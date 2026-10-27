//! Intermediate representation (IR): what the interpreter runs and what the JIT
//! compiles.
//!
//! - A function ([`Proto`]) is a graph of **basic blocks**: a sequence of
//!   instructions ending in a jump, a branch or a return ([`Term`]).
//! - Instructions read and write **registers** (locals and temporaries). Each
//!   register has a type ([`RegTy`]): `int`, `float`, `bool` or `value` (any
//!   value). An operation between typed registers is a specialized instruction
//!   (`IBin`: integer arithmetic) that the JIT turns into machine instructions;
//!   everything else goes through generic instructions (`Bin`).
//! - A dynamic value only enters a typed register through a guard (`Guard`) that
//!   checks its type: gradual typing stays sound.
//! - Variables captured by a closure live in shared **cells** (`NewCell`,
//!   `GetCell`); the closure reaches them through its **upvalues** (`GetUp`).
//! - Global variables (top level of a script, Rust functions) are slots of an
//!   array (`GetGlobal`) that persist from one script to the next (REPL).

use crate::ast::{BinOp, FnDecl, TypeDeclKind, TypeExpr};
use crate::intern::{Interner, Symbol};
use crate::span::Span;
use crate::value::Value;
use std::fmt::Write;
use std::rc::Rc;

pub type Reg = u32;
pub type BlockId = u32;

/// Type of a register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegTy {
    /// Any value.
    Value,
    Int,
    Float,
    Bool,
}

impl RegTy {
    pub fn name(self) -> &'static str {
        ty_name(self)
    }
}

/// A parsed program: its source is used for error messages.
pub struct Script {
    pub src: Rc<str>,
    pub name: Rc<str>,
}

/// A compiled function (function, method, lambda, main program…).
pub struct Proto {
    pub name: Rc<str>,
    pub script: Rc<Script>,
    pub kind: ProtoKind,
    pub params: Vec<ParamProto>,
    pub ret: Option<TypeRef>,
    /// Types of all registers. Method: register 0 = `self`; then the parameters.
    pub regs: Vec<RegTy>,
    pub ncells: u32,
    /// How each upvalue is captured from the enclosing function (`Closure`).
    pub captures: Vec<Capture>,
    pub blocks: Vec<Block>,
    pub consts: Vec<Value>,
    pub calls: Vec<CallSite>,
    pub shapes: Vec<Rc<Shape>>,
    pub types: Vec<TypeRef>,
    /// Field names of literal records.
    pub records: Vec<Rc<[Symbol]>>,
    /// Nested functions (lambdas, local `def`s).
    pub children: Vec<Rc<Proto>>,
    /// `type` / `class` / `enum` declarations (main program).
    pub decls: Vec<Decl>,
    /// Source declaration (signature for the checker, introspection).
    pub decl: Option<Rc<FnDecl>>,
    /// Native (JIT) compilation of the function, typed version.
    pub jit: std::cell::RefCell<JitState>,
    /// Baseline native compilation (untyped code): 0 = not yet, 1 = impossible,
    /// otherwise the address of the entry point `extern "C" fn(*mut Runtime, *mut NativeFrame) -> i32`.
    pub jit_dyn: std::cell::Cell<usize>,
    /// Number of calls (baseline compilation waits for the second, unless there is a loop).
    pub calls_seen: std::cell::Cell<u32>,
    /// The function contains a loop (a jump to an earlier block).
    pub has_loop: bool,
    /// No annotated parameter: arguments are stored without checks.
    pub plain_params: bool,
    /// Annotations of the function (`@[…]` before `def`).
    pub annotations: Annotations,
    /// `rescue` / `ensure` handlers, referred to by [`Block::handler`].
    pub handlers: Vec<Handler>,
    /// Number of slots where a handler keeps the error it caught (`Term::Reraise`).
    pub nslots: u32,
}

/// Where an error raised in a block goes (`begin … rescue` / `ensure`).
#[derive(Clone, Copy, Debug)]
pub struct Handler {
    pub block: BlockId,
    /// Receives the error as a value (the raised object, or an `Error`).
    pub reg: Reg,
    /// Keeps the original error, to raise it again unchanged (span and trace).
    pub slot: u32,
}

/// Annotation applied to a declaration (`@[Name(args)]`).
pub struct AnnotationIr {
    pub name: Symbol,
    /// Builds the annotation object: a parameterless function, evaluated when read.
    pub make: Rc<Proto>,
    pub span: Span,
    /// First argument, if it is a string literal (the `Deprecated` message, read by the
    /// checker without running the code).
    pub text: Option<Rc<str>>,
}

/// Annotations of a declaration, in source order.
pub type Annotations = Rc<[AnnotationIr]>;

/// Execution frame passed to baseline native code (fixed offsets).
#[repr(C)]
pub struct NativeFrame {
    pub regs: *mut Value,
    pub nregs: usize,
    pub cells: *mut crate::vm::Cell,
    pub ncells: usize,
    pub upvals: *const crate::vm::Cell,
    pub nupvals: usize,
    pub ctx: *const Rc<[u32]>,
    pub proto: *const Proto,
    /// Register and literal shape of the result (`Return`), filled in by native code.
    pub ret: u32,
    pub ret_shape: u32,
}

/// State of a function's native compilation.
#[derive(Clone, Debug, Default)]
pub enum JitState {
    /// Not tried yet (or invalidated: a called function was redefined).
    #[default]
    Untried,
    Native(NativeCode),
    /// Not compilable: the interpreter runs it.
    Rejected,
}

/// Native code of a function: entry point `extern "C" fn(ctx, args, ret) -> status`.
#[derive(Clone, Debug)]
pub struct NativeCode {
    pub entry: usize,
    pub params: Vec<RegTy>,
    pub ret: RegTy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtoKind {
    Main,
    Function,
    /// Missing argument = nil, extra arguments ignored.
    Lambda,
    /// `do` block: can serve as a builder block (`Server do … end`).
    DoBlock,
    /// Register 0 = `self`.
    Method,
    /// Default value of a `type` field (`CtxField` reads the other fields).
    FieldDefault,
    /// Construction of an annotation (`@[Name(args)]`), evaluated when read.
    Annotation,
}

pub struct ParamProto {
    pub name: Symbol,
    pub reg: Reg,
    pub ty: Option<TypeRef>,
    pub has_default: bool,
    /// `init(@field)`: index of the `self` field to initialize.
    pub field: Option<u32>,
    pub span: Span,
    pub annotations: Annotations,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capture {
    /// Cell of the enclosing function.
    Cell(u32),
    /// Upvalue of the enclosing function.
    Up(u32),
}

pub struct Block {
    pub instrs: Vec<Instr>,
    pub spans: Vec<Span>,
    pub term: Term,
    pub term_span: Span,
    /// Index in [`Proto::handlers`] of the handler for errors raised here.
    pub handler: Option<u32>,
}

/// Call: arguments in registers `start..start + npos + named.len()`.
pub struct CallSite {
    pub start: Reg,
    pub npos: u32,
    pub named: Vec<Symbol>,
    /// Literal shape of each argument (record → type conversion).
    pub shapes: Vec<Option<u32>>,
    /// The last positional argument is a `do` block.
    pub block: bool,
}

/// Shape of a literal expression, known at compile time: used to convert a **literal**
/// record to the expected type (`register({path: "/"})`, `[{path: "/a"}]`).
#[derive(Debug)]
pub enum Shape {
    Record(Vec<(Symbol, Option<Rc<Shape>>)>),
    List(Vec<Option<Rc<Shape>>>),
}

/// Compiled type annotation.
#[derive(Clone, Debug)]
pub struct TypeRef {
    pub ty: Rc<RtType>,
    /// Text of the annotation, for messages.
    pub shown: Rc<str>,
    pub span: Span,
    /// Source annotation (checker, introspection).
    pub expr: Rc<TypeExpr>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RtType {
    Any,
    Int,
    Float,
    Str,
    Bool,
    Nil,
    List(Box<RtType>),
    Hash(Box<RtType>, Box<RtType>),
    Record(Vec<(Symbol, RtType)>),
    Tuple(Vec<RtType>),
    Optional(Box<RtType>),
    Union(Vec<RtType>),
    LitStr(Rc<str>),
    LitInt(i64),
    Fn,
    /// User type: `type`, `class` or `enum`, in a global slot.
    Named(u32, Symbol),
}

/// `type` / `class` / `annotation` / `enum` declaration, executed by `DefType` /
/// `DefEnum`.
pub enum Decl {
    Type(Rc<TypeDeclIr>),
    Enum(Rc<EnumDeclIr>),
}

pub struct TypeDeclIr {
    pub name: Symbol,
    pub kind: TypeDeclKind,
    pub fields: Vec<FieldIr>,
    pub methods: Vec<(Symbol, Rc<Proto>)>,
    pub annotations: Annotations,
}

pub struct FieldIr {
    pub name: Symbol,
    pub ty: Option<TypeRef>,
    /// Default value, computed by a small function.
    pub default: Option<Rc<Proto>>,
    pub annotations: Annotations,
}

pub struct EnumDeclIr {
    pub name: Symbol,
    pub variants: Vec<Symbol>,
    /// The variant has an explicit value (`= …`), computed before `DefEnum`.
    pub explicit: Vec<bool>,
    pub repr: Option<TypeRef>,
    pub repr_is_str: bool,
    pub methods: Vec<(Symbol, Rc<Proto>)>,
    pub annotations: Annotations,
    /// Annotations of each variant.
    pub variant_annotations: Vec<Annotations>,
}

/// Resolution of a name in a `do` block outside a construction (or missing from the built type).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    None,
    Reg(Reg),
    Cell(u32),
    Up(u32),
    Global(u32),
}

#[derive(Clone, Debug)]
pub enum Instr {
    // ── Constants
    Int(Reg, i64),
    Float(Reg, f64),
    Bool(Reg, bool),
    Nil(Reg),
    Const(Reg, u32),
    Move(Reg, Reg),
    /// Runtime error (message in the constants).
    Fail(u32),
    /// `raise value`
    Raise(Reg),
    /// Does the value match the type (`rescue e: T`)?
    IsType(Reg, Reg, u32),

    // ── Variables
    GetGlobal(Reg, u32),
    SetGlobal(u32, Reg),
    /// Defines a global (`let` or `def` at the top level).
    DefGlobal {
        slot: u32,
        src: Reg,
        mutable: bool,
    },
    NewCell(u32),
    GetCell(Reg, u32),
    SetCell(u32, Reg),
    GetUp(Reg, u32),
    SetUp(u32, Reg),
    /// Name in a `do` block: field of the type under construction, otherwise `fallback`.
    GetNamed {
        dst: Reg,
        name: Symbol,
        fallback: Fallback,
    },
    SetNamed {
        name: Symbol,
        src: Reg,
        fallback: Fallback,
        shape: Option<u32>,
    },
    /// Another field of the type under construction (a field's default value).
    CtxField(Reg, u32),
    /// Dynamic value into a typed register, checked (`int` → `float` converted).
    Guard(Reg, Reg, RegTy),
    /// Check of an annotation (and conversion of literal records).
    Coerce {
        dst: Reg,
        src: Reg,
        ty: u32,
        shape: Option<u32>,
    },
    /// Parameter not provided (its default value must be computed)?
    IsUndef(Reg, Reg),

    // ── Typed arithmetic (`int` / `float` registers)
    IBin(BinOp, Reg, Reg, Reg),
    /// Integer operation with a constant: `i + 1`, `i < 1000`.
    IBinK(BinOp, Reg, Reg, i64),
    FBin(BinOp, Reg, Reg, Reg),
    INeg(Reg, Reg),
    FNeg(Reg, Reg),
    IToF(Reg, Reg),

    // ── Generic operations
    Bin(BinOp, Reg, Reg, Reg),
    Neg(Reg, Reg),
    Not(Reg, Reg),

    // ── Compound values (elements in consecutive registers)
    List(Reg, Reg, u32),
    Tuple(Reg, Reg, u32),
    /// Key, value pairs.
    Hash(Reg, Reg, u32),
    Record(Reg, Reg, u32),
    Range {
        dst: Reg,
        lo: Reg,
        hi: Reg,
        inclusive: bool,
    },
    /// Interpolation: texts of the registers, concatenated.
    Concat(Reg, Reg, u32),
    Index(Reg, Reg, Reg),
    SetIndex(Reg, Reg, Reg),
    Field(Reg, Reg, Symbol),
    /// Field whose type the checker knows (`int` / `float` / `bool`), into a
    /// register of that type.
    FieldAs(Reg, Reg, Symbol, RegTy),
    SetField {
        obj: Reg,
        name: Symbol,
        src: Reg,
        shape: Option<u32>,
    },
    /// Object field by index (`@x` in a method).
    FieldAt(Reg, Reg, u32),
    SetFieldAt {
        obj: Reg,
        idx: u32,
        src: Reg,
        shape: Option<u32>,
    },
    TupleIndex(Reg, Reg, u32),
    /// `Enum::Variant`
    Variant(Reg, Reg, Symbol),
    /// Destructuring: `n` values of a tuple / list into `dst..dst+n`
    /// (`rest`: the remainder in `dst + n`).
    Unpack {
        dst: Reg,
        src: Reg,
        n: u32,
        rest: bool,
    },

    // ── Calls
    Closure(Reg, u32),
    Call {
        dst: Reg,
        f: Reg,
        site: u32,
    },
    CallMethod {
        dst: Reg,
        recv: Reg,
        name: Symbol,
        site: u32,
    },

    // ── Loops and `case`
    /// Iterator over a list, a range, a hash (`keys`: keys only)…
    IterNew {
        dst: Reg,
        src: Reg,
        keys: bool,
    },
    /// Does a `when` pattern match the subject?
    CaseTest(Reg, Reg, Reg),

    // ── Declarations (main program)
    DefType {
        slot: u32,
        decl: u32,
    },
    /// Explicit variant values in `start..`.
    DefEnum {
        slot: u32,
        decl: u32,
        start: Reg,
    },
}

#[derive(Clone, Debug)]
pub enum Term {
    Jump(BlockId),
    /// On the truthiness of `cond` (`nil` and `false` are false).
    Branch {
        cond: Reg,
        then: BlockId,
        else_: BlockId,
    },
    /// Integer comparison and branch in one step (`if i < n`).
    IfInt {
        op: BinOp,
        a: Reg,
        b: Reg,
        then: BlockId,
        else_: BlockId,
    },
    /// Same with a constant (`if i < 1000`).
    IfIntK {
        op: BinOp,
        a: Reg,
        k: i64,
        then: BlockId,
        else_: BlockId,
    },
    Return {
        reg: Reg,
        shape: Option<u32>,
    },
    /// Next element into `dst`, then `body`; `done` when the iterator is exhausted.
    IterNext {
        iter: Reg,
        dst: Reg,
        body: BlockId,
        done: BlockId,
    },
    /// Raises again the error kept in a slot (end of an `ensure`, or no `rescue`
    /// clause matched).
    Reraise(u32),
}

// ─── Display ────────────────────────────────────────────────────────────────

impl Proto {
    /// Readable text of the IR (and of its nested functions).
    pub fn dump(&self, i: &Interner) -> String {
        let mut out = String::new();
        self.dump_into(&mut out, i);
        out
    }

    fn dump_into(&self, out: &mut String, i: &Interner) {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|p| format!("{} r{}", i.resolve(p.name), p.reg))
            .collect();
        let _ = writeln!(
            out,
            "fn {} ({:?}) ({}){}",
            self.name,
            self.kind,
            params.join(", "),
            self.ret
                .as_ref()
                .map_or(String::new(), |r| format!(": {}", r.shown))
        );
        let typed: Vec<String> = self
            .regs
            .iter()
            .enumerate()
            .filter(|(_, t)| **t != RegTy::Value)
            .map(|(n, t)| format!("r{n}:{}", ty_name(*t)))
            .collect();
        if !typed.is_empty() {
            let _ = writeln!(out, "  typed registers: {}", typed.join(" "));
        }
        for (n, b) in self.blocks.iter().enumerate() {
            match b.handler.map(|h| self.handlers[h as usize]) {
                Some(h) => {
                    let _ = writeln!(
                        out,
                        "  b{n}: (rescue r{} slot {} -> b{})",
                        h.reg, h.slot, h.block
                    );
                }
                None => {
                    let _ = writeln!(out, "  b{n}:");
                }
            }
            for ins in &b.instrs {
                let _ = writeln!(out, "    {}", self.show_instr(ins, i));
            }
            let _ = writeln!(out, "    {}", show_term(&b.term));
        }
        for c in &self.children {
            out.push('\n');
            c.dump_into(out, i);
        }
        for d in &self.decls {
            let methods = match d {
                Decl::Type(t) => {
                    for f in &t.fields {
                        if let Some(p) = &f.default {
                            out.push('\n');
                            p.dump_into(out, i);
                        }
                    }
                    &t.methods
                }
                Decl::Enum(e) => &e.methods,
            };
            for (_, m) in methods {
                out.push('\n');
                m.dump_into(out, i);
            }
        }
    }

    fn show_instr(&self, ins: &Instr, i: &Interner) -> String {
        use Instr::*;
        let s = |sym: &Symbol| i.resolve(*sym).to_string();
        match ins {
            Int(d, n) => format!("r{d} = {n}"),
            Float(d, f) => format!("r{d} = {f:?}"),
            Bool(d, b) => format!("r{d} = {b}"),
            Nil(d) => format!("r{d} = nil"),
            Const(d, c) => format!("r{d} = {}", self.consts[*c as usize].inspect(i)),
            Move(d, a) => format!("r{d} = r{a}"),
            Fail(c) => format!("fail {}", self.consts[*c as usize].inspect(i)),
            Raise(a) => format!("raise r{a}"),
            IsType(d, a, t) => format!("r{d} = r{a} is {}", self.types[*t as usize].shown),
            GetGlobal(d, g) => format!("r{d} = global[{g}]"),
            SetGlobal(g, a) => format!("global[{g}] = r{a}"),
            DefGlobal { slot, src, mutable } => {
                format!(
                    "define global[{slot}] = r{src}{}",
                    if *mutable { "" } else { " (constant)" }
                )
            }
            NewCell(c) => format!("new cell c{c}"),
            GetCell(d, c) => format!("r{d} = c{c}"),
            SetCell(c, a) => format!("c{c} = r{a}"),
            GetUp(d, u) => format!("r{d} = up{u}"),
            SetUp(u, a) => format!("up{u} = r{a}"),
            GetNamed {
                dst,
                name,
                fallback,
            } => format!("r{dst} = named {} else {fallback:?}", s(name)),
            SetNamed {
                name,
                src,
                fallback,
                ..
            } => format!("named {} = r{src} else {fallback:?}", s(name)),
            CtxField(d, f) => format!("r{d} = field under construction #{f}"),
            Guard(d, a, t) => format!("r{d} = guard {} r{a}", ty_name(*t)),
            Coerce { dst, src, ty, .. } => {
                format!(
                    "r{dst} = check r{src}: {}",
                    self.types[*ty as usize].shown
                )
            }
            IsUndef(d, a) => format!("r{d} = not provided r{a}"),
            IBin(op, d, a, b) => format!("r{d} = r{a} {}i r{b}", op.as_str()),
            IBinK(op, d, a, k) => format!("r{d} = r{a} {}i {k}", op.as_str()),
            FBin(op, d, a, b) => format!("r{d} = r{a} {}f r{b}", op.as_str()),
            INeg(d, a) => format!("r{d} = -i r{a}"),
            FNeg(d, a) => format!("r{d} = -f r{a}"),
            IToF(d, a) => format!("r{d} = float r{a}"),
            Bin(op, d, a, b) => format!("r{d} = r{a} {} r{b}", op.as_str()),
            Neg(d, a) => format!("r{d} = - r{a}"),
            Not(d, a) => format!("r{d} = not r{a}"),
            List(d, a, n) => format!("r{d} = list r{a}..+{n}"),
            Tuple(d, a, n) => format!("r{d} = tuple r{a}..+{n}"),
            Hash(d, a, n) => format!("r{d} = hash r{a}..+{}", n * 2),
            Record(d, a, r) => {
                let names: Vec<String> = self.records[*r as usize].iter().map(s).collect();
                format!("r{d} = record r{a}.. {{{}}}", names.join(", "))
            }
            Range {
                dst,
                lo,
                hi,
                inclusive,
            } => {
                format!(
                    "r{dst} = r{lo}{}r{hi}",
                    if *inclusive { ".." } else { "..." }
                )
            }
            Concat(d, a, n) => format!("r{d} = concat r{a}..+{n}"),
            Index(d, a, b) => format!("r{d} = r{a}[r{b}]"),
            SetIndex(a, b, v) => format!("r{a}[r{b}] = r{v}"),
            Field(d, a, f) => format!("r{d} = r{a}.{}", s(f)),
            FieldAs(d, a, f, t) => format!("r{d} = r{a}.{} : {}", s(f), t.name()),
            SetField { obj, name, src, .. } => format!("r{obj}.{} = r{src}", s(name)),
            FieldAt(d, a, f) => format!("r{d} = r{a}.#{f}"),
            SetFieldAt { obj, idx, src, .. } => format!("r{obj}.#{idx} = r{src}"),
            TupleIndex(d, a, n) => format!("r{d} = r{a}.{n}"),
            Variant(d, a, v) => format!("r{d} = r{a}::{}", s(v)),
            Unpack { dst, src, n, rest } => {
                format!(
                    "r{dst}..+{n} = destructure r{src}{}",
                    if *rest { " (+ rest)" } else { "" }
                )
            }
            Closure(d, p) => format!("r{d} = closure {}", self.children[*p as usize].name),
            Call { dst, f, site } => format!("r{dst} = call r{f}({})", self.show_args(*site, i)),
            CallMethod {
                dst,
                recv,
                name,
                site,
            } => {
                format!("r{dst} = r{recv}.{}({})", s(name), self.show_args(*site, i))
            }
            IterNew { dst, src, keys } => {
                format!(
                    "r{dst} = iterator r{src}{}",
                    if *keys { " (keys)" } else { "" }
                )
            }
            CaseTest(d, p, v) => format!("r{d} = pattern r{p} ~ r{v}"),
            DefType { slot, decl } => format!("define type #{decl} in global[{slot}]"),
            DefEnum { slot, decl, start } => {
                format!("define enum #{decl} in global[{slot}] (values r{start}..)")
            }
        }
    }

    fn show_args(&self, site: u32, i: &Interner) -> String {
        let c = &self.calls[site as usize];
        let mut parts: Vec<String> = (0..c.npos).map(|k| format!("r{}", c.start + k)).collect();
        for (k, n) in c.named.iter().enumerate() {
            parts.push(format!(
                "{}: r{}",
                i.resolve(*n),
                c.start + c.npos + k as u32
            ));
        }
        if c.block
            && let Some(last) = parts.get_mut(c.npos as usize - 1)
        {
            last.push_str(" (block)");
        }
        parts.join(", ")
    }
}

fn ty_name(t: RegTy) -> &'static str {
    match t {
        RegTy::Value => "value",
        RegTy::Int => "int",
        RegTy::Float => "float",
        RegTy::Bool => "bool",
    }
}

fn show_term(t: &Term) -> String {
    match t {
        Term::Jump(b) => format!("jump b{b}"),
        Term::Branch { cond, then, else_ } => format!("if r{cond} then b{then} else b{else_}"),
        Term::IfInt {
            op,
            a,
            b,
            then,
            else_,
        } => {
            format!("if r{a} {}i r{b} then b{then} else b{else_}", op.as_str())
        }
        Term::IfIntK {
            op,
            a,
            k,
            then,
            else_,
        } => {
            format!("if r{a} {}i {k} then b{then} else b{else_}", op.as_str())
        }
        Term::Return { reg, .. } => format!("return r{reg}"),
        Term::Reraise(slot) => format!("reraise slot {slot}"),
        Term::IterNext {
            iter,
            dst,
            body,
            done,
        } => {
            format!("r{dst} = next r{iter} then b{body}, done b{done}")
        }
    }
}

impl Term {
    /// Successor blocks (jump targets).
    pub fn targets_mut(&mut self) -> Vec<&mut BlockId> {
        match self {
            Term::Jump(b) => vec![b],
            Term::Branch { then, else_, .. }
            | Term::IfInt { then, else_, .. }
            | Term::IfIntK { then, else_, .. } => vec![then, else_],
            Term::IterNext { body, done, .. } => vec![body, done],
            Term::Return { .. } | Term::Reraise(_) => vec![],
        }
    }
}
