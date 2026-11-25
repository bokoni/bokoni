//! Compilation of functions to WebAssembly, for Bokoni itself running in WebAssembly,
//! where the Cranelift JIT cannot (`wasm-jit` feature; `docs/execution.md`).
//!
//! A function is written as a WebAssembly module (`wasm-encoder`), which the host compiles
//! to machine code: the embedder gives a compiler (`set_compiler`) that instantiates the
//! module against this program's memory and function table, appends its exports `f0`,
//! `f1`… to that table and returns the index of the first one. A function pointer being
//! an index in that table, Rust then calls the code like the Cranelift JIT's. Two tiers,
//! as that JIT's:
//!
//! - Typed code (`Jit::compile`): functions whose parameters are `int`, `float` or `bool`
//!   (annotated, or a copy typed for its arguments), whose result is too (annotated, or
//!   inferred: `Infer`), and whose body only computes with such values, compares, branches
//!   and calls such functions (recursion included). The same rules as the interpreter:
//!   overflow, Euclidean division, division by zero, call depth, with the same messages
//!   and the same call stack.
//!   - Each function has an internal function, `(ctx, params…) -> (status, result)`
//!     (status 0: success, 1: error, described in `ctx`), and an entry point for Rust
//!     with the C ABI of the Cranelift JIT's: `extern "C" fn(ctx, *const u64 args, *mut
//!     u64 result) -> i32`.
//!   - The functions compiled together (one calling another not compiled yet) call each
//!     other directly; one compiled before is called through the table
//!     (`call_indirect`).
//!   - A register holds a value of one native type at each point (`Plan`); one
//!     WebAssembly local per register and type.
//!   - Redefining a global function called (directly or not) invalidates the code,
//!     which is compiled again when needed.
//! - Baseline code (`Jit::compile_dynamic`): any other function without `rescue` /
//!   `ensure`, its registers left in memory, the interpreter's code called for what it
//!   does not do itself (`crate::baseline`).
//!
//! Control flow (`control_flow`): structured (`loop`, `block`, `if`) from the IR's blocks
//! when the graph is reducible, as Bokoni's are; otherwise a loop over a `br_table` on the
//! block number.

use crate::ast::BinOp;
use crate::baseline::{FAST, Fast, FieldCache, bk_drop, bk_iter_next, bk_step};
use crate::diagnostic::Error;
use crate::globals::Globals;
use crate::intern::{FxHashMap, Symbol};
use crate::ir::{
    Instr, JitState, NativeCode, NativeFrame, Proto, ProtoKind, Reg, RegTy, RtType, Script, Term,
};
use crate::span::Span;
use crate::value::layout::{
    BOOL_PAYLOAD, FIRST_HEAP_TAG, PAYLOAD, SIZE, TAG_BOOL, TAG_FLOAT, TAG_INT, TAG_NIL,
};
use crate::value::{Function as Func, Value};
use crate::vm::Runtime;
use std::cell::Cell;
use std::rc::Rc;
use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, InstructionSink, MemArg, MemoryType, Module, RefType, TableType, TypeSection,
    ValType,
};

/// Instantiates a module (its bytes) against this program's memory (`env.memory`) and
/// function table (`env.table`), appends its exports `f0`, `f1`… to that table, in order,
/// and returns the index of `f0`; `None` if it cannot.
pub type Compiler = fn(&[u8]) -> Option<u32>;

thread_local! {
    static COMPILER: Cell<Option<Compiler>> = const { Cell::new(None) };
    static COMPILED: Cell<u32> = const { Cell::new(0) };
    static BASELINE: Cell<u32> = const { Cell::new(0) };
    static DISPATCHED: Cell<u32> = const { Cell::new(0) };
    static STRUCTURED: Cell<bool> = const { Cell::new(true) };
}

/// Structured control flow when possible (the default), or always the dispatch loop
/// (tests of that loop).
#[doc(hidden)]
pub fn set_structured(on: bool) {
    STRUCTURED.with(|c| c.set(on));
}

/// Sets the compiler of the generated modules (WebAssembly only: elsewhere, nothing is
/// compiled). Without one, everything is interpreted.
pub fn set_compiler(compiler: Option<Compiler>) {
    COMPILER.with(|c| c.set(compiler));
}

/// Number of functions compiled so far, typed (tests).
pub fn compiled_functions() -> u32 {
    COMPILED.with(Cell::get)
}

/// Number of functions compiled so far to baseline code (tests).
pub fn baseline_functions() -> u32 {
    BASELINE.with(Cell::get)
}

/// Number of functions whose control flow was not structured (its graph not reducible),
/// so far (tests).
pub fn dispatched_functions() -> u32 {
    DISPATCHED.with(Cell::get)
}

/// Context shared with the generated code. The first three fields are read and written by
/// it (fixed offsets); the others are used by Rust code only.
#[repr(C)]
pub struct Ctx {
    depth: i64,
    max_depth: i64,
    /// Error that occurred: index into the error table.
    err: i64,
    /// Call stack of the error (indices into the call table), most recent first.
    traces: Vec<u32>,
}

const DEPTH: u64 = 0;
const MAX_DEPTH: u64 = 8;
const ERR: u64 = 16;

#[derive(Clone, Copy)]
enum ErrKind {
    Overflow,
    DivZero,
    Depth,
}

struct ErrInfo {
    kind: ErrKind,
    span: Span,
    script: Rc<Script>,
}

struct TraceInfo {
    name: Rc<str>,
    span: Span,
}

extern "C" fn bk_trace(ctx: *mut Ctx, id: u32) {
    // SAFETY: `ctx` is the context passed by `Jit::call`, valid during the call.
    unsafe { (*ctx).traces.push(id) }
}

extern "C" fn bk_frem(a: f64, b: f64) -> f64 {
    a.rem_euclid(b)
}

/// Index of a Rust function in the function table (what a function pointer is in
/// WebAssembly).
fn table_index(f: usize) -> i32 {
    f as i32
}

struct Compiled {
    /// (Kept alive: its address is the key.)
    _proto: Rc<Proto>,
    /// Table index of the internal function.
    index: u32,
    /// Global slots its code depends on (its direct calls, and theirs).
    slots: Vec<u32>,
}

pub struct Jit {
    /// The `len` method (an interned name), as for the Cranelift JIT (unused here).
    pub len_sym: Symbol,
    ctx: Box<Ctx>,
    errors: Vec<ErrInfo>,
    traces: Vec<TraceInfo>,
    /// Functions compiled (key: address of the `Proto`).
    compiled: FxHashMap<usize, Compiled>,
    /// Field access caches of the baseline code (addresses held by the code).
    #[allow(clippy::vec_box)]
    field_caches: Vec<Box<FieldCache>>,
}

/// Type of a register for the generated code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum T {
    Int,
    Float,
    Bool,
    /// Never read (`nil` of an expression whose value is ignored).
    Nil,
    /// Global function called directly (global slot).
    Callee(u32),
}

/// Type of a register at a program point.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lat {
    /// Not written yet.
    Undef,
    T(T),
    /// Different types depending on the path: unreadable.
    Conflict,
}

impl Lat {
    fn get(self) -> Option<T> {
        match self {
            Lat::T(t) => Some(t),
            _ => None,
        }
    }

    fn merge(self, o: Lat) -> Lat {
        if self == o { self } else { Lat::Conflict }
    }
}

fn of(t: RegTy) -> T {
    match t {
        RegTy::Int => T::Int,
        RegTy::Float => T::Float,
        RegTy::Bool => T::Bool,
        RegTy::Value => unreachable!("not a native type"),
    }
}

fn prim(rt: &RtType) -> Option<RegTy> {
    match rt {
        RtType::Int => Some(RegTy::Int),
        RtType::Float => Some(RegTy::Float),
        RtType::Bool => Some(RegTy::Bool),
        _ => None,
    }
}

fn val(t: RegTy) -> ValType {
    match t {
        RegTy::Int => ValType::I64,
        RegTy::Float => ValType::F64,
        RegTy::Bool => ValType::I32,
        RegTy::Value => unreachable!("not a native type"),
    }
}

fn val_of(t: T) -> ValType {
    match t {
        T::Int => ValType::I64,
        T::Float => ValType::F64,
        T::Bool => ValType::I32,
        _ => unreachable!("no value"),
    }
}

/// Parameters and result of a function the generated code can run; its result inferred
/// when not annotated (`Infer`).
fn native_sig(p: &Proto, globals: &Globals, inf: &mut Infer) -> Option<(Vec<RegTy>, RegTy)> {
    let params = native_sig_params(p)?;
    let ret = match &p.ret {
        Some(t) => prim(&t.ty)?,
        None => inf.ret(p, globals)?,
    };
    Some((params, ret))
}

/// Parameters of a function the generated code can run.
fn native_sig_params(p: &Proto) -> Option<Vec<RegTy>> {
    if p.kind != ProtoKind::Function
        || p.ncells != 0
        || !p.captures.is_empty()
        || !p.handlers.is_empty()
        || p.is_generic()
    {
        return None;
    }
    p.params
        .iter()
        .map(|prm| {
            if prm.has_default {
                return None;
            }
            prim(&prm.ty.as_ref()?.ty)
        })
        .collect()
}

/// Results of functions without a result annotation, inferred: the type `R` (`int`, then
/// `float`, then `bool`) such that, its calls (recursive ones included) assumed to return
/// `R`, every `return` of the function returns exactly `R`. Then by induction on the
/// depth of the calls, it does return `R` (or never returns).
#[derive(Default)]
struct Infer {
    /// Results established (`None`: no native type).
    known: FxHashMap<usize, Option<RegTy>>,
    /// Assumptions being checked (function, type), innermost last.
    assumed: Vec<(usize, RegTy)>,
    /// Assumptions read (functions), for the inferences that depend on them.
    read: Vec<usize>,
}

impl Infer {
    fn ret(&mut self, p: &Proto, globals: &Globals) -> Option<RegTy> {
        let key = p as *const Proto as usize;
        if let Some(&(_, r)) = self.assumed.iter().find(|(k, _)| *k == key) {
            self.read.push(key);
            return Some(r);
        }
        if let Some(r) = self.known.get(&key) {
            return *r;
        }
        let mut found = None;
        let mark = self.read.len();
        for r in [RegTy::Int, RegTy::Float, RegTy::Bool] {
            self.assumed.push((key, r));
            let ok = analyze(p, globals, self).is_some();
            self.assumed.pop();
            if ok {
                found = Some(r);
                break;
            }
        }
        // Depends on an assumption still being checked (mutual recursion): not kept, the
        // one that assumed it depends on it in turn.
        let deps: Vec<usize> = self.read.drain(mark..).filter(|k| *k != key).collect();
        if deps.is_empty() {
            self.known.insert(key, found);
        } else {
            self.read.extend(deps);
        }
        found
    }

    /// The type assumed for `p`'s result, while checking it.
    fn assumed(&self, p: &Proto) -> Option<RegTy> {
        let key = p as *const Proto as usize;
        self.assumed
            .iter()
            .rev()
            .find(|(k, _)| *k == key)
            .map(|(_, r)| *r)
    }
}

/// The function in global slot `slot`, if it is one written in Bokoni.
fn global_proto(globals: &Globals, slot: u32) -> Option<Rc<Proto>> {
    match &globals.at(slot).value {
        Some(Value::Function(f)) => match &**f {
            Func::Script(c) => Some(c.proto.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// `int` / `float` / `bool` arguments only (each passed in a 64-bit word).
pub fn by_ref(params: &[RegTy]) -> Option<u64> {
    params.iter().all(|t| *t != RegTy::Value).then_some(0)
}

fn is_cmp(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
    )
}

fn is_arith(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
    )
}

/// Result of analysing a function: type of the registers at each point.
struct Plan {
    params: Vec<RegTy>,
    ret: RegTy,
    /// Types at the entry of each block (`None`: never reached).
    entry: Vec<Option<Vec<Lat>>>,
    /// Type written by each instruction of each block, if any.
    defs: Vec<Vec<Option<T>>>,
    /// Function called by each `Call` (key: block, instruction).
    targets: FxHashMap<(usize, usize), Target>,
    /// Global slots read (the functions called).
    slots: Vec<u32>,
}

/// A function called, with its native signature.
#[derive(Clone)]
struct Target {
    proto: Rc<Proto>,
    params: Vec<RegTy>,
    ret: RegTy,
}

/// Type written by `ins` (`Some(None)`: nothing), operands read from `cur`; `None`: not
/// compilable. Records the function called by a `Call` into `target`.
fn transfer(
    p: &Proto,
    globals: &Globals,
    inf: &mut Infer,
    ins: &Instr,
    cur: &[Lat],
    target: &mut Option<Target>,
    slots: &mut Vec<u32>,
) -> Option<Option<T>> {
    use Instr::*;
    let g = |r: Reg| cur[r as usize].get();
    let scalar = |t: T| matches!(t, T::Int | T::Float | T::Bool);
    let numeric = |t: T| matches!(t, T::Int | T::Float);
    let fits = |want: T, t: T| t == want || (want == T::Float && t == T::Int);
    Some(Some(match ins {
        Int(..) => T::Int,
        Float(..) => T::Float,
        Bool(..) => T::Bool,
        Nil(_) => T::Nil,
        Move(_, a) => Some(g(*a)?).filter(|t| scalar(*t) || *t == T::Nil)?,
        // Into a typed register: its type checked by the analysis (`fixed`).
        Guard(_, a, _) => Some(g(*a)?).filter(|t| scalar(*t))?,
        // An annotation already satisfied (`let s: float = 0`).
        Coerce {
            src,
            ty,
            shape: None,
            ..
        } => {
            let want = of(prim(&p.types[*ty as usize].ty)?);
            if !fits(want, g(*src)?) {
                return None;
            }
            want
        }
        IBin(op, _, a, b) => {
            if g(*a)? != T::Int || g(*b)? != T::Int || !(is_cmp(*op) || is_arith(*op)) {
                return None;
            }
            if is_cmp(*op) { T::Bool } else { T::Int }
        }
        IBinK(op, _, a, _) => {
            if g(*a)? != T::Int || !(is_cmp(*op) || is_arith(*op)) {
                return None;
            }
            if is_cmp(*op) { T::Bool } else { T::Int }
        }
        FBin(op, _, a, b) => {
            if g(*a)? != T::Float || g(*b)? != T::Float || !(is_cmp(*op) || is_arith(*op)) {
                return None;
            }
            if is_cmp(*op) { T::Bool } else { T::Float }
        }
        INeg(_, a) => (g(*a)? == T::Int).then_some(T::Int)?,
        FNeg(_, a) => (g(*a)? == T::Float).then_some(T::Float)?,
        IToF(_, a) => (g(*a)? == T::Int).then_some(T::Float)?,
        Neg(_, a) => Some(g(*a)?).filter(|t| numeric(*t))?,
        Not(_, a) => Some(g(*a)?).filter(|t| scalar(*t)).map(|_| T::Bool)?,
        Bin(op, _, a, b) => match (*op, g(*a)?, g(*b)?) {
            (op, T::Int, T::Int) if is_arith(op) => T::Int,
            (op, x, y) if is_arith(op) && numeric(x) && numeric(y) => T::Float,
            (op, x, y) if is_cmp(op) && numeric(x) && numeric(y) => T::Bool,
            (BinOp::Eq | BinOp::Ne, T::Bool, T::Bool) => T::Bool,
            _ => return None,
        },
        GetGlobal(_, slot) => {
            global_proto(globals, *slot)?;
            if !slots.contains(slot) {
                slots.push(*slot);
            }
            T::Callee(*slot)
        }
        Call { f, site, .. } => {
            let T::Callee(slot) = g(*f)? else {
                return None;
            };
            let site = &p.calls[*site as usize];
            if site.block || !site.named.is_empty() {
                return None;
            }
            let args: Vec<T> = (0..site.npos)
                .map(|k| g(site.start + k))
                .collect::<Option<_>>()?;
            let mut callee = global_proto(globals, slot)?;
            // Parameters not annotated: its copy for the types of the arguments.
            if callee.untyped() {
                if args.len() != callee.params.len() {
                    return None;
                }
                let tys = args
                    .iter()
                    .map(|t| match t {
                        T::Int => Some(RtType::Int),
                        T::Float => Some(RtType::Float),
                        T::Bool => Some(RtType::Bool),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()?;
                callee = callee.specialize_params(&tys);
            }
            let (params, ret) = native_sig(&callee, globals, inf)?;
            if params.len() != args.len()
                || !params.iter().zip(&args).all(|(p, a)| fits(of(*p), *a))
            {
                return None;
            }
            *target = Some(Target {
                proto: callee,
                params,
                ret,
            });
            of(ret)
        }
        _ => return None,
    }))
}

fn analyze(p: &Proto, globals: &Globals, inf: &mut Infer) -> Option<Plan> {
    // While checking an assumed result: that one.
    let (params, ret) = match inf.assumed(p) {
        Some(r) => (native_sig_params(p)?, r),
        None => native_sig(p, globals, inf)?,
    };
    // An inferred result is returned as is (an annotated `float` converts an `int`).
    let exact = p.ret.is_none();
    // Typed IR registers: their type does not change.
    let fixed: Vec<Option<T>> = p
        .regs
        .iter()
        .map(|t| (*t != RegTy::Value).then(|| of(*t)))
        .collect();
    let mut start: Vec<Lat> = fixed.iter().map(|t| t.map_or(Lat::Undef, Lat::T)).collect();
    for (prm, t) in p.params.iter().zip(&params) {
        start[prm.reg as usize] = Lat::T(of(*t));
    }
    let nb = p.blocks.len();
    let mut entry: Vec<Option<Vec<Lat>>> = vec![None; nb];
    entry[0] = Some(start);
    let mut defs: Vec<Vec<Option<T>>> = p
        .blocks
        .iter()
        .map(|b| vec![None; b.instrs.len()])
        .collect();
    let mut targets = FxHashMap::default();
    let mut slots = Vec::new();
    loop {
        let mut changed = false;
        for (bi, b) in p.blocks.iter().enumerate() {
            let Some(mut cur) = entry[bi].clone() else {
                continue;
            };
            for (k, ins) in b.instrs.iter().enumerate() {
                let mut target = None;
                let t = transfer(p, globals, inf, ins, &cur, &mut target, &mut slots)?;
                if let Some(q) = target {
                    targets.insert((bi, k), q);
                }
                let d = written(ins);
                match (d, t) {
                    (Some(d), Some(t)) => {
                        let t = match fixed[d as usize] {
                            None => t,
                            // Typed register: same type, or `int` to `float` (converted).
                            Some(f) if f == t || (f == T::Float && t == T::Int) => f,
                            Some(_) => return None,
                        };
                        defs[bi][k] = Some(t);
                        cur[d as usize] = Lat::T(t);
                    }
                    (None, None) => {}
                    _ => return None,
                }
            }
            let g = |r: Reg| cur[r as usize].get();
            let ok = match &b.term {
                Term::Jump(_) => true,
                Term::Branch { cond, .. } => {
                    matches!(g(*cond), Some(T::Int | T::Float | T::Bool | T::Nil))
                }
                Term::IfInt { a, b: y, .. } => g(*a) == Some(T::Int) && g(*y) == Some(T::Int),
                Term::IfIntK { a, .. } => g(*a) == Some(T::Int),
                Term::Return { reg, .. } => match (ret, g(*reg)) {
                    (RegTy::Float, Some(T::Int)) => !exact,
                    (_, t) => t == Some(of(ret)),
                },
                _ => false,
            };
            if !ok {
                return None;
            }
            let mut term = b.term.clone();
            for s in term.targets_mut() {
                let s = *s as usize;
                let merged = match &entry[s] {
                    None => cur.clone(),
                    Some(old) => old.iter().zip(&cur).map(|(a, b)| a.merge(*b)).collect(),
                };
                if entry[s].as_ref() != Some(&merged) {
                    entry[s] = Some(merged);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    Some(Plan {
        params,
        ret,
        entry,
        defs,
        targets,
        slots,
    })
}

/// Register written by an instruction `transfer` accepts.
fn written(ins: &Instr) -> Option<Reg> {
    use Instr::*;
    match ins {
        Int(d, _) | Float(d, _) | Bool(d, _) | Nil(d) | Move(d, _) | Guard(d, ..) => Some(*d),
        Coerce { dst, .. } | Call { dst, .. } => Some(*dst),
        IBin(_, d, ..) | IBinK(_, d, ..) | FBin(_, d, ..) | Bin(_, d, ..) => Some(*d),
        INeg(d, _) | FNeg(d, _) | IToF(d, _) | Neg(d, _) | Not(d, _) | GetGlobal(d, _) => Some(*d),
        _ => None,
    }
}

/// The function types of a module, without duplicates.
#[derive(Default)]
struct Types {
    list: Vec<(Vec<ValType>, Vec<ValType>)>,
}

impl Types {
    fn index(&mut self, params: Vec<ValType>, results: Vec<ValType>) -> u32 {
        let t = (params, results);
        match self.list.iter().position(|x| *x == t) {
            Some(i) => i as u32,
            None => {
                self.list.push(t);
                (self.list.len() - 1) as u32
            }
        }
    }

    /// Internal function of signature `params` → `ret`: `(ctx, params…) -> (status, ret)`.
    fn internal(&mut self, params: &[RegTy], ret: RegTy) -> u32 {
        let ps = std::iter::once(ValType::I32)
            .chain(params.iter().map(|t| val(*t)))
            .collect();
        self.index(ps, vec![ValType::I32, val(ret)])
    }
}

/// How the generated code calls a function.
#[derive(Clone, Copy)]
enum Callee {
    /// In the same module (function index).
    Local(u32),
    /// Compiled before (table index).
    Table(u32),
}

impl Jit {
    pub fn new() -> Option<Jit> {
        if !cfg!(target_arch = "wasm32") {
            return None;
        }
        COMPILER.with(Cell::get)?;
        // The layout of a `Value` the generated code reads and writes (`value::layout`).
        let v = Value::Int(0x0102_0304_0506_0708);
        let b = Value::Bool(true);
        // SAFETY: reads within the values (`SIZE` bytes each).
        let ok = size_of::<Value>() == SIZE as usize
            && unsafe {
                let p = &v as *const Value as *const u8;
                let q = &b as *const Value as *const u8;
                *p == TAG_INT
                    && (p.add(PAYLOAD as usize) as *const i64).read_unaligned()
                        == 0x0102_0304_0506_0708
                    && *q == TAG_BOOL
                    && *q.add(BOOL_PAYLOAD as usize) == 1
            };
        if !ok {
            return None;
        }
        Some(Jit {
            len_sym: Symbol(0),
            ctx: Box::new(Ctx {
                depth: 0,
                max_depth: 0,
                err: -1,
                traces: Vec::new(),
            }),
            errors: Vec::new(),
            traces: Vec::new(),
            compiled: FxHashMap::default(),
            field_caches: Vec::new(),
        })
    }

    /// Compiles `root` and the functions it calls that are not compiled yet; marks each
    /// `Native` or `Rejected`.
    pub fn compile(
        &mut self,
        root: &Rc<Proto>,
        this: Option<Rc<crate::objects::TypeInfo>>,
        globals: &mut Globals,
        _env: &[crate::vm::Cell],
    ) {
        if this.is_some() {
            *root.jit.borrow_mut() = JitState::Rejected;
            return;
        }
        // 1. Functions reachable through direct calls, and their analysis.
        let mut order: Vec<Rc<Proto>> = Vec::new();
        let mut plans: FxHashMap<usize, Option<Plan>> = FxHashMap::default();
        let mut todo = vec![root.clone()];
        let mut inf = Infer::default();
        while let Some(p) = todo.pop() {
            let key = Rc::as_ptr(&p) as usize;
            if plans.contains_key(&key) {
                continue;
            }
            if self.compiled.contains_key(&key) {
                if matches!(*p.jit.borrow(), JitState::Native(_)) {
                    continue;
                }
                // Invalidated code (a function called was redefined): to compile again.
                self.compiled.remove(&key);
            }
            let plan = match *p.jit.borrow() {
                JitState::Rejected => None,
                _ => analyze(&p, globals, &mut inf),
            };
            if let Some(plan) = &plan {
                todo.extend(plan.targets.values().map(|t| t.proto.clone()));
            }
            plans.insert(key, plan);
            order.push(p);
        }
        // 2. A function calling one that is not compilable is not either (no call into the
        // interpreter from the generated code, in this phase).
        loop {
            let mut changed = false;
            for p in &order {
                let key = Rc::as_ptr(p) as usize;
                let bad = plans[&key].as_ref().is_some_and(|plan| {
                    plan.targets.values().any(|q| {
                        let k = Rc::as_ptr(&q.proto) as usize;
                        !self.compiled.contains_key(&k) && !matches!(plans.get(&k), Some(Some(_)))
                    })
                });
                if bad {
                    plans.insert(key, None);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let new: Vec<Rc<Proto>> = order
            .iter()
            .filter(|p| matches!(plans.get(&(Rc::as_ptr(p) as usize)), Some(Some(_))))
            .cloned()
            .collect();
        let base = if new.is_empty() {
            None
        } else {
            let callees: FxHashMap<usize, Callee> = new
                .iter()
                .enumerate()
                .map(|(i, p)| (Rc::as_ptr(p) as usize, Callee::Local(2 * i as u32)))
                .chain(
                    self.compiled
                        .iter()
                        .map(|(k, c)| (*k, Callee::Table(c.index))),
                )
                .collect();
            let bytes = self.module(&new, &plans, &callees);
            COMPILER.with(Cell::get).and_then(|c| c(&bytes))
        };
        // 3. Compiled: the entry points; the slots each depends on, with those of the
        // functions it calls.
        if let Some(base) = base {
            for (i, p) in new.iter().enumerate() {
                let key = Rc::as_ptr(p) as usize;
                let plan = plans[&key].as_ref().unwrap();
                *p.jit.borrow_mut() = JitState::Native(NativeCode {
                    entry: (base + 2 * i as u32 + 1) as usize,
                    params: plan.params.clone(),
                    ret: plan.ret,
                });
                self.compiled.insert(
                    key,
                    Compiled {
                        _proto: p.clone(),
                        index: base + 2 * i as u32,
                        slots: plan.slots.clone(),
                    },
                );
            }
            COMPILED.with(|c| c.set(c.get() + new.len() as u32));
            for p in &new {
                let key = Rc::as_ptr(p) as usize;
                let mut slots: Vec<u32> = Vec::new();
                let mut seen = vec![key];
                let mut todo = vec![key];
                while let Some(k) = todo.pop() {
                    let Some(c) = self.compiled.get(&k) else {
                        continue;
                    };
                    slots.extend(&c.slots);
                    for s in &c.slots {
                        if let Some(q) = global_proto(globals, *s) {
                            // (A copy typed for the arguments, if it is the one called.)
                            for q in std::iter::once(q.clone())
                                .chain(q.specs.borrow().iter().map(|(_, s)| s.clone()))
                            {
                                let k = Rc::as_ptr(&q) as usize;
                                if !seen.contains(&k) {
                                    seen.push(k);
                                    todo.push(k);
                                }
                            }
                        }
                    }
                }
                slots.sort_unstable();
                slots.dedup();
                for s in slots {
                    globals.watch(s, p);
                }
            }
        }
        // Not compiled: retried if a function they call changes.
        for p in &order {
            let key = Rc::as_ptr(p) as usize;
            if base.is_none() || matches!(plans.get(&key), Some(None)) {
                *p.jit.borrow_mut() = JitState::Rejected;
                for s in called_slots(p) {
                    globals.watch(s, p);
                }
            }
        }
    }

    /// The module of the functions `new`: for function `i`, its internal function `2i`
    /// and its entry point `2i + 1`, exported in that order.
    fn module(
        &mut self,
        new: &[Rc<Proto>],
        plans: &FxHashMap<usize, Option<Plan>>,
        callees: &FxHashMap<usize, Callee>,
    ) -> Vec<u8> {
        let mut types = Types::default();
        let mut funcs = FunctionSection::new();
        let mut code = CodeSection::new();
        let mut exports = ExportSection::new();
        for (i, p) in new.iter().enumerate() {
            let plan = plans[&(Rc::as_ptr(p) as usize)].as_ref().unwrap();
            let internal = types.internal(&plan.params, plan.ret);
            let f = self.function(p, plan, callees, &mut types);
            funcs.function(internal);
            code.function(&f);
            let entry = types.index(vec![ValType::I32; 3], vec![ValType::I32]);
            funcs.function(entry);
            code.function(&entry_point(2 * i as u32, plan));
            exports.export(&format!("f{}", 2 * i), ExportKind::Func, 2 * i as u32);
            exports.export(
                &format!("f{}", 2 * i + 1),
                ExportKind::Func,
                2 * i as u32 + 1,
            );
        }
        assemble(&types, &funcs, &exports, &code)
    }

    fn error_id(&mut self, kind: ErrKind, span: Span, script: &Rc<Script>) -> i64 {
        self.errors.push(ErrInfo {
            kind,
            span,
            script: script.clone(),
        });
        (self.errors.len() - 1) as i64
    }

    /// The internal function of `p`.
    fn function(
        &mut self,
        p: &Proto,
        plan: &Plan,
        callees: &FxHashMap<usize, Callee>,
        types: &mut Types,
    ) -> Function {
        let mut fx = Fx::new(p, plan);
        let trace_ty = types.index(vec![ValType::I32, ValType::I32], vec![]);
        let frem_ty = types.index(vec![ValType::F64, ValType::F64], vec![ValType::F64]);
        // Locals of every register and type it holds, then the scratch ones.
        for (bi, b) in p.blocks.iter().enumerate() {
            let Some(entry) = &plan.entry[bi] else {
                continue;
            };
            for (r, l) in entry.iter().enumerate() {
                if let Some(t) = l.get() {
                    fx.local(r as Reg, t);
                }
            }
            for (k, ins) in b.instrs.iter().enumerate() {
                if let (Some(d), Some(t)) = (written(ins), plan.defs[bi][k]) {
                    fx.local(d, t);
                }
            }
        }
        let s: [u32; 4] = std::array::from_fn(|_| fx.scratch(ValType::I64));
        let sf = fx.scratch(ValType::F64);
        let si = fx.scratch(ValType::I32);
        let pc = fx.scratch(ValType::I32);
        let st = fx.scratch(ValType::I32);
        let mut f = Function::new_with_locals_types(fx.locals.clone());
        let mut x = f.instructions();
        // Parameters into their registers' locals.
        for (k, (prm, t)) in p.params.iter().zip(&plan.params).enumerate() {
            x.local_get(k as u32 + 1);
            x.local_set(fx.at(prm.reg, of(*t)));
        }
        let reached: Vec<bool> = plan.entry.iter().map(Option::is_some).collect();
        let mut tb = TypedBlocks {
            jit: self,
            p,
            plan,
            callees,
            types,
            fx: &fx,
            s,
            sf,
            si,
            st,
            trace_ty,
            frem_ty,
        };
        control_flow(&mut x, &successors(p), &reached, pc, &mut tb);
        x.unreachable();
        x.end();
        f
    }
}

/// The code of a typed function's blocks.
struct TypedBlocks<'a> {
    jit: &'a mut Jit,
    p: &'a Proto,
    plan: &'a Plan,
    callees: &'a FxHashMap<usize, Callee>,
    types: &'a mut Types,
    fx: &'a Fx,
    s: [u32; 4],
    sf: u32,
    si: u32,
    st: u32,
    trace_ty: u32,
    frem_ty: u32,
}

impl Blocks for TypedBlocks<'_> {
    fn block(&mut self, x: &mut InstructionSink, bi: usize) -> Exit {
        let (p, plan, fx, s) = (self.p, self.plan, self.fx, self.s);
        let b = &p.blocks[bi];
        let Some(entry) = &plan.entry[bi] else {
            // Never reached.
            x.unreachable();
            return Exit::Return;
        };
        let mut cur = entry.clone();
        for (k, ins) in b.instrs.iter().enumerate() {
            let span = b.spans[k];
            let def = plan.defs[bi][k];
            let ctx = Ctx2 {
                fx,
                cur: &cur,
                s,
                ret: plan.ret,
            };
            match ins {
                Instr::Call { dst, site, .. } => {
                    let q = &plan.targets[&(bi, k)];
                    let (cparams, cret) = (&q.params, q.ret);
                    let depth_err = self.jit.error_id(ErrKind::Depth, span, &p.script);
                    // Depth: as the interpreter, an error at the call beyond the limit.
                    x.local_get(0).i64_load(mem(DEPTH)).local_tee(s[0]);
                    x.local_get(0).i64_load(mem(MAX_DEPTH)).i64_ge_s();
                    x.if_(BlockType::Empty);
                    ctx.fail(x, depth_err);
                    x.end();
                    x.local_get(0).local_get(s[0]).i64_const(1).i64_add();
                    x.i64_store(mem(DEPTH));
                    x.local_get(0);
                    let site = &p.calls[*site as usize];
                    for (k, pt) in cparams.iter().enumerate() {
                        let r = site.start + k as u32;
                        ctx.get(x, r);
                        convert(x, cur[r as usize].get().unwrap(), of(*pt));
                    }
                    match self.callees[&(Rc::as_ptr(&q.proto) as usize)] {
                        Callee::Local(i) => {
                            x.call(i);
                        }
                        Callee::Table(i) => {
                            let ty = self.types.internal(cparams, cret);
                            x.i32_const(i as i32).call_indirect(0, ty);
                        }
                    }
                    let tmp = match cret {
                        RegTy::Int => s[1],
                        RegTy::Float => self.sf,
                        _ => self.si,
                    };
                    x.local_set(tmp).local_set(self.st);
                    x.local_get(0).local_get(s[0]).i64_store(mem(DEPTH));
                    self.jit.traces.push(TraceInfo {
                        name: q.proto.name.clone(),
                        span,
                    });
                    let tid = (self.jit.traces.len() - 1) as i32;
                    x.local_get(self.st).if_(BlockType::Empty);
                    x.local_get(0)
                        .i32_const(tid)
                        .i32_const(table_index(bk_trace as *const () as usize))
                        .call_indirect(0, self.trace_ty);
                    x.i32_const(1);
                    zero(x, plan.ret);
                    x.return_();
                    x.end();
                    let d = def.unwrap();
                    x.local_get(tmp);
                    convert(x, of(cret), d);
                    x.local_set(fx.at(*dst, d));
                }
                ins => {
                    let jit = &mut *self.jit;
                    let mut err = |kind| jit.error_id(kind, span, &p.script);
                    ctx.instr(x, ins, def, &mut err, self.frem_ty);
                }
            }
            if let (Some(d), Some(t)) = (written(ins), def) {
                cur[d as usize] = Lat::T(t);
            }
        }
        let ctx = Ctx2 {
            fx,
            cur: &cur,
            s,
            ret: plan.ret,
        };
        match &b.term {
            Term::Jump(t) => Exit::Jump(*t),
            Term::Branch { cond, then, else_ } => {
                match cur[*cond as usize].get().unwrap() {
                    T::Bool => ctx.get(x, *cond),
                    T::Nil => {
                        x.i32_const(0);
                    }
                    // A number is true.
                    _ => {
                        x.i32_const(1);
                    }
                }
                Exit::Cond(*then, *else_)
            }
            Term::IfInt {
                op,
                a,
                b: y,
                then,
                else_,
            } => {
                ctx.get(x, *a);
                ctx.get(x, *y);
                int_cmp(x, *op);
                Exit::Cond(*then, *else_)
            }
            Term::IfIntK {
                op,
                a,
                k,
                then,
                else_,
            } => {
                ctx.get(x, *a);
                x.i64_const(*k);
                int_cmp(x, *op);
                Exit::Cond(*then, *else_)
            }
            Term::Return { reg, .. } => {
                x.i32_const(0);
                ctx.get(x, *reg);
                convert(x, cur[*reg as usize].get().unwrap(), of(plan.ret));
                x.return_();
                Exit::Return
            }
            _ => unreachable!("rejected by the analysis"),
        }
    }
}

// ─── Control flow ──────────────────────────────────────────────────────────

/// How a block ends.
enum Exit {
    Jump(u32),
    /// On the `i32` its code pushed: the first block if non-zero, the second otherwise.
    Cond(u32, u32),
    /// Its code returned (or cannot go on).
    Return,
}

/// The code of a function's blocks, for `control_flow`.
trait Blocks {
    /// The code of block `b`, then of its terminator: its condition pushed, or its return.
    fn block(&mut self, x: &mut InstructionSink, b: usize) -> Exit;
}

/// Successors of each block of `p`.
fn successors(p: &Proto) -> Vec<Vec<u32>> {
    p.blocks
        .iter()
        .map(|b| match &b.term {
            Term::Jump(t) => vec![*t],
            Term::Branch { then, else_, .. }
            | Term::IfInt { then, else_, .. }
            | Term::IfIntK { then, else_, .. } => vec![*then, *else_],
            Term::IterNext { body, done, .. } => vec![*body, *done],
            Term::Return { .. } | Term::Reraise(_) => vec![],
        })
        .collect()
}

/// Writes the control flow of the blocks (`succ`: their successors; only those `reached`
/// can run). Structured when the graph is reducible (as Bokoni's are): one `loop` per loop
/// header, one `block` per block reached by several forward edges (a merge), the others
/// written inline where they are branched to (N. Ramsey, "Beyond Relooper", 2022).
/// Otherwise a loop over a `br_table` on the block number (`pc`).
fn control_flow(
    x: &mut InstructionSink,
    succ: &[Vec<u32>],
    reached: &[bool],
    pc: u32,
    blocks: &mut impl Blocks,
) {
    match Structure::new(succ, reached).filter(|_| STRUCTURED.with(Cell::get)) {
        Some(st) => {
            let mut labels = Vec::new();
            st.tree(x, &mut labels, 0, blocks);
        }
        None => {
            DISPATCHED.with(|c| c.set(c.get() + 1));
            dispatch(x, succ.len() as u32, reached, pc, blocks);
        }
    }
}

/// The dispatch loop: `pc` the block to run.
fn dispatch(x: &mut InstructionSink, n: u32, reached: &[bool], pc: u32, blocks: &mut impl Blocks) {
    x.loop_(BlockType::Empty);
    for _ in 0..n {
        x.block(BlockType::Empty);
    }
    x.local_get(pc);
    x.br_table(0..n, n - 1);
    x.end();
    for (bi, &reach) in reached.iter().enumerate() {
        // Depth of the dispatch loop from this block's code.
        let to_loop = n - 1 - bi as u32;
        let next = bi as u32 + 1;
        let jump = |x: &mut InstructionSink, target: u32, extra: u32| {
            if target != next || extra > 0 {
                x.i32_const(target as i32).local_set(pc).br(to_loop + extra);
            }
        };
        if !reach {
            x.unreachable();
        } else {
            match blocks.block(x, bi) {
                Exit::Jump(t) => jump(x, t, 0),
                Exit::Cond(t, e) => {
                    x.if_(BlockType::Empty);
                    jump(x, t, 1);
                    x.end();
                    jump(x, e, 0);
                }
                Exit::Return => {}
            }
        }
        if bi + 1 < n as usize {
            x.end();
        }
    }
    x.end();
}

/// Labels around the code being written, innermost last.
#[derive(Clone, Copy, PartialEq)]
enum Label {
    /// The `loop` of a loop header: a branch to it goes back to the header.
    Loop(u32),
    /// The `block` followed by a merge block: a branch to it goes to that block.
    Block(u32),
    If,
}

/// A reducible control flow graph, for structured code.
struct Structure {
    /// Reverse postorder number of each block reached.
    order: Vec<u32>,
    header: Vec<bool>,
    merge: Vec<bool>,
    /// Children in the dominator tree.
    children: Vec<Vec<u32>>,
}

const UNREACHED: u32 = u32::MAX;

impl Structure {
    fn new(succ: &[Vec<u32>], reached: &[bool]) -> Option<Structure> {
        let n = succ.len();
        let ok = |b: u32| reached[b as usize];
        // Reverse postorder (iterative depth-first search).
        let mut post = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        let mut stack: Vec<(u32, usize)> = vec![(0, 0)];
        seen[0] = true;
        while let Some(&mut (b, ref mut k)) = stack.last_mut() {
            let ss = &succ[b as usize];
            if *k < ss.len() {
                let s = ss[*k];
                *k += 1;
                if ok(s) && !seen[s as usize] {
                    seen[s as usize] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
                stack.pop();
            }
        }
        let rpo: Vec<u32> = post.iter().rev().copied().collect();
        let mut order = vec![UNREACHED; n];
        for (i, &b) in rpo.iter().enumerate() {
            order[b as usize] = i as u32;
        }
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &b in &rpo {
            for &s in &succ[b as usize] {
                if order[s as usize] != UNREACHED {
                    preds[s as usize].push(b);
                }
            }
        }
        // Immediate dominators (Cooper, Harvey, Kennedy).
        let mut idom = vec![UNREACHED; n];
        idom[0] = 0;
        let intersect = |idom: &[u32], mut a: u32, mut b: u32| {
            while a != b {
                while order[a as usize] > order[b as usize] {
                    a = idom[a as usize];
                }
                while order[b as usize] > order[a as usize] {
                    b = idom[b as usize];
                }
            }
            a
        };
        loop {
            let mut changed = false;
            for &b in &rpo[1..] {
                let mut new = UNREACHED;
                for &q in &preds[b as usize] {
                    if idom[q as usize] == UNREACHED {
                        continue;
                    }
                    new = if new == UNREACHED {
                        q
                    } else {
                        intersect(&idom, q, new)
                    };
                }
                if idom[b as usize] != new {
                    idom[b as usize] = new;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let dominates = |a: u32, mut b: u32| loop {
            if a == b {
                return true;
            }
            if b == 0 {
                return false;
            }
            b = idom[b as usize];
        };
        let mut header = vec![false; n];
        let mut forward_in = vec![0u32; n];
        for &b in &rpo {
            for &s in &succ[b as usize] {
                if order[s as usize] == UNREACHED {
                    continue;
                }
                if order[s as usize] <= order[b as usize] {
                    // A back edge: to a block dominating its source, or the graph is not
                    // reducible.
                    if !dominates(s, b) {
                        return None;
                    }
                    header[s as usize] = true;
                } else {
                    forward_in[s as usize] += 1;
                }
            }
        }
        let merge: Vec<bool> = forward_in.iter().map(|&k| k >= 2).collect();
        let mut children = vec![Vec::new(); n];
        let mut depth = vec![0u32; n];
        for &b in &rpo[1..] {
            let d = idom[b as usize];
            children[d as usize].push(b);
            depth[b as usize] = depth[d as usize] + 1;
            // (The code is written recursively along the dominator tree.)
            if depth[b as usize] > 400 {
                return None;
            }
        }
        Some(Structure {
            order,
            header,
            merge,
            children,
        })
    }

    /// The code of block `b` and of the blocks it dominates.
    fn tree(
        &self,
        x: &mut InstructionSink,
        labels: &mut Vec<Label>,
        b: u32,
        blocks: &mut impl Blocks,
    ) {
        let mut merges: Vec<u32> = self.children[b as usize]
            .iter()
            .copied()
            .filter(|&c| self.merge[c as usize])
            .collect();
        // The one written last (highest number) gets the outermost `block`.
        merges.sort_by_key(|&c| std::cmp::Reverse(self.order[c as usize]));
        if self.header[b as usize] {
            x.loop_(BlockType::Empty);
            labels.push(Label::Loop(b));
            self.within(x, labels, b, &merges, blocks);
            labels.pop();
            x.end();
        } else {
            self.within(x, labels, b, &merges, blocks);
        }
    }

    fn within(
        &self,
        x: &mut InstructionSink,
        labels: &mut Vec<Label>,
        b: u32,
        merges: &[u32],
        blocks: &mut impl Blocks,
    ) {
        if let Some((&y, rest)) = merges.split_first() {
            x.block(BlockType::Empty);
            labels.push(Label::Block(y));
            self.within(x, labels, b, rest, blocks);
            labels.pop();
            x.end();
            self.tree(x, labels, y, blocks);
            return;
        }
        match blocks.block(x, b as usize) {
            Exit::Jump(t) => self.branch(x, labels, b, t, blocks),
            Exit::Cond(t, e) => {
                x.if_(BlockType::Empty);
                labels.push(Label::If);
                self.branch(x, labels, b, t, blocks);
                x.else_();
                self.branch(x, labels, b, e, blocks);
                labels.pop();
                x.end();
            }
            Exit::Return => {}
        }
    }

    /// From block `from` to block `to`: a branch, or `to`'s code here.
    fn branch(
        &self,
        x: &mut InstructionSink,
        labels: &mut Vec<Label>,
        from: u32,
        to: u32,
        blocks: &mut impl Blocks,
    ) {
        let label = if self.order[to as usize] <= self.order[from as usize] {
            Label::Loop(to)
        } else if self.merge[to as usize] {
            Label::Block(to)
        } else {
            self.tree(x, labels, to, blocks);
            return;
        };
        let at = labels
            .iter()
            .rposition(|l| *l == label)
            .expect("enclosing label");
        x.br((labels.len() - 1 - at) as u32);
    }
}

/// The bytes of a module of these functions, importing the program's memory and table.
fn assemble(
    types: &Types,
    funcs: &FunctionSection,
    exports: &ExportSection,
    code: &CodeSection,
) -> Vec<u8> {
    let mut ts = TypeSection::new();
    for (ps, rs) in &types.list {
        ts.ty().function(ps.iter().copied(), rs.iter().copied());
    }
    let mut imports = ImportSection::new();
    imports.import(
        "env",
        "memory",
        EntityType::Memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );
    imports.import(
        "env",
        "table",
        EntityType::Table(TableType {
            element_type: RefType::FUNCREF,
            table64: false,
            minimum: 1,
            maximum: None,
            shared: false,
        }),
    );
    let mut m = Module::new();
    m.section(&ts)
        .section(&imports)
        .section(funcs)
        .section(exports)
        .section(code);
    m.finish()
}

/// The entry point for Rust of internal function `internal`:
/// `(ctx, *const u64 args, *mut u64 result) -> status`.
fn entry_point(internal: u32, plan: &Plan) -> Function {
    let tmp = val(plan.ret);
    let mut f = Function::new([(1, ValType::I32), (1, tmp)]);
    let (st, out) = (3, 4);
    let mut x = f.instructions();
    x.local_get(0);
    for (k, t) in plan.params.iter().enumerate() {
        let at = MemArg {
            offset: 8 * k as u64,
            align: 3,
            memory_index: 0,
        };
        x.local_get(1);
        match t {
            RegTy::Int => x.i64_load(at),
            RegTy::Float => x.f64_load(at),
            // (The low word of the argument: 0 or 1.)
            _ => x.i32_load(MemArg { align: 2, ..at }),
        };
    }
    x.call(internal);
    x.local_set(out).local_set(st);
    x.local_get(2).local_get(out);
    match plan.ret {
        RegTy::Int => x.i64_store(mem(0)),
        RegTy::Float => x.f64_store(mem(0)),
        _ => x.i64_extend_i32_u().i64_store(mem(0)),
    };
    x.local_get(st);
    x.end();
    f
}

fn mem(offset: u64) -> MemArg {
    MemArg {
        offset,
        align: 3,
        memory_index: 0,
    }
}

/// Converts the value of type `from` on the stack to type `to` (`int` to `float`).
fn convert(x: &mut InstructionSink, from: T, to: T) {
    if from == T::Int && to == T::Float {
        x.f64_convert_i64_s();
    }
}

fn zero(x: &mut InstructionSink, t: RegTy) {
    match t {
        RegTy::Int => x.i64_const(0),
        RegTy::Float => x.f64_const(0.0.into()),
        _ => x.i32_const(0),
    };
}

fn int_cmp(x: &mut InstructionSink, op: BinOp) {
    match op {
        BinOp::Eq => x.i64_eq(),
        BinOp::Ne => x.i64_ne(),
        BinOp::Lt => x.i64_lt_s(),
        BinOp::Le => x.i64_le_s(),
        BinOp::Gt => x.i64_gt_s(),
        _ => x.i64_ge_s(),
    };
}

fn float_cmp(x: &mut InstructionSink, op: BinOp) {
    match op {
        BinOp::Eq => x.f64_eq(),
        BinOp::Ne => x.f64_ne(),
        BinOp::Lt => x.f64_lt(),
        BinOp::Le => x.f64_le(),
        BinOp::Gt => x.f64_gt(),
        _ => x.f64_ge(),
    };
}

/// The locals of a function being written.
struct Fx {
    nparams: u32,
    /// Local of each register and type.
    regs: FxHashMap<(Reg, ValType), u32>,
    locals: Vec<ValType>,
}

impl Fx {
    fn new(_p: &Proto, plan: &Plan) -> Fx {
        Fx {
            nparams: plan.params.len() as u32 + 1,
            regs: FxHashMap::default(),
            locals: Vec::new(),
        }
    }

    fn local(&mut self, r: Reg, t: T) {
        if !matches!(t, T::Int | T::Float | T::Bool) {
            return;
        }
        let v = val_of(t);
        if !self.regs.contains_key(&(r, v)) {
            let i = self.scratch(v);
            self.regs.insert((r, v), i);
        }
    }

    fn scratch(&mut self, v: ValType) -> u32 {
        self.locals.push(v);
        self.nparams + self.locals.len() as u32 - 1
    }

    fn at(&self, r: Reg, t: T) -> u32 {
        self.regs[&(r, val_of(t))]
    }
}

/// What the code of one instruction needs.
struct Ctx2<'a> {
    fx: &'a Fx,
    cur: &'a [Lat],
    /// Four scratch `i64` locals.
    s: [u32; 4],
    ret: RegTy,
}

impl Ctx2<'_> {
    fn ty(&self, r: Reg) -> T {
        self.cur[r as usize]
            .get()
            .expect("register typed by the analysis")
    }

    fn get(&self, x: &mut InstructionSink, r: Reg) {
        x.local_get(self.fx.at(r, self.ty(r)));
    }

    /// Error `id`: written into the context, status 1 returned.
    fn fail(&self, x: &mut InstructionSink, id: i64) {
        x.local_get(0).i64_const(id).i64_store(mem(ERR));
        x.i32_const(1);
        zero(x, self.ret);
        x.return_();
    }

    /// `if <i32 on the stack> { error }`.
    fn fail_if(&self, x: &mut InstructionSink, id: i64) {
        x.if_(BlockType::Empty);
        self.fail(x, id);
        x.end();
    }

    /// Integer operation of `s[0]` and `s[1]`: its result on the stack.
    fn int_bin(&self, x: &mut InstructionSink, op: BinOp, err: &mut dyn FnMut(ErrKind) -> i64) {
        let [a, b, r, q] = self.s;
        match op {
            BinOp::Add | BinOp::Sub => {
                x.local_get(a).local_get(b);
                if op == BinOp::Add {
                    x.i64_add();
                } else {
                    x.i64_sub();
                }
                x.local_set(r);
                // Overflow: (a ^ r) & (b ^ r) < 0 for `+`, (a ^ b) & (a ^ r) < 0 for `-`.
                if op == BinOp::Add {
                    x.local_get(a).local_get(r).i64_xor();
                    x.local_get(b).local_get(r).i64_xor();
                } else {
                    x.local_get(a).local_get(b).i64_xor();
                    x.local_get(a).local_get(r).i64_xor();
                }
                x.i64_and().i64_const(0).i64_lt_s();
                self.fail_if(x, err(ErrKind::Overflow));
                x.local_get(r);
            }
            BinOp::Mul => {
                x.local_get(a).local_get(b).i64_mul().local_set(r);
                // Overflow: a != 0 and (a == -1 ? b == MIN : r / a != b).
                let id = err(ErrKind::Overflow);
                x.local_get(a).i64_const(0).i64_ne();
                x.if_(BlockType::Empty);
                x.local_get(a).i64_const(-1).i64_eq();
                x.if_(BlockType::Result(ValType::I32));
                x.local_get(b).i64_const(i64::MIN).i64_eq();
                x.else_();
                x.local_get(r)
                    .local_get(a)
                    .i64_div_s()
                    .local_get(b)
                    .i64_ne();
                x.end();
                self.fail_if(x, id);
                x.end();
                x.local_get(r);
            }
            BinOp::Div | BinOp::Rem => {
                x.local_get(b).i64_eqz();
                self.fail_if(x, err(ErrKind::DivZero));
                x.local_get(a).i64_const(i64::MIN).i64_eq();
                x.local_get(b).i64_const(-1).i64_eq();
                x.i32_and();
                self.fail_if(x, err(ErrKind::Overflow));
                x.local_get(a).local_get(b).i64_div_s().local_set(q);
                x.local_get(a).local_get(b).i64_rem_s().local_set(r);
                if op == BinOp::Div {
                    // Euclidean: r < 0 ? (b > 0 ? q - 1 : q + 1) : q
                    x.local_get(q).i64_const(1).i64_sub();
                    x.local_get(q).i64_const(1).i64_add();
                    x.local_get(b).i64_const(0).i64_gt_s();
                    x.select();
                    x.local_get(q);
                } else {
                    // r < 0 ? (b < 0 ? r - b : r + b) : r
                    x.local_get(r).local_get(b).i64_sub();
                    x.local_get(r).local_get(b).i64_add();
                    x.local_get(b).i64_const(0).i64_lt_s();
                    x.select();
                    x.local_get(r);
                }
                x.local_get(r).i64_const(0).i64_lt_s();
                x.select();
            }
            op => {
                x.local_get(a).local_get(b);
                int_cmp(x, op);
            }
        }
    }

    /// Float operation of the two values on the stack.
    fn float_bin(&self, x: &mut InstructionSink, op: BinOp, frem_ty: u32) {
        match op {
            BinOp::Add => x.f64_add(),
            BinOp::Sub => x.f64_sub(),
            BinOp::Mul => x.f64_mul(),
            BinOp::Div => x.f64_div(),
            BinOp::Rem => x
                .i32_const(table_index(bk_frem as *const () as usize))
                .call_indirect(0, frem_ty),
            op => {
                float_cmp(x, op);
                x
            }
        };
    }

    /// Code of an instruction other than `Call`: its result into its register.
    fn instr(
        &self,
        x: &mut InstructionSink,
        ins: &Instr,
        def: Option<T>,
        err: &mut dyn FnMut(ErrKind) -> i64,
        frem_ty: u32,
    ) {
        use Instr::*;
        let [a0, b0, ..] = self.s;
        // Type of the value left on the stack, converted to `def`.
        let t: T = match ins {
            Int(_, v) => {
                x.i64_const(*v);
                T::Int
            }
            Float(_, v) => {
                x.f64_const((*v).into());
                T::Float
            }
            Bool(_, v) => {
                x.i32_const(*v as i32);
                T::Bool
            }
            Nil(_) | GetGlobal(..) => return,
            Move(_, a) | Guard(_, a, _) | Coerce { src: a, .. } => {
                let t = self.ty(*a);
                if t == T::Nil {
                    return;
                }
                self.get(x, *a);
                t
            }
            IBin(op, _, a, b) => {
                self.get(x, *a);
                x.local_set(a0);
                self.get(x, *b);
                x.local_set(b0);
                self.int_bin(x, *op, err);
                if is_cmp(*op) { T::Bool } else { T::Int }
            }
            IBinK(op, _, a, k) => {
                self.get(x, *a);
                x.local_set(a0);
                x.i64_const(*k).local_set(b0);
                self.int_bin(x, *op, err);
                if is_cmp(*op) { T::Bool } else { T::Int }
            }
            FBin(op, _, a, b) => {
                self.get(x, *a);
                self.get(x, *b);
                self.float_bin(x, *op, frem_ty);
                if is_cmp(*op) { T::Bool } else { T::Float }
            }
            Bin(op, _, a, b) => match (self.ty(*a), self.ty(*b)) {
                (T::Int, T::Int) => {
                    self.get(x, *a);
                    x.local_set(a0);
                    self.get(x, *b);
                    x.local_set(b0);
                    self.int_bin(x, *op, err);
                    if is_cmp(*op) { T::Bool } else { T::Int }
                }
                (T::Bool, T::Bool) => {
                    self.get(x, *a);
                    self.get(x, *b);
                    if *op == BinOp::Eq {
                        x.i32_eq();
                    } else {
                        x.i32_ne();
                    }
                    T::Bool
                }
                (ta, tb) => {
                    self.get(x, *a);
                    convert(x, ta, T::Float);
                    self.get(x, *b);
                    convert(x, tb, T::Float);
                    self.float_bin(x, *op, frem_ty);
                    if is_cmp(*op) { T::Bool } else { T::Float }
                }
            },
            INeg(_, a) | Neg(_, a) if self.ty(*a) == T::Int => {
                self.get(x, *a);
                x.local_tee(a0).i64_const(i64::MIN).i64_eq();
                self.fail_if(x, err(ErrKind::Overflow));
                x.i64_const(0).local_get(a0).i64_sub();
                T::Int
            }
            FNeg(_, a) | Neg(_, a) => {
                self.get(x, *a);
                x.f64_neg();
                T::Float
            }
            IToF(_, a) => {
                self.get(x, *a);
                x.f64_convert_i64_s();
                T::Float
            }
            Not(_, a) => {
                if self.ty(*a) == T::Bool {
                    self.get(x, *a);
                    x.i32_eqz();
                } else {
                    // A number is true.
                    x.i32_const(0);
                }
                T::Bool
            }
            _ => unreachable!("rejected by the analysis"),
        };
        let (Some(d), Some(def)) = (written(ins), def) else {
            unreachable!("an instruction with a result");
        };
        convert(x, t, def);
        x.local_set(self.fx.at(d, def));
    }
}

/// Global slots read by `p` (the functions it may call), to retry compiling it when one
/// changes.
fn called_slots(p: &Proto) -> Vec<u32> {
    let mut v: Vec<u32> = p
        .blocks
        .iter()
        .flat_map(|b| &b.instrs)
        .filter_map(|i| match i {
            Instr::GetGlobal(_, s) => Some(*s),
            _ => None,
        })
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

impl Jit {
    /// Calls the entry point `entry` (compiled by `compile`) with `args` (`n` of them).
    /// Same contract as the Cranelift JIT's `Jit::call`.
    ///
    /// # Safety
    /// `jit` belongs to the runtime `rt`, both valid; `entry` is an entry point returned
    /// in a `NativeCode` by this `Jit`, for parameters of the types of `args`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn call<'a>(
        jit: *mut Jit,
        _rt: *mut Runtime,
        entry: usize,
        n: usize,
        _by_ref: u64,
        ret: RegTy,
        args: impl IntoIterator<Item = &'a Value>,
        _env: &[crate::vm::Cell],
        depth: usize,
        max_depth: usize,
    ) -> Result<Value, Error> {
        let mut raw = [0u64; 16];
        let mut heap = Vec::new();
        let buf: &mut [u64] = if n <= raw.len() {
            &mut raw[..n]
        } else {
            heap.resize(n, 0);
            &mut heap
        };
        for (slot, v) in buf.iter_mut().zip(args) {
            *slot = match v {
                Value::Int(i) => *i as u64,
                Value::Float(f) => f.to_bits(),
                Value::Bool(b) => *b as u64,
                _ => unreachable!("arguments of the native types"),
            };
        }
        // SAFETY: see above; the context is only reached through its address.
        unsafe {
            let ctx: *mut Ctx = &mut *(*jit).ctx;
            (*ctx).depth = depth as i64;
            (*ctx).max_depth = max_depth as i64;
            (*ctx).err = -1;
            (*ctx).traces.clear();
            let mut out = 0u64;
            // `entry` is a table index of an entry point with this signature.
            let f: extern "C" fn(*mut Ctx, *const u64, *mut u64) -> i32 =
                std::mem::transmute_copy(&entry);
            let status = f(ctx, buf.as_ptr(), &mut out);
            if status == 0 {
                return Ok(match ret {
                    RegTy::Int => Value::Int(out as i64),
                    RegTy::Float => Value::Float(f64::from_bits(out)),
                    RegTy::Bool => Value::Bool(out & 1 == 1),
                    RegTy::Value => unreachable!("native result"),
                });
            }
            Err(take_error(&*jit, ctx))
        }
    }
}

/// The error that stopped the generated code, with its call stack.
///
/// # Safety
/// `ctx` is the context of `jit`, after an error.
unsafe fn take_error(jit: &Jit, ctx: *mut Ctx) -> Error {
    unsafe {
        let info = &jit.errors[(*ctx).err as usize];
        let msg = match info.kind {
            ErrKind::Overflow => "integer overflow".to_string(),
            ErrKind::DivZero => "division by zero".to_string(),
            ErrKind::Depth => format!(
                "too many nested calls ({}): infinite recursion?",
                (*ctx).max_depth
            ),
        };
        let mut e = Error::runtime(msg, info.span);
        e.src = Some((info.script.name.clone(), info.script.src.clone()));
        for &t in (*ctx).traces.iter().take(16) {
            let t = &jit.traces[t as usize];
            e.trace.push((t.name.to_string(), t.span));
        }
        (*ctx).traces.clear();
        (*ctx).err = -1;
        e
    }
}

// ─── Baseline compilation (untyped code) ───────────────────────────────────
//
// As the Cranelift JIT's (`crate::baseline`): any function without `rescue` / `ensure`.
// Its registers stay `Value`s in the call's window, in this program's memory; the code
// does the simple cases itself (constants, copies of scalar values, integer and float
// arithmetic and comparisons, branches) and asks the helpers for the rest, the
// interpreter's own code (`bk_step`) last, including the unusual cases (overflow…): the
// semantics are the same by construction. Entry point: `extern "C" fn(*mut Runtime, *mut
// NativeFrame) -> i32` (0: returned, its register in `NativeFrame::ret`; 1: error).

/// Locals of a baseline function: its two parameters, then these.
const B_REGS: u32 = 2;
const B_A: u32 = 3;
const B_B: u32 = 4;
const B_R: u32 = 5;
const B_Q: u32 = 6;
const B_TA: u32 = 7;
const B_TB: u32 = 8;
const B_F: u32 = 9;
const B_PC: u32 = 10;
const B_X: u32 = 11;

/// What the code of a baseline function needs.
struct Base<'a> {
    regs_ty: &'a [RegTy],
    /// Types of the helpers: `(i32 × 5) -> i32` (`bk_step`, the fast ones), `bk_iter_next`,
    /// `bk_drop`, `bk_frem`.
    t_step: u32,
    t_iter: u32,
    t_drop: u32,
}

/// Offset of a register's value in the window.
fn roff(r: Reg) -> u64 {
    r as u64 * SIZE as u64
}

fn at(offset: u64, align: u32) -> MemArg {
    MemArg {
        offset,
        align,
        memory_index: 0,
    }
}

fn fn_index(f: usize) -> i32 {
    table_index(f)
}

impl Base<'_> {
    fn tag(&self, x: &mut InstructionSink, r: Reg) {
        x.local_get(B_REGS).i32_load8_u(at(roff(r), 0));
    }

    fn int(&self, x: &mut InstructionSink, r: Reg) {
        x.local_get(B_REGS)
            .i64_load(at(roff(r) + PAYLOAD as u64, 3));
    }

    fn float(&self, x: &mut InstructionSink, r: Reg) {
        x.local_get(B_REGS)
            .f64_load(at(roff(r) + PAYLOAD as u64, 3));
    }

    fn boolean(&self, x: &mut InstructionSink, r: Reg) {
        x.local_get(B_REGS)
            .i32_load8_u(at(roff(r) + BOOL_PAYLOAD as u64, 0));
    }

    /// Writes a scalar into register `d`: tag `tag`, payload in local `from` (none for
    /// `nil`). A `value` register may hold a resource: dropped first.
    fn write(&self, x: &mut InstructionSink, d: Reg, tag: u8, from: Option<u32>) {
        if self.regs_ty[d as usize] == RegTy::Value {
            self.tag(x, d);
            x.i32_const(FIRST_HEAP_TAG as i32).i32_ge_u();
            x.if_(BlockType::Empty);
            x.local_get(B_REGS).i32_const(roff(d) as i32).i32_add();
            x.i32_const(fn_index(bk_drop as *const () as usize))
                .call_indirect(0, self.t_drop);
            x.end();
        }
        x.local_get(B_REGS).i32_const(tag as i32);
        x.i32_store8(at(roff(d), 0));
        if let Some(l) = from {
            x.local_get(B_REGS).local_get(l);
            match tag {
                TAG_INT => x.i64_store(at(roff(d) + PAYLOAD as u64, 3)),
                TAG_FLOAT => x.f64_store(at(roff(d) + PAYLOAD as u64, 3)),
                _ => x.i32_store8(at(roff(d) + BOOL_PAYLOAD as u64, 0)),
            };
        }
    }

    /// `bk_step` for `ins`: on an error, status 1 returned.
    fn step(&self, x: &mut InstructionSink, ins: &Instr, span: Span) {
        x.local_get(0).local_get(1);
        x.i32_const(ins as *const Instr as usize as i32);
        x.i32_const(span.start as i32).i32_const(span.end as i32);
        x.i32_const(fn_index(bk_step as *const () as usize))
            .call_indirect(0, self.t_step);
        x.if_(BlockType::Empty).i32_const(1).return_().end();
    }

    /// A fast helper; if it does not handle the case, `bk_step`.
    #[allow(clippy::too_many_arguments)]
    fn fast(
        &self,
        x: &mut InstructionSink,
        which: Fast,
        a: u32,
        b: u32,
        p: usize,
        ins: &Instr,
        span: Span,
    ) {
        x.local_get(0).local_get(1);
        x.i32_const(a as i32)
            .i32_const(b as i32)
            .i32_const(p as i32);
        x.i32_const(fn_index(FAST[which as usize].1 as *const () as usize))
            .call_indirect(0, self.t_step);
        x.if_(BlockType::Empty);
        self.step(x, ins, span);
        x.end();
    }

    /// Truthiness of a register (`nil` and `false` are false), as an `i32`.
    fn truthy(&self, x: &mut InstructionSink, r: Reg) {
        match self.regs_ty[r as usize] {
            RegTy::Bool => {
                self.boolean(x, r);
            }
            RegTy::Int | RegTy::Float => {
                x.i32_const(1);
            }
            RegTy::Value => {
                self.tag(x, r);
                x.local_tee(B_TA).i32_const(TAG_NIL as i32).i32_eq();
                x.local_get(B_TA).i32_const(TAG_BOOL as i32).i32_eq();
                self.boolean(x, r);
                x.i32_eqz().i32_and().i32_or().i32_eqz();
            }
        }
    }

    /// Integer operation of locals `B_A` and `B_B` into register `d`; an edge case
    /// (overflow, division by zero) branches out `slow` levels up, before writing.
    fn int_op(&self, x: &mut InstructionSink, op: BinOp, d: Reg, slow: u32) {
        let (a, b, r, q) = (B_A, B_B, B_R, B_Q);
        match op {
            BinOp::Add | BinOp::Sub => {
                x.local_get(a).local_get(b);
                if op == BinOp::Add {
                    x.i64_add().local_set(r);
                    x.local_get(a).local_get(r).i64_xor();
                    x.local_get(b).local_get(r).i64_xor();
                } else {
                    x.i64_sub().local_set(r);
                    x.local_get(a).local_get(b).i64_xor();
                    x.local_get(a).local_get(r).i64_xor();
                }
                x.i64_and().i64_const(0).i64_lt_s().br_if(slow);
            }
            BinOp::Mul => {
                x.local_get(a).local_get(b).i64_mul().local_set(r);
                // Overflow: a != 0 and (a == -1 ? b == MIN : r / a != b), the division
                // by 1 when it would trap.
                x.local_get(a).i64_const(0).i64_ne();
                x.local_get(a).i64_const(-1).i64_eq();
                x.local_get(b).i64_const(i64::MIN).i64_eq();
                x.i32_and();
                x.local_get(a).i64_const(-1).i64_ne();
                x.local_get(r);
                x.local_get(a).i64_const(1);
                x.local_get(a).i64_const(0).i64_ne();
                x.local_get(a).i64_const(-1).i64_ne();
                x.i32_and().select();
                x.i64_div_s().local_get(b).i64_ne();
                x.i32_and().i32_or().i32_and().br_if(slow);
            }
            BinOp::Div | BinOp::Rem => {
                x.local_get(b).i64_eqz().br_if(slow);
                x.local_get(a).i64_const(i64::MIN).i64_eq();
                x.local_get(b).i64_const(-1).i64_eq();
                x.i32_and().br_if(slow);
                x.local_get(a).local_get(b).i64_div_s().local_set(q);
                x.local_get(a).local_get(b).i64_rem_s().local_set(r);
                if op == BinOp::Div {
                    x.local_get(q).i64_const(1).i64_sub();
                    x.local_get(q).i64_const(1).i64_add();
                    x.local_get(b).i64_const(0).i64_gt_s();
                    x.select();
                    x.local_get(q);
                } else {
                    x.local_get(r).local_get(b).i64_sub();
                    x.local_get(r).local_get(b).i64_add();
                    x.local_get(b).i64_const(0).i64_lt_s();
                    x.select();
                    x.local_get(r);
                }
                x.local_get(r).i64_const(0).i64_lt_s();
                x.select().local_set(r);
            }
            op => {
                x.local_get(a).local_get(b);
                int_cmp(x, op);
                x.local_set(B_X);
                self.write(x, d, TAG_BOOL, Some(B_X));
                return;
            }
        }
        self.write(x, d, TAG_INT, Some(r));
    }

    /// Float operation (not `%`) of the two values on the stack into register `d`.
    fn float_op(&self, x: &mut InstructionSink, op: BinOp, d: Reg) {
        match op {
            BinOp::Add => x.f64_add(),
            BinOp::Sub => x.f64_sub(),
            BinOp::Mul => x.f64_mul(),
            BinOp::Div => x.f64_div(),
            op => {
                float_cmp(x, op);
                x.local_set(B_X);
                self.write(x, d, TAG_BOOL, Some(B_X));
                return;
            }
        };
        x.local_set(B_F);
        self.write(x, d, TAG_FLOAT, Some(B_F));
    }

    /// `ins` with its fast path `fast` (which branches out one level up for the slow path,
    /// two levels up once done), `bk_step` otherwise.
    fn with_slow(
        &self,
        x: &mut InstructionSink,
        ins: &Instr,
        span: Span,
        fast: impl FnOnce(&mut InstructionSink),
    ) {
        x.block(BlockType::Empty);
        x.block(BlockType::Empty);
        fast(x);
        x.br(1);
        x.end();
        self.step(x, ins, span);
        x.end();
    }

    /// Code of an instruction.
    #[allow(clippy::vec_box)]
    fn instr(
        &self,
        x: &mut InstructionSink,
        ins: &Instr,
        span: Span,
        caches: &mut Vec<Box<FieldCache>>,
    ) {
        use Instr::*;
        let ty = |r: Reg| self.regs_ty[r as usize];
        match ins {
            GetUp(r, u) => self.fast(x, Fast::GetUp, *r, *u, 0, ins, span),
            SetUp(u, a) => self.fast(x, Fast::SetUp, *u, *a, 0, ins, span),
            GetCell(r, c) => self.fast(x, Fast::GetCell, *r, *c, 0, ins, span),
            SetCell(c, a) => self.fast(x, Fast::SetCell, *c, *a, 0, ins, span),
            GetGlobal(r, g) => self.fast(x, Fast::GetGlobal, *r, *g, 0, ins, span),
            SetGlobal(g, a) => self.fast(x, Fast::SetGlobal, *g, *a, 0, ins, span),
            Field(r, a, name, _) | FieldAs(r, a, name, _) => {
                let want = match ins {
                    FieldAs(.., t) => *t,
                    _ => RegTy::Value,
                };
                caches.push(Box::new(FieldCache {
                    info: None,
                    idx: 0,
                    name: *name,
                    want,
                }));
                let p = &**caches.last().unwrap() as *const FieldCache as usize;
                self.fast(x, Fast::Field, *r, *a, p, ins, span);
            }
            Call { .. } => {
                x.local_get(0).local_get(1);
                x.i32_const(span.start as i32).i32_const(span.end as i32);
                x.i32_const(ins as *const Instr as usize as i32);
                x.i32_const(fn_index(FAST[Fast::Call as usize].1 as *const () as usize))
                    .call_indirect(0, self.t_step);
                x.if_(BlockType::Empty).i32_const(1).return_().end();
            }
            // The expected tag, or an `int` to `float`; otherwise the interpreter produces
            // the error.
            Guard(r, a, t) if *t != RegTy::Value && ty(*r) == *t => {
                self.with_slow(x, ins, span, |x| {
                    let (tag, local) = match t {
                        RegTy::Int => (TAG_INT, B_A),
                        RegTy::Float => (TAG_FLOAT, B_F),
                        _ => (TAG_BOOL, B_X),
                    };
                    self.tag(x, *a);
                    x.local_tee(B_TA).i32_const(tag as i32).i32_eq();
                    x.if_(BlockType::Empty);
                    match t {
                        RegTy::Int => self.int(x, *a),
                        RegTy::Float => self.float(x, *a),
                        _ => self.boolean(x, *a),
                    }
                    x.local_set(local);
                    self.write(x, *r, tag, Some(local));
                    x.br(2);
                    x.end();
                    if *t == RegTy::Float {
                        x.local_get(B_TA).i32_const(TAG_INT as i32).i32_eq();
                        x.if_(BlockType::Empty);
                        self.int(x, *a);
                        x.f64_convert_i64_s().local_set(B_F);
                        self.write(x, *r, TAG_FLOAT, Some(B_F));
                        x.br(2);
                        x.end();
                    }
                    x.br(0);
                });
            }
            Int(r, n) => {
                x.i64_const(*n).local_set(B_A);
                self.write(x, *r, TAG_INT, Some(B_A));
            }
            Float(r, f) => {
                x.f64_const((*f).into()).local_set(B_F);
                self.write(x, *r, TAG_FLOAT, Some(B_F));
            }
            Bool(r, b) => {
                x.i32_const(*b as i32).local_set(B_X);
                self.write(x, *r, TAG_BOOL, Some(B_X));
            }
            Nil(r) => self.write(x, *r, TAG_NIL, None),
            Move(dst, a) if ty(*a) != RegTy::Value => {
                let (tag, local) = match ty(*a) {
                    RegTy::Int => {
                        self.int(x, *a);
                        (TAG_INT, B_A)
                    }
                    RegTy::Float => {
                        self.float(x, *a);
                        (TAG_FLOAT, B_F)
                    }
                    _ => {
                        self.boolean(x, *a);
                        (TAG_BOOL, B_X)
                    }
                };
                x.local_set(local);
                self.write(x, *dst, tag, Some(local));
            }
            // Copy of a value: direct if it is scalar, otherwise through the interpreter.
            Move(dst, a) if ty(*dst) == RegTy::Value => {
                self.with_slow(x, ins, span, |x| {
                    self.tag(x, *a);
                    x.local_tee(B_TA)
                        .i32_const(FIRST_HEAP_TAG as i32)
                        .i32_ge_u();
                    x.br_if(0);
                    // Both words: the payload of a `bool` is in the first one.
                    x.local_get(B_REGS).i64_load(at(roff(*a), 3)).local_set(B_A);
                    x.local_get(B_REGS)
                        .i64_load(at(roff(*a) + 8, 3))
                        .local_set(B_B);
                    self.write(x, *dst, TAG_NIL, None);
                    x.local_get(B_REGS)
                        .local_get(B_A)
                        .i64_store(at(roff(*dst), 3));
                    x.local_get(B_REGS)
                        .local_get(B_B)
                        .i64_store(at(roff(*dst) + 8, 3));
                });
            }
            IBin(op, r, a, b) => {
                self.with_slow(x, ins, span, |x| {
                    self.int(x, *a);
                    x.local_set(B_A);
                    self.int(x, *b);
                    x.local_set(B_B);
                    self.int_op(x, *op, *r, 0);
                });
            }
            IBinK(op, r, a, k) => {
                self.with_slow(x, ins, span, |x| {
                    self.int(x, *a);
                    x.local_set(B_A);
                    x.i64_const(*k).local_set(B_B);
                    self.int_op(x, *op, *r, 0);
                });
            }
            FBin(op, r, a, b) if *op != BinOp::Rem => {
                self.float(x, *a);
                self.float(x, *b);
                self.float_op(x, *op, *r);
            }
            IToF(r, a) => {
                self.int(x, *a);
                x.f64_convert_i64_s().local_set(B_F);
                self.write(x, *r, TAG_FLOAT, Some(B_F));
            }
            FNeg(r, a) => {
                self.float(x, *a);
                x.f64_neg().local_set(B_F);
                self.write(x, *r, TAG_FLOAT, Some(B_F));
            }
            Not(r, a) => {
                self.truthy(x, *a);
                x.i32_eqz().local_set(B_X);
                self.write(x, *r, TAG_BOOL, Some(B_X));
            }
            // Generic operation: inline for two integers or two floats.
            Bin(op, r, a, b) if is_arith(*op) || is_cmp(*op) => {
                self.with_slow(x, ins, span, |x| {
                    self.tag(x, *a);
                    x.local_set(B_TA);
                    self.tag(x, *b);
                    x.local_set(B_TB);
                    x.block(BlockType::Empty);
                    x.local_get(B_TA).i32_const(TAG_INT as i32).i32_eq();
                    x.local_get(B_TB).i32_const(TAG_INT as i32).i32_eq();
                    x.i32_and().i32_eqz().br_if(0);
                    self.int(x, *a);
                    x.local_set(B_A);
                    self.int(x, *b);
                    x.local_set(B_B);
                    self.int_op(x, *op, *r, 1);
                    x.br(2);
                    x.end();
                    if *op == BinOp::Rem {
                        x.br(0);
                    } else {
                        x.local_get(B_TA).i32_const(TAG_FLOAT as i32).i32_eq();
                        x.local_get(B_TB).i32_const(TAG_FLOAT as i32).i32_eq();
                        x.i32_and().i32_eqz().br_if(0);
                        self.float(x, *a);
                        self.float(x, *b);
                        self.float_op(x, *op, *r);
                    }
                });
            }
            _ => self.step(x, ins, span),
        }
    }
}

/// The code of a baseline function's blocks.
struct BaseBlocks<'a> {
    b: &'a Base<'a>,
    p: &'a Proto,
    #[allow(clippy::vec_box)]
    caches: &'a mut Vec<Box<FieldCache>>,
}

impl Blocks for BaseBlocks<'_> {
    fn block(&mut self, x: &mut InstructionSink, bi: usize) -> Exit {
        let b = self.b;
        let blk = &self.p.blocks[bi];
        for (ins, &span) in blk.instrs.iter().zip(&blk.spans) {
            b.instr(x, ins, span, self.caches);
        }
        match &blk.term {
            Term::Reraise(_) => unreachable!("function with handlers compiled"),
            Term::Jump(t) => Exit::Jump(*t),
            Term::Branch { cond, then, else_ } => {
                b.truthy(x, *cond);
                Exit::Cond(*then, *else_)
            }
            Term::IfInt {
                op,
                a,
                b: y,
                then,
                else_,
            } => {
                b.int(x, *a);
                b.int(x, *y);
                int_cmp(x, *op);
                Exit::Cond(*then, *else_)
            }
            Term::IfIntK {
                op,
                a,
                k,
                then,
                else_,
            } => {
                b.int(x, *a);
                x.i64_const(*k);
                int_cmp(x, *op);
                Exit::Cond(*then, *else_)
            }
            Term::Return { reg, shape } => {
                x.local_get(1).i32_const(*reg as i32);
                x.i32_store(at(std::mem::offset_of!(NativeFrame, ret) as u64, 2));
                x.local_get(1).i32_const(shape.unwrap_or(u32::MAX) as i32);
                x.i32_store(at(std::mem::offset_of!(NativeFrame, ret_shape) as u64, 2));
                x.i32_const(0).return_();
                Exit::Return
            }
            Term::IterNext {
                iter,
                dst,
                body,
                done,
            } => {
                x.local_get(1)
                    .i32_const(*iter as i32)
                    .i32_const(*dst as i32);
                x.i32_const(fn_index(bk_iter_next as *const () as usize))
                    .call_indirect(0, b.t_iter);
                Exit::Cond(*body, *done)
            }
        }
    }
}

impl Jit {
    /// Compiles `p` to baseline code; returns its entry point (a table index).
    pub fn compile_dynamic(&mut self, p: &Proto) -> Option<usize> {
        // `rescue` / `ensure`: left to the interpreter.
        if !p.handlers.is_empty() {
            return None;
        }
        let compiler = COMPILER.with(Cell::get)?;
        let mut types = Types::default();
        let entry_ty = types.index(vec![ValType::I32; 2], vec![ValType::I32]);
        let b = Base {
            regs_ty: &p.regs,
            t_step: types.index(vec![ValType::I32; 5], vec![ValType::I32]),
            t_iter: types.index(vec![ValType::I32; 3], vec![ValType::I32]),
            t_drop: types.index(vec![ValType::I32], vec![]),
        };
        let mut f = Function::new([
            (1, ValType::I32),
            (4, ValType::I64),
            (2, ValType::I32),
            (1, ValType::F64),
            (2, ValType::I32),
        ]);
        let mut x = f.instructions();
        x.local_get(1)
            .i32_load(at(std::mem::offset_of!(NativeFrame, regs) as u64, 2))
            .local_set(B_REGS);
        let reached = vec![true; p.blocks.len()];
        let mut bb = BaseBlocks {
            b: &b,
            p,
            caches: &mut self.field_caches,
        };
        control_flow(&mut x, &successors(p), &reached, B_PC, &mut bb);
        x.unreachable();
        x.end();
        let mut funcs = FunctionSection::new();
        funcs.function(entry_ty);
        let mut code = CodeSection::new();
        code.function(&f);
        let mut exports = ExportSection::new();
        exports.export("f0", ExportKind::Func, 0);
        let base = compiler(&assemble(&types, &funcs, &exports, &code))?;
        BASELINE.with(|c| c.set(c.get() + 1));
        Some(base as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn structure(succ: &[&[u32]]) -> Option<Structure> {
        let succ: Vec<Vec<u32>> = succ.iter().map(|s| s.to_vec()).collect();
        Structure::new(&succ, &vec![true; succ.len()])
    }

    #[test]
    fn loops_and_merges() {
        // 0 → 1; 1 → 2 | 4; 2 → 3 | 1 (back); 3 → 1 (back); 4: return.
        let s = structure(&[&[1], &[2, 4], &[3, 1], &[1], &[]]).unwrap();
        assert_eq!(s.header, [false, true, false, false, false]);
        assert_eq!(s.merge, [false; 5]);
        // An `if` whose branches meet: 0 → 1 | 2; 1 → 3; 2 → 3.
        let s = structure(&[&[1, 2], &[3], &[3], &[]]).unwrap();
        assert_eq!(s.merge, [false, false, false, true]);
        let mut c = s.children[0].clone();
        c.sort();
        assert_eq!(c, [1, 2, 3]);
    }

    #[test]
    fn irreducible_graphs_are_not_structured() {
        // Two entries into the loop 1 ⇄ 2.
        assert!(structure(&[&[1, 2], &[2], &[1]]).is_none());
    }

    #[test]
    fn unreached_blocks_are_left_out() {
        let succ = vec![vec![2], vec![1], vec![]];
        let s = Structure::new(&succ, &[true, false, true]).unwrap();
        assert_eq!(s.order[1], UNREACHED);
        assert!(!s.header[1]);
    }
}
