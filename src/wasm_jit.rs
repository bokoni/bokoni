//! Compilation of typed functions to WebAssembly, for Bokoni itself running in
//! WebAssembly, where the Cranelift JIT cannot (`wasm-jit` feature; `docs/execution.md`).
//!
//! A function is written as a WebAssembly module (`wasm-encoder`), which the host compiles
//! to machine code: the embedder gives a compiler (`set_compiler`) that instantiates the
//! module against this program's memory and function table, appends its exports `f0`,
//! `f1`… to that table and returns the index of the first one. A function pointer being
//! an index in that table, Rust then calls the code like the Cranelift JIT's.
//!
//! Phase 1: functions whose parameters and result are annotated `int`, `float` or `bool`
//! (or a copy typed for its arguments) and whose body only computes with such values,
//! compares, branches and calls such functions (recursion included). Any other function
//! stays interpreted. Same rules as the interpreter: overflow, Euclidean division,
//! division by zero, call depth, with the same messages and the same call stack.
//!
//! - Each function has an internal function, `(ctx, params…) -> (status, result)`
//!   (status 0: success, 1: error, described in `ctx`), and an entry point for Rust with
//!   the C ABI of the Cranelift JIT's: `extern "C" fn(ctx, *const u64 args, *mut u64
//!   result) -> i32`.
//! - The functions compiled together (one calling another not compiled yet) call each
//!   other directly; one compiled before is called through the table (`call_indirect`).
//! - Control flow: the blocks of the IR are dispatched by a loop over a `br_table` on the
//!   block number; a jump to the next block falls through.
//! - A register holds a value of one native type at each point (`Plan`); one WebAssembly
//!   local per register and type.
//! - Redefining a global function called (directly or not) invalidates the code, which is
//!   compiled again when needed.

use crate::ast::BinOp;
use crate::diagnostic::Error;
use crate::globals::Globals;
use crate::intern::{FxHashMap, Symbol};
use crate::ir::{Instr, JitState, NativeCode, Proto, ProtoKind, Reg, RegTy, RtType, Script, Term};
use crate::span::Span;
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
}

/// Sets the compiler of the generated modules (WebAssembly only: elsewhere, nothing is
/// compiled). Without one, everything is interpreted.
pub fn set_compiler(compiler: Option<Compiler>) {
    COMPILER.with(|c| c.set(compiler));
}

/// Number of functions compiled so far (tests).
pub fn compiled_functions() -> u32 {
    COMPILED.with(Cell::get)
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

/// Parameters and result of a function the generated code can run.
fn native_sig(p: &Proto) -> Option<(Vec<RegTy>, RegTy)> {
    if p.kind != ProtoKind::Function
        || p.ncells != 0
        || !p.captures.is_empty()
        || !p.handlers.is_empty()
        || p.is_generic()
    {
        return None;
    }
    let params = p
        .params
        .iter()
        .map(|prm| {
            if prm.has_default {
                return None;
            }
            prim(&prm.ty.as_ref()?.ty)
        })
        .collect::<Option<Vec<_>>>()?;
    let ret = prim(&p.ret.as_ref()?.ty)?;
    Some((params, ret))
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
    targets: FxHashMap<(usize, usize), Rc<Proto>>,
    /// Global slots read (the functions called).
    slots: Vec<u32>,
}

/// Type written by `ins` (`Some(None)`: nothing), operands read from `cur`; `None`: not
/// compilable. Records the function called by a `Call` into `target`.
fn transfer(
    p: &Proto,
    globals: &Globals,
    ins: &Instr,
    cur: &[Lat],
    target: &mut Option<Rc<Proto>>,
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
            let (params, ret) = native_sig(&callee)?;
            if params.len() != args.len()
                || !params.iter().zip(&args).all(|(p, a)| fits(of(*p), *a))
            {
                return None;
            }
            *target = Some(callee);
            of(ret)
        }
        _ => return None,
    }))
}

fn analyze(p: &Proto, globals: &Globals) -> Option<Plan> {
    let (params, ret) = native_sig(p)?;
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
                let t = transfer(p, globals, ins, &cur, &mut target, &mut slots)?;
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
                    (RegTy::Float, Some(T::Int)) => true,
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
                _ => analyze(&p, globals),
            };
            if let Some(plan) = &plan {
                todo.extend(plan.targets.values().cloned());
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
                        let k = Rc::as_ptr(q) as usize;
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
            .section(&funcs)
            .section(&exports)
            .section(&code);
        m.finish()
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
        let n = p.blocks.len() as u32;
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
        x.loop_(BlockType::Empty);
        for _ in 0..n {
            x.block(BlockType::Empty);
        }
        x.local_get(pc);
        x.br_table(0..n, n - 1);
        x.end();
        for (bi, b) in p.blocks.iter().enumerate() {
            // Depth of the dispatch loop from this block's code.
            let to_loop = n - 1 - bi as u32;
            let Some(entry) = &plan.entry[bi] else {
                // Never reached.
                x.unreachable();
                if bi + 1 < n as usize {
                    x.end();
                }
                continue;
            };
            let mut cur = entry.clone();
            for (k, ins) in b.instrs.iter().enumerate() {
                let span = b.spans[k];
                let def = plan.defs[bi][k];
                let ctx = Ctx2 {
                    fx: &fx,
                    cur: &cur,
                    s,
                    ret: plan.ret,
                };
                match ins {
                    Instr::Call { dst, site, .. } => {
                        let q = &plan.targets[&(bi, k)];
                        let (cparams, cret) = native_sig(q).unwrap();
                        let depth_err = self.error_id(ErrKind::Depth, span, &p.script);
                        // Depth: as the interpreter, an error at the call beyond the limit.
                        x.local_get(0).i64_load(mem(DEPTH)).local_tee(s[0]);
                        x.local_get(0).i64_load(mem(MAX_DEPTH)).i64_ge_s();
                        x.if_(BlockType::Empty);
                        ctx.fail(&mut x, depth_err);
                        x.end();
                        x.local_get(0).local_get(s[0]).i64_const(1).i64_add();
                        x.i64_store(mem(DEPTH));
                        x.local_get(0);
                        let site = &p.calls[*site as usize];
                        for (k, pt) in cparams.iter().enumerate() {
                            let r = site.start + k as u32;
                            ctx.get(&mut x, r);
                            convert(&mut x, cur[r as usize].get().unwrap(), of(*pt));
                        }
                        match callees[&(Rc::as_ptr(q) as usize)] {
                            Callee::Local(i) => {
                                x.call(i);
                            }
                            Callee::Table(i) => {
                                let ty = types.internal(&cparams, cret);
                                x.i32_const(i as i32).call_indirect(0, ty);
                            }
                        }
                        let tmp = match cret {
                            RegTy::Int => s[1],
                            RegTy::Float => sf,
                            _ => si,
                        };
                        x.local_set(tmp).local_set(st);
                        x.local_get(0).local_get(s[0]).i64_store(mem(DEPTH));
                        self.traces.push(TraceInfo {
                            name: q.name.clone(),
                            span,
                        });
                        let tid = (self.traces.len() - 1) as i32;
                        x.local_get(st).if_(BlockType::Empty);
                        x.local_get(0)
                            .i32_const(tid)
                            .i32_const(table_index(bk_trace as *const () as usize))
                            .call_indirect(0, trace_ty);
                        x.i32_const(1);
                        zero(&mut x, plan.ret);
                        x.return_();
                        x.end();
                        let d = def.unwrap();
                        x.local_get(tmp);
                        convert(&mut x, of(cret), d);
                        x.local_set(fx.at(*dst, d));
                    }
                    ins => {
                        let mut err = |kind| self.error_id(kind, span, &p.script);
                        ctx.instr(&mut x, ins, def, &mut err, frem_ty);
                    }
                }
                if let (Some(d), Some(t)) = (written(ins), def) {
                    cur[d as usize] = Lat::T(t);
                }
            }
            let ctx = Ctx2 {
                fx: &fx,
                cur: &cur,
                s,
                ret: plan.ret,
            };
            let next = bi as u32 + 1;
            let jump = |x: &mut InstructionSink, target: u32, extra: u32| {
                if target != next || extra > 0 {
                    x.i32_const(target as i32).local_set(pc).br(to_loop + extra);
                }
            };
            match &b.term {
                Term::Jump(t) => jump(&mut x, *t, 0),
                Term::Branch { cond, then, else_ } => {
                    match cur[*cond as usize].get().unwrap() {
                        T::Bool => ctx.get(&mut x, *cond),
                        T::Nil => {
                            x.i32_const(0);
                        }
                        // A number is true.
                        _ => {
                            x.i32_const(1);
                        }
                    }
                    x.if_(BlockType::Empty);
                    jump(&mut x, *then, 1);
                    x.end();
                    jump(&mut x, *else_, 0);
                }
                Term::IfInt {
                    op,
                    a,
                    b: y,
                    then,
                    else_,
                } => {
                    ctx.get(&mut x, *a);
                    ctx.get(&mut x, *y);
                    int_cmp(&mut x, *op);
                    x.if_(BlockType::Empty);
                    jump(&mut x, *then, 1);
                    x.end();
                    jump(&mut x, *else_, 0);
                }
                Term::IfIntK {
                    op,
                    a,
                    k,
                    then,
                    else_,
                } => {
                    ctx.get(&mut x, *a);
                    x.i64_const(*k);
                    int_cmp(&mut x, *op);
                    x.if_(BlockType::Empty);
                    jump(&mut x, *then, 1);
                    x.end();
                    jump(&mut x, *else_, 0);
                }
                Term::Return { reg, .. } => {
                    x.i32_const(0);
                    ctx.get(&mut x, *reg);
                    convert(&mut x, cur[*reg as usize].get().unwrap(), of(plan.ret));
                    x.return_();
                }
                _ => unreachable!("rejected by the analysis"),
            }
            if bi + 1 < n as usize {
                x.end();
            }
        }
        // End of the dispatch loop: never reached (every block ends with a jump or a
        // return).
        x.end();
        x.unreachable();
        x.end();
        f
    }
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
