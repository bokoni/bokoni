//! Native compilation (JIT) with Cranelift.
//!
//! Compiled functions are those whose parameters and return are annotated `int`, `float`
//! or `bool`, and whose body uses only arithmetic, comparisons, branches and calls to other
//! compilable functions (recursion included). Everything else runs in the interpreter: same
//! result, only slower.
//!
//! - The IR's `value` registers are typed by an analysis: the result of calling a `: int`
//!   function is an `int`, and so on. If a register cannot be typed, the function is not
//!   compiled.
//! - Same rules as the interpreter: overflow, division by zero and call depth are checked,
//!   with the same messages and the same call stack.
//! - A call to a global function is direct (native code → native code). If the global is
//!   redefined, the calling code is invalidated, then recompiled when needed.
//!
//! Internal calling convention: `fn(ctx, params…) -> (status, result)`, status 0 = success,
//! 1 = error (described in `ctx`). Each function also has an entry point for Rust:
//! `extern "C" fn(ctx, *const u64 args, *mut u64 ret) -> i32`.

use crate::ast::BinOp;
use crate::diagnostic::Error;
use crate::globals::Globals;
use crate::intern::FxHashMap;
use crate::ir::NativeFrame;
use crate::ir::{Instr, JitState, NativeCode, Proto, ProtoKind, Reg, RegTy, RtType, Script, Term};
use crate::span::Span;
use crate::value::layout::{FIRST_HEAP_TAG, PAYLOAD, SIZE, TAG_BOOL, TAG_FLOAT, TAG_INT, TAG_NIL};
use crate::value::{Function, Value};
use crate::vm::Runtime;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    AbiParam, InstBuilder, MemFlagsData, Signature, Type, UserFuncName, types,
};
use cranelift_codegen::ir::{Block as CBlock, FuncRef, Value as CValue};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use std::rc::Rc;

/// Context shared with native code. The first three fields are read and written by the
/// generated code (fixed offsets); the others are used by Rust code only.
#[repr(C)]
pub struct JitCtx {
    pub depth: i64,
    pub max_depth: i64,
    /// Error that occurred: index into the error table.
    pub err: i64,
    /// Call stack of the error (indices into the call table), most recent first.
    pub traces: Vec<u32>,
}

const DEPTH_OFFSET: i32 = 0;
const MAX_DEPTH_OFFSET: i32 = 8;
const ERR_OFFSET: i32 = 16;

#[derive(Clone, Copy)]
enum ErrKind {
    Overflow,
    DivZero,
    Depth,
    /// An assumption of the native code did not hold (field of another type): the call is
    /// redone by the interpreter.
    Bail,
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

extern "C" fn bk_trace(ctx: *mut JitCtx, id: u32) {
    // SAFETY: `ctx` is the context passed by `Jit::call`, valid during the call.
    unsafe { (*ctx).traces.push(id) }
}

extern "C" fn bk_frem(a: f64, b: f64) -> f64 {
    a.rem_euclid(b)
}

/// Fields of an object of type `info` (passed as a parameter); null if the object is of
/// another type (type redeclared since compilation).
extern "C" fn bk_obj_fields(
    obj: *const Value,
    info: *const crate::objects::TypeInfo,
) -> *const Value {
    // SAFETY: `obj` is an argument alive during the native call. Native code has no side
    // effects: the fields neither change nor move during the call.
    unsafe {
        match &*obj {
            Value::Object(o) => match o.try_borrow() {
                Ok(b) if Rc::as_ptr(&b.info) == info => b.fields.as_ptr(),
                _ => std::ptr::null(),
            },
            _ => std::ptr::null(),
        }
    }
}

/// Native construction of an object (`info`) or a list (no `info`): for each argument,
/// the target field (object) and the conversion.
struct BuildPlan {
    info: Option<Rc<crate::objects::TypeInfo>>,
    items: Vec<(usize, Kind)>,
}

/// Script value from the bits of a native value.
///
/// # Safety
/// For `Kind::Ref`, `bits` is the address of a live value.
unsafe fn from_bits(kind: Kind, bits: u64) -> Value {
    match kind {
        Kind::Int => Value::Int(bits as i64),
        Kind::Float => Value::Float(f64::from_bits(bits)),
        Kind::IntToFloat => Value::Float(bits as i64 as f64),
        Kind::Bool => Value::Bool(bits & 0xff != 0),
        // SAFETY: see above.
        Kind::Ref => unsafe { (*(bits as *const Value)).clone() },
    }
}

/// Builds the object or list described by `plan` into slot `dst` (the old value is
/// dropped afterwards: an argument may refer to `dst`).
extern "C" fn bk_build(dst: *mut Value, args: *const u64, plan: *const BuildPlan) {
    // SAFETY: `dst` is a slot of the native frame, `args` holds one argument per element of
    // `plan` (kept alive by the `Jit`), the objects referred to live during the call.
    unsafe {
        let plan = &*plan;
        let v = match &plan.info {
            Some(info) => {
                let mut fields = vec![Value::Nil; plan.items.len()];
                for (k, (field, kind)) in plan.items.iter().enumerate() {
                    fields[*field] = from_bits(*kind, *args.add(k));
                }
                Value::Object(Rc::new(std::cell::RefCell::new(crate::objects::Object {
                    info: info.clone(),
                    fields,
                })))
            }
            None => {
                let items = plan
                    .items
                    .iter()
                    .enumerate()
                    .map(|(k, (_, kind))| from_bits(*kind, *args.add(k)))
                    .collect();
                Value::List(Rc::new(std::cell::RefCell::new(items)))
            }
        };
        *dst = v;
    }
}

/// Copies `*src` (reference counted) into slot `dst`.
extern "C" fn bk_clone_into(dst: *mut Value, src: *const Value) {
    // SAFETY: `dst` is a slot of the native frame, `src` a live value.
    unsafe {
        let v = (*src).clone();
        *dst = v;
    }
}

/// Result of a native function into the caller's slot `dst`: moved out of a slot of the
/// function (`owned`), copied otherwise (borrowed value).
extern "C" fn bk_move_out(dst: *mut Value, src: *mut Value, owned: i32) {
    // SAFETY: `dst` is a slot of the caller, `src` a live value; if `owned`, `src` is a slot
    // of the function that is returning.
    unsafe {
        let v = if owned != 0 {
            std::mem::replace(&mut *src, Value::Nil)
        } else {
            (*src).clone()
        };
        *dst = v;
    }
}

/// Drops the `n` slots of the native frame.
extern "C" fn bk_drop_slots(base: *mut Value, n: u64) {
    // SAFETY: `base` points to `n` slots of the frame, initialized (at least to `nil`).
    unsafe {
        for i in 0..n as usize {
            let slot = base.add(i);
            std::ptr::drop_in_place(slot);
            slot.write(Value::Nil);
        }
    }
}

/// Elements of a list: address in `out[0]`, count in `out[1]`; 1 if it is not a
/// list.
extern "C" fn bk_list_view(v: *const Value, out: *mut u64) -> i32 {
    // SAFETY: `v` is a value alive during the native call, `out` two slots. Native code has
    // no side effects: the list does not change during the call.
    unsafe {
        let Value::List(l) = &*v else {
            return 1;
        };
        let Ok(items) = l.try_borrow() else {
            return 1;
        };
        *out = items.as_ptr() as u64;
        *out.add(1) = items.len() as u64;
    }
    0
}

/// Field of an object passed as a parameter (`FieldAs`): 0 and the value (bits) in
/// `out`, or 1 if the object has no such field with that type.
extern "C" fn bk_field_native(obj: *const Value, cache: *mut u8, out: *mut u64) -> i32 {
    // SAFETY: `obj` is an argument alive during the native call (see `Jit::call`); `cache`
    // is a `FieldCache` kept alive by the `Jit`.
    unsafe {
        let cache = &mut *(cache as *mut FieldCache);
        let bits = match cache.get(&*obj) {
            Some(Value::Int(i)) => i as u64,
            Some(Value::Float(f)) => f.to_bits(),
            Some(Value::Bool(b)) => b as u64,
            _ => return 1,
        };
        *out = bits;
    }
    0
}

/// Compiled function: Cranelift id and signature.
struct Compiled {
    id: FuncId,
    params: Vec<RegTy>,
    ret: RegTy,
}

pub struct Jit {
    module: JITModule,
    ctx: Box<JitCtx>,
    errors: Vec<ErrInfo>,
    traces: Vec<TraceInfo>,
    /// Functions already compiled (key: address of the `Proto`).
    compiled: FxHashMap<usize, Compiled>,
    trace_fn: FuncId,
    frem_fn: FuncId,
    field_fn: FuncId,
    fields_fn: FuncId,
    list_fn: FuncId,
    /// `bk_build`, `bk_clone_into`, `bk_move_out`, `bk_drop_slots`.
    mem_fns: [FuncId; 4],
    /// Construction plans (addresses held by native code).
    #[allow(clippy::vec_box)]
    build_plans: Vec<Box<BuildPlan>>,
    /// Types whose layout native code knows (kept alive: their address serves as their
    /// identity).
    known_types: Vec<Rc<crate::objects::TypeInfo>>,
    /// The last native call was abandoned (`ErrKind::Bail`).
    pub bailed: bool,
    /// Symbol of the `len` method (lists).
    pub len_sym: crate::intern::Symbol,
    /// Helpers of the baseline code (`bk_step`, `bk_iter_next`, `bk_drop`).
    dyn_fns: Option<(FuncId, FuncId, FuncId)>,
    /// Fast helpers (`FAST`), in the order of `Fast`.
    fast_fns: Vec<FuncId>,
    /// Field access caches of the baseline code: boxed, because native code keeps their
    /// address.
    #[allow(clippy::vec_box)]
    field_caches: Vec<Box<FieldCache>>,
    next: u32,
}

/// Type of a register for native code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum JTy {
    Int,
    Float,
    Bool,
    /// Never read (`nil` of an expression whose value is ignored).
    Nil,
    /// Global function called directly (global slot).
    Callee(u32),
    /// Object of type `anns[i]`: pointer to the value (borrowed from the caller, or in a
    /// slot of the native frame).
    Obj(u32),
    /// List of elements of type `anns[i]`: pointer to the value.
    List(u32),
    /// Iterator over a list of elements of type `anns[i]`.
    Iter(u32),
    /// `type` in a global slot, called to construct an object.
    Ctor(u32),
}

fn prim(rt: &RtType) -> Option<RegTy> {
    match rt {
        RtType::Int => Some(RegTy::Int),
        RtType::Float => Some(RegTy::Float),
        RtType::Bool => Some(RegTy::Bool),
        _ => None,
    }
}

fn jty(t: RegTy) -> JTy {
    match t {
        RegTy::Int => JTy::Int,
        RegTy::Float => JTy::Float,
        RegTy::Bool => JTy::Bool,
        RegTy::Value => unreachable!(),
    }
}

/// Cranelift type of a register; `Value`: pointer to an object.
fn clif(t: RegTy) -> Type {
    match t {
        RegTy::Int | RegTy::Value => types::I64,
        RegTy::Float => types::F64,
        RegTy::Bool => types::I8,
    }
}

/// Native type of an annotation: simple, or `Value` (pointer) for an object of a declared
/// type and a list of simple elements or objects.
fn native_ty(rt: &RtType) -> Option<RegTy> {
    match rt {
        RtType::Named(..) => Some(RegTy::Value),
        RtType::List(e) if matches!(**e, RtType::Named(..)) || prim(e).is_some() => {
            Some(RegTy::Value)
        }
        t => prim(t),
    }
}

/// Native signature of a function (parameters and return), otherwise `None`.
fn native_sig(p: &Proto) -> Option<(Vec<RegTy>, RegTy)> {
    if p.kind != ProtoKind::Function || !p.captures.is_empty() || p.ncells != 0 {
        return None;
    }
    let mut params = Vec::new();
    for prm in &p.params {
        if prm.has_default || prm.field.is_some() {
            return None;
        }
        params.push(native_ty(&prm.ty.as_ref()?.ty)?);
    }
    Some((params, native_ty(&p.ret.as_ref()?.ty)?))
}

/// Annotation of parameter `i`.
fn ann_of(q: &Proto, i: usize) -> Option<&RtType> {
    q.params[i].ty.as_ref().map(|t| &*t.ty)
}

/// Same type: a declared type is identified by its global slot.
fn same_ann(a: &RtType, b: &RtType) -> bool {
    match (a, b) {
        (RtType::Named(x, _), RtType::Named(y, _)) => x == y,
        (RtType::List(x), RtType::List(y)) => same_ann(x, y),
        _ => a == b,
    }
}

/// Index of `t` in the type table (added if needed).
fn intern_ann(anns: &mut Vec<RtType>, t: &RtType) -> u32 {
    match anns.iter().position(|a| same_ann(a, t)) {
        Some(i) => i as u32,
        None => {
            anns.push(t.clone());
            (anns.len() - 1) as u32
        }
    }
}

/// Native type of a value annotated `t` (`native_ty` accepts it).
fn value_jty(anns: &mut Vec<RtType>, t: &RtType) -> JTy {
    match t {
        RtType::Named(..) => JTy::Obj(intern_ann(anns, t)),
        RtType::List(e) => JTy::List(intern_ann(anns, e)),
        t => jty(prim(t).expect("simple type")),
    }
}

/// Script function in a global slot.
fn global_proto(globals: &Globals, slot: u32) -> Option<Rc<Proto>> {
    match &globals.at(slot).value {
        Some(Value::Function(f)) => match &**f {
            Function::Script(c) => Some(c.proto.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// `type` (not `class`) in a global slot.
fn global_type(globals: &Globals, slot: u32) -> Option<Rc<crate::objects::TypeInfo>> {
    match &globals.at(slot).value {
        Some(Value::Type(info)) if info.kind == crate::ast::TypeDeclKind::Type => {
            Some(info.clone())
        }
        _ => None,
    }
}

/// Conversion of a native value to a script value (construction).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Int,
    Float,
    /// `int` given for a `float` field.
    IntToFloat,
    Bool,
    /// Object or list: the pointed-to value is copied (reference counted).
    Ref,
}

/// Native value of type `t` to a value, for a field annotated `want` (`None`: not
/// annotated). `None` if the conversion is not guaranteed.
fn kind_for(anns: &[RtType], t: JTy, want: Option<&RtType>) -> Option<Kind> {
    let kind = match t {
        JTy::Int => Kind::Int,
        JTy::Float => Kind::Float,
        JTy::Bool => Kind::Bool,
        JTy::Obj(_) | JTy::List(_) => Kind::Ref,
        _ => return None,
    };
    let Some(want) = want else {
        return Some(kind);
    };
    match (want, t) {
        (RtType::Int, JTy::Int) | (RtType::Float, JTy::Float) | (RtType::Bool, JTy::Bool) => {
            Some(kind)
        }
        (RtType::Float, JTy::Int) => Some(Kind::IntToFloat),
        (w @ RtType::Named(..), JTy::Obj(a)) if same_ann(w, &anns[a as usize]) => Some(kind),
        (RtType::List(e), JTy::List(a)) if same_ann(e, &anns[a as usize]) => Some(kind),
        _ => None,
    }
}

/// Construction `T(…)`: for each argument, (field, conversion).
fn ctor_plan(
    info: &crate::objects::TypeInfo,
    site: &crate::ir::CallSite,
    tys: &[Option<JTy>],
    anns: &[RtType],
) -> Option<Vec<(usize, Kind)>> {
    let n = site.npos as usize + site.named.len();
    if site.block || n != info.fields.len() || site.shapes.iter().any(Option::is_some) {
        return None;
    }
    let mut seen = vec![false; n];
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let field = match k.checked_sub(site.npos as usize) {
            None => k,
            Some(j) => info.field_index(site.named[j])?,
        };
        if std::mem::replace(&mut seen[field], true) {
            return None;
        }
        let t = tys[site.start as usize + k]?;
        let want = info.fields[field].ty.as_ref().map(|t| &*t.ty);
        out.push((field, kind_for(anns, t, want)?));
    }
    Some(out)
}

/// Type of a register at a program point.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lat {
    /// Not written yet.
    Undef,
    T(JTy),
    /// Different types depending on the path (or not written on one of them): unreadable.
    Conflict,
}

impl Lat {
    fn get(self) -> Option<JTy> {
        match self {
            Lat::T(t) => Some(t),
            _ => None,
        }
    }

    fn merge(self, o: Lat) -> Lat {
        if self == o { self } else { Lat::Conflict }
    }
}

/// Result of analysing a function: type of the registers at each point.
struct Plan {
    /// Types at the entry of each block.
    entry: Vec<Vec<Lat>>,
    /// Type written by each instruction of each block (and by the `IterNext` terminator),
    /// if any.
    defs: Vec<Vec<Option<JTy>>>,
    term_defs: Vec<Option<JTy>>,
    /// Functions called directly.
    callees: Vec<u32>,
    /// Constructed types (`T(…)`).
    ctors: Vec<u32>,
    /// Types of objects and list elements (`JTy::Obj` / `List` / `Iter`).
    anns: Vec<RtType>,
}

/// Type of an element of type `anns[a]`.
fn elem_jty(anns: &[RtType], a: u32) -> JTy {
    match &anns[a as usize] {
        RtType::Named(..) => JTy::Obj(a),
        t => jty(prim(t).expect("simple element or object")),
    }
}

/// Analysis of a function: context shared by the instructions.
struct Analysis<'a> {
    p: &'a Proto,
    globals: &'a Globals,
    len: crate::intern::Symbol,
    anns: Vec<RtType>,
    callees: Vec<u32>,
    ctors: Vec<u32>,
}

impl Analysis<'_> {
    fn sig_of(&self, slot: u32) -> Option<(Vec<RegTy>, RegTy)> {
        global_proto(self.globals, slot).and_then(|q| native_sig(&q))
    }

    /// Type written by `ins` (operands read from `cur`); `None`: not compilable.
    fn instr(&mut self, ins: &Instr, cur: &[Lat]) -> Option<Option<JTy>> {
        use Instr::*;
        let g = |r: Reg| cur[r as usize].get();
        let value = |t: JTy| {
            matches!(
                t,
                JTy::Int | JTy::Float | JTy::Bool | JTy::Obj(_) | JTy::List(_)
            )
        };
        let numeric = |t: JTy| matches!(t, JTy::Int | JTy::Float);
        let p = self.p;
        Some(Some(match ins {
            Int(..) => JTy::Int,
            Float(..) => JTy::Float,
            Bool(..) => JTy::Bool,
            Nil(_) => JTy::Nil,
            Move(_, a) => Some(g(*a)?).filter(|t| value(*t))?,
            Guard(_, a, _) => {
                Some(g(*a)?).filter(|t| matches!(t, JTy::Int | JTy::Float | JTy::Bool))?
            }
            FieldAs(_, a, _, t) => match g(*a)? {
                JTy::Obj(_) => jty(*t),
                _ => return None,
            },
            IterNew { src, .. } => match g(*src)? {
                JTy::List(a) => JTy::Iter(a),
                _ => return None,
            },
            Index(_, a, i) => match (g(*a)?, g(*i)?) {
                (JTy::List(e), JTy::Int) => elem_jty(&self.anns, e),
                _ => return None,
            },
            CallMethod {
                recv, name, site, ..
            } => {
                let site = &p.calls[*site as usize];
                let ok = *name == self.len
                    && matches!(g(*recv)?, JTy::List(_))
                    && site.npos == 0
                    && site.named.is_empty()
                    && !site.block;
                if !ok {
                    return None;
                }
                JTy::Int
            }
            // List literal: elements all of the same type.
            List(_, start, n) => {
                let first = g(*start)?;
                if *n == 0 || (0..*n).any(|k| g(*start + k) != Some(first)) {
                    return None;
                }
                let e = match first {
                    JTy::Int => RtType::Int,
                    JTy::Float => RtType::Float,
                    JTy::Bool => RtType::Bool,
                    JTy::Obj(a) => self.anns[a as usize].clone(),
                    _ => return None,
                };
                JTy::List(intern_ann(&mut self.anns, &e))
            }
            IBin(op, _, a, b) => {
                if g(*a)? != JTy::Int || g(*b)? != JTy::Int {
                    return None;
                }
                if is_cmp(*op) { JTy::Bool } else { JTy::Int }
            }
            IBinK(op, _, a, _) => {
                if g(*a)? != JTy::Int {
                    return None;
                }
                if is_cmp(*op) { JTy::Bool } else { JTy::Int }
            }
            FBin(op, _, a, b) => {
                if g(*a)? != JTy::Float || g(*b)? != JTy::Float {
                    return None;
                }
                if is_cmp(*op) { JTy::Bool } else { JTy::Float }
            }
            INeg(_, a) => (g(*a) == Some(JTy::Int)).then_some(JTy::Int)?,
            FNeg(_, a) => (g(*a) == Some(JTy::Float)).then_some(JTy::Float)?,
            IToF(_, a) => (g(*a) == Some(JTy::Int)).then_some(JTy::Float)?,
            Not(_, a) => match g(*a)? {
                JTy::Int | JTy::Float | JTy::Bool => JTy::Bool,
                _ => return None,
            },
            Bin(op, _, a, b) => {
                let (ta, tb) = (g(*a)?, g(*b)?);
                match (*op, ta, tb) {
                    (op, JTy::Int, JTy::Int) if is_arith(op) => JTy::Int,
                    (op, _, _) if is_arith(op) && numeric(ta) && numeric(tb) => JTy::Float,
                    (op, _, _) if is_cmp(op) && numeric(ta) && numeric(tb) => JTy::Bool,
                    (BinOp::Eq | BinOp::Ne, JTy::Bool, JTy::Bool) => JTy::Bool,
                    _ => return None,
                }
            }
            GetGlobal(_, slot) => {
                if global_type(self.globals, *slot).is_some() {
                    if !self.ctors.contains(slot) {
                        self.ctors.push(*slot);
                    }
                    JTy::Ctor(*slot)
                } else {
                    self.sig_of(*slot)?;
                    if !self.callees.contains(slot) {
                        self.callees.push(*slot);
                    }
                    JTy::Callee(*slot)
                }
            }
            Call { f, site, .. } => {
                let site = &p.calls[*site as usize];
                match g(*f)? {
                    JTy::Callee(slot) => {
                        let (cparams, _) = self.sig_of(slot)?;
                        if site.block
                            || !site.named.is_empty()
                            || site.npos as usize != cparams.len()
                        {
                            return None;
                        }
                        let callee = global_proto(self.globals, slot)?;
                        for (k, pt) in cparams.iter().enumerate() {
                            let at = g(site.start + k as u32)?;
                            let ok = match (pt, at) {
                                (RegTy::Float, JTy::Int | JTy::Float) => true,
                                // Object or list passed as is: same type, already checked.
                                (RegTy::Value, JTy::Obj(_) | JTy::List(_)) => {
                                    kind_for(&self.anns, at, ann_of(&callee, k)) == Some(Kind::Ref)
                                }
                                (t, a) => *t != RegTy::Value && jty(*t) == a,
                            };
                            if !ok {
                                return None;
                            }
                        }
                        value_jty(&mut self.anns, &callee.ret.as_ref()?.ty)
                    }
                    JTy::Ctor(slot) => {
                        let info = global_type(self.globals, slot)?;
                        let tys: Vec<Option<JTy>> = cur.iter().map(|l| l.get()).collect();
                        ctor_plan(&info, site, &tys, &self.anns)?;
                        let t = RtType::Named(slot, crate::intern::Symbol(u32::MAX));
                        JTy::Obj(intern_ann(&mut self.anns, &t))
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }))
    }
}

/// Register types of a compilable function; `None` if it is not compilable.
fn analyze(p: &Proto, globals: &Globals, len: crate::intern::Symbol) -> Option<Plan> {
    let (_, ret) = native_sig(p)?;
    let mut an = Analysis {
        p,
        globals,
        len,
        anns: Vec::new(),
        callees: Vec::new(),
        ctors: Vec::new(),
    };
    // Typed IR registers: their type does not change.
    let fixed: Vec<Option<JTy>> = p
        .regs
        .iter()
        .map(|t| (*t != RegTy::Value).then(|| jty(*t)))
        .collect();
    let mut start: Vec<Lat> = fixed.iter().map(|t| t.map_or(Lat::Undef, Lat::T)).collect();
    for prm in &p.params {
        start[prm.reg as usize] = Lat::T(value_jty(&mut an.anns, &prm.ty.as_ref()?.ty));
    }
    let ret_jty = value_jty(&mut an.anns, &p.ret.as_ref()?.ty);
    let nb = p.blocks.len();
    let mut entry: Vec<Option<Vec<Lat>>> = vec![None; nb];
    entry[0] = Some(start);
    let mut defs: Vec<Vec<Option<JTy>>> = p
        .blocks
        .iter()
        .map(|b| vec![None; b.instrs.len()])
        .collect();
    let mut term_defs: Vec<Option<JTy>> = vec![None; nb];
    loop {
        let mut changed = false;
        for (bi, b) in p.blocks.iter().enumerate() {
            let Some(mut cur) = entry[bi].clone() else {
                continue;
            };
            for (k, ins) in b.instrs.iter().enumerate() {
                let t = an.instr(ins, &cur)?;
                let d = reads_writes(p, ins)?.1;
                if let (Some(d), Some(t)) = (d, t) {
                    let t = match fixed[d as usize] {
                        None => t,
                        // Typed register: same type, or `int` to `float` (converted).
                        Some(f) if f == t || (f == JTy::Float && t == JTy::Int) => f,
                        Some(_) => return None,
                    };
                    defs[bi][k] = Some(t);
                    cur[d as usize] = Lat::T(t);
                }
            }
            let g = |r: Reg| cur[r as usize].get();
            let num = |r: Reg| matches!(g(r), Some(JTy::Int | JTy::Float));
            let ok = match &b.term {
                Term::Jump(_) => true,
                Term::Branch { cond, .. } => {
                    matches!(g(*cond), Some(JTy::Int | JTy::Float | JTy::Bool | JTy::Nil))
                }
                Term::IfInt { a, b: y, .. } => g(*a) == Some(JTy::Int) && g(*y) == Some(JTy::Int),
                Term::IfIntK { a, .. } => g(*a) == Some(JTy::Int),
                Term::Return { reg, .. } => match ret {
                    RegTy::Float => num(*reg),
                    _ => g(*reg) == Some(ret_jty),
                },
                Term::IterNext { iter, dst, .. } => match g(*iter) {
                    Some(JTy::Iter(a)) => {
                        let t = elem_jty(&an.anns, a);
                        if fixed[*dst as usize].is_some_and(|f| f != t) {
                            return None;
                        }
                        term_defs[bi] = Some(t);
                        cur[*dst as usize] = Lat::T(t);
                        true
                    }
                    _ => false,
                },
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
    let n = p.regs.len();
    Some(Plan {
        entry: entry
            .into_iter()
            .map(|e| e.unwrap_or_else(|| vec![Lat::Undef; n]))
            .collect(),
        defs,
        term_defs,
        callees: an.callees,
        ctors: an.ctors,
        anns: an.anns,
    })
}

/// Registers read and written by an instruction accepted by the typed JIT; `None` for
/// the others.
fn reads_writes(p: &Proto, ins: &Instr) -> Option<(Vec<Reg>, Option<Reg>)> {
    use Instr::*;
    Some(match ins {
        Int(d, _) | Float(d, _) | Bool(d, _) | Nil(d) | GetGlobal(d, _) => (vec![], Some(*d)),
        Move(d, a)
        | Guard(d, a, _)
        | FieldAs(d, a, _, _)
        | INeg(d, a)
        | FNeg(d, a)
        | IToF(d, a)
        | Not(d, a)
        | IBinK(_, d, a, _) => (vec![*a], Some(*d)),
        IBin(_, d, a, b) | FBin(_, d, a, b) | Bin(_, d, a, b) | Index(d, a, b) => {
            (vec![*a, *b], Some(*d))
        }
        IterNew { dst, src, .. } => (vec![*src], Some(*dst)),
        List(d, start, n) => ((*start..*start + *n).collect(), Some(*d)),
        CallMethod {
            dst, recv, site, ..
        } => {
            let site = &p.calls[*site as usize];
            let n = site.npos + site.named.len() as u32;
            let mut r: Vec<Reg> = (site.start..site.start + n).collect();
            r.push(*recv);
            (r, Some(*dst))
        }
        Call { dst, f, site } => {
            let site = &p.calls[*site as usize];
            let n = site.npos + site.named.len() as u32;
            let mut r: Vec<Reg> = (site.start..site.start + n).collect();
            r.push(*f);
            (r, Some(*dst))
        }
        _ => return None,
    })
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

fn int_cc(op: BinOp) -> IntCC {
    match op {
        BinOp::Eq => IntCC::Equal,
        BinOp::Ne => IntCC::NotEqual,
        BinOp::Lt => IntCC::SignedLessThan,
        BinOp::Le => IntCC::SignedLessThanOrEqual,
        BinOp::Gt => IntCC::SignedGreaterThan,
        _ => IntCC::SignedGreaterThanOrEqual,
    }
}

fn float_cc(op: BinOp) -> FloatCC {
    match op {
        BinOp::Eq => FloatCC::Equal,
        BinOp::Ne => FloatCC::NotEqual,
        BinOp::Lt => FloatCC::LessThan,
        BinOp::Le => FloatCC::LessThanOrEqual,
        BinOp::Gt => FloatCC::GreaterThan,
        _ => FloatCC::GreaterThanOrEqual,
    }
}

impl Jit {
    /// `None` if the machine is not supported by Cranelift.
    pub fn new() -> Option<Self> {
        let mut flags = settings::builder();
        flags.set("use_colocated_libcalls", "false").ok()?;
        flags.set("is_pic", "false").ok()?;
        flags.set("opt_level", "speed").ok()?;
        let isa = cranelift_native::builder()
            .ok()?
            .finish(settings::Flags::new(flags))
            .ok()?;
        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
        builder.symbol("bk_trace", bk_trace as *const u8);
        builder.symbol("bk_frem", bk_frem as *const u8);
        builder.symbol("bk_field_native", bk_field_native as *const u8);
        builder.symbol("bk_obj_fields", bk_obj_fields as *const u8);
        builder.symbol("bk_list_view", bk_list_view as *const u8);
        builder.symbol("bk_build", bk_build as *const u8);
        builder.symbol("bk_clone_into", bk_clone_into as *const u8);
        builder.symbol("bk_move_out", bk_move_out as *const u8);
        builder.symbol("bk_drop_slots", bk_drop_slots as *const u8);
        builder.symbol("bk_step", bk_step as *const u8);
        builder.symbol("bk_iter_next", bk_iter_next as *const u8);
        builder.symbol("bk_drop", bk_drop as *const u8);
        for (name, f) in FAST {
            builder.symbol(*name, *f as *const u8);
        }
        let mut module = JITModule::new(builder);
        let ptr = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(types::I32));
        let trace_fn = module
            .declare_function("bk_trace", Linkage::Import, &sig)
            .ok()?;
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::F64));
        sig.params.push(AbiParam::new(types::F64));
        sig.returns.push(AbiParam::new(types::F64));
        let frem_fn = module
            .declare_function("bk_frem", Linkage::Import, &sig)
            .ok()?;
        // Objects are passed by pointer in `i64` variables.
        if ptr != types::I64 {
            return None;
        }
        let mut sig = module.make_signature();
        for _ in 0..3 {
            sig.params.push(AbiParam::new(ptr));
        }
        sig.returns.push(AbiParam::new(types::I32));
        let field_fn = module
            .declare_function("bk_field_native", Linkage::Import, &sig)
            .ok()?;
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.returns.push(AbiParam::new(ptr));
        let fields_fn = module
            .declare_function("bk_obj_fields", Linkage::Import, &sig)
            .ok()?;
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.returns.push(AbiParam::new(types::I32));
        let list_fn = module
            .declare_function("bk_list_view", Linkage::Import, &sig)
            .ok()?;
        let mut mem = Vec::new();
        for (name, nparams, i32_last) in [
            ("bk_build", 3, false),
            ("bk_clone_into", 2, false),
            ("bk_move_out", 3, true),
            ("bk_drop_slots", 2, false),
        ] {
            let mut sig = module.make_signature();
            for k in 0..nparams {
                let t = if i32_last && k == nparams - 1 {
                    types::I32
                } else {
                    ptr
                };
                sig.params.push(AbiParam::new(t));
            }
            mem.push(module.declare_function(name, Linkage::Import, &sig).ok()?);
        }
        let mem_fns: [FuncId; 4] = mem.try_into().ok()?;
        Some(Jit {
            module,
            ctx: Box::new(JitCtx {
                depth: 0,
                max_depth: 0,
                err: -1,
                traces: Vec::new(),
            }),
            errors: Vec::new(),
            traces: Vec::new(),
            compiled: FxHashMap::default(),
            trace_fn,
            frem_fn,
            field_fn,
            fields_fn,
            list_fn,
            mem_fns,
            build_plans: Vec::new(),
            known_types: Vec::new(),
            bailed: false,
            len_sym: crate::intern::Symbol(u32::MAX),
            dyn_fns: None,
            fast_fns: Vec::new(),
            field_caches: Vec::new(),
            next: 0,
        })
    }

    /// Compiles `root` (and the functions it calls) if possible. Updates the JIT state of
    /// every function encountered.
    pub fn compile(&mut self, root: &Rc<Proto>, globals: &mut Globals) {
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
                // Invalidated code (a called function was redefined): to recompile.
                self.compiled.remove(&key);
            }
            let rejected = matches!(*p.jit.borrow(), JitState::Rejected);
            if rejected {
                plans.insert(key, None);
                order.push(p);
                continue;
            }
            let plan = analyze(&p, globals, self.len_sym);
            if let Some(plan) = &plan {
                for slot in &plan.callees {
                    if let Some(q) = global_proto(globals, *slot) {
                        todo.push(q);
                    }
                }
            }
            plans.insert(key, plan);
            order.push(p);
        }
        // 2. A function that calls a non-compilable function is not compilable either.
        loop {
            let mut changed = false;
            for p in &order {
                let key = Rc::as_ptr(p) as usize;
                let Some(Some(plan)) = plans.get(&key) else {
                    continue;
                };
                let blocked = plan
                    .callees
                    .iter()
                    .any(|slot| match global_proto(globals, *slot) {
                        Some(q) => {
                            let k = Rc::as_ptr(&q) as usize;
                            !self.compiled.contains_key(&k)
                                && !matches!(plans.get(&k), Some(Some(_)))
                        }
                        None => true,
                    });
                if blocked {
                    plans.insert(key, None);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // 3. Declaration, then code generation.
        let mut new = Vec::new();
        for p in &order {
            let key = Rc::as_ptr(p) as usize;
            if let Some(Some(_)) = plans.get(&key) {
                let (params, ret) = native_sig(p).unwrap();
                let sig = self.signature(&params, ret);
                self.next += 1;
                let id = match self.module.declare_function(
                    &format!("bk{}", self.next),
                    Linkage::Local,
                    &sig,
                ) {
                    Ok(id) => id,
                    Err(_) => return,
                };
                self.compiled.insert(key, Compiled { id, params, ret });
                new.push(p.clone());
            }
        }
        let mut entries = Vec::new();
        for p in &new {
            let key = Rc::as_ptr(p) as usize;
            let plan = plans.remove(&key).unwrap().unwrap();
            if self.define(p, &plan, globals).is_err() {
                // Unexpected Cranelift failure: the interpreter is still available.
                self.compiled.remove(&key);
                *p.jit.borrow_mut() = JitState::Rejected;
                continue;
            }
            match self.define_entry(key) {
                Ok(e) => entries.push((p.clone(), e, [plan.callees, plan.ctors].concat())),
                Err(_) => *p.jit.borrow_mut() = JitState::Rejected,
            }
        }
        if self.module.finalize_definitions().is_err() {
            return;
        }
        for (p, entry, callees) in entries {
            let c = &self.compiled[&(Rc::as_ptr(&p) as usize)];
            let code = self.module.get_finalized_function(entry);
            *p.jit.borrow_mut() = JitState::Native(NativeCode {
                entry: code as usize,
                params: c.params.clone(),
                ret: c.ret,
            });
            for slot in callees {
                globals.watch(slot, &p);
            }
            // Type of a parameter redeclared: field layout to recompile.
            for prm in &p.params {
                if let Some(ann) = &prm.ty
                    && let RtType::Named(slot, _) = &*ann.ty
                {
                    globals.watch(*slot, &p);
                }
            }
        }
        // Rejected functions: retried if a function they call changes.
        for p in &order {
            let key = Rc::as_ptr(p) as usize;
            if let Some(None) = plans.get(&key) {
                *p.jit.borrow_mut() = JitState::Rejected;
                if let Some(plan) = analyze_callees(p) {
                    for slot in plan {
                        globals.watch(slot, p);
                    }
                }
            }
        }
    }

    fn signature(&self, params: &[RegTy], ret: RegTy) -> Signature {
        let mut sig = self.module.make_signature();
        sig.params
            .push(AbiParam::new(self.module.target_config().pointer_type()));
        for p in params {
            sig.params.push(AbiParam::new(clif(*p)));
        }
        // Object or list returned: written into a slot provided by the caller.
        if ret == RegTy::Value {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I32));
        sig.returns.push(AbiParam::new(clif(ret)));
        sig
    }

    fn error_id(&mut self, kind: ErrKind, span: Span, script: &Rc<Script>) -> i64 {
        self.errors.push(ErrInfo {
            kind,
            span,
            script: script.clone(),
        });
        (self.errors.len() - 1) as i64
    }

    fn trace_id(&mut self, name: Rc<str>, span: Span) -> i64 {
        self.traces.push(TraceInfo { name, span });
        (self.traces.len() - 1) as i64
    }

    /// Generates the code of a function.
    fn define(&mut self, p: &Rc<Proto>, plan: &Plan, globals: &Globals) -> Result<(), ()> {
        let key = Rc::as_ptr(p) as usize;
        let (id, ret) = {
            let c = &self.compiled[&key];
            (c.id, c.ret)
        };
        let mut ctx = self.module.make_context();
        ctx.func.signature = self.signature(&self.compiled[&key].params.clone(), ret);
        ctx.func.name = UserFuncName::user(0, id.as_u32());
        // Called functions, known in this function.
        let mut callee_refs = FxHashMap::default();
        for slot in &plan.callees {
            let q = global_proto(globals, *slot).ok_or(())?;
            let cid = self.compiled.get(&(Rc::as_ptr(&q) as usize)).ok_or(())?.id;
            let r = self.module.declare_func_in_func(cid, &mut ctx.func);
            callee_refs.insert(*slot, (r, q));
        }
        let trace_ref = self
            .module
            .declare_func_in_func(self.trace_fn, &mut ctx.func);
        let frem_ref = self
            .module
            .declare_func_in_func(self.frem_fn, &mut ctx.func);
        let field_ref = self
            .module
            .declare_func_in_func(self.field_fn, &mut ctx.func);
        let fields_ref = self
            .module
            .declare_func_in_func(self.fields_fn, &mut ctx.func);
        let list_ref = self
            .module
            .declare_func_in_func(self.list_fn, &mut ctx.func);
        // Object parameters whose fields are read: their type is known now (global slot of the
        // annotation), hence the offset of every field.
        let mut obj_types: Vec<Option<Rc<crate::objects::TypeInfo>>> = vec![None; plan.anns.len()];
        for (a, ann) in plan.anns.iter().enumerate() {
            if let RtType::Named(slot, _) = ann
                && let Some(Value::Type(info)) = &globals.at(*slot).value
                && info.kind == crate::ast::TypeDeclKind::Type
            {
                obj_types[a] = Some(info.clone());
                // Kept alive: its address identifies the type in native code.
                if !self.known_types.iter().any(|t| Rc::ptr_eq(t, info)) {
                    self.known_types.push(info.clone());
                }
            }
        }
        // Registers of objects, lists and iterators: "borrowed" if they always refer to a value
        // that lives for the whole call (parameter, element of a borrowed list); otherwise
        // "owning": a slot of the native frame, dropped on exit.
        let is_ref = |t: Option<JTy>| matches!(t, Some(JTy::Obj(_) | JTy::List(_) | JTy::Iter(_)));
        let nregs = p.regs.len();
        // (register, type) pairs: parameters, writes, types known at block entries.
        let mut pairs: Vec<(Reg, JTy)> = Vec::new();
        for m in &plan.entry {
            for (r, l) in m.iter().enumerate() {
                if let Some(t) = l.get() {
                    pairs.push((r as Reg, t));
                }
            }
        }
        for (b, defs) in p.blocks.iter().zip(&plan.defs) {
            for (ins, t) in b.instrs.iter().zip(defs) {
                if let (Some((_, Some(d))), Some(t)) = (reads_writes(p, ins), t) {
                    pairs.push((d, *t));
                }
            }
        }
        for (b, t) in p.blocks.iter().zip(&plan.term_defs) {
            if let (Term::IterNext { dst, .. }, Some(t)) = (&b.term, t) {
                pairs.push((*dst, *t));
            }
        }
        pairs.sort();
        pairs.dedup();
        let mut borrowed = vec![false; nregs];
        for (r, t) in &pairs {
            if is_ref(Some(*t)) {
                borrowed[*r as usize] = true;
            }
        }
        let reg_ref = borrowed.clone();
        let mut written = vec![false; nregs];
        loop {
            let mut changed = false;
            for b in &p.blocks {
                for ins in &b.instrs {
                    let (w, from_borrowed) = match ins {
                        Instr::Move(d, a) | Instr::Index(d, a, _) => {
                            (Some(*d), borrowed[*a as usize])
                        }
                        Instr::IterNew { dst, src, .. } => (Some(*dst), borrowed[*src as usize]),
                        other => (reads_writes(p, other).and_then(|(_, w)| w), false),
                    };
                    if let Some(w) = w {
                        written[w as usize] = true;
                        if borrowed[w as usize] && !from_borrowed {
                            borrowed[w as usize] = false;
                            changed = true;
                        }
                    }
                }
                if let Term::IterNext { iter, dst, .. } = &b.term {
                    written[*dst as usize] = true;
                    if borrowed[*dst as usize] && !borrowed[*iter as usize] {
                        borrowed[*dst as usize] = false;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut slot_of: FxHashMap<Reg, i32> = FxHashMap::default();
        for r in 0..nregs {
            if reg_ref[r] && !borrowed[r] {
                let off = slot_of.len() as i32 * crate::value::layout::SIZE;
                slot_of.insert(r as Reg, off);
            }
        }
        let nslots = slot_of.len();
        let [build_fn, clone_fn, move_fn, drop_fn] = self.mem_fns;
        let build_ref = self.module.declare_func_in_func(build_fn, &mut ctx.func);
        let clone_ref = self.module.declare_func_in_func(clone_fn, &mut ctx.func);
        let move_ref = self.module.declare_func_in_func(move_fn, &mut ctx.func);
        let drop_ref = self.module.declare_func_in_func(drop_fn, &mut ctx.func);
        // Errors and calls of this function, numbered after the existing tables.
        let (err_base, trace_base) = (self.errors.len() as i64, self.traces.len() as i64);
        let mut err_sites: Vec<(ErrKind, Span)> = Vec::new();
        let mut call_sites: Vec<(Rc<str>, Span)> = Vec::new();

        let tc = self.module.target_config();
        let mut fctx = FunctionBuilderContext::new();
        let mut bx = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        // One variable per register and per type it takes.
        let mut vars: FxHashMap<(Reg, JTy), Variable> = FxHashMap::default();
        for (r, t) in &pairs {
            let ty = match t {
                JTy::Float => types::F64,
                JTy::Bool => types::I8,
                JTy::Int | JTy::Obj(_) | JTy::List(_) => types::I64,
                _ => continue,
            };
            vars.insert((*r, *t), bx.declare_var(ty));
        }
        // Iterators: address of the elements, count, position.
        let iter_vars: FxHashMap<Reg, [Variable; 3]> = pairs
            .iter()
            .filter(|(_, t)| matches!(t, JTy::Iter(_)))
            .map(|(r, _)| {
                let v = [0; 3].map(|_| bx.declare_var(types::I64));
                (*r, v)
            })
            .collect();
        let entry = bx.create_block();
        let blocks: Vec<_> = p.blocks.iter().map(|_| bx.create_block()).collect();
        let fail = bx.create_block();
        bx.append_block_param(fail, types::I64);
        bx.append_block_params_for_function_params(entry);
        bx.switch_to_block(entry);
        let ctxv = bx.block_params(entry)[0];
        // Output slot of field reads.
        let out_slot = bx.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            16,
            3,
        ));
        // Slots of owned values, `nil` (zero bytes) on entry.
        let vals = bx.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            (nslots.max(1) as i32 * crate::value::layout::SIZE) as u32,
            3,
        ));
        let vals_base = bx.ins().stack_addr(types::I64, vals, 0);
        for k in 0..nslots as i32 * crate::value::layout::SIZE / 8 {
            let z = bx.ins().iconst(types::I64, 0);
            bx.ins().store(MemFlagsData::trusted(), z, vals_base, k * 8);
        }
        let out_param = (ret == RegTy::Value).then(|| bx.block_params(entry)[1 + p.params.len()]);
        // Every register is defined from the entry (zero value).
        for v in iter_vars.values().flatten() {
            let z = bx.ins().iconst(types::I64, 0);
            bx.def_var(*v, z);
        }
        for ((_, t), v) in &vars {
            let z = match t {
                JTy::Float => bx.ins().f64const(0.0),
                JTy::Bool => bx.ins().iconst(types::I8, 0),
                _ => bx.ins().iconst(types::I64, 0),
            };
            bx.def_var(*v, z);
        }
        for (k, prm) in p.params.iter().enumerate() {
            let a = bx.block_params(entry)[k + 1];
            let t = plan.entry[0][prm.reg as usize].get().ok_or(())?;
            bx.def_var(vars[&(prm.reg, t)], a);
        }
        // Field address of each object of known type (checked once here).
        // Object parameter never reassigned whose fields are read: type checked and field
        // address taken here, once.
        let reads_fields = |reg: Reg| {
            p.blocks
                .iter()
                .flat_map(|b| &b.instrs)
                .any(|i| matches!(i, Instr::FieldAs(_, r, ..) if *r == reg))
        };
        let mut fields_ptr: FxHashMap<Reg, cranelift_codegen::ir::Value> = FxHashMap::default();
        let mut entry_checks = Vec::new();
        for (k, prm) in p.params.iter().enumerate() {
            let Some(JTy::Obj(a)) = plan.entry[0][prm.reg as usize].get() else {
                continue;
            };
            let Some(info) = &obj_types[a as usize] else {
                continue;
            };
            if written[prm.reg as usize] || !reads_fields(prm.reg) {
                continue;
            }
            let obj = bx.block_params(entry)[k + 1];
            let ip = bx.ins().iconst(types::I64, Rc::as_ptr(info) as i64);
            let call = bx.ins().call(fields_ref, &[obj, ip]);
            let fp = bx.inst_results(call)[0];
            entry_checks.push(fp);
            fields_ptr.insert(prm.reg, fp);
        }
        let start = bx.create_block();
        for fp in entry_checks {
            let null = bx.ins().icmp_imm_s(IntCC::Equal, fp, 0);
            err_sites.push((ErrKind::Bail, Span::default()));
            let eid = err_base + (err_sites.len() - 1) as i64;
            let cont = bx.create_block();
            let id = bx.ins().iconst(types::I64, eid);
            bx.ins().brif(null, fail, &[id.into()], cont, &[]);
            bx.switch_to_block(cont);
        }
        bx.ins().jump(start, &[]);
        bx.switch_to_block(start);
        bx.ins().jump(blocks[0], &[]);

        // Current types (before the instruction) and type written by the instruction.
        let mut cur: Vec<Option<JTy>>;
        let mut dty: Option<JTy>;
        macro_rules! tyof {
            ($r:expr) => {
                cur[$r as usize]
            };
        }
        macro_rules! uvar {
            ($r:expr) => {
                *vars.get(&($r, cur[$r as usize].ok_or(())?)).ok_or(())?
            };
        }
        macro_rules! dvar {
            ($r:expr) => {
                *vars.get(&($r, dty.ok_or(())?)).ok_or(())?
            };
        }
        // Branches to error `id` if `cond`.
        macro_rules! check {
            ($bx:expr, $cond:expr, $kind:expr, $span:expr) => {{
                err_sites.push(($kind, $span));
                let eid = err_base + (err_sites.len() - 1) as i64;
                let cont = $bx.create_block();
                let id = $bx.ins().iconst(types::I64, eid);
                $bx.ins().brif($cond, fail, &[id.into()], cont, &[]);
                $bx.switch_to_block(cont);
            }};
        }
        // Elements of a list: (address, count); bail if it is not a list.
        macro_rules! view {
            ($bx:expr, $list:expr, $span:expr) => {{
                let out = $bx.ins().stack_addr(types::I64, out_slot, 0);
                let call = $bx.ins().call(list_ref, &[$list, out]);
                let status = $bx.inst_results(call)[0];
                check!($bx, status, ErrKind::Bail, $span);
                let data = $bx.ins().load(types::I64, MemFlagsData::trusted(), out, 0);
                let len = $bx.ins().load(types::I64, MemFlagsData::trusted(), out, 8);
                (data, len)
            }};
        }
        // Address of the slot of owning register `r`.
        macro_rules! own {
            ($bx:expr, $r:expr) => {{
                let off = *slot_of.get(&$r).ok_or(())?;
                $bx.ins().stack_addr(types::I64, vals, off)
            }};
        }
        // Value of element `v` (object: address) into register `d`: copied into its slot if
        // `d` is owning.
        macro_rules! put_ref {
            ($bx:expr, $d:expr, $v:expr) => {{
                if borrowed[$d as usize] {
                    $bx.def_var(dvar!($d), $v);
                } else {
                    let slot = own!($bx, $d);
                    $bx.ins().call(clone_ref, &[slot, $v]);
                    $bx.def_var(dvar!($d), slot);
                }
            }};
        }
        // Drops the slots (before every exit).
        macro_rules! drop_slots {
            ($bx:expr) => {{
                if nslots > 0 {
                    let base = $bx.ins().stack_addr(types::I64, vals, 0);
                    let n = $bx.ins().iconst(types::I64, nslots as i64);
                    $bx.ins().call(drop_ref, &[base, n]);
                }
            }};
        }
        // Builds into the slot of `d` the object or list described by `items` (native values
        // of registers `regs`).
        macro_rules! build {
            ($bx:expr, $d:expr, $info:expr, $items:expr, $regs:expr) => {{
                let items: Vec<(usize, Kind)> = $items;
                let regs: Vec<Reg> = $regs;
                let args = $bx.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    (regs.len().max(1) * 8) as u32,
                    3,
                ));
                let base = $bx.ins().stack_addr(types::I64, args, 0);
                for (k, r) in regs.iter().enumerate() {
                    let v = $bx.use_var(uvar!(*r));
                    let bits = match tyof!(*r) {
                        Some(JTy::Float) => $bx.ins().bitcast(types::I64, MemFlagsData::new(), v),
                        Some(JTy::Bool) => $bx.ins().uextend(types::I64, v),
                        _ => v,
                    };
                    $bx.ins()
                        .store(MemFlagsData::trusted(), bits, base, (k * 8) as i32);
                }
                self.build_plans
                    .push(Box::new(BuildPlan { info: $info, items }));
                let plan = &**self.build_plans.last().unwrap() as *const BuildPlan;
                let plan = $bx.ins().iconst(types::I64, plan as i64);
                let slot = own!($bx, $d);
                $bx.ins().call(build_ref, &[slot, base, plan]);
                $bx.def_var(dvar!($d), slot);
            }};
        }
        // Value of the element at address `addr`, of type `anns[a]`: the address for an
        // object; otherwise the tag is checked (bail if it differs), `int` to `float`.
        macro_rules! elem {
            ($bx:expr, $addr:expr, $a:expr, $span:expr) => {{
                use crate::value::layout::{PAYLOAD, TAG_BOOL, TAG_FLOAT, TAG_INT};
                let t = elem_jty(&plan.anns, $a);
                if let JTy::Obj(_) = t {
                    $addr
                } else {
                    let tag = $bx.ins().load(types::I8, MemFlagsData::trusted(), $addr, 0);
                    let raw = $bx
                        .ins()
                        .load(types::I64, MemFlagsData::trusted(), $addr, PAYLOAD);
                    match t {
                        JTy::Int => {
                            let bad = $bx.ins().icmp_imm_s(IntCC::NotEqual, tag, TAG_INT as i64);
                            check!($bx, bad, ErrKind::Bail, $span);
                            raw
                        }
                        JTy::Bool => {
                            let bad = $bx.ins().icmp_imm_s(IntCC::NotEqual, tag, TAG_BOOL as i64);
                            check!($bx, bad, ErrKind::Bail, $span);
                            $bx.ins().ireduce(types::I8, raw)
                        }
                        _ => {
                            let isf = $bx.ins().icmp_imm_s(IntCC::Equal, tag, TAG_FLOAT as i64);
                            let isi = $bx.ins().icmp_imm_s(IntCC::Equal, tag, TAG_INT as i64);
                            let ok = $bx.ins().bor(isf, isi);
                            let bad = $bx.ins().icmp_imm_s(IntCC::Equal, ok, 0);
                            check!($bx, bad, ErrKind::Bail, $span);
                            let f = $bx.ins().bitcast(types::F64, MemFlagsData::new(), raw);
                            let i = $bx.ins().fcvt_from_sint(types::F64, raw);
                            $bx.ins().select(isf, f, i)
                        }
                    }
                }
            }};
        }
        for (bi, b) in p.blocks.iter().enumerate() {
            bx.switch_to_block(blocks[bi]);
            cur = plan.entry[bi].iter().map(|l| l.get()).collect();
            for (k, (ins, &span)) in b.instrs.iter().zip(&b.spans).enumerate() {
                use Instr::*;
                dty = plan.defs[bi][k];
                match ins {
                    Int(d, n) => {
                        let v = bx.ins().iconst(types::I64, *n);
                        bx.def_var(dvar!(*d), v);
                    }
                    Float(d, f) => {
                        let v = bx.ins().f64const(*f);
                        bx.def_var(dvar!(*d), v);
                    }
                    Bool(d, v) => {
                        let v = bx.ins().iconst(types::I8, *v as i64);
                        bx.def_var(dvar!(*d), v);
                    }
                    Nil(_) | GetGlobal(..) => {}
                    Move(_, a) if is_ref(dty) => {
                        let d = match ins {
                            Move(d, _) => d,
                            _ => unreachable!(),
                        };
                        let v = bx.use_var(uvar!(*a));
                        put_ref!(bx, *d, v);
                    }
                    Move(d, a) | Guard(d, a, _) => {
                        let v = bx.use_var(uvar!(*a));
                        let v = match (dty, tyof!(*a)) {
                            (Some(JTy::Float), Some(JTy::Int)) => {
                                bx.ins().fcvt_from_sint(types::F64, v)
                            }
                            _ => v,
                        };
                        bx.def_var(dvar!(*d), v);
                    }
                    IBin(op, d, a, b2) => {
                        let (x, y) = (bx.use_var(uvar!(*a)), bx.use_var(uvar!(*b2)));
                        let v = int_bin(&mut bx, *op, x, y, &mut |bx, c, k| check!(bx, c, k, span));
                        bx.def_var(dvar!(*d), v);
                    }
                    // Field of an object of known type: direct read and tag check.
                    FieldAs(d, a, name, t)
                        if let Some(JTy::Obj(k)) = tyof!(*a)
                            && let Some(info) = &obj_types[k as usize]
                            && let Some(idx) = info.field_index(*name) =>
                    {
                        // Parameter: field address taken on entry; list element: checked on every
                        // read.
                        let fp = match fields_ptr.get(a).copied() {
                            Some(fp) => fp,
                            None => {
                                let obj = bx.use_var(uvar!(*a));
                                let ip = bx.ins().iconst(types::I64, Rc::as_ptr(info) as i64);
                                let call = bx.ins().call(fields_ref, &[obj, ip]);
                                let fp = bx.inst_results(call)[0];
                                let null = bx.ins().icmp_imm_s(IntCC::Equal, fp, 0);
                                check!(bx, null, ErrKind::Bail, span);
                                fp
                            }
                        };
                        let off = idx as i32 * crate::value::layout::SIZE;
                        let tag = bx.ins().load(types::I8, MemFlagsData::trusted(), fp, off);
                        let (want, cty) = match t {
                            RegTy::Int => (crate::value::layout::TAG_INT, types::I64),
                            RegTy::Float => (crate::value::layout::TAG_FLOAT, types::F64),
                            _ => (crate::value::layout::TAG_BOOL, types::I8),
                        };
                        let bad = bx.ins().icmp_imm_s(IntCC::NotEqual, tag, want as i64);
                        check!(bx, bad, ErrKind::Bail, span);
                        let v = bx.ins().load(
                            cty,
                            MemFlagsData::trusted(),
                            fp,
                            off + crate::value::layout::PAYLOAD,
                        );
                        bx.def_var(dvar!(*d), v);
                    }
                    FieldAs(d, a, name, t) => {
                        self.field_caches.push(Box::new(FieldCache {
                            info: None,
                            idx: 0,
                            name: *name,
                            want: *t,
                        }));
                        let cache = &**self.field_caches.last().unwrap() as *const FieldCache;
                        let obj = bx.use_var(uvar!(*a));
                        let cache = bx.ins().iconst(types::I64, cache as i64);
                        let out = bx.ins().stack_addr(types::I64, out_slot, 0);
                        let call = bx.ins().call(field_ref, &[obj, cache, out]);
                        let status = bx.inst_results(call)[0];
                        check!(bx, status, ErrKind::Bail, span);
                        let bits = bx.ins().load(types::I64, MemFlagsData::trusted(), out, 0);
                        let v = match t {
                            RegTy::Float => bx.ins().bitcast(types::F64, MemFlagsData::new(), bits),
                            RegTy::Bool => bx.ins().ireduce(types::I8, bits),
                            _ => bits,
                        };
                        bx.def_var(dvar!(*d), v);
                    }
                    IBinK(op, d, a, k) => {
                        let x = bx.use_var(uvar!(*a));
                        let y = bx.ins().iconst(types::I64, *k);
                        let v = int_bin(&mut bx, *op, x, y, &mut |bx, c, k| check!(bx, c, k, span));
                        bx.def_var(dvar!(*d), v);
                    }
                    FBin(op, d, a, b2) => {
                        let (x, y) = (bx.use_var(uvar!(*a)), bx.use_var(uvar!(*b2)));
                        let v = float_bin(&mut bx, *op, x, y, frem_ref);
                        bx.def_var(dvar!(*d), v);
                    }
                    Bin(op, d, a, b2) => {
                        let (ta, tb) = (tyof!(*a), tyof!(*b2));
                        let (x, y) = (bx.use_var(uvar!(*a)), bx.use_var(uvar!(*b2)));
                        let v = match (ta, tb) {
                            (Some(JTy::Int), Some(JTy::Int)) => {
                                int_bin(&mut bx, *op, x, y, &mut |bx, c, k| check!(bx, c, k, span))
                            }
                            (Some(JTy::Bool), Some(JTy::Bool)) => bx.ins().icmp(int_cc(*op), x, y),
                            _ => {
                                let x = if ta == Some(JTy::Int) {
                                    bx.ins().fcvt_from_sint(types::F64, x)
                                } else {
                                    x
                                };
                                let y = if tb == Some(JTy::Int) {
                                    bx.ins().fcvt_from_sint(types::F64, y)
                                } else {
                                    y
                                };
                                float_bin(&mut bx, *op, x, y, frem_ref)
                            }
                        };
                        bx.def_var(dvar!(*d), v);
                    }
                    INeg(d, a) => {
                        let x = bx.use_var(uvar!(*a));
                        let c = bx.ins().icmp_imm_s(IntCC::Equal, x, i64::MIN);
                        check!(bx, c, ErrKind::Overflow, span);
                        let v = bx.ins().ineg(x);
                        bx.def_var(dvar!(*d), v);
                    }
                    FNeg(d, a) => {
                        let x = bx.use_var(uvar!(*a));
                        let v = bx.ins().fneg(x);
                        bx.def_var(dvar!(*d), v);
                    }
                    IToF(d, a) => {
                        let x = bx.use_var(uvar!(*a));
                        let v = bx.ins().fcvt_from_sint(types::F64, x);
                        bx.def_var(dvar!(*d), v);
                    }
                    Not(d, a) => {
                        let v = match tyof!(*a) {
                            Some(JTy::Bool) => {
                                let x = bx.use_var(uvar!(*a));
                                bx.ins().icmp_imm_s(IntCC::Equal, x, 0)
                            }
                            // A number is always truthy.
                            _ => bx.ins().iconst(types::I8, 0),
                        };
                        bx.def_var(dvar!(*d), v);
                    }
                    Call { dst, f, site } if let Some(JTy::Ctor(slot)) = tyof!(*f) => {
                        let site = &p.calls[*site as usize];
                        let info = global_type(globals, slot).ok_or(())?;
                        let items = ctor_plan(&info, site, &cur, &plan.anns).ok_or(())?;
                        let n = items.len() as u32;
                        build!(
                            bx,
                            *dst,
                            Some(info),
                            items,
                            (site.start..site.start + n).collect()
                        );
                    }
                    List(d, start, n) => {
                        let items = (0..*n)
                            .map(|k| {
                                let t = tyof!(*start + k).ok_or(())?;
                                Ok((k as usize, kind_for(&plan.anns, t, None).ok_or(())?))
                            })
                            .collect::<Result<Vec<_>, ()>>()?;
                        build!(bx, *d, None, items, (*start..*start + *n).collect());
                    }
                    Call { dst, f, site } => {
                        let Some(JTy::Callee(slot)) = tyof!(*f) else {
                            return Err(());
                        };
                        let (fref, q) = callee_refs[&slot].clone();
                        let (cparams, cret) = native_sig(&q).ok_or(())?;
                        let site = &p.calls[*site as usize];
                        let mut args = vec![ctxv];
                        for (k, pt) in cparams.iter().enumerate() {
                            let r = site.start + k as u32;
                            let v = bx.use_var(uvar!(r));
                            let v = match (pt, tyof!(r)) {
                                (RegTy::Float, Some(JTy::Int)) => {
                                    bx.ins().fcvt_from_sint(types::F64, v)
                                }
                                _ => v,
                            };
                            args.push(v);
                        }
                        // Object or list returned: in the slot of `dst`.
                        let out = if cret == RegTy::Value {
                            let slot = own!(bx, *dst);
                            args.push(slot);
                            Some(slot)
                        } else {
                            None
                        };
                        // Call depth.
                        let depth =
                            bx.ins()
                                .load(types::I64, MemFlagsData::trusted(), ctxv, DEPTH_OFFSET);
                        let max = bx.ins().load(
                            types::I64,
                            MemFlagsData::trusted(),
                            ctxv,
                            MAX_DEPTH_OFFSET,
                        );
                        let too_deep = bx.ins().icmp(IntCC::SignedGreaterThanOrEqual, depth, max);
                        check!(bx, too_deep, ErrKind::Depth, span);
                        let d1 = bx.ins().iadd_imm_s(depth, 1);
                        bx.ins()
                            .store(MemFlagsData::trusted(), d1, ctxv, DEPTH_OFFSET);
                        let call = bx.ins().call(fref, &args);
                        let (status, result) = {
                            let r = bx.inst_results(call);
                            (r[0], r[1])
                        };
                        bx.ins()
                            .store(MemFlagsData::trusted(), depth, ctxv, DEPTH_OFFSET);
                        // Error in the called function: add this call to the stack.
                        call_sites.push((q.name.clone(), span));
                        let tid = trace_base + (call_sites.len() - 1) as i64;
                        let failed = bx.create_block();
                        let cont = bx.create_block();
                        bx.ins().brif(status, failed, &[], cont, &[]);
                        bx.switch_to_block(failed);
                        let t = bx.ins().iconst(types::I32, tid);
                        bx.ins().call(trace_ref, &[ctxv, t]);
                        let none = bx.ins().iconst(types::I64, -1);
                        bx.ins().jump(fail, &[none.into()]);
                        bx.switch_to_block(cont);
                        bx.def_var(dvar!(*dst), out.unwrap_or(result));
                    }
                    IterNew { dst, src, .. } => {
                        let mut list = bx.use_var(uvar!(*src));
                        // Owned list: the iterator keeps a copy of it (it stays alive even if its register
                        // changes during the loop).
                        if !borrowed[*dst as usize] {
                            let slot = own!(bx, *dst);
                            bx.ins().call(clone_ref, &[slot, list]);
                            list = slot;
                        }
                        let (data, len) = view!(bx, list, span);
                        let [dv, lv, iv] = iter_vars[dst];
                        let zero = bx.ins().iconst(types::I64, 0);
                        bx.def_var(dv, data);
                        bx.def_var(lv, len);
                        bx.def_var(iv, zero);
                    }
                    Index(d, a, i) => {
                        let Some(JTy::List(e)) = tyof!(*a) else {
                            return Err(());
                        };
                        let list = bx.use_var(uvar!(*a));
                        let (data, len) = view!(bx, list, span);
                        // Negative index: from the end; out of bounds: the interpreter produces the
                        // error.
                        let i = bx.use_var(uvar!(*i));
                        let neg = bx.ins().icmp_imm_s(IntCC::SignedLessThan, i, 0);
                        let from_end = bx.ins().iadd(i, len);
                        let pos = bx.ins().select(neg, from_end, i);
                        let out = bx.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, pos, len);
                        check!(bx, out, ErrKind::Bail, span);
                        let off = bx.ins().imul_imm_s(pos, crate::value::layout::SIZE as i64);
                        let addr = bx.ins().iadd(data, off);
                        let v = elem!(bx, addr, e, span);
                        if is_ref(dty) {
                            put_ref!(bx, *d, v);
                        } else {
                            bx.def_var(dvar!(*d), v);
                        }
                    }
                    CallMethod { dst, recv, .. } => {
                        let list = bx.use_var(uvar!(*recv));
                        let (_, len) = view!(bx, list, span);
                        bx.def_var(dvar!(*dst), len);
                    }
                    _ => return Err(()),
                }
                if let (Some((_, Some(d))), Some(t)) = (reads_writes(p, ins), dty) {
                    cur[d as usize] = Some(t);
                }
            }
            dty = plan.term_defs[bi];
            match &b.term {
                Term::Jump(t) => {
                    bx.ins().jump(blocks[*t as usize], &[]);
                }
                Term::Branch { cond, then, else_ } => {
                    let (t, e) = (blocks[*then as usize], blocks[*else_ as usize]);
                    match tyof!(*cond) {
                        Some(JTy::Bool) => {
                            let c = bx.use_var(uvar!(*cond));
                            bx.ins().brif(c, t, &[], e, &[]);
                        }
                        Some(JTy::Nil) => {
                            bx.ins().jump(e, &[]);
                        }
                        _ => {
                            bx.ins().jump(t, &[]);
                        }
                    }
                }
                Term::IfInt {
                    op,
                    a,
                    b: y,
                    then,
                    else_,
                } => {
                    let (x, y) = (bx.use_var(uvar!(*a)), bx.use_var(uvar!(*y)));
                    let c = bx.ins().icmp(int_cc(*op), x, y);
                    bx.ins()
                        .brif(c, blocks[*then as usize], &[], blocks[*else_ as usize], &[]);
                }
                Term::IfIntK {
                    op,
                    a,
                    k,
                    then,
                    else_,
                } => {
                    let x = bx.use_var(uvar!(*a));
                    let c = bx.ins().icmp_imm_s(int_cc(*op), x, *k);
                    bx.ins()
                        .brif(c, blocks[*then as usize], &[], blocks[*else_ as usize], &[]);
                }
                Term::Return { reg, .. } => {
                    let mut v = bx.use_var(uvar!(*reg));
                    v = match (ret, tyof!(*reg)) {
                        (RegTy::Float, Some(JTy::Int)) => bx.ins().fcvt_from_sint(types::F64, v),
                        _ => v,
                    };
                    if let Some(out) = out_param {
                        // Moved if the value is in the register's slot (owned), copied otherwise
                        // (parameter…).
                        let owned = match slot_of.get(reg) {
                            Some(&off) => {
                                let slot = bx.ins().stack_addr(types::I64, vals, off);
                                let same = bx.ins().icmp(IntCC::Equal, v, slot);
                                bx.ins().uextend(types::I32, same)
                            }
                            None => bx.ins().iconst(types::I32, 0),
                        };
                        bx.ins().call(move_ref, &[out, v, owned]);
                        v = bx.ins().iconst(types::I64, 0);
                    }
                    drop_slots!(bx);
                    let ok = bx.ins().iconst(types::I32, 0);
                    bx.ins().return_(&[ok, v]);
                }
                Term::IterNext {
                    iter,
                    dst,
                    body,
                    done,
                } => {
                    let Some(JTy::Iter(a)) = tyof!(*iter) else {
                        return Err(());
                    };
                    let [dv, lv, iv] = iter_vars[iter];
                    let (i, len) = (bx.use_var(iv), bx.use_var(lv));
                    let more = bx.ins().icmp(IntCC::SignedLessThan, i, len);
                    let next = bx.create_block();
                    bx.ins().brif(more, next, &[], blocks[*done as usize], &[]);
                    bx.switch_to_block(next);
                    let data = bx.use_var(dv);
                    let off = bx.ins().imul_imm_s(i, crate::value::layout::SIZE as i64);
                    let addr = bx.ins().iadd(data, off);
                    let i1 = bx.ins().iadd_imm_s(i, 1);
                    bx.def_var(iv, i1);
                    let span = b.term_span;
                    let v = elem!(bx, addr, a, span);
                    if is_ref(dty) {
                        put_ref!(bx, *dst, v);
                    } else {
                        bx.def_var(dvar!(*dst), v);
                    }
                    bx.ins().jump(blocks[*body as usize], &[]);
                }
            }
        }
        // Error exit: the error index is stored in `ctx` (-1: already stored by the called
        // function).
        bx.switch_to_block(fail);
        let eid = bx.block_params(fail)[0];
        let set = bx.create_block();
        let done = bx.create_block();
        let fresh = bx.ins().icmp_imm_s(IntCC::SignedGreaterThanOrEqual, eid, 0);
        bx.ins().brif(fresh, set, &[], done, &[]);
        bx.switch_to_block(set);
        bx.ins()
            .store(MemFlagsData::trusted(), eid, ctxv, ERR_OFFSET);
        bx.ins().jump(done, &[]);
        bx.switch_to_block(done);
        drop_slots!(bx);
        let one = bx.ins().iconst(types::I32, 1);
        let zero = match ret {
            RegTy::Float => bx.ins().f64const(0.0),
            RegTy::Bool => bx.ins().iconst(types::I8, 0),
            _ => bx.ins().iconst(types::I64, 0),
        };
        bx.ins().return_(&[one, zero]);
        bx.seal_all_blocks();
        bx.finalize(tc);
        let script = p.script.clone();
        for (kind, span) in err_sites {
            self.error_id(kind, span, &script);
        }
        for (name, span) in call_sites {
            self.trace_id(name, span);
        }
        self.module.define_function(id, &mut ctx).map_err(|_| ())?;
        self.module.clear_context(&mut ctx);
        Ok(())
    }

    /// Entry point for Rust: `fn(ctx, args, ret) -> status`.
    fn define_entry(&mut self, key: usize) -> Result<FuncId, ()> {
        let (id, params, ret) = {
            let c = &self.compiled[&key];
            (c.id, c.params.clone(), c.ret)
        };
        let ptr = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.returns.push(AbiParam::new(types::I32));
        self.next += 1;
        let eid = self
            .module
            .declare_function(&format!("bk_entry{}", self.next), Linkage::Local, &sig)
            .map_err(|_| ())?;
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, eid.as_u32());
        let target = self.module.declare_func_in_func(id, &mut ctx.func);
        let tc = self.module.target_config();
        let mut fctx = FunctionBuilderContext::new();
        let mut bx = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let b = bx.create_block();
        bx.append_block_params_for_function_params(b);
        bx.switch_to_block(b);
        let (c, args, out) = {
            let p = bx.block_params(b);
            (p[0], p[1], p[2])
        };
        let mut call_args = vec![c];
        for (k, t) in params.iter().enumerate() {
            let v = bx
                .ins()
                .load(clif(*t), MemFlagsData::trusted(), args, (k * 8) as i32);
            call_args.push(v);
        }
        // Object or list: `out` points to the value that receives the result.
        if ret == RegTy::Value {
            call_args.push(out);
        }
        let call = bx.ins().call(target, &call_args);
        let (status, result) = {
            let r = bx.inst_results(call);
            (r[0], r[1])
        };
        if ret != RegTy::Value {
            let result = if ret == RegTy::Bool {
                bx.ins().uextend(types::I64, result)
            } else {
                result
            };
            bx.ins().store(MemFlagsData::trusted(), result, out, 0);
        }
        bx.ins().return_(&[status]);
        bx.seal_all_blocks();
        bx.finalize(tc);
        self.module.define_function(eid, &mut ctx).map_err(|_| ())?;
        self.module.clear_context(&mut ctx);
        Ok(eid)
    }

    /// Runs the native code. `depth`: current call depth.
    ///
    /// `None`: the native stack is too small for the requested depth; the call must be
    /// redone by the interpreter (whose stack grows on the heap). This is safe: compiled
    /// functions have no side effects.
    pub fn call(
        &mut self,
        code: &NativeCode,
        args: &[Value],
        depth: usize,
        max_depth: usize,
    ) -> Option<Result<Value, Error>> {
        let mut raw = [0u64; 16];
        let mut heap = Vec::new();
        let buf: &mut [u64] = if args.len() <= raw.len() {
            &mut raw[..args.len()]
        } else {
            heap.resize(args.len(), 0);
            &mut heap
        };
        for (slot, v) in buf.iter_mut().zip(args) {
            *slot = match v {
                Value::Int(i) => *i as u64,
                Value::Float(f) => f.to_bits(),
                Value::Bool(b) => *b as u64,
                // Object: pointer to the argument, alive for the whole call.
                v => v as *const Value as u64,
            };
        }
        // Depth allowed by the remaining native stack (conservative estimate per call).
        const FRAME: usize = 512;
        const MARGIN: usize = 64 * 1024;
        let stack_cap = stacker::remaining_stack().map_or(usize::MAX, |r| {
            depth.saturating_add(r.saturating_sub(MARGIN) / FRAME)
        });
        let limit = max_depth.min(stack_cap);
        self.ctx.depth = depth as i64;
        self.ctx.max_depth = limit as i64;
        self.ctx.err = -1;
        self.ctx.traces.clear();
        self.bailed = false;
        let mut out = 0u64;
        // Object or list returned: written directly into `result`.
        let mut result = Value::Nil;
        let out_ptr = match code.ret {
            RegTy::Value => &mut result as *mut Value as *mut u64,
            _ => &mut out as *mut u64,
        };
        // SAFETY: `entry` is an entry point generated by `define_entry` with this signature;
        // the arguments have the expected number and types.
        let f: extern "C" fn(*mut JitCtx, *const u64, *mut u64) -> i32 =
            unsafe { std::mem::transmute(code.entry) };
        let status = f(&mut *self.ctx, buf.as_ptr(), out_ptr);
        if status == 0 {
            return Some(Ok(match code.ret {
                RegTy::Int => Value::Int(out as i64),
                RegTy::Float => Value::Float(f64::from_bits(out)),
                RegTy::Bool => Value::Bool(out & 1 == 1),
                RegTy::Value => result,
            }));
        }
        let info = &self.errors[self.ctx.err as usize];
        if matches!(info.kind, ErrKind::Bail) {
            self.bailed = true;
            return None;
        }
        if matches!(info.kind, ErrKind::Depth) && limit < max_depth {
            return None;
        }
        let msg = match info.kind {
            ErrKind::Overflow => "integer overflow".to_string(),
            ErrKind::DivZero => "division by zero".to_string(),
            ErrKind::Depth => {
                format!("too many nested calls ({max_depth}): infinite recursion?")
            }
            ErrKind::Bail => unreachable!(),
        };
        let mut e = Error::runtime(msg, info.span);
        e.src = Some((info.script.name.clone(), info.script.src.clone()));
        for &t in self.ctx.traces.iter().take(16) {
            let t = &self.traces[t as usize];
            e.trace.push((t.name.to_string(), t.span));
        }
        Some(Err(e))
    }
}

/// Direct calls of a function (global slots), to watch for their redefinition.
fn analyze_callees(p: &Proto) -> Option<Vec<u32>> {
    let mut out = Vec::new();
    for b in &p.blocks {
        for ins in &b.instrs {
            if let Instr::GetGlobal(_, slot) = ins
                && !out.contains(slot)
            {
                out.push(*slot);
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Integer arithmetic with the interpreter's checks. `check(bx, cond, kind)` branches to
/// the error if `cond`.
fn int_bin(
    bx: &mut FunctionBuilder,
    op: BinOp,
    x: cranelift_codegen::ir::Value,
    y: cranelift_codegen::ir::Value,
    check: &mut dyn FnMut(&mut FunctionBuilder, cranelift_codegen::ir::Value, ErrKind),
) -> cranelift_codegen::ir::Value {
    match op {
        BinOp::Add => {
            let (v, of) = bx.ins().sadd_overflow(x, y);
            check(bx, of, ErrKind::Overflow);
            v
        }
        BinOp::Sub => {
            let (v, of) = bx.ins().ssub_overflow(x, y);
            check(bx, of, ErrKind::Overflow);
            v
        }
        BinOp::Mul => {
            let (v, of) = bx.ins().smul_overflow(x, y);
            check(bx, of, ErrKind::Overflow);
            v
        }
        BinOp::Div | BinOp::Rem => {
            let zero = bx.ins().icmp_imm_s(IntCC::Equal, y, 0);
            check(bx, zero, ErrKind::DivZero);
            // i64::MIN / -1 overflows (and would crash `sdiv`).
            let min = bx.ins().icmp_imm_s(IntCC::Equal, x, i64::MIN);
            let m1 = bx.ins().icmp_imm_s(IntCC::Equal, y, -1);
            let both = bx.ins().band(min, m1);
            check(bx, both, ErrKind::Overflow);
            let q = bx.ins().sdiv(x, y);
            let r = bx.ins().srem(x, y);
            let neg = bx.ins().icmp_imm_s(IntCC::SignedLessThan, r, 0);
            if op == BinOp::Div {
                // Euclidean division: the remainder is always non-negative.
                let ypos = bx.ins().icmp_imm_s(IntCC::SignedGreaterThan, y, 0);
                let minus = bx.ins().iadd_imm_s(q, -1);
                let plus = bx.ins().iadd_imm_s(q, 1);
                let adj = bx.ins().select(ypos, minus, plus);
                bx.ins().select(neg, adj, q)
            } else {
                let ay = bx.ins().iabs(y);
                let r2 = bx.ins().iadd(r, ay);
                bx.ins().select(neg, r2, r)
            }
        }
        op => bx.ins().icmp(int_cc(op), x, y),
    }
}

fn float_bin(
    bx: &mut FunctionBuilder,
    op: BinOp,
    x: cranelift_codegen::ir::Value,
    y: cranelift_codegen::ir::Value,
    frem: cranelift_codegen::ir::FuncRef,
) -> cranelift_codegen::ir::Value {
    match op {
        BinOp::Add => bx.ins().fadd(x, y),
        BinOp::Sub => bx.ins().fsub(x, y),
        BinOp::Mul => bx.ins().fmul(x, y),
        BinOp::Div => bx.ins().fdiv(x, y),
        BinOp::Rem => {
            let c = bx.ins().call(frem, &[x, y]);
            bx.inst_results(c)[0]
        }
        op => bx.ins().fcmp(float_cc(op), x, y),
    }
}

// ─── Baseline compilation (untyped code) ───────────────────────────────────
//
// Any function can be compiled this way. Registers stay `Value`s in memory (the call's
// window); native code handles the simple cases itself (constants, copies of scalar
// values, integer and float arithmetic and comparisons, branches) and calls
// `Runtime::step` — the interpreter's code — for everything else, including unusual cases
// (overflow…): the semantics are the same.

/// Runs an IR instruction with the interpreter's code. 0: success, 1: error (stored in
/// the `Runtime`).
extern "C" fn bk_step(
    rt: *mut Runtime,
    f: *mut NativeFrame,
    ins: *const Instr,
    lo: u32,
    hi: u32,
) -> i32 {
    // SAFETY: `rt`, `f` and `ins` come from `Runtime::run_native`, valid during the call;
    // the window's pointers refer to slices of the given length.
    unsafe {
        let rt = &mut *rt;
        let f = &mut *f;
        let regs = std::slice::from_raw_parts_mut(f.regs, f.nregs);
        let cells = std::slice::from_raw_parts_mut(f.cells, f.ncells);
        let upvals = std::slice::from_raw_parts(f.upvals, f.nupvals);
        match rt.step(
            &*f.proto,
            upvals,
            &*f.ctx,
            regs,
            cells,
            &*ins,
            Span::new(lo, hi),
        ) {
            Ok(()) => 0,
            Err(e) => {
                rt.set_pending(e);
                1
            }
        }
    }
}

/// Next element of a `for` iterator: 1 if there is one (stored in `dst`), 0 otherwise.
extern "C" fn bk_iter_next(f: *mut NativeFrame, iter: u32, dst: u32) -> i32 {
    // SAFETY: as for `bk_step`.
    unsafe {
        let f = &mut *f;
        let regs = std::slice::from_raw_parts_mut(f.regs, f.nregs);
        crate::vm::iter_next(regs, iter, dst) as i32
    }
}

/// Drops the value of a register before writing a scalar value into it.
extern "C" fn bk_drop(slot: *mut Value) {
    // SAFETY: `slot` is a register of the current window.
    unsafe {
        std::ptr::drop_in_place(slot);
        slot.write(Value::Nil);
    }
}

/// Cache of a field access (`p.x`): last type seen and the field's offset. The `Rc`
/// keeps the type alive, so its address cannot refer to another type.
struct FieldCache {
    info: Option<Rc<crate::objects::TypeInfo>>,
    idx: usize,
    name: crate::intern::Symbol,
    /// Expected type (`FieldAs`), or `Value`.
    want: RegTy,
}

impl FieldCache {
    /// Field `name` of object `v` (`type` only: a `class` goes through a getter).
    fn get(&mut self, v: &Value) -> Option<Value> {
        let Value::Object(o) = v else {
            return None;
        };
        let obj = o.try_borrow().ok()?;
        let idx = match &self.info {
            Some(info) if Rc::ptr_eq(info, &obj.info) => self.idx,
            _ => {
                if obj.info.kind == crate::ast::TypeDeclKind::Class {
                    return None;
                }
                let i = obj.info.field_index(self.name)?;
                self.info = Some(obj.info.clone());
                self.idx = i;
                i
            }
        };
        let v = obj.fields[idx].clone();
        match (self.want, v) {
            (RegTy::Value, v)
            | (RegTy::Int, v @ Value::Int(_))
            | (RegTy::Float, v @ Value::Float(_))
            | (RegTy::Bool, v @ Value::Bool(_)) => Some(v),
            (RegTy::Float, Value::Int(i)) => Some(Value::Float(i as f64)),
            _ => None,
        }
    }
}

/// Fast helpers: `fn(rt, window, x, y, p) -> status`. Status 0: done; other: case not
/// handled, the interpreter runs the instruction (and produces the error, if any). For
/// `Call`, 1 signals an error (stored in the `Runtime`).
#[derive(Clone, Copy)]
enum Fast {
    GetUp,
    SetUp,
    GetCell,
    SetCell,
    GetGlobal,
    SetGlobal,
    Field,
    Call,
}

type FastFn = extern "C" fn(*mut Runtime, *mut NativeFrame, u32, u32, *const u8) -> i32;

const FAST: &[(&str, FastFn)] = &[
    ("bk_get_up", bk_get_up),
    ("bk_set_up", bk_set_up),
    ("bk_get_cell", bk_get_cell),
    ("bk_set_cell", bk_set_cell),
    ("bk_get_global", bk_get_global),
    ("bk_set_global", bk_set_global),
    ("bk_field", bk_field),
    ("bk_call", bk_call),
];

/// Registers of the window.
///
/// # Safety
/// `f` comes from `Runtime::run_native`, valid during the call.
unsafe fn frame_regs<'a>(f: *mut NativeFrame) -> &'a mut [Value] {
    unsafe { std::slice::from_raw_parts_mut((*f).regs, (*f).nregs) }
}

/// Read of a captured variable or a cell: not handled if it is not initialized yet
/// (the interpreter reports the error).
fn load_cell(regs: &mut [Value], d: u32, c: &crate::vm::Cell) -> i32 {
    let v = c.borrow().clone();
    if let Value::Uninit = v {
        return 2;
    }
    regs[d as usize] = v;
    0
}

extern "C" fn bk_get_up(_: *mut Runtime, f: *mut NativeFrame, d: u32, u: u32, _: *const u8) -> i32 {
    // SAFETY: window valid during the call (see `frame_regs`).
    unsafe {
        let up = &*(*f).upvals.add(u as usize);
        load_cell(frame_regs(f), d, up)
    }
}

extern "C" fn bk_set_up(_: *mut Runtime, f: *mut NativeFrame, u: u32, a: u32, _: *const u8) -> i32 {
    // SAFETY: same.
    unsafe {
        let up = &*(*f).upvals.add(u as usize);
        *up.borrow_mut() = frame_regs(f)[a as usize].clone();
    }
    0
}

extern "C" fn bk_get_cell(
    _: *mut Runtime,
    f: *mut NativeFrame,
    d: u32,
    c: u32,
    _: *const u8,
) -> i32 {
    // SAFETY: same.
    unsafe {
        let cell = &*(*f).cells.add(c as usize);
        load_cell(frame_regs(f), d, cell)
    }
}

extern "C" fn bk_set_cell(
    _: *mut Runtime,
    f: *mut NativeFrame,
    c: u32,
    a: u32,
    _: *const u8,
) -> i32 {
    // SAFETY: same.
    unsafe {
        let cell = &*(*f).cells.add(c as usize);
        *cell.borrow_mut() = frame_regs(f)[a as usize].clone();
    }
    0
}

extern "C" fn bk_get_global(
    rt: *mut Runtime,
    f: *mut NativeFrame,
    d: u32,
    g: u32,
    _: *const u8,
) -> i32 {
    // SAFETY: `rt` and the window are valid during the call.
    unsafe {
        match &(*rt).globals.at(g).value {
            Some(v) => {
                frame_regs(f)[d as usize] = v.clone();
                0
            }
            None => 2,
        }
    }
}

extern "C" fn bk_set_global(
    rt: *mut Runtime,
    f: *mut NativeFrame,
    g: u32,
    a: u32,
    _: *const u8,
) -> i32 {
    // SAFETY: same.
    unsafe {
        let rt = &mut *rt;
        let slot = rt.globals.at(g);
        if slot.value.is_none() || !slot.mutable {
            return 2;
        }
        let v = frame_regs(f)[a as usize].clone();
        rt.globals.assign(g, v);
    }
    0
}

extern "C" fn bk_field(_: *mut Runtime, f: *mut NativeFrame, d: u32, a: u32, p: *const u8) -> i32 {
    // SAFETY: valid window; `p` is a `FieldCache` kept alive by the `Jit`.
    unsafe {
        let cache = &mut *(p as *mut FieldCache);
        let regs = frame_regs(f);
        match cache.get(&regs[a as usize]) {
            Some(v) => {
                regs[d as usize] = v;
                0
            }
            None => 2,
        }
    }
}

extern "C" fn bk_call(
    rt: *mut Runtime,
    f: *mut NativeFrame,
    lo: u32,
    hi: u32,
    ins: *const u8,
) -> i32 {
    // SAFETY: as for `bk_step`.
    unsafe {
        let rt = &mut *rt;
        let Instr::Call {
            dst,
            f: callee,
            site,
        } = &*(ins as *const Instr)
        else {
            unreachable!()
        };
        let proto = &*(*f).proto;
        match rt.call_instr(
            proto,
            frame_regs(f),
            *dst,
            *callee,
            *site,
            Span::new(lo, hi),
        ) {
            Ok(()) => 0,
            Err(e) => {
                rt.set_pending(e);
                1
            }
        }
    }
}

struct Dyn<'a> {
    regs_ty: &'a [RegTy],
    regs: CValue,
    rt: CValue,
    frame: CValue,
    step: FuncRef,
    drop: FuncRef,
    fast: Vec<FuncRef>,
    fail: CBlock,
}

impl Dyn<'_> {
    fn off(r: Reg) -> i32 {
        r as i32 * SIZE
    }

    fn tag(&self, bx: &mut FunctionBuilder, r: Reg) -> CValue {
        bx.ins()
            .load(types::I8, MemFlagsData::trusted(), self.regs, Self::off(r))
    }

    fn payload(&self, bx: &mut FunctionBuilder, r: Reg, ty: Type) -> CValue {
        bx.ins().load(
            ty,
            MemFlagsData::trusted(),
            self.regs,
            Self::off(r) + PAYLOAD,
        )
    }

    /// Writes a scalar value (tag, payload) into a register. A `value` register may hold a
    /// resource: it is dropped first.
    fn write(&self, bx: &mut FunctionBuilder, d: Reg, tag: CValue, payload: Option<CValue>) {
        if self.regs_ty[d as usize] == RegTy::Value {
            let old = self.tag(bx, d);
            let heap = bx.ins().icmp_imm_u(
                IntCC::UnsignedGreaterThanOrEqual,
                old,
                FIRST_HEAP_TAG as i64,
            );
            let free = bx.create_block();
            let cont = bx.create_block();
            bx.ins().brif(heap, free, &[], cont, &[]);
            bx.switch_to_block(free);
            let addr = bx.ins().iadd_imm_s(self.regs, Self::off(d) as i64);
            bx.ins().call(self.drop, &[addr]);
            bx.ins().jump(cont, &[]);
            bx.switch_to_block(cont);
        }
        bx.ins()
            .store(MemFlagsData::trusted(), tag, self.regs, Self::off(d));
        if let Some(p) = payload {
            bx.ins().store(
                MemFlagsData::trusted(),
                p,
                self.regs,
                Self::off(d) + PAYLOAD,
            );
        }
    }

    fn write_const(&self, bx: &mut FunctionBuilder, d: Reg, tag: u8, payload: Option<CValue>) {
        let t = bx.ins().iconst(types::I8, tag as i64);
        self.write(bx, d, t, payload);
    }

    /// Calls `bk_step` for instruction `ins`; continues in a new block.
    fn step(&self, bx: &mut FunctionBuilder, ins: &Instr, span: Span) {
        let ptr = bx.func.dfg.value_type(self.regs);
        let ip = bx.ins().iconst(ptr, ins as *const Instr as i64);
        let lo = bx.ins().iconst(types::I32, span.start as i64);
        let hi = bx.ins().iconst(types::I32, span.end as i64);
        let call = bx.ins().call(self.step, &[self.rt, self.frame, ip, lo, hi]);
        let status = bx.inst_results(call)[0];
        let cont = bx.create_block();
        bx.ins().brif(status, self.fail, &[], cont, &[]);
        bx.switch_to_block(cont);
    }

    /// Fast helper; if it does not handle the case (non-zero status), the instruction goes
    /// through the interpreter.
    #[allow(clippy::too_many_arguments)]
    fn fast(
        &self,
        bx: &mut FunctionBuilder,
        which: Fast,
        x: u32,
        y: u32,
        p: *const u8,
        ins: &Instr,
        span: Span,
    ) {
        let ptr = bx.func.dfg.value_type(self.regs);
        let x = bx.ins().iconst(types::I32, x as i64);
        let y = bx.ins().iconst(types::I32, y as i64);
        let p = bx.ins().iconst(ptr, p as i64);
        let call = bx
            .ins()
            .call(self.fast[which as usize], &[self.rt, self.frame, x, y, p]);
        let status = bx.inst_results(call)[0];
        let slow = bx.create_block();
        let next = bx.create_block();
        bx.ins().brif(status, slow, &[], next, &[]);
        bx.switch_to_block(slow);
        self.step(bx, ins, span);
        bx.ins().jump(next, &[]);
        bx.switch_to_block(next);
    }

    /// Truthiness of a register (`nil` and `false` are falsy), as an `i8` (1: truthy).
    fn truthy(&self, bx: &mut FunctionBuilder, r: Reg) -> CValue {
        match self.regs_ty[r as usize] {
            RegTy::Bool => {
                let b = self.payload(bx, r, types::I8);
                bx.ins().icmp_imm_s(IntCC::NotEqual, b, 0)
            }
            RegTy::Int | RegTy::Float => bx.ins().iconst(types::I8, 1),
            RegTy::Value => {
                let t = self.tag(bx, r);
                let b = self.payload(bx, r, types::I8);
                let nil = bx.ins().icmp_imm_s(IntCC::Equal, t, TAG_NIL as i64);
                let is_bool = bx.ins().icmp_imm_s(IntCC::Equal, t, TAG_BOOL as i64);
                let zero = bx.ins().icmp_imm_s(IntCC::Equal, b, 0);
                let fals = bx.ins().band(is_bool, zero);
                let falsy = bx.ins().bor(nil, fals);
                bx.ins().icmp_imm_s(IntCC::Equal, falsy, 0)
            }
        }
    }
}

impl Jit {
    /// Compiles `p` to baseline native code; returns the entry point's address.
    pub fn compile_dynamic(&mut self, p: &Proto) -> Option<usize> {
        let ptr = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.returns.push(AbiParam::new(types::I32));
        self.next += 1;
        let id = self
            .module
            .declare_function(&format!("bkd{}", self.next), Linkage::Local, &sig)
            .ok()?;
        let helpers = self.dyn_helpers()?;
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, id.as_u32());
        let step = self.module.declare_func_in_func(helpers.0, &mut ctx.func);
        let iter = self.module.declare_func_in_func(helpers.1, &mut ctx.func);
        let drop = self.module.declare_func_in_func(helpers.2, &mut ctx.func);
        let fast: Vec<FuncRef> = self
            .fast_fns
            .iter()
            .map(|&id| self.module.declare_func_in_func(id, &mut ctx.func))
            .collect();
        let tc = self.module.target_config();
        let mut fctx = FunctionBuilderContext::new();
        let mut bx = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let entry = bx.create_block();
        bx.append_block_params_for_function_params(entry);
        let blocks: Vec<CBlock> = p.blocks.iter().map(|_| bx.create_block()).collect();
        let fail = bx.create_block();
        bx.switch_to_block(entry);
        let (rt, frame) = {
            let ps = bx.block_params(entry);
            (ps[0], ps[1])
        };
        let regs = bx.ins().load(
            ptr,
            MemFlagsData::trusted(),
            frame,
            std::mem::offset_of!(NativeFrame, regs) as i32,
        );
        bx.ins().jump(blocks[0], &[]);
        let d = Dyn {
            regs_ty: &p.regs,
            regs,
            rt,
            frame,
            step,
            drop,
            fast,
            fail,
        };
        for (bi, b) in p.blocks.iter().enumerate() {
            bx.switch_to_block(blocks[bi]);
            for (ins, &span) in b.instrs.iter().zip(&b.spans) {
                dyn_instr(&d, &mut bx, ins, span, &mut self.field_caches);
            }
            match &b.term {
                Term::Jump(t) => {
                    bx.ins().jump(blocks[*t as usize], &[]);
                }
                Term::Branch { cond, then, else_ } => {
                    let c = d.truthy(&mut bx, *cond);
                    bx.ins()
                        .brif(c, blocks[*then as usize], &[], blocks[*else_ as usize], &[]);
                }
                Term::IfInt {
                    op,
                    a,
                    b: y,
                    then,
                    else_,
                } => {
                    let x = d.payload(&mut bx, *a, types::I64);
                    let y = d.payload(&mut bx, *y, types::I64);
                    let c = bx.ins().icmp(int_cc(*op), x, y);
                    bx.ins()
                        .brif(c, blocks[*then as usize], &[], blocks[*else_ as usize], &[]);
                }
                Term::IfIntK {
                    op,
                    a,
                    k,
                    then,
                    else_,
                } => {
                    let x = d.payload(&mut bx, *a, types::I64);
                    let c = bx.ins().icmp_imm_s(int_cc(*op), x, *k);
                    bx.ins()
                        .brif(c, blocks[*then as usize], &[], blocks[*else_ as usize], &[]);
                }
                Term::Return { reg, shape } => {
                    let r = bx.ins().iconst(types::I32, *reg as i64);
                    let s = bx
                        .ins()
                        .iconst(types::I32, shape.unwrap_or(u32::MAX) as i64);
                    bx.ins().store(
                        MemFlagsData::trusted(),
                        r,
                        frame,
                        std::mem::offset_of!(NativeFrame, ret) as i32,
                    );
                    bx.ins().store(
                        MemFlagsData::trusted(),
                        s,
                        frame,
                        std::mem::offset_of!(NativeFrame, ret_shape) as i32,
                    );
                    let ok = bx.ins().iconst(types::I32, 0);
                    bx.ins().return_(&[ok]);
                }
                Term::IterNext {
                    iter: it,
                    dst,
                    body,
                    done,
                } => {
                    let a = bx.ins().iconst(types::I32, *it as i64);
                    let b2 = bx.ins().iconst(types::I32, *dst as i64);
                    let call = bx.ins().call(iter, &[frame, a, b2]);
                    let got = bx.inst_results(call)[0];
                    bx.ins().brif(
                        got,
                        blocks[*body as usize],
                        &[],
                        blocks[*done as usize],
                        &[],
                    );
                }
            }
        }
        bx.switch_to_block(fail);
        let one = bx.ins().iconst(types::I32, 1);
        bx.ins().return_(&[one]);
        bx.seal_all_blocks();
        bx.finalize(tc);
        self.module.define_function(id, &mut ctx).ok()?;
        self.module.clear_context(&mut ctx);
        self.module.finalize_definitions().ok()?;
        Some(self.module.get_finalized_function(id) as usize)
    }

    /// `bk_step`, `bk_iter_next`, `bk_drop` (declared on first use).
    fn dyn_helpers(&mut self) -> Option<(FuncId, FuncId, FuncId)> {
        if let Some(h) = self.dyn_fns {
            return Some(h);
        }
        let ptr = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        for t in [ptr, ptr, ptr, types::I32, types::I32] {
            sig.params.push(AbiParam::new(t));
        }
        sig.returns.push(AbiParam::new(types::I32));
        let step = self
            .module
            .declare_function("bk_step", Linkage::Import, &sig)
            .ok()?;
        let mut sig = self.module.make_signature();
        for t in [ptr, types::I32, types::I32] {
            sig.params.push(AbiParam::new(t));
        }
        sig.returns.push(AbiParam::new(types::I32));
        let iter = self
            .module
            .declare_function("bk_iter_next", Linkage::Import, &sig)
            .ok()?;
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        let drop = self
            .module
            .declare_function("bk_drop", Linkage::Import, &sig)
            .ok()?;
        let mut sig = self.module.make_signature();
        for t in [ptr, ptr, types::I32, types::I32, ptr] {
            sig.params.push(AbiParam::new(t));
        }
        sig.returns.push(AbiParam::new(types::I32));
        for (name, _) in FAST {
            let id = self
                .module
                .declare_function(name, Linkage::Import, &sig)
                .ok()?;
            self.fast_fns.push(id);
        }
        self.dyn_fns = Some((step, iter, drop));
        self.dyn_fns
    }
}

/// Translates an instruction: inline for simple cases, otherwise `bk_step`.
#[allow(clippy::vec_box)]
fn dyn_instr(
    d: &Dyn,
    bx: &mut FunctionBuilder,
    ins: &Instr,
    span: Span,
    caches: &mut Vec<Box<FieldCache>>,
) {
    use Instr::*;
    let ty = |r: Reg| d.regs_ty[r as usize];
    let null = std::ptr::null::<u8>();
    match ins {
        GetUp(r, u) => d.fast(bx, Fast::GetUp, *r, *u, null, ins, span),
        SetUp(u, a) => d.fast(bx, Fast::SetUp, *u, *a, null, ins, span),
        GetCell(r, c) => d.fast(bx, Fast::GetCell, *r, *c, null, ins, span),
        SetCell(c, a) => d.fast(bx, Fast::SetCell, *c, *a, null, ins, span),
        GetGlobal(r, g) => d.fast(bx, Fast::GetGlobal, *r, *g, null, ins, span),
        SetGlobal(g, a) => d.fast(bx, Fast::SetGlobal, *g, *a, null, ins, span),
        Field(r, a, name) | FieldAs(r, a, name, _) => {
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
            let p = &**caches.last().unwrap() as *const FieldCache as *const u8;
            d.fast(bx, Fast::Field, *r, *a, p, ins, span);
        }
        Call { .. } => {
            let ptr = bx.func.dfg.value_type(d.regs);
            let ip = bx.ins().iconst(ptr, ins as *const Instr as i64);
            let lo = bx.ins().iconst(types::I32, span.start as i64);
            let hi = bx.ins().iconst(types::I32, span.end as i64);
            let call = bx
                .ins()
                .call(d.fast[Fast::Call as usize], &[d.rt, d.frame, lo, hi, ip]);
            let status = bx.inst_results(call)[0];
            let cont = bx.create_block();
            bx.ins().brif(status, d.fail, &[], cont, &[]);
            bx.switch_to_block(cont);
        }
        // Guard: the expected tag, or an `int` to `float`; otherwise the interpreter produces
        // the error.
        Guard(r, a, t) if *t != RegTy::Value && ty(*r) == *t => {
            let (tag, cty) = scalar(*t);
            let at = d.tag(bx, *a);
            let ok = bx.ins().icmp_imm_s(IntCC::Equal, at, tag as i64);
            let fast = bx.create_block();
            let other = bx.create_block();
            let slow = bx.create_block();
            let next = bx.create_block();
            bx.ins().brif(ok, fast, &[], other, &[]);
            bx.switch_to_block(fast);
            let v = d.payload(bx, *a, cty);
            d.write_const(bx, *r, tag, Some(v));
            bx.ins().jump(next, &[]);
            bx.switch_to_block(other);
            if *t == RegTy::Float {
                let conv = bx.create_block();
                let is_int = bx.ins().icmp_imm_s(IntCC::Equal, at, TAG_INT as i64);
                bx.ins().brif(is_int, conv, &[], slow, &[]);
                bx.switch_to_block(conv);
                let x = d.payload(bx, *a, types::I64);
                let v = bx.ins().fcvt_from_sint(types::F64, x);
                d.write_const(bx, *r, TAG_FLOAT, Some(v));
                bx.ins().jump(next, &[]);
            } else {
                bx.ins().jump(slow, &[]);
            }
            bx.switch_to_block(slow);
            d.step(bx, ins, span);
            bx.ins().jump(next, &[]);
            bx.switch_to_block(next);
        }
        Int(r, n) => {
            let v = bx.ins().iconst(types::I64, *n);
            d.write_const(bx, *r, TAG_INT, Some(v));
        }
        Float(r, f) => {
            let v = bx.ins().f64const(*f);
            d.write_const(bx, *r, TAG_FLOAT, Some(v));
        }
        Bool(r, b) => {
            let v = bx.ins().iconst(types::I8, *b as i64);
            d.write_const(bx, *r, TAG_BOOL, Some(v));
        }
        Nil(r) => d.write_const(bx, *r, TAG_NIL, None),
        Move(dst, a) if ty(*a) != RegTy::Value => {
            let (tag, t) = scalar(ty(*a));
            let v = d.payload(bx, *a, t);
            d.write_const(bx, *dst, tag, Some(v));
        }
        // Copy of a value: direct if it is scalar, otherwise through the interpreter.
        Move(dst, a) if ty(*dst) == RegTy::Value => {
            let t = d.tag(bx, *a);
            let scalar = bx
                .ins()
                .icmp_imm_u(IntCC::UnsignedLessThan, t, FIRST_HEAP_TAG as i64);
            let fast = bx.create_block();
            let slow = bx.create_block();
            let next = bx.create_block();
            bx.ins().brif(scalar, fast, &[], slow, &[]);
            bx.switch_to_block(fast);
            let v = d.payload(bx, *a, types::I64);
            d.write(bx, *dst, t, Some(v));
            bx.ins().jump(next, &[]);
            bx.switch_to_block(slow);
            d.step(bx, ins, span);
            bx.ins().jump(next, &[]);
            bx.switch_to_block(next);
        }
        IBin(op, r, a, b) => {
            let x = d.payload(bx, *a, types::I64);
            let y = d.payload(bx, *b, types::I64);
            int_fast(d, bx, ins, span, *op, *r, x, y);
        }
        IBinK(op, r, a, k) => {
            let x = d.payload(bx, *a, types::I64);
            let y = bx.ins().iconst(types::I64, *k);
            int_fast(d, bx, ins, span, *op, *r, x, y);
        }
        FBin(op, r, a, b) if *op != BinOp::Rem => {
            let x = d.payload(bx, *a, types::F64);
            let y = d.payload(bx, *b, types::F64);
            float_fast(d, bx, *op, *r, x, y);
        }
        IToF(r, a) => {
            let x = d.payload(bx, *a, types::I64);
            let v = bx.ins().fcvt_from_sint(types::F64, x);
            d.write_const(bx, *r, TAG_FLOAT, Some(v));
        }
        FNeg(r, a) => {
            let x = d.payload(bx, *a, types::F64);
            let v = bx.ins().fneg(x);
            d.write_const(bx, *r, TAG_FLOAT, Some(v));
        }
        Not(r, a) => {
            let t = d.truthy(bx, *a);
            let v = bx.ins().icmp_imm_s(IntCC::Equal, t, 0);
            d.write_const(bx, *r, TAG_BOOL, Some(v));
        }
        // Generic operation: inline for two integers or two floats.
        Bin(op, r, a, b) if is_arith(*op) || is_cmp(*op) => {
            let ta = d.tag(bx, *a);
            let tb = d.tag(bx, *b);
            let ia = bx.ins().icmp_imm_s(IntCC::Equal, ta, TAG_INT as i64);
            let ib = bx.ins().icmp_imm_s(IntCC::Equal, tb, TAG_INT as i64);
            let both_int = bx.ins().band(ia, ib);
            let ints = bx.create_block();
            let other = bx.create_block();
            let floats = bx.create_block();
            let slow = bx.create_block();
            let next = bx.create_block();
            bx.ins().brif(both_int, ints, &[], other, &[]);
            bx.switch_to_block(ints);
            let x = d.payload(bx, *a, types::I64);
            let y = d.payload(bx, *b, types::I64);
            int_op_into(d, bx, *op, *r, x, y, slow);
            bx.ins().jump(next, &[]);
            bx.switch_to_block(other);
            let fa = bx.ins().icmp_imm_s(IntCC::Equal, ta, TAG_FLOAT as i64);
            let fb = bx.ins().icmp_imm_s(IntCC::Equal, tb, TAG_FLOAT as i64);
            let both_float = bx.ins().band(fa, fb);
            let not_rem = *op != BinOp::Rem;
            if not_rem {
                bx.ins().brif(both_float, floats, &[], slow, &[]);
            } else {
                bx.ins().jump(slow, &[]);
            }
            bx.switch_to_block(floats);
            if not_rem {
                let x = d.payload(bx, *a, types::F64);
                let y = d.payload(bx, *b, types::F64);
                float_fast(d, bx, *op, *r, x, y);
                bx.ins().jump(next, &[]);
            } else {
                bx.ins().jump(slow, &[]);
            }
            bx.switch_to_block(slow);
            d.step(bx, ins, span);
            bx.ins().jump(next, &[]);
            bx.switch_to_block(next);
        }
        _ => d.step(bx, ins, span),
    }
}

fn scalar(t: RegTy) -> (u8, Type) {
    match t {
        RegTy::Int => (TAG_INT, types::I64),
        RegTy::Float => (TAG_FLOAT, types::F64),
        RegTy::Bool => (TAG_BOOL, types::I8),
        RegTy::Value => unreachable!(),
    }
}

/// Inline integer operation; edge case (overflow, division by zero): `bk_step`.
#[allow(clippy::too_many_arguments)]
fn int_fast(
    d: &Dyn,
    bx: &mut FunctionBuilder,
    ins: &Instr,
    span: Span,
    op: BinOp,
    r: Reg,
    x: CValue,
    y: CValue,
) {
    let slow = bx.create_block();
    let next = bx.create_block();
    int_op_into(d, bx, op, r, x, y, slow);
    bx.ins().jump(next, &[]);
    bx.switch_to_block(slow);
    d.step(bx, ins, span);
    bx.ins().jump(next, &[]);
    bx.switch_to_block(next);
}

/// Computes `x op y` and writes it to `r`; branches to `slow` for any edge case.
fn int_op_into(
    d: &Dyn,
    bx: &mut FunctionBuilder,
    op: BinOp,
    r: Reg,
    x: CValue,
    y: CValue,
    slow: CBlock,
) {
    let v = int_bin(bx, op, x, y, &mut |bx, cond, _| {
        let cont = bx.create_block();
        bx.ins().brif(cond, slow, &[], cont, &[]);
        bx.switch_to_block(cont);
    });
    let tag = if is_cmp(op) { TAG_BOOL } else { TAG_INT };
    d.write_const(bx, r, tag, Some(v));
}

fn float_fast(d: &Dyn, bx: &mut FunctionBuilder, op: BinOp, r: Reg, x: CValue, y: CValue) {
    let v = match op {
        BinOp::Add => bx.ins().fadd(x, y),
        BinOp::Sub => bx.ins().fsub(x, y),
        BinOp::Mul => bx.ins().fmul(x, y),
        BinOp::Div => bx.ins().fdiv(x, y),
        op => bx.ins().fcmp(float_cc(op), x, y),
    };
    let tag = if is_cmp(op) { TAG_BOOL } else { TAG_FLOAT };
    d.write_const(bx, r, tag, Some(v));
}
