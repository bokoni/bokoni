//! Native compilation (JIT) with Cranelift.
//!
//! Compiled functions are those whose parameters are of known types: annotated, or found
//! from the arguments of a call (a copy per types, for a parameter not annotated or a
//! `forall` type parameter). What the body does that is not compiled is done by the
//! interpreter; same result in every case, only slower.
//!
//! - The IR's `value` registers are typed by an analysis: the result of calling a `: int`
//!   function is an `int`, and so on. If a register cannot be typed, the function is not
//!   compiled.
//! - Same rules as the interpreter: overflow, division by zero and call depth are checked,
//!   with the same messages and the same call stack.
//! - Strings (`str`) are pointers to the values; their operations (`+`, comparisons,
//!   interpolation, `len`) call the interpreter's code (`bk_str_add`, `bk_concat`…).
//! - Small functions and getters are inlined (`inlinable`, `emit_inline`); an inlined
//!   method's class is checked with a cache per call site.
//! - Lists are modified in place (`xs << x`, `xs[i] = v`). Where a list may change (this
//!   function modifies one, or calls another), the elements read are copied rather than
//!   pointed to, and a `for` goes over a copy of the list.
//! - What is not compiled natively is done by the interpreter (`bk_call_boxed`,
//!   `bk_binary`…): a call of a function that is not compiled or of a built-in, any
//!   method, an operation on such a result (`JTy::Boxed`, a value of any type).
//! - When an assumption does not hold (value of another type, index out of bounds,
//!   method redefined, native stack nearly full), the native code is not abandoned: the
//!   interpreter resumes at that instruction, with the registers rebuilt (`DeoptSite`,
//!   `bk_resume`), and its result is the function's. Nothing is redone. The function is
//!   then recompiled with that instruction done by the interpreter (`Proto::deopted`);
//!   after `MAX_DEOPTS` places, it stays interpreted.
//! - A call to a global function is direct (native code → native code). If the global is
//!   redefined, the calling code is invalidated, then recompiled when needed.
//! - A method is compiled for the class that declares it: `self` is a pointer, its
//!   annotated `int` / `float` / `bool` fields (`@x`) are read and written in place. A
//!   call `obj.m()` from native code is direct, checked against a redefinition by the
//!   object's class (otherwise resumed in the interpreter). The type of a field is
//!   checked on every access (a method called may change it).
//! - A `forall` function (its annotations `T` are `any` at run time) is compiled as a copy
//!   typed for the arguments of a call (`T = int`, see `Proto::specialize`): one copy per
//!   type, called directly by native code, or by the interpreter when the arguments fit.
//!   Same for a method of a generic class (`class Box(T)`), `T` found from the fields of
//!   `self` and the arguments, or from the annotation of the receiver (`a: Acc(int)`: the
//!   type arguments are kept in compiled annotations); its fields of type `T` are
//!   checked on entry.
//! - The interpreter calls native code directly (`Runtime::native_fast`) when the
//!   arguments are exactly of its native types.
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
    /// The runtime, for a resumption in the interpreter (`bk_resume`).
    pub rt: *mut Runtime,
    /// Error raised by the interpreter during a resumption (`err` = `PENDING`).
    pub pending: Option<Error>,
    /// Result of a resumption that does not fit the native result type (status 2).
    pub returned: Option<Value>,
}

/// Resumptions in the interpreter (at distinct places) after which a function stays
/// interpreted.
const MAX_DEOPTS: usize = 4;

/// `JitCtx::err`: the error is in `JitCtx::pending`.
const PENDING: i64 = -2;

const DEPTH_OFFSET: i32 = 0;
const MAX_DEPTH_OFFSET: i32 = 8;
const ERR_OFFSET: i32 = 16;

#[derive(Clone, Copy)]
enum ErrKind {
    Overflow,
    DivZero,
}

/// How to rebuild an interpreter register from the words written by native code at a
/// resumption point.
#[derive(Clone, Copy)]
enum DKind {
    Int,
    Float,
    Bool,
    /// Address of a value (object, list, `self`): copied; null if not assigned yet.
    Ref,
    /// `for` iterator: address of the elements, count, position (3 words).
    Iter,
    /// Value of a global (function called, type constructed).
    Global(u32),
    Nil,
    /// Result of the call just made (`JitCtx::returned`).
    Returned,
}

/// Resumption point: an assumption of the native code does not hold (value of another
/// type, index out of bounds, method redefined, native stack nearly full…). Rather than
/// redoing the call (which would redo its side effects), the interpreter takes over at
/// this instruction, with the registers rebuilt.
struct DeoptSite {
    proto: Rc<Proto>,
    block: u32,
    /// Instruction of the block (its number of instructions: the terminator).
    instr: u32,
    /// Register, how to rebuild it, offset of its words in the buffer.
    regs: Vec<(Reg, DKind, u32)>,
    ret: RegTy,
    /// A check on entry (the type of an object): no instruction to give the
    /// interpreter when recompiling.
    entry: bool,
}

/// Registers live before each instruction (`[block][k]`, `k` = number of instructions:
/// before the terminator): read later before being written. Only those are written at
/// a resumption point (the others would stay alive for nothing in the native code).
fn liveness(p: &Proto) -> Vec<Vec<Vec<bool>>> {
    let n = p.regs.len();
    let term_reads = |t: &Term| -> Vec<Reg> {
        match t {
            Term::Branch { cond, .. } => vec![*cond],
            Term::IfInt { a, b, .. } => vec![*a, *b],
            Term::IfIntK { a, .. } => vec![*a],
            Term::Return { reg, .. } => vec![*reg],
            Term::IterNext { iter, .. } => vec![*iter],
            Term::Jump(_) | Term::Reraise(_) => vec![],
        }
    };
    let mut live: Vec<Vec<Vec<bool>>> = p
        .blocks
        .iter()
        .map(|b| vec![vec![false; n]; b.instrs.len() + 1])
        .collect();
    loop {
        let mut changed = false;
        for (bi, b) in p.blocks.iter().enumerate().rev() {
            // Live after the terminator: live at the entry of the successors.
            let mut cur = vec![false; n];
            let mut t = b.term.clone();
            for s in t.targets_mut() {
                for (r, l) in live[*s as usize][0].iter().enumerate() {
                    cur[r] |= *l;
                }
            }
            if let Term::IterNext { dst, .. } = &b.term {
                cur[*dst as usize] = false;
            }
            for r in term_reads(&b.term) {
                cur[r as usize] = true;
            }
            let k = b.instrs.len();
            if live[bi][k] != cur {
                live[bi][k] = cur.clone();
                changed = true;
            }
            for (k, ins) in b.instrs.iter().enumerate().rev() {
                match reads_writes(p, ins) {
                    Some((reads, write)) => {
                        if let Some(w) = write {
                            cur[w as usize] = false;
                        }
                        for r in reads {
                            cur[r as usize] = true;
                        }
                    }
                    // Not compiled: conservatively everything stays live.
                    None => cur.iter_mut().for_each(|l| *l = true),
                }
                if live[bi][k] != cur {
                    live[bi][k] = cur.clone();
                    changed = true;
                }
            }
        }
        if !changed {
            return live;
        }
    }
}

/// Resumption in the interpreter (see `DeoptSite`): rebuilds the registers from `buf`,
/// runs the rest of the function, writes its result into `out` (its native type, or the
/// value for an object or a list). 0: done; 1: error (in `JitCtx::pending`).
extern "C" fn bk_resume(
    ctx: *mut JitCtx,
    site: *const DeoptSite,
    buf: *const u64,
    out: *mut u64,
) -> i32 {
    // SAFETY: `site` is kept alive by the `Jit`; `buf` holds the words described by the
    // site, written just before the call; the addresses of `DKind::Ref` / `Iter` are
    // values alive during the call; `ctx.rt` is the runtime running this native code.
    // No reference to the context or the runtime is held while the interpreter runs
    // (it may call native code again).
    unsafe {
        let site = &*site;
        let p = &site.proto;
        let rt = (*ctx).rt;
        let mut regs: Vec<Value> = vec![Value::Nil; p.regs.len()];
        for &(r, kind, off) in &site.regs {
            let w = |j: u32| *buf.add((off + j) as usize);
            regs[r as usize] = match kind {
                DKind::Int => Value::Int(w(0) as i64),
                DKind::Float => Value::Float(f64::from_bits(w(0))),
                DKind::Bool => Value::Bool(w(0) & 0xff != 0),
                DKind::Ref if w(0) == 0 => Value::Nil,
                DKind::Ref => (*(w(0) as *const Value)).clone(),
                DKind::Iter if w(0) == 0 => Value::Nil,
                DKind::Iter => {
                    let data = w(0) as *const Value;
                    let rest: Vec<Value> = (w(2)..w(1))
                        .map(|i| (*data.add(i as usize)).clone())
                        .collect();
                    Value::Iter(Rc::new(std::cell::RefCell::new(
                        crate::vm::IterState::Values(rest.into_iter()),
                    )))
                }
                DKind::Global(slot) => (*rt).globals.at(slot).value.clone().unwrap_or(Value::Nil),
                DKind::Nil => Value::Nil,
                DKind::Returned => (*ctx).returned.take().unwrap_or(Value::Nil),
            };
        }
        // The assumption did not hold: recompiled with that instruction done by the
        // interpreter; interpreted after a few times.
        {
            let mut d = p.deopted.borrow_mut();
            let at = if site.entry {
                (u32::MAX, d.len() as u32)
            } else {
                (site.block, site.instr)
            };
            if !d.contains(&at) {
                d.push(at);
            }
            *p.jit.borrow_mut() = if d.len() >= MAX_DEOPTS {
                JitState::Rejected
            } else {
                JitState::Untried
            };
        }
        // A typed copy: the function as written runs (same code, its annotations `T`).
        let origin = p.spec.as_ref().and_then(|s| s.origin.upgrade());
        let p = origin.as_ref().unwrap_or(p);
        let saved = (
            (*ctx).depth,
            (*ctx).max_depth,
            (*ctx).err,
            std::mem::take(&mut (*ctx).traces),
        );
        let r = (*rt).resume(p, &mut regs, site.block, site.instr, saved.0 as usize);
        (*ctx).depth = saved.0;
        (*ctx).max_depth = saved.1;
        (*ctx).err = saved.2;
        (*ctx).traces = saved.3;
        (*ctx).rt = rt;
        match r {
            Ok(v) => match (site.ret, v) {
                (RegTy::Int, Value::Int(i)) => {
                    *out = i as u64;
                    0
                }
                (RegTy::Float, Value::Float(f)) => {
                    *out = f.to_bits();
                    0
                }
                (RegTy::Bool, Value::Bool(b)) => {
                    *out = b as u64;
                    0
                }
                (RegTy::Value, v) => {
                    *(out as *mut Value) = v;
                    0
                }
                // A typed copy promises its type, the function as written may return
                // another: the caller is resumed too, with this value.
                (_, v) => {
                    (*ctx).returned = Some(v);
                    2
                }
            },
            Err(e) => {
                (*ctx).pending = Some(e);
                (*ctx).err = PENDING;
                1
            }
        }
    }
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

/// Fields of `self` (method of a class or `type` `info`, or of a subclass: its fields come
/// first), writable; null if `self` is of another type or is being borrowed.
extern "C" fn bk_this_fields(
    obj: *const Value,
    info: *const crate::objects::TypeInfo,
) -> *mut Value {
    // SAFETY: `obj` is an argument alive during the native call; the field vector never
    // grows, and only this native code accesses it during the call.
    unsafe {
        let Value::Object(o) = &*obj else {
            return std::ptr::null_mut();
        };
        let Ok(mut b) = o.try_borrow_mut() else {
            return std::ptr::null_mut();
        };
        let mut t = Some(&b.info);
        while let Some(i) = t {
            if Rc::as_ptr(i) == info {
                return b.fields.as_mut_ptr();
            }
            t = i.parent.as_ref();
        }
        std::ptr::null_mut()
    }
}

/// 0 if method `name` of object `obj` is `proto` (not redefined by a subclass), 1
/// otherwise.
extern "C" fn bk_method_is(obj: *const Value, name: u32, proto: *const Proto) -> i32 {
    // SAFETY: `obj` is a value alive during the native call.
    unsafe {
        match &*obj {
            Value::Object(o) => match o.try_borrow() {
                Ok(b) => match b.info.methods.get(&crate::intern::Symbol(name)) {
                    Some(m) if Rc::as_ptr(m) == proto => 0,
                    _ => 1,
                },
                Err(_) => 1,
            },
            _ => 1,
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

/// Inline cache of an inlined method call: the class last seen whose method `name` is
/// `proto` (not redefined).
struct MethodCache {
    info: std::cell::RefCell<Option<Rc<crate::objects::TypeInfo>>>,
    name: crate::intern::Symbol,
    proto: *const Proto,
}

/// Fields of `obj` if its method `name` is the inlined one (cached by class); null
/// otherwise (resumed in the interpreter, which makes the call).
extern "C" fn bk_fields_cached(obj: *const Value, cache: *const MethodCache) -> *mut Value {
    // SAFETY: `obj` is a live value, `cache` is kept alive by the `Jit`.
    unsafe {
        let cache = &*cache;
        let Value::Object(o) = &*obj else {
            return std::ptr::null_mut();
        };
        let Ok(mut b) = o.try_borrow_mut() else {
            return std::ptr::null_mut();
        };
        let known = cache
            .info
            .borrow()
            .as_ref()
            .is_some_and(|i| Rc::ptr_eq(i, &b.info));
        if !known {
            match b.info.methods.get(&cache.name) {
                Some(m) if Rc::as_ptr(m) == cache.proto => {
                    *cache.info.borrow_mut() = Some(b.info.clone());
                }
                _ => return std::ptr::null_mut(),
            }
        }
        b.fields.as_mut_ptr()
    }
}

/// A small function, inlined at its calls: one block ending with its return, no call,
/// simple values only (arithmetic, fields of `self`), from the same script (its errors
/// point into it).
fn inlinable(q: &Proto, p: &Proto) -> bool {
    use Instr::*;
    let [b] = q.blocks.as_slice() else {
        return false;
    };
    Rc::ptr_eq(&q.script, &p.script)
        && q.handlers.is_empty()
        && b.instrs.len() <= 16
        && matches!(b.term, Term::Return { .. })
        && native_sig(q).is_some_and(|(ps, r)| {
            r != RegTy::Value
                && ps
                    .iter()
                    .skip((q.kind == ProtoKind::Method) as usize)
                    .all(|t| *t != RegTy::Value)
        })
        && b.instrs.iter().all(|i| {
            matches!(
                i,
                Int(..)
                    | Float(..)
                    | Bool(..)
                    | Move(..)
                    | Guard(..)
                    | IBin(..)
                    | IBinK(..)
                    | FBin(..)
                    | INeg(..)
                    | FNeg(..)
                    | IToF(..)
                    | FieldAt(_, 0, _)
            )
        })
}

/// Emits the body of inlinable function `q` (`plan`: its analysis) with its native
/// arguments `args` (`fields`: those of `self` for a method); its result. `err`:
/// integer overflow / division by zero (span in `q`); `bail`: a field of another type.
#[allow(clippy::too_many_arguments)]
fn emit_inline(
    bx: &mut FunctionBuilder,
    q: &Proto,
    plan: &Plan,
    args: &[CValue],
    fields: Option<CValue>,
    frem_ref: FuncRef,
    err: &mut dyn FnMut(&mut FunctionBuilder, CValue, ErrKind, Span),
    bail: &mut dyn FnMut(&mut FunctionBuilder, CValue),
) -> Option<CValue> {
    use Instr::*;
    let b = &q.blocks[0];
    let mut vals: FxHashMap<Reg, (CValue, JTy)> = FxHashMap::default();
    for (prm, v) in q.params.iter().zip(args) {
        vals.insert(prm.reg, (*v, plan.entry[0][prm.reg as usize].get()?));
    }
    for (k, (ins, &span)) in b.instrs.iter().zip(&b.spans).enumerate() {
        let dty = plan.defs[0][k];
        let get = |r: &Reg| vals.get(r).copied();
        let (d, v) = match ins {
            Int(d, n) => (*d, bx.ins().iconst(types::I64, *n)),
            Float(d, f) => (*d, bx.ins().f64const(*f)),
            Bool(d, v) => (*d, bx.ins().iconst(types::I8, *v as i64)),
            Move(d, a) | Guard(d, a, _) => {
                let (v, t) = get(a)?;
                match (dty?, t) {
                    (JTy::Float, JTy::Int) => (*d, bx.ins().fcvt_from_sint(types::F64, v)),
                    _ => (*d, v),
                }
            }
            IBin(op, d, a, c) => {
                let (x, y) = (get(a)?.0, get(c)?.0);
                (
                    *d,
                    int_bin(bx, *op, x, y, &mut |bx, c, k| err(bx, c, k, span)),
                )
            }
            IBinK(op, d, a, n) => {
                let x = get(a)?.0;
                let y = bx.ins().iconst(types::I64, *n);
                (
                    *d,
                    int_bin(bx, *op, x, y, &mut |bx, c, k| err(bx, c, k, span)),
                )
            }
            FBin(op, d, a, c) => {
                let (x, y) = (get(a)?.0, get(c)?.0);
                (*d, float_bin(bx, *op, x, y, frem_ref))
            }
            INeg(d, a) => {
                let x = get(a)?.0;
                let c = bx.ins().icmp_imm_s(IntCC::Equal, x, i64::MIN);
                err(bx, c, ErrKind::Overflow, span);
                (*d, bx.ins().ineg(x))
            }
            FNeg(d, a) => (*d, bx.ins().fneg(get(a)?.0)),
            IToF(d, a) => (*d, bx.ins().fcvt_from_sint(types::F64, get(a)?.0)),
            FieldAt(d, 0, idx) => {
                let fp = fields?;
                let off = *idx as i32 * crate::value::layout::SIZE;
                let (cty, want) = match dty? {
                    JTy::Float => (types::F64, crate::value::layout::TAG_FLOAT),
                    JTy::Bool => (types::I8, crate::value::layout::TAG_BOOL),
                    _ => (types::I64, crate::value::layout::TAG_INT),
                };
                let tag = bx.ins().load(types::I8, MemFlagsData::trusted(), fp, off);
                let bad = bx.ins().icmp_imm_s(IntCC::NotEqual, tag, want as i64);
                bail(bx, bad);
                let v = bx.ins().load(
                    cty,
                    MemFlagsData::trusted(),
                    fp,
                    off + crate::value::layout::PAYLOAD,
                );
                (*d, v)
            }
            _ => return None,
        };
        vals.insert(d, (v, dty?));
    }
    let Term::Return { reg, .. } = &b.term else {
        return None;
    };
    let (v, t) = vals.get(reg).copied()?;
    let (_, ret) = native_sig(q)?;
    Some(match (ret, t) {
        (RegTy::Float, JTy::Int) => bx.ins().fcvt_from_sint(types::F64, v),
        _ => v,
    })
}

/// Function called through the interpreter from native code.
#[derive(Clone, Copy)]
enum BoxedCallee {
    /// Value of a global (function, type constructed).
    Global(u32),
    /// The first word: the function value.
    Value,
    /// Method `name` of the first word, called from `access`.
    Method(crate::intern::Symbol, crate::ir::Access),
}

/// Call made by the interpreter for native code: a function that is not compiled, a
/// built-in, a method of an object of unknown type… (`bk_call_boxed`).
struct BoxedCall {
    callee: BoxedCallee,
    /// How to rebuild each word (callee or receiver first, then the arguments).
    words: Vec<DKind>,
    /// Native type expected for the result: 0 int, 1 float, 2 bool, 3 any (a value into
    /// the slot), 4 none (not kept).
    want: u8,
    span: Span,
}

/// Value of a word written by native code.
///
/// # Safety
/// For `DKind::Ref`, `w` is null or the address of a live value; `rt` is valid.
unsafe fn word_value(rt: *mut Runtime, kind: DKind, w: u64) -> Value {
    unsafe {
        match kind {
            DKind::Int => Value::Int(w as i64),
            DKind::Float => Value::Float(f64::from_bits(w)),
            DKind::Bool => Value::Bool(w & 0xff != 0),
            DKind::Ref if w == 0 => Value::Nil,
            DKind::Ref => (*(w as *const Value)).clone(),
            DKind::Global(slot) => (*rt).globals.at(slot).value.clone().unwrap_or(Value::Nil),
            _ => Value::Nil,
        }
    }
}

/// Runs `f` (an interpreter operation) with the native context saved around it (the
/// interpreter may run native code again), at the native call depth.
///
/// # Safety
/// `ctx` is the context of the running native code, `ctx.rt` its runtime.
unsafe fn with_interpreter<T>(ctx: *mut JitCtx, f: impl FnOnce(&mut Runtime) -> T) -> T {
    unsafe {
        let rt = (*ctx).rt;
        let saved = (
            (*ctx).depth,
            (*ctx).max_depth,
            (*ctx).err,
            std::mem::take(&mut (*ctx).traces),
        );
        let depth = (*rt).depth;
        (*rt).depth = saved.0 as usize;
        let r = f(&mut *rt);
        (*rt).depth = depth;
        (*ctx).depth = saved.0;
        (*ctx).max_depth = saved.1;
        (*ctx).err = saved.2;
        (*ctx).traces = saved.3;
        (*ctx).rt = rt;
        r
    }
}

/// Call through the interpreter (see `BoxedCall`): 0, its result in `out`; 1, an error
/// (`JitCtx::pending`); 2, a result of another type than expected (`JitCtx::returned`:
/// the native code resumes after the call with it).
extern "C" fn bk_call_boxed(
    ctx: *mut JitCtx,
    call: *const BoxedCall,
    buf: *const u64,
    out: *mut u64,
) -> i32 {
    // SAFETY: `call` is kept alive by the `Jit`; `buf` holds its words, written just
    // before; `out` is a slot of the native frame (a value for `want` 3).
    unsafe {
        let call = &*call;
        let rt = (*ctx).rt;
        let mut vals: Vec<Value> = call
            .words
            .iter()
            .enumerate()
            .map(|(k, kind)| word_value(rt, *kind, *buf.add(k)))
            .collect();
        let r = with_interpreter(ctx, |rt| match call.callee {
            BoxedCallee::Global(slot) => {
                let f = rt.globals.at(slot).value.clone().unwrap_or(Value::Nil);
                rt.call_value(&f, crate::vm::CallArgs::plain(vals), call.span)
            }
            BoxedCallee::Value => {
                let f = vals.remove(0);
                rt.call_value(&f, crate::vm::CallArgs::plain(vals), call.span)
            }
            BoxedCallee::Method(name, access) => {
                let recv = vals.remove(0);
                rt.call_method_named(
                    recv,
                    name,
                    crate::vm::CallArgs::plain(vals),
                    access,
                    call.span,
                )
            }
        });
        match r {
            Ok(v) => match (call.want, v) {
                (0, Value::Int(i)) => {
                    *out = i as u64;
                    0
                }
                (1, Value::Float(f)) => {
                    *out = f.to_bits();
                    0
                }
                (1, Value::Int(i)) => {
                    *out = (i as f64).to_bits();
                    0
                }
                (2, Value::Bool(b)) => {
                    *out = b as u64;
                    0
                }
                (3, v) => {
                    *(out as *mut Value) = v;
                    0
                }
                (4, _) => 0,
                (_, v) => {
                    (*ctx).returned = Some(v);
                    2
                }
            },
            Err(e) => {
                (*ctx).pending = Some(e);
                (*ctx).err = PENDING;
                1
            }
        }
    }
}

/// All the binary operations, by code (`op_code`).
const ALL_OPS: [BinOp; 14] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::Div,
    BinOp::Rem,
    BinOp::Eq,
    BinOp::Ne,
    BinOp::Lt,
    BinOp::Le,
    BinOp::Gt,
    BinOp::Ge,
    BinOp::And,
    BinOp::Or,
    BinOp::Push,
];

fn op_code(op: BinOp) -> i64 {
    ALL_OPS.iter().position(|o| *o == op).unwrap() as i64
}

fn dkind_code(k: DKind) -> i64 {
    match k {
        DKind::Int => 0,
        DKind::Float => 1,
        DKind::Bool => 2,
        DKind::Ref => 3,
        _ => 4,
    }
}

fn dkind_of_code(c: i32) -> DKind {
    match c {
        0 => DKind::Int,
        1 => DKind::Float,
        2 => DKind::Bool,
        3 => DKind::Ref,
        _ => DKind::Nil,
    }
}

/// `a op b` on values of any type (the interpreter's operation) into the slot `out`: 0,
/// or 1 (its error, `JitCtx::pending`).
#[allow(clippy::too_many_arguments)]
extern "C" fn bk_binary(
    ctx: *mut JitCtx,
    op: i32,
    a: u64,
    ka: i32,
    b: u64,
    kb: i32,
    out: *mut Value,
    span: u64,
) -> i32 {
    // SAFETY: the words are values written by native code (`word_value`); `out` is a
    // slot of the native frame.
    unsafe {
        let rt = (*ctx).rt;
        let (l, r) = (
            word_value(rt, dkind_of_code(ka), a),
            word_value(rt, dkind_of_code(kb), b),
        );
        let span = Span::new(span as u32, (span >> 32) as u32);
        match (*rt).binary(ALL_OPS[op as usize], l, r, span) {
            Ok(v) => {
                *out = v;
                0
            }
            Err(e) => {
                (*ctx).pending = Some(e);
                (*ctx).err = PENDING;
                1
            }
        }
    }
}

/// A native value into the slot `out` (result of a function whose result is not
/// annotated).
extern "C" fn bk_box_into(out: *mut Value, bits: u64, kind: i32) {
    // SAFETY: `out` is a value slot of the caller; for a reference, `bits` is a live value.
    unsafe {
        *out = match kind {
            4 => Value::Nil,
            k => from_bits(kind_of_code(k), bits),
        };
    }
}

/// Truthiness of a value (`if v`, `not v`).
extern "C" fn bk_truthy(v: *const Value) -> i8 {
    // SAFETY: `v` is null or a live value.
    unsafe { (!v.is_null() && (*v).truthy()) as i8 }
}

/// Value of global `slot` into the slot `out`: 0, or 1 if it is not defined (resumed in
/// the interpreter, which gives the error).
extern "C" fn bk_global(ctx: *mut JitCtx, slot: u32, out: *mut Value) -> i32 {
    // SAFETY: `out` is a slot of the native frame; `ctx.rt` is valid.
    unsafe {
        match &(*(*ctx).rt).globals.at(slot).value {
            Some(v) => {
                *out = v.clone();
                0
            }
            None => 1,
        }
    }
}

/// Binary operations on strings, by code (`str_op_code`).
const STR_OPS: [BinOp; 7] = [
    BinOp::Add,
    BinOp::Eq,
    BinOp::Ne,
    BinOp::Lt,
    BinOp::Le,
    BinOp::Gt,
    BinOp::Ge,
];

fn str_op_code(op: BinOp) -> Option<i64> {
    STR_OPS.iter().position(|o| *o == op).map(|i| i as i64)
}

/// `a + b` of two strings into the slot `dst` (the interpreter's operation): 0, or 1 if
/// one is not a string (resumed in the interpreter).
extern "C" fn bk_str_add(
    ctx: *mut JitCtx,
    dst: *mut Value,
    a: *const Value,
    b: *const Value,
) -> i32 {
    // SAFETY: `a`, `b` are live values, `dst` a slot of the native frame (possibly the
    // same as `a`: read first); `ctx.rt` is the runtime running this code.
    unsafe {
        let (l, r) = ((*a).clone(), (*b).clone());
        if !matches!((&l, &r), (Value::Str(_), Value::Str(_))) {
            return 1;
        }
        match (*(*ctx).rt).binary(BinOp::Add, l, r, Span::default()) {
            Ok(v) => {
                *dst = v;
                0
            }
            Err(_) => 1,
        }
    }
}

/// Comparison `op` (`STR_OPS`) of two strings: 0 / 1, or 2 if one is not a string.
extern "C" fn bk_str_cmp(ctx: *mut JitCtx, op: i32, a: *const Value, b: *const Value) -> i32 {
    // SAFETY: as for `bk_str_add`.
    unsafe {
        let (l, r) = ((*a).clone(), (*b).clone());
        if !matches!((&l, &r), (Value::Str(_), Value::Str(_))) {
            return 2;
        }
        match (*(*ctx).rt).binary(STR_OPS[op as usize], l, r, Span::default()) {
            Ok(Value::Bool(b)) => b as i32,
            _ => 2,
        }
    }
}

/// Number of characters of a string; -1 if it is not one.
extern "C" fn bk_str_len(s: *const Value) -> i64 {
    // SAFETY: `s` is a live value.
    unsafe {
        match &*s {
            Value::Str(s) => s.chars().count() as i64,
            _ => -1,
        }
    }
}

/// Interpolation (`"#{a} #{b}"`) of `n` native values (`words`, kinds in `kinds`: 0 int,
/// 1 float, 2 bool, 3 string) into the slot `dst`: 0, or 1 if a string is not one.
extern "C" fn bk_concat(dst: *mut Value, words: *const u64, kinds: *const u8, n: u64) -> i32 {
    // SAFETY: `words` and `kinds` hold `n` items written just before the call; the
    // strings are live values; `dst` is a slot of the native frame.
    unsafe {
        let mut out = String::new();
        for k in 0..n as usize {
            let w = *words.add(k);
            match *kinds.add(k) {
                3 => match &*(w as *const Value) {
                    Value::Str(s) => out.push_str(s),
                    _ => return 1,
                },
                c => {
                    from_bits(kind_of_code(c as i32), w).write_scalar(&mut out);
                }
            }
        }
        *dst = Value::Str(Rc::from(out.as_str()));
        0
    }
}

/// Kind of a native value passed to a list helper (`kind_code`).
fn kind_of_code(c: i32) -> Kind {
    match c {
        0 => Kind::Int,
        1 => Kind::Float,
        2 => Kind::Bool,
        _ => Kind::Ref,
    }
}

/// `list << v`: 0, or 1 if `list` is not a list (or is borrowed).
extern "C" fn bk_list_push(list: *const Value, bits: u64, kind: i32) -> i32 {
    // SAFETY: `list` is a live value; for `Kind::Ref`, `bits` is the address of a live
    // value.
    unsafe {
        let Value::List(l) = &*list else {
            return 1;
        };
        let v = from_bits(kind_of_code(kind), bits);
        match l.try_borrow_mut() {
            Ok(mut l) => {
                l.push(v);
                0
            }
            Err(_) => 1,
        }
    }
}

/// `list[i] = v` (negative: from the end): 0, or 1 if not a list or out of bounds (the
/// interpreter then gives the error).
extern "C" fn bk_list_set(list: *const Value, i: i64, bits: u64, kind: i32) -> i32 {
    // SAFETY: as for `bk_list_push`.
    unsafe {
        let Value::List(l) = &*list else {
            return 1;
        };
        let Ok(mut l) = l.try_borrow_mut() else {
            return 1;
        };
        let n = l.len() as i64;
        let pos = if i < 0 { i + n } else { i };
        if !(0..n).contains(&pos) {
            return 1;
        }
        l[pos as usize] = from_bits(kind_of_code(kind), bits);
        0
    }
}

/// Copy of the list `src` into the slot `dst` (a `for` over a list that may change, as
/// the interpreter does); `nil` if it is not a list.
extern "C" fn bk_list_snapshot(dst: *mut Value, src: *const Value) {
    // SAFETY: `dst` is a slot of the native frame, `src` a live value.
    unsafe {
        *dst = match &*src {
            Value::List(l) => Value::list(l.borrow().clone()),
            _ => Value::Nil,
        };
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
    /// Allocated once (`Box::into_raw`), freed by `drop`: native code and resumptions
    /// in the interpreter reach it by its address.
    ctx: *mut JitCtx,
    /// Resumption points (addresses held by native code).
    #[allow(clippy::vec_box)]
    deopt_sites: Vec<Box<DeoptSite>>,
    resume_fn: FuncId,
    errors: Vec<ErrInfo>,
    traces: Vec<TraceInfo>,
    /// Functions already compiled (key: address of the `Proto`).
    compiled: FxHashMap<usize, Compiled>,
    trace_fn: FuncId,
    frem_fn: FuncId,
    field_fn: FuncId,
    fields_fn: FuncId,
    list_fn: FuncId,
    /// `bk_this_fields`, `bk_method_is`.
    this_fns: [FuncId; 2],
    /// `bk_list_push`, `bk_list_set`, `bk_list_snapshot`.
    list_fns: [FuncId; 3],
    /// `bk_str_add`, `bk_str_cmp`, `bk_str_len`, `bk_concat`.
    str_fns: [FuncId; 4],
    /// `bk_call_boxed`, `bk_binary`, `bk_truthy`, `bk_global`, `bk_box_into`.
    boxed_fns: [FuncId; 5],
    /// Calls through the interpreter (addresses held by native code).
    #[allow(clippy::vec_box)]
    boxed_calls: Vec<Box<BoxedCall>>,
    /// `bk_build`, `bk_clone_into`, `bk_move_out`, `bk_drop_slots`.
    mem_fns: [FuncId; 4],
    /// Construction plans (addresses held by native code).
    #[allow(clippy::vec_box)]
    build_plans: Vec<Box<BuildPlan>>,
    /// Types whose layout native code knows (kept alive: their address serves as their
    /// identity).
    known_types: Vec<Rc<crate::objects::TypeInfo>>,
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
    /// Caches of the inlined method calls (addresses held by native code).
    #[allow(clippy::vec_box)]
    method_caches: Vec<Box<MethodCache>>,
    fields_cached_fn: FuncId,
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
    /// `self` in a method (register 0): pointer to the value; its fields are those of
    /// `Plan::this`.
    This,
    /// String: pointer to the value.
    Str,
    /// Any value, result of the interpreter (a call into it, an operation on such a
    /// value): pointer to the value, in a slot of the frame.
    Boxed,
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
        RtType::Named(..) | RtType::Str => Some(RegTy::Value),
        RtType::List(e) if elem_ok(e) => Some(RegTy::Value),
        t => prim(t),
    }
}

/// Native type of the result of `q`: its annotation, or any value (not annotated).
fn ret_jty(anns: &mut Vec<RtType>, q: &Proto) -> JTy {
    match &q.ret {
        Some(t) => value_jty(anns, &t.ty),
        None => JTy::Boxed,
    }
}

/// Native signature of a function (parameters and return), otherwise `None`.
/// For a method (not `init`): `self` first.
fn native_sig(p: &Proto) -> Option<(Vec<RegTy>, RegTy)> {
    let method = p.kind == ProtoKind::Method && !p.name.ends_with(".init");
    let function = matches!(p.kind, ProtoKind::Function | ProtoKind::Lambda);
    if !(function || method) || !p.captures.is_empty() || p.ncells != 0 {
        return None;
    }
    let mut params = Vec::new();
    if method {
        params.push(RegTy::Value);
    }
    for prm in &p.params {
        if prm.has_default || prm.field.is_some() {
            return None;
        }
        params.push(native_ty(&prm.ty.as_ref()?.ty)?);
    }
    // Not annotated: any value (in the caller's slot).
    let ret = match &p.ret {
        Some(t) => native_ty(&t.ty)?,
        None => RegTy::Value,
    };
    Some((params, ret))
}

/// Annotation of parameter `i`.
fn ann_of(q: &Proto, i: usize) -> Option<&RtType> {
    q.params[i].ty.as_ref().map(|t| &*t.ty)
}

/// Same type: a declared type is identified by its global slot (whatever its type
/// arguments: same layout).
fn same_ann(a: &RtType, b: &RtType) -> bool {
    match (a, b) {
        (RtType::Named(x, ..), RtType::Named(y, ..)) => x == y,
        (RtType::List(x), RtType::List(y)) => same_ann(x, y),
        _ => a == b,
    }
}

/// Same type and same type arguments (`Acc(int)` is not `Acc(float)`).
fn same_ann_exact(a: &RtType, b: &RtType) -> bool {
    match (a, b) {
        (RtType::Named(x, _, xs), RtType::Named(y, _, ys)) => {
            x == y
                && xs.len() == ys.len()
                && xs.iter().zip(ys.iter()).all(|(p, q)| same_ann_exact(p, q))
        }
        (RtType::List(x), RtType::List(y)) => same_ann_exact(x, y),
        _ => a == b,
    }
}

/// Index of `t` in the type table (added if needed).
fn intern_ann(anns: &mut Vec<RtType>, t: &RtType) -> u32 {
    match anns.iter().position(|a| same_ann_exact(a, t)) {
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
        RtType::Str => JTy::Str,
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
        JTy::Obj(_) | JTy::List(_) | JTy::Str => Kind::Ref,
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
        (RtType::Str, JTy::Str) => Some(kind),
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
    /// Functions called directly: their global slots.
    callees: Vec<u32>,
    /// Function called by each `Call` / `CallMethod` (key: address of the instruction):
    /// the one in the global, its copy typed for the arguments (`forall` function), or a
    /// method.
    targets: FxHashMap<usize, Target>,
    /// Method: the type of `self` (its fields).
    this: Option<Rc<crate::objects::TypeInfo>>,
    /// Constructed types (`T(…)`).
    ctors: Vec<u32>,
    /// Types of objects and list elements (`JTy::Obj` / `List` / `Iter`).
    anns: Vec<RtType>,
    /// Instructions done by the interpreter (key: address): calls of functions not
    /// compiled, operations on `JTy::Boxed` values.
    boxed: crate::intern::FxHashSet<usize>,
}

/// Type of an element of type `anns[a]`.
fn elem_jty(anns: &[RtType], a: u32) -> JTy {
    match &anns[a as usize] {
        RtType::Named(..) => JTy::Obj(a),
        RtType::Str => JTy::Str,
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
    targets: FxHashMap<usize, Target>,
    this: Option<Rc<crate::objects::TypeInfo>>,
    boxed: crate::intern::FxHashSet<usize>,
}

/// Function compiled and called directly; for a method, the type of `self` it is
/// compiled for (the class that declares it: its subclasses have the same fields first).
#[derive(Clone)]
struct Target {
    proto: Rc<Proto>,
    this: Option<Rc<crate::objects::TypeInfo>>,
    /// Method: the one declared by the class (`proto` may be its typed copy), checked
    /// against the object's when calling.
    declared: Option<Rc<Proto>>,
}

impl Target {
    fn key(&self) -> usize {
        Rc::as_ptr(&self.proto) as usize
    }
}

/// Elements a native list can hold: simple values, objects of a declared type.
fn elem_ok(e: &RtType) -> bool {
    prim(e).is_some() || matches!(e, RtType::Named(..) | RtType::Str)
}

/// Field `idx` of `info` annotated as a list (`@items: [int]`) or a string: its type.
fn field_ref(info: &crate::objects::TypeInfo, idx: u32) -> Option<RtType> {
    match &*info.fields.get(idx as usize)?.ty.as_ref()?.ty {
        t @ RtType::List(e) if elem_ok(e) => Some(t.clone()),
        RtType::Str => Some(RtType::Str),
        _ => None,
    }
}

/// Native type of field `idx` of `info`: annotated `int` / `float` / `bool`, or of a type
/// parameter replaced by one of them in the typed copy `spec` (`@value: T`, `T = int`).
/// Checked on entry of the method.
fn field_jty(
    info: &crate::objects::TypeInfo,
    idx: u32,
    spec: Option<&crate::ir::Spec>,
) -> Option<JTy> {
    let t = info.fields.get(idx as usize)?.ty.as_ref()?;
    if let Some(s) = spec
        && let Some(r) = s.param_ty(&t.expr)
    {
        return prim(r).map(jty);
    }
    prim(&t.ty).map(jty)
}

impl Analysis<'_> {
    /// An instruction not compiled natively, done by the interpreter: calls (a function
    /// not compiled, a built-in, any method) and operations on their results
    /// (`JTy::Boxed`). `None`: not even that.
    fn boxed(&mut self, ins: &Instr, cur: &[Lat]) -> Option<Option<JTy>> {
        use Instr::*;
        let g = |r: Reg| cur[r as usize].get();
        // Native values rebuilt as interpreter values.
        let value = |t: JTy| {
            matches!(
                t,
                JTy::Int
                    | JTy::Float
                    | JTy::Bool
                    | JTy::Obj(_)
                    | JTy::List(_)
                    | JTy::Str
                    | JTy::This
                    | JTy::Boxed
                    | JTy::Nil
            )
        };
        let word = |t: JTy| value(t) || matches!(t, JTy::Callee(_) | JTy::Ctor(_));
        let p = self.p;
        let t = match ins {
            Call { f, site, .. } => {
                let site = &p.calls[*site as usize];
                if site.block || !site.named.is_empty() || !word(g(*f)?) {
                    return None;
                }
                for k in 0..site.npos {
                    if !word(g(site.start + k)?) {
                        return None;
                    }
                }
                JTy::Boxed
            }
            CallMethod { recv, site, .. } => {
                let site = &p.calls[*site as usize];
                if site.block || !site.named.is_empty() || !value(g(*recv)?) {
                    return None;
                }
                for k in 0..site.npos {
                    if !word(g(site.start + k)?) {
                        return None;
                    }
                }
                JTy::Boxed
            }
            Bin(op, _, a, b)
                if !matches!(op, BinOp::Push | BinOp::And | BinOp::Or)
                    && (g(*a)? == JTy::Boxed || g(*b)? == JTy::Boxed)
                    && value(g(*a)?)
                    && value(g(*b)?) =>
            {
                JTy::Boxed
            }
            Guard(_, a, t) if g(*a)? == JTy::Boxed => jty(*t),
            Not(_, a) if g(*a)? == JTy::Boxed => JTy::Bool,
            // A global: a function called (its value read when calling), or a value.
            GetGlobal(_, slot) => match &self.globals.at(*slot).value {
                Some(Value::Function(_)) => return Some(Some(JTy::Callee(*slot))),
                Some(Value::Type(_)) => return Some(Some(JTy::Ctor(*slot))),
                _ => JTy::Boxed,
            },
            _ => return None,
        };
        self.boxed.insert(ins as *const Instr as usize);
        Some(Some(t))
    }

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
                JTy::Int
                    | JTy::Float
                    | JTy::Bool
                    | JTy::Obj(_)
                    | JTy::List(_)
                    | JTy::Str
                    | JTy::Boxed
            )
        };
        let numeric = |t: JTy| matches!(t, JTy::Int | JTy::Float);
        let p = self.p;
        let this = self.this.clone();
        Some(Some(match ins {
            Int(..) => JTy::Int,
            Float(..) => JTy::Float,
            Bool(..) => JTy::Bool,
            Nil(_) => JTy::Nil,
            // (`nil`: the value of an `if` without `else`, never read.)
            Move(_, a) => Some(g(*a)?).filter(|t| value(*t) || *t == JTy::Nil)?,
            Guard(_, a, _) => {
                Some(g(*a)?).filter(|t| matches!(t, JTy::Int | JTy::Float | JTy::Bool))?
            }
            FieldAs(_, a, _, t) => match g(*a)? {
                JTy::Obj(_) => jty(*t),
                _ => return None,
            },
            // `@x` in a method: an annotated field of a simple type.
            FieldAt(_, a, idx) => match (g(*a)?, &this) {
                (JTy::This, Some(info)) => match field_ref(info, *idx) {
                    Some(t) => value_jty(&mut self.anns, &t),
                    None => field_jty(info, *idx, p.spec.as_deref())?,
                },
                _ => return None,
            },
            SetFieldAt { obj, idx, src, .. } => {
                let (JTy::This, Some(info)) = (g(*obj)?, &this) else {
                    return None;
                };
                if let Some(t) = field_ref(info, *idx) {
                    let ok = match t {
                        RtType::Str => g(*src)? == JTy::Str,
                        _ => matches!(g(*src)?, JTy::List(_)),
                    };
                    return ok.then_some(None);
                }
                let t = field_jty(info, *idx, p.spec.as_deref())?;
                let s = g(*src)?;
                if !(s == t || (t == JTy::Float && s == JTy::Int)) {
                    return None;
                }
                return Some(None);
            }
            IterNew { src, .. } => match g(*src)? {
                JTy::List(a) if elem_ok(&self.anns[a as usize]) => JTy::Iter(a),
                _ => return None,
            },
            Index(_, a, i) => match (g(*a)?, g(*i)?) {
                (JTy::List(e), JTy::Int) if elem_ok(&self.anns[e as usize]) => {
                    elem_jty(&self.anns, e)
                }
                _ => return None,
            },
            // `xs << v`, `xs[i] = v`: any native value (the elements are checked when read).
            Bin(BinOp::Push, _, a, b) => match (g(*a)?, g(*b)?) {
                (
                    JTy::List(e),
                    JTy::Int | JTy::Float | JTy::Bool | JTy::Obj(_) | JTy::List(_),
                ) => JTy::List(e),
                _ => return None,
            },
            SetIndex(a, i, v) => match (g(*a)?, g(*i)?, g(*v)?) {
                (
                    JTy::List(_),
                    JTy::Int,
                    JTy::Int | JTy::Float | JTy::Bool | JTy::Obj(_) | JTy::List(_),
                ) => return Some(None),
                _ => return None,
            },
            // `let out: [int] = []`, an annotation already satisfied.
            // (A list literal's shape converts nothing when it has no record.)
            Coerce {
                src, ty, shape, ..
            } if shape.is_none_or(|s| {
                matches!(&*p.shapes[s as usize], crate::ir::Shape::List(items) if items.iter().all(Option::is_none))
            }) =>
            {
                let target = p.types[*ty as usize].ty.clone();
                match (&*target, g(*src)?) {
                    (t, s) if prim(t).is_some() => {
                        let want = jty(prim(t).unwrap());
                        if !(s == want || (want == JTy::Float && s == JTy::Int)) {
                            return None;
                        }
                        want
                    }
                    (RtType::List(e), JTy::List(a))
                        if elem_ok(e)
                            && (self.anns[a as usize] == RtType::Any
                                || same_ann_exact(e, &self.anns[a as usize])) =>
                    {
                        JTy::List(intern_ann(&mut self.anns, e))
                    }
                    (t @ RtType::Named(..), JTy::Obj(a)) if same_ann(t, &self.anns[a as usize]) => {
                        JTy::Obj(a)
                    }
                    _ => return None,
                }
            }
            // Declared method of an object of known type (`self`, or a parameter): called
            // directly, if not redefined by the object's class (checked when calling).
            CallMethod {
                recv, name, site, ..
            } if !matches!(g(*recv)?, JTy::List(_) | JTy::Str) => {
                let site = &p.calls[*site as usize];
                let info = match g(*recv)? {
                    JTy::This => this?,
                    JTy::Obj(a) => match &self.anns[a as usize] {
                        RtType::Named(slot, ..) => match &self.globals.at(*slot).value {
                            Some(Value::Type(info)) => info.clone(),
                            _ => return None,
                        },
                        _ => return None,
                    },
                    _ => return None,
                };
                // Type arguments of the receiver's annotation (`a: Acc(int)`), simple ones.
                let recv_args: Option<Vec<RtType>> = match g(*recv)? {
                    JTy::Obj(a) => match &self.anns[a as usize] {
                        RtType::Named(_, _, xs)
                            if !xs.is_empty() && xs.iter().all(|x| prim(x).is_some()) =>
                        {
                            Some(xs.to_vec())
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let declared = info.methods.get(name)?.clone();
                // Method of a generic class called on `self` in a typed copy: the copy for
                // the same type arguments.
                let owner = crate::objects::method_owner(&info, *name, &declared);
                let m = match (&p.spec, recv_args) {
                    (Some(s), _)
                        if !declared.class_params.is_empty()
                            && declared.class_params == p.class_params
                            && g(*recv)? == JTy::This =>
                    {
                        declared.specialize_as(&declared.class_params, &s.tys)
                    }
                    // Declared by the annotated class itself: its arguments are those of the
                    // method's type parameters (checked on entry of the copy).
                    (_, Some(args))
                        if !declared.class_params.is_empty()
                            && Rc::ptr_eq(&owner, &info)
                            && args.len() == declared.class_params.len() =>
                    {
                        declared.specialize_as(&declared.class_params, &args)
                    }
                    _ => declared.clone(),
                };
                let (cparams, _) = native_sig(&m)?;
                if m.kind != ProtoKind::Method
                    || site.block
                    || !site.named.is_empty()
                    || site.npos as usize + 1 != cparams.len()
                {
                    return None;
                }
                for (k, pt) in cparams[1..].iter().enumerate() {
                    let at = g(site.start + k as u32)?;
                    let ok = match (pt, at) {
                        (RegTy::Float, JTy::Int | JTy::Float) => true,
                        (RegTy::Value, JTy::Obj(_) | JTy::List(_) | JTy::Str) => {
                            kind_for(&self.anns, at, ann_of(&m, k)) == Some(Kind::Ref)
                        }
                        (t, a) => *t != RegTy::Value && jty(*t) == a,
                    };
                    if !ok {
                        return None;
                    }
                }
                let ret = ret_jty(&mut self.anns, &m);
                self.targets.insert(
                    ins as *const Instr as usize,
                    Target {
                        proto: m,
                        this: Some(owner),
                        declared: Some(declared),
                    },
                );
                ret
            }
            CallMethod {
                recv, name, site, ..
            } => {
                let site = &p.calls[*site as usize];
                let ok = *name == self.len
                    && matches!(g(*recv)?, JTy::List(_) | JTy::Str)
                    && site.npos == 0
                    && site.named.is_empty()
                    && !site.block;
                if !ok {
                    return None;
                }
                JTy::Int
            }
            // List literal: elements all of the same type.
            // String literal: the constant itself.
            Const(_, c) => match &p.consts[*c as usize] {
                Value::Str(_) => JTy::Str,
                _ => return None,
            },
            // Strings: `+`, comparisons, interpolation of strings and simple values.
            Bin(op, _, a, b) if g(*a)? == JTy::Str && g(*b)? == JTy::Str => match op {
                BinOp::Add => JTy::Str,
                op if str_op_code(*op).is_some() => JTy::Bool,
                _ => return None,
            },
            Concat(_, start, n) => {
                for k in 0..*n {
                    if !matches!(g(*start + k)?, JTy::Int | JTy::Float | JTy::Bool | JTy::Str) {
                        return None;
                    }
                }
                JTy::Str
            }
            // `[]`: elements of unknown type (an annotation gives them).
            List(_, _, 0) => JTy::List(intern_ann(&mut self.anns, &RtType::Any)),
            List(_, start, n) => {
                let first = g(*start)?;
                if (0..*n).any(|k| g(*start + k) != Some(first)) {
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
                    // A `forall` function: typed for the arguments at the call.
                    let generic = global_proto(self.globals, *slot).is_some_and(|q| q.is_generic());
                    if !generic {
                        self.sig_of(*slot)?;
                    }
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
                        let mut callee = global_proto(self.globals, slot)?;
                        // Parameters not annotated: its copy for the types of the arguments.
                        if callee.untyped() {
                            let anns = &self.anns;
                            let tys = (0..callee.params.len())
                                .map(|k| match &callee.params[k].ty {
                                    Some(t) => Some((*t.ty).clone()),
                                    None if k < site.npos as usize => {
                                        Some(match g(site.start + k as u32)? {
                                            JTy::Int => RtType::Int,
                                            JTy::Float => RtType::Float,
                                            JTy::Bool => RtType::Bool,
                                            JTy::Str => RtType::Str,
                                            JTy::Obj(a) => anns[a as usize].clone(),
                                            JTy::List(a) if elem_ok(&anns[a as usize]) => {
                                                RtType::List(Box::new(anns[a as usize].clone()))
                                            }
                                            _ => return None,
                                        })
                                    }
                                    None => None,
                                })
                                .collect::<Option<Vec<_>>>()?;
                            callee = callee.specialize_params(&tys);
                        }
                        if callee.is_generic() {
                            let anns = &self.anns;
                            let arg = |k: usize| {
                                if k >= site.npos as usize {
                                    return None;
                                }
                                Some(match g(site.start + k as u32)? {
                                    JTy::Int => RtType::Int,
                                    JTy::Float => RtType::Float,
                                    JTy::Bool => RtType::Bool,
                                    JTy::Obj(a) => anns[a as usize].clone(),
                                    JTy::List(a) => {
                                        RtType::List(Box::new(anns[a as usize].clone()))
                                    }
                                    _ => return None,
                                })
                            };
                            let tys = callee.type_args(&arg)?;
                            callee = callee.specialize(&tys);
                        }
                        let (cparams, _) = native_sig(&callee)?;
                        if site.block
                            || !site.named.is_empty()
                            || site.npos as usize != cparams.len()
                        {
                            return None;
                        }
                        self.targets.insert(
                            ins as *const Instr as usize,
                            Target {
                                proto: callee.clone(),
                                this: None,
                                declared: None,
                            },
                        );
                        for (k, pt) in cparams.iter().enumerate() {
                            let at = g(site.start + k as u32)?;
                            let ok = match (pt, at) {
                                (RegTy::Float, JTy::Int | JTy::Float) => true,
                                // Object or list passed as is: same type, already checked.
                                (RegTy::Value, JTy::Obj(_) | JTy::List(_) | JTy::Str) => {
                                    kind_for(&self.anns, at, ann_of(&callee, k)) == Some(Kind::Ref)
                                }
                                (t, a) => *t != RegTy::Value && jty(*t) == a,
                            };
                            if !ok {
                                return None;
                            }
                        }
                        ret_jty(&mut self.anns, &callee)
                    }
                    JTy::Ctor(slot) => {
                        let info = global_type(self.globals, slot)?;
                        let tys: Vec<Option<JTy>> = cur.iter().map(|l| l.get()).collect();
                        ctor_plan(&info, site, &tys, &self.anns)?;
                        let t = RtType::Named(slot, crate::intern::Symbol(u32::MAX), Rc::from([]));
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
fn analyze(
    p: &Proto,
    this: Option<Rc<crate::objects::TypeInfo>>,
    globals: &Globals,
    len: crate::intern::Symbol,
) -> Option<Plan> {
    // `rescue` / `ensure`: left to the interpreter.
    if !p.handlers.is_empty() {
        return None;
    }
    let (_, ret) = native_sig(p)?;
    let method = p.kind == ProtoKind::Method;
    if method != this.is_some() {
        return None;
    }
    let deopted: crate::intern::FxHashSet<usize> = p
        .deopted
        .borrow()
        .iter()
        .filter_map(|&(b, k)| p.blocks.get(b as usize)?.instrs.get(k as usize))
        .map(|i| i as *const Instr as usize)
        .collect();
    let mut an = Analysis {
        p,
        globals,
        len,
        anns: Vec::new(),
        callees: Vec::new(),
        ctors: Vec::new(),
        targets: FxHashMap::default(),
        boxed: Default::default(),
        this: this.clone(),
    };
    // Typed IR registers: their type does not change.
    let fixed: Vec<Option<JTy>> = p
        .regs
        .iter()
        .map(|t| (*t != RegTy::Value).then(|| jty(*t)))
        .collect();
    let mut start: Vec<Lat> = fixed.iter().map(|t| t.map_or(Lat::Undef, Lat::T)).collect();
    if method {
        start[0] = Lat::T(JTy::This);
    }
    for prm in &p.params {
        start[prm.reg as usize] = Lat::T(value_jty(&mut an.anns, &prm.ty.as_ref()?.ty));
    }
    let ret_jty = ret_jty(&mut an.anns, p);
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
                // Resumed in the interpreter there before: done by it.
                let forced = deopted.contains(&(ins as *const Instr as usize));
                let native = if forced { None } else { an.instr(ins, &cur) };
                let t = match native {
                    Some(t) => t,
                    // Not compiled natively: done by the interpreter, if possible.
                    None => an.boxed(ins, &cur)?,
                };
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
                Term::Branch { cond, .. } => matches!(
                    g(*cond),
                    Some(JTy::Int | JTy::Float | JTy::Bool | JTy::Nil | JTy::Boxed)
                ),
                Term::IfInt { a, b: y, .. } => g(*a) == Some(JTy::Int) && g(*y) == Some(JTy::Int),
                Term::IfIntK { a, .. } => g(*a) == Some(JTy::Int),
                Term::Return { reg, .. } => match ret {
                    RegTy::Float => num(*reg),
                    // Not annotated: any value (boxed when returned).
                    _ if ret_jty == JTy::Boxed => matches!(
                        g(*reg),
                        Some(
                            JTy::Int
                                | JTy::Float
                                | JTy::Bool
                                | JTy::Nil
                                | JTy::Obj(_)
                                | JTy::List(_)
                                | JTy::Str
                                | JTy::This
                                | JTy::Boxed
                        )
                    ),
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
                Term::Reraise(_) => false,
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
        boxed: an.boxed,
        anns: an.anns,
        targets: an.targets,
        this,
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
        FieldAt(d, a, _) => (vec![*a], Some(*d)),
        SetFieldAt { obj, src, .. } => (vec![*obj, *src], None),
        SetIndex(a, i, v) => (vec![*a, *i, *v], None),
        Const(d, _) => (vec![], Some(*d)),
        Concat(d, start, n) => ((*start..*start + *n).collect(), Some(*d)),
        Coerce { dst, src, .. } => (vec![*src], Some(*dst)),
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
        builder.symbol("bk_this_fields", bk_this_fields as *const u8);
        builder.symbol("bk_method_is", bk_method_is as *const u8);
        builder.symbol("bk_resume", bk_resume as *const u8);
        builder.symbol("bk_list_push", bk_list_push as *const u8);
        builder.symbol("bk_fields_cached", bk_fields_cached as *const u8);
        builder.symbol("bk_str_add", bk_str_add as *const u8);
        builder.symbol("bk_call_boxed", bk_call_boxed as *const u8);
        builder.symbol("bk_binary", bk_binary as *const u8);
        builder.symbol("bk_truthy", bk_truthy as *const u8);
        builder.symbol("bk_global", bk_global as *const u8);
        builder.symbol("bk_box_into", bk_box_into as *const u8);
        builder.symbol("bk_str_cmp", bk_str_cmp as *const u8);
        builder.symbol("bk_str_len", bk_str_len as *const u8);
        builder.symbol("bk_concat", bk_concat as *const u8);
        builder.symbol("bk_list_set", bk_list_set as *const u8);
        builder.symbol("bk_list_snapshot", bk_list_snapshot as *const u8);
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
        let this_fields_fn = module
            .declare_function("bk_this_fields", Linkage::Import, &sig)
            .ok()?;
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(types::I32));
        sig.params.push(AbiParam::new(ptr));
        sig.returns.push(AbiParam::new(types::I32));
        let method_fn = module
            .declare_function("bk_method_is", Linkage::Import, &sig)
            .ok()?;
        let mut sig = module.make_signature();
        for _ in 0..4 {
            sig.params.push(AbiParam::new(ptr));
        }
        sig.returns.push(AbiParam::new(types::I32));
        let resume_fn = module
            .declare_function("bk_resume", Linkage::Import, &sig)
            .ok()?;
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I32));
        sig.returns.push(AbiParam::new(types::I32));
        let push_fn = module
            .declare_function("bk_list_push", Linkage::Import, &sig)
            .ok()?;
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I32));
        sig.returns.push(AbiParam::new(types::I32));
        let set_fn = module
            .declare_function("bk_list_set", Linkage::Import, &sig)
            .ok()?;
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        let snapshot_fn = module
            .declare_function("bk_list_snapshot", Linkage::Import, &sig)
            .ok()?;
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.returns.push(AbiParam::new(ptr));
        let fields_cached_fn = module
            .declare_function("bk_fields_cached", Linkage::Import, &sig)
            .ok()?;
        let mut str_fns = Vec::new();
        for (name, params, ret) in [
            ("bk_str_add", &[ptr, ptr, ptr, ptr][..], types::I32),
            ("bk_str_cmp", &[ptr, types::I32, ptr, ptr][..], types::I32),
            ("bk_str_len", &[ptr][..], types::I64),
            ("bk_concat", &[ptr, ptr, ptr, types::I64][..], types::I32),
        ] {
            let mut sig = module.make_signature();
            for t in params {
                sig.params.push(AbiParam::new(*t));
            }
            sig.returns.push(AbiParam::new(ret));
            str_fns.push(module.declare_function(name, Linkage::Import, &sig).ok()?);
        }
        let str_fns: [FuncId; 4] = str_fns.try_into().ok()?;
        let mut boxed_fns = Vec::new();
        for (name, params, ret) in [
            ("bk_call_boxed", &[ptr, ptr, ptr, ptr][..], types::I32),
            (
                "bk_binary",
                &[
                    ptr,
                    types::I32,
                    types::I64,
                    types::I32,
                    types::I64,
                    types::I32,
                    ptr,
                    types::I64,
                ][..],
                types::I32,
            ),
            ("bk_truthy", &[ptr][..], types::I8),
            ("bk_global", &[ptr, types::I32, ptr][..], types::I32),
            (
                "bk_box_into",
                &[ptr, types::I64, types::I32][..],
                types::INVALID,
            ),
        ] {
            let mut sig = module.make_signature();
            for t in params {
                sig.params.push(AbiParam::new(*t));
            }
            if ret != types::INVALID {
                sig.returns.push(AbiParam::new(ret));
            }
            boxed_fns.push(module.declare_function(name, Linkage::Import, &sig).ok()?);
        }
        let boxed_fns: [FuncId; 5] = boxed_fns.try_into().ok()?;
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
            ctx: Box::into_raw(Box::new(JitCtx {
                depth: 0,
                max_depth: 0,
                err: -1,
                traces: Vec::new(),
                rt: std::ptr::null_mut(),
                pending: None,
                returned: None,
            })),
            deopt_sites: Vec::new(),
            resume_fn,
            errors: Vec::new(),
            traces: Vec::new(),
            compiled: FxHashMap::default(),
            trace_fn,
            frem_fn,
            field_fn,
            fields_fn,
            list_fn,
            this_fns: [this_fields_fn, method_fn],
            list_fns: [push_fn, set_fn, snapshot_fn],
            str_fns,
            boxed_fns,
            boxed_calls: Vec::new(),
            mem_fns,
            build_plans: Vec::new(),
            known_types: Vec::new(),
            len_sym: crate::intern::Symbol(u32::MAX),
            dyn_fns: None,
            fast_fns: Vec::new(),
            field_caches: Vec::new(),
            method_caches: Vec::new(),
            fields_cached_fn,
            next: 0,
        })
    }

    /// Compiles `root` (and the functions it calls) if possible; for a method, `this` is
    /// the class that declares it. Updates the JIT state of every function encountered.
    pub fn compile(
        &mut self,
        root: &Rc<Proto>,
        this: Option<Rc<crate::objects::TypeInfo>>,
        globals: &mut Globals,
    ) {
        // 1. Functions reachable through direct calls, and their analysis.
        let mut order: Vec<Rc<Proto>> = Vec::new();
        let mut plans: FxHashMap<usize, Option<Plan>> = FxHashMap::default();
        let mut todo = vec![Target {
            proto: root.clone(),
            this,
            declared: None,
        }];
        while let Some(Target { proto: p, this, .. }) = todo.pop() {
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
            let plan = analyze(&p, this, globals, self.len_sym);
            if let Some(plan) = &plan {
                todo.extend(plan.targets.values().cloned());
            }
            plans.insert(key, plan);
            order.push(p);
        }
        // 2. A function whose callee is not compilable stays compilable: that call goes
        // through the interpreter.
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
                    && let RtType::Named(slot, ..) = &*ann.ty
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
        for q in plan.targets.values() {
            let key = q.key();
            if callee_refs.contains_key(&key) {
                continue;
            }
            let cid = self.compiled.get(&key).ok_or(())?.id;
            let r = self.module.declare_func_in_func(cid, &mut ctx.func);
            callee_refs.insert(key, (r, q.proto.clone()));
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
        let [this_fields_fn, method_fn] = self.this_fns;
        let [push_fn, set_fn, snapshot_fn] = self.list_fns;
        let [call_boxed_ref, binary_ref, truthy_ref, global_ref, box_ref] = self
            .boxed_fns
            .map(|f| self.module.declare_func_in_func(f, &mut ctx.func));
        let [str_add_ref, str_cmp_ref, str_len_ref, concat_ref] = self
            .str_fns
            .map(|f| self.module.declare_func_in_func(f, &mut ctx.func));
        let push_ref = self.module.declare_func_in_func(push_fn, &mut ctx.func);
        let set_ref = self.module.declare_func_in_func(set_fn, &mut ctx.func);
        let snapshot_ref = self.module.declare_func_in_func(snapshot_fn, &mut ctx.func);
        let fields_cached_ref = self
            .module
            .declare_func_in_func(self.fields_cached_fn, &mut ctx.func);
        let this_fields_ref = self
            .module
            .declare_func_in_func(this_fields_fn, &mut ctx.func);
        let method_ref = self.module.declare_func_in_func(method_fn, &mut ctx.func);
        // Method: its native parameters start with `self`.
        let base = 1 + plan.this.is_some() as usize;
        if let Some(info) = &plan.this
            && !self.known_types.iter().any(|t| Rc::ptr_eq(t, info))
        {
            self.known_types.push(info.clone());
        }
        // Object parameters whose fields are read: their type is known now (global slot of the
        // annotation), hence the offset of every field.
        let mut obj_types: Vec<Option<Rc<crate::objects::TypeInfo>>> = vec![None; plan.anns.len()];
        for (a, ann) in plan.anns.iter().enumerate() {
            if let RtType::Named(slot, ..) = ann
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
        let is_ref = |t: Option<JTy>| {
            matches!(
                t,
                Some(JTy::Obj(_) | JTy::List(_) | JTy::Iter(_) | JTy::Str | JTy::Boxed)
            )
        };
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
        // Calls of small functions: inlined (their analysis gives the types of their
        // registers).
        let mut inlines: FxHashMap<usize, (Rc<Proto>, Plan)> = FxHashMap::default();
        for (k, t) in &plan.targets {
            if inlinable(&t.proto, p)
                && let Some(qp) = analyze(&t.proto, t.this.clone(), globals, self.len_sym)
            {
                inlines.insert(*k, (t.proto.clone(), qp));
            }
        }
        // Lists modified here, or possibly by a function called (not inlined: an inlined
        // body modifies nothing).
        let may_mutate = p.blocks.iter().flat_map(|b| &b.instrs).any(|i| {
            let called = matches!(i, Instr::Call { .. } | Instr::CallMethod { .. })
                && !inlines.contains_key(&(i as *const Instr as usize));
            called
                || matches!(
                    i,
                    Instr::Bin(BinOp::Push, ..) | Instr::SetIndex(..) | Instr::SetFieldAt { .. }
                )
        });
        let mut written = vec![false; nregs];
        loop {
            let mut changed = false;
            for b in &p.blocks {
                for ins in &b.instrs {
                    let (w, from_borrowed) = match ins {
                        // A string literal: the function's constant, alive.
                        Instr::Const(d, _) => (Some(*d), true),
                        Instr::Move(d, a)
                        | Instr::Coerce { dst: d, src: a, .. }
                        | Instr::Bin(BinOp::Push, d, a, _) => (Some(*d), borrowed[*a as usize]),
                        // A list may change during the call: its elements are copied, a
                        // `for` goes over a copy of it (as in the interpreter).
                        Instr::Index(d, a, _) => (Some(*d), borrowed[*a as usize] && !may_mutate),
                        Instr::IterNew { dst, src, .. } => {
                            (Some(*dst), borrowed[*src as usize] && !may_mutate)
                        }
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
                JTy::Int | JTy::Obj(_) | JTy::List(_) | JTy::This | JTy::Str | JTy::Boxed => {
                    types::I64
                }
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
        let out_param =
            (ret == RegTy::Value).then(|| bx.block_params(entry)[base + p.params.len()]);
        // Resumption points: the registers known there are written into `dbuf`, then
        // `bk_resume` runs the rest of the function in the interpreter; its result is
        // this function's.
        let dbuf = bx.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            ((nregs * 3).max(1) * 8) as u32,
            3,
        ));
        let dout = bx.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            8,
            3,
        ));
        let resume_ref = self.module.declare_func_in_func(self.resume_fn, bx.func);
        let live = liveness(p);
        // Checks on entry (until the first block): see `DeoptSite::entry`.
        let mut in_entry = true;
        macro_rules! deopt {
            ($bx:expr, $cond:expr, $types:expr, $b:expr, $k:expr) => {
                deopt!($bx, $cond, $types, $b, $k, None)
            };
            ($bx:expr, $cond:expr, $types:expr, $b:expr, $k:expr, $returned:expr) => {{
                let returned: Option<Reg> = $returned;
                let d = $bx.create_block();
                let cont = $bx.create_block();
                $bx.ins().brif($cond, d, &[], cont, &[]);
                $bx.switch_to_block(d);
                let buf = $bx.ins().stack_addr(types::I64, dbuf, 0);
                let mut saved = Vec::new();
                let mut off = 0u32;
                let at = &live[$b][$k];
                for (r, t) in $types.iter().enumerate() {
                    let (r, Some(t)) = (r as Reg, *t) else {
                        continue;
                    };
                    if returned == Some(r) || !at[r as usize] {
                        continue;
                    }
                    let kind = match t {
                        JTy::Int
                        | JTy::Float
                        | JTy::Bool
                        | JTy::Obj(_)
                        | JTy::List(_)
                        | JTy::This
                        | JTy::Str
                        | JTy::Boxed => {
                            let Some(&v) = vars.get(&(r, t)) else {
                                continue;
                            };
                            let v = $bx.use_var(v);
                            let bits = match t {
                                JTy::Float => $bx.ins().bitcast(types::I64, MemFlagsData::new(), v),
                                JTy::Bool => $bx.ins().uextend(types::I64, v),
                                _ => v,
                            };
                            $bx.ins()
                                .store(MemFlagsData::trusted(), bits, buf, (off * 8) as i32);
                            match t {
                                JTy::Int => DKind::Int,
                                JTy::Float => DKind::Float,
                                JTy::Bool => DKind::Bool,
                                _ => DKind::Ref,
                            }
                        }
                        JTy::Iter(_) => {
                            let Some(iv) = iter_vars.get(&r) else {
                                continue;
                            };
                            for (j, v) in iv.iter().enumerate() {
                                let x = $bx.use_var(*v);
                                $bx.ins().store(
                                    MemFlagsData::trusted(),
                                    x,
                                    buf,
                                    ((off + j as u32) * 8) as i32,
                                );
                            }
                            DKind::Iter
                        }
                        JTy::Callee(s) | JTy::Ctor(s) => DKind::Global(s),
                        JTy::Nil => DKind::Nil,
                    };
                    saved.push((r, kind, off));
                    off += match kind {
                        DKind::Iter => 3,
                        DKind::Global(_) | DKind::Nil => 0,
                        _ => 1,
                    };
                }
                if let Some(r) = returned {
                    saved.push((r, DKind::Returned, 0));
                }
                self.deopt_sites.push(Box::new(DeoptSite {
                    proto: p.clone(),
                    block: $b as u32,
                    instr: $k as u32,
                    regs: saved,
                    ret,
                    entry: in_entry,
                }));
                let site = &**self.deopt_sites.last().unwrap() as *const DeoptSite;
                let sp = $bx.ins().iconst(types::I64, site as i64);
                let outp = match out_param {
                    Some(o) => o,
                    None => $bx.ins().stack_addr(types::I64, dout, 0),
                };
                let call = $bx.ins().call(resume_ref, &[ctxv, sp, buf, outp]);
                let status = $bx.inst_results(call)[0];
                if nslots > 0 {
                    let base = $bx.ins().stack_addr(types::I64, vals, 0);
                    let n = $bx.ins().iconst(types::I64, nslots as i64);
                    $bx.ins().call(drop_ref, &[base, n]);
                }
                let v = match ret {
                    RegTy::Float => $bx.ins().stack_load(types::I64, types::F64, dout, 0),
                    RegTy::Bool => $bx.ins().stack_load(types::I64, types::I8, dout, 0),
                    RegTy::Int => $bx.ins().stack_load(types::I64, types::I64, dout, 0),
                    RegTy::Value => $bx.ins().iconst(types::I64, 0),
                };
                $bx.ins().return_(&[status, v]);
                $bx.switch_to_block(cont);
            }};
        }
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
            let a = bx.block_params(entry)[k + base];
            let t = plan.entry[0][prm.reg as usize].get().ok_or(())?;
            bx.def_var(vars[&(prm.reg, t)], a);
        }
        // Resumption on entry: the parameters (and `self`) only.
        let entry_types: Vec<Option<JTy>> = (0..nregs)
            .map(|r| {
                let param =
                    p.params.iter().any(|q| q.reg as usize == r) || (r == 0 && plan.this.is_some());
                param.then(|| plan.entry[0][r].get()).flatten()
            })
            .collect();
        // `self`: its fields (address checked on entry; the type of a field on every
        // access: a method called may change it, a field of type `T` may hold anything).
        let mut this_fp = None;
        if let Some(info) = &plan.this {
            let obj = bx.block_params(entry)[1];
            if let Some(v) = vars.get(&(0, JTy::This)) {
                bx.def_var(*v, obj);
            }
            let used: Vec<u32> = p
                .blocks
                .iter()
                .flat_map(|b| &b.instrs)
                .filter_map(|i| match i {
                    Instr::FieldAt(_, 0, idx) => Some(*idx),
                    Instr::SetFieldAt { obj: 0, idx, .. } => Some(*idx),
                    _ => None,
                })
                .collect();
            if !used.is_empty() {
                let ip = bx.ins().iconst(types::I64, Rc::as_ptr(info) as i64);
                let call = bx.ins().call(this_fields_ref, &[obj, ip]);
                let fp = bx.inst_results(call)[0];
                // Null address: resumed in the interpreter before reading the fields.
                let null = bx.ins().icmp_imm_s(IntCC::Equal, fp, 0);
                deopt!(bx, null, entry_types, 0, 0);
                this_fp = Some(fp);
            }
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
            deopt!(bx, null, entry_types, 0, 0);
        }
        bx.ins().jump(start, &[]);
        bx.switch_to_block(start);
        bx.ins().jump(blocks[0], &[]);
        in_entry = false;

        // Current types (before the instruction) and type written by the instruction.
        let mut cur: Vec<Option<JTy>>;
        let mut dty: Option<JTy>;
        // Position of the instruction being generated (resumption points).
        let (mut site_b, mut site_k): (usize, usize);
        macro_rules! bail {
            ($bx:expr, $cond:expr) => {
                deopt!($bx, $cond, cur, site_b, site_k)
            };
        }
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
                bail!($bx, status);
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
        // Error in inlined code: its frame (`tid`) added to the call stack.
        macro_rules! inline_err {
            ($bx:expr, $cond:expr, $kind:expr, $span:expr, $tid:expr) => {{
                err_sites.push(($kind, $span));
                let eid = err_base + (err_sites.len() - 1) as i64;
                let e = $bx.create_block();
                let cont = $bx.create_block();
                $bx.ins().brif($cond, e, &[], cont, &[]);
                $bx.switch_to_block(e);
                let t = $bx.ins().iconst(types::I32, $tid);
                $bx.ins().call(trace_ref, &[ctxv, t]);
                let id = $bx.ins().iconst(types::I64, eid);
                $bx.ins().jump(fail, &[id.into()]);
                $bx.switch_to_block(cont);
            }};
        }
        // A native value as a word for the interpreter (`word_value`).
        macro_rules! boxed_word {
            ($bx:expr, $r:expr) => {{
                match tyof!($r) {
                    Some(JTy::Callee(s) | JTy::Ctor(s)) => {
                        ($bx.ins().iconst(types::I64, 0), DKind::Global(s))
                    }
                    Some(JTy::Nil) | None => ($bx.ins().iconst(types::I64, 0), DKind::Nil),
                    Some(JTy::Int) => ($bx.use_var(uvar!($r)), DKind::Int),
                    Some(JTy::Float) => {
                        let v = $bx.use_var(uvar!($r));
                        (
                            $bx.ins().bitcast(types::I64, MemFlagsData::new(), v),
                            DKind::Float,
                        )
                    }
                    Some(JTy::Bool) => {
                        let v = $bx.use_var(uvar!($r));
                        ($bx.ins().uextend(types::I64, v), DKind::Bool)
                    }
                    Some(_) => ($bx.use_var(uvar!($r)), DKind::Ref),
                }
            }};
        }
        // A native value as (bits, kind) for the list helpers.
        macro_rules! native_arg {
            ($bx:expr, $r:expr) => {{
                let v = $bx.use_var(uvar!($r));
                match tyof!($r) {
                    Some(JTy::Int) => (v, 0i64),
                    Some(JTy::Float) => ($bx.ins().bitcast(types::I64, MemFlagsData::new(), v), 1),
                    Some(JTy::Bool) => ($bx.ins().uextend(types::I64, v), 2),
                    _ => (v, 3),
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
                } else if t == JTy::Str {
                    let tag = $bx.ins().load(types::I8, MemFlagsData::trusted(), $addr, 0);
                    let bad = $bx.ins().icmp_imm_s(
                        IntCC::NotEqual,
                        tag,
                        crate::value::layout::TAG_STR as i64,
                    );
                    bail!($bx, bad);
                    $addr
                } else {
                    let tag = $bx.ins().load(types::I8, MemFlagsData::trusted(), $addr, 0);
                    let raw = $bx
                        .ins()
                        .load(types::I64, MemFlagsData::trusted(), $addr, PAYLOAD);
                    match t {
                        JTy::Int => {
                            let bad = $bx.ins().icmp_imm_s(IntCC::NotEqual, tag, TAG_INT as i64);
                            bail!($bx, bad);
                            raw
                        }
                        JTy::Bool => {
                            let bad = $bx.ins().icmp_imm_s(IntCC::NotEqual, tag, TAG_BOOL as i64);
                            bail!($bx, bad);
                            $bx.ins().ireduce(types::I8, raw)
                        }
                        _ => {
                            let isf = $bx.ins().icmp_imm_s(IntCC::Equal, tag, TAG_FLOAT as i64);
                            let isi = $bx.ins().icmp_imm_s(IntCC::Equal, tag, TAG_INT as i64);
                            let ok = $bx.ins().bor(isf, isi);
                            let bad = $bx.ins().icmp_imm_s(IntCC::Equal, ok, 0);
                            bail!($bx, bad);
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
                (site_b, site_k) = (bi, k);
                dty = plan.defs[bi][k];
                let key = ins as *const Instr as usize;
                // A call through the interpreter: decided by the analysis, or its native
                // target could not be compiled.
                let boxed_call = matches!(ins, Call { .. } | CallMethod { .. })
                    && !inlines.contains_key(&key)
                    && (plan.boxed.contains(&key)
                        || plan
                            .targets
                            .get(&key)
                            .is_some_and(|t| !self.compiled.contains_key(&t.key())));
                match ins {
                    Call { dst, .. } | CallMethod { dst, .. } if boxed_call => {
                        let mut words = Vec::new();
                        let callee = match ins {
                            Call { f, site, .. } => {
                                let callee = match tyof!(*f) {
                                    Some(JTy::Callee(s) | JTy::Ctor(s)) => BoxedCallee::Global(s),
                                    _ => {
                                        words.push(boxed_word!(bx, *f));
                                        BoxedCallee::Value
                                    }
                                };
                                let site = &p.calls[*site as usize];
                                for k in 0..site.npos {
                                    words.push(boxed_word!(bx, site.start + k));
                                }
                                callee
                            }
                            CallMethod {
                                recv,
                                name,
                                site,
                                access,
                                ..
                            } => {
                                words.push(boxed_word!(bx, *recv));
                                let site = &p.calls[*site as usize];
                                for k in 0..site.npos {
                                    words.push(boxed_word!(bx, site.start + k));
                                }
                                BoxedCallee::Method(*name, *access)
                            }
                            _ => unreachable!(),
                        };
                        let buf =
                            bx.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                (words.len().max(1) * 8) as u32,
                                3,
                            ));
                        let base = bx.ins().stack_addr(types::I64, buf, 0);
                        for (j, (w, _)) in words.iter().enumerate() {
                            bx.ins()
                                .store(MemFlagsData::trusted(), *w, base, (j * 8) as i32);
                        }
                        let want = match dty {
                            Some(JTy::Int) => 0,
                            Some(JTy::Float) => 1,
                            Some(JTy::Bool) => 2,
                            Some(t) if is_ref(Some(t)) => 3,
                            _ => 4,
                        };
                        self.boxed_calls.push(Box::new(BoxedCall {
                            callee,
                            words: words.iter().map(|(_, k)| *k).collect(),
                            want,
                            span,
                        }));
                        let desc = &**self.boxed_calls.last().unwrap() as *const BoxedCall;
                        let dp = bx.ins().iconst(types::I64, desc as i64);
                        let out = if want == 3 {
                            own!(bx, *dst)
                        } else {
                            bx.ins().stack_addr(types::I64, dout, 0)
                        };
                        let call = bx.ins().call(call_boxed_ref, &[ctxv, dp, base, out]);
                        let status = bx.inst_results(call)[0];
                        // 2: a result of another type: resumed after the call with it.
                        let other = bx.ins().icmp_imm_s(IntCC::Equal, status, 2);
                        deopt!(bx, other, cur, site_b, site_k + 1, Some(*dst));
                        let failed = bx.create_block();
                        let cont = bx.create_block();
                        bx.ins().brif(status, failed, &[], cont, &[]);
                        bx.switch_to_block(failed);
                        let none = bx.ins().iconst(types::I64, -1);
                        bx.ins().jump(fail, &[none.into()]);
                        bx.switch_to_block(cont);
                        match want {
                            0 => {
                                let v = bx.ins().stack_load(types::I64, types::I64, dout, 0);
                                bx.def_var(dvar!(*dst), v);
                            }
                            1 => {
                                let v = bx.ins().stack_load(types::I64, types::F64, dout, 0);
                                bx.def_var(dvar!(*dst), v);
                            }
                            2 => {
                                let v = bx.ins().stack_load(types::I64, types::I8, dout, 0);
                                bx.def_var(dvar!(*dst), v);
                            }
                            3 => bx.def_var(dvar!(*dst), out),
                            _ => {}
                        }
                    }
                    // Operations on values of any type: the interpreter's.
                    Bin(op, d, a, b2) if plan.boxed.contains(&key) => {
                        let (wa, ka) = boxed_word!(bx, *a);
                        let (wb, kb) = boxed_word!(bx, *b2);
                        let slot = own!(bx, *d);
                        let oc = bx.ins().iconst(types::I32, op_code(*op));
                        let ka = bx.ins().iconst(types::I32, dkind_code(ka));
                        let kb = bx.ins().iconst(types::I32, dkind_code(kb));
                        let sp = bx
                            .ins()
                            .iconst(types::I64, span.start as i64 | ((span.end as i64) << 32));
                        let call = bx
                            .ins()
                            .call(binary_ref, &[ctxv, oc, wa, ka, wb, kb, slot, sp]);
                        let status = bx.inst_results(call)[0];
                        let failed = bx.create_block();
                        let cont = bx.create_block();
                        bx.ins().brif(status, failed, &[], cont, &[]);
                        bx.switch_to_block(failed);
                        let none = bx.ins().iconst(types::I64, -1);
                        bx.ins().jump(fail, &[none.into()]);
                        bx.switch_to_block(cont);
                        bx.def_var(dvar!(*d), slot);
                    }
                    // A value of any type to a simple type: its tag checked (otherwise
                    // resumed in the interpreter, which gives the error).
                    Guard(d, a, _) if plan.boxed.contains(&key) => {
                        use crate::value::layout::{PAYLOAD, TAG_BOOL, TAG_FLOAT, TAG_INT};
                        let v = bx.use_var(uvar!(*a));
                        let tag = bx.ins().load(types::I8, MemFlagsData::trusted(), v, 0);
                        let raw = bx
                            .ins()
                            .load(types::I64, MemFlagsData::trusted(), v, PAYLOAD);
                        let r = match dty {
                            Some(JTy::Int) => {
                                let bad = bx.ins().icmp_imm_s(IntCC::NotEqual, tag, TAG_INT as i64);
                                bail!(bx, bad);
                                raw
                            }
                            Some(JTy::Bool) => {
                                let bad =
                                    bx.ins().icmp_imm_s(IntCC::NotEqual, tag, TAG_BOOL as i64);
                                bail!(bx, bad);
                                bx.ins().ireduce(types::I8, raw)
                            }
                            _ => {
                                let isf = bx.ins().icmp_imm_s(IntCC::Equal, tag, TAG_FLOAT as i64);
                                let isi = bx.ins().icmp_imm_s(IntCC::Equal, tag, TAG_INT as i64);
                                let ok = bx.ins().bor(isf, isi);
                                let bad = bx.ins().icmp_imm_s(IntCC::Equal, ok, 0);
                                bail!(bx, bad);
                                let f = bx.ins().bitcast(types::F64, MemFlagsData::new(), raw);
                                let i = bx.ins().fcvt_from_sint(types::F64, raw);
                                bx.ins().select(isf, f, i)
                            }
                        };
                        bx.def_var(dvar!(*d), r);
                    }
                    Not(d, a) if plan.boxed.contains(&key) => {
                        let v = bx.use_var(uvar!(*a));
                        let call = bx.ins().call(truthy_ref, &[v]);
                        let t = bx.inst_results(call)[0];
                        let r = bx.ins().icmp_imm_s(IntCC::Equal, t, 0);
                        bx.def_var(dvar!(*d), r);
                    }
                    GetGlobal(d, slot) if plan.boxed.contains(&key) => {
                        let out = own!(bx, *d);
                        let s = bx.ins().iconst(types::I32, *slot as i64);
                        let call = bx.ins().call(global_ref, &[ctxv, s, out]);
                        let failed = bx.inst_results(call)[0];
                        bail!(bx, failed);
                        bx.def_var(dvar!(*d), out);
                    }
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
                    Move(..) if dty == Some(JTy::Nil) => {}
                    Move(_, a) if is_ref(dty) => {
                        let d = match ins {
                            Move(d, _) => d,
                            _ => unreachable!(),
                        };
                        let v = bx.use_var(uvar!(*a));
                        put_ref!(bx, *d, v);
                    }
                    Move(d, a) | Guard(d, a, _) | Coerce { dst: d, src: a, .. } => {
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
                    // String literal: the address of the constant.
                    Const(d, c) => {
                        let v = bx
                            .ins()
                            .iconst(types::I64, &p.consts[*c as usize] as *const Value as i64);
                        put_ref!(bx, *d, v);
                    }
                    // Strings: the interpreter's operations (resumed in it if a value is not a
                    // string).
                    Bin(op, d, a, b2)
                        if tyof!(*a) == Some(JTy::Str) && tyof!(*b2) == Some(JTy::Str) =>
                    {
                        let (x, y) = (bx.use_var(uvar!(*a)), bx.use_var(uvar!(*b2)));
                        if *op == BinOp::Add {
                            let slot = own!(bx, *d);
                            let call = bx.ins().call(str_add_ref, &[ctxv, slot, x, y]);
                            let failed = bx.inst_results(call)[0];
                            bail!(bx, failed);
                            bx.def_var(dvar!(*d), slot);
                        } else {
                            let code = bx.ins().iconst(types::I32, str_op_code(*op).ok_or(())?);
                            let call = bx.ins().call(str_cmp_ref, &[ctxv, code, x, y]);
                            let r = bx.inst_results(call)[0];
                            let failed = bx.ins().icmp_imm_s(IntCC::Equal, r, 2);
                            bail!(bx, failed);
                            let v = bx.ins().ireduce(types::I8, r);
                            bx.def_var(dvar!(*d), v);
                        }
                    }
                    Concat(d, start, n) => {
                        let words =
                            bx.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                (*n).max(1) * 8,
                                3,
                            ));
                        let kinds =
                            bx.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                (*n).max(1),
                                0,
                            ));
                        let wb = bx.ins().stack_addr(types::I64, words, 0);
                        let kb = bx.ins().stack_addr(types::I64, kinds, 0);
                        for k in 0..*n {
                            let (bits, kind) = native_arg!(bx, *start + k);
                            bx.ins()
                                .store(MemFlagsData::trusted(), bits, wb, (k * 8) as i32);
                            let kc = bx.ins().iconst(types::I8, kind);
                            bx.ins().store(MemFlagsData::trusted(), kc, kb, k as i32);
                        }
                        let slot = own!(bx, *d);
                        let nn = bx.ins().iconst(types::I64, *n as i64);
                        let call = bx.ins().call(concat_ref, &[slot, wb, kb, nn]);
                        let failed = bx.inst_results(call)[0];
                        bail!(bx, failed);
                        bx.def_var(dvar!(*d), slot);
                    }
                    // `xs << v`, `xs[i] = v`: the helper refuses a list being borrowed or an
                    // index out of bounds (resumed in the interpreter, which gives the error).
                    Bin(BinOp::Push, d, a, b2) => {
                        let list = bx.use_var(uvar!(*a));
                        let (bits, kind) = native_arg!(bx, *b2);
                        let k = bx.ins().iconst(types::I32, kind);
                        let call = bx.ins().call(push_ref, &[list, bits, k]);
                        let failed = bx.inst_results(call)[0];
                        bail!(bx, failed);
                        bx.def_var(dvar!(*d), list);
                    }
                    SetIndex(a, i, v) => {
                        let list = bx.use_var(uvar!(*a));
                        let i = bx.use_var(uvar!(*i));
                        let (bits, kind) = native_arg!(bx, *v);
                        let k = bx.ins().iconst(types::I32, kind);
                        let call = bx.ins().call(set_ref, &[list, i, bits, k]);
                        let failed = bx.inst_results(call)[0];
                        bail!(bx, failed);
                    }
                    // A list field of `self`: copied into its slot (the field may change).
                    FieldAt(d, 0, idx)
                        if tyof!(0) == Some(JTy::This)
                            && matches!(dty, Some(JTy::List(_) | JTy::Str)) =>
                    {
                        let fp = this_fp.ok_or(())?;
                        let off = *idx as i32 * crate::value::layout::SIZE;
                        let addr = bx.ins().iadd_imm_s(fp, off as i64);
                        put_ref!(bx, *d, addr);
                    }
                    SetFieldAt {
                        obj: 0, idx, src, ..
                    } if tyof!(0) == Some(JTy::This)
                        && matches!(tyof!(*src), Some(JTy::List(_) | JTy::Str)) =>
                    {
                        let fp = this_fp.ok_or(())?;
                        let off = *idx as i32 * crate::value::layout::SIZE;
                        let addr = bx.ins().iadd_imm_s(fp, off as i64);
                        let v = bx.use_var(uvar!(*src));
                        bx.ins().call(clone_ref, &[addr, v]);
                    }
                    // `@x` of `self`: its type was checked on entry.
                    FieldAt(d, 0, idx) if tyof!(0) == Some(JTy::This) => {
                        let fp = this_fp.ok_or(())?;
                        let off = *idx as i32 * crate::value::layout::SIZE;
                        let (cty, want) = match dty {
                            Some(JTy::Float) => (types::F64, crate::value::layout::TAG_FLOAT),
                            Some(JTy::Bool) => (types::I8, crate::value::layout::TAG_BOOL),
                            _ => (types::I64, crate::value::layout::TAG_INT),
                        };
                        let tag = bx.ins().load(types::I8, MemFlagsData::trusted(), fp, off);
                        let bad = bx.ins().icmp_imm_s(IntCC::NotEqual, tag, want as i64);
                        bail!(bx, bad);
                        let v = bx.ins().load(
                            cty,
                            MemFlagsData::trusted(),
                            fp,
                            off + crate::value::layout::PAYLOAD,
                        );
                        bx.def_var(dvar!(*d), v);
                    }
                    SetFieldAt {
                        obj: 0, idx, src, ..
                    } if tyof!(0) == Some(JTy::This) => {
                        let fp = this_fp.ok_or(())?;
                        let info = plan.this.as_ref().ok_or(())?;
                        let off = *idx as i32 * crate::value::layout::SIZE;
                        let v = bx.use_var(uvar!(*src));
                        let (tag, v) = match (
                            field_jty(info, *idx, p.spec.as_deref()).ok_or(())?,
                            tyof!(*src),
                        ) {
                            (JTy::Float, Some(JTy::Int)) => (
                                crate::value::layout::TAG_FLOAT,
                                bx.ins().fcvt_from_sint(types::F64, v),
                            ),
                            (JTy::Float, _) => (crate::value::layout::TAG_FLOAT, v),
                            (JTy::Bool, _) => (crate::value::layout::TAG_BOOL, v),
                            _ => (crate::value::layout::TAG_INT, v),
                        };
                        // The old value must be of the same simple type: nothing to drop.
                        let old = bx.ins().load(types::I8, MemFlagsData::trusted(), fp, off);
                        let bad = bx.ins().icmp_imm_s(IntCC::NotEqual, old, tag as i64);
                        bail!(bx, bad);
                        let t = bx.ins().iconst(types::I8, tag as i64);
                        bx.ins().store(MemFlagsData::trusted(), t, fp, off);
                        bx.ins().store(
                            MemFlagsData::trusted(),
                            v,
                            fp,
                            off + crate::value::layout::PAYLOAD,
                        );
                    }
                    // Inlined method: its class checked with a cache (resumed in the
                    // interpreter, which makes the call, if it redefines the method).
                    CallMethod {
                        dst,
                        recv,
                        name,
                        site,
                        ..
                    } if inlines.contains_key(&(ins as *const Instr as usize)) => {
                        let (q, qp) = &inlines[&(ins as *const Instr as usize)];
                        let target = &plan.targets[&(ins as *const Instr as usize)];
                        let declared = target.declared.as_ref().ok_or(())?;
                        self.method_caches.push(Box::new(MethodCache {
                            info: std::cell::RefCell::new(None),
                            name: *name,
                            proto: Rc::as_ptr(declared),
                        }));
                        let cache = &**self.method_caches.last().unwrap() as *const MethodCache;
                        let obj = bx.use_var(uvar!(*recv));
                        let cp = bx.ins().iconst(types::I64, cache as i64);
                        let call = bx.ins().call(fields_cached_ref, &[obj, cp]);
                        let fp = bx.inst_results(call)[0];
                        let null = bx.ins().icmp_imm_s(IntCC::Equal, fp, 0);
                        bail!(bx, null);
                        let (cparams, _) = native_sig(q).ok_or(())?;
                        let site = &p.calls[*site as usize];
                        let mut args = Vec::new();
                        for (k, pt) in cparams[1..].iter().enumerate() {
                            let r = site.start + k as u32;
                            let v = bx.use_var(uvar!(r));
                            args.push(match (pt, tyof!(r)) {
                                (RegTy::Float, Some(JTy::Int)) => {
                                    bx.ins().fcvt_from_sint(types::F64, v)
                                }
                                _ => v,
                            });
                        }
                        // Methods have `self` in register 0, not passed as an argument.
                        let tid = {
                            call_sites.push((q.name.clone(), span));
                            trace_base + (call_sites.len() - 1) as i64
                        };
                        let v = emit_inline(
                            &mut bx,
                            q,
                            qp,
                            &args,
                            Some(fp),
                            frem_ref,
                            &mut |bx, c, k, s| inline_err!(bx, c, k, s, tid),
                            &mut |bx, c| bail!(bx, c),
                        )
                        .ok_or(())?;
                        bx.def_var(dvar!(*dst), v);
                    }
                    // Method called directly: abandon if the object's class redefines it.
                    CallMethod {
                        dst,
                        recv,
                        name,
                        site,
                        ..
                    } if plan.targets.contains_key(&(ins as *const Instr as usize)) => {
                        let target = &plan.targets[&(ins as *const Instr as usize)];
                        let (fref, q) = callee_refs[&target.key()].clone();
                        let (cparams, cret) = native_sig(&q).ok_or(())?;
                        let site = &p.calls[*site as usize];
                        let obj = bx.use_var(uvar!(*recv));
                        let nm = bx.ins().iconst(types::I32, name.0 as i64);
                        let declared = target.declared.as_ref().ok_or(())?;
                        let pp = bx.ins().iconst(types::I64, Rc::as_ptr(declared) as i64);
                        let call = bx.ins().call(method_ref, &[obj, nm, pp]);
                        let other = bx.inst_results(call)[0];
                        bail!(bx, other);
                        let mut args = vec![ctxv, obj];
                        for (k, pt) in cparams[1..].iter().enumerate() {
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
                        let out = if cret == RegTy::Value {
                            let slot = own!(bx, *dst);
                            args.push(slot);
                            Some(slot)
                        } else {
                            None
                        };
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
                        bail!(bx, too_deep);
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
                        call_sites.push((q.name.clone(), span));
                        let tid = trace_base + (call_sites.len() - 1) as i64;
                        let failed = bx.create_block();
                        let cont = bx.create_block();
                        bx.ins().brif(status, failed, &[], cont, &[]);
                        bx.switch_to_block(failed);
                        // 2: the called function, resumed in the interpreter, returned a
                        // value of another type: resumed here too, after the call.
                        let other = bx.ins().icmp_imm_s(IntCC::Equal, status, 2);
                        deopt!(bx, other, cur, site_b, site_k + 1, Some(*dst));
                        let t = bx.ins().iconst(types::I32, tid);
                        bx.ins().call(trace_ref, &[ctxv, t]);
                        let none = bx.ins().iconst(types::I64, -1);
                        bx.ins().jump(fail, &[none.into()]);
                        bx.switch_to_block(cont);
                        bx.def_var(dvar!(*dst), out.unwrap_or(result));
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
                                bail!(bx, null);
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
                        bail!(bx, bad);
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
                        bail!(bx, status);
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
                    // Inlined function: its body here (it is redefined: this code is
                    // recompiled).
                    Call { dst, site, .. }
                        if inlines.contains_key(&(ins as *const Instr as usize)) =>
                    {
                        let (q, qp) = &inlines[&(ins as *const Instr as usize)];
                        let (cparams, _) = native_sig(q).ok_or(())?;
                        let site = &p.calls[*site as usize];
                        let mut args = Vec::new();
                        for (k, pt) in cparams.iter().enumerate() {
                            let r = site.start + k as u32;
                            let v = bx.use_var(uvar!(r));
                            args.push(match (pt, tyof!(r)) {
                                (RegTy::Float, Some(JTy::Int)) => {
                                    bx.ins().fcvt_from_sint(types::F64, v)
                                }
                                _ => v,
                            });
                        }
                        let tid = {
                            call_sites.push((q.name.clone(), span));
                            trace_base + (call_sites.len() - 1) as i64
                        };
                        let v = emit_inline(
                            &mut bx,
                            q,
                            qp,
                            &args,
                            None,
                            frem_ref,
                            &mut |bx, c, k, s| inline_err!(bx, c, k, s, tid),
                            &mut |bx, c| bail!(bx, c),
                        )
                        .ok_or(())?;
                        bx.def_var(dvar!(*dst), v);
                    }
                    Call { dst, f, site } => {
                        let Some(JTy::Callee(_)) = tyof!(*f) else {
                            return Err(());
                        };
                        let target = plan
                            .targets
                            .get(&(ins as *const Instr as usize))
                            .ok_or(())?;
                        let (fref, q) = callee_refs[&target.key()].clone();
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
                        bail!(bx, too_deep);
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
                        // 2: the called function, resumed in the interpreter, returned a
                        // value of another type: resumed here too, after the call.
                        let other = bx.ins().icmp_imm_s(IntCC::Equal, status, 2);
                        deopt!(bx, other, cur, site_b, site_k + 1, Some(*dst));
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
                            let f = if may_mutate { snapshot_ref } else { clone_ref };
                            bx.ins().call(f, &[slot, list]);
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
                        bail!(bx, out);
                        let off = bx.ins().imul_imm_s(pos, crate::value::layout::SIZE as i64);
                        let addr = bx.ins().iadd(data, off);
                        let v = elem!(bx, addr, e, span);
                        if is_ref(dty) {
                            put_ref!(bx, *d, v);
                        } else {
                            bx.def_var(dvar!(*d), v);
                        }
                    }
                    CallMethod { dst, recv, .. } if tyof!(*recv) == Some(JTy::Str) => {
                        let s = bx.use_var(uvar!(*recv));
                        let call = bx.ins().call(str_len_ref, &[s]);
                        let len = bx.inst_results(call)[0];
                        let failed = bx.ins().icmp_imm_s(IntCC::SignedLessThan, len, 0);
                        bail!(bx, failed);
                        bx.def_var(dvar!(*dst), len);
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
            (site_b, site_k) = (bi, b.instrs.len());
            match &b.term {
                Term::Reraise(_) => unreachable!("function with handlers compiled"),
                Term::Jump(t) => {
                    bx.ins().jump(blocks[*t as usize], &[]);
                }
                Term::Branch { cond, then, else_ } => {
                    let (t, e) = (blocks[*then as usize], blocks[*else_ as usize]);
                    match tyof!(*cond) {
                        Some(JTy::Boxed) => {
                            let v = bx.use_var(uvar!(*cond));
                            let call = bx.ins().call(truthy_ref, &[v]);
                            let c = bx.inst_results(call)[0];
                            bx.ins().brif(c, t, &[], e, &[]);
                        }
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
                // Result not annotated, of a simple type (or `nil`): a value in the slot.
                Term::Return { reg, .. }
                    if p.ret.is_none()
                        && matches!(
                            tyof!(*reg),
                            Some(JTy::Int | JTy::Float | JTy::Bool | JTy::Nil)
                        ) =>
                {
                    let out = out_param.ok_or(())?;
                    let (bits, kind) = boxed_word!(bx, *reg);
                    let k = bx.ins().iconst(types::I32, dkind_code(kind));
                    bx.ins().call(box_ref, &[out, bits, k]);
                    drop_slots!(bx);
                    let ok = bx.ins().iconst(types::I32, 0);
                    let zero = bx.ins().iconst(types::I64, 0);
                    bx.ins().return_(&[ok, zero]);
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
                    // Checked first: a resumption sees the iterator before this element.
                    let v = elem!(bx, addr, a, b.term_span);
                    let i1 = bx.ins().iadd_imm_s(i, 1);
                    bx.def_var(iv, i1);
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

    /// Calls native code (`entry`, `n` parameters, result `ret`): `args`, one per native
    /// parameter (`self` first for a method), alive during the call.
    ///
    /// # Safety
    /// `jit` belongs to the runtime `rt`, both valid; the native code may call the runtime
    /// back (resumption in the interpreter), which may call native code again: neither is
    /// borrowed during the call.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn call<'a>(
        jit: *mut Jit,
        rt: *mut Runtime,
        entry: usize,
        n: usize,
        ret: RegTy,
        args: impl IntoIterator<Item = &'a Value>,
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
        // Beyond: resumption in the interpreter, which grows the stack.
        let limit = max_depth.min(stack_cap);
        // SAFETY: see above; the context is only reached through its address.
        unsafe {
            let ctx = (*jit).ctx;
            (*ctx).depth = depth as i64;
            (*ctx).max_depth = limit as i64;
            (*ctx).err = -1;
            (*ctx).traces.clear();
            (*ctx).rt = rt;
            (*ctx).pending = None;
            let mut out = 0u64;
            // Object or list returned: written directly into `result`.
            let mut result = Value::Nil;
            let out_ptr = match ret {
                RegTy::Value => &mut result as *mut Value as *mut u64,
                _ => &mut out as *mut u64,
            };
            // `entry` is an entry point generated by `define_entry` with this signature;
            // the arguments have the expected number and types.
            let f: extern "C" fn(*mut JitCtx, *const u64, *mut u64) -> i32 =
                std::mem::transmute(entry);
            let status = f(ctx, buf.as_ptr(), out_ptr);
            if status == 2 {
                return Ok((*ctx).returned.take().unwrap_or(Value::Nil));
            }
            if status == 0 {
                return Ok(match ret {
                    RegTy::Int => Value::Int(out as i64),
                    RegTy::Float => Value::Float(f64::from_bits(out)),
                    RegTy::Bool => Value::Bool(out & 1 == 1),
                    RegTy::Value => result,
                });
            }
            let jit = &*jit;
            let mut e = match (*ctx).err {
                PENDING => (*ctx).pending.take().expect("error of a resumption"),
                err => {
                    let info = &jit.errors[err as usize];
                    let msg = match info.kind {
                        ErrKind::Overflow => "integer overflow",
                        ErrKind::DivZero => "division by zero",
                    };
                    let mut e = Error::runtime(msg, info.span);
                    e.src = Some((info.script.name.clone(), info.script.src.clone()));
                    e
                }
            };
            for &t in (*ctx).traces.iter().take(16) {
                let t = &jit.traces[t as usize];
                e.trace.push((t.name.to_string(), t.span));
            }
            Err(e)
        }
    }
}

impl Drop for Jit {
    fn drop(&mut self) {
        // SAFETY: allocated by `Box::into_raw` in `Jit::new`, freed once.
        unsafe { drop(Box::from_raw(self.ctx)) };
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
        // `rescue` / `ensure`: left to the interpreter.
        if !p.handlers.is_empty() {
            return None;
        }
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
                Term::Reraise(_) => unreachable!("function with handlers compiled"),
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
