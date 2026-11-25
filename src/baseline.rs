//! Helpers of the baseline native code (untyped functions; `jit` and `wasm_jit`): its
//! registers stay `Value`s in the call's window (`NativeFrame`), and what the code does not
//! do itself, it asks these functions, the interpreter's own code (`Runtime::step`) last.
//! Each returns a status: 0 done, otherwise the case is left to the interpreter (or, for
//! `bk_step` and `bk_call`, an error is waiting in the `Runtime`).

use crate::ir::{Instr, NativeFrame, RegTy};
use crate::span::Span;
use crate::value::Value;
use crate::vm::Runtime;
use std::rc::Rc;

/// Runs an IR instruction with the interpreter's code. 0: success, 1: error (stored in
/// the `Runtime`).
pub(crate) extern "C" fn bk_step(
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
pub(crate) extern "C" fn bk_iter_next(f: *mut NativeFrame, iter: u32, dst: u32) -> i32 {
    // SAFETY: as for `bk_step`.
    unsafe {
        let f = &mut *f;
        let regs = std::slice::from_raw_parts_mut(f.regs, f.nregs);
        crate::vm::iter_next(regs, iter, dst) as i32
    }
}

/// Drops the value of a register before writing a scalar value into it.
pub(crate) extern "C" fn bk_drop(slot: *mut Value) {
    // SAFETY: `slot` is a register of the current window.
    unsafe {
        std::ptr::drop_in_place(slot);
        slot.write(Value::Nil);
    }
}

/// Cache of a field access (`p.x`): last type seen and the field's offset. The `Rc`
/// keeps the type alive, so its address cannot refer to another type.
pub(crate) struct FieldCache {
    pub(crate) info: Option<Rc<crate::objects::TypeInfo>>,
    pub(crate) idx: usize,
    pub(crate) name: crate::intern::Symbol,
    /// Expected type (`FieldAs`), or `Value`.
    pub(crate) want: RegTy,
}

impl FieldCache {
    /// Field `name` of object `v` (`type` only: a `class` goes through a getter).
    pub(crate) fn get(&mut self, v: &Value) -> Option<Value> {
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
pub(crate) enum Fast {
    GetUp,
    SetUp,
    GetCell,
    SetCell,
    GetGlobal,
    SetGlobal,
    Field,
    Call,
}

pub(crate) type FastFn = extern "C" fn(*mut Runtime, *mut NativeFrame, u32, u32, *const u8) -> i32;

pub(crate) const FAST: &[(&str, FastFn)] = &[
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
