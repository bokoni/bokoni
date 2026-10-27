//! Runtime values.
//!
//! Base types (`nil`, `bool`, `int`, `float`) are stored inline, without allocation.
//! Compound values are shared through reference counting (`Rc`): `let b = a` does not
//! copy the list, both names refer to the same one.

use crate::intern::{FxHashMap, FxHashSet, Interner, Symbol};
use crate::objects::{EnumInfo, Object, TypeInfo};
use std::cell::RefCell;
use std::fmt::{self, Write};
use std::rc::Rc;

/// Fixed layout (`repr(C, u8)`): JIT native code reads the tag (byte 0) and the payload of
/// base types (byte 8) directly. Base types (`nil`, `bool`, `int`, `float`) have tags 0 to 3
/// and own nothing: copying or overwriting them needs no reference counting.
#[derive(Clone)]
#[repr(C, u8)]
pub enum Value {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    List(Rc<RefCell<Vec<Value>>>),
    Hash(Rc<RefCell<HashMap>>),
    Record(Rc<RefCell<Record>>),
    Tuple(Rc<[Value]>),
    Range(Rc<Range>),
    Function(Rc<Function>),
    /// Instance of a `type` or a `class`.
    Object(Rc<RefCell<Object>>),
    /// Variant of an `enum` (`Protocol::Tcp`): the enum and the variant's index.
    Variant(Rc<EnumInfo>, u32),
    /// A `type` / `class` itself (callable: construction).
    Type(Rc<TypeInfo>),
    /// An `enum` itself (`Protocol.values()`).
    Enum(Rc<EnumInfo>),
    /// Internal: object field, variable or parameter not initialized yet.
    Uninit,
    /// Internal: iterator of a `for` loop.
    Iter(Rc<RefCell<crate::vm::IterState>>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub lo: i64,
    pub hi: i64,
    pub inclusive: bool,
}

impl Range {
    /// Values of the range, in order (no overflow, even up to `i64::MAX`).
    pub fn values(&self) -> std::ops::RangeInclusive<i64> {
        match (self.inclusive, self.hi.checked_sub(1)) {
            (true, _) => self.lo..=self.hi,
            (false, Some(last)) => self.lo..=last,
            // `lo...i64::MIN`: empty (an already exhausted range).
            (false, None) => {
                let mut empty = 0..=0;
                empty.next();
                empty
            }
        }
    }

    pub fn contains(&self, n: i64) -> bool {
        n >= self.lo
            && if self.inclusive {
                n <= self.hi
            } else {
                n < self.hi
            }
    }

    pub fn len(&self) -> usize {
        let n = self.hi as i128 - self.lo as i128 + self.inclusive as i128;
        n.clamp(0, usize::MAX as i128) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Fields of a record, in declaration order.
#[derive(Clone, Default)]
pub struct Record {
    pub fields: Vec<(Symbol, Value)>,
}

impl Record {
    pub fn get(&self, name: Symbol) -> Option<&Value> {
        self.fields.iter().find(|(n, _)| *n == name).map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, name: Symbol) -> Option<&mut Value> {
        self.fields
            .iter_mut()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v)
    }
}

/// Hash key: only immutable, exactly comparable values are accepted.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum HashKey {
    Nil,
    Bool(bool),
    Int(i64),
    Str(Rc<str>),
}

impl HashKey {
    pub fn from_value(v: &Value) -> Option<HashKey> {
        Some(match v {
            Value::Nil => HashKey::Nil,
            Value::Bool(b) => HashKey::Bool(*b),
            Value::Int(n) => HashKey::Int(*n),
            Value::Str(s) => HashKey::Str(s.clone()),
            _ => return None,
        })
    }

    pub fn to_value(&self) -> Value {
        match self {
            HashKey::Nil => Value::Nil,
            HashKey::Bool(b) => Value::Bool(*b),
            HashKey::Int(n) => Value::Int(*n),
            HashKey::Str(s) => Value::Str(s.clone()),
        }
    }
}

/// Hash that preserves insertion order.
#[derive(Clone, Default)]
pub struct HashMap {
    entries: Vec<(HashKey, Value)>,
    index: FxHashMap<HashKey, usize>,
}

impl HashMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, k: &HashKey) -> Option<&Value> {
        self.index.get(k).map(|&i| &self.entries[i].1)
    }

    pub fn insert(&mut self, k: HashKey, v: Value) {
        match self.index.get(&k) {
            Some(&i) => self.entries[i].1 = v,
            None => {
                self.index.insert(k.clone(), self.entries.len());
                self.entries.push((k, v));
            }
        }
    }

    pub fn remove(&mut self, k: &HashKey) -> Option<Value> {
        let i = self.index.remove(k)?;
        let (_, v) = self.entries.remove(i);
        for idx in self.index.values_mut() {
            if *idx > i {
                *idx -= 1;
            }
        }
        Some(v)
    }

    pub fn iter(&self) -> impl Iterator<Item = &(HashKey, Value)> {
        self.entries.iter()
    }
}

/// Callable function.
pub enum Function {
    /// Script function or lambda.
    Script(crate::vm::Closure),
    /// Rust function registered with [`Engine::register_fn`](crate::Engine::register_fn).
    Native(NativeFn),
    /// Builtin function that needs the interpreter (`print`…).
    Builtin(crate::ops::Builtin),
}

/// Common signature of Rust functions callable from scripts.
pub type NativeImpl = dyn Fn(&[Value]) -> Result<Value, String>;

pub struct NativeFn {
    pub name: Rc<str>,
    pub f: Box<NativeImpl>,
}

impl Function {
    pub fn name(&self) -> &str {
        match self {
            Function::Script(c) => &c.proto.name,
            Function::Native(n) => &n.name,
            Function::Builtin(b) => crate::ops::Builtin::ALL
                .iter()
                .find(|(_, x)| x == b)
                .map_or("?", |(n, _)| n),
        }
    }
}

impl Value {
    pub fn str(s: &str) -> Value {
        Value::Str(s.into())
    }

    pub fn list(items: Vec<Value>) -> Value {
        Value::List(Rc::new(RefCell::new(items)))
    }

    /// Type name, as written in annotations.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Nil => "nil",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Hash(_) => "hash",
            Value::Record(_) => "record",
            Value::Tuple(_) => "tuple",
            Value::Range(_) => "range",
            Value::Function(_) => "function",
            Value::Object(o) => match o.borrow().info.kind {
                crate::ast::TypeDeclKind::Class => "object",
                _ => "type value",
            },
            Value::Variant(..) => "enum variant",
            Value::Type(_) => "type",
            Value::Enum(_) => "enum",
            Value::Uninit | Value::Iter(_) => "internal value",
        }
    }

    /// Type name for messages: the user type's name when there is one.
    pub fn type_label(&self) -> String {
        match self {
            Value::Object(o) => o.borrow().info.name.to_string(),
            Value::Variant(e, _) => e.name.to_string(),
            v => v.type_name().to_string(),
        }
    }

    /// `nil` and `false` are falsy, everything else is truthy (as in Ruby).
    #[inline]
    pub fn truthy(&self) -> bool {
        !matches!(self, Value::Nil | Value::Bool(false))
    }

    /// Structural equality (`==`). An `int` and a `float` of equal value are equal.
    pub fn equals(&self, other: &Value) -> bool {
        // Empty set: no allocation for simple values.
        self.equals_in(other, &mut FxHashSet::default())
    }

    /// Structural equality. `seen`: pairs of containers already compared (or in progress);
    /// a pair seen again counts as equal — this gives equality of cyclic structures, and
    /// bounds the work by the number of pairs even for heavily shared structures. The stack
    /// grows on the heap for very deep structures.
    fn equals_in(&self, other: &Value, seen: &mut FxHashSet<(usize, usize)>) -> bool {
        use Value::*;
        let pair = |a: *const (), b: *const ()| (a as usize, b as usize);
        let key = match (self, other) {
            (List(a), List(b)) => Some(pair(Rc::as_ptr(a) as _, Rc::as_ptr(b) as _)),
            (Tuple(a), Tuple(b)) => {
                Some(pair(Rc::as_ptr(a) as *const (), Rc::as_ptr(b) as *const ()))
            }
            (Hash(a), Hash(b)) => Some(pair(Rc::as_ptr(a) as _, Rc::as_ptr(b) as _)),
            (Record(a), Record(b)) => Some(pair(Rc::as_ptr(a) as _, Rc::as_ptr(b) as _)),
            (Object(a), Object(b)) => Some(pair(Rc::as_ptr(a) as _, Rc::as_ptr(b) as _)),
            _ => None,
        };
        if let Some(k) = key {
            if !seen.insert(k) {
                return true;
            }
            return stacker::maybe_grow(64 * 1024, 1024 * 1024, || self.equals_items(other, seen));
        }
        self.equals_items(other, seen)
    }

    fn equals_items(&self, other: &Value, seen: &mut FxHashSet<(usize, usize)>) -> bool {
        use Value::*;
        match (self, other) {
            (Nil, Nil) => true,
            (Bool(a), Bool(b)) => a == b,
            (Int(a), Int(b)) => a == b,
            (Float(a), Float(b)) => a == b,
            (Int(a), Float(b)) | (Float(b), Int(a)) => (*a as f64) == *b,
            (Str(a), Str(b)) => a == b,
            (List(a), List(b)) => {
                Rc::ptr_eq(a, b) || {
                    let (a, b) = (a.borrow(), b.borrow());
                    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.equals_in(y, seen))
                }
            }
            (Tuple(a), Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.equals_in(y, seen))
            }
            (Hash(a), Hash(b)) => {
                Rc::ptr_eq(a, b) || {
                    let (a, b) = (a.borrow(), b.borrow());
                    a.len() == b.len()
                        && a.iter()
                            .all(|(k, v)| b.get(k).is_some_and(|w| v.equals_in(w, seen)))
                }
            }
            (Record(a), Record(b)) => {
                Rc::ptr_eq(a, b) || {
                    let (a, b) = (a.borrow(), b.borrow());
                    a.fields.len() == b.fields.len()
                        && a.fields
                            .iter()
                            .all(|(n, v)| b.get(*n).is_some_and(|w| v.equals_in(w, seen)))
                }
            }
            (Range(a), Range(b)) => a == b,
            (Function(a), Function(b)) => Rc::ptr_eq(a, b),
            // `class`: identity; `type`: same fields (data).
            (Object(a), Object(b)) => {
                Rc::ptr_eq(a, b) || {
                    let (a, b) = (a.borrow(), b.borrow());
                    Rc::ptr_eq(&a.info, &b.info)
                        && a.info.kind != crate::ast::TypeDeclKind::Class
                        && a.fields
                            .iter()
                            .zip(&b.fields)
                            .all(|(x, y)| x.equals_in(y, seen))
                }
            }
            (Variant(a, i), Variant(b, j)) => Rc::ptr_eq(a, b) && i == j,
            (Type(a), Type(b)) => Rc::ptr_eq(a, b),
            (Enum(a), Enum(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Text for `print` and interpolation: strings without quotes.
    pub fn to_display(&self, i: &Interner) -> String {
        match self {
            Value::Str(s) => s.to_string(),
            _ => self.inspect(i),
        }
    }

    /// Appends the display of a simple value (`nil`, boolean, number) to `out`; `false` for
    /// other values.
    pub fn write_scalar(&self, out: &mut String) -> bool {
        match self {
            Value::Nil | Value::Bool(_) | Value::Int(_) | Value::Float(_) => {
                self.write_inspect(out, None, 0);
                true
            }
            _ => false,
        }
    }

    /// Source-code representation (REPL, list contents): strings in quotes.
    pub fn inspect(&self, i: &Interner) -> String {
        let mut out = String::new();
        self.write_inspect(&mut out, Some(i), 0);
        out
    }

    /// Without an interner, field names are shown by their number (`#3`).
    fn write_inspect(&self, out: &mut String, i: Option<&Interner>, depth: usize) {
        let mut cx = Inspect {
            depth,
            path: Vec::new(),
            budget: INSPECT_BUDGET,
        };
        self.write_in(out, i, &mut cx);
    }

    /// Writes the value; a container already on the current path (cyclic structure) is
    /// written `[…]`, and the total number of nodes is bounded (heavily shared structures).
    fn write_in(&self, out: &mut String, i: Option<&Interner>, cx: &mut Inspect) {
        if cx.depth > 32 || cx.budget == 0 {
            out.push('…');
            return;
        }
        cx.budget -= 1;
        let ptr = match self {
            Value::List(rc) => Some((Rc::as_ptr(rc) as *const () as usize, "[…]")),
            Value::Tuple(rc) => Some((Rc::as_ptr(rc) as *const () as usize, "(…)")),
            Value::Hash(rc) => Some((Rc::as_ptr(rc) as *const () as usize, "{…}")),
            Value::Record(rc) => Some((Rc::as_ptr(rc) as *const () as usize, "{…}")),
            Value::Object(rc) => Some((Rc::as_ptr(rc) as *const () as usize, "(…)")),
            _ => None,
        };
        if let Some((p, cycle)) = ptr {
            if cx.path.contains(&p) {
                out.push_str(cycle);
                return;
            }
            cx.path.push(p);
            cx.depth += 1;
            self.write_items(out, i, cx);
            cx.depth -= 1;
            cx.path.pop();
        } else {
            self.write_items(out, i, cx);
        }
    }

    fn write_items(&self, out: &mut String, i: Option<&Interner>, cx: &mut Inspect) {
        match self {
            Value::Nil => out.push_str("nil"),
            Value::Bool(b) => {
                let _ = write!(out, "{b}");
            }
            Value::Int(n) => {
                let _ = write!(out, "{n}");
            }
            Value::Float(f) => write_float(out, *f),
            Value::Str(s) => {
                let _ = write!(out, "{:?}", &**s);
            }
            Value::List(items) => {
                out.push('[');
                for (n, v) in items.borrow().iter().enumerate() {
                    if n > 0 {
                        out.push_str(", ");
                    }
                    v.write_in(out, i, cx);
                }
                out.push(']');
            }
            Value::Tuple(items) => {
                out.push('(');
                for (n, v) in items.iter().enumerate() {
                    if n > 0 {
                        out.push_str(", ");
                    }
                    v.write_in(out, i, cx);
                }
                if items.len() == 1 {
                    out.push(',');
                }
                out.push(')');
            }
            Value::Hash(h) => {
                out.push('{');
                for (n, (k, v)) in h.borrow().iter().enumerate() {
                    if n > 0 {
                        out.push_str(", ");
                    }
                    k.to_value().write_in(out, i, cx);
                    out.push_str(" => ");
                    v.write_in(out, i, cx);
                }
                out.push('}');
            }
            Value::Record(r) => {
                out.push('{');
                for (n, (k, v)) in r.borrow().fields.iter().enumerate() {
                    if n > 0 {
                        out.push_str(", ");
                    }
                    match i {
                        Some(i) => out.push_str(i.resolve(*k)),
                        None => {
                            let _ = write!(out, "#{}", k.0);
                        }
                    }
                    out.push_str(": ");
                    v.write_in(out, i, cx);
                }
                out.push('}');
            }
            Value::Range(r) => {
                let _ = write!(
                    out,
                    "{}{}{}",
                    r.lo,
                    if r.inclusive { ".." } else { "..." },
                    r.hi
                );
            }
            Value::Function(f) => {
                let _ = write!(out, "<function {}>", f.name());
            }
            Value::Object(o) => {
                let o = o.borrow();
                out.push_str(&o.info.name);
                out.push('(');
                for (n, (f, v)) in o.info.fields.iter().zip(&o.fields).enumerate() {
                    if n > 0 {
                        out.push_str(", ");
                    }
                    match i {
                        Some(i) => out.push_str(i.resolve(f.name)),
                        None => {
                            let _ = write!(out, "#{}", f.name.0);
                        }
                    }
                    out.push_str(": ");
                    v.write_in(out, i, cx);
                }
                out.push(')');
            }
            Value::Variant(e, idx) => {
                let name = e.variants[*idx as usize].0;
                match i {
                    Some(i) => {
                        let _ = write!(out, "{}::{}", e.name, i.resolve(name));
                    }
                    None => {
                        let _ = write!(out, "{}::#{}", e.name, name.0);
                    }
                }
            }
            Value::Type(t) => {
                let _ = write!(out, "<type {}>", t.name);
            }
            Value::Enum(e) => {
                let _ = write!(out, "<enum {}>", e.name);
            }
            Value::Uninit | Value::Iter(_) => out.push_str("<internal>"),
        }
    }
}

/// Maximum number of nodes written by one display (beyond that: `…`).
const INSPECT_BUDGET: usize = 1_000_000;

/// Display state: depth, containers on the current path, remaining nodes.
struct Inspect {
    depth: usize,
    path: Vec<usize>,
    budget: usize,
}

/// Shortest text that round-trips exactly: `3.0`, `0.1`, `1e300`, `5e-324`,
/// `1.2345678901234568e17` (scientific notation for very large and very small numbers),
/// `NaN`, `inf`. `2.0` stays `2.0` (not `2`) to tell floats from integers.
fn write_float(out: &mut String, f: f64) {
    let _ = write!(out, "{f:?}");
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = String::new();
        self.write_inspect(&mut out, None, 0);
        f.write_str(&out)
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        self.equals(other)
    }
}

/// Tags and offsets used by native code.
pub mod layout {
    pub const TAG_NIL: u8 = 0;
    pub const TAG_BOOL: u8 = 1;
    pub const TAG_INT: u8 = 2;
    pub const TAG_FLOAT: u8 = 3;
    pub const TAG_STR: u8 = 4;
    /// `Value::Uninit`: a field not initialized yet (owns nothing).
    pub const TAG_UNINIT: u8 = 15;
    /// Higher tags: the value owns a resource (reference counted).
    pub const FIRST_HEAP_TAG: u8 = 4;
    pub const PAYLOAD: i32 = 8;
    pub const SIZE: i32 = 24;
}

/// Non-recursive drop: a very deeply nested structure (list of list of list…, 300,000
/// levels) would otherwise overflow the stack while being dropped.
///
/// Only the last owner of a container takes it apart: its container elements go onto a
/// work stack (on the heap), the others are dropped in place.
impl Drop for Value {
    #[inline]
    fn drop(&mut self) {
        // Common case (number, string, shared container or one without nested containers):
        // ordinary drop, nothing to do here.
        if is_container(self) && has_nested(self) {
            drop_deep(self);
        }
    }
}

/// Takes `v` and its descendants apart without recursion.
#[inline(never)]
fn drop_deep(v: &mut Value) {
    let mut stack: Vec<Value> = Vec::new();
    take_children(v, &mut stack);
    while let Some(mut v) = stack.pop() {
        take_children(&mut v, &mut stack);
        // `v` has no elements left: dropping it does not recurse.
    }
}

/// `v` is the sole owner of a container that itself holds a container (`Rc::get_mut`:
/// uniqueness and access without a borrow check).
#[inline]
fn has_nested(v: &mut Value) -> bool {
    fn any<'a>(mut xs: impl Iterator<Item = &'a Value>) -> bool {
        xs.any(is_container)
    }
    match v {
        Value::List(rc) => Rc::get_mut(rc).is_some_and(|c| any(c.get_mut().iter())),
        Value::Tuple(rc) => Rc::get_mut(rc).is_some_and(|c| any(c.iter())),
        Value::Hash(rc) => {
            Rc::get_mut(rc).is_some_and(|c| any(c.get_mut().entries.iter().map(|(_, x)| x)))
        }
        Value::Record(rc) => {
            Rc::get_mut(rc).is_some_and(|c| any(c.get_mut().fields.iter().map(|(_, x)| x)))
        }
        Value::Object(rc) => Rc::get_mut(rc).is_some_and(|c| any(c.get_mut().fields.iter())),
        Value::Function(rc) => Rc::get_mut(rc).is_some_and(|f| match f {
            Function::Script(c) => c
                .upvals
                .iter_mut()
                .any(|u| Rc::get_mut(u).is_some_and(|u| is_container(u.get_mut()))),
            _ => false,
        }),
        Value::Iter(rc) => Rc::get_mut(rc).is_some_and(|c| match c.get_mut() {
            crate::vm::IterState::Values(items) => any(items.as_slice().iter()),
            crate::vm::IterState::List(..) => true,
            _ => false,
        }),
        _ => false,
    }
}

/// Container that may hold other values.
#[inline]
fn is_container(v: &Value) -> bool {
    matches!(
        v,
        Value::List(_)
            | Value::Tuple(_)
            | Value::Hash(_)
            | Value::Record(_)
            | Value::Object(_)
            | Value::Function(_)
            | Value::Iter(_)
    )
}

/// Removes the elements of `v` if `v` is the sole owner of a container: containers go
/// into `out`, the rest is dropped right away.
fn take_children(v: &mut Value, out: &mut Vec<Value>) {
    let mut push = |x: Value| {
        if is_container(&x) {
            out.push(x);
        }
    };
    match v {
        Value::List(rc) => {
            if let Some(cell) = Rc::get_mut(rc) {
                cell.get_mut().drain(..).for_each(&mut push);
            }
        }
        Value::Tuple(rc) => {
            if let Some(items) = Rc::get_mut(rc) {
                for x in items.iter_mut() {
                    push(std::mem::replace(x, Value::Nil));
                }
            }
        }
        Value::Hash(rc) => {
            if let Some(cell) = Rc::get_mut(rc) {
                let h = cell.get_mut();
                h.index.clear();
                h.entries.drain(..).for_each(|(_, x)| push(x));
            }
        }
        Value::Record(rc) => {
            if let Some(cell) = Rc::get_mut(rc) {
                cell.get_mut().fields.drain(..).for_each(|(_, x)| push(x));
            }
        }
        Value::Object(rc) => {
            if let Some(cell) = Rc::get_mut(rc) {
                cell.get_mut().fields.drain(..).for_each(&mut push);
            }
        }
        Value::Function(rc) => {
            if let Some(Function::Script(c)) = Rc::get_mut(rc) {
                for cell in c.upvals.iter_mut() {
                    if let Some(cell) = Rc::get_mut(cell) {
                        push(std::mem::replace(cell.get_mut(), Value::Nil));
                    }
                }
            }
        }
        Value::Iter(rc) => {
            if let Some(cell) = Rc::get_mut(rc) {
                match std::mem::replace(
                    cell.get_mut(),
                    crate::vm::IterState::Values(Vec::new().into_iter()),
                ) {
                    crate::vm::IterState::Values(items) => items.for_each(&mut push),
                    crate::vm::IterState::List(l, _) => push(Value::List(l)),
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(v: &Value) -> u8 {
        // SAFETY: `repr(C, u8)` puts the tag in the first byte.
        unsafe { *(v as *const Value as *const u8) }
    }

    fn payload<T: Copy>(v: &Value) -> T {
        // SAFETY: the variants' payload is at offset `PAYLOAD`.
        unsafe { *((v as *const Value as *const u8).add(layout::PAYLOAD as usize) as *const T) }
    }

    #[test]
    fn native_layout() {
        assert_eq!(std::mem::size_of::<Value>(), layout::SIZE as usize);
        assert_eq!(tag(&Value::Nil), layout::TAG_NIL);
        assert_eq!(tag(&Value::Bool(true)), layout::TAG_BOOL);
        assert_eq!(tag(&Value::Int(5)), layout::TAG_INT);
        assert_eq!(tag(&Value::Float(1.5)), layout::TAG_FLOAT);
        assert!(tag(&Value::str("x")) >= layout::FIRST_HEAP_TAG);
        assert_eq!(tag(&Value::str("x")), layout::TAG_STR);
        assert_eq!(tag(&Value::Uninit), layout::TAG_UNINIT);
        assert!(tag(&Value::Uninit) >= layout::FIRST_HEAP_TAG);
        assert_eq!(payload::<i64>(&Value::Int(-42)), -42);
        assert_eq!(payload::<f64>(&Value::Float(2.5)), 2.5);
        assert!(payload::<bool>(&Value::Bool(true)));
        // Zero bytes make `nil` (used to reset registers).
        let zero: Value = unsafe { std::mem::zeroed() };
        assert!(matches!(zero, Value::Nil));
    }
}
