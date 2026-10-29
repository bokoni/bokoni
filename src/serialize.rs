//! Serialization of script values with `serde` (feature `serde`), into any format the host
//! chooses (`serde_json`, `toml`…):
//!
//! ```
//! # #[cfg(feature = "serde")] {
//! let mut engine = bokoni::Engine::new();
//! let v = engine.eval("{name: \"a\", tags: [1, 2]}").unwrap();
//! let json = serde_json::to_string(&engine.serializable(&v)).unwrap();
//! assert_eq!(json, r#"{"name":"a","tags":[1,2]}"#);
//! # }
//! ```
//!
//! Lists and tuples are sequences; hashes, records and objects (their fields, in order)
//! are maps; a variant is its value for an enum with values (`enum E: str`, `A = 6`),
//! otherwise its name. Hash keys keep their type: the format decides (`serde_json`
//! turns an `int` into a string, refuses `nil`). A function, type, enum or range, and a
//! structure that contains itself, are errors.

use crate::intern::Interner;
use crate::value::{HashKey, Value};
use serde::ser::{Error, Serialize, SerializeMap, SerializeSeq, Serializer};
use std::cell::RefCell;

/// A value with what serializing it needs (the names of fields, interned).
pub struct Serializable<'a> {
    pub(crate) value: &'a Value,
    pub(crate) interner: &'a Interner,
}

impl Serialize for Serializable<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let path = RefCell::new(Vec::new());
        Node {
            value: self.value,
            interner: self.interner,
            path: &path,
        }
        .serialize(s)
    }
}

/// A value inside the one serialized; `path`: the containers it is in (addresses).
struct Node<'a> {
    value: &'a Value,
    interner: &'a Interner,
    path: &'a RefCell<Vec<usize>>,
}

impl<'a> Node<'a> {
    fn child(&self, value: &'a Value) -> Node<'a> {
        Node {
            value,
            interner: self.interner,
            path: self.path,
        }
    }
}

impl Serialize for Node<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let addr = crate::gc::addr_of_value(self.value);
        if let Some(a) = addr {
            if self.path.borrow().contains(&a) {
                return Err(S::Error::custom(
                    "cannot serialize a structure that contains itself",
                ));
            }
            self.path.borrow_mut().push(a);
        }
        // Deep nesting: the stack grows on the heap.
        let r = stacker::maybe_grow(64 * 1024, 1024 * 1024, || self.write(s));
        if addr.is_some() {
            self.path.borrow_mut().pop();
        }
        r
    }
}

impl Node<'_> {
    fn write<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let name = |sym| self.interner.resolve(sym);
        match self.value {
            Value::Nil => s.serialize_unit(),
            Value::Bool(b) => s.serialize_bool(*b),
            Value::Int(n) => s.serialize_i64(*n),
            Value::Float(f) => s.serialize_f64(*f),
            Value::Str(x) => s.serialize_str(x),
            Value::List(l) => {
                let l = l.borrow();
                let mut seq = s.serialize_seq(Some(l.len()))?;
                for v in l.iter() {
                    seq.serialize_element(&self.child(v))?;
                }
                seq.end()
            }
            Value::Tuple(t) => {
                let mut seq = s.serialize_seq(Some(t.len()))?;
                for v in t.iter() {
                    seq.serialize_element(&self.child(v))?;
                }
                seq.end()
            }
            Value::Hash(h) => {
                let h = h.borrow();
                let mut map = s.serialize_map(Some(h.len()))?;
                for (k, v) in h.iter() {
                    map.serialize_entry(&Key(k), &self.child(v))?;
                }
                map.end()
            }
            Value::Record(r) => {
                let r = r.borrow();
                let mut map = s.serialize_map(Some(r.fields.len()))?;
                for (k, v) in &r.fields {
                    map.serialize_entry(name(*k), &self.child(v))?;
                }
                map.end()
            }
            Value::Object(o) => {
                let o = o.borrow();
                let mut map = s.serialize_map(Some(o.fields.len()))?;
                for (f, v) in o.info.fields.iter().zip(&o.fields) {
                    if matches!(v, Value::Uninit) {
                        return Err(S::Error::custom(format!(
                            "field `{}` of `{}` is not initialized",
                            name(f.name),
                            o.info.name
                        )));
                    }
                    map.serialize_entry(name(f.name), &self.child(v))?;
                }
                map.end()
            }
            Value::Variant(i, e) => {
                let (vname, v) = &e.variants[*i as usize];
                if e.valued {
                    self.child(v).serialize(s)
                } else {
                    s.serialize_str(name(*vname))
                }
            }
            v => Err(S::Error::custom(format!(
                "cannot serialize a {}",
                v.type_name()
            ))),
        }
    }
}

/// A hash key, of its own type.
struct Key<'a>(&'a HashKey);

impl Serialize for Key<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            HashKey::Nil => s.serialize_unit(),
            HashKey::Bool(b) => s.serialize_bool(*b),
            HashKey::Int(n) => s.serialize_i64(*n),
            HashKey::Str(x) => s.serialize_str(x),
        }
    }
}
