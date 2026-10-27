//! Valeurs manipulées à l'exécution.
//!
//! Les types de base (`nil`, `bool`, `int`, `float`) sont stockés directement, sans
//! allocation. Les valeurs composées sont partagées par comptage de références (`Rc`) :
//! `let b = a` ne copie pas la liste, les deux noms désignent la même.

use crate::intern::{FxHashMap, Interner, Symbol};
use crate::objects::{EnumInfo, Object, TypeInfo};
use std::cell::RefCell;
use std::fmt::{self, Write};
use std::rc::Rc;

/// Disposition fixe (`repr(C, u8)`) : le code natif du JIT lit l'étiquette (octet 0) et la
/// valeur des types de base (octet 8) directement. Les types de base (`nil`, `bool`, `int`,
/// `float`) ont les étiquettes 0 à 3 et ne possèdent rien : les copier ou les écraser ne
/// demande aucun comptage de références.
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
    /// Instance d'un `type` ou d'une `class`.
    Object(Rc<RefCell<Object>>),
    /// Variante d'un `enum` (`Protocol::Tcp`) : l'enum et l'index de la variante.
    Variant(Rc<EnumInfo>, u32),
    /// Un `type` / une `class` eux-mêmes (appelables : construction).
    Type(Rc<TypeInfo>),
    /// Un `enum` lui-même (`Protocol.values()`).
    Enum(Rc<EnumInfo>),
    /// Interne : champ d'objet, variable ou paramètre pas encore initialisé.
    Uninit,
    /// Interne : itérateur d'une boucle `for`.
    Iter(Rc<RefCell<crate::vm::IterState>>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub lo: i64,
    pub hi: i64,
    pub inclusive: bool,
}

impl Range {
    /// Borne supérieure exclue.
    pub fn end(&self) -> i64 {
        if self.inclusive {
            self.hi.saturating_add(1)
        } else {
            self.hi
        }
    }

    pub fn contains(&self, n: i64) -> bool {
        n >= self.lo && n < self.end()
    }

    pub fn len(&self) -> usize {
        (self.end() - self.lo).max(0) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Champs d'un record, dans l'ordre de déclaration.
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

/// Clé de hash : seules les valeurs immuables et comparables exactement sont acceptées.
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

/// Hash qui conserve l'ordre d'insertion.
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

/// Fonction appelable.
pub enum Function {
    /// Fonction ou lambda du script.
    Script(crate::vm::Closure),
    /// Fonction Rust enregistrée avec [`Engine::register_fn`](crate::Engine::register_fn).
    Native(NativeFn),
    /// Fonction intégrée qui a besoin de l'interpréteur (`print`…).
    Builtin(crate::ops::Builtin),
}

/// Signature commune des fonctions Rust appelables depuis le script.
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

    /// Nom du type, tel qu'écrit dans les annotations.
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
            Value::Function(_) => "fonction",
            Value::Object(o) => match o.borrow().info.kind {
                crate::ast::TypeDeclKind::Class => "objet",
                _ => "valeur de type",
            },
            Value::Variant(..) => "variante d'enum",
            Value::Type(_) => "type",
            Value::Enum(_) => "enum",
            Value::Uninit | Value::Iter(_) => "valeur interne",
        }
    }

    /// Nom du type pour les messages : le nom du type utilisateur quand il y en a un.
    pub fn type_label(&self) -> String {
        match self {
            Value::Object(o) => o.borrow().info.name.to_string(),
            Value::Variant(e, _) => e.name.to_string(),
            v => v.type_name().to_string(),
        }
    }

    /// `nil` et `false` sont faux, tout le reste est vrai (comme Ruby).
    #[inline]
    pub fn truthy(&self) -> bool {
        !matches!(self, Value::Nil | Value::Bool(false))
    }

    /// Égalité structurelle (`==`). Un `int` et un `float` égaux sont égaux.
    pub fn equals(&self, other: &Value) -> bool {
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
                    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.equals(y))
                }
            }
            (Tuple(a), Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.equals(y))
            }
            (Hash(a), Hash(b)) => {
                Rc::ptr_eq(a, b) || {
                    let (a, b) = (a.borrow(), b.borrow());
                    a.len() == b.len()
                        && a.iter().all(|(k, v)| b.get(k).is_some_and(|w| v.equals(w)))
                }
            }
            (Record(a), Record(b)) => {
                Rc::ptr_eq(a, b) || {
                    let (a, b) = (a.borrow(), b.borrow());
                    a.fields.len() == b.fields.len()
                        && a.fields
                            .iter()
                            .all(|(n, v)| b.get(*n).is_some_and(|w| v.equals(w)))
                }
            }
            (Range(a), Range(b)) => a == b,
            (Function(a), Function(b)) => Rc::ptr_eq(a, b),
            // `class` : identité ; `type` : mêmes champs (données).
            (Object(a), Object(b)) => {
                Rc::ptr_eq(a, b) || {
                    let (a, b) = (a.borrow(), b.borrow());
                    Rc::ptr_eq(&a.info, &b.info)
                        && a.info.kind != crate::ast::TypeDeclKind::Class
                        && a.fields.iter().zip(&b.fields).all(|(x, y)| x.equals(y))
                }
            }
            (Variant(a, i), Variant(b, j)) => Rc::ptr_eq(a, b) && i == j,
            (Type(a), Type(b)) => Rc::ptr_eq(a, b),
            (Enum(a), Enum(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Texte pour `print` et l'interpolation : les chaînes sans guillemets.
    pub fn to_display(&self, i: &Interner) -> String {
        match self {
            Value::Str(s) => s.to_string(),
            _ => self.inspect(i),
        }
    }

    /// Ajoute l'affichage d'une valeur simple (`nil`, booléen, nombre) à `out` ; `false`
    /// pour les autres valeurs.
    pub fn write_scalar(&self, out: &mut String) -> bool {
        match self {
            Value::Nil | Value::Bool(_) | Value::Int(_) | Value::Float(_) => {
                self.write_inspect(out, None, 0);
                true
            }
            _ => false,
        }
    }

    /// Représentation en code source (REPL, contenu des listes) : chaînes entre guillemets.
    pub fn inspect(&self, i: &Interner) -> String {
        let mut out = String::new();
        self.write_inspect(&mut out, Some(i), 0);
        out
    }

    /// Sans interner, les noms de champs s'affichent par leur numéro (`#3`).
    fn write_inspect(&self, out: &mut String, i: Option<&Interner>, depth: usize) {
        if depth > 32 {
            out.push('…');
            return;
        }
        let d = depth + 1;
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
                    v.write_inspect(out, i, d);
                }
                out.push(']');
            }
            Value::Tuple(items) => {
                out.push('(');
                for (n, v) in items.iter().enumerate() {
                    if n > 0 {
                        out.push_str(", ");
                    }
                    v.write_inspect(out, i, d);
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
                    k.to_value().write_inspect(out, i, d);
                    out.push_str(" => ");
                    v.write_inspect(out, i, d);
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
                    v.write_inspect(out, i, d);
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
                let _ = write!(out, "<fonction {}>", f.name());
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
                    v.write_inspect(out, i, d);
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
            Value::Uninit | Value::Iter(_) => out.push_str("<interne>"),
        }
    }
}

/// `2.0` reste `2.0` (et non `2`) pour distinguer les décimaux des entiers.
fn write_float(out: &mut String, f: f64) {
    if f.is_finite() && f == f.trunc() && f.abs() < 1e16 {
        let _ = write!(out, "{f:.1}");
    } else {
        let _ = write!(out, "{f}");
    }
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

/// Étiquettes et positions utilisées par le code natif.
pub mod layout {
    pub const TAG_NIL: u8 = 0;
    pub const TAG_BOOL: u8 = 1;
    pub const TAG_INT: u8 = 2;
    pub const TAG_FLOAT: u8 = 3;
    /// Étiquettes supérieures : la valeur possède une ressource (comptage de références).
    pub const FIRST_HEAP_TAG: u8 = 4;
    pub const PAYLOAD: i32 = 8;
    pub const SIZE: i32 = 24;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(v: &Value) -> u8 {
        // SAFETY : `repr(C, u8)` place l'étiquette sur le premier octet.
        unsafe { *(v as *const Value as *const u8) }
    }

    fn payload<T: Copy>(v: &Value) -> T {
        // SAFETY : la valeur des variantes est à la position `PAYLOAD`.
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
        assert!(tag(&Value::Uninit) >= layout::FIRST_HEAP_TAG);
        assert_eq!(payload::<i64>(&Value::Int(-42)), -42);
        assert_eq!(payload::<f64>(&Value::Float(2.5)), 2.5);
        assert!(payload::<bool>(&Value::Bool(true)));
        // Des octets nuls forment `nil` (remise à zéro des registres).
        let zero: Value = unsafe { std::mem::zeroed() };
        assert!(matches!(zero, Value::Nil));
    }
}
