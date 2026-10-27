//! Operations on values: operators, indexing, fields, built-in functions and methods.
//! Shared by the IR interpreter and by the JIT, which calls them for anything it does
//! not compile to machine instructions.

use crate::ast::BinOp;
use crate::diagnostic::Error;
use crate::intern::Symbol;
use crate::ir::Access;
use crate::span::Span;
use crate::value::{Function, HashKey, Value};
use crate::vm::{CallArgs, Runtime};
use std::rc::Rc;

pub(crate) type R<T = Value> = Result<T, Error>;

pub(crate) fn rt_err<T>(msg: impl Into<String>, span: Span) -> R<T> {
    Err(Error::runtime(msg, span))
}

pub(crate) fn type_err<T>(msg: impl Into<String>, span: Span) -> R<T> {
    Err(Error::type_error(msg, span))
}

/// Built-in functions that need the interpreter (output, calls…).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builtin {
    Print,
    TypeOf,
    Assert,
}

impl Builtin {
    pub const ALL: [(&'static str, Builtin); 3] = [
        ("print", Builtin::Print),
        ("typeof", Builtin::TypeOf),
        ("assert", Builtin::Assert),
    ];
}

macro_rules! methods {
    ($($variant:ident = $name:literal),* $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum Method { $($variant),* }
        pub(crate) const METHODS: &[(&str, Method)] = &[$(($name, Method::$variant)),*];
    };
}

methods! {
    Len = "len", IsEmpty = "is_empty", ToS = "to_s", Inspect = "inspect",
    Push = "push", Pop = "pop", First = "first", Last = "last",
    Map = "map", Filter = "filter", Each = "each", Reduce = "reduce", Find = "find",
    Any = "any", All = "all", Join = "join", Contains = "contains",
    Sum = "sum", Min = "min", Max = "max", Sort = "sort", Reverse = "reverse",
    Keys = "keys", Values = "values", Has = "has", Get = "get", Delete = "delete",
    Upper = "upper", Lower = "lower", Capitalize = "capitalize", Trim = "trim",
    Split = "split", StartsWith = "starts_with", EndsWith = "ends_with",
    Replace = "replace", Chars = "chars", ToI = "to_i", ToF = "to_f",
    ToIOpt = "to_i?", ToFOpt = "to_f?",
    Abs = "abs", Floor = "floor", Ceil = "ceil", Round = "round", Sqrt = "sqrt",
    Call = "call", ToList = "to_list", IsNil = "nil?",
}

impl Runtime {
    pub(crate) fn hash_key(&self, v: &Value, span: Span) -> R<HashKey> {
        match HashKey::from_value(v) {
            Some(k) => Ok(k),
            None => type_err(
                format!(
                    "a hash key must be nil, bool, int or str, got {}",
                    v.type_name()
                ),
                span,
            ),
        }
    }

    pub(crate) fn field(&mut self, r: &Value, name: Symbol, access: Access, span: Span) -> R {
        match r {
            Value::Record(rec) => match rec.borrow().get(name) {
                Some(v) => Ok(v.clone()),
                None => rt_err(
                    format!("the record has no field `{}`", self.name(name)),
                    span,
                ),
            },
            Value::Object(_) | Value::Variant(..) => self.object_field(r, name, access, span),
            _ if self.methods.contains_key(&name) => rt_err(
                format!("`{n}` is a method: write `.{n}()`", n = self.name(name)),
                span,
            ),
            _ => type_err(
                format!("{} has no field `{}`", r.type_name(), self.name(name)),
                span,
            ),
        }
    }

    pub(crate) fn index(&self, r: &Value, i: &Value, span: Span) -> R {
        self.index_in(r, i, span, false)
    }

    /// `r[i]?`: `nil` when the index is out of bounds (a wrong type is still an error).
    pub(crate) fn index_opt(&self, r: &Value, i: &Value, span: Span) -> R {
        self.index_in(r, i, span, true)
    }

    fn index_in(&self, r: &Value, i: &Value, span: Span, opt: bool) -> R {
        let pos = |len: usize| -> R<Option<usize>> {
            match list_pos(i, len, span)? {
                Some(p) => Ok(Some(p)),
                None if opt => Ok(None),
                None => list_index(i, len, span).map(Some),
            }
        };
        match r {
            Value::List(l) => {
                let l = l.borrow();
                Ok(pos(l.len())?.map_or(Value::Nil, |p| l[p].clone()))
            }
            Value::Hash(h) => {
                let key = self.hash_key(i, span)?;
                Ok(h.borrow().get(&key).cloned().unwrap_or(Value::Nil))
            }
            Value::Str(s) => {
                let chars: Vec<char> = s.chars().collect();
                Ok(pos(chars.len())?
                    .map_or(Value::Nil, |p| Value::Str(chars[p].to_string().into())))
            }
            Value::Tuple(t) => Ok(pos(t.len())?.map_or(Value::Nil, |p| t[p].clone())),
            _ => type_err(
                format!("{} cannot be indexed with `[…]`", r.type_name()),
                span,
            ),
        }
    }

    // ─── Operators ───────────────────────────────────────────────────────────

    pub(crate) fn binary(&self, op: BinOp, l: Value, r: Value, span: Span) -> R {
        use BinOp::*;
        use Value::{Float, Int};
        let overflow = || Error::runtime("integer overflow", span);
        Ok(match (op, &l, &r) {
            (Eq, _, _) => Value::Bool(l.equals(&r)),
            (Ne, _, _) => Value::Bool(!l.equals(&r)),

            (Add, Int(a), Int(b)) => Int(a.checked_add(*b).ok_or_else(overflow)?),
            (Sub, Int(a), Int(b)) => Int(a.checked_sub(*b).ok_or_else(overflow)?),
            (Mul, Int(a), Int(b)) => Int(a.checked_mul(*b).ok_or_else(overflow)?),
            (Div | Rem, Int(_), Int(0)) => return rt_err("division by zero", span),
            (Div, Int(a), Int(b)) => Int(a.checked_div_euclid(*b).ok_or_else(overflow)?),
            (Rem, Int(a), Int(b)) => Int(a.checked_rem_euclid(*b).ok_or_else(overflow)?),

            (Add | Sub | Mul | Div | Rem, Int(_) | Float(_), Int(_) | Float(_)) => {
                let (a, b) = (as_f64(&l), as_f64(&r));
                Float(match op {
                    Add => a + b,
                    Sub => a - b,
                    Mul => a * b,
                    Div => a / b,
                    _ => a.rem_euclid(b),
                })
            }

            (Add, Value::Str(a), Value::Str(b)) => {
                let mut s = String::with_capacity(a.len() + b.len());
                s.push_str(a);
                s.push_str(b);
                Value::Str(s.into())
            }
            (Add, Value::Str(_), _) | (Add, _, Value::Str(_)) => {
                return type_err(
                    format!(
                        "`+` between {} and {}: use interpolation \"#{{x}}\" or `.to_s()`",
                        l.type_name(),
                        r.type_name()
                    ),
                    span,
                );
            }
            (Mul, Value::Str(s), Int(n)) => {
                if *n < 0 {
                    return rt_err("string repeated a negative number of times", span);
                }
                match s.len().checked_mul(*n as usize) {
                    Some(len) if len <= MAX_ALLOC => Value::Str(s.repeat(*n as usize).into()),
                    _ => {
                        return rt_err(format!("string too long: {} × {n} bytes", s.len()), span);
                    }
                }
            }
            (Add, Value::List(a), Value::List(b)) => {
                let mut out = a.borrow().clone();
                out.extend(b.borrow().iter().cloned());
                Value::list(out)
            }

            (Lt | Le | Gt | Ge, _, _) => {
                let ord = match (&l, &r) {
                    (Int(a), Int(b)) => a.cmp(b),
                    (Int(_) | Float(_), Int(_) | Float(_)) => {
                        match as_f64(&l).partial_cmp(&as_f64(&r)) {
                            Some(o) => o,
                            None => return Ok(Value::Bool(false)), // NaN
                        }
                    }
                    (Value::Str(a), Value::Str(b)) => a.cmp(b),
                    _ => {
                        return type_err(
                            format!(
                                "`{}` cannot compare {} and {}",
                                op.as_str(),
                                l.type_name(),
                                r.type_name()
                            ),
                            span,
                        );
                    }
                };
                use std::cmp::Ordering::*;
                Value::Bool(match op {
                    Lt => ord == Less,
                    Le => ord != Greater,
                    Gt => ord == Greater,
                    _ => ord != Less,
                })
            }

            (Push, Value::List(list), _) => {
                crate::vm::before_list_write(list);
                list.borrow_mut().push(r.clone());
                l.clone()
            }

            _ => {
                return type_err(
                    format!(
                        "`{}` does not apply to {} and {}",
                        op.as_str(),
                        l.type_name(),
                        r.type_name()
                    ),
                    span,
                );
            }
        })
    }

    pub(crate) fn builtin(&mut self, b: Builtin, args: Vec<Value>, span: Span) -> R {
        match b {
            Builtin::Print => {
                let mut line = String::new();
                for (n, a) in args.iter().enumerate() {
                    if n > 0 {
                        line.push(' ');
                    }
                    line.push_str(&self.display(a, span)?);
                }
                line.push('\n');
                (self.output)(&line);
                Ok(Value::Nil)
            }
            Builtin::TypeOf => {
                arity("typeof", &args, 1, span)?;
                Ok(Value::Str(args[0].type_label().into()))
            }
            Builtin::Assert => {
                if args.is_empty() || args.len() > 2 {
                    return rt_err(
                        "`assert(condition, message?)` expects 1 or 2 arguments",
                        span,
                    );
                }
                if args[0].truthy() {
                    Ok(Value::Nil)
                } else {
                    let msg = match args.get(1) {
                        Some(m) => format!("assertion failed: {}", m.to_display(&self.interner)),
                        None => "assertion failed".to_string(),
                    };
                    rt_err(msg, span)
                }
            }
        }
    }

    // ─── Built-in methods ────────────────────────────────────────────────────

    pub(crate) fn call1(&mut self, f: &Value, a: Value, span: Span) -> R {
        self.call_pos(f, [a], span)
    }

    /// Call with positional arguments: no allocation or generic binding when `f` is a
    /// script function with exactly these parameters.
    pub(crate) fn call_pos<const N: usize>(
        &mut self,
        f: &Value,
        args: [Value; N],
        span: Span,
    ) -> R {
        if let Value::Function(func) = f
            && let Function::Script(c) = &**func
            && c.proto.params.len() == N
        {
            return self.invoke_regs(c, &args, span);
        }
        self.call_value(f, CallArgs::plain(args.into()), span)
    }

    pub(crate) fn method(&mut self, recv: Value, name: Symbol, args: Vec<Value>, span: Span) -> R {
        use Method as M;
        let Some(&(m, mname)) = self.methods.get(&name) else {
            return type_err(
                format!("{} has no method `{}`", recv.type_name(), self.name(name)),
                span,
            );
        };
        let n = |k: usize| arity(mname, &args, k, span);
        match (m, &recv) {
            (M::ToS, _) => {
                n(0)?;
                Ok(Value::Str(recv.to_display(&self.interner).into()))
            }
            (M::Inspect, _) => {
                n(0)?;
                Ok(Value::Str(recv.inspect(&self.interner).into()))
            }
            (M::Call, Value::Function(_)) => self.call_value(&recv, CallArgs::plain(args), span),
            (M::IsNil, _) => {
                n(0)?;
                Ok(Value::Bool(matches!(recv, Value::Nil)))
            }

            // ── Numbers
            (M::Abs, Value::Int(i)) => {
                n(0)?;
                match i.checked_abs() {
                    Some(v) => Ok(Value::Int(v)),
                    None => rt_err("integer overflow", span),
                }
            }
            (M::Abs, Value::Float(f)) => {
                n(0)?;
                Ok(Value::Float(f.abs()))
            }
            (M::ToF, Value::Int(i)) => {
                n(0)?;
                Ok(Value::Float(*i as f64))
            }
            (M::ToF | M::Floor | M::Ceil | M::Round | M::ToI, Value::Float(f)) => {
                n(0)?;
                let f = *f;
                Ok(match m {
                    M::ToF => Value::Float(f),
                    _ => {
                        let r = match m {
                            M::Floor => f.floor(),
                            M::Ceil => f.ceil(),
                            M::Round => f.round(),
                            _ => f.trunc(),
                        };
                        if !r.is_finite() || r.abs() > i64::MAX as f64 {
                            return rt_err("number too large for an integer", span);
                        }
                        Value::Int(r as i64)
                    }
                })
            }
            (M::ToI | M::ToIOpt | M::Floor | M::Ceil | M::Round, Value::Int(_)) => {
                n(0)?;
                Ok(recv.clone())
            }
            // `?` variants: `nil` instead of an error.
            (M::ToIOpt, Value::Float(f)) => {
                n(0)?;
                let r = f.trunc();
                Ok(if r.is_finite() && r.abs() <= i64::MAX as f64 {
                    Value::Int(r as i64)
                } else {
                    Value::Nil
                })
            }
            (M::ToFOpt, Value::Int(i)) => {
                n(0)?;
                Ok(Value::Float(*i as f64))
            }
            (M::ToFOpt, Value::Float(_)) => {
                n(0)?;
                Ok(recv.clone())
            }
            (M::ToIOpt, Value::Str(s)) => {
                n(0)?;
                Ok(s.trim().parse::<i64>().map_or(Value::Nil, Value::Int))
            }
            (M::ToFOpt, Value::Str(s)) => {
                n(0)?;
                Ok(s.trim().parse::<f64>().map_or(Value::Nil, Value::Float))
            }
            (M::Sqrt, Value::Int(_) | Value::Float(_)) => {
                n(0)?;
                Ok(Value::Float(as_f64(&recv).sqrt()))
            }

            // ── Strings
            (M::Len, Value::Str(s)) => {
                n(0)?;
                Ok(Value::Int(s.chars().count() as i64))
            }
            (M::IsEmpty, Value::Str(s)) => {
                n(0)?;
                Ok(Value::Bool(s.is_empty()))
            }
            (M::Upper | M::Lower | M::Capitalize | M::Trim, Value::Str(s)) => {
                n(0)?;
                let out = match m {
                    M::Upper => s.to_uppercase(),
                    M::Lower => s.to_lowercase(),
                    M::Trim => s.trim().to_string(),
                    _ => {
                        let mut c = s.chars();
                        match c.next() {
                            Some(f) => f.to_uppercase().chain(c).collect(),
                            None => String::new(),
                        }
                    }
                };
                Ok(Value::Str(out.into()))
            }
            (M::Contains | M::StartsWith | M::EndsWith, Value::Str(s)) => {
                n(1)?;
                let pat = expect_str(&args[0], mname, span)?;
                Ok(Value::Bool(match m {
                    M::Contains => s.contains(&*pat),
                    M::StartsWith => s.starts_with(&*pat),
                    _ => s.ends_with(&*pat),
                }))
            }
            (M::Split, Value::Str(s)) => {
                let parts: Vec<Value> = match args.as_slice() {
                    [] => s.split_whitespace().map(Value::str).collect(),
                    [sep] => {
                        let sep = expect_str(sep, mname, span)?;
                        s.split(&*sep).map(Value::str).collect()
                    }
                    _ => return rt_err("`split(separator?)` expects 0 or 1 argument", span),
                };
                Ok(Value::list(parts))
            }
            (M::Replace, Value::Str(s)) => {
                n(2)?;
                let from = expect_str(&args[0], mname, span)?;
                let to = expect_str(&args[1], mname, span)?;
                Ok(Value::Str(s.replace(&*from, &to).into()))
            }
            (M::Chars, Value::Str(s)) => {
                n(0)?;
                Ok(Value::list(
                    s.chars()
                        .map(|c| Value::Str(c.to_string().into()))
                        .collect(),
                ))
            }
            (M::ToI, Value::Str(s)) => {
                n(0)?;
                match s.trim().parse::<i64>() {
                    Ok(v) => Ok(Value::Int(v)),
                    Err(_) => rt_err(format!("{s:?} is not an integer"), span),
                }
            }
            (M::ToF, Value::Str(s)) => {
                n(0)?;
                match s.trim().parse::<f64>() {
                    Ok(v) => Ok(Value::Float(v)),
                    Err(_) => rt_err(format!("{s:?} is not a number"), span),
                }
            }

            // ── Lists
            (M::Len, Value::List(l)) => {
                n(0)?;
                Ok(Value::Int(l.borrow().len() as i64))
            }
            (M::IsEmpty, Value::List(l)) => {
                n(0)?;
                Ok(Value::Bool(l.borrow().is_empty()))
            }
            (M::Push, Value::List(l)) => {
                n(1)?;
                crate::vm::before_list_write(l);
                l.borrow_mut().push(args[0].clone());
                Ok(recv.clone())
            }
            (M::Pop, Value::List(l)) => {
                n(0)?;
                crate::vm::before_list_write(l);
                Ok(l.borrow_mut().pop().unwrap_or(Value::Nil))
            }
            (M::First, Value::List(l)) => {
                n(0)?;
                Ok(l.borrow().first().cloned().unwrap_or(Value::Nil))
            }
            (M::Last, Value::List(l)) => {
                n(0)?;
                Ok(l.borrow().last().cloned().unwrap_or(Value::Nil))
            }
            (M::Contains, Value::List(l)) => {
                n(1)?;
                Ok(Value::Bool(l.borrow().iter().any(|v| v.equals(&args[0]))))
            }
            (M::Join, Value::List(l)) => {
                let sep = match args.as_slice() {
                    [] => Rc::from(""),
                    [s] => expect_str(s, mname, span)?,
                    _ => return rt_err("`join(separator?)` expects 0 or 1 argument", span),
                };
                let parts: Vec<String> = l
                    .borrow()
                    .iter()
                    .map(|v| v.to_display(&self.interner))
                    .collect();
                Ok(Value::Str(parts.join(&sep).into()))
            }
            (M::Reverse, Value::List(l)) => {
                n(0)?;
                let mut v = l.borrow().clone();
                v.reverse();
                Ok(Value::list(v))
            }
            (M::Reverse, Value::Str(s)) => {
                n(0)?;
                Ok(Value::Str(s.chars().rev().collect::<String>().into()))
            }
            (M::Sort, Value::List(l)) => {
                n(0)?;
                let mut v = l.borrow().clone();
                let mut err = None;
                v.sort_by(|a, b| match compare(a, b) {
                    Some(o) => o,
                    None => {
                        err.get_or_insert((a.type_name(), b.type_name()));
                        std::cmp::Ordering::Equal
                    }
                });
                if let Some((a, b)) = err {
                    return type_err(format!("`sort` cannot compare {a} and {b}"), span);
                }
                Ok(Value::list(v))
            }
            (M::Sum, Value::List(l)) => {
                n(0)?;
                let items = l.borrow().clone();
                let mut acc = Value::Int(0);
                for v in items {
                    acc = self.binary(BinOp::Add, acc, v, span)?;
                }
                Ok(acc)
            }
            (M::Min | M::Max, Value::List(l)) => {
                n(0)?;
                let items = l.borrow();
                let mut best: Option<&Value> = None;
                for v in items.iter() {
                    best = match best {
                        None => Some(v),
                        Some(b) => {
                            let Some(o) = compare(v, b) else {
                                return type_err(
                                    format!(
                                        "`{mname}` cannot compare {} and {}",
                                        v.type_name(),
                                        b.type_name()
                                    ),
                                    span,
                                );
                            };
                            let better = if m == M::Min { o.is_lt() } else { o.is_gt() };
                            Some(if better { v } else { b })
                        }
                    };
                }
                Ok(best.cloned().unwrap_or(Value::Nil))
            }
            (
                M::Map | M::Filter | M::Each | M::Find | M::Any | M::All,
                Value::List(_) | Value::Range(_),
            ) => {
                n(1)?;
                let f = args[0].clone();
                let mut out = Vec::new();
                for v in iter_lazy(&recv) {
                    match m {
                        M::Map => out.push(self.call1(&f, v, span)?),
                        M::Filter => {
                            if self.call1(&f, v.clone(), span)?.truthy() {
                                out.push(v);
                            }
                        }
                        M::Each => {
                            self.call1(&f, v, span)?;
                        }
                        M::Find => {
                            if self.call1(&f, v.clone(), span)?.truthy() {
                                return Ok(v);
                            }
                        }
                        M::Any => {
                            if self.call1(&f, v, span)?.truthy() {
                                return Ok(Value::Bool(true));
                            }
                        }
                        _ => {
                            if !self.call1(&f, v, span)?.truthy() {
                                return Ok(Value::Bool(false));
                            }
                        }
                    }
                }
                Ok(match m {
                    M::Map | M::Filter => Value::list(out),
                    M::Each => recv.clone(),
                    M::Find => Value::Nil,
                    M::Any => Value::Bool(false),
                    _ => Value::Bool(true),
                })
            }
            (M::Reduce, Value::List(_) | Value::Range(_)) => {
                n(2)?;
                let f = args[1].clone();
                let mut acc = args[0].clone();
                for v in iter_lazy(&recv) {
                    acc = self.call_pos(&f, [acc, v], span)?;
                }
                Ok(acc)
            }

            // ── Ranges
            (M::Len, Value::Range(r)) => {
                n(0)?;
                Ok(Value::Int(r.len().min(i64::MAX as usize) as i64))
            }
            (M::Contains, Value::Range(r)) => {
                n(1)?;
                Ok(Value::Bool(
                    matches!(args[0], Value::Int(i) if r.contains(i)),
                ))
            }
            (M::ToList, Value::Range(_)) => {
                n(0)?;
                Ok(Value::list(iter_values(&recv, span)?))
            }

            // ── Hash
            (M::Len, Value::Hash(h)) => {
                n(0)?;
                Ok(Value::Int(h.borrow().len() as i64))
            }
            (M::IsEmpty, Value::Hash(h)) => {
                n(0)?;
                Ok(Value::Bool(h.borrow().is_empty()))
            }
            (M::Keys, Value::Hash(h)) => {
                n(0)?;
                Ok(Value::list(
                    h.borrow().iter().map(|(k, _)| k.to_value()).collect(),
                ))
            }
            (M::Values, Value::Hash(h)) => {
                n(0)?;
                Ok(Value::list(
                    h.borrow().iter().map(|(_, v)| v.clone()).collect(),
                ))
            }
            (M::Has, Value::Hash(h)) => {
                n(1)?;
                let k = self.hash_key(&args[0], span)?;
                Ok(Value::Bool(h.borrow().get(&k).is_some()))
            }
            (M::Get, Value::Hash(h)) => {
                if args.is_empty() || args.len() > 2 {
                    return rt_err("`get(key, default?)` expects 1 or 2 arguments", span);
                }
                let k = self.hash_key(&args[0], span)?;
                let found = h.borrow().get(&k).cloned();
                Ok(found.unwrap_or_else(|| args.get(1).cloned().unwrap_or(Value::Nil)))
            }
            (M::Delete, Value::Hash(h)) => {
                n(1)?;
                let k = self.hash_key(&args[0], span)?;
                Ok(h.borrow_mut().remove(&k).unwrap_or(Value::Nil))
            }
            (M::Each, Value::Hash(h)) => {
                n(1)?;
                let f = args[0].clone();
                let entries: Vec<(Value, Value)> = h
                    .borrow()
                    .iter()
                    .map(|(k, v)| (k.to_value(), v.clone()))
                    .collect();
                for (k, v) in entries {
                    self.call_pos(&f, [k, v], span)?;
                }
                Ok(recv.clone())
            }

            // ── Tuples
            (M::Len, Value::Tuple(t)) => {
                n(0)?;
                Ok(Value::Int(t.len() as i64))
            }

            _ => type_err(
                format!("{} has no method `{mname}`", recv.type_name()),
                span,
            ),
        }
    }
}

pub(crate) fn as_f64(v: &Value) -> f64 {
    match v {
        Value::Int(i) => *i as f64,
        Value::Float(f) => *f,
        _ => f64::NAN,
    }
}

pub(crate) fn compare(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
        (Value::Int(_) | Value::Float(_), Value::Int(_) | Value::Float(_)) => {
            as_f64(a).partial_cmp(&as_f64(b))
        }
        (Value::Str(x), Value::Str(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// List index; negative counts from the end.
pub(crate) fn list_index(i: &Value, len: usize, span: Span) -> R<usize> {
    match (list_pos(i, len, span)?, i) {
        (Some(p), _) => Ok(p),
        (None, Value::Int(i)) => rt_err(format!("index {i} out of bounds (length {len})"), span),
        (None, _) => unreachable!("list_pos checks the type"),
    }
}

/// Position of index `i` (negative: from the end), `None` when out of bounds.
fn list_pos(i: &Value, len: usize, span: Span) -> R<Option<usize>> {
    let Value::Int(i) = i else {
        return type_err(
            format!("an index must be an integer, got {}", i.type_name()),
            span,
        );
    };
    let pos = if *i < 0 { len as i64 + i } else { *i };
    Ok((pos >= 0 && pos < len as i64).then_some(pos as usize))
}

/// Maximum size, in bytes, of a list created in one go (`range.to_list()`) and of a
/// repeated string (`"ab" * n`): beyond it, an error rather than the process aborting
/// for lack of memory.
pub(crate) const MAX_ALLOC: usize = 1 << 31;

/// Elements of a list or range, without materializing the range.
fn iter_lazy(v: &Value) -> Box<dyn Iterator<Item = Value>> {
    match v {
        Value::List(l) => Box::new(l.borrow().clone().into_iter()),
        Value::Range(r) => Box::new(r.values().map(Value::Int)),
        _ => Box::new(std::iter::empty()),
    }
}

fn iter_values(v: &Value, span: Span) -> R<Vec<Value>> {
    if let Value::Range(r) = v
        && r.len() > MAX_ALLOC / std::mem::size_of::<Value>()
    {
        return rt_err(
            format!("range too large to turn into a list ({} elements)", r.len()),
            span,
        );
    }
    Ok(iter_lazy(v).collect())
}

fn arity(name: &str, args: &[Value], n: usize, span: Span) -> R<()> {
    if args.len() == n {
        return Ok(());
    }
    rt_err(
        format!(
            "`{name}` expects {n} argument{}, got {}",
            if n == 1 { "" } else { "s" },
            args.len()
        ),
        span,
    )
}

fn expect_str(v: &Value, method: &str, span: Span) -> R<Rc<str>> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        _ => type_err(
            format!("`{method}` expects a string, got {}", v.type_name()),
            span,
        ),
    }
}

pub(crate) fn short(s: &str) -> String {
    if s.chars().count() > 40 {
        let mut out: String = s.chars().take(37).collect();
        out.push_str("...");
        out
    } else {
        s.to_string()
    }
}
