//! `type`, `class` and `enum` at runtime: construction, fields, methods.
//!
//! - A `type` describes data: public fields, default values, equality by value. It is
//!   built from arguments (`Route(path: "/")`) or from a block (`Server do … end`).
//! - A `class` has private fields (`@x`), an `init` constructor and methods; two
//!   instances are equal only if they are the same object.
//! - An `enum` has variants (`Protocol::Tcp`) with a value (`.value`).
//!
//! Block construction: inside the block, a field name refers to the field under
//! construction (`GetNamed` / `SetNamed` instructions). A field that is read first gets
//! its default value; fields never touched get theirs after the block, once the other
//! fields are known (`port` may depend on `protocol`).

use crate::ast::{TypeDeclKind, Visibility};
use crate::intern::{FxHashMap, Interner, Symbol};
use crate::ir::{Access, Proto, Script, Shape, TypeRef};
use crate::ops::{R, rt_err, type_err};
use crate::span::Span;
use crate::value::Value;
use crate::vm::{CallArgs, Closure, Runtime};
use std::cell::RefCell;
use std::rc::Rc;

/// A declared `type`, `class` or `annotation`.
pub struct TypeInfo {
    pub name: Rc<str>,
    pub kind: TypeDeclKind,
    /// Parent class (`class B < A`); its fields come first in `fields`, and its methods
    /// are in `methods` unless redefined.
    pub parent: Option<Rc<TypeInfo>>,
    /// Traits included, by the class or its ancestors.
    pub traits: Vec<Rc<TypeInfo>>,
    /// Methods still to be defined (`abstract def`). Empty for a class that can be
    /// instantiated.
    pub required: Vec<Required>,
    /// `abstract class` (or `trait`): cannot be instantiated.
    pub is_abstract: bool,
    pub fields: Vec<FieldInfo>,
    pub methods: FxHashMap<Symbol, Rc<Proto>>,
    pub script: Rc<Script>,
    pub annotations: crate::ir::Annotations,
}

/// A method to be defined by a class: declared `abstract` by a trait or a class.
#[derive(Clone)]
pub struct Required {
    pub name: Symbol,
    /// Trait or class that requires it.
    pub by: Rc<str>,
    pub decl: Rc<crate::ast::FnDecl>,
}

#[derive(Clone)]
pub struct FieldInfo {
    pub name: Symbol,
    pub ty: Option<TypeRef>,
    /// Default value, computed by a small compiled function.
    pub default: Option<Rc<Proto>>,
    pub annotations: crate::ir::Annotations,
}

impl TypeInfo {
    pub fn field_index(&self, name: Symbol) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    /// Is it `other`, a class that inherits from it, or one that includes it (trait)?
    pub fn is_a(self: &Rc<Self>, other: &Rc<TypeInfo>) -> bool {
        if other.kind == TypeDeclKind::Trait {
            return Rc::ptr_eq(self, other) || self.traits.iter().any(|t| Rc::ptr_eq(t, other));
        }
        let mut t = self;
        loop {
            if Rc::ptr_eq(t, other) {
                return true;
            }
            match &t.parent {
                Some(p) => t = p,
                None => return false,
            }
        }
    }
}

/// Instance of a `type` or `class`: one field per `FieldInfo`, in order.
pub struct Object {
    pub info: Rc<TypeInfo>,
    pub fields: Vec<Value>,
}

pub struct EnumInfo {
    pub name: Rc<str>,
    /// (name, value) of each variant.
    pub variants: Vec<(Symbol, Value)>,
    pub methods: FxHashMap<Symbol, Rc<Proto>>,
    pub script: Rc<Script>,
    pub annotations: crate::ir::Annotations,
    /// Annotations of each variant.
    pub variant_annotations: Vec<crate::ir::Annotations>,
}

/// A `type` under construction (arguments, then block, then default values).
pub(crate) struct ConstructCtx {
    id: u32,
    info: Rc<TypeInfo>,
    values: Vec<Option<Value>>,
    /// Default value being computed (detects circular dependencies).
    computing: Vec<bool>,
}

/// Names used by the runtime, interned once.
pub(crate) struct Syms {
    pub init: Symbol,
    pub to_s: Symbol,
    pub value: Symbol,
    pub name: Symbol,
    pub values: Symbol,
    pub from: Symbol,
    pub len: Symbol,
    pub message: Symbol,
    pub error: Symbol,
}

impl Syms {
    pub fn new(i: &mut Interner) -> Self {
        Syms {
            init: i.intern("init"),
            to_s: i.intern("to_s"),
            value: i.intern("value"),
            name: i.intern("name"),
            values: i.intern("values"),
            from: i.intern("from"),
            len: i.intern("len"),
            message: i.intern("message"),
            error: i.intern("Error"),
        }
    }
}

/// Is this a method callable without arguments (a getter)?
/// Type that declares method `m`, found as `name` in `info`: the class, one of its
/// ancestors, or a trait.
pub(crate) fn method_owner(info: &Rc<TypeInfo>, name: Symbol, m: &Rc<Proto>) -> Rc<TypeInfo> {
    let has = |t: &TypeInfo| t.methods.get(&name).is_some_and(|x| Rc::ptr_eq(x, m));
    let mut owner = info.clone();
    while let Some(p) = owner.parent.clone()
        && has(&p)
    {
        owner = p;
    }
    match owner.traits.iter().find(|t| has(t)) {
        Some(t) => t.clone(),
        None => owner,
    }
}

/// Message for a `private` / `protected` method called from where it may not be (the
/// checker gives the same).
pub(crate) fn access_error(name: &str, owner: &str, kind: TypeDeclKind, vis: Visibility) -> String {
    match vis {
        Visibility::Protected => format!(
            "`{name}` is a protected method of `{owner}`: it can only be called in the methods of `{owner}` and of {}",
            if kind == TypeDeclKind::Trait {
                "the classes that include it"
            } else {
                "its subclasses"
            }
        ),
        _ => format!("`{name}` is a private method of `{owner}`: it can only be called on `self`"),
    }
}

fn is_getter(m: &Proto) -> bool {
    m.params.iter().all(|p| p.has_default)
}

impl Runtime {
    // ─── Construction ────────────────────────────────────────────────────────

    /// Builds an instance: `Server(port: 80)`, `Server do … end`, `User("x")`.
    pub(crate) fn construct(
        &mut self,
        info: &Rc<TypeInfo>,
        args: CallArgs,
        block: Option<&Closure>,
        span: Span,
    ) -> R {
        if info.is_abstract {
            let what = if info.kind == TypeDeclKind::Trait {
                "a trait"
            } else {
                "an abstract class"
            };
            return type_err(
                format!("`{}` is {what}: it cannot be instantiated", info.name),
                span,
            );
        }
        if info.kind == TypeDeclKind::Class {
            return self.instantiate_class(info, args, block, span);
        }
        if block.is_none() && args.pos.len() + args.named.len() == info.fields.len() {
            return self.construct_direct(info, args, span);
        }
        let id = self.push_ctx(info);
        let r = self.construct_inner(id, info, args, block, span);
        let ctx = self.constructs.pop().expect("construction in progress");
        r?;
        let fields = ctx
            .values
            .into_iter()
            .map(|v| v.unwrap_or(Value::Nil))
            .collect();
        Ok(Value::Object(Rc::new(RefCell::new(Object {
            info: info.clone(),
            fields,
        }))))
    }

    /// Construction without a block where each field is given once: no default value to
    /// compute, so no construction in progress to track. Same checks, in the same order,
    /// as the general path.
    fn construct_direct(&mut self, info: &Rc<TypeInfo>, args: CallArgs, span: Span) -> R {
        let CallArgs { pos, named, shapes } = args;
        let args = pos
            .into_iter()
            .map(|v| (None, v))
            .chain(named.into_iter().map(|(n, v)| (Some(n), v)));
        self.construct_fields(info, args, &shapes, span)
    }

    /// Direct construction from a call's registers (`P(x: 1, y: 2)`), without going
    /// through `CallArgs`.
    pub(crate) fn construct_regs(
        &mut self,
        info: &Rc<TypeInfo>,
        args: &[Value],
        named: &[Symbol],
        span: Span,
    ) -> R {
        let npos = args.len() - named.len();
        let args = args
            .iter()
            .enumerate()
            .map(|(k, v)| (k.checked_sub(npos).map(|j| named[j]), v.clone()));
        self.construct_fields(info, args, &[], span)
    }

    /// Positional (`None`) then named arguments, one per field.
    fn construct_fields(
        &mut self,
        info: &Rc<TypeInfo>,
        args: impl Iterator<Item = (Option<Symbol>, Value)>,
        shapes: &[Option<Rc<Shape>>],
        span: Span,
    ) -> R {
        let mut fields: Vec<Option<Value>> = Vec::with_capacity(info.fields.len());
        fields.resize(info.fields.len(), None);
        for (k, (name, v)) in args.enumerate() {
            let i = match name {
                None => k,
                Some(name) => match info.field_index(name) {
                    Some(i) if fields[i].is_none() => i,
                    _ => return self.construct_reported(info, span, name),
                },
            };
            let v = match &info.fields[i].ty {
                Some(ty) => {
                    let shape = shapes.get(k).cloned().flatten();
                    self.coerce(v, ty, shape.as_deref(), span)?
                }
                None => v,
            };
            fields[i] = Some(v);
        }
        let fields = fields
            .into_iter()
            .map(|v| v.expect("every field is given"))
            .collect();
        Ok(Value::Object(Rc::new(RefCell::new(Object {
            info: info.clone(),
            fields,
        }))))
    }

    fn construct_reported(&mut self, info: &Rc<TypeInfo>, span: Span, name: Symbol) -> R {
        let msg = match info.field_index(name) {
            Some(_) => format!("field `{}` given twice", self.name(name)),
            None => format!("`{}` has no field `{}`", info.name, self.name(name)),
        };
        rt_err(msg, span)
    }

    fn push_ctx(&mut self, info: &Rc<TypeInfo>) -> u32 {
        self.next_ctx += 1;
        let id = self.next_ctx;
        self.constructs.push(ConstructCtx {
            id,
            info: info.clone(),
            values: vec![None; info.fields.len()],
            computing: vec![false; info.fields.len()],
        });
        id
    }

    fn construct_inner(
        &mut self,
        id: u32,
        info: &Rc<TypeInfo>,
        args: CallArgs,
        block: Option<&Closure>,
        span: Span,
    ) -> R<()> {
        let CallArgs { pos, named, shapes } = args;
        if pos.len() > info.fields.len() {
            return rt_err(
                format!(
                    "`{}` has {} field{}, got {} arguments",
                    info.name,
                    info.fields.len(),
                    if info.fields.len() == 1 { "" } else { "s" },
                    pos.len()
                ),
                span,
            );
        }
        let npos = pos.len();
        for (i, v) in pos.into_iter().enumerate() {
            let shape = shapes.get(i).cloned().flatten();
            self.set_ctx_field(id, i, v, shape.as_deref(), span)?;
        }
        for (j, (name, v)) in named.into_iter().enumerate() {
            let Some(i) = info.field_index(name) else {
                return rt_err(
                    format!("`{}` has no field `{}`", info.name, self.name(name)),
                    span,
                );
            };
            if self.ctx(id).values[i].is_some() {
                return rt_err(format!("field `{}` given twice", self.name(name)), span);
            }
            let shape = shapes.get(npos + j).cloned().flatten();
            self.set_ctx_field(id, i, v, shape.as_deref(), span)?;
        }
        if let Some(c) = block {
            if let Some(p) = c.proto.params.first() {
                return rt_err(
                    "a construction block takes no parameters: the fields are directly \
                     accessible",
                    p.span,
                );
            }
            let ctx: Rc<[u32]> = c.ctx.iter().copied().chain([id]).collect();
            self.invoke(
                &c.proto,
                &c.upvals,
                &ctx,
                None,
                CallArgs::plain(Vec::new()),
                span,
            )?;
        }
        for i in 0..info.fields.len() {
            self.ensure_field(id, i, span)?;
        }
        Ok(())
    }

    fn ctx(&self, id: u32) -> &ConstructCtx {
        self.constructs
            .iter()
            .rev()
            .find(|c| c.id == id)
            .expect("construction in progress")
    }

    fn ctx_mut(&mut self, id: u32) -> Option<&mut ConstructCtx> {
        self.constructs.iter_mut().rev().find(|c| c.id == id)
    }

    /// Field `name` of one of the visible constructions in progress (`ctx`, innermost
    /// first): (construction, field index).
    pub(crate) fn ctx_lookup(&self, ctx: &[u32], name: Symbol) -> Option<(u32, u32)> {
        for &id in ctx.iter().rev() {
            let Some(c) = self.constructs.iter().rev().find(|c| c.id == id) else {
                continue;
            };
            if let Some(i) = c.info.field_index(name) {
                return Some((id, i as u32));
            }
        }
        None
    }

    pub(crate) fn construct_type_name(&self, id: u32) -> Option<Rc<str>> {
        self.constructs
            .iter()
            .find(|c| c.id == id)
            .map(|c| c.info.name.clone())
    }

    /// Stores the value of a field under construction, after checking its type.
    fn set_ctx_field(
        &mut self,
        id: u32,
        i: usize,
        v: Value,
        shape: Option<&Shape>,
        span: Span,
    ) -> R<()> {
        let info = self.ctx(id).info.clone();
        let v = match &info.fields[i].ty {
            Some(ty) => self.coerce(v, ty, shape, span)?,
            None => v,
        };
        if let Some(ctx) = self.ctx_mut(id) {
            ctx.values[i] = Some(v);
        }
        Ok(())
    }

    /// Value of a field under construction; computes its default value if needed.
    fn ensure_field(&mut self, id: u32, i: usize, span: Span) -> R {
        let ctx = self.ctx(id);
        if let Some(v) = &ctx.values[i] {
            return Ok(v.clone());
        }
        let info = ctx.info.clone();
        let field = &info.fields[i];
        if ctx.computing[i] {
            return rt_err(
                format!(
                    "circular default value for field `{}` of `{}`",
                    self.name(field.name),
                    info.name
                ),
                span,
            );
        }
        let Some(default) = &field.default else {
            return rt_err(
                format!(
                    "field `{}` is required to construct `{}`",
                    self.name(field.name),
                    info.name
                ),
                span,
            );
        };
        self.ctx_mut(id).unwrap().computing[i] = true;
        let ctx: Rc<[u32]> = Rc::from([id]);
        let v = self.invoke(default, &[], &ctx, None, CallArgs::plain(Vec::new()), span);
        if let Some(c) = self.ctx_mut(id) {
            c.computing[i] = false;
        }
        self.set_ctx_field(id, i, v?, None, span)?;
        Ok(self.ctx(id).values[i].clone().unwrap())
    }

    pub(crate) fn field_ref_get(&mut self, id: u32, i: u32, span: Span) -> R {
        if !self.constructs.iter().any(|c| c.id == id) {
            return rt_err(
                "field of a finished construction: it is no longer accessible here",
                span,
            );
        }
        self.ensure_field(id, i as usize, span)
    }

    pub(crate) fn field_ref_set(
        &mut self,
        id: u32,
        i: u32,
        v: Value,
        shape: Option<&Shape>,
        span: Span,
    ) -> R<()> {
        self.set_ctx_field(id, i as usize, v, shape, span)
    }

    fn instantiate_class(
        &mut self,
        info: &Rc<TypeInfo>,
        mut args: CallArgs,
        block: Option<&Closure>,
        span: Span,
    ) -> R {
        // Field default values (they may refer to each other).
        let id = self.push_ctx(info);
        let mut defaults = Ok(());
        for (i, f) in info.fields.iter().enumerate() {
            if f.default.is_some()
                && let Err(e) = self.ensure_field(id, i, span)
            {
                defaults = Err(e);
                break;
            }
        }
        let ctx = self.constructs.pop().unwrap();
        defaults?;
        let fields = ctx
            .values
            .into_iter()
            .map(|v| v.unwrap_or(Value::Uninit))
            .collect();
        let obj = Rc::new(RefCell::new(Object {
            info: info.clone(),
            fields,
        }));
        let this = Value::Object(obj.clone());
        if let Some(c) = block {
            let f = Value::Function(Rc::new(crate::value::Function::Script(Closure {
                proto: c.proto.clone(),
                upvals: c.upvals.clone(),
                ctx: c.ctx.clone(),
            })));
            args.pos.push(f);
        }
        match info.methods.get(&self.syms.init).cloned() {
            Some(init) => {
                self.call_method(this.clone(), &init, args, span)?;
            }
            // Without `init`: named arguments fill the fields.
            None => {
                if !args.pos.is_empty() {
                    return rt_err(
                        format!(
                            "`{}` has no `init`: use named arguments",
                            info.name
                        ),
                        span,
                    );
                }
                for (name, v) in args.named {
                    let Some(i) = info.field_index(name) else {
                        return rt_err(
                            format!("`{}` has no field `{}`", info.name, self.name(name)),
                            span,
                        );
                    };
                    let v = match &info.fields[i].ty {
                        Some(ty) => self.coerce(v, ty, None, span)?,
                        None => v,
                    };
                    obj.borrow_mut().fields[i] = v;
                }
            }
        }
        let missing = obj
            .borrow()
            .fields
            .iter()
            .position(|v| matches!(v, Value::Uninit));
        if let Some(i) = missing {
            return rt_err(
                format!(
                    "field `@{}` of `{}` is not initialized (default value or `init`)",
                    self.name(info.fields[i].name),
                    info.name
                ),
                span,
            );
        }
        Ok(this)
    }

    // ─── Fields and methods ──────────────────────────────────────────────────

    /// Declared method of a value (object or enum variant).
    pub(crate) fn user_method(&self, v: &Value, name: Symbol) -> Option<Rc<Proto>> {
        match v {
            Value::Object(o) => o.borrow().info.methods.get(&name).cloned(),
            Value::Variant(e, _) => e.methods.get(&name).cloned(),
            _ => None,
        }
    }

    /// `obj.field` on an object or a variant.
    pub(crate) fn object_field(
        &mut self,
        v: &Value,
        name: Symbol,
        access: Access,
        span: Span,
    ) -> R {
        match v {
            Value::Object(o) => {
                let (info, idx) = {
                    let o = o.borrow();
                    (o.info.clone(), o.info.field_index(name))
                };
                // `type`: public fields. `class`: only through a getter.
                if info.kind != TypeDeclKind::Class
                    && let Some(i) = idx
                {
                    return Ok(o.borrow().fields[i].clone());
                }
                if let Some(m) = info.methods.get(&name).cloned()
                    && is_getter(&m)
                {
                    self.check_access(&info, name, &m, access, span)?;
                    return self.call_method(v.clone(), &m, CallArgs::plain(Vec::new()), span);
                }
                if idx.is_some() {
                    let n = self.name(name);
                    return rt_err(
                        format!(
                            "`{n}` is a private field of `{}`: define `def {n} = @{n}` to read it",
                            info.name
                        ),
                        span,
                    );
                }
                rt_err(
                    format!("`{}` has no field `{}`", info.name, self.name(name)),
                    span,
                )
            }
            Value::Variant(e, i) => {
                let (vname, value) = &e.variants[*i as usize];
                if name == self.syms.value {
                    return Ok(value.clone());
                }
                if name == self.syms.name {
                    return Ok(Value::str(self.name(*vname)));
                }
                if let Some(m) = e.methods.get(&name).cloned()
                    && is_getter(&m)
                {
                    return self.call_method(v.clone(), &m, CallArgs::plain(Vec::new()), span);
                }
                rt_err(
                    format!(
                        "`{}` has no field `{}` (`.value`, `.name`)",
                        e.name,
                        self.name(name)
                    ),
                    span,
                )
            }
            _ => type_err(
                format!("{} has no field `{}`", v.type_label(), self.name(name)),
                span,
            ),
        }
    }

    /// `obj.field = v`: public field of a `type`, or `field=` setter of a `class`.
    pub(crate) fn object_set_field(
        &mut self,
        obj: &Value,
        name: Symbol,
        v: Value,
        shape: Option<&Shape>,
        access: Access,
        span: Span,
    ) -> R<()> {
        let Value::Object(o) = obj else {
            return type_err(
                format!("cannot assign a field on {}", obj.type_label()),
                span,
            );
        };
        let info = o.borrow().info.clone();
        let setter = self.interner.intern(&format!("{}=", self.name(name)));
        if let Some(m) = info.methods.get(&setter).cloned() {
            self.check_access(&info, setter, &m, access, span)?;
            self.call_method(obj.clone(), &m, CallArgs::plain(vec![v]), span)?;
            return Ok(());
        }
        match info.field_index(name) {
            Some(i) if info.kind != TypeDeclKind::Class => {
                let v = match &info.fields[i].ty {
                    Some(ty) => self.coerce(v, ty, shape, span)?,
                    None => v,
                };
                o.borrow_mut().fields[i] = v;
                Ok(())
            }
            Some(_) => {
                let n = self.name(name);
                rt_err(
                    format!(
                        "`{n}` is a private field of `{}`: define `def {n}=(value)` to modify it",
                        info.name
                    ),
                    span,
                )
            }
            None => rt_err(
                format!("`{}` has no field `{}`", info.name, self.name(name)),
                span,
            ),
        }
    }

    /// May method `m` (`name` in `info`) be called from `access`? A `private` method only
    /// on `self`; a `protected` one also in the methods of the type that declares it
    /// and of its subclasses.
    pub(crate) fn check_access(
        &self,
        info: &Rc<TypeInfo>,
        name: Symbol,
        m: &Rc<Proto>,
        access: Access,
        span: Span,
    ) -> R<()> {
        if m.vis == Visibility::Public || access == Access::THIS {
            return Ok(());
        }
        let owner = method_owner(info, name, m);
        if m.vis == Visibility::Protected
            && let Some(slot) = access.class()
            && let Some(Value::Type(caller)) = &self.globals.at(slot).value
            && caller.is_a(&owner)
        {
            return Ok(());
        }
        rt_err(
            access_error(self.name(name), &owner.name, owner.kind, m.vis),
            span,
        )
    }

    /// Methods of enums themselves: `Protocol.values()`, `Protocol.from("TCP")`.
    pub(crate) fn enum_method(
        &mut self,
        e: &Rc<EnumInfo>,
        name: Symbol,
        args: &[Value],
        span: Span,
    ) -> R {
        if name == self.syms.values && args.is_empty() {
            let all = (0..e.variants.len() as u32)
                .map(|i| Value::Variant(e.clone(), i))
                .collect();
            return Ok(Value::list(all));
        }
        if name == self.syms.from && args.len() == 1 {
            let found = e.variants.iter().position(|(_, v)| v.equals(&args[0]));
            return Ok(found.map_or(Value::Nil, |i| Value::Variant(e.clone(), i as u32)));
        }
        rt_err(
            format!(
                "`{}` has no method `{}` (`values()`, `from(value)`)",
                e.name,
                self.name(name)
            ),
            span,
        )
    }

    /// Text of a value for `print` and interpolation: uses `to_s` if defined.
    pub(crate) fn display(&mut self, v: &Value, span: Span) -> R<String> {
        if let Some(m) = self.user_method(v, self.syms.to_s)
            && is_getter(&m)
        {
            let s = self.call_method(v.clone(), &m, CallArgs::plain(Vec::new()), span)?;
            return Ok(s.to_display(&self.interner));
        }
        Ok(v.to_display(&self.interner))
    }
}
