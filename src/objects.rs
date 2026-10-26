//! `type`, `class` et `enum` à l'exécution : construction, champs, méthodes.
//!
//! - Un `type` décrit des données : champs publics, valeurs par défaut, égalité par valeur.
//!   Il se construit par arguments (`Route(path: "/")`) ou par bloc (`Server do … end`).
//! - Une `class` a des champs privés (`@x`), un constructeur `init` et des méthodes ;
//!   deux instances ne sont égales que si c'est le même objet.
//! - Un `enum` a des variantes (`Protocol::Tcp`) avec une valeur (`.value`).
//!
//! Construction par bloc : dans le bloc, un nom de champ désigne le champ en construction
//! (instructions `GetNamed` / `SetNamed`). Un champ lu reçoit d'abord sa valeur par
//! défaut ; les champs jamais touchés reçoivent la leur après le bloc, quand les autres
//! champs sont connus (`port` peut dépendre de `protocol`).

use crate::ast::TypeDeclKind;
use crate::intern::{FxHashMap, Interner, Symbol};
use crate::ir::{Proto, Script, Shape, TypeRef};
use crate::ops::{R, rt_err, type_err};
use crate::span::Span;
use crate::value::Value;
use crate::vm::{CallArgs, Closure, Runtime};
use std::cell::RefCell;
use std::rc::Rc;

/// Un `type`, une `class` ou une `annotation` déclarés.
pub struct TypeInfo {
    pub name: Rc<str>,
    pub kind: TypeDeclKind,
    pub fields: Vec<FieldInfo>,
    pub methods: FxHashMap<Symbol, Rc<Proto>>,
    pub script: Rc<Script>,
}

pub struct FieldInfo {
    pub name: Symbol,
    pub ty: Option<TypeRef>,
    /// Valeur par défaut, calculée par une petite fonction compilée.
    pub default: Option<Rc<Proto>>,
}

impl TypeInfo {
    pub fn field_index(&self, name: Symbol) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }
}

/// Instance d'un `type` ou d'une `class` : un champ par `FieldInfo`, dans l'ordre.
pub struct Object {
    pub info: Rc<TypeInfo>,
    pub fields: Vec<Value>,
}

pub struct EnumInfo {
    pub name: Rc<str>,
    /// (nom, valeur) de chaque variante.
    pub variants: Vec<(Symbol, Value)>,
    pub methods: FxHashMap<Symbol, Rc<Proto>>,
    pub script: Rc<Script>,
}

/// Construction en cours d'un `type` (arguments, puis bloc, puis valeurs par défaut).
pub(crate) struct ConstructCtx {
    id: u32,
    info: Rc<TypeInfo>,
    values: Vec<Option<Value>>,
    /// Valeur par défaut en cours de calcul (détection des dépendances circulaires).
    computing: Vec<bool>,
}

/// Noms utilisés par l'exécution, internés une fois.
pub(crate) struct Syms {
    pub init: Symbol,
    pub to_s: Symbol,
    pub value: Symbol,
    pub name: Symbol,
    pub values: Symbol,
    pub from: Symbol,
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
        }
    }
}

/// Une méthode appelable sans argument (getter) ?
fn is_getter(m: &Proto) -> bool {
    m.params.iter().all(|p| p.has_default)
}

impl Runtime {
    // ─── Construction ────────────────────────────────────────────────────────

    /// Construit une instance : `Server(port: 80)`, `Server do … end`, `User("x")`.
    pub(crate) fn construct(
        &mut self,
        info: &Rc<TypeInfo>,
        args: CallArgs,
        block: Option<&Closure>,
        span: Span,
    ) -> R {
        if info.kind == TypeDeclKind::Class {
            return self.instantiate_class(info, args, block, span);
        }
        let id = self.push_ctx(info);
        let r = self.construct_inner(id, info, args, block, span);
        let ctx = self.constructs.pop().expect("construction en cours");
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
                    "`{}` a {} champ{}, reçu {} arguments",
                    info.name,
                    info.fields.len(),
                    if info.fields.len() > 1 { "s" } else { "" },
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
                    format!("`{}` n'a pas de champ `{}`", info.name, self.name(name)),
                    span,
                );
            };
            if self.ctx(id).values[i].is_some() {
                return rt_err(format!("champ `{}` donné deux fois", self.name(name)), span);
            }
            let shape = shapes.get(npos + j).cloned().flatten();
            self.set_ctx_field(id, i, v, shape.as_deref(), span)?;
        }
        if let Some(c) = block {
            if let Some(p) = c.proto.params.first() {
                return rt_err(
                    "un bloc de construction ne prend pas de paramètre : les champs sont \
                     directement accessibles",
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
            .expect("construction en cours")
    }

    fn ctx_mut(&mut self, id: u32) -> Option<&mut ConstructCtx> {
        self.constructs.iter_mut().rev().find(|c| c.id == id)
    }

    /// Champ `name` d'une des constructions en cours visibles (`ctx`, la plus proche
    /// d'abord) : (construction, index du champ).
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

    /// Range la valeur d'un champ en construction, après vérification de son type.
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

    /// Valeur d'un champ en construction ; calcule sa valeur par défaut si besoin.
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
                    "valeur par défaut circulaire pour le champ `{}` de `{}`",
                    self.name(field.name),
                    info.name
                ),
                span,
            );
        }
        let Some(default) = &field.default else {
            return rt_err(
                format!(
                    "champ `{}` obligatoire pour construire `{}`",
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
                "champ d'une construction terminée : il n'est plus accessible ici",
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
        // Valeurs par défaut des champs (elles peuvent se référencer entre elles).
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
            // Sans `init` : les arguments nommés remplissent les champs.
            None => {
                if !args.pos.is_empty() {
                    return rt_err(
                        format!(
                            "`{}` n'a pas de `init` : utiliser des arguments nommés",
                            info.name
                        ),
                        span,
                    );
                }
                for (name, v) in args.named {
                    let Some(i) = info.field_index(name) else {
                        return rt_err(
                            format!("`{}` n'a pas de champ `{}`", info.name, self.name(name)),
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
                    "le champ `@{}` de `{}` n'est pas initialisé (valeur par défaut ou `init`)",
                    self.name(info.fields[i].name),
                    info.name
                ),
                span,
            );
        }
        Ok(this)
    }

    // ─── Champs et méthodes ──────────────────────────────────────────────────

    /// Méthode déclarée d'une valeur (objet ou variante d'enum).
    pub(crate) fn user_method(&self, v: &Value, name: Symbol) -> Option<Rc<Proto>> {
        match v {
            Value::Object(o) => o.borrow().info.methods.get(&name).cloned(),
            Value::Variant(e, _) => e.methods.get(&name).cloned(),
            _ => None,
        }
    }

    /// `obj.champ` sur un objet ou une variante.
    pub(crate) fn object_field(&mut self, v: &Value, name: Symbol, span: Span) -> R {
        match v {
            Value::Object(o) => {
                let (info, idx) = {
                    let o = o.borrow();
                    (o.info.clone(), o.info.field_index(name))
                };
                // `type` : champs publics. `class` : uniquement par un getter.
                if info.kind != TypeDeclKind::Class
                    && let Some(i) = idx
                {
                    return Ok(o.borrow().fields[i].clone());
                }
                if let Some(m) = info.methods.get(&name).cloned()
                    && is_getter(&m)
                {
                    return self.call_method(v.clone(), &m, CallArgs::plain(Vec::new()), span);
                }
                if idx.is_some() {
                    let n = self.name(name);
                    return rt_err(
                        format!(
                            "`{n}` est un champ privé de `{}` : définir `def {n} = @{n}` pour le lire",
                            info.name
                        ),
                        span,
                    );
                }
                rt_err(
                    format!("`{}` n'a pas de champ `{}`", info.name, self.name(name)),
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
                        "`{}` n'a pas de champ `{}` (`.value`, `.name`)",
                        e.name,
                        self.name(name)
                    ),
                    span,
                )
            }
            _ => type_err(
                format!("{} n'a pas de champ `{}`", v.type_label(), self.name(name)),
                span,
            ),
        }
    }

    /// `obj.champ = v` : champ public d'un `type`, ou setter `champ=` d'une `class`.
    pub(crate) fn object_set_field(
        &mut self,
        obj: &Value,
        name: Symbol,
        v: Value,
        shape: Option<&Shape>,
        span: Span,
    ) -> R<()> {
        let Value::Object(o) = obj else {
            return type_err(
                format!("impossible d'assigner un champ sur {}", obj.type_label()),
                span,
            );
        };
        let info = o.borrow().info.clone();
        let setter = self.interner.intern(&format!("{}=", self.name(name)));
        if let Some(m) = info.methods.get(&setter).cloned() {
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
                        "`{n}` est un champ privé de `{}` : définir `def {n}=(valeur)` pour le modifier",
                        info.name
                    ),
                    span,
                )
            }
            None => rt_err(
                format!("`{}` n'a pas de champ `{}`", info.name, self.name(name)),
                span,
            ),
        }
    }

    /// Méthodes des enums eux-mêmes : `Protocol.values()`, `Protocol.from("TCP")`.
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
                "`{}` n'a pas de méthode `{}` (`values()`, `from(valeur)`)",
                e.name,
                self.name(name)
            ),
            span,
        )
    }

    /// Texte d'une valeur pour `print` et l'interpolation : utilise `to_s` s'il est défini.
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
