//! `type`, `class` et `enum` à l'exécution : déclarations, construction, champs, méthodes.
//!
//! - Un `type` décrit des données : champs publics, valeurs par défaut, égalité par valeur.
//!   Il se construit par arguments (`Route(path: "/")`) ou par bloc (`Server do … end`).
//! - Une `class` a des champs privés (`@x`), un constructeur `init` et des méthodes ;
//!   deux instances ne sont égales que si c'est le même objet.
//! - Un `enum` a des variantes (`Protocol::Tcp`) avec une valeur (`.value`).
//!
//! Construction par bloc : les champs sont des variables du bloc. Un champ lu dans le bloc
//! reçoit d'abord sa valeur par défaut ; les champs jamais touchés reçoivent la leur après
//! le bloc, quand les autres champs sont connus (`port` peut dépendre de `protocol`).

use crate::ast::*;
use crate::intern::{FxHashMap, Symbol};
use crate::interp::{CallArgs, Callee, Env, Flow, Runtime, Script};
use crate::span::Span;
use crate::value::{Record, Value};
use std::cell::RefCell;
use std::rc::Rc;

type R<T = Value> = Result<T, Flow>;

fn rt_err<T>(msg: impl Into<String>, span: Span) -> R<T> {
    Err(Flow::Error(crate::diagnostic::Error::runtime(msg, span)))
}

fn type_err<T>(msg: impl Into<String>, span: Span) -> R<T> {
    Err(Flow::Error(crate::diagnostic::Error::type_error(msg, span)))
}

/// Un `type`, une `class` ou une `annotation` déclarés.
pub struct TypeInfo {
    pub name: Rc<str>,
    pub kind: TypeDeclKind,
    pub fields: Vec<FieldInfo>,
    pub methods: FxHashMap<Symbol, Rc<FnDecl>>,
    /// Script et portée de la déclaration : valeurs par défaut et méthodes s'y évaluent.
    pub script: Rc<Script>,
    pub env: Env,
}

pub struct FieldInfo {
    pub name: Symbol,
    pub ty: Option<TypeExpr>,
    pub default: Option<ExprId>,
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
    pub methods: FxHashMap<Symbol, Rc<FnDecl>>,
    pub script: Rc<Script>,
    pub env: Env,
}

/// Construction en cours d'un `type` (arguments, puis bloc, puis valeurs par défaut).
pub(crate) struct ConstructCtx {
    id: u32,
    info: Rc<TypeInfo>,
    values: Vec<Option<Value>>,
    /// Valeur par défaut en cours de calcul (détection des dépendances circulaires).
    computing: Vec<bool>,
}

/// Noms utilisés par l'exécution des objets, internés une fois.
pub(crate) struct Syms {
    pub self_: Symbol,
    pub init: Symbol,
    pub to_s: Symbol,
    pub value: Symbol,
    pub name: Symbol,
    pub values: Symbol,
    pub from: Symbol,
}

impl Syms {
    pub fn new(i: &mut crate::intern::Interner) -> Self {
        Syms {
            self_: i.intern("self"),
            init: i.intern("init"),
            to_s: i.intern("to_s"),
            value: i.intern("value"),
            name: i.intern("name"),
            values: i.intern("values"),
            from: i.intern("from"),
        }
    }
}

/// Dernière expression d'un bloc (valeur de retour implicite).
pub(crate) fn tail_expr(b: &Block) -> Option<ExprId> {
    match b.stmts.last()?.kind {
        StmtKind::Expr(e) => Some(e),
        _ => None,
    }
}

impl Runtime {
    // ─── Déclarations ────────────────────────────────────────────────────────

    pub(crate) fn declare_type(&mut self, sc: &Rc<Script>, t: &TypeDecl, env: &Env) -> R<()> {
        let mut fields: Vec<FieldInfo> = Vec::new();
        let mut methods: FxHashMap<Symbol, Rc<FnDecl>> = FxHashMap::default();
        for m in &t.members {
            match m {
                Member::Field(f) => fields.push(FieldInfo {
                    name: f.name,
                    ty: f.ty.clone(),
                    default: f.default,
                }),
                Member::Method(f) => {
                    if methods.insert(f.name, Rc::new(f.clone())).is_some() {
                        return rt_err(
                            format!("méthode `{}` définie deux fois", self.name(f.name)),
                            f.name_span,
                        );
                    }
                }
                Member::Stmt(s) => {
                    return rt_err(
                        "seuls des champs et des méthodes sont permis dans un `type` / une `class` \
                         (les macros arriveront plus tard)",
                        s.span,
                    );
                }
            }
        }
        // Classe : les paramètres `@champ` de `init` déclarent aussi des champs.
        if let Some(init) = methods.get(&self.syms.init) {
            for p in init.params.iter().flatten().filter(|p| p.is_field) {
                if !fields.iter().any(|f| f.name == p.name) {
                    fields.push(FieldInfo {
                        name: p.name,
                        ty: p.ty.clone(),
                        default: None,
                    });
                }
            }
        }
        let info = TypeInfo {
            name: self.name(t.name).into(),
            kind: t.kind,
            fields,
            methods,
            script: sc.clone(),
            env: env.clone(),
        };
        env.declare(t.name, Value::Type(Rc::new(info)), false);
        Ok(())
    }

    pub(crate) fn declare_enum(&mut self, sc: &Rc<Script>, e: &EnumDecl, env: &Env) -> R<()> {
        let repr_is_str = e.repr.as_ref().is_some_and(
            |r| matches!(&r.kind, TypeKind::Named(p) if p.len() == 1 && self.name(p[0]) == "str"),
        );
        let mut variants = Vec::with_capacity(e.variants.len());
        for (i, v) in e.variants.iter().enumerate() {
            // Sans `= valeur` : le nom pour un enum `: str`, sinon l'index.
            let value = match v.value {
                Some(x) => self.eval(sc, x, env)?,
                None if repr_is_str => Value::str(self.name(v.name)),
                None => Value::Int(i as i64),
            };
            if let Some(repr) = &e.repr
                && !self.matches_type(&value, repr, env).unwrap_or(false)
            {
                return type_err(
                    format!(
                        "la valeur de `{}` doit être de type `{}`, reçu {}",
                        self.name(v.name),
                        crate::printer::type_expr(repr, &self.interner),
                        value.type_label()
                    ),
                    v.span,
                );
            }
            variants.push((v.name, value));
        }
        let info = EnumInfo {
            name: self.name(e.name).into(),
            variants,
            methods: e
                .methods
                .iter()
                .map(|m| (m.name, Rc::new(m.clone())))
                .collect(),
            script: sc.clone(),
            env: env.clone(),
        };
        env.declare(e.name, Value::Enum(Rc::new(info)), false);
        Ok(())
    }

    /// `Protocol::Tcp`
    pub(crate) fn path_value(&mut self, path: &[Symbol], env: &Env, span: Span) -> R {
        let [head, variant] = path else {
            return rt_err("seuls les chemins `Enum::Variante` sont permis", span);
        };
        match env.get(*head) {
            Some(Value::Enum(e)) => match e.variants.iter().position(|(n, _)| n == variant) {
                Some(i) => Ok(Value::Variant(e.clone(), i as u32)),
                None => rt_err(
                    format!("`{}` n'a pas de variante `{}`", e.name, self.name(*variant)),
                    span,
                ),
            },
            Some(v) => type_err(
                format!(
                    "`{}` n'est pas un enum ({})",
                    self.name(*head),
                    v.type_label()
                ),
                span,
            ),
            None => rt_err(format!("`{}` n'est pas défini", self.name(*head)), span),
        }
    }

    // ─── Construction ────────────────────────────────────────────────────────

    /// Construit une instance : `Server(port: 80)`, `Server do … end`, `User("x")`.
    pub(crate) fn construct(
        &mut self,
        info: &Rc<TypeInfo>,
        args: CallArgs,
        block: Option<(&Rc<Script>, LambdaId, &Env)>,
        span: Span,
    ) -> R {
        if info.kind == TypeDeclKind::Class {
            return self.instantiate_class(info, args, block, span);
        }
        self.next_ctx += 1;
        let id = self.next_ctx;
        self.constructs.push(ConstructCtx {
            id,
            info: info.clone(),
            values: vec![None; info.fields.len()],
            computing: vec![false; info.fields.len()],
        });
        let r = self.construct_inner(id, info, args, block, span);
        let ctx = self.constructs.pop().expect("construction en cours");
        debug_assert_eq!(ctx.id, id);
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

    fn construct_inner(
        &mut self,
        id: u32,
        info: &Rc<TypeInfo>,
        args: CallArgs,
        block: Option<(&Rc<Script>, LambdaId, &Env)>,
        span: Span,
    ) -> R<()> {
        if args.pos.len() > info.fields.len() {
            return rt_err(
                format!(
                    "`{}` a {} champ{}, reçu {} arguments",
                    info.name,
                    info.fields.len(),
                    if info.fields.len() > 1 { "s" } else { "" },
                    args.pos.len()
                ),
                span,
            );
        }
        let lits = args.lits.as_ref();
        for (i, v) in args.pos.into_iter().enumerate() {
            let lit = lits.and_then(|l| Some((&l.sc, l.pos.get(i).copied().flatten()?)));
            self.set_field(id, i, v, lit, span)?;
        }
        for (j, (name, v)) in args.named.into_iter().enumerate() {
            let Some(i) = info.field_index(name) else {
                return rt_err(
                    format!("`{}` n'a pas de champ `{}`", info.name, self.name(name)),
                    span,
                );
            };
            if self.ctx(id).values[i].is_some() {
                return rt_err(format!("champ `{}` donné deux fois", self.name(name)), span);
            }
            let lit = lits.and_then(|l| Some((&l.sc, *l.named.get(j)?)));
            self.set_field(id, i, v, lit, span)?;
        }
        if let Some((bsc, lid, benv)) = block {
            let l = bsc.program.ast.lambda(lid);
            if let Some(p) = l.params.first() {
                return rt_err(
                    "un bloc de construction ne prend pas de paramètre : les champs sont \
                     directement accessibles",
                    p.span,
                );
            }
            let scope = benv.child();
            for (i, f) in info.fields.iter().enumerate() {
                scope.declare(f.name, Value::FieldRef(id, i as u32), true);
            }
            match self.exec_block(bsc, &l.body, &scope, false) {
                Ok(_) => {}
                Err(Flow::Break(s) | Flow::Next(s)) => {
                    return rt_err("`break` / `next` en dehors d'une boucle", s);
                }
                Err(f) => return Err(f),
            }
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

    /// Range la valeur d'un champ en construction, après vérification de son type.
    fn set_field(
        &mut self,
        id: u32,
        i: usize,
        v: Value,
        lit: Option<(&Rc<Script>, ExprId)>,
        span: Span,
    ) -> R<()> {
        let info = self.ctx(id).info.clone();
        let v = match &info.fields[i].ty {
            Some(ty) => self.coerce(v, ty, &info.env, lit, span)?,
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
        let Some(default) = field.default else {
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
        // La valeur par défaut voit les autres champs (calculés à la demande).
        let env = info.env.child();
        for (j, f) in info.fields.iter().enumerate() {
            env.declare(f.name, Value::FieldRef(id, j as u32), false);
        }
        let v = self.eval(&info.script, default, &env);
        if let Some(ctx) = self.ctx_mut(id) {
            ctx.computing[i] = false;
        }
        self.set_field(id, i, v?, Some((&info.script, default)), span)?;
        Ok(self.ctx(id).values[i].clone().unwrap())
    }

    /// Lecture d'un champ dans un bloc de construction.
    pub(crate) fn field_ref_get(&mut self, id: u32, i: u32, span: Span) -> R {
        if !self.constructs.iter().any(|c| c.id == id) {
            return rt_err(
                "champ d'une construction terminée : il n'est plus accessible ici",
                span,
            );
        }
        self.ensure_field(id, i as usize, span)
    }

    /// Écriture d'un champ dans un bloc de construction.
    pub(crate) fn field_ref_set(
        &mut self,
        id: u32,
        i: u32,
        v: Value,
        lit: Option<(&Rc<Script>, ExprId)>,
        span: Span,
    ) -> R<()> {
        if !self.constructs.iter().any(|c| c.id == id) {
            return rt_err(
                "champ d'une construction terminée : il n'est plus accessible ici",
                span,
            );
        }
        self.set_field(id, i as usize, v, lit, span)
    }

    fn instantiate_class(
        &mut self,
        info: &Rc<TypeInfo>,
        mut args: CallArgs,
        block: Option<(&Rc<Script>, LambdaId, &Env)>,
        span: Span,
    ) -> R {
        let obj = Rc::new(RefCell::new(Object {
            info: info.clone(),
            fields: vec![Value::Uninit; info.fields.len()],
        }));
        let this = Value::Object(obj.clone());
        // Valeurs par défaut des champs, dans l'ordre.
        for (i, f) in info.fields.iter().enumerate() {
            if let Some(d) = f.default {
                let env = info.env.child();
                env.declare(self.syms.self_, this.clone(), false);
                let v = self.eval(&info.script, d, &env)?;
                let v = match &f.ty {
                    Some(ty) => self.coerce(v, ty, &info.env, Some((&info.script, d)), span)?,
                    None => v,
                };
                obj.borrow_mut().fields[i] = v;
            }
        }
        if let Some((bsc, lid, benv)) = block {
            args.pos.push(self.make_lambda(bsc, lid, benv));
            if let Some(l) = &mut args.lits {
                l.pos.push(None);
            }
        }
        match info.methods.get(&self.syms.init).cloned() {
            Some(init) => {
                self.call_method(this.clone(), &init, &info.script, &info.env, args, span)?;
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
                        Some(ty) => self.coerce(v, ty, &info.env, None, span)?,
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

    /// Record littéral converti vers un `type` attendu : `register({path: "/x"})`.
    fn record_to_object(
        &mut self,
        info: &Rc<TypeInfo>,
        rec: &Record,
        sc: &Rc<Script>,
        e: ExprId,
        span: Span,
    ) -> R {
        let exprs: Vec<ExprId> = match sc.program.ast.kind(e) {
            ExprKind::Record(fields) => sc.program.ast.get(fields).iter().map(|f| f.1).collect(),
            _ => Vec::new(),
        };
        let args = CallArgs {
            pos: Vec::new(),
            named: rec.fields.clone(),
            lits: Some(crate::interp::Lits {
                sc: sc.clone(),
                pos: Vec::new(),
                named: exprs,
            }),
        };
        self.construct(info, args, None, span)
    }

    // ─── Types attendus ──────────────────────────────────────────────────────

    /// `type` (ou `annotation`) nommé par une annotation de type, s'il y en a un.
    fn data_type(&self, ty: &TypeExpr, env: &Env) -> Option<Rc<TypeInfo>> {
        match &ty.kind {
            TypeKind::Named(p) if p.len() == 1 => match env.get(p[0]) {
                Some(Value::Type(info)) if info.kind != TypeDeclKind::Class => Some(info),
                _ => None,
            },
            _ => None,
        }
    }

    /// Vérifie `v` contre le type attendu, après conversion éventuelle :
    /// - `int` → `float` ;
    /// - record **littéral** (`lit` : l'expression qui a produit `v`) → `type` nominal,
    ///   y compris dans une liste littérale (`[{path: "/"}]` pour `[Route]`) et pour `T?`.
    pub(crate) fn coerce(
        &mut self,
        v: Value,
        ty: &TypeExpr,
        tenv: &Env,
        lit: Option<(&Rc<Script>, ExprId)>,
        span: Span,
    ) -> R {
        if let Some((sc, e)) = lit {
            let kind = sc.program.ast.kind(e);
            match (&kind, &ty.kind, &v) {
                (ExprKind::Record(_), TypeKind::Named(_), Value::Record(r)) => {
                    if let Some(info) = self.data_type(ty, tenv) {
                        let r = r.borrow().clone();
                        return self.record_to_object(&info, &r, sc, e, span);
                    }
                }
                (ExprKind::Record(_), TypeKind::Optional(inner), Value::Record(_)) => {
                    return self.coerce(v, inner, tenv, lit, span);
                }
                (ExprKind::Record(_), TypeKind::Union(items), Value::Record(r)) => {
                    for it in items {
                        if let Some(info) = self.data_type(it, tenv) {
                            let r = r.borrow().clone();
                            if let Ok(obj) = self.record_to_object(&info, &r, sc, e, span) {
                                return Ok(obj);
                            }
                        }
                    }
                }
                (ExprKind::List(items), TypeKind::List(inner), Value::List(l)) => {
                    let exprs = sc.program.ast.get(*items).to_vec();
                    let elems = l.borrow().clone();
                    let mut out = Vec::with_capacity(elems.len());
                    for (x, ex) in elems.into_iter().zip(exprs) {
                        out.push(self.coerce(x, inner, tenv, Some((sc, ex)), span)?);
                    }
                    return Ok(Value::list(out));
                }
                _ => {}
            }
        }
        self.check_type(v, ty, tenv, span)
    }

    // ─── Champs et méthodes ──────────────────────────────────────────────────

    /// Appelle une méthode déclarée (`type`, `class`, `enum`) avec `self` = `this`.
    pub(crate) fn call_method(
        &mut self,
        this: Value,
        m: &Rc<FnDecl>,
        script: &Rc<Script>,
        env: &Env,
        args: CallArgs,
        span: Span,
    ) -> R {
        let name = self.name(m.name).to_string();
        let callee = Callee {
            name: &name,
            script,
            params: m.params.as_deref().unwrap_or(&[]),
            body: &m.body,
            ret: m.ret.as_ref(),
            env,
            lenient: false,
            this: Some(this),
        };
        self.invoke(callee, args, span)
    }

    /// Méthode déclarée d'une valeur, avec le script et la portée où l'exécuter.
    pub(crate) fn user_method(
        &self,
        v: &Value,
        name: Symbol,
    ) -> Option<(Rc<FnDecl>, Rc<Script>, Env)> {
        match v {
            Value::Object(o) => {
                let o = o.borrow();
                let m = o.info.methods.get(&name)?;
                Some((m.clone(), o.info.script.clone(), o.info.env.clone()))
            }
            Value::Variant(e, _) => {
                let m = e.methods.get(&name)?;
                Some((m.clone(), e.script.clone(), e.env.clone()))
            }
            _ => None,
        }
    }

    /// Une méthode appelable sans argument (getter) ?
    fn is_getter(m: &FnDecl) -> bool {
        m.params
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .all(|p| p.default.is_some())
    }

    /// `obj.champ` sur un objet, une variante ou un enum.
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
                    && Self::is_getter(&m)
                {
                    return self.call_method(
                        v.clone(),
                        &m,
                        &info.script,
                        &info.env,
                        CallArgs::plain(Vec::new()),
                        span,
                    );
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
                    && Self::is_getter(&m)
                {
                    let (sc, env) = (e.script.clone(), e.env.clone());
                    return self.call_method(
                        v.clone(),
                        &m,
                        &sc,
                        &env,
                        CallArgs::plain(Vec::new()),
                        span,
                    );
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
        lit: Option<(&Rc<Script>, ExprId)>,
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
            let args = CallArgs::plain(vec![v]);
            self.call_method(obj.clone(), &m, &info.script, &info.env, args, span)?;
            return Ok(());
        }
        match info.field_index(name) {
            Some(i) if info.kind != TypeDeclKind::Class => {
                let v = match &info.fields[i].ty {
                    Some(ty) => self.coerce(v, ty, &info.env, lit, span)?,
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

    /// `self` dans une méthode.
    pub(crate) fn this(&self, env: &Env, span: Span) -> R {
        match env.get(self.syms.self_) {
            Some(v) => Ok(v),
            None => rt_err(
                "`self` et `@champ` ne s'utilisent que dans une méthode",
                span,
            ),
        }
    }

    fn ivar_slot(&self, env: &Env, name: Symbol, span: Span) -> R<(Rc<RefCell<Object>>, usize)> {
        let this = self.this(env, span)?;
        let Value::Object(o) = this else {
            return rt_err(
                format!("`@{}` : `self` n'a pas de champs ici", self.name(name)),
                span,
            );
        };
        let idx = o.borrow().info.field_index(name);
        match idx {
            Some(i) => Ok((o, i)),
            None => {
                let tname = o.borrow().info.name.clone();
                rt_err(
                    format!("`{tname}` n'a pas de champ `@{}`", self.name(name)),
                    span,
                )
            }
        }
    }

    /// `@champ`
    pub(crate) fn ivar_get(&self, env: &Env, name: Symbol, span: Span) -> R {
        let (o, i) = self.ivar_slot(env, name, span)?;
        let v = o.borrow().fields[i].clone();
        match v {
            Value::Uninit => rt_err(
                format!("`@{}` est lu avant d'être initialisé", self.name(name)),
                span,
            ),
            v => Ok(v),
        }
    }

    /// `@champ = v`
    pub(crate) fn ivar_set(
        &mut self,
        env: &Env,
        name: Symbol,
        v: Value,
        lit: Option<(&Rc<Script>, ExprId)>,
        span: Span,
    ) -> R<()> {
        let (o, i) = self.ivar_slot(env, name, span)?;
        let info = o.borrow().info.clone();
        let v = match &info.fields[i].ty {
            Some(ty) => self.coerce(v, ty, &info.env, lit, span)?,
            None => v,
        };
        o.borrow_mut().fields[i] = v;
        Ok(())
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

    /// Nom du type en cours de construction (pour les messages d'erreur).
    pub(crate) fn construct_name(&self) -> Option<Rc<str>> {
        self.constructs.last().map(|c| c.info.name.clone())
    }

    /// Texte d'une valeur pour `print` et l'interpolation : utilise `to_s` s'il est défini.
    pub(crate) fn display(&mut self, v: &Value, span: Span) -> R<String> {
        if let Some((m, sc, env)) = self.user_method(v, self.syms.to_s)
            && Self::is_getter(&m)
        {
            let s =
                self.call_method(v.clone(), &m, &sc, &env, CallArgs::plain(Vec::new()), span)?;
            return Ok(s.to_display(&self.interner));
        }
        Ok(v.to_display(&self.interner))
    }
}
