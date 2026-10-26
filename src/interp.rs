//! Interpréteur : exécute l'AST directement (v0.1).
//!
//! - Les portées sont des listes chaînées (`Env`) ; un bloc ne crée de portée que s'il
//!   déclare quelque chose (`let`, `def`).
//! - Une fonction du script est une [`Closure`] : son script (`Rc`), sa déclaration et la
//!   portée où elle a été définie. Grâce à l'AST compact, rien n'est recopié.
//! - `return`, `break` et `next` remontent comme des [`Flow`], pas comme des erreurs.

use crate::ast::*;
use crate::diagnostic::Error;
use crate::intern::{FxHashMap, Interner, Symbol};
use crate::printer;
use crate::span::Span;
use crate::value::{Function, HashKey, HashMap, Range, Record, Value};
use std::cell::RefCell;
use std::rc::Rc;

/// Un programme analysé, avec son source (pour les messages d'erreur).
pub struct Script {
    pub program: Program,
    pub src: Rc<str>,
    pub name: Rc<str>,
}

/// Fonction ou lambda du script.
pub struct Closure {
    pub name: Rc<str>,
    pub script: Rc<Script>,
    pub body: Body,
    pub env: Env,
}

pub enum Body {
    Def(Rc<FnDecl>),
    Lambda(LambdaId),
}

// ─── Portées ────────────────────────────────────────────────────────────────

struct Binding {
    name: Symbol,
    value: Value,
    mutable: bool,
}

struct Scope {
    vars: RefCell<Vec<Binding>>,
    parent: Option<Env>,
}

/// Portée de variables. Cloner un `Env` partage la même portée.
#[derive(Clone)]
pub struct Env(Rc<Scope>);

pub enum AssignError {
    NotFound,
    Immutable,
}

impl Env {
    pub fn root() -> Env {
        Env(Rc::new(Scope {
            vars: RefCell::new(Vec::new()),
            parent: None,
        }))
    }

    pub fn child(&self) -> Env {
        Env(Rc::new(Scope {
            vars: RefCell::new(Vec::new()),
            parent: Some(self.clone()),
        }))
    }

    pub fn get(&self, name: Symbol) -> Option<Value> {
        let mut scope = &self.0;
        loop {
            if let Some(b) = scope.vars.borrow().iter().rev().find(|b| b.name == name) {
                return Some(b.value.clone());
            }
            scope = &scope.parent.as_ref()?.0;
        }
    }

    /// Déclare dans cette portée (remplace une déclaration du même nom dans la même portée).
    pub fn declare(&self, name: Symbol, value: Value, mutable: bool) {
        let mut vars = self.0.vars.borrow_mut();
        let binding = Binding {
            name,
            value,
            mutable,
        };
        match vars.iter_mut().find(|b| b.name == name) {
            Some(b) => *b = binding,
            None => vars.push(binding),
        }
    }

    pub fn assign(&self, name: Symbol, value: Value) -> Result<(), AssignError> {
        let mut scope = &self.0;
        loop {
            if let Some(b) = scope.vars.borrow_mut().iter_mut().find(|b| b.name == name) {
                if !b.mutable {
                    return Err(AssignError::Immutable);
                }
                b.value = value;
                return Ok(());
            }
            match &scope.parent {
                Some(p) => scope = &p.0,
                None => return Err(AssignError::NotFound),
            }
        }
    }

    /// Vide la portée (casse les cycles portée → fonction → portée à la destruction).
    pub fn clear(&self) {
        let vars = std::mem::take(&mut *self.0.vars.borrow_mut());
        drop(vars);
    }
}

// ─── Flot de contrôle ───────────────────────────────────────────────────────

/// Ce qui interrompt l'évaluation normale.
pub enum Flow {
    Return(Value),
    Break(Span),
    Next(Span),
    Error(Error),
}

impl From<Error> for Flow {
    fn from(e: Error) -> Flow {
        Flow::Error(e)
    }
}

type R<T = Value> = Result<T, Flow>;

/// Arguments d'un appel : positionnels, puis nommés.
type Args = (Vec<Value>, Vec<(Symbol, Value)>);

const RED_ZONE: usize = 128 * 1024;
const STACK_GROW: usize = 2 * 1024 * 1024;

fn rt_err<T>(msg: impl Into<String>, span: Span) -> R<T> {
    Err(Flow::Error(Error::runtime(msg, span)))
}

fn type_err<T>(msg: impl Into<String>, span: Span) -> R<T> {
    Err(Flow::Error(Error::type_error(msg, span)))
}

// ─── Fonctions et méthodes intégrées ────────────────────────────────────────

/// Fonctions intégrées qui ont besoin de l'interpréteur (sortie, appels…).
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
        enum Method { $($variant),* }
        const METHODS: &[(&str, Method)] = &[$(($name, Method::$variant)),*];
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
    Abs = "abs", Floor = "floor", Ceil = "ceil", Round = "round", Sqrt = "sqrt",
    Call = "call", ToList = "to_list",
}

// ─── Runtime ────────────────────────────────────────────────────────────────

/// État de l'interpréteur : noms, variables globales, sortie de `print`.
pub struct Runtime {
    pub interner: Interner,
    pub globals: Env,
    pub output: Box<dyn FnMut(&str)>,
    /// Profondeur d'appels maximale (protège la pile contre la récursion infinie).
    pub max_depth: usize,
    depth: usize,
    methods: FxHashMap<Symbol, Method>,
    /// Chaînes littérales déjà converties en `Rc<str>`.
    strings: FxHashMap<Symbol, Rc<str>>,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl Runtime {
    pub fn new() -> Self {
        let mut interner = Interner::new();
        let methods = METHODS
            .iter()
            .map(|&(name, m)| (interner.intern(name), m))
            .collect();
        let globals = Env::root();
        for (name, b) in Builtin::ALL {
            let sym = interner.intern(name);
            globals.declare(sym, Value::Function(Rc::new(Function::Builtin(b))), false);
        }
        Runtime {
            interner,
            globals,
            output: Box::new(|s| print!("{s}")),
            max_depth: 1000,
            depth: 0,
            methods,
            strings: FxHashMap::default(),
        }
    }

    /// Exécute un script au niveau global : ses `let` et `def` restent visibles ensuite.
    pub fn run(&mut self, sc: &Rc<Script>) -> Result<Value, Error> {
        self.depth = 0;
        let globals = self.globals.clone();
        let r = self.exec_block(sc, &sc.program.body, &globals, false);
        let r = match r {
            Ok(v) | Err(Flow::Return(v)) => Ok(v),
            Err(Flow::Break(s)) => Err(Error::runtime("`break` en dehors d'une boucle", s)),
            Err(Flow::Next(s)) => Err(Error::runtime("`next` en dehors d'une boucle", s)),
            Err(Flow::Error(e)) => Err(e),
        };
        r.map_err(|e| attach_src(e, sc))
    }

    fn name(&self, s: Symbol) -> &str {
        self.interner.resolve(s)
    }

    fn lit_str(&mut self, s: Symbol) -> Rc<str> {
        if let Some(rc) = self.strings.get(&s) {
            return rc.clone();
        }
        let rc: Rc<str> = self.interner.resolve(s).into();
        self.strings.insert(s, rc.clone());
        rc
    }

    // ─── Blocs ───────────────────────────────────────────────────────────────

    /// Exécute un bloc ; sa valeur est celle de la dernière instruction (`nil` si vide).
    fn exec_block(&mut self, sc: &Rc<Script>, b: &Block, env: &Env, new_scope: bool) -> R {
        let declares = b
            .stmts
            .iter()
            .any(|s| matches!(s.kind, StmtKind::Let(_) | StmtKind::Def(_)));
        let local;
        let env = if new_scope && declares {
            local = env.child();
            &local
        } else {
            env
        };
        // Les `def` d'un bloc sont visibles dans tout le bloc (appel avant définition,
        // récursion mutuelle).
        for s in &b.stmts {
            if let StmtKind::Def(f) = &s.kind {
                let closure = Closure {
                    name: self.name(f.name).into(),
                    script: sc.clone(),
                    body: Body::Def(Rc::new((**f).clone())),
                    env: env.clone(),
                };
                env.declare(
                    f.name,
                    Value::Function(Rc::new(Function::Script(closure))),
                    false,
                );
            }
        }
        let mut last = Value::Nil;
        for s in &b.stmts {
            last = self.exec_stmt(sc, s, env)?;
        }
        Ok(last)
    }

    fn exec_stmt(&mut self, sc: &Rc<Script>, s: &Stmt, env: &Env) -> R {
        match &s.kind {
            StmtKind::Expr(e) => self.eval(sc, *e, env),
            StmtKind::Let(l) => {
                let mut v = self.eval(sc, l.value, env)?;
                if let Some(ty) = &l.ty {
                    v = self.check_type(v, ty, sc.program.ast.span(l.value))?;
                }
                self.bind_pattern(&l.pattern, v, env)?;
                Ok(Value::Nil)
            }
            StmtKind::Def(_) => Ok(Value::Nil),
            StmtKind::Type(t) => rt_err(
                format!(
                    "`{}` : les déclarations `type` / `class` / `annotation` arrivent en v0.2",
                    self.name(t.name)
                ),
                t.name_span,
            ),
            StmtKind::Enum(e) => rt_err(
                format!("`{}` : les `enum` arrivent en v0.2", self.name(e.name)),
                e.name_span,
            ),
        }
    }

    fn bind_pattern(&mut self, p: &Pattern, v: Value, env: &Env) -> R<()> {
        match p {
            Pattern::Name(name, _) => {
                env.declare(*name, v, true);
                Ok(())
            }
            Pattern::Tuple(items, span) => {
                let values: Vec<Value> = match &v {
                    Value::Tuple(t) => t.to_vec(),
                    Value::List(l) => l.borrow().clone(),
                    _ => return type_err(format!("tuple attendu, reçu {}", v.type_name()), *span),
                };
                if values.len() != items.len() {
                    return rt_err(
                        format!("{} valeurs attendues, reçu {}", items.len(), values.len()),
                        *span,
                    );
                }
                for (p, v) in items.iter().zip(values) {
                    self.bind_pattern(p, v, env)?;
                }
                Ok(())
            }
            Pattern::Record(names, span) => {
                let Value::Record(r) = &v else {
                    return type_err(format!("record attendu, reçu {}", v.type_name()), *span);
                };
                for (name, nspan) in names {
                    let field = r.borrow().get(*name).cloned();
                    match field {
                        Some(f) => env.declare(*name, f, true),
                        None => {
                            return rt_err(
                                format!("le record n'a pas de champ `{}`", self.name(*name)),
                                *nspan,
                            );
                        }
                    }
                }
                Ok(())
            }
            Pattern::List { items, rest, span } => {
                let Value::List(l) = &v else {
                    return type_err(format!("liste attendue, reçu {}", v.type_name()), *span);
                };
                let values = l.borrow().clone();
                if values.len() < items.len() || (rest.is_none() && values.len() != items.len()) {
                    return rt_err(
                        format!(
                            "{}{} éléments attendus, reçu {}",
                            if rest.is_some() { "au moins " } else { "" },
                            items.len(),
                            values.len()
                        ),
                        *span,
                    );
                }
                let mut it = values.into_iter();
                for p in items {
                    self.bind_pattern(p, it.next().unwrap(), env)?;
                }
                if let Some((name, _)) = rest {
                    env.declare(*name, Value::list(it.collect()), true);
                }
                Ok(())
            }
        }
    }

    // ─── Expressions ─────────────────────────────────────────────────────────

    /// Évalue une expression. Comme dans le parseur, la récursion passe sur une pile
    /// allouée sur le tas quand la pile du thread est presque pleine.
    fn eval(&mut self, sc: &Rc<Script>, e: ExprId, env: &Env) -> R {
        stacker::maybe_grow(RED_ZONE, STACK_GROW, || self.eval_inner(sc, e, env))
    }

    fn eval_inner(&mut self, sc: &Rc<Script>, e: ExprId, env: &Env) -> R {
        let ast = &sc.program.ast;
        let span = ast.span(e);
        match ast.kind(e) {
            ExprKind::Int(n) => Ok(Value::Int(n)),
            ExprKind::Float(f) => Ok(Value::Float(f)),
            ExprKind::Str(s) => Ok(Value::Str(self.lit_str(s))),
            ExprKind::Bool(b) => Ok(Value::Bool(b)),
            ExprKind::Nil => Ok(Value::Nil),
            ExprKind::Interp(parts) => {
                let mut out = String::new();
                for part in ast.get(parts) {
                    match *part {
                        StrPart::Lit(s) => out.push_str(self.name(s)),
                        StrPart::Expr(e) => {
                            let v = self.eval(sc, e, env)?;
                            out.push_str(&v.to_display(&self.interner));
                        }
                    }
                }
                Ok(Value::Str(out.into()))
            }
            ExprKind::Ident(name) => match env.get(name) {
                Some(v) => Ok(v),
                None => rt_err(format!("`{}` n'est pas défini", self.name(name)), span),
            },
            ExprKind::IVar(name) => rt_err(
                format!("`@{}` : les classes arrivent en v0.2", self.name(name)),
                span,
            ),
            ExprKind::SelfRef => rt_err("`self` : les classes arrivent en v0.2", span),
            ExprKind::Path(_) => rt_err("`A::B` : les enums et types arrivent en v0.2", span),

            ExprKind::List(items) => {
                let mut out = Vec::with_capacity(items.len as usize);
                for &it in ast.get(items) {
                    out.push(self.eval(sc, it, env)?);
                }
                Ok(Value::list(out))
            }
            ExprKind::Tuple(items) => {
                let mut out = Vec::with_capacity(items.len as usize);
                for &it in ast.get(items) {
                    out.push(self.eval(sc, it, env)?);
                }
                Ok(Value::Tuple(out.into()))
            }
            ExprKind::Hash(entries) => {
                let mut h = HashMap::new();
                for &(k, v) in ast.get(entries) {
                    let key = self.eval(sc, k, env)?;
                    let key = self.hash_key(&key, ast.span(k))?;
                    let value = self.eval(sc, v, env)?;
                    h.insert(key, value);
                }
                Ok(Value::Hash(Rc::new(RefCell::new(h))))
            }
            ExprKind::Record(fields) => {
                let mut r = Record::default();
                for &(name, v) in ast.get(fields) {
                    let value = self.eval(sc, v, env)?;
                    r.fields.push((name, value));
                }
                Ok(Value::Record(Rc::new(RefCell::new(r))))
            }

            ExprKind::Unary(op, x) => {
                let v = self.eval(sc, x, env)?;
                match (op, v) {
                    (UnaryOp::Not, v) => Ok(Value::Bool(!v.truthy())),
                    (UnaryOp::Neg, Value::Int(n)) => match n.checked_neg() {
                        Some(n) => Ok(Value::Int(n)),
                        None => rt_err("dépassement de capacité d'un entier", span),
                    },
                    (UnaryOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
                    (UnaryOp::Neg, v) => type_err(
                        format!("`-` attend un nombre, reçu {}", v.type_name()),
                        span,
                    ),
                }
            }
            ExprKind::Binary(BinOp::And, a, b) => {
                let l = self.eval(sc, a, env)?;
                if !l.truthy() {
                    Ok(l)
                } else {
                    self.eval(sc, b, env)
                }
            }
            ExprKind::Binary(BinOp::Or, a, b) => {
                let l = self.eval(sc, a, env)?;
                if l.truthy() {
                    Ok(l)
                } else {
                    self.eval(sc, b, env)
                }
            }
            ExprKind::Binary(op, a, b) => {
                let l = self.eval(sc, a, env)?;
                let r = self.eval(sc, b, env)?;
                self.binary(op, l, r, span)
            }
            ExprKind::Range { lo, hi, inclusive } => {
                let l = self.eval(sc, lo, env)?;
                let h = self.eval(sc, hi, env)?;
                match (l, h) {
                    (Value::Int(lo), Value::Int(hi)) => {
                        Ok(Value::Range(Rc::new(Range { lo, hi, inclusive })))
                    }
                    (l, h) => type_err(
                        format!(
                            "un intervalle attend deux entiers, reçu {} et {}",
                            l.type_name(),
                            h.type_name()
                        ),
                        span,
                    ),
                }
            }
            ExprKind::Assign { target, op, value } => self.assign(sc, target, op, value, env),

            ExprKind::Call {
                callee,
                args,
                block,
            } => {
                let f = self.eval(sc, callee, env)?;
                let (pos, named) = self.eval_args(sc, args, block, env)?;
                self.call_value(&f, pos, named, span)
            }
            ExprKind::MethodCall {
                recv,
                name,
                args,
                block,
            } => {
                let r = self.eval(sc, recv, env)?;
                // Un champ de record qui contient une fonction s'appelle comme une méthode.
                if let Value::Record(rec) = &r {
                    let field = rec.borrow().get(name).cloned();
                    if let Some(f) = field {
                        let (pos, named) = self.eval_args(sc, args, block, env)?;
                        return self.call_value(&f, pos, named, span);
                    }
                }
                let (pos, named) = self.eval_args(sc, args, block, env)?;
                if let Some(&(n, _)) = named.first() {
                    return rt_err(
                        format!(
                            "les méthodes intégrées n'acceptent pas d'argument nommé (`{}`)",
                            self.name(n)
                        ),
                        span,
                    );
                }
                self.method(r, name, pos, span)
            }
            ExprKind::Field(recv, name) => {
                let r = self.eval(sc, recv, env)?;
                self.field(&r, name, span)
            }
            ExprKind::TupleIndex(recv, idx) => {
                let r = self.eval(sc, recv, env)?;
                match &r {
                    Value::Tuple(t) => match t.get(idx as usize) {
                        Some(v) => Ok(v.clone()),
                        None => rt_err(
                            format!("index {idx} hors du tuple (taille {})", t.len()),
                            span,
                        ),
                    },
                    _ => type_err(
                        format!("`.{idx}` s'applique à un tuple, reçu {}", r.type_name()),
                        span,
                    ),
                }
            }
            ExprKind::Index(recv, idx) => {
                let r = self.eval(sc, recv, env)?;
                let i = self.eval(sc, idx, env)?;
                self.index(&r, &i, span)
            }
            ExprKind::Lambda(id) => Ok(self.make_lambda(sc, id, env)),

            ExprKind::If(c) => {
                let x = ast.if_(c);
                if self.eval(sc, x.cond, env)?.truthy() {
                    self.exec_block(sc, &x.then, env, true)
                } else if let Some(b) = &x.else_ {
                    self.exec_block(sc, b, env, true)
                } else {
                    Ok(Value::Nil)
                }
            }
            ExprKind::Case(c) => {
                let x = ast.case(c);
                let subject = self.eval(sc, x.subject, env)?;
                for arm in &x.arms {
                    for &p in ast.get(arm.patterns) {
                        let pv = self.eval(sc, p, env)?;
                        let hit = match (&pv, &subject) {
                            (Value::Range(r), Value::Int(n)) => r.contains(*n),
                            _ => pv.equals(&subject),
                        };
                        if hit {
                            return self.exec_block(sc, &arm.body, env, true);
                        }
                    }
                }
                match &x.else_ {
                    Some(b) => self.exec_block(sc, b, env, true),
                    None => Ok(Value::Nil),
                }
            }
            ExprKind::While(c) => {
                let x = ast.while_(c);
                while self.eval(sc, x.cond, env)?.truthy() {
                    match self.exec_block(sc, &x.body, env, true) {
                        Ok(_) | Err(Flow::Next(_)) => {}
                        Err(Flow::Break(_)) => break,
                        Err(f) => return Err(f),
                    }
                }
                Ok(Value::Nil)
            }
            ExprKind::For(c) => self.exec_for(sc, ast.for_(c), env),
            ExprKind::Return(v) => {
                let v = match v {
                    Some(v) => self.eval(sc, v, env)?,
                    None => Value::Nil,
                };
                Err(Flow::Return(v))
            }
            ExprKind::Break => Err(Flow::Break(span)),
            ExprKind::Next => Err(Flow::Next(span)),
        }
    }

    fn exec_for(&mut self, sc: &Rc<Script>, x: &For, env: &Env) -> R {
        let ast = &sc.program.ast;
        let iter_span = ast.span(x.iter);
        let iterable = self.eval(sc, x.iter, env)?;
        // Les collections sont copiées (superficiellement) avant la boucle :
        // modifier la liste pendant le parcours n'affecte pas l'itération.
        let items: Box<dyn Iterator<Item = Value>> = match &iterable {
            Value::List(l) => Box::new(l.borrow().clone().into_iter()),
            Value::Tuple(t) => {
                let t = t.clone();
                Box::new((0..t.len()).map(move |i| t[i].clone()))
            }
            Value::Range(r) => Box::new((r.lo..r.end()).map(Value::Int)),
            Value::Str(s) => Box::new(
                s.chars()
                    .map(|c| Value::Str(c.to_string().into()))
                    .collect::<Vec<_>>()
                    .into_iter(),
            ),
            Value::Hash(h) => Box::new(
                h.borrow()
                    .iter()
                    .map(|(k, v)| Value::Tuple(vec![k.to_value(), v.clone()].into()))
                    .collect::<Vec<_>>()
                    .into_iter(),
            ),
            v => {
                return type_err(
                    format!(
                        "`for` ne peut pas parcourir une valeur de type {}",
                        v.type_name()
                    ),
                    iter_span,
                );
            }
        };
        let is_hash = matches!(iterable, Value::Hash(_));
        for item in items {
            let scope = env.child();
            match x.vars.as_slice() {
                // `for k in hash` : la clé seule.
                [(name, _)] if is_hash => {
                    let Value::Tuple(t) = &item else {
                        unreachable!()
                    };
                    scope.declare(*name, t[0].clone(), true);
                }
                [(name, _)] => scope.declare(*name, item, true),
                vars => {
                    let values: Vec<Value> = match &item {
                        Value::Tuple(t) => t.to_vec(),
                        Value::List(l) => l.borrow().clone(),
                        v => {
                            return type_err(
                                format!(
                                    "{} variables attendent un tuple à chaque tour, reçu {}",
                                    vars.len(),
                                    v.type_name()
                                ),
                                iter_span,
                            );
                        }
                    };
                    if values.len() != vars.len() {
                        return rt_err(
                            format!("{} variables pour {} valeurs", vars.len(), values.len()),
                            iter_span,
                        );
                    }
                    for ((name, _), v) in vars.iter().zip(values) {
                        scope.declare(*name, v, true);
                    }
                }
            }
            match self.exec_block(sc, &x.body, &scope, false) {
                Ok(_) | Err(Flow::Next(_)) => {}
                Err(Flow::Break(_)) => break,
                Err(f) => return Err(f),
            }
        }
        Ok(Value::Nil)
    }

    fn assign(
        &mut self,
        sc: &Rc<Script>,
        target: ExprId,
        op: Option<BinOp>,
        value: ExprId,
        env: &Env,
    ) -> R {
        let ast = &sc.program.ast;
        let tspan = ast.span(target);
        match ast.kind(target) {
            ExprKind::Ident(name) => {
                let mut v = self.eval(sc, value, env)?;
                if let Some(op) = op {
                    let Some(old) = env.get(name) else {
                        return rt_err(format!("`{}` n'est pas défini", self.name(name)), tspan);
                    };
                    v = self.binary(op, old, v, ast.span(value).to(tspan))?;
                }
                match env.assign(name, v.clone()) {
                    Ok(()) => Ok(v),
                    Err(AssignError::NotFound) => {
                        let n = self.name(name);
                        // `NOM = valeur` déclare une constante.
                        if op.is_none() && n.starts_with(|c: char| c.is_ascii_uppercase()) {
                            env.declare(name, v.clone(), false);
                            Ok(v)
                        } else {
                            rt_err(
                                format!("`{n}` n'est pas déclaré : utiliser `let {n} = …`"),
                                tspan,
                            )
                        }
                    }
                    Err(AssignError::Immutable) => rt_err(
                        format!(
                            "`{}` ne peut pas être modifié (constante ou fonction)",
                            self.name(name)
                        ),
                        tspan,
                    ),
                }
            }
            ExprKind::Field(recv, name) => {
                let r = self.eval(sc, recv, env)?;
                let Value::Record(rec) = &r else {
                    return type_err(
                        format!("impossible d'assigner un champ sur {}", r.type_name()),
                        tspan,
                    );
                };
                let mut v = self.eval(sc, value, env)?;
                if let Some(op) = op {
                    let old = self.field(&r, name, tspan)?;
                    v = self.binary(op, old, v, tspan)?;
                }
                match rec.borrow_mut().get_mut(name) {
                    Some(slot) => *slot = v.clone(),
                    None => {
                        return rt_err(
                            format!("le record n'a pas de champ `{}`", self.name(name)),
                            tspan,
                        );
                    }
                }
                Ok(v)
            }
            ExprKind::Index(recv, idx) => {
                let r = self.eval(sc, recv, env)?;
                let i = self.eval(sc, idx, env)?;
                let mut v = self.eval(sc, value, env)?;
                if let Some(op) = op {
                    let old = self.index(&r, &i, tspan)?;
                    v = self.binary(op, old, v, tspan)?;
                }
                match &r {
                    Value::List(l) => {
                        let mut l = l.borrow_mut();
                        let pos = list_index(&i, l.len(), tspan)?;
                        l[pos] = v.clone();
                    }
                    Value::Hash(h) => {
                        let key = self.hash_key(&i, ast.span(idx))?;
                        h.borrow_mut().insert(key, v.clone());
                    }
                    _ => {
                        return type_err(
                            format!("impossible d'assigner par index sur {}", r.type_name()),
                            tspan,
                        );
                    }
                }
                Ok(v)
            }
            ExprKind::IVar(name) => rt_err(
                format!("`@{}` : les classes arrivent en v0.2", self.name(name)),
                tspan,
            ),
            _ => rt_err("on ne peut pas assigner à cette expression", tspan),
        }
    }

    fn hash_key(&self, v: &Value, span: Span) -> R<HashKey> {
        match HashKey::from_value(v) {
            Some(k) => Ok(k),
            None => type_err(
                format!(
                    "une clé de hash doit être nil, bool, int ou str, reçu {}",
                    v.type_name()
                ),
                span,
            ),
        }
    }

    fn field(&self, r: &Value, name: Symbol, span: Span) -> R {
        match r {
            Value::Record(rec) => match rec.borrow().get(name) {
                Some(v) => Ok(v.clone()),
                None => rt_err(
                    format!("le record n'a pas de champ `{}`", self.name(name)),
                    span,
                ),
            },
            _ if self.methods.contains_key(&name) => rt_err(
                format!(
                    "`{n}` est une méthode : écrire `.{n}()`",
                    n = self.name(name)
                ),
                span,
            ),
            _ => type_err(
                format!("{} n'a pas de champ `{}`", r.type_name(), self.name(name)),
                span,
            ),
        }
    }

    fn index(&self, r: &Value, i: &Value, span: Span) -> R {
        match r {
            Value::List(l) => {
                let l = l.borrow();
                let pos = list_index(i, l.len(), span)?;
                Ok(l[pos].clone())
            }
            Value::Hash(h) => {
                let key = self.hash_key(i, span)?;
                Ok(h.borrow().get(&key).cloned().unwrap_or(Value::Nil))
            }
            Value::Str(s) => {
                let chars: Vec<char> = s.chars().collect();
                let pos = list_index(i, chars.len(), span)?;
                Ok(Value::Str(chars[pos].to_string().into()))
            }
            Value::Tuple(t) => {
                let pos = list_index(i, t.len(), span)?;
                Ok(t[pos].clone())
            }
            _ => type_err(
                format!("{} ne s'indexe pas avec `[…]`", r.type_name()),
                span,
            ),
        }
    }

    // ─── Opérateurs ──────────────────────────────────────────────────────────

    fn binary(&self, op: BinOp, l: Value, r: Value, span: Span) -> R {
        use BinOp::*;
        use Value::{Float, Int};
        let overflow = || Flow::Error(Error::runtime("dépassement de capacité d'un entier", span));
        Ok(match (op, &l, &r) {
            (Eq, _, _) => Value::Bool(l.equals(&r)),
            (Ne, _, _) => Value::Bool(!l.equals(&r)),

            (Add, Int(a), Int(b)) => Int(a.checked_add(*b).ok_or_else(overflow)?),
            (Sub, Int(a), Int(b)) => Int(a.checked_sub(*b).ok_or_else(overflow)?),
            (Mul, Int(a), Int(b)) => Int(a.checked_mul(*b).ok_or_else(overflow)?),
            (Div | Rem, Int(_), Int(0)) => return rt_err("division par zéro", span),
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
                        "`+` entre {} et {} : utiliser l'interpolation \"#{{x}}\" ou `.to_s()`",
                        l.type_name(),
                        r.type_name()
                    ),
                    span,
                );
            }
            (Mul, Value::Str(s), Int(n)) => {
                if *n < 0 {
                    return rt_err("répétition d'une chaîne par un nombre négatif", span);
                }
                Value::Str(s.repeat(*n as usize).into())
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
                                "`{}` ne compare pas {} et {}",
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
                list.borrow_mut().push(r.clone());
                l.clone()
            }

            _ => {
                return type_err(
                    format!(
                        "`{}` ne s'applique pas à {} et {}",
                        op.as_str(),
                        l.type_name(),
                        r.type_name()
                    ),
                    span,
                );
            }
        })
    }

    // ─── Appels ──────────────────────────────────────────────────────────────

    fn make_lambda(&self, sc: &Rc<Script>, id: LambdaId, env: &Env) -> Value {
        Value::Function(Rc::new(Function::Script(Closure {
            name: "lambda".into(),
            script: sc.clone(),
            body: Body::Lambda(id),
            env: env.clone(),
        })))
    }

    /// Arguments évalués : positionnels (le bloc `do` en dernier) et nommés.
    fn eval_args(
        &mut self,
        sc: &Rc<Script>,
        args: Slice<Arg>,
        block: Option<LambdaId>,
        env: &Env,
    ) -> R<Args> {
        let mut pos = Vec::with_capacity(args.len as usize + block.is_some() as usize);
        let mut named = Vec::new();
        for a in sc.program.ast.get(args) {
            let v = self.eval(sc, a.value, env)?;
            match a.name {
                Some(n) => named.push((n, v)),
                None => pos.push(v),
            }
        }
        if let Some(b) = block {
            pos.push(self.make_lambda(sc, b, env));
        }
        Ok((pos, named))
    }

    /// Appelle une valeur avec des arguments.
    pub fn call_value(
        &mut self,
        f: &Value,
        args: Vec<Value>,
        named: Vec<(Symbol, Value)>,
        span: Span,
    ) -> R {
        let Value::Function(func) = f else {
            return type_err(format!("{} n'est pas une fonction", f.type_name()), span);
        };
        match &**func {
            Function::Native(n) => {
                if let Some(&(name, _)) = named.first() {
                    return rt_err(
                        format!(
                            "`{}` (fonction Rust) n'accepte pas d'argument nommé (`{}`)",
                            n.name,
                            self.name(name)
                        ),
                        span,
                    );
                }
                (n.f)(&args).map_err(|msg| Flow::Error(Error::runtime(msg, span)))
            }
            Function::Builtin(b) => self.builtin(*b, args, span),
            Function::Script(c) => {
                if self.depth >= self.max_depth {
                    return rt_err(
                        format!(
                            "trop d'appels imbriqués ({}) : récursion infinie ?",
                            self.max_depth
                        ),
                        span,
                    );
                }
                self.depth += 1;
                let r = self.call_closure(c, args, named, span);
                self.depth -= 1;
                r.map_err(|f| match f {
                    Flow::Error(mut e) => {
                        e = attach_src(e, &c.script);
                        if e.trace.len() < 16 {
                            e.trace.push((c.name.to_string(), span));
                        }
                        Flow::Error(e)
                    }
                    f => f,
                })
            }
        }
    }

    fn call_closure(
        &mut self,
        c: &Closure,
        args: Vec<Value>,
        named: Vec<(Symbol, Value)>,
        call_span: Span,
    ) -> R {
        let sc = &c.script;
        let (params, body, ret, lenient): (&[Param], &Block, Option<&TypeExpr>, bool) =
            match &c.body {
                Body::Def(f) => (
                    f.params.as_deref().unwrap_or(&[]),
                    &f.body,
                    f.ret.as_ref(),
                    false,
                ),
                // Les lambdas sont tolérants (comme les blocs Ruby) : argument manquant = nil,
                // arguments en trop ignorés. Pratique pour `each do |k, v|`.
                Body::Lambda(id) => {
                    let l = sc.program.ast.lambda(*id);
                    (&l.params, &l.body, None, true)
                }
            };
        if !lenient && args.len() > params.len() {
            return rt_err(
                format!(
                    "`{}` attend {} argument{}, reçu {}",
                    c.name,
                    params.len(),
                    if params.len() > 1 { "s" } else { "" },
                    args.len()
                ),
                call_span,
            );
        }
        for (n, _) in &named {
            if !params.iter().any(|p| p.name == *n) {
                return rt_err(
                    format!("`{}` n'a pas de paramètre `{}`", c.name, self.name(*n)),
                    call_span,
                );
            }
        }
        let env = c.env.child();
        let mut args = args.into_iter();
        for p in params {
            if p.is_field {
                return rt_err("les paramètres `@champ` (classes) arrivent en v0.2", p.span);
            }
            let given = args.next();
            let named_v = named
                .iter()
                .find(|(n, _)| *n == p.name)
                .map(|(_, v)| v.clone());
            if given.is_some() && named_v.is_some() {
                return rt_err(
                    format!("argument `{}` donné deux fois", self.name(p.name)),
                    call_span,
                );
            }
            let v = match given.or(named_v) {
                Some(v) => v,
                None => match p.default {
                    Some(d) => self.eval(sc, d, &env)?,
                    None if lenient => Value::Nil,
                    None => {
                        return rt_err(
                            format!(
                                "argument `{}` manquant pour `{}`",
                                self.name(p.name),
                                c.name
                            ),
                            call_span,
                        );
                    }
                },
            };
            let v = match &p.ty {
                Some(ty) => self.check_type(v, ty, p.span)?,
                None => v,
            };
            env.declare(p.name, v, true);
        }
        let result = match self.exec_block(sc, body, &env, false) {
            Ok(v) | Err(Flow::Return(v)) => v,
            Err(Flow::Break(s)) => return rt_err("`break` en dehors d'une boucle", s),
            Err(Flow::Next(s)) => return rt_err("`next` en dehors d'une boucle", s),
            Err(e) => return Err(e),
        };
        match ret {
            Some(ty) => self.check_type(result, ty, ty.span),
            None => Ok(result),
        }
    }

    fn builtin(&mut self, b: Builtin, args: Vec<Value>, span: Span) -> R {
        match b {
            Builtin::Print => {
                let mut line = String::new();
                for (n, a) in args.iter().enumerate() {
                    if n > 0 {
                        line.push(' ');
                    }
                    line.push_str(&a.to_display(&self.interner));
                }
                line.push('\n');
                (self.output)(&line);
                Ok(Value::Nil)
            }
            Builtin::TypeOf => {
                arity("typeof", &args, 1, span)?;
                Ok(Value::str(args[0].type_name()))
            }
            Builtin::Assert => {
                if args.is_empty() || args.len() > 2 {
                    return rt_err(
                        "`assert(condition, message?)` attend 1 ou 2 arguments",
                        span,
                    );
                }
                if args[0].truthy() {
                    Ok(Value::Nil)
                } else {
                    let msg = match args.get(1) {
                        Some(m) => format!("assertion échouée : {}", m.to_display(&self.interner)),
                        None => "assertion échouée".to_string(),
                    };
                    rt_err(msg, span)
                }
            }
        }
    }

    // ─── Méthodes intégrées ──────────────────────────────────────────────────

    fn call1(&mut self, f: &Value, a: Value, span: Span) -> R {
        self.call_value(f, vec![a], Vec::new(), span)
    }

    fn method(&mut self, recv: Value, name: Symbol, args: Vec<Value>, span: Span) -> R {
        use Method as M;
        let Some(&m) = self.methods.get(&name) else {
            return type_err(
                format!(
                    "{} n'a pas de méthode `{}`",
                    recv.type_name(),
                    self.name(name)
                ),
                span,
            );
        };
        let mname = self.name(name).to_string();
        let n = |k: usize| arity(&mname, &args, k, span);
        match (m, &recv) {
            (M::ToS, _) => {
                n(0)?;
                Ok(Value::Str(recv.to_display(&self.interner).into()))
            }
            (M::Inspect, _) => {
                n(0)?;
                Ok(Value::Str(recv.inspect(&self.interner).into()))
            }
            (M::Call, Value::Function(_)) => self.call_value(&recv, args, Vec::new(), span),

            // ── Nombres
            (M::Abs, Value::Int(i)) => {
                n(0)?;
                match i.checked_abs() {
                    Some(v) => Ok(Value::Int(v)),
                    None => rt_err("dépassement de capacité d'un entier", span),
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
                            return rt_err("nombre trop grand pour un entier", span);
                        }
                        Value::Int(r as i64)
                    }
                })
            }
            (M::ToI | M::Floor | M::Ceil | M::Round, Value::Int(_)) => {
                n(0)?;
                Ok(recv.clone())
            }
            (M::Sqrt, Value::Int(_) | Value::Float(_)) => {
                n(0)?;
                Ok(Value::Float(as_f64(&recv).sqrt()))
            }

            // ── Chaînes
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
                let pat = expect_str(&args[0], &mname, span)?;
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
                        let sep = expect_str(sep, &mname, span)?;
                        s.split(&*sep).map(Value::str).collect()
                    }
                    _ => return rt_err("`split(séparateur?)` attend 0 ou 1 argument", span),
                };
                Ok(Value::list(parts))
            }
            (M::Replace, Value::Str(s)) => {
                n(2)?;
                let from = expect_str(&args[0], &mname, span)?;
                let to = expect_str(&args[1], &mname, span)?;
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
                    Err(_) => rt_err(format!("{s:?} n'est pas un entier"), span),
                }
            }
            (M::ToF, Value::Str(s)) => {
                n(0)?;
                match s.trim().parse::<f64>() {
                    Ok(v) => Ok(Value::Float(v)),
                    Err(_) => rt_err(format!("{s:?} n'est pas un nombre"), span),
                }
            }

            // ── Listes
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
                l.borrow_mut().push(args[0].clone());
                Ok(recv.clone())
            }
            (M::Pop, Value::List(l)) => {
                n(0)?;
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
                    [s] => expect_str(s, &mname, span)?,
                    _ => return rt_err("`join(séparateur?)` attend 0 ou 1 argument", span),
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
                    return type_err(format!("`sort` ne compare pas {a} et {b}"), span);
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
                                        "`{mname}` ne compare pas {} et {}",
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
                let items = iter_values(&recv);
                let mut out = Vec::new();
                for v in items {
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
                for v in iter_values(&recv) {
                    acc = self.call_value(&f, vec![acc, v], Vec::new(), span)?;
                }
                Ok(acc)
            }

            // ── Intervalles
            (M::Len, Value::Range(r)) => {
                n(0)?;
                Ok(Value::Int(r.len() as i64))
            }
            (M::Contains, Value::Range(r)) => {
                n(1)?;
                Ok(Value::Bool(
                    matches!(args[0], Value::Int(i) if r.contains(i)),
                ))
            }
            (M::ToList, Value::Range(_)) => {
                n(0)?;
                Ok(Value::list(iter_values(&recv)))
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
                    return rt_err("`get(clé, défaut?)` attend 1 ou 2 arguments", span);
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
                    self.call_value(&f, vec![k, v], Vec::new(), span)?;
                }
                Ok(recv.clone())
            }

            // ── Tuples
            (M::Len, Value::Tuple(t)) => {
                n(0)?;
                Ok(Value::Int(t.len() as i64))
            }

            _ => type_err(
                format!("{} n'a pas de méthode `{mname}`", recv.type_name()),
                span,
            ),
        }
    }

    // ─── Types ───────────────────────────────────────────────────────────────

    /// Vérifie `v` contre une annotation. Renvoie la valeur, éventuellement convertie
    /// (`int` → `float` quand un `float` est attendu).
    fn check_type(&self, v: Value, ty: &TypeExpr, span: Span) -> R {
        if let (TypeKind::Named(path), Value::Int(i)) = (&ty.kind, &v)
            && path.len() == 1
            && self.name(path[0]) == "float"
        {
            return Ok(Value::Float(*i as f64));
        }
        match self.matches_type(&v, ty) {
            Ok(true) => Ok(v),
            Ok(false) => type_err(
                format!(
                    "type attendu `{}`, reçu {} ({})",
                    printer::type_expr(ty, &self.interner),
                    v.type_name(),
                    short(&v.inspect(&self.interner))
                ),
                span,
            ),
            Err(msg) => type_err(msg, ty.span),
        }
    }

    fn matches_type(&self, v: &Value, ty: &TypeExpr) -> Result<bool, String> {
        Ok(match &ty.kind {
            TypeKind::Named(path) => {
                let name = path
                    .iter()
                    .map(|s| self.name(*s))
                    .collect::<Vec<_>>()
                    .join("::");
                match (name.as_str(), v) {
                    ("any", _) => true,
                    ("int", Value::Int(_))
                    | ("float", Value::Float(_))
                    | ("str", Value::Str(_))
                    | ("bool", Value::Bool(_))
                    | ("nil", Value::Nil) => true,
                    ("int" | "float" | "str" | "bool" | "nil", _) => false,
                    _ => {
                        return Err(format!(
                            "type `{name}` inconnu (les types utilisateur arrivent en v0.2)"
                        ));
                    }
                }
            }
            TypeKind::List(inner) => match v {
                Value::List(l) => {
                    for item in l.borrow().iter() {
                        if !self.matches_type(item, inner)? {
                            return Ok(false);
                        }
                    }
                    true
                }
                _ => false,
            },
            TypeKind::Hash(kt, vt) => match v {
                Value::Hash(h) => {
                    for (k, val) in h.borrow().iter() {
                        if !self.matches_type(&k.to_value(), kt)? || !self.matches_type(val, vt)? {
                            return Ok(false);
                        }
                    }
                    true
                }
                _ => false,
            },
            TypeKind::Record(fields) => match v {
                Value::Record(r) => {
                    let r = r.borrow();
                    if r.fields.len() != fields.len() {
                        return Ok(false);
                    }
                    for (name, fty) in fields {
                        match r.get(*name) {
                            Some(fv) if self.matches_type(fv, fty)? => {}
                            _ => return Ok(false),
                        }
                    }
                    true
                }
                _ => false,
            },
            TypeKind::Tuple(items) => match v {
                Value::Tuple(t) => {
                    if t.len() != items.len() {
                        return Ok(false);
                    }
                    for (val, ity) in t.iter().zip(items) {
                        if !self.matches_type(val, ity)? {
                            return Ok(false);
                        }
                    }
                    true
                }
                _ => false,
            },
            TypeKind::Optional(inner) => matches!(v, Value::Nil) || self.matches_type(v, inner)?,
            TypeKind::Union(items) => {
                for it in items {
                    if self.matches_type(v, it)? {
                        return Ok(true);
                    }
                }
                false
            }
            TypeKind::LitStr(s) => matches!(v, Value::Str(x) if **x == **s),
            TypeKind::LitInt(n) => matches!(v, Value::Int(x) if x == n),
            TypeKind::Fn(..) => matches!(v, Value::Function(_)),
        })
    }
}

// ─── Outils ─────────────────────────────────────────────────────────────────

/// Rattache le source du script à une erreur qui n'en a pas encore.
fn attach_src(mut e: Error, sc: &Rc<Script>) -> Error {
    if e.src.is_none() {
        e.src = Some((sc.name.clone(), sc.src.clone()));
    }
    e
}

fn as_f64(v: &Value) -> f64 {
    match v {
        Value::Int(i) => *i as f64,
        Value::Float(f) => *f,
        _ => f64::NAN,
    }
}

fn compare(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
        (Value::Int(_) | Value::Float(_), Value::Int(_) | Value::Float(_)) => {
            as_f64(a).partial_cmp(&as_f64(b))
        }
        (Value::Str(x), Value::Str(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// Index de liste, négatif depuis la fin.
fn list_index(i: &Value, len: usize, span: Span) -> R<usize> {
    let Value::Int(i) = i else {
        return type_err(
            format!("un index doit être un entier, reçu {}", i.type_name()),
            span,
        );
    };
    let pos = if *i < 0 { len as i64 + i } else { *i };
    if pos < 0 || pos >= len as i64 {
        return rt_err(format!("index {i} hors limites (taille {len})"), span);
    }
    Ok(pos as usize)
}

fn iter_values(v: &Value) -> Vec<Value> {
    match v {
        Value::List(l) => l.borrow().clone(),
        Value::Range(r) => (r.lo..r.end()).map(Value::Int).collect(),
        _ => Vec::new(),
    }
}

fn arity(name: &str, args: &[Value], n: usize, span: Span) -> R<()> {
    if args.len() == n {
        return Ok(());
    }
    rt_err(
        format!(
            "`{name}` attend {n} argument{}, reçu {}",
            if n > 1 { "s" } else { "" },
            args.len()
        ),
        span,
    )
}

fn expect_str(v: &Value, method: &str, span: Span) -> R<Rc<str>> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        _ => type_err(
            format!("`{method}` attend une chaîne, reçu {}", v.type_name()),
            span,
        ),
    }
}

fn short(s: &str) -> String {
    if s.chars().count() > 40 {
        let mut out: String = s.chars().take(37).collect();
        out.push_str("...");
        out
    } else {
        s.to_string()
    }
}
