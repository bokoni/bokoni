//! Interpréteur de l'IR.
//!
//! Chaque appel réserve une fenêtre de registres sur une pile de valeurs partagée
//! (`stack`), puis exécute les blocs de base de la fonction. Les registres typés
//! contiennent toujours une valeur du bon type (`Value::Int` dans un registre `int`) :
//! les instructions spécialisées (`IBin`) n'ont pas à vérifier le type.

use crate::ast::{BinOp, TypeDeclKind};
use crate::diagnostic::Error;
use crate::globals::Globals;
use crate::intern::{FxHashMap, Interner, Symbol};
use crate::ir::*;
use crate::objects::{ConstructCtx, EnumInfo, Syms, TypeInfo};
use crate::ops::{Builtin, METHODS, Method, R, list_index, rt_err, short, type_err};
use crate::span::Span;
use crate::value::{Function, HashMap, Range, Record, Value};
use std::cell::RefCell;
use std::rc::Rc;

/// Cellule d'une variable capturée par une closure.
pub type Cell = Rc<RefCell<Value>>;

/// Fonction ou lambda du script : son code et ses variables capturées.
pub struct Closure {
    pub proto: Rc<Proto>,
    pub upvals: Box<[Cell]>,
    /// Constructions en cours au moment de la création (blocs `do` imbriqués).
    pub ctx: Rc<[u32]>,
}

/// Arguments évalués d'un appel.
pub struct CallArgs {
    pub pos: Vec<Value>,
    pub named: Vec<(Symbol, Value)>,
    /// Forme littérale de chaque argument (positionnels puis nommés), pour convertir un
    /// record littéral vers le type attendu. Vide pour un appel depuis Rust ou une
    /// méthode intégrée.
    pub shapes: Vec<Option<Rc<Shape>>>,
}

impl CallArgs {
    pub fn plain(pos: Vec<Value>) -> Self {
        CallArgs {
            pos,
            named: Vec::new(),
            shapes: Vec::new(),
        }
    }
}

/// État d'un itérateur de boucle `for` (copie des éléments au début de la boucle).
pub enum IterState {
    Values(std::vec::IntoIter<Value>),
    Range(std::ops::RangeInclusive<i64>),
}

const RED_ZONE: usize = 256 * 1024;
const STACK_GROW: usize = 2 * 1024 * 1024;
/// La pile n'est examinée qu'une fois tous les `STACK_CHECK` niveaux d'appel : la zone
/// de réserve couvre largement ces appels (moins de 32 Kio chacun).
const STACK_CHECK: usize = 8;

/// État de l'interpréteur : noms, variables globales, sortie de `print`.
pub struct Runtime {
    pub interner: Interner,
    pub globals: Globals,
    pub output: Box<dyn FnMut(&str)>,
    /// Profondeur d'appels maximale (protège contre la récursion infinie).
    pub max_depth: usize,
    pub(crate) depth: usize,
    pub(crate) methods: FxHashMap<Symbol, (Method, &'static str)>,
    /// Constructions de `type` en cours (blocs `Server do … end` imbriqués).
    pub(crate) constructs: Vec<ConstructCtx>,
    pub(crate) next_ctx: u32,
    pub(crate) syms: Syms,
    /// Fenêtres de registres libres, réutilisées d'un appel à l'autre (pas d'allocation
    /// par appel en régime établi).
    frames: Vec<Vec<Value>>,
    no_ctx: Rc<[u32]>,
    /// Compilation native des fonctions typées (créée au premier besoin).
    pub jit_enabled: bool,
    /// Compilation de base dès le premier appel (tests).
    pub jit_eager: bool,
    /// Tampon de l'interpolation de chaînes.
    concat_buf: String,
    #[cfg(feature = "jit")]
    jit: Option<crate::jit::Jit>,
    /// Erreur levée pendant l'exécution de code natif de base, en attente de remontée.
    #[cfg(feature = "jit")]
    pending: Option<Error>,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

/// Rattache le source du script à une erreur qui n'en a pas encore.
fn attach_src(mut e: Error, sc: &Script) -> Error {
    if e.src.is_none() {
        e.src = Some((sc.name.clone(), sc.src.clone()));
    }
    e
}

impl Runtime {
    pub fn new() -> Self {
        let mut interner = Interner::new();
        let methods = METHODS
            .iter()
            .map(|&(name, m)| (interner.intern(name), (m, name)))
            .collect();
        let syms = Syms::new(&mut interner);
        let mut globals = Globals::default();
        for (name, b) in Builtin::ALL {
            let sym = interner.intern(name);
            globals.define(sym, Value::Function(Rc::new(Function::Builtin(b))), false);
        }
        Runtime {
            interner,
            globals,
            output: Box::new(|s| print!("{s}")),
            max_depth: 1000,
            depth: 0,
            methods,
            constructs: Vec::new(),
            next_ctx: 0,
            syms,
            frames: Vec::new(),
            no_ctx: Rc::from([]),
            jit_enabled: cfg!(feature = "jit"),
            jit_eager: false,
            concat_buf: String::new(),
            #[cfg(feature = "jit")]
            jit: None,
            #[cfg(feature = "jit")]
            pending: None,
        }
    }

    pub(crate) fn name(&self, s: Symbol) -> &str {
        self.interner.resolve(s)
    }

    /// Exécute le programme principal d'un script.
    pub fn run(&mut self, main: &Rc<Proto>) -> Result<Value, Error> {
        self.depth = 0;
        let ctx = self.no_ctx.clone();
        self.invoke(
            main,
            &[],
            &ctx,
            None,
            CallArgs::plain(Vec::new()),
            Span::default(),
        )
        .map_err(|e| attach_src(e, &main.script))
    }

    // ─── Appels ──────────────────────────────────────────────────────────────

    /// Appelle une valeur : fonction, lambda, fonction Rust, ou type (construction).
    pub fn call_value(&mut self, f: &Value, args: CallArgs, span: Span) -> R {
        let func = match f {
            Value::Function(func) => func,
            Value::Type(info) => return self.construct(info, args, None, span),
            _ => return type_err(format!("{} n'est pas une fonction", f.type_label()), span),
        };
        match &**func {
            Function::Native(n) => {
                if let Some(&(name, _)) = args.named.first() {
                    return rt_err(
                        format!(
                            "`{}` (fonction Rust) n'accepte pas d'argument nommé (`{}`)",
                            n.name,
                            self.name(name)
                        ),
                        span,
                    );
                }
                (n.f)(&args.pos).map_err(|msg| Error::runtime(msg, span))
            }
            Function::Builtin(b) => self.builtin(*b, args.pos, span),
            Function::Script(c) => self.invoke(&c.proto, &c.upvals, &c.ctx, None, args, span),
        }
    }

    /// Appelle une méthode déclarée (`type`, `class`, `enum`) avec `self` = `this`.
    pub(crate) fn call_method(
        &mut self,
        this: Value,
        m: &Rc<Proto>,
        args: CallArgs,
        span: Span,
    ) -> R {
        let ctx = self.no_ctx.clone();
        self.invoke(m, &[], &ctx, Some(this), args, span)
    }

    /// Exécute une fonction : liaison des arguments, puis ses blocs.
    pub(crate) fn invoke(
        &mut self,
        proto: &Rc<Proto>,
        upvals: &[Cell],
        ctx: &Rc<[u32]>,
        this: Option<Value>,
        args: CallArgs,
        span: Span,
    ) -> R {
        self.invoke_with(proto, upvals, ctx, span, |rt, regs| {
            rt.bind(proto, regs, this, args, span)
        })
    }

    /// Appel rapide : arguments positionnels, autant que de paramètres, lus directement
    /// dans les registres de l'appelant.
    pub(crate) fn invoke_regs(&mut self, c: &Closure, args: &[Value], span: Span) -> R {
        #[cfg(feature = "jit")]
        if let Some(r) = self.invoke_native(c, args, span) {
            return r;
        }
        let proto = &c.proto;
        self.invoke_with(proto, &c.upvals, &c.ctx, span, |rt, regs| {
            for (p, v) in proto.params.iter().zip(args) {
                regs[p.reg as usize] = match &p.ty {
                    Some(ty) => rt.coerce(v.clone(), ty, None, p.span)?,
                    None => v.clone(),
                };
            }
            Ok(())
        })
    }

    fn invoke_with(
        &mut self,
        proto: &Rc<Proto>,
        upvals: &[Cell],
        ctx: &Rc<[u32]>,
        span: Span,
        bind: impl FnOnce(&mut Self, &mut [Value]) -> R<()>,
    ) -> R {
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
        let mut regs = self.frames.pop().unwrap_or_default();
        fill_nil(&mut regs, proto.regs.len());
        let check_stack = self.depth.is_multiple_of(STACK_CHECK);
        let body = || {
            bind(self, &mut regs)?;
            #[cfg(feature = "jit")]
            if let Some(r) = self.try_jit(proto, &regs) {
                return r;
            }
            self.exec(proto, upvals, ctx, &mut regs)
        };
        let r = if check_stack {
            stacker::maybe_grow(RED_ZONE, STACK_GROW, body)
        } else {
            body()
        };
        regs.clear();
        self.frames.push(regs);
        self.depth -= 1;
        r.map_err(|e| self.traced(e, proto, span))
    }

    /// Erreur sortant de `proto` : source et appel ajoutés à la pile.
    #[cold]
    fn traced(&self, mut e: Error, proto: &Proto, span: Span) -> Error {
        if proto.kind != ProtoKind::Main {
            e = attach_src(e, &proto.script);
            if e.trace.len() < 16 {
                e.trace.push((proto.name.to_string(), span));
            }
        }
        e
    }

    /// Range les arguments dans les registres des paramètres (avec vérification des
    /// annotations). Un paramètre absent qui a une valeur par défaut reste « non
    /// fourni » : le début de la fonction la calcule.
    fn bind(
        &mut self,
        proto: &Proto,
        regs: &mut [Value],
        this: Option<Value>,
        args: CallArgs,
        span: Span,
    ) -> R<()> {
        let CallArgs { pos, named, shapes } = args;
        let params = &proto.params;
        let lenient = matches!(proto.kind, ProtoKind::Lambda | ProtoKind::DoBlock);
        if let Some(t) = this {
            regs[0] = t;
        }
        // Cas courant : uniquement des arguments positionnels, autant que de paramètres.
        if named.is_empty() && pos.len() == params.len() && shapes.iter().all(Option::is_none) {
            for (p, v) in params.iter().zip(pos) {
                let v = match &p.ty {
                    Some(ty) => self.coerce(v, ty, None, p.span)?,
                    None => v,
                };
                regs[p.reg as usize] = v;
            }
            return Ok(());
        }
        if !lenient && pos.len() > params.len() {
            return rt_err(
                format!(
                    "`{}` attend {} argument{}, reçu {}",
                    proto.name,
                    params.len(),
                    if params.len() > 1 { "s" } else { "" },
                    pos.len()
                ),
                span,
            );
        }
        for (n, _) in &named {
            if !params.iter().any(|p| p.name == *n) {
                return rt_err(
                    format!("`{}` n'a pas de paramètre `{}`", proto.name, self.name(*n)),
                    span,
                );
            }
        }
        let npos = pos.len();
        let mut pos = pos.into_iter().map(Some).collect::<Vec<_>>();
        let mut named: Vec<Option<(Symbol, Value)>> = named.into_iter().map(Some).collect();
        for (k, p) in params.iter().enumerate() {
            let given = pos.get_mut(k).and_then(Option::take);
            let named_i = named
                .iter()
                .position(|x| x.as_ref().is_some_and(|(n, _)| *n == p.name));
            if given.is_some() && named_i.is_some() {
                return rt_err(
                    format!("argument `{}` donné deux fois", self.name(p.name)),
                    span,
                );
            }
            let (v, shape) = match (given, named_i) {
                (Some(v), _) => (v, shapes.get(k).cloned().flatten()),
                (None, Some(j)) => (
                    named[j].take().unwrap().1,
                    shapes.get(npos + j).cloned().flatten(),
                ),
                (None, None) if p.has_default => (Value::Uninit, None),
                (None, None) if lenient => (Value::Nil, None),
                (None, None) => {
                    return rt_err(
                        format!(
                            "argument `{}` manquant pour `{}`",
                            self.name(p.name),
                            proto.name
                        ),
                        span,
                    );
                }
            };
            let v = match (&p.ty, &v) {
                (_, Value::Uninit) => v,
                (Some(ty), _) => self.coerce(v, ty, shape.as_deref(), p.span)?,
                (None, _) => v,
            };
            regs[p.reg as usize] = v;
        }
        Ok(())
    }

    fn call_args(proto: &Proto, regs: &[Value], site: &CallSite) -> CallArgs {
        let start = site.start as usize;
        let npos = site.npos as usize;
        let pos = regs[start..start + npos].to_vec();
        let named = site
            .named
            .iter()
            .enumerate()
            .map(|(k, n)| (*n, regs[start + npos + k].clone()))
            .collect();
        let shapes = if site.shapes.iter().all(Option::is_none) {
            Vec::new()
        } else {
            site.shapes
                .iter()
                .map(|s| s.map(|i| proto.shapes[i as usize].clone()))
                .collect()
        };
        CallArgs { pos, named, shapes }
    }

    // ─── Exécution ───────────────────────────────────────────────────────────

    fn exec(&mut self, proto: &Proto, upvals: &[Cell], ctx: &Rc<[u32]>, regs: &mut [Value]) -> R {
        let mut cells: Vec<Cell> = (0..proto.ncells)
            .map(|_| Rc::new(RefCell::new(Value::Uninit)))
            .collect();
        #[cfg(feature = "jit")]
        if let Some(entry) = self.dyn_entry(proto) {
            return self.run_native(entry, proto, upvals, ctx, regs, &mut cells);
        }
        let mut b = 0usize;
        macro_rules! reg {
            ($r:expr) => {
                regs[$r as usize]
            };
        }
        loop {
            let block = &proto.blocks[b];
            for (ins, &span) in block.instrs.iter().zip(&block.spans) {
                self.step(proto, upvals, ctx, regs, &mut cells, ins, span)?;
            }
            match &block.term {
                Term::Jump(t) => b = *t as usize,
                Term::Branch { cond, then, else_ } => {
                    b = if reg!(*cond).truthy() { *then } else { *else_ } as usize;
                }
                Term::IfInt {
                    op,
                    a,
                    b: y,
                    then,
                    else_,
                } => {
                    let (Value::Int(x), Value::Int(y)) = (&reg!(*a), &reg!(*y)) else {
                        unreachable!("registre int qui ne contient pas un int")
                    };
                    b = if int_cmp(*op, *x, *y) { *then } else { *else_ } as usize;
                }
                Term::IfIntK {
                    op,
                    a,
                    k,
                    then,
                    else_,
                } => {
                    let Value::Int(x) = reg!(*a) else {
                        unreachable!("registre int qui ne contient pas un int")
                    };
                    b = if int_cmp(*op, x, *k) { *then } else { *else_ } as usize;
                }
                Term::Return { reg: r, shape } => {
                    let v = std::mem::replace(&mut reg!(*r), Value::Nil);
                    return match &proto.ret {
                        Some(ty) => {
                            let shape = shape.map(|s| proto.shapes[s as usize].clone());
                            self.coerce(v, ty, shape.as_deref(), ty.span)
                        }
                        None => Ok(v),
                    };
                }
                Term::IterNext {
                    iter,
                    dst,
                    body,
                    done,
                } => {
                    b = if iter_next(regs, *iter, *dst) {
                        *body
                    } else {
                        *done
                    } as usize;
                }
            }
        }
    }

    /// Exécute une instruction (hors terminateur). Utilisée par la boucle de
    /// l'interpréteur (intégrée en ligne) et par le code natif du JIT pour les
    /// instructions qu'il ne traduit pas lui-même : la sémantique est la même par
    /// construction.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step(
        &mut self,
        proto: &Proto,
        upvals: &[Cell],
        ctx: &Rc<[u32]>,
        regs: &mut [Value],
        cells: &mut [Cell],
        ins: &Instr,
        span: Span,
    ) -> R<()> {
        macro_rules! reg {
            ($r:expr) => {
                regs[$r as usize]
            };
        }
        match ins {
            Instr::Int(d, n) => reg!(*d) = Value::Int(*n),
            Instr::Float(d, f) => reg!(*d) = Value::Float(*f),
            Instr::Bool(d, v) => reg!(*d) = Value::Bool(*v),
            Instr::Nil(d) => reg!(*d) = Value::Nil,
            Instr::Const(d, c) => reg!(*d) = proto.consts[*c as usize].clone(),
            Instr::Move(d, a) => reg!(*d) = reg!(*a).clone(),
            Instr::GetGlobal(d, g) => match &self.globals.at(*g).value {
                Some(v) => reg!(*d) = v.clone(),
                None => {
                    let n = self.name(self.globals.at(*g).name).to_string();
                    return rt_err(format!("`{n}` n'est pas défini"), span);
                }
            },
            Instr::SetGlobal(g, a) => {
                let v = reg!(*a).clone();
                self.set_global(*g, v, span)?;
            }
            Instr::NewCell(c) => cells[*c as usize] = Rc::new(RefCell::new(Value::Uninit)),
            Instr::GetCell(d, c) => {
                let v = cells[*c as usize].borrow().clone();
                reg!(*d) = init_check(v, span)?;
            }
            Instr::SetCell(c, a) => *cells[*c as usize].borrow_mut() = reg!(*a).clone(),
            Instr::GetUp(d, u) => {
                let v = upvals[*u as usize].borrow().clone();
                reg!(*d) = init_check(v, span)?;
            }
            Instr::SetUp(u, a) => *upvals[*u as usize].borrow_mut() = reg!(*a).clone(),
            Instr::Guard(d, a, ty) => {
                let v = reg!(*a).clone();
                reg!(*d) = self.guard(v, *ty, span)?;
            }
            Instr::IBin(op, d, a, b) => {
                let (Value::Int(x), Value::Int(y)) = (&reg!(*a), &reg!(*b)) else {
                    unreachable!("registre int qui ne contient pas un int")
                };
                reg!(*d) = int_op(*op, *x, *y, span)?;
            }
            Instr::IBinK(op, d, a, k) => {
                let Value::Int(x) = reg!(*a) else {
                    unreachable!("registre int qui ne contient pas un int")
                };
                reg!(*d) = int_op(*op, x, *k, span)?;
            }
            Instr::FBin(op, d, a, b) => {
                let (Value::Float(x), Value::Float(y)) = (&reg!(*a), &reg!(*b)) else {
                    unreachable!("registre float qui ne contient pas un float")
                };
                reg!(*d) = float_op(*op, *x, *y);
            }
            Instr::INeg(d, a) => {
                let Value::Int(x) = reg!(*a) else {
                    unreachable!()
                };
                reg!(*d) = match x.checked_neg() {
                    Some(n) => Value::Int(n),
                    None => return rt_err("dépassement de capacité d'un entier", span),
                };
            }
            Instr::IToF(d, a) => {
                let Value::Int(x) = reg!(*a) else {
                    unreachable!()
                };
                reg!(*d) = Value::Float(x as f64);
            }

            Instr::Bin(op, d, a, b) => {
                // Cas courant du code non annoté : deux entiers.
                if let (Value::Int(x), Value::Int(y)) = (&reg!(*a), &reg!(*b))
                    && !matches!(op, BinOp::Push | BinOp::And | BinOp::Or)
                {
                    reg!(*d) = int_op(*op, *x, *y, span)?;
                    return Ok(());
                }
                let (l, r) = (reg!(*a).clone(), reg!(*b).clone());
                reg!(*d) = self.binary(*op, l, r, span)?;
            }
            Instr::Not(d, a) => reg!(*d) = Value::Bool(!reg!(*a).truthy()),

            Instr::Index(d, a, i) => {
                let (r, i) = (reg!(*a).clone(), reg!(*i).clone());
                reg!(*d) = self.index(&r, &i, span)?;
            }
            Instr::Field(d, a, name) => {
                let r = reg!(*a).clone();
                reg!(*d) = self.field(&r, *name, span)?;
            }
            Instr::FieldAs(d, a, name, t) => {
                let r = reg!(*a).clone();
                let v = self.field(&r, *name, span)?;
                reg!(*d) = self.field_as(v, *t, *name, span)?;
            }
            Instr::FieldAt(d, a, idx) => {
                let Value::Object(o) = &reg!(*a) else {
                    return rt_err("`@champ` : `self` n'est pas un objet", span);
                };
                let v = o.borrow().fields[*idx as usize].clone();
                if let Value::Uninit = v {
                    let n = self
                        .name(o.borrow().info.fields[*idx as usize].name)
                        .to_string();
                    return rt_err(format!("`@{n}` est lu avant d'être initialisé"), span);
                }
                reg!(*d) = v;
            }
            Instr::Call { dst, f, site } => self.call_instr(proto, regs, *dst, *f, *site, span)?,
            Instr::CallMethod {
                dst,
                recv,
                name,
                site,
            } => {
                let site = &proto.calls[*site as usize];
                let r = reg!(*recv).clone();
                // Valeur sans méthode ni champ déclarés (chaîne, liste, nombre…) : méthode
                // intégrée, directement.
                if !matches!(
                    r,
                    Value::Object(_) | Value::Record(_) | Value::Variant(..) | Value::Enum(_)
                ) && site.named.is_empty()
                    && !site.block
                {
                    let start = site.start as usize;
                    let args = regs[start..start + site.npos as usize].to_vec();
                    reg!(*dst) = self.method(r, *name, args, span)?;
                    return Ok(());
                }
                let args = Self::call_args(proto, regs, site);
                reg!(*dst) = self.call_method_named(r, *name, args, span)?;
            }

            // Instructions moins fréquentes : hors de la boucle principale, pour
            // qu'elle reste petite et bien optimisée.
            other => self.exec_slow(proto, upvals, ctx, regs, cells, other, span)?,
        }
        Ok(())
    }

    /// Instruction `Call` (aussi appelée par le code natif de base).
    #[inline]
    pub(crate) fn call_instr(
        &mut self,
        proto: &Proto,
        regs: &mut [Value],
        dst: Reg,
        f: Reg,
        site: u32,
        span: Span,
    ) -> R<()> {
        macro_rules! reg {
            ($r:expr) => {
                regs[$r as usize]
            };
        }
        let site = &proto.calls[site as usize];
        // Cas courant : fonction du script, arguments positionnels exacts. La
        // fonction et les arguments sont lus en place dans les registres.
        if !site.block
            && site.named.is_empty()
            && let Value::Function(func) = &reg!(f)
            && let Function::Script(c) = &**func
            && c.proto.params.len() == site.npos as usize
            && site.shapes.iter().all(Option::is_none)
        {
            let start = site.start as usize;
            let v = self.invoke_regs(c, &regs[start..start + site.npos as usize], span)?;
            reg!(dst) = v;
            return Ok(());
        }
        // Construction d'un `type` dont chaque champ est donné : directe.
        if !site.block
            && site.shapes.iter().all(Option::is_none)
            && let Value::Type(info) = &reg!(f)
            && info.kind != crate::ast::TypeDeclKind::Class
            && info.fields.len() == site.npos as usize + site.named.len()
        {
            let start = site.start as usize;
            let info = info.clone();
            let v = self.construct_regs(
                &info,
                &regs[start..start + info.fields.len()],
                &site.named,
                span,
            )?;
            reg!(dst) = v;
            return Ok(());
        }
        let f = reg!(f).clone();
        let mut args = Self::call_args(proto, regs, site);
        let v = match (&f, site.block) {
            // Construction : le bloc `do` remplit les champs.
            (Value::Type(info), true) => {
                let block = args.pos.pop().unwrap();
                if !args.shapes.is_empty() {
                    args.shapes.remove(site.npos as usize - 1);
                }
                let Value::Function(func) = block else {
                    unreachable!()
                };
                let Function::Script(c) = &*func else {
                    unreachable!()
                };
                self.construct(info, args, Some(c), span)?
            }
            _ => self.call_value(&f, args, span)?,
        };
        reg!(dst) = v;
        Ok(())
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn exec_slow(
        &mut self,
        proto: &Proto,
        upvals: &[Cell],
        ctx: &Rc<[u32]>,
        regs: &mut [Value],
        cells: &mut [Cell],
        ins: &Instr,
        span: Span,
    ) -> R<()> {
        macro_rules! reg {
            ($r:expr) => {
                regs[$r as usize]
            };
        }
        match ins {
            Instr::Fail(c) => {
                return rt_err(proto.consts[*c as usize].to_display(&self.interner), span);
            }

            Instr::DefGlobal { slot, src, mutable } => {
                let v = reg!(*src).clone();
                self.globals.set_slot(*slot, v, *mutable);
            }
            Instr::GetNamed {
                dst,
                name,
                fallback,
            } => {
                let v = match self.ctx_lookup(ctx, *name) {
                    Some((id, i)) => self.field_ref_get(id, i, span)?,
                    None => match *fallback {
                        Fallback::Reg(r) => reg!(r).clone(),
                        Fallback::Cell(c) => init_check(cells[c as usize].borrow().clone(), span)?,
                        Fallback::Up(u) => init_check(upvals[u as usize].borrow().clone(), span)?,
                        Fallback::Global(g) => match &self.globals.at(g).value {
                            Some(v) => v.clone(),
                            None => {
                                return rt_err(
                                    format!("`{}` n'est pas défini", self.name(*name)),
                                    span,
                                );
                            }
                        },
                        Fallback::None => {
                            return rt_err(
                                format!("`{}` n'est pas défini", self.name(*name)),
                                span,
                            );
                        }
                    },
                };
                reg!(*dst) = v;
            }
            Instr::SetNamed {
                name,
                src,
                fallback,
                shape,
            } => {
                let v = reg!(*src).clone();
                match self.ctx_lookup(ctx, *name) {
                    Some((id, i)) => {
                        let shape = shape.map(|s| proto.shapes[s as usize].clone());
                        self.field_ref_set(id, i, v, shape.as_deref(), span)?;
                    }
                    None => match *fallback {
                        Fallback::Reg(r) => reg!(r) = v,
                        Fallback::Cell(c) => *cells[c as usize].borrow_mut() = v,
                        Fallback::Up(u) => *upvals[u as usize].borrow_mut() = v,
                        Fallback::Global(g) => self.set_global(g, v, span)?,
                        Fallback::None => {
                            let n = self.name(*name).to_string();
                            let msg = match ctx.last().and_then(|id| self.construct_type_name(*id))
                            {
                                Some(t) => format!(
                                    "`{n}` n'est ni un champ de `{t}` ni une variable déclarée \
                                             (utiliser `let {n} = …` pour une variable locale)"
                                ),
                                None => format!("`{n}` n'est pas déclaré : utiliser `let {n} = …`"),
                            };
                            return rt_err(msg, span);
                        }
                    },
                }
            }
            Instr::CtxField(d, i) => {
                let id = *ctx.last().expect("valeur par défaut hors construction");
                reg!(*d) = self.field_ref_get(id, *i, span)?;
            }
            Instr::Coerce {
                dst,
                src,
                ty,
                shape,
            } => {
                let v = reg!(*src).clone();
                let shape = shape.map(|s| proto.shapes[s as usize].clone());
                reg!(*dst) = self.coerce(v, &proto.types[*ty as usize], shape.as_deref(), span)?;
            }
            Instr::IsUndef(d, a) => reg!(*d) = Value::Bool(matches!(reg!(*a), Value::Uninit)),

            Instr::FNeg(d, a) => {
                let Value::Float(x) = reg!(*a) else {
                    unreachable!()
                };
                reg!(*d) = Value::Float(-x);
            }
            Instr::Neg(d, a) => {
                reg!(*d) = match &reg!(*a) {
                    Value::Int(n) => match n.checked_neg() {
                        Some(n) => Value::Int(n),
                        None => return rt_err("dépassement de capacité d'un entier", span),
                    },
                    Value::Float(f) => Value::Float(-f),
                    v => {
                        return type_err(
                            format!("`-` attend un nombre, reçu {}", v.type_label()),
                            span,
                        );
                    }
                };
            }
            Instr::List(d, s, n) => {
                let s = *s as usize;
                reg!(*d) = Value::list(regs[s..s + *n as usize].to_vec());
            }
            Instr::Tuple(d, s, n) => {
                let s = *s as usize;
                reg!(*d) = Value::Tuple(regs[s..s + *n as usize].into());
            }
            Instr::Hash(d, s, n) => {
                let mut h = HashMap::new();
                for k in 0..*n as usize {
                    let key = self.hash_key(&regs[*s as usize + 2 * k], span)?;
                    h.insert(key, regs[*s as usize + 2 * k + 1].clone());
                }
                reg!(*d) = Value::Hash(Rc::new(RefCell::new(h)));
            }
            Instr::Record(d, s, r) => {
                let names = &proto.records[*r as usize];
                let fields = names
                    .iter()
                    .enumerate()
                    .map(|(k, n)| (*n, regs[*s as usize + k].clone()))
                    .collect();
                reg!(*d) = Value::Record(Rc::new(RefCell::new(Record { fields })));
            }
            Instr::Range {
                dst,
                lo,
                hi,
                inclusive,
            } => {
                reg!(*dst) = match (&reg!(*lo), &reg!(*hi)) {
                    (Value::Int(lo), Value::Int(hi)) => Value::Range(Rc::new(Range {
                        lo: *lo,
                        hi: *hi,
                        inclusive: *inclusive,
                    })),
                    (l, h) => {
                        return type_err(
                            format!(
                                "un intervalle attend deux entiers, reçu {} et {}",
                                l.type_label(),
                                h.type_label()
                            ),
                            span,
                        );
                    }
                };
            }
            Instr::Concat(d, s, n) => {
                // Tampon réutilisé : une seule allocation, celle de la chaîne finale.
                let mut out = std::mem::take(&mut self.concat_buf);
                out.clear();
                for k in 0..*n as usize {
                    let v = &regs[*s as usize + k];
                    if let Value::Str(x) = v {
                        out.push_str(x);
                    } else if !v.write_scalar(&mut out) {
                        let v = v.clone();
                        match self.display(&v, span) {
                            Ok(t) => out.push_str(&t),
                            Err(e) => {
                                self.concat_buf = out;
                                return Err(e);
                            }
                        }
                    }
                }
                reg!(*d) = Value::Str(Rc::from(out.as_str()));
                self.concat_buf = out;
            }
            Instr::SetIndex(a, i, v) => {
                let (r, i, v) = (reg!(*a).clone(), reg!(*i).clone(), reg!(*v).clone());
                self.set_index(&r, &i, v, span)?;
            }
            Instr::SetField {
                obj,
                name,
                src,
                shape,
            } => {
                let (o, v) = (reg!(*obj).clone(), reg!(*src).clone());
                let shape = shape.map(|s| proto.shapes[s as usize].clone());
                self.set_field(&o, *name, v, shape.as_deref(), span)?;
            }
            Instr::SetFieldAt {
                obj,
                idx,
                src,
                shape,
            } => {
                let (o, v) = (reg!(*obj).clone(), reg!(*src).clone());
                let shape = shape.map(|s| proto.shapes[s as usize].clone());
                self.set_field_at(&o, *idx, v, shape.as_deref(), span)?;
            }
            Instr::TupleIndex(d, a, idx) => {
                reg!(*d) = match &reg!(*a) {
                    Value::Tuple(t) => match t.get(*idx as usize) {
                        Some(v) => v.clone(),
                        None => {
                            return rt_err(
                                format!("index {idx} hors du tuple (taille {})", t.len()),
                                span,
                            );
                        }
                    },
                    r => {
                        return type_err(
                            format!("`.{idx}` s'applique à un tuple, reçu {}", r.type_label()),
                            span,
                        );
                    }
                };
            }
            Instr::Variant(d, a, name) => {
                let e = reg!(*a).clone();
                reg!(*d) = self.variant(&e, *name, span)?;
            }
            Instr::Unpack { dst, src, n, rest } => {
                let v = reg!(*src).clone();
                let items: Vec<Value> = match &v {
                    Value::Tuple(t) => t.to_vec(),
                    Value::List(l) => l.borrow().clone(),
                    v => {
                        return type_err(
                            format!("tuple ou liste attendu, reçu {}", v.type_label()),
                            span,
                        );
                    }
                };
                let n = *n as usize;
                if items.len() < n || (!*rest && items.len() != n) {
                    return rt_err(
                        format!(
                            "{}{n} valeurs attendues, reçu {}",
                            if *rest { "au moins " } else { "" },
                            items.len()
                        ),
                        span,
                    );
                }
                let mut it = items.into_iter();
                for k in 0..n {
                    reg!(*dst + k as u32) = it.next().unwrap();
                }
                if *rest {
                    reg!(*dst + n as u32) = Value::list(it.collect());
                }
            }

            Instr::Closure(d, p) => {
                let child = &proto.children[*p as usize];
                let ups: Box<[Cell]> = child
                    .captures
                    .iter()
                    .map(|c| match c {
                        Capture::Cell(c) => cells[*c as usize].clone(),
                        Capture::Up(u) => upvals[*u as usize].clone(),
                    })
                    .collect();
                reg!(*d) = Value::Function(Rc::new(Function::Script(Closure {
                    proto: child.clone(),
                    upvals: ups,
                    ctx: ctx.clone(),
                })));
            }
            Instr::IterNew { dst, src, keys } => {
                let v = reg!(*src).clone();
                let it = self.iter_new(&v, *keys, span)?;
                reg!(*dst) = Value::Iter(Rc::new(RefCell::new(it)));
            }
            Instr::CaseTest(d, p, v) => {
                let hit = match (&reg!(*p), &reg!(*v)) {
                    (Value::Range(r), Value::Int(n)) => r.contains(*n),
                    (p, v) => p.equals(v),
                };
                reg!(*d) = Value::Bool(hit);
            }
            Instr::DefType { slot, decl } => {
                let Decl::Type(t) = &proto.decls[*decl as usize] else {
                    unreachable!()
                };
                let info = TypeInfo::new(t, &proto.script, &self.interner);
                self.globals
                    .set_slot(*slot, Value::Type(Rc::new(info)), false);
            }
            Instr::DefEnum { slot, decl, start } => {
                let Decl::Enum(e) = &proto.decls[*decl as usize] else {
                    unreachable!()
                };
                let explicit = regs[*start as usize..].to_vec();
                let info = self.enum_info(e, &explicit, &proto.script, span)?;
                self.globals
                    .set_slot(*slot, Value::Enum(Rc::new(info)), false);
            }
            _ => unreachable!("instruction traitée dans la boucle principale"),
        }
        Ok(())
    }

    /// Appel direct d'une fonction déjà compilée en code de base, sans ce dont elle n'a
    /// pas besoin : liaison générique (paramètres non annotés), essai du JIT typé,
    /// cellules. `None` : chemin général (dont la vérification de la pile, un niveau
    /// d'appel sur `STACK_CHECK`).
    #[cfg(feature = "jit")]
    #[inline]
    fn invoke_native(&mut self, c: &Closure, args: &[Value], span: Span) -> Option<R> {
        let proto = &c.proto;
        let entry = proto.jit_dyn.get();
        if entry <= 1
            || proto.ncells != 0
            || !proto.plain_params
            || self.depth >= self.max_depth
            || (self.depth + 1).is_multiple_of(STACK_CHECK)
            || (proto.kind == ProtoKind::Function
                && !matches!(*proto.jit.borrow(), JitState::Rejected))
        {
            return None;
        }
        self.depth += 1;
        let mut regs = self.frames.pop().unwrap_or_default();
        fill_nil(&mut regs, proto.regs.len());
        for (p, v) in proto.params.iter().zip(args) {
            regs[p.reg as usize] = v.clone();
        }
        let r = self.run_native(entry, proto, &c.upvals, &c.ctx, &mut regs, &mut []);
        regs.clear();
        self.frames.push(regs);
        self.depth -= 1;
        Some(r.map_err(|e| self.traced(e, proto, span)))
    }

    /// Point d'entrée du code natif de base de `proto`, compilé au deuxième appel (ou au
    /// premier s'il contient une boucle).
    #[cfg(feature = "jit")]
    #[inline]
    fn dyn_entry(&mut self, proto: &Proto) -> Option<usize> {
        match proto.jit_dyn.get() {
            0 => {}
            1 => return None,
            e => return Some(e),
        }
        if !self.jit_enabled {
            return None;
        }
        let n = proto.calls_seen.get() + 1;
        proto.calls_seen.set(n);
        if n < 2 && !proto.has_loop && !self.jit_eager {
            return None;
        }
        if self.jit.is_none() {
            match crate::jit::Jit::new() {
                Some(j) => self.jit = Some(j),
                None => {
                    self.jit_enabled = false;
                    return None;
                }
            }
        }
        let e = self.jit.as_mut()?.compile_dynamic(proto).unwrap_or(1);
        proto.jit_dyn.set(e);
        (e != 1).then_some(e)
    }

    #[cfg(feature = "jit")]
    pub(crate) fn set_pending(&mut self, e: Error) {
        self.pending = Some(e);
    }

    #[cfg(feature = "jit")]
    fn run_native(
        &mut self,
        entry: usize,
        proto: &Proto,
        upvals: &[Cell],
        ctx: &Rc<[u32]>,
        regs: &mut [Value],
        cells: &mut [Cell],
    ) -> R {
        let mut f = crate::ir::NativeFrame {
            regs: regs.as_mut_ptr(),
            nregs: regs.len(),
            cells: cells.as_mut_ptr(),
            ncells: cells.len(),
            upvals: upvals.as_ptr(),
            nupvals: upvals.len(),
            ctx,
            proto,
            ret: 0,
            ret_shape: 0,
        };
        // SAFETY : `entry` vient de `compile_dynamic` pour ce `proto` ; le code natif
        // n'accède à `regs`, `cells`, `upvals` que par la fenêtre `f`.
        let code: extern "C" fn(*mut Runtime, *mut crate::ir::NativeFrame) -> i32 =
            unsafe { std::mem::transmute(entry) };
        if code(self, &mut f) != 0 {
            return Err(self.pending.take().expect("erreur du code natif"));
        }
        let v = std::mem::replace(&mut regs[f.ret as usize], Value::Nil);
        match &proto.ret {
            Some(ty) => {
                let shape =
                    (f.ret_shape != u32::MAX).then(|| proto.shapes[f.ret_shape as usize].clone());
                self.coerce(v, ty, shape.as_deref(), ty.span)
            }
            None => Ok(v),
        }
    }

    /// Exécute `proto` en natif s'il est (ou devient) compilé ; `None` sinon.
    #[cfg(feature = "jit")]
    fn try_jit(&mut self, proto: &Rc<Proto>, regs: &[Value]) -> Option<R> {
        if !self.jit_enabled || proto.kind != ProtoKind::Function {
            return None;
        }
        if matches!(*proto.jit.borrow(), JitState::Untried) {
            if self.jit.is_none() {
                self.jit = crate::jit::Jit::new();
                if self.jit.is_none() {
                    // Machine non prise en charge par Cranelift.
                    self.jit_enabled = false;
                    return None;
                }
            }
            let jit = self.jit.as_mut()?;
            jit.len_sym = self.syms.len;
            jit.compile(proto, &mut self.globals);
        }
        let state = proto.jit.borrow();
        let JitState::Native(code) = &*state else {
            return None;
        };
        let args: Vec<Value> = proto
            .params
            .iter()
            .map(|p| regs[p.reg as usize].clone())
            .collect();
        let jit = self.jit.as_mut()?;
        let r = jit.call(code, &args, self.depth, self.max_depth);
        if r.is_none() && jit.bailed {
            // Hypothèse non vérifiée : la fonction reste interprétée.
            drop(state);
            *proto.jit.borrow_mut() = JitState::Rejected;
        }
        r
    }

    // ─── Aides de l'exécution ────────────────────────────────────────────────

    fn set_global(&mut self, g: u32, v: Value, span: Span) -> R<()> {
        let slot = self.globals.at(g);
        let n = slot.name;
        match (&slot.value, slot.mutable) {
            (None, _) => {
                let n = self.name(n).to_string();
                rt_err(
                    format!("`{n}` n'est pas déclaré : utiliser `let {n} = …`"),
                    span,
                )
            }
            (Some(_), false) => rt_err(
                format!(
                    "`{}` ne peut pas être modifié (constante, fonction ou type)",
                    self.name(n)
                ),
                span,
            ),
            (Some(_), true) => {
                self.globals.assign(g, v);
                Ok(())
            }
        }
    }

    /// Valeur dynamique rangée dans un registre typé.
    /// Valeur d'un champ dont le vérificateur connaît le type. Ne peut échouer que si le
    /// type a été redéclaré depuis la vérification (REPL).
    fn field_as(&self, v: Value, ty: RegTy, name: Symbol, span: Span) -> R {
        match (ty, &v) {
            (RegTy::Int, Value::Int(_))
            | (RegTy::Float, Value::Float(_))
            | (RegTy::Bool, Value::Bool(_))
            | (RegTy::Value, _) => Ok(v),
            (RegTy::Float, Value::Int(i)) => Ok(Value::Float(*i as f64)),
            (t, v) => type_err(
                format!(
                    "le champ `{}` devait être un `{}`, reçu {} : son type a-t-il été redéclaré ?",
                    self.name(name),
                    t.name(),
                    v.type_label()
                ),
                span,
            ),
        }
    }

    fn guard(&self, v: Value, ty: RegTy, span: Span) -> R {
        match (ty, &v) {
            (RegTy::Int, Value::Int(_))
            | (RegTy::Float, Value::Float(_))
            | (RegTy::Bool, Value::Bool(_)) => Ok(v),
            (RegTy::Float, Value::Int(i)) => Ok(Value::Float(*i as f64)),
            (RegTy::Value, _) => Ok(v),
            (t, v) => type_err(
                format!(
                    "type attendu `{}`, reçu {} ({}) : une variable garde le type de sa première valeur",
                    match t {
                        RegTy::Int => "int",
                        RegTy::Float => "float",
                        _ => "bool",
                    },
                    v.type_label(),
                    short(&v.inspect(&self.interner))
                ),
                span,
            ),
        }
    }

    fn set_index(&mut self, r: &Value, i: &Value, v: Value, span: Span) -> R<()> {
        match r {
            Value::List(l) => {
                let mut l = l.borrow_mut();
                let pos = list_index(i, l.len(), span)?;
                l[pos] = v;
            }
            Value::Hash(h) => {
                let key = self.hash_key(i, span)?;
                h.borrow_mut().insert(key, v);
            }
            _ => {
                return type_err(
                    format!("impossible d'assigner par index sur {}", r.type_label()),
                    span,
                );
            }
        }
        Ok(())
    }

    fn set_field(
        &mut self,
        obj: &Value,
        name: Symbol,
        v: Value,
        shape: Option<&Shape>,
        span: Span,
    ) -> R<()> {
        match obj {
            Value::Record(rec) => match rec.borrow_mut().get_mut(name) {
                Some(slot) => {
                    *slot = v;
                    Ok(())
                }
                None => rt_err(
                    format!("le record n'a pas de champ `{}`", self.name(name)),
                    span,
                ),
            },
            _ => self.object_set_field(obj, name, v, shape, span),
        }
    }

    fn set_field_at(
        &mut self,
        obj: &Value,
        idx: u32,
        v: Value,
        shape: Option<&Shape>,
        span: Span,
    ) -> R<()> {
        let Value::Object(o) = obj else {
            return rt_err("`@champ` : `self` n'est pas un objet", span);
        };
        let info = o.borrow().info.clone();
        let v = match &info.fields[idx as usize].ty {
            Some(ty) => self.coerce(v, ty, shape, span)?,
            None => v,
        };
        o.borrow_mut().fields[idx as usize] = v;
        Ok(())
    }

    fn variant(&self, e: &Value, name: Symbol, span: Span) -> R {
        match e {
            Value::Enum(info) => match info.variants.iter().position(|(n, _)| *n == name) {
                Some(i) => Ok(Value::Variant(info.clone(), i as u32)),
                None => rt_err(
                    format!("`{}` n'a pas de variante `{}`", info.name, self.name(name)),
                    span,
                ),
            },
            v => type_err(
                format!("`::` s'applique à un enum, reçu {}", v.type_label()),
                span,
            ),
        }
    }

    fn iter_new(&self, v: &Value, keys: bool, span: Span) -> R<IterState> {
        Ok(match v {
            Value::List(l) => IterState::Values(l.borrow().clone().into_iter()),
            Value::Tuple(t) => IterState::Values(t.iter().cloned().collect::<Vec<_>>().into_iter()),
            Value::Range(r) => IterState::Range(r.values()),
            Value::Str(s) => IterState::Values(
                s.chars()
                    .map(|c| Value::Str(c.to_string().into()))
                    .collect::<Vec<_>>()
                    .into_iter(),
            ),
            Value::Hash(h) => {
                let items: Vec<Value> = h
                    .borrow()
                    .iter()
                    .map(|(k, v)| {
                        if keys {
                            k.to_value()
                        } else {
                            Value::Tuple(vec![k.to_value(), v.clone()].into())
                        }
                    })
                    .collect();
                IterState::Values(items.into_iter())
            }
            v => {
                return type_err(
                    format!(
                        "`for` ne peut pas parcourir une valeur de type {}",
                        v.type_label()
                    ),
                    span,
                );
            }
        })
    }

    /// `recv.nom(args)` : champ fonction, méthode déclarée, méthode d'enum ou intégrée.
    fn call_method_named(&mut self, r: Value, name: Symbol, args: CallArgs, span: Span) -> R {
        let field_fn = match &r {
            Value::Record(rec) => rec.borrow().get(name).cloned(),
            Value::Object(o) => {
                let o = o.borrow();
                match (o.info.kind, o.info.field_index(name)) {
                    (TypeDeclKind::Class, _) | (_, None) => None,
                    (_, Some(i)) => Some(o.fields[i].clone()),
                }
            }
            _ => None,
        };
        if let Some(f) = field_fn {
            return self.call_value(&f, args, span);
        }
        if let Some(m) = self.user_method(&r, name) {
            return self.call_method(r, &m, args, span);
        }
        if let Value::Enum(e) = &r {
            return self.enum_method(e, name, &args.pos, span);
        }
        if let Some(&(n, _)) = args.named.first() {
            return rt_err(
                format!(
                    "les méthodes intégrées n'acceptent pas d'argument nommé (`{}`)",
                    self.name(n)
                ),
                span,
            );
        }
        self.method(r, name, args.pos, span)
    }

    // ─── Vérification des types ──────────────────────────────────────────────

    /// Vérifie `v` contre une annotation, après conversion éventuelle :
    /// - `int` → `float` ;
    /// - record **littéral** (`shape`) → `type` nominal, y compris dans une liste
    ///   littérale et pour `T?` / `A | B`.
    #[inline]
    pub(crate) fn coerce(
        &mut self,
        v: Value,
        ty: &TypeRef,
        shape: Option<&Shape>,
        span: Span,
    ) -> R {
        // Cas courant : type simple déjà respecté.
        match (&*ty.ty, &v) {
            (RtType::Any, _)
            | (RtType::Int, Value::Int(_))
            | (RtType::Float, Value::Float(_))
            | (RtType::Str, Value::Str(_))
            | (RtType::Bool, Value::Bool(_)) => Ok(v),
            _ => self.coerce_rt(v, &ty.ty, ty, shape, span),
        }
    }

    fn coerce_rt(
        &mut self,
        v: Value,
        rt: &RtType,
        ty: &TypeRef,
        shape: Option<&Shape>,
        span: Span,
    ) -> R {
        match (shape, rt, &v) {
            (Some(Shape::Record(fs)), RtType::Named(..), Value::Record(r)) => {
                if let Some(info) = self.data_type(rt) {
                    let r = r.borrow().clone();
                    return self.record_to_object(&info, &r, fs, span);
                }
            }
            (Some(Shape::Record(_)), RtType::Optional(inner), Value::Record(_)) => {
                return self.coerce_rt(v, inner, ty, shape, span);
            }
            (Some(Shape::Record(fs)), RtType::Union(items), Value::Record(r)) => {
                for it in items {
                    if let Some(info) = self.data_type(it) {
                        let r = r.borrow().clone();
                        if let Ok(obj) = self.record_to_object(&info, &r, fs, span) {
                            return Ok(obj);
                        }
                    }
                }
            }
            (Some(Shape::List(shapes)), RtType::List(inner), Value::List(l)) => {
                let elems = l.borrow().clone();
                let mut out = Vec::with_capacity(elems.len());
                for (k, x) in elems.into_iter().enumerate() {
                    let s = shapes.get(k).cloned().flatten();
                    out.push(self.coerce_rt(x, inner, ty, s.as_deref(), span)?);
                }
                return Ok(Value::list(out));
            }
            _ => {}
        }
        if let (RtType::Float, Value::Int(i)) = (rt, &v) {
            return Ok(Value::Float(*i as f64));
        }
        match self.matches_rt(&v, rt) {
            Ok(true) => Ok(v),
            Ok(false) => type_err(
                format!(
                    "type attendu `{}`, reçu {} ({})",
                    ty.shown,
                    v.type_label(),
                    short(&v.inspect(&self.interner))
                ),
                span,
            ),
            Err(msg) => type_err(msg, ty.span),
        }
    }

    /// `type` (non `class`) désigné par un type nommé.
    fn data_type(&self, rt: &RtType) -> Option<Rc<TypeInfo>> {
        match rt {
            RtType::Named(slot, _) if *slot != u32::MAX => match &self.globals.at(*slot).value {
                Some(Value::Type(info)) if info.kind != TypeDeclKind::Class => Some(info.clone()),
                _ => None,
            },
            _ => None,
        }
    }

    fn record_to_object(
        &mut self,
        info: &Rc<TypeInfo>,
        rec: &Record,
        fields: &[(Symbol, Option<Rc<Shape>>)],
        span: Span,
    ) -> R {
        let shapes = rec
            .fields
            .iter()
            .map(|(n, _)| {
                fields
                    .iter()
                    .find(|(m, _)| m == n)
                    .and_then(|(_, s)| s.clone())
            })
            .collect();
        let args = CallArgs {
            pos: Vec::new(),
            named: rec.fields.clone(),
            shapes,
        };
        self.construct(info, args, None, span)
    }

    pub(crate) fn matches_rt(&self, v: &Value, rt: &RtType) -> Result<bool, String> {
        Ok(match rt {
            RtType::Any => true,
            RtType::Int => matches!(v, Value::Int(_)),
            RtType::Float => matches!(v, Value::Float(_)),
            RtType::Str => matches!(v, Value::Str(_)),
            RtType::Bool => matches!(v, Value::Bool(_)),
            RtType::Nil => matches!(v, Value::Nil),
            RtType::Fn => matches!(v, Value::Function(_) | Value::Type(_)),
            RtType::LitStr(s) => matches!(v, Value::Str(x) if **x == **s),
            RtType::LitInt(n) => matches!(v, Value::Int(x) if x == n),
            RtType::Named(slot, name) => {
                let found = if *slot == u32::MAX {
                    None
                } else {
                    self.globals.at(*slot).value.as_ref()
                };
                match found {
                    Some(Value::Type(info)) => {
                        matches!(v, Value::Object(o) if Rc::ptr_eq(&o.borrow().info, info))
                    }
                    Some(Value::Enum(e)) => matches!(v, Value::Variant(x, _) if Rc::ptr_eq(x, e)),
                    _ => return Err(format!("type `{}` inconnu", self.name(*name))),
                }
            }
            RtType::List(inner) => match v {
                Value::List(l) => {
                    for item in l.borrow().iter() {
                        if !self.matches_rt(item, inner)? {
                            return Ok(false);
                        }
                    }
                    true
                }
                _ => false,
            },
            RtType::Hash(kt, vt) => match v {
                Value::Hash(h) => {
                    for (k, val) in h.borrow().iter() {
                        if !self.matches_rt(&k.to_value(), kt)? || !self.matches_rt(val, vt)? {
                            return Ok(false);
                        }
                    }
                    true
                }
                _ => false,
            },
            RtType::Record(fields) => match v {
                Value::Record(r) => {
                    let r = r.borrow();
                    if r.fields.len() != fields.len() {
                        return Ok(false);
                    }
                    for (name, fty) in fields {
                        match r.get(*name) {
                            Some(fv) if self.matches_rt(fv, fty)? => {}
                            _ => return Ok(false),
                        }
                    }
                    true
                }
                _ => false,
            },
            RtType::Tuple(items) => match v {
                Value::Tuple(t) => {
                    if t.len() != items.len() {
                        return Ok(false);
                    }
                    for (val, ity) in t.iter().zip(items) {
                        if !self.matches_rt(val, ity)? {
                            return Ok(false);
                        }
                    }
                    true
                }
                _ => false,
            },
            RtType::Optional(inner) => matches!(v, Value::Nil) || self.matches_rt(v, inner)?,
            RtType::Union(items) => {
                for it in items {
                    if self.matches_rt(v, it)? {
                        return Ok(true);
                    }
                }
                false
            }
        })
    }

    fn enum_info(
        &self,
        e: &EnumDeclIr,
        explicit: &[Value],
        script: &Rc<Script>,
        span: Span,
    ) -> R<EnumInfo> {
        let mut values = explicit.iter();
        let mut variants = Vec::with_capacity(e.variants.len());
        for (i, (name, has)) in e.variants.iter().zip(&e.explicit).enumerate() {
            // Sans `= valeur` : le nom pour un enum `: str`, sinon l'index.
            let value = if *has {
                values.next().cloned().unwrap_or(Value::Nil)
            } else if e.repr_is_str {
                Value::str(self.name(*name))
            } else {
                Value::Int(i as i64)
            };
            if let Some(repr) = &e.repr
                && !self.matches_rt(&value, &repr.ty).unwrap_or(false)
            {
                return type_err(
                    format!(
                        "la valeur de `{}` doit être de type `{}`, reçu {}",
                        self.name(*name),
                        repr.shown,
                        value.type_label()
                    ),
                    span,
                );
            }
            variants.push((*name, value));
        }
        Ok(EnumInfo {
            name: self.name(e.name).into(),
            variants,
            methods: e.methods.iter().cloned().collect(),
            script: script.clone(),
            annotations: e.annotations.clone(),
            variant_annotations: e.variant_annotations.clone(),
        })
    }
}

fn init_check(v: Value, span: Span) -> R {
    match v {
        Value::Uninit => rt_err("variable utilisée avant sa déclaration", span),
        v => Ok(v),
    }
}

fn int_op(op: BinOp, x: i64, y: i64, span: Span) -> R {
    use BinOp::*;
    let overflow = || Error::runtime("dépassement de capacité d'un entier", span);
    Ok(match op {
        Add => Value::Int(x.checked_add(y).ok_or_else(overflow)?),
        Sub => Value::Int(x.checked_sub(y).ok_or_else(overflow)?),
        Mul => Value::Int(x.checked_mul(y).ok_or_else(overflow)?),
        Div | Rem if y == 0 => return rt_err("division par zéro", span),
        Div => Value::Int(x.checked_div_euclid(y).ok_or_else(overflow)?),
        Rem => Value::Int(x.checked_rem_euclid(y).ok_or_else(overflow)?),
        Eq => Value::Bool(x == y),
        Ne => Value::Bool(x != y),
        Lt => Value::Bool(x < y),
        Le => Value::Bool(x <= y),
        Gt => Value::Bool(x > y),
        Ge => Value::Bool(x >= y),
        And | Or | Push => unreachable!("opération non entière"),
    })
}

#[inline]
fn int_cmp(op: BinOp, x: i64, y: i64) -> bool {
    match op {
        BinOp::Eq => x == y,
        BinOp::Ne => x != y,
        BinOp::Lt => x < y,
        BinOp::Le => x <= y,
        BinOp::Gt => x > y,
        BinOp::Ge => x >= y,
        _ => unreachable!("comparaison attendue"),
    }
}

fn float_op(op: BinOp, x: f64, y: f64) -> Value {
    use BinOp::*;
    match op {
        Add => Value::Float(x + y),
        Sub => Value::Float(x - y),
        Mul => Value::Float(x * y),
        Div => Value::Float(x / y),
        Rem => Value::Float(x.rem_euclid(y)),
        Eq => Value::Bool(x == y),
        Ne => Value::Bool(x != y),
        Lt => Value::Bool(x < y),
        Le => Value::Bool(x <= y),
        Gt => Value::Bool(x > y),
        Ge => Value::Bool(x >= y),
        And | Or | Push => unreachable!("opération non décimale"),
    }
}

impl TypeInfo {
    pub(crate) fn new(t: &TypeDeclIr, script: &Rc<Script>, i: &Interner) -> Self {
        TypeInfo {
            name: i.resolve(t.name).into(),
            kind: t.kind,
            fields: t
                .fields
                .iter()
                .map(|f| crate::objects::FieldInfo {
                    name: f.name,
                    ty: f.ty.clone(),
                    default: f.default.clone(),
                    annotations: f.annotations.clone(),
                })
                .collect(),
            methods: t.methods.iter().cloned().collect(),
            annotations: t.annotations.clone(),
            script: script.clone(),
        }
    }
}

/// Avance l'itérateur du registre `iter` : `true` et l'élément dans `dst`, ou `false`.
#[inline]
pub(crate) fn iter_next(regs: &mut [Value], iter: Reg, dst: Reg) -> bool {
    let Value::Iter(it) = &regs[iter as usize] else {
        unreachable!()
    };
    let next = match &mut *it.borrow_mut() {
        IterState::Values(v) => v.next(),
        IterState::Range(r) => r.next().map(Value::Int),
    };
    match next {
        Some(v) => {
            regs[dst as usize] = v;
            true
        }
        None => false,
    }
}

/// Agrandit le vecteur vide `regs` à `n` registres `nil`. `Value::Nil` est fait d'octets
/// nuls (`repr(C, u8)`, étiquette 0) : une mise à zéro suffit, sans boucle de copies.
#[inline]
fn fill_nil(regs: &mut Vec<Value>, n: usize) {
    debug_assert!(regs.is_empty());
    regs.reserve(n);
    // SAFETY : la capacité est d'au moins `n` ; des octets nuls forment `Value::Nil`
    // (vérifié par `value::tests::native_layout`), une valeur valide qui ne possède rien.
    unsafe {
        std::ptr::write_bytes(regs.as_mut_ptr(), 0, n);
        regs.set_len(n);
    }
}
