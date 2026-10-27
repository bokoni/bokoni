//! Compilation native (JIT) avec Cranelift.
//!
//! Sont compilées les fonctions dont les paramètres et le retour sont annotés `int`,
//! `float` ou `bool`, et dont le corps n'utilise que de l'arithmétique, des comparaisons,
//! des branchements et des appels à d'autres fonctions compilables (récursion comprise).
//! Tout le reste est exécuté par l'interpréteur : le résultat est le même, plus lentement.
//!
//! - Les registres `valeur` de l'IR sont typés par une analyse : le résultat d'un appel à
//!   une fonction `: int` est un `int`, etc. Si un registre ne peut pas être typé, la
//!   fonction n'est pas compilée.
//! - Mêmes règles qu'à l'interprétation : dépassement de capacité, division par zéro et
//!   profondeur d'appels sont vérifiés, avec les mêmes messages et la même pile d'appels.
//! - Un appel à une fonction globale est direct (code natif → code natif). Si la globale
//!   est redéfinie, le code qui l'appelle est invalidé puis recompilé au besoin.
//!
//! Convention d'appel interne : `fn(ctx, params…) -> (statut, résultat)`, statut 0 = succès,
//! 1 = erreur (décrite dans `ctx`). Chaque fonction a aussi un point d'entrée pour Rust :
//! `extern "C" fn(ctx, *const u64 args, *mut u64 ret) -> i32`.

use crate::ast::BinOp;
use crate::diagnostic::Error;
use crate::globals::Globals;
use crate::intern::FxHashMap;
use crate::ir::NativeFrame;
use crate::ir::{Instr, JitState, NativeCode, Proto, ProtoKind, Reg, RegTy, RtType, Script, Term};
use crate::span::Span;
use crate::value::layout::{FIRST_HEAP_TAG, PAYLOAD, SIZE, TAG_BOOL, TAG_FLOAT, TAG_INT, TAG_NIL};
use crate::value::{Function, Value};
use crate::vm::Runtime;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    AbiParam, InstBuilder, MemFlagsData, Signature, Type, UserFuncName, types,
};
use cranelift_codegen::ir::{Block as CBlock, FuncRef, Value as CValue};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use std::rc::Rc;

/// Contexte partagé avec le code natif. Les trois premiers champs sont lus et écrits par
/// le code généré (positions fixes) ; les suivants ne servent qu'au code Rust.
#[repr(C)]
pub struct JitCtx {
    pub depth: i64,
    pub max_depth: i64,
    /// Erreur survenue : index dans la table des erreurs.
    pub err: i64,
    /// Pile d'appels de l'erreur (index dans la table des appels), plus récent en premier.
    pub traces: Vec<u32>,
}

const DEPTH_OFFSET: i32 = 0;
const MAX_DEPTH_OFFSET: i32 = 8;
const ERR_OFFSET: i32 = 16;

#[derive(Clone, Copy)]
enum ErrKind {
    Overflow,
    DivZero,
    Depth,
}

struct ErrInfo {
    kind: ErrKind,
    span: Span,
    script: Rc<Script>,
}

struct TraceInfo {
    name: Rc<str>,
    span: Span,
}

extern "C" fn bk_trace(ctx: *mut JitCtx, id: u32) {
    // SAFETY : `ctx` est le contexte passé par `Jit::call`, valide pendant l'appel.
    unsafe { (*ctx).traces.push(id) }
}

extern "C" fn bk_frem(a: f64, b: f64) -> f64 {
    a.rem_euclid(b)
}

/// Fonction compilée : identifiant Cranelift et signature.
struct Compiled {
    id: FuncId,
    params: Vec<RegTy>,
    ret: RegTy,
}

pub struct Jit {
    module: JITModule,
    ctx: Box<JitCtx>,
    errors: Vec<ErrInfo>,
    traces: Vec<TraceInfo>,
    /// Fonctions déjà compilées (clé : adresse du `Proto`).
    compiled: FxHashMap<usize, Compiled>,
    trace_fn: FuncId,
    frem_fn: FuncId,
    /// Fonctions d'aide du code de base (`bk_step`, `bk_iter_next`, `bk_drop`).
    dyn_fns: Option<(FuncId, FuncId, FuncId)>,
    next: u32,
}

/// Type d'un registre pour le code natif.
#[derive(Clone, Copy, Debug, PartialEq)]
enum JTy {
    Int,
    Float,
    Bool,
    /// Jamais lu (`nil` d'une expression dont la valeur est ignorée).
    Nil,
    /// Fonction globale appelée directement (case globale).
    Callee(u32),
}

fn prim(rt: &RtType) -> Option<RegTy> {
    match rt {
        RtType::Int => Some(RegTy::Int),
        RtType::Float => Some(RegTy::Float),
        RtType::Bool => Some(RegTy::Bool),
        _ => None,
    }
}

fn jty(t: RegTy) -> JTy {
    match t {
        RegTy::Int => JTy::Int,
        RegTy::Float => JTy::Float,
        RegTy::Bool => JTy::Bool,
        RegTy::Value => unreachable!(),
    }
}

fn clif(t: RegTy) -> Type {
    match t {
        RegTy::Int => types::I64,
        RegTy::Float => types::F64,
        RegTy::Bool => types::I8,
        RegTy::Value => unreachable!(),
    }
}

/// Signature native d'une fonction : paramètres et retour simples, sinon `None`.
fn native_sig(p: &Proto) -> Option<(Vec<RegTy>, RegTy)> {
    if p.kind != ProtoKind::Function || !p.captures.is_empty() || p.ncells != 0 {
        return None;
    }
    let mut params = Vec::new();
    for prm in &p.params {
        if prm.has_default || prm.field.is_some() {
            return None;
        }
        params.push(prim(&prm.ty.as_ref()?.ty)?);
    }
    Some((params, prim(&p.ret.as_ref()?.ty)?))
}

/// Fonction du script dans une case globale.
fn global_proto(globals: &Globals, slot: u32) -> Option<Rc<Proto>> {
    match &globals.at(slot).value {
        Some(Value::Function(f)) => match &**f {
            Function::Script(c) => Some(c.proto.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Résultat de l'analyse d'une fonction : type de chaque registre, cases appelées.
struct Plan {
    tys: Vec<Option<JTy>>,
    callees: Vec<u32>,
}

/// Types des registres d'une fonction compilable ; `None` si elle ne l'est pas.
fn analyze(p: &Proto, globals: &Globals) -> Option<Plan> {
    let (params, ret) = native_sig(p)?;
    let mut tys: Vec<Option<JTy>> = p
        .regs
        .iter()
        .map(|t| (*t != RegTy::Value).then(|| jty(*t)))
        .collect();
    for (prm, t) in p.params.iter().zip(&params) {
        tys[prm.reg as usize] = Some(jty(*t));
    }
    let mut callees = Vec::new();
    // Signature (paramètres, retour) d'une fonction appelée.
    let sig_of = |slot: u32| global_proto(globals, slot).and_then(|q| native_sig(&q));
    let numeric = |t: Option<JTy>| matches!(t, Some(JTy::Int | JTy::Float));
    // Point fixe : un registre peut être défini avant d'être lu dans l'ordre des blocs.
    for _ in 0..p.blocks.len() + 2 {
        let mut changed = false;
        let mut set = |tys: &mut Vec<Option<JTy>>, r: Reg, t: JTy| -> Option<()> {
            match tys[r as usize] {
                None => {
                    tys[r as usize] = Some(t);
                    changed = true;
                    Some(())
                }
                Some(u) if u == t => Some(()),
                // Registre `float` qui reçoit un `int` : converti.
                Some(JTy::Float) if t == JTy::Int && p.regs[r as usize] == RegTy::Float => Some(()),
                _ => None,
            }
        };
        for b in &p.blocks {
            for ins in &b.instrs {
                use Instr::*;
                match ins {
                    Int(d, _) => set(&mut tys, *d, JTy::Int)?,
                    Float(d, _) => set(&mut tys, *d, JTy::Float)?,
                    Bool(d, _) => set(&mut tys, *d, JTy::Bool)?,
                    Nil(d) => set(&mut tys, *d, JTy::Nil)?,
                    Move(d, a) | Guard(d, a, _) => {
                        if let Some(t) = tys[*a as usize] {
                            set(&mut tys, *d, t)?;
                        }
                    }
                    IBin(op, d, _, _) | IBinK(op, d, _, _) | FBin(op, d, _, _) => {
                        let t = if is_cmp(*op) {
                            JTy::Bool
                        } else if matches!(ins, FBin(..)) {
                            JTy::Float
                        } else {
                            JTy::Int
                        };
                        set(&mut tys, *d, t)?
                    }
                    INeg(d, _) => set(&mut tys, *d, JTy::Int)?,
                    FNeg(d, _) | IToF(d, _) => set(&mut tys, *d, JTy::Float)?,
                    Not(d, _) => set(&mut tys, *d, JTy::Bool)?,
                    Bin(op, d, a, b) => {
                        let (ta, tb) = (tys[*a as usize], tys[*b as usize]);
                        if ta.is_none() || tb.is_none() {
                            continue;
                        }
                        let t = match (*op, ta?, tb?) {
                            (op, JTy::Int, JTy::Int) if is_arith(op) => JTy::Int,
                            (op, _, _) if is_arith(op) && numeric(ta) && numeric(tb) => JTy::Float,
                            (op, _, _) if is_cmp(op) && numeric(ta) && numeric(tb) => JTy::Bool,
                            (BinOp::Eq | BinOp::Ne, JTy::Bool, JTy::Bool) => JTy::Bool,
                            _ => return None,
                        };
                        set(&mut tys, *d, t)?
                    }
                    GetGlobal(d, slot) => {
                        sig_of(*slot)?;
                        if !callees.contains(slot) {
                            callees.push(*slot);
                        }
                        set(&mut tys, *d, JTy::Callee(*slot))?
                    }
                    Call { dst, f, site } => {
                        let Some(JTy::Callee(slot)) = tys[*f as usize] else {
                            if tys[*f as usize].is_none() {
                                continue;
                            }
                            return None;
                        };
                        let site = &p.calls[*site as usize];
                        let (cparams, cret) = sig_of(slot)?;
                        if site.block
                            || !site.named.is_empty()
                            || site.npos as usize != cparams.len()
                        {
                            return None;
                        }
                        set(&mut tys, *dst, jty(cret))?
                    }
                    _ => return None,
                }
            }
        }
        if !changed {
            break;
        }
    }
    // Vérification des lectures : chaque opérande a le type attendu.
    let ty = |r: Reg| tys[r as usize];
    let is = |r: Reg, t: JTy| ty(r) == Some(t);
    let num = |r: Reg| matches!(ty(r), Some(JTy::Int | JTy::Float));
    for b in &p.blocks {
        for ins in &b.instrs {
            use Instr::*;
            let ok = match ins {
                Int(..) | Float(..) | Bool(..) | Nil(..) | GetGlobal(..) => true,
                Move(_, a) | Guard(_, a, _) => {
                    matches!(ty(*a), Some(JTy::Int | JTy::Float | JTy::Bool))
                }
                IBin(_, _, a, b2) => is(*a, JTy::Int) && is(*b2, JTy::Int),
                IBinK(_, _, a, _) | INeg(_, a) | IToF(_, a) => is(*a, JTy::Int),
                FBin(_, _, a, b2) => is(*a, JTy::Float) && is(*b2, JTy::Float),
                FNeg(_, a) => is(*a, JTy::Float),
                Not(_, a) => matches!(ty(*a), Some(JTy::Int | JTy::Float | JTy::Bool)),
                Bin(_, _, a, b2) => {
                    (num(*a) && num(*b2)) || (is(*a, JTy::Bool) && is(*b2, JTy::Bool))
                }
                Call { site, .. } => {
                    let site = &p.calls[*site as usize];
                    (0..site.npos).all(|k| num(site.start + k) || is(site.start + k, JTy::Bool))
                }
                _ => false,
            };
            if !ok {
                return None;
            }
        }
        let ok = match &b.term {
            Term::Jump(_) => true,
            Term::Branch { cond, .. } => matches!(
                ty(*cond),
                Some(JTy::Int | JTy::Float | JTy::Bool | JTy::Nil)
            ),
            Term::IfInt { a, b: y, .. } => is(*a, JTy::Int) && is(*y, JTy::Int),
            Term::IfIntK { a, .. } => is(*a, JTy::Int),
            Term::Return { reg, .. } => match ret {
                RegTy::Float => num(*reg),
                t => ty(*reg) == Some(jty(t)),
            },
            Term::IterNext { .. } => false,
        };
        if !ok {
            return None;
        }
    }
    // Arguments des appels : types compatibles avec les paramètres.
    for b in &p.blocks {
        for ins in &b.instrs {
            if let Instr::Call { f, site, .. } = ins {
                let Some(JTy::Callee(slot)) = ty(*f) else {
                    return None;
                };
                let (cparams, _) = sig_of(slot)?;
                let site = &p.calls[*site as usize];
                for (k, pt) in cparams.iter().enumerate() {
                    let at = ty(site.start + k as u32)?;
                    let compatible = match (pt, at) {
                        (RegTy::Float, JTy::Int | JTy::Float) => true,
                        (t, a) => jty(*t) == a,
                    };
                    if !compatible {
                        return None;
                    }
                }
            }
        }
    }
    Some(Plan { tys, callees })
}

fn is_cmp(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
    )
}

fn is_arith(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
    )
}

fn int_cc(op: BinOp) -> IntCC {
    match op {
        BinOp::Eq => IntCC::Equal,
        BinOp::Ne => IntCC::NotEqual,
        BinOp::Lt => IntCC::SignedLessThan,
        BinOp::Le => IntCC::SignedLessThanOrEqual,
        BinOp::Gt => IntCC::SignedGreaterThan,
        _ => IntCC::SignedGreaterThanOrEqual,
    }
}

fn float_cc(op: BinOp) -> FloatCC {
    match op {
        BinOp::Eq => FloatCC::Equal,
        BinOp::Ne => FloatCC::NotEqual,
        BinOp::Lt => FloatCC::LessThan,
        BinOp::Le => FloatCC::LessThanOrEqual,
        BinOp::Gt => FloatCC::GreaterThan,
        _ => FloatCC::GreaterThanOrEqual,
    }
}

impl Jit {
    /// `None` si la machine n'est pas prise en charge par Cranelift.
    pub fn new() -> Option<Self> {
        let mut flags = settings::builder();
        flags.set("use_colocated_libcalls", "false").ok()?;
        flags.set("is_pic", "false").ok()?;
        flags.set("opt_level", "speed").ok()?;
        let isa = cranelift_native::builder()
            .ok()?
            .finish(settings::Flags::new(flags))
            .ok()?;
        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
        builder.symbol("bk_trace", bk_trace as *const u8);
        builder.symbol("bk_frem", bk_frem as *const u8);
        builder.symbol("bk_step", bk_step as *const u8);
        builder.symbol("bk_iter_next", bk_iter_next as *const u8);
        builder.symbol("bk_drop", bk_drop as *const u8);
        let mut module = JITModule::new(builder);
        let ptr = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(types::I32));
        let trace_fn = module
            .declare_function("bk_trace", Linkage::Import, &sig)
            .ok()?;
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::F64));
        sig.params.push(AbiParam::new(types::F64));
        sig.returns.push(AbiParam::new(types::F64));
        let frem_fn = module
            .declare_function("bk_frem", Linkage::Import, &sig)
            .ok()?;
        Some(Jit {
            module,
            ctx: Box::new(JitCtx {
                depth: 0,
                max_depth: 0,
                err: -1,
                traces: Vec::new(),
            }),
            errors: Vec::new(),
            traces: Vec::new(),
            compiled: FxHashMap::default(),
            trace_fn,
            frem_fn,
            dyn_fns: None,
            next: 0,
        })
    }

    /// Compile `root` (et les fonctions qu'il appelle) si possible. Met à jour l'état JIT de
    /// chaque fonction rencontrée.
    pub fn compile(&mut self, root: &Rc<Proto>, globals: &mut Globals) {
        // 1. Fonctions atteignables par appels directs, et leur analyse.
        let mut order: Vec<Rc<Proto>> = Vec::new();
        let mut plans: FxHashMap<usize, Option<Plan>> = FxHashMap::default();
        let mut todo = vec![root.clone()];
        while let Some(p) = todo.pop() {
            let key = Rc::as_ptr(&p) as usize;
            if plans.contains_key(&key) {
                continue;
            }
            if self.compiled.contains_key(&key) {
                if matches!(*p.jit.borrow(), JitState::Native(_)) {
                    continue;
                }
                // Code invalidé (une fonction appelée a été redéfinie) : à recompiler.
                self.compiled.remove(&key);
            }
            let rejected = matches!(*p.jit.borrow(), JitState::Rejected);
            if rejected {
                plans.insert(key, None);
                order.push(p);
                continue;
            }
            let plan = analyze(&p, globals);
            if let Some(plan) = &plan {
                for slot in &plan.callees {
                    if let Some(q) = global_proto(globals, *slot) {
                        todo.push(q);
                    }
                }
            }
            plans.insert(key, plan);
            order.push(p);
        }
        // 2. Une fonction qui appelle une fonction non compilable ne l'est pas non plus.
        loop {
            let mut changed = false;
            for p in &order {
                let key = Rc::as_ptr(p) as usize;
                let Some(Some(plan)) = plans.get(&key) else {
                    continue;
                };
                let blocked = plan
                    .callees
                    .iter()
                    .any(|slot| match global_proto(globals, *slot) {
                        Some(q) => {
                            let k = Rc::as_ptr(&q) as usize;
                            !self.compiled.contains_key(&k)
                                && !matches!(plans.get(&k), Some(Some(_)))
                        }
                        None => true,
                    });
                if blocked {
                    plans.insert(key, None);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // 3. Déclaration, puis génération du code.
        let mut new = Vec::new();
        for p in &order {
            let key = Rc::as_ptr(p) as usize;
            if let Some(Some(_)) = plans.get(&key) {
                let (params, ret) = native_sig(p).unwrap();
                let sig = self.signature(&params, ret);
                self.next += 1;
                let id = match self.module.declare_function(
                    &format!("bk{}", self.next),
                    Linkage::Local,
                    &sig,
                ) {
                    Ok(id) => id,
                    Err(_) => return,
                };
                self.compiled.insert(key, Compiled { id, params, ret });
                new.push(p.clone());
            }
        }
        let mut entries = Vec::new();
        for p in &new {
            let key = Rc::as_ptr(p) as usize;
            let plan = plans.remove(&key).unwrap().unwrap();
            if self.define(p, &plan, globals).is_err() {
                // Échec inattendu de Cranelift : l'interpréteur reste disponible.
                self.compiled.remove(&key);
                *p.jit.borrow_mut() = JitState::Rejected;
                continue;
            }
            match self.define_entry(key) {
                Ok(e) => entries.push((p.clone(), e, plan.callees)),
                Err(_) => *p.jit.borrow_mut() = JitState::Rejected,
            }
        }
        if self.module.finalize_definitions().is_err() {
            return;
        }
        for (p, entry, callees) in entries {
            let c = &self.compiled[&(Rc::as_ptr(&p) as usize)];
            let code = self.module.get_finalized_function(entry);
            *p.jit.borrow_mut() = JitState::Native(NativeCode {
                entry: code as usize,
                params: c.params.clone(),
                ret: c.ret,
            });
            for slot in callees {
                globals.watch(slot, &p);
            }
        }
        // Les fonctions refusées : réessayées si une fonction qu'elles appellent change.
        for p in &order {
            let key = Rc::as_ptr(p) as usize;
            if let Some(None) = plans.get(&key) {
                *p.jit.borrow_mut() = JitState::Rejected;
                if let Some(plan) = analyze_callees(p) {
                    for slot in plan {
                        globals.watch(slot, p);
                    }
                }
            }
        }
    }

    fn signature(&self, params: &[RegTy], ret: RegTy) -> Signature {
        let mut sig = self.module.make_signature();
        sig.params
            .push(AbiParam::new(self.module.target_config().pointer_type()));
        for p in params {
            sig.params.push(AbiParam::new(clif(*p)));
        }
        sig.returns.push(AbiParam::new(types::I32));
        sig.returns.push(AbiParam::new(clif(ret)));
        sig
    }

    fn error_id(&mut self, kind: ErrKind, span: Span, script: &Rc<Script>) -> i64 {
        self.errors.push(ErrInfo {
            kind,
            span,
            script: script.clone(),
        });
        (self.errors.len() - 1) as i64
    }

    fn trace_id(&mut self, name: Rc<str>, span: Span) -> i64 {
        self.traces.push(TraceInfo { name, span });
        (self.traces.len() - 1) as i64
    }

    /// Génère le code d'une fonction.
    fn define(&mut self, p: &Rc<Proto>, plan: &Plan, globals: &Globals) -> Result<(), ()> {
        let key = Rc::as_ptr(p) as usize;
        let (id, ret) = {
            let c = &self.compiled[&key];
            (c.id, c.ret)
        };
        let mut ctx = self.module.make_context();
        ctx.func.signature = self.signature(&self.compiled[&key].params.clone(), ret);
        ctx.func.name = UserFuncName::user(0, id.as_u32());
        // Fonctions appelées, connues dans cette fonction.
        let mut callee_refs = FxHashMap::default();
        for slot in &plan.callees {
            let q = global_proto(globals, *slot).ok_or(())?;
            let cid = self.compiled.get(&(Rc::as_ptr(&q) as usize)).ok_or(())?.id;
            let r = self.module.declare_func_in_func(cid, &mut ctx.func);
            callee_refs.insert(*slot, (r, q));
        }
        let trace_ref = self
            .module
            .declare_func_in_func(self.trace_fn, &mut ctx.func);
        let frem_ref = self
            .module
            .declare_func_in_func(self.frem_fn, &mut ctx.func);
        // Erreurs et appels de cette fonction, numérotés à la suite des tables.
        let (err_base, trace_base) = (self.errors.len() as i64, self.traces.len() as i64);
        let mut err_sites: Vec<(ErrKind, Span)> = Vec::new();
        let mut call_sites: Vec<(Rc<str>, Span)> = Vec::new();

        let tc = self.module.target_config();
        let mut fctx = FunctionBuilderContext::new();
        let mut bx = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let vars: Vec<Option<Variable>> = plan
            .tys
            .iter()
            .map(|t| match t {
                Some(JTy::Int) => Some(bx.declare_var(types::I64)),
                Some(JTy::Float) => Some(bx.declare_var(types::F64)),
                Some(JTy::Bool) => Some(bx.declare_var(types::I8)),
                _ => None,
            })
            .collect();
        let entry = bx.create_block();
        let blocks: Vec<_> = p.blocks.iter().map(|_| bx.create_block()).collect();
        let fail = bx.create_block();
        bx.append_block_param(fail, types::I64);
        bx.append_block_params_for_function_params(entry);
        bx.switch_to_block(entry);
        let ctxv = bx.block_params(entry)[0];
        // Tous les registres sont définis dès l'entrée (valeur nulle).
        for (k, v) in vars.iter().enumerate() {
            if let Some(v) = v {
                let z = match plan.tys[k] {
                    Some(JTy::Float) => bx.ins().f64const(0.0),
                    Some(JTy::Bool) => bx.ins().iconst(types::I8, 0),
                    _ => bx.ins().iconst(types::I64, 0),
                };
                bx.def_var(*v, z);
            }
        }
        for (k, prm) in p.params.iter().enumerate() {
            let a = bx.block_params(entry)[k + 1];
            bx.def_var(vars[prm.reg as usize].unwrap(), a);
        }
        bx.ins().jump(blocks[0], &[]);

        let var = |r: Reg| vars[r as usize].expect("registre typé");
        let tyof = |r: Reg| plan.tys[r as usize];
        // Branche vers l'erreur `id` si `cond`.
        macro_rules! check {
            ($bx:expr, $cond:expr, $kind:expr, $span:expr) => {{
                err_sites.push(($kind, $span));
                let eid = err_base + (err_sites.len() - 1) as i64;
                let cont = $bx.create_block();
                let id = $bx.ins().iconst(types::I64, eid);
                $bx.ins().brif($cond, fail, &[id.into()], cont, &[]);
                $bx.switch_to_block(cont);
            }};
        }
        for (bi, b) in p.blocks.iter().enumerate() {
            bx.switch_to_block(blocks[bi]);
            for (ins, &span) in b.instrs.iter().zip(&b.spans) {
                use Instr::*;
                match ins {
                    Int(d, n) => {
                        let v = bx.ins().iconst(types::I64, *n);
                        bx.def_var(var(*d), v);
                    }
                    Float(d, f) => {
                        let v = bx.ins().f64const(*f);
                        bx.def_var(var(*d), v);
                    }
                    Bool(d, v) => {
                        let v = bx.ins().iconst(types::I8, *v as i64);
                        bx.def_var(var(*d), v);
                    }
                    Nil(_) | GetGlobal(..) => {}
                    Move(d, a) | Guard(d, a, _) => {
                        let v = bx.use_var(var(*a));
                        let v = match (tyof(*d), tyof(*a)) {
                            (Some(JTy::Float), Some(JTy::Int)) => {
                                bx.ins().fcvt_from_sint(types::F64, v)
                            }
                            _ => v,
                        };
                        bx.def_var(var(*d), v);
                    }
                    IBin(op, d, a, b2) => {
                        let (x, y) = (bx.use_var(var(*a)), bx.use_var(var(*b2)));
                        let v = int_bin(&mut bx, *op, x, y, &mut |bx, c, k| check!(bx, c, k, span));
                        bx.def_var(var(*d), v);
                    }
                    IBinK(op, d, a, k) => {
                        let x = bx.use_var(var(*a));
                        let y = bx.ins().iconst(types::I64, *k);
                        let v = int_bin(&mut bx, *op, x, y, &mut |bx, c, k| check!(bx, c, k, span));
                        bx.def_var(var(*d), v);
                    }
                    FBin(op, d, a, b2) => {
                        let (x, y) = (bx.use_var(var(*a)), bx.use_var(var(*b2)));
                        let v = float_bin(&mut bx, *op, x, y, frem_ref);
                        bx.def_var(var(*d), v);
                    }
                    Bin(op, d, a, b2) => {
                        let (ta, tb) = (tyof(*a), tyof(*b2));
                        let (x, y) = (bx.use_var(var(*a)), bx.use_var(var(*b2)));
                        let v = match (ta, tb) {
                            (Some(JTy::Int), Some(JTy::Int)) => {
                                int_bin(&mut bx, *op, x, y, &mut |bx, c, k| check!(bx, c, k, span))
                            }
                            (Some(JTy::Bool), Some(JTy::Bool)) => bx.ins().icmp(int_cc(*op), x, y),
                            _ => {
                                let x = if ta == Some(JTy::Int) {
                                    bx.ins().fcvt_from_sint(types::F64, x)
                                } else {
                                    x
                                };
                                let y = if tb == Some(JTy::Int) {
                                    bx.ins().fcvt_from_sint(types::F64, y)
                                } else {
                                    y
                                };
                                float_bin(&mut bx, *op, x, y, frem_ref)
                            }
                        };
                        bx.def_var(var(*d), v);
                    }
                    INeg(d, a) => {
                        let x = bx.use_var(var(*a));
                        let c = bx.ins().icmp_imm_s(IntCC::Equal, x, i64::MIN);
                        check!(bx, c, ErrKind::Overflow, span);
                        let v = bx.ins().ineg(x);
                        bx.def_var(var(*d), v);
                    }
                    FNeg(d, a) => {
                        let x = bx.use_var(var(*a));
                        let v = bx.ins().fneg(x);
                        bx.def_var(var(*d), v);
                    }
                    IToF(d, a) => {
                        let x = bx.use_var(var(*a));
                        let v = bx.ins().fcvt_from_sint(types::F64, x);
                        bx.def_var(var(*d), v);
                    }
                    Not(d, a) => {
                        let v = match tyof(*a) {
                            Some(JTy::Bool) => {
                                let x = bx.use_var(var(*a));
                                bx.ins().icmp_imm_s(IntCC::Equal, x, 0)
                            }
                            // Un nombre est toujours vrai.
                            _ => bx.ins().iconst(types::I8, 0),
                        };
                        bx.def_var(var(*d), v);
                    }
                    Call { dst, f, site } => {
                        let Some(JTy::Callee(slot)) = tyof(*f) else {
                            return Err(());
                        };
                        let (fref, q) = callee_refs[&slot].clone();
                        let (cparams, _) = native_sig(&q).ok_or(())?;
                        let site = &p.calls[*site as usize];
                        let mut args = vec![ctxv];
                        for (k, pt) in cparams.iter().enumerate() {
                            let r = site.start + k as u32;
                            let v = bx.use_var(var(r));
                            let v = match (pt, tyof(r)) {
                                (RegTy::Float, Some(JTy::Int)) => {
                                    bx.ins().fcvt_from_sint(types::F64, v)
                                }
                                _ => v,
                            };
                            args.push(v);
                        }
                        // Profondeur d'appels.
                        let depth =
                            bx.ins()
                                .load(types::I64, MemFlagsData::trusted(), ctxv, DEPTH_OFFSET);
                        let max = bx.ins().load(
                            types::I64,
                            MemFlagsData::trusted(),
                            ctxv,
                            MAX_DEPTH_OFFSET,
                        );
                        let too_deep = bx.ins().icmp(IntCC::SignedGreaterThanOrEqual, depth, max);
                        check!(bx, too_deep, ErrKind::Depth, span);
                        let d1 = bx.ins().iadd_imm_s(depth, 1);
                        bx.ins()
                            .store(MemFlagsData::trusted(), d1, ctxv, DEPTH_OFFSET);
                        let call = bx.ins().call(fref, &args);
                        let (status, result) = {
                            let r = bx.inst_results(call);
                            (r[0], r[1])
                        };
                        bx.ins()
                            .store(MemFlagsData::trusted(), depth, ctxv, DEPTH_OFFSET);
                        // Erreur dans la fonction appelée : on ajoute cet appel à la pile.
                        call_sites.push((q.name.clone(), span));
                        let tid = trace_base + (call_sites.len() - 1) as i64;
                        let failed = bx.create_block();
                        let cont = bx.create_block();
                        bx.ins().brif(status, failed, &[], cont, &[]);
                        bx.switch_to_block(failed);
                        let t = bx.ins().iconst(types::I32, tid);
                        bx.ins().call(trace_ref, &[ctxv, t]);
                        let none = bx.ins().iconst(types::I64, -1);
                        bx.ins().jump(fail, &[none.into()]);
                        bx.switch_to_block(cont);
                        bx.def_var(var(*dst), result);
                    }
                    _ => return Err(()),
                }
            }
            match &b.term {
                Term::Jump(t) => {
                    bx.ins().jump(blocks[*t as usize], &[]);
                }
                Term::Branch { cond, then, else_ } => {
                    let (t, e) = (blocks[*then as usize], blocks[*else_ as usize]);
                    match tyof(*cond) {
                        Some(JTy::Bool) => {
                            let c = bx.use_var(var(*cond));
                            bx.ins().brif(c, t, &[], e, &[]);
                        }
                        Some(JTy::Nil) => {
                            bx.ins().jump(e, &[]);
                        }
                        _ => {
                            bx.ins().jump(t, &[]);
                        }
                    }
                }
                Term::IfInt {
                    op,
                    a,
                    b: y,
                    then,
                    else_,
                } => {
                    let (x, y) = (bx.use_var(var(*a)), bx.use_var(var(*y)));
                    let c = bx.ins().icmp(int_cc(*op), x, y);
                    bx.ins()
                        .brif(c, blocks[*then as usize], &[], blocks[*else_ as usize], &[]);
                }
                Term::IfIntK {
                    op,
                    a,
                    k,
                    then,
                    else_,
                } => {
                    let x = bx.use_var(var(*a));
                    let c = bx.ins().icmp_imm_s(int_cc(*op), x, *k);
                    bx.ins()
                        .brif(c, blocks[*then as usize], &[], blocks[*else_ as usize], &[]);
                }
                Term::Return { reg, .. } => {
                    let v = bx.use_var(var(*reg));
                    let v = match (ret, tyof(*reg)) {
                        (RegTy::Float, Some(JTy::Int)) => bx.ins().fcvt_from_sint(types::F64, v),
                        _ => v,
                    };
                    let ok = bx.ins().iconst(types::I32, 0);
                    bx.ins().return_(&[ok, v]);
                }
                Term::IterNext { .. } => return Err(()),
            }
        }
        // Sortie en erreur : l'index de l'erreur est rangé dans `ctx` (-1 : déjà rangé
        // par la fonction appelée).
        bx.switch_to_block(fail);
        let eid = bx.block_params(fail)[0];
        let set = bx.create_block();
        let done = bx.create_block();
        let fresh = bx.ins().icmp_imm_s(IntCC::SignedGreaterThanOrEqual, eid, 0);
        bx.ins().brif(fresh, set, &[], done, &[]);
        bx.switch_to_block(set);
        bx.ins()
            .store(MemFlagsData::trusted(), eid, ctxv, ERR_OFFSET);
        bx.ins().jump(done, &[]);
        bx.switch_to_block(done);
        let one = bx.ins().iconst(types::I32, 1);
        let zero = match ret {
            RegTy::Float => bx.ins().f64const(0.0),
            RegTy::Bool => bx.ins().iconst(types::I8, 0),
            _ => bx.ins().iconst(types::I64, 0),
        };
        bx.ins().return_(&[one, zero]);
        bx.seal_all_blocks();
        bx.finalize(tc);
        let script = p.script.clone();
        for (kind, span) in err_sites {
            self.error_id(kind, span, &script);
        }
        for (name, span) in call_sites {
            self.trace_id(name, span);
        }
        self.module.define_function(id, &mut ctx).map_err(|_| ())?;
        self.module.clear_context(&mut ctx);
        Ok(())
    }

    /// Point d'entrée pour Rust : `fn(ctx, args, ret) -> statut`.
    fn define_entry(&mut self, key: usize) -> Result<FuncId, ()> {
        let (id, params, ret) = {
            let c = &self.compiled[&key];
            (c.id, c.params.clone(), c.ret)
        };
        let ptr = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.returns.push(AbiParam::new(types::I32));
        self.next += 1;
        let eid = self
            .module
            .declare_function(&format!("bk_entry{}", self.next), Linkage::Local, &sig)
            .map_err(|_| ())?;
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, eid.as_u32());
        let target = self.module.declare_func_in_func(id, &mut ctx.func);
        let tc = self.module.target_config();
        let mut fctx = FunctionBuilderContext::new();
        let mut bx = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let b = bx.create_block();
        bx.append_block_params_for_function_params(b);
        bx.switch_to_block(b);
        let (c, args, out) = {
            let p = bx.block_params(b);
            (p[0], p[1], p[2])
        };
        let mut call_args = vec![c];
        for (k, t) in params.iter().enumerate() {
            let v = bx
                .ins()
                .load(clif(*t), MemFlagsData::trusted(), args, (k * 8) as i32);
            call_args.push(v);
        }
        let call = bx.ins().call(target, &call_args);
        let (status, result) = {
            let r = bx.inst_results(call);
            (r[0], r[1])
        };
        let result = if ret == RegTy::Bool {
            bx.ins().uextend(types::I64, result)
        } else {
            result
        };
        bx.ins().store(MemFlagsData::trusted(), result, out, 0);
        bx.ins().return_(&[status]);
        bx.seal_all_blocks();
        bx.finalize(tc);
        self.module.define_function(eid, &mut ctx).map_err(|_| ())?;
        self.module.clear_context(&mut ctx);
        Ok(eid)
    }

    /// Exécute le code natif. `depth` : profondeur d'appels actuelle.
    ///
    /// `None` : la pile native ne suffit pas pour la profondeur demandée ; l'appel doit être
    /// refait par l'interpréteur (dont la pile grandit sur le tas). C'est sans risque : les
    /// fonctions compilées n'ont pas d'effet de bord.
    pub fn call(
        &mut self,
        code: &NativeCode,
        args: &[Value],
        depth: usize,
        max_depth: usize,
    ) -> Option<Result<Value, Error>> {
        let mut raw = [0u64; 16];
        let mut heap = Vec::new();
        let buf: &mut [u64] = if args.len() <= raw.len() {
            &mut raw[..args.len()]
        } else {
            heap.resize(args.len(), 0);
            &mut heap
        };
        for (slot, v) in buf.iter_mut().zip(args) {
            *slot = match v {
                Value::Int(i) => *i as u64,
                Value::Float(f) => f.to_bits(),
                Value::Bool(b) => *b as u64,
                _ => unreachable!("argument vérifié par la liaison des paramètres"),
            };
        }
        // Profondeur permise par la pile native restante (estimation prudente par appel).
        const FRAME: usize = 512;
        const MARGIN: usize = 64 * 1024;
        let stack_cap = stacker::remaining_stack().map_or(usize::MAX, |r| {
            depth.saturating_add(r.saturating_sub(MARGIN) / FRAME)
        });
        let limit = max_depth.min(stack_cap);
        self.ctx.depth = depth as i64;
        self.ctx.max_depth = limit as i64;
        self.ctx.err = -1;
        self.ctx.traces.clear();
        let mut out = 0u64;
        // SAFETY : `entry` est un point d'entrée généré par `define_entry` avec cette
        // signature ; les arguments ont le nombre et le type attendus.
        let f: extern "C" fn(*mut JitCtx, *const u64, *mut u64) -> i32 =
            unsafe { std::mem::transmute(code.entry) };
        let status = f(&mut *self.ctx, buf.as_ptr(), &mut out);
        if status == 0 {
            return Some(Ok(match code.ret {
                RegTy::Int => Value::Int(out as i64),
                RegTy::Float => Value::Float(f64::from_bits(out)),
                RegTy::Bool => Value::Bool(out & 1 == 1),
                RegTy::Value => unreachable!(),
            }));
        }
        let info = &self.errors[self.ctx.err as usize];
        if matches!(info.kind, ErrKind::Depth) && limit < max_depth {
            return None;
        }
        let msg = match info.kind {
            ErrKind::Overflow => "dépassement de capacité d'un entier".to_string(),
            ErrKind::DivZero => "division par zéro".to_string(),
            ErrKind::Depth => {
                format!("trop d'appels imbriqués ({max_depth}) : récursion infinie ?")
            }
        };
        let mut e = Error::runtime(msg, info.span);
        e.src = Some((info.script.name.clone(), info.script.src.clone()));
        for &t in self.ctx.traces.iter().take(16) {
            let t = &self.traces[t as usize];
            e.trace.push((t.name.to_string(), t.span));
        }
        Some(Err(e))
    }
}

/// Appels directs d'une fonction (cases globales), pour surveiller leur redéfinition.
fn analyze_callees(p: &Proto) -> Option<Vec<u32>> {
    let mut out = Vec::new();
    for b in &p.blocks {
        for ins in &b.instrs {
            if let Instr::GetGlobal(_, slot) = ins
                && !out.contains(slot)
            {
                out.push(*slot);
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Arithmétique entière avec les vérifications de l'interpréteur. `check(bx, cond, kind)`
/// branche vers l'erreur si `cond`.
fn int_bin(
    bx: &mut FunctionBuilder,
    op: BinOp,
    x: cranelift_codegen::ir::Value,
    y: cranelift_codegen::ir::Value,
    check: &mut dyn FnMut(&mut FunctionBuilder, cranelift_codegen::ir::Value, ErrKind),
) -> cranelift_codegen::ir::Value {
    match op {
        BinOp::Add => {
            let (v, of) = bx.ins().sadd_overflow(x, y);
            check(bx, of, ErrKind::Overflow);
            v
        }
        BinOp::Sub => {
            let (v, of) = bx.ins().ssub_overflow(x, y);
            check(bx, of, ErrKind::Overflow);
            v
        }
        BinOp::Mul => {
            let (v, of) = bx.ins().smul_overflow(x, y);
            check(bx, of, ErrKind::Overflow);
            v
        }
        BinOp::Div | BinOp::Rem => {
            let zero = bx.ins().icmp_imm_s(IntCC::Equal, y, 0);
            check(bx, zero, ErrKind::DivZero);
            // i64::MIN / -1 déborde (et ferait planter `sdiv`).
            let min = bx.ins().icmp_imm_s(IntCC::Equal, x, i64::MIN);
            let m1 = bx.ins().icmp_imm_s(IntCC::Equal, y, -1);
            let both = bx.ins().band(min, m1);
            check(bx, both, ErrKind::Overflow);
            let q = bx.ins().sdiv(x, y);
            let r = bx.ins().srem(x, y);
            let neg = bx.ins().icmp_imm_s(IntCC::SignedLessThan, r, 0);
            if op == BinOp::Div {
                // Division euclidienne : reste toujours positif.
                let ypos = bx.ins().icmp_imm_s(IntCC::SignedGreaterThan, y, 0);
                let minus = bx.ins().iadd_imm_s(q, -1);
                let plus = bx.ins().iadd_imm_s(q, 1);
                let adj = bx.ins().select(ypos, minus, plus);
                bx.ins().select(neg, adj, q)
            } else {
                let ay = bx.ins().iabs(y);
                let r2 = bx.ins().iadd(r, ay);
                bx.ins().select(neg, r2, r)
            }
        }
        op => bx.ins().icmp(int_cc(op), x, y),
    }
}

fn float_bin(
    bx: &mut FunctionBuilder,
    op: BinOp,
    x: cranelift_codegen::ir::Value,
    y: cranelift_codegen::ir::Value,
    frem: cranelift_codegen::ir::FuncRef,
) -> cranelift_codegen::ir::Value {
    match op {
        BinOp::Add => bx.ins().fadd(x, y),
        BinOp::Sub => bx.ins().fsub(x, y),
        BinOp::Mul => bx.ins().fmul(x, y),
        BinOp::Div => bx.ins().fdiv(x, y),
        BinOp::Rem => {
            let c = bx.ins().call(frem, &[x, y]);
            bx.inst_results(c)[0]
        }
        op => bx.ins().fcmp(float_cc(op), x, y),
    }
}

// ─── Compilation de base (code non typé) ────────────────────────────────────
//
// Toute fonction peut être compilée ainsi. Les registres restent des `Value` en mémoire
// (la fenêtre de l'appel) ; le code natif traite lui-même les cas simples (constantes,
// copies de valeurs scalaires, arithmétique et comparaisons d'entiers et de décimaux,
// branchements) et appelle `Runtime::step` — le code de l'interpréteur — pour tout le
// reste, y compris les cas inhabituels (dépassement…) : la sémantique est la même.

/// Exécute une instruction de l'IR avec le code de l'interpréteur. 0 : succès, 1 : erreur
/// (rangée dans le `Runtime`).
extern "C" fn bk_step(
    rt: *mut Runtime,
    f: *mut NativeFrame,
    ins: *const Instr,
    lo: u32,
    hi: u32,
) -> i32 {
    // SAFETY : `rt`, `f` et `ins` viennent de `Runtime::run_native`, valides pendant
    // l'appel ; les pointeurs de la fenêtre désignent des tranches de la longueur donnée.
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

/// Élément suivant d'un itérateur `for` : 1 s'il y en a un (rangé dans `dst`), 0 sinon.
extern "C" fn bk_iter_next(f: *mut NativeFrame, iter: u32, dst: u32) -> i32 {
    // SAFETY : comme pour `bk_step`.
    unsafe {
        let f = &mut *f;
        let regs = std::slice::from_raw_parts_mut(f.regs, f.nregs);
        crate::vm::iter_next(regs, iter, dst) as i32
    }
}

/// Libère la valeur d'un registre avant d'y écrire une valeur scalaire.
extern "C" fn bk_drop(slot: *mut Value) {
    // SAFETY : `slot` est un registre de la fenêtre en cours.
    unsafe {
        std::ptr::drop_in_place(slot);
        slot.write(Value::Nil);
    }
}

struct Dyn<'a> {
    regs_ty: &'a [RegTy],
    regs: CValue,
    rt: CValue,
    frame: CValue,
    step: FuncRef,
    drop: FuncRef,
    fail: CBlock,
}

impl Dyn<'_> {
    fn off(r: Reg) -> i32 {
        r as i32 * SIZE
    }

    fn tag(&self, bx: &mut FunctionBuilder, r: Reg) -> CValue {
        bx.ins()
            .load(types::I8, MemFlagsData::trusted(), self.regs, Self::off(r))
    }

    fn payload(&self, bx: &mut FunctionBuilder, r: Reg, ty: Type) -> CValue {
        bx.ins().load(
            ty,
            MemFlagsData::trusted(),
            self.regs,
            Self::off(r) + PAYLOAD,
        )
    }

    /// Écrit une valeur scalaire (étiquette, valeur) dans un registre. Un registre
    /// `valeur` peut contenir une ressource : elle est d'abord libérée.
    fn write(&self, bx: &mut FunctionBuilder, d: Reg, tag: CValue, payload: Option<CValue>) {
        if self.regs_ty[d as usize] == RegTy::Value {
            let old = self.tag(bx, d);
            let heap = bx.ins().icmp_imm_u(
                IntCC::UnsignedGreaterThanOrEqual,
                old,
                FIRST_HEAP_TAG as i64,
            );
            let free = bx.create_block();
            let cont = bx.create_block();
            bx.ins().brif(heap, free, &[], cont, &[]);
            bx.switch_to_block(free);
            let addr = bx.ins().iadd_imm_s(self.regs, Self::off(d) as i64);
            bx.ins().call(self.drop, &[addr]);
            bx.ins().jump(cont, &[]);
            bx.switch_to_block(cont);
        }
        bx.ins()
            .store(MemFlagsData::trusted(), tag, self.regs, Self::off(d));
        if let Some(p) = payload {
            bx.ins().store(
                MemFlagsData::trusted(),
                p,
                self.regs,
                Self::off(d) + PAYLOAD,
            );
        }
    }

    fn write_const(&self, bx: &mut FunctionBuilder, d: Reg, tag: u8, payload: Option<CValue>) {
        let t = bx.ins().iconst(types::I8, tag as i64);
        self.write(bx, d, t, payload);
    }

    /// Appel de `bk_step` pour l'instruction `ins` ; continue dans un nouveau bloc.
    fn step(&self, bx: &mut FunctionBuilder, ins: &Instr, span: Span) {
        let ptr = bx.func.dfg.value_type(self.regs);
        let ip = bx.ins().iconst(ptr, ins as *const Instr as i64);
        let lo = bx.ins().iconst(types::I32, span.start as i64);
        let hi = bx.ins().iconst(types::I32, span.end as i64);
        let call = bx.ins().call(self.step, &[self.rt, self.frame, ip, lo, hi]);
        let status = bx.inst_results(call)[0];
        let cont = bx.create_block();
        bx.ins().brif(status, self.fail, &[], cont, &[]);
        bx.switch_to_block(cont);
    }

    /// Vérité d'un registre (`nil` et `false` sont faux), en `i8` (1 : vrai).
    fn truthy(&self, bx: &mut FunctionBuilder, r: Reg) -> CValue {
        match self.regs_ty[r as usize] {
            RegTy::Bool => {
                let b = self.payload(bx, r, types::I8);
                bx.ins().icmp_imm_s(IntCC::NotEqual, b, 0)
            }
            RegTy::Int | RegTy::Float => bx.ins().iconst(types::I8, 1),
            RegTy::Value => {
                let t = self.tag(bx, r);
                let b = self.payload(bx, r, types::I8);
                let nil = bx.ins().icmp_imm_s(IntCC::Equal, t, TAG_NIL as i64);
                let is_bool = bx.ins().icmp_imm_s(IntCC::Equal, t, TAG_BOOL as i64);
                let zero = bx.ins().icmp_imm_s(IntCC::Equal, b, 0);
                let fals = bx.ins().band(is_bool, zero);
                let falsy = bx.ins().bor(nil, fals);
                bx.ins().icmp_imm_s(IntCC::Equal, falsy, 0)
            }
        }
    }
}

impl Jit {
    /// Compile `p` en code natif de base ; adresse du point d'entrée.
    pub fn compile_dynamic(&mut self, p: &Proto) -> Option<usize> {
        let ptr = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.returns.push(AbiParam::new(types::I32));
        self.next += 1;
        let id = self
            .module
            .declare_function(&format!("bkd{}", self.next), Linkage::Local, &sig)
            .ok()?;
        let helpers = self.dyn_helpers()?;
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, id.as_u32());
        let step = self.module.declare_func_in_func(helpers.0, &mut ctx.func);
        let iter = self.module.declare_func_in_func(helpers.1, &mut ctx.func);
        let drop = self.module.declare_func_in_func(helpers.2, &mut ctx.func);
        let tc = self.module.target_config();
        let mut fctx = FunctionBuilderContext::new();
        let mut bx = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let entry = bx.create_block();
        bx.append_block_params_for_function_params(entry);
        let blocks: Vec<CBlock> = p.blocks.iter().map(|_| bx.create_block()).collect();
        let fail = bx.create_block();
        bx.switch_to_block(entry);
        let (rt, frame) = {
            let ps = bx.block_params(entry);
            (ps[0], ps[1])
        };
        let regs = bx.ins().load(
            ptr,
            MemFlagsData::trusted(),
            frame,
            std::mem::offset_of!(NativeFrame, regs) as i32,
        );
        bx.ins().jump(blocks[0], &[]);
        let d = Dyn {
            regs_ty: &p.regs,
            regs,
            rt,
            frame,
            step,
            drop,
            fail,
        };
        for (bi, b) in p.blocks.iter().enumerate() {
            bx.switch_to_block(blocks[bi]);
            for (ins, &span) in b.instrs.iter().zip(&b.spans) {
                dyn_instr(&d, &mut bx, ins, span);
            }
            match &b.term {
                Term::Jump(t) => {
                    bx.ins().jump(blocks[*t as usize], &[]);
                }
                Term::Branch { cond, then, else_ } => {
                    let c = d.truthy(&mut bx, *cond);
                    bx.ins()
                        .brif(c, blocks[*then as usize], &[], blocks[*else_ as usize], &[]);
                }
                Term::IfInt {
                    op,
                    a,
                    b: y,
                    then,
                    else_,
                } => {
                    let x = d.payload(&mut bx, *a, types::I64);
                    let y = d.payload(&mut bx, *y, types::I64);
                    let c = bx.ins().icmp(int_cc(*op), x, y);
                    bx.ins()
                        .brif(c, blocks[*then as usize], &[], blocks[*else_ as usize], &[]);
                }
                Term::IfIntK {
                    op,
                    a,
                    k,
                    then,
                    else_,
                } => {
                    let x = d.payload(&mut bx, *a, types::I64);
                    let c = bx.ins().icmp_imm_s(int_cc(*op), x, *k);
                    bx.ins()
                        .brif(c, blocks[*then as usize], &[], blocks[*else_ as usize], &[]);
                }
                Term::Return { reg, shape } => {
                    let r = bx.ins().iconst(types::I32, *reg as i64);
                    let s = bx
                        .ins()
                        .iconst(types::I32, shape.unwrap_or(u32::MAX) as i64);
                    bx.ins().store(
                        MemFlagsData::trusted(),
                        r,
                        frame,
                        std::mem::offset_of!(NativeFrame, ret) as i32,
                    );
                    bx.ins().store(
                        MemFlagsData::trusted(),
                        s,
                        frame,
                        std::mem::offset_of!(NativeFrame, ret_shape) as i32,
                    );
                    let ok = bx.ins().iconst(types::I32, 0);
                    bx.ins().return_(&[ok]);
                }
                Term::IterNext {
                    iter: it,
                    dst,
                    body,
                    done,
                } => {
                    let a = bx.ins().iconst(types::I32, *it as i64);
                    let b2 = bx.ins().iconst(types::I32, *dst as i64);
                    let call = bx.ins().call(iter, &[frame, a, b2]);
                    let got = bx.inst_results(call)[0];
                    bx.ins().brif(
                        got,
                        blocks[*body as usize],
                        &[],
                        blocks[*done as usize],
                        &[],
                    );
                }
            }
        }
        bx.switch_to_block(fail);
        let one = bx.ins().iconst(types::I32, 1);
        bx.ins().return_(&[one]);
        bx.seal_all_blocks();
        bx.finalize(tc);
        self.module.define_function(id, &mut ctx).ok()?;
        self.module.clear_context(&mut ctx);
        self.module.finalize_definitions().ok()?;
        Some(self.module.get_finalized_function(id) as usize)
    }

    /// `bk_step`, `bk_iter_next`, `bk_drop` (déclarés au premier besoin).
    fn dyn_helpers(&mut self) -> Option<(FuncId, FuncId, FuncId)> {
        if let Some(h) = self.dyn_fns {
            return Some(h);
        }
        let ptr = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        for t in [ptr, ptr, ptr, types::I32, types::I32] {
            sig.params.push(AbiParam::new(t));
        }
        sig.returns.push(AbiParam::new(types::I32));
        let step = self
            .module
            .declare_function("bk_step", Linkage::Import, &sig)
            .ok()?;
        let mut sig = self.module.make_signature();
        for t in [ptr, types::I32, types::I32] {
            sig.params.push(AbiParam::new(t));
        }
        sig.returns.push(AbiParam::new(types::I32));
        let iter = self
            .module
            .declare_function("bk_iter_next", Linkage::Import, &sig)
            .ok()?;
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        let drop = self
            .module
            .declare_function("bk_drop", Linkage::Import, &sig)
            .ok()?;
        self.dyn_fns = Some((step, iter, drop));
        self.dyn_fns
    }
}

/// Traduit une instruction : en ligne pour les cas simples, sinon `bk_step`.
fn dyn_instr(d: &Dyn, bx: &mut FunctionBuilder, ins: &Instr, span: Span) {
    use Instr::*;
    let ty = |r: Reg| d.regs_ty[r as usize];
    match ins {
        Int(r, n) => {
            let v = bx.ins().iconst(types::I64, *n);
            d.write_const(bx, *r, TAG_INT, Some(v));
        }
        Float(r, f) => {
            let v = bx.ins().f64const(*f);
            d.write_const(bx, *r, TAG_FLOAT, Some(v));
        }
        Bool(r, b) => {
            let v = bx.ins().iconst(types::I8, *b as i64);
            d.write_const(bx, *r, TAG_BOOL, Some(v));
        }
        Nil(r) => d.write_const(bx, *r, TAG_NIL, None),
        Move(dst, a) if ty(*a) != RegTy::Value => {
            let (tag, t) = scalar(ty(*a));
            let v = d.payload(bx, *a, t);
            d.write_const(bx, *dst, tag, Some(v));
        }
        // Copie d'une valeur : directe si elle est scalaire, sinon par l'interpréteur.
        Move(dst, a) if ty(*dst) == RegTy::Value => {
            let t = d.tag(bx, *a);
            let scalar = bx
                .ins()
                .icmp_imm_u(IntCC::UnsignedLessThan, t, FIRST_HEAP_TAG as i64);
            let fast = bx.create_block();
            let slow = bx.create_block();
            let next = bx.create_block();
            bx.ins().brif(scalar, fast, &[], slow, &[]);
            bx.switch_to_block(fast);
            let v = d.payload(bx, *a, types::I64);
            d.write(bx, *dst, t, Some(v));
            bx.ins().jump(next, &[]);
            bx.switch_to_block(slow);
            d.step(bx, ins, span);
            bx.ins().jump(next, &[]);
            bx.switch_to_block(next);
        }
        IBin(op, r, a, b) => {
            let x = d.payload(bx, *a, types::I64);
            let y = d.payload(bx, *b, types::I64);
            int_fast(d, bx, ins, span, *op, *r, x, y);
        }
        IBinK(op, r, a, k) => {
            let x = d.payload(bx, *a, types::I64);
            let y = bx.ins().iconst(types::I64, *k);
            int_fast(d, bx, ins, span, *op, *r, x, y);
        }
        FBin(op, r, a, b) if *op != BinOp::Rem => {
            let x = d.payload(bx, *a, types::F64);
            let y = d.payload(bx, *b, types::F64);
            float_fast(d, bx, *op, *r, x, y);
        }
        IToF(r, a) => {
            let x = d.payload(bx, *a, types::I64);
            let v = bx.ins().fcvt_from_sint(types::F64, x);
            d.write_const(bx, *r, TAG_FLOAT, Some(v));
        }
        FNeg(r, a) => {
            let x = d.payload(bx, *a, types::F64);
            let v = bx.ins().fneg(x);
            d.write_const(bx, *r, TAG_FLOAT, Some(v));
        }
        Not(r, a) => {
            let t = d.truthy(bx, *a);
            let v = bx.ins().icmp_imm_s(IntCC::Equal, t, 0);
            d.write_const(bx, *r, TAG_BOOL, Some(v));
        }
        // Opération générique : en ligne pour deux entiers ou deux décimaux.
        Bin(op, r, a, b) if is_arith(*op) || is_cmp(*op) => {
            let ta = d.tag(bx, *a);
            let tb = d.tag(bx, *b);
            let ia = bx.ins().icmp_imm_s(IntCC::Equal, ta, TAG_INT as i64);
            let ib = bx.ins().icmp_imm_s(IntCC::Equal, tb, TAG_INT as i64);
            let both_int = bx.ins().band(ia, ib);
            let ints = bx.create_block();
            let other = bx.create_block();
            let floats = bx.create_block();
            let slow = bx.create_block();
            let next = bx.create_block();
            bx.ins().brif(both_int, ints, &[], other, &[]);
            bx.switch_to_block(ints);
            let x = d.payload(bx, *a, types::I64);
            let y = d.payload(bx, *b, types::I64);
            int_op_into(d, bx, *op, *r, x, y, slow);
            bx.ins().jump(next, &[]);
            bx.switch_to_block(other);
            let fa = bx.ins().icmp_imm_s(IntCC::Equal, ta, TAG_FLOAT as i64);
            let fb = bx.ins().icmp_imm_s(IntCC::Equal, tb, TAG_FLOAT as i64);
            let both_float = bx.ins().band(fa, fb);
            let not_rem = *op != BinOp::Rem;
            if not_rem {
                bx.ins().brif(both_float, floats, &[], slow, &[]);
            } else {
                bx.ins().jump(slow, &[]);
            }
            bx.switch_to_block(floats);
            if not_rem {
                let x = d.payload(bx, *a, types::F64);
                let y = d.payload(bx, *b, types::F64);
                float_fast(d, bx, *op, *r, x, y);
                bx.ins().jump(next, &[]);
            } else {
                bx.ins().jump(slow, &[]);
            }
            bx.switch_to_block(slow);
            d.step(bx, ins, span);
            bx.ins().jump(next, &[]);
            bx.switch_to_block(next);
        }
        _ => d.step(bx, ins, span),
    }
}

fn scalar(t: RegTy) -> (u8, Type) {
    match t {
        RegTy::Int => (TAG_INT, types::I64),
        RegTy::Float => (TAG_FLOAT, types::F64),
        RegTy::Bool => (TAG_BOOL, types::I8),
        RegTy::Value => unreachable!(),
    }
}

/// Opération entière en ligne ; cas limite (dépassement, division par zéro) : `bk_step`.
#[allow(clippy::too_many_arguments)]
fn int_fast(
    d: &Dyn,
    bx: &mut FunctionBuilder,
    ins: &Instr,
    span: Span,
    op: BinOp,
    r: Reg,
    x: CValue,
    y: CValue,
) {
    let slow = bx.create_block();
    let next = bx.create_block();
    int_op_into(d, bx, op, r, x, y, slow);
    bx.ins().jump(next, &[]);
    bx.switch_to_block(slow);
    d.step(bx, ins, span);
    bx.ins().jump(next, &[]);
    bx.switch_to_block(next);
}

/// Calcule `x op y` et l'écrit dans `r` ; branche vers `slow` pour tout cas limite.
fn int_op_into(
    d: &Dyn,
    bx: &mut FunctionBuilder,
    op: BinOp,
    r: Reg,
    x: CValue,
    y: CValue,
    slow: CBlock,
) {
    let v = int_bin(bx, op, x, y, &mut |bx, cond, _| {
        let cont = bx.create_block();
        bx.ins().brif(cond, slow, &[], cont, &[]);
        bx.switch_to_block(cont);
    });
    let tag = if is_cmp(op) { TAG_BOOL } else { TAG_INT };
    d.write_const(bx, r, tag, Some(v));
}

fn float_fast(d: &Dyn, bx: &mut FunctionBuilder, op: BinOp, r: Reg, x: CValue, y: CValue) {
    let v = match op {
        BinOp::Add => bx.ins().fadd(x, y),
        BinOp::Sub => bx.ins().fsub(x, y),
        BinOp::Mul => bx.ins().fmul(x, y),
        BinOp::Div => bx.ins().fdiv(x, y),
        op => bx.ins().fcmp(float_cc(op), x, y),
    };
    let tag = if is_cmp(op) { TAG_BOOL } else { TAG_FLOAT };
    d.write_const(bx, r, tag, Some(v));
}
