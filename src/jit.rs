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
    /// Hypothèse du code natif non vérifiée (champ d'un autre type) : l'appel est refait
    /// par l'interpréteur.
    Bail,
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

/// Champs d'un objet du type `info` (passé en paramètre) ; nul si l'objet est d'un autre
/// type (type redéclaré depuis la compilation).
extern "C" fn bk_obj_fields(
    obj: *const Value,
    info: *const crate::objects::TypeInfo,
) -> *const Value {
    // SAFETY : `obj` désigne un argument vivant pendant l'appel natif. Le code natif est
    // sans effet de bord : les champs ne changent pas (ni ne sont déplacés) pendant l'appel.
    unsafe {
        match &*obj {
            Value::Object(o) => match o.try_borrow() {
                Ok(b) if Rc::as_ptr(&b.info) == info => b.fields.as_ptr(),
                _ => std::ptr::null(),
            },
            _ => std::ptr::null(),
        }
    }
}

/// Éléments d'une liste : adresse dans `out[0]`, nombre dans `out[1]` ; 1 si ce n'est
/// pas une liste.
extern "C" fn bk_list_view(v: *const Value, out: *mut u64) -> i32 {
    // SAFETY : `v` désigne une valeur vivante pendant l'appel natif, `out` deux cases.
    // Le code natif est sans effet de bord : la liste ne change pas pendant l'appel.
    unsafe {
        let Value::List(l) = &*v else {
            return 1;
        };
        let Ok(items) = l.try_borrow() else {
            return 1;
        };
        *out = items.as_ptr() as u64;
        *out.add(1) = items.len() as u64;
    }
    0
}

/// Champ d'un objet passé en paramètre (`FieldAs`) : 0 et la valeur (bits) dans `out`,
/// ou 1 si l'objet n'a pas ce champ avec ce type.
extern "C" fn bk_field_native(obj: *const Value, cache: *mut u8, out: *mut u64) -> i32 {
    // SAFETY : `obj` désigne un argument vivant pendant l'appel natif (cf. `Jit::call`) ;
    // `cache` est un `FieldCache` gardé par le `Jit`.
    unsafe {
        let cache = &mut *(cache as *mut FieldCache);
        let bits = match cache.get(&*obj) {
            Some(Value::Int(i)) => i as u64,
            Some(Value::Float(f)) => f.to_bits(),
            Some(Value::Bool(b)) => b as u64,
            _ => return 1,
        };
        *out = bits;
    }
    0
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
    field_fn: FuncId,
    fields_fn: FuncId,
    list_fn: FuncId,
    /// Types dont le code natif connaît la disposition (gardés en vie : leur adresse sert
    /// d'identité).
    known_types: Vec<Rc<crate::objects::TypeInfo>>,
    /// Le dernier appel natif a été abandonné (`ErrKind::Bail`).
    pub bailed: bool,
    /// Symbole de la méthode `len` (listes).
    pub len_sym: crate::intern::Symbol,
    /// Fonctions d'aide du code de base (`bk_step`, `bk_iter_next`, `bk_drop`).
    dyn_fns: Option<(FuncId, FuncId, FuncId)>,
    /// Fonctions d'aide rapides (`FAST`), dans l'ordre de `Fast`.
    fast_fns: Vec<FuncId>,
    /// Caches des accès aux champs du code de base : en `Box`, car le code natif garde
    /// leur adresse.
    #[allow(clippy::vec_box)]
    field_caches: Vec<Box<FieldCache>>,
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
    /// Objet du type `anns[i]` : pointeur vers la valeur (paramètre ou élément d'une
    /// liste), vivante pendant tout l'appel.
    Obj(u32),
    /// Liste d'éléments du type `anns[i]` : pointeur vers la valeur.
    List(u32),
    /// Itérateur sur une liste d'éléments du type `anns[i]`.
    Iter(u32),
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

/// Type Cranelift d'un registre ; `Value` : pointeur vers un objet.
fn clif(t: RegTy) -> Type {
    match t {
        RegTy::Int | RegTy::Value => types::I64,
        RegTy::Float => types::F64,
        RegTy::Bool => types::I8,
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
        let ty = &prm.ty.as_ref()?.ty;
        // Paramètre d'un type déclaré : l'objet, par pointeur (`Value`).
        params.push(match &**ty {
            RtType::Named(..) => RegTy::Value,
            RtType::List(e) if matches!(**e, RtType::Named(..)) || prim(e).is_some() => {
                RegTy::Value
            }
            t => prim(t)?,
        });
    }
    Some((params, prim(&p.ret.as_ref()?.ty)?))
}

/// Annotation du paramètre `i`.
fn ann_of(q: &Proto, i: usize) -> Option<&RtType> {
    q.params[i].ty.as_ref().map(|t| &*t.ty)
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
    /// Types des objets et des éléments de liste (`JTy::Obj` / `List` / `Iter`).
    anns: Vec<RtType>,
    /// Annotation propre à chaque paramètre objet.
    param_ann: Vec<Option<u32>>,
}

/// Type d'un élément de type `anns[a]`.
fn elem_jty(anns: &[RtType], a: u32) -> JTy {
    match &anns[a as usize] {
        RtType::Named(..) => JTy::Obj(a),
        t => jty(prim(t).expect("élément simple ou objet")),
    }
}

/// Types des registres d'une fonction compilable ; `None` si elle ne l'est pas.
fn analyze(p: &Proto, globals: &Globals, len: crate::intern::Symbol) -> Option<Plan> {
    let (params, ret) = native_sig(p)?;
    let mut tys: Vec<Option<JTy>> = p
        .regs
        .iter()
        .map(|t| (*t != RegTy::Value).then(|| jty(*t)))
        .collect();
    let mut anns: Vec<RtType> = Vec::new();
    let mut param_ann = vec![None; params.len()];
    for (k, (prm, t)) in p.params.iter().zip(&params).enumerate() {
        tys[prm.reg as usize] = Some(match t {
            RegTy::Value => {
                let a = anns.len() as u32;
                match &*prm.ty.as_ref()?.ty {
                    t @ RtType::Named(..) => {
                        anns.push(t.clone());
                        param_ann[k] = Some(a);
                        JTy::Obj(a)
                    }
                    RtType::List(e) => {
                        anns.push((**e).clone());
                        JTy::List(a)
                    }
                    _ => return None,
                }
            }
            t => jty(*t),
        });
    }
    let dead = dead_nils(p);
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
                    // `nil` jamais lu (valeur d'un `while`…) : le registre peut servir ailleurs
                    // avec un autre type.
                    Nil(_) if dead.contains(&(ins as *const Instr)) => {}
                    Nil(d) => set(&mut tys, *d, JTy::Nil)?,
                    Move(d, a) | Guard(d, a, _) => {
                        if let Some(t) = tys[*a as usize] {
                            set(&mut tys, *d, t)?;
                        }
                    }
                    FieldAs(d, _, _, t) => set(&mut tys, *d, jty(*t))?,
                    IterNew { dst, src, .. } => match tys[*src as usize] {
                        Some(JTy::List(a)) => set(&mut tys, *dst, JTy::Iter(a))?,
                        None => {}
                        _ => return None,
                    },
                    Index(d, a, _) => match tys[*a as usize] {
                        Some(JTy::List(e)) => set(&mut tys, *d, elem_jty(&anns, e))?,
                        None => {}
                        _ => return None,
                    },
                    CallMethod {
                        dst, recv, name, ..
                    } if *name == len => match tys[*recv as usize] {
                        Some(JTy::List(_)) => set(&mut tys, *dst, JTy::Int)?,
                        None => {}
                        _ => return None,
                    },
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
            if let Term::IterNext { iter, dst, .. } = &b.term {
                match tys[*iter as usize] {
                    Some(JTy::Iter(a)) => set(&mut tys, *dst, elem_jty(&anns, a))?,
                    None => {}
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
                Move(_, a) => matches!(
                    ty(*a),
                    Some(JTy::Int | JTy::Float | JTy::Bool | JTy::Obj(_) | JTy::List(_))
                ),
                IterNew { src, .. } => matches!(ty(*src), Some(JTy::List(_))),
                Index(_, a, i) => matches!(ty(*a), Some(JTy::List(_))) && is(*i, JTy::Int),
                CallMethod {
                    recv, name, site, ..
                } => {
                    let site = &p.calls[*site as usize];
                    *name == len
                        && matches!(ty(*recv), Some(JTy::List(_)))
                        && site.npos == 0
                        && site.named.is_empty()
                        && !site.block
                }
                Guard(_, a, _) => matches!(ty(*a), Some(JTy::Int | JTy::Float | JTy::Bool)),
                FieldAs(_, a, _, _) => matches!(ty(*a), Some(JTy::Obj(_))),
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
                    (0..site.npos).all(|k| {
                        let r = site.start + k;
                        num(r)
                            || is(r, JTy::Bool)
                            || matches!(ty(r), Some(JTy::Obj(_) | JTy::List(_)))
                    })
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
            Term::IterNext { iter, .. } => matches!(ty(*iter), Some(JTy::Iter(_))),
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
                        // Objet transmis tel quel : même annotation, donc déjà vérifié.
                        (RegTy::Value, JTy::Obj(a)) => {
                            let callee = global_proto(globals, slot)?;
                            ann_of(&callee, k) == Some(&anns[a as usize])
                        }
                        // Liste transmise telle quelle : mêmes éléments.
                        (RegTy::Value, JTy::List(a)) => {
                            let callee = global_proto(globals, slot)?;
                            matches!(ann_of(&callee, k), Some(RtType::List(e)) if **e == anns[a as usize])
                        }
                        (t, a) => jty(*t) == a,
                    };
                    if !compatible {
                        return None;
                    }
                }
            }
        }
    }
    Some(Plan {
        tys,
        callees,
        anns,
        param_ann,
    })
}

/// Registres lus et écrit par une instruction acceptée par le JIT typé ; `None` pour
/// les autres.
fn reads_writes(p: &Proto, ins: &Instr) -> Option<(Vec<Reg>, Option<Reg>)> {
    use Instr::*;
    Some(match ins {
        Int(d, _) | Float(d, _) | Bool(d, _) | Nil(d) | GetGlobal(d, _) => (vec![], Some(*d)),
        Move(d, a)
        | Guard(d, a, _)
        | FieldAs(d, a, _, _)
        | INeg(d, a)
        | FNeg(d, a)
        | IToF(d, a)
        | Not(d, a)
        | IBinK(_, d, a, _) => (vec![*a], Some(*d)),
        IBin(_, d, a, b) | FBin(_, d, a, b) | Bin(_, d, a, b) | Index(d, a, b) => {
            (vec![*a, *b], Some(*d))
        }
        IterNew { dst, src, .. } => (vec![*src], Some(*dst)),
        CallMethod {
            dst, recv, site, ..
        } => {
            let site = &p.calls[*site as usize];
            let n = site.npos + site.named.len() as u32;
            let mut r: Vec<Reg> = (site.start..site.start + n).collect();
            r.push(*recv);
            (r, Some(*dst))
        }
        Call { dst, f, site } => {
            let site = &p.calls[*site as usize];
            let n = site.npos + site.named.len() as u32;
            let mut r: Vec<Reg> = (site.start..site.start + n).collect();
            r.push(*f);
            (r, Some(*dst))
        }
        _ => return None,
    })
}

/// Instructions `nil` dont la valeur n'est jamais lue (vivacité des registres).
fn dead_nils(p: &Proto) -> Vec<*const Instr> {
    let mut dead = Vec::new();
    let n = p.regs.len();
    let term_reads = |t: &Term| -> Option<Vec<Reg>> {
        Some(match t {
            Term::Jump(_) => vec![],
            Term::Branch { cond, .. } => vec![*cond],
            Term::IfInt { a, b, .. } => vec![*a, *b],
            Term::IfIntK { a, .. } => vec![*a],
            Term::Return { reg, .. } => vec![*reg],
            // L'élément (`dst`) n'est écrit que vers le corps : compté comme non écrit,
            // ce qui est prudent (moins de `nil` déclarés morts).
            Term::IterNext { iter, .. } => vec![*iter],
        })
    };
    let succs = |t: &Term| -> Vec<usize> {
        let mut t = t.clone();
        t.targets_mut().into_iter().map(|x| *x as usize).collect()
    };
    // Registres vivants à l'entrée de chaque bloc (point fixe arrière).
    let mut live_in = vec![vec![false; n]; p.blocks.len()];
    loop {
        let mut changed = false;
        for (bi, b) in p.blocks.iter().enumerate().rev() {
            let mut live = vec![false; n];
            for s in succs(&b.term) {
                for (r, l) in live_in[s].iter().enumerate() {
                    live[r] |= *l;
                }
            }
            let Some(tr) = term_reads(&b.term) else {
                return dead;
            };
            for r in tr {
                live[r as usize] = true;
            }
            for ins in b.instrs.iter().rev() {
                let Some((reads, write)) = reads_writes(p, ins) else {
                    return dead;
                };
                if let Some(w) = write {
                    live[w as usize] = false;
                }
                for r in reads {
                    live[r as usize] = true;
                }
            }
            if live != live_in[bi] {
                live_in[bi] = live;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Un `nil` est mort si son registre n'est pas vivant juste après.
    for b in &p.blocks {
        let mut live = vec![false; n];
        for s in succs(&b.term) {
            for (r, l) in live_in[s].iter().enumerate() {
                live[r] |= *l;
            }
        }
        for r in term_reads(&b.term).unwrap_or_default() {
            live[r as usize] = true;
        }
        for ins in b.instrs.iter().rev() {
            let (reads, write) = reads_writes(p, ins).unwrap_or_default();
            if let Instr::Nil(d) = ins
                && !live[*d as usize]
            {
                dead.push(ins as *const Instr);
            }
            if let Some(w) = write {
                live[w as usize] = false;
            }
            for r in reads {
                live[r as usize] = true;
            }
        }
    }
    dead
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
        builder.symbol("bk_field_native", bk_field_native as *const u8);
        builder.symbol("bk_obj_fields", bk_obj_fields as *const u8);
        builder.symbol("bk_list_view", bk_list_view as *const u8);
        builder.symbol("bk_step", bk_step as *const u8);
        builder.symbol("bk_iter_next", bk_iter_next as *const u8);
        builder.symbol("bk_drop", bk_drop as *const u8);
        for (name, f) in FAST {
            builder.symbol(*name, *f as *const u8);
        }
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
        // Les objets passent par pointeur dans des variables `i64`.
        if ptr != types::I64 {
            return None;
        }
        let mut sig = module.make_signature();
        for _ in 0..3 {
            sig.params.push(AbiParam::new(ptr));
        }
        sig.returns.push(AbiParam::new(types::I32));
        let field_fn = module
            .declare_function("bk_field_native", Linkage::Import, &sig)
            .ok()?;
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.returns.push(AbiParam::new(ptr));
        let fields_fn = module
            .declare_function("bk_obj_fields", Linkage::Import, &sig)
            .ok()?;
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        sig.returns.push(AbiParam::new(types::I32));
        let list_fn = module
            .declare_function("bk_list_view", Linkage::Import, &sig)
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
            field_fn,
            fields_fn,
            list_fn,
            known_types: Vec::new(),
            bailed: false,
            len_sym: crate::intern::Symbol(u32::MAX),
            dyn_fns: None,
            fast_fns: Vec::new(),
            field_caches: Vec::new(),
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
            let plan = analyze(&p, globals, self.len_sym);
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
            // Type d'un paramètre redéclaré : disposition des champs à recompiler.
            for prm in &p.params {
                if let Some(ann) = &prm.ty
                    && let RtType::Named(slot, _) = &*ann.ty
                {
                    globals.watch(*slot, &p);
                }
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
        let field_ref = self
            .module
            .declare_func_in_func(self.field_fn, &mut ctx.func);
        let fields_ref = self
            .module
            .declare_func_in_func(self.fields_fn, &mut ctx.func);
        let list_ref = self
            .module
            .declare_func_in_func(self.list_fn, &mut ctx.func);
        // Paramètres objets dont les champs sont lus : type connu maintenant (case globale
        // de l'annotation), donc position de chaque champ connue.
        let mut obj_types: Vec<Option<Rc<crate::objects::TypeInfo>>> = vec![None; plan.anns.len()];
        for (a, ann) in plan.anns.iter().enumerate() {
            if let RtType::Named(slot, _) = ann
                && let Some(Value::Type(info)) = &globals.at(*slot).value
                && info.kind == crate::ast::TypeDeclKind::Type
            {
                obj_types[a] = Some(info.clone());
                // Gardé en vie : son adresse identifie le type dans le code natif.
                if !self.known_types.iter().any(|t| Rc::ptr_eq(t, info)) {
                    self.known_types.push(info.clone());
                }
            }
        }
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
                Some(JTy::Obj(_) | JTy::List(_)) => Some(bx.declare_var(types::I64)),
                _ => None,
            })
            .collect();
        // Itérateurs : adresse des éléments, nombre, position.
        let iter_vars: FxHashMap<Reg, [Variable; 3]> = plan
            .tys
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t, Some(JTy::Iter(_))))
            .map(|(r, _)| {
                let v = [0; 3].map(|_| bx.declare_var(types::I64));
                (r as Reg, v)
            })
            .collect();
        let entry = bx.create_block();
        let blocks: Vec<_> = p.blocks.iter().map(|_| bx.create_block()).collect();
        let fail = bx.create_block();
        bx.append_block_param(fail, types::I64);
        bx.append_block_params_for_function_params(entry);
        bx.switch_to_block(entry);
        let ctxv = bx.block_params(entry)[0];
        // Case de sortie des lectures de champ.
        let out_slot = bx.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            16,
            3,
        ));
        // Tous les registres sont définis dès l'entrée (valeur nulle).
        for v in iter_vars.values().flatten() {
            let z = bx.ins().iconst(types::I64, 0);
            bx.def_var(*v, z);
        }
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
        // Adresse des champs de chaque objet de type connu (vérifié une fois ici).
        let reads_fields = |a: u32| {
            p.blocks.iter().flat_map(|b| &b.instrs).any(|i| {
                matches!(i, Instr::FieldAs(_, r, ..) if plan.tys[*r as usize] == Some(JTy::Obj(a)))
            })
        };
        let mut fields_ptr: Vec<Option<cranelift_codegen::ir::Value>> = vec![None; plan.anns.len()];
        let mut entry_checks = Vec::new();
        for k in 0..p.params.len() {
            let Some(a) = plan.param_ann[k] else { continue };
            let Some(info) = &obj_types[a as usize] else {
                continue;
            };
            if !reads_fields(a) {
                continue;
            }
            let obj = bx.block_params(entry)[k + 1];
            let ip = bx.ins().iconst(types::I64, Rc::as_ptr(info) as i64);
            let call = bx.ins().call(fields_ref, &[obj, ip]);
            let fp = bx.inst_results(call)[0];
            entry_checks.push(fp);
            fields_ptr[a as usize] = Some(fp);
        }
        let start = bx.create_block();
        for fp in entry_checks {
            let null = bx.ins().icmp_imm_s(IntCC::Equal, fp, 0);
            err_sites.push((ErrKind::Bail, Span::default()));
            let eid = err_base + (err_sites.len() - 1) as i64;
            let cont = bx.create_block();
            let id = bx.ins().iconst(types::I64, eid);
            bx.ins().brif(null, fail, &[id.into()], cont, &[]);
            bx.switch_to_block(cont);
        }
        bx.ins().jump(start, &[]);
        bx.switch_to_block(start);
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
        // Éléments d'une liste : (adresse, nombre) ; abandon si ce n'est pas une liste.
        macro_rules! view {
            ($bx:expr, $list:expr, $span:expr) => {{
                let out = $bx.ins().stack_addr(types::I64, out_slot, 0);
                let call = $bx.ins().call(list_ref, &[$list, out]);
                let status = $bx.inst_results(call)[0];
                check!($bx, status, ErrKind::Bail, $span);
                let data = $bx.ins().load(types::I64, MemFlagsData::trusted(), out, 0);
                let len = $bx.ins().load(types::I64, MemFlagsData::trusted(), out, 8);
                (data, len)
            }};
        }
        // Valeur de l'élément à l'adresse `addr`, du type `anns[a]` : l'adresse pour un
        // objet ; sinon étiquette vérifiée (abandon si elle diffère), `int` vers `float`.
        macro_rules! elem {
            ($bx:expr, $addr:expr, $a:expr, $span:expr) => {{
                use crate::value::layout::{PAYLOAD, TAG_BOOL, TAG_FLOAT, TAG_INT};
                let t = elem_jty(&plan.anns, $a);
                if let JTy::Obj(_) = t {
                    $addr
                } else {
                    let tag = $bx.ins().load(types::I8, MemFlagsData::trusted(), $addr, 0);
                    let raw = $bx
                        .ins()
                        .load(types::I64, MemFlagsData::trusted(), $addr, PAYLOAD);
                    match t {
                        JTy::Int => {
                            let bad = $bx.ins().icmp_imm_s(IntCC::NotEqual, tag, TAG_INT as i64);
                            check!($bx, bad, ErrKind::Bail, $span);
                            raw
                        }
                        JTy::Bool => {
                            let bad = $bx.ins().icmp_imm_s(IntCC::NotEqual, tag, TAG_BOOL as i64);
                            check!($bx, bad, ErrKind::Bail, $span);
                            $bx.ins().ireduce(types::I8, raw)
                        }
                        _ => {
                            let isf = $bx.ins().icmp_imm_s(IntCC::Equal, tag, TAG_FLOAT as i64);
                            let isi = $bx.ins().icmp_imm_s(IntCC::Equal, tag, TAG_INT as i64);
                            let ok = $bx.ins().bor(isf, isi);
                            let bad = $bx.ins().icmp_imm_s(IntCC::Equal, ok, 0);
                            check!($bx, bad, ErrKind::Bail, $span);
                            let f = $bx.ins().bitcast(types::F64, MemFlagsData::new(), raw);
                            let i = $bx.ins().fcvt_from_sint(types::F64, raw);
                            $bx.ins().select(isf, f, i)
                        }
                    }
                }
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
                    // Champ d'un objet de type connu : lecture directe et vérification de
                    // l'étiquette.
                    FieldAs(d, a, name, t)
                        if let Some(JTy::Obj(k)) = tyof(*a)
                            && let Some(info) = &obj_types[k as usize]
                            && let Some(idx) = info.field_index(*name) =>
                    {
                        // Paramètre : adresse des champs prise à l'entrée ; élément d'une
                        // liste : vérifiée à chaque lecture.
                        let fp = match fields_ptr[k as usize] {
                            Some(fp) => fp,
                            None => {
                                let obj = bx.use_var(var(*a));
                                let ip = bx.ins().iconst(types::I64, Rc::as_ptr(info) as i64);
                                let call = bx.ins().call(fields_ref, &[obj, ip]);
                                let fp = bx.inst_results(call)[0];
                                let null = bx.ins().icmp_imm_s(IntCC::Equal, fp, 0);
                                check!(bx, null, ErrKind::Bail, span);
                                fp
                            }
                        };
                        let off = idx as i32 * crate::value::layout::SIZE;
                        let tag = bx.ins().load(types::I8, MemFlagsData::trusted(), fp, off);
                        let (want, cty) = match t {
                            RegTy::Int => (crate::value::layout::TAG_INT, types::I64),
                            RegTy::Float => (crate::value::layout::TAG_FLOAT, types::F64),
                            _ => (crate::value::layout::TAG_BOOL, types::I8),
                        };
                        let bad = bx.ins().icmp_imm_s(IntCC::NotEqual, tag, want as i64);
                        check!(bx, bad, ErrKind::Bail, span);
                        let v = bx.ins().load(
                            cty,
                            MemFlagsData::trusted(),
                            fp,
                            off + crate::value::layout::PAYLOAD,
                        );
                        bx.def_var(var(*d), v);
                    }
                    FieldAs(d, a, name, t) => {
                        self.field_caches.push(Box::new(FieldCache {
                            info: None,
                            idx: 0,
                            name: *name,
                            want: *t,
                        }));
                        let cache = &**self.field_caches.last().unwrap() as *const FieldCache;
                        let obj = bx.use_var(var(*a));
                        let cache = bx.ins().iconst(types::I64, cache as i64);
                        let out = bx.ins().stack_addr(types::I64, out_slot, 0);
                        let call = bx.ins().call(field_ref, &[obj, cache, out]);
                        let status = bx.inst_results(call)[0];
                        check!(bx, status, ErrKind::Bail, span);
                        let bits = bx.ins().load(types::I64, MemFlagsData::trusted(), out, 0);
                        let v = match t {
                            RegTy::Float => bx.ins().bitcast(types::F64, MemFlagsData::new(), bits),
                            RegTy::Bool => bx.ins().ireduce(types::I8, bits),
                            _ => bits,
                        };
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
                    IterNew { dst, src, .. } => {
                        let list = bx.use_var(var(*src));
                        let (data, len) = view!(bx, list, span);
                        let [dv, lv, iv] = iter_vars[dst];
                        let zero = bx.ins().iconst(types::I64, 0);
                        bx.def_var(dv, data);
                        bx.def_var(lv, len);
                        bx.def_var(iv, zero);
                    }
                    Index(d, a, i) => {
                        let Some(JTy::List(e)) = tyof(*a) else {
                            return Err(());
                        };
                        let list = bx.use_var(var(*a));
                        let (data, len) = view!(bx, list, span);
                        // Index négatif : depuis la fin ; hors limites : l'interpréteur
                        // produit l'erreur.
                        let i = bx.use_var(var(*i));
                        let neg = bx.ins().icmp_imm_s(IntCC::SignedLessThan, i, 0);
                        let from_end = bx.ins().iadd(i, len);
                        let pos = bx.ins().select(neg, from_end, i);
                        let out = bx.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, pos, len);
                        check!(bx, out, ErrKind::Bail, span);
                        let off = bx.ins().imul_imm_s(pos, crate::value::layout::SIZE as i64);
                        let addr = bx.ins().iadd(data, off);
                        let v = elem!(bx, addr, e, span);
                        bx.def_var(var(*d), v);
                    }
                    CallMethod { dst, recv, .. } => {
                        let list = bx.use_var(var(*recv));
                        let (_, len) = view!(bx, list, span);
                        bx.def_var(var(*dst), len);
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
                Term::IterNext {
                    iter,
                    dst,
                    body,
                    done,
                } => {
                    let Some(JTy::Iter(a)) = tyof(*iter) else {
                        return Err(());
                    };
                    let [dv, lv, iv] = iter_vars[iter];
                    let (i, len) = (bx.use_var(iv), bx.use_var(lv));
                    let more = bx.ins().icmp(IntCC::SignedLessThan, i, len);
                    let next = bx.create_block();
                    bx.ins().brif(more, next, &[], blocks[*done as usize], &[]);
                    bx.switch_to_block(next);
                    let data = bx.use_var(dv);
                    let off = bx.ins().imul_imm_s(i, crate::value::layout::SIZE as i64);
                    let addr = bx.ins().iadd(data, off);
                    let i1 = bx.ins().iadd_imm_s(i, 1);
                    bx.def_var(iv, i1);
                    let span = b.term_span;
                    let v = elem!(bx, addr, a, span);
                    bx.def_var(var(*dst), v);
                    bx.ins().jump(blocks[*body as usize], &[]);
                }
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
                // Objet : pointeur vers l'argument, vivant pendant tout l'appel.
                v => v as *const Value as u64,
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
        self.bailed = false;
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
        if matches!(info.kind, ErrKind::Bail) {
            self.bailed = true;
            return None;
        }
        if matches!(info.kind, ErrKind::Depth) && limit < max_depth {
            return None;
        }
        let msg = match info.kind {
            ErrKind::Overflow => "dépassement de capacité d'un entier".to_string(),
            ErrKind::DivZero => "division par zéro".to_string(),
            ErrKind::Depth => {
                format!("trop d'appels imbriqués ({max_depth}) : récursion infinie ?")
            }
            ErrKind::Bail => unreachable!(),
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

/// Cache d'un accès à un champ (`p.x`) : dernier type vu et position du champ. Le
/// `Rc` garde le type en vie, donc son adresse ne peut pas désigner un autre type.
struct FieldCache {
    info: Option<Rc<crate::objects::TypeInfo>>,
    idx: usize,
    name: crate::intern::Symbol,
    /// Type attendu (`FieldAs`), ou `Value`.
    want: RegTy,
}

impl FieldCache {
    /// Champ `name` de l'objet `v` (`type` uniquement : une `class` passe par un getter).
    fn get(&mut self, v: &Value) -> Option<Value> {
        let Value::Object(o) = v else {
            return None;
        };
        let obj = o.try_borrow().ok()?;
        let idx = match &self.info {
            Some(info) if Rc::ptr_eq(info, &obj.info) => self.idx,
            _ => {
                if obj.info.kind == crate::ast::TypeDeclKind::Class {
                    return None;
                }
                let i = obj.info.field_index(self.name)?;
                self.info = Some(obj.info.clone());
                self.idx = i;
                i
            }
        };
        let v = obj.fields[idx].clone();
        match (self.want, v) {
            (RegTy::Value, v)
            | (RegTy::Int, v @ Value::Int(_))
            | (RegTy::Float, v @ Value::Float(_))
            | (RegTy::Bool, v @ Value::Bool(_)) => Some(v),
            (RegTy::Float, Value::Int(i)) => Some(Value::Float(i as f64)),
            _ => None,
        }
    }
}

/// Fonctions d'aide rapides : `fn(rt, fenêtre, x, y, p) -> statut`. Statut 0 : fait ;
/// autre : cas non traité, l'interpréteur exécute l'instruction (et produit l'erreur
/// éventuelle). Pour `Call`, 1 signale une erreur (rangée dans le `Runtime`).
#[derive(Clone, Copy)]
enum Fast {
    GetUp,
    SetUp,
    GetCell,
    SetCell,
    GetGlobal,
    SetGlobal,
    Field,
    Call,
}

type FastFn = extern "C" fn(*mut Runtime, *mut NativeFrame, u32, u32, *const u8) -> i32;

const FAST: &[(&str, FastFn)] = &[
    ("bk_get_up", bk_get_up),
    ("bk_set_up", bk_set_up),
    ("bk_get_cell", bk_get_cell),
    ("bk_set_cell", bk_set_cell),
    ("bk_get_global", bk_get_global),
    ("bk_set_global", bk_set_global),
    ("bk_field", bk_field),
    ("bk_call", bk_call),
];

/// Registres de la fenêtre.
///
/// # Safety
/// `f` vient de `Runtime::run_native`, valide pendant l'appel.
unsafe fn frame_regs<'a>(f: *mut NativeFrame) -> &'a mut [Value] {
    unsafe { std::slice::from_raw_parts_mut((*f).regs, (*f).nregs) }
}

/// Lecture d'une variable capturée ou d'une cellule : non traitée si elle n'est pas
/// encore initialisée (l'interpréteur signale l'erreur).
fn load_cell(regs: &mut [Value], d: u32, c: &crate::vm::Cell) -> i32 {
    let v = c.borrow().clone();
    if let Value::Uninit = v {
        return 2;
    }
    regs[d as usize] = v;
    0
}

extern "C" fn bk_get_up(_: *mut Runtime, f: *mut NativeFrame, d: u32, u: u32, _: *const u8) -> i32 {
    // SAFETY : fenêtre valide pendant l'appel (cf. `frame_regs`).
    unsafe {
        let up = &*(*f).upvals.add(u as usize);
        load_cell(frame_regs(f), d, up)
    }
}

extern "C" fn bk_set_up(_: *mut Runtime, f: *mut NativeFrame, u: u32, a: u32, _: *const u8) -> i32 {
    // SAFETY : idem.
    unsafe {
        let up = &*(*f).upvals.add(u as usize);
        *up.borrow_mut() = frame_regs(f)[a as usize].clone();
    }
    0
}

extern "C" fn bk_get_cell(
    _: *mut Runtime,
    f: *mut NativeFrame,
    d: u32,
    c: u32,
    _: *const u8,
) -> i32 {
    // SAFETY : idem.
    unsafe {
        let cell = &*(*f).cells.add(c as usize);
        load_cell(frame_regs(f), d, cell)
    }
}

extern "C" fn bk_set_cell(
    _: *mut Runtime,
    f: *mut NativeFrame,
    c: u32,
    a: u32,
    _: *const u8,
) -> i32 {
    // SAFETY : idem.
    unsafe {
        let cell = &*(*f).cells.add(c as usize);
        *cell.borrow_mut() = frame_regs(f)[a as usize].clone();
    }
    0
}

extern "C" fn bk_get_global(
    rt: *mut Runtime,
    f: *mut NativeFrame,
    d: u32,
    g: u32,
    _: *const u8,
) -> i32 {
    // SAFETY : `rt` et la fenêtre sont valides pendant l'appel.
    unsafe {
        match &(*rt).globals.at(g).value {
            Some(v) => {
                frame_regs(f)[d as usize] = v.clone();
                0
            }
            None => 2,
        }
    }
}

extern "C" fn bk_set_global(
    rt: *mut Runtime,
    f: *mut NativeFrame,
    g: u32,
    a: u32,
    _: *const u8,
) -> i32 {
    // SAFETY : idem.
    unsafe {
        let rt = &mut *rt;
        let slot = rt.globals.at(g);
        if slot.value.is_none() || !slot.mutable {
            return 2;
        }
        let v = frame_regs(f)[a as usize].clone();
        rt.globals.assign(g, v);
    }
    0
}

extern "C" fn bk_field(_: *mut Runtime, f: *mut NativeFrame, d: u32, a: u32, p: *const u8) -> i32 {
    // SAFETY : fenêtre valide ; `p` est un `FieldCache` gardé par le `Jit`.
    unsafe {
        let cache = &mut *(p as *mut FieldCache);
        let regs = frame_regs(f);
        match cache.get(&regs[a as usize]) {
            Some(v) => {
                regs[d as usize] = v;
                0
            }
            None => 2,
        }
    }
}

extern "C" fn bk_call(
    rt: *mut Runtime,
    f: *mut NativeFrame,
    lo: u32,
    hi: u32,
    ins: *const u8,
) -> i32 {
    // SAFETY : comme pour `bk_step`.
    unsafe {
        let rt = &mut *rt;
        let Instr::Call {
            dst,
            f: callee,
            site,
        } = &*(ins as *const Instr)
        else {
            unreachable!()
        };
        let proto = &*(*f).proto;
        match rt.call_instr(
            proto,
            frame_regs(f),
            *dst,
            *callee,
            *site,
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

struct Dyn<'a> {
    regs_ty: &'a [RegTy],
    regs: CValue,
    rt: CValue,
    frame: CValue,
    step: FuncRef,
    drop: FuncRef,
    fast: Vec<FuncRef>,
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

    /// Fonction d'aide rapide ; si elle ne traite pas le cas (statut non nul),
    /// l'instruction passe par l'interpréteur.
    #[allow(clippy::too_many_arguments)]
    fn fast(
        &self,
        bx: &mut FunctionBuilder,
        which: Fast,
        x: u32,
        y: u32,
        p: *const u8,
        ins: &Instr,
        span: Span,
    ) {
        let ptr = bx.func.dfg.value_type(self.regs);
        let x = bx.ins().iconst(types::I32, x as i64);
        let y = bx.ins().iconst(types::I32, y as i64);
        let p = bx.ins().iconst(ptr, p as i64);
        let call = bx
            .ins()
            .call(self.fast[which as usize], &[self.rt, self.frame, x, y, p]);
        let status = bx.inst_results(call)[0];
        let slow = bx.create_block();
        let next = bx.create_block();
        bx.ins().brif(status, slow, &[], next, &[]);
        bx.switch_to_block(slow);
        self.step(bx, ins, span);
        bx.ins().jump(next, &[]);
        bx.switch_to_block(next);
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
        let fast: Vec<FuncRef> = self
            .fast_fns
            .iter()
            .map(|&id| self.module.declare_func_in_func(id, &mut ctx.func))
            .collect();
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
            fast,
            fail,
        };
        for (bi, b) in p.blocks.iter().enumerate() {
            bx.switch_to_block(blocks[bi]);
            for (ins, &span) in b.instrs.iter().zip(&b.spans) {
                dyn_instr(&d, &mut bx, ins, span, &mut self.field_caches);
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
        let mut sig = self.module.make_signature();
        for t in [ptr, ptr, types::I32, types::I32, ptr] {
            sig.params.push(AbiParam::new(t));
        }
        sig.returns.push(AbiParam::new(types::I32));
        for (name, _) in FAST {
            let id = self
                .module
                .declare_function(name, Linkage::Import, &sig)
                .ok()?;
            self.fast_fns.push(id);
        }
        self.dyn_fns = Some((step, iter, drop));
        self.dyn_fns
    }
}

/// Traduit une instruction : en ligne pour les cas simples, sinon `bk_step`.
#[allow(clippy::vec_box)]
fn dyn_instr(
    d: &Dyn,
    bx: &mut FunctionBuilder,
    ins: &Instr,
    span: Span,
    caches: &mut Vec<Box<FieldCache>>,
) {
    use Instr::*;
    let ty = |r: Reg| d.regs_ty[r as usize];
    let null = std::ptr::null::<u8>();
    match ins {
        GetUp(r, u) => d.fast(bx, Fast::GetUp, *r, *u, null, ins, span),
        SetUp(u, a) => d.fast(bx, Fast::SetUp, *u, *a, null, ins, span),
        GetCell(r, c) => d.fast(bx, Fast::GetCell, *r, *c, null, ins, span),
        SetCell(c, a) => d.fast(bx, Fast::SetCell, *c, *a, null, ins, span),
        GetGlobal(r, g) => d.fast(bx, Fast::GetGlobal, *r, *g, null, ins, span),
        SetGlobal(g, a) => d.fast(bx, Fast::SetGlobal, *g, *a, null, ins, span),
        Field(r, a, name) | FieldAs(r, a, name, _) => {
            let want = match ins {
                FieldAs(.., t) => *t,
                _ => RegTy::Value,
            };
            caches.push(Box::new(FieldCache {
                info: None,
                idx: 0,
                name: *name,
                want,
            }));
            let p = &**caches.last().unwrap() as *const FieldCache as *const u8;
            d.fast(bx, Fast::Field, *r, *a, p, ins, span);
        }
        Call { .. } => {
            let ptr = bx.func.dfg.value_type(d.regs);
            let ip = bx.ins().iconst(ptr, ins as *const Instr as i64);
            let lo = bx.ins().iconst(types::I32, span.start as i64);
            let hi = bx.ins().iconst(types::I32, span.end as i64);
            let call = bx
                .ins()
                .call(d.fast[Fast::Call as usize], &[d.rt, d.frame, lo, hi, ip]);
            let status = bx.inst_results(call)[0];
            let cont = bx.create_block();
            bx.ins().brif(status, d.fail, &[], cont, &[]);
            bx.switch_to_block(cont);
        }
        // Garde : l'étiquette attendue, ou un `int` vers `float` ; sinon l'interpréteur
        // produit l'erreur.
        Guard(r, a, t) if *t != RegTy::Value && ty(*r) == *t => {
            let (tag, cty) = scalar(*t);
            let at = d.tag(bx, *a);
            let ok = bx.ins().icmp_imm_s(IntCC::Equal, at, tag as i64);
            let fast = bx.create_block();
            let other = bx.create_block();
            let slow = bx.create_block();
            let next = bx.create_block();
            bx.ins().brif(ok, fast, &[], other, &[]);
            bx.switch_to_block(fast);
            let v = d.payload(bx, *a, cty);
            d.write_const(bx, *r, tag, Some(v));
            bx.ins().jump(next, &[]);
            bx.switch_to_block(other);
            if *t == RegTy::Float {
                let conv = bx.create_block();
                let is_int = bx.ins().icmp_imm_s(IntCC::Equal, at, TAG_INT as i64);
                bx.ins().brif(is_int, conv, &[], slow, &[]);
                bx.switch_to_block(conv);
                let x = d.payload(bx, *a, types::I64);
                let v = bx.ins().fcvt_from_sint(types::F64, x);
                d.write_const(bx, *r, TAG_FLOAT, Some(v));
                bx.ins().jump(next, &[]);
            } else {
                bx.ins().jump(slow, &[]);
            }
            bx.switch_to_block(slow);
            d.step(bx, ins, span);
            bx.ins().jump(next, &[]);
            bx.switch_to_block(next);
        }
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
