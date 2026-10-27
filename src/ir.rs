//! Représentation intermédiaire (IR) : ce que l'interpréteur exécute, et ce que le JIT
//! compilera.
//!
//! - Une fonction ([`Proto`]) est un graphe de **blocs de base** : une suite
//!   d'instructions, terminée par un saut, un branchement ou un retour ([`Term`]).
//! - Les instructions lisent et écrivent des **registres** (variables locales et
//!   temporaires). Chaque registre a un type ([`RegTy`]) : `int`, `float`, `bool` ou
//!   `valeur` (n'importe quelle valeur). Une opération entre registres typés est une
//!   instruction spécialisée (`IBin` : arithmétique entière), que le JIT traduira en
//!   instructions machine ; le reste passe par des instructions génériques (`Bin`).
//! - Une valeur dynamique n'entre dans un registre typé que par une garde (`Guard`), qui
//!   vérifie son type : le typage graduel reste sûr.
//! - Les variables capturées par une closure vivent dans des **cellules** partagées
//!   (`NewCell`, `GetCell`) ; la closure y accède par ses **upvalues** (`GetUp`).
//! - Les variables globales (niveau principal d'un script, fonctions Rust) sont des
//!   cases d'un tableau (`GetGlobal`), qui persistent d'un script à l'autre (REPL).

use crate::ast::{BinOp, FnDecl, TypeDeclKind, TypeExpr};
use crate::intern::{Interner, Symbol};
use crate::span::Span;
use crate::value::Value;
use std::fmt::Write;
use std::rc::Rc;

pub type Reg = u32;
pub type BlockId = u32;

/// Type d'un registre.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegTy {
    /// N'importe quelle valeur.
    Value,
    Int,
    Float,
    Bool,
}

impl RegTy {
    pub fn name(self) -> &'static str {
        ty_name(self)
    }
}

/// Un programme analysé : son source sert aux messages d'erreur.
pub struct Script {
    pub src: Rc<str>,
    pub name: Rc<str>,
}

/// Une fonction compilée (fonction, méthode, lambda, programme principal…).
pub struct Proto {
    pub name: Rc<str>,
    pub script: Rc<Script>,
    pub kind: ProtoKind,
    pub params: Vec<ParamProto>,
    pub ret: Option<TypeRef>,
    /// Types de tous les registres. Méthode : registre 0 = `self` ; puis les paramètres.
    pub regs: Vec<RegTy>,
    pub ncells: u32,
    /// Comment chaque upvalue est capturée dans la fonction englobante (`Closure`).
    pub captures: Vec<Capture>,
    pub blocks: Vec<Block>,
    pub consts: Vec<Value>,
    pub calls: Vec<CallSite>,
    pub shapes: Vec<Rc<Shape>>,
    pub types: Vec<TypeRef>,
    /// Noms des champs des records littéraux.
    pub records: Vec<Rc<[Symbol]>>,
    /// Fonctions imbriquées (lambdas, `def` locaux).
    pub children: Vec<Rc<Proto>>,
    /// Déclarations de `type` / `class` / `enum` (programme principal).
    pub decls: Vec<Decl>,
    /// Déclaration source (signature pour le vérificateur, introspection).
    pub decl: Option<Rc<FnDecl>>,
    /// Compilation native (JIT) de la fonction, version typée.
    pub jit: std::cell::RefCell<JitState>,
    /// Compilation native de base (code non typé) : 0 = pas encore, 1 = impossible,
    /// sinon adresse du point d'entrée `extern "C" fn(*mut Runtime, *mut NativeFrame) -> i32`.
    pub jit_dyn: std::cell::Cell<usize>,
    /// Nombre d'appels (la compilation de base attend le deuxième, sauf s'il y a une boucle).
    pub calls_seen: std::cell::Cell<u32>,
    /// La fonction contient une boucle (saut vers un bloc précédent).
    pub has_loop: bool,
}

/// Fenêtre d'exécution passée au code natif de base (positions fixes).
#[repr(C)]
pub struct NativeFrame {
    pub regs: *mut Value,
    pub nregs: usize,
    pub cells: *mut crate::vm::Cell,
    pub ncells: usize,
    pub upvals: *const crate::vm::Cell,
    pub nupvals: usize,
    pub ctx: *const Rc<[u32]>,
    pub proto: *const Proto,
    /// Registre et forme littérale du résultat (`Return`), remplis par le code natif.
    pub ret: u32,
    pub ret_shape: u32,
}

/// État de la compilation native d'une fonction.
#[derive(Clone, Debug, Default)]
pub enum JitState {
    /// Pas encore essayée (ou invalidée : une fonction appelée a été redéfinie).
    #[default]
    Untried,
    Native(NativeCode),
    /// Non compilable : l'interpréteur l'exécute.
    Rejected,
}

/// Code natif d'une fonction : point d'entrée `extern "C" fn(ctx, args, ret) -> statut`.
#[derive(Clone, Debug)]
pub struct NativeCode {
    pub entry: usize,
    pub params: Vec<RegTy>,
    pub ret: RegTy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtoKind {
    Main,
    Function,
    /// Argument manquant = nil, arguments en trop ignorés.
    Lambda,
    /// Bloc `do` : peut servir de bloc de construction (`Server do … end`).
    DoBlock,
    /// Registre 0 = `self`.
    Method,
    /// Valeur par défaut d'un champ de `type` (`CtxField` lit les autres champs).
    FieldDefault,
}

pub struct ParamProto {
    pub name: Symbol,
    pub reg: Reg,
    pub ty: Option<TypeRef>,
    pub has_default: bool,
    /// `init(@champ)` : index du champ de `self` à initialiser.
    pub field: Option<u32>,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capture {
    /// Cellule de la fonction englobante.
    Cell(u32),
    /// Upvalue de la fonction englobante.
    Up(u32),
}

pub struct Block {
    pub instrs: Vec<Instr>,
    pub spans: Vec<Span>,
    pub term: Term,
    pub term_span: Span,
}

/// Appel : arguments dans les registres `start..start + npos + named.len()`.
pub struct CallSite {
    pub start: Reg,
    pub npos: u32,
    pub named: Vec<Symbol>,
    /// Forme littérale de chaque argument (conversion record → type).
    pub shapes: Vec<Option<u32>>,
    /// Le dernier argument positionnel est un bloc `do`.
    pub block: bool,
}

/// Forme d'une expression littérale, connue à la compilation : sert à convertir un record
/// **littéral** vers le type attendu (`register({path: "/"})`, `[{path: "/a"}]`).
#[derive(Debug)]
pub enum Shape {
    Record(Vec<(Symbol, Option<Rc<Shape>>)>),
    List(Vec<Option<Rc<Shape>>>),
}

/// Annotation de type compilée.
#[derive(Clone, Debug)]
pub struct TypeRef {
    pub ty: Rc<RtType>,
    /// Texte de l'annotation, pour les messages.
    pub shown: Rc<str>,
    pub span: Span,
    /// Annotation source (vérificateur, introspection).
    pub expr: Rc<TypeExpr>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RtType {
    Any,
    Int,
    Float,
    Str,
    Bool,
    Nil,
    List(Box<RtType>),
    Hash(Box<RtType>, Box<RtType>),
    Record(Vec<(Symbol, RtType)>),
    Tuple(Vec<RtType>),
    Optional(Box<RtType>),
    Union(Vec<RtType>),
    LitStr(Rc<str>),
    LitInt(i64),
    Fn,
    /// Type utilisateur : `type`, `class` ou `enum`, dans une case globale.
    Named(u32, Symbol),
}

/// Déclaration de `type` / `class` / `annotation` / `enum`, exécutée par `DefType` /
/// `DefEnum`.
pub enum Decl {
    Type(Rc<TypeDeclIr>),
    Enum(Rc<EnumDeclIr>),
}

pub struct TypeDeclIr {
    pub name: Symbol,
    pub kind: TypeDeclKind,
    pub fields: Vec<FieldIr>,
    pub methods: Vec<(Symbol, Rc<Proto>)>,
}

pub struct FieldIr {
    pub name: Symbol,
    pub ty: Option<TypeRef>,
    /// Valeur par défaut, calculée par une petite fonction.
    pub default: Option<Rc<Proto>>,
}

pub struct EnumDeclIr {
    pub name: Symbol,
    pub variants: Vec<Symbol>,
    /// La variante a une valeur explicite (`= …`), calculée avant `DefEnum`.
    pub explicit: Vec<bool>,
    pub repr: Option<TypeRef>,
    pub repr_is_str: bool,
    pub methods: Vec<(Symbol, Rc<Proto>)>,
}

/// Résolution d'un nom dans un bloc `do` hors construction (ou absent du type construit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    None,
    Reg(Reg),
    Cell(u32),
    Up(u32),
    Global(u32),
}

#[derive(Clone, Debug)]
pub enum Instr {
    // ── Constantes
    Int(Reg, i64),
    Float(Reg, f64),
    Bool(Reg, bool),
    Nil(Reg),
    Const(Reg, u32),
    Move(Reg, Reg),
    /// Erreur à l'exécution (message dans les constantes).
    Fail(u32),

    // ── Variables
    GetGlobal(Reg, u32),
    SetGlobal(u32, Reg),
    /// Définit une globale (`let` ou `def` au niveau principal).
    DefGlobal {
        slot: u32,
        src: Reg,
        mutable: bool,
    },
    NewCell(u32),
    GetCell(Reg, u32),
    SetCell(u32, Reg),
    GetUp(Reg, u32),
    SetUp(u32, Reg),
    /// Nom dans un bloc `do` : champ du type en construction, sinon `fallback`.
    GetNamed {
        dst: Reg,
        name: Symbol,
        fallback: Fallback,
    },
    SetNamed {
        name: Symbol,
        src: Reg,
        fallback: Fallback,
        shape: Option<u32>,
    },
    /// Autre champ du type en construction (valeur par défaut d'un champ).
    CtxField(Reg, u32),
    /// Valeur dynamique vers registre typé, avec vérification (`int` → `float` converti).
    Guard(Reg, Reg, RegTy),
    /// Vérification d'une annotation (et conversion des records littéraux).
    Coerce {
        dst: Reg,
        src: Reg,
        ty: u32,
        shape: Option<u32>,
    },
    /// Paramètre non fourni (sa valeur par défaut doit être calculée) ?
    IsUndef(Reg, Reg),

    // ── Arithmétique typée (registres `int` / `float`)
    IBin(BinOp, Reg, Reg, Reg),
    /// Opération entière avec une constante : `i + 1`, `i < 1000`.
    IBinK(BinOp, Reg, Reg, i64),
    FBin(BinOp, Reg, Reg, Reg),
    INeg(Reg, Reg),
    FNeg(Reg, Reg),
    IToF(Reg, Reg),

    // ── Opérations génériques
    Bin(BinOp, Reg, Reg, Reg),
    Neg(Reg, Reg),
    Not(Reg, Reg),

    // ── Valeurs composées (éléments dans des registres consécutifs)
    List(Reg, Reg, u32),
    Tuple(Reg, Reg, u32),
    /// Paires clé, valeur.
    Hash(Reg, Reg, u32),
    Record(Reg, Reg, u32),
    Range {
        dst: Reg,
        lo: Reg,
        hi: Reg,
        inclusive: bool,
    },
    /// Interpolation : textes des registres concaténés.
    Concat(Reg, Reg, u32),
    Index(Reg, Reg, Reg),
    SetIndex(Reg, Reg, Reg),
    Field(Reg, Reg, Symbol),
    /// Champ dont le vérificateur connaît le type (`int` / `float` / `bool`), vers un
    /// registre de ce type.
    FieldAs(Reg, Reg, Symbol, RegTy),
    SetField {
        obj: Reg,
        name: Symbol,
        src: Reg,
        shape: Option<u32>,
    },
    /// Champ d'objet par index (`@x` dans une méthode).
    FieldAt(Reg, Reg, u32),
    SetFieldAt {
        obj: Reg,
        idx: u32,
        src: Reg,
        shape: Option<u32>,
    },
    TupleIndex(Reg, Reg, u32),
    /// `Enum::Variante`
    Variant(Reg, Reg, Symbol),
    /// Déstructuration : `n` valeurs d'un tuple / d'une liste vers `dst..dst+n`
    /// (`rest` : le reste dans `dst + n`).
    Unpack {
        dst: Reg,
        src: Reg,
        n: u32,
        rest: bool,
    },

    // ── Appels
    Closure(Reg, u32),
    Call {
        dst: Reg,
        f: Reg,
        site: u32,
    },
    CallMethod {
        dst: Reg,
        recv: Reg,
        name: Symbol,
        site: u32,
    },

    // ── Boucles et `case`
    /// Itérateur sur une liste, un intervalle, un hash (`keys` : clés seules)…
    IterNew {
        dst: Reg,
        src: Reg,
        keys: bool,
    },
    /// Le motif d'un `when` correspond-il au sujet ?
    CaseTest(Reg, Reg, Reg),

    // ── Déclarations (programme principal)
    DefType {
        slot: u32,
        decl: u32,
    },
    /// Valeurs explicites des variantes dans `start..`.
    DefEnum {
        slot: u32,
        decl: u32,
        start: Reg,
    },
}

#[derive(Clone, Debug)]
pub enum Term {
    Jump(BlockId),
    /// Selon la vérité de `cond` (`nil` et `false` sont faux).
    Branch {
        cond: Reg,
        then: BlockId,
        else_: BlockId,
    },
    /// Comparaison entière et branchement en une fois (`si i < n`).
    IfInt {
        op: BinOp,
        a: Reg,
        b: Reg,
        then: BlockId,
        else_: BlockId,
    },
    /// Idem avec une constante (`si i < 1000`).
    IfIntK {
        op: BinOp,
        a: Reg,
        k: i64,
        then: BlockId,
        else_: BlockId,
    },
    Return {
        reg: Reg,
        shape: Option<u32>,
    },
    /// Élément suivant dans `dst`, puis `body` ; `done` quand l'itérateur est épuisé.
    IterNext {
        iter: Reg,
        dst: Reg,
        body: BlockId,
        done: BlockId,
    },
}

// ─── Affichage ──────────────────────────────────────────────────────────────

impl Proto {
    /// Texte lisible de l'IR (et de ses fonctions imbriquées).
    pub fn dump(&self, i: &Interner) -> String {
        let mut out = String::new();
        self.dump_into(&mut out, i);
        out
    }

    fn dump_into(&self, out: &mut String, i: &Interner) {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|p| format!("{} r{}", i.resolve(p.name), p.reg))
            .collect();
        let _ = writeln!(
            out,
            "fn {} ({:?}) ({}){}",
            self.name,
            self.kind,
            params.join(", "),
            self.ret
                .as_ref()
                .map_or(String::new(), |r| format!(": {}", r.shown))
        );
        let typed: Vec<String> = self
            .regs
            .iter()
            .enumerate()
            .filter(|(_, t)| **t != RegTy::Value)
            .map(|(n, t)| format!("r{n}:{}", ty_name(*t)))
            .collect();
        if !typed.is_empty() {
            let _ = writeln!(out, "  registres typés : {}", typed.join(" "));
        }
        for (n, b) in self.blocks.iter().enumerate() {
            let _ = writeln!(out, "  b{n}:");
            for ins in &b.instrs {
                let _ = writeln!(out, "    {}", self.show_instr(ins, i));
            }
            let _ = writeln!(out, "    {}", show_term(&b.term));
        }
        for c in &self.children {
            out.push('\n');
            c.dump_into(out, i);
        }
        for d in &self.decls {
            let methods = match d {
                Decl::Type(t) => {
                    for f in &t.fields {
                        if let Some(p) = &f.default {
                            out.push('\n');
                            p.dump_into(out, i);
                        }
                    }
                    &t.methods
                }
                Decl::Enum(e) => &e.methods,
            };
            for (_, m) in methods {
                out.push('\n');
                m.dump_into(out, i);
            }
        }
    }

    fn show_instr(&self, ins: &Instr, i: &Interner) -> String {
        use Instr::*;
        let s = |sym: &Symbol| i.resolve(*sym).to_string();
        match ins {
            Int(d, n) => format!("r{d} = {n}"),
            Float(d, f) => format!("r{d} = {f:?}"),
            Bool(d, b) => format!("r{d} = {b}"),
            Nil(d) => format!("r{d} = nil"),
            Const(d, c) => format!("r{d} = {}", self.consts[*c as usize].inspect(i)),
            Move(d, a) => format!("r{d} = r{a}"),
            Fail(c) => format!("échec {}", self.consts[*c as usize].inspect(i)),
            GetGlobal(d, g) => format!("r{d} = global[{g}]"),
            SetGlobal(g, a) => format!("global[{g}] = r{a}"),
            DefGlobal { slot, src, mutable } => {
                format!(
                    "définit global[{slot}] = r{src}{}",
                    if *mutable { "" } else { " (constante)" }
                )
            }
            NewCell(c) => format!("nouvelle cellule c{c}"),
            GetCell(d, c) => format!("r{d} = c{c}"),
            SetCell(c, a) => format!("c{c} = r{a}"),
            GetUp(d, u) => format!("r{d} = up{u}"),
            SetUp(u, a) => format!("up{u} = r{a}"),
            GetNamed {
                dst,
                name,
                fallback,
            } => format!("r{dst} = nommé {} sinon {fallback:?}", s(name)),
            SetNamed {
                name,
                src,
                fallback,
                ..
            } => format!("nommé {} = r{src} sinon {fallback:?}", s(name)),
            CtxField(d, f) => format!("r{d} = champ en construction #{f}"),
            Guard(d, a, t) => format!("r{d} = garde {} r{a}", ty_name(*t)),
            Coerce { dst, src, ty, .. } => {
                format!(
                    "r{dst} = vérifie r{src} : {}",
                    self.types[*ty as usize].shown
                )
            }
            IsUndef(d, a) => format!("r{d} = non fourni r{a}"),
            IBin(op, d, a, b) => format!("r{d} = r{a} {}i r{b}", op.as_str()),
            IBinK(op, d, a, k) => format!("r{d} = r{a} {}i {k}", op.as_str()),
            FBin(op, d, a, b) => format!("r{d} = r{a} {}f r{b}", op.as_str()),
            INeg(d, a) => format!("r{d} = -i r{a}"),
            FNeg(d, a) => format!("r{d} = -f r{a}"),
            IToF(d, a) => format!("r{d} = float r{a}"),
            Bin(op, d, a, b) => format!("r{d} = r{a} {} r{b}", op.as_str()),
            Neg(d, a) => format!("r{d} = - r{a}"),
            Not(d, a) => format!("r{d} = not r{a}"),
            List(d, a, n) => format!("r{d} = liste r{a}..+{n}"),
            Tuple(d, a, n) => format!("r{d} = tuple r{a}..+{n}"),
            Hash(d, a, n) => format!("r{d} = hash r{a}..+{}", n * 2),
            Record(d, a, r) => {
                let names: Vec<String> = self.records[*r as usize].iter().map(s).collect();
                format!("r{d} = record r{a}.. {{{}}}", names.join(", "))
            }
            Range {
                dst,
                lo,
                hi,
                inclusive,
            } => {
                format!(
                    "r{dst} = r{lo}{}r{hi}",
                    if *inclusive { ".." } else { "..." }
                )
            }
            Concat(d, a, n) => format!("r{d} = concat r{a}..+{n}"),
            Index(d, a, b) => format!("r{d} = r{a}[r{b}]"),
            SetIndex(a, b, v) => format!("r{a}[r{b}] = r{v}"),
            Field(d, a, f) => format!("r{d} = r{a}.{}", s(f)),
            FieldAs(d, a, f, t) => format!("r{d} = r{a}.{} : {}", s(f), t.name()),
            SetField { obj, name, src, .. } => format!("r{obj}.{} = r{src}", s(name)),
            FieldAt(d, a, f) => format!("r{d} = r{a}.#{f}"),
            SetFieldAt { obj, idx, src, .. } => format!("r{obj}.#{idx} = r{src}"),
            TupleIndex(d, a, n) => format!("r{d} = r{a}.{n}"),
            Variant(d, a, v) => format!("r{d} = r{a}::{}", s(v)),
            Unpack { dst, src, n, rest } => {
                format!(
                    "r{dst}..+{n} = déstructure r{src}{}",
                    if *rest { " (+ reste)" } else { "" }
                )
            }
            Closure(d, p) => format!("r{d} = closure {}", self.children[*p as usize].name),
            Call { dst, f, site } => format!("r{dst} = appel r{f}({})", self.show_args(*site, i)),
            CallMethod {
                dst,
                recv,
                name,
                site,
            } => {
                format!("r{dst} = r{recv}.{}({})", s(name), self.show_args(*site, i))
            }
            IterNew { dst, src, keys } => {
                format!(
                    "r{dst} = itérateur r{src}{}",
                    if *keys { " (clés)" } else { "" }
                )
            }
            CaseTest(d, p, v) => format!("r{d} = motif r{p} ~ r{v}"),
            DefType { slot, decl } => format!("définit type #{decl} dans global[{slot}]"),
            DefEnum { slot, decl, start } => {
                format!("définit enum #{decl} dans global[{slot}] (valeurs r{start}..)")
            }
        }
    }

    fn show_args(&self, site: u32, i: &Interner) -> String {
        let c = &self.calls[site as usize];
        let mut parts: Vec<String> = (0..c.npos).map(|k| format!("r{}", c.start + k)).collect();
        for (k, n) in c.named.iter().enumerate() {
            parts.push(format!(
                "{}: r{}",
                i.resolve(*n),
                c.start + c.npos + k as u32
            ));
        }
        if c.block
            && let Some(last) = parts.get_mut(c.npos as usize - 1)
        {
            last.push_str(" (bloc)");
        }
        parts.join(", ")
    }
}

fn ty_name(t: RegTy) -> &'static str {
    match t {
        RegTy::Value => "valeur",
        RegTy::Int => "int",
        RegTy::Float => "float",
        RegTy::Bool => "bool",
    }
}

fn show_term(t: &Term) -> String {
    match t {
        Term::Jump(b) => format!("saut b{b}"),
        Term::Branch { cond, then, else_ } => format!("si r{cond} alors b{then} sinon b{else_}"),
        Term::IfInt {
            op,
            a,
            b,
            then,
            else_,
        } => {
            format!("si r{a} {}i r{b} alors b{then} sinon b{else_}", op.as_str())
        }
        Term::IfIntK {
            op,
            a,
            k,
            then,
            else_,
        } => {
            format!("si r{a} {}i {k} alors b{then} sinon b{else_}", op.as_str())
        }
        Term::Return { reg, .. } => format!("retour r{reg}"),
        Term::IterNext {
            iter,
            dst,
            body,
            done,
        } => {
            format!("r{dst} = suivant r{iter} puis b{body}, fin b{done}")
        }
    }
}

impl Term {
    /// Blocs successeurs (cibles des sauts).
    pub fn targets_mut(&mut self) -> Vec<&mut BlockId> {
        match self {
            Term::Jump(b) => vec![b],
            Term::Branch { then, else_, .. }
            | Term::IfInt { then, else_, .. }
            | Term::IfIntK { then, else_, .. } => vec![then, else_],
            Term::IterNext { body, done, .. } => vec![body, done],
            Term::Return { .. } => vec![],
        }
    }
}
