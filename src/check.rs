//! Static checking: names and types, before execution.
//!
//! Gradual typing: only **certain** errors are reported.
//! - A value of unknown type (`any`: unannotated parameter, Rust function…) is never an
//!   error; the runtime check takes care of it.
//! - For a value that may have several types (`int?`, `int | str`), an operation is only
//!   rejected if none of those types allows it.
//!
//! All errors are collected: analysis continues after each one.
//!
//! Inference: a variable's type is that of its first value (`let x = 1`: `int`, so
//! `x = "a"` is rejected); the return type of an unannotated function is inferred from
//! its body; the parameters of a lambda passed to `map`, `filter`… get the element type.

use crate::ast::*;
use crate::diagnostic::Error;
use crate::globals::Globals;
use crate::intern::{FxHashMap, Interner, Symbol};
use crate::ir::RegTy;
use crate::objects::{EnumInfo, TypeInfo};
use crate::span::Span;
use crate::value::{Function, Value};
use std::cell::RefCell;
use std::rc::Rc;

/// Checks a program. `globals`: variables already defined (registered Rust functions,
/// scripts run before).
///
/// Returns the errors, and the inferred type of `int` / `float` / `bool` variables (by
/// position of their declaration): the compiler turns them into typed registers.
pub fn check(
    program: &Program,
    interner: &Interner,
    globals: &Globals,
) -> (Vec<Error>, Vec<Error>, Prims) {
    let mut c = Checker {
        ast: &program.ast,
        i: interner,
        globals,
        errors: Vec::new(),
        scopes: vec![Scope::default()],
        types: Vec::new(),
        enums: Vec::new(),
        fns: vec![FnCtx::default()],
        method: None,
        decl_ids: FxHashMap::default(),
        module_scopes: FxHashMap::default(),
        quiet: 0,
        imported: FxHashMap::default(),
        prims: Prims::default(),
        warnings: Vec::new(),
        deprecated_fns: FxHashMap::default(),
        deprecated_params: FxHashMap::default(),
    };
    c.block_stmts(&program.body);
    (c.errors, c.warnings, c.prims)
}

/// `int` / `float` / `bool` types known for certain, passed on to the compiler.
#[derive(Default)]
pub struct Prims {
    /// Variables, by position of their declaration.
    pub vars: FxHashMap<u32, RegTy>,
    /// Field reads (`p.x`), by expression: annotated field of a `type`, whose value is
    /// checked on every construction and assignment.
    pub fields: FxHashMap<u32, RegTy>,
}

fn prim(t: &Ty) -> Option<RegTy> {
    match t {
        Ty::Int | Ty::LitInt(_) => Some(RegTy::Int),
        Ty::Float => Some(RegTy::Float),
        Ty::Bool => Some(RegTy::Bool),
        _ => None,
    }
}

// ─── Static types ───────────────────────────────────────────────────────────

type TypeId = usize;
type EnumId = usize;

#[derive(Clone, Debug)]
enum Ty {
    /// Unknown: anything goes.
    Any,
    /// Produces no value (`return`, `break`).
    Never,
    Nil,
    Bool,
    Int,
    Float,
    Str,
    LitStr(Rc<str>),
    LitInt(i64),
    List(Box<Ty>),
    Hash(Box<Ty>, Box<Ty>),
    Record(Vec<(Symbol, Ty)>),
    Tuple(Vec<Ty>),
    Range,
    Union(Vec<Ty>),
    Fn(Rc<FnSig>),
    /// Instance of a `type` / `class`.
    Object(TypeId),
    /// The type itself (callable: construction).
    TypeValue(TypeId),
    Variant(EnumId),
    EnumValue(EnumId),
    /// A module (qualified name): not a value, only `M::x` and `M.f()`.
    Module(Symbol),
}

impl PartialEq for Ty {
    fn eq(&self, o: &Ty) -> bool {
        use Ty::*;
        match (self, o) {
            (Any, Any)
            | (Never, Never)
            | (Nil, Nil)
            | (Bool, Bool)
            | (Int, Int)
            | (Float, Float)
            | (Str, Str)
            | (Range, Range) => true,
            (LitStr(a), LitStr(b)) => a == b,
            (LitInt(a), LitInt(b)) => a == b,
            (List(a), List(b)) => a == b,
            (Hash(a, b), Hash(c, d)) => a == c && b == d,
            (Record(a), Record(b)) => a == b,
            (Tuple(a), Tuple(b)) => a == b,
            (Union(a), Union(b)) => a == b,
            (Fn(a), Fn(b)) => Rc::ptr_eq(a, b),
            (Object(a), Object(b))
            | (TypeValue(a), TypeValue(b))
            | (Variant(a), Variant(b))
            | (EnumValue(a), EnumValue(b)) => a == b,
            (Module(a), Module(b)) => a == b,
            _ => false,
        }
    }
}

impl Ty {
    fn optional(t: Ty) -> Ty {
        union(vec![t, Ty::Nil])
    }

    /// Type of a variable initialized with a value of this type: literals are widened
    /// (`"dev"` → `str`), and `nil` alone says nothing about the future type.
    fn widen(self) -> Ty {
        match self {
            Ty::LitStr(_) => Ty::Str,
            Ty::LitInt(_) => Ty::Int,
            Ty::Nil | Ty::Never => Ty::Any,
            Ty::List(t) => Ty::List(Box::new(t.widen())),
            Ty::Hash(k, v) => Ty::Hash(Box::new(k.widen()), Box::new(v.widen())),
            Ty::Record(fs) => Ty::Record(fs.into_iter().map(|(n, t)| (n, t.widen())).collect()),
            Ty::Tuple(ts) => Ty::Tuple(ts.into_iter().map(Ty::widen).collect()),
            Ty::Union(ts) => union(ts.into_iter().map(Ty::widen).collect()),
            t => t,
        }
    }

    fn members(&self) -> &[Ty] {
        match self {
            Ty::Union(ts) => ts,
            t => std::slice::from_ref(t),
        }
    }

    fn is_any(&self) -> bool {
        matches!(self, Ty::Any)
    }
}

/// Normalized union (flattened, no duplicates; `any` absorbs everything).
fn union(items: Vec<Ty>) -> Ty {
    let mut out: Vec<Ty> = Vec::new();
    for t in items {
        for m in t.members().to_vec() {
            match m {
                Ty::Any => return Ty::Any,
                Ty::Never => {}
                m if out.contains(&m) => {}
                m => out.push(m),
            }
        }
    }
    match out.len() {
        0 => Ty::Never,
        1 => out.pop().unwrap(),
        _ => Ty::Union(out),
    }
}

#[derive(Debug)]
struct FnSig {
    params: Vec<ParamSig>,
    /// Return type: annotated, or inferred once the body is checked.
    ret: RefCell<Ty>,
    /// Lambda: missing argument = nil, extra arguments ignored.
    lenient: bool,
}

#[derive(Debug, Clone)]
struct ParamSig {
    name: Symbol,
    ty: Ty,
    has_default: bool,
}

struct TypeDef {
    name: Symbol,
    kind: TypeDeclKind,
    /// Parent class (its fields and methods are included in `fields` / `methods`).
    parent: Option<TypeId>,
    /// Traits included (by the class or its ancestors).
    traits: Vec<TypeId>,
    /// `abstract def`s not defined yet (signature in `methods`): (name, who requires it).
    required: Vec<(Symbol, Rc<str>)>,
    /// `abstract class` or `trait`: cannot be instantiated.
    is_abstract: bool,
    /// Fields and methods known (a parent or a trait must be, to be used).
    ready: bool,
    fields: Vec<FieldDef>,
    methods: FxHashMap<Symbol, Rc<FnSig>>,
    deprecated: Option<Dep>,
}

/// `@[Deprecated(message?)]`: the message, if written as a literal.
type Dep = Option<Rc<str>>;

#[derive(Clone)]
struct FieldDef {
    name: Symbol,
    ty: Ty,
    has_default: bool,
    /// Annotated type (checked at runtime), rather than inferred from the default value.
    annotated: bool,
    deprecated: Option<Dep>,
}

struct EnumDef {
    name: Symbol,
    variants: Vec<Symbol>,
    value_ty: Ty,
    methods: FxHashMap<Symbol, Rc<FnSig>>,
    deprecated: Option<Dep>,
    variants_deprecated: Vec<Option<Dep>>,
}

// ─── Scopes ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
enum VarKind {
    Let,
    Const,
    Func,
    Type,
    Field,
}

#[derive(Clone)]
struct Var {
    ty: Ty,
    kind: VarKind,
    /// Declared further down in the block (`let` not reached yet).
    pending: bool,
}

#[derive(Default)]
struct Scope {
    vars: FxHashMap<Symbol, Var>,
    /// Body of a module (qualified name).
    module: Option<Symbol>,
    /// Construction block (`T do … end`): the type being built.
    construct: Option<TypeId>,
    /// Function level of the scope (0: program).
    fn_level: usize,
}

#[derive(Default)]
struct FnCtx {
    /// Annotated return type.
    ret: Option<Ty>,
    /// Types of the `return`s encountered (to infer the return type).
    returns: Vec<Ty>,
    loops: u32,
    /// Type of `self` in a method.
    this: Option<Ty>,
}

struct Checker<'a> {
    ast: &'a Ast,
    i: &'a Interner,
    globals: &'a Globals,
    errors: Vec<Error>,
    scopes: Vec<Scope>,
    types: Vec<TypeDef>,
    enums: Vec<EnumDef>,
    fns: Vec<FnCtx>,
    /// Method being checked (class, name): target of `super`.
    method: Option<(TypeId, Symbol)>,
    /// Type of each hoisted declaration (key: address of the `TypeDecl`).
    decl_ids: FxHashMap<usize, TypeId>,
    /// Scope of each module between its two phases (key: address of the `ModuleDecl`).
    module_scopes: FxHashMap<usize, (Symbol, Scope)>,
    /// > 0: errors are not reported (importing declarations already executed).
    quiet: u32,
    /// Types and enums already defined at runtime, imported (key: address of the `Rc`).
    imported: FxHashMap<usize, Ty>,
    prims: Prims,
    warnings: Vec<Error>,
    /// `@[Deprecated]` functions and methods (key: address of the signature).
    deprecated_fns: FxHashMap<usize, Dep>,
    /// `@[Deprecated]` parameters (key: address of the signature, parameter index).
    deprecated_params: FxHashMap<(usize, usize), Dep>,
}

/// Construction block of an unknown type: free assignments, no cascading errors.
const UNKNOWN_TYPE: TypeId = usize::MAX;

/// Built-in methods by receiver type (for suggestions).
const STR_METHODS: &[&str] = &[
    "len",
    "is_empty",
    "upper",
    "lower",
    "capitalize",
    "trim",
    "reverse",
    "contains",
    "starts_with",
    "ends_with",
    "split",
    "replace",
    "chars",
    "to_i",
    "to_f",
    "to_i?",
    "to_f?",
    "to_s",
    "inspect",
];
const NUM_METHODS: &[&str] = &[
    "abs", "to_f", "to_i", "to_f?", "to_i?", "floor", "ceil", "round", "sqrt", "to_s", "inspect",
];
const LIST_METHODS: &[&str] = &[
    "len", "is_empty", "push", "pop", "first", "last", "contains", "join", "reverse", "sort",
    "sum", "min", "max", "map", "filter", "each", "find", "any", "all", "reduce", "to_s",
    "inspect",
];
const RANGE_METHODS: &[&str] = &[
    "len", "contains", "to_list", "map", "filter", "each", "find", "any", "all", "reduce", "to_s",
    "inspect",
];
const HASH_METHODS: &[&str] = &[
    "len", "is_empty", "keys", "values", "has", "get", "delete", "each", "to_s", "inspect",
];

/// Edit distance (swapping two letters counts as one), to suggest a name close to a
/// misspelled one.
fn distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, x) in d[0].iter_mut().enumerate() {
        *x = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = (a[i - 1] != b[j - 1]) as usize;
            d[i][j] = (d[i - 1][j] + 1)
                .min(d[i][j - 1] + 1)
                .min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[a.len()][b.len()]
}

/// " (did you mean `len`?)" if a candidate is close: few typos for a short name, or one
/// is a prefix of the other (`lenght` → `len`).
fn suggest<'s>(name: &str, candidates: impl IntoIterator<Item = &'s str>) -> String {
    let n = name.chars().count();
    let max = match n {
        0..=2 => 0,
        3..=5 => 1,
        _ => 2,
    };
    let best = candidates
        .into_iter()
        .filter(|c| *c != name)
        .filter_map(|c| {
            let d = distance(name, c);
            let prefix =
                c.chars().count() >= 3 && n >= 3 && (name.starts_with(c) || c.starts_with(name));
            (d <= max || prefix).then_some((d, c))
        })
        .min_by_key(|(d, _)| *d);
    match best {
        Some((_, c)) => format!(" (did you mean `{c}`?)"),
        None => String::new(),
    }
}

impl Checker<'_> {
    // ─── Helpers ─────────────────────────────────────────────────────────────

    fn name(&self, s: Symbol) -> &str {
        self.i.resolve(s)
    }

    fn error(&mut self, msg: impl Into<String>, span: Span) {
        if self.quiet == 0 {
            self.errors.push(Error::type_error(msg, span));
        }
    }

    // ─── Deprecated declarations (`@[Deprecated]`) ───────────────────────────

    /// `@[Deprecated(message?)]` among the source annotations.
    fn ast_deprecation(&self, anns: &[AnnotationUse]) -> Option<Dep> {
        let a = anns
            .iter()
            .rev()
            .find(|a| matches!(a.path.as_slice(), [n] if self.name(*n) == "Deprecated"))?;
        Some(
            self.ast
                .get(a.args)
                .first()
                .and_then(|arg| match self.ast.kind(arg.value) {
                    ExprKind::Str(s) => Some(Rc::from(self.name(s))),
                    _ => None,
                }),
        )
    }

    /// `@[Deprecated(message?)]` among the annotations of an already compiled declaration.
    fn ir_deprecation(&self, anns: &crate::ir::Annotations) -> Option<Dep> {
        let a = anns
            .iter()
            .rev()
            .find(|a| self.name(a.name) == "Deprecated")?;
        Some(a.text.clone())
    }

    fn mark_fn(&mut self, sig: &Rc<FnSig>, dep: Option<Dep>) {
        if let Some(d) = dep {
            self.deprecated_fns.insert(Rc::as_ptr(sig) as usize, d);
        }
    }

    /// Deprecated parameters of a function of this program (source annotations).
    fn mark_params_ast(&mut self, sig: &Rc<FnSig>, params: Option<&[Param]>) {
        for (k, p) in params.unwrap_or(&[]).iter().enumerate() {
            if let Some(d) = self.ast_deprecation(&p.annotations) {
                self.deprecated_params
                    .insert((Rc::as_ptr(sig) as usize, k), d);
            }
        }
    }

    /// Deprecated parameters of an already compiled function (IR annotations: the AST ones
    /// point into another program's source).
    fn mark_params_ir(&mut self, sig: &Rc<FnSig>, p: &crate::ir::Proto) {
        for (k, prm) in p.params.iter().enumerate() {
            if let Some(d) = self.ir_deprecation(&prm.annotations) {
                self.deprecated_params
                    .insert((Rc::as_ptr(sig) as usize, k), d);
            }
        }
    }

    /// Warns if the declaration being used (`what`) is deprecated.
    fn warn_deprecated(&mut self, dep: Option<&Dep>, what: &str, span: Span) {
        let Some(dep) = dep else { return };
        if self.quiet > 0 {
            return;
        }
        let msg = match dep {
            Some(m) => format!("{what} is deprecated: {m}"),
            None => format!("{what} is deprecated"),
        };
        self.warnings.push(Error::warning(msg, span));
    }

    /// Warns if the value named `name` (function, type, enum) is deprecated.
    fn warn_value(&mut self, ty: &Ty, name: Symbol, span: Span) {
        let dep = match ty {
            Ty::Fn(sig) => self
                .deprecated_fns
                .get(&(Rc::as_ptr(sig) as usize))
                .cloned(),
            Ty::TypeValue(id) if *id != UNKNOWN_TYPE => self.types[*id].deprecated.clone(),
            Ty::EnumValue(id) => self.enums[*id].deprecated.clone(),
            _ => None,
        };
        let what = format!("`{}`", self.name(name));
        self.warn_deprecated(dep.as_ref(), &what, span);
    }

    fn show(&self, t: &Ty) -> String {
        match t {
            Ty::Any => "any".into(),
            Ty::Never => "never".into(),
            Ty::Nil => "nil".into(),
            Ty::Bool => "bool".into(),
            Ty::Int => "int".into(),
            Ty::Float => "float".into(),
            Ty::Str => "str".into(),
            Ty::LitStr(s) => format!("{s:?}"),
            Ty::LitInt(n) => n.to_string(),
            Ty::List(t) => format!("[{}]", self.show(t)),
            Ty::Hash(k, v) => format!("{{{} => {}}}", self.show(k), self.show(v)),
            Ty::Record(fs) => {
                let fs: Vec<String> = fs
                    .iter()
                    .map(|(n, t)| format!("{}: {}", self.name(*n), self.show(t)))
                    .collect();
                format!("{{{}}}", fs.join(", "))
            }
            Ty::Tuple(ts) => {
                let ts: Vec<String> = ts.iter().map(|t| self.show(t)).collect();
                format!("({})", ts.join(", "))
            }
            Ty::Range => "range".into(),
            Ty::Union(ts) => {
                if ts.len() == 2 && ts.contains(&Ty::Nil) {
                    let other = ts.iter().find(|t| **t != Ty::Nil).unwrap();
                    return format!("{}?", self.show(other));
                }
                let ts: Vec<String> = ts.iter().map(|t| self.show(t)).collect();
                ts.join(" | ")
            }
            Ty::Fn(_) => "function".into(),
            Ty::Object(id) => self.name(self.types[*id].name).into(),
            Ty::TypeValue(id) => format!("type {}", self.name(self.types[*id].name)),
            Ty::Variant(id) => self.name(self.enums[*id].name).into(),
            Ty::EnumValue(id) => format!("enum {}", self.name(self.enums[*id].name)),
            Ty::Module(m) => format!("module {}", self.name(*m)),
        }
    }

    /// Type of a value in a message: literals widened (`str` rather than `"a"`).
    fn show_v(&self, t: &Ty) -> String {
        self.show(&plain(t))
    }

    /// Received type, for a message: `str` rather than `"a"`, unless literals are expected
    /// (`"dev" | "prod"`), where the exact value is informative.
    fn show_got(&self, got: &Ty, expected: &Ty) -> String {
        let wants_literal = expected
            .members()
            .iter()
            .any(|m| matches!(m, Ty::LitStr(_) | Ty::LitInt(_)));
        if wants_literal {
            return self.show(got);
        }
        self.show_v(got)
    }

    fn fn_level(&self) -> usize {
        self.fns.len() - 1
    }

    fn push_scope(&mut self) {
        let fn_level = self.fn_level();
        self.scopes.push(Scope {
            fn_level,
            ..Default::default()
        });
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    // ─── Modules ─────────────────────────────────────────────────────────────

    /// `prefix::name`, if it was interned (declared somewhere).
    fn qualified(&self, prefix: Symbol, name: Symbol) -> Option<Symbol> {
        self.i
            .get(&format!("{}::{}", self.name(prefix), self.name(name)))
    }

    /// Name of a declaration of the current scope (in a module: `Module::name`).
    fn decl_name(&self, name: Symbol) -> Symbol {
        match self.scopes.last().and_then(|s| s.module) {
            Some(m) => self.qualified(m, name).unwrap_or(name),
            None => name,
        }
    }

    /// Enclosing modules, innermost first.
    fn modules(&self) -> Vec<Symbol> {
        self.scopes.iter().rev().filter_map(|s| s.module).collect()
    }

    fn is_module(&mut self, q: Symbol) -> bool {
        matches!(self.lookup(q).map(|(v, _)| v.ty.clone()), Some(Ty::Module(_)))
            || self.globals.is_module(q)
    }

    /// Module named by a path written here, looked up in the enclosing modules first.
    fn module_path(&mut self, path: &[Symbol]) -> Option<Symbol> {
        let (&head, rest) = path.split_first()?;
        let mut q = match self.lookup(head).map(|(v, _)| v.ty.clone()) {
            Some(Ty::Module(q)) => q,
            Some(_) => return None,
            None => {
                let mut found = None;
                for p in self.modules().into_iter().map(Some).chain([None]) {
                    let q = match p {
                        Some(p) => self.qualified(p, head),
                        None => Some(head),
                    };
                    if let Some(q) = q
                        && self.is_module(q)
                    {
                        found = Some(q);
                        break;
                    }
                }
                found?
            }
        };
        for &seg in rest {
            q = self.qualified(q, seg)?;
            if !self.is_module(q) {
                return None;
            }
        }
        Some(q)
    }

    /// The expression names a module (and not a variable).
    fn module_of(&mut self, e: ExprId) -> Option<Symbol> {
        let ast = self.ast;
        match ast.kind(e) {
            ExprKind::Ident(n) => self.module_path(&[n]),
            ExprKind::Path(p) => self.module_path(ast.get(p)),
            _ => None,
        }
    }

    /// Type of member `name` of module `m` (`None`: no such member).
    fn member_ty(&mut self, m: Symbol, name: Symbol) -> Option<Ty> {
        // Module being checked: its scope is still open.
        if let Some(v) = self
            .scopes
            .iter()
            .rev()
            .find(|s| s.module == Some(m))
            .and_then(|s| s.vars.get(&name))
        {
            return Some(v.ty.clone());
        }
        let q = self.qualified(m, name)?;
        match self.lookup(q).map(|(v, _)| v.ty.clone()) {
            Some(t) => Some(t),
            None => self.global_ty(q).map(|(t, _)| t),
        }
    }

    /// `M::x`, `M.f`: the member's type, or an error.
    fn member(&mut self, m: Symbol, name: Symbol, span: Span) -> Ty {
        match self.member_ty(m, name) {
            Some(t) => t,
            None => {
                let msg = format!("module `{}` has no `{}`", self.name(m), self.name(name));
                self.error(msg, span);
                Ty::Any
            }
        }
    }

    /// Module, first phase (with the enclosing block's declarations): its own
    /// declarations, visible outside by their qualified name.
    fn module_decls(&mut self, m: &ModuleDecl) {
        let outer = self.scopes.last().and_then(|s| s.module);
        let q = match outer {
            Some(o) => self.qualified(o, m.name).unwrap_or(m.name),
            None => m.name,
        };
        self.push_scope();
        self.scopes.last_mut().unwrap().module = Some(q);
        self.hoist(&m.body);
        self.close_module(m, q);
    }

    /// Second phase, once the enclosing block is declared: its statements (run when
    /// the block starts, before the block's own statements).
    fn module_body(&mut self, m: &ModuleDecl) {
        let key = m as *const ModuleDecl as usize;
        let Some((q, scope)) = self.module_scopes.remove(&key) else {
            return;
        };
        self.scopes.push(scope);
        for s in &m.body.stmts {
            if let StmtKind::Module(inner) = &s.kind {
                self.module_body(inner);
            }
        }
        for s in &m.body.stmts {
            self.stmt(s);
        }
        self.close_module(m, q);
    }

    /// Pops the module's scope (kept for its second phase) and exports its names.
    fn close_module(&mut self, m: &ModuleDecl, q: Symbol) {
        let scope = self.scopes.pop().unwrap();
        for (n, v) in &scope.vars {
            if let Some(qn) = self.qualified(q, *n) {
                self.scopes[0].vars.insert(qn, v.clone());
            }
        }
        self.module_scopes
            .insert(m as *const ModuleDecl as usize, (q, scope));
    }

    fn declare(&mut self, name: Symbol, ty: Ty, kind: VarKind) {
        self.scopes.last_mut().unwrap().vars.insert(
            name,
            Var {
                ty,
                kind,
                pending: false,
            },
        );
    }

    /// Looks up a variable; `Some((var, function level of its scope))`.
    fn lookup(&self, name: Symbol) -> Option<(&Var, usize)> {
        self.scopes
            .iter()
            .rev()
            .find_map(|s| s.vars.get(&name).map(|v| (v, s.fn_level)))
    }

    fn known_names(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for s in &self.scopes {
            for n in s.vars.keys() {
                out.push(self.name(*n));
            }
        }
        out
    }

    // ─── Already executed declarations ───────────────────────────────────────

    /// Type of a global variable defined before this program (previous script, REPL, Rust
    /// function) and whether it is mutable.
    fn global_ty(&mut self, name: Symbol) -> Option<(Ty, bool)> {
        if self.globals.is_module(name) {
            return Some((Ty::Module(name), false));
        }
        let (v, mutable) = self.globals.get_binding(name)?;
        let ty = match &v {
            Value::Type(info) => Ty::TypeValue(self.import_type(info)),
            Value::Enum(e) => Ty::EnumValue(self.import_enum(e)),
            Value::Function(f) => match &**f {
                Function::Script(c) => match &c.proto.decl {
                    Some(d) => {
                        self.quiet += 1;
                        let sig = self.signature(
                            d.params.as_deref().unwrap_or(&[]),
                            d.ret.as_ref(),
                            false,
                        );
                        self.quiet -= 1;
                        let dep = self.ir_deprecation(&c.proto.annotations);
                        self.mark_fn(&sig, dep);
                        self.mark_params_ir(&sig, &c.proto);
                        Ty::Fn(sig)
                    }
                    None => Ty::Any,
                },
                _ => Ty::Any,
            },
            v => value_ty(v),
        };
        Some((ty, mutable))
    }

    fn import_type(&mut self, info: &Rc<TypeInfo>) -> TypeId {
        let key = Rc::as_ptr(info) as usize;
        if let Some(Ty::TypeValue(id)) = self.imported.get(&key) {
            return *id;
        }
        let id = self.types.len();
        let name = self.i.get(&info.name).unwrap_or(Symbol(0));
        let deprecated = self.ir_deprecation(&info.annotations);
        self.types.push(TypeDef {
            name,
            kind: info.kind,
            parent: None,
            traits: Vec::new(),
            required: info.required.iter().map(|r| (r.name, r.by.clone())).collect(),
            is_abstract: info.is_abstract,
            ready: true,
            fields: Vec::new(),
            methods: FxHashMap::default(),
            deprecated,
        });
        self.imported.insert(key, Ty::TypeValue(id));
        self.types[id].parent = info.parent.as_ref().map(|p| self.import_type(p));
        self.types[id].traits = info.traits.iter().map(|t| self.import_type(t)).collect();
        self.quiet += 1;
        let fields = info
            .fields
            .iter()
            .map(|f| FieldDef {
                name: f.name,
                ty: f.ty.as_ref().map_or(Ty::Any, |t| self.resolve(&t.expr)),
                has_default: f.default.is_some(),
                annotated: f.ty.is_some(),
                deprecated: self.ir_deprecation(&f.annotations),
            })
            .collect();
        let mut methods: FxHashMap<Symbol, Rc<FnSig>> = info
            .methods
            .iter()
            .map(|(n, m)| (*n, self.proto_signature(m)))
            .collect();
        for r in &info.required {
            let d = &r.decl;
            let sig = self.signature(d.params.as_deref().unwrap_or(&[]), d.ret.as_ref(), false);
            methods.insert(r.name, sig);
        }
        self.quiet -= 1;
        self.types[id].fields = fields;
        self.types[id].methods = methods;
        id
    }

    fn import_enum(&mut self, e: &Rc<EnumInfo>) -> EnumId {
        let key = Rc::as_ptr(e) as usize;
        if let Some(Ty::EnumValue(id)) = self.imported.get(&key) {
            return *id;
        }
        let id = self.enums.len();
        let deprecated = self.ir_deprecation(&e.annotations);
        let variants_deprecated = e
            .variant_annotations
            .iter()
            .map(|a| self.ir_deprecation(a))
            .collect();
        self.enums.push(EnumDef {
            name: self.i.get(&e.name).unwrap_or(Symbol(0)),
            variants: e.variants.iter().map(|(n, _)| *n).collect(),
            value_ty: union(e.variants.iter().map(|(_, v)| value_ty(v)).collect()),
            methods: FxHashMap::default(),
            deprecated,
            variants_deprecated,
        });
        self.imported.insert(key, Ty::EnumValue(id));
        self.quiet += 1;
        let methods = e
            .methods
            .iter()
            .map(|(n, m)| (*n, self.proto_signature(m)))
            .collect();
        self.quiet -= 1;
        self.enums[id].methods = methods;
        id
    }

    /// Signature of an already compiled method (from its source declaration).
    fn proto_signature(&mut self, p: &crate::ir::Proto) -> Rc<FnSig> {
        match &p.decl {
            Some(d) => {
                let sig = self.signature(d.params.as_deref().unwrap_or(&[]), d.ret.as_ref(), false);
                let dep = self.ir_deprecation(&p.annotations);
                self.mark_fn(&sig, dep);
                self.mark_params_ir(&sig, p);
                sig
            }
            None => Rc::new(FnSig {
                params: Vec::new(),
                ret: RefCell::new(Ty::Any),
                lenient: true,
            }),
        }
    }

    // ─── Type annotations ────────────────────────────────────────────────────

    fn resolve(&mut self, t: &TypeExpr) -> Ty {
        match &t.kind {
            TypeKind::Named(path) => {
                if let [name] = path.as_slice() {
                    match self.name(*name) {
                        "any" => return Ty::Any,
                        "int" => return Ty::Int,
                        "float" => return Ty::Float,
                        "str" => return Ty::Str,
                        "bool" => return Ty::Bool,
                        "nil" => return Ty::Nil,
                        _ => {}
                    }
                    let mut found = self.lookup(*name).map(|(v, _)| v.ty.clone());
                    if found.is_none() {
                        for m in self.modules() {
                            if self.qualified(m, *name).is_some() {
                                found = self.member_ty(m, *name);
                                if found.is_some() {
                                    break;
                                }
                            }
                        }
                    }
                    if found.is_none() {
                        found = self.global_ty(*name).map(|g| g.0);
                    }
                    match found {
                        Some(Ty::TypeValue(id)) => return Ty::Object(id),
                        Some(Ty::EnumValue(id)) => return Ty::Variant(id),
                        _ => {}
                    }
                } else if let Some((&last, prefix)) = path.split_last()
                    && let Some(m) = self.module_path(prefix)
                {
                    match self.member_ty(m, last) {
                        Some(Ty::TypeValue(id)) => return Ty::Object(id),
                        Some(Ty::EnumValue(id)) => return Ty::Variant(id),
                        _ => {}
                    }
                }
                let shown: Vec<&str> = path.iter().map(|s| self.name(*s)).collect();
                let shown = shown.join("::");
                let candidates: Vec<String> = ["any", "int", "float", "str", "bool", "nil"]
                    .iter()
                    .map(|s| s.to_string())
                    .chain(self.types.iter().map(|t| self.name(t.name).to_string()))
                    .chain(self.enums.iter().map(|e| self.name(e.name).to_string()))
                    .collect();
                let hint = suggest(&shown, candidates.iter().map(|s| s.as_str()));
                self.error(format!("unknown type `{shown}`{hint}"), t.span);
                Ty::Any
            }
            TypeKind::List(inner) => Ty::List(Box::new(self.resolve(inner))),
            TypeKind::Hash(k, v) => Ty::Hash(Box::new(self.resolve(k)), Box::new(self.resolve(v))),
            TypeKind::Record(fs) => {
                Ty::Record(fs.iter().map(|(n, t)| (*n, self.resolve(t))).collect())
            }
            TypeKind::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| self.resolve(t)).collect()),
            TypeKind::Optional(inner) => Ty::optional(self.resolve(inner)),
            TypeKind::Union(ts) => union(ts.iter().map(|t| self.resolve(t)).collect()),
            TypeKind::LitStr(s) => Ty::LitStr(s.as_str().into()),
            TypeKind::LitInt(n) => Ty::LitInt(*n),
            TypeKind::Fn(params, ret) => {
                let params = params
                    .iter()
                    .enumerate()
                    .map(|(i, p)| ParamSig {
                        name: Symbol(u32::MAX - i as u32),
                        ty: self.resolve(p),
                        has_default: false,
                    })
                    .collect();
                let ret = self.resolve(ret);
                Ty::Fn(Rc::new(FnSig {
                    params,
                    ret: RefCell::new(ret),
                    lenient: false,
                }))
            }
        }
    }

    // ─── Compatibility ───────────────────────────────────────────────────────

    /// Can a value of type `from` be used where `to` is expected?
    /// True if it is possible for *at least one* value (gradual typing).
    fn assignable(&self, from: &Ty, to: &Ty) -> bool {
        use Ty::*;
        match (from, to) {
            (Any, _) | (_, Any) | (Never, _) => true,
            (Union(fs), _) => fs.iter().any(|f| self.assignable(f, to)),
            (_, Union(ts)) => ts.iter().any(|t| self.assignable(from, t)),
            (LitStr(_), Str) | (LitInt(_), Int) | (LitInt(_), Float) | (Int, Float) => true,
            // An arbitrary `str` may be `"dev"`: checked at runtime.
            (Str, LitStr(_)) | (Int, LitInt(_)) => true,
            (List(a), List(b)) => self.assignable(a, b),
            (Hash(a, b), Hash(c, d)) => self.assignable(a, c) && self.assignable(b, d),
            (Tuple(a), Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| self.assignable(x, y))
            }
            (Record(a), Record(b)) => {
                a.len() == b.len()
                    && b.iter()
                        .all(|(n, t)| a.iter().any(|(m, u)| m == n && self.assignable(u, t)))
            }
            (Fn(_) | TypeValue(_), Fn(_)) => true,
            (Object(a), Object(b)) => self.descends(*a, *b),
            _ => from == to,
        }
    }

    /// Is `a` the class `b` or one of its descendants?
    fn descends(&self, mut a: TypeId, b: TypeId) -> bool {
        if self.types.get(b).is_some_and(|t| t.kind == TypeDeclKind::Trait) {
            return self.types.get(a).is_some_and(|t| t.traits.contains(&b)) || a == b;
        }
        loop {
            if a == b {
                return true;
            }
            match self.types.get(a).and_then(|t| t.parent) {
                Some(p) => a = p,
                None => return false,
            }
        }
    }

    /// Checks an expression against an expected type (with literal conversion).
    fn expect(&mut self, e: ExprId, to: &Ty, what: &str) -> Ty {
        let t = self.expr(e, Some(to));
        if !self.assignable(&t, to) {
            let msg = format!(
                "expected type `{}`{what}, got `{}`",
                self.show(to),
                self.show_got(&t, to)
            );
            self.error(msg, self.ast.span(e));
        }
        t
    }

    // ─── Blocks and statements ───────────────────────────────────────────────

    /// Statements of a block in the current scope (declarations hoisted).
    fn block_stmts(&mut self, b: &Block) -> Ty {
        self.hoist(b);
        for s in &b.stmts {
            if let StmtKind::Module(m) = &s.kind {
                self.module_body(m);
            }
        }
        let mut last = Ty::Nil;
        for s in &b.stmts {
            last = self.stmt(s);
        }
        last
    }

    fn block(&mut self, b: &Block) -> Ty {
        self.push_scope();
        let t = self.block_stmts(b);
        self.pop_scope();
        t
    }

    /// Declares the block's types, enums, functions and `let`s ahead of time (like execution
    /// does for the first three; `let`s are marked "further down").
    fn hoist(&mut self, b: &Block) {
        // 1. Names of types and enums (they may refer to each other).
        let mut pending_types = Vec::new();
        let mut seen: FxHashMap<Symbol, ()> = FxHashMap::default();
        for s in &b.stmts {
            let name = match &s.kind {
                StmtKind::Type(t) => Some((t.name, t.name_span)),
                StmtKind::Enum(e) => Some((e.name, e.name_span)),
                _ => None,
            };
            if let Some((n, span)) = name
                && seen.insert(n, ()).is_some()
            {
                let msg = format!("`{}` is declared twice in this block", self.name(n));
                self.error(msg, span);
                continue;
            }
            match &s.kind {
                StmtKind::Type(t) => {
                    self.decl_ids.insert(&**t as *const TypeDecl as usize, self.types.len());
                    let id = self.types.len();
                    let deprecated = self.ast_deprecation(&t.annotations);
                    let qname = self.decl_name(t.name);
                    self.types.push(TypeDef {
                        name: qname,
                        kind: t.kind,
                        parent: None,
                        traits: Vec::new(),
                        required: Vec::new(),
                        is_abstract: t.is_abstract || t.kind == TypeDeclKind::Trait,
                        ready: false,
                        fields: Vec::new(),
                        methods: FxHashMap::default(),
                        deprecated,
                    });
                    self.declare(t.name, Ty::TypeValue(id), VarKind::Type);
                    pending_types.push((id, &**t));
                }
                StmtKind::Module(m) => {
                    let q = self.decl_name(m.name);
                    self.declare(m.name, Ty::Module(q), VarKind::Type);
                }
                StmtKind::Enum(e) => {
                    let id = self.enums.len();
                    let deprecated = self.ast_deprecation(&e.annotations);
                    let variants_deprecated = e
                        .variants
                        .iter()
                        .map(|v| self.ast_deprecation(&v.annotations))
                        .collect();
                    let qname = self.decl_name(e.name);
                    self.enums.push(EnumDef {
                        name: qname,
                        variants: e.variants.iter().map(|v| v.name).collect(),
                        value_ty: Ty::Any,
                        methods: FxHashMap::default(),
                        deprecated,
                        variants_deprecated,
                    });
                    self.declare(e.name, Ty::EnumValue(id), VarKind::Type);
                }
                _ => {}
            }
        }
        // Declarations of the modules (their bodies run before the block's types).
        for s in &b.stmts {
            if let StmtKind::Module(m) = &s.kind {
                if self.scopes.iter().skip(1).any(|s| s.module.is_none()) || self.fns.len() > 1 {
                    self.error(
                        "a `module` must be at the top level of the script (or in a module)",
                        s.span,
                    );
                } else {
                    self.module_decls(m);
                }
            }
        }
        // 2. Fields and method signatures (a parent class's first: it must be declared
        // before).
        for (id, t) in pending_types {
            let parent = t.parent.and_then(|(p, span)| {
                let found = match self.lookup(p) {
                    Some((v, _)) => Some(v.ty.clone()),
                    None => self.global_ty(p).map(|(t, _)| t),
                };
                match found {
                    Some(Ty::TypeValue(pid))
                        if self.types[pid].kind == TypeDeclKind::Class && self.types[pid].ready =>
                    {
                        Some(pid)
                    }
                    _ => {
                        let msg = format!(
                            "`{}` is not a class declared before `{}`",
                            self.name(p),
                            self.name(t.name)
                        );
                        self.error(msg, span);
                        None
                    }
                }
            });
            let (mut fields, mut methods) = match parent {
                Some(pid) => (self.types[pid].fields.clone(), self.types[pid].methods.clone()),
                None => (Vec::new(), FxHashMap::default()),
            };
            let (mut required, mut traits) = match parent {
                Some(pid) => (self.types[pid].required.clone(), self.types[pid].traits.clone()),
                None => (Vec::new(), Vec::new()),
            };
            self.types[id].parent = parent;
            // Methods with code (not only required), the parent's first.
            let mut defined: Vec<Symbol> = methods
                .keys()
                .filter(|n| !required.iter().any(|(r, _)| r == *n))
                .copied()
                .collect();
            // Traits first: the class's own methods take precedence.
            for m in &t.members {
                let Member::Include(n, span) = m else { continue };
                let found = match self.lookup(*n) {
                    Some((v, _)) => Some(v.ty.clone()),
                    None => self.global_ty(*n).map(|(t, _)| t),
                };
                match found {
                    Some(Ty::TypeValue(tid))
                        if self.types[tid].kind == TypeDeclKind::Trait && self.types[tid].ready =>
                    {
                        let tr = &self.types[tid];
                        for (name, sig) in &tr.methods {
                            if tr.required.iter().any(|(r, _)| r == name) {
                                methods.entry(*name).or_insert_with(|| sig.clone());
                            } else {
                                methods.insert(*name, sig.clone());
                                defined.push(*name);
                            }
                        }
                        required.extend(tr.required.iter().cloned());
                        if !traits.contains(&tid) {
                            traits.push(tid);
                        }
                    }
                    _ => {
                        let msg = format!(
                            "`{}` is not a trait declared before `{}`",
                            self.name(*n),
                            self.name(t.name)
                        );
                        self.error(msg, *span);
                    }
                }
            }
            for m in &t.members {
                match m {
                    Member::Field(f) => {
                        if parent.is_some() && fields.iter().any(|x: &FieldDef| x.name == f.name) {
                            let msg = format!(
                                "field `{}` is already declared by `{}`",
                                self.name(f.name),
                                self.name(t.parent.unwrap().0)
                            );
                            self.error(msg, f.span);
                            continue;
                        }
                        let ty = f.ty.as_ref().map_or(Ty::Any, |t| self.resolve(t));
                        // Type inferred from the literal default value (`port = 80`).
                        let ty = match (&f.ty, f.default) {
                            (None, Some(d)) => self.literal_ty(d),
                            _ => ty,
                        };
                        fields.push(FieldDef {
                            name: f.name,
                            ty,
                            has_default: f.default.is_some(),
                            annotated: f.ty.is_some(),
                            deprecated: self.ast_deprecation(&f.annotations),
                        });
                    }
                    Member::Method(f) => {
                        let sig = self.signature(
                            f.params.as_deref().unwrap_or(&[]),
                            f.ret.as_ref(),
                            false,
                        );
                        let dep = self.ast_deprecation(&f.annotations);
                        self.mark_fn(&sig, dep);
                        self.mark_params_ast(&sig, f.params.as_deref());
                        methods.insert(f.name, sig);
                        defined.push(f.name);
                    }
                    Member::Abstract(f) => {
                        let sig = self.signature(
                            f.params.as_deref().unwrap_or(&[]),
                            f.ret.as_ref(),
                            false,
                        );
                        methods.entry(f.name).or_insert(sig);
                        required.push((f.name, self.i.resolve(t.name).into()));
                    }
                    Member::Stmt(_) | Member::Include(..) => {}
                }
            }
            // `init(@field)` also declares fields.
            let init = self.i.get("init");
            if let Some(Member::Method(init)) = t
                .members
                .iter()
                .find(|m| matches!(m, Member::Method(f) if Some(f.name) == init))
            {
                for p in init.params.iter().flatten().filter(|p| p.is_field) {
                    if !fields.iter().any(|f| f.name == p.name) {
                        let ty = p.ty.as_ref().map_or(Ty::Any, |t| self.resolve(t));
                        fields.push(FieldDef {
                            name: p.name,
                            ty,
                            has_default: false,
                            annotated: p.ty.is_some(),
                            deprecated: self.ast_deprecation(&p.annotations),
                        });
                    }
                }
            }
            required.retain(|(n, _)| !defined.contains(n));
            let mut seen = Vec::new();
            required.retain(|(n, _)| {
                let new = !seen.contains(n);
                seen.push(*n);
                new
            });
            if !self.types[id].is_abstract
                && let Some((m, by)) = required.first()
            {
                let msg = format!(
                    "`{}` must define `{}` (required by `{by}`)",
                    self.name(t.name),
                    self.name(*m)
                );
                self.error(msg, t.name_span);
            }
            self.types[id].fields = fields;
            self.types[id].methods = methods;
            self.types[id].required = required;
            self.types[id].traits = traits;
            self.types[id].ready = true;
        }
        for s in &b.stmts {
            if let StmtKind::Enum(e) = &s.kind {
                let Some(Ty::EnumValue(id)) = self.lookup(e.name).map(|(v, _)| v.ty.clone()) else {
                    continue;
                };
                let value_ty = match &e.repr {
                    Some(r) => self.resolve(r),
                    None => {
                        let all_named = e.variants.iter().all(|v| v.value.is_none());
                        if all_named { Ty::Int } else { Ty::Any }
                    }
                };
                self.enums[id].value_ty = value_ty;
                self.enums[id].methods = e
                    .methods
                    .iter()
                    .map(|m| {
                        let sig = self.signature(
                            m.params.as_deref().unwrap_or(&[]),
                            m.ret.as_ref(),
                            false,
                        );
                        let dep = self.ast_deprecation(&m.annotations);
                        self.mark_fn(&sig, dep);
                        self.mark_params_ast(&sig, m.params.as_deref());
                        (m.name, sig)
                    })
                    .collect();
            }
        }
        // 3. Functions.
        for s in &b.stmts {
            if let StmtKind::Def(f) = &s.kind {
                let sig = self.signature(f.params.as_deref().unwrap_or(&[]), f.ret.as_ref(), false);
                let dep = self.ast_deprecation(&f.annotations);
                self.mark_fn(&sig, dep);
                self.mark_params_ast(&sig, f.params.as_deref());
                self.declare(f.name, Ty::Fn(sig), VarKind::Func);
            }
        }
        // 4. Upcoming `let`s: visible in functions (called later), but a direct use before
        //    the declaration is an error.
        for s in &b.stmts {
            if let StmtKind::Let(l) = &s.kind {
                let mut names = Vec::new();
                pattern_names(&l.pattern, &mut names);
                let scope = self.scopes.last_mut().unwrap();
                for n in names {
                    scope.vars.entry(n).or_insert(Var {
                        ty: Ty::Any,
                        kind: VarKind::Let,
                        pending: true,
                    });
                }
            }
        }
    }

    /// Type of a simple literal expression (default value of an unannotated field).
    fn literal_ty(&self, e: ExprId) -> Ty {
        match self.ast.kind(e) {
            ExprKind::Int(_) => Ty::Int,
            ExprKind::Float(_) => Ty::Float,
            ExprKind::Str(_) | ExprKind::Interp(_) => Ty::Str,
            ExprKind::Bool(_) => Ty::Bool,
            _ => Ty::Any,
        }
    }

    fn signature(&mut self, params: &[Param], ret: Option<&TypeExpr>, lenient: bool) -> Rc<FnSig> {
        let params = params
            .iter()
            .map(|p| ParamSig {
                name: p.name,
                ty: p.ty.as_ref().map_or(Ty::Any, |t| self.resolve(t)),
                has_default: p.default.is_some(),
            })
            .collect();
        // Unannotated return: `any` until the body is checked.
        let ret = ret.map_or(Ty::Any, |t| self.resolve(t));
        Rc::new(FnSig {
            params,
            ret: RefCell::new(ret),
            lenient,
        })
    }

    fn stmt(&mut self, s: &Stmt) -> Ty {
        match &s.kind {
            StmtKind::Expr(e) => self.expr(*e, None),
            StmtKind::Let(l) => {
                let ty = match &l.ty {
                    Some(t) => {
                        let declared = self.resolve(t);
                        self.expect(l.value, &declared, "");
                        declared
                    }
                    None => self.expr(l.value, None).widen(),
                };
                self.bind_pattern(&l.pattern, ty);
                Ty::Nil
            }
            StmtKind::Def(f) => {
                let sig = match self.lookup(f.name).map(|(v, _)| v.ty.clone()) {
                    Some(Ty::Fn(sig)) => sig,
                    _ => {
                        let sig = self.signature(
                            f.params.as_deref().unwrap_or(&[]),
                            f.ret.as_ref(),
                            false,
                        );
                        let dep = self.ast_deprecation(&f.annotations);
                        self.mark_fn(&sig, dep);
                        self.mark_params_ast(&sig, f.params.as_deref());
                        sig
                    }
                };
                self.function(f, &sig, None);
                Ty::Nil
            }
            // Checked with the declarations (`hoist`).
            StmtKind::Module(_) => Ty::Nil,
            StmtKind::Require(_) => {
                if self.scopes.len() > 1 || self.fns.len() > 1 {
                    self.error("`require` must be at the top level of the script", s.span);
                }
                Ty::Nil
            }
            StmtKind::Type(_) | StmtKind::Enum(_)
                if self.scopes.iter().skip(1).any(|s| s.module.is_none()) || self.fns.len() > 1 =>
            {
                self.error(
                    "`type`, `class`, `annotation` and `enum` declarations must be at the top level of the script",
                    s.span,
                );
                Ty::Nil
            }
            StmtKind::Type(t) => {
                self.type_decl(t);
                Ty::Nil
            }
            StmtKind::Enum(e) => {
                self.enum_decl(e);
                Ty::Nil
            }
        }
    }

    fn bind_pattern(&mut self, p: &Pattern, ty: Ty) {
        match p {
            Pattern::Name(n, span) => {
                if let Some(p) = prim(&ty) {
                    self.prims.vars.insert(span.start, p);
                }
                self.declare(*n, ty, VarKind::Let)
            }
            Pattern::Tuple(items, span) => {
                let elems: Vec<Ty> = match &ty {
                    Ty::Tuple(ts) if ts.len() == items.len() => ts.clone(),
                    Ty::Tuple(ts) => {
                        self.error(
                            format!(
                                "expected {} values, the tuple has {}",
                                items.len(),
                                ts.len()
                            ),
                            *span,
                        );
                        vec![Ty::Any; items.len()]
                    }
                    Ty::List(t) => vec![(**t).clone(); items.len()],
                    _ => vec![Ty::Any; items.len()],
                };
                for (p, t) in items.iter().zip(elems) {
                    self.bind_pattern(p, t);
                }
            }
            Pattern::Record(names, _) => {
                for (n, span) in names {
                    let t = self.field_ty(&ty, *n, *span);
                    self.declare(*n, t, VarKind::Let);
                }
            }
            Pattern::List { items, rest, .. } => {
                let elem = match &ty {
                    Ty::List(t) => (**t).clone(),
                    _ => Ty::Any,
                };
                for p in items {
                    self.bind_pattern(p, elem.clone());
                }
                if let Some((n, _)) = rest {
                    self.declare(*n, Ty::List(Box::new(elem)), VarKind::Let);
                }
            }
        }
    }

    /// Body of a function, a method (`this`) or a lambda.
    fn function(&mut self, f: &FnDecl, sig: &Rc<FnSig>, this: Option<Ty>) {
        self.annotations(&f.annotations);
        for p in f.params.iter().flatten() {
            self.annotations(&p.annotations);
        }
        let ret = f.ret.as_ref().map(|_| sig.ret.borrow().clone());
        let t = self.body(
            f.params.as_deref().unwrap_or(&[]),
            sig,
            &f.body,
            ret.clone(),
            this,
        );
        if ret.is_none() {
            *sig.ret.borrow_mut() = t;
        }
    }

    /// Checks a body; returns its type (last expression and `return`s).
    fn body(
        &mut self,
        params: &[Param],
        sig: &FnSig,
        body: &Block,
        ret: Option<Ty>,
        this: Option<Ty>,
    ) -> Ty {
        let in_method = this.is_some();
        self.fns.push(FnCtx {
            ret: ret.clone(),
            returns: Vec::new(),
            loops: 0,
            this,
        });
        self.push_scope();
        for (p, ps) in params.iter().zip(&sig.params) {
            if let Some(d) = p.default {
                self.expect(d, &ps.ty, "");
            }
            if p.is_field && !in_method {
                self.error(
                    "an `@field` parameter can only be used inside a method",
                    p.span,
                );
            }
            self.declare(p.name, ps.ty.clone(), VarKind::Let);
        }
        let tail = tail_expr(body);
        let last = match (tail, &ret) {
            // Last expression checked against the annotated return type (literal conversion).
            (Some(e), Some(r)) => {
                self.hoist(body);
                let n = body.stmts.len();
                for s in &body.stmts[..n - 1] {
                    self.stmt(s);
                }
                self.expect(e, r, " for the return value")
            }
            _ => self.block_stmts(body),
        };
        self.pop_scope();
        let ctx = self.fns.pop().unwrap();
        match ret {
            Some(r) => {
                if tail.is_none() && !self.assignable(&last, &r) {
                    let msg = format!(
                        "expected return type `{}`, got `{}`",
                        self.show(&r),
                        self.show(&last)
                    );
                    self.error(msg, body.span);
                }
                r
            }
            None => union(ctx.returns.into_iter().chain([last]).collect()),
        }
    }

    fn type_decl(&mut self, t: &TypeDecl) {
        // Not hoisted (declared twice): already reported.
        let Some(&id) = self.decl_ids.get(&(t as *const TypeDecl as usize)) else {
            return;
        };
        // Default values: they can see the other fields.
        self.push_scope();
        let fields: Vec<(Symbol, Ty)> = self.types[id]
            .fields
            .iter()
            .map(|f| (f.name, f.ty.clone()))
            .collect();
        for (n, ty) in &fields {
            self.declare(*n, ty.clone(), VarKind::Field);
        }
        for m in &t.members {
            if let Member::Field(f) = m
                && let Some(d) = f.default
            {
                let ty = fields
                    .iter()
                    .find(|(n, _)| *n == f.name)
                    .map(|(_, t)| t.clone())
                    .unwrap_or(Ty::Any);
                self.expect(d, &ty, &format!(" for field `{}`", self.name(f.name)));
            }
        }
        self.pop_scope();
        for m in &t.members {
            match m {
                Member::Method(f) => {
                    let sig = self.types[id].methods.get(&f.name).cloned().unwrap();
                    let saved = self.method.replace((id, f.name));
                    self.function(f, &sig, Some(Ty::Object(id)));
                    self.method = saved;
                }
                Member::Stmt(s) => self.error(
                    "only fields and methods are allowed in a `type` / `class` \
                     (macros will come later)",
                    s.span,
                ),
                Member::Field(f) => self.annotations(&f.annotations),
                Member::Abstract(f) => {
                    self.annotations(&f.annotations);
                    for p in f.params.iter().flatten() {
                        self.annotations(&p.annotations);
                    }
                }
                Member::Include(..) => {}
            }
        }
        self.annotations(&t.annotations);
    }

    fn enum_decl(&mut self, e: &EnumDecl) {
        let Some(Ty::EnumValue(id)) = self.lookup(e.name).map(|(v, _)| v.ty.clone()) else {
            return;
        };
        let value_ty = self.enums[id].value_ty.clone();
        for v in &e.variants {
            if let Some(x) = v.value {
                self.expect(
                    x,
                    &value_ty,
                    &format!(" for variant `{}`", self.name(v.name)),
                );
            }
            self.annotations(&v.annotations);
        }
        for m in &e.methods {
            // `None`: enum declared twice (reported), the name refers to the first one.
            let Some(sig) = self.enums[id].methods.get(&m.name).cloned() else {
                continue;
            };
            self.function(m, &sig, Some(Ty::Variant(id)));
        }
        self.annotations(&e.annotations);
    }

    /// `@[Name(args)]`: the annotation must be declared (`Deprecated` and `Doc` are, by the
    /// prelude), and its arguments are checked like a construction.
    fn annotations(&mut self, anns: &[AnnotationUse]) {
        for a in anns {
            let [name] = a.path.as_slice() else { continue };
            // Declared in this program, or before (prelude, previous script).
            let found = match self.lookup(*name) {
                Some((v, _)) => Some(v.ty.clone()),
                None => self.global_ty(*name).map(|(t, _)| t),
            };
            match found {
                Some(Ty::TypeValue(id)) if self.types[id].kind == TypeDeclKind::Annotation => {
                    self.construct_args(id, a.args, None, a.span);
                }
                Some(_) => {
                    let msg = format!("`{}` is not an annotation", self.name(*name));
                    self.error(msg, a.span);
                }
                None => {
                    let msg = format!("unknown annotation `{}`", self.name(*name));
                    self.error(msg, a.span);
                }
            }
        }
    }

    // ─── Expressions ─────────────────────────────────────────────────────────

    fn expr(&mut self, e: ExprId, expected: Option<&Ty>) -> Ty {
        stacker::maybe_grow(128 * 1024, 2 * 1024 * 1024, || self.expr_inner(e, expected))
    }

    fn expr_inner(&mut self, e: ExprId, expected: Option<&Ty>) -> Ty {
        let ast = self.ast;
        let span = ast.span(e);
        match ast.kind(e) {
            ExprKind::Int(n) => Ty::LitInt(n),
            ExprKind::Float(_) => Ty::Float,
            ExprKind::Str(s) => Ty::LitStr(self.name(s).into()),
            ExprKind::Interp(parts) => {
                for p in ast.get(parts) {
                    if let StrPart::Expr(x) = *p {
                        self.expr(x, None);
                    }
                }
                Ty::Str
            }
            ExprKind::Bool(_) => Ty::Bool,
            ExprKind::Nil => Ty::Nil,
            ExprKind::SelfRef => match self.fns.last().and_then(|f| f.this.clone()) {
                Some(t) => t,
                None => {
                    self.error("`self` can only be used inside a method", span);
                    Ty::Any
                }
            },
            ExprKind::Ident(name) => self.ident(name, span),
            ExprKind::IVar(name) => self.ivar(name, span),
            ExprKind::Path(p) => self.path(ast.get(p), span),

            ExprKind::List(items) => {
                let elem_expected = match expected {
                    Some(Ty::List(t)) => Some((**t).clone()),
                    _ => None,
                };
                let mut tys = Vec::new();
                for &it in ast.get(items) {
                    tys.push(match &elem_expected {
                        Some(t) => self.expect(it, t, " in the list"),
                        None => self.expr(it, None),
                    });
                }
                let elem = match elem_expected {
                    Some(t) => t,
                    None if tys.is_empty() => Ty::Any,
                    None => union(tys.into_iter().map(Ty::widen).collect()),
                };
                Ty::List(Box::new(elem))
            }
            ExprKind::Tuple(items) => {
                let exp: Option<Vec<Ty>> = match expected {
                    Some(Ty::Tuple(ts)) if ts.len() == items.len as usize => Some(ts.clone()),
                    _ => None,
                };
                let tys = ast
                    .get(items)
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| self.expr(x, exp.as_ref().map(|ts| &ts[i])))
                    .collect();
                Ty::Tuple(tys)
            }
            ExprKind::Hash(entries) => {
                let mut ks = Vec::new();
                let mut vs = Vec::new();
                for &(k, v) in ast.get(entries) {
                    let kt = self.expr(k, None);
                    let hashable = kt.members().iter().any(|m| {
                        matches!(
                            m,
                            Ty::Any
                                | Ty::Nil
                                | Ty::Bool
                                | Ty::Int
                                | Ty::LitInt(_)
                                | Ty::Str
                                | Ty::LitStr(_)
                        )
                    });
                    if !hashable {
                        let msg = format!(
                            "a hash key must be nil, bool, int or str, got `{}`",
                            self.show_v(&kt)
                        );
                        self.error(msg, ast.span(k));
                    }
                    ks.push(kt.widen());
                    vs.push(self.expr(v, None).widen());
                }
                if ks.is_empty() {
                    return Ty::Hash(Box::new(Ty::Any), Box::new(Ty::Any));
                }
                Ty::Hash(Box::new(union(ks)), Box::new(union(vs)))
            }
            ExprKind::Record(fields) => {
                // Record literal expected as a `type`: it will be converted.
                if let Some(id) = expected.and_then(|t| self.data_type_in(t)) {
                    self.record_as_type(id, fields, span);
                    return Ty::Object(id);
                }
                let exp = match expected {
                    Some(Ty::Record(fs)) => Some(fs.clone()),
                    _ => None,
                };
                let fs = ast
                    .get(fields)
                    .iter()
                    .map(|&(n, v)| {
                        let t = exp
                            .as_ref()
                            .and_then(|fs| fs.iter().find(|(m, _)| *m == n))
                            .map(|(_, t)| t.clone());
                        (n, self.expr(v, t.as_ref()))
                    })
                    .collect();
                Ty::Record(fs)
            }

            ExprKind::Unary(op, x) => {
                let t = self.expr(x, None);
                match op {
                    UnaryOp::Not => Ty::Bool,
                    UnaryOp::Neg => {
                        let ok = t
                            .members()
                            .iter()
                            .any(|m| matches!(m, Ty::Any | Ty::Int | Ty::LitInt(_) | Ty::Float));
                        if !ok {
                            let msg = format!("`-` expects a number, got `{}`", self.show_v(&t));
                            self.error(msg, span);
                        }
                        match t {
                            Ty::LitInt(_) => Ty::Int,
                            t => t,
                        }
                    }
                }
            }
            ExprKind::Binary(op, a, b) => {
                let l = self.expr(a, None);
                let r = self.expr(b, None);
                self.binary(op, &l, &r, span)
            }
            ExprKind::Range { lo, hi, .. } => {
                self.expect(lo, &Ty::Int, " for a range");
                self.expect(hi, &Ty::Int, " for a range");
                Ty::Range
            }
            ExprKind::Assign { target, op, value } => self.assign(target, op, value),

            ExprKind::Call {
                callee,
                args,
                block,
            } => {
                let f = self.expr(callee, None);
                // Call to a named function: its name in the messages.
                let label = match ast.kind(callee) {
                    ExprKind::Ident(n) => Some(format!("`{}`", self.name(n))),
                    _ => None,
                };
                self.call(&f, args, block, span, label.as_deref())
            }
            ExprKind::MethodCall {
                recv,
                name,
                args,
                block,
            } => {
                if let Some(m) = self.module_of(recv) {
                    let f = self.member(m, name, span);
                    let label = format!("`{}.{}`", self.name(m), self.name(name));
                    return self.call(&f, args, block, span, Some(&label));
                }
                let r = self.expr(recv, None);
                self.method_call(&r, name, args, block, span)
            }
            ExprKind::Field(recv, name) if self.module_of(recv).is_some() => {
                let m = self.module_of(recv).unwrap();
                self.member(m, name, span)
            }
            ExprKind::Field(recv, name) => {
                let r = self.expr(recv, None);
                if let Ty::Object(id) = r {
                    let def = &self.types[id];
                    if def.kind == TypeDeclKind::Type
                        && let Some(f) = def.fields.iter().find(|f| f.name == name)
                        && f.annotated
                        && let Some(p) = prim(&f.ty)
                    {
                        self.prims.fields.insert(e.0, p);
                    }
                }
                self.field_ty(&r, name, span)
            }
            ExprKind::TupleIndex(recv, idx) => {
                let r = self.expr(recv, None);
                match r {
                    Ty::Tuple(ts) => match ts.get(idx as usize) {
                        Some(t) => t.clone(),
                        None => {
                            self.error(
                                format!("index {idx} out of tuple bounds (length {})", ts.len()),
                                span,
                            );
                            Ty::Any
                        }
                    },
                    Ty::Any => Ty::Any,
                    r => {
                        let msg =
                            format!("`.{idx}` applies to a tuple, got `{}`", self.show_v(&r));
                        self.error(msg, span);
                        Ty::Any
                    }
                }
            }
            ExprKind::Index(recv, idx) => {
                let r = self.expr(recv, None);
                let i = self.expr(idx, None);
                self.index(&r, &i, span, ast.span(idx))
            }
            ExprKind::IndexOpt(recv, idx) => {
                let r = self.expr(recv, None);
                let i = self.expr(idx, None);
                match (&r, &i) {
                    // Literal index out of a tuple: `nil`, not an error.
                    (Ty::Tuple(ts), Ty::LitInt(n))
                        if !(-(ts.len() as i64)..ts.len() as i64).contains(n) =>
                    {
                        Ty::Nil
                    }
                    _ => {
                        let t = self.index(&r, &i, span, ast.span(idx));
                        Ty::optional(t)
                    }
                }
            }
            ExprKind::Lambda(id) => self.lambda(id, expected),

            ExprKind::If(c) => {
                let x = ast.if_(c);
                self.expr(x.cond, None);
                let t = self.block(&x.then);
                let f = match &x.else_ {
                    Some(b) => self.block(b),
                    None => Ty::Nil,
                };
                union(vec![t, f])
            }
            ExprKind::Case(c) => {
                let x = ast.case(c);
                let subject = self.expr(x.subject, None);
                let mut arms = Vec::new();
                for arm in &x.arms {
                    for &p in ast.get(arm.patterns) {
                        let pt = self.expr(p, None);
                        // A pattern that can never equal the subject (widened types:
                        // `case 8080 when 80` is not reported).
                        let (a, b) = (pt.clone().widen(), subject.clone().widen());
                        let possible = matches!(pt, Ty::Range)
                            || self.assignable(&a, &b)
                            || self.assignable(&b, &a);
                        if !possible {
                            let msg = format!(
                                "this pattern (`{}`) cannot match a value of type `{}`",
                                self.show_v(&pt),
                                self.show_v(&subject)
                            );
                            self.error(msg, ast.span(p));
                        }
                    }
                    arms.push(self.block(&arm.body));
                }
                arms.push(match &x.else_ {
                    Some(b) => self.block(b),
                    None => Ty::Nil,
                });
                union(arms)
            }
            ExprKind::While(c) => {
                let x = ast.while_(c);
                self.expr(x.cond, None);
                self.fns.last_mut().unwrap().loops += 1;
                self.block(&x.body);
                self.fns.last_mut().unwrap().loops -= 1;
                Ty::Nil
            }
            ExprKind::For(c) => {
                let x = ast.for_(c);
                let it = self.expr(x.iter, None);
                let vars = self.for_vars(&it, x.vars.len(), ast.span(x.iter));
                self.push_scope();
                for ((n, vspan), t) in x.vars.iter().zip(vars) {
                    if let Some(p) = prim(&t) {
                        self.prims.vars.insert(vspan.start, p);
                    }
                    self.declare(*n, t, VarKind::Let);
                }
                self.fns.last_mut().unwrap().loops += 1;
                self.block_stmts(&x.body);
                self.fns.last_mut().unwrap().loops -= 1;
                self.pop_scope();
                Ty::Nil
            }
            ExprKind::Begin(c) => {
                let x = ast.begin(c);
                let mut types = vec![self.block(&x.body)];
                for r in &x.rescues {
                    let t = r.ty.as_ref().map_or(Ty::Any, |t| self.resolve(t));
                    self.push_scope();
                    if let Some((n, _)) = r.var {
                        self.declare(n, t, VarKind::Let);
                    }
                    types.push(self.block_stmts(&r.body));
                    self.pop_scope();
                }
                if let Some(b) = &x.else_ {
                    // The body's value is replaced by the `else`'s.
                    types[0] = self.block(b);
                }
                if let Some(b) = &x.ensure {
                    self.block(b);
                }
                union(types)
            }
            ExprKind::Super(args) => {
                let target = self.method.and_then(|(id, m)| Some((self.types[id].parent?, m)));
                let Some((parent, m)) = target else {
                    self.error(
                        "`super` can only be used in a method of a class that inherits",
                        span,
                    );
                    for a in ast.get(args) {
                        self.expr(a.value, None);
                    }
                    return Ty::Any;
                };
                match self.types[parent].methods.get(&m).cloned() {
                    Some(sig) => {
                        let what = format!("`super` (`{}.{}`)", self.name(self.types[parent].name), self.name(m));
                        self.check_args(&sig, args, None, span, &what);
                        sig.ret.borrow().clone()
                    }
                    None => {
                        let msg = format!(
                            "`super`: `{}` has no method `{}`",
                            self.name(self.types[parent].name),
                            self.name(m)
                        );
                        self.error(msg, span);
                        Ty::Any
                    }
                }
            }
            ExprKind::Raise(v) => {
                let t = self.expr(v, None);
                let ok = |t: &Ty| {
                    matches!(
                        t,
                        Ty::Any | Ty::Never | Ty::Str | Ty::LitStr(_) | Ty::Object(_)
                    )
                };
                let fits = match &t {
                    Ty::Union(items) => items.iter().all(ok),
                    t => ok(t),
                };
                if !fits {
                    let msg = format!(
                        "`raise` expects a string or an object, got `{}`",
                        self.show_v(&t)
                    );
                    self.error(msg, ast.span(v));
                }
                Ty::Never
            }
            ExprKind::Return(v) => {
                let expected = self.fns.last().and_then(|f| f.ret.clone());
                let t = match (v, &expected) {
                    (Some(v), Some(r)) => self.expect(v, r, " for the return value"),
                    (Some(v), None) => self.expr(v, None),
                    (None, Some(r)) => {
                        if !self.assignable(&Ty::Nil, r) {
                            let msg = format!("`return` without a value, expected `{}`", self.show(r));
                            self.error(msg, span);
                        }
                        Ty::Nil
                    }
                    (None, None) => Ty::Nil,
                };
                if let Some(f) = self.fns.last_mut() {
                    f.returns.push(t);
                }
                Ty::Never
            }
            ExprKind::Break | ExprKind::Next => {
                if self.fns.last().is_some_and(|f| f.loops == 0) {
                    let kw = if matches!(ast.kind(e), ExprKind::Break) {
                        "break"
                    } else {
                        "next"
                    };
                    self.error(format!("`{kw}` outside of a loop"), span);
                }
                Ty::Never
            }
        }
    }

    fn ident(&mut self, name: Symbol, span: Span) -> Ty {
        match self.lookup(name) {
            Some((v, level)) => {
                if v.pending && level == self.fn_level() {
                    let msg = format!("`{}` is used before its declaration", self.name(name));
                    self.error(msg, span);
                    return Ty::Any;
                }
                let ty = v.ty.clone();
                if let Ty::Module(m) = ty {
                    let m = self.name(m).to_string();
                    self.error(format!("`{m}` is a module: use `{m}::Name` or `{m}.f(…)`"), span);
                    return Ty::Any;
                }
                self.warn_value(&ty, name, span);
                ty
            }
            None => {
                // Member of an enclosing module declared by an earlier script.
                for m in self.modules() {
                    if let Some(t) = self.member_ty(m, name).filter(|_| self.qualified(m, name).is_some())
                    {
                        return t;
                    }
                }
                if let Some((t, _)) = self.global_ty(name) {
                    if let Ty::Module(m) = t {
                        let m = self.name(m).to_string();
                        self.error(format!("`{m}` is a module: use `{m}::Name` or `{m}.f(…)`"), span);
                        return Ty::Any;
                    }
                    self.warn_value(&t, name, span);
                    return t;
                }
                let hint = suggest(self.name(name), self.known_names());
                let msg = format!("`{}` is not defined{hint}", self.name(name));
                self.error(msg, span);
                Ty::Any
            }
        }
    }

    /// Type of the current method's `self` object, if it has fields.
    fn this_type(&self) -> Option<TypeId> {
        match self.fns.last()?.this {
            Some(Ty::Object(id)) => Some(id),
            _ => None,
        }
    }

    fn ivar(&mut self, name: Symbol, span: Span) -> Ty {
        let Some(this) = self.fns.last().and_then(|f| f.this.clone()) else {
            self.error(
                "`self` and `@field` can only be used inside a method",
                span,
            );
            return Ty::Any;
        };
        let Ty::Object(id) = this else {
            let msg = format!("`@{}`: `self` has no fields here", self.name(name));
            self.error(msg, span);
            return Ty::Any;
        };
        match self.types[id].fields.iter().find(|f| f.name == name) {
            Some(f) => f.ty.clone(),
            None => {
                let names: Vec<&str> = self.types[id]
                    .fields
                    .iter()
                    .map(|f| self.name(f.name))
                    .collect();
                let hint = suggest(self.name(name), names);
                let msg = format!(
                    "`{}` has no field `@{}`{hint}",
                    self.name(self.types[id].name),
                    self.name(name)
                );
                self.error(msg, span);
                Ty::Any
            }
        }
    }

    fn path(&mut self, path: &[Symbol], span: Span) -> Ty {
        let (&last, prefix) = path.split_last().unwrap();
        // Member of a module: `Http::Request`.
        if let Some(m) = self.module_path(prefix) {
            return self.member(m, last, span);
        }
        if self.module_path(path).is_some() {
            self.error("a module is not a value", span);
            return Ty::Any;
        }
        // `Enum::Variant`, the enum possibly in a module (`Http::Status::Ok`).
        let (head, variant) = (&prefix[prefix.len() - 1], &last);
        let head_ty = match prefix {
            [h] => match self.lookup(*h).map(|(v, _)| v.ty.clone()) {
                Some(t) => Some(t),
                None => self.global_ty(*h).map(|g| g.0),
            },
            _ => match self.module_path(&prefix[..prefix.len() - 1]) {
                Some(m) => self.member_ty(m, *head),
                None => {
                    self.error("unknown module in this path", span);
                    return Ty::Any;
                }
            },
        };
        match head_ty {
            Some(Ty::EnumValue(id)) => {
                if let Some(k) = self.enums[id].variants.iter().position(|v| v == variant) {
                    let dep = self.enums[id].variants_deprecated.get(k).cloned().flatten();
                    let what = format!("`{}::{}`", self.name(*head), self.name(*variant));
                    self.warn_deprecated(dep.as_ref(), &what, span);
                } else {
                    let names: Vec<&str> = self.enums[id]
                        .variants
                        .iter()
                        .map(|v| self.name(*v))
                        .collect();
                    let hint = suggest(self.name(*variant), names);
                    let msg = format!(
                        "`{}` has no variant `{}`{hint}",
                        self.name(*head),
                        self.name(*variant)
                    );
                    self.error(msg, span);
                }
                Ty::Variant(id)
            }
            Some(Ty::Any) => Ty::Any,
            Some(t) => {
                let msg = format!(
                    "`{}` is not an enum (`{}`)",
                    self.name(*head),
                    self.show(&t)
                );
                self.error(msg, span);
                Ty::Any
            }
            None => {
                let msg = format!("`{}` is not defined", self.name(*head));
                self.error(msg, span);
                Ty::Any
            }
        }
    }

    fn binary(&mut self, op: BinOp, l: &Ty, r: &Ty, span: Span) -> Ty {
        use BinOp::*;
        match op {
            Eq | Ne => return Ty::Bool,
            And | Or => return union(vec![l.clone(), r.clone()]),
            _ => {}
        }
        // Result for each possible combination; error if none fits.
        let mut results = Vec::new();
        for a in l.members() {
            for b in r.members() {
                if let Some(t) = binary_result(op, a, b) {
                    results.push(t);
                }
            }
        }
        if results.is_empty() {
            let msg = match (
                op,
                l.members().iter().any(is_str),
                r.members().iter().any(is_str),
            ) {
                (Add, true, false) | (Add, false, true) => format!(
                    "`+` between `{}` and `{}`: use interpolation \"#{{x}}\" or `.to_s()`",
                    self.show_v(l),
                    self.show_v(r)
                ),
                (Lt | Le | Gt | Ge, ..) => format!(
                    "`{}` cannot compare `{}` and `{}`",
                    op.as_str(),
                    self.show_v(l),
                    self.show_v(r)
                ),
                _ => format!(
                    "`{}` does not apply to `{}` and `{}`",
                    op.as_str(),
                    self.show_v(l),
                    self.show_v(r)
                ),
            };
            self.error(msg, span);
            return Ty::Any;
        }
        if op == Push
            && let (Ty::List(elem), false) = (l, r.is_any())
            && !self.assignable(r, elem)
        {
            let msg = format!(
                "`<<` appends `{}` to a list of `{}`",
                self.show_v(r),
                self.show_v(elem)
            );
            self.error(msg, span);
        }
        union(results)
    }

    fn assign(&mut self, target: ExprId, op: Option<BinOp>, value: ExprId) -> Ty {
        let ast = self.ast;
        let tspan = ast.span(target);
        // Type expected by the target.
        let target_ty: Option<Ty> = match ast.kind(target) {
            ExprKind::Ident(name) => match self.lookup(name).map(|(v, l)| (v.clone(), l)) {
                Some((v, level)) => {
                    if v.pending && level == self.fn_level() {
                        let msg = format!("`{}` is used before its declaration", self.name(name));
                        self.error(msg, tspan);
                    }
                    match v.kind {
                        VarKind::Const | VarKind::Func | VarKind::Type => {
                            let msg = format!(
                                "`{}` cannot be modified (constant, function or type)",
                                self.name(name)
                            );
                            self.error(msg, tspan);
                            None
                        }
                        _ => Some(v.ty),
                    }
                }
                None if self.globals.get(name).is_some() => {
                    let (t, mutable) = self.global_ty(name).unwrap();
                    if !mutable {
                        let msg = format!(
                            "`{}` cannot be modified (constant, function or type)",
                            self.name(name)
                        );
                        self.error(msg, tspan);
                    }
                    Some(t)
                }
                None => {
                    let n = self.name(name).to_string();
                    if op.is_none() && n.starts_with(|c: char| c.is_ascii_uppercase()) {
                        // `NAME = value`: constant.
                        let t = self.expr(value, None).widen();
                        self.declare(name, t.clone(), VarKind::Const);
                        return t;
                    }
                    let construct = self.scopes.iter().rev().find_map(|s| s.construct);
                    // Block of an unknown type (already reported): no cascading error.
                    if construct == Some(UNKNOWN_TYPE) {
                        let t = self.expr(value, None).widen();
                        self.declare(name, t.clone(), VarKind::Field);
                        return t;
                    }
                    let msg = match construct {
                        Some(id) => {
                            let fields: Vec<&str> = self.types[id]
                                .fields
                                .iter()
                                .map(|f| self.name(f.name))
                                .collect();
                            let hint = suggest(&n, fields);
                            format!(
                                "`{n}` is neither a field of `{}` nor a declared variable{hint} \
                                 (use `let {n} = …` for a local variable)",
                                self.name(self.types[id].name)
                            )
                        }
                        None => {
                            let hint = suggest(&n, self.known_names());
                            format!("`{n}` is not declared: use `let {n} = …`{hint}")
                        }
                    };
                    self.error(msg, tspan);
                    None
                }
            },
            ExprKind::IVar(name) => Some(self.ivar(name, tspan)),
            ExprKind::Field(recv, name) => {
                let r = self.expr(recv, None);
                Some(self.field_target(&r, name, tspan))
            }
            ExprKind::Index(recv, idx) => {
                let r = self.expr(recv, None);
                let i = self.expr(idx, None);
                Some(match &r {
                    Ty::Hash(_, v) => (**v).clone(),
                    _ => self.index(&r, &i, tspan, ast.span(idx)),
                })
            }
            _ => None,
        };
        let target_ty = target_ty.unwrap_or(Ty::Any);
        match op {
            None => self.expect(value, &target_ty, ""),
            Some(op) => {
                let v = self.expr(value, None);
                let result = self.binary(op, &target_ty, &v, ast.span(value).to(tspan));
                if !self.assignable(&result, &target_ty) {
                    let msg = format!(
                        "expected type `{}`, got `{}`",
                        self.show(&target_ty),
                        self.show(&result)
                    );
                    self.error(msg, tspan);
                }
                result
            }
        }
    }

    // ─── Calls ───────────────────────────────────────────────────────────────

    /// Call of `f`. `this`: receiver of a declared method (for messages).
    fn call(
        &mut self,
        f: &Ty,
        args: Slice<Arg>,
        block: Option<LambdaId>,
        span: Span,
        method: Option<&str>,
    ) -> Ty {
        match f {
            Ty::Fn(sig) => {
                let sig = sig.clone();
                self.check_args(&sig, args, block, span, method.unwrap_or("the function"));
                sig.ret.borrow().clone()
            }
            Ty::TypeValue(id) => {
                let id = *id;
                if self.types[id].is_abstract {
                    let what = if self.types[id].kind == TypeDeclKind::Trait {
                        "a trait"
                    } else {
                        "an abstract class"
                    };
                    let msg = format!(
                        "`{}` is {what}: it cannot be instantiated",
                        self.name(self.types[id].name)
                    );
                    self.error(msg, span);
                    return Ty::Any;
                }
                if self.types[id].kind == TypeDeclKind::Class {
                    let init = self
                        .i
                        .get("init")
                        .and_then(|s| self.types[id].methods.get(&s).cloned());
                    match init {
                        Some(sig) => {
                            let name = format!("`{}`", self.name(self.types[id].name));
                            self.check_args(&sig, args, block, span, &name);
                        }
                        None => self.construct_args(id, args, block, span),
                    }
                } else {
                    self.construct_args(id, args, block, span);
                }
                Ty::Object(id)
            }
            Ty::Any => {
                // Possibly the construction of an unknown type: block checked leniently.
                for a in self.ast.get(args) {
                    self.expr(a.value, None);
                }
                if let Some(b) = block {
                    let l = self.ast.lambda(b);
                    if l.params.is_empty() {
                        self.push_scope();
                        self.scopes.last_mut().unwrap().construct = Some(UNKNOWN_TYPE);
                        self.block_stmts(&l.body);
                        self.pop_scope();
                    } else {
                        self.lambda(b, None);
                    }
                }
                Ty::Any
            }
            Ty::Union(_) => {
                self.args_any(args, block);
                Ty::Any
            }
            t => {
                self.args_any(args, block);
                let msg = format!("`{}` is not a function", self.show_v(t));
                self.error(msg, span);
                Ty::Any
            }
        }
    }

    fn args_any(&mut self, args: Slice<Arg>, block: Option<LambdaId>) {
        for a in self.ast.get(args) {
            self.expr(a.value, None);
        }
        if let Some(b) = block {
            self.lambda(b, None);
        }
    }

    fn check_args(
        &mut self,
        sig: &FnSig,
        args: Slice<Arg>,
        block: Option<LambdaId>,
        span: Span,
        what: &str,
    ) {
        let ast = self.ast;
        let args = ast.get(args);
        let positional =
            args.iter().filter(|a| a.name.is_none()).count() + block.is_some() as usize;
        if !sig.lenient && positional > sig.params.len() {
            self.error(
                format!(
                    "{what} expects {} argument{}, got {positional}",
                    sig.params.len(),
                    if sig.params.len() != 1 { "s" } else { "" }
                ),
                span,
            );
        }
        let mut given = vec![false; sig.params.len()];
        let mut pos_i = 0;
        for a in args {
            let param = match a.name {
                None => {
                    let i = pos_i;
                    pos_i += 1;
                    sig.params.get(i).map(|p| (i, p))
                }
                Some(n) => match sig.params.iter().position(|p| p.name == n) {
                    Some(i) => Some((i, &sig.params[i])),
                    None => {
                        let names: Vec<&str> =
                            sig.params.iter().map(|p| self.name(p.name)).collect();
                        let hint = suggest(self.name(n), names);
                        let msg = format!("{what} has no parameter `{}`{hint}", self.name(n));
                        self.error(msg, ast.span(a.value));
                        None
                    }
                },
            };
            match param {
                Some((i, p)) => {
                    if given[i] {
                        let msg = format!("argument `{}` given twice", self.name(p.name));
                        self.error(msg, ast.span(a.value));
                    }
                    let key = (sig as *const FnSig as usize, i);
                    if let Some(dep) = self.deprecated_params.get(&key).cloned() {
                        let param = format!("parameter `{}` of {what}", self.name(p.name));
                        self.warn_deprecated(Some(&dep), &param, ast.span(a.value));
                    }
                    given[i] = true;
                    let label = format!(" for `{}`", self.name(p.name));
                    self.expect(a.value, &p.ty.clone(), &label);
                }
                None => {
                    self.expr(a.value, None);
                }
            }
        }
        if let Some(b) = block {
            let i = pos_i;
            let expected = sig.params.get(i).map(|p| p.ty.clone());
            if i < given.len() {
                given[i] = true;
            }
            self.lambda(b, expected.as_ref());
        }
        if !sig.lenient {
            for (p, g) in sig.params.iter().zip(&given) {
                if !g && !p.has_default {
                    let msg = format!("missing argument `{}` for {what}", self.name(p.name));
                    self.error(msg, span);
                }
            }
        }
    }

    /// Construction of a `type` (or a `class` without `init`): arguments and block.
    fn construct_args(
        &mut self,
        id: TypeId,
        args: Slice<Arg>,
        block: Option<LambdaId>,
        span: Span,
    ) {
        let ast = self.ast;
        let tname = self.name(self.types[id].name).to_string();
        let nfields = self.types[id].fields.len();
        let mut given = vec![false; nfields];
        let mut pos_i = 0;
        for a in ast.get(args) {
            let idx = match a.name {
                None => {
                    let i = pos_i;
                    pos_i += 1;
                    if i >= nfields {
                        self.error(
                            format!(
                                "`{tname}` has {nfields} field{}, too many arguments",
                                if nfields != 1 { "s" } else { "" }
                            ),
                            ast.span(a.value),
                        );
                        None
                    } else {
                        Some(i)
                    }
                }
                Some(n) => {
                    let idx = self.types[id].fields.iter().position(|f| f.name == n);
                    if idx.is_none() {
                        let names: Vec<&str> = self.types[id]
                            .fields
                            .iter()
                            .map(|f| self.name(f.name))
                            .collect();
                        let hint = suggest(self.name(n), names);
                        let msg = format!("`{tname}` has no field `{}`{hint}", self.name(n));
                        self.error(msg, ast.span(a.value));
                    }
                    idx
                }
            };
            match idx {
                Some(i) => {
                    given[i] = true;
                    let (fname, fty, dep) = {
                        let f = &self.types[id].fields[i];
                        (f.name, f.ty.clone(), f.deprecated.clone())
                    };
                    let what = format!("`{tname}.{}`", self.name(fname));
                    self.warn_deprecated(dep.as_ref(), &what, ast.span(a.value));
                    let label = format!(" for field `{}`", self.name(fname));
                    self.expect(a.value, &fty, &label);
                }
                None => {
                    self.expr(a.value, None);
                }
            }
        }
        match block {
            Some(b) => {
                let l = ast.lambda(b);
                if let Some(p) = l.params.first() {
                    self.error(
                        "a construction block takes no parameters: the fields are \
                         directly accessible",
                        p.span,
                    );
                }
                self.push_scope();
                self.scopes.last_mut().unwrap().construct = Some(id);
                let fields: Vec<(Symbol, Ty)> = self.types[id]
                    .fields
                    .iter()
                    .map(|f| (f.name, f.ty.clone()))
                    .collect();
                for (n, t) in fields {
                    self.declare(n, t, VarKind::Field);
                }
                self.block_stmts(&l.body);
                self.pop_scope();
            }
            // Without a block, required fields must be given.
            None => {
                let missing: Vec<Symbol> = self.types[id]
                    .fields
                    .iter()
                    .zip(&given)
                    .filter(|(f, g)| !**g && !f.has_default)
                    .map(|(f, _)| f.name)
                    .collect();
                for name in missing {
                    let msg = format!(
                        "field `{}` is required to construct `{tname}`",
                        self.name(name)
                    );
                    self.error(msg, span);
                }
            }
        }
    }

    /// A `type` (not a `class`) among the possible types of `t`.
    fn data_type_in(&self, t: &Ty) -> Option<TypeId> {
        t.members().iter().find_map(|m| match m {
            Ty::Object(id) if self.types[*id].kind != TypeDeclKind::Class => Some(*id),
            _ => None,
        })
    }

    /// Record literal converted to `type` `id`: known fields, types, required ones.
    fn record_as_type(&mut self, id: TypeId, fields: Slice<(Symbol, ExprId)>, span: Span) {
        let ast = self.ast;
        let tname = self.name(self.types[id].name).to_string();
        let mut given = Vec::new();
        for &(n, v) in ast.get(fields) {
            match self.types[id]
                .fields
                .iter()
                .find(|f| f.name == n)
                .map(|f| f.ty.clone())
            {
                Some(fty) => {
                    given.push(n);
                    let label = format!(" for field `{}`", self.name(n));
                    self.expect(v, &fty, &label);
                }
                None => {
                    let names: Vec<&str> = self.types[id]
                        .fields
                        .iter()
                        .map(|f| self.name(f.name))
                        .collect();
                    let hint = suggest(self.name(n), names);
                    let msg = format!("`{tname}` has no field `{}`{hint}", self.name(n));
                    self.error(msg, ast.span(v));
                }
            }
        }
        let missing: Vec<Symbol> = self.types[id]
            .fields
            .iter()
            .filter(|f| !f.has_default && !given.contains(&f.name))
            .map(|f| f.name)
            .collect();
        for name in missing {
            let msg = format!(
                "field `{}` is required to construct `{tname}`",
                self.name(name)
            );
            self.error(msg, span);
        }
    }

    /// Lambda, with the expected parameter types if known.
    fn lambda(&mut self, id: LambdaId, expected: Option<&Ty>) -> Ty {
        let l = self.ast.lambda(id);
        let exp_params: Vec<Ty> = match expected {
            Some(Ty::Fn(sig)) => sig.params.iter().map(|p| p.ty.clone()).collect(),
            _ => Vec::new(),
        };
        let params = l
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| ParamSig {
                name: p.name,
                ty: match &p.ty {
                    Some(t) => self.resolve(t),
                    None => exp_params.get(i).cloned().unwrap_or(Ty::Any),
                },
                has_default: p.default.is_some(),
            })
            .collect();
        let sig = FnSig {
            params,
            ret: RefCell::new(Ty::Any),
            lenient: true,
        };
        let this = self.fns.last().and_then(|f| f.this.clone());
        let ret = self.body(&l.params, &sig, &l.body, None, this);
        *sig.ret.borrow_mut() = ret;
        Ty::Fn(Rc::new(sig))
    }

    /// Signature of a lambda expected by a built-in method: parameters `params`.
    fn fn_of(params: Vec<Ty>) -> Ty {
        Ty::Fn(Rc::new(FnSig {
            params: params
                .into_iter()
                .enumerate()
                .map(|(i, ty)| ParamSig {
                    name: Symbol(u32::MAX - i as u32),
                    ty,
                    has_default: false,
                })
                .collect(),
            ret: RefCell::new(Ty::Any),
            lenient: true,
        }))
    }

    fn fn_ret(t: &Ty) -> Ty {
        match t {
            Ty::Fn(sig) => sig.ret.borrow().clone(),
            _ => Ty::Any,
        }
    }

    // ─── Fields and methods ──────────────────────────────────────────────────

    fn field_ty(&mut self, r: &Ty, name: Symbol, span: Span) -> Ty {
        let n = self.name(name).to_string();
        match r {
            Ty::Any | Ty::Union(_) => Ty::Any,
            Ty::Record(fs) => match fs.iter().find(|(m, _)| *m == name) {
                Some((_, t)) => t.clone(),
                None => {
                    let names: Vec<&str> = fs.iter().map(|(m, _)| self.name(*m)).collect();
                    let hint = suggest(&n, names);
                    self.error(format!("the record has no field `{n}`{hint}"), span);
                    Ty::Any
                }
            },
            Ty::Object(id) => {
                let id = *id;
                let def = &self.types[id];
                let field = def
                    .fields
                    .iter()
                    .find(|f| f.name == name)
                    .map(|f| f.ty.clone());
                let method = def.methods.get(&name).cloned();
                let is_class = def.kind == TypeDeclKind::Class;
                let inside = self.this_type() == Some(id);
                let dep = def
                    .fields
                    .iter()
                    .find(|f| f.name == name)
                    .and_then(|f| f.deprecated.clone());
                let full = format!("`{}.{n}`", self.name(def.name));
                self.warn_deprecated(dep.as_ref(), &full, span);
                let def = &self.types[id];
                match (field, method) {
                    (Some(t), _) if !is_class || inside => t,
                    (_, Some(sig)) if sig.params.iter().all(|p| p.has_default) => {
                        sig.ret.borrow().clone()
                    }
                    (Some(_), _) => {
                        let msg = format!(
                            "`{n}` is a private field of `{}`: define `def {n} = @{n}` to read it",
                            self.name(def.name)
                        );
                        self.error(msg, span);
                        Ty::Any
                    }
                    (None, _) => {
                        let names: Vec<&str> = def
                            .fields
                            .iter()
                            .map(|f| self.name(f.name))
                            .chain(def.methods.keys().map(|m| self.name(*m)))
                            .collect();
                        let hint = suggest(&n, names);
                        let msg = format!("`{}` has no field `{n}`{hint}", self.name(def.name));
                        self.error(msg, span);
                        Ty::Any
                    }
                }
            }
            Ty::Variant(id) => match n.as_str() {
                "value" => self.enums[*id].value_ty.clone(),
                "name" => Ty::Str,
                _ => match self.enums[*id].methods.get(&name) {
                    Some(sig) => sig.ret.borrow().clone(),
                    None => {
                        let msg = format!(
                            "`{}` has no field `{n}` (`.value`, `.name`)",
                            self.name(self.enums[*id].name)
                        );
                        self.error(msg, span);
                        Ty::Any
                    }
                },
            },
            t => {
                let has_method = self.builtin_methods(t).contains(&n.as_str());
                let msg = if has_method {
                    format!("`{n}` is a method: write `.{n}()`")
                } else {
                    format!("`{}` has no field `{n}`", self.show_v(t))
                };
                self.error(msg, span);
                Ty::Any
            }
        }
    }

    /// Type expected by `recv.name = value`.
    fn field_target(&mut self, r: &Ty, name: Symbol, span: Span) -> Ty {
        let n = self.name(name).to_string();
        if let Ty::Object(id) = r {
            let def = &self.types[*id];
            let setter = self
                .i
                .get(&format!("{n}="))
                .and_then(|s| def.methods.get(&s).cloned());
            if let Some(sig) = setter {
                return sig.params.first().map_or(Ty::Any, |p| p.ty.clone());
            }
            if def.kind == TypeDeclKind::Class
                && self.this_type() != Some(*id)
                && def.fields.iter().any(|f| f.name == name)
            {
                {
                    let msg = format!(
                        "`{n}` is a private field of `{}`: define `def {n}=(value)` to modify it",
                        self.name(def.name)
                    );
                    self.error(msg, span);
                    return Ty::Any;
                }
            }
        }
        self.field_ty(r, name, span)
    }

    fn builtin_methods(&self, t: &Ty) -> &'static [&'static str] {
        match t {
            Ty::Str | Ty::LitStr(_) => STR_METHODS,
            Ty::Int | Ty::LitInt(_) | Ty::Float => NUM_METHODS,
            Ty::List(_) => LIST_METHODS,
            Ty::Range => RANGE_METHODS,
            Ty::Hash(..) => HASH_METHODS,
            _ => &["to_s", "inspect"],
        }
    }

    fn method_call(
        &mut self,
        r: &Ty,
        name: Symbol,
        args: Slice<Arg>,
        block: Option<LambdaId>,
        span: Span,
    ) -> Ty {
        let n = self.name(name).to_string();
        // Declared methods, function fields, enums.
        match r {
            Ty::Any | Ty::Union(_) | Ty::Never => {
                self.args_any(args, block);
                return Ty::Any;
            }
            Ty::Object(id) => {
                let id = *id;
                if let Some(sig) = self.types[id].methods.get(&name).cloned() {
                    let what = format!("`{n}`");
                    let dep = self
                        .deprecated_fns
                        .get(&(Rc::as_ptr(&sig) as usize))
                        .cloned();
                    let full = format!("`{}.{n}`", self.name(self.types[id].name));
                    self.warn_deprecated(dep.as_ref(), &full, span);
                    self.check_args(&sig, args, block, span, &what);
                    let ret = sig.ret.borrow().clone();
                    return ret;
                }
                if self.types[id].kind != TypeDeclKind::Class
                    && let Some(f) = self.types[id].fields.iter().find(|f| f.name == name)
                {
                    let ft = f.ty.clone();
                    return self.call(&ft, args, block, span, None);
                }
            }
            Ty::Record(fs) => {
                if let Some((_, t)) = fs.iter().find(|(m, _)| *m == name) {
                    let t = t.clone();
                    return self.call(&t, args, block, span, None);
                }
            }
            Ty::Variant(id) => {
                if let Some(sig) = self.enums[*id].methods.get(&name).cloned() {
                    let what = format!("`{n}`");
                    let dep = self
                        .deprecated_fns
                        .get(&(Rc::as_ptr(&sig) as usize))
                        .cloned();
                    let full = format!("`{}.{n}`", self.name(self.enums[*id].name));
                    self.warn_deprecated(dep.as_ref(), &full, span);
                    self.check_args(&sig, args, block, span, &what);
                    let ret = sig.ret.borrow().clone();
                    return ret;
                }
            }
            Ty::EnumValue(id) => {
                let id = *id;
                self.args_any(args, block);
                return match n.as_str() {
                    "values" => Ty::List(Box::new(Ty::Variant(id))),
                    "from" => Ty::optional(Ty::Variant(id)),
                    _ => {
                        let msg = format!(
                            "`{}` has no method `{n}` (`values()`, `from(value)`)",
                            self.name(self.enums[id].name)
                        );
                        self.error(msg, span);
                        Ty::Any
                    }
                };
            }
            _ => {}
        }
        self.builtin_method(r, &n, args, block, span)
    }

    /// Built-in methods: checks the number and types of arguments, returns the result.
    fn builtin_method(
        &mut self,
        r: &Ty,
        n: &str,
        args: Slice<Arg>,
        block: Option<LambdaId>,
        span: Span,
    ) -> Ty {
        let ast = self.ast;
        let arg_list: Vec<Arg> = ast.get(args).to_vec();
        if let Some(a) = arg_list.iter().find(|a| a.name.is_some()) {
            self.error(
                format!(
                    "built-in methods do not accept named arguments (`{}`)",
                    self.name(a.name.unwrap())
                ),
                ast.span(a.value),
            );
        }
        // Arguments: expressions, then the optional `do` block.
        let mut exprs: Vec<Option<ExprId>> = arg_list.iter().map(|a| Some(a.value)).collect();
        if block.is_some() {
            exprs.push(None);
        }
        let count = exprs.len();
        let elem = match r {
            Ty::List(t) => Some((**t).clone()),
            Ty::Range => Some(Ty::Int),
            _ => None,
        };
        let str_ty = matches!(r, Ty::Str | Ty::LitStr(_));
        let num = matches!(r, Ty::Int | Ty::LitInt(_) | Ty::Float);
        // (expected argument types: Some = fixed count, result)
        let sig: Option<(Vec<Ty>, Ty)> = match (n, r) {
            ("to_s" | "inspect", _) => Some((vec![], Ty::Str)),
            ("call", Ty::Fn(f)) => {
                let f = f.clone();
                self.check_args_dyn(&f, &exprs, block, span);
                return f.ret.borrow().clone();
            }
            ("abs", Ty::Int | Ty::LitInt(_)) => Some((vec![], Ty::Int)),
            ("abs", Ty::Float) => Some((vec![], Ty::Float)),
            ("to_f" | "sqrt", _) if num => Some((vec![], Ty::Float)),
            ("to_i" | "floor" | "ceil" | "round", _) if num => Some((vec![], Ty::Int)),
            ("to_f?", _) if num => Some((vec![], Ty::Float)),
            ("to_i?", Ty::Float) => Some((vec![], Ty::optional(Ty::Int))),
            ("to_i?", _) if num => Some((vec![], Ty::Int)),
            ("len", _)
                if str_ty || matches!(r, Ty::List(_) | Ty::Range | Ty::Hash(..) | Ty::Tuple(_)) =>
            {
                Some((vec![], Ty::Int))
            }
            ("is_empty", _) if str_ty || matches!(r, Ty::List(_) | Ty::Hash(..)) => {
                Some((vec![], Ty::Bool))
            }
            ("upper" | "lower" | "capitalize" | "trim" | "reverse", _) if str_ty => {
                Some((vec![], Ty::Str))
            }
            ("contains" | "starts_with" | "ends_with", _) if str_ty => {
                Some((vec![Ty::Str], Ty::Bool))
            }
            ("split", _) if str_ty => {
                if count > 1 {
                    self.error("`split(separator?)` expects 0 or 1 argument", span);
                }
                if let Some(Some(e)) = exprs.first() {
                    self.expect(*e, &Ty::Str, "");
                }
                return Ty::List(Box::new(Ty::Str));
            }
            ("replace", _) if str_ty => Some((vec![Ty::Str, Ty::Str], Ty::Str)),
            ("chars", _) if str_ty => Some((vec![], Ty::List(Box::new(Ty::Str)))),
            ("to_i", _) if str_ty => Some((vec![], Ty::Int)),
            ("to_f", _) if str_ty => Some((vec![], Ty::Float)),
            ("to_i?", _) if str_ty => Some((vec![], Ty::optional(Ty::Int))),
            ("to_f?", _) if str_ty => Some((vec![], Ty::optional(Ty::Float))),

            ("push", Ty::List(t)) => Some((vec![(**t).clone()], r.clone())),
            ("pop" | "first" | "last", Ty::List(t)) => Some((vec![], Ty::optional((**t).clone()))),
            ("contains", Ty::List(_)) => Some((vec![Ty::Any], Ty::Bool)),
            ("contains", Ty::Range) => Some((vec![Ty::Int], Ty::Bool)),
            ("join", Ty::List(_)) => {
                if count > 1 {
                    self.error("`join(separator?)` expects 0 or 1 argument", span);
                }
                if let Some(Some(e)) = exprs.first() {
                    self.expect(*e, &Ty::Str, "");
                }
                return Ty::Str;
            }
            ("reverse" | "sort", Ty::List(_)) => Some((vec![], r.clone())),
            ("sum", Ty::List(t)) => Some((
                vec![],
                match &**t {
                    Ty::Float => Ty::Float,
                    Ty::Int => Ty::Int,
                    _ => Ty::Any,
                },
            )),
            ("min" | "max", Ty::List(t)) => Some((vec![], Ty::optional((**t).clone()))),
            ("to_list", Ty::Range) => Some((vec![], Ty::List(Box::new(Ty::Int)))),
            ("map" | "filter" | "each" | "find" | "any" | "all", Ty::List(_) | Ty::Range) => {
                let elem = elem.clone().unwrap();
                if count != 1 {
                    self.error(format!("`{n}` expects 1 argument, got {count}"), span);
                }
                let f = self.fn_arg(
                    exprs.first().copied().flatten(),
                    block,
                    Self::fn_of(vec![elem.clone()]),
                );
                return match n {
                    "map" => Ty::List(Box::new(Self::fn_ret(&f).widen())),
                    "filter" => Ty::List(Box::new(elem)),
                    "each" => r.clone(),
                    "find" => Ty::optional(elem),
                    _ => Ty::Bool,
                };
            }
            ("reduce", Ty::List(_) | Ty::Range) => {
                let elem = elem.clone().unwrap();
                if count != 2 {
                    self.error(
                        format!("`reduce(initial, f)` expects 2 arguments, got {count}"),
                        span,
                    );
                }
                let init = match exprs.first().copied().flatten() {
                    Some(e) => self.expr(e, None).widen(),
                    None => Ty::Any,
                };
                let f = self.fn_arg(
                    exprs.get(1).copied().flatten(),
                    if count == 2 { block } else { None },
                    Self::fn_of(vec![init.clone(), elem]),
                );
                let ret = Self::fn_ret(&f);
                return union(vec![init, ret]);
            }

            ("keys", Ty::Hash(k, _)) => Some((vec![], Ty::List(k.clone()))),
            ("values", Ty::Hash(_, v)) => Some((vec![], Ty::List(v.clone()))),
            ("has" | "delete", Ty::Hash(k, v)) => Some((
                vec![(**k).clone()],
                if n == "has" {
                    Ty::Bool
                } else {
                    Ty::optional((**v).clone())
                },
            )),
            ("get", Ty::Hash(k, v)) => {
                if !(1..=2).contains(&count) {
                    self.error("`get(key, default?)` expects 1 or 2 arguments", span);
                }
                if let Some(Some(e)) = exprs.first() {
                    self.expect(*e, k, "");
                }
                return match exprs.get(1).copied().flatten() {
                    Some(d) => {
                        let dt = self.expr(d, None);
                        union(vec![(**v).clone(), dt])
                    }
                    None => Ty::optional((**v).clone()),
                };
            }
            ("each", Ty::Hash(k, v)) => {
                if count != 1 {
                    self.error(format!("`each` expects 1 argument, got {count}"), span);
                }
                self.fn_arg(
                    exprs.first().copied().flatten(),
                    block,
                    Self::fn_of(vec![(**k).clone(), (**v).clone()]),
                );
                return r.clone();
            }
            _ => None,
        };
        match sig {
            Some((params, ret)) => {
                if count != params.len() {
                    self.error(
                        format!(
                            "`{n}` expects {} argument{}, got {count}",
                            params.len(),
                            if params.len() != 1 { "s" } else { "" }
                        ),
                        span,
                    );
                }
                for (e, p) in exprs
                    .iter()
                    .zip(params.iter().chain(std::iter::repeat(&Ty::Any)))
                {
                    match e {
                        Some(e) => {
                            self.expect(*e, p, "");
                        }
                        None => {
                            self.lambda(block.unwrap(), Some(p));
                        }
                    }
                }
                ret
            }
            None => {
                for e in exprs.iter().flatten() {
                    self.expr(*e, None);
                }
                if let Some(b) = block {
                    self.lambda(b, None);
                }
                let hint = suggest(n, self.builtin_methods(r).iter().copied());
                let msg = format!("`{}` has no method `{n}`{hint}", self.show_v(r));
                self.error(msg, span);
                Ty::Any
            }
        }
    }

    /// Function argument of a built-in method (lambda argument or `do` block).
    fn fn_arg(&mut self, e: Option<ExprId>, block: Option<LambdaId>, expected: Ty) -> Ty {
        match (e, block) {
            (Some(e), _) => self.expect(e, &expected, ""),
            (None, Some(b)) => self.lambda(b, Some(&expected)),
            (None, None) => Ty::Any,
        }
    }

    fn check_args_dyn(
        &mut self,
        f: &FnSig,
        exprs: &[Option<ExprId>],
        block: Option<LambdaId>,
        span: Span,
    ) {
        if !f.lenient && exprs.len() > f.params.len() {
            self.error(
                format!(
                    "the function expects {} argument{}, got {}",
                    f.params.len(),
                    if f.params.len() != 1 { "s" } else { "" },
                    exprs.len()
                ),
                span,
            );
        }
        for (i, e) in exprs.iter().enumerate() {
            let p = f.params.get(i).map(|p| p.ty.clone()).unwrap_or(Ty::Any);
            match e {
                Some(e) => {
                    self.expect(*e, &p, "");
                }
                None => {
                    self.lambda(block.unwrap(), Some(&p));
                }
            }
        }
    }

    fn index(&mut self, r: &Ty, i: &Ty, span: Span, ispan: Span) -> Ty {
        match r {
            Ty::Any | Ty::Union(_) => Ty::Any,
            Ty::List(t) => {
                if !self.assignable(i, &Ty::Int) {
                    let msg = format!("an index must be an integer, got `{}`", self.show_v(i));
                    self.error(msg, ispan);
                }
                (**t).clone()
            }
            Ty::Str | Ty::LitStr(_) => {
                if !self.assignable(i, &Ty::Int) {
                    let msg = format!("an index must be an integer, got `{}`", self.show_v(i));
                    self.error(msg, ispan);
                }
                Ty::Str
            }
            Ty::Tuple(ts) => match i {
                Ty::LitInt(n) => {
                    let len = ts.len() as i64;
                    let pos = if *n < 0 { len + n } else { *n };
                    match usize::try_from(pos).ok().and_then(|p| ts.get(p)) {
                        Some(t) => t.clone(),
                        None => {
                            self.error(format!("index {n} out of tuple bounds (length {len})"), span);
                            Ty::Any
                        }
                    }
                }
                _ => union(ts.clone()),
            },
            Ty::Hash(k, v) => {
                if !self.assignable(i, k) {
                    let msg = format!(
                        "expected key of type `{}`, got `{}`",
                        self.show(k),
                        self.show(i)
                    );
                    self.error(msg, ispan);
                }
                Ty::optional((**v).clone())
            }
            t => {
                let msg = format!("`{}` cannot be indexed with `[…]`", self.show_v(t));
                self.error(msg, span);
                Ty::Any
            }
        }
    }

    /// Types of a `for` loop's variables, depending on what is iterated.
    fn for_vars(&mut self, it: &Ty, n: usize, span: Span) -> Vec<Ty> {
        let elem: Ty = match it {
            Ty::List(t) => (**t).clone(),
            Ty::Range => Ty::Int,
            Ty::Str | Ty::LitStr(_) => Ty::Str,
            Ty::Hash(k, v) => {
                return match n {
                    1 => vec![(**k).clone()],
                    2 => vec![(**k).clone(), (**v).clone()],
                    _ => {
                        self.error(
                            format!("a hash is iterated with 1 or 2 variables, not {n}"),
                            span,
                        );
                        vec![Ty::Any; n]
                    }
                };
            }
            Ty::Any | Ty::Union(_) | Ty::Tuple(_) => Ty::Any,
            t => {
                let msg = format!(
                    "`for` cannot iterate over a value of type `{}`",
                    self.show_v(t)
                );
                self.error(msg, span);
                Ty::Any
            }
        };
        if n == 1 {
            return vec![elem];
        }
        match elem {
            Ty::Tuple(ts) if ts.len() == n => ts,
            Ty::Any => vec![Ty::Any; n],
            Ty::Tuple(ts) => {
                self.error(
                    format!("{n} variables for tuples of {} values", ts.len()),
                    span,
                );
                vec![Ty::Any; n]
            }
            t => {
                let msg = format!(
                    "{n} variables expect a tuple on each iteration, got `{}`",
                    self.show(&t)
                );
                self.error(msg, span);
                vec![Ty::Any; n]
            }
        }
    }
}

fn is_str(t: &Ty) -> bool {
    matches!(t, Ty::Str | Ty::LitStr(_))
}

/// Result of `a op b` for two specific types, `None` if the operation is impossible.
fn binary_result(op: BinOp, a: &Ty, b: &Ty) -> Option<Ty> {
    use BinOp::*;
    use Ty::*;
    let int = |t: &Ty| matches!(t, Int | LitInt(_));
    let num = |t: &Ty| matches!(t, Int | LitInt(_) | Float);
    if a.is_any() || b.is_any() {
        return Some(match op {
            Lt | Le | Gt | Ge => Bool,
            Push if matches!(a, List(_)) => a.clone(),
            _ => Any,
        });
    }
    Some(match op {
        Add | Sub | Mul | Div | Rem if int(a) && int(b) => Int,
        Add | Sub | Mul | Div | Rem if num(a) && num(b) => Float,
        Add if is_str(a) && is_str(b) => Str,
        Mul if is_str(a) && int(b) => Str,
        Add => match (a, b) {
            (List(x), List(y)) => List(Box::new(union(vec![(**x).clone(), (**y).clone()]))),
            _ => return None,
        },
        Lt | Le | Gt | Ge if (num(a) && num(b)) || (is_str(a) && is_str(b)) => Bool,
        Push if matches!(a, List(_)) => a.clone(),
        _ => return None,
    })
}

fn pattern_names(p: &Pattern, out: &mut Vec<Symbol>) {
    match p {
        Pattern::Name(n, _) => out.push(*n),
        Pattern::Tuple(items, _) => items.iter().for_each(|p| pattern_names(p, out)),
        Pattern::Record(names, _) => out.extend(names.iter().map(|(n, _)| *n)),
        Pattern::List { items, rest, .. } => {
            items.iter().for_each(|p| pattern_names(p, out));
            out.extend(rest.iter().map(|(n, _)| *n));
        }
    }
}

fn tail_expr(b: &Block) -> Option<ExprId> {
    match b.stmts.last()?.kind {
        StmtKind::Expr(e) => Some(e),
        _ => None,
    }
}

/// Static type of an already computed value (global variable).
fn value_ty(v: &Value) -> Ty {
    match v {
        Value::Nil => Ty::Any,
        Value::Bool(_) => Ty::Bool,
        Value::Int(_) => Ty::Int,
        Value::Float(_) => Ty::Float,
        Value::Str(_) => Ty::Str,
        Value::Range(_) => Ty::Range,
        _ => Ty::Any,
    }
}

/// Literals replaced by their type (`"a"` → `str`), recursively.
fn plain(t: &Ty) -> Ty {
    match t {
        Ty::LitStr(_) => Ty::Str,
        Ty::LitInt(_) => Ty::Int,
        Ty::List(x) => Ty::List(Box::new(plain(x))),
        Ty::Hash(k, v) => Ty::Hash(Box::new(plain(k)), Box::new(plain(v))),
        Ty::Record(fs) => Ty::Record(fs.iter().map(|(n, t)| (*n, plain(t))).collect()),
        Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(plain).collect()),
        Ty::Union(ts) => union(ts.iter().map(plain).collect()),
        t => t.clone(),
    }
}
