//! Arbre syntaxique compact.
//!
//! Les expressions (l'essentiel des nœuds) sont rangées dans un seul `Vec` de l'[`Ast`] et
//! désignées par un [`ExprId`] (u32). Les listes de longueur variable (arguments, éléments,
//! champs…) sont des tranches [`Slice`] de réserves partagées. Conséquences :
//!
//! - aucune allocation par nœud, et une destruction qui ne libère que quelques `Vec` ;
//! - 32 octets par expression, rangées dans l'ordre du source (bonne localité de cache) ;
//! - un nœud se désigne par un entier : une closure de l'interpréteur ne garde qu'un
//!   index, et l'AST se sérialise facilement (prévu pour les macros).
//!
//! Les déclarations et les blocs (`def`, `type`, `if`…), bien moins nombreux, restent des
//! structures possédées classiques. Chaque nœud porte son `Span` ; les noms et les chaînes
//! littérales sont des `Symbol` internés.

use crate::intern::Symbol;
use crate::span::Span;
use std::marker::PhantomData;
use std::num::NonZeroU32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExprId(pub u32);

/// Index d'un lambda (bloc `do` ou `|x| …`). `NonZeroU32` : `Option<LambdaId>` fait 4 octets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LambdaId(NonZeroU32);

impl LambdaId {
    #[inline]
    pub fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

/// Index d'une structure de contrôle (`if`, `case`, `while`, `for`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ControlId(pub u32);

/// Tranche `[start, start + len)` d'une réserve de l'[`Ast`].
pub struct Slice<T> {
    pub start: u32,
    pub len: u32,
    _t: PhantomData<fn() -> T>,
}

impl<T> Slice<T> {
    pub const EMPTY: Slice<T> = Slice {
        start: 0,
        len: 0,
        _t: PhantomData,
    };

    #[inline]
    fn range(self) -> std::ops::Range<usize> {
        self.start as usize..(self.start + self.len) as usize
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

impl<T> Clone for Slice<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Slice<T> {}
impl<T> PartialEq for Slice<T> {
    fn eq(&self, o: &Self) -> bool {
        self.start == o.start && self.len == o.len
    }
}
impl<T> std::fmt::Debug for Slice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}..+{}", self.start, self.len)
    }
}

/// Réserves de nœuds d'un programme.
#[derive(Clone, Debug, Default)]
pub struct Ast {
    exprs: Vec<Expr>,
    expr_lists: Vec<ExprId>,
    args: Vec<Arg>,
    fields: Vec<(Symbol, ExprId)>,
    entries: Vec<(ExprId, ExprId)>,
    parts: Vec<StrPart>,
    syms: Vec<Symbol>,
    lambdas: Vec<Lambda>,
    controls: Vec<Control>,
}

/// Une réserve de l'AST, adressable par `Slice<T>`.
pub trait Pool<T> {
    fn pool(&self) -> &Vec<T>;
    fn pool_mut(&mut self) -> &mut Vec<T>;
}

macro_rules! pool {
    ($t:ty, $field:ident) => {
        impl Pool<$t> for Ast {
            #[inline]
            fn pool(&self) -> &Vec<$t> {
                &self.$field
            }
            #[inline]
            fn pool_mut(&mut self) -> &mut Vec<$t> {
                &mut self.$field
            }
        }
    };
}
pool!(ExprId, expr_lists);
pool!(Arg, args);
pool!((Symbol, ExprId), fields);
pool!((ExprId, ExprId), entries);
pool!(StrPart, parts);
pool!(Symbol, syms);

impl Ast {
    #[inline]
    pub fn expr(&self, id: ExprId) -> &Expr {
        &self.exprs[id.0 as usize]
    }

    #[inline]
    pub fn expr_mut(&mut self, id: ExprId) -> &mut Expr {
        &mut self.exprs[id.0 as usize]
    }

    #[inline]
    pub fn kind(&self, id: ExprId) -> ExprKind {
        self.exprs[id.0 as usize].kind
    }

    #[inline]
    pub fn span(&self, id: ExprId) -> Span {
        self.exprs[id.0 as usize].span
    }

    #[inline]
    pub fn push(&mut self, kind: ExprKind, span: Span) -> ExprId {
        let id = ExprId(self.exprs.len() as u32);
        self.exprs.push(Expr { kind, span });
        id
    }

    /// Éléments d'une tranche.
    #[inline]
    pub fn get<T>(&self, s: Slice<T>) -> &[T]
    where
        Self: Pool<T>,
    {
        &self.pool()[s.range()]
    }

    /// Copie `items` dans la réserve et renvoie la tranche.
    pub fn alloc<T: Clone>(&mut self, items: &[T]) -> Slice<T>
    where
        Self: Pool<T>,
    {
        if items.is_empty() {
            return Slice::EMPTY;
        }
        let pool = self.pool_mut();
        let start = pool.len() as u32;
        pool.extend_from_slice(items);
        Slice {
            start,
            len: items.len() as u32,
            _t: PhantomData,
        }
    }

    pub fn lambda(&self, id: LambdaId) -> &Lambda {
        &self.lambdas[id.index()]
    }

    pub fn push_lambda(&mut self, l: Lambda) -> LambdaId {
        self.lambdas.push(l);
        LambdaId(NonZeroU32::new(self.lambdas.len() as u32).unwrap())
    }

    pub fn push_control(&mut self, c: Control, span: Span) -> ExprId {
        let id = ControlId(self.controls.len() as u32);
        let kind = match &c {
            Control::If(_) => ExprKind::If(id),
            Control::Case(_) => ExprKind::Case(id),
            Control::While(_) => ExprKind::While(id),
            Control::For(_) => ExprKind::For(id),
        };
        self.controls.push(c);
        self.push(kind, span)
    }

    pub fn control(&self, id: ControlId) -> &Control {
        &self.controls[id.0 as usize]
    }

    pub fn if_(&self, id: ControlId) -> &If {
        match self.control(id) {
            Control::If(x) => x,
            _ => unreachable!("ControlId ne désigne pas un `if`"),
        }
    }

    pub fn case(&self, id: ControlId) -> &Case {
        match self.control(id) {
            Control::Case(x) => x,
            _ => unreachable!("ControlId ne désigne pas un `case`"),
        }
    }

    pub fn while_(&self, id: ControlId) -> &While {
        match self.control(id) {
            Control::While(x) => x,
            _ => unreachable!("ControlId ne désigne pas un `while`"),
        }
    }

    pub fn for_(&self, id: ControlId) -> &For {
        match self.control(id) {
            Control::For(x) => x,
            _ => unreachable!("ControlId ne désigne pas un `for`"),
        }
    }

    /// Nombre d'expressions.
    pub fn len(&self) -> usize {
        self.exprs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.exprs.is_empty()
    }

    /// Peut se trouver à gauche d'un `=`.
    pub fn is_assignable(&self, id: ExprId) -> bool {
        matches!(
            self.kind(id),
            ExprKind::Ident(_) | ExprKind::IVar(_) | ExprKind::Field(..) | ExprKind::Index(..)
        )
    }
}

#[derive(Clone, Debug)]
pub struct Program {
    pub ast: Ast,
    pub body: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StmtKind {
    Expr(ExprId),
    Let(Box<Let>),
    Def(Box<FnDecl>),
    Type(Box<TypeDecl>),
    Enum(Box<EnumDecl>),
}

/// `let motif(: type)? = valeur`
#[derive(Clone, Debug, PartialEq)]
pub struct Let {
    pub pattern: Pattern,
    pub ty: Option<TypeExpr>,
    pub value: ExprId,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    Name(Symbol, Span),
    /// `(a, b)`
    Tuple(Vec<Pattern>, Span),
    /// `{x, y}`
    Record(Vec<(Symbol, Span)>, Span),
    /// `[first, ...rest]`
    List {
        items: Vec<Pattern>,
        rest: Option<(Symbol, Span)>,
        span: Span,
    },
}

impl Pattern {
    pub fn span(&self) -> Span {
        match self {
            Pattern::Name(_, s) | Pattern::Tuple(_, s) | Pattern::Record(_, s) => *s,
            Pattern::List { span, .. } => *span,
        }
    }
}

/// `@[Nom(args)]`
#[derive(Clone, Debug, PartialEq)]
pub struct AnnotationUse {
    pub path: Vec<Symbol>,
    pub args: Slice<Arg>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: Symbol,
    /// `@nom` dans un constructeur : déclare et assigne le champ.
    pub is_field: bool,
    pub ty: Option<TypeExpr>,
    pub default: Option<ExprId>,
    pub annotations: Vec<AnnotationUse>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FnDecl {
    /// Pour un setter `def nom=(v)`, le nom est `nom=`.
    pub name: Symbol,
    pub name_span: Span,
    /// `None` pour une méthode déclarée sans parenthèses (`def to_s: str = …`).
    pub params: Option<Vec<Param>>,
    pub ret: Option<TypeExpr>,
    pub body: Block,
    pub annotations: Vec<AnnotationUse>,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeDeclKind {
    Type,
    Class,
    Annotation,
}

/// `type`, `class` ou `annotation`.
#[derive(Clone, Debug, PartialEq)]
pub struct TypeDecl {
    pub kind: TypeDeclKind,
    pub name: Symbol,
    pub name_span: Span,
    pub members: Vec<Member>,
    pub annotations: Vec<AnnotationUse>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Member {
    Field(Field),
    Method(FnDecl),
    /// Autre instruction dans le corps (appel de macro, type imbriqué…).
    Stmt(Stmt),
}

/// `nom: type = défaut`, `nom: type` ou `nom = défaut`
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub name: Symbol,
    pub ty: Option<TypeExpr>,
    pub default: Option<ExprId>,
    pub annotations: Vec<AnnotationUse>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnumDecl {
    pub name: Symbol,
    pub name_span: Span,
    pub repr: Option<TypeExpr>,
    pub variants: Vec<Variant>,
    pub methods: Vec<FnDecl>,
    pub annotations: Vec<AnnotationUse>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Variant {
    pub name: Symbol,
    pub value: Option<ExprId>,
    pub annotations: Vec<AnnotationUse>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypeExpr {
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeKind {
    /// `int`, `Server`, `Protocol::Tcp`
    Named(Vec<Symbol>),
    /// `[T]`
    List(Box<TypeExpr>),
    /// `{K => V}`
    Hash(Box<TypeExpr>, Box<TypeExpr>),
    /// `{a: T, b: U}`
    Record(Vec<(Symbol, TypeExpr)>),
    /// `(A, B)`
    Tuple(Vec<TypeExpr>),
    /// `T?`
    Optional(Box<TypeExpr>),
    /// `A | B`
    Union(Vec<TypeExpr>),
    /// `"dev"`
    LitStr(String),
    /// `2`
    LitInt(i64),
    /// `(A, B): R`
    Fn(Vec<TypeExpr>, Box<TypeExpr>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arg {
    /// Argument nommé : `f(nom: valeur)`
    pub name: Option<Symbol>,
    pub value: ExprId,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StrPart {
    /// Texte littéral (interné).
    Lit(Symbol),
    Expr(ExprId),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Lambda {
    pub params: Vec<Param>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WhenArm {
    pub patterns: Slice<ExprId>,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    /// `<<` : ajout à une liste
    Push,
}

impl BinOp {
    pub fn as_str(self) -> &'static str {
        use BinOp::*;
        match self {
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Rem => "%",
            Eq => "==",
            Ne => "!=",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            And => "and",
            Or => "or",
            Push => "<<",
        }
    }
}

/// Expression. `Copy` : les enfants sont des index, pas des pointeurs possédés.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExprKind {
    Int(i64),
    Float(f64),
    Str(Symbol),
    Interp(Slice<StrPart>),
    Bool(bool),
    Nil,
    SelfRef,
    Ident(Symbol),
    /// `@nom`
    IVar(Symbol),
    /// `Protocol::Tcp`
    Path(Slice<Symbol>),

    List(Slice<ExprId>),
    Hash(Slice<(ExprId, ExprId)>),
    Record(Slice<(Symbol, ExprId)>),
    Tuple(Slice<ExprId>),

    Unary(UnaryOp, ExprId),
    Binary(BinOp, ExprId, ExprId),
    Range {
        lo: ExprId,
        hi: ExprId,
        inclusive: bool,
    },
    /// `cible = valeur`, `cible += valeur` (`op = Some(Add)`)
    Assign {
        target: ExprId,
        op: Option<BinOp>,
        value: ExprId,
    },

    /// `f(args)` ou `Server do … end` ; le bloc `do` est passé en dernier argument.
    Call {
        callee: ExprId,
        args: Slice<Arg>,
        block: Option<LambdaId>,
    },
    /// `recv.nom(args)`
    MethodCall {
        recv: ExprId,
        name: Symbol,
        args: Slice<Arg>,
        block: Option<LambdaId>,
    },
    /// `recv.nom`
    Field(ExprId, Symbol),
    /// `recv.0`
    TupleIndex(ExprId, u32),
    /// `recv[index]`
    Index(ExprId, ExprId),

    Lambda(LambdaId),

    If(ControlId),
    Case(ControlId),
    While(ControlId),
    For(ControlId),
    Return(Option<ExprId>),
    Break,
    Next,
}

#[derive(Clone, Debug, PartialEq)]
pub struct If {
    pub cond: ExprId,
    pub then: Block,
    pub else_: Option<Block>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Case {
    pub subject: ExprId,
    pub arms: Vec<WhenArm>,
    pub else_: Option<Block>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct While {
    pub cond: ExprId,
    pub body: Block,
}

#[derive(Clone, Debug, PartialEq)]
pub struct For {
    pub vars: Vec<(Symbol, Span)>,
    pub iter: ExprId,
    pub body: Block,
}

/// Structures de contrôle, rangées à part car elles contiennent des blocs.
#[derive(Clone, Debug, PartialEq)]
pub enum Control {
    If(If),
    Case(Case),
    While(While),
    For(For),
}
