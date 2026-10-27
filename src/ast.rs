//! Compact syntax tree.
//!
//! Expressions (most of the nodes) live in a single `Vec` of the [`Ast`] and are referred
//! to by an [`ExprId`] (u32). Variable-length lists (arguments, elements, fields…) are
//! [`Slice`]s of shared pools. Consequences:
//!
//! - no allocation per node, and dropping the tree only frees a few `Vec`s;
//! - 32 bytes per expression, stored in source order (good cache locality);
//! - a node is identified by an integer: an interpreter closure only keeps an index, and
//!   the AST is easy to serialize (planned for macros).
//!
//! Declarations and blocks (`def`, `type`, `if`…), far less numerous, remain ordinary
//! owned structures. Every node carries its `Span`; names and string literals are
//! interned `Symbol`s.

use crate::intern::Symbol;
use crate::span::Span;
use std::marker::PhantomData;
use std::num::NonZeroU32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExprId(pub u32);

/// Index of a lambda (`do` block or `|x| …`). `NonZeroU32`: `Option<LambdaId>` is 4 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LambdaId(NonZeroU32);

impl LambdaId {
    #[inline]
    pub fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

/// Index of a control structure (`if`, `case`, `while`, `for`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ControlId(pub u32);

/// Range `[start, start + len)` of an [`Ast`] pool.
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

/// Node pools of a program.
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

/// An AST pool, addressable by `Slice<T>`.
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

    /// Elements of a slice.
    #[inline]
    pub fn get<T>(&self, s: Slice<T>) -> &[T]
    where
        Self: Pool<T>,
    {
        &self.pool()[s.range()]
    }

    /// Copies `items` into the pool and returns the slice.
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
            _ => unreachable!("ControlId does not refer to an `if`"),
        }
    }

    pub fn case(&self, id: ControlId) -> &Case {
        match self.control(id) {
            Control::Case(x) => x,
            _ => unreachable!("ControlId does not refer to a `case`"),
        }
    }

    pub fn while_(&self, id: ControlId) -> &While {
        match self.control(id) {
            Control::While(x) => x,
            _ => unreachable!("ControlId does not refer to a `while`"),
        }
    }

    pub fn for_(&self, id: ControlId) -> &For {
        match self.control(id) {
            Control::For(x) => x,
            _ => unreachable!("ControlId does not refer to a `for`"),
        }
    }

    /// Number of expressions.
    pub fn len(&self) -> usize {
        self.exprs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.exprs.is_empty()
    }

    /// Can appear on the left of an `=`.
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

/// `let pattern(: type)? = value`
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

/// `@[Name(args)]`
#[derive(Clone, Debug, PartialEq)]
pub struct AnnotationUse {
    pub path: Vec<Symbol>,
    pub args: Slice<Arg>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: Symbol,
    /// `@name` in a constructor: declares and assigns the field.
    pub is_field: bool,
    pub ty: Option<TypeExpr>,
    pub default: Option<ExprId>,
    pub annotations: Vec<AnnotationUse>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FnDecl {
    /// For a setter `def name=(v)`, the name is `name=`.
    pub name: Symbol,
    pub name_span: Span,
    /// `None` for a method declared without parentheses (`def to_s: str = …`).
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

/// `type`, `class` or `annotation`.
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
    /// Any other statement in the body (macro call, nested type…).
    Stmt(Stmt),
}

/// `name: type = default`, `name: type` or `name = default`
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
    /// Named argument: `f(name: value)`
    pub name: Option<Symbol>,
    pub value: ExprId,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StrPart {
    /// Literal text (interned).
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
    /// `<<`: append to a list
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

/// Expression. `Copy`: children are indices, not owned pointers.
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
    /// `@name`
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
    /// `target = value`, `target += value` (`op = Some(Add)`)
    Assign {
        target: ExprId,
        op: Option<BinOp>,
        value: ExprId,
    },

    /// `f(args)` or `Server do … end`; the `do` block is passed as the last argument.
    Call {
        callee: ExprId,
        args: Slice<Arg>,
        block: Option<LambdaId>,
    },
    /// `recv.name(args)`
    MethodCall {
        recv: ExprId,
        name: Symbol,
        args: Slice<Arg>,
        block: Option<LambdaId>,
    },
    /// `recv.name`
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

/// Control structures, stored separately because they contain blocks.
#[derive(Clone, Debug, PartialEq)]
pub enum Control {
    If(If),
    Case(Case),
    While(While),
    For(For),
}
