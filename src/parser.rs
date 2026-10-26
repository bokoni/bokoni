//! Parseur écrit à la main : descente récursive, et Pratt pour les opérateurs.
//!
//! - Les jetons sont lus à la demande (un seul jeton d'avance), jamais de retour en
//!   arrière : chaque jeton est lu une fois.
//! - Les expressions sont ajoutées à l'[`Ast`] compact ; les listes (arguments,
//!   éléments…) sont accumulées sur des piles temporaires réutilisées, puis copiées d'un
//!   bloc dans les réserves de l'AST : aucune allocation par liste.
//!
//! Précédence (de la plus faible à la plus forte) :
//!
//! | niveau | opérateurs                     | associativité |
//! |--------|--------------------------------|---------------|
//! | 1      | `=` `+=` `-=` …                | droite        |
//! | 2      | `or`                           | gauche        |
//! | 3      | `and`                          | gauche        |
//! | 4      | `not` (préfixe)                |               |
//! | 5      | `==` `!=` `<` `<=` `>` `>=`    | aucune        |
//! | 6      | `..` `...`                     | aucune        |
//! | 7      | `<<`                           | gauche        |
//! | 8      | `+` `-`                        | gauche        |
//! | 9      | `*` `/` `%`                    | gauche        |
//! | 10     | `-` (préfixe)                  |               |
//! | 11     | `f()` `x.y` `x[i]` `A::B` `do` | gauche        |

use crate::ast::*;
use crate::diagnostic::{Error, Result};
use crate::intern::{Interner, Symbol};
use crate::lexer::{Lexer, Token, TokenKind};
use crate::span::{Span, line_col};

use TokenKind as T;

/// Profondeur maximale d'imbrication.
const MAX_DEPTH: u32 = 1000;

/// Si la pile restante descend sous `RED_ZONE`, la récursion continue sur une nouvelle
/// pile de `STACK_GROW` allouée sur le tas : aucune imbrication ne peut faire déborder
/// la pile du thread appelant.
const RED_ZONE: usize = 128 * 1024;
const STACK_GROW: usize = 2 * 1024 * 1024;

pub fn parse(src: &str, interner: &mut Interner) -> Result<Program> {
    let mut lexer = Lexer::new(src);
    let cur = lexer.next_token();
    let mut p = Parser {
        src,
        lexer,
        cur,
        ahead: None,
        prev_end: 0,
        interner,
        ast: Ast::default(),
        s_exprs: Vec::new(),
        s_args: Vec::new(),
        s_fields: Vec::new(),
        s_entries: Vec::new(),
        s_parts: Vec::new(),
        depth: 0,
        no_do: false,
    };
    let result = p.block(&[]).and_then(|body| {
        p.expect(T::Eof, "")?;
        Ok(body)
    });
    // Une erreur du lexer est remontée en priorité : le parseur n'a vu qu'un `Eof` à sa place.
    if let Some(e) = p.lexer.error.take() {
        return Err(e);
    }
    Ok(Program {
        body: result?,
        ast: p.ast,
    })
}

struct Parser<'s, 'i> {
    src: &'s str,
    /// Jetons lus à la demande : seuls le jeton courant et un jeton d'avance existent.
    lexer: Lexer<'s>,
    cur: Token,
    ahead: Option<Token>,
    /// Fin du dernier jeton consommé.
    prev_end: u32,
    interner: &'i mut Interner,
    ast: Ast,
    // Piles temporaires pour les listes en cours de construction (imbrication : chaque
    // liste note la hauteur de départ et ne retire que ses propres éléments).
    s_exprs: Vec<ExprId>,
    s_args: Vec<Arg>,
    s_fields: Vec<(Symbol, ExprId)>,
    s_entries: Vec<(ExprId, ExprId)>,
    s_parts: Vec<StrPart>,
    depth: u32,
    /// Dans l'en-tête d'un `while` / `for`, un `do` ne s'attache pas à l'expression.
    no_do: bool,
}

#[derive(Clone, Copy)]
enum Infix {
    Bin(BinOp),
    Range(bool),
}

/// Puissance de liaison des opérateurs binaires : (gauche, droite).
#[inline]
fn infix_bp(kind: TokenKind) -> Option<(u8, u8, Infix)> {
    use Infix::*;
    Some(match kind {
        T::Or => (2, 3, Bin(BinOp::Or)),
        T::And => (4, 5, Bin(BinOp::And)),
        T::EqEq => (8, 9, Bin(BinOp::Eq)),
        T::NotEq => (8, 9, Bin(BinOp::Ne)),
        T::Lt => (8, 9, Bin(BinOp::Lt)),
        T::Le => (8, 9, Bin(BinOp::Le)),
        T::Gt => (8, 9, Bin(BinOp::Gt)),
        T::Ge => (8, 9, Bin(BinOp::Ge)),
        T::DotDot => (10, 11, Range(true)),
        T::DotDotDot => (10, 11, Range(false)),
        T::Shl => (12, 13, Bin(BinOp::Push)),
        T::Plus => (14, 15, Bin(BinOp::Add)),
        T::Minus => (14, 15, Bin(BinOp::Sub)),
        T::Star => (16, 17, Bin(BinOp::Mul)),
        T::Slash => (16, 17, Bin(BinOp::Div)),
        T::Percent => (16, 17, Bin(BinOp::Rem)),
        _ => return None,
    })
}

const BP_NOT: u8 = 6;
const BP_NEG: u8 = 18;

#[inline]
fn assign_op(kind: TokenKind) -> Option<Option<BinOp>> {
    Some(match kind {
        T::Eq => None,
        T::PlusEq => Some(BinOp::Add),
        T::MinusEq => Some(BinOp::Sub),
        T::StarEq => Some(BinOp::Mul),
        T::SlashEq => Some(BinOp::Div),
        T::PercentEq => Some(BinOp::Rem),
        _ => return None,
    })
}

fn is_comparison(kind: TokenKind) -> bool {
    matches!(kind, T::EqEq | T::NotEq | T::Lt | T::Le | T::Gt | T::Ge)
}

/// Jetons qui ne peuvent pas commencer une expression : `return` sans valeur avant eux.
fn ends_expr(kind: TokenKind) -> bool {
    matches!(
        kind,
        T::Newline
            | T::Semi
            | T::Eof
            | T::End
            | T::If
            | T::Unless
            | T::Else
            | T::Elsif
            | T::When
            | T::Then
            | T::RParen
            | T::RBracket
            | T::RBrace
            | T::Comma
            | T::Pipe
    )
}

/// Mots-clés acceptés comme nom de méthode ou de champ après `.` (`x.type`, `x.class`).
fn is_keyword(kind: TokenKind) -> bool {
    matches!(
        kind,
        T::Let
            | T::Def
            | T::End
            | T::If
            | T::Elsif
            | T::Else
            | T::Unless
            | T::Then
            | T::Case
            | T::When
            | T::While
            | T::For
            | T::In
            | T::Do
            | T::Break
            | T::Next
            | T::Return
            | T::Type
            | T::Class
            | T::Enum
            | T::Annotation
            | T::Macro
            | T::True
            | T::False
            | T::Nil
            | T::And
            | T::Or
            | T::Not
            | T::SelfKw
    )
}

/// Bloc d'une seule instruction-expression.
fn expr_block(e: ExprId, span: Span) -> Block {
    Block {
        span,
        stmts: vec![Stmt {
            span,
            kind: StmtKind::Expr(e),
        }],
    }
}

impl Parser<'_, '_> {
    // ─── Jetons ──────────────────────────────────────────────────────────────

    #[inline]
    fn peek(&self) -> TokenKind {
        self.cur.kind
    }

    /// Type du jeton qui suit le jeton courant.
    #[inline]
    fn peek2(&mut self) -> TokenKind {
        match self.ahead {
            Some(t) => t.kind,
            None => {
                let t = self.lexer.next_token();
                self.ahead = Some(t);
                t.kind
            }
        }
    }

    #[inline]
    fn tok(&self) -> Token {
        self.cur
    }

    #[inline]
    fn at(&self, kind: TokenKind) -> bool {
        self.cur.kind == kind
    }

    #[inline]
    fn bump(&mut self) -> Token {
        let t = self.cur;
        if t.kind != T::Eof {
            self.prev_end = t.span.end;
            self.cur = match self.ahead.take() {
                Some(next) => next,
                None => self.lexer.next_token(),
            };
        }
        t
    }

    #[inline]
    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.at(kind) {
            self.bump();
            true
        } else {
            false
        }
    }

    #[inline]
    fn span_from(&self, start: Span) -> Span {
        Span::new(start.start, self.prev_end.max(start.end))
    }

    #[inline]
    fn span(&self, e: ExprId) -> Span {
        self.ast.span(e)
    }

    #[inline]
    fn mk(&mut self, kind: ExprKind, span: Span) -> ExprId {
        self.ast.push(kind, span)
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), T::Newline | T::Semi) {
            self.bump();
        }
    }

    // ─── Erreurs ─────────────────────────────────────────────────────────────

    fn found(&self) -> String {
        let t = self.tok();
        match t.kind {
            T::Ident | T::Int | T::Float => format!("`{}`", t.span.text(self.src)),
            k => k.describe().to_string(),
        }
    }

    fn error_at<R>(&self, msg: impl Into<String>, span: Span) -> Result<R> {
        Err(Error::syntax(msg, span))
    }

    fn error_here<R>(&self, msg: impl Into<String>) -> Result<R> {
        let mut e = Error::syntax(msg, self.tok().span);
        e.incomplete = self.at(T::Eof);
        Err(e)
    }

    /// Consomme un jeton du type attendu, ou produit « X attendu `contexte`, trouvé Y ».
    fn expect(&mut self, kind: TokenKind, context: &str) -> Result<Token> {
        if self.at(kind) {
            return Ok(self.bump());
        }
        let sep = if context.is_empty() { "" } else { " " };
        self.error_here(format!(
            "{} attendu{sep}{context}, trouvé {}",
            kind.describe(),
            self.found()
        ))
    }

    fn unclosed<R>(&self, opener: &str, opener_span: Span) -> Result<R> {
        let (line, _) = line_col(self.src, opener_span.start);
        self.error_here(format!(
            "`end` attendu pour fermer `{opener}` (ligne {line}), trouvé {}",
            self.found()
        ))
    }

    fn enter(&mut self) -> Result<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return self.error_here("imbrication trop profonde");
        }
        Ok(())
    }

    // ─── Noms ────────────────────────────────────────────────────────────────

    #[inline]
    fn intern(&mut self, span: Span) -> Symbol {
        let src = self.src;
        self.interner.intern(span.text(src))
    }

    fn ident(&mut self, context: &str) -> Result<(Symbol, Span)> {
        let t = self.expect(T::Ident, context)?;
        Ok((self.intern(t.span), t.span))
    }

    // ─── Blocs et instructions ───────────────────────────────────────────────

    /// Suite d'instructions jusqu'à l'un des `terminators` (non consommé) ou la fin.
    fn block(&mut self, terminators: &[TokenKind]) -> Result<Block> {
        stacker::maybe_grow(RED_ZONE, STACK_GROW, || self.block_inner(terminators))
    }

    fn block_inner(&mut self, terminators: &[TokenKind]) -> Result<Block> {
        self.enter()?;
        let start = self.tok().span;
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            let k = self.peek();
            if k == T::Eof || terminators.contains(&k) {
                break;
            }
            stmts.push(self.stmt()?);
            let k = self.peek();
            if !matches!(k, T::Newline | T::Semi | T::Eof) && !terminators.contains(&k) {
                return self.error_here(format!(
                    "fin de ligne attendue après l'instruction, trouvé {}",
                    self.found()
                ));
            }
        }
        self.depth -= 1;
        let span = match (stmts.first(), stmts.last()) {
            (Some(a), Some(b)) => a.span.to(b.span),
            _ => Span::new(start.start, start.start),
        };
        Ok(Block { stmts, span })
    }

    /// Bloc terminé par `end` (consommé).
    fn block_end(&mut self, opener: &str, opener_span: Span) -> Result<Block> {
        let body = self.block(&[T::End])?;
        self.end(opener, opener_span)?;
        Ok(body)
    }

    fn end(&mut self, opener: &str, opener_span: Span) -> Result<()> {
        if self.eat(T::End) {
            Ok(())
        } else {
            self.unclosed(opener, opener_span)
        }
    }

    fn stmt(&mut self) -> Result<Stmt> {
        let annotations = self.annotations()?;
        self.stmt_with(annotations)
    }

    fn stmt_with(&mut self, annotations: Vec<AnnotationUse>) -> Result<Stmt> {
        let start = self.tok().span;
        let kind = match self.peek() {
            T::Def => StmtKind::Def(Box::new(self.def(annotations)?)),
            T::Type | T::Class | T::Annotation => {
                StmtKind::Type(Box::new(self.type_decl(annotations)?))
            }
            T::Enum => StmtKind::Enum(Box::new(self.enum_decl(annotations)?)),
            T::Macro => {
                return self
                    .error_here("les macros ne sont pas encore supportées (prévu : version 3)");
            }
            _ if !annotations.is_empty() => {
                return self.error_here(format!(
                    "une annotation doit précéder une déclaration (`def`, `type`, `class`, `enum`, champ), trouvé {}",
                    self.found()
                ));
            }
            T::Let => StmtKind::Let(Box::new(self.let_stmt()?)),
            _ => StmtKind::Expr(self.expr_stmt()?),
        };
        Ok(Stmt {
            kind,
            span: self.span_from(start),
        })
    }

    /// Expression suivie d'éventuels modificateurs : `expr if cond`, `expr unless cond`.
    fn expr_stmt(&mut self) -> Result<ExprId> {
        let mut e = self.expr()?;
        while matches!(self.peek(), T::If | T::Unless) {
            let negate = self.bump().kind == T::Unless;
            let mut cond = self.expr()?;
            if negate {
                let span = self.span(cond);
                cond = self.mk(ExprKind::Unary(UnaryOp::Not, cond), span);
            }
            let e_span = self.span(e);
            let span = e_span.to(self.span(cond));
            let c = Control::If(If {
                cond,
                then: expr_block(e, e_span),
                else_: None,
            });
            e = self.ast.push_control(c, span);
        }
        Ok(e)
    }

    fn let_stmt(&mut self) -> Result<Let> {
        self.bump();
        let pattern = self.pattern()?;
        let ty = if self.eat(T::Colon) {
            Some(self.type_expr(true)?)
        } else {
            None
        };
        self.expect(T::Eq, "après la déclaration `let`")?;
        let value = self.expr()?;
        Ok(Let { pattern, ty, value })
    }

    fn pattern(&mut self) -> Result<Pattern> {
        let start = self.tok().span;
        match self.peek() {
            T::Ident => {
                let (name, span) = self.ident("")?;
                Ok(Pattern::Name(name, span))
            }
            T::LParen => {
                self.bump();
                let mut items = Vec::new();
                while !self.at(T::RParen) {
                    items.push(self.pattern()?);
                    if !self.eat(T::Comma) {
                        break;
                    }
                }
                self.expect(T::RParen, "pour fermer le motif")?;
                Ok(Pattern::Tuple(items, self.span_from(start)))
            }
            T::LBrace => {
                self.bump();
                let mut names = Vec::new();
                while !self.at(T::RBrace) {
                    names.push(self.ident("dans le motif `{…}`")?);
                    if !self.eat(T::Comma) {
                        break;
                    }
                }
                self.expect(T::RBrace, "pour fermer le motif")?;
                Ok(Pattern::Record(names, self.span_from(start)))
            }
            T::LBracket => {
                self.bump();
                let mut items = Vec::new();
                let mut rest = None;
                while !self.at(T::RBracket) {
                    if self.eat(T::DotDotDot) {
                        rest = Some(self.ident("après `...`")?);
                        break;
                    }
                    items.push(self.pattern()?);
                    if !self.eat(T::Comma) {
                        break;
                    }
                }
                self.expect(T::RBracket, "pour fermer le motif")?;
                Ok(Pattern::List {
                    items,
                    rest,
                    span: self.span_from(start),
                })
            }
            _ => self.error_here(format!(
                "nom de variable attendu après `let`, trouvé {}",
                self.found()
            )),
        }
    }

    fn annotations(&mut self) -> Result<Vec<AnnotationUse>> {
        let mut out = Vec::new();
        while self.at(T::AtLBracket) {
            let start = self.bump().span;
            let mut path = vec![self.ident("après `@[`")?.0];
            while self.eat(T::ColonColon) {
                path.push(self.ident("après `::`")?.0);
            }
            let args = if self.at(T::LParen) {
                self.args()?
            } else {
                Slice::EMPTY
            };
            self.expect(T::RBracket, "pour fermer l'annotation")?;
            out.push(AnnotationUse {
                path,
                args,
                span: self.span_from(start),
            });
            self.skip_newlines();
        }
        Ok(out)
    }

    // ─── Déclarations ────────────────────────────────────────────────────────

    fn def(&mut self, annotations: Vec<AnnotationUse>) -> Result<FnDecl> {
        let start = self.bump().span;
        let name_tok = self.tok();
        if name_tok.kind != T::Ident && !is_keyword(name_tok.kind) {
            return self.error_here(format!(
                "nom de fonction attendu après `def`, trouvé {}",
                self.found()
            ));
        }
        self.bump();
        let mut name_span = name_tok.span;
        // Setter : `def nom=(valeur)`, le `=` collé au nom.
        if self.at(T::Eq) && self.tok().span.start == name_tok.span.end && self.peek2() == T::LParen
        {
            name_span = name_span.to(self.bump().span);
        }
        let name = self.intern(name_span);
        let params = if self.at(T::LParen) {
            Some(self.params(T::LParen, T::RParen)?)
        } else {
            None
        };
        let ret = if self.eat(T::Colon) {
            Some(self.type_expr(true)?)
        } else {
            None
        };
        let body = if self.eat(T::Eq) {
            let e = self.expr()?;
            expr_block(e, self.span(e))
        } else {
            if !matches!(self.peek(), T::Newline | T::Semi) {
                return self.error_here(format!(
                    "fin de ligne ou `=` attendu après la signature de `{}`, trouvé {}",
                    self.interner.resolve(name),
                    self.found()
                ));
            }
            self.block_end("def", start)?
        };
        Ok(FnDecl {
            name,
            name_span,
            params,
            ret,
            body,
            annotations,
            span: self.span_from(start),
        })
    }

    /// Paramètres entre `open` et `close` : `(a: int, b = 0)` ou `|x, y|`.
    fn params(&mut self, open: TokenKind, close: TokenKind) -> Result<Vec<Param>> {
        self.expect(open, "")?;
        let in_lambda = close == T::Pipe;
        let mut params: Vec<Param> = Vec::new();
        loop {
            self.skip_newlines();
            if self.eat(close) {
                break;
            }
            let annotations = self.annotations()?;
            let start = self.tok().span;
            let (name, is_field) = match self.peek() {
                T::Ident => (self.intern(start), false),
                T::IVar if !in_lambda => (self.intern(Span::new(start.start + 1, start.end)), true),
                _ => {
                    return self
                        .error_here(format!("nom de paramètre attendu, trouvé {}", self.found()));
                }
            };
            self.bump();
            if params.iter().any(|p| p.name == name) {
                return self.error_at(
                    format!(
                        "paramètre `{}` déclaré deux fois",
                        self.interner.resolve(name)
                    ),
                    start,
                );
            }
            // Dans `|…|`, une union doit être entre parenthèses : `|x: (int | str)|`.
            let ty = if self.eat(T::Colon) {
                Some(self.type_expr(!in_lambda)?)
            } else {
                None
            };
            let default = if self.eat(T::Eq) {
                Some(self.expr()?)
            } else {
                None
            };
            params.push(Param {
                name,
                is_field,
                ty,
                default,
                annotations,
                span: self.span_from(start),
            });
            self.skip_newlines();
            if !self.eat(T::Comma) {
                self.skip_newlines();
                self.expect(close, "pour fermer les paramètres")?;
                break;
            }
        }
        Ok(params)
    }

    fn type_decl(&mut self, annotations: Vec<AnnotationUse>) -> Result<TypeDecl> {
        let kw = self.bump();
        let (kind, opener) = match kw.kind {
            T::Type => (TypeDeclKind::Type, "type"),
            T::Class => (TypeDeclKind::Class, "class"),
            _ => (TypeDeclKind::Annotation, "annotation"),
        };
        let (name, name_span) = self.ident(&format!("après `{opener}`"))?;
        let mut members: Vec<Member> = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.peek(), T::End | T::Eof) {
                break;
            }
            let annotations = self.annotations()?;
            let start = self.tok().span;
            let member = match (self.peek(), self.peek2()) {
                (T::Def, _) => Member::Method(self.def(annotations)?),
                (T::Ident, T::Colon | T::Eq) => {
                    let (name, _) = self.ident("")?;
                    let ty = if self.eat(T::Colon) {
                        Some(self.type_expr(true)?)
                    } else {
                        None
                    };
                    let default = if self.eat(T::Eq) {
                        Some(self.expr()?)
                    } else {
                        None
                    };
                    let span = self.span_from(start);
                    let dup = members
                        .iter()
                        .any(|m| matches!(m, Member::Field(g) if g.name == name));
                    if dup {
                        return self.error_at(
                            format!("champ `{}` déclaré deux fois", self.interner.resolve(name)),
                            span,
                        );
                    }
                    Member::Field(Field {
                        name,
                        ty,
                        default,
                        annotations,
                        span,
                    })
                }
                _ => Member::Stmt(self.stmt_with(annotations)?),
            };
            members.push(member);
            if !matches!(self.peek(), T::Newline | T::Semi | T::End) {
                return self.error_here(format!(
                    "fin de ligne attendue après le membre, trouvé {}",
                    self.found()
                ));
            }
        }
        self.end(opener, kw.span)?;
        Ok(TypeDecl {
            kind,
            name,
            name_span,
            members,
            annotations,
            span: self.span_from(kw.span),
        })
    }

    fn enum_decl(&mut self, annotations: Vec<AnnotationUse>) -> Result<EnumDecl> {
        let kw = self.bump();
        let (name, name_span) = self.ident("après `enum`")?;
        let repr = if self.eat(T::Colon) {
            Some(self.type_expr(true)?)
        } else {
            None
        };
        let mut variants: Vec<Variant> = Vec::new();
        let mut methods = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.peek(), T::End | T::Eof) {
                break;
            }
            let annotations = self.annotations()?;
            if self.at(T::Def) {
                methods.push(self.def(annotations)?);
            } else {
                let (vname, start) = self.ident("(variante de l'enum)")?;
                if variants.iter().any(|v| v.name == vname) {
                    return self.error_at(
                        format!(
                            "variante `{}` déclarée deux fois",
                            self.interner.resolve(vname)
                        ),
                        start,
                    );
                }
                let value = if self.eat(T::Eq) {
                    Some(self.expr()?)
                } else {
                    None
                };
                variants.push(Variant {
                    name: vname,
                    value,
                    annotations,
                    span: self.span_from(start),
                });
            }
            if !matches!(self.peek(), T::Newline | T::Semi | T::End) {
                return self.error_here(format!(
                    "fin de ligne attendue après la variante, trouvé {}",
                    self.found()
                ));
            }
        }
        self.end("enum", kw.span)?;
        Ok(EnumDecl {
            name,
            name_span,
            repr,
            variants,
            methods,
            annotations,
            span: self.span_from(kw.span),
        })
    }

    // ─── Types ───────────────────────────────────────────────────────────────

    /// `allow_union` est faux dans `|…|` où `|` fermerait la liste.
    fn type_expr(&mut self, allow_union: bool) -> Result<TypeExpr> {
        stacker::maybe_grow(RED_ZONE, STACK_GROW, || self.type_expr_inner(allow_union))
    }

    fn type_expr_inner(&mut self, allow_union: bool) -> Result<TypeExpr> {
        self.enter()?;
        let first = self.type_postfix()?;
        let ty = if allow_union && self.at(T::Pipe) {
            let start = first.span;
            let mut items = vec![first];
            while self.eat(T::Pipe) {
                items.push(self.type_postfix()?);
            }
            TypeExpr {
                kind: TypeKind::Union(items),
                span: self.span_from(start),
            }
        } else {
            first
        };
        self.depth -= 1;
        Ok(ty)
    }

    fn type_postfix(&mut self) -> Result<TypeExpr> {
        let mut ty = self.type_atom()?;
        while self.at(T::Question) {
            let end = self.bump().span;
            let span = ty.span.to(end);
            ty = TypeExpr {
                kind: TypeKind::Optional(Box::new(ty)),
                span,
            };
        }
        Ok(ty)
    }

    fn type_atom(&mut self) -> Result<TypeExpr> {
        let start = self.tok().span;
        let kind = match self.peek() {
            T::Ident => {
                let mut path = vec![self.ident("")?.0];
                while self.eat(T::ColonColon) {
                    path.push(self.ident("après `::`")?.0);
                }
                TypeKind::Named(path)
            }
            T::Nil => {
                self.bump();
                TypeKind::Named(vec![self.interner.intern("nil")])
            }
            T::Str => {
                let t = self.bump();
                TypeKind::LitStr(self.str_content(t)?)
            }
            T::Int | T::Minus => {
                let neg = self.eat(T::Minus);
                let t = self.expect(T::Int, "après `-`")?;
                TypeKind::LitInt(self.int_value(t, neg)?)
            }
            T::LBracket => {
                self.bump();
                let inner = self.type_expr(true)?;
                self.expect(T::RBracket, "pour fermer le type liste `[T]`")?;
                TypeKind::List(Box::new(inner))
            }
            T::LBrace => {
                self.bump();
                if self.peek() == T::Ident && self.peek2() == T::Colon {
                    let mut fields: Vec<(Symbol, TypeExpr)> = Vec::new();
                    loop {
                        self.skip_newlines();
                        if self.at(T::RBrace) {
                            break;
                        }
                        let (name, _) = self.ident("(champ du record)")?;
                        self.expect(T::Colon, "après le nom du champ")?;
                        fields.push((name, self.type_expr(true)?));
                        self.skip_newlines();
                        if !self.eat(T::Comma) {
                            break;
                        }
                    }
                    self.skip_newlines();
                    self.expect(T::RBrace, "pour fermer le type record")?;
                    TypeKind::Record(fields)
                } else {
                    let k = self.type_expr(true)?;
                    self.expect(
                        T::FatArrow,
                        "dans le type hash `{K => V}` (pour un record : `{nom: T}`)",
                    )?;
                    let v = self.type_expr(true)?;
                    self.expect(T::RBrace, "pour fermer le type hash")?;
                    TypeKind::Hash(Box::new(k), Box::new(v))
                }
            }
            T::LParen => {
                self.bump();
                let mut items = Vec::new();
                let mut trailing_comma = false;
                while !self.at(T::RParen) {
                    items.push(self.type_expr(true)?);
                    trailing_comma = self.eat(T::Comma);
                    if !trailing_comma {
                        break;
                    }
                }
                self.expect(T::RParen, "pour fermer le type")?;
                if self.eat(T::Colon) {
                    let ret = self.type_postfix()?;
                    TypeKind::Fn(items, Box::new(ret))
                } else if items.len() == 1 && !trailing_comma {
                    // Simple parenthésage : `(int | str)?`
                    return Ok(items.pop().unwrap());
                } else {
                    TypeKind::Tuple(items)
                }
            }
            _ => return self.error_here(format!("type attendu, trouvé {}", self.found())),
        };
        Ok(TypeExpr {
            kind,
            span: self.span_from(start),
        })
    }

    // ─── Expressions ─────────────────────────────────────────────────────────

    fn expr(&mut self) -> Result<ExprId> {
        stacker::maybe_grow(RED_ZONE, STACK_GROW, || self.expr_inner())
    }

    fn expr_inner(&mut self) -> Result<ExprId> {
        self.enter()?;
        let lhs = self.binary(0)?;
        let e = match assign_op(self.peek()) {
            Some(op) => {
                if !self.ast.is_assignable(lhs) {
                    return self
                        .error_at("on ne peut pas assigner à cette expression", self.span(lhs));
                }
                self.bump();
                let value = self.expr()?;
                let span = self.span(lhs).to(self.span(value));
                self.mk(
                    ExprKind::Assign {
                        target: lhs,
                        op,
                        value,
                    },
                    span,
                )
            }
            None => lhs,
        };
        self.depth -= 1;
        Ok(e)
    }

    fn binary(&mut self, min_bp: u8) -> Result<ExprId> {
        let mut lhs = self.prefix()?;
        loop {
            let kind = self.peek();
            let Some((l_bp, r_bp, op)) = infix_bp(kind) else {
                break;
            };
            if l_bp < min_bp {
                break;
            }
            let op_span = self.bump().span;
            let rhs = self.binary(r_bp)?;
            let span = self.span(lhs).to(self.span(rhs));
            let node = match op {
                Infix::Bin(op) => ExprKind::Binary(op, lhs, rhs),
                Infix::Range(inclusive) => ExprKind::Range {
                    lo: lhs,
                    hi: rhs,
                    inclusive,
                },
            };
            lhs = self.mk(node, span);
            // Comparaisons et intervalles ne s'enchaînent pas : `a < b < c` est refusé.
            let next = self.peek();
            let chained = (is_comparison(kind) && is_comparison(next))
                || (matches!(kind, T::DotDot | T::DotDotDot)
                    && matches!(next, T::DotDot | T::DotDotDot));
            if chained {
                return self.error_at(
                    format!(
                        "{} ne peut pas suivre directement {} : ajouter des parenthèses",
                        next.describe(),
                        kind.describe()
                    ),
                    op_span.to(self.tok().span),
                );
            }
        }
        Ok(lhs)
    }

    fn prefix(&mut self) -> Result<ExprId> {
        let start = self.tok().span;
        match self.peek() {
            T::Minus => {
                self.bump();
                // `-5` est un littéral (permet i64::MIN), sauf `-5.abs()` = `-(5.abs())`.
                if self.at(T::Int) && !matches!(self.peek2(), T::Dot | T::LBracket | T::LParen) {
                    let t = self.bump();
                    let n = self.int_value(t, true)?;
                    return Ok(self.mk(ExprKind::Int(n), start.to(t.span)));
                }
                let operand = self.binary(BP_NEG)?;
                let span = start.to(self.span(operand));
                Ok(self.mk(ExprKind::Unary(UnaryOp::Neg, operand), span))
            }
            T::Not => {
                self.bump();
                let operand = self.binary(BP_NOT)?;
                let span = start.to(self.span(operand));
                Ok(self.mk(ExprKind::Unary(UnaryOp::Not, operand), span))
            }
            T::Pipe => {
                let params = self.params(T::Pipe, T::Pipe)?;
                let body = self.expr()?;
                let body_span = self.span(body);
                let span = start.to(body_span);
                let id = self.ast.push_lambda(Lambda {
                    params,
                    body: expr_block(body, body_span),
                    span,
                });
                Ok(self.mk(ExprKind::Lambda(id), span))
            }
            _ => {
                let e = self.primary()?;
                self.postfix(e)
            }
        }
    }

    fn postfix(&mut self, mut e: ExprId) -> Result<ExprId> {
        loop {
            match self.peek() {
                T::LParen => {
                    let args = self.args()?;
                    let span = self.span_from(self.span(e));
                    match self.ast.kind(e) {
                        // `recv.nom(args)` : le champ devient un appel de méthode.
                        ExprKind::Field(recv, name) => {
                            *self.ast.expr_mut(e) = Expr {
                                kind: ExprKind::MethodCall {
                                    recv,
                                    name,
                                    args,
                                    block: None,
                                },
                                span,
                            };
                        }
                        _ => {
                            e = self.mk(
                                ExprKind::Call {
                                    callee: e,
                                    args,
                                    block: None,
                                },
                                span,
                            );
                        }
                    }
                }
                T::Dot => {
                    self.bump();
                    let t = self.tok();
                    let span = self.span(e).to(t.span);
                    if t.kind == T::Int {
                        self.bump();
                        let idx = t
                            .span
                            .text(self.src)
                            .replace('_', "")
                            .parse()
                            .map_err(|_| Error::syntax("index de tuple invalide", t.span))?;
                        e = self.mk(ExprKind::TupleIndex(e, idx), span);
                    } else if t.kind == T::Ident || is_keyword(t.kind) {
                        self.bump();
                        let name = self.intern(t.span);
                        e = self.mk(ExprKind::Field(e, name), span);
                    } else {
                        return self.error_here(format!(
                            "nom de méthode ou de champ attendu après `.`, trouvé {}",
                            self.found()
                        ));
                    }
                }
                T::ColonColon => {
                    let ExprKind::Ident(first) = self.ast.kind(e) else {
                        return self.error_at("`::` doit suivre un nom de type", self.span(e));
                    };
                    let mut path = vec![first];
                    while self.eat(T::ColonColon) {
                        path.push(self.ident("après `::`")?.0);
                    }
                    let span = self.span_from(self.span(e));
                    let path = self.ast.alloc(&path);
                    *self.ast.expr_mut(e) = Expr {
                        kind: ExprKind::Path(path),
                        span,
                    };
                }
                T::LBracket => {
                    self.bump();
                    self.skip_newlines();
                    let index = self.expr()?;
                    self.skip_newlines();
                    self.expect(T::RBracket, "pour fermer l'index")?;
                    let span = self.span_from(self.span(e));
                    e = self.mk(ExprKind::Index(e, index), span);
                }
                T::Do if !self.no_do => {
                    let do_span = self.tok().span;
                    let lambda = self.do_block()?;
                    let block = Some(self.ast.push_lambda(lambda));
                    let span = self.span_from(self.span(e));
                    let kind = match self.ast.kind(e) {
                        ExprKind::Call {
                            callee,
                            args,
                            block: None,
                        } => ExprKind::Call {
                            callee,
                            args,
                            block,
                        },
                        ExprKind::MethodCall {
                            recv,
                            name,
                            args,
                            block: None,
                        } => ExprKind::MethodCall {
                            recv,
                            name,
                            args,
                            block,
                        },
                        ExprKind::Field(recv, name) => ExprKind::MethodCall {
                            recv,
                            name,
                            args: Slice::EMPTY,
                            block,
                        },
                        ExprKind::Ident(_) | ExprKind::Path(_) => {
                            e = self.mk(
                                ExprKind::Call {
                                    callee: e,
                                    args: Slice::EMPTY,
                                    block,
                                },
                                span,
                            );
                            continue;
                        }
                        _ => {
                            return self.error_at(
                                "un bloc `do` doit suivre un appel de fonction ou de méthode",
                                do_span,
                            );
                        }
                    };
                    *self.ast.expr_mut(e) = Expr { kind, span };
                }
                _ => return Ok(e),
            }
        }
    }

    /// `do |params| … end`
    fn do_block(&mut self) -> Result<Lambda> {
        let start = self.bump().span;
        let params = if self.at(T::Pipe) {
            self.params(T::Pipe, T::Pipe)?
        } else {
            Vec::new()
        };
        let saved = std::mem::replace(&mut self.no_do, false);
        let body = self.block_end("do", start);
        self.no_do = saved;
        Ok(Lambda {
            params,
            body: body?,
            span: self.span_from(start),
        })
    }

    /// `(a, nom: b)`
    fn args(&mut self) -> Result<Slice<Arg>> {
        self.bump();
        let saved = std::mem::replace(&mut self.no_do, false);
        let base = self.s_args.len();
        let r = self.args_inner(base);
        self.no_do = saved;
        r?;
        let slice = self.ast.alloc(&self.s_args[base..]);
        self.s_args.truncate(base);
        Ok(slice)
    }

    fn args_inner(&mut self, base: usize) -> Result<()> {
        loop {
            self.skip_newlines();
            if self.eat(T::RParen) {
                return Ok(());
            }
            let name = if self.peek() == T::Ident && self.peek2() == T::Colon {
                let t = self.bump();
                self.bump();
                let name = self.intern(t.span);
                if self.s_args[base..].iter().any(|a| a.name == Some(name)) {
                    return self.error_at(
                        format!("argument `{}` donné deux fois", self.interner.resolve(name)),
                        t.span,
                    );
                }
                Some(name)
            } else {
                if self.s_args[base..].last().is_some_and(|a| a.name.is_some()) {
                    return self.error_here("argument positionnel après un argument nommé");
                }
                None
            };
            let value = self.expr()?;
            self.s_args.push(Arg { name, value });
            self.skip_newlines();
            if !self.eat(T::Comma) {
                self.skip_newlines();
                self.expect(T::RParen, "pour fermer les arguments")?;
                return Ok(());
            }
        }
    }

    fn primary(&mut self) -> Result<ExprId> {
        let t = self.tok();
        let kind = match t.kind {
            T::Int => {
                self.bump();
                ExprKind::Int(self.int_value(t, false)?)
            }
            T::Float => {
                self.bump();
                let text = t.span.text(self.src).replace('_', "");
                ExprKind::Float(
                    text.parse()
                        .map_err(|_| Error::syntax("nombre décimal invalide", t.span))?,
                )
            }
            T::Str => {
                self.bump();
                ExprKind::Str(self.str_symbol(t)?)
            }
            T::StrHead => return self.interpolation(),
            T::True => {
                self.bump();
                ExprKind::Bool(true)
            }
            T::False => {
                self.bump();
                ExprKind::Bool(false)
            }
            T::Nil => {
                self.bump();
                ExprKind::Nil
            }
            T::SelfKw => {
                self.bump();
                ExprKind::SelfRef
            }
            T::Ident => {
                self.bump();
                ExprKind::Ident(self.intern(t.span))
            }
            T::IVar => {
                self.bump();
                ExprKind::IVar(self.intern(Span::new(t.span.start + 1, t.span.end)))
            }
            T::LParen => return self.paren(),
            T::LBracket => {
                self.bump();
                let saved = std::mem::replace(&mut self.no_do, false);
                let items = self.comma_list(T::RBracket, "pour fermer la liste");
                self.no_do = saved;
                ExprKind::List(items?)
            }
            T::LBrace => return self.brace(),
            T::If => return self.if_expr(),
            T::Unless => return self.unless_expr(),
            T::Case => return self.case_expr(),
            T::While => return self.while_expr(),
            T::For => return self.for_expr(),
            T::Return => {
                self.bump();
                let value = if ends_expr(self.peek()) {
                    None
                } else {
                    Some(self.expr()?)
                };
                ExprKind::Return(value)
            }
            T::Break => {
                self.bump();
                ExprKind::Break
            }
            T::Next => {
                self.bump();
                ExprKind::Next
            }
            T::Do if self.no_do => {
                return self.error_here(
                    "`do` n'est pas utilisé ici : passer à la ligne après la condition",
                );
            }
            T::Let => {
                return self.error_here(
                    "`let` est une instruction : il ne peut pas être dans une expression",
                );
            }
            _ => return self.error_here(format!("expression attendue, trouvé {}", self.found())),
        };
        let span = self.span_from(t.span);
        Ok(self.mk(kind, span))
    }

    /// Éléments séparés par des virgules jusqu'à `close` (virgule finale permise).
    fn comma_list(&mut self, close: TokenKind, context: &str) -> Result<Slice<ExprId>> {
        let base = self.s_exprs.len();
        self.comma_items(close, context)?;
        Ok(self.take_exprs(base))
    }

    /// Empile les éléments sur `s_exprs` jusqu'à `close` (consommé).
    fn comma_items(&mut self, close: TokenKind, context: &str) -> Result<()> {
        loop {
            self.skip_newlines();
            if self.eat(close) {
                return Ok(());
            }
            let e = self.expr()?;
            self.s_exprs.push(e);
            self.skip_newlines();
            if !self.eat(T::Comma) {
                self.skip_newlines();
                self.expect(close, context)?;
                return Ok(());
            }
        }
    }

    /// Retire de `s_exprs` les éléments empilés depuis `base` et les range dans l'AST.
    fn take_exprs(&mut self, base: usize) -> Slice<ExprId> {
        let slice = self.ast.alloc(&self.s_exprs[base..]);
        self.s_exprs.truncate(base);
        slice
    }

    /// `(e)`, `()`, `(a, b)`, `(a,)`
    fn paren(&mut self) -> Result<ExprId> {
        let start = self.bump().span;
        let saved = std::mem::replace(&mut self.no_do, false);
        let r = self.paren_inner(start);
        self.no_do = saved;
        r
    }

    fn paren_inner(&mut self, start: Span) -> Result<ExprId> {
        self.skip_newlines();
        if self.eat(T::RParen) {
            let span = self.span_from(start);
            return Ok(self.mk(ExprKind::Tuple(Slice::EMPTY), span));
        }
        let first = self.expr()?;
        self.skip_newlines();
        if self.eat(T::Comma) {
            // `(a, b, …)` : `first` est déjà lu, le reste s'empile à sa suite.
            let base = self.s_exprs.len();
            self.s_exprs.push(first);
            self.comma_items(T::RParen, "pour fermer le tuple")?;
            let items = self.take_exprs(base);
            let span = self.span_from(start);
            return Ok(self.mk(ExprKind::Tuple(items), span));
        }
        self.expect(T::RParen, "pour fermer la parenthèse")?;
        Ok(first)
    }

    /// `{}` (hash vide), `{k => v}` (hash) ou `{nom: v, autre}` (record).
    fn brace(&mut self) -> Result<ExprId> {
        let start = self.bump().span;
        let saved = std::mem::replace(&mut self.no_do, false);
        let r = self.brace_inner(start);
        self.no_do = saved;
        r
    }

    fn brace_inner(&mut self, start: Span) -> Result<ExprId> {
        self.skip_newlines();
        if self.eat(T::RBrace) {
            let span = self.span_from(start);
            return Ok(self.mk(ExprKind::Hash(Slice::EMPTY), span));
        }
        let is_record = self.peek() == T::Ident
            && matches!(self.peek2(), T::Colon | T::Comma | T::RBrace | T::Newline);
        if is_record {
            let base = self.s_fields.len();
            loop {
                self.skip_newlines();
                if self.eat(T::RBrace) {
                    break;
                }
                let (name, span) = self.ident("(champ du record)")?;
                if self.s_fields[base..].iter().any(|(n, _)| *n == name) {
                    return self.error_at(
                        format!("champ `{}` donné deux fois", self.interner.resolve(name)),
                        span,
                    );
                }
                let value = if self.eat(T::Colon) {
                    self.expr()?
                } else {
                    // Raccourci `{x}` = `{x: x}`
                    self.mk(ExprKind::Ident(name), span)
                };
                self.s_fields.push((name, value));
                self.skip_newlines();
                if !self.eat(T::Comma) {
                    self.skip_newlines();
                    self.expect(T::RBrace, "pour fermer le record")?;
                    break;
                }
            }
            let fields = self.ast.alloc(&self.s_fields[base..]);
            self.s_fields.truncate(base);
            let span = self.span_from(start);
            return Ok(self.mk(ExprKind::Record(fields), span));
        }
        let base = self.s_entries.len();
        loop {
            self.skip_newlines();
            if self.eat(T::RBrace) {
                break;
            }
            let key = self.expr()?;
            self.expect(
                T::FatArrow,
                "entre la clé et la valeur (pour un record : `{nom: valeur}`)",
            )?;
            let value = self.expr()?;
            self.s_entries.push((key, value));
            self.skip_newlines();
            if !self.eat(T::Comma) {
                self.skip_newlines();
                self.expect(T::RBrace, "pour fermer le hash")?;
                break;
            }
        }
        let entries = self.ast.alloc(&self.s_entries[base..]);
        self.s_entries.truncate(base);
        let span = self.span_from(start);
        Ok(self.mk(ExprKind::Hash(entries), span))
    }

    fn interpolation(&mut self) -> Result<ExprId> {
        let head = self.bump();
        let base = self.s_parts.len();
        let lit = self.str_content(head)?;
        if !lit.is_empty() {
            let sym = self.interner.intern(&lit);
            self.s_parts.push(StrPart::Lit(sym));
        }
        let saved = std::mem::replace(&mut self.no_do, false);
        let r = self.interpolation_inner();
        self.no_do = saved;
        r?;
        let parts = self.ast.alloc(&self.s_parts[base..]);
        self.s_parts.truncate(base);
        let span = self.span_from(head.span);
        Ok(self.mk(ExprKind::Interp(parts), span))
    }

    fn interpolation_inner(&mut self) -> Result<()> {
        loop {
            self.skip_newlines();
            let e = self.expr()?;
            self.s_parts.push(StrPart::Expr(e));
            self.skip_newlines();
            let t = self.tok();
            if !matches!(t.kind, T::StrMid | T::StrTail) {
                return self.error_here(format!(
                    "`}}` attendu pour fermer l'interpolation, trouvé {}",
                    self.found()
                ));
            }
            self.bump();
            let lit = self.str_content(t)?;
            if !lit.is_empty() {
                let sym = self.interner.intern(&lit);
                self.s_parts.push(StrPart::Lit(sym));
            }
            if t.kind == T::StrTail {
                return Ok(());
            }
        }
    }

    fn if_expr(&mut self) -> Result<ExprId> {
        let start = self.bump().span;
        self.if_tail(start, "if")
    }

    /// Suite d'un `if` / `elsif` après le mot-clé.
    fn if_tail(&mut self, start: Span, opener: &str) -> Result<ExprId> {
        let cond = self.condition(opener)?;
        let then = self.block(&[T::Elsif, T::Else, T::End])?;
        let else_ = match self.peek() {
            T::Elsif => {
                let s = self.bump().span;
                let nested = self.if_tail(s, "elsif")?;
                Some(expr_block(nested, self.span(nested)))
            }
            T::Else => {
                self.bump();
                Some(self.block_end(opener, start)?)
            }
            _ => {
                self.end(opener, start)?;
                None
            }
        };
        let span = self.span_from(start);
        Ok(self
            .ast
            .push_control(Control::If(If { cond, then, else_ }), span))
    }

    /// Condition suivie de `then` ou d'une fin de ligne.
    fn condition(&mut self, opener: &str) -> Result<ExprId> {
        let cond = self.expr()?;
        if !self.eat(T::Then) && !matches!(self.peek(), T::Newline | T::Semi) {
            return self.error_here(format!(
                "`then` ou fin de ligne attendu après la condition du `{opener}`, trouvé {}",
                self.found()
            ));
        }
        Ok(cond)
    }

    fn unless_expr(&mut self) -> Result<ExprId> {
        let start = self.bump().span;
        let cond = self.condition("unless")?;
        let then = self.block(&[T::Else, T::End])?;
        let else_ = if self.eat(T::Else) {
            Some(self.block_end("unless", start)?)
        } else {
            self.end("unless", start)?;
            None
        };
        let cond_span = self.span(cond);
        let cond = self.mk(ExprKind::Unary(UnaryOp::Not, cond), cond_span);
        let span = self.span_from(start);
        Ok(self
            .ast
            .push_control(Control::If(If { cond, then, else_ }), span))
    }

    fn case_expr(&mut self) -> Result<ExprId> {
        let start = self.bump().span;
        let subject = self.expr()?;
        self.skip_newlines();
        let mut arms = Vec::new();
        while self.at(T::When) {
            let arm_start = self.bump().span;
            let base = self.s_exprs.len();
            loop {
                let p = self.expr()?;
                self.s_exprs.push(p);
                if !self.eat(T::Comma) {
                    break;
                }
            }
            let patterns = self.take_exprs(base);
            if !self.eat(T::Then) && !matches!(self.peek(), T::Newline | T::Semi) {
                return self.error_here(format!(
                    "`then` ou fin de ligne attendu après `when`, trouvé {}",
                    self.found()
                ));
            }
            let body = self.block(&[T::When, T::Else, T::End])?;
            arms.push(WhenArm {
                patterns,
                body,
                span: self.span_from(arm_start),
            });
        }
        if arms.is_empty() {
            return self.error_here(format!(
                "`when` attendu après `case`, trouvé {}",
                self.found()
            ));
        }
        let else_ = if self.eat(T::Else) {
            Some(self.block_end("case", start)?)
        } else {
            self.end("case", start)?;
            None
        };
        let span = self.span_from(start);
        Ok(self.ast.push_control(
            Control::Case(Case {
                subject,
                arms,
                else_,
            }),
            span,
        ))
    }

    /// Expression d'en-tête de boucle : `do` interdit, fin de ligne obligatoire.
    fn loop_head(&mut self, opener: &str) -> Result<ExprId> {
        let saved = std::mem::replace(&mut self.no_do, true);
        let e = self.expr();
        self.no_do = saved;
        let e = e?;
        if self.at(T::Do) {
            return self.error_here(format!(
                "`do` n'est pas utilisé après `{opener}` : passer à la ligne"
            ));
        }
        if !matches!(self.peek(), T::Newline | T::Semi) {
            return self.error_here(format!(
                "fin de ligne attendue après l'en-tête du `{opener}`, trouvé {}",
                self.found()
            ));
        }
        Ok(e)
    }

    fn while_expr(&mut self) -> Result<ExprId> {
        let start = self.bump().span;
        let cond = self.loop_head("while")?;
        let body = self.block_end("while", start)?;
        let span = self.span_from(start);
        Ok(self
            .ast
            .push_control(Control::While(While { cond, body }), span))
    }

    fn for_expr(&mut self) -> Result<ExprId> {
        let start = self.bump().span;
        let mut vars = vec![self.ident("après `for`")?];
        while self.eat(T::Comma) {
            vars.push(self.ident("après `,`")?);
        }
        self.expect(T::In, "après les variables du `for`")?;
        let iter = self.loop_head("for")?;
        let body = self.block_end("for", start)?;
        let span = self.span_from(start);
        Ok(self
            .ast
            .push_control(Control::For(For { vars, iter, body }), span))
    }

    // ─── Littéraux ───────────────────────────────────────────────────────────

    fn int_value(&self, t: Token, negative: bool) -> Result<i64> {
        let too_big = || Error::syntax("entier trop grand (64 bits max)", t.span);
        let mut n: u64 = 0;
        for b in t.span.text(self.src).bytes() {
            if b != b'_' {
                n = n
                    .checked_mul(10)
                    .and_then(|n| n.checked_add((b - b'0') as u64))
                    .ok_or_else(too_big)?;
            }
        }
        if negative {
            if n > i64::MAX as u64 + 1 {
                return Err(too_big());
            }
            Ok((n as i64).wrapping_neg())
        } else {
            i64::try_from(n).map_err(|_| too_big())
        }
    }

    /// Intervalle du contenu d'un jeton de chaîne, sans délimiteurs.
    fn str_inner(t: Token) -> Span {
        let (skip_start, skip_end) = match t.kind {
            T::Str | T::StrTail => (1, 1),
            T::StrHead | T::StrMid => (1, 2),
            _ => unreachable!(),
        };
        Span::new(t.span.start + skip_start, t.span.end - skip_end)
    }

    /// Contenu d'un jeton de chaîne, échappements traités.
    fn str_content(&self, t: Token) -> Result<String> {
        let inner = Self::str_inner(t);
        unescape(inner.text(self.src), inner.start)
    }

    /// Chaîne littérale internée ; sans échappement, aucune allocation.
    fn str_symbol(&mut self, t: Token) -> Result<Symbol> {
        let inner = Self::str_inner(t);
        let src = self.src;
        let text = inner.text(src);
        if text.contains('\\') {
            let s = unescape(text, inner.start)?;
            Ok(self.interner.intern(&s))
        } else {
            Ok(self.interner.intern(text))
        }
    }
}

/// Traite `\n \t \r \0 \\ \" \# \u{…}`. `offset` sert aux messages d'erreur.
fn unescape(s: &str, offset: u32) -> Result<String> {
    if !s.contains('\\') {
        return Ok(s.to_string());
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.char_indices();
    while let Some((i, c)) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        let at = |len: usize| Span::new(offset + i as u32, offset + (i + len) as u32);
        match chars.next() {
            Some((_, 'n')) => out.push('\n'),
            Some((_, 't')) => out.push('\t'),
            Some((_, 'r')) => out.push('\r'),
            Some((_, '0')) => out.push('\0'),
            Some((_, '\\')) => out.push('\\'),
            Some((_, '"')) => out.push('"'),
            Some((_, '#')) => out.push('#'),
            Some((_, '\n')) => {} // `\` en fin de ligne : la chaîne continue sans saut de ligne
            Some((_, 'u')) => {
                let rest = &s[i + 2..];
                let Some(close) = rest.strip_prefix('{').and_then(|r| r.find('}')) else {
                    return Err(Error::syntax("`\\u{…}` attendu", at(2)));
                };
                let hex = &rest[1..close + 1];
                let c = u32::from_str_radix(hex, 16)
                    .ok()
                    .and_then(char::from_u32)
                    .ok_or_else(|| Error::syntax("code Unicode invalide", at(close + 4)))?;
                out.push(c);
                for _ in 0..close + 2 {
                    chars.next();
                }
            }
            Some((_, other)) => {
                return Err(Error::syntax(
                    format!("échappement inconnu `\\{other}`"),
                    at(1 + other.len_utf8()),
                ));
            }
            None => return Err(Error::syntax("`\\` en fin de chaîne", at(1))),
        }
    }
    Ok(out)
}
