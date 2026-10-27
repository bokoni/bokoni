//! Hand-written parser: recursive descent, and Pratt parsing for operators.
//!
//! - Tokens are read on demand (a single token of lookahead), never backtracking: each
//!   token is read once.
//! - Expressions are appended to the compact [`Ast`]; lists (arguments, elements…) are
//!   accumulated on reused temporary stacks, then copied in one go into the AST pools:
//!   no allocation per list.
//!
//! Precedence (from lowest to highest):
//!
//! | level | operators                      | associativity |
//! |-------|--------------------------------|---------------|
//! | 1     | `=` `+=` `-=` …                | right         |
//! | 2     | `or`                           | left          |
//! | 3     | `and`                          | left          |
//! | 4     | `not` (prefix)                 |               |
//! | 5     | `==` `!=` `<` `<=` `>` `>=`    | none          |
//! | 6     | `..` `...`                     | none          |
//! | 7     | `<<`                           | left          |
//! | 8     | `+` `-`                        | left          |
//! | 9     | `*` `/` `%`                    | left          |
//! | 10    | `-` (prefix)                   |               |
//! | 11    | `f()` `x.y` `x[i]` `A::B` `do` | left          |

use crate::ast::*;
use crate::diagnostic::{Error, Errors, Result};
use crate::intern::{Interner, Symbol};
use crate::lexer::{Lexer, Token, TokenKind};
use crate::span::{Span, line_col};

use TokenKind as T;

/// Maximum nesting depth.
const MAX_DEPTH: u32 = 1000;

/// When the remaining stack drops below `RED_ZONE`, recursion continues on a new
/// heap-allocated stack of `STACK_GROW`: no nesting can overflow the calling thread's
/// stack.
const RED_ZONE: usize = 128 * 1024;
const STACK_GROW: usize = 2 * 1024 * 1024;

/// Maximum number of errors reported in one pass.
const MAX_ERRORS: usize = 50;

/// Parses a program. On error, parsing resumes at the next statement: all the errors in
/// the file are returned, sorted by position.
pub fn parse(src: &str, interner: &mut Interner) -> std::result::Result<Program, Errors> {
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
        errors: Vec::new(),
        abort: false,
    };
    // When parsing is aborted (`Err`), the error is already in `p.errors`.
    let body = p.block(&[]).ok();
    let mut errors = std::mem::take(&mut p.errors);
    // A parser error that follows a lexer error on the same line is a consequence of it
    // (the offending character was skipped): it is not shown.
    let lexer_errors = std::mem::take(&mut p.lexer.errors);
    let line_of = |e: &Error| e.span.map(|s| line_col(src, s.start).0);
    errors.retain(|e| {
        !lexer_errors
            .iter()
            .any(|l| line_of(l) == line_of(e) && l.span.map(|s| s.start) < e.span.map(|s| s.start))
    });
    errors.extend(lexer_errors);
    // After a fatal lexer error (unclosed string…), the parser only saw an `Eof`: its
    // errors located further on are meaningless.
    if let Some(fatal) = p.lexer.error.take() {
        let at = fatal.span.map_or(0, |s| s.start);
        errors.retain(|e| !e.incomplete && e.span.is_some_and(|s| s.start < at));
        errors.push(fatal);
    }
    match body {
        Some(body) if errors.is_empty() => Ok(Program { body, ast: p.ast }),
        _ => Err(Errors::new(errors)),
    }
}

struct Parser<'s, 'i> {
    src: &'s str,
    /// Tokens read on demand: only the current token and one lookahead token exist.
    lexer: Lexer<'s>,
    cur: Token,
    ahead: Option<Token>,
    /// End of the last consumed token.
    prev_end: u32,
    interner: &'i mut Interner,
    ast: Ast,
    // Temporary stacks for lists under construction (nesting: each list records its
    // starting height and only removes its own elements).
    s_exprs: Vec<ExprId>,
    s_args: Vec<Arg>,
    s_fields: Vec<(Symbol, ExprId)>,
    s_entries: Vec<(ExprId, ExprId)>,
    s_parts: Vec<StrPart>,
    depth: u32,
    /// In a `while` / `for` header, a `do` does not attach to the expression.
    no_do: bool,
    /// Reported errors; parsing goes on after each one.
    errors: Vec<Error>,
    /// Parsing stopped (premature end of source, too many errors).
    abort: bool,
}

/// State to restore when a statement fails.
struct Snapshot {
    depth: u32,
    no_do: bool,
    lens: [usize; 5],
}

/// Tracks open constructs (`if`, `def`, `do`… up to their `end`) in a token stream: used
/// to cleanly skip an erroneous statement.
#[derive(Default)]
struct Tracker {
    stack: Vec<Frame>,
    brackets: i32,
    prev: Option<Token>,
}

#[derive(Clone, Copy, PartialEq)]
enum Frame {
    /// Construct closed by `end`.
    Block,
    /// Signature of a `def`: becomes `Block`, or `Endless` (`def f = …`, without `end`).
    DefSig {
        name_end: Option<u32>,
        brackets: i32,
    },
    /// `def f = …`: ends at the line end.
    Endless { brackets: i32 },
}

impl Tracker {
    fn depth(&self) -> usize {
        self.stack.len()
    }

    /// `if` / `unless` / `while` open a construct, except as modifiers (`x = 0 if c`),
    /// that is after a token that ends an expression.
    fn opens(&self, kind: TokenKind) -> bool {
        match kind {
            T::Def
            | T::Type
            | T::Class
            | T::Enum
            | T::Annotation
            | T::Macro
            | T::Case
            | T::For
            | T::Begin
            | T::Do => true,
            T::If | T::Unless | T::While => !self.prev.is_some_and(|p| {
                matches!(
                    p.kind,
                    T::Ident
                        | T::IVar
                        | T::Int
                        | T::Float
                        | T::Str
                        | T::StrTail
                        | T::RParen
                        | T::RBracket
                        | T::RBrace
                        | T::True
                        | T::False
                        | T::Nil
                        | T::SelfKw
                        | T::End
                        | T::Break
                        | T::Next
                        | T::Return
                )
            }),
            _ => false,
        }
    }

    fn feed(&mut self, t: Token) {
        let top = self.stack.last().copied();
        match t.kind {
            T::LParen | T::LBracket | T::LBrace | T::AtLBracket | T::StrHead => self.brackets += 1,
            T::RParen | T::RBracket | T::RBrace | T::StrTail => {
                self.brackets = (self.brackets - 1).max(0)
            }
            T::End => {
                self.stack.pop();
            }
            T::Newline | T::Semi => match top {
                Some(Frame::DefSig { brackets, .. }) if brackets == self.brackets => {
                    *self.stack.last_mut().unwrap() = Frame::Block;
                }
                Some(Frame::Endless { brackets }) if brackets == self.brackets => {
                    self.stack.pop();
                }
                _ => {}
            },
            // `=` in a `def` signature: one-line body, no `end` (except the `=` attached
            // to a setter name, `def name=(v)`).
            T::Eq => {
                if let Some(Frame::DefSig { name_end, brackets }) = top
                    && brackets == self.brackets
                    && name_end != Some(t.span.start)
                {
                    *self.stack.last_mut().unwrap() = Frame::Endless { brackets };
                }
            }
            T::Def => self.stack.push(Frame::DefSig {
                name_end: None,
                brackets: self.brackets,
            }),
            k if self.opens(k) => self.stack.push(Frame::Block),
            _ => {}
        }
        // Name of the `def`: the token after `def`.
        if self.prev.is_some_and(|p| p.kind == T::Def)
            && let Some(Frame::DefSig { name_end, .. }) = self.stack.last_mut()
        {
            *name_end = Some(t.span.end);
        }
        self.prev = Some(t);
    }
}

#[derive(Clone, Copy)]
enum Infix {
    Bin(BinOp),
    Range(bool),
}

/// Binding power of binary operators: (left, right).
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

/// Tokens that cannot start an expression: `return` without a value before them.
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
            | T::Rescue
            | T::Ensure
            | T::RParen
            | T::RBracket
            | T::RBrace
            | T::Comma
            | T::Pipe
    )
}

/// Keywords accepted as method or field names after `.` (`x.type`, `x.class`).
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
            | T::Begin
            | T::Rescue
            | T::Ensure
            | T::Raise
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

/// Block made of a single expression statement.
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
    // ─── Tokens ──────────────────────────────────────────────────────────────

    #[inline]
    fn peek(&self) -> TokenKind {
        self.cur.kind
    }

    /// Kind of the token after the current one.
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
            // Line ends do not count: `prev_end` stays on the last meaningful token.
            if !matches!(t.kind, T::Newline | T::Semi) {
                self.prev_end = t.span.end;
            }
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

    // ─── Errors ──────────────────────────────────────────────────────────────

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
        // At the end of the source, the error is placed right after the last token read
        // (where something is missing) rather than on the last line, often empty.
        let span = if self.at(T::Eof) {
            Span::new(self.prev_end, self.prev_end)
        } else {
            self.tok().span
        };
        let mut e = Error::syntax(msg, span);
        e.incomplete = self.at(T::Eof);
        Err(e)
    }

    /// Consumes a token of the expected kind, or produces "expected X `context`, found Y".
    fn expect(&mut self, kind: TokenKind, context: &str) -> Result<Token> {
        if self.at(kind) {
            return Ok(self.bump());
        }
        let sep = if context.is_empty() { "" } else { " " };
        self.error_here(format!(
            "expected {}{sep}{context}, found {}",
            kind.describe(),
            self.found()
        ))
    }

    /// The current token is on another line than the previous one while an expression
    /// was expected: the previous line ends with an operator (or `,`, `(`…).
    fn continuation_hint(&self) -> &'static str {
        let gap = &self.src[(self.prev_end as usize).min(self.cur.span.start as usize)
            ..self.cur.span.start as usize];
        if self.prev_end > 0 && gap.contains('\n') {
            " (the previous line ends with an operator: the expression continues on this line)"
        } else {
            ""
        }
    }

    fn unclosed<R>(&self, opener: &str, opener_span: Span) -> Result<R> {
        let (line, _) = line_col(self.src, opener_span.start);
        self.error_here(format!(
            "expected `end` to close `{opener}` (line {line}), found {}",
            self.found()
        ))
    }

    // ─── Error recovery ──────────────────────────────────────────────────────

    /// Records an error. Returns `Err` (and stops parsing) if it comes from a premature
    /// end of source or if there are too many.
    fn report(&mut self, e: Error) -> Result<()> {
        if e.incomplete || self.errors.len() + 1 >= MAX_ERRORS {
            self.abort = true;
            self.errors.push(e.clone());
            return Err(e);
        }
        self.errors.push(e);
        Ok(())
    }

    /// Parses an item (statement, member…) with `f`. On error, reports it and skips to the
    /// end of the item: `Ok(None)`, parsing goes on. `Err` only when parsing must stop
    /// (premature end of source, too many errors).
    fn recovering<V>(
        &mut self,
        terminators: &[TokenKind],
        what: &str,
        f: impl FnOnce(&mut Self) -> Result<V>,
    ) -> Result<Option<V>> {
        let snap = self.snapshot();
        let start = self.tok().span.start;
        match f(self) {
            Ok(v) => {
                let k = self.peek();
                if !matches!(k, T::Newline | T::Semi | T::Eof) && !terminators.contains(&k) {
                    let e = Error::syntax(
                        format!(
                            "expected a line end after {what}, found {}",
                            self.found()
                        ),
                        self.tok().span,
                    );
                    self.report(e)?;
                    self.recover(start, terminators);
                }
                Ok(Some(v))
            }
            Err(e) => {
                if self.abort {
                    return Err(e);
                }
                self.restore(snap);
                self.report(e)?;
                self.recover(start, terminators);
                Ok(None)
            }
        }
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            depth: self.depth,
            no_do: self.no_do,
            lens: [
                self.s_exprs.len(),
                self.s_args.len(),
                self.s_fields.len(),
                self.s_entries.len(),
                self.s_parts.len(),
            ],
        }
    }

    fn restore(&mut self, s: Snapshot) {
        self.depth = s.depth;
        self.no_do = s.no_do;
        self.s_exprs.truncate(s.lens[0]);
        self.s_args.truncate(s.lens[1]);
        self.s_fields.truncate(s.lens[2]);
        self.s_entries.truncate(s.lens[3]);
        self.s_parts.truncate(s.lens[4]);
    }

    /// Skips the rest of an erroneous statement started at `stmt_start`: up to the line
    /// end, or up to the `end` of the constructs it opened, or up to a terminator of the
    /// current block (`end`, `else`…).
    fn recover(&mut self, stmt_start: u32, terminators: &[TokenKind]) {
        // Constructs opened in the part already read: lex it again.
        let mut tracker = Tracker::default();
        let consumed =
            &self.src[stmt_start as usize..(self.cur.span.start as usize).max(stmt_start as usize)];
        let mut relex = Lexer::new(consumed);
        loop {
            let t = relex.next_token();
            if t.kind == T::Eof {
                break;
            }
            tracker.feed(t);
        }
        // A parenthesis left open in the broken part must not swallow everything.
        tracker.brackets = 0;
        for f in &mut tracker.stack {
            match f {
                Frame::DefSig { brackets, .. } | Frame::Endless { brackets } => *brackets = 0,
                Frame::Block => {}
            }
        }
        loop {
            let t = self.tok();
            if t.kind == T::Eof {
                return;
            }
            if tracker.depth() == 0 && tracker.brackets == 0 {
                if matches!(t.kind, T::Newline | T::Semi) {
                    self.bump();
                    return;
                }
                if terminators.contains(&t.kind) {
                    return;
                }
            }
            let before = tracker.depth();
            tracker.feed(t);
            self.bump();
            // A one-line construct (`def f = …`) has just closed.
            if before > 0 && tracker.depth() == 0 && matches!(t.kind, T::Newline | T::Semi) {
                return;
            }
        }
    }

    fn enter(&mut self) -> Result<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return self.error_here("nesting too deep");
        }
        Ok(())
    }

    // ─── Names ───────────────────────────────────────────────────────────────

    #[inline]
    fn intern(&mut self, span: Span) -> Symbol {
        let src = self.src;
        self.interner.intern(span.text(src))
    }

    fn ident(&mut self, context: &str) -> Result<(Symbol, Span)> {
        let t = self.expect(T::Ident, context)?;
        Ok((self.intern(t.span), t.span))
    }

    // ─── Blocks and statements ───────────────────────────────────────────────

    /// Sequence of statements up to one of the `terminators` (not consumed) or the end.
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
            if let Some(stmt) = self.recovering(terminators, "the statement", Self::stmt)? {
                stmts.push(stmt);
            }
        }
        self.depth -= 1;
        let span = match (stmts.first(), stmts.last()) {
            (Some(a), Some(b)) => a.span.to(b.span),
            _ => Span::new(start.start, start.start),
        };
        Ok(Block { stmts, span })
    }

    /// Block ended by `end` (consumed).
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
                    .error_here("macros are not supported yet (planned: version 3)");
            }
            _ if !annotations.is_empty() => {
                return self.error_here(format!(
                    "an annotation must precede a declaration (`def`, `type`, `class`, `enum`, field), found {}",
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

    /// Expression followed by optional modifiers: `expr if cond`, `expr unless cond`.
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
        self.expect(T::Eq, "after the `let` declaration")?;
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
                self.expect(T::RParen, "to close the pattern")?;
                Ok(Pattern::Tuple(items, self.span_from(start)))
            }
            T::LBrace => {
                self.bump();
                let mut names = Vec::new();
                while !self.at(T::RBrace) {
                    names.push(self.ident("in the `{…}` pattern")?);
                    if !self.eat(T::Comma) {
                        break;
                    }
                }
                self.expect(T::RBrace, "to close the pattern")?;
                Ok(Pattern::Record(names, self.span_from(start)))
            }
            T::LBracket => {
                self.bump();
                let mut items = Vec::new();
                let mut rest = None;
                while !self.at(T::RBracket) {
                    if self.eat(T::DotDotDot) {
                        rest = Some(self.ident("after `...`")?);
                        break;
                    }
                    items.push(self.pattern()?);
                    if !self.eat(T::Comma) {
                        break;
                    }
                }
                self.expect(T::RBracket, "to close the pattern")?;
                Ok(Pattern::List {
                    items,
                    rest,
                    span: self.span_from(start),
                })
            }
            _ => self.error_here(format!(
                "expected a variable name after `let`, found {}",
                self.found()
            )),
        }
    }

    fn annotations(&mut self) -> Result<Vec<AnnotationUse>> {
        let mut out = Vec::new();
        while self.at(T::AtLBracket) {
            let start = self.bump().span;
            let mut path = vec![self.ident("after `@[`")?.0];
            while self.eat(T::ColonColon) {
                path.push(self.ident("after `::`")?.0);
            }
            let args = if self.at(T::LParen) {
                self.args()?
            } else {
                Slice::EMPTY
            };
            self.expect(T::RBracket, "to close the annotation")?;
            out.push(AnnotationUse {
                path,
                args,
                span: self.span_from(start),
            });
            self.skip_newlines();
        }
        Ok(out)
    }

    // ─── Declarations ────────────────────────────────────────────────────────

    fn def(&mut self, annotations: Vec<AnnotationUse>) -> Result<FnDecl> {
        let start = self.bump().span;
        let name_tok = self.tok();
        if name_tok.kind != T::Ident && !is_keyword(name_tok.kind) {
            return self.error_here(format!(
                "expected a function name after `def`, found {}",
                self.found()
            ));
        }
        self.bump();
        let mut name_span = self.question_suffix(name_tok.span);
        // Setter: `def name=(value)`, the `=` attached to the name.
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
                    "expected a line end or `=` after the signature of `{}`, found {}",
                    self.interner.resolve(name),
                    self.found()
                ));
            }
            self.body_end("def", start)?
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

    /// A `?` attached to the end of `span` (`to_i?`, `xs[i]?`) is consumed and added
    /// to it.
    fn question_suffix(&mut self, span: Span) -> Span {
        if self.at(T::Question) && self.tok().span.start == span.end {
            span.to(self.bump().span)
        } else {
            span
        }
    }

    /// Parameters between `open` and `close`: `(a: int, b = 0)` or `|x, y|`.
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
                        .error_here(format!("expected a parameter name, found {}", self.found()));
                }
            };
            self.bump();
            if params.iter().any(|p| p.name == name) {
                return self.error_at(
                    format!(
                        "parameter `{}` declared twice",
                        self.interner.resolve(name)
                    ),
                    start,
                );
            }
            // In `|…|`, a union must be parenthesized: `|x: (int | str)|`.
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
                self.expect(close, "to close the parameters")?;
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
        let (name, name_span) = self.ident(&format!("after `{opener}`"))?;
        let mut members: Vec<Member> = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.peek(), T::End | T::Eof) {
                break;
            }
            if let Some(m) = self.recovering(&[T::End], "the member", |p| p.type_member(&members))? {
                members.push(m);
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

    /// Member of a `type` / `class`: field, method or other statement.
    fn type_member(&mut self, members: &[Member]) -> Result<Member> {
        let annotations = self.annotations()?;
        let start = self.tok().span;
        Ok(match (self.peek(), self.peek2()) {
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
                        format!("field `{}` declared twice", self.interner.resolve(name)),
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
        })
    }

    fn enum_decl(&mut self, annotations: Vec<AnnotationUse>) -> Result<EnumDecl> {
        let kw = self.bump();
        let (name, name_span) = self.ident("after `enum`")?;
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
            match self.recovering(&[T::End], "the variant", |p| p.enum_member(&variants))? {
                Some(Ok(v)) => variants.push(v),
                Some(Err(m)) => methods.push(m),
                None => {}
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

    /// Variant (`Ok`) or method (`Err`) of an `enum`.
    fn enum_member(
        &mut self,
        variants: &[Variant],
    ) -> Result<std::result::Result<Variant, FnDecl>> {
        let annotations = self.annotations()?;
        if self.at(T::Def) {
            return Ok(Err(self.def(annotations)?));
        }
        let (vname, start) = self.ident("(enum variant)")?;
        if variants.iter().any(|v| v.name == vname) {
            return self.error_at(
                format!(
                    "variant `{}` declared twice",
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
        Ok(Ok(Variant {
            name: vname,
            value,
            annotations,
            span: self.span_from(start),
        }))
    }

    // ─── Types ───────────────────────────────────────────────────────────────

    /// `allow_union` is false in `|…|`, where `|` would close the list.
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
                    path.push(self.ident("after `::`")?.0);
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
                let t = self.expect(T::Int, "after `-`")?;
                TypeKind::LitInt(self.int_value(t, neg)?)
            }
            T::LBracket => {
                self.bump();
                let inner = self.type_expr(true)?;
                self.expect(T::RBracket, "to close the list type `[T]`")?;
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
                        let (name, _) = self.ident("(record field)")?;
                        self.expect(T::Colon, "after the field name")?;
                        fields.push((name, self.type_expr(true)?));
                        self.skip_newlines();
                        if !self.eat(T::Comma) {
                            break;
                        }
                    }
                    self.skip_newlines();
                    self.expect(T::RBrace, "to close the record type")?;
                    TypeKind::Record(fields)
                } else {
                    let k = self.type_expr(true)?;
                    self.expect(
                        T::FatArrow,
                        "in the hash type `{K => V}` (for a record: `{name: T}`)",
                    )?;
                    let v = self.type_expr(true)?;
                    self.expect(T::RBrace, "to close the hash type")?;
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
                self.expect(T::RParen, "to close the type")?;
                if self.eat(T::Colon) {
                    let ret = self.type_postfix()?;
                    TypeKind::Fn(items, Box::new(ret))
                } else if items.len() == 1 && !trailing_comma {
                    // Plain parenthesizing: `(int | str)?`
                    return Ok(items.pop().unwrap());
                } else {
                    TypeKind::Tuple(items)
                }
            }
            _ => return self.error_here(format!("expected a type, found {}", self.found())),
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
                        .error_at("cannot assign to this expression", self.span(lhs));
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
        // Nested prefix operators (`- - - x`): recursion that does not go through `expr`.
        stacker::maybe_grow(RED_ZONE, STACK_GROW, || self.binary_inner(min_bp))
    }

    fn binary_inner(&mut self, min_bp: u8) -> Result<ExprId> {
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
            // Comparisons and ranges do not chain: `a < b < c` is rejected.
            let next = self.peek();
            let chained = (is_comparison(kind) && is_comparison(next))
                || (matches!(kind, T::DotDot | T::DotDotDot)
                    && matches!(next, T::DotDot | T::DotDotDot));
            if chained {
                return self.error_at(
                    format!(
                        "{} cannot directly follow {}: add parentheses",
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
                // `-5` is a literal (allows i64::MIN), except `-5.abs()` = `-(5.abs())`.
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
                        // `recv.name(args)`: the field becomes a method call.
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
                            .map_err(|_| Error::syntax("invalid tuple index", t.span))?;
                        e = self.mk(ExprKind::TupleIndex(e, idx), span);
                    } else if t.kind == T::Ident || is_keyword(t.kind) {
                        self.bump();
                        let name_span = self.question_suffix(t.span);
                        let name = self.intern(name_span);
                        e = self.mk(ExprKind::Field(e, name), span.to(name_span));
                    } else {
                        return self.error_here(format!(
                            "expected a method or field name after `.`, found {}",
                            self.found()
                        ));
                    }
                }
                T::ColonColon => {
                    let ExprKind::Ident(first) = self.ast.kind(e) else {
                        return self.error_at("`::` must follow a type name", self.span(e));
                    };
                    let mut path = vec![first];
                    while self.eat(T::ColonColon) {
                        path.push(self.ident("after `::`")?.0);
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
                    let close = self.expect(T::RBracket, "to close the index")?;
                    // `xs[i]?`: `nil` rather than an error when out of bounds.
                    let optional = self.question_suffix(close.span) != close.span;
                    let span = self.span_from(self.span(e));
                    let kind = if optional {
                        ExprKind::IndexOpt(e, index)
                    } else {
                        ExprKind::Index(e, index)
                    };
                    e = self.mk(kind, span);
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
                                "a `do` block must follow a function or method call",
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
        let body = self.body_end("do", start);
        self.no_do = saved;
        Ok(Lambda {
            params,
            body: body?,
            span: self.span_from(start),
        })
    }

    /// `(a, name: b)`
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
                        format!("argument `{}` given twice", self.interner.resolve(name)),
                        t.span,
                    );
                }
                Some(name)
            } else {
                if self.s_args[base..].last().is_some_and(|a| a.name.is_some()) {
                    return self.error_here("positional argument after a named argument");
                }
                None
            };
            let value = self.expr()?;
            self.s_args.push(Arg { name, value });
            self.skip_newlines();
            if !self.eat(T::Comma) {
                self.skip_newlines();
                self.expect(T::RParen, "to close the arguments")?;
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
                        .map_err(|_| Error::syntax("invalid float", t.span))?,
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
                // `valid?(x)`: call of a function whose name ends with `?`.
                let mut span = t.span;
                if self.at(T::Question)
                    && self.tok().span.start == span.end
                    && self.peek2() == T::LParen
                {
                    span = span.to(self.bump().span);
                }
                ExprKind::Ident(self.intern(span))
            }
            T::IVar => {
                self.bump();
                ExprKind::IVar(self.intern(Span::new(t.span.start + 1, t.span.end)))
            }
            T::LParen => return self.paren(),
            T::LBracket => {
                self.bump();
                let saved = std::mem::replace(&mut self.no_do, false);
                let items = self.comma_list(T::RBracket, "to close the list");
                self.no_do = saved;
                ExprKind::List(items?)
            }
            T::LBrace => return self.brace(),
            T::If => return self.if_expr(),
            T::Unless => return self.unless_expr(),
            T::Case => return self.case_expr(),
            T::While => return self.while_expr(),
            T::For => return self.for_expr(),
            T::Begin => return self.begin_expr(),
            T::Raise => {
                self.bump();
                ExprKind::Raise(self.expr()?)
            }
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
                    "`do` is not used here: start a new line after the condition",
                );
            }
            T::Rescue | T::Ensure => {
                return self.error_here(format!(
                    "{} outside of a `begin`, `def` or `do` block",
                    t.kind.describe()
                ));
            }
            T::Let => {
                return self.error_here(format!(
                    "`let` is a statement: it cannot appear inside an expression{}",
                    self.continuation_hint()
                ));
            }
            _ => {
                return self.error_here(format!(
                    "expected an expression, found {}{}",
                    self.found(),
                    self.continuation_hint()
                ));
            }
        };
        let span = self.span_from(t.span);
        Ok(self.mk(kind, span))
    }

    /// Comma-separated elements up to `close` (trailing comma allowed).
    fn comma_list(&mut self, close: TokenKind, context: &str) -> Result<Slice<ExprId>> {
        let base = self.s_exprs.len();
        self.comma_items(close, context)?;
        Ok(self.take_exprs(base))
    }

    /// Pushes the elements onto `s_exprs` up to `close` (consumed).
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

    /// Pops the elements pushed onto `s_exprs` since `base` and stores them in the AST.
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
            // `(a, b, …)`: `first` is already read, the rest is pushed after it.
            let base = self.s_exprs.len();
            self.s_exprs.push(first);
            self.comma_items(T::RParen, "to close the tuple")?;
            let items = self.take_exprs(base);
            let span = self.span_from(start);
            return Ok(self.mk(ExprKind::Tuple(items), span));
        }
        self.expect(T::RParen, "to close the parenthesis")?;
        Ok(first)
    }

    /// `{}` (empty hash), `{k => v}` (hash) or `{name: v, other}` (record).
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
                let (name, span) = self.ident("(record field)")?;
                if self.s_fields[base..].iter().any(|(n, _)| *n == name) {
                    return self.error_at(
                        format!("field `{}` given twice", self.interner.resolve(name)),
                        span,
                    );
                }
                let value = if self.eat(T::Colon) {
                    self.expr()?
                } else {
                    // Shorthand `{x}` = `{x: x}`
                    self.mk(ExprKind::Ident(name), span)
                };
                self.s_fields.push((name, value));
                self.skip_newlines();
                if !self.eat(T::Comma) {
                    self.skip_newlines();
                    self.expect(T::RBrace, "to close the record")?;
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
                "between the key and the value (for a record: `{name: value}`)",
            )?;
            let value = self.expr()?;
            self.s_entries.push((key, value));
            self.skip_newlines();
            if !self.eat(T::Comma) {
                self.skip_newlines();
                self.expect(T::RBrace, "to close the hash")?;
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
                    "expected `}}` to close the interpolation, found {}",
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

    /// Rest of an `if` / `elsif` after the keyword.
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

    /// Condition followed by `then` or a line end.
    fn condition(&mut self, opener: &str) -> Result<ExprId> {
        let cond = self.expr()?;
        if !self.eat(T::Then) && !matches!(self.peek(), T::Newline | T::Semi) {
            return self.error_here(format!(
                "expected `then` or a line end after the `{opener}` condition, found {}",
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
                    "expected `then` or a line end after `when`, found {}",
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
                "expected `when` after `case`, found {}",
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

    /// Loop header expression: `do` forbidden, line end required.
    fn loop_head(&mut self, opener: &str) -> Result<ExprId> {
        let saved = std::mem::replace(&mut self.no_do, true);
        let e = self.expr();
        self.no_do = saved;
        let e = e?;
        if self.at(T::Do) {
            return self.error_here(format!(
                "`do` is not used after `{opener}`: start a new line"
            ));
        }
        if !matches!(self.peek(), T::Newline | T::Semi) {
            return self.error_here(format!(
                "expected a line end after the `{opener}` header, found {}",
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
        let mut vars = vec![self.ident("after `for`")?];
        while self.eat(T::Comma) {
            vars.push(self.ident("after `,`")?);
        }
        self.expect(T::In, "after the `for` variables")?;
        let iter = self.loop_head("for")?;
        let body = self.block_end("for", start)?;
        let span = self.span_from(start);
        Ok(self
            .ast
            .push_control(Control::For(For { vars, iter, body }), span))
    }

    fn begin_expr(&mut self) -> Result<ExprId> {
        let start = self.bump().span;
        let body = self.block(&[T::Rescue, T::Else, T::Ensure, T::End])?;
        self.begin_tail(body, start, "begin")
    }

    /// Body of a `def` / `do` block, up to its `end`: `rescue` / `ensure` clauses make it
    /// an implicit `begin`.
    fn body_end(&mut self, opener: &str, opener_span: Span) -> Result<Block> {
        let body = self.block(&[T::End, T::Rescue, T::Ensure])?;
        if !matches!(self.peek(), T::Rescue | T::Ensure) {
            self.end(opener, opener_span)?;
            return Ok(body);
        }
        let e = self.begin_tail(body, opener_span, opener)?;
        let span = self.span(e);
        Ok(expr_block(e, span))
    }

    /// `rescue` clauses, `else`, `ensure` and `end` after the body of a `begin`.
    fn begin_tail(&mut self, body: Block, start: Span, opener: &str) -> Result<ExprId> {
        let mut rescues = Vec::new();
        while self.at(T::Rescue) {
            let clause_start = self.bump().span;
            let mut var = None;
            let mut ty = None;
            if self.at(T::Ident) {
                let first = self.tok().span.text(self.src).chars().next();
                if first.is_some_and(char::is_uppercase) {
                    ty = Some(self.type_expr(true)?);
                } else {
                    var = Some(self.ident("after `rescue`")?);
                    if self.eat(T::Colon) {
                        ty = Some(self.type_expr(true)?);
                    }
                }
            }
            if !self.eat(T::Then) && !matches!(self.peek(), T::Newline | T::Semi) {
                return self.error_here(format!(
                    "expected a line end after `rescue`, found {}",
                    self.found()
                ));
            }
            let body = self.block(&[T::Rescue, T::Else, T::Ensure, T::End])?;
            rescues.push(Rescue {
                var,
                ty,
                body,
                span: self.span_from(clause_start),
            });
        }
        let else_ = if self.at(T::Else) {
            if rescues.is_empty() {
                return self.error_here(format!("`else` in a `{opener}` requires a `rescue`"));
            }
            self.bump();
            Some(self.block(&[T::Ensure, T::End])?)
        } else {
            None
        };
        let ensure = if self.eat(T::Ensure) {
            Some(self.block(&[T::End])?)
        } else {
            None
        };
        self.end(opener, start)?;
        let span = self.span_from(start);
        Ok(self.ast.push_control(
            Control::Begin(Begin {
                body,
                rescues,
                else_,
                ensure,
            }),
            span,
        ))
    }

    // ─── Literals ────────────────────────────────────────────────────────────

    fn int_value(&self, t: Token, negative: bool) -> Result<i64> {
        let too_big = || Error::syntax("integer too large (64 bits max)", t.span);
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

    /// Range of a string token's content, without delimiters.
    fn str_inner(t: Token) -> Span {
        let (skip_start, skip_end) = match t.kind {
            T::Str | T::StrTail => (1, 1),
            T::StrHead | T::StrMid => (1, 2),
            _ => unreachable!(),
        };
        Span::new(t.span.start + skip_start, t.span.end - skip_end)
    }

    /// Content of a string token, with escapes processed.
    fn str_content(&self, t: Token) -> Result<String> {
        let inner = Self::str_inner(t);
        unescape(inner.text(self.src), inner.start)
    }

    /// Interned string literal; without escapes, no allocation.
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

/// Processes `\n \t \r \0 \\ \" \# \u{…}`. `offset` is used for error messages.
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
            Some((_, '\n')) => {} // `\` at the end of a line: the string continues without a line break
            Some((_, 'u')) => {
                let rest = &s[i + 2..];
                let Some(close) = rest.strip_prefix('{').and_then(|r| r.find('}')) else {
                    return Err(Error::syntax("expected `\\u{…}`", at(2)));
                };
                let hex = &rest[1..close + 1];
                let c = u32::from_str_radix(hex, 16)
                    .ok()
                    .and_then(char::from_u32)
                    .ok_or_else(|| Error::syntax("invalid Unicode code point", at(close + 4)))?;
                out.push(c);
                for _ in 0..close + 2 {
                    chars.next();
                }
            }
            Some((_, other)) => {
                return Err(Error::syntax(
                    format!("unknown escape `\\{other}`"),
                    at(1 + other.len_utf8()),
                ));
            }
            None => return Err(Error::syntax("`\\` at the end of a string", at(1))),
        }
    }
    Ok(out)
}
