//! Lexer: source → tokens.
//!
//! Works on bytes and allocates nothing per token: a token is just a kind and a range;
//! the text (number, name, string) is read back from the source by the parser when it
//! needs it.
//!
//! Interpolated strings: `"a #{x} b #{y} c"` gives
//! `StrHead("a #{)`, `x`, `StrMid(} b #{)`, `y`, `StrTail(} c")`.
//! A string without interpolation gives a single `Str` token.
//!
//! Line ends: a `Newline` ends a statement. It is dropped after a token that cannot end
//! an expression (operator, `,`, `(`…), or when the next line starts with `.` (method
//! chaining).

use crate::diagnostic::{Error, Result};
use crate::span::Span;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // Literals and names
    Int,
    Float,
    Str,
    StrHead,
    StrMid,
    StrTail,
    Ident,
    /// `@name`
    IVar,

    // Keywords
    Let,
    Def,
    End,
    If,
    Elsif,
    Else,
    Unless,
    Then,
    Case,
    When,
    While,
    For,
    In,
    Do,
    Break,
    Next,
    Return,
    Begin,
    Rescue,
    Ensure,
    Raise,
    Type,
    Class,
    Enum,
    Annotation,
    Macro,
    True,
    False,
    Nil,
    And,
    Or,
    Not,
    SelfKw,

    // Punctuation
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    /// `@[`
    AtLBracket,
    Comma,
    Dot,
    DotDot,
    DotDotDot,
    Colon,
    ColonColon,
    Semi,
    Pipe,
    Question,
    FatArrow,

    // Operators
    Eq,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Shl,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,

    Newline,
    Eof,
}

impl TokenKind {
    /// After these tokens, a line end does not end the statement.
    fn continues_line(self) -> bool {
        use TokenKind::*;
        matches!(
            self,
            Comma
                | Dot
                | ColonColon
                | Colon
                | LParen
                | LBracket
                | LBrace
                | AtLBracket
                | FatArrow
                | Eq
                | EqEq
                | NotEq
                | Lt
                | Le
                | Gt
                | Ge
                | Plus
                | Minus
                | Star
                | Slash
                | Percent
                | Shl
                | PlusEq
                | MinusEq
                | StarEq
                | SlashEq
                | PercentEq
                | DotDot
                | DotDotDot
                | And
                | Or
                | Not
                | StrHead
                | StrMid
        )
    }

    /// Readable name, for error messages.
    pub fn describe(self) -> &'static str {
        use TokenKind::*;
        match self {
            Int => "an integer",
            Float => "a float",
            Str | StrHead => "a string",
            StrMid | StrTail => "the rest of a string",
            Ident => "an identifier",
            IVar => "an `@…` attribute",
            Let => "`let`",
            Def => "`def`",
            End => "`end`",
            If => "`if`",
            Elsif => "`elsif`",
            Else => "`else`",
            Unless => "`unless`",
            Then => "`then`",
            Case => "`case`",
            When => "`when`",
            While => "`while`",
            For => "`for`",
            In => "`in`",
            Do => "`do`",
            Break => "`break`",
            Next => "`next`",
            Return => "`return`",
            Begin => "`begin`",
            Rescue => "`rescue`",
            Ensure => "`ensure`",
            Raise => "`raise`",
            Type => "`type`",
            Class => "`class`",
            Enum => "`enum`",
            Annotation => "`annotation`",
            Macro => "`macro`",
            True => "`true`",
            False => "`false`",
            Nil => "`nil`",
            And => "`and`",
            Or => "`or`",
            Not => "`not`",
            SelfKw => "`self`",
            LParen => "`(`",
            RParen => "`)`",
            LBracket => "`[`",
            RBracket => "`]`",
            LBrace => "`{`",
            RBrace => "`}`",
            AtLBracket => "`@[`",
            Comma => "`,`",
            Dot => "`.`",
            DotDot => "`..`",
            DotDotDot => "`...`",
            Colon => "`:`",
            ColonColon => "`::`",
            Semi => "`;`",
            Pipe => "`|`",
            Question => "`?`",
            FatArrow => "`=>`",
            Eq => "`=`",
            EqEq => "`==`",
            NotEq => "`!=`",
            Lt => "`<`",
            Le => "`<=`",
            Gt => "`>`",
            Ge => "`>=`",
            Plus => "`+`",
            Minus => "`-`",
            Star => "`*`",
            Slash => "`/`",
            Percent => "`%`",
            Shl => "`<<`",
            PlusEq => "`+=`",
            MinusEq => "`-=`",
            StarEq => "`*=`",
            SlashEq => "`/=`",
            PercentEq => "`%=`",
            Newline => "a line end",
            Eof => "end of file",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[inline]
fn keyword(s: &[u8]) -> Option<TokenKind> {
    use TokenKind::*;
    // Fast reject: keywords are 2 to 10 letters long and start with one of these.
    if !(2..=10).contains(&s.len())
        || !matches!(
            s[0],
            b'a'..=b'f' | b'i' | b'l' | b'm' | b'n' | b'o' | b'r' | b's' | b't' | b'u' | b'w'
        )
    {
        return None;
    }
    Some(match s {
        b"let" => Let,
        b"def" => Def,
        b"end" => End,
        b"if" => If,
        b"elsif" => Elsif,
        b"else" => Else,
        b"unless" => Unless,
        b"then" => Then,
        b"case" => Case,
        b"when" => When,
        b"while" => While,
        b"for" => For,
        b"in" => In,
        b"do" => Do,
        b"break" => Break,
        b"next" => Next,
        b"return" => Return,
        b"begin" => Begin,
        b"rescue" => Rescue,
        b"ensure" => Ensure,
        b"raise" => Raise,
        b"type" => Type,
        b"class" => Class,
        b"enum" => Enum,
        b"annotation" => Annotation,
        b"macro" => Macro,
        b"true" => True,
        b"false" => False,
        b"nil" => Nil,
        b"and" => And,
        b"or" => Or,
        b"not" => Not,
        b"self" => SelfKw,
        _ => return None,
    })
}

/// Splits the whole source into tokens. The last token is always `Eof`.
///
/// The parser does not use this function: it reads tokens one by one with
/// [`Lexer::next_token`], which keeps the token stream in the CPU cache.
pub fn tokenize(src: &str) -> Result<Vec<Token>> {
    let mut lx = Lexer::new(src);
    let mut tokens = Vec::with_capacity(src.len() / 4 + 16);
    loop {
        let t = lx.next_token();
        tokens.push(t);
        if t.kind == TokenKind::Eof {
            break;
        }
    }
    let mut errors = std::mem::take(&mut lx.errors);
    errors.extend(lx.error.take());
    match errors
        .into_iter()
        .min_by_key(|e| e.span.map_or(0, |s| s.start))
    {
        Some(e) => Err(e),
        None => Ok(tokens),
    }
}

/// On-demand lexer: each call to [`next_token`](Lexer::next_token) produces one token.
pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    /// Brace depth of each open `#{` interpolation.
    interp: Vec<u32>,
    /// Kind of the last token produced (line-end rules). `Newline` at the start: blank
    /// lines at the top of the file are skipped.
    last: TokenKind,
    /// Local errors (unknown character, `&&`…): reported, then the lexer continues.
    pub errors: Vec<Error>,
    /// Fatal error (unclosed string…): the lexer then only produces `Eof`.
    pub error: Option<Error>,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        let mut lx = Lexer {
            src: src.as_bytes(),
            pos: 0,
            interp: Vec::new(),
            last: TokenKind::Newline,
            errors: Vec::new(),
            error: None,
        };
        if src.len() > u32::MAX as usize {
            lx.error = Some(Error::syntax(
                "source too large (4 GiB max)",
                Span::new(0, 0),
            ));
        }
        lx
    }

    /// Next token. After an error (stored in `error`) or the end, returns `Eof`.
    #[inline]
    pub fn next_token(&mut self) -> Token {
        if self.error.is_none() {
            match self.step() {
                Ok(t) => {
                    self.last = t.kind;
                    return t;
                }
                Err(e) => self.error = Some(e),
            }
        }
        let end = self.src.len().min(u32::MAX as usize) as u32;
        Token {
            kind: TokenKind::Eof,
            span: Span::new(end, end),
        }
    }
}

impl Lexer<'_> {
    #[inline]
    fn peek_at(&self, offset: usize) -> u8 {
        self.src.get(self.pos + offset).copied().unwrap_or(0)
    }

    #[inline]
    fn token(&self, kind: TokenKind, start: usize) -> Token {
        Token {
            kind,
            span: Span::new(start as u32, self.pos as u32),
        }
    }

    /// Reports a local error; the lexer continues.
    fn report(&mut self, msg: impl Into<String>, start: usize) {
        self.errors.push(Error::syntax(
            msg,
            Span::new(start as u32, (self.pos.max(start + 1)) as u32),
        ));
    }

    /// Byte length of the identifier character at `pos`, 0 if it is not one.
    /// Identifiers: letters (Unicode included: `é`, `ß`, `π`), `_`, then digits.
    #[inline]
    fn ident_char(&self, pos: usize, first: bool) -> usize {
        match self.src.get(pos) {
            Some(&b) if b < 0x80 => {
                (b.is_ascii_alphabetic() || b == b'_' || (!first && b.is_ascii_digit())) as usize
            }
            Some(&b) => {
                let len = match b {
                    0xF0.. => 4,
                    0xE0.. => 3,
                    _ => 2,
                };
                let bytes = &self.src[pos..(pos + len).min(self.src.len())];
                std::str::from_utf8(bytes)
                    .ok()
                    .and_then(|s| s.chars().next())
                    .filter(|c| c.is_alphabetic() || (!first && c.is_numeric()))
                    .map_or(0, char::len_utf8)
            }
            None => 0,
        }
    }

    #[inline]
    fn skip_ident(&mut self) {
        loop {
            let n = self.ident_char(self.pos, false);
            if n == 0 {
                return;
            }
            self.pos += n;
        }
    }

    fn step(&mut self) -> Result<Token> {
        use TokenKind::*;
        loop {
            let Some(&b) = self.src.get(self.pos) else {
                return self.finish();
            };
            let start = self.pos;
            match b {
                b' ' | b'\t' | b'\r' => self.pos += 1,
                b'\\' if self.peek_at(1) == b'\n' => self.pos += 2,
                b'\\' if self.peek_at(1) == b'\r' && self.peek_at(2) == b'\n' => self.pos += 3,
                b'\n' => {
                    self.pos += 1;
                    if self.newline_ends_stmt() {
                        return Ok(self.token(Newline, start));
                    }
                }
                b'#' => {
                    while let Some(&c) = self.src.get(self.pos)
                        && c != b'\n'
                    {
                        self.pos += 1;
                    }
                }
                b'"' => {
                    self.pos += 1;
                    return self.string(start, true);
                }
                b'{' => {
                    if let Some(depth) = self.interp.last_mut() {
                        *depth += 1;
                    }
                    self.pos += 1;
                    return Ok(self.token(LBrace, start));
                }
                b'}' => {
                    self.pos += 1;
                    match self.interp.last_mut() {
                        Some(0) => {
                            self.interp.pop();
                            return self.string(start, false);
                        }
                        Some(depth) => *depth -= 1,
                        None => {}
                    }
                    return Ok(self.token(RBrace, start));
                }
                b'0'..=b'9' => return self.number(start),
                b'@' => {
                    self.pos += 1;
                    if self.peek_at(0) == b'[' {
                        self.pos += 1;
                        return Ok(self.token(AtLBracket, start));
                    }
                    if self.ident_char(self.pos, true) > 0 {
                        self.skip_ident();
                        return Ok(self.token(IVar, start));
                    }
                    self.report("`@` must be followed by a name or `[`", start);
                }
                _ if self.ident_char(self.pos, true) > 0 => {
                    self.skip_ident();
                    let kind = keyword(&self.src[start..self.pos]).unwrap_or(Ident);
                    return Ok(self.token(kind, start));
                }
                _ => {
                    if let Some(t) = self.punct(start, b) {
                        return Ok(t);
                    }
                }
            }
        }
    }

    /// End of source: a final `Newline` if needed, then `Eof`.
    fn finish(&mut self) -> Result<Token> {
        use TokenKind::*;
        if !self.interp.is_empty() {
            let mut e = Error::syntax(
                "unclosed `#{` interpolation",
                Span::new(self.pos as u32, self.pos as u32),
            );
            e.incomplete = true;
            return Err(e);
        }
        // No final line end after an operator: the expression is incomplete.
        let kind = if matches!(self.last, Newline | Eof) || self.last.continues_line() {
            Eof
        } else {
            Newline
        };
        Ok(self.token(kind, self.pos))
    }

    /// Does a line end (already consumed) end the statement?
    fn newline_ends_stmt(&self) -> bool {
        if self.last == TokenKind::Newline || self.last.continues_line() {
            return false;
        }
        // Next line starts with `.` (and not `..`): method chaining.
        let mut p = self.pos;
        loop {
            match self.src.get(p) {
                Some(b' ' | b'\t' | b'\r' | b'\n') => p += 1,
                Some(b'#') => {
                    while p < self.src.len() && self.src[p] != b'\n' {
                        p += 1;
                    }
                }
                _ => break,
            }
        }
        !(self.src.get(p) == Some(&b'.') && self.src.get(p + 1) != Some(&b'.'))
    }

    /// Reads a string segment, after `"` (`from_quote`) or after the `}` of an interpolation.
    fn string(&mut self, start: usize, from_quote: bool) -> Result<Token> {
        use TokenKind::*;
        loop {
            match self.src.get(self.pos) {
                None => {
                    let mut e = Error::syntax(
                        "unclosed string",
                        Span::new(start as u32, self.pos as u32),
                    );
                    e.incomplete = true;
                    return Err(e);
                }
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(self.token(if from_quote { Str } else { StrTail }, start));
                }
                Some(b'\\') => self.pos += 2,
                Some(b'#') if self.peek_at(1) == b'{' => {
                    self.pos += 2;
                    self.interp.push(0);
                    return Ok(self.token(if from_quote { StrHead } else { StrMid }, start));
                }
                Some(_) => self.pos += 1,
            }
        }
    }

    fn number(&mut self, start: usize) -> Result<Token> {
        let digits = |lx: &mut Self| {
            while let Some(b'0'..=b'9' | b'_') = lx.src.get(lx.pos) {
                lx.pos += 1;
            }
        };
        digits(self);
        // `t.0.1`: after a `.`, only an integer is possible (tuple index).
        let after_dot = self.last == TokenKind::Dot;
        let mut float = false;
        if !after_dot && self.peek_at(0) == b'.' && self.peek_at(1).is_ascii_digit() {
            float = true;
            self.pos += 1;
            digits(self);
        }
        if !after_dot && matches!(self.peek_at(0), b'e' | b'E') {
            let sign = matches!(self.peek_at(1), b'+' | b'-') as usize;
            if self.peek_at(1 + sign).is_ascii_digit() {
                float = true;
                self.pos += 1 + sign;
                digits(self);
            }
        }
        let kind = if float {
            TokenKind::Float
        } else {
            TokenKind::Int
        };
        let number = self.token(kind, start);
        if self.ident_char(self.pos, true) > 0 {
            // `12abc`: reported, then read as the number `12`.
            self.skip_ident();
            self.report("malformed number", start);
        }
        Ok(number)
    }

    /// Punctuation; `None` for an unknown character (reported and skipped).
    fn punct(&mut self, start: usize, b: u8) -> Option<Token> {
        use TokenKind::*;
        let next = self.peek_at(1);
        let (kind, len) = match (b, next) {
            (b'(', _) => (LParen, 1),
            (b')', _) => (RParen, 1),
            (b'[', _) => (LBracket, 1),
            (b']', _) => (RBracket, 1),
            (b',', _) => (Comma, 1),
            (b';', _) => (Semi, 1),
            (b'|', _) => (Pipe, 1),
            (b'?', _) => (Question, 1),
            (b'.', b'.') if self.peek_at(2) == b'.' => (DotDotDot, 3),
            (b'.', b'.') => (DotDot, 2),
            (b'.', _) => (Dot, 1),
            (b':', b':') => (ColonColon, 2),
            (b':', _) => (Colon, 1),
            (b'=', b'=') => (EqEq, 2),
            (b'=', b'>') => (FatArrow, 2),
            (b'=', _) => (Eq, 1),
            (b'!', b'=') => (NotEq, 2),
            (b'<', b'<') => (Shl, 2),
            (b'<', b'=') => (Le, 2),
            (b'<', _) => (Lt, 1),
            (b'>', b'=') => (Ge, 2),
            (b'>', _) => (Gt, 1),
            (b'+', b'=') => (PlusEq, 2),
            (b'+', _) => (Plus, 1),
            (b'-', b'=') => (MinusEq, 2),
            (b'-', _) => (Minus, 1),
            (b'*', b'=') => (StarEq, 2),
            (b'*', _) => (Star, 1),
            (b'/', b'=') => (SlashEq, 2),
            (b'/', _) => (Slash, 1),
            (b'%', b'=') => (PercentEq, 2),
            (b'%', _) => (Percent, 1),
            // Reported, then read as their equivalent so parsing can go on.
            (b'!', _) => {
                self.pos += 1;
                self.report("`!` does not exist: use `not`", start);
                return Some(self.token(Not, start));
            }
            (b'&', b'&') => {
                self.pos += 2;
                self.report("`&&` does not exist: use `and`", start);
                return Some(self.token(And, start));
            }
            _ => {
                // Whole (UTF-8) character for the message.
                let len = match b {
                    0xF0.. => 4,
                    0xE0.. => 3,
                    0xC0.. => 2,
                    _ => 1,
                };
                let end = (start + len).min(self.src.len());
                let ch = String::from_utf8_lossy(&self.src[start..end]).into_owned();
                self.pos = end;
                self.report(format!("unexpected character `{ch}`"), start);
                return None;
            }
        };
        self.pos += len;
        Some(self.token(kind, start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use TokenKind::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        tokenize(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn basics() {
        assert_eq!(
            kinds("let x = 1 + 2.5"),
            [Let, Ident, Eq, Int, Plus, Float, Newline, Eof]
        );
    }

    #[test]
    fn ranges_and_tuple_index() {
        assert_eq!(kinds("1..10"), [Int, DotDot, Int, Newline, Eof]);
        assert_eq!(kinds("0...n"), [Int, DotDotDot, Ident, Newline, Eof]);
        assert_eq!(kinds("t.0.1"), [Ident, Dot, Int, Dot, Int, Newline, Eof]);
    }

    #[test]
    fn interpolation() {
        let src = r##""a #{x + "#{y}"} b""##;
        assert_eq!(
            kinds(src),
            [
                StrHead, Ident, Plus, StrHead, Ident, StrTail, StrTail, Newline, Eof
            ]
        );
        assert_eq!(
            kinds(r##""#{ {a: 1} }""##),
            [
                StrHead, LBrace, Ident, Colon, Int, RBrace, StrTail, Newline, Eof
            ]
        );
        assert_eq!(kinds(r##""\#{x}""##), [Str, Newline, Eof]);
    }

    #[test]
    fn newlines() {
        // Blank lines and comments merged, continuation after an operator.
        assert_eq!(
            kinds("a +\n  b\n\n# c\nd"),
            [Ident, Plus, Ident, Newline, Ident, Newline, Eof]
        );
        // Method chaining on the next line.
        assert_eq!(
            kinds("xs\n  .map(f)"),
            [Ident, Dot, Ident, LParen, Ident, RParen, Newline, Eof]
        );
    }

    #[test]
    fn ivars_and_annotations() {
        assert_eq!(
            kinds("@[Tag] @name"),
            [AtLBracket, Ident, RBracket, IVar, Newline, Eof]
        );
    }

    #[test]
    fn unicode_identifiers() {
        assert_eq!(kinds("déjà π"), [Ident, Ident, Newline, Eof]);
        assert!(
            tokenize("5 € 2")
                .unwrap_err()
                .message
                .contains("unexpected character `€`")
        );
    }

    #[test]
    fn errors() {
        assert!(tokenize("\"abc").unwrap_err().incomplete);
        assert!(tokenize("\"#{x").unwrap_err().incomplete);
        assert!(tokenize("a && b").is_err());
        assert!(tokenize("12abc").is_err());
    }
}
