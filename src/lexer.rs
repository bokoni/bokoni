//! Lexer : source → jetons.
//!
//! Travaille sur les octets, ne fait aucune allocation par jeton : un jeton n'est qu'un
//! type et un intervalle ; le texte (nombre, nom, chaîne) est relu dans le source par le
//! parseur quand il en a besoin.
//!
//! Chaînes interpolées : `"a #{x} b #{y} c"` donne
//! `StrHead("a #{)`, `x`, `StrMid(} b #{)`, `y`, `StrTail(} c")`.
//! Une chaîne sans interpolation donne un seul jeton `Str`.
//!
//! Fins de ligne : un `Newline` termine une instruction. Il est supprimé après un jeton qui
//! ne peut pas finir une expression (opérateur, `,`, `(`…), ou si la ligne suivante commence
//! par `.` (chaînage de méthodes).

use crate::diagnostic::{Error, Result};
use crate::span::Span;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // Littéraux et noms
    Int,
    Float,
    Str,
    StrHead,
    StrMid,
    StrTail,
    Ident,
    /// `@nom`
    IVar,

    // Mots-clés
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

    // Ponctuation
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

    // Opérateurs
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
    /// Après ces jetons, une fin de ligne ne termine pas l'instruction.
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

    /// Nom lisible, pour les messages d'erreur.
    pub fn describe(self) -> &'static str {
        use TokenKind::*;
        match self {
            Int => "un entier",
            Float => "un nombre décimal",
            Str | StrHead => "une chaîne",
            StrMid | StrTail => "la suite d'une chaîne",
            Ident => "un identifiant",
            IVar => "un attribut `@…`",
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
            Newline => "une fin de ligne",
            Eof => "la fin du fichier",
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
    // Rejet rapide : les mots-clés font 2 à 10 lettres et commencent par l'une de celles-ci.
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

#[inline]
fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b >= 0x80
}

#[inline]
fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80
}

/// Découpe tout le source en jetons. Le dernier jeton est toujours `Eof`.
///
/// Le parseur n'utilise pas cette fonction : il lit les jetons un par un avec
/// [`Lexer::next_token`], ce qui garde le flux de jetons dans le cache du processeur.
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
    match lx.error.take() {
        Some(e) => Err(e),
        None => Ok(tokens),
    }
}

/// Lexer à la demande : chaque appel à [`next_token`](Lexer::next_token) produit un jeton.
pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    /// Profondeur d'accolades de chaque interpolation `#{` ouverte.
    interp: Vec<u32>,
    /// Type du dernier jeton produit (règles des fins de ligne). `Newline` au départ :
    /// les lignes vides en tête de fichier sont ignorées.
    last: TokenKind,
    /// Première erreur rencontrée ; le lexer ne produit ensuite plus que `Eof`.
    pub error: Option<Error>,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        let mut lx = Lexer {
            src: src.as_bytes(),
            pos: 0,
            interp: Vec::new(),
            last: TokenKind::Newline,
            error: None,
        };
        if src.len() > u32::MAX as usize {
            lx.error = Some(Error::syntax(
                "source trop grand (4 Gio max)",
                Span::new(0, 0),
            ));
        }
        lx
    }

    /// Jeton suivant. Après une erreur (rangée dans `error`) ou la fin, renvoie `Eof`.
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

    fn err<T>(&self, msg: impl Into<String>, start: usize) -> Result<T> {
        Err(Error::syntax(
            msg,
            Span::new(start as u32, (self.pos.max(start + 1)) as u32),
        ))
    }

    #[inline]
    fn skip_ident(&mut self) {
        while let Some(&c) = self.src.get(self.pos)
            && is_ident_continue(c)
        {
            self.pos += 1;
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
                    if is_ident_start(self.peek_at(0)) {
                        self.skip_ident();
                        return Ok(self.token(IVar, start));
                    }
                    return self.err("`@` doit être suivi d'un nom ou de `[`", start);
                }
                _ if is_ident_start(b) => {
                    self.pos += 1;
                    self.skip_ident();
                    let kind = keyword(&self.src[start..self.pos]).unwrap_or(Ident);
                    return Ok(self.token(kind, start));
                }
                _ => return self.punct(start, b),
            }
        }
    }

    /// Fin du source : `Newline` final si besoin, puis `Eof`.
    fn finish(&mut self) -> Result<Token> {
        use TokenKind::*;
        if !self.interp.is_empty() {
            let mut e = Error::syntax(
                "interpolation `#{` non fermée",
                Span::new(self.pos as u32, self.pos as u32),
            );
            e.incomplete = true;
            return Err(e);
        }
        let kind = if matches!(self.last, Newline | Eof) {
            Eof
        } else {
            Newline
        };
        Ok(self.token(kind, self.pos))
    }

    /// Une fin de ligne (déjà consommée) termine-t-elle l'instruction ?
    fn newline_ends_stmt(&self) -> bool {
        if self.last == TokenKind::Newline || self.last.continues_line() {
            return false;
        }
        // Ligne suivante qui commence par `.` (et pas `..`) : chaînage de méthodes.
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

    /// Lit un segment de chaîne, après `"` (`from_quote`) ou après le `}` d'une interpolation.
    fn string(&mut self, start: usize, from_quote: bool) -> Result<Token> {
        use TokenKind::*;
        loop {
            match self.src.get(self.pos) {
                None => {
                    let mut e = Error::syntax(
                        "chaîne non fermée",
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
        // `t.0.1` : après un `.`, seul un entier est possible (index de tuple).
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
        if is_ident_start(self.peek_at(0)) {
            self.pos += 1;
            return self.err("nombre mal formé", start);
        }
        Ok(self.token(
            if float {
                TokenKind::Float
            } else {
                TokenKind::Int
            },
            start,
        ))
    }

    fn punct(&mut self, start: usize, b: u8) -> Result<Token> {
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
            (b'!', _) => {
                self.pos += 1;
                return self.err("`!` n'existe pas : utiliser `not`", start);
            }
            (b'&', b'&') => {
                self.pos += 2;
                return self.err("`&&` n'existe pas : utiliser `and`", start);
            }
            _ => {
                // Caractère complet (UTF-8) pour le message.
                let len = match b {
                    0xF0.. => 4,
                    0xE0.. => 3,
                    0xC0.. => 2,
                    _ => 1,
                };
                let end = (start + len).min(self.src.len());
                let ch = String::from_utf8_lossy(&self.src[start..end]).into_owned();
                self.pos = end;
                return self.err(format!("caractère inattendu `{ch}`"), start);
            }
        };
        self.pos += len;
        Ok(self.token(kind, start))
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
        // Lignes vides et commentaires fusionnés, continuation après un opérateur.
        assert_eq!(
            kinds("a +\n  b\n\n# c\nd"),
            [Ident, Plus, Ident, Newline, Ident, Newline, Eof]
        );
        // Chaînage de méthodes sur la ligne suivante.
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
    fn errors() {
        assert!(tokenize("\"abc").unwrap_err().incomplete);
        assert!(tokenize("\"#{x").unwrap_err().incomplete);
        assert!(tokenize("a && b").is_err());
        assert!(tokenize("12abc").is_err());
    }
}
