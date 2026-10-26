//! Erreurs et leur affichage avec extrait du source.

use crate::span::{Span, line_col};
use std::fmt;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Syntax,
    Runtime,
    Type,
}

/// Erreur de compilation ou d'exécution.
///
/// Réduite à un pointeur (8 octets) : un `Result<Expr, Error>` reste petit, ce qui compte
/// dans le parseur où chaque fonction en renvoie un. Les champs sont accessibles
/// directement grâce à `Deref` (`e.message`, `e.span`…).
#[derive(Clone, Debug, PartialEq)]
pub struct Error(Box<ErrorData>);

#[derive(Clone, Debug, PartialEq)]
pub struct ErrorData {
    pub kind: ErrorKind,
    pub message: String,
    pub span: Option<Span>,
    /// Vrai si l'erreur vient d'une fin de source prématurée (le REPL lit alors la suite).
    pub incomplete: bool,
    /// Pile d'appels (noms de fonctions, plus récent en premier), pour les erreurs d'exécution.
    pub trace: Vec<(String, Span)>,
    /// Script où l'erreur s'est produite : (nom, source). Renseigné par l'interpréteur,
    /// car une fonction peut venir d'un autre script que celui en cours.
    pub src: Option<(Rc<str>, Rc<str>)>,
}

impl std::ops::Deref for Error {
    type Target = ErrorData;
    fn deref(&self) -> &ErrorData {
        &self.0
    }
}

impl std::ops::DerefMut for Error {
    fn deref_mut(&mut self) -> &mut ErrorData {
        &mut self.0
    }
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>, span: Option<Span>) -> Self {
        Error(Box::new(ErrorData {
            kind,
            message: message.into(),
            span,
            incomplete: false,
            trace: Vec::new(),
            src: None,
        }))
    }

    pub fn syntax(message: impl Into<String>, span: Span) -> Self {
        Error::new(ErrorKind::Syntax, message, Some(span))
    }

    pub fn runtime(message: impl Into<String>, span: Span) -> Self {
        Error::new(ErrorKind::Runtime, message, Some(span))
    }

    pub fn type_error(message: impl Into<String>, span: Span) -> Self {
        Error::new(ErrorKind::Type, message, Some(span))
    }

    /// Affiche l'erreur avec la ligne fautive soulignée.
    ///
    /// ```text
    /// erreur de syntaxe: `end` attendu
    ///  --> script.bk:3:5
    ///   |
    /// 3 |     x + 1
    ///   |     ^^^^^
    /// ```
    ///
    /// Si l'erreur connaît son script (`src`), il est utilisé à la place de `src` / `file`.
    pub fn render(&self, src: &str, file: &str) -> String {
        let (file, src) = match &self.src {
            Some((name, text)) => (&**name, &**text),
            None => (file, src),
        };
        let title = match self.kind {
            ErrorKind::Syntax => "erreur de syntaxe",
            ErrorKind::Runtime => "erreur d'exécution",
            ErrorKind::Type => "erreur de type",
        };
        let mut out = format!("{title}: {}\n", self.message);
        let Some(span) = self.span else {
            return out;
        };
        let start = (span.start as usize).min(src.len());
        let (line, col) = line_col(src, span.start);
        let line_start = src[..start].rfind('\n').map_or(0, |i| i + 1);
        let line_end = src[start..].find('\n').map_or(src.len(), |i| start + i);
        let line_text = &src[line_start..line_end];
        let pad = " ".repeat(line.to_string().len());
        // Soulignement limité à la ligne de départ.
        let end = (span.end as usize).clamp(start, line_end);
        let width = src[start..end].chars().count().max(1);
        out.push_str(&format!("{pad}--> {file}:{line}:{col}\n"));
        out.push_str(&format!("{pad} |\n"));
        out.push_str(&format!("{line} | {line_text}\n"));
        out.push_str(&format!(
            "{pad} | {}{}\n",
            " ".repeat(col - 1),
            "^".repeat(width)
        ));
        for (name, call) in &self.trace {
            let (l, c) = line_col(src, call.start);
            out.push_str(&format!(
                "{pad} = dans `{name}` appelé depuis {file}:{l}:{c}\n"
            ));
        }
        out
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Une ou plusieurs erreurs, triées par position. L'analyse (parseur, vérificateur de
/// types) continue après une erreur pour toutes les signaler en une passe.
#[derive(Clone, Debug, PartialEq)]
pub struct Errors(pub Vec<Error>);

impl Errors {
    /// Trie par position et retire les doublons (même position, même message).
    pub fn new(mut errors: Vec<Error>) -> Self {
        errors.sort_by_key(|e| e.span.map_or(u32::MAX, |s| s.start));
        errors.dedup_by(|b, a| a.span == b.span && a.message == b.message);
        Errors(errors)
    }

    /// Vrai si la première erreur vient d'une fin de source prématurée (le REPL lit la suite).
    pub fn is_incomplete(&self) -> bool {
        self.0.first().is_some_and(|e| e.incomplete)
    }

    /// Toutes les erreurs, suivies d'un résumé s'il y en a plusieurs.
    pub fn render(&self, src: &str, file: &str) -> String {
        let mut out = String::new();
        for (n, e) in self.0.iter().enumerate() {
            if n > 0 {
                out.push('\n');
            }
            out.push_str(&e.render(src, file));
        }
        if self.0.len() > 1 {
            out.push_str(&format!("\n{} erreurs\n", self.0.len()));
        }
        out
    }
}

impl std::ops::Deref for Errors {
    type Target = [Error];
    fn deref(&self) -> &[Error] {
        &self.0
    }
}

impl From<Error> for Errors {
    fn from(e: Error) -> Self {
        Errors(vec![e])
    }
}

impl fmt::Display for Errors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (n, e) in self.0.iter().enumerate() {
            if n > 0 {
                f.write_str("\n")?;
            }
            f.write_str(&e.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for Errors {}
