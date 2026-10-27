//! Errors and how they are rendered with a source excerpt.

use crate::span::{Span, line_col};
use std::fmt;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Syntax,
    Runtime,
    Type,
    /// Warning (does not prevent execution): use of a `@[Deprecated]` declaration…
    Warning,
}

/// Compile-time or runtime error.
///
/// Boxed down to a pointer (8 bytes): a `Result<Expr, Error>` stays small, which matters
/// in the parser where every function returns one. Fields are reachable directly
/// through `Deref` (`e.message`, `e.span`…).
#[derive(Clone, Debug, PartialEq)]
pub struct Error(Box<ErrorData>);

#[derive(Clone, Debug, PartialEq)]
pub struct ErrorData {
    pub kind: ErrorKind,
    pub message: String,
    pub span: Option<Span>,
    /// True if the error comes from a premature end of source (the REPL then reads more).
    pub incomplete: bool,
    /// Call stack (function names, most recent first), for runtime errors.
    pub trace: Vec<(String, Span)>,
    /// Script where the error occurred: (name, source). Set by the interpreter, since a
    /// function may come from a script other than the current one.
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

/// Maximum width of the displayed source excerpt (in characters).
const WINDOW: usize = 120;

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

    pub fn warning(message: impl Into<String>, span: Span) -> Self {
        Error::new(ErrorKind::Warning, message, Some(span))
    }

    /// Renders the error with the offending line underlined.
    ///
    /// ```text
    /// syntax error: expected `end`
    ///  --> script.bk:3:5
    ///   |
    /// 3 |     x + 1
    ///   |     ^^^^^
    /// ```
    ///
    /// If the error knows its script (`src`), it is used instead of `src` / `file`.
    pub fn render(&self, src: &str, file: &str) -> String {
        let (file, src) = match &self.src {
            Some((name, text)) => (&**name, &**text),
            None => (file, src),
        };
        let title = match self.kind {
            ErrorKind::Syntax => "syntax error",
            ErrorKind::Runtime => "runtime error",
            ErrorKind::Type => "type error",
            ErrorKind::Warning => "warning",
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
        // Underline limited to the starting line.
        let end = (span.end as usize).clamp(start, line_end);
        let width = src[start..end].chars().count().max(1);
        // Very long line: only a window around the position.
        let chars: Vec<char> = line_text.chars().collect();
        let (from, to) = if chars.len() > WINDOW {
            let from = (col - 1).saturating_sub(WINDOW / 3);
            (from, (from + WINDOW).min(chars.len()))
        } else {
            (0, chars.len())
        };
        let mut shown: String = chars[from..to].iter().collect();
        let mut caret = col - 1 - from;
        if from > 0 {
            shown.insert(0, '…');
            caret += 1;
        }
        if to < chars.len() {
            shown.push('…');
        }
        let width = width.min(to.saturating_sub(col - 1).max(1));
        out.push_str(&format!("{pad}--> {file}:{line}:{col}\n"));
        out.push_str(&format!("{pad} |\n"));
        out.push_str(&format!("{line} | {shown}\n"));
        out.push_str(&format!(
            "{pad} | {}{}\n",
            " ".repeat(caret),
            "^".repeat(width)
        ));
        for (name, call) in &self.trace {
            let (l, c) = line_col(src, call.start);
            out.push_str(&format!(
                "{pad} = in `{name}` called from {file}:{l}:{c}\n"
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

/// One or more errors, sorted by position. Analysis (parser, type checker) keeps going
/// after an error to report them all in one pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Errors(pub Vec<Error>);

impl Errors {
    /// Sorts by position and removes duplicates (same position, same message).
    pub fn new(mut errors: Vec<Error>) -> Self {
        errors.sort_by_key(|e| e.span.map_or(u32::MAX, |s| s.start));
        errors.dedup_by(|b, a| a.span == b.span && a.message == b.message);
        Errors(errors)
    }

    /// True if the first error comes from a premature end of source (the REPL reads more).
    pub fn is_incomplete(&self) -> bool {
        self.0.first().is_some_and(|e| e.incomplete)
    }

    /// All the errors, followed by a summary if there are several.
    pub fn render(&self, src: &str, file: &str) -> String {
        let mut out = String::new();
        for (n, e) in self.0.iter().enumerate() {
            if n > 0 {
                out.push('\n');
            }
            out.push_str(&e.render(src, file));
        }
        if self.0.len() > 1 {
            out.push_str(&format!("\n{} errors\n", self.0.len()));
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
