//! Bokoni in the browser (interpreter only: native code cannot be generated inside
//! WebAssembly).
//!
//! A minimal interface for JavaScript, without `wasm-bindgen`: the source is written
//! (UTF-8) into memory obtained with `alloc`, `run` or `check` is called, and the answer,
//! a JSON object, is read at `result_ptr()` for `result_len()` bytes. What the script
//! prints goes to the imported `env.bk_output` as it happens.
//!
//! ```json
//! {"value": "[1, 2]" | null,
//!  "diagnostics": [{"severity": "error", "kind": "type", "message": "…",
//!                   "rendered": "…", "line": 1, "col": 5, "endLine": 1, "endCol": 9}]}
//! ```
//! Columns count UTF-16 units (as JavaScript strings do), from 1.

use bokoni::{Engine, Error, ErrorKind};
use std::cell::RefCell;
use std::fmt::Write;

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// Text printed by the script (UTF-8).
    fn bk_output(ptr: *const u8, len: usize);
}

thread_local! {
    static RESULT: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Name of the script in messages.
const FILE: &str = "main.bk";

/// Calls of the script into each other at most: deeper, an error rather than a crash.
/// Each one takes room on the browser's native stack (wasm frames), which cannot grow and
/// is small in a Chrome worker (about 1 MB): measured there, about 680 nested calls fit
/// (fewer through `map` and other built-ins).
const MAX_CALL_DEPTH: usize = 400;

/// `len` bytes for JavaScript to write into.
#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Frees memory obtained with `alloc(len)`.
///
/// # Safety
/// `ptr` comes from `alloc(len)` and is not used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    // SAFETY: allocated by `alloc` with this capacity; no element to drop.
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
}

#[unsafe(no_mangle)]
pub extern "C" fn result_ptr() -> *const u8 {
    RESULT.with(|r| r.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn result_len() -> usize {
    RESULT.with(|r| r.borrow().len())
}

/// Runs the script `len` bytes at `ptr` (in a new engine).
///
/// # Safety
/// `ptr` points to `len` bytes written by JavaScript.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn run(ptr: *const u8, len: usize) {
    // SAFETY: see above.
    let src = unsafe { source(ptr, len) };
    let mut e = engine();
    // SAFETY: the imported function only reads the `len` bytes during the call.
    e.set_output(|s| unsafe { bk_output(s.as_ptr(), s.len()) });
    let r = e.eval_named(src, FILE);
    let (value, mut errors) = match r {
        Ok(v) => (Some(e.inspect(&v)), Vec::new()),
        Err(errs) => (None, errs.0),
    };
    errors.extend(e.take_warnings());
    answer(src, value.as_deref(), &errors);
}

/// Static checks only (the script is compiled, not run): for the editor, as one types.
///
/// # Safety
/// `ptr` points to `len` bytes written by JavaScript.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn check(ptr: *const u8, len: usize) {
    // SAFETY: see above.
    let src = unsafe { source(ptr, len) };
    let mut e = engine();
    let mut errors = match e.check(src, FILE) {
        Ok(()) => Vec::new(),
        Err(errs) => errs.0,
    };
    errors.extend(e.take_warnings());
    answer(src, None, &errors);
}

/// # Safety
/// `ptr` points to `len` bytes.
unsafe fn source<'a>(ptr: *const u8, len: usize) -> &'a str {
    // SAFETY: see above.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    // JavaScript's `TextEncoder` writes valid UTF-8.
    std::str::from_utf8(bytes).unwrap_or("")
}

fn engine() -> Engine {
    let mut e = Engine::new();
    e.set_max_call_depth(MAX_CALL_DEPTH);
    e
}

/// Writes the JSON answer into `RESULT`.
fn answer(src: &str, value: Option<&str>, errors: &[Error]) {
    let mut out = String::from("{\"value\":");
    match value {
        Some(v) => string(&mut out, v),
        None => out.push_str("null"),
    }
    out.push_str(",\"diagnostics\":[");
    for (k, err) in errors.iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        diagnostic(&mut out, src, err);
    }
    out.push_str("]}");
    RESULT.with(|r| *r.borrow_mut() = out);
}

fn diagnostic(out: &mut String, src: &str, err: &Error) {
    let (severity, kind) = match err.kind {
        ErrorKind::Syntax => ("error", "syntax"),
        ErrorKind::Type => ("error", "type"),
        ErrorKind::Runtime => ("error", "runtime"),
        ErrorKind::Warning => ("warning", "warning"),
    };
    let _ = write!(
        out,
        "{{\"severity\":\"{severity}\",\"kind\":\"{kind}\",\"message\":"
    );
    string(out, &err.message);
    out.push_str(",\"rendered\":");
    string(out, &err.render(src, FILE));
    // A position in this script (not in a file it requires).
    let here = err.src.as_ref().is_none_or(|(name, _)| &**name == FILE);
    if let Some(span) = err.span.filter(|_| here) {
        let (line, col) = position(src, span.start);
        let (end_line, end_col) = position(src, span.end.max(span.start + 1));
        let _ = write!(
            out,
            ",\"line\":{line},\"col\":{col},\"endLine\":{end_line},\"endCol\":{end_col}"
        );
    }
    out.push('}');
}

/// Line (from 1) and column (from 1, in UTF-16 units) of byte `offset`.
fn position(src: &str, offset: u32) -> (usize, usize) {
    let mut offset = (offset as usize).min(src.len());
    while !src.is_char_boundary(offset) {
        offset -= 1;
    }
    let before = &src[..offset];
    let line = before.bytes().filter(|&b| b == b'\n').count() + 1;
    let start = before.rfind('\n').map_or(0, |i| i + 1);
    let col = before[start..].encode_utf16().count() + 1;
    (line, col)
}

/// `s` as a JSON string.
fn string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_count_utf16_units() {
        let src = "let é = 1\n😀x";
        assert_eq!(position(src, 0), (1, 1));
        assert_eq!(position(src, 6), (1, 6));
        assert_eq!(position(src, 11), (2, 1));
        assert_eq!(position(src, 15), (2, 3));
    }

    #[test]
    fn json_strings_are_escaped() {
        let mut out = String::new();
        string(&mut out, "a\"b\\c\nd\u{1}");
        assert_eq!(out, r#""a\"b\\c\nd\u0001""#);
    }
}
