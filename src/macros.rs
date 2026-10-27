//! Macros (Crystal style): run at parse time, they generate source code inserted in
//! place of their call.
//!
//! ```text
//! macro getters(*names)
//!   {% for name in names %}
//!     def {{name}} = @{{name}}
//!   {% end %}
//! end
//! ```
//!
//! - `{{ expr }}` inserts the text of a value: an argument's source code, a string's
//!   characters, the items of a list separated by `, `.
//! - `{% for x in list %} … {% end %}`, `{% if c %} … {% elsif c %} … {% else %} … {% end %}`,
//!   `{% let x = expr %}`: logic at expansion time.
//! - Expressions: names (parameters, `let`, loop variables), `"…#{expr}…"` (`\#` for a
//!   literal `#`), integers, `true` / `false` / `nil`, `[a, b]`, `==` `!=` `+` `and` `or`
//!   `not`, and methods: `.id` (an identifier), `.stringify` (a string literal of the
//!   source), `.name` / `.type` (of a field), `.map(|v| …)`, `.join(sep)`, `.size`,
//!   `.first`, `.last`, `.empty?`, `.upper`, `.lower`, `.capitalize`.
//! - `@type`, in a type body: `.name`, `.fields` (with `.name`, `.type`), `.methods`.
//! - Calls: `m(a, b: x)`, or `m a, b: x` (arguments up to the end of the line), or `m`;
//!   parameters may have a default (`macro m(a, b = 1)`), those after `*rest` are given
//!   by name only.
//! - Hygiene: a variable declared with `let` in the macro's own text is renamed, so it
//!   does not clash with the caller's variables.
//!
//! The parser parses the generated text (`parser.rs`); errors in it point into it.

use std::collections::HashMap;
use std::rc::Rc;

/// A declared macro.
#[derive(Debug)]
pub struct MacroDef {
    pub name: String,
    pub params: Vec<String>,
    /// Default value of each parameter, as written (`macro m(a, b = 1)`), if any.
    pub defaults: Vec<Option<String>>,
    /// `*rest`: the remaining arguments, as a list.
    pub splat: Option<String>,
    /// Parameters before `*rest` (those after it are given by name only).
    pub positional: usize,
    /// Text of the body (between the signature and `end`).
    pub body: String,
}

/// Macros known (from this script and the previous ones), and the source code generated
/// by their calls (after the script's own source).
#[derive(Default)]
pub struct Macros {
    pub defs: HashMap<String, Rc<MacroDef>>,
    /// The script's source, then the code generated so far: spans point into it.
    pub(crate) full: String,
    /// Number of expansions (for hygienic names).
    pub(crate) count: u32,
    /// Fields of the annotations declared (name, default value as written: empty if
    /// none), in order: positional arguments are read by field.
    pub(crate) annotations: HashMap<String, Rc<[(String, String)]>>,
}

/// Where the call is: the type (`@type`) and the method (`@def`) whose body contains it.
#[derive(Clone, Copy, Default)]
pub struct Ctx<'a> {
    pub ty: Option<&'a TypeCtx>,
    pub def: Option<&'a DefCtx>,
}

/// The type whose body contains the call (`@type`): as declared before the call.
#[derive(Clone, Debug, Default)]
pub struct TypeCtx {
    pub name: String,
    pub fields: Vec<Rc<VarCtx>>,
    pub methods: Vec<Rc<DefCtx>>,
    pub annotations: Vec<Rc<AnnCtx>>,
}

/// A method (`@def`, items of `@type.methods`).
#[derive(Clone, Debug, Default)]
pub struct DefCtx {
    pub name: String,
    pub params: Vec<Rc<VarCtx>>,
    /// Return type as written (empty: not annotated).
    pub ret: String,
    pub annotations: Vec<Rc<AnnCtx>>,
}

/// A field or a parameter.
#[derive(Clone, Debug, Default)]
pub struct VarCtx {
    pub name: String,
    /// Type as written (empty: not annotated).
    pub ty: String,
    /// Default value as written (empty: none).
    pub default: String,
    pub annotations: Vec<Rc<AnnCtx>>,
}

/// An annotation attached to a declaration (`@[Validate(min: 1)]`).
#[derive(Clone, Debug, Default)]
pub struct AnnCtx {
    pub name: String,
    /// Arguments: (name if named, source code).
    pub args: Vec<(Option<String>, String)>,
    /// Fields of the annotation's declaration, if known (see [`Macros::annotations`]).
    pub fields: Option<Rc<[(String, String)]>>,
}

impl AnnCtx {
    /// Value of field `f`: named argument, positional one in field order, or the
    /// declared default; `None` if `f` is not a field of the declaration.
    fn field(&self, f: &str) -> Option<MVal> {
        if let Some((_, v)) = self.args.iter().find(|(n, _)| n.as_deref() == Some(f)) {
            return Some(MVal::Node(v.clone()));
        }
        let Some(fields) = &self.fields else {
            return Some(MVal::Nil);
        };
        let k = fields.iter().position(|(n, _)| n == f)?;
        let positional: Vec<&String> = self
            .args
            .iter()
            .filter(|(n, _)| n.is_none())
            .map(|(_, v)| v)
            .collect();
        Some(match positional.get(k) {
            Some(v) => MVal::Node((*v).clone()),
            None if fields[k].1.is_empty() || fields[k].1 == "nil" => MVal::Nil,
            None => MVal::Node(fields[k].1.clone()),
        })
    }
}

/// Value at expansion time.
#[derive(Clone, Debug)]
enum MVal {
    /// Source code of an argument (or a name).
    Node(String),
    Str(String),
    Int(i64),
    Bool(bool),
    Nil,
    List(Vec<MVal>),
    Var(Rc<VarCtx>),
    Def(Rc<DefCtx>),
    Ann(Rc<AnnCtx>),
    Type(Rc<TypeCtx>),
    /// `|v| expr` (argument of `map`).
    Lambda(String, Rc<Expr>),
}

impl MVal {
    /// Text inserted by `{{ }}`.
    fn text(&self) -> String {
        match self {
            MVal::Node(s) | MVal::Str(s) => s.clone(),
            MVal::Int(n) => n.to_string(),
            MVal::Bool(b) => b.to_string(),
            MVal::Nil => String::new(),
            MVal::List(items) => items.iter().map(MVal::text).collect::<Vec<_>>().join(", "),
            MVal::Var(v) => v.name.clone(),
            MVal::Def(d) => d.name.clone(),
            MVal::Ann(a) => a.name.clone(),
            MVal::Type(t) => t.name.clone(),
            MVal::Lambda(..) => "<lambda>".into(),
        }
    }

    fn truthy(&self) -> bool {
        !matches!(self, MVal::Nil | MVal::Bool(false))
    }
}

/// Expands macro `def` called with the positional arguments `args` (source code of
/// each), at `ctx`: the generated source, or an error message.
#[cfg(test)]
pub fn expand(def: &MacroDef, args: Vec<String>, ctx: Ctx, count: u32) -> Result<String, String> {
    expand_call(
        def,
        args.into_iter().map(|a| (None, a)).collect(),
        ctx,
        count,
    )
}

/// Expands macro `def` called with `args` (name if named, source code), at `ctx`: the
/// generated source, or an error message.
pub fn expand_call(
    def: &MacroDef,
    args: Vec<(Option<String>, String)>,
    ctx: Ctx,
    count: u32,
) -> Result<String, String> {
    let mut env: HashMap<String, MVal> = HashMap::new();
    for (p, v) in def.params.iter().chain(&def.splat).zip(bind(def, args)?) {
        env.insert(p.clone(), v);
    }
    if let Some(t) = ctx.ty {
        env.insert("@type".into(), MVal::Type(Rc::new(t.clone())));
    }
    if let Some(d) = ctx.def {
        env.insert("@def".into(), MVal::Def(Rc::new(d.clone())));
    }
    let nodes = parse_template(&def.body)?;
    let mut out = Vec::new();
    run(&nodes, &mut vec![env], &mut out)?;
    Ok(hygiene(out, count))
}

/// Values of the parameters of `def` (the last one: `*rest`, if declared) for `args`:
/// positional ones in order, named ones by name, the others their default.
fn bind(def: &MacroDef, args: Vec<(Option<String>, String)>) -> Result<Vec<MVal>, String> {
    let n = def.params.len();
    let np = def.positional;
    let mut bound: Vec<Option<String>> = vec![None; n];
    let mut rest = Vec::new();
    let mut k = 0;
    let positional = args.iter().filter(|(name, _)| name.is_none()).count();
    for (name, text) in args {
        match name {
            None if k < np => {
                bound[k] = Some(text);
                k += 1;
            }
            None if def.splat.is_some() => rest.push(MVal::Node(text)),
            None => {
                let required = def.defaults[..np].iter().filter(|d| d.is_none()).count();
                let expected = if required == np {
                    format!("{np}")
                } else {
                    format!("{required} to {np}")
                };
                return Err(format!(
                    "macro `{}` expects {expected} argument{}, got {positional}",
                    def.name,
                    if np == 1 { "" } else { "s" },
                ));
            }
            Some(name) => {
                let i = def
                    .params
                    .iter()
                    .position(|p| *p == name)
                    .ok_or_else(|| format!("macro `{}` has no parameter `{name}`", def.name))?;
                if bound[i].is_some() {
                    return Err(format!("argument `{name}` given twice"));
                }
                bound[i] = Some(text);
            }
        }
    }
    let mut out = Vec::with_capacity(n + 1);
    for (i, b) in bound.into_iter().enumerate() {
        let text = match b.or_else(|| def.defaults.get(i).cloned().flatten()) {
            Some(t) => t,
            None => {
                return Err(format!(
                    "macro `{}`: missing argument `{}`",
                    def.name, def.params[i]
                ));
            }
        };
        out.push(MVal::Node(text));
    }
    if def.splat.is_some() {
        out.push(MVal::List(rest));
    }
    Ok(out)
}

// ─── Template ───────────────────────────────────────────────────────────────

#[derive(Debug)]
enum Node {
    Text(String),
    Insert(Expr),
    Let(String, Expr),
    For(String, Expr, Vec<Node>),
    /// Branches (condition, body), then the `else` body.
    If(Vec<(Expr, Vec<Node>)>, Vec<Node>),
}

/// A piece of the template: text, `{{ … }}` or `{% … %}`.
enum Piece {
    Text(String),
    Insert(String),
    Tag(String),
}

/// `src` (the source from a macro's body on) with its tags masked, byte for byte: what a
/// `{{ }}` contains is not code (`f.type`), a `{% for %}` / `{% if %}` / `{% end %}`
/// opens or closes like the keyword. An unclosed tag is left as it is.
pub fn mask(src: &str) -> String {
    let mut b = src.as_bytes().to_vec();
    let mut i = 0;
    while i + 1 < b.len() {
        if b[i] != b'{' || !matches!(b[i + 1], b'{' | b'%') {
            i += 1;
            continue;
        }
        let close = if b[i + 1] == b'{' { b'}' } else { b'%' };
        let mut j = i + 2;
        let mut in_str = false;
        while j + 1 < b.len() {
            match b[j] {
                b'\\' if in_str => j += 1,
                b'"' => in_str = !in_str,
                c if !in_str && c == close && b[j + 1] == b'}' => break,
                _ => {}
            }
            j += 1;
        }
        if j + 1 >= b.len() {
            break;
        }
        let end = j + 2;
        let kw: &[u8] = if close == b'}' {
            b"x"
        } else {
            let inner = src[i + 2..j].trim_start();
            let word = inner
                .split(|c: char| !c.is_alphanumeric())
                .next()
                .unwrap_or("");
            match word {
                "for" => b";for",
                "if" => b";if",
                "end" => b";end",
                _ => b"",
            }
        };
        for (k, byte) in b[i..end].iter_mut().enumerate() {
            *byte = kw.get(k).copied().unwrap_or(b' ');
        }
        i = end;
    }
    // Only ASCII bytes replaced whole tags (`{`…`}`): the text stays UTF-8.
    String::from_utf8(b).expect("masked text")
}

/// Splits the body into pieces. `}}` / `%}` inside a string literal of an expression do
/// not end it.
fn pieces(src: &str) -> Result<Vec<Piece>, String> {
    let mut out = Vec::new();
    let b = src.as_bytes();
    let (mut i, mut text_start) = (0, 0);
    while i + 1 < b.len() {
        let open = (b[i], b[i + 1]);
        if open == (b'{', b'{') || open == (b'{', b'%') {
            out.push(Piece::Text(src[text_start..i].to_string()));
            let close = if open.1 == b'{' { b'}' } else { b'%' };
            let start = i + 2;
            let mut j = start;
            let mut in_str = false;
            loop {
                if j + 1 >= b.len() {
                    return Err(format!(
                        "`{}` without its `{}}}`",
                        &src[i..i + 2],
                        close as char
                    ));
                }
                match b[j] {
                    b'\\' if in_str => j += 1,
                    b'"' => in_str = !in_str,
                    c if !in_str && c == close && b[j + 1] == b'}' => break,
                    _ => {}
                }
                j += 1;
            }
            let inner = src[start..j].trim().to_string();
            out.push(if open.1 == b'{' {
                Piece::Insert(inner)
            } else {
                Piece::Tag(inner)
            });
            i = j + 2;
            text_start = i;
        } else {
            i += 1;
        }
    }
    out.push(Piece::Text(src[text_start..].to_string()));
    Ok(out)
}

fn parse_template(src: &str) -> Result<Vec<Node>, String> {
    let pieces = pieces(src)?;
    let mut it = pieces.into_iter().peekable();
    let (nodes, end) = nodes_until(&mut it)?;
    match end {
        None => Ok(nodes),
        Some(tag) => Err(format!(
            "`{{% {tag} %}}` without its `{{% for %}}` / `{{% if %}}`"
        )),
    }
}

/// Nodes up to a closing tag (`end`, `else`, `elsif …`), returned with it.
fn nodes_until(
    it: &mut std::iter::Peekable<std::vec::IntoIter<Piece>>,
) -> Result<(Vec<Node>, Option<String>), String> {
    let mut out = Vec::new();
    while let Some(p) = it.next() {
        match p {
            Piece::Text(t) => out.push(Node::Text(t)),
            Piece::Insert(e) => out.push(Node::Insert(parse_expr(&e)?)),
            Piece::Tag(tag) => {
                let (kw, rest) = match tag.split_once(char::is_whitespace) {
                    Some((k, r)) => (k, r.trim()),
                    None => (tag.as_str(), ""),
                };
                match kw {
                    "end" | "else" | "elsif" => return Ok((out, Some(tag))),
                    "let" => {
                        let (name, e) = rest.split_once('=').ok_or_else(|| {
                            format!("expected `{{% let name = … %}}`, found `{tag}`")
                        })?;
                        out.push(Node::Let(name.trim().to_string(), parse_expr(e)?));
                    }
                    "for" => {
                        let (var, list) = rest.split_once(" in ").ok_or_else(|| {
                            format!("expected `{{% for x in list %}}`, found `{tag}`")
                        })?;
                        let (body, end) = nodes_until(it)?;
                        if end.as_deref() != Some("end") {
                            return Err("`{% for %}` without its `{% end %}`".into());
                        }
                        out.push(Node::For(var.trim().to_string(), parse_expr(list)?, body));
                    }
                    "if" => {
                        let mut branches = Vec::new();
                        let mut cond = parse_expr(rest)?;
                        let else_ = loop {
                            let (body, end) = nodes_until(it)?;
                            branches.push((cond, body));
                            match end
                                .as_deref()
                                .map(|t| t.split_once(char::is_whitespace).map_or(t, |(k, _)| k))
                            {
                                Some("end") => break Vec::new(),
                                Some("else") => {
                                    let (body, end) = nodes_until(it)?;
                                    if end.as_deref() != Some("end") {
                                        return Err("`{% else %}` without its `{% end %}`".into());
                                    }
                                    break body;
                                }
                                Some("elsif") => {
                                    let t = end.unwrap();
                                    cond = parse_expr(t["elsif".len()..].trim())?;
                                }
                                _ => return Err("`{% if %}` without its `{% end %}`".into()),
                            }
                        };
                        out.push(Node::If(branches, else_));
                    }
                    _ => return Err(format!("unknown macro tag `{{% {tag} %}}`")),
                }
            }
        }
    }
    Ok((out, None))
}

/// Generated text: (from the macro's own text?, text).
type Out = Vec<(bool, String)>;

fn run(nodes: &[Node], env: &mut Vec<HashMap<String, MVal>>, out: &mut Out) -> Result<(), String> {
    for n in nodes {
        match n {
            Node::Text(t) => out.push((true, t.clone())),
            Node::Insert(e) => out.push((false, eval(e, env)?.text())),
            Node::Let(name, e) => {
                let v = eval(e, env)?;
                env.last_mut().unwrap().insert(name.clone(), v);
            }
            Node::For(var, list, body) => {
                let items = match eval(list, env)? {
                    MVal::List(items) => items,
                    v => return Err(format!("`{{% for %}}` over {}, not a list", kind(&v))),
                };
                for item in items {
                    env.push(HashMap::from([(var.clone(), item)]));
                    let r = run(body, env, out);
                    env.pop();
                    r?;
                }
            }
            Node::If(branches, else_) => {
                let mut done = false;
                for (c, body) in branches {
                    if eval(c, env)?.truthy() {
                        run(body, env, out)?;
                        done = true;
                        break;
                    }
                }
                if !done {
                    run(else_, env, out)?;
                }
            }
        }
    }
    Ok(())
}

/// Variables declared with `let` in the macro's own text are renamed (`x` → `x__m3`) in
/// that text: they do not clash with the caller's.
fn hygiene(out: Out, count: u32) -> String {
    let mut names: Vec<String> = Vec::new();
    for (own, t) in &out {
        if !*own {
            continue;
        }
        let mut it = t.split_whitespace();
        while let Some(w) = it.next() {
            if w == "let"
                && let Some(n) = it.next()
            {
                let n: String = n
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !n.is_empty() && !names.contains(&n) {
                    names.push(n);
                }
            }
        }
    }
    let mut s = String::new();
    for (own, t) in out {
        if own && !names.is_empty() {
            s.push_str(&rename(&t, &names, count));
        } else {
            s.push_str(&t);
        }
    }
    s
}

/// `t` with the identifiers `names` (whole words, not a field `@x`, a method `.x` or a
/// named argument `x:`) suffixed.
fn rename(t: &str, names: &[String], count: u32) -> String {
    let mut out = String::new();
    let chars: Vec<char> = t.chars().collect();
    let mut i = 0;
    let mut in_str = false;
    // Open `#{` interpolations: braces opened in each.
    let mut interp: Vec<u32> = Vec::new();
    let mut prev_word = String::new();
    while i < chars.len() {
        let c = chars[i];
        if in_str {
            if c == '\\' {
                out.push(c);
                if let Some(&n) = chars.get(i + 1) {
                    out.push(n);
                }
                i += 2;
                continue;
            }
            if c == '"' {
                in_str = false;
            } else if c == '#' && chars.get(i + 1) == Some(&'{') {
                out.push_str("#{");
                i += 2;
                in_str = false;
                interp.push(0);
                continue;
            }
            out.push(c);
            i += 1;
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' if !interp.is_empty() => *interp.last_mut().unwrap() += 1,
            '}' if !interp.is_empty() => {
                let top = interp.last_mut().unwrap();
                if *top == 0 {
                    interp.pop();
                    in_str = true;
                } else {
                    *top -= 1;
                }
            }
            _ => {}
        }
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let before = if start > 0 { chars[start - 1] } else { ' ' };
            let after = chars.get(i).copied().unwrap_or(' ');
            // `name: value` is a named argument, but `let x: T` declares `x`.
            let named_arg = after == ':' && chars.get(i + 1) != Some(&':') && prev_word != "let";
            let renamable = names.contains(&word) && before != '@' && before != '.' && !named_arg;
            out.push_str(&word);
            if renamable {
                out.push_str(&format!("__m{count}"));
            }
            prev_word = word;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

// ─── Expressions ────────────────────────────────────────────────────────────

#[derive(Debug)]
enum Expr {
    Name(String),
    Str(Vec<StrPart>),
    Int(i64),
    Bool(bool),
    Nil,
    List(Vec<Expr>),
    Call(Box<Expr>, String, Vec<Expr>),
    Bin(String, Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    Lambda(String, Rc<Expr>),
    Assign(String, Box<Expr>),
}

#[derive(Debug)]
enum StrPart {
    Text(String),
    Expr(Expr),
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Name(String),
    Str(String),
    Int(i64),
    Sym(&'static str),
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let mut out = Vec::new();
    let c: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
        } else if ch == '"' {
            let mut s = String::new();
            i += 1;
            let mut depth = 0;
            while i < c.len() && (c[i] != '"' || depth > 0) {
                if c[i] == '\\' && i + 1 < c.len() {
                    s.push('\\');
                    s.push(c[i + 1]);
                    i += 2;
                    continue;
                }
                if c[i] == '#' && c.get(i + 1) == Some(&'{') {
                    depth += 1;
                } else if c[i] == '}' && depth > 0 {
                    depth -= 1;
                }
                s.push(c[i]);
                i += 1;
            }
            if i >= c.len() {
                return Err("unclosed string in a macro expression".into());
            }
            i += 1;
            out.push(Tok::Str(s));
        } else if ch.is_ascii_digit() {
            let start = i;
            while i < c.len() && c[i].is_ascii_digit() {
                i += 1;
            }
            let t: String = c[start..i].iter().collect();
            out.push(Tok::Int(
                t.parse().map_err(|_| format!("bad integer `{t}`"))?,
            ));
        } else if ch.is_alphabetic() || ch == '_' || ch == '@' {
            let start = i;
            i += 1;
            while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            if i < c.len() && c[i] == '?' {
                i += 1;
            }
            out.push(Tok::Name(c[start..i].iter().collect()));
        } else {
            let two: String = c[i..(i + 2).min(c.len())].iter().collect();
            let sym = match two.as_str() {
                "==" => Some("=="),
                "!=" => Some("!="),
                _ => None,
            };
            if let Some(s) = sym {
                out.push(Tok::Sym(s));
                i += 2;
                continue;
            }
            let s = match ch {
                '.' => ".",
                '(' => "(",
                ')' => ")",
                '[' => "[",
                ']' => "]",
                ',' => ",",
                '|' => "|",
                '+' => "+",
                '=' => "=",
                _ => return Err(format!("unexpected `{ch}` in a macro expression")),
            };
            out.push(Tok::Sym(s));
            i += 1;
        }
    }
    Ok(out)
}

struct EParser {
    toks: Vec<Tok>,
    pos: usize,
}

fn parse_expr(src: &str) -> Result<Expr, String> {
    let mut p = EParser {
        toks: lex(src)?,
        pos: 0,
    };
    let e = p.assign()?;
    if p.pos < p.toks.len() {
        return Err(format!(
            "unexpected `{:?}` in the macro expression `{src}`",
            p.toks[p.pos]
        ));
    }
    Ok(e)
}

impl EParser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn eat_sym(&mut self, s: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Sym(x)) if *x == s) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn eat_name(&mut self, n: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Name(x)) if x == n) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, s: &str) -> Result<(), String> {
        if self.eat_sym(s) {
            Ok(())
        } else {
            Err(format!("expected `{s}` in a macro expression"))
        }
    }

    /// `name = expr` (binds `name`, its value is the expression's: `{% if v = … %}`), or
    /// an expression.
    fn assign(&mut self) -> Result<Expr, String> {
        if let (Some(Tok::Name(n)), Some(Tok::Sym("="))) = (
            self.toks.get(self.pos).cloned(),
            self.toks.get(self.pos + 1),
        ) {
            self.pos += 2;
            return Ok(Expr::Assign(n, Box::new(self.or()?)));
        }
        self.or()
    }

    fn or(&mut self) -> Result<Expr, String> {
        let mut e = self.and()?;
        while self.eat_name("or") {
            e = Expr::Bin("or".into(), Box::new(e), Box::new(self.and()?));
        }
        Ok(e)
    }

    fn and(&mut self) -> Result<Expr, String> {
        let mut e = self.not()?;
        while self.eat_name("and") {
            e = Expr::Bin("and".into(), Box::new(e), Box::new(self.not()?));
        }
        Ok(e)
    }

    fn not(&mut self) -> Result<Expr, String> {
        if self.eat_name("not") {
            return Ok(Expr::Not(Box::new(self.not()?)));
        }
        let e = self.add()?;
        for op in ["==", "!="] {
            if self.eat_sym(op) {
                return Ok(Expr::Bin(op.into(), Box::new(e), Box::new(self.add()?)));
            }
        }
        Ok(e)
    }

    fn add(&mut self) -> Result<Expr, String> {
        let mut e = self.postfix()?;
        while self.eat_sym("+") {
            e = Expr::Bin("+".into(), Box::new(e), Box::new(self.postfix()?));
        }
        Ok(e)
    }

    fn postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.primary()?;
        while self.eat_sym(".") {
            let name = match self.peek().cloned() {
                Some(Tok::Name(n)) => {
                    self.pos += 1;
                    n
                }
                _ => return Err("expected a method name after `.`".into()),
            };
            let mut args = Vec::new();
            if self.eat_sym("(") {
                while !self.eat_sym(")") {
                    args.push(self.arg()?);
                    if !self.eat_sym(",") {
                        self.expect(")")?;
                        break;
                    }
                }
            }
            e = Expr::Call(Box::new(e), name, args);
        }
        Ok(e)
    }

    /// An argument: an expression, or `|v| expr`.
    fn arg(&mut self) -> Result<Expr, String> {
        if self.eat_sym("|") {
            let v = match self.peek().cloned() {
                Some(Tok::Name(n)) => {
                    self.pos += 1;
                    n
                }
                _ => return Err("expected a name in `|v|`".into()),
            };
            self.expect("|")?;
            return Ok(Expr::Lambda(v, Rc::new(self.or()?)));
        }
        self.or()
    }

    fn primary(&mut self) -> Result<Expr, String> {
        match self.peek().cloned() {
            Some(Tok::Int(n)) => {
                self.pos += 1;
                Ok(Expr::Int(n))
            }
            Some(Tok::Str(s)) => {
                self.pos += 1;
                Ok(Expr::Str(str_parts(&s)?))
            }
            Some(Tok::Name(n)) => {
                self.pos += 1;
                Ok(match n.as_str() {
                    "true" => Expr::Bool(true),
                    "false" => Expr::Bool(false),
                    "nil" => Expr::Nil,
                    _ => Expr::Name(n),
                })
            }
            Some(Tok::Sym("(")) => {
                self.pos += 1;
                let e = self.or()?;
                self.expect(")")?;
                Ok(e)
            }
            Some(Tok::Sym("[")) => {
                self.pos += 1;
                let mut items = Vec::new();
                while !self.eat_sym("]") {
                    items.push(self.or()?);
                    if !self.eat_sym(",") {
                        self.expect("]")?;
                        break;
                    }
                }
                Ok(Expr::List(items))
            }
            t => Err(format!("unexpected {t:?} in a macro expression")),
        }
    }
}

/// Parts of a string literal: text (escapes resolved, `\#` for `#`) and `#{expr}`.
fn str_parts(s: &str) -> Result<Vec<StrPart>, String> {
    let c: Vec<char> = s.chars().collect();
    let mut parts = Vec::new();
    let mut text = String::new();
    let mut i = 0;
    while i < c.len() {
        match c[i] {
            '\\' if i + 1 < c.len() => {
                text.push(match c[i + 1] {
                    'n' => '\n',
                    't' => '\t',
                    other => other,
                });
                i += 2;
            }
            '#' if c.get(i + 1) == Some(&'{') => {
                let start = i + 2;
                let mut depth = 1;
                let mut j = start;
                while j < c.len() && depth > 0 {
                    match c[j] {
                        '{' => depth += 1,
                        '}' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                if depth > 0 {
                    return Err("unclosed `#{` in a macro string".into());
                }
                if !text.is_empty() {
                    parts.push(StrPart::Text(std::mem::take(&mut text)));
                }
                let inner: String = c[start..j - 1].iter().collect();
                parts.push(StrPart::Expr(parse_expr(&inner)?));
                i = j;
            }
            ch => {
                text.push(ch);
                i += 1;
            }
        }
    }
    if !text.is_empty() {
        parts.push(StrPart::Text(text));
    }
    Ok(parts)
}

fn kind(v: &MVal) -> &'static str {
    match v {
        MVal::Node(_) => "a node",
        MVal::Str(_) => "a string",
        MVal::Int(_) => "an integer",
        MVal::Bool(_) => "a boolean",
        MVal::Nil => "nil",
        MVal::List(_) => "a list",
        MVal::Var(_) => "a field or a parameter",
        MVal::Def(_) => "a method",
        MVal::Ann(_) => "an annotation",
        MVal::Type(_) => "a type",
        MVal::Lambda(..) => "a lambda",
    }
}

fn lookup(env: &[HashMap<String, MVal>], name: &str) -> Result<MVal, String> {
    for scope in env.iter().rev() {
        if let Some(v) = scope.get(name) {
            return Ok(v.clone());
        }
    }
    if name == "@type" {
        return Err("`@type` is only available in the body of a type".into());
    }
    if name == "@def" {
        return Err("`@def` is only available in the body of a method".into());
    }
    Err(format!("unknown name `{name}` in a macro"))
}

fn eval(e: &Expr, env: &mut Vec<HashMap<String, MVal>>) -> Result<MVal, String> {
    Ok(match e {
        Expr::Name(n) => lookup(env, n)?,
        Expr::Int(n) => MVal::Int(*n),
        Expr::Bool(b) => MVal::Bool(*b),
        Expr::Nil => MVal::Nil,
        Expr::Str(parts) => {
            let mut s = String::new();
            for p in parts {
                match p {
                    StrPart::Text(t) => s.push_str(t),
                    StrPart::Expr(e) => s.push_str(&eval(e, env)?.text()),
                }
            }
            MVal::Str(s)
        }
        Expr::List(items) => MVal::List(
            items
                .iter()
                .map(|i| eval(i, env))
                .collect::<Result<_, _>>()?,
        ),
        Expr::Lambda(v, body) => MVal::Lambda(v.clone(), body.clone()),
        Expr::Assign(name, e) => {
            let v = eval(e, env)?;
            env.last_mut().unwrap().insert(name.clone(), v.clone());
            v
        }
        Expr::Not(x) => MVal::Bool(!eval(x, env)?.truthy()),
        Expr::Bin(op, a, b) => {
            let a = eval(a, env)?;
            match op.as_str() {
                "and" => return if a.truthy() { eval(b, env) } else { Ok(a) },
                "or" => return if a.truthy() { Ok(a) } else { eval(b, env) },
                _ => {}
            }
            let b = eval(b, env)?;
            match op.as_str() {
                "==" => MVal::Bool(a.text() == b.text()),
                "!=" => MVal::Bool(a.text() != b.text()),
                _ => match (a, b) {
                    (MVal::Int(x), MVal::Int(y)) => MVal::Int(x + y),
                    (MVal::List(mut x), MVal::List(y)) => {
                        x.extend(y);
                        MVal::List(x)
                    }
                    (a, b) => MVal::Str(a.text() + &b.text()),
                },
            }
        }
        Expr::Call(recv, name, args) => {
            let r = eval(recv, env)?;
            method(r, name, args, env)?
        }
    })
}

fn method(
    r: MVal,
    name: &str,
    args: &[Expr],
    env: &mut Vec<HashMap<String, MVal>>,
) -> Result<MVal, String> {
    let arg = |k: usize, env: &mut Vec<HashMap<String, MVal>>| -> Result<MVal, String> {
        match args.get(k) {
            Some(e) => eval(e, env),
            None => Err(format!("`.{name}` expects an argument")),
        }
    };
    let or_any = |t: &str| MVal::Node(if t.is_empty() { "any".into() } else { t.into() });
    // `annotation(Validate)`: the annotation's name (a name bound in the macro: its text).
    let ann_name = |env: &mut Vec<HashMap<String, MVal>>| -> Result<Option<String>, String> {
        Ok(match args.first() {
            None => None,
            Some(Expr::Name(n)) if lookup(env, n).is_err() => Some(n.clone()),
            Some(e) => Some(eval(e, env)?.text()),
        })
    };
    let anns = match &r {
        MVal::Type(t) => Some(&t.annotations),
        MVal::Def(d) => Some(&d.annotations),
        MVal::Var(v) => Some(&v.annotations),
        _ => None,
    };
    if let Some(anns) = anns
        && matches!(name, "annotation" | "annotations")
    {
        let wanted = ann_name(env)?;
        let mut found = anns
            .iter()
            .filter(|a| wanted.as_ref().is_none_or(|w| *w == a.name))
            .map(|a| MVal::Ann(a.clone()));
        return Ok(if name == "annotation" {
            found.next_back().unwrap_or(MVal::Nil)
        } else {
            MVal::List(found.collect())
        });
    }
    if let MVal::Ann(a) = &r {
        match a.field(name) {
            Some(v) => return Ok(v),
            None if name == "name" => return Ok(MVal::Node(a.name.clone())),
            None if name == "args" => {
                return Ok(MVal::List(
                    a.args.iter().map(|(_, v)| MVal::Node(v.clone())).collect(),
                ));
            }
            None => return Err(format!("annotation `{}` has no field `{name}`", a.name)),
        }
    }
    Ok(match (name, r) {
        ("id", v) => MVal::Node(v.text().trim().to_string()),
        ("stringify", v) => MVal::Str(format!("{:?}", v.text().trim())),
        ("name", MVal::Var(v)) => MVal::Node(v.name.clone()),
        ("type", MVal::Var(v)) => or_any(&v.ty),
        ("default", MVal::Var(v)) if v.default.is_empty() => MVal::Nil,
        ("default", MVal::Var(v)) => MVal::Node(v.default.clone()),
        ("name", MVal::Def(d)) => MVal::Node(d.name.clone()),
        ("params", MVal::Def(d)) => {
            MVal::List(d.params.iter().map(|p| MVal::Var(p.clone())).collect())
        }
        ("return_type", MVal::Def(d)) => or_any(&d.ret),
        ("name", MVal::Type(t)) => MVal::Node(t.name.clone()),
        ("fields", MVal::Type(t)) => {
            MVal::List(t.fields.iter().map(|f| MVal::Var(f.clone())).collect())
        }
        ("methods", MVal::Type(t)) => {
            MVal::List(t.methods.iter().map(|m| MVal::Def(m.clone())).collect())
        }
        ("size" | "len", MVal::List(items)) => MVal::Int(items.len() as i64),
        ("size" | "len", v) => MVal::Int(v.text().chars().count() as i64),
        ("empty?" | "is_empty", MVal::List(items)) => MVal::Bool(items.is_empty()),
        ("empty?" | "is_empty", v) => MVal::Bool(v.text().is_empty()),
        ("first", MVal::List(items)) => items.into_iter().next().unwrap_or(MVal::Nil),
        ("last", MVal::List(items)) => items.into_iter().last().unwrap_or(MVal::Nil),
        ("join", MVal::List(items)) => {
            let sep = if args.is_empty() {
                String::new()
            } else {
                arg(0, env)?.text()
            };
            MVal::Str(items.iter().map(MVal::text).collect::<Vec<_>>().join(&sep))
        }
        ("map", MVal::List(items)) => {
            let MVal::Lambda(v, body) = arg(0, env)? else {
                return Err("`.map` expects `|v| expr`".into());
            };
            let mut out = Vec::new();
            for item in items {
                env.push(HashMap::from([(v.clone(), item)]));
                let r = eval(&body, env);
                env.pop();
                out.push(r?);
            }
            MVal::List(out)
        }
        ("upper", v) => MVal::Str(v.text().to_uppercase()),
        ("lower", v) => MVal::Str(v.text().to_lowercase()),
        ("capitalize", v) => {
            let t = v.text();
            let mut c = t.chars();
            MVal::Str(match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            })
        }
        (_, v) => return Err(format!("{} has no macro method `.{name}`", kind(&v))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(params: &[&str], splat: Option<&str>, body: &str) -> MacroDef {
        MacroDef {
            name: "m".into(),
            params: params.iter().map(|s| s.to_string()).collect(),
            defaults: vec![None; params.len()],
            splat: splat.map(str::to_string),
            positional: params.len(),
            body: body.into(),
        }
    }

    #[test]
    fn templates() {
        let d = def(
            &[],
            Some("names"),
            "{% for n in names %}def {{n}} = @{{n}}\n{% end %}",
        );
        assert_eq!(
            expand(&d, vec!["a".into(), "b".into()], Ctx::default(), 1).unwrap(),
            "def a = @a\ndef b = @b\n"
        );
        let d = def(
            &["x"],
            None,
            "{% if x == \"1\" %}one{% elsif x == \"2\" %}two{% else %}other{% end %}",
        );
        assert_eq!(
            expand(&d, vec!["2".into()], Ctx::default(), 1).unwrap(),
            "two"
        );
        assert_eq!(
            expand(&d, vec!["5".into()], Ctx::default(), 1).unwrap(),
            "other"
        );
        let d = def(
            &["x"],
            None,
            "{% let s = [x, \"b\"].map(|v| v.upper).join(\"-\") %}{{s}} {{x.stringify}}",
        );
        assert_eq!(
            expand(&d, vec!["a".into()], Ctx::default(), 1).unwrap(),
            "A-B \"a\""
        );
        // Hygiene: the macro's own `let t` is renamed, not the inserted `t`.
        let d = def(&["v"], None, "let t = {{v}}\nt + 1");
        assert_eq!(
            expand(&d, vec!["t".into()], Ctx::default(), 4).unwrap(),
            "let t__m4 = t\nt__m4 + 1"
        );
        let d = def(&[], None, "let t: int = 1\nprint(\"#{t} t\", t: t)");
        assert_eq!(
            expand(&d, vec![], Ctx::default(), 2).unwrap(),
            "let t__m2: int = 1\nprint(\"#{t__m2} t\", t: t__m2)"
        );
        let ty = TypeCtx {
            name: "P".into(),
            fields: ["x", "y"]
                .iter()
                .zip(["int", ""])
                .map(|(n, ty)| {
                    Rc::new(VarCtx {
                        name: n.to_string(),
                        ty: ty.to_string(),
                        ..Default::default()
                    })
                })
                .collect(),
            ..Default::default()
        };
        let d = def(
            &[],
            None,
            "{{@type.name}}: {{@type.fields.map(|f| \"#{f.name}=\\#{@#{f.name}}\").join(\", \")}}",
        );
        assert_eq!(
            expand(
                &d,
                vec![],
                Ctx {
                    ty: Some(&ty),
                    def: None
                },
                1
            )
            .unwrap(),
            "P: x=#{@x}, y=#{@y}"
        );
        assert!(
            expand(&d, vec![], Ctx::default(), 1)
                .unwrap_err()
                .contains("`@type` is only available")
        );
        assert!(
            expand(&def(&["a"], None, ""), vec![], Ctx::default(), 1)
                .unwrap_err()
                .contains("missing argument `a`")
        );
    }
}
