use bokoni::intern::Interner;
use bokoni::{parser, printer};

fn sexp(src: &str) -> String {
    let mut i = Interner::new();
    match parser::parse(src, &mut i) {
        Ok(p) => printer::program(&p, &i).trim_end().to_string(),
        Err(e) => panic!("{}", e.render(src, "test")),
    }
}

fn err(src: &str) -> String {
    let mut i = Interner::new();
    match parser::parse(src, &mut i) {
        Ok(p) => panic!("should fail: {}", printer::program(&p, &i)),
        Err(e) => e.to_string(),
    }
}

#[test]
fn precedence() {
    assert_eq!(sexp("1 + 2 * 3 - 4"), "(- (+ 1 (* 2 3)) 4)");
    assert_eq!(sexp("-x.abs() * 2"), "(* (- (.abs x)) 2)");
    assert_eq!(sexp("-2 * 3"), "(* -2 3)");
    assert_eq!(
        sexp("not a == b and c or d"),
        "(or (and (not (== a b)) c) d)"
    );
    assert_eq!(sexp("a < b + 1"), "(< a (+ b 1))");
    assert_eq!(sexp("0..n - 1"), "(.. 0 (- n 1))");
    assert_eq!(sexp("xs << a + b"), "(<< xs (+ a b))");
    assert_eq!(sexp("a = b = 1 + 2"), "(= a (= b (+ 1 2)))");
    assert_eq!(sexp("x += 1"), "(+= x 1)");
    assert_eq!(
        sexp("debug = mode == \"dev\""),
        "(= debug (== mode \"dev\"))"
    );
}

#[test]
fn literals() {
    assert_eq!(sexp("1_000"), "1000");
    assert_eq!(sexp("2.5e3"), "2500.0");
    assert_eq!(sexp("-9223372036854775808"), "-9223372036854775808");
    assert_eq!(sexp(r#""a\tb\u{e9}""#), r#""a\tbé""#);
    assert_eq!(
        sexp(r##""Bonjour #{name}, #{n + 1} ans""##),
        r#"(str "Bonjour " name ", " (+ n 1) " ans")"#
    );
    assert_eq!(sexp("[1, 2,\n 3,]"), "(list 1 2 3)");
    assert_eq!(sexp("(1, \"x\")"), "(tuple 1 \"x\")");
    assert_eq!(sexp("(1,)"), "(tuple 1)");
    assert_eq!(sexp("(1)"), "1");
    assert_eq!(sexp("{}"), "(hash)");
    assert_eq!(
        sexp(r#"{"a" => 1, k => v}"#),
        r#"(hash ("a" => 1) (k => v))"#
    );
    assert_eq!(sexp("{x: 1, y}"), "(record (x: 1) (y: y))");
    assert_eq!(sexp("{\n  x: 1,\n  y: 2\n}"), "(record (x: 1) (y: 2))");
}

#[test]
fn postfix() {
    assert_eq!(sexp("a.b.c(1)[0].d"), "(. (index (.c (. a b) 1) 0) d)");
    assert_eq!(sexp("pair.0"), "(. pair 0)");
    assert_eq!(sexp("t.0.1"), "(. (. t 0) 1)");
    assert_eq!(sexp("Protocol::Tcp.value"), "(. Protocol::Tcp value)");
    assert_eq!(sexp("add(a: 1, b: 2)"), "(call add a: 1 b: 2)");
    assert_eq!(
        sexp("xs\n  .filter(|n| n > 0)\n  .map(|n| n * 2)"),
        "(.map (.filter xs (fn (n) (> n 0))) (fn (n) (* n 2)))"
    );
}

#[test]
fn lambdas_and_blocks() {
    assert_eq!(sexp("|x| x + 1"), "(fn (x) (+ x 1))");
    assert_eq!(sexp("|| 1"), "(fn () 1)");
    assert_eq!(sexp("|a: int, b| a"), "(fn ((a : int) b) a)");
    assert_eq!(
        sexp("[1, 2].each do |n|\n  print(n)\nend"),
        "(.each (list 1 2) (fn (n) (call print n)))"
    );
    assert_eq!(
        sexp("Server do\n  port = 80\nend"),
        "(call Server (fn () (= port 80)))"
    );
    assert_eq!(
        sexp("routes << Route do\n  path = \"/\"\nend"),
        "(<< routes (call Route (fn () (= path \"/\"))))"
    );
}

#[test]
fn control() {
    assert_eq!(
        sexp("if a then 1 elsif b then 2 else 3 end"),
        "(if a (then 1) (else (if b (then 2) (else 3))))"
    );
    assert_eq!(sexp("if a\n  x\n  y\nend"), "(if a (then x y))");
    assert_eq!(sexp("unless a\n  x\nend"), "(if (not a) (then x))");
    assert_eq!(
        sexp("next if i % 2 == 0"),
        "(if (== (% i 2) 0) (then (next)))"
    );
    assert_eq!(
        sexp("return x unless ok"),
        "(if (not ok) (then (return x)))"
    );
    assert_eq!(sexp("return"), "(return)");
    assert_eq!(
        sexp(
            "case port\n  when 80, 443 then \"web\"\n  when 8000..8999\n    \"dev\"\n  else \"other\"\nend"
        ),
        "(case port (when (80 443) \"web\") (when ((.. 8000 8999)) \"dev\") (else \"other\"))"
    );
    assert_eq!(
        sexp("while x < 3\n  x += 1\nend"),
        "(while (< x 3) (+= x 1))"
    );
    assert_eq!(
        sexp("for k, v in h\n  print(k)\nend"),
        "(for (k v) h (call print k))"
    );
    assert_eq!(sexp("for i in 0...n\nend"), "(for (i) (... 0 n))");
    assert_eq!(
        sexp(
            "begin\n  f(1)\nrescue e: A | B\n  1\nrescue Error\n  2\nrescue\n  3\nelse\n  4\nensure\n  5\nend"
        ),
        "(begin (call f 1) (rescue e : A | B 1) (rescue : Error 2) (rescue 3) (else 4) (ensure 5))"
    );
    assert_eq!(sexp("raise \"x\" if bad"), "(if bad (then (raise \"x\")))");
    assert_eq!(sexp("xs[i]?"), "(index? xs i)");
    assert_eq!(sexp("s.to_i?() or 0"), "(or (.to_i? s) 0)");
    assert_eq!(sexp("def ok?(x) = x"), "(def ok? (x) x)");
    // `def` and `do` blocks with clauses: an implicit `begin`.
    assert_eq!(
        sexp("def g()\n  1\nrescue\n  2\nend"),
        "(def g () (begin 1 (rescue 2)))"
    );
    assert_eq!(
        sexp("xs.map do |x|\n  x\nensure\n  y\nend"),
        "(.map xs (fn (x) (begin x (ensure y))))"
    );
}

#[test]
fn let_and_patterns() {
    assert_eq!(sexp("let x = 1"), "(let x 1)");
    assert_eq!(sexp("let x: int? = nil"), "(let x : int? nil)");
    assert_eq!(sexp("let (a, b) = t"), "(let (tuple a b) t)");
    assert_eq!(sexp("let {x, y} = p"), "(let (record x y) p)");
    assert_eq!(sexp("let [a, ...rest] = xs"), "(let (list a ...rest) xs)");
}

#[test]
fn types() {
    assert_eq!(sexp("let a: [int] = []"), "(let a : [int] (list))");
    assert_eq!(
        sexp("let a: {str => int} = {}"),
        "(let a : {str => int} (hash))"
    );
    assert_eq!(
        sexp("let a: {x: int, y: str} = r"),
        "(let a : {x: int, y: str} r)"
    );
    assert_eq!(sexp("let a: (int, str) = t"), "(let a : (int, str) t)");
    assert_eq!(
        sexp("let a: \"dev\" | \"prod\" = m"),
        "(let a : \"dev\" | \"prod\" m)"
    );
    assert_eq!(sexp("let a: (int | str)? = m"), "(let a : (int | str)? m)");
    assert_eq!(
        sexp("let f: (int, int): int = g"),
        "(let f : (int, int): int g)"
    );
    assert_eq!(
        sexp("let p: Protocol::Kind = k"),
        "(let p : Protocol::Kind k)"
    );
}

#[test]
fn defs() {
    assert_eq!(
        sexp("def add(a: int, b: int = 0): int\n  a + b\nend"),
        "(def add ((a : int) (b : int = 0)) : int (+ a b))"
    );
    assert_eq!(
        sexp("def double(x: int): int = x * 2"),
        "(def double ((x : int)) : int (* x 2))"
    );
    assert_eq!(sexp("def name: str = @name"), "(def name : str @name)");
    assert_eq!(sexp("def name = @name"), "(def name @name)");
    assert_eq!(
        sexp("def name=(value: str)\n  @name = value\nend"),
        "(def name= ((value : str)) (= @name value))"
    );
    assert_eq!(
        sexp("def init(@name: str, @email: str? = nil)\nend"),
        "(def init ((@name : str) (@email : str? = nil)))"
    );
}

#[test]
fn declarations() {
    assert_eq!(
        sexp("type Server\n  host: str = \"localhost\"\n  port = 80\n  path: str\nend"),
        "(type Server (field host : str = \"localhost\") (field port = 80) (field path : str))"
    );
    assert_eq!(
        sexp("enum Protocol: str\n  Tcp = \"TCP\"\n  Udp\nend"),
        "(enum Protocol : str (Tcp = \"TCP\") Udp)"
    );
    assert_eq!(
        sexp("@[Tag]\ntype S\n  @[Validate(min: 1, max: 9)]\n  port: int = 80\nend"),
        "@[Tag] (type S @[Validate min: 1 max: 9] (field port : int = 80))"
    );
    assert_eq!(
        sexp("def f(@[V(min: 0)] t: int)\nend"),
        "(def f (@[V min: 0] (t : int)))"
    );
    assert_eq!(
        sexp("class User\n  def init(@name: str)\n  end\n  getters(name)\nend"),
        "(class User (def init ((@name : str))) (call getters name))"
    );
    assert_eq!(sexp("annotation Tag\nend"), "(annotation Tag)");
}

#[test]
fn errors() {
    assert!(err("1 +").contains("expected an expression"));
    assert!(err("if x\n  1\n").contains("expected `end` to close `if` (line 1)"));
    assert!(err("a < b < c").contains("add parentheses"));
    assert!(err("1 = 2").contains("cannot assign"));
    assert!(err("{\"a\": 1}").contains("expected `=>`"));
    assert!(err("f(a: 1, 2)").contains("positional argument after a named argument"));
    assert!(err("f(a: 1, a: 2)").contains("twice"));
    assert!(err("{x: 1, x: 2}").contains("twice"));
    assert!(err("while x do\nend").contains("`do` is not used"));
    assert!(err("1.(").contains("method or field name"));
    assert!(err("@[Tag] x = 1").contains("annotation"));
    assert!(err("x y").contains("expected a line end"));
    assert!(err("\"abc #{x\"").contains("unclosed"));

    // Errors at the end of the source are marked "incomplete" (the REPL reads more input).
    let mut i = Interner::new();
    assert!(
        parser::parse("def f\n  1", &mut i)
            .unwrap_err()
            .is_incomplete()
    );
    assert!(parser::parse("[1,", &mut i).unwrap_err().is_incomplete());
    assert!(!parser::parse("1 + )", &mut i).unwrap_err().is_incomplete());
}

#[test]
fn deep_nesting_does_not_overflow() {
    // 900 levels pass (the stack grows on the heap if needed)…
    let src = "(".repeat(900) + "1" + &")".repeat(900);
    assert_eq!(sexp(&src), "1");
    let src = "[".repeat(900) + &"]".repeat(900);
    assert!(sexp(&src).starts_with("(list (list"));
    // …beyond the limit, a clean error rather than a stack overflow.
    let src = "(".repeat(10_000) + "1" + &")".repeat(10_000);
    assert!(err(&src).contains("nesting too deep"));
}

/// Every ```ruby block of docs/syntax.md must be accepted by the parser.
/// Except: macros (`{%`, `{{`) and blocks with holes (`…`).
#[test]
fn spec_examples_parse() {
    let spec = include_str!("../docs/syntax.md");
    let mut count = 0;
    let mut rest = spec;
    while let Some(start) = rest.find("```ruby\n") {
        let body = &rest[start + 8..];
        let end = body.find("```").expect("unclosed block");
        let code = &body[..end];
        rest = &body[end + 3..];
        if code.contains("{%") || code.contains("{{") || code.contains('…') {
            continue;
        }
        let mut i = Interner::new();
        if let Err(e) = parser::parse(code, &mut i) {
            panic!(
                "syntax.md example rejected:\n{}\n{code}",
                e.render(code, "syntax.md")
            );
        }
        count += 1;
    }
    assert!(count >= 15, "only {count} examples found");
}

/// Lines (1-based) of the reported errors.
fn error_lines(src: &str) -> Vec<usize> {
    let mut i = Interner::new();
    match parser::parse(src, &mut i) {
        Ok(_) => Vec::new(),
        Err(errs) => errs
            .iter()
            .map(|e| bokoni::span::line_col(src, e.span.unwrap().start).0)
            .collect(),
    }
}

#[test]
fn reports_every_error_once() {
    let src = r#"let a = 1 +* 2
def f(x y)
  x + 1
end
if a b
  print(a)
end
let c = [1, 2,, 3]
x = 0 if a > 1 b
while a && b
  a = a - 1
end
[1, 2].each do |n|
  print(n n)
end
let s = {"a": 1}
let d = 5 € 2
def g = 1
def h(x) = x * 2
print(g)
"#;
    assert_eq!(error_lines(src), [1, 2, 5, 8, 9, 10, 14, 16, 17]);
}

#[test]
fn recovery_skips_whole_broken_construct() {
    // Broken header of a `def` / `if` / `case`: its body and its `end` are skipped,
    // without cascading errors.
    assert_eq!(
        error_lines("def f(a b)\n  if a\n    1\n  end\nend\nf(1)\n"),
        [1]
    );
    assert_eq!(
        error_lines("case x y\nwhen 1 then 2\nend\nlet z = 1 +\n"),
        [1, 4]
    );
    // A modifier `if` in the skipped part does not open a construct.
    assert_eq!(error_lines("let a = 1 2 if b\nlet c = (\n"), [1, 2]);
    // A `do` block opened on the faulty line is skipped up to its `end`.
    assert_eq!(error_lines("foo(1 2) do |x|\n  x\nend\nbar(\n"), [1, 4]);
}

#[test]
fn recovery_inside_bodies() {
    // Errors in a body are reported and the rest of the body is parsed.
    assert_eq!(error_lines("def f\n  1 +\n  let x = 2 3\n  x\nend\n"), [3]);
    assert_eq!(
        error_lines("type S\n  a: = 1\n  b: int\n  c: [int = 2\nend\nlet ok = 1\n"),
        [2, 4]
    );
    assert_eq!(error_lines("enum E\n  A = \n  B C\n  D\nend\n"), [3]);
    assert_eq!(error_lines("if x\n  a b\nelse\n  c d\nend\n"), [2, 4]);
}

#[test]
fn incomplete_source_stops_cleanly() {
    let mut i = Interner::new();
    let e = parser::parse("let a = 1 2\ndef f\n  1\n", &mut i).unwrap_err();
    assert_eq!(e.len(), 2);
    assert!(e[1].incomplete);
    // Unclosed string: parser errors after it are meaningless.
    let e = parser::parse("let a = 1 2\nlet s = \"abc\nlet b = 1 2\n", &mut i).unwrap_err();
    assert_eq!(e.len(), 2, "{e}");
    assert!(e[1].message.contains("unclosed string"));
}

#[test]
fn private_and_protected() {
    assert_eq!(
        sexp(
            "class A\n  private def f() = 1\n  protected def g = 2\n  @[X] private def h=(v)\n  end\nend"
        ),
        "(class A (private def f () 1) (protected def g 2) @[X] (private def h= (v)))"
    );
    assert_eq!(
        sexp("trait T\n  private abstract def f(): int\nend"),
        "(trait T (abstract (private def f () : int)))"
    );
    assert_eq!(
        sexp(
            "module M\n  private let x = 1\n  private class C\n  end\n  private module N\n  end\nend"
        ),
        "(module M (private (let x 1)) (private (class C)) (private (module N)))"
    );
    assert!(err("class A\n  protected x: int\nend").contains("expected `def` after `protected`"));
}
