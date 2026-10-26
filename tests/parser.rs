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
        Ok(p) => panic!("devrait échouer : {}", printer::program(&p, &i)),
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
            "case port\n  when 80, 443 then \"web\"\n  when 8000..8999\n    \"dev\"\n  else \"autre\"\nend"
        ),
        "(case port (when (80 443) \"web\") (when ((.. 8000 8999)) \"dev\") (else \"autre\"))"
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
    assert!(err("1 +").contains("expression attendue"));
    assert!(err("if x\n  1\n").contains("`end` attendu pour fermer `if` (ligne 1)"));
    assert!(err("a < b < c").contains("ajouter des parenthèses"));
    assert!(err("1 = 2").contains("assigner"));
    assert!(err("{\"a\": 1}").contains("`=>` attendu"));
    assert!(err("f(a: 1, 2)").contains("positionnel après un argument nommé"));
    assert!(err("f(a: 1, a: 2)").contains("deux fois"));
    assert!(err("{x: 1, x: 2}").contains("deux fois"));
    assert!(err("while x do\nend").contains("`do` n'est pas utilisé"));
    assert!(err("1.(").contains("nom de méthode"));
    assert!(err("@[Tag] x = 1").contains("annotation"));
    assert!(err("x y").contains("fin de ligne attendue"));
    assert!(err("\"abc #{x\"").contains("non fermée"));

    // Les erreurs en fin de source sont marquées « incomplètes » (le REPL lit la suite).
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
    // 900 niveaux passent (la pile grandit sur le tas si besoin)…
    let src = "(".repeat(900) + "1" + &")".repeat(900);
    assert_eq!(sexp(&src), "1");
    let src = "[".repeat(900) + &"]".repeat(900);
    assert!(sexp(&src).starts_with("(list (list"));
    // …au-delà de la limite, une erreur propre plutôt qu'un débordement de pile.
    let src = "(".repeat(10_000) + "1" + &")".repeat(10_000);
    assert!(err(&src).contains("imbrication trop profonde"));
}

/// Chaque bloc ```ruby de docs/syntax.md doit être accepté par le parseur.
/// Sauf : les macros (`{%`, `{{`) et les blocs à trous (`…`).
#[test]
fn spec_examples_parse() {
    let spec = include_str!("../docs/syntax.md");
    let mut count = 0;
    let mut rest = spec;
    while let Some(start) = rest.find("```ruby\n") {
        let body = &rest[start + 8..];
        let end = body.find("```").expect("bloc non fermé");
        let code = &body[..end];
        rest = &body[end + 3..];
        if code.contains("{%") || code.contains("{{") || code.contains('…') {
            continue;
        }
        let mut i = Interner::new();
        if let Err(e) = parser::parse(code, &mut i) {
            panic!(
                "exemple de syntax.md refusé :\n{}\n{code}",
                e.render(code, "syntax.md")
            );
        }
        count += 1;
    }
    assert!(count >= 15, "seulement {count} exemples trouvés");
}

/// Lignes (à partir de 1) des erreurs signalées.
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
    // L'en-tête cassé d'un `def` / `if` / `case` : son corps et son `end` sont sautés,
    // sans erreur en cascade.
    assert_eq!(
        error_lines("def f(a b)\n  if a\n    1\n  end\nend\nf(1)\n"),
        [1]
    );
    assert_eq!(
        error_lines("case x y\nwhen 1 then 2\nend\nlet z = 1 +\n"),
        [1, 4]
    );
    // `if` en modificateur dans la partie sautée : n'ouvre pas de construction.
    assert_eq!(error_lines("let a = 1 2 if b\nlet c = (\n"), [1, 2]);
    // Un bloc `do` ouvert sur la ligne erronée est sauté jusqu'à son `end`.
    assert_eq!(error_lines("foo(1 2) do |x|\n  x\nend\nbar(\n"), [1, 4]);
}

#[test]
fn recovery_inside_bodies() {
    // Les erreurs dans un corps sont signalées et le reste du corps est analysé.
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
    // Chaîne non fermée : les erreurs du parseur situées après n'ont pas de sens.
    let e = parser::parse("let a = 1 2\nlet s = \"abc\nlet b = 1 2\n", &mut i).unwrap_err();
    assert_eq!(e.len(), 2, "{e}");
    assert!(e[1].message.contains("chaîne non fermée"));
}
