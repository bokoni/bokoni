//! Static checking: errors reported before execution, and no false positives.

use bokoni::Engine;

/// Static error messages (the program is not run if there are any).
fn errors(src: &str) -> Vec<String> {
    let mut e = Engine::new();
    e.set_output(|_| {});
    match e.eval(src) {
        Ok(_) => Vec::new(),
        Err(errs) => errs.iter().map(|e| e.message.clone()).collect(),
    }
}

/// The program passes the checks AND runs without error.
fn ok(src: &str) {
    let mut e = Engine::new();
    e.set_output(|_| {});
    if let Err(errs) = e.eval(src) {
        panic!("false positive?\n{}", errs.render(src, "test"));
    }
}

fn has(errs: &[String], needle: &str) -> bool {
    errs.iter().any(|e| e.contains(needle))
}

#[test]
fn reports_all_errors_before_running() {
    let errs = errors(
        r#"print("must not be printed")
let x = 1
x = "a"
y = 2
let n: int = "three"
def add(a: int, b: int): int = a + b
add(1)
add(1, "2")
"abc".lenght()
let l = [1, 2]
l.push("a")
print(unknown)
"#,
    );
    assert_eq!(errs.len(), 8, "{errs:#?}");
    assert!(has(&errs, "expected type `int`, got `str`")); // x = "a"
    assert!(has(&errs, "`y` is not declared"));
    assert!(has(&errs, "missing argument `b`"));
    assert!(has(&errs, "expected type `int` for `b`, got `str`"));
    assert!(has(
        &errs,
        "`str` has no method `lenght` (did you mean `len`?)"
    ));
    assert!(has(&errs, "expected type `int`, got `str`"));
    assert!(has(&errs, "`unknown` is not defined"));
}

#[test]
fn names() {
    assert!(has(
        &errors("let total = 1\nprint(totla)"),
        "(did you mean `total`?)"
    ));
    assert!(has(
        &errors("print(x)\nlet x = 1"),
        "`x` is used before its declaration"
    ));
    assert!(has(&errors("MAX = 1\nMAX = 2"), "cannot be modified"));
    assert!(has(&errors("def f = 1\nf = 2"), "cannot be modified"));
    assert!(has(&errors("break"), "`break` outside of a loop"));
    assert!(has(&errors("@x"), "only be used inside a method"));
    assert!(has(&errors("let s: Servr = 1"), "unknown type `Servr`"));
    assert!(has(
        &errors("let s: strr = 1"),
        "(did you mean `str`?)"
    ));
    // A function may use a variable declared further down (called afterwards).
    ok("def show() = config + 1\nlet config = 1\nshow()");
    // Block scope.
    assert!(has(
        &errors("if true\n  let z = 1\nend\nprint(z)"),
        "`z` is not defined"
    ));
}

#[test]
fn inference() {
    // A variable's type is that of its first value.
    assert!(has(
        &errors("let x = 1\nx = 2.5"),
        "expected type `int`, got `float`"
    ));
    ok("let x = 1.5\nx = 2"); // int → float accepted
    ok("let x = nil\nx = 1\nx = \"a\""); // nil alone: unknown type
    ok("let xs = []\nxs << 1\nxs << \"a\"");
    assert!(has(
        &errors("let xs = [1, 2]\nxs << \"a\""),
        "`<<` appends `str` to a list of `int`"
    ));
    // Return type inferred from the body.
    assert!(has(
        &errors("def five = 5\nfive() + \"a\""),
        "`+` between `int` and `str`"
    ));
    // Lambda parameters inferred from the elements.
    assert!(has(
        &errors("[1, 2].map(|n| n.upper())"),
        "`int` has no method `upper`"
    ));
    assert!(has(
        &errors("[\"a\"].map(|s| s.len()).map(|n| n.upper())"),
        "`int` has no method `upper`"
    ));
    assert!(has(
        &errors("{\"a\" => 1}.each do |k, v|\n  v.upper()\nend"),
        "`int` has no method `upper`"
    ));
    assert!(has(
        &errors("for i in 0..3\n  i.upper()\nend"),
        "`int` has no method `upper`"
    ));
}

#[test]
fn gradual_typing_has_no_false_positives() {
    // Unannotated parameters: `any`, anything goes.
    ok("def f(x) = x.whatever_method()\nlet g = |a| a.b.c");
    // A value that *may* fit is not rejected (checked at runtime).
    ok("let m: \"dev\" | \"prod\" = \"dev\"\nlet s = \"prod\"\nm = s");
    ok("let h = {\"a\" => 1}\nlet v = h[\"a\"]\nlet n: int = v");
    ok("let xs = [1, \"a\"]\nlet first: int = xs[0]");
    ok("let x: int | str = 1\nx = \"a\"");
    // Rust functions: unknown type.
    let mut e = Engine::new();
    e.register_fn("ext", |x: i64| x);
    e.eval("let r = ext(1)\nr.anything").unwrap_err(); // error… at runtime only
    assert!(
        e.eval("let r: str = ext(1)").unwrap_err()[0]
            .message
            .contains("expected type `str`")
    );
    // `case` on a constant.
    ok("let k = case 8080\n  when 80 then 1\n  else 2\nend");
}

#[test]
fn types_and_classes() {
    let route = "type Route\n  path: str\n  methods: [str] = [\"GET\"]\nend\n";
    let e = |src: &str| errors(&format!("{route}{src}"));
    assert!(has(&e("Route()"), "field `path` is required"));
    assert!(has(
        &e("Route(path: \"/\", verb: 1)"),
        "`Route` has no field `verb`"
    ));
    assert!(has(
        &e("Route(path: 1)"),
        "expected type `str` for field `path`"
    ));
    assert!(has(
        &e("Route do\n  pth = \"/\"\nend"),
        "(did you mean `path`?)"
    ));
    assert!(has(
        &e("Route(path: \"/\").pth"),
        "(did you mean `path`?)"
    ));
    assert!(has(
        &e("def reg(r: Route) = r\nreg({path: \"/\", verb: 1})"),
        "`Route` has no field `verb`"
    ));
    assert!(has(
        &e("def reg(r: Route) = r\nreg({methods: []})"),
        "field `path` is required"
    ));
    assert!(has(
        &e("let rs: [Route] = [{path: 1}]"),
        "expected type `str` for field `path`"
    ));
    // A block may fill in the required fields: no static error.
    ok(&format!(
        "{route}let r = Route do\n  path = \"/\"\nend\nr.methods.len()"
    ));
    ok(&format!(
        "{route}def reg(r: Route) = r.path\nreg({{path: \"/\"}})"
    ));

    let user = "class User\n  def init(@name: str)\n  end\n  def name: str = @name\n  def rename(n: str) = @name = n\nend\n";
    let e = |src: &str| errors(&format!("{user}{src}"));
    assert!(has(&e("User(1)"), "expected type `str` for `name`"));
    assert!(has(
        &e("User(\"a\").rename(1)"),
        "expected type `str` for `n`"
    ));
    assert!(has(
        &e("User(\"a\").name.len() + \"x\""),
        "`+` between `int` and `str`"
    ));
    assert!(has(
        &errors("class C\n  def init(@n: int)\n  end\n  def get = @m\nend"),
        "`C` has no field `@m`"
    ));
    assert!(has(
        &errors("class C\n  def init(@n: int)\n  end\nend\nC(1).n"),
        "private field"
    ));
    ok(&format!(
        "{user}let u = User(\"a\")\nu.rename(\"b\")\nu.name.upper()"
    ));
}

#[test]
fn enums() {
    let en = "enum Protocol: str\n  Tcp = \"TCP\"\n  Udp = \"UDP\"\nend\n";
    let e = |src: &str| errors(&format!("{en}{src}"));
    assert!(has(&e("Protocol::Tpc"), "(did you mean `Tcp`?)"));
    assert!(has(
        &e("Protocol::Tcp.value + 1"),
        "`+` between `str` and `int`"
    ));
    assert!(has(
        &e("let p: Protocol = \"TCP\""),
        "expected type `Protocol`, got `str`"
    ));
    assert!(has(
        &errors("enum E: int\n  A = \"a\"\nend"),
        "expected type `int` for variant `A`"
    ));
    ok(&format!(
        "{en}let p = Protocol.from(\"TCP\")\nProtocol.values().len()"
    ));
}

#[test]
fn annotations() {
    assert!(has(
        &errors("@[Unknown]\ndef f = 1"),
        "unknown annotation `Unknown`"
    ));
    assert!(has(
        &errors("annotation Validate\n  min: int? = nil\nend\n@[Validate(min: \"a\")]\ndef f = 1"),
        "expected type `int?` for field `min`"
    ));
    ok("annotation Tag\nend\n@[Tag]\n@[Deprecated(\"use g\")]\ndef f = 1");
}

#[test]
fn checks_use_earlier_definitions() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("type Server\n  port: int = 80\nend\ndef double(x: int): int = x * 2")
        .unwrap();
    // The type and function defined earlier are known to the checker.
    let errs = e.eval("Server do\n  prot = 1\nend").unwrap_err();
    assert!(
        errs[0].message.contains("(did you mean `port`?)"),
        "{errs}"
    );
    assert!(
        e.eval("double(\"a\")").unwrap_err()[0]
            .message
            .contains("expected type `int`")
    );
    assert_eq!(
        e.eval_as::<i64>("Server do\n  port = 1\nend.port").unwrap(),
        1
    );
}

#[test]
fn static_checks_can_be_disabled() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.set_static_checks(false);
    // Without static checks, the error happens at runtime (after the print).
    let out = std::rc::Rc::new(std::cell::RefCell::new(String::new()));
    let sink = out.clone();
    e.set_output(move |s| sink.borrow_mut().push_str(s));
    let errs = e.eval("print(\"before\")\nlet x: int = \"a\"").unwrap_err();
    assert_eq!(errs.len(), 1);
    assert_eq!(out.borrow().as_str(), "before\n");
}

/// Varied correct programs: no error (neither static nor at runtime).
#[test]
fn valid_programs_pass() {
    ok(r##"
enum Protocol: str
  Tcp = "TCP"
  Udp = "UDP"
end

type Route
  path: str
  methods: [str] = ["GET"]
end

type Server
  host: str = "localhost"
  protocol = Protocol::Tcp
  port: int = if protocol == Protocol::Tcp then 80 else 40 end
  debug: bool = false
  mode: "dev" | "prod" = "dev"
  routes: [Route] = []
end

let env = "prod"
let s = Server do
  let base = 8000
  host = "api.example.com"
  mode = env
  port = if mode == "prod" then 443 else base + 80 end
  debug = mode == "dev"
  routes << Route do
    path = "/users"
    methods = ["GET", "POST"]
  end
  for p in ["/health", "/metrics"]
    routes << Route(path: p)
  end
end

def register(r: Route): str = r.path
register({path: "/x"})
let rs: [Route] = [{path: "/a"}, {path: "/b"}]

class User
  def init(@name: str, @email: str? = nil)
  end
  def name: str = @name
  def name=(value: str)
    @name = value.capitalize()
  end
  def to_s: str = "User(#{@name})"
end

let u = User("vincent")
u.name = "bob"
print(u.name, u)

def fib(n: int): int
  if n < 2 then n else fib(n - 1) + fib(n - 2) end
end
let evens = [1, 2, 3, 4].filter(|n| n % 2 == 0).map(|n| n * n)
let ages = {"alice" => 30}
ages["bob"] = 25
for name, age in ages
  print("#{name.capitalize()} is #{age}")
end
let {x, y} = {x: 3, y: 4}
let d = (x * x + y * y).sqrt()
let (lo, hi) = ([3, 1].min(), [3, 1].max())
let total = 0
[1, 2, 3].each do |n|
  total += n
end
let kind = case 8080
  when 80, 443 then "web"
  when 8000..8999 then "dev"
  else "other"
end
def counter()
  let n = 0
  || n += 1
end
let c = counter()
c()
"##);
}

#[test]
fn nil_tests_refine_types() {
    // In each branch, the variable's type without `nil`, or `nil`: errors found there.
    for (src, msg) in [
        (
            "def f(a: int?): int\n  if a.nil?\n    a + 1\n  else\n    0\n  end\nend",
            "`+` does not apply to `nil` and `int`",
        ),
        (
            "def f(a: int?): str\n  if a != nil\n    a.upper()\n  else\n    \"\"\n  end\nend",
            "`int` has no method `upper`",
        ),
        (
            "def f(a: int?): bool = a != nil and a.upper()",
            "`int` has no method `upper`",
        ),
        (
            "def f(a: int?): bool = a == nil or a.upper()",
            "`int` has no method `upper`",
        ),
        // After a branch that does not finish: the other condition.
        (
            "def f(a: str?): int\n  return 0 if a.nil?\n  a.nope()\nend",
            "`str` has no method `nope`",
        ),
        (
            "def f(a: int?): int\n  unless a\n    raise \"none\"\n  end\n  let s: str = a\n  0\nend",
            "expected type `str`, got `int`",
        ),
        // `is_a?`, `responds_to?`: in both branches, with `and`, after a `return`.
        (
            "def f(v: int | str): int\n  if v.is_a?(int)\n    v.len()\n  else\n    0\n  end\nend",
            "`int` has no method `len`",
        ),
        (
            "def f(v: int | str): int\n  if v.is_a?(int)\n    0\n  else\n    v + 1\n  end\nend",
            "`+` between `str` and `int`",
        ),
        (
            "def f(v: int | str): int\n  return v.len() unless v.is_a?(int)\n  v.upper()\nend",
            "`int` has no method `upper`",
        ),
        (
            "class A\n  def a = 1\nend\nclass B < A\n  def b = 2\nend\ndef f(x: A): int\n  if x.is_a?(B) then x.b + x.nope else x.a end\nend",
            "`B` has no field `nope`",
        ),
        (
            "def f(v: int | str): int\n  if v.responds_to?(abs) then v.len() else 0 end\nend",
            "`int` has no method `len`",
        ),
        // `case` by type: each arm, the arms not taken, a pattern that cannot match.
        (
            "def f(v: int | str): int\n  case v\n  when int then v.len()\n  else 0\n  end\nend",
            "`int` has no method `len`",
        ),
        (
            "def f(v: int | str | nil): int\n  case v\n  when int, nil then 0\n  else v + 1\n  end\nend",
            "`+` between `str` and `int`",
        ),
        (
            "def f(v: str): int\n  case v\n  when int then 1\n  else 2\n  end\nend",
            "this pattern (`int`) cannot match a value of type `str`",
        ),
        // `if let a = @a`: a local copy, refined.
        (
            "class A\n  def init(@a: int?)\n  end\n  def f(): int\n    if let a = @a then a.upper() else 0 end\n  end\nend",
            "`int` has no method `upper`",
        ),
        (
            "class A\n  def init(@a: int | str)\n  end\n  def f(): int\n    if (let a = @a).is_a?(str) then a + 1 else a end\n  end\nend",
            "`+` between `str` and `int`",
        ),
        // `as(T)` that can never succeed.
        ("\"x\".as(int)", "`str` can never be `int`"),
        // `??` and `?.`: the type without `nil`, or optional.
        (
            "def f(a: int?): int\n  let n = a ?? 0\n  let s: str = n\n  0\nend",
            "expected type `str`, got `int`",
        ),
    ] {
        let errs = errors(src);
        assert!(
            has(&errs, msg),
            "expected `{msg}` for:\n{src}\ngot {errs:?}"
        );
    }
    // An assignment, a loop, a lambda: no longer refined (no false positive).
    ok("def f(a: int?): int\n  return 0 if a.nil?\n  a = nil\n  a ?? 1\nend\nf(nil) + f(2)");
    ok(
        "def f(a: int?): int\n  if a == nil\n    let i = 0\n    while i < 2\n      i += 1\n      a = i\n    end\n    return a + 1\n  end\n  a\nend\nf(nil) + f(2)",
    );
    ok(
        "def f(a: int?): int\n  if a == nil\n    a = 5\n    return a + 1\n  end\n  a\nend\nf(nil) + f(2)",
    );
    ok(
        "def f(a: int?)\n  if a == nil\n    let g = || a + 1\n    a = 1\n    return g()\n  end\n  a\nend\nf(nil)",
    );
    ok("let a: int? = nil\nlet r = [a.nil?, 3.nil?, \"x\".nil?]\nr");
    ok(
        "def f(v: int | str): int\n  if v.is_a?(int)\n    v + 1\n  else\n    v.len()\n  end\nend\nf(1) + f(\"ab\")",
    );
    // Every type covered: no `nil` without `else`.
    ok(
        "def f(v: int | str): int\n  case v\n  when int then v + 1\n  when str then v.len()\n  end\nend\nf(1) + f(\"ab\")",
    );
}
