use bokoni::{Engine, ErrorKind, Value};
use std::cell::RefCell;
use std::rc::Rc;

/// Runs `src`; returns the `print` output followed by the result (`=> …`) unless it is nil.
fn run(src: &str) -> String {
    let mut e = Engine::new();
    let out = Rc::new(RefCell::new(String::new()));
    let sink = out.clone();
    e.set_output(move |s| sink.borrow_mut().push_str(s));
    match e.eval(src) {
        Ok(v) => {
            let mut s = out.borrow().clone();
            if !matches!(v, Value::Nil) {
                s.push_str(&format!("=> {}", e.inspect(&v)));
            }
            s.trim_end().to_string()
        }
        Err(err) => panic!("{}", err.render(src, "test")),
    }
}

fn err(src: &str) -> (ErrorKind, String) {
    let mut e = Engine::new();
    e.set_output(|_| {});
    match e.eval(src) {
        Ok(v) => panic!("should fail, gave {}", e.inspect(&v)),
        // The kind of the first error; the messages of all.
        Err(err) => (err[0].kind, err.to_string()),
    }
}

#[test]
fn arithmetic() {
    assert_eq!(run("1 + 2 * 3"), "=> 7");
    assert_eq!(run("7 / 2"), "=> 3");
    assert_eq!(run("-7 / 2"), "=> -4");
    assert_eq!(run("-7 % 3"), "=> 2");
    assert_eq!(run("7 / 2.0"), "=> 3.5");
    assert_eq!(run("2.0 * 3"), "=> 6.0");
    assert_eq!(run("\"ab\" + \"cd\""), "=> \"abcd\"");
    assert_eq!(run("\"ab\" * 3"), "=> \"ababab\"");
    assert_eq!(run("[1] + [2, 3]"), "=> [1, 2, 3]");
    assert_eq!(run("1 == 1.0"), "=> true");
    assert_eq!(run("\"a\" < \"b\""), "=> true");
    assert_eq!(run("not nil"), "=> true");
    assert_eq!(run("nil or \"default\""), "=> \"default\"");
    assert_eq!(run("1 and 2"), "=> 2");
    assert!(err("1 / 0").1.contains("division by zero"));
    assert!(err("9223372036854775807 + 1").1.contains("overflow"));
    assert_eq!(err("1 < \"a\"").0, ErrorKind::Type);
}

#[test]
fn variables_and_constants() {
    assert_eq!(run("let x = 1\nx = x + 1\nx += 10\nx"), "=> 12");
    assert_eq!(run("MAX = 10\nMAX * 2"), "=> 20");
    assert!(err("x = 1").1.contains("use `let x = …`"));
    assert!(
        err("MAX = 1\nMAX = 2")
            .1
            .contains("cannot be modified")
    );
    assert!(err("y").1.contains("`y` is not defined"));
    // Block scope: a `let` inside an `if` does not leak.
    assert!(
        err("if true\n  let z = 1\nend\nz")
            .1
            .contains("`z` is not defined")
    );
    // But an outer variable can be modified.
    assert_eq!(run("let n = 0\nif true\n  n = 5\nend\nn"), "=> 5");
}

#[test]
fn strings() {
    assert_eq!(
        run(r##"let n = "world"
"Hello #{n.upper()} (#{1 + 1})""##),
        "=> \"Hello WORLD (2)\""
    );
    assert_eq!(run(r#""a,b,c".split(",")"#), r#"=> ["a", "b", "c"]"#);
    assert_eq!(run(r#""  x ".trim().len()"#), "=> 1");
    assert_eq!(run(r#""héllo".len()"#), "=> 5");
    assert_eq!(run(r#""abc"[-1]"#), "=> \"c\"");
    assert_eq!(run(r#""42".to_i() + 1"#), "=> 43");
    assert_eq!(run(r#""bob".capitalize()"#), "=> \"Bob\"");
    assert_eq!(run(r#""abc".reverse()"#), "=> \"cba\"");
}

#[test]
fn functions() {
    assert_eq!(
        run("def add(a: int, b: int = 10): int = a + b\nadd(1)"),
        "=> 11"
    );
    assert_eq!(run("def add(a, b) = a - b\nadd(b: 1, a: 10)"), "=> 9");
    // Call before definition and mutual recursion.
    assert_eq!(
        run(
            "def even(n) = if n == 0 then true else odd(n - 1) end\ndef odd(n) = if n == 0 then false else even(n - 1) end\neven(10)"
        ),
        "=> true"
    );
    assert_eq!(
        run("def f(x)\n  return 1 if x > 0\n  2\nend\n[f(5), f(-5)]"),
        "=> [1, 2]"
    );
    assert!(
        err("def f(a) = a\nf(1, 2)")
            .1
            .contains("expects 1 argument, got 2")
    );
    assert!(
        err("def f(a, b) = a\nf(1)")
            .1
            .contains("missing argument `b`")
    );
    assert!(
        err("def f(a) = a\nf(z: 1)")
            .1
            .contains("has no parameter `z`")
    );
    assert!(
        err("def f = 1\nf = 2")
            .1
            .contains("cannot be modified")
    );
}

#[test]
fn closures() {
    assert_eq!(
        run("def counter()\n  let n = 0\n  || n += 1\nend\nlet c = counter()\nc()\nc()\nc()"),
        "=> 3"
    );
    assert_eq!(run("let add = |a, b| a + b\nadd(2, 3)"), "=> 5");
    assert_eq!(
        run("let k = 10\nlet f = |x| x * k\n[1, 2].map(f)"),
        "=> [10, 20]"
    );
    // Lambdas are lenient: a missing argument is nil.
    assert_eq!(run("let f = |a, b| b\nf(1)"), "");
    assert_eq!(run("let f = |x| x + 1\nf.call(1)"), "=> 2");
}

#[test]
fn blocks() {
    assert_eq!(
        run("let t = 0\n[1, 2, 3].each do |n|\n  t += n\nend\nt"),
        "=> 6"
    );
    assert_eq!(
        run("def twice(f) = [f(1), f(2)]\ntwice() do |x|\n  x * 10\nend"),
        "=> [10, 20]"
    );
    assert_eq!(
        run("[1, 2, 3, 4].filter(|n| n % 2 == 0).map(|n| n * n)"),
        "=> [4, 16]"
    );
    assert_eq!(run("[1, 2, 3].reduce(0, |acc, n| acc + n)"), "=> 6");
    assert_eq!(run("[3, 1, 2].find(|n| n < 3)"), "=> 1");
    assert_eq!(run("(1..4).map(|n| n * 2)"), "=> [2, 4, 6, 8]");
}

#[test]
fn control_flow() {
    assert_eq!(
        run("if 1 > 2 then \"a\" elsif 2 > 1 then \"b\" else \"c\" end"),
        "=> \"b\""
    );
    assert_eq!(run("unless false\n  1\nend"), "=> 1");
    // `c ? a : b` (nested to the right), `a ?? b` (`b` only for `nil`, `a` evaluated
    // once), next to the `?` variants.
    assert_eq!(
        run("let n = 0\n[n > 0 ? \"pos\" : n == 0 ? \"zero\" : \"neg\", true ?\n  1 : 2]"),
        "=> [\"zero\", 1]"
    );
    assert_eq!(
        run("let xs = [1]\nlet k = 0\ndef f(): int?\n  k += 1\n  nil\nend\n[xs[3]? ?? 9, xs[0]? ?? 9, false ?? 1, f() ?? 2, k, nil ?? 1 + 1, nil ?? 0 > 5, \"7\".to_i?() ?? 0]"),
        "=> [9, 1, false, 2, 1, 2, false, 7]"
    );
    assert!(err("true ? 1").1.contains("between the two values of `? :`"));
    // Type reflection: `is_a?`, `as`, `as?`, `responds_to?`.
    let classes = "trait Greet\n  def hi(): str = \"hi\"\nend\nclass Animal\n  def name = \"animal\"\nend\nclass Dog < Animal\n  include Greet\n  def bark = \"woof\"\nend\ntype Point\n  x: int = 0\nend\ndef id(x) = x\nlet d = Dog()\n";
    assert_eq!(
        run(&format!(
            "{classes}[d.is_a?(Dog), d.is_a?(Animal), d.is_a?(Greet), Animal().is_a?(Dog), 1.is_a?(int | str), nil.is_a?(int?), [1, \"a\"].is_a?([int]), [1].is_a?([int])]"
        )),
        "=> [true, true, true, false, true, true, false, true]"
    );
    assert_eq!(
        run(&format!(
            "{classes}let m: int | str = \"abc\"\nlet n: int? = nil\n[m.as(str).len(), id(1).as?(str), \"x\".as?(str), n.as?(int) ?? 7, d.as(Animal).name]"
        )),
        "=> [3, nil, \"x\", 7, \"animal\"]"
    );
    assert_eq!(
        run(&format!(
            "{classes}[1.responds_to?(abs), \"s\".responds_to?(abs), d.responds_to?(bark), d.responds_to?(hi), Animal().responds_to?(bark), Point().responds_to?(x), {{a: 1}}.responds_to?(a), [1].responds_to?(len), nil.responds_to?(nil?)]"
        )),
        "=> [true, false, true, true, false, true, true, true, true]"
    );
    // `case` by type (`MAX`, `Color::Red`: values).
    assert_eq!(
        run(&format!(
            "{classes}enum Color\n  Red\n  Blue\nend\nMAX = 3\ndef k(v) = case v\n  when int | float then \"num\"\n  when Dog then \"dog \" + v.bark\n  when Animal, nil then \"animal\"\n  when str then \"str\"\n  else \"other\"\nend\n[k(1), k(2.5), k(d), k(Animal()), k(nil), k(\"s\"), k(true), case 3 when MAX then 1 else 2 end, case Color::Blue when Color::Red then 1 when Color::Blue then 2 end, case [1].len() when int then \"i\" end]"
        )),
        "=> [\"num\", \"num\", \"dog woof\", \"animal\", \"animal\", \"str\", \"other\", 1, 2, \"i\"]"
    );
    assert!(
        err("def id(x) = x\nid(\"x\").as(int)")
            .1
            .contains("expected type `int`, got str")
    );
    // `a?.b`: `nil` for a `nil` `a`, the rest of the chain skipped, `a` evaluated once.
    assert_eq!(
        run("class U\n  def init(@n: str, @up: U?)\n  end\n  def n = @n\n  def up = @up\n  def hi(x: str): str = x + @n\nend\nlet a = U(\"a\", nil)\nlet b = U(\"b\", a)\nlet k = 0\ndef f(u: U?): U?\n  k += 1\n  u\nend\nlet none: U? = nil\nlet ps = [{x: 1}]\n[b.up?.n, a.up?.n, a.up?.up?.n, none?.up.n, none?.n ?? \"?\", f(b)?.hi(\"h\"), f(nil)?.hi(\"h\"), k, ps[0]?.x, ps[5]?.x, b\n  ?.up\n  ?.n]"),
        "=> [\"a\", nil, nil, nil, \"?\", \"hb\", nil, 2, 1, nil, \"a\"]"
    );
    // `a?[i]` (`c ?[1] : [2]`: a ternary).
    assert_eq!(
        run("let xs: [int]? = [1, 2]\nlet none: [int]? = nil\nlet h: {str => [int]}? = {\"a\" => [5]}\nlet c = true\n[xs?[0], none?[0], none?[0].abs(), xs?[1].abs(), h?[\"a\"]?[0], none?[0] ?? 9, c ?[1] : [2], xs?[5]?]"),
        "=> [1, nil, nil, 2, 5, 9, [1], nil]"
    );
    // A method ending with `?`, without parentheses.
    assert_eq!(
        run("class U\n  def ok? = true\nend\n[\"5\".to_i?, \"x\".to_i? ?? 0, U().ok?, \"5\".to_i?().abs()]"),
        "=> [5, 0, true, 5]"
    );
    assert!(err("let x: int = false ? 1 : \"a\"").1.contains("expected type `int`"));
    assert_eq!(run("let x = 5\nx = 0 if x > 3\nx"), "=> 0");
    assert_eq!(
        run(
            "let out = []\nfor i in 0..10\n  next if i % 2 == 0\n  break if i > 7\n  out << i\nend\nout"
        ),
        "=> [1, 3, 5, 7]"
    );
    assert_eq!(run("let i = 0\nwhile i < 5\n  i += 1\nend\ni"), "=> 5");
    assert_eq!(run("let s = 0\nfor i in 0...3\n  s += i\nend\ns"), "=> 3");
    assert_eq!(
        run(
            "def k(p) = case p\n  when 80, 443 then \"web\"\n  when 8000..8999 then \"dev\"\n  else \"other\"\nend\n[k(443), k(8080), k(1)]"
        ),
        "=> [\"web\", \"dev\", \"other\"]"
    );
    assert!(err("break").1.contains("`break` outside"));
    assert!(err("for x in 5\nend").1.contains("cannot iterate"));
}

#[test]
fn collections() {
    assert_eq!(run("let l = [1, 2, 3]\nl[-1]"), "=> 3");
    assert_eq!(run("let l = [1, 2]\nl[0] = 9\nl << 3\nl"), "=> [9, 2, 3]");
    assert!(err("[1][5]").1.contains("out of bounds"));
    assert_eq!(run("[3, 1, 2].sort()"), "=> [1, 2, 3]");
    assert_eq!(run("[\"b\", \"a\"].sort().join(\"-\")"), "=> \"a-b\"");
    assert_eq!(run("[2, 9, 4].max()"), "=> 9");
    assert_eq!(run("[1, 2].contains(2)"), "=> true");
    // Lists are shared (like Ruby, Python…).
    assert_eq!(run("let a = [1]\nlet b = a\nb << 2\na"), "=> [1, 2]");
}

#[test]
fn hashes() {
    assert_eq!(
        run("let h = {\"a\" => 1}\nh[\"b\"] = 2\nh[\"a\"] += 10\nh"),
        "=> {\"a\" => 11, \"b\" => 2}"
    );
    assert_eq!(run("let h = {\"a\" => 1}\nh[\"zz\"]"), "");
    assert_eq!(
        run("let h = {1 => \"one\"}\n[h.has(1), h.get(2, \"?\"), h.keys()]"),
        "=> [true, \"?\", [1]]"
    );
    assert_eq!(
        run("let h = {\"a\" => 1, \"b\" => 2, \"c\" => 3}\nh.delete(\"b\")\nh.keys()"),
        "=> [\"a\", \"c\"]"
    );
    assert_eq!(
        run(
            "let h = {\"x\" => 1, \"y\" => 2}\nlet s = []\nfor k, v in h\n  s << \"#{k}=#{v}\"\nend\ns.join(\" \")"
        ),
        "=> \"x=1 y=2\""
    );
    assert!(err("{[1] => 2}").1.contains("hash key"));
    // `for` with a pattern, as `let`: a tuple, nested, a record, a list.
    assert_eq!(
        run(
            "let h = {\"x\" => 1}\nlet s = []\nfor (k, v) in h\n  s << \"#{k}=#{v}\"\nend\nfor (a, b) in [(1, 2), (3, 4)]\n  s << a * b\nend\ns"
        ),
        "=> [\"x=1\", 2, 12]"
    );
    assert_eq!(
        run(
            "let s = []\nfor (i, (a, b)) in [(0, (1, 2)), (1, (3, 4))]\n  s << i * 100 + a * 10 + b\nend\nfor i, {x, y} in [(5, {x: 1, y: 2})]\n  s << i + x + y\nend\nfor [a, ...r] in [[1, 2, 3]]\n  s << r\nend\ns"
        ),
        "=> [12, 134, 8, [2, 3]]"
    );
    assert!(
        err("for (i, (a, b)) in [(0, 5)]\n  a\nend")
            .1
            .contains("expected a tuple or list, got int")
    );
}

#[test]
fn records_and_tuples() {
    assert_eq!(run("let p = {x: 1, y: 2}\np.x = 5\np"), "=> {x: 5, y: 2}");
    assert!(
        err("let p = {x: 1}\np.z = 3")
            .1
            .contains("has no field `z`")
    );
    assert_eq!(run("let x = 10\n{x, y: 20}"), "=> {x: 10, y: 20}");
    assert_eq!(run("let p = {x: 3, y: 4}\nlet {x, y} = p\nx * y"), "=> 12");
    assert_eq!(run("let t = (1, \"one\")\nt.1"), "=> \"one\"");
    assert_eq!(run("let (a, b) = (1, 2)\na + b"), "=> 3");
    assert_eq!(run("let [h, ...rest] = [1, 2, 3]\nrest"), "=> [2, 3]");
    // A function field is called like a method.
    assert_eq!(run("let o = {double: |x| x * 2}\no.double(21)"), "=> 42");
    assert!(err("[1].len").1.contains("write `.len()`"));
}

#[test]
fn type_annotations() {
    assert_eq!(run("let x: int = 1\nx"), "=> 1");
    assert_eq!(run("let x: float = 1\nx"), "=> 1.0");
    assert_eq!(run("let x: int? = nil\nx"), "");
    assert_eq!(run("let m: \"dev\" | \"prod\" = \"dev\"\nm"), "=> \"dev\"");
    assert_eq!(run("let h: {str => int} = {\"a\" => 1}\nh.len()"), "=> 1");
    assert_eq!(run("let p: {x: int, y: int} = {x: 1, y: 2}\np.y"), "=> 2");
    assert_eq!(run("let t: (int, str) = (1, \"a\")\nt.0"), "=> 1");
    let (kind, msg) = err("let x: int = \"a\"");
    assert_eq!(kind, ErrorKind::Type);
    assert!(msg.contains("expected type `int`, got `str`"), "{msg}");
    assert!(
        err("let m: \"dev\" | \"prod\" = \"test\"")
            .1
            .contains("expected type")
    );
    assert!(
        err("def f(n: int) = n\nf(\"a\")")
            .1
            .contains("expected type `int`")
    );
    assert!(
        err("def f(): str = 1\nf()")
            .1
            .contains("expected type `str`")
    );
    assert!(err("let s: Server = 1").1.contains("unknown type `Server`"));
    assert!(
        err("let s: Sever = 1")
            .1
            .contains("unknown type `Sever`")
    );
}

const ROUTE: &str = "type Route\n  path: str\n  methods: [str] = [\"GET\"]\nend\n";

#[test]
fn types_and_construction() {
    let r = |src: &str| run(&format!("{ROUTE}{src}"));
    assert_eq!(
        r("Route(path: \"/\")"),
        "=> Route(path: \"/\", methods: [\"GET\"])"
    );
    assert_eq!(r("Route(\"/a\", [\"PUT\"]).methods"), "=> [\"PUT\"]");
    assert_eq!(
        r("let x = Route do\n  path = \"/b\"\nend\nx.path"),
        "=> \"/b\""
    );
    // Public fields are writable, with a type check.
    assert_eq!(
        r("let x = Route(path: \"/\")\nx.path = \"/z\"\nx.path"),
        "=> \"/z\""
    );
    assert!(
        err(&format!("{ROUTE}let x = Route(path: \"/\")\nx.path = 1"))
            .1
            .contains("expected type `str`")
    );
    // Value equality for a `type`.
    assert_eq!(r("Route(path: \"/\") == Route(path: \"/\")"), "=> true");
    assert_eq!(r("typeof(Route(path: \"/\"))"), "=> \"Route\"");
    assert!(
        err(&format!("{ROUTE}Route()"))
            .1
            .contains("field `path` is required")
    );
    assert!(
        err(&format!("{ROUTE}Route(path: \"/\", verb: 1)"))
            .1
            .contains("has no field `verb`")
    );
    assert!(
        err(&format!("{ROUTE}Route(path: 1)"))
            .1
            .contains("expected type `str`")
    );
}

#[test]
fn construction_blocks() {
    let server = "type Server\n  protocol = \"tcp\"\n  port: int = if protocol == \"tcp\" then 80 else 40 end\n  routes: [str] = []\nend\n";
    let r = |src: &str| run(&format!("{server}{src}"));
    // Default computed after the block, from the fields the block set.
    assert_eq!(r("Server do\n  protocol = \"udp\"\nend.port"), "=> 40");
    assert_eq!(r("Server().port"), "=> 80");
    // A field read in the block first gets its default value.
    assert_eq!(
        r("let s = Server do\n  routes << \"/a\"\n  routes << \"/b\"\nend\ns.routes"),
        "=> [\"/a\", \"/b\"]"
    );
    // Local variables, outer variables, loops.
    assert_eq!(
        r("let base = 8000\nServer do\n  let off = 80\n  port = base + off\nend.port"),
        "=> 8080"
    );
    assert!(
        err(&format!("{server}Server do\n  prot = 1\nend"))
            .1
            .contains("neither a field of `Server`")
    );
    assert!(
        err(&format!("{server}Server do\n  port = \"x\"\nend"))
            .1
            .contains("expected type `int`")
    );
    assert!(
        err("type A\n  x: int = y\n  y: int = x\nend\nA()")
            .1
            .contains("circular")
    );
}

#[test]
fn record_literals_convert_to_types() {
    let r = |src: &str| run(&format!("{ROUTE}{src}"));
    assert_eq!(
        r("def reg(r: Route) = typeof(r)\nreg({path: \"/\"})"),
        "=> \"Route\""
    );
    assert_eq!(r("let x: Route = {path: \"/\"}\nx.methods"), "=> [\"GET\"]");
    assert_eq!(
        r("let xs: [Route] = [{path: \"/a\"}, {path: \"/b\"}]\nxs[1].path"),
        "=> \"/b\""
    );
    assert_eq!(
        r("let x: Route? = {path: \"/\"}\ntypeof(x)"),
        "=> \"Route\""
    );
    assert_eq!(
        r("def mk(): Route = {path: \"/r\"}\nmk().path"),
        "=> \"/r\""
    );
    // Only literals are converted.
    assert!(
        err(&format!(
            "{ROUTE}def reg(r: Route) = r\nlet x = {{path: \"/\"}}\nreg(x)"
        ))
        .1
        .contains("expected type `Route` for `r`, got `{path: str}`")
    );
    assert!(
        err(&format!(
            "{ROUTE}def reg(r: Route) = r\nreg({{path: \"/\", verb: 1}})"
        ))
        .1
        .contains("has no field `verb`")
    );
}

#[test]
fn classes() {
    let user = "class User\n  count = 0\n  def init(@name: str, @email: str? = nil)\n  end\n  def name: str = @name\n  def name=(v: str)\n    @name = v.capitalize()\n  end\n  def greet(other) = \"#{@name} greets #{other.name}\"\n  def bump()\n    @count += 1\n    self\n  end\n  def to_s: str = \"User(#{@name})\"\nend\n";
    let r = |src: &str| run(&format!("{user}{src}"));
    assert_eq!(
        r("let u = User(\"ana\")\nu.name = \"bob\"\nu.name"),
        "=> \"Bob\""
    );
    assert_eq!(r("User(\"a\").greet(User(\"b\"))"), "=> \"a greets b\"");
    assert_eq!(r("print(User(\"zoé\"))"), "User(zoé)");
    assert_eq!(r("\"#{User(\"x\")}\""), "=> \"User(x)\"");
    assert_eq!(r("User(\"a\").bump().bump().to_s()"), "=> \"User(a)\"");
    // Identity, not value equality.
    assert_eq!(
        r("let a = User(\"a\")\n[a == a, a == User(\"a\")]"),
        "=> [true, false]"
    );
    assert!(
        err(&format!("{user}User(\"a\").email"))
            .1
            .contains("private field")
    );
    assert!(
        err(&format!("{user}let u = User(\"a\")\nu.email = \"x\""))
            .1
            .contains("private field")
    );
    assert!(
        err(&format!("{user}User(1)"))
            .1
            .contains("expected type `str`")
    );
    assert!(
        err("class C\n  x: int\nend\nC()")
            .1
            .contains("field `x` is required")
    );
    assert!(err("@x").1.contains("only be used inside a method"));
}

#[test]
fn enums() {
    let e = "enum Protocol: str\n  Tcp = \"TCP\"\n  Udp = \"UDP\"\n  def secure = self == Protocol::Tcp\nend\nenum Color\n  Red\n  Green\nend\n";
    let r = |src: &str| run(&format!("{e}{src}"));
    assert_eq!(r("Protocol::Tcp.value"), "=> \"TCP\"");
    assert_eq!(
        r("[Color::Green.value, Color::Green.name]"),
        "=> [1, \"Green\"]"
    );
    assert_eq!(r("Protocol::Udp"), "=> Protocol::Udp");
    assert_eq!(
        r("[Protocol::Tcp.secure, Protocol::Udp.secure]"),
        "=> [true, false]"
    );
    assert_eq!(r("Protocol.from(\"UDP\") == Protocol::Udp"), "=> true");
    assert_eq!(r("Protocol.from(\"x\")"), "");
    assert_eq!(r("Color.values().len()"), "=> 2");
    assert_eq!(
        r("case Protocol::Udp\n  when Protocol::Tcp then 1\n  when Protocol::Udp then 2\nend"),
        "=> 2"
    );
    assert_eq!(r("let p: Protocol = Protocol::Tcp\np.name"), "=> \"Tcp\"");
    assert!(
        err(&format!("{e}Protocol::Ftp"))
            .1
            .contains("has no variant `Ftp`")
    );
    assert!(
        err(&format!("{e}let p: Protocol = \"TCP\""))
            .1
            .contains("expected type `Protocol`")
    );
    assert!(
        err("enum E: int\n  A = \"a\"\nend")
            .1
            .contains("expected type `int` for variant `A`")
    );
}

#[test]
fn recursion_limit() {
    let (_, msg) = err("def f(n) = f(n + 1)\nf(0)");
    assert!(msg.contains("infinite recursion"), "{msg}");
    assert_eq!(
        run("def fib(n) = if n < 2 then n else fib(n - 1) + fib(n - 2) end\nfib(15)"),
        "=> 610"
    );
    // 900 nested calls succeed, even in a test thread (2 MiB stack).
    assert_eq!(
        run("def down(n) = if n == 0 then 0 else 1 + down(n - 1) end\ndown(900)"),
        "=> 900"
    );
}

#[test]
fn print_and_builtins() {
    assert_eq!(
        run("print(\"a\", 1, [2, \"b\"], nil)"),
        "a 1 [2, \"b\"] nil"
    );
    assert_eq!(run("typeof(1.5)"), "=> \"float\"");
    assert!(
        err("assert(1 == 2, \"math\")")
            .1
            .contains("assertion failed: math")
    );
}

#[test]
fn engine_api() {
    let mut e = Engine::new();
    e.register_fn("double", |x: i64| x * 2);
    e.register_fn("greet", |name: String, excl: bool| {
        format!("Hello {name}{}", if excl { "!" } else { "" })
    });
    e.register_fn("safe_div", |a: f64, b: f64| {
        if b == 0.0 {
            Err("division by zero")
        } else {
            Ok(a / b)
        }
    });
    e.register_fn("total", |xs: Vec<i64>| xs.iter().sum::<i64>());
    e.register_fn("maybe", |x: Option<i64>| x.unwrap_or(-1));

    assert_eq!(e.eval_as::<i64>("double(21)").unwrap(), 42);
    assert_eq!(
        e.eval_as::<String>("greet(\"Vincent\", true)").unwrap(),
        "Hello Vincent!"
    );
    assert_eq!(e.eval_as::<f64>("safe_div(1, 4)").unwrap(), 0.25);
    assert_eq!(e.eval_as::<i64>("total([1, 2, 3])").unwrap(), 6);
    assert_eq!(e.eval_as::<i64>("maybe(nil)").unwrap(), -1);

    let msg = e.eval("safe_div(1, 0)").unwrap_err()[0].message.clone();
    assert_eq!(msg, "division by zero");
    let msg = e.eval("double(\"a\")").unwrap_err()[0].message.clone();
    assert_eq!(msg, "argument 1 of `double`: expected int, got str");
    let msg = e.eval("double(1, 2)").unwrap_err()[0].message.clone();
    assert_eq!(msg, "`double` expects 1 argument, got 2");

    // State persists between calls; calling a script function from Rust.
    e.eval("let base = 100\ndef plus(x: int): int = x + base")
        .unwrap();
    assert_eq!(e.call_as::<i64>("plus", (5,)).unwrap(), 105);
    e.set_global("base", 1);
    assert_eq!(e.call_as::<i64>("plus", (5,)).unwrap(), 6);
    assert!(matches!(e.get_global("base"), Some(Value::Int(1))));

    // Result conversion refused.
    assert!(e.eval_as::<i64>("\"text\"").is_err());
}

#[test]
fn errors_from_earlier_scripts_show_their_own_source() {
    let mut e = Engine::new();
    e.eval_named("def boom(x) = x + \"!\"", "lib.bk").unwrap();
    let err = e.eval_named("\n\nboom(1)", "main.bk").unwrap_err();
    let shown = err.render("ignored", "ignored");
    assert!(shown.contains("lib.bk:1:15"), "{shown}");
    assert!(shown.contains("def boom(x) = x + \"!\""), "{shown}");
}

/// The runnable examples of docs/syntax.md run (those that don't use v0.2 features
/// yet and don't illustrate an error).
#[test]
fn spec_examples_run() {
    let examples = [
        "let name = \"Vincent\"\nlet age: int = 42\nage = 43\nMAX_PORT = 65535",
        "def add(a: int, b: int = 0): int\n  a + b\nend\ndef double(x: int): int = x * 2\nadd(1, 2)\nadd(a: 1, b: 2)\ndouble(4)",
        "let inc = |x| x + 1\nlet add = |a: int, b: int| a + b\n[1, 2, 3].map(|n| n * 2)\n[1, 2, 3].each do |n|\n  n\nend",
        "let nums = [1, 2, 3]\nnums[0]\nnums[-1]\nnums << 4\nnums.len()",
        "let ages = {\"alice\" => 30, \"bob\" => 25}\nlet key = \"carol\"\nlet scores: {str => int} = {}\nages[\"alice\"]\nages[\"dave\"] = 41\nages.has(\"bob\")\nages.delete(\"bob\")\nages.keys()\nscores[key] = 10",
        "let point = {x: 1, y: 2}\npoint.x\npoint.x = 5\nlet x = 10\nlet p2 = {x, y: 20}\ndef dist(p: {x: int, y: int}): float\n  (p.x * p.x + p.y * p.y).sqrt()\nend\ndist(point)",
        "let pair = (1, \"one\")\npair.0\ndef min_max(nums: [int]): (int, int)\n  (nums.min(), nums.max())\nend\nlet (lo, hi) = min_max([3, 1, 4])\nlet point = {x: 1, y: 2}\nlet {x, y} = point\nlet nums = [1, 2]\nlet [first, ...rest] = nums",
    ];
    for src in examples {
        let mut e = Engine::new();
        e.set_output(|_| {});
        if let Err(err) = e.eval(src) {
            panic!("{}", err.render(src, "syntax.md"));
        }
    }
}

#[test]
fn float_display() {
    let show = |src: &str| {
        let mut e = Engine::new();
        let v = e.eval(src).unwrap();
        (e.inspect(&v), e.display(&v))
    };
    for (src, want) in [
        ("3.0", "3.0"),
        ("-0.0", "-0.0"),
        ("0.1 + 0.2", "0.30000000000000004"),
        ("1e15", "1000000000000000.0"),
        ("1e16", "1e16"),
        ("1.5e16", "1.5e16"),
        ("1e300", "1e300"),
        ("-1e300", "-1e300"),
        ("5e-324", "5e-324"),
        ("0.0001", "0.0001"),
        ("0.00001", "1e-5"),
        ("1.7976931348623157e308", "1.7976931348623157e308"),
        ("1e308 * 10.0", "inf"),
        ("-1e308 * 10.0", "-inf"),
        ("0.0 / 0.0", "NaN"),
    ] {
        assert_eq!(show(src), (want.to_string(), want.to_string()), "{src}");
        // Interpolation: same text.
        assert_eq!(show(&format!("\"#{{{src}}}\"")).1, want, "{src}");
    }
    // Every finite rendering reads back as a literal and gives exactly the same number.
    let mut e = Engine::new();
    for src in [
        "1e300",
        "5e-324",
        "1.2345678901234568e17",
        "0.1 + 0.2",
        "1e-7",
        "2.0 / 3.0",
        "123456.789e10",
    ] {
        let v = e.eval(src).unwrap();
        let shown = e.inspect(&v);
        let again = e.eval(&shown).unwrap();
        let (bokoni::Value::Float(a), bokoni::Value::Float(b)) = (&v, &again) else {
            panic!("{src} → {shown}")
        };
        assert_eq!(a.to_bits(), b.to_bits(), "{src} → {shown}");
    }
}

#[test]
fn range_limits() {
    let run = |src: &str| {
        let mut e = Engine::new();
        e.set_output(|_| {});
        e.eval(src)
            .map(|v| e.inspect(&v))
            .unwrap_or_else(|err| err.to_string())
    };
    let max = i64::MAX;
    let min = i64::MIN;
    // The `i64::MAX` bound is included, without overflow.
    assert_eq!(
        run(&format!("({}..{max}).to_list()", max - 2)),
        format!("[{}, {}, {max}]", max - 2, max - 1)
    );
    assert_eq!(
        run(&format!(
            "let t = []\nfor i in {}..{max}\n  t << i\nend\nt",
            max - 1
        )),
        format!("[{}, {max}]", max - 1)
    );
    assert_eq!(
        run(&format!(
            "let r = {}..{max}\nlet t = []\nfor i in r\n  t << i\nend\nt",
            max - 1
        )),
        format!("[{}, {max}]", max - 1)
    );
    assert_eq!(run(&format!("(0..{max}).contains({max})")), "true");
    assert_eq!(run(&format!("(0...{max}).contains({max})")), "false");
    // Actual length 2^64 - 1: capped at i64::MAX.
    assert_eq!(run(&format!("({min}..{max}).len()")), max.to_string());
    assert_eq!(run(&format!("({min}...{max}).len()")), max.to_string());
    assert_eq!(run("(3...1).len()"), "0");
    assert_eq!(run(&format!("({min}...{min}).to_list()")), "[]");
    assert_eq!(run("(5..1).to_list()"), "[]");
    assert_eq!(
        run(&format!("({}..{max}).map(|i| i - {max})", max - 1)),
        "[-1, 0]"
    );
    // Huge allocations are errors, not a process abort.
    assert!(run(&format!("(0...{max}).to_list()")).contains("range too large"));
    assert!(run(&format!("\"ab\" * {max}")).contains("string too long"));
    assert!(run(&format!("\"ab\" * {}", max / 2 + 1)).contains("string too long"));
    assert_eq!(run(&format!("\"\" * {max}")), "\"\"");
    assert_eq!(run("\"ab\" * 3"), "\"ababab\"");
    // Iterating over a huge range without materializing it.
    assert_eq!(run(&format!("(0...{max}).find(|i| i > 3)")), "4");
    assert_eq!(
        run(&format!("({}..{max}).reduce(0, |a, i| a + 1)", max - 9)),
        "10"
    );
}

/// Runs `f` on a thread with a small stack: unbounded recursion would overflow it.
fn small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn deep_structures_do_not_overflow() {
    // Data nested 200,000 times: built, compared, displayed, dropped
    // (recursive dropping used to overflow the stack).
    let out = small_stack(|| {
        let mut out = Vec::new();
        for src in [
            "let l = []\nfor i in 0...200000\n  l = [l]\nend\nl.len()",
            "let a = []\nlet b = []\nfor i in 0...200000\n  a = [a]\n  b = [b]\nend\na == b",
            "let a = []\nlet b = [1]\nfor i in 0...200000\n  a = [a]\n  b = [b]\nend\na == b",
            "let h = {}\nfor i in 0...200000\n  h = {\"k\" => h}\nend\nh.len()",
            "let r = {v: 0}\nfor i in 0...200000\n  r = {v: r}\nend\nr == r",
            "let t = (0, 0)\nfor i in 0...200000\n  t = (t, i)\nend\nt.1",
            "type N\n  next_node = nil\nend\nlet n = N()\nfor i in 0...200000\n  n = N(next_node: n)\nend\nn == n",
            "let f = || 0\nfor i in 0...200000\n  let g = f\n  f = || g()\nend\n1",
            "let a = []\nfor i in 0...200000\n  a = [a, i]\nend\na.to_s().len() > 10",
        ] {
            let mut e = Engine::new();
            e.set_output(|_| {});
            // Variables change type on each iteration: dynamic checking.
            e.set_static_checks(false);
            out.push(
                e.eval(src)
                    .map(|v| e.inspect(&v))
                    .unwrap_or_else(|err| err.to_string()),
            );
        }
        out
    });
    assert_eq!(
        out,
        [
            "1", "true", "false", "1", "true", "199999", "true", "1", "true"
        ]
    );
}

#[test]
fn deep_sources_do_not_overflow() {
    let out = small_stack(|| {
        let mut out = Vec::new();
        for src in [
            format!("1{}", " + 1".repeat(200_000)),
            format!("{}1", "- ".repeat(100_000)),
            format!("{}true", "not ".repeat(100_001)),
            format!("let a = [1]\na{}", ".len().to_s()".repeat(20_000)),
            format!("{}1{}", "(".repeat(100_000), ")".repeat(100_000)),
        ] {
            let mut e = Engine::new();
            e.set_output(|_| {});
            out.push(match e.eval(&src) {
                Ok(v) => e.inspect(&v),
                Err(err) => {
                    // Bounded message, even for a 200,000-character line.
                    let shown = err.render(&src, "test");
                    assert!(shown.len() < 1000, "{} bytes", shown.len());
                    err[0].message.clone()
                }
            });
        }
        out
    });
    assert_eq!(out[0], "200001");
    assert_eq!(out[1], "1");
    assert_eq!(out[2], "false");
    assert_eq!(out[3], "\"1\"");
    assert_eq!(out[4], "nesting too deep");
}

#[test]
fn cyclic_and_shared_structures() {
    let run = |src: &str| {
        let mut e = Engine::new();
        e.set_output(|_| {});
        e.set_static_checks(false);
        e.eval(src)
            .map(|v| e.inspect(&v))
            .unwrap_or_else(|err| err.to_string())
    };
    // Cycles: displayed `[…]`, compared without endless recursion.
    assert_eq!(run("let l = []\nl << l\nl"), "[[…]]");
    assert_eq!(run("let l = []\nl << l\nl << l\nl"), "[[…], […]]");
    assert_eq!(run("let h = {}\nh[\"x\"] = h\nh"), "{\"x\" => {…}}");
    assert_eq!(
        run("let a = []\na << a\nlet b = []\nb << b\n[a == b, a == [a], a == [1], a == a]"),
        "[true, true, false, true]"
    );
    assert_eq!(
        run("let a = [1]\na << a\nlet b = [2]\nb << b\na == b"),
        "false"
    );
    assert_eq!(
        run("type N\n  v = nil\nend\nlet n = N()\nn.v = n\n[n, n == n, N(v: n) == N(v: n)]"),
        "[N(v: (…)), true, true]"
    );
    // Sharing: 2^40 paths, but few distinct nodes.
    assert_eq!(
        run(
            "let a = [1]\nlet b = [1]\nfor i in 0...40\n  a = [a, a]\n  b = [b, b]\nend\n[a == b, a.to_s().len() > 1000]"
        ),
        "[true, true]"
    );
    assert_eq!(
        run("let a = [1]\nlet b = [1]\nfor i in 0...40\n  a = [a, a]\n  b = [b, [b]]\nend\na == b"),
        "false"
    );
}
