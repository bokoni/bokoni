use bokoni::{Engine, ErrorKind, Value};
use std::cell::RefCell;
use std::rc::Rc;

/// Exécute `src` ; renvoie la sortie de `print` suivie du résultat (`=> …`) s'il n'est pas nil.
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
        Ok(v) => panic!("devrait échouer, a donné {}", e.inspect(&v)),
        // Le type de la première erreur ; les messages de toutes.
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
    assert_eq!(run("nil or \"défaut\""), "=> \"défaut\"");
    assert_eq!(run("1 and 2"), "=> 2");
    assert!(err("1 / 0").1.contains("division par zéro"));
    assert!(err("9223372036854775807 + 1").1.contains("dépassement"));
    assert_eq!(err("1 < \"a\"").0, ErrorKind::Type);
}

#[test]
fn variables_and_constants() {
    assert_eq!(run("let x = 1\nx = x + 1\nx += 10\nx"), "=> 12");
    assert_eq!(run("MAX = 10\nMAX * 2"), "=> 20");
    assert!(err("x = 1").1.contains("utiliser `let x = …`"));
    assert!(
        err("MAX = 1\nMAX = 2")
            .1
            .contains("ne peut pas être modifié")
    );
    assert!(err("y").1.contains("`y` n'est pas défini"));
    // Portée de bloc : un `let` dans un `if` ne fuit pas.
    assert!(
        err("if true\n  let z = 1\nend\nz")
            .1
            .contains("`z` n'est pas défini")
    );
    // Mais on peut modifier une variable extérieure.
    assert_eq!(run("let n = 0\nif true\n  n = 5\nend\nn"), "=> 5");
}

#[test]
fn strings() {
    assert_eq!(
        run(r##"let n = "monde"
"Bonjour #{n.upper()} (#{1 + 1})""##),
        "=> \"Bonjour MONDE (2)\""
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
    // Appel avant définition et récursion mutuelle.
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
            .contains("attend 1 argument, reçu 2")
    );
    assert!(
        err("def f(a, b) = a\nf(1)")
            .1
            .contains("argument `b` manquant")
    );
    assert!(
        err("def f(a) = a\nf(z: 1)")
            .1
            .contains("pas de paramètre `z`")
    );
    assert!(
        err("def f = 1\nf = 2")
            .1
            .contains("ne peut pas être modifié")
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
    // Les lambdas sont tolérants : argument manquant = nil.
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
            "def k(p) = case p\n  when 80, 443 then \"web\"\n  when 8000..8999 then \"dev\"\n  else \"autre\"\nend\n[k(443), k(8080), k(1)]"
        ),
        "=> [\"web\", \"dev\", \"autre\"]"
    );
    assert!(err("break").1.contains("`break` en dehors"));
    assert!(err("for x in 5\nend").1.contains("ne peut pas parcourir"));
}

#[test]
fn collections() {
    assert_eq!(run("let l = [1, 2, 3]\nl[-1]"), "=> 3");
    assert_eq!(run("let l = [1, 2]\nl[0] = 9\nl << 3\nl"), "=> [9, 2, 3]");
    assert!(err("[1][5]").1.contains("hors limites"));
    assert_eq!(run("[3, 1, 2].sort()"), "=> [1, 2, 3]");
    assert_eq!(run("[\"b\", \"a\"].sort().join(\"-\")"), "=> \"a-b\"");
    assert_eq!(run("[2, 9, 4].max()"), "=> 9");
    assert_eq!(run("[1, 2].contains(2)"), "=> true");
    // Les listes sont partagées (comme Ruby, Python…).
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
        run("let h = {1 => \"un\"}\n[h.has(1), h.get(2, \"?\"), h.keys()]"),
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
    assert!(err("{[1] => 2}").1.contains("clé de hash"));
}

#[test]
fn records_and_tuples() {
    assert_eq!(run("let p = {x: 1, y: 2}\np.x = 5\np"), "=> {x: 5, y: 2}");
    assert!(
        err("let p = {x: 1}\np.z = 3")
            .1
            .contains("pas de champ `z`")
    );
    assert_eq!(run("let x = 10\n{x, y: 20}"), "=> {x: 10, y: 20}");
    assert_eq!(run("let p = {x: 3, y: 4}\nlet {x, y} = p\nx * y"), "=> 12");
    assert_eq!(run("let t = (1, \"un\")\nt.1"), "=> \"un\"");
    assert_eq!(run("let (a, b) = (1, 2)\na + b"), "=> 3");
    assert_eq!(run("let [h, ...rest] = [1, 2, 3]\nrest"), "=> [2, 3]");
    // Un champ fonction s'appelle comme une méthode.
    assert_eq!(run("let o = {double: |x| x * 2}\no.double(21)"), "=> 42");
    assert!(err("[1].len").1.contains("écrire `.len()`"));
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
    assert!(msg.contains("type attendu `int`, reçu `str`"), "{msg}");
    assert!(
        err("let m: \"dev\" | \"prod\" = \"test\"")
            .1
            .contains("type attendu")
    );
    assert!(
        err("def f(n: int) = n\nf(\"a\")")
            .1
            .contains("type attendu `int`")
    );
    assert!(
        err("def f(): str = 1\nf()")
            .1
            .contains("type attendu `str`")
    );
    assert!(err("let s: Server = 1").1.contains("type `Server` inconnu"));
    assert!(
        err("let s: Serveur = 1")
            .1
            .contains("type `Serveur` inconnu")
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
    // Champs publics modifiables, avec vérification du type.
    assert_eq!(
        r("let x = Route(path: \"/\")\nx.path = \"/z\"\nx.path"),
        "=> \"/z\""
    );
    assert!(
        err(&format!("{ROUTE}let x = Route(path: \"/\")\nx.path = 1"))
            .1
            .contains("type attendu `str`")
    );
    // Égalité par valeur pour un `type`.
    assert_eq!(r("Route(path: \"/\") == Route(path: \"/\")"), "=> true");
    assert_eq!(r("typeof(Route(path: \"/\"))"), "=> \"Route\"");
    assert!(
        err(&format!("{ROUTE}Route()"))
            .1
            .contains("champ `path` obligatoire")
    );
    assert!(
        err(&format!("{ROUTE}Route(path: \"/\", verb: 1)"))
            .1
            .contains("pas de champ `verb`")
    );
    assert!(
        err(&format!("{ROUTE}Route(path: 1)"))
            .1
            .contains("type attendu `str`")
    );
}

#[test]
fn construction_blocks() {
    let server = "type Server\n  protocol = \"tcp\"\n  port: int = if protocol == \"tcp\" then 80 else 40 end\n  routes: [str] = []\nend\n";
    let r = |src: &str| run(&format!("{server}{src}"));
    // Défaut calculé après le bloc, en fonction des champs fixés par le bloc.
    assert_eq!(r("Server do\n  protocol = \"udp\"\nend.port"), "=> 40");
    assert_eq!(r("Server().port"), "=> 80");
    // Un champ lu dans le bloc reçoit d'abord sa valeur par défaut.
    assert_eq!(
        r("let s = Server do\n  routes << \"/a\"\n  routes << \"/b\"\nend\ns.routes"),
        "=> [\"/a\", \"/b\"]"
    );
    // Variables locales, variables extérieures, boucles.
    assert_eq!(
        r("let base = 8000\nServer do\n  let off = 80\n  port = base + off\nend.port"),
        "=> 8080"
    );
    assert!(
        err(&format!("{server}Server do\n  prot = 1\nend"))
            .1
            .contains("ni un champ de `Server`")
    );
    assert!(
        err(&format!("{server}Server do\n  port = \"x\"\nend"))
            .1
            .contains("type attendu `int`")
    );
    assert!(
        err("type A\n  x: int = y\n  y: int = x\nend\nA()")
            .1
            .contains("circulaire")
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
    // Seuls les littéraux sont convertis.
    assert!(
        err(&format!(
            "{ROUTE}def reg(r: Route) = r\nlet x = {{path: \"/\"}}\nreg(x)"
        ))
        .1
        .contains("type attendu `Route` pour `r`, reçu `{path: str}`")
    );
    assert!(
        err(&format!(
            "{ROUTE}def reg(r: Route) = r\nreg({{path: \"/\", verb: 1}})"
        ))
        .1
        .contains("pas de champ `verb`")
    );
}

#[test]
fn classes() {
    let user = "class User\n  count = 0\n  def init(@name: str, @email: str? = nil)\n  end\n  def name: str = @name\n  def name=(v: str)\n    @name = v.capitalize()\n  end\n  def greet(other) = \"#{@name} salue #{other.name}\"\n  def bump()\n    @count += 1\n    self\n  end\n  def to_s: str = \"User(#{@name})\"\nend\n";
    let r = |src: &str| run(&format!("{user}{src}"));
    assert_eq!(
        r("let u = User(\"ana\")\nu.name = \"bob\"\nu.name"),
        "=> \"Bob\""
    );
    assert_eq!(r("User(\"a\").greet(User(\"b\"))"), "=> \"a salue b\"");
    assert_eq!(r("print(User(\"zoé\"))"), "User(zoé)");
    assert_eq!(r("\"#{User(\"x\")}\""), "=> \"User(x)\"");
    assert_eq!(r("User(\"a\").bump().bump().to_s()"), "=> \"User(a)\"");
    // Identité, pas égalité de valeur.
    assert_eq!(
        r("let a = User(\"a\")\n[a == a, a == User(\"a\")]"),
        "=> [true, false]"
    );
    assert!(
        err(&format!("{user}User(\"a\").email"))
            .1
            .contains("champ privé")
    );
    assert!(
        err(&format!("{user}let u = User(\"a\")\nu.email = \"x\""))
            .1
            .contains("champ privé")
    );
    assert!(
        err(&format!("{user}User(1)"))
            .1
            .contains("type attendu `str`")
    );
    assert!(
        err("class C\n  x: int\nend\nC()")
            .1
            .contains("champ `x` obligatoire")
    );
    assert!(err("@x").1.contains("que dans une méthode"));
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
            .contains("pas de variante `Ftp`")
    );
    assert!(
        err(&format!("{e}let p: Protocol = \"TCP\""))
            .1
            .contains("type attendu `Protocol`")
    );
    assert!(
        err("enum E: int\n  A = \"a\"\nend")
            .1
            .contains("type attendu `int` pour la variante `A`")
    );
}

#[test]
fn recursion_limit() {
    let (_, msg) = err("def f(n) = f(n + 1)\nf(0)");
    assert!(msg.contains("récursion infinie"), "{msg}");
    assert_eq!(
        run("def fib(n) = if n < 2 then n else fib(n - 1) + fib(n - 2) end\nfib(15)"),
        "=> 610"
    );
    // 900 appels imbriqués passent, même dans un thread de test (pile de 2 Mio).
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
        err("assert(1 == 2, \"maths\")")
            .1
            .contains("assertion échouée : maths")
    );
}

#[test]
fn engine_api() {
    let mut e = Engine::new();
    e.register_fn("double", |x: i64| x * 2);
    e.register_fn("greet", |name: String, excl: bool| {
        format!("Bonjour {name}{}", if excl { "!" } else { "" })
    });
    e.register_fn("safe_div", |a: f64, b: f64| {
        if b == 0.0 {
            Err("division par zéro")
        } else {
            Ok(a / b)
        }
    });
    e.register_fn("total", |xs: Vec<i64>| xs.iter().sum::<i64>());
    e.register_fn("maybe", |x: Option<i64>| x.unwrap_or(-1));

    assert_eq!(e.eval_as::<i64>("double(21)").unwrap(), 42);
    assert_eq!(
        e.eval_as::<String>("greet(\"Vincent\", true)").unwrap(),
        "Bonjour Vincent!"
    );
    assert_eq!(e.eval_as::<f64>("safe_div(1, 4)").unwrap(), 0.25);
    assert_eq!(e.eval_as::<i64>("total([1, 2, 3])").unwrap(), 6);
    assert_eq!(e.eval_as::<i64>("maybe(nil)").unwrap(), -1);

    let msg = e.eval("safe_div(1, 0)").unwrap_err()[0].message.clone();
    assert_eq!(msg, "division par zéro");
    let msg = e.eval("double(\"a\")").unwrap_err()[0].message.clone();
    assert_eq!(msg, "argument 1 de `double` : int attendu, reçu str");
    let msg = e.eval("double(1, 2)").unwrap_err()[0].message.clone();
    assert_eq!(msg, "`double` attend 1 argument, reçu 2");

    // L'état persiste entre les appels ; appel d'une fonction du script depuis Rust.
    e.eval("let base = 100\ndef plus(x: int): int = x + base")
        .unwrap();
    assert_eq!(e.call_as::<i64>("plus", (5,)).unwrap(), 105);
    e.set_global("base", 1);
    assert_eq!(e.call_as::<i64>("plus", (5,)).unwrap(), 6);
    assert!(matches!(e.get_global("base"), Some(Value::Int(1))));

    // Conversion de résultat refusée.
    assert!(e.eval_as::<i64>("\"texte\"").is_err());
}

#[test]
fn errors_from_earlier_scripts_show_their_own_source() {
    let mut e = Engine::new();
    e.eval_named("def boom(x) = x + \"!\"", "lib.bk").unwrap();
    let err = e.eval_named("\n\nboom(1)", "main.bk").unwrap_err();
    let shown = err.render("ignoré", "ignoré");
    assert!(shown.contains("lib.bk:1:15"), "{shown}");
    assert!(shown.contains("def boom(x) = x + \"!\""), "{shown}");
}

/// Les exemples exécutables de docs/syntax.md tournent (ceux qui n'utilisent pas encore
/// de fonctionnalité v0.2 et n'illustrent pas une erreur).
#[test]
fn spec_examples_run() {
    let examples = [
        "let name = \"Vincent\"\nlet age: int = 42\nage = 43\nMAX_PORT = 65535",
        "def add(a: int, b: int = 0): int\n  a + b\nend\ndef double(x: int): int = x * 2\nadd(1, 2)\nadd(a: 1, b: 2)\ndouble(4)",
        "let inc = |x| x + 1\nlet add = |a: int, b: int| a + b\n[1, 2, 3].map(|n| n * 2)\n[1, 2, 3].each do |n|\n  n\nend",
        "let nums = [1, 2, 3]\nnums[0]\nnums[-1]\nnums << 4\nnums.len()",
        "let ages = {\"alice\" => 30, \"bob\" => 25}\nlet key = \"carol\"\nlet scores: {str => int} = {}\nages[\"alice\"]\nages[\"dave\"] = 41\nages.has(\"bob\")\nages.delete(\"bob\")\nages.keys()\nscores[key] = 10",
        "let point = {x: 1, y: 2}\npoint.x\npoint.x = 5\nlet x = 10\nlet p2 = {x, y: 20}\ndef dist(p: {x: int, y: int}): float\n  (p.x * p.x + p.y * p.y).sqrt()\nend\ndist(point)",
        "let pair = (1, \"un\")\npair.0\ndef min_max(nums: [int]): (int, int)\n  (nums.min(), nums.max())\nend\nlet (lo, hi) = min_max([3, 1, 4])\nlet point = {x: 1, y: 2}\nlet {x, y} = point\nlet nums = [1, 2]\nlet [first, ...rest] = nums",
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
        // Interpolation : même écriture.
        assert_eq!(show(&format!("\"#{{{src}}}\"")).1, want, "{src}");
    }
    // Chaque écriture finie se relit comme littéral et redonne exactement le nombre.
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
    // La borne `i64::MAX` est incluse, sans débordement.
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
    // Longueur réelle 2^64 - 1 : plafonnée à i64::MAX.
    assert_eq!(run(&format!("({min}..{max}).len()")), max.to_string());
    assert_eq!(run(&format!("({min}...{max}).len()")), max.to_string());
    assert_eq!(run("(3...1).len()"), "0");
    assert_eq!(run(&format!("({min}...{min}).to_list()")), "[]");
    assert_eq!(run("(5..1).to_list()"), "[]");
    assert_eq!(
        run(&format!("({}..{max}).map(|i| i - {max})", max - 1)),
        "[-1, 0]"
    );
    // Les allocations démesurées sont des erreurs, pas un arrêt du processus.
    assert!(run(&format!("(0...{max}).to_list()")).contains("intervalle trop grand"));
    assert!(run(&format!("\"ab\" * {max}")).contains("chaîne trop longue"));
    assert!(run(&format!("\"ab\" * {}", max / 2 + 1)).contains("chaîne trop longue"));
    assert_eq!(run(&format!("\"\" * {max}")), "\"\"");
    assert_eq!(run("\"ab\" * 3"), "\"ababab\"");
    // Parcourir un intervalle immense sans tout matérialiser.
    assert_eq!(run(&format!("(0...{max}).find(|i| i > 3)")), "4");
    assert_eq!(
        run(&format!("({}..{max}).reduce(0, |a, i| a + 1)", max - 9)),
        "10"
    );
}

/// Exécute `f` dans un fil à petite pile : une récursion non maîtrisée y déborderait.
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
    // Données imbriquées 200 000 fois : construites, comparées, affichées, libérées
    // (la libération récursive faisait déborder la pile).
    let out = small_stack(|| {
        let mut out = Vec::new();
        for src in [
            "let l = []\nfor i in 0...200000\n  l = [l]\nend\nl.len()",
            "let a = []\nlet b = []\nfor i in 0...200000\n  a = [a]\n  b = [b]\nend\na == b",
            "let a = []\nlet b = [1]\nfor i in 0...200000\n  a = [a]\n  b = [b]\nend\na == b",
            "let h = {}\nfor i in 0...200000\n  h = {\"k\" => h}\nend\nh.len()",
            "let r = {v: 0}\nfor i in 0...200000\n  r = {v: r}\nend\nr == r",
            "let t = (0, 0)\nfor i in 0...200000\n  t = (t, i)\nend\nt.1",
            "type N\n  suiv = nil\nend\nlet n = N()\nfor i in 0...200000\n  n = N(suiv: n)\nend\nn == n",
            "let f = || 0\nfor i in 0...200000\n  let g = f\n  f = || g()\nend\n1",
            "let a = []\nfor i in 0...200000\n  a = [a, i]\nend\na.to_s().len() > 10",
        ] {
            let mut e = Engine::new();
            e.set_output(|_| {});
            // Variables qui changent de type à chaque tour : vérification dynamique.
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
                    // Message borné, même pour une ligne de 200 000 caractères.
                    let shown = err.render(&src, "test");
                    assert!(shown.len() < 1000, "{} octets", shown.len());
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
    assert_eq!(out[4], "imbrication trop profonde");
}
