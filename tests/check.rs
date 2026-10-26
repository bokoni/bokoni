//! Vérification statique : erreurs signalées avant l'exécution, et absence de faux positifs.

use bokoni::Engine;

/// Messages des erreurs statiques (le programme n'est pas exécuté s'il y en a).
fn errors(src: &str) -> Vec<String> {
    let mut e = Engine::new();
    e.set_output(|_| {});
    match e.eval(src) {
        Ok(_) => Vec::new(),
        Err(errs) => errs.iter().map(|e| e.message.clone()).collect(),
    }
}

/// Le programme passe la vérification ET s'exécute sans erreur.
fn ok(src: &str) {
    let mut e = Engine::new();
    e.set_output(|_| {});
    if let Err(errs) = e.eval(src) {
        panic!("faux positif ?\n{}", errs.render(src, "test"));
    }
}

fn has(errs: &[String], needle: &str) -> bool {
    errs.iter().any(|e| e.contains(needle))
}

#[test]
fn reports_all_errors_before_running() {
    let errs = errors(
        r#"print("ne doit pas s'afficher")
let x = 1
x = "a"
y = 2
let n: int = "trois"
def add(a: int, b: int): int = a + b
add(1)
add(1, "2")
"abc".lenght()
let l = [1, 2]
l.push("a")
print(inconnu)
"#,
    );
    assert_eq!(errs.len(), 8, "{errs:#?}");
    assert!(has(&errs, "type attendu `int`, reçu `str`")); // x = "a"
    assert!(has(&errs, "`y` n'est pas déclaré"));
    assert!(has(&errs, "argument `b` manquant"));
    assert!(has(&errs, "type attendu `int` pour `b`, reçu `str`"));
    assert!(has(
        &errs,
        "`str` n'a pas de méthode `lenght` (voulez-vous dire `len` ?)"
    ));
    assert!(has(&errs, "type attendu `int`, reçu `str`"));
    assert!(has(&errs, "`inconnu` n'est pas défini"));
}

#[test]
fn names() {
    assert!(has(
        &errors("let total = 1\nprint(totla)"),
        "(voulez-vous dire `total` ?)"
    ));
    assert!(has(
        &errors("print(x)\nlet x = 1"),
        "`x` est utilisé avant sa déclaration"
    ));
    assert!(has(&errors("MAX = 1\nMAX = 2"), "ne peut pas être modifié"));
    assert!(has(&errors("def f = 1\nf = 2"), "ne peut pas être modifié"));
    assert!(has(&errors("break"), "`break` en dehors d'une boucle"));
    assert!(has(&errors("@x"), "que dans une méthode"));
    assert!(has(&errors("let s: Serveur = 1"), "type `Serveur` inconnu"));
    assert!(has(
        &errors("let s: strr = 1"),
        "(voulez-vous dire `str` ?)"
    ));
    // Une fonction peut utiliser une variable déclarée plus loin (appelée après).
    ok("def show() = config + 1\nlet config = 1\nshow()");
    // Portée de bloc.
    assert!(has(
        &errors("if true\n  let z = 1\nend\nprint(z)"),
        "`z` n'est pas défini"
    ));
}

#[test]
fn inference() {
    // Type de la variable : celui de sa première valeur.
    assert!(has(
        &errors("let x = 1\nx = 2.5"),
        "type attendu `int`, reçu `float`"
    ));
    ok("let x = 1.5\nx = 2"); // int → float accepté
    ok("let x = nil\nx = 1\nx = \"a\""); // nil seul : type inconnu
    ok("let xs = []\nxs << 1\nxs << \"a\"");
    assert!(has(
        &errors("let xs = [1, 2]\nxs << \"a\""),
        "`<<` ajoute `str` à une liste de `int`"
    ));
    // Retour déduit du corps.
    assert!(has(
        &errors("def five = 5\nfive() + \"a\""),
        "`+` entre `int` et `str`"
    ));
    // Paramètres des lambdas déduits des éléments.
    assert!(has(
        &errors("[1, 2].map(|n| n.upper())"),
        "`int` n'a pas de méthode `upper`"
    ));
    assert!(has(
        &errors("[\"a\"].map(|s| s.len()).map(|n| n.upper())"),
        "`int` n'a pas de méthode `upper`"
    ));
    assert!(has(
        &errors("{\"a\" => 1}.each do |k, v|\n  v.upper()\nend"),
        "`int` n'a pas de méthode `upper`"
    ));
    assert!(has(
        &errors("for i in 0..3\n  i.upper()\nend"),
        "`int` n'a pas de méthode `upper`"
    ));
}

#[test]
fn gradual_typing_has_no_false_positives() {
    // Paramètres non annotés : `any`, tout est permis.
    ok("def f(x) = x.whatever_method()\nlet g = |a| a.b.c");
    // Une valeur qui *peut* convenir n'est pas refusée (vérifiée à l'exécution).
    ok("let m: \"dev\" | \"prod\" = \"dev\"\nlet s = \"prod\"\nm = s");
    ok("let h = {\"a\" => 1}\nlet v = h[\"a\"]\nlet n: int = v");
    ok("let xs = [1, \"a\"]\nlet first: int = xs[0]");
    ok("let x: int | str = 1\nx = \"a\"");
    // Fonctions Rust : type inconnu.
    let mut e = Engine::new();
    e.register_fn("ext", |x: i64| x);
    e.eval("let r = ext(1)\nr.anything").unwrap_err(); // erreur… à l'exécution seulement
    assert!(
        e.eval("let r: str = ext(1)").unwrap_err()[0]
            .message
            .contains("type attendu `str`")
    );
    // `case` sur une constante.
    ok("let k = case 8080\n  when 80 then 1\n  else 2\nend");
}

#[test]
fn types_and_classes() {
    let route = "type Route\n  path: str\n  methods: [str] = [\"GET\"]\nend\n";
    let e = |src: &str| errors(&format!("{route}{src}"));
    assert!(has(&e("Route()"), "champ `path` obligatoire"));
    assert!(has(
        &e("Route(path: \"/\", verb: 1)"),
        "`Route` n'a pas de champ `verb`"
    ));
    assert!(has(
        &e("Route(path: 1)"),
        "type attendu `str` pour le champ `path`"
    ));
    assert!(has(
        &e("Route do\n  pth = \"/\"\nend"),
        "(voulez-vous dire `path` ?)"
    ));
    assert!(has(
        &e("Route(path: \"/\").pth"),
        "(voulez-vous dire `path` ?)"
    ));
    assert!(has(
        &e("def reg(r: Route) = r\nreg({path: \"/\", verb: 1})"),
        "`Route` n'a pas de champ `verb`"
    ));
    assert!(has(
        &e("def reg(r: Route) = r\nreg({methods: []})"),
        "champ `path` obligatoire"
    ));
    assert!(has(
        &e("let rs: [Route] = [{path: 1}]"),
        "type attendu `str` pour le champ `path`"
    ));
    // Un `bloc` peut remplir les champs obligatoires : pas d'erreur statique.
    ok(&format!(
        "{route}let r = Route do\n  path = \"/\"\nend\nr.methods.len()"
    ));
    ok(&format!(
        "{route}def reg(r: Route) = r.path\nreg({{path: \"/\"}})"
    ));

    let user = "class User\n  def init(@name: str)\n  end\n  def name: str = @name\n  def rename(n: str) = @name = n\nend\n";
    let e = |src: &str| errors(&format!("{user}{src}"));
    assert!(has(&e("User(1)"), "type attendu `str` pour `name`"));
    assert!(has(
        &e("User(\"a\").rename(1)"),
        "type attendu `str` pour `n`"
    ));
    assert!(has(
        &e("User(\"a\").name.len() + \"x\""),
        "`+` entre `int` et `str`"
    ));
    assert!(has(
        &errors("class C\n  def init(@n: int)\n  end\n  def get = @m\nend"),
        "`C` n'a pas de champ `@m`"
    ));
    assert!(has(
        &errors("class C\n  def init(@n: int)\n  end\nend\nC(1).n"),
        "champ privé"
    ));
    ok(&format!(
        "{user}let u = User(\"a\")\nu.rename(\"b\")\nu.name.upper()"
    ));
}

#[test]
fn enums() {
    let en = "enum Protocol: str\n  Tcp = \"TCP\"\n  Udp = \"UDP\"\nend\n";
    let e = |src: &str| errors(&format!("{en}{src}"));
    assert!(has(&e("Protocol::Tpc"), "(voulez-vous dire `Tcp` ?)"));
    assert!(has(
        &e("Protocol::Tcp.value + 1"),
        "`+` entre `str` et `int`"
    ));
    assert!(has(
        &e("let p: Protocol = \"TCP\""),
        "type attendu `Protocol`, reçu `str`"
    ));
    assert!(has(
        &errors("enum E: int\n  A = \"a\"\nend"),
        "type attendu `int` pour la variante `A`"
    ));
    ok(&format!(
        "{en}let p = Protocol.from(\"TCP\")\nProtocol.values().len()"
    ));
}

#[test]
fn annotations() {
    assert!(has(
        &errors("@[Inconnue]\ndef f = 1"),
        "annotation `Inconnue` inconnue"
    ));
    assert!(has(
        &errors("annotation Validate\n  min: int? = nil\nend\n@[Validate(min: \"a\")]\ndef f = 1"),
        "type attendu `int?` pour le champ `min`"
    ));
    ok("annotation Tag\nend\n@[Tag]\n@[Deprecated(\"utiliser g\")]\ndef f = 1");
}

#[test]
fn checks_use_earlier_definitions() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("type Server\n  port: int = 80\nend\ndef double(x: int): int = x * 2")
        .unwrap();
    // Le type et la fonction définis avant sont connus du vérificateur.
    let errs = e.eval("Server do\n  prot = 1\nend").unwrap_err();
    assert!(
        errs[0].message.contains("(voulez-vous dire `port` ?)"),
        "{errs}"
    );
    assert!(
        e.eval("double(\"a\")").unwrap_err()[0]
            .message
            .contains("type attendu `int`")
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
    // Sans vérification statique, l'erreur arrive à l'exécution (après le print).
    let out = std::rc::Rc::new(std::cell::RefCell::new(String::new()));
    let sink = out.clone();
    e.set_output(move |s| sink.borrow_mut().push_str(s));
    let errs = e.eval("print(\"avant\")\nlet x: int = \"a\"").unwrap_err();
    assert_eq!(errs.len(), 1);
    assert_eq!(out.borrow().as_str(), "avant\n");
}

/// Programmes corrects et variés : aucune erreur (statique ni à l'exécution).
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
  print("#{name.capitalize()} a #{age} ans")
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
  else "autre"
end
def counter()
  let n = 0
  || n += 1
end
let c = counter()
c()
"##);
}
