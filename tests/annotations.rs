//! Annotations : déclaration, application, vérification, lecture depuis Rust.

use bokoni::{DeclKind, Engine, Value};

const SPEC: &str = r##"
annotation Validate
  min: int? = nil
  max: int? = nil
end

annotation Column
  name: str? = nil
  primary: bool = false
end

annotation Tag
end

@[Tag]
@[Doc("Un serveur HTTP")]
type Server
  @[Validate(min: 1, max: 65535)]
  port: int = 80

  @[Column(name: "host_name")]
  @[Deprecated("utiliser address")]
  host: str = "localhost"

  @[Validate(0, 10)]
  @[Validate(max: 5)]
  retries: int = 3

  @[Doc("Adresse complète")]
  def address(): str = "#{self.host}:#{self.port}"
end

@[Deprecated("utiliser add")]
def plus(a: int, b: int): int = a + b

def connect(@[Validate(min: 0)] timeout: int, host = "x")
  timeout
end

enum Protocol: str
  @[Deprecated]
  Ftp = "FTP"
  Tcp = "TCP"
end
"##;

fn engine() -> Engine {
    let mut e = Engine::new();
    e.set_output(|_| {});
    if let Err(err) = e.eval(SPEC) {
        panic!("{}", err.render(SPEC, "spec"));
    }
    e
}

#[test]
fn types_fields_and_methods() {
    let mut e = engine();
    let ty = e.type_info("Server").unwrap();
    assert_eq!(ty.name, "Server");
    assert_eq!(ty.kind, DeclKind::Type);
    assert!(ty.annotation("Tag").is_some());
    assert_eq!(ty.doc(), Some("Un serveur HTTP"));
    assert_eq!(ty.all_annotations().len(), 2);

    let names: Vec<&str> = ty.fields().iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["port", "host", "retries"]);

    // Exemple de la spécification.
    for field in ty.fields() {
        if let Some(v) = field.annotation("Validate") {
            let min: Option<i64> = v.get("min").unwrap();
            assert!(min.is_some() || field.name == "retries");
        }
    }
    let port = ty.field("port").unwrap();
    assert_eq!(port.ty.as_deref(), Some("int"));
    assert!(port.has_default);
    let v = port.annotation("Validate").unwrap();
    assert_eq!(v.get::<Option<i64>>("min").unwrap(), Some(1));
    assert_eq!(v.get::<i64>("max").unwrap(), 65535);
    assert!(
        v.get::<i64>("nope")
            .unwrap_err()
            .contains("pas de champ `nope`")
    );
    assert!(v.get::<String>("min").is_err());

    let host = ty.field("host").unwrap();
    let col = host.annotation("Column").unwrap();
    assert_eq!(col.get::<String>("name").unwrap(), "host_name");
    assert!(!col.get::<bool>("primary").unwrap());
    assert_eq!(host.deprecated(), Some(Some("utiliser address")));
    assert_eq!(port.deprecated(), None);

    // Positionnels dans l'ordre des champs ; plusieurs fois la même : la dernière.
    let retries = ty.field("retries").unwrap();
    let all: Vec<_> = retries.annotations("Validate").collect();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].get::<Option<i64>>("min").unwrap(), Some(0));
    assert_eq!(all[0].get::<Option<i64>>("max").unwrap(), Some(10));
    let last = retries.annotation("Validate").unwrap();
    assert_eq!(last.get::<Option<i64>>("max").unwrap(), Some(5));
    assert_eq!(last.get::<Option<i64>>("min").unwrap(), None);

    let m = ty.method("address").unwrap();
    assert_eq!(m.ret.as_deref(), Some("str"));
    assert_eq!(m.doc(), Some("Adresse complète"));
}

#[test]
fn functions_params_and_enums() {
    let mut e = engine();
    let f = e.function_info("plus").unwrap();
    assert_eq!(f.deprecated(), Some(Some("utiliser add")));
    let ps: Vec<(&str, Option<&str>)> = f
        .params
        .iter()
        .map(|p| (p.name.as_str(), p.ty.as_deref()))
        .collect();
    assert_eq!(ps, [("a", Some("int")), ("b", Some("int"))]);
    assert_eq!(f.ret.as_deref(), Some("int"));

    let c = e.function_info("connect").unwrap();
    let t = c.param("timeout").unwrap();
    assert_eq!(
        t.annotation("Validate")
            .unwrap()
            .get::<Option<i64>>("min")
            .unwrap(),
        Some(0)
    );
    assert!(c.param("host").unwrap().has_default);
    assert!(c.all_annotations().is_empty());

    let en = e.enum_info("Protocol").unwrap();
    let ftp = en.variant("Ftp").unwrap();
    assert_eq!(ftp.deprecated(), Some(None));
    assert!(matches!(&ftp.value, Value::Str(s) if &**s == "FTP"));
    assert_eq!(en.variant("Tcp").unwrap().deprecated(), None);

    let a = e.type_info("Validate").unwrap();
    assert_eq!(a.kind, DeclKind::Annotation);

    // Introuvable.
    assert!(e.type_info("Nope").is_err());
    assert!(e.function_info("Server").is_err());
    assert!(e.enum_info("plus").is_err());
}

#[test]
fn annotations_do_not_change_execution() {
    let mut e = engine();
    assert_eq!(e.eval_as::<i64>("plus(2, 3)").unwrap(), 5);
    assert_eq!(
        e.eval_as::<String>("Server(port: 8080).address()").unwrap(),
        "localhost:8080"
    );
    assert_eq!(e.eval_as::<i64>("connect(7)").unwrap(), 7);
}

#[test]
fn arguments_are_checked() {
    let err = |src: &str| {
        let mut e = Engine::new();
        let full = format!("annotation Validate\n  min: int? = nil\nend\n{src}");
        e.eval(&full).unwrap_err().to_string()
    };
    assert!(err("@[Nope]\ndef f() = 1").contains("annotation `Nope` inconnue"));
    assert!(err("@[Validate(min: \"a\")]\ndef f() = 1").contains("int"));
    assert!(err("@[Validate(mini: 1)]\ndef f() = 1").contains("mini"));
    assert!(err("@[Doc()]\ndef f() = 1").contains("text"));
    assert!(err("@[Doc(1)]\ndef f() = 1").contains("str"));
    assert!(err("type T\nend\n@[T]\ndef f() = 1").contains("n'est pas une annotation"));
    assert!(err("@[plus]\ndef f() = 1").contains("plus"));
}

#[test]
fn constants_and_later_scripts() {
    let mut e = Engine::new();
    e.eval("annotation Limit\n  n: int\nend\nlet MAX = 40\n")
        .unwrap();
    // Annotation déclarée dans un script précédent ; argument calculé avec une globale.
    e.eval("@[Limit(MAX + 2)]\ndef f() = 1").unwrap();
    let f = e.function_info("f").unwrap();
    assert_eq!(f.annotation("Limit").unwrap().get::<i64>("n").unwrap(), 42);
    // Les arguments sont évalués à la lecture.
    e.eval("MAX = 0").unwrap();
    let f = e.function_info("f").unwrap();
    assert_eq!(f.annotation("Limit").unwrap().get::<i64>("n").unwrap(), 2);
    // Erreur pendant l'évaluation : signalée à la lecture.
    e.eval("@[Limit(10 / MAX)]\ndef g() = 1").unwrap();
    let err = e.function_info("g").unwrap_err().to_string();
    assert!(err.contains("division par zéro"), "{err}");
}

#[test]
fn deprecated_uses_are_warned() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    // Déclarations dans un premier script…
    e.eval(SPEC).unwrap();
    // Une déclaration n'est pas une utilisation ; `address` lit bien `host`.
    let w: Vec<String> = e
        .take_warnings()
        .iter()
        .map(|w| w.message.clone())
        .collect();
    assert_eq!(w, ["`Server.host` est obsolète : utiliser address"]);
    // …utilisations dans le suivant (déclarations importées) : avertissements, et le
    // script s'exécute.
    let v = e
        .eval("let s = Server(host: \"a\")\n[plus(1, 2), s.host, Protocol::Ftp.value, Protocol::Tcp.value, s.port]")
        .unwrap();
    assert_eq!(e.inspect(&v), "[3, \"a\", \"FTP\", \"TCP\", 80]");
    let w: Vec<String> = e
        .take_warnings()
        .iter()
        .map(|w| w.message.clone())
        .collect();
    assert_eq!(
        w,
        [
            "`Server.host` est obsolète : utiliser address",
            "`plus` est obsolète : utiliser add",
            "`Server.host` est obsolète : utiliser address",
            "`Protocol::Ftp` est obsolète",
        ]
    );
    assert!(e.take_warnings().is_empty());

    // Dans le même script, avec un gestionnaire : appelé avant l'exécution.
    let mut e = Engine::new();
    let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let (l1, l2) = (log.clone(), log.clone());
    e.set_output(move |s| l1.borrow_mut().push(format!("print {s}")));
    e.set_warning_handler(move |w| l2.borrow_mut().push(w.to_string()));
    e.eval("@[Deprecated(\"non\")]\ndef old() = 1\nclass C\n  @[Deprecated]\n  def m() = 2\nend\nprint(old() + C().m())").unwrap();
    let log = log.borrow();
    assert_eq!(log.len(), 3, "{log:?}");
    assert!(log[0].contains("`old` est obsolète : non"), "{log:?}");
    assert!(log[1].contains("`C.m` est obsolète"), "{log:?}");
    assert!(log[2].starts_with("print 3"), "{log:?}");
}
