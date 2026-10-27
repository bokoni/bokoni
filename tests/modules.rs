//! Modules: `module M … end`, `M::Name`, `M.f()`.
//!
//! Every program runs in each execution mode (interpreter, JIT, JIT from the first
//! call): the output and the result must be the same.

use bokoni::Engine;
use std::cell::RefCell;
use std::rc::Rc;

fn run_mode(src: &str, jit: bool, eager: bool, checks: bool) -> String {
    let mut e = Engine::new();
    let out = Rc::new(RefCell::new(String::new()));
    let o = out.clone();
    e.set_output(move |s| o.borrow_mut().push_str(s));
    e.set_jit(jit);
    e.set_jit_eager(eager);
    e.set_static_checks(checks);
    let r = match e.eval(src) {
        Ok(v) => format!("=> {}", e.inspect(&v)),
        Err(err) => err.render(src, "test"),
    };
    let out = out.borrow();
    format!("{out}{r}")
}

/// Same outcome in every mode.
fn run(src: &str) -> String {
    let a = run_mode(src, false, false, true);
    for (jit, eager) in [(true, false), (true, true)] {
        if cfg!(feature = "jit") || !jit {
            let b = run_mode(src, jit, eager, true);
            assert_eq!(a, b, "mode jit={jit} eager={eager} differs for:\n{src}");
        }
    }
    a
}

const HTTP: &str = "module Http
  let VERSION = \"1.1\"
  enum Status
    Ok
    NotFound
  end
  class Request
    def init(@url: str)
    end
    def url = @url
  end
  class Failure < Error
  end
  trait Named
    abstract def name(): str
  end
  def get(url: str): Request = Request(url)
  def status(code: int): Status = if code == 200 then Status::Ok else Status::NotFound end
  module Mime
    let JSON = \"application/json\"
    def of(ext: str): str = if ext == \"json\" then JSON else \"text/plain\" end
  end
  def describe(): str = \"http #{VERSION} #{Mime::JSON}\"
end
";

#[test]
fn members_by_path_and_functions_by_dot() {
    let src = format!(
        "{HTTP}let r: Http::Request = Http.get(\"x.org\")
[r.url, Http::VERSION, Http.status(200), Http::Status::NotFound, Http::Mime::JSON,
 Http::Mime.of(\"json\"), Http.describe(), typeof(r)]"
    );
    assert_eq!(
        run(&src),
        "=> [\"x.org\", \"1.1\", Http::Status::Ok, Http::Status::NotFound, \"application/json\", \"application/json\", \"http 1.1 application/json\", \"Http::Request\"]"
    );
    // A module function in a hot loop (compiled code calls it directly).
    let src = "module M\n  def sq(x: int): int = x * x\nend\nlet t = 0\nfor i in 0..1000\n  t += M.sq(i)\nend\nt";
    assert_eq!(run(src), "=> 333833500");
}

#[test]
fn classes_traits_and_errors_across_modules() {
    let src = format!(
        "{HTTP}class Api < Http::Failure
end
class Person
  include Http::Named
  def name(): str = \"ann\"
end
def greet(n: Http::Named): str = \"hi #{{n.name()}}\"
[begin
  raise Api(\"boom\")
rescue e: Http::Failure
  \"#{{typeof(e)}}: #{{e.message}}\"
end, greet(Person())]"
    );
    assert_eq!(run(&src), "=> [\"Api: boom\", \"hi ann\"]");
}

#[test]
fn names_inside_a_module() {
    // Inside: short names, members first; the enclosing script's functions are visible
    // from the module's functions.
    let src = "def helper() = 40
let name = \"outer\"
module U
  let name = \"inner\"
  def f() = helper() + 2
  def g() = name
end
[U.f(), U.g(), name]";
    assert_eq!(run(src), "=> [42, \"inner\", \"outer\"]");
}

#[test]
fn modules_from_a_previous_script() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(HTTP).unwrap();
    assert_eq!(e.eval_as::<String>("Http.get(\"a\").url").unwrap(), "a");
    assert_eq!(
        e.eval_as::<String>("Http::Mime::JSON").unwrap(),
        "application/json"
    );
    // The checker knows the members of the module.
    let err = e.eval("Http.get(1)").unwrap_err().to_string();
    assert!(
        err.contains("expected type `str` for `url`, got `int`"),
        "{err}"
    );
    let err = e.eval("Http::Nope").unwrap_err().to_string();
    assert!(err.contains("module `Http` has no `Nope`"), "{err}");
    // A module can be reopened: new members, and short names for the old ones.
    e.eval("module Http\n  def head(url: str): str = \"HEAD #{get(url).url}\"\nend")
        .unwrap();
    assert_eq!(e.eval_as::<String>("Http.head(\"b\")").unwrap(), "HEAD b");
    // Introspection by qualified name.
    assert!(e.type_info("Http::Request").is_ok());
}

#[test]
fn module_errors() {
    let check = |src: &str, msg: &str| {
        let out = run(&format!("{HTTP}{src}"));
        assert!(out.contains(msg), "expected `{msg}` in:\n{out}");
    };
    check(
        "Http",
        "`Http` is a module: use `Http::Name` or `Http.f(…)`",
    );
    check("let m = Http\nm", "`Http` is a module");
    check("Http.nope(1)", "module `Http` has no `nope`");
    check("Http::Nope", "module `Http` has no `Nope`");
    check("Http.get(1)", "expected type `str` for `url`, got `int`");
    check(
        "def f()\n  module X\n  end\nend",
        "a `module` must be at the top level of the script (or in a module)",
    );
    // A module runs before the script's statements (like the declarations).
    check(
        "let top = 1\nmodule U\n  let y = top\nend",
        "`top` is used before its declaration",
    );
    check(
        "class Top\nend\nmodule U\n  class B < Top\n  end\nend",
        "`Top` is not a class declared before `B`",
    );
    // Without static checks: a runtime error.
    let out = run_mode(&format!("{HTTP}Http::nope"), false, false, false);
    assert!(out.contains("`Http::nope` is not defined"), "{out}");
}
