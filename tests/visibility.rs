//! `private` and `protected`: methods of classes and traits, members of modules.
//!
//! Every program runs in each execution mode (interpreter, JIT, JIT from the first
//! call): the output and the result must be the same.

use bokoni::{Engine, Visibility};
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

/// Same error with static checks (before running) and without (when running).
fn both(src: &str, msg: &str) {
    let out = run(src);
    assert!(out.contains(msg), "expected `{msg}` (static) in:\n{out}");
    let out = run_mode(src, false, false, false);
    assert!(out.contains(msg), "expected `{msg}` (run time) in:\n{out}");
}

const ACCOUNT: &str = "class Account
  def init(@cents: int)
  end
  private def valid(n: int): bool = n > 0
  private def log = \"#{@cents}\"
  private def cents=(v: int)
    @cents = v
  end
  protected def cents = @cents
  def deposit(n: int): int
    raise Error(\"invalid amount\") unless self.valid(n)
    self.cents = @cents + n
    @cents
  end
  def richer(o: Account): bool = self.cents > o.cents
  def report(): str = self.log
  def total(all: [Account]): int = all.map(|a| a.cents).sum() + [1].map(|x| self.cents).sum()
end
class Savings < Account
  def bonus(): int = self.deposit(1) + (if self.valid(2) then 100 else 0 end)
  def cmp(o: Account): int = o.cents
end
";

#[test]
fn private_methods_on_self() {
    let src = format!(
        "{ACCOUNT}let a = Account(10)
[a.deposit(5), a.report(), Savings(1).bonus()]"
    );
    assert_eq!(run(&src), "=> [15, \"15\", 102]");
    // In a hot loop (compiled code).
    let src =
        format!("{ACCOUNT}let a = Account(0)\nfor i in 0..1000\n  a.deposit(1)\nend\na.report()");
    assert_eq!(run(&src), "=> \"1001\"");
}

#[test]
fn protected_methods_on_other_objects() {
    // Same class, a subclass, and inside a lambda of a method.
    let src = format!(
        "{ACCOUNT}let a = Account(10)
[a.richer(Account(3)), Account(1).richer(a), Savings(1).cmp(a), a.total([Account(2), Savings(3)])]"
    );
    assert_eq!(run(&src), "=> [true, false, 10, 15]");
}

#[test]
fn private_and_protected_errors() {
    let private = "`valid` is a private method of `Account`: it can only be called on `self`";
    let protected = "`cents` is a protected method of `Account`: it can only be called in the methods of `Account` and of its subclasses";
    both(&format!("{ACCOUNT}Account(1).valid(2)"), private);
    // A getter, a setter.
    both(
        &format!("{ACCOUNT}Account(1).log"),
        "`log` is a private method of `Account`",
    );
    both(
        &format!("{ACCOUNT}let a = Account(1)\na.cents = 5"),
        "`cents=` is a private method of `Account`",
    );
    both(&format!("{ACCOUNT}Account(1).cents"), protected);
    // Private: not even on another object of the same class.
    both(
        &format!(
            "{ACCOUNT}class Audit < Account\n  def check(o: Account) = o.valid(1)\nend\nAudit(1).check(Account(2))"
        ),
        private,
    );
    // Protected: not from an unrelated class.
    both(
        &format!(
            "{ACCOUNT}class Bank\n  def peek(a: Account) = a.cents\nend\nBank().peek(Account(1))"
        ),
        protected,
    );
    // Statically, also in a function never called.
    let out = run(&format!("{ACCOUNT}def f(a: Account) = a.valid(1)"));
    assert!(out.contains(private), "{out}");
}

#[test]
fn traits_with_private_and_protected_methods() {
    let src = "trait Named
  abstract def name(): str
  private def shout(): str = self.name().upper()
  protected def tag(): str = \"<#{self.name()}>\"
  def hello(): str = \"hello #{self.shout()}\"
  def pair(o: Named): str = self.tag() + o.tag()
end
class User
  include Named
  def init(@n: str)
  end
  def name(): str = @n
end
[User(\"ann\").hello(), User(\"a\").pair(User(\"b\"))]";
    assert_eq!(run(src), "=> [\"hello ANN\", \"<a><b>\"]");
    both(
        &format!("{src}\nUser(\"x\").shout()"),
        "`shout` is a private method of `Named`: it can only be called on `self`",
    );
    both(
        &format!("{src}\nUser(\"x\").tag()"),
        "`tag` is a protected method of `Named`: it can only be called in the methods of `Named` and of the classes that include it",
    );
}

#[test]
fn visibility_from_a_previous_script() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(ACCOUNT).unwrap();
    // The checker knows the visibility of the methods of an imported class.
    let err = e.eval("Account(1).valid(1)").unwrap_err().to_string();
    assert!(
        err.contains("`valid` is a private method of `Account`"),
        "{err}"
    );
    let err = e
        .eval("def f(a: Account) = a.cents")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("`cents` is a protected method of `Account`"),
        "{err}"
    );
    // Inherited by a class of the next script.
    e.eval("class Joint < Account\n  def sum(o: Account): int = self.cents + o.cents\nend")
        .unwrap();
    assert_eq!(e.eval_as::<i64>("Joint(1).sum(Account(2))").unwrap(), 3);
    // Introspection.
    let ty = e.type_info("Account").unwrap();
    assert_eq!(ty.method("valid").unwrap().visibility, Visibility::Private);
    assert_eq!(
        ty.method("cents").unwrap().visibility,
        Visibility::Protected
    );
    assert_eq!(ty.method("deposit").unwrap().visibility, Visibility::Public);
}

const HTTP: &str = "module Http
  private let TOKEN = \"t0k\"
  private def sign(s: str): str = \"#{s}+#{TOKEN}\"
  private class Conn
    def init(@host: str)
    end
    def host = @host
  end
  private module Wire
    def size(s: str): int = s.len()
  end
  def get(url: str): str = \"#{sign(Conn(url).host)} #{Wire.size(url)} #{Http::TOKEN}\"
  module Debug
    def signed(s: str): str = Http.sign(s)
  end
end
";

#[test]
fn private_module_members() {
    let src = format!("{HTTP}[Http.get(\"a.org\"), Http::Debug.signed(\"x\")]");
    assert_eq!(run(&src), "=> [\"a.org+t0k 5 t0k\", \"x+t0k\"]");
    let err = |src: &str, name: &str| {
        let msg = format!("`{name}` is private to module `Http`: it can only be used inside it");
        both(&format!("{HTTP}{src}"), &msg);
    };
    err("Http.sign(\"a\")", "sign");
    err("Http::TOKEN", "TOKEN");
    err("Http::Conn(\"a\")", "Conn");
    err("let c: Http::Conn? = nil", "Conn");
    err("class Pool < Http::Conn\nend", "Conn");
    err("Http::Wire.size(\"a\")", "Wire");
    // Also in a function that is never called.
    err("def f() = Http.sign(\"a\")", "sign");
}

#[test]
fn private_module_members_from_a_previous_script() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(HTTP).unwrap();
    let err = e.eval("Http.sign(\"a\")").unwrap_err().to_string();
    assert!(err.contains("`sign` is private to module `Http`"), "{err}");
    // A reopened module uses its private members.
    e.eval("module Http\n  def head(url: str): str = \"HEAD #{sign(url)}\"\nend")
        .unwrap();
    assert_eq!(
        e.eval_as::<String>("Http.head(\"b\")").unwrap(),
        "HEAD b+t0k"
    );
}

#[test]
fn misplaced_private_and_protected() {
    let check = |src: &str, msg: &str| {
        let out = run(src);
        assert!(out.contains(msg), "expected `{msg}` in:\n{out}");
    };
    check(
        "private def f() = 1",
        "`private` is only allowed in a module (before a declaration) or in a class (before a method)",
    );
    check(
        "def f()\n  private let x = 1\nend",
        "`private` is only allowed in a module",
    );
    check(
        "protected def f() = 1",
        "`protected` is only allowed before a method (`def`) of a `class` or a `trait`",
    );
    check(
        "module M\n  private 3\nend",
        "expected a declaration after `private` (`def`, `let`, `class`, `module`…), found `3`",
    );
    check(
        "class A\n  private x: int = 1\nend",
        "expected `def` after `private` (only methods can be private)",
    );
    check(
        "class A\n  private def init()\n  end\nend",
        "`init` cannot be private: `Name(…)` calls it",
    );
    check(
        "enum E\n  X\n  private def f() = 1\nend",
        "`private` methods are only allowed in a `class`, a `trait` or a `type`",
    );
}
