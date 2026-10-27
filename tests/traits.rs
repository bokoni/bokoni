//! Traits (`trait`, `include`), `abstract def` and abstract classes.
//!
//! Every program runs in each execution mode (interpreter, JIT, JIT from the first
//! call): the output and the result must be the same.

use bokoni::{DeclKind, Engine};
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

const GREET: &str = "trait Greet
  abstract def name(): str
  def greet(): str = \"Hello #{self.name()}\"
end
trait Shout
  abstract def greet(): str
  def shout(): str = self.greet().upper()
end
class User
  include Greet
  include Shout
  def init(@n: str)
  end
  def name(): str = @n
end
class Robot
  include Greet
  def name(): str = \"R2\"
  def greet(): str = \"beep\"
end
def welcome(x: Greet): str = x.greet()
";

#[test]
fn default_and_required_methods() {
    // `greet` comes from `Greet` (and satisfies `Shout`); `Robot` redefines it.
    let src = format!("{GREET}[welcome(User(\"ann\")), welcome(Robot()), User(\"bob\").shout()]");
    assert_eq!(run(&src), "=> [\"Hello ann\", \"beep\", \"HELLO BOB\"]");
    // In a hot loop, through a trait-typed parameter.
    let src = format!(
        "{GREET}let t = 0
for i in 0..300
  let g: Greet = if i % 3 == 0 then Robot() else User(\"x\") end
  t += welcome(g).len()
end
t"
    );
    assert_eq!(run(&src), "=> 1804");
}

#[test]
fn traits_are_types() {
    let src = format!("{GREET}let gs: [Greet] = [User(\"a\"), Robot()]\ngs.map(|g| g.greet())");
    assert_eq!(run(&src), "=> [\"Hello a\", \"beep\"]");
    let out = run(&format!("{GREET}class Plain\nend\nwelcome(Plain())"));
    assert!(
        out.contains("expected type `Greet` for `x`, got `Plain`"),
        "{out}"
    );
    // At run time too (static checks off).
    let src = format!("{GREET}class Plain\nend\ndef any(x) = x\nwelcome(any(Plain()))");
    let out = run_mode(&src, false, false, false);
    assert!(out.contains("expected type `Greet`, got Plain"), "{out}");
}

#[test]
fn traits_on_errors() {
    let src = "trait Retryable
  def attempts(): int = 3
end
class Timeout < Error
  include Retryable
end
class Fatal < Error
end
def run(e)
  begin
    raise e
  rescue r: Retryable
    \"retry #{r.attempts()}x\"
  rescue f: Error
    \"give up: #{f.message}\"
  end
end
[run(Timeout(\"slow\")), run(Fatal(\"disk\"))]";
    assert_eq!(run(src), "=> [\"retry 3x\", \"give up: disk\"]");
}

#[test]
fn abstract_classes() {
    let src = "abstract class Shape
  abstract def area(): float
  def describe(): str = \"#{typeof(self)} #{self.area()}\"
end
abstract class Polygon < Shape
  abstract def sides(): int
end
class Square < Polygon
  def init(@s: float)
  end
  def area(): float = @s * @s
  def sides(): int = 4
end
class Circle < Shape
  def init(@r: float)
  end
  def area(): float = 3.0 * @r * @r
end
let shapes: [Shape] = [Square(2.0), Circle(1.0)]
[shapes.map(|s| s.describe()), Square(1.0).sides()]";
    assert_eq!(run(src), "=> [[\"Square 4.0\", \"Circle 3.0\"], 4]");
}

#[test]
fn traits_from_a_previous_script() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(GREET).unwrap();
    // The required methods of an imported trait are known to the checker.
    e.eval("class Bot\n  include Greet\n  def name(): str = \"bot\"\nend")
        .unwrap();
    assert_eq!(e.eval_as::<String>("welcome(Bot())").unwrap(), "Hello bot");
    let err = e
        .eval("class Mute\n  include Greet\nend")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("`Mute` must define `name` (required by `Greet`)"),
        "{err}"
    );
    let err = e
        .eval("def f(g: Greet): int = g.name()")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("expected return type `int`, got `str`") || err.contains("`int`"),
        "{err}"
    );
    // Introspection.
    let ty = e.type_info("User").unwrap();
    assert_eq!(ty.traits, ["Greet", "Shout"]);
    assert!(!ty.is_abstract);
    let g = e.type_info("Greet").unwrap();
    assert_eq!(g.kind, DeclKind::Trait);
    assert!(g.is_abstract);
}

#[test]
fn trait_errors() {
    let t = "trait Greet\n  abstract def name(): str\nend\n";
    let check = |src: &str, msg: &str| {
        let out = run(&format!("{t}{src}"));
        assert!(out.contains(msg), "expected `{msg}` in:\n{out}");
    };
    check(
        "class U\n  include Greet\nend",
        "`U` must define `name` (required by `Greet`)",
    );
    check("Greet()", "`Greet` is a trait: it cannot be instantiated");
    check(
        "abstract class S\nend\nS()",
        "`S` is an abstract class: it cannot be instantiated",
    );
    check(
        "class A\nend\nclass U\n  include A\nend",
        "`A` is not a trait declared before `U`",
    );
    check(
        "class U\n  include Later\nend\ntrait Later\nend",
        "`Later` is not a trait declared before `U`",
    );
    check(
        "class U\n  abstract def f()\nend",
        "`abstract def` is only allowed in a `trait` or an `abstract class`",
    );
    check(
        "trait X\n  x: int = 1\nend",
        "a `trait` has no fields: it only describes methods",
    );
    check(
        "trait X\n  include Greet\nend",
        "`include` is only allowed in a `class`",
    );
    check("trait X\n  def f() = @x\nend", "`X` has no field `@x`");
    check(
        "trait X < Greet\nend",
        "only a `class` can inherit (a `trait` cannot)",
    );
    check(
        "abstract class S\n  abstract def f(): int\nend\nclass T < S\nend",
        "`T` must define `f` (required by `S`)",
    );
    // Without static checks: the same errors when running.
    let out = run_mode(
        &format!("{t}class U\n  include Greet\nend"),
        false,
        false,
        false,
    );
    assert!(
        out.contains("`U` must define `name` (required by `Greet`)"),
        "{out}"
    );
    let out = run_mode(
        &format!("{t}def make(t) = t()\nmake(Greet)"),
        false,
        false,
        false,
    );
    assert!(
        out.contains("`Greet` is a trait: it cannot be instantiated"),
        "{out}"
    );
}
