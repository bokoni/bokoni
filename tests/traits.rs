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
        "type X\n  include Greet\nend",
        "`include` is only allowed in a `class` or a `trait`",
    );
    check(
        "trait X\n  include X\nend",
        "`X` is not a trait declared before `X`",
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

const NAMED: &str = "trait Named
  abstract def name(): str
  def tag(): str = \"<#{self.name()}>\"
end
trait Greet
  include Named
  def greet(): str = \"Hello #{self.name()} #{self.tag()}\"
end
";

#[test]
fn traits_that_include_traits() {
    // A class including `Greet` is also a `Named`: its methods, its type.
    let src = format!(
        "{NAMED}class User
  include Greet
  def init(@n: str)
  end
  def name(): str = @n
end
def show(x: Named): str = x.tag()
def up(x: Greet): Named = x
let gs: [Named] = [User(\"a\"), up(User(\"b\"))]
[User(\"ann\").greet(), show(User(\"bo\")), gs.map(|g| g.tag()), User(\"c\").is_a?(Named), User(\"c\").is_a?(Greet)]"
    );
    assert_eq!(
        run(&src),
        "=> [\"Hello ann <ann>\", \"<bo>\", [\"<a>\", \"<b>\"], true, true]"
    );
    // A method required by the included trait: provided by the trait including it, or
    // by the class; otherwise an error naming the trait that requires it.
    let src = format!(
        "{NAMED}trait Fixed\n  include Named\n  def name(): str = \"f\"\nend\nclass F\n  include Fixed\nend\nF().tag()"
    );
    assert_eq!(run(&src), "=> \"<f>\"");
    let err = run(&format!("{NAMED}class U\n  include Greet\nend"));
    assert!(
        err.contains("`U` must define `name` (required by `Named`)"),
        "{err}"
    );
    let err = run_mode(
        &format!("{NAMED}class U\n  include Greet\nend"),
        false,
        false,
        false,
    );
    assert!(
        err.contains("`U` must define `name` (required by `Named`)"),
        "{err}"
    );
    // Not a `Greet` for including only `Named`.
    let err = run(&format!(
        "{NAMED}class U\n  include Named\n  def name(): str = \"u\"\nend\ndef f(x: Greet) = x\nf(U())"
    ));
    assert!(
        err.contains("expected type `Greet` for `x`, got `U`"),
        "{err}"
    );
    // Without static checks: checked when calling.
    let out = run_mode(
        &format!(
            "{NAMED}class U\n  include Named\n  def name(): str = \"u\"\nend\ndef f(x: Greet) = x\nf(U())"
        ),
        false,
        false,
        false,
    );
    assert!(out.contains("Greet"), "{out}");
    // Generic: the arguments passed on (`Ord(T)` includes `Eq(T)`).
    let src = "trait Eq(T)
  abstract def eq(o: T): bool
  def ne(o: T): bool = not self.eq(o)
end
trait Ord(T)
  include Eq(T)
  abstract def cmp(o: T): int
  def eq(o: T): bool = self.cmp(o) == 0
end
class M
  include Ord(M)
  def init(@c: int)
  end
  def c = @c
  def cmp(o: M): int = @c - o.c
end
def same(a: T, b: T): bool forall T: Eq(T) = a.eq(b)
[M(1).ne(M(2)), same(M(3), M(3)), M(1).is_a?(Eq)]";
    assert_eq!(run(src), "=> [true, true, true]");
    let err = run(&format!("{src}\ndef f(x: Eq(int)) = x\nf(M(1))"));
    assert!(
        err.contains("expected type `Eq(int)` for `x`, got `M`"),
        "{err}"
    );
}

#[test]
fn included_traits_from_a_previous_script() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(NAMED).unwrap();
    e.eval("trait Loud\n  include Greet\n  def loud(): str = self.greet().upper()\nend")
        .unwrap();
    e.eval("class Bot\n  include Loud\n  def name(): str = \"bot\"\nend\ndef show(x: Named): str = x.tag()")
        .unwrap();
    assert_eq!(
        e.eval_as::<String>("Bot().loud() + show(Bot())").unwrap(),
        "HELLO BOT <BOT><bot>"
    );
    let err = e
        .eval("class Mute\n  include Loud\nend")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("`Mute` must define `name` (required by `Named`)"),
        "{err}"
    );
    let ty = e.type_info("Bot").unwrap();
    assert_eq!(ty.traits, ["Named", "Greet", "Loud"]);
    assert_eq!(e.type_info("Loud").unwrap().traits, ["Named", "Greet"]);
}
