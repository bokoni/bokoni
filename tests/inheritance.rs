//! Single inheritance of classes: `class B < A`, `super`, subtypes.
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

const ANIMALS: &str = "class Animal
  def init(@name: str)
  end
  def speak(): str = \"...\"
  def describe(): str = \"#{@name} says #{self.speak()}\"
  def name = @name
end
class Dog < Animal
  def speak(): str = \"woof\"
end
class Puppy < Dog
  def init(@name: str, @age: int)
  end
  def speak(): str = super() + \" (#{@age})\"
end
";

#[test]
fn methods_are_inherited_and_redefined() {
    let src = format!(
        "{ANIMALS}let all: [Animal] = [Animal(\"a\"), Dog(\"rex\"), Puppy(\"bit\", 1)]
all.map(|x| x.describe())"
    );
    assert_eq!(
        run(&src),
        "=> [\"a says ...\", \"rex says woof\", \"bit says woof (1)\"]"
    );
    // The parent's `init` is inherited; its methods read the inherited fields.
    let src = format!("{ANIMALS}[Dog(\"rex\").name, typeof(Puppy(\"p\", 2))]");
    assert_eq!(run(&src), "=> [\"rex\", \"Puppy\"]");
}

#[test]
fn super_calls_the_parent_version() {
    // `super(…)` passes its own arguments (named ones included); `@x` set by the parent.
    let src = "class A
  def init(@x: int)
  end
  def f(k: int, scale: int = 1): int = (@x + k) * scale
end
class B < A
  def init(@x: int, @y: int)
    super(x * 10)
  end
  def f(k: int, scale: int = 1): int = super(k, scale: 2) + @y
end
[B(1, 5).f(1), A(1).f(1)]";
    assert_eq!(run(src), "=> [27, 2]");
    // `super` inside a `do` block of a method.
    let src = "class A
  def f(v: int): int = v * 100
end
class B < A
  def f(v: int): int = [v].map do |x|
    super(x) + 1
  end.first() or 0
end
B().f(3)";
    assert_eq!(run(src), "=> 301");
}

#[test]
fn subclasses_are_subtypes() {
    let src = format!(
        "{ANIMALS}def loud(a: Animal): str = a.speak().upper()
let t = 0
for i in 0..300
  let a: Animal = if i % 2 == 0 then Dog(\"d\") else Animal(\"a\") end
  t += loud(a).len()
end
[loud(Puppy(\"p\", 3)), t]"
    );
    assert_eq!(run(&src), "=> [\"WOOF (3)\", 1054]");
    // The reverse is a type error, statically and at run time.
    let out = run(&format!("{ANIMALS}let d: Dog = Animal(\"a\")"));
    assert!(out.contains("expected type `Dog`, got `Animal`"), "{out}");
    let src =
        format!("{ANIMALS}def as_dog(d: Dog) = d\ndef any(x) = x\nas_dog(any(Animal(\"a\")))");
    let out = run_mode(&src, false, false, false);
    assert!(out.contains("expected type `Dog`, got Animal"), "{out}");
}

#[test]
fn custom_errors_are_errors() {
    let src = "class IoError < Error
end
class NotFound < IoError
  def init(@path: str)
    super(\"not found: #{path}\")
  end
  def path = @path
end
def open(p: str)
  raise NotFound(p) if p == \"missing\"
  raise IoError(\"disk\") if p == \"bad\"
  1 / 0 if p == \"zero\"
  p
end
def try(p)
  begin
    open(p)
  rescue e: NotFound
    \"not found #{e.path}\"
  rescue e: IoError
    \"io: #{e.message}\"
  rescue e: Error
    \"other: #{e.message}\"
  end
end
[try(\"missing\"), try(\"bad\"), try(\"zero\"), try(\"ok\")]";
    assert_eq!(
        run(src),
        "=> [\"not found missing\", \"io: disk\", \"other: division by zero\", \"ok\"]"
    );
    let src = "class Oops < Error\nend\nraise Oops(\"boom\")";
    assert!(run(src).starts_with("runtime error: Oops: boom"));
    // `Error` itself, by name or by position.
    assert_eq!(
        run("[Error(\"a\").message, Error(message: \"b\").message]"),
        "=> [\"a\", \"b\"]"
    );
}

#[test]
fn fields_of_parent_and_child() {
    let src = "class A
  x: int = 1
  def sum(): int = @x
end
class B < A
  y: int = 2
  def sum(): int = super() + @y
end
[B().sum(), B(x: 10, y: 20).sum(), A(x: 5).sum()]";
    assert_eq!(run(src), "=> [3, 30, 5]");
}

#[test]
fn parent_from_a_previous_script() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(ANIMALS).unwrap();
    e.eval("class Cat < Animal\n  def speak(): str = super() + \" meow\"\nend")
        .unwrap();
    assert_eq!(
        e.eval_as::<String>("Cat(\"tom\").describe()").unwrap(),
        "tom says ... meow"
    );
    // The checker knows the inherited fields and methods.
    let err = e.eval("Cat(1)").unwrap_err().to_string();
    assert!(err.contains("expected type `str`"), "{err}");
    let err = e.eval("let c: Dog = Cat(\"x\")").unwrap_err().to_string();
    assert!(err.contains("expected type `Dog`, got `Cat`"), "{err}");
    // Introspection.
    let ty = e.type_info("Cat").unwrap();
    assert_eq!(ty.parent.as_deref(), Some("Animal"));
    assert_eq!(ty.fields()[0].name, "name");
    assert!(ty.method("describe").is_some());
}

#[test]
fn inheritance_errors() {
    let err = |src: &str| run(src);
    assert!(
        err("type A\nend\nclass B < A\nend").contains("`A` is not a class declared before `B`")
    );
    assert!(
        err("class B < A\nend\nclass A\nend").contains("`A` is not a class declared before `B`")
    );
    assert!(err("class B < Nope\nend").contains("`Nope` is not a class declared before `B`"));
    assert!(err("type T < Error\nend").contains("only a `class` can inherit (a `type` cannot)"));
    assert!(
        err("class A\n  x: int = 1\nend\nclass B < A\n  x: int = 2\nend")
            .contains("field `x` is already declared by `A`")
    );
    assert!(
        err("class A\nend\nclass B < A\n  def f() = super()\nend\nB().f()")
            .contains("`super`: `A` has no method `f`")
    );
    assert!(
        err("def f() = super(1)")
            .contains("`super` can only be used in a method of a class that inherits")
    );
    assert!(
        err("class A\n  def f(x: int) = x\nend\nclass B < A\n  def f(x: int) = super(\"s\")\nend")
            .contains("expected type `int` for `x`, got `str`")
    );
    assert!(
        err("class A\nend\nclass B < A\n  def f() = super\nend")
            .contains("expected `(` after `super`")
    );
    // Without static checks: the same errors when compiling or running.
    let out = run_mode("type A\nend\nclass B < A\nend", false, false, false);
    assert!(
        out.contains("`A` is not a class declared before `B`"),
        "{out}"
    );
    let out = run_mode(
        "class A\nend\nclass B < A\n  def f() = super()\nend\nB().f()",
        false,
        false,
        false,
    );
    assert!(out.contains("`super`: `A` has no method `f`"), "{out}");
}

#[test]
fn a_name_declared_twice_is_an_error() {
    // Found by the fuzzer: the checker used to panic.
    let out = run("class A\n  def f() = 1\nend\nclass A\n  def g() = 2\nend");
    assert!(out.contains("`A` is declared twice in this block"), "{out}");
    let out = run("enum E\n  X\n  def f() = 1\nend\nenum E\n  Y\n  def g() = 2\nend");
    assert!(out.contains("`E` is declared twice in this block"), "{out}");
}
