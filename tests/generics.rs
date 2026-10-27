//! Generics: `forall T` functions, generic types (`class Box(T)`, `type Pair(A, B)`),
//! generic traits (`Comparable(T)`) and bounds (`forall T: Comparable(T)`).
//!
//! Checked statically, erased at run time. Every program runs in each execution mode
//! (interpreter, JIT, JIT from the first call): the output and the result must be the
//! same.

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

/// Same outcome in every mode, with and without static checks.
fn run(src: &str) -> String {
    let a = run_mode(src, false, false, true);
    for (jit, eager) in [(true, false), (true, true)] {
        if cfg!(feature = "jit") || !jit {
            let b = run_mode(src, jit, eager, true);
            assert_eq!(a, b, "mode jit={jit} eager={eager} differs for:\n{src}");
        }
    }
    if a.starts_with("=>") {
        assert_eq!(
            a,
            run_mode(src, false, false, false),
            "differs without checks:\n{src}"
        );
    }
    a
}

fn err(src: &str, msg: &str) {
    let out = run(src);
    assert!(out.contains(msg), "expected `{msg}` in:\n{out}");
}

const BOX: &str = "class Box(T)
  def init(@value: T)
  end
  def get(): T = @value
  def set(v: T) = @value = v
  def map(f: (T): U): Box(U) forall U = Box(f(@value))
end
";

#[test]
fn generic_functions() {
    let src = "def first(xs: [T]): T? forall T = xs[0]?
def pair(a: A, b: B): (A, B) forall A, B = (a, b)
def map(xs: [T], f: (T): U): [U] forall T, U
  let out: [U] = []
  for x in xs
    out << f(x)
  end
  out
end
let n: int = first([1, 2]) or 0
let lens: [int] = map([\"a\", \"bcd\"], |s| s.len())
[n + 1, pair(1, \"a\"), lens, map([1, 2], |x| x * 1.5), first([]) or \"none\"]";
    assert_eq!(run(src), "=> [2, (1, \"a\"), [1, 3], [1.5, 3.0], \"none\"]");
    // Inferred types: wrong uses are reported.
    err(
        &format!("{src}\nlet s: str = first([1, 2])"),
        "expected type `str`, got `int?`",
    );
    err(
        &format!("{src}\nmap([1, 2], |x| x * 2).first().upper()"),
        "has no method `upper`",
    );
    err(
        "def same(a: T, b: T): T forall T = a\nsame(1, \"a\")",
        "expected type `int` for `b`, got `str`",
    );
    // In the body, `T` is any type: not an `int`.
    err(
        "def bad(x: T): T forall T = 1",
        "expected type `T` for the return value, got `int`",
    );
    // A hot loop calling a generic function.
    let src = "def id(x: T): T forall T = x\nlet t = 0\nfor i in 0..1000\n  t += id(i)\nend\nt";
    assert_eq!(run(src), "=> 500500");
}

#[test]
fn generic_classes_and_types() {
    let src = format!(
        "{BOX}let b: Box(int) = Box(1)
b.set(b.get() + 41)
let s = Box(\"abc\").map(|v| v.len())
type Pair(A, B)
  a: A
  b: B
end
let p = Pair(a: 1, b: \"z\")
let q: Pair(int, str) = {{a: 2, b: \"w\"}}
[b.get(), s.get() + 1, p.b.upper(), q.a + 1, typeof(b)]"
    );
    assert_eq!(run(&src), "=> [42, 4, \"Z\", 3, \"Box\"]");
    err(
        &format!("{BOX}let b: Box(int) = Box(1)\nb.set(\"x\")"),
        "expected type `int` for `v`, got `str`",
    );
    err(
        &format!("{BOX}let s: str = Box(1).get()"),
        "expected type `str`, got `int`",
    );
    err(
        &format!("{BOX}let b: Box(str) = Box(1)"),
        "expected type `Box(str)`, got `Box(int)`",
    );
    // From the expected type, when the values say nothing.
    let src = "class Stack(T)
  items: [T] = []
  def push(x: T) = @items << x
  def size(): int = @items.len()
end
let s: Stack(int) = Stack()
s.push(1)
s.size()";
    assert_eq!(run(src), "=> 1");
    err(
        &format!("{src}\ns.push(\"x\")"),
        "expected type `int` for `x`, got `str`",
    );
}

#[test]
fn generic_parents() {
    let src = format!(
        "{BOX}class IntBox < Box(int)
  def double(): int = self.get() * 2
end
class Labeled(T) < Box(T)
  def init(@value: T, @label: str)
  end
  def show(): str = \"#{{@label}}=#{{self.get()}}\"
  def get(): T = super()
end
let b: Box(int) = IntBox(4)
[IntBox(4).double(), Labeled(\"x\", \"k\").show(), Labeled(1, \"n\").get() + 1, b.get()]"
    );
    assert_eq!(run(&src), "=> [8, \"k=x\", 2, 4]");
    err(
        &format!("{BOX}class IntBox < Box(int)\nend\nlet s: str = IntBox(1).get()"),
        "expected type `str`, got `int`",
    );
    err(
        &format!("{BOX}class IntBox < Box(int)\nend\nlet c: Box(str) = IntBox(2)"),
        "expected type `Box(str)`, got `IntBox`",
    );
}

const COMPARABLE: &str = "trait Comparable(T)
  abstract def cmp(o: T): int
  def lt(o: T): bool = self.cmp(o) < 0
end
class Money
  include Comparable(Money)
  def init(@cents: int)
  end
  def cents = @cents
  def cmp(o: Money): int = @cents - o.cents
end
def max(a: T, b: T): T forall T: Comparable(T) = if a.cmp(b) >= 0 then a else b end
";

#[test]
fn generic_traits_and_bounds() {
    let src = format!(
        "{COMPARABLE}def sort(xs: [T]): [T] forall T: Comparable(T)
  let out: [T] = []
  for x in xs
    out << x
  end
  let n = out.len()
  for i in 0...n
    for j in 0...(n - 1 - i)
      if out[j + 1].lt(out[j])
        let t = out[j]
        out[j] = out[j + 1]
        out[j + 1] = t
      end
    end
  end
  out
end
[max(Money(3), Money(5)).cents, sort([Money(3), Money(1), Money(2)]).map(|m| m.cents)]"
    );
    assert_eq!(run(&src), "=> [5, [1, 2, 3]]");
    err(
        &format!("{COMPARABLE}max(1, 2)"),
        "`int` (for `T`) is not a `Comparable(int)`",
    );
    err(
        &format!("{COMPARABLE}class Sorted(T: Comparable(T))\nend\nlet s: Sorted(int)? = nil"),
        "`int` (for `T`) is not a `Comparable(int)`",
    );
    // The trait's methods take the class's arguments.
    err(
        &format!("{COMPARABLE}Money(1).lt(2)"),
        "expected type `Money` for `o`, got `int`",
    );
}

#[test]
fn generics_from_a_previous_script() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(BOX).unwrap();
    e.eval(COMPARABLE).unwrap();
    e.eval("class IntBox < Box(int)\nend\ndef first(xs: [T]): T? forall T = xs[0]?")
        .unwrap();
    // The checker knows the type parameters of what was declared before.
    assert_eq!(
        e.eval_as::<i64>("let b: Box(int) = Box(1)\nb.get() + 1")
            .unwrap(),
        2
    );
    for (src, msg) in [
        (
            "Box(1).set(\"x\")",
            "expected type `int` for `v`, got `str`",
        ),
        (
            "let s: str = IntBox(1).get()",
            "expected type `str`, got `int`",
        ),
        ("let s: str = first([1])", "expected type `str`, got `int?`"),
        ("max(1, 2)", "`int` (for `T`) is not a `Comparable(int)`"),
        ("Money(1).lt(2)", "expected type `Money` for `o`, got `int`"),
    ] {
        let err = e.eval(src).unwrap_err().to_string();
        assert!(err.contains(msg), "{src}: {err}");
    }
    assert_eq!(
        e.eval_as::<i64>("max(Money(2), Money(7)).cents").unwrap(),
        7
    );
    // Introspection.
    assert_eq!(e.type_info("Box").unwrap().params, ["T"]);
    assert!(e.type_info("Money").unwrap().params.is_empty());
}

#[test]
fn generic_errors() {
    err(
        &format!("{BOX}let b: Box(int, str) = Box(1)"),
        "`Box` takes 1 type argument, got 2",
    );
    err("let w: int(str) = 1", "`int` is not generic");
    err("class A\nend\nlet a: A(int) = A()", "`A` is not generic");
    err("let x: Nope(int) = 1", "unknown type `Nope`");
    err("annotation A(T)\nend", "an `annotation` cannot be generic");
    // Calling a parameter of function type (it used to crash the checker).
    err(
        "def apply(f: (int): int) = f(\"x\")",
        "expected type `int` for argument 1, got `str`",
    );
}

#[test]
#[cfg(feature = "jit")]
fn forall_functions_are_compiled_per_type() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    let src = "def max(a: T, b: T): T forall T = if a > b then a else b end
def sum(xs: [T], zero: T): T forall T
  let t = zero
  for x in xs
    t = t + x
  end
  t
end
def clamp(x: int, lo: int, hi: int): int = max(lo, if x > hi then hi else x end)
let ints = 0
let floats = 0.0
for i in 0..300
  ints += max(i, 150)
  floats += max(i * 0.5, 10.0)
end
[ints, floats, sum([1, 2, 3], 0), sum([0.5, 1.5], 0.0), clamp(7, 0, 5), clamp(-3, 0, 5)]";
    let v = e.eval(src).unwrap();
    assert_eq!(e.inspect(&v), "[56475, 22680.0, 6, 2.0, 5, 0]");
    // A copy per type: `max` called by the interpreter (`int`, `float`) and directly by
    // the native code of `clamp`.
    assert!(e.is_native("max"));
    assert!(e.is_native("sum"));
    assert!(e.is_native("clamp"));
    // Not for every value: strings stay interpreted, same result.
    assert_eq!(e.eval_as::<String>("max(\"a\", \"b\")").unwrap(), "b");
    // Arguments that do not fit the copy (`T = int` from the first one; without static
    // checks, which refuse it): interpreted.
    e.set_static_checks(false);
    assert_eq!(e.eval_as::<f64>("max(1, 2.5)").unwrap(), 2.5);
    e.eval("def mixed(x: int): float = max(x, 2.5)").unwrap();
    assert_eq!(e.eval_as::<f64>("mixed(3)").unwrap(), 3.0);
    assert!(!e.is_native("mixed"));
    e.set_static_checks(true);
    // Recursion: the typed copy calls itself.
    e.eval("def total(xs: [T], i: int): int forall T = if i >= xs.len() then 0 else 1 + total(xs, i + 1) end")
        .unwrap();
    assert_eq!(
        e.eval_as::<i64>("total([5, 6, 7], 0) + total([0.5], 0)")
            .unwrap(),
        4
    );
    assert!(e.is_native("total"));
}
