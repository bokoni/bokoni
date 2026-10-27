//! Error handling: `raise`, `begin` / `rescue` / `else` / `ensure`.
//!
//! Every program runs in each execution mode (interpreter, JIT, JIT from the first
//! call): the output and the result must be the same.

use bokoni::Engine;
use std::cell::RefCell;
use std::rc::Rc;

/// `print` output and result (or rendered error) of a program in one mode.
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

#[test]
fn rescue_gives_the_value_of_the_clause() {
    let src = "def div(a, b) = a / b
let r = begin
  div(1, 0)
rescue e
  print(e.message)
  -1
end
[r, begin
  div(6, 3)
rescue
  0
end]";
    assert_eq!(run(src), "division by zero\n=> [-1, 2]");
}

#[test]
fn clauses_are_tried_in_order_by_type() {
    let src = "type ConfigError
  message: str
  path: str
end
def load(path: str): str
  raise ConfigError(message: \"missing #{path}\", path: path) if path == \"x\"
  raise \"bad #{path}\" if path == \"y\"
  \"ok #{path}\"
end
def try(path)
  begin
    load(path)
  rescue e: Error
    \"error: #{e.message}\"
  rescue e: ConfigError
    \"config: #{e.path}\"
  end
end
[try(\"x\"), try(\"y\"), try(\"z\")]";
    assert_eq!(run(src), "=> [\"config: x\", \"error: bad y\", \"ok z\"]");
    // A union of types in one clause, a type without a variable.
    let src = "type A
  n: int
end
type B
  n: int
end
def f(k)
  begin
    raise A(n: k) if k == 1
    raise B(n: k) if k == 2
    raise \"other\"
  rescue e: A | B
    e.n
  rescue Error
    0
  end
end
[f(1), f(2), f(3)]";
    assert_eq!(run(src), "=> [1, 2, 0]");
}

#[test]
fn else_and_ensure_run_in_order() {
    let src = "def f(fail)
  begin
    print(\"body\")
    raise \"x\" if fail
    \"body value\"
  rescue
    print(\"rescue\")
    \"rescue value\"
  else
    print(\"else\")
    \"else value\"
  ensure
    print(\"ensure\")
    \"ignored\"
  end
end
[f(false), f(true)]";
    assert_eq!(
        run(src),
        "body\nelse\nensure\nbody\nrescue\nensure\n=> [\"else value\", \"rescue value\"]"
    );
    // An error in `else` is not rescued by the same `begin`, but `ensure` runs.
    let src = "begin
  begin
    1
  rescue
    print(\"no\")
  else
    raise \"from else\"
  ensure
    print(\"ensure\")
  end
rescue e
  e.message
end";
    assert_eq!(run(src), "ensure\n=> \"from else\"");
}

#[test]
fn unmatched_errors_keep_their_origin() {
    let src = "type Other
  message: str
end
def f(x) = 10 / x
begin
  f(0)
rescue e: Other
  print(\"no\")
ensure
  print(\"ensure\")
end";
    let out = run(src);
    assert!(
        out.starts_with("ensure\nruntime error: division by zero"),
        "{out}"
    );
    assert!(out.contains("<eval>:4:12"), "{out}");
    assert!(out.contains("= in `f` called from <eval>:6:3"), "{out}");
}

#[test]
fn uncaught_raise_messages() {
    assert!(run("raise \"fatal\"").starts_with("runtime error: fatal\n --> <eval>:1:1"));
    let src = "type ConfigError\n  message: str\nend\nraise ConfigError(message: \"missing\")";
    assert!(run(src).starts_with("runtime error: ConfigError: missing"));
    let src = "type Oops\n  code: int\nend\nraise Oops(code: 3)";
    assert!(run(src).starts_with("runtime error: raised Oops(code: 3)"));
    // `Error` itself: its message only.
    assert!(run("raise Error(message: \"plain\")").starts_with("runtime error: plain"));
}

#[test]
fn raise_needs_a_string_or_an_object() {
    let out = run("raise 42");
    assert!(
        out.starts_with("type error: `raise` expects a string or an object, got `int`"),
        "{out}"
    );
    // Without static checks: the same error at run time, which can be rescued.
    let src = "def v() = 42\nbegin\n  raise v()\nrescue e\n  e.message\nend";
    let out = run_mode(src, false, false, false);
    assert_eq!(out, "=> \"`raise` expects a string or an object, got int\"");
}

#[test]
fn ensure_runs_on_next_break_and_return() {
    let src = "let log = []
for i in 0..5
  begin
    next if i == 1
    break if i == 3
    log << i
  ensure
    log << \"e#{i}\"
  end
end
log";
    assert_eq!(run(src), "=> [0, \"e0\", \"e1\", 2, \"e2\", \"e3\"]");
    // `return` through two `ensure`s, innermost first; the value is computed before.
    let src = "def f()
  let x = 1
  begin
    begin
      return x
    ensure
      x = 2
      print(\"inner\")
    end
  ensure
    print(\"outer #{x}\")
  end
  99
end
f()";
    assert_eq!(run(src), "inner\nouter 2\n=> 1");
    // A loop inside the `begin`: its `break` does not leave the `begin`.
    let src = "let n = 0
begin
  while true
    n += 1
    break if n == 3
  end
  n += 10
ensure
  n += 100
end
n";
    assert_eq!(run(src), "=> 113");
    // `return` from a `rescue` clause.
    let src = "def f()
  begin
    raise \"x\"
  rescue
    return \"from rescue\"
  ensure
    print(\"ensure\")
  end
  \"after\"
end
f()";
    assert_eq!(run(src), "ensure\n=> \"from rescue\"");
}

#[test]
fn an_error_in_ensure_replaces_the_first() {
    let src = "begin
  begin
    raise \"first\"
  ensure
    raise \"second\"
  end
rescue e
  e.message
end";
    assert_eq!(run(src), "=> \"second\"");
}

#[test]
fn def_and_do_blocks_are_implicit_begins() {
    let src = "def safe(n)
  10 / n
rescue
  0
ensure
  print(\"done #{n}\")
end
let xs = [1, 2, 0].map do |x|
  100 / x
rescue
  nil
end
[safe(2), safe(0), xs]";
    assert_eq!(run(src), "done 2\ndone 0\n=> [5, 0, [100, 50, nil]]");
}

#[test]
fn errors_cross_calls_and_compiled_code() {
    // Raised deep in a recursion, in a function hot enough to be compiled.
    let src = "def h(n)
  raise \"bottom\" if n == 0
  h(n - 1) + 1
end
def g(n: int): int
  raise \"negative\" if n < 0
  n * 2
end
let s = 0
for i in 0..2000
  s += g(i)
  s += begin
    h(i % 7)
  rescue e
    i
  end
end
[s, begin
  g(-1)
rescue e
  e.message
end]";
    assert_eq!(run(src), "=> [6003000, \"negative\"]");
    // A Rust-side method (`map`) calling a block that raises.
    let src = "begin\n  [1, 0].map do |x|\n    1 / x\n  end\nrescue e\n  e.message\nend";
    assert_eq!(run(src), "=> \"division by zero\"");
    // Infinite recursion is rescued; the engine is usable afterwards.
    let src = "def loop_forever(n) = loop_forever(n + 1)
let m = begin
  loop_forever(0)
rescue e
  e.message
end
[m, 1 + 1]";
    assert_eq!(
        run(src),
        "=> [\"too many nested calls (1000): infinite recursion?\", 2]"
    );
}

#[test]
fn raised_objects_keep_their_fields() {
    let src = "type Failure
  message: str
  code: int
end
def check(n)
  raise Failure(message: \"too big\", code: n) if n > 10
  n
end
def wrap(n)
  begin
    check(n)
  rescue e: Failure
    raise e
  end
end
begin
  wrap(42)
rescue f: Failure
  [f.code, f.message]
end";
    assert_eq!(run(src), "=> [42, \"too big\"]");
}

#[test]
fn rescue_variables_are_scoped_and_capturable() {
    let src = "let fs = []
for i in 0..3
  begin
    raise \"n#{i}\"
  rescue e
    fs << [1].map do |_|
      e.message
    end
  end
end
fs";
    assert_eq!(run(src), "=> [[\"n0\"], [\"n1\"], [\"n2\"], [\"n3\"]]");
    let out = run("begin\n  raise \"x\"\nrescue e\n  1\nend\ne");
    assert!(out.contains("`e` is not defined"), "{out}");
}

#[test]
fn syntax_of_clauses() {
    let out = run("begin\n  1\nelse\n  2\nend");
    assert!(
        out.contains("`else` in a `begin` requires a `rescue`"),
        "{out}"
    );
    let out = run("rescue\n  1\nend");
    assert!(
        out.starts_with("syntax error: `rescue` outside of a `begin`, `def` or `do` block"),
        "{out}"
    );
    // `then` on the same line.
    assert_eq!(run("begin raise \"x\" rescue e then 5 end"), "=> 5");
}

#[test]
fn question_variants_return_nil() {
    let src = "let xs = [1, 2, 3]
[xs[1]?, xs[7]?, xs[-3]?, xs[-4]?, \"ab\"[1]?, \"ab\"[5]?, {\"a\" => 1}[\"z\"]?, (1, 2)[9]?]";
    assert_eq!(run(src), "=> [2, nil, 1, nil, \"b\", nil, nil, nil]");
    let src = "[\"42\".to_i?(), \" 7 \".to_i?(), \"x\".to_i?(), \"1.5\".to_f?(), \"no\".to_f?(),
 2.7.to_i?(), 1e300.to_i?(), 3.to_i?(), 3.to_f?()]";
    assert_eq!(run(src), "=> [42, 7, nil, 1.5, nil, 2, nil, 3, 3.0]");
    // The usual pattern: a default value.
    assert_eq!(
        run("let port: int = \"80x\".to_i?() or 8080\nport"),
        "=> 8080"
    );
    // In a hot loop (compiled code), the same results.
    let src = "def at(xs, i) = xs[i]? or -1
let t = 0
for i in 0..2000
  t += at([1, 2, 3], i % 5)
end
t";
    assert_eq!(run(src), "=> 1601");
    // A wrong type is still an error.
    let out = run("let xs = [1]\nxs[\"a\"]?");
    assert!(
        out.contains("an index must be an integer, got `str`"),
        "{out}"
    );
    let out = run("[1].to_i?()");
    assert!(out.contains("has no method `to_i?`"), "{out}");
    // Not assignable.
    let out = run("let xs = [1]\nxs[0]? = 2");
    assert!(out.contains("cannot assign to this expression"), "{out}");
}

#[test]
fn names_can_end_with_a_question_mark() {
    let src = "def valid?(s: str): bool = s.to_i?() != nil
class Box
  def init(@v: int)
  end
  def empty?() = @v == 0
end
[valid?(\"12\"), valid?(\"a\"), Box(0).empty?(), Box(1).empty?()]";
    assert_eq!(run(src), "=> [true, false, true, false]");
    // `T?` in types is unchanged.
    assert_eq!(run("def f(x: int?): str? = nil\nf(nil)"), "=> nil");
}
