//! Compilation to IR: semantics of tricky cases, shape of the generated IR.

use bokoni::{Engine, Value};

fn run(src: &str) -> String {
    let mut e = Engine::new();
    e.set_output(|_| {});
    match e.eval(src) {
        Ok(Value::Nil) => String::new(),
        Ok(v) => e.inspect(&v),
        Err(err) => panic!("{}", err.render(src, "test")),
    }
}

fn err(src: &str) -> String {
    let mut e = Engine::new();
    e.set_output(|_| {});
    match e.eval(src) {
        Ok(v) => panic!("should fail, gave {}", e.inspect(&v)),
        Err(err) => err.to_string(),
    }
}

fn ir(src: &str) -> String {
    Engine::new().ir(src).unwrap()
}

#[test]
fn evaluation_order() {
    // The left operand is read before the assignment on the right modifies it.
    assert_eq!(run("let x = 1\nx + (x = 2)"), "3");
    assert_eq!(run("let x = 1\nlet l = [x, (x = 5), x]\nl"), "[1, 5, 5]");
    // The value of an assignment does not change with the variable.
    assert_eq!(run("let a = 0\nlet b = (a = 3)\na = 9\nb"), "3");
}

#[test]
fn closures_capture_variables() {
    // Each loop iteration has its own variable.
    assert_eq!(
        run("let fs = []\nfor i in 0..2\n  fs << || i\nend\nfs.map(|f| f())"),
        "[0, 1, 2]"
    );
    assert_eq!(
        run("let fs = []\nfor x in [\"a\", \"b\"]\n  fs << || x\nend\nfs.map(|f| f())"),
        "[\"a\", \"b\"]"
    );
    // A closure sees later modifications of the captured variable.
    assert_eq!(
        run("def f()\n  let n = 1\n  let g = || n\n  n = 5\n  g()\nend\nf()"),
        "5"
    );
    // Capture across several levels, and modification from the closure.
    assert_eq!(
        run(
            "def outer()\n  let total = 0\n  let add = |x| [1].each do |_|\n    total += x\n  end\n  add(2)\n  add(3)\n  total\nend\nouter()"
        ),
        "5"
    );
    // A local function can call itself.
    assert_eq!(
        run(
            "def f()\n  def fact(n) = if n < 2 then 1 else n * fact(n - 1) end\n  fact(5)\nend\nf()"
        ),
        "120"
    );
    // A variable declared later in the function is visible to closures…
    assert_eq!(
        run("def f()\n  let g = || x\n  let x = 5\n  g()\nend\nf()"),
        "5"
    );
    // …but reading it before its declaration is an error.
    assert!(
        err("def f()\n  let g = || x\n  g()\n  let x = 5\nend\nf()")
            .contains("before its declaration")
    );
}

#[test]
fn typed_registers_keep_their_type() {
    // `n` is an `int` (inferred): a dynamic value of another type is rejected at
    // runtime, even though the checker could not know it.
    let e = err("def any(x) = x\ndef f()\n  let n = 0\n  n = any(\"a\")\nend\nf()");
    assert!(
        e.contains("a variable keeps the type of its first value"),
        "{e}"
    );
    // `int` to `float` is converted.
    assert_eq!(
        run("def f(): float\n  let x = 1.5\n  x = 2\n  x\nend\nf()"),
        "2.0"
    );
    // Without static checks, no typed registers: dynamic behavior.
    let mut en = Engine::new();
    en.set_static_checks(false);
    assert!(
        en.eval("def f()\n  let n = 0\n  n = \"a\"\n  n\nend\nf()")
            .is_ok()
    );
}

#[test]
fn construction_blocks_resolve_names() {
    let src = "type Route\n  path: str\nend\ntype Server\n  mode = \"dev\"\n  routes: [Route] = []\nend\n";
    // In a nested block, a field of the outer block is readable.
    assert_eq!(
        run(&format!(
            "{src}let s = Server do\n  mode = \"prod\"\n  routes << Route do\n    path = \"/#{{mode}}\"\n  end\nend\ns.routes[0].path"
        )),
        "\"/prod\""
    );
    // A field shadows an outer variable of the same name.
    assert_eq!(
        run(&format!(
            "{src}let mode = \"outer\"\nServer do\n  mode = \"field\"\nend.mode"
        )),
        "\"field\""
    );
    assert_eq!(
        run(&format!(
            "{src}def f()\n  let mode = \"x\"\n  Server do\n  end\nend\nf().mode"
        )),
        "\"dev\""
    );
    // An ordinary `do` block sees outer variables.
    assert_eq!(
        run("def f()\n  let t = 0\n  [1, 2].each do |n|\n    t += n\n  end\n  t\nend\nf()"),
        "3"
    );
}

#[test]
fn state_persists_between_evaluations() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("let n = 1\ndef inc() = n += 1").unwrap();
    e.eval("inc()\ninc()").unwrap();
    assert_eq!(e.eval_as::<i64>("n").unwrap(), 3);
    // Redefining a function replaces the old one.
    e.eval("def inc() = n += 10").unwrap();
    e.eval("inc()").unwrap();
    assert_eq!(e.eval_as::<i64>("n").unwrap(), 13);
}

#[test]
fn declarations_are_top_level() {
    let e = err("def f()\n  type T\n    x: int\n  end\nend");
    assert!(e.contains("at the top level"), "{e}");
}

#[test]
fn ir_uses_typed_instructions() {
    let typed =
        ir("def f(n: int): int\n  let s = 0\n  for i in 0...n\n    s += i * 2\n  end\n  s\nend");
    assert!(typed.contains("*i 2"), "{typed}"); // immediate operand
    assert!(typed.contains("+i"), "{typed}");
    assert!(typed.contains("if r"), "{typed}"); // fused compare and branch
    assert!(typed.contains("typed registers"), "{typed}");
    // Unannotated parameter: generic operation.
    let generic = ir("def g(a, b) = a + b");
    assert!(generic.contains(" + r"), "{generic}");
    assert!(!generic.contains("+i"), "{generic}");
}

#[test]
fn annotated_fields_are_typed() {
    let src = "type P\n  x: int = 0\n  port = 80\nend\ndef f(p: P): int = p.x * 2\ndef g(p: P) = p.port * 2";
    let out = ir(src);
    assert!(out.contains("r0.x : int"), "{out}");
    assert!(out.contains("*i 2"), "{out}");
    // Type inferred from the default value, not checked at runtime: generic.
    assert!(out.contains("r0.port\n"), "{out}");
}
