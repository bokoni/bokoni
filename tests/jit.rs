//! Native compilation: same results and same errors as the interpreter.
#![cfg(feature = "jit")]

use bokoni::Engine;
use std::rc::Rc;

/// Result (or displayed error message, call stack included) of a program.
fn outcome(src: &str, jit: bool) -> String {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.set_jit(jit);
    match e.eval(src) {
        Ok(v) => e.inspect(&v),
        Err(err) => err.render(src, "test"),
    }
}

/// The program gives the same thing with and without the JIT.
fn same(src: &str) -> String {
    let (a, b) = (outcome(src, false), outcome(src, true));
    assert_eq!(a, b, "JIT differs from the interpreter for:\n{src}");
    b
}

#[test]
fn integer_operations_match_the_interpreter() {
    let values = [i64::MIN, i64::MIN + 1, -7, -2, -1, 0, 1, 2, 3, 7, i64::MAX];
    for op in ["+", "-", "*", "/", "%"] {
        let src = format!("def f(a: int, b: int): int = a {op} b\n");
        for &a in &values {
            for &b in &values {
                same(&format!("{src}f({a}, {b})"));
            }
        }
    }
    for op in ["<", "<=", ">", ">=", "==", "!="] {
        for &a in &values[..6] {
            same(&format!(
                "def f(a: int, b: int): bool = a {op} b\nf({a}, -1)"
            ));
        }
    }
    same(&format!("def neg(a: int): int = -a\nneg({})", i64::MIN));
    same("def k(a: int): int = a * 3 + 10 / 4 - a % 5\nk(-17)");
}

#[test]
fn float_operations_match_the_interpreter() {
    let values = ["-7.5", "-1.0", "0.0", "0.5", "3.0", "1e300"];
    for op in ["+", "-", "*", "/", "%", "<", "=="] {
        for a in values {
            for b in values {
                let ret = if matches!(op, "<" | "==") {
                    "bool"
                } else {
                    "float"
                };
                same(&format!(
                    "def f(a: float, b: float): {ret} = a {op} b\nf({a}, {b})"
                ));
            }
        }
    }
    // `int` passed to a `float` parameter, `int` mixed with a `float`.
    same("def h(x: float): float = x * 2 + 1\nh(3)");
    same(
        "def s(n: int): float\n  let t = 0.0\n  for i in 1..n\n    t += 1.0 / i\n  end\n  t\nend\ns(1000)",
    );
}

#[test]
fn programs_match_and_are_compiled() {
    let src = "def fib(n: int): int\n  if n < 2 then n else fib(n - 1) + fib(n - 2) end\nend\n\
               def even(n: int): bool = if n == 0 then true else odd(n - 1) end\n\
               def odd(n: int): bool = if n == 0 then false else even(n - 1) end\n\
               def collatz(n: int): int\n  let steps = 0\n  let x = n\n  while x != 1\n    x = if x % 2 == 0 then x / 2 else 3 * x + 1 end\n    steps += 1\n  end\n  steps\nend\n";
    assert_eq!(
        same(&format!("{src}[fib(20), even(11), odd(11), collatz(27)]")),
        "[6765, false, true, 111]"
    );
    let mut e = Engine::new();
    e.eval(&format!("{src}fib(5)\neven(3)\ncollatz(6)"))
        .unwrap();
    for f in ["fib", "even", "odd", "collatz"] {
        assert!(e.is_native(f), "`{f}` should be compiled");
    }
    // Compiled too: an unannotated parameter (a copy per type of its argument), an
    // unannotated return (any value), a built-in function (called through the
    // interpreter).
    e.set_output(|_| {});
    e.eval("def untyped(x) = x + 1\nuntyped(1)\ndef noret(x: int) = x\nnoret(1)\ndef pr(x: int): int\n  print(x)\n  x\nend\npr(1)").unwrap();
    for f in ["untyped", "noret", "pr"] {
        assert!(e.is_native(f), "`{f}` should be compiled");
    }
    // One copy per type; the same results and errors.
    same(
        "def u(x) = x + 1\ndef run(n: int): float\n  let t = 0.0\n  for i in 0...n\n    t += u(i) + u(i * 0.5)\n  end\n  t\nend\n[u(1), u(1.5), run(10)]",
    );
    same("def u(x) = x + 1\nu(\"a\")");
}

#[test]
fn errors_match_the_interpreter() {
    // Identical message, position and call stack.
    let e = same(
        "def a(x: int): int = b(x) + 1\ndef b(x: int): int = c(x) * 2\ndef c(x: int): int = 10 / x\na(0)",
    );
    assert!(e.contains("division by zero"), "{e}");
    assert!(e.contains("in `c` called from <eval>:2:22"), "{e}");
    assert!(e.contains("in `a` called from <eval>:4:1"), "{e}");
    let e = same("def inf(n: int): int = inf(n + 1)\ninf(0)");
    assert!(e.contains("too many nested calls (1000)"), "{e}");
    same("def big(n: int): int = n * n * n * n * n\nbig(10000)");
}

#[test]
fn redefinition_invalidates_native_code() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("def sq(x: int): int = x * x\ndef f(x: int): int = sq(x) + 1")
        .unwrap();
    assert_eq!(e.eval_as::<i64>("f(3)").unwrap(), 10);
    assert!(e.is_native("f"));
    // New version of `sq`: `f` is recompiled and uses it.
    e.eval("def sq(x: int): int = x * x * x").unwrap();
    assert_eq!(e.eval_as::<i64>("f(3)").unwrap(), 28);
    assert!(e.is_native("f"));
    // `sq` can no longer be compiled: `f` calls it through the interpreter (its result
    // checked when `f` returns it), still correct.
    e.eval("def sq(x) = x").unwrap();
    assert_eq!(e.eval_as::<i64>("f(3)").unwrap(), 4);
    assert!(e.is_native("f"));
    e.eval("def sq(x) = 0.5").unwrap();
    assert!(e.eval("f(3)").unwrap_err().to_string().contains("int"));
}

#[test]
fn deep_recursion_beyond_the_native_stack() {
    // The native stack is not enough for 200,000 calls: the interpreter takes over (its
    // stack grows on the heap), with the same result.
    let src = "def down(n: int): int = if n == 0 then 0 else 1 + down(n - 1) end\ndown(200000)";
    let mut e = Engine::new();
    e.set_max_call_depth(1_000_000);
    assert_eq!(e.eval_as::<i64>(src).unwrap(), 200_000);
    // And the user's limit is respected, natively as elsewhere.
    let mut e = Engine::new();
    e.set_max_call_depth(50);
    let err = e
        .eval("def down(n: int): int = if n == 0 then 0 else 1 + down(n - 1) end\ndown(100)")
        .unwrap_err();
    assert!(
        err.to_string().contains("too many nested calls (50)"),
        "{err}"
    );
}

/// Like `same`, with baseline compilation from the first call of each function.
fn same_eager(src: &str) -> String {
    let interp = outcome(src, false);
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.set_jit_eager(true);
    let native = match e.eval(src) {
        Ok(v) => e.inspect(&v),
        Err(err) => err.render(src, "test"),
    };
    assert_eq!(interp, native, "baseline native code differs for:\n{src}");
    native
}

#[test]
fn untyped_code_matches_the_interpreter() {
    let programs = [
        // Generic arithmetic: integers, floats, mixes, strings, edge cases.
        "def f(a, b) = [a + b, a - b, a * b, a / b, a % b, a < b, a == b]\n[f(7, -2), f(-7.5, 2.0), f(3, 0.5), f(-7, 2)]",
        &format!("def f(a, b) = a + b\nf({}, 1)", i64::MAX),
        &format!("def f(a, b) = a / b\nf({}, -1)", i64::MIN),
        "def f(a, b) = a % b\nf(5, 0)",
        "def f(a, b) = a + b\n[f(\"a\", \"b\"), f([1], [2])]",
        "def f(a) = not a\n[f(nil), f(false), f(0), f(\"\"), f([])]",
        "def f(a) = if a then 1 else 2 end\n[f(nil), f(false), f(true), f(0), f([1])]",
        // Closures, modified captures, counters.
        "def counter()\n  let n = 0\n  || n += 1\nend\nlet c = counter()\nlet last = 0\nfor i in 0...1000\n  last = c()\nend\nlast",
        "def make(k) = |x| x * k\nlet fs = [make(2), make(3.5)]\nfs.map(|f| f(4))",
        "def f()\n  let fs = []\n  for i in 0..3\n    fs << || i * i\n  end\n  fs.map(|g| g())\nend\nf()",
        // Lists, iterators, hashes.
        "let xs = (0...200).to_list()\nxs.map(|x| x * 2).filter(|x| x % 3 == 0).sum()",
        "let t = 0\nfor x in [1, 2.5, 3]\n  t = t + x\nend\nt",
        "let h = {\"a\" => 1, \"b\" => 2}\nlet s = 0\nh.each do |k, v|\n  s += v\nend\ns",
        // Objects, classes, enums.
        "type P\n  x: int = 0\n  y: int = 0\nend\nlet s = 0\nfor i in 0...500\n  let p = P(x: i, y: 2)\n  s += p.x + p.y\nend\ns",
        "type P\n  x: int = 0\n  y = x + 1\nend\n[P(x: 3), P(), P(4, 5), P do\n  x = 9\nend]",
        "type P\n  x: int\nend\nP(x: \"a\")",
        "type P\n  x: int\nend\nP(x: 1, x: 2)",
        "type P\n  x: int\nend\nP(z: 1)",
        "type P\n  x: int\nend\nP()",
        "class C\n  def init(@n: int)\n  end\n  def get() = @n\nend\nlet c = C(4)\nc.get() + C(5).get()",
        // Runtime errors in untyped code, with the call stack.
        "def a(x) = b(x) + 1\ndef b(x) = x.nope()\na(1)",
        "def a(x) = x + \"s\"\na(1)",
        "def f()\n  let g = || x\n  g()\n  let x = 5\nend\nf()",
        "def any(x) = x\ndef f()\n  let n = 0\n  n = any(\"a\")\nend\nf()",
        "def inf(n) = inf(n + 1)\ninf(0)",
        // Field access (cache): several types at the same site, field at another
        // position, class (getter), record, variant, value modified between two reads.
        "type A\n  x: int = 1\nend\ntype B\n  y: int = 2\n  x: int = 3\nend\ndef gx(o) = o.x\nlet t = 0\nfor i in 0...20\n  t += gx(if i % 3 == 0 then A() else B() end)\nend\nt",
        "class C\n  def init(@x: int)\n  end\n  def x = @x * 10\nend\ntype A\n  x: int = 1\nend\ndef gx(o) = o.x\n[gx(A()), gx(C(4)), gx(A()), gx({x: 7})]",
        "class C\n  def init(@x: int)\n  end\nend\ndef gx(o) = o.x\ngx(C(1))",
        "type A\n  x: int = 1\nend\ndef gx(o) = o.x\nlet a = A()\nlet r = [gx(a)]\na.x = 5\nr << gx(a)\nr",
        "type A\n  x: int = 1\nend\ndef gx(o) = o.y\ngx(A())",
        // Globals and captured variables: error cases.
        "def f() = g + 1\nf()",
        "def f()\n  undefined_x = 1\nend\nf()",
        "def f()\n  print = 1\nend\nf()",
        "def f()\n  let g = || y\n  let r = g()\n  let y = 1\n  r\nend\nf()",
        "def f()\n  let n = 0\n  let inc = || n += 1\n  for i in 0...10\n    inc()\n  end\n  n\nend\nf()",
        // Strings.
        "let s = \"\"\nfor i in 0...50\n  s = \"#{s}#{i % 10}\"\nend\ns.len()",
    ];
    for src in programs {
        same_eager(src);
        same(src);
    }
}

#[test]
fn construction_errors_at_run_time() {
    // Without static checks, the errors come from execution.
    let run = |src: &str| {
        let mut e = Engine::new();
        e.set_static_checks(false);
        e.eval(src)
            .map(|v| e.inspect(&v))
            .unwrap_or_else(|err| err.to_string())
    };
    let ty = "type P\n  x: int = 0\n  y: int = 0\nend\n";
    assert!(run(&format!("{ty}P(z: 1, y: 2)")).contains("`P` has no field `z`"));
    assert!(run(&format!("{ty}P(1, x: 2)")).contains("field `x` given twice"));
    assert!(
        run(&format!("{ty}P(y: \"a\", x: 1)")).contains("expected type `int`, got str (\"a\")")
    );
    assert!(run(&format!("{ty}P(1, 2, 3)")).contains("`P` has 2 fields, got 3 arguments"));
    assert_eq!(run(&format!("{ty}P(y: 1, x: 2)")), "P(x: 2, y: 1)");
}

#[test]
fn deep_untyped_recursion() {
    // Deep recursion through every call path: direct, through `map`, a method, a
    // closure. The stack grows on the heap, without overflowing.
    let programs = [
        "def down(n) = if n == 0 then 0 else 1 + down(n - 1) end\ndown(100000)",
        "def down(n) = if n == 0 then 0 else [n].map(|x| down(x - 1))[0] + 1 end\ndown(30000)",
        "class C\n  def down(n) = if n == 0 then 0 else 1 + self.down(n - 1) end\nend\nC().down(50000)",
        "let f = nil\nf = |n| if n == 0 then 0 else 1 + f(n - 1) end\nf(100000)",
        "type T\n  x: int = 0\nend\ndef down(n) = if n == 0 then T() else T do\n  x = down(n - 1).x + 1\nend end\ndown(20000).x",
    ];
    for src in programs {
        for eager in [false, true] {
            let mut e = Engine::new();
            e.set_max_call_depth(10_000_000);
            e.set_jit_eager(eager);
            let r = e
                .eval(src)
                .map(|v| e.inspect(&v))
                .unwrap_or_else(|err| err.to_string());
            assert!(r.chars().all(|c| c.is_ascii_digit()), "{src} → {r}");
        }
    }
}

#[test]
fn field_cache_sees_mutations() {
    assert_eq!(
        same_eager(
            "type A\n  x: int = 1\nend\ndef gx(o) = o.x\nlet a = A()\nlet r = [gx(a)]\na.x = 5\nr << gx(a)\nr"
        ),
        "[1, 5]"
    );
}

#[test]
fn typed_functions_over_objects() {
    let src = "type P\n  x: int = 0\n  y: float = 0.5\n  ok: bool = true\nend\n\
               def dot(a: P, b: P): float = a.x * b.x + a.y * b.y\n\
               def score(p: P): float = if p.ok then dot(p, p) else 0.0 end\n\
               def bad(p: P): int = p.x * 4611686018427387904\n";
    let r = same(&format!(
        "{src}[score(P(x: 3, y: 2)), score(P(x: 1, ok: false)), dot(P(x: 2), P(x: 5, y: 4))]"
    ));
    assert_eq!(r, "[13.0, 0.0, 12.0]");
    // Error in native code on an object: same message, same stack.
    same(&format!("{src}bad(P(x: 3))"));
    let mut e = Engine::new();
    e.eval(&format!("{src}score(P(x: 1))")).unwrap();
    assert!(
        e.is_native("score") && e.is_native("dot"),
        "functions on objects compiled"
    );
}

#[test]
fn redeclared_type_falls_back_to_the_interpreter() {
    // `f` is compiled assuming `P.x: int`; `P` is then redeclared with `x: str`.
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("type P\n  x: int = 1\nend\ndef f(p: P): int = p.x + 1\nf(P())")
        .unwrap();
    assert!(e.is_native("f"));
    e.eval("type P\n  x: str = \"a\"\nend").unwrap();
    let err = e.eval("f(P())").unwrap_err().to_string();
    assert!(
        err.contains("field `x` should be of type `int`, got str"),
        "{err}"
    );
    assert!(!e.is_native("f"));
    // Redeclared with the same shape, then another layout: recompiled, correct.
    e.eval("type P\n  x: int = 2\nend").unwrap();
    assert_eq!(e.eval_as::<i64>("f(P())").unwrap(), 3);
    e.eval("type P\n  pad: str = \"\"\n  x: int = 7\nend")
        .unwrap();
    assert_eq!(e.eval_as::<i64>("f(P())").unwrap(), 8);
    assert!(e.is_native("f"));
}

#[test]
fn typed_fields_in_untyped_code() {
    // Typed fields read in functions the typed JIT does not compile (unannotated
    // return, loop over a list, local object): baseline code, same result.
    let ty = "type P\n  x: int = 0\n  y: float = 1.5\n  name: str = \"p\"\nend\n";
    for body in [
        "def f(p: P) = p.x + p.y\nf(P(x: 2))",
        "def f(ps: [P]): float\n  let t = 0.0\n  for p in ps\n    t += p.x * p.y\n  end\n  t\nend\nf([P(x: 1), P(x: 2, y: 2.0)])",
        "def f(n: int): int\n  let p = P(x: n)\n  p.x * 2\nend\nf(21)",
        "def f(p: P) = \"#{p.name}:#{p.x}\"\nf(P(x: 4))",
        "def f(p: P): int = p.x * 4611686018427387904\nf(P(x: 3))",
    ] {
        same_eager(&format!("{ty}{body}"));
        same(&format!("{ty}{body}"));
    }
}

#[test]
fn typed_functions_over_lists() {
    let src = "type P\n  x: int = 0\n  w: float = 1.0\nend\n\
               def total(xs: [int]): int\n  let t = 0\n  for x in xs\n    t += x\n  end\n  t\nend\n\
               def mean(xs: [float]): float\n  let t = 0.0\n  for x in xs\n    t += x\n  end\n  t / xs.len()\nend\n\
               def wsum(ps: [P]): float\n  let t = 0.0\n  for p in ps\n    t += p.x * p.w\n  end\n  t\nend\n\
               def at(xs: [int], i: int): int = xs[i] * 10\n\
               def first(ps: [P]): int = ps[0].x + ps[-1].x\n\
               def pass(xs: [int]): int = total(xs) + xs.len()\n";
    let cases = [
        "[total([1, 2, 3]), total([]), mean([1.0, 2, 4.5]), wsum([P(x: 2), P(x: 3, w: 0.5)])]",
        "[at([5, 6, 7], 1), at([5, 6, 7], -1), first([P(x: 9), P(x: 1)]), pass([4, 5])]",
        // Edge cases: the interpreter takes over again, same errors.
        "at([5, 6, 7], 3)",
        "at([5, 6, 7], -4)",
        "first([])",
        &format!("total([{}, 1])", i64::MAX),
    ];
    for c in cases {
        same(&format!("{src}{c}"));
    }
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(src).unwrap();
    e.eval(cases[0]).unwrap();
    e.eval(cases[1]).unwrap();
    for f in ["total", "mean", "wsum", "at", "first", "pass"] {
        assert!(e.is_native(f), "`{f}` should be compiled");
    }
}

#[test]
fn list_with_unexpected_values_falls_back() {
    // A list annotated `[int]` that later receives a string (untyped code): rejected at
    // the call (the annotation checks the elements), natively as when interpreted.
    let src = "def total(xs: [int]): int\n  let t = 0\n  for x in xs\n    t += x\n  end\n  t\nend\n\
               def push(l, v) = l << v\n\
               let xs: [int] = [1, 2]\ntotal(xs)\npush(xs, \"a\")\ntotal(xs)";
    let r = same(src);
    assert!(r.contains("expected type `[int]`"), "{r}");
}

#[test]
fn interpolation() {
    let ty = "type P\n  x: int = 0\n  def to_s = \"P<#{self.x}|#{[self.x, 1.5]}>\"\nend\n\
              type Q\n  def to_s = 1 / 0\nend\n";
    assert_eq!(
        same_eager(&format!(
            "{ty}let w = \"word\"\n\"#{{1}} #{{-2.5}} #{{0.1}} #{{nil}} #{{true}} #{{[1, w]}} #{{w}} #{{P(x: 3)}}\""
        )),
        "\"1 -2.5 0.1 nil true [1, \\\"word\\\"] word P<3|[3, 1.5]>\""
    );
    // Error during display, then a normal interpolation: nothing remains of the text
    // that was started.
    let mut e = Engine::new();
    e.eval(ty).unwrap();
    assert!(e.eval("\"a #{Q()} b\"").is_err());
    assert_eq!(e.eval_as::<String>("\"x#{1}y\"").unwrap(), "x1y");
}

const VEC: &str = "type V\n  x: float = 0.0\n  y: float = 0.0\nend\n\
    def add(a: V, b: V): V = V(x: a.x + b.x, y: a.y + b.y)\n\
    def scale(v: V, k: float): V = V(x: v.x * k, y: v.y * k)\n\
    def id(v: V): V = v\n\
    def pick(a: V, b: V, c: bool): V = if c then a else b end\n\
    def run(n: int): float\n  let p = V(x: 0.0, y: 0.0)\n  let d = V(x: 1.0, y: 0.5)\n  let i = 0\n  while i < n\n    p = add(p, scale(d, 2.0))\n    p = id(p)\n    i += 1\n  end\n  p.x + p.y\nend\n\
    def pair(a: int, b: int): [int] = [a, b]\n\
    def both(v: V): [V] = [v, scale(v, 3.0)]\n\
    def firsts(xs: [int]): int = pair(xs[0], xs[-1])[1] + pair(1, 2).len()\n\
    def boom(v: V, n: int): V\n  let w = V(x: v.x, y: 1.0)\n  let k = 10 / n\n  scale(w, k)\nend\n\
    def bail(xs: [int]): V\n  let w = V(x: 2.0, y: 1.0)\n  let y = xs[5]\n  scale(w, y)\nend\n";

#[test]
fn typed_functions_create_objects_and_lists() {
    for c in [
        "[run(1000), add(V(x: 1.0), V(y: 2.0)), id(V(x: 3.0)), pick(V(x: 1.0), V(x: 2.0), false)]",
        "[pair(3, 4), both(V(x: 1.0)), firsts([7, 8, 9])]",
        "boom(V(x: 1.0), 2)",
        "boom(V(x: 1.0), 0)",
        "bail([1, 2])",
        "bail([1, 2, 3, 4, 5, 6])",
    ] {
        same_eager(&format!("{VEC}{c}"));
        same(&format!("{VEC}{c}"));
    }
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(VEC).unwrap();
    e.eval("run(10)\nboth(V())\nfirsts([1, 2])\npick(V(), V(), true)")
        .unwrap();
    for f in [
        "run", "add", "scale", "id", "pick", "pair", "both", "firsts",
    ] {
        assert!(e.is_native(f), "`{f}` should be compiled");
    }
}

#[test]
fn native_allocations_are_freed() {
    // Values created or copied by native code are freed exactly once: the reference
    // count of shared values returns to its starting value.
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(VEC).unwrap();
    e.eval("let vs = [V(x: 1.0), V(x: 2.0)]\nlet v = V(x: 5.0)\n\
            def keep(xs: [V]): [V] = [xs[0], xs[1], xs[0]]\n\
            def again(xs: [V]): V\n  let w = xs[0]\n  w = xs[1]\n  w\nend\n\
            def loop2(xs: [V], n: int): float\n  let t = 0.0\n  let i = 0\n  while i < n\n    for p in keep(xs)\n      t += add(p, again(xs)).x\n    end\n    i += 1\n  end\n  t\nend")
        .unwrap();
    let counts = |e: &mut Engine| -> Vec<usize> {
        let (bokoni::Value::List(ref l), bokoni::Value::Object(ref v)) =
            (e.eval("vs").unwrap(), e.eval("v").unwrap())
        else {
            panic!()
        };
        let mut c = vec![Rc::strong_count(l), Rc::strong_count(v)];
        c.extend(l.borrow().iter().map(|x| match x {
            bokoni::Value::Object(o) => Rc::strong_count(o),
            _ => 0,
        }));
        c
    };
    let before = counts(&mut e);
    for call in [
        "loop2(vs, 1000)",
        "keep(vs)",
        "again(vs)",
        "id(v)",
        "pick(v, v, true)",
        "both(v)",
        "[keep(vs), both(v)]",
        "boom(v, 0)",
        "bail([1])",
    ] {
        let _ = e.eval(call);
        assert_eq!(counts(&mut e), before, "after `{call}`");
    }
    assert!(e.is_native("loop2") && e.is_native("keep") && e.is_native("again"));
}

const SHAPES: &str = "class Vec2
  def init(@x: float, @y: float)
  end
  def norm2(): float = @x * @x + @y * @y
  def dot(o: Vec2): float = @x * o.x() + @y * o.y()
  def x(): float = @x
  def y(): float = @y
  def scaled(): float = self.norm2() * 2.0
end
class Vec3 < Vec2
  def init(@x: float, @y: float, @z: float)
  end
  def norm2(): float = @x * @x + @y * @y + @z * @z
end
class Counter
  def init(@count: int, @total: float)
  end
  def add(n: int): int
    @count += n
    @total = @total + n
    @count
  end
  def count(): int = @count
end
def total(ps: [Vec2]): float
  let t = 0.0
  for p in ps
    t += p.norm2()
  end
  t
end
def bump(c: Counter): int = c.add(1)
";

#[test]
fn methods_match_and_are_compiled() {
    // From the interpreter, from native code (`total`, `scaled`), overridden by a
    // subclass (abandon: the interpreter redoes the call), fields written.
    let src = format!(
        "{SHAPES}let v = Vec2(3.0, 4.0)
let s = 0.0
for i in 0..200
  s += v.norm2() + v.scaled()
end
let c = Counter(0, 0.5)
for i in 1..100
  c.add(i)
end
[s, total([v, Vec2(1.0, 1.0)]), total([v, Vec3(1.0, 2.0, 2.0)]), v.dot(Vec2(2.0, 1.0)),
 c.count(), bump(c), Vec3(1.0, 1.0, 1.0).scaled()]"
    );
    assert_eq!(same(&src), "[15075.0, 27.0, 34.0, 10.0, 5050, 5051, 6.0]");
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(&src).unwrap();
    for (t, m) in [("Vec2", "norm2"), ("Vec2", "dot"), ("Counter", "add")] {
        assert!(e.is_native_method(t, m), "`{t}.{m}` should be compiled");
    }
    // A native caller of a method that writes fields: compiled too (a resumption in the
    // interpreter redoes nothing).
    assert!(e.is_native("bump"));
    // `total` and `scaled` call `norm2` directly; on a `Vec3` (which redefines it) the
    // interpreter resumes, and they are recompiled on their next call.
    let mut e = Engine::new();
    e.eval(&format!(
        "{SHAPES}Vec2(1.0, 2.0).scaled() + total([Vec2(1.0, 1.0)])"
    ))
    .unwrap();
    assert!(e.is_native_method("Vec2", "scaled"));
    assert!(e.is_native("total"));
    assert_eq!(
        e.eval_as::<f64>("Vec3(1.0, 1.0, 1.0).scaled()").unwrap(),
        6.0
    );
    assert!(!e.is_native_method("Vec2", "scaled"));
}

#[test]
fn methods_errors_match_the_interpreter() {
    // Field read before being initialized (method called by `init`): abandoned on entry.
    same(
        "class A\n  def init(@a: int)\n    @b = self.twice()\n  end\n  b: int = 0\n  def twice(): int = @b * 2\nend\nA(1)",
    );
    same(
        "class A\n  def init(@n: int)\n  end\n  def big(): int = @n * 9223372036854775807\nend\nA(2).big()",
    );
    same(
        "class A\n  def init(@n: int)\n  end\n  def half(d: int): int = @n / d\nend\nlet a = A(4)\n[a.half(2), a.half(0)]",
    );
}

#[test]
fn methods_of_generic_classes_are_compiled_per_type() {
    let src = "class Acc(T)
  def init(@sum: T, @n: int)
  end
  def add(x: T): T
    @sum = @sum + x
    @n += 1
    @sum
  end
  def get(): T = @sum
  def twice(): T = self.get() + self.get()
end
def run(a: Acc(int), k: int): int
  let t = 0
  for i in 0...k
    t += a.twice()
  end
  t
end
let i = Acc(0, 0)
let f = Acc(0.5, 0)
let s = Acc(\"\", 0)
for k in 0...100
  i.add(k)
  f.add(k * 0.5)
  s.add(\"x\")
end
[i.get(), f.get(), s.get().len(), i.twice(), f.twice(), run(i, 10)]";
    assert_eq!(same(src), "[4950, 2475.5, 100, 9900, 4951.0, 99000]");
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(src).unwrap();
    // A copy for `int` and one for `float`; the strings stay interpreted.
    for m in ["add", "get", "twice"] {
        assert!(e.is_native_method("Acc", m), "`Acc.{m}` should be compiled");
    }
    // A field of type `T` holding something else than expected: checked on entry.
    assert_eq!(
        e.eval_as::<String>("let z = Acc(1, 0)\nz.add(2)\nlet w = Acc(\"a\", 0)\nw.add(\"b\")")
            .unwrap(),
        "ab"
    );
}

#[test]
fn typed_code_calls_methods_of_generic_objects() {
    let src = "class Acc(T)
  def init(@sum: T, @n: int)
  end
  def get(): T = @sum
  def scaled(k: T): T = @sum * k
end
def run(a: Acc(int), k: int): int
  let t = 0
  for i in 0...k
    t += a.get() + a.scaled(2)
  end
  t
end
def runf(a: Acc(float), k: int): float
  let t = 0.0
  for i in 0...k
    t += a.scaled(0.5)
  end
  t
end
[run(Acc(3, 0), 100), runf(Acc(4.0, 0), 10)]";
    assert_eq!(same(src), "[900, 20.0]");
    let mut e = Engine::new();
    e.eval(src).unwrap();
    // The annotation `Acc(int)` gives the copy of the methods to call directly.
    assert!(e.is_native("run"));
    assert!(e.is_native("runf"));
    // An object that does not match (static checks off): checked on entry of the
    // method's copy, the call is redone by the interpreter, which gives its error.
    let bad = |jit: bool| {
        let mut e = Engine::new();
        e.set_jit(jit);
        e.set_static_checks(false);
        e.eval(src).unwrap();
        e.eval("run(Acc(1.5, 0), 2)").unwrap_err().to_string()
    };
    assert_eq!(bad(true), bad(false));
    assert!(bad(true).contains("expected type `int`, got float"));
}

#[test]
fn native_code_resumes_in_the_interpreter() {
    // In the middle of a `for` over a list: an element whose class redefines the method
    // called; the interpreter goes on from that element (not from the start).
    let src = "class A
  def init(@v: int)
  end
  def f(): int = @v
end
class B < A
  def f(): int = @v * 100
end
def total(xs: [A]): int
  let t = 0
  let k = 0
  for x in xs
    k += 1
    t += x.f() * k
  end
  t
end
[total([A(1), A(2), B(3), A(4)]), total([A(1), A(1)])]";
    assert_eq!(same(src), "[921, 3]");
    // Index out of bounds after a few iterations: the interpreter's error, its stack.
    same(
        "def g(xs: [int], n: int): int\n  let t = 0\n  for i in 0..n\n    t += xs[i]\n  end\n  t\nend\ndef h(): int = g([1, 2, 3], 5)\nh()",
    );
    // Deep recursion (beyond what the native stack allows at once) and infinite
    // recursion: the same result and error.
    same("def d(n: int): int = if n == 0 then 0 else 1 + d(n - 1) end\nd(900)");
    same("def inf(n: int): int = inf(n + 1) + 1\ninf(0)");
    // A typed copy (`T = int`) resumed returns a `float`: its native caller resumes too.
    same(
        "class Acc(T)\n  def init(@s: T)\n  end\n  def get(): T = @s\nend\ndef run(a: Acc(int)): int = a.get() + 1\nrun(Acc(2))",
    );
}

#[test]
fn native_code_calls_methods_that_write_fields() {
    // Called in a typed loop; resumed in the middle (a subclass redefines `add`): the
    // writes already done are not redone.
    let src = "class Counter
  def init(@n: int, @total: float)
  end
  def add(k: int): int
    @n += k
    @total = @total + k * 0.5
    @n
  end
  def n(): int = @n
  def total(): float = @total
end
class Loud < Counter
  def add(k: int): int
    @n += k * 10
    @n
  end
end
def run(cs: [Counter], k: int): int
  let t = 0
  for i in 0...k
    for c in cs
      t += c.add(i)
    end
  end
  t
end
let a = Counter(0, 0.0)
let b = Counter(5, 1.0)
let r1 = run([a, b], 100)
let l = Loud(0, 0.0)
let r2 = run([a, l, b], 10)
[r1, r2, a.n(), a.total(), b.n(), l.n()]";
    assert_eq!(same(src), "[333800, 101030, 4995, 2497.5, 5000, 450]");
    let mut e = Engine::new();
    e.eval(src).unwrap();
    assert!(e.is_native_method("Counter", "add"));
}

#[test]
fn lists_modified_in_native_code() {
    let src = "def doubled(xs: [int]): [int]
  let out: [int] = []
  for x in xs
    out << x * 2
  end
  out
end
def bump(xs: [float]): int
  let i = 0
  for x in xs
    xs[i] = x + 0.5
    i += 1
  end
  xs.len()
end
def grow(xs: [int], n: int): int
  for x in xs
    xs << x
  end
  let t = 0
  for i in 0...n
    xs << i
    t += xs[-1]
  end
  xs.len()
end
class Bag
  def init(@items: [int], @n: int)
  end
  def add(x: int): int
    @items << x
    @n += 1
    @items.len()
  end
  def reset(): int
    @items = []
    @n
  end
  def total(): int
    let t = 0
    for x in @items
      t += x
    end
    t
  end
end
def fill(b: Bag, k: int): int
  let t = 0
  for i in 0...k
    t += b.add(i)
  end
  t + b.total()
end
let fs = [1.0, 2.0]
let b = Bag([], 0)
[doubled([1, 2, 3]), bump(fs), fs, grow([1, 2], 3), fill(b, 4), b.reset(), b.total(),
 doubled([]), bump([1.0, 2])]";
    assert_eq!(same(src), "[[2, 4, 6], 2, [1.5, 2.5], 7, 16, 4, 0, [], 2]");
    let mut e = Engine::new();
    e.eval(src).unwrap();
    for f in ["doubled", "bump", "grow", "fill"] {
        assert!(e.is_native(f), "`{f}` should be compiled");
    }
    for m in ["add", "reset", "total"] {
        assert!(e.is_native_method("Bag", m), "`Bag.{m}` should be compiled");
    }
    // Out of bounds: the interpreter's error.
    same("def w(xs: [int]): int\n  xs[5] = 1\n  0\nend\nw([1])");
}

#[test]
fn small_functions_and_getters_are_inlined() {
    let src = "class P
  def init(@x: float, @y: float)
  end
  def x(): float = @x
  def y(): float = @y
  def norm2(): float = @x * @x + @y * @y
end
class Q < P
  def x(): float = @x * 10.0
end
def sq(v: float): float = v * v
def total(ps: [P]): float
  let t = 0.0
  for p in ps
    t += sq(p.x()) + p.y() + p.norm2()
  end
  t
end
def big(n: int): int = n * 4611686018427387904
def use_big(n: int): int = big(n) + 1
[total([P(1.0, 2.0), P(3.0, 4.0)]), total([P(1.0, 1.0), Q(1.0, 1.0), P(2.0, 0.0)])]";
    assert_eq!(same(src), "[46.0, 115.0]");
    // (Without the call on a `Q`, which resumes in the interpreter.)
    let defs = &src[..src.rfind("\n[").unwrap()];
    let mut e = Engine::new();
    e.eval(&format!("{defs}\ntotal([P(1.0, 2.0)])")).unwrap();
    assert!(e.is_native("total"));
    // An error in inlined code: the same message and call stack.
    same(&format!("{src}\nuse_big(1)"));
    same(&format!("{src}\nuse_big(2)"));
}

#[test]
fn strings_in_native_code() {
    let src = "def rep(s: str, n: int): str
  let out = \"\"
  for i in 0...n
    out = out + s
  end
  if out == \"x\" then \"y\" else \"#{out}:#{n}:#{out.len()}:#{1.5}:#{true}\" end
end
def longest(xs: [str]): str
  let best = \"\"
  for x in xs
    if x.len() > best.len() then best = x end
  end
  best
end
def count(xs: [str], w: str): int
  let n = 0
  for x in xs
    if x == w or x < \"b\" then n += 1 end
  end
  n
end
class Greeter
  def init(@name: str, @n: int)
  end
  def greet(who: str): str
    @n += 1
    @name = @name + \"!\"
    \"#{@name} #{who} #{@n}\"
  end
end
def many(g: Greeter, k: int): int
  let t = 0
  for i in 0...k
    t += g.greet(\"bob\").len()
  end
  t
end
[rep(\"ab\", 3), rep(\"x\", 1), longest([\"a\", \"ccc\", \"bb\"]), count([\"a\", \"b\", \"c\", \"b\"], \"b\"),
 many(Greeter(\"hi\", 0), 3), \"é\".len()]";
    assert_eq!(
        same(src),
        "[\"ababab:3:6:1.5:true\", \"y\", \"ccc\", 3, 30, 1]"
    );
    let mut e = Engine::new();
    e.eval(src).unwrap();
    for f in ["rep", "longest", "count", "many"] {
        assert!(e.is_native(f), "`{f}` should be compiled");
    }
    // A non-string where a string is expected (static checks off): the interpreter's
    // error.
    let bad = |jit: bool| {
        let mut e = Engine::new();
        e.set_jit(jit);
        e.set_static_checks(false);
        e.eval(src).unwrap();
        e.eval("longest([\"a\", 3])")
            .map(|v| e.inspect(&v))
            .unwrap_or_else(|e| e.to_string())
    };
    assert_eq!(bad(true), bad(false));
}

/// Result and printed output of a program.
fn outcome_out(src: &str, jit: bool) -> String {
    let mut e = Engine::new();
    let out = Rc::new(std::cell::RefCell::new(String::new()));
    let o = out.clone();
    e.set_output(move |s| o.borrow_mut().push_str(s));
    e.set_jit(jit);
    let r = match e.eval(src) {
        Ok(v) => e.inspect(&v),
        Err(err) => err.render(src, "test"),
    };
    format!("{}{r}", out.borrow())
}

#[test]
fn native_code_calls_the_interpreter() {
    // `print`, an unannotated function, a built-in method, a function value: done by the
    // interpreter, the rest natively.
    let src = "def untyped(x) = x * 2
def show(n: int): int
  let t = 0
  for i in 0...n
    print(\"i=#{i}\")
    t += untyped(i) + [i, 1].len()
  end
  t
end
def twice(f, x: int): int = f(f(x))
def inc(x: int): int = x + 1
[show(3), twice(inc, 5), twice(|v| v * 3, 2)]";
    assert_eq!(outcome_out(src, true), outcome_out(src, false));
    assert!(outcome_out(src, true).ends_with("[12, 7, 18]"));
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval(src).unwrap();
    assert!(e.is_native("show"));
    // An error in the interpreter: the same message and call stack.
    same("def bad(x) = x + nil\ndef f(n: int): int = n + bad(n)\nf(1)");
    // A result of another type than the typed code expects: resumed after the call.
    same(
        "def any(x) = if x > 1 then 1.5 else x end\ndef g(n: int): int\n  let t = 0\n  for i in 0...n\n    t += any(i)\n  end\n  t\nend\ng(3)",
    );
}

#[test]
fn recompiled_after_a_resumption() {
    // `total` resumes in the interpreter on a `B` (it redefines `f`), then is recompiled
    // with that call done by the interpreter: native again, and right for both.
    let src = "class A
  def init(@v: int)
  end
  def f(): int = @v
end
class B < A
  def f(): int = @v * 100
end
def total(xs: [A]): int
  let t = 0
  for x in xs
    t += x.f()
  end
  t
end
";
    let mut e = Engine::new();
    e.eval(&format!("{src}total([A(1)])")).unwrap();
    assert!(e.is_native("total"));
    assert_eq!(e.eval_as::<i64>("total([A(1), B(2)])").unwrap(), 201);
    assert!(!e.is_native("total"));
    assert_eq!(e.eval_as::<i64>("total([B(1), A(2), B(3)])").unwrap(), 402);
    assert!(e.is_native("total"));
    assert_eq!(e.eval_as::<i64>("total([A(5), B(1)])").unwrap(), 105);
    assert!(e.is_native("total"));
}

#[test]
fn classes_constructed_in_native_code() {
    let src = "class V
  def init(@x: float, @y: float)
  end
  def n2(): float = @x * @x + @y * @y
end
class Named
  def init(@name: str, @tags: [str])
    @size = @tags.len()
  end
  size: int = 0
  def size(): int = @size
end
class Half
  def init(@a: int, b: int)
    if b > 0 then @b = b end
  end
  b: int
end
def total(n: int): float
  let t = 0.0
  for i in 0...n
    let v = V(i * 1.0, 2.0)
    t += v.n2()
  end
  t
end
def half(b: int): int
  let h = Half(1, b)
  1
end
[total(100), Named(\"x\", [\"a\", \"b\"]).size()]";
    assert_eq!(same(src), "[328750.0, 2]");
    let mut e = Engine::new();
    e.eval(src).unwrap();
    assert!(e.is_native("total"));
    // A field not initialized by `init`: the interpreter's error.
    same(&format!("{src}\nhalf(1)"));
    same(&format!("{src}\nhalf(0)"));
}

#[test]
fn for_sees_the_list_as_it_was() {
    // The list is not copied when the loop starts: modified during the loop (by the loop
    // itself, by a function, by an inner loop), the loop still sees it as it was.
    let src = "let xs = [1, 2, 3]
let seen = []
for x in xs
  seen << x
  if x == 1
    xs << 10
    xs[2] = 30
  end
end
def grow(l)
  l << 99
  l.pop()
  l.pop()
end
let ys = [1, 2, 3, 4]
let total = 0
for y in ys
  total += y
  grow(ys)
end
let zs = [1, 2]
let pairs = []
for a in zs
  for b in zs
    pairs << a * 10 + b
    zs << 0 if a == 1 and b == 1
  end
end
let ws = [5, 6, 7]
for w in ws
  break if w == 6
end
ws << 8
for w in ws
  ws[0] = 0
  total += w
end
def typed(xs: [int]): int
  let t = 0
  for x in xs
    xs << x if x < 3
    t += x
  end
  t
end
[seen, xs, total, pairs, ws, typed([1, 2, 3, 4]), typed([5])]";
    assert_eq!(
        same(src),
        "[[1, 2, 3], [1, 2, 30, 10], 36, [11, 12, 21, 22, 20], [0, 6, 7, 8], 10, 5]"
    );
}

#[test]
fn loops_of_the_main_program() {
    // Outlined into functions compiled natively, globals read in place: same results.
    for src in [
        "let t = 0\nfor i in 0...1000\n  t += i % 7\nend\nt",
        // A global that changes type during the loop (resumed in the interpreter).
        "let t = 0\nfor i in 0...100\n  t += 1\n  t = 0.5 if i == 50\nend\nt",
        "let t = 0\nlet s = \"\"\nfor i in 0...20\n  if i == 10\n    t = \"x\"\n  end\n  s = \"#{t}\"\nend\n[t, s]",
        // `break`, `next`, nested loops, `while`, a list, a register read after.
        "let n = 0\nlet k = 0\nwhile k < 50\n  k += 1\n  next if k % 3 == 0\n  for j in 0..k\n    n += j\n    break if j > 5\n  end\nend\n[n, k]",
        "let xs = [3, 1, 2]\nlet t = 0\nfor x in xs\n  t = t * 10 + x\nend\nt",
        "print(\"a\")\nlet t = 1.5\nfor i in 0...10\n  t = t * 2.0\nend\nprint(t)\nt",
        "let flag = false\nlet c = 0\nfor i in 0...10\n  flag = not flag\n  c += 1 if flag\nend\n[flag, c]",
        "def f(x: int): int = x * 2\nlet t = 0\nfor i in 0...100\n  t += f(i)\nend\nt",
        "let t = 0\nlet r = [for i in 0...3\n  t += i\nend, t]\nr",
        // Errors: the global is up to date, the call stack is the main program's.
        "let t = 0\nfor i in 0...10\n  t += 100 / (5 - i)\nend",
        "def g(x) = x.nope()\nlet t = 0\nfor i in 0...10\n  t += 1\n  g(i) if i == 3\nend",
        "const K = 1\nfor i in 0...3\n  K = i\nend",
        "let t = 9223372036854775800\nfor i in 0...100\n  t += 1\nend",
        // Objects: their methods called directly, the class checked on every read.
        "class A\n  def init(@n: int)\n  end\n  def add(k: int): int\n    @n += k\n    @n\n  end\nend\nclass B < A\n  def add(k: int): int = 0\nend\nlet a = A(0)\nlet t = 0\nfor i in 0...100\n  t += a.add(i)\n  a = B(1) if i == 60\nend\n[t, a.add(1)]",
        "class Acc(T)\n  def init(@sum: T)\n  end\n  def add(x: T): T\n    @sum = @sum + x\n    @sum\n  end\nend\nlet a = Acc(0.0)\nfor i in 0...100\n  a.add(i * 0.5)\nend\na.add(0.0)",
    ] {
        same(src);
    }
    // After an error, the global has the last value assigned.
    let mut e = Engine::new();
    e.set_output(|_| {});
    assert!(
        e.eval("let t = 0\nfor i in 0...10\n  t += 1\n  t = t / (5 - i)\nend")
            .is_err()
    );
    assert_eq!(e.eval_as::<i64>("t").unwrap(), 2);
    e.eval("let u = 0\nfor i in 0...100000\n  u += i\nend")
        .unwrap();
    assert_eq!(e.eval_as::<i64>("u").unwrap(), 4999950000);
    // A global of a previous script.
    e.eval("for i in 0...10\n  u -= i\nend").unwrap();
    assert_eq!(e.eval_as::<i64>("u").unwrap(), 4999949955);
}

#[test]
fn closures_with_captured_variables() {
    for src in [
        // Two counters: same code, other captured variables.
        "def counter(start: int)\n  let n = start\n  || n += 1\nend\nlet a = counter(0)\nlet b = counter(100)\nlet t = 0\nfor i in 0...1000\n  t += a()\n  t += b() if i % 2 == 0\nend\n[t, a(), b()]",
        // A captured variable that changes type (resumed in the interpreter).
        "def acc()\n  let s = 0\n  |x| s = s + x\nend\nlet f = acc()\nlet r = []\nfor i in 0...50\n  r << f(if i == 30 then 0.5 else 1 end)\nend\nr.last()",
        "def acc()\n  let s = 0\n  |x| s = if x > 10 then \"big\" else s + x end\nend\nlet f = acc()\nlet r = nil\nfor i in 0...20\n  r = f(i)\nend\nr",
        // Captured float, untyped parameter, called by a built-in.
        "def make(k: float)\n  |x| x * k + 1.0\nend\nlet f = make(0.5)\nlet t = 0.0\nfor i in 0...1000\n  t += f(i * 1.0)\nend\n[t, [1.0, 2.0].map(f)]",
        "def scale(xs: [int], m: int): int = xs.map(|x| x * m).sum()\nscale([1, 2, 3], 4)",
        // Modified by the closure, seen by the function that declared it.
        "def run(): int\n  let total = 0\n  let add = |x: int| total += x\n  for i in 0...100\n    add(i)\n  end\n  total\nend\nrun()",
        // Errors in a closure: same message and call stack.
        "def make(d: int)\n  |x: int| x / d\nend\nlet f = make(0)\nlet t = 0\nfor i in 0...10\n  t += f(i)\nend",
        "def make(xs: [int])\n  |i: int| xs[i]\nend\nlet f = make([1, 2, 3])\nlet t = 0\nfor i in 0...10\n  t += f(i)\nend",
        // Read before being initialized.
        "def outer()\n  let f = |x| y + x\n  let r = f(1)\n  let y = 2\n  r\nend\nouter()",
        // Nested closures.
        "def adder(a: int)\n  |b: int| |c: int| a + b + c\nend\nlet t = 0\nfor i in 0...100\n  t += adder(1)(i)(2)\nend\nt",
    ] {
        same(src);
    }
    // Functions whose variables are captured by their lambdas.
    for src in [
        // A new cell on each turn: each closure its own `j`.
        "def make_all(): int\n  let fs = []\n  for i in 0...5\n    let j = i * 10\n    fs << || j\n  end\n  fs.map(|f| f()).sum()\nend\nmake_all()",
        // Resumed in the interpreter after the closure was created: same cells.
        "let g = 1\ndef run()\n  let total = 0\n  let add = |x| total += x\n  for i in 0...20\n    add(g)\n    g = 1.5 if i == 10\n  end\n  [total, add(0)]\nend\nrun()",
        "def run(n: int): int\n  let seen = 0\n  let f = |x: int| seen = seen + 1\n  let t = 0\n  for i in 0...n\n    f(i)\n    t += [1, 2][i]\n  end\n  t + seen\nend\nrun(5)",
        "def lazy(): int\n  let f = || y * 2\n  let y = 21\n  f()\nend\nlazy()",
    ] {
        same(src);
    }
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("def run(): int\n  let total = 0\n  let add = |x: int| total += x\n  for i in 0...100\n    add(i)\n  end\n  total\nend\ndef scale(xs: [int], m: int): int = xs.map(|x| x * m).sum()")
        .unwrap();
    assert_eq!(e.eval_as::<i64>("run() + scale([1, 2], 3)").unwrap(), 4959);
    assert!(e.is_native("run"));
    assert!(e.is_native("scale"));
    // The closure called in the loop is compiled.
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("def counter()\n  let n = 0\n  || n += 1\nend\nlet c = counter()\nlet last = 0\nfor i in 0...1000\n  last = c()\nend")
        .unwrap();
    assert!(e.is_native("c"));
    assert_eq!(e.eval_as::<i64>("c()").unwrap(), 1001);
}

#[test]
fn hashes_records_tuples_in_native_code() {
    // Done by the interpreter's code from native code: same results and errors.
    for src in [
        "def tally(xs: [int]): int\n  let h = {}\n  for x in xs\n    let k = x % 3\n    if h.has(k)\n      h[k] = h[k] + 1\n    else\n      h[k] = 1\n    end\n  end\n  let t = 0\n  for k, v in h\n    t += k * 10 + v\n  end\n  t\nend\ntally([1, 2, 3, 4, 5, 6, 7])",
        "def norm(x: float, y: float): float\n  let p = {x: x, y: y}\n  p.x * p.x + p.y * p.y\nend\nlet t = 0.0\nfor i in 0...100\n  t += norm(i * 1.0, 2.0)\nend\nt",
        "def swap(a: int, b: int): int\n  let (x, y) = (b, a)\n  x * 10 + y\nend\nswap(1, 2)",
        "def kind(n: int): str\n  case n % 4\n  when 0 then \"zero\"\n  when 1, 2 then \"low\"\n  else \"high\"\n  end\nend\n(0...6).map(|i| kind(i))",
        "def over(r): int\n  let t = 0\n  for i in r\n    t += i\n  end\n  t\nend\n[over(0..10), over([1, 2]), over(\"ab\".chars().map(|c| c.len()))]",
        "def keys(h: {str => int}): [str]\n  let out: [str] = []\n  for k in h.keys()\n    out << k.upper()\n  end\n  out\nend\nkeys({\"a\" => 1, \"b\" => 2})",
        // Errors in such an instruction.
        "def bad(h: {int => int}, k: int): int = h[k] + 1\nbad({1 => 2}, 3)",
        "def bad(): int\n  let (a, b) = (1, 2, 3)\n  a\nend\nbad()",
        "def bad(x: int): int\n  let h = {}\n  h[[x]] = 1\n  0\nend\nbad(1)",
    ] {
        same(src);
    }
    // Compiled when such instructions are outside its loops (in a loop, the baseline code
    // does them better).
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("def area(w: float, h: float): float\n  let r = {w: w, h: h}\n  let t = 0.0\n  for i in 0...10\n    t += w * h\n  end\n  t + r.w\nend\ndef tally(xs: [int]): int\n  let h = {}\n  for x in xs\n    h[x % 3] = 1\n  end\n  h.len()\nend")
        .unwrap();
    assert_eq!(e.eval_as::<f64>("area(2.0, 3.0)").unwrap(), 62.0);
    assert!(e.is_native("area"));
    assert_eq!(e.eval_as::<i64>("tally([1, 2, 3])").unwrap(), 3);
    assert!(!e.is_native("tally"));
}

#[test]
fn rescue_and_ensure_in_native_code() {
    const ERRS: &str = "class Oops < Error\nend\nclass Other < Error\nend\n";
    for src in [
        // Errors detected by native code, by a function called (native or not), raised.
        "def safe_div(a: int, b: int): int\n  begin\n    a / b\n  rescue e: Error\n    -1\n  end\nend\n[safe_div(7, 2), safe_div(1, 0)]",
        "def add(a: int, b: int): int\n  begin\n    a + b\n  rescue\n    0\n  end\nend\nadd(9223372036854775807, 1)",
        "def f(x: int): int = 10 / x\ndef g(x: int): int\n  begin\n    f(x) + 1\n  rescue e: Error\n    e.message.len()\n  end\nend\n[g(5), g(0)]",
        "def h(x) = x.nope()\ndef g(x: int): str\n  begin\n    h(x)\n    \"ok\"\n  rescue e: Error\n    e.message\n  end\nend\ng(1)",
        &format!(
            "{ERRS}def g(x: int): str\n  begin\n    raise Oops(\"o#{{x}}\") if x > 0\n    \"none\"\n  rescue e: Oops\n    e.message\n  end\nend\n[g(0), g(3)]"
        ),
        // A clause that does not match: raised again (with its call stack).
        &format!(
            "{ERRS}def g(x: int): str\n  begin\n    raise Other(\"x\")\n  rescue e: Oops\n    \"no\"\n  end\nend\ng(1)"
        ),
        // `ensure`: on success, on error, with a `return`.
        "let log = []\ndef g(x: int): int\n  begin\n    100 / x\n  ensure\n    log << x\n  end\nend\n[g(5), begin\n  g(0)\nrescue e: Error\n  -1\nend, log]",
        "let log = []\ndef g(x: int): int\n  begin\n    return 1 if x > 0\n    2\n  ensure\n    log << \"e\"\n  end\nend\n[g(1), g(0), log]",
        // Nested, in a loop, errors counted.
        "def count(xs: [int]): int\n  let bad = 0\n  for x in xs\n    begin\n      begin\n        let y = 10 / x\n      rescue e: Error\n        bad += 1\n        raise e if x == 0 and bad > 5\n      end\n    rescue\n      bad += 100\n    end\n  end\n  bad\nend\ncount([1, 0, 2, 0, 0])",
        // Resumed in the interpreter inside a protected block.
        "def g(xs: [int], i: int): int\n  begin\n    xs[i] * 2\n  rescue e: Error\n    -1\n  end\nend\n[g([1, 2], 1), g([1, 2], 5)]",
        // An error that is not caught (a type error is): out.
        "def g(x: int): int\n  begin\n    x / 0\n  rescue e: Error\n    raise e\n  end\nend\ng(1)",
    ] {
        same(src);
    }
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("def safe_div(a: int, b: int): int\n  begin\n    a / b\n  rescue e: Error\n    -1\n  end\nend")
        .unwrap();
    assert_eq!(
        e.eval_as::<i64>("safe_div(1, 0) + safe_div(9, 3)").unwrap(),
        2
    );
    assert!(e.is_native("safe_div"));
    // In a loop.
    e.eval("def bad(xs: [int]): int\n  let n = 0\n  for x in xs\n    begin\n      n += 10 / x\n    rescue e: Error\n      n -= 1\n    end\n  end\n  n\nend")
        .unwrap();
    assert_eq!(e.eval_as::<i64>("bad([1, 0, 2, 0])").unwrap(), 13);
    assert!(e.is_native("bad"));
}
