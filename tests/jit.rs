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
    // Not compiled: unannotated parameter, unannotated return, built-in function.
    e.eval("def untyped(x) = x + 1\nuntyped(1)\ndef noret(x: int) = x\nnoret(1)\ndef pr(x: int): int\n  print(x)\n  x\nend").unwrap();
    for f in ["untyped", "noret", "pr"] {
        assert!(!e.is_native(f), "`{f}` should not be compiled");
    }
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
    // `sq` can no longer be compiled: `f` is interpreted, still correct.
    e.eval("def sq(x) = x").unwrap();
    assert_eq!(e.eval_as::<i64>("f(3)").unwrap(), 4);
    assert!(!e.is_native("f"));
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
