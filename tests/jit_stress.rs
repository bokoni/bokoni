//! JIT stress test: randomly generated programs (fixed seed, so reproducible) on edge
//! values. Each call is run by the interpreter, by the typed JIT and by the baseline
//! native code (compiled from the first call): results and error messages must be
//! identical.
//!
//! `BK_STRESS=<n>` multiplies the number of programs (default 1).
#![cfg(feature = "jit")]

use bokoni::Engine;

/// Pseudo-random generator (xorshift64*): no dependency, reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len())]
    }
}

const INTS: &[&str] = &[
    "0",
    "1",
    "-1",
    "2",
    "-2",
    "3",
    "7",
    "-7",
    "1000003",
    "4294967296",
    "-4294967296",
    "9007199254740992",
    "9007199254740993",
    "-9007199254740993",
    "4611686018427387904",
    "9223372036854775807",
    "-9223372036854775807",
    "-9223372036854775808",
];

const FLOATS: &[&str] = &[
    "0.0",
    "-0.0",
    "1.0",
    "-1.0",
    "0.5",
    "-2.5",
    "3.0",
    "0.1",
    "1e-300",
    "5e-324",
    "1e300",
    "1.7976931348623157e308",
    "(1e308 * 10.0)",
    "(-1e308 * 10.0)",
    "(0.0 / 0.0)",
    "9007199254740993.0",
];

/// Three engines: interpreter, JIT (typed and baseline), baseline code from the first call.
struct Trio([Engine; 3]);

impl Trio {
    fn new(defs: &str) -> Trio {
        let mk = |jit: bool, eager: bool| {
            let mut e = Engine::new();
            e.set_output(|_| {});
            e.set_jit(jit);
            e.set_jit_eager(eager);
            e.set_max_call_depth(400);
            if let Err(err) = e.eval(defs) {
                panic!(
                    "definitions rejected:\n{}\n{defs}",
                    err.render(defs, "defs")
                );
            }
            e
        };
        Trio([mk(false, false), mk(true, false), mk(true, true)])
    }

    /// Evaluates `call` in the three engines; fails unless the results are identical.
    fn check(&mut self, defs: &str, call: &str) -> String {
        let mut outs: Vec<String> = self
            .0
            .iter_mut()
            .map(|e| match e.eval(call) {
                Ok(v) => e.inspect(&v),
                Err(err) => err.render(call, "appel"),
            })
            .collect();
        assert!(
            outs[0] == outs[1] && outs[0] == outs[2],
            "different results for `{call}`\n\
             interpreter: {}\nJIT: {}\nbaseline: {}\n--- definitions ---\n{defs}",
            outs[0],
            outs[1],
            outs[2]
        );
        outs.swap_remove(0)
    }
}

fn scale() -> usize {
    std::env::var("BK_STRESS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

/// Integer expression over `a`, `b`, `c`.
fn int_expr(r: &mut Rng, depth: u32) -> String {
    if depth == 0 || r.below(4) == 0 {
        return match r.below(3) {
            0 => r.pick(&["a", "b", "c"]).to_string(),
            _ => r.pick(INTS).to_string(),
        };
    }
    let d = depth - 1;
    match r.below(10) {
        0 => format!("(-{})", int_expr(r, d)),
        1 => {
            let op = r.pick(&["<", "<=", ">", ">=", "==", "!="]);
            format!(
                "(if {} {op} {} then {} else {} end)",
                int_expr(r, d),
                int_expr(r, d),
                int_expr(r, d),
                int_expr(r, d)
            )
        }
        _ => {
            let op = r.pick(&["+", "-", "*", "/", "%", "+", "-", "*"]);
            format!("({} {op} {})", int_expr(r, d), int_expr(r, d))
        }
    }
}

/// Float expression over `x`, `y` (with integers mixed in).
fn float_expr(r: &mut Rng, depth: u32) -> String {
    if depth == 0 || r.below(4) == 0 {
        return match r.below(5) {
            0 | 1 => r.pick(&["x", "y"]).to_string(),
            2 => r.pick(INTS).to_string(),
            _ => r.pick(FLOATS).to_string(),
        };
    }
    let d = depth - 1;
    match r.below(10) {
        0 => format!("(-{})", float_expr(r, d)),
        1 => {
            let op = r.pick(&["<", "<=", ">", ">=", "==", "!="]);
            format!(
                "(if {} {op} {} then {} else {} end)",
                float_expr(r, d),
                float_expr(r, d),
                float_expr(r, d),
                float_expr(r, d)
            )
        }
        _ => {
            let op = r.pick(&["+", "-", "*", "/", "%"]);
            format!("({} {op} {})", float_expr(r, d), float_expr(r, d))
        }
    }
}

#[test]
fn random_integer_arithmetic() {
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..150 * scale() {
        let e = int_expr(&mut r, 4);
        // Same expression: typed function (typed JIT) and untyped one (baseline code).
        let defs = format!("def f(a: int, b: int, c: int): int = {e}\ndef g(a, b, c) = {e}\n");
        let mut t = Trio::new(&defs);
        for _ in 0..4 {
            let (a, b, c) = (r.pick(INTS), r.pick(INTS), r.pick(INTS));
            let typed = t.check(&defs, &format!("f({a}, {b}, {c})"));
            let untyped = t.check(&defs, &format!("g({a}, {b}, {c})"));
            // Apart from errors (whose call stack names the function), same value.
            if !typed.contains("error") {
                assert_eq!(typed, untyped, "{defs}");
            }
        }
    }
}

#[test]
fn random_float_arithmetic() {
    let mut r = Rng(0xD1B5_4A32_D192_ED03);
    for _ in 0..150 * scale() {
        let e = float_expr(&mut r, 4);
        let defs = format!("def f(x: float, y: float): float = {e}\ndef g(x, y) = {e}\n");
        let mut t = Trio::new(&defs);
        for _ in 0..4 {
            let (x, y) = (r.pick(FLOATS), r.pick(FLOATS));
            t.check(&defs, &format!("f({x}, {y})"));
            t.check(&defs, &format!("g({x}, {y})"));
            // `int` passed to a `float` parameter.
            t.check(&defs, &format!("f({}, {y})", r.pick(INTS)));
        }
    }
}

#[test]
fn random_mixed_comparisons() {
    // Integers compared with floats: conversions and `int` / `float` equality.
    let mut r = Rng(0x0123_4567_89AB_CDEF);
    let ops = ["<", "<=", ">", ">=", "==", "!="];
    for _ in 0..40 * scale() {
        let op = r.pick(&ops);
        let defs = format!(
            "def f(a: int, x: float): bool = a {op} x\n\
             def h(a: int, x: float): float = a * x + a / 3\n\
             def g(a, x) = a {op} x\n"
        );
        let mut t = Trio::new(&defs);
        for _ in 0..6 {
            let (a, x) = (r.pick(INTS), r.pick(FLOATS));
            t.check(&defs, &format!("f({a}, {x})"));
            t.check(&defs, &format!("h({a}, {x})"));
            t.check(&defs, &format!("g({a}, {x})"));
        }
    }
}

#[test]
fn big_numbers_in_loops_and_recursion() {
    let defs = "def sum(n: int, k: int): int\n  let t = 0\n  let i = 0\n  while i < n\n    t += k * i\n    i += 1\n  end\n  t\nend\n\
                def range_sum(n: int, k: int): int\n  let t = 0\n  for i in 0...n\n    t += i * k - (k / 2)\n  end\n  t\nend\n\
                def tri(n: int, acc: int): int = if n == 0 then acc else tri(n - 1, acc * 3 + n) end\n\
                def pw(x: float, n: int): float = if n == 0 then 1.0 else x * pw(x, n - 1) end\n\
                def tail(n: int): int\n  let t = 0\n  for i in n..9223372036854775807\n    t += i - n + 1\n  end\n  t\nend\n\
                def head(n: int): int\n  let t = 0\n  for i in -9223372036854775808...n\n    t += 1\n  end\n  t\nend\n\
                def total(xs: [int]): int\n  let t = 0\n  for x in xs\n    t += x\n  end\n  t\nend\n\
                def avg(xs: [float]): float\n  let t = 0.0\n  for x in xs\n    t += x\n  end\n  t / xs.len()\nend\n";
    let mut t = Trio::new(defs);
    for call in [
        "sum(1000, 3)",
        "sum(100000, 1000000007)",
        "sum(10, 4611686018427387904)",
        "sum(3, -9223372036854775808)",
        "range_sum(100000, 92233720368)",
        "range_sum(2, -9223372036854775807)",
        "tri(30, 1)",
        "tri(39, 1)",
        "tri(41, 1)",
        "tri(399, 0)",
        "tri(1000, 0)",
        "pw(2.0, 1023)",
        "pw(2.0, 1024)",
        "pw(-1e10, 31)",
        "pw(0.5, 1074)",
        "pw(0.5, 1075)",
        "tail(9223372036854775800)",
        "tail(9223372036854775807)",
        "head(-9223372036854775800)",
        "head(-9223372036854775808)",
        "total([9223372036854775807, 0, -1])",
        "total([9223372036854775807, 1])",
        "total([-9223372036854775808, -1])",
        "total([4611686018427387904, 4611686018427387903, 1])",
        "avg([1e308, 1e308])",
        "avg([1e308, -1e308, 5e-324])",
        "avg([(0.0 / 0.0), 1.0])",
        "avg([1, 2.5, 9007199254740993])",
        "avg([])",
    ] {
        t.check(defs, call);
    }
}

#[test]
fn big_numbers_in_objects() {
    let defs = "type P\n  n: int = 0\n  x: float = 0.0\nend\n\
                def mk(n: int, x: float): P = P(n: n * 2, x: x / 3)\n\
                def grow(p: P, k: int): P = P(n: p.n * k, x: p.x * k)\n\
                def chain(n: int): int\n  let p = mk(1, 1.0)\n  let i = 0\n  while i < n\n    p = grow(p, 3)\n    i += 1\n  end\n  p.n\nend\n\
                def pairs(a: int, b: int): [int] = [a + b, a - b, a * b]\n";
    let mut t = Trio::new(defs);
    for call in [
        "mk(4611686018427387903, 1e308)",
        "mk(4611686018427387904, 1.0)",
        "grow(mk(3, (0.0 / 0.0)), -1)",
        "chain(38)",
        "chain(39)",
        "chain(40)",
        "pairs(9223372036854775807, 0)",
        "pairs(9223372036854775807, 1)",
        "pairs(-9223372036854775808, -1)",
        "pairs(3037000499, 3037000499)",
        "pairs(3037000500, 3037000500)",
    ] {
        t.check(defs, call);
    }
}
