//! Execution speed on a few typical programs, with and without the JIT.
//!
//! `cargo run --release --example bench_interp`

use bokoni::Engine;
use std::time::Instant;

const PROGRAMS: &[(&str, &str)] = &[
    (
        "recursive fib(25)",
        "def fib(n: int): int\n  if n < 2 then n else fib(n - 1) + fib(n - 2) end\nend\nfib(25)",
    ),
    (
        "while loop, 1 M iterations",
        "def run(): int\n  let i = 0\n  let s = 0\n  while i < 1000000\n    s += i % 7\n    i += 1\n  end\n  s\nend\nrun()",
    ),
    (
        "for over a range, 1 M iterations",
        "def run(): int\n  let s = 0\n  for i in 0...1000000\n    s += i\n  end\n  s\nend\nrun()",
    ),
    (
        "generic function (forall T), 1 M calls",
        "def max(a: T, b: T): T forall T = if a > b then a else b end\ndef run(): int\n  let s = 0\n  for i in 0...1000000\n    s += max(i % 7, 3)\n  end\n  s\nend\nrun()",
    ),
    (
        "methods: fields read and written, 1 M calls",
        "class Acc\n  def init(@n: int, @sum: float)\n  end\n  def add(x: float): float\n    @n += 1\n    @sum = @sum + x\n    @sum\n  end\n  def mean(): float = @sum / @n\nend\nlet a = Acc(0, 0.0)\nfor i in 0...1000000\n  a.add(i * 0.5)\nend\na.mean()",
    ),
    (
        "generic class (Acc(T)), method, 1 M calls",
        "class Acc(T)\n  def init(@sum: T, @n: int)\n  end\n  def add(x: T): T\n    @n += 1\n    @sum = @sum + x\n    @sum\n  end\nend\nlet a = Acc(0.0, 0)\nfor i in 0...1000000\n  a.add(i * 0.5)\nend\na.add(0.0)",
    ),
    (
        "lists built in typed code (100 k, 10 passes)",
        "def squares(n: int): [int]\n  let out: [int] = []\n  for i in 0...n\n    out << i * i\n  end\n  out\nend\ndef run(): int\n  let t = 0\n  for k in 0...10\n    let xs = squares(100000)\n    xs[0] = k\n    t += xs[k] + xs.len()\n  end\n  t\nend\nrun()",
    ),
    (
        "getters and small functions inlined (1 M)",
        "class P\n  def init(@x: float, @y: float)\n  end\n  def x(): float = @x\n  def y(): float = @y\nend\ndef sq(v: float): float = v * v\ndef total(ps: [P], k: int): float\n  let t = 0.0\n  for i in 0...k\n    for p in ps\n      t += sq(p.x()) + p.y()\n    end\n  end\n  t\nend\ntotal([P(1.0, 2.0), P(3.0, 4.0)], 500000)",
    ),
    (
        "strings in typed code: interpolation, len (100 k)",
        "def label(i: int, name: str): str = \"#{name}-#{i}\"\ndef run(n: int): int\n  let t = 0\n  for i in 0...n\n    let s = label(i, \"item\")\n    if s != \"x\" then t += s.len() end\n  end\n  t\nend\nrun(100000)",
    ),
    (
        "float: harmonic series (1 M)",
        "def h(n: int): float\n  let s = 0.0\n  for i in 1..n\n    s += 1.0 / i\n  end\n  s\nend\nh(1000000)",
    ),
    (
        "lists: map / filter / sum (100 k)",
        "let xs = (0...100000).to_list()\nxs.map(|x| x * 2).filter(|x| x % 3 == 0).sum()",
    ),
    (
        "closures and counter (300 k calls)",
        "def counter()\n  let n = 0\n  || n += 1\nend\nlet c = counter()\nlet last = 0\nfor i in 0...300000\n  last = c()\nend\nlast",
    ),
    (
        "objects: construction and fields (100 k)",
        "type P\n  x: int = 0\n  y: int = 0\nend\nlet s = 0\nfor i in 0...100000\n  let p = P(x: i, y: 2)\n  s += p.x + p.y\nend\ns",
    ),
    (
        "typed objects: dot product (1 M)",
        "type V\n  x: float = 0.0\n  y: float = 0.0\nend\ndef dot(a: V, b: V): float = a.x * b.x + a.y * b.y\ndef run(v: V, n: int): float\n  let t = 0.0\n  let i = 0\n  while i < n\n    t += dot(v, v)\n    i += 1\n  end\n  t\nend\nrun(V(x: 1.5, y: 2.0), 1000000)",
    ),
    (
        "typed lists: 100 k objects created, 10 passes",
        "type P\n  x: int = 0\n  w: float = 1.0\nend\ndef wsum(ps: [P]): float\n  let t = 0.0\n  for p in ps\n    t += p.x * p.w\n  end\n  t\nend\nlet ps = (0...100000).map(|i| P(x: i, w: 0.5))\nlet t = 0.0\nfor k in 0...10\n  t += wsum(ps)\nend\nt",
    ),
    (
        "objects created in typed code: vectors (1 M)",
        "type V\n  x: float = 0.0\n  y: float = 0.0\nend\ndef add(a: V, b: V): V = V(x: a.x + b.x, y: a.y + b.y)\ndef scale(v: V, k: float): V = V(x: v.x * k, y: v.y * k)\ndef run(n: int): float\n  let p = V(x: 0.0, y: 0.0)\n  let d = V(x: 1.0, y: 0.5)\n  let i = 0\n  while i < n\n    p = add(p, scale(d, 0.001))\n    i += 1\n  end\n  p.x + p.y\nend\nrun(1000000)",
    ),
    (
        "strings: interpolation (100 k)",
        "let n = 0\nfor i in 0...100000\n  let s = \"item #{i}\"\n  n += s.len()\nend\nn",
    ),
];

fn main() {
    println!("{:<42} {:>12} {:>12}", "program", "interpreter", "JIT");
    for (name, src) in PROGRAMS {
        let mut times = [0.0; 2];
        let mut result = String::new();
        for (k, jit) in [false, true].into_iter().enumerate() {
            let mut best = f64::MAX;
            for _ in 0..7 {
                let mut e = Engine::new();
                e.set_jit(jit);
                let t = Instant::now();
                let v = e.eval(src).unwrap_or_else(|err| panic!("{name}: {err}"));
                best = best.min(t.elapsed().as_secs_f64());
                result = e.inspect(&v);
            }
            times[k] = best;
        }
        println!(
            "{name:<42} {:>10.1}ms {:>10.1}ms  ×{:<5.1} => {result}",
            times[0] * 1e3,
            times[1] * 1e3,
            times[0] / times[1]
        );
    }
}
