//! Vitesse d'exécution sur quelques programmes typiques, avec et sans JIT.
//!
//! `cargo run --release --example bench_interp`

use bokoni::Engine;
use std::time::Instant;

const PROGRAMS: &[(&str, &str)] = &[
    (
        "fib(25) récursif",
        "def fib(n: int): int\n  if n < 2 then n else fib(n - 1) + fib(n - 2) end\nend\nfib(25)",
    ),
    (
        "boucle while, 1 M tours",
        "def run(): int\n  let i = 0\n  let s = 0\n  while i < 1000000\n    s += i % 7\n    i += 1\n  end\n  s\nend\nrun()",
    ),
    (
        "for sur un intervalle, 1 M tours",
        "def run(): int\n  let s = 0\n  for i in 0...1000000\n    s += i\n  end\n  s\nend\nrun()",
    ),
    (
        "float : série harmonique (1 M)",
        "def h(n: int): float\n  let s = 0.0\n  for i in 1..n\n    s += 1.0 / i\n  end\n  s\nend\nh(1000000)",
    ),
    (
        "listes : map / filter / sum (100 k)",
        "let xs = (0...100000).to_list()\nxs.map(|x| x * 2).filter(|x| x % 3 == 0).sum()",
    ),
    (
        "closures et compteur (300 k appels)",
        "def counter()\n  let n = 0\n  || n += 1\nend\nlet c = counter()\nlet last = 0\nfor i in 0...300000\n  last = c()\nend\nlast",
    ),
    (
        "objets : construction et champs (100 k)",
        "type P\n  x: int = 0\n  y: int = 0\nend\nlet s = 0\nfor i in 0...100000\n  let p = P(x: i, y: 2)\n  s += p.x + p.y\nend\ns",
    ),
    (
        "chaînes : interpolation (100 k)",
        "let n = 0\nfor i in 0...100000\n  let s = \"item #{i}\"\n  n += s.len()\nend\nn",
    ),
];

fn main() {
    println!("{:<42} {:>12} {:>12}", "programme", "interpréteur", "JIT");
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
