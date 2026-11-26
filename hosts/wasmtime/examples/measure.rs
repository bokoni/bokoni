//! Time and fuel of a few scripts in wasmtime, interpreted and compiled:
//! `cargo run --release --example measure` (after building `playground/wasm`).

use bokoni_wasmtime::{Host, Limits};
use std::time::Instant;

const SCRIPTS: &[(&str, &str)] = &[
    (
        "fib(25), typed",
        "def fib(n: int): int = if n < 2 then n else fib(n - 1) + fib(n - 2) end\nfib(25)",
    ),
    (
        "objects, 100,000",
        "type P\n  x: int\n  y: int\nend\ndef f(n)\n  let s = 0\n  for i in 0..n\n    let p = P(x: i, y: 2)\n    s += p.x * p.y\n  end\n  s\nend\nf(100000)",
    ),
    (
        "a configuration: records, helpers, a list",
        "def job(name, steps, env = {}) = {name: name, steps: steps, env: env}\n\
         def cargo(id, cmd) = job(\"cargo-#{id}\", [\"cargo #{cmd}\", \"cargo sweep --time 7 .\"])\n\
         let jobs = [cargo(\"test\", \"test\"), cargo(\"clippy\", \"clippy\"), cargo(\"fmt\", \"fmt --check\")]\n\
         jobs.map(|j| j.name)",
    ),
];

fn main() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../playground/wasm/target/wasm32-unknown-unknown/release/bokoni_playground.wasm"
    );
    let wasm = std::fs::read(path).expect("build playground/wasm first");
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(move || {
            let t = Instant::now();
            let host = Host::new(&wasm, Limits::default().stack).unwrap();
            println!(
                "module compiled in {:.0} ms\n",
                t.elapsed().as_secs_f64() * 1e3
            );
            for (name, src) in SCRIPTS {
                println!("{name}");
                for (jit, jit_threshold, label) in [
                    (0, 0, "interpreted"),
                    (1, 0, "compiled"),
                    (1, Limits::default().jit_threshold, "after calls"),
                ] {
                    let limits = Limits {
                        jit,
                        jit_threshold,
                        ..Limits::default()
                    };
                    host.run(src, limits).unwrap();
                    let t = Instant::now();
                    let r = host.run(src, limits).unwrap();
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    println!(
                        "  {:<12} {ms:>7.2} ms  {:>11} fuel  {} modules",
                        label,
                        r.fuel,
                        r.modules
                    );
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
