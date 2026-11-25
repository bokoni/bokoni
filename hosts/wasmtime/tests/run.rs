//! Bokoni in wasmtime: the same answers interpreted and compiled, and the limits stop the
//! compiled code too.

use bokoni_wasmtime::{Host, Limits, Run};
use std::sync::OnceLock;

/// The playground's module, built once.
fn host() -> &'static Host {
    static HOST: OnceLock<Host> = OnceLock::new();
    HOST.get_or_init(|| {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../playground/wasm");
        let ok = std::process::Command::new("cargo")
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .current_dir(dir)
            .status()
            .expect("cargo")
            .success();
        assert!(ok, "building the playground's module");
        let wasm = std::fs::read(format!(
            "{dir}/target/wasm32-unknown-unknown/release/bokoni_playground.wasm"
        ))
        .unwrap();
        Host::new(&wasm, Limits::default().stack).unwrap()
    })
}

/// `src` run with `limits`, on a thread whose stack has room for wasmtime's.
fn run(src: &str, limits: Limits) -> wasmtime::Result<Run> {
    let src = src.to_string();
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(move || host().run(&src, limits))
        .unwrap()
        .join()
        .unwrap()
}

fn mode(jit: u32) -> Limits {
    Limits {
        jit,
        ..Limits::default()
    }
}

const PROGRAMS: &[&str] = &[
    "def fib(n: int): int = if n < 2 then n else fib(n - 1) + fib(n - 2) end\nfib(20)",
    "def fib(n) = if n < 2 then n else fib(n - 1) + fib(n - 2) end\nfib(18)",
    "def f(a: int, b: int): int = a * b\nf(4611686018427387904, 2)",
    "def d(a: int, b: int): int = a / b\n[d(7, -2), d(-7, 2), d(1, 0)]",
    "let xs = (1..1000).to_list()\nprint(xs.map(|x| x * 2).filter(|x| x % 3 == 0).len())\nxs.len()",
    "type P\n  x: int\n  y: int\nend\ndef f(n)\n  let s = 0\n  for i in 0..n\n    let p = P(x: i, y: 2)\n    s += p.x * p.y\n  end\n  s\nend\nf(1000)",
    "def f(n)\n  let out = \"\"\n  for i in 0..n\n    out = \"#{out}#{i % 10}\"\n  end\n  out\nend\nf(50)",
    "def down(n: int): int = if n == 0 then 0 else 1 + down(n - 1) end\ndown(100000)",
];

#[test]
fn same_answers_interpreted_and_compiled() {
    let mut modules = 0;
    for src in PROGRAMS {
        let interpreted = run(src, mode(0)).unwrap();
        assert_eq!(interpreted.modules, 0);
        for jit in [1, 2] {
            let compiled = run(src, mode(jit)).unwrap();
            assert_eq!(
                (&compiled.json, &compiled.printed),
                (&interpreted.json, &interpreted.printed),
                "{src}"
            );
            modules += compiled.modules;
        }
    }
    assert!(modules > PROGRAMS.len() as u32);
}

#[test]
fn compiled_code_is_faster_in_fuel() {
    let src = "def fib(n: int): int = if n < 2 then n else fib(n - 1) + fib(n - 2) end\nfib(22)";
    let interpreted = run(src, mode(0)).unwrap();
    let compiled = run(src, mode(1)).unwrap();
    assert_eq!(compiled.json, interpreted.json);
    assert!(
        compiled.fuel * 10 < interpreted.fuel,
        "{} vs {}",
        compiled.fuel,
        interpreted.fuel
    );
}

#[test]
fn fuel_stops_an_endless_loop_compiled_or_not() {
    let src = "def spin(n: int): int\n  let i = 0\n  while i >= 0\n    i = i + 1 - 1\n  end\n  i\nend\nspin(1)";
    for jit in [0, 1, 2] {
        let limits = Limits {
            fuel: 50_000_000,
            ..mode(jit)
        };
        let e = run(src, limits).unwrap_err();
        assert!(format!("{e:?}").contains("fuel"), "{e:?}");
    }
}

#[test]
fn memory_limit_stops_a_list_growing_forever() {
    let src = "let xs = [1]\nwhile true\n  xs = xs + xs\nend";
    for jit in [0, 1] {
        let limits = Limits {
            memory: 32 << 20,
            ..mode(jit)
        };
        assert!(run(src, limits).is_err());
    }
}
