# Bokoni in wasmtime

The browser playground's module (`playground/wasm`: the interpreter compiled to
WebAssembly, with `wasm-jit`) run by [wasmtime](https://wasmtime.dev) 48, as a native host
such as the platform's kernel would run scripts in a sandbox: fuel, a memory limit, a
stack limit, and nothing else to call.

The module's functions compiled to WebAssembly (`bokoni::wasm_jit`, see
[docs/execution.md](../../docs/execution.md)) are instantiated by `env.bk_compile` in the
same store, against the module's memory and function table (`src/lib.rs`, `compile`):
they use the same fuel and memory as the rest. Components do not share memory, so this
links core modules.

```rust
let host = Host::new(&wasm, Limits::default().stack)?;
let run = host.run("def fib(n: int): int = …\nfib(25)", Limits::default())?;
// run.json: {"value": "75025", "diagnostics": []}; run.printed, run.fuel, run.modules
```

wasmtime runs WebAssembly on the calling thread's stack: the thread needs
`Limits::stack` (4 MiB) more than its own use (the tests use a thread of 16 MiB).

## Use

```sh
cd hosts/wasmtime
cargo test --release                    # builds playground/wasm, then runs the tests
cargo run --release --example measure   # time and fuel, interpreted and compiled
```

## Interpreted or compiled

Measured on hulk (x86_64), a new store per run, the module itself compiled once (150 ms):

| Script | Interpreted | Compiled from the first call | From the 50th (default) |
|---|---:|---:|---:|
| recursive `fib(25)`, typed | 42 ms, 575 M fuel | 2.9 ms, 28 M fuel | 2.7 ms, 28 M fuel |
| 100,000 `type` objects created and read | 20.7 ms, 374 M fuel | 9.3 ms, 143 M fuel | 9.3 ms, 143 M fuel |
| a configuration: records, helpers, a list | 0.10 ms, 0.58 M fuel | 4.6 ms, 11.6 M fuel | 0.11 ms, 0.59 M fuel |

Compiling a function costs a few million units of fuel (the JIT builds its code and
translates it to WebAssembly inside the module) and a module wasmtime compiles with
Cranelift (about a millisecond): for a script evaluated in a fraction of a millisecond,
such as a pipeline's configuration, that is forty times its run. So a function is
compiled only from its 50th call (`Limits::jit_threshold`, `Engine::set_jit_threshold`),
or on its first if it has a loop: a short script stays interpreted, one that computes is
compiled.
