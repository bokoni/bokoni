# How a script runs

Three ways to turn a `.bk` file into work done by the processor. The third is for Bokoni
itself running in WebAssembly (the browser playground), where the second cannot.

| | Path | Who produces the machine code |
|---|---|---|
| **1. Interpreter** | `.bk` → IR → run instruction by instruction by the interpreter | nobody, for the script: only the interpreter (Rust) is compiled |
| **2. Cranelift JIT** | `.bk` → IR → Cranelift → machine code | Bokoni itself, in its own process |
| **3. WebAssembly backend** | `.bk` → IR → `.wasm` bytes (`wasm-encoder`) → instantiated by the host | the host: the browser's engine (V8, SpiderMonkey) or wasmtime |

## 1. Interpreter

Always present, on every target, and the reference: the JIT is tested against it (same
values, same errors). In the browser playground, the interpreter is compiled from Rust to
WebAssembly once, the browser turns that into machine code, and the script is then
interpreted by it, unless its functions are compiled (section 3). `fib(25)`: about 28 ms
natively, 27 ms in Chrome.

## 2. Cranelift JIT (`jit` feature, on by default)

Bokoni writes machine code into memory and jumps to it. Two tiers:

- **Typed code**: a function whose parameters and return are annotated (`int`, `float`,
  `bool`, a `type`, a list of them…) is compiled on its first call into straight native
  code: values in machine registers, fields read at their offset. 14 to 54 times faster
  than the interpreter (`fib(25)`: 0.5 ms).
- **Baseline code**: any other function, compiled on its second call (or its first if it
  loops). Registers stay `Value`s in memory; native code does the simple cases
  (constants, copies, arithmetic, comparisons, branches) and calls the interpreter's own
  code (`Runtime::step`) for the rest, so the semantics are the same by construction.
  About twice as fast as the interpreter.

Where an assumption of the native code does not hold, the interpreter takes over at that
point.

**Not possible inside WebAssembly** (a browser, a wasm module of a host such as
wasmtime): a module's code is fixed when it is instantiated; it cannot make memory
executable nor jump into bytes it has just written. Built without the JIT
(`--no-default-features`), Bokoni runs there with the interpreter, and with the
WebAssembly backend if its host compiles modules for it (section 3).

## 3. WebAssembly backend (`wasm-jit` feature)

The way to get compiled code inside WebAssembly ([`src/wasm_jit.rs`](../src/wasm_jit.rs)):
Bokoni writes a small `.wasm` module for a function (with
[`wasm-encoder`](https://crates.io/crates/wasm-encoder), which only writes the binary
format) and hands its bytes to the host, which compiles it to machine code like any
other module and links it to the program's memory and function table. Rust then calls
the code through a function pointer, which in WebAssembly is an index in that table.

- The embedder gives a compiler (`bokoni::wasm_jit::set_compiler`): a function that
  instantiates a module's bytes against the program's memory (`env.memory`) and function
  table (`env.table`), appends its exports `f0`, `f1`… to the table and returns the index
  of the first one. The playground's is `bk_compile` in
  [`playground/web/src/bokoni.ts`](../playground/web/src/bokoni.ts); the crate is built
  with `-C link-arg=--export-table -C link-arg=--growable-table`. Without a compiler (or
  outside WebAssembly), everything is interpreted.
- In a wasmtime host, the compiler would be an import of the same kind; the host
  instantiates the module under the same fuel and memory limits. Components do not share
  memory, so the host links the generated module at the level of core modules.
- The VM sees it as it sees the Cranelift JIT (`Jit::compile`, `Jit::call`,
  `Jit::compile_dynamic`, `JitState::Native`), so the same paths call it.

The same two tiers as the Cranelift JIT:

- **Typed code.** A function whose parameters and result are `int`, `float` or `bool`,
  whose body only computes with such values, compares, branches, loops and calls such
  functions (recursion included), as the Cranelift JIT compiled in its first version
  (v0.4). Annotations may be left out: parameters without one get a copy typed for the
  arguments, and a result without one is inferred. For that, `int`, then `float`, then
  `bool` is assumed, recursive calls included, and kept if every `return` gives exactly
  that type (`def fib(n) = …` is compiled). The values live in WebAssembly locals. The
  functions compiled together call each other directly, one compiled earlier through the
  table; each has an internal function `(ctx, params…) -> (status, result)` and an entry
  point for Rust with the Cranelift JIT's C signature. The interpreter's rules: overflow
  of `+`, `-`, `*` and negation, Euclidean `/` and `%`, division by zero,
  `i64::MIN / -1`, the call depth, the same messages and call stack. Redefining a global
  function invalidates the code calling it, directly or through other compiled
  functions.
- **Baseline code.** Any other function without `rescue` / `ensure`, as the Cranelift
  JIT's, with the same helpers ([`src/baseline.rs`](../src/baseline.rs)): its registers
  stay `Value`s in the call's window, the code does the simple cases itself (constants,
  copies of scalar values, arithmetic and comparisons of two integers or two floats,
  branches) and calls the interpreter's code for everything else, lists, strings,
  objects, closures and calls included; the semantics are the same by construction.

**Control flow**, for both: structured, as a compiler from a structured language would
write it, from the IR's blocks (N. Ramsey, "Beyond Relooper", 2022): a `loop` for each
loop header, a `block` before each block reached by several forward edges, the others
written inline where they are branched to; along the dominator tree, in reverse
postorder. A graph that is not reducible (none from Bokoni's code: checked on every
program of the tests) gets a loop over a `br_table` on the block number instead.

Measured in Chrome (V8), the interpreter against the compiled code of the playground
(the best of three runs):

| Program | Interpreter | Compiled |
|---|---:|---:|
| recursive `fib(25)` | 26.5 ms | 0.3 ms |
| recursive `fib(30)` | 295 ms | 2.8 ms |
| the same, without annotations | 315 ms | 2.8 ms |
| `float`: harmonic series, 1 M iterations | 25.3 ms | 0.8 ms |
| sum of a list of 200,000 `int`, twice (baseline) | 14.5 ms | 6.4 ms |
| `map` / `filter` over 100,000 elements | 14.1 ms | 10.8 ms |
| `type` objects created and read, 100,000 | 14.3 ms | 9.1 ms |
| string interpolation, 100,000 | 5.6 ms | 4.1 ms |
| a closure called 300,000 times | 26.0 ms | 15.8 ms |

The natively compiled Cranelift JIT takes 0.5 ms for `fib(25)`.

Tested under Bun ([`playground/web/test/`](../playground/web/test)), each program run
interpreted, compiled as usual, and with every function compiled from its first call
(baseline code for the untyped ones), the answers compared (value, errors, positions,
call stack, what it printed):

- `jit.test.ts`: edge cases of arithmetic, redefinitions, every example of the
  playground, and random programs of typed functions, annotations left out at times
  ([`generate.ts`](../playground/web/test/generate.ts); 400 in CI, `BK_GEN=3000 bun
  test` for more);
- `corpus.test.ts`: every Bokoni program of the Rust tests (the string literals of
  `tests/*.rs`) and of [`syntax.md`](syntax.md), nearly 900, also with the dispatch loop
  instead of structured control flow.

Not done: the typed tier stops at `int`, `float` and `bool`. The Cranelift JIT's typed
code also reads lists, strings and objects natively and resumes in the interpreter when
an assumption fails (`DeoptSite`); here such functions run as baseline code (1.3 to 2.3
times faster than the interpreter rather than 5 to 50).

### Phase 0: done ([`playground/jit-spike/`](../playground/jit-spike/src/lib.rs))

The mechanism the backend rests on works, under Bun (JavaScriptCore) and in Chrome (V8).
Rust (built with `--export-table --growable-table`) writes a module with `wasm-encoder`.
The host instantiates it against Rust's memory and function table and appends its
functions to that table. Rust then calls them through ordinary `extern "C" fn` pointers,
which on wasm32 are indices in that table. The generated code:

- reads Rust's memory (a `Vec<i64>` summed);
- calls back into Rust with `call_indirect` on a function pointer it is given;
- returns a `#[repr(C)]` structure through the pointer the C ABI passes first, which is
  how the JIT's `(status, result)` would come back.

Speed of a recursive `fib`, without overflow checks:

| | generated | rustc |
|---|---:|---:|
| Chrome, `fib(25)` | 0.20 ms | 0.20 ms |
| Chrome, `fib(30)` | 1.2 ms | 1.0 ms |
| Bun, `fib(25)` | 0.41 ms | 0.17 ms |
| Bun, `fib(30)` | 2.0 ms | 1.5 ms |

For comparison, the playground's interpreter takes about 40 ms for `fib(25)`.

`cd playground/jit-spike && cargo build --release --target wasm32-unknown-unknown && bun
host.js`; or serve the directory and open `index.html`. The module is compiled
synchronously (`new WebAssembly.Module`), which a browser allows for small modules on the
main thread and for any size in a worker, where the playground runs.
