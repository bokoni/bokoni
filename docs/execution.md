# How a script runs

Three ways to turn a `.bk` file into work done by the processor. The third, for Bokoni
itself running in WebAssembly, compiles typed functions only so far (phase 1 below).

| | Path | Who produces the machine code |
|---|---|---|
| **1. Interpreter** | `.bk` → IR → run instruction by instruction by the interpreter | nobody, for the script: only the interpreter (Rust) is compiled |
| **2. Cranelift JIT** | `.bk` → IR → Cranelift → machine code | Bokoni itself, in its own process |
| **3. WebAssembly backend** | `.bk` → IR → `.wasm` bytes (`wasm-encoder`) → instantiated by the host | the host: the browser's engine (V8, SpiderMonkey) or wasmtime |

## 1. Interpreter

Always present, on every target, and the reference: the JIT is tested against it (same
values, same errors). In the browser playground, the interpreter is compiled from Rust to
WebAssembly once, the browser turns that into machine code, and the script is then
interpreted by it (apart from its typed functions, section 3). `fib(25)`: about 28 ms
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

## 3. WebAssembly backend (`wasm-jit` feature; typed functions)

The way to get compiled code inside WebAssembly: Bokoni writes a small `.wasm` module for
the script's functions (with [`wasm-encoder`](https://crates.io/crates/wasm-encoder),
which only writes the binary format) and hands its bytes to the host, which compiles it to
machine code like any other module and links it to the interpreter's memory and functions:

- in a browser, JavaScript calls `WebAssembly.instantiate(bytes, imports)`, the imports
  being the interpreter's memory and helper functions;
- in a wasmtime host, an import such as `compile(bytes)`; the host instantiates the
  module under the same fuel and memory limits. Components do not share memory, so the
  host links the generated module at the level of core modules.

The same two tiers as the JIT would apply:

- typed code maps almost one to one onto WebAssembly (`int` → `i64`, `float` → `f64`,
  locals, `loop`, `call`): the realistic part;
- baseline code would keep calling the interpreter's code for anything beyond scalars;
  compiling values, strings, lists, closures and objects themselves means writing the
  runtime again in WebAssembly (linear memory or wasm GC): a much larger project.

Worth it only for scripts that compute: a configuration evaluated in about a millisecond
gains nothing over the interpreter.

### Phase 1: typed functions ([`src/wasm_jit.rs`](../src/wasm_jit.rs))

What the Cranelift JIT compiled in its first version (v0.4), the same way: a function
whose parameters and result are annotated `int`, `float` or `bool` (or a copy typed for
its arguments, for parameters without annotations), whose body only computes with such
values, compares, branches, loops and calls such functions (recursion included). Any
other function stays interpreted, and so does a function calling one, since the generated
code does not call the interpreter yet.

- The embedder gives a compiler (`bokoni::wasm_jit::set_compiler`): a function that
  instantiates a module's bytes against the program's memory (`env.memory`) and function
  table (`env.table`), appends its exports `f0`, `f1`… to the table and returns the index
  of the first one. The playground's is `bk_compile` in
  [`playground/web/src/bokoni.ts`](../playground/web/src/bokoni.ts); the crate is built
  with `-C link-arg=--export-table -C link-arg=--growable-table`. Without a compiler
  (or outside WebAssembly), everything is interpreted.
- The VM sees it as it sees the Cranelift JIT (`Jit::compile`, `Jit::call`,
  `JitState::Native`), so the same paths call it.
- One module per compilation: the function called and those it calls that are not
  compiled yet, calling each other directly; one compiled earlier is called through the
  table. Each function has an internal function `(ctx, params…) -> (status, result)` and
  an entry point for Rust with the Cranelift JIT's C signature.
- Control flow: a loop over a `br_table` on the IR block's number, a jump to the next
  block falling through (a structured rewrite, "Beyond Relooper", is for later).
- The interpreter's rules: overflow of `+`, `-`, `*` and negation, Euclidean `/` and `%`,
  division by zero, `i64::MIN / -1`, the call depth, the same messages and call stack.
- Redefining a global function invalidates the code calling it, directly or through
  other compiled functions.

Measured in Chrome (V8), the interpreter against the compiled code of the playground:

| Program | Interpreter | Compiled |
|---|---:|---:|
| recursive `fib(25)` | 26.6 ms | 0.5 ms |
| recursive `fib(30)` | 290 ms | 4.2 ms |
| `float`: harmonic series, 1 M iterations | 28.6 ms | 0.9 ms |
| `for` over a range, 1 M iterations | 31.5 ms | 1.6 ms |

The natively compiled Cranelift JIT takes 0.5 ms for `fib(25)` too.

Tested in [`playground/web/test/jit.test.ts`](../playground/web/test/jit.test.ts), each
program run interpreted, then compiled, the answers compared (value, errors, positions,
call stack): edge cases of arithmetic, every example of the playground, and random
programs of typed functions ([`generate.ts`](../playground/web/test/generate.ts); 400 in
CI, `BK_GEN=3000 bun test` for more).

Next: a result without annotation (inferred), calls into the interpreter from the
generated code (for what it does not compile), then lists, strings and objects through
the interpreter's helpers, and a structured control flow.

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
