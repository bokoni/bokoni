# How a script runs

Three ways to turn a `.bk` file into work done by the processor. The first two exist
today; the third is a possible backend, not started.

| | Path | Who produces the machine code |
|---|---|---|
| **1. Interpreter** | `.bk` → IR → run instruction by instruction by the interpreter | nobody, for the script: only the interpreter (Rust) is compiled |
| **2. Cranelift JIT** | `.bk` → IR → Cranelift → machine code | Bokoni itself, in its own process |
| **3. WebAssembly backend** | `.bk` → IR → `.wasm` bytes (`wasm-encoder`) → instantiated by the host | the host: the browser's engine (V8, SpiderMonkey) or wasmtime |

## 1. Interpreter

Always present, on every target, and the reference: the JIT is tested against it (same
values, same errors). It is what runs in the browser playground: the interpreter is
compiled from Rust to WebAssembly once, the browser turns that into machine code, and the
script is then interpreted by it. `fib(25)`: about 28 ms natively, 40 ms in the browser.

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
(`--no-default-features`), Bokoni runs there with the interpreter only.

## 3. WebAssembly backend (possible, not started)

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
