# How a script runs

Three ways to turn a `.bk` file into work done by the processor. The third is for Bokoni
itself running in WebAssembly (the browser playground), where the second cannot.

| | Path | Who produces the machine code |
|---|---|---|
| **1. Interpreter** | `.bk` → IR → run instruction by instruction by the interpreter | nobody, for the script: only the interpreter (Rust) is compiled |
| **2. Cranelift JIT** | `.bk` → IR → Cranelift → machine code | Bokoni itself, in its own process |
| **3. WebAssembly backend** | `.bk` → IR → Cranelift's IR → `.wasm` bytes (`wasm-encoder`) → instantiated by the host | the host: the browser's engine (V8, SpiderMonkey) or wasmtime |

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
  than the interpreter (`fib(25)`: 0.5 ms). Parameters without annotation get a copy
  typed for the arguments; a result without one is guessed (`guess_ret`: the type every
  `return` gives, the calls assumed to return their own guess), `def fib(n) = …`
  included. A wrong guess costs a resumption, as a typed copy returning another type
  does (status 2).
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

The Cranelift JIT itself, inside WebAssembly ([`src/wasm_jit.rs`](../src/wasm_jit.rs)).
The JIT builds its functions in Cranelift's IR (CLIF), as natively, both tiers with
everything they do (typed code on lists, strings and objects, its resumptions in the
interpreter, baseline code); only the last step differs. In place of Cranelift's machine
code backend (`cranelift_jit::JITModule`), `WasmModule` translates each function from
CLIF into WebAssembly (`wasm-encoder`, which only writes the binary format) and hands the
module to the host, which compiles it to machine code and links it to the program's
memory and function table. Rust calls the code through a function pointer, which in
WebAssembly is an index in that table.

- The embedder gives a compiler (`bokoni::wasm_jit::set_compiler`): a function that
  instantiates a module's bytes against the program's memory (`env.memory`) and function
  table (`env.table`), appends its exports `f0`, `f1`… to the table and returns the index
  of the first one. The playground's is `bk_compile` in
  [`playground/web/src/bokoni.ts`](../playground/web/src/bokoni.ts); the crate is built
  with `-C link-arg=--export-table -C link-arg=--growable-table`. Without a compiler (or
  outside WebAssembly), everything is interpreted.
- In a wasmtime host ([`hosts/wasmtime`](../hosts/wasmtime/README.md)), the same import:
  the host instantiates each module in the same store, under the same fuel and memory
  limits; components do not share memory, so it links core modules. Compiling a function
  costs a few million units of fuel and about a millisecond: `fib(25)` goes from 42 ms to
  2.9 ms, but a configuration evaluated in 0.1 ms takes 4 ms compiled, so such scripts
  run interpreted there.

The translation:

- CLIF values become WebAssembly locals (an `i8` in an `i32`, its low byte); a block's
  parameters are locals assigned on the edges into it. The JIT handles addresses as `i64`
  (as on a 64-bit machine), 32 bits in WebAssembly's memory: converted at each memory
  access, and at each call of a Rust helper, whose WebAssembly parameters come from its
  Rust type (`sig_of`). The JIT reads and writes addresses in memory at their real width
  (`Addresses`).
- Stack slots live in an area of the program's memory used as a stack, a function's
  frame taken there on its first use (a typed function's slots mostly serve its
  resumptions, rarely taken). Values nothing uses are not computed, stores into slots
  nothing reads not made.
- Control flow is structured, as a compiler from a structured language would write it
  (N. Ramsey, "Beyond Relooper", 2022): a `loop` for each loop header, a `block` before
  each block reached by several forward edges, the others written inline where they are
  branched to, along the dominator tree, in reverse postorder. A graph that is not
  reducible (none from Bokoni's code) gets a loop over a `br_table` instead.
- Each function has an adapter for Rust (its addresses as `i32`).
- An instruction the translation does not know leaves the function interpreted (none in
  the tests).

Measured in Chrome (V8), the interpreter against the compiled code of the playground
(the best of three runs, compilation included):

| Program | Interpreter | Compiled |
|---|---:|---:|
| recursive `fib(25)` | 27.1 ms | 1.3 ms |
| recursive `fib(30)` | 301 ms | 4.4 ms |
| the same, without annotations | 314 ms | 4.7 ms |
| `float`: harmonic series, 1 M iterations | 27.8 ms | 0.9 ms |
| sum of a list of 200,000 `int`, twice | 13.7 ms | 2.6 ms |
| `type` objects created and read, 100,000 | 14.8 ms | 4.9 ms |
| dot product of two `type` objects, 1 M | 230 ms | 28 ms |
| `map` / `filter` over 100,000 elements | 14.3 ms | 11.4 ms |
| string interpolation, 100,000 | 5.7 ms | 5.0 ms |
| a closure called 300,000 times | 29.3 ms | 30.2 ms |

The closure gains nothing natively either (21 ms → 19 ms): a call of a closure value goes
through the interpreter.

Tested under Bun ([`playground/web/test/`](../playground/web/test)), each program run
interpreted, compiled as usual, and with every function compiled from its first call,
the answers compared (value, errors, positions, call stack, what it printed):

- `jit.test.ts`: edge cases of arithmetic, redefinitions, every example of the
  playground, and random programs of typed functions, annotations left out at times
  ([`generate.ts`](../playground/web/test/generate.ts); 400 in CI, `BK_GEN=10000 bun
  test` for more);
- `corpus.test.ts`: every Bokoni program of the Rust tests (the string literals of
  `tests/*.rs`) and of [`syntax.md`](syntax.md), nearly 900, also with the dispatch loop
  instead of structured control flow; none left untranslated.

Running the JIT in WebAssembly found two bugs of the native one, fixed for both: a call
the analysis left to the interpreter (an argument of any type) was still inlined, which
made Cranelift panic; redefining a function did not reach the callers of its callers
(their code kept calling the old one).

### Phase 0 ([`playground/jit-spike/`](../playground/jit-spike/src/lib.rs))

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
