# Roadmap

End goal: a scripting language embedded in Rust, with a JIT (Cranelift) for typed code.

## Target architecture

```
source ─► lexer ─► parser ─► AST ─► type checker ─► typed IR ─┬─► interpreter
                                                               └─► JIT (Cranelift)
```

- **Typed IR**: a simple intermediate representation where every value has a known type.
  A fully typed function (`int`, `float`, `bool`) is compiled to native code; `any` code
  handles generic ("boxed") values checked at run time.
- **The interpreter stays** even with the JIT: it runs `any` code, platforms without a
  JIT (WebAssembly, iOS), and is the reference for testing the JIT (same results).

## Milestones

### v0.1 — Executable core

- [x] Hand-written lexer: bytes, on demand (one token of lookahead), `"#{…}"` interpolation
- [x] Hand-written parser (recursive descent + Pratt for operators)
- [x] Compact AST: expressions in a `Vec`, referenced by index (`ExprId`), 32 bytes each,
      source positions on every node, printable (S-expressions)
- [x] No stack overflow possible (parser and interpreter: `stacker`)
- [x] The parser accepts all of the `syntax.md` syntax (including `type`, `class`, `enum`, `@[…]`)
- [x] Error messages with line, column and source excerpt
- [x] AST interpreter: `let`, `def`, `if`, `case`, `while`, `for`, lambdas, `do` blocks
- [x] Values: `int`, `float`, `str`, `bool`, `nil`, lists, hashes, records, tuples, ranges
- [x] Type annotations checked at run time (`let`, parameters, return)
- [x] `Engine` API for Rust: `eval`, `register_fn` with automatic type conversion
- [x] Command line: run a file, REPL, `--ast` to print the tree
- [x] Error recovery in the parser (all errors in one pass) — done in v0.2

### v0.2 — Static typing

- [x] All errors reported in one pass: the lexer and the parser resume after an error
      (skipping to the `end` of the open constructs, without cascades)
- [x] Name checking (scopes, `let`, constants, use before declaration, fields of
      construction blocks), with suggestions (`len` for `lenght`)
- [x] Type checker: local inference, `any`, `T?`, unions, literal types; only certain
      errors are reported (gradual typing)
- [x] Declarations from previous scripts (REPL) are known to the checker
- [x] `type`, `enum`, `class` at run time; `Server do … end` blocks
- [x] Record literal → nominal type conversion
- [x] Generic types (v0.12), inheritance (v0.8), traits (v0.9), modules (v0.10)

### v0.3 — Typed IR

- [x] Register-based IR in basic blocks (`src/ir.rs`), registers typed `int` / `float` /
      `bool` / `value`; readable output: `bokoni --ir file.bk`
- [x] AST → IR compilation (`src/compile.rs`): capture analysis (captured variables in
      cells, the others in registers), globals in slots resolved at compile time
- [x] Registers typed from annotations and from the types inferred by the checker; a
      dynamic value only enters them through a guard (safe gradual typing)
- [x] Optimizations: direct write into the variable, immediate operands (`i + 1`), fused
      compare-and-branch, short-circuited jumps, merged blocks
- [x] The interpreter runs the IR (`src/vm.rs`); the AST interpreter is removed
- [x] `type`, `class`, `enum`: declarations at the top level of the script

| Program (`cargo run --release --example bench_interp`)    | AST (v0.2) | IR (v0.3) |
|-----------------------------------------------------------|-----------:|----------:|
| recursive `fib(25)`                                        | 57 ms      | 28 ms     |
| `while` loop, 1 M iterations                               | 115 ms     | 29 ms     |
| `for` over a range, 1 M iterations                         | 56 ms      | 20 ms     |
| lists: `map` / `filter` / `sum` (100 k)                    | 19 ms      | 17 ms     |
| closures and counter (300 k calls)                         | 36 ms      | 27 ms     |
| objects: construction and fields (100 k)                   | 30 ms      | 17 ms     |
| strings: interpolation (100 k)                             | 16 ms      | 12 ms     |

Programs dominated by built-in functions (lists) or by calls gain little: that is the
JIT's job (direct calls, unboxed values).

### v0.4 — Cranelift JIT

- [x] Native compilation (`src/jit.rs`, Cranelift 0.136) of functions whose parameters and
      return are annotated `int` / `float` / `bool` and whose body only computes,
      compares, branches and calls other compilable functions (recursion included)
- [x] IR analysis: `value` registers are typed (result of a call to a `: int`
      function…); otherwise the function stays interpreted
- [x] Same results and same errors as the interpreter: overflow, Euclidean division,
      division by zero, call depth, call stack (verified by equivalence tests)
- [x] Direct native → native calls; redefining a global function invalidates the code
      that calls it, which is recompiled when needed
- [x] Compilation on the first call; a function called only once but that loops
      benefits too
- [x] Insufficient native stack (very deep recursion): resumed in the interpreter,
      whose stack grows on the heap (v0.14)
- [x] Cargo feature `jit` (on by default); `Engine::set_jit(false)`, `bokoni --no-jit`
- [ ] Compile more: methods, lists, strings, calls into the interpreter from native code
      (boxed values)

| Program (`cargo run --release --example bench_interp`)    | Interpreter | JIT    |
|-----------------------------------------------------------|------------:|-------:|
| recursive `fib(25)`                                        | 28 ms       | 0.5 ms |
| `while` loop, 1 M iterations                               | 29 ms       | 1.2 ms |
| `for` over a range, 1 M iterations                         | 20 ms       | 0.5 ms |
| `float`: harmonic series, 1 M iterations                   | 33 ms       | 0.9 ms |
| lists, closures, objects, strings                          | unchanged (interpreted) | |

Times include compilation.

### v0.5 — JIT for untyped code and objects

- [x] Baseline compilation of any function (second call, or first if it has a loop):
      `Value` registers in memory (fixed `repr(C, u8)` layout), constants, copies,
      `int` / `float` arithmetic and comparisons, branches and `for` in native code;
      everything else calls the interpreter's instruction (`Runtime::step`), so the
      semantics are the same by construction
- [x] Lighter calls (interpreter and JIT): registers cleared in one go, `map` / `filter` /
      `each` / `reduce` without allocation or generic binding, method names without
      allocation, direct construction of a `type` when all its fields are given
- [x] Equivalence tests on untyped code (closures, lists, objects, errors), compiled from
      the first call
- [x] Dedicated helper functions instead of the interpreter for frequent instructions:
      captured variables and cells, globals, calls; type guards in native code
- [x] Field access with a per-site cache (last type seen, field position)
- [x] Construction of a `type` directly from the call's registers; stack checked every 8
      call levels; JIT watchers stored in the globals' slots
- Tried and dropped: object fields stored inline (`SmallVec`) — not faster, small
  allocations are cheap
- [x] Typed fields in the IR: `p.x` of an **annotated** field of a `type` yields an
      `int` / `float` / `bool` register (`FieldAs` instruction); a field whose type is only
      inferred (`port = 80`) is not checked at run time, so it stays generic
- [x] Typed JIT: parameters of a declared type (`def dot(a: V, b: V): float`), passed by
      pointer; the object's type is checked once on entry, fields are read directly at
      their offset; passed as is to native functions with the same annotation
- [x] Unverified assumption (redeclared type): resumed in the interpreter (v0.14);
      redeclaring the type recompiles the function for the new layout
- [x] Liveness analysis: a `nil` that is never read (value of a `while`) no longer
      prevents typed compilation
- [x] Typed JIT: lists as parameters (`[int]`, `[float]`, `[P]`…) — `for x in xs`,
      `xs[i]` (negative index included), `xs.len()`, passing to native functions; edge
      cases (index out of bounds…): the interpreter takes over the call and produces the
      exact error
- [x] Typed JIT: creating objects (`V(x: …, y: …)`) and list literals, returning objects
      and lists (`def add(a: V, b: V): V`). Owning registers: a slot in the native frame,
      released on every exit (return, error, bail-out); borrowed registers (parameters,
      elements of a received list): a plain pointer. Tested with reference counts after
      thousands of calls
- [x] Flow-sensitive register typing (one type per register and per point, one native
      variable per pair): reusing a temporary for different types no longer prevents
      compilation (replaces the dead-`nil` analysis)
- [x] Direct call to a function already compiled to baseline code; string interpolation
      with a reused buffer; built-in methods called directly
- Deliberately not done: typing lambda parameters from the list (`xs.map(|x| …)`) — an
  `[int]` list may receive another value from untyped code, and the remaining cost is the
  call, not the computation

| Program                                  | v0.4    | now     | gain |
|------------------------------------------|--------:|--------:|-----:|
| lists: map / filter / sum (100 k)        | 16.9 ms | 8.0 ms  | ×2.1 |
| closures and counter (300 k calls)       | 27.4 ms | 13.6 ms | ×2.0 |
| objects: construction and fields (100 k) | 17.9 ms | 9.2 ms  | ×1.9 |
| strings: interpolation (100 k)           | 11.2 ms | 6.0 ms  | ×1.9 |
| typed objects: dot product (1 M)         | 172 ms  | 3.2 ms  | ×54  |
| typed loop over 100 k objects (`wsum`)   | ~6.4 ms | 0.47 ms | ×14  |
| vectors created in typed code (1 M)      | 409 ms  | 73 ms   | ×5.6 |
| interpreted `fib(25)` (`--no-jit`)       | 28 ms   | 18 ms   | ×1.6 |

### v0.6 — Annotations

- [x] Arguments checked before running (like a `type` construction), including for
      annotations declared in a previous script
- [x] Kept on every target (type, class, enum, variant, field, method, function,
      parameter): each `@[…]` is compiled into a small function evaluated when read
- [x] Rust API: `Engine::type_info` / `function_info` / `enum_info`, descriptions with
      `annotation(name)`, `annotations(name)`, `doc()`, `deprecated()`,
      `Annotation::get::<T>(field)` (`reflect` module)
- [x] `Deprecated` and `Doc` declared by a prelude; a warning on every use of an
      `@[Deprecated]` declaration (`ErrorKind::Warning`, `Engine::take_warnings`,
      `Engine::set_warning_handler`, printed by the command line)
- [x] `@[Deprecated]` parameter: a warning when the argument is passed (by position or
      by name)

### v0.7 — Error handling

Modeled on Crystal (see `syntax.md`, *Error handling*).

- [x] `raise "message"` (an `Error`, declared by the prelude) or `raise obj` (any
      object); `begin` / `rescue e: T` / `else` / `ensure`, also directly on the body of
      a `def` or a `do` block
- [x] Runtime errors (division by zero, index, too many nested calls…) rescued as an
      `Error`; an error no clause matches goes on up unchanged (position and call stack)
- [x] `ensure` on every way out: end of the body, `rescue`, `return` / `break` / `next`
      (its code is repeated before each jump), error going on up
- [x] Zero cost when nothing is raised: each IR block knows its handler (exception
      table); the interpreter only looks at it when an instruction fails. Functions
      with a `rescue` / `ensure` stay interpreted; errors raised in compiled code
      (typed or baseline) reach them as before
- [x] Static checks: type of the `rescue` variable, `raise` of a value that is neither a
      string nor an object
- [x] Custom errors inherit from `Error` (see v0.8)
- [x] `?` variants that return `nil` instead of raising: `to_i?`, `to_f?`, `xs[i]?`
      (the other built-in methods already return `nil`); names of functions and
      methods may end with `?`

### v0.8 — Single inheritance

- [x] `class B < A` (classes only): fields of the parent first in the objects, so its
      methods keep reading them by index; methods inherited, `init` included, and
      redefinable
- [x] `super(args)` in any method (and its `do` blocks): the parent's version
- [x] Subtypes: checked statically (`[Animal]` accepts a `Dog`) and at run time
      (annotations, `rescue e: Error`)
- [x] `Error` is a class of the prelude: custom errors inherit from it
- [x] Parent from a previous script; `TypeDesc::parent` in the Rust API
- [x] A type or enum declared twice in a block is an error (the checker used to panic,
      found by the fuzzer)
- [x] Abstract classes and methods, traits: see v0.9

### v0.9 — Traits and abstract classes

Decided: two separate notions, unlike Crystal's single `module` — `module` for
namespaces, `trait` for shared behavior.

- [x] `trait`: required methods (`abstract def`, no body) and provided methods;
      `include` in a class, several allowed; precedence: own, traits, parent
- [x] Nominal: a trait is a type (`x: Greet`, `[Greet]`, `rescue e: Retryable`),
      checked statically and at run time
- [x] A missing required method is reported before running (and when declaring the
      class, without static checks); signatures of `abstract def`s kept at run time
      for later scripts
- [x] `abstract class`: not instantiable, may declare `abstract def`s
- [x] `TypeDesc::traits`, `TypeDesc::is_abstract`, `DeclKind::Trait`
- [ ] A trait that includes another one; generic traits (`Comparable(T)`)

### v0.10 — Modules and files

- [x] `module M … end`: constants, functions, types, classes, traits, enums, nested
      modules; `M::Name` for a member, `M.f(…)` for a function (Crystal's notation)
- [x] Short names inside a module (its members first); qualified names everywhere a
      type is expected (`x: Http::Request`, `class A < Http::Failure`,
      `include Http::Named`, `rescue e: Http::Failure`)
- [x] Members are globals named `M::name`: no runtime cost, compiled code calls
      module functions directly
- [x] Module bodies run with the declarations, before the block's statements; a
      module can be reopened (also by a later script)
- [x] `require "file"`: loaded once, before the script is checked (its declarations
      known as a previous script's); nested requires, cycles reported; files provided
      by the host (`Engine::set_loader`), from disk for the `bokoni` command

### v0.11 — Visibility

- [x] `private def` / `protected def` in a class, a trait or a `type` (Ruby's and
      Crystal's rules: `private` on `self` only, `protected` from the methods of the
      declaring type and of its subclasses), getters and setters included
- [x] Checked before running, and again when running (static checks off): each call
      site carries where it is called from, the check costs nothing for a public
      method
- [x] `private` declarations of a module: used only inside it (and its nested
      modules), checked when compiling
- [x] `FnDesc::visibility`

### v0.12 — Generics

- [x] `forall T, U` on functions and methods (Crystal's syntax); type arguments found
      from the arguments, then from the expected type
- [x] Generic `class`, `type` and `trait`: `class Box(T)`, `Box(int)` in annotations,
      found from the values at construction
- [x] Generic parents and traits: `class IntBox < Box(int)`, `include Comparable(T)`
- [x] Bounds: `forall T: Comparable(T)`, `class Sorted(T: Comparable(T))`, checked at
      calls and in annotations; a bounded `T` has the bound's methods
- [x] Checked statically only, erased at run time (no cost); declarations of previous
      scripts known with their parameters; `TypeDesc::params`
- [x] Fix: calling a parameter of function type (`f: (int): int`) crashed the checker
- [x] Typed JIT for `forall` functions: a copy per type of the arguments
      (`T = int`, `[float]`, objects), called directly by native code or by the
      interpreter; ×37 on a generic `max` in a hot loop
- [x] Methods of generic classes (fields of type `T`), compiled per type argument
      (v0.13): `T` found from the fields of `self` and the arguments

### v0.13 — Methods in the typed JIT

- [x] Methods compiled like typed functions, for the class that declares it (its
      subclasses share its first fields): `self` as a pointer, annotated `int` /
      `float` / `bool` fields read and written in place
- [x] `obj.m()` / `self.m()` from native code: direct call, checked against a
      redefinition by the object's class (abandoned to the interpreter otherwise);
      ×49 for a typed loop calling a method
- [x] No abandon after a side effect: a method that writes fields contains only
      arithmetic, branches and fields (their types checked on entry), and native code
      does not call it
- [x] Interpreter → native code called directly when the arguments are of its native
      types (no frame, no conversion): a method called in a loop of the script ×1.5 →
      ×5.8 against the interpreter, a typed function 56 → 28 ns per call
- [x] Type arguments kept in compiled annotations (`a: Acc(int)`), ignored by the
      run-time checks: typed native code calls the copy of the methods for them
      directly (checked on entry of the copy); 0.13 s → 0.02 s on a typed loop

### v0.14 — JIT without abandons, side effects in native code

- [x] Resumption in the interpreter where an assumption does not hold, instead of
      redoing the whole call: the live registers (liveness analysis) are written to a
      buffer, `bk_resume` rebuilds the interpreter's registers (`for` iterators
      included) and runs the rest of the function from that instruction; a typed copy
      resumes as the function written (`Spec::origin`); a result of another type than
      the native one is passed to the native caller, which resumes after the call
- [ ] Lists modified in native code (`xs << x`, `xs[i] = v`, list fields)
- [ ] Native calls to methods that write fields
- [ ] Inlining of small functions and getters
- [ ] Strings in native code

### Next

In suggested order:

1. **Crystal-style macros** (v3 of `syntax.md`): hygiene and AST exposure still open;
   benefit from generics and modules.

No design decision needed:

- Free local functions that capture themselves (reference cycle).
- `for` without copying the list it walks.
- Typed registers for globals, not only locals.

## Decisions that commit the JIT

| Topic                    | v0.1 choice                                         |
|--------------------------|-----------------------------------------------------|
| Value representation     | `enum Value` (24 bytes), basic types without allocation; NaN-boxing (8 bytes) to be studied for the JIT |
| Memory                   | reference counting (`Rc`), no garbage collector     |
| Rust functions           | single convention `Fn(&[Value]) -> Result<Value, String>` |
| Names                    | interning: `Symbol(u32)`, O(1) comparisons          |
| Positions                | `Span { start: u32, end: u32 }` in bytes            |
| AST                      | compact, nodes referenced by index: a closure only keeps `Rc<Script>` + index |

## Parser performance

`cargo run --release --example bench_parser` compares the parser with an equivalent
[`peg`](https://crates.io/crates/peg) grammar (common subset: `let`, assignment,
operators, calls, fields, indexing, lists, strings). The peg version does *less* work
(no positions, no interning).

| Measure                                 | bokoni        | peg          |
|-----------------------------------------|---------------|--------------|
| 150,000-line file (9 MB)                | ~60 ms (150 MB/s) | ~93 ms (100 MB/s) |
| Small script (60 lines) × 20,000        | ~400 ms       | ~820 ms      |
| Allocations per line                    | 0.34          | 14.7         |

What mattered, as measured with the profiler: the amount of memory touched. Moving from
one `Box` per node to the compact AST halved parsing time (120 ms → 60 ms); the other
optimizations (on-demand lexer, interning of short names) only brought a few %.

## Robustness

- Fuzzing (`tests/fuzz.rs`): random text, mutations of valid programs, generated
  programs; interpreter, JIT and baseline code, with and without static checking: no
  panic, no hang, same results. Long campaigns:
  `BK_FUZZ=50 BK_FUZZ_SEED=3 cargo test --release --test fuzz`
- No recursion on the stack for oversized inputs: nested source (parser nesting limit,
  long operator chains), data nested hundreds of thousands of times (iterative release
  of `Value`, comparison on a stack that grows on the heap)
- Oversized allocations are refused with an error rather than aborting the process:
  string repeated beyond 2 GiB, range converted to a list beyond 2^27 elements; `map` /
  `filter` / `each` / `reduce` / `find` walk a range without copying it
- Cyclic and heavily shared structures: `[…]` displayed for a cycle, at most one million
  nodes displayed; equality by bisimulation (a pair of containers seen again counts as
  equal): terminates, in time bounded by the number of distinct pairs
- Bounded error messages: only a 120-character window of a long line is displayed
- `Value` implements `Drop` (iterative release): from Rust, bind its fields by reference
  (`if let Value::List(ref l) = v`) or clone them; measured cost of 3 to 8% on programs
  that allocate a lot

## Known limitations (v0.1)

- Reference cycles: a local function that captures itself is never freed (global
  functions are, when the `Engine` is dropped).
- A `for` loop copies the list it walks before starting (safe, but costly for a very
  large list).
- A typed register is only used for a local variable; globals (top level) stay dynamic.
