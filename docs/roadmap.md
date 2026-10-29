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
- [x] Compile more: lists (v0.5, v0.14), methods (v0.13), strings and calls into the
      interpreter from native code, with boxed values (v0.15)

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
- [x] Generic traits (`Comparable(T)`, v0.12)
- [ ] A trait that includes another one

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
- [x] Lists modified in native code: `xs << x`, `xs[i] = v`, `[]` given its type by
      an annotation (`let out: [int] = []`), list fields of `self` read and assigned;
      in a function that modifies lists or calls another, the elements read are copied
      and a `for` goes over a copy of the list (as the interpreter does); building a
      list in a typed loop ×5.9
- [x] Native calls to methods that write fields (and no restriction on what such a
      method contains): the type of a field is checked on every access, a method
      called may change it; a typed loop calling `c.add(k)`: 0.11 s → 0.02 s
- [x] Inlining of small functions and getters (one block, arithmetic and fields of
      `self`, same script): their body in the caller, an error keeps their frame in
      the call stack; a method's class checked by a cache per call site
      (`bk_fields_cached`, resumed in the interpreter if it redefines the method); an
      inlined call does not count as a call that may modify lists (no copy); a typed
      loop calling `sq(p.x()) + p.y()`: 0.18 s → 0.01 s
- [x] Native code calls the interpreter (v0.15) for what it does not compile: a
      function that is not compiled, a built-in (`print`), any method, a function
      value; its result is a value of any type (`JTy::Boxed`) on which operations are
      the interpreter's (`Guard` to a simple type checks its tag); a callee that cannot
      be compiled no longer prevents its callers from being compiled
- [x] Recompiled after a resumption, rather than interpreted for good: the place is
      remembered and done by the interpreter in the new code (a call that turned out
      polymorphic, an inlined method redefined…); interpreted after 4 places; a
      redefinition of what the code depends on forgets them
- [x] Functions and lambdas without annotations (lambdas without captured variables):
      a copy per types of the arguments (`Proto::specialize_params`), like `forall`; a
      result not annotated is any value (boxed in the caller's slot, a `Guard` reads
      it)
- [x] Classes constructed in native code (no field with a default value): the object,
      its `init` called directly (methods `init` are compiled), the check that every
      field is initialized; a typed loop constructing objects: 0.56 s → 0.08 s
- [x] Strings in native code: `str` parameters, results, fields, list elements,
      literals (the function's constants, not copied), `+`, comparisons, interpolation
      of strings and simple values, `len()` — the interpreter's operations, called by
      helpers (resumed in the interpreter if a value is not a string); a `nil` copied
      (value of an `if` without `else`) no longer prevents compilation; ×2.6 on a typed
      loop building strings

### v0.16 — Macros

- [x] `macro name(a, *rest) … end`, Crystal style (`syntax.md`): a template with
      `{{ }}` and `{% for / if / let %}`, expanded at parse time; the generated code is
      parsed after the script's source (its spans point into it, errors name the macro
      and its call)
- [x] Calls as statements, as type members (with `@type`: name, fields, methods) and
      as expressions (a `begin … end`)
- [x] Hygiene: the `let`s of the macro's own text renamed (in interpolations too)
- [x] `@def` (name, parameters, return type) and annotations read by macros
      (`f.annotation(Validate)`, `{% if v = … %}`): the `validate_fields` example works
- [x] Macros kept by the engine: usable by later scripts and by the files that
      `require` the one declaring them

### v0.17 — Main program in native code

- [x] `for` over a list without copying it (copied lazily, only if modified during the
      loop): a loop over 10^6 elements, peak memory 78 MB → 32 MB
- [x] Loops of the main program outlined into hidden functions (`outline.rs`): the
      registers read from before the loop are their parameters, compiled like any
      function typed for its arguments; not shown in call stacks
- [x] Globals in native code: a global of a simple type read and written in place (its
      tag checked, otherwise resumed in the interpreter; slots no longer move), an object
      global of the class of its value (its methods called directly, the class checked on
      every read); `global = v` the interpreter's assignment otherwise. A top-level loop
      summing into a global: 0.25 s → 0.03 s (same as in a function); a method called on a
      global object in a loop: ×2 to ×14
- [x] Closures with captured variables compiled, and functions whose variables are
      captured or that create lambdas (their cells in the native frame, closures built
      by a helper; unless a nested function reads a `do` block's context): their cells
      passed by the caller (`JitCtx::env`), a
      captured variable of a simple type read and written in place (type of its value
      when compiled, checked on every read), the closure in a global called directly;
      a counter closure called in a loop: 14.7 ms → 10.6 ms (300 k calls)
- [x] A value of any type returned by a function with a simple result: its tag checked
      (otherwise resumed in the interpreter at the `return`)
- [x] Fix: a function calling directly the typed copy of a function without
      annotations did not watch its global (stale code after a redefinition)
- [x] Any instruction the native code does not do (hashes, records, tuples,
      `destructure`, `case`, a `for` over a hash or a range value…) done by the
      interpreter's code on values rebuilt from the native ones (`BoxedStep`): such a
      function is compiled, unless one of these instructions is in a loop (the baseline
      code, which keeps values as they are, does them better there)
- [x] Fix: a `check` (annotated `let`) of a list or object shared the register it read
      in native code, which could be written again
- [x] `begin … rescue … ensure` in native code: an error raised in a protected block
      (detected by native code, by a function called, by the interpreter) goes to its
      handler (`bk_catch`: the value it receives, the error kept for `Term::Reraise`);
      `x is T` by a helper; the errors kept given to the interpreter on a resumption. A
      loop with a `rescue`: ×3 when 1 turn in 10 fails, ×20 or more otherwise

### v0.18 — Reference cycles collected

- [x] Cycle collector (`gc.rs`, the algorithm of CPython's): the containers created
      (lists, hashes, records, objects, cells of captured variables) registered by weak
      references; those only reachable from cycles are emptied, which frees them. Safe
      by construction (a container that cannot be read, or referenced by a value not
      registered, counts as referenced from outside). Run after a script, and at a
      function call once enough containers were created (threshold growing with the
      survivors); `Engine::collect_cycles()` from Rust. Objects whose fields all have
      simple types are not registered (they cannot be in a cycle). No measurable cost
      on the benchmarks.

### v0.19 — Limits lifted

- [x] Hashes and records in the loops of typed code: indexing, fields, hash and record
      literals call the interpreter's function for them directly, on borrowed values
      (`direct_op`, no registers rebuilt): such a function is compiled even with them in
      a loop. A value of any type with a number (`h[k] + 1`) is speculated to be a
      number (tag checked, otherwise resumed in the interpreter, which then does that
      instruction): `t += h[1] + 1`, 3 M turns, 0.08 s (baseline) → 0.05 s
- [x] Native code calling the interpreter grows the stack when needed (a deep recursion
      through `map` and native functions)
- [x] A global list of ints, floats, bools or strings (its first element) read by native
      code as such a list, each element checked as it is read: a loop over a global list
      of 1,000 ints, 3,000 times: 0.03 s → under 0.01 s
- [x] Cycle collector: closures that capture and tuples holding containers registered
      too (a cycle through a closure referenced several times is freed); a value built
      by the host is registered when it reaches the engine (`set_global`, the result of a
      Rust function, `Engine::call`); registered twice, it is counted once
- [x] Macros: calls without parentheses (`getters name, email`, `derive_to_s`: the
      arguments up to the end of the line), named arguments (`prop(host, kind: str)`),
      parameters with a default value, parameters after `*rest` given by name

### v0.20 — Nil and type tests

- [x] Patterns in `for`: `for (k, v) in [(1, "a"), …]`, `for {x, y} in points`,
      nested (`for (i, (a, b)) in …`): a hidden variable destructured by a `let` at the
      top of the body; typed and compiled like any `let`
- [x] `cond ? a : b` (an `if`) and `a ?? b` (`a`, or `b` when `a` is `nil`, `a`
      evaluated once): desugared by the parser, no new instruction
- [x] Safe navigation `a?.b`, `a?.m(x)`, `a?[i]`: `nil` as soon as a link of the
      chain is `nil`, the rest of the chain skipped; `a?.b ?? d`; a line starting with
      `?.` continues the previous one
- [x] Methods ending with `?` called without parentheses (`x.nil?`)
- [x] The checker refines a variable's type by its tests: `x.nil?`, `x == nil`,
      `x != nil`, `if x`, `not`, `unless`, `and` / `or`; after a branch ending with
      `return`, `raise`, `break` or `next`, the opposite holds for the rest of the block
      (`return 0 if s.nil?` then `s.len()`); ended by an assignment and by a loop; local
      variables and parameters only (not `@a` fields, not inside a lambda)
- [x] `if let a = @a` (not `nil` in `then`; `if let n = @n and n > 0`) and
      `if (let a = @a).is_a?(T)`: a local copy declared for the whole `if`, refined
      by the condition; fix: `let a = b` with `b: int?` was an `any` (`nil` widened in
      a union), it is an `int?`
- [x] Type reflection (Crystal's): `x.is_a?(T)` (refines too), `x.as(T)` (checked as
      an annotation: an error at run time, or before running if it can never be),
      `x.as?(T)` (`nil` if it is not a `T`), `x.responds_to?(name)` (refines to the
      types that have it); `T` any annotation type (`int | str`, `[int]`, a class, a
      trait, a generic type checked as its class)
- [x] Refined types in native code: a variable the checker refined to `int`, `float`
      or `bool` (`if x != nil … x * 2`) read into a typed register (its tag checked);
      parameters of any value (`x: int?`, `x: int | str`, `x: any`) compiled (a
      pointer to the argument, checked by the interpreter before the call); `x == nil`
      on such a value is a test of its tag. A loop of 5 M turns over an `int?`
      parameter: 0.09 s (baseline code) → under 0.01 s, as with an `int`
- [x] `case` by type: `when int`, `when Animal`, `when int | str` (an `is_a?` of the
      subject, which is refined in that arm and, in the next ones and `else`, to the
      types left); not `nil` without `else` when every type is covered
- [x] `Value` at 16 bytes (24 before): `Str` and `Tuple` are one pointer wide (thin
      reference-counted slices, the length in the allocation), `Variant(index, enum)`
      beside the tag (`repr(u8)`: a `bool`'s payload at byte 1). A list of 2 M
      integers: 149 MB → 102 MB of memory; a concatenation is a single allocation;
      speed otherwise unchanged. NaN-boxing (8 bytes) left aside: an `i64` does not fit

## Decisions that commit the JIT

| Topic                    | v0.1 choice                                         |
|--------------------------|-----------------------------------------------------|
| Value representation     | `enum Value` (16 bytes since v0.20, 24 before), basic types without allocation; NaN-boxing (8 bytes) left aside: `int` is a full `i64` |
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

- Reference cycles are freed by a collection (see v0.18), not as soon as they become
  unreachable; a value the host built and keeps (a `Value::List` it still holds) counts
  as referenced from outside, as does a container nested in it that the host shares.
- A `for` loop over a list does not copy it: the loop copies what it has left only when
  the list is modified (`<<`, `push`, `pop`, `xs[i] = v`) while it walks it. A list
  modified from Rust (`Value::List`) during a loop is seen modified.
- Globals are typed in native code (by their value, checked on every read) only in the
  loops of the main program and the functions compiled natively; the interpreter keeps
  them dynamic. A global list of objects (or empty when compiled) read in native code is
  any value.
