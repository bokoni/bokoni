# Bokoni

A scripting language embedded in Rust: Ruby-like syntax, gradual typing (optional
annotations), with a Cranelift JIT for typed code as the end goal.

```ruby
def fib(n: int): int
  if n < 2 then n else fib(n - 1) + fib(n - 2) end
end

let ages = {"alice" => 30, "bob" => 25}
for name, age in ages
  print("#{name.capitalize()} is #{age}")
end

[1, 2, 3, 4].filter(|n| n % 2 == 0).map(|n| n * n)   # [4, 16]
```

Before running, the whole program is checked (syntax, names, types) and **all** errors
are reported at once, with suggestions:

```text
type error: `str` has no method `lenght` (did you mean `len`?)
 --> script.bk:12:1
   |
12 | "abc".lenght()
   | ^^^^^^^^^^^^^^
```

Gradual typing: only certain errors are reported; everything else is checked at run
time.

Errors are exceptions, as in Crystal (custom errors are classes that inherit from
`Error`):

```ruby
let ratio = begin
  total / count
rescue e: Error
  print("cannot compute: #{e.message}")
  0
ensure
  close(file)
end
```

Fully typed functions are compiled to native code (Cranelift JIT) on their first call:
`fib(25)` goes from 28 ms (interpreted) to 0.5 ms. A generic function
(`def max(a: T, b: T): T forall T`) is compiled once per type it is called with, and
methods, lists and strings are compiled too; where an assumption of the native code does
not hold, the interpreter takes over at that point.
Everything else is interpreted, with exactly the same result.

Macros, as in Crystal, generate code at parse time:

```ruby
macro getters(*names)
  {% for name in names %}
    def {{name}} = @{{name}}
  {% end %}
end

class User
  def init(@name: str, @email: str)
  end

  getters(name, email)
end
```

## Using it from Rust

```rust
use bokoni::Engine;

let mut engine = Engine::new();
engine.register_fn("double", |x: i64| x * 2);

let v: i64 = engine.eval_as("double(21)")?;          // 42
engine.eval("def square(x: int): int = x * x")?;
let s: i64 = engine.call_as("square", (7,))?;        // 49
```

Arguments and return values are converted automatically (`i64`, `f64`, `bool`, `String`,
`Vec<T>`, `Option<T>`, `Value`…); a function may return `Result<T, E: Display>`.

The script's declarations and their annotations can be read from Rust:

```rust
engine.eval("annotation Column\n  name: str\nend\n\
             type User\n  @[Column(\"user_name\")]\n  name: str = \"\"\nend")?;
let ty = engine.type_info("User")?;
let col = ty.field("name").unwrap().annotation("Column").unwrap();
let column: String = col.get("name")?;               // "user_name"
```

`require "name"` asks the host for the file (from disk, an archive, the network…):

```rust
engine.set_loader(|name, _from| match assets.get(name) {
    Some(text) => Ok(Source { name: format!("{name}.bk"), text: text.clone() }),
    None => Err(format!("unknown file `{name}`")),
});
```

## Command line

```sh
cargo run                         # REPL
cargo run -- script.bk            # run a file
cargo run -- -e '1 + 2'           # => 3
cargo run -- --ast -e '1 + 2 * 3' # (+ 1 (* 2 3))
cargo run -- --ir script.bk       # IR being executed (typed registers, basic blocks)
cargo run -- --no-jit script.bk   # without native compilation
```

## Documentation

- [docs/syntax.md](docs/syntax.md) — the syntax (specification)
- [docs/roadmap.md](docs/roadmap.md) — roadmap to the JIT, progress, performance

## Development

```sh
cargo test                                       # tests (including every example in syntax.md)
cargo run --release --example bench_parser       # parser vs peg
cargo run --release --example bench_interp       # execution speed (with / without JIT)
cargo build --no-default-features                # without the JIT (platforms without Cranelift)
```
