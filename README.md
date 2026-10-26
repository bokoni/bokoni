# Bokoni

Langage de script intégré à Rust : syntaxe à la Ruby, typage graduel (annotations
optionnelles), objectif final un JIT Cranelift pour le code typé.

```ruby
def fib(n: int): int
  if n < 2 then n else fib(n - 1) + fib(n - 2) end
end

let ages = {"alice" => 30, "bob" => 25}
for name, age in ages
  print("#{name.capitalize()} a #{age} ans")
end

[1, 2, 3, 4].filter(|n| n % 2 == 0).map(|n| n * n)   # [4, 16]
```

Avant l'exécution, tout le programme est vérifié (syntaxe, noms, types) et **toutes** les
erreurs sont signalées d'un coup, avec suggestions :

```text
erreur de type: `str` n'a pas de méthode `lenght` (voulez-vous dire `len` ?)
 --> script.bk:12:1
   |
12 | "abc".lenght()
   | ^^^^^^^^^^^^^^
```

Typage graduel : seules les erreurs certaines sont signalées, le reste est vérifié à
l'exécution.

Les fonctions entièrement typées sont compilées en code natif (JIT Cranelift) à leur
premier appel : `fib(25)` passe de 28 ms (interprété) à 0,5 ms. Le reste est interprété,
avec exactement le même résultat.

## Utilisation depuis Rust

```rust
use bokoni::Engine;

let mut engine = Engine::new();
engine.register_fn("double", |x: i64| x * 2);

let v: i64 = engine.eval_as("double(21)")?;          // 42
engine.eval("def square(x: int): int = x * x")?;
let s: i64 = engine.call_as("square", (7,))?;        // 49
```

Les arguments et retours sont convertis automatiquement (`i64`, `f64`, `bool`, `String`,
`Vec<T>`, `Option<T>`, `Value`…) ; une fonction peut renvoyer `Result<T, E: Display>`.

## Exécutable

```sh
cargo run                         # REPL
cargo run -- script.bk            # exécute un fichier
cargo run -- -e '1 + 2'           # => 3
cargo run -- --ast -e '1 + 2 * 3' # (+ 1 (* 2 3))
cargo run -- --ir script.bk       # IR exécutée (registres typés, blocs de base)
cargo run -- --no-jit script.bk   # sans compilation native
```

## Documentation

- [docs/syntax.md](docs/syntax.md) — la syntaxe (spécification)
- [docs/roadmap.md](docs/roadmap.md) — feuille de route vers le JIT, état d'avancement, performances

## Développement

```sh
cargo test                                       # tests (dont tous les exemples de syntax.md)
cargo run --release --example bench_parser       # parseur vs peg
cargo run --release --example bench_interp       # vitesse d'exécution (avec / sans JIT)
cargo build --no-default-features                # sans le JIT (plateformes sans Cranelift)
```
