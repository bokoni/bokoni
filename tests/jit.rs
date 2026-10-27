//! Compilation native : mêmes résultats et mêmes erreurs que l'interpréteur.
#![cfg(feature = "jit")]

use bokoni::Engine;

/// Résultat (ou message d'erreur affiché, pile d'appels comprise) d'un programme.
fn outcome(src: &str, jit: bool) -> String {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.set_jit(jit);
    match e.eval(src) {
        Ok(v) => e.inspect(&v),
        Err(err) => err.render(src, "test"),
    }
}

/// Le programme donne la même chose avec et sans JIT.
fn same(src: &str) -> String {
    let (a, b) = (outcome(src, false), outcome(src, true));
    assert_eq!(a, b, "JIT différent de l'interpréteur pour :\n{src}");
    b
}

#[test]
fn integer_operations_match_the_interpreter() {
    let values = [i64::MIN, i64::MIN + 1, -7, -2, -1, 0, 1, 2, 3, 7, i64::MAX];
    for op in ["+", "-", "*", "/", "%"] {
        let src = format!("def f(a: int, b: int): int = a {op} b\n");
        for &a in &values {
            for &b in &values {
                same(&format!("{src}f({a}, {b})"));
            }
        }
    }
    for op in ["<", "<=", ">", ">=", "==", "!="] {
        for &a in &values[..6] {
            same(&format!(
                "def f(a: int, b: int): bool = a {op} b\nf({a}, -1)"
            ));
        }
    }
    same(&format!("def neg(a: int): int = -a\nneg({})", i64::MIN));
    same("def k(a: int): int = a * 3 + 10 / 4 - a % 5\nk(-17)");
}

#[test]
fn float_operations_match_the_interpreter() {
    let values = ["-7.5", "-1.0", "0.0", "0.5", "3.0", "1e300"];
    for op in ["+", "-", "*", "/", "%", "<", "=="] {
        for a in values {
            for b in values {
                let ret = if matches!(op, "<" | "==") {
                    "bool"
                } else {
                    "float"
                };
                same(&format!(
                    "def f(a: float, b: float): {ret} = a {op} b\nf({a}, {b})"
                ));
            }
        }
    }
    // `int` passé à un paramètre `float`, `int` mêlé à un `float`.
    same("def h(x: float): float = x * 2 + 1\nh(3)");
    same(
        "def s(n: int): float\n  let t = 0.0\n  for i in 1..n\n    t += 1.0 / i\n  end\n  t\nend\ns(1000)",
    );
}

#[test]
fn programs_match_and_are_compiled() {
    let src = "def fib(n: int): int\n  if n < 2 then n else fib(n - 1) + fib(n - 2) end\nend\n\
               def even(n: int): bool = if n == 0 then true else odd(n - 1) end\n\
               def odd(n: int): bool = if n == 0 then false else even(n - 1) end\n\
               def collatz(n: int): int\n  let steps = 0\n  let x = n\n  while x != 1\n    x = if x % 2 == 0 then x / 2 else 3 * x + 1 end\n    steps += 1\n  end\n  steps\nend\n";
    assert_eq!(
        same(&format!("{src}[fib(20), even(11), odd(11), collatz(27)]")),
        "[6765, false, true, 111]"
    );
    let mut e = Engine::new();
    e.eval(&format!("{src}fib(5)\neven(3)\ncollatz(6)"))
        .unwrap();
    for f in ["fib", "even", "odd", "collatz"] {
        assert!(e.is_native(f), "`{f}` devrait être compilée");
    }
    // Non compilées : paramètre non annoté, retour non annoté, fonction intégrée.
    e.eval("def untyped(x) = x + 1\nuntyped(1)\ndef noret(x: int) = x\nnoret(1)\ndef pr(x: int): int\n  print(x)\n  x\nend").unwrap();
    for f in ["untyped", "noret", "pr"] {
        assert!(!e.is_native(f), "`{f}` ne devrait pas être compilée");
    }
}

#[test]
fn errors_match_the_interpreter() {
    // Message, position et pile d'appels identiques.
    let e = same(
        "def a(x: int): int = b(x) + 1\ndef b(x: int): int = c(x) * 2\ndef c(x: int): int = 10 / x\na(0)",
    );
    assert!(e.contains("division par zéro"), "{e}");
    assert!(e.contains("dans `c` appelé depuis <eval>:2:22"), "{e}");
    assert!(e.contains("dans `a` appelé depuis <eval>:4:1"), "{e}");
    let e = same("def inf(n: int): int = inf(n + 1)\ninf(0)");
    assert!(e.contains("trop d'appels imbriqués (1000)"), "{e}");
    same("def big(n: int): int = n * n * n * n * n\nbig(10000)");
}

#[test]
fn redefinition_invalidates_native_code() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("def sq(x: int): int = x * x\ndef f(x: int): int = sq(x) + 1")
        .unwrap();
    assert_eq!(e.eval_as::<i64>("f(3)").unwrap(), 10);
    assert!(e.is_native("f"));
    // Nouvelle version de `sq` : `f` est recompilée et l'utilise.
    e.eval("def sq(x: int): int = x * x * x").unwrap();
    assert_eq!(e.eval_as::<i64>("f(3)").unwrap(), 28);
    assert!(e.is_native("f"));
    // `sq` n'est plus compilable : `f` est interprétée, toujours correcte.
    e.eval("def sq(x) = x").unwrap();
    assert_eq!(e.eval_as::<i64>("f(3)").unwrap(), 4);
    assert!(!e.is_native("f"));
}

#[test]
fn deep_recursion_beyond_the_native_stack() {
    // La pile native ne suffit pas pour 200 000 appels : l'interpréteur prend le relais
    // (sa pile grandit sur le tas), avec le même résultat.
    let src = "def down(n: int): int = if n == 0 then 0 else 1 + down(n - 1) end\ndown(200000)";
    let mut e = Engine::new();
    e.set_max_call_depth(1_000_000);
    assert_eq!(e.eval_as::<i64>(src).unwrap(), 200_000);
    // Et la limite de l'utilisateur est respectée, en natif comme ailleurs.
    let mut e = Engine::new();
    e.set_max_call_depth(50);
    let err = e
        .eval("def down(n: int): int = if n == 0 then 0 else 1 + down(n - 1) end\ndown(100)")
        .unwrap_err();
    assert!(
        err.to_string().contains("trop d'appels imbriqués (50)"),
        "{err}"
    );
}

/// Comme `same`, avec la compilation de base dès le premier appel de chaque fonction.
fn same_eager(src: &str) -> String {
    let interp = outcome(src, false);
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.set_jit_eager(true);
    let native = match e.eval(src) {
        Ok(v) => e.inspect(&v),
        Err(err) => err.render(src, "test"),
    };
    assert_eq!(interp, native, "code natif de base différent pour :\n{src}");
    native
}

#[test]
fn untyped_code_matches_the_interpreter() {
    let programs = [
        // Arithmétique générique : entiers, décimaux, mélanges, chaînes, cas limites.
        "def f(a, b) = [a + b, a - b, a * b, a / b, a % b, a < b, a == b]\n[f(7, -2), f(-7.5, 2.0), f(3, 0.5), f(-7, 2)]",
        &format!("def f(a, b) = a + b\nf({}, 1)", i64::MAX),
        &format!("def f(a, b) = a / b\nf({}, -1)", i64::MIN),
        "def f(a, b) = a % b\nf(5, 0)",
        "def f(a, b) = a + b\n[f(\"a\", \"b\"), f([1], [2])]",
        "def f(a) = not a\n[f(nil), f(false), f(0), f(\"\"), f([])]",
        "def f(a) = if a then 1 else 2 end\n[f(nil), f(false), f(true), f(0), f([1])]",
        // Closures, captures modifiées, compteurs.
        "def counter()\n  let n = 0\n  || n += 1\nend\nlet c = counter()\nlet last = 0\nfor i in 0...1000\n  last = c()\nend\nlast",
        "def make(k) = |x| x * k\nlet fs = [make(2), make(3.5)]\nfs.map(|f| f(4))",
        "def f()\n  let fs = []\n  for i in 0..3\n    fs << || i * i\n  end\n  fs.map(|g| g())\nend\nf()",
        // Listes, itérateurs, hash.
        "let xs = (0...200).to_list()\nxs.map(|x| x * 2).filter(|x| x % 3 == 0).sum()",
        "let t = 0\nfor x in [1, 2.5, 3]\n  t = t + x\nend\nt",
        "let h = {\"a\" => 1, \"b\" => 2}\nlet s = 0\nh.each do |k, v|\n  s += v\nend\ns",
        // Objets, classes, enum.
        "type P\n  x: int = 0\n  y: int = 0\nend\nlet s = 0\nfor i in 0...500\n  let p = P(x: i, y: 2)\n  s += p.x + p.y\nend\ns",
        "type P\n  x: int = 0\n  y = x + 1\nend\n[P(x: 3), P(), P(4, 5), P do\n  x = 9\nend]",
        "type P\n  x: int\nend\nP(x: \"a\")",
        "type P\n  x: int\nend\nP(x: 1, x: 2)",
        "type P\n  x: int\nend\nP(z: 1)",
        "type P\n  x: int\nend\nP()",
        "class C\n  def init(@n: int)\n  end\n  def get() = @n\nend\nlet c = C(4)\nc.get() + C(5).get()",
        // Erreurs à l'exécution dans du code non typé, avec la pile d'appels.
        "def a(x) = b(x) + 1\ndef b(x) = x.nope()\na(1)",
        "def a(x) = x + \"s\"\na(1)",
        "def f()\n  let g = || x\n  g()\n  let x = 5\nend\nf()",
        "def any(x) = x\ndef f()\n  let n = 0\n  n = any(\"a\")\nend\nf()",
        "def inf(n) = inf(n + 1)\ninf(0)",
        // Accès aux champs (cache) : plusieurs types au même endroit, champ à une autre
        // position, classe (getter), record, variante, valeur modifiée entre deux lectures.
        "type A\n  x: int = 1\nend\ntype B\n  y: int = 2\n  x: int = 3\nend\ndef gx(o) = o.x\nlet t = 0\nfor i in 0...20\n  t += gx(if i % 3 == 0 then A() else B() end)\nend\nt",
        "class C\n  def init(@x: int)\n  end\n  def x = @x * 10\nend\ntype A\n  x: int = 1\nend\ndef gx(o) = o.x\n[gx(A()), gx(C(4)), gx(A()), gx({x: 7})]",
        "class C\n  def init(@x: int)\n  end\nend\ndef gx(o) = o.x\ngx(C(1))",
        "type A\n  x: int = 1\nend\ndef gx(o) = o.x\nlet a = A()\nlet r = [gx(a)]\na.x = 5\nr << gx(a)\nr",
        "type A\n  x: int = 1\nend\ndef gx(o) = o.y\ngx(A())",
        // Globales et variables capturées : cas d'erreur.
        "def f() = g + 1\nf()",
        "def f()\n  undefined_x = 1\nend\nf()",
        "def f()\n  print = 1\nend\nf()",
        "def f()\n  let g = || y\n  let r = g()\n  let y = 1\n  r\nend\nf()",
        "def f()\n  let n = 0\n  let inc = || n += 1\n  for i in 0...10\n    inc()\n  end\n  n\nend\nf()",
        // Chaînes.
        "let s = \"\"\nfor i in 0...50\n  s = \"#{s}#{i % 10}\"\nend\ns.len()",
    ];
    for src in programs {
        same_eager(src);
        same(src);
    }
}

#[test]
fn construction_errors_at_run_time() {
    // Sans vérification statique, les erreurs viennent de l'exécution.
    let run = |src: &str| {
        let mut e = Engine::new();
        e.set_static_checks(false);
        e.eval(src)
            .map(|v| e.inspect(&v))
            .unwrap_or_else(|err| err.to_string())
    };
    let ty = "type P\n  x: int = 0\n  y: int = 0\nend\n";
    assert!(run(&format!("{ty}P(z: 1, y: 2)")).contains("`P` n'a pas de champ `z`"));
    assert!(run(&format!("{ty}P(1, x: 2)")).contains("champ `x` donné deux fois"));
    assert!(
        run(&format!("{ty}P(y: \"a\", x: 1)")).contains("type attendu `int`, reçu str (\"a\")")
    );
    assert!(run(&format!("{ty}P(1, 2, 3)")).contains("`P` a 2 champs, reçu 3 arguments"));
    assert_eq!(run(&format!("{ty}P(y: 1, x: 2)")), "P(x: 2, y: 1)");
}

#[test]
fn deep_untyped_recursion() {
    // Récursion profonde par tous les chemins d'appel : directe, à travers `map`, une
    // méthode, une closure. La pile grandit sur le tas, sans débordement.
    let programs = [
        "def down(n) = if n == 0 then 0 else 1 + down(n - 1) end\ndown(100000)",
        "def down(n) = if n == 0 then 0 else [n].map(|x| down(x - 1))[0] + 1 end\ndown(30000)",
        "class C\n  def down(n) = if n == 0 then 0 else 1 + self.down(n - 1) end\nend\nC().down(50000)",
        "let f = nil\nf = |n| if n == 0 then 0 else 1 + f(n - 1) end\nf(100000)",
        "type T\n  x: int = 0\nend\ndef down(n) = if n == 0 then T() else T do\n  x = down(n - 1).x + 1\nend end\ndown(20000).x",
    ];
    for src in programs {
        for eager in [false, true] {
            let mut e = Engine::new();
            e.set_max_call_depth(10_000_000);
            e.set_jit_eager(eager);
            let r = e
                .eval(src)
                .map(|v| e.inspect(&v))
                .unwrap_or_else(|err| err.to_string());
            assert!(r.chars().all(|c| c.is_ascii_digit()), "{src} → {r}");
        }
    }
}

#[test]
fn field_cache_sees_mutations() {
    assert_eq!(
        same_eager(
            "type A\n  x: int = 1\nend\ndef gx(o) = o.x\nlet a = A()\nlet r = [gx(a)]\na.x = 5\nr << gx(a)\nr"
        ),
        "[1, 5]"
    );
}

#[test]
fn typed_functions_over_objects() {
    let src = "type P\n  x: int = 0\n  y: float = 0.5\n  ok: bool = true\nend\n\
               def dot(a: P, b: P): float = a.x * b.x + a.y * b.y\n\
               def score(p: P): float = if p.ok then dot(p, p) else 0.0 end\n\
               def bad(p: P): int = p.x * 4611686018427387904\n";
    let r = same(&format!(
        "{src}[score(P(x: 3, y: 2)), score(P(x: 1, ok: false)), dot(P(x: 2), P(x: 5, y: 4))]"
    ));
    assert_eq!(r, "[13.0, 0.0, 12.0]");
    // Erreur dans du code natif sur un objet : même message, même pile.
    same(&format!("{src}bad(P(x: 3))"));
    let mut e = Engine::new();
    e.eval(&format!("{src}score(P(x: 1))")).unwrap();
    assert!(
        e.is_native("score") && e.is_native("dot"),
        "fonctions sur objets compilées"
    );
}

#[test]
fn redeclared_type_falls_back_to_the_interpreter() {
    // `f` est compilée en supposant `P.x : int` ; `P` est ensuite redéclaré avec `x: str`.
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("type P\n  x: int = 1\nend\ndef f(p: P): int = p.x + 1\nf(P())")
        .unwrap();
    assert!(e.is_native("f"));
    e.eval("type P\n  x: str = \"a\"\nend").unwrap();
    let err = e.eval("f(P())").unwrap_err().to_string();
    assert!(
        err.contains("le champ `x` devait être un `int`, reçu str"),
        "{err}"
    );
    assert!(!e.is_native("f"));
    // Redéclaré avec la même forme, puis une autre disposition : recompilé, correct.
    e.eval("type P\n  x: int = 2\nend").unwrap();
    assert_eq!(e.eval_as::<i64>("f(P())").unwrap(), 3);
    e.eval("type P\n  pad: str = \"\"\n  x: int = 7\nend")
        .unwrap();
    assert_eq!(e.eval_as::<i64>("f(P())").unwrap(), 8);
    assert!(e.is_native("f"));
}

#[test]
fn typed_fields_in_untyped_code() {
    // Champs typés lus dans des fonctions que le JIT typé ne compile pas (retour non
    // annoté, boucle sur une liste, objet local) : code de base, même résultat.
    let ty = "type P\n  x: int = 0\n  y: float = 1.5\n  name: str = \"p\"\nend\n";
    for body in [
        "def f(p: P) = p.x + p.y\nf(P(x: 2))",
        "def f(ps: [P]): float\n  let t = 0.0\n  for p in ps\n    t += p.x * p.y\n  end\n  t\nend\nf([P(x: 1), P(x: 2, y: 2.0)])",
        "def f(n: int): int\n  let p = P(x: n)\n  p.x * 2\nend\nf(21)",
        "def f(p: P) = \"#{p.name}:#{p.x}\"\nf(P(x: 4))",
        "def f(p: P): int = p.x * 4611686018427387904\nf(P(x: 3))",
    ] {
        same_eager(&format!("{ty}{body}"));
        same(&format!("{ty}{body}"));
    }
}
