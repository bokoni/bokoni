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
