//! Compilation vers l'IR : sémantique des cas délicats, forme de l'IR produite.

use bokoni::{Engine, Value};

fn run(src: &str) -> String {
    let mut e = Engine::new();
    e.set_output(|_| {});
    match e.eval(src) {
        Ok(Value::Nil) => String::new(),
        Ok(v) => e.inspect(&v),
        Err(err) => panic!("{}", err.render(src, "test")),
    }
}

fn err(src: &str) -> String {
    let mut e = Engine::new();
    e.set_output(|_| {});
    match e.eval(src) {
        Ok(v) => panic!("devrait échouer, a donné {}", e.inspect(&v)),
        Err(err) => err.to_string(),
    }
}

fn ir(src: &str) -> String {
    Engine::new().ir(src).unwrap()
}

#[test]
fn evaluation_order() {
    // L'opérande gauche est lu avant que l'assignation à droite ne le modifie.
    assert_eq!(run("let x = 1\nx + (x = 2)"), "3");
    assert_eq!(run("let x = 1\nlet l = [x, (x = 5), x]\nl"), "[1, 5, 5]");
    // La valeur d'une assignation ne change pas avec la variable.
    assert_eq!(run("let a = 0\nlet b = (a = 3)\na = 9\nb"), "3");
}

#[test]
fn closures_capture_variables() {
    // Chaque tour de boucle a sa propre variable.
    assert_eq!(
        run("let fs = []\nfor i in 0..2\n  fs << || i\nend\nfs.map(|f| f())"),
        "[0, 1, 2]"
    );
    assert_eq!(
        run("let fs = []\nfor x in [\"a\", \"b\"]\n  fs << || x\nend\nfs.map(|f| f())"),
        "[\"a\", \"b\"]"
    );
    // Une closure voit les modifications ultérieures de la variable capturée.
    assert_eq!(
        run("def f()\n  let n = 1\n  let g = || n\n  n = 5\n  g()\nend\nf()"),
        "5"
    );
    // Capture à travers plusieurs niveaux, et modification depuis la closure.
    assert_eq!(
        run(
            "def outer()\n  let total = 0\n  let add = |x| [1].each do |_|\n    total += x\n  end\n  add(2)\n  add(3)\n  total\nend\nouter()"
        ),
        "5"
    );
    // Une fonction locale peut s'appeler elle-même.
    assert_eq!(
        run(
            "def f()\n  def fact(n) = if n < 2 then 1 else n * fact(n - 1) end\n  fact(5)\nend\nf()"
        ),
        "120"
    );
    // Une variable déclarée plus loin dans la fonction est visible des closures…
    assert_eq!(
        run("def f()\n  let g = || x\n  let x = 5\n  g()\nend\nf()"),
        "5"
    );
    // …mais la lire avant sa déclaration est une erreur.
    assert!(
        err("def f()\n  let g = || x\n  g()\n  let x = 5\nend\nf()")
            .contains("avant sa déclaration")
    );
}

#[test]
fn typed_registers_keep_their_type() {
    // `n` est un `int` (déduit) : une valeur dynamique d'un autre type est refusée à
    // l'exécution, même si le vérificateur ne pouvait pas le savoir.
    let e = err("def any(x) = x\ndef f()\n  let n = 0\n  n = any(\"a\")\nend\nf()");
    assert!(
        e.contains("une variable garde le type de sa première valeur"),
        "{e}"
    );
    // `int` vers `float` est converti.
    assert_eq!(
        run("def f(): float\n  let x = 1.5\n  x = 2\n  x\nend\nf()"),
        "2.0"
    );
    // Sans vérification statique, pas de registre typé : comportement dynamique.
    let mut en = Engine::new();
    en.set_static_checks(false);
    assert!(
        en.eval("def f()\n  let n = 0\n  n = \"a\"\n  n\nend\nf()")
            .is_ok()
    );
}

#[test]
fn construction_blocks_resolve_names() {
    let src = "type Route\n  path: str\nend\ntype Server\n  mode = \"dev\"\n  routes: [Route] = []\nend\n";
    // Dans un bloc imbriqué, un champ du bloc extérieur est lisible.
    assert_eq!(
        run(&format!(
            "{src}let s = Server do\n  mode = \"prod\"\n  routes << Route do\n    path = \"/#{{mode}}\"\n  end\nend\ns.routes[0].path"
        )),
        "\"/prod\""
    );
    // Un champ masque une variable extérieure du même nom.
    assert_eq!(
        run(&format!(
            "{src}let mode = \"externe\"\nServer do\n  mode = \"champ\"\nend.mode"
        )),
        "\"champ\""
    );
    assert_eq!(
        run(&format!(
            "{src}def f()\n  let mode = \"x\"\n  Server do\n  end\nend\nf().mode"
        )),
        "\"dev\""
    );
    // Un bloc `do` ordinaire voit les variables extérieures.
    assert_eq!(
        run("def f()\n  let t = 0\n  [1, 2].each do |n|\n    t += n\n  end\n  t\nend\nf()"),
        "3"
    );
}

#[test]
fn state_persists_between_evaluations() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("let n = 1\ndef inc() = n += 1").unwrap();
    e.eval("inc()\ninc()").unwrap();
    assert_eq!(e.eval_as::<i64>("n").unwrap(), 3);
    // Redéfinir une fonction remplace l'ancienne.
    e.eval("def inc() = n += 10").unwrap();
    e.eval("inc()").unwrap();
    assert_eq!(e.eval_as::<i64>("n").unwrap(), 13);
}

#[test]
fn declarations_are_top_level() {
    let e = err("def f()\n  type T\n    x: int\n  end\nend");
    assert!(e.contains("au niveau principal"), "{e}");
}

#[test]
fn ir_uses_typed_instructions() {
    let typed =
        ir("def f(n: int): int\n  let s = 0\n  for i in 0...n\n    s += i * 2\n  end\n  s\nend");
    assert!(typed.contains("*i 2"), "{typed}"); // opérande immédiat
    assert!(typed.contains("+i"), "{typed}");
    assert!(typed.contains("si r"), "{typed}"); // comparaison et branchement fusionnés
    assert!(typed.contains("registres typés"), "{typed}");
    // Paramètre non annoté : opération générique.
    let generic = ir("def g(a, b) = a + b");
    assert!(generic.contains(" + r"), "{generic}");
    assert!(!generic.contains("+i"), "{generic}");
}
