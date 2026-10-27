//! Fuzzing : textes aléatoires, mutations de programmes valides et programmes générés.
//! Pour chacun, aucun plantage (panique), aucun blocage, et le même résultat — ou le
//! même message d'erreur — avec l'interpréteur, le JIT et le code natif de base.
//!
//! Graine fixe : un échec se reproduit. `BK_FUZZ=<n>` multiplie le nombre de cas,
//! `BK_FUZZ_SEED=<n>` change la graine (autres programmes).

use std::sync::mpsc;
use std::time::Duration;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len())]
    }
}

/// Graine du générateur : `base`, modifiée par `BK_FUZZ_SEED`.
fn seed(base: u64) -> Rng {
    let extra: u64 = std::env::var("BK_FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    Rng(base ^ extra.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
}

/// Nombre de cas : `n` (un cinquième en debug, pour que `cargo test` reste rapide),
/// multiplié par `BK_FUZZ`.
fn count(n: usize) -> usize {
    match std::env::var("BK_FUZZ")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(k) => n * k,
        None if cfg!(debug_assertions) => n / 5,
        None => n,
    }
}

/// Résultat d'un programme dans un mode : valeur affichée ou erreurs.
fn outcome(src: &str, jit: bool, eager: bool, checks: bool) -> String {
    let mut e = bokoni::Engine::new();
    e.set_output(|_| {});
    e.set_static_checks(checks);
    e.set_jit(jit);
    #[cfg(feature = "jit")]
    e.set_jit_eager(eager);
    let _ = eager;
    e.set_max_call_depth(200);
    match e.eval(src) {
        Ok(v) => e.inspect(&v),
        Err(err) => err.render(src, "fuzz"),
    }
}

/// Exécute `src` dans les trois modes (interpréteur, JIT, code de base dès le premier
/// appel), avec puis sans vérification statique (le code mal typé atteint alors
/// l'exécution), dans un fil séparé (délai maximal, pile large) ; échoue en cas de
/// panique, de blocage ou de résultats différents.
fn run(src: &str) {
    let code = src.to_string();
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(move || {
            let r = std::panic::catch_unwind(|| {
                [true, false].map(|checks| {
                    [
                        outcome(&code, false, false, checks),
                        outcome(&code, true, false, checks),
                        outcome(&code, true, true, checks),
                    ]
                })
            });
            let _ = tx.send(r.map_err(|p| {
                p.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default()
            }));
        })
        .unwrap();
    match rx.recv_timeout(Duration::from_secs(20)) {
        Ok(Ok(modes)) => {
            for [a, b, c] in modes {
                assert!(
                    a == b && a == c,
                    "résultats différents pour :\n{src}\n--- interpréteur ---\n{a}\n--- JIT ---\n{b}\n--- base ---\n{c}"
                );
            }
        }
        Ok(Err(msg)) => panic!("plantage ({msg}) pour :\n{src}"),
        Err(_) => panic!("bloqué (plus de 20 s) :\n{src}"),
    }
}

const VOCAB: &[&str] = &[
    "let ",
    "def ",
    "end",
    "if ",
    "then ",
    "else ",
    "elsif ",
    "unless ",
    "case ",
    "when ",
    "for ",
    " in ",
    "do ",
    "return ",
    "break",
    "next",
    "type ",
    "class ",
    "enum ",
    "annotation ",
    "true",
    "false",
    "nil",
    "and ",
    "or ",
    "not ",
    "self",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ",",
    ".",
    ":",
    "::",
    " = ",
    " == ",
    " != ",
    " + ",
    " - ",
    " * ",
    " / ",
    " % ",
    " < ",
    " <= ",
    " > ",
    " >= ",
    " << ",
    "|",
    "||",
    "->",
    "=>",
    "...",
    "..",
    "@",
    "@[",
    "#",
    "?",
    "!",
    "\n",
    "\n",
    "\n",
    " ",
    "a",
    "b",
    "f",
    "x",
    "P",
    "int",
    "float",
    "str",
    "0",
    "1",
    "-1",
    "42",
    "9223372036854775807",
    "-9223372036854775808",
    "1.5",
    "1e300",
    "\"s\"",
    "\"#{a}\"",
    "\"",
    "'",
    "_",
    "init",
    "@x",
    "len()",
    "map",
    "|v|",
    "print",
];

/// Programmes valides servant de base aux mutations. Sans `while` : une mutation en fait
/// facilement une boucle infinie, indiscernable d'un blocage.
const CORPUS: &[&str] = &[
    "let a = 1\nlet b = a + 2\nb * 3",
    "def f(x: int): int = if x < 2 then x else f(x - 1) + f(x - 2) end\nf(10)",
    "def g(a, b) = a * b + 1\n[g(2, 3), g(1.5, 2), g(2, -3)]",
    "let xs = [1, 2, 3, 4]\nxs.map(|v| v * 2).filter(|v| v > 2).sum()",
    "let h = {\"a\" => 1, \"b\" => 2}\nh[\"c\"] = 3\nh.keys()",
    "let p = {x: 1, y: 2}\np.x = 5\np.x + p.y",
    "type P\n  x: int = 0\n  y: float = 1.5\nend\nlet p = P(x: 3)\np.x * p.y",
    "type S\n  host = \"h\"\n  port: int = 80\nend\nS do\n  port = 8080\nend",
    "class C\n  def init(@n: int)\n  end\n  def get() = @n * 2\nend\nC(21).get()",
    "enum E: str\n  A = \"a\"\n  B = \"b\"\nend\n[E::A.value, E::B, E.values()]",
    "def k()\n  let n = 0\n  let inc = || n += 1\n  inc()\n  inc()\n  n\nend\nk()",
    "let t = 0\nfor v in [1, 2, 3]\n  t += v\nend\nt",
    "case 3\nwhen 1 then \"un\"\nwhen 3 then \"trois\"\nelse \"autre\"\nend",
    "let (a, b) = (1, \"x\")\nlet [c, ...d] = [1, 2, 3]\n[a, b, c, d]",
    "def m(xs: [int]): int\n  let t = 0\n  for x in xs\n    t += x * x\n  end\n  t\nend\nm([1, 2, 3])",
    "type V\n  x: float = 0.0\nend\ndef add(a: V, b: V): V = V(x: a.x + b.x)\nadd(V(x: 1.0), V(x: 2.0)).x",
    "annotation A\n  n: int = 0\nend\n@[A(n: 2)]\ndef f() = \"#{1 + 1} et #{[1, 2]}\"\nf()",
    "let n = 9223372036854775807\n[n - 1, n / 2, -n, 7 % -3, -7 / 2, 2.0 / 3]",
];

/// Mutation d'un programme : suppression, duplication ou échange de lignes, insertion
/// d'un jeton, suppression ou remplacement de caractères.
fn mutate(r: &mut Rng, src: &str) -> String {
    let mut s = src.to_string();
    for _ in 0..1 + r.below(3) {
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        let at = r.below(n + 1);
        s = match r.below(6) {
            0 => {
                let len = 1 + r.below(5);
                chars[..at]
                    .iter()
                    .chain(chars[(at + len).min(n)..].iter())
                    .collect()
            }
            1 | 2 => {
                let tok = r.pick(VOCAB);
                let mut t: String = chars[..at].iter().collect();
                t.push_str(tok);
                t.extend(&chars[at..]);
                t
            }
            3 => {
                let mut lines: Vec<&str> = s.lines().collect();
                if !lines.is_empty() {
                    let i = r.below(lines.len());
                    lines.insert(i, lines[i]);
                }
                lines.join("\n")
            }
            4 => {
                let mut lines: Vec<&str> = s.lines().collect();
                if lines.len() > 1 {
                    let (i, j) = (r.below(lines.len()), r.below(lines.len()));
                    lines.swap(i, j);
                }
                lines.join("\n")
            }
            _ => {
                let mut t = chars.clone();
                if at < n {
                    t[at] = r
                        .pick(&["(", ")", "\"", "e", "1", " ", "\n", ".", ":", "|"])
                        .chars()
                        .next()
                        .unwrap();
                }
                t.into_iter().collect()
            }
        };
    }
    s
}

#[test]
fn random_token_soup() {
    let mut r = seed(0xA076_1D64_78BD_642F);
    for _ in 0..count(300) {
        let n = 1 + r.below(40);
        let src: String = (0..n).map(|_| r.pick(VOCAB)).collect();
        run(&src);
    }
}

#[test]
fn mutated_programs() {
    let mut r = seed(0xE703_7ED1_A0B4_28DB);
    // Le corpus lui-même s'exécute sans erreur.
    for src in CORPUS {
        let mut e = bokoni::Engine::new();
        e.set_output(|_| {});
        if let Err(err) = e.eval(src) {
            panic!("corpus invalide :\n{}", err.render(src, "corpus"));
        }
    }
    for _ in 0..count(400) {
        let base = CORPUS[r.below(CORPUS.len())];
        let src = mutate(&mut r, base);
        run(&src);
    }
}

/// Programme plausible, surtout bien typé (pour aller loin dans l'exécution), avec une
/// part de mélanges hasardeux : déclarations, fonctions typées (JIT) et non typées,
/// objets, listes, closures, boucles.
fn program(r: &mut Rng) -> String {
    // Expression entière (variables `a`, `n` entières) ; `calls` : peut appeler `g` et
    // `h` (pas dans leur propre corps : récursion sans fin).
    fn int(r: &mut Rng, d: u32, calls: bool) -> String {
        if d == 0 {
            // Grands nombres rares : sinon presque tout déborde.
            if r.below(30) == 0 {
                return r
                    .pick(&["9223372036854775807", "-9223372036854775808"])
                    .to_string();
            }
            return r
                .pick(&[
                    "a", "n", "1", "-2", "0", "7", "3", "p.x", "xs.len()", "xs[1]",
                ])
                .to_string();
        }
        let e = |r: &mut Rng| int(r, d - 1, calls);
        match r.below(9) {
            4 | 5 if !calls => format!("({} + 1)", e(r)),
            0..=2 => format!("({} {} {})", e(r), r.pick(&["+", "-", "*", "/", "%"]), e(r)),
            3 => format!(
                "(if {} {} {} then {} else {} end)",
                e(r),
                r.pick(&["<", "==", ">="]),
                e(r),
                e(r),
                e(r)
            ),
            4 => format!("g({})", e(r)),
            5 => format!("h({}, {})", e(r), e(r)),
            6 => format!("P(x: {}).x", e(r)),
            7 => format!("(|v| v {} {})({})", r.pick(&["+", "*", "-"]), e(r), e(r)),
            _ => format!("(-{})", e(r)),
        }
    }
    // Valeur quelconque : mélanges (types différents, `nil`, chaînes…).
    fn wild(r: &mut Rng, d: u32) -> String {
        if d == 0 {
            return r
                .pick(&[
                    "a", "\"s\"", "nil", "true", "2.5", "[1]", "p", "{x: 1}", "xs",
                ])
                .to_string();
        }
        let e = |r: &mut Rng| wild(r, d - 1);
        match r.below(6) {
            0 => format!(
                "({} {} {})",
                e(r),
                r.pick(&["+", "*", "==", "<", "and", "or", "<<"]),
                e(r)
            ),
            1 => format!(
                "{}.{}",
                e(r),
                r.pick(&["x", "len()", "to_s()", "first()", "map(|v| v)", "nope()"])
            ),
            2 => format!("{}[{}]", e(r), e(r)),
            3 => format!("h({}, {})", e(r), e(r)),
            4 => format!("\"#{{{}}}\"", e(r)),
            _ => int(r, d, true),
        }
    }
    let mut out = String::from(
        "type P\n  x: int = 0\n  y = 1\nend\nlet xs = [3, 1, 4, 1, 5]\nlet p = P(x: 2)\n",
    );
    out.push_str(&format!("def g(a: int): int = {}\n", int(r, 3, false)));
    out.push_str(&format!("def h(a, n) = {}\n", int(r, 3, false)));
    out.push_str(&format!(
        "def k(ps: [P], n: int): int\n  let t = 0\n  for q in ps\n    t += {}\n  end\n  t\nend\n",
        int(r, 2, false).replace("p.x", "q.x")
    ));
    out.push_str("let a = 3\nlet n = 10\n");
    for _ in 0..2 + r.below(5) {
        let wild_line = r.below(10) == 0;
        let e = if wild_line {
            wild(r, 2)
        } else {
            int(r, 3, true)
        };
        match r.below(5) {
            0 => out.push_str(&format!("a = {e}\n")),
            1 => out.push_str(&format!(
                "for v in xs\n  n = {}\nend\n",
                e.replace('a', "v")
            )),
            2 => out.push_str(&format!("xs << {e}\n")),
            3 => out.push_str(&format!("p = P(x: {e})\n")),
            _ => out.push_str(&format!("let c = k([p, P(x: {e})], n)\n")),
        }
    }
    out.push_str(&format!("[a, n, p.x, k([p], 3), {}]", int(r, 2, true)));
    out
}

#[test]
fn generated_programs() {
    let mut r = seed(0x8EBC_6AF0_9C88_C6E3);
    for _ in 0..count(300) {
        let src = program(&mut r);
        run(&src);
    }
}
