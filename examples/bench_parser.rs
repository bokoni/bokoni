//! Compare le parseur écrit à la main avec une grammaire `peg` équivalente.
//!
//! `cargo run --release --example bench_parser`
//!
//! La grammaire peg couvre un sous-ensemble commun (let, assignation, opérateurs avec
//! précédence, appels, champs, index, listes, chaînes). Elle fait volontairement *moins*
//! de travail que le parseur Bokoni : pas de positions, pas d'interning (noms en `&str`),
//! pas de messages d'erreur détaillés. La comparaison est donc favorable à peg.

use bokoni::intern::Interner;
use std::time::Instant;

include!("common/peg_grammar.rs");
include!("common/bench_gen.rs");

fn best_of<T>(runs: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::MAX;
    let mut last = None;
    for _ in 0..runs {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed().as_secs_f64());
        last = Some(r);
    }
    (best, last.unwrap())
}

fn main() {
    for (title, names) in [
        ("noms réalistes (64 variables)", 64),
        (
            "un nom différent par ligne (défavorable à l'interning)",
            usize::MAX,
        ),
    ] {
        println!("── {title} ──");
        run(&generate(150_000, names));
        println!();
    }
}

fn run(src: &str) {
    let src = src.to_string();
    let mb = src.len() as f64 / 1e6;
    println!("entrée : {} lignes, {mb:.1} Mo\n", src.lines().count());

    let (t_lex, n_tokens) = best_of(5, || bokoni::lexer::tokenize(&src).unwrap().len());
    let (t_ours, n_ours) = best_of(5, || {
        let mut i = Interner::new();
        bokoni::parser::parse(&src, &mut i)
            .unwrap()
            .body
            .stmts
            .len()
    });
    let (t_peg, n_peg) = best_of(5, || pg::program(&src).unwrap().len());
    assert_eq!(
        n_ours, n_peg,
        "les deux parseurs doivent lire le même nombre d'instructions"
    );

    println!("{:<36} {:>9} {:>10}", "", "temps", "débit");
    println!(
        "{:<36} {:>7.1}ms {:>7.0} Mo/s   ({n_tokens} jetons)",
        "bokoni : lexer seul",
        t_lex * 1e3,
        mb / t_lex
    );
    println!(
        "{:<36} {:>7.1}ms {:>7.0} Mo/s",
        "bokoni : lexer + parseur + AST",
        t_ours * 1e3,
        mb / t_ours
    );
    println!(
        "{:<36} {:>7.1}ms {:>7.0} Mo/s",
        "peg (sans positions ni interning)",
        t_peg * 1e3,
        mb / t_peg
    );
    println!(
        "rapport : bokoni / peg = {:.2}× (> 1 : bokoni plus rapide)",
        t_peg / t_ours
    );
}
