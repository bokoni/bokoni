//! Compares the hand-written parser with an equivalent `peg` grammar.
//!
//! `cargo run --release --example bench_parser`
//!
//! The peg grammar covers a common subset (let, assignment, operators with precedence,
//! calls, fields, indexing, lists, strings). It deliberately does *less* work than the
//! Bokoni parser: no positions, no interning (names as `&str`), no detailed error
//! messages. The comparison therefore favours peg.

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
        ("realistic names (64 variables)", 64),
        (
            "a different name on every line (worst case for interning)",
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
    println!("input: {} lines, {mb:.1} MB\n", src.lines().count());

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
        "both parsers must read the same number of statements"
    );

    println!("{:<36} {:>9} {:>10}", "", "time", "throughput");
    println!(
        "{:<36} {:>7.1}ms {:>7.0} MB/s   ({n_tokens} tokens)",
        "bokoni: lexer only",
        t_lex * 1e3,
        mb / t_lex
    );
    println!(
        "{:<36} {:>7.1}ms {:>7.0} MB/s",
        "bokoni: lexer + parser + AST",
        t_ours * 1e3,
        mb / t_ours
    );
    println!(
        "{:<36} {:>7.1}ms {:>7.0} MB/s",
        "peg (no positions, no interning)",
        t_peg * 1e3,
        mb / t_peg
    );
    println!(
        "ratio: bokoni / peg = {:.2}× (> 1: bokoni is faster)",
        t_peg / t_ours
    );
}
