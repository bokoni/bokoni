//! `bokoni` : exécute un script, une expression, ou lance le REPL.

use bokoni::{Engine, Value};
use std::io::{self, BufRead, IsTerminal, Write};
use std::process::ExitCode;

const USAGE: &str = "\
usage :
  bokoni                    REPL
  bokoni fichier.bk         exécute un fichier
  bokoni -e \"code\"          exécute du code et affiche le résultat
  bokoni --ast fichier.bk   affiche l'arbre syntaxique (S-expressions)
  bokoni --ir fichier.bk    affiche l'IR (ce qu'exécute l'interpréteur)
  (--ast et --ir acceptent aussi -e \"code\")
  --no-jit                  sans compilation native (interpréteur seul)";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut ast = false;
    let mut ir = false;
    let mut jit = true;
    let mut code: Option<(String, String)> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--ast" => ast = true,
            "--ir" => ir = true,
            "--no-jit" => jit = false,
            "-e" => match it.next() {
                Some(src) => code = Some((src.clone(), "<-e>".into())),
                None => return usage_error("`-e` attend du code"),
            },
            path if !path.starts_with('-') && code.is_none() => match std::fs::read_to_string(path)
            {
                Ok(src) => code = Some((src, path.to_string())),
                Err(e) => {
                    eprintln!("impossible de lire {path} : {e}");
                    return ExitCode::FAILURE;
                }
            },
            other => return usage_error(&format!("argument inconnu `{other}`")),
        }
    }

    let mut engine = Engine::new();
    // Avertissements (déclarations `@[Deprecated]`…) : affichés avant l'exécution.
    engine.set_warning_handler(|w| eprint!("{}", w.render("", "")));
    engine.set_jit(jit);
    match code {
        None if ast || ir => usage_error("`--ast` / `--ir` attendent un fichier ou `-e`"),
        Some((src, name)) if ir => match engine.ir(&src) {
            Ok(s) => {
                print!("{s}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprint!("{}", e.render(&src, &name));
                ExitCode::FAILURE
            }
        },
        None => repl(&mut engine),
        Some((src, name)) if ast => match engine.ast(&src) {
            Ok(s) => {
                print!("{s}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprint!("{}", e.render(&src, &name));
                ExitCode::FAILURE
            }
        },
        Some((src, name)) => match engine.eval_named(&src, &name) {
            Ok(v) => {
                // Avec `-e`, le résultat est affiché (sauf nil) ; un fichier ne l'affiche pas.
                if name == "<-e>" && !matches!(v, Value::Nil) {
                    println!("{}", engine.inspect(&v));
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprint!("{}", e.render(&src, &name));
                ExitCode::FAILURE
            }
        },
    }
}

fn usage_error(msg: &str) -> ExitCode {
    eprintln!("{msg}\n\n{USAGE}");
    ExitCode::from(2)
}

fn repl(engine: &mut Engine) -> ExitCode {
    let interactive = io::stdin().is_terminal();
    if interactive {
        println!(
            "bokoni {} — `:q` ou Ctrl-D pour quitter",
            env!("CARGO_PKG_VERSION")
        );
    }
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut buffer = String::new();
    let mut n = 0;
    loop {
        if interactive {
            print!("{}", if buffer.is_empty() { ">> " } else { ".. " });
            let _ = io::stdout().flush();
        }
        let Some(Ok(line)) = lines.next() else {
            if interactive {
                println!();
            }
            return ExitCode::SUCCESS;
        };
        if buffer.is_empty() && matches!(line.trim(), ":q" | ":quit" | "exit") {
            return ExitCode::SUCCESS;
        }
        buffer.push_str(&line);
        buffer.push('\n');
        if buffer.trim().is_empty() {
            buffer.clear();
            continue;
        }
        n += 1;
        let name = format!("<repl:{n}>");
        match engine.eval_named(&buffer, &name) {
            Ok(Value::Nil) => {}
            Ok(v) => println!("=> {}", engine.inspect(&v)),
            // Construction pas encore terminée (`def` sans `end`…) : on lit la suite.
            Err(e) if e.is_incomplete() => {
                n -= 1;
                continue;
            }
            Err(e) => eprint!("{}", e.render(&buffer, &name)),
        }
        buffer.clear();
    }
}
