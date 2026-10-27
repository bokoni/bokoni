//! `bokoni`: runs a script or an expression, or starts the REPL.

use bokoni::{Engine, Value};
use std::io::{self, BufRead, IsTerminal, Write};
use std::process::ExitCode;

const USAGE: &str = "\
usage:
  bokoni                    REPL
  bokoni file.bk            runs a file
  bokoni -e \"code\"          runs code and prints the result
  bokoni --ast file.bk      prints the syntax tree (S-expressions)
  bokoni --ir file.bk       prints the IR (what the interpreter runs)
  (--ast and --ir also accept -e \"code\")
  --no-jit                  no native compilation (interpreter only)";

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
                None => return usage_error("`-e` expects code"),
            },
            path if !path.starts_with('-') && code.is_none() => match std::fs::read_to_string(path)
            {
                Ok(src) => code = Some((src, path.to_string())),
                Err(e) => {
                    eprintln!("cannot read {path}: {e}");
                    return ExitCode::FAILURE;
                }
            },
            other => return usage_error(&format!("unknown argument `{other}`")),
        }
    }

    let mut engine = Engine::new();
    // Warnings (`@[Deprecated]` declarations…): printed before execution.
    engine.set_warning_handler(|w| eprint!("{}", w.render("", "")));
    engine.set_jit(jit);
    match code {
        None if ast || ir => usage_error("`--ast` / `--ir` expect a file or `-e`"),
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
                // With `-e`, the result is printed (unless nil); a file does not print it.
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
            "bokoni {} — `:q` or Ctrl-D to quit",
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
            // Construct not finished yet (`def` without `end`…): read the next line.
            Err(e) if e.is_incomplete() => {
                n -= 1;
                continue;
            }
            Err(e) => eprint!("{}", e.render(&buffer, &name)),
        }
        buffer.clear();
    }
}
