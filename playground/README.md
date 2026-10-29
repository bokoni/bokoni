# Bokoni playground

Write and run Bokoni in the browser: a [Monaco](https://github.com/microsoft/monaco-editor)
editor, the interpreter compiled to WebAssembly, run in a worker:
**https://bokoni.github.io/bokoni/** (published by `.github/workflows/playground.yml` on
every push to `main` that touches it).

- `wasm/`: a small Rust crate around `bokoni` (without the JIT: native code cannot be
  generated inside WebAssembly). It exports `run` (runs a script) and `check` (static
  checks only), each answering in JSON; what the script prints goes to an imported
  function as it happens. No `wasm-bindgen`.
- `web/`: the page, in TypeScript, built with [Bun](https://bun.sh) and
  [Vite](https://vite.dev).

## Use

```sh
cd playground/web
bun install
bun run dev        # builds the wasm module, then serves the page (http://localhost:5173)
bun run build      # the site in dist/ (static files, relative paths: any host)
bun run test       # the wasm module under Bun: output, errors, positions, examples
```

Needs the `wasm32-unknown-unknown` Rust target (`rustup target add wasm32-unknown-unknown`).

## What it does

- Bokoni highlighting, indentation of `… end` blocks, snippets (`def`, `class`, `case`…).
- Static errors underlined as one types (a second worker runs the checks).
- **Run** (Ctrl+Enter): printed text as it comes, the program's value, errors with
  their excerpt (a click shows the place in the editor). **Stop**, and a 10 s limit: a
  loop that never ends does not freeze the page.
- Examples, **Share** (the code, compressed, in the link), light / dark theme, a
  divider that can be dragged, a layout for phones. The code is kept in the browser.

## Limits

- Interpreter only: 5 to 60 times slower than the native code of the desktop JIT on hot
  loops (`fib(25)`: about 40 ms).
- 400 nested calls at most (`MAX_CALL_DEPTH` in `wasm/src/lib.rs`): each one takes room
  on the browser's own stack, which cannot grow (about 1 MB in a Chrome worker, where
  about 680 fit). Deeper, a Bokoni error; a crash beyond (through other built-ins) is
  caught and reported.
- No `require` (no files).

## Upgrading Monaco

`src/monaco.ts` imports the editor's features without the services of other languages
(megabytes of workers); it is generated from Monaco's own entry point:
`bun run monaco` after `bun update monaco-editor`.
