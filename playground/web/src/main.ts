// The playground page: a Monaco editor, Bokoni run in a worker, its output beside.

import * as monaco from "./monaco";
import EditorWorker from "monaco-editor/editor/editor.worker?worker";
import "./style.css";
import type { Answer, Diagnostic } from "./bokoni";
import { examples } from "./examples";
import { defineThemes, LANGUAGE, registerBokoni } from "./language";
import { Runner, type Outcome } from "./runner";
import { decodeShared, encodeShared } from "./share";

self.MonacoEnvironment = { getWorker: () => new EditorWorker() };

const RUN_TIMEOUT = 10_000;
const CHECK_DELAY = 300;
const STORE = "bokoni.playground.code";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const runButton = $<HTMLButtonElement>("run");
const shareButton = $<HTMLButtonElement>("share");
const themeButton = $<HTMLButtonElement>("theme");
const exampleSelect = $<HTMLSelectElement>("examples");
const output = $<HTMLDivElement>("output");
const runStatus = $<HTMLSpanElement>("run-status");
const problems = $<HTMLSpanElement>("problems");
const cursor = $<HTMLSpanElement>("cursor");

// ─── Storage (may be unavailable: private windows, blocked site data)

function load(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function save(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Not kept: the page works without it.
  }
}

// ─── Theme: the system's, or the one chosen

type Theme = "auto" | "light" | "dark";

const svg = (body: string) =>
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
const icons: Record<Theme, string> = {
  auto: svg('<circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 0 0 18z" fill="currentColor"/>'),
  light: svg(
    '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  ),
  dark: svg('<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>'),
};
const dark = matchMedia("(prefers-color-scheme: dark)");
let theme: Theme = (load("bokoni.playground.theme") as Theme | null) ?? "auto";

function applyTheme() {
  const isDark = theme === "dark" || (theme === "auto" && dark.matches);
  if (theme === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  monaco.editor.setTheme(isDark ? "bokoni-dark" : "bokoni-light");
  themeButton.title = `Theme: ${theme}`;
  themeButton.innerHTML = icons[theme];
}

themeButton.onclick = () => {
  theme = theme === "auto" ? "light" : theme === "light" ? "dark" : "auto";
  save("bokoni.playground.theme", theme);
  applyTheme();
};
dark.addEventListener("change", applyTheme);

// ─── Editor

registerBokoni(monaco);
defineThemes(monaco);

const editor = monaco.editor.create($("editor"), {
  language: LANGUAGE,
  automaticLayout: true,
  fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  fontLigatures: true,
  fontSize: 14,
  lineHeight: 22,
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  tabSize: 2,
  padding: { top: 12 },
  renderLineHighlight: "all",
  smoothScrolling: true,
  bracketPairColorization: { enabled: true },
});
const model = editor.getModel()!;
applyTheme();

for (const [i, ex] of examples.entries()) {
  exampleSelect.add(new Option(ex.name, String(i)));
}
exampleSelect.onchange = () => {
  const ex = examples[Number(exampleSelect.value)];
  if (!ex) return;
  model.setValue(ex.code);
  history.replaceState(null, "", location.pathname);
  clearOutput();
  editor.focus();
};

// ─── Diagnostics: as one types (static checks), and those of the last run

const checker = new Runner(2_000);
let checkTimer: ReturnType<typeof setTimeout> | undefined;

function markers(diags: Diagnostic[]) {
  monaco.editor.setModelMarkers(
    model,
    "bokoni",
    diags
      .filter((d) => d.line !== undefined)
      .map((d) => ({
        severity:
          d.severity === "error" ? monaco.MarkerSeverity.Error : monaco.MarkerSeverity.Warning,
        message: d.message,
        startLineNumber: d.line!,
        startColumn: d.col!,
        endLineNumber: d.endLine!,
        endColumn: d.endCol!,
      })),
  );
  const errors = diags.filter((d) => d.severity === "error").length;
  const warnings = diags.length - errors;
  problems.className = errors ? "bad" : warnings ? "warn" : "good";
  problems.textContent =
    errors || warnings
      ? [plural(errors, "error"), plural(warnings, "warning")].filter(Boolean).join(", ")
      : "✓ No problems";
}

function plural(n: number, word: string): string {
  return n === 0 ? "" : `${n} ${word}${n > 1 ? "s" : ""}`;
}

async function check() {
  const src = model.getValue();
  const o = await checker.request("check", src);
  // (A newer text replaces this result.)
  if (o.kind === "done" && model.getValue() === src) markers(o.answer.diagnostics);
}

model.onDidChangeContent(() => {
  save(STORE, model.getValue());
  clearTimeout(checkTimer);
  checkTimer = setTimeout(check, CHECK_DELAY);
});

editor.onDidChangeCursorPosition((e) => {
  cursor.textContent = `Ln ${e.position.lineNumber}, Col ${e.position.column}`;
});

// ─── Running

const runner = new Runner(RUN_TIMEOUT);

function clearOutput() {
  output.replaceChildren(hint());
  runStatus.textContent = "";
  runStatus.className = "";
}

function hint(): HTMLElement {
  const p = document.createElement("p");
  p.className = "hint";
  p.innerHTML = "Press <kbd>Run</kbd> or <kbd>Ctrl</kbd> + <kbd>Enter</kbd> to run the program.";
  return p;
}

function block(tag: string, className: string, text: string): HTMLElement {
  const el = document.createElement(tag);
  el.className = className;
  el.textContent = text;
  return el;
}

function reveal(d: Diagnostic) {
  if (d.line === undefined) return;
  editor.setSelection({
    startLineNumber: d.line,
    startColumn: d.col!,
    endLineNumber: d.endLine!,
    endColumn: d.endCol!,
  });
  editor.revealLineInCenterIfOutsideViewport(d.line);
  editor.focus();
}

function show(answer: Answer, printed: boolean) {
  // (`nil` after printed text: the value of a final `print`, not worth showing.)
  if (answer.value !== null && !(answer.value === "nil" && printed)) {
    const v = block("div", "value", "");
    v.append(block("span", "arrow", "⇒ "), block("span", "", answer.value));
    output.append(v);
  }
  for (const d of answer.diagnostics) {
    const el = block("pre", `diag ${d.severity}`, d.rendered);
    if (d.line !== undefined) {
      el.title = "Show in the editor";
      el.onclick = () => reveal(d);
    }
    output.append(el);
  }
  markers(answer.diagnostics);
}

function status(o: Outcome) {
  const time = (ms: number) => (ms < 1000 ? `${ms.toFixed(ms < 10 ? 1 : 0)} ms` : `${(ms / 1000).toFixed(2)} s`);
  switch (o.kind) {
    case "done": {
      const failed = o.answer.diagnostics.some((d) => d.severity === "error");
      runStatus.className = failed ? "bad" : "good";
      runStatus.textContent = `${failed ? "Error after" : "Finished in"} ${time(o.ms)}`;
      break;
    }
    case "crash":
      runStatus.className = "bad";
      runStatus.textContent = "Stopped";
      output.append(block("pre", "diag error", o.message));
      break;
    case "stopped":
      runStatus.className = "warn";
      runStatus.textContent =
        o.reason === "timeout" ? `Stopped after ${RUN_TIMEOUT / 1000} s` : "Stopped";
      break;
  }
}

async function run() {
  if (runner.busy) {
    runner.stop();
    return;
  }
  output.replaceChildren();
  runStatus.className = "busy";
  runStatus.textContent = "Running…";
  runButton.classList.add("running");
  runButton.querySelector(".label")!.textContent = "Stop";
  let consoleEl: HTMLElement | undefined;
  const o = await runner.request("run", model.getValue(), (text) => {
    consoleEl ??= output.appendChild(block("pre", "console", ""));
    consoleEl.append(text);
  });
  runButton.classList.remove("running");
  runButton.querySelector(".label")!.textContent = "Run";
  if (o.kind === "done") show(o.answer, consoleEl !== undefined);
  status(o);
  if (!output.childElementCount) output.append(block("p", "hint", "(no output)"));
}

runButton.onclick = run;
editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, run);
window.addEventListener("keydown", (e: KeyboardEvent) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    run();
  }
});
$("clear").onclick = clearOutput;

// ─── Sharing: the code in the link

shareButton.onclick = async () => {
  const url = `${location.origin}${location.pathname}#code=${await encodeShared(model.getValue())}`;
  history.replaceState(null, "", url);
  try {
    await navigator.clipboard.writeText(url);
    toast("Link copied");
  } catch {
    toast("Link in the address bar");
  }
};

function toast(text: string) {
  const el = block("div", "toast", text);
  document.body.append(el);
  setTimeout(() => el.remove(), 1800);
}

// ─── Splitter between the editor and the output

const main = $("main");
const divider = $("divider");
const split = Number(load("bokoni.playground.split")) || 0.58;
main.style.setProperty("--split", String(split));
divider.onpointerdown = (e) => {
  divider.setPointerCapture(e.pointerId);
  divider.classList.add("dragging");
  divider.onpointermove = (m) => {
    const r = main.getBoundingClientRect();
    const f = Math.min(0.85, Math.max(0.2, (m.clientX - r.left) / r.width));
    main.style.setProperty("--split", f.toFixed(3));
  };
  divider.onpointerup = () => {
    divider.onpointermove = null;
    divider.classList.remove("dragging");
    save("bokoni.playground.split", main.style.getPropertyValue("--split"));
  };
};

// ─── Initial program: a shared link, the last one written, or the first example

async function start() {
  const shared = location.hash.startsWith("#code=")
    ? await decodeShared(location.hash.slice(6)).catch(() => null)
    : null;
  const code = shared ?? load(STORE) ?? examples[0]!.code;
  model.setValue(code);
  const i = examples.findIndex((e) => e.code === code);
  if (i < 0) exampleSelect.add(new Option("Your program", ""), 0);
  exampleSelect.value = i >= 0 ? String(i) : "";
  clearOutput();
  editor.focus();
}

start();
