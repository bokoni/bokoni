// The programs of the Rust tests (`tests/*.rs`, as string literals) and of the
// specification (`docs/syntax.md`), run interpreted and with every function compiled to
// WebAssembly from its first call (with structured control flow, then with the dispatch
// loop): the same answers (errors included).

import { beforeAll, expect, test } from "bun:test";
import { Glob } from "bun";
import { Bokoni } from "../src/bokoni";

let module: WebAssembly.Module;
let bk: Bokoni;
let printed = "";

const fresh = async () => {
  bk = await Bokoni.instantiate(module, (t) => (printed += t));
};

beforeAll(async () => {
  const bytes = await Bun.file(new URL("../src/generated/bokoni.wasm", import.meta.url)).arrayBuffer();
  module = await WebAssembly.compile(bytes);
  await fresh();
});

const root = new URL("../../../", import.meta.url).pathname;

/** The string literals of a Rust source (plain and raw), unescaped. */
function literals(src: string): string[] {
  const out: string[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i]!;
    if (c === "/" && src[i + 1] === "/") {
      i = src.indexOf("\n", i);
      if (i < 0) break;
      continue;
    }
    if (c === "'" ) {
      // A character literal ('"', '\\'…) or a lifetime.
      const m = /^'(\\.|[^\\'])'/.exec(src.slice(i, i + 4)) ?? /^'\\u\{[0-9a-f]+\}'/.exec(src.slice(i, i + 12));
      i += m ? m[0].length : 1;
      continue;
    }
    const raw = /^r(#*)"/.exec(src.slice(i, i + 10));
    if (raw && !/[A-Za-z0-9_]/.test(src[i - 1] ?? "")) {
      const end = '"' + raw[1];
      const start = i + raw[0].length;
      const j = src.indexOf(end, start);
      out.push(src.slice(start, j));
      i = j + end.length;
      continue;
    }
    if (c === '"') {
      let s = "";
      i++;
      while (i < src.length && src[i] !== '"') {
        if (src[i] === "\\") {
          const e = src[i + 1]!;
          if (e === "n") s += "\n";
          else if (e === "t") s += "\t";
          else if (e === "r") s += "\r";
          else if (e === "0") s += "\0";
          else if (e === "\n") {
            // A continued line: the next one's leading spaces dropped.
            i += 2;
            while (/\s/.test(src[i]!)) i++;
            continue;
          } else if (e === "u") {
            const m = /^\\u\{([0-9a-fA-F]+)\}/.exec(src.slice(i, i + 12))!;
            s += String.fromCodePoint(parseInt(m[1]!, 16));
            i += m[0].length;
            continue;
          } else if (e === "x") {
            s += String.fromCharCode(parseInt(src.slice(i + 2, i + 4), 16));
            i += 4;
            continue;
          } else s += e;
          i += 2;
        } else s += src[i++];
      }
      out.push(s);
      i++;
      continue;
    }
    i++;
  }
  return out;
}

/** Programs: literals that look like Bokoni code. */
async function corpus(): Promise<string[]> {
  const progs = new Set<string>();
  for await (const f of new Glob("tests/*.rs").scan(root)) {
    // (The fuzzer's: fragments for its generator.)
    if (f.endsWith("fuzz.rs")) continue;
    for (const s of literals(await Bun.file(root + f).text())) {
      if (s.length >= 8 && /\b(def|let|class|type|enum|while|for|case|if)\b|\n/.test(s)) progs.add(s);
    }
  }
  const md = await Bun.file(root + "docs/syntax.md").text();
  for (const m of md.matchAll(/```ruby\n([\s\S]*?)```/g)) progs.add(m[1]!);
  return [...progs];
}

test("the programs of the Rust tests and of syntax.md: the same answers", async () => {
  const progs = await corpus();
  let compiled = 0;
  let dispatched = 0;
  // Programs with a function the translation into WebAssembly does not know.
  const untranslated: string[] = [];
  for (const src of progs) {
    // The browser's own stack overflowing (very deep data, recursion through
    // built-ins): a crash, after which the instance is replaced (as the playground does).
    const run = async (mode: boolean | "eager" | "dispatch") => {
      bk.setJit(mode);
      printed = "";
      try {
        return { answer: bk.run(src), printed };
      } catch (e) {
        await fresh();
        return { crash: String(e) };
      }
    };
    const interpreted = await run(false);
    const before = bk.baseline + bk.compiled;
    const unstructured = bk.dispatched;
    const failures = bk.untranslated;
    const eager = await run("eager");
    compiled += Math.max(0, bk.baseline + bk.compiled - before);
    dispatched += Math.max(0, bk.dispatched - unstructured);
    if (bk.untranslated > failures) untranslated.push(src);
    for (const got of [eager, await run("dispatch")]) {
      if (JSON.stringify(got) !== JSON.stringify(interpreted)) {
        expect({ src, ...got }).toEqual({ src, ...interpreted });
      }
    }
  }
  expect(progs.length).toBeGreaterThan(500);
  // Not vacuous: most programs compile functions (many literals are fragments).
  expect(compiled).toBeGreaterThan(progs.length / 2);
  // Bokoni's control flow graphs are reducible: all structured.
  expect(dispatched).toBe(0);
  // Every function the JIT builds is translated.
  expect(untranslated.slice(0, 3)).toEqual([]);
}, 120_000);
