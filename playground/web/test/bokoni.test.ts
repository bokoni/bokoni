// The WebAssembly interpreter, as the worker uses it (`bun test`, after `bun run wasm`).

import { beforeAll, describe, expect, test } from "bun:test";
import { Bokoni } from "../src/bokoni";
import { examples } from "../src/examples";
import { decodeShared, encodeShared } from "../src/share";

let module: WebAssembly.Module;
let printed = "";
let bk: Bokoni;

beforeAll(async () => {
  const bytes = await Bun.file(new URL("../src/generated/bokoni.wasm", import.meta.url)).arrayBuffer();
  module = await WebAssembly.compile(bytes);
  bk = await Bokoni.instantiate(module, (t) => (printed += t));
});

const run = (src: string) => {
  printed = "";
  return bk.run(src);
};

describe("run", () => {
  test("value and output", () => {
    const a = run('print("héllo #{1 + 1}")\n[1, 2].map(|x| x * 10)');
    expect(a.value).toBe("[10, 20]");
    expect(a.diagnostics).toEqual([]);
    expect(printed).toBe("héllo 2\n");
  });

  test("a type error, located (UTF-16 columns)", () => {
    const a = run('let s = "😀"\ndef f(x: int): int = x\nf(s)');
    expect(a.value).toBeNull();
    const [d] = a.diagnostics;
    expect(d).toMatchObject({ severity: "error", kind: "type", line: 3, col: 3, endLine: 3, endCol: 4 });
    expect(d!.rendered).toContain("main.bk:3:3");
  });

  test("a runtime error, with its call stack", () => {
    const a = run("def d(a, b) = a / b\nd(1, 0)");
    expect(a.diagnostics[0]).toMatchObject({ kind: "runtime", line: 1 });
    expect(a.diagnostics[0]!.rendered).toContain("called from");
  });

  test("a syntax error", () => {
    expect(run("def f(\n").diagnostics[0]!.kind).toBe("syntax");
  });

  test("deep recursion: an error, not a crash", () => {
    const a = run("def down(n) = if n == 0 then 0 else 1 + down(n - 1) end\ndown(1000000)");
    expect(a.diagnostics[0]!.message).toContain("too many nested calls (400)");
    // The instance still works.
    expect(run("1 + 1").value).toBe("2");
  });

  test("recursion within the limit", () => {
    expect(run("def down(n) = if n == 0 then 0 else 1 + down(n - 1) end\ndown(350)").value).toBe("350");
  });

  test("every example runs without errors", () => {
    for (const ex of examples) {
      const a = run(ex.code);
      expect({ name: ex.name, diagnostics: a.diagnostics }).toEqual({ name: ex.name, diagnostics: [] });
    }
  });
});

describe("check", () => {
  test("static errors without running", () => {
    printed = "";
    const a = bk.check('print("not run")\nlet x: int = "a"');
    expect(printed).toBe("");
    expect(a.value).toBeNull();
    expect(a.diagnostics[0]).toMatchObject({ kind: "type", line: 2 });
  });

  test("a correct program", () => {
    expect(bk.check("def f(x: int): int = x * 2\nf(2)").diagnostics).toEqual([]);
  });
});

test("shared links decode to the same code", async () => {
  const code = examples[1]!.code + "\n# é 😀";
  const link = await encodeShared(code);
  expect(link).toMatch(/^[A-Za-z0-9_-]+$/);
  expect(await decodeShared(link)).toBe(code);
});
