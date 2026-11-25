// Functions compiled to WebAssembly (`bokoni::wasm_jit`) against the interpreter: the
// same answer (value, errors, positions, call stack) for each program, run interpreted,
// compiled as usual, and with every function compiled from its first call (`eager`: the
// baseline code of the untyped ones).

import { beforeAll, describe, expect, test } from "bun:test";
import { Bokoni } from "../src/bokoni";
import { examples } from "../src/examples";
import { program } from "./generate";

let bk: Bokoni;
let printed = "";

beforeAll(async () => {
  const bytes = await Bun.file(new URL("../src/generated/bokoni.wasm", import.meta.url)).arrayBuffer();
  bk = await Bokoni.instantiate(await WebAssembly.compile(bytes), (t) => (printed += t));
});

/**
 * Runs `src` interpreted, compiled, then compiled eagerly: the answers and what each
 * printed; the functions compiled as usual, typed and to baseline code.
 */
function both(src: string) {
  const run = (mode: boolean | "eager") => {
    bk.setJit(mode);
    printed = "";
    return { answer: bk.run(src), printed };
  };
  const interpreted = run(false);
  const [typed, base] = [bk.compiled, bk.baseline];
  const compiled = run(true);
  const [functions, baseline] = [bk.compiled - typed, bk.baseline - base];
  return { interpreted, compiled, eager: run("eager"), functions, baseline };
}

/** The same answers; returns the number of typed functions compiled. */
function same(src: string): number {
  const r = both(src);
  expect(r.compiled).toEqual(r.interpreted);
  expect(r.eager).toEqual(r.interpreted);
  return r.functions;
}

const fib = "def fib(n: int): int = if n < 2 then n else fib(n - 1) + fib(n - 2) end\n";

describe("compiled", () => {
  test("a recursive function", () => {
    expect(same(fib + "fib(20)")).toBe(1);
    expect(both(fib + "fib(20)").compiled.answer.value).toBe("6765");
  });

  test("loops, floats, mixed arithmetic", () => {
    const src = `
def harm(n: int): float
  let s = 0.0
  let i = 1
  while i <= n
    s = s + 1.0 / i
    i += 1
  end
  s
end
def sum(n: int): int
  let t = 0
  for i in 0..n
    t += i % 7 - i / 3
  end
  t
end
def mix(a: int, b: float): float
  let x = a
  let y = b
  x * y - x / 2 + y % 3.0
end
[harm(1000), sum(1000), mix(7, -2.5), mix(-7, 0.0)]`;
    expect(same(src)).toBe(3);
  });

  test("booleans, comparisons, `not`, `and` / `or`", () => {
    const src = `
def between(x: float, lo: float, hi: float): bool = x >= lo and x <= hi
def odd(n: int): bool = n % 2 == 1
def pick(b: bool, n: int): int = if not b or n > 3 then n else -n end
def same(a: bool, b: bool): bool = a == b and not (a != b)
[between(1.5, 1.0, 2.0), between(0.0 / 0.0, 0.0, 1.0), odd(-3), pick(true, 2),
 pick(false, 1), same(true, false)]`;
    expect(same(src)).toBe(4);
  });

  test("calls between compiled functions, mutual recursion", () => {
    const src = `
def even(n: int): bool = if n == 0 then true else odd(n - 1) end
def odd(n: int): bool = if n == 0 then false else even(n - 1) end
def sq(x: float): float = x * x
def norm(a: float, b: float): float = sq(a) + sq(b)
[even(100), odd(7), norm(3.0, 4.0)]`;
    expect(same(src)).toBeGreaterThanOrEqual(3);
  });

  test("parameters without annotations: a copy typed for the arguments", () => {
    expect(same("def add(a, b): int = a + b\ndef f(n: int): int = add(n, 1) * 2\nf(20)")).toBe(2);
  });

  test("no annotation at all: copies typed for the arguments", () => {
    expect(same("def add(a, b) = a + b\ndef f(n: int): int = add(n, 1) * 2\nf(20)")).toBe(2);
    expect(same("def fib(n) = if n < 2 then n else fib(n - 1) + fib(n - 2) end\nfib(20)")).toBe(1);
    expect(same("def half(x) = x / 2.0\ndef f(n) = half(n) + half(n + 1)\n[f(3), f(4)]")).toBeGreaterThan(0);
  });

  test("mutual recursion without annotations: baseline code", () => {
    const src = "def even(n) = if n == 0 then true else odd(n - 1) end\n" +
      "def odd(n) = if n == 0 then false else even(n - 1) end\n[even(10), odd(7)]";
    same(src);
    expect(both(src).baseline).toBe(2);
  });

  test("results of different types: baseline code", () => {
    // (An `int` and a `float`: no one native type, unlike `: float`, which converts.)
    expect(same("def f(n) = if n > 0 then 1 else 2.5 end\n[f(1), f(-1)]")).toBe(0);
    expect(same("def f(n): float = n * 2\n[f(1), f(-1)]")).toBe(1);
    expect(same("def f(n) = if n > 0 then f(n - 1) else 0.5 end\nf(3)")).toBe(0);
    expect(both("def f(n) = if n > 0 then f(n - 1) else 0.5 end\nf(3)").baseline).toBe(1);
  });

  test("more than sixteen arguments", () => {
    const ps = Array.from({ length: 18 }, (_, i) => `a${i}: int`).join(", ");
    const sum = Array.from({ length: 18 }, (_, i) => `a${i}`).join(" + ");
    const args = Array.from({ length: 18 }, (_, i) => `${i}`).join(", ");
    expect(same(`def s(${ps}): int = ${sum}\ns(${args})`)).toBe(1);
  });

  test("redefining a function called by compiled code (directly or not)", () => {
    for (const jit of [false, true]) {
      bk.setJit(jit);
      const before = bk.compiled;
      bk.runAgain(`
def f(n: int): int = n + 1
def g(n: int): int = f(n) * 10
def h(n: int): int = g(n) + 1
def k(n: int): int = n * 2`);
      expect(bk.runAgain("[h(1), k(4)]").value).toBe("[21, 8]");
      bk.runAgain("def f(n: int): int = n + 2");
      expect(bk.runAgain("[h(1), k(4)]").value).toBe("[31, 8]");
      if (jit) expect(bk.compiled - before).toBe(7);
    }
  });
});

describe("errors, as the interpreter gives them", () => {
  const cases: Record<string, string> = {
    "`+` overflow": "def f(a: int, b: int): int = a + b\nf(9223372036854775807, 1)",
    "`-` overflow": "def f(a: int, b: int): int = a - b\nf(-9223372036854775807, 2)",
    "`*` overflow": "def f(a: int, b: int): int = a * b\nf(4611686018427387904, 2)",
    "`*` overflow, -1": "def f(a: int, b: int): int = a * b\nf(-1, -9223372036854775807 - 1)",
    "`*` no overflow": "def f(a: int, b: int): int = a * b\n[f(-1, 9223372036854775807), f(0, -9223372036854775807 - 1), f(-3, 7)]",
    "negation overflow": "def f(a: int): int = -a\nf(-9223372036854775807 - 1)",
    "division by zero": "def f(a: int, b: int): int = a / b\nf(1, 0)",
    "remainder by zero": "def f(a: int, b: int): int = a % b\nf(1, 0)",
    "MIN / -1": "def f(a: int, b: int): int = a / b\nf(-9223372036854775807 - 1, -1)",
    "MIN % -1": "def f(a: int, b: int): int = a % b\nf(-9223372036854775807 - 1, -1)",
    "Euclidean division":
      "def d(a: int, b: int): int = a / b\ndef r(a: int, b: int): int = a % b\n" +
      "[d(7, 2), d(-7, 2), d(7, -2), d(-7, -2), r(7, 2), r(-7, 2), r(7, -2), r(-7, -2), r(-1, -9223372036854775807 - 1)]",
    "float remainder": "def r(a: float, b: float): float = a % b\n[r(7.5, 2.0), r(-7.5, 2.0), r(7.5, -2.0), r(1.0, 0.0)]",
    "an error deep in compiled calls":
      "def a(n: int): int = 10 / n\ndef b(n: int): int = a(n - 1) + 1\ndef c(n: int): int = b(n) * 2\nc(1)",
    "too many nested calls": "def down(n: int): int = if n == 0 then 0 else 1 + down(n - 1) end\ndown(100000)",
    "just within the call depth": "def down(n: int): int = if n == 0 then 0 else 1 + down(n - 1) end\ndown(390)",
  };
  for (const [name, src] of Object.entries(cases)) {
    test(name, () => {
      expect(same(src)).toBeGreaterThan(0);
    });
  }
});

test("every example: the same output", () => {
  for (const ex of examples) {
    const r = both(ex.code);
    expect({ name: ex.name, ...r.compiled }).toEqual({ name: ex.name, ...r.interpreted });
    expect({ name: ex.name, ...r.eager }).toEqual({ name: ex.name, ...r.interpreted });
  }
});

test("generated programs: the same answer", () => {
  const n = Number(process.env.BK_GEN ?? 400);
  let compiled = 0;
  let errors = 0;
  for (let seed = 1; seed <= n; seed++) {
    const src = program(seed);
    const r = both(src);
    for (const got of [r.compiled, r.eager]) {
      if (JSON.stringify(got) !== JSON.stringify(r.interpreted)) {
        // The program, to reproduce it.
        expect({ seed, src, ...got }).toEqual({ seed, src, ...r.interpreted });
      }
    }
    compiled += r.functions;
    if (r.interpreted.answer.diagnostics.length) errors++;
  }
  // Not vacuous: most programs compile, some fail at run time.
  expect(compiled).toBeGreaterThan(n);
  expect(errors).toBeGreaterThan(n / 20);
}, 600_000);
