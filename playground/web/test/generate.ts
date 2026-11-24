// Random programs of typed functions, for the equivalence test of `jit.test.ts`.

/** A small deterministic random generator (mulberry32). */
function rng(seed: number) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const int = (n: number) => Math.floor(next() * n);
  const pick = <T>(xs: T[]): T => xs[int(xs.length)]!;
  return { next, int, pick };
}

type Ty = "int" | "float" | "bool";

/**
 * A program of typed functions (each may call the ones before it, and itself with a
 * decreasing budget), then a call of the last one.
 */
export function program(seed: number): string {
  const r = rng(seed);
  const tys: Ty[] = ["int", "float", "bool"];
  const fns: { name: string; params: Ty[]; ret: Ty }[] = [];
  // (As text: a JavaScript number loses the low digits of the large ones.)
  const ints = ["0", "1", "2", "3", "-1", "-7", "10", "1000", "4611686018427387904", "9223372036854775807", "-9223372036854775807"];
  const floats = ["0.0", "1.5", "-2.25", "1000000.0", "0.1", "3.0"];

  const out: string[] = [];
  const nf = 1 + r.int(4);
  for (let f = 0; f < nf; f++) {
    const params: Ty[] = ["int", ...Array.from({ length: r.int(3) }, () => r.pick(tys))];
    const ret = r.pick(tys);
    const name = `f${f}`;
    const vars: { name: string; ty: Ty }[] = params.map((ty, i) => ({ name: `p${i}`, ty }));
    const self = { name, params, ret };

    const expr = (ty: Ty, depth: number): string => {
      const of = vars.filter((v) => v.ty === ty);
      const leaf = (): string => {
        if (of.length && r.next() < 0.6) return r.pick(of).name;
        if (ty === "int") return r.pick(ints);
        if (ty === "float") return r.pick(floats);
        return r.pick(["true", "false"]);
      };
      if (depth <= 0) return leaf();
      const d = depth - 1;
      const callable = fns.filter((g) => g.ret === ty);
      const k = r.next();
      if (k < 0.15 && callable.length) {
        const g = r.pick(callable);
        return `${g.name}(${g.params.map((p) => expr(p, d)).join(", ")})`;
      }
      if (k < 0.22 && ret === ty) {
        // Itself, on a smaller budget (two levels at most: the calls stay few).
        const rest = params.slice(1).map((p) => expr(p, 0));
        return `(if p0 > 0 and p0 < 3 then ${name}(${["p0 - 1", ...rest].join(", ")}) else ${leaf()} end)`;
      }
      if (k < 0.32) return `(if ${expr("bool", d)} then ${expr(ty, d)} else ${expr(ty, d)} end)`;
      if (ty === "int") {
        if (r.next() < 0.1) return `(-(${expr("int", d)}))`;
        return `(${expr("int", d)} ${r.pick(["+", "-", "*", "/", "%"])} ${expr("int", d)})`;
      }
      if (ty === "float") {
        if (r.next() < 0.1) return `(-(${expr("float", d)}))`;
        const [a, b] = r.pick([["float", "float"], ["int", "float"], ["float", "int"]] as [Ty, Ty][]);
        return `(${expr(a, d)} ${r.pick(["+", "-", "*", "/", "%"])} ${expr(b, d)})`;
      }
      const c = r.next();
      if (c < 0.2) return `(not ${expr("bool", d)})`;
      if (c < 0.4) return `(${expr("bool", d)} ${r.pick(["and", "or", "==", "!="])} ${expr("bool", d)})`;
      const [a, b] = r.pick([["int", "int"], ["float", "float"], ["int", "float"]] as [Ty, Ty][]);
      return `(${expr(a, d)} ${r.pick(["<", "<=", ">", ">=", "==", "!="])} ${expr(b, d)})`;
    };

    const body: string[] = [];
    // Locals, a bounded loop updating them.
    const nl = r.int(3);
    for (let i = 0; i < nl; i++) {
      const ty = r.pick(tys);
      body.push(`  let v${i} = ${expr(ty, 2)}`);
      vars.push({ name: `v${i}`, ty });
    }
    if (nl && r.next() < 0.6) {
      body.push(`  let k = 0`);
      body.push(`  while k < ${1 + r.int(3)}`);
      const v = vars[vars.length - 1 - r.int(nl)]!;
      body.push(`    ${v.name} = ${expr(v.ty, 2)}`);
      body.push(`    k += 1`);
      body.push(`  end`);
    }
    body.push(`  ${expr(ret, 3)}`);
    // Annotations left out at times: the result inferred, a copy typed for the arguments.
    const bare = r.next() < 0.2;
    const sig = params.map((t, i) => (bare && i > 0 ? `p${i}` : `p${i}: ${t}`)).join(", ");
    const result = r.next() < 0.35 ? "" : `: ${ret}`;
    out.push(`def ${name}(${sig})${result}\n${body.join("\n")}\nend`);
    fns.push(self);
  }
  const last = fns[fns.length - 1]!;
  const arg = (t: Ty) =>
    t === "int" ? `${r.pick([0, 1, 2, 3, 5, -4])}` : t === "float" ? r.pick(floats) : r.pick(["true", "false"]);
  out.push(`${last.name}(${last.params.map(arg).join(", ")})`);
  return out.join("\n");
}
