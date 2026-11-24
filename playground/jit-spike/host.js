// The host side of the spike: `bk_compile` instantiates the module Rust wrote against
// Rust's own memory and table, and appends its functions to that table. Runs under Bun
// (`bun host.js`) and in a browser (`index.html`).

export async function spike(wasmBytes, log) {
  let x;
  const { instance } = await WebAssembly.instantiate(wasmBytes, {
    env: {
      bk_compile(ptr, len) {
        const bytes = new Uint8Array(x.memory.buffer, ptr, len).slice();
        const m = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
          env: { memory: x.memory, table: x.__indirect_function_table },
        });
        const fs = Object.keys(m.exports).filter((k) => /^f\d+$/.test(k));
        const table = x.__indirect_function_table;
        const base = table.grow(fs.length);
        fs.forEach((k, i) => table.set(base + i, m.exports[`f${i}`]));
        return base;
      },
    },
  });
  x = instance.exports;
  const results = [];
  const check = (name, got, want) => {
    const ok = got === want;
    results.push(ok);
    log(`${ok ? "ok  " : "FAIL"} ${name}: ${got}${ok ? "" : ` (expected ${want})`}`);
  };

  log(`module written by Rust: ${x.spike_compile()} bytes`);
  check("add(40, 2)", x.spike_add(40n, 2n), 42n);
  check("add(i64 max, 1) wraps", x.spike_add(9223372036854775807n, 1n), -9223372036854775808n);
  check("sum of 1..=1000 in Rust's memory", x.spike_sum(1000n), 500500n);
  check("call back into Rust: helper(5)", x.spike_helper(5n), 16n);
  check("struct returned by pointer: 17 / 5", x.spike_div(17n, 5n), 3n);
  check("struct returned by pointer: error status", x.spike_div(1n, 0n), -9223372036854775808n);
  check("fib(25)", x.spike_fib(25n), 75025n);

  const time = (f, n) => {
    f(n); // warm up (the browser may tier up)
    const t = performance.now();
    const r = f(n);
    return [r, performance.now() - t];
  };
  for (const n of [25n, 30n]) {
    const [, gen] = time(x.spike_fib, n);
    const [, rust] = time(x.rust_fib, n);
    log(`fib(${n}): generated ${gen.toFixed(2)} ms, rustc ${rust.toFixed(2)} ms`);
  }
  return results.every(Boolean);
}

if (typeof Bun !== "undefined" && import.meta.main) {
  const bytes = await Bun.file(
    new URL("target/wasm32-unknown-unknown/release/bokoni_jit_spike.wasm", import.meta.url),
  ).arrayBuffer();
  const ok = await spike(bytes, console.log);
  process.exit(ok ? 0 : 1);
}
