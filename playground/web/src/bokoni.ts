// The Bokoni interpreter compiled to WebAssembly (see `playground/wasm`): a thin wrapper
// around its exports. Used by the worker, and by the tests under Bun.
//
// Its typed functions are compiled to WebAssembly modules as it runs (`bk_compile`): each
// is instantiated against the interpreter's memory and function table, and its functions
// appended to that table, where the interpreter calls them.

export type Severity = "error" | "warning";

export interface Diagnostic {
  severity: Severity;
  kind: "syntax" | "type" | "runtime" | "warning";
  message: string;
  /** The message with a source excerpt, as the command line shows it. */
  rendered: string;
  /**
   * Position in the script (absent for an error in no particular place). Lines and
   * columns from 1, columns in UTF-16 units.
   */
  line?: number;
  col?: number;
  endLine?: number;
  endCol?: number;
}

export interface Answer {
  /** The program's value, as `inspect` shows it (`null` after an error, or for `check`). */
  value: string | null;
  diagnostics: Diagnostic[];
}

interface Exports {
  memory: WebAssembly.Memory;
  __indirect_function_table: WebAssembly.Table;
  alloc(len: number): number;
  dealloc(ptr: number, len: number): void;
  run(ptr: number, len: number): void;
  run_again(ptr: number, len: number): void;
  check(ptr: number, len: number): void;
  result_ptr(): number;
  result_len(): number;
  set_jit(on: number): void;
  jit_compiled(): number;
  jit_baseline(): number;
  jit_dispatched(): number;
  jit_untranslated(): number;
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

export class Bokoni {
  private constructor(private readonly x: Exports) {}

  /**
   * A new instance of the compiled module. `onOutput` receives what the script prints,
   * as it prints it.
   */
  static async instantiate(
    module: WebAssembly.Module,
    onOutput: (text: string) => void,
  ): Promise<Bokoni> {
    let exports: Exports | undefined;
    const instance = await WebAssembly.instantiate(module, {
      env: {
        bk_output(ptr: number, len: number) {
          const bytes = new Uint8Array(exports!.memory.buffer, ptr, len);
          onOutput(decoder.decode(bytes));
        },
        bk_compile(ptr: number, len: number): number {
          const x = exports!;
          const bytes = new Uint8Array(x.memory.buffer, ptr, len).slice();
          try {
            const table = x.__indirect_function_table;
            const m = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
              env: { memory: x.memory, table },
            });
            const fs = Object.keys(m.exports).length;
            const base = table.grow(fs);
            for (let i = 0; i < fs; i++) table.set(base + i, m.exports[`f${i}`] as Function);
            return base;
          } catch (e) {
            // (The function stays interpreted.)
            console.warn("bokoni: compiling a function failed", e);
            return -1;
          }
        },
      },
    });
    exports = instance.exports as unknown as Exports;
    return new Bokoni(exports);
  }

  /** Runs the script in a new engine. */
  run(src: string): Answer {
    return this.call("run", src);
  }

  /**
   * Runs the script in the engine of the previous `runAgain` (its definitions kept), or
   * a new one (tests).
   */
  runAgain(src: string): Answer {
    return this.call("run_again", src);
  }

  /**
   * Functions compiled to WebAssembly (the default), every one from its first call
   * (`"eager"`, for tests; `"dispatch"`, also without structured control flow), or none.
   */
  setJit(on: boolean | "eager" | "dispatch") {
    this.x.set_jit(on === "dispatch" ? 3 : on === "eager" ? 2 : on ? 1 : 0);
  }

  /** Number of typed functions compiled to WebAssembly so far. */
  get compiled(): number {
    return this.x.jit_compiled();
  }

  /** Number of other functions compiled to WebAssembly (baseline code) so far. */
  get baseline(): number {
    return this.x.jit_baseline();
  }

  /** Number of functions that could not be translated to WebAssembly. */
  get untranslated(): number {
    return this.x.jit_untranslated();
  }

  /** Number of compiled functions whose control flow could not be structured. */
  get dispatched(): number {
    return this.x.jit_dispatched();
  }

  /** Static checks only: the script is compiled, not run. */
  check(src: string): Answer {
    return this.call("check", src);
  }

  private call(f: "run" | "run_again" | "check", src: string): Answer {
    const bytes = encoder.encode(src);
    const len = Math.max(bytes.length, 1);
    const ptr = this.x.alloc(len);
    new Uint8Array(this.x.memory.buffer, ptr, bytes.length).set(bytes);
    try {
      this.x[f](ptr, bytes.length);
    } finally {
      this.x.dealloc(ptr, len);
    }
    // (Memory may have grown during the call: views are taken afterwards.)
    const out = new Uint8Array(this.x.memory.buffer, this.x.result_ptr(), this.x.result_len());
    return JSON.parse(decoder.decode(out)) as Answer;
  }
}
