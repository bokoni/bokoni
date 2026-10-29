// The Bokoni interpreter compiled to WebAssembly (see `playground/wasm`): a thin wrapper
// around its exports. Used by the worker, and by the tests under Bun.

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
  alloc(len: number): number;
  dealloc(ptr: number, len: number): void;
  run(ptr: number, len: number): void;
  check(ptr: number, len: number): void;
  result_ptr(): number;
  result_len(): number;
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
      },
    });
    exports = instance.exports as unknown as Exports;
    return new Bokoni(exports);
  }

  /** Runs the script in a new engine. */
  run(src: string): Answer {
    return this.call("run", src);
  }

  /** Static checks only: the script is compiled, not run. */
  check(src: string): Answer {
    return this.call("check", src);
  }

  private call(f: "run" | "check", src: string): Answer {
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
