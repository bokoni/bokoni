// Runs Bokoni off the main thread: a script that loops forever does not freeze the page,
// and the page can stop it (by terminating this worker).

import wasmUrl from "./generated/bokoni.wasm?url";
import { Bokoni } from "./bokoni";
import type { Reply, Request } from "./protocol";

/** Printed text beyond this is dropped (a script printing in a loop). */
const MAX_OUTPUT = 1 << 20;
/** Printed text is sent in pieces, at most this often (ms). */
const FLUSH_EVERY = 40;

const post = (r: Reply) => self.postMessage(r);

const module = WebAssembly.compileStreaming(fetch(wasmUrl));

let current = 0;
let pending = "";
let sent = 0;
let lastFlush = 0;
let truncated = false;

function flush() {
  if (pending) {
    post({ id: current, kind: "output", text: pending });
    sent += pending.length;
    pending = "";
  }
  lastFlush = performance.now();
}

function onOutput(text: string) {
  if (sent + pending.length + text.length > MAX_OUTPUT) {
    if (!truncated) {
      truncated = true;
      pending += text.slice(0, Math.max(0, MAX_OUTPUT - sent - pending.length));
      pending += "\n… (output truncated)\n";
    }
  } else {
    pending += text;
  }
  // (The script runs synchronously: no timer fires before it ends.)
  if (performance.now() - lastFlush > FLUSH_EVERY) {
    flush();
  }
}

/** The instance, recreated after a crash (its memory may be inconsistent). */
let instance: Promise<Bokoni> | undefined;

self.onmessage = async (e: MessageEvent<Request>) => {
  const { id, kind, src } = e.data;
  current = id;
  pending = "";
  sent = 0;
  truncated = false;
  lastFlush = performance.now();
  try {
    instance ??= module.then((m) => Bokoni.instantiate(m, onOutput));
    const bk = await instance;
    const start = performance.now();
    const answer = kind === "run" ? bk.run(src) : bk.check(src);
    const ms = performance.now() - start;
    flush();
    post({ id, kind: "done", answer, ms });
  } catch (err) {
    instance = undefined;
    flush();
    post({ id, kind: "crash", message: describe(err) });
  }
};

function describe(err: unknown): string {
  if (err instanceof RangeError || String(err).includes("call stack")) {
    return "the script recursed too deeply: the browser's stack is exhausted";
  }
  return `internal error of the interpreter: ${err}`;
}
