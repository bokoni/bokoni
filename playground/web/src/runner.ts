// Page side of the worker: one request at a time; a run that takes too long, or that the
// user stops, terminates the worker (a new one is started for the next request).

import type { Answer } from "./bokoni";
import type { Reply, Request } from "./protocol";

export type Outcome =
  | { kind: "done"; answer: Answer; ms: number }
  | { kind: "crash"; message: string }
  | { kind: "stopped"; reason: "user" | "timeout" };

interface Pending {
  id: number;
  onOutput: (text: string) => void;
  resolve: (o: Outcome) => void;
  timer?: ReturnType<typeof setTimeout>;
}

export class Runner {
  private worker?: Worker;
  private next = 1;
  private pending?: Pending;

  /** `timeout`: ms after which the run is stopped (none if 0). */
  constructor(private readonly timeout = 0) {}

  get busy(): boolean {
    return this.pending !== undefined;
  }

  /** Runs (or checks) `src`; a request still going is stopped first. */
  request(
    kind: Request["kind"],
    src: string,
    onOutput: (text: string) => void = () => {},
  ): Promise<Outcome> {
    if (this.pending) this.stop("user");
    const worker = (this.worker ??= this.spawn());
    const id = this.next++;
    return new Promise((resolve) => {
      const p: Pending = { id, onOutput, resolve };
      if (this.timeout > 0) p.timer = setTimeout(() => this.stop("timeout"), this.timeout);
      this.pending = p;
      worker.postMessage({ id, kind, src } satisfies Request);
    });
  }

  /** Stops the request going, if any. */
  stop(reason: "user" | "timeout" = "user") {
    const p = this.pending;
    if (!p) return;
    this.worker?.terminate();
    this.worker = undefined;
    this.finish(p, { kind: "stopped", reason });
  }

  private spawn(): Worker {
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (e: MessageEvent<Reply>) => {
      const r = e.data;
      const p = this.pending;
      if (!p || r.id !== p.id) return;
      if (r.kind === "output") p.onOutput(r.text);
      else if (r.kind === "done") this.finish(p, { kind: "done", answer: r.answer, ms: r.ms });
      else this.finish(p, { kind: "crash", message: r.message });
    };
    w.onerror = (e) => {
      const p = this.pending;
      this.worker = undefined;
      if (p) this.finish(p, { kind: "crash", message: e.message || "the worker failed" });
    };
    return w;
  }

  private finish(p: Pending, o: Outcome) {
    clearTimeout(p.timer);
    if (this.pending === p) this.pending = undefined;
    p.resolve(o);
  }
}
