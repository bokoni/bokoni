// Messages between the page and the worker that runs Bokoni.

import type { Answer } from "./bokoni";

export interface Request {
  id: number;
  kind: "run" | "check";
  src: string;
}

export type Reply =
  /** Text printed by the script, as it runs. */
  | { id: number; kind: "output"; text: string }
  | { id: number; kind: "done"; answer: Answer; ms: number }
  /** The interpreter stopped abnormally (stack exhausted, internal error). */
  | { id: number; kind: "crash"; message: string };
