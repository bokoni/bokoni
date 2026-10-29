// Builds the Bokoni interpreter for WebAssembly (`playground/wasm`) and copies it into
// `src/generated/`, where the worker imports it.

import { $ } from "bun";

const crate = new URL("../../wasm/", import.meta.url).pathname;
await $`cargo build --release --target wasm32-unknown-unknown`.cwd(crate);

const built = Bun.file(`${crate}target/wasm32-unknown-unknown/release/bokoni_playground.wasm`);
await Bun.write(new URL("../src/generated/bokoni.wasm", import.meta.url), built);
console.log(`bokoni.wasm: ${Math.round(built.size / 1024)} KB`);
