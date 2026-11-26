//! Bokoni compiled to WebAssembly (the playground's module, `playground/wasm`), run by
//! wasmtime: what a native host such as the platform's kernel would do to run scripts in a
//! sandbox. The script's code and its functions compiled to WebAssembly
//! (`bokoni::wasm_jit`) run in one store, under its fuel and memory limits; the host
//! offers nothing else (no clock, no files, no network).
//!
//! The module imports `env.bk_output` (what the script prints) and `env.bk_compile`: a
//! module written by Bokoni, instantiated here against the module's own memory and
//! function table, its exports appended to the table (`compile`). Components do not
//! share memory, so this links core modules.
//!
//! wasmtime runs WebAssembly on the calling thread's stack: `Config::max_wasm_stack` is
//! set to `Limits::stack`, and the thread needs that much more than its own use (a
//! thread of 16 MiB for the default).

use wasmtime::{
    Caller, Config, Engine, Extern, Instance, Linker, Memory, Module, Ref, Result, Store,
    StoreLimits, StoreLimitsBuilder, Table, TypedFunc, format_err,
};

/// How a script may run.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Units of fuel (about one per WebAssembly instruction).
    pub fuel: u64,
    /// Bytes of memory.
    pub memory: usize,
    /// Bytes of the native stack WebAssembly may use.
    pub stack: usize,
    /// Functions compiled to WebAssembly (1), from their first call (2), or none (0).
    pub jit: u32,
    /// Calls of a function before it is compiled (0: typed code on the first call); a
    /// function with a loop is compiled on its first call. Compiling a function costs
    /// about a millisecond: a function called a few times is better interpreted.
    pub jit_threshold: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            fuel: 2_000_000_000,
            memory: 64 << 20,
            stack: 4 << 20,
            jit: 1,
            jit_threshold: 50,
        }
    }
}

/// The answer of a run.
#[derive(Debug)]
pub struct Run {
    /// The module's answer: `{"value": …, "diagnostics": […]}` (see `playground/wasm`).
    pub json: String,
    /// What the script printed.
    pub printed: String,
    /// Fuel used.
    pub fuel: u64,
    /// Modules compiled from the script's functions.
    pub modules: u32,
}

/// The compiled Bokoni module, ready to run scripts.
pub struct Host {
    engine: Engine,
    module: Module,
}

struct State {
    limits: StoreLimits,
    printed: String,
    modules: u32,
}

impl Host {
    /// `wasm`: the playground's module (`bokoni_playground.wasm`). Its compilation takes
    /// a while (a few tenths of a second), once.
    pub fn new(wasm: &[u8], stack: usize) -> Result<Host> {
        let mut config = Config::new();
        // (Must stay below the stack of asynchronous calls, unused here.)
        config
            .consume_fuel(true)
            .async_stack_size(stack + (1 << 20))
            .max_wasm_stack(stack);
        let engine = Engine::new(&config)?;
        let module = Module::new(&engine, wasm)?;
        Ok(Host { engine, module })
    }

    /// Runs `src` in a new store. A limit reached is an error (`Err`), as a trap is.
    pub fn run(&self, src: &str, limits: Limits) -> Result<Run> {
        let state = State {
            limits: StoreLimitsBuilder::new().memory_size(limits.memory).build(),
            printed: String::new(),
            modules: 0,
        };
        let mut store = Store::new(&self.engine, state);
        store.limiter(|s| &mut s.limits);
        store.set_fuel(limits.fuel)?;
        let mut linker = Linker::new(&self.engine);
        linker.func_wrap(
            "env",
            "bk_output",
            |mut caller: Caller<'_, State>, ptr: u32, len: u32| -> Result<()> {
                let memory = memory(&mut caller)?;
                let bytes = read(&memory, &caller, ptr, len)?;
                caller
                    .data_mut()
                    .printed
                    .push_str(&String::from_utf8_lossy(&bytes));
                Ok(())
            },
        )?;
        linker.func_wrap(
            "env",
            "bk_compile",
            |mut caller: Caller<'_, State>, ptr: u32, len: u32| -> i32 {
                // A module that cannot be compiled: its function stays interpreted.
                compile(&mut caller, ptr, len).map_or(-1, |base| base as i32)
            },
        )?;
        let instance = linker.instantiate(&mut store, &self.module)?;
        let set_jit: TypedFunc<u32, ()> = instance.get_typed_func(&mut store, "set_jit")?;
        set_jit.call(&mut store, limits.jit)?;
        let set_threshold: TypedFunc<u32, ()> =
            instance.get_typed_func(&mut store, "set_jit_threshold")?;
        set_threshold.call(&mut store, limits.jit_threshold)?;
        let alloc = instance.get_typed_func::<u32, u32>(&mut store, "alloc")?;
        let run = instance.get_typed_func::<(u32, u32), ()>(&mut store, "run")?;
        let result_ptr = instance.get_typed_func::<(), u32>(&mut store, "result_ptr")?;
        let result_len = instance.get_typed_func::<(), u32>(&mut store, "result_len")?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| format_err!("no memory"))?;
        let len = src.len() as u32;
        let ptr = alloc.call(&mut store, len.max(1))?;
        memory.write(&mut store, ptr as usize, src.as_bytes())?;
        run.call(&mut store, (ptr, len))?;
        let (p, n) = (
            result_ptr.call(&mut store, ())?,
            result_len.call(&mut store, ())?,
        );
        let json = String::from_utf8_lossy(&read(&memory, &store, p, n)?).into_owned();
        let fuel = limits.fuel - store.get_fuel()?;
        let state = store.into_data();
        Ok(Run {
            json,
            printed: state.printed,
            fuel,
            modules: state.modules,
        })
    }
}

fn memory(caller: &mut Caller<'_, State>) -> Result<Memory> {
    caller
        .get_export("memory")
        .and_then(Extern::into_memory)
        .ok_or_else(|| format_err!("no memory"))
}

fn read(memory: &Memory, store: impl wasmtime::AsContext, ptr: u32, len: u32) -> Result<Vec<u8>> {
    let data = memory.data(&store);
    let (start, end) = (ptr as usize, ptr as usize + len as usize);
    let bytes = data
        .get(start..end)
        .ok_or_else(|| format_err!("out of bounds"))?;
    Ok(bytes.to_vec())
}

/// `env.bk_compile`: the module at `ptr`, instantiated against the caller's memory and
/// table, its exports `f0`, `f1`… appended to the table; the index of `f0`.
fn compile(caller: &mut Caller<'_, State>, ptr: u32, len: u32) -> Result<u64> {
    let memory = memory(caller)?;
    let table: Table = caller
        .get_export("__indirect_function_table")
        .and_then(Extern::into_table)
        .ok_or_else(|| format_err!("no function table"))?;
    let bytes = read(&memory, &*caller, ptr, len)?;
    let module = Module::new(caller.engine(), &bytes)?;
    let instance = Instance::new(&mut *caller, &module, &[memory.into(), table.into()])?;
    let n = module.exports().len() as u64;
    let base = table.grow(&mut *caller, n, Ref::Func(None))?;
    for i in 0..n {
        let f = instance
            .get_func(&mut *caller, &format!("f{i}"))
            .ok_or_else(|| format_err!("missing export"))?;
        table.set(&mut *caller, base + i, Ref::Func(Some(f)))?;
    }
    caller.data_mut().modules += 1;
    Ok(base)
}
