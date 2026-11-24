//! Phase 0 of a WebAssembly backend (`docs/execution.md`): can code written at run time
//! with `wasm-encoder` be called like the Cranelift JIT's, through a function pointer?
//!
//! Rust writes a module, the host instantiates it (`env.bk_compile`) against this
//! module's memory and function table, appends its functions to the table and returns
//! the index of the first one. In WebAssembly a Rust function pointer is an index in that
//! table, so Rust then calls them as ordinary `extern "C" fn`s. Checked here:
//!
//! - a call with scalar arguments (`add`), and a recursive one (`fib`, for speed);
//! - reading this module's memory (`sum` over a `Vec<i64>`);
//! - calling back into Rust through the shared table (`call_helper`);
//! - a structure returned by pointer, as the JIT's `(status, result)` would be (`div`).

use std::sync::atomic::{AtomicU32, Ordering};
use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, MemArg, MemoryType, Module, RefType, TableType, TypeSection, ValType,
};

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// Instantiates the module (`len` bytes at `ptr`) and appends its exports `f0`, `f1`…
    /// to the function table; returns the index of `f0`.
    fn bk_compile(ptr: *const u8, len: usize) -> u32;
}

/// Table index of the first generated function (0: not compiled yet).
static BASE: AtomicU32 = AtomicU32::new(0);

/// What `div` returns, written by the generated code at the address it is given.
#[repr(C)]
struct Ret {
    status: u32,
    value: i64,
}

/// Function types of the generated module, by index.
const T_BIN: u32 = 0; // (i64, i64) -> i64
const T_SLICE: u32 = 1; // (i32, i32) -> i64
const T_UN: u32 = 2; // (i64) -> i64
const T_HELPER: u32 = 3; // (i32, i64) -> i64
const T_DIV: u32 = 4; // (i32, i64, i64) -> ()

const FIB: u32 = 2;

fn module() -> Vec<u8> {
    use ValType::{I32, I64};
    let mut types = TypeSection::new();
    types.ty().function([I64, I64], [I64]);
    types.ty().function([I32, I32], [I64]);
    types.ty().function([I64], [I64]);
    types.ty().function([I32, I64], [I64]);
    types.ty().function([I32, I64, I64], []);

    let mut imports = ImportSection::new();
    imports.import(
        "env",
        "memory",
        EntityType::Memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );
    imports.import(
        "env",
        "table",
        EntityType::Table(TableType {
            element_type: RefType::FUNCREF,
            table64: false,
            minimum: 1,
            maximum: None,
            shared: false,
        }),
    );

    let mut funcs = FunctionSection::new();
    let mut exports = ExportSection::new();
    let mut code = CodeSection::new();
    let mut add = |ty: u32, f: Function| {
        let k = funcs.len();
        funcs.function(ty);
        exports.export(&format!("f{k}"), ExportKind::Func, k);
        code.function(&f);
    };

    // f0: add(a, b) = a + b
    let mut f = Function::new([]);
    f.instructions().local_get(0).local_get(1).i64_add().end();
    add(T_BIN, f);

    // f1: sum(ptr, len) = the `len` i64 at `ptr`, added
    let mut f = Function::new([(1, I32), (1, I64)]); // i = 2, acc = 3
    let elem = MemArg { offset: 0, align: 3, memory_index: 0 };
    f.instructions()
        .block(BlockType::Empty)
        .loop_(BlockType::Empty)
        .local_get(2)
        .local_get(1)
        .i32_ge_u()
        .br_if(1)
        .local_get(3)
        .local_get(0)
        .local_get(2)
        .i32_const(3)
        .i32_shl()
        .i32_add()
        .i64_load(elem)
        .i64_add()
        .local_set(3)
        .local_get(2)
        .i32_const(1)
        .i32_add()
        .local_set(2)
        .br(0)
        .end()
        .end()
        .local_get(3)
        .end();
    add(T_SLICE, f);

    // f2: fib(n) = if n < 2 then n else fib(n - 1) + fib(n - 2) end
    let mut f = Function::new([]);
    f.instructions()
        .local_get(0)
        .i64_const(2)
        .i64_lt_s()
        .if_(BlockType::Result(I64))
        .local_get(0)
        .else_()
        .local_get(0)
        .i64_const(1)
        .i64_sub()
        .call(FIB)
        .local_get(0)
        .i64_const(2)
        .i64_sub()
        .call(FIB)
        .i64_add()
        .end()
        .end();
    add(T_UN, f);

    // f3: call_helper(h, x) = h(x), `h` a Rust function's index in the shared table
    let mut f = Function::new([]);
    f.instructions().local_get(1).local_get(0).call_indirect(0, T_UN).end();
    add(T_HELPER, f);

    // f4: div(ret, a, b): `ret` = {status 1} if b == 0, otherwise {0, a / b}
    let mut f = Function::new([]);
    let status = MemArg { offset: 0, align: 2, memory_index: 0 };
    let value = MemArg { offset: 8, align: 3, memory_index: 0 };
    f.instructions()
        .local_get(2)
        .i64_eqz()
        .if_(BlockType::Empty)
        .local_get(0)
        .i32_const(1)
        .i32_store(status)
        .else_()
        .local_get(0)
        .i32_const(0)
        .i32_store(status)
        .local_get(0)
        .local_get(1)
        .local_get(2)
        .i64_div_s()
        .i64_store(value)
        .end()
        .end();
    add(T_DIV, f);

    let mut m = Module::new();
    m.section(&types)
        .section(&imports)
        .section(&funcs)
        .section(&exports)
        .section(&code);
    m.finish()
}

/// The generated function `k` as a function pointer of type `F`.
///
/// # Safety
/// `F` is an `extern "C" fn` type matching the function's WebAssembly type (a mismatch
/// traps at the call: `call_indirect` checks it).
unsafe fn generated<F: Copy>(k: u32) -> F {
    let base = BASE.load(Ordering::Relaxed);
    assert!(base != 0, "spike_compile first");
    assert_eq!(size_of::<F>(), size_of::<usize>());
    // SAFETY: a function pointer is an index in the table (wasm32); see above for `F`.
    unsafe { std::mem::transmute_copy(&((base + k) as usize)) }
}

/// Writes and compiles the module; returns its size in bytes.
#[unsafe(no_mangle)]
pub extern "C" fn spike_compile() -> u32 {
    let bytes = module();
    // SAFETY: the host only reads the bytes during the call.
    let base = unsafe { bk_compile(bytes.as_ptr(), bytes.len()) };
    BASE.store(base, Ordering::Relaxed);
    bytes.len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn spike_add(a: i64, b: i64) -> i64 {
    // SAFETY: f0 is (i64, i64) -> i64.
    let f: extern "C" fn(i64, i64) -> i64 = unsafe { generated(0) };
    f(a, b)
}

#[unsafe(no_mangle)]
pub extern "C" fn spike_sum(n: i64) -> i64 {
    let xs: Vec<i64> = (1..=n).collect();
    // SAFETY: f1 is (i32, i32) -> i64; a pointer and a length are i32 on wasm32.
    let f: extern "C" fn(*const i64, usize) -> i64 = unsafe { generated(1) };
    f(xs.as_ptr(), xs.len())
}

#[unsafe(no_mangle)]
pub extern "C" fn spike_fib(n: i64) -> i64 {
    // SAFETY: f2 is (i64) -> i64.
    let f: extern "C" fn(i64) -> i64 = unsafe { generated(2) };
    f(n)
}

extern "C" fn helper(x: i64) -> i64 {
    x * 3 + 1
}

#[unsafe(no_mangle)]
pub extern "C" fn spike_helper(x: i64) -> i64 {
    // SAFETY: f3 is (i32, i64) -> i64; a function pointer is an i32 index.
    let f: extern "C" fn(extern "C" fn(i64) -> i64, i64) -> i64 = unsafe { generated(3) };
    f(helper, x)
}

/// `a / b`, or `i64::MIN` for the error status.
#[unsafe(no_mangle)]
pub extern "C" fn spike_div(a: i64, b: i64) -> i64 {
    // SAFETY: f4 is (i32, i64, i64) -> (): the C ABI returns `Ret` through a pointer
    // passed first.
    let f: extern "C" fn(i64, i64) -> Ret = unsafe { generated(4) };
    let r = f(a, b);
    if r.status != 0 { i64::MIN } else { r.value }
}

/// The same `fib` compiled by rustc, for comparison.
#[unsafe(no_mangle)]
pub extern "C" fn rust_fib(n: i64) -> i64 {
    if n < 2 { n } else { rust_fib(n - 1) + rust_fib(n - 2) }
}
