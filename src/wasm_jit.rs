//! The Cranelift JIT (`crate::jit`) for Bokoni itself running in WebAssembly, where native
//! code cannot be generated (`wasm-jit` feature; `docs/execution.md`).
//!
//! The JIT builds its functions in Cranelift's IR (CLIF) as it does natively, both tiers
//! (typed code with its resumptions in the interpreter, baseline code); only the last step
//! differs. Instead of Cranelift's machine code backend, `WasmModule` (in place of
//! `cranelift_jit::JITModule`) translates each function into WebAssembly (`wasm-encoder`)
//! and hands the module to the host, which compiles it to machine code: the embedder
//! gives a compiler (`set_compiler`) that instantiates the module against this program's
//! memory and function table, appends its exports `f0`, `f1`… to that table and returns
//! the index of the first one. A function pointer being an index in that table, Rust then
//! calls the code as it calls the native JIT's.
//!
//! - CLIF values are WebAssembly locals (an `i8` in an `i32`: its low byte, the others
//!   zero); a block's parameters are locals assigned on the edges to it.
//! - The CLIF blocks are written as structured control flow (`control_flow`).
//! - Stack slots live in an area of this program's memory used as a stack (`ARENA_SP`),
//!   a function's frame taken there when first used.
//! - Values nothing uses are not computed, stores into stack slots nothing reads not
//!   made (`Translator::liveness`).
//! - The JIT handles addresses as `i64` (as on a 64-bit machine): they are 32 bits in
//!   WebAssembly's memory, converted at each access and at each call of a Rust helper,
//!   whose parameters are known from its Rust type (`sig_of`).
//! - Each function has an adapter for Rust (its `i64` addresses as `i32`): the function
//!   pointer `get_finalized_function` gives.

// (Built with the native JIT too, as `cargo test --features wasm-jit` does: unused then.)
#![cfg_attr(feature = "jit", allow(dead_code))]

use crate::intern::{FxHashMap, FxHashSet};
use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    self, Block, ExternalName, InstructionData, Opcode, Signature, Type, Value, types,
};
use cranelift_codegen::isa::{CallConv, TargetFrontendConfig};
use cranelift_module::{FuncId, Linkage};
use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};
use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, InstructionSink, MemArg, MemoryType, Module, RefType, TableType, TypeSection,
    ValType,
};

/// Instantiates a module (its bytes) against this program's memory (`env.memory`) and
/// function table (`env.table`), appends its exports `f0`, `f1`… to that table, in order,
/// and returns the index of `f0`; `None` if it cannot.
pub type Compiler = fn(&[u8]) -> Option<u32>;

thread_local! {
    static COMPILER: Cell<Option<Compiler>> = const { Cell::new(None) };
    static COMPILED: Cell<u32> = const { Cell::new(0) };
    static BASELINE: Cell<u32> = const { Cell::new(0) };
    static DISPATCHED: Cell<u32> = const { Cell::new(0) };
    static UNTRANSLATED: Cell<u32> = const { Cell::new(0) };
    static STRUCTURED: Cell<bool> = const { Cell::new(true) };
}

/// Sets the compiler of the generated modules (WebAssembly only: elsewhere, nothing is
/// compiled). Without one, everything is interpreted.
pub fn set_compiler(compiler: Option<Compiler>) {
    COMPILER.with(|c| c.set(compiler));
}

/// The compiler of the generated modules, inside WebAssembly.
pub(crate) fn compiler() -> Option<Compiler> {
    if !cfg!(target_arch = "wasm32") {
        return None;
    }
    COMPILER.with(Cell::get)
}

/// Number of functions compiled so far to typed code (tests).
pub fn compiled_functions() -> u32 {
    COMPILED.with(Cell::get)
}

/// Number of functions compiled so far to baseline code (tests).
pub fn baseline_functions() -> u32 {
    BASELINE.with(Cell::get)
}

/// Number of functions whose control flow was not structured (its graph not reducible),
/// so far (tests).
pub fn dispatched_functions() -> u32 {
    DISPATCHED.with(Cell::get)
}

/// Number of functions that could not be translated (an instruction the translation does
/// not know), so far (tests).
pub fn untranslated_functions() -> u32 {
    UNTRANSLATED.with(Cell::get)
}

pub(crate) fn count_typed(n: usize) {
    COMPILED.with(|c| c.set(c.get() + n as u32));
}

pub(crate) fn count_baseline() {
    BASELINE.with(|c| c.set(c.get() + 1));
}

/// Structured control flow when possible (the default), or always the dispatch loop
/// (tests of that loop).
#[doc(hidden)]
pub fn set_structured(on: bool) {
    STRUCTURED.with(|c| c.set(on));
}

// ─── Signatures of the Rust helpers ────────────────────────────────────────

/// A function's WebAssembly parameters and results.
pub(crate) type Sig = (Vec<ValType>, Vec<ValType>);

/// WebAssembly type of an address in memory.
const PTR: ValType = if size_of::<usize>() == 4 {
    ValType::I32
} else {
    ValType::I64
};

/// WebAssembly type of a parameter of a Rust `extern "C"` function.
pub(crate) trait WasmArg {
    const TY: ValType;
}

impl<T> WasmArg for *const T {
    const TY: ValType = PTR;
}
impl<T> WasmArg for *mut T {
    const TY: ValType = PTR;
}
impl WasmArg for usize {
    const TY: ValType = PTR;
}

macro_rules! wasm_arg {
    ($ty:expr; $($t:ty),*) => {
        $(impl WasmArg for $t {
            const TY: ValType = $ty;
        })*
    };
}
wasm_arg!(ValType::I64; u64, i64);
wasm_arg!(ValType::I32; u32, i32, u16, i16, u8, i8, bool);
wasm_arg!(ValType::F64; f64);

/// WebAssembly results of a Rust `extern "C"` function.
pub(crate) trait WasmRet {
    fn tys() -> Vec<ValType>;
}

impl WasmRet for () {
    fn tys() -> Vec<ValType> {
        Vec::new()
    }
}
impl<T: WasmArg> WasmRet for T {
    fn tys() -> Vec<ValType> {
        vec![T::TY]
    }
}

/// A Rust function the generated code calls: an `extern "C" fn` pointer.
pub(crate) trait HelperFn {
    fn sig() -> Sig;
}

macro_rules! helper_fn {
    ($($a:ident),*) => {
        impl<R: WasmRet, $($a: WasmArg),*> HelperFn for extern "C" fn($($a),*) -> R {
            fn sig() -> Sig {
                (vec![$($a::TY),*], R::tys())
            }
        }
    };
}
helper_fn!();
helper_fn!(A);
helper_fn!(A, B);
helper_fn!(A, B, C);
helper_fn!(A, B, C, D);
helper_fn!(A, B, C, D, E);
helper_fn!(A, B, C, D, E, G);
helper_fn!(A, B, C, D, E, G, H);
helper_fn!(A, B, C, D, E, G, H, I);

/// The WebAssembly signature of Rust function `f`, from its type.
pub(crate) fn sig_of<F: HelperFn>(_: F) -> Sig {
    F::sig()
}

// ─── Stack slots ───────────────────────────────────────────────────────────

/// Bytes of the area the generated functions' stack slots live in (a stack: each call
/// takes its frame on entry and gives it back before returning). The call depth bounds
/// it; beyond, the code traps.
const ARENA_BYTES: u32 = 8 << 20;

/// Top of the area, and its end.
static ARENA_SP: AtomicU32 = AtomicU32::new(0);
static ARENA_END: AtomicU32 = AtomicU32::new(0);

/// The address of the area's top, and its end.
fn arena() -> (u32, u32) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // Never freed: the generated code uses it for as long as the program runs.
        let area: &'static mut [u64] =
            Box::leak(vec![0u64; ARENA_BYTES as usize / 8].into_boxed_slice());
        let base = area.as_mut_ptr() as usize as u32;
        ARENA_SP.store(base, Ordering::Relaxed);
        ARENA_END.store(base + ARENA_BYTES, Ordering::Relaxed);
    });
    (
        &ARENA_SP as *const AtomicU32 as usize as u32,
        ARENA_END.load(Ordering::Relaxed),
    )
}

// ─── The module ────────────────────────────────────────────────────────────

struct Decl {
    linkage: Linkage,
    sig: Signature,
}

/// A function translated and compiled: the table indices of its function and of its
/// adapter for Rust, and its WebAssembly type.
struct Done {
    func: u32,
    adapter: u32,
    ty: Sig,
}

/// In place of `cranelift_jit::JITModule`, with the methods the JIT uses: functions are
/// declared, defined (their CLIF kept), then translated together into one WebAssembly
/// module by `finalize_definitions`.
pub(crate) struct WasmModule {
    decls: Vec<Decl>,
    /// Rust functions the code calls (imported functions), by name: table index and
    /// signature.
    helpers: FxHashMap<&'static str, (u32, Sig)>,
    /// Imported functions by id.
    imports: FxHashMap<FuncId, &'static str>,
    defined: Vec<(FuncId, ir::Function)>,
    done: FxHashMap<FuncId, Done>,
}

impl WasmModule {
    /// `helpers`: the Rust functions the code may call (name, address, signature).
    pub(crate) fn new(helpers: impl IntoIterator<Item = (&'static str, usize, Sig)>) -> Self {
        WasmModule {
            decls: Vec::new(),
            helpers: helpers
                .into_iter()
                .map(|(name, addr, sig)| (name, (addr as u32, sig)))
                .collect(),
            imports: FxHashMap::default(),
            defined: Vec::new(),
            done: FxHashMap::default(),
        }
    }

    /// As on a 64-bit machine: addresses are `i64` in the IR.
    pub(crate) fn target_config(&self) -> TargetFrontendConfig {
        TargetFrontendConfig {
            default_call_conv: CallConv::SystemV,
            pointer_width: target_lexicon::PointerWidth::U64,
            page_size_align_log2: 12,
        }
    }

    pub(crate) fn make_signature(&self) -> Signature {
        Signature::new(CallConv::SystemV)
    }

    pub(crate) fn declare_function(
        &mut self,
        name: &str,
        linkage: Linkage,
        sig: &Signature,
    ) -> Result<FuncId, ()> {
        let id = FuncId::from_u32(self.decls.len() as u32);
        if linkage == Linkage::Import {
            let (name, _) = self.helpers.get_key_value(name).ok_or(())?;
            self.imports.insert(id, name);
        }
        self.decls.push(Decl {
            linkage,
            sig: sig.clone(),
        });
        Ok(id)
    }

    /// As `cranelift_module::Module::declare_func_in_func`.
    pub(crate) fn declare_func_in_func(
        &mut self,
        id: FuncId,
        func: &mut ir::Function,
    ) -> ir::FuncRef {
        let decl = &self.decls[id.as_u32() as usize];
        let signature = func.import_signature(decl.sig.clone());
        let name = func.declare_imported_user_function(ir::UserExternalName {
            namespace: 0,
            index: id.as_u32(),
        });
        func.import_function(ir::ExtFuncData {
            name: ExternalName::user(name),
            signature,
            colocated: decl.linkage.is_final(),
            patchable: false,
        })
    }

    pub(crate) fn make_context(&self) -> Context {
        Context::new()
    }

    pub(crate) fn clear_context(&self, ctx: &mut Context) {
        ctx.clear();
    }

    pub(crate) fn define_function(&mut self, id: FuncId, ctx: &mut Context) -> Result<(), ()> {
        self.defined.push((id, ctx.func.clone()));
        Ok(())
    }

    /// Translates the functions defined since the last call into one module and has the
    /// host compile it. An error: none of them can run (an instruction the translation
    /// does not know, no compiler).
    pub(crate) fn finalize_definitions(&mut self) -> Result<(), ()> {
        let defined = std::mem::take(&mut self.defined);
        if defined.is_empty() {
            return Ok(());
        }
        let compiler = compiler().ok_or(())?;
        let mut types = Types::default();
        let mut funcs = FunctionSection::new();
        let mut code = CodeSection::new();
        let mut exports = ExportSection::new();
        // This module's functions: function `2i`, its adapter `2i + 1`.
        let local: FxHashMap<FuncId, u32> = defined
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, 2 * i as u32))
            .collect();
        let mut sigs = Vec::new();
        // The cold regions' functions, after those (`Region`).
        let mut next = 2 * defined.len() as u32;
        let mut cold: Vec<(u32, Function)> = Vec::new();
        for (i, (_, func)) in defined.iter().enumerate() {
            let sig = wasm_sig(&func.signature)?;
            let ty = types.index(sig.0.clone(), sig.1.clone());
            let body = self
                .translate(func, &local, &mut types, &sig, &mut next, &mut cold)
                .inspect_err(|()| UNTRANSLATED.with(|c| c.set(c.get() + 1)))?;
            funcs.function(ty);
            code.function(&body);
            let (adapter, (aps, ars)) = adapter(func, 2 * i as u32)?;
            funcs.function(types.index(aps, ars));
            code.function(&adapter);
            for k in [2 * i, 2 * i + 1] {
                exports.export(&format!("f{k}"), ExportKind::Func, k as u32);
            }
            sigs.push(sig);
        }
        for (ty, body) in &cold {
            funcs.function(*ty);
            code.function(body);
        }
        let base = compiler(&assemble(&types, &funcs, &exports, &code)).ok_or(())?;
        for (i, ((id, _), ty)) in defined.into_iter().zip(sigs).enumerate() {
            self.done.insert(
                id,
                Done {
                    func: base + 2 * i as u32,
                    adapter: base + 2 * i as u32 + 1,
                    ty,
                },
            );
        }
        Ok(())
    }

    /// The code of `func` (signature `sig`), and of its cold regions, appended to `cold`
    /// (their type, their code), numbered from `next`.
    fn translate(
        &self,
        func: &ir::Function,
        local: &FxHashMap<FuncId, u32>,
        types: &mut Types,
        sig: &Sig,
        next: &mut u32,
        cold: &mut Vec<(u32, Function)>,
    ) -> Result<Function, ()> {
        let mut t = Translator::new(self, func, local, types, None)?;
        let (regions, kept) = t.cold_regions();
        for r in &regions {
            t.stubs.insert(r.blocks[0], (*next, r.params.clone()));
            *next += 1;
        }
        // (A block of a region also written here, from a region too small to be apart,
        // stays.)
        for r in &regions {
            let hidden = r
                .blocks
                .iter()
                .filter(|b| !t.stubs.contains_key(b) && !kept.contains(b));
            t.hidden.extend(hidden.copied().collect::<Vec<_>>());
        }
        let body = t.run()?;
        for r in &regions {
            let mut ps = Vec::with_capacity(r.params.len() + 1);
            for v in &r.params {
                ps.push(val_type(func.dfg.value_type(*v))?);
            }
            ps.push(ValType::I32);
            let ty = types.index(ps, sig.1.clone());
            let code = Translator::new(self, func, local, types, Some(r))?.run()?;
            cold.push((ty, code));
        }
        Ok(body)
    }

    /// The function pointer of the adapter of function `id` (its table index).
    pub(crate) fn get_finalized_function(&self, id: FuncId) -> *const u8 {
        self.done[&id].adapter as usize as *const u8
    }
}

fn val_type(t: Type) -> Result<ValType, ()> {
    Ok(match t {
        types::I8 | types::I32 => ValType::I32,
        types::I64 => ValType::I64,
        types::F64 => ValType::F64,
        _ => return Err(()),
    })
}

/// The WebAssembly signature of a generated function (an `i8` as an `i32`).
fn wasm_sig(sig: &Signature) -> Result<Sig, ()> {
    let ps = sig
        .params
        .iter()
        .map(|p| val_type(p.value_type))
        .collect::<Result<_, _>>()?;
    let rs = sig
        .returns
        .iter()
        .map(|p| val_type(p.value_type))
        .collect::<Result<_, _>>()?;
    Ok((ps, rs))
}

/// The adapter of function `index` for Rust: its `i64` parameters are addresses, as Rust
/// passes them (the functions Rust calls take addresses only: the entry points of typed
/// code, baseline code).
fn adapter(func: &ir::Function, index: u32) -> Result<(Function, Sig), ()> {
    let (ps, rs) = wasm_sig(&func.signature)?;
    let aps: Vec<ValType> = ps
        .iter()
        .map(|t| if *t == ValType::I64 { PTR } else { *t })
        .collect();
    let mut f = Function::new([]);
    let mut x = f.instructions();
    for (k, (p, a)) in ps.iter().zip(&aps).enumerate() {
        x.local_get(k as u32);
        if p != a {
            x.i64_extend_i32_u();
        }
    }
    x.call(index);
    x.end();
    Ok((f, (aps, rs)))
}

/// The function types of a module, without duplicates.
#[derive(Default)]
struct Types {
    list: Vec<Sig>,
}

impl Types {
    fn index(&mut self, params: Vec<ValType>, results: Vec<ValType>) -> u32 {
        let t = (params, results);
        match self.list.iter().position(|x| *x == t) {
            Some(i) => i as u32,
            None => {
                self.list.push(t);
                (self.list.len() - 1) as u32
            }
        }
    }
}

/// The bytes of a module of these functions, importing the program's memory and table.
fn assemble(
    types: &Types,
    funcs: &FunctionSection,
    exports: &ExportSection,
    code: &CodeSection,
) -> Vec<u8> {
    let mut ts = TypeSection::new();
    for (ps, rs) in &types.list {
        ts.ty().function(ps.iter().copied(), rs.iter().copied());
    }
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
    let mut m = Module::new();
    m.section(&ts)
        .section(&imports)
        .section(funcs)
        .section(exports)
        .section(code);
    m.finish()
}

// ─── Translation ───────────────────────────────────────────────────────────

/// Translation of one CLIF function.
struct Translator<'a> {
    module: &'a WasmModule,
    func: &'a ir::Function,
    /// Functions of the module being written (their function index).
    local: &'a FxHashMap<FuncId, u32>,
    types: &'a mut Types,
    /// Blocks in layout order (the entry first), and their number.
    blocks: Vec<Block>,
    number: FxHashMap<Block, u32>,
    /// Local of each value.
    locals: FxHashMap<Value, u32>,
    /// Types of the locals after the parameters.
    decl: Vec<ValType>,
    nparams: u32,
    /// Offset of each stack slot in the frame, and the frame's size.
    slots: Vec<u32>,
    frame: u32,
    /// Locals: the frame's address, scratch ones, the block of the dispatch loop.
    fp: u32,
    s64: [u32; 3],
    s32: u32,
    pc: u32,
    /// Something could not be translated.
    failed: bool,
    /// Values used (the others are not computed: `liveness`).
    live: FxHashSet<Value>,
    /// Stores into stack slots nothing reads.
    dead_stores: FxHashSet<ir::Inst>,
    /// Values computed where they are used rather than kept in a local (`trees`).
    inline: FxHashSet<Value>,
    /// Entries of the cold regions written as functions of their own: the function's
    /// index, and its arguments (`cold_regions`).
    stubs: FxHashMap<Block, (u32, Vec<Value>)>,
    /// The other blocks of those regions, not written in this function.
    hidden: FxHashSet<Block>,
}

/// Cold code that only returns (resumptions in the interpreter, errors), written as a
/// function of its own: called by the function it comes from, which returns its result.
/// The function stays small, and the host inlines it into its callers more readily
/// (V8: `fib(30)` 3.6 → 2.8 ms).
struct Region {
    /// Its first block, and the blocks reached from it.
    blocks: Vec<Block>,
    /// The values it reads that are computed before it; the frame's address follows.
    params: Vec<Value>,
}

/// Instructions a cold region has at least, to be written apart.
const MIN_REGION: usize = 8;

/// Instructions whose effect is more than their results.
fn effect(op: Opcode) -> bool {
    matches!(op, Opcode::Store | Opcode::Call) || op.is_terminator()
}

impl<'a> Translator<'a> {
    /// The translation of function `func`, or of its cold `region`.
    fn new(
        module: &'a WasmModule,
        func: &'a ir::Function,
        local: &'a FxHashMap<FuncId, u32>,
        types: &'a mut Types,
        region: Option<&Region>,
    ) -> Result<Self, ()> {
        let blocks: Vec<Block> = match region {
            Some(r) => r.blocks.clone(),
            None => func.layout.blocks().collect(),
        };
        let number = blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (*b, i as u32))
            .collect();
        let mut t = Translator {
            module,
            func,
            local,
            types,
            blocks,
            number,
            locals: FxHashMap::default(),
            decl: Vec::new(),
            nparams: match region {
                Some(r) => r.params.len() as u32 + 1,
                None => func.signature.params.len() as u32,
            },
            slots: Vec::new(),
            frame: 0,
            fp: 0,
            s64: [0; 3],
            s32: 0,
            pc: 0,
            failed: false,
            live: FxHashSet::default(),
            dead_stores: FxHashSet::default(),
            inline: FxHashSet::default(),
            stubs: FxHashMap::default(),
            hidden: FxHashSet::default(),
        };
        t.liveness();
        t.trees();
        // The parameters: those of the entry block, or the values the region reads.
        let entry = *t.blocks.first().ok_or(())?;
        let params = match region {
            Some(r) => r.params.clone(),
            None => func.dfg.block_params(entry).to_vec(),
        };
        for (k, v) in params.iter().enumerate() {
            t.locals.insert(*v, k as u32);
        }
        for (slot, data) in func.sized_stack_slots.iter() {
            let align = (1u32 << data.align_shift).max(8);
            t.frame = t.frame.next_multiple_of(align);
            debug_assert_eq!(slot.as_u32() as usize, t.slots.len());
            t.slots.push(t.frame);
            t.frame += data.size;
        }
        t.frame = t.frame.next_multiple_of(16);
        // No slot used: no frame.
        let used = func.layout.blocks().any(|b| {
            func.layout.block_insts(b).any(|i| {
                matches!(func.dfg.insts[i], InstructionData::StackAddr { .. })
                    && t.live.contains(&func.dfg.first_result(i))
            })
        });
        if !used {
            t.frame = 0;
        }
        // (A region is given the frame of its function.)
        t.fp = match region {
            Some(_) => t.nparams - 1,
            None => t.scratch(ValType::I32),
        };
        t.s64 = std::array::from_fn(|_| 0);
        for k in 0..3 {
            t.s64[k] = t.scratch(ValType::I64);
        }
        t.s32 = t.scratch(ValType::I32);
        t.pc = t.scratch(ValType::I32);
        Ok(t)
    }

    /// The values the function uses: those an instruction with an effect (a store, a
    /// call, a branch, a return) reads, and those they are computed from; a block's
    /// parameter is used if it is read, and then the arguments given to it. Before that,
    /// the stores into a stack slot whose address only serves to write to it.
    fn liveness(&mut self) {
        let f = self.func;
        let dfg = &f.dfg;
        let pool = &dfg.value_lists;
        let insts: Vec<ir::Inst> = f
            .layout
            .blocks()
            .flat_map(|b| f.layout.block_insts(b))
            .collect();
        // Stack slots read (an address used other than to write to it).
        let mut slot_of: FxHashMap<Value, ir::StackSlot> = FxHashMap::default();
        for &i in &insts {
            if let InstructionData::StackAddr { stack_slot, .. } = dfg.insts[i] {
                slot_of.insert(dfg.first_result(i), stack_slot);
            }
        }
        let mut read: FxHashSet<ir::StackSlot> = FxHashSet::default();
        for &i in &insts {
            let args: Vec<Value> = dfg.inst_values(i).map(|v| dfg.resolve_aliases(v)).collect();
            let store_addr = match dfg.insts[i] {
                InstructionData::Store { args: [v, a], .. } => {
                    Some((dfg.resolve_aliases(v), dfg.resolve_aliases(a)))
                }
                _ => None,
            };
            for (k, v) in args.iter().enumerate() {
                let Some(slot) = slot_of.get(v) else {
                    continue;
                };
                // (The address operand of a store, not the value stored.)
                let written = store_addr.is_some_and(|(sv, a)| k == 1 && a == *v && sv != *v);
                if !written {
                    read.insert(*slot);
                }
            }
        }
        for &i in &insts {
            if let InstructionData::Store { args: [_, a], .. } = dfg.insts[i]
                && let Some(slot) = slot_of.get(&dfg.resolve_aliases(a))
                && !read.contains(slot)
            {
                self.dead_stores.insert(i);
            }
        }
        // Edges into each block (terminator, successor number).
        let mut incoming: FxHashMap<Block, Vec<ir::BlockCall>> = FxHashMap::default();
        for &i in &insts {
            match dfg.insts[i] {
                InstructionData::Jump { destination, .. } => {
                    incoming
                        .entry(destination.block(pool))
                        .or_default()
                        .push(destination);
                }
                InstructionData::Brif { blocks, .. } => {
                    for c in blocks {
                        incoming.entry(c.block(pool)).or_default().push(c);
                    }
                }
                _ => {}
            }
        }
        let mut todo: Vec<Value> = Vec::new();
        for &i in &insts {
            let data = dfg.insts[i];
            if !effect(data.opcode()) || self.dead_stores.contains(&i) {
                continue;
            }
            // (A branch's arguments to its targets: used if their parameter is.)
            todo.extend(dfg.inst_args(i).iter().map(|v| dfg.resolve_aliases(*v)));
        }
        while let Some(v) = todo.pop() {
            if !self.live.insert(v) {
                continue;
            }
            match dfg.value_def(v) {
                ir::ValueDef::Result(i, _) => {
                    todo.extend(dfg.inst_args(i).iter().map(|a| dfg.resolve_aliases(*a)));
                }
                ir::ValueDef::Param(b, k) => {
                    for c in incoming.get(&b).into_iter().flatten() {
                        if let Some(ir::BlockArg::Value(a)) = c.args(pool).nth(k) {
                            todo.push(dfg.resolve_aliases(a));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// The values computed where they are used, as a compiler from expressions would
    /// write them: constants, at each use; a value used once, by an instruction of the
    /// same block, when nothing with an effect (a store, a call) comes in between. Their
    /// operands keep their value there: each value has its own local, and a block's
    /// parameters are assigned only on the edges into it.
    fn trees(&mut self) {
        let f = self.func;
        let dfg = &f.dfg;
        // Uses of each value (a block's arguments included), and the last user.
        let mut uses: FxHashMap<Value, (u32, ir::Inst)> = FxHashMap::default();
        for b in f.layout.blocks() {
            for i in f.layout.block_insts(b) {
                for v in dfg.inst_values(i) {
                    let e = uses.entry(dfg.resolve_aliases(v)).or_insert((0, i));
                    e.0 += 1;
                    e.1 = i;
                }
            }
        }
        for b in f.layout.blocks() {
            let insts: Vec<ir::Inst> = f.layout.block_insts(b).collect();
            let at: FxHashMap<ir::Inst, usize> =
                insts.iter().enumerate().map(|(k, i)| (*i, k)).collect();
            // Where each instruction's code is written: its place, or its user's.
            let mut pos: Vec<usize> = (0..insts.len()).collect();
            for k in (0..insts.len()).rev() {
                let i = insts[k];
                let data = dfg.insts[i];
                let [r] = dfg.inst_results(i) else {
                    continue;
                };
                let r = *r;
                if !self.live.contains(&r) {
                    continue;
                }
                if matches!(
                    data,
                    InstructionData::UnaryImm { .. } | InstructionData::UnaryIeee64 { .. }
                ) {
                    self.inline.insert(r);
                    continue;
                }
                let pure = match data {
                    InstructionData::Binary { opcode, .. } => !matches!(
                        opcode,
                        Opcode::SaddOverflow
                            | Opcode::SsubOverflow
                            | Opcode::SmulOverflow
                            | Opcode::Sdiv
                            | Opcode::Srem
                    ),
                    // (`iabs` uses a scratch local.)
                    InstructionData::Unary { opcode, .. } => opcode != Opcode::Iabs,
                    InstructionData::LoadNoOffset { .. }
                    | InstructionData::IntCompare { .. }
                    | InstructionData::FloatCompare { .. }
                    | InstructionData::Ternary { .. }
                    | InstructionData::Load { .. } => true,
                    _ => false,
                };
                let Some(&(1, user)) = uses.get(&r) else {
                    continue;
                };
                let Some(&u) = at.get(&user) else {
                    continue;
                };
                if !pure || u <= k {
                    continue;
                }
                let end = pos[u];
                let clear = insts[k + 1..end]
                    .iter()
                    .all(|j| !effect(dfg.insts[*j].opcode()));
                if clear {
                    self.inline.insert(r);
                    pos[k] = end;
                }
            }
        }
    }

    /// The cold regions of the function: blocks marked cold (`set_cold_block`) and those
    /// only they lead to, from which every path returns. One region per block of them
    /// entered from the rest of the function, with the blocks reached from it (a block
    /// reached from several is written in each). Also the blocks of the regions too small
    /// to be apart, written in the function.
    fn cold_regions(&self) -> (Vec<Region>, FxHashSet<Block>) {
        let f = self.func;
        let n = self.blocks.len();
        let succ: Vec<Vec<u32>> = self.blocks.iter().map(|b| self.successors(*b)).collect();
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (b, ss) in succ.iter().enumerate() {
            for &s in ss {
                preds[s as usize].push(b as u32);
            }
        }
        let mut cold: Vec<bool> = self.blocks.iter().map(|b| f.layout.is_cold(*b)).collect();
        cold[0] = false;
        // Blocks only cold ones lead to.
        loop {
            let mut changed = false;
            for b in 1..n {
                if !cold[b] && !preds[b].is_empty() && preds[b].iter().all(|p| cold[*p as usize]) {
                    cold[b] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // Those from which every path stays cold (and so returns).
        loop {
            let mut changed = false;
            for b in 0..n {
                if cold[b] && succ[b].iter().any(|s| !cold[*s as usize]) {
                    cold[b] = false;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let dfg = &f.dfg;
        let mut regions = Vec::new();
        let mut kept = FxHashSet::default();
        for e in 0..n {
            if !cold[e] || preds[e].iter().all(|p| cold[*p as usize]) {
                continue;
            }
            let mut seen = vec![false; n];
            let mut order = vec![e as u32];
            seen[e] = true;
            let mut k = 0;
            while k < order.len() {
                for &s in &succ[order[k] as usize] {
                    if !seen[s as usize] {
                        seen[s as usize] = true;
                        order.push(s);
                    }
                }
                k += 1;
            }
            let blocks: Vec<Block> = order.iter().map(|b| self.blocks[*b as usize]).collect();
            let size: usize = blocks
                .iter()
                .map(|b| f.layout.block_insts(*b).count())
                .sum();
            if size < MIN_REGION {
                kept.extend(blocks);
                continue;
            }
            // The values read there and computed before: those of the instructions
            // written there (a value computed at its use, by its own instruction, is
            // computed in the same block; a constant needs nothing).
            let mut defined: FxHashSet<Value> = FxHashSet::default();
            for (k, b) in blocks.iter().enumerate() {
                if k > 0 {
                    defined.extend(dfg.block_params(*b).iter().copied());
                }
                for i in f.layout.block_insts(*b) {
                    defined.extend(dfg.inst_results(i).iter().copied());
                }
            }
            let mut params: Vec<Value> = Vec::new();
            let mut taken: FxHashSet<Value> = FxHashSet::default();
            for b in &blocks {
                for i in f.layout.block_insts(*b) {
                    for v in dfg.inst_values(i) {
                        let v = dfg.resolve_aliases(v);
                        if defined.contains(&v) || !self.live.contains(&v) {
                            continue;
                        }
                        if self.inline.contains(&v) {
                            // (Only a constant: other values computed at their use are
                            // used in their own block.)
                            continue;
                        }
                        if taken.insert(v) {
                            params.push(v);
                        }
                    }
                }
            }
            regions.push(Region { blocks, params });
        }
        (regions, kept)
    }

    fn scratch(&mut self, t: ValType) -> u32 {
        self.decl.push(t);
        self.nparams + self.decl.len() as u32 - 1
    }

    fn value(&self, v: Value) -> Value {
        self.func.dfg.resolve_aliases(v)
    }

    fn ty(&self, v: Value) -> Type {
        self.func.dfg.value_type(self.value(v))
    }

    /// The local of value `v`.
    fn local(&mut self, v: Value) -> u32 {
        let v = self.value(v);
        if let Some(l) = self.locals.get(&v) {
            return *l;
        }
        let t = val_type(self.func.dfg.value_type(v)).unwrap_or_else(|()| {
            self.failed = true;
            ValType::I32
        });
        let l = self.scratch(t);
        self.locals.insert(v, l);
        l
    }

    fn get(&mut self, x: &mut InstructionSink, v: Value) {
        let v = self.value(v);
        if self.inline.contains(&v) {
            let ir::ValueDef::Result(i, _) = self.func.dfg.value_def(v) else {
                unreachable!("an instruction's result");
            };
            if self.emit(x, i, false).is_err() {
                self.failed = true;
            }
            return;
        }
        let l = self.local(v);
        x.local_get(l);
    }

    fn set(&mut self, x: &mut InstructionSink, v: Value) {
        let l = self.local(v);
        x.local_set(l);
    }

    /// The function's code; its locals are known once it is written.
    fn run(mut self) -> Result<Function, ()> {
        let succ: Vec<Vec<u32>> = self
            .blocks
            .iter()
            .map(|b| match self.stubs.contains_key(b) {
                true => Vec::new(),
                false => self.successors(*b),
            })
            .collect();
        let reached: Vec<bool> = self
            .blocks
            .iter()
            .map(|b| !self.hidden.contains(b))
            .collect();
        let mut bytes = Vec::new();
        {
            let mut x = InstructionSink::new(&mut bytes);
            let pc = self.pc;
            control_flow(&mut x, &succ, &reached, pc, &mut self);
            x.unreachable();
            x.end();
        }
        if self.failed {
            return Err(());
        }
        let mut f = Function::new_with_locals_types(self.decl.iter().copied());
        f.raw(bytes);
        Ok(f)
    }

    /// The frame: taken from the area on its first use (a trap if the area is full);
    /// `fp` is 0 until then.
    fn frame(&mut self, x: &mut InstructionSink) {
        let (sp, end) = arena();
        x.local_get(self.fp).i32_eqz();
        x.if_(BlockType::Empty);
        x.i32_const(sp as i32)
            .i32_load(mem(0, 2))
            .local_tee(self.fp);
        x.i32_const(self.frame as i32).i32_add().local_tee(self.s32);
        x.i32_const(end as i32).i32_gt_u();
        x.if_(BlockType::Empty).unreachable().end();
        x.i32_const(sp as i32)
            .local_get(self.s32)
            .i32_store(mem(0, 2));
        x.end();
    }

    /// The frame given back, if taken (before a return).
    fn epilogue(&mut self, x: &mut InstructionSink) {
        if self.frame == 0 {
            return;
        }
        let (sp, _) = arena();
        x.local_get(self.fp).if_(BlockType::Empty);
        x.i32_const(sp as i32)
            .local_get(self.fp)
            .i32_store(mem(0, 2));
        x.end();
    }

    fn successors(&self, b: Block) -> Vec<u32> {
        let Some(term) = self.func.layout.last_inst(b) else {
            return Vec::new();
        };
        let pool = &self.func.dfg.value_lists;
        match &self.func.dfg.insts[term] {
            InstructionData::Jump { destination, .. } => {
                vec![self.number[&destination.block(pool)]]
            }
            InstructionData::Brif { blocks, .. } => {
                blocks.iter().map(|c| self.number[&c.block(pool)]).collect()
            }
            _ => Vec::new(),
        }
    }

    /// The block call of the terminator of `b` to its successor `which`.
    fn block_call(&self, b: Block, which: usize) -> ir::BlockCall {
        let term = self.func.layout.last_inst(b).unwrap();
        match &self.func.dfg.insts[term] {
            InstructionData::Jump { destination, .. } => *destination,
            InstructionData::Brif { blocks, .. } => blocks[which],
            _ => unreachable!("a branch"),
        }
    }

    /// Address `addr` (an `i64`) plus `off`: WebAssembly's `i32` and the offset left.
    fn address(&mut self, x: &mut InstructionSink, addr: Value, off: i32) -> u64 {
        self.get(x, addr);
        if off < 0 {
            x.i64_const(off as i64).i64_add();
        }
        x.i32_wrap_i64();
        off.max(0) as u64
    }

    /// An `i32`, non-zero if `v` is (a condition).
    fn condition(&mut self, x: &mut InstructionSink, v: Value) {
        let t = self.ty(v);
        self.get(x, v);
        if t == types::I64 {
            x.i64_const(0).i64_ne();
        }
    }

    /// Writes instruction `inst` (not a terminator), unless its result is computed where
    /// it is used.
    fn inst(&mut self, x: &mut InstructionSink, inst: ir::Inst) -> Result<(), ()> {
        let dfg = &self.func.dfg;
        if let [r] = dfg.inst_results(inst)
            && self.inline.contains(r)
        {
            return Ok(());
        }
        self.emit(x, inst, true)
    }

    /// The code of instruction `inst`: its result left on the stack, or assigned to its
    /// local if `keep`.
    fn emit(&mut self, x: &mut InstructionSink, inst: ir::Inst, keep: bool) -> Result<(), ()> {
        let dfg = &self.func.dfg;
        let data = dfg.insts[inst];
        if keep
            && (self.dead_stores.contains(&inst)
                || (!effect(data.opcode())
                    && dfg
                        .inst_results(inst)
                        .iter()
                        .all(|r| !self.live.contains(r))))
        {
            return Ok(());
        }
        let args: Vec<Value> = dfg.inst_args(inst).to_vec();
        let results: Vec<Value> = dfg.inst_results(inst).to_vec();
        let rty = results.first().map(|r| dfg.value_type(*r));
        // An `i8` result kept in the low byte.
        let mask8 = |x: &mut InstructionSink, t: Option<Type>| {
            if t == Some(types::I8) {
                x.i32_const(0xff).i32_and();
            }
        };
        match data {
            InstructionData::UnaryImm { imm, .. } => {
                let n = imm.bits();
                match rty {
                    Some(types::I64) => x.i64_const(n),
                    Some(types::I8) => x.i32_const((n & 0xff) as i32),
                    _ => x.i32_const(n as i32),
                };
            }
            InstructionData::UnaryIeee64 { imm, .. } => {
                x.f64_const(f64::from_bits(imm.bits()).into());
            }
            InstructionData::Binary { opcode, .. }
                if matches!(
                    opcode,
                    Opcode::SaddOverflow | Opcode::SsubOverflow | Opcode::SmulOverflow
                ) =>
            {
                if self.ty(args[0]) != types::I64 {
                    return Err(());
                }
                let [a, b, r] = self.s64;
                self.get(x, args[0]);
                x.local_set(a);
                self.get(x, args[1]);
                x.local_set(b);
                x.local_get(a).local_get(b);
                match opcode {
                    Opcode::SaddOverflow => x.i64_add(),
                    Opcode::SsubOverflow => x.i64_sub(),
                    _ => x.i64_mul(),
                };
                x.local_set(r);
                // The overflow flag (an `i8`).
                match opcode {
                    Opcode::SaddOverflow => {
                        // (a ^ r) & (b ^ r) < 0
                        x.local_get(a).local_get(r).i64_xor();
                        x.local_get(b).local_get(r).i64_xor();
                        x.i64_and().i64_const(0).i64_lt_s();
                    }
                    Opcode::SsubOverflow => {
                        // (a ^ b) & (a ^ r) < 0
                        x.local_get(a).local_get(b).i64_xor();
                        x.local_get(a).local_get(r).i64_xor();
                        x.i64_and().i64_const(0).i64_lt_s();
                    }
                    _ => {
                        // a != 0 and (a == -1 ? b == MIN : r / a != b), the division by 1
                        // where it would trap.
                        x.local_get(a).i64_const(0).i64_ne();
                        x.local_get(a).i64_const(-1).i64_eq();
                        x.local_get(b).i64_const(i64::MIN).i64_eq();
                        x.i32_and();
                        x.local_get(a).i64_const(-1).i64_ne();
                        x.local_get(r);
                        x.local_get(a).i64_const(1);
                        x.local_get(a).i64_const(0).i64_ne();
                        x.local_get(a).i64_const(-1).i64_ne();
                        x.i32_and().select();
                        x.i64_div_s().local_get(b).i64_ne();
                        x.i32_and().i32_or().i32_and();
                    }
                }
                self.set(x, results[1]);
                x.local_get(r);
                self.set(x, results[0]);
                return Ok(());
            }
            InstructionData::Binary { opcode, .. } => {
                let t = self.ty(args[0]);
                self.get(x, args[0]);
                self.get(x, args[1]);
                match (opcode, t) {
                    (Opcode::Iadd, types::I64) => x.i64_add(),
                    (Opcode::Isub, types::I64) => x.i64_sub(),
                    (Opcode::Imul, types::I64) => x.i64_mul(),
                    (Opcode::Sdiv, types::I64) => x.i64_div_s(),
                    (Opcode::Srem, types::I64) => x.i64_rem_s(),
                    (Opcode::Band, types::I64) => x.i64_and(),
                    (Opcode::Bor, types::I64) => x.i64_or(),
                    (Opcode::Bxor, types::I64) => x.i64_xor(),
                    (Opcode::Iadd, types::I8 | types::I32) => x.i32_add(),
                    (Opcode::Isub, types::I8 | types::I32) => x.i32_sub(),
                    (Opcode::Imul, types::I8 | types::I32) => x.i32_mul(),
                    (Opcode::Band, types::I8 | types::I32) => x.i32_and(),
                    (Opcode::Bor, types::I8 | types::I32) => x.i32_or(),
                    (Opcode::Bxor, types::I8 | types::I32) => x.i32_xor(),
                    (Opcode::Fadd, types::F64) => x.f64_add(),
                    (Opcode::Fsub, types::F64) => x.f64_sub(),
                    (Opcode::Fmul, types::F64) => x.f64_mul(),
                    (Opcode::Fdiv, types::F64) => x.f64_div(),
                    _ => return Err(()),
                };
                mask8(x, rty);
            }
            InstructionData::Unary { opcode, arg } => {
                let from = self.ty(arg);
                let to = rty.ok_or(())?;
                match opcode {
                    Opcode::Ineg if to == types::I64 => {
                        x.i64_const(0);
                        self.get(x, arg);
                        x.i64_sub();
                    }
                    Opcode::Iabs if to == types::I64 => {
                        // v < 0 ? -v : v
                        let s = self.s64[0];
                        self.get(x, arg);
                        x.local_set(s);
                        x.i64_const(0).local_get(s).i64_sub();
                        x.local_get(s);
                        x.local_get(s).i64_const(0).i64_lt_s();
                        x.select();
                    }
                    Opcode::Fneg if to == types::F64 => {
                        self.get(x, arg);
                        x.f64_neg();
                    }
                    Opcode::FcvtFromSint if from == types::I64 && to == types::F64 => {
                        self.get(x, arg);
                        x.f64_convert_i64_s();
                    }
                    Opcode::Uextend => {
                        self.get(x, arg);
                        if to == types::I64 {
                            x.i64_extend_i32_u();
                        }
                    }
                    Opcode::Sextend => {
                        self.get(x, arg);
                        if from == types::I8 {
                            x.i32_extend8_s();
                        }
                        if to == types::I64 {
                            x.i64_extend_i32_s();
                        }
                    }
                    Opcode::Ireduce => {
                        self.get(x, arg);
                        if from == types::I64 {
                            x.i32_wrap_i64();
                        }
                        mask8(x, Some(to));
                    }
                    _ => return Err(()),
                }
            }
            InstructionData::LoadNoOffset {
                opcode: Opcode::Bitcast,
                arg,
                ..
            } => {
                self.get(x, arg);
                match (self.ty(arg), rty) {
                    (types::I64, Some(types::F64)) => x.f64_reinterpret_i64(),
                    (types::F64, Some(types::I64)) => x.i64_reinterpret_f64(),
                    _ => return Err(()),
                };
            }
            InstructionData::IntCompare {
                cond, args: [a, b], ..
            } => {
                let t = self.ty(a);
                let signed = matches!(
                    cond,
                    IntCC::SignedLessThan
                        | IntCC::SignedLessThanOrEqual
                        | IntCC::SignedGreaterThan
                        | IntCC::SignedGreaterThanOrEqual
                );
                for v in [a, b] {
                    self.get(x, v);
                    if t == types::I8 && signed {
                        x.i32_extend8_s();
                    }
                }
                if t == types::I64 {
                    match cond {
                        IntCC::Equal => x.i64_eq(),
                        IntCC::NotEqual => x.i64_ne(),
                        IntCC::SignedLessThan => x.i64_lt_s(),
                        IntCC::SignedLessThanOrEqual => x.i64_le_s(),
                        IntCC::SignedGreaterThan => x.i64_gt_s(),
                        IntCC::SignedGreaterThanOrEqual => x.i64_ge_s(),
                        IntCC::UnsignedLessThan => x.i64_lt_u(),
                        IntCC::UnsignedLessThanOrEqual => x.i64_le_u(),
                        IntCC::UnsignedGreaterThan => x.i64_gt_u(),
                        IntCC::UnsignedGreaterThanOrEqual => x.i64_ge_u(),
                    };
                } else {
                    match cond {
                        IntCC::Equal => x.i32_eq(),
                        IntCC::NotEqual => x.i32_ne(),
                        IntCC::SignedLessThan => x.i32_lt_s(),
                        IntCC::SignedLessThanOrEqual => x.i32_le_s(),
                        IntCC::SignedGreaterThan => x.i32_gt_s(),
                        IntCC::SignedGreaterThanOrEqual => x.i32_ge_s(),
                        IntCC::UnsignedLessThan => x.i32_lt_u(),
                        IntCC::UnsignedLessThanOrEqual => x.i32_le_u(),
                        IntCC::UnsignedGreaterThan => x.i32_gt_u(),
                        IntCC::UnsignedGreaterThanOrEqual => x.i32_ge_u(),
                    };
                }
            }
            InstructionData::FloatCompare {
                cond, args: [a, b], ..
            } => {
                self.get(x, a);
                self.get(x, b);
                match cond {
                    FloatCC::Equal => x.f64_eq(),
                    FloatCC::NotEqual => x.f64_ne(),
                    FloatCC::LessThan => x.f64_lt(),
                    FloatCC::LessThanOrEqual => x.f64_le(),
                    FloatCC::GreaterThan => x.f64_gt(),
                    FloatCC::GreaterThanOrEqual => x.f64_ge(),
                    _ => return Err(()),
                };
            }
            InstructionData::Ternary {
                opcode: Opcode::Select,
                args: [c, a, b],
            } => {
                self.get(x, a);
                self.get(x, b);
                self.condition(x, c);
                x.select();
            }
            InstructionData::Load {
                opcode: Opcode::Load,
                arg,
                offset,
                ..
            } => {
                let off = self.address(x, arg, offset.into());
                match rty {
                    Some(types::I8) => x.i32_load8_u(mem(off, 0)),
                    Some(types::I32) => x.i32_load(mem(off, 0)),
                    Some(types::I64) => x.i64_load(mem(off, 0)),
                    Some(types::F64) => x.f64_load(mem(off, 0)),
                    _ => return Err(()),
                };
            }
            InstructionData::Store {
                opcode: Opcode::Store,
                args: [v, addr],
                offset,
                ..
            } => {
                let off = self.address(x, addr, offset.into());
                self.get(x, v);
                match self.ty(v) {
                    types::I8 => x.i32_store8(mem(off, 0)),
                    types::I32 => x.i32_store(mem(off, 0)),
                    types::I64 => x.i64_store(mem(off, 0)),
                    types::F64 => x.f64_store(mem(off, 0)),
                    _ => return Err(()),
                };
                return Ok(());
            }
            InstructionData::StackAddr {
                stack_slot, offset, ..
            } => {
                let at =
                    self.slots[stack_slot.as_u32() as usize] as i64 + i64::from(i32::from(offset));
                self.frame(x);
                x.local_get(self.fp).i64_extend_i32_u();
                x.i64_const(at).i64_add();
            }
            InstructionData::Call { func_ref, .. } => {
                return self.call(x, func_ref, &args, &results);
            }
            _ => return Err(()),
        }
        match results.as_slice() {
            [_] if !keep => Ok(()),
            [r] => {
                self.set(x, *r);
                Ok(())
            }
            _ => Err(()),
        }
    }

    /// A call: of a function of this module, of one compiled before (through the table),
    /// or of a Rust helper (through the table, its addresses as `i32`).
    fn call(
        &mut self,
        x: &mut InstructionSink,
        func_ref: ir::FuncRef,
        args: &[Value],
        results: &[Value],
    ) -> Result<(), ()> {
        let ext = &self.func.dfg.ext_funcs[func_ref];
        let ExternalName::User(name) = ext.name else {
            return Err(());
        };
        let id = FuncId::from_u32(self.func.params.user_named_funcs()[name].index);
        if let Some(helper) = self.module.imports.get(&id) {
            let (index, (ps, rs)) = self.module.helpers[helper].clone();
            if ps.len() != args.len() || rs.len() != results.len() {
                return Err(());
            }
            for (a, p) in args.iter().zip(&ps) {
                let t = self.ty(*a);
                self.get(x, *a);
                match (t, *p) {
                    (types::I64, ValType::I32) => {
                        x.i32_wrap_i64();
                    }
                    (types::I64, ValType::I64)
                    | (types::I8 | types::I32, ValType::I32)
                    | (types::F64, ValType::F64) => {}
                    _ => return Err(()),
                }
            }
            let ty = self.types.index(ps, rs.clone());
            x.i32_const(index as i32).call_indirect(0, ty);
            for (r, w) in results.iter().zip(&rs).rev() {
                match (self.ty(*r), *w) {
                    (types::I64, ValType::I32) => {
                        x.i64_extend_i32_u();
                    }
                    (types::I8, ValType::I32) => {
                        x.i32_const(0xff).i32_and();
                    }
                    (types::I64, ValType::I64)
                    | (types::I32, ValType::I32)
                    | (types::F64, ValType::F64) => {}
                    _ => return Err(()),
                }
                self.set(x, *r);
            }
            return Ok(());
        }
        for a in args {
            self.get(x, *a);
        }
        if let Some(i) = self.local.get(&id) {
            x.call(*i);
        } else {
            let done = self.module.done.get(&id).ok_or(())?;
            let ty = self.types.index(done.ty.0.clone(), done.ty.1.clone());
            x.i32_const(done.func as i32).call_indirect(0, ty);
        }
        for r in results.iter().rev() {
            self.set(x, *r);
        }
        Ok(())
    }
}

impl Blocks for Translator<'_> {
    fn block(&mut self, x: &mut InstructionSink, bi: usize) -> Exit {
        let b = self.blocks[bi];
        if let Some((index, args)) = self.stubs.get(&b).cloned() {
            // A cold region: its function returns this one's result (and gives the frame
            // back).
            for a in args {
                self.get(x, a);
            }
            x.local_get(self.fp);
            x.call(index);
            x.return_();
            return Exit::Return;
        }
        let insts: Vec<ir::Inst> = self.func.layout.block_insts(b).collect();
        let Some((&term, body)) = insts.split_last() else {
            self.failed = true;
            return Exit::Return;
        };
        for &inst in body {
            if self.inst(x, inst).is_err() {
                self.failed = true;
                x.unreachable();
                return Exit::Return;
            }
        }
        let data = self.func.dfg.insts[term];
        let pool = &self.func.dfg.value_lists;
        match data {
            InstructionData::Jump { destination, .. } => {
                Exit::Jump(self.number[&destination.block(pool)])
            }
            InstructionData::Brif { arg, blocks, .. } => {
                let (t, e) = (
                    self.number[&blocks[0].block(pool)],
                    self.number[&blocks[1].block(pool)],
                );
                self.condition(x, arg);
                Exit::Cond(t, e)
            }
            InstructionData::MultiAry {
                opcode: Opcode::Return,
                ..
            } => {
                let args: Vec<Value> = self.func.dfg.inst_args(term).to_vec();
                self.epilogue(x);
                for a in args {
                    self.get(x, a);
                }
                x.return_();
                Exit::Return
            }
            _ => {
                self.failed = true;
                x.unreachable();
                Exit::Return
            }
        }
    }

    /// The arguments of the edge: the target's parameters assigned.
    fn edge(&mut self, x: &mut InstructionSink, from: usize, which: usize) {
        let call = self.block_call(self.blocks[from], which);
        let pool = &self.func.dfg.value_lists;
        let target = call.block(pool);
        let args: Option<Vec<Value>> = call
            .args(pool)
            .map(|a| match a {
                ir::BlockArg::Value(v) => Some(v),
                _ => None,
            })
            .collect();
        let Some(args) = args else {
            self.failed = true;
            return;
        };
        // The parameters used, all read, then all written (a parallel assignment).
        let pairs: Vec<(Value, Value)> = self
            .func
            .dfg
            .block_params(target)
            .iter()
            .zip(args)
            .filter(|(p, _)| self.live.contains(*p))
            .map(|(p, a)| (*p, a))
            .collect();
        for (_, a) in &pairs {
            self.get(x, *a);
        }
        for (p, _) in pairs.iter().rev() {
            self.set(x, *p);
        }
    }
}

fn mem(offset: u64, align: u32) -> MemArg {
    MemArg {
        offset,
        align,
        memory_index: 0,
    }
}

// ─── Control flow ──────────────────────────────────────────────────────────

/// How a block ends.
enum Exit {
    Jump(u32),
    /// On the `i32` its code pushed: the first block if non-zero, the second otherwise.
    Cond(u32, u32),
    /// Its code returned (or cannot go on).
    Return,
}

/// The code of a function's blocks, for `control_flow`.
trait Blocks {
    /// The code of block `b`, then of its terminator: its condition pushed, or its return.
    fn block(&mut self, x: &mut InstructionSink, b: usize) -> Exit;

    /// Before leaving block `from` for its successor `which` (0: the jump, or the first
    /// of a condition; 1: the second).
    fn edge(&mut self, _x: &mut InstructionSink, _from: usize, _which: usize) {}
}

/// Writes the control flow of the blocks (`succ`: their successors; only those `reached`
/// can run). Structured when the graph is reducible: one `loop` per loop header, one
/// `block` per block reached by several forward edges (a merge), the others written
/// inline where they are branched to (N. Ramsey, "Beyond Relooper", 2022). Otherwise a
/// loop over a `br_table` on the block number (`pc`).
fn control_flow(
    x: &mut InstructionSink,
    succ: &[Vec<u32>],
    reached: &[bool],
    pc: u32,
    blocks: &mut impl Blocks,
) {
    match Structure::new(succ, reached).filter(|_| STRUCTURED.with(Cell::get)) {
        Some(st) => {
            let mut labels = Vec::new();
            st.tree(x, &mut labels, 0, blocks);
        }
        None => {
            DISPATCHED.with(|c| c.set(c.get() + 1));
            dispatch(x, succ.len() as u32, reached, pc, blocks);
        }
    }
}

/// The dispatch loop: `pc` the block to run.
fn dispatch(x: &mut InstructionSink, n: u32, reached: &[bool], pc: u32, blocks: &mut impl Blocks) {
    x.loop_(BlockType::Empty);
    for _ in 0..n {
        x.block(BlockType::Empty);
    }
    x.local_get(pc);
    x.br_table(0..n, n - 1);
    x.end();
    for (bi, &reach) in reached.iter().enumerate() {
        // Depth of the dispatch loop from this block's code.
        let to_loop = n - 1 - bi as u32;
        let next = bi as u32 + 1;
        let jump = |x: &mut InstructionSink, target: u32, extra: u32| {
            if target != next || extra > 0 {
                x.i32_const(target as i32).local_set(pc).br(to_loop + extra);
            }
        };
        if !reach {
            x.unreachable();
        } else {
            match blocks.block(x, bi) {
                Exit::Jump(t) => {
                    blocks.edge(x, bi, 0);
                    jump(x, t, 0);
                }
                Exit::Cond(t, e) => {
                    x.if_(BlockType::Empty);
                    blocks.edge(x, bi, 0);
                    jump(x, t, 1);
                    x.end();
                    blocks.edge(x, bi, 1);
                    jump(x, e, 0);
                }
                Exit::Return => {}
            }
        }
        if bi + 1 < n as usize {
            x.end();
        }
    }
    x.end();
}

/// Labels around the code being written, innermost last.
#[derive(Clone, Copy, PartialEq)]
enum Label {
    /// The `loop` of a loop header: a branch to it goes back to the header.
    Loop(u32),
    /// The `block` followed by a merge block: a branch to it goes to that block.
    Block(u32),
    If,
}

/// A reducible control flow graph, for structured code.
struct Structure {
    /// Reverse postorder number of each block reached.
    order: Vec<u32>,
    header: Vec<bool>,
    merge: Vec<bool>,
    /// Children in the dominator tree.
    children: Vec<Vec<u32>>,
}

const UNREACHED: u32 = u32::MAX;

impl Structure {
    fn new(succ: &[Vec<u32>], reached: &[bool]) -> Option<Structure> {
        let n = succ.len();
        let ok = |b: u32| reached[b as usize];
        // Reverse postorder (iterative depth-first search).
        let mut post = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        let mut stack: Vec<(u32, usize)> = vec![(0, 0)];
        seen[0] = true;
        while let Some(&mut (b, ref mut k)) = stack.last_mut() {
            let ss = &succ[b as usize];
            if *k < ss.len() {
                let s = ss[*k];
                *k += 1;
                if ok(s) && !seen[s as usize] {
                    seen[s as usize] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
                stack.pop();
            }
        }
        let rpo: Vec<u32> = post.iter().rev().copied().collect();
        let mut order = vec![UNREACHED; n];
        for (i, &b) in rpo.iter().enumerate() {
            order[b as usize] = i as u32;
        }
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &b in &rpo {
            for &s in &succ[b as usize] {
                if order[s as usize] != UNREACHED {
                    preds[s as usize].push(b);
                }
            }
        }
        // Immediate dominators (Cooper, Harvey, Kennedy).
        let mut idom = vec![UNREACHED; n];
        idom[0] = 0;
        let intersect = |idom: &[u32], mut a: u32, mut b: u32| {
            while a != b {
                while order[a as usize] > order[b as usize] {
                    a = idom[a as usize];
                }
                while order[b as usize] > order[a as usize] {
                    b = idom[b as usize];
                }
            }
            a
        };
        loop {
            let mut changed = false;
            for &b in &rpo[1..] {
                let mut new = UNREACHED;
                for &q in &preds[b as usize] {
                    if idom[q as usize] == UNREACHED {
                        continue;
                    }
                    new = if new == UNREACHED {
                        q
                    } else {
                        intersect(&idom, q, new)
                    };
                }
                if idom[b as usize] != new {
                    idom[b as usize] = new;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let dominates = |a: u32, mut b: u32| loop {
            if a == b {
                return true;
            }
            if b == 0 {
                return false;
            }
            b = idom[b as usize];
        };
        let mut header = vec![false; n];
        let mut forward_in = vec![0u32; n];
        for &b in &rpo {
            for &s in &succ[b as usize] {
                if order[s as usize] == UNREACHED {
                    continue;
                }
                if order[s as usize] <= order[b as usize] {
                    // A back edge: to a block dominating its source, or the graph is not
                    // reducible.
                    if !dominates(s, b) {
                        return None;
                    }
                    header[s as usize] = true;
                } else {
                    forward_in[s as usize] += 1;
                }
            }
        }
        let merge: Vec<bool> = forward_in.iter().map(|&k| k >= 2).collect();
        let mut children = vec![Vec::new(); n];
        let mut depth = vec![0u32; n];
        for &b in &rpo[1..] {
            let d = idom[b as usize];
            children[d as usize].push(b);
            depth[b as usize] = depth[d as usize] + 1;
            // (The code is written recursively along the dominator tree.)
            if depth[b as usize] > 400 {
                return None;
            }
        }
        Some(Structure {
            order,
            header,
            merge,
            children,
        })
    }

    /// The code of block `b` and of the blocks it dominates.
    fn tree(
        &self,
        x: &mut InstructionSink,
        labels: &mut Vec<Label>,
        b: u32,
        blocks: &mut impl Blocks,
    ) {
        let mut merges: Vec<u32> = self.children[b as usize]
            .iter()
            .copied()
            .filter(|&c| self.merge[c as usize])
            .collect();
        // The one written last (highest number) gets the outermost `block`.
        merges.sort_by_key(|&c| std::cmp::Reverse(self.order[c as usize]));
        if self.header[b as usize] {
            x.loop_(BlockType::Empty);
            labels.push(Label::Loop(b));
            self.within(x, labels, b, &merges, blocks);
            labels.pop();
            x.end();
        } else {
            self.within(x, labels, b, &merges, blocks);
        }
    }

    fn within(
        &self,
        x: &mut InstructionSink,
        labels: &mut Vec<Label>,
        b: u32,
        merges: &[u32],
        blocks: &mut impl Blocks,
    ) {
        if let Some((&y, rest)) = merges.split_first() {
            x.block(BlockType::Empty);
            labels.push(Label::Block(y));
            self.within(x, labels, b, rest, blocks);
            labels.pop();
            x.end();
            self.tree(x, labels, y, blocks);
            return;
        }
        match blocks.block(x, b as usize) {
            Exit::Jump(t) => {
                blocks.edge(x, b as usize, 0);
                self.branch(x, labels, b, t, blocks);
            }
            Exit::Cond(t, e) => {
                x.if_(BlockType::Empty);
                labels.push(Label::If);
                blocks.edge(x, b as usize, 0);
                self.branch(x, labels, b, t, blocks);
                x.else_();
                blocks.edge(x, b as usize, 1);
                self.branch(x, labels, b, e, blocks);
                labels.pop();
                x.end();
            }
            Exit::Return => {}
        }
    }

    /// From block `from` to block `to`: a branch, or `to`'s code here.
    fn branch(
        &self,
        x: &mut InstructionSink,
        labels: &mut Vec<Label>,
        from: u32,
        to: u32,
        blocks: &mut impl Blocks,
    ) {
        let label = if self.order[to as usize] <= self.order[from as usize] {
            Label::Loop(to)
        } else if self.merge[to as usize] {
            Label::Block(to)
        } else {
            self.tree(x, labels, to, blocks);
            return;
        };
        let at = labels
            .iter()
            .rposition(|l| *l == label)
            .expect("enclosing label");
        x.br((labels.len() - 1 - at) as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn structure(succ: &[&[u32]]) -> Option<Structure> {
        let succ: Vec<Vec<u32>> = succ.iter().map(|s| s.to_vec()).collect();
        Structure::new(&succ, &vec![true; succ.len()])
    }

    #[test]
    fn loops_and_merges() {
        // 0 → 1; 1 → 2 | 4; 2 → 3 | 1 (back); 3 → 1 (back); 4: return.
        let s = structure(&[&[1], &[2, 4], &[3, 1], &[1], &[]]).unwrap();
        assert_eq!(s.header, [false, true, false, false, false]);
        assert_eq!(s.merge, [false; 5]);
        // An `if` whose branches meet: 0 → 1 | 2; 1 → 3; 2 → 3.
        let s = structure(&[&[1, 2], &[3], &[3], &[]]).unwrap();
        assert_eq!(s.merge, [false, false, false, true]);
        let mut c = s.children[0].clone();
        c.sort();
        assert_eq!(c, [1, 2, 3]);
    }

    #[test]
    fn irreducible_graphs_are_not_structured() {
        // Two entries into the loop 1 ⇄ 2.
        assert!(structure(&[&[1, 2], &[2], &[1]]).is_none());
    }

    #[test]
    fn unreached_blocks_are_left_out() {
        let succ = vec![vec![2], vec![1], vec![]];
        let s = Structure::new(&succ, &[true, false, true]).unwrap();
        assert_eq!(s.order[1], UNREACHED);
        assert!(!s.header[1]);
    }

    #[test]
    fn helper_signatures_follow_rust_types() {
        extern "C" fn f(_: *const u8, _: u64, _: i32, _: f64) -> i8 {
            0
        }
        extern "C" fn g(_: *mut u8) {}
        assert_eq!(
            sig_of(f as extern "C" fn(_, _, _, _) -> _),
            (
                vec![PTR, ValType::I64, ValType::I32, ValType::F64],
                vec![ValType::I32]
            )
        );
        assert_eq!(sig_of(g as extern "C" fn(_)), (vec![PTR], vec![]));
    }
}
