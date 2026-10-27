//! IR interpreter.
//!
//! Each call reserves a window of registers on a shared value stack (`stack`), then runs
//! the function's basic blocks. Typed registers always hold a value of the right type
//! (`Value::Int` in an `int` register), so specialized instructions (`IBin`) need not
//! check the type.

use crate::ast::{BinOp, TypeDeclKind, Visibility};
use crate::diagnostic::{Error, ErrorKind, Raised};
use crate::globals::Globals;
use crate::intern::{FxHashMap, Interner, Symbol};
use crate::ir::*;
use crate::objects::{ConstructCtx, EnumInfo, Syms, TypeInfo};
use crate::ops::{Builtin, METHODS, Method, R, list_index, rt_err, short, type_err};
use crate::span::Span;
use crate::value::{Function, HashMap, Range, Record, Value};
use std::cell::RefCell;
use std::rc::Rc;

/// Cell of a variable captured by a closure.
pub type Cell = Rc<RefCell<Value>>;

/// Script function or lambda: its code and its captured variables.
pub struct Closure {
    pub proto: Rc<Proto>,
    pub upvals: Box<[Cell]>,
    /// Constructions in progress when it was created (nested `do` blocks).
    pub ctx: Rc<[u32]>,
}

/// Evaluated arguments of a call.
pub struct CallArgs {
    pub pos: Vec<Value>,
    pub named: Vec<(Symbol, Value)>,
    /// Literal shape of each argument (positional then named), used to convert a record
    /// literal to the expected type. Empty for a call from Rust or a built-in method.
    pub shapes: Vec<Option<Rc<Shape>>>,
}

impl CallArgs {
    pub fn plain(pos: Vec<Value>) -> Self {
        CallArgs {
            pos,
            named: Vec::new(),
            shapes: Vec::new(),
        }
    }
}

/// State of a `for` loop iterator: the loop sees the elements as they were when it
/// started.
pub enum IterState {
    Values(std::vec::IntoIter<Value>),
    Range(std::ops::RangeInclusive<i64>),
    /// Over a list, not copied: (list, next index). Before the list is modified, the
    /// rest of it is copied (`Values`, see [`before_list_write`]).
    List(Rc<RefCell<Vec<Value>>>, usize),
}

thread_local! {
    /// Iterators over a list not copied (`IterState::List`): (address of the list,
    /// iterator).
    static LIST_ITERS: RefCell<Vec<(usize, std::rc::Weak<RefCell<IterState>>)>> =
        const { RefCell::new(Vec::new()) };
}

/// Iterator over `l`, without copying it: registered, so that it copies the rest before
/// `l` is modified.
fn list_iter(l: &Rc<RefCell<Vec<Value>>>) -> Value {
    let it = Rc::new(RefCell::new(IterState::List(l.clone(), 0)));
    LIST_ITERS.with(|iters| {
        let mut iters = iters.borrow_mut();
        // Those that ended (their register reused) are forgotten from time to time.
        if iters.len() >= 32 {
            iters.retain(|(_, w)| w.strong_count() > 0);
        }
        iters.push((Rc::as_ptr(l) as usize, Rc::downgrade(&it)));
    });
    Value::Iter(it)
}

/// To call before modifying list `l` (`<<`, `push`, `pop`, `xs[i] = v`): the loops
/// that walk it copy what they have left, so they see it as it was.
#[inline]
pub(crate) fn before_list_write(l: &Rc<RefCell<Vec<Value>>>) {
    LIST_ITERS.with(|iters| {
        let Ok(mut iters) = iters.try_borrow_mut() else {
            return;
        };
        if iters.is_empty() {
            return;
        }
        let p = Rc::as_ptr(l) as usize;
        if !iters.iter().any(|(q, _)| *q == p) {
            return;
        }
        iters.retain(|(q, w)| {
            if *q != p {
                return w.strong_count() > 0;
            }
            if let Some(it) = w.upgrade()
                && let Ok(mut st) = it.try_borrow_mut()
                && let IterState::List(list, i) = &*st
            {
                let rest: Vec<Value> = list.borrow().get(*i..).unwrap_or_default().to_vec();
                *st = IterState::Values(rest.into_iter());
            }
            false
        });
    });
}

/// Calls between two checks of whether a collection of cycles is due.
const GC_CALLS: u32 = 1024;

pub(crate) const RED_ZONE: usize = 256 * 1024;
pub(crate) const STACK_GROW: usize = 2 * 1024 * 1024;
/// The stack is only checked once every `STACK_CHECK` call levels: the red zone easily
/// covers those calls (under 32 KiB each).
const STACK_CHECK: usize = 8;

/// Interpreter state: names, global variables, `print` output.
pub struct Runtime {
    pub interner: Interner,
    pub globals: Globals,
    pub output: Box<dyn FnMut(&str)>,
    /// Maximum call depth (guards against infinite recursion).
    pub max_depth: usize,
    pub(crate) depth: usize,
    pub(crate) methods: FxHashMap<Symbol, (Method, &'static str)>,
    /// `type` constructions in progress (nested `Server do … end` blocks).
    pub(crate) constructs: Vec<ConstructCtx>,
    pub(crate) next_ctx: u32,
    pub(crate) syms: Syms,
    /// Free register windows, reused from one call to the next (no per-call allocation
    /// in steady state).
    frames: Vec<Vec<Value>>,
    pub(crate) no_ctx: Rc<[u32]>,
    /// Native compilation of typed functions (created on first use).
    pub jit_enabled: bool,
    /// Baseline compilation from the first call (tests).
    pub jit_eager: bool,
    /// String interpolation buffer.
    concat_buf: String,
    /// Errors kept by `rescue` / `ensure` handlers, by call (`Term::Reraise`).
    pub(crate) caught: Vec<Option<Error>>,
    /// Calls left before checking whether a collection of cycles is due.
    gc_calls: u32,
    #[cfg(feature = "jit")]
    pub(crate) jit: Option<crate::jit::Jit>,
    /// Error raised while running baseline native code, waiting to be propagated.
    #[cfg(feature = "jit")]
    pending: Option<Error>,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

/// Attaches the script source to an error that has none yet.
fn attach_src(mut e: Error, sc: &Script) -> Error {
    if e.src.is_none() {
        e.src = Some((sc.name.clone(), sc.src.clone()));
    }
    e
}

impl Runtime {
    pub fn new() -> Self {
        let mut interner = Interner::new();
        let methods = METHODS
            .iter()
            .map(|&(name, m)| (interner.intern(name), (m, name)))
            .collect();
        let syms = Syms::new(&mut interner);
        let mut globals = Globals::default();
        for (name, b) in Builtin::ALL {
            let sym = interner.intern(name);
            globals.define(sym, Value::Function(Rc::new(Function::Builtin(b))), false);
        }
        Runtime {
            interner,
            globals,
            output: Box::new(|s| print!("{s}")),
            max_depth: 1000,
            depth: 0,
            methods,
            constructs: Vec::new(),
            next_ctx: 0,
            syms,
            frames: Vec::new(),
            no_ctx: Rc::from([]),
            jit_enabled: cfg!(feature = "jit"),
            jit_eager: false,
            concat_buf: String::new(),
            caught: Vec::new(),
            gc_calls: GC_CALLS,
            #[cfg(feature = "jit")]
            jit: None,
            #[cfg(feature = "jit")]
            pending: None,
        }
    }

    pub(crate) fn name(&self, s: Symbol) -> &str {
        self.interner.resolve(s)
    }

    /// Runs a script's main program.
    pub fn run(&mut self, main: &Rc<Proto>) -> Result<Value, Error> {
        self.depth = 0;
        let ctx = self.no_ctx.clone();
        self.invoke(
            main,
            &[],
            &ctx,
            None,
            CallArgs::plain(Vec::new()),
            Span::default(),
        )
        .map_err(|e| attach_src(e, &main.script))
    }

    // ─── Calls ───────────────────────────────────────────────────────────────

    /// Calls a value: function, lambda, Rust function, or type (construction).
    pub fn call_value(&mut self, f: &Value, args: CallArgs, span: Span) -> R {
        let func = match f {
            Value::Function(func) => func,
            Value::Type(info) => return self.construct(info, args, None, span),
            _ => return type_err(format!("{} is not a function", f.type_label()), span),
        };
        match &**func {
            Function::Native(n) => {
                if let Some(&(name, _)) = args.named.first() {
                    return rt_err(
                        format!(
                            "`{}` (Rust function) does not accept named arguments (`{}`)",
                            n.name,
                            self.name(name)
                        ),
                        span,
                    );
                }
                (n.f)(&args.pos).map_err(|msg| Error::runtime(msg, span))
            }
            Function::Builtin(b) => self.builtin(*b, args.pos, span),
            Function::Script(c) => self.invoke(&c.proto, &c.upvals, &c.ctx, None, args, span),
        }
    }

    /// Calls a declared method (`type`, `class`, `enum`) with `self` = `this`.
    pub(crate) fn call_method(
        &mut self,
        this: Value,
        m: &Rc<Proto>,
        args: CallArgs,
        span: Span,
    ) -> R {
        let ctx = self.no_ctx.clone();
        self.invoke(m, &[], &ctx, Some(this), args, span)
    }

    /// Runs a function: binds the arguments, then runs its blocks.
    pub(crate) fn invoke(
        &mut self,
        proto: &Rc<Proto>,
        upvals: &[Cell],
        ctx: &Rc<[u32]>,
        this: Option<Value>,
        args: CallArgs,
        span: Span,
    ) -> R {
        self.invoke_with(proto, upvals, ctx, span, |rt, regs| {
            rt.bind(proto, regs, this, args, span)
        })
    }

    /// Fast call: positional arguments, as many as parameters, read directly from the
    /// caller's registers.
    pub(crate) fn invoke_regs(&mut self, c: &Closure, args: &[Value], span: Span) -> R {
        #[cfg(feature = "jit")]
        if let Some(r) = self.native_fast(&c.proto, None, args, span, &c.upvals) {
            return r;
        }
        #[cfg(feature = "jit")]
        if let Some(r) = self.invoke_native(c, args, span) {
            return r;
        }
        let proto = &c.proto;
        self.invoke_with(proto, &c.upvals, &c.ctx, span, |rt, regs| {
            for (p, v) in proto.params.iter().zip(args) {
                regs[p.reg as usize] = match &p.ty {
                    Some(ty) => rt.coerce(v.clone(), ty, None, p.span)?,
                    None => v.clone(),
                };
            }
            Ok(())
        })
    }

    fn invoke_with(
        &mut self,
        proto: &Rc<Proto>,
        upvals: &[Cell],
        ctx: &Rc<[u32]>,
        span: Span,
        bind: impl FnOnce(&mut Self, &mut [Value]) -> R<()>,
    ) -> R {
        if self.depth >= self.max_depth {
            return rt_err(
                format!(
                    "too many nested calls ({}): infinite recursion?",
                    self.max_depth
                ),
                span,
            );
        }
        // A safe point (no value is held other than by its references): every so often,
        // a collection of cycles if due.
        self.gc_calls = self.gc_calls.wrapping_sub(1);
        if self.gc_calls == 0 {
            self.gc_calls = GC_CALLS;
            crate::gc::maybe_collect();
        }
        self.depth += 1;
        let mut regs = self.frames.pop().unwrap_or_default();
        fill_nil(&mut regs, proto.regs.len());
        let check_stack = self.depth.is_multiple_of(STACK_CHECK);
        let body = || {
            bind(self, &mut regs)?;
            #[cfg(feature = "jit")]
            if let Some(r) = self.try_jit(proto, &regs, upvals) {
                return r;
            }
            self.exec(proto, upvals, ctx, &mut regs)
        };
        let r = if check_stack {
            stacker::maybe_grow(RED_ZONE, STACK_GROW, body)
        } else {
            body()
        };
        regs.clear();
        self.frames.push(regs);
        self.depth -= 1;
        r.map_err(|e| self.traced(e, proto, span))
    }

    /// Error leaving `proto`: source and call added to the trace.
    #[cold]
    fn traced(&self, mut e: Error, proto: &Proto, span: Span) -> Error {
        if proto.kind != ProtoKind::Main && !proto.outlined {
            e = attach_src(e, &proto.script);
            if e.trace.len() < 16 {
                e.trace.push((proto.name.to_string(), span));
            }
        }
        e
    }

    /// Stores the arguments in the parameter registers (checking annotations). A missing
    /// parameter that has a default value stays "not provided": the function's prologue
    /// computes it.
    fn bind(
        &mut self,
        proto: &Proto,
        regs: &mut [Value],
        this: Option<Value>,
        args: CallArgs,
        span: Span,
    ) -> R<()> {
        let CallArgs { pos, named, shapes } = args;
        let params = &proto.params;
        let lenient = matches!(proto.kind, ProtoKind::Lambda | ProtoKind::DoBlock);
        if let Some(t) = this {
            regs[0] = t;
        }
        // Common case: only positional arguments, as many as parameters.
        if named.is_empty() && pos.len() == params.len() && shapes.iter().all(Option::is_none) {
            for (p, v) in params.iter().zip(pos) {
                let v = match &p.ty {
                    Some(ty) => self.coerce(v, ty, None, p.span)?,
                    None => v,
                };
                regs[p.reg as usize] = v;
            }
            return Ok(());
        }
        if !lenient && pos.len() > params.len() {
            return rt_err(
                format!(
                    "`{}` expects {} argument{}, got {}",
                    proto.name,
                    params.len(),
                    if params.len() == 1 { "" } else { "s" },
                    pos.len()
                ),
                span,
            );
        }
        for (n, _) in &named {
            if !params.iter().any(|p| p.name == *n) {
                return rt_err(
                    format!("`{}` has no parameter `{}`", proto.name, self.name(*n)),
                    span,
                );
            }
        }
        let npos = pos.len();
        let mut pos = pos.into_iter().map(Some).collect::<Vec<_>>();
        let mut named: Vec<Option<(Symbol, Value)>> = named.into_iter().map(Some).collect();
        for (k, p) in params.iter().enumerate() {
            let given = pos.get_mut(k).and_then(Option::take);
            let named_i = named
                .iter()
                .position(|x| x.as_ref().is_some_and(|(n, _)| *n == p.name));
            if given.is_some() && named_i.is_some() {
                return rt_err(
                    format!("argument `{}` given twice", self.name(p.name)),
                    span,
                );
            }
            let (v, shape) = match (given, named_i) {
                (Some(v), _) => (v, shapes.get(k).cloned().flatten()),
                (None, Some(j)) => (
                    named[j].take().unwrap().1,
                    shapes.get(npos + j).cloned().flatten(),
                ),
                (None, None) if p.has_default => (Value::Uninit, None),
                (None, None) if lenient => (Value::Nil, None),
                (None, None) => {
                    return rt_err(
                        format!(
                            "missing argument `{}` for `{}`",
                            self.name(p.name),
                            proto.name
                        ),
                        span,
                    );
                }
            };
            let v = match (&p.ty, &v) {
                (_, Value::Uninit) => v,
                (Some(ty), _) => self.coerce(v, ty, shape.as_deref(), p.span)?,
                (None, _) => v,
            };
            regs[p.reg as usize] = v;
        }
        Ok(())
    }

    fn call_args(proto: &Proto, regs: &[Value], site: &CallSite) -> CallArgs {
        let start = site.start as usize;
        let npos = site.npos as usize;
        let pos = regs[start..start + npos].to_vec();
        let named = site
            .named
            .iter()
            .enumerate()
            .map(|(k, n)| (*n, regs[start + npos + k].clone()))
            .collect();
        let shapes = if site.shapes.iter().all(Option::is_none) {
            Vec::new()
        } else {
            site.shapes
                .iter()
                .map(|s| s.map(|i| proto.shapes[i as usize].clone()))
                .collect()
        };
        CallArgs { pos, named, shapes }
    }

    // ─── Execution ───────────────────────────────────────────────────────────

    fn exec(&mut self, proto: &Proto, upvals: &[Cell], ctx: &Rc<[u32]>, regs: &mut [Value]) -> R {
        self.exec_at(proto, upvals, ctx, regs, 0, 0, None, None)
    }

    /// Runs `proto` from instruction `k0` of block `b0` (resumption of native code).
    #[allow(clippy::too_many_arguments)]
    fn exec_at(
        &mut self,
        proto: &Proto,
        upvals: &[Cell],
        ctx: &Rc<[u32]>,
        regs: &mut [Value],
        b0: usize,
        k0: usize,
        cells: Option<Vec<Cell>>,
        caught: Option<Vec<Option<Error>>>,
    ) -> R {
        // (Resumption of native code: its cells, shared with the lambdas it created.)
        let mut cells: Vec<Cell> = match cells {
            Some(c) => c,
            None if proto.ncells == 0 => Vec::new(),
            None => new_cells(proto.ncells),
        };
        #[cfg(feature = "jit")]
        if b0 == 0
            && k0 == 0
            && let Some(entry) = self.dyn_entry(proto)
        {
            return self.run_native(entry, proto, upvals, ctx, regs, &mut cells);
        }
        let mut b = b0;
        let mut skip = k0;
        // This call's slots of caught errors: `self.caught[base..]` (resumption of native
        // code: those it kept).
        let base = self.caught.len();
        if let Some(c) = caught {
            self.caught.extend(c);
        }
        macro_rules! reg {
            ($r:expr) => {
                regs[$r as usize]
            };
        }
        'run: loop {
            let block = &proto.blocks[b];
            let first = std::mem::take(&mut skip);
            for (ins, &span) in block.instrs.iter().zip(&block.spans).skip(first) {
                if let Err(e) = self.step(proto, upvals, ctx, regs, &mut cells, ins, span) {
                    b = self.catch(proto, block, e, regs, base)?;
                    continue 'run;
                }
            }
            match &block.term {
                Term::Reraise(slot) => {
                    let e = self.caught[base + *slot as usize]
                        .take()
                        .expect("caught error");
                    b = self.catch(proto, block, e, regs, base)?;
                }
                Term::Jump(t) => b = *t as usize,
                Term::Branch { cond, then, else_ } => {
                    b = if reg!(*cond).truthy() { *then } else { *else_ } as usize;
                }
                Term::IfInt {
                    op,
                    a,
                    b: y,
                    then,
                    else_,
                } => {
                    let (Value::Int(x), Value::Int(y)) = (&reg!(*a), &reg!(*y)) else {
                        unreachable!("int register that does not hold an int")
                    };
                    b = if int_cmp(*op, *x, *y) { *then } else { *else_ } as usize;
                }
                Term::IfIntK {
                    op,
                    a,
                    k,
                    then,
                    else_,
                } => {
                    let Value::Int(x) = reg!(*a) else {
                        unreachable!("int register that does not hold an int")
                    };
                    b = if int_cmp(*op, x, *k) { *then } else { *else_ } as usize;
                }
                Term::Return { reg: r, shape } => {
                    if self.caught.len() > base {
                        self.caught.truncate(base);
                    }
                    let v = std::mem::replace(&mut reg!(*r), Value::Nil);
                    return match &proto.ret {
                        Some(ty) => {
                            let shape = shape.map(|s| proto.shapes[s as usize].clone());
                            self.coerce(v, ty, shape.as_deref(), ty.span)
                        }
                        None => Ok(v),
                    };
                }
                Term::IterNext {
                    iter,
                    dst,
                    body,
                    done,
                } => {
                    b = if iter_next(regs, *iter, *dst) {
                        *body
                    } else {
                        *done
                    } as usize;
                }
            }
        }
    }

    /// Error raised in `block`: to its handler (the error as a value in its register),
    /// otherwise returned (the call's slots of caught errors are freed). Syntax errors
    /// and warnings cannot be caught.
    #[cold]
    #[inline(never)]
    fn catch(
        &mut self,
        proto: &Proto,
        block: &Block,
        e: Error,
        regs: &mut [Value],
        base: usize,
    ) -> R<usize> {
        let h = match block.handler {
            Some(h) if matches!(e.kind, ErrorKind::Runtime | ErrorKind::Type) => {
                proto.handlers[h as usize]
            }
            _ => {
                self.caught.truncate(base);
                return Err(e);
            }
        };
        regs[h.reg as usize] = self.error_value(&e);
        let end = base + proto.nslots as usize;
        if self.caught.len() < end {
            self.caught.resize_with(end, || None);
        }
        self.caught[base + h.slot as usize] = Some(e);
        Ok(h.block as usize)
    }

    /// What a `rescue` receives: the raised object, otherwise an `Error`.
    pub(crate) fn error_value(&mut self, e: &Error) -> Value {
        if let Some(Raised(v)) = &e.raised {
            return v.clone();
        }
        let msg = Value::str(&e.message);
        let slot = self.globals.slot(self.syms.error);
        match &self.globals.at(slot).value {
            Some(Value::Type(info)) => {
                let info = info.clone();
                let span = e.span.unwrap_or_default();
                self.construct(&info, CallArgs::plain(vec![msg.clone()]), None, span)
                    .unwrap_or(msg)
            }
            _ => msg,
        }
    }

    /// Error of `raise v`: a string becomes the message of an `Error`; an object is
    /// carried as is (its `message` field, if any, is the text of the error).
    #[cold]
    fn raise(&self, v: Value, span: Span) -> Error {
        let message = match &v {
            Value::Str(s) => return Error::runtime(s.to_string(), span),
            Value::Object(o) => {
                let o = o.borrow();
                let text = o
                    .info
                    .field_index(self.syms.message)
                    .and_then(|i| match &o.fields[i] {
                        Value::Str(s) => Some(s.clone()),
                        _ => None,
                    });
                match text {
                    Some(t) if o.info.name.as_ref() == "Error" => t.to_string(),
                    Some(t) => format!("{}: {t}", o.info.name),
                    None => format!("raised {}", short(&v.inspect(&self.interner))),
                }
            }
            _ => {
                return Error::type_error(
                    format!(
                        "`raise` expects a string or an object, got {}",
                        v.type_label()
                    ),
                    span,
                );
            }
        };
        let mut e = Error::runtime(message, span);
        e.raised = Some(Raised(v));
        e
    }

    /// Runs one instruction (not a terminator). Used by the interpreter loop (inlined)
    /// and by the JIT's native code for the instructions it does not translate itself:
    /// the semantics are the same by construction.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step(
        &mut self,
        proto: &Proto,
        upvals: &[Cell],
        ctx: &Rc<[u32]>,
        regs: &mut [Value],
        cells: &mut [Cell],
        ins: &Instr,
        span: Span,
    ) -> R<()> {
        macro_rules! reg {
            ($r:expr) => {
                regs[$r as usize]
            };
        }
        match ins {
            Instr::Int(d, n) => reg!(*d) = Value::Int(*n),
            Instr::Float(d, f) => reg!(*d) = Value::Float(*f),
            Instr::Bool(d, v) => reg!(*d) = Value::Bool(*v),
            Instr::Nil(d) => reg!(*d) = Value::Nil,
            Instr::Const(d, c) => reg!(*d) = proto.consts[*c as usize].clone(),
            Instr::Move(d, a) => reg!(*d) = reg!(*a).clone(),
            Instr::GetGlobal(d, g) => match &self.globals.at(*g).value {
                Some(v) => reg!(*d) = v.clone(),
                None => {
                    let n = self.name(self.globals.at(*g).name).to_string();
                    return rt_err(format!("`{n}` is not defined"), span);
                }
            },
            Instr::SetGlobal(g, a) => {
                let v = reg!(*a).clone();
                self.set_global(*g, v, span)?;
            }
            Instr::NewCell(c) => {
                cells[*c as usize] = Rc::new(RefCell::new(Value::Uninit));
                crate::gc::track_cell(&cells[*c as usize]);
            }
            Instr::GetCell(d, c) => {
                let v = cells[*c as usize].borrow().clone();
                reg!(*d) = init_check(v, span)?;
            }
            Instr::SetCell(c, a) => *cells[*c as usize].borrow_mut() = reg!(*a).clone(),
            Instr::GetUp(d, u) => {
                let v = upvals[*u as usize].borrow().clone();
                reg!(*d) = init_check(v, span)?;
            }
            Instr::SetUp(u, a) => *upvals[*u as usize].borrow_mut() = reg!(*a).clone(),
            Instr::Guard(d, a, ty) => {
                let v = reg!(*a).clone();
                reg!(*d) = self.guard(v, *ty, span)?;
            }
            Instr::IBin(op, d, a, b) => {
                let (Value::Int(x), Value::Int(y)) = (&reg!(*a), &reg!(*b)) else {
                    unreachable!("int register that does not hold an int")
                };
                reg!(*d) = int_op(*op, *x, *y, span)?;
            }
            Instr::IBinK(op, d, a, k) => {
                let Value::Int(x) = reg!(*a) else {
                    unreachable!("int register that does not hold an int")
                };
                reg!(*d) = int_op(*op, x, *k, span)?;
            }
            Instr::FBin(op, d, a, b) => {
                let (Value::Float(x), Value::Float(y)) = (&reg!(*a), &reg!(*b)) else {
                    unreachable!("float register that does not hold a float")
                };
                reg!(*d) = float_op(*op, *x, *y);
            }
            Instr::INeg(d, a) => {
                let Value::Int(x) = reg!(*a) else {
                    unreachable!()
                };
                reg!(*d) = match x.checked_neg() {
                    Some(n) => Value::Int(n),
                    None => return rt_err("integer overflow", span),
                };
            }
            Instr::IToF(d, a) => {
                let Value::Int(x) = reg!(*a) else {
                    unreachable!()
                };
                reg!(*d) = Value::Float(x as f64);
            }

            Instr::Bin(op, d, a, b) => {
                // Common case in unannotated code: two integers.
                if let (Value::Int(x), Value::Int(y)) = (&reg!(*a), &reg!(*b))
                    && !matches!(op, BinOp::Push | BinOp::And | BinOp::Or)
                {
                    reg!(*d) = int_op(*op, *x, *y, span)?;
                    return Ok(());
                }
                let (l, r) = (reg!(*a).clone(), reg!(*b).clone());
                reg!(*d) = self.binary(*op, l, r, span)?;
            }
            Instr::Not(d, a) => reg!(*d) = Value::Bool(!reg!(*a).truthy()),

            Instr::Index(d, a, i) => {
                let (r, i) = (reg!(*a).clone(), reg!(*i).clone());
                reg!(*d) = self.index(&r, &i, span)?;
            }
            Instr::IndexOpt(d, a, i) => {
                let (r, i) = (reg!(*a).clone(), reg!(*i).clone());
                reg!(*d) = self.index_opt(&r, &i, span)?;
            }
            Instr::Field(d, a, name, access) => {
                let r = reg!(*a).clone();
                reg!(*d) = self.field(&r, *name, *access, span)?;
            }
            Instr::FieldAs(d, a, name, t) => {
                let r = reg!(*a).clone();
                // A public field of a `type`.
                let v = self.field(&r, *name, Access::THIS, span)?;
                reg!(*d) = self.field_as(v, *t, *name, span)?;
            }
            Instr::FieldAt(d, a, idx) => {
                let Value::Object(o) = &reg!(*a) else {
                    return rt_err("`@field`: `self` is not an object", span);
                };
                let v = o.borrow().fields[*idx as usize].clone();
                if let Value::Uninit = v {
                    let n = self
                        .name(o.borrow().info.fields[*idx as usize].name)
                        .to_string();
                    return rt_err(format!("`@{n}` is read before being initialized"), span);
                }
                reg!(*d) = v;
            }
            Instr::Call { dst, f, site } => self.call_instr(proto, regs, *dst, *f, *site, span)?,
            Instr::CallMethod {
                dst,
                recv,
                name,
                site,
                access,
            } => {
                let site = &proto.calls[*site as usize];
                // Declared method already compiled: called directly.
                #[cfg(feature = "jit")]
                if self.jit_enabled
                    && site.named.is_empty()
                    && !site.block
                    && let Value::Object(o) = &reg!(*recv)
                {
                    let m = {
                        let b = o.borrow();
                        let info = &b.info;
                        // A `type`'s field holding a function is called instead.
                        let plain =
                            info.kind == TypeDeclKind::Class || info.field_index(*name).is_none();
                        plain.then(|| info.methods.get(name).cloned()).flatten()
                    };
                    if let Some(m) = m
                        && (m.vis == Visibility::Public || *access == Access::THIS)
                    {
                        let start = site.start as usize;
                        let args = &regs[start..start + site.npos as usize];
                        let this = &regs[*recv as usize];
                        // Method of a generic class: its copy for these fields and arguments.
                        let m = match m.class_params.is_empty() {
                            true => Some(m),
                            false => method_spec(&m, this, args),
                        };
                        if let Some(m) = m
                            && let Some(r) = self.native_fast(&m, Some(this), args, span, &[])
                        {
                            reg!(*dst) = r?;
                            return Ok(());
                        }
                    }
                }
                let r = reg!(*recv).clone();
                // Value with no declared method or field (string, list, number…): built-in
                // method, directly.
                if !matches!(
                    r,
                    Value::Object(_) | Value::Record(_) | Value::Variant(..) | Value::Enum(_)
                ) && site.named.is_empty()
                    && !site.block
                {
                    let start = site.start as usize;
                    let args = regs[start..start + site.npos as usize].to_vec();
                    reg!(*dst) = self.method(r, *name, args, span)?;
                    return Ok(());
                }
                let args = Self::call_args(proto, regs, site);
                reg!(*dst) = self.call_method_named(r, *name, args, *access, span)?;
            }

            // Less frequent instructions: outside the main loop, so that it stays small
            // and well optimized.
            other => self.exec_slow(proto, upvals, ctx, regs, cells, other, span)?,
        }
        Ok(())
    }

    /// `Call` instruction (also called by baseline native code).
    #[inline]
    pub(crate) fn call_instr(
        &mut self,
        proto: &Proto,
        regs: &mut [Value],
        dst: Reg,
        f: Reg,
        site: u32,
        span: Span,
    ) -> R<()> {
        macro_rules! reg {
            ($r:expr) => {
                regs[$r as usize]
            };
        }
        let site = &proto.calls[site as usize];
        // Common case: script function, exact positional arguments. The function and
        // the arguments are read in place from the registers.
        if !site.block
            && site.named.is_empty()
            && let Value::Function(func) = &reg!(f)
            && let Function::Script(c) = &**func
            && c.proto.params.len() == site.npos as usize
            && site.shapes.iter().all(Option::is_none)
        {
            let start = site.start as usize;
            let v = self.invoke_regs(c, &regs[start..start + site.npos as usize], span)?;
            reg!(dst) = v;
            return Ok(());
        }
        // Construction of a `type` where every field is given: direct.
        if !site.block
            && site.shapes.iter().all(Option::is_none)
            && let Value::Type(info) = &reg!(f)
            && info.kind != crate::ast::TypeDeclKind::Class
            && !info.is_abstract
            && info.fields.len() == site.npos as usize + site.named.len()
        {
            let start = site.start as usize;
            let info = info.clone();
            let v = self.construct_regs(
                &info,
                &regs[start..start + info.fields.len()],
                &site.named,
                span,
            )?;
            reg!(dst) = v;
            return Ok(());
        }
        let f = reg!(f).clone();
        let mut args = Self::call_args(proto, regs, site);
        let v = match (&f, site.block) {
            // Construction: the `do` block fills the fields.
            (Value::Type(info), true) => {
                let block = args.pos.pop().unwrap();
                if !args.shapes.is_empty() {
                    args.shapes.remove(site.npos as usize - 1);
                }
                let Value::Function(func) = &block else {
                    unreachable!()
                };
                let Function::Script(c) = &**func else {
                    unreachable!()
                };
                self.construct(info, args, Some(c), span)?
            }
            _ => self.call_value(&f, args, span)?,
        };
        reg!(dst) = v;
        Ok(())
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn exec_slow(
        &mut self,
        proto: &Proto,
        upvals: &[Cell],
        ctx: &Rc<[u32]>,
        regs: &mut [Value],
        cells: &mut [Cell],
        ins: &Instr,
        span: Span,
    ) -> R<()> {
        macro_rules! reg {
            ($r:expr) => {
                regs[$r as usize]
            };
        }
        match ins {
            Instr::Fail(c) => {
                return rt_err(proto.consts[*c as usize].to_display(&self.interner), span);
            }
            Instr::Raise(a) => return Err(self.raise(reg!(*a).clone(), span)),
            Instr::IsType(d, a, t) => {
                let ok = self
                    .matches_rt(&reg!(*a), &proto.types[*t as usize].ty)
                    .map_err(|m| Error::type_error(m, span))?;
                reg!(*d) = Value::Bool(ok);
            }

            Instr::DefGlobal { slot, src, mutable } => {
                let v = reg!(*src).clone();
                self.globals.set_slot(*slot, v, *mutable);
            }
            Instr::GetNamed {
                dst,
                name,
                fallback,
            } => {
                let v = match self.ctx_lookup(ctx, *name) {
                    Some((id, i)) => self.field_ref_get(id, i, span)?,
                    None => match *fallback {
                        Fallback::Reg(r) => reg!(r).clone(),
                        Fallback::Cell(c) => init_check(cells[c as usize].borrow().clone(), span)?,
                        Fallback::Up(u) => init_check(upvals[u as usize].borrow().clone(), span)?,
                        Fallback::Global(g) => match &self.globals.at(g).value {
                            Some(v) => v.clone(),
                            None => {
                                return rt_err(
                                    format!("`{}` is not defined", self.name(*name)),
                                    span,
                                );
                            }
                        },
                        Fallback::None => {
                            return rt_err(format!("`{}` is not defined", self.name(*name)), span);
                        }
                    },
                };
                reg!(*dst) = v;
            }
            Instr::SetNamed {
                name,
                src,
                fallback,
                shape,
            } => {
                let v = reg!(*src).clone();
                match self.ctx_lookup(ctx, *name) {
                    Some((id, i)) => {
                        let shape = shape.map(|s| proto.shapes[s as usize].clone());
                        self.field_ref_set(id, i, v, shape.as_deref(), span)?;
                    }
                    None => match *fallback {
                        Fallback::Reg(r) => reg!(r) = v,
                        Fallback::Cell(c) => *cells[c as usize].borrow_mut() = v,
                        Fallback::Up(u) => *upvals[u as usize].borrow_mut() = v,
                        Fallback::Global(g) => self.set_global(g, v, span)?,
                        Fallback::None => {
                            let n = self.name(*name).to_string();
                            let msg = match ctx.last().and_then(|id| self.construct_type_name(*id))
                            {
                                Some(t) => format!(
                                    "`{n}` is neither a field of `{t}` nor a declared variable \
                                             (use `let {n} = …` for a local variable)"
                                ),
                                None => format!("`{n}` is not declared: use `let {n} = …`"),
                            };
                            return rt_err(msg, span);
                        }
                    },
                }
            }
            Instr::CtxField(d, i) => {
                let id = *ctx.last().expect("default value outside a construction");
                reg!(*d) = self.field_ref_get(id, *i, span)?;
            }
            Instr::Coerce {
                dst,
                src,
                ty,
                shape,
            } => {
                let v = reg!(*src).clone();
                let shape = shape.map(|s| proto.shapes[s as usize].clone());
                reg!(*dst) = self.coerce(v, &proto.types[*ty as usize], shape.as_deref(), span)?;
            }
            Instr::IsUndef(d, a) => reg!(*d) = Value::Bool(matches!(reg!(*a), Value::Uninit)),

            Instr::FNeg(d, a) => {
                let Value::Float(x) = reg!(*a) else {
                    unreachable!()
                };
                reg!(*d) = Value::Float(-x);
            }
            Instr::Neg(d, a) => {
                reg!(*d) = match &reg!(*a) {
                    Value::Int(n) => match n.checked_neg() {
                        Some(n) => Value::Int(n),
                        None => return rt_err("integer overflow", span),
                    },
                    Value::Float(f) => Value::Float(-f),
                    v => {
                        return type_err(
                            format!("`-` expects a number, got {}", v.type_label()),
                            span,
                        );
                    }
                };
            }
            Instr::List(d, s, n) => {
                let s = *s as usize;
                reg!(*d) = Value::list(regs[s..s + *n as usize].to_vec());
            }
            Instr::Tuple(d, s, n) => {
                let s = *s as usize;
                reg!(*d) = Value::Tuple(regs[s..s + *n as usize].into());
            }
            Instr::Hash(d, s, n) => {
                let mut h = HashMap::new();
                for k in 0..*n as usize {
                    let key = self.hash_key(&regs[*s as usize + 2 * k], span)?;
                    h.insert(key, regs[*s as usize + 2 * k + 1].clone());
                }
                reg!(*d) = Value::Hash(Rc::new(RefCell::new(h)));
                crate::gc::track(&reg!(*d));
            }
            Instr::Record(d, s, r) => {
                let names = &proto.records[*r as usize];
                let fields = names
                    .iter()
                    .enumerate()
                    .map(|(k, n)| (*n, regs[*s as usize + k].clone()))
                    .collect();
                reg!(*d) = Value::Record(Rc::new(RefCell::new(Record { fields })));
                crate::gc::track(&reg!(*d));
            }
            Instr::Range {
                dst,
                lo,
                hi,
                inclusive,
            } => {
                reg!(*dst) = match (&reg!(*lo), &reg!(*hi)) {
                    (Value::Int(lo), Value::Int(hi)) => Value::Range(Rc::new(Range {
                        lo: *lo,
                        hi: *hi,
                        inclusive: *inclusive,
                    })),
                    (l, h) => {
                        return type_err(
                            format!(
                                "a range expects two integers, got {} and {}",
                                l.type_label(),
                                h.type_label()
                            ),
                            span,
                        );
                    }
                };
            }
            Instr::Concat(d, s, n) => {
                // Reused buffer: a single allocation, for the final string.
                let mut out = std::mem::take(&mut self.concat_buf);
                out.clear();
                for k in 0..*n as usize {
                    let v = &regs[*s as usize + k];
                    if let Value::Str(x) = v {
                        out.push_str(x);
                    } else if !v.write_scalar(&mut out) {
                        let v = v.clone();
                        match self.display(&v, span) {
                            Ok(t) => out.push_str(&t),
                            Err(e) => {
                                self.concat_buf = out;
                                return Err(e);
                            }
                        }
                    }
                }
                reg!(*d) = Value::Str(Rc::from(out.as_str()));
                self.concat_buf = out;
            }
            Instr::SetIndex(a, i, v) => {
                let (r, i, v) = (reg!(*a).clone(), reg!(*i).clone(), reg!(*v).clone());
                self.set_index(&r, &i, v, span)?;
            }
            Instr::SetField {
                obj,
                name,
                src,
                shape,
                access,
            } => {
                let (o, v) = (reg!(*obj).clone(), reg!(*src).clone());
                let shape = shape.map(|s| proto.shapes[s as usize].clone());
                self.set_field(&o, *name, v, shape.as_deref(), *access, span)?;
            }
            Instr::SetFieldAt {
                obj,
                idx,
                src,
                shape,
            } => {
                let (o, v) = (reg!(*obj).clone(), reg!(*src).clone());
                let shape = shape.map(|s| proto.shapes[s as usize].clone());
                self.set_field_at(&o, *idx, v, shape.as_deref(), span)?;
            }
            Instr::TupleIndex(d, a, idx) => {
                reg!(*d) = match &reg!(*a) {
                    Value::Tuple(t) => match t.get(*idx as usize) {
                        Some(v) => v.clone(),
                        None => {
                            return rt_err(
                                format!("index {idx} out of tuple bounds (length {})", t.len()),
                                span,
                            );
                        }
                    },
                    r => {
                        return type_err(
                            format!("`.{idx}` applies to a tuple, got {}", r.type_label()),
                            span,
                        );
                    }
                };
            }
            Instr::Variant(d, a, name) => {
                let e = reg!(*a).clone();
                reg!(*d) = self.variant(&e, *name, span)?;
            }
            Instr::Unpack { dst, src, n, rest } => {
                let v = reg!(*src).clone();
                let items: Vec<Value> = match &v {
                    Value::Tuple(t) => t.to_vec(),
                    Value::List(l) => l.borrow().clone(),
                    v => {
                        return type_err(
                            format!("expected a tuple or list, got {}", v.type_label()),
                            span,
                        );
                    }
                };
                let n = *n as usize;
                if items.len() < n || (!*rest && items.len() != n) {
                    return rt_err(
                        format!(
                            "expected {}{n} values, got {}",
                            if *rest { "at least " } else { "" },
                            items.len()
                        ),
                        span,
                    );
                }
                let mut it = items.into_iter();
                for k in 0..n {
                    reg!(*dst + k as u32) = it.next().unwrap();
                }
                if *rest {
                    reg!(*dst + n as u32) = Value::list(it.collect());
                }
            }

            Instr::Closure(d, p) => {
                let child = &proto.children[*p as usize];
                let ups: Box<[Cell]> = child
                    .captures
                    .iter()
                    .map(|c| match c {
                        Capture::Cell(c) => cells[*c as usize].clone(),
                        Capture::Up(u) => upvals[*u as usize].clone(),
                    })
                    .collect();
                reg!(*d) = Value::Function(Rc::new(Function::Script(Closure {
                    proto: child.clone(),
                    upvals: ups,
                    ctx: ctx.clone(),
                })));
                crate::gc::track(&reg!(*d));
            }
            Instr::IterNew { dst, src, keys } => {
                let v = reg!(*src).clone();
                reg!(*dst) = match &v {
                    Value::List(l) => list_iter(l),
                    _ => Value::Iter(Rc::new(RefCell::new(self.iter_new(&v, *keys, span)?))),
                };
            }
            Instr::CaseTest(d, p, v) => {
                let hit = match (&reg!(*p), &reg!(*v)) {
                    (Value::Range(r), Value::Int(n)) => r.contains(*n),
                    (p, v) => p.equals(v),
                };
                reg!(*d) = Value::Bool(hit);
            }
            Instr::CallSuper {
                dst,
                this,
                parent,
                name,
                site,
            } => {
                let site = &proto.calls[*site as usize];
                let args = Self::call_args(proto, regs, site);
                let method = match &self.globals.at(*parent).value {
                    Some(Value::Type(info)) => info.methods.get(name).cloned(),
                    _ => None,
                };
                let Some(m) = method else {
                    let p = self.name(self.globals.at(*parent).name).to_string();
                    return rt_err(
                        format!("`super`: `{p}` has no method `{}`", self.name(*name)),
                        span,
                    );
                };
                let this = reg!(*this).clone();
                reg!(*dst) = self.call_method(this, &m, args, span)?;
            }
            Instr::DefType { slot, decl } => {
                let Decl::Type(t) = &proto.decls[*decl as usize] else {
                    unreachable!()
                };
                let parent = match t.parent {
                    Some((p, nfields)) => match &self.globals.at(p).value {
                        Some(Value::Type(info))
                            if info.kind == TypeDeclKind::Class
                                && info.fields.len() == nfields as usize =>
                        {
                            Some(info.clone())
                        }
                        _ => {
                            let n = self.name(self.globals.at(p).name).to_string();
                            return type_err(
                                format!(
                                    "`{n}` is not the class `{}` was compiled with",
                                    self.name(t.name)
                                ),
                                span,
                            );
                        }
                    },
                    None => None,
                };
                let mut traits = Vec::with_capacity(t.traits.len());
                for &s in &t.traits {
                    match &self.globals.at(s).value {
                        Some(Value::Type(info)) if info.kind == TypeDeclKind::Trait => {
                            traits.push(info.clone());
                        }
                        _ => {
                            let n = self.name(self.globals.at(s).name).to_string();
                            return type_err(format!("`{n}` is not a trait"), span);
                        }
                    }
                }
                let info = TypeInfo::new(t, parent, traits, &proto.script, &self.interner)
                    .map_err(|m| Error::type_error(m, span))?;
                self.globals
                    .set_slot(*slot, Value::Type(Rc::new(info)), false);
            }
            Instr::DefEnum { slot, decl, start } => {
                let Decl::Enum(e) = &proto.decls[*decl as usize] else {
                    unreachable!()
                };
                let explicit = regs[*start as usize..].to_vec();
                let info = self.enum_info(e, &explicit, &proto.script, span)?;
                self.globals
                    .set_slot(*slot, Value::Enum(Rc::new(info)), false);
            }
            _ => unreachable!("instruction handled in the main loop"),
        }
        Ok(())
    }

    /// Direct call of a function already compiled to baseline code, skipping what it
    /// does not need: generic binding (unannotated parameters), typed JIT attempt,
    /// cells. `None`: general path (including the stack check, one call level out of
    /// `STACK_CHECK`).
    #[cfg(feature = "jit")]
    #[inline]
    fn invoke_native(&mut self, c: &Closure, args: &[Value], span: Span) -> Option<R> {
        let proto = &c.proto;
        let entry = proto.jit_dyn.get();
        if entry <= 1
            || proto.ncells != 0
            || !proto.plain_params
            || self.depth >= self.max_depth
            || (self.depth + 1).is_multiple_of(STACK_CHECK)
            || (proto.kind == ProtoKind::Function
                && !matches!(*proto.jit.borrow(), JitState::Rejected))
        {
            return None;
        }
        self.depth += 1;
        let mut regs = self.frames.pop().unwrap_or_default();
        fill_nil(&mut regs, proto.regs.len());
        for (p, v) in proto.params.iter().zip(args) {
            regs[p.reg as usize] = v.clone();
        }
        let r = self.run_native(entry, proto, &c.upvals, &c.ctx, &mut regs, &mut []);
        regs.clear();
        self.frames.push(regs);
        self.depth -= 1;
        Some(r.map_err(|e| self.traced(e, proto, span)))
    }

    /// Entry point of `proto`'s baseline native code, compiled on the second call (or the
    /// first if it contains a loop).
    #[cfg(feature = "jit")]
    #[inline]
    fn dyn_entry(&mut self, proto: &Proto) -> Option<usize> {
        match proto.jit_dyn.get() {
            0 => {}
            1 => return None,
            e => return Some(e),
        }
        if !self.jit_enabled {
            return None;
        }
        let n = proto.calls_seen.get() + 1;
        proto.calls_seen.set(n);
        if n < 2 && !proto.has_loop && !self.jit_eager {
            return None;
        }
        if self.jit.is_none() {
            match crate::jit::Jit::new() {
                Some(j) => self.jit = Some(j),
                None => {
                    self.jit_enabled = false;
                    return None;
                }
            }
        }
        let e = self.jit.as_mut()?.compile_dynamic(proto).unwrap_or(1);
        proto.jit_dyn.set(e);
        (e != 1).then_some(e)
    }

    #[cfg(feature = "jit")]
    pub(crate) fn set_pending(&mut self, e: Error) {
        self.pending = Some(e);
    }

    #[cfg(feature = "jit")]
    fn run_native(
        &mut self,
        entry: usize,
        proto: &Proto,
        upvals: &[Cell],
        ctx: &Rc<[u32]>,
        regs: &mut [Value],
        cells: &mut [Cell],
    ) -> R {
        let mut f = crate::ir::NativeFrame {
            regs: regs.as_mut_ptr(),
            nregs: regs.len(),
            cells: cells.as_mut_ptr(),
            ncells: cells.len(),
            upvals: upvals.as_ptr(),
            nupvals: upvals.len(),
            ctx,
            proto,
            ret: 0,
            ret_shape: 0,
        };
        // SAFETY: `entry` comes from `compile_dynamic` for this `proto`; the native code
        // only accesses `regs`, `cells`, `upvals` through the window `f`.
        let code: extern "C" fn(*mut Runtime, *mut crate::ir::NativeFrame) -> i32 =
            unsafe { std::mem::transmute(entry) };
        if code(self, &mut f) != 0 {
            return Err(self.pending.take().expect("native code error"));
        }
        let v = std::mem::replace(&mut regs[f.ret as usize], Value::Nil);
        match &proto.ret {
            Some(ty) => {
                let shape =
                    (f.ret_shape != u32::MAX).then(|| proto.shapes[f.ret_shape as usize].clone());
                self.coerce(v, ty, shape.as_deref(), ty.span)
            }
            None => Ok(v),
        }
    }

    /// Direct call of the typed native code of `proto`, already compiled, when the
    /// arguments are of exactly its native types (`int`, `float`, `bool`; `this` first for
    /// a method): no frame, no conversion. `None`: general path (which also produces the
    /// errors of wrong arguments).
    #[cfg(feature = "jit")]
    #[inline]
    fn native_fast(
        &mut self,
        proto: &Rc<Proto>,
        this: Option<&Value>,
        args: &[Value],
        span: Span,
        upvals: &[Cell],
    ) -> Option<R> {
        if !self.jit_enabled {
            return None;
        }
        // Parameters not annotated: the copy for these arguments.
        let spec;
        let proto = if proto.untyped() {
            spec = untyped_spec(proto, |k| args.get(k))?;
            &spec
        } else {
            proto
        };
        // Not borrowed during the call: a resumption may mark it interpreted.
        let (entry, n, ret) = {
            let state = proto.jit.borrow();
            let JitState::Native(code) = &*state else {
                return None;
            };
            let skip = this.is_some() as usize;
            if code.params.len() != skip + args.len() || self.depth + 1 >= self.max_depth {
                return None;
            }
            let exact = code.params[skip..].iter().zip(args).all(|(t, v)| {
                matches!(
                    (t, v),
                    (RegTy::Int, Value::Int(_))
                        | (RegTy::Float, Value::Float(_))
                        | (RegTy::Bool, Value::Bool(_))
                )
            });
            if !exact {
                return None;
            }
            (code.entry, code.params.len(), code.ret)
        };
        let jit: *mut crate::jit::Jit = self.jit.as_mut()?;
        let (depth, max) = (self.depth + 1, self.max_depth);
        // SAFETY: `jit` belongs to this runtime; neither is borrowed during the call.
        let r = unsafe {
            crate::jit::Jit::call(
                jit,
                self,
                entry,
                n,
                ret,
                this.into_iter().chain(args),
                upvals,
                depth,
                max,
            )
        };
        Some(r.map_err(|e| self.traced(e, proto, span)))
    }

    /// Runs `proto` natively if it is (or becomes) compiled; `None` otherwise.
    #[cfg(feature = "jit")]
    fn try_jit(&mut self, proto: &Rc<Proto>, regs: &[Value], upvals: &[Cell]) -> Option<R> {
        if !self.jit_enabled {
            return None;
        }
        let method = match proto.kind {
            ProtoKind::Function => false,
            ProtoKind::Method if matches!(regs[0], Value::Object(_)) => true,
            _ => return None,
        };
        let declared = proto;
        // `forall` function, method of a generic class: its copy typed for these
        // arguments and fields (`T = int`).
        let spec;
        let proto = if proto.is_generic() {
            spec = specialization(proto, regs)?;
            &spec
        } else if proto.untyped() {
            spec = untyped_spec(proto, |k| {
                proto.params.get(k).map(|p| &regs[p.reg as usize])
            })?;
            if !fits_params(&spec, regs) {
                return None;
            }
            &spec
        } else if method && !proto.class_params.is_empty() {
            let args: Vec<Value> = proto
                .params
                .iter()
                .map(|p| regs[p.reg as usize].clone())
                .collect();
            spec = method_spec(proto, &regs[0], &args)?;
            if !fits_params(&spec, regs) {
                return None;
            }
            &spec
        } else {
            proto
        };
        if matches!(*proto.jit.borrow(), JitState::Untried) {
            // Method: compiled for the class that declares it (`self` in register 0).
            let this = match &regs[0] {
                Value::Object(o) if method => {
                    let info = o.borrow().info.clone();
                    let name = proto.decl.as_ref()?.name;
                    Some(crate::objects::method_owner(&info, name, declared))
                }
                _ => None,
            };
            if self.jit.is_none() {
                self.jit = crate::jit::Jit::new();
                if self.jit.is_none() {
                    // Machine not supported by Cranelift.
                    self.jit_enabled = false;
                    return None;
                }
            }
            let jit = self.jit.as_mut()?;
            jit.len_sym = self.syms.len;
            jit.compile(proto, this, &mut self.globals, upvals);
        }
        let (entry, n, ret) = match &*proto.jit.borrow() {
            JitState::Native(code) => (code.entry, code.params.len(), code.ret),
            _ => return None,
        };
        let args: Vec<Value> = method
            .then(|| regs[0].clone())
            .into_iter()
            .chain(proto.params.iter().map(|p| regs[p.reg as usize].clone()))
            .collect();
        let jit: *mut crate::jit::Jit = self.jit.as_mut()?;
        let (depth, max) = (self.depth, self.max_depth);
        // SAFETY: `jit` belongs to this runtime; neither is borrowed during the call.
        Some(unsafe { crate::jit::Jit::call(jit, self, entry, n, ret, &args, upvals, depth, max) })
    }

    /// Resumption in the interpreter of native code (see `jit::DeoptSite`): `proto` runs
    /// from instruction `instr` of block `block`, with the registers rebuilt, at call
    /// depth `depth`.
    #[cfg(feature = "jit")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resume(
        &mut self,
        proto: &Rc<Proto>,
        regs: &mut [Value],
        block: u32,
        instr: u32,
        depth: usize,
        upvals: &[Cell],
        cells: Vec<Cell>,
        caught: Vec<Option<Error>>,
    ) -> R {
        let saved = self.depth;
        self.depth = depth;
        let ctx = self.no_ctx.clone();
        let cells = (proto.ncells > 0).then_some(cells);
        let caught = (proto.nslots > 0).then_some(caught);
        let r = self.exec_at(
            proto,
            upvals,
            &ctx,
            regs,
            block as usize,
            instr as usize,
            cells,
            caught,
        );
        self.depth = saved;
        r.map_err(|e| attach_src(e, &proto.script))
    }

    // ─── Runtime helpers ─────────────────────────────────────────────────────

    pub(crate) fn set_global(&mut self, g: u32, v: Value, span: Span) -> R<()> {
        let slot = self.globals.at(g);
        let n = slot.name;
        match (&slot.value, slot.mutable) {
            (None, _) => {
                let n = self.name(n).to_string();
                rt_err(format!("`{n}` is not declared: use `let {n} = …`"), span)
            }
            (Some(_), false) => rt_err(
                format!(
                    "`{}` cannot be modified (constant, function or type)",
                    self.name(n)
                ),
                span,
            ),
            (Some(_), true) => {
                self.globals.assign(g, v);
                Ok(())
            }
        }
    }

    /// Dynamic value stored in a typed register.
    /// Value of a field whose type the checker knows. Can only fail if the type was
    /// redeclared since the check (REPL).
    fn field_as(&self, v: Value, ty: RegTy, name: Symbol, span: Span) -> R {
        match (ty, &v) {
            (RegTy::Int, Value::Int(_))
            | (RegTy::Float, Value::Float(_))
            | (RegTy::Bool, Value::Bool(_))
            | (RegTy::Value, _) => Ok(v),
            (RegTy::Float, Value::Int(i)) => Ok(Value::Float(*i as f64)),
            (t, v) => type_err(
                format!(
                    "field `{}` should be of type `{}`, got {}: was its type redeclared?",
                    self.name(name),
                    t.name(),
                    v.type_label()
                ),
                span,
            ),
        }
    }

    fn guard(&self, v: Value, ty: RegTy, span: Span) -> R {
        match (ty, &v) {
            (RegTy::Int, Value::Int(_))
            | (RegTy::Float, Value::Float(_))
            | (RegTy::Bool, Value::Bool(_)) => Ok(v),
            (RegTy::Float, Value::Int(i)) => Ok(Value::Float(*i as f64)),
            (RegTy::Value, _) => Ok(v),
            (t, v) => type_err(
                format!(
                    "expected type `{}`, got {} ({}): a variable keeps the type of its first value",
                    match t {
                        RegTy::Int => "int",
                        RegTy::Float => "float",
                        _ => "bool",
                    },
                    v.type_label(),
                    short(&v.inspect(&self.interner))
                ),
                span,
            ),
        }
    }

    pub(crate) fn set_index(&mut self, r: &Value, i: &Value, v: Value, span: Span) -> R<()> {
        match r {
            Value::List(l) => {
                before_list_write(l);
                let mut l = l.borrow_mut();
                let pos = list_index(i, l.len(), span)?;
                l[pos] = v;
            }
            Value::Hash(h) => {
                let key = self.hash_key(i, span)?;
                h.borrow_mut().insert(key, v);
            }
            _ => {
                return type_err(
                    format!("cannot assign by index on {}", r.type_label()),
                    span,
                );
            }
        }
        Ok(())
    }

    pub(crate) fn set_field(
        &mut self,
        obj: &Value,
        name: Symbol,
        v: Value,
        shape: Option<&Shape>,
        access: Access,
        span: Span,
    ) -> R<()> {
        match obj {
            Value::Record(rec) => match rec.borrow_mut().get_mut(name) {
                Some(slot) => {
                    *slot = v;
                    Ok(())
                }
                None => rt_err(
                    format!("the record has no field `{}`", self.name(name)),
                    span,
                ),
            },
            _ => self.object_set_field(obj, name, v, shape, access, span),
        }
    }

    fn set_field_at(
        &mut self,
        obj: &Value,
        idx: u32,
        v: Value,
        shape: Option<&Shape>,
        span: Span,
    ) -> R<()> {
        let Value::Object(o) = obj else {
            return rt_err("`@field`: `self` is not an object", span);
        };
        let info = o.borrow().info.clone();
        let v = match &info.fields[idx as usize].ty {
            Some(ty) => self.coerce(v, ty, shape, span)?,
            None => v,
        };
        o.borrow_mut().fields[idx as usize] = v;
        Ok(())
    }

    fn variant(&self, e: &Value, name: Symbol, span: Span) -> R {
        match e {
            Value::Enum(info) => match info.variants.iter().position(|(n, _)| *n == name) {
                Some(i) => Ok(Value::Variant(info.clone(), i as u32)),
                None => rt_err(
                    format!("`{}` has no variant `{}`", info.name, self.name(name)),
                    span,
                ),
            },
            v => type_err(
                format!("`::` applies to an enum, got {}", v.type_label()),
                span,
            ),
        }
    }

    fn iter_new(&self, v: &Value, keys: bool, span: Span) -> R<IterState> {
        Ok(match v {
            Value::List(l) => IterState::Values(l.borrow().clone().into_iter()),
            Value::Tuple(t) => IterState::Values(t.iter().cloned().collect::<Vec<_>>().into_iter()),
            Value::Range(r) => IterState::Range(r.values()),
            Value::Str(s) => IterState::Values(
                s.chars()
                    .map(|c| Value::Str(c.to_string().into()))
                    .collect::<Vec<_>>()
                    .into_iter(),
            ),
            Value::Hash(h) => {
                let items: Vec<Value> = h
                    .borrow()
                    .iter()
                    .map(|(k, v)| {
                        if keys {
                            k.to_value()
                        } else {
                            Value::Tuple(vec![k.to_value(), v.clone()].into())
                        }
                    })
                    .collect();
                IterState::Values(items.into_iter())
            }
            v => {
                return type_err(
                    format!(
                        "`for` cannot iterate over a value of type {}",
                        v.type_label()
                    ),
                    span,
                );
            }
        })
    }

    /// `recv.name(args)`: function field, declared method, enum method or built-in.
    pub(crate) fn call_method_named(
        &mut self,
        r: Value,
        name: Symbol,
        args: CallArgs,
        access: Access,
        span: Span,
    ) -> R {
        let field_fn = match &r {
            Value::Record(rec) => rec.borrow().get(name).cloned(),
            Value::Object(o) => {
                let o = o.borrow();
                match (o.info.kind, o.info.field_index(name)) {
                    (TypeDeclKind::Class, _) | (_, None) => None,
                    (_, Some(i)) => Some(o.fields[i].clone()),
                }
            }
            _ => None,
        };
        if let Some(f) = field_fn {
            return self.call_value(&f, args, span);
        }
        if let Some(m) = self.user_method(&r, name) {
            if m.vis != Visibility::Public
                && let Value::Object(o) = &r
            {
                let info = o.borrow().info.clone();
                self.check_access(&info, name, &m, access, span)?;
            }
            return self.call_method(r, &m, args, span);
        }
        if let Value::Enum(e) = &r {
            return self.enum_method(e, name, &args.pos, span);
        }
        if let Some(&(n, _)) = args.named.first() {
            return rt_err(
                format!(
                    "built-in methods do not accept named arguments (`{}`)",
                    self.name(n)
                ),
                span,
            );
        }
        self.method(r, name, args.pos, span)
    }

    // ─── Type checks ─────────────────────────────────────────────────────────

    /// Checks `v` against an annotation, after any conversion:
    /// - `int` → `float`;
    /// - record **literal** (`shape`) → nominal `type`, including inside a list literal
    ///   and for `T?` / `A | B`.
    #[inline]
    pub(crate) fn coerce(
        &mut self,
        v: Value,
        ty: &TypeRef,
        shape: Option<&Shape>,
        span: Span,
    ) -> R {
        // Common case: simple type already satisfied.
        match (&*ty.ty, &v) {
            (RtType::Any, _)
            | (RtType::Int, Value::Int(_))
            | (RtType::Float, Value::Float(_))
            | (RtType::Str, Value::Str(_))
            | (RtType::Bool, Value::Bool(_)) => Ok(v),
            _ => self.coerce_rt(v, &ty.ty, ty, shape, span),
        }
    }

    fn coerce_rt(
        &mut self,
        v: Value,
        rt: &RtType,
        ty: &TypeRef,
        shape: Option<&Shape>,
        span: Span,
    ) -> R {
        match (shape, rt, &v) {
            (Some(Shape::Record(fs)), RtType::Named(..), Value::Record(r)) => {
                if let Some(info) = self.data_type(rt) {
                    let r = r.borrow().clone();
                    return self.record_to_object(&info, &r, fs, span);
                }
            }
            (Some(Shape::Record(_)), RtType::Optional(inner), Value::Record(_)) => {
                return self.coerce_rt(v, inner, ty, shape, span);
            }
            (Some(Shape::Record(fs)), RtType::Union(items), Value::Record(r)) => {
                for it in items {
                    if let Some(info) = self.data_type(it) {
                        let r = r.borrow().clone();
                        if let Ok(obj) = self.record_to_object(&info, &r, fs, span) {
                            return Ok(obj);
                        }
                    }
                }
            }
            (Some(Shape::List(shapes)), RtType::List(inner), Value::List(l)) => {
                let elems = l.borrow().clone();
                let mut out = Vec::with_capacity(elems.len());
                for (k, x) in elems.into_iter().enumerate() {
                    let s = shapes.get(k).cloned().flatten();
                    out.push(self.coerce_rt(x, inner, ty, s.as_deref(), span)?);
                }
                return Ok(Value::list(out));
            }
            _ => {}
        }
        if let (RtType::Float, Value::Int(i)) = (rt, &v) {
            return Ok(Value::Float(*i as f64));
        }
        match self.matches_rt(&v, rt) {
            Ok(true) => Ok(v),
            Ok(false) => type_err(
                format!(
                    "expected type `{}`, got {} ({})",
                    ty.shown,
                    v.type_label(),
                    short(&v.inspect(&self.interner))
                ),
                span,
            ),
            Err(msg) => type_err(msg, ty.span),
        }
    }

    /// `type` (not `class`) named by a named type.
    fn data_type(&self, rt: &RtType) -> Option<Rc<TypeInfo>> {
        match rt {
            RtType::Named(slot, ..) if *slot != u32::MAX => match &self.globals.at(*slot).value {
                Some(Value::Type(info)) if info.kind != TypeDeclKind::Class => Some(info.clone()),
                _ => None,
            },
            _ => None,
        }
    }

    fn record_to_object(
        &mut self,
        info: &Rc<TypeInfo>,
        rec: &Record,
        fields: &[(Symbol, Option<Rc<Shape>>)],
        span: Span,
    ) -> R {
        let shapes = rec
            .fields
            .iter()
            .map(|(n, _)| {
                fields
                    .iter()
                    .find(|(m, _)| m == n)
                    .and_then(|(_, s)| s.clone())
            })
            .collect();
        let args = CallArgs {
            pos: Vec::new(),
            named: rec.fields.clone(),
            shapes,
        };
        self.construct(info, args, None, span)
    }

    pub(crate) fn matches_rt(&self, v: &Value, rt: &RtType) -> Result<bool, String> {
        Ok(match rt {
            RtType::Any => true,
            RtType::Int => matches!(v, Value::Int(_)),
            RtType::Float => matches!(v, Value::Float(_)),
            RtType::Str => matches!(v, Value::Str(_)),
            RtType::Bool => matches!(v, Value::Bool(_)),
            RtType::Nil => matches!(v, Value::Nil),
            RtType::Fn => matches!(v, Value::Function(_) | Value::Type(_)),
            RtType::LitStr(s) => matches!(v, Value::Str(x) if **x == **s),
            RtType::LitInt(n) => matches!(v, Value::Int(x) if x == n),
            RtType::Named(slot, name, _) => {
                let found = if *slot == u32::MAX {
                    None
                } else {
                    self.globals.at(*slot).value.as_ref()
                };
                match found {
                    Some(Value::Type(info)) => {
                        matches!(v, Value::Object(o) if o.borrow().info.is_a(info))
                    }
                    Some(Value::Enum(e)) => matches!(v, Value::Variant(x, _) if Rc::ptr_eq(x, e)),
                    _ => return Err(format!("unknown type `{}`", self.name(*name))),
                }
            }
            RtType::List(inner) => match v {
                Value::List(l) => {
                    for item in l.borrow().iter() {
                        if !self.matches_rt(item, inner)? {
                            return Ok(false);
                        }
                    }
                    true
                }
                _ => false,
            },
            RtType::Hash(kt, vt) => match v {
                Value::Hash(h) => {
                    for (k, val) in h.borrow().iter() {
                        if !self.matches_rt(&k.to_value(), kt)? || !self.matches_rt(val, vt)? {
                            return Ok(false);
                        }
                    }
                    true
                }
                _ => false,
            },
            RtType::Record(fields) => match v {
                Value::Record(r) => {
                    let r = r.borrow();
                    if r.fields.len() != fields.len() {
                        return Ok(false);
                    }
                    for (name, fty) in fields {
                        match r.get(*name) {
                            Some(fv) if self.matches_rt(fv, fty)? => {}
                            _ => return Ok(false),
                        }
                    }
                    true
                }
                _ => false,
            },
            RtType::Tuple(items) => match v {
                Value::Tuple(t) => {
                    if t.len() != items.len() {
                        return Ok(false);
                    }
                    for (val, ity) in t.iter().zip(items) {
                        if !self.matches_rt(val, ity)? {
                            return Ok(false);
                        }
                    }
                    true
                }
                _ => false,
            },
            RtType::Optional(inner) => matches!(v, Value::Nil) || self.matches_rt(v, inner)?,
            RtType::Union(items) => {
                for it in items {
                    if self.matches_rt(v, it)? {
                        return Ok(true);
                    }
                }
                false
            }
        })
    }

    fn enum_info(
        &self,
        e: &EnumDeclIr,
        explicit: &[Value],
        script: &Rc<Script>,
        span: Span,
    ) -> R<EnumInfo> {
        let mut values = explicit.iter();
        let mut variants = Vec::with_capacity(e.variants.len());
        for (i, (name, has)) in e.variants.iter().zip(&e.explicit).enumerate() {
            // Without `= value`: the name for a `: str` enum, otherwise the index.
            let value = if *has {
                values.next().cloned().unwrap_or(Value::Nil)
            } else if e.repr_is_str {
                Value::str(self.name(*name))
            } else {
                Value::Int(i as i64)
            };
            if let Some(repr) = &e.repr
                && !self.matches_rt(&value, &repr.ty).unwrap_or(false)
            {
                return type_err(
                    format!(
                        "the value of `{}` must be of type `{}`, got {}",
                        self.name(*name),
                        repr.shown,
                        value.type_label()
                    ),
                    span,
                );
            }
            variants.push((*name, value));
        }
        Ok(EnumInfo {
            name: self.name(e.name).into(),
            variants,
            methods: e.methods.iter().cloned().collect(),
            script: script.clone(),
            annotations: e.annotations.clone(),
            variant_annotations: e.variant_annotations.clone(),
        })
    }
}

fn init_check(v: Value, span: Span) -> R {
    match v {
        Value::Uninit => rt_err("variable used before its declaration", span),
        v => Ok(v),
    }
}

fn int_op(op: BinOp, x: i64, y: i64, span: Span) -> R {
    use BinOp::*;
    let overflow = || Error::runtime("integer overflow", span);
    Ok(match op {
        Add => Value::Int(x.checked_add(y).ok_or_else(overflow)?),
        Sub => Value::Int(x.checked_sub(y).ok_or_else(overflow)?),
        Mul => Value::Int(x.checked_mul(y).ok_or_else(overflow)?),
        Div | Rem if y == 0 => return rt_err("division by zero", span),
        Div => Value::Int(x.checked_div_euclid(y).ok_or_else(overflow)?),
        Rem => Value::Int(x.checked_rem_euclid(y).ok_or_else(overflow)?),
        Eq => Value::Bool(x == y),
        Ne => Value::Bool(x != y),
        Lt => Value::Bool(x < y),
        Le => Value::Bool(x <= y),
        Gt => Value::Bool(x > y),
        Ge => Value::Bool(x >= y),
        And | Or | Push => unreachable!("non-integer operation"),
    })
}

#[inline]
fn int_cmp(op: BinOp, x: i64, y: i64) -> bool {
    match op {
        BinOp::Eq => x == y,
        BinOp::Ne => x != y,
        BinOp::Lt => x < y,
        BinOp::Le => x <= y,
        BinOp::Gt => x > y,
        BinOp::Ge => x >= y,
        _ => unreachable!("expected a comparison"),
    }
}

fn float_op(op: BinOp, x: f64, y: f64) -> Value {
    use BinOp::*;
    match op {
        Add => Value::Float(x + y),
        Sub => Value::Float(x - y),
        Mul => Value::Float(x * y),
        Div => Value::Float(x / y),
        Rem => Value::Float(x.rem_euclid(y)),
        Eq => Value::Bool(x == y),
        Ne => Value::Bool(x != y),
        Lt => Value::Bool(x < y),
        Le => Value::Bool(x <= y),
        Gt => Value::Bool(x > y),
        Ge => Value::Bool(x >= y),
        And | Or | Push => unreachable!("non-float operation"),
    }
}

impl TypeInfo {
    /// Error: a method required by a trait or an abstract parent is not defined.
    pub(crate) fn new(
        t: &TypeDeclIr,
        parent: Option<Rc<TypeInfo>>,
        traits: Vec<Rc<TypeInfo>>,
        script: &Rc<Script>,
        i: &Interner,
    ) -> Result<Self, String> {
        let name: Rc<str> = i.resolve(t.name).into();
        let own = t.fields.iter().map(|f| crate::objects::FieldInfo {
            name: f.name,
            ty: f.ty.clone(),
            default: f.default.clone(),
            annotations: f.annotations.clone(),
        });
        let (mut fields, mut methods) = match &parent {
            Some(p) => (p.fields.clone(), p.methods.clone()),
            None => (Vec::new(), FxHashMap::default()),
        };
        fields.extend(own);
        // Precedence: own methods, then the traits' (in order), then the parent's.
        let mut required: Vec<crate::objects::Required> =
            parent.as_ref().map_or(Vec::new(), |p| p.required.clone());
        let mut all_traits: Vec<Rc<TypeInfo>> =
            parent.as_ref().map_or(Vec::new(), |p| p.traits.clone());
        for tr in &traits {
            methods.extend(tr.methods.iter().map(|(n, m)| (*n, m.clone())));
            required.extend(tr.required.iter().cloned());
            if !all_traits.iter().any(|x| Rc::ptr_eq(x, tr)) {
                all_traits.push(tr.clone());
            }
        }
        methods.extend(t.methods.iter().cloned());
        required.extend(t.abstracts.iter().map(|d| crate::objects::Required {
            name: d.name,
            by: name.clone(),
            decl: d.clone(),
        }));
        required.retain(|r| !methods.contains_key(&r.name));
        let mut seen = Vec::new();
        required.retain(|r| {
            let new = !seen.contains(&r.name);
            seen.push(r.name);
            new
        });
        let is_abstract = t.is_abstract || t.kind == crate::ast::TypeDeclKind::Trait;
        if !is_abstract && let Some(r) = required.first() {
            return Err(format!(
                "`{name}` must define `{}` (required by `{}`)",
                i.resolve(r.name),
                r.by
            ));
        }
        // (A simple type: what it holds refers to no container.)
        fn simple(t: &RtType) -> bool {
            match t {
                RtType::Int
                | RtType::Float
                | RtType::Str
                | RtType::Bool
                | RtType::Nil
                | RtType::LitStr(_)
                | RtType::LitInt(_) => true,
                RtType::Optional(x) => simple(x),
                RtType::Union(xs) | RtType::Tuple(xs) => xs.iter().all(simple),
                _ => false,
            }
        }
        let acyclic = fields
            .iter()
            .all(|f| f.ty.as_ref().is_some_and(|t| simple(&t.ty)));
        Ok(TypeInfo {
            acyclic,
            name,
            kind: t.kind,
            parent,
            traits: all_traits,
            includes: traits,
            generics: t.generics.clone(),
            required,
            is_abstract,
            fields,
            methods,
            annotations: t.annotations.clone(),
            script: script.clone(),
        })
    }
}

/// Copy of `forall` function `proto` typed for its arguments in `regs`: `T` found from a
/// simple value (`int`, `float`, `bool`) or a list of them, every argument then of
/// exactly the type of its parameter (the generic function's arguments are not checked:
/// its annotations are `any`). `None`: no typed copy for these arguments.
#[cfg(feature = "jit")]
fn specialization(proto: &Rc<Proto>, regs: &[Value]) -> Option<Rc<Proto>> {
    let arg = |k: usize| {
        let v = &regs[proto.params.get(k)?.reg as usize];
        match v {
            Value::List(l) => {
                let first = simple_ty(l.borrow().first()?)?;
                Some(RtType::List(Box::new(first)))
            }
            v => simple_ty(v),
        }
    };
    let tys = proto.type_args(&arg)?;
    let spec = proto.specialize(&tys);
    fits_params(&spec, regs).then_some(spec)
}

/// Copy of `proto` (parameters not annotated) typed by its arguments (`arg(k)`): simple
/// values, strings, lists of simple values. `None` if one is of another type.
#[cfg(feature = "jit")]
fn untyped_spec<'a>(
    proto: &Rc<Proto>,
    arg: impl Fn(usize) -> Option<&'a Value>,
) -> Option<Rc<Proto>> {
    // A copy already made (no allocation on every call).
    let kind_fits = |v: &Value, t: &RtType| match (t, v) {
        (RtType::Int, Value::Int(_))
        | (RtType::Float, Value::Float(_))
        | (RtType::Bool, Value::Bool(_))
        | (RtType::Str, Value::Str(_)) => true,
        (RtType::List(e), Value::List(l)) => l
            .borrow()
            .first()
            .is_some_and(|x| simple_ty(x).as_ref() == Some(&**e)),
        _ => false,
    };
    for (tys, p) in proto.specs.borrow().iter() {
        let same = proto
            .params
            .iter()
            .zip(tys)
            .enumerate()
            .all(|(k, (prm, t))| prm.ty.is_some() || arg(k).is_some_and(|v| kind_fits(v, t)));
        if same {
            return Some(p.clone());
        }
    }
    let tys = proto
        .params
        .iter()
        .enumerate()
        .map(|(k, p)| match &p.ty {
            Some(t) => Some((*t.ty).clone()),
            None => match arg(k)? {
                Value::Str(_) => Some(RtType::Str),
                Value::List(l) => Some(RtType::List(Box::new(simple_ty(l.borrow().first()?)?))),
                v => simple_ty(v),
            },
        })
        .collect::<Option<Vec<_>>>()?;
    Some(proto.specialize_params(&tys))
}

/// The arguments in `regs` are exactly of the types of the parameters of typed copy
/// `spec` (`int` for an `int`, a list of `int` for an `[int]`…).
#[cfg(feature = "jit")]
fn fits_params(spec: &Proto, regs: &[Value]) -> bool {
    fn fits(v: &Value, t: &RtType) -> bool {
        match (t, v) {
            (RtType::Int, Value::Int(_))
            | (RtType::Float, Value::Float(_))
            | (RtType::Bool, Value::Bool(_)) => true,
            (RtType::List(e), Value::List(l)) => l.borrow().iter().all(|x| fits(x, e)),
            (RtType::Str, Value::Str(_)) | (RtType::Any, _) => true,
            _ => false,
        }
    }
    spec.params.iter().all(|p| match &p.ty {
        Some(t) => fits(&regs[p.reg as usize], &t.ty),
        None => true,
    })
}

/// Typed copy of method `m` of a generic class for `this` and its arguments: each type
/// parameter found from the first field written `T` (the class's fields come first), or
/// from a parameter. `None` if one is not a simple type.
#[cfg(feature = "jit")]
fn method_spec(m: &Rc<Proto>, this: &Value, args: &[Value]) -> Option<Rc<Proto>> {
    let Value::Object(o) = this else {
        return None;
    };
    let o = o.borrow();
    let deduce = |n: Symbol| {
        let field = o.info.fields.iter().zip(&o.fields).find(|(f, _)| {
            f.ty.as_ref().is_some_and(
                |t| matches!(&t.expr.kind, crate::ast::TypeKind::Named(p) if p[..] == [n]),
            )
        });
        match field {
            Some((_, v)) => simple_ty(v),
            None => m.type_arg(n, &|k| simple_ty(args.get(k)?)),
        }
    };
    // A copy already made (no allocation on every call).
    for (tys, p) in m.specs.borrow().iter() {
        if m.class_params
            .iter()
            .zip(tys)
            .all(|(n, t)| deduce(*n).as_ref() == Some(t))
        {
            return Some(p.clone());
        }
    }
    let tys = m
        .class_params
        .iter()
        .map(|n| deduce(*n))
        .collect::<Option<Vec<_>>>()?;
    Some(m.specialize_as(&m.class_params, &tys))
}

/// `int` / `float` / `bool` of a value.
#[cfg(feature = "jit")]
fn simple_ty(v: &Value) -> Option<RtType> {
    match v {
        Value::Int(_) => Some(RtType::Int),
        Value::Float(_) => Some(RtType::Float),
        Value::Bool(_) => Some(RtType::Bool),
        _ => None,
    }
}

/// Advances the iterator in register `iter`: `true` and the element in `dst`, or `false`.
#[inline]
pub(crate) fn iter_next(regs: &mut [Value], iter: Reg, dst: Reg) -> bool {
    let Value::Iter(it) = &regs[iter as usize] else {
        unreachable!()
    };
    match iter_advance(it) {
        Some(v) => {
            regs[dst as usize] = v;
            true
        }
        None => false,
    }
}

/// `n` cells not initialized yet (variables captured by lambdas), registered for the
/// cycle collector. (Out of line: the interpreter's loop stays small.)
#[inline(never)]
fn new_cells(n: u32) -> Vec<Cell> {
    (0..n)
        .map(|_| {
            let c = Rc::new(RefCell::new(Value::Uninit));
            crate::gc::track_cell(&c);
            c
        })
        .collect()
}

/// Next element of a `for` iterator.
#[inline]
pub(crate) fn iter_advance(it: &RefCell<IterState>) -> Option<Value> {
    match &mut *it.borrow_mut() {
        IterState::Values(v) => v.next(),
        IterState::Range(r) => r.next().map(Value::Int),
        IterState::List(l, i) => {
            let v = l.borrow().get(*i).cloned();
            *i += 1;
            v
        }
    }
}

/// Grows the empty vector `regs` to `n` `nil` registers. `Value::Nil` is all zero bytes
/// (`repr(C, u8)`, tag 0): zeroing is enough, no copy loop.
#[inline]
fn fill_nil(regs: &mut Vec<Value>, n: usize) {
    debug_assert!(regs.is_empty());
    regs.reserve(n);
    // SAFETY: the capacity is at least `n`; zero bytes form `Value::Nil` (checked by
    // `value::tests::native_layout`), a valid value that owns nothing.
    unsafe {
        std::ptr::write_bytes(regs.as_mut_ptr(), 0, n);
        regs.set_len(n);
    }
}
