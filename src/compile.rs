//! Compilation from the AST to the IR ([`crate::ir`]).
//!
//! Two passes:
//! 1. Analysis ([`Resolver`]): which variables are captured by a closure (they go into
//!    cells, the others into registers), and which expressions contain an assignment
//!    (to preserve the evaluation order of `x + (x = 2)`).
//! 2. Translation ([`Compiler`]): one IR function per function, lambda, method and field
//!    default value; the main program is the `<main>` function.
//!
//! Typed registers: a variable is `int` / `float` / `bool` if its annotation says so or
//! if the checker inferred it; temporaries follow the rules of the operations
//! (`int + int`: `int`). A dynamic value only enters them through a guard.

use crate::ast::*;
use crate::diagnostic::{Error, Errors};
use crate::globals::Globals;
use crate::intern::{FxBuild, FxHashMap, Interner, Symbol};
use crate::ir::{
    AnnotationIr, Annotations, Block as IrBlock, BlockId, CallSite, Capture, Decl, EnumDeclIr,
    Fallback, FieldIr, Handler, Instr, ParamProto, Proto, ProtoKind, Reg, RegTy, RtType, Script,
    Shape, Term, TypeDeclIr, TypeRef,
};
use crate::span::Span;
use crate::value::Value;
use std::collections::HashSet;
use std::rc::Rc;

type FxHashSet<T> = HashSet<T, FxBuild>;

/// Reserved name of `self` in scopes.
const SELF: Symbol = Symbol(u32::MAX);

pub use crate::check::Prims;

pub fn compile(
    program: &Program,
    script: &Rc<Script>,
    interner: &mut Interner,
    globals: &mut Globals,
    prims: Option<&Prims>,
) -> Result<Rc<Proto>, Errors> {
    let mut r = Resolver {
        ast: &program.ast,
        scopes: Vec::new(),
        depth: 0,
        captured: FxHashSet::default(),
        assign: vec![false; program.ast.len()],
    };
    r.program(program);
    let empty = Prims::default();
    let mut c = Compiler {
        ast: &program.ast,
        script,
        i: interner,
        globals,
        prims: prims.unwrap_or(&empty),
        captured: r.captured,
        assign: r.assign,
        fns: Vec::new(),
        errors: Vec::new(),
        classes: FxHashMap::default(),
        traits: FxHashSet::default(),
    };
    let main = c.main(&program.body);
    if c.errors.is_empty() {
        Ok(main)
    } else {
        Err(Errors::new(c.errors))
    }
}

// ─── Analysis: captures and assignments ─────────────────────────────────────

struct RScope {
    names: Vec<(Symbol, u32)>,
    depth: usize,
    global: bool,
}

struct Resolver<'a> {
    ast: &'a Ast,
    scopes: Vec<RScope>,
    /// Current function depth.
    depth: usize,
    /// Declarations (by position) captured by a nested function.
    captured: FxHashSet<u32>,
    /// Expressions containing an assignment.
    assign: Vec<bool>,
}

/// Names declared by a block (visible in the whole block): `let`, `def`, types.
fn block_decls(b: &Block) -> Vec<(Symbol, u32)> {
    let mut out = Vec::new();
    for s in &b.stmts {
        match &s.kind {
            StmtKind::Let(l) => pattern_bindings(&l.pattern, &mut out),
            StmtKind::Def(f) => out.push((f.name, f.name_span.start)),
            StmtKind::Type(t) => out.push((t.name, t.name_span.start)),
            StmtKind::Enum(e) => out.push((e.name, e.name_span.start)),
            StmtKind::Expr(_) | StmtKind::Module(_) | StmtKind::Require(_) => {}
        }
    }
    out
}

/// `prefix::name`
pub(crate) fn qualify(i: &mut Interner, prefix: Option<Symbol>, name: Symbol) -> Symbol {
    match prefix {
        Some(p) => {
            let q = format!("{}::{}", i.resolve(p), i.resolve(name));
            i.intern(&q)
        }
        None => name,
    }
}

/// Interns the qualified names of the modules' declarations (`Http::get`), so that the
/// checker (which does not intern) can find them.
pub fn intern_module_names(program: &Program, i: &mut Interner) {
    fn walk(b: &Block, prefix: Option<Symbol>, i: &mut Interner) {
        for s in &b.stmts {
            match &s.kind {
                StmtKind::Module(m) => {
                    let q = qualify(i, prefix, m.name);
                    walk(&m.body, Some(q), i);
                }
                _ if prefix.is_none() => {}
                StmtKind::Let(l) => {
                    let mut names = Vec::new();
                    pattern_bindings(&l.pattern, &mut names);
                    for (n, _) in names {
                        qualify(i, prefix, n);
                    }
                }
                StmtKind::Def(f) => {
                    qualify(i, prefix, f.name);
                }
                StmtKind::Type(t) => {
                    qualify(i, prefix, t.name);
                }
                StmtKind::Enum(e) => {
                    qualify(i, prefix, e.name);
                }
                StmtKind::Expr(_) | StmtKind::Require(_) => {}
            }
        }
    }
    walk(&program.body, None, i);
}

fn pattern_bindings(p: &Pattern, out: &mut Vec<(Symbol, u32)>) {
    match p {
        Pattern::Name(n, s) => out.push((*n, s.start)),
        Pattern::Tuple(items, _) => items.iter().for_each(|p| pattern_bindings(p, out)),
        Pattern::Record(names, _) => out.extend(names.iter().map(|(n, s)| (*n, s.start))),
        Pattern::List { items, rest, .. } => {
            items.iter().for_each(|p| pattern_bindings(p, out));
            out.extend(rest.iter().map(|(n, s)| (*n, s.start)));
        }
    }
}

impl Resolver<'_> {
    fn push(&mut self, global: bool) {
        self.scopes.push(RScope {
            names: Vec::new(),
            depth: self.depth,
            global,
        });
    }

    fn declare(&mut self, name: Symbol, key: u32) {
        self.scopes.last_mut().unwrap().names.push((name, key));
    }

    fn refer(&mut self, name: Symbol) {
        for s in self.scopes.iter().rev() {
            if let Some(&(_, key)) = s.names.iter().rev().find(|(n, _)| *n == name) {
                if !s.global && s.depth != self.depth {
                    self.captured.insert(key);
                }
                return;
            }
        }
    }

    fn program(&mut self, p: &Program) {
        self.push(true);
        for (n, k) in block_decls(&p.body) {
            self.declare(n, k);
        }
        for s in &p.body.stmts {
            self.stmt(s);
        }
    }

    fn block(&mut self, b: &Block) {
        self.push(false);
        self.block_in_scope(b);
        self.scopes.pop();
    }

    fn block_in_scope(&mut self, b: &Block) {
        for (n, k) in block_decls(b) {
            self.declare(n, k);
        }
        for s in &b.stmts {
            self.stmt(s);
        }
    }

    fn function(&mut self, params: &[Param], body: &Block, self_key: Option<u32>) {
        self.depth += 1;
        self.push(false);
        if let Some(k) = self_key {
            self.declare(SELF, k);
        }
        for p in params {
            self.declare(p.name, p.span.start);
        }
        for p in params {
            if let Some(d) = p.default {
                self.expr(d);
            }
        }
        self.block_in_scope(body);
        self.scopes.pop();
        self.depth -= 1;
    }

    fn stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Require(_) => {}
            StmtKind::Module(m) => {
                self.push(true);
                for (n, k) in block_decls(&m.body) {
                    self.declare(n, k);
                }
                for s in &m.body.stmts {
                    self.stmt(s);
                }
                self.scopes.pop();
            }
            StmtKind::Expr(e) => {
                self.expr(*e);
            }
            StmtKind::Let(l) => {
                self.expr(l.value);
            }
            StmtKind::Def(f) => self.function(f.params.as_deref().unwrap_or(&[]), &f.body, None),
            StmtKind::Type(t) => {
                for m in &t.members {
                    match m {
                        Member::Field(f) => {
                            if let Some(d) = f.default {
                                self.depth += 1;
                                self.expr(d);
                                self.depth -= 1;
                            }
                        }
                        Member::Method(f) => self.function(
                            f.params.as_deref().unwrap_or(&[]),
                            &f.body,
                            Some(f.span.start),
                        ),
                        Member::Stmt(s) => self.stmt(s),
                        Member::Abstract(_) | Member::Include(..) => {}
                    }
                }
            }
            StmtKind::Enum(e) => {
                for v in &e.variants {
                    if let Some(x) = v.value {
                        self.expr(x);
                    }
                }
                for m in &e.methods {
                    self.function(
                        m.params.as_deref().unwrap_or(&[]),
                        &m.body,
                        Some(m.span.start),
                    );
                }
            }
        }
    }

    fn lambda(&mut self, id: LambdaId) {
        let l = self.ast.lambda(id);
        self.function(&l.params, &l.body, None);
    }

    fn args(&mut self, args: Slice<Arg>, block: Option<LambdaId>) -> bool {
        let mut a = false;
        for arg in self.ast.get(args) {
            a |= self.expr(arg.value);
        }
        if let Some(b) = block {
            self.lambda(b);
        }
        a
    }

    /// Analyzes an expression; true if it contains an assignment.
    fn expr(&mut self, e: ExprId) -> bool {
        stacker::maybe_grow(128 * 1024, 2 * 1024 * 1024, || self.expr_inner(e))
    }

    fn expr_inner(&mut self, e: ExprId) -> bool {
        let ast = self.ast;
        let a = match ast.kind(e) {
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Bool(_)
            | ExprKind::Nil
            | ExprKind::Break
            | ExprKind::Next => false,
            ExprKind::Ident(n) => {
                self.refer(n);
                false
            }
            ExprKind::IVar(_) | ExprKind::SelfRef => {
                self.refer(SELF);
                false
            }
            ExprKind::Path(p) => {
                if let Some(&h) = ast.get(p).first() {
                    self.refer(h);
                }
                false
            }
            ExprKind::Interp(parts) => {
                let mut a = false;
                for p in ast.get(parts) {
                    if let StrPart::Expr(x) = *p {
                        a |= self.expr(x);
                    }
                }
                a
            }
            ExprKind::List(items) | ExprKind::Tuple(items) => {
                let mut a = false;
                for &x in ast.get(items) {
                    a |= self.expr(x);
                }
                a
            }
            ExprKind::Hash(entries) => {
                let mut a = false;
                for &(k, v) in ast.get(entries) {
                    a |= self.expr(k);
                    a |= self.expr(v);
                }
                a
            }
            ExprKind::Record(fields) => {
                let mut a = false;
                for &(_, v) in ast.get(fields) {
                    a |= self.expr(v);
                }
                a
            }
            ExprKind::Unary(_, x) => self.expr(x),
            ExprKind::Binary(_, x, y) | ExprKind::Index(x, y) | ExprKind::IndexOpt(x, y) => {
                let a = self.expr(x);
                self.expr(y) || a
            }
            ExprKind::Range { lo, hi, .. } => {
                let a = self.expr(lo);
                self.expr(hi) || a
            }
            ExprKind::Assign { target, value, .. } => {
                match ast.kind(target) {
                    ExprKind::Ident(n) => self.refer(n),
                    _ => {
                        self.expr(target);
                    }
                }
                self.expr(value);
                true
            }
            ExprKind::Call {
                callee,
                args,
                block,
            } => {
                let a = self.expr(callee);
                self.args(args, block) || a
            }
            ExprKind::MethodCall {
                recv, args, block, ..
            } => {
                let a = self.expr(recv);
                self.args(args, block) || a
            }
            ExprKind::Field(x, _) | ExprKind::TupleIndex(x, _) => self.expr(x),
            ExprKind::Lambda(id) => {
                self.lambda(id);
                false
            }
            ExprKind::If(c) => {
                let x = ast.if_(c);
                self.expr(x.cond);
                self.block(&x.then);
                if let Some(b) = &x.else_ {
                    self.block(b);
                }
                true
            }
            ExprKind::Case(c) => {
                let x = ast.case(c);
                self.expr(x.subject);
                for arm in &x.arms {
                    for &p in ast.get(arm.patterns) {
                        self.expr(p);
                    }
                    self.block(&arm.body);
                }
                if let Some(b) = &x.else_ {
                    self.block(b);
                }
                true
            }
            ExprKind::While(c) => {
                let x = ast.while_(c);
                self.expr(x.cond);
                self.block(&x.body);
                true
            }
            ExprKind::For(c) => {
                let x = ast.for_(c);
                self.expr(x.iter);
                self.push(false);
                for (n, s) in &x.vars {
                    self.declare(*n, s.start);
                }
                self.block_in_scope(&x.body);
                self.scopes.pop();
                true
            }
            ExprKind::Begin(c) => {
                let x = ast.begin(c);
                self.block(&x.body);
                for r in &x.rescues {
                    self.push(false);
                    if let Some((n, s)) = r.var {
                        self.declare(n, s.start);
                    }
                    self.block_in_scope(&r.body);
                    self.scopes.pop();
                }
                for b in x.else_.iter().chain(&x.ensure) {
                    self.block(b);
                }
                true
            }
            ExprKind::Raise(v) => self.expr(v),
            ExprKind::Super(args) => {
                self.refer(SELF);
                self.args(args, None)
            }
            ExprKind::Return(v) => {
                if let Some(v) = v {
                    self.expr(v);
                }
                false
            }
        };
        self.assign[e.0 as usize] = a;
        a
    }
}

// ─── Translation ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
enum Store {
    Reg(Reg),
    Cell(u32),
    Up(u32),
    Global(u32),
}

#[derive(Clone, Copy)]
struct Var {
    name: Symbol,
    store: Store,
    /// Type of the register (`Store::Reg`), `Value` otherwise.
    ty: RegTy,
    mutable: bool,
    /// Type enforced on values stored in the variable (cells included).
    guard: Option<RegTy>,
}

struct Scope {
    vars: Vec<Var>,
    /// Main scope: its variables are globals.
    global: bool,
    /// Body of a module (qualified name): its globals are named `Module::name`.
    module: Option<Symbol>,
}

struct BB {
    instrs: Vec<Instr>,
    spans: Vec<Span>,
    term: Option<(Term, Span)>,
    handler: Option<u32>,
}

/// `ensure` of an enclosing `begin`: run again before a `return` / `break` / `next`
/// that leaves it.
#[derive(Clone, Copy)]
struct EnsureCtx {
    begin: ControlId,
    /// Loops open outside the `begin`.
    loops: usize,
    /// Handler outside the `begin`.
    handler: Option<u32>,
}

struct Fn_ {
    kind: ProtoKind,
    name: Rc<str>,
    regs: Vec<RegTy>,
    is_var: Vec<bool>,
    ncells: u32,
    captures: Vec<Capture>,
    /// Name → (upvalue, variable, declaring function). A given outer variable has a
    /// single visible name here.
    upvals: Vec<(Symbol, u32, Var, usize)>,
    blocks: Vec<BB>,
    cur: BlockId,
    consts: Vec<Value>,
    str_consts: FxHashMap<Symbol, u32>,
    calls: Vec<CallSite>,
    shapes: Vec<Rc<Shape>>,
    types: Vec<TypeRef>,
    records: Vec<Rc<[Symbol]>>,
    children: Vec<Rc<Proto>>,
    decls: Vec<Decl>,
    scopes: Vec<Scope>,
    /// (target of `next`, target of `break`)
    loops: Vec<(BlockId, BlockId)>,
    handlers: Vec<Handler>,
    nslots: u32,
    /// Handler of the blocks being created.
    handler: Option<u32>,
    ensures: Vec<EnsureCtx>,
    free: [Vec<Reg>; 4],
    temps: Vec<Reg>,
    params: Vec<ParamProto>,
    ret: Option<TypeRef>,
    /// Fields of `self` (method of a `type` / `class`, and its lambdas).
    this: Option<Rc<[Symbol]>>,
    /// Method of a class that inherits (and its lambdas): (global slot of the parent,
    /// name of the method), for `super`.
    sup: Option<(u32, Symbol)>,
    /// Register 0 is `self`.
    has_self: bool,
    /// Fields of the type whose default value is being computed.
    ctx_fields: Option<Rc<[Symbol]>>,
    /// Index of the nearest enclosing `do` block function (names → fields).
    region: Option<usize>,
    decl: Option<Rc<FnDecl>>,
    annotations: Annotations,
}

fn ty_index(t: RegTy) -> usize {
    match t {
        RegTy::Value => 0,
        RegTy::Int => 1,
        RegTy::Float => 2,
        RegTy::Bool => 3,
    }
}

struct Compiler<'a> {
    ast: &'a Ast,
    script: &'a Rc<Script>,
    i: &'a mut Interner,
    globals: &'a mut Globals,
    prims: &'a Prims,
    captured: FxHashSet<u32>,
    assign: Vec<bool>,
    fns: Vec<Fn_>,
    errors: Vec<Error>,
    /// Fields (parents' included) of the classes compiled so far.
    classes: FxHashMap<Symbol, Rc<[Symbol]>>,
    /// Traits compiled so far.
    traits: FxHashSet<Symbol>,
}

type Op = (Reg, RegTy);

impl Compiler<'_> {
    // ─── Functions under construction ────────────────────────────────────────

    fn f(&mut self) -> &mut Fn_ {
        self.fns.last_mut().unwrap()
    }

    fn begin(&mut self, kind: ProtoKind, name: Rc<str>) {
        let parent = self.fns.last();
        let this = parent.and_then(|p| p.this.clone());
        let sup = parent.and_then(|p| p.sup);
        let ctx_fields = parent.and_then(|p| p.ctx_fields.clone());
        let mut region = parent.and_then(|p| p.region);
        if kind == ProtoKind::DoBlock {
            region = Some(self.fns.len());
        }
        self.fns.push(Fn_ {
            kind,
            name,
            regs: Vec::new(),
            is_var: Vec::new(),
            ncells: 0,
            captures: Vec::new(),
            upvals: Vec::new(),
            blocks: vec![BB {
                instrs: Vec::new(),
                spans: Vec::new(),
                term: None,
                handler: None,
            }],
            cur: 0,
            consts: Vec::new(),
            str_consts: FxHashMap::default(),
            calls: Vec::new(),
            shapes: Vec::new(),
            types: Vec::new(),
            records: Vec::new(),
            children: Vec::new(),
            decls: Vec::new(),
            scopes: Vec::new(),
            loops: Vec::new(),
            handlers: Vec::new(),
            nslots: 0,
            handler: None,
            ensures: Vec::new(),
            free: Default::default(),
            temps: Vec::new(),
            params: Vec::new(),
            ret: None,
            this,
            sup,
            has_self: false,
            ctx_fields,
            region,
            decl: None,
            annotations: Rc::new([]),
        });
    }

    fn finish(&mut self, ret: Op, ret_shape: Option<u32>, span: Span) -> Rc<Proto> {
        let ret = self.boxed(ret, span);
        self.term(
            Term::Return {
                reg: ret,
                shape: ret_shape,
            },
            span,
        );
        let mut f = self.fns.pop().unwrap();
        thread_jumps(&mut f.blocks, &mut f.handlers);
        merge_blocks(&mut f.blocks, &f.handlers);
        let reachable = compact(&mut f.blocks, &mut f.handlers);
        f.blocks = reachable;
        let plain_params = f.params.iter().all(|p| p.ty.is_none());
        // Loop: a jump to a block that does not come later.
        let has_loop = f.blocks.iter().enumerate().any(|(i, b)| {
            b.term.as_ref().is_some_and(|(t, _)| {
                let mut t = t.clone();
                t.targets_mut().into_iter().any(|x| (*x as usize) <= i)
            })
        });
        let blocks = f
            .blocks
            .into_iter()
            .map(|b| {
                let (term, term_span) = b.term.unwrap_or((Term::Jump(0), span));
                IrBlock {
                    instrs: b.instrs,
                    spans: b.spans,
                    term,
                    term_span,
                    handler: b.handler,
                }
            })
            .collect();
        Rc::new(Proto {
            name: f.name,
            script: self.script.clone(),
            kind: f.kind,
            params: f.params,
            ret: f.ret,
            regs: f.regs,
            ncells: f.ncells,
            captures: f.captures,
            blocks,
            consts: f.consts,
            calls: f.calls,
            shapes: f.shapes,
            types: f.types,
            records: f.records,
            children: f.children,
            decls: f.decls,
            decl: f.decl,
            annotations: f.annotations,
            handlers: f.handlers,
            nslots: f.nslots,
            jit: Default::default(),
            jit_dyn: Default::default(),
            calls_seen: Default::default(),
            has_loop,
            plain_params,
        })
    }

    fn emit(&mut self, ins: Instr, span: Span) {
        let f = self.f();
        let b = &mut f.blocks[f.cur as usize];
        if b.term.is_none() {
            b.instrs.push(ins);
            b.spans.push(span);
        }
    }

    fn emit_in(&mut self, block: BlockId, ins: Instr, span: Span) {
        let b = &mut self.f().blocks[block as usize];
        b.instrs.push(ins);
        b.spans.push(span);
    }

    fn term(&mut self, t: Term, span: Span) {
        let f = self.f();
        let b = &mut f.blocks[f.cur as usize];
        if b.term.is_some() {
            return;
        }
        // `r = a <i b` followed by `if r`: comparison and branch fused.
        let t = match t {
            Term::Branch { cond, then, else_ } if !f.is_var[cond as usize] => match b.instrs.last()
            {
                Some(&Instr::IBin(op, d, x, y)) if d == cond && is_cmp(op) => {
                    b.instrs.pop();
                    b.spans.pop();
                    Term::IfInt {
                        op,
                        a: x,
                        b: y,
                        then,
                        else_,
                    }
                }
                Some(&Instr::IBinK(op, d, x, k)) if d == cond && is_cmp(op) => {
                    b.instrs.pop();
                    b.spans.pop();
                    Term::IfIntK {
                        op,
                        a: x,
                        k,
                        then,
                        else_,
                    }
                }
                _ => Term::Branch { cond, then, else_ },
            },
            t => t,
        };
        b.term = Some((t, span));
    }

    fn term_in(&mut self, block: BlockId, t: Term, span: Span) {
        let b = &mut self.f().blocks[block as usize];
        if b.term.is_none() {
            b.term = Some((t, span));
        }
    }

    /// New block, protected by the current handler.
    fn new_block(&mut self) -> BlockId {
        let f = self.f();
        f.blocks.push(BB {
            instrs: Vec::new(),
            spans: Vec::new(),
            term: None,
            handler: f.handler,
        });
        (f.blocks.len() - 1) as BlockId
    }

    fn switch(&mut self, b: BlockId) {
        self.f().cur = b;
    }

    fn cur(&mut self) -> BlockId {
        self.f().cur
    }

    // ─── Registers ───────────────────────────────────────────────────────────

    fn new_reg(&mut self, ty: RegTy, var: bool) -> Reg {
        let f = self.f();
        f.regs.push(ty);
        f.is_var.push(var);
        (f.regs.len() - 1) as Reg
    }

    /// Temporary (reused after the current instruction).
    fn tmp(&mut self, ty: RegTy) -> Reg {
        let r = match self.f().free[ty_index(ty)].pop() {
            Some(r) => r,
            None => self.new_reg(ty, false),
        };
        self.f().temps.push(r);
        r
    }

    /// `n` consecutive temporaries (`value`).
    fn range(&mut self, n: u32) -> Reg {
        let start = self.f().regs.len() as Reg;
        for _ in 0..n {
            let r = self.new_reg(RegTy::Value, false);
            self.f().temps.push(r);
        }
        start
    }

    fn var_reg(&mut self, ty: RegTy) -> Reg {
        self.new_reg(ty, true)
    }

    fn temps_mark(&mut self) -> usize {
        self.f().temps.len()
    }

    fn release(&mut self, mark: usize) {
        let f = self.f();
        let freed: Vec<Reg> = f.temps.drain(mark..).collect();
        for r in freed {
            let t = f.regs[r as usize];
            f.free[ty_index(t)].push(r);
        }
    }

    fn konst(&mut self, v: Value) -> u32 {
        let f = self.f();
        f.consts.push(v);
        (f.consts.len() - 1) as u32
    }

    fn str_const(&mut self, s: Symbol) -> u32 {
        if let Some(&c) = self.f().str_consts.get(&s) {
            return c;
        }
        let v = Value::str(self.i.resolve(s));
        let c = self.konst(v);
        self.f().str_consts.insert(s, c);
        c
    }

    /// Runtime error (the checker already reported it, if enabled).
    fn fail(&mut self, msg: impl Into<String>, span: Span) -> Op {
        let c = self.konst(Value::Str(msg.into().into()));
        self.emit(Instr::Fail(c), span);
        let r = self.tmp(RegTy::Value);
        (r, RegTy::Value)
    }

    /// Operand in a `value` register (for a list, a call…).
    fn boxed(&mut self, op: Op, _span: Span) -> Reg {
        op.0
    }

    /// Copies into `dst`, converting if needed (`int` → `float`, guard into typed).
    ///
    /// If `src` is a temporary produced by the previous instruction, that instruction
    /// writes directly into `dst`: `s += i` gives `r0 = r0 +i r7`, not two instructions.
    fn store_to(&mut self, dst: Reg, src: Op, span: Span) {
        let dty = self.f().regs[dst as usize];
        if dst == src.0 {
            return;
        }
        if dty == src.1 && !self.f().is_var[src.0 as usize] {
            let f = self.f();
            let b = &mut f.blocks[f.cur as usize];
            if b.term.is_none()
                && let Some(last) = b.instrs.last_mut()
                && let Some(d) = dest_mut(last)
                && *d == src.0
            {
                *d = dst;
                return;
            }
        }
        let ins = match (dty, src.1) {
            (a, b) if a == b => Instr::Move(dst, src.0),
            (RegTy::Value, _) => Instr::Move(dst, src.0),
            (RegTy::Float, RegTy::Int) => Instr::IToF(dst, src.0),
            (t, _) => Instr::Guard(dst, src.0, t),
        };
        self.emit(ins, span);
    }

    /// Value conforming to the enforced type `guard` (for a cell or a global).
    fn guarded(&mut self, src: Op, guard: Option<RegTy>, span: Span) -> Op {
        match guard {
            Some(g) if g != src.1 => {
                let t = self.tmp(g);
                self.store_to(t, src, span);
                (t, g)
            }
            _ => src,
        }
    }

    // ─── Variables ───────────────────────────────────────────────────────────

    fn push_scope(&mut self, global: bool) {
        self.f().scopes.push(Scope {
            vars: Vec::new(),
            global,
            module: None,
        });
    }

    // ─── Modules ─────────────────────────────────────────────────────────────

    /// Enclosing modules, innermost first.
    fn modules(&self) -> Vec<Symbol> {
        self.fns[0].scopes.iter().rev().filter_map(|s| s.module).collect()
    }

    /// Module named by a path written here (`Http`, `Http::Client`): looked up in the
    /// enclosing modules first.
    fn module_path(&mut self, path: &[Symbol]) -> Option<Symbol> {
        let (&head, rest) = path.split_first()?;
        let mut q = self
            .modules()
            .into_iter()
            .map(Some)
            .chain([None])
            .map(|p| qualify(self.i, p, head))
            .find(|q| self.globals.is_module(*q))?;
        for &seg in rest {
            q = qualify(self.i, Some(q), seg);
            if !self.globals.is_module(q) {
                return None;
            }
        }
        Some(q)
    }

    /// The expression names a module (and not a variable).
    fn module_of(&mut self, e: ExprId) -> Option<Symbol> {
        match self.ast.kind(e) {
            ExprKind::Ident(n) if self.resolve(n).is_none() => self.module_path(&[n]),
            ExprKind::Path(p) => {
                let p = self.ast.get(p).to_vec();
                self.module_path(&p)
            }
            _ => None,
        }
    }

    /// Global slot of a member: `path` = module(s) then the member's name.
    fn member_slot(&mut self, path: &[Symbol]) -> Option<u32> {
        let (&last, prefix) = path.split_last()?;
        let m = self.module_path(prefix)?;
        let q = qualify(self.i, Some(m), last);
        Some(self.globals.slot(q))
    }

    /// Global slot of a name written here: variable in scope, member of an enclosing
    /// module (also declared by an earlier script), path, or plain global.
    fn global_slot(&mut self, path: &[Symbol]) -> u32 {
        if let [name] = path {
            if let Some((v, _)) = self.resolve(*name)
                && let Store::Global(s) = v.store
            {
                return s;
            }
            for m in self.modules() {
                let q = qualify(self.i, Some(m), *name);
                if self.globals.has_slot(q) {
                    return self.globals.slot(q);
                }
            }
            return self.globals.slot(*name);
        }
        match self.member_slot(path) {
            Some(s) => s,
            None => {
                let mut q = path[0];
                for &seg in &path[1..] {
                    q = qualify(self.i, Some(q), seg);
                }
                self.globals.slot(q)
            }
        }
    }

    /// Slot of a declaration of the current scope (in a module: `Module::name`).
    fn decl_slot(&mut self, name: Symbol) -> u32 {
        let m = self.f().scopes.last().and_then(|s| s.module);
        let q = qualify(self.i, m, name);
        self.globals.slot(q)
    }

    /// `Http::Base` written as one name (after `<`, `include`): its segments.
    fn split_path(&mut self, name: Symbol) -> Vec<Symbol> {
        let text = self.i.resolve(name).to_string();
        text.split("::").map(|s| self.i.intern(s)).collect()
    }

    fn module_decl(&mut self, m: &ModuleDecl, span: Span) {
        let at_top = self.fns.len() == 1 && self.f().scopes.iter().all(|s| s.global);
        if !at_top {
            self.errors.push(Error::syntax(
                "a `module` must be at the top level of the script (or in a module)",
                span,
            ));
            return;
        }
        let outer = self.f().scopes.last().and_then(|s| s.module);
        let q = qualify(self.i, outer, m.name);
        self.globals.add_module(q);
        self.f().scopes.push(Scope {
            vars: Vec::new(),
            global: true,
            module: Some(q),
        });
        let mark = self.temps_mark();
        self.block_body(&m.body, true);
        self.release(mark);
        self.pop_scope();
    }

    fn pop_scope(&mut self) {
        self.f().scopes.pop();
    }

    /// Type of a variable: its annotation, otherwise the checker's inference.
    fn var_prim(&self, key: u32, ann: Option<&TypeExpr>) -> Option<RegTy> {
        if let Some(t) = ann {
            return prim_of(t, self.i);
        }
        self.prims.vars.get(&key).copied()
    }

    /// Declares a local variable in the current scope.
    fn declare_local(
        &mut self,
        name: Symbol,
        key: u32,
        prim: Option<RegTy>,
        mutable: bool,
        span: Span,
    ) -> Var {
        let var = if self.captured.contains(&key) {
            let c = self.f().ncells;
            self.f().ncells += 1;
            self.emit(Instr::NewCell(c), span);
            Var {
                name,
                store: Store::Cell(c),
                ty: RegTy::Value,
                mutable,
                guard: prim,
            }
        } else {
            let ty = prim.unwrap_or(RegTy::Value);
            let r = self.var_reg(ty);
            Var {
                name,
                store: Store::Reg(r),
                ty,
                mutable,
                guard: prim,
            }
        };
        self.f().scopes.last_mut().unwrap().vars.push(var);
        var
    }

    fn declare_global(&mut self, name: Symbol, mutable: bool) -> Var {
        let slot = self.decl_slot(name);
        let var = Var {
            name,
            store: Store::Global(slot),
            ty: RegTy::Value,
            mutable,
            guard: None,
        };
        self.f().scopes.last_mut().unwrap().vars.push(var);
        var
    }

    /// Looks up `name` in function `fi` and its enclosing functions. Returns the variable
    /// as seen from `fi` and the index of the function that declares it.
    fn resolve_in(&mut self, fi: usize, name: Symbol) -> Option<(Var, usize)> {
        for s in self.fns[fi].scopes.iter().rev() {
            if let Some(v) = s.vars.iter().rev().find(|v| v.name == name) {
                return Some((*v, fi));
            }
        }
        if let Some(&(_, _, v, origin)) = self.fns[fi].upvals.iter().find(|u| u.0 == name) {
            return Some((v, origin));
        }
        if fi == 0 {
            return None;
        }
        let (outer, origin) = self.resolve_in(fi - 1, name)?;
        let cap = match outer.store {
            Store::Global(_) => return Some((outer, origin)),
            Store::Cell(c) => Capture::Cell(c),
            Store::Up(u) => Capture::Up(u),
            Store::Reg(_) => {
                // The analysis should have put this variable in a cell.
                let n = self.i.resolve(name).to_string();
                self.errors.push(Error::syntax(
                    format!("internal compiler error: capture of `{n}`"),
                    Span::default(),
                ));
                return None;
            }
        };
        let f = &mut self.fns[fi];
        f.captures.push(cap);
        let u = (f.captures.len() - 1) as u32;
        let v = Var {
            store: Store::Up(u),
            ty: RegTy::Value,
            ..outer
        };
        f.upvals.push((name, u, v, origin));
        Some((v, origin))
    }

    fn resolve(&mut self, name: Symbol) -> Option<(Var, usize)> {
        let fi = self.fns.len() - 1;
        self.resolve_in(fi, name)
    }

    /// Must the name be resolved dynamically (field of a construction in progress)?
    fn named_access(&self, name: Symbol, origin: Option<usize>, store: Option<Store>) -> bool {
        if name == SELF {
            return false;
        }
        let Some(region) = self.fns.last().unwrap().region else {
            return false;
        };
        match (origin, store) {
            (_, Some(Store::Global(_))) | (None, _) => true,
            (Some(o), _) => o < region,
        }
    }

    fn fallback(store: Store) -> Fallback {
        match store {
            Store::Reg(r) => Fallback::Reg(r),
            Store::Cell(c) => Fallback::Cell(c),
            Store::Up(u) => Fallback::Up(u),
            Store::Global(g) => Fallback::Global(g),
        }
    }

    /// Field of the type whose default value is being computed?
    fn ctx_field(&self, name: Symbol) -> Option<u32> {
        let fields = self.fns.last()?.ctx_fields.as_ref()?;
        fields.iter().position(|f| *f == name).map(|i| i as u32)
    }

    fn read_var(&mut self, name: Symbol, span: Span) -> Op {
        let found = self.resolve(name);
        let is_global = matches!(
            found,
            Some((
                Var {
                    store: Store::Global(_),
                    ..
                },
                _
            )) | None
        );
        if is_global && let Some(i) = self.ctx_field(name) {
            let d = self.tmp(RegTy::Value);
            self.emit(Instr::CtxField(d, i), span);
            return (d, RegTy::Value);
        }
        let store = found.map(|(v, _)| v.store);
        if self.named_access(name, found.map(|(_, o)| o), store) {
            let fallback = match store {
                Some(s) => Self::fallback(s),
                None => Fallback::Global(self.global_slot(&[name])),
            };
            let dst = self.tmp(RegTy::Value);
            self.emit(
                Instr::GetNamed {
                    dst,
                    name,
                    fallback,
                },
                span,
            );
            return (dst, RegTy::Value);
        }
        match found {
            Some((v, _)) => match v.store {
                Store::Reg(r) => (r, v.ty),
                Store::Cell(c) => {
                    let d = self.tmp(RegTy::Value);
                    self.emit(Instr::GetCell(d, c), span);
                    (d, RegTy::Value)
                }
                Store::Up(u) => {
                    let d = self.tmp(RegTy::Value);
                    self.emit(Instr::GetUp(d, u), span);
                    (d, RegTy::Value)
                }
                Store::Global(g) => {
                    let d = self.tmp(RegTy::Value);
                    self.emit(Instr::GetGlobal(d, g), span);
                    (d, RegTy::Value)
                }
            },
            None => {
                if let Some(m) = self.module_path(&[name]) {
                    let m = self.i.resolve(m).to_string();
                    return self.fail(
                        format!("`{m}` is a module: use `{m}::Name` or `{m}.f(…)`"),
                        span,
                    );
                }
                let g = self.global_slot(&[name]);
                let d = self.tmp(RegTy::Value);
                self.emit(Instr::GetGlobal(d, g), span);
                (d, RegTy::Value)
            }
        }
    }

    /// Stores a value in an existing variable (`init`: first value of a `let`,
    /// allowed even if the variable is constant).
    fn write(&mut self, v: Var, src: Op, init: bool, span: Span) {
        match v.store {
            Store::Reg(r) => self.store_to(r, src, span),
            Store::Cell(c) => {
                let s = self.guarded(src, v.guard, span);
                self.emit(Instr::SetCell(c, s.0), span);
            }
            Store::Up(u) => {
                let s = self.guarded(src, v.guard, span);
                self.emit(Instr::SetUp(u, s.0), span);
            }
            Store::Global(g) if init => self.emit(
                Instr::DefGlobal {
                    slot: g,
                    src: src.0,
                    mutable: v.mutable,
                },
                span,
            ),
            Store::Global(g) => self.emit(Instr::SetGlobal(g, src.0), span),
        }
    }

    fn write_var(&mut self, name: Symbol, src: Op, shape: Option<u32>, span: Span) {
        let found = self.resolve(name);
        let store = found.map(|(v, _)| v.store);
        if self.named_access(name, found.map(|(_, o)| o), store) {
            let fallback = match found {
                Some((v, _)) if v.mutable => Self::fallback(v.store),
                Some(_) => Fallback::None,
                // Global defined by a previous script (REPL).
                None if self.globals.get(name).is_some() => {
                    Fallback::Global(self.globals.slot(name))
                }
                None => Fallback::None,
            };
            self.emit(
                Instr::SetNamed {
                    name,
                    src: src.0,
                    fallback,
                    shape,
                },
                span,
            );
            return;
        }
        match found {
            Some((v, _)) if !v.mutable => {
                let n = self.i.resolve(name).to_string();
                self.fail(
                    format!("`{n}` cannot be modified (constant, function or type)"),
                    span,
                );
            }
            Some((v, _)) => self.write(v, src, false, span),
            None => {
                let n = self.i.resolve(name).to_string();
                let defined = self.globals.get(name).is_some();
                if n.starts_with(|c: char| c.is_ascii_uppercase()) && !defined {
                    // `NAME = value`: constant.
                    let global = self.f().scopes.last().is_some_and(|s| s.global);
                    let v = if global {
                        self.declare_global(name, false)
                    } else {
                        self.declare_local(name, span.start, None, false, span)
                    };
                    self.write(v, src, true, span);
                } else {
                    // Global (from a previous script); execution checks that it exists.
                    let g = self.globals.slot(name);
                    self.emit(Instr::SetGlobal(g, src.0), span);
                }
            }
        }
    }

    // ─── Types and literal shapes ────────────────────────────────────────────

    fn type_ref(&mut self, t: &TypeExpr) -> TypeRef {
        let ty = self.rt_type(t);
        TypeRef {
            ty: Rc::new(ty),
            shown: crate::printer::type_expr(t, self.i).into(),
            span: t.span,
            expr: Rc::new(t.clone()),
        }
    }

    fn type_idx(&mut self, t: &TypeExpr) -> u32 {
        let r = self.type_ref(t);
        let f = self.f();
        f.types.push(r);
        (f.types.len() - 1) as u32
    }

    fn rt_type(&mut self, t: &TypeExpr) -> RtType {
        match &t.kind {
            TypeKind::Named(path) => {
                if let [name] = path.as_slice() {
                    match self.i.resolve(*name) {
                        "any" => RtType::Any,
                        "int" => RtType::Int,
                        "float" => RtType::Float,
                        "str" => RtType::Str,
                        "bool" => RtType::Bool,
                        "nil" => RtType::Nil,
                        _ => RtType::Named(self.global_slot(&[*name]), *name),
                    }
                } else {
                    match self.member_slot(path) {
                        Some(slot) => RtType::Named(slot, *path.last().unwrap()),
                        None => RtType::Named(u32::MAX, path[0]),
                    }
                }
            }
            TypeKind::List(x) => RtType::List(Box::new(self.rt_type(x))),
            TypeKind::Hash(k, v) => {
                RtType::Hash(Box::new(self.rt_type(k)), Box::new(self.rt_type(v)))
            }
            TypeKind::Record(fs) => {
                RtType::Record(fs.iter().map(|(n, t)| (*n, self.rt_type(t))).collect())
            }
            TypeKind::Tuple(ts) => RtType::Tuple(ts.iter().map(|t| self.rt_type(t)).collect()),
            TypeKind::Optional(x) => RtType::Optional(Box::new(self.rt_type(x))),
            TypeKind::Union(ts) => RtType::Union(ts.iter().map(|t| self.rt_type(t)).collect()),
            TypeKind::LitStr(s) => RtType::LitStr(s.as_str().into()),
            TypeKind::LitInt(n) => RtType::LitInt(*n),
            TypeKind::Fn(..) => RtType::Fn,
        }
    }

    fn shape_rc(&self, e: ExprId) -> Option<Rc<Shape>> {
        let ast = self.ast;
        match ast.kind(e) {
            ExprKind::Record(fields) => Some(Rc::new(Shape::Record(
                ast.get(fields)
                    .iter()
                    .map(|&(n, v)| (n, self.shape_rc(v)))
                    .collect(),
            ))),
            ExprKind::List(items) => Some(Rc::new(Shape::List(
                ast.get(items).iter().map(|&x| self.shape_rc(x)).collect(),
            ))),
            _ => None,
        }
    }

    /// Index of the literal shape of `e` in the current function.
    fn shape(&mut self, e: ExprId) -> Option<u32> {
        let s = self.shape_rc(e)?;
        let f = self.f();
        f.shapes.push(s);
        Some((f.shapes.len() - 1) as u32)
    }

    // ─── Program, blocks, statements ─────────────────────────────────────────

    fn main(&mut self, body: &Block) -> Rc<Proto> {
        self.begin(ProtoKind::Main, "<main>".into());
        self.push_scope(true);
        let v = self.block_body(body, true);
        let span = body.span;
        self.finish(v, None, span)
    }

    /// Statements of a block in the current scope (declarations hoisted to the top).
    fn block_body(&mut self, b: &Block, top: bool) -> Op {
        self.hoist(b, top);
        let n = b.stmts.len();
        let mut last = None;
        for (k, s) in b.stmts.iter().enumerate() {
            let mark = self.temps_mark();
            let v = self.stmt(s);
            if k + 1 == n {
                // The value of the last statement stays alive for the caller.
                last = v;
            } else {
                self.release(mark);
            }
        }
        match last {
            Some(v) => v,
            None => self.nil(b.span),
        }
    }

    fn block(&mut self, b: &Block) -> Op {
        self.push_scope(false);
        let v = self.block_body(b, false);
        self.pop_scope();
        v
    }

    fn nil(&mut self, span: Span) -> Op {
        let r = self.tmp(RegTy::Value);
        self.emit(Instr::Nil(r), span);
        (r, RegTy::Value)
    }

    /// Declares the block's `let`s (variables), types, enums and `def`s (closures).
    fn hoist(&mut self, b: &Block, top: bool) {
        let mut let_count: FxHashMap<Symbol, u32> = FxHashMap::default();
        for s in &b.stmts {
            if let StmtKind::Let(l) = &s.kind {
                let mut names = Vec::new();
                pattern_bindings(&l.pattern, &mut names);
                for (n, _) in names {
                    *let_count.entry(n).or_default() += 1;
                }
            }
        }
        // 1. Variables of the `let`s.
        for s in &b.stmts {
            if let StmtKind::Let(l) = &s.kind {
                let mut names = Vec::new();
                pattern_bindings(&l.pattern, &mut names);
                let ann = match (&l.pattern, &l.ty) {
                    (Pattern::Name(..), Some(t)) => Some(t),
                    _ => None,
                };
                for (n, key) in names {
                    if self
                        .f()
                        .scopes
                        .last()
                        .unwrap()
                        .vars
                        .iter()
                        .any(|v| v.name == n)
                    {
                        continue;
                    }
                    if top {
                        self.declare_global(n, true);
                    } else {
                        // Several `let`s with the same name: untyped.
                        let prim = if let_count[&n] == 1 {
                            self.var_prim(key, ann)
                        } else {
                            None
                        };
                        self.declare_local(n, key, prim, true, s.span);
                    }
                }
            }
        }
        // Modules: known first (they may refer to each other), then their bodies run,
        // in order, before the block's types (which may inherit from their classes).
        if top {
            let outer = self.f().scopes.last().and_then(|s| s.module);
            for s in &b.stmts {
                if let StmtKind::Module(m) = &s.kind {
                    let q = qualify(self.i, outer, m.name);
                    self.globals.add_module(q);
                }
            }
        }
        for s in &b.stmts {
            if let StmtKind::Module(m) = &s.kind {
                self.module_decl(m, s.span);
            }
        }
        // 2. Types and enums (main program only).
        for s in &b.stmts {
            match &s.kind {
                StmtKind::Type(t) if top => self.type_decl(t),
                StmtKind::Enum(e) if top => self.enum_decl(e),
                StmtKind::Type(_) | StmtKind::Enum(_) => self.errors.push(Error::syntax(
                    "`type`, `class`, `annotation` and `enum` declarations must be at the top level of the script",
                    s.span,
                )),
                _ => {}
            }
        }
        // 3. Functions: all declared, then their closures created.
        let mut defs = Vec::new();
        for s in &b.stmts {
            if let StmtKind::Def(f) = &s.kind {
                let v = if top {
                    self.declare_global(f.name, false)
                } else {
                    self.declare_local(f.name, f.name_span.start, None, false, f.span)
                };
                defs.push((v, f));
            }
        }
        for (v, f) in defs {
            let name = self.i.resolve(f.name).into();
            let idx = self.function(
                ProtoKind::Function,
                name,
                f.params.as_deref().unwrap_or(&[]),
                &f.body,
                f.ret.as_ref(),
                Some(Rc::new((**f).clone())),
                false,
            );
            let r = self.tmp(RegTy::Value);
            self.emit(Instr::Closure(r, idx), f.span);
            self.write(v, (r, RegTy::Value), true, f.span);
        }
    }

    /// Statement; returns its value (`None` for a declaration: it evaluates to `nil`).
    fn stmt(&mut self, s: &Stmt) -> Option<Op> {
        match &s.kind {
            StmtKind::Expr(e) => Some(self.expr(*e)),
            // Run with the declarations (`hoist`).
            StmtKind::Module(_) => None,
            // Loaded by the engine before compiling.
            StmtKind::Require(_) => {
                let top = self.fns.len() == 1 && self.f().scopes.len() == 1;
                if !top {
                    self.errors.push(Error::syntax(
                        "`require` must be at the top level of the script",
                        s.span,
                    ));
                }
                None
            }
            StmtKind::Let(l) => {
                let span = self.ast.span(l.value);
                let mut v = self.expr(l.value);
                if let Some(t) = &l.ty {
                    let ty = self.type_idx(t);
                    let shape = self.shape(l.value);
                    let dst = self.tmp(prim_of(t, self.i).unwrap_or(RegTy::Value));
                    self.emit(
                        Instr::Coerce {
                            dst,
                            src: v.0,
                            ty,
                            shape,
                        },
                        span,
                    );
                    v = (dst, self.f().regs[dst as usize]);
                }
                self.bind(&l.pattern, v, span);
                None
            }
            // Declared at the top of the block.
            StmtKind::Def(_) | StmtKind::Type(_) | StmtKind::Enum(_) => None,
        }
    }

    /// Stores a value into the variables of a `let` pattern.
    fn bind(&mut self, p: &Pattern, v: Op, span: Span) {
        match p {
            Pattern::Name(n, _) => {
                let var = self
                    .f()
                    .scopes
                    .last()
                    .unwrap()
                    .vars
                    .iter()
                    .rev()
                    .find(|x| x.name == *n)
                    .copied();
                match var {
                    Some(var) => self.write(var, v, true, span),
                    None => self.write_var(*n, v, None, span),
                }
            }
            Pattern::Tuple(items, pspan) => {
                let n = items.len() as u32;
                let dst = self.range(n);
                self.emit(
                    Instr::Unpack {
                        dst,
                        src: v.0,
                        n,
                        rest: false,
                    },
                    *pspan,
                );
                for (k, item) in items.iter().enumerate() {
                    self.bind(item, (dst + k as u32, RegTy::Value), span);
                }
            }
            Pattern::List {
                items,
                rest,
                span: pspan,
            } => {
                let n = items.len() as u32;
                let dst = self.range(n + rest.is_some() as u32);
                self.emit(
                    Instr::Unpack {
                        dst,
                        src: v.0,
                        n,
                        rest: rest.is_some(),
                    },
                    *pspan,
                );
                for (k, item) in items.iter().enumerate() {
                    self.bind(item, (dst + k as u32, RegTy::Value), span);
                }
                if let Some((r, rs)) = rest {
                    self.bind(&Pattern::Name(*r, *rs), (dst + n, RegTy::Value), span);
                }
            }
            Pattern::Record(names, _) => {
                for (n, nspan) in names {
                    let d = self.tmp(RegTy::Value);
                    self.emit(Instr::Field(d, v.0, *n), *nspan);
                    self.bind(&Pattern::Name(*n, *nspan), (d, RegTy::Value), span);
                }
            }
        }
    }

    // ─── Functions ───────────────────────────────────────────────────────────

    /// Compiles a nested function; returns its index in the current function.
    #[allow(clippy::too_many_arguments)]
    fn function(
        &mut self,
        kind: ProtoKind,
        name: Rc<str>,
        params: &[Param],
        body: &Block,
        ret: Option<&TypeExpr>,
        decl: Option<Rc<FnDecl>>,
        method: bool,
    ) -> u32 {
        let p = self.function_proto(kind, name, params, body, ret, decl, method, None);
        let f = self.f();
        f.children.push(p);
        (f.children.len() - 1) as u32
    }

    #[allow(clippy::too_many_arguments)]
    fn function_proto(
        &mut self,
        kind: ProtoKind,
        name: Rc<str>,
        params: &[Param],
        body: &Block,
        ret: Option<&TypeExpr>,
        decl: Option<Rc<FnDecl>>,
        method: bool,
        self_key: Option<u32>,
    ) -> Rc<Proto> {
        let annotations = match &decl {
            Some(d) => self.annotations_ir(&d.annotations),
            None => Rc::new([]),
        };
        let param_annotations: Vec<Annotations> = params
            .iter()
            .map(|p| self.annotations_ir(&p.annotations))
            .collect();
        self.begin(kind, name);
        self.f().decl = decl;
        self.f().annotations = annotations;
        self.push_scope(false);
        let span = body.span;
        if method {
            self.f().has_self = true;
            let r0 = self.var_reg(RegTy::Value);
            let key = self_key.unwrap_or(u32::MAX);
            if self.captured.contains(&key) {
                let c = self.f().ncells;
                self.f().ncells += 1;
                self.emit(Instr::NewCell(c), span);
                self.emit(Instr::SetCell(c, r0), span);
                self.f().scopes.last_mut().unwrap().vars.push(Var {
                    name: SELF,
                    store: Store::Cell(c),
                    ty: RegTy::Value,
                    mutable: false,
                    guard: None,
                });
            } else {
                self.f().scopes.last_mut().unwrap().vars.push(Var {
                    name: SELF,
                    store: Store::Reg(r0),
                    ty: RegTy::Value,
                    mutable: false,
                    guard: None,
                });
            }
        }
        // Parameters: consecutive registers.
        let mut regs = Vec::new();
        for (p, annotations) in params.iter().zip(param_annotations) {
            let prim = p.ty.as_ref().and_then(|t| prim_of(t, self.i));
            let r = self.var_reg(prim.unwrap_or(RegTy::Value));
            regs.push((r, prim));
            let ty = p.ty.as_ref().map(|t| self.type_ref(t));
            let field = if p.is_field {
                let this = self.f().this.clone();
                match this.and_then(|fs| fs.iter().position(|f| *f == p.name)) {
                    Some(i) => Some(i as u32),
                    None => {
                        self.fail(
                            "an `@field` parameter can only be used inside a method",
                            p.span,
                        );
                        None
                    }
                }
            } else {
                None
            };
            self.f().params.push(ParamProto {
                name: p.name,
                reg: r,
                ty,
                has_default: p.default.is_some(),
                field,
                span: p.span,
                annotations,
            });
        }
        // Parameter variables (register, or cell if captured).
        let mut vars = Vec::new();
        for (p, &(r, prim)) in params.iter().zip(&regs) {
            let v = Var {
                name: p.name,
                store: Store::Reg(r),
                ty: prim.unwrap_or(RegTy::Value),
                mutable: true,
                guard: prim,
            };
            self.f().scopes.last_mut().unwrap().vars.push(v);
            vars.push(v);
        }
        // Default values of parameters not provided.
        for (k, p) in params.iter().enumerate() {
            let Some(d) = p.default else { continue };
            let (r, prim) = regs[k];
            let mark = self.temps_mark();
            let t = self.tmp(RegTy::Bool);
            self.emit(Instr::IsUndef(t, r), p.span);
            let bdef = self.new_block();
            let bnext = self.new_block();
            self.term(
                Term::Branch {
                    cond: t,
                    then: bdef,
                    else_: bnext,
                },
                p.span,
            );
            self.switch(bdef);
            let v = self.expr(d);
            match &p.ty {
                Some(ty) => {
                    let ty = self.type_idx(ty);
                    let shape = self.shape(d);
                    self.emit(
                        Instr::Coerce {
                            dst: r,
                            src: v.0,
                            ty,
                            shape,
                        },
                        p.span,
                    );
                }
                None => self.store_to(r, v, p.span),
            }
            let _ = prim;
            self.term(Term::Jump(bnext), p.span);
            self.switch(bnext);
            self.release(mark);
        }
        // `@x` fields of `init`, captured parameters.
        for (k, p) in params.iter().enumerate() {
            let r = regs[k].0;
            if let Some(idx) = self.f().params[k].field {
                self.emit(
                    Instr::SetFieldAt {
                        obj: 0,
                        idx,
                        src: r,
                        shape: None,
                    },
                    p.span,
                );
            }
            if self.captured.contains(&p.span.start) {
                let c = self.f().ncells;
                self.f().ncells += 1;
                self.emit(Instr::NewCell(c), p.span);
                self.emit(Instr::SetCell(c, r), p.span);
                let scope = self.f().scopes.last_mut().unwrap();
                if let Some(v) = scope.vars.iter_mut().rev().find(|v| v.name == p.name) {
                    v.store = Store::Cell(c);
                    v.ty = RegTy::Value;
                }
            }
        }
        if let Some(t) = ret {
            let r = self.type_ref(t);
            self.f().ret = Some(r);
        }
        let v = self.block_body(body, false);
        let shape = tail(body).and_then(|e| self.shape(e));
        self.pop_scope();
        self.finish(v, shape, span)
    }

    fn lambda(&mut self, id: LambdaId, kind: ProtoKind) -> u32 {
        let l = self.ast.lambda(id);
        let name: Rc<str> = if kind == ProtoKind::DoBlock {
            "block".into()
        } else {
            "lambda".into()
        };
        self.function(kind, name, &l.params, &l.body, None, None, false)
    }

    // ─── Annotations ─────────────────────────────────────────────────────────

    /// Compiles `@[Name(args)]`: a parameterless function that builds the annotation.
    /// It is compiled separately, without an enclosing function: its names refer to
    /// globals (an annotation's arguments are constants, not local variables).
    fn annotations_ir(&mut self, uses: &[AnnotationUse]) -> Annotations {
        if uses.is_empty() {
            return Rc::new([]);
        }
        let saved = std::mem::take(&mut self.fns);
        let mut out = Vec::with_capacity(uses.len());
        for u in uses {
            let [name] = u.path.as_slice() else {
                self.errors.push(Error::syntax(
                    "expected an annotation name: `@[Name]` or `@[Name(…)]`",
                    u.span,
                ));
                continue;
            };
            let text = self
                .ast
                .get(u.args)
                .first()
                .and_then(|a| match self.ast.kind(a.value) {
                    ExprKind::Str(s) => Some(Rc::from(self.i.resolve(s))),
                    _ => None,
                });
            let label = format!("@[{}]", self.i.resolve(*name));
            self.begin(ProtoKind::Annotation, label.into());
            self.push_scope(false);
            let f = self.read_var(*name, u.span);
            let site = self.call_site(u.args, None);
            let dst = self.tmp(RegTy::Value);
            self.emit(Instr::Call { dst, f: f.0, site }, u.span);
            self.pop_scope();
            let make = self.finish((dst, RegTy::Value), None, u.span);
            out.push(AnnotationIr {
                name: *name,
                make,
                span: u.span,
                text,
            });
        }
        self.fns = saved;
        out.into()
    }

    // ─── Type declarations ───────────────────────────────────────────────────

    /// Fields: (name, type, default, annotations).
    #[allow(clippy::type_complexity)]
    fn type_fields(
        &self,
        t: &TypeDecl,
    ) -> Vec<(Symbol, Option<TypeExpr>, Option<ExprId>, Vec<AnnotationUse>)> {
        let mut fields: Vec<(Symbol, Option<TypeExpr>, Option<ExprId>, Vec<AnnotationUse>)> =
            Vec::new();
        for m in &t.members {
            if let Member::Field(f) = m {
                fields.push((f.name, f.ty.clone(), f.default, f.annotations.clone()));
            }
        }
        let init = self.i.get("init");
        for m in &t.members {
            if let Member::Method(f) = m
                && Some(f.name) == init
            {
                for p in f.params.iter().flatten().filter(|p| p.is_field) {
                    if !fields.iter().any(|(n, ..)| *n == p.name) {
                        fields.push((p.name, p.ty.clone(), None, p.annotations.clone()));
                    }
                }
            }
        }
        fields
    }

    /// Fields of the parent class (declared before, in this script or a previous one).
    fn parent_fields(&mut self, t: &TypeDecl) -> Option<(u32, Rc<[Symbol]>)> {
        let (p, span) = t.parent?;
        let path = self.split_path(p);
        let slot = self.global_slot(&path);
        if let Some(fields) = self.classes.get(&self.globals.at(slot).name) {
            return Some((slot, fields.clone()));
        }
        match &self.globals.at(slot).value {
            Some(Value::Type(info)) if info.kind == TypeDeclKind::Class => {
                Some((slot, info.fields.iter().map(|f| f.name).collect()))
            }
            _ => {
                self.errors.push(Error::type_error(
                    format!(
                        "`{}` is not a class declared before `{}`",
                        self.i.resolve(p),
                        self.i.resolve(t.name)
                    ),
                    span,
                ));
                None
            }
        }
    }

    fn type_decl(&mut self, t: &TypeDecl) {
        let parent = self.parent_fields(t);
        let inherited: Rc<[Symbol]> = parent.as_ref().map_or(Rc::from([]), |(_, f)| f.clone());
        let mut fields = self.type_fields(t);
        // `init(@x)` for a field of the parent: an assignment, not a new field.
        fields.retain(|(n, ..)| !inherited.contains(n));
        for m in &t.members {
            if let Member::Field(f) = m
                && inherited.contains(&f.name)
            {
                self.errors.push(Error::type_error(
                    format!(
                        "field `{}` is already declared by `{}`",
                        self.i.resolve(f.name),
                        self.i.resolve(t.parent.unwrap().0)
                    ),
                    f.span,
                ));
            }
        }
        let names: Rc<[Symbol]> = inherited
            .iter()
            .copied()
            .chain(fields.iter().map(|(n, ..)| *n))
            .collect();
        if t.kind == TypeDeclKind::Class {
            let slot = self.decl_slot(t.name);
            let qname = self.globals.at(slot).name;
            self.classes.insert(qname, names.clone());
        }
        let mut fir = Vec::new();
        for (name, ty, default, anns) in &fields {
            let annotations = self.annotations_ir(anns);
            let ty = ty.as_ref().map(|t| self.type_ref(t));
            let default = default.map(|d| {
                let span = self.ast.span(d);
                self.begin(
                    ProtoKind::FieldDefault,
                    format!("default of {}", self.i.resolve(*name)).into(),
                );
                self.f().ctx_fields = Some(names.clone());
                self.push_scope(false);
                let v = self.expr(d);
                self.pop_scope();
                self.finish(v, None, span)
            });
            fir.push(FieldIr {
                name: *name,
                ty,
                default,
                annotations,
            });
        }
        let mut methods = Vec::new();
        let mut abstracts = Vec::new();
        let mut traits = Vec::new();
        for m in &t.members {
            match m {
                Member::Abstract(f) => abstracts.push(Rc::new(f.clone())),
                Member::Include(name, span) => {
                    let path = self.split_path(*name);
                    let slot = self.global_slot(&path);
                    let known = self.traits.contains(&self.globals.at(slot).name)
                        || matches!(&self.globals.at(slot).value,
                            Some(Value::Type(info)) if info.kind == TypeDeclKind::Trait);
                    if known {
                        traits.push(slot);
                    } else {
                        self.errors.push(Error::type_error(
                            format!(
                                "`{}` is not a trait declared before `{}`",
                                self.i.resolve(*name),
                                self.i.resolve(t.name)
                            ),
                            *span,
                        ));
                    }
                }
                Member::Method(f) => {
                    let saved = self.f().this.replace(names.clone());
                    let saved_sup = std::mem::replace(
                        &mut self.f().sup,
                        parent.as_ref().map(|(slot, _)| (*slot, f.name)),
                    );
                    let pname = format!("{}.{}", self.i.resolve(t.name), self.i.resolve(f.name));
                    let p = self.function_proto(
                        ProtoKind::Method,
                        pname.into(),
                        f.params.as_deref().unwrap_or(&[]),
                        &f.body,
                        f.ret.as_ref(),
                        Some(Rc::new(f.clone())),
                        true,
                        Some(f.span.start),
                    );
                    self.f().this = saved;
                    self.f().sup = saved_sup;
                    methods.push((f.name, p));
                }
                Member::Stmt(s) => self.errors.push(Error::syntax(
                    "only fields and methods are allowed in a `type` / `class` \
                     (macros will come later)",
                    s.span,
                )),
                Member::Field(_) => {}
            }
        }
        let annotations = self.annotations_ir(&t.annotations);
        let slot = self.decl_slot(t.name);
        let qname = self.globals.at(slot).name;
        if t.kind == TypeDeclKind::Trait {
            self.traits.insert(qname);
        }
        let decl = TypeDeclIr {
            name: qname,
            kind: t.kind,
            parent: parent.map(|(slot, f)| (slot, f.len() as u32)),
            traits,
            abstracts,
            is_abstract: t.is_abstract,
            fields: fir,
            methods,
            annotations,
        };
        let f = self.f();
        f.decls.push(Decl::Type(Rc::new(decl)));
        let idx = (f.decls.len() - 1) as u32;
        self.emit(Instr::DefType { slot, decl: idx }, t.name_span);
    }

    fn enum_decl(&mut self, e: &EnumDecl) {
        let explicit: Vec<bool> = e.variants.iter().map(|v| v.value.is_some()).collect();
        let n = explicit.iter().filter(|x| **x).count() as u32;
        let start = self.range(n);
        let mut k = 0;
        for v in &e.variants {
            if let Some(x) = v.value {
                let val = self.expr(x);
                self.store_to(start + k, val, v.span);
                k += 1;
            }
        }
        let repr = e.repr.as_ref().map(|r| self.type_ref(r));
        let repr_is_str = e
            .repr
            .as_ref()
            .is_some_and(|r| prim_name(r, self.i) == Some("str"));
        let mut methods = Vec::new();
        for m in &e.methods {
            let saved = self.f().this.take();
            let pname = format!("{}.{}", self.i.resolve(e.name), self.i.resolve(m.name));
            let p = self.function_proto(
                ProtoKind::Method,
                pname.into(),
                m.params.as_deref().unwrap_or(&[]),
                &m.body,
                m.ret.as_ref(),
                Some(Rc::new(m.clone())),
                true,
                Some(m.span.start),
            );
            self.f().this = saved;
            methods.push((m.name, p));
        }
        let annotations = self.annotations_ir(&e.annotations);
        let variant_annotations = e
            .variants
            .iter()
            .map(|v| self.annotations_ir(&v.annotations))
            .collect();
        let slot = self.decl_slot(e.name);
        let decl = EnumDeclIr {
            name: self.globals.at(slot).name,
            variants: e.variants.iter().map(|v| v.name).collect(),
            explicit,
            repr,
            repr_is_str,
            methods,
            annotations,
            variant_annotations,
        };
        let f = self.f();
        f.decls.push(Decl::Enum(Rc::new(decl)));
        let idx = (f.decls.len() - 1) as u32;
        self.emit(
            Instr::DefEnum {
                slot,
                decl: idx,
                start,
            },
            e.name_span,
        );
    }

    // ─── Expressions ─────────────────────────────────────────────────────────

    fn expr(&mut self, e: ExprId) -> Op {
        stacker::maybe_grow(128 * 1024, 2 * 1024 * 1024, || self.expr_inner(e))
    }

    /// Successive operands; an operand that is a variable is copied if a later
    /// expression may modify it (`x + (x = 2)` is `1 + 2`).
    fn operands(&mut self, es: &[ExprId]) -> Vec<Op> {
        let mut out = Vec::with_capacity(es.len());
        for (k, &e) in es.iter().enumerate() {
            let mut op = self.expr(e);
            let later_assign = es[k + 1..].iter().any(|x| self.assign[x.0 as usize]);
            if later_assign && self.f().is_var[op.0 as usize] {
                let t = self.tmp(op.1);
                self.emit(Instr::Move(t, op.0), self.ast.span(e));
                op = (t, op.1);
            }
            out.push(op);
        }
        out
    }

    /// Evaluates `es` into consecutive `value` registers; returns the first.
    fn eval_range(&mut self, es: &[ExprId]) -> Reg {
        let start = self.range(es.len() as u32);
        for (k, &e) in es.iter().enumerate() {
            let v = self.expr(e);
            self.store_to(start + k as u32, v, self.ast.span(e));
        }
        start
    }

    fn expr_inner(&mut self, e: ExprId) -> Op {
        let ast = self.ast;
        let span = ast.span(e);
        match ast.kind(e) {
            ExprKind::Int(n) => {
                let r = self.tmp(RegTy::Int);
                self.emit(Instr::Int(r, n), span);
                (r, RegTy::Int)
            }
            ExprKind::Float(x) => {
                let r = self.tmp(RegTy::Float);
                self.emit(Instr::Float(r, x), span);
                (r, RegTy::Float)
            }
            ExprKind::Bool(b) => {
                let r = self.tmp(RegTy::Bool);
                self.emit(Instr::Bool(r, b), span);
                (r, RegTy::Bool)
            }
            ExprKind::Nil => self.nil(span),
            ExprKind::Str(s) => {
                let c = self.str_const(s);
                let r = self.tmp(RegTy::Value);
                self.emit(Instr::Const(r, c), span);
                (r, RegTy::Value)
            }
            ExprKind::Interp(parts) => {
                let parts = ast.get(parts).to_vec();
                let start = self.range(parts.len() as u32);
                for (k, p) in parts.iter().enumerate() {
                    let dst = start + k as u32;
                    match *p {
                        StrPart::Lit(s) => {
                            let c = self.str_const(s);
                            self.emit(Instr::Const(dst, c), span);
                        }
                        StrPart::Expr(x) => {
                            let v = self.expr(x);
                            self.store_to(dst, v, ast.span(x));
                        }
                    }
                }
                let r = self.tmp(RegTy::Value);
                self.emit(Instr::Concat(r, start, parts.len() as u32), span);
                (r, RegTy::Value)
            }
            ExprKind::Ident(name) => self.read_var(name, span),
            ExprKind::SelfRef => self.self_value(span),
            ExprKind::IVar(name) => {
                let Some(idx) = self.this_field(name, span) else {
                    return (self.tmp(RegTy::Value), RegTy::Value);
                };
                let obj = self.self_value(span);
                let d = self.tmp(RegTy::Value);
                self.emit(Instr::FieldAt(d, obj.0, idx), span);
                (d, RegTy::Value)
            }
            ExprKind::Path(p) => {
                let p = ast.get(p).to_vec();
                // Member of a module: `Http::Request`, `Http::VERSION`.
                if let Some(slot) = self.member_slot(&p) {
                    let d = self.tmp(RegTy::Value);
                    self.emit(Instr::GetGlobal(d, slot), span);
                    return (d, RegTy::Value);
                }
                if self.module_path(&p).is_some() {
                    return self.fail("a module is not a value", span);
                }
                // `Enum::Variant`, the enum possibly in a module (`Http::Status::Ok`).
                let (&variant, head) = p.split_last().unwrap();
                let head = match head {
                    [name] => self.read_var(*name, span),
                    _ => match self.member_slot(head) {
                        Some(slot) => {
                            let d = self.tmp(RegTy::Value);
                            self.emit(Instr::GetGlobal(d, slot), span);
                            (d, RegTy::Value)
                        }
                        None => return self.fail("unknown module in this path", span),
                    },
                };
                let d = self.tmp(RegTy::Value);
                self.emit(Instr::Variant(d, head.0, variant), span);
                (d, RegTy::Value)
            }
            ExprKind::List(items) => {
                let items = ast.get(items).to_vec();
                let start = self.eval_range(&items);
                let d = self.tmp(RegTy::Value);
                self.emit(Instr::List(d, start, items.len() as u32), span);
                (d, RegTy::Value)
            }
            ExprKind::Tuple(items) => {
                let items = ast.get(items).to_vec();
                let start = self.eval_range(&items);
                let d = self.tmp(RegTy::Value);
                self.emit(Instr::Tuple(d, start, items.len() as u32), span);
                (d, RegTy::Value)
            }
            ExprKind::Hash(entries) => {
                let flat: Vec<ExprId> =
                    ast.get(entries).iter().flat_map(|&(k, v)| [k, v]).collect();
                let start = self.eval_range(&flat);
                let d = self.tmp(RegTy::Value);
                self.emit(Instr::Hash(d, start, (flat.len() / 2) as u32), span);
                (d, RegTy::Value)
            }
            ExprKind::Record(fields) => {
                let fields = ast.get(fields).to_vec();
                let values: Vec<ExprId> = fields.iter().map(|f| f.1).collect();
                let start = self.eval_range(&values);
                let names: Rc<[Symbol]> = fields.iter().map(|f| f.0).collect();
                let f = self.f();
                f.records.push(names);
                let idx = (f.records.len() - 1) as u32;
                let d = self.tmp(RegTy::Value);
                self.emit(Instr::Record(d, start, idx), span);
                (d, RegTy::Value)
            }
            ExprKind::Unary(op, x) => {
                let a = self.expr(x);
                match (op, a.1) {
                    (UnaryOp::Not, _) => {
                        let d = self.tmp(RegTy::Bool);
                        self.emit(Instr::Not(d, a.0), span);
                        (d, RegTy::Bool)
                    }
                    (UnaryOp::Neg, RegTy::Int) => {
                        let d = self.tmp(RegTy::Int);
                        self.emit(Instr::INeg(d, a.0), span);
                        (d, RegTy::Int)
                    }
                    (UnaryOp::Neg, RegTy::Float) => {
                        let d = self.tmp(RegTy::Float);
                        self.emit(Instr::FNeg(d, a.0), span);
                        (d, RegTy::Float)
                    }
                    (UnaryOp::Neg, _) => {
                        let d = self.tmp(RegTy::Value);
                        self.emit(Instr::Neg(d, a.0), span);
                        (d, RegTy::Value)
                    }
                }
            }
            ExprKind::Binary(op @ (BinOp::And | BinOp::Or), a, b) => {
                let d = self.tmp(RegTy::Value);
                let l = self.expr(a);
                self.store_to(d, l, span);
                let rhs = self.new_block();
                let end = self.new_block();
                let (then, else_) = if op == BinOp::And {
                    (rhs, end)
                } else {
                    (end, rhs)
                };
                self.term(
                    Term::Branch {
                        cond: d,
                        then,
                        else_,
                    },
                    span,
                );
                self.switch(rhs);
                let r = self.expr(b);
                self.store_to(d, r, span);
                self.term(Term::Jump(end), span);
                self.switch(end);
                (d, RegTy::Value)
            }
            ExprKind::Binary(op, a, b) => {
                let ops = self.operands(&[a, b]);
                self.binary(op, ops[0], ops[1], span)
            }
            ExprKind::Range { lo, hi, inclusive } => {
                let ops = self.operands(&[lo, hi]);
                let d = self.tmp(RegTy::Value);
                self.emit(
                    Instr::Range {
                        dst: d,
                        lo: ops[0].0,
                        hi: ops[1].0,
                        inclusive,
                    },
                    span,
                );
                (d, RegTy::Value)
            }
            ExprKind::Assign { target, op, value } => self.assign(target, op, value, span),
            ExprKind::Call {
                callee,
                args,
                block,
            } => {
                let mut f = self.expr(callee);
                if self.args_assign(args) && self.f().is_var[f.0 as usize] {
                    let t = self.tmp(RegTy::Value);
                    self.emit(Instr::Move(t, f.0), span);
                    f = (t, RegTy::Value);
                }
                let site = self.call_site(args, block);
                let dst = self.tmp(RegTy::Value);
                self.emit(Instr::Call { dst, f: f.0, site }, span);
                (dst, RegTy::Value)
            }
            ExprKind::MethodCall {
                recv,
                name,
                args,
                block,
            } if self.module_of(recv).is_some() => {
                // Function of a module: `Http.get(url)`.
                let m = self.module_of(recv).unwrap();
                let q = qualify(self.i, Some(m), name);
                let slot = self.globals.slot(q);
                let f = self.tmp(RegTy::Value);
                self.emit(Instr::GetGlobal(f, slot), span);
                let site = self.call_site(args, block);
                let dst = self.tmp(RegTy::Value);
                self.emit(Instr::Call { dst, f, site }, span);
                (dst, RegTy::Value)
            }
            ExprKind::Field(recv, name) if self.module_of(recv).is_some() => {
                let m = self.module_of(recv).unwrap();
                let q = qualify(self.i, Some(m), name);
                let slot = self.globals.slot(q);
                let d = self.tmp(RegTy::Value);
                self.emit(Instr::GetGlobal(d, slot), span);
                (d, RegTy::Value)
            }
            ExprKind::MethodCall {
                recv,
                name,
                args,
                block,
            } => {
                let mut r = self.expr(recv);
                if self.args_assign(args) && self.f().is_var[r.0 as usize] {
                    let t = self.tmp(r.1);
                    self.emit(Instr::Move(t, r.0), span);
                    r = (t, r.1);
                }
                let site = self.call_site(args, block);
                let dst = self.tmp(RegTy::Value);
                self.emit(
                    Instr::CallMethod {
                        dst,
                        recv: r.0,
                        name,
                        site,
                    },
                    span,
                );
                (dst, RegTy::Value)
            }
            ExprKind::Field(recv, name) => {
                let r = self.expr(recv);
                // Field type known: read and check into a typed register.
                if let Some(&t) = self.prims.fields.get(&e.0) {
                    let d = self.tmp(t);
                    self.emit(Instr::FieldAs(d, r.0, name, t), span);
                    return (d, t);
                }
                let d = self.tmp(RegTy::Value);
                self.emit(Instr::Field(d, r.0, name), span);
                (d, RegTy::Value)
            }
            ExprKind::TupleIndex(recv, idx) => {
                let r = self.expr(recv);
                let d = self.tmp(RegTy::Value);
                self.emit(Instr::TupleIndex(d, r.0, idx), span);
                (d, RegTy::Value)
            }
            ExprKind::Index(recv, idx) | ExprKind::IndexOpt(recv, idx) => {
                let ops = self.operands(&[recv, idx]);
                let d = self.tmp(RegTy::Value);
                let (r, i) = (ops[0].0, ops[1].0);
                let ins = match ast.kind(e) {
                    ExprKind::IndexOpt(..) => Instr::IndexOpt(d, r, i),
                    _ => Instr::Index(d, r, i),
                };
                self.emit(ins, span);
                (d, RegTy::Value)
            }
            ExprKind::Lambda(id) => {
                let idx = self.lambda(id, ProtoKind::Lambda);
                let d = self.tmp(RegTy::Value);
                self.emit(Instr::Closure(d, idx), span);
                (d, RegTy::Value)
            }
            ExprKind::If(c) => {
                let x = ast.if_(c);
                let cond = self.expr(x.cond);
                let bt = self.new_block();
                let be = self.new_block();
                self.term(
                    Term::Branch {
                        cond: cond.0,
                        then: bt,
                        else_: be,
                    },
                    span,
                );
                self.switch(bt);
                let tv = self.block(&x.then);
                let t_end = self.cur();
                self.switch(be);
                let ev = match &x.else_ {
                    Some(b) => self.block(b),
                    None => self.nil(span),
                };
                let e_end = self.cur();
                self.merge(&[(t_end, tv), (e_end, ev)], span)
            }
            ExprKind::Case(c) => {
                let x = ast.case(c);
                let subject = self.expr(x.subject);
                let mut results = Vec::new();
                for arm in &x.arms {
                    let body = self.new_block();
                    for &p in ast.get(arm.patterns) {
                        let pv = self.expr(p);
                        let t = self.tmp(RegTy::Bool);
                        self.emit(Instr::CaseTest(t, pv.0, subject.0), ast.span(p));
                        let next = self.new_block();
                        self.term(
                            Term::Branch {
                                cond: t,
                                then: body,
                                else_: next,
                            },
                            ast.span(p),
                        );
                        self.switch(next);
                    }
                    let after = self.cur();
                    self.switch(body);
                    let v = self.block(&arm.body);
                    results.push((self.cur(), v));
                    self.switch(after);
                }
                let v = match &x.else_ {
                    Some(b) => self.block(b),
                    None => self.nil(span),
                };
                results.push((self.cur(), v));
                self.merge(&results, span)
            }
            ExprKind::While(c) => {
                let x = ast.while_(c);
                let head = self.new_block();
                let body = self.new_block();
                let exit = self.new_block();
                self.term(Term::Jump(head), span);
                self.switch(head);
                let cond = self.expr(x.cond);
                self.term(
                    Term::Branch {
                        cond: cond.0,
                        then: body,
                        else_: exit,
                    },
                    span,
                );
                self.switch(body);
                self.f().loops.push((head, exit));
                let mark = self.temps_mark();
                self.block(&x.body);
                self.release(mark);
                self.f().loops.pop();
                self.term(Term::Jump(head), span);
                self.switch(exit);
                self.nil(span)
            }
            ExprKind::For(c) => self.for_loop(ast.for_(c), span),
            ExprKind::Begin(c) => self.begin_expr(c, span),
            ExprKind::Super(args) => {
                let Some((parent, name)) = self.f().sup else {
                    return self.fail(
                        "`super` can only be used in a method of a class that inherits",
                        span,
                    );
                };
                let this = self.self_value(span);
                let this = self.boxed(this, span);
                let site = self.call_site(args, None);
                let dst = self.tmp(RegTy::Value);
                self.emit(
                    Instr::CallSuper {
                        dst,
                        this,
                        parent,
                        name,
                        site,
                    },
                    span,
                );
                (dst, RegTy::Value)
            }
            ExprKind::Raise(v) => {
                let v = self.expr(v);
                let r = self.boxed(v, span);
                self.emit(Instr::Raise(r), span);
                let dead = self.new_block();
                self.switch(dead);
                self.nil(span)
            }
            ExprKind::Return(v) => {
                let (mut r, shape) = match v {
                    Some(v) => {
                        let r = self.expr(v);
                        (r, self.shape(v))
                    }
                    None => (self.nil(span), None),
                };
                if !self.f().ensures.is_empty() {
                    // The value is computed before the `ensure`s, which may change it.
                    let t = self.tmp(r.1);
                    self.emit(Instr::Move(t, r.0), span);
                    r = (t, r.1);
                    self.run_ensures(0);
                }
                self.term(Term::Return { reg: r.0, shape }, span);
                let dead = self.new_block();
                self.switch(dead);
                self.nil(span)
            }
            ExprKind::Break | ExprKind::Next => {
                let is_break = matches!(ast.kind(e), ExprKind::Break);
                match self.f().loops.last().copied() {
                    Some((cont, brk)) => {
                        let depth = self.f().loops.len();
                        self.run_ensures(depth);
                        self.term(Term::Jump(if is_break { brk } else { cont }), span);
                        let dead = self.new_block();
                        self.switch(dead);
                        self.nil(span)
                    }
                    None => self.fail(
                        format!(
                            "`{}` outside of a loop",
                            if is_break { "break" } else { "next" }
                        ),
                        span,
                    ),
                }
            }
        }
    }

    /// `begin`: the body in the region of the `rescue` handler, itself in the region of
    /// the `ensure` handler.
    ///
    /// ```text
    /// body (rescue → R) → else → ensure → after
    /// R: clause 1? clause 2? … otherwise reraise → ensure
    /// E (ensure handler): ensure; reraise
    /// ```
    fn begin_expr(&mut self, id: ControlId, span: Span) -> Op {
        let ast = self.ast;
        let x = ast.begin(id);
        let outer = self.f().handler;
        // Region of the `ensure`: the body, the `rescue` clauses and the `else`.
        let ensure = x.ensure.as_ref().map(|_| {
            let h = self.handler_block();
            let loops = self.f().loops.len();
            self.f().ensures.push(EnsureCtx {
                begin: id,
                loops,
                handler: outer,
            });
            self.f().handler = Some(h.0);
            h
        });
        let inner = self.f().handler;
        let rescue = (!x.rescues.is_empty()).then(|| self.handler_block());
        if let Some((h, _)) = rescue {
            self.f().handler = Some(h);
        }
        let body = self.new_block();
        self.term(Term::Jump(body), span);
        self.switch(body);
        let mut results = Vec::new();
        let mut v = self.block(&x.body);
        self.f().handler = inner;
        if let Some(b) = &x.else_ {
            // Outside the `rescue` region.
            let next = self.new_block();
            self.term(Term::Jump(next), span);
            self.switch(next);
            v = self.block(b);
        }
        results.push((self.cur(), v));
        if let Some((h, slot)) = rescue {
            let reg = self.f().handlers[h as usize].reg;
            let entry = self.f().handlers[h as usize].block;
            self.switch(entry);
            let mut caught = false;
            for r in &x.rescues {
                let next = r.ty.as_ref().map(|t| {
                    let ty = self.type_idx(t);
                    let c = self.tmp(RegTy::Bool);
                    self.emit(Instr::IsType(c, reg, ty), r.span);
                    let (then, next) = (self.new_block(), self.new_block());
                    self.term(
                        Term::Branch {
                            cond: c,
                            then,
                            else_: next,
                        },
                        r.span,
                    );
                    self.switch(then);
                    next
                });
                self.push_scope(false);
                if let Some((name, vspan)) = r.var {
                    let v = self.declare_local(name, vspan.start, None, true, vspan);
                    self.write(v, (reg, RegTy::Value), true, vspan);
                }
                let mark = self.temps_mark();
                let v = self.block_body(&r.body, false);
                self.release(mark);
                self.pop_scope();
                results.push((self.cur(), v));
                match next {
                    Some(n) => self.switch(n),
                    None => {
                        caught = true;
                        break;
                    }
                }
            }
            if !caught {
                self.term(Term::Reraise(slot), span);
            }
        }
        let v = self.merge(&results, span);
        let Some((h, slot)) = ensure else {
            return v;
        };
        self.f().ensures.pop();
        self.f().handler = outer;
        let after = self.new_block();
        self.term(Term::Jump(after), span);
        self.switch(after);
        let ens = x.ensure.as_ref().unwrap();
        let mark = self.temps_mark();
        self.block(ens);
        self.release(mark);
        let cont = self.cur();
        let entry = self.f().handlers[h as usize].block;
        self.switch(entry);
        self.block(ens);
        self.term(Term::Reraise(slot), span);
        self.switch(cont);
        v
    }

    /// Handler (block, slot, register) for the current region; its block is outside.
    fn handler_block(&mut self) -> (u32, u32) {
        let block = self.new_block();
        let reg = self.new_reg(RegTy::Value, false);
        let f = self.f();
        let slot = f.nslots;
        f.nslots += 1;
        f.handlers.push(Handler { block, reg, slot });
        ((f.handlers.len() - 1) as u32, slot)
    }

    /// Before a jump out of `begin`s: runs their `ensure`s, innermost first (those
    /// entered inside the `loops`-th loop, all of them for `return`).
    fn run_ensures(&mut self, loops: usize) {
        let saved_handler = self.f().handler;
        let saved_ensures = self.f().ensures.clone();
        let saved_loops = self.f().loops.clone();
        for (k, e) in saved_ensures.iter().enumerate().rev() {
            if e.loops < loops {
                break;
            }
            // The `ensure` runs outside its `begin` (and the jumps it contains as well).
            let f = self.f();
            f.handler = e.handler;
            f.ensures.truncate(k);
            f.loops.truncate(e.loops);
            let b = self.new_block();
            self.term(Term::Jump(b), Span::default());
            self.switch(b);
            let ens = self.ast.begin(e.begin).ensure.as_ref().unwrap();
            let mark = self.temps_mark();
            self.block(ens);
            self.release(mark);
        }
        let f = self.f();
        f.handler = saved_handler;
        f.ensures = saved_ensures;
        f.loops = saved_loops;
    }

    /// Gathers the values of the branches (`if`, `case`) into one register.
    fn merge(&mut self, branches: &[(BlockId, Op)], span: Span) -> Op {
        let blocks = &self.fns.last().unwrap().blocks;
        let live: Vec<&(BlockId, Op)> = branches
            .iter()
            .filter(|(b, _)| blocks[*b as usize].term.is_none())
            .collect();
        let ty = match live.first() {
            Some((_, (_, t))) if live.iter().all(|(_, (_, u))| u == t) => *t,
            _ => RegTy::Value,
        };
        let d = self.new_reg(ty, false);
        self.f().temps.push(d);
        let end = self.new_block();
        for (b, v) in branches {
            if self.f().blocks[*b as usize].term.is_some() {
                continue;
            }
            self.emit_in(*b, Instr::Move(d, v.0), span);
            self.term_in(*b, Term::Jump(end), span);
        }
        self.switch(end);
        (d, ty)
    }

    fn binary(&mut self, op: BinOp, a: Op, b: Op, span: Span) -> Op {
        use BinOp::*;
        let cmp = matches!(op, Eq | Ne | Lt | Le | Gt | Ge);
        let arith = matches!(op, Add | Sub | Mul | Div | Rem);
        match (a.1, b.1) {
            (RegTy::Int, RegTy::Int) if arith || cmp => {
                let ty = if cmp { RegTy::Bool } else { RegTy::Int };
                // Constant on the right, just loaded: immediate operand.
                let k = {
                    let f = self.f();
                    let blk = &mut f.blocks[f.cur as usize];
                    match blk.instrs.last() {
                        Some(&Instr::Int(r, k))
                            if r == b.0 && !f.is_var[r as usize] && blk.term.is_none() =>
                        {
                            blk.instrs.pop();
                            blk.spans.pop();
                            Some(k)
                        }
                        _ => None,
                    }
                };
                let d = self.tmp(ty);
                match k {
                    Some(k) => self.emit(Instr::IBinK(op, d, a.0, k), span),
                    None => self.emit(Instr::IBin(op, d, a.0, b.0), span),
                }
                (d, ty)
            }
            (RegTy::Int | RegTy::Float, RegTy::Int | RegTy::Float) if arith || cmp => {
                let fa = self.as_float(a, span);
                let fb = self.as_float(b, span);
                let ty = if cmp { RegTy::Bool } else { RegTy::Float };
                let d = self.tmp(ty);
                self.emit(Instr::FBin(op, d, fa, fb), span);
                (d, ty)
            }
            _ => {
                let ty = if cmp { RegTy::Bool } else { RegTy::Value };
                let d = self.tmp(ty);
                self.emit(Instr::Bin(op, d, a.0, b.0), span);
                (d, ty)
            }
        }
    }

    fn as_float(&mut self, a: Op, span: Span) -> Reg {
        if a.1 == RegTy::Float {
            return a.0;
        }
        let d = self.tmp(RegTy::Float);
        self.emit(Instr::IToF(d, a.0), span);
        d
    }

    fn self_value(&mut self, span: Span) -> Op {
        match self.resolve(SELF) {
            Some(_) => self.read_var(SELF, span),
            None => self.fail(
                "`self` and `@field` can only be used inside a method",
                span,
            ),
        }
    }

    /// Index of the `@name` field of `self`.
    fn this_field(&mut self, name: Symbol, span: Span) -> Option<u32> {
        let this = self.f().this.clone();
        match this {
            Some(fields) => match fields.iter().position(|f| *f == name) {
                Some(i) => Some(i as u32),
                None => {
                    let n = self.i.resolve(name).to_string();
                    self.fail(format!("no field `@{n}`"), span);
                    None
                }
            },
            None => {
                self.fail(
                    "`self` and `@field` can only be used inside a method",
                    span,
                );
                None
            }
        }
    }

    fn args_assign(&self, args: Slice<Arg>) -> bool {
        self.ast
            .get(args)
            .iter()
            .any(|a| self.assign[a.value.0 as usize])
    }

    fn call_site(&mut self, args: Slice<Arg>, block: Option<LambdaId>) -> u32 {
        let ast = self.ast;
        let list = ast.get(args).to_vec();
        let npos = list.iter().filter(|a| a.name.is_none()).count() as u32 + block.is_some() as u32;
        let named: Vec<Symbol> = list.iter().filter_map(|a| a.name).collect();
        let start = self.range(npos + named.len() as u32);
        let mut shapes = vec![None; (npos as usize) + named.len()];
        let (mut p, mut n) = (0u32, 0u32);
        for a in &list {
            let slot = match a.name {
                None => {
                    p += 1;
                    p - 1
                }
                Some(_) => {
                    n += 1;
                    npos + n - 1
                }
            };
            let v = self.expr(a.value);
            self.store_to(start + slot, v, ast.span(a.value));
            shapes[slot as usize] = self.shape(a.value);
        }
        if let Some(b) = block {
            let idx = self.lambda(b, ProtoKind::DoBlock);
            self.emit(Instr::Closure(start + npos - 1, idx), ast.lambda(b).span);
        }
        let f = self.f();
        f.calls.push(CallSite {
            start,
            npos,
            named,
            shapes,
            block: block.is_some(),
        });
        (f.calls.len() - 1) as u32
    }

    fn assign(&mut self, target: ExprId, op: Option<BinOp>, value: ExprId, span: Span) -> Op {
        let ast = self.ast;
        let tspan = ast.span(target);
        match ast.kind(target) {
            ExprKind::Ident(name) => {
                let mut v = self.expr(value);
                let mut shape = self.shape(value);
                if let Some(op) = op {
                    let old = self.read_var(name, tspan);
                    v = self.binary(op, old, v, span);
                    shape = None;
                }
                // Variable in a register: the value is stored there directly, and the
                // register is the value of the assignment.
                if let Some((var, origin)) = self.resolve(name)
                    && var.mutable
                    && let Store::Reg(r) = var.store
                    && !self.named_access(name, Some(origin), Some(var.store))
                {
                    self.store_to(r, v, tspan);
                    return (r, var.ty);
                }
                // Copy: the value of the assignment must not change if the variable changes.
                let result = if self.f().is_var[v.0 as usize] {
                    let t = self.tmp(v.1);
                    self.emit(Instr::Move(t, v.0), span);
                    (t, v.1)
                } else {
                    v
                };
                self.write_var(name, result, shape, tspan);
                result
            }
            ExprKind::IVar(name) => {
                let Some(idx) = self.this_field(name, tspan) else {
                    return (self.tmp(RegTy::Value), RegTy::Value);
                };
                let obj = self.self_value(tspan);
                let mut v = self.expr(value);
                let mut shape = self.shape(value);
                if let Some(op) = op {
                    let old = self.tmp(RegTy::Value);
                    self.emit(Instr::FieldAt(old, obj.0, idx), tspan);
                    v = self.binary(op, (old, RegTy::Value), v, span);
                    shape = None;
                }
                self.emit(
                    Instr::SetFieldAt {
                        obj: obj.0,
                        idx,
                        src: v.0,
                        shape,
                    },
                    tspan,
                );
                v
            }
            ExprKind::Field(recv, name) => {
                let obj = self.expr(recv);
                let mut v = self.expr(value);
                let mut shape = self.shape(value);
                if let Some(op) = op {
                    let old = self.tmp(RegTy::Value);
                    self.emit(Instr::Field(old, obj.0, name), tspan);
                    v = self.binary(op, (old, RegTy::Value), v, span);
                    shape = None;
                }
                self.emit(
                    Instr::SetField {
                        obj: obj.0,
                        name,
                        src: v.0,
                        shape,
                    },
                    tspan,
                );
                v
            }
            ExprKind::Index(recv, idx) => {
                let ops = self.operands(&[recv, idx, value]);
                let mut v = ops[2];
                if let Some(op) = op {
                    let old = self.tmp(RegTy::Value);
                    self.emit(Instr::Index(old, ops[0].0, ops[1].0), tspan);
                    v = self.binary(op, (old, RegTy::Value), v, span);
                }
                self.emit(Instr::SetIndex(ops[0].0, ops[1].0, v.0), tspan);
                v
            }
            _ => self.fail("cannot assign to this expression", tspan),
        }
    }

    fn for_loop(&mut self, x: &For, span: Span) -> Op {
        let ast = self.ast;
        let exit = self.new_block();
        self.push_scope(false);
        // `for i in a..b`: integer counter, no iterator.
        if let (ExprKind::Range { lo, hi, inclusive }, [(name, vspan)]) =
            (ast.kind(x.iter), x.vars.as_slice())
        {
            let ops = self.operands(&[lo, hi]);
            let k = self.var_reg(RegTy::Int);
            self.store_to(k, ops[0], ast.span(lo));
            let end = self.var_reg(RegTy::Int);
            self.store_to(end, ops[1], ast.span(hi));
            let head = self.new_block();
            let body = self.new_block();
            let step = self.new_block();
            self.term(Term::Jump(head), span);
            self.switch(head);
            let c = self.tmp(RegTy::Bool);
            let cmp = if inclusive { BinOp::Le } else { BinOp::Lt };
            self.emit(Instr::IBin(cmp, c, k, end), span);
            self.term(
                Term::Branch {
                    cond: c,
                    then: body,
                    else_: exit,
                },
                span,
            );
            self.switch(body);
            let prim = self.var_prim(vspan.start, None).or(Some(RegTy::Int));
            if !self.captured.contains(&vspan.start)
                && !assigns(ast, &x.body, *name)
                && prim == Some(RegTy::Int)
            {
                // The body does not modify the variable: it is the counter itself.
                self.f().scopes.last_mut().unwrap().vars.push(Var {
                    name: *name,
                    store: Store::Reg(k),
                    ty: RegTy::Int,
                    mutable: true,
                    guard: Some(RegTy::Int),
                });
            } else {
                let v = self.declare_local(*name, vspan.start, prim, true, *vspan);
                self.write(v, (k, RegTy::Int), true, *vspan);
            }
            self.f().loops.push((step, exit));
            let mark = self.temps_mark();
            self.block_body(&x.body, false);
            self.release(mark);
            self.f().loops.pop();
            self.term(Term::Jump(step), span);
            self.switch(step);
            // Inclusive range: exit after the bound without incrementing it (no
            // overflow for `a..i64::MAX`).
            if inclusive {
                let last = self.tmp(RegTy::Bool);
                self.emit(Instr::IBin(BinOp::Eq, last, k, end), span);
                let inc = self.new_block();
                self.term(
                    Term::Branch {
                        cond: last,
                        then: exit,
                        else_: inc,
                    },
                    span,
                );
                self.switch(inc);
            }
            self.emit(Instr::IBinK(BinOp::Add, k, k, 1), span);
            self.term(Term::Jump(head), span);
            self.pop_scope();
            self.switch(exit);
            return self.nil(span);
        }
        let src = self.expr(x.iter);
        let it = self.var_reg(RegTy::Value);
        self.emit(
            Instr::IterNew {
                dst: it,
                src: src.0,
                keys: x.vars.len() == 1,
            },
            ast.span(x.iter),
        );
        let item = self.var_reg(RegTy::Value);
        let head = self.new_block();
        let body = self.new_block();
        self.term(Term::Jump(head), span);
        self.switch(head);
        self.term(
            Term::IterNext {
                iter: it,
                dst: item,
                body,
                done: exit,
            },
            ast.span(x.iter),
        );
        self.switch(body);
        match x.vars.as_slice() {
            [(name, vspan)] => {
                let prim = self.var_prim(vspan.start, None);
                let v = self.declare_local(*name, vspan.start, prim, true, *vspan);
                self.write(v, (item, RegTy::Value), true, *vspan);
            }
            vars => {
                let n = vars.len() as u32;
                let dst = self.range(n);
                self.emit(
                    Instr::Unpack {
                        dst,
                        src: item,
                        n,
                        rest: false,
                    },
                    ast.span(x.iter),
                );
                for (k, (name, vspan)) in vars.iter().enumerate() {
                    let prim = self.var_prim(vspan.start, None);
                    let v = self.declare_local(*name, vspan.start, prim, true, *vspan);
                    self.write(v, (dst + k as u32, RegTy::Value), true, *vspan);
                }
            }
        }
        self.f().loops.push((head, exit));
        let mark = self.temps_mark();
        self.block_body(&x.body, false);
        self.release(mark);
        self.f().loops.pop();
        self.term(Term::Jump(head), span);
        self.pop_scope();
        self.switch(exit);
        self.nil(span)
    }
}

fn tail(b: &Block) -> Option<ExprId> {
    match b.stmts.last()?.kind {
        StmtKind::Expr(e) => Some(e),
        _ => None,
    }
}

fn prim_name<'a>(t: &TypeExpr, i: &'a Interner) -> Option<&'a str> {
    match &t.kind {
        TypeKind::Named(p) if p.len() == 1 => Some(i.resolve(p[0])),
        _ => None,
    }
}

/// Register type of a simple annotation (`int`, `float`, `bool`).
fn prim_of(t: &TypeExpr, i: &Interner) -> Option<RegTy> {
    match prim_name(t, i)? {
        "int" => Some(RegTy::Int),
        "float" => Some(RegTy::Float),
        "bool" => Some(RegTy::Bool),
        _ => None,
    }
}

/// Register written by a single-destination instruction (for `store_to`).
fn dest_mut(ins: &mut Instr) -> Option<&mut Reg> {
    use Instr::*;
    Some(match ins {
        Int(d, _) | Float(d, _) | Bool(d, _) | Nil(d) | Const(d, _) | Move(d, _) => d,
        GetGlobal(d, _) | GetCell(d, _) | GetUp(d, _) | CtxField(d, _) => d,
        GetNamed { dst, .. } | Coerce { dst, .. } | Range { dst, .. } => dst,
        IsUndef(d, _)
        | IBin(_, d, _, _)
        | IBinK(_, d, _, _)
        | FBin(_, d, _, _)
        | INeg(d, _)
        | FNeg(d, _)
        | IToF(d, _) => d,
        Bin(_, d, _, _) | Neg(d, _) | Not(d, _) => d,
        List(d, _, _) | Tuple(d, _, _) | Hash(d, _, _) | Record(d, _, _) | Concat(d, _, _) => d,
        Index(d, _, _)
        | IndexOpt(d, _, _)
        | Field(d, _, _)
        | FieldAs(d, _, _, _)
        | FieldAt(d, _, _)
        | TupleIndex(d, _, _)
        | Variant(d, _, _) => d,
        Closure(d, _) | Call { dst: d, .. } | CallMethod { dst: d, .. } => d,
        _ => return None,
    })
}

/// Does the block (or its lambdas) assign the variable `name`?
fn assigns(ast: &Ast, b: &Block, name: Symbol) -> bool {
    let mut found = false;
    visit_block(ast, b, &mut |e| {
        if let ExprKind::Assign { target, .. } = ast.kind(e)
            && matches!(ast.kind(target), ExprKind::Ident(n) if n == name)
        {
            found = true;
        }
    });
    found
}

/// Calls `f` on each expression of a block, sub-expressions and lambdas included.
fn visit_block(ast: &Ast, b: &Block, f: &mut dyn FnMut(ExprId)) {
    for s in &b.stmts {
        match &s.kind {
            StmtKind::Expr(e) => visit(ast, *e, f),
            StmtKind::Module(m) => visit_block(ast, &m.body, f),
            StmtKind::Require(_) => {}
            StmtKind::Let(l) => visit(ast, l.value, f),
            StmtKind::Def(d) => {
                for p in d.params.iter().flatten() {
                    if let Some(x) = p.default {
                        visit(ast, x, f);
                    }
                }
                visit_block(ast, &d.body, f);
            }
            StmtKind::Type(_) | StmtKind::Enum(_) => {}
        }
    }
}

fn visit(ast: &Ast, e: ExprId, f: &mut dyn FnMut(ExprId)) {
    f(e);
    let lambda = |id: LambdaId, f: &mut dyn FnMut(ExprId)| {
        let l = ast.lambda(id);
        for p in &l.params {
            if let Some(x) = p.default {
                visit(ast, x, f);
            }
        }
        visit_block(ast, &l.body, f);
    };
    match ast.kind(e) {
        ExprKind::Interp(parts) => {
            for p in ast.get(parts) {
                if let StrPart::Expr(x) = *p {
                    visit(ast, x, f);
                }
            }
        }
        ExprKind::List(items) | ExprKind::Tuple(items) => {
            for &x in ast.get(items) {
                visit(ast, x, f);
            }
        }
        ExprKind::Hash(entries) => {
            for &(k, v) in ast.get(entries) {
                visit(ast, k, f);
                visit(ast, v, f);
            }
        }
        ExprKind::Record(fields) => {
            for &(_, v) in ast.get(fields) {
                visit(ast, v, f);
            }
        }
        ExprKind::Unary(_, x) | ExprKind::Field(x, _) | ExprKind::TupleIndex(x, _) => {
            visit(ast, x, f)
        }
        ExprKind::Binary(_, x, y) | ExprKind::Index(x, y) | ExprKind::IndexOpt(x, y) => {
            visit(ast, x, f);
            visit(ast, y, f);
        }
        ExprKind::Range { lo, hi, .. } => {
            visit(ast, lo, f);
            visit(ast, hi, f);
        }
        ExprKind::Assign { target, value, .. } => {
            visit(ast, target, f);
            visit(ast, value, f);
        }
        ExprKind::Call {
            callee: x,
            args,
            block,
        }
        | ExprKind::MethodCall {
            recv: x,
            args,
            block,
            ..
        } => {
            visit(ast, x, f);
            for a in ast.get(args) {
                visit(ast, a.value, f);
            }
            if let Some(b) = block {
                lambda(b, f);
            }
        }
        ExprKind::Lambda(id) => lambda(id, f),
        ExprKind::If(c) => {
            let x = ast.if_(c);
            visit(ast, x.cond, f);
            visit_block(ast, &x.then, f);
            if let Some(b) = &x.else_ {
                visit_block(ast, b, f);
            }
        }
        ExprKind::Case(c) => {
            let x = ast.case(c);
            visit(ast, x.subject, f);
            for arm in &x.arms {
                for &p in ast.get(arm.patterns) {
                    visit(ast, p, f);
                }
                visit_block(ast, &arm.body, f);
            }
            if let Some(b) = &x.else_ {
                visit_block(ast, b, f);
            }
        }
        ExprKind::While(c) => {
            let x = ast.while_(c);
            visit(ast, x.cond, f);
            visit_block(ast, &x.body, f);
        }
        ExprKind::For(c) => {
            let x = ast.for_(c);
            visit(ast, x.iter, f);
            visit_block(ast, &x.body, f);
        }
        ExprKind::Begin(c) => {
            let x = ast.begin(c);
            visit_block(ast, &x.body, f);
            for r in &x.rescues {
                visit_block(ast, &r.body, f);
            }
            for b in x.else_.iter().chain(&x.ensure) {
                visit_block(ast, b, f);
            }
        }
        ExprKind::Return(Some(x)) | ExprKind::Raise(x) => visit(ast, x, f),
        ExprKind::Super(args) => {
            for a in ast.get(args) {
                visit(ast, a.value, f);
            }
        }
        _ => {}
    }
}

fn is_cmp(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
    )
}

/// Jumps to an empty block that only jumps go straight to its target.
fn thread_jumps(blocks: &mut [BB], handlers: &mut [Handler]) {
    let n = blocks.len();
    let target = |blocks: &[BB], mut b: BlockId| {
        for _ in 0..n {
            match &blocks[b as usize] {
                BB {
                    instrs,
                    term: Some((Term::Jump(t), _)),
                    ..
                } if instrs.is_empty() && *t != b => b = *t,
                _ => break,
            }
        }
        b
    };
    let finals: Vec<BlockId> = (0..n as BlockId).map(|b| target(blocks, b)).collect();
    for b in blocks.iter_mut() {
        if let Some((t, _)) = &mut b.term {
            for x in t.targets_mut() {
                *x = finals[*x as usize];
            }
        }
    }
    for h in handlers {
        h.block = finals[h.block as usize];
    }
}

/// A block that jumps unconditionally to a block whose only predecessor it is absorbs
/// that block: fewer jumps to run. The absorbed block becomes unreachable.
/// Only within the same handler: the absorbed instructions keep theirs.
fn merge_blocks(blocks: &mut [BB], handlers: &[Handler]) {
    loop {
        let mut preds = vec![0u32; blocks.len()];
        preds[0] = 1; // entry point
        for b in blocks.iter_mut() {
            if let Some((t, _)) = &mut b.term {
                for x in t.targets_mut() {
                    preds[*x as usize] += 1;
                }
            }
        }
        for h in handlers {
            preds[h.block as usize] += 1;
        }
        let found = (0..blocks.len()).find_map(|a| match &blocks[a].term {
            Some((Term::Jump(t), _))
                if *t as usize != a
                    && *t != 0
                    && preds[*t as usize] == 1
                    && blocks[*t as usize].handler == blocks[a].handler =>
            {
                Some((a, *t as usize))
            }
            _ => None,
        });
        let Some((a, t)) = found else { return };
        let BB {
            instrs,
            spans,
            term,
            handler,
        } = std::mem::replace(
            &mut blocks[t],
            BB {
                instrs: Vec::new(),
                spans: Vec::new(),
                // Unreachable: it jumps to itself and is never reached again.
                term: Some((Term::Jump(t as BlockId), Span::default())),
                handler: None,
            },
        );
        debug_assert_eq!(handler, blocks[a].handler);
        blocks[a].instrs.extend(instrs);
        blocks[a].spans.extend(spans);
        blocks[a].term = term;
    }
}

/// Keeps only the blocks reachable from the entry, renumbered in order (handlers of
/// unreachable blocks are kept, pointing to a block that stays).
fn compact(blocks: &mut Vec<BB>, handlers: &mut [Handler]) -> Vec<BB> {
    let n = blocks.len();
    let mut seen = vec![false; n];
    let mut todo = vec![0 as BlockId];
    while let Some(b) = todo.pop() {
        if std::mem::replace(&mut seen[b as usize], true) {
            continue;
        }
        let block = &mut blocks[b as usize];
        if let Some(h) = block.handler {
            todo.push(handlers[h as usize].block);
        }
        if let Some((t, _)) = &mut block.term {
            todo.extend(t.targets_mut().into_iter().map(|x| *x));
        }
    }
    let mut index = vec![0 as BlockId; n];
    let mut next = 0;
    for (b, keep) in seen.iter().enumerate() {
        if *keep {
            index[b] = next;
            next += 1;
        }
    }
    for h in handlers.iter_mut() {
        h.block = if seen[h.block as usize] {
            index[h.block as usize]
        } else {
            0
        };
    }
    let mut out = Vec::with_capacity(next as usize);
    for (b, block) in std::mem::take(blocks).into_iter().enumerate() {
        if seen[b] {
            let mut block = block;
            if let Some((t, _)) = &mut block.term {
                for x in t.targets_mut() {
                    *x = index[*x as usize];
                }
            }
            out.push(block);
        }
    }
    out
}
