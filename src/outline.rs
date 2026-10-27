//! Loops of the main program outlined into functions, for the typed JIT.
//!
//! At the top level, variables are globals and the main program is not compiled by the
//! typed JIT (it declares types, functions…). A loop of the main program whose registers
//! live only in the loop becomes a hidden function (`Proto::outlined`): the registers
//! it reads from before the loop are its parameters, the main program calls it, then goes
//! on after the loop. The function is compiled like any other, typed for its arguments,
//! the globals it reads typed by their values (see `jit.rs`).
//!
//! Not outlined: a loop that `return`s, raises again (`ensure`), is in a `begin … rescue`,
//! uses cells (variables captured by a lambda) or the context of a `do` block, or leaves
//! a register that is read after it.

use crate::intern::Symbol;
use crate::ir::{Block, BlockId, CallSite, Instr, ParamProto, Proto, Reg, RegTy, Term};
use std::rc::Rc;

/// Registers read and written by `ins` (all of them), and whether it can be outlined.
pub(crate) fn regs_of(p: &Proto, ins: &Instr) -> (Vec<Reg>, Vec<Reg>, bool) {
    use Instr::*;
    let range = |s: Reg, n: u32| (s..s + n).collect::<Vec<_>>();
    let site_args = |site: u32| {
        let s = &p.calls[site as usize];
        range(s.start, s.npos + s.named.len() as u32)
    };
    let (reads, writes): (Vec<Reg>, Vec<Reg>) = match ins {
        Int(d, _) | Float(d, _) | Bool(d, _) | Nil(d) | Const(d, _) | GetGlobal(d, _) => {
            (vec![], vec![*d])
        }
        Move(d, a) | IsType(d, a, _) | Guard(d, a, _) | IsUndef(d, a) => (vec![*a], vec![*d]),
        Coerce { dst, src, .. } => (vec![*src], vec![*dst]),
        Fail(_) => (vec![], vec![]),
        Raise(a) | SetGlobal(_, a) => (vec![*a], vec![]),
        DefGlobal { src, .. } => (vec![*src], vec![]),
        IBin(_, d, a, b) | FBin(_, d, a, b) | Bin(_, d, a, b) => (vec![*a, *b], vec![*d]),
        IBinK(_, d, a, _) | INeg(d, a) | FNeg(d, a) | IToF(d, a) | Neg(d, a) | Not(d, a) => {
            (vec![*a], vec![*d])
        }
        List(d, s, n) | Tuple(d, s, n) | Concat(d, s, n) => (range(*s, *n), vec![*d]),
        Hash(d, s, n) => (range(*s, 2 * n), vec![*d]),
        Record(d, s, r) => (range(*s, p.records[*r as usize].len() as u32), vec![*d]),
        Range { dst, lo, hi, .. } => (vec![*lo, *hi], vec![*dst]),
        Index(d, a, b) | IndexOpt(d, a, b) | CaseTest(d, a, b) => (vec![*a, *b], vec![*d]),
        SetIndex(a, b, c) => (vec![*a, *b, *c], vec![]),
        Field(d, a, ..) | FieldAs(d, a, ..) | FieldAt(d, a, _) | TupleIndex(d, a, _) => {
            (vec![*a], vec![*d])
        }
        Variant(d, a, _) => (vec![*a], vec![*d]),
        SetField { obj, src, .. } | SetFieldAt { obj, src, .. } => (vec![*obj, *src], vec![]),
        Unpack { dst, src, n, rest } => (vec![*src], range(*dst, n + *rest as u32)),
        // A lambda that captures reads the program's cells.
        Closure(d, c) if p.children[*c as usize].captures.is_empty() => (vec![], vec![*d]),
        Call { dst, f, site } => {
            let mut r = site_args(*site);
            r.push(*f);
            (r, vec![*dst])
        }
        CallSuper {
            dst, this, site, ..
        } => {
            let mut r = site_args(*site);
            r.push(*this);
            (r, vec![*dst])
        }
        CallMethod {
            dst, recv, site, ..
        } => {
            let mut r = site_args(*site);
            r.push(*recv);
            (r, vec![*dst])
        }
        IterNew { dst, src, .. } => (vec![*src], vec![*dst]),
        // Cells, `do` block context, declarations: stay in the main program.
        GetCell(d, _) | GetUp(d, _) | CtxField(d, _) | GetNamed { dst: d, .. } | Closure(d, _) => {
            return (vec![], vec![*d], false);
        }
        SetCell(_, a) | SetUp(_, a) | SetNamed { src: a, .. } => return (vec![*a], vec![], false),
        NewCell(_) | DefType { .. } => return (vec![], vec![], false),
        // Reads the explicit values that follow `start`: all of them, to be safe.
        DefEnum { start, .. } => {
            return ((*start..p.regs.len() as Reg).collect(), vec![], false);
        }
    };
    (reads, writes, true)
}

fn successors(t: &Term) -> Vec<BlockId> {
    match t {
        Term::Jump(b) => vec![*b],
        Term::Branch { then, else_, .. }
        | Term::IfInt { then, else_, .. }
        | Term::IfIntK { then, else_, .. } => vec![*then, *else_],
        Term::IterNext { body, done, .. } => vec![*body, *done],
        Term::Return { .. } | Term::Reraise(_) => vec![],
    }
}

fn term_reads(t: &Term) -> Vec<Reg> {
    match t {
        Term::Branch { cond, .. } => vec![*cond],
        Term::IfInt { a, b, .. } => vec![*a, *b],
        Term::IfIntK { a, .. } => vec![*a],
        Term::Return { reg, .. } => vec![*reg],
        Term::IterNext { iter, .. } => vec![*iter],
        Term::Jump(_) | Term::Reraise(_) => vec![],
    }
}

/// Registers live at the start of each block.
fn liveness(p: &Proto) -> Vec<Vec<bool>> {
    let n = p.regs.len();
    let nb = p.blocks.len();
    let mut live_in = vec![vec![false; n]; nb];
    loop {
        let mut changed = false;
        for b in (0..nb).rev() {
            let blk = &p.blocks[b];
            let mut live = vec![false; n];
            for s in successors(&blk.term) {
                for (r, l) in live_in[s as usize].iter().enumerate() {
                    live[r] |= *l;
                }
            }
            // A handler receives the error in its register: what its block reads is live
            // anywhere in the protected blocks.
            if let Some(h) = blk.handler {
                for (r, l) in live_in[p.handlers[h as usize].block as usize]
                    .iter()
                    .enumerate()
                {
                    live[r] |= *l;
                }
            }
            if let Term::IterNext { dst, .. } = blk.term {
                live[dst as usize] = false;
            }
            for r in term_reads(&blk.term) {
                live[r as usize] = true;
            }
            for ins in blk.instrs.iter().rev() {
                let (reads, writes, _) = regs_of(p, ins);
                for w in writes {
                    if let Some(l) = live.get_mut(w as usize) {
                        *l = false;
                    }
                }
                for r in reads {
                    if let Some(l) = live.get_mut(r as usize) {
                        *l = true;
                    }
                }
            }
            if live != live_in[b] {
                live_in[b] = live;
                changed = true;
            }
        }
        if !changed {
            return live_in;
        }
    }
}

/// Strongly connected components of the block graph (Tarjan), those that are loops.
pub(crate) fn loops(p: &Proto) -> Vec<Vec<BlockId>> {
    struct T<'a> {
        succ: &'a [Vec<BlockId>],
        index: Vec<Option<u32>>,
        low: Vec<u32>,
        on: Vec<bool>,
        stack: Vec<BlockId>,
        next: u32,
        out: Vec<Vec<BlockId>>,
    }
    fn visit(t: &mut T, v: BlockId) {
        let vi = v as usize;
        t.index[vi] = Some(t.next);
        t.low[vi] = t.next;
        t.next += 1;
        t.stack.push(v);
        t.on[vi] = true;
        for &w in &t.succ[vi] {
            let wi = w as usize;
            match t.index[wi] {
                None => {
                    visit(t, w);
                    t.low[vi] = t.low[vi].min(t.low[wi]);
                }
                Some(i) if t.on[wi] => t.low[vi] = t.low[vi].min(i),
                _ => {}
            }
        }
        if Some(t.low[vi]) == t.index[vi] {
            let mut scc = Vec::new();
            loop {
                let w = t.stack.pop().unwrap();
                t.on[w as usize] = false;
                scc.push(w);
                if w == v {
                    break;
                }
            }
            let is_loop = scc.len() > 1 || t.succ[vi].contains(&v);
            if is_loop {
                t.out.push(scc);
            }
        }
    }
    let succ: Vec<Vec<BlockId>> = p.blocks.iter().map(|b| successors(&b.term)).collect();
    let nb = p.blocks.len();
    let mut t = T {
        succ: &succ,
        index: vec![None; nb],
        low: vec![0; nb],
        on: vec![false; nb],
        stack: Vec::new(),
        next: 0,
        out: Vec::new(),
    };
    // Iterative depth would be safer, but the main program's blocks are few enough
    // (one per branch): recursion depth = length of the longest path.
    if nb > 20_000 {
        return Vec::new();
    }
    visit(&mut t, 0);
    t.out
}

/// A loop to outline: its blocks, header, pre-header, exit, the `iterator` instruction
/// moved from the pre-header, and the registers it reads from before.
struct Region {
    blocks: Vec<bool>,
    header: BlockId,
    pre: BlockId,
    exit: BlockId,
    moved: Option<(Instr, crate::span::Span)>,
    params: Vec<Reg>,
}

fn region(p: &Proto, scc: &[BlockId], live_in: &[Vec<bool>], in_loop: &[bool]) -> Option<Region> {
    let nb = p.blocks.len();
    let mut inside = vec![false; nb];
    for &b in scc {
        inside[b as usize] = true;
    }
    // Single entry (a `jump` from outside) and single exit.
    let mut header = None;
    let mut pre = None;
    let mut exit = None;
    for (b, blk) in p.blocks.iter().enumerate() {
        for s in successors(&blk.term) {
            match (inside[b], inside[s as usize]) {
                (false, true) => {
                    if pre.is_some() || !matches!(blk.term, Term::Jump(_)) {
                        return None;
                    }
                    pre = Some(b as BlockId);
                    header = Some(s);
                }
                (true, false) => {
                    if exit.is_some_and(|x| x != s) {
                        return None;
                    }
                    exit = Some(s);
                }
                _ => {}
            }
        }
    }
    let (header, pre, exit) = (header?, pre?, exit?);
    if inside[0] || in_loop[pre as usize] {
        return None;
    }
    let mut written = vec![false; p.regs.len()];
    for &b in scc {
        let blk = &p.blocks[b as usize];
        if blk.handler.is_some() || matches!(blk.term, Term::Return { .. } | Term::Reraise(_)) {
            return None;
        }
        for ins in &blk.instrs {
            let (_, writes, ok) = regs_of(p, ins);
            if !ok {
                return None;
            }
            for w in writes {
                written[w as usize] = true;
            }
        }
        if let Term::IterNext { dst, .. } = blk.term {
            written[dst as usize] = true;
        }
    }
    // The `for`'s iterator, created just before the loop: created in the function.
    let pb = &p.blocks[pre as usize];
    let moved = match pb.instrs.last() {
        Some(ins @ Instr::IterNew { dst, .. }) if live_in[header as usize][*dst as usize] => {
            written[*dst as usize] = true;
            Some((ins.clone(), *pb.spans.last().unwrap()))
        }
        _ => None,
    };
    let mut params: Vec<Reg> = (0..p.regs.len() as Reg)
        .filter(|&r| live_in[header as usize][r as usize])
        .collect();
    if let Some((Instr::IterNew { dst, src, .. }, _)) = &moved {
        params.retain(|r| r != dst);
        if !params.contains(src) {
            params.push(*src);
            params.sort_unstable();
        }
    }
    // What the loop writes must not be read after it.
    if live_in[exit as usize]
        .iter()
        .enumerate()
        .any(|(r, l)| *l && written[r])
    {
        return None;
    }
    Some(Region {
        blocks: inside,
        header,
        pre,
        exit,
        moved,
        params,
    })
}

/// Outlines the loops of the main program `p` (see the module documentation).
/// `name`: that of the parameters.
pub fn outline_loops(p: &mut Proto, name: Symbol) {
    if !p.has_loop {
        return;
    }
    let sccs = loops(p);
    if sccs.is_empty() {
        return;
    }
    let live_in = liveness(p);
    let mut in_loop = vec![false; p.blocks.len()];
    for scc in &sccs {
        for &b in scc {
            in_loop[b as usize] = true;
        }
    }
    let regions: Vec<Region> = sccs
        .iter()
        .filter_map(|scc| region(p, scc, &live_in, &in_loop))
        .collect();
    for r in regions {
        let child = outlined(p, &r, name);
        let idx = p.children.len() as u32;
        p.children.push(Rc::new(child));
        let new_reg = |p: &mut Proto| {
            p.regs.push(RegTy::Value);
            (p.regs.len() - 1) as Reg
        };
        let f = new_reg(p);
        let dst = new_reg(p);
        let start = p.regs.len() as Reg;
        for _ in &r.params {
            new_reg(p);
        }
        let site = p.calls.len() as u32;
        p.calls.push(CallSite {
            start,
            npos: r.params.len() as u32,
            named: Vec::new(),
            shapes: vec![None; r.params.len()],
            block: false,
        });
        let pb = &mut p.blocks[r.pre as usize];
        if r.moved.is_some() {
            pb.instrs.pop();
            pb.spans.pop();
        }
        let span = pb.term_span;
        pb.instrs.push(Instr::Closure(f, idx));
        pb.spans.push(span);
        for (k, &a) in r.params.iter().enumerate() {
            pb.instrs.push(Instr::Move(start + k as Reg, a));
            pb.spans.push(span);
        }
        pb.instrs.push(Instr::Call { dst, f, site });
        pb.spans.push(span);
        pb.term = Term::Jump(r.exit);
    }
}

/// The function of loop `r`: the loop's blocks, an entry block that jumps to it, the
/// others replaced by `return nil` (its exit; unreachable).
fn outlined(p: &Proto, r: &Region, name: Symbol) -> Proto {
    let mut regs = p.regs.clone();
    regs.push(RegTy::Value);
    let nil = (regs.len() - 1) as Reg;
    let pre = &p.blocks[r.pre as usize];
    let blocks = p
        .blocks
        .iter()
        .enumerate()
        .map(|(b, blk)| {
            if r.blocks[b] {
                blk.clone()
            } else if b == 0 {
                let (instrs, spans) = match &r.moved {
                    Some((ins, span)) => (vec![ins.clone()], vec![*span]),
                    None => (vec![], vec![]),
                };
                Block {
                    instrs,
                    spans,
                    term: Term::Jump(r.header),
                    term_span: pre.term_span,
                    handler: None,
                }
            } else {
                Block {
                    instrs: vec![Instr::Nil(nil)],
                    spans: vec![blk.term_span],
                    term: Term::Return {
                        reg: nil,
                        shape: None,
                    },
                    term_span: blk.term_span,
                    handler: None,
                }
            }
        })
        .collect();
    Proto {
        name: p.name.clone(),
        script: p.script.clone(),
        kind: crate::ir::ProtoKind::Function,
        params: r
            .params
            .iter()
            .map(|&reg| ParamProto {
                name,
                reg,
                ty: None,
                has_default: false,
                field: None,
                span: pre.term_span,
                annotations: Rc::from([]),
            })
            .collect(),
        ret: None,
        regs,
        ncells: 0,
        captures: Vec::new(),
        blocks,
        consts: p.consts.clone(),
        calls: p.calls.clone(),
        shapes: p.shapes.clone(),
        types: p.types.clone(),
        records: p.records.clone(),
        children: p.children.clone(),
        decls: p.decls.clone(),
        decl: None,
        vis: p.vis,
        jit: Default::default(),
        jit_dyn: Default::default(),
        calls_seen: Default::default(),
        has_loop: true,
        plain_params: true,
        annotations: Rc::from([]),
        handlers: Vec::new(),
        nslots: 0,
        specs: Default::default(),
        class_params: Vec::new(),
        spec: None,
        deopted: Default::default(),
        outlined: true,
    }
}
