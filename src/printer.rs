//! Prints the AST as S-expressions: `(+ 1 (* 2 3))`.
//!
//! Used by the parser tests and by `bokoni --ast`.

use crate::ast::*;
use crate::intern::{Interner, Symbol};
use std::fmt::Write;

pub fn program(p: &Program, i: &Interner) -> String {
    let mut out = Printer {
        ast: &p.ast,
        i,
        out: String::new(),
    };
    for s in &p.body.stmts {
        out.stmt(s);
        out.out.push('\n');
    }
    out.out
}

pub fn expr(ast: &Ast, e: ExprId, i: &Interner) -> String {
    let mut p = Printer {
        ast,
        i,
        out: String::new(),
    };
    p.expr(e);
    p.out
}

pub fn type_expr(t: &TypeExpr, i: &Interner) -> String {
    let ast = Ast::default();
    let mut p = Printer {
        ast: &ast,
        i,
        out: String::new(),
    };
    p.ty(t);
    p.out
}

struct Printer<'a> {
    ast: &'a Ast,
    i: &'a Interner,
    out: String,
}

impl Printer<'_> {
    fn name(&mut self, s: Symbol) {
        self.out.push_str(self.i.resolve(s));
    }

    fn path(&mut self, path: &[Symbol]) {
        for (n, s) in path.iter().enumerate() {
            if n > 0 {
                self.out.push_str("::");
            }
            self.name(*s);
        }
    }

    fn quoted(&mut self, s: Symbol) {
        let _ = write!(self.out, "{:?}", self.i.resolve(s));
    }

    fn block(&mut self, b: &Block) {
        for s in &b.stmts {
            self.out.push(' ');
            self.stmt(s);
        }
    }

    fn annotations(&mut self, anns: &[AnnotationUse]) {
        for a in anns {
            self.out.push_str("@[");
            self.path(&a.path);
            for arg in self.ast.get(a.args) {
                self.out.push(' ');
                self.arg(arg);
            }
            self.out.push_str("] ");
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Expr(e) => self.expr(*e),
            StmtKind::Let(l) => {
                self.out.push_str("(let ");
                self.pattern(&l.pattern);
                if let Some(t) = &l.ty {
                    self.out.push_str(" : ");
                    self.ty(t);
                }
                self.out.push(' ');
                self.expr(l.value);
                self.out.push(')');
            }
            StmtKind::Def(f) => self.def(f),
            StmtKind::Type(t) => {
                self.annotations(&t.annotations);
                self.out.push('(');
                self.out.push_str(match t.kind {
                    TypeDeclKind::Type => "type",
                    TypeDeclKind::Class => "class",
                    TypeDeclKind::Annotation => "annotation",
                });
                self.out.push(' ');
                self.name(t.name);
                if let Some((p, _)) = t.parent {
                    self.out.push_str(" < ");
                    self.name(p);
                }
                for m in &t.members {
                    self.out.push(' ');
                    match m {
                        Member::Field(f) => {
                            self.annotations(&f.annotations);
                            self.out.push_str("(field ");
                            self.name(f.name);
                            if let Some(t) = &f.ty {
                                self.out.push_str(" : ");
                                self.ty(t);
                            }
                            if let Some(d) = f.default {
                                self.out.push_str(" = ");
                                self.expr(d);
                            }
                            self.out.push(')');
                        }
                        Member::Method(f) => self.def(f),
                        Member::Stmt(s) => self.stmt(s),
                    }
                }
                self.out.push(')');
            }
            StmtKind::Enum(e) => {
                self.annotations(&e.annotations);
                self.out.push_str("(enum ");
                self.name(e.name);
                if let Some(r) = &e.repr {
                    self.out.push_str(" : ");
                    self.ty(r);
                }
                for v in &e.variants {
                    self.out.push(' ');
                    self.annotations(&v.annotations);
                    if let Some(val) = v.value {
                        self.out.push('(');
                        self.name(v.name);
                        self.out.push_str(" = ");
                        self.expr(val);
                        self.out.push(')');
                    } else {
                        self.name(v.name);
                    }
                }
                for m in &e.methods {
                    self.out.push(' ');
                    self.def(m);
                }
                self.out.push(')');
            }
        }
    }

    fn def(&mut self, f: &FnDecl) {
        self.annotations(&f.annotations);
        self.out.push_str("(def ");
        self.name(f.name);
        if let Some(params) = &f.params {
            self.out.push_str(" (");
            self.params(params);
            self.out.push(')');
        }
        if let Some(r) = &f.ret {
            self.out.push_str(" : ");
            self.ty(r);
        }
        self.block(&f.body);
        self.out.push(')');
    }

    fn params(&mut self, params: &[Param]) {
        for (n, p) in params.iter().enumerate() {
            if n > 0 {
                self.out.push(' ');
            }
            self.annotations(&p.annotations);
            let simple = p.ty.is_none() && p.default.is_none();
            if !simple {
                self.out.push('(');
            }
            if p.is_field {
                self.out.push('@');
            }
            self.name(p.name);
            if let Some(t) = &p.ty {
                self.out.push_str(" : ");
                self.ty(t);
            }
            if let Some(d) = p.default {
                self.out.push_str(" = ");
                self.expr(d);
            }
            if !simple {
                self.out.push(')');
            }
        }
    }

    fn pattern(&mut self, p: &Pattern) {
        match p {
            Pattern::Name(n, _) => self.name(*n),
            Pattern::Tuple(items, _) => {
                self.out.push_str("(tuple");
                for it in items {
                    self.out.push(' ');
                    self.pattern(it);
                }
                self.out.push(')');
            }
            Pattern::Record(names, _) => {
                self.out.push_str("(record");
                for (n, _) in names {
                    self.out.push(' ');
                    self.name(*n);
                }
                self.out.push(')');
            }
            Pattern::List { items, rest, .. } => {
                self.out.push_str("(list");
                for it in items {
                    self.out.push(' ');
                    self.pattern(it);
                }
                if let Some((r, _)) = rest {
                    self.out.push_str(" ...");
                    self.name(*r);
                }
                self.out.push(')');
            }
        }
    }

    fn sep_types(&mut self, items: &[TypeExpr]) {
        for (n, it) in items.iter().enumerate() {
            if n > 0 {
                self.out.push_str(", ");
            }
            self.ty(it);
        }
    }

    fn ty(&mut self, t: &TypeExpr) {
        match &t.kind {
            TypeKind::Named(path) => self.path(path),
            TypeKind::List(inner) => {
                self.out.push('[');
                self.ty(inner);
                self.out.push(']');
            }
            TypeKind::Hash(k, v) => {
                self.out.push('{');
                self.ty(k);
                self.out.push_str(" => ");
                self.ty(v);
                self.out.push('}');
            }
            TypeKind::Record(fields) => {
                self.out.push('{');
                for (n, (name, ty)) in fields.iter().enumerate() {
                    if n > 0 {
                        self.out.push_str(", ");
                    }
                    self.name(*name);
                    self.out.push_str(": ");
                    self.ty(ty);
                }
                self.out.push('}');
            }
            TypeKind::Tuple(items) => {
                self.out.push('(');
                self.sep_types(items);
                self.out.push(')');
            }
            TypeKind::Optional(inner) => {
                let paren = matches!(inner.kind, TypeKind::Union(_) | TypeKind::Fn(..));
                if paren {
                    self.out.push('(');
                }
                self.ty(inner);
                if paren {
                    self.out.push(')');
                }
                self.out.push('?');
            }
            TypeKind::Union(items) => {
                for (n, it) in items.iter().enumerate() {
                    if n > 0 {
                        self.out.push_str(" | ");
                    }
                    self.ty(it);
                }
            }
            TypeKind::LitStr(s) => {
                let _ = write!(self.out, "{s:?}");
            }
            TypeKind::LitInt(n) => {
                let _ = write!(self.out, "{n}");
            }
            TypeKind::Fn(params, ret) => {
                self.out.push('(');
                self.sep_types(params);
                self.out.push_str("): ");
                self.ty(ret);
            }
        }
    }

    fn arg(&mut self, a: &Arg) {
        if let Some(n) = a.name {
            self.name(n);
            self.out.push_str(": ");
        }
        self.expr(a.value);
    }

    fn args(&mut self, args: Slice<Arg>, block: Option<LambdaId>) {
        for a in self.ast.get(args) {
            self.out.push(' ');
            self.arg(a);
        }
        if let Some(b) = block {
            self.out.push(' ');
            self.lambda(self.ast.lambda(b));
        }
    }

    fn lambda(&mut self, l: &Lambda) {
        self.out.push_str("(fn (");
        self.params(&l.params);
        self.out.push(')');
        self.block(&l.body);
        self.out.push(')');
    }

    fn list(&mut self, head: &str, items: Slice<ExprId>) {
        self.out.push('(');
        self.out.push_str(head);
        for it in self.ast.get(items) {
            self.out.push(' ');
            self.expr(*it);
        }
        self.out.push(')');
    }

    fn else_block(&mut self, else_: &Option<Block>) {
        if let Some(e) = else_ {
            self.out.push_str(" (else");
            self.block(e);
            self.out.push(')');
        }
    }

    fn expr(&mut self, e: ExprId) {
        let ast = self.ast;
        match ast.kind(e) {
            ExprKind::Int(n) => {
                let _ = write!(self.out, "{n}");
            }
            ExprKind::Float(f) => {
                let _ = write!(self.out, "{f:?}");
            }
            ExprKind::Str(s) => self.quoted(s),
            ExprKind::Interp(parts) => {
                self.out.push_str("(str");
                for p in ast.get(parts) {
                    self.out.push(' ');
                    match *p {
                        StrPart::Lit(s) => self.quoted(s),
                        StrPart::Expr(e) => self.expr(e),
                    }
                }
                self.out.push(')');
            }
            ExprKind::Bool(b) => {
                let _ = write!(self.out, "{b}");
            }
            ExprKind::Nil => self.out.push_str("nil"),
            ExprKind::SelfRef => self.out.push_str("self"),
            ExprKind::Ident(s) => self.name(s),
            ExprKind::IVar(s) => {
                self.out.push('@');
                self.name(s);
            }
            ExprKind::Path(p) => self.path(ast.get(p)),
            ExprKind::List(items) => self.list("list", items),
            ExprKind::Tuple(items) => self.list("tuple", items),
            ExprKind::Hash(entries) => {
                self.out.push_str("(hash");
                for &(k, v) in ast.get(entries) {
                    self.out.push_str(" (");
                    self.expr(k);
                    self.out.push_str(" => ");
                    self.expr(v);
                    self.out.push(')');
                }
                self.out.push(')');
            }
            ExprKind::Record(fields) => {
                self.out.push_str("(record");
                for &(n, v) in ast.get(fields) {
                    self.out.push_str(" (");
                    self.name(n);
                    self.out.push_str(": ");
                    self.expr(v);
                    self.out.push(')');
                }
                self.out.push(')');
            }
            ExprKind::Unary(op, x) => {
                self.out.push_str(match op {
                    UnaryOp::Neg => "(- ",
                    UnaryOp::Not => "(not ",
                });
                self.expr(x);
                self.out.push(')');
            }
            ExprKind::Binary(op, a, b) => {
                self.out.push('(');
                self.out.push_str(op.as_str());
                self.out.push(' ');
                self.expr(a);
                self.out.push(' ');
                self.expr(b);
                self.out.push(')');
            }
            ExprKind::Range { lo, hi, inclusive } => {
                self.out.push_str(if inclusive { "(.. " } else { "(... " });
                self.expr(lo);
                self.out.push(' ');
                self.expr(hi);
                self.out.push(')');
            }
            ExprKind::Assign { target, op, value } => {
                self.out.push('(');
                if let Some(op) = op {
                    self.out.push_str(op.as_str());
                }
                self.out.push_str("= ");
                self.expr(target);
                self.out.push(' ');
                self.expr(value);
                self.out.push(')');
            }
            ExprKind::Call {
                callee,
                args,
                block,
            } => {
                self.out.push_str("(call ");
                self.expr(callee);
                self.args(args, block);
                self.out.push(')');
            }
            ExprKind::MethodCall {
                recv,
                name,
                args,
                block,
            } => {
                self.out.push_str("(.");
                self.name(name);
                self.out.push(' ');
                self.expr(recv);
                self.args(args, block);
                self.out.push(')');
            }
            ExprKind::Field(recv, name) => {
                self.out.push_str("(. ");
                self.expr(recv);
                self.out.push(' ');
                self.name(name);
                self.out.push(')');
            }
            ExprKind::TupleIndex(recv, idx) => {
                self.out.push_str("(. ");
                self.expr(recv);
                let _ = write!(self.out, " {idx})");
            }
            ExprKind::Index(recv, idx) | ExprKind::IndexOpt(recv, idx) => {
                let opt = matches!(ast.kind(e), ExprKind::IndexOpt(..));
                self.out.push_str(if opt { "(index? " } else { "(index " });
                self.expr(recv);
                self.out.push(' ');
                self.expr(idx);
                self.out.push(')');
            }
            ExprKind::Lambda(l) => self.lambda(ast.lambda(l)),
            ExprKind::If(c) => {
                let x = ast.if_(c);
                self.out.push_str("(if ");
                self.expr(x.cond);
                self.out.push_str(" (then");
                self.block(&x.then);
                self.out.push(')');
                self.else_block(&x.else_);
                self.out.push(')');
            }
            ExprKind::Case(c) => {
                let x = ast.case(c);
                self.out.push_str("(case ");
                self.expr(x.subject);
                for arm in &x.arms {
                    self.out.push_str(" (when (");
                    for (n, p) in ast.get(arm.patterns).iter().enumerate() {
                        if n > 0 {
                            self.out.push(' ');
                        }
                        self.expr(*p);
                    }
                    self.out.push(')');
                    self.block(&arm.body);
                    self.out.push(')');
                }
                self.else_block(&x.else_);
                self.out.push(')');
            }
            ExprKind::While(c) => {
                let x = ast.while_(c);
                self.out.push_str("(while ");
                self.expr(x.cond);
                self.block(&x.body);
                self.out.push(')');
            }
            ExprKind::For(c) => {
                let x = ast.for_(c);
                self.out.push_str("(for (");
                for (n, (v, _)) in x.vars.iter().enumerate() {
                    if n > 0 {
                        self.out.push(' ');
                    }
                    self.name(*v);
                }
                self.out.push_str(") ");
                self.expr(x.iter);
                self.block(&x.body);
                self.out.push(')');
            }
            ExprKind::Begin(c) => {
                let x = ast.begin(c);
                self.out.push_str("(begin");
                self.block(&x.body);
                for r in &x.rescues {
                    self.out.push_str(" (rescue");
                    if let Some((v, _)) = r.var {
                        self.out.push(' ');
                        self.name(v);
                    }
                    if let Some(t) = &r.ty {
                        self.out.push_str(" : ");
                        self.ty(t);
                    }
                    self.block(&r.body);
                    self.out.push(')');
                }
                self.else_block(&x.else_);
                if let Some(b) = &x.ensure {
                    self.out.push_str(" (ensure");
                    self.block(b);
                    self.out.push(')');
                }
                self.out.push(')');
            }
            ExprKind::Super(args) => {
                self.out.push_str("(super");
                self.args(args, None);
                self.out.push(')');
            }
            ExprKind::Raise(v) => {
                self.out.push_str("(raise ");
                self.expr(v);
                self.out.push(')');
            }
            ExprKind::Return(v) => {
                self.out.push_str("(return");
                if let Some(v) = v {
                    self.out.push(' ');
                    self.expr(v);
                }
                self.out.push(')');
            }
            ExprKind::Break => self.out.push_str("(break)"),
            ExprKind::Next => self.out.push_str("(next)"),
        }
    }
}
