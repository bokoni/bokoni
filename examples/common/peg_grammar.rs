#[derive(Debug)]
#[allow(dead_code)]
enum PExpr<'a> {
    Int(i64),
    Str(&'a str),
    Ident(&'a str),
    List(Vec<PExpr<'a>>),
    Neg(Box<PExpr<'a>>),
    Bin(&'static str, Box<PExpr<'a>>, Box<PExpr<'a>>),
    Call(Box<PExpr<'a>>, Vec<PExpr<'a>>),
    Field(Box<PExpr<'a>>, &'a str),
    Index(Box<PExpr<'a>>, Box<PExpr<'a>>),
}

#[derive(Debug)]
#[allow(dead_code)]
enum PStmt<'a> {
    Let(&'a str, PExpr<'a>),
    Assign(&'a str, PExpr<'a>),
    Expr(PExpr<'a>),
}

enum PostOp<'a> {
    Call(Vec<PExpr<'a>>),
    Field(&'a str),
    Index(PExpr<'a>),
}

fn bin<'a>(op: &'static str, a: PExpr<'a>, b: PExpr<'a>) -> PExpr<'a> {
    PExpr::Bin(op, Box::new(a), Box::new(b))
}

peg::parser!(grammar pg() for str {
    rule _() = [' ' | '\t']*
    rule nl() = _ ("\n" _)+
    rule ident_char() = ['a'..='z' | 'A'..='Z' | '0'..='9' | '_']
    rule keyword() = "let" !ident_char()

    pub rule program() -> Vec<PStmt<'input>> =
        "\n"* s:(stmt() ** nl()) nl()? _ ![_] { s }

    rule stmt() -> PStmt<'input> =
        _ s:(let_stmt() / assign() / e:expr() { PStmt::Expr(e) }) _ { s }

    rule let_stmt() -> PStmt<'input> =
        "let" !ident_char() _ n:ident() _ "=" _ e:expr() { PStmt::Let(n, e) }

    rule assign() -> PStmt<'input> =
        n:ident() _ "=" !"=" _ e:expr() { PStmt::Assign(n, e) }

    rule expr() -> PExpr<'input> = precedence!{
        a:(@) _ "==" _ b:@ { bin("==", a, b) }
        a:(@) _ "<" _ b:@ { bin("<", a, b) }
        --
        a:(@) _ "+" _ b:@ { bin("+", a, b) }
        a:(@) _ "-" _ b:@ { bin("-", a, b) }
        --
        a:(@) _ "*" _ b:@ { bin("*", a, b) }
        a:(@) _ "/" _ b:@ { bin("/", a, b) }
        --
        "-" _ a:@ { PExpr::Neg(Box::new(a)) }
        --
        p:postfix() { p }
    }

    rule postfix() -> PExpr<'input> =
        p:primary() ops:post_op()* {
            ops.into_iter().fold(p, |e, op| match op {
                PostOp::Call(args) => PExpr::Call(Box::new(e), args),
                PostOp::Field(n) => PExpr::Field(Box::new(e), n),
                PostOp::Index(i) => PExpr::Index(Box::new(e), Box::new(i)),
            })
        }

    rule post_op() -> PostOp<'input> =
        "(" _ a:(expr() ** (_ "," _)) _ ")" { PostOp::Call(a) }
        / "." n:ident() { PostOp::Field(n) }
        / "[" _ e:expr() _ "]" { PostOp::Index(e) }

    rule primary() -> PExpr<'input> =
        n:$(['0'..='9']+) { PExpr::Int(n.parse().unwrap()) }
        / "\"" s:$([^ '"']*) "\"" { PExpr::Str(s) }
        / "[" _ a:(expr() ** (_ "," _)) _ "]" { PExpr::List(a) }
        / "(" _ e:expr() _ ")" { e }
        / i:ident() { PExpr::Ident(i) }

    rule ident() -> &'input str =
        quiet!{ !keyword() s:$(['a'..='z' | 'A'..='Z' | '_'] ident_char()*) { s } }
        / expected!("identifiant")
});

