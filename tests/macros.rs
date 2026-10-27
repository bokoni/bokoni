//! Macros: `macro name(params) … end`, expanded where they are called.

use bokoni::Engine;
use std::cell::RefCell;
use std::rc::Rc;

fn run(src: &str) -> String {
    let mut e = Engine::new();
    let out = Rc::new(RefCell::new(String::new()));
    let o = out.clone();
    e.set_output(move |s| o.borrow_mut().push_str(s));
    let r = match e.eval(src) {
        Ok(v) => format!("=> {}", e.inspect(&v)),
        Err(err) => err.render(src, "test"),
    };
    format!("{}{r}", out.borrow())
}

#[test]
fn getters_and_to_s() {
    // The two examples of `docs/syntax.md`.
    let src = "macro getters(*names)
  {% for name in names %}
    def {{name}} = @{{name}}
  {% end %}
end

class User
  def init(@name: str, @email: str)
  end

  getters(name, email)
end

macro derive_to_s
  def to_s: str
    {% let parts = @type.fields.map(|f| \"#{f.name}=\\#{@#{f.name}}\") %}
    \"{{@type.name}}({{parts.join(\", \")}})\"
  end
end

type Point
  x: int = 0
  y: int = 0
  derive_to_s()
end

let u = User(\"ann\", \"a@x.org\")
[u.name, u.email, Point(x: 1).to_s()]";
    assert_eq!(run(src), "=> [\"ann\", \"a@x.org\", \"Point(x=1, y=0)\"]");
    // Fields declared by `init(@x)`, with their types; methods declared before the call.
    let src = "macro describe
  def describe(): str = {{@type.fields.map(|f| \"#{f.name}:#{f.type}\").join(\" \").stringify}} + \" / \" + {{@type.methods.join(\",\").stringify}}
end
class C
  def init(@a: int, b: str, @c)
  end
  def m() = 1
  describe()
end
C(1, \"x\", 2).describe()";
    assert_eq!(run(src), "=> \"a:int c:any / init,m\"");
}

#[test]
fn statements_conditions_and_hygiene() {
    let src = "macro twice(expr)
  let tmp = {{expr}}
  tmp + tmp
end

macro repeat(n, body)
  {% if n == \"0\" %}
    0
  {% else %}
    {{body}} * {{n}}
  {% end %}
end

macro constants(*names)
  {% for n in names %}
  let {{n.upper}} = {{n.stringify}}
  {% end %}
end

let tmp = 10
let a = twice(tmp + 1)
constants(red, green)
[a, tmp, repeat(3, 7), repeat(0, 7), RED, GREEN]";
    // The macro's own `tmp` does not hide the caller's.
    assert_eq!(run(src), "=> [22, 10, 21, 0, \"red\", \"green\"]");
}

#[test]
fn macros_from_previous_scripts_and_required_files() {
    let mut e = Engine::new();
    e.eval("macro square_fn(name, k)\n  def {{name}}(x: int): int = x * x * {{k}}\nend")
        .unwrap();
    e.eval("square_fn(sq3, 3)").unwrap();
    assert_eq!(e.eval_as::<i64>("sq3(2)").unwrap(), 12);
    // A required file declares a macro used by the file that requires it.
    let files = [(
        "lib",
        "macro double_fn(name)\n  def {{name}}(x: int): int = x * 2\nend",
    )];
    e.set_loader(
        move |name, _| match files.iter().find(|(n, _)| *n == name) {
            Some((n, t)) => Ok(bokoni::Source {
                name: format!("{n}.bk"),
                text: t.to_string(),
            }),
            None => Err("no such file".into()),
        },
    );
    assert_eq!(
        e.eval_as::<i64>("require \"lib\"\ndouble_fn(dbl)\ndbl(21)")
            .unwrap(),
        42
    );
}

#[test]
fn annotations_and_def() {
    // The example of `docs/syntax.md`: fields' annotations, positional or named.
    let src = "annotation Validate
  min: int? = nil
  max: int? = nil
end

macro type_doc
  def doc(): str = {{@type.annotation(Doc).text}}
end

macro validate_fields
  def validate(): [str]
    let errors: [str] = []
    {% for f in @type.fields %}
      {% if v = f.annotation(Validate) %}
        {% if v.min %}
          errors << \"{{f.name}} < {{v.min}}\" if @{{f.name}} < {{v.min}}
        {% end %}
        {% if v.max %}
          errors << \"{{f.name}} > {{v.max}}\" if @{{f.name}} > {{v.max}}
        {% end %}
      {% end %}
    {% end %}
    errors
  end
end

@[Doc(\"A server\")]
type Server
  @[Validate(min: 1, max: 65535)]
  port: int = 0
  @[Validate(0)]
  workers: int = 0
  name: str = \"\"
  validate_fields()
  type_doc()
end

let s = Server(workers: -2)
[s.validate(), s.doc()]";
    assert_eq!(
        run(src),
        "=> [[\"port < 1\", \"workers < 0\"], \"A server\"]"
    );
    // `@def`: the method whose body contains the call.
    let src = "macro trace
  print({{\"#{@def.name}(#{@def.params.map(|p| \"#{p.name}: #{p.type}\").join(\", \")}): #{@def.return_type}\".stringify}})
  {% if a = @def.annotation(Deprecated) %}
  print(\"deprecated: \" + {{a.message}})
  {% end %}
end
@[Deprecated(\"use plus\")]
def add(a: int, b = 2): int
  trace()
  a + b
end
def f
  trace()
end
add(1)
f()";
    assert_eq!(
        run(src),
        "add(a: int, b: any): int\ndeprecated: use plus\nf(): any\n=> nil"
    );
    let out = run("macro m\n  {{@def.name}}\nend\nm()");
    assert!(
        out.contains("`@def` is only available in the body of a method"),
        "{out}"
    );
    let out = run(
        "annotation A\n  x: int = 0\nend\nmacro m\n  {{@type.annotation(A).y}}\nend\n@[A]\ntype T\n  m()\nend",
    );
    assert!(out.contains("annotation `A` has no field `y`"), "{out}");
}

#[test]
fn macro_errors() {
    let check = |src: &str, msg: &str| {
        let out = run(src);
        assert!(out.contains(msg), "expected `{msg}` in:\n{out}");
    };
    check(
        "macro m(a)\n  {{a}}\nend\nm(1, 2)",
        "macro `m`: macro `m` expects 1 argument, got 2",
    );
    check(
        "macro m(a)\n  {{b}}\nend\nm(1)",
        "macro `m`: unknown name `b` in a macro",
    );
    check(
        "macro m()\n  {% for x in [1] %}\n  x\n  end\nend\nm()",
        "`{% for %}` without its `{% end %}`",
    );
    check(
        "macro m()\n  {{@type.name}}\nend\nm()",
        "`@type` is only available in the body of a type",
    );
    // An error in the generated code: pointed into it, with the macro and its call.
    let out = run("macro m(a)\n  let x = {{a}} * * 2\nend\nm(1)");
    assert!(
        out.contains("in the code generated by macro `m`, line 4"),
        "{out}"
    );
    assert!(out.contains("let x__m1 = 1 * * 2"), "{out}");
    check("macro m()\n  1\n", "expected `end` to close `macro`");
    check("macro m()\n  m()\nend\nm()", "expansion too deep");
}
