//! Script values serialized with `serde` (feature `serde`).
#![cfg(feature = "serde")]

use bokoni::Engine;

/// JSON of the program's value, or the serialization error.
fn json(src: &str) -> String {
    let mut e = Engine::new();
    let v = e
        .eval(src)
        .unwrap_or_else(|err| panic!("{}", err.render(src, "test")));
    serde_json::to_string(&e.serializable(&v)).unwrap_or_else(|err| format!("error: {err}"))
}

#[test]
fn base_values_and_containers() {
    assert_eq!(json("nil"), "null");
    assert_eq!(
        json("[true, -3, 1.5, \"é\\\"\\n\"]"),
        r#"[true,-3,1.5,"é\"\n"]"#
    );
    assert_eq!(json("9223372036854775807"), "9223372036854775807");
    assert_eq!(json("(1, \"a\", [nil])"), r#"[1,"a",[null]]"#);
    assert_eq!(json("{x: 1, y: {z: []}}"), r#"{"x":1,"y":{"z":[]}}"#);
    assert_eq!(json("{\"b\" => 1, \"a\" => 2}"), r#"{"b":1,"a":2}"#);
    // Shared, not cyclic: written twice.
    assert_eq!(json("let a = [1]\n[a, a]"), "[[1],[1]]");
}

#[test]
fn objects_and_enums() {
    let src = "type P\n  x: int = 0\n  tags: [str] = []\nend\n\
               class C\n  def init(@n: int, @p: P)\n  end\nend\n\
               C(2, P(x: 1, tags: [\"t\"]))";
    assert_eq!(json(src), r#"{"n":2,"p":{"x":1,"tags":["t"]}}"#);
    let src = "enum Plain\n  A\n  B\nend\n\
               enum S: str\n  X = \"x\"\n  Y\nend\n\
               enum N\n  T = 6\n  U = 17\nend\n\
               [Plain::B, S::X, S::Y, N::U]";
    assert_eq!(json(src), r#"["B","x","Y",17]"#);
}

#[test]
fn hash_keys_are_left_to_the_format() {
    assert_eq!(
        json("{1 => \"a\", true => \"b\"}"),
        r#"{"1":"a","true":"b"}"#
    );
    assert!(json("{nil => 1}").starts_with("error: key must be a string"));
}

#[test]
fn what_cannot_be_serialized() {
    assert_eq!(json("def f() = 1\nf"), "error: cannot serialize a function");
    assert_eq!(json("1..3"), "error: cannot serialize a range");
    let cyclic = json("let l: [any] = [1]\nl << l\nl");
    assert_eq!(
        cyclic,
        "error: cannot serialize a structure that contains itself"
    );
    let cyclic = json("let h: {str => any} = {}\nh[\"self\"] = {inner: [h]}\nh");
    assert_eq!(
        cyclic,
        "error: cannot serialize a structure that contains itself"
    );
}

#[test]
fn deep_nesting() {
    let v = json("let l = []\nfor i in 0...100000\n  l = [l]\nend\nl");
    // `serde_json` writes it (the stack grows); 100,001 levels.
    assert_eq!(v.len(), 200_002);
}
