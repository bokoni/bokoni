//! Reference cycles: freed when nothing else refers to them, kept otherwise.

use bokoni::Engine;

/// Containers still alive after a collection.
fn alive_after(e: &mut Engine) -> usize {
    e.collect_cycles();
    bokoni::gc::tracked()
}

#[test]
fn unreachable_cycles_are_freed() {
    for jit in [false, true] {
        let mut e = Engine::new();
        e.set_output(|_| {});
        e.set_jit(jit);
        e.set_static_checks(false);
        e.eval(
            "class Node\n  def init(@peer)\n  end\n  def set(p) = @peer = p\nend
def self_capture(): int
  let down = |n: int| if n == 0 then 0 else 1 + down(n - 1) end
  down(3)
end
def pair(): int
  let a = Node(nil)
  let b = Node(a)
  a.set(b)
  1
end
def lists(): int
  let xs = [1]
  xs << xs
  let h = {}
  h[1] = h
  let r = {me: nil}
  r.me = r
  xs.len()
end
def callback(): int
  let n = Node(nil)
  n.set(|| n)
  1
end",
        )
        .unwrap();
        let base = alive_after(&mut e);
        e.eval("let t = 0\nfor i in 0...2000\n  t += self_capture() + pair() + lists() + callback()\nend")
            .unwrap();
        let after = alive_after(&mut e);
        assert!(
            after <= base + 10,
            "jit={jit}: {base} containers before, {after} after"
        );
    }
}

#[test]
fn reachable_cycles_are_kept() {
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.set_static_checks(false);
    e.eval("class Node\n  def init(@peer, @v: int)\n  end\n  def peer = @peer\n  def v = @v\n  def set(p) = @peer = p\nend
let a = Node(nil, 1)
let b = Node(a, 2)
a.set(b)
let xs = [10]
xs << xs
def make()
  let f = |n: int| if n == 0 then 0 else n + f(n - 1) end
  f
end
let g = make()")
        .unwrap();
    // A cycle held by the host only.
    let held = e
        .eval("let c = Node(nil, 7)\nc.set(c)\nlet r = c\nc = nil\nr")
        .unwrap();
    e.eval("r = nil").unwrap();
    for _ in 0..3 {
        e.collect_cycles();
    }
    assert_eq!(
        e.eval_as::<i64>("a.peer.peer.peer.v + xs[1][1][0] + g(4)")
            .unwrap(),
        2 + 10 + 10
    );
    e.set_global("h", held);
    assert_eq!(e.eval_as::<i64>("h.peer.peer.v").unwrap(), 7);
}

#[test]
fn collected_during_a_long_script() {
    // Without calling `collect_cycles`: done automatically as values are created.
    let mut e = Engine::new();
    e.set_output(|_| {});
    e.eval("def pair(): int\n  let xs = []\n  let ys = [xs]\n  xs << ys\n  1\nend\nlet t = 0\nfor i in 0...300000\n  t += pair()\nend")
        .unwrap();
    assert!(bokoni::gc::tracked() < 250_000, "{}", bokoni::gc::tracked());
}

#[test]
fn cycles_through_shared_closures_and_tuples() {
    for jit in [false, true] {
        let mut e = Engine::new();
        e.set_output(|_| {});
        e.set_jit(jit);
        e.set_static_checks(false);
        e.eval(
            "class Node\n  def init(@peer)\n  end\n  def set(p) = @peer = p\nend
def shared(): int
  let n = Node(nil)
  let f = || n
  n.set([f, f])
  1
end
def tuple(): int
  let xs = [0]
  let t = (xs, xs)
  xs << t
  1
end
def handlers(): int
  let hs = {}
  let me = |x| hs
  hs[1] = me
  hs[2] = me
  1
end",
        )
        .unwrap();
        let base = alive_after(&mut e);
        e.eval("let t = 0\nfor i in 0...2000\n  t += shared() + tuple() + handlers()\nend")
            .unwrap();
        let after = alive_after(&mut e);
        assert!(
            after <= base + 10,
            "jit={jit}: {base} containers before, {after} after"
        );
    }
}

#[test]
fn cycles_built_by_the_host() {
    use bokoni::Value;
    use std::cell::RefCell;
    use std::rc::{Rc, Weak};
    let mut e = Engine::new();
    e.set_output(|_| {});
    // Lists the host built (not registered when created), put into cycles by the script,
    // then dropped: freed.
    type Made = Rc<RefCell<Vec<Weak<RefCell<Vec<Value>>>>>>;
    let made: Made = Rc::default();
    let m = made.clone();
    e.register_fn("make", move || {
        let l = Rc::new(RefCell::new(vec![Value::Int(1)]));
        m.borrow_mut().push(Rc::downgrade(&l));
        Value::List(l)
    });
    e.eval("def f(): int\n  let xs = make()\n  xs << xs\n  1\nend\nlet t = 0\nfor i in 0...1000\n  t += f()\nend")
        .unwrap();
    e.collect_cycles();
    let alive = made
        .borrow()
        .iter()
        .filter(|w| w.strong_count() > 0)
        .count();
    assert_eq!(alive, 0);
    // Kept while reachable from a global.
    e.set_global("g", Value::List(Rc::new(RefCell::new(vec![]))));
    e.eval("g << g\ng << 5").unwrap();
    e.collect_cycles();
    assert_eq!(e.eval_as::<i64>("g[0][0][1]").unwrap(), 5);
}
