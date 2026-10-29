//! Collection of reference cycles.
//!
//! Values are reference counted (`Rc`): a cycle (a closure that captures itself, two
//! objects that point to each other, a list that contains itself) is never freed by
//! counting alone. The containers created by scripts (lists, hashes, records, objects,
//! cells of captured variables, closures that capture, tuples of containers) are
//! registered here (weak references), and
//! [`collect`] frees those that are only reachable from cycles — the algorithm of
//! CPython's collector, which fits reference counting:
//!
//! 1. each container's count, minus the references it gets from the other containers
//!    registered: what remains comes from outside (a register, a global, the host…);
//! 2. the containers referenced from outside, and all those reachable from them, are
//!    alive;
//! 3. the others are only referenced by cycles: their contents are taken out, which
//!    frees the cycle.
//!
//! It is safe by construction: a container that cannot be read (borrowed at that
//! moment) or that is referenced by a value not registered (one a host built without
//! [`Value::list`] / [`Value::hash`] / [`Value::record`], a tuple shared by several
//! owners) counts as referenced from outside, so it and what it reaches stay alive. A collection runs at safe points: after a script, and at a function call once
//! enough containers were created since the last one.

use crate::objects::Object;
use crate::value::{Function, HashMap, Record, ThinRc, ThinWeak, Value};
use crate::vm::Cell;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// A registered container.
enum Node {
    List(Weak<RefCell<Vec<Value>>>),
    Hash(Weak<RefCell<HashMap>>),
    Record(Weak<RefCell<Record>>),
    Object(Weak<RefCell<Object>>),
    Cell(Weak<RefCell<Value>>),
    Closure(Weak<Function>),
    Tuple(ThinWeak<Value>),
}

/// A registered container, alive during a collection.
enum Live {
    List(Rc<RefCell<Vec<Value>>>),
    Hash(Rc<RefCell<HashMap>>),
    Record(Rc<RefCell<Record>>),
    Object(Rc<RefCell<Object>>),
    Cell(Rc<RefCell<Value>>),
    Closure(Rc<Function>),
    Tuple(ThinRc<Value>),
}

struct Registry {
    nodes: Vec<Node>,
    collecting: bool,
}

const MIN_THRESHOLD: usize = 20_000;

thread_local! {
    static REGISTRY: RefCell<Registry> = const {
        RefCell::new(Registry {
            nodes: Vec::new(),
            collecting: false,
        })
    };
    /// Registrations left before a collection is due (0: due), checked at every call.
    static LEFT: std::cell::Cell<usize> = const { std::cell::Cell::new(MIN_THRESHOLD) };
}

fn register(n: Node) {
    REGISTRY.with(|r| {
        if let Ok(mut r) = r.try_borrow_mut() {
            r.nodes.push(n);
        }
    });
    LEFT.with(|l| l.set(l.get().saturating_sub(1)));
}

/// Registers the container `v` (a value just created by a script).
#[inline]
pub fn track(v: &Value) {
    match v {
        Value::List(l) => register(Node::List(Rc::downgrade(l))),
        Value::Hash(h) => register(Node::Hash(Rc::downgrade(h))),
        Value::Record(r) => register(Node::Record(Rc::downgrade(r))),
        // (Not one whose fields all have simple types: it cannot be in a cycle.)
        Value::Object(o) if !o.try_borrow().is_ok_and(|o| o.info.acyclic) => {
            register(Node::Object(Rc::downgrade(o)))
        }
        // (A closure that captures nothing, a tuple of simple values: never in a cycle.)
        Value::Function(f) if matches!(&**f, Function::Script(c) if !c.upvals.is_empty()) => {
            register(Node::Closure(Rc::downgrade(f)))
        }
        Value::Tuple(t) if t.iter().any(|v| addr_of_value(v).is_some()) => {
            register(Node::Tuple(ThinRc::downgrade(t)))
        }
        _ => {}
    }
}

/// Registers `v` and the containers in it that only it holds (a value built by the
/// host: its containers were not registered when created). Those already registered are
/// found again once at the next collection.
pub fn track_new(v: &Value) {
    let mut work = vec![v.clone()];
    while let Some(v) = work.pop() {
        // (The copy on the work stack, plus the owner.)
        let unique = match &v {
            Value::List(l) => Rc::strong_count(l) <= 2,
            Value::Hash(h) => Rc::strong_count(h) <= 2,
            Value::Record(r) => Rc::strong_count(r) <= 2,
            Value::Object(o) => Rc::strong_count(o) <= 2,
            Value::Tuple(t) => ThinRc::strong_count(t) <= 2,
            _ => false,
        };
        if !unique {
            continue;
        }
        track(&v);
        let mut push = |x: &Value| {
            if addr_of_value(x).is_some() {
                work.push(x.clone());
            }
        };
        match &v {
            Value::List(l) => {
                if let Ok(l) = l.try_borrow() {
                    l.iter().for_each(&mut push);
                }
            }
            Value::Hash(h) => {
                if let Ok(h) = h.try_borrow() {
                    h.iter().for_each(|(_, x)| push(x));
                }
            }
            Value::Record(r) => {
                if let Ok(r) = r.try_borrow() {
                    r.fields.iter().for_each(|(_, x)| push(x));
                }
            }
            Value::Object(o) => {
                if let Ok(o) = o.try_borrow() {
                    o.fields.iter().for_each(&mut push);
                }
            }
            Value::Tuple(t) => t.iter().for_each(&mut push),
            _ => {}
        }
    }
}

/// Registers the cell of a captured variable.
#[inline]
pub fn track_cell(c: &Cell) {
    register(Node::Cell(Rc::downgrade(c)));
}

/// A collection is due (many containers created since the last one).
#[inline]
pub fn due() -> bool {
    LEFT.with(|l| l.get() == 0)
}

/// Collects if due.
#[inline]
pub fn maybe_collect() {
    if due() {
        collect();
    }
}

/// Address identifying a registered container.
pub(crate) fn addr_of_value(v: &Value) -> Option<usize> {
    Some(match v {
        Value::List(l) => Rc::as_ptr(l) as *const () as usize,
        Value::Hash(h) => Rc::as_ptr(h) as *const () as usize,
        Value::Record(r) => Rc::as_ptr(r) as *const () as usize,
        Value::Object(o) => Rc::as_ptr(o) as *const () as usize,
        Value::Function(f) => Rc::as_ptr(f) as *const () as usize,
        Value::Tuple(t) => ThinRc::as_ptr(t) as *const () as usize,
        _ => return None,
    })
}

impl Live {
    fn addr(&self) -> usize {
        match self {
            Live::List(x) => Rc::as_ptr(x) as *const () as usize,
            Live::Hash(x) => Rc::as_ptr(x) as *const () as usize,
            Live::Record(x) => Rc::as_ptr(x) as *const () as usize,
            Live::Object(x) => Rc::as_ptr(x) as *const () as usize,
            Live::Cell(x) => Rc::as_ptr(x) as *const () as usize,
            Live::Closure(x) => Rc::as_ptr(x) as *const () as usize,
            Live::Tuple(x) => ThinRc::as_ptr(x) as *const () as usize,
        }
    }

    /// Calls `f` on the address of each value it references that may be registered
    /// (directly, or through a tuple or a closure not registered that it alone owns).
    /// `false` if it cannot be read now.
    fn children(&self, registered: &dyn Fn(usize) -> bool, f: &mut dyn FnMut(usize)) -> bool {
        fn value(v: &Value, registered: &dyn Fn(usize) -> bool, f: &mut dyn FnMut(usize)) {
            let Some(a) = addr_of_value(v) else {
                return;
            };
            if registered(a) {
                return f(a);
            }
            match v {
                // A tuple owned by this container only: part of it.
                Value::Tuple(t) if ThinRc::strong_count(t) == 1 => {
                    for x in t.iter() {
                        value(x, registered, f);
                    }
                }
                Value::Function(c) if Rc::strong_count(c) == 1 => {
                    if let Function::Script(c) = &**c {
                        for u in c.upvals.iter() {
                            f(Rc::as_ptr(u) as *const () as usize);
                        }
                    }
                }
                _ => {}
            }
        }
        let mut value = |v: &Value| value(v, registered, f);
        match self {
            Live::List(l) => match l.try_borrow() {
                Ok(l) => l.iter().for_each(value),
                Err(_) => return false,
            },
            Live::Hash(h) => match h.try_borrow() {
                Ok(h) => h.iter().for_each(|(_, v)| value(v)),
                Err(_) => return false,
            },
            Live::Record(r) => match r.try_borrow() {
                Ok(r) => r.fields.iter().for_each(|(_, v)| value(v)),
                Err(_) => return false,
            },
            Live::Object(o) => match o.try_borrow() {
                Ok(o) => o.fields.iter().for_each(value),
                Err(_) => return false,
            },
            Live::Cell(c) => match c.try_borrow() {
                Ok(v) => value(&v),
                Err(_) => return false,
            },
            // Its cells (registered).
            Live::Closure(c) => {
                if let Function::Script(c) = &**c {
                    for u in c.upvals.iter() {
                        f(Rc::as_ptr(u) as *const () as usize);
                    }
                }
            }
            Live::Tuple(t) => t.iter().for_each(value),
        }
        true
    }

    /// Takes its contents out (it is garbage): into `trash`.
    fn clear(&self, trash: &mut Vec<Value>) {
        match self {
            Live::List(l) => {
                if let Ok(mut l) = l.try_borrow_mut() {
                    trash.append(&mut l);
                }
            }
            Live::Hash(h) => {
                if let Ok(mut h) = h.try_borrow_mut() {
                    trash.extend(std::mem::take(&mut *h).into_values());
                }
            }
            Live::Record(r) => {
                if let Ok(mut r) = r.try_borrow_mut() {
                    trash.extend(
                        r.fields
                            .iter_mut()
                            .map(|(_, v)| std::mem::replace(v, Value::Nil)),
                    );
                }
            }
            Live::Object(o) => {
                if let Ok(mut o) = o.try_borrow_mut() {
                    trash.extend(
                        o.fields
                            .iter_mut()
                            .map(|v| std::mem::replace(v, Value::Nil)),
                    );
                }
            }
            Live::Cell(c) => {
                if let Ok(mut c) = c.try_borrow_mut() {
                    trash.push(std::mem::replace(&mut *c, Value::Nil));
                }
            }
            // Immutable: freed when the mutable containers of their cycle are emptied.
            Live::Closure(_) | Live::Tuple(_) => {}
        }
    }
}

/// Frees the containers only reachable from cycles. Returns how many were freed.
pub fn collect() -> usize {
    // The registered containers still alive, with their counts.
    let mut live: Vec<(Live, usize)> = Vec::new();
    let ok = REGISTRY.with(|r| {
        let Ok(mut r) = r.try_borrow_mut() else {
            return false;
        };
        if r.collecting {
            return false;
        }
        r.collecting = true;
        let nodes = std::mem::take(&mut r.nodes);
        let mut seen = crate::intern::FxHashSet::default();
        for n in nodes {
            // (The count before `upgrade`, which adds one.)
            let item = match &n {
                Node::List(w) => w.upgrade().map(|x| (Live::List(x), w.strong_count())),
                Node::Hash(w) => w.upgrade().map(|x| (Live::Hash(x), w.strong_count())),
                Node::Record(w) => w.upgrade().map(|x| (Live::Record(x), w.strong_count())),
                Node::Object(w) => w.upgrade().map(|x| (Live::Object(x), w.strong_count())),
                Node::Cell(w) => w.upgrade().map(|x| (Live::Cell(x), w.strong_count())),
                Node::Closure(w) => w.upgrade().map(|x| (Live::Closure(x), w.strong_count())),
                Node::Tuple(w) => w.upgrade().map(|x| (Live::Tuple(x), w.strong_count())),
            };
            // (Registered twice, a value from the host: once.)
            if let Some((x, count)) = item
                && seen.insert(x.addr())
            {
                live.push((x, count - 1));
                r.nodes.push(n);
            }
        }
        true
    });
    if !ok {
        return 0;
    }
    let index: crate::intern::FxHashMap<usize, usize> = live
        .iter()
        .enumerate()
        .map(|(i, (x, _))| (x.addr(), i))
        .collect();
    // 1. References from outside: the count minus those from registered containers.
    let mut outside: Vec<isize> = live.iter().map(|(_, c)| *c as isize).collect();
    let registered = |a: usize| index.contains_key(&a);
    let mut readable = vec![true; live.len()];
    for (i, (x, _)) in live.iter().enumerate() {
        readable[i] = x.children(&registered, &mut |a| {
            if let Some(&j) = index.get(&a) {
                outside[j] -= 1;
            }
        });
    }
    // 2. Alive: referenced from outside (or not readable), and what they reach.
    let mut alive = vec![false; live.len()];
    let mut work: Vec<usize> = (0..live.len())
        .filter(|&i| outside[i] > 0 || !readable[i])
        .collect();
    for &i in &work {
        alive[i] = true;
    }
    while let Some(i) = work.pop() {
        live[i].0.children(&registered, &mut |a| {
            if let Some(&j) = index.get(&a)
                && !alive[j]
            {
                alive[j] = true;
                work.push(j);
            }
        });
    }
    // 3. The others: their contents taken out, then dropped (the cycles are broken).
    let mut trash = Vec::new();
    let mut freed = 0;
    for (i, (x, _)) in live.iter().enumerate() {
        if !alive[i] {
            x.clear(&mut trash);
            freed += 1;
        }
    }
    let survivors = live.len() - freed;
    drop(live);
    drop(trash);
    REGISTRY.with(|r| {
        let mut r = r.borrow_mut();
        r.nodes.retain(|n| match n {
            Node::List(w) => w.strong_count() > 0,
            Node::Hash(w) => w.strong_count() > 0,
            Node::Record(w) => w.strong_count() > 0,
            Node::Object(w) => w.strong_count() > 0,
            Node::Cell(w) => w.strong_count() > 0,
            Node::Closure(w) => w.strong_count() > 0,
            Node::Tuple(w) => w.strong_count() > 0,
        });
        // Next collection after as many new containers as survived (at least the minimum).
        LEFT.with(|l| l.set(survivors.max(MIN_THRESHOLD)));
        r.collecting = false;
    });
    freed
}

/// Number of registered containers still alive (for tests).
pub fn tracked() -> usize {
    REGISTRY.with(|r| {
        r.borrow()
            .nodes
            .iter()
            .filter(|n| match n {
                Node::List(w) => w.strong_count() > 0,
                Node::Hash(w) => w.strong_count() > 0,
                Node::Record(w) => w.strong_count() > 0,
                Node::Object(w) => w.strong_count() > 0,
                Node::Cell(w) => w.strong_count() > 0,
                Node::Closure(w) => w.strong_count() > 0,
                Node::Tuple(w) => w.strong_count() > 0,
            })
            .count()
    })
}
