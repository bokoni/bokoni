//! Global variables: one slot per name, persisting from one script to the next (REPL).
//!
//! The compiler resolves each global name to a slot index; execution reads the slot
//! directly, without any lookup by name.

use crate::intern::{FxHashMap, FxHashSet, Symbol};
use crate::ir::{JitState, Proto};
use crate::value::Value;
use std::rc::{Rc, Weak};

pub struct GlobalSlot {
    pub name: Symbol,
    /// `None`: name known to the compiler but not defined yet.
    pub value: Option<Value>,
    pub mutable: bool,
    /// Functions whose native compilation depends on the slot's value (direct call of a
    /// global function): invalidated when it changes.
    watchers: Vec<Weak<Proto>>,
}

#[derive(Default)]
pub struct Globals {
    slots: Vec<GlobalSlot>,
    index: FxHashMap<Symbol, u32>,
    /// Declared modules, by qualified name (`Http`, `Http::Client`): their members are
    /// the globals `Http::name`.
    modules: FxHashSet<Symbol>,
    /// Members declared `private` in their module (`Http::parse`).
    private: FxHashSet<Symbol>,
}

impl Globals {
    /// Slot of the name (created empty if needed).
    pub fn slot(&mut self, name: Symbol) -> u32 {
        if let Some(&i) = self.index.get(&name) {
            return i;
        }
        let i = self.slots.len() as u32;
        self.slots.push(GlobalSlot {
            name,
            value: None,
            mutable: true,
            watchers: Vec::new(),
        });
        self.index.insert(name, i);
        i
    }

    pub fn define(&mut self, name: Symbol, value: Value, mutable: bool) {
        let i = self.slot(name);
        self.set_slot(i, value, mutable);
    }

    #[inline]
    pub fn set_slot(&mut self, i: u32, value: Value, mutable: bool) {
        let s = &mut self.slots[i as usize];
        s.value = Some(value);
        s.mutable = mutable;
        self.invalidate(i);
    }

    /// Changes the value of an already defined slot.
    #[inline]
    pub fn assign(&mut self, i: u32, value: Value) {
        self.slots[i as usize].value = Some(value);
        self.invalidate(i);
    }

    /// `proto` (compiled natively, or rejected) depends on the value of slot `i`.
    pub fn watch(&mut self, i: u32, proto: &Rc<Proto>) {
        self.slots[i as usize].watchers.push(Rc::downgrade(proto));
    }

    #[inline]
    fn invalidate(&mut self, i: u32) {
        let ws = &mut self.slots[i as usize].watchers;
        if ws.is_empty() {
            return;
        }
        for w in std::mem::take(ws) {
            if let Some(p) = w.upgrade() {
                // What it depends on changed: its past resumptions no longer count.
                *p.jit.borrow_mut() = JitState::Untried;
                p.deopted.borrow_mut().clear();
            }
        }
    }

    #[inline]
    pub fn at(&self, i: u32) -> &GlobalSlot {
        &self.slots[i as usize]
    }

    pub fn get(&self, name: Symbol) -> Option<Value> {
        self.get_binding(name).map(|(v, _)| v)
    }

    /// Value and mutability of a defined global.
    pub fn add_module(&mut self, name: Symbol) {
        self.modules.insert(name);
    }

    pub fn is_module(&self, name: Symbol) -> bool {
        self.modules.contains(&name)
    }

    /// Visibility of a member of a module (the last declaration wins).
    pub fn set_private(&mut self, name: Symbol, private: bool) {
        if private {
            self.private.insert(name);
        } else {
            self.private.remove(&name);
        }
    }

    pub fn is_private(&self, name: Symbol) -> bool {
        self.private.contains(&name)
    }

    /// Is the name known (declared by a script compiled before)?
    pub fn has_slot(&self, name: Symbol) -> bool {
        self.index.contains_key(&name)
    }

    pub fn get_binding(&self, name: Symbol) -> Option<(Value, bool)> {
        let s = &self.slots[*self.index.get(&name)? as usize];
        Some((s.value.clone()?, s.mutable))
    }

    /// Clears every slot (breaks function → global cycles on teardown).
    pub fn clear(&mut self) {
        for s in &mut self.slots {
            s.value = None;
        }
    }
}
