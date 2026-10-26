//! Variables globales : une case par nom, qui persiste d'un script à l'autre (REPL).
//!
//! Le compilateur résout chaque nom global en index de case ; l'exécution lit la case
//! directement, sans recherche par nom.

use crate::intern::{FxHashMap, Symbol};
use crate::ir::{JitState, Proto};
use crate::value::Value;
use std::rc::{Rc, Weak};

pub struct GlobalSlot {
    pub name: Symbol,
    /// `None` : nom connu du compilateur mais pas encore défini.
    pub value: Option<Value>,
    pub mutable: bool,
}

#[derive(Default)]
pub struct Globals {
    slots: Vec<GlobalSlot>,
    index: FxHashMap<Symbol, u32>,
    /// Fonctions dont la compilation native dépend de la valeur d'une case (appel direct
    /// d'une fonction globale) : invalidées si la case change.
    watchers: FxHashMap<u32, Vec<Weak<Proto>>>,
}

impl Globals {
    /// Case du nom (créée vide si besoin).
    pub fn slot(&mut self, name: Symbol) -> u32 {
        if let Some(&i) = self.index.get(&name) {
            return i;
        }
        let i = self.slots.len() as u32;
        self.slots.push(GlobalSlot {
            name,
            value: None,
            mutable: true,
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

    /// Change la valeur d'une case déjà définie.
    #[inline]
    pub fn assign(&mut self, i: u32, value: Value) {
        self.slots[i as usize].value = Some(value);
        self.invalidate(i);
    }

    /// `proto` (compilé en natif, ou refusé) dépend de la valeur de la case `i`.
    pub fn watch(&mut self, i: u32, proto: &Rc<Proto>) {
        self.watchers
            .entry(i)
            .or_default()
            .push(Rc::downgrade(proto));
    }

    #[inline]
    fn invalidate(&mut self, i: u32) {
        if self.watchers.is_empty() {
            return;
        }
        if let Some(ws) = self.watchers.remove(&i) {
            for w in ws {
                if let Some(p) = w.upgrade() {
                    *p.jit.borrow_mut() = JitState::Untried;
                }
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

    /// Valeur et caractère modifiable d'une globale définie.
    pub fn get_binding(&self, name: Symbol) -> Option<(Value, bool)> {
        let s = &self.slots[*self.index.get(&name)? as usize];
        Some((s.value.clone()?, s.mutable))
    }

    /// Vide toutes les cases (casse les cycles fonction → globale à la destruction).
    pub fn clear(&mut self) {
        for s in &mut self.slots {
            s.value = None;
        }
    }
}
