//! Variables globales : une case par nom, qui persiste d'un script à l'autre (REPL).
//!
//! Le compilateur résout chaque nom global en index de case ; l'exécution lit la case
//! directement, sans recherche par nom.

use crate::intern::{FxHashMap, Symbol};
use crate::value::Value;

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
    }

    #[inline]
    pub fn at(&self, i: u32) -> &GlobalSlot {
        &self.slots[i as usize]
    }

    #[inline]
    pub fn at_mut(&mut self, i: u32) -> &mut GlobalSlot {
        &mut self.slots[i as usize]
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
