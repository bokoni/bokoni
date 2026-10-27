//! Identifier interning: each name becomes a `Symbol` (u32) comparable in O(1).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(pub u32);

/// Symbol table.
///
/// All strings are concatenated into a single arena (no allocation per name).
///
/// - Short names (1 to 8 bytes, almost every identifier): the name fits in a zero-padded
///   `u64`. An identifier never contains a NUL byte, so this word identifies it exactly:
///   lookup is one integer multiplication and comparison, without reading the arena.
/// - Long names: a table storing 32 bits of hash next to the symbol; the arena is read
///   only when the hashes are equal.
pub struct Interner {
    arena: String,
    /// `Symbol` → (start, length) in `arena`.
    spans: Vec<(u32, u32)>,
    /// Short names: `(word, symbol)`, word `0` = free. Size: power of 2.
    short: Vec<(u64, u32)>,
    short_shift: u32,
    /// Long names: `(hash << 32) | symbol`, `EMPTY` if free. Size: power of 2.
    table: Vec<u64>,
    /// `64 - log2(table.len())`: the index is taken from the high bits of the hash.
    shift: u32,
    /// Number of long names (load factor of `table`).
    long_count: usize,
}

const EMPTY: u64 = u64::MAX;

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

impl Interner {
    pub fn new() -> Self {
        const INITIAL: usize = 256;
        Interner {
            arena: String::new(),
            spans: Vec::new(),
            short: vec![(0, 0); INITIAL],
            short_shift: 64 - INITIAL.trailing_zeros(),
            table: vec![EMPTY; INITIAL],
            shift: 64 - INITIAL.trailing_zeros(),
            long_count: 0,
        }
    }

    /// Zero-padded 8-byte word, if `s` is a short name without a NUL byte.
    #[inline]
    fn short_word(s: &str) -> Option<u64> {
        let b = s.as_bytes();
        if b.is_empty() || b.len() > 8 || b.contains(&0) {
            return None;
        }
        let mut buf = [0u8; 8];
        buf[..b.len()].copy_from_slice(b);
        Some(u64::from_le_bytes(buf))
    }

    #[inline]
    fn short_slot(&self, word: u64) -> usize {
        (word.wrapping_mul(SEED) >> self.short_shift) as usize
    }

    /// Looks up a short name; returns the symbol or the free slot.
    #[inline]
    fn find_short(&self, word: u64) -> Result<Symbol, usize> {
        let mask = self.short.len() - 1;
        let mut i = self.short_slot(word);
        loop {
            let (w, sym) = self.short[i];
            if w == word {
                return Ok(Symbol(sym));
            }
            if w == 0 {
                return Err(i);
            }
            i = (i + 1) & mask;
        }
    }

    fn push_str(&mut self, s: &str) -> u32 {
        let sym = self.spans.len() as u32;
        self.spans.push((self.arena.len() as u32, s.len() as u32));
        self.arena.push_str(s);
        sym
    }

    fn grow_short(&mut self) {
        let old = std::mem::take(&mut self.short);
        self.short = vec![(0, 0); old.len() * 2];
        self.short_shift -= 1;
        let mask = self.short.len() - 1;
        for (w, sym) in old {
            if w != 0 {
                let mut i = self.short_slot(w);
                while self.short[i].0 != 0 {
                    i = (i + 1) & mask;
                }
                self.short[i] = (w, sym);
            }
        }
    }

    #[inline]
    fn slot(&self, hash: u64) -> usize {
        (hash >> self.shift) as usize
    }

    #[inline]
    fn str_of(&self, sym: u32) -> &str {
        let (start, len) = self.spans[sym as usize];
        &self.arena[start as usize..(start + len) as usize]
    }

    /// Looks up `s`; returns the symbol or the free slot to insert it into.
    #[inline]
    fn find(&self, s: &str, hash: u64) -> Result<Symbol, usize> {
        let mask = self.table.len() - 1;
        let tag = hash as u32;
        let mut i = self.slot(hash);
        loop {
            let e = self.table[i];
            if e == EMPTY {
                return Err(i);
            }
            if (e >> 32) as u32 == tag && self.str_of(e as u32) == s {
                return Ok(Symbol(e as u32));
            }
            i = (i + 1) & mask;
        }
    }

    #[inline]
    pub fn intern(&mut self, s: &str) -> Symbol {
        if let Some(word) = Self::short_word(s) {
            let slot = match self.find_short(word) {
                Ok(sym) => return sym,
                Err(slot) => slot,
            };
            let sym = self.push_str(s);
            self.short[slot] = (word, sym);
            // Max load factor 1/2 (approximated by the total number of names).
            if (self.spans.len() - self.long_count) * 2 > self.short.len() {
                self.grow_short();
            }
            return Symbol(sym);
        }
        self.intern_long(s)
    }

    fn intern_long(&mut self, s: &str) -> Symbol {
        let hash = fx_hash(s.as_bytes());
        let slot = match self.find(s, hash) {
            Ok(sym) => return sym,
            Err(slot) => slot,
        };
        let sym = self.push_str(s);
        self.table[slot] = ((hash as u32 as u64) << 32) | sym as u64;
        self.long_count += 1;
        if self.long_count * 2 > self.table.len() {
            self.grow();
        }
        Symbol(sym)
    }

    fn grow(&mut self) {
        let len = self.table.len() * 2;
        self.table = vec![EMPTY; len];
        self.shift -= 1;
        let mask = len - 1;
        for sym in 0..self.spans.len() as u32 {
            if Self::short_word(self.str_of(sym)).is_some() {
                continue;
            }
            let hash = fx_hash(self.str_of(sym).as_bytes());
            let mut i = self.slot(hash);
            while self.table[i] != EMPTY {
                i = (i + 1) & mask;
            }
            self.table[i] = ((hash as u32 as u64) << 32) | sym as u64;
        }
    }

    /// Looks up a symbol without adding it.
    pub fn get(&self, s: &str) -> Option<Symbol> {
        match Self::short_word(s) {
            Some(word) => self.find_short(word).ok(),
            None => self.find(s, fx_hash(s.as_bytes())).ok(),
        }
    }

    #[inline]
    pub fn resolve(&self, sym: Symbol) -> &str {
        self.str_of(sym.0)
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

/// FxHash of a byte sequence, one 8-byte word at a time.
#[inline]
pub fn fx_hash(bytes: &[u8]) -> u64 {
    let mut h = FxHasher::default();
    h.write(bytes);
    h.add(bytes.len() as u64);
    h.hash
}

/// FxHash hashing (rustc): much faster than SipHash for short keys.
#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for w in words {
            self.add(u64::from_le_bytes(*w));
        }
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

pub type FxBuild = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = HashMap<K, V, FxBuild>;
pub type FxHashSet<T> = std::collections::HashSet<T, FxBuild>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_and_resolve() {
        let mut i = Interner::new();
        let names: Vec<String> = (0..10_000).map(|n| format!("name{n}")).collect();
        let syms: Vec<Symbol> = names.iter().map(|n| i.intern(n)).collect();
        for (n, s) in names.iter().zip(&syms) {
            assert_eq!(i.intern(n), *s);
            assert_eq!(i.resolve(*s), n);
            assert_eq!(i.get(n), Some(*s));
        }
        assert_eq!(i.len(), 10_000);
        assert_eq!(i.get("absent"), None);
        // Long names (> 8 bytes) and names containing a NUL byte: general table.
        let long = i.intern("a_fairly_long_name");
        assert_eq!(i.intern("a_fairly_long_name"), long);
        assert_eq!(i.resolve(long), "a_fairly_long_name");
        let nul = i.intern("a\0b");
        assert_ne!(nul, i.intern("a"));
        assert_eq!(i.intern("a\0b"), nul);
        let empty = i.intern("");
        assert_eq!(i.resolve(empty), "");
    }
}
