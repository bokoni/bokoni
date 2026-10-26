//! Positions dans le code source.

/// Intervalle d'octets `[start, end)` dans le source.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    #[inline]
    pub const fn new(start: u32, end: u32) -> Self {
        Span { start, end }
    }

    /// Intervalle couvrant `self` et `other`.
    #[inline]
    pub fn to(self, other: Span) -> Span {
        Span::new(self.start.min(other.start), self.end.max(other.end))
    }

    #[inline]
    pub fn len(self) -> u32 {
        self.end - self.start
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.start == self.end
    }

    #[inline]
    pub fn text(self, src: &str) -> &str {
        &src[self.start as usize..self.end as usize]
    }
}

/// Ligne et colonne (à partir de 1) d'un décalage en octets.
pub fn line_col(src: &str, offset: u32) -> (usize, usize) {
    let offset = (offset as usize).min(src.len());
    let before = &src[..offset];
    let line = before.bytes().filter(|&b| b == b'\n').count() + 1;
    let line_start = before.rfind('\n').map_or(0, |i| i + 1);
    let col = src[line_start..offset].chars().count() + 1;
    (line, col)
}
