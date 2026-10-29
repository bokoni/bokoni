//! Thin reference-counted slices: one pointer wide (`Rc<[T]>` is two: pointer and
//! length). The length lives in the allocation, next to the counts, which keeps
//! `Value` at 16 bytes.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::borrow::Borrow;
use std::cell::Cell;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::rc::Rc;

/// Start of every allocation, followed by the `len` elements.
#[repr(C)]
struct Header {
    strong: Cell<usize>,
    /// Weak references, plus one held collectively by the strong ones (as in `Rc`).
    weak: Cell<usize>,
    len: usize,
}

/// Shared, immutable slice, counted like `Rc<[T]>`.
pub struct ThinRc<T> {
    ptr: NonNull<Header>,
    _owns: PhantomData<T>,
}

/// Weak reference to a `ThinRc`: does not keep the elements alive.
pub struct ThinWeak<T> {
    ptr: NonNull<Header>,
    _owns: PhantomData<T>,
}

/// Offset of the elements from the start of the allocation.
#[inline]
const fn data_offset<T>() -> usize {
    let a = align_of::<T>();
    (size_of::<Header>() + a - 1) & !(a - 1)
}

#[inline]
fn layout<T>(len: usize) -> Layout {
    let size = Layout::array::<T>(len)
        .ok()
        .and_then(|a| a.size().checked_add(data_offset::<T>()))
        .expect("slice too large");
    let align = align_of::<Header>().max(align_of::<T>());
    Layout::from_size_align(size, align)
        .expect("slice too large")
        .pad_to_align()
}

#[inline]
fn header<'a>(ptr: NonNull<Header>) -> &'a Header {
    // SAFETY: `ptr` points to a live header for as long as a strong or weak reference
    // exists, and every caller holds one.
    unsafe { ptr.as_ref() }
}

#[inline]
fn inc(c: &Cell<usize>) {
    let n = c.get().wrapping_add(1);
    if n == 0 {
        // As `Rc`: the count cannot overflow without leaking `usize::MAX` references.
        std::process::abort();
    }
    c.set(n);
}

/// Frees the allocation (elements already dropped) once no reference remains.
#[inline]
fn release_weak<T>(ptr: NonNull<Header>) {
    let h = header(ptr);
    h.weak.set(h.weak.get() - 1);
    if h.weak.get() == 0 {
        let len = h.len;
        // SAFETY: allocated by `ThinRc::alloc` with this layout; no reference is left.
        unsafe { dealloc(ptr.as_ptr() as *mut u8, layout::<T>(len)) }
    }
}

impl<T> ThinRc<T> {
    /// Allocation for `len` elements written by `init`, which must write all of them.
    ///
    /// SAFETY: `init` writes exactly `len` elements at the pointer it receives. If it
    /// panics, the allocation leaks (nothing reads it).
    #[inline]
    unsafe fn alloc(len: usize, init: impl FnOnce(*mut T)) -> Self {
        let lay = layout::<T>(len);
        // SAFETY: the layout is never zero-sized (it holds the header).
        let raw = unsafe { alloc(lay) };
        let Some(ptr) = NonNull::new(raw as *mut Header) else {
            handle_alloc_error(lay)
        };
        // SAFETY: freshly allocated, sized and aligned for the header then the elements.
        unsafe {
            ptr.as_ptr().write(Header {
                strong: Cell::new(1),
                weak: Cell::new(1),
                len,
            });
            init(raw.add(data_offset::<T>()) as *mut T);
        }
        ThinRc {
            ptr,
            _owns: PhantomData,
        }
    }

    #[inline]
    fn data(&self) -> *mut T {
        // SAFETY: the elements start `data_offset` bytes into the allocation.
        unsafe { (self.ptr.as_ptr() as *mut u8).add(data_offset::<T>()) as *mut T }
    }

    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        header(this.ptr).strong.get()
    }

    /// Address of the elements: identifies the allocation.
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        this.data()
    }

    #[inline]
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        a.ptr == b.ptr
    }

    #[inline]
    pub fn downgrade(this: &Self) -> ThinWeak<T> {
        inc(&header(this.ptr).weak);
        ThinWeak {
            ptr: this.ptr,
            _owns: PhantomData,
        }
    }

    /// The elements, mutably, if `this` is the only strong reference. Weak references
    /// are not checked (unlike `Rc::get_mut`): the caller must not upgrade one while it
    /// holds the result.
    #[inline]
    pub(crate) fn sole_mut(this: &mut Self) -> Option<&mut [T]> {
        (Self::strong_count(this) == 1).then(|| {
            // SAFETY: the only strong reference, held mutably; see above for weak ones.
            unsafe { std::slice::from_raw_parts_mut(this.data(), header(this.ptr).len) }
        })
    }
}

impl<T> Deref for ThinRc<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        // SAFETY: `len` initialized elements, alive while a strong reference exists.
        unsafe { std::slice::from_raw_parts(self.data(), header(self.ptr).len) }
    }
}

impl<T> Clone for ThinRc<T> {
    #[inline]
    fn clone(&self) -> Self {
        inc(&header(self.ptr).strong);
        ThinRc {
            ptr: self.ptr,
            _owns: PhantomData,
        }
    }
}

impl<T> Drop for ThinRc<T> {
    #[inline]
    fn drop(&mut self) {
        let h = header(self.ptr);
        h.strong.set(h.strong.get() - 1);
        if h.strong.get() == 0 {
            // SAFETY: the last strong reference: the elements are dropped once. A weak
            // reference cannot upgrade any more (the count is zero).
            unsafe {
                std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(self.data(), h.len))
            };
            release_weak::<T>(self.ptr);
        }
    }
}

impl<T> From<Vec<T>> for ThinRc<T> {
    #[inline]
    fn from(mut v: Vec<T>) -> Self {
        let len = v.len();
        // SAFETY: the `len` elements are moved out of `v`, whose length is then zero
        // (it frees its buffer without dropping them).
        unsafe {
            let r = ThinRc::alloc(len, |dst| {
                std::ptr::copy_nonoverlapping(v.as_ptr(), dst, len)
            });
            v.set_len(0);
            r
        }
    }
}

impl<T: Clone> From<&[T]> for ThinRc<T> {
    #[inline]
    fn from(xs: &[T]) -> Self {
        // SAFETY: writes the `xs.len()` clones in order.
        unsafe {
            ThinRc::alloc(xs.len(), |dst: *mut T| {
                for (i, x) in xs.iter().enumerate() {
                    dst.add(i).write(x.clone());
                }
            })
        }
    }
}

impl<T> FromIterator<T> for ThinRc<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<T>>().into()
    }
}

impl<T: fmt::Debug> fmt::Debug for ThinRc<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<T> ThinWeak<T> {
    #[inline]
    pub fn upgrade(&self) -> Option<ThinRc<T>> {
        let h = header(self.ptr);
        (h.strong.get() > 0).then(|| {
            inc(&h.strong);
            ThinRc {
                ptr: self.ptr,
                _owns: PhantomData,
            }
        })
    }

    #[inline]
    pub fn strong_count(&self) -> usize {
        header(self.ptr).strong.get()
    }
}

impl<T> Clone for ThinWeak<T> {
    #[inline]
    fn clone(&self) -> Self {
        inc(&header(self.ptr).weak);
        ThinWeak {
            ptr: self.ptr,
            _owns: PhantomData,
        }
    }
}

impl<T> Drop for ThinWeak<T> {
    #[inline]
    fn drop(&mut self) {
        release_weak::<T>(self.ptr);
    }
}

/// Shared, immutable string, one pointer wide: the payload of `Value::Str`.
#[derive(Clone)]
pub struct Str(ThinRc<u8>);

impl Str {
    #[inline]
    pub fn as_str(&self) -> &str {
        self
    }

    /// The strings `parts` end to end, in a single allocation.
    pub fn concat(parts: &[&str]) -> Str {
        let len = parts.iter().map(|p| p.len()).sum();
        // SAFETY: writes the `len` bytes of the parts, in order (valid UTF-8).
        Str(unsafe {
            ThinRc::alloc(len, |dst: *mut u8| {
                let mut at = 0;
                for p in parts {
                    std::ptr::copy_nonoverlapping(p.as_ptr(), dst.add(at), p.len());
                    at += p.len();
                }
            })
        })
    }
}

impl Deref for Str {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        // SAFETY: built only from a `str`.
        unsafe { std::str::from_utf8_unchecked(&self.0) }
    }
}

impl From<&str> for Str {
    #[inline]
    fn from(s: &str) -> Str {
        Str(s.as_bytes().into())
    }
}

impl From<String> for Str {
    #[inline]
    fn from(s: String) -> Str {
        Str::from(s.as_str())
    }
}

impl From<&String> for Str {
    #[inline]
    fn from(s: &String) -> Str {
        Str::from(s.as_str())
    }
}

impl From<Rc<str>> for Str {
    #[inline]
    fn from(s: Rc<str>) -> Str {
        Str::from(&*s)
    }
}

impl From<&Rc<str>> for Str {
    #[inline]
    fn from(s: &Rc<str>) -> Str {
        Str::from(&**s)
    }
}

impl From<char> for Str {
    #[inline]
    fn from(c: char) -> Str {
        Str::from(c.encode_utf8(&mut [0; 4]) as &str)
    }
}

impl From<Str> for Rc<str> {
    #[inline]
    fn from(s: Str) -> Rc<str> {
        Rc::from(&*s)
    }
}

impl From<&Str> for Rc<str> {
    #[inline]
    fn from(s: &Str) -> Rc<str> {
        Rc::from(&**s)
    }
}

impl From<Str> for String {
    #[inline]
    fn from(s: Str) -> String {
        s.to_string()
    }
}

impl AsRef<str> for Str {
    #[inline]
    fn as_ref(&self) -> &str {
        self
    }
}

impl Borrow<str> for Str {
    #[inline]
    fn borrow(&self) -> &str {
        self
    }
}

impl PartialEq for Str {
    #[inline]
    fn eq(&self, other: &Str) -> bool {
        ThinRc::ptr_eq(&self.0, &other.0) || **self == **other
    }
}

impl Eq for Str {}

impl PartialEq<str> for Str {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        &**self == other
    }
}

impl PartialEq<&str> for Str {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        &**self == *other
    }
}

impl PartialOrd for Str {
    #[inline]
    fn partial_cmp(&self, other: &Str) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Str {
    #[inline]
    fn cmp(&self, other: &Str) -> std::cmp::Ordering {
        (**self).cmp(&**other)
    }
}

/// Same hash as the `str` (required by `Borrow<str>`).
impl Hash for Str {
    #[inline]
    fn hash<H: Hasher>(&self, h: &mut H) {
        (**self).hash(h)
    }
}

impl fmt::Display for Str {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl fmt::Debug for Str {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_word() {
        assert_eq!(size_of::<ThinRc<u64>>(), 8);
        assert_eq!(size_of::<Str>(), 8);
        assert_eq!(size_of::<Option<Str>>(), 8);
    }

    #[test]
    fn strings() {
        let s = Str::from("héllo");
        assert_eq!(&*s, "héllo");
        assert_eq!(s.len(), 6);
        let t = s.clone();
        assert_eq!(ThinRc::strong_count(&s.0), 2);
        drop(t);
        assert_eq!(ThinRc::strong_count(&s.0), 1);
        assert_eq!(Str::from(""), Str::from(String::new()));
        assert!(Str::from("a").cmp(&Str::from("b")).is_lt());
        assert_eq!(&*Str::concat(&["ab", "", "é"]), "abé");
    }

    #[test]
    fn elements_dropped_once() {
        let marker = Rc::new(());
        let a: ThinRc<Rc<()>> = vec![marker.clone(), marker.clone()].into();
        let b: ThinRc<Rc<()>> = (&a[..]).into();
        assert_eq!(Rc::strong_count(&marker), 5);
        let w = ThinRc::downgrade(&a);
        let a2 = a.clone();
        drop(a);
        assert_eq!(Rc::strong_count(&marker), 5);
        drop(a2);
        assert_eq!(Rc::strong_count(&marker), 3);
        assert!(w.upgrade().is_none());
        assert_eq!(w.strong_count(), 0);
        drop(w);
        drop(b);
        assert_eq!(Rc::strong_count(&marker), 1);
    }

    #[test]
    fn weak_upgrade() {
        let a: ThinRc<String> = vec!["x".to_string()].into();
        let w = ThinRc::downgrade(&a);
        let w2 = w.clone();
        let b = w.upgrade().unwrap();
        assert_eq!(&b[0], "x");
        assert_eq!(w2.strong_count(), 2);
        drop(w);
        drop(a);
        drop(b);
        assert!(w2.upgrade().is_none());
    }

    #[test]
    fn sole_mut_only_alone() {
        let mut a: ThinRc<i32> = vec![1, 2].into();
        ThinRc::sole_mut(&mut a).unwrap()[0] = 5;
        assert_eq!(&a[..], &[5, 2]);
        let b = a.clone();
        assert!(ThinRc::sole_mut(&mut a).is_none());
        drop(b);
        let empty: ThinRc<i32> = Vec::new().into();
        assert!(empty.is_empty());
    }
}
