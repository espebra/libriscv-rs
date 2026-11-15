//! Port of `guest/guest_rust_box.hpp`: Rust's owning pointers, `Box<T>`,
//! `Box<[T]>` and `Box<str>`, in guest memory.
//!
//! `Box` is the one Rust smart pointer whose layout the language
//! guarantees: for a sized `T` it is one non-null pointer. The two unsized
//! forms are the fat pointers of a slice and a `&str`, `{ ptr, len }`, with
//! ownership. `Box::from_raw` on a host allocation is sound when the guest's
//! global allocator is the native heap, which is what makes ownership cross
//! the boundary in both directions.
//!
//! A null box isn't a `Box<T>`: it is `Option<Box<T>>::None`, the one
//! `Option` whose null representation is documented. A default or freed
//! [`GuestRustBox`] is exactly that, and must not be handed to the guest as
//! a `Box<T>`.

use std::marker::PhantomData;

use super::rust_string::{array_bytes, guest_bytes, rust_dangling, utf8, word};
use super::rust_vec::{elem, element, elements, within_limit};
use super::{
    GuestInit, GuestPtr, GuestRustSlice, GuestRustStr, GuestType, ToHost, container_error,
    guest_alloc, guest_free, nested,
};
use crate::error::Result;
use crate::machine::Machine;
use crate::memory::{GuestPod, GuestSlice};
use crate::vmcall::{ArgRegs, VmCallArg};

/// Rust's `Box<T>` for a sized `T`: one pointer to a `T` on the native
/// heap, or null for `Option<Box<T>>::None`.
///
/// It dereferences to the [`GuestPtr`] of the boxed value, whose methods
/// read and change it in place, and it is passed to a VM call as that
/// pointer (a `Box<T>`, `&T` or `*const T` parameter). This makes it the
/// host's way to own a guest value on the native heap:
///
/// ```ignore
/// let mut text = GuestRustBox::<GuestRustString>::new(m, "hello")?;
/// text.push_str(m, ", world")?;              // in place
/// m.vmcall("takes_string", &text)?;          // fn takes_string(s: &String)
/// text.free(m)?;                             // drop the String and the box
/// ```
///
/// `T` may be the type containing the box (a tree), as the box's layout
/// doesn't depend on it. Copies are shallow: only one of them may free.
///
/// Its own methods are few on purpose (`new`, `from_raw`, `into_raw`,
/// `is_null`, `free`), so that the boxed value's in-place methods are
/// reachable through the dereference without being shadowed.
pub struct GuestRustBox<T> {
    ptr: GuestPtr<T>,
}

impl<T> GuestRustBox<T> {
    /// The box owning the `T` at `addr` (`Box::from_raw`); 0 is
    /// `Option<Box<T>>::None`.
    pub const fn from_raw(addr: u64) -> Self {
        Self {
            ptr: GuestPtr::new(addr),
        }
    }
    /// Give up ownership without freeing anything (`Box::into_raw`): the
    /// caller owns the allocation from here on.
    pub const fn into_raw(self) -> u64 {
        self.ptr.address()
    }
    /// True for `Option<Box<T>>::None`.
    pub const fn is_null(&self) -> bool {
        self.ptr.address() == 0
    }
}

impl<T> std::ops::Deref for GuestRustBox<T> {
    type Target = GuestPtr<T>;
    fn deref(&self) -> &GuestPtr<T> {
        &self.ptr
    }
}

impl<T> Default for GuestRustBox<T> {
    /// `Option<Box<T>>::None`.
    fn default() -> Self {
        Self::from_raw(0)
    }
}
impl<T> Clone for GuestRustBox<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for GuestRustBox<T> {}
impl<T> PartialEq for GuestRustBox<T> {
    fn eq(&self, o: &Self) -> bool {
        self.ptr == o.ptr
    }
}
impl<T> Eq for GuestRustBox<T> {}
impl<T> std::fmt::Debug for GuestRustBox<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GuestRustBox({:#x})", self.ptr.address())
    }
}
impl<T> GuestPod for GuestRustBox<T> {
    const SIZE: usize = 8;
    const ALIGN: usize = 8;
    fn write_guest(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.ptr.address().to_le_bytes());
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Self::from_raw(word(bytes, 0))
    }
}

impl<T: GuestType> GuestRustBox<T> {
    /// `Box::new` of a `T` constructed from `value`: a host value, or a `T`
    /// (a shallow copy that takes over what it owns).
    pub fn new<A: GuestInit<T> + ?Sized>(m: &mut Machine, value: &A) -> Result<Self> {
        if T::SIZE == 0 {
            // Rust doesn't allocate a zero-sized value: the box dangles
            let addr = rust_dangling::<T>();
            value.init_at(m, addr)?;
            return Ok(Self::from_raw(addr));
        }
        let addr = guest_alloc(m, T::SIZE)?;
        if let Err(e) = value.init_at(m, addr) {
            let _ = guest_free(m, addr);
            return Err(e);
        }
        Ok(Self::from_raw(addr))
    }

    /// Drop the boxed value and its allocation, recursively, the way the
    /// guest's drop glue would (`drop(b)`). Freeing a null box does
    /// nothing.
    pub fn free(self, m: &mut Machine) -> Result<()> {
        if !self.is_null() {
            let addr = self.ptr.address();
            nested(|| T::free_at(m, addr))?;
            if T::SIZE != 0 {
                guest_free(m, addr)?;
            }
        }
        Ok(())
    }
}

impl<T: GuestType + ToHost> ToHost for GuestRustBox<T> {
    type Host = T::Host;
    fn to_host(&self, m: &Machine) -> Result<T::Host> {
        if self.is_null() {
            return Err(container_error("Guest Rust Box is null"));
        }
        nested(|| self.ptr.get(m)?.to_host(m))
    }
}

/// Freeing leaves a null box behind.
impl<T: GuestType> GuestType for GuestRustBox<T> {
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::<Self>::new(addr).modify(m, |b, m| std::mem::take(b).free(m))
    }
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        std::mem::take(self).free(m)
    }
}

/// A box is passed as its pointer.
impl<T> VmCallArg for GuestRustBox<T> {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.int(m, self.ptr.address())
    }
}

/// Rust's `Box<[T]>`: an owned `{ ptr, len }`, one allocation of exactly
/// `len` elements.
pub struct GuestRustBoxedSlice<T> {
    pub ptr: u64,
    pub len: u64,
    _type: PhantomData<fn() -> T>,
}

impl<T> GuestRustBoxedSlice<T> {
    /// The boxed slice owning `len` elements at `ptr`, an allocation from
    /// the native heap (`Box::from_raw` of a slice).
    pub const fn from_raw_parts(ptr: u64, len: u64) -> Self {
        Self {
            ptr,
            len,
            _type: PhantomData,
        }
    }
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub const fn len(&self) -> usize {
        self.len as usize
    }
    /// The guest address of the first element.
    pub const fn as_ptr(&self) -> u64 {
        self.ptr
    }
    /// Borrow the whole thing as a guest slice.
    pub const fn as_slice(&self) -> GuestRustSlice<T> {
        GuestRustSlice::from_raw_parts(self.ptr, self.len)
    }
}

impl<T: GuestPod> Default for GuestRustBoxedSlice<T> {
    /// An empty boxed slice: dangling, no allocation.
    fn default() -> Self {
        Self::from_raw_parts(Self::DANGLING, 0)
    }
}
impl<T> Clone for GuestRustBoxedSlice<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for GuestRustBoxedSlice<T> {}
impl<T> PartialEq for GuestRustBoxedSlice<T> {
    fn eq(&self, o: &Self) -> bool {
        (self.ptr, self.len) == (o.ptr, o.len)
    }
}
impl<T> Eq for GuestRustBoxedSlice<T> {}
impl<T> std::fmt::Debug for GuestRustBoxedSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestRustBoxedSlice")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}
impl<T> GuestPod for GuestRustBoxedSlice<T> {
    const SIZE: usize = 16;
    const ALIGN: usize = 8;
    fn write_guest(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.ptr.to_le_bytes());
        out.extend_from_slice(&self.len.to_le_bytes());
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Self::from_raw_parts(word(bytes, 0), word(bytes, 1))
    }
}

impl<T: GuestPod> GuestRustBoxedSlice<T> {
    /// The pointer of an empty boxed slice.
    pub const DANGLING: u64 = rust_dangling::<T>();

    /// Element `index`, in place, or `None` when out of bounds.
    pub fn get(&self, index: usize) -> Option<GuestPtr<T>> {
        element(self.ptr, self.len, index)
    }
    /// The elements, borrowed from guest memory.
    pub fn view<'a>(&self, m: &'a Machine) -> Result<GuestSlice<'a, T>> {
        elements(m, self.ptr, self.len, "Guest Rust Box<[T]> is too large")
    }
    /// Copy the elements out.
    pub fn to_vec(&self, m: &Machine) -> Result<Vec<T>> {
        Ok(self.view(m)?.to_vec())
    }
}

impl<T: GuestType> GuestRustBoxedSlice<T> {
    /// A boxed slice holding `values`, each constructed from a host value,
    /// in an allocation of exactly that many (none when empty).
    pub fn from_slice<A: GuestInit<T>>(m: &mut Machine, values: &[A]) -> Result<Self> {
        let mut b = Self::default();
        if values.is_empty() {
            return Ok(b);
        }
        if T::SIZE == 0 {
            // No allocation for zero-sized elements, as in Rust
            for value in values {
                value.init_at(m, Self::DANGLING)?;
            }
            b.len = values.len() as u64;
            return Ok(b);
        }
        b.ptr = guest_alloc(m, array_bytes(values.len() as u64, T::SIZE)?)?;
        for value in values {
            if let Err(e) = value.init_at(m, b.ptr.wrapping_add(b.len * elem::<T>())) {
                let _ = b.free(m);
                return Err(e);
            }
            b.len += 1;
        }
        Ok(b)
    }

    /// Drop every element and release the allocation, leaving an empty
    /// boxed slice. The pointer, not the length, says whether there is an
    /// allocation: a zero-length one is reachable through
    /// [`from_raw_parts`](Self::from_raw_parts). Zero-sized elements are
    /// never allocated and own nothing, whatever the length says.
    pub fn free(&mut self, m: &mut Machine) -> Result<()> {
        if T::SIZE != 0 && self.ptr != Self::DANGLING {
            if self.len != 0 {
                if !within_limit::<T>(self.len) {
                    return Err(container_error("Guest Rust Box<[T]> is too large"));
                }
                let ptr = self.ptr;
                nested(|| {
                    for i in 0..self.len {
                        T::free_at(m, ptr.wrapping_add(i * elem::<T>()))?;
                    }
                    Ok(())
                })?;
            }
            guest_free(m, self.ptr)?;
        }
        *self = Self::default();
        Ok(())
    }
}

impl<T: GuestType> GuestType for GuestRustBoxedSlice<T> {
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::<Self>::new(addr).modify(m, |b, m| b.free(m))
    }
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        self.free(m)
    }
}

impl<T: GuestPod + ToHost> ToHost for GuestRustBoxedSlice<T> {
    type Host = Vec<T::Host>;
    fn to_host(&self, m: &Machine) -> Result<Vec<T::Host>> {
        nested(|| self.view(m)?.iter().map(|v| v.to_host(m)).collect())
    }
}

/// Construct from host values; assigning frees the old contents and
/// allocates exactly the new ones.
impl<T: GuestType, A: GuestInit<T>> GuestInit<GuestRustBoxedSlice<T>> for [A] {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        let b = GuestRustBoxedSlice::from_slice(m, self)?;
        GuestPtr::new(addr).set(m, &b)
    }
}
impl<T: GuestType, A: GuestInit<T>, const N: usize> GuestInit<GuestRustBoxedSlice<T>> for [A; N] {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestInit::<GuestRustBoxedSlice<T>>::init_at(self.as_slice(), m, addr)
    }
}

/// A boxed slice in guest memory.
impl<T: GuestType> GuestPtr<GuestRustBoxedSlice<T>> {
    pub fn len(&self, m: &Machine) -> Result<usize> {
        Ok(self.get(m)?.len())
    }
    pub fn is_empty(&self, m: &Machine) -> Result<bool> {
        Ok(self.get(m)?.is_empty())
    }
    /// Copy the elements out.
    pub fn to_vec(&self, m: &Machine) -> Result<Vec<T>> {
        self.get(m)?.to_vec(m)
    }
}

/// Rust's `Box<str>`: an owned `{ ptr, len }` of UTF-8 bytes, a `String`
/// without spare capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestRustBoxedStr {
    pub ptr: u64,
    pub len: u64,
}

impl Default for GuestRustBoxedStr {
    /// An empty boxed str: dangling, no allocation.
    fn default() -> Self {
        Self {
            ptr: Self::DANGLING,
            len: 0,
        }
    }
}

impl GuestPod for GuestRustBoxedStr {
    const SIZE: usize = 16;
    const ALIGN: usize = 8;
    fn write_guest(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.ptr.to_le_bytes());
        out.extend_from_slice(&self.len.to_le_bytes());
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Self {
            ptr: word(bytes, 0),
            len: word(bytes, 1),
        }
    }
}

impl GuestRustBoxedStr {
    pub const DANGLING: u64 = rust_dangling::<u8>();

    /// A boxed str holding `s`, in an allocation of exactly its length
    /// (`Box::from`).
    pub fn from_str(m: &mut Machine, s: &str) -> Result<Self> {
        Self::from_bytes(m, s.as_bytes())
    }
    fn from_bytes(m: &mut Machine, bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Ok(Self::default());
        }
        let ptr = guest_alloc(m, bytes.len())?;
        m.memory.memcpy(ptr, bytes)?;
        Ok(Self {
            ptr,
            len: bytes.len() as u64,
        })
    }
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub const fn len(&self) -> usize {
        self.len as usize
    }
    /// The guest address of the bytes.
    pub const fn as_ptr(&self) -> u64 {
        self.ptr
    }
    /// Borrow as a guest `&str`.
    pub const fn as_str(&self) -> GuestRustStr {
        GuestRustStr::from_raw_parts(self.ptr, self.len)
    }
    /// The bytes, borrowed from guest memory.
    pub fn as_bytes<'a>(&self, m: &'a Machine) -> Result<&'a [u8]> {
        guest_bytes(m, self.ptr, self.len, "Guest Rust Box<str> is too large")
    }
    /// The text, borrowed from guest memory, or an error when not UTF-8.
    pub fn to_str<'a>(&self, m: &'a Machine) -> Result<&'a str> {
        utf8(self.as_bytes(m)?)
    }
    /// The text as a host string, with invalid UTF-8 replaced.
    pub fn to_string_lossy(&self, m: &Machine) -> Result<String> {
        Ok(String::from_utf8_lossy(self.as_bytes(m)?).into_owned())
    }

    /// Release the allocation, leaving an empty boxed str.
    pub fn free(&mut self, m: &mut Machine) -> Result<()> {
        if self.len != 0 {
            guest_free(m, self.ptr)?;
        }
        *self = Self::default();
        Ok(())
    }
}

impl GuestType for GuestRustBoxedStr {
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::<Self>::new(addr).modify(m, |b, m| b.free(m))
    }
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        self.free(m)
    }
}

impl ToHost for GuestRustBoxedStr {
    type Host = String;
    fn to_host(&self, m: &Machine) -> Result<String> {
        self.to_string_lossy(m)
    }
}

/// Construct from text; assigning frees the old allocation and makes one of
/// exactly the new length.
macro_rules! boxed_str_sources {
    ($($t:ty),*) => {$(
        impl GuestInit<GuestRustBoxedStr> for $t {
            fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
                let b = GuestRustBoxedStr::from_bytes(m, self.as_bytes())?;
                GuestPtr::new(addr).set(m, &b)
            }
        }
    )*};
}
boxed_str_sources!(str, &str, String);

/// A boxed str in guest memory.
impl GuestPtr<GuestRustBoxedStr> {
    /// The text as a host string, with invalid UTF-8 replaced.
    pub fn to_string_lossy(&self, m: &Machine) -> Result<String> {
        self.get(m)?.to_string_lossy(m)
    }
}
