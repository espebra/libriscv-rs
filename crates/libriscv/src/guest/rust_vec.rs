//! Port of `guest/guest_rust_vec.hpp`: Rust's `Vec<T>` and `&[T]` in guest
//! memory.

use std::marker::PhantomData;

use super::rust_string::{
    array_bytes, copy_guest, grown_len, rawvec_grow, rust_dangling, rust_min_capacity, word,
};
use super::{
    GuestInit, GuestPtr, GuestType, MAX_BYTES, ToHost, container_error, guest_alloc, guest_free,
    nested, pass_by_value,
};
use crate::error::Result;
use crate::machine::Machine;
use crate::memory::{GuestPod, GuestSlice};
use crate::vmcall::{ArgRegs, VmCallArg};

/// `size_of::<T>()`, never zero (for address arithmetic).
pub(crate) const fn elem<T: GuestPod>() -> u64 {
    if T::SIZE == 0 { 1 } else { T::SIZE as u64 }
}

/// True when `len` elements of `T` fit in [`MAX_BYTES`]. A zero-sized
/// element counts as one byte, so that a loop over a guest-supplied length
/// stays bounded.
pub(crate) fn within_limit<T: GuestPod>(len: u64) -> bool {
    len.checked_mul(elem::<T>())
        .is_some_and(|b| b <= MAX_BYTES as u64)
}

/// The elements `ptr[0..len]`, borrowed from guest memory.
pub(crate) fn elements<'a, T: GuestPod>(
    m: &'a Machine,
    ptr: u64,
    len: u64,
    too_large: &'static str,
) -> Result<GuestSlice<'a, T>> {
    if !within_limit::<T>(len) {
        return Err(container_error(too_large));
    }
    m.memory.memarray(ptr, len, MAX_BYTES as u64)
}

/// The handle of element `index` of `ptr[0..len]`, if in bounds.
pub(crate) fn element<T: GuestPod>(ptr: u64, len: u64, index: usize) -> Option<GuestPtr<T>> {
    let index = index as u64;
    (index < len).then(|| GuestPtr::new(ptr.wrapping_add(index.wrapping_mul(elem::<T>()))))
}

/// A borrowed Rust slice `&[T]`: a pointer and an element count.
///
/// Like [`GuestRustStr`](super::GuestRustStr) this is the layout *stored
/// in memory*. As a function argument the two halves go in separate
/// registers.
pub struct GuestRustSlice<T> {
    pub ptr: u64,
    pub len: u64,
    _type: PhantomData<fn() -> T>,
}

impl<T> GuestRustSlice<T> {
    /// The slice of `len` elements at `ptr` (`slice::from_raw_parts`).
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
}

impl<T: GuestPod> GuestRustSlice<T> {
    /// Element `index`, in place, or `None` when out of bounds.
    pub fn get(&self, index: usize) -> Option<GuestPtr<T>> {
        element(self.ptr, self.len, index)
    }
    /// The elements, borrowed from guest memory.
    pub fn view<'a>(&self, m: &'a Machine) -> Result<GuestSlice<'a, T>> {
        elements(m, self.ptr, self.len, "Guest Rust slice is too large")
    }
    /// Copy the elements out.
    pub fn to_vec(&self, m: &Machine) -> Result<Vec<T>> {
        super::copy_view(&self.view(m)?)
    }
}

impl GuestRustSlice<u8> {
    /// The bytes, borrowed from guest memory.
    pub fn as_bytes<'a>(&self, m: &'a Machine) -> Result<&'a [u8]> {
        super::rust_string::guest_bytes(m, self.ptr, self.len, "Guest Rust slice is too large")
    }
}

impl<T: GuestPod> Default for GuestRustSlice<T> {
    /// `&[]`: dangling, not null.
    fn default() -> Self {
        Self::from_raw_parts(rust_dangling::<T>(), 0)
    }
}
impl<T> Clone for GuestRustSlice<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for GuestRustSlice<T> {}
impl<T> PartialEq for GuestRustSlice<T> {
    fn eq(&self, o: &Self) -> bool {
        (self.ptr, self.len) == (o.ptr, o.len)
    }
}
impl<T> Eq for GuestRustSlice<T> {}
impl<T> std::fmt::Debug for GuestRustSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestRustSlice")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}
impl<T> GuestPod for GuestRustSlice<T> {
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
impl<T> GuestType for GuestRustSlice<T> {}

/// Rust's `alloc::vec::Vec<T>`: `{ cap, ptr, len }`.
///
/// The elements may be guest collections themselves (`Vec<String>`,
/// `Vec<Vec<u32>>`); they are freed along with the vector. This is a
/// host-side copy of the three words: copies are shallow, and changes reach
/// the guest when the copy is written back. [`GuestPtr<GuestRustVec<T>>`]
/// has the same methods for a vector in guest memory.
///
/// An empty vector still has a non-null (dangling) pointer. Only a
/// non-zero capacity means an allocation. A vector of a zero-sized `T`
/// never allocates and stores a capacity of 0 whatever its length, like
/// Rust's `RawVec`.
pub struct GuestRustVec<T> {
    pub cap: u64,
    pub ptr: u64,
    pub len: u64,
    _type: PhantomData<fn() -> T>,
}

impl<T> GuestRustVec<T> {
    /// The vector of `len` elements at `ptr`, in an allocation of `cap`
    /// (`Vec::from_raw_parts`), which must come from the native heap
    /// unless `cap` is 0. Nothing is checked until the vector is used.
    pub const fn from_raw_parts(ptr: u64, len: u64, cap: u64) -> Self {
        Self {
            cap,
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
    pub const fn capacity(&self) -> usize {
        self.cap as usize
    }
    /// The guest address of the first element.
    pub const fn as_ptr(&self) -> u64 {
        self.ptr
    }
    /// Borrow the whole vector as a guest slice.
    pub const fn as_slice(&self) -> GuestRustSlice<T> {
        GuestRustSlice::from_raw_parts(self.ptr, self.len)
    }
}

impl<T: GuestPod> Default for GuestRustVec<T> {
    /// `Vec::new()`: no allocation, a dangling pointer.
    fn default() -> Self {
        Self::from_raw_parts(Self::DANGLING, 0, 0)
    }
}
impl<T> Clone for GuestRustVec<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for GuestRustVec<T> {}
impl<T> PartialEq for GuestRustVec<T> {
    fn eq(&self, o: &Self) -> bool {
        (self.cap, self.ptr, self.len) == (o.cap, o.ptr, o.len)
    }
}
impl<T> Eq for GuestRustVec<T> {}
impl<T> std::fmt::Debug for GuestRustVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestRustVec")
            .field("cap", &self.cap)
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}
impl<T> GuestPod for GuestRustVec<T> {
    const SIZE: usize = 24;
    const ALIGN: usize = 8;
    fn write_guest(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.cap.to_le_bytes());
        out.extend_from_slice(&self.ptr.to_le_bytes());
        out.extend_from_slice(&self.len.to_le_bytes());
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Self::from_raw_parts(word(bytes, 1), word(bytes, 2), word(bytes, 0))
    }
}

impl<T: GuestPod> GuestRustVec<T> {
    /// The pointer of a vector that hasn't allocated anything.
    pub const DANGLING: u64 = rust_dangling::<T>();
    /// The smallest allocation `RawVec` makes for `T`.
    pub const MIN_CAPACITY: u64 = rust_min_capacity::<T>();

    /// Element `index`, in place, or `None` when out of bounds.
    pub fn get(&self, index: usize) -> Option<GuestPtr<T>> {
        element(self.ptr, self.len, index)
    }

    /// Every element, in place.
    pub fn iter(&self) -> Result<impl Iterator<Item = GuestPtr<T>> + use<T>> {
        if !within_limit::<T>(self.len) {
            return Err(container_error("Guest Rust Vec is too large"));
        }
        let (ptr, n) = (self.ptr, self.len);
        Ok((0..n).map(move |i| GuestPtr::new(ptr.wrapping_add(i * elem::<T>()))))
    }

    /// The elements, borrowed from guest memory.
    pub fn view<'a>(&self, m: &'a Machine) -> Result<GuestSlice<'a, T>> {
        if T::SIZE != 0 && self.len > self.cap {
            return Err(container_error("Guest Rust Vec has len > capacity"));
        }
        elements(m, self.ptr, self.len, "Guest Rust Vec is too large")
    }

    /// Copy the elements out: shallow copies, still owned by the vector.
    pub fn to_vec(&self, m: &Machine) -> Result<Vec<T>> {
        super::copy_view(&self.view(m)?)
    }
}

impl GuestRustVec<u8> {
    /// The bytes, borrowed from guest memory.
    pub fn as_bytes<'a>(&self, m: &'a Machine) -> Result<&'a [u8]> {
        if self.len > self.cap {
            return Err(container_error("Guest Rust Vec has len > capacity"));
        }
        self.as_slice().as_bytes(m)
    }
}

impl<T: GuestType> GuestRustVec<T> {
    /// A vector holding `values`, each constructed from a host value, in
    /// one allocation of exactly that many (none when empty), like
    /// `to_vec` and `vec![..]`.
    pub fn from_slice<A: GuestInit<T>>(m: &mut Machine, values: &[A]) -> Result<Self> {
        let mut v = Self::default();
        v.reserve_exact(m, values.len())?;
        if let Err(e) = v.extend_from_slice(m, values) {
            let _ = v.free(m);
            return Err(e);
        }
        Ok(v)
    }

    fn slot(&self, index: u64) -> u64 {
        self.ptr.wrapping_add(index.wrapping_mul(elem::<T>()))
    }

    /// Append an element constructed from `value`: a host value, or a `T`
    /// (a shallow copy that takes over what it owns).
    pub fn push<A: GuestInit<T> + ?Sized>(&mut self, m: &mut Machine, value: &A) -> Result<()> {
        let total = grown_len(self.len, 1, "Guest Rust Vec too large")?;
        self.reserve_total(m, total)?;
        value.init_at(m, self.slot(self.len))?;
        self.len = total;
        Ok(())
    }

    /// Append every value, reserving once.
    pub fn extend_from_slice<A: GuestInit<T>>(
        &mut self,
        m: &mut Machine,
        values: &[A],
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        self.reserve(m, values.len())?;
        for value in values {
            value.init_at(m, self.slot(self.len))?;
            self.len += 1;
        }
        Ok(())
    }

    /// Remove the last element and hand it over: the caller owns what it
    /// owns from here on.
    pub fn pop(&mut self, m: &Machine) -> Result<Option<T>> {
        let Some(last) = self.len.checked_sub(1) else {
            return Ok(None);
        };
        let value = GuestPtr::<T>::new(self.slot(last)).get(m)?;
        self.len = last;
        Ok(Some(value))
    }

    /// Drop the elements from `len` on, keeping the allocation.
    pub fn truncate(&mut self, m: &mut Machine, len: usize) -> Result<()> {
        let len = len as u64;
        if len >= self.len {
            return Ok(());
        }
        // A zero-sized element has no bytes, so it owns nothing to drop
        if T::SIZE != 0 {
            if !within_limit::<T>(self.len) {
                return Err(container_error("Guest Rust Vec is too large"));
            }
            let this = &*self;
            nested(|| {
                for i in len..this.len {
                    T::free_at(m, this.slot(i))?;
                }
                Ok(())
            })?;
        }
        self.len = len;
        Ok(())
    }

    /// Drop every element, keeping the allocation.
    pub fn clear(&mut self, m: &mut Machine) -> Result<()> {
        self.truncate(m, 0)
    }

    /// Grow or shrink to `new_len`, filling with default elements
    /// (`resize_with(new_len, Default::default)`).
    pub fn resize(&mut self, m: &mut Machine, new_len: usize) -> Result<()>
    where
        T: Default,
    {
        let new_len_u64 = new_len as u64;
        if new_len_u64 <= self.len {
            return self.truncate(m, new_len);
        }
        self.reserve_total(m, new_len_u64)?;
        let value = T::default();
        for i in self.len..new_len_u64 {
            value.init_at(m, self.slot(i))?;
        }
        self.len = new_len_u64;
        Ok(())
    }

    /// Make room for at least `additional` more elements, growing the way
    /// Rust's `RawVec` does. The elements move bytewise, which is all a
    /// Rust move is.
    pub fn reserve(&mut self, m: &mut Machine, additional: usize) -> Result<()> {
        let total = grown_len(self.len, additional, "Guest Rust Vec too large")?;
        self.reserve_total(m, total)
    }

    /// Make room for exactly `additional` more elements, if there isn't
    /// room already (`Vec::reserve_exact`).
    pub fn reserve_exact(&mut self, m: &mut Machine, additional: usize) -> Result<()> {
        let total = grown_len(self.len, additional, "Guest Rust Vec too large")?;
        if total <= self.cap {
            return Ok(());
        }
        self.grow_to(m, total)
    }

    pub(crate) fn reserve_total(&mut self, m: &mut Machine, elements: u64) -> Result<()> {
        if elements <= self.cap {
            return Ok(());
        }
        self.grow_to(m, rawvec_grow(self.cap, elements, Self::MIN_CAPACITY))
    }

    /// Move the elements to a new allocation of `new_cap` (at least `len`).
    fn grow_to(&mut self, m: &mut Machine, new_cap: u64) -> Result<()> {
        if T::SIZE == 0 {
            // Room for any number of zero-sized elements, without allocating
            return Ok(());
        }
        let new_ptr = guest_alloc(m, array_bytes(new_cap, T::SIZE)?)?;
        if self.len != 0 {
            // The old pointer is the guest's word: it may not be readable
            let copied = array_bytes(self.len, T::SIZE)
                .and_then(|bytes| copy_guest(m, new_ptr, self.ptr, bytes as u64));
            if let Err(e) = copied {
                let _ = guest_free(m, new_ptr);
                return Err(e);
            }
        }
        if self.cap != 0 {
            guest_free(m, self.ptr)?;
        }
        self.ptr = new_ptr;
        self.cap = new_cap;
        Ok(())
    }

    /// Drop every element and release the allocation, leaving `Vec::new()`:
    /// what dropping the vector does in the guest.
    pub fn free(&mut self, m: &mut Machine) -> Result<()> {
        self.clear(m)?;
        if self.cap != 0 && T::SIZE != 0 {
            guest_free(m, self.ptr)?;
        }
        *self = Self::default();
        Ok(())
    }
}

impl<T: GuestType> GuestType for GuestRustVec<T> {
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::<Self>::new(addr).modify(m, |v, m| v.free(m))
    }
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        self.free(m)
    }
}

/// Every element copied to the host (`Vec<String>` for a `Vec<String>`).
impl<T: GuestPod + ToHost> ToHost for GuestRustVec<T> {
    type Host = Vec<T::Host>;
    fn to_host(&self, m: &Machine) -> Result<Vec<T::Host>> {
        super::host_copy(|| nested(|| super::to_host_all(m, &self.view(m)?)))
    }
}

/// Construct from host values; assigning replaces the contents, reusing
/// the allocation when it is big enough.
impl<T: GuestType, A: GuestInit<T>> GuestInit<GuestRustVec<T>> for [A] {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        let v = GuestRustVec::from_slice(m, self)?;
        GuestPtr::new(addr).set(m, &v)
    }
    fn assign_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::<GuestRustVec<T>>::new(addr).modify(m, |v, m| {
            v.clear(m)?;
            v.extend_from_slice(m, self)
        })
    }
}
impl<T: GuestType, A: GuestInit<T>> GuestInit<GuestRustVec<T>> for Vec<A> {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestInit::<GuestRustVec<T>>::init_at(self.as_slice(), m, addr)
    }
    fn assign_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestInit::<GuestRustVec<T>>::assign_at(self.as_slice(), m, addr)
    }
}
impl<T: GuestType, A: GuestInit<T>, const N: usize> GuestInit<GuestRustVec<T>> for [A; N] {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestInit::<GuestRustVec<T>>::init_at(self.as_slice(), m, addr)
    }
    fn assign_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestInit::<GuestRustVec<T>>::assign_at(self.as_slice(), m, addr)
    }
}

/// A vector in guest memory, changed in place.
impl<T: GuestType> GuestPtr<GuestRustVec<T>> {
    pub fn len(&self, m: &Machine) -> Result<usize> {
        Ok(self.get(m)?.len())
    }
    pub fn is_empty(&self, m: &Machine) -> Result<bool> {
        Ok(self.get(m)?.is_empty())
    }
    pub fn capacity(&self, m: &Machine) -> Result<usize> {
        Ok(self.get(m)?.capacity())
    }
    /// Copy the elements out.
    pub fn to_vec(&self, m: &Machine) -> Result<Vec<T>> {
        self.get(m)?.to_vec(m)
    }
    pub fn push<A: GuestInit<T> + ?Sized>(&self, m: &mut Machine, value: &A) -> Result<()> {
        self.modify(m, |v, m| v.push(m, value))
    }
    pub fn extend_from_slice<A: GuestInit<T>>(&self, m: &mut Machine, values: &[A]) -> Result<()> {
        self.modify(m, |v, m| v.extend_from_slice(m, values))
    }
    /// Remove the last element and hand it over.
    pub fn pop(&self, m: &mut Machine) -> Result<Option<T>> {
        self.modify(m, |v, m| v.pop(m))
    }
    pub fn truncate(&self, m: &mut Machine, len: usize) -> Result<()> {
        self.modify(m, |v, m| v.truncate(m, len))
    }
    pub fn clear(&self, m: &mut Machine) -> Result<()> {
        self.modify(m, |v, m| v.clear(m))
    }
    pub fn resize(&self, m: &mut Machine, new_len: usize) -> Result<()>
    where
        T: Default,
    {
        self.modify(m, |v, m| v.resize(m, new_len))
    }
    pub fn reserve(&self, m: &mut Machine, additional: usize) -> Result<()> {
        self.modify(m, |v, m| v.reserve(m, additional))
    }
}

/// By value: pushed onto the stack, and the callee gets its address, like a
/// `&Vec<T>` argument.
impl<T: GuestType> VmCallArg for GuestRustVec<T> {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        pass_by_value(self, m, regs)
    }
}
