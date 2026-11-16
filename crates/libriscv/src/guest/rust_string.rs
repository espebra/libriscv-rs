//! Port of `guest/guest_rust_string.hpp` (and the Rust half of
//! `guest/guest_rust_common.hpp`): Rust's `String` and `&str` in guest
//! memory.
//!
//! Mirrors the layouts emitted by rustc for `riscv64gc-unknown-linux-gnu`,
//! verified by `tests/native_rust.rs`. Rust collection layouts are unstable:
//!
//! ```text
//!   Vec<T>  = { cap: usize, ptr: *mut T, len: usize }
//!   String  = Vec<u8> of UTF-8 bytes, the same three words
//!   &[T]    = { ptr: *const T, len: usize }
//!   &str    = &[u8] of UTF-8 bytes, the same two words
//! ```
//!
//! These layouts have no self-pointers and support bytewise moves. Strings
//! have no inline storage or NUL terminator. Empty collections have zero
//! capacity and an aligned, non-null dangling pointer.

use super::{GuestInit, GuestPtr, GuestType, MAX_BYTES, ToHost, container_error, guest_alloc};
use crate::error::Result;
use crate::machine::Machine;
use crate::memory::GuestPod;
use crate::vmcall::{ArgRegs, VmCallArg};

/// The pointer Rust uses for a collection of `T` that has not allocated:
/// dangling but aligned (`NonNull::dangling`).
pub const fn rust_dangling<T: GuestPod>() -> u64 {
    T::ALIGN as u64
}

/// The smallest allocation Rust's `RawVec` makes, in elements: 8 for bytes,
/// 4 up to 1024 bytes, else 1.
pub const fn rust_min_capacity<T: GuestPod>() -> u64 {
    if T::SIZE == 1 {
        8
    } else if T::SIZE <= 1024 {
        4
    } else {
        1
    }
}

/// Rust `RawVec` growth: at least `wanted`, at least double the current
/// capacity, at least the minimum.
pub(crate) fn rawvec_grow(cap: u64, wanted: u64, min: u64) -> u64 {
    wanted.max(cap.saturating_mul(2)).max(min)
}

/// `count * size` as an allocation size, or an error.
pub(crate) fn array_bytes(count: u64, size: usize) -> Result<usize> {
    count
        .checked_mul(size as u64)
        .and_then(|b| usize::try_from(b).ok())
        .ok_or(container_error("Guest Rust collection: too many elements"))
}

/// `len + additional`, or an error.
pub(crate) fn grown_len(len: u64, additional: usize, what: &'static str) -> Result<u64> {
    len.checked_add(additional as u64)
        .ok_or(container_error(what))
}

/// Copy `len` bytes within guest memory (between two blocks of the arena).
pub(crate) fn copy_guest(m: &mut Machine, dst: u64, src: u64, len: u64) -> Result<()> {
    m.memory.memcpy_within(dst, src, len)
}

/// Read two or three guest words out of a mirror's bytes.
pub(crate) fn word(bytes: &[u8], index: usize) -> u64 {
    u64::read_guest(bytes.get(index * 8..).unwrap_or_default())
}

/// `len` bytes at `ptr`, borrowed from guest memory, at most [`MAX_BYTES`].
pub(crate) fn guest_bytes<'a>(
    m: &'a Machine,
    ptr: u64,
    len: u64,
    too_large: &'static str,
) -> Result<&'a [u8]> {
    if len > MAX_BYTES as u64 {
        return Err(container_error(too_large));
    }
    if len == 0 {
        return Ok(&[]);
    }
    m.memory.memview(ptr, len)
}

pub(crate) fn utf8(bytes: &[u8]) -> Result<&str> {
    std::str::from_utf8(bytes).map_err(|_| container_error("Guest Rust string is not valid UTF-8"))
}

/// A borrowed Rust `&str`: a pointer and a length, without ownership.
///
/// This is the layout of a `&str` *stored in memory*, e.g. as a field of a
/// `#[repr(C)]` struct. As a function argument the RISC-V ABI passes the
/// two halves in separate registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestRustStr {
    pub ptr: u64,
    pub len: u64,
}

impl Default for GuestRustStr {
    /// `""`: dangling, not null.
    fn default() -> Self {
        Self {
            ptr: GuestRustString::DANGLING,
            len: 0,
        }
    }
}

impl GuestPod for GuestRustStr {
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
impl GuestType for GuestRustStr {}

impl GuestRustStr {
    /// The `&str` of `len` bytes at `ptr` (`str::from_raw_parts`).
    pub const fn from_raw_parts(ptr: u64, len: u64) -> Self {
        Self { ptr, len }
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
    /// The bytes, borrowed from guest memory.
    pub fn as_bytes<'a>(&self, m: &'a Machine) -> Result<&'a [u8]> {
        guest_bytes(m, self.ptr, self.len, "Guest Rust &str is too large")
    }
    /// Borrow the text from guest memory, returning an error for invalid UTF-8.
    pub fn to_str<'a>(&self, m: &'a Machine) -> Result<&'a str> {
        utf8(self.as_bytes(m)?)
    }
    /// The text as a host string, with invalid UTF-8 replaced.
    pub fn to_string_lossy(&self, m: &Machine) -> Result<String> {
        super::host_string(self.as_bytes(m)?)
    }
}

impl ToHost for GuestRustStr {
    type Host = String;
    fn to_host(&self, m: &Machine) -> Result<String> {
        self.to_string_lossy(m)
    }
}

/// Rust's `alloc::string::String`: a `Vec<u8>` of UTF-8 bytes,
/// `{ cap, ptr, len }`.
///
/// This is a host-side copy of the three words: copies are shallow (they
/// share the bytes, and only one of them may free them), and changes reach
/// the guest when the copy is written back. [`GuestPtr<GuestRustString>`]
/// has the same methods for a string in guest memory.
///
/// Rust code may assume the bytes are valid UTF-8; it is undefined
/// behavior in the guest when they are not. Everything that takes a `&str`
/// keeps it valid; only raw bytes (`&[u8]` as a [`GuestInit`] source) can
/// break that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuestRustString {
    pub cap: u64,
    pub ptr: u64,
    pub len: u64,
}

impl Default for GuestRustString {
    /// `String::new()`: no allocation, a dangling pointer.
    fn default() -> Self {
        Self {
            cap: 0,
            ptr: Self::DANGLING,
            len: 0,
        }
    }
}

impl GuestPod for GuestRustString {
    const SIZE: usize = 24;
    const ALIGN: usize = 8;
    fn write_guest(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.cap.to_le_bytes());
        out.extend_from_slice(&self.ptr.to_le_bytes());
        out.extend_from_slice(&self.len.to_le_bytes());
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Self {
            cap: word(bytes, 0),
            ptr: word(bytes, 1),
            len: word(bytes, 2),
        }
    }
}

impl GuestRustString {
    /// The pointer of a string that has not allocated anything.
    pub const DANGLING: u64 = rust_dangling::<u8>();
    /// The smallest allocation `RawVec` makes for bytes.
    pub const MIN_CAPACITY: u64 = rust_min_capacity::<u8>();

    /// A string holding `s`, its bytes on the native heap in an allocation
    /// of exactly its length (`String::from`).
    pub fn from_str(m: &mut Machine, s: &str) -> Result<Self> {
        Self::from_bytes(m, s.as_bytes())
    }

    /// A string of raw bytes (unchecked), allocated exactly.
    pub(crate) fn from_bytes(m: &mut Machine, bytes: &[u8]) -> Result<Self> {
        let mut string = Self::default();
        string.reserve_exact(m, bytes.len())?;
        if let Err(e) = string.push_bytes(m, bytes) {
            let _ = string.free(m);
            return Err(e);
        }
        Ok(string)
    }

    /// The string of `len` bytes at `ptr`, in an allocation of `cap`
    /// (`String::from_raw_parts`), which must come from the native heap
    /// unless `cap` is 0.
    pub const fn from_raw_parts(ptr: u64, len: u64, cap: u64) -> Self {
        Self { cap, ptr, len }
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
    /// The guest address of the bytes, meaningful when there is a capacity.
    pub const fn as_ptr(&self) -> u64 {
        self.ptr
    }
    /// Borrow the string as a guest `&str`.
    pub const fn as_str(&self) -> GuestRustStr {
        GuestRustStr::from_raw_parts(self.ptr, self.len)
    }

    /// The bytes, borrowed from guest memory.
    pub fn as_bytes<'a>(&self, m: &'a Machine) -> Result<&'a [u8]> {
        if self.len > self.cap {
            return Err(container_error("Guest Rust String has len > capacity"));
        }
        guest_bytes(m, self.ptr, self.len, "Guest Rust String is too large")
    }
    /// The text, borrowed from guest memory, or an error when the bytes
    /// are not UTF-8.
    pub fn to_str<'a>(&self, m: &'a Machine) -> Result<&'a str> {
        utf8(self.as_bytes(m)?)
    }
    /// The text as a host string, with invalid UTF-8 replaced.
    pub fn to_string_lossy(&self, m: &Machine) -> Result<String> {
        super::host_string(self.as_bytes(m)?)
    }

    /// Append `s`, growing the allocation the way the guest would.
    pub fn push_str(&mut self, m: &mut Machine, s: &str) -> Result<()> {
        self.push_bytes(m, s.as_bytes())
    }

    /// Append raw bytes, unchecked.
    pub(crate) fn push_bytes(&mut self, m: &mut Machine, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let total = grown_len(self.len, bytes.len(), "Guest Rust String too large")?;
        self.reserve_total(m, total)?;
        m.memory.memcpy(self.ptr.wrapping_add(self.len), bytes)?;
        self.len = total;
        Ok(())
    }

    /// Truncate to zero length, keeping the allocation.
    pub const fn clear(&mut self) {
        self.len = 0;
    }

    /// Reserve at least `additional` bytes using Rust `RawVec` growth rules,
    /// matching capacities produced by the guest.
    pub fn reserve(&mut self, m: &mut Machine, additional: usize) -> Result<()> {
        let total = grown_len(self.len, additional, "Guest Rust String too large")?;
        self.reserve_total(m, total)
    }

    /// Reserve exactly `additional` bytes if capacity is insufficient
    /// (`String::reserve_exact`).
    pub fn reserve_exact(&mut self, m: &mut Machine, additional: usize) -> Result<()> {
        let total = grown_len(self.len, additional, "Guest Rust String too large")?;
        if total <= self.cap {
            return Ok(());
        }
        self.grow_to(m, total)
    }

    fn reserve_total(&mut self, m: &mut Machine, capacity: u64) -> Result<()> {
        if capacity <= self.cap {
            return Ok(());
        }
        self.grow_to(m, rawvec_grow(self.cap, capacity, Self::MIN_CAPACITY))
    }

    /// Move the bytes to a new allocation of `new_cap` (at least `len`).
    fn grow_to(&mut self, m: &mut Machine, new_cap: u64) -> Result<()> {
        let new_ptr = guest_alloc(m, array_bytes(new_cap, 1)?)?;
        if self.len != 0 {
            // The old pointer is the guest's word: it may not be readable
            if let Err(e) = copy_guest(m, new_ptr, self.ptr, self.len) {
                let _ = super::guest_free(m, new_ptr);
                return Err(e);
            }
        }
        if self.cap != 0 {
            super::guest_free(m, self.ptr)?;
        }
        self.ptr = new_ptr;
        self.cap = new_cap;
        Ok(())
    }

    /// Free the allocation and reset to `String::new()`, matching guest drop behavior.
    pub fn free(&mut self, m: &mut Machine) -> Result<()> {
        if self.cap != 0 {
            super::guest_free(m, self.ptr)?;
        }
        *self = Self::default();
        Ok(())
    }
}

impl GuestType for GuestRustString {
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::<Self>::new(addr).modify(m, |s, m| s.free(m))
    }
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        self.free(m)
    }
}

impl ToHost for GuestRustString {
    type Host = String;
    fn to_host(&self, m: &Machine) -> Result<String> {
        self.to_string_lossy(m)
    }
}

/// Construct a string from raw bytes (unchecked), allocated exactly like
/// `String::from`; assigning reuses the allocation when it is big enough.
impl GuestInit<GuestRustString> for [u8] {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        let s = GuestRustString::from_bytes(m, self)?;
        GuestPtr::new(addr).set(m, &s)
    }
    fn assign_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::<GuestRustString>::new(addr).modify(m, |s, m| {
            s.clear();
            s.push_bytes(m, self)
        })
    }
}

macro_rules! string_sources {
    ($($t:ty),*) => {$(
        impl GuestInit<GuestRustString> for $t {
            fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
                GuestInit::<GuestRustString>::init_at(self.as_bytes(), m, addr)
            }
            fn assign_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
                GuestInit::<GuestRustString>::assign_at(self.as_bytes(), m, addr)
            }
        }
    )*};
}
string_sources!(str, &str, String);

/// A string in guest memory, changed in place.
impl GuestPtr<GuestRustString> {
    pub fn len(&self, m: &Machine) -> Result<usize> {
        Ok(self.get(m)?.len())
    }
    pub fn is_empty(&self, m: &Machine) -> Result<bool> {
        Ok(self.get(m)?.is_empty())
    }
    pub fn capacity(&self, m: &Machine) -> Result<usize> {
        Ok(self.get(m)?.capacity())
    }
    /// The text as a host string, with invalid UTF-8 replaced.
    pub fn to_string_lossy(&self, m: &Machine) -> Result<String> {
        self.get(m)?.to_string_lossy(m)
    }
    /// A copy of the bytes.
    pub fn to_bytes(&self, m: &Machine) -> Result<Vec<u8>> {
        super::host_bytes(self.get(m)?.as_bytes(m)?)
    }
    pub fn push_str(&self, m: &mut Machine, s: &str) -> Result<()> {
        self.modify(m, |string, m| string.push_str(m, s))
    }
    pub fn clear(&self, m: &mut Machine) -> Result<()> {
        self.modify(m, |s, _| {
            s.clear();
            Ok(())
        })
    }
    pub fn reserve(&self, m: &mut Machine, additional: usize) -> Result<()> {
        self.modify(m, |s, m| s.reserve(m, additional))
    }
}

/// By value: pushed onto the stack, and the callee gets its address, like a
/// `&String` argument.
impl VmCallArg for GuestRustString {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        super::pass_by_value(self, m, regs)
    }
}
