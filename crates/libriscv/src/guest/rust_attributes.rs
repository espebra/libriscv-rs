//! Port of `guest/guest_rust_attributes.hpp`: a guest-side map of named,
//! dynamically typed attributes, zero-copy in both directions:
//!
//! ```text
//!   #[repr(C)] pub struct Attr { pub key: String, pub value: Value }
//!   #[repr(C)] pub struct Attributes { entries: Vec<Attr> }   // sorted by key
//! ```
//!
//! A sorted vector provides one allocation and deterministic bytewise key
//! ordering without a shared hasher. Rust's `HashMap` has an unspecified
//! table layout and a process-seeded default hasher.
//!
//! Freeing a map drops each entry, its key and its value, and a value
//! owning a nested `Box<Attributes>` drops that too, through
//! [`GuestType::free_at`]: the host-side counterpart of the guest's drop
//! glue.

use std::collections::BTreeMap;

use super::rust_string::array_bytes;
use super::{
    GuestInit, GuestPtr, GuestRustString, GuestRustVec, GuestType, ToHost, container_error,
    round_up,
};
use crate::error::Result;
use crate::machine::Machine;
use crate::memory::GuestPod;

/// One entry of a guest attribute map, `#[repr(C)] struct Attr`: the key,
/// followed by the value. `#[repr(C)]` fixes the field layout.
#[derive(Debug, Clone, PartialEq)]
pub struct GuestRustAttr<V> {
    pub key: GuestRustString,
    pub value: V,
}

impl<V: GuestPod> GuestRustAttr<V> {
    /// `offsetof(Attr, value)`.
    pub const VALUE_OFFSET: usize = round_up(GuestRustString::SIZE, V::ALIGN);
}

impl<V: GuestPod> GuestPod for GuestRustAttr<V> {
    const ALIGN: usize = if V::ALIGN > GuestRustString::ALIGN {
        V::ALIGN
    } else {
        GuestRustString::ALIGN
    };
    const SIZE: usize = round_up(Self::VALUE_OFFSET + V::SIZE, Self::ALIGN);
    fn write_guest(&self, out: &mut Vec<u8>) {
        let start = out.len();
        self.key.write_guest(out);
        out.resize(start + Self::VALUE_OFFSET, 0);
        self.value.write_guest(out);
        out.resize(start + Self::SIZE, 0);
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Self {
            key: GuestRustString::read_guest(bytes),
            value: V::read_guest(bytes.get(Self::VALUE_OFFSET..).unwrap_or_default()),
        }
    }
}

impl<V: GuestType> GuestType for GuestRustAttr<V> {
    /// Drop the key and the value, recursively.
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        GuestRustString::free_at(m, addr)?;
        V::free_at(m, addr.wrapping_add(Self::VALUE_OFFSET as u64))
    }
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        self.key.free(m)?;
        self.value.free_value(m)
    }
}

impl<V: GuestType + Default> GuestRustAttr<V> {
    /// An entry with a default value.
    pub fn new(m: &mut Machine, key: &str) -> Result<Self> {
        Self::with_key(m, key.as_bytes())
    }
    fn with_key(m: &mut Machine, key: &[u8]) -> Result<Self> {
        Ok(Self {
            key: GuestRustString::from_bytes(m, key)?,
            value: V::default(),
        })
    }
}

/// A guest attribute map, `#[repr(C)] struct Attributes`: a `Vec` of
/// entries kept sorted by key, with the methods of a map.
///
/// This is a host-side copy of the vector's three words: copies are
/// shallow, and changes reach the guest when the copy is written back.
/// [`GuestPtr<GuestRustAttributes<V>>`] changes a map in guest memory in
/// place.
///
/// Validate guest-supplied maps with [`validate`](Self::validate) and
/// [`is_sorted`](Self::is_sorted) before traversal. All accesses are bounded;
/// unsorted keys may cause lookups to miss entries.
#[derive(Debug, Clone, PartialEq)]
pub struct GuestRustAttributes<V> {
    pub entries: GuestRustVec<GuestRustAttr<V>>,
}

impl<V: GuestPod> Default for GuestRustAttributes<V> {
    fn default() -> Self {
        Self {
            entries: GuestRustVec::default(),
        }
    }
}

impl<V> GuestPod for GuestRustAttributes<V> {
    const SIZE: usize = 24;
    const ALIGN: usize = 8;
    fn write_guest(&self, out: &mut Vec<u8>) {
        self.entries.write_guest(out);
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Self {
            entries: GuestRustVec::read_guest(bytes),
        }
    }
}

impl<V: GuestType> GuestType for GuestRustAttributes<V> {
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        GuestRustVec::<GuestRustAttr<V>>::free_at(m, addr)
    }
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        self.free(m)
    }
}

impl<V: GuestType> GuestRustAttributes<V> {
    const ENTRY: u64 = GuestRustAttr::<V>::SIZE as u64;

    pub const fn len(&self) -> usize {
        self.entries.len()
    }
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub const fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    /// Make room for at least `additional` more entries.
    pub fn reserve(&mut self, m: &mut Machine, additional: usize) -> Result<()> {
        self.entries.reserve(m, additional)
    }

    fn entry_addr(&self, index: u64) -> u64 {
        self.entries
            .ptr
            .wrapping_add(index.wrapping_mul(Self::ENTRY))
    }

    /// The entry array must be readable.
    fn check_entries(&self, m: &Machine) -> Result<()> {
        self.entries.view(m).map(|_| ())
    }

    /// The key of entry `index`, borrowed from guest memory.
    fn key_of<'a>(&self, m: &'a Machine, index: u64) -> Result<&'a [u8]> {
        let key: GuestRustString = m.memory.read_value(self.entry_addr(index))?;
        key.as_bytes(m)
    }

    /// The index of the first entry whose key is not less than `key`.
    fn lower_bound(&self, m: &Machine, count: u64, key: &[u8]) -> Result<u64> {
        let (mut lo, mut hi) = (0, count);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.key_of(m, mid)? < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        Ok(lo)
    }

    fn value_ptr(&self, index: u64) -> GuestPtr<V> {
        GuestPtr::new(
            self.entry_addr(index)
                .wrapping_add(GuestRustAttr::<V>::VALUE_OFFSET as u64),
        )
    }

    /// The index of `key`, if it is there.
    fn position(&self, m: &Machine, key: &[u8]) -> Result<Option<u64>> {
        let count = self.entries.len;
        if count == 0 {
            return Ok(None);
        }
        self.check_entries(m)?;
        let i = self.lower_bound(m, count, key)?;
        Ok((i < count && self.key_of(m, i)? == key).then_some(i))
    }

    /// The value of `key`, in place, or `None`.
    pub fn get(&self, m: &Machine, key: &str) -> Result<Option<GuestPtr<V>>> {
        Ok(self.position(m, key.as_bytes())?.map(|i| self.value_ptr(i)))
    }

    pub fn contains_key(&self, m: &Machine, key: &str) -> Result<bool> {
        Ok(self.position(m, key.as_bytes())?.is_some())
    }

    /// Set the value of `key`, constructed from `value`. An existing value
    /// is dropped (freed) and replaced, so unlike `BTreeMap::insert` the
    /// new value comes back, in place.
    pub fn insert<A: GuestInit<V> + ?Sized>(
        &mut self,
        m: &mut Machine,
        key: &str,
        value: &A,
    ) -> Result<GuestPtr<V>>
    where
        V: Default,
    {
        let dst = self.entry_or_default(m, key)?;
        value.assign_at(m, dst.address())?;
        Ok(dst)
    }

    /// The value of `key`, in place, inserting a default one in sorted
    /// position when it is not there (`entry(key).or_default()`).
    pub fn entry_or_default(&mut self, m: &mut Machine, key: &str) -> Result<GuestPtr<V>>
    where
        V: Default,
    {
        let key = key.as_bytes();
        let count = self.entries.len;
        let mut i = count;
        if count != 0 {
            self.check_entries(m)?;
            i = self.lower_bound(m, count, key)?;
            if i < count && self.key_of(m, i)? == key {
                return Ok(self.value_ptr(i));
            }
        }
        // Make room, and build the entry, before moving anything: reserving
        // may move the array.
        self.entries.reserve(m, 1)?;
        let entry = GuestRustAttr::<V>::with_key(m, key)?;
        if i < count {
            self.move_entries(m, i + 1, i, count - i)?;
        }
        // A shallow copy, which hands over the key's allocation.
        m.memory.write_value(self.entry_addr(i), &entry)?;
        self.entries.len += 1;
        Ok(self.value_ptr(i))
    }

    /// Remove `key` and transfer its value to the caller. Frees the key;
    /// the caller owns any allocations retained by the returned value.
    pub fn remove(&mut self, m: &mut Machine, key: &str) -> Result<Option<V>> {
        let Some(i) = self.position(m, key.as_bytes())? else {
            return Ok(None);
        };
        let count = self.entries.len;
        let value = self.value_ptr(i).get(m)?;
        GuestRustString::free_at(m, self.entry_addr(i))?;
        if i + 1 < count {
            self.move_entries(m, i, i + 1, count - i - 1)?;
        }
        self.entries.len -= 1;
        Ok(Some(value))
    }

    /// Drop every entry, keeping the allocation.
    pub fn clear(&mut self, m: &mut Machine) -> Result<()> {
        self.entries.clear(m)
    }

    /// Every entry in key order: the key's bytes, borrowed from guest
    /// memory, and the value in place. Reading a key can fault, so each
    /// item is a `Result`.
    pub fn iter<'a>(
        &self,
        m: &'a Machine,
    ) -> Result<impl Iterator<Item = Result<(&'a [u8], GuestPtr<V>)>> + use<'a, V>> {
        let count = self.entries.len;
        if count != 0 {
            self.check_entries(m)?;
        }
        let this = Self {
            entries: self.entries,
        };
        Ok((0..count).map(move |i| Ok((this.key_of(m, i)?, this.value_ptr(i)))))
    }

    /// Insert every key/value pair, reserving once.
    pub fn extend<K, A, I>(&mut self, m: &mut Machine, pairs: I) -> Result<()>
    where
        K: AsRef<str>,
        A: GuestInit<V>,
        I: IntoIterator<Item = (K, A)>,
        V: Default,
    {
        let pairs = pairs.into_iter();
        self.reserve(m, pairs.size_hint().0)?;
        for (key, value) in pairs {
            self.insert(m, key.as_ref(), &value)?;
        }
        Ok(())
    }

    /// Drop every entry and release the allocation, recursively: what
    /// dropping the map does in the guest.
    pub fn free(&mut self, m: &mut Machine) -> Result<()> {
        self.entries.free(m)
    }

    /// Check that the map is usable and has at most `max_entries` entries.
    pub fn validate(&self, m: &Machine, max_entries: usize) -> Result<()> {
        if self.entries.len > self.entries.cap {
            return Err(container_error("Guest Rust Attributes has len > capacity"));
        }
        if self.entries.len > max_entries as u64 {
            return Err(container_error(
                "Guest Rust Attributes has more entries than allowed",
            ));
        }
        if !self.entries.is_empty() {
            self.check_entries(m)?;
        }
        Ok(())
    }

    /// True when the keys are strictly increasing, the order lookups need.
    pub fn is_sorted(&self, m: &Machine) -> Result<bool> {
        let count = self.entries.len;
        if count < 2 {
            return Ok(true);
        }
        self.check_entries(m)?;
        for i in 1..count {
            if self.key_of(m, i - 1)? >= self.key_of(m, i)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Move `count` entries from `src` to `dst` by copying bytes.
    /// Entries contain no self-pointers and require no relocation.
    fn move_entries(&self, m: &mut Machine, dst: u64, src: u64, count: u64) -> Result<()> {
        let len = array_bytes(count, GuestRustAttr::<V>::SIZE)?;
        let mut bytes = vec![0; len];
        m.memory.memcpy_out(&mut bytes, self.entry_addr(src))?;
        m.memory.memcpy(self.entry_addr(dst), &bytes)
    }
}

impl<V: GuestType + ToHost> GuestRustAttributes<V> {
    /// Copy the whole map to the host. Only for value types that become
    /// host values on their own; walk a tree with [`iter`](Self::iter).
    /// Keys that are not UTF-8 are replaced.
    pub fn to_map(&self, m: &Machine) -> Result<BTreeMap<String, V::Host>> {
        super::host_copy(|| {
            let mut out = BTreeMap::new();
            for entry in self.iter(m)? {
                let (key, value) = entry?;
                super::charge_host_elements::<(String, V::Host)>(1)?;
                let host = value.get(m)?.to_host(m)?;
                out.insert(super::host_string(key)?, host);
            }
            Ok(out)
        })
    }
}

/// A map in guest memory, changed in place.
impl<V: GuestType> GuestPtr<GuestRustAttributes<V>> {
    pub fn len(&self, m: &Machine) -> Result<usize> {
        Ok(self.get(m)?.len())
    }
    pub fn is_empty(&self, m: &Machine) -> Result<bool> {
        Ok(self.get(m)?.is_empty())
    }
    pub fn contains_key(&self, m: &Machine, key: &str) -> Result<bool> {
        self.get(m)?.contains_key(m, key)
    }
    pub fn insert<A: GuestInit<V> + ?Sized>(
        &self,
        m: &mut Machine,
        key: &str,
        value: &A,
    ) -> Result<GuestPtr<V>>
    where
        V: Default,
    {
        self.modify(m, |a, m| a.insert(m, key, value))
    }
    pub fn entry_or_default(&self, m: &mut Machine, key: &str) -> Result<GuestPtr<V>>
    where
        V: Default,
    {
        self.modify(m, |a, m| a.entry_or_default(m, key))
    }
    pub fn remove(&self, m: &mut Machine, key: &str) -> Result<Option<V>> {
        self.modify(m, |a, m| a.remove(m, key))
    }
    pub fn clear(&self, m: &mut Machine) -> Result<()> {
        self.modify(m, |a, m| a.clear(m))
    }
    pub fn extend<K, A, I>(&self, m: &mut Machine, pairs: I) -> Result<()>
    where
        K: AsRef<str>,
        A: GuestInit<V>,
        I: IntoIterator<Item = (K, A)>,
        V: Default,
    {
        self.modify(m, |a, m| a.extend(m, pairs))
    }
}
