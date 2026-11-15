//! Port of `guest/guest_rust_enum.hpp`: a Rust enum with fields, declared
//! `#[repr(C, uN)]` by the guest, in guest memory.
//!
//! Unlike `std::variant` the layout is *requested*, not discovered:
//! `#[repr(C, uN)]` is a tag followed by the union of the variants. A plain
//! `#[repr(Rust)]` enum isn't mirrorable (rustc reorders it and hides the
//! tag in a niche).
//!
//! ```text
//!   #[repr(C, u64)]            libriscv::guest_rust_enum! {
//!   pub enum Value {               pub enum Value {
//!       Nil,                           Nil(()),
//!       Int(i64),                      Int(i64),
//!       Float(f64),                    Float(f64),
//!       Str(String),                   Str(GuestRustString),
//!   }                              }
//!                              }
//!                              type GValue = GuestRustEnum<Value, u64>;
//! ```
//!
//! The tag comes first, and the discriminants are the declaration order.
//! Unit variants are `()`.
//!
//! The tag must be at least as wide as the alignment of the widest variant
//! (a compile-time error otherwise). RFC 2195 puts a `bool` payload at
//! offset 1 behind a `u8` tag, while rustc emits one uniformly aligned
//! payload at offset 8; the readings only differ when the tag is narrower
//! than the payload alignment. Use `u64` for pointer-sized variants: the
//! padding after a `u8` tag would be there anyway.

use std::marker::PhantomData;

use super::{
    GuestInit, GuestPtr, GuestType, GuestVariant, ToHost, VariantAlternative, container_error,
    round_up,
};
use crate::error::Result;
use crate::machine::Machine;
use crate::memory::GuestPod;
use crate::vmcall::{ArgRegs, VmCallArg};

/// The variants of a [`GuestRustEnum`]: a [`GuestVariant`] whose first
/// variant has a default, which is what freeing a [`GuestRustEnum`]
/// leaves behind (a Rust enum has no valueless state). Implemented by
/// [`guest_rust_enum!`](crate::guest_rust_enum).
pub trait RustEnumVariants: GuestVariant {
    /// The first alternative in its default state.
    fn first() -> Self;
}

/// The tag of a [`GuestRustEnum`]: `u8`, `u16`, `u32` or `u64`.
pub trait RustEnumTag: GuestPod + Copy {
    fn to_index(self) -> u64;
    fn from_index(index: u64) -> Self;
}
macro_rules! tags {
    ($($t:ty),*) => {$(
        impl RustEnumTag for $t {
            fn to_index(self) -> u64 {
                u64::from(self)
            }
            fn from_index(index: u64) -> Self {
                index as $t
            }
        }
    )*};
}
tags!(u8, u16, u32, u64);

/// Declare an enum mirroring a guest `#[repr(C, uN)]` enum: one tuple
/// variant per guest variant, in the guest's order, each holding a guest
/// type (`()` for a unit variant). The first one must implement `Default`.
/// Variant types must be distinct.
///
/// With `=> HostName`, it also declares `HostName`, the same enum over
/// [`ToHost`] types, for enums whose variants can all be copied to the host
/// on their own ([`ToHost::to_host`]).
///
/// ```
/// use libriscv::guest::{GuestRustEnum, GuestRustString};
/// libriscv::guest_rust_enum! {
///     #[derive(Debug, Clone, PartialEq)]
///     pub enum Value => HostValue {
///         Nil(()),
///         Int(i64),
///         Str(GuestRustString),
///     }
/// }
/// type GValue = GuestRustEnum<Value, u64>;
/// assert_eq!(<GValue as libriscv::GuestPod>::SIZE, 8 + 24);
/// ```
#[macro_export]
macro_rules! guest_rust_enum {
    ($(#[$meta:meta])* $vis:vis enum $name:ident => $host:ident {
        $first:ident($fty:ty) $(, $variant:ident($ty:ty))* $(,)?
    }) => {
        $crate::guest_rust_enum!(@enum $(#[$meta])* $vis $name; $first($fty) $(, $variant($ty))*);
        /// The host-side copy of the guest enum's active variant.
        #[derive(Debug, Clone, PartialEq)]
        $vis enum $host {
            $first(<$fty as $crate::guest::ToHost>::Host),
            $($variant(<$ty as $crate::guest::ToHost>::Host)),*
        }
        impl $crate::guest::ToHost for $name {
            type Host = $host;
            fn to_host(&self, m: &$crate::Machine) -> $crate::Result<$host> {
                Ok(match self {
                    Self::$first(v) => $host::$first($crate::guest::ToHost::to_host(v, m)?),
                    $(Self::$variant(v) => $host::$variant($crate::guest::ToHost::to_host(v, m)?),)*
                })
            }
        }
    };
    ($(#[$meta:meta])* $vis:vis enum $name:ident {
        $first:ident($fty:ty) $(, $variant:ident($ty:ty))* $(,)?
    }) => {
        $crate::guest_rust_enum!(@enum $(#[$meta])* $vis $name; $first($fty) $(, $variant($ty))*);
    };
    (@enum $(#[$meta:meta])* $vis:vis $name:ident; $first:ident($fty:ty) $(, $variant:ident($ty:ty))*) => {
        $crate::guest_variant! {
            $(#[$meta])*
            $vis enum $name {
                $first($fty),
                $($variant($ty)),*
            }
        }
        impl $crate::guest::RustEnumVariants for $name {
            fn first() -> Self {
                Self::$first(<$fty as ::core::default::Default>::default())
            }
        }
    };
}

/// A Rust enum `#[repr(C, uN)]`: the tag, then the active variant at the
/// payload offset. The value is `None` for an out-of-range tag, which only
/// memory the guest never initialized (or lies about) has.
///
/// The usual way in is the enum itself: `match value.value()`. The
/// generic accessors (`is::<T>`, `get::<T>`) are for code that only knows
/// the variant's type. Copies are shallow.
#[derive(Debug, Clone, PartialEq)]
pub struct GuestRustEnum<E, Tag = u64> {
    value: Option<E>,
    _tag: PhantomData<fn() -> Tag>,
}

impl<E: RustEnumVariants, Tag: RustEnumTag> GuestRustEnum<E, Tag> {
    /// Where the payload starts: right after the tag, which is at least as
    /// wide as the payload's alignment.
    pub const PAYLOAD_OFFSET: usize = {
        assert!(
            Tag::SIZE >= E::UNION_ALIGN,
            "The tag of a guest Rust enum must be at least as wide as the alignment of its \
             widest variant: declare the guest enum #[repr(C, u64)] (or u32/u16 to match)"
        );
        assert!(
            !E::SELF_REFERENCING,
            "A guest Rust enum variant must not point into itself (the C++ containers do)"
        );
        Tag::SIZE
    };
    /// The number of variants.
    pub const VARIANTS: usize = E::COUNT;

    pub const fn new(value: E) -> Self {
        Self {
            value: Some(value),
            _tag: PhantomData,
        }
    }
    /// The active variant, or `None` for an out-of-range tag.
    pub const fn value(&self) -> Option<&E> {
        self.value.as_ref()
    }
    pub const fn value_mut(&mut self) -> Option<&mut E> {
        self.value.as_mut()
    }
    pub fn into_inner(self) -> Option<E> {
        self.value
    }
    /// The discriminant, the index of the active variant, or `None` for an
    /// out-of-range tag.
    pub fn discriminant(&self) -> Option<usize> {
        self.value.as_ref().map(E::index)
    }
    /// The discriminant of the variant holding a `T`.
    pub const fn discriminant_of<T: VariantAlternative<E>>() -> usize {
        T::INDEX
    }
    /// True when the active variant holds a `T`.
    pub fn is<T: VariantAlternative<E>>(&self) -> bool {
        self.get::<T>().is_some()
    }
    /// The `T` of the active variant, or `None` when another one is active.
    pub fn get<T: VariantAlternative<E>>(&self) -> Option<&T> {
        self.value.as_ref().and_then(T::get)
    }
}

impl<E: RustEnumVariants, Tag: RustEnumTag> ToHost for GuestRustEnum<E, Tag>
where
    E: ToHost,
{
    type Host = E::Host;
    fn to_host(&self, m: &Machine) -> Result<E::Host> {
        match &self.value {
            Some(v) => v.to_host(m),
            None => Err(container_error(
                "Guest Rust enum has an out-of-range discriminant",
            )),
        }
    }
}

impl<E: RustEnumVariants, Tag: RustEnumTag> Default for GuestRustEnum<E, Tag> {
    /// The first variant in its default state (it must allocate nothing).
    fn default() -> Self {
        Self::new(E::first())
    }
}

impl<E: RustEnumVariants, Tag: RustEnumTag> From<E> for GuestRustEnum<E, Tag> {
    fn from(value: E) -> Self {
        Self::new(value)
    }
}

impl<E: RustEnumVariants, Tag: RustEnumTag> GuestPod for GuestRustEnum<E, Tag> {
    const ALIGN: usize = if Tag::ALIGN > E::UNION_ALIGN {
        Tag::ALIGN
    } else {
        E::UNION_ALIGN
    };
    const SIZE: usize = round_up(Self::PAYLOAD_OFFSET + E::UNION_SIZE, Self::ALIGN);
    fn write_guest(&self, out: &mut Vec<u8>) {
        let start = out.len();
        let index = self.value.as_ref().map_or(E::COUNT, E::index);
        Tag::from_index(index as u64).write_guest(out);
        out.resize(start + Self::PAYLOAD_OFFSET, 0);
        if let Some(v) = &self.value {
            v.write_alternative(out);
        }
        out.resize(start + Self::SIZE, 0);
    }
    fn read_guest(bytes: &[u8]) -> Self {
        let index = Tag::read_guest(bytes).to_index();
        let value = usize::try_from(index)
            .ok()
            .filter(|&i| i < E::COUNT)
            .and_then(|i| {
                E::read_alternative(i, bytes.get(Self::PAYLOAD_OFFSET..).unwrap_or_default())
            });
        Self {
            value,
            _tag: PhantomData,
        }
    }
}

impl<E: RustEnumVariants, Tag: RustEnumTag> GuestType for GuestRustEnum<E, Tag> {
    /// Drop the active variant and everything it owns, recursively, leaving
    /// the first variant in its default state. Freeing twice
    /// is a no-op, not a double free.
    fn free_at(m: &mut Machine, addr: u64) -> Result<()> {
        let ptr = GuestPtr::<Self>::new(addr);
        if let Some(index) = ptr.discriminant(m)? {
            E::free_alternative(index, m, ptr.payload_addr())?;
        }
        ptr.set(m, &Self::default())
    }
    /// The same, for a host-side value.
    fn free_value(&mut self, m: &mut Machine) -> Result<()> {
        if let Some(v) = &mut self.value {
            v.free_value(m)?;
        }
        *self = Self::default();
        Ok(())
    }
}

/// Construct the enum holding `self` (a shallow copy of the variant, which
/// takes over what it owns); assigning frees the old variant first.
impl<E: RustEnumVariants, Tag: RustEnumTag> GuestInit<GuestRustEnum<E, Tag>> for E {
    fn init_at(&self, m: &mut Machine, addr: u64) -> Result<()> {
        GuestPtr::new(addr).set(m, &GuestRustEnum::<E, Tag>::new(self.clone()))
    }
}

/// An enum in guest memory, read and changed in place.
impl<E: RustEnumVariants, Tag: RustEnumTag> GuestPtr<GuestRustEnum<E, Tag>> {
    fn payload_addr(&self) -> u64 {
        self.address()
            .wrapping_add(GuestRustEnum::<E, Tag>::PAYLOAD_OFFSET as u64)
    }

    /// The discriminant, or `None` for an out-of-range tag.
    pub fn discriminant(&self, m: &Machine) -> Result<Option<usize>> {
        let tag: Tag = m.memory.read_value(self.address())?;
        Ok(usize::try_from(tag.to_index())
            .ok()
            .filter(|&i| i < E::COUNT))
    }
    /// True when the active variant holds a `T`.
    pub fn is<T: VariantAlternative<E>>(&self, m: &Machine) -> Result<bool> {
        Ok(self.discriminant(m)? == Some(T::INDEX))
    }
    /// The `T` of the active variant, in place, or `None` when another one
    /// is active.
    pub fn variant<T: VariantAlternative<E>>(&self, m: &Machine) -> Result<Option<GuestPtr<T>>> {
        Ok(self.is::<T>(m)?.then(|| GuestPtr::new(self.payload_addr())))
    }
    /// Drop the active variant and make the one holding a `T` active,
    /// constructed from `value` (`Option::insert`, for one variant). The
    /// old variant is freed first, so a failed construction leaves a valid
    /// enum (the first variant). Returns the new `T`, in place.
    ///
    /// To replace the value with a whole `E`, use [`GuestPtr::assign`].
    pub fn insert<T, A>(&self, m: &mut Machine, value: &A) -> Result<GuestPtr<T>>
    where
        T: VariantAlternative<E>,
        A: GuestInit<T> + ?Sized,
    {
        <GuestRustEnum<E, Tag> as GuestType>::free_at(m, self.address())?;
        value.init_at(m, self.payload_addr())?;
        m.memory
            .write_value(self.address(), &Tag::from_index(T::INDEX as u64))?;
        Ok(GuestPtr::new(self.payload_addr()))
    }
}

/// By value: pushed onto the stack, and the callee gets its address.
impl<E: RustEnumVariants, Tag: RustEnumTag> VmCallArg for GuestRustEnum<E, Tag> {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        super::pass_by_value(self, m, regs)
    }
}

/// The layout of `#[repr(C, uN)]`: the tag
/// first, then the payload at the first offset that satisfies the
/// alignment of the widest variant.
#[cfg(test)]
mod layout {
    use super::GuestRustEnum;
    use crate::guest::GuestRustString;
    use crate::memory::GuestPod;

    crate::guest_rust_enum! {
        #[derive(Debug, Clone, PartialEq)]
        enum NilInt { Nil(()), Int(u64) }
    }
    crate::guest_rust_enum! {
        #[derive(Debug, Clone, PartialEq)]
        enum NilStr { Nil(()), Str(GuestRustString) }
    }
    crate::guest_rust_enum! {
        #[derive(Debug, Clone, PartialEq)]
        enum Narrow { Nil(()), Small(u16) }
    }
    crate::guest_rust_enum! {
        #[derive(Debug, Clone, PartialEq)]
        enum NilF64 { Nil(()), Float(f64), Str(GuestRustString) }
    }

    type NilIntT = GuestRustEnum<NilInt, u64>;
    type NilStrT = GuestRustEnum<NilStr, u64>;
    type NarrowT = GuestRustEnum<Narrow, u16>;
    type NilF64T = GuestRustEnum<NilF64, u64>;

    // The payload follows the tag: enum { Nil, Int(usize) }
    const _: () = assert!(NilIntT::PAYLOAD_OFFSET == 8);
    const _: () = assert!(NilIntT::SIZE == 16);
    const _: () = assert!(NilIntT::ALIGN == 8);
    // The tag plus the widest variant: enum { Nil, Str(String) }
    const _: () = assert!(NilStrT::SIZE == 8 + GuestRustString::SIZE);
    // A narrow tag is wide enough for a payload of the same alignment
    const _: () = assert!(NarrowT::PAYLOAD_OFFSET == 2);
    const _: () = assert!(NarrowT::SIZE == 4);
    // The payload is padded up to its own alignment
    const _: () = assert!(NilF64T::PAYLOAD_OFFSET == 8);
    const _: () = assert!(NilF64T::SIZE == 8 + GuestRustString::SIZE.div_ceil(8) * 8);

    #[test]
    fn round_trip_keeps_the_variant() {
        let value = NilStrT::new(NilStr::Str(GuestRustString {
            cap: 8,
            ptr: 0x1000,
            len: 3,
        }));
        let mut bytes = Vec::new();
        value.write_guest(&mut bytes);
        assert_eq!(bytes.len(), NilStrT::SIZE);
        assert_eq!(&bytes[..8], &1u64.to_le_bytes());
        assert_eq!(NilStrT::read_guest(&bytes), value);
        // An out-of-range tag reads as no variant at all
        bytes[0] = 7;
        assert_eq!(NilStrT::read_guest(&bytes).discriminant(), None);
    }
}
