//! Fuzz host access to Rust guest strings, vectors, boxes, enums, and attribute
//! maps. Input controls pointers, lengths, capacities, and tags, including cyclic
//! and nested attribute trees. Allocations use the native heap; no compiled
//! Rust guest is required.

use libriscv::guest::{
    GuestPtr, GuestRustAttributes, GuestRustBox, GuestRustBoxedSlice, GuestRustBoxedStr,
    GuestRustEnum, GuestRustSlice, GuestRustStr, GuestRustString, GuestRustVec,
};
use libriscv::{Machine, MachineOptions};

/// Where the fuzz input is placed. Every operation works on the object at
/// the start of it; its pointers usually point back into it. The
/// dictionary has this address and a few offsets from it.
pub const OBJECT: u64 = 0x10000;
/// At most this much input goes into guest memory.
const OBJECT_MAX: usize = 0x10000;
/// The native heap system calls (unused by the harness, but they set up
/// the arena).
const HEAP_SYSCALLS: usize = 200;

libriscv::guest_rust_enum! {
    /// The examples' attribute value, `#[repr(C, u64)] enum Value`: a
    /// group holds a nested map, a list holds values.
    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        Nil(()),
        Int(i64),
        Float(f64),
        Bool(bool),
        Str(GuestRustString),
        Group(GuestRustBox<GuestRustAttributes<GuestRustEnum<Value, u64>>>),
        List(GuestRustBox<GuestRustVec<GuestRustEnum<Value, u64>>>),
    }
}
type RustValue = GuestRustEnum<Value, u64>;
type Attrs = GuestRustAttributes<RustValue>;

libriscv::guest_rust_enum! {
    /// A flat value with a host mirror: maps of these copy to the host
    /// (`to_map`).
    #[derive(Debug, Clone, PartialEq)]
    pub enum Flat => HostFlat {
        Nil(()),
        Int(i64),
        Str(GuestRustString),
    }
}
type FlatValue = GuestRustEnum<Flat, u64>;
type FlatMap = GuestPtr<GuestRustAttributes<FlatValue>>;

libriscv::guest_rust_enum! {
    /// `#[repr(C, u32)] enum Small { A, B(u32), C(u16) }`: a 4-byte tag.
    #[derive(Debug, Clone, PartialEq)]
    pub enum Small {
        A(()),
        B(u32),
        C(u16),
    }
}
type SmallValue = GuestPtr<GuestRustEnum<Small, u32>>;

type Str = GuestPtr<GuestRustString>;
type Words = GuestPtr<GuestRustVec<u32>>;
type Strings = GuestPtr<GuestRustVec<GuestRustString>>;
type Values = GuestPtr<RustValue>;
type Map = GuestPtr<Attrs>;
type ListBox = GuestPtr<GuestRustBox<GuestRustVec<RustValue>>>;

/// A string longer than any small capacity, so it reallocates.
const LONG: &str = "a string long enough to make the vector grow more than once";

/// The collection types [`operate`] views the object as.
const TYPES: u8 = 12;

/// One operation on the object at [`OBJECT`], viewed as one of the
/// collection types: `(op >> 3) % TYPES` picks the type, `op & 7` the
/// operation. Errors are expected; panics (and stack overflows) are bugs.
fn operate(m: &mut Machine, op: u8) -> libriscv::Result<()> {
    use libriscv::guest::{GuestType, ToHost};
    let arg = u64::from(op);
    match (op >> 3) % TYPES {
        0 => {
            let s = Str::new(OBJECT);
            match op & 7 {
                0 => {
                    let _ = s.to_string_lossy(m)?;
                }
                1 => {
                    let _ = s.to_bytes(m)?;
                }
                2 => s.push_str(m, "short")?,
                3 => s.push_str(m, LONG)?,
                4 => s.reserve(m, (arg & 63) as usize)?,
                5 => {
                    let _ = (s.len(m)?, s.capacity(m)?);
                }
                6 => s.assign(m, LONG)?,
                _ => s.free(m)?,
            }
        }
        1 => {
            let v = Words::new(OBJECT);
            match op & 7 {
                0 => {
                    let _ = v.to_vec(m)?;
                }
                1 => v.push(m, &(arg as u32))?,
                2 => {
                    let _ = v.pop(m)?;
                }
                3 => v.truncate(m, (arg & 3) as usize)?,
                4 => v.resize(m, (arg & 31) as usize)?,
                5 => v.reserve(m, (arg & 63) as usize)?,
                6 => v.extend_from_slice(m, &[1u32, 2, 3])?,
                _ => v.free(m)?,
            }
        }
        2 => {
            let v = Strings::new(OBJECT);
            match op & 7 {
                0 => {
                    let _ = v.to_vec(m)?;
                }
                1 => {
                    let s = GuestRustString::from_str(m, LONG)?;
                    v.push(m, &s)?;
                }
                2 => {
                    if let Some(mut s) = v.pop(m)? {
                        s.free(m)?;
                    }
                }
                3 => v.truncate(m, (arg & 1) as usize)?,
                4 => v.reserve(m, (arg & 7) as usize)?,
                5 => {
                    let _ = v.get(m)?.to_host(m)?;
                }
                6 => v.clear(m)?,
                _ => v.free(m)?,
            }
        }
        3 => {
            // Borrowed views at the object: a &str, a &[u16] and its bytes, and
            // boxed slices and strs.
            let raw = GuestPtr::<[u64; 2]>::new(OBJECT).get(m)?;
            match op & 7 {
                0 => {
                    let _ = GuestRustStr::from_raw_parts(raw[0], raw[1]).to_str(m)?;
                }
                1 => {
                    let _ = GuestRustStr::from_raw_parts(raw[0], raw[1]).to_string_lossy(m)?;
                }
                2 => {
                    let _ = GuestRustSlice::<u16>::from_raw_parts(raw[0], raw[1]).to_vec(m)?;
                }
                3 => {
                    let _ = GuestRustSlice::<u8>::from_raw_parts(raw[0], raw[1]).as_bytes(m)?;
                }
                4 => {
                    let b = GuestPtr::<GuestRustBoxedSlice<u64>>::new(OBJECT).get(m)?;
                    let _ = b.to_vec(m)?;
                }
                5 => GuestPtr::<GuestRustBoxedSlice<u64>>::new(OBJECT).free(m)?,
                6 => {
                    let b = GuestPtr::<GuestRustBoxedStr>::new(OBJECT).get(m)?;
                    let _ = b.to_string_lossy(m)?;
                }
                _ => GuestPtr::<GuestRustBoxedStr>::new(OBJECT).free(m)?,
            }
        }
        4 => {
            let v = Values::new(OBJECT);
            match op & 7 {
                0 => {
                    let _ = v.discriminant(m)?;
                }
                1 => {
                    if let Some(s) = v.variant::<GuestRustString>(m)? {
                        let _ = s.to_string_lossy(m)?;
                    }
                }
                2 => {
                    if let Some(g) = v.variant::<GuestRustBox<Attrs>>(m)? {
                        let map = g.get(m)?.get(m)?;
                        map.validate(m, 64)?;
                        let _ = map.is_sorted(m)?;
                    }
                }
                3 => {
                    if let Some(l) = v.variant::<GuestRustBox<GuestRustVec<RustValue>>>(m)? {
                        let list = l.get(m)?.get(m)?;
                        let _ = list.view(m)?.len();
                    }
                }
                4 => {
                    let _ = v.insert::<i64, _>(m, &(arg as i64))?;
                }
                5 => {
                    let _ = v.insert::<GuestRustString, _>(m, LONG)?;
                }
                6 => {
                    let list =
                        GuestRustVec::<RustValue>::from_slice(m, &[RustValue::new(Value::Int(1))])?;
                    let boxed = GuestRustBox::new(m, &list)?;
                    let _ = v.insert::<GuestRustBox<GuestRustVec<RustValue>>, _>(m, &boxed)?;
                }
                _ => v.free(m)?,
            }
        }
        5 => {
            let map = Map::new(OBJECT);
            match op & 7 {
                0 => {
                    let attrs = map.get(m)?;
                    attrs.validate(m, 64)?;
                    let _ = attrs.is_sorted(m)?;
                }
                1 => {
                    let _ = map.get(m)?.get(m, "key")?;
                }
                2 => {
                    let _ = map.contains_key(m, LONG)?;
                }
                3 => {
                    let _ = map.insert(m, "key", &RustValue::new(Value::Int(arg as i64)))?;
                }
                4 => {
                    let _ = map.entry_or_default(m, LONG)?;
                }
                5 => {
                    if let Some(mut old) = map.remove(m, "key")? {
                        old.free_value(m)?;
                    }
                }
                6 => map.clear(m)?,
                _ => map.get(m)?.free(m)?,
            }
        }
        6 => {
            // A boxed list: the cycle- and depth-prone shape.
            let list = ListBox::new(OBJECT);
            match op & 7 {
                0 | 1 => {
                    let b = list.get(m)?;
                    if !b.is_null() {
                        let _ = b.get(m)?.view(m)?.len();
                    }
                }
                _ => list.free(m)?,
            }
        }
        7 => {
            let v = SmallValue::new(OBJECT);
            match op & 7 {
                0 => {
                    let _ = v.discriminant(m)?;
                }
                1 => {
                    if let Some(b) = v.variant::<u32>(m)? {
                        let _ = b.get(m)?;
                    }
                }
                2 => {
                    let _ = v.get(m)?.get::<u16>();
                }
                3 => {
                    let _ = v.insert::<u16, _>(m, &(arg as u16))?;
                }
                4 => {
                    let _ = v.insert::<u32, _>(m, &(arg as u32))?;
                }
                _ => v.free(m)?,
            }
        }
        8 => {
            let map = FlatMap::new(OBJECT);
            match op & 7 {
                0 | 1 => {
                    let _ = map.get(m)?.to_map(m)?;
                }
                2 => {
                    let attrs = map.get(m)?;
                    for entry in attrs.iter(m)? {
                        let (_, value) = entry?;
                        let _ = value.get(m)?.to_host(m)?;
                    }
                }
                3 => map.extend(
                    m,
                    [
                        ("a", FlatValue::new(Flat::Int(1))),
                        ("key", FlatValue::new(Flat::Nil(()))),
                        ("z", FlatValue::new(Flat::Int(arg as i64))),
                    ],
                )?,
                4 => {
                    let _ = map.insert(m, LONG, &FlatValue::new(Flat::Int(2)))?;
                }
                5 => {
                    if let Some(mut old) = map.remove(m, "a")? {
                        old.free_value(m)?;
                    }
                }
                6 => map.clear(m)?,
                _ => map.get(m)?.free(m)?,
            }
        }
        9 => {
            let b = GuestPtr::<GuestRustBoxedSlice<GuestRustString>>::new(OBJECT);
            match op & 7 {
                0 | 1 => {
                    let _ = b.get(m)?.to_host(m)?;
                }
                2 => {
                    let _ = b.len(m)?;
                }
                3 | 4 => {
                    let strings = [GuestRustString::from_str(m, LONG)?];
                    b.assign(m, &strings[..])?;
                }
                _ => b.free(m)?,
            }
        }
        10 => {
            let v = GuestPtr::<GuestRustVec<GuestRustVec<u8>>>::new(OBJECT);
            match op & 7 {
                0 | 1 => {
                    let _ = v.get(m)?.to_host(m)?;
                }
                2 => {
                    let bytes = GuestRustVec::<u8>::from_slice(m, LONG.as_bytes())?;
                    v.push(m, &bytes)?;
                }
                3 => {
                    if let Some(mut last) = v.pop(m)? {
                        last.free(m)?;
                    }
                }
                4 => v.truncate(m, (arg & 1) as usize)?,
                5 => v.reserve(m, (arg & 7) as usize)?,
                6 => v.clear(m)?,
                _ => v.free(m)?,
            }
        }
        _ => {
            // Free whatever the object is, as each type: the walks of the
            // guest's drop glue.
            match op & 7 {
                0 => Values::new(OBJECT).free(m)?,
                1 => Map::new(OBJECT).free(m)?,
                2 => Strings::new(OBJECT).free(m)?,
                3 => ListBox::new(OBJECT).free(m)?,
                4 => GuestPtr::<GuestRustBox<Attrs>>::new(OBJECT).free(m)?,
                5 => GuestPtr::<GuestRustVec<RustValue>>::new(OBJECT).free(m)?,
                6 => GuestPtr::<GuestRustBoxedSlice<RustValue>>::new(OBJECT).free(m)?,
                _ => GuestPtr::<GuestRustBox<RustValue>>::new(OBJECT).free(m)?,
            }
        }
    }
    Ok(())
}

/// A machine with a native heap and `memory` at [`OBJECT`].
fn rust_machine(memory: &[u8]) -> Option<Machine> {
    let memory = memory.get(..OBJECT_MAX).unwrap_or(memory);
    let options = MachineOptions {
        memory_max: 16 << 20,
        ..crate::options()
    };
    let mut m = Machine::new([].into(), options).ok()?;
    crate::quiet(&mut m);
    m.setup_native_heap(HEAP_SYSCALLS, crate::HEAP, crate::HEAP_SIZE)
        .ok()?;
    m.copy_to_guest(OBJECT, memory).ok()?;
    Some(m)
}

/// `data[..4]` are four operations (see [`operate`]) on the object at
/// [`OBJECT`]; `data[4..]` is the guest memory there.
pub fn fuzz_rust_collections(data: &[u8]) {
    let Some((ops, memory)) = data.split_first_chunk::<4>() else {
        return;
    };
    let Some(mut m) = rust_machine(memory) else {
        return;
    };
    for &op in ops {
        let _ = operate(&mut m, op);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collections_are_built_in_guest_memory() {
        // Type 0 (String), operation 3: push a long string.
        let mut m = rust_machine(&[0; 64]).unwrap();
        operate(&mut m, 3).unwrap();
        assert_eq!(Str::new(OBJECT).to_string_lossy(&m).unwrap(), LONG);
        // Type 1 (Vec<u32>) on fresh memory: push.
        let mut m = rust_machine(&[0; 64]).unwrap();
        operate(&mut m, 8 | 1).unwrap();
        assert_eq!(Words::new(OBJECT).to_vec(&m).unwrap(), [9]);
        // Type 5 (attribute map): insert, then look it up.
        let mut m = rust_machine(&[0; 64]).unwrap();
        operate(&mut m, 40 | 3).unwrap();
        assert!(Map::new(OBJECT).contains_key(&m, "key").unwrap());
        operate(&mut m, 40 | 7).unwrap();
    }

    /// A value whose list holds the value itself, and a chain of lists
    /// deeper than `MAX_NESTING`: every free operation on them fails (an
    /// error, not a stack overflow).
    #[test]
    fn cyclic_and_deep_values_are_errors() {
        let link = |m: &mut Machine, value: u64, vec: u64, next: u64| {
            GuestPtr::<GuestRustVec<RustValue>>::new(vec)
                .set(m, &GuestRustVec::from_raw_parts(next, 1, 1))
                .unwrap();
            let list = Value::List(GuestRustBox::from_raw(vec));
            Values::new(value).set(m, &RustValue::new(list)).unwrap();
        };
        // The cycle: the object's list is at OBJECT + 0x40, holding OBJECT.
        // (A failed free may have nulled boxes on the way: fresh memory for
        // each.)
        for op in [32 | 7, 88, 88 | 7] {
            let mut m = rust_machine(&[0; 0x100]).unwrap();
            link(&mut m, OBJECT, OBJECT + 0x40, OBJECT);
            assert!(operate(&mut m, op).is_err(), "op {op}");
        }
        // The chain: 1000 levels in the input's 64 KiB (32-byte values in
        // the lower half, their vectors in the upper), the last one empty.
        let mut m = rust_machine(&[0; OBJECT_MAX]).unwrap();
        let at = |i: u64| OBJECT + 32 * i;
        for i in 0..1000 {
            link(&mut m, at(i), at(i) + 0x8000, at(i + 1));
        }
        assert!(operate(&mut m, 88).is_err());
    }
}
