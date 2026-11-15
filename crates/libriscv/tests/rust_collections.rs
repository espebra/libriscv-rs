#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]
//! The Rust guest collections (`libriscv::guest::GuestRust*`) on objects
//! written into guest memory, without a Rust guest: they allocate from the
//! native heap. What a hostile guest can build, so these run everywhere
//! (`native_rust.rs` needs the Rust RISC-V toolchain).

use libriscv::guest::{
    GuestPtr, GuestRustBox, GuestRustEnum, GuestRustString, GuestRustVec, MAX_NESTING,
};
use libriscv::{Machine, MachineOptions};

libriscv::guest_rust_enum! {
    /// The recursive value of the attribute trees: a list holds values.
    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        Nil(()),
        Int(i64),
        Str(GuestRustString),
        List(GuestRustBox<GuestRustVec<GuestRustEnum<Value, u64>>>),
    }
}
type RustValue = GuestRustEnum<Value, u64>;
type List = GuestRustVec<RustValue>;

const HEAP: u64 = 0x200_0000;
const HEAP_SIZE: u64 = 0x100_0000;

fn machine() -> Machine {
    let mut m = Machine::new(
        &[],
        MachineOptions {
            memory_max: 64 << 20,
            ..Default::default()
        },
    )
    .unwrap();
    m.setup_native_heap(200, HEAP, HEAP_SIZE).unwrap();
    m
}

/// `Value::List` at `value`, its box pointing at a one-element `Vec`
/// at `vec` whose element is at `next`.
fn link(m: &mut Machine, value: u64, vec: u64, next: u64) {
    GuestPtr::<List>::new(vec)
        .set(m, &List::from_raw_parts(next, 1, 1))
        .unwrap();
    let list = Value::List(GuestRustBox::from_raw(vec));
    GuestPtr::<RustValue>::new(value)
        .set(m, &RustValue::new(list))
        .unwrap();
}

fn is_too_deep(e: &libriscv::MachineError) -> bool {
    e.to_string().contains("nest too deeply")
}

/// A value whose list holds the value itself: freeing it (the guest's drop
/// glue) is an error, not a stack overflow that aborts the host.
#[test]
fn freeing_a_cycle_of_guest_collections_is_an_error() {
    let mut m = machine();
    let (value, vec) = (0x2_0000, 0x2_1000);
    link(&mut m, value, vec, value);
    let err = GuestPtr::<RustValue>::new(value).free(&mut m).unwrap_err();
    assert!(is_too_deep(&err), "{err}");
    // The limit is per operation: the next one starts from the top again.
    let plain = 0x3_0000;
    GuestPtr::<RustValue>::new(plain)
        .set(&mut m, &RustValue::new(Value::Int(7)))
        .unwrap();
    GuestPtr::<RustValue>::new(plain).free(&mut m).unwrap();
}

/// No cycle, just depth: a chain of lists far deeper than the limit.
#[test]
fn freeing_a_very_deep_chain_of_guest_collections_is_an_error() {
    let mut m = machine();
    let depth: u64 = 200_000;
    let at = |i: u64| 0x10_0000 + 64 * i;
    for i in 0..depth {
        link(&mut m, at(i), at(i) + 32, at(i + 1));
    }
    GuestPtr::<RustValue>::new(at(depth))
        .set(&mut m, &RustValue::new(Value::Nil(())))
        .unwrap();
    let err = GuestPtr::<RustValue>::new(at(0)).free(&mut m).unwrap_err();
    assert!(is_too_deep(&err), "{err}");
}

/// Real nesting below the limit is freed completely, within an ordinary
/// thread's stack: every level is a box and a vector, two pointers.
#[test]
fn nesting_below_the_limit_is_freed_completely() {
    let mut m = machine();
    let used = m.arena().unwrap().bytes_used();
    let levels = MAX_NESTING / 2 - 1;
    let mut value = RustValue::new(Value::Int(1));
    for _ in 0..levels {
        let list = List::from_slice(&mut m, &[value]).unwrap();
        let boxed = GuestRustBox::new(&mut m, &list).unwrap();
        value = RustValue::new(Value::List(boxed));
    }
    let top = GuestRustBox::new(&mut m, &value).unwrap();
    top.free(&mut m).unwrap();
    assert_eq!(m.arena().unwrap().bytes_used(), used);
}
