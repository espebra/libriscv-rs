#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]
//! The Rust guest collections (`libriscv::guest::GuestRust*`) on objects
//! written into guest memory, without a Rust guest: they allocate from the
//! native heap. What a hostile guest can build, so these run everywhere
//! (`native_rust.rs` needs the Rust RISC-V toolchain). The budget for
//! copies to the host is tested here for the C++ containers too.

use libriscv::guest::{
    self, GuestPtr, GuestRustBox, GuestRustBoxedSlice, GuestRustEnum, GuestRustString,
    GuestRustVec, GuestStdString, GuestStdUnorderedMap, GuestStdVector, MAX_HOST_COPY, MAX_NESTING,
    ScopedArenaObject, ToHost,
};
use libriscv::{GuestPod, Machine, MachineOptions};

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

// --- the budget for copies to the host ----------------------------------

/// One guest buffer that every aliased element points at.
const BIG: u64 = 0x100_0000;
const BIG_LEN: u64 = 4 << 20;
/// Where the element headers go.
const HEADERS: u64 = 0x80_0000;

fn big_buffer(m: &mut Machine) {
    m.memory.memset(BIG, b'x', BIG_LEN).unwrap();
}

/// `result` failed for running out of the budget. Without one, it would
/// have succeeded: say so without printing the (large) copy.
fn assert_over_budget<T>(result: libriscv::Result<T>, what: &str) {
    match result {
        Ok(_) => panic!("copied {what}, over the budget"),
        Err(e) => assert!(
            e.to_string().contains("too large to copy to the host"),
            "{e}"
        ),
    }
}

/// A `Vec<String>` of `count` headers that all point at the one buffer.
fn aliased_rust_strings(m: &mut Machine, count: u64) -> GuestRustVec<GuestRustString> {
    big_buffer(m);
    let s = GuestRustString::from_raw_parts(BIG, BIG_LEN, BIG_LEN);
    for i in 0..count {
        GuestPtr::new(HEADERS + 24 * i).set(m, &s).unwrap();
    }
    GuestRustVec::from_raw_parts(HEADERS, count, count)
}

/// Copying more than the budget, with guest memory used only once, fails
/// instead of allocating `count` × 4 MiB on the host (128 MiB here; the
/// limits on each piece alone would allow TiBs).
#[test]
fn aliased_strings_of_a_rust_vec_exceed_the_host_copy_budget() {
    let mut m = machine();
    let count = 2 * MAX_HOST_COPY as u64 / BIG_LEN;
    let vec = aliased_rust_strings(&mut m, count);
    assert_over_budget(vec.to_host(&m), "the strings");
    // A boxed slice of the same headers too.
    let slice = GuestRustBoxedSlice::<GuestRustString>::from_raw_parts(HEADERS, count);
    assert_over_budget(slice.to_host(&m), "the strings");
}

/// Data within the budget copies, and the budget is per top-level copy:
/// the next one starts afresh.
#[test]
fn rust_vec_within_the_host_copy_budget_copies() {
    let mut m = machine();
    let count = MAX_HOST_COPY as u64 / BIG_LEN / 2;
    let vec = aliased_rust_strings(&mut m, count);
    for _ in 0..3 {
        let host = vec.to_host(&m).unwrap();
        assert_eq!(host.len() as u64, count);
        assert!(host.iter().all(|s| s.len() as u64 == BIG_LEN));
    }
    // A smaller limit for one operation, shared by everything in it.
    assert_over_budget(
        guest::with_host_copy_limit(BIG_LEN as usize, || vec.to_host(&m)),
        "the strings with a 4 MiB limit",
    );
    let one = GuestRustString::from_raw_parts(BIG, BIG_LEN, BIG_LEN);
    let copies = |n| {
        guest::with_host_copy_limit(5 * BIG_LEN as usize / 2, || {
            (0..n)
                .map(|_| one.to_host(&m))
                .collect::<libriscv::Result<Vec<_>>>()
        })
    };
    assert_eq!(copies(2).unwrap().len(), 2);
    assert_over_budget(copies(3), "three strings with room for two and a half");
    // And afterwards, the default again.
    vec.to_host(&m).unwrap();
}

/// Nested containers share one budget: the inner vector fits on its own,
/// but the outer one, holding the same inner vector twice, doesn't.
#[test]
fn nested_rust_vecs_share_the_host_copy_budget() {
    let mut m = machine();
    let count = MAX_HOST_COPY as u64 * 3 / 4 / BIG_LEN;
    let inner = aliased_rust_strings(&mut m, count);
    inner.to_host(&m).unwrap();
    let outer_elems = HEADERS - 0x1000;
    for i in 0..2 {
        GuestPtr::new(outer_elems + 24 * i)
            .set(&mut m, &inner)
            .unwrap();
    }
    let outer = GuestRustVec::<GuestRustVec<GuestRustString>>::from_raw_parts(outer_elems, 2, 2);
    assert_over_budget(outer.to_host(&m), "the inner vector twice");
}

libriscv::guest_rust_enum! {
    /// [`Value`] with a host type.
    #[derive(Debug, Clone, PartialEq)]
    pub enum Tree => HostTree {
        Leaf(i64),
        Node(GuestRustBox<GuestRustVec<GuestRustEnum<Tree, u64>>>),
    }
}
type RustTree = GuestRustEnum<Tree, u64>;

/// `depth` levels of lists of two elements that both hold the list of the
/// level below: a DAG of 2^`depth` leaves in a few KiB of guest memory.
fn shared_sublists(m: &mut Machine, at: u64, depth: u64) -> RustTree {
    let mut value = RustTree::new(Tree::Leaf(1));
    for i in 0..depth {
        let elems = at + 0x100 * i;
        for e in 0..2 {
            GuestPtr::<RustTree>::new(elems + <RustTree as GuestPod>::SIZE as u64 * e)
                .set(m, &value)
                .unwrap();
        }
        GuestPtr::<GuestRustVec<RustTree>>::new(elems + 0x80)
            .set(m, &GuestRustVec::from_raw_parts(elems, 2, 2))
            .unwrap();
        value = RustTree::new(Tree::Node(GuestRustBox::from_raw(elems + 0x80)));
    }
    value
}

/// Shared sublists are charged for every copy: the budget bounds the walk
/// as well as the memory.
#[test]
fn shared_sublists_are_charged_for_every_copy() {
    let mut m = machine();
    let value = shared_sublists(&mut m, 0x40_0000, 16);
    assert_over_budget(
        guest::with_host_copy_limit(1 << 20, || value.to_host(&m)),
        "2^16 leaves",
    );
    let small = shared_sublists(&mut m, 0x50_0000, 4);
    guest::with_host_copy_limit(1 << 20, || small.to_host(&m)).unwrap();
}

/// A C++ `std::string` that points at the one buffer.
fn aliased_std_string() -> GuestStdString {
    let mut data = [0; 16];
    data[..8].copy_from_slice(&BIG_LEN.to_le_bytes());
    GuestStdString {
        ptr: BIG,
        size: BIG_LEN,
        data,
    }
}

/// C++ `to_string_vector`: the same, for a `std::vector<std::string>`.
#[test]
fn aliased_strings_of_a_std_vector_exceed_the_host_copy_budget() {
    let mut m = machine();
    big_buffer(&mut m);
    let count = 2 * MAX_HOST_COPY as u64 / BIG_LEN;
    for i in 0..count {
        GuestPtr::new(HEADERS + 32 * i)
            .set(&mut m, &aliased_std_string())
            .unwrap();
    }
    let end = HEADERS + 32 * count;
    let vec = GuestStdVector::<GuestStdString>::from_raw(HEADERS, end, end);
    assert_over_budget(vec.to_string_vector(&m), "the strings");
    // Within the budget, it copies.
    let part = HEADERS + 32 * (count / 4);
    let vec = GuestStdVector::<GuestStdString>::from_raw(HEADERS, part, part);
    assert_eq!(vec.to_string_vector(&m).unwrap().len() as u64, count / 4);
}

/// C++ `to_map`: the values of a `std::unordered_map` all point at the one
/// buffer.
#[test]
fn aliased_values_of_a_std_unordered_map_exceed_the_host_copy_budget() {
    let mut m = machine();
    big_buffer(&mut m);
    let count = 2 * MAX_HOST_COPY as u64 / BIG_LEN;
    let map = ScopedArenaObject::<GuestStdUnorderedMap<u64, GuestStdString>>::new(&mut m).unwrap();
    for key in 0..count {
        map.insert_or_assign(&mut m, &key, "small").unwrap();
    }
    assert_eq!(map.to_map(&m).unwrap().len() as u64, count);
    let mut values = Vec::new();
    map.for_each(&m, |_, v| {
        values.push(v);
        Ok(())
    })
    .unwrap();
    for v in values {
        // The small strings are inline: nothing to free.
        v.set(&mut m, &aliased_std_string()).unwrap();
    }
    assert_over_budget(map.to_map(&m), "the strings");
}
