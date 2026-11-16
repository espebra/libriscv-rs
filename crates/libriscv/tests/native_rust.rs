//! Rust guest collections, from both sides: port of
//! `tests/unit/native_rust.cpp` (with `tests/unit/rustbuilder.cpp` as
//! `testutil::build_rust`).
//!
//! The "example_" tests come first: each shows one way a Rust value crosses
//! the boundary. The tests below them keep the host mirrors honest: field
//! order, `RawVec` growth, UTF-8 validation and allocation balance.
//!
//! What makes all of it work is at the top of the guest program: the guest
//! installs a `#[global_allocator]` that forwards to the native heap, so
//! both sides allocate out of the same heap.

#![cfg(all(feature = "ext-a", feature = "ext-c", feature = "ext-f"))]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::collections::BTreeMap;
use std::sync::OnceLock;

use libriscv::guest::{
    GuestPtr, GuestRustAttr, GuestRustAttributes, GuestRustBox, GuestRustBoxedSlice,
    GuestRustBoxedStr, GuestRustEnum, GuestRustSlice, GuestRustStr, GuestRustString, GuestRustVec,
    GuestType, ToHost,
};
use libriscv::{GuestPod, Machine, MachineOptions, SyscallHandler};

const MAX_INSTRUCTIONS: u64 = 100_000_000;
const HEAP_SIZE: u64 = 4 << 20;

// The host functions the guest calls with counted dyncalls. A dyncall is the
// system call numbered by its immediate, so these are system call numbers.
const NATIVE_HEAP: usize = 470;
const NATIVE_MEMORY: usize = 475;
const HOST_PAUSE: usize = 480;
const HOST_FILL_STRING: usize = 481;
const HOST_FILL_VEC: usize = 482;
const HOST_TAKE_STR: usize = 483;
const HOST_TAKE_VEC: usize = 484;
const HOST_TAKE_ATTRS: usize = 485;
const HOST_FILL_ATTRS: usize = 486;

type RustString = GuestRustString;
type RustVec<T> = GuestRustVec<T>;
type RustBox<T> = GuestRustBox<T>;

// ---------------------------------------------------------------------------
// The host mirror of the guest's attribute tree. The variant order and the
// tag width ARE the ABI: they must match the guest's #[repr(C, u64)] enum,
// and the layout probe below checks that they do.
// ---------------------------------------------------------------------------

libriscv::guest_rust_enum! {
    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        Nil(()),
        Int(i64),
        Float(f64),
        Bool(bool),
        Str(RustString),
        Group(RustGroup),
        List(RustList),
    }
}
type RustValue = GuestRustEnum<Value, u64>;
type RustAttrs = GuestRustAttributes<RustValue>;
type RustGroup = GuestRustBox<RustAttrs>;
type RustList = GuestRustBox<RustVec<RustValue>>;

// The layout is requested, not discovered: the host mirror is the size of
// what the guest declared.
const _: () = assert!(
    RustValue::SIZE == 8 + RustString::SIZE,
    "tag + the widest variant"
);
const _: () = assert!(
    GuestRustAttr::<RustValue>::SIZE == RustString::SIZE + RustValue::SIZE,
    "Attr is a String and a Value"
);
const _: () = assert!(
    RustAttrs::SIZE == RustVec::<i32>::SIZE,
    "Attributes is a Vec of entries"
);

// What the host system calls write into the guest's own collections
const HOST_TEXT: &str = "A string the host wrote into a guest-owned String";
const GUEST_SUFFIX: &str = " and the guest appended to it";

// What the guest lends the host in the worked examples
const LENT_TEXT: &str = "A borrowed &str lent out by the guest";
const LENT_NAMES: [&str; 3] = ["alpha", "beta", "gamma"];

// What the guest's rust_attrs_build() puts in the tree it lends the host
const GUEST_BETA: &str = "a string the guest allocated";
const GUEST_NESTED: &str = "inside the group";

/// The FNV-1a the guest computes over the same bytes.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

const GUEST_PROGRAM: &str = r##"
#![allow(improper_ctypes_definitions)]
use core::alloc::{GlobalAlloc, Layout};
use core::arch::asm;

// The host functions, by dyncall index: a dyncall is the system call
// numbered by its immediate, so these are the numbers the host installed its
// handlers under.
const NATIVE_HEAP: i32 = 470; // malloc, calloc, realloc, free
const NATIVE_MEMORY: i32 = 475; // memcpy, memset, memmove, memcmp
const HOST_PAUSE: i32 = 480;
const HOST_FILL_STRING: i32 = 481;
const HOST_FILL_VEC: i32 = 482;
const HOST_TAKE_STR: i32 = 483;
const HOST_TAKE_VEC: i32 = 484;
const HOST_TAKE_ATTRS: i32 = 485;
const HOST_FILL_ATTRS: i32 = 486;

// Every call into the host is a counted dynamic call: one custom-2
// instruction (opcode 0x5b, funct3 0b111) carrying the system call number in
// its immediate, and how many argument registers the host reads (rd = x16 +
// inputs) and writes back (rs1 = x24 + outputs, no floating point). Unlike
// an ecall it needs no a7, and the operands are exactly the registers the
// host touches, so the compiler keeps everything else live in registers
// across the call instead of spilling and reloading it. A function the host
// doesn't write a result for leaves a0 alone, so memcpy can return dest
// without saving it first.
macro_rules! dyncall {
	($index:expr, $inputs:literal => $outputs:literal $(, $($operands:tt)+)?) => {
		asm!(".insn i 0x5b, 7, x{rd}, x{rs1}, {index}",
			index = const $index,
			rd = const 16 + $inputs,
			rs1 = const 24 + $outputs,
			$($($operands)+,)?
			options(nostack))
	};
}

// The arena hands out 16-byte aligned blocks, which covers every alignment
// that the standard collections ask for.
struct SysAllocator;

unsafe impl GlobalAlloc for SysAllocator {
	#[inline]
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		let ret: *mut u8;
		dyncall!(NATIVE_HEAP + 0, 1 => 1, in("a0") layout.size(), lateout("a0") ret);
		ret
	}
	#[inline]
	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		let ret: *mut u8;
		dyncall!(NATIVE_HEAP + 1, 2 => 1, in("a0") 1usize, in("a1") layout.size(), lateout("a0") ret);
		ret
	}
	#[inline]
	unsafe fn realloc(&self, ptr: *mut u8, _layout: Layout, new_size: usize) -> *mut u8 {
		let ret: *mut u8;
		dyncall!(NATIVE_HEAP + 2, 2 => 1, in("a0") ptr, in("a1") new_size, lateout("a0") ret);
		ret
	}
	#[inline]
	unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
		dyncall!(NATIVE_HEAP + 3, 1 => 0, in("a0") ptr);
	}
}

#[global_allocator]
static ALLOCATOR: SysAllocator = SysAllocator;

// The block memory operations, handed to the host the same way the heap is.
// The link step passes --wrap=memcpy and friends, which rewrites every
// reference to memcpy in the program - Rust's own and the ones inside glibc -
// to __wrap_memcpy, so these four take over without any calling code knowing.
// The host then copies with a native memcpy instead of emulating the loop.
//
// The bodies are pure inline assembly on purpose: a hand-written byte loop
// here would be recognised by LLVM as a memcpy and turned back into a call to
// this very function.

#[no_mangle]
pub unsafe extern "C" fn __wrap_memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
	dyncall!(NATIVE_MEMORY + 0, 3 => 0, in("a0") dest, in("a1") src, in("a2") n);
	dest
}

#[no_mangle]
pub unsafe extern "C" fn __wrap_memset(dest: *mut u8, value: i32, n: usize) -> *mut u8 {
	dyncall!(NATIVE_MEMORY + 1, 3 => 0, in("a0") dest, in("a1") value, in("a2") n);
	dest
}

#[no_mangle]
pub unsafe extern "C" fn __wrap_memmove(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
	dyncall!(NATIVE_MEMORY + 2, 3 => 0, in("a0") dest, in("a1") src, in("a2") n);
	dest
}

#[no_mangle]
pub unsafe extern "C" fn __wrap_memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
	let result: i32;
	dyncall!(NATIVE_MEMORY + 3, 3 => 1, in("a0") s1, in("a1") s2, in("a2") n, lateout("a0") result);
	result
}

// Comparing two slices with == does not call memcmp: LLVM knows the caller
// only wants to know whether they differ, and emits the equality-only entry
// point instead - __memcmpeq on a current toolchain, bcmp on an older one.
// Both are satisfied by a memcmp result, since all either one promises is
// zero or non-zero.

#[no_mangle]
pub unsafe extern "C" fn __wrap___memcmpeq(s1: *const u8, s2: *const u8, n: usize) -> i32 {
	__wrap_memcmp(s1, s2, n)
}

#[no_mangle]
pub unsafe extern "C" fn __wrap_bcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
	__wrap_memcmp(s1, s2, n)
}

/// The same FNV-1a that the host computes, so both sides can prove they are
/// looking at the same bytes.
fn fnv1a(bytes: &[u8]) -> u64 {
	let mut hash: u64 = 0xcbf29ce484222325;
	for b in bytes {
		hash ^= *b as u64;
		hash = hash.wrapping_mul(0x100000001b3);
	}
	hash
}

unsafe fn borrow<'a>(ptr: *const u8, len: usize) -> &'a str {
	core::str::from_utf8(core::slice::from_raw_parts(ptr, len))
		.expect("The host passed bytes that are not UTF-8")
}

// ---------------------------------------------------------------------------
// The guest half of the worked examples. Everything below this block is the
// guest half of the exhaustive tests further down in native_rust.cpp.
// ---------------------------------------------------------------------------

/// A String returned by value. It is too large for a register, so the caller
/// allocates room for it and passes the address as a hidden first argument.
#[no_mangle]
pub extern "C" fn rust_make_greeting() -> String {
	String::from("A String returned by value from the guest")
}

/// The same, but with an argument: the hidden pointer takes the first register,
/// so `count` arrives in the second one.
#[no_mangle]
pub extern "C" fn rust_make_squares(count: usize) -> Vec<u32> {
	(0..count as u32).map(|i| i * i).collect()
}

/// The guest lends the host a &str, which is two registers: address and length.
#[no_mangle]
pub extern "C" fn rust_lends_str() {
	let text = String::from("A borrowed &str lent out by the guest");
	unsafe {
		dyncall!(HOST_TAKE_STR, 2 => 0, in("a0") text.as_ptr(), in("a1") text.len());
	}
	// Still owned here, and dropped at the end of the function
}

/// The guest lends the host a whole collection, which is one register holding
/// the address of its three words.
#[no_mangle]
pub extern "C" fn rust_lends_vec() {
	let names: Vec<String> = vec![
		String::from("alpha"), String::from("beta"), String::from("gamma")];
	unsafe {
		dyncall!(HOST_TAKE_VEC, 1 => 0, in("a0") &names as *const Vec<String>);
	}
}

/// Collections nest: every element of the outer Vec is itself the three words
/// of a Vec, which is exactly what the host pushes into it.
#[no_mangle]
pub extern "C" fn rust_nested_sum(vec: *const Vec<Vec<u32>>) -> u64 {
	let vec: &Vec<Vec<u32>> = unsafe { &*vec };
	vec.iter().map(|row| row.iter().map(|x| *x as u64).sum::<u64>()).sum()
}

// ---------------------------------------------------------------------------
// Layout probe: the raw machine words of every collection, followed by what
// the Rust API says they hold, so the host can prove the field order it uses.
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn rust_probe_layout(out: *mut usize) {
	let vec: Vec<u32> = vec![1, 2, 3];
	let string = String::from("A string that is definitely heap allocated");
	let slice: &[u32] = &vec[..];
	let strref: &str = &string[..];
	let empty_string = String::new();
	let empty_vec: Vec<u32> = Vec::new();

	unsafe {
		let vw: [usize; 3] = core::mem::transmute_copy(&vec);
		let sw: [usize; 3] = core::mem::transmute_copy(&string);
		let slw: [usize; 2] = core::mem::transmute_copy(&slice);
		let stw: [usize; 2] = core::mem::transmute_copy(&strref);
		let ew: [usize; 3] = core::mem::transmute_copy(&empty_string);
		let evw: [usize; 3] = core::mem::transmute_copy(&empty_vec);
		let words: [usize; 26] = [
			vw[0], vw[1], vw[2], vec.as_ptr() as usize, vec.len(), vec.capacity(),
			sw[0], sw[1], sw[2], string.as_ptr() as usize, string.len(), string.capacity(),
			slw[0], slw[1], slice.as_ptr() as usize, slice.len(),
			stw[0], stw[1], strref.as_ptr() as usize, strref.len(),
			ew[0], ew[1], ew[2],
			evw[0], evw[1], evw[2],
		];
		core::ptr::copy_nonoverlapping(words.as_ptr(), out, words.len());
	}
}

/// Forces the allocations that the runtime makes lazily on its first use of
/// each collection, so that the host can take an allocation baseline which
/// does not move underneath it afterwards.
#[no_mangle]
pub extern "C" fn rust_warmup() {
	let mut string = String::from("warm");
	string.push_str("up");
	let mut vec: Vec<u32> = Vec::new();
	vec.push(1);
	let strings: Vec<String> = vec![String::from("warm"), String::from("up")];
	let joined = strings.join(", ");
	core::hint::black_box((&string, &vec, &strings, &joined));
}

/// The capacity that Rust's RawVec ends up with after pushing one element at
/// a time. The host's reserve() has to arrive at the same number.
#[no_mangle]
pub extern "C" fn rust_vec_capacity_after(pushes: usize) -> usize {
	let mut vec: Vec<u32> = Vec::new();
	for i in 0..pushes {
		vec.push(i as u32);
	}
	vec.capacity()
}

#[no_mangle]
pub extern "C" fn rust_string_capacity_after(pushes: usize) -> usize {
	let mut string = String::new();
	for _ in 0..pushes {
		string.push('x');
	}
	string.capacity()
}

// The smallest allocation RawVec makes depends on the size of the element:
// 8 for a byte, 4 for anything up to 1024 bytes, and 1 above that. One
// function per branch, so the host constant can be checked against each.

#[no_mangle]
pub extern "C" fn rust_u8vec_capacity_after(pushes: usize) -> usize {
	let mut vec: Vec<u8> = Vec::new();
	for i in 0..pushes {
		vec.push(i as u8);
	}
	vec.capacity()
}

#[no_mangle]
pub extern "C" fn rust_u64vec_capacity_after(pushes: usize) -> usize {
	let mut vec: Vec<u64> = Vec::new();
	for i in 0..pushes {
		vec.push(i as u64);
	}
	vec.capacity()
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Big { pub data: [u8; 2048] }

#[no_mangle]
pub extern "C" fn rust_bigvec_capacity_after(pushes: usize) -> usize {
	let mut vec: Vec<Big> = Vec::new();
	for _ in 0..pushes {
		vec.push(Big { data: [0u8; 2048] });
	}
	vec.capacity()
}

// ---------------------------------------------------------------------------
// Strings
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn rust_string_len(string: *const String) -> usize {
	let string: &String = unsafe { &*string };
	string.len()
}

#[no_mangle]
pub extern "C" fn rust_string_checksum(string: *const String) -> u64 {
	let string: &String = unsafe { &*string };
	fnv1a(string.as_bytes())
}

#[no_mangle]
pub extern "C" fn rust_string_is_utf8(ptr: *const u8, len: usize) -> i32 {
	let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
	core::str::from_utf8(bytes).is_ok() as i32
}

/// Append to a host-built string, which reallocates it in the arena.
#[no_mangle]
pub extern "C" fn rust_string_append(string: *mut String, ptr: *const u8, len: usize) {
	let string: &mut String = unsafe { &mut *string };
	string.push_str(unsafe { borrow(ptr, len) });
}

/// Build a string in the guest, at an address the host picked.
#[no_mangle]
pub extern "C" fn rust_string_make(out: *mut String, ptr: *const u8, len: usize) {
	let mut string = String::from(unsafe { borrow(ptr, len) });
	string.push_str(" (built by the guest)");
	unsafe { core::ptr::write(out, string) };
}

/// Drop a host-built string in the guest, leaving an empty one behind.
#[no_mangle]
pub extern "C" fn rust_string_drop(string: *mut String) {
	unsafe {
		core::ptr::drop_in_place(string);
		core::ptr::write(string, String::new());
	}
}

// ---------------------------------------------------------------------------
// Ownership: the guest keeps a string the host allocated
// ---------------------------------------------------------------------------

static mut STORED: Option<String> = None;

fn stored() -> &'static mut Option<String> {
	unsafe { &mut *core::ptr::addr_of_mut!(STORED) }
}

#[no_mangle]
pub extern "C" fn rust_take_string(string: *mut String) {
	let owned = unsafe { core::ptr::read(string) };
	unsafe { core::ptr::write(string, String::new()) };
	*stored() = Some(owned);
}

#[no_mangle]
pub extern "C" fn rust_stored_checksum() -> u64 {
	match stored() {
		Some(string) => fnv1a(string.as_bytes()),
		None => 0,
	}
}

#[no_mangle]
pub extern "C" fn rust_release_stored() {
	*stored() = None;
}

// ---------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn rust_vec_sum(vec: *const Vec<u32>) -> u64 {
	let vec: &Vec<u32> = unsafe { &*vec };
	vec.iter().map(|x| *x as u64).sum()
}

#[no_mangle]
pub extern "C" fn rust_vec_push(vec: *mut Vec<u32>, value: u32) {
	let vec: &mut Vec<u32> = unsafe { &mut *vec };
	vec.push(value);
}

#[no_mangle]
pub extern "C" fn rust_vec_make(out: *mut Vec<u32>, count: usize) {
	let vec: Vec<u32> = (0..count as u32).map(|i| i * i).collect();
	unsafe { core::ptr::write(out, vec) };
}

#[no_mangle]
pub extern "C" fn rust_strvec_checksum(vec: *const Vec<String>) -> u64 {
	let vec: &Vec<String> = unsafe { &*vec };
	let mut hash = fnv1a(&[vec.len() as u8]);
	for string in vec.iter() {
		hash = hash.wrapping_add(fnv1a(string.as_bytes()));
	}
	hash
}

#[no_mangle]
pub extern "C" fn rust_strvec_push(vec: *mut Vec<String>, ptr: *const u8, len: usize) {
	let vec: &mut Vec<String> = unsafe { &mut *vec };
	vec.push(String::from(unsafe { borrow(ptr, len) }));
}

#[no_mangle]
pub extern "C" fn rust_strvec_join(vec: *const Vec<String>, out: *mut String) {
	let vec: &Vec<String> = unsafe { &*vec };
	let joined = vec.join(", ");
	unsafe { core::ptr::write(out, joined) };
}

// ---------------------------------------------------------------------------
// Borrowed slices. As arguments the two halves arrive in separate registers,
// and stored in memory they are a { ptr, len } pair.
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn rust_slice_sum(ptr: *const u32, len: usize) -> u64 {
	let slice = unsafe { core::slice::from_raw_parts(ptr, len) };
	slice.iter().map(|x| *x as u64).sum()
}

#[no_mangle]
pub extern "C" fn rust_str_checksum(ptr: *const u8, len: usize) -> u64 {
	fnv1a(unsafe { borrow(ptr, len) }.as_bytes())
}

#[repr(C)]
pub struct FatPointer {
	ptr: usize,
	len: usize,
}

/// Store a &str and a &[u32] borrowed from guest collections, so the host can
/// read them back as GuestRustStr and GuestRustSlice.
#[no_mangle]
pub extern "C" fn rust_borrow_into(string: *const String, vec: *const Vec<u32>, out: *mut FatPointer) {
	let string: &String = unsafe { &*string };
	let vec: &Vec<u32> = unsafe { &*vec };
	unsafe {
		let strref: &str = string.as_str();
		let slice: &[u32] = vec.as_slice();
		let stw: [usize; 2] = core::mem::transmute_copy(&strref);
		let slw: [usize; 2] = core::mem::transmute_copy(&slice);
		core::ptr::write(out.add(0), FatPointer { ptr: stw[0], len: stw[1] });
		core::ptr::write(out.add(1), FatPointer { ptr: slw[0], len: slw[1] });
	}
}

// ---------------------------------------------------------------------------
// The host writing into collections the guest owns, through a system call
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn rust_host_fills_string() -> u64 {
	let mut string = String::new();
	unsafe {
		dyncall!(HOST_FILL_STRING, 1 => 0, in("a0") &mut string as *mut String);
	}
	// The guest owns the bytes the host wrote: it can grow them and drop them
	let mut hash = fnv1a(string.as_bytes());
	string.push_str(" and the guest appended to it");
	hash = hash.wrapping_add(fnv1a(string.as_bytes()));
	hash
}

#[no_mangle]
pub extern "C" fn rust_host_fills_vec() -> u64 {
	let mut vec: Vec<u32> = Vec::new();
	unsafe {
		dyncall!(HOST_FILL_VEC, 1 => 0, in("a0") &mut vec as *mut Vec<u32>);
	}
	let mut sum: u64 = vec.iter().map(|x| *x as u64).sum();
	vec.push(1000);
	sum += vec.len() as u64;
	sum
}

// ---------------------------------------------------------------------------
// Box: the one Rust smart pointer with a guaranteed layout. A Box the host
// allocated really did come from this program's global allocator, so
// Box::from_raw() on it is sound and not merely convenient - and then the
// ordinary drop at the end of the scope returns it to the host arena.
// ---------------------------------------------------------------------------

/// Take a Box<i64> the host built, and let it drop
#[no_mangle]
pub extern "C" fn rust_box_take_i64(ptr: *mut i64) -> i64 {
	let boxed: Box<i64> = unsafe { Box::from_raw(ptr) };
	*boxed
}

/// Build a Box<i64> and hand ownership to the host
#[no_mangle]
pub extern "C" fn rust_box_make_i64(value: i64) -> *mut i64 {
	Box::into_raw(Box::new(value * 2))
}

/// A struct of the two unsized boxes, so the host can build them in place.
#[repr(C)]
pub struct Boxes {
	pub numbers: Box<[u32]>,
	pub name: Box<str>,
}

/// Take a Box<Boxes> the host built. Dropping it frees three allocations: the
/// struct, the boxed slice and the boxed str, all through the drop glue.
#[no_mangle]
pub extern "C" fn rust_boxes_take(ptr: *mut Boxes) -> u64 {
	let boxes: Box<Boxes> = unsafe { Box::from_raw(ptr) };
	let mut hash = fnv1a(boxes.name.as_bytes());
	for n in boxes.numbers.iter() {
		hash = hash.wrapping_mul(131).wrapping_add(*n as u64);
	}
	hash
}

// ---------------------------------------------------------------------------
// The attribute tree.
//
// Ordinary Rust, with two attributes on it. #[repr(C, u64)] asks for the
// tagged-union layout that the host mirrors - a plain #[repr(Rust)] enum would
// be reordered and niche-packed - and #[repr(C)] does the same for the entry
// struct, which a tuple could not promise.
//
// The entries are kept sorted by key, so a lookup is a binary search. That is
// not a marshalling compromise: it is one allocation for the whole map instead
// of a node per key, it needs no hasher that both sides have to agree on, and
// std::HashMap is one .collect() away for a guest that wants one.
// ---------------------------------------------------------------------------

#[repr(C, u64)]
pub enum Value {
	Nil,
	Int(i64),
	Float(f64),
	Bool(bool),
	Str(String),
	Group(Box<Attributes>),
	List(Box<Vec<Value>>),
}

#[repr(C)]
pub struct Attr {
	pub key: String,
	pub value: Value,
}

#[repr(C)]
pub struct Attributes {
	entries: Vec<Attr>,
}

impl Attributes {
	pub fn new() -> Self {
		Attributes { entries: Vec::new() }
	}

	pub fn get(&self, key: &str) -> Option<&Value> {
		self.entries
			.binary_search_by(|a| a.key.as_str().cmp(key))
			.ok()
			.map(|i| &self.entries[i].value)
	}

	pub fn set(&mut self, key: &str, value: Value) {
		match self.entries.binary_search_by(|a| a.key.as_str().cmp(key)) {
			Ok(i) => self.entries[i].value = value,
			Err(i) => self.entries.insert(i, Attr { key: key.to_string(), value }),
		}
	}

	pub fn len(&self) -> usize {
		self.entries.len()
	}

	pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
		self.entries.iter().map(|a| (a.key.as_str(), &a.value))
	}
}

/// The checksum the host computes over the same tree, with the same operations
/// in the same (sorted) order, so that both sides can prove they are looking at
/// the same thing.
fn checksum_value(value: &Value) -> u64 {
	match value {
		Value::Nil => 1,
		Value::Int(i) => 2u64.wrapping_mul(*i as u64).wrapping_add(0x10),
		Value::Float(f) => 3u64.wrapping_mul(f.to_bits()).wrapping_add(0x20),
		Value::Bool(b) => if *b { 4 } else { 5 },
		Value::Str(s) => fnv1a(s.as_bytes()),
		Value::Group(g) => checksum_attrs(g).wrapping_mul(31),
		Value::List(l) => l
			.iter()
			.fold(7u64, |acc, v| acc.wrapping_mul(131).wrapping_add(checksum_value(v))),
	}
}

fn checksum_attrs(attrs: &Attributes) -> u64 {
	let mut hash = attrs.len() as u64;
	for (key, value) in attrs.iter() {
		hash = hash.wrapping_mul(0x100000001b3);
		hash ^= fnv1a(key.as_bytes());
		hash = hash.wrapping_add(checksum_value(value));
	}
	hash
}

/// The host built a tree on the guest heap and handed over the pointer. The
/// guest takes the whole thing with one from_raw, and drops it on the way out:
/// no rebuilding, and no host-side teardown function either, because the drop
/// glue walks the tree.
#[no_mangle]
pub extern "C" fn rust_attrs_take(ptr: *mut Attributes) -> u64 {
	let attrs: Box<Attributes> = unsafe { Box::from_raw(ptr) };
	checksum_attrs(&attrs)
}

/// The host owns the tree and only lends it out for the duration of the call,
/// which is the shape an engine handing attributes to a script every frame
/// wants: nothing is allocated, nothing is freed.
#[no_mangle]
pub extern "C" fn rust_attrs_borrow(ptr: *const Attributes) -> u64 {
	checksum_attrs(unsafe { &*ptr })
}

/// The guest builds a tree of its own and lends it to the host, which walks it
/// in place. Seven instructions of guest work, whatever the size of the tree.
#[no_mangle]
pub extern "C" fn rust_attrs_build() -> u64 {
	let mut attrs = Attributes::new();
	attrs.set("alpha", Value::Int(-1234));
	attrs.set("beta", Value::Str(String::from("a string the guest allocated")));
	attrs.set("gamma", Value::Float(2.5));
	attrs.set("delta", Value::Bool(true));
	attrs.set("epsilon", Value::Nil);

	let mut group = Attributes::new();
	group.set("nested_int", Value::Int(77));
	group.set("nested_str", Value::Str(String::from("inside the group")));
	attrs.set("zeta", Value::Group(Box::new(group)));

	attrs.set("eta", Value::List(Box::new(vec![
		Value::Int(1), Value::Int(2), Value::Str(String::from("three"))])));

	let checksum = checksum_attrs(&attrs);
	unsafe {
		dyncall!(HOST_TAKE_ATTRS, 1 => 0, in("a0") &attrs as *const Attributes);
	}
	checksum
	// The guest still owns it, and drops the whole tree here
}

/// The host fills in a tree the guest owns, through a system call, and the
/// guest then uses it as its own - including growing it.
#[no_mangle]
pub extern "C" fn rust_attrs_host_fills() -> u64 {
	let mut attrs = Attributes::new();
	unsafe {
		dyncall!(HOST_FILL_ATTRS, 1 => 0, in("a0") &mut attrs as *mut Attributes);
	}
	let mut hash = checksum_attrs(&attrs);
	// The entries the host allocated are the guest's now: it can insert into
	// the same vector, and it drops all of it at the end of the function
	attrs.set("guest_added", Value::Int(999));
	hash = hash.wrapping_add(checksum_attrs(&attrs));
	hash
}

/// A guest that would rather have a real HashMap gets one for the price of a
/// collect(). The tree is the interchange shape, not a cage.
#[no_mangle]
pub extern "C" fn rust_attrs_into_hashmap(ptr: *mut Attributes) -> u64 {
	let attrs: Box<Attributes> = unsafe { Box::from_raw(ptr) };
	let map: std::collections::HashMap<String, Value> =
		attrs.entries.into_iter().map(|a| (a.key, a.value)).collect();

	let mut total = map.len() as u64;
	for key in ["alpha", "beta", "gamma"] {
		if let Some(value) = map.get(key) {
			total = total.wrapping_add(checksum_value(value));
		}
	}
	total
}

// ---------------------------------------------------------------------------
// Layout probe for the enum and the entry struct, so that a rustc release
// which lays out #[repr(C, u64)] differently fails here instead of silently
// reading garbage everywhere else.
// ---------------------------------------------------------------------------

/// The offset of the payload of a variant, measured against the enum itself
macro_rules! payload_offset {
	($value:expr, $pattern:pat => $field:expr) => {{
		let value = $value;
		let base = &value as *const Value as usize;
		match &value {
			$pattern => ($field as *const _ as usize) - base,
			#[allow(unreachable_patterns)]
			_ => usize::MAX,
		}
	}};
}

/// The discriminant of a variant, read as the first bytes of the enum
fn tag_of(value: &Value) -> usize {
	unsafe { *(value as *const Value as *const u64) as usize }
}

#[no_mangle]
pub extern "C" fn rust_probe_enum_layout(out: *mut usize) {
	let attr = Attr { key: String::from("k"), value: Value::Int(1) };
	let attr_base = &attr as *const Attr as usize;
	let empty = Attributes::new();

	let words: [usize; 21] = [
		core::mem::size_of::<Value>(),
		core::mem::align_of::<Value>(),
		// The discriminants, which must be the declaration order 0..n
		tag_of(&Value::Nil),
		tag_of(&Value::Int(0)),
		tag_of(&Value::Float(0.0)),
		tag_of(&Value::Bool(false)),
		tag_of(&Value::Str(String::new())),
		tag_of(&Value::Group(Box::new(Attributes::new()))),
		tag_of(&Value::List(Box::new(Vec::new()))),
		// The payload offsets, which must all be sizeof(tag)
		payload_offset!(Value::Int(1), Value::Int(x) => x),
		payload_offset!(Value::Float(1.0), Value::Float(x) => x),
		payload_offset!(Value::Bool(true), Value::Bool(x) => x),
		payload_offset!(Value::Str(String::new()), Value::Str(x) => x),
		payload_offset!(Value::Group(Box::new(Attributes::new())), Value::Group(x) => x),
		payload_offset!(Value::List(Box::new(Vec::new())), Value::List(x) => x),
		// The entry struct: a String followed by a Value, in that order
		core::mem::size_of::<Attr>(),
		(&attr.key as *const String as usize) - attr_base,
		(&attr.value as *const Value as usize) - attr_base,
		// Attributes is a newtype over the Vec, so it is the same three words
		core::mem::size_of::<Attributes>(),
		core::mem::size_of::<Box<Attributes>>(),
		empty.len(),
	];
	unsafe { core::ptr::copy_nonoverlapping(words.as_ptr(), out, words.len()) };
}

// ---------------------------------------------------------------------------
// The block memory operations, from the guest side.
//
// Every length here comes in as an argument, which is what keeps LLVM from
// expanding the operation inline: a runtime length has to become a call, and
// the call is what --wrap redirects into the host. Each function returns a
// checksum of the bytes the host produced, so a copy that lands in the wrong
// place cannot pass.
// ---------------------------------------------------------------------------

/// fill() on a byte slice is a memset.
#[no_mangle]
pub extern "C" fn rust_memset_fill(len: usize, byte: u8) -> u64 {
	let mut buffer = vec![0u8; len];
	buffer.fill(byte);
	fnv1a(&buffer)
}

/// copy_from_slice() between two buffers that cannot overlap is a memcpy.
#[no_mangle]
pub extern "C" fn rust_memcpy_roundtrip(len: usize) -> u64 {
	let source: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
	let mut dest = vec![0u8; len];
	dest.copy_from_slice(&source);
	fnv1a(&dest)
}

/// copy_within() overlaps, so it is a memmove. Which way it shifts decides
/// which of the host's two paths runs: forwards when the destination is below
/// the source, backwards when it is above.
#[no_mangle]
pub extern "C" fn rust_memmove_overlap(len: usize, shift: usize, downwards: bool) -> u64 {
	let mut buffer: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
	if downwards {
		buffer.copy_within(shift.., 0);
	} else {
		buffer.copy_within(..len - shift, shift);
	}
	fnv1a(&buffer)
}

/// Comparing two byte slices of a runtime length is a memcmp, and the sign of
/// what the host returns is what Ord turns into Less, Equal or Greater.
#[no_mangle]
pub extern "C" fn rust_memcmp_order(len: usize, index: usize, value: u8) -> i64 {
	let left: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
	let mut right = left.clone();
	if index < len {
		right[index] = value;
	}
	match left.cmp(&right) {
		core::cmp::Ordering::Less => -1,
		core::cmp::Ordering::Equal => 0,
		core::cmp::Ordering::Greater => 1,
	}
}

/// Slice equality, which is the common case and does *not* go to memcmp:
/// LLVM emits the equality-only entry point for it, so this only reaches the
/// host if __memcmpeq (or bcmp on an older toolchain) is wrapped as well.
#[no_mangle]
pub extern "C" fn rust_memcmp_equality(len: usize, index: usize, value: u8) -> u64 {
	let left: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
	let mut right = left.clone();
	if index < len {
		right[index] = value;
	}
	if left == right { 1 } else { 0 }
}

/// A String built out of parts, which is what the collections do all day: the
/// reserve is a realloc into the host arena and the append is a memcpy.
#[no_mangle]
pub extern "C" fn rust_memops_build_string(repeats: usize) -> String {
	let piece = "The guest appends this through the host's memcpy. ";
	let mut text = String::with_capacity(piece.len() * repeats);
	for _ in 0..repeats {
		text.push_str(piece);
	}
	text
}

fn main() {
	println!("Rust guest is ready");
	// Pause here with everything initialized, so that the host can make calls
	unsafe {
		dyncall!(HOST_PAUSE, 0 => 0);
	}
}
"##;

// ===========================================================================
// The host half of the attribute tree: the checksum the guest computes, and
// the tree the guest builds, from the host side. Neither serializes
// anything: reading borrows keys and strings from guest memory, writing
// allocates the entries in their final shape on the guest heap.
// ===========================================================================

fn host_checksum_value(m: &Machine, value: &RustValue) -> u64 {
    match value.value().expect("a valid discriminant") {
        Value::Nil(()) => 1,
        Value::Int(i) => 2u64.wrapping_mul(*i as u64).wrapping_add(0x10),
        Value::Float(f) => 3u64.wrapping_mul(f.to_bits()).wrapping_add(0x20),
        Value::Bool(b) => {
            if *b {
                4
            } else {
                5
            }
        }
        Value::Str(s) => fnv1a(s.as_bytes(m).unwrap()),
        Value::Group(g) => host_checksum_attrs(m, &g.get(m).unwrap()).wrapping_mul(31),
        Value::List(l) => {
            let list = l.get(m).unwrap();
            list.to_vec(m).unwrap().iter().fold(7u64, |acc, v| {
                acc.wrapping_mul(131)
                    .wrapping_add(host_checksum_value(m, v))
            })
        }
    }
}

fn host_checksum_attrs(m: &Machine, attrs: &RustAttrs) -> u64 {
    let mut hash = attrs.len() as u64;
    for entry in attrs.iter(m).unwrap() {
        let (key, value) = entry.unwrap();
        hash = hash.wrapping_mul(0x0100_0000_01b3);
        hash ^= fnv1a(key);
        hash = hash.wrapping_add(host_checksum_value(m, &value.get(m).unwrap()));
    }
    hash
}

/// Build the tree the guest's rust_attrs_build() builds, on the guest heap.
/// Nothing is copied afterwards: this is the final shape.
fn host_build_attrs(attrs: &mut RustAttrs, m: &mut Machine) -> libriscv::Result<()> {
    attrs.insert(m, "alpha", &Value::Int(-1234))?;
    let beta = RustString::from_str(m, GUEST_BETA)?;
    attrs.insert(m, "beta", &Value::Str(beta))?;
    attrs.insert(m, "gamma", &Value::Float(2.5))?;
    attrs.insert(m, "delta", &Value::Bool(true))?;
    attrs.insert(m, "epsilon", &Value::Nil(()))?;

    // A nested group: one Box holding an Attributes of its own, filled in
    // where it lives
    let group = RustGroup::new(m, &RustAttrs::default())?;
    attrs.insert(m, "zeta", &Value::Group(group))?;
    group.insert(m, "nested_int", &Value::Int(77))?;
    let nested = RustString::from_str(m, GUEST_NESTED)?;
    group.insert(m, "nested_str", &Value::Str(nested))?;

    // A list: one Box holding a Vec of values
    let list = RustList::new(m, &RustVec::<RustValue>::default())?;
    attrs.insert(m, "eta", &Value::List(list))?;
    list.push(m, &Value::Int(1))?;
    list.push(m, &Value::Int(2))?;
    // A variant can also be built in place
    list.push(m, &Value::Nil(()))?;
    list.get(m)?
        .get(2)
        .unwrap()
        .insert::<RustString, _>(m, "three")?;
    Ok(())
}

/// The guest program, built once per test binary. `--wrap` makes the
/// linker rewrite every reference to memcpy, memset, memmove and memcmp
/// (Rust's own and glibc's) to the `__wrap_*` stubs at the top of the
/// guest, which turn them into the host's native memory system calls.
fn rust_guest_binary() -> &'static [u8] {
    static BINARY: OnceLock<Vec<u8>> = OnceLock::new();
    BINARY.get_or_init(|| {
        let args = "-C opt-level=2 \
            -C link-arg=-Wl,--wrap=memcpy \
            -C link-arg=-Wl,--wrap=memset \
            -C link-arg=-Wl,--wrap=memmove \
            -C link-arg=-Wl,--wrap=memcmp \
            -C link-arg=-Wl,--wrap=__memcmpeq \
            -C link-arg=-Wl,--wrap=bcmp";
        testutil::build_rust_and_load(GUEST_PROGRAM, args).unwrap_or_else(|e| panic!("{e}"))
    })
}

// ---------------------------------------------------------------------------
// Counting the block memory operations: the native memory system calls with
// a counter chained onto each, so that a test can prove the work happened on
// the host (a guest-local memcpy would get the bytes right too). C++ keeps
// the counters in globals; tests run concurrently here, so they live in the
// machine's userdata.
// ---------------------------------------------------------------------------

const MEMCPY: usize = 0;
const MEMSET: usize = 1;
const MEMMOVE: usize = 2;
const MEMCMP: usize = 3;
const MEMOP_COUNT: usize = 4;

/// The test's state, in the machine's userdata.
#[derive(Default)]
struct GuestState {
    memop_counts: [u64; MEMOP_COUNT],
    memop_originals: [Option<SyscallHandler>; MEMOP_COUNT],
    lent_str: String,
    lent_vec: Vec<String>,
    lent_attrs_checksum: u64,
    lent_attrs_size: usize,
}

fn state(m: &mut Machine) -> &mut GuestState {
    m.userdata_mut::<GuestState>().unwrap()
}

fn counting_memory_syscall<const OP: usize>(m: &mut Machine) -> libriscv::Result<()> {
    let s = state(m);
    s.memop_counts[OP] += 1;
    let original = s.memop_originals[OP].unwrap();
    original(m)
}

/// The native memory system calls, with a counter chained onto each.
fn install_counted_memory_syscalls(m: &mut Machine) {
    m.setup_native_memory(NATIVE_MEMORY);
    for op in 0..MEMOP_COUNT {
        let original = m.syscall_handler(NATIVE_MEMORY + op);
        state(m).memop_originals[op] = original;
    }
    m.install_syscall_handlers([
        (
            NATIVE_MEMORY + MEMCPY,
            counting_memory_syscall::<MEMCPY> as SyscallHandler,
        ),
        (NATIVE_MEMORY + MEMSET, counting_memory_syscall::<MEMSET>),
        (NATIVE_MEMORY + MEMMOVE, counting_memory_syscall::<MEMMOVE>),
        (NATIVE_MEMORY + MEMCMP, counting_memory_syscall::<MEMCMP>),
    ]);
}

/// A baseline of the counters, so a test can count its own calls.
struct MemoryOpCounter([u64; MEMOP_COUNT]);

impl MemoryOpCounter {
    fn new(guest: &mut RustGuest) -> Self {
        Self(state(&mut guest.machine).memop_counts)
    }
    fn get(&self, guest: &mut RustGuest, op: usize) -> u64 {
        state(&mut guest.machine).memop_counts[op] - self.0[op]
    }
}

/// A block of raw bytes on the native heap. A VM call string argument is
/// zero-terminated, which a Rust &str isn't, so the tests hand the guest an
/// address and a length instead.
struct GuestBytes {
    addr: u64,
    len: u64,
}

impl GuestBytes {
    fn new(m: &mut Machine, bytes: &[u8]) -> Self {
        let addr = u64::from(
            m.arena_mut()
                .unwrap()
                .malloc(bytes.len() as u64 + 1)
                .unwrap(),
        );
        assert_ne!(addr, 0);
        m.copy_to_guest(addr, bytes).unwrap();
        Self {
            addr,
            len: bytes.len() as u64,
        }
    }
    fn free(self, m: &mut Machine) {
        m.arena_mut().unwrap().free(self.addr as u32);
    }
}

// ---------------------------------------------------------------------------
// The host functions of the guest program, called with counted dyncalls
// ---------------------------------------------------------------------------

fn host_pause(m: &mut Machine) -> libriscv::Result<()> {
    m.stop();
    Ok(())
}

/// The guest hands the host a String it owns, and the host fills it in.
fn host_fill_string(m: &mut Machine) -> libriscv::Result<()> {
    let string = GuestPtr::<RustString>::new(m.sysarg(0));
    string.assign(m, HOST_TEXT)
}

/// The same for a Vec<u32>.
fn host_fill_vec(m: &mut Machine) -> libriscv::Result<()> {
    let vec = GuestPtr::<RustVec<u32>>::new(m.sysarg(0));
    for i in 1..=10u32 {
        vec.push(m, &i)?;
    }
    Ok(())
}

/// The guest lends the host a &str: the address and the length are two
/// registers, which is how Rust passes one.
fn host_take_str(m: &mut Machine) -> libriscv::Result<()> {
    let text = GuestRustStr::from_raw_parts(m.sysarg(0), m.sysarg(1)).to_string_lossy(m)?;
    state(m).lent_str = text;
    Ok(())
}

/// The guest lends the host a whole Vec<String>, by address. The host only
/// reads it: the guest still owns it.
fn host_take_vec(m: &mut Machine) -> libriscv::Result<()> {
    let vec = GuestPtr::<RustVec<RustString>>::new(m.sysarg(0));
    let names = vec.get(m)?.to_host(m)?;
    state(m).lent_vec = names;
    Ok(())
}

/// The guest lends the host its whole attribute tree, by address, and the
/// host walks it in place.
fn host_take_attrs(m: &mut Machine) -> libriscv::Result<()> {
    let attrs = GuestPtr::<RustAttrs>::new(m.sysarg(0)).get(m)?;
    // A script is adversarial input, so the shape is checked first
    attrs.validate(m, 4096)?;
    assert!(attrs.is_sorted(m)?);

    let len = attrs.len();
    let checksum = host_checksum_attrs(m, &attrs);

    // Reading individual keys goes through the same guest memory
    let at = |key: &str| attrs.get(m, key).unwrap().unwrap().get(m).unwrap();
    assert_eq!(at("alpha").get::<i64>(), Some(&-1234));
    assert_eq!(
        at("beta").get::<RustString>().unwrap().to_str(m)?,
        GUEST_BETA
    );
    assert_eq!(at("gamma").get::<f64>(), Some(&2.5));
    assert_eq!(at("delta").get::<bool>(), Some(&true));
    assert!(at("epsilon").is::<()>());
    assert!(attrs.get(m, "missing")?.is_none());

    let group = at("zeta").get::<RustGroup>().unwrap().get(m)?;
    assert_eq!(group.len(), 2);
    let gat = |key: &str| group.get(m, key).unwrap().unwrap().get(m).unwrap();
    assert_eq!(gat("nested_int").get::<i64>(), Some(&77));
    assert_eq!(
        gat("nested_str").get::<RustString>().unwrap().to_str(m)?,
        GUEST_NESTED
    );

    let list = at("eta").get::<RustList>().unwrap().get(m)?;
    assert_eq!(list.len(), 3);
    assert_eq!(
        list.get(2)
            .unwrap()
            .get(m)?
            .get::<RustString>()
            .unwrap()
            .to_str(m)?,
        "three"
    );

    let s = state(m);
    s.lent_attrs_size = len;
    s.lent_attrs_checksum = checksum;
    Ok(())
}

/// The guest hands the host an attribute tree it owns, and the host builds
/// the whole thing into it.
fn host_fill_attrs(m: &mut Machine) -> libriscv::Result<()> {
    let ptr = GuestPtr::<RustAttrs>::new(m.sysarg(0));
    ptr.modify(m, host_build_attrs)
}

/// A booted Rust guest, paused inside main() with its runtime initialized
/// and its heap taken over by the native heap.
struct RustGuest {
    machine: Machine,
}

impl RustGuest {
    fn new() -> Self {
        Self::with_options(MachineOptions::default())
    }

    fn with_options(options: MachineOptions) -> Self {
        let mut machine = Machine::new(rust_guest_binary().into(), options).unwrap();
        machine.set_userdata(GuestState::default());
        machine.set_printer(|_, _| {});
        let heap = machine.memory.mmap_allocate(HEAP_SIZE);
        machine
            .setup_native_heap(NATIVE_HEAP, heap, HEAP_SIZE)
            .unwrap();
        // memcpy, memset, memmove and memcmp, done natively by the host
        install_counted_memory_syscalls(&mut machine);
        machine.setup_linux_syscalls(true, true);
        // Rust's standard library locks with futexes even single-threaded
        machine.setup_posix_threads();
        machine
            .setup_linux(&["rust_guest"], &["LC_ALL=C", "USER=root"])
            .unwrap();
        machine.install_syscall_handlers([
            (HOST_PAUSE, host_pause as SyscallHandler),
            (HOST_FILL_STRING, host_fill_string),
            (HOST_FILL_VEC, host_fill_vec),
            (HOST_TAKE_STR, host_take_str),
            (HOST_TAKE_VEC, host_take_vec),
            (HOST_TAKE_ATTRS, host_take_attrs),
            (HOST_FILL_ATTRS, host_fill_attrs),
        ]);

        machine.simulate(MAX_INSTRUCTIONS).unwrap();
        // main() doesn't return: it stops the machine from the pause
        // dyncall, with everything initialized and ready to be called into
        assert!(!machine.instruction_limit_reached());
        assert!(machine.stopped());
        Self { machine }
    }

    /// Booted, or `None` (the test is skipped) without a Rust toolchain.
    fn boot() -> Option<Self> {
        testutil::need_rust_toolchain().then(Self::new)
    }

    fn call(&mut self, func: &str, args: impl libriscv::VmCallArgs) -> u64 {
        self.machine.vmcall(func, args).unwrap()
    }

    /// Run through every collection once, so that the runtime's lazy
    /// allocations aren't counted against a test.
    fn warmup(&mut self) {
        self.call("rust_warmup", ());
    }

    /// Live native heap allocations, to prove that a test leaks nothing.
    fn live_allocations(&self) -> u32 {
        let arena = self.machine.arena().unwrap();
        arena.allocation_counter() - arena.deallocation_counter()
    }
}

// ===========================================================================
// Worked examples: one test per way of moving a Rust value across the
// boundary. Everything after them is verification.
// ===========================================================================

/// Everything the guest does through the host, under the JIT: counted
/// dyncalls are compiled as calls of their handlers, which see the exact
/// instruction counter and can stop the machine on the spot. The results
/// and the work done on the host match the interpreter.
///
/// So do the instruction counts: both run a dyncall as a system call with
/// the counter published, so the handlers' penalties (the native heap's
/// 2000 per call, memcpy's 2 per byte) count in both.
#[test]
#[cfg(feature = "jit")]
fn the_host_functions_run_under_the_jit() {
    if !testutil::need_rust_toolchain() {
        return;
    }
    let run = |jit: bool| {
        let mut guest = RustGuest::with_options(MachineOptions {
            jit,
            ..Default::default()
        });
        let boot = guest.machine.instruction_counter();
        guest.warmup();
        let allocs_before = guest.live_allocations();
        let ops = MemoryOpCounter::new(&mut guest);
        // Counted calls, so that the instruction counts can be compared
        let mut counted = |func: &str, arg: usize| {
            let m = &mut guest.machine;
            let result = m.vmcall_limited(1 << 30, func, arg).unwrap();
            (result, m.instruction_counter())
        };
        let counts = [
            counted("rust_vec_capacity_after", 9),
            counted("rust_memcpy_roundtrip", 4096),
            counted("rust_u8vec_capacity_after", 17),
        ];
        let results = [
            guest.call("rust_host_fills_string", ()),
            guest.call("rust_host_fills_vec", ()),
            guest.call("rust_attrs_build", ()),
            guest.call("rust_attrs_host_fills", ()),
            guest.call("rust_memset_fill", (4096usize, 7u8)),
            guest.call("rust_memcpy_roundtrip", 4096usize),
            guest.call("rust_memmove_overlap", (4096usize, 13usize, true)),
            guest.call("rust_memcmp_order", (4096usize, 100usize, 0u8)),
        ];
        guest.call("rust_lends_str", ());
        guest.call("rust_lends_vec", ());
        let memops: Vec<u64> = (0..MEMOP_COUNT).map(|op| ops.get(&mut guest, op)).collect();
        assert_eq!(guest.live_allocations(), allocs_before, "jit={jit}");
        let s = state(&mut guest.machine);
        assert_eq!(s.lent_str, LENT_TEXT, "jit={jit}");
        assert_eq!(s.lent_vec, LENT_NAMES, "jit={jit}");
        assert_eq!(s.lent_attrs_checksum, results[2], "jit={jit}");
        (boot, counts, results, memops)
    };
    assert_eq!(run(true), run(false));
}

/// The host builds a Rust String on the native heap and calls the guest
/// with it. A box owns the three words *and* the bytes, and passing it to a
/// VM call passes its address, which is what a `&String` or
/// `*const String` parameter expects.
#[test]
fn example_the_host_builds_a_string_and_a_vec_for_the_guest() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    const TEXT: &str = "Hello from the host";
    // The Rust runtime allocates lazily, which would move the allocation
    // balance underneath the test
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    // fn rust_string_checksum(string: *const String) -> u64
    let text = RustBox::<RustString>::new(m, TEXT).unwrap();
    assert_eq!(
        m.vmcall("rust_string_checksum", text).unwrap(),
        fnv1a(TEXT.as_bytes())
    );

    // fn rust_vec_sum(vec: *const Vec<u32>) -> u64
    let vec = RustBox::<RustVec<u32>>::new(m, &[1u32, 2, 3, 4][..]).unwrap();
    assert_eq!(m.vmcall("rust_vec_sum", vec).unwrap(), 10);

    // A borrowed &str is two registers, the address and the length
    let borrowed = text.get(m).unwrap().as_str();
    assert_eq!(
        m.vmcall("rust_str_checksum", (borrowed.as_ptr(), borrowed.len()))
            .unwrap(),
        fnv1a(TEXT.as_bytes())
    );

    text.free(m).unwrap();
    vec.free(m).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// A String or a Vec is too large for a register, so a Rust function that
/// returns one takes a hidden pointer to the return value as its first
/// argument: on the host, a box passed first.
#[test]
fn example_a_rust_function_that_returns_a_string_or_a_vec() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    // fn rust_make_greeting() -> String
    let greeting = RustBox::new(m, &RustString::default()).unwrap();
    m.vmcall("rust_make_greeting", greeting).unwrap();
    assert_eq!(
        greeting.get(m).unwrap().to_str(m).unwrap(),
        "A String returned by value from the guest"
    );

    // fn rust_make_squares(count: usize) -> Vec<u32>: the hidden pointer
    // takes the first register, so `count` is the second argument
    let squares = RustBox::new(m, &RustVec::<u32>::default()).unwrap();
    m.vmcall("rust_make_squares", (&squares, 5u32)).unwrap();
    assert_eq!(squares.to_vec(m).unwrap(), [0, 1, 4, 9, 16]);

    // Allocated by the guest, freed by the host, in the one shared heap
    greeting.free(m).unwrap();
    squares.free(m).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// The guest hands the host something in a dyncall. Neither transfers
/// ownership: the host reads the bytes where they are.
#[test]
fn example_the_guest_lends_data_to_the_host() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    // A &str, as an address and a length in two registers
    guest.call("rust_lends_str", ());
    assert_eq!(state(&mut guest.machine).lent_str, LENT_TEXT);

    // A whole Vec<String>, as the address of its three words
    guest.call("rust_lends_vec", ());
    assert_eq!(state(&mut guest.machine).lent_vec, LENT_NAMES);

    assert_eq!(guest.live_allocations(), allocs_before);
}

/// Collections nest: an element of a Vec<Vec<u32>> is itself the three
/// words of a Vec. Freeing the outer one frees them all.
#[test]
fn example_nested_rust_collections() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    let rows = RustBox::new(m, &RustVec::<RustVec<u32>>::default()).unwrap();
    for r in 0..3u32 {
        // A shallow push: the row hands its allocation over to the array
        let row = RustVec::<u32>::from_slice(m, &[r, r + 1, r + 2]).unwrap();
        rows.push(m, &row).unwrap();
    }
    assert_eq!(rows.len(m).unwrap(), 3);
    let third = rows.get(m).unwrap().get(2).unwrap();
    assert_eq!(third.to_vec(m).unwrap(), [2, 3, 4]);

    // fn rust_nested_sum(vec: *const Vec<Vec<u32>>) -> u64
    assert_eq!(
        m.vmcall("rust_nested_sum", rows).unwrap(),
        (1 + 2) + (1 + 2 + 3) + (2 + 3 + 4)
    );
    // The outer vector frees the three inner ones along with itself
    rows.free(m).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// Ownership crosses in both directions, because both sides allocate out of
/// the same heap: the host hands over a raw block it filled in, and the
/// guest grows and frees it through its global allocator.
#[test]
fn example_the_guest_takes_ownership_of_a_host_buffer() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    // A plain heap block, filled in by the host
    let buffer = u64::from(m.arena_mut().unwrap().malloc(16).unwrap());
    assert_ne!(buffer, 0);
    for i in 0..4u32 {
        m.memory
            .write_value(buffer + u64::from(i) * 4, &(10 * (i + 1)))
            .unwrap();
    }

    // A vector adopts it unchanged (Vec::from_raw_parts), boxed so that the
    // guest can be handed a &mut Vec<u32>
    let vec = RustBox::new(m, &RustVec::<u32>::from_raw_parts(buffer, 4, 4)).unwrap();
    let v = vec.get(m).unwrap();
    assert_eq!(v.as_ptr(), buffer);
    assert_eq!(v.len(), 4);
    assert_eq!(v.capacity(), 4);
    assert_eq!(m.vmcall("rust_vec_sum", vec).unwrap(), 100);

    // Pushing past the capacity makes the guest reallocate it through its
    // global allocator, which is the same heap
    m.vmcall("rust_vec_push", (&vec, 50u32)).unwrap();
    assert_eq!(vec.len(m).unwrap(), 5);
    assert!(vec.capacity(m).unwrap() >= 5);
    assert_eq!(m.vmcall("rust_vec_sum", vec).unwrap(), 150);

    vec.free(m).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

// ===========================================================================
// Verification of the host mirrors
// ===========================================================================

/// Rust doesn't stabilize the layout of its collections, so the guest is
/// asked: a release that moves a field fails here instead of silently
/// reading garbage everywhere else.
#[test]
fn rust_collection_layout_matches_the_host_mirrors() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    let m = &mut guest.machine;
    const PROBE_STRING_LEN: u64 = "A string that is definitely heap allocated".len() as u64;

    // 3 raw words + ptr/len/capacity for the Vec and the String, 2 raw words
    // + ptr/len for the slice and the &str, and the raw words of an empty
    // String and an empty Vec
    let addr = u64::from(m.arena_mut().unwrap().malloc(26 * 8).unwrap());
    m.vmcall("rust_probe_layout", addr).unwrap();
    let w = m.memory.memarray::<u64>(addr, 26, 26 * 8).unwrap().to_vec();

    // Vec<u32> = { capacity, ptr, len }
    assert_eq!(w[0], w[5]); // the capacity comes first
    assert_eq!(w[1], w[3]); // then the pointer
    assert_eq!(w[2], w[4]); // then the length
    assert_eq!(w[4], 3);
    assert!(w[5] >= 3);

    // String = Vec<u8>, the same three words
    assert_eq!(w[6], w[11]);
    assert_eq!(w[7], w[9]);
    assert_eq!(w[8], w[10]);
    assert_eq!(w[10], PROBE_STRING_LEN);

    // &[u32] and &str = { ptr, len }
    assert_eq!(w[12], w[14]);
    assert_eq!(w[13], w[15]);
    assert_eq!(w[15], 3);
    assert_eq!(w[16], w[18]);
    assert_eq!(w[17], w[19]);
    assert_eq!(w[19], PROBE_STRING_LEN);

    // The raw words read through the host mirrors agree with what the Rust
    // API reported for the same collection
    let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    let vec = RustVec::<u32>::read_guest(&bytes[0..]);
    assert_eq!(vec.capacity() as u64, w[5]);
    assert_eq!(vec.as_ptr(), w[3]);
    assert_eq!(vec.len() as u64, w[4]);

    let string = RustString::read_guest(&bytes[6 * 8..]);
    assert_eq!(string.capacity() as u64, w[11]);
    assert_eq!(string.as_ptr(), w[9]);
    assert_eq!(string.len() as u64, w[10]);

    let strref = GuestRustStr::read_guest(&bytes[16 * 8..]);
    assert_eq!(strref.as_ptr(), w[18]);
    assert_eq!(strref.len() as u64, w[19]);

    // An empty collection has a capacity of zero and a dangling, aligned,
    // never null pointer
    assert_eq!(w[20], 0);
    assert_eq!(w[21], RustString::DANGLING);
    assert_eq!(w[22], 0);
    assert_eq!(w[23], 0);
    assert_eq!(w[24], RustVec::<u32>::DANGLING);
    assert_eq!(w[25], 0);

    m.arena_mut().unwrap().free(addr as u32);
}

const GROWTH_PUSHES: [usize; 12] = [0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 33, 100];

/// The host grows collections exactly like Rust's RawVec: both sides push one element
/// at a time and must end up with the same capacity, or a collection the
/// host builds reallocates the moment the guest touches it.
#[test]
fn rust_rawvec_growth_matches_the_host() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    let m = &mut guest.machine;

    for pushes in GROWTH_PUSHES {
        let vec = RustBox::new(m, &RustVec::<u32>::default()).unwrap();
        for i in 0..pushes {
            vec.push(m, &(i as u32)).unwrap();
        }
        let guest_capacity = m.vmcall("rust_vec_capacity_after", pushes).unwrap();
        assert_eq!(
            vec.len(m).unwrap(),
            pushes,
            "Vec<u32> after {pushes} pushes"
        );
        assert_eq!(
            vec.capacity(m).unwrap() as u64,
            guest_capacity,
            "Vec<u32> after {pushes} pushes"
        );
        vec.free(m).unwrap();

        let string = RustBox::new(m, &RustString::default()).unwrap();
        for _ in 0..pushes {
            string.push_str(m, "x").unwrap();
        }
        let guest_capacity = m.vmcall("rust_string_capacity_after", pushes).unwrap();
        let s = string.get(m).unwrap();
        assert_eq!(s.len(), pushes, "String after {pushes} pushes");
        assert_eq!(
            s.capacity() as u64,
            guest_capacity,
            "String after {pushes} pushes"
        );
        string.free(m).unwrap();
    }

    // The smallest allocation Rust makes: 8 bytes, 4 elements of 4 bytes
    assert_eq!(RustString::MIN_CAPACITY, 8);
    assert_eq!(RustVec::<u32>::MIN_CAPACITY, 4);
}

#[test]
fn vm_calls_with_a_rust_string() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    const TEXT: &str = "Hello from the host, Rust World!";
    const APPENDED: &str = " ...and some more text on the end";
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    for _ in 0..10 {
        let string = RustBox::new(m, &RustString::default()).unwrap();
        let s = string.get(m).unwrap();
        assert!(s.is_empty());
        assert_eq!(s.capacity(), 0);
        assert_eq!(s.as_ptr(), RustString::DANGLING);

        string.assign(m, TEXT).unwrap();
        let s = string.get(m).unwrap();
        assert_eq!(s.len(), TEXT.len());
        assert_eq!(s.as_bytes(m).unwrap(), TEXT.as_bytes());
        assert_eq!(s.to_string_lossy(m).unwrap(), TEXT);

        // The guest reads the string the host built
        assert_eq!(
            m.vmcall("rust_string_len", string).unwrap(),
            TEXT.len() as u64
        );
        assert_eq!(
            m.vmcall("rust_string_checksum", string).unwrap(),
            fnv1a(TEXT.as_bytes())
        );

        // The guest appends to it, which reallocates it in the heap
        let more = GuestBytes::new(m, APPENDED.as_bytes());
        m.vmcall("rust_string_append", (&string, more.addr, more.len))
            .unwrap();
        let s = string.get(m).unwrap();
        assert_eq!(s.to_str(m).unwrap(), format!("{TEXT}{APPENDED}"));
        assert!(s.capacity() >= TEXT.len() + APPENDED.len());
        more.free(m);

        // A string passed by value ends up on the guest stack, and the
        // guest gets a pointer to it, like a &String argument
        let mut onstack = RustString::from_str(m, TEXT).unwrap();
        assert_eq!(
            m.vmcall("rust_string_checksum", onstack).unwrap(),
            fnv1a(TEXT.as_bytes())
        );
        onstack.free(m).unwrap();

        string.free(m).unwrap();
    }
    assert_eq!(guest.live_allocations(), allocs_before);
}

#[test]
fn a_rust_string_the_guest_builds_and_the_host_frees() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    const TEXT: &str = "A string that starts life on the host side";
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    for _ in 0..10 {
        // The guest writes a String of its own at an address the host picked
        let string = RustBox::new(m, &RustString::default()).unwrap();
        let text = GuestBytes::new(m, TEXT.as_bytes());
        m.vmcall("rust_string_make", (&string, text.addr, text.len))
            .unwrap();
        let s = string.get(m).unwrap();
        assert_eq!(s.to_str(m).unwrap(), format!("{TEXT} (built by the guest)"));
        assert!(s.capacity() >= s.len());
        text.free(m);
        // Freed with the guest's own allocation
        string.free(m).unwrap();

        // And the other way around: the guest drops a host-built string
        let hostside = RustBox::<RustString>::new(m, TEXT).unwrap();
        assert_ne!(hostside.get(m).unwrap().capacity(), 0);
        m.vmcall("rust_string_drop", hostside).unwrap();
        let s = hostside.get(m).unwrap();
        assert!(s.is_empty());
        assert_eq!(s.capacity(), 0);
        assert_eq!(s.as_ptr(), RustString::DANGLING);
        hostside.free(m).unwrap();
    }
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// The guest can take ownership of memory the host allocated, and keep it
/// after the call returns. The moved-from string is left empty.
#[test]
fn rust_string_ownership_moves_to_the_guest() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    const TEXT: &str = "This string is heap-allocated and becomes guest-owned!";
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    let string = RustBox::<RustString>::new(m, TEXT).unwrap();
    assert_eq!(string.get(m).unwrap().to_str(m).unwrap(), TEXT);

    m.vmcall("rust_take_string", string).unwrap();
    // The guest moved out of it, leaving String::new() behind
    let s = string.get(m).unwrap();
    assert!(s.is_empty());
    assert_eq!(s.capacity(), 0);
    assert_eq!(s.as_ptr(), RustString::DANGLING);
    // ... and it still owns the bytes the host allocated
    assert_eq!(
        m.vmcall("rust_stored_checksum", ()).unwrap(),
        fnv1a(TEXT.as_bytes())
    );

    string.free(m).unwrap();
    // The scoped string is gone, and the guest's copy is still valid
    assert_eq!(
        m.vmcall("rust_stored_checksum", ()).unwrap(),
        fnv1a(TEXT.as_bytes())
    );
    assert_eq!(guest.live_allocations(), allocs_before + 1);

    // The guest drops it, which returns the host's allocation to the heap
    guest.call("rust_release_stored", ());
    assert_eq!(guest.call("rust_stored_checksum", ()), 0);
    assert_eq!(guest.live_allocations(), allocs_before);
}

#[test]
fn vm_calls_with_a_rust_vec() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    for _ in 0..10 {
        let vec = RustBox::<RustVec<u32>>::new(m, &[1u32, 2, 3, 4, 5][..]).unwrap();
        assert_eq!(vec.len(m).unwrap(), 5);
        assert_eq!(vec.to_vec(m).unwrap(), [1, 2, 3, 4, 5]);
        assert_eq!(m.vmcall("rust_vec_sum", vec).unwrap(), 15);

        // The guest pushes, which reallocates the vector in the heap
        for v in 6..=20u32 {
            m.vmcall("rust_vec_push", (&vec, v)).unwrap();
        }
        assert_eq!(vec.len(m).unwrap(), 20);
        assert_eq!(m.vmcall("rust_vec_sum", vec).unwrap(), 20 * 21 / 2);
        assert_eq!(vec.to_vec(m).unwrap()[19], 20);

        // The host reads it back, straight out of guest memory
        let v = vec.get(m).unwrap();
        let sum: u64 = v.view(m).unwrap().iter().map(u64::from).sum();
        assert_eq!(sum, 20 * 21 / 2);

        // Shrinking keeps the allocation
        let capacity = vec.capacity(m).unwrap();
        vec.truncate(m, 3).unwrap();
        assert_eq!(vec.len(m).unwrap(), 3);
        assert_eq!(vec.capacity(m).unwrap(), capacity);
        assert_eq!(m.vmcall("rust_vec_sum", vec).unwrap(), 6);

        vec.clear(m).unwrap();
        assert!(vec.is_empty(m).unwrap());
        assert_eq!(m.vmcall("rust_vec_sum", vec).unwrap(), 0);

        // A vector the guest builds at an address the host picked
        let made = RustBox::new(m, &RustVec::<u32>::default()).unwrap();
        m.vmcall("rust_vec_make", (&made, 8u32)).unwrap();
        let squares = made.get(m).unwrap();
        assert_eq!(squares.len(), 8);
        for k in 0..8 {
            let square = squares.get(k).unwrap();
            assert_eq!(square.get(m).unwrap(), (k * k) as u32);
        }
        made.free(m).unwrap();
        vec.free(m).unwrap();
    }
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// The checksum the guest's rust_strvec_checksum computes.
fn strvec_checksum<S: AsRef<str>>(strings: &[S]) -> u64 {
    let mut hash = fnv1a(&[strings.len() as u8]);
    for s in strings {
        hash = hash.wrapping_add(fnv1a(s.as_ref().as_bytes()));
    }
    hash
}

#[test]
fn vm_calls_with_a_rust_vec_of_strings() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    const NAMES: [&str; 4] = [
        "one",
        "two",
        "three",
        "a name long enough to need its own allocation",
    ];
    const EXTRA: &str = "an element the guest pushed on the end";
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    for _ in 0..10 {
        let names = RustBox::<RustVec<RustString>>::new(m, &NAMES[..]).unwrap();
        assert_eq!(names.len(m).unwrap(), NAMES.len());
        assert_eq!(names.get(m).unwrap().to_host(m).unwrap(), NAMES);
        assert_eq!(
            m.vmcall("rust_strvec_checksum", names).unwrap(),
            strvec_checksum(&NAMES)
        );

        // The guest pushes a String of its own into the host's vector
        let extra = GuestBytes::new(m, EXTRA.as_bytes());
        m.vmcall("rust_strvec_push", (&names, extra.addr, extra.len))
            .unwrap();
        extra.free(m);
        let all = [&NAMES[..], &[EXTRA]].concat();
        assert_eq!(names.get(m).unwrap().to_host(m).unwrap(), all);
        assert_eq!(
            m.vmcall("rust_strvec_checksum", names).unwrap(),
            strvec_checksum(&all)
        );

        // And joins them into a String at an address the host picked
        let joined = RustBox::new(m, &RustString::default()).unwrap();
        m.vmcall("rust_strvec_join", (&names, &joined)).unwrap();
        assert_eq!(joined.get(m).unwrap().to_str(m).unwrap(), all.join(", "));

        joined.free(m).unwrap();
        names.free(m).unwrap();
    }
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// A &str or &[T] argument is two registers; stored in memory it is
/// `{ ptr, len }`.
#[test]
fn borrowed_rust_str_and_slices() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    const TEXT: &str = "Borrowed by the guest, owned by the host";
    let m = &mut guest.machine;

    let string = RustBox::<RustString>::new(m, TEXT).unwrap();
    let vec = RustBox::<RustVec<u32>>::new(m, &[10u32, 20, 30, 40][..]).unwrap();

    // As arguments: two registers, not a pointer to a fat pointer
    let borrowed = string.get(m).unwrap().as_str();
    assert_eq!(borrowed.len(), TEXT.len());
    assert_eq!(borrowed.to_str(m).unwrap(), TEXT);
    assert_eq!(
        m.vmcall("rust_str_checksum", (borrowed.as_ptr(), borrowed.len()))
            .unwrap(),
        fnv1a(TEXT.as_bytes())
    );

    let slice = vec.get(m).unwrap().as_slice();
    assert_eq!(slice.len(), 4);
    assert_eq!(slice.view(m).unwrap().get(2), Some(30));
    assert_eq!(
        m.vmcall("rust_slice_sum", (slice.as_ptr(), slice.len()))
            .unwrap(),
        100
    );

    // Stored in memory they are { ptr, len }: the guest writes them, the
    // host mirrors read them back
    let addr = u64::from(m.arena_mut().unwrap().malloc(32).unwrap());
    m.vmcall("rust_borrow_into", (&string, &vec, addr)).unwrap();

    let guest_str: GuestRustStr = m.memory.read_value(addr).unwrap();
    assert_eq!(guest_str.as_ptr(), string.get(m).unwrap().as_ptr());
    assert_eq!(guest_str.len(), TEXT.len());
    assert_eq!(guest_str.to_str(m).unwrap(), TEXT);

    let guest_slice: GuestRustSlice<u32> = m.memory.read_value(addr + 16).unwrap();
    assert_eq!(guest_slice.as_ptr(), vec.get(m).unwrap().as_ptr());
    assert_eq!(guest_slice.len(), 4);
    assert_eq!(guest_slice.view(m).unwrap().get(3), Some(40));

    // An empty borrow is a dangling pointer and a zero length
    let empty = GuestRustStr::default();
    assert!(empty.is_empty());
    assert_eq!(empty.as_ptr(), RustString::DANGLING);
    assert!(empty.as_bytes(m).unwrap().is_empty());

    m.arena_mut().unwrap().free(addr as u32);
    string.free(m).unwrap();
    vec.free(m).unwrap();
}

/// The guest owns the collection, and hands the host its address in a
/// dyncall.
#[test]
fn host_fills_in_guest_owned_rust_collections() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    for _ in 0..10 {
        let expected = fnv1a(HOST_TEXT.as_bytes())
            .wrapping_add(fnv1a(format!("{HOST_TEXT}{GUEST_SUFFIX}").as_bytes()));
        assert_eq!(guest.call("rust_host_fills_string", ()), expected);
        // 1..10 summed, plus the length after the guest pushed one more
        assert_eq!(guest.call("rust_host_fills_vec", ()), 55 + 11);
    }
    // The guest dropped every collection, releasing the host's memory
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// Raw bytes go into a String unchecked, and to_str() is the check on the
/// way out: it must agree with the guest on every edge case.
#[test]
fn rust_utf8_validation_matches_the_guest() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    let cases: [&[u8]; 25] = [
        b"",                       // empty
        b"Hello, World!",          // plain ASCII
        b"a\0b",                   // an embedded zero is just a character
        b"\x7F",                   // last single byte
        b"\xC2\x80",               // U+0080, the first two-byte sequence
        b"\xDF\xBF",               // U+07FF, the last one
        b"\xE0\xA0\x80",           // U+0800, the first three-byte sequence
        b"\xEF\xBF\xBF",           // U+FFFF
        b"\xF0\x90\x80\x80",       // U+10000, the first four-byte sequence
        b"\xF4\x8F\xBF\xBF",       // U+10FFFF, the last code point
        b"\xF0\x9F\x98\x80 smile", // an emoji
        b"\x80",                   // a lone continuation byte
        b"\xC0\xAF",               // an overlong '/'
        b"\xC1\xBF",               // an overlong sequence
        b"\xE0\x9F\xBF",           // an overlong three-byte sequence
        b"\xF0\x8F\xBF\xBF",       // an overlong four-byte sequence
        b"\xED\xA0\x80",           // U+D800, a surrogate half
        b"\xED\xBF\xBF",           // U+DFFF, the other end
        b"\xF4\x90\x80\x80",       // beyond U+10FFFF
        b"\xF5\x80\x80\x80",       // beyond the four-byte range entirely
        b"\xFE",                   // never valid
        b"\xFF",                   // never valid
        b"\xE2\x82",               // truncated three-byte sequence
        b"\xF0\x9F\x98",           // truncated four-byte sequence
        b"ok\xE2\x82",             // truncated at the very end
    ];
    let m = &mut guest.machine;
    for (i, bytes) in cases.iter().enumerate() {
        let buffer = GuestBytes::new(m, bytes);
        let guest_says = m
            .vmcall("rust_string_is_utf8", (buffer.addr, buffer.len))
            .unwrap()
            != 0;
        buffer.free(m);
        let string = RustBox::<RustString>::new(m, *bytes).unwrap();
        let host_says = string.get(m).unwrap().to_str(m).is_ok();
        string.free(m).unwrap();
        assert_eq!(host_says, guest_says, "case {i}");
    }
}

/// A 2048-byte element, the guest's `Big`.
#[derive(Clone, Copy)]
struct Big([u8; 2048]);

impl Default for Big {
    fn default() -> Self {
        Self([0; 2048])
    }
}
impl GuestPod for Big {
    const SIZE: usize = 2048;
    const ALIGN: usize = 1;
    fn write_guest(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Self(bytes[..2048].try_into().unwrap())
    }
}
impl libriscv::guest::GuestType for Big {}

/// The minimum capacity has three branches: 8 for a byte, 4 up to 1024
/// bytes, and 1 above that.
#[test]
fn rust_rawvec_minimum_capacity_per_element_size() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    assert_eq!(RustVec::<u8>::MIN_CAPACITY, 8);
    assert_eq!(RustVec::<u64>::MIN_CAPACITY, 4);
    assert_eq!(RustVec::<Big>::MIN_CAPACITY, 1);
    let m = &mut guest.machine;

    for pushes in [1usize, 2, 3, 5, 9, 17] {
        let vec = RustBox::new(m, &RustVec::<u8>::default()).unwrap();
        for i in 0..pushes {
            vec.push(m, &(i as u8)).unwrap();
        }
        assert_eq!(
            vec.capacity(m).unwrap() as u64,
            m.vmcall("rust_u8vec_capacity_after", pushes).unwrap(),
            "after {pushes} pushes"
        );
        vec.free(m).unwrap();

        let vec = RustBox::new(m, &RustVec::<u64>::default()).unwrap();
        for i in 0..pushes {
            vec.push(m, &(i as u64)).unwrap();
        }
        assert_eq!(
            vec.capacity(m).unwrap() as u64,
            m.vmcall("rust_u64vec_capacity_after", pushes).unwrap(),
            "after {pushes} pushes"
        );
        vec.free(m).unwrap();
    }

    // A large element gets no minimum: the first push allocates one
    for pushes in [1usize, 2, 3] {
        let vec = RustBox::new(m, &RustVec::<Big>::default()).unwrap();
        for _ in 0..pushes {
            vec.push(m, &Big::default()).unwrap();
        }
        assert_eq!(
            vec.capacity(m).unwrap() as u64,
            m.vmcall("rust_bigvec_capacity_after", pushes).unwrap(),
            "after {pushes} large pushes"
        );
        vec.free(m).unwrap();
    }
}

/// The remaining GuestRustVec operations, checked against the guest where
/// it can see them and against the heap where it can't.
#[test]
fn rust_vec_host_api_surface() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    // Constructing allocates exactly the length, like to_vec and vec![..]
    let mut exact = RustVec::<u32>::from_slice(m, &[1, 2, 3]).unwrap();
    assert_eq!(exact.capacity(), 3);
    exact.reserve_exact(m, 2).unwrap();
    assert_eq!(exact.capacity(), 5);
    assert_eq!(exact.to_vec(m).unwrap(), [1, 2, 3]);
    exact.free(m).unwrap();

    // Resizing an empty vector default-initializes its elements
    let vec = RustBox::new(m, &RustVec::<u32>::default()).unwrap();
    vec.resize(m, 4).unwrap();
    let v = vec.get(m).unwrap();
    assert_eq!(v.len(), 4);
    assert!(v.capacity() >= 4);
    assert_eq!(v.to_vec(m).unwrap(), [0, 0, 0, 0]);

    // Assigning replaces the contents, reusing the allocation when it fits
    let data = v.as_ptr();
    vec.assign(m, &[5u32, 6, 7][..]).unwrap();
    let v = vec.get(m).unwrap();
    assert_eq!(v.len(), 3);
    assert_eq!(v.as_ptr(), data);
    assert_eq!(m.vmcall("rust_vec_sum", vec).unwrap(), 18);

    // get() is the element in place, at its guest address
    assert_eq!(v.get(0).unwrap().address(), v.as_ptr());
    assert_eq!(v.get(2).unwrap().address(), v.as_ptr() + 8);
    assert_eq!(v.get(1).unwrap().get(m).unwrap(), 6);
    assert!(v.get(3).is_none());

    // Iterating reads the elements straight out of guest memory
    let sum: u64 = v
        .iter()
        .unwrap()
        .map(|p| u64::from(p.get(m).unwrap()))
        .sum();
    assert_eq!(sum, 18);
    let span = v.view(m).unwrap();
    assert_eq!(span.len(), 3);
    assert_eq!(span.get(2), Some(7));

    // pop() hands the last element over and keeps the allocation
    let capacity = v.capacity();
    assert_eq!(vec.pop(m).unwrap(), Some(7));
    assert_eq!(vec.len(m).unwrap(), 2);
    assert_eq!(vec.capacity(m).unwrap(), capacity);
    assert_eq!(m.vmcall("rust_vec_sum", vec).unwrap(), 11);

    // Growing with resize() default-initializes what it adds
    vec.resize(m, 5).unwrap();
    assert_eq!(vec.to_vec(m).unwrap(), [5, 6, 0, 0, 0]);
    assert_eq!(m.vmcall("rust_vec_sum", vec).unwrap(), 11);

    // reserve() makes room for more, and only ever grows
    vec.reserve(m, 64).unwrap();
    assert!(vec.capacity(m).unwrap() >= 5 + 64);
    let grown = vec.capacity(m).unwrap();
    vec.reserve(m, 2).unwrap();
    assert_eq!(vec.capacity(m).unwrap(), grown);
    assert_eq!(vec.len(m).unwrap(), 5);

    // An empty vector has nothing left to pop
    vec.clear(m).unwrap();
    assert!(vec.is_empty(m).unwrap());
    assert_eq!(vec.pop(m).unwrap(), None);

    // A slice of the whole vector, and the empty borrows: dangling but
    // never null, like the collections themselves
    vec.assign(m, &[1u32, 2][..]).unwrap();
    assert_eq!(vec.get(m).unwrap().as_slice().len(), 2);
    let empty_slice = GuestRustSlice::<u32>::default();
    assert!(empty_slice.is_empty());
    assert_eq!(empty_slice.as_ptr(), RustVec::<u32>::DANGLING);

    vec.free(m).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// The remaining GuestRustString operations. The capacity says whether an
/// allocation is reused or replaced, which the length can't.
#[test]
fn rust_string_host_api_surface() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    // Constructing allocates exactly the length, like String::from
    let string = RustBox::<RustString>::new(m, "initial").unwrap();
    let s = string.get(m).unwrap();
    assert_eq!(s.len(), 7);
    assert_eq!(s.capacity(), 7);
    let mut exact = RustString::from_str(m, "abc").unwrap();
    assert_eq!(exact.capacity(), 3);
    exact.reserve_exact(m, 2).unwrap();
    assert_eq!(exact.capacity(), 5);
    exact.free(m).unwrap();

    // Assigning reuses the allocation when the new text fits
    let (capacity, data) = (s.capacity(), s.as_ptr());
    string.assign(m, "short").unwrap();
    let s = string.get(m).unwrap();
    assert_eq!(s.to_str(m).unwrap(), "short");
    assert_eq!(s.capacity(), capacity);
    assert_eq!(s.as_ptr(), data);

    // clear() truncates to nothing and keeps the allocation
    string.clear(m).unwrap();
    let s = string.get(m).unwrap();
    assert!(s.is_empty());
    assert_eq!(s.capacity(), capacity);
    assert_eq!(s.as_ptr(), data);
    assert_eq!(m.vmcall("rust_string_len", string).unwrap(), 0);

    // ... and can be filled again without allocating
    string.push_str(m, "back").unwrap();
    let s = string.get(m).unwrap();
    assert_eq!(s.to_str(m).unwrap(), "back");
    assert_eq!(s.as_ptr(), data);

    // reserve() makes room for more, and only ever grows
    string.reserve(m, 256).unwrap();
    let grown = string.get(m).unwrap().capacity();
    assert!(grown >= 4 + 256);
    string.reserve(m, 1).unwrap();
    let s = string.get(m).unwrap();
    assert_eq!(s.capacity(), grown);
    assert_eq!(s.to_str(m).unwrap(), "back");
    assert_eq!(
        m.vmcall("rust_string_checksum", string).unwrap(),
        fnv1a(b"back")
    );

    // A borrowed &str copies out as an owned String, and an empty borrow
    // is dangling rather than null
    assert_eq!(s.as_str().to_string_lossy(m).unwrap(), "back");
    let empty_str = GuestRustStr::default();
    assert!(empty_str.is_empty());
    assert_eq!(empty_str.as_ptr(), RustString::DANGLING);
    assert!(empty_str.to_string_lossy(m).unwrap().is_empty());

    string.free(m).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// Every element of a Vec<String> owns an allocation, so every operation
/// that removes one has to release it (a Vec<u32> can't catch that).
#[test]
fn rust_vec_of_strings_releases_the_elements_it_drops() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    const NAMES: [&str; 3] = [
        "a string long enough to need its own allocation",
        "and another one just like it",
        "and a third",
    ];
    guest.warmup();
    let allocs_before = guest.live_allocations();

    let names = RustBox::<RustVec<RustString>>::new(&mut guest.machine, &NAMES[..]).unwrap();
    // The box itself, the element array, and one per string
    assert_eq!(
        guest.live_allocations(),
        allocs_before + 2 + NAMES.len() as u32
    );

    // pop() hands the last String over, and the host drops it
    let mut last = names.pop(&mut guest.machine).unwrap().unwrap();
    assert_eq!(last.to_str(&guest.machine).unwrap(), NAMES[2]);
    assert_eq!(guest.live_allocations(), allocs_before + 5);
    last.free(&mut guest.machine).unwrap();
    assert_eq!(names.len(&guest.machine).unwrap(), 2);
    assert_eq!(guest.live_allocations(), allocs_before + 4);

    // truncate() releases everything it drops
    names.truncate(&mut guest.machine, 1).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before + 3);

    // resize() growing leaves empty Strings, which own nothing yet
    names.resize(&mut guest.machine, 3).unwrap();
    assert_eq!(names.len(&guest.machine).unwrap(), 3);
    assert_eq!(guest.live_allocations(), allocs_before + 3);
    let m = &mut guest.machine;
    let v = names.get(m).unwrap();
    let last = v.get(2).unwrap().get(m).unwrap();
    assert!(last.is_empty());
    assert_eq!(last.as_ptr(), RustString::DANGLING);
    assert_eq!(v.to_host(m).unwrap(), [NAMES[0], "", ""]);

    // clear() releases them all, and keeps the array
    names.clear(m).unwrap();
    assert!(names.is_empty(m).unwrap());
    assert_eq!(guest.live_allocations(), allocs_before + 2);

    // extend_from_slice() builds a String per host string
    let m = &mut guest.machine;
    names.extend_from_slice(m, &NAMES).unwrap();
    assert_eq!(names.get(m).unwrap().to_host(m).unwrap(), NAMES);
    // Assigning replaces the contents, releasing the old elements
    names.assign(m, &NAMES[1..]).unwrap();
    assert_eq!(names.get(m).unwrap().to_host(m).unwrap(), NAMES[1..]);

    names.free(m).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

// ===========================================================================
// Box, the guest Rust enum, and the attribute tree
// ===========================================================================

/// Box<T> is one non-null pointer. The block the host allocates came from
/// the guest's global allocator, so Box::from_raw() on it is sound, and the
/// guest's drop returns it to the heap.
#[test]
fn example_a_box_crosses_the_boundary_in_both_directions() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    // The host builds a Box<i64> and gives it away: into_raw() hands over
    // ownership without freeing
    let boxed = GuestRustBox::<i64>::new(&mut guest.machine, &4321).unwrap();
    assert!(!boxed.is_null());
    assert_eq!(boxed.get(&guest.machine).unwrap(), 4321);
    assert_eq!(guest.live_allocations(), allocs_before + 1);

    let addr = boxed.into_raw();
    assert_eq!(guest.call("rust_box_take_i64", addr), 4321);
    // The guest dropped it
    assert_eq!(guest.live_allocations(), allocs_before);

    // And the other way: the guest allocates, the host takes ownership
    let addr = guest.call("rust_box_make_i64", 21i64);
    assert_eq!(guest.live_allocations(), allocs_before + 1);

    let m = &mut guest.machine;
    let from_guest = GuestRustBox::<i64>::from_raw(addr);
    assert_eq!(from_guest.get(m).unwrap(), 42);
    from_guest.free(m).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// `#[repr(C)] struct Boxes { numbers: Box<[u32]>, name: Box<str> }`
#[derive(Debug, Clone, Copy, Default)]
struct HostBoxes {
    numbers: GuestRustBoxedSlice<u32>,
    name: GuestRustBoxedStr,
}
impl GuestPod for HostBoxes {
    const SIZE: usize = 32;
    const ALIGN: usize = 8;
    fn write_guest(&self, out: &mut Vec<u8>) {
        self.numbers.write_guest(out);
        self.name.write_guest(out);
    }
    fn read_guest(bytes: &[u8]) -> Self {
        Self {
            numbers: GuestRustBoxedSlice::read_guest(bytes),
            name: GuestRustBoxedStr::read_guest(&bytes[16..]),
        }
    }
}

/// Box<[T]> and Box<str> are the fat pointers of a slice and a &str, with
/// ownership. The host builds a struct of both, hands over one pointer, and
/// the guest's drop glue frees all three allocations.
#[test]
fn example_the_host_builds_a_boxed_slice_and_a_boxed_str() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    const NUMBERS: [u32; 4] = [5, 10, 15, 20];
    const NAME: &str = "a boxed str the host allocated";
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    let addr = u64::from(
        m.arena_mut()
            .unwrap()
            .malloc(HostBoxes::SIZE as u64)
            .unwrap(),
    );
    assert_ne!(addr, 0);
    let boxes = HostBoxes {
        numbers: GuestRustBoxedSlice::from_slice(m, &NUMBERS).unwrap(),
        name: GuestRustBoxedStr::from_str(m, NAME).unwrap(),
    };
    m.memory.write_value(addr, &boxes).unwrap();

    assert_eq!(boxes.numbers.len(), NUMBERS.len());
    assert_eq!(boxes.numbers.to_vec(m).unwrap(), NUMBERS);
    assert_eq!(boxes.name.to_str(m).unwrap(), NAME);
    // The struct itself, the boxed slice and the boxed str
    assert_eq!(guest.live_allocations(), allocs_before + 3);

    let mut expected = fnv1a(NAME.as_bytes());
    for n in NUMBERS {
        expected = expected.wrapping_mul(131).wrapping_add(u64::from(n));
    }
    assert_eq!(guest.call("rust_boxes_take", addr), expected);
    // One from_raw and one drop released all three
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// The attribute tree, host to guest: built on the guest heap in its final
/// shape, taken by the guest with one from_raw, and dropped by the guest's
/// drop glue, nested group and list included.
#[test]
fn example_the_host_builds_an_attribute_tree_and_the_guest_takes_it() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    // A Box<Attributes>: what the guest calls from_raw() on
    let root = RustGroup::new(m, &RustAttrs::default()).unwrap();
    root.modify(m, host_build_attrs).unwrap();

    let tree = root.get(m).unwrap();
    assert_eq!(tree.len(), 7);
    assert!(tree.is_sorted(m).unwrap());
    let expected = host_checksum_attrs(m, &tree);

    // The guest computes the same checksum over the same bytes, and drops
    // the tree on its way out
    let addr = root.into_raw();
    assert_eq!(guest.call("rust_attrs_take", addr), expected);
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// The host keeps the tree and lends it for the call: build once, pass the
/// same address every time, allocate nothing per call.
#[test]
fn example_the_guest_borrows_a_host_owned_attribute_tree() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    let attrs = RustBox::new(m, &RustAttrs::default()).unwrap();
    attrs.modify(m, host_build_attrs).unwrap();
    let expected = host_checksum_attrs(m, &attrs.get(m).unwrap());

    // Twice with the same address: nothing is allocated or freed
    let allocs_built = guest.live_allocations();
    let m = &mut guest.machine;
    assert_eq!(m.vmcall("rust_attrs_borrow", attrs).unwrap(), expected);
    assert_eq!(m.vmcall("rust_attrs_borrow", attrs).unwrap(), expected);
    assert_eq!(guest.live_allocations(), allocs_built);

    // Freeing the box frees the tree, recursively
    attrs.free(&mut guest.machine).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// The guest builds a tree and lends it to the host, which walks it in
/// place.
#[test]
fn example_the_guest_lends_its_attribute_tree_to_the_host() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    // The guest returns its checksum, and the dyncall stored the one
    // the host computed over the same bytes
    let guest_checksum = guest.call("rust_attrs_build", ());
    let s = state(&mut guest.machine);
    assert_eq!(s.lent_attrs_size, 7);
    assert_eq!(s.lent_attrs_checksum, guest_checksum);

    // The guest still owned it and dropped it
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// The host fills in a tree the guest owns; the guest then inserts into it
/// and drops all of it.
#[test]
fn example_the_host_fills_in_a_guest_owned_attribute_tree() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    // Non-zero: the host-built entries survived the guest growing the same
    // vector
    assert_ne!(guest.call("rust_attrs_host_fills", ()), 0);
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// A guest that wants a real HashMap gets one for the price of a collect().
#[test]
fn the_guest_can_collect_the_tree_into_a_hashmap() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    let root = RustGroup::new(m, &RustAttrs::default()).unwrap();
    root.modify(m, host_build_attrs).unwrap();

    let tree = root.get(m).unwrap();
    let mut expected = tree.len() as u64;
    for key in ["alpha", "beta", "gamma"] {
        let value = tree.get(m, key).unwrap().unwrap().get(m).unwrap();
        expected = expected.wrapping_add(host_checksum_value(m, &value));
    }
    let addr = root.into_raw();
    assert_eq!(guest.call("rust_attrs_into_hashmap", addr), expected);
    // The guest's HashMap owned the keys and values, and dropping it
    // released everything the host allocated
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// `#[repr(C, u64)]` is requested, but only rustc can say whether it
/// delivered: the tags, payload offsets and sizes, from the guest.
#[test]
fn rust_enum_layout_matches_the_host_mirror() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    const WORDS: u64 = 21;
    let m = &mut guest.machine;
    let addr = u64::from(m.arena_mut().unwrap().malloc(WORDS * 8).unwrap());
    m.vmcall("rust_probe_enum_layout", addr).unwrap();
    let w = m
        .memory
        .memarray::<u64>(addr, WORDS, WORDS * 8)
        .unwrap()
        .to_vec();

    // The tag plus the widest variant, aligned like the payload
    assert_eq!(w[0], RustValue::SIZE as u64);
    assert_eq!(w[1], RustValue::ALIGN as u64);

    // The discriminants are the declaration order
    assert_eq!(w[2], RustValue::discriminant_of::<()>() as u64);
    assert_eq!(w[3], RustValue::discriminant_of::<i64>() as u64);
    assert_eq!(w[4], RustValue::discriminant_of::<f64>() as u64);
    assert_eq!(w[5], RustValue::discriminant_of::<bool>() as u64);
    assert_eq!(w[6], RustValue::discriminant_of::<RustString>() as u64);
    assert_eq!(w[7], RustValue::discriminant_of::<RustGroup>() as u64);
    assert_eq!(w[8], RustValue::discriminant_of::<RustList>() as u64);

    // Every payload starts right after the tag
    for &offset in &w[9..15] {
        assert_eq!(offset, RustValue::PAYLOAD_OFFSET as u64);
    }
    assert_eq!(w[9], 8);

    // The entry struct is a String followed by a Value
    type HostAttr = GuestRustAttr<RustValue>;
    assert_eq!(w[15], HostAttr::SIZE as u64);
    assert_eq!(w[16], 0);
    assert_eq!(w[17], HostAttr::VALUE_OFFSET as u64);

    // Attributes is a newtype over the Vec, and a Box is one word
    assert_eq!(w[18], RustAttrs::SIZE as u64);
    assert_eq!(w[19], RustGroup::SIZE as u64);
    assert_eq!(w[20], 0);

    m.arena_mut().unwrap().free(addr as u32);
}

/// The map on its own: sorted insertion, lookup, overwrite and removal.
#[test]
fn rust_attributes_insertion_lookup_and_removal() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    let attrs = RustBox::new(m, &RustAttrs::default()).unwrap();
    let mut map = attrs.get(m).unwrap();
    assert!(map.is_empty());
    assert!(map.get(m, "nothing").unwrap().is_none());

    // Inserted out of order, kept in key order
    map.insert(m, "gamma", &Value::Int(3)).unwrap();
    map.insert(m, "alpha", &Value::Int(1)).unwrap();
    map.insert(m, "beta", &Value::Int(2)).unwrap();
    assert_eq!(map.len(), 3);
    assert!(map.is_sorted(m).unwrap());

    let keys: Vec<&[u8]> = map.iter(m).unwrap().map(|e| e.unwrap().0).collect();
    assert_eq!(keys, [&b"alpha"[..], b"beta", b"gamma"]);

    // Overwriting with another variant, built in place
    let alpha = map.get(m, "alpha").unwrap().unwrap();
    assert!(alpha.is::<i64>(m).unwrap());
    map.entry_or_default(m, "alpha")
        .unwrap()
        .insert::<RustString, _>(m, "now a string")
        .unwrap();
    assert!(alpha.is::<RustString>(m).unwrap());
    let text = alpha.variant::<RustString>(m).unwrap().unwrap();
    assert_eq!(text.to_string_lossy(m).unwrap(), "now a string");
    assert!(alpha.variant::<i64>(m).unwrap().is_none());
    assert_eq!(map.len(), 3);

    // Overwriting releases what the old variant owned
    let allocs_with_string = guest.live_allocations();
    let m = &mut guest.machine;
    map.insert(m, "alpha", &Value::Float(1.5)).unwrap();
    assert_eq!(guest.live_allocations(), allocs_with_string - 1);
    let m = &mut guest.machine;
    let alpha = map.get(m, "alpha").unwrap().unwrap().get(m).unwrap();
    assert_eq!(alpha.get::<f64>(), Some(&1.5));

    // remove() closes the gap, keeps the order, and hands the value over
    let beta = map.remove(m, "beta").unwrap().unwrap();
    assert_eq!(beta.value(), Some(&Value::Int(2)));
    assert!(map.remove(m, "beta").unwrap().is_none());
    assert_eq!(map.len(), 2);
    assert!(map.is_sorted(m).unwrap());
    assert!(map.contains_key(m, "alpha").unwrap());
    assert!(map.contains_key(m, "gamma").unwrap());
    assert!(map.get(m, "beta").unwrap().is_none());

    // A whole host map at once, in one allocation for the entries
    let host_map = BTreeMap::from([
        ("one", Value::Int(1)),
        ("two", Value::Int(2)),
        ("three", Value::Int(3)),
    ]);
    map.clear(m).unwrap();
    map.extend(m, host_map).unwrap();
    assert_eq!(map.len(), 3);
    assert!(map.is_sorted(m).unwrap());
    let two = map.get(m, "two").unwrap().unwrap().get(m).unwrap();
    assert_eq!(two.get::<i64>(), Some(&2));

    map.clear(m).unwrap();
    assert!(map.is_empty());
    attrs.set(m, &map).unwrap();

    attrs.free(m).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// free() on a tree walks it: the group, the list, every string in both.
#[test]
fn rust_attributes_free_the_whole_tree_recursively() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    let mut attrs = RustAttrs::default();
    host_build_attrs(&mut attrs, &mut guest.machine).unwrap();
    assert!(guest.live_allocations() > allocs_before);

    // One call, and the entries, the keys, the strings, the nested group and
    // the list are all gone
    attrs.free(&mut guest.machine).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
    assert!(attrs.is_empty());

    // A freed map is a valid empty one, so freeing twice isn't a double free
    attrs.free(&mut guest.machine).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// A Rust enum has no valueless state: free() leaves the first variant in
/// its default state, which the guest can still drop.
#[test]
fn a_freed_rust_enum_holds_its_first_variant() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    let value = RustBox::new(&mut guest.machine, &RustValue::default()).unwrap();
    let m = &mut guest.machine;
    assert!(value.is::<()>(m).unwrap());
    assert_eq!(value.discriminant(m).unwrap(), Some(0));

    value
        .insert::<RustString, _>(m, "a heap allocated string")
        .unwrap();
    assert!(value.is::<RustString>(m).unwrap());
    // The boxed enum, and the string's bytes
    assert_eq!(guest.live_allocations(), allocs_before + 2);

    // Freeing the value in place (not the box: GuestPtr::free, through the
    // box's GuestPtr)
    let m = &mut guest.machine;
    (*value).free(m).unwrap();
    // Back to Nil, which owns nothing, so a second free changes nothing
    assert!(value.is::<()>(m).unwrap());
    assert_eq!(guest.live_allocations(), allocs_before + 1);
    let m = &mut guest.machine;
    (*value).free(m).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before + 1);

    // Every variant can be selected explicitly
    let m = &mut guest.machine;
    value.insert::<i64, _>(m, &-5).unwrap();
    let v = value.get(m).unwrap();
    assert_eq!(v.get::<i64>(), Some(&-5));
    assert!(v.get::<f64>().is_none());
    assert_eq!(v.value(), Some(&Value::Int(-5)));
    assert_eq!(v.discriminant(), Some(RustValue::discriminant_of::<i64>()));
    // ... or replaced as a whole value
    value.assign(m, &Value::Bool(true)).unwrap();
    assert_eq!(value.get(m).unwrap().value(), Some(&Value::Bool(true)));

    // The same on a host-side value
    let mut host = RustValue::new(Value::Str(RustString::from_str(m, "on the host").unwrap()));
    assert!(host.is::<RustString>());
    host.free_value(m).unwrap();
    assert!(host.is::<()>());

    // Freeing the box frees the enum and the box
    value.free(&mut guest.machine).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// A guest can hand over a map whose length exceeds its capacity, or whose
/// keys are out of order: neither may corrupt the host, nor go unnoticed.
#[test]
fn rust_attributes_reject_a_malformed_guest_map() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    let mut attrs = RustAttrs::default();
    attrs.insert(m, "alpha", &Value::Int(1)).unwrap();
    attrs.insert(m, "beta", &Value::Int(2)).unwrap();
    attrs.validate(m, 4096).unwrap();
    assert!(attrs.is_sorted(m).unwrap());

    // A length beyond the capacity is caught before anything is read
    let real_len = attrs.entries.len;
    attrs.entries.len = attrs.entries.cap + 1;
    assert!(attrs.validate(m, 4096).is_err());
    attrs.entries.len = real_len;

    // So is a map with more entries than the caller allows
    assert!(attrs.validate(m, 1).is_err());
    attrs.validate(m, 2).unwrap();

    // Keys out of order can't corrupt anything (a binary search that misses
    // only fails to find), but is_sorted() reports it
    let first = attrs.entries.get(0).unwrap();
    let second = attrs.entries.get(1).unwrap();
    let (a, b) = (first.get(m).unwrap(), second.get(m).unwrap());
    first.set(m, &b).unwrap();
    second.set(m, &a).unwrap();
    assert!(!attrs.is_sorted(m).unwrap());

    attrs.free(m).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

libriscv::guest_rust_enum! {
    #[derive(Debug, Clone, PartialEq)]
    pub enum FlatValue => HostFlatValue {
        Nil(()),
        Int(i64),
        Float(f64),
        Str(RustString),
    }
}
type FlatRustValue = GuestRustEnum<FlatValue, u64>;
type FlatAttrs = GuestRustAttributes<FlatRustValue>;
// The shape of the recursive value, minus the two boxed variants
const _: () = assert!(FlatRustValue::SIZE == 8 + RustString::SIZE, "tag + String");

/// A map whose values are all plain values or strings can be copied to the
/// host in one call. A tree with nested groups can't (its enum has no host
/// copy: `guest_rust_enum!` without `=> Host`), so iter() is the only
/// way in, which keeps a nested read from returning a pointer as a number.
#[test]
fn a_flat_rust_attribute_map_can_be_copied_to_the_host() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    let attrs = RustBox::new(m, &FlatAttrs::default()).unwrap();
    attrs.insert(m, "count", &FlatValue::Int(7)).unwrap();
    attrs.insert(m, "ratio", &FlatValue::Float(0.25)).unwrap();
    let label = RustString::from_str(m, "a label").unwrap();
    attrs.insert(m, "label", &FlatValue::Str(label)).unwrap();
    attrs.insert(m, "nothing", &FlatValue::Nil(())).unwrap();

    // A single variant on its own
    let map = attrs.get(m).unwrap();
    let value = map.get(m, "count").unwrap().unwrap().get(m).unwrap();
    assert_eq!(value.to_host(m).unwrap(), HostFlatValue::Int(7));

    // And the whole map at once, with the guest strings copied out
    let map = attrs.get(m).unwrap().to_map(m).unwrap();
    assert_eq!(map.len(), 4);
    assert_eq!(map["count"], HostFlatValue::Int(7));
    assert_eq!(map["ratio"], HostFlatValue::Float(0.25));
    assert_eq!(map["label"], HostFlatValue::Str("a label".to_owned()));
    assert_eq!(map["nothing"], HostFlatValue::Nil(()));

    attrs.free(m).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

// ===========================================================================
// Block memory operations
//
// The guest is linked with --wrap=memcpy and friends, so a Rust program that
// never mentions memcpy still ends up in the host for every bulk copy: Vec
// growth, slice copies and String appends. Each test checks that the bytes
// are right *and* that the operation reached the host: a guest that kept its
// own memcpy would pass every other check while emulating the whole loop.
// ===========================================================================

/// The byte pattern the guest fills its buffers with.
fn memop_pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

#[test]
fn booting_a_rust_guest_goes_through_the_host_memory_functions() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    // glibc's startup and the Rust runtime copy and zero a lot before
    // main(), and --wrap catches all of it
    let counts = state(&mut guest.machine).memop_counts;
    assert!(counts[MEMCPY] > 0, "boot: {counts:?}");
    assert!(counts[MEMSET] > 0, "boot: {counts:?}");
}

#[test]
fn rust_memset_runs_on_the_host() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    for len in [1usize, 15, 64, 4096, 256 * 1024] {
        let ops = MemoryOpCounter::new(&mut guest);
        let checksum = guest.call("rust_memset_fill", (len, 0xA5u8));
        assert_eq!(checksum, fnv1a(&vec![0xA5; len]), "memset of {len} bytes");
        assert!(ops.get(&mut guest, MEMSET) > 0, "memset of {len} bytes");
    }
    assert_eq!(guest.live_allocations(), allocs_before);
}

#[test]
fn rust_memcpy_runs_on_the_host() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    for len in [1usize, 15, 64, 4096, 256 * 1024] {
        let ops = MemoryOpCounter::new(&mut guest);
        let checksum = guest.call("rust_memcpy_roundtrip", len);
        assert_eq!(
            checksum,
            fnv1a(&memop_pattern(len)),
            "memcpy of {len} bytes"
        );
        assert!(ops.get(&mut guest, MEMCPY) > 0, "memcpy of {len} bytes");
    }
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// The expected result of the guest's rust_memmove_overlap.
fn memmove_expected(len: usize, shift: usize, downwards: bool) -> Vec<u8> {
    let mut expected = memop_pattern(len);
    if downwards {
        expected.copy_within(shift.., 0);
    } else {
        expected.copy_within(..len - shift, shift);
    }
    expected
}

/// The host has one path for a destination below the source and one for a
/// destination above it (which copies backwards).
#[test]
fn rust_memmove_runs_on_the_host_both_directions() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    for len in [64usize, 4096, 256 * 1024] {
        for shift in [1usize, 7, 8, 32] {
            for downwards in [true, false] {
                let ops = MemoryOpCounter::new(&mut guest);
                let checksum = guest.call("rust_memmove_overlap", (len, shift, downwards));
                let what = format!("memmove of {len} bytes shifted {shift} (down: {downwards})");
                assert_eq!(
                    checksum,
                    fnv1a(&memmove_expected(len, shift, downwards)),
                    "{what}"
                );
                assert!(ops.get(&mut guest, MEMMOVE) > 0, "{what}");
            }
        }
    }
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// The sign of the host's memcmp is what Ord turns into Less, Equal or
/// Greater.
#[test]
fn rust_memcmp_runs_on_the_host() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    const LEN: usize = 4096;
    // An index past the end leaves the buffers equal; otherwise the byte at
    // that index decides the order
    let cases: [(usize, u8, i64); 6] = [
        (LEN, 0x00, 0),      // untouched: equal
        (0, 0xFF, -1),       // right is larger at the very first byte
        (0, 0x00, 0),        // pattern[0] is already 0
        (LEN - 1, 0xFF, -1), // and at the very last one
        (LEN - 1, 0x00, 1),  // pattern[LEN-1] isn't 0, so left wins
        (LEN / 2, 0xFF, -1),
    ];
    for (index, value, expected) in cases {
        let ops = MemoryOpCounter::new(&mut guest);
        let order = guest.call("rust_memcmp_order", (LEN, index, value)) as i64;
        assert_eq!(order, expected, "memcmp with byte {index} set to {value}");
        assert!(
            ops.get(&mut guest, MEMCMP) > 0,
            "memcmp with byte {index} set to {value}"
        );
    }
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// `a == b` doesn't compile to memcmp: LLVM emits the equality-only entry
/// point, __memcmpeq, which has to be wrapped separately.
#[test]
fn rust_slice_equality_runs_on_the_host() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    const LEN: usize = 4096;
    for len in [1usize, 64, 4096] {
        let ops = MemoryOpCounter::new(&mut guest);
        // index == len leaves the two buffers identical
        assert_eq!(
            guest.call("rust_memcmp_equality", (len, len, 0u8)),
            1,
            "{len} bytes"
        );
        assert!(
            ops.get(&mut guest, MEMCMP) > 0,
            "equal slices of {len} bytes"
        );
    }
    // A difference at either end has to be found
    for index in [0, LEN / 2, LEN - 1] {
        let ops = MemoryOpCounter::new(&mut guest);
        assert_eq!(guest.call("rust_memcmp_equality", (LEN, index, 0xFFu8)), 0);
        assert!(
            ops.get(&mut guest, MEMCMP) > 0,
            "slices differing at byte {index}"
        );
    }
    assert_eq!(guest.live_allocations(), allocs_before);
}

#[test]
fn rust_block_memory_operations_handle_the_edges() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    // A zero length is accepted and does nothing (Rust may not even call)
    assert_eq!(guest.call("rust_memset_fill", (0usize, 0xA5u8)), fnv1a(b""));
    assert_eq!(guest.call("rust_memcpy_roundtrip", 0usize), fnv1a(b""));

    // A whole-buffer overlap: source and destination are the same address
    const LEN: usize = 1024;
    for downwards in [true, false] {
        assert_eq!(
            guest.call("rust_memmove_overlap", (LEN, 0usize, downwards)),
            fnv1a(&memop_pattern(LEN)),
            "memmove onto itself (down: {downwards})"
        );
    }
    // A one-byte overlap, where the copy is almost the whole buffer
    assert_eq!(
        guest.call("rust_memmove_overlap", (LEN, 1usize, true)),
        fnv1a(&memmove_expected(LEN, 1, true))
    );
    // Two empty slices compare equal
    assert_eq!(guest.call("rust_memcmp_order", (0usize, 0usize, 0u8)), 0);

    assert_eq!(guest.live_allocations(), allocs_before);
}

/// A Rust program that never mentions memcpy spends most of its bulk memory
/// work inside one: the String is built in the shared heap through the
/// host's memcpy, and read back by the host without a copy.
#[test]
fn a_rust_string_grown_through_the_host_memcpy() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();

    const REPEATS: usize = 64;
    const PIECE: &str = "The guest appends this through the host's memcpy. ";
    let ops = MemoryOpCounter::new(&mut guest);
    let text = RustBox::new(&mut guest.machine, &RustString::default()).unwrap();
    guest
        .machine
        .vmcall("rust_memops_build_string", (&text, REPEATS))
        .unwrap();

    let m = &guest.machine;
    assert_eq!(
        text.get(m).unwrap().to_str(m).unwrap(),
        PIECE.repeat(REPEATS)
    );
    let memcpys = ops.get(&mut guest, MEMCPY);
    assert!(memcpys >= REPEATS as u64, "String append: memcpy={memcpys}");

    text.free(&mut guest.machine).unwrap();
    assert_eq!(guest.live_allocations(), allocs_before);
}

/// A guest controls every word of a collection it lends the host. Corrupt lengths, capacities, pointers and tags are errors, never a
/// host panic, and nothing is freed on the way.
#[test]
fn corrupt_guest_collections_are_errors() {
    let Some(mut guest) = RustGuest::boot() else {
        return;
    };
    guest.warmup();
    let allocs_before = guest.live_allocations();
    let m = &mut guest.machine;

    // A length beyond the capacity, and one that overflows when scaled
    for len in [5, u64::MAX, u64::MAX / 4 + 1] {
        let vec = RustVec::<u32>::from_raw_parts(RustVec::<u32>::DANGLING, len, 4);
        assert!(vec.to_vec(m).is_err(), "len {len:#x}");
        assert!(vec.view(m).is_err(), "len {len:#x}");
        let string = RustString {
            cap: 4,
            ptr: RustString::DANGLING,
            len,
        };
        assert!(string.as_bytes(m).is_err(), "len {len:#x}");
        assert!(string.to_str(m).is_err(), "len {len:#x}");
    }
    let huge = RustVec::<u64>::from_raw_parts(0x1000, u64::MAX, u64::MAX);
    assert!(huge.iter().is_err());
    assert!(huge.view(m).is_err());
    assert!(
        GuestRustSlice::<u64>::from_raw_parts(0x1000, u64::MAX)
            .to_vec(m)
            .is_err()
    );
    assert!(
        GuestRustStr::from_raw_parts(0, u64::MAX)
            .as_bytes(m)
            .is_err()
    );

    // Freeing or growing one fails instead of walking a bogus array
    let mut bogus = RustVec::<RustString>::from_raw_parts(0x10, u64::MAX, u64::MAX);
    assert!(bogus.clear(m).is_err());
    let mut bogus = RustVec::<u32>::from_raw_parts(0x10, 1, 1);
    assert!(bogus.reserve(m, usize::MAX).is_err());
    // Growing from an unreadable pointer releases the new allocation (the
    // live-allocation count at the end checks it)
    assert!(bogus.reserve(m, 4).is_err());
    assert!(bogus.reserve_exact(m, 4).is_err());
    let mut bogus = RustString {
        cap: 1,
        ptr: 0x10,
        len: 1,
    };
    assert!(bogus.push_str(m, "more").is_err());
    assert!(bogus.reserve_exact(m, 4).is_err());

    // Zero-sized elements own nothing and are never allocated: whatever
    // length a guest claims (Rust stores a capacity of 0 for them), dropping
    // them does no work and frees nothing, and walking them is bounded
    let zst_len = 1 << 63;
    let mut units = RustVec::<()>::from_raw_parts(RustVec::<()>::DANGLING, zst_len, 0);
    assert!(units.iter().is_err());
    assert!(units.view(m).is_err());
    units.truncate(m, 1).unwrap();
    assert_eq!(units.len(), 1);
    units.free(m).unwrap();
    let mut units = GuestRustBoxedSlice::<()>::from_raw_parts(1, zst_len);
    units.free(m).unwrap();
    let mut units = RustVec::<()>::from_slice(m, &[(), (), ()]).unwrap();
    units.push(m, &()).unwrap();
    assert_eq!((units.len(), units.capacity()), (4, 0));
    assert_eq!(units.as_ptr(), RustVec::<()>::DANGLING);
    units.free(m).unwrap();
    let units = GuestRustBoxedSlice::<()>::from_slice(m, &[(), ()]).unwrap();
    assert_eq!((units.len(), units.as_ptr()), (2, 1));
    let unit = GuestRustBox::<()>::new(m, &()).unwrap();
    assert_eq!(unit.into_raw(), 1);
    unit.free(m).unwrap();

    // A null box is Option<Box<T>>::None, not a box: nothing to read
    let null = GuestRustBox::<i64>::default();
    assert!(null.is_null());
    assert!(null.get(m).is_err());
    assert!(null.to_host(m).is_err());

    // Invalid UTF-8 is an error from to_str(), and replaced by
    // to_string_lossy()
    let bytes = GuestBytes::new(m, b"ok\xFF");
    let invalid = GuestRustStr::from_raw_parts(bytes.addr, bytes.len);
    assert!(invalid.to_str(m).is_err());
    assert_eq!(invalid.to_string_lossy(m).unwrap(), "ok\u{FFFD}");
    bytes.free(m);

    // An out-of-range tag is no variant; freeing it leaves the first one
    let value = RustBox::new(m, &RustValue::default()).unwrap();
    m.memory.write_value(value.address(), &99u64).unwrap();
    assert_eq!(value.discriminant(m).unwrap(), None);
    assert_eq!(value.get(m).unwrap().value(), None);
    assert!(value.variant::<i64>(m).unwrap().is_none());
    (*value).free(m).unwrap();
    assert!(value.is::<()>(m).unwrap());
    value.free(m).unwrap();

    assert_eq!(guest.live_allocations(), allocs_before);
}
