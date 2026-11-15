//! The memory arena's contents through writes and discards, against a plain byte array (`libriscv_fuzz::arena::fuzz_arena_discard`).
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    libriscv_fuzz::arena::fuzz_arena_discard(data);
});
