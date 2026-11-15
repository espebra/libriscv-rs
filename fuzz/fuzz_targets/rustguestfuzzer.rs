//! The Rust guest collections of the scripting API, on guest-controlled bytes (`libriscv_fuzz::rust_guest::fuzz_rust_collections`).
#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    libriscv_fuzz::rust_guest::fuzz_rust_collections(data);
});
