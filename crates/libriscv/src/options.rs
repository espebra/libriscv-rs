//! Port of C++ `MachineOptions<W>` (`common.hpp`).
//!
//! Options cover the interpreter, Cranelift JIT, and AOT translation.
//! C++ binary-translation, libtcc, and asmjit options are replaced by the
//! corresponding Rust backends; see `PORTING.md`.

/// Options passed when constructing a machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineOptions {
    /// Maximum guest memory, rounded down to the page size.
    pub memory_max: u64,
    /// Maximum host bytes held by execute segments: code copies and decoder caches.
    /// Storage is approximately 9 bytes per code byte, or 5 without the C extension.
    /// Creating a segment beyond this limit returns `OutOfMemory`.
    ///
    /// `None` defaults to the page-rounded `memory_max`, with a minimum of
    /// [`EXECUTE_SEGMENT_MEMORY_MIN`](crate::memory::EXECUTE_SEGMENT_MEMORY_MIN).
    /// Shared segments count in full for each machine. Native code is excluded;
    /// background compilation temporarily retains an additional segment copy.
    /// New segments evict earlier segments they cover. C++ has no equivalent limit
    /// and retains covered segments.
    pub execute_segment_memory_max: Option<u64>,
    /// Virtual memory reserved for the main stack at construction.
    pub stack_size: u32,
    /// Load the binary as a RISC-V ELF at construction.
    pub load_program: bool,
    /// Apply page protections from the ELF program headers.
    pub protect_segments: bool,
    /// Allow read-write-execute segments.
    pub allow_write_exec_segment: bool,
    /// Enforce execute-only segments (X without R).
    pub enforce_exec_only: bool,
    /// Ignore the `.text` section and load all executable segments as-is.
    pub ignore_text_section: bool,
    /// Print verbose loader information.
    pub verbose_loader: bool,
    /// Skip copy-on-write page sharing when forking.
    pub minimal_fork: bool,
    /// Use a linear memory arena for main memory.
    pub use_memory_arena: bool,
    /// Preserve vector registers when copying register state.
    pub preserve_vector_registers: bool,
    /// Compile execute segments to native code with Cranelift.
    /// Requires the `jit` feature; otherwise `Machine::new` returns
    /// `FEATURE_DISABLED`. Unsupported hosts use the interpreter.
    pub jit: bool,
    /// Compile execute segments on background threads when `jit` is enabled
    /// (C++ `asmjit_background_callback`). The interpreter runs each segment
    /// until compiled code is available, preserving results and instruction counts.
    /// While compilation is pending, `simulate` checks for completed code every
    /// `jit_background_slice` instructions. Compilation may exceed execution time
    /// for short programs.
    pub jit_background: bool,
    /// Bytes of executable code that may be decoded at runtime without an
    /// instruction charge. Includes newly mapped code and stale-segment rebuilds
    /// after FENCE.I. Each decode copies, hashes, and decodes the surrounding
    /// executable area. Beyond this allowance, decoding costs `len / 4`
    /// instructions, bounding repeated decodes by the instruction limit.
    ///
    /// C++ does not charge for runtime decoding; instruction counts match only
    /// within this allowance. Load-time segments and `init_execute_area` are excluded.
    pub free_decode_bytes: u64,
    /// Instructions between checks for completed background compilations,
    /// clamped to at least 1. Smaller slices reduce the delay before native execution.
    pub jit_background_slice: u64,
    /// Mask native guest addresses with `arena_size - 1` instead of checking
    /// bounds (C++ `translate_automatic_nbit_address_space`). Applies to JIT code
    /// and translations generated through `aot_output`.
    ///
    /// Masked accesses can read or write the entire arena, including read-only
    /// data and page zero. Addresses outside the arena wrap instead of using
    /// the page table. Interpreter accesses retain their usual checks.
    ///
    /// Requires a power-of-two arena. `memory_max` is rounded up; the `rvlinux`
    /// default of 4000 MiB becomes 4 GiB. Rounding by more than 256 MiB makes
    /// `Machine::new` return `OutOfMemory`. Translations require the same setting.
    pub jit_automatic_nbit_address_space: bool,
    /// Disable instruction counting and limit checks in native code
    /// (C++ `translate_ignore_instruction_limit`). Applies to JIT code and
    /// translations generated through `aot_output`.
    ///
    /// Native loops run until they exit native code. System calls still allow
    /// `stop()` to take effect. The interpreter counts its own blocks and enforces
    /// the limit, so the counter reports only partial progress. `rvlinux` enables
    /// this unless `--accurate` is set. Translations require the same setting.
    pub jit_ignore_instruction_limit: bool,
    /// Guard JIT bounds checks and entry-table lookups against Spectre v1
    /// (default: enabled). Conditional moves constrain addresses before access,
    /// including during speculative execution. Disabling guards is intended for
    /// trusted guests and improves checked CoreMark performance by about 25%.
    /// Masked accesses (`jit_automatic_nbit_address_space`) require no guards.
    /// Architectural results are unchanged.
    pub jit_spectre_guards: bool,
    /// Use previously compiled JIT code for matching execute segments
    /// (default: enabled), including linked objects and shared objects registered
    /// with `jit_cache::load_shared_object`.
    /// Cached code takes precedence over compilation and can run with `jit: false`.
    /// Requires the `jit` feature and preserves JIT results and instruction counts.
    pub jit_cache: bool,
    /// Record JIT-compiled segments for saving across runs; see [`JitCacheOutput`].
    /// The machine compiles its own segments instead of using cached code or
    /// segments shared by other machines. Requires `jit`.
    ///
    /// [`JitCacheOutput`]: crate::jit_cache::JitCacheOutput
    #[cfg(feature = "jit")]
    pub jit_cache_output: Option<crate::jit_cache::JitCacheOutput>,
    /// Directory for loading and saving compiled segments covered by [`CacheDir`].
    /// Requires `jit` and `jit_cache`. Saving is part of compilation; with
    /// `jit_background`, call [`jit_cache::wait_for_saves`] before exiting.
    ///
    /// [`CacheDir`]: crate::jit_cache::CacheDir
    /// [`jit_cache::wait_for_saves`]: crate::jit_cache::wait_for_saves
    #[cfg(feature = "jit")]
    pub jit_cache_dir: Option<crate::jit_cache::CacheDir>,
    /// Use registered AOT Rust translations for matching segments and memory
    /// layouts (default: enabled). Translations take precedence over the interpreter
    /// and JIT and preserve interpreter results and instruction counts.
    /// Requires the `aot` feature. Register embedded code with `aot::register`
    /// or load shared objects with `aot::load_shared_object` (`aot-dylib`).
    pub aot: bool,
    /// Translate each new execute segment to Rust and record it for export
    /// as source or a shared object; see [`AotOutput`](crate::aot::AotOutput).
    /// The machine creates its own segments instead of using registered
    /// translations or shared segments. Execution continues with the interpreter
    /// or JIT.
    #[cfg(feature = "aot")]
    pub aot_output: Option<crate::aot::AotOutput>,
    /// ELF symbol used as the VM call return address instead of the injected
    /// host code page (C++ `default_exit_function`). The function must stop the
    /// call with `.insn i SYSTEM, 0, x0, x0, 0x7ff`. An exit in the program's
    /// execute segment avoids a segment switch.
    ///
    /// If the symbol is absent, the host page is used. If present, the host page
    /// and its signal-return trampoline are not installed, matching C++.
    ///
    /// [`Machine::vmcall`]: crate::Machine::vmcall
    pub default_exit_function: Option<String>,
    /// Share decoded execute segments and JIT code among machines in the
    /// process with matching code and addresses (C++ `use_shared_execute_segments`,
    /// default: enabled). Forks always share their parent's segments.
    pub use_shared_execute_segments: bool,
    /// Share whole host pages of program text and read-only data below the first
    /// writable segment among machines with the same program and arena size
    /// (C++ `use_shared_rodata`, default: enabled).
    ///
    /// Requires `jit` and Linux on x86_64, aarch64, or riscv64; otherwise has no
    /// effect. JIT execution is not required. The image is mapped copy-on-write,
    /// so stores enabled by paging or N-bit masking remain private to each machine.
    pub use_shared_rodata: bool,
    /// Install EBREAKs in decoder caches at the specified locations without
    /// changing guest memory (C++ `ebreak_locations`). Breakpoints invoke the
    /// EBREAK system call handler. Segments with breakpoints are not shared.
    pub ebreak_locations: Vec<crate::EbreakLocation>,
    /// Replace known libc functions in the program's initial execute segment
    /// with native implementations (C++ `libc_fastpath`), charged as native memory
    /// system calls. Functions are identified by ELF symbols: memcpy, memset,
    /// memmove, memcmp, memchr, rawmemchr, strlen, strnlen, strcmp, strncmp, strcpy,
    /// stpcpy, strncpy, strchr, strchrnul, and aliases. Only decoder caches are
    /// patched; other EBREAKs still invoke the installed EBREAK handler.
    pub libc_fastpath: bool,
}

impl Default for MachineOptions {
    fn default() -> Self {
        Self {
            memory_max: 64 << 20,
            execute_segment_memory_max: None,
            stack_size: 1 << 20,
            load_program: true,
            protect_segments: true,
            allow_write_exec_segment: false,
            enforce_exec_only: false,
            ignore_text_section: false,
            verbose_loader: false,
            minimal_fork: false,
            use_memory_arena: true,
            preserve_vector_registers: true,
            jit: false,
            jit_background: false,
            free_decode_bytes: crate::FREE_DECODE_BYTES,
            jit_background_slice: crate::JIT_BACKGROUND_SLICE,
            jit_automatic_nbit_address_space: false,
            jit_ignore_instruction_limit: false,
            jit_spectre_guards: true,
            jit_cache: true,
            #[cfg(feature = "jit")]
            jit_cache_output: None,
            #[cfg(feature = "jit")]
            jit_cache_dir: None,
            aot: true,
            #[cfg(feature = "aot")]
            aot_output: None,
            default_exit_function: None,
            use_shared_execute_segments: true,
            use_shared_rodata: true,
            ebreak_locations: Vec::new(),
            libc_fastpath: false,
        }
    }
}
