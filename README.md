# libriscv-rs

A Rust port of [libriscv](https://github.com/libriscv/libriscv), a sandboxed
RISC-V userspace emulator for embedding and scripting. The architecture follows
the C++ implementation, with Cranelift replacing its JIT backend. The library
uses `#![forbid(unsafe_code)]`. Executable memory and native library loading
are isolated in `libriscv-jit-rt`.

[![CI](https://github.com/libriscv/libriscv-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/libriscv/libriscv-rs/actions/workflows/ci.yml) [![Fuzzing](https://github.com/libriscv/libriscv-rs/actions/workflows/fuzzing.yml/badge.svg)](https://github.com/libriscv/libriscv-rs/actions/workflows/fuzzing.yml)

Supports full [RVA23U64](https://docs.riscv.org/reference/rva23/v1.0/rva23-profiles.html#3-1-1-rva23u64-profile). See [PORTING.md](PORTING.md) for extension coverage and
implementation status.

## Benchmarks

| ELF | Rust interpreter | C++ threaded | C++ tailcall | Rust JIT | C++ asmjit |
|---|---:|---:|---:|---:|---:|
| rv64imafdc | 2425 | 2440 | 2197 | 19076 | 19326 |
| rv64imafdc_zba_zbb_zbs | 2777 | 2442 | 2290 | 19778 | 19640 |
| coremark-rv64g | 2492 | 2680 | 2380 | 18412 | 18836 |
| coremark-rv64gc | 2416 | 2330 | 2255 | 18700 | 19479 |
| coremark-rv64g_b | 2744 | 2716 | 2618 | 20839 | 20095 |
| coremark.rv64gb.gcc14.2.elf | 2809 | 2863 | 2596 | 17893 | 19131 |
| coremark-rv64g_zba_zbb | 2655 | 2779 | 2496 | 19559 | 19445 |

CoreMark iterations per second with instruction counting and bounds-checked
memory. JIT execution reaches approximately 2/3 of native performance. See
[PORTING.md](PORTING.md#coremark-10) for measurement conditions.

## Building and testing

Builds on stable Rust (the version pinned in `rust-toolchain.toml`),
including the interpreter and the JIT. The opt-in `nightly-dispatch` feature
is the fastest interpreter: it uses guaranteed tail calls (`become`) and
needs the pinned nightly, as does AOT translation:

```sh
cargo +nightly-2026-09-28 build --features libriscv/nightly-dispatch,rvlinux/nightly-dispatch
```

```sh
cargo build
cargo test
```

Most tests need a RISC-V cross compiler (`riscv64-linux-gnu-gcc-14`, override
with `RCC`) and the C++ reference emulator (`../libriscv/emulator/rvlinux`,
override with `LIBRISCV_CPP` or `LIBRISCV_CPP_EMULATOR`).

## JIT code cache

The default `jit` feature supports saving Cranelift output for reuse across runs.

```sh
rvlinux --jit-save-shared prog.so prog.elf   # run and save compiled code
rvlinux --jit-load prog.so prog.elf          # reuse compiled code
rvlinux -Q --jit-save-object prog.o prog.elf # save an object for linking into rvlinux
RVLINUX_JIT_EMBED=prog.o cargo build -p rvlinux --features jit
```

Saved code is only used with the same build, host CPU and memory layout it
was compiled for. Shared objects are linked with the system `cc`. See `libriscv::jit_cache` and `MachineOptions::jit_cache_output`.

## Ahead-of-time translation to Rust

The `aot` feature translates execute segments to safe Rust for compilation
into the host program or a shared object:

```sh
rvlinux -Q --aot-save prog.rs prog.elf          # translate to Rust source
RVLINUX_AOT_EMBED=prog.rs cargo +nightly-2026-09-28 build -p rvlinux --no-default-features --features aot   # embed the translation
rvlinux -Q --aot-save-shared prog.so prog.elf   # compile a shared object with rustc
rvlinux --aot-load prog.so prog.elf             # requires aot-dylib
```

Translations preserve interpreter results and instruction counts and require
matching segments and memory layouts. AOT requires the pinned nightly
(`cargo +nightly-2026-09-28`, also for building `rvlinux` with `aot`) for
`become`. See `libriscv::aot` and `MachineOptions::aot_output`.

## License

BSD-3-Clause, same as libriscv.
