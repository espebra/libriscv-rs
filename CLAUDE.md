# libriscv-rs

Rust port of C++ libriscv, a RISC-V userspace emulator for sandboxing and
scripting. Read `PORTING.md` for scope, architecture, implementation status,
and the C++ to Rust file mapping.

## Repository layout

- C++ reference source: `../libriscv` (or `$LIBRISCV_CPP`). Library code is in
  `../libriscv/lib/libriscv/`; `../libriscv/AGENTS.md` is its architecture guide.
- C++ reference emulator (the test oracle): `../libriscv/emulator/rvlinux`.
- CoreMark sources (difftest and benchmark): `../coremark` (or `$COREMARK_DIR`).
- `crates/libriscv` — the library. `crates/rvlinux` — CLI. `crates/testutil` —
  guest cross-compilation and oracle helpers (test-only). `examples/asteroids` —
  a game on the scripting API; its RISC-V guest (`guest/`) is a separate cargo
  project that the example builds.

## Commands

```sh
cargo test                                   # all tests, JIT included (tool-dependent tests skip if tools missing)
LIBRISCV_REQUIRE_TOOLCHAIN=1 cargo test      # same, but fail instead of skipping
cargo test -- --ignored                      # include not-yet-enabled difftests
cargo test -p rvlinux --test difftest rva23  # C++'s RVA23 suite; needs an RVA23 gcc ($RVCC, default ~/riscv/bin/riscv64-unknown-linux-gnu-gcc), else skips (LIBRISCV_REQUIRE_RVA23=1 fails)
cargo clippy --all-targets -- -D warnings    # must be clean (also with --no-default-features)
cargo test --workspace --no-default-features # interpreter only: no Cranelift, no unsafe in the build
cargo test --workspace --features libriscv/aot-dylib,rvlinux/aot-dylib  # + AOT (needs rustc)
rvlinux --jit-save-shared x.so prog.elf; rvlinux --jit-load x.so prog.elf   # JIT code cache (also --jit-save-object, RVLINUX_JIT_EMBED)
rvlinux --aot-save-shared x.so prog.elf; rvlinux --aot-load x.so prog.elf   # AOT: translate to safe Rust (also --aot-save x.rs, RVLINUX_AOT_EMBED)
cargo run --release -p asteroids             # example game: Rust engine + Rust RV64GCB script (examples/asteroids)
rvlinux dynamic.elf                          # JIT by default (-n interprets); caches ld.so + system libraries in ~/.cache/rvlinux/jit by itself (--jit-cache-dir, --no-jit-cache)
```

Expected durations for a warm build on 32 cores: approximately 48 s with
JIT, 42 s without an RVA23 toolchain, and 9 s with `--no-default-features`.
Test binaries run sequentially: `difftest.rs` takes about 18 s, including an
initial JIT compilation of static libstdc++; `libriscv` tests take about
21 s, including 7 s for `jit_runs_on_shared_rodata`.

Cranelift is optimized through `[profile.dev.package."*"]`; workspace crates
use opt-level 0. Split independent test runs into separate tests so the
harness can execute them concurrently; see `difftests!`. Never use `git stash`
to benchmark HEAD.

```sh
cargo fmt --all
scripts/difftest.sh prog.elf [args...]       # diff one ELF against the C++ emulator
scripts/build-cpp-reference.sh               # C++ threaded/tailcall/switch builds -> target/cpp-ref/
cargo bench -p rvlinux --bench coremark   # CoreMark: Rust interpreter/JIT vs C++
fuzz/do_fuzz.sh vmfuzzer -- -max_total_time=300   # libFuzzer (vmfuzzer/elffuzzer/syshelperfuzzer); needs cargo-fuzz
```

Toolchain: pinned nightly (`rust-toolchain.toml`) for `become`. Do not bump it
as part of unrelated work; see "Toolchain" in `PORTING.md`.

Guests compile with `riscv64-linux-gnu-gcc-14` (`$RCC`); outputs are cached in
`target/riscv-guests/`. Rust guests (`testutil::build_rust`, `tests/native_rust.rs`)
need `rustup target add riscv64gc-unknown-linux-gnu` for the pinned toolchain.

## Rules

- **Enforce safe Rust.** Do not bypass `forbid(unsafe_code)` with `allow`
  or helper crates. Only `crates/libriscv-jit-rt` may contain unsafe code,
  for executable memory, AOT shared-object loading, and guest arena mappings.
  Keep that boundary minimal and documented. JIT generation and generated
  AOT bodies must remain safe Rust.
- **Guest input must not panic the host.** Return `MachineError`. Use checked
  access, checked or wrapping arithmetic, and fallible conversions for
  guest-controlled values. Resolve clippy warnings for `indexing_slicing`,
  `unwrap_used`, `expect_used`, and `panic`. A locally provable invariant may
  justify `#[allow]`; document it in a comment.
- **Preserve C++ interpreter behavior:** results, exception kinds, system
  calls, and retired-instruction counts. Resolve ambiguities using the C++
  source and reference emulator. Record upstream bugs in the PR and preserve
  compatibility unless instructed otherwise.
- Port corresponding `../libriscv/tests/unit/*.cpp` cases with each implementation.
- Use `become` for every dispatch tail call. Handler chains are unbounded;
  ordinary calls can overflow the stack. Use the shared signature described
  in `PORTING.md` and the structure in
  `../libriscv/lib/libriscv/tailcall_dispatch.cpp`.
- Start new modules with `//! Port of <C++ files>.`.
- Preserve byte-compatible `rvlinux` summaries for `testutil::parse_run_output`:
  `>>> Program exited, exit code = N (0xN)` and `Instructions executed: N ...`.
- Update `PORTING.md` status entries with implementation changes.
- Ask before adding runtime dependencies to `libriscv`. Dev dependencies
  require no approval.
- Follow the exclusions in `PORTING.md`: C translation, libtcc, asmjit,
  assembly dispatch, Windows shims, RV32, RV128, and the C API. Use Cranelift
  for JIT and safe Rust for AOT. Keep the port RV64 only, without width
  generics or `W == 4` paths.
- Where C++ tail-call and threaded dispatch differ, follow the reference's
  threaded implementation (`cpu_dispatch.cpp`); see "Dispatch" in `PORTING.md`.

## Integration tests

Integration test files start with
`#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]`
because clippy's in-test exemptions do not cover helper functions. Gate tests
needing external tools with `if !testutil::need_compiler() { return; }` /
`testutil::need_reference()`.
