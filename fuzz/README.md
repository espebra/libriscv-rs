# libriscv-rs fuzzing

[cargo-fuzz] harnesses using libFuzzer. The first four targets port
libriscv's `fuzz/fuzz.cpp`; the remaining targets cover additional library
components. Harnesses are defined in `src/`, with wrappers in `fuzz_targets/`.

| Target | C++ mode | Coverage |
| ------ | -------- | -------- |
| `vmfuzzer` | `FUZZ_VM` | Decoder and interpreter; input code at `0x2000`, 5,000 instructions |
| `elffuzzer` | `FUZZ_ELF` | ELF loader, execute segments, and Linux system calls; 5,000 instructions, no filesystem or sockets |
| `nativefuzzer` | `FUZZ_NAT` | Native heap and memory system calls; input selects a system call and code at `0x2000`, run for 10,000 instructions followed by another invocation of the selected call |
| `syshelperfuzzer` | `FUZZ_SYSH` | `memstring`, `gather_buffers`, `memview`, `memcmp`, `memcpy`, and `gather_writable_buffers`, invoked through system call 1 |
| `linuxfuzzer` | | Linux system calls, POSIX threads, signal delivery after faults, and a read-only filesystem sandbox for `/dev/null`, `/dev/zero`, `/dev/urandom`, and `/etc/hosts`; minimal ELF with argv/envp/auxv followed by an additional system call, no sockets or proxy mode |
| `libcfuzzer` | | `libc_fastpath`; input supplies and calls functions patched by ELF symbol |
| `forkfuzzer` | | Forks and copy-on-write; parent and fork must produce identical final states, and neither may affect a fork of the fork |
| `guestfuzzer` | | Host operations on guest C++ `std::string`, `vector`, `unordered_map`, and `variant` objects supplied by the input |
| `rustguestfuzzer` | | Host operations and recursive freeing of Rust guest strings, vectors, boxes, enums, attribute maps, and recursive attribute trees; no compiled Rust guest required |
| `arenafuzzer` | | Arena writes and discards compared with a byte array; `--features jit` uses the mapped arena with page reclamation |
| `vmcallfuzzer` | | VM arguments, nested preemption, dyncalls, and system call argument readers, including Rust strings, slices, `ByPointer`, and pointers to Rust collections |
| `jitdifffuzzer` | | JIT and interpreter registers, PC, counter, errors, and memory; requires `--features jit` and x86-64. Skips NaN-producing FP arithmetic because NaN payloads depend on the host |

ELF-based targets place code at `0x10000` and a zero-filled writable segment
at `0x200000`, with selected argument registers pointing into that segment.
`do_fuzz.sh` loads `dicts/<target>.dict`, which supplies instruction words,
system call numbers, sandbox paths, and guest pointers. Harness tests verify
that setup succeeds and each target reaches its intended operations.

Native thread system calls are not ported and are excluded from `nativefuzzer`.
All targets are RV64 only. Any host panic is a bug; machine errors are expected
and ignored, matching exception handling in the C++ harness.

## Setup

```sh
cargo install cargo-fuzz
```

The fuzz crate uses the repository's pinned nightly and a separate workspace
to exclude libFuzzer from the main test build.

## Running

```sh
./do_fuzz.sh vmfuzzer                            # until a crash or Ctrl-C
./do_fuzz.sh vmfuzzer -- -max_total_time=300     # 5 minutes
./do_fuzz.sh elffuzzer -- -fork=16               # 16 worker processes
cargo fuzz run -s none -a syshelperfuzzer        # without the script
```

`do_fuzz.sh` runs `cargo fuzz run -s none -a` (override with `FUZZ_FLAGS`)
and forwards arguments after `--` to libFuzzer:

- `-s none` disables sanitizers. Bounds checks detect invalid accesses in
  safe Rust. AddressSanitizer cannot instrument the dispatcher's
  `preserve_none` functions: LLVM reports "Stack realignment in presence
  of dynamic allocas is not supported with this calling convention".
  This also prevents sanitizer coverage of `libriscv-jit-rt`.
- `-a` enables debug assertions and overflow checks. Arithmetic overflow
  on guest-controlled values is a bug, including when release builds wrap.

The corpus is kept in `corpus/<target>/`, crashing inputs in
`artifacts/<target>/` (both git-ignored). Replay one with:

```sh
cargo fuzz run -s none -a vmfuzzer artifacts/vmfuzzer/crash-<hash>
```

Add a regression test beside the affected code.

Use `-fork=N` for parallel workers. Fuzzing stops at the first crash.

Test the 4-byte decoder layout without the C extension:

```sh
FUZZ_FLAGS="-s none -a --no-default-features --target-dir target/no-c" ./do_fuzz.sh vmfuzzer
```

### Long runs

`overnight.sh` runs all targets concurrently for a specified duration
(default: 8 hours), distributing cores among them. Crashes, OOMs above 2 GB,
and timeouts above 30 s are saved to `artifacts/<target>/` without stopping
the run. Logs use `logs/`; corpora persist in `corpus/`. A final table lists
findings per target.

```sh
./overnight.sh                          # all targets, 8 hours, all cores
JOBS=16 ./overnight.sh 3600 guestfuzzer # one target, 16 workers, an hour
```

`jitdifffuzzer` is built in `target/jit`: fork mode re-runs the target
binary for each job, so rebuilding the shared `target/` with the JIT would
switch the other targets to it. Set `NO_JIT=1` to skip it where the JIT
does not run (anything but x86-64).

### Coverage

```sh
rustup component add llvm-tools-preview   # once, for the pinned nightly
cargo fuzz coverage -s none -a linuxfuzzer corpus/linuxfuzzer
```

This writes `coverage/linuxfuzzer/coverage.profdata`. Generate a report with
the toolchain's `llvm-cov report` against
`target/<triple>/coverage/<triple>/release/linuxfuzzer`.

### Symbolizer hangs

libFuzzer runs `llvm-symbolizer` for `NEW_FUNC` lines and stack traces and
waits for it. On distros with a default debuginfod URL (Ubuntu sets
`DEBUGINFOD_URLS` from `/etc/profile.d`) its debuginfod client can hang
forever. `do_fuzz.sh` clears `DEBUGINFOD_URLS`; do the same when running
`cargo fuzz` directly.

## Seeding the ELF loader fuzzer

An empty `elffuzzer` corpus primarily exercises ELF header rejection. `do_fuzz.sh elffuzzer` seeds it with the 64-bit RISC-V
binaries in `$LIBRISCV_CPP/tests/unit/elf` (default `../../libriscv`),
copied into `seeds/elffuzzer/` and passed after the writable
`corpus/elffuzzer/`, so libFuzzer never writes into the seeds. Do not set
`-max_len`: libFuzzer skips seed files larger than it, which drops most of
these binaries. Large seeds increase coverage at the cost of throughput;
see the C++ `fuzz/README.md` for measurements.

## Fuzzing in CI

`.github/workflows/fuzzing.yml` lints and builds the fuzz crate, then runs
each target (and `vmfuzzer` without the C extension) for 30 seconds on
push and pull request (from an empty corpus,
`elffuzzer` seeded as above), failing the job on a crash and uploading the
reproducer as an artifact. The workflow's *Run workflow* button allows longer
runs. Use extended local runs for broader coverage.

[cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz
