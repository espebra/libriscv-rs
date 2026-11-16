# Asteroid Dodge

A terminal game with a Rust host engine and Rust game logic running in a
sandboxed RV64GCB Linux machine with static glibc. Adapted from
`libloong/examples/gamedev`, it demonstrates dyncalls, typed system call
arguments, userdata, VM calls, and the native heap.

```sh
cargo run --release -p asteroids          # build and run with the JIT
cargo run --release -p asteroids -- -n    # run with the interpreter
cargo test -p asteroids                  # compare interpreter and JIT frames
```

Requires the `riscv64gc-unknown-linux-gnu` target for the pinned toolchain
(`rustup target add riscv64gc-unknown-linux-gnu`) and a RISC-V cross linker
(`riscv64-linux-gnu-gcc-14`, overridden by
`CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER`).

Controls: **A**/**D** or arrow keys to move; **Q** or **Esc** to quit.
The script writes to `game_log.txt`.

## Architecture

```text
Host engine (src/)                         Guest script (guest/src/)
Game::new: load ELF, configure heap,
  Linux environment, and host functions
Run main()                         ─────▶ main(): initialize runtime, ready()
                                   ◀───── ready(): stop the machine
VM call game_init                  ─────▶ game_init()
Each frame: clear screen,
  VM call game_update              ─────▶ game_update(): input, physics,
  (32M instruction limit)                    collisions, drawing
Host system call handlers          ◀───── draw_pixel(), get_delta_time(), ...
  read sysargs, update HostState,           via dyncalls
  set_result
```

- `src/api.rs` defines the system call numbers shared by host and guest.
- `src/lib.rs` implements the screen, host functions, and `Game`. Handlers
  access `HostState` through machine userdata. `draw_text` and `log` use
  `Machine::with_userdata` to update state while borrowing guest strings.
- `src/main.rs` implements the terminal interface. It configures raw mode
  through `stty` without unsafe code.
- `guest/src/main.rs` implements the game logic.
- `guest/src/host.rs` provides reusable guest bindings and allocator macros.

## Guest bindings

`host_functions!` generates safe functions from declarations:

```rust
host::host_functions! {
    api::DRAW_TEXT => fn draw_text(x: i32, y: i32, text: &str);
    api::SIN_F32 => fn sin_f32(x: f32) -> f32;
}
```

Each function uses a naked stub containing a counted dyncall and `ret`.
The compiler passes arguments according to the guest calling convention,
which matches the host's `sysargs` decoding. A `&str` occupies two integer
registers; an `f32` occupies one FP register. The macro derives dyncall
register counts from argument types.

`native_heap!` installs the host's native heap as the global allocator.
`native_memory!` forwards `memcpy`, `memset`, `memmove`, and `memcmp` to the
host, including Rust and glibc calls linked with `--wrap` in
`guest/.cargo/config.toml`.

A seeded host random number generator makes games reproducible. Headless
tests compare the interpreter and JIT screen and instruction count on every
frame.
