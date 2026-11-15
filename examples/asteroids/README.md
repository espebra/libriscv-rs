# Asteroid Dodge

A terminal game where the engine is a Rust program and the game logic is a
Rust script running in a sandboxed RISC-V machine (RV64GCB Linux, static
glibc). It's the libriscv-rs version of `libloong/examples/gamedev`, built
on this crate's scripting API: dyncalls are system calls, typed `sysargs`,
userdata, VM calls and the native heap.

```sh
cargo run --release -p asteroids          # builds the script, then plays (JIT)
cargo run --release -p asteroids -- -n    # interpreted
cargo test -p asteroids                   # plays headless: interpreter vs JIT, frame by frame
```

You need the `riscv64gc-unknown-linux-gnu` target for the pinned toolchain
(`rustup target add riscv64gc-unknown-linux-gnu`) and a RISC-V cross linker
(`riscv64-linux-gnu-gcc-14`; override it with
`CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER`).

Controls: **A**/**D** or the arrow keys move, **Q** or **Esc** quits. The
script logs to `game_log.txt`.

## How it fits together

```
 engine (src/)                                script (guest/src/)
 ───────────────────────────────              ────────────────────────────────
 Game::new: load ELF, native heap,            main(): runtime up, then ready()
   Linux, host functions; run main()  ─────▶    → host stops the machine
   until ready(), vmcall game_init
 every frame: clear screen,           vmcall  game_update(): input, physics,
   vmcall game_update (32M insn cap)  ─────▶    collisions, draw
 host functions (system calls)       ◀─────   draw_pixel(..), get_delta_time(), ...
   read sysargs, change HostState,    dyncall   one instruction each, no ecall
   set_result
```

- **`src/api.rs`** is the ABI: the system call number of every host
  function. Both sides include it, so they can't disagree on a number.
- **`src/lib.rs`** is the engine: the screen, the host functions
  (`fn(&mut Machine) -> Result<()>` handlers over `HostState`, the
  machine's userdata) and `Game`, which boots the script and runs frames.
  `draw_text` and `log` use `Machine::with_userdata`, which lends a handler
  its state and the machine together, so the `&str` it borrows from guest
  memory can go straight onto the screen.
- **`src/main.rs`** is the terminal front end (raw mode through `stty`, so
  the whole engine stays free of `unsafe`).
- **`guest/src/host.rs`** is the guest half of the scripting API, and not
  specific to the game:
  - `host_functions!` turns a list of declarations into safe functions:

    ```rust
    host::host_functions! {
        api::DRAW_TEXT => fn draw_text(x: i32, y: i32, text: &str);
        api::SIN_F32 => fn sin_f32(x: f32) -> f32;
    }
    ```

    Each calls a naked stub, a counted dyncall and a `ret`, so the compiler
    places the arguments by the calling convention, which is exactly how the
    host's `sysargs` reads them (a `&str` is a pointer and a length in two
    integer registers, an `f32` an FP register). The register counts in the
    dyncall are computed from the argument types.
  - `native_heap!` makes the host's native heap the global allocator, and
    `native_memory!` sends memcpy, memset, memmove and memcmp (Rust's and
    glibc's, through `--wrap` in `guest/.cargo/config.toml`) to the host.
- **`guest/src/main.rs`** is the game, ported from the libloong example.

The random numbers come from a seeded generator on the host, so a seed
replays a game exactly; the tests use that to check the JIT against the
interpreter, screen and instruction count, on every frame.
