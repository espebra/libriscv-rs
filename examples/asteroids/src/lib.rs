//! Asteroid Dodge's engine: a terminal renderer and frame loop in Rust,
//! with the game logic in a Rust script that runs inside a sandboxed
//! RISC-V machine (`guest/`).
//!
//! The engine owns the screen, the input, the clock and the score. The
//! script gets them through host functions, which are system calls (see
//! [`api`]): the guest calls each with one dyncall, and the engine's
//! handler reads the arguments with [`Machine::sysargs`], works on
//! [`HostState`] (the machine's userdata) and stores the result with
//! [`Machine::set_result`]. Every frame the engine calls the script's
//! `game_update` with a VM call, under an instruction limit.

pub mod api;

use std::path::{Path, PathBuf};
use std::process::Command;

use libriscv::{ExceptionKind, Machine, MachineError, MachineOptions, Result, SyscallHandler};

pub const SCREEN_WIDTH: usize = 80;
pub const SCREEN_HEIGHT: usize = 24;

/// The native heap the script allocates from.
const HEAP_SIZE: u64 = 4 << 20;
/// Instructions the script may run before it's ready (`main()`).
const BOOT_INSTRUCTIONS: u64 = 100_000_000;
/// Instructions a call into the script may run.
pub const MAX_INSTRUCTIONS: u64 = 32_000_000;

/// The ASCII frame buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Screen {
    rows: [[u8; SCREEN_WIDTH]; SCREEN_HEIGHT],
}

impl Default for Screen {
    fn default() -> Self {
        Self {
            rows: [[b' '; SCREEN_WIDTH]; SCREEN_HEIGHT],
        }
    }
}

impl Screen {
    /// Blank, inside a border.
    pub fn clear(&mut self) {
        *self = Self::default();
        let (right, bottom) = (SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1);
        for (y, row) in self.rows.iter_mut().enumerate() {
            for (x, cell) in row.iter_mut().enumerate() {
                let edge_x = x == 0 || x == right;
                let edge_y = y == 0 || y == bottom;
                *cell = match (edge_x, edge_y) {
                    (true, true) => b'+',
                    (false, true) => b'=',
                    (true, false) => b'|',
                    (false, false) => b' ',
                };
            }
        }
    }

    /// Draw a character; positions off the screen are ignored. Only
    /// printable ASCII is drawn, so a script can't send escape codes to
    /// the terminal.
    pub fn put(&mut self, x: i32, y: i32, c: u8) {
        let (Ok(x), Ok(y)) = (usize::try_from(x), usize::try_from(y)) else {
            return;
        };
        if let Some(cell) = self.rows.get_mut(y).and_then(|row| row.get_mut(x)) {
            *cell = if c.is_ascii_graphic() || c == b' ' {
                c
            } else {
                b'?'
            };
        }
    }

    /// Draw text from `(x, y)` rightwards, clipped at the edge.
    pub fn text(&mut self, x: i32, y: i32, text: &str) {
        for (i, c) in text.bytes().enumerate() {
            let Some(x) = i32::try_from(i).ok().and_then(|i| x.checked_add(i)) else {
                return;
            };
            self.put(x, y, c);
        }
    }

    /// The character at `(x, y)`.
    pub fn at(&self, x: usize, y: usize) -> Option<u8> {
        self.rows.get(y)?.get(x).copied()
    }

    /// The rows, as text.
    pub fn lines(&self) -> impl Iterator<Item = &str> {
        // Only ASCII is ever stored.
        self.rows
            .iter()
            .map(|row| std::str::from_utf8(row).unwrap_or_default())
    }
}

/// A small deterministic generator (xorshift64*), so that a seed replays a
/// game exactly.
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform enough in `min..=max` (`min` when the range is empty).
    fn range(&mut self, min: i32, max: i32) -> i32 {
        if max <= min {
            return min;
        }
        let span = (i64::from(max) - i64::from(min) + 1) as u64;
        (i64::from(min) + (self.next() % span) as i64) as i32
    }
}

/// What the host functions work on: the machine's userdata.
#[derive(Debug)]
pub struct HostState {
    pub screen: Screen,
    pub score: i32,
    /// False once the script called `game_over`.
    pub running: bool,
    /// Seconds since the last frame, as the script sees it.
    pub delta_time: f32,
    /// The last key pressed, until the script reads it.
    pub input: Option<u8>,
    /// Lines from the script's `log`, until taken.
    pub log: Vec<String>,
    /// True once the script called `ready`.
    ready: bool,
    rng: Rng,
}

fn state(m: &mut Machine) -> Result<&mut HostState> {
    m.userdata_mut::<HostState>().ok_or(NO_STATE)
}

const NO_STATE: MachineError =
    MachineError::new(ExceptionKind::IllegalOperation, "No engine state", 0);

// The host functions. Each is the system call the guest's dyncall names,
// with the signature `guest/src/main.rs` declares for it.

fn ready(m: &mut Machine) -> Result<()> {
    state(m)?.ready = true;
    m.stop();
    Ok(())
}

fn draw_pixel(m: &mut Machine) -> Result<()> {
    let (x, y, c): (i32, i32, u8) = m.sysargs()?;
    state(m)?.screen.put(x, y, c);
    Ok(())
}

fn draw_text(m: &mut Machine) -> Result<()> {
    // The text is borrowed from guest memory while the screen changes
    m.with_userdata(|state: &mut HostState, m| {
        let (x, y, text): (i32, i32, &str) = m.sysargs()?;
        state.screen.text(x, y, text);
        Ok(())
    })
    .unwrap_or(Err(NO_STATE))
}

fn has_input(m: &mut Machine) -> Result<()> {
    let available = state(m)?.input.is_some();
    m.set_result(available);
    Ok(())
}

fn get_input(m: &mut Machine) -> Result<()> {
    let key = state(m)?.input.take().unwrap_or(0);
    m.set_result(key);
    Ok(())
}

fn get_delta_time(m: &mut Machine) -> Result<()> {
    let dt = state(m)?.delta_time;
    m.set_result(dt);
    Ok(())
}

fn get_score(m: &mut Machine) -> Result<()> {
    let score = state(m)?.score;
    m.set_result(score);
    Ok(())
}

fn add_score(m: &mut Machine) -> Result<()> {
    let (points,): (i32,) = m.sysargs()?;
    let s = state(m)?;
    s.score = s.score.saturating_add(points);
    Ok(())
}

fn game_over(m: &mut Machine) -> Result<()> {
    state(m)?.running = false;
    Ok(())
}

fn random_int(m: &mut Machine) -> Result<()> {
    let (min, max): (i32, i32) = m.sysargs()?;
    let value = state(m)?.rng.range(min, max);
    m.set_result(value);
    Ok(())
}

fn sin_f32(m: &mut Machine) -> Result<()> {
    let (x,): (f32,) = m.sysargs()?;
    m.set_result(x.sin());
    Ok(())
}

fn cos_f32(m: &mut Machine) -> Result<()> {
    let (x,): (f32,) = m.sysargs()?;
    m.set_result(x.cos());
    Ok(())
}

/// Lines a script may keep in the log before the engine takes them.
const MAX_LOG_LINES: usize = 1000;

fn log(m: &mut Machine) -> Result<()> {
    m.with_userdata(|state: &mut HostState, m| {
        let (message,): (&str,) = m.sysargs()?;
        if state.log.len() < MAX_LOG_LINES {
            state.log.push(message.to_owned());
        }
        Ok(())
    })
    .unwrap_or(Err(NO_STATE))
}

const HOST_FUNCTIONS: [(usize, SyscallHandler); 13] = [
    (api::READY, ready),
    (api::DRAW_PIXEL, draw_pixel),
    (api::DRAW_TEXT, draw_text),
    (api::HAS_INPUT, has_input),
    (api::GET_INPUT, get_input),
    (api::GET_DELTA_TIME, get_delta_time),
    (api::GET_SCORE, get_score),
    (api::ADD_SCORE, add_score),
    (api::GAME_OVER, game_over),
    (api::RANDOM_INT, random_int),
    (api::SIN_F32, sin_f32),
    (api::COS_F32, cos_f32),
    (api::LOG, log),
];

/// How to run the script.
#[derive(Debug, Clone)]
pub struct GameOptions {
    /// Compile the script with the JIT (with the `jit` feature).
    pub jit: bool,
    /// Compile in the background, interpreting meanwhile: starts at once.
    pub jit_background: bool,
    /// Seeds the random numbers the script gets.
    pub seed: u64,
    /// Let the script print to stdout.
    pub print: bool,
}

impl Default for GameOptions {
    fn default() -> Self {
        Self {
            jit: cfg!(feature = "jit"),
            jit_background: false,
            seed: 1,
            print: false,
        }
    }
}

/// One frame's cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameStats {
    /// Instructions `game_update` ran.
    pub instructions: u64,
}

/// The game script in its machine, booted and initialized.
#[derive(Debug)]
pub struct Game {
    machine: Machine,
    update: u64,
}

fn missing(message: &'static str) -> MachineError {
    MachineError::new(ExceptionKind::InvalidProgram, message, 0)
}

impl Game {
    /// Load the script (a static RV64 Linux ELF), run its `main()` until it
    /// is ready, and call its `game_init`.
    pub fn new(elf: &[u8], options: &GameOptions) -> Result<Self> {
        let mut m = Machine::new(
            elf,
            MachineOptions {
                jit: options.jit,
                jit_background: options.jit_background,
                ..MachineOptions::default()
            },
        )?;
        if !options.print {
            m.set_printer(|_, _| {});
        }
        m.set_userdata(HostState {
            screen: Screen::default(),
            score: 0,
            running: true,
            delta_time: 0.0,
            input: None,
            log: Vec::new(),
            ready: false,
            rng: Rng::new(options.seed),
        });
        // The script's allocator and memcpy and friends are the host's
        let heap = m.memory.mmap_allocate(HEAP_SIZE);
        m.setup_native_heap(api::NATIVE_HEAP, heap, HEAP_SIZE)?;
        m.setup_native_memory(api::NATIVE_MEMORY);
        // A Rust program needs Linux to start, but no files
        m.setup_linux_syscalls(false, false);
        // Rust's standard library locks with futexes even single-threaded
        m.setup_posix_threads();
        m.setup_linux(&["asteroids"], &["LC_ALL=C"])?;
        m.install_syscall_handlers(HOST_FUNCTIONS);

        m.simulate(BOOT_INSTRUCTIONS)?;
        if !state(&mut m)?.ready {
            return Err(missing("The script ended without calling ready()"));
        }
        let init = m.address_of("game_init")?;
        let update = m.address_of("game_update")?;
        if init == 0 || update == 0 {
            return Err(missing("The script has no game_init or game_update"));
        }
        m.vmcall_limited(MAX_INSTRUCTIONS, init, ())?;
        Ok(Self { machine: m, update })
    }

    /// Run one frame: clear the screen, let the script update and draw.
    /// `input` replaces a key the script hasn't read yet.
    pub fn frame(&mut self, delta_time: f32, input: Option<u8>) -> Result<FrameStats> {
        let s = self.state_mut();
        s.delta_time = delta_time;
        if input.is_some() {
            s.input = input;
        }
        s.screen.clear();
        self.machine
            .vmcall_limited(MAX_INSTRUCTIONS, self.update, ())?;
        Ok(FrameStats {
            instructions: self.machine.instruction_counter(),
        })
    }

    pub fn screen(&self) -> &Screen {
        &self.state().screen
    }

    pub fn score(&self) -> i32 {
        self.state().score
    }

    /// False once the script called `game_over`.
    pub fn running(&self) -> bool {
        self.state().running
    }

    /// Take the lines the script logged.
    pub fn take_log(&mut self) -> Vec<String> {
        std::mem::take(&mut self.state_mut().log)
    }

    /// True when the script is compiled to native code.
    pub fn jit(&self) -> bool {
        cfg!(feature = "jit") && self.machine.options().jit
    }

    pub fn machine(&self) -> &Machine {
        &self.machine
    }

    #[allow(clippy::expect_used)]
    fn state(&self) -> &HostState {
        // Invariant: set in `new` and never replaced or taken.
        self.machine.userdata().expect("engine state")
    }

    #[allow(clippy::expect_used)]
    fn state_mut(&mut self) -> &mut HostState {
        // Invariant: as in `state`.
        self.machine.userdata_mut().expect("engine state")
    }
}

/// Where the script's sources are.
pub fn guest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("guest")
}

/// Build the script with cargo (from `guest/`, so its `.cargo/config.toml`
/// applies: target, linker and flags), returning the ELF's path. Cargo
/// rebuilds only what changed.
pub fn build_guest() -> std::io::Result<PathBuf> {
    let dir = guest_dir();
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .current_dir(&dir)
        .args(["build", "--release", "--quiet", "--target-dir", "target"])
        .status()?;
    if !status.success() {
        return Err(std::io::Error::other(format!(
            "building the script in {} failed ({status})",
            dir.display()
        )));
    }
    Ok(dir.join("target/riscv64gc-unknown-linux-gnu/release/asteroids-guest"))
}
