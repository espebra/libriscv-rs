//! Asteroid Dodge: the engine's terminal front end. Builds the script,
//! then runs the game at ~60 frames per second until the player is hit or
//! quits.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::process::{Command, ExitCode, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use asteroids::{Game, GameOptions, SCREEN_HEIGHT, build_guest};

fn usage() {
    println!("Usage: asteroids [OPTIONS]\n");
    println!("Options:");
    println!("  -n, --no-jit       Interpret the script instead of compiling it");
    println!("      --seed N       Seed the asteroids (default: the clock)");
    println!("      --guest FILE   Run this script instead of building guest/");
    println!("  -h, --help         Show this help\n");
    println!("Controls:");
    println!("  A/D or arrow keys  Move left/right");
    println!("  Q or Esc           Quit");
}

/// A key, from the terminal.
enum Key {
    Char(u8),
    Quit,
}

/// The terminal in non-canonical mode without echo, restored when dropped.
/// `stty` does the work, which keeps this program free of `unsafe`.
struct RawTerminal {
    saved: String,
}

impl RawTerminal {
    fn enable() -> Option<Self> {
        // `output()` gives the child a null stdin unless told otherwise,
        // and stty works on the terminal that is its stdin
        let out = Command::new("stty")
            .arg("-g")
            .stdin(Stdio::inherit())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let saved = String::from_utf8(out.stdout).ok()?.trim().to_owned();
        let ok = Command::new("stty")
            .args(["-icanon", "-echo", "min", "1"])
            .status()
            .is_ok_and(|s| s.success());
        // Hide the cursor
        print!("\x1b[?25l");
        ok.then_some(Self { saved })
    }
}

impl Drop for RawTerminal {
    fn drop(&mut self) {
        let _ = Command::new("stty").arg(&self.saved).status();
        print!("\x1b[?25h");
        let _ = std::io::stdout().flush();
    }
}

/// Read keys on a thread of their own, so the frame loop never blocks.
fn spawn_input() -> Receiver<Key> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut stdin = std::io::stdin().lock();
        let mut buf = [0u8; 32];
        while let Ok(n) = stdin.read(&mut buf) {
            if n == 0 {
                break;
            }
            for key in parse_keys(buf.get(..n).unwrap_or_default()) {
                if tx.send(key).is_err() {
                    return;
                }
            }
        }
    });
    rx
}

/// Arrow keys arrive as escape sequences in one read; a lone Esc quits.
fn parse_keys(bytes: &[u8]) -> Vec<Key> {
    let mut keys = Vec::new();
    let mut rest = bytes;
    while let Some((&first, tail)) = rest.split_first() {
        rest = tail;
        keys.push(match first {
            0x1b => match rest {
                [b'[', b'D', tail @ ..] => {
                    rest = tail;
                    Key::Char(b'a')
                }
                [b'[', b'C', tail @ ..] => {
                    rest = tail;
                    Key::Char(b'd')
                }
                [b'[', _, tail @ ..] => {
                    rest = tail;
                    continue;
                }
                _ => Key::Quit,
            },
            b'q' | b'Q' => Key::Quit,
            c => Key::Char(c),
        });
    }
    keys
}

/// The frame, with a status line over the bottom border.
fn render(
    game: &Game,
    update_time: Duration,
    avg: &mut [f64; 64],
    frame: usize,
    instructions: u64,
) {
    let micros = update_time.as_secs_f64() * 1e6;
    if let Some(slot) = avg.get_mut(frame % 64) {
        *slot = micros;
    }
    let average = avg.iter().sum::<f64>() / 64.0;
    let mut out = String::with_capacity(4096);
    out.push_str("\x1b[H");
    for line in game.screen().lines().take(SCREEN_HEIGHT - 1) {
        out.push_str(line);
        out.push('\n');
    }
    let status = format!(
        "+= libriscv {}  Time: {micros:.2}us (avg={average:.2}us)  Instr: {instructions} ",
        if game.jit() { "JIT" } else { "Interp" },
    );
    out.push_str(&format!("{status:=<80}"));
    let mut stdout = std::io::stdout().lock();
    let _ = stdout.write_all(out.as_bytes());
    let _ = stdout.flush();
}

fn append_log(lines: &[String]) {
    if lines.is_empty() {
        return;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("game_log.txt");
    if let Ok(mut file) = file {
        for line in lines {
            let _ = writeln!(file, "[GAME LOG] {line}");
        }
    }
}

fn main() -> ExitCode {
    let mut options = GameOptions {
        jit_background: true,
        print: true,
        seed: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |d| d.as_nanos() as u64),
        ..GameOptions::default()
    };
    let mut guest = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                usage();
                return ExitCode::SUCCESS;
            }
            "-n" | "--no-jit" => options.jit = false,
            "--seed" => match args.next().and_then(|s| s.parse().ok()) {
                Some(seed) => options.seed = seed,
                None => {
                    eprintln!("Error: --seed needs a number");
                    return ExitCode::FAILURE;
                }
            },
            "--guest" => guest = args.next(),
            _ => {
                eprintln!("Error: Unknown option '{arg}'\n");
                usage();
                return ExitCode::FAILURE;
            }
        }
    }

    let path = match guest {
        Some(path) => path.into(),
        None => {
            println!("Building the script...");
            match build_guest() {
                Ok(path) => path,
                Err(e) => {
                    eprintln!("Error: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
    };
    let elf = match std::fs::read(&path) {
        Ok(elf) => elf,
        Err(e) => {
            eprintln!("Error: {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                     ASTEROID DODGE - libriscv Game                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");
    println!("Loading game from: {}", path.display());
    println!("Controls: A/D or Arrow Keys to move, Q to quit");
    println!("Press Enter to start...");
    let _ = std::io::stdin().read_line(&mut String::new());

    let mut game = match Game::new(&elf, &options) {
        Ok(game) => game,
        Err(e) => {
            eprintln!("Error: {e}");
            return ExitCode::FAILURE;
        }
    };
    append_log(&game.take_log());

    let terminal = RawTerminal::enable();
    let keys = spawn_input();
    print!("\x1b[2J");

    let mut avg = [0.0; 64];
    let mut last = Instant::now();
    let mut frame = 0;
    let mut quit = false;
    let mut pending = VecDeque::new();
    let mut result = Ok(());
    while game.running() && !quit {
        let now = Instant::now();
        let delta_time = (now - last).as_secs_f32();
        last = now;

        // The script reads one key per frame; the rest wait their turn
        for key in keys.try_iter() {
            match key {
                Key::Quit => quit = true,
                Key::Char(c) => pending.push_back(c),
            }
        }
        let input = pending.pop_front();

        let start = Instant::now();
        let stats = match game.frame(delta_time, input) {
            Ok(stats) => stats,
            Err(e) => {
                result = Err(e);
                break;
            }
        };
        let update_time = start.elapsed();
        append_log(&game.take_log());
        render(&game, update_time, &mut avg, frame, stats.instructions);
        frame += 1;

        std::thread::sleep(Duration::from_millis(16));
    }
    drop(terminal);

    print!("\x1b[2J\x1b[H");
    if let Err(e) = result {
        eprintln!("Error in the script: {e}");
        return ExitCode::FAILURE;
    }
    println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                              GAME OVER                                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!(
        "\nFinal Score: {}  JIT: {}",
        game.score(),
        if game.jit() { "Enabled" } else { "Disabled" }
    );
    println!("\nThanks for playing!");
    ExitCode::SUCCESS
}
