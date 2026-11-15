//! The game, played without a terminal: frames at a fixed rate with
//! scripted input. Needs the Rust guest toolchain (skipped without it).

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::sync::OnceLock;

use asteroids::{Game, GameOptions, SCREEN_HEIGHT, SCREEN_WIDTH, build_guest};

const DT: f32 = 1.0 / 60.0;
/// The row the ship is on (`PLAYER_Y` in the script).
const PLAYER_Y: usize = SCREEN_HEIGHT - 3;

fn script() -> Option<&'static [u8]> {
    static ELF: OnceLock<Vec<u8>> = OnceLock::new();
    if !testutil::need_rust_toolchain() {
        return None;
    }
    Some(ELF.get_or_init(|| std::fs::read(build_guest().unwrap()).unwrap()))
}

fn game(jit: bool, seed: u64) -> Option<Game> {
    let options = GameOptions {
        jit,
        seed,
        ..GameOptions::default()
    };
    Some(Game::new(script()?, &options).unwrap())
}

/// Where the ship's 'A' is.
fn ship_x(game: &Game) -> usize {
    (1..SCREEN_WIDTH - 1)
        .find(|&x| game.screen().at(x, PLAYER_Y) == Some(b'A'))
        .expect("the ship is drawn")
}

#[test]
fn plays_until_hit() {
    let Some(mut game) = game(false, 7) else {
        return;
    };
    assert_eq!(game.take_log(), ["Game initialized"]);
    let mut frames = 0;
    while game.running() {
        game.frame(DT, None).unwrap();
        frames += 1;
        assert!(frames < 20_000, "a ship that never moves gets hit");
    }
    // The engine's border, the script's HUD and ship
    let lines: Vec<&str> = game.screen().lines().collect();
    assert!(lines[0].starts_with("+===") && lines[0].ends_with("==+"));
    assert!(lines[1].contains(&format!("Score: {}", game.score())));
    assert!(lines[1].contains("Difficulty: "));
    assert!(lines[SCREEN_HEIGHT - 2].contains("A/D: Move  Q: Quit"));
    assert!(game.score() > 0 && game.score() % 10 == 0);
    let log = game.take_log();
    assert_eq!(log.len(), 1);
    assert!(log[0].starts_with(&format!("Hit after {frames} frames")));
}

#[test]
fn keys_move_the_ship() {
    let Some(mut game) = game(false, 3) else {
        return;
    };
    game.frame(DT, None).unwrap();
    let start = ship_x(&game);
    assert_eq!(start, SCREEN_WIDTH / 2);
    for _ in 0..5 {
        game.frame(DT, Some(b'a')).unwrap();
    }
    assert_eq!(ship_x(&game), start - 5);
    for _ in 0..3 {
        game.frame(DT, Some(b'l')).unwrap();
    }
    assert_eq!(ship_x(&game), start - 2);
    // Other keys do nothing, and the border stops the ship
    game.frame(DT, Some(b'x')).unwrap();
    assert_eq!(ship_x(&game), start - 2);
    for _ in 0..SCREEN_WIDTH {
        game.frame(DT, Some(b'D')).unwrap();
    }
    assert_eq!(ship_x(&game), SCREEN_WIDTH - 3);
}

/// The same game under the JIT and the interpreter: the same screen and the
/// same instruction count, frame by frame.
#[test]
fn jit_plays_like_the_interpreter() {
    if !cfg!(feature = "jit") {
        return;
    }
    let (Some(mut interp), Some(mut jit)) = (game(false, 11), game(true, 11)) else {
        return;
    };
    assert!(jit.jit() && !interp.jit());
    let keys = b"aaadddd hlhl";
    for frame in 0..1500 {
        let input = (frame % 7 == 0).then(|| keys[frame / 7 % keys.len()]);
        let a = interp.frame(DT, input).unwrap();
        let b = jit.frame(DT, input).unwrap();
        assert_eq!(a, b, "frame {frame}");
        assert_eq!(interp.screen(), jit.screen(), "frame {frame}");
        assert_eq!(interp.running(), jit.running(), "frame {frame}");
        if !interp.running() {
            break;
        }
    }
    assert_eq!(interp.score(), jit.score());
    assert_eq!(interp.take_log(), jit.take_log());
}

/// Each frame is a VM call with an instruction limit, and the native heap
/// keeps the script's allocations balanced across frames.
#[test]
fn frames_are_bounded() {
    let Some(mut game) = game(false, 5) else {
        return;
    };
    let arena = |g: &Game| {
        let arena = g.machine().arena().unwrap();
        arena.allocation_counter() - arena.deallocation_counter()
    };
    game.frame(DT, None).unwrap();
    let live = arena(&game);
    for _ in 0..100 {
        let stats = game.frame(DT, None).unwrap();
        assert!(stats.instructions > 0 && stats.instructions < asteroids::MAX_INSTRUCTIONS);
    }
    // Particles come and go, but the frame's strings are freed
    assert!(arena(&game) <= live + 1, "{} > {live}", arena(&game));
}
