//! The ABI between the engine and the game script: the system call number
//! of every host function. Both sides include this file (the guest with
//! `#[path]`), so they cannot disagree on a number; the signatures are in
//! `guest/src/main.rs` (`host_functions!`) and the handlers in `engine.rs`.
//!
//! A Rust guest calls a host function with a dyncall, which is the system
//! call numbered by its immediate. Linux system calls end below 470, and
//! the table ends at 511 (the EBREAK handler), which leaves 490..=510 to
//! the game.

/// The native heap: malloc, calloc, realloc, free, meminfo (5 numbers).
pub const NATIVE_HEAP: usize = 470;
/// The native memory functions: memcpy, memset, memmove, memcmp, ...
/// (15 numbers).
pub const NATIVE_MEMORY: usize = 475;

/// `fn ready()`: the guest is initialized; stops the machine.
pub const READY: usize = 490;
/// `fn draw_pixel(x: i32, y: i32, c: u8)`
pub const DRAW_PIXEL: usize = 491;
/// `fn draw_text(x: i32, y: i32, text: &str)`
pub const DRAW_TEXT: usize = 492;
/// `fn has_input() -> bool`
pub const HAS_INPUT: usize = 493;
/// `fn get_input() -> u8`: the last key pressed, consuming it.
pub const GET_INPUT: usize = 494;
/// `fn get_delta_time() -> f32`: seconds since the last frame.
pub const GET_DELTA_TIME: usize = 495;
/// `fn get_score() -> i32`
pub const GET_SCORE: usize = 496;
/// `fn add_score(points: i32)`
pub const ADD_SCORE: usize = 497;
/// `fn game_over()`
pub const GAME_OVER: usize = 498;
/// `fn random_int(min: i32, max: i32) -> i32`: uniform in `min..=max`.
pub const RANDOM_INT: usize = 499;
/// `fn sin_f32(x: f32) -> f32`
pub const SIN_F32: usize = 500;
/// `fn cos_f32(x: f32) -> f32`
pub const COS_F32: usize = 501;
/// `fn log(message: &str)`
pub const LOG: usize = 502;
