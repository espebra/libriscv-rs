//! Asteroid Dodge: the game logic, running as a script inside the engine's
//! RISC-V machine. The engine calls `game_init` once and `game_update`
//! every frame; the game draws and reads input through host functions.

#[path = "../../src/api.rs"]
#[allow(dead_code)]
mod api;
mod host;

use std::sync::{LazyLock, Mutex};

host::native_heap!(api::NATIVE_HEAP);
host::native_memory!(api::NATIVE_MEMORY);

host::host_functions! {
    /// Tell the host the program is initialized; it stops the machine here,
    /// and calls the game's functions from then on.
    api::READY => fn ready();
    /// Draw a character at a screen position.
    api::DRAW_PIXEL => fn draw_pixel(x: i32, y: i32, c: u8);
    /// Draw text at a screen position.
    api::DRAW_TEXT => fn draw_text(x: i32, y: i32, text: &str);
    /// True when a key was pressed since the last `get_input`.
    api::HAS_INPUT => fn has_input() -> bool;
    /// The last key pressed.
    api::GET_INPUT => fn get_input() -> u8;
    /// Seconds since the last frame.
    api::GET_DELTA_TIME => fn get_delta_time() -> f32;
    api::GET_SCORE => fn get_score() -> i32;
    api::ADD_SCORE => fn add_score(points: i32);
    api::GAME_OVER => fn game_over();
    /// A random number in `min..=max`.
    api::RANDOM_INT => fn random_int(min: i32, max: i32) -> i32;
    api::SIN_F32 => fn sin_f32(x: f32) -> f32;
    api::COS_F32 => fn cos_f32(x: f32) -> f32;
    /// Append a line to the engine's log file.
    api::LOG => fn log(message: &str);
}

// Game constants
const SCREEN_WIDTH: i32 = 80;
const SCREEN_HEIGHT: i32 = 24;
const MAX_ASTEROIDS: usize = 10;
const PLAYER_Y: i32 = SCREEN_HEIGHT - 3;

#[derive(Clone, Copy)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    const fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }
}

struct Player {
    pos: Vec2,
    velocity: f32,
    sprite: u8,
}

impl Player {
    fn new() -> Self {
        Player {
            pos: Vec2::new((SCREEN_WIDTH / 2) as f32, PLAYER_Y as f32),
            velocity: 0.0,
            sprite: b'A',
        }
    }

    fn update(&mut self, delta_time: f32) {
        self.pos.x += self.velocity * delta_time * 30.0;
        // Clamp to the screen, inside the border
        self.pos.x = self.pos.x.clamp(2.0, (SCREEN_WIDTH - 3) as f32);
        // Friction
        self.velocity *= 0.85;
    }

    fn move_left(&mut self) {
        self.pos.x -= 1.0;
    }

    fn move_right(&mut self) {
        self.pos.x += 1.0;
    }

    fn draw(&self) {
        let (x, y) = (self.pos.x as i32, self.pos.y as i32);
        draw_pixel(x, y, self.sprite);
        draw_pixel(x - 1, y, b'/');
        draw_pixel(x + 1, y, b'\\');
    }
}

struct Asteroid {
    pos: Vec2,
    velocity: Vec2,
    active: bool,
    sprite: u8,
    rotation_time: f32,
}

impl Asteroid {
    const fn new() -> Self {
        Asteroid {
            pos: Vec2::new(0.0, 0.0),
            velocity: Vec2::new(0.0, 0.0),
            active: false,
            sprite: b'*',
            rotation_time: 0.0,
        }
    }

    fn spawn(&mut self, x: f32, speed: f32) {
        self.pos = Vec2::new(x, 1.0);
        self.velocity = Vec2::new(random_int(-10, 10) as f32 * 0.1, speed);
        self.active = true;
        self.rotation_time = 0.0;
        // Vary the appearance
        self.sprite = match random_int(0, 3) {
            0 => b'*',
            1 => b'o',
            2 => b'O',
            _ => b'@',
        };
    }

    fn update(&mut self, delta_time: f32) {
        if !self.active {
            return;
        }
        self.pos.x += self.velocity.x * delta_time * 10.0;
        self.pos.y += self.velocity.y * delta_time * 10.0;
        self.rotation_time += delta_time * 4.0;

        if self.pos.y >= (SCREEN_HEIGHT - 1) as f32 {
            self.active = false;
            // The player dodged it
            add_score(10);
        }
    }

    fn draw(&self) {
        if !self.active {
            return;
        }
        let (x, y) = (self.pos.x as i32, self.pos.y as i32);
        // A tumbling effect
        match (-2.0 + self.rotation_time.sin() * 2.0) as i32 {
            -1 => {
                draw_pixel(x - 1, y, b'.');
                draw_pixel(x, y, self.sprite);
                draw_pixel(x + 1, y, b'.');
            }
            0 | 1 => {
                draw_pixel(x - 1, y, self.sprite);
                draw_pixel(x + 1, y, self.sprite);
            }
            _ => draw_pixel(x, y, self.sprite),
        }
    }

    fn collides_with(&self, player: &Player) -> bool {
        if !self.active {
            return false;
        }
        let dx = (self.pos.x - player.pos.x).abs();
        let dy = (self.pos.y - player.pos.y).abs();
        dx < 2.0 && dy < 1.5
    }
}

/// Explosion debris.
struct Particle {
    pos: Vec2,
    velocity: Vec2,
    lifetime: f32,
}

impl Particle {
    fn new(x: f32, y: f32) -> Self {
        let angle = random_int(0, 360) as f32 * core::f32::consts::PI / 180.0;
        let speed = random_int(5, 15) as f32 * 0.1;
        Particle {
            pos: Vec2::new(x, y),
            velocity: Vec2::new(cos_f32(angle) * speed, sin_f32(angle) * speed),
            lifetime: 2.5,
        }
    }

    /// Returns whether the particle is still alive.
    fn update(&mut self, delta_time: f32) -> bool {
        self.pos.x += self.velocity.x * delta_time * 10.0;
        // Debris falls
        self.pos.y += 0.2;
        self.lifetime -= delta_time;
        self.lifetime > 0.0
    }

    fn draw(&self) {
        let c = if self.lifetime > 0.3 { b'*' } else { b'.' };
        draw_pixel(self.pos.x as i32, self.pos.y as i32, c);
    }
}

struct Game {
    player: Player,
    asteroids: [Asteroid; MAX_ASTEROIDS],
    particles: Vec<Particle>,
    spawn_timer: f32,
    spawn_rate: f32,
    difficulty: f32,
    frames: u32,
}

impl Game {
    fn new() -> Self {
        Game {
            player: Player::new(),
            asteroids: [const { Asteroid::new() }; MAX_ASTEROIDS],
            particles: Vec::new(),
            spawn_timer: 0.0,
            spawn_rate: 0.5,
            difficulty: 1.0,
            frames: 0,
        }
    }

    fn spawn_asteroid(&mut self) {
        if let Some(asteroid) = self.asteroids.iter_mut().find(|a| !a.active) {
            let x = random_int(5, SCREEN_WIDTH - 5) as f32;
            asteroid.spawn(x, 1.0 + self.difficulty * 0.5);
        }
    }

    fn spawn_explosion(&mut self, x: f32, y: f32) {
        self.particles.extend((0..8).map(|_| Particle::new(x, y)));
    }

    fn update(&mut self, delta_time: f32) {
        self.frames += 1;

        if has_input() {
            match get_input() {
                b'a' | b'A' | b'h' => self.player.move_left(),
                b'd' | b'D' | b'l' => self.player.move_right(),
                _ => {}
            }
        }

        self.player.update(delta_time);

        let mut collision = false;
        for asteroid in &mut self.asteroids {
            asteroid.update(delta_time);
            collision |= asteroid.collides_with(&self.player);
        }
        if collision {
            self.spawn_explosion(self.player.pos.x, self.player.pos.y);
            log(&format!(
                "Hit after {} frames, score {}",
                self.frames,
                get_score()
            ));
            game_over();
        }

        self.particles.retain_mut(|p| p.update(delta_time));

        self.spawn_timer += delta_time;
        if self.spawn_timer >= self.spawn_rate {
            self.spawn_timer = 0.0;
            self.spawn_asteroid();
        }

        // Harder over time
        if self.frames.is_multiple_of(300) {
            self.difficulty += 0.1;
            self.spawn_rate = (self.spawn_rate * 0.95).max(0.3);
        }
    }

    fn draw(&self) {
        self.player.draw();
        self.asteroids.iter().for_each(Asteroid::draw);
        self.particles.iter().for_each(Particle::draw);

        // The HUD
        draw_text(3, 1, &format!("Score: {}", get_score()));
        draw_text(
            SCREEN_WIDTH - 20,
            1,
            &format!("Difficulty: {:.1}", self.difficulty),
        );
        draw_text(3, SCREEN_HEIGHT - 2, "A/D: Move  Q: Quit");
    }
}

static GAME: LazyLock<Mutex<Game>> = LazyLock::new(|| Mutex::new(Game::new()));

// The functions the engine calls

#[unsafe(no_mangle)]
pub extern "C" fn game_init() {
    LazyLock::force(&GAME);
    log("Game initialized");
}

#[unsafe(no_mangle)]
pub extern "C" fn game_update() {
    // A poisoned lock can only come from a panic, which aborts
    let Ok(mut game) = GAME.lock() else { return };
    game.update(get_delta_time());
    game.draw();
}

/// The engine calls these by name, which nothing in the program does: this
/// keeps them (and their symbols) through dead code elimination and LTO.
#[used]
static ENTRY_POINTS: [extern "C" fn(); 2] = [game_init, game_update];

fn main() {
    println!(">>> Asteroid Dodge Game - Guest Module");
    // The host stops the machine here, with the runtime initialized
    ready();
}
