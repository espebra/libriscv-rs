//! Cached JIT code can't stand in for other code: a unit is only used for
//! a segment with exactly the bytes it was compiled from, even when its
//! key (whose CRC32-C and hash a guest can forge) says otherwise, and a
//! cache directory doesn't save code that didn't come from a file the host
//! chose. In its own test binary, since the registry is process-wide.
//! Runs native code: meaningful on x86-64 Linux only (skipped elsewhere).

#![cfg(feature = "jit")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use libriscv::jit_cache::{self, CacheDir, JitCacheOutput};
use libriscv::{Machine, MachineOptions};

const A: u64 = 0x10000;

const fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
    ((imm as u32 & 0xFFF) << 20) | (rs1 << 15) | (rd << 7) | 0x13
}
const fn add(rd: u32, rs1: u32, rs2: u32) -> u32 {
    (rs2 << 20) | (rs1 << 15) | (rd << 7) | 0x33
}
const fn bne(rs1: u32, rs2: u32, off: i32) -> u32 {
    let o = off as u32;
    ((o >> 12 & 1) << 31)
        | ((o >> 5 & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (1 << 12)
        | ((o >> 1 & 0xF) << 8)
        | ((o >> 11 & 1) << 7)
        | 0x63
}

/// A loop adding `step` to a0 and a0 to a1 500 times, then spinning.
fn segment(step: i32) -> Vec<u8> {
    [
        addi(5, 0, 500),
        addi(10, 10, step),
        add(11, 11, 10),
        addi(5, 5, -1),
        bne(5, 0, -12),
        0x0000_006F, // j .
    ]
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect()
}

/// `codegen::hash64`, the key's hash of the code.
fn hash64(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    let (words, rest) = bytes.as_chunks::<8>();
    for &w in words {
        h = (h ^ u64::from_le_bytes(w)).wrapping_mul(PRIME);
    }
    for &b in rest {
        h = (h ^ u64::from(b)).wrapping_mul(PRIME);
    }
    (h ^ bytes.len() as u64).wrapping_mul(PRIME)
}

/// Machines that don't share segments: each looks its own up.
fn options() -> MachineOptions {
    MachineOptions {
        memory_max: 1 << 20,
        use_shared_execute_segments: false,
        ..Default::default()
    }
}

fn machine(code: &[u8], options: MachineOptions) -> Machine {
    let mut m = Machine::new([].into(), options).unwrap();
    m.init_execute_area(code, A).unwrap();
    m.cpu.jump(A).unwrap();
    m
}

fn state(m: &mut Machine) -> Vec<u64> {
    let result = m.try_simulate(10_000, 0);
    assert!(result.is_ok(), "{result:?}");
    let mut s = m.cpu.registers().all().to_vec();
    s.extend([m.cpu.pc(), m.instruction_counter()]);
    s
}

fn cached(m: &Machine) -> bool {
    m.memory.exec_segment_for(A).unwrap().jit_is_cached()
}

/// The JIT runs here and a host C compiler driver links shared objects.
fn supported() -> bool {
    if !cfg!(all(target_arch = "x86_64", target_os = "linux")) {
        eprintln!("SKIPPED: the JIT cache needs x86-64 Linux");
        return false;
    }
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let have_cc = Command::new(&cc)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    testutil::tool_available(&format!("host `{cc}`"), have_cc)
}

fn scratch(name: &str) -> PathBuf {
    let dir = testutil::workspace_root()
        .join("target")
        .join("jit-cache-tests")
        .join("integrity");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn link(object: &Path, so: &Path) {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(cc)
        .args(["-shared", "-nostdlib", "-Wl,-z,noexecstack", "-o"])
        .arg(so)
        .arg(object)
        .arg("-lm")
        .status()
        .unwrap();
    assert!(status.success());
}

/// A shared object whose unit claims the key of one segment but holds
/// the code of another (as a guest that forged a collision would have it
/// saved) is never used for the first, which runs like the interpreter.
#[test]
fn a_forged_key_does_not_select_other_code() {
    if !supported() {
        return;
    }
    let (victim, forged) = (segment(3), segment(5));
    assert_eq!(victim.len(), forged.len());
    let expected = state(&mut machine(
        &victim,
        MachineOptions {
            jit_cache: false,
            ..options()
        },
    ));

    let output = JitCacheOutput::new();
    let mut m = machine(
        &forged,
        MachineOptions {
            jit: true,
            jit_background: false,
            jit_cache_output: Some(output.clone()),
            ..options()
        },
    );
    let forged_state = state(&mut m);
    assert_ne!(forged_state, expected);
    assert_eq!(output.len(), 1);

    // Put the victim's CRC32-C and hash in the unit's key: the words after
    // the metadata magic are build, vaddr, len, crc32c and hash.
    let victim_crc = machine(&victim, options())
        .memory
        .exec_segment_for(A)
        .unwrap()
        .crc32c_hash();
    let mut object = output.object().unwrap();
    let at = object
        .windows(8)
        .position(|w| w == b"RVJITM02")
        .expect("metadata magic");
    let word = |i: usize| at + 8 + i * 8;
    assert_eq!(object[word(1)..word(2)], A.to_le_bytes());
    object[word(3)..word(4)].copy_from_slice(&u64::from(victim_crc).to_le_bytes());
    object[word(4)..word(5)].copy_from_slice(&hash64(&victim).to_le_bytes());
    let o = scratch("forged.o");
    let so = scratch("forged.so");
    std::fs::write(&o, &object).unwrap();
    link(&o, &so);
    assert_eq!(jit_cache::load_shared_object(&so).unwrap(), 1);

    let mut m = machine(&victim, options());
    assert!(!cached(&m), "code compiled for other bytes was installed");
    assert_eq!(state(&mut m), expected);

    // The unmodified unit is used for its own code.
    let so = scratch("genuine.so");
    output.write_shared_object(&so).unwrap();
    assert_eq!(jit_cache::load_shared_object(&so).unwrap(), 1);
    let mut m = machine(&forged, options());
    assert!(cached(&m));
    assert_eq!(state(&mut m), forged_state);
    let mut m = machine(&victim, options());
    assert!(!cached(&m));
    assert_eq!(state(&mut m), expected);
}

/// A cache directory saves no segment the host didn't map from a file:
/// not code given to `init_execute_area` in its ranges (where a guest's
/// own code would be), only the initial segment it is told to.
#[test]
fn a_cache_directory_saves_only_host_files() {
    if !supported() {
        return;
    }
    let dir = scratch("cache");
    let cache_dir = CacheDir {
        path: dir.clone(),
        initial_segment: false,
        ranges: std::iter::once(A..A + 0x10000).collect(),
    };
    let mut m = machine(
        &segment(7),
        MachineOptions {
            jit: true,
            jit_background: false,
            jit_cache_dir: Some(cache_dir),
            ..options()
        },
    );
    state(&mut m);
    jit_cache::wait_for_saves();
    assert!(
        !dir.exists(),
        "{:?}",
        std::fs::read_dir(&dir).map(|d| d.count())
    );
}
