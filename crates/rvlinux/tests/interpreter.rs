//! A dynamic executable's `PT_INTERP` must not choose a host file:
//! rvlinux always runs its own dynamic linker. Before, it read whatever
//! path the guest named, so `/dev/zero` exhausted host memory and a FIFO
//! hung rvlinux.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

const WARNING: &str = "Ignoring the program's interpreter";

/// A minimal RV64 ELF (`ET_DYN`) whose only program header is a
/// `PT_INTERP` naming `interp`.
fn elf_with_interp(interp: &str) -> Vec<u8> {
    let mut e = Vec::new();
    e.extend_from_slice(b"\x7fELF\x02\x01\x01\0\0\0\0\0\0\0\0\0");
    e.extend_from_slice(&3u16.to_le_bytes()); // e_type: ET_DYN
    e.extend_from_slice(&0xf3u16.to_le_bytes()); // e_machine: RISC-V
    e.extend_from_slice(&1u32.to_le_bytes()); // e_version
    e.extend_from_slice(&0x1000u64.to_le_bytes()); // e_entry
    e.extend_from_slice(&64u64.to_le_bytes()); // e_phoff
    e.extend_from_slice(&0u64.to_le_bytes()); // e_shoff
    e.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    e.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
    e.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
    e.extend_from_slice(&1u16.to_le_bytes()); // e_phnum
    e.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
    e.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
    e.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx
    assert_eq!(e.len(), 64);
    let len = interp.len() as u64 + 1;
    e.extend_from_slice(&3u32.to_le_bytes()); // p_type: PT_INTERP
    e.extend_from_slice(&4u32.to_le_bytes()); // p_flags: R
    e.extend_from_slice(&120u64.to_le_bytes()); // p_offset
    e.extend_from_slice(&0u64.to_le_bytes()); // p_vaddr
    e.extend_from_slice(&0u64.to_le_bytes()); // p_paddr
    e.extend_from_slice(&len.to_le_bytes()); // p_filesz
    e.extend_from_slice(&len.to_le_bytes()); // p_memsz
    e.extend_from_slice(&1u64.to_le_bytes()); // p_align
    assert_eq!(e.len(), 120);
    e.extend_from_slice(interp.as_bytes());
    e.push(0);
    e
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("interpreter");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// Runs rvlinux on an ELF whose `PT_INTERP` is `interp`; it must finish
/// within the time limit (it used to hang or run out of memory).
fn run_with_interp(name: &str, interp: &str) -> Output {
    let elf = scratch(name);
    std::fs::write(&elf, elf_with_interp(interp)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_rvlinux"))
        .arg("-n")
        .arg(&elf)
        .current_dir(elf.parent().unwrap())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if start.elapsed() > Duration::from_secs(60) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("rvlinux didn't finish with PT_INTERP {interp:?}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    child.wait_with_output().unwrap()
}

#[test]
fn interp_dev_zero_is_ignored() {
    let out = run_with_interp("dev_zero.elf", "/dev/zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(WARNING) && stderr.contains("/dev/zero"),
        "{stderr}"
    );
}

#[cfg(unix)]
#[test]
fn interp_fifo_is_ignored() {
    let fifo = scratch("fifo");
    let _ = std::fs::remove_file(&fifo);
    let status = Command::new("mkfifo").arg(&fifo).status().unwrap();
    assert!(status.success());
    let out = run_with_interp("fifo.elf", fifo.to_str().unwrap());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains(WARNING), "{stderr}");
}

#[test]
fn interp_other_file_is_ignored() {
    // Another readable file (here: the program itself, by a relative and
    // by an absolute path) must not become the initial segment.
    let out = run_with_interp("self.elf", "self.elf");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains(WARNING), "{stderr}");
    let path = scratch("self_abs.elf");
    let out = run_with_interp("self_abs.elf", path.to_str().unwrap());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains(WARNING), "{stderr}");
}

#[test]
fn interp_canonical_linker_is_accepted() {
    for (name, interp) in [
        ("canonical_lib.elf", "/lib/ld-linux-riscv64-lp64d.so.1"),
        (
            "canonical_usr.elf",
            "/usr/riscv64-linux-gnu/lib/ld-linux-riscv64-lp64d.so.1",
        ),
    ] {
        let out = run_with_interp(name, interp);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!stderr.contains(WARNING), "{stderr}");
    }
}
