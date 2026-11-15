//! The file system sandbox of the Linux system calls (`FileDescriptors`):
//! what `openat`, `faccessat` and `fstatat` let through to the host. The
//! system calls are invoked directly on an empty machine, so these need
//! no guest toolchain.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::registers::REG_ARG0;
use libriscv::{FileDescriptors, Machine, MachineOptions};
use std::path::PathBuf;
use std::time::Duration;

const SYS_FACCESSAT: u64 = 48;
const SYS_OPENAT: u64 = 56;
const SYS_READ: u64 = 63;
const SYS_FSTATAT: u64 = 79;
const AT_FDCWD: u64 = -100i64 as u64;
const O_RDONLY: u64 = 0;
const R_OK: u64 = 4;
const EPERM: i64 = 1;
const ENOENT: i64 = 2;
const EBADF: i64 = 9;
const EACCES: i64 = 13;

/// A fresh temporary directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("libriscv-fs-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // The filters see the path as given; temp_dir may be behind a
        // symlink (macOS), which is fine: links are followed on open.
        Self(dir)
    }
    fn path(&self, rel: &str) -> String {
        format!("{}/{rel}", self.0.to_str().unwrap())
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn machine(setup: impl FnOnce(&mut FileDescriptors)) -> Machine {
    let options = MachineOptions {
        memory_max: 8 << 20,
        ..Default::default()
    };
    let mut m = Machine::new(&[], options).unwrap();
    m.setup_linux_syscalls(true, false);
    let fds = m.fds_mut().unwrap();
    fds.permit_filesystem = true;
    setup(fds);
    m
}

fn syscall(m: &mut Machine, number: u64, args: &[u64]) -> i64 {
    for (i, &a) in args.iter().enumerate() {
        m.cpu.set_reg(REG_ARG0 + i as u32, a);
    }
    m.system_call(number).unwrap();
    m.return_value() as i64
}

/// Scratch guest memory: a bump allocator over the (writable) arena of
/// an empty machine, per thread. Machines are short-lived and only hold
/// strings and buffers, so reuse across machines doesn't matter.
fn alloc(len: u64) -> u64 {
    thread_local!(static NEXT: std::cell::Cell<u64> = const { std::cell::Cell::new(0x40000) });
    NEXT.with(|next| {
        let addr = next.get();
        let end = (addr + len + 15) & !15;
        next.set(if end > 0x40_0000 { 0x40000 } else { end });
        addr
    })
}

fn guest_str(m: &mut Machine, s: &str) -> u64 {
    let addr = alloc(s.len() as u64 + 1);
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    m.copy_to_guest(addr, &bytes).unwrap();
    addr
}

fn openat(m: &mut Machine, dir_fd: u64, path: &str) -> i64 {
    let p = guest_str(m, path);
    syscall(m, SYS_OPENAT, &[dir_fd, p, O_RDONLY, 0])
}

fn open(m: &mut Machine, path: &str) -> i64 {
    openat(m, AT_FDCWD, path)
}

fn read_all(m: &mut Machine, fd: i64) -> Vec<u8> {
    let buf = alloc(4096);
    let n = syscall(m, SYS_READ, &[fd as u64, buf, 4096]);
    assert!(n >= 0, "read: {n}");
    let mut out = vec![0u8; n as usize];
    m.copy_from_guest(&mut out, buf).unwrap();
    out
}

#[test]
fn openat_without_a_filter_is_denied() {
    let dir = TempDir::new("nofilter");
    let file = dir.path("file");
    std::fs::write(&file, b"secret").unwrap();

    // permit_filesystem alone admits nothing, as for faccessat and fstatat.
    let mut m = machine(|_| {});
    assert_eq!(open(&mut m, &file), -EPERM);
    let p = guest_str(&mut m, &file);
    assert_eq!(
        syscall(&mut m, SYS_FACCESSAT, &[AT_FDCWD, p, R_OK]),
        -EACCES
    );

    // Proxy mode admits every path.
    let mut m = machine(|fds| fds.proxy_mode = true);
    let fd = open(&mut m, &file);
    assert!(fd >= 0, "open in proxy mode: {fd}");
    assert_eq!(read_all(&mut m, fd), b"secret");
}

#[test]
fn filter_sees_normalised_absolute_paths() {
    let dir = TempDir::new("normalise");
    std::fs::create_dir_all(dir.path("allowed")).unwrap();
    std::fs::write(dir.path("allowed/x"), b"allowed").unwrap();
    std::fs::write(dir.path("secret"), b"secret").unwrap();
    let allowed_dir = dir.path("allowed");
    let prefix = format!("{allowed_dir}/");
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let log = seen.clone();
    let mut m = machine(move |fds| {
        fds.filter_open = Some(Box::new(move |p: &mut String| {
            log.lock().unwrap().push(p.clone());
            *p == allowed_dir || p.starts_with(&prefix)
        }));
    });

    // `..` can't climb out of the admitted prefix.
    let escape = dir.path("allowed/../secret");
    assert_eq!(open(&mut m, &escape), -EPERM);
    assert_eq!(open(&mut m, &dir.path("allowed/x/../../secret")), -EPERM);
    let p = guest_str(&mut m, &escape);
    assert_eq!(
        syscall(&mut m, SYS_FACCESSAT, &[AT_FDCWD, p, R_OK]),
        -EACCES
    );
    let stat = alloc(256);
    assert_eq!(
        syscall(&mut m, SYS_FSTATAT, &[AT_FDCWD, p, stat, 0]),
        -EPERM
    );
    assert_eq!(seen.lock().unwrap()[0], dir.path("secret"));

    // `.` and repeated slashes are collapsed too.
    let fd = open(&mut m, &dir.path("allowed/././/x"));
    assert!(fd >= 0, "open: {fd}");
    assert_eq!(read_all(&mut m, fd), b"allowed");

    // Relative to a directory descriptor: the filter judges the joined path.
    let dfd = open(&mut m, &dir.path("allowed"));
    assert!(dfd >= 0, "open dir: {dfd}");
    let fd = openat(&mut m, dfd as u64, "x");
    assert!(fd >= 0, "openat(dir, x): {fd}");
    assert_eq!(read_all(&mut m, fd), b"allowed");
    assert_eq!(openat(&mut m, dfd as u64, "../secret"), -EPERM);
    assert_eq!(seen.lock().unwrap().last().unwrap(), &dir.path("secret"));

    // Relative to AT_FDCWD there's no host directory; empty is ENOENT.
    assert_eq!(open(&mut m, "x"), -EBADF);
    assert_eq!(open(&mut m, ""), -ENOENT);
}

/// Run `f` on another thread, failing (instead of hanging) if it takes
/// longer than a few seconds; `unblock` is called then to release it.
fn bounded<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static, unblock: impl FnOnce()) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(Duration::from_secs(5)) {
        Ok(v) => {
            handle.join().unwrap();
            v
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            unblock();
            panic!("the system call blocked the host thread");
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            // `f` panicked: report its panic.
            std::panic::resume_unwind(handle.join().unwrap_err())
        }
    }
}

fn mkfifo(path: &str) -> bool {
    std::process::Command::new("mkfifo")
        .arg(path)
        .status()
        .is_ok_and(|s| s.success())
}

#[test]
fn special_files_are_refused_without_blocking() {
    let dir = TempDir::new("fifo");
    let fifo = dir.path("fifo");
    if !mkfifo(&fifo) {
        eprintln!("mkfifo unavailable; skipping");
        return;
    }
    let link = dir.path("link-to-fifo");
    std::os::unix::fs::symlink(&fifo, &link).unwrap();
    std::fs::write(dir.path("file"), b"regular").unwrap();
    let file_link = dir.path("link-to-file");
    std::os::unix::fs::symlink(dir.path("file"), &file_link).unwrap();

    let (fifo2, link2) = (fifo.clone(), link.clone());
    let release = fifo.clone();
    let results = bounded(
        move || {
            let mut m = machine(|fds| fds.filter_open = Some(Box::new(|_| true)));
            let p = guest_str(&mut m, &fifo2);
            [
                open(&mut m, &fifo2),
                open(&mut m, &link2),
                syscall(&mut m, SYS_FACCESSAT, &[AT_FDCWD, p, R_OK]),
            ]
        },
        // Open the write side, so a blocked reader gets released.
        move || {
            let _ = std::fs::OpenOptions::new().write(true).open(release);
        },
    );
    // The FIFO, and a link to it, are refused; faccessat doesn't probe it.
    assert_eq!(results, [-EACCES, -EACCES, -EACCES]);

    // A link to a regular file is followed.
    let mut m = machine(|fds| fds.filter_open = Some(Box::new(|_| true)));
    let fd = open(&mut m, &file_link);
    assert!(fd >= 0, "open link: {fd}");
    assert_eq!(read_all(&mut m, fd), b"regular");
    // Devices are special files too: refused unless permitted.
    assert_eq!(open(&mut m, "/dev/null"), -EACCES);
    let mut m = machine(|fds| {
        fds.filter_open = Some(Box::new(|_| true));
        fds.permit_special_files = true;
    });
    assert!(open(&mut m, "/dev/null") >= 0);
}
