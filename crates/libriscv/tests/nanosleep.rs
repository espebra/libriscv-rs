//! What nanosleep and clock_nanosleep charge: one instruction per
//! microsecond requested, saturating, so a huge request can't be free while
//! the host still sleeps. A suspended second thread keeps each host sleep
//! at 1 ms.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::registers::REG_ARG0;
use libriscv::{Machine, MachineOptions};

const TS: u64 = 0x10000;
const NANOSLEEP: u64 = 101;
const CLOCK_NANOSLEEP: u64 = 115;
/// Outside proxy mode, the low nanosecond bits are masked off.
const MASK_NANOS: i64 = !0x3FFF;

/// A machine with a second thread, so that sleeps are short.
fn machine() -> Machine {
    let mut m = Machine::new([].into(), MachineOptions::default()).unwrap();
    m.setup_linux_syscalls(false, false);
    m.setup_posix_threads();
    // clone: the main thread is suspended and the child runs.
    syscall(&mut m, 220, &[0, 0x8000]);
    assert_eq!(m.threads().unwrap().suspended_threads().len(), 1);
    m
}

fn syscall(m: &mut Machine, number: u64, args: &[u64]) -> i64 {
    for (i, &a) in args.iter().enumerate() {
        m.cpu.set_reg(REG_ARG0 + i as u32, a);
    }
    m.system_call(number).unwrap();
    m.return_value() as i64
}

/// Sleep for `(sec, nsec)` with system call `number`; returns the charge.
fn charge(m: &mut Machine, number: u64, sec: i64, nsec: i64) -> u64 {
    let mut ts = [0u8; 16];
    ts[..8].copy_from_slice(&sec.to_le_bytes());
    ts[8..].copy_from_slice(&nsec.to_le_bytes());
    m.copy_to_guest(TS, &ts).unwrap();
    let before = m.instruction_counter();
    let t0 = std::time::Instant::now();
    let tid = m.gettid();
    match number {
        NANOSLEEP => syscall(m, number, &[TS, 0]),
        _ => syscall(m, number, &[0, 0, TS, 0]),
    };
    // The other thread runs now; the sleeper resumes with 0.
    assert_ne!(m.gettid(), tid);
    let sleeper = m.threads().unwrap().get_thread(tid).unwrap();
    assert_eq!(sleeper.stored_regs.get(REG_ARG0), 0);
    assert!(t0.elapsed().as_millis() < 500, "{:?}", t0.elapsed());
    m.instruction_counter() - before
}

fn written_back(m: &Machine) -> (i64, i64) {
    let mut ts = [0u8; 16];
    m.copy_from_guest(&mut ts, TS).unwrap();
    (
        i64::from_le_bytes(ts[..8].try_into().unwrap()),
        i64::from_le_bytes(ts[8..].try_into().unwrap()),
    )
}

#[test]
fn normal_requests_charge_one_instruction_per_microsecond() {
    for number in [NANOSLEEP, CLOCK_NANOSLEEP] {
        let mut m = machine();
        for (sec, nsec) in [
            (0, 0),
            (0, 16384),
            (0, 999_424),
            (0, 999_999_999),
            (2, 0),
            (3, 500_000_000),
            (86_400, 1),
            (9_000_000_000, 0),
        ] {
            let expected = (sec * 1_000_000_000 + (nsec & MASK_NANOS)) / 1000;
            assert_eq!(
                charge(&mut m, number, sec, nsec),
                expected as u64,
                "{sec} {nsec}"
            );
        }
    }
}

#[test]
fn nanosleep_writes_the_clamped_request_back() {
    // Alone, the thread really sleeps: the whole duration is clamped to a
    // second, but the request written back is C++'s, seconds clamped.
    let mut m = Machine::new([].into(), MachineOptions::default()).unwrap();
    m.setup_linux_syscalls(false, false);
    let nsec = 500_000_000i64;
    let mut ts = [0u8; 16];
    ts[..8].copy_from_slice(&(1i64 << 55).to_le_bytes());
    ts[8..].copy_from_slice(&nsec.to_le_bytes());
    m.copy_to_guest(TS, &ts).unwrap();
    let t0 = std::time::Instant::now();
    assert_eq!(syscall(&mut m, NANOSLEEP, &[TS, 0]), 0);
    assert!(t0.elapsed().as_millis() >= 900, "{:?}", t0.elapsed());
    assert_eq!(m.instruction_counter(), u64::MAX / 1000);
    assert_eq!(written_back(&m), (1, nsec & MASK_NANOS));
    // Sleeping again on the written-back request charges what it says.
    m.set_instruction_counter(0);
    assert_eq!(syscall(&mut m, NANOSLEEP, &[TS, 0]), 0);
    assert_eq!(
        m.instruction_counter(),
        (1_000_000_000 + (nsec & MASK_NANOS)) as u64 / 1000
    );
}

#[test]
fn huge_requests_charge_the_saturated_duration() {
    // 2^55 seconds is 0 nanoseconds modulo 2^64: it used to cost nothing,
    // while the host slept for a second.
    let saturated = u64::MAX / 1000;
    for number in [NANOSLEEP, CLOCK_NANOSLEEP] {
        let mut m = machine();
        for (sec, nsec) in [(1 << 55, 0), (1 << 55, 999_999_999), (i64::MAX, 0)] {
            assert_eq!(charge(&mut m, number, sec, nsec), saturated, "{sec} {nsec}");
        }
    }
}
