//! The cooperative thread scheduler's bookkeeping, driven by hand-assembled
//! guests: a thread is in at most one of the suspended and blocked lists,
//! and never while it runs.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::{ExceptionKind, Machine, MachineOptions};

const CODE: u64 = 0x2000;
const FUTEX: u64 = 0x10000;
const ECALL: u32 = 0x73;

const fn itype(imm: i32, rs1: u32, f3: u32, rd: u32, op: u32) -> u32 {
    ((imm as u32 & 0xfff) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op
}
const fn li(rd: u32, imm: i32) -> u32 {
    itype(imm, 0, 0, rd, 0x13)
}
const fn bnez(rs1: u32, off: i32) -> u32 {
    let o = off as u32;
    (((o >> 12) & 1) << 31)
        | (((o >> 5) & 0x3f) << 25)
        | (rs1 << 15)
        | (1 << 12)
        | (((o >> 1) & 0xf) << 8)
        | (((o >> 11) & 1) << 7)
        | 0x63
}
const fn jal_x0(off: i32) -> u32 {
    let o = off as u32;
    (((o >> 20) & 1) << 31)
        | (((o >> 1) & 0x3ff) << 21)
        | (((o >> 11) & 1) << 20)
        | (((o >> 12) & 0xff) << 12)
        | 0x6f
}

fn machine(program: &[u32]) -> Machine {
    let mut m = Machine::new([].into(), MachineOptions::default()).unwrap();
    m.setup_linux_syscalls(false, false);
    m.setup_posix_threads();
    m.copy_to_guest(FUTEX, &[0u8; 8]).unwrap();
    let code: Vec<u8> = program.iter().flat_map(|w| w.to_le_bytes()).collect();
    m.init_execute_area(&code, CODE).unwrap();
    m.cpu.jump(CODE).unwrap();
    m
}

/// Main clones a child and blocks on a futex while the child is
/// suspended; the child yields and exits. With nothing suspended, the
/// child's exit resumes the blocked main thread. 16 instructions a round.
const BLOCK_EXIT_LOOP: [u32; 16] = [
    li(17, 220),                     // 0: li a7, 220 (clone)
    li(10, 0),                       // 4: li a0, 0
    itype(0, 2, 0, 11, 0x13),        // 8: mv a1, sp
    ECALL,                           // 12
    bnez(10, 24),                    // 16: parent -> 40
    li(17, 124),                     // 20: child: sched_yield
    ECALL,                           // 24
    li(17, 93),                      // 28: exit
    ECALL,                           // 32
    jal_x0(0),                       // 36: (not reached)
    (0x10 << 12) | (10 << 7) | 0x37, // 40: lui a0, 0x10 (FUTEX)
    li(11, 0),                       // 44: FUTEX_WAIT
    li(12, 0),                       // 48: expected value
    li(17, 98),                      // 52: futex
    ECALL,                           // 56
    jal_x0(-60),                     // 60: -> 0
];

#[test]
fn resumed_main_thread_leaves_the_blocked_list() {
    let mut m = machine(&BLOCK_EXIT_LOOP);
    let mut max_queued = 0;
    for _ in 0..1000 {
        let err = m.resume(16).unwrap_err();
        assert_eq!(err.kind, ExceptionKind::MaxInstructionsReached, "{err}");
        let mt = m.threads().unwrap();
        max_queued = max_queued.max(mt.blocked_threads().len() + mt.suspended_threads().len());
        assert!(mt.len() <= 2);
    }
    // Without the fix, the main thread gained a stale blocked entry every
    // round (1000 here), which every exit and wakeup then scanned.
    assert!(max_queued <= 1, "{max_queued} queued threads");
    let mt = m.threads().unwrap();
    assert!(
        mt.blocked_threads().is_empty(),
        "{:?}",
        mt.blocked_threads()
    );
}

/// Main blocks while a child is suspended, the child exits (resuming the
/// blocked main), then main clones a second child that wakes the futex
/// main once waited on. The running main thread must not be woken: the
/// wakeup finds no waiter, and main is queued once, when it was suspended
/// by the clone.
#[test]
fn running_main_thread_is_not_woken() {
    let program = [
        li(17, 220),                     // 0: clone
        li(10, 0),                       // 4
        itype(0, 2, 0, 11, 0x13),        // 8: mv a1, sp
        ECALL,                           // 12
        bnez(10, 24),                    // 16: parent -> 40
        li(17, 124),                     // 20: child 1: sched_yield
        ECALL,                           // 24
        li(17, 93),                      // 28: exit
        ECALL,                           // 32
        jal_x0(0),                       // 36
        (0x10 << 12) | (10 << 7) | 0x37, // 40: lui a0, 0x10
        li(11, 0),                       // 44: FUTEX_WAIT
        li(12, 0),                       // 48
        li(17, 98),                      // 52
        ECALL,                           // 56: blocks, child 1 runs
        li(17, 220),                     // 60: main again: clone
        li(10, 0),                       // 64
        itype(0, 2, 0, 11, 0x13),        // 68
        ECALL,                           // 72
        bnez(10, 24),                    // 76: parent -> 100
        (0x10 << 12) | (10 << 7) | 0x37, // 80: child 2: lui a0, 0x10
        li(11, 1),                       // 84: FUTEX_WAKE
        li(12, -1),                      // 88: everyone
        li(17, 98),                      // 92
        ECALL,                           // 96: the result stays in a0
        jal_x0(0),                       // 100: both spin here
    ];
    let mut m = machine(&program);
    let err = m.simulate(200).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::MaxInstructionsReached, "{err}");
    let mt = m.threads().unwrap();
    assert_eq!(mt.len(), 2);
    assert_ne!(mt.get_tid(), 1, "child 2 runs");
    // FUTEX_WAKE found no waiter.
    assert_eq!(m.return_value(), 0);
    assert_eq!(mt.suspended_threads(), &[1]);
    assert!(mt.blocked_threads().is_empty());
}
