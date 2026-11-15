//! Per-thread signal state lives only as long as its thread: a guest that
//! keeps creating threads that touch it mustn't grow host memory.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::{ExceptionKind, Machine, MachineOptions};

const CODE: u64 = 0x2000;
const STACK_T: u64 = 0x10000;
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

/// Main keeps cloning; each child calls sigaltstack and exits, which
/// resumes main, about a dozen instructions a round.
fn run(set_altstack: bool, instructions: u64) -> Machine {
    let ss = if set_altstack {
        (0x10 << 12) | (10 << 7) | 0x37 // lui a0, 0x10 (STACK_T)
    } else {
        li(10, 0) // NULL: only query
    };
    let program = [
        li(17, 220),              // 0: clone
        li(10, 0),                // 4
        itype(0, 2, 0, 11, 0x13), // 8: mv a1, sp
        ECALL,                    // 12
        bnez(10, -16),            // 16: parent: clone again
        li(17, 132),              // 20: child: sigaltstack(ss, NULL)
        ss,                       // 24
        li(11, 0),                // 28
        ECALL,                    // 32
        li(17, 93),               // 36: exit
        ECALL,                    // 40
        0x6f,                     // 44: (not reached)
    ];
    let mut m = Machine::new(&[], MachineOptions::default()).unwrap();
    m.setup_linux_syscalls(false, false);
    m.setup_posix_threads();
    let mut stack_t = [0u8; 24];
    stack_t[..8].copy_from_slice(&0x20000u64.to_le_bytes());
    stack_t[16..].copy_from_slice(&0x4000u64.to_le_bytes());
    m.copy_to_guest(STACK_T, &stack_t).unwrap();
    let code: Vec<u8> = program.iter().flat_map(|w| w.to_le_bytes()).collect();
    m.init_execute_area(&code, CODE).unwrap();
    m.cpu.jump(CODE).unwrap();
    let err = m.simulate(instructions).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::MaxInstructionsReached, "{err}");
    assert!(m.threads().unwrap().len() <= 2);
    m
}

#[test]
fn querying_the_altstack_creates_no_state() {
    let mut m = run(false, 12_000);
    // Without the fix, every child left an entry behind (over 900).
    assert_eq!(m.signals().threads_with_state(), 0);
}

#[test]
fn exited_threads_leave_no_state_behind() {
    // Stop at every point of a round: at most the running child has state
    // (without the fix, every child that ever ran: over 900).
    let mut child_with_altstack = false;
    for extra in 0..16 {
        let mut m = run(true, 12_000 + extra);
        let n = m.signals().threads_with_state();
        assert!(n <= 1, "{n} threads with signal state");
        let tid = m.gettid();
        if let Some(pt) = m.signals().thread_state(tid) {
            assert_ne!(tid, 1);
            assert_eq!(pt.stack.ss_sp, 0x20000);
            assert_eq!(pt.stack.ss_size, 0x4000);
            child_with_altstack = true;
        }
    }
    assert!(child_with_altstack);
}
