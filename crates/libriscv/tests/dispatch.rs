//! Dispatcher behaviour that the difftests cannot reach through `rvlinux`:
//! system call handlers that change PC or stop the machine, and agreement
//! between the block-counting dispatcher and the precise interpreter.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::registers::*;
use libriscv::{ExceptionKind, Machine, MachineOptions, SYSCALL_EBREAK_NR};

const V: u64 = 0x2000;

/// li a7, n
const fn li_a7(n: u32) -> u32 {
    (n << 20) | (REG_ARG7 << 7) | 0x13
}
/// addi a0, a0, n
const fn addi_a0(n: u32) -> u32 {
    (n << 20) | (REG_ARG0 << 15) | (REG_ARG0 << 7) | 0x13
}
const ECALL: u32 = 0x0000_0073;
const EBREAK: u32 = 0x0010_0073;
const NOP: u32 = 0x0000_0013;
/// `j .` (spin)
const JSELF: u32 = 0x0000_006F;

fn machine_with(program: &[u32]) -> Machine {
    let mut m = Machine::new([].into(), MachineOptions::default()).unwrap();
    let code: Vec<u8> = program.iter().flat_map(|w| w.to_le_bytes()).collect();
    m.init_execute_area(&code, V).unwrap();
    m.cpu.jump(V).unwrap();
    m
}

#[test]
fn syscall_that_changes_pc_resumes_after_it() {
    // The handler sets PC to the second ECALL: execution resumes at PC + 4,
    // skipping both the instruction after the first ECALL and the second
    // ECALL itself.
    let program = [
        li_a7(1),    // 0x2000
        ECALL,       // 0x2004
        addi_a0(1),  // 0x2008: skipped
        ECALL,       // 0x200C: the new PC
        addi_a0(10), // 0x2010: resumes here
        li_a7(2),
        ECALL,
        JSELF,
    ];
    let mut m = machine_with(&program);
    m.install_syscall_handler(1, |m| {
        m.cpu.aligned_jump(V + 0xC);
        Ok(())
    });
    m.install_syscall_handler(2, |m| {
        m.stop();
        Ok(())
    });
    m.simulate(100).unwrap();
    assert_eq!(m.return_value(), 10);
    assert_eq!(m.cpu.pc(), V + 0x1C);
    // li, ecall | addi, li, ecall
    assert_eq!(m.instruction_counter(), 5);
}

#[test]
fn stopping_from_ebreak_stays_on_the_ebreak() {
    // Unlike ECALL, a SYSTEM instruction that stops the machine leaves PC on
    // itself (C++ cpu_dispatch.cpp), so resuming would execute it again.
    let program = [NOP, EBREAK, addi_a0(1), JSELF];
    let mut m = machine_with(&program);
    m.install_syscall_handler(SYSCALL_EBREAK_NR, |m| {
        m.set_result(m.return_value() + 100);
        m.stop();
        Ok(())
    });
    m.simulate(100).unwrap();
    assert_eq!(m.cpu.pc(), V + 4);
    assert_eq!(m.return_value(), 100);
    assert_eq!(m.instruction_counter(), 2);
}

#[test]
fn syscall_errors_propagate_with_the_counter_applied() {
    let program = [NOP, NOP, li_a7(5), ECALL, JSELF];
    let mut m = machine_with(&program);
    m.install_syscall_handler(5, |_| {
        Err(libriscv::MachineError::new(
            ExceptionKind::SystemCallFailed,
            "boom",
            5,
        ))
    });
    let err = m.simulate(100).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::SystemCallFailed);
    assert_eq!(m.cpu.pc(), V + 12);
    assert_eq!(m.instruction_counter(), 4);
}

#[test]
fn unhandled_syscalls_go_to_the_callback() {
    let program = [
        li_a7(300),
        ECALL,
        li_a7(2000),
        ECALL,
        li_a7(2),
        ECALL,
        JSELF,
    ];
    let mut m = machine_with(&program);
    m.set_on_unhandled_syscall(|m, n| {
        m.set_result(m.return_value() + n);
        Ok(())
    });
    m.install_syscall_handler(2, |m| {
        m.stop();
        Ok(())
    });
    m.simulate(100).unwrap();
    assert_eq!(m.return_value(), 2300);
}

#[test]
fn resume_continues_counting() {
    let mut m = machine_with(&[JSELF]);
    assert!(m.simulate(1000).unwrap_err().is_timeout());
    assert_eq!(m.instruction_counter(), 1000);
    assert!(m.resume(500).unwrap_err().is_timeout());
    assert_eq!(m.instruction_counter(), 1500);
    assert!(!m.try_simulate(10, 0).unwrap());
    assert!(m.instruction_limit_reached());
}

#[test]
fn code_outside_any_segment_faults() {
    let mut m = machine_with(&[NOP, JSELF]);
    m.cpu.jump(0x10_0000).unwrap();
    let err = m.simulate(100).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::ExecutionSpaceProtectionFault);
    assert_eq!(err.data, 0x10_0000);
}

/// The dispatcher counts whole blocks on entry; the precise interpreter
/// counts every instruction. For a program that runs to completion, the
/// totals and results must agree.
#[test]
fn dispatcher_and_precise_interpreter_agree() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = testutil::build_and_load(
        r#"
        static long sys(long n, long a) {
            register long a0 asm("a0") = a; register long a7 asm("a7") = n;
            asm volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
            return a0;
        }
        static unsigned long collatz(unsigned long n) {
            unsigned long steps = 0;
            while (n != 1) { n = (n & 1) ? 3 * n + 1 : n / 2; steps++; }
            return steps;
        }
        void _start(void) {
            unsigned long total = 0;
            for (unsigned long i = 1; i < 3000; i++) total += collatz(i) * (i % 7);
            sys(93, (long)(total % 100003));
        }"#,
        "-march=rv64im -mabi=lp64 -static -nostdlib -ffreestanding -O2",
    )
    .unwrap_or_else(|e| panic!("{e}"));

    let mut fast = Machine::new(binary.as_slice().into(), MachineOptions::default()).unwrap();
    fast.setup_minimal_syscalls();
    fast.simulate(100_000_000).unwrap();

    let mut precise = Machine::new(binary.as_slice().into(), MachineOptions::default()).unwrap();
    precise.setup_minimal_syscalls();
    precise.set_max_instructions(100_000_000);
    precise.simulate_precise().unwrap();

    assert_eq!(fast.return_value(), precise.return_value());
    assert_eq!(fast.instruction_counter(), precise.instruction_counter());
    assert_eq!(fast.cpu.registers().all(), precise.cpu.registers().all());
    assert!(fast.instruction_counter() > 100_000);
}

/// Random programs must not panic either interpreter. Uses deterministic
/// inputs to complement the fuzzing harness.
#[test]
fn random_code_never_panics() {
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 32) as u32
    };
    for round in 0..400 {
        let words = 4 + (next() % 128) as usize;
        let mut code: Vec<u8> = (0..words).flat_map(|_| next().to_le_bytes()).collect();
        // Bias towards real opcodes some of the time.
        if round % 2 == 0 {
            for chunk in code.chunks_mut(4) {
                chunk[0] |= 0x3;
            }
        }
        let align = if libriscv::COMPRESSED_ENABLED { 2 } else { 4 };
        let base = 0x2000 + u64::from(next() % 64) * align;
        let mut m = Machine::new([].into(), MachineOptions::default()).unwrap();
        m.set_on_unhandled_syscall(|_, _| Ok(()));
        if m.init_execute_area(&code, base).is_err() {
            continue;
        }
        for (i, r) in [REG_SP, REG_ARG0, REG_ARG1, REG_T0].into_iter().enumerate() {
            m.cpu.set_reg(r, u64::from(next()) << (i * 8));
        }
        m.cpu.jump(base).unwrap();
        let _ = m.simulate(10_000);
        m.cpu.jump(base).unwrap();
        m.set_instruction_counter(0);
        m.set_max_instructions(10_000);
        let _ = m.simulate_precise();
    }
}

/// Instructions preceding a breakpoint or libc fast-path patch must count
/// toward the limit. A zero count with the C extension allowed fall-through
/// loops to bypass the limit. Regression found by `libcfuzzer`.
#[test]
#[cfg(feature = "ext-c")]
fn instructions_before_a_breakpoint_are_counted() {
    // V: c.addi a0, 1; c.addi a0, 1; V+4: breakpoint (c.nop); c.j back.
    let code: Vec<u8> = [0x0505u16, 0x0505, 0x0001, 0xbff5]
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect();
    let options = MachineOptions {
        memory_max: 1 << 20,
        ebreak_locations: vec![libriscv::EbreakLocation::Address(V + 4)],
        ..Default::default()
    };
    let mut m = Machine::new([].into(), options).unwrap();
    m.init_execute_area(&code, V).unwrap();
    // The breakpoint loops back to V, as a fast-path function returning to
    // its caller does.
    m.install_syscall_handler(SYSCALL_EBREAK_NR, |m| m.cpu.jump(V));
    m.cpu.jump(V).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = m.simulate(3000);
        let _ = tx.send((result.map_err(|e| e.kind), m.cpu.reg(REG_ARG0)));
    });
    let (result, a0) = rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("the loop ran past the instruction limit");
    assert_eq!(result, Err(ExceptionKind::MaxInstructionsReached));
    // Three instructions per round (the two before it and the breakpoint),
    // as without the C extension.
    assert_eq!(a0, 2 * 1000);
}

/// A breakpoint or libc patch after a 4-byte instruction must end the block.
/// The backward scan must skip the instruction's empty second decoder slot;
/// stopping there leaves the patch mid-block and publishes the wrong PC.
/// Regression found while investigating `libcfuzzer`.
#[test]
#[cfg(feature = "ext-c")]
fn a_breakpoint_after_a_4_byte_instruction_ends_its_block() {
    // V: addi a0, a0, 1; V+4: c.addi a0, 1; V+6: breakpoint (c.nop);
    // V+8: c.j back.
    let mut code = 0x0015_0513u32.to_le_bytes().to_vec();
    for half in [0x0505u16, 0x0001, 0xbfe5] {
        code.extend(half.to_le_bytes());
    }
    let options = MachineOptions {
        memory_max: 1 << 20,
        ebreak_locations: vec![libriscv::EbreakLocation::Address(V + 6)],
        ..Default::default()
    };
    let mut m = Machine::new([].into(), options).unwrap();
    m.init_execute_area(&code, V).unwrap();
    m.set_userdata(std::sync::atomic::AtomicU64::new(0));
    m.install_syscall_handler(SYSCALL_EBREAK_NR, |m| {
        let pc = m.cpu.pc();
        if let Some(seen) = m.userdata::<std::sync::atomic::AtomicU64>() {
            seen.store(pc, std::sync::atomic::Ordering::Relaxed);
        }
        m.cpu.jump(V)
    });
    m.cpu.jump(V).unwrap();
    let err = m.simulate(3000).unwrap_err();
    assert_eq!(err.kind, ExceptionKind::MaxInstructionsReached);
    let seen = m.userdata::<std::sync::atomic::AtomicU64>().unwrap();
    assert_eq!(seen.load(std::sync::atomic::Ordering::Relaxed), V + 6);
    // Three instructions per round, the breakpoint included.
    assert_eq!(m.cpu.reg(REG_ARG0), 2 * 1000);
}

/// Port of C++ "A breakpoint on the second half of a fused pair": the
/// decoder fuses ADDI + ADDI, and the first entry then runs the second
/// instruction too, without dispatching it. A breakpoint installed on the
/// second instruction must unfuse the pair, or it is never hit.
#[test]
fn a_breakpoint_on_the_second_half_of_a_fused_pair() {
    let program: Vec<u8> = [NOP, addi_a0(1), addi_a0(2), addi_a0(4), JSELF]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let options = MachineOptions {
        memory_max: 1 << 20,
        ebreak_locations: vec![libriscv::EbreakLocation::Address(V + 8)],
        ..Default::default()
    };
    let mut m = Machine::new([].into(), options).unwrap();
    m.init_execute_area(&program, V).unwrap();
    m.install_syscall_handler(SYSCALL_EBREAK_NR, |m| {
        m.stop();
        Ok(())
    });
    m.cpu.jump(V).unwrap();
    m.simulate(100).unwrap();
    // Stopped in the breakpoint: only the instructions before it have run.
    assert_eq!(m.cpu.pc(), V + 8);
    assert_eq!(m.cpu.reg(REG_ARG0), 1);
    assert_eq!(m.instruction_counter(), 3);
}

/// Host stack depths seen by the system call handler [`record_depth`].
#[derive(Default)]
struct Depth {
    first: usize,
    deepest: usize,
    calls: u32,
}

/// System call 500: record how deep the host stack is.
fn record_depth(m: &mut Machine) -> libriscv::Result<()> {
    let local = 0u8;
    let here = std::hint::black_box(&local) as *const u8 as usize;
    let d = m.userdata_mut::<Depth>().unwrap();
    if d.calls == 0 {
        d.first = here;
        d.deepest = here;
    }
    d.deepest = d.deepest.min(here);
    d.calls += 1;
    Ok(())
}

/// A loop through the handlers that call host code (ECALL, a CSR read, an
/// atomic through its precise handler) and a branch, `n` times, then stop.
/// Each iteration records the host stack depth. The atomic is only included
/// with `ext-a`; without it a `nop` keeps the loop length (and the branch
/// offset) the same.
fn depth_program(n: u32) -> Vec<u32> {
    const RDCYCLE_T0: u32 = 0xC000_22F3; // csrrs t0, cycle, x0
    const LUI_T2_0X100: u32 = (0x100 << 12) | (7 << 7) | 0x37; // lui t2, 0x100
    #[cfg(feature = "ext-a")]
    const AMOADD_W: u32 = (6 << 20) | (7 << 15) | (2 << 12) | 0x2F; // amoadd.w zero, t1, (t2)
    #[cfg(not(feature = "ext-a"))]
    const AMOADD_W: u32 = 0x13; // nop: no atomics without ext-a
    const BNEZ_A0_BACK: u32 = {
        // bnez a0, -16 (back to the ECALL)
        let o = -16i32 as u32;
        (((o >> 12) & 1) << 31)
            | (((o >> 5) & 0x3F) << 25)
            | (REG_ARG0 << 15)
            | (1 << 12)
            | (((o >> 1) & 0xF) << 8)
            | (((o >> 11) & 1) << 7)
            | 0x63
    };
    vec![
        (n << 20) | (REG_ARG0 << 7) | 0x13, // li a0, n
        li_a7(500),
        LUI_T2_0X100,
        ECALL, // loop:
        RDCYCLE_T0,
        AMOADD_W,
        addi_a0(-1i32 as u32 & 0xFFF),
        BNEZ_A0_BACK,
        li_a7(2),
        ECALL,
        JSELF,
    ]
}

/// The growth of the host stack over `n` iterations of [`depth_program`].
fn stack_growth(n: u32, options: MachineOptions) -> usize {
    let mut m = Machine::new([].into(), options).unwrap();
    let code: Vec<u8> = depth_program(n)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    m.init_execute_area(&code, V).unwrap();
    m.cpu.jump(V).unwrap();
    m.set_userdata(Depth::default());
    m.install_syscall_handler(500, record_depth);
    m.install_syscall_handler(2, |m| {
        m.stop();
        Ok(())
    });
    m.simulate(1_000_000).unwrap();
    let d = m.userdata::<Depth>().unwrap();
    assert_eq!(d.calls, n);
    d.first - d.deepest
}

/// True if handler chains must not grow the host stack at all: tail calls
/// are guaranteed with `nightly-dispatch`, and otherwise optimized builds
/// turn them into jumps (unoptimized builds rely on `dispatch_slice`).
const fn chains_are_flat() -> bool {
    cfg!(feature = "nightly-dispatch") || !cfg!(debug_assertions)
}

/// Handler chains do not grow the host stack: each handler tail-calls the
/// next, including those that run host code first. Without
/// `nightly-dispatch` that holds only where LLVM turns the calls into jumps
/// (optimized builds; `cargo test --release` checks it). Unoptimized,
/// `dispatch_slice` must keep the stack well within a test thread's 2 MiB.
#[test]
fn handler_chains_do_not_grow_the_stack() {
    let growth = stack_growth(2000, MachineOptions::default());
    if chains_are_flat() {
        assert!(growth < 1024, "the host stack grew by {growth} bytes");
    } else {
        assert!(growth < 1 << 20, "the host stack grew by {growth} bytes");
    }
}

/// [`handler_chains_do_not_grow_the_stack`] in and out of native code.
#[cfg(feature = "jit")]
#[test]
fn native_code_chains_do_not_grow_the_stack() {
    let options = MachineOptions {
        jit: true,
        jit_background: false,
        ..Default::default()
    };
    let growth = stack_growth(2000, options);
    if chains_are_flat() {
        assert!(growth < 1024, "the host stack grew by {growth} bytes");
    } else {
        assert!(growth < 1 << 20, "the host stack grew by {growth} bytes");
    }
}

/// Code that differs between the dispatch modes stays in `dispatch/abi.rs`:
/// the rest of the crate must build on stable Rust, and handlers are written
/// once for both modes. (The AOT emitter writes `become` into generated code,
/// and `tail_call_tests.rs` tests it with `nightly-dispatch`.)
#[test]
fn nightly_dispatch_syntax_stays_in_abi() {
    fn visit(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    visit(&src, &mut files);
    let allowed = ["dispatch/abi.rs", "aot/emit.rs", "tail_call_tests.rs"];
    let mut found = Vec::new();
    for file in files {
        let rel = file
            .strip_prefix(&src)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        if allowed.contains(&rel.as_str()) {
            continue;
        }
        for (n, line) in std::fs::read_to_string(&file).unwrap().lines().enumerate() {
            let code = line.split("//").next().unwrap();
            if ["become ", "rust-preserve-none", "rustc_align"]
                .iter()
                .any(|s| code.contains(s))
            {
                found.push(format!("{rel}:{}: {}", n + 1, line.trim()));
            }
        }
    }
    assert!(
        found.is_empty(),
        "outside dispatch/abi.rs:\n{}",
        found.join("\n")
    );
}
