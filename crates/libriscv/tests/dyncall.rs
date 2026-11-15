//! Dyncalls: port of `tests/unit/dyncall.cpp`, adapted to dyncalls being
//! system calls here. A valid dyncall word runs the system call numbered by
//! its immediate, in the interpreter and the JIT alike: the handler sees the
//! exact PC and instruction counter, and can stop or penalize the machine.
//! (In C++ a dyncall is a custom instruction, whose handler the decoder
//! hook picks, and only asmjit publishes the counter to it.)

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

#[cfg(feature = "jit")]
mod support;

use libriscv::registers::REG_ARG0;
use libriscv::{Dyncall, ExceptionKind, Instr, Machine, MachineError, MachineOptions};

const CALL: usize = 500;
const FP_CALL: usize = 501;

#[test]
fn counted_dyncall_fp_flag_encoding() {
    for funct3 in 0..8u32 {
        for rd in 0..32u32 {
            for rs1 in 0..32u32 {
                let word = (4095 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x5b;
                let legacy = funct3 == 0 && rd == 0 && rs1 == 0;
                let counted = funct3 == 7
                    && (16..=24).contains(&rd)
                    && ((24..=26).contains(&rs1) || (28..=30).contains(&rs1));
                assert_eq!(Dyncall::valid(word), legacy || counted, "{word:#x}");
                if counted {
                    assert_eq!(Dyncall::inputs(word), rd & 15);
                    assert_eq!(Dyncall::outputs(word), rs1 & 3);
                    assert_eq!(Dyncall::floats(word), rs1 & 4 != 0);
                    assert_eq!(Dyncall::encode(4095, rd & 15, rs1 & 3, rs1 & 4 != 0), word);
                }
            }
        }
    }
    assert!(Dyncall::floats(Dyncall::OPCODE));
    assert!(Dyncall::floats(Dyncall::encode(0, 1, 1, true)));
}

const REG_FA0: u32 = 10;

#[derive(Debug, Default)]
struct CallState {
    calls: u32,
    scenario: u32,
}

/// System call 500: a0 += 7. 501 also reads all of fa0-fa7 and writes
/// fa0-fa1.
///
/// Like the C++ test's decoder hook, the handler rejects a counted word
/// whose promise doesn't fit it: one input and one output, and FP
/// registers live for 501. The word is read from instruction memory at PC.
fn call(m: &mut Machine, fp: bool) -> libriscv::Result<()> {
    let word: u32 = m.memory.read_value(m.cpu.pc())?;
    let fits = !Dyncall::is_counted(word)
        || (Dyncall::inputs(word) == 1
            && Dyncall::outputs(word) == 1
            && (!fp || Dyncall::floats(word)));
    if !fits {
        return Err(MachineError::trigger(
            ExceptionKind::UnimplementedInstruction,
            u64::from(word),
        ));
    }
    let state = m.userdata_mut::<CallState>().unwrap();
    state.calls += 1;
    let scenario = state.scenario;
    let a0 = m.cpu.reg(REG_ARG0);
    m.cpu.set_reg(REG_ARG0, a0.wrapping_add(7));
    if fp {
        let regs = m.cpu.registers_mut();
        let sum: f64 = (REG_FA0..REG_FA0 + 8)
            .map(|r| f64::from_bits(regs.fp_bits(r)))
            .sum();
        regs.set_fp_bits(REG_FA0, (sum / 16.0).to_bits());
        regs.set_fp_bits(REG_FA0 + 1, (sum / 32.0).to_bits());
    }
    match scenario {
        1 => m.penalize(100_000),
        2 => m.stop(),
        3 => {
            return Err(MachineError::trigger(ExceptionKind::IllegalOperation, 0));
        }
        _ => {}
    }
    Ok(())
}

/// A custom-2 word that isn't a dyncall still goes to the decoder hook.
fn on_unimplemented(i: Instr) -> Option<libriscv::CustomInstruction> {
    (i.whole() == NOT_A_DYNCALL).then_some(not_a_dyncall)
}
const NOT_A_DYNCALL: u32 = 0b1000011011011;
fn not_a_dyncall(m: &mut Machine, _: Instr) -> libriscv::Result<()> {
    m.cpu.set_reg(REG_ARG0, 0x1234);
    Ok(())
}

/// The guest: dirty every FP argument and a non-argument. Each loop also
/// mixes in an FP-enabled handler, and a branch skips an integer call.
fn program(word: u32) -> Vec<u8> {
    let mut body = String::from("li t0, 7\n1:\naddi a0, a0, 1\n");
    for r in 10..18 {
        body += &format!("fadd.d f{r}, f{r}, f31\n");
    }
    body += &format!("fadd.d f2, f2, f31\n.word {word}\n");
    body += &format!(".word {}\n", Dyncall::encode(501, 1, 1, true));
    body += &format!("andi t1, t0, 1\nbeqz t1, 2f\n.word {word}\n2:\n");
    body += "addi t0, t0, -1\nbnez t0, 1b\n.word 0x7ff00073\n";
    let source = format!(
        "__attribute__((naked)) void _start(void) {{ __asm__ volatile(\"{}\"); }}",
        body.replace('\n', "\\n")
    );
    // The guest is hand-written assembly: rv64g, no compressed instructions.
    testutil::build_and_load(
        &source,
        "-nostdlib -static -march=rv64g -mabi=lp64d -Wl,-e,_start",
    )
    .unwrap()
}

/// The final state of one run.
#[derive(Debug, PartialEq)]
struct Outcome {
    gpr: [u64; 32],
    fp: [u64; 32],
    pc: u64,
    counter: u64,
    calls: u32,
    exception: Option<ExceptionKind>,
}

fn run(binary: &[u8], scenario: u32, translated: bool) -> (Machine, Outcome) {
    let options = MachineOptions {
        memory_max: 8 << 20,
        use_shared_execute_segments: false,
        jit: translated,
        ..Default::default()
    };
    let mut machine = Machine::new(binary, options).unwrap();
    if translated {
        assert!(machine.cpu.current_execute_segment().jit_entry_count() > 0);
    }
    machine.install_syscall_handler(CALL, |m| call(m, false));
    machine.install_syscall_handler(FP_CALL, |m| call(m, true));
    machine.set_userdata(CallState { calls: 0, scenario });
    for r in 0..32 {
        machine
            .cpu
            .registers_mut()
            .set_fp_bits(r, (f64::from(r) + 0.25).to_bits());
    }
    let exception = machine.simulate(10_000).err().map(|e| e.kind);
    let outcome = Outcome {
        gpr: std::array::from_fn(|r| machine.cpu.reg(r as u32)),
        fp: std::array::from_fn(|r| machine.cpu.registers().fp_bits(r as u32)),
        pc: machine.cpu.pc(),
        counter: machine.instruction_counter(),
        calls: machine.userdata::<CallState>().unwrap().calls,
        exception,
    };
    (machine, outcome)
}

/// C++ "Counted integer calls preserve live FP values", for one dyncall
/// word.
fn counted_calls(word: u32) {
    if !testutil::need_compiler() {
        return;
    }
    let binary = program(word);
    for scenario in 0..4 {
        let mut expected = None;
        for translated in [false, true] {
            if translated && !cfg!(feature = "jit") {
                continue;
            }
            let (machine, got) = run(&binary, scenario, translated);
            let ctx = format!("word {word:#x} scenario {scenario} translated {translated}");
            if word >> 20 == FP_CALL as u32 && !Dyncall::floats(word) {
                // The FP handler rejects the very first call, before it
                // changes anything
                assert_eq!(got.calls, 0, "{ctx}");
                let want = Some(ExceptionKind::UnimplementedInstruction);
                assert_eq!(got.exception, want, "{ctx}");
                assert_eq!(got.gpr[REG_ARG0 as usize], 1, "{ctx}");
                assert_eq!(got.pc, machine.address_of("_start").unwrap() + 44, "{ctx}");
            } else if scenario == 1 || scenario == 2 {
                // The first call stops the machine on the spot, or
                // penalizes it beyond the limit.
                assert_eq!(got.calls, 1, "{ctx}");
                let want = (scenario == 1).then_some(ExceptionKind::MaxInstructionsReached);
                assert_eq!(got.exception, want, "{ctx}");
                assert_eq!(got.gpr[REG_ARG0 as usize], 8, "{ctx}");
                for r in 0..32 {
                    let dirty = r == 2 || (10..18).contains(&r);
                    let want = f64::from(r) + 0.25 + if dirty { 31.25 } else { 0.0 };
                    assert_eq!(f64::from_bits(got.fp[r as usize]), want, "{ctx} f{r}");
                }
            }
            if scenario == 3 {
                // The faulting dyncall itself, in both
                assert_eq!(got.pc, machine.address_of("_start").unwrap() + 44, "{ctx}");
            }
            let Some(exp) = &expected else {
                expected = Some(got);
                continue;
            };
            // The interpreter and the JIT agree on everything, the
            // instruction count included
            assert_eq!(&got, exp, "{ctx}");
        }
    }
}

#[test]
#[cfg_attr(not(feature = "ext-f"), ignore = "the guest uses D")]
fn counted_integer_call_without_fp() {
    counted_calls(Dyncall::encode(500, 1, 1, false));
}

#[test]
#[cfg_attr(not(feature = "ext-f"), ignore = "the guest uses D")]
fn counted_call_with_fp() {
    counted_calls(Dyncall::encode(500, 1, 1, true));
}

/// An integer call to the FP handler breaks the word's promise, which the
/// handler checks.
#[test]
#[cfg_attr(not(feature = "ext-f"), ignore = "the guest uses D")]
fn integer_call_to_fp_handler_is_unimplemented() {
    counted_calls(Dyncall::encode(FP_CALL as u32, 1, 1, false));
}

#[test]
#[cfg_attr(not(feature = "ext-f"), ignore = "the guest uses D")]
fn legacy_call() {
    counted_calls((500 << 20) | Dyncall::OPCODE);
}

/// The index beyond the system call table is an unhandled system call, and
/// a custom-2 word that isn't a dyncall is still the decoder hook's.
#[test]
fn unknown_dyncall_index_and_other_custom_instructions() {
    if !testutil::need_compiler() {
        return;
    }
    let source = format!(
        "__attribute__((naked)) void _start(void) {{ __asm__ volatile(\".word {}\\n.word {}\\n.word 0x7ff00073\"); }}",
        Dyncall::encode(4000, 0, 0, false),
        NOT_A_DYNCALL
    );
    let binary = testutil::build_and_load(
        &source,
        "-nostdlib -static -march=rv64g -mabi=lp64d -Wl,-e,_start",
    )
    .unwrap();
    for jit in [false, cfg!(feature = "jit")] {
        let mut machine = Machine::new(
            &binary,
            MachineOptions {
                memory_max: 8 << 20,
                jit,
                ..Default::default()
            },
        )
        .unwrap();
        machine.set_on_unimplemented_instruction(on_unimplemented);
        machine.set_on_unhandled_syscall(|m, number| {
            assert_eq!(number, 4000);
            m.cpu.set_reg(REG_ARG0 + 1, number);
            Ok(())
        });
        machine.simulate(1000).unwrap();
        assert_eq!(machine.cpu.reg(REG_ARG0 + 1), 4000, "jit={jit}");
        assert_eq!(machine.cpu.reg(REG_ARG0), 0x1234, "jit={jit}");
    }
}

/// A counted call that stops the machine leaves it after the dyncall with
/// the exact count, so resuming finishes like an uninterrupted run.
#[test]
#[cfg(all(feature = "jit", feature = "ext-f"))]
fn stop_in_counted_call_resumes_exactly() {
    if !testutil::need_compiler() {
        return;
    }
    let binary = program(Dyncall::encode(500, 1, 1, true));
    let (_, uninterrupted) = run(&binary, 0, false);

    let (mut machine, stopped) = run(&binary, 2, true);
    assert_eq!(stopped.exception, None);
    assert_eq!(stopped.pc, machine.address_of("_start").unwrap() + 48);
    // li, addi, 9 x fadd.d and the dyncall.
    assert_eq!(stopped.counter, 12);
    let mut stops = 1;
    while machine.cpu.reg(5) != 0 || stops < 2 {
        machine.resume(10_000).unwrap();
        stops += 1;
        assert!(stops < 100);
    }
    // The remaining calls didn't stop (scenario 2 stops at every call, so
    // the loop above resumed once per call).
    assert_eq!(machine.instruction_counter(), uninterrupted.counter);
    assert_eq!(machine.cpu.pc(), uninterrupted.pc);
    for r in 0..32 {
        assert_eq!(machine.cpu.reg(r), uninterrupted.gpr[r as usize], "x{r}");
    }
}

/// Straight-line blocks longer than the decoder allows are forced to end,
/// and one that ends at a dyncall keeps the dyncall's bytecode there: sweep
/// a dyncall across that point. The handler returns, penalizes beyond the
/// limit, or stops; the interpreter and the JIT must agree exactly, and a
/// stop or penalty must halt right after the dyncall with its exact count.
#[test]
#[cfg(feature = "jit")]
fn dyncalls_around_a_forced_block_end() {
    use support::*;
    let addi = itype(0x13, 0, 10, 10, 1); // addi a0, a0, 1
    for number in [CALL, CALL + 2, CALL + 3] {
        for at in 110..140usize {
            let mut code = vec![addi; at];
            code.push(Dyncall::encode(number as u32, 1, 1, false));
            code.extend([addi; 40]);
            code.push(0x7ff0_0073); // stop
            let code = words(&code);
            let outcome = |jit: bool, fuel: u64| {
                let mut m = machine_sized(
                    &code,
                    1 << 20,
                    MachineOptions {
                        jit,
                        ..Default::default()
                    },
                );
                m.install_syscall_handler(CALL, |_| Ok(()));
                m.install_syscall_handler(CALL + 2, |m| {
                    m.penalize(100_000);
                    Ok(())
                });
                m.install_syscall_handler(CALL + 3, |m| {
                    m.stop();
                    Ok(())
                });
                run(&mut m, &[0; 64], fuel)
            };
            for fuel in [at as u64, at as u64 + 1, at as u64 + 2, 10_000] {
                let interp = outcome(false, fuel);
                let jit = outcome(true, fuel);
                let ctx = format!("number {number} at {at} fuel {fuel}");
                assert!(jit == interp, "{ctx}:\n{}", describe(&jit, &interp));
                if number != CALL && fuel == 10_000 {
                    assert_eq!(interp.pc, V + 4 * at as u64 + 4, "{ctx}");
                    assert_eq!(
                        interp.counter,
                        at as u64 + 1 + 100_000 * u64::from(number == CALL + 2),
                        "{ctx}"
                    );
                }
            }
        }
    }
}
