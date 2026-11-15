//! Harness for calls into the guest (`vmcall.rs`, `dyncall.rs`,
//! `sysargs.rs`): VM calls with host arguments, preemption from system
//! call handlers (nested), dyncalls (system calls), and system call argument
//! parsing (the Rust scripting API's readers included), all against
//! guest-chosen code and data.

use libriscv::guest::{GuestPtr, GuestRustStr, GuestRustString, GuestRustVec};
use libriscv::registers::REG_ARG0;
use libriscv::{
    Buffer, ByPointer, ExceptionKind, GuestSlice, Machine, MachineError, MachineOptions, OnStack,
};

use crate::elf::{self, CODE};
use crate::{MAX_CYCLES, quiet};

/// System calls the guest can make.
const SYS_PREEMPT: usize = 1;
const SYS_TRY_PREEMPT: usize = 2;
const SYS_ARGS: usize = 3;
/// 4-7: the dyncall behaviours of `sys_dyncall`.
const SYS_DYNCALL: usize = 4;
/// The argument readers of the Rust scripting API: borrowed strings and
/// slices, and guest pointers to Rust collections.
const SYS_RUST_ARGS: usize = 8;
const SYS_RUST_PTRS: usize = 9;
/// Preemptions nest (a preempted function may preempt again): each level
/// is a host stack frame, so the embedder has to bound the depth. Like an
/// embedder, the harness refuses beyond this.
const MAX_DEPTH: u32 = 8;

#[derive(Default)]
struct State {
    depth: u32,
}

fn enter(m: &mut Machine) -> bool {
    let Some(state) = m.userdata_mut::<State>() else {
        return false;
    };
    if state.depth >= MAX_DEPTH {
        return false;
    }
    state.depth += 1;
    true
}

fn leave(m: &mut Machine) {
    if let Some(state) = m.userdata_mut::<State>() {
        state.depth = state.depth.saturating_sub(1);
    }
}

/// `preempt(a0, a1, "string", 1.5)`, restoring the registers.
fn sys_preempt(m: &mut Machine) -> libriscv::Result<()> {
    let (func, arg): (u64, u64) = m.sysargs()?;
    if !enter(m) {
        return Ok(());
    }
    let r = m.preempt(MAX_CYCLES, func, (arg, "string", 1.5f64));
    leave(m);
    m.set_result(r?);
    Ok(())
}

/// `try_preempt` (errors discarded), clobbering the registers if a2 is 0.
fn sys_try_preempt(m: &mut Machine) -> libriscv::Result<()> {
    let (func, data, store): (u64, String, u64) = m.sysargs()?;
    if !enter(m) {
        return Ok(());
    }
    let r = m.try_preempt(
        MAX_CYCLES,
        func,
        (data.as_str(), OnStack([1u64, 2, 3])),
        store != 0,
    );
    leave(m);
    m.set_result(r?);
    Ok(())
}

/// The system call argument readers, on guest-chosen pointers.
fn sys_args(m: &mut Machine) -> libriscv::Result<()> {
    let n = {
        let (s, view, buf): (String, &[u8], Buffer<'_>) = m.sysargs()?;
        s.len() + view.len() + buf.len()
    };
    m.set_result(n as u64);
    Ok(())
}

/// The readers a Rust guest's host functions use, on guest-chosen pointers
/// and lengths: a UTF-8-checked `&str`, an unread `GuestRustStr` (then
/// read both ways) and a `GuestSlice<u32>`.
fn sys_rust_args(m: &mut Machine) -> libriscv::Result<()> {
    let n = {
        let (s, raw, words): (&str, GuestRustStr, GuestSlice<'_, u32>) = m.sysargs()?;
        let lossy = raw.to_string_lossy(m)?.len();
        let checked = raw.to_str(m).map_or(0, str::len);
        let sum = words.iter().fold(0u64, |a, w| a.wrapping_add(u64::from(w)));
        (s.len() + lossy + checked) as u64 ^ sum
    };
    m.set_result(n);
    Ok(())
}

/// A value read through a guest pointer, and guest pointers to a Rust
/// `String` and `Vec<u32>`, read and changed in place.
fn sys_rust_ptrs(m: &mut Machine) -> libriscv::Result<()> {
    let (ByPointer(pair), s, v): (
        ByPointer<[u64; 2]>,
        GuestPtr<GuestRustString>,
        GuestPtr<GuestRustVec<u32>>,
    ) = m.sysargs()?;
    let text = s.to_string_lossy(m)?;
    let words = v.to_vec(m)?;
    if pair[0] & 1 != 0 {
        s.push_str(m, "!")?;
        v.push(m, &(text.len() as u32))?;
    }
    m.set_result(pair[1] ^ (words.len() as u64));
    Ok(())
}

/// System calls 4-7, which dyncalls (system calls numbered by their
/// immediate) reach too: read the arguments as a string and a buffer, then
/// return, penalize, stop or fail.
fn sys_dyncall<const BEHAVIOUR: usize>(m: &mut Machine) -> libriscv::Result<()> {
    let n = m
        .sysargs::<(String, &[u8])>()
        .map_or(0, |(s, v)| s.len() + v.len());
    m.cpu.set_reg(REG_ARG0, n as u64);
    match BEHAVIOUR {
        0 => {}
        1 => m.penalize(1000),
        2 => m.stop(),
        _ => return Err(MachineError::trigger(ExceptionKind::IllegalOperation, 0)),
    }
    Ok(())
}

/// A machine running `code`, with the harness's system calls and dyncalls.
fn vmcall_machine(code: &[u8]) -> Option<Machine> {
    let binary = elf::executable(code, &[]);
    let options = MachineOptions {
        memory_max: 16 << 20,
        ..crate::options()
    };
    let mut m = Machine::new(&binary, options).ok()?;
    quiet(&mut m);
    m.set_userdata(State::default());
    m.setup_minimal_syscalls();
    m.install_syscall_handler(SYS_PREEMPT, sys_preempt);
    m.install_syscall_handler(SYS_TRY_PREEMPT, sys_try_preempt);
    m.install_syscall_handler(SYS_ARGS, sys_args);
    m.install_syscall_handler(SYS_RUST_ARGS, sys_rust_args);
    m.install_syscall_handler(SYS_RUST_PTRS, sys_rust_ptrs);
    m.install_syscall_handlers([
        (SYS_DYNCALL, sys_dyncall::<0> as libriscv::SyscallHandler),
        (SYS_DYNCALL + 1, sys_dyncall::<1>),
        (SYS_DYNCALL + 2, sys_dyncall::<2>),
        (SYS_DYNCALL + 3, sys_dyncall::<3>),
    ]);
    m.setup_argv(&["program"], &["LC_ALL=C"]).ok()?;
    Some(m)
}

/// `data[2..]` runs as a program that can preempt itself through system
/// calls, use dyncalls, and pass pointers to the argument readers. Then
/// the host makes VM calls to `CODE + 4 * data[0]` with the argument set
/// `data[1]` picks.
pub fn fuzz_vmcall(data: &[u8]) {
    let Some(([entry, args], code)) = data.split_first_chunk::<2>().map(|(a, b)| (*a, b)) else {
        return;
    };
    let Some(mut m) = vmcall_machine(code) else {
        return;
    };
    let _ = m.simulate(MAX_CYCLES);

    let func = CODE + 4 * u64::from(entry);
    let _ = match args % 6 {
        0 => m.vmcall_limited(MAX_CYCLES, func, ()),
        1 => m.vmcall_limited(MAX_CYCLES, func, (1u64, -2i32, 3.0f32, 4.0f64)),
        2 => m.vmcall_limited(MAX_CYCLES, func, ("a string", String::from("another"))),
        3 => m.vmcall_limited(MAX_CYCLES, func, (OnStack([7u64; 9]), vec![1u8, 2, 3])),
        // More arguments than registers: the rest go onto the stack.
        4 => m.vmcall_limited(MAX_CYCLES, func, (1, 2, 3, 4, 5, 6, 7, 8, 9u64, 10.0, "x")),
        _ => m.try_vmcall(MAX_CYCLES, func, (func, "try")),
    };
    // From the host, between calls.
    let _ = m.preempt(MAX_CYCLES, func, (u64::from(args),));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn vmcalls_and_preemption_reach_the_guest() {
        let code = words(&[
            // 0: a0 + 5
            addi(10, 10, 5),
            RET,
            // 8: preempt(CODE, 37)
            lui(10, CODE),
            addi(11, 0, 37),
            addi(17, 0, SYS_PREEMPT as i32),
            ECALL,
            RET,
        ]);
        let mut m = vmcall_machine(&code).unwrap();
        assert_eq!(m.vmcall_limited(1000, CODE, (37u64,)).unwrap(), 42);
        assert_eq!(m.vmcall_limited(1000, CODE + 8, ()).unwrap(), 42);
    }

    /// The Rust scripting API's readers, on arguments set up by hand in
    /// the data segment.
    #[test]
    fn the_rust_argument_readers_read_guest_memory() {
        use libriscv::registers::{REG_ARG1, REG_ARG2, REG_ARG3, REG_ARG4, REG_ARG5};
        let mut m = vmcall_machine(&words(&[RET])).unwrap();
        let data = crate::elf::DATA;
        m.copy_to_guest(data, b"hi").unwrap();
        m.copy_to_guest(data + 0x10, &5u32.to_le_bytes()).unwrap();
        // ("hi", GuestRustStr "hi", [5u32])
        for (reg, value) in [
            (REG_ARG0, data),
            (REG_ARG1, 2),
            (REG_ARG2, data),
            (REG_ARG3, 2),
            (REG_ARG4, data + 0x10),
            (REG_ARG5, 1),
        ] {
            m.cpu.set_reg(reg, value);
        }
        m.system_call(SYS_RUST_ARGS as u64).unwrap();
        assert_eq!(m.cpu.reg(REG_ARG0), (2 + 2 + 2) ^ 5);
        // ByPointer([0, 9]) at data + 0x20, a String "hi" at data + 0x40
        // ({ cap, ptr, len }) and an empty Vec<u32> at data + 0x60
        let words64 = |m: &mut Machine, at: u64, w: &[u64]| {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            m.copy_to_guest(at, &bytes).unwrap();
        };
        words64(&mut m, data + 0x20, &[0, 9]);
        words64(&mut m, data + 0x40, &[2, data, 2]);
        words64(&mut m, data + 0x60, &[0, 4, 0]);
        for (reg, value) in [
            (REG_ARG0, data + 0x20),
            (REG_ARG1, data + 0x40),
            (REG_ARG2, data + 0x60),
        ] {
            m.cpu.set_reg(reg, value);
        }
        m.system_call(SYS_RUST_PTRS as u64).unwrap();
        assert_eq!(m.cpu.reg(REG_ARG0), 9);
    }
}
