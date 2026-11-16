//! Port of `tailcall_dispatch.cpp` and the slow-path and system handlers in
//! `bytecode_impl.cpp`. Fast bytecodes are defined in `bytecode.rs`.
//!
//! Each decoder entry stores a [`Handler`] function pointer. Handlers execute
//! one instruction and tail-call the next handler, providing one indirect
//! branch per instruction without stack growth. A chain returns an [`Exit`]
//! on timeout, stop, segment change, fault, or the end of a slice.
//!
//! Handlers are written once for both dispatch modes (see [`abi`]): with
//! `nightly-dispatch` they use `become` and `preserve_none`; otherwise
//! ordinary calls in tail position, which optimized builds turn into jumps,
//! with chains bounded by `MachineOptions::dispatch_slice`. Handlers receive
//! instruction bits as an argument. Exits are tail calls too, allowing
//! handlers without other calls to omit stack frames.
//!
//! On block entry, `BEGIN_BLOCK` adds the block's instruction count. Only
//! block-ending instructions use the PC (`Frame::end_pc`).
//!
//! Observable behavior follows `cpu_dispatch.cpp`, the default C++ reference:
//! illegal-opcode faults publish the counter; system calls that stop or change
//! PC resume after PC; other SYSTEM instructions retrieve both counters.

pub(crate) mod abi;

use std::sync::Arc;

pub(crate) use abi::Handler;
use abi::{
    begin_block, cold_handler, entry, execute, handler, next_block, overflow_checked_jump, resume,
    tail, unchecked_jump,
};

use crate::decode;
use crate::decoder_cache::{COMPRESSED, DecodedSegment};
use crate::error::{ExceptionKind, MachineError, Result};
use crate::instruction::Instr;
use crate::isa::Op;
use crate::machine::Machine;
use crate::registers::REG_ECALL;

/// The instruction counter and limit, kept in registers while dispatching
/// (C++ `InstrCounter`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Counter {
    pub value: u64,
    pub max: u64,
}

impl Counter {
    #[inline(always)]
    pub(crate) const fn overflowed(self) -> bool {
        self.value >= self.max
    }
}

/// Why a handler chain returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exit {
    /// The counter reached its limit, or the machine stopped (limit 0).
    Halt,
    /// Execution continues at an address outside the current execute
    /// segment (C++ tail-calls `next_execute_segment`).
    Jump,
    /// A fault; the error is in [`DispatchState::error`].
    Fault,
}

/// Where an [`Exit`] leaves its details.
#[derive(Debug, Default)]
pub(crate) struct DispatchState {
    pc: u64,
    counter: Counter,
    error: Option<MachineError>,
    /// The running chain's limit, which handlers keep in memory without
    /// `nightly-dispatch` (see `abi`). Saved across nested runs.
    #[cfg(not(feature = "nightly-dispatch"))]
    pub(crate) limit: u64,
    /// The slice being run while a background compilation is pending.
    slice: Option<Slice>,
    /// A background compilation began during the run (an execute segment
    /// created at runtime): begin a slice at the next chance, so that its
    /// code gets installed.
    #[cfg(feature = "jit")]
    pub(crate) jit_spawned: bool,
}

/// A dispatcher run that stops early, at `end`, on the way to the real limit
/// `max` (see `Machine::dispatch_simulate`).
#[derive(Debug, Clone, Copy)]
struct Slice {
    /// The counter the slice began at (for deferred compilations).
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    begin: u64,
    end: u64,
    max: u64,
    /// Instructions charged for segment decodes during the slice: they
    /// do not make deferred compilations hot (a decode of a large area
    /// must not get itself compiled by its own charge).
    charged: u64,
}

impl DispatchState {
    /// True if `c` is past the limit of the slice being run but not past
    /// the real limit: only the slicing stops the chain here.
    #[inline]
    fn only_slice_overflowed(&self, c: Counter) -> bool {
        matches!(self.slice, Some(s) if c.max == s.end && c.value < s.max)
    }

    /// The limit host code (system calls) sees for the chain's limit `max`:
    /// the real one while running a slice, so slicing is not observable.
    #[inline]
    fn visible_max(&self, max: u64) -> u64 {
        match self.slice {
            Some(s) if max == s.end => s.max,
            _ => max,
        }
    }

    /// Back to the chain's limit after host code ran. If the host changed
    /// the limit (`stop()` or a new one), that one counts and the slice ends.
    #[inline]
    fn chain_max(&mut self, visible: u64) -> u64 {
        match self.slice {
            Some(s) if visible == s.max => s.end,
            Some(_) => {
                self.slice = None;
                visible
            }
            None => visible,
        }
    }
}

/// The chain's limit after host code ran (`DispatchState::chain_max`),
/// beginning a slice if a background compilation started meanwhile, or if
/// the chain runs without one (the host set a new limit, or the run had
/// none), so that `dispatch_slice` bounds the rest of the chain too.
#[inline]
fn chain_max(m: &mut Machine) -> u64 {
    #[cfg(feature = "jit")]
    if m.dispatch.jit_spawned && m.dispatch.slice.is_none() {
        m.dispatch.jit_spawned = false;
        let len = slice_len(m, true);
        return begin_slice(m, m.counter, m.max_counter, len);
    }
    let max = m.dispatch.chain_max(m.max_counter);
    if m.dispatch.slice.is_none() {
        let len = slice_len(m, false);
        return begin_slice(m, m.counter, max, len);
    }
    max
}

/// The length of a slice: `dispatch_slice`, and at most
/// `jit_background_slice` while background compilations are pending.
#[inline]
fn slice_len(m: &Machine, jit_pending: bool) -> u64 {
    let len = m.options().dispatch_slice.max(1);
    #[cfg(feature = "jit")]
    if jit_pending {
        return len.min(m.options().jit_background_slice.max(1));
    }
    #[cfg(not(feature = "jit"))]
    let _ = jit_pending;
    len
}

/// Begin a slice of `len` instructions at `counter` on the way to the real
/// limit `max`, as `Machine::dispatch_sliced` does, in a run that has none
/// (the slice loop goes on from where the slice ends). Returns the chain's
/// new limit.
#[inline(never)]
fn begin_slice(m: &mut Machine, counter: u64, max: u64, len: u64) -> u64 {
    let end = counter.saturating_add(len);
    if end >= max {
        return max;
    }
    m.dispatch.slice = Some(Slice {
        begin: counter,
        end,
        max,
        charged: 0,
    });
    end
}

/// Decoder entries per 4-byte instruction (C++ `NEXT_INSTR`).
pub(crate) const STEP: usize = if COMPRESSED { 2 } else { 1 };

#[cold]
#[inline(never)]
pub(crate) fn halt(m: &mut Machine, pc: u64, c: Counter) -> Exit {
    m.dispatch.pc = pc;
    m.dispatch.counter = c;
    Exit::Halt
}

#[cold]
#[inline(never)]
pub(crate) fn leave_segment(m: &mut Machine, pc: u64, c: Counter) -> Exit {
    m.dispatch.pc = pc;
    m.dispatch.counter = c;
    Exit::Jump
}

#[cold]
#[inline(never)]
pub(crate) fn fault(m: &mut Machine, error: MachineError) -> Exit {
    m.dispatch.error = Some(error);
    Exit::Fault
}

/// Report an out-of-range decoder index, even though decoder invariants
/// normally prevent it.
#[cold]
#[inline(never)]
pub(crate) fn bad_entry(m: &mut Machine, pc: u64) -> Exit {
    fault(
        m,
        MachineError::trigger(ExceptionKind::ExecutionSpaceProtectionFault, pc),
    )
}

cold_handler! {
    /// [`halt`] as a handler, for tail calls from handlers: a plain call would
    /// make the caller set up a stack frame (`nightly-dispatch` only).
    #[cfg(feature = "nightly-dispatch")]
    pub(crate) fn halt_exit(f) {
        let c = f.counter();
        halt(f.m, f.pc, c)
    }
}

cold_handler! {
    /// [`leave_segment`] as a handler (see [`halt_exit`]).
    #[cfg(feature = "nightly-dispatch")]
    pub(crate) fn leave_exit(f) {
        let c = f.counter();
        leave_segment(f.m, f.pc, c)
    }
}

cold_handler! {
    /// [`bad_entry`] as a handler (see [`halt_exit`]).
    #[cfg(feature = "nightly-dispatch")]
    pub(crate) fn bad_entry_exit(f) {
        bad_entry(f.m, f.pc)
    }
}

handler! {
    /// An instruction executed through its precise handler, continuing with the
    /// next entry. Used for C++ bytecodes whose handler does the same, without
    /// FUNCTION's stale-segment check (FMADD). The entry holds the original
    /// instruction bits.
    #[cfg(feature = "ext-f")]
    pub(crate) fn seq<O: Op>(f) {
        let instr = Instr(f.ins());
        if abi::host(f.m, move |m| O::exec(m, instr)).is_none() {
            return Exit::Fault;
        }
        execute!(f, f.i() + STEP)
    }
}

handler! {
    /// C++ `RV32C_BC_FUNCTION`: a compressed instruction executed through its
    /// precise handler, continuing with the next entry. Unlike
    /// `RV32I_BC_FUNCTION` it does not check for a stale segment afterwards.
    pub(crate) fn cseq<O: Op>(f) {
        let instr = Instr(f.ins());
        if abi::host(f.m, move |m| O::exec(m, instr)).is_none() {
            return Exit::Fault;
        }
        execute!(f, f.i() + 1)
    }
}

handler! {
    /// C++ `RV32I_BC_FUNCTION`: a slow-path instruction in the middle of a
    /// block. Like C++ it re-checks the segment afterwards, since the
    /// instruction may have made it stale (FENCE.I).
    pub(crate) fn function<O: Op>(f) {
        let instr = Instr(f.ins());
        if abi::host(f.m, move |m| O::exec(m, instr)).is_none() {
            return Exit::Fault;
        }
        if f.exec.is_stale() {
            // C++ ENTER_NEW_EXECUTE_SEGMENT
            let c = f.counter();
            return leave_segment(f.m, f.exec.address_of(f.i()).wrapping_add(4), c);
        }
        execute!(f, f.i() + STEP)
    }
}

handler! {
    /// C++ `RV32I_BC_FUNCBLOCK`: the last instruction of an over-long block,
    /// executed precisely (with `pc` visible to it).
    pub(crate) fn funcblock<O: Op>(f) {
        let instr = Instr(f.ins());
        f.m.cpu.regs.pc = f.exec.address_of(f.i());
        if abi::host(f.m, move |m| O::exec(m, instr)).is_none() {
            return Exit::Fault;
        }
        let len = i64::from(instr.length());
        if f.exec.is_stale() {
            let c = f.counter();
            return leave_segment(f.m, f.exec.address_of(f.i()).wrapping_add_signed(len), c);
        }
        next_block!(f, len, true)
    }
}

handler! {
    /// ECALL (C++ `rv32i_syscall`).
    pub(crate) fn syscall(f) {
        let pc = f.end_pc();
        // Make the current PC and the counters visible to the handler.
        f.m.cpu.regs.pc = pc;
        f.m.counter = f.counter().value;
        f.m.max_counter = f.m.dispatch.visible_max(f.counter().max);
        let number = f.m.cpu.regs.get(REG_ECALL);
        if abi::host(f.m, move |m| m.system_call(number)).is_none() {
            return Exit::Fault;
        }
        let c = Counter {
            value: f.m.counter,
            max: chain_max(f.m),
        };
        f.set_counter(c);
        // Stopped, or a clone-like system call changed PC. System calls are
        // always full-length instructions, so execution resumes after PC.
        if c.overflowed() || f.m.cpu.regs.pc != pc {
            let new_pc = f.m.cpu.regs.pc.wrapping_add(4);
            overflow_checked_jump!(f, new_pc)
        }
        next_block!(f, 4, false)
    }
}

/// What [`run_dyncall`] leaves behind.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DyncallDone {
    /// The counter and limit after the handler, the dyncall's block counted.
    pub c: Counter,
    /// The counter to halt with right after the dyncall: up to and
    /// including it.
    pub halt_after: Counter,
    /// The handler stopped or penalized the machine, or changed its limit.
    pub touched: bool,
    /// The dyncall ends its block: the segment's last instruction, or an
    /// over-long block forced to end there.
    pub ends_block: bool,
}

/// Run the dyncall `ins` at `pc`, whose block `c` has counted: the system
/// call numbered by its immediate, with the PC, the counter and the limit
/// published as for an ECALL. The handler can stop or penalize the machine,
/// but not change PC (a dyncall is never clone-like or an exit), so its
/// block goes on after it. Shared by the [`dyncall`] bytecode and the JIT's
/// dyncall exit; `counts` is false for native code that does not count.
pub(crate) fn run_dyncall(
    exec: &DecodedSegment,
    m: &mut Machine,
    pc: u64,
    ins: u32,
    c: Counter,
    counts: bool,
) -> std::result::Result<DyncallDone, Exit> {
    match call_dyncall(m, pc, ins, c) {
        Ok((c, touched)) => Ok(dyncall_done(exec, pc, c, touched, counts)),
        Err(err) => Err(fault(m, err)),
    }
}

/// The handler part of [`run_dyncall`]: the counter and limit after it,
/// and whether it stopped or penalized the machine, or changed its limit.
/// The JIT's host call (see `jit::run`).
#[inline]
pub(crate) fn call_dyncall(
    m: &mut Machine,
    pc: u64,
    ins: u32,
    c: Counter,
) -> Result<(Counter, bool)> {
    let visible_max = m.dispatch.visible_max(c.max);
    m.cpu.regs.pc = pc;
    m.counter = c.value;
    m.max_counter = visible_max;
    m.system_call(crate::Dyncall::index(ins))?;
    let touched = m.counter != c.value || m.max_counter != visible_max;
    let c = Counter {
        value: m.counter,
        max: chain_max(m),
    };
    Ok((c, touched))
}

/// The rest of [`run_dyncall`], after the dyncall at `pc` left `c`.
pub(crate) fn dyncall_done(
    exec: &DecodedSegment,
    pc: u64,
    c: Counter,
    touched: bool,
    counts: bool,
) -> DyncallDone {
    let ends_block = exec
        .cache
        .get(exec.index(pc))
        .is_none_or(|e| e.block_bytes() == 0);
    // The block is counted to its end: its instructions after the dyncall
    // are not retired yet.
    let retired = if ends_block || !counts {
        c.value
    } else {
        c.value
            .wrapping_sub(crate::native_code::block_icount(exec, pc.wrapping_add(4)))
    };
    DyncallDone {
        c,
        halt_after: Counter {
            value: retired,
            max: c.max,
        },
        touched,
        ends_block,
    }
}

handler! {
    /// Execute the system call encoded by a [`Dyncall`](crate::Dyncall) immediate.
    /// A stop or penalty reaching the retired-instruction limit halts immediately
    /// after the call with the exact count. Otherwise execution continues, checking
    /// a limit crossed by block counting at the block end. JIT dyncalls use the
    /// same rules. C++ instead treats dyncalls as custom instructions.
    pub(crate) fn dyncall(f) {
        let at = f.exec.address_of(f.i());
        let next = at.wrapping_add(4);
        let d = match run_dyncall(f.exec, f.m, at, f.ins(), f.counter(), true) {
            Ok(d) => d,
            Err(exit) => return exit,
        };
        // Only when the retired count reaches the limit: the block's count
        // crossing it is checked at the block end, penalty or not.
        if d.touched && d.halt_after.overflowed() {
            return halt(f.m, next, d.halt_after);
        }
        // FUNCTION's stale-segment check.
        if f.exec.is_stale() {
            return leave_segment(f.m, next, d.c);
        }
        f.set_counter(d.c);
        if d.ends_block {
            // The segment's last instruction in the 4-byte layout falls
            // through into the padding entry past the end, like FUNCTION.
            if !f.exec.is_within(next, 2) {
                execute!(f, f.i() + STEP)
            }
            // A forced block end checks the limit before the next block, like
            // FUNCBLOCK.
            if d.c.overflowed() {
                return halt(f.m, next, d.halt_after);
            }
            begin_block!(f, f.exec.index(next), next)
        }
        execute!(f, f.i() + STEP)
    }
}

handler! {
    /// WFI and the STOP instruction (C++ `rv32i_stop`).
    pub(crate) fn stop(f) {
        // Complete the STOP instruction, and stop.
        let c = Counter {
            value: f.counter().value,
            max: 0,
        };
        halt(f.m, f.end_pc().wrapping_add(4), c)
    }
}

handler! {
    /// Other SYSTEM instructions: EBREAK, CSRs (C++ `rv32i_system`).
    pub(crate) fn system(f) {
        let instr = Instr(f.ins());
        let pc = f.end_pc();
        let c = f.counter();
        f.m.cpu.regs.pc = pc;
        f.m.counter = c.value;
        f.m.max_counter = f.m.dispatch.visible_max(c.max);
        if abi::host(f.m, move |m| m.system(instr)).is_none() {
            return Exit::Fault;
        }
        let c = Counter {
            value: f.m.counter,
            max: chain_max(f.m),
        };
        f.set_counter(c);
        // Stopped, or PC changed: continue *at* PC (so a stopped machine
        // resumes at this instruction). C++ halts there at the limit too, after
        // running the instruction. Only the limit of a slice is different: the
        // next slice must go on as an uninterrupted run would, so the slice
        // ends at the next block instead, without running this one twice.
        if (c.overflowed() && !f.m.dispatch.only_slice_overflowed(c)) || f.m.cpu.regs.pc != pc {
            overflow_checked_jump!(f, f.m.cpu.regs.pc)
        }
        next_block!(f, 4, true)
    }
}

handler! {
    /// C++ `execute_invalid`: an entry that cannot be executed from the cache.
    /// Entries without instruction bits (the middle of a 4-byte instruction)
    /// are executed precisely from the segment's code instead.
    pub(crate) fn execute_invalid(f) {
        // The current PC, mid block.
        let pc = f.exec.address_of(f.i());
        let ins = f.ins();
        let c = f.counter();
        if ins == 0 && f.exec.is_within(pc, 2) {
            f.m.counter = c.value;
            f.m.max_counter = f.m.dispatch.visible_max(c.max);
            let exec = f.exec;
            let Some(pc) = abi::host(f.m, move |m| simulate_undecoded(m, exec, pc)) else {
                return Exit::Fault;
            };
            let c = Counter {
                value: f.m.counter,
                max: chain_max(f.m),
            };
            if c.overflowed() {
                return halt(f.m, pc, c);
            }
            if f.exec.is_stale() {
                return leave_segment(f.m, pc, c);
            }
            f.set_counter(c);
            unchecked_jump!(f, pc)
        }
        f.m.counter = c.value;
        f.m.cpu.regs.pc = pc;
        fault(
            f.m,
            MachineError::trigger(ExceptionKind::IllegalOpcode, u64::from(ins)),
        )
    }
}

/// Where the chain goes on after native code ([`native_step`]). The
/// counter and the address are left in `DispatchState::counter` and
/// `DispatchState::pc`.
#[derive(Clone, Copy)]
enum NativeNext {
    /// Leave the chain.
    Exit(Exit),
    /// Interpret the entry the native code was entered at, with its
    /// original handler (the frame's instruction bits and block end).
    Retry,
    /// Begin the block at the address (`BEGIN_BLOCK`).
    Begin,
    /// Jump to the address, which may be outside the segment.
    Indirect,
    /// Run the entry at the address with its own handler, in the same block
    /// (`EXECUTE_CURRENT`).
    Execute,
    /// Finish the block from the address with the interpreter's handler,
    /// the block counted already.
    Resume,
}

/// Run native code entered at entry `i` with counter `c`, until it leaves
/// for the interpreter or leaves the chain (`native_enter`'s loop). Outlined
/// without `nightly-dispatch`, like `abi::host`: `jit.run` returns through
/// memory, which would keep `native_enter`'s tail calls from becoming jumps.
#[cfg_attr(feature = "nightly-dispatch", inline(always))]
#[cfg_attr(not(feature = "nightly-dispatch"), inline(never))]
fn native_step(exec: &DecodedSegment, m: &mut Machine, i: usize, c: Counter) -> NativeNext {
    use crate::native_code::NativeExit;

    let start = exec.address_of(i);
    let (Some(e), Some(jit)) = (exec.cache.get(i), exec.native.as_ref()) else {
        return NativeNext::Exit(bad_entry(m, start));
    };
    let go = |m: &mut Machine, next: NativeNext, pc: u64, c: Counter| {
        m.dispatch.pc = pc;
        m.dispatch.counter = c;
        next
    };
    // A stale segment is left at the next slow-path instruction, which
    // native code does not check for: interpret it instead. (Nothing native
    // code runs can make the segment stale, so checking once is enough.)
    if exec.is_stale() {
        return go(m, NativeNext::Retry, start, c);
    }
    let (mut slot, mut entry_pc) = (i, start);
    // Native code counts the block it begins at, so it is entered with the
    // counter from before the block. Code ignoring the limit counts
    // nothing: the block stays counted, and the counter never goes down.
    let counts = !jit.unlimited();
    let mut c = c;
    let mut before = if counts {
        c.value.wrapping_sub(e.instruction_count())
    } else {
        c.value
    };
    loop {
        let max = c.max;
        let (pc, d) = match jit.run(exec, m, slot, entry_pc, before, max) {
            NativeExit::Halt { pc, counter } => {
                return NativeNext::Exit(halt(
                    m,
                    pc,
                    Counter {
                        value: counter,
                        max,
                    },
                ));
            }
            NativeExit::Jump { pc, counter } => {
                return go(
                    m,
                    NativeNext::Begin,
                    pc,
                    Counter {
                        value: counter,
                        max,
                    },
                );
            }
            NativeExit::Indirect { pc, counter } => {
                return go(
                    m,
                    NativeNext::Indirect,
                    pc,
                    Counter {
                        value: counter,
                        max,
                    },
                );
            }
            // Continue the chain at that instruction with the interpreter's
            // own handler (the block is counted already).
            NativeExit::Slow { pc, counter } => {
                return go(
                    m,
                    NativeNext::Resume,
                    pc,
                    Counter {
                        value: counter,
                        max,
                    },
                );
            }
            NativeExit::Dyncall { pc, counter } => {
                // What the interpreter's `dyncall` bytecode does: the
                // handler sees the exact PC, the counter and the limit, and
                // a stop or penalty takes effect right after it.
                let at_dyncall = Counter {
                    value: counter,
                    max,
                };
                match run_dyncall(exec, m, pc, exec.read_u32(pc), at_dyncall, counts) {
                    Ok(d) => (pc, d),
                    Err(exit) => return NativeNext::Exit(exit),
                }
            }
            // The JIT ran it, and could not go on after it.
            NativeExit::Dyncalled { pc, done } => (pc, done),
            NativeExit::Fault(err) => return NativeNext::Exit(fault(m, err)),
            NativeExit::Unavailable if slot == i => {
                return go(m, NativeNext::Retry, start, c);
            }
            // Re-entry after a dyncall: finish its block in the interpreter.
            NativeExit::Unavailable => return go(m, NativeNext::Resume, entry_pc, c),
        };
        c = d.c;
        let next = pc.wrapping_add(4);
        // As in the `dyncall` bytecode: the retired count decides.
        if d.touched && d.halt_after.overflowed() {
            return NativeNext::Exit(halt(m, next, d.halt_after));
        }
        // The handler may have made the segment stale: leave it.
        if exec.is_stale() {
            return NativeNext::Exit(leave_segment(m, next, c));
        }
        // The segment's last instruction in the 4-byte layout: its
        // block ends only because the segment does, and it falls
        // through into the padding entry after it, without
        // counting it or checking the limit. (With compressed
        // instructions the decoder makes that instruction invalid,
        // so it is never compiled.)
        if d.ends_block && !exec.is_within(next, 2) {
            return go(m, NativeNext::Execute, next, c);
        }
        // A forced block end checks the limit before the next
        // block. In the middle of a block, the limit is only
        // checked at its end, so the block's count crossing it is
        // no reason to stop here.
        if d.ends_block && c.overflowed() {
            return NativeNext::Exit(halt(m, next, d.halt_after));
        }
        let si = exec.index(next);
        if exec.is_within(next, 2) && jit.has_entry(si) {
            (slot, entry_pc, before) = (si, next, d.halt_after.value);
            continue;
        }
        if d.ends_block {
            return go(m, NativeNext::Begin, next, c);
        }
        // Finish the block in the interpreter, as after EXIT_SLOW.
        return go(m, NativeNext::Resume, next, c);
    }
}

handler! {
    /// The handler of decoder entries that begin a block with native code, from
    /// the JIT or translated ahead of time (C++ patches the decoder entry with
    /// the asmjit or binary translation function the same way). Only installed
    /// at block starts, which are only reached through `BEGIN_BLOCK`, so the
    /// counter already includes this block; native code counts it itself.
    ///
    /// The dispatcher loop also calls it directly for unpatched entries that
    /// native code can be entered at (`dispatch_chain`): native entries inside
    /// a decoder block, such as a loop head, which cannot be patched because the
    /// interpreter falls through them. Then the entry's own handler is the
    /// interpreter's.
    ///
    /// Native code runs in [`native_step`].
    pub(crate) fn native_enter(f) {
        let i = f.i();
        let (exec, c) = (f.exec, f.counter());
        let next = native_step(exec, f.m, i, c);
        let NativeNext::Exit(exit) = next else {
            let (pc, c) = (f.m.dispatch.pc, f.m.dispatch.counter);
            f.set_counter(c);
            // The interpreter's own handler of an entry native code patched.
            let original = |f: &abi::Frame<'_>, si: usize, own: Handler| {
                f.exec
                    .native
                    .as_ref()
                    .and_then(|jit| jit.original_handler(si))
                    .unwrap_or(own)
            };
            match next {
                NativeNext::Retry => {
                    let e = entry!(f, i, pc);
                    let handler = original(&f, i, e.handler);
                    tail!(f, handler)
                }
                NativeNext::Begin => begin_block!(f, f.exec.index(pc), pc),
                NativeNext::Indirect => unchecked_jump!(f, pc),
                NativeNext::Execute => execute!(f, f.exec.index(pc)),
                NativeNext::Resume | NativeNext::Exit(_) => {
                    let si = f.exec.index(pc);
                    let e = entry!(f, si, pc);
                    let handler = original(&f, si, e.handler);
                    resume!(f, handler, si, pc)
                }
            }
        };
        exit
    }
}

/// Port of `CPU::simulate_undecoded`: execute directly (and slowly) from the
/// segment's code until reaching a decoded entry. Uses the machine's
/// counters.
fn simulate_undecoded(m: &mut Machine, exec: &DecodedSegment, mut pc: u64) -> Result<u64> {
    loop {
        if guest_rewrote_code(m, exec, pc) {
            exec.set_stale(true);
            break;
        }
        m.cpu.regs.pc = pc;
        let instr = Instr(exec.read_u32(pc));
        m.execute(instr)?;
        m.counter = m.counter.wrapping_add(1);
        let len = if COMPRESSED {
            u64::from(instr.length())
        } else {
            4
        };
        m.cpu.regs.pc = m.cpu.regs.pc.wrapping_add(len);
        pc = m.cpu.regs.pc;

        if !std::ptr::eq(Arc::as_ptr(&m.cpu.exec), exec) {
            break;
        }
        if !exec.is_within(pc, 2) || exec.is_stale() {
            break;
        }
        if exec.cache.get(exec.index(pc)).is_none_or(|e| !e.invalid) {
            break;
        }
        if m.stopped() {
            break;
        }
    }
    Ok(pc)
}

/// Port of `CPU::guest_rewrote_code`.
fn guest_rewrote_code(m: &Machine, exec: &DecodedSegment, pc: u64) -> bool {
    if exec.is_execute_only() || !exec.is_within(pc, 2) {
        return false;
    }
    m.memory
        .read_u16(pc)
        .is_ok_and(|current| current != exec.read_u16(pc))
}

impl Machine {
    /// Port of `CPU::simulate` (tail-call dispatch): run from `pc` with the
    /// counter starting at `counter`, until it reaches `max` or the machine
    /// stops. Returns the final counters; a limit of zero means the machine
    /// stopped (C++: "stopped normally"). Sets PC, but like C++ leaves
    /// recording the counter to the caller.
    ///
    /// Reentrant: a system call handler may run the machine again (a
    /// `preempt`), which gets its own slicing state.
    pub(crate) fn dispatch_simulate(&mut self, pc: u64, counter: u64, max: u64) -> Result<Counter> {
        let outer = self.dispatch.slice.take();
        #[cfg(not(feature = "nightly-dispatch"))]
        let outer_limit = self.dispatch.limit;
        let result = self.dispatch_sliced(pc, counter, max);
        self.dispatch.slice = outer;
        #[cfg(not(feature = "nightly-dispatch"))]
        {
            self.dispatch.limit = outer_limit;
        }
        // A compilation this (nested) run started concerns the outer run too.
        #[cfg(feature = "jit")]
        if outer.is_none() && (self.jit_pending() || self.memory.jit_deferred()) {
            self.dispatch.jit_spawned = true;
        }
        result
    }

    /// `dispatch_simulate` in slices: chains of at most `dispatch_slice`
    /// instructions (up to the next limit check), which bounds the host stack
    /// of handlers chained by ordinary calls, and while background
    /// compilations are pending, of `jit_background_slice` instructions,
    /// installing finished native code in between. A slice ends at an
    /// instruction-limit check, exactly where a run with that limit would
    /// halt, and the next one begins the block there, as the uninterrupted
    /// run would have: results and counts do not depend on where the slices
    /// fall. A compilation that starts during an unsliced run (a segment
    /// created at runtime) begins a slice there (`begin_slice`). Segments
    /// whose compilation is deferred until they are hot also run in slices,
    /// each of which counts towards it.
    fn dispatch_sliced(&mut self, pc: u64, counter: u64, max: u64) -> Result<Counter> {
        let (mut pc, mut counter, mut max) = (pc, counter, max);
        loop {
            #[cfg(feature = "jit")]
            let pending = {
                self.dispatch.jit_spawned = false;
                (self.jit_pending() && {
                    self.refresh_execute_segments(false);
                    self.jit_pending()
                }) || self.memory.jit_deferred()
            };
            #[cfg(not(feature = "jit"))]
            let pending = false;
            let end = counter.saturating_add(slice_len(self, pending));
            let limit = if end < max {
                self.dispatch.slice = Some(Slice {
                    begin: counter,
                    end,
                    max,
                    charged: 0,
                });
                end
            } else {
                max
            };
            let result = self.dispatch_chain(pc, counter, limit);
            let slice = self.dispatch.slice.take();
            let (c, halt_pc) = result?;
            // The slice ran out: nothing stopped the machine or changed the
            // limit. (A slice begun during the run has the limit of then.)
            if let Some(s) = slice
                && c.max == s.end
            {
                // Deferred compilations: the instructions of this slice.
                #[cfg(feature = "jit")]
                {
                    let executed = c.value.saturating_sub(s.begin).saturating_sub(s.charged);
                    let hot = self.options().jit_background_slice.max(1);
                    self.memory.run_deferred_jit(executed, hot);
                }
                max = s.max;
                // At this check, the real limit is reached too: an
                // uninterrupted run would halt here as well.
                if c.value >= max {
                    self.cpu.regs.pc = halt_pc;
                    return Ok(Counter {
                        value: c.value,
                        max,
                    });
                }
                // Continue without publishing the PC: an uninterrupted run
                // would not have stored it (it is observable after a fault).
                pc = halt_pc;
                counter = c.value;
                continue;
            }
            self.cpu.regs.pc = halt_pc;
            return Ok(c);
        }
    }

    /// The dispatcher loop. Returns the final counters and the PC it halted
    /// at, which the caller stores.
    fn dispatch_chain(&mut self, pc: u64, counter: u64, max: u64) -> Result<(Counter, u64)> {
        let mut c = Counter {
            value: counter,
            max,
        };
        let mut pc = pc;
        // Select the execute segment containing the current PC.
        if !(pc >= self.cpu.exec.exec_begin() && pc < self.cpu.exec.exec_end()) {
            let charge;
            (pc, charge) = self.next_execute_segment(pc)?;
            if self.charge_decode(&mut c, charge) {
                return Ok((c, pc));
            }
        }
        loop {
            let exec = Arc::clone(&self.cpu.exec);
            let i = exec.index(pc);
            let Some(e) = exec.cache.get(i) else {
                return Err(MachineError::trigger(
                    ExceptionKind::ExecutionSpaceProtectionFault,
                    pc,
                ));
            };
            // BEGIN_BLOCK
            let block_pc = pc.wrapping_add(e.block_bytes());
            c.value = c.value.wrapping_add(e.instruction_count());
            // Native code can be entered at more places than the decoder
            // entries patched with `native_enter`: those are only the decoder
            // block starts, since the interpreter falls through the others.
            // A block beginning here at a native entry inside a decoder block
            // (resuming after a background-compilation slice or a halt, at a
            // loop head reached by a backward branch) enters native code too;
            // otherwise the interpreter would run the whole loop.
            let handler = match &exec.native {
                Some(native) if native.has_entry(i) && !exec.is_stale() => native_enter,
                _ => e.handler,
            };
            match abi::enter(handler, &exec, i, self, c, e.instr, block_pc) {
                Exit::Halt => return Ok((self.dispatch.counter, self.dispatch.pc)),
                Exit::Jump => {
                    c = self.dispatch.counter;
                    if let Some(stopped) = self.stop_in_other_segment(self.dispatch.pc, c) {
                        return Ok(stopped);
                    }
                    let charge;
                    (pc, charge) = self.next_execute_segment(self.dispatch.pc)?;
                    if self.charge_decode(&mut c, charge) {
                        return Ok((c, pc));
                    }
                    // The new segment may be compiling in the background.
                    #[cfg(feature = "jit")]
                    if self.dispatch.jit_spawned && self.dispatch.slice.is_none() {
                        self.dispatch.jit_spawned = false;
                        let len = slice_len(self, true);
                        c.max = begin_slice(self, c.value, c.max, len);
                    }
                }
                Exit::Fault => {
                    return Err(self
                        .dispatch
                        .error
                        .take()
                        .unwrap_or(MachineError::trigger(ExceptionKind::UnknownException, 0)));
                }
            }
        }
    }

    /// Charge a segment decode (`next_execute_segment`; not C++, see
    /// `MachineOptions::free_decode_bytes`). True if that reaches the chain's limit: the
    /// dispatcher then halts before the new block, as at a limit check.
    fn charge_decode(&mut self, c: &mut Counter, charge: u64) -> bool {
        if charge == 0 {
            return false;
        }
        c.value = c.value.saturating_add(charge);
        if let Some(s) = &mut self.dispatch.slice {
            s.charged = s.charged.saturating_add(charge);
        }
        c.value >= c.max
    }

    /// Precise handler for any instruction (C++ `CPU::execute(format_t)`).
    pub(crate) fn execute(&mut self, instr: Instr) -> Result<()> {
        (decode::decode(instr).exec)(self, instr)
    }
}
