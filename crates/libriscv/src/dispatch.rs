//! Port of `tailcall_dispatch.cpp` and the slow-path and system handlers of
//! `bytecode_impl.cpp` (`RV32I_BC_FUNCTION`, `FUNCBLOCK`, `SYSCALL`, ...).
//! The fast-path bytecodes are in `bytecode.rs`.
//!
//! Every decoder cache entry stores a handler with the [`Handler`]
//! signature. A handler executes its instruction and then tail-calls the
//! handler of the next entry with `become`, so each instruction gets its own
//! indirect branch and the chain never grows the stack. The chain ends by
//! returning an [`Exit`] when the instruction counter overflows, the machine
//! stops, execution leaves the current execute segment, or a fault occurs.
//!
//! Handlers use the `preserve_none` calling convention (as in C++), so they
//! save no registers, and get their entry's instruction bits as an argument
//! (C++ reads `d->instr`), so they don't look up their own entry. Exits are
//! tail calls too (`halt_exit` etc.): a handler that makes no calls needs no
//! stack frame.
//!
//! As in C++, `pc` is not tracked per instruction. On entering a block
//! (`BEGIN_BLOCK`) `pc` is advanced to the block's last instruction and the
//! block's instruction count is added to the counter. Only block-ending
//! instructions (branches, jumps, system instructions) look at `pc`; AUIPC
//! reconstructs its own address from the entry's distance to the block end.
//!
//! The structure follows `tailcall_dispatch.cpp`, but where its observable
//! behaviour differs from `cpu_dispatch.cpp` (the threaded and switch
//! dispatch of the default C++ build, which is the test oracle), this
//! follows `cpu_dispatch.cpp`: the counter is recorded before an
//! illegal-opcode fault, system calls that stop or change PC resume after
//! PC, and other SYSTEM instructions retrieve both counters.

use std::sync::Arc;

use crate::decode;
use crate::decoder_cache::{COMPRESSED, DecodedSegment, DecoderEntry};
use crate::error::{ExceptionKind, MachineError, Result};
use crate::instruction::Instr;
use crate::isa::Op;
use crate::machine::Machine;
use crate::registers::REG_ECALL;

/// The signature shared by all handlers: the current execute segment, its
/// decoder cache (in registers, which saves two loads per instruction), the
/// index of this handler's decoder entry, the machine, the counters, the
/// entry's instruction bits (re-encoded by the rewriter for fast bytecodes,
/// the original instruction otherwise), and the dispatcher's `pc` (the last
/// instruction of the current block).
///
/// The order decides the registers (`preserve_none` on x86-64: r12, r13,
/// r14, r15, rdi, rsi, rdx, rcx, r8, ...). It puts the instruction bits in
/// rcx: a variable shift takes its count in cl, and the shift immediates
/// are the low bits of the re-encoded instruction, so the shift handlers
/// use them in place instead of saving and restoring a live rcx (CoreMark
/// rv64imafdc: 4.5% fewer cycles than with the counters in rdx/rcx).
pub(crate) type Handler = extern "rust-preserve-none" fn(
    &DecodedSegment,
    &[DecoderEntry],
    usize,
    &mut Machine,
    Counter,
    u32,
    u64,
) -> Exit;

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
    end: u64,
    max: u64,
}

impl DispatchState {
    /// The limit host code (system calls) sees for the chain's limit `max`:
    /// the real one while running a slice, so slicing isn't observable.
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
/// beginning a slice if a background compilation started meanwhile.
#[inline]
fn chain_max(m: &mut Machine) -> u64 {
    #[cfg(feature = "jit")]
    if m.dispatch.jit_spawned && m.dispatch.slice.is_none() {
        return begin_slice(m, m.counter, m.max_counter);
    }
    m.dispatch.chain_max(m.max_counter)
}

/// Begin a slice at `counter` on the way to the real limit `max`, as
/// `Machine::dispatch_sliced` does, for a background compilation that
/// started during an unsliced run (the slice loop goes on from where the
/// slice ends). Returns the chain's new limit.
#[cfg(feature = "jit")]
#[cold]
#[inline(never)]
fn begin_slice(m: &mut Machine, counter: u64, max: u64) -> u64 {
    m.dispatch.jit_spawned = false;
    let end = counter.saturating_add(m.options().jit_background_slice.max(1));
    if end >= max {
        return max;
    }
    m.dispatch.slice = Some(Slice { end, max });
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

/// A decoder cache index outside the cache. The decoder guarantees this
/// can't happen; it is handled anyway rather than trusted.
#[cold]
#[inline(never)]
pub(crate) fn bad_entry(m: &mut Machine, pc: u64) -> Exit {
    fault(
        m,
        MachineError::trigger(ExceptionKind::ExecutionSpaceProtectionFault, pc),
    )
}

/// [`halt`] as a handler, for tail calls from handlers: a plain call would
/// make the caller set up a stack frame.
#[cold]
#[inline(never)]
pub(crate) extern "rust-preserve-none" fn halt_exit(
    _exec: &DecodedSegment,
    _cache: &[DecoderEntry],
    _i: usize,
    m: &mut Machine,
    c: Counter,
    _ins: u32,
    pc: u64,
) -> Exit {
    halt(m, pc, c)
}

/// [`leave_segment`] as a handler (see [`halt_exit`]).
#[cold]
#[inline(never)]
pub(crate) extern "rust-preserve-none" fn leave_exit(
    _exec: &DecodedSegment,
    _cache: &[DecoderEntry],
    _i: usize,
    m: &mut Machine,
    c: Counter,
    _ins: u32,
    pc: u64,
) -> Exit {
    leave_segment(m, pc, c)
}

/// [`bad_entry`] as a handler (see [`halt_exit`]).
#[cold]
#[inline(never)]
pub(crate) extern "rust-preserve-none" fn bad_entry_exit(
    _exec: &DecodedSegment,
    _cache: &[DecoderEntry],
    _i: usize,
    m: &mut Machine,
    _c: Counter,
    _ins: u32,
    pc: u64,
) -> Exit {
    bad_entry(m, pc)
}

/// The entry at index `$i`, or return a fault.
macro_rules! entry {
    ($exec:ident, $cache:ident, $i:expr, $m:ident, $pc:expr, $c:expr) => {
        match $cache.get($i) {
            Some(e) => e,
            None => become $crate::dispatch::bad_entry_exit($exec, $cache, 0, $m, $c, 0, $pc),
        }
    };
}
pub(crate) use entry;

/// C++ `EXECUTE_CURRENT`: tail-call the handler of entry `$i`, handing it
/// the entry's (re-encoded) instruction bits.
macro_rules! execute {
    ($exec:ident, $cache:ident, $i:expr, $m:ident, $pc:expr, $c:expr) => {{
        let i: usize = $i;
        let e = $crate::dispatch::entry!($exec, $cache, i, $m, $pc, $c);
        become (e.handler)($exec, $cache, i, $m, $c, e.instr, $pc)
    }};
}
pub(crate) use execute;

/// C++ `BEGIN_BLOCK` + `EXECUTE_CURRENT`: enter the block at entry `$i`.
macro_rules! begin_block {
    ($exec:ident, $cache:ident, $i:expr, $m:ident, $pc:expr, $c:expr) => {{
        let i: usize = $i;
        let c: $crate::dispatch::Counter = $c;
        let e = $crate::dispatch::entry!($exec, $cache, i, $m, $pc, c);
        let pc: u64 = ($pc).wrapping_add(e.block_bytes());
        let c = $crate::dispatch::Counter {
            value: c.value.wrapping_add(e.instruction_count()),
            max: c.max,
        };
        become (e.handler)($exec, $cache, i, $m, c, e.instr, pc)
    }};
}
pub(crate) use begin_block;

/// C++ `NEXT_BLOCK(len, OF)` (and `PERFORM_BRANCH`, which is
/// `NEXT_BLOCK(imm, true)`): continue `$len` bytes from the current block's
/// last instruction, checking the counter first if `$of`.
macro_rules! next_block {
    ($exec:ident, $cache:ident, $i:expr, $m:ident, $pc:expr, $c:expr, $len:expr, $of:expr) => {{
        let len: i64 = $len;
        let c: $crate::dispatch::Counter = $c;
        let pc: u64 = ($pc).wrapping_add_signed(len);
        let i: usize = ($i).wrapping_add_signed((len >> $crate::decoder_cache::SHIFT) as isize);
        if $of && c.overflowed() {
            become $crate::dispatch::halt_exit($exec, $cache, 0, $m, c, 0, pc);
        }
        $crate::dispatch::begin_block!($exec, $cache, i, $m, pc, c)
    }};
}
pub(crate) use next_block;

/// C++ `UNCHECKED_JUMP` (with its `QUICK_EXEC_CHECK`): continue at `$pc`,
/// leaving the chain if it is outside the current segment.
macro_rules! unchecked_jump {
    ($exec:ident, $cache:ident, $m:ident, $pc:expr, $c:expr) => {{
        let pc: u64 = $pc;
        let c: $crate::dispatch::Counter = $c;
        if !(pc >= $exec.exec_begin() && pc < $exec.exec_end()) {
            become $crate::dispatch::leave_exit($exec, $cache, 0, $m, c, 0, pc);
        }
        $crate::dispatch::begin_block!($exec, $cache, $exec.index(pc), $m, pc, c)
    }};
}
pub(crate) use unchecked_jump;

/// C++ `OVERFLOW_CHECKED_JUMP`.
macro_rules! overflow_checked_jump {
    ($exec:ident, $cache:ident, $m:ident, $pc:expr, $c:expr) => {{
        let pc: u64 = $pc;
        let c: $crate::dispatch::Counter = $c;
        if c.overflowed() {
            become $crate::dispatch::halt_exit($exec, $cache, 0, $m, c, 0, pc);
        }
        $crate::dispatch::unchecked_jump!($exec, $cache, $m, pc, c)
    }};
}
pub(crate) use overflow_checked_jump;

/// An instruction executed through its precise handler, continuing with the
/// next entry. Used for C++ bytecodes whose handler does the same, without
/// FUNCTION's stale-segment check (FMADD). The entry holds the original
/// instruction bits.
#[cfg(feature = "ext-f")]
#[rustc_align(64)]
pub(crate) extern "rust-preserve-none" fn seq<O: Op>(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    c: Counter,
    ins: u32,
    pc: u64,
) -> Exit {
    if let Err(err) = O::exec(m, Instr(ins)) {
        return fault(m, err);
    }
    execute!(exec, cache, i + STEP, m, pc, c)
}

/// C++ `RV32C_BC_FUNCTION`: a compressed instruction executed through its
/// precise handler, continuing with the next entry. Unlike
/// `RV32I_BC_FUNCTION` it doesn't check for a stale segment afterwards.
#[rustc_align(64)]
pub(crate) extern "rust-preserve-none" fn cseq<O: Op>(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    c: Counter,
    ins: u32,
    pc: u64,
) -> Exit {
    if let Err(err) = O::exec(m, Instr(ins)) {
        return fault(m, err);
    }
    execute!(exec, cache, i + 1, m, pc, c)
}

/// C++ `RV32I_BC_FUNCTION`: a slow-path instruction in the middle of a
/// block. Like C++ it re-checks the segment afterwards, since the
/// instruction may have made it stale (FENCE.I).
#[rustc_align(64)]
pub(crate) extern "rust-preserve-none" fn function<O: Op>(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    c: Counter,
    ins: u32,
    pc: u64,
) -> Exit {
    if let Err(err) = O::exec(m, Instr(ins)) {
        return fault(m, err);
    }
    if exec.is_stale() {
        // C++ ENTER_NEW_EXECUTE_SEGMENT
        return leave_segment(m, exec.address_of(i).wrapping_add(4), c);
    }
    execute!(exec, cache, i + STEP, m, pc, c)
}

/// C++ `RV32I_BC_FUNCBLOCK`: the last instruction of an over-long block,
/// executed precisely (with `pc` visible to it).
#[rustc_align(64)]
pub(crate) extern "rust-preserve-none" fn funcblock<O: Op>(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    c: Counter,
    ins: u32,
    pc: u64,
) -> Exit {
    let instr = Instr(ins);
    m.cpu.regs.pc = exec.address_of(i);
    if let Err(err) = O::exec(m, instr) {
        return fault(m, err);
    }
    let len = i64::from(instr.length());
    if exec.is_stale() {
        return leave_segment(m, exec.address_of(i).wrapping_add_signed(len), c);
    }
    next_block!(exec, cache, i, m, pc, c, len, true)
}

/// ECALL (C++ `rv32i_syscall`).
#[rustc_align(64)]
pub(crate) extern "rust-preserve-none" fn syscall(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    c: Counter,
    _ins: u32,
    pc: u64,
) -> Exit {
    // Make the current PC and the counters visible to the handler.
    m.cpu.regs.pc = pc;
    m.counter = c.value;
    m.max_counter = m.dispatch.visible_max(c.max);
    let number = m.cpu.regs.get(REG_ECALL);
    if let Err(err) = m.system_call(number) {
        return fault(m, err);
    }
    let c = Counter {
        value: m.counter,
        max: chain_max(m),
    };
    // Stopped, or a clone-like system call changed PC. System calls are
    // always full-length instructions, so execution resumes after PC.
    if c.overflowed() || m.cpu.regs.pc != pc {
        let new_pc = m.cpu.regs.pc.wrapping_add(4);
        overflow_checked_jump!(exec, cache, m, new_pc, c)
    }
    next_block!(exec, cache, i, m, pc, c, 4, false)
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
/// dyncall exit; `counts` is false for native code that doesn't count.
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
    // aren't retired yet.
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

/// A [`Dyncall`](crate::Dyncall) word: the system call numbered by its
/// immediate, in the middle of its block (not in C++, where a dyncall is a
/// custom instruction that runs without the counter). A stop, or a penalty
/// that takes the retired count to the limit, halts right after it with the
/// exact count; otherwise
/// the block goes on, and a limit merely crossed by the block's count is
/// checked at its end, as for any instruction. The JIT's dyncall exit does
/// the same.
#[rustc_align(64)]
pub(crate) extern "rust-preserve-none" fn dyncall(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    c: Counter,
    ins: u32,
    pc: u64,
) -> Exit {
    let at = exec.address_of(i);
    let next = at.wrapping_add(4);
    let d = match run_dyncall(exec, m, at, ins, c, true) {
        Ok(d) => d,
        Err(exit) => return exit,
    };
    // Only when the retired count reaches the limit: the block's count
    // crossing it is checked at the block end, penalty or not.
    if d.touched && d.halt_after.overflowed() {
        return halt(m, next, d.halt_after);
    }
    // FUNCTION's stale-segment check.
    if exec.is_stale() {
        return leave_segment(m, next, d.c);
    }
    if d.ends_block {
        // The segment's last instruction in the 4-byte layout falls
        // through into the padding entry past the end, like FUNCTION.
        if !exec.is_within(next, 2) {
            execute!(exec, cache, i + STEP, m, pc, d.c)
        }
        // A forced block end checks the limit before the next block, like
        // FUNCBLOCK.
        if d.c.overflowed() {
            return halt(m, next, d.halt_after);
        }
        begin_block!(exec, cache, exec.index(next), m, next, d.c)
    }
    execute!(exec, cache, i + STEP, m, pc, d.c)
}

/// WFI and the STOP instruction (C++ `rv32i_stop`).
#[rustc_align(64)]
pub(crate) extern "rust-preserve-none" fn stop(
    _exec: &DecodedSegment,
    _cache: &[DecoderEntry],
    _i: usize,
    m: &mut Machine,
    c: Counter,
    _ins: u32,
    pc: u64,
) -> Exit {
    // Complete the STOP instruction, and stop.
    halt(
        m,
        pc.wrapping_add(4),
        Counter {
            value: c.value,
            max: 0,
        },
    )
}

/// Other SYSTEM instructions: EBREAK, CSRs (C++ `rv32i_system`).
#[rustc_align(64)]
pub(crate) extern "rust-preserve-none" fn system(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    c: Counter,
    ins: u32,
    pc: u64,
) -> Exit {
    let instr = Instr(ins);
    m.cpu.regs.pc = pc;
    m.counter = c.value;
    m.max_counter = m.dispatch.visible_max(c.max);
    if let Err(err) = m.system(instr) {
        return fault(m, err);
    }
    let c = Counter {
        value: m.counter,
        max: chain_max(m),
    };
    // Stopped, or PC changed: continue *at* PC (so a stopped machine
    // resumes at this instruction).
    if c.overflowed() || m.cpu.regs.pc != pc {
        overflow_checked_jump!(exec, cache, m, m.cpu.regs.pc, c)
    }
    next_block!(exec, cache, i, m, pc, c, 4, true)
}

/// C++ `execute_invalid`: an entry that can't be executed from the cache.
/// Entries without instruction bits (the middle of a 4-byte instruction)
/// are executed precisely from the segment's code instead.
#[rustc_align(64)]
pub(crate) extern "rust-preserve-none" fn execute_invalid(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    c: Counter,
    ins: u32,
    _pc: u64,
) -> Exit {
    // The current PC, mid block.
    let pc = exec.address_of(i);
    if ins == 0 && exec.is_within(pc, 2) {
        m.counter = c.value;
        m.max_counter = m.dispatch.visible_max(c.max);
        let pc = match simulate_undecoded(m, exec, pc) {
            Ok(pc) => pc,
            Err(err) => return fault(m, err),
        };
        let c = Counter {
            value: m.counter,
            max: chain_max(m),
        };
        if c.overflowed() {
            return halt(m, pc, c);
        }
        if exec.is_stale() {
            return leave_segment(m, pc, c);
        }
        unchecked_jump!(exec, cache, m, pc, c)
    }
    m.counter = c.value;
    m.cpu.regs.pc = pc;
    fault(
        m,
        MachineError::trigger(ExceptionKind::IllegalOpcode, u64::from(ins)),
    )
}

/// The handler of decoder entries that begin a block with native code, from
/// the JIT or translated ahead of time (C++ patches the decoder entry with
/// the asmjit or binary translation function the same way). Only installed
/// at block starts, which are only reached through `BEGIN_BLOCK`, so the
/// counter already includes this block; native code counts it itself.
///
/// The dispatcher loop also calls it directly for unpatched entries that
/// native code can be entered at (`dispatch_chain`): native entries inside
/// a decoder block, such as a loop head, which can't be patched because the
/// interpreter falls through them. Then the entry's own handler is the
/// interpreter's.
#[rustc_align(64)]
pub(crate) extern "rust-preserve-none" fn native_enter(
    exec: &DecodedSegment,
    cache: &[DecoderEntry],
    i: usize,
    m: &mut Machine,
    c: Counter,
    ins: u32,
    pc: u64,
) -> Exit {
    use crate::native_code::NativeExit;

    let e = entry!(exec, cache, i, m, pc, c);
    let start = exec.address_of(i);
    let Some(jit) = exec.native.as_ref() else {
        return bad_entry(m, start);
    };
    // A stale segment is left at the next slow-path instruction, which
    // native code doesn't check for: interpret it instead. (Nothing native
    // code runs can make the segment stale, so checking once is enough.)
    if exec.is_stale() {
        let handler = jit.original_handler(i).unwrap_or(e.handler);
        become handler(exec, cache, i, m, c, ins, pc)
    }
    let (mut slot, mut entry_pc) = (i, start);
    // Native code counts the block it begins at, so it's entered with the
    // counter from before the block. Code ignoring the limit counts
    // nothing: the block stays counted, and the counter never goes down.
    let counts = !jit.unlimited();
    let mut before = if counts {
        c.value.wrapping_sub(e.instruction_count())
    } else {
        c.value
    };
    let mut c = c;
    loop {
        let (pc, d) = match jit.run(exec, m, slot, entry_pc, before, c.max) {
            NativeExit::Halt { pc, counter } => {
                return halt(
                    m,
                    pc,
                    Counter {
                        value: counter,
                        max: c.max,
                    },
                );
            }
            NativeExit::Jump { pc, counter } => begin_block!(
                exec,
                cache,
                exec.index(pc),
                m,
                pc,
                Counter {
                    value: counter,
                    max: c.max,
                }
            ),
            NativeExit::Indirect { pc, counter } => unchecked_jump!(
                exec,
                cache,
                m,
                pc,
                Counter {
                    value: counter,
                    max: c.max,
                }
            ),
            NativeExit::Slow { pc, counter } => {
                // Continue the chain at that instruction with the interpreter's
                // own handler (the block is counted already).
                let si = exec.index(pc);
                let e = entry!(exec, cache, si, m, pc, c);
                let handler = jit.original_handler(si).unwrap_or(e.handler);
                let block_pc = pc.wrapping_add(e.block_bytes());
                become handler(
                    exec,
                    cache,
                    si,
                    m,
                    Counter {
                        value: counter,
                        max: c.max,
                    },
                    e.instr,
                    block_pc,
                )
            }
            NativeExit::Dyncall { pc, counter } => {
                // What the interpreter's `dyncall` bytecode does: the
                // handler sees the exact PC, the counter and the limit, and
                // a stop or penalty takes effect right after it.
                let at_dyncall = Counter {
                    value: counter,
                    max: c.max,
                };
                match run_dyncall(exec, m, pc, exec.read_u32(pc), at_dyncall, counts) {
                    Ok(d) => (pc, d),
                    Err(exit) => return exit,
                }
            }
            // The JIT ran it, and couldn't go on after it.
            NativeExit::Dyncalled { pc, done } => (pc, done),
            NativeExit::Fault(err) => return fault(m, err),
            NativeExit::Unavailable if slot == i => {
                let handler = jit.original_handler(i).unwrap_or(e.handler);
                become handler(exec, cache, i, m, c, ins, pc)
            }
            NativeExit::Unavailable => {
                // Re-entry after a dyncall: finish its block in the interpreter.
                let e = entry!(exec, cache, slot, m, entry_pc, c);
                let handler = jit.original_handler(slot).unwrap_or(e.handler);
                let block_pc = entry_pc.wrapping_add(e.block_bytes());
                become handler(exec, cache, slot, m, c, e.instr, block_pc)
            }
        };
        c = d.c;
        let next = pc.wrapping_add(4);
        // As in the `dyncall` bytecode: the retired count decides.
        if d.touched && d.halt_after.overflowed() {
            return halt(m, next, d.halt_after);
        }
        // The handler may have made the segment stale: leave it.
        if exec.is_stale() {
            return leave_segment(m, next, c);
        }
        // The segment's last instruction in the 4-byte layout: its
        // block ends only because the segment does, and it falls
        // through into the padding entry after it, without
        // counting it or checking the limit. (With compressed
        // instructions the decoder makes that instruction invalid,
        // so it is never compiled.)
        if d.ends_block && !exec.is_within(next, 2) {
            execute!(exec, cache, exec.index(next), m, pc, c)
        }
        // A forced block end checks the limit before the next
        // block. In the middle of a block, the limit is only
        // checked at its end, so the block's count crossing it is
        // no reason to stop here.
        if d.ends_block && c.overflowed() {
            return halt(m, next, d.halt_after);
        }
        let si = exec.index(next);
        if exec.is_within(next, 2) && jit.has_entry(si) {
            (slot, entry_pc, before) = (si, next, d.halt_after.value);
            continue;
        }
        if d.ends_block {
            begin_block!(exec, cache, si, m, next, c)
        }
        // Finish the block in the interpreter, as after EXIT_SLOW.
        let e = entry!(exec, cache, si, m, next, c);
        let handler = jit.original_handler(si).unwrap_or(e.handler);
        let block_pc = next.wrapping_add(e.block_bytes());
        become handler(exec, cache, si, m, c, e.instr, block_pc)
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
        #[cfg(feature = "jit")]
        let result = self.dispatch_sliced(pc, counter, max);
        #[cfg(not(feature = "jit"))]
        let result = self.dispatch_run(pc, counter, max);
        self.dispatch.slice = outer;
        // A compilation this (nested) run started concerns the outer run too.
        #[cfg(feature = "jit")]
        if outer.is_none() && (self.jit_pending() || self.memory.jit_deferred()) {
            self.dispatch.jit_spawned = true;
        }
        result
    }

    /// `dispatch_simulate` with the JIT: while background compilations are
    /// pending, run in slices of `jit_background_slice` instructions,
    /// installing finished native code in between. A slice ends at an
    /// instruction-limit check, exactly where a run with that limit would
    /// halt, and the next one begins the block there, as the uninterrupted
    /// run would have: results and counts don't depend on where the slices
    /// fall. A compilation that starts during an unsliced run (a segment
    /// created at runtime) begins a slice there (`begin_slice`). Segments
    /// whose compilation is deferred until they are hot also run in slices,
    /// each of which counts towards it.
    #[cfg(feature = "jit")]
    fn dispatch_sliced(&mut self, pc: u64, counter: u64, max: u64) -> Result<Counter> {
        let (mut pc, mut counter, mut max) = (pc, counter, max);
        let slice_len = self.options().jit_background_slice.max(1);
        loop {
            self.dispatch.jit_spawned = false;
            let pending = (self.jit_pending() && {
                self.refresh_execute_segments(false);
                self.jit_pending()
            }) || self.memory.jit_deferred();
            let end = counter.saturating_add(slice_len);
            let limit = if pending && end < max {
                self.dispatch.slice = Some(Slice { end, max });
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
                // Deferred compilations: the instructions of this slice (it
                // may have begun after `counter`, see `begin_slice`).
                let begun = s.end.saturating_sub(slice_len).max(counter);
                self.memory
                    .run_deferred_jit(c.value.saturating_sub(begun), slice_len);
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
                // wouldn't have stored it (it's observable after a fault).
                pc = halt_pc;
                counter = c.value;
                continue;
            }
            self.cpu.regs.pc = halt_pc;
            return Ok(c);
        }
    }

    #[cfg(not(feature = "jit"))]
    fn dispatch_run(&mut self, pc: u64, counter: u64, max: u64) -> Result<Counter> {
        let (c, halt_pc) = self.dispatch_chain(pc, counter, max)?;
        self.cpu.regs.pc = halt_pc;
        Ok(c)
    }

    /// The dispatcher loop. Returns the final counters and the PC it halted
    /// at, which the caller stores.
    fn dispatch_chain(&mut self, pc: u64, counter: u64, max: u64) -> Result<(Counter, u64)> {
        let mut c = Counter {
            value: counter,
            max,
        };
        let mut pc = pc;
        // We need an execute segment matching the current PC.
        if !(pc >= self.cpu.exec.exec_begin() && pc < self.cpu.exec.exec_end()) {
            pc = self.next_execute_segment(pc)?;
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
            match handler(&exec, &exec.cache, i, self, c, e.instr, block_pc) {
                Exit::Halt => return Ok((self.dispatch.counter, self.dispatch.pc)),
                Exit::Jump => {
                    c = self.dispatch.counter;
                    if let Some(stopped) = self.stop_in_other_segment(self.dispatch.pc, c) {
                        return Ok(stopped);
                    }
                    pc = self.next_execute_segment(self.dispatch.pc)?;
                    // The new segment may be compiling in the background.
                    #[cfg(feature = "jit")]
                    if self.dispatch.jit_spawned && self.dispatch.slice.is_none() {
                        c.max = begin_slice(self, c.value, c.max);
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

    /// Precise handler for any instruction (C++ `CPU::execute(format_t)`).
    pub(crate) fn execute(&mut self, instr: Instr) -> Result<()> {
        (decode::decode(instr).exec)(self, instr)
    }
}
