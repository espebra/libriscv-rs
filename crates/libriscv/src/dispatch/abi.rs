//! Port of the handler conventions of `tailcall_dispatch.cpp`
//! (`INSTRUCTION`, `MUSTTAIL`, `preserve_none`, `NEXT_BLOCK` and friends).
//!
//! This is the only module whose code differs between the two dispatch
//! modes. Handlers in `dispatch.rs`, `bytecode.rs` and `fused.rs` are written once,
//! against a [`Frame`] and the transfer macros here, and compile to either:
//!
//! - **`nightly-dispatch`** (opt-in, pinned nightly): guaranteed tail calls
//!   with `become`, the `rust-preserve-none` ABI, and 64-byte aligned hot
//!   handlers. Every value of the frame lives in a register.
//! - **default** (stable Rust): ordinary calls in tail position, which LLVM
//!   turns into jumps (sibling calls) as long as all arguments fit in
//!   argument registers: at most six integer registers on x86-64. The
//!   counter's limit therefore lives in memory (`DispatchState::limit`,
//!   read only at limit checks), the block-end PC is recomputed from the
//!   entry index where needed, and the entry index and instruction bits share
//!   one register. Sibling calls are an optimization, not a guarantee: the
//!   dispatcher also bounds every chain with `MachineOptions::dispatch_slice`,
//!   so a call that does not become a jump only costs bounded stack.
//!
//! Handler bodies use only the [`Frame`] methods and fields documented as
//! shared, and transfer control only through the macros below.

use crate::decoder_cache::{DecodedSegment, DecoderEntry};
use crate::dispatch::{Counter, Exit};
use crate::machine::Machine;

// ---------------------------------------------------------------------------
// nightly-dispatch: `become`, preserve_none, everything in registers.
// ---------------------------------------------------------------------------

/// Shared handler signature: execute segment, decoder cache, entry index,
/// machine, counters, instruction bits, and block-ending PC.
/// Fast bytecodes use re-encoded bits; slow paths retain the original instruction.
/// Keeping the cache in registers avoids two loads per instruction.
///
/// Argument order places instruction bits in rcx under x86-64 `preserve_none`.
/// Shift immediates occupy the low bits, so variable shifts can use cl without
/// saving another live value. This reduced CoreMark rv64imafdc cycles by 4.5%
/// compared with counters in rdx/rcx.
#[cfg(feature = "nightly-dispatch")]
pub(crate) type Handler = extern "rust-preserve-none" fn(
    &DecodedSegment,
    &[DecoderEntry],
    usize,
    &mut Machine,
    Counter,
    u32,
    u64,
) -> Exit;

/// A handler's arguments (C++ `DecoderData* d`, `exec`, `cpu`, `pc`,
/// `counter`), as one value the handler body works with. Building it is
/// free: it is taken apart again for the next tail call.
///
/// Shared by both modes: `exec`, `cache` and `m`, and the methods.
#[cfg(feature = "nightly-dispatch")]
pub(crate) struct Frame<'a> {
    pub(crate) exec: &'a DecodedSegment,
    pub(crate) cache: &'a [DecoderEntry],
    pub(crate) i: usize,
    pub(crate) m: &'a mut Machine,
    pub(crate) c: Counter,
    pub(crate) ins: u32,
    pub(crate) pc: u64,
}

#[cfg(feature = "nightly-dispatch")]
impl Frame<'_> {
    /// This handler's decoder entry.
    #[inline(always)]
    pub(crate) const fn i(&self) -> usize {
        self.i
    }
    /// The entry's (re-encoded) instruction bits.
    #[inline(always)]
    pub(crate) const fn ins(&self) -> u32 {
        self.ins
    }
    /// The counter and its limit.
    #[inline(always)]
    pub(crate) const fn counter(&self) -> Counter {
        self.c
    }
    #[inline(always)]
    pub(crate) const fn overflowed(&self) -> bool {
        self.c.overflowed()
    }
    /// Continue with this counter and limit.
    #[inline(always)]
    pub(crate) const fn set_counter(&mut self, c: Counter) {
        self.c = c;
    }
    /// Add a block's instructions (`BEGIN_BLOCK`).
    #[inline(always)]
    pub(crate) const fn count(&mut self, n: u64) {
        self.c.value = self.c.value.wrapping_add(n);
    }
    /// The address of the current block's last instruction. Only
    /// block-ending handlers may use it: it is their own address.
    #[inline(always)]
    pub(crate) const fn end_pc(&self) -> u64 {
        self.pc
    }
    /// This instruction's address, from the block end (C++ `rv32i_auipc`).
    /// The entry exists: its handler is running.
    #[inline(always)]
    pub(crate) fn this_pc(&self) -> u64 {
        let to_end = self.cache.get(self.i).map_or(0, DecoderEntry::block_bytes);
        self.pc.wrapping_sub(to_end)
    }
}

/// Define a hot handler: `handler! { /// docs  fn name(f) { body } }`, with
/// an optional single generic parameter. The body sees the [`Frame`] as `f`.
#[cfg(feature = "nightly-dispatch")]
macro_rules! handler {
    ($(#[$attr:meta])* $vis:vis fn $name:ident $(<$g:ident: $b:path>)? ($f:ident) $body:block) => {
        $(#[$attr])*
        #[rustc_align(64)]
        $vis extern "rust-preserve-none" fn $name $(<$g: $b>)? (
            exec: &$crate::decoder_cache::DecodedSegment,
            cache: &[$crate::decoder_cache::DecoderEntry],
            i: usize,
            m: &mut $crate::machine::Machine,
            c: $crate::dispatch::Counter,
            ins: u32,
            pc: u64,
        ) -> $crate::dispatch::Exit {
            #[allow(unused_mut)]
            let mut $f = $crate::dispatch::abi::Frame { exec, cache, i, m, c, ins, pc };
            $body
        }
    };
}

/// [`handler!`] for cold slow paths: not aligned, never inlined.
#[cfg(feature = "nightly-dispatch")]
macro_rules! cold_handler {
    ($(#[$attr:meta])* $vis:vis fn $name:ident ($f:ident) $body:block) => {
        $(#[$attr])*
        #[cold]
        #[inline(never)]
        $vis extern "rust-preserve-none" fn $name(
            exec: &$crate::decoder_cache::DecodedSegment,
            cache: &[$crate::decoder_cache::DecoderEntry],
            i: usize,
            m: &mut $crate::machine::Machine,
            c: $crate::dispatch::Counter,
            ins: u32,
            pc: u64,
        ) -> $crate::dispatch::Exit {
            #[allow(unused_mut)]
            let mut $f = $crate::dispatch::abi::Frame { exec, cache, i, m, c, ins, pc };
            $body
        }
    };
}

/// C++ `MUSTTAIL return handler(...)`: continue with handler `$h` at entry
/// `$i`, with instruction bits `$ins` and block-end PC `$pc`.
#[cfg(feature = "nightly-dispatch")]
macro_rules! call {
    ($f:ident, $h:expr, $i:expr, $ins:expr, $pc:expr) => {
        become ($h)($f.exec, $f.cache, $i, $f.m, $f.c, $ins, $pc)
    };
}

/// Leave the chain through `$crate::dispatch::$exit` (`halt` or
/// `leave_segment`) at `$pc`. A tail call to a cold handler, so that fast
/// handlers make no calls and need no stack frame.
#[cfg(feature = "nightly-dispatch")]
macro_rules! exit {
    ($f:ident, halt, $pc:expr) => {
        become $crate::dispatch::halt_exit($f.exec, $f.cache, 0, $f.m, $f.c, 0, $pc)
    };
    ($f:ident, leave_segment, $pc:expr) => {
        become $crate::dispatch::leave_exit($f.exec, $f.cache, 0, $f.m, $f.c, 0, $pc)
    };
    ($f:ident, bad_entry, $pc:expr) => {
        become $crate::dispatch::bad_entry_exit($f.exec, $f.cache, 0, $f.m, $f.c, 0, $pc)
    };
}

/// Run the chain from entry `i` (handler `handler`), the block's counter
/// already added: the dispatcher's way in.
#[cfg(feature = "nightly-dispatch")]
#[inline(always)]
pub(crate) fn enter(
    handler: Handler,
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    c: Counter,
    ins: u32,
    pc: u64,
) -> Exit {
    handler(exec, &exec.cache, i, m, c, ins, pc)
}

// ---------------------------------------------------------------------------
// Stable: sibling calls within six argument registers.
// ---------------------------------------------------------------------------

/// Shared handler signature without `nightly-dispatch`: execute segment,
/// entry index, machine, instruction bits, and counter value. The bits are
/// fourth, in rcx under x86-64 SysV, where variable shifts want their count
/// (as in the `nightly-dispatch` signature). At most six
/// integer registers, the most x86-64 passes in registers: with any
/// argument on the stack, LLVM no longer turns the tail call into a jump.
/// So the limit is `DispatchState::limit`, read only at limit checks, and
/// handlers read the decoder cache's address and length from the segment.
/// Those loads depend only on the segment, not on the previous instruction.
/// (Packing the index and the instruction bits into one register to pass
/// the cache as well put the load of the next entry's bits on the chain of
/// dependent operations from one index to the next: CoreMark took ~1.9x the
/// cycles of `nightly-dispatch`.)
#[cfg(not(feature = "nightly-dispatch"))]
pub(crate) type Handler = fn(&DecodedSegment, usize, &mut Machine, u32, u64) -> Exit;

/// A handler's arguments (see the `nightly-dispatch` version).
#[cfg(not(feature = "nightly-dispatch"))]
pub(crate) struct Frame<'a> {
    pub(crate) exec: &'a DecodedSegment,
    pub(crate) cache: &'a [DecoderEntry],
    pub(crate) i: usize,
    pub(crate) m: &'a mut Machine,
    pub(crate) value: u64,
    pub(crate) ins: u32,
}

#[cfg(not(feature = "nightly-dispatch"))]
impl Frame<'_> {
    #[inline(always)]
    pub(crate) const fn i(&self) -> usize {
        self.i
    }
    #[inline(always)]
    pub(crate) const fn ins(&self) -> u32 {
        self.ins
    }
    #[inline(always)]
    pub(crate) fn counter(&self) -> Counter {
        Counter {
            value: self.value,
            max: self.m.dispatch.limit,
        }
    }
    #[inline(always)]
    pub(crate) fn overflowed(&self) -> bool {
        self.value >= self.m.dispatch.limit
    }
    #[inline(always)]
    pub(crate) fn set_counter(&mut self, c: Counter) {
        self.value = c.value;
        self.m.dispatch.limit = c.max;
    }
    #[inline(always)]
    pub(crate) const fn count(&mut self, n: u64) {
        self.value = self.value.wrapping_add(n);
    }
    /// The address of the current block's last instruction. Only
    /// block-ending handlers may use it: it is their own address, which is
    /// what this mode computes.
    #[inline(always)]
    pub(crate) fn end_pc(&self) -> u64 {
        debug_assert!(
            self.cache.get(self.i).is_none_or(|e| e.block_bytes() == 0),
            "end_pc in the middle of a block"
        );
        self.exec.address_of(self.i)
    }
    /// This instruction's address.
    #[inline(always)]
    pub(crate) const fn this_pc(&self) -> u64 {
        self.exec.address_of(self.i)
    }
}

#[cfg(not(feature = "nightly-dispatch"))]
macro_rules! handler {
    ($(#[$attr:meta])* $vis:vis fn $name:ident $(<$g:ident: $b:path>)? ($f:ident) $body:block) => {
        $(#[$attr])*
        $vis fn $name $(<$g: $b>)? (
            exec: &$crate::decoder_cache::DecodedSegment,
            i: usize,
            m: &mut $crate::machine::Machine,
            ins: u32,
            value: u64,
        ) -> $crate::dispatch::Exit {
            #[allow(unused_mut)]
            let mut $f = $crate::dispatch::abi::Frame { exec, cache: &exec.cache, i, m, value, ins };
            $body
        }
    };
}

#[cfg(not(feature = "nightly-dispatch"))]
macro_rules! cold_handler {
    ($(#[$attr:meta])* $vis:vis fn $name:ident ($f:ident) $body:block) => {
        $(#[$attr])*
        #[cold]
        #[inline(never)]
        $vis fn $name(
            exec: &$crate::decoder_cache::DecodedSegment,
            i: usize,
            m: &mut $crate::machine::Machine,
            ins: u32,
            value: u64,
        ) -> $crate::dispatch::Exit {
            #[allow(unused_mut)]
            let mut $f = $crate::dispatch::abi::Frame { exec, cache: &exec.cache, i, m, value, ins };
            $body
        }
    };
}

/// `$pc` is not passed (nor evaluated): handlers recompute it.
#[cfg(not(feature = "nightly-dispatch"))]
macro_rules! call {
    ($f:ident, $h:expr, $i:expr, $ins:expr, $pc:expr) => {
        return ($h)($f.exec, $i, $f.m, $ins, $f.value)
    };
}

/// Exits are plain calls in tail position: fewer arguments than a handler,
/// so they become jumps too.
#[cfg(not(feature = "nightly-dispatch"))]
macro_rules! exit {
    ($f:ident, bad_entry, $pc:expr) => {{
        let pc: u64 = $pc;
        return $crate::dispatch::bad_entry($f.m, pc);
    }};
    ($f:ident, $exit:ident, $pc:expr) => {{
        let pc: u64 = $pc;
        let c = $f.counter();
        return $crate::dispatch::$exit($f.m, pc, c);
    }};
}

#[cfg(not(feature = "nightly-dispatch"))]
#[inline(always)]
pub(crate) fn enter(
    handler: Handler,
    exec: &DecodedSegment,
    i: usize,
    m: &mut Machine,
    c: Counter,
    ins: u32,
    _pc: u64,
) -> Exit {
    m.dispatch.limit = c.max;
    handler(exec, i, m, ins, c.value)
}

// ---------------------------------------------------------------------------
// Shared by both modes.
// ---------------------------------------------------------------------------

/// Run fallible host code from a handler (an instruction's precise handler,
/// a system call), leaving a failure in the machine as a fault: `None` then,
/// and the handler returns [`Exit::Fault`].
///
/// Without `nightly-dispatch` this is never inlined, so the `Result` lives
/// in this function's frame. Once a stack value's address escapes to a call
/// (such as a large `Result` returned through memory), LLVM no longer turns
/// the handler's later calls into jumps. For the same reason `run` is a
/// `move` closure of a few register-sized values: one that borrows the
/// handler's locals makes their addresses escape.
#[cfg_attr(feature = "nightly-dispatch", inline(always))]
#[cfg_attr(not(feature = "nightly-dispatch"), inline(never))]
pub(crate) fn host<T>(
    m: &mut Machine,
    run: impl FnOnce(&mut Machine) -> crate::error::Result<T>,
) -> Option<T> {
    match run(m) {
        Ok(v) => Some(v),
        Err(err) => {
            m.dispatch.error = Some(err);
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Transfers, shared by both modes.
// ---------------------------------------------------------------------------

/// The entry at index `$i`, or leave the chain with a fault at `$pc`.
macro_rules! entry {
    ($f:ident, $i:expr, $pc:expr) => {
        match $f.cache.get($i) {
            Some(e) => e,
            None => $crate::dispatch::abi::exit!($f, bad_entry, $pc),
        }
    };
}

/// Continue with the same frame in handler `$h` (C++ calling another
/// handler of the same instruction, such as a slow path).
macro_rules! tail {
    ($f:ident, $h:expr) => {{
        let i = $f.i();
        let ins = $f.ins();
        $crate::dispatch::abi::call!($f, $h, i, ins, $f.pc)
    }};
}

/// C++ `EXECUTE_CURRENT`: tail-call the handler of entry `$i` in the same
/// block, handing it the entry's (re-encoded) instruction bits.
macro_rules! execute {
    ($f:ident, $i:expr) => {{
        let i: usize = $i;
        let e = $crate::dispatch::abi::entry!($f, i, $f.exec.address_of(i));
        // Same block: the block-end PC is unchanged (stable: not passed).
        $crate::dispatch::abi::call!($f, e.handler, i, e.instr, $f.pc)
    }};
}

/// C++ `NEXT_INSTR_TO(bc)`: continue at entry `$i` of the same block in
/// handler `$h`, known statically, instead of the entry's own. A fused
/// pair's handler continues in its second half this way (a direct jump),
/// while that entry keeps its own handler for when it is jumped to.
macro_rules! execute_to {
    ($f:ident, $h:expr, $i:expr) => {{
        let i: usize = $i;
        let e = $crate::dispatch::abi::entry!($f, i, $f.exec.address_of(i));
        $crate::dispatch::abi::call!($f, $h, i, e.instr, $f.pc)
    }};
}

/// C++ `BEGIN_BLOCK` + `EXECUTE_CURRENT`: enter the block at entry `$i`,
/// address `$pc`.
macro_rules! begin_block {
    ($f:ident, $i:expr, $pc:expr) => {{
        let i: usize = $i;
        let pc: u64 = $pc;
        let e = $crate::dispatch::abi::entry!($f, i, pc);
        $f.count(e.instruction_count());
        $crate::dispatch::abi::call!($f, e.handler, i, e.instr, pc.wrapping_add(e.block_bytes()))
    }};
}

/// Continue at entry `$i`, address `$pc`, in the middle of a block that is
/// counted already, with handler `$h` instead of the entry's own.
macro_rules! resume {
    ($f:ident, $h:expr, $i:expr, $pc:expr) => {{
        let i: usize = $i;
        let pc: u64 = $pc;
        let e = $crate::dispatch::abi::entry!($f, i, pc);
        $crate::dispatch::abi::call!($f, $h, i, e.instr, pc.wrapping_add(e.block_bytes()))
    }};
}

/// C++ `NEXT_BLOCK(len, OF)` (and `PERFORM_BRANCH`, which is
/// `NEXT_BLOCK(imm, true)`): continue `$len` bytes from the current block's
/// last instruction, checking the counter first if `$of`.
macro_rules! next_block {
    ($f:ident, $len:expr, $of:expr) => {{
        let len: i64 = $len;
        let pc: u64 = $f.end_pc().wrapping_add_signed(len);
        let i: usize = $f
            .i()
            .wrapping_add_signed((len >> $crate::decoder_cache::SHIFT) as isize);
        if $of && $f.overflowed() {
            $crate::dispatch::abi::exit!($f, halt, pc);
        }
        $crate::dispatch::abi::begin_block!($f, i, pc)
    }};
}

/// C++ `UNCHECKED_JUMP` (with its `QUICK_EXEC_CHECK`): continue at `$pc`,
/// leaving the chain if it is outside the current segment.
macro_rules! unchecked_jump {
    ($f:ident, $pc:expr) => {{
        let pc: u64 = $pc;
        if !(pc >= $f.exec.exec_begin() && pc < $f.exec.exec_end()) {
            $crate::dispatch::abi::exit!($f, leave_segment, pc);
        }
        $crate::dispatch::abi::begin_block!($f, $f.exec.index(pc), pc)
    }};
}

/// C++ `OVERFLOW_CHECKED_JUMP`.
macro_rules! overflow_checked_jump {
    ($f:ident, $pc:expr) => {{
        let pc: u64 = $pc;
        if $f.overflowed() {
            $crate::dispatch::abi::exit!($f, halt, pc);
        }
        $crate::dispatch::abi::unchecked_jump!($f, pc)
    }};
}

pub(crate) use {
    begin_block, call, cold_handler, entry, execute, execute_to, exit, handler, next_block,
    overflow_checked_jump, resume, tail, unchecked_jump,
};
