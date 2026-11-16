//! Port of `cpu.hpp`, `cpu.cpp`, `cpu_inline.hpp` and
//! `instruction_counter.hpp` (the counter lives in `dispatch::Counter`).
//!
//! C++ `CPU` holds a reference to its `Machine` and reaches memory through
//! it. Here the CPU is plain state, and operations that need memory too are
//! methods on [`Machine`].

use std::sync::Arc;

use crate::decoder_cache::{COMPRESSED, DecodedSegment};
use crate::dispatch::{self, Counter};
use crate::error::{ExceptionKind, MachineError, Result};
use crate::instruction::Instr;
use crate::machine::Machine;
use crate::memory::PAGE_SIZE;
use crate::registers::{REG_SP, Registers};

/// Registers plus the current execute segment.
#[derive(Debug)]
pub struct Cpu {
    pub(crate) regs: Registers,
    /// The current execute segment (an empty one before any code is loaded).
    pub(crate) exec: Arc<DecodedSegment>,
    /// Guards against no-progress execute segment rebuild loops.
    stale_restart_pc: u64,
    /// Bytes execute segments may still be decoded from at runtime without
    /// a charge (`MachineOptions::free_decode_bytes`).
    free_decode_bytes: u64,
}

impl Cpu {
    pub(crate) fn new(free_decode_bytes: u64) -> Self {
        Self {
            regs: Registers::default(),
            exec: Arc::new(DecodedSegment::empty()),
            stale_restart_pc: u64::MAX,
            free_decode_bytes,
        }
    }

    /// Copy registers and the current execute segment for a fork.
    pub(crate) fn fork(&self) -> Self {
        Self {
            regs: self.regs.clone(),
            exec: Arc::clone(&self.exec),
            stale_restart_pc: self.stale_restart_pc,
            free_decode_bytes: self.free_decode_bytes,
        }
    }

    pub const fn registers(&self) -> &Registers {
        &self.regs
    }
    pub const fn registers_mut(&mut self) -> &mut Registers {
        &mut self.regs
    }
    pub const fn reg(&self, idx: u32) -> u64 {
        self.regs.get(idx)
    }
    pub const fn set_reg(&mut self, idx: u32, value: u64) {
        self.regs.set(idx, value);
    }
    pub const fn pc(&self) -> u64 {
        self.regs.pc
    }

    /// Port of `CPU::jump`: set PC, faulting on a misaligned address.
    pub fn jump(&mut self, dst: u64) -> Result<()> {
        let mask = if COMPRESSED { 0x1 } else { 0x3 };
        if dst & mask != 0 {
            return Err(MachineError::trigger(
                ExceptionKind::MisalignedInstruction,
                dst,
            ));
        }
        self.regs.pc = dst;
        Ok(())
    }

    /// Port of `CPU::aligned_jump`: set PC without checks.
    pub const fn aligned_jump(&mut self, dst: u64) {
        self.regs.pc = dst;
    }

    pub const fn increment_pc(&mut self, delta: i64) {
        self.regs.pc = self.regs.pc.wrapping_add_signed(delta);
    }

    /// The current execute segment.
    pub fn current_execute_segment(&self) -> &DecodedSegment {
        &self.exec
    }

    /// True if `addr` is in the current execute segment.
    pub fn is_executable(&self, addr: u64) -> bool {
        self.exec.is_within(addr, 2)
    }
}

impl Machine {
    /// Port of `CPU::reset`: clear the registers, set up the stack pointer
    /// and jump to the program's entry point.
    pub fn reset_cpu(&mut self) -> Result<()> {
        self.cpu.regs = Registers::default();
        self.cpu.regs.set(REG_SP, self.memory.stack_initial());
        // Resetting an unloaded machine must not jump to an ELF entry point.
        if !self.cpu.exec.is_empty() {
            let initial_pc = self.memory.start_address();
            // Check that the initial PC is executable.
            if !self.cpu.exec.is_within(initial_pc, 2) {
                let (attr, _) = self.memory.get_exec_pageno(initial_pc / PAGE_SIZE)?;
                if !attr.exec {
                    return Err(MachineError::trigger(
                        ExceptionKind::ExecutionSpaceProtectionFault,
                        initial_pc,
                    ));
                }
            }
            self.cpu.jump(initial_pc)?;
        }
        Ok(())
    }

    /// Port of `CPU::init_execute_area`: decode `code` as a new execute
    /// segment at `begin`, and make it the current one.
    ///
    /// Like C++, this does not copy the code into guest memory.
    pub fn init_execute_area(&mut self, code: &[u8], begin: u64) -> Result<&DecodedSegment> {
        self.init_execute_area_from(code, begin, None)
    }

    /// [`init_execute_area`](Self::init_execute_area) for a segment
    /// replacing the stale segment `rebuilt_from`.
    fn init_execute_area_from(
        &mut self,
        code: &[u8],
        begin: u64,
        rebuilt_from: Option<&DecodedSegment>,
    ) -> Result<&DecodedSegment> {
        if code.len() < 4 {
            return Err(MachineError::trigger(
                ExceptionKind::ExecutionSpaceProtectionFault,
                begin,
            ));
        }
        self.cpu.exec =
            self.memory
                .create_execute_segment(code, begin, false, false, rebuilt_from)?;
        // Compiling in the background: the dispatcher has to run in slices
        // to switch to the native code (`dispatch_sliced`).
        // So does one deferred until it is hot (to count towards it).
        #[cfg(feature = "jit")]
        if self.cpu.exec.jit_outstanding() || self.cpu.exec.jit_deferred() {
            self.dispatch.jit_spawned = true;
        }
        Ok(&self.cpu.exec)
    }

    /// Select the execute segment containing `pc` (C++ `CPU::next_execute_segment`).
    /// Missing or stale segments are decoded from adjacent executable pages;
    /// non-executable targets fault. Supports runtime mappings and self-modifying code.
    ///
    /// Returns the continuation PC and runtime decoding charge; see
    /// [`MachineOptions::free_decode_bytes`](crate::MachineOptions::free_decode_bytes).
    /// The caller adds the charge and checks its instruction limit.
    pub(crate) fn next_execute_segment(&mut self, pc: u64) -> Result<(u64, u64)> {
        if let Some(segment) = self.memory.exec_segment_ref(pc)
            && !segment.is_stale()
        {
            if !Arc::ptr_eq(segment, &self.cpu.exec) {
                self.cpu.exec = Arc::clone(segment);
            }
            // Reached a usable segment, reset the no-progress guard.
            self.cpu.stale_restart_pc = u64::MAX;
            return Ok((pc, 0));
        }
        let found = self.memory.exec_segment_for(pc);
        // Read-fault handlers could be reading PC.
        self.cpu.regs.pc = pc;
        self.cpu.exec = found
            .clone()
            .unwrap_or_else(|| Arc::new(DecodedSegment::empty()));

        // Look at the page first, to verify that it is executable. (The
        // default execute fault handler raises the fault; custom handlers
        // and page traps are not ported yet.)
        let pageno = pc / PAGE_SIZE;
        if !self.memory.get_pageno(pageno)?.0.exec {
            return Err(MachineError::trigger(
                ExceptionKind::ExecutionSpaceProtectionFault,
                pc,
            ));
        }

        // The stale segment, compiled if a background compilation finished:
        // the rebuild may reuse its native code.
        #[cfg(feature = "jit")]
        let rebuilt_from = found.as_ref().map(DecodedSegment::latest);
        #[cfg(not(feature = "jit"))]
        let rebuilt_from: Option<Arc<DecodedSegment>> = None;
        // Evict stale execute segments.
        if let Some(segment) = found {
            // If this PC was rebuilt already and the segment is stale again,
            // trap instead of spinning forever.
            if pc == self.cpu.stale_restart_pc {
                return Err(MachineError::trigger(
                    ExceptionKind::ExecutionLoopDetected,
                    pc,
                ));
            }
            self.cpu.stale_restart_pc = pc;
            self.memory.evict_execute_segment(&segment);
        }

        // Decode every executable page around PC as one new segment.
        let max_pages = self.memory.max_exec_pages();
        let Some((base, code)) = self.memory.executable_area(pageno, max_pages)? else {
            return Err(MachineError::trigger(
                ExceptionKind::ExecutionSpaceProtectionFault,
                pc,
            ));
        };
        let segment =
            self.init_execute_area_from(&code, base * PAGE_SIZE, rebuilt_from.as_deref())?;
        // The rebuild produced a usable segment, so re-arm the guard.
        if !segment.is_stale() {
            self.cpu.stale_restart_pc = u64::MAX;
        }
        // Charge the decoded bytes past the free allowance (new segments
        // and rebuilds alike): a quarter instruction per byte, as
        // `flush_execute_segments` charges.
        let bytes = code.len() as u64;
        let free = bytes.min(self.cpu.free_decode_bytes);
        self.cpu.free_decode_bytes -= free;
        Ok((pc, (bytes - free) / 4))
    }

    /// Handle a jump to a block-start STOP in another usable segment without
    /// switching segments. Returns the same result as `BEGIN_BLOCK` followed by
    /// [`dispatch::stop`], avoiding `Arc` updates on VM call returns.
    /// Returns `None` if switching or executing the block requires other work.
    pub(crate) fn stop_in_other_segment(&mut self, pc: u64, c: Counter) -> Option<(Counter, u64)> {
        let exec = self.memory.exec_segment_ref(pc)?;
        if exec.is_stale() {
            return None;
        }
        let i = exec.index(pc);
        let e = exec.cache.get(i)?;
        if !std::ptr::fn_addr_eq(e.handler, dispatch::stop as dispatch::Handler)
            || exec.native.as_ref().is_some_and(|n| n.has_entry(i))
        {
            return None;
        }
        // As next_execute_segment does on reaching a usable segment.
        self.cpu.stale_restart_pc = u64::MAX;
        let block_pc = pc.wrapping_add(e.block_bytes());
        let value = c.value.wrapping_add(e.instruction_count());
        Some((Counter { value, max: 0 }, block_pc.wrapping_add(4)))
    }

    /// Port of `CPU::simulate_precise`: execute one instruction at a time
    /// with the precise handlers, counting every instruction, until the
    /// machine's counter reaches its limit. Slow; for debugging and testing.
    pub fn simulate_precise(&mut self) -> Result<()> {
        // Always have at least the current segment.
        if !self.cpu.is_executable(self.cpu.regs.pc) {
            let (_, charge) = self.next_execute_segment(self.cpu.regs.pc)?;
            self.penalize(charge);
        }
        while self.counter < self.max_counter {
            let mut pc = self.cpu.regs.pc;
            if !self.cpu.exec.is_within(pc, 2) || self.cpu.exec.is_stale() {
                // Faults if the new location is not executable.
                let charge;
                (pc, charge) = self.next_execute_segment(pc)?;
                if charge > 0 {
                    self.penalize(charge);
                    if self.counter >= self.max_counter {
                        self.cpu.regs.pc = pc;
                        break;
                    }
                }
            }
            let instr = Instr(self.cpu.exec.read_u32(pc));
            self.execute(instr)?;
            let len = if COMPRESSED {
                u64::from(instr.length())
            } else {
                4
            };
            self.cpu.regs.pc = self.cpu.regs.pc.wrapping_add(len);
            self.counter = self.counter.wrapping_add(1);
        }
        Ok(())
    }

    /// Port of `CPU::read_next_instruction_slowpath` with paging: read the
    /// instruction at `pc` directly from executable page memory.
    fn read_next_instruction_slowpath(&self, pc: u64) -> Result<Instr> {
        let pageno = pc / PAGE_SIZE;
        let (attr, data) = self.memory.get_exec_pageno(pageno)?;
        if !attr.exec {
            return Err(MachineError::trigger(
                ExceptionKind::ExecutionSpaceProtectionFault,
                pc,
            ));
        }
        let off = (pc % PAGE_SIZE) as usize;
        if let Some(b) = data.get(off..).and_then(|s| s.first_chunk::<4>()) {
            return Ok(Instr(u32::from_le_bytes(*b)));
        }
        // Only 16 bits are left of the page (PC is at least 2-aligned).
        let lo = data
            .get(off..)
            .and_then(|s| s.first_chunk::<2>())
            .map_or(0, |b| u16::from_le_bytes(*b));
        let mut instr = Instr(u32::from(lo));
        // A 32-bit instruction at a page border: the upper half is on the
        // next page (which C++ does not check for execute permission).
        if instr.is_long() {
            let (_, next) = self.memory.get_exec_pageno(pageno.wrapping_add(1))?;
            let hi = next
                .first_chunk::<2>()
                .map_or(0, |b| u16::from_le_bytes(*b));
            instr = Instr(u32::from(lo) | u32::from(hi) << 16);
        }
        Ok(instr)
    }

    /// Port of `CPU::step_one`: execute the instruction at PC precisely.
    pub fn step_one(&mut self, use_instruction_counter: bool) -> Result<()> {
        let pc = self.cpu.regs.pc;
        let instr = if self.cpu.exec.is_within(pc, 2) {
            Instr(self.cpu.exec.read_u32(pc))
        } else {
            self.read_next_instruction_slowpath(pc)?
        };
        self.execute(instr)?;
        let len = if COMPRESSED {
            u64::from(instr.length())
        } else {
            4
        };
        self.cpu.regs.pc = self.cpu.regs.pc.wrapping_add(len);
        self.counter = self
            .counter
            .wrapping_add(u64::from(use_instruction_counter));
        Ok(())
    }
}
