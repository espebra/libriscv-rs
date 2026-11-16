//! Port of `machine_vmcall.hpp`, `CPU::preempt_internal` (`cpu.cpp`) and
//! the call helpers of `machine_inline.hpp` / `cpu_inline.hpp`
//! (`realign_stack`, `reset_stack_pointer`): calling guest functions from
//! the host.
//!
//! C++ passes arguments as a variadic template pack and picks what to do
//! with each one by type. Here the arguments are a tuple of [`VmCallArg`]s
//! (a single argument does not need the tuple, `()` is none), with the same
//! per-type rules:
//!
//! | Argument | C++ | Goes to |
//! |---|---|---|
//! | integers, `bool`, `char` | integral | the next integer register (sign- or zero-extended) |
//! | `f32`, `f64` | `float`, `double` | the next FP register |
//! | `str`, `String` | `const char*`, `std::string` | pushed with its NUL, address in the next integer register |
//! | `[T]`, `[T; N]`, `Vec<T>` of [`GuestPod`] | `std::vector`, `std::array` | the elements pushed, address in the next integer register |
//! | [`OnStack`] of a [`GuestPod`] | other standard-layout types | pushed, address in the next integer register |
//!
//! The C++ template parameters become methods: `vmcall` (no limit),
//! `vmcall_limited` (`vmcall<MAXI>`), `try_vmcall` (`Throw = false`),
//! `preempt` (`StoreRegs = true`), `preempt_clobbering`
//! (`StoreRegs = false`) and `try_preempt` (`Throw = false`).
//!
//! Like C++, arguments past `a7`/`fa7` keep counting up the register file
//! (`s2`, `s3`, ... and `ft8`, ...); more than that is an error here (C++
//! writes past the register file).

use crate::error::{ExceptionKind, MachineError, Result};
use crate::machine::Machine;
use crate::memory::GuestPod;
use crate::registers::{REG_ARG0, REG_RA, REG_SP};

/// C++ `REG_FA0`.
const REG_FA0: u32 = 10;
/// Registers past this are out of the register file.
const REG_LAST: u32 = 31;

/// The registers the next arguments go to (C++ `iarg`/`farg` in
/// `setup_call`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArgRegs {
    iarg: u32,
    farg: u32,
}

impl Default for ArgRegs {
    fn default() -> Self {
        Self {
            iarg: REG_ARG0,
            farg: REG_FA0,
        }
    }
}

fn too_many_args() -> MachineError {
    MachineError::new(
        ExceptionKind::IllegalOperation,
        "Too many arguments for a VM call",
        0,
    )
}

impl ArgRegs {
    /// Pass `value` in the next integer register.
    pub fn int(&mut self, m: &mut Machine, value: u64) -> Result<()> {
        if self.iarg > REG_LAST {
            return Err(too_many_args());
        }
        m.cpu.set_reg(self.iarg, value);
        self.iarg += 1;
        Ok(())
    }

    /// Pass `value` in the next FP register (C++ `set_float`: the low 32
    /// bits, NaN-boxed only in the `debug-checks` build).
    pub fn float(&mut self, m: &mut Machine, value: f32) -> Result<()> {
        if self.farg > REG_LAST {
            return Err(too_many_args());
        }
        m.cpu.regs.load_fp_u32(self.farg, value.to_bits());
        self.farg += 1;
        Ok(())
    }

    /// Pass `value` in the next FP register (C++ `f64 = value`).
    pub fn double(&mut self, m: &mut Machine, value: f64) -> Result<()> {
        if self.farg > REG_LAST {
            return Err(too_many_args());
        }
        m.cpu.regs.set_fp_bits(self.farg, value.to_bits());
        self.farg += 1;
        Ok(())
    }

    /// Push `data` onto the guest stack and pass its address in the next
    /// integer register.
    pub fn push(&mut self, m: &mut Machine, data: &[u8]) -> Result<()> {
        let addr = m.stack_push(data)?;
        self.int(m, addr)
    }
}

/// One VM call argument. Implement this trait for custom argument types;
/// see the module documentation for built-in implementations.
pub trait VmCallArg {
    /// Put the argument where the callee expects it.
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()>;
}

/// The arguments of a VM call: a tuple of up to 12 [`VmCallArg`]s, a single
/// one, or `()` for none.
pub trait VmCallArgs {
    /// Pass every argument, in order.
    fn pass_all(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()>;
}

impl<T: VmCallArg + ?Sized> VmCallArg for &T {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        (**self).pass(m, regs)
    }
}

macro_rules! int_args {
    ($($t:ty),*) => {$(
        impl VmCallArg for $t {
            fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
                // C++ assigns to an address-sized register: signed types
                // sign-extend, unsigned ones zero-extend.
                #[allow(clippy::cast_sign_loss, clippy::cast_lossless)]
                regs.int(m, *self as u64)
            }
        }
    )*};
}
int_args!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl VmCallArg for bool {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.int(m, u64::from(*self))
    }
}
impl VmCallArg for char {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.int(m, u64::from(*self))
    }
}
impl VmCallArg for f32 {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.float(m, *self)
    }
}
impl VmCallArg for f64 {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.double(m, *self)
    }
}

/// C++ `stack_push(str, strlen(str) + 1)`.
impl VmCallArg for str {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        let addr = m.stack_push_str(self)?;
        regs.int(m, addr)
    }
}
impl VmCallArg for String {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        self.as_str().pass(m, regs)
    }
}

fn pod_bytes<T: GuestPod>(values: &[T]) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        v.write_guest(&mut out);
    }
    out
}

/// C++ `stack_push(vec.data(), vec.size() * sizeof(vec[0]))`.
impl<T: GuestPod> VmCallArg for [T] {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        regs.push(m, &pod_bytes(self))
    }
}
impl<T: GuestPod, const N: usize> VmCallArg for [T; N] {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        self.as_slice().pass(m, regs)
    }
}
impl<T: GuestPod> VmCallArg for Vec<T> {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        self.as_slice().pass(m, regs)
    }
}

/// Pass a [`GuestPod`] by pointer: its bytes are pushed onto the guest
/// stack and the address is the argument (C++ does this for any
/// standard-layout argument that is not a number).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnStack<T>(pub T);

impl<T: GuestPod> VmCallArg for OnStack<T> {
    fn pass(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        let mut out = Vec::new();
        self.0.write_guest(&mut out);
        regs.push(m, &out)
    }
}

impl<T: VmCallArg> VmCallArgs for T {
    fn pass_all(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
        self.pass(m, regs)
    }
}

macro_rules! tuple_args {
    ($($name:ident)*) => {
        impl<$($name: VmCallArg),*> VmCallArgs for ($($name,)*) {
            #[allow(non_snake_case, unused_variables)]
            fn pass_all(&self, m: &mut Machine, regs: &mut ArgRegs) -> Result<()> {
                let ($($name,)*) = self;
                $($name.pass(m, regs)?;)*
                Ok(())
            }
        }
    };
}
tuple_args!();
tuple_args!(A);
tuple_args!(A B);
tuple_args!(A B C);
tuple_args!(A B C D);
tuple_args!(A B C D E);
tuple_args!(A B C D E F);
tuple_args!(A B C D E F G);
tuple_args!(A B C D E F G H);
tuple_args!(A B C D E F G H I);
tuple_args!(A B C D E F G H I J);
tuple_args!(A B C D E F G H I J K);
tuple_args!(A B C D E F G H I J K L);

/// VM call target: a guest address or ELF symbol name.
/// Symbol names are resolved on every call. Cache [`Machine::address_of`]
/// for repeated calls. An unknown symbol resolves to address 0 and faults
/// when executed, matching C++.
pub trait CallTarget {
    /// The function's address.
    fn address(&self, m: &Machine) -> Result<u64>;
}
impl CallTarget for u64 {
    fn address(&self, _: &Machine) -> Result<u64> {
        Ok(*self)
    }
}
impl CallTarget for &str {
    fn address(&self, m: &Machine) -> Result<u64> {
        m.memory.resolve_address(self)
    }
}
impl CallTarget for String {
    fn address(&self, m: &Machine) -> Result<u64> {
        m.memory.resolve_address(self)
    }
}
impl CallTarget for &String {
    fn address(&self, m: &Machine) -> Result<u64> {
        m.memory.resolve_address(self)
    }
}

impl Machine {
    /// Port of `CPU::reset_stack_pointer`: back to the initial stack
    /// pointer (below argv, envp and the auxiliary vector).
    pub const fn reset_stack_pointer(&mut self) {
        self.cpu.set_reg(REG_SP, self.memory.stack_initial());
    }

    /// Port of `Machine::realign_stack`: 16-byte align the stack pointer, as
    /// the RISC-V calling convention requires.
    pub const fn realign_stack(&mut self) {
        let sp = self.cpu.reg(REG_SP) & !0xF;
        self.cpu.set_reg(REG_SP, sp);
    }

    /// Port of `Machine::setup_call`: return address to the exit function,
    /// arguments into registers and onto the stack, stack realigned.
    pub fn setup_call(&mut self, args: &impl VmCallArgs) -> Result<()> {
        let exit = self.memory.exit_address();
        self.cpu.set_reg(REG_RA, exit);
        args.pass_all(self, &mut ArgRegs::default())?;
        self.realign_stack();
        Ok(())
    }

    /// Call a guest function with a fresh stack and return its `a0` value.
    /// The function returns through the exit function, which stops the machine.
    ///
    /// Execution has no instruction limit or counting (C++ `simulate_inaccurate`).
    /// An infinite loop blocks the calling host thread. Use only with trusted code;
    /// for untrusted code, use [`vmcall_limited`](Self::vmcall_limited) or
    /// [`try_vmcall`](Self::try_vmcall) with a nonzero limit.
    /// The machine must be idle; use [`preempt`](Self::preempt) from a system call.
    pub fn vmcall(&mut self, func: impl CallTarget, args: impl VmCallArgs) -> Result<u64> {
        self.vmcall_limited(u64::MAX, func, args)
    }

    /// Call with an instruction limit counted from zero.
    /// Reaching the limit returns `MAX_INSTRUCTIONS_REACHED` (C++ `vmcall<MAXI>`).
    ///
    /// Both 0 and `u64::MAX` disable counting and the limit, matching
    /// [`vmcall`](Self::vmcall). Check exhausted budgets before calling:
    /// passing 0 runs untrusted code without a limit.
    pub fn vmcall_limited(
        &mut self,
        max_instructions: u64,
        func: impl CallTarget,
        args: impl VmCallArgs,
    ) -> Result<u64> {
        let pc = self.prepare_vmcall(&func, &args)?;
        if max_instructions == u64::MAX || max_instructions == 0 {
            self.simulate_inaccurate(pc)?;
        } else {
            self.simulate_or_timeout(max_instructions, 0, pc)?;
        }
        Ok(self.return_value())
    }

    /// Call with a limit, returning the current `a0` on timeout
    /// (C++ `vmcall<MAXI, false>`). Use
    /// [`instruction_limit_reached`](Self::instruction_limit_reached) to detect timeout.
    ///
    /// Both 0 and `u64::MAX` disable the limit, as with
    /// [`vmcall_limited`](Self::vmcall_limited). Do not pass 0 for untrusted code.
    pub fn try_vmcall(
        &mut self,
        max_instructions: u64,
        func: impl CallTarget,
        args: impl VmCallArgs,
    ) -> Result<u64> {
        let pc = self.prepare_vmcall(&func, &args)?;
        if max_instructions == u64::MAX || max_instructions == 0 {
            self.simulate_inaccurate(pc)?;
        } else {
            self.simulate_with(max_instructions, 0, pc)?;
        }
        Ok(self.return_value())
    }

    fn prepare_vmcall(&mut self, func: &impl CallTarget, args: &impl VmCallArgs) -> Result<u64> {
        let pc = func.address(self)?;
        // Reset the stack pointer to its initial location (deliberately).
        self.reset_stack_pointer();
        self.setup_call(args)?;
        Ok(pc)
    }

    /// Call a guest function on the current stack below a 16-byte red zone
    /// (C++ `Machine::preempt`, `StoreRegs = true`). Restore all registers,
    /// including PC, after return or execution failure. Returns the function's `a0`.
    /// Can be called from a system call handler.
    ///
    /// The limit is `max_instructions` beyond the current counter. Executed
    /// instructions also consume the caller's budget. Reaching the limit returns
    /// `MAX_INSTRUCTIONS_REACHED`.
    pub fn preempt(
        &mut self,
        max_instructions: u64,
        func: impl CallTarget,
        args: impl VmCallArgs,
    ) -> Result<u64> {
        self.preempt_impl(max_instructions, &func, &args, true, true)
    }

    /// Preempt without saving or restoring registers (C++ `preempt<true, false>`).
    /// Registers and PC retain the called function's final state. From a system
    /// call handler, execution resumes 4 bytes after that PC. The caller must
    /// restore any state needed to resume interrupted code.
    pub fn preempt_clobbering(
        &mut self,
        max_instructions: u64,
        func: impl CallTarget,
        args: impl VmCallArgs,
    ) -> Result<u64> {
        self.preempt_impl(max_instructions, &func, &args, false, true)
    }

    /// Preempt while discarding execution faults and timeouts
    /// (C++ `preempt<false, StoreRegs>`). If `store_regs` is true, restore
    /// registers before returning `a0`; otherwise return the failed call's `a0`.
    ///
    /// Argument setup can still fail. Setup errors leave partially modified
    /// registers unrestored, matching all other call variants and C++.
    pub fn try_preempt(
        &mut self,
        max_instructions: u64,
        func: impl CallTarget,
        args: impl VmCallArgs,
        store_regs: bool,
    ) -> Result<u64> {
        self.preempt_impl(max_instructions, &func, &args, store_regs, false)
    }

    fn preempt_impl(
        &mut self,
        max_instructions: u64,
        func: &impl CallTarget,
        args: &impl VmCallArgs,
        store_regs: bool,
        forward_errors: bool,
    ) -> Result<u64> {
        let pc = func.address(self)?;
        let old_regs = store_regs.then(|| self.cpu.regs.clone());
        // Reserve stack space for the preempted call.
        let sp = self.cpu.reg(REG_SP).wrapping_sub(16);
        self.cpu.set_reg(REG_SP, sp);
        // If pushing an argument fails, the registers keep the partial setup
        // (SP, RA, the arguments so far), as in C++.
        self.setup_call(args)?;
        self.preempt_internal(old_regs, forward_errors, pc, max_instructions)
    }

    /// Port of `CPU::preempt_internal`.
    fn preempt_internal(
        &mut self,
        old_regs: Option<crate::registers::Registers>,
        forward_errors: bool,
        pc: u64,
        max_instructions: u64,
    ) -> Result<u64> {
        let prev_max = self.max_counter;
        // Execute by extending the max instruction counter (resuming).
        // Resume the existing counter to preserve the caller's instruction budget.
        // Repeatedly invoking preemption must lead to timeouts on either
        // preempt() or the caller.
        let counter = self.counter;
        let result = self.simulate_or_timeout(counter.wrapping_add(max_instructions), counter, pc);
        self.max_counter = prev_max;
        if let Err(err) = result {
            if let Some(regs) = &old_regs {
                self.cpu.regs = regs.clone();
            }
            if forward_errors {
                return Err(err);
            }
        }
        // Restore registers and return the return value.
        let retval = self.cpu.reg(REG_ARG0);
        if let Some(regs) = old_regs {
            self.cpu.regs = regs;
        }
        Ok(retval)
    }
}
