//! Port of `threaded_bytecodes.hpp` (the re-encoded operand formats) and
//! `bytecode_impl.cpp` (the fast-path bytecode handlers).
//!
//! When a segment is decoded, `decode::fast_handler` picks a bytecode
//! handler for each instruction and re-encodes the instruction's operands
//! into one of the `Faster*` formats below (C++ `threaded_rewrite`). The
//! re-encoded bits are stored in the decoder entry, and the dispatcher hands
//! them to the handler as its `ins` argument. So a handler never decodes an
//! instruction, never re-reads its own decoder entry, and never checks
//! anything the rewriter already checked: `rd != x0`, branch targets inside
//! the segment, shift amounts in range.
//!
//! Like C++, the handlers have no error paths of their own. The only
//! fallible bytecodes are the loads and stores, and their fast path, the flat
//! arena, cannot fail. Anything else (the page table, faults) is a separate
//! cold handler that the fast one tail-calls, so the fast handlers carry
//! no `Result` and need no stack frame.
//!
//! The bytecodes of fused pairs (`fused.rs`) also exist as marker types
//! ([`Bytecode`], [`First`]), like C++ `bytecode_bodies.hpp`: a fused handler
//! runs the body of the first and continues directly in the handler of the
//! second. The macros below expand each body into both the handler and the
//! marker, so it is still written once.

use std::marker::PhantomData;

use crate::decoder_cache::COMPRESSED;
use crate::dispatch::abi::{
    Frame, cold_handler, execute, handler, next_block, overflow_checked_jump, tail,
};
use crate::dispatch::{Handler, STEP, fault};
use crate::error::{ExceptionKind, MachineError};
use crate::isa::rvi::{Add, Addw, Mul, RtypeAlu, Sh1addUw, Sub, sext32};
use crate::machine::Machine;
use crate::registers::REG_RA;

/// A bytecode handler as a type: the second half of a fused pair, which the
/// fused handler continues in directly (C++ `NEXT_INSTR_TO(B)`).
pub(crate) trait Bytecode {
    const HANDLER: Handler;
}

/// A bytecode that can be the first half of a fused pair: its body without
/// the transfer to the next instruction (C++ `bytecode_body`). Only
/// sequential bytecodes qualify.
pub(crate) trait First: Bytecode {
    /// Decoder entries to the next instruction: [`STEP`] (C++
    /// `NEXT_INSTR`) or 1 for compressed instructions (`NEXT_C_INSTR`).
    const STEP: usize;
    /// Where the instruction continues, with the same frame, when [`run`]
    /// declines it: the slow path of a load or store outside the arena,
    /// which then dispatches the next entry itself.
    ///
    /// [`run`]: First::run
    const MISS: Handler;
    /// Execute the instruction. `false` (only loads and stores) means
    /// nothing happened yet: continue in [`First::MISS`].
    fn run(f: &mut Frame<'_>) -> bool;
}

/// [`Bytecode`] markers for the block-ending second halves of fused pairs.
macro_rules! markers {
    ($($marker:ident => $h:expr,)*) => {$(
        pub(crate) struct $marker;
        impl Bytecode for $marker {
            const HANDLER: Handler = $h;
        }
    )*};
}

/// C++ `FasterItype`: `imm:16 | rs2:8 | rs1:8`. For most bytecodes `rs1` is
/// the destination and `rs2` the source; for branches and stores they are
/// the two source registers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FasterItype(pub u32);

impl FasterItype {
    pub(crate) const fn encode(rs1: u32, rs2: u32, imm: i32) -> u32 {
        (imm as u32 & 0xFFFF) | ((rs2 & 0xFF) << 16) | ((rs1 & 0xFF) << 24)
    }
    #[inline(always)]
    const fn rs1(self) -> u32 {
        self.0 >> 24
    }
    #[inline(always)]
    const fn rs2(self) -> u32 {
        (self.0 >> 16) & 0xFF
    }
    #[inline(always)]
    const fn signed_imm(self) -> i64 {
        self.0 as u16 as i16 as i64
    }
    #[inline(always)]
    const fn unsigned_imm(self) -> u32 {
        self.0 & 0xFFFF
    }
}

/// C++ `FasterOpType`: `rd:16 | rs2:8 | rs1:8`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FasterOpType(pub u32);

impl FasterOpType {
    pub(crate) const fn encode(rd: u32, rs1: u32, rs2: u32) -> u32 {
        (rd & 0xFFFF) | ((rs2 & 0xFF) << 16) | ((rs1 & 0xFF) << 24)
    }
    #[inline(always)]
    const fn rd(self) -> u32 {
        self.0 & 0xFFFF
    }
    #[inline(always)]
    const fn rs1(self) -> u32 {
        self.0 >> 24
    }
    #[inline(always)]
    const fn rs2(self) -> u32 {
        (self.0 >> 16) & 0xFF
    }
}

/// C++ `FasterImmediate`: `rd:8 | (padding):8 | imm:16`. C++ leaves the
/// padding byte uninitialised; it is zero here.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FasterImmediate(pub u32);

impl FasterImmediate {
    pub(crate) const fn encode(rd: u32, imm: i32) -> u32 {
        (rd & 0xFF) | ((imm as u32 & 0xFFFF) << 16)
    }
    #[inline(always)]
    const fn rd(self) -> u32 {
        self.0 & 0xFF
    }
    #[inline(always)]
    const fn signed_imm(self) -> i64 {
        (self.0 >> 16) as u16 as i16 as i64
    }
}

/// C++ `FasterMove`: `rs1:8 | rd:8` (the upper half uninitialised in C++,
/// zero here).
#[derive(Debug, Clone, Copy)]
pub(crate) struct FasterMove(pub u32);

impl FasterMove {
    pub(crate) const fn encode(rd: u32, rs1: u32) -> u32 {
        (rs1 & 0xFF) | ((rd & 0xFF) << 8)
    }
    #[inline(always)]
    const fn rd(self) -> u32 {
        (self.0 >> 8) & 0xFF
    }
    #[inline(always)]
    const fn rs1(self) -> u32 {
        self.0 & 0xFF
    }
}

/// C++ `FasterJtype`: `offset:24 | rd:8`. JAL keeps its offset there; LUI
/// and AUIPC keep `imm << 4`, so that `offset << 8` is the upper immediate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FasterJtype(pub u32);

impl FasterJtype {
    pub(crate) const fn encode(offset: u32, rd: u32) -> u32 {
        (offset & 0xFF_FFFF) | ((rd & 0xFF) << 24)
    }
    #[inline(always)]
    const fn rd(self) -> u32 {
        self.0 >> 24
    }
    #[inline(always)]
    const fn signed_imm(self) -> i64 {
        (((self.0 << 8) as i32) >> 8) as i64
    }
    #[inline(always)]
    const fn upper_imm(self) -> u64 {
        (self.0 << 8) as i32 as i64 as u64
    }
}

/// C++ `FasterFloatType`: `func:8 | rd:8 | rs2:8 | rs1:8`.
#[cfg(feature = "ext-f")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct FasterFloatType(pub u32);

#[cfg(feature = "ext-f")]
impl FasterFloatType {
    pub(crate) const fn encode(func: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        (func & 0xFF) | ((rd & 0xFF) << 8) | ((rs2 & 0xFF) << 16) | ((rs1 & 0xFF) << 24)
    }
    #[inline(always)]
    const fn func(self) -> u32 {
        self.0 & 0xFF
    }
    #[inline(always)]
    const fn rd(self) -> u32 {
        (self.0 >> 8) & 0xFF
    }
    #[inline(always)]
    const fn rs2(self) -> u32 {
        (self.0 >> 16) & 0xFF
    }
    #[inline(always)]
    const fn rs1(self) -> u32 {
        self.0 >> 24
    }
}

#[inline(always)]
fn reg(m: &Machine, r: u32) -> u64 {
    m.cpu.regs.get(r)
}
#[inline(always)]
fn set(m: &mut Machine, r: u32, v: u64) {
    m.cpu.regs.set(r, v);
}

/// A sequential bytecode: `|m, ins| body`, then continue `$step` decoder
/// entries further on (C++ `NEXT_INSTR` is [`STEP`], `NEXT_C_INSTR` is 1).
/// `name: Marker` also defines the [`First`] marker type.
macro_rules! sequential {
    ($($(#[$attr:meta])* $name:ident $(: $marker:ident)?, $step:expr, |$m:ident, $ins:ident| $body:block)*) => {$(
        handler! {
            $(#[$attr])*
            pub(crate) fn $name(f) {
                {
                    let $ins = f.ins();
                    let $m = &mut *f.m;
                    $body
                }
                execute!(f, f.i() + $step)
            }
        }
        $(
            pub(crate) struct $marker;
            impl Bytecode for $marker {
                const HANDLER: Handler = $name;
            }
            impl First for $marker {
                const STEP: usize = $step;
                const MISS: Handler = $name;
                #[inline(always)]
                fn run(f: &mut Frame<'_>) -> bool {
                    let $ins = f.ins();
                    let $m = &mut *f.m;
                    $body
                    true
                }
            }
        )?
    )*};
}

sequential! {
    /// C++ `RV32I_BC_ADDI`
    addi: Addi, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs2()).wrapping_add(fi.signed_imm() as u64));
    }
    /// C++ `RV32I_BC_LI`
    li: Li, STEP, |m, ins| {
        let fi = FasterImmediate(ins);
        set(m, fi.rd(), fi.signed_imm() as u64);
    }
    /// C++ `RV32I_BC_MV`
    mv: Mv, STEP, |m, ins| {
        let fi = FasterMove(ins);
        set(m, fi.rd(), reg(m, fi.rs1()));
    }
    /// C++ `RV64I_BC_ADDIW`
    addiw: Addiw, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), sext32((reg(m, fi.rs2()) as u32).wrapping_add(fi.signed_imm() as u32)));
    }
    /// C++ `RV32I_BC_SLLI`
    slli: Slli, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs2()).wrapping_shl(fi.unsigned_imm()));
    }
    /// C++ `RV32I_BC_SLTI`
    slti, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), u64::from((reg(m, fi.rs2()) as i64) < fi.signed_imm()));
    }
    /// C++ `RV32I_BC_SLTIU`
    sltiu, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), u64::from(reg(m, fi.rs2()) < fi.signed_imm() as u64));
    }
    /// C++ `RV32I_BC_XORI`
    xori, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs2()) ^ fi.signed_imm() as u64);
    }
    /// C++ `RV32I_BC_SRLI`
    srli: Srli, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs2()).wrapping_shr(fi.unsigned_imm()));
    }
    /// C++ `RV32I_BC_SRAI`
    srai, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), (reg(m, fi.rs2()) as i64).wrapping_shr(fi.unsigned_imm()) as u64);
    }
    /// C++ `RV32I_BC_ORI`
    ori, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs2()) | fi.signed_imm() as u64);
    }
    /// C++ `RV32I_BC_ANDI`
    andi: Andi, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs2()) & fi.signed_imm() as u64);
    }
    /// C++ `RV32I_BC_LUI`
    lui: Lui, STEP, |m, ins| {
        let fi = FasterJtype(ins);
        set(m, fi.rd(), fi.upper_imm());
    }
    /// C++ `RV32I_BC_SEXT_B`
    sext_b, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs2()) as i8 as i64 as u64);
    }
    /// C++ `RV32I_BC_SEXT_H`
    sext_h, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs2()) as i16 as i64 as u64);
    }
    /// C++ `RV32I_BC_BSETI`
    bseti, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs2()) | 1u64.wrapping_shl(fi.unsigned_imm()));
    }
    /// C++ `RV32I_BC_BEXTI`
    bexti, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs2()).wrapping_shr(fi.unsigned_imm()) & 1);
    }
    /// C++ `RV64I_BC_SLLIW`
    slliw, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), sext32((reg(m, fi.rs2()) as u32).wrapping_shl(fi.unsigned_imm())));
    }
    /// C++ `RV64I_BC_SRLIW`
    srliw, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), sext32((reg(m, fi.rs2()) as u32).wrapping_shr(fi.unsigned_imm())));
    }
    /// C++ `RV64I_BC_SRAIW`
    sraiw, STEP, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), (reg(m, fi.rs2()) as i32).wrapping_shr(fi.unsigned_imm()) as i64 as u64);
    }

    /// C++ `RV32C_BC_ADDI` (C.ADDI, C.ADDI4SPN, C.ADDI16SP and C.LI)
    c_addi: CAddi, 1, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs2()).wrapping_add(fi.signed_imm() as u64));
    }
    /// C++ `RV32C_BC_MV`
    c_mv: CMv, 1, |m, ins| {
        let fi = FasterMove(ins);
        set(m, fi.rd(), reg(m, fi.rs1()));
    }
    /// C++ `RV32C_BC_SLLI`
    c_slli: CSlli, 1, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs1()).wrapping_shl(fi.unsigned_imm()));
    }
    /// C++ `RV32C_BC_SRLI`
    c_srli: CSrli, 1, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs1()).wrapping_shr(fi.unsigned_imm()));
    }
    /// C++ `RV32C_BC_ANDI`
    c_andi, 1, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs1()) & fi.signed_imm() as u64);
    }
    /// C++ `RV32C_BC_ADD`
    c_add: CAdd, 1, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs1()).wrapping_add(reg(m, fi.rs2())));
    }
    /// C++ `RV32C_BC_XOR`
    c_xor: CXor, 1, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs1()) ^ reg(m, fi.rs2()));
    }
    /// C++ `RV32C_BC_OR`
    c_or, 1, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), reg(m, fi.rs1()) | reg(m, fi.rs2()));
    }
    /// C++ `RV32C_BC_JAL_ADDIW` (C.ADDIW on RV64)
    c_addiw, 1, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), sext32((reg(m, fi.rs1()) as u32).wrapping_add(fi.signed_imm() as u32)));
    }
}

handler! {
    /// The register-register bytecodes (C++ `RV32I_BC_OP_*`, `RV64I_BC_OP_*`),
    /// sharing their arithmetic with the precise handlers.
    pub(crate) fn op<O: RtypeAlu>(f) {
        Op::<O>::run(&mut f);
        execute!(f, f.i() + STEP)
    }
}

/// The [`First`] marker of [`op`].
pub(crate) struct Op<O>(PhantomData<O>);

impl<O: RtypeAlu> Bytecode for Op<O> {
    const HANDLER: Handler = op::<O>;
}

impl<O: RtypeAlu> First for Op<O> {
    const STEP: usize = STEP;
    const MISS: Handler = op::<O>;
    #[inline(always)]
    fn run(f: &mut Frame<'_>) -> bool {
        let fi = FasterOpType(f.ins());
        let v = O::compute(reg(f.m, fi.rs1()), reg(f.m, fi.rs2()));
        set(f.m, fi.rd(), v);
        true
    }
}

/// C++ `RV32I_BC_OP_ADD`
pub(crate) type OpAdd = Op<Add>;
/// C++ `RV32I_BC_OP_SUB`
pub(crate) type OpSub = Op<Sub>;
/// C++ `RV32I_BC_OP_MUL`
pub(crate) type OpMul = Op<Mul>;
/// C++ `RV64I_BC_OP_ADDW`
pub(crate) type OpAddw = Op<Addw>;
/// C++ `RV64I_BC_OP_SH1ADD_UW`
pub(crate) type OpSh1addUw = Op<Sh1addUw>;

handler! {
    /// The CA-format register-register ops `rs1 = rs1 op rs2`: C++
    /// `RV32C_BC_SUB`, `AND`, `SUBW` and `ADDW`, and C.MUL, which C++ leaves
    /// to `RV32C_BC_FUNCTION`.
    pub(crate) fn c_op<O: RtypeAlu>(f) {
        let fi = FasterItype(f.ins());
        let v = O::compute(reg(f.m, fi.rs1()), reg(f.m, fi.rs2()));
        set(f.m, fi.rs1(), v);
        execute!(f, f.i() + 1)
    }
}

sequential! {
    /// C++ `RV32C_BC_SRAI`
    c_srai, 1, |m, ins| {
        let fi = FasterItype(ins);
        set(m, fi.rs1(), (reg(m, fi.rs1()) as i64).wrapping_shr(fi.unsigned_imm()) as u64);
    }
    /// C++ `RV32C_BC_LUI`, in the `FasterJtype` format of `RV32I_BC_LUI`
    c_lui: CLui, 1, |m, ins| {
        let fi = FasterJtype(ins);
        set(m, fi.rd(), fi.upper_imm());
    }
}

handler! {
    /// AUIPC reconstructs its address from the block end (C++ `rv32i_auipc`).
    pub(crate) fn auipc(f) {
        Auipc::run(&mut f);
        execute!(f, f.i() + STEP)
    }
}

/// The [`First`] marker of [`auipc`].
pub(crate) struct Auipc;

impl Bytecode for Auipc {
    const HANDLER: Handler = auipc;
}

impl First for Auipc {
    const STEP: usize = STEP;
    const MISS: Handler = auipc;
    #[inline(always)]
    fn run(f: &mut Frame<'_>) -> bool {
        let fi = FasterJtype(f.ins());
        let here = f.this_pc();
        set(f.m, fi.rd(), here.wrapping_add(fi.upper_imm()));
        true
    }
}

/// Loads: `rs1 = extend(read(rs2 + imm))` (C++ `RV32I_BC_LD*`,
/// `RV32C_BC_LD*`, `RV32F_BC_FL*`). The fast handler reads the flat arena;
/// everything else goes through `$slow`, which uses the page table and
/// raises faults.
macro_rules! loads {
    ($($(#[$attr:meta])* $name:ident $(: $marker:ident)?, $cached:ident, $slow:ident, $step:expr, $n:literal, |$m:ident, $rd:ident, $b:ident| $set:expr;)*) => {$(
        $(
            /// The [`First`] marker: the arena path, else `$cached`.
            pub(crate) struct $marker;
            impl Bytecode for $marker {
                const HANDLER: Handler = $name;
            }
            impl First for $marker {
                const STEP: usize = $step;
                const MISS: Handler = $cached;
                #[inline(always)]
                fn run(f: &mut Frame<'_>) -> bool {
                    let fi = FasterItype(f.ins());
                    let $m = &mut *f.m;
                    let addr = reg($m, fi.rs2()).wrapping_add(fi.signed_imm() as u64);
                    if let Some($b) = $m.memory.arena_read::<$n>(addr) {
                        let $rd = fi.rs1();
                        $set;
                        return true;
                    }
                    false
                }
            }
        )?

        handler! {
            $(#[$attr])*
            pub(crate) fn $name(f) {
                let fi = FasterItype(f.ins());
                {
                    let $m = &mut *f.m;
                    let addr = reg($m, fi.rs2()).wrapping_add(fi.signed_imm() as u64);
                    if let Some($b) = $m.memory.arena_read::<$n>(addr) {
                        let $rd = fi.rs1();
                        $set;
                        execute!(f, f.i() + $step)
                    }
                }
                tail!(f, $cached)
            }
        }

        handler! {
            /// Outside the arena: a page cache hit, without the calls (and
            /// so the stack frame) of the full page-table path in `$slow`.
            #[inline(never)]
            fn $cached(f) {
                let fi = FasterItype(f.ins());
                {
                    let $m = &mut *f.m;
                    let addr = reg($m, fi.rs2()).wrapping_add(fi.signed_imm() as u64);
                    if let Some($b) = $m.memory.read_cached::<$n>(addr) {
                        let $rd = fi.rs1();
                        $set;
                        execute!(f, f.i() + $step)
                    }
                }
                tail!(f, $slow)
            }
        }

        cold_handler! {
            fn $slow(f) {
                let fi = FasterItype(f.ins());
                {
                    let $m = &mut *f.m;
                    let addr = reg($m, fi.rs2()).wrapping_add(fi.signed_imm() as u64);
                    let $b = match $m.memory.read_slow::<$n>(addr) {
                        Ok(b) => b,
                        Err(err) => return fault($m, err),
                    };
                    let $rd = fi.rs1();
                    $set;
                }
                execute!(f, f.i() + $step)
            }
        }
    )*};
}

loads! {
    /// C++ `RV32I_BC_LDB`
    ldb, ldb_cached, ldb_slow, STEP, 1, |m, rd, b| set(m, rd, b[0] as i8 as i64 as u64);
    /// C++ `RV32I_BC_LDBU`
    ldbu: Ldbu, ldbu_cached, ldbu_slow, STEP, 1, |m, rd, b| set(m, rd, u64::from(b[0]));
    /// C++ `RV32I_BC_LDH`
    ldh: Ldh, ldh_cached, ldh_slow, STEP, 2, |m, rd, b| set(m, rd, i16::from_le_bytes(b) as i64 as u64);
    /// C++ `RV32I_BC_LDHU`
    ldhu: Ldhu, ldhu_cached, ldhu_slow, STEP, 2, |m, rd, b| set(m, rd, u64::from(u16::from_le_bytes(b)));
    /// C++ `RV32I_BC_LDW`
    ldw: Ldw, ldw_cached, ldw_slow, STEP, 4, |m, rd, b| set(m, rd, sext32(u32::from_le_bytes(b)));
    /// C++ `RV32I_BC_LDWU`
    ldwu, ldwu_cached, ldwu_slow, STEP, 4, |m, rd, b| set(m, rd, u64::from(u32::from_le_bytes(b)));
    /// C++ `RV32I_BC_LDD`
    ldd: Ldd, ldd_cached, ldd_slow, STEP, 8, |m, rd, b| set(m, rd, u64::from_le_bytes(b));
    /// C++ `RV32C_BC_LDW` (C.LW, C.LWSP)
    c_ldw: CLdw, c_ldw_cached, c_ldw_slow, 1, 4, |m, rd, b| set(m, rd, sext32(u32::from_le_bytes(b)));
    /// C++ `RV32C_BC_LDD` (C.LD, C.LDSP)
    c_ldd: CLdd, c_ldd_cached, c_ldd_slow, 1, 8, |m, rd, b| set(m, rd, u64::from_le_bytes(b));
    /// C++ `RV32C_BC_FLD` (C.FLD, C.FLDSP). Like the precise C.FLD handlers,
    /// not gated on the F extension.
    c_fld: CFld, c_fld_cached, c_fld_slow, 1, 8, |m, rd, b| m.cpu.regs.set_fp_bits(rd, u64::from_le_bytes(b));
}

#[cfg(feature = "ext-f")]
loads! {
    /// C++ `RV32F_BC_FLW`
    flw, flw_cached, flw_slow, STEP, 4, |m, rd, b| crate::isa::rvfd::load_u32(m, rd, u32::from_le_bytes(b));
    /// C++ `RV32F_BC_FLD`
    fld: Fld, fld_cached, fld_slow, STEP, 8, |m, rd, b| m.cpu.regs.set_fp_bits(rd, u64::from_le_bytes(b));
}

/// Stores: `write(rs1 + imm, rs2)` (C++ `RV32I_BC_ST*`, `RV32C_BC_ST*`,
/// `RV32F_BC_FS*`), with a cold page-table handler like [`loads!`].
macro_rules! stores {
    ($($(#[$attr:meta])* $name:ident $(: $marker:ident)?, $cached:ident, $slow:ident, $step:expr, |$m:ident, $rs2:ident| $value:expr;)*) => {$(
        $(
            /// The [`First`] marker: the arena path, else `$cached`.
            pub(crate) struct $marker;
            impl Bytecode for $marker {
                const HANDLER: Handler = $name;
            }
            impl First for $marker {
                const STEP: usize = $step;
                const MISS: Handler = $cached;
                #[inline(always)]
                fn run(f: &mut Frame<'_>) -> bool {
                    let fi = FasterItype(f.ins());
                    let $m = &mut *f.m;
                    let addr = reg($m, fi.rs1()).wrapping_add(fi.signed_imm() as u64);
                    let $rs2 = fi.rs2();
                    let value = $value;
                    $m.memory.arena_write(addr, value)
                }
            }
        )?

        handler! {
            $(#[$attr])*
            pub(crate) fn $name(f) {
                let fi = FasterItype(f.ins());
                {
                    let $m = &mut *f.m;
                    let addr = reg($m, fi.rs1()).wrapping_add(fi.signed_imm() as u64);
                    let $rs2 = fi.rs2();
                    let value = $value;
                    if !$m.memory.arena_write(addr, value) {
                        tail!(f, $cached)
                    }
                }
                execute!(f, f.i() + $step)
            }
        }

        handler! {
            /// Outside the arena: a page cache hit on a page of its own, like
            /// the loads' `$cached`.
            #[inline(never)]
            fn $cached(f) {
                let fi = FasterItype(f.ins());
                {
                    let $m = &mut *f.m;
                    let addr = reg($m, fi.rs1()).wrapping_add(fi.signed_imm() as u64);
                    let $rs2 = fi.rs2();
                    let value = $value;
                    if !$m.memory.write_cached(addr, value) {
                        tail!(f, $slow)
                    }
                }
                execute!(f, f.i() + $step)
            }
        }

        cold_handler! {
            fn $slow(f) {
                let fi = FasterItype(f.ins());
                {
                    let $m = &mut *f.m;
                    let addr = reg($m, fi.rs1()).wrapping_add(fi.signed_imm() as u64);
                    let $rs2 = fi.rs2();
                    let value = $value;
                    if let Err(err) = $m.memory.write_slow(addr, value) {
                        return fault($m, err);
                    }
                }
                execute!(f, f.i() + $step)
            }
        }
    )*};
}

stores! {
    /// C++ `RV32I_BC_STB`
    stb: Stb, stb_cached, stb_slow, STEP, |m, rs2| (reg(m, rs2) as u8).to_le_bytes();
    /// C++ `RV32I_BC_STH`
    sth: Sth, sth_cached, sth_slow, STEP, |m, rs2| (reg(m, rs2) as u16).to_le_bytes();
    /// C++ `RV32I_BC_STW`
    stw: Stw, stw_cached, stw_slow, STEP, |m, rs2| (reg(m, rs2) as u32).to_le_bytes();
    /// C++ `RV32I_BC_STD`
    std: Std, std_cached, std_slow, STEP, |m, rs2| reg(m, rs2).to_le_bytes();
    /// C++ `RV32C_BC_STW` (C.SW, C.SWSP)
    c_stw: CStw, c_stw_cached, c_stw_slow, 1, |m, rs2| (reg(m, rs2) as u32).to_le_bytes();
    /// C++ `RV32C_BC_STD` (C.SD, C.SDSP)
    c_std: CStd, c_std_cached, c_std_slow, 1, |m, rs2| reg(m, rs2).to_le_bytes();
    /// C++ `RV32C_BC_FSD` (C.FSD, C.FSDSP)
    c_fsd, c_fsd_cached, c_fsd_slow, 1, |m, rs2| m.cpu.regs.fp_bits(rs2).to_le_bytes();
}

#[cfg(feature = "ext-f")]
stores! {
    /// C++ `RV32F_BC_FSW`
    fsw, fsw_cached, fsw_slow, STEP, |m, rs2| (m.cpu.regs.fp_bits(rs2) as u32).to_le_bytes();
    /// C++ `RV32F_BC_FSD`
    fsd: Fsd, fsd_cached, fsd_slow, STEP, |m, rs2| m.cpu.regs.fp_bits(rs2).to_le_bytes();
}

/// FADD/FSUB/FMUL/FDIV (C++ `RV32F_BC_F*`). The rewriter only picks these
/// for the S (`func == 0`) and D (`func == 1`) formats.
#[cfg(feature = "ext-f")]
macro_rules! fp_arith {
    ($($(#[$attr:meta])* $name:ident, |$a:ident, $b:ident| $e:expr;)*) => {$(
        handler! {
            $(#[$attr])*
            pub(crate) fn $name(f) {
                use crate::isa::rvfd::{CANONICAL_NAN_F32, NANBOXING, load_u32};
                let fi = FasterFloatType(f.ins());
                let m = &mut *f.m;
                let x = m.cpu.regs.fp_bits(fi.rs1());
                let y = m.cpu.regs.fp_bits(fi.rs2());
                if fi.func() == 0 {
                    if NANBOXING && ((x >> 32) as u32 != u32::MAX || (y >> 32) as u32 != u32::MAX) {
                        load_u32(m, fi.rd(), CANONICAL_NAN_F32);
                    } else {
                        let ($a, $b) = (f32::from_bits(x as u32), f32::from_bits(y as u32));
                        let r: f32 = $e;
                        load_u32(m, fi.rd(), r.to_bits());
                    }
                } else {
                    let ($a, $b) = (f64::from_bits(x), f64::from_bits(y));
                    let r: f64 = $e;
                    m.cpu.regs.set_fp_bits(fi.rd(), r.to_bits());
                }
                execute!(f, f.i() + STEP)
            }
        }
    )*};
}

#[cfg(feature = "ext-f")]
fp_arith! {
    /// C++ `RV32F_BC_FADD`
    fadd, |a, b| a + b;
    /// C++ `RV32F_BC_FSUB`
    fsub, |a, b| a - b;
    /// C++ `RV32F_BC_FMUL`
    fmul, |a, b| a * b;
    /// C++ `RV32F_BC_FDIV`
    fdiv, |a, b| a / b;
}

/// Conditional branches (C++ `RV32I_BC_B*`): `PERFORM_BRANCH`, or for the
/// forward variants `PERFORM_FORWARD_BRANCH`, which skips the counter check
/// (forward branches cannot loop).
macro_rules! branches {
    ($($(#[$attr:meta])* $name:ident, forward: $fw:expr, |$a:ident, $b:ident| $cond:expr;)*) => {$(
        handler! {
            $(#[$attr])*
            pub(crate) fn $name(f) {
                let fi = FasterItype(f.ins());
                let $a = reg(f.m, fi.rs1());
                let $b = reg(f.m, fi.rs2());
                if $cond {
                    next_block!(f, fi.signed_imm(), !$fw)
                }
                next_block!(f, 4, false)
            }
        }
    )*};
}

branches! {
    /// C++ `RV32I_BC_BEQ`
    beq, forward: false, |a, b| a == b;
    /// C++ `RV32I_BC_BNE`
    bne, forward: false, |a, b| a != b;
    /// C++ `RV32I_BC_BEQ_FW`
    beq_fw, forward: true, |a, b| a == b;
    /// C++ `RV32I_BC_BNE_FW`
    bne_fw, forward: true, |a, b| a != b;
    /// C++ `RV32I_BC_BLT`
    blt, forward: false, |a, b| (a as i64) < (b as i64);
    /// C++ `RV32I_BC_BGE`
    bge, forward: false, |a, b| (a as i64) >= (b as i64);
    /// C++ `RV32I_BC_BLTU`
    bltu, forward: false, |a, b| a < b;
    /// C++ `RV32I_BC_BGEU`
    bgeu, forward: false, |a, b| a >= b;
}

markers! {
    BeqFw => beq_fw,
    Bne => bne,
    BneFw => bne_fw,
    Bltu => bltu,
    Bgeu => bgeu,
    CBeqz => c_beqz,
    CJr => c_jr,
    FastJal => fast_jal,
    FastCall => fast_call,
    Jalr => jalr,
}

handler! {
    /// C++ `RV32C_BC_BEQZ`. Unlike the 4-byte branches, C++ checks the counter
    /// on forward branches too (`PERFORM_BRANCH`).
    pub(crate) fn c_beqz(f) {
        let fi = FasterItype(f.ins());
        if reg(f.m, fi.rs1()) == 0 {
            next_block!(f, fi.signed_imm(), true)
        }
        next_block!(f, 2, false)
    }
}

handler! {
    /// C++ `RV32C_BC_BNEZ`
    pub(crate) fn c_bnez(f) {
        let fi = FasterItype(f.ins());
        if reg(f.m, fi.rs1()) != 0 {
            next_block!(f, fi.signed_imm(), true)
        }
        next_block!(f, 2, false)
    }
}

handler! {
    /// C++ `RV32C_BC_JMP`: the offset is the whole entry.
    pub(crate) fn c_jmp(f) {
        next_block!(f, i64::from(f.ins() as i32), true)
    }
}

handler! {
    /// C++ `RV32C_BC_JR`: the entry is the register.
    pub(crate) fn c_jr(f) {
        let target = reg(f.m, f.ins()) & !1;
        overflow_checked_jump!(f, target)
    }
}

handler! {
    /// C++ `RV32C_BC_JALR`. Reads the target before writing RA.
    pub(crate) fn c_jalr(f) {
        let target = reg(f.m, f.ins());
        let link = f.end_pc().wrapping_add(2);
        set(f.m, REG_RA, link);
        overflow_checked_jump!(f, target & !1)
    }
}

handler! {
    /// C++ `RV32I_BC_JAL`, with a link register other than `x0`/`ra`.
    pub(crate) fn jal(f) {
        let fi = FasterJtype(f.ins());
        let link = f.end_pc().wrapping_add(4);
        set(f.m, fi.rd(), link);
        next_block!(f, fi.signed_imm(), true)
    }
}

handler! {
    /// C++ `RV32I_BC_FAST_JAL` (JAL with `rd = x0`): the offset is the whole entry.
    pub(crate) fn fast_jal(f) {
        next_block!(f, i64::from(f.ins() as i32), true)
    }
}

handler! {
    /// C++ `RV32I_BC_FAST_CALL` (JAL with `rd = ra`).
    pub(crate) fn fast_call(f) {
        let link = f.end_pc().wrapping_add(4);
        set(f.m, REG_RA, link);
        next_block!(f, i64::from(f.ins() as i32), true)
    }
}

handler! {
    /// C++ `RV32I_BC_JALR`.
    pub(crate) fn jalr(f) {
        let fi = FasterItype(f.ins());
        // If rs1 == rd, avoid the clobber by computing the address first.
        let address = reg(f.m, fi.rs2()).wrapping_add(fi.signed_imm() as u64);
        if fi.rs1() != 0 {
            let link = f.end_pc().wrapping_add(4);
            set(f.m, fi.rs1(), link);
        }
        let target = address & !1;
        if !COMPRESSED && target & 0x3 != 0 {
            return fault(
                f.m,
                MachineError::trigger(ExceptionKind::MisalignedInstruction, target),
            );
        }
        overflow_checked_jump!(f, target)
    }
}
