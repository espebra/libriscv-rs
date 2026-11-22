//! Port of `fused_bytecodes.hpp`, `fused_pairs.inc` and `fuse_bytecodes`
//! (`decoder_cache.cpp`): 128 common pairs of adjacent instructions get
//! bytecodes of their own.
//!
//! The decoder gives an entry whose next instruction is in the same block,
//! and whose bytecode and the next one form a pair, the handler of the pair:
//! [`fused`] runs the first instruction's body and continues in the second
//! instruction's handler with a direct jump instead of an indirect one. The
//! second entry is unchanged, as it may also be a jump target. Instruction
//! counts are added when a block is entered, so they do not change.
//!
//! An entry's bytecode is not its handler's address: generic handlers
//! (`op::<Add>`) may exist more than once, and identical ones may share an
//! address. So the decoder tags the entries it picks a fusable bytecode for
//! with a [`Bc`], and fusing goes by those tags. Patching a decoder entry
//! later (a breakpoint, the libc fast path) unfuses the entry before it, by
//! decoding it again (C++ `unfuse_entry_before`; see
//! `DecodedSegment::create_block_ending_entry_at`).
//!
//! Pairs follow C++ exactly; only RV64 ones exist, and those with a
//! compressed half never match without the C extension.

use crate::bytecode::{self as bc, Bytecode, First};
use crate::dispatch::Handler;
use crate::dispatch::abi::{execute_to, handler, tail};

/// A fused pair of bytecodes, `(first, second)`.
pub(crate) trait Pair {
    type A: First;
    type B: Bytecode;
}

impl<A: First, B: Bytecode> Pair for (A, B) {
    type A = A;
    type B = B;
}

handler! {
    /// C++ `rv_fused_<name>`: the first instruction's body, then the second
    /// instruction's handler (`NEXT_INSTR_TO`). A load or store outside
    /// the arena continues in its own slow path, which dispatches the second
    /// entry as usual.
    pub(crate) fn fused<P: Pair>(f) {
        if !P::A::run(&mut f) {
            tail!(f, <P::A as First>::MISS)
        }
        execute_to!(f, <P::B as Bytecode>::HANDLER, f.i() + <P::A as First>::STEP)
    }
}

/// Defines [`Bc`]: one variant per bytecode type in `bytecode.rs`.
macro_rules! bytecodes {
    ($($(#[$m:meta])* $name:ident,)*) => {
        /// A fusable bytecode, which the decoder tags its entries with (C++
        /// `get_bytecode()`, for the bytecodes of fused pairs only).
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub(crate) enum Bc {
            $($(#[$m])* $name,)*
        }

        impl Bc {
            /// The bytecode's handler.
            #[inline]
            pub(crate) const fn handler(self) -> Handler {
                match self {
                    $($(#[$m])* Self::$name => <bc::$name as Bytecode>::HANDLER,)*
                }
            }

            /// Every fusable bytecode.
            #[cfg(test)]
            pub(crate) const ALL: &[Self] = &[$($(#[$m])* Self::$name),*];
        }
    };
}

/// Defines [`fused_handler`] from the pairs of `fused_pairs.inc`.
macro_rules! pairs {
    ($($(#[$m:meta])* $a:ident $b:ident,)*) => {
        /// The handler of the fused pair `(a, b)`, if they are one.
        pub(crate) fn fused_handler(a: Bc, b: Bc) -> Option<Handler> {
            Some(match (a, b) {
                $($(#[$m])* (Bc::$a, Bc::$b) => fused::<(bc::$a, bc::$b)>,)*
                _ => return None,
            })
        }

        /// The number of pairs (C++ `RV32I_BC_FUSED_COUNT`).
        #[cfg(test)]
        pub(crate) const COUNT: usize = [$($(#[$m])* stringify!($a)),*].len();
    };
}

bytecodes! {
    Addiw,
    Andi,
    Mv,
    Bne,
    Std,
    Addi,
    OpMul,
    OpAdd,
    Ldw,
    Stw,
    Slli,
    Srli,
    CAdd,
    OpAddw,
    Ldh,
    #[cfg(feature = "ext-f")]
    Fld,
    #[cfg(feature = "ext-f")]
    Fsd,
    CFld,
    CAddi,
    Li,
    CLdw,
    CStw,
    Lui,
    CLui,
    CLdd,
    CMv,
    FastCall,
    CStd,
    CSlli,
    Ldd,
    Ldbu,
    OpSh1addUw,
    Ldhu,
    BeqFw,
    CSrli,
    CJr,
    Jalr,
    Bltu,
    Bgeu,
    FastJal,
    BneFw,
    Auipc,
    Stb,
    CBeqz,
    OpSub,
    Sth,
    CXor,
}

pairs! {
    Addiw Andi,
    Andi Andi,
    Mv Bne,
    Std Mv,
    Addi Addi,
    OpMul OpAdd,
    Mv Ldw,
    Stw Mv,
    Addi OpAdd,
    Slli Srli,
    Srli CAdd,
    OpAddw Slli,
    CAdd Ldh,
    #[cfg(feature = "ext-f")]
    Fld Addi,
    #[cfg(feature = "ext-f")]
    Fsd Bne,
    #[cfg(feature = "ext-f")]
    Addi Fsd,
    CFld CAddi,
    CAddi CAddi,
    #[cfg(feature = "ext-f")]
    CAddi Fsd,
    Stw Ldw,
    Ldw Stw,
    Li Stw,
    Ldw Ldw,
    CLdw CStw,
    CLdw CAddi,
    CAddi CStw,
    Stw Stw,
    Lui Addi,
    Std Std,
    Addi Std,
    Addi CAdd,
    CLui Addi,
    CAdd CLdd,
    CLdd CAddi,
    CMv FastCall,
    CStd CMv,
    CLdd CSlli,
    Slli OpAdd,
    Ldd Addi,
    Ldd Ldd,
    Mv Mv,
    CSlli CAdd,
    Andi CSlli,
    CAddi CStd,
    CMv CMv,
    OpAdd Ldbu,
    Ldbu Bne,
    OpSh1addUw Ldhu,
    Addiw BeqFw,
    Slli CSrli,
    Ldw Addi,
    CAddi CJr,
    Addi BeqFw,
    CMv CAddi,
    Mv FastCall,
    Ldbu Addi,
    Addi Jalr,
    Ldd Std,
    OpAdd OpAdd,
    Ldbu CAddi,
    Li BeqFw,
    CLdd CStd,
    OpAdd Bltu,
    CLdd CLdd,
    CStd CStd,
    CAddi CMv,
    CMv CLdd,
    Li Addi,
    CLdd CMv,
    CAddi BeqFw,
    Addi Andi,
    Mv Li,
    Ldd Mv,
    Li Li,
    Li Bgeu,
    OpAdd Addi,
    Addi FastJal,
    Ldd Ldw,
    CStd CAddi,
    Addi Li,
    Std Addi,
    Andi BeqFw,
    Mv Addi,
    Li Mv,
    Addi Bltu,
    Addi Bne,
    CStd CLdd,
    Mv Ldd,
    Ldhu Addi,
    Ldd Addiw,
    Li Bltu,
    CAdd CAdd,
    Ldbu Ldbu,
    Ldbu BeqFw,
    Andi BneFw,
    CAddi FastCall,
    Ldw BeqFw,
    CSlli CSrli,
    Std Ldd,
    Ldw Ldd,
    Auipc Ldd,
    Ldw Addiw,
    Addi BneFw,
    CLdd Addi,
    Stb Bne,
    Addiw Stw,
    CLdd CBeqz,
    Ldd CStd,
    Addi Stb,
    OpAdd Ldd,
    Ldw CAddi,
    Addi OpSub,
    Sth Bne,
    Srli BeqFw,
    CAddi Bgeu,
    CAddi Bne,
    Stw Addi,
    Mv BneFw,
    Ldd OpAdd,
    CAdd Ldbu,
    Addi Ldd,
    Ldd BeqFw,
    Ldd Slli,
    CAddi Bltu,
    Ldw Ldbu,
    CXor CAddi,
    CAdd CStd,
    Addi FastCall,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]
mod tests {
    use super::*;
    use crate::{Machine, MachineOptions};

    /// The whole of `fused_pairs.inc` (with the F extension: four pairs have
    /// FLD or FSD halves), and no duplicates.
    #[test]
    #[cfg(feature = "ext-f")]
    fn all_pairs_of_cpp() {
        assert_eq!(COUNT, 128);
        let pairs = Bc::ALL
            .iter()
            .flat_map(|&a| Bc::ALL.iter().map(move |&b| (a, b)))
            .filter(|&(a, b)| fused_handler(a, b).is_some())
            .count();
        assert_eq!(pairs, COUNT);
    }

    const CODE: u64 = 0x2000;
    const DATA: u64 = 0x10000;
    /// Where block-ending second halves jump backwards to: `j .`.
    const BACK: u64 = 4;
    /// The sled after a pair: two NOPs and `j .`.
    const SLED: [u32; 3] = [0x0000_0013, 0x0000_0013, 0x0000_006F];
    /// x5: the target of JALR and C.JR. x8: the loads' and stores' base.
    /// x14, x15: the branches' operands. The first halves write only
    /// x10..x13, fa0 and memory.
    const T0: u32 = 5;
    const A4: u32 = 14;
    const A5: u32 = 15;

    const fn btype(f3: u32, rs1: u32, rs2: u32, imm: i64) -> u32 {
        let imm = imm as u32;
        (((imm >> 12) & 1) << 31)
            | (((imm >> 5) & 0x3F) << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (f3 << 12)
            | (((imm >> 1) & 0xF) << 8)
            | (((imm >> 11) & 1) << 7)
            | 0x63
    }
    const fn jal(rd: u32, imm: i64) -> u32 {
        let imm = imm as u32;
        (((imm >> 20) & 1) << 31)
            | (((imm >> 1) & 0x3FF) << 21)
            | (((imm >> 11) & 1) << 20)
            | (((imm >> 12) & 0xFF) << 12)
            | (rd << 7)
            | 0x6F
    }
    /// C.BEQZ rs1', imm
    const fn c_beqz(rs1: u32, imm: i64) -> u16 {
        let imm = imm as u32;
        (0b110 << 13
            | ((imm >> 8) & 1) << 12
            | ((imm >> 3) & 3) << 10
            | (rs1 - 8) << 7
            | ((imm >> 6) & 3) << 5
            | ((imm >> 1) & 3) << 3
            | ((imm >> 5) & 1) << 2
            | 0b01) as u16
    }

    /// An instruction of bytecode `bc` at offset `at`, which jumps (if it
    /// does) to `fwd` or [`BACK`].
    fn sample(bc: Bc, at: u64, fwd: u64) -> Vec<u8> {
        let (fwd, back) = (fwd as i64 - at as i64, BACK as i64 - at as i64);
        let long = |w: u32| w.to_le_bytes().to_vec();
        let short = |h: u16| h.to_le_bytes().to_vec();
        match bc {
            Bc::Addi => long(0x0035_8513),       // addi a0, a1, 3
            Bc::Addiw => long(0x0075_851b),      // addiw a0, a1, 7
            Bc::Andi => long(0x03f6_7593),       // andi a1, a2, 63
            Bc::Mv => long(0x0006_8613),         // mv a2, a3
            Bc::OpAdd => long(0x00c5_8533),      // add a0, a1, a2
            Bc::OpSub => long(0x40d6_05b3),      // sub a1, a2, a3
            Bc::OpMul => long(0x02a6_8633),      // mul a2, a3, a0
            Bc::OpAddw => long(0x00b5_06bb),     // addw a3, a0, a1
            Bc::OpSh1addUw => long(0x20c5_a53b), // sh1add.uw a0, a1, a2
            Bc::Ldw => long(0x0084_2503),        // lw a0, 8(s0)
            Bc::Ldd => long(0x0104_3583),        // ld a1, 16(s0)
            Bc::Ldbu => long(0x0034_4603),       // lbu a2, 3(s0)
            Bc::Ldhu => long(0x0064_5683),       // lhu a3, 6(s0)
            Bc::Ldh => long(0x00a4_1503),        // lh a0, 10(s0)
            Bc::Stw => long(0x00b4_2c23),        // sw a1, 24(s0)
            Bc::Std => long(0x02c4_3023),        // sd a2, 32(s0)
            Bc::Stb => long(0x02d4_04a3),        // sb a3, 41(s0)
            Bc::Sth => long(0x02a4_1523),        // sh a0, 42(s0)
            #[cfg(feature = "ext-f")]
            Bc::Fld => long(0x0304_3507), // fld fa0, 48(s0)
            #[cfg(feature = "ext-f")]
            Bc::Fsd => long(0x02b4_3c27), // fsd fa1, 56(s0)
            Bc::Slli => long(0x00d5_9513),       // slli a0, a1, 13
            Bc::Srli => long(0x0076_5593),       // srli a1, a2, 7
            Bc::Auipc => long(0x0001_2517),      // auipc a0, 0x12
            Bc::Lui => long(0x1234_55b7),        // lui a1, 0x12345
            Bc::Li => long(0xffb0_0693),         // li a3, -5
            Bc::Jalr => long(0x0002_80e7),       // jalr ra, 0(t0)
            Bc::BeqFw => long(btype(0, A4, A5, fwd)),
            Bc::BneFw => long(btype(1, A4, A5, fwd)),
            Bc::Bne => long(btype(1, A4, A5, back)),
            Bc::Bltu => long(btype(6, A4, A5, back)),
            Bc::Bgeu => long(btype(7, A4, A5, fwd)),
            Bc::FastJal => long(jal(0, fwd)),
            Bc::FastCall => long(jal(1, fwd)),
            Bc::CAdd => short(0x952e),  // c.add a0, a1
            Bc::CAddi => short(0x0595), // c.addi a1, 5
            Bc::CMv => short(0x8636),   // c.mv a2, a3
            Bc::CSlli => short(0x068e), // c.slli a3, 3
            Bc::CSrli => short(0x8109), // c.srli a0, 2
            Bc::CXor => short(0x8db1),  // c.xor a1, a2
            Bc::CLui => short(0x6605),  // c.lui a2, 1
            Bc::CLdw => short(0x4408),  // c.lw a0, 8(s0)
            Bc::CStw => short(0xc44c),  // c.sw a1, 12(s0)
            Bc::CLdd => short(0x6810),  // c.ld a2, 16(s0)
            Bc::CStd => short(0xec14),  // c.sd a3, 24(s0)
            Bc::CFld => short(0x3008),  // c.fld fa0, 32(s0)
            Bc::CJr => short(0x8282),   // c.jr t0
            Bc::CBeqz => short(c_beqz(A4, fwd)),
        }
    }

    /// `j +8; j .; A; B; nop; nop; j .`, and the offsets of A and the sled.
    fn program(a: Bc, b: Bc) -> (Vec<u8>, u64, u64) {
        let mut code = Vec::new();
        code.extend(jal(0, 8).to_le_bytes());
        code.extend(jal(0, 0).to_le_bytes());
        // The lengths do not depend on the offsets.
        let la = sample(a, 0, 0).len() as u64;
        let lb = sample(b, 0, 0).len() as u64;
        let sled = 8 + la + lb;
        code.extend(sample(a, 8, sled + 4));
        code.extend(sample(b, 8 + la, sled + 4));
        for w in SLED {
            code.extend(w.to_le_bytes());
        }
        (code, 8, sled)
    }

    struct Outcome {
        x: Vec<u64>,
        f: Vec<u64>,
        data: Vec<u8>,
        pc: u64,
    }

    fn run(code: &[u8], sled: u64, taken: bool, precise: bool) -> Outcome {
        let mut m = Machine::new([].into(), MachineOptions::default()).unwrap();
        m.init_execute_area(code, CODE).unwrap();
        for k in 0..64 {
            m.memory
                .write_u64(DATA + k * 8, 0x0123_4567_89AB_CDEF ^ (k << 40) ^ k)
                .unwrap();
        }
        let regs = m.cpu.registers_mut();
        for r in 1..32 {
            regs.set(r, 0x8000_0000_0000_0000 | (u64::from(r) * 0x1_0001_0101));
            regs.set_fp_bits(r, u64::from(r).wrapping_mul(0x3FF0_0000_0100_0001));
        }
        regs.set(T0, CODE + sled + 4);
        regs.set(8, DATA);
        regs.set(A4, if taken { 0 } else { 1 });
        regs.set(A5, if taken { 0 } else { 2 });
        m.cpu.jump(CODE).unwrap();
        if precise {
            m.set_instruction_counter(0);
            m.set_max_instructions(12);
            m.simulate_precise().unwrap();
        } else {
            m.try_simulate(112, 0).unwrap();
        }
        let regs = m.cpu.registers();
        Outcome {
            x: (0..32).map(|i| regs.get(i)).collect(),
            f: (0..32).map(|i| regs.fp_bits(i)).collect(),
            data: (DATA..DATA + 512)
                .map(|a| m.memory.read_u8(a).unwrap())
                .collect(),
            pc: m.cpu.pc(),
        }
    }

    /// Every pair is fused by the decoder, and runs like the precise
    /// interpreter, its branches both taken and not.
    #[test]
    fn every_pair_fuses_and_runs_like_the_precise_interpreter() {
        let mut tested = 0;
        for &a in Bc::ALL {
            for &b in Bc::ALL {
                if fused_handler(a, b).is_none() {
                    continue;
                }
                let (la, lb) = (sample(a, 0, 0).len(), sample(b, 0, 0).len());
                if !crate::decoder_cache::COMPRESSED && (la == 2 || lb == 2) {
                    continue;
                }
                let (code, at, sled) = program(a, b);
                let context = format!("pair {a:?} {b:?}");
                // The decoder tags both halves as expected, and fuses them.
                let m = {
                    let mut m = Machine::new([].into(), MachineOptions::default()).unwrap();
                    m.init_execute_area(&code, CODE).unwrap();
                    m
                };
                let exec = m.cpu.current_execute_segment();
                let end = CODE + code.len() as u64;
                for (bc, off) in [(a, at), (b, at + la as u64)] {
                    let instr = exec.read_instruction(CODE + off, end);
                    let decoded = crate::decode::fast_handler(instr, CODE + off, CODE, end);
                    assert_eq!(decoded.and_then(|d| d.2), Some(bc), "{context}");
                }
                let first = &exec.cache[exec.index(CODE + at)];
                assert!(
                    !std::ptr::fn_addr_eq(first.handler, a.handler()),
                    "{context}: not fused"
                );
                for taken in [false, true] {
                    let fast = run(&code, sled, taken, false);
                    let precise = run(&code, sled, taken, true);
                    let context = format!("{context}, taken: {taken}");
                    assert_eq!(fast.pc, precise.pc, "{context}");
                    assert_eq!(fast.x, precise.x, "{context}");
                    assert_eq!(fast.f, precise.f, "{context}");
                    assert!(fast.data == precise.data, "memory differs, {context}");
                }
                tested += 1;
            }
        }
        if crate::decoder_cache::COMPRESSED && cfg!(feature = "ext-f") {
            assert_eq!(tested, COUNT);
        }
    }
}
