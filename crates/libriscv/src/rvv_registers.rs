//! Port of `rvv_registers.hpp`: the vector register file and the vector
//! CSR state (`vl`, `vtype`, `vstart`, `vxrm`, `vxsat`).
//!
//! VLEN is 256 bits (C++ `RISCV_EXT_VECTOR` = 32 bytes). The 32 registers
//! are one flat little-endian byte array, so an element index that runs
//! past the end of a register group continues into the next register
//! exactly as C++'s `element_at` does (which wraps modulo the whole file).

/// VLEN in bytes (C++ `VectorLane::VSIZE`), as the `vlenb` CSR reports it.
pub const VLENB: usize = 32;
const FILE_BYTES: usize = 32 * VLENB;

/// A vector element type: an integer or floating-point type of 1, 2, 4 or
/// 8 bytes, stored little-endian (C++ `VectorLane::elem<T>`).
pub(crate) trait Elem: Copy {
    const BYTES: usize;
    fn load(bytes: &[u8]) -> Self;
    fn store(self, bytes: &mut [u8]);
}

macro_rules! elem {
    ($($t:ty),*) => {$(
        impl Elem for $t {
            const BYTES: usize = std::mem::size_of::<$t>();
            #[inline(always)]
            fn load(bytes: &[u8]) -> Self {
                bytes.try_into().map(<$t>::from_le_bytes).unwrap_or_default()
            }
            #[inline(always)]
            fn store(self, bytes: &mut [u8]) {
                bytes.copy_from_slice(&self.to_le_bytes());
            }
        }
    )*};
}
elem!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

/// The vector registers and vector CSRs (C++ `VectorRegisters<W>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorRegisters {
    v: [u8; FILE_BYTES],
    vl: u64,
    vstart: u64,
    /// Encoded SEW: log2(SEW / 8), 0..=3.
    vsew: u32,
    /// The raw vtype encoding, for the vtype CSR.
    vtype: u32,
    /// log2(LMUL); negative for the fractional LMULs.
    lmul: i32,
    vxrm: u8,
    vxsat: bool,
    vta: bool,
    vma: bool,
    vill: bool,
}

impl Default for VectorRegisters {
    fn default() -> Self {
        Self {
            v: [0; FILE_BYTES],
            vl: 0,
            vstart: 0,
            vsew: 0,
            vtype: 0,
            lmul: 0,
            vxrm: 0,
            vxsat: false,
            vta: false,
            vma: false,
            // Invalid until the first vsetvli.
            vill: true,
        }
    }
}

/// log2 of VLEN in bytes.
const VLEN_LOG2: i32 = VLENB.trailing_zeros() as i32;

impl VectorRegisters {
    /// Register `idx` (only the low 5 bits are used) as raw bytes.
    pub fn lane(&self, idx: u32) -> &[u8; VLENB] {
        let start = (idx as usize & 31) * VLENB;
        self.v
            .get(start..start + VLENB)
            .and_then(|s| s.try_into().ok())
            .unwrap_or(&[0; VLENB])
    }

    /// Register `idx` (only the low 5 bits are used) as raw bytes, mutably.
    pub fn lane_mut(&mut self, idx: u32) -> &mut [u8; VLENB] {
        let start = (idx as usize & 31) * VLENB;
        // The index is masked to 0..32, so the range is in the file.
        #[allow(clippy::indexing_slicing, clippy::unwrap_used)]
        (&mut self.v[start..start + VLENB]).try_into().unwrap()
    }

    /// All 32 registers as one byte array (v0 first).
    pub const fn file(&self) -> &[u8; FILE_BYTES] {
        &self.v
    }

    /// Byte offset of element `i` of the group at `vreg`, wrapped to the
    /// register file as C++ `element_at` wraps it.
    #[inline(always)]
    const fn offset<T: Elem>(vreg: u32, i: u64) -> usize {
        let per_reg = (VLENB / T::BYTES) as u64;
        let elements = 32 * per_reg;
        ((vreg as u64 * per_reg).wrapping_add(i) & (elements - 1)) as usize * T::BYTES
    }

    /// Element `i` of the register group starting at `vreg` (C++
    /// `element_at`).
    #[inline(always)]
    pub(crate) fn get<T: Elem>(&self, vreg: u32, i: u64) -> T {
        let o = Self::offset::<T>(vreg, i);
        T::load(self.v.get(o..o + T::BYTES).unwrap_or_default())
    }

    #[inline(always)]
    pub(crate) fn set<T: Elem>(&mut self, vreg: u32, i: u64, value: T) {
        let o = Self::offset::<T>(vreg, i);
        if let Some(b) = self.v.get_mut(o..o + T::BYTES) {
            value.store(b);
        }
    }

    /// Byte offset of element `i` of `bytes` bytes, as [`offset`](Self::offset).
    #[inline(always)]
    const fn offset_bytes(vreg: u32, i: u64, bytes: usize) -> usize {
        let per_reg = (VLENB / bytes) as u64;
        let elements = 32 * per_reg;
        ((vreg as u64 * per_reg).wrapping_add(i) & (elements - 1)) as usize * bytes
    }

    /// The bytes of element `i` (`bytes` wide: 1, 2, 4 or 8) of the group
    /// at `vreg`, for memory transfers.
    #[inline(always)]
    pub(crate) fn elem_bytes(&self, vreg: u32, i: u64, bytes: usize) -> &[u8] {
        let o = Self::offset_bytes(vreg, i, bytes);
        self.v.get(o..o + bytes).unwrap_or_default()
    }

    #[inline(always)]
    pub(crate) fn elem_bytes_mut(&mut self, vreg: u32, i: u64, bytes: usize) -> &mut [u8] {
        let o = Self::offset_bytes(vreg, i, bytes);
        self.v.get_mut(o..o + bytes).unwrap_or_default()
    }

    /// `len` contiguous bytes starting at register `vreg`, if they are in
    /// the file.
    pub(crate) fn group_bytes(&self, vreg: u32, len: usize) -> Option<&[u8]> {
        let start = vreg as usize * VLENB;
        self.v.get(start..start.checked_add(len)?)
    }

    pub(crate) fn group_bytes_mut(&mut self, vreg: u32, len: usize) -> Option<&mut [u8]> {
        let start = vreg as usize * VLENB;
        self.v.get_mut(start..start.checked_add(len)?)
    }

    /// Bit `i` of register `vreg` used as a mask (C++ `VectorLane::mask`).
    #[inline(always)]
    pub(crate) fn mask(&self, vreg: u32, i: u64) -> bool {
        let byte = self.get::<u8>(vreg, i / 8);
        (byte >> (i % 8)) & 1 != 0
    }

    #[inline(always)]
    pub(crate) fn set_mask(&mut self, vreg: u32, i: u64, value: bool) {
        // A mask register holds VLEN bits; C++ indexes the lane's bytes
        // directly, so the byte index stays within register `vreg`.
        let start = (vreg as usize & 31) * VLENB;
        let o = start + (i / 8) as usize % VLENB;
        if let Some(b) = self.v.get_mut(o) {
            let bit = 1u8 << (i % 8);
            *b = if value { *b | bit } else { *b & !bit };
        }
    }

    pub const fn vl(&self) -> u64 {
        self.vl
    }
    pub(crate) const fn set_vl(&mut self, value: u64) {
        self.vl = value;
    }
    pub const fn vill(&self) -> bool {
        self.vill
    }
    /// The current SEW in bits (8, 16, 32 or 64).
    pub const fn sew(&self) -> u32 {
        8 << self.vsew
    }
    /// The SEW as vtype encodes it: log2(SEW / 8).
    pub const fn encoded_sew(&self) -> u32 {
        self.vsew
    }
    /// log2(LMUL). Negative values are fractional LMULs.
    pub const fn lmul_shift(&self) -> i32 {
        self.lmul
    }
    /// The number of registers in a register group (0 when fractional).
    pub const fn group_regs(&self) -> u32 {
        if self.lmul >= 0 { 1 << self.lmul } else { 0 }
    }
    pub const fn vta(&self) -> bool {
        self.vta
    }
    pub const fn vma(&self) -> bool {
        self.vma
    }
    /// VLEN in bytes, as the vlenb CSR reports it.
    pub const fn vlenb() -> u64 {
        VLENB as u64
    }

    /// The vtype CSR: the encoding vsetvl was given, or vill alone (the
    /// top bit) when that encoding was illegal.
    pub const fn vtype(&self) -> u64 {
        if self.vill {
            1 << 63
        } else {
            self.vtype as u64
        }
    }

    /// The fixed-point rounding mode.
    pub const fn vxrm(&self) -> u32 {
        self.vxrm as u32
    }
    pub(crate) const fn set_vxrm(&mut self, value: u32) {
        self.vxrm = (value & 3) as u8;
    }
    /// The fixed-point saturation flag.
    pub const fn vxsat(&self) -> bool {
        self.vxsat
    }
    pub(crate) const fn set_vxsat(&mut self, value: bool) {
        self.vxsat = value;
    }
    /// vcsr: vxsat in bit 0 and vxrm in bits 2:1.
    pub const fn vcsr(&self) -> u32 {
        ((self.vxrm as u32) << 1) | self.vxsat as u32
    }
    pub(crate) const fn set_vcsr(&mut self, value: u32) {
        self.vxsat = value & 1 != 0;
        self.vxrm = ((value >> 1) & 3) as u8;
    }
    /// Always zero here: every instruction completes.
    pub const fn vstart(&self) -> u64 {
        self.vstart
    }
    pub(crate) const fn set_vstart(&mut self, value: u64) {
        self.vstart = value;
    }

    /// VLMAX = LMUL * VLEN / SEW, at least 1.
    pub const fn vlmax(&self) -> u64 {
        let shift = VLEN_LOG2 - self.vsew as i32 + self.lmul;
        if shift > 0 { 1 << shift } else { 1 }
    }

    /// Set vtype from the vtypei encoding bits. Returns false if the
    /// encoding is reserved or unsupported, in which case vill is set.
    pub(crate) const fn set_vtype(&mut self, vtypei: u32) -> bool {
        let vlmul = vtypei & 0x7;
        let vsew = (vtypei >> 3) & 0x7;
        self.vta = vtypei & 0x40 != 0;
        self.vma = vtypei & 0x80 != 0;
        // Bits above vma/vta are reserved, as is vlmul == 100. SEW is
        // limited to the widths implemented here.
        self.vill = (vtypei >> 8) != 0 || vlmul == 0b100 || vsew > 0b011;
        if !self.vill {
            self.vsew = vsew;
            // vlmul is log2(LMUL) as a three-bit signed field.
            self.lmul = vlmul as i32 - ((vlmul & 0b100) << 1) as i32;
            self.vtype = vtypei;
        }
        !self.vill
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vtype_encodings() {
        let mut r = VectorRegisters::default();
        assert!(r.vill());
        assert_eq!(r.vtype(), 1 << 63);
        // e32, m1
        assert!(r.set_vtype(0b010_000));
        assert_eq!((r.sew(), r.lmul_shift(), r.vlmax()), (32, 0, 8));
        // e8, mf8: VLEN/8/8 = 4
        assert!(r.set_vtype(0b000_101));
        assert_eq!((r.sew(), r.lmul_shift(), r.vlmax()), (8, -3, 4));
        // e64, mf8 clamps to one element
        assert!(r.set_vtype(0b011_101));
        assert_eq!(r.vlmax(), 1);
        // e64, m8
        assert!(r.set_vtype(0b011_011));
        assert_eq!((r.vlmax(), r.group_regs()), (32, 8));
        // Reserved LMUL, SEW > 64, bits above vma
        assert!(!r.set_vtype(0b000_100));
        assert!(!r.set_vtype(0b100_000));
        assert!(!r.set_vtype(0x100));
        assert_eq!(r.vtype(), 1 << 63);
    }

    #[test]
    fn elements_wrap_around_the_file() {
        let mut r = VectorRegisters::default();
        r.set::<u32>(31, 8, 0xdead_beef);
        // Element 8 of v31 at e32 is element 0 of v0.
        assert_eq!(r.get::<u32>(0, 0), 0xdead_beef);
        assert_eq!(r.lane(0)[..4], [0xef, 0xbe, 0xad, 0xde]);
        r.set_mask(3, 9, true);
        assert!(r.mask(3, 9));
        assert_eq!(r.lane(3)[1], 2);
    }

    #[test]
    fn vcsr_packs_vxrm_and_vxsat() {
        let mut r = VectorRegisters::default();
        r.set_vcsr(0b111);
        assert_eq!((r.vxrm(), r.vxsat()), (3, true));
        r.set_vxrm(6);
        assert_eq!(r.vcsr(), 0b101);
    }
}
