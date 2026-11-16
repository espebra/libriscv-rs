//! Guest bindings for host functions, native heap allocation, and memory helpers.
//!
//! Calls use counted custom-2 dyncalls: opcode `0x5b`, funct3 `0b111`, and a
//! system call number in the immediate. Register fields specify integer input
//! and output counts (`rd = x16 + inputs`, `rs1 = x24 + outputs`), with 4 added
//! to rs1 for FP access. No `a7` argument is required.
//!
//! Macros accept system call numbers chosen by the embedder and are independent
//! of the game.

/// One dyncall as inline assembly, with exactly the registers the host
/// touches as operands, so the compiler keeps everything else live across
/// it. Integer registers only.
macro_rules! dyncall {
    ($index:expr, $inputs:literal => $outputs:literal $(, $($operands:tt)+)?) => {
        core::arch::asm!(".insn i 0x5b, 7, x{rd}, x{rs1}, {index}",
            index = const $index,
            rd = const 16 + $inputs,
            rs1 = const 24 + $outputs,
            $($($operands)+,)?
            options(nostack))
    };
}
#[allow(unused_imports)]
pub(crate) use dyncall;

/// A host function argument: how the C calling convention passes it (an
/// integer register, an FP register, or a pair of integer registers for a
/// `&str` or a slice), which is also how the host's `sysargs` reads it.
pub trait HostArg {
    /// The type the dyncall stub is declared with.
    type Abi;
    /// Integer argument registers it takes.
    const INTS: u32;
    /// FP argument registers it takes.
    const FLOATS: u32;
    fn into_abi(self) -> Self::Abi;
}

/// A host function result, read back from `a0` or `fa0`.
pub trait HostRet {
    type Abi;
    /// Integer result registers.
    const INTS: u32;
    /// FP result registers.
    const FLOATS: u32;
    fn from_abi(abi: Self::Abi) -> Self;
}

macro_rules! int_args {
    ($($t:ty),*) => {$(
        impl HostArg for $t {
            type Abi = $t;
            const INTS: u32 = 1;
            const FLOATS: u32 = 0;
            #[inline(always)]
            fn into_abi(self) -> $t { self }
        }
        /// Read from all of `a0` and truncated: the host sign-extends every
        /// integer narrower than 64 bits (libriscv's convention), where the
        /// RISC-V ABI zero-extends unsigned 8- and 16-bit ones.
        impl HostRet for $t {
            type Abi = u64;
            const INTS: u32 = 1;
            const FLOATS: u32 = 0;
            #[inline(always)]
            fn from_abi(abi: u64) -> $t { abi as $t }
        }
    )*};
}
int_args!(i8, u8, i16, u16, i32, u32, i64, u64, isize, usize);

impl HostArg for bool {
    type Abi = bool;
    const INTS: u32 = 1;
    const FLOATS: u32 = 0;
    #[inline(always)]
    fn into_abi(self) -> bool {
        self
    }
}
/// Any non-zero `a0` is true: the host is trusted, but a stray bit must not
/// make an invalid `bool`.
impl HostRet for bool {
    type Abi = u64;
    const INTS: u32 = 1;
    const FLOATS: u32 = 0;
    #[inline(always)]
    fn from_abi(abi: u64) -> bool {
        abi != 0
    }
}

macro_rules! float_args {
    ($($t:ty),*) => {$(
        impl HostArg for $t {
            type Abi = $t;
            const INTS: u32 = 0;
            const FLOATS: u32 = 1;
            #[inline(always)]
            fn into_abi(self) -> $t { self }
        }
        impl HostRet for $t {
            type Abi = $t;
            const INTS: u32 = 0;
            const FLOATS: u32 = 1;
            #[inline(always)]
            fn from_abi(abi: $t) -> $t { abi }
        }
    )*};
}
float_args!(f32, f64);

impl HostRet for () {
    type Abi = ();
    const INTS: u32 = 0;
    const FLOATS: u32 = 0;
    #[inline(always)]
    fn from_abi(_: ()) {}
}

/// A pointer and a length. The calling convention passes a two-word
/// struct in two integer registers, exactly as the host reads a `&str` or
/// a `GuestSlice`.
#[repr(C)]
pub struct RawParts {
    ptr: *const u8,
    len: usize,
}

impl HostArg for &str {
    type Abi = RawParts;
    const INTS: u32 = 2;
    const FLOATS: u32 = 0;
    #[inline(always)]
    fn into_abi(self) -> RawParts {
        RawParts {
            ptr: self.as_ptr(),
            len: self.len(),
        }
    }
}

impl<T: Copy> HostArg for &[T] {
    type Abi = RawParts;
    const INTS: u32 = 2;
    const FLOATS: u32 = 0;
    #[inline(always)]
    fn into_abi(self) -> RawParts {
        RawParts {
            ptr: self.as_ptr().cast(),
            len: self.len(),
        }
    }
}

/// Declare host functions: each becomes a safe Rust function that calls the
/// host with one counted dyncall.
///
/// ```ignore
/// host_functions! {
///     api::DRAW_TEXT => fn draw_text(x: i32, y: i32, text: &str);
///     api::SIN_F32 => fn sin_f32(x: f32) -> f32;
/// }
/// ```
///
/// The dyncall sits in a naked function of its own (the instruction and a
/// `ret`), so the compiler places the arguments by the calling convention,
/// and the register counts in the instruction are computed from the
/// argument types.
macro_rules! host_functions {
    ($($(#[$attr:meta])* $number:path => fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?;)*) => {$(
        $crate::host::host_functions!(@one [$(#[$attr])*] $number, $name, [$($arg: $ty),*], [$($ret)?]);
    )*};
    (@one [$($attr:tt)*] $number:expr, $name:ident, [$($arg:ident: $ty:ty),*], []) => {
        $crate::host::host_functions!(@one [$($attr)*] $number, $name, [$($arg: $ty),*], [()]);
    };
    (@one [$($attr:tt)*] $number:expr, $name:ident, [$($arg:ident: $ty:ty),*], [$ret:ty]) => {
        $($attr)*
        #[inline]
        #[allow(dead_code)]
        pub fn $name($($arg: $ty),*) -> $ret {
            #[allow(unused_imports)]
            use $crate::host::{HostArg, HostRet};
            const INPUTS: u32 = 0 $(+ <$ty as HostArg>::INTS)*;
            const OUTPUTS: u32 = <$ret as HostRet>::INTS;
            const FLOATS: bool = (0 $(+ <$ty as HostArg>::FLOATS)* + <$ret as HostRet>::FLOATS) > 0;
            const _: () = assert!(INPUTS <= 8 && OUTPUTS <= 2, "too many registers for a dyncall");
            #[unsafe(naked)]
            extern "C" fn call($(_: <$ty as HostArg>::Abi),*) -> <$ret as HostRet>::Abi {
                core::arch::naked_asm!(
                    ".insn i 0x5b, 7, x{rd}, x{rs1}, {number}",
                    "ret",
                    number = const $number,
                    rd = const 16 + INPUTS,
                    rs1 = const 24 + OUTPUTS + if FLOATS { 4 } else { 0 },
                )
            }
            <$ret as HostRet>::from_abi(call($(HostArg::into_abi($arg)),*))
        }
    };
}
pub(crate) use host_functions;

/// Make the host's native heap the global allocator (the host installs it
/// with `setup_native_heap(base, ..)`), so both sides allocate from the
/// same heap and either can grow or free what the other allocated.
macro_rules! native_heap {
    ($base:expr) => {
        struct NativeHeap;

        // The arena hands out 16-byte aligned blocks.
        unsafe impl core::alloc::GlobalAlloc for NativeHeap {
            #[inline]
            unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
                if layout.align() > 16 {
                    return core::ptr::null_mut();
                }
                let ret: *mut u8;
                unsafe {
                    $crate::host::dyncall!($base + 0, 1 => 1, in("a0") layout.size(), lateout("a0") ret);
                }
                ret
            }
            #[inline]
            unsafe fn alloc_zeroed(&self, layout: core::alloc::Layout) -> *mut u8 {
                if layout.align() > 16 {
                    return core::ptr::null_mut();
                }
                let ret: *mut u8;
                unsafe {
                    $crate::host::dyncall!($base + 1, 2 => 1,
                        in("a0") 1usize, in("a1") layout.size(), lateout("a0") ret);
                }
                ret
            }
            #[inline]
            unsafe fn realloc(&self, ptr: *mut u8, layout: core::alloc::Layout, new_size: usize) -> *mut u8 {
                if layout.align() > 16 {
                    return core::ptr::null_mut();
                }
                let ret: *mut u8;
                unsafe {
                    $crate::host::dyncall!($base + 2, 2 => 1,
                        in("a0") ptr, in("a1") new_size, lateout("a0") ret);
                }
                ret
            }
            #[inline]
            unsafe fn dealloc(&self, ptr: *mut u8, _layout: core::alloc::Layout) {
                unsafe {
                    $crate::host::dyncall!($base + 3, 1 => 0, in("a0") ptr);
                }
            }
        }

        #[global_allocator]
        static ALLOCATOR: NativeHeap = NativeHeap;
    };
}
pub(crate) use native_heap;

/// Hand memcpy, memset, memmove and memcmp to the host's native memory
/// functions (installed with `setup_native_memory(base)`). The link step
/// passes `--wrap=memcpy` and friends (see `.cargo/config.toml`), which
/// sends every call, Rust's and glibc's, to these.
///
/// Inline assembly prevents byte-loop recognition from generating recursive
/// calls to these wrappers.
macro_rules! native_memory {
    ($base:expr) => {
        #[unsafe(no_mangle)]
        unsafe extern "C" fn __wrap_memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
            unsafe { $crate::host::dyncall!($base + 0, 3 => 0, in("a0") dest, in("a1") src, in("a2") n) };
            dest
        }
        #[unsafe(no_mangle)]
        unsafe extern "C" fn __wrap_memset(dest: *mut u8, value: i32, n: usize) -> *mut u8 {
            unsafe { $crate::host::dyncall!($base + 1, 3 => 0, in("a0") dest, in("a1") value, in("a2") n) };
            dest
        }
        #[unsafe(no_mangle)]
        unsafe extern "C" fn __wrap_memmove(dest: *mut u8, src: *const u8, n: usize) -> *mut u8 {
            unsafe { $crate::host::dyncall!($base + 2, 3 => 0, in("a0") dest, in("a1") src, in("a2") n) };
            dest
        }
        #[unsafe(no_mangle)]
        unsafe extern "C" fn __wrap_memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
            let result: i32;
            unsafe {
                $crate::host::dyncall!($base + 3, 3 => 1,
                    in("a0") s1, in("a1") s2, in("a2") n, lateout("a0") result)
            };
            result
        }
        // Slice equality calls the equality-only entry points, which a
        // memcmp result satisfies.
        #[unsafe(no_mangle)]
        unsafe extern "C" fn __wrap___memcmpeq(s1: *const u8, s2: *const u8, n: usize) -> i32 {
            unsafe { __wrap_memcmp(s1, s2, n) }
        }
        #[unsafe(no_mangle)]
        unsafe extern "C" fn __wrap_bcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
            unsafe { __wrap_memcmp(s1, s2, n) }
        }
    };
}
pub(crate) use native_memory;
