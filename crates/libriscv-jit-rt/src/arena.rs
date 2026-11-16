//! Port of shared-memory support in `shared_rodata.cpp`: guest arena mappings,
//! sealed memfd images, and page reclamation for `libriscv`'s `jit` feature.
//!
//! [`GuestArena`] is an anonymous private mapping exposed as a contiguous
//! `[u8]`. [`GuestArena::attach`] maps a [`RodataImage`] over its initial pages,
//! sharing program text and read-only data across machines.
//! [`GuestArena::zero_pages`] reclaims anonymous pages with
//! `madvise(MADV_DONTNEED)` without touching their bytes.
//!
//! # Copy-on-write images
//!
//! Images use `PROT_READ | PROT_WRITE | MAP_PRIVATE`. Reads share memfd page-cache
//! pages; writes create private copies. This preserves ordinary arena semantics
//! with paging, N-bit masking, and host writes. C++ uses a read-only shared
//! mapping and disables sharing when stores can bypass the write window.
//! Seals prohibit writes and size changes, preventing image mutation and
//! truncation-induced `SIGBUS`.
//!
//! The arena uses `memmap2` on all platforms. On Linux, inaccessible regions of
//! [`GUARD`] bytes surround it as additional protection against native access
//! errors. Native code must still enforce its own bounds checks.
//!
//! Sealable memfds and fixed-address mappings require Linux calls declared in
//! [`sys`] for architectures with supported `mmap` flag values.

use std::fs::File;
use std::io;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use memmap2::{Mmap, MmapMut, MmapOptions};

/// True when [`RodataImage`]s can be created on this platform (sealable
/// anonymous files, ie. Linux memfds).
pub const fn shared_rodata_supported() -> bool {
    sys::SUPPORTED
}

/// The host page size, which is what an image's length must be a multiple
/// of to be attached (4096 where images are not supported).
pub fn host_page_size() -> usize {
    sys::page_size()
}

/// Inaccessible bytes on each side of an arena (Linux): more than any
/// offset an access instruction encodes, and a multiple of every host page
/// size. Address space only: `PROT_NONE`, `MAP_NORESERVE`.
pub(crate) const GUARD: usize = if sys::SUPPORTED { 1 << 20 } else { 0 };

/// Zero-initialized guest memory: an anonymous private mapping of `len`
/// bytes (`MAP_NORESERVE` where the platform has it, so untouched pages
/// cost nothing), owned exclusively like a `Box<[u8]>`. The low part of it
/// can be replaced by a shared image ([`GuestArena::attach`]).
pub struct GuestArena {
    /// The mapping's base and length, cached so that `deref` is two loads
    /// (the arena is on every interpreter memory access).
    ptr: NonNull<u8>,
    len: usize,
    /// The mapping `ptr` points into (none for the empty arena), unmapped
    /// on drop, attached images included.
    map: Option<MmapMut>,
    /// The length of the image attached at the start, 0 for none: pages
    /// that [`GuestArena::zero_pages`] must not drop.
    attached: usize,
    /// Where `ptr` lies in `map`: after the low guard ([`GUARD`], or 0).
    guard: usize,
}

// SAFETY: the arena exclusively owns its mapping, like a `Box<[u8]>`, and
// only hands it out through `&self`/`&mut self` borrows.
#[allow(unsafe_code)]
unsafe impl Send for GuestArena {}
// SAFETY: as for `Send`: shared access is only ever to `&[u8]`.
#[allow(unsafe_code)]
unsafe impl Sync for GuestArena {}

impl std::fmt::Debug for GuestArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestArena")
            .field("base", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

impl GuestArena {
    /// An arena of no bytes, without a mapping.
    pub const fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            map: None,
            attached: 0,
            guard: 0,
        }
    }

    /// `len` bytes of zeroes.
    pub fn new(len: usize) -> io::Result<Self> {
        if len == 0 {
            return Ok(Self::empty());
        }
        // [low guard][arena, up to a page boundary][high guard]
        let too_big = || io::Error::from(io::ErrorKind::OutOfMemory);
        let high = len
            .checked_next_multiple_of(host_page_size())
            .and_then(|l| l.checked_add(GUARD))
            .ok_or_else(too_big)?;
        let total = high.checked_add(GUARD).ok_or_else(too_big)?;
        let mut map = MmapOptions::new().len(total).no_reserve_swap().map_anon()?;
        let base = map.as_mut_ptr();
        if GUARD != 0 {
            // SAFETY: the low guard is the first whole pages of the mapping
            // just made, which nothing borrows yet; the arena is not touched.
            #[allow(unsafe_code)]
            unsafe {
                sys::protect_none(base, GUARD)?;
            }
            // SAFETY: the high guard is the last whole pages of that
            // mapping (`high` is a multiple of the page size), past the
            // arena.
            #[allow(unsafe_code)]
            unsafe {
                sys::protect_none(base.wrapping_add(high), GUARD)?;
            }
        }
        let ptr = NonNull::new(base.wrapping_add(GUARD))
            .ok_or_else(|| io::Error::other("null mapping"))?;
        Ok(Self {
            ptr,
            len,
            map: Some(map),
            attached: 0,
            guard: GUARD,
        })
    }

    /// Replace the first `image.len()` bytes of the arena with `image`,
    /// mapped copy-on-write: they read as the image, and a store to one of
    /// its pages makes a private copy of that page.
    ///
    /// The image must fit in the arena and be a whole number of host pages
    /// long. On failure the range reads as the image's bytes anyway, as
    /// private anonymous memory: callers attach only an image that holds
    /// what the range holds already (just published from it) or is about
    /// to (verified against the program), so either way the guest sees the
    /// same bytes. Every step that can fail for want of resources (mapping
    /// the view to restore from) comes before the arena is touched; the
    /// process aborts only if re-mapping the arena's own range fails, ie.
    /// it could not access its own memory any more.
    pub fn attach(&mut self, image: &RodataImage) -> io::Result<()> {
        let len = image.len();
        if !sys::SUPPORTED
            || len == 0
            || len > self.len
            || !len.is_multiple_of(host_page_size())
            || self.map.is_none()
        {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        // The bytes to restore if the mapping fails (and the kernel had
        // already dropped the old pages).
        let view = image.view()?;
        let base = self.ptr.as_ptr();
        // SAFETY: `[base, base + len)` is inside our own mapping (checked
        // above) and `&mut self` means nothing borrows it. The image is
        // the whole sealed file, a multiple of the page size, so every
        // page is backed and can neither change nor be truncated away.
        #[allow(unsafe_code)]
        let mapped = unsafe { sys::map_image_fixed(base, len, &image.file) };
        let Err(error) = mapped else {
            self.attached = self.attached.max(len);
            return Ok(());
        };
        // SAFETY: as above, our own range, borrowed by nothing.
        #[allow(unsafe_code)]
        let zeroed = unsafe { sys::map_zeroes_fixed(base, len) };
        if zeroed.is_err() {
            // Part of the arena may be unmapped: `deref` would be unsound.
            std::process::abort();
        }
        if let Some(range) = self.get_mut(..len) {
            range.copy_from_slice(&view);
        }
        Err(error)
    }
}

impl GuestArena {
    /// Zero `[start, start + len)` by handing its pages back to the kernel
    /// (`madvise(MADV_DONTNEED)`, what C++ `memdiscard` does to its
    /// arena): the next read of each page gets a fresh zero page, and the
    /// host memory behind them is released. Pages that were never touched
    /// cost nothing, where writing (or even reading) them would fault
    /// every one in.
    ///
    /// True when the range now reads as zeroes. False, with the arena
    /// unchanged, when it cannot be done this way and the caller has to
    /// zero the bytes itself: not on Linux (elsewhere `MADV_DONTNEED` may
    /// keep the contents), `start` or `len` not a multiple of the host
    /// page size, the range not inside the arena, or overlapping an
    /// attached image (its pages would read as the image again, not as
    /// zeroes), or the call failing.
    pub fn zero_pages(&mut self, start: usize, len: usize) -> bool {
        let page = host_page_size();
        let Some(end) = start.checked_add(len) else {
            return false;
        };
        if !cfg!(target_os = "linux")
            || len == 0
            || !start.is_multiple_of(page)
            || !len.is_multiple_of(page)
            || end > self.len
            || start < self.attached
        {
            return false;
        }
        let Some(map) = self.map.as_ref() else {
            return false;
        };
        #[cfg(target_os = "linux")]
        {
            // SAFETY: `[start, end)` is whole pages of our own private
            // anonymous mapping (checked above: `ptr` is `guard` bytes, a
            // whole number of pages, into it, so the offset is page-aligned
            // and memmap2 does not widen the range to align it), past any
            // attached image, and `&mut self` means nothing borrows them. On Linux, such pages read as zeroes after
            // `MADV_DONTNEED`, which is the effect wanted; the mapping
            // itself, and so `ptr` and `len`, stay as they are.
            #[allow(unsafe_code)]
            let dropped = unsafe {
                map.unchecked_advise_range(
                    memmap2::UncheckedAdvice::DontNeed,
                    self.guard + start,
                    len,
                )
            };
            dropped.is_ok()
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (map, self.guard);
            false
        }
    }
}

impl Deref for GuestArena {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` is `len` readable bytes owned by `map` for as long
        // as `self` lives (or dangling and aligned with `len` 0), and only
        // mutated through `&mut self`.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
}

impl DerefMut for GuestArena {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as for `deref`; every page is writable, attached image
        // pages included (copy-on-write), and `&mut self` is exclusive.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }
}

/// A sealed, immutable image of a program's read-only memory (C++
/// `SharedRodataImage`): a memfd named `libriscv-rodata`. Its contents are
/// read through a temporary [`RodataImage::view`], so that a live image
/// costs no mapping besides the arenas it is attached to.
pub struct RodataImage {
    file: File,
    len: usize,
}

impl std::fmt::Debug for RodataImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RodataImage")
            .field("len", &self.len)
            .finish()
    }
}

impl RodataImage {
    /// A sealed image holding `bytes`, which must not be empty.
    pub fn new(bytes: &[u8]) -> io::Result<Self> {
        use std::io::Write;
        if bytes.is_empty() {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let mut file = sys::new_memfd()?;
        // Filled with write(), so no writable mapping ever exists, which
        // F_SEAL_WRITE requires.
        file.write_all(bytes)?;
        sys::seal(&file)?;
        Ok(Self {
            file,
            len: bytes.len(),
        })
    }

    /// A read-only mapping of the image's contents (C++ `inspect`), for as
    /// long as the caller keeps it.
    pub fn view(&self) -> io::Result<impl Deref<Target = [u8]> + use<>> {
        // SAFETY: the file is sealed: its contents and length cannot change
        // for as long as the view lives.
        #[allow(unsafe_code)]
        let view = unsafe { Mmap::map(&self.file)? };
        if view.len() != self.len {
            return Err(io::Error::other("image changed size"));
        }
        Ok(view)
    }

    /// The image's length in bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// True for an empty image (never the case for a created one).
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The Linux calls `memmap2` does not cover, declared by hand with the
/// generic flag values of the architectures listed (mips, powerpc and sparc
/// spell `MAP_ANONYMOUS`/`MAP_NORESERVE` differently, so they go without).
#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )
))]
mod sys {
    use std::ffi::{c_char, c_int, c_long, c_uint, c_void};
    use std::fs::File;
    use std::io;
    use std::os::fd::{AsRawFd, FromRawFd};

    pub(super) const SUPPORTED: bool = true;

    const PROT_NONE: c_int = 0;
    const PROT_READ: c_int = 1;
    const PROT_WRITE: c_int = 2;
    const MAP_PRIVATE: c_int = 0x02;
    const MAP_FIXED: c_int = 0x10;
    const MAP_ANONYMOUS: c_int = 0x20;
    const MAP_NORESERVE: c_int = 0x4000;
    const MFD_CLOEXEC: c_uint = 1;
    const MFD_ALLOW_SEALING: c_uint = 2;
    const F_ADD_SEALS: c_int = 1033;
    const F_SEAL_SEAL: c_int = 1;
    const F_SEAL_SHRINK: c_int = 2;
    const F_SEAL_GROW: c_int = 4;
    const F_SEAL_WRITE: c_int = 8;
    /// glibc and musl agree on Linux.
    const SC_PAGESIZE: c_int = 30;

    // SAFETY: the Linux declarations of these libc functions (`off_t` is
    // 64-bit on the architectures above).
    #[allow(unsafe_code)]
    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        fn memfd_create(name: *const c_char, flags: c_uint) -> c_int;
        fn mprotect(addr: *mut c_void, len: usize, prot: c_int) -> c_int;
        fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
        safe fn sysconf(name: c_int) -> c_long;
    }

    pub(super) fn page_size() -> usize {
        usize::try_from(sysconf(SC_PAGESIZE))
            .ok()
            .filter(|&s| s > 0)
            .unwrap_or(4096)
    }

    /// `mmap(MAP_FIXED)` over `[base, base + len)`; Ok when it landed there.
    ///
    /// # Safety
    ///
    /// The range must be a mapping the caller owns and nothing borrows:
    /// its pages are replaced.
    #[allow(unsafe_code)]
    unsafe fn map_fixed(base: *mut u8, len: usize, flags: c_int, fd: c_int) -> io::Result<()> {
        // SAFETY: the caller's contract; the result is checked.
        let addr = unsafe {
            mmap(
                base.cast(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_FIXED | MAP_NORESERVE | flags,
                fd,
                0,
            )
        };
        if addr == base.cast() {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Make `[base, base + len)` inaccessible (`PROT_NONE`).
    ///
    /// # Safety
    ///
    /// The range must be whole pages of a mapping the caller owns and
    /// nothing borrows: any access to them faults afterwards.
    #[allow(unsafe_code)]
    pub(super) unsafe fn protect_none(base: *mut u8, len: usize) -> io::Result<()> {
        // SAFETY: the caller's contract; the result is checked.
        if unsafe { mprotect(base.cast(), len, PROT_NONE) } == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Map `file` copy-on-write over `[base, base + len)`.
    ///
    /// # Safety
    ///
    /// As for `map_fixed`, and `file` must be at least `len` bytes long
    /// for as long as the mapping lives.
    #[allow(unsafe_code)]
    pub(super) unsafe fn map_image_fixed(base: *mut u8, len: usize, file: &File) -> io::Result<()> {
        #[cfg(test)]
        if FAIL_AFTER_UNMAP.take() {
            // What a failing kernel may leave behind: the old pages gone.
            // SAFETY: the caller's contract.
            unsafe { map_zeroes_fixed(base, len)? };
            return Err(io::Error::from(io::ErrorKind::OutOfMemory));
        }
        // SAFETY: the caller's contract.
        unsafe { map_fixed(base, len, 0, file.as_raw_fd()) }
    }

    #[cfg(test)]
    thread_local! {
        /// Make the next `map_image_fixed` on this thread fail after
        /// dropping the range's pages.
        pub(super) static FAIL_AFTER_UNMAP: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
    }

    /// Map fresh zeroes over `[base, base + len)`.
    ///
    /// # Safety
    ///
    /// As for `map_fixed`.
    #[allow(unsafe_code)]
    pub(super) unsafe fn map_zeroes_fixed(base: *mut u8, len: usize) -> io::Result<()> {
        // SAFETY: the caller's contract.
        unsafe { map_fixed(base, len, MAP_ANONYMOUS, -1) }
    }

    /// A new empty sealable memfd named `libriscv-rodata`.
    pub(super) fn new_memfd() -> io::Result<File> {
        // SAFETY: the name is a valid C string; no memory is affected.
        #[allow(unsafe_code)]
        let fd =
            unsafe { memfd_create(c"libriscv-rodata".as_ptr(), MFD_CLOEXEC | MFD_ALLOW_SEALING) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `fd` is a freshly created descriptor that nothing else
        // owns.
        #[allow(unsafe_code)]
        let file = unsafe { File::from_raw_fd(fd) };
        Ok(file)
    }

    /// Seal `file` against writes, growing, shrinking and further seals.
    pub(super) fn seal(file: &File) -> io::Result<()> {
        let seals = F_SEAL_SEAL | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE;
        // SAFETY: fcntl on a descriptor we own, affecting no memory.
        #[allow(unsafe_code)]
        let sealed = unsafe { fcntl(file.as_raw_fd(), F_ADD_SEALS, seals) };
        if sealed == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

/// Images are not supported here: arenas still work (they are `memmap2`
/// mappings), nothing is ever attached.
#[cfg(not(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )
)))]
mod sys {
    use std::fs::File;
    use std::io;

    pub(super) const SUPPORTED: bool = false;

    pub(super) const fn page_size() -> usize {
        4096
    }

    fn unsupported<T>() -> io::Result<T> {
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }

    /// Never called: `attach` checks `SUPPORTED` first.
    #[allow(unsafe_code)]
    pub(super) unsafe fn map_image_fixed(_: *mut u8, _: usize, _: &File) -> io::Result<()> {
        unsupported()
    }

    /// Never called: `attach` checks `SUPPORTED` first.
    #[allow(unsafe_code)]
    pub(super) unsafe fn map_zeroes_fixed(_: *mut u8, _: usize) -> io::Result<()> {
        unsupported()
    }

    /// Never called: there are no guards (`GUARD` is 0) where this is.
    #[allow(unsafe_code)]
    pub(super) unsafe fn protect_none(_: *mut u8, _: usize) -> io::Result<()> {
        unsupported()
    }

    pub(super) fn new_memfd() -> io::Result<File> {
        unsupported()
    }

    pub(super) fn seal(_: &File) -> io::Result<()> {
        unsupported()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::*;

    #[test]
    fn arena_is_zeroed_and_writable() {
        let mut arena = GuestArena::new(3 * 4096 + 5).unwrap();
        assert_eq!(arena.len(), 3 * 4096 + 5);
        assert!(arena.iter().all(|&b| b == 0));
        arena[3 * 4096 + 4] = 9;
        assert_eq!(arena[3 * 4096 + 4], 9);
        assert!(GuestArena::new(0).unwrap().is_empty());
        assert!(GuestArena::empty().is_empty());
    }

    #[test]
    fn zero_pages_zeroes_whole_pages_on_linux() {
        let page = host_page_size();
        let mut arena = GuestArena::new(8 * page).unwrap();
        arena.fill(7);
        // Unaligned, empty, out of range: refused, nothing changes.
        assert!(!arena.zero_pages(page + 1, page));
        assert!(!arena.zero_pages(page, page - 1));
        assert!(!arena.zero_pages(page, 0));
        assert!(!arena.zero_pages(7 * page, 2 * page));
        assert!(!arena.zero_pages(usize::MAX - page + 1, page));
        assert!(arena.iter().all(|&b| b == 7));
        let done = arena.zero_pages(2 * page, 3 * page);
        assert_eq!(done, cfg!(target_os = "linux"));
        if done {
            assert!(arena[..2 * page].iter().all(|&b| b == 7));
            assert!(arena[2 * page..5 * page].iter().all(|&b| b == 0));
            assert!(arena[5 * page..].iter().all(|&b| b == 7));
            // Still writable.
            arena[3 * page] = 1;
            assert_eq!(arena[3 * page], 1);
        }
        assert!(!GuestArena::empty().zero_pages(0, page));
    }

    /// On Linux the arena lies between two inaccessible guards, also with
    /// a length that is not a whole number of pages.
    #[test]
    fn arenas_have_guard_pages() {
        let len = 3 * host_page_size() + 100;
        let mut arena = GuestArena::new(len).unwrap();
        assert_eq!(arena.len(), len);
        arena[0] = 1;
        arena[len - 1] = 2;
        assert_eq!((arena[0], arena[len - 1]), (1, 2));
        if GUARD == 0 {
            return;
        }
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        // Whether `[from, to)` lies in one inaccessible mapping.
        let inaccessible = |from: usize, to: usize| {
            maps.lines().any(|line| {
                let mut fields = line.split_whitespace();
                let (Some(range), Some(perms)) = (fields.next(), fields.next()) else {
                    return false;
                };
                let Some((lo, hi)) = range.split_once('-') else {
                    return false;
                };
                let lo = usize::from_str_radix(lo, 16).unwrap();
                let hi = usize::from_str_radix(hi, 16).unwrap();
                perms.starts_with("---") && lo <= from && to <= hi
            })
        };
        let start = arena.as_ptr() as usize;
        let end = (start + len).next_multiple_of(host_page_size());
        assert!(inaccessible(start - GUARD, start), "{maps}");
        assert!(inaccessible(end, end + GUARD), "{maps}");
        assert!(!inaccessible(start, start + 1));
    }

    #[test]
    fn zero_pages_leaves_an_attached_image_alone() {
        if !shared_rodata_supported() {
            return;
        }
        let page = host_page_size();
        let image = RodataImage::new(&vec![5u8; 2 * page]).unwrap();
        let mut arena = GuestArena::new(6 * page).unwrap();
        arena.attach(&image).unwrap();
        arena[0] = 9; // a private copy of the first image page
        assert!(!arena.zero_pages(0, page));
        assert!(!arena.zero_pages(page, 2 * page));
        assert_eq!(arena[0], 9);
        assert!(arena[page..2 * page].iter().all(|&b| b == 5));
        arena[3 * page] = 4;
        assert!(arena.zero_pages(2 * page, 2 * page));
        assert_eq!(arena[3 * page], 0);
        assert!(arena[1..2 * page].iter().all(|&b| b == 5) && arena[0] == 9);
    }

    #[test]
    fn attached_image_reads_through_and_copies_on_write() {
        if !shared_rodata_supported() {
            return;
        }
        let page = host_page_size();
        let mut content = vec![0u8; 2 * page];
        content[0] = 0xAA;
        content[page + 1] = 0xBB;
        let image = RodataImage::new(&content).unwrap();
        assert_eq!(&*image.view().unwrap(), &content[..]);

        let mut a = GuestArena::new(4 * page).unwrap();
        let mut b = GuestArena::new(4 * page).unwrap();
        a[3 * page] = 7;
        a.attach(&image).unwrap();
        b.attach(&image).unwrap();
        assert_eq!(&a[..2 * page], &content[..]);
        assert_eq!(&b[..2 * page], &content[..]);
        // Above the image the arena is as it was.
        assert_eq!(a[3 * page], 7);

        // A store is private to the arena that made it.
        a[1] = 0xCC;
        assert_eq!(a[1], 0xCC);
        assert_eq!(b[1], 0);
        assert_eq!(image.view().unwrap()[1], 0);
    }

    #[test]
    fn attach_refuses_partial_pages_and_oversized_images() {
        if !shared_rodata_supported() {
            return;
        }
        let page = host_page_size();
        let image = RodataImage::new(&vec![1u8; page + 1]).unwrap();
        let mut arena = GuestArena::new(4 * page).unwrap();
        assert!(arena.attach(&image).is_err());
        assert!(arena.iter().all(|&b| b == 0));

        let image = RodataImage::new(&vec![1u8; 8 * page]).unwrap();
        assert!(arena.attach(&image).is_err());
        assert!(GuestArena::empty().attach(&image).is_err());
    }

    #[test]
    fn image_is_sealed() {
        if !shared_rodata_supported() {
            return;
        }
        use std::io::Write;
        let page = host_page_size();
        let image = RodataImage::new(&vec![3u8; page]).unwrap();
        assert!((&image.file).write_all(b"x").is_err());
        assert!(image.file.set_len(0).is_err());
        assert!(image.file.set_len(2 * page as u64).is_err());
    }

    /// A mapping that fails after the kernel dropped the old pages leaves
    /// the image's bytes in place, privately.
    #[cfg(all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )
    ))]
    #[test]
    fn failed_attach_restores_the_image_bytes() {
        let page = host_page_size();
        let content: Vec<u8> = (0..2 * page).map(|i| (i % 251) as u8).collect();
        let image = RodataImage::new(&content).unwrap();
        let mut arena = GuestArena::new(4 * page).unwrap();
        arena[..2 * page].copy_from_slice(&content);
        arena[3 * page] = 7;
        sys::FAIL_AFTER_UNMAP.set(true);
        assert!(arena.attach(&image).is_err());
        assert_eq!(&arena[..2 * page], &content[..]);
        assert_eq!(arena[3 * page], 7);
        // Private memory: stores stay in this arena.
        arena[0] ^= 0xFF;
        assert_eq!(image.view().unwrap()[0], content[0]);
    }

    #[test]
    fn images_are_unsupported_elsewhere() {
        if shared_rodata_supported() {
            return;
        }
        assert!(RodataImage::new(&[1]).is_err());
    }
}
