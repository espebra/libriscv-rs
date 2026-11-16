//! Port of `linux/syscalls_mman.cpp`: munmap, mmap, mremap, mprotect and
//! madvise, with virtual paging (mappings can be anywhere in the address
//! space, as in the reference emulator).

use std::os::unix::fs::FileExt;

use crate::error::{ExceptionKind, MachineError, Result};
use crate::machine::Machine;
use crate::memory::{PAGE_SIZE, PageAttributes, page_round_up};
use crate::syscalls::errno::{EEXIST, EINVAL, ENOMEM, neg};
use crate::syscalls::fds::Target;

const MAP_ANONYMOUS: i32 = 0x20;
const MAP_NORESERVE: i32 = 0x04000;
const MAP_FIXED: i32 = 0x10;
const MAP_FIXED_NOREPLACE: i32 = 0x10_0000;
const MAP_FAILED: u64 = u64::MAX;

pub(super) fn syscall_munmap(m: &mut Machine) -> Result<()> {
    let addr = m.sysarg(0);
    let len = m.sysarg(1);
    let Some(end) = addr.checked_add(len) else {
        return Err(MachineError::new(
            ExceptionKind::SystemCallFailed,
            "munmap() arguments overflow",
            0,
        ));
    };
    m.memory.free_pages(addr, len);
    if addr >= m.memory.mmap_start() && end <= m.memory.mmap_address() {
        m.memory.mmap_unmap(addr, len);
    }
    m.set_result(0);
    Ok(())
}

/// Read file `vfd` from `offset` into `[dst, dst + length)`: the file
/// mapping of C++ (`lseek` + `readv` into at most 256 gathered buffers). A
/// short file leaves the rest as it was. Returns false on failure.
fn map_file(m: &mut Machine, vfd: i32, offset: u64, dst: u64, length: u64) -> Result<bool> {
    let Some(fds) = m.fds.as_deref_mut() else {
        return Ok(false);
    };
    let Some(Target::File(f)) = fds.translate(vfd) else {
        return Ok(false);
    };
    let file = &f.file;
    let mut pos = offset;
    for buf in m.memory.gather_writable_buffers(dst, length, 256)? {
        let mut done = 0usize;
        while let Some(rest) = buf.get_mut(done..) {
            if rest.is_empty() {
                break;
            }
            match file.read_at(rest, pos) {
                Ok(0) => return Ok(true),
                Ok(n) => {
                    done += n;
                    pos = pos.wrapping_add(n as u64);
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => return Ok(false),
            }
        }
    }
    Ok(true)
}

/// Take `[addr, addr + length)` from the mmap area if all of it is free
/// (`MAP_FIXED_NOREPLACE`): above the cursor, which then moves past it
/// (the skipped gap goes to the free list), or carved out of the free list.
/// Changes nothing and returns false otherwise.
fn claim_free_range(m: &mut Machine, addr: u64, length: u64, nextfree: &mut u64) -> bool {
    let Some(end) = addr.checked_add(length) else {
        return false;
    };
    if addr < m.memory.mmap_start() {
        return false;
    }
    if addr >= *nextfree {
        // Untouched territory: move the cursor above it, and donate the
        // skipped gap to the free list.
        if addr > *nextfree {
            m.memory
                .mmap_cache_mut()
                .insert(*nextfree, addr - *nextfree);
        }
        *nextfree = end;
        true
    } else {
        // Below the cursor: only if not occupied
        m.memory.mmap_cache_mut().carve(addr, length)
    }
}

/// Where a file mapping without an address goes when its file is pinned
/// ([`FileDescriptors::pinned_mappings`](crate::FileDescriptors)): the pinned
/// address, claimed from the mmap area, if the whole mapping fits in the
/// arena there and nothing else is mapped there. Not a Linux feature: it
/// keeps shared libraries at the same address in every program, so code
/// saved for them (the JIT cache, AOT translations) is found again.
fn pinned_address(m: &mut Machine, vfd: i32, length: u64) -> Option<u64> {
    let fds = m.fds.as_deref_mut()?;
    let path = fds.get(vfd)?.path.clone();
    let addr = fds
        .pinned_mappings
        .iter()
        .find_map(|(p, a)| (*p == path).then_some(*a))?;
    let end = addr.checked_add(length)?;
    if !addr.is_multiple_of(PAGE_SIZE) || end > m.memory.memory_arena_size() {
        return None;
    }
    let mut nextfree = m.memory.mmap_address();
    claim_free_range(m, addr, length, &mut nextfree).then(|| {
        m.memory.set_mmap_address(nextfree);
        addr
    })
}

/// If file `vfd` is one of the pinned files, remember that it was mapped
/// at `dst` (`length` bytes from `offset`): segments there whose bytes are
/// the file's may be saved to the JIT cache directory
/// ([`CacheDir`](crate::jit_cache::CacheDir)), at any address.
#[cfg(feature = "jit")]
fn record_pinned_file(m: &mut Machine, vfd: i32, dst: u64, length: u64, offset: u64) {
    let Some(fds) = m.fds.as_deref_mut() else {
        return;
    };
    let Some(path) = fds.get(vfd).map(|f| f.path.clone()) else {
        return;
    };
    if fds.pinned_mappings.iter().any(|(p, _)| *p == path) {
        m.memory.record_pinned_file(dst, length, &path, offset);
    }
}

pub(super) fn syscall_mmap(m: &mut Machine) -> Result<()> {
    let addr_g = m.sysarg(0);
    let length = m.sysarg(1);
    let prot = m.sysarg(2) as i32;
    let flags = m.sysarg(3) as i32;
    let vfd = m.sysarg(4) as i32;
    let voff = m.sysarg(5);
    let attr = PageAttributes::from_prot(prot);
    let failed = |m: &mut Machine| {
        m.set_result(MAP_FAILED);
        Ok(())
    };
    let taken = |m: &mut Machine| {
        m.set_result(neg(EEXIST));
        Ok(())
    };

    if !addr_g.is_multiple_of(PAGE_SIZE) {
        return failed(m);
    }
    let prev_nextfree = m.memory.mmap_address();
    let mut nextfree = prev_nextfree;
    let length = page_round_up(length);
    if length == 0 {
        return failed(m);
    }

    if vfd != -1 {
        if m.fds.is_none() {
            // File descriptors are disabled: fail like the kernel would.
            return failed(m);
        }
        let pinned = if addr_g == 0 && voff == 0 {
            pinned_address(m, vfd, length)
        } else {
            None
        };
        let dst = if let Some(pin) = pinned {
            pin
        } else if addr_g == 0 {
            let Some(end) = nextfree.checked_add(length) else {
                return failed(m);
            };
            m.memory.set_mmap_address(end);
            nextfree
        } else {
            if addr_g.checked_add(length).is_none() {
                return failed(m);
            }
            addr_g
        };
        // Make the area read-write, read the file into it, then apply the
        // protections.
        m.memory
            .set_page_attr(dst, length, PageAttributes::default())?;
        if !map_file(m, vfd, voff, dst, length)? {
            return failed(m);
        }
        #[cfg(feature = "jit")]
        record_pinned_file(m, vfd, dst, length, voff);
        m.memory.set_page_attr(dst, length, attr)?;
        m.set_result(dst);
        return Ok(());
    }

    let result;
    if addr_g != 0 && flags & MAP_FIXED_NOREPLACE != 0 {
        // MAP_FIXED_NOREPLACE maps at exactly the given address or fails
        // with EEXIST, and is never relocated.
        if !claim_free_range(m, addr_g, length, &mut nextfree) {
            return taken(m);
        }
        result = addr_g;
    } else if addr_g == 0 || flags & MAP_FIXED == 0 {
        // Without MAP_FIXED the address is only a hint, which is ignored.
        let range = m.memory.mmap_cache_mut().find(length);
        if range.size == 0 {
            let Some(end) = nextfree.checked_add(length) else {
                return failed(m);
            };
            result = nextfree;
            nextfree = end;
        } else {
            result = range.addr;
        }
    } else {
        match addr_g.checked_add(length) {
            None => return failed(m),
            Some(_) if addr_g < m.memory.mmap_start() => {
                // A fixed range below the mmap area: just return it
                result = addr_g;
            }
            Some(e) if e <= nextfree => {
                // Inside the mmap area: it may cover a range on the free
                // list, which must not be handed out a second time.
                let cache = m.memory.mmap_cache_mut();
                if !cache.carve(addr_g, length) {
                    cache.invalidate(addr_g, length);
                }
                result = addr_g;
            }
            Some(e) => {
                // Ending after the current end of the mmap area
                let cache = m.memory.mmap_cache_mut();
                if addr_g > nextfree {
                    cache.insert(nextfree, addr_g - nextfree);
                } else {
                    cache.invalidate(addr_g, length);
                }
                result = addr_g;
                nextfree = nextfree.max(e);
            }
        }
    }
    m.memory.set_mmap_address(nextfree);

    let untouched = result >= prev_nextfree;
    // Anonymous pages need to be zeroed
    if flags & MAP_ANONYMOUS != 0 {
        m.memory.memdiscard(result, length, true)?;
    }
    // Avoid creating pages for MAP_NORESERVE and untouched PROT_NONE
    if flags & MAP_NORESERVE == 0 && !(prot == 0 && untouched) {
        m.memory.set_page_attr(result, length, attr)?;
    }
    m.set_result(result);
    Ok(())
}

pub(super) fn syscall_mremap(m: &mut Machine) -> Result<()> {
    const MREMAP_FIXED: i32 = 0x0002;
    let old_addr = m.sysarg(0);
    let old_size = m.sysarg(1);
    let new_size = m.sysarg(2);
    let flags = m.sysarg(3) as i32;
    // Only the common case of growing (or shrinking) the last mapping.
    if flags & MREMAP_FIXED == 0
        && let Some(old_end) = old_addr.checked_add(old_size)
        && old_end == m.memory.mmap_address()
        && let Some(new_end) = old_addr.checked_add(new_size)
    {
        m.memory.set_mmap_address(new_end);
        m.set_result(old_addr);
        return Ok(());
    }
    m.set_result(MAP_FAILED);
    Ok(())
}

pub(super) fn syscall_mprotect(m: &mut Machine) -> Result<()> {
    let addr = m.sysarg(0);
    let len = m.sysarg(1);
    let prot = m.sysarg(2) as i32;
    if !addr.is_multiple_of(PAGE_SIZE) {
        m.set_result(neg(EINVAL));
        return Ok(());
    }
    if addr.checked_add(len).is_none() {
        m.set_result(neg(ENOMEM));
        return Ok(());
    }
    m.memory
        .set_page_attr(addr, len, PageAttributes::from_prot(prot))?;
    m.set_result(0);
    Ok(())
}

pub(super) fn syscall_madvise(m: &mut Machine) -> Result<()> {
    let addr = m.sysarg(0);
    let len = m.sysarg(1);
    let advice = m.sysarg(2) as i32;
    match advice {
        // NORMAL, RANDOM, SEQUENTIAL, WILLNEED, DONTFORK, DOFORK,
        // MERGEABLE, NOHUGEPAGE, WIPEONFORK
        0..=3 | 10 | 11 | 12 | 15 | 18 => m.set_result(0),
        4 => {
            // MADV_DONTNEED
            m.memory.memdiscard(addr, len, true)?;
            m.set_result(0);
        }
        8 | 9 => {
            // MADV_FREE, MADV_REMOVE
            m.memory.free_pages(addr, len);
            m.set_result(0);
        }
        // MADV_GUARD_INSTALL/REMOVE: unsupported, as on Linux < 6.13. The
        // -1 is a work-around for Zig.
        102 | 103 | -1 => m.set_result(neg(EINVAL)),
        _ => {
            return Err(MachineError::new(
                ExceptionKind::SystemCallFailed,
                "Unimplemented madvise() advice",
                advice as u64,
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[cfg(feature = "jit")]
mod tests {
    use super::*;
    use crate::MachineOptions;
    use crate::jit_cache::CacheDir;
    use crate::registers::REG_ARG0;
    use crate::syscalls::fds::OpenFile;

    const PIN: u64 = 32 << 20;
    const PROT_RW: u64 = 3;
    const PROT_RX: u64 = 5;
    const MAP_PRIVATE: u64 = 2;
    const FIXED: u64 = MAP_PRIVATE | MAP_FIXED as u64;

    fn call(m: &mut Machine, f: fn(&mut Machine) -> Result<()>, args: &[u64]) -> u64 {
        for (i, &a) in (0u32..).zip(args) {
            m.cpu.set_reg(REG_ARG0 + i, a);
        }
        f(m).unwrap();
        m.return_value()
    }

    /// Whether the segment of `len` bytes at `vaddr` in `m`'s memory
    /// would be saved to `dir` (not initial, not a rebuild).
    fn saves(m: &Machine, dir: &CacheDir, vaddr: u64, len: usize) -> bool {
        let mut code = vec![0; len];
        m.copy_from_guest(&mut code, vaddr).unwrap();
        m.memory.saves_to_cache_dir(dir, vaddr, &code, false, false)
    }

    /// A guest can't get code of its own saved to a JIT cache directory
    /// (where later runs would use it in place of a library: the key's
    /// hashes can be forged): only segments whose bytes are a pinned
    /// file's, mapped there, are saved. Its own code in the pinned range
    /// isn't, nor a library's code it changed.
    #[test]
    fn only_code_from_pinned_files_is_saved_to_the_cache_dir() {
        let dir =
            std::env::temp_dir().join(format!("libriscv-pinned-origin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lib = dir.join("lib.so");
        let contents: Vec<u8> = (0..3 * PAGE_SIZE).map(|i| (i * 7 + 3) as u8).collect();
        std::fs::write(&lib, &contents).unwrap();
        let path = lib.to_str().unwrap().to_owned();
        let mut m = Machine::new(
            [].into(),
            MachineOptions {
                memory_max: 64 << 20,
                ..Default::default()
            },
        )
        .unwrap();
        m.setup_linux_syscalls(true, false);
        let fds = m.fds_mut().unwrap();
        fds.pinned_mappings = vec![(path.clone(), PIN)];
        let file = OpenFile {
            file: std::fs::File::open(&lib).unwrap(),
            guest_path: path.clone(),
            path,
            access: 0,
        };
        let vfd = u64::from(fds.assign_file(file).unwrap().cast_unsigned());
        let cache = CacheDir {
            path: dir.join("cache"),
            initial_segment: false,
            ranges: std::iter::once(PIN..PIN + (16 << 20)).collect(),
        };
        let len = contents.len() as u64;

        // The guest writes the library's very bytes into the range itself:
        // not saved (nothing was mapped from the file).
        let anonymous = FIXED | MAP_ANONYMOUS as u64;
        let args = [PIN, len, PROT_RW, anonymous, u64::MAX, 0];
        assert_eq!(call(&mut m, syscall_mmap, &args), PIN);
        m.copy_to_guest(PIN, &contents).unwrap();
        call(&mut m, syscall_mprotect, &[PIN, len, PROT_RX]);
        assert!(!saves(&m, &cache, PIN, contents.len()));

        // The file mapped there: saved, all of it or a part.
        assert_eq!(
            call(&mut m, syscall_mmap, &[PIN, len, PROT_RX, FIXED, vfd, 0]),
            PIN
        );
        assert!(saves(&m, &cache, PIN, contents.len()));
        assert!(saves(&m, &cache, PIN + PAGE_SIZE, PAGE_SIZE as usize));
        // Not past the mapping, nor outside the ranges, nor as a rebuild.
        assert!(!saves(&m, &cache, PIN, contents.len() + PAGE_SIZE as usize));
        let elsewhere = CacheDir {
            ranges: Vec::new(),
            ..cache.clone()
        };
        assert!(!saves(&m, &elsewhere, PIN, contents.len()));
        assert!(
            !m.memory
                .saves_to_cache_dir(&cache, PIN, &contents, false, true)
        );

        // The guest changes a byte of the mapped file: that page is no
        // longer the file's, the others still are.
        call(&mut m, syscall_mprotect, &[PIN, len, PROT_RW]);
        m.copy_to_guest(PIN + 100, &[!contents[100]]).unwrap();
        call(&mut m, syscall_mprotect, &[PIN, len, PROT_RX]);
        assert!(!saves(&m, &cache, PIN, contents.len()));
        assert!(!saves(&m, &cache, PIN, PAGE_SIZE as usize));
        assert!(saves(&m, &cache, PIN + PAGE_SIZE, 2 * PAGE_SIZE as usize));

        // A mapping from an offset in the file is compared at that offset.
        let at = PIN + (1 << 20);
        let args = [at, PAGE_SIZE, PROT_RX, FIXED, vfd, PAGE_SIZE];
        assert_eq!(call(&mut m, syscall_mmap, &args), at);
        assert!(saves(&m, &cache, at, PAGE_SIZE as usize));
        // Anonymous memory over it: guest code again.
        let args = [at, PAGE_SIZE, PROT_RW, anonymous, u64::MAX, 0];
        assert_eq!(call(&mut m, syscall_mmap, &args), at);
        m.copy_to_guest(at, &[1, 2, 3, 4]).unwrap();
        assert!(!saves(&m, &cache, at, PAGE_SIZE as usize));

        // The initial segment (the host's program): only if the directory
        // says so.
        assert!(
            !m.memory
                .saves_to_cache_dir(&cache, 0x1000, &contents, true, false)
        );
        let initial = CacheDir {
            initial_segment: true,
            ..cache.clone()
        };
        assert!(
            m.memory
                .saves_to_cache_dir(&initial, 0x1000, &contents, true, false)
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
