//! Not a port: checks the contents of the memory arena against a plain
//! byte array through writes and discards (`memdiscard`, what
//! `madvise(MADV_DONTNEED)`, anonymous `mmap` and `munmap` do). With the
//! `jit` feature the arena is a mapping and large discards hand its pages
//! back to the kernel (`GuestArena::zero_pages`); a discard that zeroes too
//! much, too little or the wrong pages shows up here as a mismatch, which
//! the other harnesses (they only look for panics) would not notice.

use libriscv::{Machine, MachineOptions};

/// The arena: big enough for discards well above the size where pages are
/// dropped (256 KiB), small enough to compare after every input quickly.
const ARENA: u64 = 8 << 20;
/// Below this the arena is not writable without a program.
const LOW: u64 = 0x10_000;

struct Input<'a>(&'a [u8]);

impl Input<'_> {
    fn byte(&mut self) -> Option<u8> {
        let (&b, rest) = self.0.split_first()?;
        self.0 = rest;
        Some(b)
    }

    fn u32(&mut self) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..4 {
            v = (v << 8) | u32::from(self.byte()?);
        }
        Some(v)
    }

    /// An address in or a little past the arena, often page-aligned (as a
    /// guest's `madvise` is) and sometimes 2 MiB-aligned.
    fn addr(&mut self) -> Option<u64> {
        let kind = self.byte()?;
        let raw = u64::from(self.u32()?) % (ARENA + (1 << 20));
        Some(match kind % 4 {
            0 => raw,
            1 | 2 => raw & !0xfff,
            _ => raw & !0x1f_ffff,
        })
    }

    /// A length: small, page multiples, or up to the whole arena and past.
    fn len(&mut self) -> Option<u64> {
        let kind = self.byte()?;
        let raw = u64::from(self.u32()?);
        Some(match kind % 4 {
            0 => raw % 0x2000,
            1 => (raw % 0x400) << 12,
            2 => raw % (2 * ARENA),
            _ => u64::MAX - raw,
        })
    }
}

/// The part of `[addr, addr + len)` inside the shadow array.
fn clip(addr: u64, len: u64) -> std::ops::Range<usize> {
    let end = addr.saturating_add(len).min(ARENA);
    let begin = addr.min(end);
    begin as usize..end as usize
}

/// Run the operations in `data` on a machine's arena and on a shadow copy,
/// and panic where they differ.
pub fn fuzz_arena_discard(data: &[u8]) {
    let options = MachineOptions {
        memory_max: ARENA,
        ..MachineOptions::default()
    };
    let Ok(mut m) = Machine::new([].into(), options) else {
        return;
    };
    if m.memory.memory_arena_size() != ARENA {
        return;
    }
    let mut shadow = vec![0u8; ARENA as usize];
    let mut input = Input(data);
    let mut ops = 0;
    while let Some(op) = input.byte() {
        ops += 1;
        if ops > 64 {
            break;
        }
        match op % 4 {
            0 | 1 => {
                // Fill (the guest writing): only where the arena is
                // writable, so that a failed write cannot desync the shadow.
                let (Some(addr), Some(len), Some(value)) =
                    (input.addr(), input.len(), input.byte())
                else {
                    break;
                };
                let addr = addr.max(LOW);
                let r = clip(addr, len.min(1 << 20));
                if r.is_empty() {
                    continue;
                }
                let n = (r.end - r.start) as u64;
                if op % 4 == 0 {
                    m.memory
                        .memset(addr, value, n)
                        .expect("memset in the arena");
                    shadow[r].fill(value);
                } else {
                    let bytes: Vec<u8> = (0..n).map(|i| value ^ (i as u8)).collect();
                    m.memory.memcpy(addr, &bytes).expect("memcpy in the arena");
                    shadow[r].copy_from_slice(&bytes);
                }
            }
            _ => {
                let (Some(addr), Some(len)) = (input.addr(), input.len()) else {
                    break;
                };
                if m.memory.memdiscard(addr, len, true).is_ok() {
                    shadow[clip(addr, len)].fill(0);
                } else {
                    // A failed discard may have zeroed part of the range:
                    // resynchronise from the arena (and check nothing else).
                    return;
                }
            }
        }
    }
    let view = m
        .memory
        .memview(LOW, ARENA - LOW)
        .expect("the arena is readable");
    if let Some(i) = view
        .iter()
        .zip(&shadow[LOW as usize..])
        .position(|(a, b)| a != b)
    {
        let at = LOW + i as u64;
        panic!(
            "arena byte {at:#x} is {:#x}, expected {:#x}",
            view[i], shadow[at as usize]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An address operand: kind 0 (as is), then the value.
    fn addr(a: u32) -> Vec<u8> {
        let mut v = vec![0];
        v.extend(a.to_be_bytes());
        v
    }

    /// A length operand: kind 2 (`% 2 * ARENA`), then the value.
    fn len(n: u32) -> Vec<u8> {
        let mut v = vec![2];
        v.extend(n.to_be_bytes());
        v
    }

    #[test]
    fn large_discards_between_writes_match_the_shadow() {
        let mut data = Vec::new();
        // Fill most of the arena, discard an unaligned 3 MiB range in the
        // middle (dropped pages plus hand-zeroed edges), refill part of it,
        // and discard past the end of the arena.
        for (start, value) in [(0x10_000, 0xaa), (0x110_000, 0xbb), (0x600_000, 0xcc)] {
            data.push(0);
            data.extend(addr(start));
            data.extend(len(0xf_ffff));
            data.push(value);
        }
        data.push(2);
        data.extend(addr(0x12_345));
        data.extend(len(0x30_0123));
        data.push(1);
        data.extend(addr(0x20_0800));
        data.extend(len(0x5000));
        data.push(7);
        data.push(3);
        data.extend(addr(0x70_0100));
        data.extend(len(0x40_0000));
        fuzz_arena_discard(&data);
    }

    #[test]
    fn a_large_discard_keeps_the_neighbouring_pages() {
        let options = MachineOptions {
            memory_max: ARENA,
            ..MachineOptions::default()
        };
        let mut m = Machine::new([].into(), options).unwrap();
        m.memory.memset(0x40_0000, 1, 0x40_0000).unwrap();
        m.memory.memdiscard(0x40_1000, 0x30_0000, true).unwrap();
        let view = m.memory.memview(0x40_0000, 0x40_0000).unwrap();
        // The page below and above the discard keep their bytes.
        assert!(view[..0x1000].iter().all(|&b| b == 1));
        assert!(view[0x1000..0x30_1000].iter().all(|&b| b == 0));
        assert!(view[0x30_1000..].iter().all(|&b| b == 1));
    }
}
