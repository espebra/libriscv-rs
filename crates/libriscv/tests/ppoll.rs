//! ppoll: timeouts are validated, descriptors the guest does not have are
//! invalid (and ready), negative ones are ignored, and revents is always
//! written back. Port of the C14 ppoll cases of `tests/unit/security.cpp`
//! that do not need a host pipe (here descriptors are never forwarded to
//! the host, and nothing waits).

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use libriscv::registers::REG_ARG0;
use libriscv::{Machine, MachineOptions};

const FDS: u64 = 0x10000;
const TS: u64 = 0x10800;
const PPOLL: u64 = 73;
const POLLIN: i16 = 0x1;
const POLLOUT: i16 = 0x4;
const POLLNVAL: i16 = 0x20;
const EINVAL: i64 = 22;

fn machine() -> Machine {
    let mut m = Machine::new([].into(), MachineOptions::default()).unwrap();
    // With a file descriptor table, as in C++.
    m.setup_linux_syscalls(true, false);
    m
}

/// ppoll on `(fd, events, revents)` with an optional `(sec, nsec)` timeout:
/// the result and the revents written back.
fn ppoll(m: &mut Machine, fds: &[(i32, i16, i16)], ts: Option<(i64, i64)>) -> (i64, Vec<i16>) {
    let raw: Vec<u8> = fds
        .iter()
        .flat_map(|&(fd, events, revents)| {
            let mut pfd = fd.to_le_bytes().to_vec();
            pfd.extend(events.to_le_bytes());
            pfd.extend(revents.to_le_bytes());
            pfd
        })
        .collect();
    m.copy_to_guest(FDS, &raw).unwrap();
    if let Some((sec, nsec)) = ts {
        let mut t = sec.to_le_bytes().to_vec();
        t.extend(nsec.to_le_bytes());
        m.copy_to_guest(TS, &t).unwrap();
    }
    let args = [FDS, fds.len() as u64, ts.map_or(0, |_| TS), 0];
    for (i, &a) in args.iter().enumerate() {
        m.cpu.set_reg(REG_ARG0 + i as u32, a);
    }
    m.system_call(PPOLL).unwrap();
    let mut out = vec![0u8; raw.len()];
    m.memory.memcpy_out(&mut out, FDS).unwrap();
    let revents = out
        .chunks(8)
        .map(|c| i16::from_le_bytes([c[6], c[7]]))
        .collect();
    (m.return_value() as i64, revents)
}

#[test]
fn ppoll_timeouts_are_validated() {
    let mut m = machine();
    let fds = [(1, POLLOUT, 0)];
    assert_eq!(ppoll(&mut m, &fds, Some((0, 1_000_000_000))).0, -EINVAL);
    assert_eq!(ppoll(&mut m, &fds, Some((-1, 0))).0, -EINVAL);
    assert_eq!(ppoll(&mut m, &fds, Some((0, -1))).0, -EINVAL);
    // A NULL timeout (an infinite wait) and a huge one are fine.
    assert_eq!(ppoll(&mut m, &fds, None), (1, vec![POLLOUT]));
    let huge = Some((i64::MAX, 999_999_999));
    assert_eq!(ppoll(&mut m, &fds, huge), (1, vec![POLLOUT]));
}

#[test]
fn ppoll_on_descriptors_the_guest_does_not_have() {
    let mut m = machine();
    // Unknown descriptors are invalid and count as ready; negative ones
    // are ignored.
    let fds = [(1234, POLLIN, 0), (-1, POLLIN, 0), (1, POLLOUT, 0)];
    assert_eq!(ppoll(&mut m, &fds, None), (2, vec![POLLNVAL, 0, POLLOUT]));
}

#[test]
fn ppoll_always_writes_back_revents() {
    let mut m = machine();
    // Nothing is ready (stdout is not readable): the stale revents is
    // cleared anyway.
    let fds = [(1, POLLIN, POLLIN), (-5, POLLIN, POLLOUT)];
    assert_eq!(ppoll(&mut m, &fds, Some((0, 0))), (0, vec![0, 0]));
}
