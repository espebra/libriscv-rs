//! Port of `posix/filedesc.hpp`: the sandboxed file descriptor table.
//!
//! The guest sees virtual file descriptors. 0, 1 and 2 are the standard
//! streams, files opened by the guest get numbers from `FILE_D_BASE` up.
//! C++ maps them to raw host file descriptors and calls the host's POSIX
//! functions on them; here they own `std::fs::File`s, so there is no raw
//! file descriptor (and no `unsafe`) anywhere. What a guest can reach is
//! decided by the system call handlers (`permit_filesystem`, `proxy_mode`)
//! and the filters, which are closures instead of `std::function`s taking
//! the userdata pointer.

use std::collections::BTreeMap;
use std::fs::File;
use std::io;

/// Decides whether a path may be opened, and may rewrite it
/// (C++ `filter_open` / `filter_readlink`).
pub type PathFilter = Box<dyn FnMut(&mut String) -> bool + Send + Sync>;
/// Decides whether a path may be inspected (C++ `filter_stat`).
pub type StatFilter = Box<dyn FnMut(&str) -> bool + Send + Sync>;
/// Decides whether an ioctl request is allowed (C++ `filter_ioctl`).
pub type IoctlFilter = Box<dyn FnMut(u64) -> bool + Send + Sync>;

/// An open guest file.
#[derive(Debug)]
pub struct OpenFile {
    pub file: File,
    /// The host path it was opened with (what `filter_open` returned).
    pub path: String,
    /// The guest path it was opened as: absolute and normalised, before
    /// `filter_open` rewrote it. Paths relative to a directory descriptor
    /// are joined to it before they go through the filters.
    pub guest_path: String,
    /// The access mode bits (`O_ACCMODE`) it was opened with, for `F_GETFL`.
    pub access: i32,
}

/// What a virtual file descriptor refers to (C++ `translate`).
#[derive(Debug)]
pub enum Target<'a> {
    /// Host standard input, output or error.
    Stdio(i32),
    File(&'a mut OpenFile),
}

/// The file descriptor table and sandbox policy (C++ `FileDescriptors`).
pub struct FileDescriptors {
    files: BTreeMap<i32, OpenFile>,
    /// The working directory the guest sees (a fake root by default).
    pub cwd: String,
    file_counter: i32,
    /// Allow the path-based system calls (`openat`, `fstatat`,
    /// `faccessat`, `readlinkat`) at all. On its own this opens nothing:
    /// outside [`proxy_mode`](Self::proxy_mode), a path reaches the host
    /// only when [`filter_open`](Self::filter_open) (or `filter_stat`,
    /// `filter_readlink`) admits it.
    pub permit_filesystem: bool,
    /// Allow sockets (not ported yet; kept for API parity).
    pub permit_sockets: bool,
    /// Proxy mode: the guest may write to files and see host details
    /// (exact time), and with no filter installed every path is admitted.
    /// Disables most of the sandbox.
    pub proxy_mode: bool,
    /// Allow opening files that are neither regular files nor directories:
    /// devices (`/dev/urandom`, but also `/dev/tty`), FIFOs and sockets.
    /// Off by default, because opening or reading one can block the host
    /// thread indefinitely, outside any instruction limit. The type is
    /// checked on the file a path resolves to, so a symbolic link to such
    /// a file is refused too. Not in C++, which opens anything admitted.
    pub permit_special_files: bool,
    /// Decides which paths `openat` and `faccessat` may use (and `fstatat`
    /// without `filter_stat`). With no filter, every path is denied unless
    /// [`proxy_mode`](Self::proxy_mode) is on.
    ///
    /// The filter is given an absolute, lexically normalised guest path
    /// (see [`normalize_path`](Self::normalize_path)): `.`, `..` and
    /// repeated slashes are collapsed before it runs, and a path relative
    /// to a directory descriptor is joined to the directory's guest path
    /// first. So a prefix check like `path.starts_with("/srv/data/")`
    /// cannot be escaped with `..`. Relative paths through `AT_FDCWD` are
    /// `EBADF` (as in C++, the guest has no host working directory) and
    /// never reach the filter. The filter may rewrite the path: what it
    /// leaves in the string is the host path that is opened, as is (a
    /// relative one is relative to the host process's working directory).
    ///
    /// The host still resolves symbolic links when it opens the path, so
    /// a link inside an admitted directory leads wherever it points:
    /// admit only paths whose contents you trust. Files that are not
    /// regular files or directories are refused unless
    /// [`permit_special_files`](Self::permit_special_files) is set.
    pub filter_open: Option<PathFilter>,
    pub filter_readlink: Option<PathFilter>,
    pub filter_stat: Option<StatFilter>,
    pub filter_ioctl: Option<IoctlFilter>,
    /// Files whose `mmap` without an address (`ld.so` mapping a shared
    /// library) goes to a fixed guest address: (filtered path, address).
    /// The address is used when it is page aligned, the whole mapping fits
    /// in the arena and that range of the mmap area is free; otherwise the
    /// mapping goes where it would have gone. Not in C++: it keeps libraries
    /// at the same address in every program, so code saved for them (the
    /// JIT cache, AOT translations, both keyed by address) is found again.
    pub pinned_mappings: Vec<(String, u64)>,
}

impl core::fmt::Debug for FileDescriptors {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FileDescriptors")
            .field("files", &self.files)
            .field("cwd", &self.cwd)
            .field("permit_filesystem", &self.permit_filesystem)
            .field("permit_sockets", &self.permit_sockets)
            .field("proxy_mode", &self.proxy_mode)
            .field("permit_special_files", &self.permit_special_files)
            .finish_non_exhaustive()
    }
}

impl Default for FileDescriptors {
    fn default() -> Self {
        Self {
            files: BTreeMap::new(),
            cwd: "/home".to_owned(),
            file_counter: Self::FILE_D_BASE,
            permit_filesystem: false,
            permit_sockets: false,
            proxy_mode: false,
            permit_special_files: false,
            filter_open: None,
            filter_readlink: None,
            filter_stat: None,
            filter_ioctl: None,
            pinned_mappings: Vec::new(),
        }
    }
}

impl FileDescriptors {
    /// The first virtual file descriptor handed out for files.
    pub const FILE_D_BASE: i32 = 0x1000;
    /// Files the guest may have open at once (C++ has no limit). Guests
    /// leaking descriptors would otherwise exhaust the host's.
    pub const MAX_FILES: usize = 1024;

    /// Lexically normalise a guest path into the absolute form the filters
    /// are given: empty, `.` and `..` components are collapsed (`..` at the
    /// root stays at the root), and the result starts with `/`. A relative
    /// path is treated as relative to `/`. A trailing slash (or a final
    /// `.` or `..`) is kept as a trailing slash, so `"/etc/hosts/"` stays
    /// distinct from `"/etc/hosts"`. Symbolic links are not consulted: the
    /// normalised path is the one that is opened.
    pub fn normalize_path(path: &str) -> String {
        let mut parts: Vec<&str> = Vec::new();
        let mut trailing = false;
        for component in path.split('/') {
            trailing = matches!(component, "" | "." | "..");
            match component {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                c => parts.push(c),
            }
        }
        let mut out = String::with_capacity(path.len() + 1);
        for part in parts {
            out.push('/');
            out.push_str(part);
        }
        if out.is_empty() || trailing {
            out.push('/');
        }
        out
    }

    /// Register an open file and return its virtual file descriptor, or
    /// `None` when the table is full.
    pub fn assign_file(&mut self, file: OpenFile) -> Option<i32> {
        if self.files.len() >= Self::MAX_FILES {
            return None;
        }
        // Skip numbers still in use after the counter wrapped.
        loop {
            let vfd = self.file_counter;
            self.file_counter = match self.file_counter.checked_add(1) {
                Some(n) => n,
                None => Self::FILE_D_BASE,
            };
            if let std::collections::btree_map::Entry::Vacant(e) = self.files.entry(vfd) {
                e.insert(file);
                return Some(vfd);
            }
        }
    }

    /// What `vfd` refers to: an open file, or one of the standard streams
    /// (C++ `translate`, which only lets 0, 1 and 2 through to the host).
    pub fn translate(&mut self, vfd: i32) -> Option<Target<'_>> {
        if let Some(f) = self.files.get_mut(&vfd) {
            return Some(Target::File(f));
        }
        (0..=2).contains(&vfd).then_some(Target::Stdio(vfd))
    }

    /// An open file (C++ `get`, which does not let the standard streams through).
    pub fn get(&mut self, vfd: i32) -> Option<&mut OpenFile> {
        self.files.get_mut(&vfd)
    }

    /// Close a virtual file descriptor (C++ `erase`). Returns false if it
    /// was not open.
    pub fn erase(&mut self, vfd: i32) -> bool {
        self.files.remove(&vfd).is_some()
    }

    /// Number of open files.
    pub fn len(&self) -> usize {
        self.files.len()
    }
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Writing to files is only allowed in proxy mode (C++ `permit_write`;
    /// sockets are not ported yet).
    pub const fn permit_write(&self) -> bool {
        self.proxy_mode
    }
}

/// A host file handle for standard stream `n`, for `fstat`.
pub(crate) fn stdio_file(n: i32) -> io::Result<File> {
    use std::os::fd::AsFd;
    let owned = match n {
        0 => io::stdin().as_fd().try_clone_to_owned()?,
        1 => io::stdout().as_fd().try_clone_to_owned()?,
        _ => io::stderr().as_fd().try_clone_to_owned()?,
    };
    Ok(File::from(owned))
}

/// Size of the RISC-V `struct stat` (C++ `riscv_stat`).
pub(crate) const RISCV_STAT_SIZE: usize = 128;

/// Host metadata as the RISC-V `struct stat` (C++ `copy_stat_buffer`).
pub(crate) fn riscv_stat(md: &std::fs::Metadata) -> [u8; RISCV_STAT_SIZE] {
    use std::os::unix::fs::MetadataExt;
    let mut b = [0u8; RISCV_STAT_SIZE];
    let mut put = |ofs: usize, bytes: &[u8]| {
        if let Some(dst) = b.get_mut(ofs..ofs + bytes.len()) {
            dst.copy_from_slice(bytes);
        }
    };
    put(0, &md.dev().to_le_bytes());
    put(8, &md.ino().to_le_bytes());
    put(16, &md.mode().to_le_bytes());
    put(20, &(md.nlink() as u32).to_le_bytes());
    put(24, &md.uid().to_le_bytes());
    put(28, &md.gid().to_le_bytes());
    put(32, &md.rdev().to_le_bytes());
    put(48, &(md.size() as i64).to_le_bytes());
    put(56, &(md.blksize() as i32).to_le_bytes());
    put(64, &(md.blocks() as i64).to_le_bytes());
    put(72, &md.atime().to_le_bytes());
    put(80, &(md.atime_nsec() as u64).to_le_bytes());
    put(88, &md.mtime().to_le_bytes());
    put(96, &(md.mtime_nsec() as u64).to_le_bytes());
    put(104, &md.ctime().to_le_bytes());
    put(112, &(md.ctime_nsec() as u64).to_le_bytes());
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_streams_translate() {
        let mut fds = FileDescriptors::default();
        assert!(matches!(fds.translate(1), Some(Target::Stdio(1))));
        assert!(fds.translate(3).is_none());
        assert!(fds.translate(-100).is_none());
        assert!(fds.get(0).is_none());
        assert!(!fds.erase(0x1000));
    }

    #[test]
    fn files_get_virtual_numbers() {
        let mut fds = FileDescriptors::default();
        let file = File::open("/dev/null").unwrap();
        let vfd = fds
            .assign_file(OpenFile {
                file,
                path: "/dev/null".into(),
                guest_path: "/dev/null".into(),
                access: 0,
            })
            .unwrap();
        assert_eq!(vfd, FileDescriptors::FILE_D_BASE);
        assert!(matches!(fds.translate(vfd), Some(Target::File(_))));
        assert!(fds.erase(vfd));
        assert!(fds.translate(vfd).is_none());
    }

    #[test]
    fn paths_are_normalised() {
        let n = FileDescriptors::normalize_path;
        assert_eq!(n("/etc/hosts"), "/etc/hosts");
        assert_eq!(n("//etc///./hosts"), "/etc/hosts");
        assert_eq!(n("/srv/data/../../etc/shadow"), "/etc/shadow");
        assert_eq!(n("/../../.."), "/");
        assert_eq!(n("/a/b/.."), "/a/");
        assert_eq!(n("/a/b/."), "/a/b/");
        assert_eq!(n("/a/b/"), "/a/b/");
        assert_eq!(n("/"), "/");
        assert_eq!(n(""), "/");
        assert_eq!(n("a/./b"), "/a/b");
        assert_eq!(n("/a/..b/.c"), "/a/..b/.c");
    }
}
