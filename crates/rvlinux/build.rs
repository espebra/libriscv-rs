//! Link JIT objects listed in `RVLINUX_JIT_EMBED` into `rvlinux` (`jit`).
//! Paths use the platform's `PATH` separator. Generate objects with
//! `rvlinux --jit-save-object`; linked code is discovered automatically.
//!
//! Compile and register Rust translations listed in `RVLINUX_AOT_EMBED`
//! with `aot`. Generate source with `rvlinux --aot-save`. Paths use the same
//! separator; embedded translations contain only safe Rust.

use std::fmt::Write as _;

fn main() {
    embed_translations();
    link_jit_code();
}

fn embed_translations() {
    println!("cargo:rustc-check-cfg=cfg(rvlinux_aot_embed)");
    println!("cargo:rerun-if-env-changed=RVLINUX_AOT_EMBED");
    let Some(files) = std::env::var_os("RVLINUX_AOT_EMBED") else {
        return;
    };
    let files: Vec<_> = std::env::split_paths(&files)
        .filter(|f| !f.as_os_str().is_empty())
        .collect();
    if files.is_empty() {
        return;
    }
    assert!(
        std::env::var_os("CARGO_FEATURE_AOT").is_some(),
        "RVLINUX_AOT_EMBED needs the `aot` feature"
    );
    // One module per translation (each defines `mod translation`), and a
    // function registering them all.
    let mut code = String::new();
    let mut register = String::from("pub(crate) fn register() -> usize {\n    0");
    for (n, file) in files.iter().enumerate() {
        let file = std::fs::canonicalize(file).unwrap_or_else(|_| file.clone());
        println!("cargo:rerun-if-changed={}", file.display());
        let _ = writeln!(code, "mod t{n} {{\n    include!({:?});\n}}", file);
        let _ = write!(
            register,
            "\n        + libriscv::aot::register(t{n}::translation::META, t{n}::translation::enter)"
        );
    }
    register.push_str("\n}\n");
    code.push_str(&register);
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap_or_default());
    if let Err(e) = std::fs::write(out.join("aot_embed.rs"), code) {
        eprintln!("writing aot_embed.rs: {e}");
        std::process::exit(1);
    }
    println!("cargo:rustc-cfg=rvlinux_aot_embed");
}

fn link_jit_code() {
    println!("cargo:rerun-if-env-changed=RVLINUX_JIT_EMBED");
    let Some(objects) = std::env::var_os("RVLINUX_JIT_EMBED") else {
        return;
    };
    for object in std::env::split_paths(&objects) {
        if object.as_os_str().is_empty() {
            continue;
        }
        let object = std::fs::canonicalize(&object).unwrap_or(object);
        println!("cargo:rerun-if-changed={}", object.display());
        println!("cargo:rustc-link-arg-bins={}", object.display());
    }
}
