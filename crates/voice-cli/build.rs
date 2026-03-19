//! Build script for voice CLI.
//!
//! Finds the `mlx.metallib` compiled by mlx-sys during the build and copies it
//! to a stable location (`~/.mlx/lib/`) that persists after `cargo install`
//! cleans up its temporary build directory.
//!
//! At runtime, `main()` copies the metallib from `~/.mlx/lib/` to sit next to
//! the voice binary so that MLX's co-located library search finds it before
//! falling back to the (potentially broken) compile-time `METAL_PATH`.

use std::path::{Path, PathBuf};
use std::{env, fs};

/// Walk up from a starting directory looking for `mlx-sys-*/out/build/lib/mlx.metallib`.
fn find_metallib(build_dir: &Path) -> Option<PathBuf> {
    // build_dir is something like: target/release/build/voice-HASH/out
    // We want:                     target/release/build/mlx-sys-HASH/out/build/lib/mlx.metallib
    // So go up to target/release/build/ and glob from there.
    let build_root = build_dir
        .parent()? // voice-HASH/
        .parent()?; // build/

    let entries = fs::read_dir(build_root).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("mlx-sys-") {
            // Check the known locations where CMake puts the metallib
            let candidates = [
                entry.path().join("out/build/lib/mlx.metallib"),
                entry
                    .path()
                    .join("out/build/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib"),
            ];
            for candidate in &candidates {
                if candidate.exists() {
                    return Some(candidate.clone());
                }
            }
        }
    }

    None
}

fn stable_metallib_dir() -> Option<PathBuf> {
    let home = env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".mlx").join("lib"))
}

fn main() {
    let out_dir = match env::var("OUT_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => return,
    };

    // Find the metallib from mlx-sys's build output
    let metallib_src = match find_metallib(&out_dir) {
        Some(p) => p,
        None => {
            println!(
                "cargo:warning=Could not find mlx.metallib in build output. \
                 `cargo install` users may need to copy it manually."
            );
            return;
        }
    };

    // Copy to stable location
    let stable_dir = match stable_metallib_dir() {
        Some(d) => d,
        None => return,
    };

    if let Err(e) = fs::create_dir_all(&stable_dir) {
        println!(
            "cargo:warning=Failed to create {}: {}",
            stable_dir.display(),
            e
        );
        return;
    }

    let target = stable_dir.join("mlx.metallib");
    match fs::copy(&metallib_src, &target) {
        Ok(_) => {
            println!(
                "cargo:warning=Copied mlx.metallib to {} for runtime discovery",
                target.display()
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=Failed to copy mlx.metallib to {}: {}",
                target.display(),
                e
            );
        }
    }
}
