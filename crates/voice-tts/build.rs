//! Build-time check for Git LFS data.
//!
//! The embedded voice files (`data/voices/*.safetensors`) are tracked by Git
//! LFS. If `git lfs` wasn't installed before cloning, these files will contain
//! tiny LFS pointers instead of actual safetensor weights. The build will
//! succeed (the bytes are valid for `include_bytes!`), but the binary will
//! fail at runtime when it tries to deserialize the pointers as safetensors.
//!
//! This script catches that at build time with a clear error message.

use std::path::Path;

/// LFS pointer files always start with this line.
const LFS_SIGNATURE: &[u8] = b"version https://git-lfs.github.com/spec/v1";

fn main() {
    let lfs_files: &[&str] = &[
        "data/voices/af_heart.safetensors",
        "data/voices/af_bella.safetensors",
        "data/voices/af_sarah.safetensors",
        "data/voices/af_sky.safetensors",
        "data/voices/am_michael.safetensors",
        "data/voices/am_adam.safetensors",
        "data/voices/bf_emma.safetensors",
    ];

    for rel_path in lfs_files {
        let path = Path::new(rel_path);

        // Re-run this check if the file changes (e.g. after `git lfs pull`).
        println!("cargo:rerun-if-changed={}", rel_path);

        let contents = match std::fs::read(path) {
            Ok(c) => c,
            Err(e) => {
                println!(
                    "cargo:warning=Could not read {}: {}. \
                     Build may fail if the file is missing.",
                    rel_path, e
                );
                continue;
            }
        };

        if contents.starts_with(LFS_SIGNATURE) {
            eprintln!();
            eprintln!(
                "error: {} is a Git LFS pointer, not the actual data.",
                rel_path
            );
            eprintln!();
            eprintln!("  Git Large File Storage (git-lfs) was not installed when you");
            eprintln!("  cloned this repository. The file contains:");
            eprintln!();
            // LFS pointers are small text — safe to display as UTF-8.
            let text = String::from_utf8_lossy(&contents);
            for line in text.lines().take(3) {
                eprintln!("    {}", line);
            }
            eprintln!();
            eprintln!("  To fix this, install git-lfs and re-fetch the data:");
            eprintln!();
            eprintln!("    brew install git-lfs");
            eprintln!("    git lfs install");
            eprintln!("    git lfs pull");
            eprintln!();
            eprintln!("  Then rebuild:");
            eprintln!();
            eprintln!("    cargo install --path crates/voice-cli");
            eprintln!();
            std::process::exit(1);
        }
    }
}
