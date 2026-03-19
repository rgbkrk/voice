//! Build-time check for Git LFS data.
//!
//! The tagger weights file (`data/tagger/weights.json`) is tracked by Git LFS.
//! If `git lfs` wasn't installed before cloning, the file will contain a tiny
//! LFS pointer instead of the actual JSON weights. The build will succeed
//! (it's valid UTF-8, so `include_str!` is happy), but the binary will panic
//! at runtime when it tries to parse the pointer as JSON.
//!
//! This script catches that at build time with a clear error message.

use std::path::Path;

/// LFS pointer files always start with this line.
const LFS_SIGNATURE: &str = "version https://git-lfs.github.com/spec/v1";

fn main() {
    let lfs_files: &[&str] = &["data/tagger/weights.json"];

    for rel_path in lfs_files {
        let path = Path::new(rel_path);

        // Re-run this check if the file changes (e.g. after `git lfs pull`).
        println!("cargo:rerun-if-changed={}", rel_path);

        let contents = match std::fs::read_to_string(path) {
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
            for line in contents.lines().take(3) {
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
