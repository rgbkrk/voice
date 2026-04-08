//! Generate the bronze pronunciation dictionary by running espeak-ng in batch
//! mode against a word list, then applying E2M conversion to produce Kokoro
//! phonemes. The output is a JSON file suitable for embedding in voice-g2p.
//!
//! Usage:
//!   cargo run --bin generate-bronze -- [OPTIONS]
//!
//! Options:
//!   --wordlist <PATH>    Word list file (default: /usr/share/dict/words)
//!   --output <PATH>      Output JSON file (default: data/us_bronze.json)
//!   --espeak-path <PATH> Path to espeak-ng binary (default: espeak-ng)

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::process::{Command, Stdio};

use voice_g2p::espeak::apply_e2m_us;

const US_GOLD_JSON: &str = include_str!("../../data/us_gold.json");
const US_SILVER_JSON: &str = include_str!("../../data/us_silver.json");

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let wordlist_path = get_arg(&args, "--wordlist")
        .unwrap_or_else(|| "/usr/share/dict/words".to_string());
    let output_path = get_arg(&args, "--output")
        .unwrap_or_else(|| "data/us_bronze.json".to_string());
    let espeak_path = get_arg(&args, "--espeak-path")
        .unwrap_or_else(|| "espeak-ng".to_string());

    // Load existing dictionaries to skip known words
    eprintln!("Loading gold and silver dictionaries...");
    let gold: serde_json::Value =
        serde_json::from_str(US_GOLD_JSON).expect("failed to parse us_gold.json");
    let silver: serde_json::Value =
        serde_json::from_str(US_SILVER_JSON).expect("failed to parse us_silver.json");

    let mut known: HashSet<String> = HashSet::new();
    if let serde_json::Value::Object(map) = &gold {
        for key in map.keys() {
            known.insert(key.to_lowercase());
        }
    }
    if let serde_json::Value::Object(map) = &silver {
        for key in map.keys() {
            known.insert(key.to_lowercase());
        }
    }
    eprintln!("  {} known words from gold+silver", known.len());

    // Read and filter word list
    eprintln!("Reading word list from {}...", wordlist_path);
    let wordlist_file =
        std::fs::read_to_string(&wordlist_path).expect("failed to read word list");

    let mut words: Vec<String> = wordlist_file
        .lines()
        .map(|l| l.trim().to_lowercase())
        .filter(|w| !w.is_empty())
        .filter(|w| w.chars().all(|c| c.is_ascii_alphabetic()))
        .filter(|w| !known.contains(w))
        .collect();

    words.sort();
    words.dedup();
    eprintln!("  {} new words to process", words.len());

    // Process in chunks to avoid pipe buffer deadlock.
    // Each chunk spawns espeak-ng, writes words to stdin, reads all stdout.
    const CHUNK_SIZE: usize = 5000;
    let mut result: BTreeMap<String, String> = BTreeMap::new();
    let mut skipped = 0;

    eprintln!("Running espeak-ng in chunks of {}...", CHUNK_SIZE);
    for (chunk_idx, chunk) in words.chunks(CHUNK_SIZE).enumerate() {
        let chunk_start = chunk_idx * CHUNK_SIZE;
        eprint!(
            "\r  {}/{} words processed...",
            chunk_start,
            words.len()
        );

        let mut child = Command::new(&espeak_path)
            .args(["--ipa", "-q", "-v", "en-us", "--tie=^"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to start espeak-ng — is it installed?");

        {
            let stdin = child.stdin.as_mut().expect("failed to open stdin");
            for word in chunk {
                writeln!(stdin, "{}", word).expect("failed to write to stdin");
            }
        }

        let output = child.wait_with_output().expect("espeak-ng failed");
        if !output.status.success() {
            eprintln!("\nespeak-ng failed on chunk {}, falling back to per-word", chunk_idx);
            let fallback = process_per_word(chunk, &espeak_path);
            result.extend(fallback);
            continue;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = stdout.lines().collect();

        if lines.len() != chunk.len() {
            eprintln!(
                "\nWARNING: chunk {} got {} lines for {} words, falling back",
                chunk_idx, lines.len(), chunk.len()
            );
            let fallback = process_per_word(chunk, &espeak_path);
            result.extend(fallback);
            continue;
        }

        for (word, ipa_line) in chunk.iter().zip(lines.iter()) {
            let ipa = ipa_line.trim();
            if ipa.is_empty() {
                skipped += 1;
                continue;
            }
            let phonemes = apply_e2m_us(ipa);
            if phonemes.trim().is_empty() {
                skipped += 1;
                continue;
            }
            result.insert(word.clone(), phonemes);
        }
    }

    eprintln!(
        "\r  {} entries generated, {} skipped (empty output)     ",
        result.len(),
        skipped
    );

    write_output(&result, &output_path);
}

/// Fallback: process words one at a time if batch mode has line count mismatch.
fn process_per_word(words: &[String], espeak_path: &str) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    let total = words.len();

    for (i, word) in words.iter().enumerate() {
        if i % 10000 == 0 {
            eprintln!("  processing word {}/{}", i, total);
        }
        let output = Command::new(espeak_path)
            .args(["--ipa", "-q", "-v", "en-us", "--tie=^", word])
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let raw = String::from_utf8_lossy(&output.stdout);
                let ipa = raw.trim();
                if !ipa.is_empty() {
                    let phonemes = apply_e2m_us(ipa);
                    if !phonemes.trim().is_empty() {
                        result.insert(word.clone(), phonemes);
                    }
                }
            }
        }
    }

    result
}

fn write_output(result: &BTreeMap<String, String>, output_path: &str) {
    let json =
        serde_json::to_string_pretty(result).expect("failed to serialize JSON");
    std::fs::write(output_path, &json).expect("failed to write output file");
    let size_mb = json.len() as f64 / 1_048_576.0;
    eprintln!(
        "Wrote {} entries to {} ({:.1} MB)",
        result.len(),
        output_path,
        size_mb
    );
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
