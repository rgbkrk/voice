//! Embedded model config and tokenizer for instant startup.
//!
//! The distil-medium.en, distil-large-v3, and distil-large-v3.5 configs and
//! tokenizers are compiled into the binary via `include_str!`, enabling instant
//! model initialization with no network or filesystem access for metadata.
//! Only the model weights are downloaded from HuggingFace Hub on first run.

use crate::error::{Result, SttError};

/// Default HuggingFace repo ID for STT.
///
/// distil-large-v3.5: English transcription on the large-v3 architecture
/// (~1.5GB weights). Drop-in upgrade over distil-large-v3 with markedly fewer
/// repetition errors on trailing silence. Override with `STT_MODEL`, e.g.
/// `openai/whisper-large-v3` for multilingual or
/// `distil-whisper/distil-medium.en` for smaller/faster.
pub const DEFAULT_MODEL_REPO: &str = "distil-whisper/distil-large-v3.5";

// Embedded data for zero-fetch init
const DISTIL_MEDIUM_EN_CONFIG: &str = include_str!("../data/distil_medium_en_config.json");
const DISTIL_MEDIUM_EN_TOKENIZER: &str = include_str!("../data/distil_medium_en_tokenizer.json");
const DISTIL_LARGE_V3_CONFIG: &str = include_str!("../data/distil_large_v3_config.json");
const DISTIL_LARGE_V3_TOKENIZER: &str = include_str!("../data/distil_large_v3_tokenizer.json");
const DISTIL_LARGE_V3_5_CONFIG: &str = include_str!("../data/distil_large_v3_5_config.json");
const DISTIL_LARGE_V3_5_TOKENIZER: &str = include_str!("../data/distil_large_v3_5_tokenizer.json");

/// Known model repo IDs and whether they are multilingual.
pub fn is_multilingual(repo_id: &str) -> bool {
    let en_only = [
        "openai/whisper-tiny.en",
        "openai/whisper-base.en",
        "openai/whisper-small.en",
        "openai/whisper-medium.en",
        "distil-whisper/distil-medium.en",
    ];
    !en_only.contains(&repo_id)
}

/// Get the embedded config for a repo ID, if available.
pub fn config_for_repo(repo_id: &str) -> Option<Result<voice_whisper::Config>> {
    let json = match repo_id {
        "distil-whisper/distil-medium.en" => DISTIL_MEDIUM_EN_CONFIG,
        "distil-whisper/distil-large-v3" => DISTIL_LARGE_V3_CONFIG,
        "distil-whisper/distil-large-v3.5" => DISTIL_LARGE_V3_5_CONFIG,
        _ => return None,
    };
    Some(serde_json::from_str(json).map_err(SttError::Json))
}

/// Get the embedded tokenizer for a repo ID, if available.
pub fn tokenizer_for_repo(repo_id: &str) -> Option<Result<tokenizers::Tokenizer>> {
    let json = match repo_id {
        "distil-whisper/distil-medium.en" => DISTIL_MEDIUM_EN_TOKENIZER,
        "distil-whisper/distil-large-v3" => DISTIL_LARGE_V3_TOKENIZER,
        "distil-whisper/distil-large-v3.5" => DISTIL_LARGE_V3_5_TOKENIZER,
        _ => return None,
    };
    Some(
        tokenizers::Tokenizer::from_bytes(json.as_bytes())
            .map_err(|e| SttError::Tokenizer(format!("embedded tokenizer: {e}"))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default model must ship embedded, parseable metadata so the daemon
    /// starts without a config/tokenizer fetch. Guards future default changes.
    #[test]
    fn default_model_metadata_is_embedded_and_parses() {
        config_for_repo(DEFAULT_MODEL_REPO)
            .unwrap_or_else(|| panic!("no embedded config for default {DEFAULT_MODEL_REPO}"))
            .expect("default config parses");
        tokenizer_for_repo(DEFAULT_MODEL_REPO)
            .unwrap_or_else(|| panic!("no embedded tokenizer for default {DEFAULT_MODEL_REPO}"))
            .expect("default tokenizer parses");
    }
}
