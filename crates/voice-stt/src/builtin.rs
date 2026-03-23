//! Embedded model config and tokenizer for instant startup.
//!
//! The distil-medium.en config (2KB) and tokenizer (2.4MB) are compiled into
//! the binary via `include_str!`, enabling instant model initialization with
//! no network or filesystem access for metadata. Only the model weights
//! (~789MB) are downloaded from HuggingFace Hub on first run.

use crate::error::{Result, SttError};

/// Default HuggingFace repo ID for STT.
/// distil-medium.en: English-only, fast, good accuracy.
pub const DEFAULT_MODEL_REPO: &str = "distil-whisper/distil-medium.en";

// Embedded data for zero-fetch init
const DISTIL_MEDIUM_EN_CONFIG: &str = include_str!("../data/distil_medium_en_config.json");
const DISTIL_MEDIUM_EN_TOKENIZER: &str = include_str!("../data/distil_medium_en_tokenizer.json");

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
    match repo_id {
        "distil-whisper/distil-medium.en" => {
            Some(serde_json::from_str(DISTIL_MEDIUM_EN_CONFIG).map_err(SttError::Json))
        }
        _ => None,
    }
}

/// Get the embedded tokenizer for a repo ID, if available.
pub fn tokenizer_for_repo(repo_id: &str) -> Option<Result<tokenizers::Tokenizer>> {
    match repo_id {
        "distil-whisper/distil-medium.en" => Some(
            tokenizers::Tokenizer::from_bytes(DISTIL_MEDIUM_EN_TOKENIZER.as_bytes())
                .map_err(|e| SttError::Tokenizer(format!("embedded tokenizer: {e}"))),
        ),
        _ => None,
    }
}
