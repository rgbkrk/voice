//! Embedded model config and tokenizer for instant startup.
//!
//! Both distil-medium.en and distil-large-v3 configs and tokenizers are
//! compiled into the binary via `include_str!`, enabling instant model
//! initialization with no network or filesystem access for metadata.
//! Only the model weights are downloaded from HuggingFace Hub on first run.

use crate::error::{Result, SttError};

/// Default HuggingFace repo ID for STT.
/// distil-large-v3: multilingual, best accuracy (~3GB weights).
/// Override with STT_MODEL=distil-whisper/distil-medium.en for faster/smaller.
pub const DEFAULT_MODEL_REPO: &str = "distil-whisper/distil-large-v3";

// Embedded data for zero-fetch init
const DISTIL_MEDIUM_EN_CONFIG: &str = include_str!("../data/distil_medium_en_config.json");
const DISTIL_MEDIUM_EN_TOKENIZER: &str = include_str!("../data/distil_medium_en_tokenizer.json");
const DISTIL_LARGE_V3_CONFIG: &str = include_str!("../data/distil_large_v3_config.json");
const DISTIL_LARGE_V3_TOKENIZER: &str = include_str!("../data/distil_large_v3_tokenizer.json");

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
        _ => return None,
    };
    Some(serde_json::from_str(json).map_err(SttError::Json))
}

/// Get the embedded tokenizer for a repo ID, if available.
pub fn tokenizer_for_repo(repo_id: &str) -> Option<Result<tokenizers::Tokenizer>> {
    let json = match repo_id {
        "distil-whisper/distil-medium.en" => DISTIL_MEDIUM_EN_TOKENIZER,
        "distil-whisper/distil-large-v3" => DISTIL_LARGE_V3_TOKENIZER,
        _ => return None,
    };
    Some(
        tokenizers::Tokenizer::from_bytes(json.as_bytes())
            .map_err(|e| SttError::Tokenizer(format!("embedded tokenizer: {e}"))),
    )
}
