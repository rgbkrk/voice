//! Embedded model config and tokenizer data for instant startup.
//!
//! The Moonshine-tiny config and tokenizer are compiled into the binary via
//! `include_str!`, enabling instant model initialization with no network or
//! filesystem access for these small files. Model weights (~108MB) are still
//! downloaded from HuggingFace Hub on first run.

use crate::error::{Result, SttError};
use crate::moonshine::MoonshineConfig;

// Embedded data
const MOONSHINE_TINY_CONFIG_JSON: &str = include_str!("../data/moonshine_tiny_config.json");
const MOONSHINE_TINY_TOKENIZER_JSON: &str = include_str!("../data/moonshine_tiny_tokenizer.json");

/// Default HuggingFace repo ID for Moonshine-tiny.
pub const MOONSHINE_TINY_REPO: &str = "UsefulSensors/moonshine-tiny";

/// Default HuggingFace repo ID for Moonshine-base.
pub const MOONSHINE_BASE_REPO: &str = "UsefulSensors/moonshine-base";

/// All known model repo IDs with embedded configs.
pub const BUILTIN_MODELS: &[&str] = &[MOONSHINE_TINY_REPO];

/// Parse the embedded Moonshine-tiny model config.
pub fn builtin_config() -> Result<MoonshineConfig> {
    let config: MoonshineConfig =
        serde_json::from_str(MOONSHINE_TINY_CONFIG_JSON).map_err(SttError::Json)?;
    Ok(config)
}

/// Return the raw JSON string for the embedded Moonshine-tiny tokenizer.
///
/// This can be passed to `tokenizers::Tokenizer::from_bytes()` to create
/// a tokenizer without any filesystem or network access.
pub fn builtin_tokenizer_json() -> &'static str {
    MOONSHINE_TINY_TOKENIZER_JSON
}

/// Load the embedded Moonshine-tiny tokenizer.
pub fn builtin_tokenizer() -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_bytes(MOONSHINE_TINY_TOKENIZER_JSON.as_bytes())
        .map_err(|e| SttError::Tokenizer(format!("Failed to load embedded tokenizer: {e}")))
}

/// Check if a repo ID has an embedded config available.
pub fn has_builtin_config(repo_id: &str) -> bool {
    BUILTIN_MODELS.contains(&repo_id)
}

/// Get the embedded config for a repo ID, if available.
pub fn config_for_repo(repo_id: &str) -> Option<Result<MoonshineConfig>> {
    match repo_id {
        MOONSHINE_TINY_REPO => Some(builtin_config()),
        _ => None,
    }
}

/// Get the embedded tokenizer for a repo ID, if available.
pub fn tokenizer_for_repo(repo_id: &str) -> Option<Result<tokenizers::Tokenizer>> {
    match repo_id {
        MOONSHINE_TINY_REPO => Some(builtin_tokenizer()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_config_parses() {
        let config = builtin_config().unwrap();
        assert_eq!(config.model_type, "moonshine");
        assert_eq!(config.hidden_size, 288);
        assert_eq!(config.vocab_size, 32768);
        assert_eq!(config.encoder_num_hidden_layers, 6);
        assert_eq!(config.decoder_num_hidden_layers, 6);
    }

    #[test]
    fn test_builtin_tokenizer_loads() {
        let tokenizer = builtin_tokenizer().unwrap();
        let vocab_size = tokenizer.get_vocab_size(true);
        assert_eq!(vocab_size, 32768);
    }

    #[test]
    fn test_builtin_tokenizer_decodes() {
        let tokenizer = builtin_tokenizer().unwrap();
        // Token 1 = BOS (<s>), Token 2 = EOS (</s>)
        // Try decoding some basic tokens
        let result = tokenizer.decode(&[1], true);
        assert!(result.is_ok(), "Should decode BOS token");
    }

    #[test]
    fn test_has_builtin() {
        assert!(has_builtin_config("UsefulSensors/moonshine-tiny"));
        assert!(!has_builtin_config("UsefulSensors/moonshine-base"));
        assert!(!has_builtin_config("openai/whisper-tiny"));
    }

    #[test]
    fn test_config_for_repo() {
        let config = config_for_repo("UsefulSensors/moonshine-tiny");
        assert!(config.is_some());
        assert!(config.unwrap().is_ok());

        let config = config_for_repo("unknown/model");
        assert!(config.is_none());
    }
}
