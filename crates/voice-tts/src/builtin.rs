//! Embedded model config and voice data.
//!
//! Popular voices and the model config are compiled into the binary via
//! `include_str!`/`include_bytes!`, enabling instant loading with no
//! network or filesystem access (model weights still require HF Hub).

use quill_mlx::Array;

use crate::config::ModelConfig;
use crate::error::{Result, VoicersError};

// Embedded config
const BUILTIN_CONFIG_JSON: &str = include_str!("../data/config.json");

// Embedded voices
const VOICE_AF_HEART: &[u8] = include_bytes!("../data/voices/af_heart.safetensors");
const VOICE_AF_BELLA: &[u8] = include_bytes!("../data/voices/af_bella.safetensors");
const VOICE_AF_SARAH: &[u8] = include_bytes!("../data/voices/af_sarah.safetensors");
const VOICE_AF_SKY: &[u8] = include_bytes!("../data/voices/af_sky.safetensors");
const VOICE_AM_MICHAEL: &[u8] = include_bytes!("../data/voices/am_michael.safetensors");
const VOICE_AM_ADAM: &[u8] = include_bytes!("../data/voices/am_adam.safetensors");
const VOICE_BF_EMMA: &[u8] = include_bytes!("../data/voices/bf_emma.safetensors");

/// All embedded voice names, for discoverability.
pub const BUILTIN_VOICES: &[&str] = &[
    "af_heart",
    "af_bella",
    "af_sarah",
    "af_sky",
    "am_michael",
    "am_adam",
    "bf_emma",
];

/// Parse the embedded model config.
pub fn builtin_config() -> Result<ModelConfig> {
    let config: ModelConfig = serde_json::from_str(BUILTIN_CONFIG_JSON)?;
    Ok(config)
}

/// Return the raw safetensors bytes for a builtin voice, or `None` if unknown.
pub fn builtin_voice_bytes(name: &str) -> Option<&'static [u8]> {
    match name {
        "af_heart" => Some(VOICE_AF_HEART),
        "af_bella" => Some(VOICE_AF_BELLA),
        "af_sarah" => Some(VOICE_AF_SARAH),
        "af_sky" => Some(VOICE_AF_SKY),
        "am_michael" => Some(VOICE_AM_MICHAEL),
        "am_adam" => Some(VOICE_AM_ADAM),
        "bf_emma" => Some(VOICE_BF_EMMA),
        _ => None,
    }
}

/// Load a builtin voice embedding as an [`Array`].
///
/// Returns `None` if `name` isn't a builtin voice. Returns `Some(Err(...))`
/// on deserialization or reshape failures.
///
/// Internally writes to a temp file because `Array::load_safetensors` only
/// accepts a `Path`. The temp file is cleaned up after loading.
pub fn load_builtin_voice(name: &str) -> Option<Result<Array>> {
    let bytes = builtin_voice_bytes(name)?;
    Some(load_voice_from_bytes(name, bytes))
}

fn load_voice_from_bytes(name: &str, bytes: &[u8]) -> Result<Array> {
    let path = std::env::temp_dir().join(format!("voicers_{name}.safetensors"));

    std::fs::write(&path, bytes)?;

    let tensors = Array::load_safetensors(&path).map_err(|e| {
        let _ = std::fs::remove_file(&path);
        VoicersError::Weight(e.to_string())
    })?;

    let _ = std::fs::remove_file(&path);

    let voice = tensors
        .into_values()
        .next()
        .ok_or_else(|| VoicersError::Weight(format!("Empty builtin voice file: {name}")))?;

    // Ensure shape is (1, 256) — matches logic in voice.rs
    let voice = if voice.ndim() == 1 {
        voice.reshape(&[1, -1])?
    } else {
        voice
    };

    Ok(voice)
}
