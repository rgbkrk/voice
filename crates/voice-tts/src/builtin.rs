//! Embedded model config and voice data.

use voice_kokoro::ModelConfig;
use crate::error::Result;

const BUILTIN_CONFIG_JSON: &str = include_str!("../data/config.json");

const VOICE_AF_HEART: &[u8] = include_bytes!("../data/voices/af_heart.safetensors");
const VOICE_AF_BELLA: &[u8] = include_bytes!("../data/voices/af_bella.safetensors");
const VOICE_AF_SARAH: &[u8] = include_bytes!("../data/voices/af_sarah.safetensors");
const VOICE_AF_SKY: &[u8] = include_bytes!("../data/voices/af_sky.safetensors");
const VOICE_AM_MICHAEL: &[u8] = include_bytes!("../data/voices/am_michael.safetensors");
const VOICE_AM_ADAM: &[u8] = include_bytes!("../data/voices/am_adam.safetensors");
const VOICE_BF_EMMA: &[u8] = include_bytes!("../data/voices/bf_emma.safetensors");

pub const BUILTIN_VOICES: &[&str] = &[
    "af_heart",
    "af_bella",
    "af_sarah",
    "af_sky",
    "am_michael",
    "am_adam",
    "bf_emma",
];

pub fn builtin_config() -> Result<ModelConfig> {
    let config: ModelConfig = serde_json::from_str(BUILTIN_CONFIG_JSON)?;
    Ok(config)
}

pub fn get_builtin_voice_bytes(name: &str) -> Option<&'static [u8]> {
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
