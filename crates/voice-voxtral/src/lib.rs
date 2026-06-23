//! Voxtral TTS support primitives.
//!
//! This crate starts the native Voxtral integration with configuration,
//! voice metadata, and asset discovery. It deliberately does not claim a
//! working forward pass yet; `VoxtralModel` marks the boundary for the future
//! Candle implementation.

mod assets;
mod config;
mod error;
mod model;
mod tokenizer;
mod voices;
mod weights;

pub use assets::{
    VoxtralAssetPaths, VoxtralAssetResolver, VoxtralSource, CONFIG_FILE, DEFAULT_REPO,
    TOKENIZER_FILE, VOICE_EMBEDDING_DIR, WEIGHTS_FILE,
};
pub use config::{
    AcousticTransformerConfig, AudioEncodingConfig, AudioModelConfig, AudioTokenizerConfig,
    MultimodalConfig, VoxtralConfig,
};
pub use error::{Result, VoxtralError};
pub use model::VoxtralModel;
pub use tokenizer::{
    TekkenAudioEncodingConfig, TekkenAudioMetadata, TekkenConfig, TekkenSpecialToken,
    TekkenVocabToken, VoxtralTokenizerMetadata,
};
pub use voices::{get_preset_voice, PresetVoice, VOXTRAL_PRESET_VOICES};
pub use weights::{
    ExpectedTensor, TensorInfo, VoxtralCheckpointSummary, VoxtralWeightMetadata, WeightComponent,
};

/// Voxtral TTS emits 24 kHz mono audio.
pub const SAMPLE_RATE: u32 = 24_000;
