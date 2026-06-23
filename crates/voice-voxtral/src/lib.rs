//! Voxtral TTS support primitives.
//!
//! This crate starts the native Voxtral integration with configuration,
//! voice metadata, asset discovery, checkpoint validation, typed Candle module
//! loading, and the first executable acoustic-transformer forward helpers.
//! End-to-end text-to-audio generation is still follow-up work.

mod assets;
mod config;
mod error;
mod model;
mod streaming;
mod tokenizer;
mod transformer;
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
pub use streaming::{
    plan_codec_chunk, VoxtralCodecChunk, VoxtralStreamingConfig, DEFAULT_CODEC_CHUNK_FRAMES,
    DEFAULT_CODEC_CHUNK_FRAMES_AT_BEGIN, DEFAULT_CODEC_LEFT_CONTEXT_FRAMES,
};
pub use tokenizer::{
    TekkenAudioEncodingConfig, TekkenAudioMetadata, TekkenConfig, TekkenSpecialToken,
    TekkenVocabToken, VoxtralTokenizerMetadata,
};
pub use transformer::{
    VoxtralAcousticTransformer, VoxtralAttention, VoxtralFeedForward, VoxtralInferenceModules,
    VoxtralLanguageBackbone, VoxtralMultimodalEmbeddings, VoxtralTransformerBlock,
};
pub use voices::{get_preset_voice, PresetVoice, VOXTRAL_PRESET_VOICES};
pub use weights::{
    ExpectedTensor, TensorInfo, VoxtralCheckpointSummary, VoxtralWeightMetadata, WeightComponent,
};

/// Voxtral TTS emits 24 kHz mono audio.
pub const SAMPLE_RATE: u32 = 24_000;
