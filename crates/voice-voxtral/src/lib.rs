//! Voxtral TTS support primitives.
//!
//! This crate starts the native Voxtral integration with configuration,
//! voice metadata, asset discovery, checkpoint validation, typed Candle module
//! loading, and the first executable acoustic-transformer forward helpers.
//! End-to-end text-to-audio generation is still follow-up work.

mod assets;
mod codec;
mod config;
mod error;
mod generation;
mod model;
mod prompt;
mod realtime;
mod realtime_audio;
mod realtime_inference;
mod streaming;
mod text;
mod tokenizer;
mod transformer;
mod voice_embedding;
mod voices;
mod weights;

pub use assets::{
    VoxtralAssetPaths, VoxtralAssetResolver, VoxtralSource, CONFIG_FILE, DEFAULT_REPO,
    TOKENIZER_FILE, VOICE_EMBEDDING_DIR, WEIGHTS_FILE,
};
pub use codec::{
    VoxtralAudioCodebook, VoxtralAudioTokenizer, VoxtralCodecAttention, VoxtralCodecStage,
    VoxtralCodecTransformerBlock,
};
pub use config::{
    AcousticTransformerConfig, AudioEncodingConfig, AudioModelConfig, AudioTokenizerConfig,
    MultimodalConfig, VoxtralConfig,
};
pub use error::{Result, VoxtralError};
pub use generation::{
    VoxtralGeneratedAudio, VoxtralGeneratedAudioChunk, VoxtralGenerationOptions,
    VoxtralGenerationTrace, VoxtralRuntimeLoadTrace, VoxtralTtsRuntime,
};
pub use model::{VoxtralModel, VoxtralModelLoadTrace};
pub use prompt::{
    build_prompt_embeddings, build_prompt_token_ids, VoxtralPrompt,
    VOXTRAL_NEXT_AUDIO_TEXT_TOKEN_ID, VOXTRAL_REPEAT_AUDIO_TEXT_TOKEN_ID,
};
pub use realtime::{
    build_realtime_streaming_prompt, build_realtime_streaming_prompt_with_left_pad,
    expected_realtime_tensors, pad_realtime_audio, plan_realtime_audio_padding,
    plan_realtime_audio_padding_with_left_pad, realtime_audio_frames_per_token,
    realtime_num_audio_tokens_for_samples, realtime_num_delay_tokens,
    realtime_raw_audio_length_per_token, validate_realtime_checkpoint, VoxtralRealtimeAssetPaths,
    VoxtralRealtimeAssetResolver, VoxtralRealtimeAudioEncoderConfig,
    VoxtralRealtimeAudioEncodingConfig, VoxtralRealtimeConfig, VoxtralRealtimeDownsampleConfig,
    VoxtralRealtimeExpectedTensor, VoxtralRealtimeModel, VoxtralRealtimeMultimodalConfig,
    VoxtralRealtimePaddingPlan, VoxtralRealtimePrompt, VoxtralRealtimeTransformersAudioConfig,
    VoxtralRealtimeTransformersConfig, VoxtralRealtimeTransformersTextConfig,
    VoxtralRealtimeWhisperModelConfig, REALTIME_AUDIO_TOKEN_ID, REALTIME_BEGIN_AUDIO_TOKEN_ID,
    REALTIME_BOS_TOKEN_ID, REALTIME_CONFIG_FILE, REALTIME_DEFAULT_LEFT_PAD_TOKENS,
    REALTIME_DEFAULT_OFFLINE_BUFFER_TOKENS, REALTIME_DEFAULT_REPO, REALTIME_EOS_TOKEN_ID,
    REALTIME_EXPECTED_TENSOR_COUNT, REALTIME_HF_CONFIG_FILE, REALTIME_NUM_MEL_BINS,
    REALTIME_PROCESSOR_CONFIG_FILE, REALTIME_REPEAT_AUDIO_TEXT_TOKEN_ID, REALTIME_SAMPLE_RATE,
    REALTIME_STREAMING_PAD_TOKEN_ID, REALTIME_STREAMING_WORD_TOKEN_ID, REALTIME_TOKENIZER_FILE,
    REALTIME_TRANSCRIPTION_FORMAT, REALTIME_WEIGHTS_FILE,
};
pub use realtime_audio::{
    realtime_log_mel_spectrogram, realtime_log_mel_spectrogram_with_center, realtime_mel_filters,
    VoxtralRealtimeMelFilters, VoxtralRealtimeMelSpectrogram, REALTIME_MIN_MEL_VALUE,
};
pub use realtime_inference::{
    VoxtralRealtimeAudioAttention, VoxtralRealtimeAudioFeedForward, VoxtralRealtimeAudioModules,
    VoxtralRealtimeAudioProjector, VoxtralRealtimeAudioStem, VoxtralRealtimeAudioTransformer,
    VoxtralRealtimeAudioTransformerBlock, VoxtralRealtimeInferenceModules,
    VoxtralRealtimeTextAdaRmsNorm, VoxtralRealtimeTextAttention, VoxtralRealtimeTextDecoder,
    VoxtralRealtimeTextDecoderBlock, VoxtralRealtimeTextFeedForward, VoxtralRealtimeTextGeneration,
    VoxtralRealtimeTokenEmbeddings, VoxtralRealtimeTranscriber, VoxtralRealtimeTranscription,
    VoxtralRealtimeTranscriptionOptions,
};
pub use streaming::{
    plan_codec_chunk, VoxtralCodecChunk, VoxtralStreamingConfig, DEFAULT_CODEC_CHUNK_FRAMES,
    DEFAULT_CODEC_CHUNK_FRAMES_AT_BEGIN, DEFAULT_CODEC_LEFT_CONTEXT_FRAMES,
};
pub use text::{
    normalize_tts_text, normalize_tts_text_with_options, suggest_max_frames_for_text,
    VoxtralTextNormalizationOptions, DEFAULT_SUGGESTED_MAX_FRAMES, SUGGESTED_MAX_FRAMES_CAP,
};
pub use tokenizer::{
    TekkenAudioEncodingConfig, TekkenAudioMetadata, TekkenConfig, TekkenSpecialToken,
    TekkenVocabToken, VoxtralTekkenDecoder, VoxtralTekkenEncoder, VoxtralTokenizerMetadata,
};
pub use transformer::{
    VoxtralAcousticTransformer, VoxtralAttention, VoxtralFeedForward, VoxtralInferenceModules,
    VoxtralLanguageBackbone, VoxtralLanguageCache, VoxtralModuleLoadTrace,
    VoxtralMultimodalEmbeddings, VoxtralTransformerBlock,
};
pub use voice_embedding::{
    load_voice_embedding, load_voice_embedding_with_hidden_dim, VOXTRAL_VOICE_EMBEDDING_HIDDEN_DIM,
};
pub use voices::{get_preset_voice, PresetVoice, VOXTRAL_PRESET_VOICES};
pub use weights::{
    ExpectedTensor, TensorInfo, VoxtralCheckpointSummary, VoxtralWeightMetadata, WeightComponent,
};

/// Voxtral TTS emits 24 kHz mono audio.
pub const SAMPLE_RATE: u32 = 24_000;
