//! Whisper speech-to-text on candle with Metal acceleration.
//!
//! Wraps `candle_transformers::models::whisper` with a greedy decode pipeline,
//! mel filter loading, and KV-cached inference. Supports all Whisper variants
//! including distil-whisper.

mod decoder;
mod mel;

pub use candle_transformers::models::whisper::Config;
pub use decoder::{DecodingResult, WhisperDecoder};
pub use mel::{load_mel_filters, pcm_to_mel, GpuMelSpec};

/// Re-export the model types from candle-transformers for loading.
pub use candle_transformers::models::whisper::model::Whisper;
pub use candle_transformers::models::whisper::DTYPE;
