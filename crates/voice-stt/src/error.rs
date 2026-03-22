//! Error types for voice-stt.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SttError {
    #[error("MLX error: {0}")]
    Mlx(#[from] quill_mlx::error::Exception),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Hub error: {0}")]
    Hub(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Audio error: {0}")]
    Audio(String),

    #[error("Weight error: {0}")]
    Weight(String),

    #[error("Model error: {0}")]
    Model(String),
}

pub type Result<T> = std::result::Result<T, SttError>;
