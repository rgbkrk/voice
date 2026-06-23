#[derive(Debug, thiserror::Error)]
pub enum VoxtralError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Hub error: {0}")]
    Hub(String),

    #[error("Candle error: {0}")]
    Candle(String),

    #[error("Invalid Voxtral config: {0}")]
    InvalidConfig(String),

    #[error("Invalid Voxtral checkpoint: {0}")]
    InvalidCheckpoint(String),

    #[error("Invalid Voxtral tokenizer: {0}")]
    InvalidTokenizer(String),

    #[error("Native Voxtral inference is not implemented yet: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, VoxtralError>;
