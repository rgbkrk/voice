#[derive(Debug, thiserror::Error)]
pub enum VoxtralError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Hub error: {0}")]
    Hub(String),

    #[error("Invalid Voxtral config: {0}")]
    InvalidConfig(String),

    #[error("Native Voxtral inference is not implemented yet: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, VoxtralError>;
