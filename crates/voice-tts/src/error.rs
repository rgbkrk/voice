#[derive(Debug, thiserror::Error)]
pub enum VoicersError {
    #[error("Model error: {0}")]
    Model(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Hub error: {0}")]
    Hub(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, VoicersError>;
