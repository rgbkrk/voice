use mlx_rs::error::Exception;

#[derive(Debug, thiserror::Error)]
pub enum VoicersError {
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Weight error: {0}")]
    Weight(String),

    #[error("Hub error: {0}")]
    Hub(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, VoicersError>;
