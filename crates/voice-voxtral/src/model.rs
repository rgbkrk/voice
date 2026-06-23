use crate::{Result, VoxtralConfig, VoxtralError};

/// Boundary for a future native Candle implementation of Voxtral TTS.
#[derive(Debug, Clone)]
pub struct VoxtralModel {
    config: VoxtralConfig,
}

impl VoxtralModel {
    pub fn new(config: VoxtralConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &VoxtralConfig {
        &self.config
    }

    pub fn generate(&mut self, _text: &str, _voice: &str) -> Result<Vec<f32>> {
        Err(VoxtralError::Unsupported(
            "the config/asset layer exists, but the native audio-generation and audio-tokenizer stages have not been ported".to_string(),
        ))
    }
}
