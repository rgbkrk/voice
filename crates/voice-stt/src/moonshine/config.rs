//! Moonshine model configuration.
//!
//! Matches the `config.json` structure from HuggingFace's
//! `UsefulSensors/moonshine-tiny` and `UsefulSensors/moonshine-base`.

use serde::Deserialize;

/// Configuration for a Moonshine speech-to-text model.
///
/// Two official variants exist:
///
/// | Field | Tiny | Base |
/// |-------|------|------|
/// | `hidden_size` | 288 | 416 |
/// | `intermediate_size` | 1152 | 1664 |
/// | `encoder_num_hidden_layers` | 6 | 8 |
/// | `decoder_num_hidden_layers` | 6 | 8 |
/// | `partial_rotary_factor` | 0.9 | 0.62 |
#[derive(Debug, Clone, Deserialize)]
pub struct MoonshineConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    #[serde(default = "default_encoder_num_hidden_layers")]
    pub encoder_num_hidden_layers: usize,

    #[serde(default = "default_decoder_num_hidden_layers")]
    pub decoder_num_hidden_layers: usize,

    #[serde(default = "default_num_attention_heads")]
    pub encoder_num_attention_heads: usize,

    #[serde(default = "default_num_attention_heads")]
    pub decoder_num_attention_heads: usize,

    /// Number of key-value heads for encoder attention (GQA).
    /// Defaults to `encoder_num_attention_heads` if not set.
    pub encoder_num_key_value_heads: Option<usize>,

    /// Number of key-value heads for decoder attention (GQA).
    /// Defaults to `decoder_num_attention_heads` if not set.
    pub decoder_num_key_value_heads: Option<usize>,

    #[serde(default = "default_encoder_hidden_act")]
    pub encoder_hidden_act: String,

    #[serde(default = "default_decoder_hidden_act")]
    pub decoder_hidden_act: String,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default)]
    pub attention_bias: bool,

    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,

    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,

    #[serde(default = "default_decoder_start_token_id")]
    pub decoder_start_token_id: u32,

    /// EOS is used as the pad token.
    pub pad_token_id: Option<u32>,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    /// Pad head dimension to a multiple of this value for hardware efficiency.
    pub pad_head_dim_to_multiple_of: Option<usize>,
}

impl MoonshineConfig {
    /// Effective number of KV heads for the encoder.
    /// Falls back to `encoder_num_attention_heads` when not explicitly set.
    pub fn encoder_kv_heads(&self) -> usize {
        self.encoder_num_key_value_heads
            .unwrap_or(self.encoder_num_attention_heads)
    }

    /// Effective number of KV heads for the decoder.
    /// Falls back to `decoder_num_attention_heads` when not explicitly set.
    pub fn decoder_kv_heads(&self) -> usize {
        self.decoder_num_key_value_heads
            .unwrap_or(self.decoder_num_attention_heads)
    }

    /// Per-head dimension: `hidden_size / num_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.encoder_num_attention_heads
    }

    /// Number of dimensions that receive rotary positional encoding.
    ///
    /// Computed as `floor(head_dim * partial_rotary_factor)` rounded down
    /// to the nearest even number.
    pub fn rotary_ndims(&self) -> usize {
        let raw = (self.head_dim() as f32 * self.partial_rotary_factor) as usize;
        raw - (raw % 2)
    }

    /// Number of KV groups for GQA in the encoder.
    /// Returns 1 for standard multi-head attention.
    pub fn encoder_kv_groups(&self) -> usize {
        self.encoder_num_attention_heads / self.encoder_kv_heads()
    }

    /// Number of KV groups for GQA in the decoder.
    /// Returns 1 for standard multi-head attention.
    pub fn decoder_kv_groups(&self) -> usize {
        self.decoder_num_attention_heads / self.decoder_kv_heads()
    }

    /// Audio sample rate expected by the model.
    pub fn sample_rate(&self) -> u32 {
        16000
    }
}

// -- Serde defaults ----------------------------------------------------------

fn default_model_type() -> String {
    "moonshine".to_string()
}

fn default_vocab_size() -> usize {
    32768
}

fn default_hidden_size() -> usize {
    288
}

fn default_intermediate_size() -> usize {
    1152
}

fn default_encoder_num_hidden_layers() -> usize {
    6
}

fn default_decoder_num_hidden_layers() -> usize {
    6
}

fn default_num_attention_heads() -> usize {
    8
}

fn default_encoder_hidden_act() -> String {
    "gelu".to_string()
}

fn default_decoder_hidden_act() -> String {
    "silu".to_string()
}

fn default_max_position_embeddings() -> usize {
    512
}

fn default_partial_rotary_factor() -> f32 {
    0.9
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_bos_token_id() -> u32 {
    1
}

fn default_eos_token_id() -> u32 {
    2
}

fn default_decoder_start_token_id() -> u32 {
    1
}

fn default_tie_word_embeddings() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_config() {
        let json = r#"{
            "model_type": "moonshine",
            "hidden_size": 288,
            "intermediate_size": 1152,
            "encoder_num_hidden_layers": 6,
            "decoder_num_hidden_layers": 6,
            "encoder_num_attention_heads": 8,
            "decoder_num_attention_heads": 8,
            "partial_rotary_factor": 0.9,
            "max_position_embeddings": 194,
            "vocab_size": 32768,
            "tie_word_embeddings": true
        }"#;
        let config: MoonshineConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.hidden_size, 288);
        assert_eq!(config.head_dim(), 36);
        assert_eq!(config.rotary_ndims(), 32);
        assert_eq!(config.encoder_kv_heads(), 8);
        assert_eq!(config.encoder_kv_groups(), 1);
        assert_eq!(config.sample_rate(), 16000);
    }

    #[test]
    fn test_base_config() {
        let json = r#"{
            "model_type": "moonshine",
            "hidden_size": 416,
            "intermediate_size": 1664,
            "encoder_num_hidden_layers": 8,
            "decoder_num_hidden_layers": 8,
            "encoder_num_attention_heads": 8,
            "decoder_num_attention_heads": 8,
            "partial_rotary_factor": 0.62,
            "max_position_embeddings": 194,
            "vocab_size": 32768
        }"#;
        let config: MoonshineConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.hidden_size, 416);
        assert_eq!(config.head_dim(), 52);
        // floor(52 * 0.62) = 32, already even
        assert_eq!(config.rotary_ndims(), 32);
        assert_eq!(config.encoder_num_hidden_layers, 8);
    }

    #[test]
    fn test_defaults() {
        let config: MoonshineConfig = serde_json::from_str("{}").unwrap();

        assert_eq!(config.model_type, "moonshine");
        assert_eq!(config.vocab_size, 32768);
        assert_eq!(config.hidden_size, 288);
        assert!(!config.attention_bias);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.bos_token_id, 1);
        assert_eq!(config.eos_token_id, 2);
        assert_eq!(config.encoder_kv_heads(), 8);
        assert_eq!(config.decoder_kv_heads(), 8);
    }

    #[test]
    fn test_gqa_config() {
        let json = r#"{
            "encoder_num_attention_heads": 8,
            "decoder_num_attention_heads": 8,
            "encoder_num_key_value_heads": 4,
            "decoder_num_key_value_heads": 2
        }"#;
        let config: MoonshineConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.encoder_kv_heads(), 4);
        assert_eq!(config.encoder_kv_groups(), 2);
        assert_eq!(config.decoder_kv_heads(), 2);
        assert_eq!(config.decoder_kv_groups(), 4);
    }
}
