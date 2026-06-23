use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::{Result, VoxtralError, SAMPLE_RATE};

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub norm_eps: f64,
    pub max_seq_len: usize,
    pub model_type: String,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    pub multimodal: MultimodalConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MultimodalConfig {
    pub bos_token_id: i64,
    pub audio_model_args: AudioModelConfig,
    pub audio_tokenizer_args: AudioTokenizerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioModelConfig {
    pub semantic_codebook_size: usize,
    pub acoustic_codebook_size: usize,
    pub n_acoustic_codebook: usize,
    pub audio_encoding_args: AudioEncodingConfig,
    pub audio_token_id: i64,
    pub begin_audio_token_id: i64,
    pub input_embedding_concat_type: String,
    pub acoustic_transformer_args: AcousticTransformerConfig,
    #[serde(default)]
    pub p_uncond: Option<f64>,
    #[serde(default)]
    pub text_feature_bugged: Option<bool>,
    #[serde(default)]
    pub condition_dropped_token_id: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncodingConfig {
    pub codebook_pattern: String,
    pub interleave_audio_tokens_per_segment: usize,
    pub interleave_text_tokens_per_segment: usize,
    pub single_trailing_segment: bool,
    pub num_codebooks: usize,
    pub sampling_rate: u32,
    pub frame_rate: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AcousticTransformerConfig {
    pub input_dim: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub use_biases: bool,
    pub rope_theta: f64,
    #[serde(default)]
    pub sigma: Option<f64>,
    #[serde(default)]
    pub sigma_max: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioTokenizerConfig {
    pub channels: usize,
    pub sampling_rate: u32,
    pub pretransform_patch_size: usize,
    pub patch_proj_kernel_size: usize,
    pub semantic_codebook_size: usize,
    pub semantic_dim: usize,
    pub acoustic_codebook_size: usize,
    pub acoustic_dim: usize,
    pub conv_weight_norm: bool,
    pub causal: bool,
    pub attn_sliding_window_size: usize,
    pub half_attn_window_upon_downsampling: bool,
    pub dim: usize,
    pub hidden_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub qk_norm_eps: f64,
    pub qk_norm: bool,
    pub use_biases: bool,
    pub norm_eps: f64,
    pub layer_scale: bool,
    pub layer_scale_init: f64,
    pub decoder_transformer_lengths_str: String,
    pub decoder_convs_kernels_str: String,
    pub decoder_convs_strides_str: String,
    pub voice: BTreeMap<String, usize>,
}

impl VoxtralConfig {
    pub fn from_json_str(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        Self::from_json_str(&json)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "voxtral_tts" {
            return Err(VoxtralError::InvalidConfig(format!(
                "expected model_type voxtral_tts, got {}",
                self.model_type
            )));
        }

        let encoding = &self.multimodal.audio_model_args.audio_encoding_args;
        let tokenizer = &self.multimodal.audio_tokenizer_args;
        let audio_model = &self.multimodal.audio_model_args;
        let expected_codebooks = self.multimodal.audio_model_args.n_acoustic_codebook + 1;

        if encoding.sampling_rate != SAMPLE_RATE || tokenizer.sampling_rate != SAMPLE_RATE {
            return Err(VoxtralError::InvalidConfig(format!(
                "expected 24 kHz audio, got encoding={} tokenizer={}",
                encoding.sampling_rate, tokenizer.sampling_rate
            )));
        }

        if encoding.num_codebooks != expected_codebooks {
            return Err(VoxtralError::InvalidConfig(format!(
                "num_codebooks={} but semantic + acoustic codebooks imply {}",
                encoding.num_codebooks, expected_codebooks
            )));
        }

        if tokenizer.semantic_codebook_size != audio_model.semantic_codebook_size
            || tokenizer.acoustic_codebook_size != audio_model.acoustic_codebook_size
            || tokenizer.acoustic_dim != audio_model.n_acoustic_codebook
        {
            return Err(VoxtralError::InvalidConfig(
                "audio model and tokenizer codebook dimensions disagree".to_string(),
            ));
        }

        if tokenizer.voice.is_empty() {
            return Err(VoxtralError::InvalidConfig(
                "no preset voices listed in audio_tokenizer_args.voice".to_string(),
            ));
        }

        Ok(())
    }

    pub fn sample_rate(&self) -> u32 {
        self.multimodal
            .audio_model_args
            .audio_encoding_args
            .sampling_rate
    }

    pub fn frame_rate(&self) -> f64 {
        self.multimodal
            .audio_model_args
            .audio_encoding_args
            .frame_rate
    }

    pub fn num_codebooks(&self) -> usize {
        self.multimodal
            .audio_model_args
            .audio_encoding_args
            .num_codebooks
    }

    pub fn voice_id(&self, name: &str) -> Option<usize> {
        self.multimodal
            .audio_tokenizer_args
            .voice
            .get(name)
            .copied()
    }

    pub fn voices(&self) -> impl Iterator<Item = (&str, usize)> {
        self.multimodal
            .audio_tokenizer_args
            .voice
            .iter()
            .map(|(name, id)| (name.as_str(), *id))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::VOXTRAL_PRESET_VOICES;

    pub(crate) const PARAMS_JSON: &str = r#"{
      "dim": 3072,
      "n_layers": 26,
      "head_dim": 128,
      "hidden_dim": 9216,
      "n_heads": 32,
      "n_kv_heads": 8,
      "rope_theta": 1000000.0,
      "norm_eps": 1e-05,
      "vocab_size": 131072,
      "max_seq_len": 65536,
      "model_type": "voxtral_tts",
      "max_position_embeddings": 128000,
      "multimodal": {
        "bos_token_id": 1,
        "audio_model_args": {
          "semantic_codebook_size": 8192,
          "acoustic_codebook_size": 21,
          "n_acoustic_codebook": 36,
          "audio_encoding_args": {
            "codebook_pattern": "parallel",
            "interleave_audio_tokens_per_segment": 8192,
            "interleave_text_tokens_per_segment": 8192,
            "single_trailing_segment": false,
            "num_codebooks": 37,
            "sampling_rate": 24000,
            "frame_rate": 12.5
          },
          "audio_token_id": 24,
          "begin_audio_token_id": 25,
          "input_embedding_concat_type": "sum",
          "acoustic_transformer_args": {
            "input_dim": 3072,
            "dim": 3072,
            "n_layers": 3,
            "head_dim": 128,
            "hidden_dim": 9216,
            "n_heads": 32,
            "n_kv_heads": 8,
            "use_biases": false,
            "rope_theta": 10000.0,
            "sigma": 1e-05,
            "sigma_max": 1.0
          },
          "p_uncond": 0.0,
          "text_feature_bugged": false,
          "condition_dropped_token_id": 42
        },
        "audio_tokenizer_args": {
          "channels": 1,
          "sampling_rate": 24000,
          "pretransform_patch_size": 240,
          "patch_proj_kernel_size": 7,
          "semantic_codebook_size": 8192,
          "semantic_dim": 256,
          "acoustic_codebook_size": 21,
          "acoustic_dim": 36,
          "conv_weight_norm": true,
          "causal": true,
          "attn_sliding_window_size": 16,
          "half_attn_window_upon_downsampling": true,
          "dim": 1024,
          "hidden_dim": 4096,
          "head_dim": 128,
          "n_heads": 8,
          "n_kv_heads": 8,
          "qk_norm_eps": 1e-06,
          "qk_norm": true,
          "use_biases": false,
          "norm_eps": 0.01,
          "layer_scale": true,
          "layer_scale_init": 0.01,
          "decoder_transformer_lengths_str": "2,2,2,2",
          "decoder_convs_kernels_str": "3,4,4,4",
          "decoder_convs_strides_str": "1,2,2,2",
          "voice": {
            "casual_female": 0,
            "casual_male": 1,
            "cheerful_female": 2,
            "neutral_female": 3,
            "neutral_male": 4,
            "pt_male": 5,
            "pt_female": 6,
            "nl_male": 7,
            "nl_female": 8,
            "it_male": 9,
            "it_female": 10,
            "fr_male": 11,
            "fr_female": 12,
            "es_male": 13,
            "es_female": 14,
            "de_male": 15,
            "de_female": 16,
            "ar_male": 17,
            "hi_male": 18,
            "hi_female": 19
          }
        }
      }
    }"#;

    #[test]
    fn parses_official_config_shape() {
        let config = VoxtralConfig::from_json_str(PARAMS_JSON).unwrap();

        assert_eq!(config.model_type, "voxtral_tts");
        assert_eq!(config.sample_rate(), 24_000);
        assert_eq!(config.frame_rate(), 12.5);
        assert_eq!(config.num_codebooks(), 37);
        assert_eq!(config.voice_id("casual_male"), Some(1));
        assert_eq!(config.voice_id("hi_female"), Some(19));
        assert_eq!(config.voices().count(), 20);
    }

    #[test]
    fn preset_catalog_matches_config_fixture() {
        let config = VoxtralConfig::from_json_str(PARAMS_JSON).unwrap();
        let config_voices = config
            .voices()
            .map(|(name, _)| name)
            .collect::<BTreeSet<_>>();
        let catalog_voices = VOXTRAL_PRESET_VOICES
            .iter()
            .map(|voice| voice.id)
            .collect::<BTreeSet<_>>();

        assert_eq!(catalog_voices, config_voices);
    }

    #[test]
    fn rejects_non_voxtral_model_type() {
        let json = PARAMS_JSON.replace("\"voxtral_tts\"", "\"kokoro\"");
        let err = VoxtralConfig::from_json_str(&json).unwrap_err();

        assert!(matches!(err, VoxtralError::InvalidConfig(_)));
    }
}
