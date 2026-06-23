use candle_core::Result;
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::{AcousticTransformerConfig, VoxtralConfig};

pub struct VoxtralInferenceModules {
    pub embeddings: VoxtralMultimodalEmbeddings,
    pub language: VoxtralLanguageBackbone,
    pub acoustic: VoxtralAcousticTransformer,
}

pub struct VoxtralMultimodalEmbeddings {
    pub tok_embeddings: Embedding,
    pub audio_codebook_embeddings: Embedding,
    pub audio_vocab_size: usize,
}

pub struct VoxtralLanguageBackbone {
    pub layers: Vec<VoxtralTransformerBlock>,
    pub norm: RmsNorm,
}

pub struct VoxtralAcousticTransformer {
    pub input_projection: Linear,
    pub time_projection: Linear,
    pub llm_projection: Linear,
    pub semantic_codebook_output: Linear,
    pub acoustic_codebook_output: Linear,
    pub layers: Vec<VoxtralTransformerBlock>,
    pub norm: RmsNorm,
}

pub struct VoxtralTransformerBlock {
    pub attention: VoxtralAttention,
    pub feed_forward: VoxtralFeedForward,
    pub attention_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
}

pub struct VoxtralAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

pub struct VoxtralFeedForward {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
}

impl VoxtralInferenceModules {
    pub fn load(config: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = VoxtralMultimodalEmbeddings::load(config, vb.clone())?;
        let language = VoxtralLanguageBackbone::load(config, vb.clone())?;
        let acoustic = VoxtralAcousticTransformer::load(config, vb)?;

        Ok(Self {
            embeddings,
            language,
            acoustic,
        })
    }
}

impl VoxtralMultimodalEmbeddings {
    pub fn load(config: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        let audio_vocab_size = audio_codebook_vocab_size(config);
        let vb = vb.pp("mm_audio_embeddings");
        let tok_embeddings = embedding(config.vocab_size, config.dim, vb.pp("tok_embeddings"))?;
        let audio_codebook_embeddings = embedding(
            audio_vocab_size,
            config.dim,
            vb.pp("audio_codebook_embeddings").pp("embeddings"),
        )?;

        Ok(Self {
            tok_embeddings,
            audio_codebook_embeddings,
            audio_vocab_size,
        })
    }
}

impl VoxtralLanguageBackbone {
    pub fn load(config: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.n_layers);
        for layer_idx in 0..config.n_layers {
            layers.push(VoxtralTransformerBlock::load(
                config.dim,
                config.hidden_dim,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                config.norm_eps,
                vb.pp(format!("layers.{layer_idx}")),
            )?);
        }
        let norm = rms_norm(config.dim, config.norm_eps, vb.pp("norm"))?;

        Ok(Self { layers, norm })
    }
}

impl VoxtralAcousticTransformer {
    pub fn load(config: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        let audio_model = &config.multimodal.audio_model_args;
        let acoustic = &audio_model.acoustic_transformer_args;
        let vb = vb.pp("acoustic_transformer");

        let input_projection = linear_no_bias(
            audio_model.n_acoustic_codebook,
            acoustic.dim,
            vb.pp("input_projection"),
        )?;
        let time_projection = linear_no_bias(acoustic.dim, acoustic.dim, vb.pp("time_projection"))?;
        let llm_projection =
            linear_no_bias(acoustic.input_dim, acoustic.dim, vb.pp("llm_projection"))?;
        let semantic_codebook_output = linear_no_bias(
            acoustic.dim,
            semantic_codebook_output_size(config),
            vb.pp("semantic_codebook_output"),
        )?;
        let acoustic_codebook_output = linear_no_bias(
            acoustic.dim,
            audio_model.n_acoustic_codebook,
            vb.pp("acoustic_codebook_output"),
        )?;

        let mut layers = Vec::with_capacity(acoustic.n_layers);
        for layer_idx in 0..acoustic.n_layers {
            layers.push(VoxtralTransformerBlock::load_acoustic(
                acoustic,
                config.norm_eps,
                vb.pp(format!("layers.{layer_idx}")),
            )?);
        }
        let norm = rms_norm(acoustic.dim, config.norm_eps, vb.pp("norm"))?;

        Ok(Self {
            input_projection,
            time_projection,
            llm_projection,
            semantic_codebook_output,
            acoustic_codebook_output,
            layers,
            norm,
        })
    }
}

impl VoxtralTransformerBlock {
    pub fn load(
        dim: usize,
        hidden_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attention =
            VoxtralAttention::load(dim, n_heads, n_kv_heads, head_dim, vb.pp("attention"))?;
        let feed_forward = VoxtralFeedForward::load(dim, hidden_dim, vb.pp("feed_forward"))?;
        let attention_norm = rms_norm(dim, norm_eps, vb.pp("attention_norm"))?;
        let ffn_norm = rms_norm(dim, norm_eps, vb.pp("ffn_norm"))?;

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }

    fn load_acoustic(
        config: &AcousticTransformerConfig,
        norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::load(
            config.dim,
            config.hidden_dim,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            norm_eps,
            vb,
        )
    }
}

impl VoxtralAttention {
    pub fn load(
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let wq = linear_no_bias(dim, q_dim, vb.pp("wq"))?;
        let wk = linear_no_bias(dim, kv_dim, vb.pp("wk"))?;
        let wv = linear_no_bias(dim, kv_dim, vb.pp("wv"))?;
        let wo = linear_no_bias(q_dim, dim, vb.pp("wo"))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }
}

impl VoxtralFeedForward {
    pub fn load(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w1 = linear_no_bias(dim, hidden_dim, vb.pp("w1"))?;
        let w2 = linear_no_bias(hidden_dim, dim, vb.pp("w2"))?;
        let w3 = linear_no_bias(dim, hidden_dim, vb.pp("w3"))?;

        Ok(Self { w1, w2, w3 })
    }
}

fn audio_codebook_vocab_size(config: &VoxtralConfig) -> usize {
    let audio_model = &config.multimodal.audio_model_args;
    round_up_to_multiple(
        audio_model.semantic_codebook_size
            + 2
            + (audio_model.acoustic_codebook_size + 2) * audio_model.n_acoustic_codebook,
        128,
    )
}

fn semantic_codebook_output_size(config: &VoxtralConfig) -> usize {
    let audio_model = &config.multimodal.audio_model_args;
    round_up_to_multiple(audio_model.semantic_codebook_size + 2, 128)
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    multiple * value.div_ceil(multiple)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    use super::*;

    fn tiny_config() -> VoxtralConfig {
        VoxtralConfig::from_json_str(
            r#"{
              "dim": 8,
              "n_layers": 2,
              "head_dim": 4,
              "hidden_dim": 16,
              "n_heads": 2,
              "n_kv_heads": 1,
              "rope_theta": 1000000.0,
              "norm_eps": 1e-05,
              "vocab_size": 32,
              "max_seq_len": 128,
              "model_type": "voxtral_tts",
              "multimodal": {
                "bos_token_id": 1,
                "audio_model_args": {
                  "semantic_codebook_size": 8,
                  "acoustic_codebook_size": 3,
                  "n_acoustic_codebook": 2,
                  "audio_encoding_args": {
                    "codebook_pattern": "parallel",
                    "interleave_audio_tokens_per_segment": 8,
                    "interleave_text_tokens_per_segment": 8,
                    "single_trailing_segment": false,
                    "num_codebooks": 3,
                    "sampling_rate": 24000,
                    "frame_rate": 12.5
                  },
                  "audio_token_id": 24,
                  "begin_audio_token_id": 25,
                  "input_embedding_concat_type": "sum",
                  "acoustic_transformer_args": {
                    "input_dim": 8,
                    "dim": 8,
                    "n_layers": 1,
                    "head_dim": 4,
                    "hidden_dim": 16,
                    "n_heads": 2,
                    "n_kv_heads": 1,
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
                  "semantic_codebook_size": 8,
                  "semantic_dim": 4,
                  "acoustic_codebook_size": 3,
                  "acoustic_dim": 2,
                  "conv_weight_norm": true,
                  "causal": true,
                  "attn_sliding_window_size": 16,
                  "half_attn_window_upon_downsampling": true,
                  "dim": 8,
                  "hidden_dim": 16,
                  "head_dim": 4,
                  "n_heads": 2,
                  "n_kv_heads": 1,
                  "qk_norm_eps": 1e-06,
                  "qk_norm": true,
                  "use_biases": false,
                  "norm_eps": 0.01,
                  "layer_scale": true,
                  "layer_scale_init": 0.01,
                  "decoder_transformer_lengths_str": "1",
                  "decoder_convs_kernels_str": "3",
                  "decoder_convs_strides_str": "1",
                  "voice": {"casual_male": 1}
                }
              }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn loads_tiny_inference_modules_from_varbuilder() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

        let modules = VoxtralInferenceModules::load(&config, vb).unwrap();

        assert_eq!(modules.embeddings.tok_embeddings.hidden_size(), 8);
        assert_eq!(modules.embeddings.audio_vocab_size, 128);
        assert_eq!(modules.language.layers.len(), 2);
        assert_eq!(modules.acoustic.layers.len(), 1);
        assert_eq!(
            modules.language.layers[0].attention.wq.weight().dims(),
            &[8, 8]
        );
        assert_eq!(
            modules.language.layers[0].attention.wk.weight().dims(),
            &[4, 8]
        );
        assert_eq!(
            modules.language.layers[0].feed_forward.w1.weight().dims(),
            &[16, 8]
        );
        assert_eq!(modules.acoustic.input_projection.weight().dims(), &[8, 2]);
        assert_eq!(
            modules.acoustic.semantic_codebook_output.weight().dims(),
            &[128, 8]
        );
    }
}
