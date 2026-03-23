use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub istftnet: IstftNetConfig,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    pub hidden_dim: usize,
    #[serde(default = "default_max_dur")]
    pub max_dur: usize,
    pub n_layer: usize,
    #[serde(default = "default_n_mels")]
    pub n_mels: usize,
    pub n_token: usize,
    pub style_dim: usize,
    #[serde(default = "default_text_encoder_kernel_size")]
    pub text_encoder_kernel_size: usize,
    pub plbert: PlbertConfig,
    #[serde(default)]
    pub vocab: HashMap<String, i64>,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IstftNetConfig {
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_rates: Vec<usize>,
    pub gen_istft_hop_size: usize,
    pub gen_istft_n_fft: usize,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlbertConfig {
    pub hidden_size: usize,
    #[serde(default = "default_embedding_size")]
    pub embedding_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_plbert_dropout")]
    pub dropout: f64,
}

fn default_dropout() -> f64 {
    0.2
}
fn default_max_dur() -> usize {
    50
}
fn default_n_mels() -> usize {
    80
}
fn default_text_encoder_kernel_size() -> usize {
    5
}
fn default_embedding_size() -> usize {
    128
}
fn default_plbert_dropout() -> f64 {
    0.1
}
fn default_sample_rate() -> u32 {
    24000
}
