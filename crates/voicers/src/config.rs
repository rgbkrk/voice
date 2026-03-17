use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub istftnet: ISTFTNetConfig,
    pub dim_in: i32,
    pub dropout: f32,
    pub hidden_dim: i32,
    pub max_conv_dim: i32,
    pub max_dur: i32,
    pub multispeaker: bool,
    pub n_layer: i32,
    pub n_mels: i32,
    pub n_token: i32,
    pub style_dim: i32,
    pub text_encoder_kernel_size: i32,
    pub plbert: AlbertConfig,
    pub vocab: HashMap<String, i32>,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: i32,
}

fn default_sample_rate() -> i32 {
    24000
}

#[derive(Debug, Clone, Deserialize)]
pub struct AlbertConfig {
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub max_position_embeddings: i32,
    #[serde(default = "default_embedding_size")]
    pub embedding_size: i32,
    #[serde(default = "default_inner_group_num")]
    pub inner_group_num: i32,
    #[serde(default = "default_num_hidden_groups")]
    pub num_hidden_groups: i32,
    #[serde(default = "default_hidden_dropout_prob")]
    pub hidden_dropout_prob: f32,
    #[serde(default = "default_attention_probs_dropout_prob")]
    pub attention_probs_dropout_prob: f32,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: i32,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
}

fn default_embedding_size() -> i32 {
    128
}
fn default_inner_group_num() -> i32 {
    1
}
fn default_num_hidden_groups() -> i32 {
    1
}
fn default_hidden_dropout_prob() -> f32 {
    0.1
}
fn default_attention_probs_dropout_prob() -> f32 {
    0.1
}
fn default_type_vocab_size() -> i32 {
    2
}
fn default_layer_norm_eps() -> f32 {
    1e-12
}
fn default_vocab_size() -> i32 {
    30522
}

#[derive(Debug, Clone, Deserialize)]
pub struct ISTFTNetConfig {
    pub resblock_kernel_sizes: Vec<i32>,
    pub upsample_rates: Vec<i32>,
    pub upsample_initial_channel: i32,
    pub resblock_dilation_sizes: Vec<Vec<i32>>,
    pub upsample_kernel_sizes: Vec<i32>,
    pub gen_istft_n_fft: i32,
    pub gen_istft_hop_size: i32,
}
