use serde::Deserialize;
use std::collections::HashMap;

pub use voicers_nn::albert::AlbertConfig;

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
pub struct ISTFTNetConfig {
    pub resblock_kernel_sizes: Vec<i32>,
    pub upsample_rates: Vec<i32>,
    pub upsample_initial_channel: i32,
    pub resblock_dilation_sizes: Vec<Vec<i32>>,
    pub upsample_kernel_sizes: Vec<i32>,
    pub gen_istft_n_fft: i32,
    pub gen_istft_hop_size: i32,
}
