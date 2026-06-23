use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{self as nn, linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use crate::{AudioTokenizerConfig, VoxtralConfig, VoxtralFeedForward};

const AUDIO_SPECIAL_TOKEN_COUNT: f64 = 2.0;

pub struct VoxtralAudioTokenizer {
    pub codebook: VoxtralAudioCodebook,
    pub input_conv: nn::Conv1d,
    pub stages: Vec<VoxtralCodecStage>,
    pub output_proj: nn::Conv1d,
    pub patch_size: usize,
    pub latent_dim: usize,
    pub frame_rate: f64,
}

pub struct VoxtralAudioCodebook {
    pub semantic_embedding: Tensor,
    pub semantic_dim: usize,
    pub acoustic_dim: usize,
    pub semantic_codebook_size: usize,
    pub acoustic_codebook_size: usize,
}

pub struct VoxtralCodecStage {
    pub layers: Vec<VoxtralCodecTransformerBlock>,
    pub upsample: Option<nn::ConvTranspose1d>,
    pub window_size: usize,
}

pub struct VoxtralCodecTransformerBlock {
    pub attention: VoxtralCodecAttention,
    pub feed_forward: VoxtralFeedForward,
    pub attention_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
    pub attention_scale: Tensor,
    pub ffn_scale: Tensor,
    pub window_size: usize,
}

pub struct VoxtralCodecAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl VoxtralAudioTokenizer {
    pub fn load(config: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        let tokenizer = &config.multimodal.audio_tokenizer_args;
        let vb = vb.pp("audio_tokenizer");
        let codebook = VoxtralAudioCodebook::load(tokenizer, vb.pp("quantizer"))?;
        let kernels = parse_csv_usize(&tokenizer.decoder_convs_kernels_str);
        let strides = parse_csv_usize(&tokenizer.decoder_convs_strides_str);
        let transformer_lengths = parse_csv_usize(&tokenizer.decoder_transformer_lengths_str);
        if kernels.is_empty() || strides.is_empty() || kernels.len() != strides.len() {
            candle_core::bail!(
                "invalid codec decoder conv config: kernels={:?} strides={:?}",
                kernels,
                strides
            );
        }
        if transformer_lengths.is_empty() {
            candle_core::bail!("invalid codec decoder transformer config: no stages");
        }
        if kernels.len() != transformer_lengths.len() {
            candle_core::bail!(
                "codec decoder expects one conv kernel per transformer stage, got kernels={:?} transformers={:?}",
                kernels,
                transformer_lengths
            );
        }

        let input_conv = load_weight_norm_conv1d(
            tokenizer.semantic_dim + tokenizer.acoustic_dim,
            tokenizer.dim,
            kernels[0],
            strides[0],
            vb.pp("decoder_blocks.0"),
        )?;

        let mut stages = Vec::with_capacity(transformer_lengths.len());
        let mut block_idx = 1usize;
        let mut window_size = initial_decoder_window(tokenizer, &strides)?;
        for (stage_idx, &n_layers) in transformer_lengths.iter().enumerate() {
            let transformer_vb = vb.pp(format!("decoder_blocks.{block_idx}"));
            let mut layers = Vec::with_capacity(n_layers);
            for layer_idx in 0..n_layers {
                layers.push(VoxtralCodecTransformerBlock::load(
                    tokenizer,
                    window_size,
                    transformer_vb.pp(format!("layers.{layer_idx}")),
                )?);
            }
            block_idx += 1;

            let upsample = if stage_idx + 1 < transformer_lengths.len()
                && (kernels[stage_idx + 1] != 1 || strides[stage_idx + 1] != 1)
            {
                let upsample = load_weight_norm_conv_transpose1d(
                    tokenizer.dim,
                    tokenizer.dim,
                    kernels[stage_idx + 1],
                    strides[stage_idx + 1],
                    vb.pp(format!("decoder_blocks.{block_idx}")),
                )?;
                block_idx += 1;
                Some(upsample)
            } else {
                None
            };

            stages.push(VoxtralCodecStage {
                layers,
                upsample,
                window_size,
            });

            if stage_idx + 1 < transformer_lengths.len() {
                window_size *= strides[stage_idx + 1].max(1);
            }
        }

        let output_proj = load_weight_norm_conv1d(
            tokenizer.dim,
            tokenizer.pretransform_patch_size,
            tokenizer.patch_proj_kernel_size,
            1,
            vb.pp("output_proj"),
        )?;
        let scale_factor: usize = strides.iter().product();
        let frame_rate = tokenizer.sampling_rate as f64
            / (tokenizer.pretransform_patch_size * scale_factor) as f64;

        Ok(Self {
            codebook,
            input_conv,
            stages,
            output_proj,
            patch_size: tokenizer.pretransform_patch_size,
            latent_dim: tokenizer.semantic_dim + tokenizer.acoustic_dim,
            frame_rate,
        })
    }

    /// Decode discrete codec IDs into the continuous codec latent space.
    ///
    /// Input shape is `[batch, semantic + acoustic_codebooks, frames]`; output
    /// shape is `[batch, semantic_dim + acoustic_dim, frames]`.
    pub fn decode_code_embeddings(&self, codes: &Tensor) -> Result<Tensor> {
        self.codebook.decode(codes)
    }

    pub fn decode_codes_to_waveform(&self, codes: &Tensor) -> Result<Tensor> {
        let mut hidden = self.decode_code_embeddings(codes)?;
        hidden = self.forward_input_projection(&hidden)?;
        for stage_idx in 0..self.stages.len() {
            hidden = self.forward_stage_transformers(stage_idx, &hidden)?;
            if let Some(upsampled) = self.forward_stage_upsample(stage_idx, &hidden)? {
                hidden = upsampled;
            }
        }

        let patches = causal_conv1d(&self.output_proj, &hidden)?;
        let (batch, patch_size, frames) = patches.dims3()?;
        patches
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, 1, frames * patch_size))
    }

    /// Run the first decoder projection: `[B, 292, T] -> [B, 1024, T]`.
    pub fn forward_input_projection(&self, latents: &Tensor) -> Result<Tensor> {
        causal_conv1d(&self.input_conv, latents)
    }

    pub fn forward_stage_upsample(
        &self,
        stage_idx: usize,
        hidden: &Tensor,
    ) -> Result<Option<Tensor>> {
        let Some(stage) = self.stages.get(stage_idx) else {
            candle_core::bail!("codec stage {stage_idx} is out of range");
        };
        stage
            .upsample
            .as_ref()
            .map(|conv| causal_conv_transpose1d(conv, hidden))
            .transpose()
    }

    pub fn forward_stage_transformers(&self, stage_idx: usize, hidden: &Tensor) -> Result<Tensor> {
        let Some(stage) = self.stages.get(stage_idx) else {
            candle_core::bail!("codec stage {stage_idx} is out of range");
        };
        let mut hidden = hidden.transpose(1, 2)?.contiguous()?;
        for layer in &stage.layers {
            hidden = layer.forward(&hidden)?;
        }
        hidden.transpose(1, 2)
    }

    pub fn semantic_dim(&self) -> usize {
        self.codebook.semantic_dim
    }

    pub fn acoustic_dim(&self) -> usize {
        self.codebook.acoustic_dim
    }
}

impl VoxtralAudioCodebook {
    pub fn load(config: &AudioTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let semantic_vb = vb.pp("semantic_codebook");
        let embedding_sum = semantic_vb.get(
            (config.semantic_codebook_size, config.semantic_dim),
            "embedding_sum",
        )?;
        let cluster_usage = semantic_vb
            .get(config.semantic_codebook_size, "cluster_usage")?
            .to_dtype(embedding_sum.dtype())?
            .clamp(1e-5f32, f32::MAX)?
            .reshape((config.semantic_codebook_size, 1))?;
        let semantic_embedding = embedding_sum.broadcast_div(&cluster_usage)?;

        Ok(Self {
            semantic_embedding,
            semantic_dim: config.semantic_dim,
            acoustic_dim: config.acoustic_dim,
            semantic_codebook_size: config.semantic_codebook_size,
            acoustic_codebook_size: config.acoustic_codebook_size,
        })
    }

    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let (batch, codebooks, frames) = codes.dims3()?;
        let expected_codebooks = 1 + self.acoustic_dim;
        if codebooks != expected_codebooks {
            candle_core::bail!("expected {expected_codebooks} codebooks, got {codebooks}");
        }

        let semantic_ids = codes
            .narrow(1, 0, 1)?
            .reshape((batch, frames))?
            .to_dtype(DType::F32)?;
        let semantic_ids = ((semantic_ids - AUDIO_SPECIAL_TOKEN_COUNT)?
            .clamp(0.0f32, (self.semantic_codebook_size - 1) as f32)?
            .round()?)
        .to_dtype(DType::U32)?
        .reshape((batch * frames,))?
        .contiguous()?;
        let semantic = self
            .semantic_embedding
            .embedding(&semantic_ids)?
            .reshape((batch, frames, self.semantic_dim))?
            .transpose(1, 2)?;

        let acoustic = codes
            .narrow(1, 1, self.acoustic_dim)?
            .to_dtype(DType::F32)?;
        let acoustic = ((acoustic - AUDIO_SPECIAL_TOKEN_COUNT)?
            .clamp(0.0f32, (self.acoustic_codebook_size - 1) as f32)?
            * (2.0 / (self.acoustic_codebook_size - 1) as f64))?
            - 1.0;
        let acoustic = acoustic?.to_dtype(semantic.dtype())?;

        Tensor::cat(&[&semantic, &acoustic], 1)
    }
}

impl VoxtralCodecTransformerBlock {
    pub fn load(config: &AudioTokenizerConfig, window_size: usize, vb: VarBuilder) -> Result<Self> {
        let attention = VoxtralCodecAttention::load(config, vb.pp("attention"))?;
        let feed_forward =
            VoxtralFeedForward::load(config.dim, config.hidden_dim, vb.pp("feed_forward"))?;
        let attention_norm = rms_norm(config.dim, config.norm_eps, vb.pp("attention_norm"))?;
        let ffn_norm = rms_norm(config.dim, config.norm_eps, vb.pp("ffn_norm"))?;
        let attention_scale = vb.get(config.dim, "attention_scale")?;
        let ffn_scale = vb.get(config.dim, "ffn_scale")?;

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            attention_scale,
            ffn_scale,
            window_size,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let dim = hidden_states.dim(D::Minus1)?;
        let attention_input = self.attention_norm.forward(hidden_states)?;
        let attention = self.attention.forward(&attention_input, self.window_size)?;
        let attention_scale = self
            .attention_scale
            .to_dtype(attention.dtype())?
            .reshape((1, 1, dim))?;
        let hidden_states = (hidden_states + attention.broadcast_mul(&attention_scale)?)?;

        let ffn_input = self.ffn_norm.forward(&hidden_states)?;
        let ffn = self.feed_forward.forward(&ffn_input)?;
        let ffn_scale = self.ffn_scale.to_dtype(ffn.dtype())?.reshape((1, 1, dim))?;
        hidden_states + ffn.broadcast_mul(&ffn_scale)?
    }
}

impl VoxtralCodecAttention {
    pub fn load(config: &AudioTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.n_kv_heads * config.head_dim;

        let wq = linear_no_bias(config.dim, q_dim, vb.pp("wq"))?;
        let wk = linear_no_bias(config.dim, kv_dim, vb.pp("wk"))?;
        let wv = linear_no_bias(config.dim, kv_dim, vb.pp("wv"))?;
        let wo = linear_no_bias(q_dim, config.dim, vb.pp("wo"))?;
        let q_norm = rms_norm(q_dim, config.qk_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(kv_dim, config.qk_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, window_size: usize) -> Result<Tensor> {
        let (batch, seq_len, _dim) = hidden_states.dims3()?;
        let repeat = self.n_heads / self.n_kv_heads;
        let q_dim = self.n_heads * self.head_dim;

        let query = linear_forward(&self.wq, hidden_states)?;
        let key = linear_forward(&self.wk, hidden_states)?;
        let value = linear_forward(&self.wv, hidden_states)?;
        let query = self
            .q_norm
            .forward(&query)?
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key =
            self.k_norm
                .forward(&key)?
                .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?;
        let value = value.reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?;
        let key = repeat_kv(&key, repeat)?.transpose(1, 2)?.contiguous()?;
        let value = repeat_kv(&value, repeat)?.transpose(1, 2)?.contiguous()?;

        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let scores = (query.matmul(&key.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let mask = alibi_causal_sliding_mask(
            self.n_heads,
            seq_len,
            window_size,
            scores.dtype(),
            scores.device(),
        )?;
        let scores = scores.broadcast_add(&mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let output = weights
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape((batch, seq_len, q_dim))?;

        linear_forward(&self.wo, &output)
    }
}

fn linear_forward(linear: &Linear, xs: &Tensor) -> Result<Tensor> {
    let dims = xs.dims();
    if dims.len() <= 2 {
        return linear.forward(xs);
    }

    let input_dim = *dims.last().expect("tensor rank checked above");
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let flat = xs.reshape((batch, input_dim))?;
    let flat = linear.forward(&flat)?;
    let output_dim = linear.weight().dim(0)?;
    let mut output_shape = dims.to_vec();
    *output_shape.last_mut().expect("tensor rank checked above") = output_dim;
    flat.reshape(output_shape)
}

fn repeat_kv(hidden_states: &Tensor, repeat: usize) -> Result<Tensor> {
    if repeat == 1 {
        return Ok(hidden_states.clone());
    }
    let (batch, seq_len, n_kv_heads, head_dim) = hidden_states.dims4()?;
    let expanded = hidden_states
        .unsqueeze(3)?
        .expand((batch, seq_len, n_kv_heads, repeat, head_dim))?;
    expanded.reshape((batch, seq_len, n_kv_heads * repeat, head_dim))
}

fn alibi_causal_sliding_mask(
    n_heads: usize,
    seq_len: usize,
    window_size: usize,
    dtype: DType,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let slopes = alibi_slopes(n_heads);
    let mut values = Vec::with_capacity(n_heads * seq_len * seq_len);
    for &slope in &slopes {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let rel = j as isize - i as isize;
                let outside_window = rel > 0 || rel < -(window_size as isize);
                if outside_window {
                    values.push(f32::NEG_INFINITY);
                } else {
                    values.push(slope * rel as f32);
                }
            }
        }
    }
    Tensor::from_vec(values, (1, n_heads, seq_len, seq_len), device)?.to_dtype(dtype)
}

fn alibi_slopes(n_heads: usize) -> Vec<f32> {
    if n_heads == 0 {
        return Vec::new();
    }
    if n_heads.is_power_of_two() {
        let ratio = 2f32.powf(-8.0 / n_heads as f32);
        return (0..n_heads).map(|idx| ratio.powi(idx as i32)).collect();
    }

    let lower_power = 1usize << (usize::BITS - n_heads.leading_zeros() - 1);
    let mut slopes = alibi_slopes(lower_power);
    let extra = alibi_slopes(lower_power * 2);
    slopes.extend(extra.into_iter().step_by(2).take(n_heads - lower_power));
    slopes
}

fn load_weight_norm_conv1d(
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<nn::Conv1d> {
    let weight_v = vb.get(
        (out_ch, in_ch, kernel_size),
        "conv.parametrizations.weight.original1",
    )?;
    let weight_g = vb.get((out_ch, 1, 1), "conv.parametrizations.weight.original0")?;
    let v_norm = weight_v.sqr()?.sum_keepdim(&[1usize, 2][..])?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&v_norm)?;
    let config = nn::Conv1dConfig {
        stride,
        ..Default::default()
    };
    Ok(nn::Conv1d::new(weight, None, config))
}

fn load_weight_norm_conv_transpose1d(
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<nn::ConvTranspose1d> {
    let weight_v = vb.get(
        (in_ch, out_ch, kernel_size),
        "conv.parametrizations.weight.original1",
    )?;
    let weight_g = vb.get((in_ch, 1, 1), "conv.parametrizations.weight.original0")?;
    let v_norm = weight_v.sqr()?.sum_keepdim(&[1usize, 2][..])?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&v_norm)?;
    let config = nn::ConvTranspose1dConfig {
        stride,
        ..Default::default()
    };
    Ok(nn::ConvTranspose1d::new(weight, None, config))
}

fn causal_conv1d(conv: &nn::Conv1d, xs: &Tensor) -> Result<Tensor> {
    let (_, _, length) = xs.dims3()?;
    let config = conv.config();
    let effective_kernel = (conv.weight().dim(2)? - 1) * config.dilation + 1;
    let padding_total = effective_kernel.saturating_sub(config.stride);
    let n_frames = ((length + padding_total).saturating_sub(effective_kernel) as f64
        / config.stride as f64)
        + 1.0;
    let out_length = n_frames.ceil().max(0.0) as usize;
    if out_length == 0 {
        return Tensor::zeros(
            (xs.dim(0)?, conv.weight().dim(0)?, 0usize),
            xs.dtype(),
            xs.device(),
        );
    }
    let target_length = (out_length - 1) * config.stride + effective_kernel - padding_total;
    let extra_right = target_length.saturating_sub(length);
    let padded = replicate_pad_last_dim(xs, padding_total, extra_right)?;
    conv.forward(&padded)
}

fn causal_conv_transpose1d(conv: &nn::ConvTranspose1d, xs: &Tensor) -> Result<Tensor> {
    let config = conv.config();
    let kernel_size = conv.weight().dim(2)?;
    let total_padding = kernel_size.saturating_sub(config.stride);
    let right_padding = total_padding;
    let left_padding = total_padding.saturating_sub(right_padding);
    let out = conv.forward(xs)?;
    let out_len = out.dim(2)?;
    if left_padding + right_padding > out_len {
        candle_core::bail!(
            "cannot trim conv transpose output length {out_len} by left={left_padding} right={right_padding}"
        );
    }
    out.narrow(2, left_padding, out_len - left_padding - right_padding)
}

fn replicate_pad_last_dim(xs: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let length = xs.dim(2)?;
    if length == 0 && (left > 0 || right > 0) {
        candle_core::bail!("cannot replicate-pad empty temporal dimension");
    }

    let mut parts = Vec::with_capacity(3);
    if left > 0 {
        parts.push(xs.narrow(2, 0, 1)?.repeat((1, 1, left))?);
    }
    parts.push(xs.clone());
    if right > 0 {
        parts.push(xs.narrow(2, length - 1, 1)?.repeat((1, 1, right))?);
    }
    let refs: Vec<_> = parts.iter().collect();
    Tensor::cat(&refs, 2)
}

fn initial_decoder_window(config: &AudioTokenizerConfig, strides: &[usize]) -> Result<usize> {
    if !config.half_attn_window_upon_downsampling {
        return Ok(config.attn_sliding_window_size);
    }
    let upsample_after_first: usize = strides.iter().skip(1).product();
    let window = config.attn_sliding_window_size / upsample_after_first.max(1);
    if window == 0 {
        candle_core::bail!(
            "codec decoder window collapsed to zero: base={} strides={:?}",
            config.attn_sliding_window_size,
            strides
        );
    }
    Ok(window)
}

fn parse_csv_usize(raw: &str) -> Vec<usize> {
    raw.split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use super::*;
    use crate::transformer::tests::{tiny_config, tiny_config_json};

    #[test]
    fn loads_tiny_codec_decoder_from_varbuilder() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

        let codec = VoxtralAudioTokenizer::load(&config, vb).unwrap();

        assert_eq!(codec.semantic_dim(), 4);
        assert_eq!(codec.acoustic_dim(), 2);
        assert_eq!(codec.latent_dim, 6);
        assert_eq!(codec.input_conv.weight().dims(), &[8, 6, 3]);
        assert_eq!(codec.output_proj.weight().dims(), &[240, 8, 7]);
        assert_eq!(codec.stages.len(), 1);
        assert_eq!(codec.stages[0].layers.len(), 1);
        assert_eq!(codec.stages[0].window_size, 16);
    }

    #[test]
    fn decodes_tiny_codec_code_embeddings() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let codec = VoxtralAudioTokenizer::load(&config, vb).unwrap();
        let device = Device::Cpu;
        let codes = Tensor::new(&[2u32, 2, 3], &device)
            .unwrap()
            .reshape((1, 3, 1))
            .unwrap();

        let embeddings = codec.decode_code_embeddings(&codes).unwrap();
        let values = embeddings.to_vec3::<f32>().unwrap();

        assert_eq!(embeddings.dims(), &[1, 6, 1]);
        assert_eq!(values[0][0], vec![0.0]);
        assert_eq!(values[0][3], vec![0.0]);
        assert_eq!(values[0][4], vec![-1.0]);
        assert_eq!(values[0][5], vec![0.0]);
    }

    #[test]
    fn runs_tiny_codec_input_projection() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let codec = VoxtralAudioTokenizer::load(&config, vb).unwrap();
        let device = Device::Cpu;
        let latents = Tensor::zeros((1, codec.latent_dim, 3), DType::F32, &device).unwrap();

        let projected = codec.forward_input_projection(&latents).unwrap();

        assert_eq!(
            projected.dims(),
            &[1, config.multimodal.audio_tokenizer_args.dim, 3]
        );
    }

    #[test]
    fn trims_tiny_codec_upsample_projection() {
        let mut value: serde_json::Value = serde_json::from_str(tiny_config_json()).unwrap();
        let tokenizer = value
            .get_mut("multimodal")
            .and_then(|v| v.get_mut("audio_tokenizer_args"))
            .unwrap();
        tokenizer["decoder_transformer_lengths_str"] = serde_json::json!("1,1");
        tokenizer["decoder_convs_kernels_str"] = serde_json::json!("3,4");
        tokenizer["decoder_convs_strides_str"] = serde_json::json!("1,2");
        let config = VoxtralConfig::from_json_str(&serde_json::to_string(&value).unwrap()).unwrap();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let codec = VoxtralAudioTokenizer::load(&config, vb).unwrap();
        let device = Device::Cpu;
        let hidden = Tensor::zeros(
            (1, config.multimodal.audio_tokenizer_args.dim, 3),
            DType::F32,
            &device,
        )
        .unwrap();

        let upsampled = codec.forward_stage_upsample(0, &hidden).unwrap().unwrap();

        assert_eq!(
            upsampled.dims(),
            &[1, config.multimodal.audio_tokenizer_args.dim, 6]
        );
        assert_eq!(codec.stages[0].window_size, 8);
        assert_eq!(codec.stages[1].window_size, 16);
    }

    #[test]
    fn runs_tiny_codec_transformer_stage() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let codec = VoxtralAudioTokenizer::load(&config, vb).unwrap();
        let device = Device::Cpu;
        let hidden = Tensor::zeros(
            (1, config.multimodal.audio_tokenizer_args.dim, 3),
            DType::F32,
            &device,
        )
        .unwrap();

        let transformed = codec.forward_stage_transformers(0, &hidden).unwrap();

        assert_eq!(
            transformed.dims(),
            &[1, config.multimodal.audio_tokenizer_args.dim, 3]
        );
    }

    #[test]
    fn decodes_tiny_codec_waveform_shape() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let codec = VoxtralAudioTokenizer::load(&config, vb).unwrap();
        let device = Device::Cpu;
        let codes = Tensor::new(&[2u32, 2, 3, 2, 2, 3], &device)
            .unwrap()
            .reshape((1, 3, 2))
            .unwrap();

        let waveform = codec.decode_codes_to_waveform(&codes).unwrap();

        assert_eq!(waveform.dims(), &[1, 1, 480]);
    }

    #[test]
    fn builds_alibi_causal_sliding_mask() {
        let device = Device::Cpu;

        let mask = alibi_causal_sliding_mask(2, 4, 2, DType::F32, &device).unwrap();
        let values = mask
            .reshape((2 * 4 * 4,))
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let idx = |head: usize, row: usize, col: usize| head * 16 + row * 4 + col;
        assert_eq!(values[idx(0, 0, 0)], 0.0);
        assert!(values[idx(0, 0, 1)].is_infinite());
        assert!(values[idx(0, 3, 0)].is_infinite());
        assert_eq!(values[idx(0, 3, 1)], -2.0);
        assert_eq!(values[idx(1, 3, 1)], -0.125);
    }
}
