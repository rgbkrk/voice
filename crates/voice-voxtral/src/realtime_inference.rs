use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{
    conv1d, embedding, linear, linear_no_bias, rms_norm, Activation, Conv1d, Conv1dConfig,
    Embedding, Linear, RmsNorm, VarBuilder,
};

use crate::realtime::{REALTIME_ENCODER_PREFIX, REALTIME_STREAMS_PREFIX};
use crate::VoxtralRealtimeConfig;

pub struct VoxtralRealtimeInferenceModules {
    pub token_embeddings: VoxtralRealtimeTokenEmbeddings,
    pub audio_stem: VoxtralRealtimeAudioStem,
    pub audio_projector: VoxtralRealtimeAudioProjector,
}

pub struct VoxtralRealtimeAudioModules {
    pub stem: VoxtralRealtimeAudioStem,
    pub transformer: VoxtralRealtimeAudioTransformer,
    pub projector: VoxtralRealtimeAudioProjector,
}

pub struct VoxtralRealtimeTokenEmbeddings {
    pub tok_embeddings: Embedding,
}

pub struct VoxtralRealtimeAudioStem {
    pub conv1: Conv1d,
    pub conv2: Conv1d,
    pub num_mel_bins: usize,
    pub hidden_size: usize,
}

pub struct VoxtralRealtimeAudioProjector {
    pub linear_1: Linear,
    pub linear_2: Linear,
    pub downsample_factor: usize,
    pub audio_hidden_size: usize,
    pub text_hidden_size: usize,
}

pub struct VoxtralRealtimeAudioTransformer {
    pub layers: Vec<VoxtralRealtimeAudioTransformerBlock>,
    pub norm: RmsNorm,
    pub rope_theta: f64,
    pub sliding_window: usize,
}

pub struct VoxtralRealtimeAudioTransformerBlock {
    pub attention: VoxtralRealtimeAudioAttention,
    pub feed_forward: VoxtralRealtimeAudioFeedForward,
    pub attention_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
}

pub struct VoxtralRealtimeAudioAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

pub struct VoxtralRealtimeAudioFeedForward {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
}

impl VoxtralRealtimeInferenceModules {
    pub fn load(config: &VoxtralRealtimeConfig, vb: VarBuilder) -> Result<Self> {
        let token_embeddings = VoxtralRealtimeTokenEmbeddings::load(config, vb.clone())?;
        let audio_stem = VoxtralRealtimeAudioStem::load(config, vb.clone())?;
        let audio_projector = VoxtralRealtimeAudioProjector::load(config, vb)?;

        Ok(Self {
            token_embeddings,
            audio_stem,
            audio_projector,
        })
    }
}

impl VoxtralRealtimeAudioModules {
    pub fn load(config: &VoxtralRealtimeConfig, vb: VarBuilder) -> Result<Self> {
        let stem = VoxtralRealtimeAudioStem::load(config, vb.clone())?;
        let transformer = VoxtralRealtimeAudioTransformer::load(config, vb.clone())?;
        let projector = VoxtralRealtimeAudioProjector::load(config, vb)?;

        Ok(Self {
            stem,
            transformer,
            projector,
        })
    }

    pub fn forward(&self, input_features: &Tensor, start_pos: usize) -> Result<Tensor> {
        let hidden = self.stem.forward_features(input_features)?;
        let hidden = self.transformer.forward(&hidden, start_pos)?;
        self.projector.forward(&hidden)
    }
}

impl VoxtralRealtimeAudioTransformer {
    pub fn load(config: &VoxtralRealtimeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = &config.multimodal.whisper_model_args.encoder_args;
        let mut layers = Vec::with_capacity(encoder.n_layers);
        for layer_idx in 0..encoder.n_layers {
            layers.push(VoxtralRealtimeAudioTransformerBlock::load(
                config,
                vb.pp(REALTIME_ENCODER_PREFIX)
                    .pp("transformer")
                    .pp(format!("layers.{layer_idx}")),
            )?);
        }
        let norm = rms_norm(
            encoder.dim,
            encoder.norm_eps,
            vb.pp(REALTIME_ENCODER_PREFIX).pp("transformer.norm"),
        )?;

        Ok(Self {
            layers,
            norm,
            rope_theta: encoder.rope_theta,
            sliding_window: encoder.sliding_window,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, start_pos: usize) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                start_pos,
                self.rope_theta,
                self.sliding_window,
            )?;
        }
        self.norm.forward(&hidden_states)
    }
}

impl VoxtralRealtimeAudioTransformerBlock {
    pub fn load(config: &VoxtralRealtimeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = &config.multimodal.whisper_model_args.encoder_args;
        let attention = VoxtralRealtimeAudioAttention::load(config, vb.pp("attention"))?;
        let feed_forward = VoxtralRealtimeAudioFeedForward::load(config, vb.pp("feed_forward"))?;
        let attention_norm = rms_norm(encoder.dim, encoder.norm_eps, vb.pp("attention_norm"))?;
        let ffn_norm = rms_norm(encoder.dim, encoder.norm_eps, vb.pp("ffn_norm"))?;

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        rope_theta: f64,
        sliding_window: usize,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let attention_input = self.attention_norm.forward(hidden_states)?;
        let attention_output =
            self.attention
                .forward(&attention_input, start_pos, rope_theta, sliding_window)?;
        let hidden_states = (attention_output + residual)?;

        let residual = hidden_states.clone();
        let ffn_input = self.ffn_norm.forward(&hidden_states)?;
        let ffn_output = self.feed_forward.forward(&ffn_input)?;
        ffn_output + residual
    }
}

impl VoxtralRealtimeAudioAttention {
    pub fn load(config: &VoxtralRealtimeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = &config.multimodal.whisper_model_args.encoder_args;
        let q_dim = encoder.n_heads * encoder.head_dim;
        let kv_dim = encoder.n_kv_heads * encoder.head_dim;
        let wq = linear(encoder.dim, q_dim, vb.pp("wq"))?;
        let wk = linear_no_bias(encoder.dim, kv_dim, vb.pp("wk"))?;
        let wv = linear(encoder.dim, kv_dim, vb.pp("wv"))?;
        let wo = linear(q_dim, encoder.dim, vb.pp("wo"))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            n_heads: encoder.n_heads,
            n_kv_heads: encoder.n_kv_heads,
            head_dim: encoder.head_dim,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        rope_theta: f64,
        sliding_window: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _dim) = hidden_states.dims3()?;
        let repeat = self.n_heads / self.n_kv_heads;
        let query = self
            .wq
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key = self
            .wk
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value = self.wv.forward(hidden_states)?.reshape((
            batch,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
        ))?;

        let (cos, sin) = rope_frequencies(
            seq_len,
            self.head_dim,
            start_pos,
            rope_theta,
            query.dtype(),
            query.device(),
        )?;
        let query = candle_nn::rotary_emb::rope_i(&query, &cos, &sin)?;
        let key = candle_nn::rotary_emb::rope_i(&key, &cos, &sin)?;
        let key = repeat_kv_heads(&key, repeat)?;
        let value = repeat_kv(&value, repeat)?.transpose(1, 2)?.contiguous()?;

        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let scores = (query
            .matmul(&key.transpose(candle_core::D::Minus2, candle_core::D::Minus1)?)?
            * scale)?;
        let mask = sliding_causal_mask(
            seq_len,
            start_pos,
            sliding_window,
            scores.dtype(),
            scores.device(),
        )?;
        let scores = scores.broadcast_add(&mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let output = weights.matmul(&value)?.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ))?;

        self.wo.forward(&output)
    }
}

impl VoxtralRealtimeAudioFeedForward {
    pub fn load(config: &VoxtralRealtimeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = &config.multimodal.whisper_model_args.encoder_args;
        let w1 = linear_no_bias(encoder.dim, encoder.hidden_dim, vb.pp("w1"))?;
        let w2 = linear(encoder.hidden_dim, encoder.dim, vb.pp("w2"))?;
        let w3 = linear_no_bias(encoder.dim, encoder.hidden_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(hidden_states)?;
        let gate = Activation::Silu.forward(&gate)?;
        let up = self.w3.forward(hidden_states)?;
        let hidden_states = gate.broadcast_mul(&up)?;
        self.w2.forward(&hidden_states)
    }
}

impl VoxtralRealtimeTokenEmbeddings {
    pub fn load(config: &VoxtralRealtimeConfig, vb: VarBuilder) -> Result<Self> {
        let tok_embeddings = embedding(
            config.vocab_size,
            config.dim,
            vb.pp(REALTIME_STREAMS_PREFIX).pp("tok_embeddings"),
        )?;
        Ok(Self { tok_embeddings })
    }

    pub fn forward(&self, token_ids: &[usize], device: &Device) -> Result<Tensor> {
        let token_ids = token_ids.iter().map(|id| *id as u32).collect::<Vec<_>>();
        let input_ids = Tensor::new(token_ids.as_slice(), device)?.reshape((1, token_ids.len()))?;
        self.tok_embeddings.forward(&input_ids)
    }
}

impl VoxtralRealtimeAudioStem {
    pub fn load(config: &VoxtralRealtimeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = &config.multimodal.whisper_model_args.encoder_args;
        let encoding = &encoder.audio_encoding_args;
        let conv1 = conv1d(
            encoding.num_mel_bins,
            encoder.dim,
            3,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                ..Default::default()
            },
            vb.pp(REALTIME_ENCODER_PREFIX).pp("conv_layers.0.conv"),
        )?;
        let conv2 = conv1d(
            encoder.dim,
            encoder.dim,
            3,
            Conv1dConfig {
                padding: 0,
                stride: 2,
                ..Default::default()
            },
            vb.pp(REALTIME_ENCODER_PREFIX).pp("conv_layers.1.conv"),
        )?;

        Ok(Self {
            conv1,
            conv2,
            num_mel_bins: encoding.num_mel_bins,
            hidden_size: encoder.dim,
        })
    }

    /// Run the causal conv stem.
    ///
    /// Input shape is `[batch, mel_bins, mel_frames]`; output shape is
    /// `[batch, ceil(mel_frames / 2), hidden_size]`.
    pub fn forward_features(&self, input_features: &Tensor) -> Result<Tensor> {
        let (_batch, mel_bins, _frames) = input_features.dims3()?;
        if mel_bins != self.num_mel_bins {
            candle_core::bail!(
                "input_features has {mel_bins} mel bins, expected {}",
                self.num_mel_bins
            );
        }

        let hidden = zero_causal_conv1d(&self.conv1, input_features)?;
        let hidden = Activation::Gelu.forward(&hidden)?;
        let hidden = zero_causal_conv1d(&self.conv2, &hidden)?;
        let hidden = Activation::Gelu.forward(&hidden)?;
        hidden.transpose(1, 2)?.contiguous()
    }
}

impl VoxtralRealtimeAudioProjector {
    pub fn load(config: &VoxtralRealtimeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = &config.multimodal.whisper_model_args.encoder_args;
        let downsample_factor = config.downsample_factor();
        let audio_hidden_size = encoder.dim;
        let text_hidden_size = config.dim;
        let adapter_input_dim = audio_hidden_size * downsample_factor;
        let vb = vb
            .pp(REALTIME_STREAMS_PREFIX)
            .pp("audio_language_projection");
        let linear_1 = linear_no_bias(adapter_input_dim, text_hidden_size, vb.pp("0"))?;
        let linear_2 = linear_no_bias(text_hidden_size, text_hidden_size, vb.pp("2"))?;

        Ok(Self {
            linear_1,
            linear_2,
            downsample_factor,
            audio_hidden_size,
            text_hidden_size,
        })
    }

    /// Left-truncate to a downsample multiple, then project `[B, S, H]`
    /// encoder states to `[B, S / downsample, text_hidden]`.
    pub fn forward(&self, encoder_hidden: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = encoder_hidden.dims3()?;
        if hidden_size != self.audio_hidden_size {
            candle_core::bail!(
                "encoder hidden size {hidden_size} does not match expected {}",
                self.audio_hidden_size
            );
        }
        if self.downsample_factor == 0 {
            candle_core::bail!("downsample_factor must be greater than zero");
        }

        let trunc = seq_len % self.downsample_factor;
        let kept = seq_len - trunc;
        if kept == 0 {
            candle_core::bail!(
                "encoder sequence length {seq_len} is shorter than downsample factor {}",
                self.downsample_factor
            );
        }
        let hidden = if trunc > 0 {
            encoder_hidden.narrow(1, trunc, kept)?
        } else {
            encoder_hidden.clone()
        };
        let hidden = hidden.reshape((
            batch,
            kept / self.downsample_factor,
            self.audio_hidden_size * self.downsample_factor,
        ))?;
        let hidden = self.linear_1.forward(&hidden)?;
        let hidden = Activation::Gelu.forward(&hidden)?;
        self.linear_2.forward(&hidden)
    }
}

fn zero_causal_conv1d(conv: &Conv1d, xs: &Tensor) -> Result<Tensor> {
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
    let padded = zero_pad_last_dim(xs, padding_total, extra_right)?;
    conv.forward(&padded)
}

fn zero_pad_last_dim(xs: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let (batch, channels, _length) = xs.dims3()?;
    let mut parts = Vec::with_capacity(3);
    if left > 0 {
        parts.push(Tensor::zeros(
            (batch, channels, left),
            xs.dtype(),
            xs.device(),
        )?);
    }
    parts.push(xs.clone());
    if right > 0 {
        parts.push(Tensor::zeros(
            (batch, channels, right),
            xs.dtype(),
            xs.device(),
        )?);
    }
    Tensor::cat(&parts, 2)
}

fn repeat_kv(hidden_states: &Tensor, repeat: usize) -> Result<Tensor> {
    if repeat == 1 {
        return Ok(hidden_states.clone());
    }

    let (_batch, _seq_len, n_kv_heads, _head_dim) = hidden_states.dims4()?;
    let mut repeated = Vec::with_capacity(n_kv_heads * repeat);
    for head_idx in 0..n_kv_heads {
        let head = hidden_states.narrow(2, head_idx, 1)?;
        for _ in 0..repeat {
            repeated.push(head.clone());
        }
    }
    Tensor::cat(&repeated, 2)
}

fn repeat_kv_heads(hidden_states: &Tensor, repeat: usize) -> Result<Tensor> {
    if repeat == 1 {
        return Ok(hidden_states.clone());
    }

    let (batch, n_kv_heads, seq_len, head_dim) = hidden_states.dims4()?;
    let expanded = hidden_states
        .unsqueeze(2)?
        .expand((batch, n_kv_heads, repeat, seq_len, head_dim))?;
    expanded.reshape((batch, n_kv_heads * repeat, seq_len, head_dim))
}

fn rope_frequencies(
    seq_len: usize,
    head_dim: usize,
    start_pos: usize,
    theta: f64,
    dtype: candle_core::DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let mut cos = Vec::with_capacity(seq_len * half_dim);
    let mut sin = Vec::with_capacity(seq_len * half_dim);
    for pos in start_pos..start_pos + seq_len {
        for idx in 0..half_dim {
            let freq = 1.0 / theta.powf((2 * idx) as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }

    let cos = Tensor::from_vec(cos, (seq_len, half_dim), device)?.to_dtype(dtype)?;
    let sin = Tensor::from_vec(sin, (seq_len, half_dim), device)?.to_dtype(dtype)?;
    Ok((cos, sin))
}

fn sliding_causal_mask(
    seq_len: usize,
    start_pos: usize,
    sliding_window: usize,
    dtype: candle_core::DType,
    device: &Device,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(seq_len * seq_len);
    for query_pos in 0..seq_len {
        let query_abs = start_pos + query_pos;
        let valid_start = if sliding_window == 0 {
            0
        } else {
            query_abs.saturating_sub(sliding_window - 1)
        };
        for key_pos in 0..seq_len {
            let key_abs = start_pos + key_pos;
            if key_abs <= query_abs && key_abs >= valid_start {
                values.push(0.0);
            } else {
                values.push(f32::NEG_INFINITY);
            }
        }
    }
    Tensor::from_vec(values, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn tiny_realtime_config() -> VoxtralRealtimeConfig {
        VoxtralRealtimeConfig {
            dim: 8,
            n_layers: 1,
            head_dim: 4,
            hidden_dim: 16,
            n_heads: 2,
            n_kv_heads: 1,
            use_biases: false,
            causal: true,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-5,
            vocab_size: 64,
            model_parallel: 1,
            tied_embeddings: true,
            sliding_window: 32,
            model_max_length: 128,
            multimodal: crate::VoxtralRealtimeMultimodalConfig {
                whisper_model_args: crate::VoxtralRealtimeWhisperModelConfig {
                    encoder_args: crate::VoxtralRealtimeAudioEncoderConfig {
                        audio_encoding_args: crate::VoxtralRealtimeAudioEncodingConfig {
                            sampling_rate: crate::REALTIME_SAMPLE_RATE,
                            frame_rate: 12.5,
                            num_mel_bins: 4,
                            hop_length: 160,
                            window_size: 400,
                            chunk_length_s: None,
                            global_log_mel_max: 1.5,
                            transcription_format: crate::REALTIME_TRANSCRIPTION_FORMAT.to_string(),
                        },
                        dim: 8,
                        n_layers: 1,
                        head_dim: 4,
                        hidden_dim: 16,
                        n_heads: 2,
                        vocab_size: 64,
                        n_kv_heads: 1,
                        use_biases: true,
                        use_cache: false,
                        rope_theta: 1_000_000.0,
                        causal: true,
                        norm_eps: 1e-5,
                        pos_embed: "rope".to_string(),
                        max_source_positions: None,
                        ffn_type: "swiglu".to_string(),
                        norm_type: "rms_norm".to_string(),
                        sliding_window: 16,
                    },
                    downsample_args: crate::VoxtralRealtimeDownsampleConfig {
                        downsample_factor: 2,
                    },
                },
            },
            ada_rms_norm_t_cond: true,
            ada_rms_norm_t_cond_dim: Some(32),
        }
    }

    #[test]
    fn runs_tiny_realtime_conv_stem() {
        let config = tiny_realtime_config();
        let modules = VoxtralRealtimeInferenceModules::load(
            &config,
            VarBuilder::zeros(DType::F32, &Device::Cpu),
        )
        .unwrap();
        let input = Tensor::zeros((1, 4, 9), DType::F32, &Device::Cpu).unwrap();

        let hidden = modules.audio_stem.forward_features(&input).unwrap();

        assert_eq!(hidden.dims(), &[1, 5, 8]);
        assert_eq!(modules.audio_stem.conv1.weight().dims(), &[8, 4, 3]);
        assert_eq!(modules.audio_stem.conv2.weight().dims(), &[8, 8, 3]);
    }

    #[test]
    fn projects_tiny_realtime_audio_states_after_left_truncation() {
        let config = tiny_realtime_config();
        let modules = VoxtralRealtimeInferenceModules::load(
            &config,
            VarBuilder::zeros(DType::F32, &Device::Cpu),
        )
        .unwrap();
        let encoder_hidden = Tensor::zeros((1, 5, 8), DType::F32, &Device::Cpu).unwrap();

        let projected = modules.audio_projector.forward(&encoder_hidden).unwrap();

        assert_eq!(projected.dims(), &[1, 2, 8]);
        assert_eq!(modules.audio_projector.linear_1.weight().dims(), &[8, 16]);
        assert_eq!(modules.audio_projector.linear_2.weight().dims(), &[8, 8]);
    }

    #[test]
    fn runs_tiny_realtime_audio_transformer() {
        let config = tiny_realtime_config();
        let transformer = VoxtralRealtimeAudioTransformer::load(
            &config,
            VarBuilder::zeros(DType::F32, &Device::Cpu),
        )
        .unwrap();
        let hidden_states = Tensor::zeros((1, 5, 8), DType::F32, &Device::Cpu).unwrap();

        let hidden = transformer.forward(&hidden_states, 0).unwrap();

        assert_eq!(transformer.layers.len(), 1);
        assert_eq!(transformer.layers[0].attention.wq.weight().dims(), &[8, 8]);
        assert_eq!(transformer.layers[0].attention.wk.weight().dims(), &[4, 8]);
        assert_eq!(
            transformer.layers[0].feed_forward.w2.weight().dims(),
            &[8, 16]
        );
        assert_eq!(hidden.dims(), &[1, 5, 8]);
    }

    #[test]
    fn runs_tiny_realtime_audio_modules() {
        let config = tiny_realtime_config();
        let audio_modules =
            VoxtralRealtimeAudioModules::load(&config, VarBuilder::zeros(DType::F32, &Device::Cpu))
                .unwrap();
        let input_features = Tensor::zeros((1, 4, 8), DType::F32, &Device::Cpu).unwrap();

        let audio_embeds = audio_modules.forward(&input_features, 0).unwrap();

        assert_eq!(audio_embeds.dims(), &[1, 2, 8]);
    }

    #[test]
    fn embeds_tiny_realtime_tokens() {
        let config = tiny_realtime_config();
        let modules = VoxtralRealtimeInferenceModules::load(
            &config,
            VarBuilder::zeros(DType::F32, &Device::Cpu),
        )
        .unwrap();

        let embeddings = modules
            .token_embeddings
            .forward(
                &[
                    crate::REALTIME_BOS_TOKEN_ID,
                    crate::REALTIME_STREAMING_PAD_TOKEN_ID,
                ],
                &Device::Cpu,
            )
            .unwrap();

        assert_eq!(embeddings.dims(), &[1, 2, 8]);
    }

    #[test]
    fn loads_local_realtime_inference_modules_when_env_is_set() {
        let Ok(dir) = std::env::var("VOXTRAL_REALTIME_LOCAL_DIR") else {
            return;
        };
        if std::env::var("VOXTRAL_REALTIME_LOAD_FULL").as_deref() != Ok("1") {
            return;
        }

        let model = crate::VoxtralRealtimeModel::load_from_dir(dir).unwrap();
        let modules = model
            .load_inference_modules(DType::F32, &Device::Cpu)
            .unwrap();

        assert_eq!(
            modules.token_embeddings.tok_embeddings.embeddings().dims(),
            &[model.config().vocab_size, model.config().dim]
        );
        assert_eq!(modules.audio_stem.conv1.weight().dims(), &[1280, 128, 3]);
        assert_eq!(modules.audio_stem.conv2.weight().dims(), &[1280, 1280, 3]);
        assert_eq!(
            modules.audio_projector.linear_1.weight().dims(),
            &[3072, 5120]
        );
        assert_eq!(
            modules.audio_projector.linear_2.weight().dims(),
            &[3072, 3072]
        );
    }

    #[test]
    fn runs_local_realtime_mel_stem_projector_when_env_is_set() {
        let Ok(dir) = std::env::var("VOXTRAL_REALTIME_LOCAL_DIR") else {
            return;
        };
        if std::env::var("VOXTRAL_REALTIME_LOAD_FULL").as_deref() != Ok("1") {
            return;
        }

        let model = crate::VoxtralRealtimeModel::load_from_dir(dir).unwrap();
        let modules = model
            .load_inference_modules(DType::F32, &Device::Cpu)
            .unwrap();
        let samples = vec![0.0f32; 1280];
        let mel = crate::realtime_log_mel_spectrogram(model.config(), &samples).unwrap();
        let input_features = Tensor::from_vec(
            mel.to_channel_major(),
            (1, mel.mel_bins, mel.frames),
            &Device::Cpu,
        )
        .unwrap();

        let encoder_stem_hidden = modules
            .audio_stem
            .forward_features(&input_features)
            .unwrap();
        let audio_embeds = modules
            .audio_projector
            .forward(&encoder_stem_hidden)
            .unwrap();

        assert_eq!(mel.frames, 8);
        assert_eq!(encoder_stem_hidden.dims(), &[1, 4, 1280]);
        assert_eq!(audio_embeds.dims(), &[1, 1, 3072]);
    }

    #[test]
    fn loads_local_realtime_audio_transformer_when_env_is_set() {
        let Ok(dir) = std::env::var("VOXTRAL_REALTIME_LOCAL_DIR") else {
            return;
        };
        if std::env::var("VOXTRAL_REALTIME_LOAD_FULL").as_deref() != Ok("1") {
            return;
        }

        let model = crate::VoxtralRealtimeModel::load_from_dir(dir).unwrap();
        let transformer = model
            .load_audio_transformer(DType::BF16, &Device::Cpu)
            .unwrap();

        assert_eq!(transformer.layers.len(), 32);
        assert_eq!(
            transformer.layers[0].attention.wq.weight().dims(),
            &[2048, 1280]
        );
        assert_eq!(
            transformer.layers[0].attention.wk.weight().dims(),
            &[2048, 1280]
        );
        assert_eq!(
            transformer.layers[0].attention.wv.weight().dims(),
            &[2048, 1280]
        );
        assert_eq!(
            transformer.layers[0].attention.wo.weight().dims(),
            &[1280, 2048]
        );
        assert_eq!(
            transformer.layers[31].feed_forward.w2.weight().dims(),
            &[1280, 5120]
        );
        assert_eq!(transformer.norm.weight().dims(), &[1280]);
    }

    #[test]
    fn loads_local_realtime_audio_modules_when_env_is_set() {
        let Ok(dir) = std::env::var("VOXTRAL_REALTIME_LOCAL_DIR") else {
            return;
        };
        if std::env::var("VOXTRAL_REALTIME_LOAD_FULL").as_deref() != Ok("1") {
            return;
        }

        let model = crate::VoxtralRealtimeModel::load_from_dir(dir).unwrap();
        let audio_modules = model.load_audio_modules(DType::BF16, &Device::Cpu).unwrap();

        assert_eq!(audio_modules.transformer.layers.len(), 32);
        assert_eq!(audio_modules.stem.conv1.weight().dims(), &[1280, 128, 3]);
        assert_eq!(
            audio_modules.projector.linear_1.weight().dims(),
            &[3072, 5120]
        );
    }

    #[test]
    fn runs_local_realtime_audio_modules_forward_when_env_is_set() {
        let Ok(dir) = std::env::var("VOXTRAL_REALTIME_LOCAL_DIR") else {
            return;
        };
        if std::env::var("VOXTRAL_REALTIME_RUN_AUDIO_FORWARD").as_deref() != Ok("1") {
            return;
        }

        let model = crate::VoxtralRealtimeModel::load_from_dir(dir).unwrap();
        let audio_modules = model.load_audio_modules(DType::F32, &Device::Cpu).unwrap();
        let samples = vec![0.0f32; 1280];
        let mel = crate::realtime_log_mel_spectrogram(model.config(), &samples).unwrap();
        let input_features = Tensor::from_vec(
            mel.to_channel_major(),
            (1, mel.mel_bins, mel.frames),
            &Device::Cpu,
        )
        .unwrap();

        let audio_embeds = audio_modules.forward(&input_features, 0).unwrap();

        assert_eq!(audio_embeds.dims(), &[1, 1, 3072]);
    }
}
