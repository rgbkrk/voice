use std::time::{Duration, Instant};

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    embedding, linear_no_bias, rms_norm, Activation, Embedding, Linear, RmsNorm, VarBuilder,
};

use crate::{AcousticTransformerConfig, VoxtralAudioTokenizer, VoxtralConfig};

const EMPTY_AUDIO_TOKEN_ID: usize = 0;
const END_AUDIO_TOKEN_ID: usize = 1;
const AUDIO_SPECIAL_TOKEN_COUNT: usize = 2;
const TIME_EMBEDDING_THETA: f64 = 10_000.0;

pub struct VoxtralInferenceModules {
    pub embeddings: VoxtralMultimodalEmbeddings,
    pub language: VoxtralLanguageBackbone,
    pub acoustic: VoxtralAcousticTransformer,
    pub codec: VoxtralAudioTokenizer,
}

#[derive(Debug, Clone, Default)]
pub struct VoxtralModuleLoadTrace {
    pub embeddings: Duration,
    pub language: Duration,
    pub language_layers: Duration,
    pub language_norm: Duration,
    pub language_layer_count: usize,
    pub acoustic: Duration,
    pub codec: Duration,
    pub total: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct VoxtralLanguageLoadTrace {
    pub layers: Duration,
    pub norm: Duration,
    pub layer_count: usize,
    pub total: Duration,
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

#[derive(Debug, Clone)]
pub struct VoxtralLanguageCache {
    layers: Vec<VoxtralAttentionCache>,
}

#[derive(Debug, Clone, Default)]
struct VoxtralAttentionCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
}

pub struct VoxtralFeedForward {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
}

impl VoxtralInferenceModules {
    pub fn load(config: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self::load_with_trace(config, vb)?.0)
    }

    pub fn load_with_trace(
        config: &VoxtralConfig,
        vb: VarBuilder,
    ) -> Result<(Self, VoxtralModuleLoadTrace)> {
        let total_start = Instant::now();
        let embeddings_start = Instant::now();
        let embeddings = VoxtralMultimodalEmbeddings::load(config, vb.clone())?;
        let embeddings_load = embeddings_start.elapsed();

        let language_start = Instant::now();
        let (language, language_trace) =
            VoxtralLanguageBackbone::load_with_trace(config, vb.clone())?;
        let language_load = language_start.elapsed();

        let acoustic_start = Instant::now();
        let acoustic = VoxtralAcousticTransformer::load(config, vb.clone())?;
        let acoustic_load = acoustic_start.elapsed();

        let codec_start = Instant::now();
        let codec = VoxtralAudioTokenizer::load(config, vb)?;
        let codec_load = codec_start.elapsed();

        let trace = VoxtralModuleLoadTrace {
            embeddings: embeddings_load,
            language: language_load,
            language_layers: language_trace.layers,
            language_norm: language_trace.norm,
            language_layer_count: language_trace.layer_count,
            acoustic: acoustic_load,
            codec: codec_load,
            total: total_start.elapsed(),
        };

        Ok((
            Self {
                embeddings,
                language,
                acoustic,
                codec,
            },
            trace,
        ))
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

    pub fn token_embeddings(&self, token_ids: &[usize], device: &Device) -> Result<Tensor> {
        let token_ids = token_ids.iter().map(|id| *id as u32).collect::<Vec<_>>();
        let input_ids = Tensor::new(token_ids.as_slice(), device)?.reshape((1, token_ids.len()))?;
        self.tok_embeddings.forward(&input_ids)
    }

    pub fn audio_codes_embedding(&self, config: &VoxtralConfig, codes: &Tensor) -> Result<Tensor> {
        let (batch, codebooks) = codes.dims2()?;
        if codebooks != config.num_codebooks() {
            candle_core::bail!(
                "audio code frame has {codebooks} codebooks, expected {}",
                config.num_codebooks()
            );
        }

        let offsets = audio_codebook_offsets(config);
        let codes = codes.to_vec2::<u32>()?;
        let mut global_ids = Vec::with_capacity(batch * codebooks);
        for row in codes {
            for (codebook, code) in row.into_iter().enumerate() {
                global_ids.push(offsets[codebook] as u32 + code);
            }
        }

        let global_ids = Tensor::new(
            global_ids.as_slice(),
            self.audio_codebook_embeddings.embeddings().device(),
        )?
        .reshape((batch, codebooks))?;
        let embeddings = self.audio_codebook_embeddings.forward(&global_ids)?;
        embeddings.sum(1)
    }
}

impl VoxtralLanguageBackbone {
    pub fn load(config: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self::load_with_trace(config, vb)?.0)
    }

    pub fn load_with_trace(
        config: &VoxtralConfig,
        vb: VarBuilder,
    ) -> Result<(Self, VoxtralLanguageLoadTrace)> {
        let total_start = Instant::now();
        let layers_start = Instant::now();
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
        let layers_load = layers_start.elapsed();

        let norm_start = Instant::now();
        let norm = rms_norm(config.dim, config.norm_eps, vb.pp("norm"))?;
        let norm_load = norm_start.elapsed();

        let trace = VoxtralLanguageLoadTrace {
            layers: layers_load,
            norm: norm_load,
            layer_count: config.n_layers,
            total: total_start.elapsed(),
        };

        Ok((Self { layers, norm }, trace))
    }

    pub fn forward_causal(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        rope_theta: f64,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward_causal(&hidden_states, start_pos, rope_theta)?;
        }
        self.norm.forward(&hidden_states)
    }

    pub fn new_cache(&self) -> VoxtralLanguageCache {
        VoxtralLanguageCache::new(self.layers.len())
    }

    pub fn forward_causal_cached(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        rope_theta: f64,
        cache: &mut VoxtralLanguageCache,
    ) -> Result<Tensor> {
        if cache.layers.len() != self.layers.len() {
            candle_core::bail!(
                "language cache has {} layers, expected {}",
                cache.layers.len(),
                self.layers.len()
            );
        }

        let mut hidden_states = hidden_states.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward_causal_cached(
                &hidden_states,
                start_pos,
                rope_theta,
                cache.layer_mut(layer_idx)?,
            )?;
        }
        self.norm.forward(&hidden_states)
    }
}

impl VoxtralLanguageCache {
    pub fn new(layer_count: usize) -> Self {
        Self {
            layers: vec![VoxtralAttentionCache::default(); layer_count],
        }
    }

    pub fn len(&self) -> usize {
        self.layers
            .first()
            .map(VoxtralAttentionCache::len)
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn layer_mut(&mut self, layer_idx: usize) -> Result<&mut VoxtralAttentionCache> {
        self.layers.get_mut(layer_idx).ok_or_else(|| {
            candle_core::Error::Msg(format!("missing language cache for layer {layer_idx}"))
        })
    }
}

impl VoxtralAttentionCache {
    fn len(&self) -> usize {
        self.key.as_ref().map(|key| key.dims()[2]).unwrap_or(0)
    }

    fn update(&mut self, key: Tensor, value: Tensor) -> Result<(Tensor, Tensor)> {
        let key_dims = key.dims();
        let value_dims = value.dims();
        if key_dims.len() != 4 || value_dims.len() != 4 {
            candle_core::bail!(
                "attention cache expects rank-4 key/value tensors, got {:?} and {:?}",
                key_dims,
                value_dims
            );
        }
        if key_dims[0] != value_dims[0]
            || key_dims[1] != value_dims[1]
            || key_dims[2] != value_dims[2]
            || key_dims[3] != value_dims[3]
        {
            candle_core::bail!("key/value cache dims differ: {key_dims:?} vs {value_dims:?}");
        }

        match (&self.key, &self.value) {
            (None, None) => {
                self.key = Some(key.clone());
                self.value = Some(value.clone());
                Ok((key, value))
            }
            (Some(cached_key), Some(cached_value)) => {
                let cached_key_dims = cached_key.dims();
                let cached_value_dims = cached_value.dims();
                if cached_key_dims.len() != 4 || cached_value_dims.len() != 4 {
                    candle_core::bail!(
                        "cached key/value tensors must be rank 4, got {:?} and {:?}",
                        cached_key_dims,
                        cached_value_dims
                    );
                }
                if cached_key_dims[0] != key_dims[0]
                    || cached_key_dims[1] != key_dims[1]
                    || cached_key_dims[3] != key_dims[3]
                    || cached_value_dims[0] != value_dims[0]
                    || cached_value_dims[1] != value_dims[1]
                    || cached_value_dims[3] != value_dims[3]
                {
                    candle_core::bail!(
                        "new key/value dims {:?}/{:?} are incompatible with cached dims {:?}/{:?}",
                        key_dims,
                        value_dims,
                        cached_key_dims,
                        cached_value_dims
                    );
                }
                let full_key = Tensor::cat(&[cached_key, &key], 2)?;
                let full_value = Tensor::cat(&[cached_value, &value], 2)?;
                self.key = Some(full_key.clone());
                self.value = Some(full_value.clone());
                Ok((full_key, full_value))
            }
            _ => candle_core::bail!("attention cache is partially initialized"),
        }
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

    pub fn forward_attention_layers(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward_bidirectional(&hidden_states)?;
        }
        Ok(hidden_states)
    }

    /// Predict the flow-matching velocity for one acoustic frame.
    ///
    /// Mirrors vLLM-Omni's `_predict_velocity`: concatenate the noised acoustic
    /// state, sinusoidal time embedding, and projected LLM hidden state, then run
    /// the bidirectional acoustic transformer and predict acoustic codebook values
    /// from the first sequence position.
    pub fn predict_velocity(
        &self,
        x_t: &Tensor,
        llm_hidden: &Tensor,
        timestep: &Tensor,
    ) -> Result<Tensor> {
        let (batch, acoustic_codebooks) = x_t.dims2()?;
        let (hidden_batch, _) = llm_hidden.dims2()?;
        if hidden_batch != batch {
            candle_core::bail!("x_t batch {batch} does not match llm_hidden batch {hidden_batch}");
        }
        let time_emb = time_embedding(timestep, self.time_projection.weight().dims()[0])?
            .to_dtype(llm_hidden.dtype())?;
        let time_emb = linear_forward(&self.time_projection, &time_emb)?;
        let llm_hidden = linear_forward(&self.llm_projection, llm_hidden)?;
        let hidden_dim = llm_hidden.dim(1)?;
        let time_dim = time_emb.dim(1)?;
        if hidden_dim != time_dim {
            candle_core::bail!("projected llm/time dims differ: {hidden_dim} vs {time_dim}");
        }

        let inputs = Tensor::cat(
            &[
                linear_forward(
                    &self.input_projection,
                    &x_t.reshape((batch, 1, acoustic_codebooks))?,
                )?,
                time_emb.reshape((batch, 1, time_dim))?,
                llm_hidden.reshape((batch, 1, hidden_dim))?,
            ],
            1,
        )?;
        if inputs.dims() != [batch, 3, hidden_dim] {
            candle_core::bail!(
                "expected acoustic transformer input dims [{batch}, 3, {hidden_dim}], got {:?}",
                inputs.dims()
            );
        }

        let hidden = self.forward_attention_layers(&inputs)?;
        let hidden = self.norm.forward(&hidden)?;
        let acoustic_hidden = hidden.narrow(1, 0, 1)?.reshape((batch, hidden_dim))?;
        linear_forward(&self.acoustic_codebook_output, &acoustic_hidden)
    }

    /// Compute masked semantic-codebook logits for one generated frame.
    ///
    /// `[EMPTY_AUDIO]` is never sampled. `[END_AUDIO]` remains valid, and padded
    /// positions beyond the semantic codebook are masked out.
    pub fn semantic_logits(&self, config: &VoxtralConfig, llm_hidden: &Tensor) -> Result<Tensor> {
        let logits = self
            .semantic_codebook_output
            .forward(llm_hidden)?
            .to_dtype(DType::F32)?;
        let mask = semantic_logits_mask(config, logits.device())?;
        logits.broadcast_add(&mask)
    }

    /// Predict one frame of semantic + acoustic audio codebook IDs.
    ///
    /// This mirrors vLLM-Omni's `forward`/`decode_one_frame` boundary but keeps
    /// the stochastic policy outside this module: callers provide the initial
    /// acoustic noise and the Euler timestep schedule.
    pub fn predict_frame_codes_from_noise(
        &self,
        config: &VoxtralConfig,
        llm_hidden: &Tensor,
        initial_noise: &Tensor,
        timesteps: &[f32],
        cfg_alpha: f32,
    ) -> Result<Tensor> {
        if timesteps.len() < 2 {
            candle_core::bail!("expected at least two timesteps, got {}", timesteps.len());
        }

        let semantic_code = self
            .semantic_logits(config, llm_hidden)?
            .argmax_keepdim(D::Minus1)?;
        let acoustic_codes = self.decode_acoustic_codes_from_noise(
            config,
            &semantic_code.squeeze(1)?,
            llm_hidden,
            initial_noise,
            timesteps,
            cfg_alpha,
        )?;

        Tensor::cat(&[semantic_code.to_dtype(DType::U32)?, acoustic_codes], 1)
    }

    fn decode_acoustic_codes_from_noise(
        &self,
        config: &VoxtralConfig,
        semantic_code: &Tensor,
        llm_hidden: &Tensor,
        initial_noise: &Tensor,
        timesteps: &[f32],
        cfg_alpha: f32,
    ) -> Result<Tensor> {
        let audio_model = &config.multimodal.audio_model_args;
        let (batch, acoustic_codebooks) = initial_noise.dims2()?;
        let (hidden_batch, hidden_dim) = llm_hidden.dims2()?;
        if batch != hidden_batch {
            candle_core::bail!(
                "initial_noise batch {} does not match llm_hidden batch {}",
                batch,
                hidden_batch
            );
        }
        if acoustic_codebooks != audio_model.n_acoustic_codebook {
            candle_core::bail!(
                "initial_noise has {} acoustic codebooks, expected {}",
                acoustic_codebooks,
                audio_model.n_acoustic_codebook
            );
        }
        if hidden_dim != audio_model.acoustic_transformer_args.input_dim {
            candle_core::bail!(
                "llm_hidden dim {} does not match acoustic input dim {}",
                hidden_dim,
                audio_model.acoustic_transformer_args.input_dim
            );
        }

        let device = llm_hidden.device();
        let dtype = llm_hidden.dtype();
        let llm_hidden_zero = Tensor::zeros_like(llm_hidden)?;
        let mut sampled = initial_noise.to_dtype(dtype)?;

        for step in timesteps.windows(2) {
            let t = step[0];
            let dt = step[1] - step[0];
            let t_batched = vec![t; batch * 2];
            let t_batched = Tensor::new(t_batched.as_slice(), device)?.to_dtype(dtype)?;
            let x_batched = Tensor::cat(&[&sampled, &sampled], 0)?;
            let llm_batched = Tensor::cat(&[llm_hidden, &llm_hidden_zero], 0)?;

            let velocity = self.predict_velocity(&x_batched, &llm_batched, &t_batched)?;
            let conditional = velocity.narrow(0, 0, batch)?;
            let unconditional = velocity.narrow(0, batch, batch)?;
            let guided =
                ((conditional * cfg_alpha as f64)? + (unconditional * (1.0 - cfg_alpha) as f64)?)?;
            sampled = (sampled + (guided * dt as f64)?)?;
        }

        let scaled = (((sampled.clamp(-1.0f32, 1.0f32)? + 1.0)? * 0.5)?
            * (audio_model.acoustic_codebook_size - 1) as f64)?;
        let shifted_codes =
            (scaled.round()? + AUDIO_SPECIAL_TOKEN_COUNT as f64)?.to_dtype(DType::U32)?;
        let empty_codes = Tensor::new(AUDIO_SPECIAL_TOKEN_COUNT as u32, device)?
            .broadcast_as(shifted_codes.shape())?;
        let should_decode = semantic_code
            .ne(END_AUDIO_TOKEN_ID as u32)?
            .reshape((batch, 1))?
            .broadcast_as(shifted_codes.shape())?;

        should_decode.where_cond(&shifted_codes, &empty_codes)
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

    pub fn forward_bidirectional(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let attention_input = self.attention_norm.forward(hidden_states)?;
        let attention_output = self.attention.forward_bidirectional(&attention_input)?;
        let hidden_states = (attention_output + residual)?;

        let residual = hidden_states.clone();
        let ffn_input = self.ffn_norm.forward(&hidden_states)?;
        let ffn_output = self.feed_forward.forward(&ffn_input)?;
        ffn_output + residual
    }

    pub fn forward_causal(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        rope_theta: f64,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let attention_input = self.attention_norm.forward(hidden_states)?;
        let attention_output =
            self.attention
                .forward_causal(&attention_input, start_pos, rope_theta)?;
        let hidden_states = (attention_output + residual)?;

        let residual = hidden_states.clone();
        let ffn_input = self.ffn_norm.forward(&hidden_states)?;
        let ffn_output = self.feed_forward.forward(&ffn_input)?;
        ffn_output + residual
    }

    fn forward_causal_cached(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        rope_theta: f64,
        cache: &mut VoxtralAttentionCache,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let attention_input = self.attention_norm.forward(hidden_states)?;
        let attention_output =
            self.attention
                .forward_causal_cached(&attention_input, start_pos, rope_theta, cache)?;
        let hidden_states = (attention_output + residual)?;

        let residual = hidden_states.clone();
        let ffn_input = self.ffn_norm.forward(&hidden_states)?;
        let ffn_output = self.feed_forward.forward(&ffn_input)?;
        ffn_output + residual
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

    pub fn forward_bidirectional(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _dim) = hidden_states.dims3()?;
        let repeat = self.n_heads / self.n_kv_heads;

        let query = self
            .wq
            .forward_compat(hidden_states)?
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key = self.wk.forward_compat(hidden_states)?.reshape((
            batch,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
        ))?;
        let value = self.wv.forward_compat(hidden_states)?.reshape((
            batch,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
        ))?;
        let key = repeat_kv(&key, repeat)?.transpose(1, 2)?.contiguous()?;
        let value = repeat_kv(&value, repeat)?.transpose(1, 2)?.contiguous()?;

        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let scores = (query.matmul(&key.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let output = weights.matmul(&value)?.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ))?;

        self.wo.forward_compat(&output)
    }

    pub fn forward_causal(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        rope_theta: f64,
    ) -> Result<Tensor> {
        let (batch, seq_len, _dim) = hidden_states.dims3()?;
        let repeat = self.n_heads / self.n_kv_heads;

        let query = self
            .wq
            .forward_compat(hidden_states)?
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key = self
            .wk
            .forward_compat(hidden_states)?
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value = self.wv.forward_compat(hidden_states)?.reshape((
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
        let scores = (query.matmul(&key.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let mask = causal_mask(seq_len, scores.dtype(), scores.device())?;
        let scores = scores.broadcast_add(&mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let output = weights.matmul(&value)?.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ))?;

        self.wo.forward_compat(&output)
    }

    fn forward_causal_cached(
        &self,
        hidden_states: &Tensor,
        start_pos: usize,
        rope_theta: f64,
        cache: &mut VoxtralAttentionCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, _dim) = hidden_states.dims3()?;
        if cache.len() != start_pos {
            candle_core::bail!(
                "attention cache length {} does not match start_pos {start_pos}",
                cache.len()
            );
        }
        let repeat = self.n_heads / self.n_kv_heads;

        let query = self
            .wq
            .forward_compat(hidden_states)?
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key = self
            .wk
            .forward_compat(hidden_states)?
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value = self
            .wv
            .forward_compat(hidden_states)?
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

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
        let (key, value) = cache.update(key, value)?;
        let key_len = key.dim(2)?;
        let key = repeat_kv_heads(&key, repeat)?;
        let value = repeat_kv_heads(&value, repeat)?;

        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let scores = (query.matmul(&key.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let mask =
            causal_mask_with_offset(seq_len, key_len, start_pos, scores.dtype(), scores.device())?;
        let scores = scores.broadcast_add(&mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let output = weights.matmul(&value)?.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ))?;

        self.wo.forward_compat(&output)
    }
}

impl VoxtralFeedForward {
    pub fn load(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w1 = linear_no_bias(dim, hidden_dim, vb.pp("w1"))?;
        let w2 = linear_no_bias(hidden_dim, dim, vb.pp("w2"))?;
        let w3 = linear_no_bias(dim, hidden_dim, vb.pp("w3"))?;

        Ok(Self { w1, w2, w3 })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward_compat(hidden_states)?;
        let gate = Activation::Silu.forward(&gate)?;
        let up = self.w3.forward_compat(hidden_states)?;
        let hidden_states = gate.broadcast_mul(&up)?;
        self.w2.forward_compat(&hidden_states)
    }
}

trait LinearCompat {
    fn forward_compat(&self, xs: &Tensor) -> Result<Tensor>;
}

impl LinearCompat for Linear {
    fn forward_compat(&self, xs: &Tensor) -> Result<Tensor> {
        linear_forward(self, xs)
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

fn audio_codebook_vocab_size(config: &VoxtralConfig) -> usize {
    let audio_model = &config.multimodal.audio_model_args;
    round_up_to_multiple(
        audio_model.semantic_codebook_size
            + 2
            + (audio_model.acoustic_codebook_size + 2) * audio_model.n_acoustic_codebook,
        128,
    )
}

fn audio_codebook_offsets(config: &VoxtralConfig) -> Vec<usize> {
    let audio_model = &config.multimodal.audio_model_args;
    let semantic_size = audio_model.semantic_codebook_size + AUDIO_SPECIAL_TOKEN_COUNT;
    let acoustic_size = audio_model.acoustic_codebook_size + AUDIO_SPECIAL_TOKEN_COUNT;
    let mut offsets = Vec::with_capacity(config.num_codebooks());
    offsets.push(0);
    for idx in 1..config.num_codebooks() {
        offsets.push(semantic_size + (idx - 1) * acoustic_size);
    }
    offsets
}

fn semantic_codebook_output_size(config: &VoxtralConfig) -> usize {
    let audio_model = &config.multimodal.audio_model_args;
    round_up_to_multiple(audio_model.semantic_codebook_size + 2, 128)
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    multiple * value.div_ceil(multiple)
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
    dtype: DType,
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

fn causal_mask(seq_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(seq_len * seq_len);
    for query_pos in 0..seq_len {
        for key_pos in 0..seq_len {
            if key_pos <= query_pos {
                values.push(0.0);
            } else {
                values.push(f32::NEG_INFINITY);
            }
        }
    }
    Tensor::from_vec(values, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

fn causal_mask_with_offset(
    query_len: usize,
    key_len: usize,
    start_pos: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(query_len * key_len);
    for query_pos in 0..query_len {
        let absolute_query_pos = start_pos + query_pos;
        for key_pos in 0..key_len {
            if key_pos <= absolute_query_pos {
                values.push(0.0);
            } else {
                values.push(f32::NEG_INFINITY);
            }
        }
    }
    Tensor::from_vec(values, (1, 1, query_len, key_len), device)?.to_dtype(dtype)
}

fn time_embedding(timestep: &Tensor, dim: usize) -> Result<Tensor> {
    let timestep = match timestep.dims() {
        [_batch] => timestep.unsqueeze(1)?,
        [_batch, 1] => timestep.clone(),
        _ => candle_core::bail!(
            "expected timestep shape [batch] or [batch, 1], got {:?}",
            timestep.dims()
        ),
    };
    let half_dim = dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|idx| (-TIME_EMBEDDING_THETA.ln() * idx as f64 / half_dim as f64).exp() as f32)
        .collect();
    let inv_freq =
        Tensor::new(inv_freq.as_slice(), timestep.device())?.to_dtype(timestep.dtype())?;
    let angles = timestep.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
    Tensor::cat(&[angles.cos()?, angles.sin()?], D::Minus1)
}

fn semantic_logits_mask(config: &VoxtralConfig, device: &candle_core::Device) -> Result<Tensor> {
    debug_assert_eq!(END_AUDIO_TOKEN_ID, AUDIO_SPECIAL_TOKEN_COUNT - 1);
    let valid_len =
        AUDIO_SPECIAL_TOKEN_COUNT + config.multimodal.audio_model_args.semantic_codebook_size;
    let output_len = semantic_codebook_output_size(config);
    let mask: Vec<f32> = (0..output_len)
        .map(|idx| {
            if idx == EMPTY_AUDIO_TOKEN_ID || idx >= valid_len {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();
    Tensor::new(mask.as_slice(), device)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    use super::*;

    pub(crate) fn tiny_config_json() -> &'static str {
        r#"{
              "dim": 8,
              "n_layers": 2,
              "head_dim": 4,
              "hidden_dim": 16,
              "n_heads": 2,
              "n_kv_heads": 1,
              "rope_theta": 1000000.0,
              "norm_eps": 1e-05,
              "vocab_size": 64,
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
            }"#
    }

    pub(crate) fn tiny_config() -> VoxtralConfig {
        VoxtralConfig::from_json_str(tiny_config_json()).unwrap()
    }

    #[test]
    fn loads_tiny_inference_modules_from_varbuilder() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

        let (modules, trace) = VoxtralInferenceModules::load_with_trace(&config, vb).unwrap();

        assert!(trace.total >= trace.embeddings);
        assert!(trace.total >= trace.language);
        assert!(trace.language >= trace.language_layers);
        assert!(trace.language >= trace.language_norm);
        assert_eq!(trace.language_layer_count, config.n_layers);
        assert!(trace.total >= trace.acoustic);
        assert!(trace.total >= trace.codec);

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

    #[test]
    fn runs_tiny_language_causal_forward_path() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let modules = VoxtralInferenceModules::load(&config, vb).unwrap();
        let device = Device::Cpu;

        let token_embeddings = modules
            .embeddings
            .token_embeddings(&[1, 2, 3], &device)
            .unwrap();
        let hidden = modules
            .language
            .forward_causal(&token_embeddings, 0, config.rope_theta)
            .unwrap();

        assert_eq!(token_embeddings.dims(), &[1, 3, 8]);
        assert_eq!(hidden.dims(), &[1, 3, 8]);
    }

    #[test]
    fn cached_language_forward_matches_full_prefix() {
        let config = tiny_config();
        let device = Device::Cpu;
        let tensors = deterministic_language_tensors(&config, &device);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let language = VoxtralLanguageBackbone::load(&config, vb).unwrap();
        let input_values = (0..4 * config.dim)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.03)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input_values, (1, 4, config.dim), &device).unwrap();

        let full = language
            .forward_causal(&input, 0, config.rope_theta)
            .unwrap();
        let mut cache = language.new_cache();
        let prefill_input = input.narrow(1, 0, 3).unwrap();
        let step_input = input.narrow(1, 3, 1).unwrap();
        let prefill = language
            .forward_causal_cached(&prefill_input, 0, config.rope_theta, &mut cache)
            .unwrap();
        let step = language
            .forward_causal_cached(&step_input, 3, config.rope_theta, &mut cache)
            .unwrap();

        assert_eq!(cache.len(), 4);
        assert_eq!(prefill.dims(), &[1, 3, config.dim]);
        assert_eq!(step.dims(), &[1, 1, config.dim]);
        assert!(
            max_abs_diff(&full.narrow(1, 0, 3).unwrap(), &prefill) < 1e-4,
            "cached prefill diverged from full-prefix forward"
        );
        assert!(
            max_abs_diff(&full.narrow(1, 3, 1).unwrap(), &step) < 1e-4,
            "cached decode step diverged from full-prefix forward"
        );
    }

    #[test]
    fn embeds_tiny_audio_codes_for_next_language_step() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let modules = VoxtralInferenceModules::load(&config, vb).unwrap();
        let device = Device::Cpu;

        let codes = Tensor::new(&[2u32, 2, 3], &device)
            .unwrap()
            .reshape((1, config.num_codebooks()))
            .unwrap();
        let audio_embedding = modules
            .embeddings
            .audio_codes_embedding(&config, &codes)
            .unwrap();

        assert_eq!(audio_embedding.dims(), &[1, 8]);
    }

    #[test]
    fn runs_tiny_acoustic_transformer_velocity_path() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let modules = VoxtralInferenceModules::load(&config, vb).unwrap();
        let device = Device::Cpu;

        let x_t = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let llm_hidden = Tensor::zeros((2, 8), DType::F32, &device).unwrap();
        let timestep = Tensor::new(&[0.0f32, 0.5], &device).unwrap();

        let velocity = modules
            .acoustic
            .predict_velocity(&x_t, &llm_hidden, &timestep)
            .unwrap();

        assert_eq!(velocity.dims(), &[2, 2]);
    }

    #[test]
    fn masks_tiny_semantic_logits_like_reference() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let modules = VoxtralInferenceModules::load(&config, vb).unwrap();
        let device = Device::Cpu;

        let llm_hidden = Tensor::zeros((1, 8), DType::F32, &device).unwrap();
        let logits = modules
            .acoustic
            .semantic_logits(&config, &llm_hidden)
            .unwrap();
        let logits = logits.to_vec2::<f32>().unwrap();
        let row = &logits[0];

        assert_eq!(row.len(), 128);
        assert!(
            row[EMPTY_AUDIO_TOKEN_ID].is_infinite() && row[EMPTY_AUDIO_TOKEN_ID].is_sign_negative()
        );
        assert_eq!(row[END_AUDIO_TOKEN_ID], 0.0);
        assert_eq!(
            row[AUDIO_SPECIAL_TOKEN_COUNT
                + config.multimodal.audio_model_args.semantic_codebook_size
                - 1],
            0.0
        );
        assert!(row[AUDIO_SPECIAL_TOKEN_COUNT
            + config.multimodal.audio_model_args.semantic_codebook_size]
            .is_infinite());
    }

    #[test]
    fn predicts_tiny_frame_codes_from_supplied_noise() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let modules = VoxtralInferenceModules::load(&config, vb).unwrap();
        let device = Device::Cpu;

        let llm_hidden = Tensor::zeros((1, 8), DType::F32, &device).unwrap();
        let initial_noise = Tensor::zeros((1, 2), DType::F32, &device).unwrap();
        let frame_codes = modules
            .acoustic
            .predict_frame_codes_from_noise(&config, &llm_hidden, &initial_noise, &[0.0, 1.0], 1.2)
            .unwrap();

        assert_eq!(frame_codes.dims(), &[1, 3]);
        assert_eq!(frame_codes.to_vec2::<u32>().unwrap(), vec![vec![1, 2, 2]]);
    }

    #[test]
    fn reshapes_acoustic_frame_as_single_sequence_position() {
        let device = Device::Cpu;
        let x_t = Tensor::zeros((2, 36), DType::F32, &device).unwrap();
        let (batch, acoustic_codebooks) = x_t.dims2().unwrap();

        let acoustic_input = x_t.reshape((batch, 1, acoustic_codebooks)).unwrap();

        assert_eq!(acoustic_input.dims(), &[2, 1, 36]);
    }

    #[test]
    fn runs_local_acoustic_forward_when_env_is_set() {
        let Ok(dir) = std::env::var("VOXTRAL_LOCAL_DIR") else {
            return;
        };
        if std::env::var("VOXTRAL_FORWARD_SMOKE").as_deref() != Ok("1") {
            return;
        }

        let device = Device::Cpu;
        let model = crate::VoxtralModel::load_from_dir(dir).unwrap();
        let vb = model.var_builder(DType::F32, &device).unwrap();
        let acoustic_module = VoxtralAcousticTransformer::load(model.config(), vb).unwrap();
        let audio_model = &model.config().multimodal.audio_model_args;
        let acoustic = &audio_model.acoustic_transformer_args;

        let x_t = Tensor::zeros((1, audio_model.n_acoustic_codebook), DType::F32, &device).unwrap();
        let llm_hidden = Tensor::zeros((1, acoustic.input_dim), DType::F32, &device).unwrap();
        let timestep = Tensor::new(&[0.0f32], &device).unwrap();

        let velocity = acoustic_module
            .predict_velocity(&x_t, &llm_hidden, &timestep)
            .unwrap();
        let semantic_logits = acoustic_module
            .semantic_logits(model.config(), &llm_hidden)
            .unwrap();
        let frame_codes = acoustic_module
            .predict_frame_codes_from_noise(model.config(), &llm_hidden, &x_t, &[0.0, 1.0], 1.2)
            .unwrap();

        assert_eq!(velocity.dims(), &[1, audio_model.n_acoustic_codebook]);
        assert_eq!(
            semantic_logits.dims(),
            &[1, semantic_codebook_output_size(model.config())]
        );
        assert_eq!(frame_codes.dims(), &[1, model.config().num_codebooks()]);
    }

    fn deterministic_language_tensors(
        config: &VoxtralConfig,
        device: &Device,
    ) -> HashMap<String, Tensor> {
        let mut tensors = HashMap::new();
        for layer_idx in 0..config.n_layers {
            let prefix = format!("layers.{layer_idx}");
            insert_test_tensor(
                &mut tensors,
                format!("{prefix}.attention.wq.weight"),
                &[config.n_heads * config.head_dim, config.dim],
                device,
            );
            insert_test_tensor(
                &mut tensors,
                format!("{prefix}.attention.wk.weight"),
                &[config.n_kv_heads * config.head_dim, config.dim],
                device,
            );
            insert_test_tensor(
                &mut tensors,
                format!("{prefix}.attention.wv.weight"),
                &[config.n_kv_heads * config.head_dim, config.dim],
                device,
            );
            insert_test_tensor(
                &mut tensors,
                format!("{prefix}.attention.wo.weight"),
                &[config.dim, config.n_heads * config.head_dim],
                device,
            );
            insert_test_tensor(
                &mut tensors,
                format!("{prefix}.feed_forward.w1.weight"),
                &[config.hidden_dim, config.dim],
                device,
            );
            insert_test_tensor(
                &mut tensors,
                format!("{prefix}.feed_forward.w2.weight"),
                &[config.dim, config.hidden_dim],
                device,
            );
            insert_test_tensor(
                &mut tensors,
                format!("{prefix}.feed_forward.w3.weight"),
                &[config.hidden_dim, config.dim],
                device,
            );
            insert_norm_tensor(
                &mut tensors,
                format!("{prefix}.attention_norm.weight"),
                config.dim,
                device,
            );
            insert_norm_tensor(
                &mut tensors,
                format!("{prefix}.ffn_norm.weight"),
                config.dim,
                device,
            );
        }
        insert_norm_tensor(&mut tensors, "norm.weight".to_string(), config.dim, device);
        tensors
    }

    fn insert_test_tensor(
        tensors: &mut HashMap<String, Tensor>,
        name: String,
        dims: &[usize],
        device: &Device,
    ) {
        let len = dims.iter().product::<usize>();
        let seed = name.bytes().fold(0usize, |acc, byte| {
            acc.wrapping_mul(31).wrapping_add(byte as usize)
        });
        let values = (0..len)
            .map(|idx| (((idx + seed) % 29) as f32 - 14.0) * 0.015)
            .collect::<Vec<_>>();
        tensors.insert(
            name,
            Tensor::from_vec(values, dims.to_vec(), device).unwrap(),
        );
    }

    fn insert_norm_tensor(
        tensors: &mut HashMap<String, Tensor>,
        name: String,
        dim: usize,
        device: &Device,
    ) {
        tensors.insert(
            name,
            Tensor::from_vec(vec![1.0f32; dim], (dim,), device).unwrap(),
        );
    }

    fn max_abs_diff(left: &Tensor, right: &Tensor) -> f32 {
        left.broadcast_sub(right)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .fold(0.0, f32::max)
    }
}
