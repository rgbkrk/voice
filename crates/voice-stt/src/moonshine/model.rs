//! Moonshine speech-to-text model architecture.
//!
//! Ported from mlx-audio's `mlx_audio/stt/models/moonshine/moonshine.py`.
//!
//! Moonshine is a compact encoder-decoder transformer designed for on-device
//! speech recognition. Key architectural features:
//!
//! - **Learned audio frontend**: Three Conv1d layers with progressive stride
//!   (64×3×2 = 384× total downsampling) replace mel spectrograms.
//! - **Partial RoPE**: Only a fraction of head dimensions receive rotary
//!   positional encoding; the rest pass through unmodified.
//! - **SwiGLU decoder MLP**: Gated SiLU activation in the decoder for better
//!   gradient flow.
//! - **Tied embeddings**: Output projection reuses the token embedding matrix.

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{self, Embedding, GroupNorm, LayerNorm, LayerNormBuilder, Linear, LinearBuilder};
use mlx_rs::ops::indexing::{IndexOp, IntoStrideBy};
use mlx_rs::ops::{concatenate_axis, expand_dims, stack_axis};
use mlx_rs::Array;

use super::config::MoonshineConfig;

// ---------------------------------------------------------------------------
// Rotary Positional Embedding
// ---------------------------------------------------------------------------

/// Precomputes inverse frequency bands for RoPE.
///
/// At call time, produces `(cos, sin)` tensors for the given position IDs.
#[derive(Debug)]
pub struct MoonshineRotaryEmbedding {
    inv_freq: Array,
    dim: usize,
}

impl MoonshineRotaryEmbedding {
    pub fn new(dim: usize, _max_position_embeddings: usize, base: f32) -> Self {
        // inv_freq = 1.0 / (base^(arange(0, dim, 2) / dim))
        let half = (dim / 2) as i32;
        let arange = Array::from_iter((0..dim).step_by(2).map(|i| i as f32 / dim as f32), &[half]);
        let base_arr = Array::from_f32(base);
        let inv_freq = Array::from_f32(1.0) / base_arr.power(&arange).unwrap();
        Self { inv_freq, dim }
    }

    /// Compute (cos, sin) embeddings for the given position IDs.
    ///
    /// - `position_ids`: shape `[B, T]`
    /// - Returns: `(cos, sin)` each shape `[B, T, rotary_ndims]`
    pub fn forward(&self, position_ids: &Array) -> Result<(Array, Array), Exception> {
        // position_ids: [B, T] -> [B, T, 1]
        let pos = position_ids.reshape(&[position_ids.shape()[0], position_ids.shape()[1], 1])?;
        let pos = pos.as_dtype(mlx_rs::Dtype::Float32)?;

        // inv_freq: [dim/2] -> [1, 1, dim/2]
        let inv = self.inv_freq.reshape(&[1, 1, (self.dim / 2) as i32])?;

        // freqs: [B, T, dim/2]
        let freqs = pos.multiply(&inv)?;

        // emb: [B, T, dim]  (concatenate freqs with itself)
        let emb = concatenate_axis(&[&freqs, &freqs], -1)?;

        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok((cos, sin))
    }
}

// ---------------------------------------------------------------------------
// RoPE application helpers
// ---------------------------------------------------------------------------

/// Rotate interleaved pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
fn rotate_half(x: &Array) -> Result<Array, Exception> {
    // x1 = x[..., 0::2], x2 = x[..., 1::2]
    let x1 = x.index((.., .., .., (0i32..).stride_by(2)));
    let x2 = x.index((.., .., .., (1i32..).stride_by(2)));

    // rotated = stack([-x2, x1], axis=-1) then reshape to x.shape
    let neg_x2 = x2.negative()?;
    let stacked = stack_axis(&[&neg_x2, &x1], -1)?;
    stacked.reshape(x.shape())
}

/// Apply partial rotary positional embeddings to Q and K.
///
/// Only the first `rotary_dim` dimensions are rotated; the rest pass through.
fn apply_rotary_pos_emb(
    q: &Array,
    k: &Array,
    cos: &Array,
    sin: &Array,
    rotary_dim: usize,
) -> Result<(Array, Array), Exception> {
    // cos/sin: [B, T, rotary_dim] -> [B, 1, T, rotary_dim] (unsqueeze head dim)
    let cos = expand_dims(cos, 1)?;
    let sin = expand_dims(sin, 1)?;

    // The cos/sin are [B, 1, T, dim] from concatenation.
    // We need to expand them to cover interleaved pairs.
    // half = rotary_dim / 2, then repeat each value twice to match interleaved layout.
    let half = rotary_dim / 2;
    let cos_half = cos.index((.., .., .., ..half as i32));
    let sin_half = sin.index((.., .., .., ..half as i32));
    // repeat(2, axis=-1) to interleave
    let cos_exp = Array::repeat_axis::<f32>(cos_half, 2, -1)?;
    let sin_exp = Array::repeat_axis::<f32>(sin_half, 2, -1)?;

    let rd = rotary_dim as i32;

    // Split Q into rotary and pass-through portions
    let q_rot = q.index((.., .., .., ..rd));
    let q_pass = q.index((.., .., .., rd..));
    let k_rot = k.index((.., .., .., ..rd));
    let k_pass = k.index((.., .., .., rd..));

    // Apply rotation: x_embed = x * cos + rotate_half(x) * sin
    let q_embed = q_rot
        .multiply(&cos_exp)?
        .add(&rotate_half(&q_rot)?.multiply(&sin_exp)?)?;
    let k_embed = k_rot
        .multiply(&cos_exp)?
        .add(&rotate_half(&k_rot)?.multiply(&sin_exp)?)?;

    // Concatenate rotary and pass-through parts
    let q_out = concatenate_axis(&[&q_embed, &q_pass], -1)?;
    let k_out = concatenate_axis(&[&k_embed, &k_pass], -1)?;

    Ok((q_out, k_out))
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Multi-head attention with partial RoPE and optional GQA.
///
/// Used in both encoder (non-causal, self-attention only) and decoder
/// (causal self-attention + non-causal cross-attention).
#[derive(Debug, ModuleParameters)]
pub struct MoonshineAttention {
    #[param]
    pub q_proj: Linear,
    #[param]
    pub k_proj: Linear,
    #[param]
    pub v_proj: Linear,
    #[param]
    pub o_proj: Linear,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_kv_groups: usize,
    pub is_causal: bool,
    pub scale: f32,
    pub rotary_ndims: usize,
    rotary_emb: MoonshineRotaryEmbedding,
}

impl MoonshineAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        bias: bool,
        is_causal: bool,
        partial_rotary_factor: f32,
        max_position_embeddings: usize,
        rope_theta: f32,
    ) -> Result<Self, Exception> {
        let head_dim = hidden_size / num_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let scale = (head_dim as f32).powf(-0.5);

        let raw_rotary = (head_dim as f32 * partial_rotary_factor) as usize;
        let rotary_ndims = raw_rotary - (raw_rotary % 2);

        let q_proj = LinearBuilder::new(hidden_size as i32, (num_heads * head_dim) as i32)
            .bias(bias)
            .build()?;
        let k_proj = LinearBuilder::new(hidden_size as i32, (num_kv_heads * head_dim) as i32)
            .bias(bias)
            .build()?;
        let v_proj = LinearBuilder::new(hidden_size as i32, (num_kv_heads * head_dim) as i32)
            .bias(bias)
            .build()?;
        let o_proj = LinearBuilder::new((num_heads * head_dim) as i32, hidden_size as i32)
            .bias(false)
            .build()?;

        let rotary_emb =
            MoonshineRotaryEmbedding::new(rotary_ndims, max_position_embeddings, rope_theta);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            num_kv_groups,
            is_causal,
            scale,
            rotary_ndims,
            rotary_emb,
        })
    }

    /// Forward pass.
    ///
    /// - `x`: input hidden states `[B, T, hidden_size]`
    /// - `encoder_hidden_states`: if `Some`, this is cross-attention (K/V come from encoder)
    /// - `cache`: optional `(K, V)` from previous steps
    /// - `position_ids`: optional position IDs for RoPE; auto-computed if `None`
    ///
    /// Returns `(output, (new_K, new_V))`.
    pub fn forward(
        &mut self,
        x: &Array,
        encoder_hidden_states: Option<&Array>,
        cache: Option<&(Array, Array)>,
        position_ids: Option<&Array>,
    ) -> Result<(Array, (Array, Array)), Exception> {
        let b = x.shape()[0];
        let t = x.shape()[1];
        let is_cross_attention = encoder_hidden_states.is_some();

        let q = self.q_proj.forward(x)?;
        let (k, v) = if let Some(enc) = encoder_hidden_states {
            (self.k_proj.forward(enc)?, self.v_proj.forward(enc)?)
        } else {
            (self.k_proj.forward(x)?, self.v_proj.forward(x)?)
        };

        let hd = self.head_dim as i32;
        let nh = self.num_heads as i32;
        let nkv = self.num_kv_heads as i32;

        // Reshape to [B, T, num_heads, head_dim] then transpose to [B, num_heads, T, head_dim]
        let q = q.reshape(&[b, t, nh, hd])?.transpose_axes(&[0, 2, 1, 3])?;
        let s = k.shape()[1];
        let mut k = k.reshape(&[b, s, nkv, hd])?.transpose_axes(&[0, 2, 1, 3])?;
        let mut v = v.reshape(&[b, s, nkv, hd])?.transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE to self-attention only (not cross-attention)
        let q = if !is_cross_attention {
            let pos = if let Some(p) = position_ids {
                p.clone()
            } else {
                let offset = if let Some(c) = cache {
                    c.0.shape()[2]
                } else {
                    0
                };
                let pos_range = Array::from_iter(offset..offset + t, &[t]);
                pos_range.reshape(&[1, t])?
            };

            let (cos, sin) = self.rotary_emb.forward(&pos)?;
            let (q_rot, k_rot) = apply_rotary_pos_emb(&q, &k, &cos, &sin, self.rotary_ndims)?;
            k = k_rot;
            q_rot
        } else {
            q
        };

        // Apply KV cache
        if let Some(c) = cache {
            if is_cross_attention {
                // Cross-attention: reuse frozen K/V from first step
                k = c.0.clone();
                v = c.1.clone();
            } else {
                // Self-attention: concatenate with previous K/V
                k = concatenate_axis(&[&c.0, &k], 2)?;
                v = concatenate_axis(&[&c.1, &v], 2)?;
            }
        }

        // Expand KV for GQA if needed
        let (k_exp, v_exp) = if self.num_kv_groups > 1 {
            (
                Array::repeat_axis::<f32>(k.clone(), self.num_kv_groups as i32, 1)?,
                Array::repeat_axis::<f32>(v.clone(), self.num_kv_groups as i32, 1)?,
            )
        } else {
            (k.clone(), v.clone())
        };

        // Causal mask for decoder self-attention
        let mask = if self.is_causal && t > 1 {
            let causal = nn::MultiHeadAttention::create_additive_causal_mask::<f32>(t)?;
            let klen = k_exp.shape()[2];
            if klen > t {
                // Prefix zeros for cached keys
                let prefix = Array::zeros::<f32>(&[t, klen - t])?;
                Some(concatenate_axis(&[&prefix, &causal], 1)?)
            } else {
                Some(causal)
            }
        } else {
            None
        };

        // Scaled dot-product attention
        let o = if let Some(ref m) = mask {
            mlx_rs::fast::scaled_dot_product_attention(&q, &k_exp, &v_exp, self.scale, m)?
        } else {
            mlx_rs::fast::scaled_dot_product_attention(
                &q,
                &k_exp,
                &v_exp,
                self.scale,
                None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            )?
        };

        // Reshape back: [B, num_heads, T, head_dim] -> [B, T, num_heads * head_dim]
        let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, t, -1])?;

        let out = self.o_proj.forward(&o)?;
        Ok((out, (k, v)))
    }
}

// ---------------------------------------------------------------------------
// MLPs
// ---------------------------------------------------------------------------

/// Encoder MLP: simple two-layer feedforward with GELU activation.
///
/// `dim → intermediate_size → dim`
#[derive(Debug, ModuleParameters)]
pub struct MoonshineEncoderMLP {
    #[param]
    pub fc1: Linear,
    #[param]
    pub fc2: Linear,
    pub use_gelu: bool,
}

impl MoonshineEncoderMLP {
    pub fn new(hidden_size: usize, intermediate_size: usize, act: &str) -> Result<Self, Exception> {
        let fc1 = LinearBuilder::new(hidden_size as i32, intermediate_size as i32).build()?;
        let fc2 = LinearBuilder::new(intermediate_size as i32, hidden_size as i32).build()?;
        Ok(Self {
            fc1,
            fc2,
            use_gelu: act == "gelu",
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = self.fc1.forward(x)?;
        let h = if self.use_gelu {
            nn::gelu(&h)?
        } else {
            nn::silu(&h)?
        };
        self.fc2.forward(&h)
    }
}

/// Decoder MLP: SwiGLU (gated SiLU) activation.
///
/// `fc1` projects to `2 * intermediate_size`, splits into value and gate,
/// applies `silu(gate) * value`, then `fc2` projects back.
#[derive(Debug, ModuleParameters)]
pub struct MoonshineDecoderMLP {
    #[param]
    pub fc1: Linear,
    #[param]
    pub fc2: Linear,
}

impl MoonshineDecoderMLP {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Result<Self, Exception> {
        let fc1 = LinearBuilder::new(hidden_size as i32, (2 * intermediate_size) as i32).build()?;
        let fc2 = LinearBuilder::new(intermediate_size as i32, hidden_size as i32).build()?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = self.fc1.forward(x)?;
        let mid = h.shape().last().copied().unwrap_or(0);
        let half = mid / 2;

        // Split into value and gate halves
        let value = h.index((.., .., ..half));
        let gate = h.index((.., .., half..));

        // SwiGLU: silu(gate) * value
        let activated = nn::silu(&gate)?.multiply(&value)?;
        self.fc2.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// Encoder layer
// ---------------------------------------------------------------------------

/// A single Moonshine encoder transformer layer.
///
/// Pre-LN architecture: LayerNorm → Self-Attention → residual → LayerNorm → MLP → residual
#[derive(Debug, ModuleParameters)]
pub struct MoonshineEncoderLayer {
    #[param]
    pub self_attn: MoonshineAttention,
    #[param]
    pub mlp: MoonshineEncoderMLP,
    #[param]
    pub input_layernorm: LayerNorm,
    #[param]
    pub post_attention_layernorm: LayerNorm,
}

impl MoonshineEncoderLayer {
    pub fn new(config: &MoonshineConfig) -> Result<Self, Exception> {
        let self_attn = MoonshineAttention::new(
            config.hidden_size,
            config.encoder_num_attention_heads,
            config.encoder_kv_heads(),
            config.attention_bias,
            false, // encoder is non-causal
            config.partial_rotary_factor,
            config.max_position_embeddings,
            config.rope_theta,
        )?;
        let mlp = MoonshineEncoderMLP::new(
            config.hidden_size,
            config.intermediate_size,
            &config.encoder_hidden_act,
        )?;
        let input_layernorm = LayerNormBuilder::new(config.hidden_size as i32)
            .affine(true)
            .build()?;
        let post_attention_layernorm = LayerNormBuilder::new(config.hidden_size as i32)
            .affine(true)
            .build()?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&mut self, x: &Array, position_ids: Option<&Array>) -> Result<Array, Exception> {
        // Self-attention with pre-norm
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let (h, _) = self.self_attn.forward(&h, None, None, position_ids)?;
        let x = residual.add(&h)?;

        // MLP with pre-norm
        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        residual.add(&h)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

/// Output of a single decoder layer: `(hidden_states, self_attn_cache, cross_attn_cache)`.
type DecoderLayerOutput = (Array, (Array, Array), (Array, Array));

/// A single Moonshine decoder transformer layer.
///
/// Three sub-blocks:
/// 1. Causal self-attention with RoPE
/// 2. Cross-attention to encoder output (no RoPE)
/// 3. SwiGLU MLP
#[derive(Debug, ModuleParameters)]
pub struct MoonshineDecoderLayer {
    #[param]
    pub self_attn: MoonshineAttention,
    #[param]
    pub encoder_attn: MoonshineAttention,
    #[param]
    pub mlp: MoonshineDecoderMLP,
    #[param]
    pub input_layernorm: LayerNorm,
    #[param]
    pub post_attention_layernorm: LayerNorm,
    #[param]
    pub final_layernorm: LayerNorm,
}

impl MoonshineDecoderLayer {
    pub fn new(config: &MoonshineConfig) -> Result<Self, Exception> {
        let self_attn = MoonshineAttention::new(
            config.hidden_size,
            config.decoder_num_attention_heads,
            config.decoder_kv_heads(),
            config.attention_bias,
            true, // decoder self-attention is causal
            config.partial_rotary_factor,
            config.max_position_embeddings,
            config.rope_theta,
        )?;
        let encoder_attn = MoonshineAttention::new(
            config.hidden_size,
            config.decoder_num_attention_heads,
            config.decoder_kv_heads(),
            config.attention_bias,
            false, // cross-attention is non-causal
            config.partial_rotary_factor,
            config.max_position_embeddings,
            config.rope_theta,
        )?;
        let mlp = MoonshineDecoderMLP::new(config.hidden_size, config.intermediate_size)?;

        let input_layernorm = LayerNormBuilder::new(config.hidden_size as i32)
            .affine(true)
            .build()?;
        let post_attention_layernorm = LayerNormBuilder::new(config.hidden_size as i32)
            .affine(true)
            .build()?;
        let final_layernorm = LayerNormBuilder::new(config.hidden_size as i32)
            .affine(true)
            .build()?;

        Ok(Self {
            self_attn,
            encoder_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            final_layernorm,
        })
    }

    /// Forward pass.
    ///
    /// Returns `(output, new_self_attn_cache, new_cross_attn_cache)`.
    pub fn forward(
        &mut self,
        x: &Array,
        encoder_hidden_states: &Array,
        self_attn_cache: Option<&(Array, Array)>,
        cross_attn_cache: Option<&(Array, Array)>,
    ) -> Result<DecoderLayerOutput, Exception> {
        // 1. Causal self-attention
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let (h, new_self_cache) = self.self_attn.forward(&h, None, self_attn_cache, None)?;
        let x = residual.add(&h)?;

        // 2. Cross-attention to encoder output
        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(&x)?;
        let (h, new_cross_cache) =
            self.encoder_attn
                .forward(&h, Some(encoder_hidden_states), cross_attn_cache, None)?;
        let x = residual.add(&h)?;

        // 3. SwiGLU MLP
        let residual = x.clone();
        let h = self.final_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        let x = residual.add(&h)?;

        Ok((x, new_self_cache, new_cross_cache))
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Moonshine encoder: learned Conv1d audio frontend + transformer layers.
///
/// Takes raw 16kHz audio and produces contextual hidden representations.
/// Total downsampling factor: 64 × 3 × 2 = 384 (24ms per output frame).
#[derive(Debug, ModuleParameters)]
pub struct MoonshineEncoder {
    #[param]
    pub conv1: nn::Conv1d,
    #[param]
    pub groupnorm: GroupNorm,
    #[param]
    pub conv2: nn::Conv1d,
    #[param]
    pub conv3: nn::Conv1d,
    #[param]
    pub layers: Vec<MoonshineEncoderLayer>,
    #[param]
    pub layer_norm: LayerNorm,
}

impl MoonshineEncoder {
    pub fn new(config: &MoonshineConfig) -> Result<Self, Exception> {
        let dim = config.hidden_size;

        let conv1 = nn::Conv1dBuilder::new(1, dim as i32, 127)
            .stride(64)
            .bias(false)
            .build()?;

        let groupnorm = nn::GroupNormBuilder::new(1, dim as i32).build()?;

        let conv2 = nn::Conv1dBuilder::new(dim as i32, (2 * dim) as i32, 7)
            .stride(3)
            .bias(true)
            .build()?;

        let conv3 = nn::Conv1dBuilder::new((2 * dim) as i32, dim as i32, 3)
            .stride(2)
            .bias(true)
            .build()?;

        let layers = (0..config.encoder_num_hidden_layers)
            .map(|_| MoonshineEncoderLayer::new(config))
            .collect::<Result<Vec<_>, _>>()?;

        let layer_norm = LayerNormBuilder::new(dim as i32).affine(true).build()?;

        Ok(Self {
            conv1,
            groupnorm,
            conv2,
            conv3,
            layers,
            layer_norm,
        })
    }

    /// Encode raw audio waveform to hidden representations.
    ///
    /// - `audio`: shape `[samples]` (1D) or `[B, samples]` (2D, batched)
    /// - Returns: `[B, T', hidden_size]` where `T' ≈ samples / 384`
    pub fn forward(&mut self, audio: &Array) -> Result<Array, Exception> {
        // Ensure [B, T] shape
        let audio = if audio.ndim() == 1 {
            audio.reshape(&[1, -1])?
        } else {
            audio.clone()
        };

        // [B, T] -> [B, T, 1] for Conv1d (channel-last in MLX)
        let x = audio.reshape(&[audio.shape()[0], audio.shape()[1], 1])?;

        // Conv frontend with progressive downsampling
        let x = self.conv1.forward(&x)?;
        let x = mlx_rs::ops::tanh(&x)?;
        let x = self.groupnorm.forward(&x)?;
        let x = self.conv2.forward(&x)?;
        let x = nn::gelu(&x)?;
        let x = self.conv3.forward(&x)?;
        let x = nn::gelu(&x)?;

        // Position IDs for transformer layers
        let seq_len = x.shape()[1] as i32;
        let position_ids = Array::from_iter(0..seq_len, &[seq_len]).reshape(&[1, seq_len])?;

        // Transformer layers
        let mut x = x;
        for layer in &mut self.layers {
            x = layer.forward(&x, Some(&position_ids))?;
        }

        self.layer_norm.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Per-layer KV cache for the decoder.
#[derive(Clone)]
pub struct DecoderLayerCache {
    pub self_attn: Option<(Array, Array)>,
    pub cross_attn: Option<(Array, Array)>,
}

/// Moonshine decoder: token embedding + transformer layers with cross-attention.
#[derive(Debug, ModuleParameters)]
pub struct MoonshineDecoder {
    #[param]
    pub embed_tokens: Embedding,
    #[param]
    pub layers: Vec<MoonshineDecoderLayer>,
    #[param]
    pub norm: LayerNorm,
}

impl MoonshineDecoder {
    pub fn new(config: &MoonshineConfig) -> Result<Self, Exception> {
        let embed_tokens = Embedding::new(config.vocab_size as i32, config.hidden_size as i32)?;

        let layers = (0..config.decoder_num_hidden_layers)
            .map(|_| MoonshineDecoderLayer::new(config))
            .collect::<Result<Vec<_>, _>>()?;

        let norm = LayerNormBuilder::new(config.hidden_size as i32)
            .affine(true)
            .build()?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass.
    ///
    /// - `tokens`: token IDs `[B, T]`
    /// - `encoder_hidden_states`: encoder output `[B, T_enc, hidden_size]`
    /// - `cache`: optional per-layer KV cache
    ///
    /// Returns `(hidden_states, new_cache)`.
    pub fn forward(
        &mut self,
        tokens: &Array,
        encoder_hidden_states: &Array,
        cache: Option<&[DecoderLayerCache]>,
    ) -> Result<(Array, Vec<DecoderLayerCache>), Exception> {
        let x = self.embed_tokens.forward(tokens)?;

        let num_layers = self.layers.len();
        let empty_cache = vec![
            DecoderLayerCache {
                self_attn: None,
                cross_attn: None,
            };
            num_layers
        ];
        let layer_caches = cache.unwrap_or(&empty_cache);

        let mut new_cache = Vec::with_capacity(num_layers);
        let mut x = x;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let lc = &layer_caches[i];
            let (out, new_self, new_cross) = layer.forward(
                &x,
                encoder_hidden_states,
                lc.self_attn.as_ref(),
                lc.cross_attn.as_ref(),
            )?;
            x = out;
            new_cache.push(DecoderLayerCache {
                self_attn: Some(new_self),
                cross_attn: Some(new_cross),
            });
        }

        let x = self.norm.forward(&x)?;
        Ok((x, new_cache))
    }
}

// ---------------------------------------------------------------------------
// Full Model
// ---------------------------------------------------------------------------

/// Complete Moonshine speech-to-text model.
///
/// Encoder-decoder architecture with a learned audio frontend.
/// Supports greedy autoregressive decoding.
#[derive(Debug, ModuleParameters)]
pub struct MoonshineModel {
    #[param]
    pub encoder: MoonshineEncoder,
    #[param]
    pub decoder: MoonshineDecoder,

    pub config: MoonshineConfig,
}

impl MoonshineModel {
    pub fn new(config: &MoonshineConfig) -> Result<Self, Exception> {
        let encoder = MoonshineEncoder::new(config)?;
        let decoder = MoonshineDecoder::new(config)?;

        Ok(Self {
            encoder,
            decoder,
            config: config.clone(),
        })
    }

    /// Compute logits from decoder hidden states.
    ///
    /// When `tie_word_embeddings` is true, reuses the embedding matrix
    /// as the output projection (no separate `proj_out` weight).
    pub fn get_logits(&self, hidden_states: &Array) -> Result<Array, Exception> {
        if self.config.tie_word_embeddings {
            self.decoder.embed_tokens.as_linear(hidden_states)
        } else {
            // Non-tied case would need a separate proj_out linear layer.
            // All official Moonshine models use tied embeddings.
            Err(Exception::from(
                "Non-tied embeddings not implemented for Moonshine",
            ))
        }
    }

    /// Encode audio to hidden representations.
    pub fn encode(&mut self, audio: &Array) -> Result<Array, Exception> {
        self.encoder.forward(audio)
    }

    /// Run one decoder step.
    ///
    /// - `tokens`: token IDs for this step `[B, 1]`
    /// - `encoder_out`: encoder hidden states
    /// - `cache`: per-layer KV cache from previous step
    ///
    /// Returns `(hidden_states, new_cache)`.
    pub fn decode(
        &mut self,
        tokens: &Array,
        encoder_out: &Array,
        cache: Option<&[DecoderLayerCache]>,
    ) -> Result<(Array, Vec<DecoderLayerCache>), Exception> {
        self.decoder.forward(tokens, encoder_out, cache)
    }

    /// Greedy autoregressive transcription.
    ///
    /// - `audio`: raw 16kHz waveform, shape `[samples]` or `[1, samples]`
    /// - `max_tokens`: maximum number of tokens to generate
    ///
    /// Returns the generated token IDs (excluding BOS, excluding EOS).
    pub fn generate(&mut self, audio: &Array, max_tokens: usize) -> Result<Vec<u32>, Exception> {
        // Encode
        let encoder_out = self.encode(audio)?;
        encoder_out.eval()?;

        let bos = self.config.decoder_start_token_id;
        let eos = self.config.eos_token_id;

        let mut tokens: Vec<u32> = vec![bos];
        let mut cache: Option<Vec<DecoderLayerCache>> = None;

        for _ in 0..max_tokens {
            let last_token = *tokens.last().unwrap();
            let token_ids = Array::from_int(last_token as i32).reshape(&[1, 1])?;

            let (hidden, new_cache) = self.decode(&token_ids, &encoder_out, cache.as_deref())?;
            hidden.eval()?;

            // Get logits for the last position
            let last_hidden = hidden.index((.., -1, ..));
            let logits = self.get_logits(&last_hidden)?;

            // Greedy: argmax
            let next_token = mlx_rs::ops::indexing::argmax_axis(&logits, -1, false)?;
            let next_token_val: u32 = next_token.item();

            if next_token_val == eos {
                break;
            }

            tokens.push(next_token_val);
            cache = Some(new_cache);
        }

        // Remove BOS token
        Ok(tokens[1..].to_vec())
    }

    /// Sanitize weight keys from HuggingFace format to our module structure.
    ///
    /// - Strips `model.` prefix from `model.encoder.*` and `model.decoder.*`
    /// - Transposes Conv1d weights from PyTorch `[out, in, kernel]` to MLX
    ///   `[out, kernel, in]`
    /// - Skips `proj_out.weight` when `tie_word_embeddings` is true
    pub fn sanitize(
        &self,
        weights: std::collections::HashMap<String, Array>,
    ) -> Result<std::collections::HashMap<String, Array>, Exception> {
        let mut sanitized = std::collections::HashMap::new();

        for (key, value) in weights {
            // Strip "model." prefix
            let new_key = if key.starts_with("model.") {
                key.strip_prefix("model.").unwrap().to_string()
            } else if key.starts_with("proj_out.") && self.config.tie_word_embeddings {
                continue;
            } else {
                key
            };

            // Transpose Conv1d weights: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
            let value =
                if new_key.contains("conv") && new_key.contains("weight") && value.ndim() == 3 {
                    value.transpose_axes(&[0, 2, 1])?
                } else {
                    value
                };

            sanitized.insert(new_key, value);
        }

        Ok(sanitized)
    }
}
