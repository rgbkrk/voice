use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::nn::{Conv1d, Conv1dBuilder, Linear, LinearBuilder};
use mlx_rs::ops::{broadcast_to, concatenate_axis, squeeze_axes, which, zeros};
use mlx_rs::Array;

use super::ada_norm::AdaLayerNorm;
use super::lstm::BiLstm;
use crate::modules::vocoder::decoder::AdainResBlk1d;

// ---------------------------------------------------------------------------
// Input types
// ---------------------------------------------------------------------------

/// Input for `ProsodyPredictor::forward`.
pub struct ProsodyPredictorInput<'a> {
    pub x: &'a Array,
    pub style: &'a Array,
    pub text_lengths: &'a Array,
    pub mask: &'a Array,
}

/// Input for `ProsodyPredictor::f0_n_train`.
pub struct F0NTrainInput<'a> {
    pub x: &'a Array,
    pub s: &'a Array,
}

/// Input for `DurationEncoder::forward`.
pub struct DurationEncoderInput<'a> {
    pub x: &'a Array,
    pub style: &'a Array,
    pub text_lengths: &'a Array,
    pub mask: &'a Array,
}

// ---------------------------------------------------------------------------
// DurationEncoderBlock
// ---------------------------------------------------------------------------

/// Interleaved blocks for DurationEncoder: either a BiLSTM or an AdaLayerNorm.
#[derive(Debug)]
pub enum DurationEncoderBlock {
    Lstm(BiLstm),
    Norm(AdaLayerNorm),
}

impl ModuleParameters for DurationEncoderBlock {
    fn num_parameters(&self) -> usize {
        match self {
            DurationEncoderBlock::Lstm(l) => l.num_parameters(),
            DurationEncoderBlock::Norm(n) => n.num_parameters(),
        }
    }

    fn parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            DurationEncoderBlock::Lstm(l) => l.parameters(),
            DurationEncoderBlock::Norm(n) => n.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> mlx_rs::module::ModuleParamMut<'_> {
        match self {
            DurationEncoderBlock::Lstm(l) => l.parameters_mut(),
            DurationEncoderBlock::Norm(n) => n.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            DurationEncoderBlock::Lstm(l) => l.trainable_parameters(),
            DurationEncoderBlock::Norm(n) => n.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            DurationEncoderBlock::Lstm(l) => l.freeze_parameters(recursive),
            DurationEncoderBlock::Norm(n) => n.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            DurationEncoderBlock::Lstm(l) => l.unfreeze_parameters(recursive),
            DurationEncoderBlock::Norm(n) => n.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            DurationEncoderBlock::Lstm(l) => l.all_frozen(),
            DurationEncoderBlock::Norm(n) => n.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            DurationEncoderBlock::Lstm(l) => l.any_frozen(),
            DurationEncoderBlock::Norm(n) => n.any_frozen(),
        }
    }
}

// ---------------------------------------------------------------------------
// DurationEncoder
// ---------------------------------------------------------------------------

/// Encodes text features with style conditioning and masking for duration prediction.
///
/// Interleaves BiLSTM layers and AdaLayerNorm layers, concatenating style
/// information and applying masking at each normalization step.
#[derive(Debug, ModuleParameters)]
pub struct DurationEncoder {
    #[param]
    pub lstms: Vec<DurationEncoderBlock>,

    pub dropout: f32,
    pub d_model: i32,
    pub sty_dim: i32,
}

impl DurationEncoder {
    /// Create a new `DurationEncoder`.
    ///
    /// * `sty_dim` - style embedding dimension
    /// * `d_model` - hidden dimension
    /// * `nlayers` - number of (BiLSTM + AdaLayerNorm) pairs
    /// * `dropout` - dropout probability (stored but not applied in inference)
    pub fn new(sty_dim: i32, d_model: i32, nlayers: i32, dropout: f32) -> Result<Self, Exception> {
        let mut blocks = Vec::new();
        for _ in 0..nlayers {
            let lstm = BiLstm::new(d_model + sty_dim, d_model / 2)?;
            let norm = AdaLayerNorm::new(sty_dim, d_model)?;
            blocks.push(DurationEncoderBlock::Lstm(lstm));
            blocks.push(DurationEncoderBlock::Norm(norm));
        }

        Ok(Self {
            lstms: blocks,
            dropout,
            d_model,
            sty_dim,
        })
    }
}

impl Module<DurationEncoderInput<'_>> for DurationEncoder {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: DurationEncoderInput<'_>) -> Result<Array, Exception> {
        let DurationEncoderInput {
            x,
            style,
            text_lengths: _,
            mask,
        } = input;

        // x: (B, C, T) -> transpose to (T, B, C)
        let x_t = x.transpose_axes(&[2, 0, 1])?;
        let t = x_t.dim(0);
        let b = x_t.dim(1);

        // s: broadcast style (1, 1, sty_dim) -> (T, B, sty_dim)
        let s = broadcast_to(style, &[t, b, style.dim(-1)])?;

        // Concatenate x with style: (T, B, C + sty_dim)
        let mut x_cat = concatenate_axis(&[&x_t, &s], -1)?;

        // Apply mask: mask is (B, T), expand -> (T, B, 1) then use where
        // m[..., None].transpose(1, 0, 2) -> (T, B, 1)
        let mask_expanded = mask
            .expand_dims(-1)?
            .transpose_axes(&[1, 0, 2])?;
        let zero = Array::from_f32(0.0);
        x_cat = which(&mask_expanded, &zero, &x_cat)?;

        // Transpose to (B, C+sty_dim, T) for processing
        let mut x_proc = x_cat.transpose_axes(&[1, 2, 0])?;

        // s transposed for re-concatenation: (T, B, sty_dim) -> (B, sty_dim, T)
        let s_bct = s.transpose_axes(&[1, 2, 0])?;

        // mask for (B, 1, T) dimension
        let mask_b1t = mask.expand_dims(-1)?.transpose_axes(&[0, 2, 1])?;

        for block in self.lstms.iter_mut() {
            match block {
                DurationEncoderBlock::Norm(norm) => {
                    // x_proc: (B, C, T) -> transpose to (B, T, C), apply norm, transpose back
                    let xt = x_proc.transpose_axes(&[0, 2, 1])?;
                    let normed = norm.forward((&xt, style))?;
                    x_proc = normed.transpose_axes(&[0, 2, 1])?;

                    // Re-concatenate style: (B, C+sty_dim, T)
                    x_proc = concatenate_axis(&[&x_proc, &s_bct], 1)?;

                    // Apply mask
                    x_proc = which(&mask_b1t, &zero, &x_proc)?;
                }
                DurationEncoderBlock::Lstm(lstm) => {
                    // x_proc: (B, C, T) -> (B, T, C) for LSTM, take hidden output
                    let xt = x_proc.transpose_axes(&[0, 2, 1])?;
                    let (h, _c) = lstm.forward(&xt)?;
                    // h: (B, T, hidden) -> (B, hidden, T)
                    x_proc = h.transpose_axes(&[0, 2, 1])?;

                    // Pad to match mask length if needed
                    let x_len = x_proc.dim(-1);
                    let m_len = mask.dim(-1);
                    if x_len < m_len {
                        let pad_shape = [x_proc.dim(0), x_proc.dim(1), m_len - x_len];
                        let pad_zeros = zeros::<f32>(&pad_shape)?;
                        x_proc = concatenate_axis(&[&x_proc, &pad_zeros], -1)?;
                    }
                }
            }
        }

        // Return (B, T, C)
        x_proc.transpose_axes(&[0, 2, 1])
    }

    fn training_mode(&mut self, mode: bool) {
        for block in self.lstms.iter_mut() {
            match block {
                DurationEncoderBlock::Lstm(l) => l.training_mode(mode),
                DurationEncoderBlock::Norm(n) => n.training_mode(mode),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ProsodyPredictor
// ---------------------------------------------------------------------------

/// Predicts duration, F0 (pitch), and energy (N) from text features and style.
///
/// Uses a `DurationEncoder` for text encoding, BiLSTM layers for shared and
/// duration-specific features, and `AdainResBlk1d` blocks with style conditioning
/// for F0 and energy prediction.
#[derive(Debug, ModuleParameters)]
pub struct ProsodyPredictor {
    #[param]
    pub text_encoder: DurationEncoder,

    #[param]
    pub lstm: BiLstm,

    #[param]
    pub duration_proj: Linear,

    #[param]
    pub shared: BiLstm,

    #[param]
    pub f0_blocks: Vec<AdainResBlk1d>,

    #[param]
    pub n_blocks: Vec<AdainResBlk1d>,

    #[param]
    pub f0_proj: Conv1d,

    #[param]
    pub n_proj: Conv1d,
}

impl ProsodyPredictor {
    /// Create a new `ProsodyPredictor`.
    ///
    /// * `style_dim` - style embedding dimension
    /// * `d_hid` - hidden dimension
    /// * `nlayers` - number of layers in the DurationEncoder
    /// * `max_dur` - maximum duration (output dim of duration projection)
    /// * `dropout` - dropout probability
    pub fn new(
        style_dim: i32,
        d_hid: i32,
        nlayers: i32,
        max_dur: i32,
        dropout: f32,
    ) -> Result<Self, Exception> {
        let text_encoder = DurationEncoder::new(style_dim, d_hid, nlayers, dropout)?;

        // BiLSTM: input = d_hid + style_dim, hidden = d_hid / 2 (each direction)
        let lstm = BiLstm::new(d_hid + style_dim, d_hid / 2)?;

        // Duration projection: d_hid -> max_dur (just a Linear layer)
        let duration_proj = LinearBuilder::new(d_hid, max_dur).build()?;

        // Shared BiLSTM for F0 and N
        let shared = BiLstm::new(d_hid + style_dim, d_hid / 2)?;

        // F0 blocks: three AdainResBlk1d with progressively smaller dimensions
        let f0_blocks = vec![
            AdainResBlk1d::new(d_hid, d_hid, style_dim, false, dropout)?,
            AdainResBlk1d::new(d_hid, d_hid / 2, style_dim, true, dropout)?,
            AdainResBlk1d::new(d_hid / 2, d_hid / 2, style_dim, false, dropout)?,
        ];

        // N blocks: same structure as F0
        let n_blocks = vec![
            AdainResBlk1d::new(d_hid, d_hid, style_dim, false, dropout)?,
            AdainResBlk1d::new(d_hid, d_hid / 2, style_dim, true, dropout)?,
            AdainResBlk1d::new(d_hid / 2, d_hid / 2, style_dim, false, dropout)?,
        ];

        // Projection convolutions: reduce to 1 channel
        let f0_proj = Conv1dBuilder::new(d_hid / 2, 1, 1).padding(0).build()?;
        let n_proj = Conv1dBuilder::new(d_hid / 2, 1, 1).padding(0).build()?;

        Ok(Self {
            text_encoder,
            lstm,
            duration_proj,
            shared,
            f0_blocks,
            n_blocks,
            f0_proj,
            n_proj,
        })
    }

    /// Predict F0 and energy N from shared LSTM features and style.
    ///
    /// * `x` - features from the text encoder, shape (B, C, T)
    /// * `s` - style embedding
    ///
    /// Returns `(F0, N)`, each of shape `(B, T)`.
    pub fn f0_n_train(&mut self, x: &Array, s: &Array) -> Result<(Array, Array), Exception> {
        // shared LSTM: input (B, C, T) -> (B, T, C)
        let x_t = x.transpose_axes(&[0, 2, 1])?;
        let (shared_out, _) = self.shared.forward(&x_t)?;
        // shared_out: (B, T, hidden) -> (B, hidden, T)
        let shared_bct = shared_out.transpose_axes(&[0, 2, 1])?;

        // F0 path
        let mut f0 = shared_bct.clone();
        for block in self.f0_blocks.iter_mut() {
            f0 = block.forward((&f0, s))?;
        }
        // f0: (B, C, T) -> swap to (B, T, C) for conv1d, then back
        let f0 = f0.swap_axes(2, 1)?;
        let f0 = self.f0_proj.forward(&f0)?;
        let f0 = f0.swap_axes(2, 1)?;
        // Squeeze channel dim: (B, 1, T) -> (B, T)
        let f0 = squeeze_axes(&f0, &[1])?;

        // N (energy) path
        let mut n = shared_bct;
        for block in self.n_blocks.iter_mut() {
            n = block.forward((&n, s))?;
        }
        let n = n.swap_axes(2, 1)?;
        let n = self.n_proj.forward(&n)?;
        let n = n.swap_axes(2, 1)?;
        let n = squeeze_axes(&n, &[1])?;

        Ok((f0, n))
    }
}

impl Module<ProsodyPredictorInput<'_>> for ProsodyPredictor {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: ProsodyPredictorInput<'_>) -> Result<Array, Exception> {
        let ProsodyPredictorInput {
            x,
            style,
            text_lengths,
            mask,
        } = input;

        // Encode text with style and mask
        let encoded = self.text_encoder.forward(DurationEncoderInput {
            x,
            style,
            text_lengths,
            mask,
        })?;

        // encoded: (B, T, d_hid)
        // Concatenate style: broadcast style -> (B, T, style_dim)
        let b = encoded.dim(0);
        let t = encoded.dim(1);
        let s_broad = broadcast_to(style, &[b, t, style.dim(-1)])?;
        let enc_cat = concatenate_axis(&[&encoded, &s_broad], -1)?;

        // Duration LSTM
        let (dur_out, _) = self.lstm.forward(&enc_cat)?;
        // dur_out: (B, T, d_hid) -> project to (B, T, max_dur)
        let dur = self.duration_proj.forward(&dur_out)?;

        Ok(dur)
    }

    fn training_mode(&mut self, mode: bool) {
        self.text_encoder.training_mode(mode);
        self.lstm.training_mode(mode);
        self.shared.training_mode(mode);
        for block in self.f0_blocks.iter_mut() {
            block.training_mode(mode);
        }
        for block in self.n_blocks.iter_mut() {
            block.training_mode(mode);
        }
    }
}
