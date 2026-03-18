use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{leaky_relu, Dropout, DropoutBuilder, Upsample, UpsampleMode};
use mlx_rs::ops::{concatenate_axis, pad, sqrt, PadWidth};
use mlx_rs::Array;

use super::generator::Generator;
use crate::ada_norm::AdaIN1d;
use crate::conv_weighted::ConvWeighted;

// ---------------------------------------------------------------------------
// UpSample1d helper
// ---------------------------------------------------------------------------

/// Simple 1D nearest-neighbor upsampling by factor 2, or pass-through if `upsample` is false.
fn upsample_1d(x: &Array, do_upsample: bool) -> Result<Array, Exception> {
    if !do_upsample {
        return Ok(x.clone());
    }
    // x: (B, C, T) -- use nn::Upsample which expects (B, spatial, C)
    // Swap to (B, T, C), upsample, swap back
    let x_t = x.swap_axes(2, 1)?;
    let mut up = Upsample::new(2.0_f32, UpsampleMode::Nearest);
    let out = up.forward(&x_t)?;
    out.swap_axes(2, 1)
}

// ---------------------------------------------------------------------------
// AdainResBlk1d
// ---------------------------------------------------------------------------

/// Input for `AdainResBlk1d::forward`.
pub type AdainResBlk1dInput<'a> = (&'a Array, &'a Array);

/// Adaptive Instance Normalization residual block for 1D signals.
///
/// Applies style-conditioned normalization with learned shortcut and optional
/// upsampling via transposed convolution.
#[derive(Debug, ModuleParameters)]
pub struct AdainResBlk1d {
    pub learned_sc: bool,
    pub upsample_type: bool,

    #[param]
    pub conv1: ConvWeighted,

    #[param]
    pub conv2: ConvWeighted,

    #[param]
    pub norm1: AdaIN1d,

    #[param]
    pub norm2: AdaIN1d,

    #[param]
    pub conv1x1: Option<ConvWeighted>,

    #[param]
    pub dropout: Dropout,

    /// Transposed convolution used as upsampling pool when `upsample_type` is true.
    #[param]
    pub pool: Option<ConvWeighted>,
}

impl AdainResBlk1d {
    /// Create a new `AdainResBlk1d`.
    ///
    /// * `dim_in` - input channel dimension
    /// * `dim_out` - output channel dimension
    /// * `style_dim` - style embedding dimension
    /// * `upsample` - whether to upsample by factor 2
    /// * `dropout_p` - dropout probability
    pub fn new(
        dim_in: i32,
        dim_out: i32,
        style_dim: i32,
        upsample: bool,
        dropout_p: f32,
    ) -> Result<Self, Exception> {
        let learned_sc = dim_in != dim_out;

        let conv1 = ConvWeighted::new(dim_in, dim_out, 3, 1, 1, 1, 1, true, false)?;
        let conv2 = ConvWeighted::new(dim_out, dim_out, 3, 1, 1, 1, 1, true, false)?;

        let norm1 = AdaIN1d::new(style_dim, dim_in)?;
        let norm2 = AdaIN1d::new(style_dim, dim_out)?;

        let conv1x1 = if learned_sc {
            Some(ConvWeighted::new(
                dim_in, dim_out, 1, 1, 0, 1, 1, false, false,
            )?)
        } else {
            None
        };

        let dropout = DropoutBuilder::new()
            .p(dropout_p)
            .build()
            .map_err(|e| Exception::custom(e.to_string()))?;

        let pool = if upsample {
            // ConvWeighted used as transposed conv for upsampling
            // groups=dim_in for depthwise
            Some(ConvWeighted::new(
                1, dim_in, 3, 2, 1, 1, dim_in, true, false,
            )?)
        } else {
            None
        };

        Ok(Self {
            learned_sc,
            upsample_type: upsample,
            conv1,
            conv2,
            norm1,
            norm2,
            conv1x1,
            dropout,
            pool,
        })
    }

    /// Compute the shortcut connection, with optional upsampling and 1x1 conv.
    fn shortcut(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut x = upsample_1d(x, self.upsample_type)?;
        if let Some(ref mut conv) = self.conv1x1 {
            // (B, C, T) -> (B, T, C) for conv1d, then back
            let xt = x.swap_axes(2, 1)?;
            let xt = conv.forward_conv1d(&xt)?;
            x = xt.swap_axes(2, 1)?;
        }
        Ok(x)
    }

    /// Compute the residual path with normalization, activation, and convolution.
    fn residual(&mut self, x: &Array, s: &Array) -> Result<Array, Exception> {
        let mut x = self.norm1.forward((x, s))?;
        x = leaky_relu(&x, 0.2)?;

        if self.upsample_type {
            if let Some(ref mut pool) = self.pool {
                // Upsample via transposed convolution
                // x: (B, C, T) -> swap to (B, T, C) for conv_transpose1d
                let xt = x.swap_axes(2, 1)?;
                let xt = pool.forward_conv_transpose1d(&xt)?;
                // Pad time dim while still in (B, T, C) layout
                // Python: mx.pad(x, ((0,0), (1,0), (0,0))) pads dim 1 (time) with 1 at start
                let widths: &[(i32, i32)] = &[(0, 0), (1, 0), (0, 0)];
                let xt = pad(&xt, PadWidth::Widths(widths), None, None)?;
                // Swap back to (B, C, T)
                x = xt.swap_axes(2, 1)?;
            }
        }

        // dropout + conv1
        let x_drop = self.dropout.forward(&x)?;
        let xt = x_drop.swap_axes(2, 1)?;
        let xt = self.conv1.forward_conv1d(&xt)?;
        let mut x = xt.swap_axes(2, 1)?;

        x = self.norm2.forward((&x, s))?;
        x = leaky_relu(&x, 0.2)?;

        let xt = x.swap_axes(2, 1)?;
        let xt = self.conv2.forward_conv1d(&xt)?;
        xt.swap_axes(2, 1)
    }
}

impl Module<(&Array, &Array)> for AdainResBlk1d {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, (x, s): (&Array, &Array)) -> Result<Array, Exception> {
        let residual = self.residual(x, s)?;
        let shortcut = self.shortcut(x)?;
        let sum = &residual + &shortcut;
        let sqrt2 = sqrt(Array::from_f32(2.0))?;
        sum.divide(&sqrt2)
    }

    fn training_mode(&mut self, mode: bool) {
        self.dropout.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Input types for Decoder
// ---------------------------------------------------------------------------

/// Input for `Decoder::forward`.
pub struct DecoderInput<'a> {
    /// ASR (text encoder) features, shape (B, C, T)
    pub asr: &'a Array,
    /// F0 curve, shape (B, T)
    pub f0_curve: &'a Array,
    /// Energy (N), shape (B, T)
    pub n: &'a Array,
    /// Style embedding
    pub s: &'a Array,
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Full decoder that combines ASR features, F0, and energy with style conditioning
/// to produce audio via a `Generator`.
#[derive(Debug, ModuleParameters)]
pub struct Decoder {
    #[param]
    pub encode: AdainResBlk1d,

    #[param]
    pub decode: Vec<AdainResBlk1d>,

    #[param]
    pub f0_conv: ConvWeighted,

    #[param]
    pub n_conv: ConvWeighted,

    #[param]
    pub asr_res: Vec<ConvWeighted>,

    #[param]
    pub generator: Generator,
}

impl Decoder {
    /// Create a new `Decoder`.
    ///
    /// * `dim_in` - input ASR feature dimension
    /// * `style_dim` - style embedding dimension
    /// * `dim_out` - (unused, kept for API compat)
    /// * `resblock_kernel_sizes` - passed to Generator
    /// * `upsample_rates` - passed to Generator
    /// * `upsample_initial_channel` - passed to Generator
    /// * `resblock_dilation_sizes` - passed to Generator
    /// * `upsample_kernel_sizes` - passed to Generator
    /// * `gen_istft_n_fft` - FFT size for iSTFT
    /// * `gen_istft_hop_size` - hop size for iSTFT
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim_in: i32,
        style_dim: i32,
        _dim_out: i32,
        resblock_kernel_sizes: &[i32],
        upsample_rates: &[i32],
        upsample_initial_channel: i32,
        resblock_dilation_sizes: &[Vec<i32>],
        upsample_kernel_sizes: &[i32],
        gen_istft_n_fft: i32,
        gen_istft_hop_size: i32,
    ) -> Result<Self, Exception> {
        // Encoder: dim_in + 2 (F0 + N) -> 1024
        let encode = AdainResBlk1d::new(dim_in + 2, 1024, style_dim, false, 0.0)?;

        // Decoder blocks: 1024 + 2 + 64 -> 1024 (x3), then upsample to 512
        let decode = vec![
            AdainResBlk1d::new(1024 + 2 + 64, 1024, style_dim, false, 0.0)?,
            AdainResBlk1d::new(1024 + 2 + 64, 1024, style_dim, false, 0.0)?,
            AdainResBlk1d::new(1024 + 2 + 64, 1024, style_dim, false, 0.0)?,
            AdainResBlk1d::new(1024 + 2 + 64, 512, style_dim, true, 0.0)?,
        ];

        // F0 and N convolutions: downsample by stride 2
        let f0_conv = ConvWeighted::new(1, 1, 3, 2, 1, 1, 1, true, false)?;
        let n_conv = ConvWeighted::new(1, 1, 3, 2, 1, 1, 1, true, false)?;

        // ASR residual projection
        let asr_res = vec![ConvWeighted::new(512, 64, 1, 1, 0, 1, 1, true, false)?];

        let generator = Generator::new(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
            gen_istft_n_fft,
            gen_istft_hop_size,
        )?;

        Ok(Self {
            encode,
            decode,
            f0_conv,
            n_conv,
            asr_res,
            generator,
        })
    }
}

impl Module<DecoderInput<'_>> for Decoder {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: DecoderInput<'_>) -> Result<Array, Exception> {
        let DecoderInput {
            asr,
            f0_curve,
            n,
            s,
        } = input;

        // Process F0: (B, T) -> (B, 1, T) -> swap -> conv1d -> swap -> (B, 1, T_down)
        let f0_in = f0_curve.expand_dims(1)?;
        let f0_swapped = f0_in.swap_axes(2, 1)?;
        let f0_conv_out = self.f0_conv.forward_conv1d(&f0_swapped)?;
        let f0_feat = f0_conv_out.swap_axes(2, 1)?;

        // Process N: (B, T) -> (B, 1, T) -> swap -> conv1d -> swap -> (B, 1, T_down)
        let n_in = n.expand_dims(1)?;
        let n_swapped = n_in.swap_axes(2, 1)?;
        let n_conv_out = self.n_conv.forward_conv1d(&n_swapped)?;
        let n_feat = n_conv_out.swap_axes(2, 1)?;

        // Concatenate ASR features with F0 and N: (B, C+2, T)
        let x = concatenate_axis(&[asr, &f0_feat, &n_feat], 1)?;

        // Encode
        let mut x = self.encode.forward((&x, s))?;

        // ASR residual: (B, 512, T) -> (B, T, 512) -> conv1d -> (B, T, 64) -> (B, 64, T)
        let asr_swapped = asr.swap_axes(2, 1)?;
        let asr_proj = self.asr_res[0].forward_conv1d(&asr_swapped)?;
        let asr_res = asr_proj.swap_axes(2, 1)?;

        // Decode with residual injection
        let mut res = true;
        for block in self.decode.iter_mut() {
            if res {
                x = concatenate_axis(&[&x, &asr_res, &f0_feat, &n_feat], 1)?;
            }
            x = block.forward((&x, s))?;
            if block.upsample_type {
                res = false;
            }
        }

        // Generate audio
        use super::generator::GeneratorInput;
        self.generator.forward(GeneratorInput {
            x: &x,
            s,
            f0: f0_curve,
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.encode.training_mode(mode);
        for block in self.decode.iter_mut() {
            block.training_mode(mode);
        }
        self.generator.training_mode(mode);
    }
}
