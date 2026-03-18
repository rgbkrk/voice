use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{Linear, LinearBuilder};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{cumsum, sin, tanh};
use mlx_rs::Array;

use crate::ada_norm::AdaIN1d;
use crate::conv_weighted::ConvWeighted;
use voice_dsp::interpolate;

// ---------------------------------------------------------------------------
// SineGen (not a Module -- no trainable parameters)
// ---------------------------------------------------------------------------

/// Sine waveform generator for harmonic-plus-noise source signals.
///
/// Generates sine waves at the fundamental frequency and its harmonics,
/// adding noise for unvoiced regions. This struct has no trainable parameters.
#[derive(Debug, Clone)]
pub struct SineGen {
    pub sine_amp: f32,
    pub noise_std: f32,
    pub harmonic_num: i32,
    pub dim: i32,
    pub sampling_rate: i32,
    pub voiced_threshold: f32,
    pub upsample_scale: i32,
}

impl SineGen {
    /// Create a new SineGen.
    ///
    /// * `samp_rate` - audio sampling rate
    /// * `upsample_scale` - total upsampling factor from F0 frame rate to sample rate
    /// * `harmonic_num` - number of harmonics (0 = fundamental only)
    /// * `sine_amp` - amplitude of the sine signal
    /// * `noise_std` - standard deviation of noise added to voiced regions
    /// * `voiced_threshold` - F0 threshold below which a frame is unvoiced
    pub fn new(
        samp_rate: i32,
        upsample_scale: i32,
        harmonic_num: i32,
        sine_amp: f32,
        noise_std: f32,
        voiced_threshold: f32,
    ) -> Self {
        Self {
            sine_amp,
            noise_std,
            harmonic_num,
            dim: harmonic_num + 1,
            sampling_rate: samp_rate,
            voiced_threshold,
            upsample_scale,
        }
    }

    /// Compute voiced/unvoiced mask from F0.
    /// Returns 1.0 where f0 > voiced_threshold, 0.0 otherwise.
    fn f02uv(&self, f0: &Array) -> Result<Array, Exception> {
        let threshold = Array::from_f32(self.voiced_threshold);
        let uv = f0.gt(&threshold)?;
        uv.as_type::<f32>()
    }

    /// Convert F0 values to sine waveforms via phase accumulation.
    fn f02sine(&self, f0_values: &Array) -> Result<Array, Exception> {
        // f0_values: (B, T, dim)
        let sr = Array::from_f32(self.sampling_rate as f32);

        // Normalized frequency: f0 / sr, then mod 1
        let rad_values = f0_values.divide(&sr)?.remainder(Array::from_f32(1.0))?;

        // Random initial phase for each harmonic, with first column zeroed
        let b = f0_values.dim(0);
        let dim = f0_values.dim(2);
        let zero_col = Array::zeros::<f32>(&[b, 1])?;
        let rest_cols = if dim > 1 {
            let rest = mlx_rs::random::normal::<f32>(&[b, dim - 1], None, None, None)?;
            mlx_rs::ops::concatenate_axis(&[&zero_col, &rest], -1)?
        } else {
            zero_col
        };

        // rad_values[:, 0, :] += rand_ini
        // We'll add rand_ini expanded to (B, 1, dim) to the first time step
        let rand_expanded = rest_cols.expand_dims(1)?;

        // Add random init to first frame only by splitting and concatenating
        let t = rad_values.dim(1);
        let first_frame = rad_values.index((.., ..1_i32, ..));
        let first_frame = &first_frame + &rand_expanded;

        let rad_values = if t > 1 {
            let rest_frames = rad_values.index((.., 1_i32.., ..));
            mlx_rs::ops::concatenate_axis(&[&first_frame, &rest_frames], 1)?
        } else {
            first_frame
        };

        // Downsample for interpolation: (B, T, dim) -> (B, dim, T) for interp
        let rad_t = rad_values.transpose_axes(&[0, 2, 1])?;
        let scale_down = 1.0 / self.upsample_scale as f32;
        let rad_down = interpolate(&rad_t, scale_down, "linear")?;
        let rad_down = rad_down.transpose_axes(&[0, 2, 1])?;

        // Cumulative sum to get phase
        let phase = cumsum(&rad_down, 1, None, None)?;
        let two_pi = Array::from_f32(2.0 * std::f32::consts::PI);
        let phase = phase.multiply(&two_pi)?;

        // Upsample phase back: (B, T_down, dim) -> (B, dim, T_down) for interp
        let phase_t = phase.transpose_axes(&[0, 2, 1])?;
        let scale_up_factor = Array::from_f32(self.upsample_scale as f32);
        let phase_up = interpolate(
            &phase_t.multiply(&scale_up_factor)?,
            self.upsample_scale as f32,
            "linear",
        )?;
        let phase_up = phase_up.transpose_axes(&[0, 2, 1])?;

        // Generate sine
        sin(&phase_up)
    }

    /// Generate sine waves, voiced/unvoiced mask, and noise.
    ///
    /// * `f0` - fundamental frequency, shape (B, T, 1)
    ///
    /// Returns `(sine_waves, uv, noise)`.
    pub fn call(&self, f0: &Array) -> Result<(Array, Array, Array), Exception> {
        // Create harmonics: f0 * [1, 2, ..., harmonic_num+1]
        let harmonics = Array::arange::<i32, f32>(1, self.harmonic_num + 2, None)?;
        // harmonics shape: (harmonic_num+1,) -> (1, 1, harmonic_num+1)
        let harmonics = harmonics.reshape(&[1, 1, -1])?;
        let fn_freqs = f0.multiply(&harmonics)?;

        // Generate sine waves and scale by amplitude
        let sine_waves = self.f02sine(&fn_freqs)?;
        let amp = Array::from_f32(self.sine_amp);
        let sine_waves = sine_waves.multiply(&amp)?;

        // Voiced/unvoiced mask
        let uv = self.f02uv(f0)?;

        // Noise amplitude: uv * noise_std + (1 - uv) * sine_amp / 3
        let noise_std_arr = Array::from_f32(self.noise_std);
        let sine_amp_third = Array::from_f32(self.sine_amp / 3.0);
        let one = Array::from_f32(1.0);
        let one_minus_uv = &one - &uv;
        let noise_amp = (&uv * &noise_std_arr).add(one_minus_uv.multiply(&sine_amp_third)?)?;

        // Generate noise
        let noise_raw = mlx_rs::random::normal::<f32>(sine_waves.shape(), None, None, None)?;
        let noise = noise_amp.multiply(&noise_raw)?;

        // Mix: voiced regions get sine, unvoiced get noise
        let sine_waves = uv.multiply(&sine_waves)?.add(&noise)?;

        Ok((sine_waves, uv, noise))
    }
}

// ---------------------------------------------------------------------------
// SourceModuleHnNSF
// ---------------------------------------------------------------------------

/// Harmonic-plus-noise source module that generates a source signal from F0.
///
/// Combines a `SineGen` with a learned linear projection to merge harmonics
/// into a single source signal.
#[derive(Debug, ModuleParameters)]
pub struct SourceModuleHnNSF {
    pub sine_gen: SineGen,

    #[param]
    pub l_linear: Linear,

    pub sine_amp: f32,
}

impl SourceModuleHnNSF {
    /// Create a new `SourceModuleHnNSF`.
    ///
    /// * `sampling_rate` - audio sampling rate
    /// * `upsample_scale` - total upsampling factor
    /// * `harmonic_num` - number of harmonics
    /// * `sine_amp` - sine wave amplitude
    /// * `add_noise_std` - noise standard deviation
    /// * `voiced_threshold` - F0 threshold for voiced/unvoiced
    pub fn new(
        sampling_rate: i32,
        upsample_scale: i32,
        harmonic_num: i32,
        sine_amp: f32,
        add_noise_std: f32,
        voiced_threshold: f32,
    ) -> Result<Self, Exception> {
        let sine_gen = SineGen::new(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshold,
        );

        let l_linear = LinearBuilder::new(harmonic_num + 1, 1).build()?;

        Ok(Self {
            sine_gen,
            l_linear,
            sine_amp,
        })
    }
}

/// Output of `SourceModuleHnNSF::forward`.
pub struct SourceModuleOutput {
    pub sine_merge: Array,
    pub noise: Array,
    pub uv: Array,
}

impl Module<&Array> for SourceModuleHnNSF {
    type Error = Exception;
    type Output = SourceModuleOutput;

    fn forward(&mut self, x: &Array) -> Result<SourceModuleOutput, Exception> {
        let (sine_wavs, uv, _) = self.sine_gen.call(x)?;

        // Linear merge of harmonics: (B, T, dim) -> (B, T, 1)
        let sine_merge = self.l_linear.forward(&sine_wavs)?;
        let sine_merge = tanh(&sine_merge)?;

        // Generate noise for unvoiced regions
        let noise_amp = Array::from_f32(self.sine_amp / 3.0);
        let noise =
            mlx_rs::random::normal::<f32>(uv.shape(), None, None, None)?.multiply(&noise_amp)?;

        Ok(SourceModuleOutput {
            sine_merge,
            noise,
            uv,
        })
    }

    fn training_mode(&mut self, _mode: bool) {}
}

// ---------------------------------------------------------------------------
// AdaINResBlock1 (vocoder residual block with AdaIN and Snake activation)
// ---------------------------------------------------------------------------

/// Input for `AdaINResBlock1::forward`.
pub struct AdaINResBlock1Input<'a> {
    pub x: &'a Array,
    pub s: &'a Array,
}

/// Adaptive Instance Normalization residual block with Snake activation.
///
/// Each block has 3 sub-blocks. Each sub-block applies:
/// 1. AdaIN normalization with style
/// 2. Snake activation: `x + (1/alpha) * sin(alpha * x)^2`
/// 3. Dilated weighted convolution
/// 4. AdaIN normalization with style
/// 5. Snake activation
/// 6. Standard weighted convolution
/// 7. Residual addition
#[derive(Debug, ModuleParameters)]
pub struct AdaINResBlock1 {
    #[param]
    pub convs1: Vec<ConvWeighted>,

    #[param]
    pub convs2: Vec<ConvWeighted>,

    #[param]
    pub adain1: Vec<AdaIN1d>,

    #[param]
    pub adain2: Vec<AdaIN1d>,

    pub alpha1: Vec<Array>,

    pub alpha2: Vec<Array>,
}

/// Compute padding for a given kernel size and dilation: (k - 1) * d / 2
fn get_padding(kernel_size: i32, dilation: i32) -> i32 {
    (kernel_size - 1) * dilation / 2
}

impl AdaINResBlock1 {
    /// Create a new `AdaINResBlock1`.
    ///
    /// * `channels` - number of input/output channels
    /// * `kernel_size` - convolution kernel size
    /// * `dilation` - list of dilation factors for each sub-block
    /// * `style_dim` - style embedding dimension
    pub fn new(
        channels: i32,
        kernel_size: i32,
        dilation: &[i32],
        style_dim: i32,
    ) -> Result<Self, Exception> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut adain1 = Vec::new();
        let mut adain2 = Vec::new();
        let mut alpha1 = Vec::new();
        let mut alpha2 = Vec::new();

        for &d in dilation {
            convs1.push(ConvWeighted::new(
                channels,
                channels,
                kernel_size,
                1,
                get_padding(kernel_size, d),
                d,
                1,
                true,
                false,
            )?);
            convs2.push(ConvWeighted::new(
                channels,
                channels,
                kernel_size,
                1,
                get_padding(kernel_size, 1),
                1,
                1,
                true,
                false,
            )?);
            adain1.push(AdaIN1d::new(style_dim, channels)?);
            adain2.push(AdaIN1d::new(style_dim, channels)?);
            alpha1.push(Array::ones::<f32>(&[1, channels, 1])?);
            alpha2.push(Array::ones::<f32>(&[1, channels, 1])?);
        }

        Ok(Self {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
        })
    }
}

/// Snake1D activation: `x + (1/alpha) * sin(alpha * x)^2`
fn snake1d(x: &Array, alpha: &Array) -> Result<Array, Exception> {
    let ax = x.multiply(alpha)?;
    let s = sin(&ax)?;
    let s2 = s.multiply(&s)?;
    let inv_alpha = Array::from_f32(1.0).divide(alpha)?;
    let term = inv_alpha.multiply(&s2)?;
    x.add(&term)
}

impl Module<(&Array, &Array)> for AdaINResBlock1 {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, (x, s): (&Array, &Array)) -> Result<Array, Exception> {
        let mut x = x.clone();

        for i in 0..self.convs1.len() {
            // AdaIN1 + Snake1D activation
            let xt = self.adain1[i].forward((&x, s))?;
            let xt = snake1d(&xt, &self.alpha1[i])?;

            // Dilated conv1: (B, C, T) -> swap to (B, T, C) for conv, swap back
            let xt = xt.swap_axes(2, 1)?;
            let xt = self.convs1[i].forward_conv1d(&xt)?;
            let xt = xt.swap_axes(2, 1)?;

            // AdaIN2 + Snake1D activation
            let xt = self.adain2[i].forward((&xt, s))?;
            let xt = snake1d(&xt, &self.alpha2[i])?;

            // Standard conv2: (B, C, T) -> swap -> conv -> swap
            let xt = xt.swap_axes(2, 1)?;
            let xt = self.convs2[i].forward_conv1d(&xt)?;
            let xt = xt.swap_axes(2, 1)?;

            // Residual connection
            x = &xt + &x;
        }

        Ok(x)
    }

    fn training_mode(&mut self, _mode: bool) {}
}
