use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{
    Conv1d, Conv1dBuilder, Upsample, UpsampleMode,
    leaky_relu,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate_axis, exp, pad, sin, squeeze_axes, PadWidth};
use mlx_rs::Array;

use super::source::{AdaINResBlock1, SourceModuleHnNSF};
use voicers_dsp::MlxStft;
use crate::conv_weighted::ConvWeighted;

// ---------------------------------------------------------------------------
// Input types
// ---------------------------------------------------------------------------

/// Input for `Generator::forward`.
pub struct GeneratorInput<'a> {
    /// Encoded features, shape (B, C, T)
    pub x: &'a Array,
    /// Style embedding
    pub s: &'a Array,
    /// F0 curve (fundamental frequency), shape (B, T)
    pub f0: &'a Array,
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// iSTFT-based neural vocoder generator.
///
/// Takes encoded features, style, and F0 as input; upsamples with transposed
/// convolutions while mixing in harmonic source information; and produces
/// audio via inverse STFT.
#[derive(Debug, ModuleParameters)]
pub struct Generator {
    pub num_kernels: usize,
    pub num_upsamples: usize,

    #[param]
    pub m_source: SourceModuleHnNSF,

    pub f0_upsamp: Upsample,

    #[param]
    pub noise_convs: Vec<Conv1d>,

    #[param]
    pub noise_res: Vec<AdaINResBlock1>,

    #[param]
    pub ups: Vec<ConvWeighted>,

    #[param]
    pub resblocks: Vec<AdaINResBlock1>,

    #[param]
    pub conv_post: ConvWeighted,

    pub post_n_fft: i32,
    pub stft: MlxStft,

    /// Reflection padding amounts (left, right).
    pub reflection_pad: (i32, i32),
}

impl Generator {
    /// Create a new `Generator`.
    ///
    /// * `style_dim` - style embedding dimension
    /// * `resblock_kernel_sizes` - kernel sizes for each residual block type
    /// * `upsample_rates` - upsampling rate at each stage
    /// * `upsample_initial_channel` - channel count before first upsample
    /// * `resblock_dilation_sizes` - dilation patterns for each residual block type
    /// * `upsample_kernel_sizes` - kernel size for each upsampling stage
    /// * `gen_istft_n_fft` - FFT size for the output iSTFT
    /// * `gen_istft_hop_size` - hop size for the output iSTFT
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        style_dim: i32,
        resblock_kernel_sizes: &[i32],
        upsample_rates: &[i32],
        upsample_initial_channel: i32,
        resblock_dilation_sizes: &[Vec<i32>],
        upsample_kernel_sizes: &[i32],
        gen_istft_n_fft: i32,
        gen_istft_hop_size: i32,
    ) -> Result<Self, Exception> {
        let num_kernels = resblock_kernel_sizes.len();
        let num_upsamples = upsample_rates.len();

        // Total upsampling scale for the source module
        let total_upsample: i32 = upsample_rates.iter().product::<i32>() * gen_istft_hop_size;

        let m_source = SourceModuleHnNSF::new(
            24000,
            total_upsample,
            8,     // harmonic_num
            0.1,   // sine_amp
            0.003, // add_noise_std
            10.0,  // voiced_threshold
        )?;

        // F0 upsampling layer
        let f0_upsamp = Upsample::new(
            total_upsample as f32,
            UpsampleMode::Nearest,
        );

        // Build upsampling layers (transposed convolutions)
        let mut ups = Vec::new();
        for (i, (&u, &k)) in upsample_rates.iter().zip(upsample_kernel_sizes.iter()).enumerate() {
            let ch_out = upsample_initial_channel / (1 << (i + 1));
            let ch_in = upsample_initial_channel / (1 << i);
            let padding = (k - u) / 2;
            // encode=true means this is used as a transposed conv
            ups.push(ConvWeighted::new(
                ch_out,     // "in" for transposed = output channels
                ch_in,      // "out" for transposed = input channels
                k,
                u,          // stride
                padding,
                1,          // dilation
                1,          // groups
                true,       // bias
                true,       // encode (transpose mode)
            )?);
        }

        // Build residual blocks
        let mut resblocks = Vec::new();
        for i in 0..num_upsamples {
            let ch = upsample_initial_channel / (1 << (i + 1));
            for (_, (k, d)) in resblock_kernel_sizes
                .iter()
                .zip(resblock_dilation_sizes.iter())
                .enumerate()
            {
                resblocks.push(AdaINResBlock1::new(ch, *k, d, style_dim)?);
            }
        }

        // Build noise convolutions and noise residual blocks
        let mut noise_convs = Vec::new();
        let mut noise_res = Vec::new();
        for i in 0..num_upsamples {
            let c_cur = upsample_initial_channel / (1 << (i + 1));
            if i + 1 < num_upsamples {
                let stride_f0: i32 = upsample_rates[i + 1..].iter().product();
                let kernel = stride_f0 * 2;
                let pad = (stride_f0 + 1) / 2;
                noise_convs.push(
                    Conv1dBuilder::new(gen_istft_n_fft + 2, c_cur, kernel)
                        .stride(stride_f0)
                        .padding(pad)
                        .build()?,
                );
                noise_res.push(AdaINResBlock1::new(c_cur, 7, &[1, 3, 5], style_dim)?);
            } else {
                noise_convs.push(
                    Conv1dBuilder::new(gen_istft_n_fft + 2, c_cur, 1).build()?,
                );
                noise_res.push(AdaINResBlock1::new(c_cur, 11, &[1, 3, 5], style_dim)?);
            }
        }

        // Post-processing convolution
        let ch_final = upsample_initial_channel / (1 << num_upsamples);
        let conv_post = ConvWeighted::new(
            ch_final,
            gen_istft_n_fft + 2,
            7,
            1,
            3,
            1,
            1,
            true,
            false,
        )?;

        let stft = MlxStft::new(gen_istft_n_fft, gen_istft_hop_size, gen_istft_n_fft)?;

        Ok(Self {
            num_kernels,
            num_upsamples,
            m_source,
            f0_upsamp,
            noise_convs,
            noise_res,
            ups,
            resblocks,
            conv_post,
            post_n_fft: gen_istft_n_fft,
            stft,
            reflection_pad: (1, 0),
        })
    }
}

/// Apply reflection padding to the last dimension: pad with (left, right) on axis 2.
///
/// Equivalent to `mx.pad(x, ((0,0), (0,0), (left, right)))` in channel-first layout.
fn reflection_pad_1d(x: &Array, left: i32, right: i32) -> Result<Array, Exception> {
    let widths: &[(i32, i32)] = &[(0, 0), (0, 0), (left, right)];
    pad(x, PadWidth::Widths(widths), None, None)
}

impl Module<GeneratorInput<'_>> for Generator {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: GeneratorInput<'_>) -> Result<Array, Exception> {
        let GeneratorInput { x, s, f0 } = input;

        // Upsample F0: (B, T) -> (B, 1, T) -> upsample -> (B, T_up, 1)
        let f0_expanded = f0.expand_dims(1)?;
        let f0_t = f0_expanded.transpose_axes(&[0, 2, 1])?;
        // Upsample expects (B, spatial..., C) so we have (B, T, 1)
        let f0_up = self.f0_upsamp.forward(&f0_t)?;

        // Generate harmonic source from upsampled F0
        let source_out = self.m_source.forward(&f0_up)?;

        // har_source: (B, T_up, 1) -> (B, 1, T_up) -> squeeze -> (B, T_up)
        let har_source = source_out.sine_merge.transpose_axes(&[0, 2, 1])?;
        let har_source = squeeze_axes(&har_source, &[1])?;

        // STFT of harmonic source
        let (har_spec, har_phase) = self.stft.transform(&har_source)?;
        // har_spec, har_phase: (B, freq, T_stft)
        // Concatenate along freq axis and swap to (B, T_stft, freq*2)
        let har = concatenate_axis(&[&har_spec, &har_phase], 1)?;
        let har = har.swap_axes(2, 1)?;

        let mut x = x.clone();

        for i in 0..self.num_upsamples {
            x = leaky_relu(&x, 0.1)?;

            // Noise source from harmonic features
            let x_source = self.noise_convs[i].forward(&har)?;
            let x_source = x_source.swap_axes(2, 1)?;
            let x_source = self.noise_res[i].forward((&x_source, s))?;

            // Upsample with transposed convolution
            let x_swapped = x.swap_axes(2, 1)?;
            let x_up = self.ups[i].forward_conv_transpose1d(&x_swapped)?;
            x = x_up.swap_axes(2, 1)?;

            // Apply reflection padding at the last upsample stage
            if i == self.num_upsamples - 1 {
                x = reflection_pad_1d(&x, self.reflection_pad.0, self.reflection_pad.1)?;
            }

            // Add noise source
            x = &x + &x_source;

            // Apply residual blocks and average
            let mut xs: Option<Array> = None;
            for j in 0..self.num_kernels {
                let block_idx = i * self.num_kernels + j;
                let block_out = self.resblocks[block_idx].forward((&x, s))?;
                xs = Some(match xs {
                    None => block_out,
                    Some(prev) => &prev + &block_out,
                });
            }
            let num_k = Array::from_f32(self.num_kernels as f32);
            x = xs.unwrap().divide(&num_k)?;
        }

        // Final activation and convolution
        x = leaky_relu(&x, 0.01)?;
        let x_swapped = x.swap_axes(2, 1)?;
        let x_conv = self.conv_post.forward_conv1d(&x_swapped)?;
        x = x_conv.swap_axes(2, 1)?;

        // Split into magnitude spectrum and phase
        let half = self.post_n_fft / 2 + 1;
        let spec = x.index((.., ..half, ..));
        let spec = exp(&spec)?;
        let phase = x.index((.., half.., ..));
        let phase = sin(&phase)?;

        // Inverse STFT to get audio
        self.stft.inverse(&spec, &phase)
    }

    fn training_mode(&mut self, mode: bool) {
        for block in self.resblocks.iter_mut() {
            block.training_mode(mode);
        }
        for block in self.noise_res.iter_mut() {
            block.training_mode(mode);
        }
    }
}
