//! iSTFTNet decoder: Generator with harmonic source, AdaINResBlock1 with snake activation.
//!
//! Ported from kokoro/istftnet.py

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

use crate::modules::{
    load_weight_norm_conv1d, load_weight_norm_conv_transpose1d, AdaIN1d, AdainResBlk1d,
};

// ---------------------------------------------------------------------------
// Snake activation: x + (1/alpha) * sin(alpha * x)^2
// ---------------------------------------------------------------------------

fn snake_activation(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let sin_val = (x.broadcast_mul(alpha))?.sin()?;
    let sin_sq = sin_val.sqr()?;
    let recip_alpha = alpha.recip()?;
    x + sin_sq.broadcast_mul(&recip_alpha)?
}

// ---------------------------------------------------------------------------
// AdaINResBlock1 — Generator residual block with snake activation
// ---------------------------------------------------------------------------

pub struct AdaINResBlock1 {
    convs1: Vec<nn::Conv1d>,
    convs2: Vec<nn::Conv1d>,
    adain1: Vec<AdaIN1d>,
    adain2: Vec<AdaIN1d>,
    alpha1: Vec<Tensor>,
    alpha2: Vec<Tensor>,
}

impl AdaINResBlock1 {
    pub fn load(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        style_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num = dilations.len();
        let mut convs1 = Vec::with_capacity(num);
        let mut convs2 = Vec::with_capacity(num);
        let mut adain1 = Vec::with_capacity(num);
        let mut adain2 = Vec::with_capacity(num);
        let mut alpha1 = Vec::with_capacity(num);
        let mut alpha2 = Vec::with_capacity(num);

        for (i, &d) in dilations.iter().enumerate() {
            let padding = (kernel_size * d - d) / 2;
            let cfg1 = nn::Conv1dConfig {
                padding,
                dilation: d,
                ..Default::default()
            };
            convs1.push(load_weight_norm_conv1d(
                channels,
                channels,
                kernel_size,
                cfg1,
                vb.pp(format!("convs1.{i}")),
            )?);

            let cfg2 = nn::Conv1dConfig {
                padding: (kernel_size - 1) / 2,
                dilation: 1,
                ..Default::default()
            };
            convs2.push(load_weight_norm_conv1d(
                channels,
                channels,
                kernel_size,
                cfg2,
                vb.pp(format!("convs2.{i}")),
            )?);

            adain1.push(AdaIN1d::load(
                style_dim,
                channels,
                vb.pp(format!("adain1.{i}")),
            )?);
            adain2.push(AdaIN1d::load(
                style_dim,
                channels,
                vb.pp(format!("adain2.{i}")),
            )?);

            alpha1.push(vb.get((1, channels, 1), &format!("alpha1.{i}"))?);
            alpha2.push(vb.get((1, channels, 1), &format!("alpha2.{i}"))?);
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

    /// x: [B, C, T], s: [B, style_dim]
    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let mut xt = self.adain1[i].forward(&x, s)?;
            xt = snake_activation(&xt, &self.alpha1[i])?;
            xt = self.convs1[i].forward(&xt)?;
            xt = self.adain2[i].forward(&xt, s)?;
            xt = snake_activation(&xt, &self.alpha2[i])?;
            xt = self.convs2[i].forward(&xt)?;
            x = (&xt + &x)?;
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// SineGen — sine waveform generator for F0
// ---------------------------------------------------------------------------

struct SineGen {
    sine_amp: f64,
    noise_std: f64,
    harmonic_num: usize,
    sampling_rate: f64,
    voiced_threshold: f64,
    upsample_scale: usize,
}

impl SineGen {
    fn new(
        sampling_rate: usize,
        upsample_scale: usize,
        harmonic_num: usize,
        sine_amp: f64,
        noise_std: f64,
        voiced_threshold: f64,
    ) -> Self {
        Self {
            sine_amp,
            noise_std,
            harmonic_num,
            sampling_rate: sampling_rate as f64,
            voiced_threshold,
            upsample_scale,
        }
    }

    /// f0: [B, T, 1]
    /// Returns: (sine_waves: [B, T, dim], uv: [B, T, 1])
    fn forward(&self, f0: &Tensor, device: &Device) -> Result<(Tensor, Tensor)> {
        let (b, t, _) = f0.dims3()?;
        let dim = self.harmonic_num + 1;

        // Create harmonic multipliers [1, 2, ..., harmonic_num+1]
        let harmonics: Vec<f32> = (1..=(self.harmonic_num + 1)).map(|i| i as f32).collect();
        let harmonics = Tensor::new(&harmonics[..], device)?
            .reshape((1, 1, dim))?
            .to_dtype(f0.dtype())?;

        // fn = f0 * [1, 2, ..., harmonic_num+1]  -> [B, T, dim]
        let fn_vals = f0.broadcast_mul(&harmonics)?;

        // rad_values = (fn / sampling_rate) % 1
        let sr = Tensor::new(self.sampling_rate as f32, device)?.to_dtype(f0.dtype())?;
        let mut rad_values = fn_vals.broadcast_div(&sr)?;
        // Modulo 1: fract
        rad_values = (&rad_values - &rad_values.floor()?)?;

        // Fix 2: Add initial random phase for non-fundamental harmonics
        // Python: rand_ini = torch.rand(B, dim); rand_ini[:, 0] = 0
        //         rad_values[:, 0, :] += rand_ini
        let rand_ini = Tensor::rand(0f32, 1f32, &[b, dim], device)?.to_dtype(f0.dtype())?;
        // Zero out fundamental (column 0)
        let mask_data: Vec<f32> = (0..dim).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect();
        let mask = Tensor::new(&mask_data[..], device)?
            .reshape((1, dim))?
            .to_dtype(f0.dtype())?;
        let rand_ini = rand_ini.broadcast_mul(&mask)?; // [B, dim]
        let rand_ini = rand_ini.unsqueeze(1)?; // [B, 1, dim]

        // Add random phase to first timestep only
        // rad_values[:, 0:1, :] += rand_ini
        let first_step = rad_values.narrow(1, 0, 1)?; // [B, 1, dim]
        let first_step_updated = (first_step + rand_ini)?;
        let rest = rad_values.narrow(1, 1, t - 1)?; // [B, T-1, dim]
        rad_values = Tensor::cat(&[&first_step_updated, &rest], 1)?; // [B, T, dim]

        // Fix 1: Downsample → cumsum → upsample (Python's interpolation trick)
        // This prevents phase drift on long sequences.
        // Python: rad_values = F.interpolate(rad_values.T, scale_factor=1/upsample_scale, mode="linear").T
        //         phase = cumsum(rad_values, dim=1) * 2*pi
        //         phase = F.interpolate(phase.T * upsample_scale, scale_factor=upsample_scale, mode="linear").T

        // Transpose to [B, dim, T] for interpolation along T axis
        let rv_bdt = rad_values.transpose(1, 2)?; // [B, dim, T]

        // Downsample by 1/upsample_scale using exact scale_factor
        let down_scale = 1.0 / self.upsample_scale as f64;
        let rv_down = linear_interpolate_1d_scale(&rv_bdt.contiguous()?, down_scale)?; // [B, dim, T_down]

        // Transpose back to [B, T_down, dim] for cumsum along dim 1
        let rv_down_btd = rv_down.transpose(1, 2)?; // [B, T_down, dim]

        // Cumsum at reduced resolution
        let phase_down = rv_down_btd.cumsum(1)?; // GPU matmul-based cumsum

        // Transpose to [B, dim, T_down], multiply by upsample_scale, upsample back
        let phase_bdt = phase_down.transpose(1, 2)?; // [B, dim, T_down]
        let scale_t = Tensor::new(self.upsample_scale as f32, device)?.to_dtype(f0.dtype())?;
        let phase_scaled = phase_bdt.broadcast_mul(&scale_t)?;
        let up_scale = self.upsample_scale as f64;
        let phase_up = linear_interpolate_1d_scale(&phase_scaled.contiguous()?, up_scale)?; // [B, dim, T]

        // Transpose back to [B, T, dim] and multiply by 2*pi
        let phase = phase_up.transpose(1, 2)?; // [B, T, dim]
        let two_pi = Tensor::new(2.0f32 * std::f32::consts::PI, device)?.to_dtype(f0.dtype())?;
        let phase = phase.broadcast_mul(&two_pi)?;

        let sine_amp_t = Tensor::new(self.sine_amp as f32, device)?.to_dtype(f0.dtype())?;
        let sines = phase.sin()?.broadcast_mul(&sine_amp_t)?;

        // UV (unvoiced) detection
        let uv_f64 = f0.to_dtype(DType::F32)?;
        let threshold =
            Tensor::new(self.voiced_threshold as f32, device)?.broadcast_as(uv_f64.shape())?;
        let uv = uv_f64.gt(&threshold)?.to_dtype(f0.dtype())?;

        // Noise for unvoiced regions
        let noise_amp = ((&uv * self.noise_std)? + ((1.0 - &uv)? * (self.sine_amp / 3.0))?)?;
        let noise = Tensor::randn(0f32, 1f32, sines.shape(), device)?.to_dtype(sines.dtype())?;
        let noise = noise.broadcast_mul(&noise_amp)?;

        // Final: sine * uv + noise
        let sine_waves = (sines.broadcast_mul(&uv)? + noise)?;

        Ok((sine_waves, uv))
    }
}

// ---------------------------------------------------------------------------
// SourceModuleHnNSF
// ---------------------------------------------------------------------------

pub struct SourceModuleHnNSF {
    sine_gen: SineGen,
    l_linear: nn::Linear,
    sine_amp: f64,
}

impl SourceModuleHnNSF {
    pub fn load(
        sampling_rate: usize,
        upsample_scale: usize,
        harmonic_num: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let sine_gen = SineGen::new(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            0.1,
            0.003,
            10.0,
        );
        let l_linear = nn::linear(harmonic_num + 1, 1, vb.pp("l_linear"))?;

        Ok(Self {
            sine_gen,
            l_linear,
            sine_amp: 0.1,
        })
    }

    /// x: [B, T, 1] (F0 values)
    /// Returns: (sine_merge: [B, T, 1], noise: [B, T, 1], uv: [B, T, 1])
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let device = x.device();
        let (sine_wavs, uv) = self.sine_gen.forward(x, device)?;

        // Linear merge of harmonics -> single channel
        let sine_merge = self.l_linear.forward(&sine_wavs)?;
        let sine_merge = sine_merge.tanh()?;

        // Noise source
        let noise = Tensor::randn(0f32, 1f32, uv.shape(), device)?.to_dtype(x.dtype())?;
        let noise = (noise * (self.sine_amp / 3.0))?;

        Ok((sine_merge, noise, uv))
    }
}

// ---------------------------------------------------------------------------
// TorchSTFT — STFT/iSTFT using candle operations
// ---------------------------------------------------------------------------

pub struct TorchSTFT {
    filter_length: usize,
    hop_length: usize,
    win_length: usize,
    /// Hann window coefficients
    window: Vec<f32>,
}

impl TorchSTFT {
    pub fn new(filter_length: usize, hop_length: usize, win_length: usize) -> Self {
        // Generate Hann window
        let window: Vec<f32> = (0..win_length)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / win_length as f64).cos())
                    as f32
            })
            .collect();

        Self {
            filter_length,
            hop_length,
            win_length,
            window,
        }
    }

    /// Compute STFT magnitude and phase via DFT convolution.
    ///
    /// input_data: [B, L] (time-domain signal)
    /// Returns: (magnitude: [B, n_fft/2+1, T], phase: [B, n_fft/2+1, T])
    pub fn transform(&self, input_data: &Tensor) -> Result<(Tensor, Tensor)> {
        let device = input_data.device();
        let dtype = input_data.dtype();
        let n_fft = self.filter_length;
        let n_bins = n_fft / 2 + 1;

        // Build DFT basis as Conv1d filters
        // Real and imaginary parts: [n_bins, 1, n_fft]
        let mut cos_basis = Vec::with_capacity(n_bins * n_fft);
        let mut sin_basis = Vec::with_capacity(n_bins * n_fft);

        for k in 0..n_bins {
            for n in 0..n_fft {
                let angle = 2.0 * std::f64::consts::PI * k as f64 * n as f64 / n_fft as f64;
                let w = if n < self.win_length {
                    self.window[n] as f64
                } else {
                    0.0
                };
                cos_basis.push((angle.cos() * w) as f32);
                sin_basis.push((-angle.sin() * w) as f32);
            }
        }

        let cos_kernel = Tensor::new(&cos_basis[..], device)?
            .reshape((n_bins, 1, n_fft))?
            .to_dtype(dtype)?;
        let sin_kernel = Tensor::new(&sin_basis[..], device)?
            .reshape((n_bins, 1, n_fft))?
            .to_dtype(dtype)?;

        // Pad input for STFT framing
        let pad_amount = n_fft / 2;
        let input_padded = reflection_pad_1d(input_data, pad_amount)?;

        // [B, L] -> [B, 1, L] for conv1d
        let x = input_padded.unsqueeze(1)?;

        let conv_cfg = nn::Conv1dConfig {
            stride: self.hop_length,
            ..Default::default()
        };

        // Apply DFT via conv1d
        let real = x.conv1d(&cos_kernel, 0, conv_cfg.stride, 1, 1)?;
        let imag = x.conv1d(&sin_kernel, 0, conv_cfg.stride, 1, 1)?;

        // Magnitude and phase
        let magnitude = (real.sqr()? + imag.sqr()?)?.sqrt()?;
        // atan2(imag, real) = atan(imag/real) with quadrant correction
        // We'll compute this element-wise via atan(imag/real) and handle signs
        // For simplicity, use the identity: phase = atan2(y, x) via:
        // Compute atan2(imag, real) on CPU since candle doesn't have atan
        let real_data = real.flatten_all()?.to_vec1::<f32>()?;
        let imag_data = imag.flatten_all()?.to_vec1::<f32>()?;
        let phase_data: Vec<f32> = imag_data
            .iter()
            .zip(real_data.iter())
            .map(|(y, x)| y.atan2(*x))
            .collect();
        let phase = Tensor::from_vec(phase_data, real.shape(), device)?.to_dtype(dtype)?;

        Ok((magnitude, phase))
    }

    /// Inverse STFT via GPU conv_transpose1d (iDFT + overlap-add in one operation).
    ///
    /// magnitude: [B, n_fft/2+1, T], phase: [B, n_fft/2+1, T]
    /// Returns: [B, 1, L] (time-domain signal with channel dim)
    pub fn inverse(&self, magnitude: &Tensor, phase: &Tensor) -> Result<Tensor> {
        let device = magnitude.device();
        let dtype = magnitude.dtype();
        let n_fft = self.filter_length;
        let n_bins = n_fft / 2 + 1;
        let (_b, _freq, num_frames) = magnitude.dims3()?;

        // Phase unwrapping along time axis (axis 2) — matches mlx-audio behavior.
        // This removes 2π discontinuities for smoother spectral reconstruction.
        // We immediately reduce the result modulo 2π to avoid precision loss in cos/sin
        // for large accumulated phase values.
        let phase = phase_unwrap_mod2pi(phase, 2)?;

        // Reconstruct complex spectrum: real + imag
        let real: Tensor = (magnitude * phase.cos()?)?;
        let imag: Tensor = (magnitude * phase.sin()?)?;

        // Build windowed inverse DFT kernels for conv_transpose1d.
        // Kernel shape: [n_bins, 1, n_fft] (in_channels=n_bins, out_channels/groups=1, kernel_size=n_fft)
        // kernel[k, 0, n] = scale * window[n] * cos(2πkn/N)  (cos_kernel)
        //                  = scale * window[n] * sin(2πkn/N)  (sin_kernel)
        // where scale = 1/N for DC and Nyquist (k=0, k=N/2), 2/N otherwise.
        // conv_transpose1d with stride=hop_length performs iDFT + overlap-add in one GPU op.
        let win_len = self.win_length;
        let mut cos_data = Vec::with_capacity(n_bins * n_fft);
        let mut sin_data = Vec::with_capacity(n_bins * n_fft);

        for k in 0..n_bins {
            let scale = if k == 0 || k == n_fft / 2 {
                1.0 / n_fft as f64
            } else {
                2.0 / n_fft as f64
            };
            for n in 0..n_fft {
                let w = if n < win_len {
                    self.window[n] as f64
                } else {
                    0.0
                };
                let angle = 2.0 * std::f64::consts::PI * k as f64 * n as f64 / n_fft as f64;
                cos_data.push((scale * w * angle.cos()) as f32);
                sin_data.push((scale * w * angle.sin()) as f32);
            }
        }

        let cos_kernel = Tensor::new(&cos_data[..], device)?
            .reshape((n_bins, 1, n_fft))?
            .to_dtype(dtype)?;
        let sin_kernel = Tensor::new(&sin_data[..], device)?
            .reshape((n_bins, 1, n_fft))?
            .to_dtype(dtype)?;

        // iDFT + window + overlap-add via conv_transpose1d on GPU.
        // Each input channel (freq bin) is convolved with its kernel and results are summed,
        // with stride=hop_length providing the overlap-add automatically.
        let output = (real.conv_transpose1d(&cos_kernel, 0, 0, self.hop_length, 1, 1)?
            - imag.conv_transpose1d(&sin_kernel, 0, 0, self.hop_length, 1, 1)?)?;
        // output: [B, 1, output_len] where output_len = (T-1)*hop_length + n_fft

        // COLA normalization: compute window squared sum on CPU (small, depends only on
        // window params and frame count, not on the audio data).
        let output_len = output.dim(2)?;
        let mut window_sum = vec![0f32; output_len];
        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_length;
            for n in 0..win_len.min(n_fft) {
                if start + n < output_len {
                    let w = self.window[n];
                    window_sum[start + n] += w * w;
                }
            }
        }

        // Clamp to avoid division by zero (leaves near-zero samples unchanged)
        for v in window_sum.iter_mut() {
            if *v < 1e-8 {
                *v = 1.0;
            }
        }
        let window_sum_tensor = Tensor::new(&window_sum[..], device)?
            .to_dtype(dtype)?
            .reshape((1, 1, output_len))?;
        let output = output.broadcast_div(&window_sum_tensor)?;

        // Strip padding: the forward STFT added n_fft/2 reflection padding on each side.
        let pad = n_fft / 2;
        let trimmed_len = output_len.saturating_sub(2 * pad);
        let output = output.narrow(2, pad, trimmed_len)?;

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Generator — the main vocoder generator
// ---------------------------------------------------------------------------

pub struct Generator {
    m_source: SourceModuleHnNSF,
    ups: Vec<nn::ConvTranspose1d>,
    noise_convs: Vec<nn::Conv1d>,
    noise_res: Vec<AdaINResBlock1>,
    resblocks: Vec<AdaINResBlock1>,
    conv_post: nn::Conv1d,
    num_kernels: usize,
    num_upsamples: usize,
    post_n_fft: usize,
    stft: TorchSTFT,
    upsample_scale: usize,
}

impl Generator {
    pub fn load(
        style_dim: usize,
        resblock_kernel_sizes: &[usize],
        upsample_rates: &[usize],
        upsample_initial_channel: usize,
        resblock_dilation_sizes: &[Vec<usize>],
        upsample_kernel_sizes: &[usize],
        gen_istft_n_fft: usize,
        gen_istft_hop_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_kernels = resblock_kernel_sizes.len();
        let num_upsamples = upsample_rates.len();
        let total_upsample: usize = upsample_rates.iter().product();
        let upsample_scale = total_upsample * gen_istft_hop_size;

        let m_source = SourceModuleHnNSF::load(24000, upsample_scale, 8, vb.pp("m_source"))?;

        // Upsample layers (ConvTranspose1d with weight_norm)
        let mut ups = Vec::with_capacity(num_upsamples);
        for (i, (&u, &k)) in upsample_rates
            .iter()
            .zip(upsample_kernel_sizes.iter())
            .enumerate()
        {
            let in_ch = upsample_initial_channel / (1 << i);
            let out_ch = upsample_initial_channel / (1 << (i + 1));
            let cfg = nn::ConvTranspose1dConfig {
                stride: u,
                padding: (k - u) / 2,
                ..Default::default()
            };
            ups.push(load_weight_norm_conv_transpose1d(
                in_ch,
                out_ch,
                k,
                cfg,
                vb.pp(format!("ups.{i}")),
            )?);
        }

        // Residual blocks
        let mut resblocks = Vec::new();
        for i in 0..num_upsamples {
            let ch = upsample_initial_channel / (1 << (i + 1));
            for (j, (k, d)) in resblock_kernel_sizes
                .iter()
                .zip(resblock_dilation_sizes.iter())
                .enumerate()
            {
                resblocks.push(AdaINResBlock1::load(
                    ch,
                    *k,
                    d,
                    style_dim,
                    vb.pp(format!("resblocks.{}", i * num_kernels + j)),
                )?);
            }
        }

        // Noise convolutions and residual blocks
        let mut noise_convs = Vec::new();
        let mut noise_res = Vec::new();
        for i in 0..num_upsamples {
            let c_cur = upsample_initial_channel / (1 << (i + 1));
            if i + 1 < num_upsamples {
                let stride_f0: usize = upsample_rates[i + 1..].iter().product();
                let kernel_size = stride_f0 * 2;
                let padding = stride_f0.div_ceil(2);
                let cfg = nn::Conv1dConfig {
                    stride: stride_f0,
                    padding,
                    ..Default::default()
                };
                noise_convs.push(nn::conv1d(
                    gen_istft_n_fft + 2,
                    c_cur,
                    kernel_size,
                    cfg,
                    vb.pp(format!("noise_convs.{i}")),
                )?);
                noise_res.push(AdaINResBlock1::load(
                    c_cur,
                    7,
                    &[1, 3, 5],
                    style_dim,
                    vb.pp(format!("noise_res.{i}")),
                )?);
            } else {
                let cfg = nn::Conv1dConfig::default();
                noise_convs.push(nn::conv1d(
                    gen_istft_n_fft + 2,
                    c_cur,
                    1,
                    cfg,
                    vb.pp(format!("noise_convs.{i}")),
                )?);
                noise_res.push(AdaINResBlock1::load(
                    c_cur,
                    11,
                    &[1, 3, 5],
                    style_dim,
                    vb.pp(format!("noise_res.{i}")),
                )?);
            }
        }

        // Post convolution
        let ch = upsample_initial_channel / (1 << num_upsamples);
        let conv_post = load_weight_norm_conv1d(
            ch,
            gen_istft_n_fft + 2,
            7,
            nn::Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_post"),
        )?;

        let stft = TorchSTFT::new(gen_istft_n_fft, gen_istft_hop_size, gen_istft_n_fft);

        Ok(Self {
            m_source,
            ups,
            noise_convs,
            noise_res,
            resblocks,
            conv_post,
            num_kernels,
            num_upsamples,
            post_n_fft: gen_istft_n_fft,
            stft,
            upsample_scale,
        })
    }

    /// x: [B, C, T], s: [B, style_dim], f0: [B, T_f0]
    pub fn forward(&self, x: &Tensor, s: &Tensor, f0: &Tensor) -> Result<Tensor> {
        // Upsample F0 to full resolution
        let f0_up = upsample_nearest_1d_single(f0, self.upsample_scale)?; // [B, T_full]
        let f0_3d = f0_up.unsqueeze(2)?; // [B, T_full, 1]

        // Generate harmonic source
        let (har_source, _noise_source, _uv) = self.m_source.forward(&f0_3d)?;

        // har_source: [B, T_full, 1] -> [B, 1, T_full] -> [B, T_full]
        let har_source = har_source.transpose(1, 2)?.contiguous()?;
        let har_source = har_source.squeeze(1)?;

        // STFT of harmonic source
        let (har_spec, har_phase) = self.stft.transform(&har_source)?;
        // har: [B, n_fft/2+1 + n_fft/2+1, T_stft] = [B, n_fft+2, T_stft]
        let har = Tensor::cat(&[&har_spec, &har_phase], 1)?;

        let mut x = x.clone();
        for i in 0..self.num_upsamples {
            x = nn::Activation::LeakyRelu(0.1).forward(&x)?;

            // Noise conditioning
            let mut x_source = self.noise_convs[i].forward(&har)?;
            x_source = self.noise_res[i].forward(&x_source, s)?;

            x = self.ups[i].forward(&x)?;

            // Reflection pad on last upsample
            if i == self.num_upsamples - 1 {
                x = reflection_pad_1d_channels(&x, 1, 0)?;
            }

            // Check length match — Python adds directly without trimming
            let x_len = x.dim(2)?;
            let xs_len = x_source.dim(2)?;
            if x_len != xs_len {
                // Pad the shorter one with zeros instead of truncating the longer one
                if x_len < xs_len {
                    let pad = Tensor::zeros(
                        &[x.dim(0)?, x.dim(1)?, xs_len - x_len],
                        x.dtype(),
                        x.device(),
                    )?;
                    x = Tensor::cat(&[&x, &pad], 2)?;
                } else {
                    let pad = Tensor::zeros(
                        &[x_source.dim(0)?, x_source.dim(1)?, x_len - xs_len],
                        x_source.dtype(),
                        x_source.device(),
                    )?;
                    x_source = Tensor::cat(&[&x_source, &pad], 2)?;
                }
            }
            x = (x + x_source)?;

            // Sum residual blocks
            let mut xs: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let rb_out = self.resblocks[i * self.num_kernels + j].forward(&x, s)?;
                xs = Some(match xs {
                    Some(prev) => (prev + rb_out)?,
                    None => rb_out,
                });
            }
            let nk = Tensor::new(self.num_kernels as f32, x.device())?.to_dtype(x.dtype())?;
            x = xs.unwrap().broadcast_div(&nk)?;
        }

        x = nn::Activation::LeakyRelu(0.01).forward(&x)?;
        x = self.conv_post.forward(&x)?;

        // Split into spec and phase
        let spec = x.narrow(1, 0, self.post_n_fft / 2 + 1)?.exp()?;
        let phase = x
            .narrow(1, self.post_n_fft / 2 + 1, self.post_n_fft / 2 + 1)?
            .sin()?;

        self.stft.inverse(&spec, &phase)
    }
}

// ---------------------------------------------------------------------------
// Decoder — top-level decoder wrapping Generator
// ---------------------------------------------------------------------------

pub struct Decoder {
    encode: AdainResBlk1d,
    decode: Vec<AdainResBlk1d>,
    f0_conv: nn::Conv1d,
    n_conv: nn::Conv1d,
    asr_res: nn::Conv1d,
    pub generator: Generator,
}

impl Decoder {
    pub fn load(
        dim_in: usize,
        style_dim: usize,
        _dim_out: usize,
        resblock_kernel_sizes: &[usize],
        upsample_rates: &[usize],
        upsample_initial_channel: usize,
        resblock_dilation_sizes: &[Vec<usize>],
        upsample_kernel_sizes: &[usize],
        gen_istft_n_fft: usize,
        gen_istft_hop_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // encode: AdainResBlk1d(dim_in + 2, 1024, style_dim)
        let encode = AdainResBlk1d::load(dim_in + 2, 1024, style_dim, false, vb.pp("encode"))?;

        // decode blocks
        let mut decode = Vec::new();
        decode.push(AdainResBlk1d::load(
            1024 + 2 + 64,
            1024,
            style_dim,
            false,
            vb.pp("decode.0"),
        )?);
        decode.push(AdainResBlk1d::load(
            1024 + 2 + 64,
            1024,
            style_dim,
            false,
            vb.pp("decode.1"),
        )?);
        decode.push(AdainResBlk1d::load(
            1024 + 2 + 64,
            1024,
            style_dim,
            false,
            vb.pp("decode.2"),
        )?);
        decode.push(AdainResBlk1d::load(
            1024 + 2 + 64,
            512,
            style_dim,
            true,
            vb.pp("decode.3"),
        )?);

        // F0_conv and N_conv: weight_norm Conv1d(1, 1, 3, stride=2, padding=1)
        let conv_cfg = nn::Conv1dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let f0_conv = load_weight_norm_conv1d(1, 1, 3, conv_cfg, vb.pp("F0_conv"))?;
        let n_conv = load_weight_norm_conv1d(1, 1, 3, conv_cfg, vb.pp("N_conv"))?;

        // asr_res: Sequential(weight_norm(Conv1d(512, 64, 1)))
        let asr_res = load_weight_norm_conv1d(512, 64, 1, Default::default(), vb.pp("asr_res.0"))?;

        let generator = Generator::load(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
            gen_istft_n_fft,
            gen_istft_hop_size,
            vb.pp("generator"),
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

    /// asr: [B, C, T], f0_curve: [B, T], n: [B, T], s: [B, style_dim]
    pub fn forward(
        &self,
        asr: &Tensor,
        f0_curve: &Tensor,
        n: &Tensor,
        s: &Tensor,
    ) -> Result<Tensor> {
        // F0 and N convolutions: [B, T] -> [B, 1, T] -> conv -> [B, 1, T/2]
        let f0 = self.f0_conv.forward(&f0_curve.unsqueeze(1)?)?;
        let n_out = self.n_conv.forward(&n.unsqueeze(1)?)?;

        // Concatenate [asr, F0, N] along channel dim
        let x = Tensor::cat(&[asr, &f0, &n_out], 1)?;

        let mut x = self.encode.forward(&x, s)?;

        let asr_res = self.asr_res.forward(asr)?;

        let mut res = true;
        for block in &self.decode {
            if res {
                x = Tensor::cat(&[&x, &asr_res, &f0, &n_out], 1)?;
            }
            x = block.forward(&x, s)?;
            if block.upsample {
                res = false;
            }
        }

        self.generator.forward(&x, s, f0_curve)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Cumulative sum along dimension 1.
#[allow(dead_code)]
fn cumsum_dim1(x: &Tensor) -> Result<Tensor> {
    let (_b, seq_len, _d) = x.dims3()?;
    if seq_len <= 1 {
        return Ok(x.clone());
    }

    // Build cumsum by iterative addition
    let mut slices = Vec::with_capacity(seq_len);
    slices.push(x.narrow(1, 0, 1)?);
    for i in 1..seq_len {
        let prev = &slices[i - 1];
        let curr = x.narrow(1, i, 1)?;
        slices.push((prev + curr)?);
    }
    Tensor::cat(&slices, 1)
}

/// Linear interpolation along the last dimension of a [B, C, T_in] tensor.
///
/// Uses PyTorch's F.interpolate scale_factor convention:
///   src_pos = (i + 0.5) / scale_factor - 0.5
/// where scale_factor is provided explicitly (not computed from sizes).
fn linear_interpolate_1d_scale(x: &Tensor, scale_factor: f64) -> Result<Tensor> {
    let (b, c, t_in) = x.dims3()?;
    let target_len = (t_in as f64 * scale_factor).floor() as usize;
    if t_in == target_len {
        return Ok(x.clone());
    }
    if t_in == 0 || target_len == 0 {
        return Tensor::zeros(&[b, c, target_len], x.dtype(), x.device());
    }

    let data = x
        .to_dtype(DType::F32)?
        .to_device(&candle_core::Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut out = vec![0f32; b * c * target_len];

    for batch in 0..b {
        for ch in 0..c {
            let src_offset = (batch * c + ch) * t_in;
            let dst_offset = (batch * c + ch) * target_len;

            for i in 0..target_len {
                // PyTorch scale_factor formula:
                // src_pos = (i + 0.5) / scale_factor - 0.5
                let src_pos = (i as f64 + 0.5) / scale_factor - 0.5;
                let src_pos = src_pos.max(0.0).min((t_in - 1) as f64);
                let lo = src_pos.floor() as usize;
                let hi = (lo + 1).min(t_in - 1);
                let frac = (src_pos - lo as f64) as f32;

                out[dst_offset + i] =
                    data[src_offset + lo] * (1.0 - frac) + data[src_offset + hi] * frac;
            }
        }
    }

    let device = x.device();
    let dtype = x.dtype();
    Tensor::new(&out[..], &candle_core::Device::Cpu)?
        .reshape(&[b, c, target_len])?
        .to_device(device)?
        .to_dtype(dtype)
}

/// Nearest-neighbor 1D upsampling for a 2D tensor [B, T] -> [B, T*factor]
fn upsample_nearest_1d_single(x: &Tensor, factor: usize) -> Result<Tensor> {
    let (b, t) = x.dims2()?;
    let x = x.unsqueeze(2)?; // [B, T, 1]
    let x = x.expand(&[b, t, factor])?; // [B, T, factor]
    x.reshape(&[b, t * factor])
}

/// Reflection padding for 1D signal [B, L]
fn reflection_pad_1d(x: &Tensor, pad: usize) -> Result<Tensor> {
    let (_b, l) = x.dims2()?;
    if pad == 0 {
        return Ok(x.clone());
    }
    // Left pad: reflect indices [pad, pad-1, ..., 1]
    let mut parts = Vec::new();
    if pad > 0 {
        let left = x.narrow(1, 1, pad)?.flip(&[1])?;
        parts.push(left);
    }
    parts.push(x.clone());
    if pad > 0 {
        let right = x.narrow(1, l - pad - 1, pad)?.flip(&[1])?;
        parts.push(right);
    }
    Tensor::cat(&parts, 1)
}

/// Reflection padding for 3D channels-first [B, C, T] with asymmetric padding
fn reflection_pad_1d_channels(x: &Tensor, pad_left: usize, pad_right: usize) -> Result<Tensor> {
    let (_b, _c, t) = x.dims3()?;
    let mut parts = Vec::new();

    if pad_left > 0 {
        let left = x.narrow(2, 1, pad_left)?.flip(&[2])?;
        parts.push(left);
    }
    parts.push(x.clone());
    if pad_right > 0 {
        let right = x.narrow(2, t - pad_right - 1, pad_right)?.flip(&[2])?;
        parts.push(right);
    }
    Tensor::cat(&parts, 2)
}

/// Phase unwrapping along a given axis.
/// Removes 2π discontinuities from a phase signal.
/// Phase unwrap + reduce modulo 2π.
///
/// Performs numpy-style phase unwrapping (removing 2π discontinuities) then
/// reduces the result modulo 2π so that cos/sin remain numerically accurate.
/// The cumulative correction is always an integer multiple of 2π, so reducing
/// the unwrapped phase mod 2π is equivalent to reducing the *original* phase
/// mod 2π. We exploit this by applying the reduction to the original data
/// (which is already in [-π, π] and thus avoids any precision loss).
fn phase_unwrap_mod2pi(phase: &Tensor, axis: usize) -> Result<Tensor> {
    let shape = phase.shape().dims().to_vec();
    let n = shape[axis];
    if n <= 1 {
        return Ok(phase.clone());
    }

    // Phase unwrap only adds integer multiples of 2π as corrections.
    // Since cos/sin are 2π-periodic, unwrap(phase) ≡ phase (mod 2π).
    // Therefore we can skip the actual unwrap computation and just reduce
    // the original phase to [0, 2π) for numerically accurate cos/sin.
    let data = phase
        .to_dtype(DType::F32)?
        .to_device(&candle_core::Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let two_pi = 2.0f32 * std::f32::consts::PI;
    let result: Vec<f32> = data
        .iter()
        .map(|&v| {
            let r = v % two_pi;
            if r < 0.0 { r + two_pi } else { r }
        })
        .collect();

    let device = phase.device();
    let dtype = phase.dtype();
    Tensor::new(&result[..], &candle_core::Device::Cpu)?
        .reshape(shape.as_slice())?
        .to_device(device)?
        .to_dtype(dtype)
}

/// Equivalent to numpy.unwrap / mlx_unwrap.
#[allow(dead_code)]
fn phase_unwrap(phase: &Tensor, axis: usize) -> Result<Tensor> {
    let shape = phase.shape().dims().to_vec();
    let n = shape[axis];
    if n <= 1 {
        return Ok(phase.clone());
    }

    // Work on CPU for simplicity
    let data = phase
        .to_dtype(DType::F32)?
        .to_device(&candle_core::Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let total: usize = shape.iter().product();
    let mut result = data.clone();

    // Compute strides for the given layout
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let _axis_stride = strides[axis];
    let axis_size = shape[axis];

    // Number of independent sequences to unwrap
    let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);

    let pi = std::f32::consts::PI;
    let two_pi = 2.0 * pi;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let _base = outer
                * strides.get(axis.wrapping_sub(1)).copied().unwrap_or(total)
                * if axis > 0 { 1 } else { 0 }
                + inner;

            // Simpler: compute the flat index for element [outer_indices..., t, inner_indices...]
            // For axis=2 on [B, F, T]: base = b * F * T + f * T, step along T with stride 1
            let start_idx = outer * (axis_size * inner_size) + inner;

            let mut cumulative_correction = 0.0f32;
            for t in 1..axis_size {
                let prev_idx = start_idx + (t - 1) * inner_size;
                let curr_idx = start_idx + t * inner_size;

                let diff = result[curr_idx] - result[prev_idx];
                // Wrap diff to [-pi, pi]
                let wrapped = ((diff + pi) % two_pi + two_pi) % two_pi - pi;
                // Handle edge case where wrapped == -pi and diff > 0
                let wrapped = if wrapped == -pi && diff > 0.0 {
                    pi
                } else {
                    wrapped
                };

                cumulative_correction += wrapped - diff;
                result[curr_idx] += cumulative_correction;
            }
        }
    }

    let device = phase.device();
    let dtype = phase.dtype();
    Tensor::new(&result[..], &candle_core::Device::Cpu)?
        .reshape(shape.as_slice())?
        .to_device(device)?
        .to_dtype(dtype)
}
