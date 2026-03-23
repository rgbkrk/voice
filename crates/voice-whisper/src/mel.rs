//! Mel spectrogram computation for Whisper.
//!
//! Two implementations:
//! - `pcm_to_mel()`: CPU-based via candle-transformers (reference)
//! - `GpuMelSpec`: GPU-based via DFT matrix multiplication on Metal

use candle_core::{Device, Tensor};
use candle_transformers::models::whisper::{audio, Config, HOP_LENGTH, N_FFT};

/// Pre-computed mel filter banks embedded in the binary.
const MEL_FILTERS_80: &[u8] = include_bytes!("../data/melfilters.bytes");
const MEL_FILTERS_128: &[u8] = include_bytes!("../data/melfilters128.bytes");

/// Load mel filter coefficients for the given config.
///
/// Returns the filter bank as a flat `Vec<f32>`. The filters are embedded
/// in the binary so no filesystem or network access is needed.
pub fn load_mel_filters(config: &Config) -> Result<Vec<f32>, String> {
    let mel_bytes = match config.num_mel_bins {
        80 => MEL_FILTERS_80,
        128 => MEL_FILTERS_128,
        n => {
            return Err(format!(
                "unsupported num_mel_bins: {n} (expected 80 or 128)"
            ))
        }
    };

    let mut filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut filters);
    Ok(filters)
}

/// Convert PCM audio samples to a mel spectrogram tensor on the given device.
/// CPU-based reference implementation.
///
/// Input: mono f32 samples at 16kHz.
/// Output: tensor of shape `(1, num_mel_bins, num_frames)`.
pub fn pcm_to_mel(
    config: &Config,
    samples: &[f32],
    mel_filters: &[f32],
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mel = audio::pcm_to_mel(config, samples, mel_filters);
    let mel_len = mel.len();
    Tensor::from_vec(
        mel,
        (1, config.num_mel_bins, mel_len / config.num_mel_bins),
        device,
    )
}

/// GPU-accelerated mel spectrogram using DFT matrix multiplication.
///
/// Precomputes the DFT basis, Hanning window, and mel filterbank as GPU
/// tensors. At inference time, the entire STFT + mel + log pipeline runs
/// on Metal via batched matmul — no CPU FFT needed.
pub struct GpuMelSpec {
    /// Hanning window [N_FFT]
    hann_window: Tensor,
    /// DFT cosine basis [N_FFT, N_FFT/2+1]
    dft_real: Tensor,
    /// DFT sine basis [N_FFT, N_FFT/2+1]
    dft_imag: Tensor,
    /// Mel filterbank [num_mel_bins, N_FFT/2+1]
    mel_filters: Tensor,
    /// Device
    device: Device,
}

impl GpuMelSpec {
    /// Create a new GPU mel spectrogram processor.
    ///
    /// Precomputes and uploads DFT matrices, window, and mel filters to the device.
    pub fn new(
        config: &Config,
        mel_filters_flat: &[f32],
        device: &Device,
    ) -> candle_core::Result<Self> {
        let n_fft = N_FFT; // 400
        let n_freq = n_fft / 2 + 1; // 201
        let num_mel_bins = config.num_mel_bins;

        // Hanning window
        let hann: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
            .collect();
        let hann_window = Tensor::from_vec(hann, (n_fft,), device)?;

        // DFT basis matrices
        // For each frequency bin k and time index n:
        //   dft_real[n, k] = cos(2*pi*k*n/N)
        //   dft_imag[n, k] = -sin(2*pi*k*n/N)
        let mut real_data = vec![0f32; n_fft * n_freq];
        let mut imag_data = vec![0f32; n_fft * n_freq];
        let two_pi = 2.0 * std::f32::consts::PI;

        for n in 0..n_fft {
            for k in 0..n_freq {
                let angle = two_pi * k as f32 * n as f32 / n_fft as f32;
                real_data[n * n_freq + k] = angle.cos();
                imag_data[n * n_freq + k] = -angle.sin();
            }
        }

        let dft_real = Tensor::from_vec(real_data, (n_fft, n_freq), device)?;
        let dft_imag = Tensor::from_vec(imag_data, (n_fft, n_freq), device)?;

        // Mel filterbank: [num_mel_bins, n_freq]
        let mel_filters =
            Tensor::from_vec(mel_filters_flat.to_vec(), (num_mel_bins, n_freq), device)?;

        Ok(Self {
            hann_window,
            dft_real,
            dft_imag,
            mel_filters,
            device: device.clone(),
        })
    }

    /// Compute mel spectrogram on GPU.
    ///
    /// Input: mono f32 samples at 16kHz.
    /// Output: tensor of shape `(1, num_mel_bins, num_frames)`.
    pub fn compute(&self, samples: &[f32]) -> candle_core::Result<Tensor> {
        let n_fft = N_FFT;
        let hop = HOP_LENGTH;

        // Calculate number of frames with padding (match CPU implementation)
        let n_len = samples.len() / hop;
        let pad = 100 * 30 / 2; // 100 * CHUNK_LENGTH / 2 = 1500
        let n_len = if n_len % pad != 0 {
            (n_len / pad + 1) * pad
        } else {
            n_len
        };
        let n_len = n_len + pad;

        // Pad samples
        let total_samples = n_len * hop + n_fft;
        let mut padded = vec![0.0f32; total_samples];
        let copy_len = samples.len().min(total_samples);
        padded[..copy_len].copy_from_slice(&samples[..copy_len]);

        // Frame the audio into overlapping windows: [num_frames, N_FFT]
        // This is done on CPU since it's just index reshuffling
        let num_frames = n_len;
        let mut frames_data = vec![0.0f32; num_frames * n_fft];
        for i in 0..num_frames {
            let offset = i * hop;
            let end = (offset + n_fft).min(padded.len());
            let len = end - offset;
            frames_data[i * n_fft..i * n_fft + len].copy_from_slice(&padded[offset..offset + len]);
        }

        // Upload frames to GPU: [num_frames, N_FFT]
        let frames = Tensor::from_vec(frames_data, (num_frames, n_fft), &self.device)?;

        // Apply Hanning window (broadcast multiply)
        let frames = frames.broadcast_mul(&self.hann_window)?;

        // DFT via matrix multiply — all on GPU
        // real_part = frames @ dft_real  → [num_frames, n_freq]
        // imag_part = frames @ dft_imag  → [num_frames, n_freq]
        let real_part = frames.matmul(&self.dft_real)?;
        let imag_part = frames.matmul(&self.dft_imag)?;

        // Power spectrum: real² + imag²
        let magnitude_sq = (real_part.sqr()? + imag_part.sqr()?)?;

        // Mel filterbank: [num_frames, n_freq] @ [n_freq, num_mel_bins] → [num_frames, num_mel_bins]
        let mel = magnitude_sq.matmul(&self.mel_filters.t()?)?;

        // Log scale: log10(max(mel, 1e-10))
        let mel = mel.clamp(1e-10f32, f32::MAX)?;
        // log10(x) = ln(x) / ln(10)
        let mel = (mel.log()? / f64::from(10.0f32.ln()))?;

        // Normalize: mmax = max(mel) - 8, mel = max(mel, mmax) / 4 + 1
        let mel_max = mel
            .max(candle_core::D::Minus1)?
            .max(0)?
            .to_scalar::<f32>()?;
        let mmax = mel_max - 8.0;
        let mel = mel.clamp(mmax, f32::MAX)?;
        let mel = ((mel / 4.0)? + 1.0)?;

        // Transpose to [1, num_mel_bins, num_frames] for Whisper encoder
        let mel = mel.t()?.unsqueeze(0)?;

        Ok(mel)
    }
}
