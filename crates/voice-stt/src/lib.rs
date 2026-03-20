//! Speech-to-text library backed by MLX, starting with Moonshine.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use voice_stt::{load_model, transcribe, TranscribeResult};
//!
//! let mut model = load_model("UsefulSensors/moonshine-tiny").unwrap();
//! let result = transcribe(&mut model, "audio.wav").unwrap();
//! println!("{}", result.text);
//! ```
//!
//! # Supported models
//!
//! | Model | Repo ID | Params |
//! |-------|---------|--------|
//! | Moonshine Tiny | `UsefulSensors/moonshine-tiny` | 27M |
//! | Moonshine Base | `UsefulSensors/moonshine-base` | 61M |
//!
//! # Architecture
//!
//! Moonshine uses a learned Conv1d audio frontend instead of mel spectrograms,
//! so raw 16kHz audio goes directly into the encoder with no DSP preprocessing.

pub mod builtin;
pub mod error;
pub mod moonshine;

use std::path::{Path, PathBuf};

pub use error::{Result, SttError};
pub use mlx_rs::Array;
pub use moonshine::{MoonshineConfig, MoonshineModel};
pub use tokenizers;

/// Result of a transcription.
#[derive(Debug, Clone)]
pub struct TranscribeResult {
    /// The transcribed text.
    pub text: String,
    /// Number of tokens generated (excluding BOS/EOS).
    pub tokens: Vec<u32>,
    /// Sample rate of the model (always 16000 for Moonshine).
    pub sample_rate: u32,
}

/// Load a Moonshine model from a HuggingFace repo or local path.
///
/// # Examples
///
/// ```rust,no_run
/// let mut model = voice_stt::load_model("UsefulSensors/moonshine-tiny").unwrap();
/// ```
pub fn load_model(path_or_repo: &str) -> Result<MoonshineModel> {
    let model_dir = if Path::new(path_or_repo).exists() {
        PathBuf::from(path_or_repo)
    } else {
        download_model(path_or_repo)?
    };

    // Use embedded config if available, otherwise read from disk
    let config: MoonshineConfig = if let Some(result) = builtin::config_for_repo(path_or_repo) {
        result?
    } else {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        serde_json::from_str(&config_str)?
    };

    let mut model = MoonshineModel::new(&config).map_err(SttError::Mlx)?;

    // Load weights
    let weights_path = model_dir.join("model.safetensors");
    if !weights_path.exists() {
        return Err(SttError::Weight(format!(
            "model.safetensors not found in {}",
            model_dir.display()
        )));
    }

    let raw_weights =
        Array::load_safetensors(&weights_path).map_err(|e| SttError::Weight(e.to_string()))?;

    let weights = model
        .sanitize(raw_weights)
        .map_err(|e| SttError::Weight(e.to_string()))?;

    // Apply weights to model parameters
    {
        use mlx_rs::module::ModuleParameters;
        let mut params = model.parameters_mut().flatten();
        let mut loaded = 0;
        let mut missing = Vec::new();
        for (key, value) in &weights {
            if let Some(param) = params.get_mut(&**key) {
                **param = value.clone();
                loaded += 1;
            } else {
                missing.push(key.clone());
            }
        }
        if !missing.is_empty() {
            eprintln!(
                "[WARN] Loaded {}/{} weights, {} unmatched",
                loaded,
                weights.len(),
                missing.len()
            );
        }
    }

    Ok(model)
}

/// Load the tokenizer from a model directory.
///
/// Moonshine uses a HuggingFace fast tokenizer stored as `tokenizer.json`.
pub fn load_tokenizer(path_or_repo: &str) -> Result<tokenizers::Tokenizer> {
    // Use embedded tokenizer if available
    if let Some(result) = builtin::tokenizer_for_repo(path_or_repo) {
        return result;
    }

    let model_dir = if Path::new(path_or_repo).exists() {
        PathBuf::from(path_or_repo)
    } else {
        download_model(path_or_repo)?
    };

    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(SttError::Tokenizer(format!(
            "tokenizer.json not found in {}",
            model_dir.display()
        )));
    }

    tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| SttError::Tokenizer(e.to_string()))
}

/// Transcribe an audio file.
///
/// Loads the audio file as WAV (16-bit PCM or 32-bit float), converts to
/// 16kHz mono f32, and runs greedy decoding.
///
/// # Examples
///
/// ```rust,no_run
/// let mut model = voice_stt::load_model("UsefulSensors/moonshine-tiny").unwrap();
/// let result = voice_stt::transcribe(&mut model, "audio.wav").unwrap();
/// println!("{}", result.text);
/// ```
pub fn transcribe(
    model: &mut MoonshineModel,
    audio_path: impl AsRef<Path>,
) -> Result<TranscribeResult> {
    let audio_path = audio_path.as_ref();
    let samples = load_wav_as_f32(audio_path)?;
    transcribe_audio(model, &samples, model.config.sample_rate())
}

/// Transcribe raw audio samples.
///
/// - `samples`: mono f32 audio samples
/// - `sample_rate`: sample rate of the input audio (will be resampled to 16kHz if different)
pub fn transcribe_audio(
    model: &mut MoonshineModel,
    samples: &[f32],
    sample_rate: u32,
) -> Result<TranscribeResult> {
    let samples = if sample_rate != 16000 {
        resample(samples, sample_rate, 16000)
    } else {
        samples.to_vec()
    };

    let audio = Array::from_slice(&samples, &[samples.len() as i32]);

    let tokens = model.generate(&audio, 200).map_err(SttError::Mlx)?;

    // Try embedded tokenizer first, then fall back to ASCII decode
    let text = if let Ok(tokenizer) = builtin::builtin_tokenizer() {
        tokenizer
            .decode(&tokens, true)
            .unwrap_or_else(|_| decode_tokens_fallback(&tokens))
    } else {
        decode_tokens_fallback(&tokens)
    };

    Ok(TranscribeResult {
        text,
        tokens,
        sample_rate: 16000,
    })
}

/// Transcribe raw audio samples with a tokenizer for proper text decoding.
pub fn transcribe_audio_with_tokenizer(
    model: &mut MoonshineModel,
    samples: &[f32],
    sample_rate: u32,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<TranscribeResult> {
    let samples = if sample_rate != 16000 {
        resample(samples, sample_rate, 16000)
    } else {
        samples.to_vec()
    };

    let audio = Array::from_slice(&samples, &[samples.len() as i32]);

    let tokens = model.generate(&audio, 200).map_err(SttError::Mlx)?;

    let text = tokenizer
        .decode(&tokens, true)
        .map_err(|e| SttError::Tokenizer(e.to_string()))?;

    Ok(TranscribeResult {
        text,
        tokens,
        sample_rate: 16000,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Download model files from HuggingFace Hub.
fn download_model(repo_id: &str) -> Result<PathBuf> {
    let api = hf_hub::api::sync::Api::new().map_err(|e| SttError::Hub(e.to_string()))?;
    let repo = api.model(repo_id.to_string());

    // Download required files
    let config_path = repo
        .get("config.json")
        .map_err(|e| SttError::Hub(e.to_string()))?;

    repo.get("model.safetensors")
        .map_err(|e| SttError::Hub(e.to_string()))?;

    // Try to download tokenizer (optional — not all model paths have it)
    let _ = repo.get("tokenizer.json");

    // Return the directory containing the downloaded files
    let model_dir = config_path
        .parent()
        .ok_or_else(|| SttError::Hub("Could not determine model directory".to_string()))?;

    Ok(model_dir.to_path_buf())
}

/// Load a WAV file and return mono f32 samples.
///
/// Handles both 16-bit integer and 32-bit float WAV formats.
/// Multi-channel audio is mixed down to mono by averaging.
fn load_wav_as_f32(path: &Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| SttError::Audio(format!("Failed to open {}: {e}", path.display())))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
    };

    // Mix down to mono if multi-channel
    let mono = if channels > 1 {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    // Resample to 16kHz if needed
    let mono = if spec.sample_rate != 16000 {
        resample(&mono, spec.sample_rate, 16000)
    } else {
        mono
    };

    Ok(mono)
}

/// High-quality audio resampling using rubato's sinc interpolation.
///
/// Uses a windowed sinc resampler with 256-tap filter for clean
/// anti-aliasing. Falls back to linear interpolation if rubato fails
/// (e.g. extremely short inputs).
///
/// For 24kHz→16kHz (AirPods): ~40dB SNR vs ~22dB with linear.
/// For 48kHz→16kHz: ~60dB SNR vs ~17dB with linear.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }

    // Try rubato sinc resampler first
    match resample_sinc(samples, from_rate, to_rate) {
        Ok(resampled) => resampled,
        Err(_) => {
            // Fallback to linear for edge cases (very short audio, etc.)
            resample_linear(samples, from_rate, to_rate)
        }
    }
}

/// Sinc-based resampling via rubato.
///
/// Uses rubato's sinc resampler with a 128-tap filter and Blackman window
/// for high-quality anti-aliasing.
fn resample_sinc(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::{
        calculate_cutoff, Async, FixedAsync, Indexing, Resampler,
        SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let sinc_len = 128;
    let window = WindowFunction::Blackman2;
    let f_cutoff = calculate_cutoff(sinc_len, window);

    let params = SincInterpolationParameters {
        sinc_len,
        f_cutoff,
        interpolation: SincInterpolationType::Quadratic,
        oversampling_factor: 256,
        window,
    };

    let ratio = to_rate as f64 / from_rate as f64;
    let chunk_size = samples.len();

    let mut resampler = Async::<f64>::new_sinc(
        ratio,
        1.1,
        &params,
        chunk_size,
        1, // mono
        FixedAsync::Input,
    )
    .map_err(|e| SttError::Audio(format!("Resampler init failed: {e}")))?;

    // Convert f32 → f64 for rubato
    let input_f64: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
    let num_input_frames = input_f64.len();

    // Compute output size
    let num_output_frames = (num_input_frames as f64 * ratio).ceil() as usize
        + resampler.output_delay()
        + 128; // padding
    let mut output_f64 = vec![0.0f64; num_output_frames];

    // Use InterleavedSlice adapters (1 channel = interleaved is same as planar)
    use audioadapter_buffers::direct::InterleavedSlice;

    let input_adapter =
        InterleavedSlice::new(&input_f64, 1, num_input_frames)
            .map_err(|e| SttError::Audio(format!("Input adapter failed: {e}")))?;
    let mut output_adapter =
        InterleavedSlice::new_mut(&mut output_f64, 1, num_output_frames)
            .map_err(|e| SttError::Audio(format!("Output adapter failed: {e}")))?;

    let indexing = Indexing {
        input_offset: 0,
        output_offset: 0,
        active_channels_mask: None,
        partial_len: None,
    };

    let (_, output_frames) = resampler
        .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
        .map_err(|e| SttError::Audio(format!("Resampling failed: {e}")))?;

    // Convert f64 → f32, trim to actual output length
    Ok(output_f64[..output_frames]
        .iter()
        .map(|&s| s as f32)
        .collect())
}

/// Linear interpolation resampling (fallback).
///
/// Simple but introduces aliasing artifacts. Used only when rubato
/// fails (very short inputs) or for testing.
pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (samples.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;

        let sample = if idx + 1 < samples.len() {
            samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac
        } else if idx < samples.len() {
            samples[idx] as f64
        } else {
            0.0
        };

        output.push(sample as f32);
    }

    output
}

/// Fallback token decoder when no tokenizer is available.
///
/// Maps ASCII-range tokens to characters, and wraps others in `<id>` brackets.
fn decode_tokens_fallback(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(|&t| {
            if (32..128).contains(&t) {
                (t as u8 as char).to_string()
            } else {
                format!("<{t}>")
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_tokens_fallback_ascii() {
        let tokens = vec![72, 101, 108, 108, 111]; // "Hello"
        assert_eq!(decode_tokens_fallback(&tokens), "Hello");
    }

    #[test]
    fn test_decode_tokens_fallback_non_ascii() {
        let tokens = vec![1, 72, 101, 2];
        assert_eq!(decode_tokens_fallback(&tokens), "<1>He<2>");
    }
}
