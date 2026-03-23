//! Speech-to-text library backed by candle + Metal, using Whisper.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use voice_stt::{load_model, transcribe, TranscribeResult};
//!
//! let mut model = load_model("distil-whisper/distil-medium.en").unwrap();
//! let result = transcribe(&mut model, "audio.wav").unwrap();
//! println!("{}", result.text);
//! ```
//!
//! # Supported models
//!
//! Any Whisper or distil-whisper model on HuggingFace with safetensors weights.
//! Default: `distil-whisper/distil-medium.en` (English-only, fast, accurate).
//!
//! # Architecture
//!
//! Whisper uses mel-spectrogram preprocessing followed by a transformer
//! encoder-decoder. Audio is processed in 30-second chunks. For typical
//! voice commands (<30s), a single chunk suffices.

pub mod builtin;
pub mod error;

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub use error::{Result, SttError};
pub use tokenizers;

/// Result of a transcription.
#[derive(Debug, Clone)]
pub struct TranscribeResult {
    /// The transcribed text.
    pub text: String,
    /// Token IDs generated (including special tokens).
    pub tokens: Vec<u32>,
    /// Sample rate of the model input (always 16000 for Whisper).
    pub sample_rate: u32,
}

/// Loaded Whisper STT model ready for transcription.
pub struct WhisperModel {
    decoder: voice_whisper::WhisperDecoder,
}

/// Load a Whisper model from a HuggingFace repo or local path.
///
/// Creates a Metal GPU device and loads the model weights via mmap.
///
/// # Examples
///
/// ```rust,no_run
/// let mut model = voice_stt::load_model("distil-whisper/distil-medium.en").unwrap();
/// ```
pub fn load_model(path_or_repo: &str) -> Result<WhisperModel> {
    let device = Device::new_metal(0).map_err(|e| SttError::Model(e.to_string()))?;

    let (config_path, tokenizer_path, weights_path) = if Path::new(path_or_repo).exists() {
        let dir = PathBuf::from(path_or_repo);
        (
            dir.join("config.json"),
            dir.join("tokenizer.json"),
            dir.join("model.safetensors"),
        )
    } else {
        download_model(path_or_repo)?
    };

    let config_str = std::fs::read_to_string(&config_path)?;
    let config: voice_whisper::Config = serde_json::from_str(&config_str)?;

    let mel_filters = voice_whisper::load_mel_filters(&config).map_err(SttError::Model)?;

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| SttError::Tokenizer(e.to_string()))?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
            .map_err(|e| SttError::Weight(e.to_string()))?
    };

    let model = voice_whisper::Whisper::load(&vb, config.clone())
        .map_err(|e| SttError::Model(e.to_string()))?;

    // Determine language token for multilingual models
    let language_token = if builtin::is_multilingual(path_or_repo) {
        // Default to English for multilingual models
        tokenizer.token_to_id("<|en|>")
    } else {
        None
    };

    let decoder = voice_whisper::WhisperDecoder::new(
        model,
        config,
        tokenizer,
        mel_filters,
        device,
        language_token,
    )
    .map_err(|e| SttError::Model(e.to_string()))?;

    Ok(WhisperModel { decoder })
}

/// Load the tokenizer from a model directory or HuggingFace repo.
///
/// Whisper uses a HuggingFace fast tokenizer stored as `tokenizer.json`.
pub fn load_tokenizer(path_or_repo: &str) -> Result<tokenizers::Tokenizer> {
    let tokenizer_path = if Path::new(path_or_repo).exists() {
        PathBuf::from(path_or_repo).join("tokenizer.json")
    } else {
        let api = hf_hub::api::sync::Api::new().map_err(|e| SttError::Hub(e.to_string()))?;
        let repo = api.model(path_or_repo.to_string());
        repo.get("tokenizer.json")
            .map_err(|e| SttError::Hub(e.to_string()))?
    };

    if !tokenizer_path.exists() {
        return Err(SttError::Tokenizer(format!(
            "tokenizer.json not found in {}",
            tokenizer_path.display()
        )));
    }

    tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| SttError::Tokenizer(e.to_string()))
}

/// Transcribe an audio file.
///
/// Loads the audio file as WAV (16-bit PCM or 32-bit float), converts to
/// 16kHz mono f32, and runs greedy decoding.
pub fn transcribe(
    model: &mut WhisperModel,
    audio_path: impl AsRef<Path>,
) -> Result<TranscribeResult> {
    let samples = load_wav_as_f32(audio_path.as_ref())?;
    transcribe_audio(model, &samples, 16000)
}

/// Transcribe raw audio samples.
///
/// - `samples`: mono f32 audio samples
/// - `sample_rate`: sample rate of the input audio (will be resampled to 16kHz if different)
pub fn transcribe_audio(
    model: &mut WhisperModel,
    samples: &[f32],
    sample_rate: u32,
) -> Result<TranscribeResult> {
    let samples = if sample_rate != 16000 {
        resample(samples, sample_rate, 16000)
    } else {
        samples.to_vec()
    };

    let result = model
        .decoder
        .transcribe(&samples)
        .map_err(|e| SttError::Model(e.to_string()))?;

    Ok(TranscribeResult {
        text: result.text,
        tokens: result.tokens,
        sample_rate: 16000,
    })
}

/// Transcribe raw audio samples with a tokenizer for proper text decoding.
///
/// Note: The tokenizer parameter is accepted for API compatibility but
/// Whisper's tokenizer is loaded with the model. The provided tokenizer
/// is not used — decoding uses the model's built-in tokenizer.
pub fn transcribe_audio_with_tokenizer(
    model: &mut WhisperModel,
    samples: &[f32],
    sample_rate: u32,
    _tokenizer: &tokenizers::Tokenizer,
) -> Result<TranscribeResult> {
    transcribe_audio(model, samples, sample_rate)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Download model files from HuggingFace Hub.
/// Returns (config_path, tokenizer_path, weights_path).
fn download_model(repo_id: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let api = hf_hub::api::sync::Api::new().map_err(|e| SttError::Hub(e.to_string()))?;
    let repo = api.model(repo_id.to_string());

    let config_path = repo
        .get("config.json")
        .map_err(|e| SttError::Hub(e.to_string()))?;

    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| SttError::Hub(e.to_string()))?;

    let weights_path = repo
        .get("model.safetensors")
        .map_err(|e| SttError::Hub(e.to_string()))?;

    Ok((config_path, tokenizer_path, weights_path))
}

/// Load a WAV file and return mono f32 samples at 16kHz.
///
/// Handles both 16-bit integer and 32-bit float WAV formats.
/// Multi-channel audio is mixed down to mono by averaging.
pub(crate) fn load_wav_as_f32(path: &Path) -> Result<Vec<f32>> {
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
/// Uses a windowed sinc resampler with 128-tap filter for clean
/// anti-aliasing. Falls back to linear interpolation if rubato fails
/// (e.g. extremely short inputs).
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }

    match resample_sinc(samples, from_rate, to_rate) {
        Ok(resampled) => resampled,
        Err(_) => resample_linear(samples, from_rate, to_rate),
    }
}

/// Sinc-based resampling via rubato.
fn resample_sinc(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::{
        calculate_cutoff, Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters,
        SincInterpolationType, WindowFunction,
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

    let mut resampler =
        Async::<f64>::new_sinc(ratio, 1.1, &params, chunk_size, 1, FixedAsync::Input)
            .map_err(|e| SttError::Audio(format!("Resampler init failed: {e}")))?;

    let input_f64: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
    let num_input_frames = input_f64.len();

    let num_output_frames =
        (num_input_frames as f64 * ratio).ceil() as usize + resampler.output_delay() + 128;
    let mut output_f64 = vec![0.0f64; num_output_frames];

    use audioadapter_buffers::direct::InterleavedSlice;

    let input_adapter = InterleavedSlice::new(&input_f64, 1, num_input_frames)
        .map_err(|e| SttError::Audio(format!("Input adapter failed: {e}")))?;
    let mut output_adapter = InterleavedSlice::new_mut(&mut output_f64, 1, num_output_frames)
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

    Ok(output_f64[..output_frames]
        .iter()
        .map(|&s| s as f32)
        .collect())
}

/// Linear interpolation resampling (fallback).
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_resample_identity() {
        let sr = 16000u32;
        let freq = 440.0f32;
        let input: Vec<f32> = (0..sr as usize)
            .map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect();

        let output = resample_linear(&input, sr, sr);

        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6, "samples differ: {a} vs {b}");
        }
    }

    #[test]
    fn test_resample_downsample_length() {
        let input = vec![0.0f32; 100];
        let output = resample_linear(&input, 48000, 16000);
        assert!(
            (output.len() as i64 - 34).abs() <= 1,
            "expected ~34 samples, got {}",
            output.len()
        );
    }

    #[test]
    fn test_resample_upsample_length() {
        let input = vec![0.0f32; 100];
        let output = resample_linear(&input, 8000, 16000);
        assert!(
            (output.len() as i64 - 200).abs() <= 1,
            "expected ~200 samples, got {}",
            output.len()
        );
    }

    #[test]
    fn test_resample_empty() {
        let output = resample_linear(&[], 44100, 16000);
        assert!(output.is_empty());
    }

    #[test]
    fn test_resample_preserves_sine() {
        let sr_in = 48000u32;
        let sr_out = 16000u32;
        let freq = 440.0f32;
        let duration_samples = sr_in as usize;

        let input: Vec<f32> = (0..duration_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sr_in as f32).sin())
            .collect();

        let output = resample_linear(&input, sr_in, sr_out);

        let expected_len = sr_out as usize;
        assert!(
            (output.len() as i64 - expected_len as i64).abs() <= 1,
            "expected ~{expected_len} samples, got {}",
            output.len()
        );

        let rms = (output.iter().map(|s| s * s).sum::<f32>() / output.len() as f32).sqrt();
        assert!(rms > 0.5, "RMS of resampled sine is too low: {rms}");
    }

    #[test]
    fn test_wav_16bit_roundtrip() {
        let path = temp_wav_path("i16");
        let sample_rate = 16000u32;
        let i16_samples: Vec<i16> = vec![0, 16383, -16384, 32767, -32768, 1000, -1000];
        let expected_f32: Vec<f32> = i16_samples.iter().map(|&s| s as f32 / 32768.0).collect();

        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&path, spec).unwrap();
            for &s in &i16_samples {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }

        let loaded = load_wav_as_f32(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.len(), expected_f32.len());
        for (i, (got, want)) in loaded.iter().zip(expected_f32.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "sample {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_wav_32float_roundtrip() {
        let path = temp_wav_path("f32");
        let sample_rate = 16000u32;
        let f32_samples: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.123, -0.987];

        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer = hound::WavWriter::create(&path, spec).unwrap();
            for &s in &f32_samples {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }

        let loaded = load_wav_as_f32(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.len(), f32_samples.len());
        for (i, (got, want)) in loaded.iter().zip(f32_samples.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "sample {i}: got {got}, want {want}"
            );
        }
    }

    fn temp_wav_path(label: &str) -> std::path::PathBuf {
        let pid = std::process::id();
        let tid = std::thread::current().id();
        std::path::PathBuf::from(format!("/tmp/voice_test_{label}_{pid}_{tid:?}.wav"))
    }
}
