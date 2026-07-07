//! Speech-to-text library backed by candle, using Whisper.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use voice_stt::{load_model, transcribe, TranscribeResult};
//!
//! let mut model = load_model("distil-whisper/distil-medium.en").unwrap();
//! let result = transcribe(&mut model, "audio.ogg").unwrap();
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
use std::process::Command;

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

/// Decoded mono audio samples suitable for STT.
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Mono f32 samples in the range `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Sample rate of `samples`.
    pub sample_rate: u32,
}

/// Loaded Whisper STT model ready for transcription.
pub struct WhisperModel {
    decoder: voice_whisper::WhisperDecoder,
}

/// Supported speech-to-text backend families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttBackend {
    Whisper,
    Voxtral,
}

/// Backend-neutral STT model.
///
/// Existing callers can continue using `WhisperModel` directly via
/// `load_model`; this enum is for opt-in backend selection.
pub enum SttModel {
    Whisper(Box<WhisperModel>),
    Voxtral(Box<VoxtralRealtimeSttModel>),
}

/// Loaded Voxtral Realtime STT model ready for transcription.
pub struct VoxtralRealtimeSttModel {
    transcriber: voice_voxtral::VoxtralRealtimeTranscriber,
    options: voice_voxtral::VoxtralRealtimeTranscriptionOptions,
}

impl SttBackend {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "whisper" => Ok(Self::Whisper),
            "voxtral" | "voxtral-realtime" => Ok(Self::Voxtral),
            other => Err(SttError::Model(format!(
                "unsupported STT backend {other:?}; expected whisper or voxtral"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Whisper => "whisper",
            Self::Voxtral => "voxtral",
        }
    }
}

impl SttModel {
    pub fn backend(&self) -> SttBackend {
        match self {
            Self::Whisper(_) => SttBackend::Whisper,
            Self::Voxtral(_) => SttBackend::Voxtral,
        }
    }

    pub fn transcribe_audio(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<TranscribeResult> {
        match self {
            Self::Whisper(model) => transcribe_audio(model.as_mut(), samples, sample_rate),
            Self::Voxtral(model) => model.transcribe_audio(samples, sample_rate),
        }
    }

    pub fn set_max_new_tokens(&mut self, max_new_tokens: usize) {
        if let Self::Voxtral(model) = self {
            model.set_max_new_tokens(max_new_tokens);
        }
    }
}

impl VoxtralRealtimeSttModel {
    pub fn set_max_new_tokens(&mut self, max_new_tokens: usize) {
        self.options.max_new_tokens = max_new_tokens;
    }

    pub fn transcribe_audio(&self, samples: &[f32], sample_rate: u32) -> Result<TranscribeResult> {
        let samples = if sample_rate != voice_voxtral::REALTIME_SAMPLE_RATE {
            resample(samples, sample_rate, voice_voxtral::REALTIME_SAMPLE_RATE)
        } else {
            samples.to_vec()
        };
        let result = self
            .transcriber
            .transcribe_16khz(&samples, self.options)
            .map_err(|e| SttError::Model(e.to_string()))?;
        let tokens = result
            .tokens
            .into_iter()
            .map(|token| {
                u32::try_from(token)
                    .map_err(|_| SttError::Model(format!("Voxtral token id {token} exceeds u32")))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(TranscribeResult {
            text: result.text,
            tokens,
            sample_rate: voice_voxtral::REALTIME_SAMPLE_RATE,
        })
    }
}

/// Return the default inference device for STT.
///
/// On macOS this preserves the existing Apple Silicon Metal path. All other
/// builds use Candle's CPU backend, which keeps Whisper usable on Linux hosts
/// with no GPU.
pub fn default_stt_device() -> Result<Device> {
    #[cfg(target_os = "macos")]
    {
        Device::new_metal(0).map_err(|e| SttError::Model(e.to_string()))
    }

    #[cfg(not(target_os = "macos"))]
    {
        Ok(Device::Cpu)
    }
}

/// Load a Whisper model from a HuggingFace repo or local path.
///
/// Creates the default STT device and loads the model weights via mmap.
///
/// # Examples
///
/// ```rust,no_run
/// let mut model = voice_stt::load_model("distil-whisper/distil-medium.en").unwrap();
/// ```
pub fn load_model(path_or_repo: &str) -> Result<WhisperModel> {
    let device = default_stt_device()?;
    load_model_on_device(path_or_repo, device)
}

/// Load a Whisper model on an explicitly supplied Candle device.
pub fn load_model_on_device(path_or_repo: &str, device: Device) -> Result<WhisperModel> {
    // Use embedded config/tokenizer when available (zero network fetch).
    // Only the weights need downloading from HuggingFace.
    let config: voice_whisper::Config = if let Some(result) = builtin::config_for_repo(path_or_repo)
    {
        result?
    } else if Path::new(path_or_repo).exists() {
        let config_str = std::fs::read_to_string(Path::new(path_or_repo).join("config.json"))?;
        serde_json::from_str(&config_str)?
    } else {
        let api = hf_hub::api::sync::Api::new().map_err(|e| SttError::Hub(e.to_string()))?;
        let repo = api.model(path_or_repo.to_string());
        let config_path = repo
            .get("config.json")
            .map_err(|e| SttError::Hub(e.to_string()))?;
        let config_str = std::fs::read_to_string(config_path)?;
        serde_json::from_str(&config_str)?
    };

    let tokenizer: tokenizers::Tokenizer =
        if let Some(result) = builtin::tokenizer_for_repo(path_or_repo) {
            result?
        } else if Path::new(path_or_repo).exists() {
            tokenizers::Tokenizer::from_file(Path::new(path_or_repo).join("tokenizer.json"))
                .map_err(|e| SttError::Tokenizer(e.to_string()))?
        } else {
            let api = hf_hub::api::sync::Api::new().map_err(|e| SttError::Hub(e.to_string()))?;
            let repo = api.model(path_or_repo.to_string());
            let tokenizer_path = repo
                .get("tokenizer.json")
                .map_err(|e| SttError::Hub(e.to_string()))?;
            tokenizers::Tokenizer::from_file(tokenizer_path)
                .map_err(|e| SttError::Tokenizer(e.to_string()))?
        };

    let mel_filters = voice_whisper::load_mel_filters(&config).map_err(SttError::Model)?;

    // Only the weights need a network fetch (or local path)
    let weights_path = if Path::new(path_or_repo).exists() {
        PathBuf::from(path_or_repo).join("model.safetensors")
    } else {
        download_weights(path_or_repo)?
    };

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
            .map_err(|e| SttError::Weight(e.to_string()))?
    };

    let model = voice_whisper::Whisper::load(&vb, config.clone())
        .map_err(|e| SttError::Model(e.to_string()))?;

    let language_token = if builtin::is_multilingual(path_or_repo) {
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

pub fn default_model_for_backend(backend: SttBackend) -> &'static str {
    match backend {
        SttBackend::Whisper => builtin::DEFAULT_MODEL_REPO,
        SttBackend::Voxtral => voice_voxtral::REALTIME_DEFAULT_REPO,
    }
}

pub fn infer_backend_for_model(path_or_repo: &str) -> SttBackend {
    let value = path_or_repo.to_ascii_lowercase();
    if value.contains("voxtral") && (value.contains("realtime") || value.contains("2602")) {
        SttBackend::Voxtral
    } else {
        SttBackend::Whisper
    }
}

pub fn resolve_backend_and_model(
    backend: Option<SttBackend>,
    path_or_repo: Option<&str>,
) -> (SttBackend, String) {
    match (backend, path_or_repo) {
        (Some(backend), Some(path_or_repo)) => (backend, path_or_repo.to_string()),
        (Some(backend), None) => (backend, default_model_for_backend(backend).to_string()),
        (None, Some(path_or_repo)) => (
            infer_backend_for_model(path_or_repo),
            path_or_repo.to_string(),
        ),
        (None, None) => (
            SttBackend::Whisper,
            default_model_for_backend(SttBackend::Whisper).to_string(),
        ),
    }
}

pub fn load_backend_model(backend: SttBackend, path_or_repo: &str) -> Result<SttModel> {
    let device = default_stt_device()?;
    load_backend_model_on_device(backend, path_or_repo, device)
}

pub fn load_backend_model_on_device(
    backend: SttBackend,
    path_or_repo: &str,
    device: Device,
) -> Result<SttModel> {
    match backend {
        SttBackend::Whisper => load_model_on_device(path_or_repo, device)
            .map(|model| SttModel::Whisper(Box::new(model))),
        SttBackend::Voxtral => load_voxtral_realtime_model_on_device(path_or_repo, device)
            .map(|model| SttModel::Voxtral(Box::new(model))),
    }
}

fn load_voxtral_realtime_model_on_device(
    path_or_repo: &str,
    device: Device,
) -> Result<VoxtralRealtimeSttModel> {
    let model = voice_voxtral::VoxtralRealtimeModel::load(path_or_repo)
        .map_err(|e| SttError::Model(e.to_string()))?;
    let delay_tokens = model
        .default_delay_tokens()
        .map_err(|e| SttError::Model(e.to_string()))?;
    let dtype = default_voxtral_dtype(&device);
    let transcriber = model
        .load_transcriber(dtype, &device)
        .map_err(|e| SttError::Model(e.to_string()))?;
    Ok(VoxtralRealtimeSttModel {
        transcriber,
        options: voice_voxtral::VoxtralRealtimeTranscriptionOptions {
            delay_tokens,
            max_new_tokens: usize::MAX,
        },
    })
}

fn default_voxtral_dtype(device: &Device) -> DType {
    match device {
        Device::Cpu => DType::F32,
        _ => DType::F16,
    }
}

/// Load the tokenizer from a model directory or HuggingFace repo.
///
/// Whisper uses a HuggingFace fast tokenizer stored as `tokenizer.json`.
pub fn load_tokenizer(path_or_repo: &str) -> Result<tokenizers::Tokenizer> {
    // Use embedded tokenizer when available
    if let Some(result) = builtin::tokenizer_for_repo(path_or_repo) {
        return result;
    }

    let tokenizer_path = if Path::new(path_or_repo).exists() {
        PathBuf::from(path_or_repo).join("tokenizer.json")
    } else {
        let api = hf_hub::api::sync::Api::new().map_err(|e| SttError::Hub(e.to_string()))?;
        let repo = api.model(path_or_repo.to_string());
        repo.get("tokenizer.json")
            .map_err(|e| SttError::Hub(e.to_string()))?
    };

    tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| SttError::Tokenizer(e.to_string()))
}

/// Transcribe an audio file.
///
/// Loads WAV directly. If the input is another container/codec such as
/// Ogg/Opus, falls back to `ffmpeg` and decodes to mono f32 audio before
/// running greedy decoding.
pub fn transcribe(
    model: &mut WhisperModel,
    audio_path: impl AsRef<Path>,
) -> Result<TranscribeResult> {
    let audio = load_audio_file(audio_path.as_ref())?;
    transcribe_audio(model, &audio.samples, audio.sample_rate)
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

/// Download only model weights from HuggingFace Hub.
///
/// Config and tokenizer are embedded for known models, so only the
/// weights safetensors file needs to be fetched.
fn download_weights(repo_id: &str) -> Result<PathBuf> {
    let api = hf_hub::api::sync::Api::new().map_err(|e| SttError::Hub(e.to_string()))?;
    let repo = api.model(repo_id.to_string());

    repo.get("model.safetensors")
        .map_err(|e| SttError::Hub(e.to_string()))
}

/// Load an audio file and return mono f32 samples.
///
/// WAV input is decoded in-process. Other formats are decoded with `ffmpeg`
/// into 16 kHz mono float PCM so WhatsApp-ready Ogg/Opus files can be
/// transcribed without a manual conversion step.
pub fn load_audio_file(path: &Path) -> Result<AudioData> {
    match load_wav_audio_file(path) {
        Ok(audio) => Ok(audio),
        Err(wav_error) => decode_audio_with_ffmpeg(path).map_err(|ffmpeg_error| {
            SttError::Audio(format!(
                "Failed to decode {} as WAV ({wav_error}); ffmpeg fallback also failed: {ffmpeg_error}",
                path.display()
            ))
        }),
    }
}

fn load_wav_audio_file(path: &Path) -> Result<AudioData> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| SttError::Audio(format!("Failed to open {}: {e}", path.display())))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;

    if spec.sample_rate == 0 {
        return Err(SttError::Audio("Invalid WAV: sample rate is 0".into()));
    }
    if channels == 0 {
        return Err(SttError::Audio("Invalid WAV: channel count is 0".into()));
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            if bits == 0 || bits > 32 {
                return Err(SttError::Audio(format!(
                    "Unsupported bits_per_sample {bits}; expected 1..=32"
                )));
            }
            let max_val = (1u32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| SttError::Audio(format!("Failed to read WAV samples: {e}")))?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| SttError::Audio(format!("Failed to read WAV samples: {e}")))?,
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

    Ok(AudioData {
        samples: mono,
        sample_rate: spec.sample_rate,
    })
}

fn decode_audio_with_ffmpeg(path: &Path) -> Result<AudioData> {
    let output = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(path)
        .arg("-f")
        .arg("f32le")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("pipe:1")
        .output()
        .map_err(|e| SttError::Audio(format!("spawn ffmpeg: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(SttError::Audio(format!(
            "ffmpeg exited with {}: {}",
            output.status,
            stderr.trim()
        )));
    }

    if output.stdout.is_empty() {
        return Err(SttError::Audio("ffmpeg produced no audio samples".into()));
    }
    if output.stdout.len() % std::mem::size_of::<f32>() != 0 {
        return Err(SttError::Audio(format!(
            "ffmpeg produced {} bytes, not a whole number of f32 samples",
            output.stdout.len()
        )));
    }

    let samples = output
        .stdout
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(AudioData {
        samples,
        sample_rate: 16000,
    })
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

    use rubato::audioadapter_buffers::direct::InterleavedSlice;

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
    use candle_core::Device;
    use std::f32::consts::PI;

    #[test]
    fn test_default_stt_device_is_cpu_on_non_macos() {
        let device = default_stt_device().unwrap();

        #[cfg(target_os = "macos")]
        assert!(
            matches!(device, Device::Metal(_)),
            "macOS should keep using Metal by default"
        );

        #[cfg(not(target_os = "macos"))]
        assert!(
            matches!(device, Device::Cpu),
            "non-macOS STT should default to CPU"
        );
    }

    #[test]
    fn parses_stt_backends() {
        assert_eq!(SttBackend::parse("whisper").unwrap(), SttBackend::Whisper);
        assert_eq!(SttBackend::parse("voxtral").unwrap(), SttBackend::Voxtral);
        assert_eq!(
            SttBackend::parse("voxtral-realtime").unwrap(),
            SttBackend::Voxtral
        );
        assert!(SttBackend::parse("kokoro").is_err());
        assert_eq!(SttBackend::Whisper.as_str(), "whisper");
        assert_eq!(SttBackend::Voxtral.as_str(), "voxtral");
    }

    #[test]
    fn resolves_default_model_for_backend() {
        assert_eq!(
            default_model_for_backend(SttBackend::Whisper),
            builtin::DEFAULT_MODEL_REPO
        );
        assert_eq!(
            default_model_for_backend(SttBackend::Voxtral),
            voice_voxtral::REALTIME_DEFAULT_REPO
        );
    }

    #[test]
    fn infers_backend_from_model_name() {
        assert_eq!(
            infer_backend_for_model("mistralai/Voxtral-Mini-4B-Realtime-2602"),
            SttBackend::Voxtral
        );
        assert_eq!(
            infer_backend_for_model("/models/voxtral-realtime"),
            SttBackend::Voxtral
        );
        assert_eq!(
            infer_backend_for_model("distil-whisper/distil-large-v3.5"),
            SttBackend::Whisper
        );
    }

    #[test]
    fn resolves_backend_and_model_together() {
        assert_eq!(
            resolve_backend_and_model(None, None),
            (SttBackend::Whisper, builtin::DEFAULT_MODEL_REPO.to_string())
        );
        assert_eq!(
            resolve_backend_and_model(Some(SttBackend::Voxtral), None),
            (
                SttBackend::Voxtral,
                voice_voxtral::REALTIME_DEFAULT_REPO.to_string()
            )
        );
        assert_eq!(
            resolve_backend_and_model(None, Some("mistralai/Voxtral-Mini-4B-Realtime-2602")),
            (
                SttBackend::Voxtral,
                "mistralai/Voxtral-Mini-4B-Realtime-2602".to_string()
            )
        );
        assert_eq!(
            resolve_backend_and_model(
                Some(SttBackend::Whisper),
                Some("mistralai/Voxtral-Mini-4B-Realtime-2602")
            ),
            (
                SttBackend::Whisper,
                "mistralai/Voxtral-Mini-4B-Realtime-2602".to_string()
            )
        );
    }

    #[test]
    #[ignore = "downloads Whisper weights and runs CPU inference"]
    fn test_cpu_model_transcribes_silence_smoke() {
        let mut model =
            load_model_on_device("distil-whisper/distil-medium.en", Device::Cpu).unwrap();
        let samples = vec![0.0f32; 16_000];
        let result = transcribe_audio(&mut model, &samples, 16_000).unwrap();
        assert_eq!(result.sample_rate, 16_000);
    }

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

        let loaded = load_audio_file(&path).unwrap().samples;
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

        let loaded = load_audio_file(&path).unwrap().samples;
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.len(), f32_samples.len());
        for (i, (got, want)) in loaded.iter().zip(f32_samples.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "sample {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_load_audio_file_preserves_wav_sample_rate() {
        let path = temp_wav_path("audio_file_wav");
        let sample_rate = 24_000u32;
        let f32_samples: Vec<f32> = vec![0.0, 0.25, -0.25, 0.5, -0.5];

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

        let loaded = load_audio_file(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.sample_rate, sample_rate);
        assert_eq!(loaded.samples.len(), f32_samples.len());
        for (i, (got, want)) in loaded.samples.iter().zip(f32_samples.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "sample {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn test_load_audio_file_decodes_ogg_opus_with_ffmpeg() {
        if !command_available("ffmpeg") {
            eprintln!("skipping Ogg/Opus decode test because ffmpeg is not on PATH");
            return;
        }

        let wav_path = temp_wav_path("ogg_source");
        let ogg_path = temp_audio_path("ogg_opus", "ogg");
        let sample_rate = 24_000u32;
        let samples: Vec<f32> = (0..sample_rate / 4)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.25)
            .collect();

        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
            for &sample in &samples {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }

        let encode = Command::new("ffmpeg")
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-y")
            .arg("-i")
            .arg(&wav_path)
            .arg("-ac")
            .arg("1")
            .arg("-ar")
            .arg("48000")
            .arg("-c:a")
            .arg("libopus")
            .arg(&ogg_path)
            .output()
            .unwrap();

        if !encode.status.success() {
            eprintln!(
                "skipping Ogg/Opus decode test because ffmpeg encode failed: {}",
                String::from_utf8_lossy(&encode.stderr)
            );
            let _ = std::fs::remove_file(&wav_path);
            let _ = std::fs::remove_file(&ogg_path);
            return;
        }

        let loaded = load_audio_file(&ogg_path).unwrap();
        let _ = std::fs::remove_file(&wav_path);
        let _ = std::fs::remove_file(&ogg_path);

        assert_eq!(loaded.sample_rate, 16_000);
        assert!(!loaded.samples.is_empty());
        let rms = (loaded.samples.iter().map(|s| s * s).sum::<f32>() / loaded.samples.len() as f32)
            .sqrt();
        assert!(rms > 0.01, "decoded Ogg/Opus RMS is too low: {rms}");
    }

    fn temp_wav_path(label: &str) -> std::path::PathBuf {
        temp_audio_path(label, "wav")
    }

    fn temp_audio_path(label: &str, ext: &str) -> std::path::PathBuf {
        let pid = std::process::id();
        let tid = std::thread::current().id();
        std::path::PathBuf::from(format!("/tmp/voice_test_{label}_{pid}_{tid:?}.{ext}"))
    }

    fn command_available(command: &str) -> bool {
        Command::new(command).arg("-version").output().is_ok()
    }
}
