pub mod builtin;
pub mod catalog;
pub mod config;
pub mod error;

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;

pub use config::ModelConfig;
pub use error::{Result, VoicersError};

const DEFAULT_REPO: &str = "prince-canuma/Kokoro-82M";

/// Loaded Kokoro TTS model ready for generation.
pub struct KokoroModel {
    model: kokoro_candle::KModel,
    config: kokoro_candle::ModelConfig,
    device: Device,
    pub sample_rate: u32,
}

/// Load a Kokoro TTS model from a HuggingFace repo or local path.
pub fn load_model(path_or_repo: &str) -> Result<KokoroModel> {
    let device = Device::new_metal(0).map_err(|e| VoicersError::Model(e.to_string()))?;

    let (config_path, weights_path) = if Path::new(path_or_repo).exists() {
        let dir = PathBuf::from(path_or_repo);
        (dir.join("config.json"), find_safetensors(&dir)?)
    } else {
        let api = Api::new().map_err(|e| VoicersError::Hub(e.to_string()))?;
        let repo = api.model(path_or_repo.to_string());
        let config = repo
            .get("config.json")
            .map_err(|e| VoicersError::Hub(e.to_string()))?;
        let weights = repo
            .get("kokoro-v1_0.safetensors")
            .map_err(|e| VoicersError::Hub(e.to_string()))?;
        (config, weights)
    };

    let config_str = std::fs::read_to_string(&config_path)?;
    let config: kokoro_candle::ModelConfig = serde_json::from_str(&config_str)?;

    let weights_data = std::fs::read(&weights_path)?;
    let vb = VarBuilder::from_buffered_safetensors(weights_data, DType::F32, &device)
        .map_err(|e| VoicersError::Model(e.to_string()))?;

    let model =
        kokoro_candle::KModel::load(&config, vb).map_err(|e| VoicersError::Model(e.to_string()))?;

    let sample_rate = config.sample_rate;

    Ok(KokoroModel {
        model,
        config,
        device,
        sample_rate,
    })
}

impl KokoroModel {
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Load a voice embedding using this model's device.
    pub fn load_voice(&self, voice_name: &str, repo_id: Option<&str>) -> Result<Tensor> {
        load_voice(voice_name, repo_id, &self.device)
    }
}

/// Load a voice embedding by name, using the model's device.
pub fn load_voice(voice_name: &str, repo_id: Option<&str>, device: &Device) -> Result<Tensor> {
    // Try builtin first
    if let Some(data) = builtin::get_builtin_voice_bytes(voice_name) {
        return load_voice_from_bytes(data, device);
    }

    // Download from HF
    let repo_id = repo_id.unwrap_or(DEFAULT_REPO);
    let api = Api::new().map_err(|e| VoicersError::Hub(e.to_string()))?;
    let repo = api.model(repo_id.to_string());
    let voice_path = repo
        .get(&format!("voices/{voice_name}.safetensors"))
        .map_err(|e| VoicersError::Hub(e.to_string()))?;

    let data = std::fs::read(&voice_path)?;
    load_voice_from_bytes(&data, device)
}

fn load_voice_from_bytes(data: &[u8], device: &Device) -> Result<Tensor> {
    let tensors = safetensors::SafeTensors::deserialize(data)
        .map_err(|e| VoicersError::Model(e.to_string()))?;
    let (_, view) = tensors
        .iter()
        .next()
        .ok_or_else(|| VoicersError::Model("empty voice file".into()))?;

    let shape: Vec<usize> = view.shape().to_vec();
    let f32_data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let tensor = Tensor::from_vec(f32_data, shape.as_slice(), device)
        .map_err(|e| VoicersError::Model(e.to_string()))?;

    // Voice files have shape [N, 1, 256] — a "voice pack" with per-length embeddings.
    // Return the full pack; the generate function selects by phoneme length.
    Ok(tensor)
}

/// Generate audio from phonemes using a loaded model and voice.
///
/// Returns f32 audio samples at 24kHz.
pub fn generate(
    model: &mut KokoroModel,
    phonemes: &str,
    voice: &Tensor,
    speed: f32,
) -> Result<Vec<f32>> {
    let vocab = &model.config.vocab;

    // Convert phonemes to token IDs (character by character)
    let input_ids: Vec<i64> = phonemes
        .chars()
        .filter_map(|c| vocab.get(&c.to_string()).copied())
        .collect();

    if input_ids.is_empty() {
        return Err(VoicersError::Model("no valid tokens from phonemes".into()));
    }

    // Select the right voice embedding from the pack based on token count.
    // Voice packs have shape [N, 1, 256] where index = token_count - 1.
    // Matches voice (MLX): `ref_s.index(input_ids.len() - 1)`
    // Matches hexgrad KPipeline: `pack[len(ps) - 1]`
    let ref_s = if voice.dims().len() == 3 {
        let pack_len = voice
            .dim(0)
            .map_err(|e| VoicersError::Model(e.to_string()))?;
        let idx = (input_ids.len() - 1).min(pack_len - 1);
        voice
            .i(idx)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| VoicersError::Model(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| VoicersError::Model(e.to_string()))?
    } else {
        voice.clone()
    };

    let audio = model
        .model
        .forward(&input_ids, &ref_s, speed, &model.device)
        .map_err(|e| VoicersError::Model(e.to_string()))?;

    let samples = audio
        .to_vec1::<f32>()
        .map_err(|e| VoicersError::Model(e.to_string()))?;

    Ok(samples)
}

/// Save audio samples to a WAV file (24kHz, 32-bit float).
pub fn save_wav(samples: &[f32], path: &Path, sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| VoicersError::Io(std::io::Error::other(e)))?;
    for &sample in samples {
        writer
            .write_sample(sample)
            .map_err(|e| VoicersError::Io(std::io::Error::other(e)))?;
    }
    writer
        .finalize()
        .map_err(|e| VoicersError::Io(std::io::Error::other(e)))?;
    Ok(())
}

fn find_safetensors(dir: &Path) -> Result<PathBuf> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "safetensors")
            && !path.to_string_lossy().contains("voices/")
        {
            return Ok(path);
        }
    }
    Err(VoicersError::Model(format!("no safetensors in {:?}", dir)))
}
