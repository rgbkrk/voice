pub mod builtin;
pub mod config;
pub mod error;
pub mod model;
pub mod voice;
pub mod weights;

use std::path::Path;

use mlx_rs::Array;

pub use builtin::BUILTIN_VOICES;
pub use config::ModelConfig;
pub use error::{Result, VoicersError};
pub use model::KokoroModel;

/// Load a Kokoro TTS model from a HuggingFace repo or local path.
///
/// Uses `~/.cache/huggingface/hub/` for caching.
pub fn load_model(path_or_repo: &str) -> Result<KokoroModel> {
    weights::load_model(path_or_repo)
}

/// Load a voice embedding by name.
///
/// Builtin voices (af_heart, af_bella, af_sarah, af_sky, am_michael, am_adam,
/// bf_emma) are embedded in the binary and load instantly. All other voices
/// are fetched from HuggingFace Hub.
pub fn load_voice(voice_name: &str, repo_id: Option<&str>) -> Result<Array> {
    if let Some(result) = builtin::load_builtin_voice(voice_name) {
        return result;
    }
    voice::load_voice(voice_name, repo_id)
}

/// Generate audio from phonemes using a loaded model and voice.
///
/// Returns a 1D audio array at 24kHz sample rate.
pub fn generate(
    model: &mut KokoroModel,
    phonemes: &str,
    voice: &Array,
    speed: f32,
) -> Result<Array> {
    model
        .generate(phonemes, voice, speed)
        .map_err(VoicersError::Mlx)
}

/// Save audio array to a WAV file (24kHz, 32-bit float).
pub fn save_wav(audio: &Array, path: &Path, sample_rate: u32) -> Result<()> {
    audio.eval().map_err(VoicersError::Mlx)?;
    let samples: &[f32] = audio.as_slice();

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
