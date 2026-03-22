use std::path::Path;

use hf_hub::api::sync::Api;
use quill_mlx::Array;

use crate::error::{Result, VoicersError};

const DEFAULT_REPO: &str = "prince-canuma/Kokoro-82M";

/// Load a voice embedding from a local .safetensors file.
pub fn load_voice_from_file(path: &Path) -> Result<Array> {
    let tensors = Array::load_safetensors(path).map_err(|e| VoicersError::Weight(e.to_string()))?;

    // Voice files typically have a single tensor
    let voice = tensors
        .into_values()
        .next()
        .ok_or_else(|| VoicersError::Weight("Empty voice file".to_string()))?;

    // Ensure shape is (1, 256) for the model
    let voice = if voice.ndim() == 1 {
        voice.reshape(&[1, -1]).map_err(VoicersError::Mlx)?
    } else {
        voice
    };

    Ok(voice)
}

/// Load a voice embedding by name from HuggingFace Hub.
///
/// Voice files are stored as `voices/{name}.safetensors` in the model repo.
pub fn load_voice(voice_name: &str, repo_id: Option<&str>) -> Result<Array> {
    let repo_id = repo_id.unwrap_or(DEFAULT_REPO);
    let api = Api::new().map_err(|e| VoicersError::Hub(e.to_string()))?;
    let repo = api.model(repo_id.to_string());

    let voice_path = format!("voices/{}.safetensors", voice_name);
    let local_path = repo
        .get(&voice_path)
        .map_err(|e| VoicersError::Hub(format!("Voice '{}' not found: {}", voice_name, e)))?;

    load_voice_from_file(&local_path)
}
