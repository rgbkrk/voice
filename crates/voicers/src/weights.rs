use std::collections::HashMap;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::Api;
use mlx_rs::Array;

use crate::config::ModelConfig;
use crate::error::{Result, VoicersError};
use crate::model::KokoroModel;

const DEFAULT_REPO: &str = "prince-canuma/Kokoro-82M";

/// Download a model from HuggingFace Hub (or use cached version).
pub fn download_model(repo_id: &str) -> Result<PathBuf> {
    let api = Api::new().map_err(|e| VoicersError::Hub(e.to_string()))?;
    let repo = api.model(repo_id.to_string());

    // Download config.json
    let config_path = repo
        .get("config.json")
        .map_err(|e| VoicersError::Hub(e.to_string()))?;

    // Download the model weights file
    let _weights_path = repo
        .get("kokoro-v1_0.safetensors")
        .map_err(|e| VoicersError::Hub(e.to_string()))?;

    // Return the directory containing config.json
    Ok(config_path
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf())
}

/// Load model config from a directory or HF repo.
pub fn load_config(path: &Path) -> Result<ModelConfig> {
    let config_path = path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: ModelConfig = serde_json::from_str(&config_str)?;
    Ok(config)
}

/// Find all safetensors files in a directory.
fn find_safetensors(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "safetensors") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// Check if a 3D array shape looks like it's already in MLX conv format.
/// Returns true if out_channels >= kH and out_channels >= kW and kH == kW.
fn check_array_shape(shape: &[i32]) -> bool {
    if shape.len() != 3 {
        return false;
    }
    let (out_ch, kh, kw) = (shape[0], shape[1], shape[2]);
    out_ch >= kh && out_ch >= kw && kh == kw
}

/// Sanitize weights from PyTorch format to our Rust module key format.
///
/// This handles:
/// - Removing position_ids
/// - Renaming .gamma/.beta to .weight/.bias for LayerNorm
/// - Combining bias_ih + bias_hh for LSTM
/// - Remapping LSTM keys (weight_ih_l0 -> forward.wx, etc.)
/// - Transposing conv weights where needed
pub fn sanitize_weights(
    weights: HashMap<String, Array>,
) -> Result<HashMap<String, Array>> {
    let mut sanitized = HashMap::new();
    let mut bias_ih_cache: HashMap<String, Array> = HashMap::new();
    let mut bias_hh_cache: HashMap<String, Array> = HashMap::new();

    for (key, value) in &weights {
        // Skip position_ids
        if key.contains("position_ids") {
            continue;
        }

        // Handle LSTM bias combining - collect first, combine later
        if key.ends_with(".bias_ih_l0")
            || key.ends_with(".bias_ih_l0_reverse")
            || key.ends_with(".bias_hh_l0")
            || key.ends_with(".bias_hh_l0_reverse")
        {
            if key.contains("bias_ih") {
                bias_ih_cache.insert(key.clone(), value.clone());
            } else {
                bias_hh_cache.insert(key.clone(), value.clone());
            }
            continue;
        }

        let new_key;
        let new_value;

        if key.ends_with(".gamma") {
            new_key = key.replace(".gamma", ".weight");
            new_value = value.clone();
        } else if key.ends_with(".beta") {
            new_key = key.replace(".beta", ".bias");
            new_value = value.clone();
        } else if key.ends_with(".weight_ih_l0") {
            let base = key.strip_suffix(".weight_ih_l0").unwrap();
            new_key = format!("{}.forward.wx", base);
            new_value = value.clone();
        } else if key.ends_with(".weight_hh_l0") {
            let base = key.strip_suffix(".weight_hh_l0").unwrap();
            new_key = format!("{}.forward.wh", base);
            new_value = value.clone();
        } else if key.ends_with(".weight_ih_l0_reverse") {
            let base = key.strip_suffix(".weight_ih_l0_reverse").unwrap();
            new_key = format!("{}.backward.wx", base);
            new_value = value.clone();
        } else if key.ends_with(".weight_hh_l0_reverse") {
            let base = key.strip_suffix(".weight_hh_l0_reverse").unwrap();
            new_key = format!("{}.backward.wh", base);
            new_value = value.clone();
        } else if key.contains("weight_v") {
            let shape = value.shape();
            if check_array_shape(shape) {
                new_key = key.clone();
                new_value = value.clone();
            } else {
                new_key = key.clone();
                new_value = value.transpose_axes(&[0, 2, 1]).unwrap_or_else(|_| value.clone());
            }
        } else if key.contains("F0_proj.weight") || key.contains("N_proj.weight") {
            new_key = key.clone();
            new_value = value.transpose_axes(&[0, 2, 1]).unwrap_or_else(|_| value.clone());
        } else if key.starts_with("decoder.") && key.contains("noise_convs") && key.ends_with(".weight") {
            new_key = key.clone();
            new_value = value.transpose_axes(&[0, 2, 1]).unwrap_or_else(|_| value.clone());
        } else {
            new_key = key.clone();
            new_value = value.clone();
        }

        sanitized.insert(new_key, new_value);
    }

    // Combine LSTM biases: bias = bias_ih + bias_hh
    for (ih_key, ih_val) in &bias_ih_cache {
        let hh_key = ih_key.replace("bias_ih", "bias_hh");
        if let Some(hh_val) = bias_hh_cache.get(&hh_key) {
            let combined = &ih_val.clone() + &hh_val.clone();
            let base_key;
            let direction;
            if ih_key.ends_with("_reverse") {
                let stripped = ih_key.strip_suffix(".bias_ih_l0_reverse").unwrap();
                base_key = stripped.to_string();
                direction = "backward";
            } else {
                let stripped = ih_key.strip_suffix(".bias_ih_l0").unwrap();
                base_key = stripped.to_string();
                direction = "forward";
            }
            sanitized.insert(format!("{}.{}.bias", base_key, direction), combined);
        }
    }

    Ok(sanitized)
}

/// Load a KokoroModel from a HuggingFace repo or local path.
pub fn load_model(path_or_repo: &str) -> Result<KokoroModel> {
    // Resolve path: local or download from HF
    let model_dir = if Path::new(path_or_repo).exists() {
        PathBuf::from(path_or_repo)
    } else {
        download_model(path_or_repo)?
    };

    // Load config
    let config = load_config(&model_dir)?;

    // Create model
    let mut model =
        KokoroModel::new(&config).map_err(|e| VoicersError::Weight(e.to_string()))?;

    // Load and sanitize weights
    let weight_files = find_safetensors(&model_dir)?;
    if weight_files.is_empty() {
        return Err(VoicersError::Weight(format!(
            "No safetensors files found in {:?}",
            model_dir
        )));
    }

    let mut all_weights = HashMap::new();
    for wf in &weight_files {
        let tensors = Array::load_safetensors(wf)
            .map_err(|e| VoicersError::Weight(e.to_string()))?;
        all_weights.extend(tensors);
    }

    let sanitized = sanitize_weights(all_weights)?;

    // Load weights into model by matching parameter keys
    {
        use mlx_rs::module::ModuleParameters;
        let mut params = model.parameters_mut().flatten();
        for (key, value) in sanitized {
            if let Some(param) = params.get_mut(&*key) {
                **param = value;
            }
        }
    }

    // Evaluate loaded params
    use mlx_rs::module::ModuleParametersExt;
    model
        .eval()
        .map_err(|e| VoicersError::Weight(e.to_string()))?;

    Ok(model)
}

/// Load model with default Kokoro repo.
pub fn load_default_model() -> Result<KokoroModel> {
    load_model(DEFAULT_REPO)
}
