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

/// Remap a weight key from PyTorch/safetensors naming to Rust module naming.
///
/// Handles all systematic naming differences between the original model and
/// our Rust module field names.
fn remap_key(key: &str) -> String {
    let mut k = key.to_string();

    // LayerNorm case: LayerNorm -> layer_norm
    k = k.replace("LayerNorm", "layer_norm");

    // .gamma/.beta -> .weight/.bias (old LayerNorm naming)
    if k.ends_with(".gamma") {
        k = k[..k.len() - 6].to_string() + ".weight";
    } else if k.ends_with(".beta") {
        k = k[..k.len() - 5].to_string() + ".bias";
    }

    // Decoder conv naming: F0_conv -> f0_conv, N_conv -> n_conv
    k = k.replace("decoder.F0_conv", "decoder.f0_conv");
    k = k.replace("decoder.N_conv", "decoder.n_conv");

    // Predictor field naming: F0 -> f0_blocks, N -> n_blocks, F0_proj -> f0_proj, N_proj -> n_proj
    k = k.replace("predictor.F0_proj", "predictor.f0_proj");
    k = k.replace("predictor.N_proj", "predictor.n_proj");
    // Must come after F0_proj/N_proj to avoid partial matches
    k = k.replace("predictor.F0.", "predictor.f0_blocks.");
    k = k.replace("predictor.N.", "predictor.n_blocks.");

    // TextEncoder CNN: cnn.N.0. -> cnn.N.conv., cnn.N.1. -> cnn.N.norm.
    if k.starts_with("text_encoder.cnn.") {
        // Pattern: text_encoder.cnn.{idx}.0.{rest} -> text_encoder.cnn.{idx}.conv.{rest}
        //          text_encoder.cnn.{idx}.1.{rest} -> text_encoder.cnn.{idx}.norm.{rest}
        let parts: Vec<&str> = k.splitn(4, '.').collect(); // ["text_encoder", "cnn", idx, rest]
        if parts.len() == 4 {
            let rest = parts[3];
            if let Some(stripped) = rest.strip_prefix("0.") {
                k = format!("text_encoder.cnn.{}.conv.{}", parts[2], stripped);
            } else if let Some(stripped) = rest.strip_prefix("1.") {
                k = format!("text_encoder.cnn.{}.norm.{}", parts[2], stripped);
            }
        }
    }

    // mlx-rs Linear wrapping: duration_proj.linear_layer. -> duration_proj.
    k = k.replace("duration_proj.linear_layer.", "duration_proj.");

    k
}

/// Sanitize weights from PyTorch format to our Rust module key format.
///
/// This handles:
/// - Removing position_ids
/// - Key renaming (CamelCase -> snake_case, field name differences)
/// - Combining bias_ih + bias_hh for LSTM
/// - Remapping LSTM keys (weight_ih_l0 -> forward_lstm.wx, etc.)
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

        if key.ends_with(".weight_ih_l0") {
            let base = remap_key(key.strip_suffix(".weight_ih_l0").unwrap());
            new_key = format!("{}.forward_lstm.wx", base);
            new_value = value.clone();
        } else if key.ends_with(".weight_hh_l0") {
            let base = remap_key(key.strip_suffix(".weight_hh_l0").unwrap());
            new_key = format!("{}.forward_lstm.wh", base);
            new_value = value.clone();
        } else if key.ends_with(".weight_ih_l0_reverse") {
            let base = remap_key(key.strip_suffix(".weight_ih_l0_reverse").unwrap());
            new_key = format!("{}.backward_lstm.wx", base);
            new_value = value.clone();
        } else if key.ends_with(".weight_hh_l0_reverse") {
            let base = remap_key(key.strip_suffix(".weight_hh_l0_reverse").unwrap());
            new_key = format!("{}.backward_lstm.wh", base);
            new_value = value.clone();
        } else if key.contains("weight_v") {
            let shape = value.shape();
            new_key = remap_key(key);
            if check_array_shape(shape) {
                new_value = value.clone();
            } else {
                new_value = value.transpose_axes(&[0, 2, 1]).unwrap_or_else(|_| value.clone());
            }
        } else if key.contains("F0_proj.weight") || key.contains("N_proj.weight") {
            new_key = remap_key(key);
            new_value = value.transpose_axes(&[0, 2, 1]).unwrap_or_else(|_| value.clone());
        } else if key.starts_with("decoder.") && key.contains("noise_convs") && key.ends_with(".weight") {
            new_key = remap_key(key);
            new_value = value.transpose_axes(&[0, 2, 1]).unwrap_or_else(|_| value.clone());
        } else {
            new_key = remap_key(key);
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
                base_key = remap_key(stripped);
                direction = "backward_lstm";
            } else {
                let stripped = ih_key.strip_suffix(".bias_ih_l0").unwrap();
                base_key = remap_key(stripped);
                direction = "forward_lstm";
            }
            sanitized.insert(format!("{}.{}.bias", base_key, direction), combined);
        }
    }

    Ok(sanitized)
}

/// Load alpha (snake1d activation) parameters that aren't exposed via #[param].
///
/// These are Vec<Array> fields on AdaINResBlock1 structs in:
/// - decoder.generator.resblocks[i].alpha{1,2}[j]
/// - decoder.generator.noise_res[i].alpha{1,2}[j]
fn load_alpha_params(model: &mut KokoroModel, weights: &HashMap<String, Array>) -> std::collections::HashSet<String> {
    let mut loaded = std::collections::HashSet::new();
    for (key, value) in weights {
        // Pattern: decoder.generator.resblocks.{i}.alpha{1|2}.{j}
        if let Some(rest) = key.strip_prefix("decoder.generator.resblocks.") {
            if let Some((idx_str, alpha_rest)) = rest.split_once('.') {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if let Some((alpha_name, j_str)) = alpha_rest.split_once('.') {
                        if let Ok(j) = j_str.parse::<usize>() {
                            if idx < model.decoder.generator.resblocks.len() {
                                let block = &mut model.decoder.generator.resblocks[idx];
                                match alpha_name {
                                    "alpha1" if j < block.alpha1.len() => {
                                        block.alpha1[j] = value.clone();
                                        loaded.insert(key.clone());
                                    }
                                    "alpha2" if j < block.alpha2.len() => {
                                        block.alpha2[j] = value.clone();
                                        loaded.insert(key.clone());
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }
        // Pattern: decoder.generator.noise_res.{i}.alpha{1|2}.{j}
        if let Some(rest) = key.strip_prefix("decoder.generator.noise_res.") {
            if let Some((idx_str, alpha_rest)) = rest.split_once('.') {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if let Some((alpha_name, j_str)) = alpha_rest.split_once('.') {
                        if let Ok(j) = j_str.parse::<usize>() {
                            if idx < model.decoder.generator.noise_res.len() {
                                let block = &mut model.decoder.generator.noise_res[idx];
                                match alpha_name {
                                    "alpha1" if j < block.alpha1.len() => {
                                        block.alpha1[j] = value.clone();
                                        loaded.insert(key.clone());
                                    }
                                    "alpha2" if j < block.alpha2.len() => {
                                        block.alpha2[j] = value.clone();
                                        loaded.insert(key.clone());
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    loaded
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

    // Load alpha parameters manually (not exposed via #[param])
    let alpha_loaded = load_alpha_params(&mut model, &sanitized);

    // Load weights into model by matching parameter keys
    {
        use mlx_rs::module::ModuleParameters;
        let mut params = model.parameters_mut().flatten();
        let mut loaded = 0;
        let mut missing = Vec::new();
        let total = sanitized.len();
        for (key, value) in &sanitized {
            if let Some(param) = params.get_mut(&**key) {
                **param = value.clone();
                loaded += 1;
            } else if !alpha_loaded.contains(key.as_str()) {
                missing.push(key.clone());
            } else {
                loaded += 1; // count alpha as loaded
            }
        }
        if !missing.is_empty() {
            missing.sort();
            eprintln!("[WARN] Loaded {}/{} weights, {} unmatched:", loaded, total, missing.len());
            for k in &missing {
                eprintln!("  unmatched weight: {}", k);
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
