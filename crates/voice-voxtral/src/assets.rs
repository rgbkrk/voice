use std::path::{Path, PathBuf};

use hf_hub::api::sync::Api;

use crate::{Result, VoxtralConfig, VoxtralError};

pub const DEFAULT_REPO: &str = "mistralai/Voxtral-4B-TTS-2603";
pub const CONFIG_FILE: &str = "params.json";
pub const TOKENIZER_FILE: &str = "tekken.json";
pub const WEIGHTS_FILE: &str = "consolidated.safetensors";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoxtralSource {
    Local(PathBuf),
    Hub(String),
}

impl VoxtralSource {
    pub fn from_path_or_repo(path_or_repo: impl AsRef<str>) -> Self {
        let value = path_or_repo.as_ref();
        if Path::new(value).exists() {
            Self::Local(PathBuf::from(value))
        } else {
            Self::Hub(value.to_string())
        }
    }

    pub fn default_hub() -> Self {
        Self::Hub(DEFAULT_REPO.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralAssetPaths {
    pub params_json: PathBuf,
    pub tokenizer_json: Option<PathBuf>,
    pub weights: Option<PathBuf>,
    pub voice_embedding_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct VoxtralAssetResolver {
    source: VoxtralSource,
}

impl Default for VoxtralAssetResolver {
    fn default() -> Self {
        Self::new(VoxtralSource::default_hub())
    }
}

impl VoxtralAssetResolver {
    pub fn new(source: VoxtralSource) -> Self {
        Self { source }
    }

    pub fn source(&self) -> &VoxtralSource {
        &self.source
    }

    /// Resolve only `params.json`.
    pub fn resolve_config(&self) -> Result<PathBuf> {
        match &self.source {
            VoxtralSource::Local(dir) => require_file(dir.join(CONFIG_FILE), CONFIG_FILE),
            VoxtralSource::Hub(repo_id) => {
                let api = Api::new().map_err(|e| VoxtralError::Hub(e.to_string()))?;
                let repo = api.model(repo_id.to_string());
                repo.get(CONFIG_FILE)
                    .map_err(|e| VoxtralError::Hub(e.to_string()))
            }
        }
    }

    /// Resolve the lightweight metadata files needed to inspect the model.
    ///
    /// For HuggingFace sources this downloads `params.json` and `tekken.json`,
    /// but intentionally does not download the 8 GB weight file.
    pub fn resolve_metadata(&self) -> Result<VoxtralAssetPaths> {
        match &self.source {
            VoxtralSource::Local(dir) => self.resolve_local(dir, false),
            VoxtralSource::Hub(repo_id) => self.resolve_hub(repo_id, false),
        }
    }

    /// Resolve all files needed for native inference, including the weight file.
    pub fn resolve_all(&self) -> Result<VoxtralAssetPaths> {
        match &self.source {
            VoxtralSource::Local(dir) => self.resolve_local(dir, true),
            VoxtralSource::Hub(repo_id) => self.resolve_hub(repo_id, true),
        }
    }

    pub fn load_config(&self) -> Result<VoxtralConfig> {
        VoxtralConfig::from_path(self.resolve_config()?)
    }

    fn resolve_local(&self, dir: &Path, include_weights: bool) -> Result<VoxtralAssetPaths> {
        let params_json = require_file(dir.join(CONFIG_FILE), CONFIG_FILE)?;
        let tokenizer_json = optional_file(dir.join(TOKENIZER_FILE));
        let weights = if include_weights {
            Some(require_file(dir.join(WEIGHTS_FILE), WEIGHTS_FILE)?)
        } else {
            optional_file(dir.join(WEIGHTS_FILE))
        };
        let voice_embedding_dir = dir.join("voice_embedding");
        let voice_embedding_dir = voice_embedding_dir.is_dir().then_some(voice_embedding_dir);

        Ok(VoxtralAssetPaths {
            params_json,
            tokenizer_json,
            weights,
            voice_embedding_dir,
        })
    }

    fn resolve_hub(&self, repo_id: &str, include_weights: bool) -> Result<VoxtralAssetPaths> {
        let api = Api::new().map_err(|e| VoxtralError::Hub(e.to_string()))?;
        let repo = api.model(repo_id.to_string());

        let params_json = repo
            .get(CONFIG_FILE)
            .map_err(|e| VoxtralError::Hub(e.to_string()))?;
        let tokenizer_json = repo.get(TOKENIZER_FILE).ok();
        let weights = if include_weights {
            Some(
                repo.get(WEIGHTS_FILE)
                    .map_err(|e| VoxtralError::Hub(e.to_string()))?,
            )
        } else {
            None
        };

        Ok(VoxtralAssetPaths {
            params_json,
            tokenizer_json,
            weights,
            voice_embedding_dir: None,
        })
    }
}

fn require_file(path: PathBuf, label: &str) -> Result<PathBuf> {
    if path.is_file() {
        Ok(path)
    } else {
        Err(VoxtralError::InvalidConfig(format!(
            "missing required {label} at {}",
            path.display()
        )))
    }
}

fn optional_file(path: PathBuf) -> Option<PathBuf> {
    path.is_file().then_some(path)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn default_source_is_official_hub_repo() {
        assert_eq!(
            VoxtralSource::default_hub(),
            VoxtralSource::Hub(DEFAULT_REPO.to_string())
        );
    }

    #[test]
    fn local_metadata_resolution_does_not_require_weights() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "voice-voxtral-assets-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(CONFIG_FILE), "{}").unwrap();
        std::fs::write(dir.join(TOKENIZER_FILE), "{}").unwrap();

        let resolver = VoxtralAssetResolver::new(VoxtralSource::Local(dir.clone()));
        let assets = resolver.resolve_metadata().unwrap();

        assert_eq!(assets.params_json, dir.join(CONFIG_FILE));
        assert_eq!(assets.tokenizer_json, Some(dir.join(TOKENIZER_FILE)));
        assert_eq!(assets.weights, None);
        assert!(resolver.resolve_all().is_err());

        std::fs::remove_dir_all(dir).unwrap();
    }
}
