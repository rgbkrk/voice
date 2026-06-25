use std::path::Path;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{
    Result, VoxtralAssetPaths, VoxtralAssetResolver, VoxtralConfig, VoxtralError,
    VoxtralInferenceModules, VoxtralModuleLoadTrace, VoxtralSource, VoxtralTokenizerMetadata,
    VoxtralWeightMetadata,
};

/// Boundary for a future native Candle implementation of Voxtral TTS.
#[derive(Debug, Clone)]
pub struct VoxtralModel {
    config: VoxtralConfig,
    resolver: Option<VoxtralAssetResolver>,
    assets: Option<VoxtralAssetPaths>,
    tokenizer: Option<VoxtralTokenizerMetadata>,
    weights: Option<VoxtralWeightMetadata>,
}

#[derive(Debug, Clone, Default)]
pub struct VoxtralModelLoadTrace {
    pub resolve_assets: Duration,
    pub config_load: Duration,
    pub tokenizer_load: Duration,
    pub tokenizer_validate: Duration,
    pub weight_metadata: Duration,
    pub weight_validate: Duration,
    pub total: Duration,
}

impl VoxtralModel {
    pub fn new(config: VoxtralConfig) -> Self {
        Self {
            config,
            resolver: None,
            assets: None,
            tokenizer: None,
            weights: None,
        }
    }

    /// Load config and checkpoint metadata from a local directory or HF repo.
    ///
    /// This resolves `consolidated.safetensors` and validates its tensor layout
    /// against `params.json`. It does not instantiate the full generation graph yet.
    pub fn load(path_or_repo: &str) -> Result<Self> {
        let resolver = VoxtralAssetResolver::new(VoxtralSource::from_path_or_repo(path_or_repo));
        Ok(Self::load_from_resolver_with_trace(&resolver)?.0)
    }

    pub fn load_with_trace(path_or_repo: &str) -> Result<(Self, VoxtralModelLoadTrace)> {
        let resolver = VoxtralAssetResolver::new(VoxtralSource::from_path_or_repo(path_or_repo));
        Self::load_from_resolver_with_trace(&resolver)
    }

    pub fn load_from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let resolver = VoxtralAssetResolver::new(VoxtralSource::Local(dir.as_ref().to_path_buf()));
        Ok(Self::load_from_resolver_with_trace(&resolver)?.0)
    }

    pub fn load_from_dir_with_trace(
        dir: impl AsRef<Path>,
    ) -> Result<(Self, VoxtralModelLoadTrace)> {
        let resolver = VoxtralAssetResolver::new(VoxtralSource::Local(dir.as_ref().to_path_buf()));
        Self::load_from_resolver_with_trace(&resolver)
    }

    pub fn load_from_resolver(resolver: &VoxtralAssetResolver) -> Result<Self> {
        Ok(Self::load_from_resolver_with_trace(resolver)?.0)
    }

    pub fn load_from_resolver_with_trace(
        resolver: &VoxtralAssetResolver,
    ) -> Result<(Self, VoxtralModelLoadTrace)> {
        let total_start = Instant::now();
        let resolve_start = Instant::now();
        let assets = resolver.resolve_all()?;
        let resolve_assets = resolve_start.elapsed();

        let config_start = Instant::now();
        let config = VoxtralConfig::from_path(&assets.params_json)?;
        let config_load = config_start.elapsed();

        let tokenizer_start = Instant::now();
        let tokenizer = assets
            .tokenizer_json
            .as_ref()
            .map(VoxtralTokenizerMetadata::from_path)
            .transpose()?;
        let tokenizer_load = tokenizer_start.elapsed();

        let tokenizer_validate_start = Instant::now();
        if let Some(tokenizer) = &tokenizer {
            tokenizer.validate_for_config(&config)?;
        }
        let tokenizer_validate = tokenizer_validate_start.elapsed();

        let weights_path = assets.weights.as_ref().ok_or_else(|| {
            VoxtralError::InvalidCheckpoint("missing consolidated.safetensors".to_string())
        })?;
        let weight_metadata_start = Instant::now();
        let weights = VoxtralWeightMetadata::from_safetensors_file(weights_path)?;
        let weight_metadata = weight_metadata_start.elapsed();
        let weight_validate_start = Instant::now();
        weights.validate_for_config(&config)?;
        let weight_validate = weight_validate_start.elapsed();

        let trace = VoxtralModelLoadTrace {
            resolve_assets,
            config_load,
            tokenizer_load,
            tokenizer_validate,
            weight_metadata,
            weight_validate,
            total: total_start.elapsed(),
        };

        Ok((
            Self {
                config,
                resolver: Some(resolver.clone()),
                assets: Some(assets),
                tokenizer,
                weights: Some(weights),
            },
            trace,
        ))
    }

    pub fn config(&self) -> &VoxtralConfig {
        &self.config
    }

    pub fn assets(&self) -> Option<&VoxtralAssetPaths> {
        self.assets.as_ref()
    }

    pub fn tokenizer(&self) -> Option<&VoxtralTokenizerMetadata> {
        self.tokenizer.as_ref()
    }

    pub fn weights(&self) -> Option<&VoxtralWeightMetadata> {
        self.weights.as_ref()
    }

    pub fn checkpoint_summary(&self) -> Option<crate::VoxtralCheckpointSummary> {
        self.weights.as_ref().map(|weights| weights.summary())
    }

    pub fn resolve_voice_embedding_path(&self, voice: &str) -> Result<std::path::PathBuf> {
        if let Some(path) = self
            .assets
            .as_ref()
            .and_then(|assets| assets.voice_embeddings.get(voice))
        {
            return Ok(path.clone());
        }
        let resolver = self.resolver.as_ref().ok_or_else(|| {
            VoxtralError::InvalidCheckpoint("model was created without asset resolver".into())
        })?;
        resolver.resolve_voice_embedding(voice)
    }

    /// Open a Candle VarBuilder over the mmaped safetensors checkpoint.
    pub fn var_builder(&self, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
        let assets = self.assets.as_ref().ok_or_else(|| {
            VoxtralError::InvalidCheckpoint("model was created without checkpoint assets".into())
        })?;
        let weights = assets.weights.as_ref().ok_or_else(|| {
            VoxtralError::InvalidCheckpoint("missing consolidated.safetensors".into())
        })?;
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device) }
            .map_err(|e| VoxtralError::Candle(e.to_string()))
    }

    /// Load a tiny required tensor through Candle to verify mmaped access works.
    pub fn load_norm_weight(&self, dtype: DType, device: &Device) -> Result<Tensor> {
        let vb = self.var_builder(dtype, device)?;
        vb.get(&[self.config.dim], "norm.weight")
            .map_err(|e| VoxtralError::Candle(e.to_string()))
    }

    /// Instantiate the native Candle module skeleton from the checkpoint.
    ///
    /// This loads the multimodal embeddings, language backbone, and acoustic
    /// transformer tensors. The autoregressive generation loop and audio
    /// tokenizer/vocoder decoder are still separate follow-up work.
    pub fn load_inference_modules(
        &self,
        dtype: DType,
        device: &Device,
    ) -> Result<VoxtralInferenceModules> {
        Ok(self.load_inference_modules_with_trace(dtype, device)?.0)
    }

    pub fn load_inference_modules_with_trace(
        &self,
        dtype: DType,
        device: &Device,
    ) -> Result<(VoxtralInferenceModules, VoxtralModuleLoadTrace)> {
        let vb = self.var_builder(dtype, device)?;
        VoxtralInferenceModules::load_with_trace(&self.config, vb)
            .map_err(|e| VoxtralError::Candle(e.to_string()))
    }

    pub fn generate(&mut self, _text: &str, _voice: &str) -> Result<Vec<f32>> {
        Err(VoxtralError::Unsupported(
            "the config/asset layer exists, but the native audio-generation and audio-tokenizer stages have not been ported".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;
    use crate::WeightComponent;

    #[test]
    fn validates_local_checkpoint_when_env_is_set() {
        let Ok(dir) = std::env::var("VOXTRAL_LOCAL_DIR") else {
            return;
        };

        let (model, trace) = VoxtralModel::load_from_dir_with_trace(dir).unwrap();
        assert!(trace.total >= trace.resolve_assets);
        assert!(trace.total >= trace.config_load);
        assert!(trace.total >= trace.tokenizer_load);
        assert!(trace.total >= trace.tokenizer_validate);
        assert!(trace.total >= trace.weight_metadata);
        assert!(trace.total >= trace.weight_validate);

        let assets = model.assets().unwrap();
        assert_eq!(assets.voice_embeddings.len(), 20);
        assert!(assets.voice_embeddings.contains_key("casual_male"));
        let summary = model.checkpoint_summary().unwrap();

        assert_eq!(summary.tensor_count, 386);
        assert_eq!(
            summary
                .component_counts
                .get(&WeightComponent::LanguageModel)
                .copied(),
            Some(234)
        );
        assert_eq!(
            summary
                .component_counts
                .get(&WeightComponent::AcousticTransformer)
                .copied(),
            Some(33)
        );
        assert_eq!(
            summary
                .component_counts
                .get(&WeightComponent::AudioTokenizer)
                .copied(),
            Some(116)
        );
        let tokenizer = model.tokenizer().unwrap();
        assert_eq!(tokenizer.special_token_id("[OUTPUT_AUDIO]"), Some(26));
        assert_eq!(tokenizer.voice_audio_tokens("casual_male"), Some(147));

        let norm = model.load_norm_weight(DType::F32, &Device::Cpu).unwrap();
        assert_eq!(norm.dims(), &[model.config().dim]);
    }

    #[test]
    fn loads_local_inference_modules_when_env_is_set() {
        let Ok(dir) = std::env::var("VOXTRAL_LOCAL_DIR") else {
            return;
        };
        if std::env::var("VOXTRAL_LOAD_FULL").as_deref() != Ok("1") {
            return;
        }

        let model = VoxtralModel::load_from_dir(dir).unwrap();
        let modules = model
            .load_inference_modules(DType::BF16, &Device::Cpu)
            .unwrap();

        assert_eq!(modules.language.layers.len(), model.config().n_layers);
        assert_eq!(
            modules.acoustic.layers.len(),
            model
                .config()
                .multimodal
                .audio_model_args
                .acoustic_transformer_args
                .n_layers
        );
        assert_eq!(
            modules.embeddings.tok_embeddings.embeddings().dims(),
            &[model.config().vocab_size, model.config().dim]
        );
    }
}
