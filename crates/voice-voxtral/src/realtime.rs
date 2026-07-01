use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use serde::Deserialize;

use crate::{
    Result, VoxtralCheckpointSummary, VoxtralError, VoxtralRealtimeAudioModules,
    VoxtralRealtimeAudioTransformer, VoxtralRealtimeInferenceModules, VoxtralRealtimeTextDecoder,
    VoxtralRealtimeTokenEmbeddings, VoxtralRealtimeTranscriber, VoxtralSource,
    VoxtralTokenizerMetadata, VoxtralWeightMetadata,
};

pub const REALTIME_DEFAULT_REPO: &str = "mistralai/Voxtral-Mini-4B-Realtime-2602";
pub const REALTIME_CONFIG_FILE: &str = "params.json";
pub const REALTIME_HF_CONFIG_FILE: &str = "config.json";
pub const REALTIME_PROCESSOR_CONFIG_FILE: &str = "processor_config.json";
pub const REALTIME_TOKENIZER_FILE: &str = "tekken.json";
pub const REALTIME_WEIGHTS_FILE: &str = "consolidated.safetensors";
pub const REALTIME_SAMPLE_RATE: u32 = 16_000;
pub const REALTIME_NUM_MEL_BINS: usize = 128;
pub const REALTIME_TRANSCRIPTION_FORMAT: &str = "streaming";
pub const REALTIME_EXPECTED_TENSOR_COUNT: usize = 711;
pub const REALTIME_BOS_TOKEN_ID: usize = 1;
pub const REALTIME_EOS_TOKEN_ID: usize = 2;
pub const REALTIME_AUDIO_TOKEN_ID: usize = 24;
pub const REALTIME_BEGIN_AUDIO_TOKEN_ID: usize = 25;
pub const REALTIME_STREAMING_PAD_TOKEN_ID: usize = 32;
pub const REALTIME_STREAMING_WORD_TOKEN_ID: usize = 33;
pub const REALTIME_REPEAT_AUDIO_TEXT_TOKEN_ID: usize = 34;
pub const REALTIME_DEFAULT_LEFT_PAD_TOKENS: usize = 32;
pub const REALTIME_DEFAULT_OFFLINE_BUFFER_TOKENS: usize = 10;
const REALTIME_TEXT_ADA_NORM_DIM: usize = 32;
pub(crate) const REALTIME_ENCODER_PREFIX: &str =
    "mm_streams_embeddings.embedding_module.whisper_encoder";
pub(crate) const REALTIME_STREAMS_PREFIX: &str = "mm_streams_embeddings.embedding_module";
const SAFETENSORS_BF16: &str = "BF16";

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralRealtimeConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub use_biases: bool,
    pub causal: bool,
    pub rope_theta: f64,
    pub norm_eps: f64,
    pub vocab_size: usize,
    pub model_parallel: usize,
    pub tied_embeddings: bool,
    pub sliding_window: usize,
    pub model_max_length: usize,
    pub multimodal: VoxtralRealtimeMultimodalConfig,
    #[serde(default)]
    pub ada_rms_norm_t_cond: bool,
    #[serde(default)]
    pub ada_rms_norm_t_cond_dim: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralRealtimeMultimodalConfig {
    pub whisper_model_args: VoxtralRealtimeWhisperModelConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralRealtimeWhisperModelConfig {
    pub encoder_args: VoxtralRealtimeAudioEncoderConfig,
    pub downsample_args: VoxtralRealtimeDownsampleConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralRealtimeAudioEncoderConfig {
    pub audio_encoding_args: VoxtralRealtimeAudioEncodingConfig,
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub n_kv_heads: usize,
    pub use_biases: bool,
    pub use_cache: bool,
    pub rope_theta: f64,
    pub causal: bool,
    pub norm_eps: f64,
    pub pos_embed: String,
    pub max_source_positions: Option<usize>,
    pub ffn_type: String,
    pub norm_type: String,
    pub sliding_window: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralRealtimeAudioEncodingConfig {
    pub sampling_rate: u32,
    pub frame_rate: f64,
    pub num_mel_bins: usize,
    pub hop_length: usize,
    pub window_size: usize,
    pub chunk_length_s: Option<f64>,
    pub global_log_mel_max: f64,
    pub transcription_format: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralRealtimeDownsampleConfig {
    pub downsample_factor: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralRealtimeTransformersConfig {
    pub architectures: Vec<String>,
    pub audio_config: VoxtralRealtimeTransformersAudioConfig,
    pub audio_length_per_tok: usize,
    pub default_num_delay_tokens: usize,
    pub downsample_factor: usize,
    pub dtype: String,
    pub hidden_size: usize,
    pub model_type: String,
    pub text_config: VoxtralRealtimeTransformersTextConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralRealtimeTransformersAudioConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_mel_bins: usize,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub vocab_size: usize,
    pub model_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralRealtimeTransformersTextConfig {
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralRealtimeAssetPaths {
    pub params_json: PathBuf,
    pub hf_config_json: Option<PathBuf>,
    pub processor_config_json: Option<PathBuf>,
    pub tokenizer_json: Option<PathBuf>,
    pub weights: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct VoxtralRealtimeAssetResolver {
    source: VoxtralSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralRealtimePrompt {
    pub input_ids: Vec<usize>,
    pub left_pad_tokens: usize,
    pub delay_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralRealtimePaddingPlan {
    pub input_samples: usize,
    pub raw_audio_length_per_token: usize,
    pub left_pad_tokens: usize,
    pub delay_tokens: usize,
    pub right_pad_tokens: usize,
    pub align_pad_samples: usize,
    pub left_pad_samples: usize,
    pub right_pad_samples: usize,
    pub padded_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralRealtimeExpectedTensor {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct VoxtralRealtimeModel {
    config: VoxtralRealtimeConfig,
    transformers_config: Option<VoxtralRealtimeTransformersConfig>,
    assets: VoxtralRealtimeAssetPaths,
    tokenizer: Option<VoxtralTokenizerMetadata>,
    weights: Option<VoxtralWeightMetadata>,
}

impl VoxtralRealtimeConfig {
    pub fn from_json_str(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json_str(&json)
    }

    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 || self.n_layers == 0 || self.vocab_size == 0 {
            return Err(VoxtralError::InvalidConfig(
                "realtime text backbone dimensions must be non-zero".into(),
            ));
        }
        if !self.causal {
            return Err(VoxtralError::InvalidConfig(
                "realtime text backbone must be causal".into(),
            ));
        }

        let whisper = &self.multimodal.whisper_model_args;
        let encoder = &whisper.encoder_args;
        let encoding = &encoder.audio_encoding_args;
        if encoding.sampling_rate != REALTIME_SAMPLE_RATE {
            return Err(VoxtralError::InvalidConfig(format!(
                "expected realtime audio sample rate {REALTIME_SAMPLE_RATE}, got {}",
                encoding.sampling_rate
            )));
        }
        if encoding.num_mel_bins != REALTIME_NUM_MEL_BINS {
            return Err(VoxtralError::InvalidConfig(format!(
                "expected {REALTIME_NUM_MEL_BINS} mel bins, got {}",
                encoding.num_mel_bins
            )));
        }
        if encoding.transcription_format != REALTIME_TRANSCRIPTION_FORMAT {
            return Err(VoxtralError::InvalidConfig(format!(
                "expected transcription_format {REALTIME_TRANSCRIPTION_FORMAT:?}, got {:?}",
                encoding.transcription_format
            )));
        }
        if !encoder.causal || encoder.use_cache {
            return Err(VoxtralError::InvalidConfig(
                "realtime audio encoder must be causal and cache-free".into(),
            ));
        }
        if whisper.downsample_args.downsample_factor == 0 {
            return Err(VoxtralError::InvalidConfig(
                "downsample_factor must be greater than zero".into(),
            ));
        }
        Ok(())
    }

    pub fn model_type(&self) -> &'static str {
        "voxtral_realtime"
    }

    pub fn sample_rate(&self) -> u32 {
        self.multimodal
            .whisper_model_args
            .encoder_args
            .audio_encoding_args
            .sampling_rate
    }

    pub fn frame_rate(&self) -> f64 {
        self.multimodal
            .whisper_model_args
            .encoder_args
            .audio_encoding_args
            .frame_rate
    }

    pub fn num_mel_bins(&self) -> usize {
        self.multimodal
            .whisper_model_args
            .encoder_args
            .audio_encoding_args
            .num_mel_bins
    }

    pub fn downsample_factor(&self) -> usize {
        self.multimodal
            .whisper_model_args
            .downsample_args
            .downsample_factor
    }
}

impl VoxtralRealtimeTransformersConfig {
    pub fn from_json_str(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json_str(&json)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "voxtral_realtime" {
            return Err(VoxtralError::InvalidConfig(format!(
                "expected model_type voxtral_realtime, got {}",
                self.model_type
            )));
        }
        if !self
            .architectures
            .iter()
            .any(|name| name == "VoxtralRealtimeForConditionalGeneration")
        {
            return Err(VoxtralError::InvalidConfig(
                "missing VoxtralRealtimeForConditionalGeneration architecture".into(),
            ));
        }
        if self.audio_config.model_type != "voxtral_realtime_encoder"
            || self.text_config.model_type != "voxtral_realtime_text"
        {
            return Err(VoxtralError::InvalidConfig(
                "unexpected realtime audio/text config model_type".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_against_params(&self, params: &VoxtralRealtimeConfig) -> Result<()> {
        let encoder = &params.multimodal.whisper_model_args.encoder_args;
        if self.hidden_size != params.dim
            || self.text_config.hidden_size != params.dim
            || self.text_config.num_hidden_layers != params.n_layers
            || self.text_config.vocab_size != params.vocab_size
            || self.audio_config.hidden_size != encoder.dim
            || self.audio_config.num_hidden_layers != encoder.n_layers
            || self.audio_config.num_mel_bins != params.num_mel_bins()
            || self.downsample_factor != params.downsample_factor()
        {
            return Err(VoxtralError::InvalidConfig(
                "realtime config.json does not match params.json".into(),
            ));
        }
        Ok(())
    }
}

impl Default for VoxtralRealtimeAssetResolver {
    fn default() -> Self {
        Self::new(VoxtralSource::Hub(REALTIME_DEFAULT_REPO.to_string()))
    }
}

impl VoxtralRealtimeAssetResolver {
    pub fn new(source: VoxtralSource) -> Self {
        Self { source }
    }

    pub fn source(&self) -> &VoxtralSource {
        &self.source
    }

    pub fn resolve_metadata(&self) -> Result<VoxtralRealtimeAssetPaths> {
        match &self.source {
            VoxtralSource::Local(dir) => self.resolve_local(dir, false),
            VoxtralSource::Hub(repo_id) => self.resolve_hub(repo_id, false),
        }
    }

    pub fn resolve_all(&self) -> Result<VoxtralRealtimeAssetPaths> {
        match &self.source {
            VoxtralSource::Local(dir) => self.resolve_local(dir, true),
            VoxtralSource::Hub(repo_id) => self.resolve_hub(repo_id, true),
        }
    }

    fn resolve_local(
        &self,
        dir: &Path,
        include_weights: bool,
    ) -> Result<VoxtralRealtimeAssetPaths> {
        let params_json = require_file(dir.join(REALTIME_CONFIG_FILE), REALTIME_CONFIG_FILE)?;
        let hf_config_json = optional_file(dir.join(REALTIME_HF_CONFIG_FILE));
        let processor_config_json = optional_file(dir.join(REALTIME_PROCESSOR_CONFIG_FILE));
        let tokenizer_json = optional_file(dir.join(REALTIME_TOKENIZER_FILE));
        let weights = if include_weights {
            Some(require_file(
                dir.join(REALTIME_WEIGHTS_FILE),
                REALTIME_WEIGHTS_FILE,
            )?)
        } else {
            optional_file(dir.join(REALTIME_WEIGHTS_FILE))
        };

        Ok(VoxtralRealtimeAssetPaths {
            params_json,
            hf_config_json,
            processor_config_json,
            tokenizer_json,
            weights,
        })
    }

    fn resolve_hub(
        &self,
        repo_id: &str,
        include_weights: bool,
    ) -> Result<VoxtralRealtimeAssetPaths> {
        let api = Api::new().map_err(|e| VoxtralError::Hub(e.to_string()))?;
        let repo = api.model(repo_id.to_string());
        let params_json = repo
            .get(REALTIME_CONFIG_FILE)
            .map_err(|e| VoxtralError::Hub(e.to_string()))?;
        let hf_config_json = repo.get(REALTIME_HF_CONFIG_FILE).ok();
        let processor_config_json = repo.get(REALTIME_PROCESSOR_CONFIG_FILE).ok();
        let tokenizer_json = repo.get(REALTIME_TOKENIZER_FILE).ok();
        let weights = if include_weights {
            Some(
                repo.get(REALTIME_WEIGHTS_FILE)
                    .map_err(|e| VoxtralError::Hub(e.to_string()))?,
            )
        } else {
            None
        };

        Ok(VoxtralRealtimeAssetPaths {
            params_json,
            hf_config_json,
            processor_config_json,
            tokenizer_json,
            weights,
        })
    }
}

impl VoxtralRealtimeModel {
    pub fn load_metadata(path_or_repo: &str) -> Result<Self> {
        let resolver =
            VoxtralRealtimeAssetResolver::new(VoxtralSource::from_path_or_repo(path_or_repo));
        Self::load_metadata_from_resolver(&resolver)
    }

    pub fn load_metadata_from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let resolver =
            VoxtralRealtimeAssetResolver::new(VoxtralSource::Local(dir.as_ref().to_path_buf()));
        Self::load_metadata_from_resolver(&resolver)
    }

    pub fn load_metadata_from_resolver(resolver: &VoxtralRealtimeAssetResolver) -> Result<Self> {
        Self::load_from_assets(resolver.resolve_metadata()?, false)
    }

    pub fn load(path_or_repo: &str) -> Result<Self> {
        let resolver =
            VoxtralRealtimeAssetResolver::new(VoxtralSource::from_path_or_repo(path_or_repo));
        Self::load_from_resolver(&resolver)
    }

    pub fn load_from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let resolver =
            VoxtralRealtimeAssetResolver::new(VoxtralSource::Local(dir.as_ref().to_path_buf()));
        Self::load_from_resolver(&resolver)
    }

    pub fn load_from_resolver(resolver: &VoxtralRealtimeAssetResolver) -> Result<Self> {
        Self::load_from_assets(resolver.resolve_all()?, true)
    }

    pub fn config(&self) -> &VoxtralRealtimeConfig {
        &self.config
    }

    pub fn transformers_config(&self) -> Option<&VoxtralRealtimeTransformersConfig> {
        self.transformers_config.as_ref()
    }

    pub fn assets(&self) -> &VoxtralRealtimeAssetPaths {
        &self.assets
    }

    pub fn tokenizer(&self) -> Option<&VoxtralTokenizerMetadata> {
        self.tokenizer.as_ref()
    }

    pub fn weights(&self) -> Option<&VoxtralWeightMetadata> {
        self.weights.as_ref()
    }

    pub fn checkpoint_summary(&self) -> Option<crate::VoxtralCheckpointSummary> {
        self.weights.as_ref().map(|weights| {
            weights.summary_with_expected_tensor_count(REALTIME_EXPECTED_TENSOR_COUNT)
        })
    }

    pub fn var_builder(&self, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
        let weights = self.assets.weights.as_ref().ok_or_else(|| {
            VoxtralError::InvalidCheckpoint("missing realtime consolidated.safetensors".into())
        })?;
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, device) }
            .map_err(|e| VoxtralError::Candle(e.to_string()))
    }

    pub fn load_inference_modules(
        &self,
        dtype: DType,
        device: &Device,
    ) -> Result<VoxtralRealtimeInferenceModules> {
        let vb = self.var_builder(dtype, device)?;
        VoxtralRealtimeInferenceModules::load(&self.config, vb)
            .map_err(|e| VoxtralError::Candle(e.to_string()))
    }

    pub fn load_audio_transformer(
        &self,
        dtype: DType,
        device: &Device,
    ) -> Result<VoxtralRealtimeAudioTransformer> {
        let vb = self.var_builder(dtype, device)?;
        VoxtralRealtimeAudioTransformer::load(&self.config, vb)
            .map_err(|e| VoxtralError::Candle(e.to_string()))
    }

    pub fn load_audio_modules(
        &self,
        dtype: DType,
        device: &Device,
    ) -> Result<VoxtralRealtimeAudioModules> {
        let vb = self.var_builder(dtype, device)?;
        VoxtralRealtimeAudioModules::load(&self.config, vb)
            .map_err(|e| VoxtralError::Candle(e.to_string()))
    }

    pub fn load_text_decoder(
        &self,
        dtype: DType,
        device: &Device,
    ) -> Result<VoxtralRealtimeTextDecoder> {
        let vb = self.var_builder(dtype, device)?;
        VoxtralRealtimeTextDecoder::load(&self.config, vb)
            .map_err(|e| VoxtralError::Candle(e.to_string()))
    }

    pub fn load_transcriber(
        &self,
        dtype: DType,
        device: &Device,
    ) -> Result<VoxtralRealtimeTranscriber> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            VoxtralError::InvalidTokenizer("missing realtime tekken tokenizer".into())
        })?;
        let token_decoder = tokenizer.decoder()?;
        let vb = self.var_builder(dtype, device)?;
        let token_embeddings = VoxtralRealtimeTokenEmbeddings::load(&self.config, vb.clone())
            .map_err(|e| VoxtralError::Candle(e.to_string()))?;
        let audio_modules = VoxtralRealtimeAudioModules::load(&self.config, vb.clone())
            .map_err(|e| VoxtralError::Candle(e.to_string()))?;
        let text_decoder = VoxtralRealtimeTextDecoder::load(&self.config, vb)
            .map_err(|e| VoxtralError::Candle(e.to_string()))?;
        Ok(VoxtralRealtimeTranscriber::new(
            self.config.clone(),
            token_embeddings,
            audio_modules,
            text_decoder,
            token_decoder,
        ))
    }

    pub fn default_delay_tokens(&self) -> Result<usize> {
        if let Some(config) = &self.transformers_config {
            Ok(config.default_num_delay_tokens)
        } else if let Some(tokenizer) = &self.tokenizer {
            if let Some(delay_ms) = tokenizer.audio.transcription_delay_ms {
                realtime_num_delay_tokens(&self.config, delay_ms)
            } else {
                Ok(6)
            }
        } else {
            Ok(6)
        }
    }

    fn load_from_assets(assets: VoxtralRealtimeAssetPaths, require_weights: bool) -> Result<Self> {
        let config = VoxtralRealtimeConfig::from_path(&assets.params_json)?;
        let transformers_config = assets
            .hf_config_json
            .as_ref()
            .map(VoxtralRealtimeTransformersConfig::from_path)
            .transpose()?;
        if let Some(transformers_config) = &transformers_config {
            transformers_config.validate_against_params(&config)?;
        }
        let tokenizer = assets
            .tokenizer_json
            .as_ref()
            .map(VoxtralTokenizerMetadata::from_path)
            .transpose()?;
        if let Some(tokenizer) = &tokenizer {
            validate_tokenizer_metadata(tokenizer, &config)?;
        }
        let weights = if let Some(weights) = &assets.weights {
            let weights = VoxtralWeightMetadata::from_safetensors_file(weights)?;
            validate_realtime_checkpoint(&weights, &config)?;
            Some(weights)
        } else if require_weights {
            return Err(VoxtralError::InvalidCheckpoint(format!(
                "missing {REALTIME_WEIGHTS_FILE}"
            )));
        } else {
            None
        };

        Ok(Self {
            config,
            transformers_config,
            assets,
            tokenizer,
            weights,
        })
    }
}

pub fn validate_realtime_checkpoint(
    weights: &VoxtralWeightMetadata,
    config: &VoxtralRealtimeConfig,
) -> Result<VoxtralCheckpointSummary> {
    if weights.tensor_count() != REALTIME_EXPECTED_TENSOR_COUNT {
        return Err(VoxtralError::InvalidCheckpoint(format!(
            "realtime checkpoint has {} tensors, expected {REALTIME_EXPECTED_TENSOR_COUNT}",
            weights.tensor_count()
        )));
    }

    for expected in expected_realtime_tensors(config) {
        let tensor = weights.tensor(&expected.name).ok_or_else(|| {
            VoxtralError::InvalidCheckpoint(format!("missing required tensor {}", expected.name))
        })?;
        if tensor.dtype != expected.dtype {
            return Err(VoxtralError::InvalidCheckpoint(format!(
                "tensor {} has dtype {}, expected {}",
                expected.name, tensor.dtype, expected.dtype
            )));
        }
        if tensor.shape != expected.shape {
            return Err(VoxtralError::InvalidCheckpoint(format!(
                "tensor {} has shape {:?}, expected {:?}",
                expected.name, tensor.shape, expected.shape
            )));
        }
    }

    Ok(weights.summary_with_expected_tensor_count(REALTIME_EXPECTED_TENSOR_COUNT))
}

pub fn realtime_raw_audio_length_per_token(config: &VoxtralRealtimeConfig) -> Result<usize> {
    let sample_rate = config.sample_rate() as f64;
    let raw = sample_rate / config.frame_rate();
    let rounded = raw.round();
    if (raw - rounded).abs() > f64::EPSILON {
        return Err(VoxtralError::InvalidConfig(format!(
            "sample_rate/frame_rate must be integral, got {raw}"
        )));
    }
    let raw = rounded as usize;
    if raw == 0 {
        return Err(VoxtralError::InvalidConfig(
            "raw audio length per token must be greater than zero".into(),
        ));
    }
    Ok(raw)
}

pub fn realtime_audio_frames_per_token(config: &VoxtralRealtimeConfig) -> Result<usize> {
    let raw = realtime_raw_audio_length_per_token(config)?;
    let hop = config
        .multimodal
        .whisper_model_args
        .encoder_args
        .audio_encoding_args
        .hop_length;
    if hop == 0 || !raw.is_multiple_of(hop) {
        return Err(VoxtralError::InvalidConfig(format!(
            "raw audio length per token {raw} must be divisible by hop_length {hop}"
        )));
    }
    Ok(raw / hop)
}

pub fn realtime_num_audio_tokens_for_samples(
    config: &VoxtralRealtimeConfig,
    sample_count: usize,
) -> Result<usize> {
    let hop = config
        .multimodal
        .whisper_model_args
        .encoder_args
        .audio_encoding_args
        .hop_length;
    let frames_per_token = realtime_audio_frames_per_token(config)?;
    let mel_frames = if sample_count.is_multiple_of(hop) {
        sample_count / hop
    } else {
        sample_count.div_ceil(hop).saturating_sub(1)
    };
    Ok(mel_frames.div_ceil(frames_per_token))
}

pub fn realtime_num_delay_tokens(config: &VoxtralRealtimeConfig, delay_ms: usize) -> Result<usize> {
    let delay_samples = delay_ms
        .checked_mul(config.sample_rate() as usize)
        .ok_or_else(|| VoxtralError::InvalidConfig("delay_ms overflow".into()))?
        / 1000;
    realtime_num_audio_tokens_for_samples(config, delay_samples)
}

pub fn build_realtime_streaming_prompt(delay_tokens: usize) -> VoxtralRealtimePrompt {
    build_realtime_streaming_prompt_with_left_pad(REALTIME_DEFAULT_LEFT_PAD_TOKENS, delay_tokens)
}

pub fn build_realtime_streaming_prompt_with_left_pad(
    left_pad_tokens: usize,
    delay_tokens: usize,
) -> VoxtralRealtimePrompt {
    let mut input_ids = Vec::with_capacity(1 + left_pad_tokens + delay_tokens);
    input_ids.push(REALTIME_BOS_TOKEN_ID);
    input_ids.extend(std::iter::repeat_n(
        REALTIME_STREAMING_PAD_TOKEN_ID,
        left_pad_tokens + delay_tokens,
    ));
    VoxtralRealtimePrompt {
        input_ids,
        left_pad_tokens,
        delay_tokens,
    }
}

pub fn plan_realtime_audio_padding(
    config: &VoxtralRealtimeConfig,
    sample_count: usize,
    delay_tokens: usize,
) -> Result<VoxtralRealtimePaddingPlan> {
    plan_realtime_audio_padding_with_left_pad(
        config,
        sample_count,
        REALTIME_DEFAULT_LEFT_PAD_TOKENS,
        delay_tokens,
    )
}

pub fn plan_realtime_audio_padding_with_left_pad(
    config: &VoxtralRealtimeConfig,
    sample_count: usize,
    left_pad_tokens: usize,
    delay_tokens: usize,
) -> Result<VoxtralRealtimePaddingPlan> {
    let raw_audio_length_per_token = realtime_raw_audio_length_per_token(config)?;
    let align_pad_samples = (raw_audio_length_per_token
        - (sample_count % raw_audio_length_per_token))
        % raw_audio_length_per_token;
    let right_pad_tokens = delay_tokens + 1 + REALTIME_DEFAULT_OFFLINE_BUFFER_TOKENS;
    let left_pad_samples = left_pad_tokens
        .checked_mul(raw_audio_length_per_token)
        .ok_or_else(|| VoxtralError::InvalidConfig("left pad overflow".into()))?;
    let right_pad_samples = right_pad_tokens
        .checked_mul(raw_audio_length_per_token)
        .and_then(|padding| padding.checked_add(align_pad_samples))
        .ok_or_else(|| VoxtralError::InvalidConfig("right pad overflow".into()))?;
    let padded_samples = sample_count
        .checked_add(left_pad_samples)
        .and_then(|samples| samples.checked_add(right_pad_samples))
        .ok_or_else(|| VoxtralError::InvalidConfig("padded sample count overflow".into()))?;

    Ok(VoxtralRealtimePaddingPlan {
        input_samples: sample_count,
        raw_audio_length_per_token,
        left_pad_tokens,
        delay_tokens,
        right_pad_tokens,
        align_pad_samples,
        left_pad_samples,
        right_pad_samples,
        padded_samples,
    })
}

pub fn pad_realtime_audio(samples: &[f32], plan: &VoxtralRealtimePaddingPlan) -> Vec<f32> {
    let mut padded = vec![0.0; plan.padded_samples];
    let start = plan.left_pad_samples;
    let end = start + samples.len();
    padded[start..end].copy_from_slice(samples);
    padded
}

pub fn expected_realtime_tensors(
    config: &VoxtralRealtimeConfig,
) -> Vec<VoxtralRealtimeExpectedTensor> {
    let mut tensors = Vec::with_capacity(REALTIME_EXPECTED_TENSOR_COUNT);
    add_realtime_audio_tensors(config, &mut tensors);
    add_realtime_text_tensors(config, &mut tensors);
    expected_bf16(&mut tensors, "norm.weight", [config.dim]);
    tensors
}

fn add_realtime_audio_tensors(
    config: &VoxtralRealtimeConfig,
    tensors: &mut Vec<VoxtralRealtimeExpectedTensor>,
) {
    let encoder = &config.multimodal.whisper_model_args.encoder_args;
    let encoding = &encoder.audio_encoding_args;
    let dim = encoder.dim;
    let qkv_dim = encoder.n_heads * encoder.head_dim;
    let hidden = encoder.hidden_dim;
    let adapter_input_dim = encoder.dim * config.downsample_factor();

    expected_bf16(
        tensors,
        format!("{REALTIME_ENCODER_PREFIX}.conv_layers.0.conv.weight"),
        [dim, encoding.num_mel_bins, 3],
    );
    expected_bf16(
        tensors,
        format!("{REALTIME_ENCODER_PREFIX}.conv_layers.0.conv.bias"),
        [dim],
    );
    expected_bf16(
        tensors,
        format!("{REALTIME_ENCODER_PREFIX}.conv_layers.1.conv.weight"),
        [dim, dim, 3],
    );
    expected_bf16(
        tensors,
        format!("{REALTIME_ENCODER_PREFIX}.conv_layers.1.conv.bias"),
        [dim],
    );

    for layer in 0..encoder.n_layers {
        let prefix = format!("{REALTIME_ENCODER_PREFIX}.transformer.layers.{layer}");
        expected_bf16(
            tensors,
            format!("{prefix}.attention.wq.weight"),
            [qkv_dim, dim],
        );
        expected_bf16(tensors, format!("{prefix}.attention.wq.bias"), [qkv_dim]);
        expected_bf16(
            tensors,
            format!("{prefix}.attention.wk.weight"),
            [qkv_dim, dim],
        );
        expected_bf16(
            tensors,
            format!("{prefix}.attention.wv.weight"),
            [qkv_dim, dim],
        );
        expected_bf16(tensors, format!("{prefix}.attention.wv.bias"), [qkv_dim]);
        expected_bf16(
            tensors,
            format!("{prefix}.attention.wo.weight"),
            [dim, qkv_dim],
        );
        expected_bf16(tensors, format!("{prefix}.attention.wo.bias"), [dim]);
        expected_bf16(tensors, format!("{prefix}.attention_norm.weight"), [dim]);
        expected_bf16(tensors, format!("{prefix}.ffn_norm.weight"), [dim]);
        expected_bf16(
            tensors,
            format!("{prefix}.feed_forward.w1.weight"),
            [hidden, dim],
        );
        expected_bf16(
            tensors,
            format!("{prefix}.feed_forward.w2.weight"),
            [dim, hidden],
        );
        expected_bf16(tensors, format!("{prefix}.feed_forward.w2.bias"), [dim]);
        expected_bf16(
            tensors,
            format!("{prefix}.feed_forward.w3.weight"),
            [hidden, dim],
        );
    }

    expected_bf16(
        tensors,
        format!("{REALTIME_ENCODER_PREFIX}.transformer.norm.weight"),
        [dim],
    );
    expected_bf16(
        tensors,
        format!("{REALTIME_STREAMS_PREFIX}.audio_language_projection.0.weight"),
        [config.dim, adapter_input_dim],
    );
    expected_bf16(
        tensors,
        format!("{REALTIME_STREAMS_PREFIX}.audio_language_projection.2.weight"),
        [config.dim, config.dim],
    );
    expected_bf16(
        tensors,
        format!("{REALTIME_STREAMS_PREFIX}.tok_embeddings.weight"),
        [config.vocab_size, config.dim],
    );
}

fn add_realtime_text_tensors(
    config: &VoxtralRealtimeConfig,
    tensors: &mut Vec<VoxtralRealtimeExpectedTensor>,
) {
    let dim = config.dim;
    let hidden = config.hidden_dim;
    let q_dim = config.n_heads * config.head_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;

    for layer in 0..config.n_layers {
        let prefix = format!("layers.{layer}");
        expected_bf16(
            tensors,
            format!("{prefix}.ada_rms_norm_t_cond.0.weight"),
            [REALTIME_TEXT_ADA_NORM_DIM, dim],
        );
        expected_bf16(
            tensors,
            format!("{prefix}.ada_rms_norm_t_cond.2.weight"),
            [dim, REALTIME_TEXT_ADA_NORM_DIM],
        );
        expected_bf16(
            tensors,
            format!("{prefix}.attention.wq.weight"),
            [q_dim, dim],
        );
        expected_bf16(
            tensors,
            format!("{prefix}.attention.wk.weight"),
            [kv_dim, dim],
        );
        expected_bf16(
            tensors,
            format!("{prefix}.attention.wv.weight"),
            [kv_dim, dim],
        );
        expected_bf16(
            tensors,
            format!("{prefix}.attention.wo.weight"),
            [dim, q_dim],
        );
        expected_bf16(tensors, format!("{prefix}.attention_norm.weight"), [dim]);
        expected_bf16(tensors, format!("{prefix}.ffn_norm.weight"), [dim]);
        expected_bf16(
            tensors,
            format!("{prefix}.feed_forward.w1.weight"),
            [hidden, dim],
        );
        expected_bf16(
            tensors,
            format!("{prefix}.feed_forward.w2.weight"),
            [dim, hidden],
        );
        expected_bf16(
            tensors,
            format!("{prefix}.feed_forward.w3.weight"),
            [hidden, dim],
        );
    }
}

fn expected_bf16<const N: usize>(
    tensors: &mut Vec<VoxtralRealtimeExpectedTensor>,
    name: impl Into<String>,
    shape: [usize; N],
) {
    expected(tensors, name, SAFETENSORS_BF16, shape);
}

fn expected<const N: usize>(
    tensors: &mut Vec<VoxtralRealtimeExpectedTensor>,
    name: impl Into<String>,
    dtype: &'static str,
    shape: [usize; N],
) {
    tensors.push(VoxtralRealtimeExpectedTensor {
        name: name.into(),
        dtype,
        shape: shape.to_vec(),
    });
}

fn validate_tokenizer_metadata(
    tokenizer: &VoxtralTokenizerMetadata,
    config: &VoxtralRealtimeConfig,
) -> Result<()> {
    let encoding = &config
        .multimodal
        .whisper_model_args
        .encoder_args
        .audio_encoding_args;
    if tokenizer.config.default_vocab_size != config.vocab_size {
        return Err(VoxtralError::InvalidTokenizer(format!(
            "tokenizer default_vocab_size={} but params vocab_size={}",
            tokenizer.config.default_vocab_size, config.vocab_size
        )));
    }
    if tokenizer.audio.sampling_rate != encoding.sampling_rate {
        return Err(VoxtralError::InvalidTokenizer(format!(
            "tokenizer sampling_rate={} but params sampling_rate={}",
            tokenizer.audio.sampling_rate, encoding.sampling_rate
        )));
    }
    if tokenizer.audio.audio_encoding_config.num_mel_bins != encoding.num_mel_bins
        || tokenizer.audio.audio_encoding_config.hop_length != encoding.hop_length
        || tokenizer.audio.audio_encoding_config.window_size != encoding.window_size
    {
        return Err(VoxtralError::InvalidTokenizer(
            "tokenizer audio encoding config does not match realtime params".into(),
        ));
    }
    expect_special_token(tokenizer, "<s>", 1)?;
    expect_special_token(tokenizer, "</s>", 2)?;
    expect_special_token(tokenizer, "[AUDIO]", 24)?;
    expect_special_token(tokenizer, "[BEGIN_AUDIO]", 25)?;
    expect_special_token(tokenizer, "[STREAMING_PAD]", 32)?;
    expect_special_token(tokenizer, "[STREAMING_WORD]", 33)?;
    expect_special_token(tokenizer, "[REPEAT_AUDIO_TEXT]", 34)?;
    Ok(())
}

fn expect_special_token(
    tokenizer: &VoxtralTokenizerMetadata,
    token: &str,
    expected_id: usize,
) -> Result<()> {
    match tokenizer.special_token_id(token) {
        Some(id) if id == expected_id => Ok(()),
        Some(id) => Err(VoxtralError::InvalidTokenizer(format!(
            "special token {token} has id {id}, expected {expected_id}"
        ))),
        None => Err(VoxtralError::InvalidTokenizer(format!(
            "missing special token {token}"
        ))),
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

    const REALTIME_PARAMS_JSON: &str = r#"{
      "dim": 3072,
      "n_layers": 26,
      "head_dim": 128,
      "hidden_dim": 9216,
      "n_heads": 32,
      "n_kv_heads": 8,
      "use_biases": false,
      "causal": true,
      "rope_theta": 1000000.0,
      "norm_eps": 0.00001,
      "vocab_size": 131072,
      "model_parallel": 1,
      "tied_embeddings": true,
      "sliding_window": 8192,
      "model_max_length": 131072,
      "multimodal": {
        "whisper_model_args": {
          "encoder_args": {
            "audio_encoding_args": {
              "sampling_rate": 16000,
              "frame_rate": 12.5,
              "num_mel_bins": 128,
              "hop_length": 160,
              "window_size": 400,
              "chunk_length_s": null,
              "global_log_mel_max": 1.5,
              "transcription_format": "streaming"
            },
            "dim": 1280,
            "n_layers": 32,
            "head_dim": 64,
            "hidden_dim": 5120,
            "n_heads": 32,
            "vocab_size": 131072,
            "n_kv_heads": 32,
            "use_biases": true,
            "use_cache": false,
            "rope_theta": 1000000.0,
            "causal": true,
            "norm_eps": 0.00001,
            "pos_embed": "rope",
            "max_source_positions": null,
            "ffn_type": "swiglu",
            "norm_type": "rms_norm",
            "sliding_window": 750
          },
          "downsample_args": {
            "downsample_factor": 4
          }
        }
      },
      "ada_rms_norm_t_cond": true,
      "ada_rms_norm_t_cond_dim": 32
    }"#;

    const REALTIME_HF_CONFIG_JSON: &str = r#"{
      "architectures": ["VoxtralRealtimeForConditionalGeneration"],
      "audio_config": {
        "head_dim": 64,
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "model_type": "voxtral_realtime_encoder",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "num_mel_bins": 128,
        "rms_norm_eps": 0.00001,
        "sliding_window": 750,
        "vocab_size": 131072
      },
      "audio_length_per_tok": 8,
      "default_num_delay_tokens": 6,
      "downsample_factor": 4,
      "dtype": "bfloat16",
      "hidden_size": 3072,
      "model_type": "voxtral_realtime",
      "text_config": {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 128,
        "hidden_size": 3072,
        "intermediate_size": 9216,
        "max_position_embeddings": 131072,
        "model_type": "voxtral_realtime_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 26,
        "num_key_value_heads": 8,
        "rms_norm_eps": 0.00001,
        "sliding_window": 8192,
        "tie_word_embeddings": true,
        "use_cache": true,
        "vocab_size": 131072
      }
    }"#;

    #[test]
    fn parses_realtime_params_shape() {
        let config = VoxtralRealtimeConfig::from_json_str(REALTIME_PARAMS_JSON).unwrap();

        assert_eq!(config.model_type(), "voxtral_realtime");
        assert_eq!(config.dim, 3072);
        assert_eq!(config.sample_rate(), 16_000);
        assert_eq!(config.num_mel_bins(), 128);
        assert_eq!(config.frame_rate(), 12.5);
        assert_eq!(config.downsample_factor(), 4);
        assert_eq!(
            config
                .multimodal
                .whisper_model_args
                .encoder_args
                .sliding_window,
            750
        );
    }

    #[test]
    fn parses_realtime_transformers_config_shape() {
        let params = VoxtralRealtimeConfig::from_json_str(REALTIME_PARAMS_JSON).unwrap();
        let config =
            VoxtralRealtimeTransformersConfig::from_json_str(REALTIME_HF_CONFIG_JSON).unwrap();

        config.validate_against_params(&params).unwrap();
        assert_eq!(config.audio_length_per_tok, 8);
        assert_eq!(config.default_num_delay_tokens, 6);
        assert_eq!(config.dtype, "bfloat16");
    }

    #[test]
    fn expected_realtime_tensor_contract_matches_references() {
        let config = VoxtralRealtimeConfig::from_json_str(REALTIME_PARAMS_JSON).unwrap();
        let tensors = expected_realtime_tensors(&config);

        assert_eq!(tensors.len(), REALTIME_EXPECTED_TENSOR_COUNT);
        assert!(tensors.iter().any(|tensor| {
            tensor.name == "mm_streams_embeddings.embedding_module.tok_embeddings.weight"
                && tensor.dtype == SAFETENSORS_BF16
                && tensor.shape == [131072, 3072]
        }));
        assert!(tensors.iter().any(|tensor| {
            tensor.name == "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.weight"
                && tensor.dtype == SAFETENSORS_BF16
                && tensor.shape == [1280, 128, 3]
        }));
        assert!(tensors.iter().any(|tensor| {
            tensor.name
                == "mm_streams_embeddings.embedding_module.audio_language_projection.0.weight"
                && tensor.dtype == SAFETENSORS_BF16
                && tensor.shape == [3072, 5120]
        }));
        assert!(tensors.iter().any(|tensor| {
            tensor.name == "layers.0.ada_rms_norm_t_cond.0.weight"
                && tensor.dtype == SAFETENSORS_BF16
                && tensor.shape == [32, 3072]
        }));
        assert!(tensors.iter().any(|tensor| {
            tensor.name == "layers.25.feed_forward.w3.weight"
                && tensor.dtype == SAFETENSORS_BF16
                && tensor.shape == [9216, 3072]
        }));
        assert!(tensors
            .iter()
            .any(|tensor| tensor.name == "norm.weight" && tensor.dtype == SAFETENSORS_BF16));
        assert!(tensors
            .iter()
            .all(|tensor| tensor.dtype == SAFETENSORS_BF16));
    }

    #[test]
    fn computes_realtime_audio_token_timing() {
        let config = VoxtralRealtimeConfig::from_json_str(REALTIME_PARAMS_JSON).unwrap();

        assert_eq!(realtime_raw_audio_length_per_token(&config).unwrap(), 1280);
        assert_eq!(realtime_audio_frames_per_token(&config).unwrap(), 8);
        assert_eq!(
            realtime_num_audio_tokens_for_samples(&config, 7_680).unwrap(),
            6
        );
        assert_eq!(realtime_num_delay_tokens(&config, 480).unwrap(), 6);
        assert_eq!(realtime_num_delay_tokens(&config, 80).unwrap(), 1);
        assert_eq!(realtime_num_delay_tokens(&config, 2_400).unwrap(), 30);
        assert_eq!(
            realtime_num_audio_tokens_for_samples(&config, 16_000).unwrap(),
            13
        );
    }

    #[test]
    fn builds_default_realtime_streaming_prompt() {
        let prompt = build_realtime_streaming_prompt(6);

        assert_eq!(prompt.left_pad_tokens, 32);
        assert_eq!(prompt.delay_tokens, 6);
        assert_eq!(prompt.input_ids.len(), 39);
        assert_eq!(prompt.input_ids[0], REALTIME_BOS_TOKEN_ID);
        assert!(prompt.input_ids[1..]
            .iter()
            .all(|id| *id == REALTIME_STREAMING_PAD_TOKEN_ID));
    }

    #[test]
    fn plans_and_applies_offline_streaming_audio_padding() {
        let config = VoxtralRealtimeConfig::from_json_str(REALTIME_PARAMS_JSON).unwrap();
        let samples = vec![0.25; 16_000];
        let plan = plan_realtime_audio_padding(&config, samples.len(), 6).unwrap();

        assert_eq!(plan.raw_audio_length_per_token, 1280);
        assert_eq!(plan.left_pad_samples, 40_960);
        assert_eq!(plan.align_pad_samples, 640);
        assert_eq!(plan.right_pad_tokens, 17);
        assert_eq!(plan.right_pad_samples, 22_400);
        assert_eq!(plan.padded_samples, 79_360);

        let padded = pad_realtime_audio(&samples, &plan);
        assert_eq!(padded.len(), plan.padded_samples);
        assert!(padded[..plan.left_pad_samples]
            .iter()
            .all(|sample| *sample == 0.0));
        assert_eq!(
            &padded[plan.left_pad_samples..plan.left_pad_samples + samples.len()],
            samples.as_slice()
        );
        assert!(padded[plan.left_pad_samples + samples.len()..]
            .iter()
            .all(|sample| *sample == 0.0));
    }

    #[test]
    fn local_realtime_metadata_resolution_does_not_require_weights() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "voice-voxtral-realtime-assets-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(REALTIME_CONFIG_FILE), REALTIME_PARAMS_JSON).unwrap();
        std::fs::write(dir.join(REALTIME_HF_CONFIG_FILE), REALTIME_HF_CONFIG_JSON).unwrap();

        let model = VoxtralRealtimeModel::load_metadata_from_dir(&dir).unwrap();

        assert_eq!(model.config().model_type(), "voxtral_realtime");
        assert!(model.transformers_config().is_some());
        assert!(model.weights().is_none());
        assert!(VoxtralRealtimeModel::load_from_dir(&dir).is_err());

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn loads_local_realtime_metadata_when_env_is_set() {
        let Ok(dir) = std::env::var("VOXTRAL_REALTIME_LOCAL_DIR") else {
            return;
        };

        let model = VoxtralRealtimeModel::load_metadata_from_dir(dir).unwrap();
        let tokenizer = model.tokenizer().unwrap();

        assert_eq!(model.config().sample_rate(), 16_000);
        assert_eq!(model.config().num_mel_bins(), 128);
        assert_eq!(model.config().downsample_factor(), 4);
        assert_eq!(tokenizer.special_token_id("[STREAMING_PAD]"), Some(32));
        assert_eq!(tokenizer.special_token_id("[STREAMING_WORD]"), Some(33));
        assert_eq!(tokenizer.special_token_id("[REPEAT_AUDIO_TEXT]"), Some(34));
        assert_eq!(tokenizer.audio.transcription_delay_ms, Some(480));
    }

    #[test]
    fn validates_local_realtime_checkpoint_when_env_is_set() {
        let Ok(dir) = std::env::var("VOXTRAL_REALTIME_LOCAL_DIR") else {
            return;
        };
        if std::env::var("VOXTRAL_REALTIME_LOAD_FULL").as_deref() != Ok("1") {
            return;
        }

        let model = VoxtralRealtimeModel::load_from_dir(dir).unwrap();
        let summary = model.checkpoint_summary().unwrap();

        assert_eq!(summary.tensor_count, REALTIME_EXPECTED_TENSOR_COUNT);
        assert_eq!(
            summary.expected_tensor_count,
            REALTIME_EXPECTED_TENSOR_COUNT
        );
        assert_eq!(
            summary
                .component_counts
                .get(&crate::WeightComponent::LanguageModel)
                .copied(),
            Some(286)
        );
        assert_eq!(
            summary
                .component_counts
                .get(&crate::WeightComponent::RealtimeStreams)
                .copied(),
            Some(424)
        );
        assert_eq!(
            summary
                .component_counts
                .get(&crate::WeightComponent::FinalNorm)
                .copied(),
            Some(1)
        );
    }
}
