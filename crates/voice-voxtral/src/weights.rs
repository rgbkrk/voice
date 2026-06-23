use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;

use crate::{Result, VoxtralConfig, VoxtralError};

const SAFETENSORS_HEADER_LEN_BYTES: usize = 8;
const EXPECTED_DTYPE: &str = "BF16";
const VOXTRAL_TTS_CHECKPOINT_TENSORS: usize = 386;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WeightComponent {
    AcousticTransformer,
    AudioTokenizer,
    LanguageModel,
    MultimodalEmbeddings,
    RealtimeStreams,
    FinalNorm,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedTensor {
    pub name: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralCheckpointSummary {
    pub tensor_count: usize,
    pub expected_tensor_count: usize,
    pub component_counts: BTreeMap<WeightComponent, usize>,
    pub file_len: u64,
    pub header_len: usize,
    pub data_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralWeightMetadata {
    tensors: BTreeMap<String, TensorInfo>,
    file_len: u64,
    header_len: usize,
    data_len: u64,
}

#[derive(Debug, Deserialize)]
struct RawTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl VoxtralWeightMetadata {
    pub fn from_safetensors_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();

        let mut header_len_bytes = [0u8; SAFETENSORS_HEADER_LEN_BYTES];
        file.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes)
            .try_into()
            .map_err(|_| VoxtralError::InvalidCheckpoint("safetensors header too large".into()))?;

        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header)?;

        let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)?;
        let mut tensors = BTreeMap::new();
        let mut data_len = 0u64;

        for (name, value) in raw {
            if name == "__metadata__" {
                continue;
            }
            let raw_info: RawTensorInfo = serde_json::from_value(value)?;
            data_len = data_len.max(raw_info.data_offsets[1]);
            tensors.insert(
                name,
                TensorInfo {
                    dtype: raw_info.dtype,
                    shape: raw_info.shape,
                    data_offsets: raw_info.data_offsets,
                },
            );
        }

        let expected_file_len = SAFETENSORS_HEADER_LEN_BYTES as u64 + header_len as u64 + data_len;
        if expected_file_len != file_len {
            return Err(VoxtralError::InvalidCheckpoint(format!(
                "safetensors file length mismatch: header implies {expected_file_len} bytes, file has {file_len}"
            )));
        }

        Ok(Self {
            tensors,
            file_len,
            header_len,
            data_len,
        })
    }

    #[cfg(test)]
    fn from_tensors_for_test(tensors: BTreeMap<String, TensorInfo>) -> Self {
        Self {
            tensors,
            file_len: 0,
            header_len: 0,
            data_len: 0,
        }
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.tensors
            .iter()
            .map(|(name, info)| (name.as_str(), info))
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    pub fn header_len(&self) -> usize {
        self.header_len
    }

    pub fn data_len(&self) -> u64 {
        self.data_len
    }

    pub fn component_counts(&self) -> BTreeMap<WeightComponent, usize> {
        let mut counts = BTreeMap::new();
        for name in self.tensors.keys() {
            *counts.entry(component_for_name(name)).or_insert(0) += 1;
        }
        counts
    }

    pub fn summary(&self) -> VoxtralCheckpointSummary {
        self.summary_with_expected_tensor_count(VOXTRAL_TTS_CHECKPOINT_TENSORS)
    }

    pub fn summary_with_expected_tensor_count(
        &self,
        expected_tensor_count: usize,
    ) -> VoxtralCheckpointSummary {
        VoxtralCheckpointSummary {
            tensor_count: self.tensor_count(),
            expected_tensor_count,
            component_counts: self.component_counts(),
            file_len: self.file_len,
            header_len: self.header_len,
            data_len: self.data_len,
        }
    }

    pub fn validate_for_config(&self, config: &VoxtralConfig) -> Result<VoxtralCheckpointSummary> {
        for expected in expected_tensors(config) {
            self.expect_tensor(&expected.name, &expected.shape)?;
        }
        Ok(self.summary())
    }

    fn expect_tensor(&self, name: &str, shape: &[usize]) -> Result<()> {
        let tensor = self.tensors.get(name).ok_or_else(|| {
            VoxtralError::InvalidCheckpoint(format!("missing required tensor {name}"))
        })?;
        if tensor.dtype != EXPECTED_DTYPE {
            return Err(VoxtralError::InvalidCheckpoint(format!(
                "tensor {name} has dtype {}, expected {EXPECTED_DTYPE}",
                tensor.dtype
            )));
        }
        if tensor.shape != shape {
            return Err(VoxtralError::InvalidCheckpoint(format!(
                "tensor {name} has shape {:?}, expected {:?}",
                tensor.shape, shape
            )));
        }
        Ok(())
    }
}

pub fn expected_tensors(config: &VoxtralConfig) -> Vec<ExpectedTensor> {
    let mut tensors = Vec::new();
    add_language_model_expected_tensors(config, &mut tensors);
    add_multimodal_embedding_expected_tensors(config, &mut tensors);
    add_acoustic_transformer_expected_tensors(config, &mut tensors);
    add_audio_tokenizer_expected_tensors(config, &mut tensors);
    tensors
}

fn add_language_model_expected_tensors(config: &VoxtralConfig, tensors: &mut Vec<ExpectedTensor>) {
    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let q_dim = config.n_heads * config.head_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;

    for layer in 0..config.n_layers {
        let prefix = format!("layers.{layer}");
        expected(
            tensors,
            format!("{prefix}.attention.wq.weight"),
            [q_dim, dim],
        );
        expected(
            tensors,
            format!("{prefix}.attention.wk.weight"),
            [kv_dim, dim],
        );
        expected(
            tensors,
            format!("{prefix}.attention.wv.weight"),
            [kv_dim, dim],
        );
        expected(
            tensors,
            format!("{prefix}.attention.wo.weight"),
            [dim, q_dim],
        );
        expected(tensors, format!("{prefix}.attention_norm.weight"), [dim]);
        expected(tensors, format!("{prefix}.ffn_norm.weight"), [dim]);
        expected(
            tensors,
            format!("{prefix}.feed_forward.w1.weight"),
            [hidden_dim, dim],
        );
        expected(
            tensors,
            format!("{prefix}.feed_forward.w2.weight"),
            [dim, hidden_dim],
        );
        expected(
            tensors,
            format!("{prefix}.feed_forward.w3.weight"),
            [hidden_dim, dim],
        );
    }

    expected(tensors, "norm.weight", [dim]);
}

fn add_multimodal_embedding_expected_tensors(
    config: &VoxtralConfig,
    tensors: &mut Vec<ExpectedTensor>,
) {
    let audio_model = &config.multimodal.audio_model_args;
    let audio_vocab = round_up_to_multiple(
        audio_model.semantic_codebook_size
            + 2
            + (audio_model.acoustic_codebook_size + 2) * audio_model.n_acoustic_codebook,
        128,
    );

    expected(
        tensors,
        "mm_audio_embeddings.tok_embeddings.weight",
        [config.vocab_size, config.dim],
    );
    expected(
        tensors,
        "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight",
        [audio_vocab, config.dim],
    );
}

fn add_acoustic_transformer_expected_tensors(
    config: &VoxtralConfig,
    tensors: &mut Vec<ExpectedTensor>,
) {
    let audio_model = &config.multimodal.audio_model_args;
    let acoustic = &audio_model.acoustic_transformer_args;
    let q_dim = acoustic.n_heads * acoustic.head_dim;
    let kv_dim = acoustic.n_kv_heads * acoustic.head_dim;
    let semantic_output = round_up_to_multiple(audio_model.semantic_codebook_size + 2, 128);

    expected(
        tensors,
        "acoustic_transformer.input_projection.weight",
        [acoustic.dim, audio_model.n_acoustic_codebook],
    );
    expected(
        tensors,
        "acoustic_transformer.time_projection.weight",
        [acoustic.dim, acoustic.dim],
    );
    expected(
        tensors,
        "acoustic_transformer.llm_projection.weight",
        [acoustic.dim, acoustic.input_dim],
    );
    expected(
        tensors,
        "acoustic_transformer.semantic_codebook_output.weight",
        [semantic_output, acoustic.dim],
    );
    expected(
        tensors,
        "acoustic_transformer.acoustic_codebook_output.weight",
        [audio_model.n_acoustic_codebook, acoustic.dim],
    );
    expected(tensors, "acoustic_transformer.norm.weight", [acoustic.dim]);

    for layer in 0..acoustic.n_layers {
        let prefix = format!("acoustic_transformer.layers.{layer}");
        expected(
            tensors,
            format!("{prefix}.attention.wq.weight"),
            [q_dim, acoustic.dim],
        );
        expected(
            tensors,
            format!("{prefix}.attention.wk.weight"),
            [kv_dim, acoustic.dim],
        );
        expected(
            tensors,
            format!("{prefix}.attention.wv.weight"),
            [kv_dim, acoustic.dim],
        );
        expected(
            tensors,
            format!("{prefix}.attention.wo.weight"),
            [acoustic.dim, q_dim],
        );
        expected(
            tensors,
            format!("{prefix}.attention_norm.weight"),
            [acoustic.dim],
        );
        expected(tensors, format!("{prefix}.ffn_norm.weight"), [acoustic.dim]);
        expected(
            tensors,
            format!("{prefix}.feed_forward.w1.weight"),
            [acoustic.hidden_dim, acoustic.dim],
        );
        expected(
            tensors,
            format!("{prefix}.feed_forward.w2.weight"),
            [acoustic.dim, acoustic.hidden_dim],
        );
        expected(
            tensors,
            format!("{prefix}.feed_forward.w3.weight"),
            [acoustic.hidden_dim, acoustic.dim],
        );
    }
}

fn add_audio_tokenizer_expected_tensors(config: &VoxtralConfig, tensors: &mut Vec<ExpectedTensor>) {
    let tokenizer = &config.multimodal.audio_tokenizer_args;
    let latent_dim = tokenizer.semantic_dim + tokenizer.acoustic_dim;
    let first_decoder_kernel = parse_csv_usize(&tokenizer.decoder_convs_kernels_str)
        .first()
        .copied()
        .unwrap_or(3);

    expected(
        tensors,
        "audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original0",
        [tokenizer.dim, 1, 1],
    );
    expected(
        tensors,
        "audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original1",
        [tokenizer.dim, latent_dim, first_decoder_kernel],
    );
    expected(
        tensors,
        "audio_tokenizer.quantizer.semantic_codebook.cluster_usage",
        [tokenizer.semantic_codebook_size],
    );
    expected(
        tensors,
        "audio_tokenizer.quantizer.semantic_codebook.embedding_sum",
        [tokenizer.semantic_codebook_size, tokenizer.semantic_dim],
    );
}

fn expected<const N: usize>(
    tensors: &mut Vec<ExpectedTensor>,
    name: impl Into<String>,
    shape: [usize; N],
) {
    tensors.push(ExpectedTensor {
        name: name.into(),
        shape: shape.to_vec(),
    });
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    multiple * value.div_ceil(multiple)
}

fn parse_csv_usize(raw: &str) -> Vec<usize> {
    raw.split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect()
}

fn component_for_name(name: &str) -> WeightComponent {
    if name.starts_with("acoustic_transformer.") {
        WeightComponent::AcousticTransformer
    } else if name.starts_with("audio_tokenizer.") {
        WeightComponent::AudioTokenizer
    } else if name.starts_with("layers.") {
        WeightComponent::LanguageModel
    } else if name.starts_with("mm_audio_embeddings.") {
        WeightComponent::MultimodalEmbeddings
    } else if name.starts_with("mm_streams_embeddings.") {
        WeightComponent::RealtimeStreams
    } else if name == "norm.weight" {
        WeightComponent::FinalNorm
    } else {
        WeightComponent::Other
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    const PARAMS_JSON: &str = crate::config::tests::PARAMS_JSON;

    #[test]
    fn expected_tensor_contract_matches_voxtral_config() {
        let config = VoxtralConfig::from_json_str(PARAMS_JSON).unwrap();
        let expected = expected_tensors(&config);

        assert_eq!(expected.len(), 274);
        assert!(expected.iter().any(|tensor| {
            tensor.name == "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
                && tensor.shape == [9088, 3072]
        }));
        assert!(expected.iter().any(|tensor| {
            tensor.name == "acoustic_transformer.semantic_codebook_output.weight"
                && tensor.shape == [8320, 3072]
        }));
    }

    #[test]
    fn validates_required_tensor_shapes() {
        let config = VoxtralConfig::from_json_str(PARAMS_JSON).unwrap();
        let tensors = expected_tensors(&config)
            .into_iter()
            .map(|expected| {
                (
                    expected.name,
                    TensorInfo {
                        dtype: EXPECTED_DTYPE.to_string(),
                        shape: expected.shape,
                        data_offsets: [0, 0],
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let metadata = VoxtralWeightMetadata::from_tensors_for_test(tensors);

        metadata.validate_for_config(&config).unwrap();
    }

    #[test]
    fn rejects_shape_mismatch() {
        let config = VoxtralConfig::from_json_str(PARAMS_JSON).unwrap();
        let mut tensors = expected_tensors(&config)
            .into_iter()
            .map(|expected| {
                (
                    expected.name,
                    TensorInfo {
                        dtype: EXPECTED_DTYPE.to_string(),
                        shape: expected.shape,
                        data_offsets: [0, 0],
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        tensors
            .get_mut("norm.weight")
            .unwrap()
            .shape
            .push(config.dim);
        let metadata = VoxtralWeightMetadata::from_tensors_for_test(tensors);
        let err = metadata.validate_for_config(&config).unwrap_err();

        assert!(matches!(err, VoxtralError::InvalidCheckpoint(_)));
    }
}
