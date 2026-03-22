use quill_mlx::builder::Builder;
use quill_mlx::error::Exception;
use quill_mlx::module::Module;
use quill_mlx::nn::{
    Dropout, DropoutBuilder, Embedding, Gelu, LayerNorm, LayerNormBuilder, Linear, LinearBuilder,
};
use quill_mlx::ops::{expand_dims, softmax_axis, zeros_like};
use quill_mlx::Array;
use quill_mlx_macros::ModuleParameters;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AlbertConfig {
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub max_position_embeddings: i32,
    #[serde(default = "default_embedding_size")]
    pub embedding_size: i32,
    #[serde(default = "default_inner_group_num")]
    pub inner_group_num: i32,
    #[serde(default = "default_num_hidden_groups")]
    pub num_hidden_groups: i32,
    #[serde(default = "default_hidden_dropout_prob")]
    pub hidden_dropout_prob: f32,
    #[serde(default = "default_attention_probs_dropout_prob")]
    pub attention_probs_dropout_prob: f32,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: i32,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
}

fn default_embedding_size() -> i32 {
    128
}
fn default_inner_group_num() -> i32 {
    1
}
fn default_num_hidden_groups() -> i32 {
    1
}
fn default_hidden_dropout_prob() -> f32 {
    0.1
}
fn default_attention_probs_dropout_prob() -> f32 {
    0.1
}
fn default_type_vocab_size() -> i32 {
    2
}
fn default_layer_norm_eps() -> f32 {
    1e-12
}
fn default_vocab_size() -> i32 {
    30522
}

// ---------------------------------------------------------------------------
// Input types
// ---------------------------------------------------------------------------

pub struct AlbertEmbeddingsInput<'a> {
    pub input_ids: &'a Array,
    pub token_type_ids: Option<&'a Array>,
    pub position_ids: Option<&'a Array>,
}

pub struct AlbertLayerInput<'a> {
    pub hidden_states: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

// ---------------------------------------------------------------------------
// AlbertEmbeddings
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
pub struct AlbertEmbeddings {
    #[param]
    pub word_embeddings: Embedding,
    #[param]
    pub position_embeddings: Embedding,
    #[param]
    pub token_type_embeddings: Embedding,
    #[param]
    pub layer_norm: LayerNorm,
    #[param]
    pub dropout: Dropout,
}

impl AlbertEmbeddings {
    pub fn new(config: &AlbertConfig) -> Result<Self, Exception> {
        let word_embeddings = Embedding::new(config.vocab_size, config.embedding_size)?;
        let position_embeddings =
            Embedding::new(config.max_position_embeddings, config.embedding_size)?;
        let token_type_embeddings = Embedding::new(config.type_vocab_size, config.embedding_size)?;
        let layer_norm = LayerNormBuilder::new(config.embedding_size)
            .eps(config.layer_norm_eps)
            .build()?;
        let dropout = DropoutBuilder::new()
            .p(config.hidden_dropout_prob)
            .build()
            .map_err(|e| Exception::custom(e.to_string()))?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        })
    }
}

impl<'a> Module<AlbertEmbeddingsInput<'a>> for AlbertEmbeddings {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: AlbertEmbeddingsInput<'a>) -> Result<Array, Exception> {
        let input_ids = input.input_ids;
        let seq_length = input_ids.shape()[1];

        // position_ids: default to arange(seq_length)[None, :]
        let position_ids_owned;
        let position_ids = match input.position_ids {
            Some(p) => p,
            None => {
                position_ids_owned =
                    Array::arange::<_, i32>(None, seq_length, None)?.reshape(&[1, -1])?;
                &position_ids_owned
            }
        };

        // token_type_ids: default to zeros_like(input_ids)
        let token_type_ids_owned;
        let token_type_ids = match input.token_type_ids {
            Some(t) => t,
            None => {
                token_type_ids_owned = zeros_like(input_ids)?;
                &token_type_ids_owned
            }
        };

        let words = self.word_embeddings.forward(input_ids)?;
        let positions = self.position_embeddings.forward(position_ids)?;
        let token_types = self.token_type_embeddings.forward(token_type_ids)?;

        let embeddings = &(&words + &positions) + &token_types;
        let normed = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&normed)
    }

    fn training_mode(&mut self, mode: bool) {
        self.dropout.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// AlbertSelfAttention
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
pub struct AlbertSelfAttention {
    pub num_attention_heads: i32,
    pub attention_head_size: i32,
    pub all_head_size: i32,

    #[param]
    pub query: Linear,
    #[param]
    pub key: Linear,
    #[param]
    pub value: Linear,
    #[param]
    pub dense: Linear,
    #[param]
    pub layer_norm: LayerNorm,
    #[param]
    pub dropout: Dropout,
}

impl AlbertSelfAttention {
    pub fn new(config: &AlbertConfig) -> Result<Self, Exception> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;

        let query = LinearBuilder::new(config.hidden_size, all_head_size).build()?;
        let key = LinearBuilder::new(config.hidden_size, all_head_size).build()?;
        let value = LinearBuilder::new(config.hidden_size, all_head_size).build()?;
        let dense = LinearBuilder::new(config.hidden_size, config.hidden_size).build()?;
        let layer_norm = LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .build()?;
        let dropout = DropoutBuilder::new()
            .p(config.attention_probs_dropout_prob)
            .build()
            .map_err(|e| Exception::custom(e.to_string()))?;

        Ok(Self {
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
            query,
            key,
            value,
            dense,
            layer_norm,
            dropout,
        })
    }

    /// Reshape from (batch, seq, all_head_size) to (batch, num_heads, seq, head_size)
    fn transpose_for_scores(&self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        // shape[:-1] + (num_heads, head_size)
        let batch = shape[0];
        let seq = shape[1];
        let reshaped = x.reshape(&[
            batch,
            seq,
            self.num_attention_heads,
            self.attention_head_size,
        ])?;
        // transpose to (batch, num_heads, seq, head_size)
        reshaped.transpose_axes(&[0, 2, 1, 3])
    }
}

impl<'a> Module<AlbertLayerInput<'a>> for AlbertSelfAttention {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: AlbertLayerInput<'a>) -> Result<Array, Exception> {
        let hidden_states = input.hidden_states;

        let q_proj = self.query.forward(hidden_states)?;
        let k_proj = self.key.forward(hidden_states)?;
        let v_proj = self.value.forward(hidden_states)?;
        let q = self.transpose_for_scores(&q_proj)?;
        let k = self.transpose_for_scores(&k_proj)?;
        let v = self.transpose_for_scores(&v_proj)?;

        // scores = q @ k^T / sqrt(head_size)
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let scale = Array::from_f32((self.attention_head_size as f32).sqrt());
        let mut scores = q.matmul(&k_t)?;
        scores = &scores / &scale;

        // Apply attention mask if provided
        if let Some(mask) = input.attention_mask {
            scores = &scores + mask;
        }

        // softmax + dropout
        let probs = softmax_axis(&scores, -1, None)?;
        let probs = self.dropout.forward(&probs)?;

        // context = probs @ v -> (batch, num_heads, seq, head_size)
        let context = probs.matmul(&v)?;
        // transpose back to (batch, seq, num_heads, head_size)
        let context = context.transpose_axes(&[0, 2, 1, 3])?;
        // reshape to (batch, seq, all_head_size)
        let shape = context.shape();
        let batch = shape[0];
        let seq = shape[1];
        let context = context.reshape(&[batch, seq, self.all_head_size])?;

        // output projection + residual + layer norm
        let context = self.dense.forward(&context)?;
        self.layer_norm.forward(&(&context + hidden_states))
    }

    fn training_mode(&mut self, mode: bool) {
        self.dropout.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// AlbertLayer
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
pub struct AlbertLayer {
    #[param]
    pub attention: AlbertSelfAttention,
    #[param]
    pub full_layer_layer_norm: LayerNorm,
    #[param]
    pub ffn: Linear,
    #[param]
    pub ffn_output: Linear,
    #[param]
    pub activation: Gelu,
}

impl AlbertLayer {
    pub fn new(config: &AlbertConfig) -> Result<Self, Exception> {
        let attention = AlbertSelfAttention::new(config)?;
        let full_layer_layer_norm = LayerNormBuilder::new(config.hidden_size)
            .eps(config.layer_norm_eps)
            .build()?;
        let ffn = LinearBuilder::new(config.hidden_size, config.intermediate_size).build()?;
        let ffn_output =
            LinearBuilder::new(config.intermediate_size, config.hidden_size).build()?;
        let activation = Gelu::new();

        Ok(Self {
            attention,
            full_layer_layer_norm,
            ffn,
            ffn_output,
            activation,
        })
    }
}

impl<'a> Module<AlbertLayerInput<'a>> for AlbertLayer {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: AlbertLayerInput<'a>) -> Result<Array, Exception> {
        let attention_output = self.attention.forward(AlbertLayerInput {
            hidden_states: input.hidden_states,
            attention_mask: input.attention_mask,
        })?;

        let ffn_out = self.ffn.forward(&attention_output)?;
        let ffn_out = self.activation.forward(&ffn_out)?;
        let ffn_out = self.ffn_output.forward(&ffn_out)?;

        self.full_layer_layer_norm
            .forward(&(&ffn_out + &attention_output))
    }

    fn training_mode(&mut self, mode: bool) {
        self.attention.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// AlbertLayerGroup
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
pub struct AlbertLayerGroup {
    #[param]
    pub albert_layers: Vec<AlbertLayer>,
}

impl AlbertLayerGroup {
    pub fn new(config: &AlbertConfig) -> Result<Self, Exception> {
        let mut albert_layers = Vec::with_capacity(config.inner_group_num as usize);
        for _ in 0..config.inner_group_num {
            albert_layers.push(AlbertLayer::new(config)?);
        }
        Ok(Self { albert_layers })
    }
}

impl<'a> Module<AlbertLayerInput<'a>> for AlbertLayerGroup {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: AlbertLayerInput<'a>) -> Result<Array, Exception> {
        let mut hidden_states = input.hidden_states.clone();
        for layer in self.albert_layers.iter_mut() {
            hidden_states = layer.forward(AlbertLayerInput {
                hidden_states: &hidden_states,
                attention_mask: input.attention_mask,
            })?;
        }
        Ok(hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        for layer in self.albert_layers.iter_mut() {
            layer.training_mode(mode);
        }
    }
}

// ---------------------------------------------------------------------------
// AlbertEncoder
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
pub struct AlbertEncoder {
    pub num_hidden_layers: i32,
    pub num_hidden_groups: i32,

    #[param]
    pub embedding_hidden_mapping_in: Linear,
    #[param]
    pub albert_layer_groups: Vec<AlbertLayerGroup>,
}

impl AlbertEncoder {
    pub fn new(config: &AlbertConfig) -> Result<Self, Exception> {
        let embedding_hidden_mapping_in =
            LinearBuilder::new(config.embedding_size, config.hidden_size).build()?;
        let mut albert_layer_groups = Vec::with_capacity(config.num_hidden_groups as usize);
        for _ in 0..config.num_hidden_groups {
            albert_layer_groups.push(AlbertLayerGroup::new(config)?);
        }
        Ok(Self {
            num_hidden_layers: config.num_hidden_layers,
            num_hidden_groups: config.num_hidden_groups,
            embedding_hidden_mapping_in,
            albert_layer_groups,
        })
    }
}

impl<'a> Module<AlbertLayerInput<'a>> for AlbertEncoder {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: AlbertLayerInput<'a>) -> Result<Array, Exception> {
        let mut hidden_states = self
            .embedding_hidden_mapping_in
            .forward(input.hidden_states)?;

        let layers_per_group = self.num_hidden_layers / self.num_hidden_groups;
        for i in 0..self.num_hidden_layers {
            let group_idx = (i / layers_per_group) as usize;
            hidden_states = self.albert_layer_groups[group_idx].forward(AlbertLayerInput {
                hidden_states: &hidden_states,
                attention_mask: input.attention_mask,
            })?;
        }
        Ok(hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        for group in self.albert_layer_groups.iter_mut() {
            group.training_mode(mode);
        }
    }
}

// ---------------------------------------------------------------------------
// CustomAlbert
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
pub struct CustomAlbert {
    pub num_hidden_layers: i32,
    pub num_hidden_groups: i32,

    #[param]
    pub embeddings: AlbertEmbeddings,
    #[param]
    pub encoder: AlbertEncoder,
    #[param]
    pub pooler: Linear,
}

pub struct CustomAlbertInput<'a> {
    pub input_ids: &'a Array,
    pub token_type_ids: Option<&'a Array>,
    pub attention_mask: Option<&'a Array>,
}

pub struct CustomAlbertOutput {
    pub encoder_output: Array,
    pub pooled_output: Array,
}

impl CustomAlbert {
    pub fn new(config: &AlbertConfig) -> Result<Self, Exception> {
        let embeddings = AlbertEmbeddings::new(config)?;
        let encoder = AlbertEncoder::new(config)?;
        let pooler = LinearBuilder::new(config.hidden_size, config.hidden_size).build()?;

        Ok(Self {
            num_hidden_layers: config.num_hidden_layers,
            num_hidden_groups: config.num_hidden_groups,
            embeddings,
            encoder,
            pooler,
        })
    }
}

impl<'a> Module<CustomAlbertInput<'a>> for CustomAlbert {
    type Error = Exception;
    type Output = CustomAlbertOutput;

    fn forward(&mut self, input: CustomAlbertInput<'a>) -> Result<CustomAlbertOutput, Exception> {
        let embedding_output = self.embeddings.forward(AlbertEmbeddingsInput {
            input_ids: input.input_ids,
            token_type_ids: input.token_type_ids,
            position_ids: None,
        })?;

        // Process attention mask: expand dims and invert
        let attention_mask = match input.attention_mask {
            Some(mask) => {
                // mask[:, None, None, :] -> expand to (batch, 1, 1, seq)
                let mask = expand_dims(mask, 1)?;
                let mask = expand_dims(&mask, 1)?;
                // (1.0 - mask) * -10000.0
                let ones = Array::from_f32(1.0);
                let inverted = &ones - &mask;
                let scale = Array::from_f32(-10000.0);
                Some(&inverted * &scale)
            }
            None => None,
        };

        let encoder_output = self.encoder.forward(AlbertLayerInput {
            hidden_states: &embedding_output,
            attention_mask: attention_mask.as_ref(),
        })?;

        // Pool: take first token, apply pooler + tanh
        use quill_mlx::ops::indexing::IndexOp;
        let first_token = encoder_output.index((.., 0));
        let pooled = self.pooler.forward(&first_token)?;
        let pooled = quill_mlx::ops::tanh(&pooled)?;

        Ok(CustomAlbertOutput {
            encoder_output,
            pooled_output: pooled,
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.embeddings.training_mode(mode);
        self.encoder.training_mode(mode);
    }
}
