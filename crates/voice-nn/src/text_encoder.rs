use quill_mlx_macros::ModuleParameters;
use quill_mlx::builder::Builder;
use quill_mlx::error::Exception;
use quill_mlx::module::Module;
use quill_mlx::nn::{Dropout, DropoutBuilder, Embedding, LayerNorm, LayerNormBuilder};
use quill_mlx::ops::{concatenate_axis, expand_dims, zeros};
use quill_mlx::Array;

use super::conv_weighted::ConvWeighted;
use super::lstm::BiLstm;

// ---------------------------------------------------------------------------
// Input type
// ---------------------------------------------------------------------------

pub struct TextEncoderInput<'a> {
    pub x: &'a Array,
    pub input_lengths: &'a Array,
    pub mask: &'a Array,
}

// ---------------------------------------------------------------------------
// ConvBlock: one CNN block (conv + layernorm + leaky_relu + dropout)
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
pub struct ConvBlock {
    #[param]
    pub conv: ConvWeighted,
    #[param]
    pub norm: LayerNorm,
    #[param]
    pub dropout: Dropout,
}

impl ConvBlock {
    pub fn new(channels: i32, kernel_size: i32, padding: i32) -> Result<Self, Exception> {
        let conv = ConvWeighted::new_simple(channels, channels, kernel_size, padding)?;
        let norm = LayerNormBuilder::new(channels).build()?;
        let dropout = DropoutBuilder::new()
            .p(0.2)
            .build()
            .map_err(|e| Exception::custom(e.to_string()))?;
        Ok(Self {
            conv,
            norm,
            dropout,
        })
    }
}

/// Apply leaky ReLU with negative slope 0.2.
fn leaky_relu_02(x: &Array) -> Result<Array, Exception> {
    quill_mlx::nn::leaky_relu(x, 0.2)
}

// ---------------------------------------------------------------------------
// TextEncoder
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
pub struct TextEncoder {
    #[param]
    pub embedding: Embedding,
    #[param]
    pub cnn: Vec<ConvBlock>,
    #[param]
    pub lstm: BiLstm,
}

impl TextEncoder {
    pub fn new(
        channels: i32,
        kernel_size: i32,
        depth: i32,
        n_symbols: i32,
    ) -> Result<Self, Exception> {
        let embedding = Embedding::new(n_symbols, channels)?;
        let padding = (kernel_size - 1) / 2;

        let mut cnn = Vec::with_capacity(depth as usize);
        for _ in 0..depth {
            cnn.push(ConvBlock::new(channels, kernel_size, padding)?);
        }

        let lstm = BiLstm::new(channels, channels / 2)?;

        Ok(Self {
            embedding,
            cnn,
            lstm,
        })
    }
}

impl<'a> Module<TextEncoderInput<'a>> for TextEncoder {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: TextEncoderInput<'a>) -> Result<Array, Exception> {
        let x = input.x;
        let mask = input.mask;

        // Embedding: (B, seq) -> (B, seq, channels)
        let mut x = self.embedding.forward(x)?;

        // Transpose to (B, channels, seq)
        x = x.transpose_axes(&[0, 2, 1])?;

        // Expand mask: (B, seq) -> (B, 1, seq)
        let m = expand_dims(mask, 1)?;

        // Apply mask: zero out masked positions
        let zero = Array::from_f32(0.0);
        x = quill_mlx::ops::r#where(&m, &zero, &x)?;

        // CNN blocks
        for block in self.cnn.iter_mut() {
            // Conv expects (B, seq, channels), so swap axes
            x = x.swap_axes(2, 1)?; // (B, seq, channels)
            x = block.conv.forward(&x)?;
            x = x.swap_axes(2, 1)?; // (B, channels, seq)

            // LayerNorm expects (B, seq, channels)
            x = x.swap_axes(2, 1)?; // (B, seq, channels)
            x = block.norm.forward(&x)?;
            x = x.swap_axes(2, 1)?; // (B, channels, seq)

            // LeakyReLU (works on any layout)
            x = leaky_relu_02(&x)?;

            // Dropout (works on any layout)
            x = block.dropout.forward(&x)?;

            // Re-apply mask
            x = quill_mlx::ops::r#where(&m, &zero, &x)?;
        }

        // LSTM: needs (B, seq, channels)
        x = x.swap_axes(2, 1)?; // (B, seq, channels)
        let (lstm_out, _) = self.lstm.forward(&x)?;
        x = lstm_out;
        x = x.swap_axes(2, 1)?; // (B, channels, seq)

        // Pad output to match mask length if needed
        let x_channels = x.shape()[1];
        let x_seq = x.shape()[2];
        let m_seq = m.shape()[2]; // m is (B, 1, seq)

        if x_seq < m_seq {
            // Pad with zeros along the sequence dimension to match mask length
            let batch = x.shape()[0];
            let pad_len = m_seq - x_seq;
            let pad_zeros = zeros::<f32>(&[batch, x_channels, pad_len])?;
            x = concatenate_axis(&[&x, &pad_zeros], -1)?;
        }

        // Final mask application
        x = quill_mlx::ops::r#where(&m, &zero, &x)?;

        Ok(x)
    }

    fn training_mode(&mut self, mode: bool) {
        for block in self.cnn.iter_mut() {
            block.dropout.training_mode(mode);
        }
    }
}
