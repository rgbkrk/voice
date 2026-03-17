use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{Lstm, LstmBuilder, LstmInput};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use mlx_macros::ModuleParameters;

/// Bidirectional LSTM that processes a sequence in both forward and backward directions
/// and concatenates the outputs.
///
/// Given input of shape `(batch, seq_len, features)`, the output shape is
/// `(batch, seq_len, 2 * hidden_size)`.
#[derive(Debug, ModuleParameters)]
pub struct BiLstm {
    #[param]
    pub forward_lstm: Lstm,
    #[param]
    pub backward_lstm: Lstm,
}

impl BiLstm {
    /// Create a new BiLstm with the given input and hidden sizes.
    ///
    /// Each direction has its own LSTM with `hidden_size` units, so the concatenated
    /// output has dimension `2 * hidden_size`.
    pub fn new(input_size: i32, hidden_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            forward_lstm: LstmBuilder::new(input_size, hidden_size).build()?,
            backward_lstm: LstmBuilder::new(input_size, hidden_size).build()?,
        })
    }

    /// Run the bidirectional LSTM on input `x`.
    ///
    /// Input shape: `(batch, seq_len, features)` or `(seq_len, features)`.
    /// Output shape: `(batch, seq_len, 2 * hidden_size)`.
    ///
    /// Returns `(output, ())` to match the expected LSTM return convention.
    pub fn forward(&mut self, x: &Array) -> Result<(Array, ()), Exception> {
        // Ensure 3-D input: (batch, seq_len, features)
        let x = if x.ndim() == 2 {
            x.reshape(&[1, x.shape()[0], x.shape()[1]])?
        } else {
            x.clone()
        };

        let seq_len = x.shape()[1];

        // Forward direction: process the full sequence
        // Lstm.step processes (batch, seq_len, features) -> ((batch, seq_len, hidden_size), cell)
        let fwd_input = LstmInput::from(&x);
        let (fwd_out, _fwd_cell) = self.forward_lstm.forward(fwd_input)?;

        // Backward direction: reverse the input along the sequence dimension (axis 1)
        let rev_indices: Vec<i32> = (0..seq_len).rev().collect();
        let rev_idx = Array::from_slice(&rev_indices, &[seq_len]);
        // x_rev = x[:, rev_indices, :]
        let x_rev = x.index((0.., &rev_idx, 0..));

        let bwd_input = LstmInput::from(&x_rev);
        let (bwd_out_rev, _bwd_cell) = self.backward_lstm.forward(bwd_input)?;

        // Reverse the backward output back to the original temporal order
        let bwd_out = bwd_out_rev.index((0.., &rev_idx, 0..));

        // Concatenate forward and backward outputs along the feature dimension (axis 2)
        let output = mlx_rs::ops::concatenate_axis(&[&fwd_out, &bwd_out], 2)?;
        Ok((output, ()))
    }

    /// Set training mode on both LSTMs.
    pub fn training_mode(&mut self, _mode: bool) {
        // mlx-rs Lstm doesn't have training mode, but we keep the interface
    }
}
