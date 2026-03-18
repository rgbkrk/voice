use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::nn::{Lstm, LstmBuilder};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{addmm, split, stack_axis};
use mlx_rs::Array;

use mlx_macros::ModuleParameters;

/// Run a single-direction LSTM with proper hidden state propagation.
///
/// The mlx-rs built-in `Lstm::step` doesn't propagate hidden/cell state between
/// timesteps (each step uses the initial hidden state). This function fixes that
/// by manually iterating over timesteps and feeding back the hidden/cell state.
fn lstm_forward_recurrent(
    lstm: &Lstm,
    x: &Array, // (batch, seq_len, features)
) -> Result<(Array, Array), Exception> {
    // Pre-compute input projection: x @ wx.T + bias
    let x_proj = if let Some(b) = &lstm.bias.value {
        addmm(b, x, lstm.wx.value.t(), None, None)?
    } else {
        x.matmul(lstm.wx.value.t())?
    };

    let seq_len = x.dim(-2);
    let mut all_hidden = Vec::with_capacity(seq_len as usize);
    let mut all_cell = Vec::with_capacity(seq_len as usize);

    // No initial hidden/cell state — start from zeros (implicit in the math)
    let mut hidden: Option<Array> = None;
    let mut cell: Option<Array> = None;

    for idx in 0..seq_len {
        let mut ifgo = x_proj.index((.., idx, ..));

        // Add hidden state contribution if we have one from the previous step
        if let Some(ref h) = hidden {
            ifgo = addmm(&ifgo, h, lstm.wh.value.t(), None, None)?;
        }

        let pieces = split(&ifgo, 4, -1)?;
        let i = mlx_rs::ops::sigmoid(&pieces[0])?;
        let f = mlx_rs::ops::sigmoid(&pieces[1])?;
        let g = mlx_rs::ops::tanh(&pieces[2])?;
        let o = mlx_rs::ops::sigmoid(&pieces[3])?;

        let new_cell = match &cell {
            Some(c) => f.multiply(c)?.add(i.multiply(&g)?)?,
            None => i.multiply(&g)?,
        };

        let new_hidden = o.multiply(mlx_rs::ops::tanh(&new_cell)?)?;

        all_hidden.push(new_hidden.clone());
        all_cell.push(new_cell.clone());

        hidden = Some(new_hidden);
        cell = Some(new_cell);
    }

    let hidden_refs: Vec<&Array> = all_hidden.iter().collect();
    let cell_refs: Vec<&Array> = all_cell.iter().collect();

    Ok((stack_axis(&hidden_refs, -2)?, stack_axis(&cell_refs, -2)?))
}

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

        // Forward direction with proper recurrence
        let (fwd_out, _fwd_cell) = lstm_forward_recurrent(&self.forward_lstm, &x)?;

        // Backward direction: reverse input, run forward, reverse output
        let rev_indices: Vec<i32> = (0..seq_len).rev().collect();
        let rev_idx = Array::from_slice(&rev_indices, &[seq_len]);
        let x_rev = x.index((0.., &rev_idx, 0..));

        let (bwd_out_rev, _bwd_cell) = lstm_forward_recurrent(&self.backward_lstm, &x_rev)?;

        // Reverse the backward output back to the original temporal order
        let bwd_out = bwd_out_rev.index((0.., &rev_idx, 0..));

        // Concatenate forward and backward outputs along the feature dimension (axis 2)
        let output = mlx_rs::ops::concatenate_axis(&[&fwd_out, &bwd_out], 2)?;
        Ok((output, ()))
    }

    /// Set training mode on both LSTMs.
    pub fn training_mode(&mut self, _mode: bool) {
        // Lstm doesn't have dropout, nothing to toggle
    }
}
