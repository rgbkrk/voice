use mlx_rs::error::Exception;
use mlx_rs::module::Param;
use mlx_rs::Array;
use mlx_macros::ModuleParameters;

/// Which convolution operation to use.
#[derive(Debug, Clone, Copy)]
pub enum ConvOp {
    Conv1d,
    ConvTranspose1d,
}

/// A weight-normalized 1-D convolution layer.
///
/// Stores separate `weight_g` (gain) and `weight_v` (direction) parameters.
/// The effective weight is computed as: `w = g * v / ||v||` (weight normalization).
#[derive(Debug, ModuleParameters)]
pub struct ConvWeighted {
    #[param]
    pub weight_g: Param<Array>,
    #[param]
    pub weight_v: Param<Array>,
    #[param]
    pub bias: Param<Option<Array>>,

    pub stride: i32,
    pub padding: i32,
    pub dilation: i32,
    pub groups: i32,
    pub encode: bool,
}

impl ConvWeighted {
    /// Create a new weight-normalized Conv1d with all parameters specified.
    ///
    /// - `in_channels`: number of input channels
    /// - `out_channels`: number of output channels
    /// - `kernel_size`: size of the convolution kernel
    /// - `stride`, `padding`, `dilation`, `groups`: convolution parameters
    /// - `bias`: whether to include a bias term
    /// - `encode`: if true, bias dimension is `in_channels` instead of `out_channels`
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        bias: bool,
        encode: bool,
    ) -> Result<Self, Exception> {
        let weight_g = Array::ones::<f32>(&[out_channels, 1, 1])?;
        let weight_v = Array::ones::<f32>(&[out_channels, kernel_size, in_channels])?;
        let bias_arr = if bias {
            let bias_dim = if encode { in_channels } else { out_channels };
            Some(Array::zeros::<f32>(&[bias_dim])?)
        } else {
            None
        };

        Ok(Self {
            weight_g: Param::new(weight_g),
            weight_v: Param::new(weight_v),
            bias: Param::new(bias_arr),
            stride,
            padding,
            dilation,
            groups,
            encode,
        })
    }

    /// Convenience constructor with only (in_ch, out_ch, kernel_size, padding).
    /// Defaults: stride=1, dilation=1, groups=1, bias=true, encode=false.
    pub fn new_simple(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        padding: i32,
    ) -> Result<Self, Exception> {
        Self::new(in_channels, out_channels, kernel_size, 1, padding, 1, 1, true, false)
    }

    /// Compute the weight-normalized weight: `g * v / ||v||`.
    fn normalized_weight(&self) -> Result<Array, Exception> {
        let v = &*self.weight_v;
        let g = &*self.weight_g;

        // L2 norm of v along dims [1, 2], keeping dims for broadcast
        let v_norm = mlx_rs::linalg::norm_l2(v, &[1, 2], true)?;

        // Avoid division by zero
        let eps = Array::from_slice(&[1e-12f32], &[1]);
        let v_norm = mlx_rs::ops::maximum(&v_norm, &eps)?;

        let weight = g * v / v_norm;
        Ok(weight)
    }

    /// Resolve the effective weight, optionally transposing for shape compatibility.
    fn resolve_weight(&self, x: &Array) -> Result<Array, Exception> {
        let weight = self.normalized_weight()?;
        let x_last = x.shape()[x.ndim() - 1];
        let w_last = weight.shape()[weight.ndim() - 1];

        if x_last == w_last || self.groups > 1 {
            Ok(weight)
        } else {
            weight.transpose_axes(&[0, 2, 1])
        }
    }

    /// Add bias to the result if present.
    fn add_bias(&self, result: Array) -> Result<Array, Exception> {
        if let Some(b) = &*self.bias {
            Ok(result + b)
        } else {
            Ok(result)
        }
    }

    /// Run as a standard Conv1d (no transposition).
    ///
    /// Input: `(batch, length, channels)` -- MLX channel-last layout.
    pub fn forward_conv1d(&self, x: &Array) -> Result<Array, Exception> {
        let weight = self.resolve_weight(x)?;
        let result = mlx_rs::ops::conv1d(
            x, &weight, self.stride, self.padding, self.dilation, self.groups,
        )?;
        self.add_bias(result)
    }

    /// Run as a transposed Conv1d.
    ///
    /// Input: `(batch, length, channels)` -- MLX channel-last layout.
    /// Weight stored as (out_ch, kernel, in_ch). For conv_transpose1d,
    /// MLX expects weight (out_ch, kernel, in_ch) where in_ch matches
    /// the input's last dimension. Use .T if the input channels don't
    /// match weight's last dim (same logic as regular conv1d, but
    /// the transpose is full-reverse to match Python's .T behavior).
    pub fn forward_conv_transpose1d(&self, x: &Array) -> Result<Array, Exception> {
        let weight = self.normalized_weight()?;
        let x_last = x.shape()[x.ndim() - 1];
        let w_last = weight.shape()[weight.ndim() - 1];

        // Same logic as Python: if x.shape[-1] == weight.shape[-1], use as-is
        // Otherwise transpose weight via .T (full reverse: [2, 1, 0])
        let weight = if x_last == w_last || self.groups > 1 {
            weight
        } else {
            weight.transpose_axes(&[2, 1, 0])?
        };

        let result = mlx_rs::ops::conv_transpose1d(
            x, &weight, self.stride, self.padding, self.dilation, None, self.groups,
        )?;
        self.add_bias(result)
    }

    /// Run the weight-normalized convolution with an explicit operation selector.
    pub fn forward_with_op(&self, x: &Array, conv_op: ConvOp) -> Result<Array, Exception> {
        match conv_op {
            ConvOp::Conv1d => self.forward_conv1d(x),
            ConvOp::ConvTranspose1d => self.forward_conv_transpose1d(x),
        }
    }

    /// Run as Conv1d (default forward for modules that just need `forward(&x)`).
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        self.forward_conv1d(x)
    }
}
