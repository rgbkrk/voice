use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{Linear, LinearBuilder};
use mlx_rs::Array;

// ---------------------------------------------------------------------------
// InstanceNorm1d
// ---------------------------------------------------------------------------

/// Instance normalization over the last (time) dimension, per-channel.
///
/// Given input of shape `(batch, channels, time)`, normalizes each `(batch, channel)`
/// slice to zero mean and unit variance.
#[derive(Debug, ModuleParameters)]
pub struct InstanceNorm1d {
    pub num_features: i32,
    pub eps: f32,
}

impl InstanceNorm1d {
    pub fn new(num_features: i32, eps: Option<f32>) -> Self {
        Self {
            num_features,
            eps: eps.unwrap_or(1e-5),
        }
    }

    /// Normalize input of shape `(batch, channels, time)`.
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // Mean and variance over the time axis (axis=2), keepdims
        let mean = x.mean_axes(&[2], true)?;
        let var = x.var_axes(&[2], true, 0)?;

        let eps = Array::from_slice(&[self.eps], &[1]);
        let std = (var + eps).sqrt()?;

        Ok((x - mean) / std)
    }
}

// ---------------------------------------------------------------------------
// AdaIN1d
// ---------------------------------------------------------------------------

/// Adaptive Instance Normalization for 1-D signals.
///
/// Applies instance normalization, then modulates with a style vector projected
/// to produce per-channel scale (gamma) and shift (beta).
///
/// Input: `(x, s)` where `x` is `(batch, channels, time)` and `s` is `(batch, style_dim)`.
/// Output: `gamma * instance_norm(x) + beta`.
#[derive(Debug, ModuleParameters)]
pub struct AdaIN1d {
    pub norm: InstanceNorm1d,
    #[param]
    pub fc: Linear,
}

impl AdaIN1d {
    /// Create a new AdaIN1d layer.
    ///
    /// - `style_dim`: dimension of the style/conditioning vector
    /// - `num_features`: number of channels in the input
    pub fn new(style_dim: i32, num_features: i32) -> Result<Self, Exception> {
        let fc = LinearBuilder::new(style_dim, num_features * 2).build()?;
        Ok(Self {
            norm: InstanceNorm1d::new(num_features, None),
            fc,
        })
    }

    /// Apply adaptive instance normalization.
    ///
    /// - `x`: input tensor of shape `(batch, channels, time)`
    /// - `s`: style tensor of shape `(batch, style_dim)`
    pub fn forward_xy(&mut self, x: &Array, s: &Array) -> Result<Array, Exception> {
        // Project style to (batch, 2 * num_features)
        let h = self.fc.forward(s)?;

        let num_features = self.norm.num_features;
        let parts = mlx_rs::ops::split(&h, 2, -1)?;
        let gamma = &parts[0]; // (batch, num_features)
        let beta = &parts[1];

        // Reshape for broadcasting: (batch, num_features, 1)
        let batch = x.shape()[0];
        let gamma = gamma.reshape(&[batch, num_features, 1])?;
        let beta = beta.reshape(&[batch, num_features, 1])?;

        let normed = self.norm.forward(x)?;
        let one = Array::from_f32(1.0);
        Ok((&one + gamma) * normed + beta)
    }
}

impl Module<(&Array, &Array)> for AdaIN1d {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, (x, s): (&Array, &Array)) -> Result<Array, Exception> {
        self.forward_xy(x, s)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

// ---------------------------------------------------------------------------
// AdaLayerNorm
// ---------------------------------------------------------------------------

/// Adaptive Layer Normalization.
///
/// Normalizes the input using mean/variance over the last dimension,
/// then applies a style-conditioned affine transform (gamma, beta).
///
/// Input: `(x, s)` where `x` is `(batch, seq_len, dim)` and `s` is `(batch, style_dim)`.
#[derive(Debug, ModuleParameters)]
pub struct AdaLayerNorm {
    pub eps: f32,
    pub d_model: i32,
    #[param]
    pub fc: Linear,
}

impl AdaLayerNorm {
    /// Create a new AdaLayerNorm.
    ///
    /// - `style_dim`: dimension of the style/conditioning vector
    /// - `d_model`: last dimension of the input features
    /// - `eps`: epsilon for normalization stability (default 1e-5)
    pub fn new(style_dim: i32, d_model: i32) -> Result<Self, Exception> {
        Self::with_eps(style_dim, d_model, 1e-5)
    }

    pub fn with_eps(style_dim: i32, d_model: i32, eps: f32) -> Result<Self, Exception> {
        let fc = LinearBuilder::new(style_dim, d_model * 2).build()?;
        Ok(Self { eps, d_model, fc })
    }

    /// Apply adaptive layer normalization.
    ///
    /// - `x`: input tensor, shape `(batch, seq_len, d_model)` or `(batch, d_model)`
    /// - `s`: style tensor, shape `(batch, style_dim)`
    pub fn forward_xy(&mut self, x: &Array, s: &Array) -> Result<Array, Exception> {
        let last_axis = (x.ndim() as i32) - 1;
        let mean = x.mean_axes(&[last_axis], true)?;
        let var = x.var_axes(&[last_axis], true, 0)?;
        let eps = Array::from_slice(&[self.eps], &[1]);
        let normed = (x - &mean) / (var + eps).sqrt()?;

        let h = self.fc.forward(s)?;
        let d_model = self.d_model;
        let parts = mlx_rs::ops::split(&h, 2, -1)?;
        let gamma = &parts[0];
        let beta = &parts[1];

        let one = Array::from_f32(1.0);
        if x.ndim() == 3 {
            let batch = x.shape()[0];
            let gamma = gamma.reshape(&[batch, 1, d_model])?;
            let beta = beta.reshape(&[batch, 1, d_model])?;
            Ok((&one + gamma) * normed + beta)
        } else {
            Ok((&one + gamma) * normed + beta)
        }
    }
}

impl Module<(&Array, &Array)> for AdaLayerNorm {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, (x, s): (&Array, &Array)) -> Result<Array, Exception> {
        self.forward_xy(x, s)
    }

    fn training_mode(&mut self, _mode: bool) {}
}
