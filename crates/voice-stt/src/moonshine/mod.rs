//! Moonshine speech-to-text model.
//!
//! A compact encoder-decoder transformer designed for on-device speech
//! recognition by [Useful Sensors](https://usefulsensors.com/).
//!
//! Two official variants are available on HuggingFace:
//!
//! | Variant | Params | Hidden | Layers (enc/dec) |
//! |---------|--------|--------|------------------|
//! | [`moonshine-tiny`](https://huggingface.co/UsefulSensors/moonshine-tiny) | 27M | 288 | 6 / 6 |
//! | [`moonshine-base`](https://huggingface.co/UsefulSensors/moonshine-base) | 61M | 416 | 8 / 8 |
//!
//! # Architecture
//!
//! - **Learned audio frontend**: Three Conv1d layers with progressive stride
//!   (64×3×2 = 384× total downsampling) replace mel spectrograms. Raw 16kHz
//!   audio goes directly into the encoder.
//! - **Partial RoPE**: Only a fraction of head dimensions receive rotary
//!   positional encoding; the rest pass through unmodified.
//! - **SwiGLU decoder MLP**: Gated SiLU activation for better gradient flow.
//! - **Tied embeddings**: Output projection reuses the token embedding matrix.

pub mod config;
pub mod model;

pub use config::MoonshineConfig;
pub use model::{DecoderLayerCache, MoonshineModel};
