# voice-nn

Neural network modules for [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) TTS, built on [mlx-rs](https://github.com/oxiglade/mlx-rs).

This crate provides the building blocks used by [`voice-tts`](https://crates.io/crates/voice-tts) — you probably want that crate unless you're doing something custom with the model architecture.

## Modules

| Module | Description |
|--------|-------------|
| `albert` | ALBERT transformer encoder for text understanding |
| `lstm` | Bidirectional LSTM with proper recurrence |
| `text_encoder` | Conv + BiLSTM text feature extraction |
| `prosody` | Duration and F0/voicing prediction |
| `vocoder` | HiFi-GAN style decoder, generator, and harmonic source |
| `conv_weighted` | Weight-normalized 1D convolution |
| `ada_norm` | Adaptive instance normalization layers |

## Usage

```toml
[dependencies]
voice-nn = "0.1"
```

```rust
use voice_nn::lstm;
use voice_nn::albert;
use voice_nn::vocoder;
```

All modules operate on `mlx_rs::Array` tensors and use `mlx_macros::ModuleParameters` for weight loading from safetensors checkpoints.

## License

MIT