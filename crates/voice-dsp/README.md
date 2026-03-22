# voice-dsp

DSP primitives for the [voice](https://github.com/rgbkrk/voicers) TTS pipeline, built on [mlx-rs](https://github.com/oxiglade/mlx-rs) (Apple MLX).

## Install

```toml
[dependencies]
voice-dsp = "0.1"
```

## What's inside

- **STFT / iSTFT** — Short-Time Fourier Transform and its inverse, matching PyTorch conventions
- **`MlxStft`** — batched STFT wrapper used by the vocoder pipeline (`transform` → magnitude + phase, `inverse` → audio)
- **Windowing** — Hanning window generation
- **Interpolation** — 1-D nearest/linear interpolation for upsampling tensors
- **Phase utilities** — `mlx_angle` (complex argument) and `mlx_unwrap` (phase unwrapping)

## Usage

```rust
use voice_dsp::{stft, istft, hanning, MlxStft};

// Batched STFT for the vocoder
let stft = MlxStft::new(1024, 256, 1024)?;
let (magnitude, phase) = stft.transform(&audio_batch)?;
let reconstructed = stft.inverse(&magnitude, &phase)?;
```

All functions operate on `quill_mlx::Array` and return `Result<_, quill_mlx::error::Exception>`.

## Requirements

- macOS with Apple Silicon (MLX requirement)

## License

MIT