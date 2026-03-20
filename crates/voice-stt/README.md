# voice-stt

Speech-to-text on Apple Silicon, powered by [MLX](https://github.com/oxiglade/mlx-rs). Ships with [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny) support — a compact encoder-decoder transformer designed for on-device speech recognition.

## Install

```toml
[dependencies]
voice-stt = "0.1"
```

> **Note:** Requires macOS with Apple Silicon (MLX requirement). Model weights (~108-246MB) are downloaded from HuggingFace Hub on first run and cached locally.

## Usage

### Transcribe a file

```rust
fn main() -> voice_stt::Result<()> {
    let mut model = voice_stt::load_model("UsefulSensors/moonshine-tiny")?;
    let result = voice_stt::transcribe(&mut model, "audio.wav")?;
    println!("{}", result.text);
    Ok(())
}
```

### Transcribe raw audio samples

```rust
fn main() -> voice_stt::Result<()> {
    let mut model = voice_stt::load_model("UsefulSensors/moonshine-base")?;
    let tokenizer = voice_stt::load_tokenizer("UsefulSensors/moonshine-base")?;

    let samples: Vec<f32> = /* 16kHz mono f32 samples */;
    let result = voice_stt::transcribe_audio_with_tokenizer(
        &mut model, &samples, 16000, &tokenizer
    )?;
    println!("{}", result.text);
    Ok(())
}
```

Audio at any sample rate is automatically resampled to 16kHz using a high-quality sinc resampler.

## Supported Models

| Model | Repo ID | Params | Size | Notes |
|-------|---------|--------|------|-------|
| Moonshine Tiny | `UsefulSensors/moonshine-tiny` | 27M | 108MB | Fast, embedded config + tokenizer |
| Moonshine Base | `UsefulSensors/moonshine-base` | 61M | 246MB | Better accuracy for real mic audio |

## Performance

On Apple Silicon (cached model):

| Audio Length | Transcription Time | RTF |
|-------------|-------------------|-----|
| 3s | ~60ms | 0.02× (50× real-time) |
| 9s | ~100ms | 0.01× (89× real-time) |
| 17s | ~200ms | 0.01× |

## Architecture

Moonshine uses a **learned audio frontend** instead of mel spectrograms — raw 16kHz audio goes directly into the encoder:

- **Encoder**: Three Conv1d layers (384× total downsampling) → transformer layers with partial RoPE
- **Decoder**: Token embedding → transformer layers with causal self-attention + cross-attention + SwiGLU MLP
- **Output**: Tied embedding projection → greedy autoregressive decoding with KV cache

The Moonshine-tiny config and tokenizer are embedded in the binary for instant startup. Model weights are downloaded from HuggingFace on first use.

## Features

- **High-quality resampling**: Sinc interpolation via [rubato](https://crates.io/crates/rubato) — accepts audio at any sample rate
- **WAV loading**: 16-bit integer and 32-bit float WAV files, with automatic mono mixdown
- **Embedded tokenizer**: SentencePiece BPE tokenizer compiled into the binary (no filesystem access needed)
- **HuggingFace Hub**: Automatic model download and caching via [hf-hub](https://crates.io/crates/hf-hub)
- **Weight sanitization**: Loads HuggingFace safetensors directly, handles PyTorch→MLX conv weight transposition

## Requirements

- macOS with Apple Silicon
- Rust 1.85+
- Xcode command line tools (for MLX Metal compilation)

## License

MIT