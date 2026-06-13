# voice-stt

Speech-to-text on candle using Whisper / distil-whisper models. Uses Metal on
macOS when available and CPU on Linux or other non-macOS hosts.

## Install

```toml
[dependencies]
voice-stt = "0.2"
```

Model configs and tokenizers for known distil-whisper models are embedded in the
binary. Model weights are downloaded from HuggingFace Hub on first use and cached
locally.

## Usage

### Transcribe a file

```rust
fn main() -> voice_stt::Result<()> {
    let mut model = voice_stt::load_model("distil-whisper/distil-large-v3")?;
    let result = voice_stt::transcribe(&mut model, "audio.wav")?;
    println!("{}", result.text);
    Ok(())
}
```

### Force CPU

```rust
fn main() -> voice_stt::Result<()> {
    let mut model = voice_stt::load_model_on_device(
        "distil-whisper/distil-medium.en",
        candle_core::Device::Cpu,
    )?;
    let samples: Vec<f32> = vec![0.0; 16_000];
    let result = voice_stt::transcribe_audio(&mut model, &samples, 16_000)?;
    println!("{}", result.text);
    Ok(())
}
```

Audio at any sample rate is automatically resampled to 16kHz using a high-quality
sinc resampler, with linear interpolation as a fallback for very short inputs.

## Supported Models

| Model | Repo ID | Params | Notes |
|-------|---------|--------|-------|
| Distil Large v3 | `distil-whisper/distil-large-v3` | 756M | Multilingual, default in the CLI |
| Distil Medium English | `distil-whisper/distil-medium.en` | 394M | English-only, better for CPU experiments |

## Performance

On Apple Silicon with Metal, transcription is roughly 50x real-time for short
voice-command audio. CPU inference is supported for portability and testing, but
is substantially slower; use the medium English model for practical CPU smoke
tests.

## Architecture

Whisper uses mel-spectrogram preprocessing followed by an encoder-decoder
transformer:

- **Mel frontend**: Metal-accelerated on macOS, CPU fallback elsewhere.
- **Encoder-decoder transformer**: candle-transformers Whisper backend.
- **Greedy decode with KV cache**: encoder output is computed once, then cached
  cross-attention keys/values are reused across decoder steps.
- **Embedded metadata**: configs and tokenizers for known distil-whisper models
  are compiled into the binary.

## Features

- **High-quality resampling**: Sinc interpolation via [rubato](https://crates.io/crates/rubato).
- **WAV loading**: 16-bit integer and 32-bit float WAV files, with automatic mono mixdown.
- **HuggingFace Hub**: Automatic model weight download and caching via `hf-hub`.
- **Portable inference**: Metal by default on macOS, CPU by default elsewhere.

## Requirements

- Rust 1.85+
- macOS with Apple Silicon for fast Metal inference, or a CPU-only host for slower portable inference

## License

MIT
