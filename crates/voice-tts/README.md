# voice-tts

Core TTS library for [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) model inference on Apple Silicon via [mlx-rs](https://github.com/oxiglade/mlx-rs).

## Install

```toml
[dependencies]
voice-tts = "0.1"
```

## Usage

```rust
use std::path::Path;

fn main() -> voice_tts::Result<()> {
    // Load model and voice (downloaded from HuggingFace Hub, cached locally)
    let mut model = voice_tts::load_model("prince-canuma/Kokoro-82M")?;
    let voice = voice_tts::load_voice("af_heart", None)?;

    // Generate audio from phonemes
    let audio = voice_tts::generate(&mut model, "həlˈO wˈɜɹld", &voice, 1.0)?;

    // Save to WAV
    voice_tts::save_wav(&audio, Path::new("output.wav"), 24000)?;

    Ok(())
}
```

Pair with [`voice-g2p`](https://crates.io/crates/voice-g2p) to go from English text to phonemes:

```rust
let chunks = voice_g2p::text_to_phoneme_chunks("Hello world, this is a test.")?;

let mut all_samples: Vec<f32> = Vec::new();
for phonemes in &chunks {
    let audio = voice_tts::generate(&mut model, phonemes, &voice, 1.0)?;
    all_samples.extend_from_slice(audio.as_slice());
}
```

## What's inside

- **Model loading** — downloads and caches Kokoro weights from HuggingFace Hub
- **Voice loading** — loads voice embeddings by name (e.g. `af_heart`, `am_adam`)
- **Inference** — full TTS forward pass: text encoding → duration/pitch prediction → vocoder
- **WAV output** — write generated audio to disk

## Requirements

- macOS with Apple Silicon (MLX requirement)
- Xcode command line tools (for Metal compilation)

## License

MIT