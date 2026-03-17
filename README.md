# voicers

Rust TTS library backed by [mlx-rs](https://github.com/oxiglade/mlx-rs) (Apple MLX). Currently implements the [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) 82M-parameter TTS model.

## Quick start

```bash
# Generate audio from phonemes
cargo run --release -p voicers-cli -- \
  --phonemes "h ɛ l oʊ w ɜː l d" \
  --voice af_heart \
  --output hello.wav \
  --play

# Different voice and speed
cargo run --release -p voicers-cli -- \
  --phonemes "ð ɪ s ɪ z ɐ t ɛ s t" \
  --voice am_onyx \
  --speed 0.8 \
  --output test.wav
```

Model weights are automatically downloaded from HuggingFace Hub and cached in `~/.cache/huggingface/hub/` (shared with Python's `huggingface_hub`).

## Library usage

```rust
use std::path::Path;

fn main() -> voicers::Result<()> {
    let mut model = voicers::load_model("prince-canuma/Kokoro-82M")?;
    let voice = voicers::load_voice("af_heart", None)?;
    let audio = voicers::generate(&mut model, "h ɛ l oʊ", &voice, 1.0)?;
    voicers::save_wav(&audio, Path::new("output.wav"), 24000)?;
    Ok(())
}
```

## Workspace structure

```
crates/
  voicers/        Core TTS library (no audio playback deps)
  voicers-cli/    CLI with --play support via rodio
  voicers-g2p/    Grapheme-to-phoneme (stub, future espeak-ng)
```

## Available voices

Voices from the Kokoro model repo: `af_heart`, `af_bella`, `af_nova`, `af_sky`, `af_alloy`, `af_aoede`, `af_kore`, `af_river`, `af_sarah`, `am_onyx`, `bf_alice`, `bf_emma`, `bm_daniel`, `bm_george`, `ff_siwis`, `if_sara`, `im_nicola`, `jf_alpha`, `jf_gongitsune`, `zf_xiaobei`.

## Phoneme input

Kokoro expects IPA phoneme strings. Each character maps to a token via the model's vocab. Common English phonemes:

| Sound | IPA | Example |
|-------|-----|---------|
| h | h | **h**ello |
| e as in bed | ɛ | h**e**llo |
| l | l | he**ll**o |
| long o | oʊ | hell**o** |
| w | w | **w**orld |
| er (rhotic) | ɜː | w**or**ld |
| d | d | worl**d** |
| th (voiced) | ð | **th**is |
| s | s | **s**ee |
| t | t | **t**est |

A G2P (grapheme-to-phoneme) crate is planned to convert plain text to phonemes automatically.

## Requirements

- macOS with Apple Silicon (MLX requirement)
- Rust 1.85+
- Xcode command line tools (for MLX compilation)

## Architecture

The Kokoro model pipeline:

```
Phonemes -> ALBERT encoder (6-layer transformer)
         -> Prosody predictor (duration, F0, voicing)
         -> Text encoder (Conv1d + BiLSTM)
         -> Duration alignment (expand phonemes to frames)
         -> Decoder (style-conditioned residual blocks)
         -> Generator (upsampling + iSTFT vocoder)
         -> 24kHz audio waveform
```

All neural network layers use mlx-rs with Metal GPU acceleration on Apple Silicon.
