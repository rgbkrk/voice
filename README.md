# voice

Rust TTS library backed by [mlx-rs](https://github.com/oxiglade/mlx-rs) (Apple MLX). Currently implements the [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) 82M-parameter TTS model with misaki-compatible G2P.

## Quick start

```bash
# Text to speech
voice --text "Hello world, this is voice speaking." --voice af_heart --play

# Save to file with different voice and speed
voice --text "The quick brown fox jumps over the lazy dog." \
  --voice am_onyx --speed 0.9 --output speech.wav

# Raw phoneme input (advanced)
voice --phonemes "həlˈO wˈɜɹld" --voice af_heart --play
```

### From source

```bash
cargo run --release -p voice-cli -- --text "Hello world" --voice af_heart --play
```

Model weights are automatically downloaded from HuggingFace Hub and cached in `~/.cache/huggingface/hub/` (shared with Python's `huggingface_hub`).

## Library usage

```rust
use std::path::Path;

fn main() -> voice_tts::Result<()> {
    // Load model and voice (cached from HuggingFace Hub)
    let mut model = voice_tts::load_model("prince-canuma/Kokoro-82M")?;
    let voice = voice_tts::load_voice("af_heart", None)?;

    // Generate from phonemes
    let audio = voice_tts::generate(&mut model, "həlˈO wˈɜɹld", &voice, 1.0)?;
    voice_tts::save_wav(&audio, Path::new("output.wav"), 24000)?;

    Ok(())
}
```

### With G2P (text to phonemes)

```rust
fn main() -> voice_tts::Result<()> {
    let mut model = voice_tts::load_model("prince-canuma/Kokoro-82M")?;
    let voice = voice_tts::load_voice("af_heart", None)?;

    // Convert text to phoneme chunks (handles the 510-char model limit)
    let chunks = voice_g2p::text_to_phoneme_chunks("Hello world, this is a test.")
        .expect("G2P failed");

    // Generate and concatenate audio for each chunk
    let mut all_samples: Vec<f32> = Vec::new();
    for phonemes in &chunks {
        let audio = voice_tts::generate(&mut model, phonemes, &voice, 1.0)?;
        all_samples.extend_from_slice(audio.as_slice());
    }

    // Write WAV
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("output.wav", spec).unwrap();
    for &s in &all_samples {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();

    Ok(())
}
```

## CLI options

```
voice [OPTIONS]

Options:
  --text <TEXT>        Plain English text to synthesize
  --phonemes <IPA>     Raw phoneme string (IPA format)
  --voice <NAME>       Voice name [default: af_heart]
  --speed <FLOAT>      Speech speed factor [default: 1.0]
  --output <PATH>      Output WAV file [default: output.wav]
  --play               Play audio after generation
  --model <REPO>       HuggingFace repo [default: prince-canuma/Kokoro-82M]
```

`--text` and `--phonemes` are mutually exclusive. `--text` uses the G2P pipeline to convert English to phonemes automatically.

## Workspace

```
crates/
  voice-tts/    Core TTS library — model, config, weights, voice loading
  voice-nn/     Neural network modules — ALBERT, BiLSTM, vocoder, prosody
  voice-dsp/    DSP primitives — STFT, iSTFT, overlap-add, windowing
  voice-g2p/    Grapheme-to-phoneme — misaki dictionary + espeak-ng fallback
  voice-cli/    CLI binary (installs as `voice`)
```

## G2P pipeline

The `voice-g2p` crate ports [misaki](https://github.com/hexgrad/misaki)'s English G2P, which Kokoro was trained on:

- **Dictionary lookup**: 90k gold + 93k silver pronunciation entries embedded at compile time
- **Morphological decomposition**: -s, -ed, -ing suffix rules with voicing logic
- **Number handling**: cardinals, ordinals, years, currency
- **POS tagging**: optional spaCy subprocess (via `uv run`) for context-dependent pronunciation
- **Fallback**: espeak-ng per-word for unknown words

## Available voices

American: `af_heart`, `af_alloy`, `af_aoede`, `af_bella`, `af_kore`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_liam`, `am_michael`, `am_onyx`

British: `bf_alice`, `bf_emma`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

Other: `ff_siwis` (French), `if_sara` / `im_nicola` (Italian), `jf_alpha` / `jf_gongitsune` (Japanese), `zf_xiaobei` / `zf_xiaoni` / `zf_xiaoxiao` (Chinese)

## Requirements

- macOS with Apple Silicon (MLX requirement)
- Rust 1.85+
- Xcode command line tools (for MLX Metal compilation)
- espeak-ng (optional, for G2P fallback on unknown words)
