# voice

Rust TTS on Apple Silicon, powered by [MLX](https://github.com/oxiglade/mlx-rs). Ships the [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) 82M-parameter model with a full English G2P pipeline.

## Install

```bash
cargo install voice-cli
```

This puts the `voice` binary on your `$PATH`. Model weights are downloaded from HuggingFace Hub on first run and cached in `~/.cache/huggingface/hub/`.

## Usage

```bash
# Just talk
voice Hello world, this is voice speaking.

# Pick a voice
voice -v am_adam "How are you today?"

# Pipe text in
echo "The quick brown fox jumps over the lazy dog." | voice

# Save to file instead of playing
voice -o speech.wav "Good morning everyone."

# Read from a file
voice -f script.txt

# Adjust speed
voice -s 0.8 "Take it slow."

# Raw phoneme input (advanced)
voice --phonemes "həlˈO wˈɜɹld"
```

### From source

```bash
cargo install --path crates/voice-cli
```

## CLI options

```
Usage: voice [OPTIONS] [TEXT]...

Arguments:
  [TEXT]...  Text to speak

Options:
  -f, --input-file <FILE>   Read text from a file (use - for stdin)
      --phonemes <IPA>      Raw phoneme string (IPA)
  -v, --voice <VOICE>       Voice name [default: af_heart]
  -o, --output <PATH>       Write WAV to file instead of playing
  -s, --speed <SPEED>       Speech speed factor [default: 1.0]
  -h, --help                Print help
```

If no text, `--phonemes`, or `-f` is given, reads from stdin.

## Library usage

Add the crates you need:

```toml
[dependencies]
voice-tts = "0.1"
voice-g2p = "0.1"  # if you need text-to-phoneme conversion
```

### Generate from phonemes

```rust
use std::path::Path;

fn main() -> voice_tts::Result<()> {
    let mut model = voice_tts::load_model("prince-canuma/Kokoro-82M")?;
    let voice = voice_tts::load_voice("af_heart", None)?;

    let audio = voice_tts::generate(&mut model, "həlˈO wˈɜɹld", &voice, 1.0)?;
    voice_tts::save_wav(&audio, Path::new("output.wav"), 24000)?;

    Ok(())
}
```

### With G2P (text → phonemes)

```rust
fn main() -> voice_tts::Result<()> {
    let mut model = voice_tts::load_model("prince-canuma/Kokoro-82M")?;
    let voice = voice_tts::load_voice("af_heart", None)?;

    // Convert text to phoneme chunks (handles the 510-token model limit)
    let chunks = voice_g2p::text_to_phoneme_chunks("Hello world, this is a test.")
        .expect("G2P failed");

    let mut all_samples: Vec<f32> = Vec::new();
    for phonemes in &chunks {
        let audio = voice_tts::generate(&mut model, phonemes, &voice, 1.0)?;
        all_samples.extend_from_slice(audio.as_slice());
    }

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

## Crates

| Crate | Description |
|-------|-------------|
| [`voice-cli`](https://crates.io/crates/voice-cli) | CLI binary — installs as `voice` |
| [`voice-tts`](https://crates.io/crates/voice-tts) | Core TTS library — model loading, inference, WAV output |
| [`voice-nn`](https://crates.io/crates/voice-nn) | Neural network modules — ALBERT, BiLSTM, vocoder, prosody |
| [`voice-dsp`](https://crates.io/crates/voice-dsp) | DSP primitives — STFT, iSTFT, overlap-add, windowing |
| [`voice-g2p`](https://crates.io/crates/voice-g2p) | Grapheme-to-phoneme — misaki dictionary + espeak-ng fallback |

## G2P pipeline

The `voice-g2p` crate ports [misaki](https://github.com/hexgrad/misaki)'s English G2P, which Kokoro was trained on:

- **Dictionary lookup**: 90k gold + 93k silver pronunciation entries embedded at compile time
- **Morphological decomposition**: -s, -ed, -ing suffix rules with voicing logic
- **Number handling**: cardinals, ordinals, years, currency
- **POS tagging**: optional spaCy subprocess (via `uv run`) for context-dependent pronunciation
- **Fallback**: espeak-ng per-word for unknown words

## Available voices

**American**: `af_heart`, `af_alloy`, `af_aoede`, `af_bella`, `af_kore`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_liam`, `am_michael`, `am_onyx`

**British**: `bf_alice`, `bf_emma`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

**Other**: `ff_siwis` (French), `if_sara` / `im_nicola` (Italian), `jf_alpha` / `jf_gongitsune` (Japanese), `zf_xiaobei` / `zf_xiaoni` / `zf_xiaoxiao` (Chinese)

## Requirements

- macOS with Apple Silicon (MLX requirement)
- Rust 1.85+
- Xcode command line tools (for MLX Metal compilation)
- espeak-ng (optional, for G2P fallback on unknown words)

## License

MIT