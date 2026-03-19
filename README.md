# voice

Rust TTS on Apple Silicon, powered by [MLX](https://github.com/oxiglade/mlx-rs). Ships the [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) 82M-parameter model with a full English G2P pipeline.

Faster time-to-first-speech than macOS `say`, with dramatically better audio quality.

## Install

Build from source (requires Git LFS for embedded voice/model data):

```bash
# Install git-lfs if you don't have it
brew install git-lfs
git lfs install

# Clone and build
git clone https://github.com/rgbkrk/voicers.git
cd voicers
cargo install --path crates/voice-cli
```

> **Why not `cargo install voice`?** The Metal shader library path is baked in at compile time by mlx-sys and points to a temp directory that gets cleaned up after install. This is an [upstream mlx-rs issue](https://github.com/oxiglade/mlx-rs/issues/327). Building from source avoids this entirely.

> **Why git-lfs?** Voice data (`.safetensors`) and tagger weights are stored with Git LFS. Without it, those files are tiny pointers instead of actual data — the build will catch this and tell you what to do.

This puts the `voice` binary on your `$PATH`. Model weights (~312MB) are downloaded from HuggingFace Hub on first run and cached in `~/.cache/huggingface/hub/`. Seven popular voices and the model config are embedded in the binary — no network needed for common use.

## Usage

```bash
# Just talk
voice Hello world, this is voice speaking.

# Pick a voice
voice -v am_michael "How are you today?"

# Pipe text in
echo "The quick brown fox jumps over the lazy dog." | voice

# Save to file instead of playing
voice -o speech.wav "Good morning everyone."

# Read from a file
voice -f script.txt

# Adjust speed
voice -s 0.8 "Take it slow."

# Strip markdown before speaking
voice --markdown -f blog-post.mdx

# Raw phoneme input
voice --phonemes "həlˈO wˈɜɹld"
```

## CLI options

```
Usage: voice [OPTIONS] [TEXT]...

Arguments:
  [TEXT]...                       Text to speak

Options:
  -f, --input-file <FILE>        Read text from a file (use - for stdin)
      --phonemes <IPA>           Raw phoneme string (IPA)
  -v, --voice <VOICE>            Voice name [default: af_heart]
  -o, --output <PATH>            Write WAV to file instead of playing
  -s, --speed <SPEED>            Speech speed factor [default: 1.0]
      --markdown                 Strip markdown/MDX formatting before speaking
      --sub <WORD=REPLACEMENT>   Word substitution (repeatable)
      --sub-file <PATH>          Load substitutions from a file
  -q, --quiet                    Suppress progress output (errors still print)
  -h, --help                     Print help
```

If no text, `--phonemes`, or `-f` is given, reads from stdin.

## LLM-friendly design

`voice` is built to work well with AI agents and coding assistants:

- **Phoneme output**: The CLI emits phoneme chunks to stderr, so agents can see the IPA representation of what's being spoken
- **Phoneme input**: `--phonemes` accepts raw IPA strings, giving agents precise control over pronunciation without going through G2P
- **Stdin pipe**: `echo "text" | voice` lets agents speak from any script or tool
- **Markdown stripping**: `--markdown` cleans up LLM-generated markdown before speaking
- **Word substitutions**: `--sub` and `.voice-subs` files let you fix pronunciation of project-specific terms (package names, acronyms, etc.)

See [SKILL.md](SKILL.md) for a lightweight reference card that AI agents can use to learn the `voice` tool.

## Word substitutions

Fix pronunciation of names, acronyms, or technical terms.

### Inline

```bash
voice --sub nteract=enteract --sub PyTorch=pie-torch "nteract uses PyTorch"
```

### `.voice-subs` file

Create a `.voice-subs` file in your project root. `voice` auto-discovers it by walking up from the working directory.

```bash
# .voice-subs — one WORD=REPLACEMENT per line
nteract=enteract
PyTorch=pie-torch
MLX=M L X
kubectl=cube-cuddle

# Wrap in /slashes/ for phoneme overrides (bypass G2P entirely)
Kokoro=/kˈOkəɹO/
```

Text substitutions are applied before G2P. Phoneme overrides (the `/slash/` syntax) are injected directly into the phoneme stream.

```bash
# Uses .voice-subs automatically
voice --markdown -f README.md

# Or specify a file explicitly
voice --sub-file my-project.subs -f notes.txt
```

## Builtin voices

These voices are embedded in the binary and load instantly (no network):

| Voice | Description |
|-------|-------------|
| `af_heart` | American female — warm, natural (default) |
| `af_bella` | American female — expressive |
| `af_sarah` | American female — clear, professional |
| `af_sky` | American female — bright |
| `am_michael` | American male — clear |
| `am_adam` | American male — deeper |
| `bf_emma` | British female — natural |

All other voices are fetched from HuggingFace Hub on first use:

**American**: `af_alloy`, `af_aoede`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_onyx`, `am_puck`

**British**: `bf_alice`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

**Other languages**: French (`ff_siwis`), Hindi (`hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`), Italian (`if_sara`, `im_nicola`), Japanese (`jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`), Portuguese (`pf_dora`, `pm_alex`, `pm_santa`), Spanish (`ef_dora`, `em_alex`, `em_santa`), Chinese (`zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`)

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
| [`voice`](https://crates.io/crates/voice) | CLI binary — installs as `voice` |
| [`voice-tts`](https://crates.io/crates/voice-tts) | Core TTS library — model loading, inference, WAV output |
| [`voice-nn`](https://crates.io/crates/voice-nn) | Neural network modules — ALBERT, BiLSTM, vocoder, prosody |
| [`voice-dsp`](https://crates.io/crates/voice-dsp) | DSP primitives — STFT, iSTFT, overlap-add, windowing |
| [`voice-g2p`](https://crates.io/crates/voice-g2p) | Grapheme-to-phoneme — misaki dictionary + espeak-ng fallback |

## Architecture

### G2P pipeline

The `voice-g2p` crate ports [misaki](https://github.com/hexgrad/misaki)'s English G2P:

- **POS tagging**: Embedded averaged perceptron tagger (no Python/spaCy dependency)
- **Dictionary lookup**: 90k gold + 93k silver pronunciation entries embedded at compile time
- **Morphological decomposition**: -s, -ed, -ing suffix rules with voicing logic
- **Number handling**: Cardinals, ordinals, years, currency
- **Fallback**: espeak-ng subprocess for unknown words

### Startup pipeline

Model loading runs in a background thread while text resolution, G2P, and voice loading happen on the main thread. Audio chunks stream to the speakers as they're generated — the first chunk plays immediately while subsequent chunks are still being synthesized.

## Requirements

- macOS with Apple Silicon (MLX requirement)
- Rust 1.85+
- Git LFS (`brew install git-lfs && git lfs install`)
- Xcode command line tools (for MLX Metal compilation)
- Xcode license accepted: `sudo xcodebuild -license`
- Metal Toolchain (Xcode 17+): `xcodebuild -downloadComponent MetalToolchain`
- espeak-ng (optional, for G2P fallback on unknown words): `brew install espeak-ng`

> **Fresh Mac?** If the build fails with linker errors mentioning "You have not
> agreed to the Xcode license agreements", run `sudo xcodebuild -license`.
> If it fails with "cannot execute tool 'metal' due to missing Metal Toolchain",
> run `xcodebuild -downloadComponent MetalToolchain`. Then retry the build.

## License

MIT