# voice

Rust TTS & STT on Apple Silicon, powered by [candle](https://github.com/huggingface/candle) with Metal GPU acceleration. Ships the [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) 82M-parameter TTS model with a full English G2P pipeline, and [Whisper](https://huggingface.co/distil-whisper/distil-large-v3) for speech-to-text.

Faster time-to-first-speech than macOS `say`, with dramatically better audio quality. STT runs at ~50× real-time on Apple Silicon.

## Install

### Pre-built binary (recommended)

Install with [cargo-binstall](https://github.com/cargo-bins/cargo-binstall) to get a pre-built binary — no compilation required:

```bash
# Install cargo-binstall if you don't have it
cargo install cargo-binstall

# Install voice
cargo binstall voice
```

### Build from source

Requires Git LFS for embedded voice/model data:

```bash
# Install git-lfs if you don't have it
brew install git-lfs
git lfs install

# Clone and build
git clone https://github.com/rgbkrk/voice.git
cd voice
cargo install --path crates/voice-cli
```

> **Why git-lfs?** Voice data (`.safetensors`) and tagger weights are stored with Git LFS. Without it, those files are tiny pointers instead of actual data — the build will catch this and tell you what to do.

This puts the `voice` binary on your `$PATH`. Model weights are downloaded from HuggingFace Hub on first run and cached in `~/.cache/huggingface/hub/`. Seven popular voices and the model config are embedded in the binary — no network needed for common use.

## Usage

```bash
# Just talk (backward compatible — no subcommand needed)
voice Hello world, this is voice speaking.

# Explicit say subcommand with options
voice say -v am_michael "How are you today?"

# Pipe text in
echo "The quick brown fox jumps over the lazy dog." | voice say

# Save to file instead of playing
voice say -o speech.wav "Good morning everyone."

# Read from a file, strip markdown
voice say --markdown -f blog-post.mdx

# Adjust speed
voice say -s 0.8 "Take it slow."

# Raw phoneme input
voice say --phonemes "həlˈO wˈɜɹld"

# Listen — single-shot speech-to-text from mic
voice listen

# Continuous listening — segments split on silence
voice listen --continuous

# Transcribe an audio file
voice transcribe recording.wav

# JSON-RPC server for agent integration
voice serve
```

## CLI options

### `voice say`

```
Speak text aloud (default when no subcommand given)

Usage: voice say [OPTIONS] [TEXT]...

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
  -q, --quiet                    Suppress progress output
  -h, --help                     Print help
```

### `voice listen`

```
Record from microphone and transcribe (speech-to-text)

Usage: voice listen [OPTIONS]

Options:
      --continuous   Continuous mode — record and transcribe segments
                     as you speak. Segments split on silence.
  -q, --quiet        Suppress progress output
  -h, --help         Print help
```

### `voice transcribe`

```
Transcribe a WAV audio file

Usage: voice transcribe <FILE>
```

### `voice serve`

```
Run as a JSON-RPC 2.0 server on stdin/stdout

Usage: voice serve [OPTIONS]

Options:
  -v, --voice <VOICE>            Voice name [default: af_heart]
  -s, --speed <SPEED>            Speech speed factor [default: 1.0]
      --sub <WORD=REPLACEMENT>   Word substitution (repeatable)
      --sub-file <PATH>          Load substitutions from a file
```

## Speech-to-text

STT uses [distil-whisper](https://huggingface.co/distil-whisper/distil-large-v3) models running on Metal GPU via candle — knowledge-distilled versions of OpenAI's Whisper optimized for fast on-device transcription.

| Model | Repo ID | Params | Notes |
|-------|---------|--------|-------|
| Distil Large v3 | `distil-whisper/distil-large-v3` | 756M | Multilingual (default) |
| Distil Medium English | `distil-whisper/distil-medium.en` | 394M | English-only, faster |

Performance is ~50× real-time on Apple Silicon (a 10-second recording transcribes in ~200ms). Configs and tokenizers for known models are embedded in the binary.

**Adaptive noise floor**: Before recording, `voice listen` calibrates against ambient noise for ~500ms, then sets a silence threshold relative to the noise floor. This avoids false triggers in noisy environments and missed speech in quiet ones. A **ding** sound plays when the mic is ready.

**Model selection**: The default model is `distil-whisper/distil-large-v3`. Override with the `STT_MODEL` environment variable:

```bash
STT_MODEL=distil-whisper/distil-medium.en voice listen
```

## JSON-RPC server

`voice serve` runs a JSON-RPC 2.0 server on stdin/stdout, designed for integration with AI agents and tool-using LLMs.

### Methods

| Method | Description |
|--------|-------------|
| `speak` | Speak text or phonemes. Params: `text`, `phonemes`, `voice`, `speed`, `markdown`, `detail` |
| `listen` | Record from mic and transcribe. Params: `max_duration_ms`, `silence_timeout_ms`, `silence_threshold`, `noise_multiplier`, `calibration_ms` |
| `cancel` | Interrupt current speak playback |
| `set_voice` | Change the default voice. Params: `voice` |
| `set_speed` | Change the default speed. Params: `speed` |
| `list_voices` | List available builtin voices |
| `ping` | Health check — returns `"pong"` |

When `detail` is `"full"`, `speak` emits `speak.progress` notifications with chunk/phoneme info as audio streams.

### Example session

```jsonc
// Client → Server
{"jsonrpc": "2.0", "method": "speak", "params": {"text": "Hello! I'm listening."}, "id": 1}
// Server → Client
{"jsonrpc": "2.0", "result": {"duration_ms": 1520, "chunks": 1}, "id": 1}

// Client → Server
{"jsonrpc": "2.0", "method": "listen", "params": {"silence_timeout_ms": 2000}, "id": 2}
// Server → Client (after user speaks and silence is detected)
{"jsonrpc": "2.0", "result": {"text": "What's the weather like?", "tokens": 6, "duration_ms": 3200}, "id": 2}
```

See [`examples/conversation.py`](examples/conversation.py) for a full speak/listen conversation loop.

## LLM-friendly design

`voice` is built to work well with AI agents and coding assistants:

- **Phoneme output**: The CLI emits phoneme chunks to stderr, so agents can see the IPA representation of what's being spoken
- **Phoneme input**: `--phonemes` accepts raw IPA strings, giving agents precise control over pronunciation without going through G2P
- **Stdin pipe**: `echo "text" | voice say` lets agents speak from any script or tool
- **Markdown stripping**: `--markdown` cleans up LLM-generated markdown before speaking
- **Word substitutions**: `--sub` and `.voice-subs` files let you fix pronunciation of project-specific terms
- **JSON-RPC server**: `voice serve` gives agents structured, bidirectional TTS + STT over stdin/stdout

See [SKILL.md](SKILL.md) for a lightweight reference card that AI agents can use to learn the `voice` tool.

## Word substitutions

Fix pronunciation of names, acronyms, or technical terms.

### Inline

```bash
voice say --sub nteract=enteract --sub PyTorch=pie-torch "nteract uses PyTorch"
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
voice say --markdown -f README.md

# Or specify a file explicitly
voice say --sub-file my-project.subs -f notes.txt
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

## Crates

| Crate | Description |
|-------|-------------|
| [`voice`](https://crates.io/crates/voice) | CLI binary — installs as `voice` |
| [`voice-tts`](https://crates.io/crates/voice-tts) | Core TTS library — model loading, inference, WAV output |
| [`voice-stt`](https://crates.io/crates/voice-stt) | Speech-to-text library — Whisper transcription, resampling |
| [`voice-kokoro`](https://crates.io/crates/voice-kokoro) | Kokoro TTS backend — ALBERT encoder, prosody predictor, iSTFT decoder |
| [`voice-whisper`](https://crates.io/crates/voice-whisper) | Whisper STT backend — greedy decoding, GPU mel spectrogram |
| [`voice-g2p`](https://crates.io/crates/voice-g2p) | Grapheme-to-phoneme — misaki dictionary + espeak-ng fallback |

## Library usage

Add the crates you need:

```toml
[dependencies]
voice-tts = "0.2"
voice-stt = "0.1"   # if you need speech-to-text
voice-g2p = "0.2"   # if you need text-to-phoneme conversion
```

### Text-to-speech

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

    let chunks = voice_g2p::text_to_phoneme_chunks("Hello world, this is a test.")
        .expect("G2P failed");

    let mut all_samples: Vec<f32> = Vec::new();
    for phonemes in &chunks {
        let audio = voice_tts::generate(&mut model, phonemes, &voice, 1.0)?;
        all_samples.extend_from_slice(audio.as_slice());
    }

    voice_tts::save_wav_samples(&all_samples, std::path::Path::new("output.wav"), 24000)?;
    Ok(())
}
```

### Speech-to-text

```rust
fn main() -> voice_stt::Result<()> {
    let mut model = voice_stt::load_model("distil-whisper/distil-large-v3")?;
    let result = voice_stt::transcribe(&mut model, "audio.wav")?;
    println!("{}", result.text);
    Ok(())
}
```

## Architecture

### TTS: Kokoro (82M)

- **G2P pipeline**: Ports [misaki](https://github.com/hexgrad/misaki)'s English G2P — POS tagging (embedded averaged perceptron), 90k gold + 93k silver dictionary entries, morphological decomposition, number/currency handling, espeak-ng fallback
- **Inference**: StyleTTS2-based model with ISTFT vocoder head. Audio chunks stream to speakers as they're generated — the first chunk plays while subsequent chunks are still synthesizing
- **Startup**: Model loads in a background thread while text resolution, G2P, and voice loading happen on the main thread

### STT: Whisper (distil-large-v3 / distil-medium.en)

- **GPU mel spectrogram**: Preprocessing runs on Metal GPU via candle
- **Encoder-decoder transformer**: Standard Whisper architecture with knowledge distillation for faster inference
- **Greedy decode with KV cache**: Encoder output is computed once, then cached cross-attention keys/values are reused across all decoder steps
- **Embedded configs**: Tokenizers and model configs for known distil-whisper models are built into the binary

## Requirements

- macOS with Apple Silicon (Metal GPU acceleration)
- Rust 1.85+
- Git LFS (`brew install git-lfs && git lfs install`)
- Xcode command line tools
- espeak-ng (optional, for G2P fallback on unknown words): `brew install espeak-ng`

## License

MIT