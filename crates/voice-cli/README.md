# voice

Like `say`, but with [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) TTS and [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny) STT. A command-line speech tool for macOS, powered by MLX on Apple Silicon.

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

> **Note:** `cargo install voice` compiles from source on crates.io, but the Metal shader library path can break — see the [main README](../../README.md) for details. Use `cargo binstall voice` or build from a local clone instead.

## Usage

```bash
# Just talk (backward compatible — no subcommand needed)
voice Hello world

# Text-to-speech with the say subcommand
voice say -v am_michael "How are you today?"
voice say -f script.txt -o output.wav
echo "Hello" | voice say
voice say --markdown -f post.mdx
voice phonemes "ChatGPT uses RuntimeStateDoc"
voice stream --sample-rate 48000 --frame-ms 20 --raw-output output.s16le "Hello"

# Speech-to-text from microphone
voice listen
voice listen --continuous

# Transcribe an audio file
voice transcribe recording.wav

# JSON-RPC 2.0 server on stdin/stdout
voice serve -v am_michael
```

## Options

### Top-level

```
Usage: voice [OPTIONS] [COMMAND] [TEXT]...

Commands:
  say         Speak text aloud (default when no subcommand given)
  phonemes    Convert text to phoneme chunks without synthesis
  stream      Stream TTS audio chunks from the voice daemon
  converse    Speak text aloud, then listen for a response
  listen      Record from microphone and transcribe (speech-to-text)
  transcribe  Transcribe a WAV audio file
  serve       Run as a JSON-RPC 2.0 server on stdin/stdout
  mcp         Run as an MCP server on stdin/stdout
  daemon      Inspect and control a running voice daemon

Arguments:
  [TEXT]...  Text to speak (shorthand for `voice say <text>`)

Options:
  -q, --quiet  Suppress progress output
  -h, --help   Print help
```

### `voice say`

```
Usage: voice say [OPTIONS] [TEXT]...

Options:
  -f, --input-file <FILE>        Read text from a file (use - for stdin)
      --phonemes <IPA>           Raw phoneme string (IPA)
  -v, --voice <VOICE>            Voice name [default: af_heart]
  -o, --output <PATH>            Write WAV to file instead of playing
  -s, --speed <SPEED>            Speech speed factor [default: 1.0]
      --markdown                 Strip markdown/MDX formatting before speaking
      --sub <WORD=REPLACEMENT>   Word substitution (repeatable)
      --sub-file <PATH>          Load substitutions from a file
```

### `voice phonemes`

```
Usage: voice phonemes [OPTIONS] [TEXT]...

Options:
  -f, --input-file <FILE>        Read text from a file (use - for stdin)
      --markdown                 Strip markdown/MDX formatting before conversion
      --sub <WORD=REPLACEMENT>   Word substitution (repeatable)
      --sub-file <PATH>          Load substitutions from a file
      --json                     Print a JSON object with preprocessed text and phoneme chunks
```

### `voice stream`

Requires a running `voiced` daemon. Emits ordered signed 16-bit little-endian
mono PCM frames as compact summaries or full JSON events.

```
Usage: voice stream [OPTIONS] [TEXT]...

Options:
  -f, --input-file <FILE>          Read text from a file (use - for stdin)
  -v, --voice <VOICE>              Voice name [default: af_heart]
  -s, --speed <SPEED>              Speech speed factor [default: 1.0]
      --sample-rate <SAMPLE_RATE>  Target stream sample rate [default: 24000]
      --frame-ms <FRAME_MS>        Target frame duration in milliseconds [default: 20]
  -o, --raw-output <PATH>          Write raw signed 16-bit little-endian mono PCM
      --json                       Print full JSON stream events
      --markdown                   Strip markdown/MDX formatting before speaking
      --sub <WORD=REPLACEMENT>     Word substitution (repeatable)
      --sub-file <PATH>            Load substitutions from a file
```

### `voice listen`

```
Usage: voice listen [OPTIONS]

Options:
      --continuous  Record and transcribe segments continuously
```

### `voice transcribe`

```
Usage: voice transcribe <FILE>
```

## Voices

**American**: `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`, `am_adam`, `am_michael`

**British**: `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`

See the full list in the [main README](../../README.md#builtin-voices).

## License

MIT
