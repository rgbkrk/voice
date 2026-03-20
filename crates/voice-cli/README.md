# voice

Like `say`, but with [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) TTS and [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny) STT. A command-line speech tool for macOS, powered by MLX on Apple Silicon.

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

> **Note:** `cargo install voice` is the package name on crates.io, but building from source is recommended — see the [main README](../../README.md) for details on why.

## Usage

```bash
# Just talk (backward compatible — no subcommand needed)
voice Hello world

# Text-to-speech with the say subcommand
voice say -v am_michael "How are you today?"
voice say -f script.txt -o output.wav
echo "Hello" | voice say
voice say --markdown -f post.mdx

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
  listen      Record from microphone and transcribe (speech-to-text)
  transcribe  Transcribe a WAV audio file
  serve       Run as a JSON-RPC 2.0 server on stdin/stdout

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