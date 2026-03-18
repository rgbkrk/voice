# voice

Like `say`, but with [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M). A command-line TTS tool for macOS powered by MLX.

## Install

```bash
cargo install voice
```

This puts the `voice` binary on your `$PATH`.

## Usage

```bash
# Just talk
voice Hello world

# Pick a voice
voice -v am_adam "How are you today?"

# Pipe text in
echo "The quick brown fox jumps over the lazy dog." | voice

# Read from a file
voice -f script.txt

# Save to WAV instead of playing
voice -o speech.wav "Good morning everyone."

# Adjust speed
voice -s 0.8 "Take it slow."

# Raw phonemes (advanced)
voice --phonemes "həlˈO wˈɜɹld"
```

## Options

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

## Voices

**American**: `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`, `am_adam`, `am_michael`

**British**: `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`

See the full list in the [main README](../../README.md#available-voices).

## License

MIT