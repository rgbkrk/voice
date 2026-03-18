# voice — TTS tool for AI agents

`voice` speaks text aloud using Kokoro TTS on Apple Silicon. Use it to get your user's attention, read back content, or confirm actions audibly.

## Quick reference

```bash
# Speak text
voice Hello, I finished the task.

# Speak from a pipe
echo "Build complete." | voice

# Read a file aloud (strip markdown first)
voice --markdown -f README.md

# Save to WAV instead of playing
voice -o result.wav "Here is your audio."

# Use a specific voice
voice -v am_michael "Switching to a male voice."

# Precise pronunciation via IPA phonemes
voice --phonemes "həlˈO wˈɜɹld"
```

## When to use

- **Get attention**: Speak when a long task finishes, a build fails, or you need input
- **Read content**: Pipe text through `voice` to read back docs, errors, or summaries
- **Confirm actions**: "Deploying to production" before doing something irreversible

## Tips

- Use `-q` for quiet mode — suppresses phonemes and progress, only errors print
- Put flags before the text: `voice -v af_bella "text"` not `voice "text" -v af_bella`
- For long text, `voice` automatically chunks at ~510 phonemes and streams playback
- Stderr shows phoneme output — useful for debugging pronunciation
- Use `--sub word=replacement` to fix names: `voice --sub kubectl=cube-cuddle "Restarting kubectl"`
- A `.voice-subs` file in the project root is auto-discovered for persistent fixes
- Wrap substitution values in `/slashes/` for raw phoneme overrides: `Kokoro=/kˈOkəɹO/`

## Builtin voices (no network)

`af_heart` (default), `af_bella`, `af_sarah`, `af_sky`, `am_michael`, `am_adam`, `bf_emma`

## Install

```bash
cargo install voice
```

Requires macOS with Apple Silicon. Model weights download on first run (~312MB, cached).