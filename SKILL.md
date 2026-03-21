# voice — TTS & STT tool for AI agents

`voice` speaks text aloud using Kokoro TTS and transcribes speech using Moonshine STT on Apple Silicon. Use it to talk to your user, listen for their response, or run a full voice conversation loop.

## Quick reference

### Speak (TTS)

```bash
# Speak text (backward compatible — no subcommand needed)
voice Hello, I finished the task.

# Explicit say subcommand with options
voice say -v am_michael "Switching to a male voice."

# Speak from a pipe
echo "Build complete." | voice say

# Read a file aloud (strip markdown first)
voice say --markdown -f README.md

# Save to WAV instead of playing
voice say -o result.wav "Here is your audio."

# Precise pronunciation via IPA phonemes
voice say --phonemes "həlˈO wˈɜɹld"
```

### Converse (speak + listen)

```bash
# Speak text, then immediately listen for a response
voice converse "How are you today?"

# With voice and speed options
voice converse -v am_michael -s 1.2 "What do you think about that?"
```

### Listen (STT)

```bash
# Record from mic, transcribe on Enter/Ctrl+C
voice listen

# Continuous mode — transcribe segments as you speak, split on silence
voice listen --continuous

# Transcribe a WAV file
voice transcribe recording.wav
```

### JSON-RPC server

```bash
# Start the server (for programmatic control)
voice serve -v am_michael
```

```jsonl
# Speak
→ {"jsonrpc":"2.0","method":"speak","params":{"text":"Hello"},"id":1}
← {"jsonrpc":"2.0","result":{"duration_ms":1800,"chunks":1},"id":1}

# Listen (ding plays, records, auto-stops on silence)
→ {"jsonrpc":"2.0","method":"listen","id":2}
← {"jsonrpc":"2.0","result":{"text":"I heard you","tokens":4,"duration_ms":3200},"id":2}

# Cancel current playback or recording
→ {"jsonrpc":"2.0","method":"cancel","id":3}

# Other methods: set_voice, set_speed, list_voices, ping
```

## When to use

- **Get attention**: Speak when a long task finishes, a build fails, or you need input
- **Read content**: Pipe text through `voice say` to read back docs, errors, or summaries
- **Confirm actions**: "Deploying to production" before doing something irreversible
- **Listen for input**: Use `voice listen` to capture a spoken response from the user
- **Voice conversation**: Use `voice converse` to speak then listen in one shot, or `voice serve` for programmatic control
- **Transcribe recordings**: Use `voice transcribe` to convert audio files to text

## Tips

### TTS tips

- Use `-q` for quiet mode — suppresses phonemes and progress, only errors print
- For long text, `voice` automatically chunks at ~510 phonemes and streams playback
- Stderr shows phoneme output — useful for debugging pronunciation
- Use `--sub word=replacement` to fix names: `voice say --sub kubectl=cube-cuddle "Restarting kubectl"`
- A `.voice-subs` file in the project root is auto-discovered for persistent fixes
- Wrap substitution values in `/slashes/` for raw phoneme overrides: `Kokoro=/kˈOkəɹO/`

### STT tips

- A ding sound plays when the mic is ready — wait for it before speaking
- Bluetooth mics (AirPods) have ~0.5s latency; the ding helps you time it
- Noise floor is calibrated automatically — works with MacBook mic or AirPods
- Use `STT_MODEL=UsefulSensors/moonshine-tiny` for faster (but less accurate) transcription
- Default model is `moonshine-base` (61M params, ~50× real-time on Apple Silicon)

### JSON-RPC tips

- `voice serve` loads the TTS model at startup; STT model loads lazily on first `listen`
- `cancel` interrupts the current speak or listen mid-operation
- `speak` supports per-request `voice` and `speed` overrides without changing defaults
- `listen` params are tunable: `noise_multiplier`, `calibration_ms`, `silence_timeout_ms`
- Notifications (requests without `id`) are fire-and-forget — no response returned

## Subcommands

| Command | What it does |
|---------|-------------|
| `voice <text>` | Speak text (implicit `say`, backward compatible) |
| `voice say` | Speak text with full TTS options |
| `voice converse` | Speak text, then listen for a response |
| `voice listen` | Record from mic, transcribe once |
| `voice listen --continuous` | Record and transcribe segments continuously |
| `voice transcribe <file>` | Transcribe a WAV file |
| `voice serve` | Start JSON-RPC server on stdin/stdout |

## Builtin voices (no network)

`af_heart` (default), `af_bella`, `af_sarah`, `af_sky`, `am_michael`, `am_adam`, `bf_emma`

## Install

```bash
git clone https://github.com/rgbkrk/voice.git
cd voice
cargo install --path crates/voice-cli
```

Requires macOS with Apple Silicon, Git LFS, and Rust 1.85+. TTS model weights download on first `voice say` (~312MB, cached). STT model weights download on first `voice listen` (~246MB, cached).