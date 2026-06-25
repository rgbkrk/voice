# voice — TTS & STT tool for AI agents

`voice` speaks text aloud using Kokoro or Voxtral TTS and transcribes speech with Whisper STT on Apple Silicon. Prefer the daemon-backed UI queue for agent messages: MCP `speak` and `converse` requests are held in the Voice UI by default, while CLI `voice say` and `voice converse` remain immediate user-facing commands.

## Quick reference

### Speak (TTS)

```bash
# Speak text (backward compatible — no subcommand needed)
voice Hello, I finished the task.

# Explicit say subcommand with options
voice say -v af_sky "Switching to a brighter voice."

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
voice converse -v af_sky -s 1.2 "What do you think about that?"
```

### Daemon UI queue

```bash
# Start the daemon locally. Without --tts-only it eagerly loads TTS and STT.
voice daemon start

# Inspect queued, current, and recent daemon items.
voice daemon status
voice daemon status --json

# Open the browser UI served by the daemon.
open http://127.0.0.1:8767/

# Replay audio saved for a UI queue item.
voice daemon replay --part question <queue_id>
voice daemon replay --part answer <queue_id>
```

For agents, use `voice mcp`. Its `speak` and `converse` tools enqueue held UI messages unless `immediate=true` is supplied. Use held messages when the user should decide when to listen or respond; use `immediate=true` only for intentional interruption or direct playback.

### Listen (STT)

```bash
# Record from mic, transcribe on Enter/Ctrl+C
voice listen

# Continuous mode — transcribe segments as you speak, split on silence
voice listen --continuous

# Transcribe a WAV file
voice transcribe recording.wav
```

### MCP server

```bash
# Start the MCP server for agent integration.
voice mcp
```

```jsonl
# Queue a held UI playback item.
→ {"jsonrpc":"2.0","method":"tools/call","params":{"name":"speak","arguments":{"text":"Build finished."}},"id":1}

# Queue a held UI item that needs the user to respond.
→ {"jsonrpc":"2.0","method":"tools/call","params":{"name":"converse","arguments":{"text":"Should I open the PR?"}},"id":2}

# Play immediately instead of queueing.
→ {"jsonrpc":"2.0","method":"tools/call","params":{"name":"speak","arguments":{"text":"Urgent interrupt.","immediate":true}},"id":3}
```

## When to use

- **Leave a message**: Use MCP `speak` to add a held item to the daemon UI queue
- **Ask for input**: Use MCP `converse` to add a response-needed item to the daemon UI queue
- **Get attention immediately**: Use `immediate=true` only when interruption is intended
- **Read content locally**: Pipe text through `voice say` to read back docs, errors, or summaries
- **Confirm actions**: "Deploying to production" before doing something irreversible
- **Listen for input**: Use `voice listen` to capture a spoken response from the user
- **Voice conversation**: Use `voice converse` for a foreground CLI speak/listen turn
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
- Default model is `distil-whisper/distil-large-v3`
- Use `STT_MODEL=distil-whisper/distil-medium.en` for a faster English-only fallback

### Daemon/MCP tips

- `voice daemon start` eagerly loads TTS and STT. `voice daemon start --tts-only` skips eager STT load for lightweight servers.
- The daemon serves the Voice UI at `http://127.0.0.1:8767/`.
- MCP `speak` and `converse` default to held UI queue items.
- Set MCP `immediate=true` to bypass the held UI queue and perform the action now.
- Browser UI playback uses saved WAV files from `~/.voice/audio/<queue_id>-q.wav` and response audio from `~/.voice/audio/<queue_id>-a.wav`.

## Subcommands

| Command | What it does |
|---------|-------------|
| `voice <text>` | Speak text (implicit `say`, backward compatible) |
| `voice say` | Speak text with full TTS options |
| `voice converse` | Speak text, then listen for a response |
| `voice listen` | Record from mic, transcribe once |
| `voice listen --continuous` | Record and transcribe segments continuously |
| `voice transcribe <file>` | Transcribe a WAV file |
| `voice mcp` | Start MCP server on stdin/stdout |
| `voice daemon start` | Start daemon, socket API, and browser UI |
| `voice daemon status` | Inspect daemon queue state |

## Builtin voices (no network)

`af_heart` (default), `af_bella`, `af_sarah`, `af_sky`, `am_michael`, `am_adam`, `bf_emma`

## Install

```bash
git clone https://github.com/rgbkrk/voice.git
cd voice
cargo install --path crates/voice-cli
```

Requires macOS with Apple Silicon, Git LFS, and Rust 1.85+. TTS model weights download on first `voice say` (~312MB, cached). STT model weights download on first `voice listen` (~246MB, cached).
