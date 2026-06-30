# voice — TTS & STT tool for AI agents

`voice` speaks text aloud using Kokoro or Voxtral TTS and transcribes speech with Whisper STT on Apple Silicon. A background daemon keeps the models warm and owns the audio hardware, serializing playback so utterances never overlap. MCP `speak` is fire-and-forget; MCP `converse` is async — it returns a `converse_id` immediately and you poll `converse_result` for the human's reply.

## Quick reference

### Speak (TTS)

```bash
# Speak text (the `say` subcommand is required)
voice say "I finished the task."

# With voice and other options
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
# Speak text, then listen for a spoken response (foreground CLI turn)
voice converse "How are you today?"

# With voice and speed options
voice converse -v af_sky -s 1.2 "What do you think about that?"
```

### Daemon

```bash
# Start the daemon locally. Without --tts-only it eagerly loads TTS and STT.
voice daemon start
voice daemon start --tts-only

# Inspect queued, current, and recent daemon items.
voice daemon status
voice daemon status --json

# Cancel a queued item by id.
voice daemon cancel <queue_id>
```

The daemon listens on a Unix socket at `~/.voice/daemon.sock`. It has no browser UI. The CLI and MCP server both talk to it over that socket.

For agents, use `voice mcp`. Its `speak` tool enqueues audio and returns immediately; its `converse` tool returns a `converse_id` you then poll with `converse_result`. Neither blocks on the human, so the calls are safe to retry.

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
# Speak text aloud. Returns a queue_id immediately; the daemon plays it.
→ {"jsonrpc":"2.0","method":"tools/call","params":{"name":"speak","arguments":{"text":"Build finished."}},"id":1}

# Ask the human a question. Returns a converse_id immediately.
→ {"jsonrpc":"2.0","method":"tools/call","params":{"name":"converse","arguments":{"text":"Should I open the PR?"}},"id":2}

# Poll for the spoken reply. wait_ms long-polls up to that long (cap 30000) for a terminal result.
→ {"jsonrpc":"2.0","method":"tools/call","params":{"name":"converse_result","arguments":{"converse_id":"<id>","wait_ms":30000}},"id":3}
```

## When to use

- **Announce progress**: Use MCP `speak` to play a message; it returns at once and the daemon serializes playback
- **Ask for spoken input**: Use MCP `converse`, then poll `converse_result` for the transcript
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
- Default model is `distil-whisper/distil-large-v3.5` (English; fewer repetition errors than v3)
- Use `STT_MODEL=distil-whisper/distil-medium.en` for a smaller/faster model, or `STT_MODEL=openai/whisper-large-v3` for multilingual

### Daemon/MCP tips

- `voice daemon start` eagerly loads TTS and STT. `voice daemon start --tts-only` skips eager STT load for lightweight servers.
- The daemon serializes audio through a single worker, so MCP `speak` calls queue behind each other instead of overlapping.
- MCP `converse` never blocks on the human. Poll `converse_result` with the returned `converse_id`; `phase` walks queued → speaking → listening → completed, and `mic_active` is true while recording.
- Fetching by `converse_id` is idempotent, so a retried `converse_result` returns the same transcript instead of re-running anything.

## Subcommands

| Command | What it does |
|---------|-------------|
| `voice say` | Speak text with full TTS options |
| `voice converse` | Speak text, then listen for a response |
| `voice listen` | Record from mic, transcribe once |
| `voice listen --continuous` | Record and transcribe segments continuously |
| `voice transcribe <file>` | Transcribe a WAV file |
| `voice mcp` | Start MCP server on stdin/stdout |
| `voice daemon start` | Start the daemon and its socket API |
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
