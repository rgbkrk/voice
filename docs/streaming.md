# Streaming TTS

`voice` exposes file and streaming surfaces relevant to integrations:

- `say --output`: write a completed audio file. `.wav` writes float WAV, while
  `.ogg` / `.opus` or `--format ogg-opus` writes real
  `audio/ogg; codecs=opus` for WhatsApp-style voice notes. OGG/Opus output
  requires `ffmpeg` on `PATH`.
- `synthesize`: daemon RPC for completed files. It infers `.wav`, `.ogg`, and
  `.opus` output paths the same way as the CLI.
- `stream_speak`: emit ordered audio events over the daemon socket. This is the
  transport-neutral path for WebRTC, voice bridges, or any client that wants
  audio before a final file exists. `voice stream --output reply.ogg` can also
  encode those frames to streamed Ogg/Opus without a WAV intermediate.
- `stream_transcribe`: ingest ordered PCM events over the daemon socket and
  return a transcript after the client sends `stt.end`. This is the first
  inbound-audio contract for WebRTC sidecars. Partial transcripts are a later
  layer.

## Hermes / WhatsApp

Hermes' current WhatsApp path sends local media files to the WhatsApp bridge.
Use OGG/Opus output for native `ptt` voice-note delivery without an extra
Hermes-side conversion:

```yaml
tts:
  provider: kokoro
  providers:
    kokoro:
      type: command
      command: /path/to/voice say --format ogg-opus --input-file {input_path} --output {output_path} --voice {voice} --speed {speed}
      output_format: ogg
      voice_compatible: true
      voice: af_heart
      speed: 1.0
      timeout: 180
      max_text_length: 2000
```

Run the daemon first so the Kokoro model stays resident:

```bash
voice daemon start --tts-only
```

The macOS release archive includes the single `voice` binary. From source,
install it with `cargo install --path crates/voice-cli`.

`voice say -o ...` falls back to local synthesis if the daemon is unavailable,
so the same command still works outside Hermes.

Set `voice_compatible: true` so Hermes routes the returned `.ogg` file as a
native voice note. `output_format: wav` remains valid as a compatibility
fallback; Hermes and the WhatsApp bridge can still convert non-Opus audio when
needed. Telegram accepts the same Ogg/Opus voice-message shape for `sendVoice`
alongside MP3 and M4A uploads, so Voice does not need a Telegram-specific
encoder.

For live WhatsApp Calling, keep Ogg/Opus as the file path and bridge WebRTC
media through 48 kHz 20 ms PCM frames. See
[whatsapp-calling-webrtc.md](whatsapp-calling-webrtc.md).

The repository also includes `examples/hermes-command-tts.sh`, which matches
Hermes' command-provider argument order and explicitly asks `voice` for
Ogg/Opus output by default:

```yaml
command: /path/to/voice/examples/hermes-command-tts.sh {input_path} {output_path} {voice} {speed}
```

Set `VOICE_BIN=/path/to/voice` if `voice` is not on `PATH`.
Set `VOICE_FORMAT=wav` for a WAV compatibility provider, or set
`VOICE_FORMAT=` to let `voice` infer the format from `{output_path}`.

For command-provider STT, use the matching voice-owned shim:

```yaml
stt:
  enabled: true
  provider: voice
  providers:
    voice:
      type: command
      command: /path/to/voice/examples/hermes-command-stt.sh {input_path}
      format: txt
      timeout: 300
```

The direct equivalent is:

```yaml
command: /path/to/voice stream-transcribe --quiet {input_path}
```

Both forms are accepted by `scripts/verify_hermes_voice_config.py`; arbitrary
wrappers are rejected so config drift is visible before restarting Hermes.
Pass `--stt-audio ~/.hermes/audio_cache/aud_...ogg` to execute the configured
STT command against a cached inbound WhatsApp voice note; the verifier only
reports transcript size, not transcript text.

To generate or install the voice-owned provider blocks, run:

```bash
scripts/install_hermes_voice_config.py --print-snippet --voice-bin "$(command -v voice)"
scripts/install_hermes_voice_config.py --config ~/.hermes/config.yaml --voice-bin "$(command -v voice)"
scripts/install_hermes_voice_config.py --config ~/.hermes/config.yaml --voice-bin "$(command -v voice)" --apply
```

The default run is a dry run: it patches a temporary copy and verifies the
resulting shape. `--apply` writes the config, keeps a timestamped backup unless
`--no-backup` is passed, and reruns the Hermes config verifier.

Validate the command-provider shape locally before wiring or restarting
Hermes:

```bash
VOICE_BIN=/path/to/voice scripts/verify_hermes_command_tts.sh
```

The script simulates Hermes' `{input_path} {output_path} {voice} {speed}`
invocation, then checks the result with `ffprobe` to confirm an Ogg container
with Opus audio, mono, at 48 kHz.

To validate an installed Hermes tree without touching a running gateway, run
Hermes' TTS tool directly from that install after confirming its config points
at a voice-native provider:

```bash
cd ~/.hermes/hermes-agent
HERMES_HOME=~/.hermes ./venv/bin/python - <<'PY'
import json
from tools.tts_tool import text_to_speech_tool

result = text_to_speech_tool("Hermes voice native Ogg Opus smoke test.")
print(json.dumps(json.loads(result), indent=2, ensure_ascii=False))
PY
```

Inspect the returned file with `ffprobe`; a voice-compatible WhatsApp path
should return `.ogg` / Opus audio and should not need another Hermes-side
conversion.

On a Linux host that has the Hermes gateway installed as a user service, also
check the live service drop-in:

```bash
VOICE_BIN=/path/to/voice scripts/verify_hermes_gateway_service.py
VOICE_BIN=/path/to/voice scripts/verify_local_hermes_voice_stack.sh
```

The gateway verifier confirms `hermes-gateway.service` is active, points at the
expected Hermes home, exports `WHATSAPP_CLOUD_CALLING_SIDECAR_URL`, and uses a
`voice stream --raw-output -` command for the WhatsApp Calling sidecar path.
The aggregate stack verifier also dry-runs `install_hermes_voice_config.py`
against the active config, so one command now checks both config drift
detection and the voice-owned repair path.
If any step fails, it emits `failure_category=...` so automation can distinguish
voice runtime, Hermes config/service, WhatsApp bridge or credential, inbound
audio, and WebRTC sidecar failures.

## CLI Smoke Test

Use `voice stream` to inspect the streaming event flow:

```bash
voice stream-contract
voice stream "Hello from the stream"
voice stream --json "Hello from the stream"
voice say --format ogg-opus -o reply.ogg "Hello"
voice stream --output streamed.ogg --format ogg-opus "Hello"
voice stream --output - --format ogg-opus "Hello" > streamed.ogg
voice stream --sample-rate 48000 --frame-ms 20 --raw-output reply.s16le "Hello"
voice stream-transcribe recording.ogg
voice stream-transcribe --raw-input webrtc-in.s16le --sample-rate 48000 --frame-ms 20
voice stream-transcribe --json recording.ogg
```

For a single voice-side WhatsApp preflight, run:

```bash
VOICE_BIN=/path/to/voice scripts/verify_whatsapp_voice_contract.sh
```

For the equivalent Telegram preflight, run:

```bash
VOICE_BIN=/path/to/voice scripts/verify_telegram_voice_contract.sh
VOICE_BIN=/path/to/voice scripts/verify_telegram_voice_contract.sh --require-telegram-credentials
```

Those verifiers check `voice stream-contract` against the checked-in sidecar
contract, prove `voice say --format ogg-opus` and `.ogg` extension inference
write real mono 48 kHz Ogg/Opus, verify misleading `.wav`/`ogg-opus`
combinations are rejected before writing a file, and, when the daemon is
running, check both raw 48 kHz 20 ms PCM streaming and streamed Ogg/Opus
encoding to both named files and stdout. The Telegram verifier also checks the
active Hermes command-provider config unless `--skip-hermes-config` is passed,
and reports whether the Hermes env file contains Telegram credentials without
printing secret values. Pass `--require-telegram-credentials` when setup should
fail unless `TELEGRAM_BOT_TOKEN` is present. Pass `--require-daemon` when the
daemon stream path must be covered, or `--skip-daemon` for a file-only
preflight. Pass `--run-stt-smoke` with `--require-daemon` when the inbound
WebRTC/STT path must also be covered; it creates a tiny WAV fixture and requires
`voice stream-transcribe --json` to return a terminal `stt.transcribed` event.
This is optional because it may lazily load the Whisper model.

The raw output is signed 16-bit little-endian mono PCM with no container header.
Use the event metadata for sample rate, frame duration, and stream ID.
When `--raw-output -` is used, stdout is reserved for PCM bytes and compact
progress lines move to stderr.

`--output` is the encoded stream path. It currently accepts `.ogg` / `.opus` or
`--format ogg-opus`, requires `ffmpeg`, and preserves the daemon's low-latency
PCM frame contract while producing a valid `audio/ogg; codecs=opus` file.
Use `--output - --format ogg-opus` to pipe that encoded Ogg stream to stdout;
`--format` is required with `-` so binary stdout is unambiguous, and `--json`
cannot be combined with binary stdout.
Use `--raw-output` when the consumer is a WebRTC sidecar or another process
that wants raw PCM frames.

`voice stream-transcribe` reads an audio file or explicit raw `pcm_s16le` input,
splits it into the same ordered PCM frames a WebRTC sidecar would send, and
returns the terminal STT event from the daemon. WAV input is decoded directly;
Ogg/Opus and other compressed formats use `ffmpeg` when available. Use
`--raw-input -` to pipe decoded WebRTC PCM directly from another process. It is
a transport smoke test; `voice transcribe` remains the direct file transcription
command.

`voice stream-contract` prints the same machine-readable sidecar contract used
by the Python WebRTC example. Besides the fixed PCM shape and HTTP endpoint
schema, including the outbound-audio clear endpoint used for barge-in, the
`voice_surfaces` object maps integration modes to commands:
`completed_voice_note` for WhatsApp-ready Ogg/Opus files, `streamed_voice_note`
for Ogg/Opus encoded from daemon frames to a named file,
`streamed_voice_note_stdout` for pipeable Ogg/Opus stdout, `raw_outbound_pcm`
for WebRTC TTS frames, `raw_inbound_pcm` for decoded WebRTC audio entering STT,
and `file_transcription_smoke` for replaying an audio file through the inbound
stream contract.

## Daemon Protocol

### TTS Output

Send a JSON-RPC request in a `Request` frame:

```json
{
  "jsonrpc": "2.0",
  "method": "stream_speak",
  "params": {
    "text": "Hello",
    "voice": "af_heart",
    "speed": 1.0,
    "sample_rate": 48000,
    "frame_ms": 20
  },
  "id": 1
}
```

The daemon replies with a normal JSON-RPC `Response` frame:

```json
{
  "jsonrpc": "2.0",
  "result": {
    "queue_id": "abcd1234",
    "stream_id": "...",
    "status": "queued"
  },
  "id": 1
}
```

It then emits `Event` frames until a terminal event:

- `tts.started`: stream metadata, sample rate, encoding, frame duration, voice,
  speed, and total phoneme chunks.
- `tts.audio`: one ordered PCM frame with `sequence`, `offset_samples`,
  `sample_count`, `padding_samples`, and `samples`.
- `tts.ended`: frame count, sample count, audio duration, and elapsed synthesis
  time.
- `tts.error`: terminal failure with a message.
- `tts.cancelled`: terminal cancellation with a reason.

Audio frames are fixed-duration PCM packets. The last frame is padded with
silence when needed so clients can feed frames directly into an Opus encoder.
Use `sample_rate: 48000` and `frame_ms: 20` for a WebRTC-friendly stream.
The WebRTC-friendly constants live in the `voice-stream` crate and are checked
against the machine-readable sidecar v1 shape in
[`docs/contracts/webrtc-sidecar-v1.json`](contracts/webrtc-sidecar-v1.json).
Installed binaries expose the same object with:

```bash
voice stream-contract
```

### STT Input

Start a client-to-daemon transcription stream with a `Request` frame:

```json
{
  "jsonrpc": "2.0",
  "method": "stream_transcribe",
  "params": {
    "sample_rate": 48000,
    "channels": 1,
    "encoding": "pcm_s16le",
    "frame_ms": 20,
    "max_duration_ms": 300000
  },
  "id": 1
}
```

The daemon replies with the stream metadata:

```json
{
  "jsonrpc": "2.0",
  "result": {
    "stream_id": "...",
    "status": "receiving",
    "sample_rate": 48000,
    "channels": 1,
    "encoding": "pcm_s16le",
    "frame_ms": 20,
    "max_duration_ms": 300000
  },
  "id": 1
}
```

The client then sends `Event` frames with `stt.audio` payloads. The frame shape
matches `tts.audio` closely, but only `stream_id`, `sample_rate`, `channels`,
`encoding`, and `samples` are required today:

```json
{
  "event": "stt.audio",
  "data": {
    "frame": {
      "stream_id": "...",
      "sequence": 0,
      "sample_rate": 48000,
      "channels": 1,
      "encoding": "pcm_s16le",
      "frame_ms": 20,
      "sample_count": 960,
      "samples": [0, 12, -8]
    }
  }
}
```

End the stream with:

```json
{
  "event": "stt.end",
  "data": {
    "stream_id": "..."
  }
}
```

The daemon enqueues one STT job and emits one terminal event:

```json
{
  "event": "stt.transcribed",
  "data": {
    "stream_id": "...",
    "queue_id": "...",
    "frames": 42,
    "text": "hello",
    "tokens": 8,
    "sample_rate": 48000,
    "audio_duration_ms": 840,
    "elapsed_ms": 120
  }
}
```

If validation or transcription fails, the terminal event is `stt.error` with a
`message` field.

`max_duration_ms` defaults to 60 seconds and is capped at 300 seconds. Treat
each `stream_transcribe` session as one utterance or segment from the WebRTC
sidecar rather than a whole call.
