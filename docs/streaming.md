# Streaming TTS

`voice` exposes three TTS surfaces relevant to integrations:

- `say --output`: write a completed audio file. `.wav` writes float WAV, while
  `.ogg` / `.opus` or `--format ogg-opus` writes real
  `audio/ogg; codecs=opus` for WhatsApp-style voice notes. OGG/Opus output
  requires `ffmpeg` on `PATH`.
- `synthesize`: daemon RPC for completed files. It infers `.wav`, `.ogg`, and
  `.opus` output paths the same way as the CLI.
- `stream_speak`: emit ordered audio events over the daemon socket. This is the
  transport-neutral path for WebRTC, voice bridges, or any client that wants
  audio before a final file exists.

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
voiced --tts-only
```

The macOS release archive includes `voiced` alongside `voice`. From source,
install the daemon with `cargo install --path crates/voice-daemon`.

`voice say -o ...` falls back to local synthesis if the daemon is unavailable,
so the same command still works outside Hermes.

Set `voice_compatible: true` so Hermes routes the returned `.ogg` file as a
native voice note. `output_format: wav` remains valid as a compatibility
fallback; Hermes and the WhatsApp bridge can still convert non-Opus audio when
needed.

For live WhatsApp Calling, keep Ogg/Opus as the file path and bridge WebRTC
media through 48 kHz 20 ms PCM frames. See
[whatsapp-calling-webrtc.md](whatsapp-calling-webrtc.md).

The repository also includes `examples/hermes-command-tts.sh`, which matches
Hermes' command-provider argument order:

```yaml
command: /path/to/voice/examples/hermes-command-tts.sh {input_path} {output_path} {voice} {speed}
```

Set `VOICE_BIN=/path/to/voice` if `voice` is not on `PATH`.

## CLI Smoke Test

Use `voice stream` to inspect the streaming event flow:

```bash
voice stream "Hello from the stream"
voice stream --json "Hello from the stream"
voice say --format ogg-opus -o reply.ogg "Hello"
voice stream --sample-rate 48000 --frame-ms 20 --raw-output reply.s16le "Hello"
voice stream --sample-rate 48000 --frame-ms 20 --raw-output - "Hello" \
  | ffmpeg -f s16le -ar 48000 -ac 1 -i - -c:a libopus reply.ogg
```

The raw output is signed 16-bit little-endian mono PCM with no container header.
Use the event metadata for sample rate, frame duration, and stream ID.
When `--raw-output -` is used, stdout is reserved for PCM bytes and compact
progress lines move to stderr.

## Daemon Protocol

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
