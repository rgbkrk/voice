# Streaming TTS

`voice` exposes two daemon-backed TTS surfaces:

- `synthesize`: write a completed WAV file. This is the compatibility path for
  tools that expect `{input_path} -> {output_path}`, including Hermes command
  TTS providers.
- `stream_speak`: emit ordered audio events over the daemon socket. This is the
  transport-neutral path for WebRTC, voice bridges, or any client that wants
  audio before a final file exists.

## Hermes / WhatsApp

Hermes' current WhatsApp path sends local media files to the WhatsApp bridge.
The bridge reads the completed file and converts non-Opus audio to OGG/Opus for
native `ptt` voice-note delivery. Use daemon synthesis for that path:

```yaml
tts:
  provider: kokoro
  providers:
    kokoro:
      type: command
      command: /path/to/voice say --input-file {input_path} --output {output_path} --voice {voice} --speed {speed}
      output_format: wav
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

Set `voice_compatible: true` so Hermes converts the WAV output to OGG/Opus
when it needs native voice-note delivery. This requires `ffmpeg` in Hermes'
environment and lets WhatsApp/Telegram paths receive an Opus voice note instead
of a generic audio attachment.

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
