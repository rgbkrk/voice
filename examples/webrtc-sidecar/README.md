# WebRTC Sidecar Spike

This is a minimal Python `aiortc` sidecar for proving the live-call media
boundary without pulling WebRTC into the Rust workspace yet.

It accepts a remote SDP offer over local HTTP, creates a WebRTC answer, sends a
48 kHz mono 20 ms PCM source as an outbound audio track, and writes decoded
inbound audio to a raw PCM file. `aiortc` handles Opus RTP, ICE, DTLS, and SRTP.

This is a spike. It does not call the WhatsApp Graph API, authenticate requests,
run VAD, or manage Hermes sessions. Hermes should still own WhatsApp Cloud
webhooks and Graph actions such as `pre_accept`, `accept`, and `terminate`.

## Install

```bash
python3 -m venv /tmp/voice-webrtc-venv
/tmp/voice-webrtc-venv/bin/pip install -r examples/webrtc-sidecar/requirements.txt
```

## Run

Create a FIFO for outbound audio from `voice`:

```bash
mkfifo /tmp/voice-webrtc-out.s16le
/tmp/voice-webrtc-venv/bin/python examples/webrtc-sidecar/sidecar.py \
  --tx-pcm /tmp/voice-webrtc-out.s16le \
  --rx-pcm /tmp/voice-webrtc-in.s16le
```

Feed TTS frames into the FIFO:

```bash
voice stream --sample-rate 48000 --frame-ms 20 \
  --raw-output /tmp/voice-webrtc-out.s16le \
  "Hello from the WebRTC sidecar."
```

You can also omit `--tx-pcm` and POST outbound frames over local HTTP once a
call session exists. This is the shape Hermes can use when it receives
`stream_speak` frames from the voice daemon:

```bash
curl -sS -X POST http://127.0.0.1:8787/calls/local-test/audio \
  -H 'content-type: application/json' \
  -d '{
    "sample_rate": 48000,
    "channels": 1,
    "frame_ms": 20,
    "encoding": "pcm_s16le",
    "pcm_s16le_base64": "AAAAAA=="
  }'
```

Each 20 ms frame is 1920 bytes before base64 encoding. If both `--tx-pcm` and
HTTP input are idle, the sidecar sends silence. That is useful for checking SDP,
ICE, and call timing before TTS is wired in.

## SDP API

Check process health and the fixed local audio contract:

```bash
curl -sS http://127.0.0.1:8787/health
```

Post a remote offer:

```bash
curl -sS http://127.0.0.1:8787/offer \
  -H 'content-type: application/json' \
  -d '{"call_id":"local-test","type":"offer","sdp":"v=0..."}'
```

Response:

```json
{
  "call_id": "local-test",
  "type": "answer",
  "sdp": "v=0...",
  "audio": {
    "sample_rate": 48000,
    "channels": 1,
    "frame_ms": 20,
    "encoding": "pcm_s16le"
  }
}
```

Inspect a live session:

```bash
curl -sS http://127.0.0.1:8787/calls/local-test
```

Queue outbound PCM for a live session:

```bash
curl -sS -X POST http://127.0.0.1:8787/calls/local-test/audio \
  -H 'content-type: application/json' \
  -d '{"sample_rate":48000,"channels":1,"frame_ms":20,"encoding":"pcm_s16le","pcm_s16le_base64":"AAAAAA=="}'
```

Close a session:

```bash
curl -X POST http://127.0.0.1:8787/calls/local-test/close
```

## WhatsApp Calling Placement

For WhatsApp Cloud Calling, Hermes should translate a `connect` webhook into a
local `/offer` call, then pass the returned SDP answer to the Cloud API
`pre_accept` and `accept` actions. The sidecar should already have an outbound
audio track attached and ready before Hermes calls `accept`; the track sends
silence until real TTS PCM arrives.

Inbound decoded PCM is written as raw signed 16-bit little-endian mono samples.
A later bridge layer can segment that stream with VAD and submit each segment
directly to the daemon STT stream contract:

```bash
voice stream-transcribe \
  --raw-input /tmp/voice-webrtc-in.s16le \
  --sample-rate 48000 \
  --frame-ms 20
```

Use `--raw-input -` when piping decoded PCM from another process.

## Tests

The sidecar tests are optional because they need the Python WebRTC dependency
set:

```bash
python3 -m venv /tmp/voice-webrtc-venv
/tmp/voice-webrtc-venv/bin/pip install -r examples/webrtc-sidecar/requirements.txt pytest
/tmp/voice-webrtc-venv/bin/python -m pytest -q examples/webrtc-sidecar/test_sidecar.py
```
