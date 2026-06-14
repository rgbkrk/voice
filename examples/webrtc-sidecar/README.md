# WebRTC Sidecar Spike

This is a minimal Python `aiortc` sidecar for proving the live-call media
boundary without pulling WebRTC into the Rust workspace yet.

It accepts a remote SDP offer over local HTTP, creates a WebRTC answer, sends a
48 kHz mono 20 ms PCM source as an outbound audio track, and writes decoded
inbound audio to a raw PCM file. `aiortc` handles Opus RTP, ICE, DTLS, and SRTP.
The machine-readable v1 contract lives at
[`docs/contracts/webrtc-sidecar-v1.json`](../../docs/contracts/webrtc-sidecar-v1.json)
and can be printed from an installed binary with `voice stream-contract`. It is
also exposed by the sidecar at `GET /contract`.
When the checked-in JSON is not present, the Python sidecar helpers fall back to
`$VOICE_BIN stream-contract` or `voice stream-contract` so a packaged sidecar can
discover the same PCM contract from an installed binary.

This is a spike. It does not call the WhatsApp Graph API, authenticate requests,
run VAD, or manage Hermes sessions. Hermes should still own WhatsApp Cloud
webhooks and Graph actions such as `pre_accept`, `accept`, and `terminate`.
Do not expose the HTTP control API publicly; bind it to localhost or a private
socket and let only the local Hermes process call it. The WebRTC media path may
still need normal outbound ICE/STUN/TURN network access.
The example enforces that by default: `--host` must be a loopback address such
as `127.0.0.1`, `::1`, or `localhost`. Use `--allow-nonlocal` only behind a
trusted local network or private socket boundary.

## Install

```bash
python3 -m venv /tmp/voice-webrtc-venv
/tmp/voice-webrtc-venv/bin/pip install -r examples/webrtc-sidecar/requirements.txt
```

## Run

Start the sidecar:

```bash
/tmp/voice-webrtc-venv/bin/python examples/webrtc-sidecar/sidecar.py \
  --rx-pcm /tmp/voice-webrtc-in.s16le
```

Once Hermes or a local test has created a call session, stream TTS frames into
the sidecar's per-call outbound queue:

```bash
python examples/webrtc-sidecar/post_voice_stream.py local-test \
  "Hello from the WebRTC sidecar."
```

The helper runs `voice stream --raw-output - --sample-rate 48000 --frame-ms 20`
and POSTs each 1920-byte PCM frame to `POST /calls/{call_id}/audio`. Use
`--sidecar-url` when the sidecar is not listening on `http://127.0.0.1:8787`,
and pass normal voice options such as `--voice`, `--speed`, `--markdown`,
`--input-file`, `--sub`, and `--sub-file`.

You can also POST outbound frames yourself. This is the shape Hermes can use
when it receives `stream_speak` frames from the voice daemon:

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

Each 20 ms frame is 1920 bytes before base64 encoding. HTTP input is queued per
`call_id`; `--tx-pcm` remains a process-level fallback source. The per-call
outbound queue is bounded by the contract's `max_outbound_queue_bytes` value;
`POST /calls/{call_id}/audio` returns HTTP 429 when Hermes needs to slow down
instead of allowing unbounded TTS buffering. If both sources are idle, the
sidecar sends silence. That is useful for checking SDP, ICE, and call timing
before TTS is wired in.

For FIFO-based smoke tests, start the sidecar with `--tx-pcm`:

```bash
mkfifo /tmp/voice-webrtc-out.s16le
/tmp/voice-webrtc-venv/bin/python examples/webrtc-sidecar/sidecar.py \
  --tx-pcm /tmp/voice-webrtc-out.s16le \
  --rx-pcm /tmp/voice-webrtc-in.s16le
voice stream --sample-rate 48000 --frame-ms 20 \
  --raw-output /tmp/voice-webrtc-out.s16le \
  "Hello from the WebRTC sidecar."
```

## SDP API

Check process health and the fixed local audio contract:

```bash
curl -sS http://127.0.0.1:8787/contract
curl -sS http://127.0.0.1:8787/health
voice stream-contract
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

Drain decoded inbound PCM for a live session:

```bash
curl -sS 'http://127.0.0.1:8787/calls/local-test/audio?wait_ms=500'
```

For raw PCM output that can be piped into STT tooling:

```bash
python examples/webrtc-sidecar/drain_sidecar_audio.py local-test \
  --duration-ms 5000 \
  --stop-after-empty 3 \
  --output /tmp/voice-webrtc-in.s16le
```

By default the drain helper requests the contract's `default_drain_bytes`
window, currently 96000 bytes or about one second of mono 48 kHz s16le audio.
Pass `--max-bytes 1920` for exact frame-by-frame polling.

For a minimal live-call echo bot, drain inbound PCM and queue it back to the
same sidecar call:

```bash
python examples/webrtc-sidecar/echo_sidecar_audio.py local-test \
  --stop-after-empty 10
```

This proves the sidecar's local media bridge before adding STT, Hermes turns,
or `voice stream` TTS. It exits with code 75 if the outbound queue is full, so
Hermes can treat that as retryable audio backpressure.

Queue outbound PCM for a live session:

```bash
curl -sS -X POST http://127.0.0.1:8787/calls/local-test/audio \
  -H 'content-type: application/json' \
  -d '{"sample_rate":48000,"channels":1,"frame_ms":20,"encoding":"pcm_s16le","pcm_s16le_base64":"AAAAAA=="}'
```

Response:

```json
{
  "call_id": "local-test",
  "accepted_bytes": 1920,
  "queued_tx_bytes": 1920,
  "max_tx_queue_bytes": 960000,
  "audio": {
    "sample_rate": 48000,
    "channels": 1,
    "frame_ms": 20,
    "encoding": "pcm_s16le"
  }
}
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

Inbound decoded PCM is kept in a bounded per-call queue and can also be mirrored
to a raw signed 16-bit little-endian mono sink with `--rx-pcm`. A later bridge
layer can drain or read that stream, segment it with VAD, and submit each segment
directly to the daemon STT stream contract:

```bash
voice stream-transcribe \
  --raw-input /tmp/voice-webrtc-in.s16le \
  --sample-rate 48000 \
  --frame-ms 20
```

Use `--raw-input -` when piping decoded PCM from another process:

```bash
python examples/webrtc-sidecar/drain_sidecar_audio.py local-test \
  --duration-ms 5000 \
  --stop-after-empty 3 \
  --quiet \
| voice stream-transcribe --raw-input - --sample-rate 48000 --frame-ms 20
```

## Tests

The sidecar tests are optional because they need the Python WebRTC dependency
set:

```bash
python3 -m venv /tmp/voice-webrtc-venv
/tmp/voice-webrtc-venv/bin/pip install -r examples/webrtc-sidecar/requirements.txt pytest
/tmp/voice-webrtc-venv/bin/python examples/webrtc-sidecar/loopback_smoke.py
/tmp/voice-webrtc-venv/bin/python examples/webrtc-sidecar/voice_stream_loopback_smoke.py --voice-bin /path/to/voice
/tmp/voice-webrtc-venv/bin/python -m pytest -q examples/webrtc-sidecar/test_sidecar.py
/tmp/voice-webrtc-venv/bin/python -m pytest -q examples/webrtc-sidecar/test_post_voice_stream.py
/tmp/voice-webrtc-venv/bin/python -m pytest -q examples/webrtc-sidecar/test_drain_sidecar_audio.py
/tmp/voice-webrtc-venv/bin/python -m pytest -q examples/webrtc-sidecar/test_echo_sidecar_audio.py
```

`loopback_smoke.py` starts the sidecar in-process, completes a local SDP
offer/answer, verifies that HTTP-queued PCM reaches a WebRTC audio track, and
verifies that inbound WebRTC audio can be drained back as local PCM. It does
not contact WhatsApp or the Graph API.

`voice_stream_loopback_smoke.py` runs the real `voice stream` command through
`post_voice_stream.py`, queues those PCM frames into the sidecar, and waits for
non-silent audio at a local WebRTC peer. It also does not contact WhatsApp or
the Graph API.
