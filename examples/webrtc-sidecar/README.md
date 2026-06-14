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

For a one-off development run:

```bash
python3 -m venv /tmp/voice-webrtc-venv
/tmp/voice-webrtc-venv/bin/pip install -r examples/webrtc-sidecar/requirements.txt
```

For a persistent local Hermes/WhatsApp Calling host on Linux, install the voice
daemon first, then install the sidecar as a systemd user service:

```bash
voice daemon install
scripts/install_webrtc_sidecar_service.sh --voice-bin "$(command -v voice)"
scripts/verify_webrtc_sidecar_service.py --voice-bin "$(command -v voice)"
```

The installer creates a venv under the XDG data directory, writes
`~/.config/systemd/user/voice-webrtc-sidecar.service`, enables it, and restarts
the service. It pins `VOICE_BIN` in the unit so the sidecar can discover the
same `voice stream-contract` object Hermes expects. Use `--print-unit` to
inspect the generated service without writing files, `--no-start` when another
process will start it, and `--uninstall` to remove the user unit.
The verifier compares the running sidecar's `/contract` response against
`voice stream-contract`, checks `/health`, and confirms the sidecar plus voice
daemon user services are active. Pass `--skip-systemd` for a contract-only
check against a manually run sidecar.

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
For barge-in or cancellation, `POST /calls/{call_id}/audio/clear` drops queued
per-call outbound PCM while leaving the WebRTC session alive.

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

The contract includes `voice_surfaces`, a machine-readable map from integration
mode to command: completed Ogg/Opus voice notes, streamed Ogg/Opus files, raw
outbound PCM, raw inbound PCM, and file-based stream-transcribe smokes.
`post_voice_stream.py` validates the raw PCM surface metadata when it is
present so frame-size or transport drift fails before a live call. It posts
frames on the contract's `frame_ms` cadence, so a fast `voice stream` process
does not build a large outbound playback queue in the sidecar.

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
  "accepted_ms": 20,
  "queued_tx_bytes": 1920,
  "queued_tx_ms": 20,
  "max_tx_queue_bytes": 960000,
  "max_tx_queue_ms": 10000,
  "audio": {
    "sample_rate": 48000,
    "channels": 1,
    "frame_ms": 20,
    "encoding": "pcm_s16le"
  }
}
```

Clear queued outbound PCM without closing the call:

```bash
curl -sS -X POST http://127.0.0.1:8787/calls/local-test/audio/clear
```

Response:

```json
{
  "call_id": "local-test",
  "dropped_tx_bytes": 3840,
  "dropped_tx_ms": 40,
  "queued_tx_bytes": 0,
  "queued_tx_ms": 0,
  "max_tx_queue_bytes": 960000,
  "max_tx_queue_ms": 10000,
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
silence until real TTS PCM arrives. The `/offer` response and `GET
/calls/{call_id}` status include `ready_for_accept` plus per-check `readiness`
booleans. Hermes should treat `ready_for_accept: true` as the local gate before
it sends the Graph `accept` action; the remote WebRTC connection state can
still be `new` until Meta applies the accepted SDP answer.

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
/tmp/voice-webrtc-venv/bin/python examples/webrtc-sidecar/stream_transcribe_loopback_smoke.py --voice-bin /path/to/voice
/tmp/voice-webrtc-venv/bin/python examples/webrtc-sidecar/full_duplex_loopback_smoke.py --voice-bin /path/to/voice
/tmp/voice-webrtc-venv/bin/python -m pytest -q examples/webrtc-sidecar/test_sidecar.py
/tmp/voice-webrtc-venv/bin/python -m pytest -q examples/webrtc-sidecar/test_post_voice_stream.py
/tmp/voice-webrtc-venv/bin/python -m pytest -q examples/webrtc-sidecar/test_drain_sidecar_audio.py
/tmp/voice-webrtc-venv/bin/python -m pytest -q examples/webrtc-sidecar/test_echo_sidecar_audio.py
/tmp/voice-webrtc-venv/bin/python -m pytest -q examples/webrtc-sidecar/test_stream_transcribe_loopback_smoke.py
/tmp/voice-webrtc-venv/bin/python -m pytest -q examples/webrtc-sidecar/test_full_duplex_loopback_smoke.py
```

`loopback_smoke.py` starts the sidecar in-process, completes a local SDP
offer/answer, verifies that HTTP-queued PCM reaches a WebRTC audio track, and
verifies that inbound WebRTC audio can be drained back as local PCM. It does
not contact WhatsApp or the Graph API.

`voice_stream_loopback_smoke.py` runs the real `voice stream` command through
`post_voice_stream.py`, queues those PCM frames into the sidecar, and waits for
non-silent audio at a local WebRTC peer. It also does not contact WhatsApp or
the Graph API.

`stream_transcribe_loopback_smoke.py` sends real `voice stream` PCM through a
local WebRTC sender into the sidecar, drains the sidecar-decoded PCM, and runs
`voice stream-transcribe` on that decoded audio. It validates the inbound
WebRTC-to-STT path without WhatsApp or the Graph API.

`full_duplex_loopback_smoke.py` exercises both directions on one sidecar call:
local WebRTC audio drains through `voice stream-transcribe`, while
`post_voice_stream.py` queues outbound `voice stream` PCM back to the same
WebRTC peer. By default it starts an in-process sidecar; pass `--sidecar-url`
to exercise an already running sidecar service such as
`http://127.0.0.1:8787`. It is the closest local smoke to a single WhatsApp
call turn. It
also queues a small outbound PCM probe on the live call and calls
`POST /calls/{call_id}/audio/clear`, so the same smoke covers the local
barge-in/cancellation primitive. It closes the sidecar call and verifies the
session is removed from both `GET /calls/{call_id}` and `/health`. Pass
`--skip-clear-audio-smoke` when checking only media round-trip behavior. The
smoke fails when the sidecar still has more than one second of outbound audio
queued at the end of the turn; use `--max-queued-tx-ms` to tighten or loosen
that local latency budget. When available, it uses the sidecar-reported
`queued_tx_ms` field; older sidecars fall back to deriving the duration from
`queued_tx_bytes` and the audio contract.
