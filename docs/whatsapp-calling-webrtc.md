# WhatsApp Calling / WebRTC Architecture

Status: architecture and spike document, current as of 2026-06-14.

This document describes how `voice` should fit into Hermes + WhatsApp beyond
file-based voice notes. The short version:

- WhatsApp voice notes are a file problem. `voice` should emit
  `audio/ogg; codecs=opus` directly, and Hermes should send that file with
  `ptt: true` or upload it to the WhatsApp Cloud API.
- WhatsApp live calls are a WebRTC media problem. `voice` should not own Meta
  signaling, ICE, DTLS, SRTP, or RTP. It should own local speech synthesis,
  transcription, and a stable PCM frame contract.
- A small sidecar should bridge WhatsApp Cloud Calling SDP/webhooks to `voice`
  PCM streams.

## References

- Meta user-initiated calls:
  https://developers.facebook.com/documentation/business-messaging/whatsapp/calling/user-initiated-calls/
- Meta business-initiated calls:
  https://developers.facebook.com/documentation/business-messaging/whatsapp/calling/business-initiated-calls
- Meta calling API/webhook reference:
  https://developers.facebook.com/documentation/business-messaging/whatsapp/calling/reference
- Pipecat WhatsApp Calling + SmallWebRTC example:
  https://docs.pipecat.ai/pipecat/features/whatsapp
- 360dialog calling overview:
  https://docs.360dialog.com/docs/messaging/calling

## Existing Surfaces

`voice` now exposes three relevant surfaces:

- Completed files: `voice say --format ogg-opus -o reply.ogg "hello"`
  produces WhatsApp-ready Ogg/Opus.
- Daemon completed files: `synthesize` can write `.wav`, `.ogg`, or `.opus`
  using the same format resolver.
- Streaming frames: `stream_speak` emits ordered signed 16-bit little-endian
  mono PCM frames. Use `sample_rate: 48000` and `frame_ms: 20` for WebRTC.

The streaming frame contract is documented in [streaming.md](streaming.md).

## Deployment Split

### Voice Notes

Use this path for Hermes replies over the current local WhatsApp bridge and for
Cloud API media uploads:

```text
Hermes text response
  -> TTS command provider
  -> voice say --format ogg-opus -o reply.ogg
  -> WhatsApp bridge / Cloud media upload
  -> ptt voice note
```

No PCM stream is needed. The Ogg container is useful here because WhatsApp
expects an uploaded media file.

### Live Calls

Use this path for real-time WhatsApp Calling:

```text
WhatsApp user
  <-> Meta RTC infrastructure
  <-> WebRTC sidecar
  <-> voice daemon PCM frames
  <-> Hermes agent loop
```

The Ogg container is not useful in the live path. The sidecar should accept and
produce RTP media through WebRTC, while `voice` stays on raw PCM frames locally.

## Proposed Components

### Hermes WhatsApp Cloud Adapter

Owns Cloud API/webhook concerns:

- receive call webhooks with `call_id` and SDP
- call Graph API actions: `pre_accept`, `accept`, `reject`, `terminate`
- track call lifecycle and map calls to Hermes sessions
- route transcripts and spoken responses into the existing agent loop

### WebRTC Sidecar

Owns real-time media concerns:

- create a peer connection from Meta's remote SDP
- produce the SDP answer used for `pre_accept` and `accept`
- complete ICE/DTLS/SRTP negotiation
- receive Opus RTP from WhatsApp and expose decoded PCM frames locally
- accept local PCM frames from `voice` and send Opus RTP back to WhatsApp
- terminate quickly and cleanly when the Graph lifecycle ends

The repository includes a first Python spike in
[`examples/webrtc-sidecar`](../examples/webrtc-sidecar/). It exposes the local
SDP/PCM boundary described below, sends silence until outbound PCM is available,
and lets `aiortc` handle Opus RTP, ICE, DTLS, and SRTP.

### `voice` Daemon

Owns speech model concerns:

- TTS: emit 48 kHz 20 ms PCM frames from `stream_speak`
- STT: consume 16 kHz or 48 kHz PCM frames through `stream_transcribe`
- cancellation: stop current TTS on barge-in or call termination
- backpressure: avoid unbounded audio buffering

For local smoke tests, decoded sidecar PCM can go straight into the CLI wrapper
for the daemon contract:

```bash
voice stream-transcribe --raw-input /tmp/voice-webrtc-in.s16le --sample-rate 48000 --frame-ms 20
```

## Inbound Call Flow

1. WhatsApp sends a `connect` webhook containing `call_id` and an SDP offer.
2. Hermes creates a WebRTC sidecar session for that `call_id`.
3. The sidecar creates a peer connection, attaches inbound and outbound audio
   tracks, sets the remote SDP, and creates a local SDP answer.
4. Hermes calls the Graph API with `action: pre_accept` and the SDP answer.
5. The sidecar waits until its media path is ready enough to send/receive audio.
6. Hermes calls the Graph API with `action: accept` and the same SDP answer.
7. The sidecar starts forwarding inbound PCM to STT and outbound PCM from TTS.
8. Either side can terminate; Hermes calls `terminate` if it ends locally.

The important timing rule is that `accept` should not happen before the media
sender is ready. Meta's troubleshooting docs call out media-flow timing as a
common failure mode, and their API rejects mismatched SDP answers between
`pre_accept` and `accept`.

## Sidecar API Sketch

The exact transport can be Unix socket, local HTTP, or WebSocket. The first
implementation should favor debuggability over abstraction. The canonical v1
PCM and endpoint contract is machine-readable at
[`docs/contracts/webrtc-sidecar-v1.json`](contracts/webrtc-sidecar-v1.json), and
installed `voice` binaries print the same object with `voice stream-contract`.
The Python sidecar exposes the same object at `GET /contract`. The contract
defines the fixed PCM audio shape plus the `/offer`, call-state, audio send,
audio drain, close, and error payloads Hermes needs to drive WhatsApp Calling
without hard-coding sidecar response shapes.

The sidecar HTTP API is a local control plane, not a public web API. It should
bind to localhost or a private socket, with Hermes as the caller. The Python
spike rejects non-loopback `--host` values unless `--allow-nonlocal` is passed
explicitly. Only the WebRTC media negotiation needs normal network access for
ICE, DTLS, SRTP, and possibly TURN.

```http
POST /offer
{
  "call_id": "wamid-call-id",
  "type": "offer",
  "sdp": "v=0..."
}
```

Response:

```json
{
  "call_id": "wamid-call-id",
  "type": "answer",
  "sdp": "v=0...",
  "audio": {
    "sample_rate": 48000,
    "channels": 1,
    "frame_ms": 20,
    "encoding": "pcm_s16le"
  },
  "state": {
    "call_id": "wamid-call-id",
    "closed": false,
    "connection_state": "new",
    "ice_connection_state": "new",
    "ice_gathering_state": "complete",
    "signaling_state": "stable"
  }
}
```

Debug a live session with `GET /calls/{call_id}`. Drain decoded inbound audio
with `GET /calls/{call_id}/audio`. Queue outbound audio for the WebRTC track
with `POST /calls/{call_id}/audio`. Terminate a local session with
`POST /calls/{call_id}/close`. The outbound queue is deliberately bounded;
`POST /calls/{call_id}/audio` returns HTTP 429 when the sidecar is already
holding `max_outbound_queue_bytes` for that call, which lets Hermes pause or
cancel TTS instead of building an unbounded latency backlog.

Runtime events:

```json
{"type": "connected", "call_id": "..."}
{"type": "inbound_audio", "sequence": 42, "pcm_s16le_base64": "..."}
{"type": "dtmf", "digit": "1"}
{"type": "ended", "reason": "remote_hangup"}
```

Inbound audio:

```http
GET /calls/{call_id}/audio?max_bytes=1920&wait_ms=500
```

Response:

```json
{
  "call_id": "wamid-call-id",
  "returned_bytes": 1920,
  "queued_rx_bytes": 3840,
  "pcm_s16le_base64": "...",
  "audio": {
    "sample_rate": 48000,
    "channels": 1,
    "frame_ms": 20,
    "encoding": "pcm_s16le"
  }
}
```

`wait_ms` is optional, capped by the sidecar, and lets Hermes long-poll for the
next decoded PCM chunk instead of running a hot polling loop.

Outbound audio:

```http
POST /calls/{call_id}/audio
```

Request:

```json
{
  "sequence": 17,
  "sample_rate": 48000,
  "channels": 1,
  "frame_ms": 20,
  "encoding": "pcm_s16le",
  "pcm_s16le_base64": "..."
}
```

Response:

```json
{
  "call_id": "wamid-call-id",
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

Each 20 ms frame is 1920 bytes: 960 signed 16-bit little-endian mono samples at
48 kHz. This mirrors the current `voice` daemon stream shape closely enough
that the bridge can be mostly mechanical: Hermes can receive `stream_speak` raw
frames, base64-encode each frame, and POST them to the sidecar while the WebRTC
track sends per-call queued audio or silence.

## Implementation Options

### Python `aiortc`

Best first spike if the goal is fast integration with Hermes:

- easy to run next to Hermes
- known examples exist for WebRTC bots
- simpler SDP and media debugging than Rust
- enough performance for a first single-call bot

Risk: long-term CPU overhead and Python dependency surface.

### Pipecat / SmallWebRTC

Best reference implementation path:

- already documents WhatsApp Calling integration
- gives a working transport abstraction for AI voice bots
- useful to compare call lifecycle and timeout handling

Risk: may pull the architecture toward Pipecat's frame model instead of the
small `voice` daemon contract.

### Rust `webrtc-rs`

Best long-term sidecar if we want one Rust deployment:

- aligns with `voice` runtime and Tokio
- avoids Python media-loop overhead
- can share typed frame structs with `voice-protocol`

Risk: more time spent on WebRTC plumbing before validating Meta call behavior.

Recommendation: spike in Python or Pipecat first, keep the local audio boundary
as 48 kHz 20 ms PCM, then port the sidecar to Rust only after the signaling and
timing behavior is proven. The local `examples/webrtc-sidecar` spike follows
that path.

## PR Sequence

1. Keep direct Ogg/Opus file output in `voice` and Hermes. Done for `voice`;
   Hermes wiring is staged separately.
2. Add this architecture document and track open questions. Done.
3. Add a `voice` streaming STT input API that accepts fixed PCM frames. Done.
4. Prototype a sidecar that accepts a synthetic SDP offer and round-trips PCM.
   The initial `examples/webrtc-sidecar` artifact covers the SDP answer,
   outbound PCM-to-Opus/WebRTC track, inbound decoded PCM sink, and local HTTP
   send/drain endpoints. Done as a spike.
5. Wire Hermes WhatsApp Cloud `connect` webhooks to the sidecar.
6. Build an inbound-call echo bot: WhatsApp audio in, same audio out. The
   local `examples/webrtc-sidecar/echo_sidecar_audio.py` helper now covers the
   sidecar-local drain-and-post loop; exercising it against a real WhatsApp
   Cloud call still depends on a calling-enabled number.
7. Replace echo with STT -> Hermes turn -> `stream_speak` TTS.
8. Add interruption/barge-in: inbound voice cancels outbound TTS.

## Open Questions

- Do we have a Cloud API number with calling enabled, or only the local Baileys
  bridge? The local bridge is enough for voice notes, but not a stable live
  calling target.
- Which sidecar runtime should own the first spike: `aiortc`, Pipecat, Node, or
  Rust `webrtc-rs`?
- Should `stream_transcribe` grow partial transcript events, or should the first
  sidecar keep one transcript per completed utterance?
- How much silence should the sidecar send before the agent's first TTS frame
  is ready? WebRTC calls should have media flowing immediately after accept.
- What barge-in policy should Hermes use: immediate cancel on VAD, cancel after
  partial transcript, or full-duplex overlap?
