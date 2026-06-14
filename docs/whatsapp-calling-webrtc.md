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

## Local WhatsApp Runtime Check

The local Hermes bridge uses Baileys multi-file auth, not the WhatsApp Cloud
API. A paired Baileys session is enough for text messages, inbound voice-note
downloads, and outbound native voice notes. It is not proof that the Meta Cloud
webhook or WhatsApp Calling product is configured.

Use the voice-owned verifier to inspect the local bridge without printing auth
keys or session material:

```bash
scripts/verify_whatsapp_bridge_runtime.py --hermes-home ~/.hermes
```

The verifier checks:

- the bridge health endpoint is connected
- `~/.hermes/whatsapp/session/creds.json` has a paired WhatsApp identity
- the running `whatsapp-bridge/bridge.js` process uses the expected session
- LID-to-phone mapping files match the paired identity
- WhatsApp Cloud and Calling environment keys are present or missing

For a full local release gate, use:

```bash
scripts/verify_local_hermes_voice_stack.sh --hermes-home ~/.hermes
```

That command verifies installed `voice`, daemon-aware CLI/MCP behavior,
WhatsApp-ready Ogg/Opus output, the Baileys bridge identity, the Hermes gateway
voice-stream service, and the WebRTC sidecar contract. Add
`--require-whatsapp-cloud` or `--require-whatsapp-calling` only when the host is
expected to have real Meta Cloud credentials. Once those credentials are on
disk, add `--check-whatsapp-cloud-api` to make an authenticated Graph API
phone-number request and confirm the configured `WHATSAPP_CLOUD_PHONE_NUMBER_ID`
is reachable without printing token or phone-number values.

The same stack gate can run the categorized alpha report as an explicit opt-in:

```bash
scripts/verify_local_hermes_voice_stack.sh \
  --hermes-home ~/.hermes \
  --whatsapp-alpha-profile cached-receive \
  --whatsapp-alpha-json-output ./whatsapp-alpha.json
```

Use `--whatsapp-alpha-profile send` only when it is acceptable to post a real
voice note through the paired bridge. Prefer
`--whatsapp-alpha-profile attended-cache-receive` for an attended receive test
while Hermes is already running; it watches `~/.hermes/audio_cache` for a fresh
`aud_*` artifact and does not drain the bridge queue. Use
`--whatsapp-alpha-profile attended-send-receive` only when the verifier itself
should poll and drain the bridge `/messages` queue. The stack gate also accepts
`--whatsapp-alpha-chat-id`, `--whatsapp-alpha-wait-audio-cache-seconds`, and
`--whatsapp-alpha-wait-inbound-seconds` so attended tests do not need to drop
down to the lower-level alpha script. When `--whatsapp-alpha-json-output` is
set, the stack gate runs the alpha profile with `--json` and saves that
structured report for another agent or runbook step to consume.

Cached receive profiles also pass the newest cached inbound `aud_*` file to
the Hermes config verifier, so the configured STT command provider is exercised
against bridge-downloaded audio when a cache file is available. Add
`--skip-hermes-stt-smoke` when that part should remain shape-only.

For a categorized alpha-readiness report, use:

```bash
scripts/verify_whatsapp_alpha_readiness.py --hermes-home ~/.hermes
```

That report runs the voice/Hermes/bridge/sidecar checks as separate components
and groups failures as voice runtime, Hermes runtime/config, bridge pairing,
voice-note, live-call local sidecar, or external Meta setup. Use
`--require-whatsapp-calling` when a host is expected to be ready for real Cloud
Calling; otherwise missing Meta credentials are reported as external setup
still required, not as a local Baileys voice-note failure. Add
`--check-whatsapp-cloud-api` when credentials are expected to work and the
report should prove the configured Cloud phone-number node is reachable. The
named profiles are `unattended`, `cached-receive`, `send`,
`attended-cache-receive`, and `attended-send-receive`. Use `cached-receive`
when the report should also replay a cached inbound WhatsApp voice note through
`voice stream-transcribe`:

```bash
scripts/verify_whatsapp_alpha_readiness.py \
  --hermes-home ~/.hermes \
  --profile cached-receive
```

The JSON report includes `pending_gates` for checks that are intentionally not
part of unattended local readiness: fresh attended inbound receive, WhatsApp
Cloud API credentials, and WhatsApp Cloud Calling. A default local pass can
still report `pending_gates.attended_fresh_receive.status=pending_attended`
until someone sends a fresh WhatsApp voice note during the guarded receive
window.

For the external Meta gates, the JSON report includes safe setup handoffs under
`pending_gates.whatsapp_cloud.setup_handoff` and
`pending_gates.whatsapp_cloud_calling.setup_handoff`. These handoffs list the
required key names, which keys are missing, redacted source labels for values
that were found, and the exact follow-up verifier commands. Secret values are
never printed. The Cloud handoff also reports the resolved local webhook
binding (`host`, `port`, `path`, and `api_version`), which values came from env
sources versus Hermes defaults, and malformed webhook keys under `invalid`.
`readiness_summary.next_actions` repeats the same non-secret commands,
missing-key groups, and invalid-key groups so another agent can consume the
report without scraping human output. The human output mirrors the same
information with:

- `whatsapp_cloud_setup`
- `whatsapp_cloud_verify_command`
- `whatsapp_calling_setup`
- `whatsapp_calling_verify_command`
- `whatsapp_calling_complete_command`

When the top-level stack gate is run with `--whatsapp-alpha-json-output`, it
saves the alpha report and then echoes compact `whatsapp_alpha_json_*` summary
lines for the saved artifact. Those lines include the alpha profile, readiness
status, completion state, next action IDs, attended receive status, and
Cloud/Calling missing or invalid key groups.

Use the complete gate only when the local bridge, attended receive, Cloud API,
and Calling setup are all expected to pass:

```bash
scripts/verify_whatsapp_alpha_readiness.py \
  --hermes-home ~/.hermes \
  --profile attended-cache-receive \
  --require-complete
```

Until Meta Cloud/Calling credentials are installed, that command should fail
with the missing external keys instead of reporting a local Baileys or voice
runtime regression.

The same alpha report can also drive the real bridge voice-note operation when
running an attended test. Add `--send-voice-note` to post the generated
Ogg/Opus note to `WHATSAPP_HOME_CHANNEL`, or pass `--voice-note-chat-id` to
override the destination:

```bash
scripts/verify_whatsapp_alpha_readiness.py \
  --hermes-home ~/.hermes \
  --profile send
```

For a full attended send/receive pass while Hermes is already watching the
bridge, use the cache-watching guarded receive profile:

```bash
scripts/verify_whatsapp_alpha_readiness.py \
  --hermes-home ~/.hermes \
  --profile attended-cache-receive
```

That profile sends a real voice note, then watches the Hermes audio cache for a
fresh inbound `aud_*` file without polling the bridge message queue. If Hermes
is not running and the verifier itself must consume bridge events, use the
draining profile instead. The alpha readiness handoff prints both commands with
explicit 60-second wait windows so copied commands and saved JSON artifacts are
self-contained:

```bash
scripts/verify_whatsapp_alpha_readiness.py \
  --hermes-home ~/.hermes \
  --profile attended-send-receive
```

To prove the outbound voice-note path without sending a WhatsApp message, run:

```bash
scripts/verify_whatsapp_voice_note_bridge.py --hermes-home ~/.hermes
```

That generates an Ogg/Opus file with `voice`, checks the bridge's media payload
builder maps it to `audio/ogg; codecs=opus` with `ptt: true`, and stops before
posting. Add `--send` to post the generated file to the configured
`WHATSAPP_HOME_CHANNEL` through the local bridge:

```bash
scripts/verify_whatsapp_voice_note_bridge.py --hermes-home ~/.hermes --send
```

Use attended inbound receive polling only when someone can reply with a real
WhatsApp voice note during the wait window:

```bash
scripts/verify_whatsapp_voice_note_bridge.py \
  --hermes-home ~/.hermes \
  --wait-inbound-seconds 60 \
  --require-inbound-audio \
  --drain-bridge-messages
```

The explicit `--drain-bridge-messages` flag is required because this path polls
the bridge's `GET /messages` endpoint, which consumes queued bridge messages.
Do not use it as a background health check while Hermes is expected to process
the same queue.

If the bridge has already downloaded inbound WhatsApp audio, validate that
receive artifact without draining the live `/messages` queue:

```bash
scripts/verify_whatsapp_inbound_audio_cache.py \
  --hermes-home ~/.hermes \
  --require-cache \
  --run-stt
```

The bridge writes inbound voice/audio media as `aud_*` files under
`~/.hermes/audio_cache` by default. The cache verifier checks those files look
like bridge downloads, validates the audio stream with `ffprobe`, and can replay
one through `voice stream-transcribe --json`. Saved verifier JSON records audio
frames, duration, token count, and transcript length, but redacts the transcript
text. The full stack gate exposes the same receive-side smoke behind an
explicit opt-in:

```bash
scripts/verify_local_hermes_voice_stack.sh \
  --hermes-home ~/.hermes \
  --run-whatsapp-inbound-cache-smoke
```

For attended fresh receive without draining bridge messages, use the cache
watch mode:

```bash
scripts/verify_whatsapp_inbound_audio_cache.py \
  --hermes-home ~/.hermes \
  --wait-fresh-seconds 60 \
  --require-fresh-audio \
  --run-stt
```

Cloud/Calling readiness requires these external Meta settings in addition to
the local sidecar:

- `WHATSAPP_CLOUD_PHONE_NUMBER_ID`
- `WHATSAPP_CLOUD_ACCESS_TOKEN`
- `WHATSAPP_CLOUD_APP_SECRET`
- `WHATSAPP_CLOUD_VERIFY_TOKEN`
- `WHATSAPP_CLOUD_CALLING_SIDECAR_URL`
- `WHATSAPP_CLOUD_CALLING_SIDECAR_TTS_STREAM_COMMAND`

Hermes defaults the local Cloud webhook bind to `0.0.0.0:8090` at
`/whatsapp/webhook` with Graph API version `v20.0` when the matching optional
`WHATSAPP_CLOUD_WEBHOOK_*` keys are not set. The voice verifier reports those
resolved defaults and fails strict Cloud/Calling gates if the configured
webhook port, path, or API version is malformed.

The external Meta setup behind those keys is:

- create or select a WhatsApp Business Platform app and WABA
- attach a phone number that is eligible for WhatsApp Cloud API
- generate a permanent System User access token with WhatsApp permissions
- configure webhook verify token and app secret for signed inbound webhooks
- enable or approve WhatsApp Calling for the Cloud phone number
- route Cloud Calling webhooks to the Hermes WhatsApp Cloud adapter

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
5. Hermes checks the sidecar call state. `ready_for_accept` should be true
   before `accept`; this means the local SDP answer exists, the outbound audio
   track is live, ICE gathering completed, and signaling is stable.
6. Hermes calls the Graph API with `action: accept` and the same SDP answer.
7. The sidecar starts forwarding inbound PCM to STT and outbound PCM from TTS.
8. Either side can terminate; Hermes calls `terminate` if it ends locally.

The important timing rule is that `accept` should not happen before the media
sender is ready. Meta's troubleshooting docs call out media-flow timing as a
common failure mode, and their API rejects mismatched SDP answers between
`pre_accept` and `accept`. `ready_for_accept` is a local readiness signal, not
proof that the remote WhatsApp peer is connected; connection state still
progresses after Meta receives the accepted SDP answer.

## Sidecar API Sketch

The exact transport can be Unix socket, local HTTP, or WebSocket. The first
implementation should favor debuggability over abstraction. The canonical v1
PCM and endpoint contract is machine-readable at
[`docs/contracts/webrtc-sidecar-v1.json`](contracts/webrtc-sidecar-v1.json), and
installed `voice` binaries print the same object with `voice stream-contract`.
The Python sidecar exposes the same object at `GET /contract`. The contract
defines the fixed PCM audio shape plus the `/offer`, call-state, audio send,
audio drain, outbound-audio clear, close, and error payloads Hermes needs to
drive WhatsApp Calling without hard-coding sidecar response shapes. Its
`voice_surfaces` section also maps each local integration mode to the expected
`voice` command: completed Ogg/Opus voice notes, streamed Ogg/Opus files, raw
outbound PCM frames, raw inbound PCM transcription, and file-based inbound
stream smokes.

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
    "ready_for_accept": true,
    "readiness": {
      "not_closed": true,
      "local_sdp_answer": true,
      "signaling_stable": true,
      "ice_gathering_complete": true,
      "outbound_audio_track": true
    },
    "connection_state": "new",
    "ice_connection_state": "new",
    "ice_gathering_state": "complete",
    "signaling_state": "stable"
  }
}
```

Debug a live session with `GET /calls/{call_id}`. Drain decoded inbound audio
with `GET /calls/{call_id}/audio`. Queue outbound audio for the WebRTC track
with `POST /calls/{call_id}/audio`. Drop queued outbound audio with
`POST /calls/{call_id}/audio/clear` when inbound speech should barge in without
tearing down the call. Terminate a local session with `POST
/calls/{call_id}/close`. The outbound queue is deliberately bounded; `POST
/calls/{call_id}/audio` returns HTTP 429 when the sidecar is already holding
`max_outbound_queue_bytes` for that call, which lets Hermes pause or cancel TTS
instead of building an unbounded latency backlog.

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
  "returned_ms": 20,
  "queued_rx_bytes": 3840,
  "queued_rx_ms": 40,
  "max_rx_queue_bytes": 960000,
  "max_rx_queue_ms": 10000,
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

Each 20 ms frame is 1920 bytes: 960 signed 16-bit little-endian mono samples at
48 kHz. This mirrors the current `voice` daemon stream shape closely enough
that the bridge can be mostly mechanical: Hermes can receive `stream_speak` raw
frames, base64-encode each frame, and POST them to the sidecar while the WebRTC
track sends per-call queued audio or silence.

Barge-in / cancellation:

```http
POST /calls/{call_id}/audio/clear
```

Response:

```json
{
  "call_id": "wamid-call-id",
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

Hermes should call this when it decides inbound speech interrupts an in-flight
spoken reply. That clears stale PCM already accepted by the sidecar while the
WebRTC track keeps sending silence until fresh TTS arrives.

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
   send/drain/clear endpoints. The full-duplex smoke also requires the sidecar
   answer state to be `ready_for_accept` before media proceeds. Done as a spike.
5. Wire Hermes WhatsApp Cloud `connect` webhooks to the sidecar.
6. Build an inbound-call echo bot: WhatsApp audio in, same audio out. The
   local `examples/webrtc-sidecar/echo_sidecar_audio.py` helper now covers the
   sidecar-local drain-and-post loop; exercising it against a real WhatsApp
   Cloud call still depends on a calling-enabled number.
7. Replace echo with STT -> Hermes turn -> `stream_speak` TTS.
8. Add interruption/barge-in: inbound voice clears queued outbound sidecar PCM
   and cancels in-flight TTS. The local full-duplex sidecar smoke now exercises
   `/calls/{call_id}/audio/clear`; Hermes still needs the policy that decides
   when to invoke it during real calls.

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
- What barge-in policy should Hermes use: immediate clear/cancel on VAD, cancel
  after partial transcript, or full-duplex overlap?
