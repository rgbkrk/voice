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

If `--tx-pcm` is omitted, the sidecar sends silence. That is useful for checking
SDP, ICE, and call timing before TTS is wired in.

## SDP API

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
to `voice stream-transcribe` or the daemon `stream_transcribe` RPC.
