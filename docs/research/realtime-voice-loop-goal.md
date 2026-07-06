# Goal: Local Realtime Voice Loop

Date: 2026-07-06

Branch: `quod/realtime-local-voice-loop`

## Outcome

Build and verify a fully local realtime voice loop for this repo:

```text
human/mic or PCM fixture -> VAD/turn detection -> STT -> assistant text stream
-> streaming TTS -> speaker/PCM frames, with barge-in cancellation
```

The first usable target is a local session surface that can run without cloud
services, expose ordered realtime events, and report latency/performance metrics
for Apple Silicon Metal execution. It should be suitable for a human talking to
an agent, but the first verifier may use prerecorded PCM fixtures so the core
loop can be tested repeatably.

## Baseline

Observed on 2026-07-06 from `origin/main`:

- `voice-stream` defines ordered `tts.started`, `tts.audio`, and terminal TTS
  events plus fixed 20 ms PCM frame packetization.
- The daemon already implements `stream_speak` and can emit streamed TTS frames.
- The daemon already implements `stream_transcribe`, accepting inbound PCM
  frames and returning terminal `stt.transcribed` after `stt.end`.
- STT is final-segment only today; inbound streaming does not yet emit partial
  transcripts.
- `voice converse` is speak-then-listen, not a full-duplex session.
- Queue cancellation exists, and stream TTS checks cancellation cooperatively.
- The local WebRTC sidecar contract already uses 48 kHz, mono, 20 ms,
  `pcm_s16le` frames plus an audio-clear primitive useful for barge-in.
- Prior repo memory says voice-only UX work should treat daemon shutdown as
  reversible, keep pause/stop-listening controls first-class, and verify browser
  plus daemon behavior together when queue/playback boundaries are involved.

## User Requirements

- Prioritize a seamless back-and-forth conversation, not button-driven dictation.
- Push local Apple Silicon performance as far as practical using Candle/Metal
  paths for STT/TTS.
- Keep the first milestone fully local. External LLM services, paid APIs, or
  public network dependencies are optional later backends, not the primary gate.
- Treat Gemma 4 as a future local reasoning backend, not a blocker for the first
  realtime loop.
- The user likes voice UIs that expose a movable tolerance control for
  speech-to-text / VAD sensitivity, but this is optional polish. The goal should
  preserve a tunable turn-detection threshold in the API and metrics even if no
  visible slider is built in the first pass.

## Constraints

- Work from `origin/main` on `quod/realtime-local-voice-loop`.
- Prefer the existing daemon, protocol, stream, CLI, and audio crates over a new
  parallel runtime.
- Preserve existing `voice stream`, `voice stream-transcribe`, `voice listen`,
  `voice converse`, MCP, and daemon behavior unless deliberately extending them.
- Do not weaken existing stream contracts or tests to make the new surface pass.
- Do not rely on cloud services for the primary verifier.
- Do not require real microphone input for automated tests.
- Do not start, install, unload, or alter a global LaunchAgent daemon without
  explicit user approval. Running a repo-local foreground daemon for verification
  is allowed if it does not mutate global service registration.

## Non-goals

- Do not implement native Gemma 4 as the first realtime loop backend.
- Do not chase perfect partial ASR before the final-only local loop works.
- Do not build a polished native UI before the daemon/protocol loop is verified.
- Do not replace Kokoro, Voxtral, Whisper, or the existing daemon queue as part
  of the first milestone.
- Do not make the visual VAD/tolerance slider a completion blocker.

## Proposed Shape

Add a realtime layer over the existing daemon contracts. The likely first shape
is a `voice realtime-smoke` or `voice realtime` CLI plus reusable library module
that emits JSON events compatible with a future WebSocket/session API:

- `session.started`
- `input_audio_buffer.appended`
- `input_audio_buffer.committed`
- `input_audio_buffer.speech_started`
- `conversation.item.input_audio_transcription.completed`
- `response.output_text.delta`
- `response.output_audio.delta`
- `response.output_audio.done`
- `response.cancelled`
- `response.done`
- `session.metrics`

If a UI surface is added during this goal, it should make listening state,
assistant speaking state, cancellation, and turn-detection tolerance visible.
The draggable tolerance-dot style shown in the user's reference screenshots is a
good direction, but an API/config-level tolerance control is sufficient for the
first verifier.

The assistant text backend for the first automated verifier may be a local
deterministic echo/script backend, but the code must be shaped so a local LLM
backend can stream text deltas later. The deterministic backend verifies the
audio loop and event choreography only; it must not be presented as proof of
local reasoning quality.

## Primary Verifier

A repeatable local smoke command must pass from a clean checkout on this branch:

```bash
cargo run -p voice -- realtime-smoke \
  --input-audio <fixture-or-generated-wav> \
  --assistant-backend echo \
  --tts-engine kokoro \
  --sample-rate 48000 \
  --frame-ms 20 \
  --json
```

It must prove all of the following without cloud services:

- input audio is chunked into 48 kHz, mono, 20 ms PCM frames;
- the session commits an utterance and returns a non-empty STT transcript for a
  speech fixture, or an explicit empty/no-speech result for a silence fixture;
- assistant text is emitted as ordered text deltas;
- TTS emits ordered audio deltas before `response.done`;
- output audio frame metadata is consistent: sample rate, frame duration,
  monotonic sequence numbers, offsets, and terminal frame count;
- the command emits `session.metrics` with at least:
  - `stt_audio_ms`
  - `stt_elapsed_ms`
  - `assistant_first_delta_ms`
  - `tts_first_audio_ms`
  - `response_total_ms`
  - `output_audio_ms`
  - `underrun_count`
  - selected STT/TTS engine names
  - selected device/backend when available, including Metal/CPU;
- the process exits nonzero on malformed event order, missing terminal events,
  frame-shape mismatch, empty transcript for the speech fixture, or TTS without
  audio frames.

## Barge-in Verifier

A deterministic barge-in test must pass:

```bash
cargo run -p voice -- realtime-smoke \
  --input-audio <first-speech-fixture> \
  --barge-in-audio <second-speech-fixture> \
  --barge-in-at-ms <during-assistant-audio> \
  --assistant-backend echo \
  --tts-engine kokoro \
  --json
```

It must prove:

- second input speech triggers active response cancellation;
- queued outbound audio is cleared or marked truncated;
- no further audio frames from the cancelled response are emitted after the
  cancellation boundary;
- a second user turn can be transcribed and answered in the same session;
- metrics report cancellation latency and dropped output audio duration.

## Supporting Checks

- `cargo test --workspace` or a documented narrower set if the full workspace is
  too slow for the current turn.
- Focused tests for realtime event ordering, frame validation, cancellation, and
  metrics.
- Existing stream contract checks still pass:
  - `voice stream-contract`
  - `voice stream --json "Hello from the stream"`
  - `voice stream-transcribe --json <fixture>`
- A manual hardware run is attempted and reported separately:

```bash
cargo run -p voice -- realtime --mic --speaker --turns 2 --json
```

If microphone, speaker, TCC, or local hardware constraints prevent the manual
run, record the exact blocker and keep automated fixture verification separate.

## Performance Targets

Initial targets are deliberately empirical and may be tightened only with
evidence:

- warm-model `tts_first_audio_ms` should be low enough to feel conversational;
- STT should remain faster than realtime on the default distil-whisper path;
- output streaming should produce enough buffered audio to avoid underruns
  during normal response playback;
- all metrics must clearly identify whether Metal or CPU was used, so Candle
  Metal regressions are visible.

Do not declare completion on performance vibes. Report exact metrics and the
machine/backend used.

## Iteration Loop

1. Inspect the current daemon/protocol/CLI seams for the smallest extension.
2. Add one vertical slice: event schema, session state, fixture input, STT,
   assistant text, TTS, cancellation, or metrics.
3. Run the focused verifier for that slice.
4. Record command output and current next action in this file or a nearby
   worklog section.
5. If a verifier fails, fix the cause or narrow the claim; do not weaken the
   verifier.
6. After the fixture loop passes, try the manual mic/speaker verifier and record
   what was and was not proven.

## Anti-cheating Rules

- Do not replace real STT/TTS with mocks in the primary verifier.
- Do not accept event streams that have no terminal event.
- Do not count TTS completion as realtime streaming unless at least one audio
  delta appears before `response.done`.
- Do not hide cancellation failures by dropping all events silently.
- Do not call the deterministic assistant backend a local LLM.
- Do not treat manual mic failure as automated loop failure when fixture
  verification passed; report the two surfaces separately.
- Do not change frame size, sample rate, or contract defaults without updating
  `voice-stream`, docs, tests, and the verifier together.

## Approval Gates

Ask before:

- installing, uninstalling, unloading, or modifying a global LaunchAgent daemon;
- using paid, cloud, or external network LLM/STT/TTS APIs;
- downloading multi-GB model files beyond already-approved caches;
- pushing branches, opening PRs, or publishing packages;
- recording or saving real microphone audio beyond transient verifier use.

## Blocker Standard

The goal is blocked only if the same external blocker recurs for at least three
goal turns and no meaningful local progress remains. Examples:

- macOS TCC denies microphone access and the manual verifier cannot run;
- required local model files are unavailable and download approval is needed;
- hardware audio output is unavailable.

Difficulty, slow model startup, missing partial ASR, or imperfect latency is not
a blocker while fixture-based progress is still possible.

## Completion Proof

Before marking the goal complete, provide:

- final changed files and purpose;
- exact automated verifier commands and pass/fail summaries;
- exact cargo test commands and pass/fail summaries;
- the final event sequence emitted by the fixture smoke;
- metrics from at least one successful warm local run;
- barge-in verifier evidence;
- explicit statement of which surfaces are supported:
  - fixture/offline realtime loop;
  - live mic/speaker loop;
  - browser/WebSocket loop, if implemented;
  - local LLM backend, if implemented;
- remaining risks, especially latency, partial STT, TCC, model warmup, and
  Metal/CPU fallback behavior.

## Current Next Step

Continue from the first fixture-based smoke slice. The next implementation
slice is deterministic barge-in: inject a second audio fixture while assistant
audio is streaming, cancel the active response, prove no further cancelled
audio frames are emitted, then answer the second turn.

## Worklog

### 2026-07-06: First Realtime Smoke Slice

Implemented `voice realtime-smoke` inside `voice-cli` as the smallest vertical
slice over existing daemon APIs. It:

- accepts a file fixture with `--input-audio`;
- resamples/chunks input into local 48 kHz, mono, 20 ms PCM frames;
- emits ordered realtime-style JSON events;
- reports fixture VAD metrics through `--vad-threshold`;
- uses daemon `stream_transcribe` for real STT;
- uses a deterministic `--assistant-backend echo` text stream;
- uses daemon `stream_speak` for real streamed TTS;
- validates output frame sample rate, frame duration, mono channel count,
  `pcm_s16le` encoding, sequence numbers, offsets, frame payload length, and
  terminal frame/sample counts;
- exits nonzero on missing terminal events, STT terminal errors, empty speech
  transcripts, malformed output frames, or TTS without audio.

Verification run:

```bash
cargo fmt --check -p voice
cargo test -p voice realtime -- --nocapture
cargo check -p voice
git diff --check
```

Result: all passed. Focused realtime tests covered CLI parsing, fixture
peak/RMS speech detection, streamed text delta reconstruction, output frame
shape validation, and required event sequencing. The primary end-to-end
`cargo run -p voice -- realtime-smoke ...` command still needs a daemon-backed
audio fixture run before this goal can be called complete.

### Reference Clones

- Silero VAD: `/Users/kylekelley/code/src/github.com/snakers4/silero-vad`
  at `b163605b3f44c3aadf28f97b125a2f7c461e9a7f`.
- NVIDIA Parakeet HF model repo:
  `/Users/kylekelley/code/src/huggingface.co/nvidia/parakeet-tdt-1.1b`
  at `53276c6469d1f17a1352e30c4d11be3d0d7e9575`.

Parakeet was cloned with LFS smudge disabled; `parakeet-tdt-1.1b.nemo` is
currently a Git LFS pointer in that checkout, not the downloaded model file.
Its README describes a 1.1B English FastConformer-TDT NeMo ASR model that
accepts 16 kHz mono audio. It is useful ASR research, but it is not a near-term
Candle/Metal drop-in without a NeMo/FastConformer/TDT port.

### VAD Comparison

Current repo VAD is an amplitude gate:

- mic paths calibrate a peak noise floor, then use
  `max(noise_floor * noise_multiplier, silence_threshold)`;
- recording state is `waiting -> speech_started -> silence_timeout`;
- continuous mode force-splits long segments and trims silence with an RMS
  pass;
- the new fixture smoke uses simple peak/RMS thresholding only to emit
  `input_audio_buffer.speech_started` and metrics.

Silero VAD is materially stronger for realtime turn detection:

- model output is speech probability per short chunk, not raw amplitude;
- default threshold is 0.5 with a lower negative threshold for hysteresis;
- it enforces minimum speech duration, minimum trailing silence, and speech
  padding;
- it has a streaming `VADIterator`;
- bundled artifacts include ONNX, TorchScript, safetensors, and OpenVINO
  variants;
- project docs claim sub-millisecond processing per 30+ ms chunk on one CPU
  thread, with ONNX sometimes faster.

Conclusion: keep the current energy VAD as a dependency-free fallback and test
bootstrap, but use a Silero-compatible neural VAD backend for real barge-in and
noisy-room turn taking. This should start on CPU because VAD cost is tiny
compared with Whisper/Kokoro/Voxtral/LLM inference; Metal effort belongs first
on STT/TTS/LLM paths.

### Gemma 4 Speed Read

The high Apple Silicon Gemma 4 numbers found online are plausible only in the
right configuration. Official Google/Ollama material attributes the recent
speed jump to MTP/speculative decoding, often with quantized MLX paths. That is
not directly comparable to the prior Candle BF16 greedy E2B probe.

For this repo, the voice loop should not wait on Gemma 4. The local reasoning
backend can be added behind the deterministic assistant seam later. Matching
the newest Ollama-class throughput in Candle likely requires separate work:

- MTP drafter loading and speculative decode;
- quantized weight formats compatible with Gemma 4 releases;
- Metal kernels that avoid redundant weight reads during multi-token verify;
- benchmark harnesses that report prefill, decode, accepted draft tokens,
  context length, quantization, and backend.
