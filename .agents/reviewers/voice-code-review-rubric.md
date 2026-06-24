# voice Code Review Rubric

Use this rubric for Pullfrog and other second-pass review runs. It is reviewer
guidance, not general implementation guidance.

## Operating Mode

Review as a senior voice code reviewer. Stay read-only. Inspect the diff first,
then the nearest owning code, tests, `AGENTS.md`, release scripts, docs, and
recent repository patterns when they are relevant. Be terse, adversarial, and
evidence-backed.

Primary goals:

- Find concrete bugs, behavioral regressions, audio or transcript quality
  regressions, model-loading mistakes, platform breakage, security issues,
  concurrency hazards, broken tests, and missing tests that could allow the diff
  to regress.
- Also report repo-invariant architecture/style nits when they protect
  maintainability or user-visible behavior. Examples: changing tensor shapes or
  device placement without checking Kokoro/Whisper expectations, moving
  CPU-only work onto a hot audio path, duplicating stream/daemon protocol
  contracts instead of sharing `voice-stream` or `voice-protocol`, weakening
  zero-network startup for builtin voices/configs, or letting docs/scripts drift
  from the shipped CLI surface.

Suppress comments that are only subjective preference, formatting, naming, or
"could be cleaner" with no concrete invariant, user-visible behavior, runtime
risk, or test gap. Group repeated instances under one finding. Prefer at most 12
findings.

## voice Checklist

- Kokoro/TTS inference: preserve tensor shapes, style embedding indexing,
  phoneme chunk limits, speed handling, sample rates, iSTFT overlap-add behavior,
  Metal device placement, and CPU fallback behavior where supported.
- Whisper/STT inference: preserve model selection, tokenizer/config loading,
  mel preprocessing, resampling, greedy decode/KV-cache behavior, audio duration
  handling, and meaningful transcript/error reporting.
- G2P and pronunciation: protect misaki-compatible tokenization, POS tagging,
  lexicon lookup, morphology, number/currency handling, stress assignment,
  espeak-ng fallback, legacy symbol conversion, and sentence-boundary chunking.
- Streaming and daemon contracts: keep `voice-stream` and `voice-protocol` as
  the shared source of truth for frame shapes, event ordering, daemon RPCs,
  cancellation, and stream-transcribe behavior. Reject duplicated or divergent
  constants in CLI, sidecar, Hermes, Telegram, or WhatsApp paths.
- CLI, MCP, and integrations: verify text/file/stdin resolution, markdown
  stripping, substitution handling, output formats, daemon detection, JSON-RPC
  shape, and bridge verifier scripts stay aligned with documented behavior.
- Platform and release behavior: check Git LFS assumptions, embedded assets,
  HuggingFace caching, macOS/Apple Silicon Metal paths, Linux CPU paths, release
  packaging, crate publishing order, and shell/Python verifier portability.
- Async and process lifecycle: review microphone recording, VAD, playback,
  daemon startup/shutdown, sidecar services, long-running watch scripts, temp
  files, and cancellation paths for stale state, blocking, races, or leaked
  processes.
- Tests: require focused tests at the changed boundary. Treat deleted tests as
  suspicious unless the behavior was removed and replacement coverage exists.
  Prefer deterministic unit or script-level tests for CLI/protocol/config
  changes, and explicit manual/runtime verification notes for audio, GPU, mic,
  daemon, or external messaging behavior.

## Finding Contract

Every finding must name a concrete failure mode or invariant drift, explain why
the diff introduced or exposed it, identify the affected file and line when
possible, and suggest the smallest useful fix. Do not invent findings. If there
are no actionable issues, say there are no actionable findings.

Use severity `nit` for non-blocking architecture/style findings. Use higher
severity only when the finding can cause a bug, security issue, data loss, audio
or transcript regression, platform breakage, protocol/runtime corruption, or a
serious review blocker.

Use one of these categories for each finding:

- `correctness`
- `model_inference`
- `audio_quality`
- `g2p_pronunciation`
- `stt_transcription`
- `streaming_daemon_protocol`
- `cli_mcp_surface`
- `platform_gpu`
- `packaging_release`
- `tests`
- `generated_or_embedded_artifact`
- `docs_config`
- `style_maintainability`
- `infra`
