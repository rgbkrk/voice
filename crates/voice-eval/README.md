# voice-eval

`voice-eval` runs a local intelligibility loop:

1. synthesize text with the current TTS backend,
2. transcribe the generated samples with Whisper STT,
3. normalize both strings and report word error rate.

It is intentionally narrow. It measures whether generated speech can be
recognized by the repo's STT path; it does not score naturalness, speaker
similarity, latency, or prosody.

```bash
cargo run -p voice-eval -- \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice am_adam \
  --output-wav /tmp/voice-eval.wav
```

Use `--json` for machine-readable output, and `--stt-model
distil-whisper/distil-medium.en` for a faster smaller transcriber.
