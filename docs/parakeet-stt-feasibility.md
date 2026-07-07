# Parakeet STT feasibility

This note records the focused feasibility pass for using NVIDIA Parakeet as an
alternative STT backend for `voice`.

## Status

Parakeet is feasible on this Mac today through MLX, but not yet as a native
Rust/Candle backend. The practical near-term use is an external, explicit eval
or final-pass transcription harness. It should not replace Whisper in
`voice listen`, `voice transcribe`, or the daemon until a larger quality pass
proves better behavior on live voice recordings.

The local smoke on the existing `eval/recordings` fixtures did not show a
clear quality win. Parakeet TDT v3 loaded and ran locally, but it missed the
start of both fixture utterances. The current Whisper default
`distil-whisper/distil-large-v3.5` also missed leading words on these fixtures,
so the next quality slice should use freshly recorded Vodex-style utterances
and inspect the fixture leading audio rather than overfitting to these two
short files.

## Current artifacts

| Model | Runtime artifact | Size evidence | Status |
| --- | --- | --- | --- |
| `mlx-community/parakeet-tdt-0.6b-v3` | MLX `model.safetensors`, config, tokenizer | HF API reports `model.safetensors` as 2,508,288,736 bytes | Tested locally |
| `mlx-community/parakeet-tdt-0.6b-v2` | MLX `model.safetensors`, config, tokenizer | HF API reports `model.safetensors` as 2,471,559,904 bytes | Artifact confirmed; local smoke attempted and interrupted |
| `mlx-community/parakeet-ctc-1.1b` | MLX `model.safetensors`, config, tokenizer | HF API reports `model.safetensors` as 4,250,695,964 bytes | Artifact confirmed, not smoke-tested |
| `nvidia/parakeet-tdt-0.6b-v3` | Transformers/HF safetensors plus `.nemo` | HF API reports `model.safetensors` as 2,508,311,120 bytes and `.nemo` as 2,509,332,480 bytes | Source artifact confirmed |

Primary source pointers:

- NVIDIA Parakeet TDT v3 model card:
  https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- NVIDIA Parakeet TDT v2 model card:
  https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
- NVIDIA Parakeet CTC 1.1B model card:
  https://huggingface.co/nvidia/parakeet-ctc-1.1b
- MLX Community Parakeet collection:
  https://huggingface.co/collections/mlx-community/parakeet
- `parakeet-mlx` implementation:
  https://github.com/senstella/parakeet-mlx
- Apple MLX project:
  https://opensource.apple.com/projects/mlx

## Local smoke evidence

Machine/runtime prerequisites:

```bash
which uv
which ffmpeg
uname -m
```

Observed:

- `uv`: `/opt/homebrew/bin/uv`
- `ffmpeg`: `/opt/homebrew/bin/ffmpeg`
- architecture: `arm64`

First-run Parakeet TDT v3 smoke:

```bash
rm -rf /tmp/voice-parakeet-smoke-v3
mkdir -p /tmp/voice-parakeet-smoke-v3
/usr/bin/time -p uvx --from parakeet-mlx parakeet-mlx \
  eval/recordings/001.wav \
  --model mlx-community/parakeet-tdt-0.6b-v3 \
  --output-dir /tmp/voice-parakeet-smoke-v3 \
  --output-format json \
  --verbose
```

Result:

- First run wall time including package/model download: `real 810.15`
- Transcript: `Fox jumps over the lazy dog.`
- Expected: `The quick brown fox jumps over the lazy dog.`
- Output JSON: `/tmp/voice-parakeet-smoke-v3/001.json`
- Cached model path:
  `~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3`
- Cached model blob:
  `blobs/05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592`
  at 2,508,288,736 bytes

Cached two-fixture Parakeet TDT v3 eval:

```bash
/usr/bin/time -p uv run --with parakeet-mlx python eval/evaluate.py \
  --recordings eval/recordings \
  --engine parakeet-mlx \
  --json-out /tmp/voice-parakeet-eval-v3.json
```

Result:

- Model: `mlx-community/parakeet-tdt-0.6b-v3`
- Wall time: `real 5.30`
- Audio duration: `17.1s`
- Aggregate RTF: `0.2816`
- Exact: `0/2`
- Mean WER: `41.7%`
- `001`: `Fox jumps over the lazy dog.`
- `002`: `Shells by the seashore.`

Parakeet TDT v2 attempted smoke:

```bash
/usr/bin/time -p uv run --with parakeet-mlx python eval/evaluate.py \
  --recordings eval/recordings \
  --engine parakeet-mlx \
  --model mlx-community/parakeet-tdt-0.6b-v2 \
  --json-out /tmp/voice-parakeet-eval-v2.json
```

Result:

- Interrupted after `real 182.06` because no visible model-cache progress was
  observed beyond a 40 KB repo stub.
- No `parakeet-mlx` child process remained after interrupt.
- v2 remains artifact-confirmed through Hugging Face, but not locally
  smoke-tested in this branch.

Current Rust Whisper comparison from this checkout:

```bash
cargo build -p voice
/usr/bin/time -p python3 eval/evaluate.py \
  --recordings eval/recordings \
  --voice target/debug/voice \
  --model distil-whisper/distil-large-v3.5 \
  --json-out /tmp/voice-whisper-eval-large-v3-5.json
```

Result:

- Build initially required `git lfs pull` because the Codex worktree had LFS
  pointer files for embedded data; after that, `cargo build -p voice` passed.
- Model: `distil-whisper/distil-large-v3.5`
- Wall time: `real 7.67`
- Audio duration: `17.1s`
- Aggregate RTF: `0.4445`
- Exact: `0/2`
- Mean WER: `36.1%`
- `001`: `The fox jumps over the lazy dog.`
- `002`: `Shells by the seashore.`

## Harness

`eval/evaluate.py` now supports an explicit Parakeet MLX engine:

```bash
uv run --with parakeet-mlx python eval/evaluate.py \
  --recordings eval/recordings \
  --engine parakeet-mlx \
  --model mlx-community/parakeet-tdt-0.6b-v3 \
  --json-out eval/results/parakeet_mlx_v3.json
```

The default engine remains the existing `voice` CLI path. The shell wrappers
also keep their default Whisper model matrix unchanged; set `PARAKEET_MLX=1`
to add an opt-in Parakeet pass. The wrappers run that Parakeet pass through
`uv run --with parakeet-mlx`, so a clean checkout needs `uv` but does not need a
preinstalled `parakeet-mlx` console script:

```bash
PARAKEET_MLX=1 ./eval/compare.sh target/debug/voice
PARAKEET_MLX=1 ./eval/synth_eval.sh target/debug/voice
```

Parakeet infrastructure failures are strict by default. If the Parakeet binary
is missing or the subprocess exits nonzero, `eval/evaluate.py` records the item
error and exits nonzero. Use `--allow-errors` only for exploratory scoring where
empty/error transcripts should not fail the command.

## Streaming assessment

Treat Parakeet TDT v2/v3 MLX as final-pass transcription for now. The public
`parakeet-mlx` CLI supports long-audio chunking with overlap, but it is not a
streaming partial-transcript API for the current `voice` daemon/listen loop.

NVIDIA's v2 model-card discussion points to chunked buffered inference and
separately mentions dedicated cache-aware streaming architectures. That is not
the same integration surface as the current foreground Whisper partial path in
`voice-cli`, which owns VAD snapshots and calls `voice_stt::WhisperModel`
directly.

## Native Rust/Candle blockers

A native implementation is plausible but non-trivial. It is not a loader swap.

Main blockers:

- FastConformer encoder in Candle: convolutional subsampling, depthwise
  separable convolution modules, relative-position attention, global/local
  attention masks, and efficient long-audio memory behavior.
- Decoder choice:
  - CTC is the shortest first native target because decoding is simpler.
  - TDT/RNNT requires decoder, joint network, duration prediction, and greedy or
    beam search with timestamp handling.
- Tokenizer support: Parakeet MLX artifacts use `tokenizer.model` and
  `vocab.txt`, not the Whisper `tokenizer.json` path currently embedded in
  `voice-stt`.
- Preprocessing parity: 16 kHz mono audio, mel/filterbank feature settings,
  normalization, padding, and chunking need to match NeMo/Transformers configs.
- Weight mapping: HF safetensors and `.nemo` names must be mapped into Candle
  modules with shape checks and test fixtures.
- Streaming/cache behavior: native final-pass transcription can come first;
  low-latency partials would require buffered or cache-aware inference design.

## Recommended next slice

1. Record a small Vodex/live-voice fixture suite with leading-speech and
   trailing-silence cases that reproduce the Whisper repetition problem.
2. Run the new eval harness across `distil-large-v3.5`,
   `mlx-community/parakeet-tdt-0.6b-v2`, `mlx-community/parakeet-tdt-0.6b-v3`,
   and `mlx-community/parakeet-ctc-1.1b`.
3. If Parakeet wins quality, add an explicit final-pass STT backend boundary
   before daemon/listen integration.
4. For native Rust, prototype a `voice-parakeet` crate against a CTC model
   first, then decide whether TDT is worth the extra decoder work.
