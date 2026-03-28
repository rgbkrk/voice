# Agent context for voice

## What this project is

A Rust TTS and STT toolkit for Apple Silicon, implementing [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) 82M-parameter TTS and Whisper STT using [candle](https://github.com/huggingface/candle) with Metal GPU acceleration. Includes a misaki-compatible G2P pipeline for English text input.

## Current state

Production-quality TTS audio output. STT via distil-whisper (distil-large-v3 default). G2P handles dictionary lookup, embedded perceptron POS tagger, number handling, stress assignment, and espeak-ng fallback. 7 voices embedded in the binary for zero-network startup; 50+ downloadable from HuggingFace.

## Workspace layout

| Crate | Purpose |
|-------|---------|
| `crates/voice-kokoro/` | Kokoro-82M model — ALBERT encoder, prosody predictor, text encoder, iSTFT decoder |
| `crates/voice-tts/` | High-level TTS API — model loading, voice embedding, builtin voices, catalog |
| `crates/voice-stt/` | High-level STT API — Whisper model loading, transcription, resampling |
| `crates/voice-whisper/` | Whisper model backend — greedy decoding, GPU mel spectrogram |
| `crates/voice-g2p/` | Grapheme-to-phoneme — misaki dictionary + espeak-ng fallback |
| `crates/voice-cli/` | CLI binary (installs as `voice`) |

## Key files

| File | What it does |
|------|-------------|
| `voice-kokoro/src/model.rs` | `KModel::forward()` — full Kokoro inference pass |
| `voice-kokoro/src/albert.rs` | Custom ALBERT transformer encoder |
| `voice-kokoro/src/bilstm.rs` | Bidirectional LSTM modules |
| `voice-kokoro/src/modules.rs` | ProsodyPredictor, TextEncoder, conv/activation blocks |
| `voice-kokoro/src/istftnet.rs` | iSTFT decoder — GPU synthesis via conv_transpose1d, phase unwrapping |
| `voice-kokoro/src/config.rs` | ModelConfig (loaded from JSON) |
| `voice-tts/src/lib.rs` | `load_model()`, `load_voice()`, `generate()`, `save_wav()` |
| `voice-tts/src/builtin.rs` | 7 embedded voices + config for zero-network startup |
| `voice-tts/src/catalog.rs` | Metadata for all 50+ Kokoro voices |
| `voice-stt/src/lib.rs` | `load_model()`, `transcribe()` — Whisper STT API |
| `voice-stt/src/builtin.rs` | Embedded configs/tokenizers for distil-whisper models |
| `voice-whisper/src/decoder.rs` | Greedy decode loop with KV-caching, GPU mel preprocessing |
| `voice-g2p/src/lib.rs` | G2P pipeline — `english_to_phonemes()`, `text_to_phoneme_chunks()` |
| `voice-g2p/src/lexicon.rs` | Misaki dictionary lookup with morphology |
| `voice-g2p/src/tagger.rs` | Embedded perceptron POS tagger |
| `voice-g2p/src/espeak.rs` | Per-word espeak-ng fallback with E2M mapping |
| `voice-cli/src/main.rs` | CLI entry point, text resolution, markdown stripping, substitutions |
| `voice-cli/src/listen.rs` | Microphone recording, VAD, continuous transcription |
| `voice-cli/src/mcp.rs` | MCP server for AI agent integration |

## Critical implementation details

### Candle + Metal

- All inference runs on Metal GPU via candle's `Device::new_metal(0)`
- Weight loading uses `VarBuilder::from_mmaped_safetensors` for zero-copy
- iSTFT uses GPU-native `conv_transpose1d` with stride for overlap-add synthesis
- GPU linear interpolation for up/downsampling (no CPU fallback)
- Phase unwrap via atan2 is CPU-side (candle lacks atan2) — small data, negligible cost

### Voice embeddings

`(510, 1, 256)` lookup table indexed by `phoneme_count - 1`. First 128 dims = speaker style, last 128 = prosody style.

### G2P pipeline (voice-g2p)

Ported from [misaki](https://github.com/hexgrad/misaki)'s `en.py`:
1. Tokenize with embedded perceptron POS tagger
2. `fold_left` — merge non-head tokens
3. `retokenize` — subtokenize, handle punctuation/currency
4. Right-to-left lexicon lookup with `TokenContext` (future_vowel, future_to)
5. Morphological decomposition: `-s`, `-ed`, `-ing` suffix rules
6. espeak-ng per-word fallback with E2M mapping table
7. Legacy conversion: `ɾ→T`, `ʔ→t`
8. Long text chunked at 500-char phoneme limit, respects sentence boundaries

### STT (voice-stt / voice-whisper)

- Default model: `distil-whisper/distil-large-v3` (multilingual)
- Fallback: `distil-whisper/distil-medium.en` (English-only, faster)
- Configs and tokenizers embedded in binary for known models
- GPU mel spectrogram preprocessing on Metal
- Greedy decoding with KV-caching

## Build and test

```bash
# Build
cargo build --release -p voice

# Run TTS
voice "Hello world"
voice say -v am_adam "How are you today?"

# Speak then listen
voice converse "Please repeat after me"

# Listen (single-shot or continuous)
voice listen
voice listen --continuous

# Transcribe a file
voice transcribe recording.wav

# Run G2P tests
cargo test -p voice-g2p

# Run all tests
cargo test --workspace
```

## Python reference codebases

| Reference | Workspace path |
|-----------|---------------|
| mlx-audio (Kokoro) | `colombo-v2` — `mlx_audio/tts/models/kokoro/` |
| misaki (G2P) | `abuja-v4` — `misaki/en.py` |
| kokoro (PyTorch) | `montevideo` — `kokoro/` |

## Dependencies

- `candle-core 0.9` — Metal GPU inference, tensor ops
- `candle-nn 0.9` — Neural network layers
- `candle-transformers 0.9` — Whisper model implementation
- `candle-metal-kernels 0.9` — Metal compute kernels
- `safetensors 0.6` — Weight loading
- `hf-hub 0.5` — HuggingFace model downloads + caching
- `hound 3` — WAV file I/O
- `serde` / `serde_json` — config + dictionary parsing
- `rodio 0.22` — audio playback
- `clap 4` — CLI args
- `tokenizers 0.21` — HuggingFace fast tokenizers (Whisper)
- `rubato 1.0` — Audio resampling (STT)
- `pulldown-cmark 0.13` — Markdown stripping
