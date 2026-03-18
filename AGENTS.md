# Agent context for voice

## What this project is

A Rust TTS (text-to-speech) library implementing the [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) 82M-parameter model using [mlx-rs](https://github.com/oxiglade/mlx-rs) (Rust bindings for Apple's MLX framework). Includes a full misaki-compatible G2P pipeline for English text input.

## Current state

Production-quality audio output. Whisper STT validates 7/7 test phrases correctly, matching Python mlx-audio reference. G2P handles dictionary lookup (183k entries), morphological decomposition, number handling, POS tagging via spaCy, and espeak-ng fallback.

## Workspace layout

| Crate | Purpose |
|-------|---------|
| `crates/voice-tts/` | Core TTS library — model orchestration, config, weights, voice loading |
| `crates/voice-nn/` | Neural network modules — ALBERT, BiLSTM, vocoder, prosody, text encoder |
| `crates/voice-dsp/` | DSP primitives — STFT, iSTFT, overlap-add, hanning window |
| `crates/voice-g2p/` | Grapheme-to-phoneme — misaki dictionary + espeak-ng fallback |
| `crates/voice-cli/` | CLI binary (installs as `voice`) |

## Key files

| File | What it does |
|------|-------------|
| `voice-tts/src/model.rs` | `KokoroModel::generate()` — full forward pass |
| `voice-tts/src/weights.rs` | HF Hub download, weight sanitization (PyTorch -> mlx-rs) |
| `voice-tts/src/config.rs` | `ModelConfig`, `AlbertConfig`, `ISTFTNetConfig` |
| `voice-dsp/src/lib.rs` | STFT, iSTFT (with CPU overlap-add), hanning window, interpolate |
| `voice-nn/src/lstm.rs` | BiLstm with custom `lstm_forward_recurrent()` — fixes mlx-rs recurrence bug |
| `voice-nn/src/albert.rs` | ALBERT transformer encoder + `AlbertConfig` |
| `voice-nn/src/prosody.rs` | ProsodyPredictor, DurationEncoder |
| `voice-nn/src/text_encoder.rs` | TextEncoder (Conv1d + BiLSTM) |
| `voice-nn/src/vocoder/` | Decoder, Generator (iSTFT synthesis), SineGen (harmonic source) |
| `voice-g2p/src/lib.rs` | G2P pipeline + `G2PConfig` for custom tool paths |
| `voice-g2p/src/lexicon.rs` | 90k gold + 93k silver dictionary lookup with morphology |
| `voice-g2p/src/tokenizer.rs` | spaCy POS tagging (via `uv run`) + simple fallback |
| `voice-g2p/src/espeak.rs` | Per-word espeak-ng fallback with misaki E2M mapping |

## Critical implementation details

### mlx-rs workarounds (voice-nn)

- **LSTM recurrence**: mlx-rs `Lstm::step` doesn't propagate hidden/cell state between timesteps (Rust scoping issue — `let hidden = ...` shadows function param but doesn't feed back). Custom `lstm_forward_recurrent()` in `lstm.rs` fixes this.
- **Dropout eval mode**: mlx-rs `Dropout` defaults to `training=true`. Must call `training_mode(false)` on all modules before inference. Python MLX defaults to eval mode.
- **Intentional inference dropout**: The prosody predictor uses `mx.dropout(x, p=0.5)` at inference BY DESIGN. This is a raw function call, NOT `nn.Dropout`, so it's not affected by `training_mode`. Do not remove it.

### iSTFT normalization

- `normalized=false` (simple window, not squared) matches Python mlx-audio
- iSTFT uses CPU overlap-add because `scatter_add_single` isn't in mlx-rs 0.25.3

### Weight sanitization (`voice-tts/src/weights.rs`)

- `*.position_ids` -> skip
- `.gamma` -> `.weight`, `.beta` -> `.bias` (LayerNorm)
- LSTM: `weight_ih_l0` -> `forward_lstm.wx`, biases combined: `bias_ih + bias_hh`
- Conv weights with wrong shape: transpose `(0, 2, 1)`
- `duration_proj.linear_layer.` -> `duration_proj.`

### Voice embeddings

`(510, 1, 256)` lookup table indexed by `phoneme_count - 1`. First 128 dims = speaker style, last 128 = prosody style.

### G2P pipeline (voice-g2p)

Ported from [misaki](https://github.com/hexgrad/misaki)'s `en.py`:
1. Tokenize (spaCy via `uv run` preferred, whitespace fallback)
2. `fold_left` — merge non-head tokens
3. `retokenize` — subtokenize, handle punctuation/currency
4. Right-to-left lexicon lookup with `TokenContext` (future_vowel, future_to)
5. Morphological decomposition: `-s`, `-ed`, `-ing` suffix rules
6. espeak-ng per-word fallback with E2M mapping table
7. Legacy conversion: `ɾ→T`, `ʔ→t`

Configurable tool paths via `G2PConfig { uv_path, espeak_path }`.

## Build and test

```bash
# Build
cargo build --release -p voice

# Run TTS
voice --text "Hello world" --voice af_heart --play

# Run from source
cargo run --release -p voice -- --text "Hello world" --voice af_heart --play

# Run G2P tests (112 tests)
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
| spaCy | `edinburgh-v3` |
| mlx-rs | `miami` — `mlx-rs/src/nn/recurrent.rs` (LSTM bug reference) |

## Dependencies

- `mlx-rs 0.25.3` — Metal GPU, safetensors (crates.io)
- `mlx-macros 0.25.3` — `ModuleParameters` derive
- `hf-hub 0.5` — HuggingFace model downloads + caching
- `hound 3` — WAV writing
- `serde` / `serde_json` — config + dictionary parsing
- `fancy-regex 0.14` — subtokenize with lookahead (G2P)
- `unicode-normalization 0.1` — NFKC normalization (G2P)
- `rodio 0.20` — audio playback (CLI only)
- `clap 4` — CLI args (CLI only)
