# Agent context for voicers

## What this project is

A Rust TTS (text-to-speech) library porting the Python [mlx-audio](https://github.com/lucasnewman/mlx-audio) Kokoro model to Rust using [mlx-rs](https://github.com/oxiglade/mlx-rs) (Rust bindings for Apple's MLX framework).

## Current state

End-to-end audio generation works. The model loads from HuggingFace Hub, processes phoneme input through the full Kokoro pipeline, and outputs 24kHz WAV audio. Audio quality has not been validated against the Python reference.

## Crate layout

| Crate | Purpose |
|-------|---------|
| `crates/voicers/` | Core library: model, modules, weights, DSP, public API |
| `crates/voicers-cli/` | CLI binary with `--play` via rodio |
| `crates/voicers-g2p/` | Stub for future grapheme-to-phoneme (espeak-ng) |

## Key files

| File | What it does |
|------|-------------|
| `voicers/src/model.rs` | `KokoroModel` struct and `generate()` forward pass |
| `voicers/src/weights.rs` | HF Hub download, weight sanitization (PyTorch -> mlx-rs key mapping) |
| `voicers/src/config.rs` | `ModelConfig`, `AlbertConfig`, `ISTFTNetConfig` (serde from config.json) |
| `voicers/src/dsp.rs` | STFT, iSTFT, hanning window, interpolate, mlx_angle, mlx_unwrap |
| `voicers/src/voice.rs` | Voice embedding loading from safetensors |
| `voicers/src/modules/albert.rs` | ALBERT transformer encoder (embeddings, attention, FFN) |
| `voicers/src/modules/prosody.rs` | ProsodyPredictor, DurationEncoder (F0/voicing prediction) |
| `voicers/src/modules/text_encoder.rs` | TextEncoder (Conv1d blocks + BiLSTM) |
| `voicers/src/modules/lstm.rs` | BiLstm (bidirectional LSTM wrapping two mlx-rs Lstm) |
| `voicers/src/modules/conv_weighted.rs` | Weight-normalized Conv1d/ConvTranspose1d |
| `voicers/src/modules/ada_norm.rs` | InstanceNorm1d, AdaIN1d, AdaLayerNorm |
| `voicers/src/modules/vocoder/decoder.rs` | Vocoder Decoder + AdainResBlk1d |
| `voicers/src/modules/vocoder/generator.rs` | Generator (upsampling + iSTFT synthesis) |
| `voicers/src/modules/vocoder/source.rs` | SineGen, SourceModuleHnNSF, AdaINResBlock1 |

## Python reference files

The Python mlx-audio source was used as reference (attached to the workspace at the mlx-audio colombo-v2 workspace):

| Rust module | Python source |
|-------------|--------------|
| model.rs | `mlx_audio/tts/models/kokoro/kokoro.py` |
| modules/*.rs | `mlx_audio/tts/models/kokoro/modules.py` |
| vocoder/*.rs | `mlx_audio/tts/models/kokoro/istftnet.py` |
| dsp.rs | `mlx_audio/dsp.py` + `mlx_audio/tts/models/interpolate.py` |
| weights.rs | `mlx_audio/tts/utils.py` + `kokoro.py::sanitize()` |

## Weight sanitization

The model weights from HuggingFace are in PyTorch format. Key transformations in `weights.rs::sanitize_weights()`:

- `*.position_ids` -> skip
- `.gamma` -> `.weight`, `.beta` -> `.bias` (LayerNorm)
- LSTM keys: `weight_ih_l0` -> `forward.wx`, `weight_hh_l0` -> `forward.wh`, reverse variants -> `backward.*`
- LSTM biases: `bias_ih + bias_hh` combined into single `bias`
- Conv weights with wrong shape: transpose `(0, 2, 1)`
- `F0_proj.weight`, `N_proj.weight`: transpose `(0, 2, 1)`
- Decoder noise_convs weights: transpose `(0, 2, 1)`

## Voice embeddings

Voice files are `(510, 1, 256)` tensors — a lookup table indexed by phoneme count. To get the style vector for N phonemes: `voice[N - 1]` gives `(1, 256)`. The first 128 dims are speaker style, the last 128 are prosody style.

## Known issues and next steps

### Audio quality
- Output audio plays but quality hasn't been A/B tested against Python
- Weight sanitization may have edge cases (check `check_array_shape` logic for conv weight transposition)
- The `istft` runs overlap-add on CPU because `scatter_add_single` isn't in mlx-rs 0.25.3 (only in unreleased local version). This works but is slower than a GPU-native approach.

### Missing features
- **G2P**: `voicers-g2p` is a stub. Need espeak-ng bindings to convert text -> phonemes
- **Streaming**: No streaming audio output yet
- **Multiple languages**: Pipeline supports English phonemes only (no misaki/espeak G2P integration)
- **Speed**: Release builds are fast but the CPU overlap-add in istft could be moved to GPU once mlx-rs publishes scatter_add support

### mlx-rs API gaps (as of 0.25.3)
- No `scatter_add_single` (available in local unreleased source at the mlx-rs workspace)
- `Module` trait requires `training_mode()` method on all impls
- `Builder` trait must be imported for `.build()` calls
- `Array::from_f32()` for scalars, `Array::from_int()` for i32 scalars
- `as_dtype()` instead of `as_type()` for type conversion
- ConvWeighted uses `.T` (full axis reverse `[2,1,0]`) for conv_transpose1d weight matching

## Build and test

```bash
# Check compilation
cargo check --workspace

# Build release
cargo build --release -p voicers-cli

# Run TTS
cargo run --release -p voicers-cli -- \
  --phonemes "h ɛ l oʊ" --voice af_heart --output test.wav --play
```

## Dependencies

- `mlx-rs 0.25.3` from crates.io (Metal GPU, safetensors support)
- `mlx-macros 0.25.3` for `ModuleParameters` derive macro
- `hf-hub 0.5` for HuggingFace model downloads
- `hound 3` for WAV writing
- `rodio 0.20` for audio playback (cli only)
- `clap 4` for CLI args (cli only)
