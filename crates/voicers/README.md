# voicers

Core TTS library for Kokoro model inference on Apple Silicon via mlx-rs.

## API

```rust
// Load model (downloads from HF Hub on first use, cached after)
let mut model = voicers::load_model("prince-canuma/Kokoro-82M")?;

// Load voice embedding
let voice = voicers::load_voice("af_heart", None)?;

// Generate audio from phonemes
let audio = voicers::generate(&mut model, "h ɛ l oʊ", &voice, 1.0)?;

// Save to WAV
voicers::save_wav(&audio, Path::new("output.wav"), 24000)?;
```

## Module organization

- `config` — Model configuration structs (deserialized from HF config.json)
- `model` — `KokoroModel` with the full TTS forward pass
- `weights` — Weight downloading, caching, and sanitization
- `voice` — Voice embedding loading
- `dsp` — STFT/iSTFT, windowing, interpolation
- `modules/` — Neural network building blocks:
  - `albert` — ALBERT transformer encoder
  - `prosody` — Duration and F0/voicing prediction
  - `text_encoder` — Conv + BiLSTM text features
  - `lstm` — Bidirectional LSTM wrapper
  - `conv_weighted` — Weight-normalized convolution
  - `ada_norm` — Adaptive normalization layers
  - `vocoder/` — Decoder, Generator, harmonic source modules
