# voice-g2p

English grapheme-to-phoneme conversion for [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) TTS. A Rust port of [misaki](https://github.com/hexgrad/misaki)'s English G2P pipeline.

## Install

```toml
[dependencies]
voice-g2p = "0.1"
```

## Usage

```rust
// Convert text to Kokoro-compatible phonemes
let phonemes = voice_g2p::english_to_phonemes("Hello world")?;
// => "həlˈO wˈɜɹld"

// For long text, split into chunks that fit the model's 510-token limit
let chunks = voice_g2p::text_to_phoneme_chunks("A very long paragraph...")?;
for chunk in &chunks {
    // Each chunk is ≤500 phoneme characters
    println!("{chunk}");
}
```

### Custom configuration

If `uv` or `espeak-ng` aren't on your `$PATH`:

```rust
let config = voice_g2p::G2PConfig {
    uv_path: "/opt/homebrew/bin/uv".into(),
    espeak_path: "/opt/homebrew/bin/espeak-ng".into(),
};
let g2p = voice_g2p::G2P::with_config(config);
let phonemes = g2p.convert("Hello world")?;
```

## What's inside

- **Dictionary lookup** — 90k gold + 93k silver pronunciation entries embedded at compile time
- **Morphological decomposition** — `-s`, `-ed`, `-ing` suffix rules with voicing logic
- **Number handling** — cardinals, ordinals, years, currency, phone numbers
- **POS tagging** — optional spaCy subprocess (via `uv run`) for context-dependent pronunciation
- **Fallback** — espeak-ng per-word for unknown words

## Optional dependencies

- **espeak-ng** — fallback pronunciation for words not in the dictionary (`brew install espeak-ng`)
- **uv** — runs spaCy for POS-based disambiguation (e.g. "read" as past vs. present tense)

Both are optional. Without them, the pipeline still works using dictionary lookup alone.

## License

MIT