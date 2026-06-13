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

## What's inside

- **Dictionary lookup** — 90k gold + 93k silver pronunciation entries embedded at compile time
- **Morphological decomposition** — `-s`, `-ed`, `-ing` suffix rules with voicing logic
- **Number handling** — cardinals, ordinals, years, currency, phone numbers
- **POS tagging** — embedded averaged perceptron model for context-dependent pronunciation
- **Fallback** — embedded OOV phonemizer for unknown words, acronyms, and mixed alphanumeric tokens

## Offline tooling

The runtime G2P pipeline is self-contained. The `generate-bronze` helper binary
still uses `espeak-ng` to regenerate the embedded bronze dictionary data.

## License

MIT
