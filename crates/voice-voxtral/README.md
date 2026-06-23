# voice-voxtral

Foundation crate for adding Mistral Voxtral TTS support to `voice`.

This first slice is intentionally small:

- parse the official `params.json` shape used by `mistralai/Voxtral-4B-TTS-2603`
- expose the 20 preset voice IDs from the model config
- resolve local or HuggingFace metadata files without downloading the 8 GB weights by default
- define the native model boundary that a future Candle implementation will fill in

Native inference is not implemented yet. The reference implementation is a two-stage vLLM-Omni pipeline:

1. a Mistral text/audio generation stage
2. an audio tokenizer stage that decodes generated audio codebooks to 24 kHz audio

The next native milestones are tokenizer parity, audio codebook parsing, the acoustic transformer, and the tokenizer/vocoder decoder.
