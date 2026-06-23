# voice-voxtral

Foundation crate for adding Mistral Voxtral TTS support to `voice`.

Current support is still pre-generation, but it now establishes the native load path and the first executable acoustic-transformer boundary:

- parse the official `params.json` shape used by `mistralai/Voxtral-4B-TTS-2603`
- validate `tekken.json` tokenizer/audio metadata against the model config
- expose the 20 preset voice IDs from the model config
- resolve local or HuggingFace metadata files without downloading the 8 GB weights by default
- require the 20 official voice prompt files when resolving the full inference asset set
- validate the official safetensors checkpoint layout and open it through Candle mmap loading
- instantiate typed Candle modules for the multimodal embeddings, language backbone, and acoustic transformer
- run the acoustic transformer's bidirectional attention blocks, semantic-codebook logits, and flow-matching velocity prediction in Candle
- define the native model boundary that the autoregressive generation loop and audio decoder will fill in

End-to-end native inference is not implemented yet. The reference implementation is a two-stage vLLM-Omni pipeline:

1. a Mistral text/audio generation stage
2. an audio tokenizer stage that decodes generated audio codebooks to 24 kHz audio

The next native milestones are text/token prompt construction, audio codebook parsing, the autoregressive generation loop, and the tokenizer/vocoder decoder.
