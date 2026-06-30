# voice-voxtral

Native Mistral Voxtral TTS for `voice`, built on Candle with Metal acceleration.

It runs end-to-end text-to-audio generation against `mistralai/Voxtral-4B-TTS-2603`:

- parse the official `params.json` shape and validate the `tekken.json` tokenizer/audio metadata against it
- resolve local or HuggingFace assets, including the 20 official voice prompt files, downloading the ~8 GB weights only when full inference is requested
- load the safetensors checkpoint through Candle mmap into typed modules for the multimodal embeddings, language backbone, and acoustic transformer
- build the text/audio prompt, run the autoregressive generation loop, decode semantic codebooks and flow-matching velocities, and vocode to 24 kHz audio
- expose the 20 preset voice IDs from the model config

The runtime offers batch generation (`VoxtralTtsRuntime::generate_audio`) and frame-streaming generation (`generate_audio_streaming_with_trace`) for low-latency playback. The CLI reaches it through `voxtral say` / `voice say --engine voxtral`.

The reference implementation is a two-stage vLLM-Omni pipeline (a Mistral text/audio generation stage feeding an audio tokenizer that decodes codebooks to 24 kHz). This crate reimplements both stages natively in Candle.
