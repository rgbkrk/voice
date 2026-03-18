# voicers-cli

Command-line interface for voicers TTS.

## Usage

```bash
voicers-cli --phonemes "h ɛ l oʊ w ɜː l d" --voice af_heart --output hello.wav --play
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `prince-canuma/Kokoro-82M` | HF repo or local path |
| `--phonemes` | (required) | IPA phoneme string |
| `--voice` | `af_heart` | Voice name |
| `--output` | `output.wav` | Output WAV path |
| `--play` | off | Play audio after generation |
| `--speed` | `1.0` | Speech speed factor |
