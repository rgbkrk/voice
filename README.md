# voice

Rust TTS & STT powered by [candle](https://github.com/huggingface/candle), with Metal GPU acceleration on Apple Silicon and CPU fallback on Linux. Ships the [Kokoro](https://huggingface.co/prince-canuma/Kokoro-82M) 82M-parameter TTS model with a full English G2P pipeline, and [Whisper](https://huggingface.co/distil-whisper/distil-large-v3) for speech-to-text.

Faster time-to-first-speech than macOS `say`, with dramatically better audio quality. STT runs at ~50× real-time on Apple Silicon.

## Install

### Pre-built binary (recommended)

Install with [cargo-binstall](https://github.com/cargo-bins/cargo-binstall) to get a pre-built binary — no compilation required:

```bash
# Install cargo-binstall if you don't have it
cargo install cargo-binstall

# Install voice
cargo binstall voice
```

The tagged macOS release archive includes the `voice` binary. Its optional
daemon runs through `voice daemon start` and is used by `voice stream` and
daemon-backed file synthesis.

### Build from source

Requires Git LFS for embedded voice/model data:

```bash
# Install git-lfs if you don't have it
brew install git-lfs
git lfs install

# Clone and build
git clone https://github.com/rgbkrk/voice.git
cd voice
cargo install --path crates/voice-cli
```

> **Why git-lfs?** Voice data (`.safetensors`) and tagger weights are stored with Git LFS. Without it, those files are tiny pointers instead of actual data — the build will catch this and tell you what to do.

This puts the `voice` binary on your `$PATH`. Model weights are downloaded from HuggingFace Hub on first run and cached in `~/.cache/huggingface/hub/`. Seven popular voices and the model config are embedded in the binary — no network needed for common use.

## Usage

```bash
# Just talk (backward compatible — no subcommand needed)
voice Hello world, this is voice speaking.

# Explicit say subcommand with options
voice say -v am_michael "How are you today?"

# Pipe text in
echo "The quick brown fox jumps over the lazy dog." | voice say

# Save to file instead of playing
voice say -o speech.wav "Good morning everyone."
voice say --format ogg-opus -o speech.ogg "Good morning everyone."

# Stream daemon TTS frames for bridge/WebRTC experiments
voice stream-contract
voice stream --sample-rate 48000 --frame-ms 20 --raw-output speech.s16le "Good morning everyone."
voice stream --sample-rate 48000 --frame-ms 20 --output streamed.ogg --format ogg-opus "Good morning everyone."

# Replay audio through daemon streaming STT
voice stream-transcribe recording.ogg

# Verify messaging voice-note output and stream contract from an installed binary
VOICE_BIN=/path/to/voice scripts/verify_whatsapp_voice_contract.sh
VOICE_BIN=/path/to/voice scripts/verify_whatsapp_voice_contract.sh --require-daemon --run-stt-smoke
VOICE_BIN=/path/to/voice scripts/verify_telegram_voice_contract.sh --skip-hermes-config
VOICE_BIN=/path/to/voice scripts/verify_telegram_voice_contract.sh --require-telegram-credentials

# Read from a file, strip markdown
voice say --markdown -f blog-post.mdx

# Inspect G2P output without synthesis
voice phonemes "ChatGPT uses RuntimeStateDoc"

# Adjust speed
voice say -s 0.8 "Take it slow."

# Raw phoneme input
voice say --phonemes "həlˈO wˈɜɹld"

# Listen — single-shot speech-to-text from mic
voice listen

# Continuous listening — segments split on silence
voice listen --continuous

# Transcribe an audio file
voice transcribe recording.ogg

# JSON-RPC server for agent integration
voice serve
```

## CLI options

### `voice say`

```
Speak text aloud (default when no subcommand given)

Usage: voice say [OPTIONS] [TEXT]...

Arguments:
  [TEXT]...                       Text to speak

Options:
  -f, --input-file <FILE>        Read text from a file (use - for stdin)
      --phonemes <IPA>           Raw phoneme string (IPA)
  -v, --voice <VOICE>            Voice name [default: af_heart]
  -o, --output <PATH>            Write audio to file instead of playing
      --format <FORMAT>          Output format: wav, ogg-opus
  -s, --speed <SPEED>            Speech speed factor [default: 1.0]
      --markdown                 Strip markdown/MDX formatting before speaking
      --sub <WORD=REPLACEMENT>   Word substitution (repeatable)
      --sub-file <PATH>          Load substitutions from a file
  -q, --quiet                    Suppress progress output
  -h, --help                     Print help
```

### `voice phonemes`

```
Convert text to phoneme chunks without synthesis

Usage: voice phonemes [OPTIONS] [TEXT]...

Arguments:
  [TEXT]...                       Text to convert

Options:
  -f, --input-file <FILE>        Read text from a file (use - for stdin)
      --markdown                 Strip markdown/MDX formatting before conversion
      --sub <WORD=REPLACEMENT>   Word substitution (repeatable)
      --sub-file <PATH>          Load substitutions from a file
      --json                     Print a JSON object with preprocessed text and phoneme chunks
  -q, --quiet                    Suppress progress output
  -h, --help                     Print help
```

### `voice listen`

```
Record from microphone and transcribe (speech-to-text)

Usage: voice listen [OPTIONS]

Options:
      --continuous   Continuous mode — record and transcribe segments
                     as you speak. Segments split on silence.
  -q, --quiet        Suppress progress output
  -h, --help         Print help
```

### `voice stream`

Streams ordered PCM frames from a daemon started with `voice daemon start`. Use
this for bridge and WebRTC experiments where clients need audio chunks instead
of a completed WAV file. `--raw-output` writes headerless PCM for a sidecar or
test harness; `--output` writes streamed Ogg/Opus without first creating a WAV.

```
Usage: voice stream [OPTIONS] [TEXT]...

Options:
  -f, --input-file <FILE>          Read text from a file (use - for stdin)
  -v, --voice <VOICE>              Voice name [default: af_heart]
  -s, --speed <SPEED>              Speech speed factor [default: 1.0]
      --sample-rate <SAMPLE_RATE>  Target stream sample rate [default: 24000]
      --frame-ms <FRAME_MS>        Target frame duration in milliseconds [default: 20]
  -o, --raw-output <PATH>          Write raw signed 16-bit little-endian mono PCM
                                   (use - for stdout)
      --output <PATH>              Write streamed audio to an Ogg/Opus file
                                   (use - for stdout with --format ogg-opus)
      --format <FORMAT>            Output container/codec for --output
                                   [possible values: ogg-opus]
      --json                       Print full JSON stream events
      --markdown                   Strip markdown/MDX formatting before speaking
      --sub <WORD=REPLACEMENT>     Word substitution (repeatable)
      --sub-file <PATH>            Load substitutions from a file
```

### `voice stream-contract`

Prints the machine-readable WebRTC sidecar v1 contract generated from the
`voice-stream` constants. Use this when Hermes, a sidecar, or a test harness is
installed without a source checkout but still needs the exact PCM frame shape
and the `voice_surfaces` command map for voice notes, raw PCM, WebRTC, and
streaming smokes.

```
Usage: voice stream-contract
```

### `voice stream-transcribe`

Replays an audio file or raw PCM as ordered frames into a daemon started with
`voice daemon start`. Use this to smoke-test the inbound STT stream contract
that WebRTC and bridge clients use.

```
Usage: voice stream-transcribe [OPTIONS] [FILE]

Arguments:
  [FILE]                 Path to an audio file

Options:
      --raw-input <PATH>           Read raw signed 16-bit little-endian mono PCM
                                   from this path (use - for stdin)
      --sample-rate <SAMPLE_RATE>  Sample rate for --raw-input [default: 48000]
      --frame-ms <MS>              Target stream frame duration in milliseconds
                                   [default: 20]
      --json                       Print full JSON STT events
  -q, --quiet                      Suppress progress output
  -h, --help                       Print help
```

### `voice transcribe`

```
Transcribe an audio file. WAV input is decoded directly; Ogg/Opus and other
compressed formats use `ffmpeg` when available.

Usage: voice transcribe <FILE>
```

### `voice serve`

```
Run as a JSON-RPC 2.0 server on stdin/stdout

Usage: voice serve [OPTIONS]

Options:
  -v, --voice <VOICE>            Voice name [default: af_heart]
  -s, --speed <SPEED>            Speech speed factor [default: 1.0]
      --sub <WORD=REPLACEMENT>   Word substitution (repeatable)
      --sub-file <PATH>          Load substitutions from a file
```

## Speech-to-text

STT uses [distil-whisper](https://huggingface.co/distil-whisper/distil-large-v3) models via candle — knowledge-distilled versions of OpenAI's Whisper optimized for fast on-device transcription. On macOS, STT runs on Metal by default. On Linux and other non-macOS hosts, STT runs on CPU.

| Model | Repo ID | Params | Notes |
|-------|---------|--------|-------|
| Distil Large v3 | `distil-whisper/distil-large-v3` | 756M | Multilingual (default) |
| Distil Medium English | `distil-whisper/distil-medium.en` | 394M | English-only, faster |

Performance is ~50× real-time on Apple Silicon (a 10-second recording transcribes in ~200ms). CPU transcription is supported for portability but is much slower; use `distil-whisper/distil-medium.en` for faster CPU experiments. Configs and tokenizers for known models are embedded in the binary.

**Adaptive noise floor**: Before recording, `voice listen` calibrates against ambient noise for ~500ms, then sets a silence threshold relative to the noise floor. This avoids false triggers in noisy environments and missed speech in quiet ones. A **ding** sound plays when the mic is ready.

**Model selection**: The default model is `distil-whisper/distil-large-v3`. Override with the `STT_MODEL` environment variable:

```bash
STT_MODEL=distil-whisper/distil-medium.en voice listen
```

## Evaluation

The `eval/` scripts provide optional local STT/TTS evaluation. They may
download model weights and are not part of normal CI.

```bash
cargo build --release -p voice
./eval/compare.sh ./target/release/voice
./eval/synth_eval.sh ./target/release/voice
```

`compare.sh` scores recordings from `eval/recordings/`. `synth_eval.sh`
generates deterministic temporary TTS audio from `eval/phrases.txt` first. Both
write JSON results with WER, CER, exact-match counts, elapsed time, audio
duration, and real-time factor under `eval/results/`.

Before tagging a macOS release, compare the current checkout against the latest
stable release on local Apple Silicon hardware:

```bash
scripts/verify_cli_mcp_surface.py --voice-bin ./target/release/voice
scripts/macos_release_compare.py --keep
```

The fast verifier checks `stream-contract` and MCP startup with the daemon
deliberately hidden, then checks MCP daemon detection when a daemon is running.
The macOS comparison builds the current CLI, downloads the latest stable release
binary, forces the benchmark path to run without a daemon, and reports TTS
timing plus optional STT timing if `eval/recordings/*.wav` fixtures are present.
It also verifies plain `voice say` and `voice mcp` startup with the daemon
deliberately hidden, checks that `voice mcp` reports a daemon connection when
one is actually running, and synthesizes/transcribes `Wait, what. Wait what?`
as a file-based articulation smoke for issue #110. Pass
`--skip-articulation-smoke` only when you intentionally want a timing-only run.

## JSON-RPC server

`voice serve` runs a JSON-RPC 2.0 server on stdin/stdout, designed for integration with AI agents and tool-using LLMs.

### Methods

| Method | Description |
|--------|-------------|
| `speak` | Speak text or phonemes. Params: `text`, `phonemes`, `voice`, `speed`, `markdown`, `detail` |
| `listen` | Record from mic and transcribe. Params: `max_duration_ms`, `silence_timeout_ms`, `silence_threshold`, `noise_multiplier`, `calibration_ms` |
| `cancel` | Interrupt current speak playback |
| `set_voice` | Change the default voice. Params: `voice` |
| `set_speed` | Change the default speed. Params: `speed` |
| `list_voices` | List available builtin voices |
| `ping` | Health check — returns `"pong"` |

When `detail` is `"full"`, `speak` emits `speak.progress` notifications with chunk/phoneme info as audio streams.

### Example session

```jsonc
// Client → Server
{"jsonrpc": "2.0", "method": "speak", "params": {"text": "Hello! I'm listening."}, "id": 1}
// Server → Client
{"jsonrpc": "2.0", "result": {"duration_ms": 1520, "chunks": 1}, "id": 1}

// Client → Server
{"jsonrpc": "2.0", "method": "listen", "params": {"silence_timeout_ms": 2000}, "id": 2}
// Server → Client (after user speaks and silence is detected)
{"jsonrpc": "2.0", "result": {"text": "What's the weather like?", "tokens": 6, "duration_ms": 3200}, "id": 2}
```

See [`examples/conversation.py`](examples/conversation.py) for a full speak/listen conversation loop.

## Daemon TTS for Hermes and Streaming

Run `voice daemon start --tts-only` to keep Kokoro warm without eagerly loading STT. When the
daemon is running, `voice say -o output.wav ...` uses the daemon `synthesize`
RPC and waits until the WAV exists. For WhatsApp-ready voice notes, use
`voice say --format ogg-opus -o output.ogg ...` or an `.ogg` / `.opus` output
path; the CLI writes real `audio/ogg; codecs=opus` instead of a WAV with a
misleading extension. OGG/Opus output requires `ffmpeg` on `PATH`.

The macOS release archive and source install both expose the daemon through the
single `voice` binary. Use `voice daemon install` to register the daemon as a
service when the daemon or `voice stream` surface is needed.

See [docs/daemon.md](docs/daemon.md) for systemd user service and macOS
LaunchAgent examples.

For Linux Hermes/WhatsApp Calling experiments, install the Python WebRTC
sidecar as a separate user service after the daemon:

```bash
scripts/install_webrtc_sidecar_service.sh --voice-bin "$(command -v voice)"
scripts/verify_webrtc_sidecar_service.py --voice-bin "$(command -v voice)"
```

For WhatsApp or Telegram voice-note delivery through Hermes, prefer
`output_format: ogg` with `voice_compatible: true` so Hermes can send Voice's
Opus output directly. `output_format: wav` remains a compatibility fallback,
with Hermes converting to OGG/Opus when needed.

Telegram voice messages accept the same Ogg/Opus file shape as WhatsApp voice
notes, even though Telegram also accepts MP3 and M4A uploads. Run
`scripts/verify_telegram_voice_contract.sh` before adding the bot token; it
does not contact Telegram, but it verifies Voice's Ogg/Opus output contract
and, unless skipped, the active Hermes command-provider config. Add
`--require-telegram-credentials` when you want the preflight to fail unless
`TELEGRAM_BOT_TOKEN` is present in the Hermes env file. This covers Telegram bot
voice messages via `sendVoice`; it does not verify Telegram voice calls or a
live streaming transport.

Hermes can call `voice` directly, or use the voice-owned command-provider
shims:

```yaml
tts:
  provider: kokoro
  providers:
    kokoro:
      type: command
      command: /path/to/voice/examples/hermes-command-tts.sh {input_path} {output_path} {voice} {speed}
      output_format: ogg
      voice_compatible: true

stt:
  enabled: true
  provider: voice
  providers:
    voice:
      type: command
      command: /path/to/voice/examples/hermes-command-stt.sh {input_path}
      format: txt
```

To avoid hand-editing `~/.hermes/config.yaml`, generate or install the same
voice-native provider blocks with:

```bash
scripts/install_hermes_voice_config.py --print-snippet --voice-bin "$(command -v voice)"
scripts/install_hermes_voice_config.py --config ~/.hermes/config.yaml --voice-bin "$(command -v voice)"
scripts/install_hermes_voice_config.py --config ~/.hermes/config.yaml --voice-bin "$(command -v voice)" --apply
```

Without `--apply`, the installer performs a dry run and verifies the patched
config in a temporary file. With `--apply`, it keeps a timestamped backup and
runs `scripts/verify_hermes_voice_config.py` on the result.

`scripts/verify_hermes_voice_config.py` accepts either these shims or direct
`voice say --format ogg-opus` / `voice stream-transcribe --quiet` commands, and
rejects arbitrary wrappers so config drift is caught locally. Add
`--stt-audio ~/.hermes/audio_cache/aud_...ogg` when you also want to execute the
configured Hermes STT command against a cached inbound WhatsApp voice note. STT
smoke verifiers report transcript length and audio metrics, not transcript
text.

Use the local Hermes stack verifier before a release or host update:

```bash
scripts/verify_local_hermes_voice_stack.sh \
  --voice-bin "$(command -v voice)" \
  --hermes-config ~/.hermes/config.yaml
```

That single gate checks plain CLI/MCP daemon behavior, Hermes' active command
provider, the voice-owned config installer dry-run, direct Ogg/Opus voice note
output, daemon-backed streaming, `stream-transcribe`, and the local WebRTC
sidecar service. It also reports active and saved attended WhatsApp receive
watchers so the same gate shows whether a fresh-reply window is already
running. It requires the daemon and sidecar by default. For narrower checks, run
`scripts/verify_cli_mcp_surface.py`, `scripts/verify_hermes_voice_config.py`,
`scripts/verify_whatsapp_voice_contract.sh`,
`scripts/verify_telegram_voice_contract.sh`, or
`scripts/verify_webrtc_sidecar_service.py` directly.
Add `--run-telegram-voice-contract` to include Telegram bot voice-message
preflight output in the aggregate gate; add `--require-telegram-credentials`
when the local Telegram bot token should already be configured.
When a step fails, the aggregate verifier prints `failure_category=...` with
values such as `voice_runtime`, `hermes_config`, `upstream_hermes`,
`whatsapp_bridge_or_credentials`, `whatsapp_attended_watch`,
`telegram_setup`, or `webrtc_sidecar`. When the stack continues after a
classified nonzero check, such as external Meta setup or an alpha profile, the
final footer also prints `stack_failure_category=...`.

To include the categorized WhatsApp alpha report in the same command, add
`--whatsapp-alpha-profile unattended`, `cached-receive`, `send`,
`attended-cache-receive`, or `attended-send-receive`. Cached receive profiles
automatically run the configured Hermes STT command against the newest cached
inbound `aud_*` file when one is present; add `--skip-hermes-stt-smoke` to keep
the Hermes config check shape-only. The `send` and attended profiles perform
real bridge operations. Prefer `attended-cache-receive` when Hermes is already
watching the bridge; it waits for a fresh `aud_*` cache artifact without
draining queued messages. Use `attended-send-receive` only when the verifier
itself should poll and drain the bridge message queue. Add
`--whatsapp-alpha-chat-id` to override `WHATSAPP_HOME_CHANNEL`, or adjust the
attended wait with `--whatsapp-alpha-wait-audio-cache-seconds` /
`--whatsapp-alpha-wait-inbound-seconds`. The generated attended receive
handoff commands include explicit 60-second wait windows by default.

For a longer unattended receive window, start the non-draining cache watcher as
a transient user service:

```bash
scripts/start_whatsapp_attended_cache_watch.py \
  --voice-bin "$(command -v voice)" \
  --hermes-home ~/.hermes \
  --hermes-config ~/.hermes/config.yaml
```

It prints the unit name plus JSON/log/manifest artifact paths, then sends the
attended prompt voice note and waits for a fresh `aud_*` file without polling
the bridge `/messages` queue. The manifest is written before `systemd-run`
starts the long watch, so a later session can inspect the prompt text, send
intent, launched command, wait window, expected agent metadata, audio cache
directory, and artifact paths even while JSON/log output is still empty.
Use `--status <unit-or-service>` with the printed unit name to summarize
whether the watch is still waiting, failed, completed, or verified fresh
receive evidence. Status and list output also include the computed deadline and
remaining wait time from the manifest, so long attended windows can be checked
without hand-parsing timestamps. When the manifest names an audio cache
directory, status and list output also show the latest cached `aud_*` file and
whether it is newer than the watch start time. Use `--list` to discover active
watches and matching artifacts from a later session. Use `--stop
<unit-or-service>` to stop a stale watch without deleting its JSON/log/manifest
artifacts.
When the aggregate stack gate runs with `--whatsapp-alpha-profile` and attended
watch status is enabled, it passes the watch artifact directory and unit prefix
into the alpha readiness script. A previously verified non-draining watch can
therefore satisfy `pending_gates.attended_fresh_receive` in the saved alpha JSON
without asking the operator to rerun the attended receive window.

To fail unless every alpha gate is complete, include the strict completion flag:

```bash
scripts/verify_local_hermes_voice_stack.sh \
  --hermes-home ~/.hermes \
  --whatsapp-alpha-profile attended-cache-receive \
  --whatsapp-alpha-json-output ./whatsapp-alpha.json \
  --require-whatsapp-alpha-complete
```

That strict gate is expected to fail until the host has a fresh attended inbound
voice note plus the required Meta WhatsApp Cloud and Calling credentials. The
alpha output prints `whatsapp_cloud_setup`, `whatsapp_cloud_verify_command`,
`whatsapp_calling_setup`, `whatsapp_calling_verify_command`, and
`whatsapp_calling_complete_command` handoff lines so missing external setup is
separated from local daemon, bridge, and Hermes failures. It also prints the
resolved Cloud webhook bind defaults and malformed webhook keys, so strict
Cloud/Calling failures can distinguish missing credentials from invalid
`WHATSAPP_CLOUD_WEBHOOK_*` settings. The generated complete-alpha handoff
keeps the selected `voice` binary, Hermes config, bridge URL, sidecar URL,
audio cache override, and expected agent identity from the current run. It also
includes `--require-whatsapp-cloud`, `--require-whatsapp-calling`,
`--check-whatsapp-cloud-api`, `--check-whatsapp-cloud-health`, and
`--check-whatsapp-cloud-webhook`, so after real Cloud credentials are present it
also makes an authenticated Graph API phone-number request, verifies the local
Cloud adapter health contract, and verifies the local webhook challenge echo
without printing token or phone-number values. Failures from those explicit
Cloud probes are reported as external Meta setup, not as local Baileys bridge or
voice runtime failures.
The stack verifier runs alpha profiles through JSON internally so nonzero alpha
results can still be classified as local runtime, attended receive, or external
Meta setup. Use `--whatsapp-alpha-json-output` with any alpha profile when
another local agent should consume the same structured
`readiness_summary.next_actions` without rerunning the full stack gate; without
that flag the temporary JSON report is removed after the compact summary is
printed. The stack verifier echoes compact `whatsapp_alpha_json_*` summary
lines for every alpha profile, including readiness status, completion state,
next actions, attended receive status, and Cloud/Calling missing or invalid key
groups. When a verified attended watch is available, the compact summary also
reports whether the watch sent the prompt as a voice note, the local PTT
transport and Ogg/Opus MIME it used, and whether it used the non-draining
audio-cache receive path. A verified attended receive also stores compact
redacted proof under `pending_gates.attended_fresh_receive.evidence`, including
the attended-watch send format, send transport, and receive-watch mode.
Calling summaries split missing setup into `cloud_missing` for Meta credentials
and `sidecar_missing` for local WebRTC sidecar environment keys.

When the WebRTC Python dependencies are installed, add
`--run-webrtc-loopback-smoke --webrtc-python /tmp/voice-webrtc-venv/bin/python`
to run one full-duplex local media turn through the sidecar spike as well.

For lower-level streaming, `stream_speak` emits `tts.started`, `tts.audio`, and
terminal `tts.ended` / `tts.error` / `tts.cancelled` events over the same daemon
frame protocol. `stream_transcribe` accepts client-sent `stt.audio` frames and
returns a terminal `stt.transcribed` event. Use `voice stream-transcribe` to
replay WAV or Ogg/Opus audio through that inbound stream path. See
[docs/streaming.md](docs/streaming.md) for the event schema and Hermes/WebRTC
notes. See
[docs/whatsapp-calling-webrtc.md](docs/whatsapp-calling-webrtc.md) for the
WhatsApp Calling architecture.

## LLM-friendly design

`voice` is built to work well with AI agents and coding assistants:

- **Phoneme inspection**: `voice phonemes` emits the exact G2P chunks without loading the TTS model or playing audio
- **Phoneme output**: `voice say` emits phoneme chunks to stderr, so agents can see the IPA representation of what's being spoken
- **Phoneme input**: `--phonemes` accepts raw IPA strings, giving agents precise control over pronunciation without going through G2P
- **Stdin pipe**: `echo "text" | voice say` lets agents speak from any script or tool
- **Markdown stripping**: `--markdown` cleans up LLM-generated markdown before speaking
- **Word substitutions**: `--sub` and `.voice-subs` files let you fix pronunciation of project-specific terms
- **JSON-RPC server**: `voice serve` gives agents structured, bidirectional TTS + STT over stdin/stdout

See [SKILL.md](SKILL.md) for a lightweight reference card that AI agents can use to learn the `voice` tool.

## Word substitutions

Fix pronunciation of names, acronyms, or technical terms.

### Inline

```bash
voice say --sub nteract=enteract --sub PyTorch=pie-torch "nteract uses PyTorch"
```

### `.voice-subs` file

Create a `.voice-subs` file in your project root. `voice` auto-discovers it by walking up from the working directory.

```bash
# .voice-subs — one WORD=REPLACEMENT per line
nteract=enteract
PyTorch=pie-torch
MLX=M L X
kubectl=cube-cuddle

# Wrap in /slashes/ for phoneme overrides (bypass G2P entirely)
Kokoro=/kˈOkəɹO/
```

Text substitutions are applied before G2P. Phoneme overrides (the `/slash/` syntax) are injected directly into the phoneme stream.

```bash
# Uses .voice-subs automatically
voice say --markdown -f README.md

# Or specify a file explicitly
voice say --sub-file my-project.subs -f notes.txt
```

## Builtin voices

These voices are embedded in the binary and load instantly (no network):

| Voice | Description |
|-------|-------------|
| `af_heart` | American female — warm, natural (default) |
| `af_bella` | American female — expressive |
| `af_sarah` | American female — clear, professional |
| `af_sky` | American female — bright |
| `am_michael` | American male — clear |
| `am_adam` | American male — deeper |
| `bf_emma` | British female — natural |

All other voices are fetched from HuggingFace Hub on first use:

**American**: `af_alloy`, `af_aoede`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_onyx`, `am_puck`

**British**: `bf_alice`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

**Other languages**: French (`ff_siwis`), Hindi (`hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`), Italian (`if_sara`, `im_nicola`), Japanese (`jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`), Portuguese (`pf_dora`, `pm_alex`, `pm_santa`), Spanish (`ef_dora`, `em_alex`, `em_santa`), Chinese (`zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`)

## Crates

| Crate | Description |
|-------|-------------|
| [`voice`](https://crates.io/crates/voice) | CLI binary — installs as `voice` |
| [`voice-audio`](https://crates.io/crates/voice-audio) | Audio container helpers — WAV and OGG/Opus output |
| [`voice-tts`](https://crates.io/crates/voice-tts) | Core TTS library — model loading and inference |
| [`voice-stt`](https://crates.io/crates/voice-stt) | Speech-to-text library — Whisper transcription, resampling |
| [`voice-kokoro`](https://crates.io/crates/voice-kokoro) | Kokoro TTS backend — ALBERT encoder, prosody predictor, iSTFT decoder |
| [`voice-whisper`](https://crates.io/crates/voice-whisper) | Whisper STT backend — greedy decoding, GPU mel spectrogram |
| [`voice-g2p`](https://crates.io/crates/voice-g2p) | Grapheme-to-phoneme — misaki dictionary + embedded OOV fallback |

## Library usage

Add the crates you need:

```toml
[dependencies]
voice-tts = "0.2"
voice-stt = "0.1"   # if you need speech-to-text
voice-g2p = "0.2"   # if you need text-to-phoneme conversion
```

### Text-to-speech

```rust
use std::path::Path;

fn main() -> voice_tts::Result<()> {
    let mut model = voice_tts::load_model("prince-canuma/Kokoro-82M")?;
    let voice = voice_tts::load_voice("af_heart", None)?;

    let audio = voice_tts::generate(&mut model, "həlˈO wˈɜɹld", &voice, 1.0)?;
    voice_tts::save_wav(&audio, Path::new("output.wav"), 24000)?;

    Ok(())
}
```

### With G2P (text → phonemes)

```rust
fn main() -> voice_tts::Result<()> {
    let mut model = voice_tts::load_model("prince-canuma/Kokoro-82M")?;
    let voice = voice_tts::load_voice("af_heart", None)?;

    let chunks = voice_g2p::text_to_phoneme_chunks("Hello world, this is a test.")
        .expect("G2P failed");

    let mut all_samples: Vec<f32> = Vec::new();
    for phonemes in &chunks {
        let audio = voice_tts::generate(&mut model, phonemes, &voice, 1.0)?;
        all_samples.extend_from_slice(audio.as_slice());
    }

    voice_tts::save_wav_samples(&all_samples, std::path::Path::new("output.wav"), 24000)?;
    Ok(())
}
```

### Speech-to-text

```rust
fn main() -> voice_stt::Result<()> {
    let mut model = voice_stt::load_model("distil-whisper/distil-large-v3")?;
    let result = voice_stt::transcribe(&mut model, "audio.ogg")?;
    println!("{}", result.text);
    Ok(())
}
```

## Architecture

### TTS: Kokoro (82M)

- **G2P pipeline**: Ports [misaki](https://github.com/hexgrad/misaki)'s English G2P — POS tagging (embedded averaged perceptron), 90k gold + 93k silver dictionary entries, morphological decomposition, number/currency handling, embedded OOV fallback
- **Inference**: StyleTTS2-based model with ISTFT vocoder head. Audio chunks stream to speakers as they're generated — the first chunk plays while subsequent chunks are still synthesizing
- **Startup**: Model loads in a background thread while text resolution, G2P, and voice loading happen on the main thread

### STT: Whisper (distil-large-v3 / distil-medium.en)

- **Mel spectrogram**: Preprocessing runs on Metal GPU via candle on macOS, with CPU mel preprocessing on CPU-only hosts
- **Encoder-decoder transformer**: Standard Whisper architecture with knowledge distillation for faster inference
- **Greedy decode with KV cache**: Encoder output is computed once, then cached cross-attention keys/values are reused across all decoder steps
- **Embedded configs**: Tokenizers and model configs for known distil-whisper models are built into the binary

## Requirements

- macOS with Apple Silicon for Metal acceleration, or Linux/other hosts with CPU inference
- Rust 1.85+
- Git LFS (`brew install git-lfs && git lfs install`)
- Xcode command line tools

## License

MIT
