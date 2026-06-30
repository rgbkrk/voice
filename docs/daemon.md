# Voice Daemon

`voice daemon start` keeps the TTS model warm for low-latency synthesis and streaming.
The `voice` CLI still works without it: `voice say -o output.wav ...` falls
back to local synthesis, `voice mcp` initializes without a daemon, and
`voice stream` requires a running daemon.

The daemon listens on a Unix socket at `~/.voice/daemon.sock` and processes
requests through a single worker, so audio from concurrent MCP clients never
overlaps. There is no network listener or browser UI.

Use the fast verifier when changing CLI, MCP, or daemon detection behavior:

```bash
scripts/verify_cli_mcp_surface.py --voice-bin "$(command -v voice)"
```

It checks `voice stream-contract` and `voice mcp` with the daemon deliberately
hidden, then checks that MCP reports a daemon connection when
`voice daemon status --json` detects one. Add `--require-daemon` for release
hosts where the daemon must be installed.

## Install

Install `voice`, then register the daemon as a system service:

```bash
cargo install --path crates/voice-cli
voice daemon install
```

`voice daemon install` auto-detects the platform and writes the appropriate
service file (macOS LaunchAgent or Linux systemd user unit), loads it, and
starts the daemon. It also prints `voice daemon status` output on success.
On Linux, it disables the older `voice-daemon.service` unit if one exists so
only the current `voiced.service` registration owns the daemon socket.

On an interactive terminal it prints the detected chip and RAM, then prompts
for the speech-to-text model. Pressing Enter takes the default
(`distil-whisper/distil-large-v3.5`, English). When stdin is not a TTY (CI,
piped input, a provisioning script) it skips the prompt and installs the
default, so unattended installs just work.

To pick a model up front and skip the prompt, or to pin a model in an
unattended install:

```bash
voice daemon install --stt-model openai/whisper-large-v3   # multilingual
voice daemon install --yes                                 # accept defaults, no prompt
```

A non-default `--stt-model` is written into the service environment as
`STT_MODEL`, so the daemon loads it on every restart. The default tracks the
compiled-in default and writes no `STT_MODEL`, so future upgrades move with the
binary.

To install the service file without starting immediately:

```bash
voice daemon install --no-start
```

To remove the service:

```bash
voice daemon uninstall
```

Check status at any time:

```bash
voice daemon status
voice daemon status --json
```

## WebRTC Sidecar Service

For Hermes/WhatsApp Calling experiments on Linux, keep the daemon warm and run
the Python WebRTC sidecar as a separate user service:

```bash
voice daemon install
scripts/install_webrtc_sidecar_service.sh --voice-bin "$(command -v voice)"
scripts/verify_webrtc_sidecar_service.py --voice-bin "$(command -v voice)"
```

The helper creates or updates the sidecar venv, writes
`~/.config/systemd/user/voice-webrtc-sidecar.service`, enables it, and restarts
the service. The generated unit depends on `voiced.service`, binds the sidecar
control API to `127.0.0.1:8787`, mirrors inbound PCM under the XDG state
directory, and sets `VOICE_BIN` to the exact binary passed with `--voice-bin`.

Useful options:

```bash
scripts/install_webrtc_sidecar_service.sh --print-unit
scripts/install_webrtc_sidecar_service.sh --no-start
scripts/install_webrtc_sidecar_service.sh --uninstall
scripts/verify_webrtc_sidecar_service.py --skip-systemd
```

If Hermes is installed locally for WhatsApp, verify the running gateway service
is using the same voice stream contract and sidecar URL:

```bash
scripts/verify_hermes_gateway_service.py --voice-bin "$(command -v voice)"
scripts/verify_local_hermes_voice_stack.sh --voice-bin "$(command -v voice)"
```

The aggregate stack verifier checks the Hermes config, the running
`hermes-gateway.service` drop-in, CLI/MCP daemon behavior, Ogg/Opus voice-note
output, raw stream frames, stream STT, and the WebRTC sidecar service. When
`--run-webrtc-loopback-smoke` is enabled, the media smoke targets the configured
sidecar service URL instead of an in-process sidecar. Use `--skip-systemd` only
for non-service CI-style runs.

---

The sections below document the manual setup steps performed by
`voice daemon install`, for reference or non-standard installs.

## Linux systemd user service

Create `~/.config/systemd/user/voiced.service`:

```ini
[Unit]
Description=Voice daemon
After=default.target

[Service]
Type=simple
ExecStart=%h/.cargo/bin/voice daemon start --tts-only
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
```

If `voice` is installed somewhere else, replace `ExecStart` with the absolute
path from `command -v voice`. To pin a non-default STT model, add
`Environment=STT_MODEL=openai/whisper-large-v3` under `[Service]`.

Enable and start it:

```bash
systemctl --user daemon-reload
systemctl --user enable --now voiced.service
systemctl --user status voiced.service
```

Allow it to start at login without an active terminal session:

```bash
loginctl enable-linger "$USER"
```

## macOS LaunchAgent

Create `~/Library/LaunchAgents/com.rgbkrk.voice.voiced.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.rgbkrk.voice.voiced</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/.cargo/bin/voice</string>
    <string>daemon</string>
    <string>start</string>
    <string>--tts-only</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/voiced.out.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/voiced.err.log</string>
</dict>
</plist>
```

Replace `/Users/YOU/.cargo/bin/voice` with the absolute path from
`command -v voice`. To pin a non-default STT model, add an
`EnvironmentVariables` dict:

```xml
  <key>EnvironmentVariables</key>
  <dict>
    <key>STT_MODEL</key>
    <string>openai/whisper-large-v3</string>
  </dict>
```

Load and start it:

```bash
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.rgbkrk.voice.voiced.plist
launchctl enable "gui/$(id -u)/com.rgbkrk.voice.voiced"
launchctl kickstart -k "gui/$(id -u)/com.rgbkrk.voice.voiced"
```

Check status:

```bash
launchctl print "gui/$(id -u)/com.rgbkrk.voice.voiced"
voice daemon status
```

Unload it:

```bash
launchctl bootout "gui/$(id -u)" ~/Library/LaunchAgents/com.rgbkrk.voice.voiced.plist
```
