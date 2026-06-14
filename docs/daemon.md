# Voice Daemon

`voice daemon start` keeps the TTS model warm for low-latency synthesis and streaming.
The `voice` CLI still works without it: `voice say -o output.wav ...` falls
back to local synthesis, `voice mcp` initializes without a daemon, and
`voice stream` requires a running daemon.

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
path from `command -v voice`.

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
`command -v voice`.

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
