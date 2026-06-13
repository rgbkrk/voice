# Voice Daemon

`voiced` keeps the TTS model warm for low-latency synthesis and streaming.
The `voice` CLI still works without it: `voice say -o output.wav ...` falls
back to local synthesis, `voice mcp` initializes without a daemon, and
`voice stream` requires a running daemon.

## Install

The macOS release archive includes both `voice` and `voiced`. Source installs
should install both binaries when daemon-backed synthesis or streaming is
needed:

```bash
cargo install --path crates/voice-cli
cargo install --path crates/voice-daemon
```

Start manually:

```bash
voiced --tts-only
```

Check status:

```bash
voice daemon status
voice daemon status --json
```

## Linux systemd user service

Create `~/.config/systemd/user/voiced.service`:

```ini
[Unit]
Description=Voice daemon
After=default.target

[Service]
Type=simple
ExecStart=%h/.cargo/bin/voiced --tts-only
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
```

If `voiced` is installed somewhere else, replace `ExecStart` with the absolute
path from `command -v voiced`.

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
    <string>/Users/YOU/.cargo/bin/voiced</string>
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

Replace `/Users/YOU/.cargo/bin/voiced` with the absolute path from
`command -v voiced`.

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
