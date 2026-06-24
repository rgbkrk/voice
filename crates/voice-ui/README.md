# voice-ui

React components and static assets for the Voice queue/player surface.

The web app is intentionally separate from `voice-tray`: it can run as a browser prototype today and can later be embedded into Rust binaries.

## Development

```bash
npm install
npm run dev
```

The dev app uses mock queue data unless `/api/daemon/status` returns the
daemon status JSON shape from `voice daemon status --json`.

## Build Web Assets

```bash
npm run build
```

This writes static assets to `crates/voice-ui/dist`.

## Embed In Rust

After `npm run build`, Cargo embeds the current `dist` files:

```bash
cargo test -p voice-ui
```

Rust consumers can call `voice_ui::assets()` or `voice_ui::get("index.html")`.
