# Daemon-backed Voice UI worklog

## 2026-06-24

- Baseline: branch `quod/daemon-backed-voice-ui`, draft PR #288, commit `8eebdf1`.
- Local checks before goal activation passed: `npm run build`, `npm run check`, `cargo check -p voice-daemon`, `cargo test -p voice-daemon -p voice-protocol -p voice-ui`, `cargo fmt --check`, `git diff --check`.
- GitHub checks before goal activation passed: `check`, `sidecar-python`, `python-helpers`.
- Known gap: UI `respond` command currently changes transport state only. It does not record/transcribe and attach answer state to the selected track.
- Known gap: UI playlist entries are projected from the legacy queue/recent state. Agent submissions are not yet fully held/synthesized into prompt WAVs without autoplay.
- Next action: add daemon-side UI track mechanics and a no-mic verifier for prompt/answer audio state.

## 2026-06-24 held-track slice

- Added `held_for_ui` to protocol queue items and daemon queue entries. Old snapshots deserialize with the default `false`.
- Added `RequestQueue::enqueue_ui_held`; worker `dequeue` skips held entries so mailbox tracks stay visible until command handling moves them.
- Added `RequestQueue::complete_held_item` to move a held response track into recent history with a stored result.
- Added explicit socket `ui_hold` support for non-waiting `speak`, `listen`, and `converse` requests. Existing CLI requests are unchanged because they do not send `ui_hold`.
- Added `VOICE_AUDIO_DIR` override for isolated tests and local E2E.
- Added env-gated no-mic response completion through `VOICE_UI_TEST_RESPONSE_TEXT`; normal `respond` still enters listening state.
- Added in-process HTTP verifier: snapshot -> POST `/api/ui/commands/respond` -> snapshot -> GET `/api/ui/audio/:track_id/answer`.
- Passing checks:
  - `cargo test -p voice-daemon -p voice-protocol -p voice-ui`
  - `npm run check` in `crates/voice-ui`
  - `npm run build` in `crates/voice-ui`
- Remaining gap: production `respond` still needs daemon-owned native mic/STT completion instead of only the test response path.
- Remaining gap: held prompt audio still depends on pre-existing question WAVs or future synthesis wiring; the worker does not yet prepare prompt WAVs for held tracks automatically.
