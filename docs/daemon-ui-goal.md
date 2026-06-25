# Daemon-backed Voice UI goal

## Outcome

Make draft PR #288 a working daemon-backed Voice UI integration. The daemon should serve the embedded app, expose UI state over HTTP and SSE, accept UI commands, keep playlist state outside React, and support enough end-to-end playback and response flow that the UI can be exercised without mock data.

## Baseline

Commit `8eebdf1` adds the first bridge: embedded assets, snapshot/events endpoints, command stubs, audio URLs, and an RxJS frontend store. Local checks and GitHub checks pass. The deeper mailbox behavior is incomplete: response commands only enter listening state, held prompt synthesis is not fully modeled, and there is no non-human E2E path for answer recording/transcription.

## Constraints

- Keep PR #288 draft until the integration is genuinely usable.
- Preserve the legacy Unix socket/CLI path unless a focused compatibility change is needed.
- The browser UI uses HTTP POST commands plus SSE state; do not make React own daemon state again.
- Browser playback should use served WAVs so seek/rewind/pause are real media controls.
- Do not require a human at the microphone for automated E2E. Use synthetic WAV input or a daemon test hook when needed.
- Do not weaken existing tests, replace real protocol checks with mocks, or hide failures by narrowing the verifier.

## Non-goals

- Public deployment or production release.
- Browser-side Automerge sync.
- Rebuilding the visual design unless a behavior requires a small UI affordance.
- Removing the legacy socket protocol wholesale in this slice.

## Primary verifier

An automated daemon/UI flow can start the daemon or an in-memory UI server, create at least one real UI track, observe it through `/api/ui/snapshot` and `/api/ui/events`, play prompt audio via `/api/ui/audio/...`, issue response-oriented commands, and end with state that matches a fresh snapshot. The verifier should run without a live microphone.

## Supporting checks

- `cargo test -p voice-daemon -p voice-protocol -p voice-ui`
- `cargo check -p voice-daemon`
- `npm run build` and `npm run check` in `crates/voice-ui`
- `cargo fmt --check`
- `git diff --check`
- PR #288 checks remain green.

## Iteration loop

1. Inspect the smallest daemon/UI seam that blocks the primary verifier.
2. Change one meaningful behavior: queue semantics, command execution, SSE state, audio serving, frontend bridge, or tests.
3. Run the narrowest failing verifier first, then the supporting checks before committing.
4. Record evidence and remaining gaps in `docs/daemon-ui-worklog.md`.
5. Keep the PR draft unless the completion proof is satisfied.

## Approval gates

- Ask before marking PR #288 ready for review.
- Ask before merging, force-pushing, deleting branches, or changing persistent user daemon files outside test-scoped paths.
- Ask before introducing a major new runtime dependency or replacing the daemon protocol outside the UI surface.

## Blocker standard

Only mark blocked after the same external condition prevents progress for three consecutive goal turns and no smaller local verifier or implementation slice remains. Missing live microphone access is not a blocker; use synthetic/file-based audio paths.

## Completion proof

Before marking the goal complete, record:

- The commit range and PR URL.
- Exact commands and passing outputs for all supporting checks.
- Evidence from the primary verifier, including request/track IDs and snapshot/event agreement.
- A short list of residual risks, if any.
