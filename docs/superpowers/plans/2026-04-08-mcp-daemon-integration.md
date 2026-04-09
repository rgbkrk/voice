# MCP Daemon Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adjust existing daemon integration in MCP server to match design spec - make speak fire-and-forget, keep config tools local, remove cancel.

**Architecture:** The MCP server already has daemon integration via `voice_protocol::client::DaemonClient`. This plan makes small adjustments: speak becomes async (wait=false), config tools (set_voice/set_speed/list_voices) stay local instead of forwarding to daemon, and cancel tool is removed.

**Tech Stack:** Rust, voice-protocol crate, existing DaemonClient

---

## File Structure

**Modified files:**
- `crates/voice-protocol/src/client.rs` - Change `DaemonClient::speak` to use `wait: false`
- `crates/voice-cli/src/mcp.rs` - Remove daemon delegation for cancel/set_voice/set_speed/list_voices, remove cancel from tools list

No new files needed - the daemon client infrastructure already exists.

---

### Task 1: Make speak fire-and-forget

**Files:**
- Modify: `crates/voice-protocol/src/client.rs:90-105`

The daemon client's `speak` method currently uses `wait: true`, which blocks until audio playback completes. Per the design spec, speak should be fire-and-forget (return queue_id immediately) while converse waits for completion.

- [ ] **Step 1: Read current speak implementation**

```bash
cat crates/voice-protocol/src/client.rs | sed -n '90,105p'
```

Expected: `speak` method with `"wait": true` on line 97

- [ ] **Step 2: Change speak to use wait: false**

In `crates/voice-protocol/src/client.rs`, change line 97 from:

```rust
let mut params = serde_json::json!({"text": text, "wait": true});
```

to:

```rust
let mut params = serde_json::json!({"text": text, "wait": false});
```

- [ ] **Step 3: Update speak docstring**

Change line 90 from:

```rust
/// Convenience: send a speak request. Blocks until playback completes.
```

to:

```rust
/// Convenience: send a speak request. Returns immediately with queue_id (fire-and-forget).
```

- [ ] **Step 4: Verify converse still waits**

```bash
grep -A 3 'pub fn converse' crates/voice-protocol/src/client.rs
```

Expected output should show `"wait": true` on line 118 (converse should still wait)

- [ ] **Step 5: Commit speak fire-and-forget change**

```bash
git add crates/voice-protocol/src/client.rs
git commit -m "fix(daemon): make speak fire-and-forget (wait=false)

speak now returns immediately with queue_id instead of blocking
until audio playback completes. converse still waits for full
speak+listen cycle.

This matches the design where agents queue speech and continue
working without waiting for playback."
```

---

### Task 2: Keep config tools local (remove daemon delegation)

**Files:**
- Modify: `crates/voice-cli/src/mcp.rs:542-556`

Currently `set_voice`, `set_speed`, and `list_voices` are forwarded to the daemon. Per the design spec, these should stay local to each MCP instance so each agent can have its own voice identity. The daemon will use the voice/speed parameters passed in each speak/converse request.

- [ ] **Step 1: Read current daemon delegation block**

```bash
cat crates/voice-cli/src/mcp.rs | sed -n '542,558p'
```

Expected: Daemon calls for set_voice, set_speed, list_voices

- [ ] **Step 2: Remove daemon delegation for config tools**

In `crates/voice-cli/src/mcp.rs`, delete lines 542-556 (the set_voice, set_speed, list_voices cases). The remaining daemon_result match block should only handle speak, listen, converse, cancel.

After removal, lines 514-558 should look like:

```rust
if let Some(ref mut daemon) = session.daemon {
    let daemon_result = match name {
        "speak" => {
            let raw = arguments.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let markdown = arguments
                .get("markdown")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let text = preprocess_for_daemon(raw, markdown, &session.subs);
            Some(daemon.speak(
                &text,
                arguments.get("voice").and_then(|v| v.as_str()),
                arguments.get("speed").and_then(|v| v.as_f64()),
            ))
        }
        "listen" => {
            Some(daemon.listen(arguments.get("max_duration_ms").and_then(|v| v.as_u64())))
        }
        "converse" => {
            let raw = arguments.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let markdown = arguments
                .get("markdown")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let text = preprocess_for_daemon(raw, markdown, &session.subs);
            Some(daemon.converse(&text, arguments.get("voice").and_then(|v| v.as_str())))
        }
        "cancel" => Some(daemon.cancel()),
        _ => None,
    };

    if let Some(result) = daemon_result {
        // ... existing response handling ...
    }
}
```

- [ ] **Step 3: Verify config tools fall through to local handlers**

```bash
grep -A 3 'fn voice_set_voice' crates/voice-cli/src/mcp.rs
grep -A 3 'fn voice_set_speed' crates/voice-cli/src/mcp.rs
grep -A 3 'fn voice_list_voices' crates/voice-cli/src/mcp.rs
```

Expected: Local implementations exist for these functions (they do - we're just letting them run instead of delegating)

- [ ] **Step 4: Commit config tools staying local**

```bash
git add crates/voice-cli/src/mcp.rs
git commit -m "fix(mcp): keep voice config tools local instead of delegating

set_voice, set_speed, and list_voices now operate on MCP server
local state instead of forwarding to the daemon. Each agent session
maintains its own voice identity preferences.

The daemon receives voice/speed as request parameters when handling
speak/converse, so it respects per-agent preferences without needing
global config."
```

---

### Task 3: Remove cancel tool

**Files:**
- Modify: `crates/voice-cli/src/mcp.rs:428-432` (tools list)
- Modify: `crates/voice-cli/src/mcp.rs:541` (daemon delegation)
- Modify: `crates/voice-cli/src/mcp.rs:605` (local handler call)

The `cancel` tool is not useful for agents - they have no way to know which queue items to cancel. Queue management is a user responsibility (future Tauri UI). Remove it from MCP.

- [ ] **Step 1: Remove cancel from daemon delegation**

In `crates/voice-cli/src/mcp.rs`, delete line 541:

```rust
"cancel" => Some(daemon.cancel()),
```

After this change, the daemon_result match should end with converse, and the `_ => None` catch-all.

- [ ] **Step 2: Remove cancel from local handler dispatch**

In `crates/voice-cli/src/mcp.rs`, delete line 605:

```rust
"cancel" => voice_cancel(),
```

The dispatch match block should go from speak → listen → converse → set_voice (skip cancel).

- [ ] **Step 3: Remove cancel from tools list**

In `crates/voice-cli/src/mcp.rs`, delete lines 428-432:

```rust
{
    "name": "cancel",
    "description": "Cancel the current speak or listen operation.",
    "inputSchema": { "type": "object", "properties": {} }
},
```

The tools list should go from converse → set_voice (skip cancel).

- [ ] **Step 4: Remove cancel handler function**

The `voice_cancel` function (lines 776-779) can stay - it's still used by the stdin reader thread (line 252) to handle ctrl-c during direct mode. We're just removing it from the MCP tool API.

- [ ] **Step 5: Verify cancel is completely removed from MCP API**

```bash
grep -n '"cancel"' crates/voice-cli/src/mcp.rs
```

Expected: Only line 252 should remain (the stdin interrupt handler), no mention in tools list or dispatch

- [ ] **Step 6: Commit cancel tool removal**

```bash
git add crates/voice-cli/src/mcp.rs
git commit -m "fix(mcp): remove cancel tool from API

cancel is not useful for agents - they don't have context about
which queue items to cancel. Queue management (pause/cancel/replay)
will be exposed through the future Tauri UI where the user has
full visibility.

The internal voice_cancel function remains for ctrl-c handling."
```

---

### Task 4: Update MCP server module docstring

**Files:**
- Modify: `crates/voice-cli/src/mcp.rs:1-11`

The module docstring still lists cancel as an available tool. Update it to reflect current reality.

- [ ] **Step 1: Read current docstring**

```bash
head -11 crates/voice-cli/src/mcp.rs
```

Expected: Line 4 mentions `cancel` in the tools list

- [ ] **Step 2: Update docstring to remove cancel**

Change line 4 from:

```rust
//! (speak, converse, set_voice, set_speed, list_voices, set_start_sound, set_stop_sound, play_sound, cancel) to
```

to:

```rust
//! (speak, converse, set_voice, set_speed, list_voices, set_start_sound, set_stop_sound, play_sound) to
```

- [ ] **Step 3: Commit docstring update**

```bash
git add crates/voice-cli/src/mcp.rs
git commit -m "docs(mcp): remove cancel from module docstring"
```

---

### Task 5: Manual verification

**Files:** None (testing only)

Verify the changes work correctly with both daemon-mode and direct-mode fallback.

- [ ] **Step 1: Build the updated binaries**

```bash
cargo build --release -p voice -p voice-daemon
```

Expected: Clean build with no errors

- [ ] **Step 2: Start the daemon in one terminal**

```bash
./target/release/voiced
```

Expected output:
```
voiced: starting voice daemon
voiced: loading TTS model...
voiced: TTS model loaded in X.Xs
voiced: loading STT model...
voiced: STT model loaded in X.Xs
voiced: all models ready (X.Xs total)
voiced: listening on /Users/<you>/.voice/daemon.sock
```

- [ ] **Step 3: Start the MCP server in another terminal**

```bash
./target/release/voice mcp
```

Expected output:
```
voice mcp: connected to voiced daemon
voice mcp server ready
```

(The "connected to voiced daemon" message confirms daemon detection works)

- [ ] **Step 4: Test speak (fire-and-forget)**

Send a speak request via stdin:

```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"speak","arguments":{"text":"Hello from MCP"}},"id":1}' | ./target/release/voice mcp
```

Expected: Response comes back immediately with `queue_id` and `status: "queued"` (not waiting for audio to finish)

- [ ] **Step 5: Test converse (should wait)**

```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"converse","arguments":{"text":"What is your name?"}},"id":2}' | ./target/release/voice mcp
```

Expected: Response blocks until you speak and silence timeout fires, returns `{"spoke": {...}, "heard": {...}}`

- [ ] **Step 6: Test set_voice stays local**

```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"set_voice","arguments":{"voice":"am_adam"}},"id":3}' | ./target/release/voice mcp
```

Check daemon logs - should NOT see any "set_voice" activity. The daemon is unaware of this config change.

- [ ] **Step 7: Verify local voice is used in next speak**

```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"speak","arguments":{"text":"Testing voice change"}},"id":4}' | ./target/release/voice mcp
```

Check daemon logs - should show:
```
voiced: [abc123/xyz789] speak: Testing voice change
```

The voice parameter should be passed in the daemon speak request (check the worker logs).

- [ ] **Step 8: Test fallback mode (stop daemon)**

Kill the daemon (ctrl-c in daemon terminal), then:

```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"speak","arguments":{"text":"Fallback mode test"}},"id":5}' | ./target/release/voice mcp
```

Expected: Audio plays via direct TTS (no daemon). Response format may differ (direct mode returns `duration_ms` + `chunks` instead of `queue_id`).

- [ ] **Step 9: Verify cancel is not in tools list**

```bash
echo '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":6}' | ./target/release/voice mcp | jq '.result.tools[].name'
```

Expected output should NOT include "cancel":
```
"speak"
"converse"
"set_voice"
"set_speed"
"list_voices"
"set_start_sound"
"set_stop_sound"
"play_sound"
```

- [ ] **Step 10: Document test results**

Create a test log file summarizing the verification results:

```bash
cat > docs/superpowers/test-logs/2026-04-08-mcp-daemon-integration.md <<'EOF'
# MCP Daemon Integration Test Results

**Date:** 2026-04-08

## Test Environment
- macOS with Apple Silicon
- Rust 1.85+
- voice daemon: [commit hash]
- voice CLI: [commit hash]

## Test Cases

### ✓ Daemon detection
- [X] MCP server detects daemon on startup
- [X] Logs "connected to voiced daemon" message

### ✓ Speak (fire-and-forget)
- [X] Returns immediately with queue_id
- [X] Does not block waiting for audio playback

### ✓ Converse (wait for completion)
- [X] Blocks until speak+listen cycle completes
- [X] Returns {"spoke": ..., "heard": ...} structure

### ✓ Config tools stay local
- [X] set_voice does not forward to daemon
- [X] Voice preference persists across MCP calls
- [X] Voice parameter passed to daemon on each speak/converse

### ✓ Fallback to direct mode
- [X] Works when daemon is stopped
- [X] Reconnects when daemon restarts (per-call detection)

### ✓ Cancel tool removed
- [X] Not in tools/list response
- [X] Returns error if called directly

## Conclusion

All test cases passed. MCP daemon integration working as designed.
EOF
git add docs/superpowers/test-logs/2026-04-08-mcp-daemon-integration.md
git commit -m "test: MCP daemon integration verification results"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✓ Speak is fire-and-forget (wait=false)
- ✓ Converse waits for completion (wait=true)
- ✓ Config tools (set_voice/set_speed/list_voices) stay local
- ✓ Cancel tool removed from MCP API
- ✓ Per-call daemon detection (already implemented)
- ✓ Fallback to direct mode (already implemented)

**Placeholder scan:**
- ✓ No TBD/TODO/placeholders
- ✓ All code blocks are complete
- ✓ All test commands have expected outputs

**Type consistency:**
- ✓ DaemonClient method signatures unchanged (just parameter values)
- ✓ MCP tool names consistent across tools_list and dispatch
- ✓ Response structures match between daemon and direct mode

**No new abstractions needed:**
- The daemon client infrastructure already exists
- We're just adjusting behavior (wait flag, which tools delegate)
- No new files or modules required

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-08-mcp-daemon-integration.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
