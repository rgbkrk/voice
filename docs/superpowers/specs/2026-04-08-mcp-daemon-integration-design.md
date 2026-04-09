# MCP Daemon Integration Design

**Date:** 2026-04-08  
**Author:** Kyle Kelley (with Claude Code)

## Problem

The voice MCP server currently does TTS/STT directly in-process. When multiple Claude Code sessions use the voice MCP tools simultaneously, their audio overlaps and creates chaos. The voice daemon was built to serialize audio operations through a shared queue, but the MCP server doesn't use it yet.

## Goals

1. Make the MCP server prefer the daemon when available (serialize audio across multiple agent sessions)
2. Fall back gracefully to direct TTS/STT when the daemon isn't running
3. Handle daemon restarts during development without requiring MCP server restart
4. Maintain per-agent voice/speed preferences while letting the daemon manage the global queue

## Non-Goals

- Persistent connections between MCP server and daemon
- Exposing queue management to agents (that's for the future Tauri UI)
- Changing the MCP tool signatures (transparent backend swap)

## Architecture

### Connection Strategy

**Per-call daemon detection with fallback:**

On every `speak` and `converse` call:
1. Try to connect to `~/.voice/daemon.sock`
2. If connection succeeds, forward the request to daemon
3. If connection fails (socket missing or daemon down), fall back to direct TTS/STT

This approach handles daemon lifecycle gracefully:
- Daemon starts after MCP server → next call automatically uses daemon
- Daemon crashes → immediate fallback to direct mode
- No stale connection state to manage

**Why per-call instead of startup-time check?**  
During development, the daemon frequently restarts. Per-call detection ensures the MCP server always uses the daemon when available without requiring MCP server restart. The connection overhead is minimal (Unix domain socket on localhost).

### Method Behavior

#### `speak(text, voice?, speed?, markdown?)`

**Daemon mode:**
- Send JSON-RPC request: `{"method": "speak", "params": {"text": "...", "voice": "...", "speed": 1.0, "wait": false}}`
- Return immediately with: `{"queue_id": "abc123", "status": "queued"}`
- Agent gets queue_id but doesn't wait for audio completion

**Direct mode:**
- Run TTS in-process (current behavior)
- Return after completion: `{"duration_ms": 1520, "chunks": 1}`

**Rationale for fire-and-forget:**  
`speak` is used for agent narration. The agent doesn't need to wait for audio to finish - it just wants to queue speech and continue working. The daemon handles playback asynchronously.

#### `converse(text, voice?, speed?)`

**Daemon mode:**
- Send JSON-RPC request: `{"method": "converse", "params": {"text": "...", "voice": "...", "wait": true}}`
- Block until daemon responds with: `{"spoke": {"duration_ms": 1520, "chunks": 1}, "heard": {"text": "user reply", "tokens": 5, "duration_ms": 3200}}`

**Direct mode:**
- Run TTS then STT in-process (current behavior)
- Return combined result with same structure

**Rationale for wait:**  
`converse` is a question-and-answer exchange. The agent needs the user's transcribed reply to continue, so it must wait for the full cycle (speak → listen → transcribe) to complete.

#### Removed: `listen`

Raw `listen` without context creates "hot mic" moments where the user doesn't know why recording started. It's not useful for agents. Remove it entirely from MCP tools. Users who need standalone transcription can use `voice listen` from the CLI.

#### Removed: `cancel`

Without persistent session state, `cancel` has no clear target. The daemon's `cancel` method cancels queued items for a specific client_id, but each MCP call is a fresh connection with a new client_id. Remove this tool. Cancellation will be exposed through the Tauri UI later.

### Configuration Tools

**`set_voice(voice)`**, **`set_speed(speed)`**, **`list_voices()`**

These remain **local to the MCP server instance**. Each agent session (Claude Code session) maintains its own voice/speed preferences. When forwarding `speak` or `converse` to the daemon, the MCP server includes these as request parameters.

**Why local?**  
- Each agent session can have its own voice identity (e.g., one agent uses "am_adam", another uses "af_bella")
- No need for daemon to track per-client preferences
- Simpler state management

The daemon has its own global defaults for when clients don't specify voice/speed, but per-request parameters always take precedence.

## Implementation

### Code Structure

**New module:** `crates/voice-cli/src/mcp_daemon.rs`

Handles daemon communication:
- `try_connect_daemon() -> Option<UnixStream>` - attempt socket connection
- `daemon_speak(stream: UnixStream, text: &str, voice: Option<&str>, speed: Option<f32>) -> Result<Value>` - forward speak (wait=false)
- `daemon_converse(stream: UnixStream, text: &str, voice: Option<&str>, speed: Option<f32>) -> Result<Value>` - forward converse (wait=true)
- Uses `voice-protocol` frame codec to speak the daemon's protocol

**Modified:** `crates/voice-cli/src/mcp.rs`

Update tool handlers:
- `handle_speak()` - try daemon first, fallback to direct
- `handle_converse()` - try daemon first, fallback to direct
- Remove `handle_listen()` and `handle_cancel()`
- Keep `handle_set_voice()`, `handle_set_speed()`, `handle_list_voices()` as-is (local state)

### Dependencies

Add to `crates/voice-cli/Cargo.toml`:
```toml
voice-protocol = { path = "../voice-protocol" }
```

This gives the MCP server access to:
- `voice_protocol::frames::{read_frame, write_frame, Frame, FrameType}`
- `voice_protocol::rpc::{Request, Response, ...}`

### Connection Pattern

```rust
// Pseudocode for the daemon-first pattern
fn handle_speak(params: SpeakParams) -> Result<Value> {
    if let Some(stream) = try_connect_daemon() {
        eprintln!("mcp: using daemon");
        return daemon_speak(stream, &params.text, params.voice, params.speed);
    }
    
    eprintln!("mcp: daemon unavailable, using direct mode");
    // Current direct TTS implementation
    direct_speak(params)
}
```

Short-lived connections: connect, send frame, read response, close. No connection pooling or persistent state.

### Frame Protocol

The daemon expects:
1. **Request frame:** `Frame::request(json_rpc_bytes)` where JSON-RPC is `{"jsonrpc": "2.0", "method": "speak", "params": {...}, "id": 1}`
2. **Response frame:** `Frame::response(json_rpc_bytes)` where JSON-RPC is `{"jsonrpc": "2.0", "result": {...}, "id": 1}`

Frames are length-prefixed:
```
[4 bytes: frame_type][4 bytes: payload_length][N bytes: payload]
```

This is already implemented in `voice-protocol::frames` - just call `write_frame()` and `read_frame()`.

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Socket doesn't exist | Silent fallback to direct mode |
| Connection fails (daemon crashed) | Silent fallback to direct mode |
| Daemon returns JSON-RPC error | Forward error to MCP client |
| Direct mode fails | Return error to MCP client (same as current) |
| Invalid request | Return JSON-RPC error to MCP client |

**Stderr logging for debugging:**
- `"mcp: using daemon"` when socket connects
- `"mcp: daemon unavailable, using direct mode"` when falling back
- Existing TTS/STT progress output remains unchanged

## Future Extensions

**Q&A Voicemail Queue (not in this design):**

Later, we could add a `speak_and_wait_for_reply(text, timeout_ms)` tool that:
1. Queues speech to daemon
2. Returns a queue_id immediately
3. Agent can later call `get_reply(queue_id)` to check if user responded
4. Tauri UI shows pending questions, user clicks to play audio + reply via mic
5. Agent polls until reply appears or timeout expires

This turns the queue into an async Q&A system where multiple agents can ask questions and the user responds at their own pace. Not implementing this now - just noting the future direction.

**Status visibility:**

The daemon's `status` method returns `{current, pending, recent}` queue state. We could expose this as an MCP tool later so agents can see the queue, but it's not essential for v1.

## Testing Plan

1. **Manual testing:** Start daemon, use MCP tools from Claude Code, verify audio serializes
2. **Daemon down:** Stop daemon, verify MCP falls back to direct mode
3. **Daemon restart:** Stop daemon, use MCP (direct mode), start daemon, use MCP again (should switch to daemon)
4. **Multiple sessions:** Open two Claude Code sessions, have both use voice tools simultaneously, verify serialization
5. **Voice preferences:** Set different voices in two sessions, verify each agent's voice persists across calls

## Open Questions

None - design is complete and approved.

## Summary

This design makes the MCP server daemon-aware while maintaining backward compatibility. Agents get automatic audio serialization when the daemon runs, with zero-config fallback to direct mode when it doesn't. Per-call connection detection handles daemon restarts gracefully during development. Voice/speed preferences stay local to each agent session for identity separation.
