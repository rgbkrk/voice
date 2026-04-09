# Tauri Status Bar with Automerge State Sync Design

**Date:** 2026-04-09  
**Author:** Kyle Kelley (with Claude Code)  
**Status:** Daemon implementation complete (PR #68). Ready for Tauri app implementation.

## Implementation Status

✅ **Daemon-side complete** (merged in PR #68):
- Automerge CRDT state synchronization (`~/.voice/state.automerge`)
- Audio persistence (`~/.voice/audio/{queue_id}-{q|a}.wav`)
- Background cleanup task (30s auto-clear)
- RPC methods: `replay_audio`, `cancel_item`
- Queue state tracking: current, pending, recent with timestamps

🚧 **Tauri app pending**:
- macOS system tray application
- React frontend with voicemail-style UX
- File watcher for Automerge sync
- Unix socket client for daemon RPC

## Problem

The voice daemon currently operates as a "black box" - when multiple agents queue speech/questions simultaneously, users have no visibility into what's queued, who it's from, or the ability to manage responses. Users need a way to:
- See pending questions from agents
- Answer questions in their own time (voicemail-style UX)
- Replay audio (both questions and their own answers)
- Know which repo/project each question came from
- Pause/cancel items and switch between them

## Goals

1. **Voicemail UX** - Visual queue of pending questions, click to hear + answer
2. **System tray integration** - Lives in macOS menu bar, badge shows pending count
3. **Audio replay** - Play back questions and answers with transcripts
4. **Multi-source visibility** - See which repo/MCP client each item came from
5. **Automerge state sync** - Real-time state sync between daemon and UI
6. **Queue management** - Pause, cancel, reorder, auto-clear after 30s

## Non-Goals

- Cross-device sync (Automerge enables this later, but not MVP scope)
- Rich text editing of transcripts
- Audio speed controls (just play/pause for MVP)
- Custom notification sounds
- Keyboard shortcuts (mouse-driven UI for MVP)

## Architecture

### Automerge as State Sync Layer

The daemon maintains an **Automerge CRDT document** representing complete queue state. This follows the pattern from nteract's RuntimeStateDoc:
- Daemon is authoritative (owns the document)
- Tauri UI subscribes to changes (read-only with request actions)
- Real-time sync via file watcher (daemon writes doc to disk)
- Enables future multi-client scenarios without architecture changes

**Why Automerge vs polling:**
- Avoids polling lag and unnecessary RPC calls
- Proven pattern from nteract (already validated in production)
- Easier to add offline/multi-device support later
- Built-in conflict-free semantics for concurrent updates

### Component Architecture

```
┌─────────────────────────────────────────────────┐
│  Tauri UI (React + TypeScript)                  │
│  - System tray icon with badge                  │
│  - Dropdown panel with queue items              │
│  - Audio playback controls                      │
│  - Automerge state consumer                     │
└────────────┬────────────────────────────────────┘
             │ Tauri commands (#[tauri::command])
             │
┌────────────▼────────────────────────────────────┐
│  Tauri Backend (Rust)                           │
│  - Unix socket client to daemon                 │
│  - Automerge doc reader                         │
│  - Audio playback wrapper                       │
│  - File watcher for state changes               │
└────────────┬────────────────────────────────────┘
             │ JSON-RPC over Unix socket
             │
┌────────────▼────────────────────────────────────┐
│  Voice Daemon (voiced)                          │
│  - Automerge document owner                     │
│  - Audio recorder (questions + answers)         │
│  - Queue worker (existing)                      │
│  - Cleanup task (30s auto-clear)                │
└─────────────────────────────────────────────────┘
```

## Automerge Document Schema

The daemon maintains a single Automerge document at `~/.voice/state.automerge`:

```typescript
{
  status: "idle" | "speaking" | "listening" | "conversing" | "queued",
  current: QueueItem | null,
  pending: QueueItem[],
  recent: QueueItem[],
  audio: {
    [queue_id]: {
      question_path: string,      // ~/.voice/audio/{queue_id}-q.wav
      answer_path: string | null,  // ~/.voice/audio/{queue_id}-a.wav
      duration_ms: number
    }
  }
}

interface QueueItem {
  id: string,               // 8-char UUID prefix
  client_id: string,        // MCP session ID
  method: "speak" | "listen" | "converse",
  status: "Queued" | "Processing" | "Completed" | "Failed",
  created_at: number,       // unix timestamp (seconds)
  completed_at: number | null,
  text_preview: string | null,  // first 80 chars of text
  result: string | null,    // JSON with transcription/error
  repo: string | null,      // from MCP metadata (e.g., "rgbkrk/voice")
  auto_clear_at: number | null  // unix timestamp when to auto-remove
}
```

**Key additions beyond current daemon state:**
- `audio` map: Tracks audio file paths for replay
- `repo` field: Identifies which project/MCP client originated the request
- `auto_clear_at`: Enables 30-second auto-removal of answered items
- `completed_at`: Timestamp for when item finished (for auto-clear calculation)

**Type safety with automorph:**
Use `automorph` crate to generate Rust structs with `#[derive(Automorph)]` for type-safe Automerge operations.

```rust
use automorph::Automorph;

#[derive(Automorph, Clone, Debug)]
struct QueueItem {
    id: String,
    client_id: String,
    method: String,
    status: ItemStatus,
    created_at: u64,
    completed_at: Option<u64>,
    text_preview: Option<String>,
    result: Option<String>,
    repo: Option<String>,
    auto_clear_at: Option<u64>,
}

#[derive(Automorph, Clone, Debug)]
struct VoiceState {
    status: String,
    current: Option<QueueItem>,
    pending: Vec<QueueItem>,
    recent: Vec<QueueItem>,
    audio: HashMap<String, AudioInfo>,
}
```

## Audio Persistence Strategy

### Storage Layout

**Directory:** `~/.voice/audio/`

**File naming:**
- Question audio: `{queue_id}-q.wav` (what the agent said)
- Answer audio: `{queue_id}-a.wav` (what the user replied)

**Format:** 24kHz mono WAV (matches daemon's existing TTS output)

### Recording Points

1. **Question audio (converse speak phase)**
   - Worker captures rodio playback stream while speaking
   - Writes to `{queue_id}-q.wav`
   - Updates Automerge doc with path + duration

2. **Answer audio (converse listen phase)**
   - Worker captures microphone input while listening
   - Writes to `{queue_id}-a.wav`
   - Updates Automerge doc with path

3. **Speak-only requests**
   - Only record question audio (no answer needed)

### Cleanup Strategy

Background task runs every 10 seconds:
1. Check `recent` items where `auto_clear_at < now()`
2. Remove item from Automerge doc
3. Delete audio files: `{queue_id}-q.wav` and `{queue_id}-a.wav`
4. Keep recent list capped at 20 items (existing behavior)

### Replay Implementation

**Daemon method:** `replay_audio(queue_id: String, part: "question" | "answer") -> Result<()>`

1. Look up audio path in Automerge doc
2. Read WAV file from disk
3. Play through rodio (reuse existing playback code)
4. Return when playback completes

**Tauri calls this method** when user clicks "Play Question" or "Play Answer" button.

## Daemon Modifications

### ✅ Completed in PR #68

All daemon-side changes are implemented and merged:

**New Files (PR #68):**

**`crates/voice-daemon/src/automerge_state.rs`** ✓
- Maintains Automerge document with `VoiceState`, `AutomergeQueueItem`, `AudioInfo` types
- API: `new()`, `load_or_create()`, `update(&DaemonState)`, `remove_from_recent(id)`, `save()`
- Uses automorph 0.2 derive macros for type-safe CRDT operations
- Syncs on every queue change (enqueue, dequeue, complete, fail)

**`crates/voice-daemon/src/audio_recorder.rs`** ✓
- Saves TTS audio to WAV: `save_wav(path, samples, sample_rate)`
- Paths: `question_path(queue_id)` → `{queue_id}-q.wav`, `answer_path(queue_id)` → `{queue_id}-a.wav`
- Format: 32-bit float mono WAV at 24kHz
- Cleanup: `delete_audio(queue_id)` removes both q/a files

**`crates/voice-daemon/src/cleanup.rs`** ✓
- Background task runs every 10 seconds
- Checks `recent` items where `auto_clear_at <= now()`
- Deletes audio files via `audio_recorder::delete_audio()`
- Removes from Automerge doc and in-memory queue

**Modified Files (PR #68):**

**`crates/voice-daemon/src/worker.rs`** ✓
- Records TTS audio during speak phase (accumulates audio samples, saves after playback)
- Records microphone audio during listen phase (saves captured samples)
- Calls `sync_automerge()` after every queue state change (including Processing state)
- Sets `auto_clear_at` via `complete(result, Some(30))` for Converse requests

**`crates/voice-daemon/src/queue.rs`** ✓
- Added fields: `repo: Option<String>`, `completed_at: Option<u64>`, `auto_clear_at: Option<u64>`
- `complete()` accepts `auto_clear_secs: Option<u64>` for atomic timestamp setting
- `cancel_item(queue_id)` removes from pending or current, signals waiters
- `remove_recent(queue_id)` for cleanup task

**`crates/voice-daemon/src/socket.rs`** ✓
- `replay_audio(queue_id, part)` - reads WAV file, plays through rodio, returns duration
- `cancel_item(queue_id)` - removes from queue, updates Automerge, returns success bool
- Automerge state threaded through serve() → handle_client() → dispatch()

**`crates/voice-daemon/src/main.rs`** ✓
- Initializes Automerge doc via `AutomergeState::load_or_create()`
- Spawns cleanup task: `tokio::spawn(cleanup::run(queue, automerge))`
- Passes automerge to worker and socket handler

**Dependencies Added (PR #68):**

```toml
[dependencies]
automerge = "0.7"     # CRDT document (updated from 0.5 in spec)
automorph = "0.2"     # Derive macros for type-safe Automerge
```

**Tauri-side dependency:**

```toml
[dependencies]
notify = "6"          # File watcher for ~/.voice/state.automerge
```

## Tauri Application Architecture

### Project Structure

```
crates/voice-tray/
├── src-tauri/                  # Rust backend
│   ├── main.rs                 # Tauri app setup, system tray
│   ├── daemon.rs               # Unix socket client to voiced
│   ├── automerge.rs            # Automerge doc reader + file watcher
│   ├── audio.rs                # Audio playback controls
│   └── commands.rs             # Tauri commands for frontend
├── src/                        # Frontend (React + TypeScript)
│   ├── App.tsx                 # Main component, system tray panel
│   ├── QueuePanel.tsx          # Dropdown panel with queue list
│   ├── QueueItem.tsx           # Individual item component
│   ├── AudioControls.tsx       # Play/pause buttons
│   ├── store.ts                # Zustand store for Automerge state
│   └── types.ts                # TypeScript types (mirrors Rust)
├── public/
│   └── icon.png                # System tray icon
├── tauri.conf.json             # Tauri config (system tray settings)
├── package.json                # Frontend dependencies
└── Cargo.toml                  # Rust dependencies
```

### Tauri Backend Responsibilities

1. **Daemon connection** (`daemon.rs`)
   - Connect to `~/.voice/daemon.sock`
   - Send JSON-RPC commands: `replay_audio`, `cancel_item`
   - Handle connection errors, auto-reconnect

2. **Automerge sync** (`automerge.rs`)
   - Watch `~/.voice/state.automerge` with `notify` crate
   - Load document on change
   - Expose as reactive state to frontend

3. **Audio controls** (`audio.rs`)
   - Call daemon's `replay_audio` method
   - Track playback state (playing, paused, stopped)
   - Emit events to frontend for UI updates

4. **Tauri commands** (`commands.rs`)
   - `get_queue_state() -> VoiceState`
   - `play_question(queue_id: String)`
   - `play_answer(queue_id: String)`
   - `pause_audio()`
   - `answer_item(queue_id: String)` - triggers daemon converse flow
   - `cancel_item(queue_id: String)`

### Frontend (React + TypeScript)

**Component hierarchy:**
```
App
├── SystemTrayIcon (badge with count)
└── QueuePanel (dropdown, 280px wide)
    ├── PendingSection
    │   └── QueueItem[] (collapsed by default)
    ├── CurrentSection (if item is processing)
    │   └── QueueItem (expanded, shows progress)
    └── RecentSection (answered items, auto-clear 30s)
        └── QueueItem[] (collapsed by default)
```

**QueueItem component:**
- Collapsed: Shows preview text (50 chars), repo badge, timestamp
- Expanded: Full text, play/pause buttons, transcript (if available)
- Actions: "Answer" button (pending), "Play Question/Answer" buttons (recent)

**State management:**
- Use Zustand store
- Subscribe to Tauri events for Automerge updates
- Local UI state: expanded items, playback position

**Styling:**
- Tailwind CSS for rapid prototyping
- System-native look (match macOS menu bar dropdowns)
- Dark mode support

## Communication Protocol

### Daemon ← Tauri (JSON-RPC over Unix socket)

**Existing methods (no changes):**
- `speak(text, voice, speed, wait)` - Already implemented
- `converse(text, voice, wait)` - Already implemented
- `status()` - Returns queue state (already has current/pending/recent)

**New methods:**
```typescript
// Replay audio file
replay_audio(queue_id: string, part: "question" | "answer") -> Result<{duration_ms: number}>

// Cancel a queued or processing item
cancel_item(queue_id: string) -> Result<{cancelled: boolean}>
```

### Daemon → Tauri (Automerge file sync)

**File:** `~/.voice/state.automerge`

**Update flow:**
1. Daemon modifies Automerge doc (enqueue, complete, fail, etc.)
2. Daemon calls `automerge_state.save_to_disk()`
3. File watcher in Tauri detects change
4. Tauri loads new document
5. Tauri emits event to React frontend
6. UI re-renders with new state

**Why file-based vs socket streaming:**
- Simpler than maintaining WebSocket connection
- Automerge doc is small (<100KB even with full queue)
- File watcher is efficient (OS-level notifications)
- Matches nteract pattern (proven in production)
- Easy to inspect/debug (`automerge` CLI tools can read file)

### Tauri ← Frontend (Tauri commands)

**Commands exposed to React:**
```rust
#[tauri::command]
async fn get_queue_state() -> Result<VoiceState, String>

#[tauri::command]
async fn play_question(queue_id: String) -> Result<(), String>

#[tauri::command]
async fn play_answer(queue_id: String) -> Result<(), String>

#[tauri::command]
async fn pause_audio() -> Result<(), String>

#[tauri::command]
async fn answer_item(queue_id: String) -> Result<(), String>

#[tauri::command]
async fn cancel_item(queue_id: String) -> Result<(), String>
```

**Events emitted to React:**
```typescript
// Fired when Automerge doc changes
event: "queue-updated", payload: VoiceState

// Fired when audio playback state changes
event: "audio-state", payload: { playing: boolean, queue_id: string | null }
```

## User Flows

### Flow 1: Answer a Pending Question

**Scenario:** Agent asks "What should the function name be?" via MCP converse tool

1. **Daemon queues the converse request**
   - Worker plays question audio, records to `{queue_id}-q.wav`
   - But agent specified `wait: false` (fire-and-forget)
   - Item goes to pending, updates Automerge doc

2. **User sees notification**
   - System tray icon badge shows "1"
   - User clicks tray icon
   - Dropdown panel opens, shows pending item:
     ```
     📍 rgbkrk/voice
     "What should the function name be?"
     2s ago
     ```

3. **User clicks "Answer" button**
   - Tauri sends `answer_item(queue_id)` command
   - Daemon plays question audio (user hears it again)
   - After question, daemon plays "ding" (ready to listen)
   - Daemon starts recording microphone
   - User speaks: "Let's call it process_queue_item"
   - After silence timeout, daemon plays "dong" (done listening)

4. **Daemon processes response**
   - Transcribes audio to text
   - Saves answer audio to `{queue_id}-a.wav`
   - Sets `result = {"text": "Let's call it process_queue_item", ...}`
   - Sets `status = Completed`, `completed_at = now()`, `auto_clear_at = now() + 30`
   - Updates Automerge doc

5. **Item moves to Recent section**
   - UI shows item in "Recent" with checkmark
   - Badge count decrements
   - After 30s, cleanup task removes item + audio files

### Flow 2: Replay Previous Audio

**Scenario:** User wants to hear their own response again to verify what they said

1. **User opens tray panel**
   - "Recent" section shows answered items (within 30s window)
   - Item shows: preview, timestamp, checkmark (completed)

2. **User clicks item to expand**
   - Shows full text preview
   - Shows transcript of answer
   - Two buttons: "Play Question" | "Play Answer"

3. **User clicks "Play Answer"**
   - Tauri sends `play_answer(queue_id)` command
   - Daemon reads `{queue_id}-a.wav` from disk
   - Plays audio through speakers
   - UI shows pulsing indicator during playback

4. **User can read transcript while listening**
   - Transcript displayed below audio controls
   - Scrollable if text is long

### Flow 3: Pause and Switch Items

**Scenario:** Multiple agents ask questions simultaneously, user wants to prioritize

1. **Three items in pending queue**
   - Item A: "What's the return type?" (from voice)
   - Item B: "Should we add tests?" (from other-project)
   - Item C: "Approve this PR?" (from third-project)

2. **User clicks Answer on Item A**
   - Question starts playing
   - UI shows "Playing..." indicator on Item A

3. **User realizes Item C is urgent**
   - Clicks "Pause" button on Item A
   - Daemon stops playback immediately
   - Item A returns to pending (status remains Queued)

4. **User clicks Answer on Item C**
   - Item C's question plays + listens
   - User answers, item completes

5. **User returns to Item A later**
   - Clicks Answer again
   - Question replays from beginning
   - User can now answer it

### Flow 4: Audio File Missing

**Scenario:** Audio file was deleted manually or cleanup race condition

1. **User clicks "Play Question"**
   - Tauri sends `play_question(queue_id)`
   - Daemon tries to read `{queue_id}-q.wav`
   - File not found error

2. **Daemon returns error**
   - JSON-RPC error response: `{"code": -32000, "message": "Audio file not found"}`

3. **Tauri shows fallback UI**
   - Toast notification: "Audio unavailable"
   - "Play Question" button disabled
   - Transcript still visible (read-only)

## Data Flow

### Enqueue Flow (Agent → Daemon → UI)

```
Agent (MCP)
  → calls converse(text, voice, wait=false)
  → Daemon receives request
  → Worker starts processing (plays question, records to WAV)
  → Queue adds to pending, updates Automerge doc
  → Automerge doc saved to disk
  → Tauri file watcher detects change
  → React UI re-renders with new pending item
  → Badge count increments
```

### Answer Flow (User → UI → Daemon → Agent)

```
User clicks "Answer"
  → React calls Tauri command: answer_item(queue_id)
  → Tauri calls daemon: converse_resume(queue_id)
  → Daemon plays question audio
  → Daemon listens for user response
  → Daemon transcribes, saves answer audio
  → Daemon updates queue entry: status=Completed, result=transcript
  → Automerge doc updated, saved to disk
  → Tauri detects change, notifies React
  → Item moves from Pending to Recent
  → Auto-clear timer starts (30s)
```

### Cleanup Flow (Daemon background task)

```
Cleanup task wakes (every 10s)
  → Iterate recent items
  → Find items where auto_clear_at < now()
  → For each expired item:
    - Delete {queue_id}-q.wav
    - Delete {queue_id}-a.wav
    - Remove from Automerge doc recent array
  → Save Automerge doc
  → Tauri detects change, UI updates
```

## Error Handling

### Daemon Connection Lost

**Detection:** Tauri's Unix socket connection fails or times out

**UI Response:**
- Show "Disconnected" banner in tray panel
- Badge shows last known pending count (grayed out)
- Disable all action buttons
- Show last known Automerge state (read-only)

**Recovery:**
- Auto-reconnect every 5 seconds
- When reconnected, load latest Automerge doc
- Resume normal operation

### Audio File Missing

**Detection:** Daemon's `replay_audio` returns file-not-found error

**UI Response:**
- Toast notification: "Audio unavailable for this item"
- Disable play buttons for this item
- Keep transcript visible
- Item can still be auto-cleared after 30s

### Transcription Failed

**Detection:** Worker's STT call fails (model error, no speech detected, etc.)

**Daemon Response:**
- Mark item status as Failed
- Set result = JSON error: `{"error": "Transcription failed: <reason>"}`
- Still record answer audio (in case user wants to replay it)
- Update Automerge doc

**UI Response:**
- Show item in Recent with ⚠️ icon
- Display error message
- "Retry" button to answer again
- Still allow playing question/answer audio

### Microphone Access Denied

**Detection:** Tauri/daemon can't open microphone (permission denied)

**Initial handling:**
- When daemon starts, check mic permissions
- If denied, show macOS permission prompt

**Runtime handling:**
- If permission lost mid-session, fail gracefully
- Show error in UI: "Microphone access required"
- Link to System Preferences → Privacy → Microphone

**Fallback option (future):**
- Text input field to type answer instead of speaking
- Not in MVP scope, but architecture supports it

### Auto-Clear Race Condition

**Scenario:** User clicks item just as cleanup task removes it

**Protection:**
1. Tauri loads Automerge doc on every action
2. If queue_id not found, show toast: "Item no longer available"
3. UI removes item from display
4. No crash, graceful degradation

### Automerge Conflict

**Scenario:** File modified externally while daemon has unsaved changes

**Resolution:**
- Shouldn't happen (daemon is sole writer)
- If it does, Automerge automatically merges
- Last-write-wins semantics for queue operations

## Testing Strategy

### Unit Tests

**Daemon side:**
- Audio recorder: write test WAV, verify format/duration
- Cleanup task: mock time, verify expired items removed
- Automerge integration: verify doc mutations produce correct state

**Tauri side:**
- Daemon client: mock Unix socket, verify RPC calls
- File watcher: mock file changes, verify state updates
- Command handlers: verify error handling

### Integration Tests

1. **End-to-end flow test:**
   - Start daemon + Tauri app
   - Enqueue converse via daemon socket
   - Verify item appears in UI
   - Call answer_item, verify audio plays + records
   - Verify item completes and auto-clears

2. **Reconnection test:**
   - Start daemon + Tauri
   - Kill daemon
   - Verify UI shows disconnected state
   - Restart daemon
   - Verify UI reconnects and syncs state

3. **Audio replay test:**
   - Complete a converse item
   - Call play_question and play_answer
   - Verify correct audio files play

### Manual Testing

**User flows to validate:**
- ✅ Answer pending question (happy path)
- ✅ Replay previous audio
- ✅ Pause and switch between items
- ✅ Auto-clear after 30 seconds
- ✅ Multiple items from different repos
- ✅ Badge count accuracy
- ✅ Audio file missing (delete manually, verify fallback)
- ✅ Microphone permission denied

## Future Enhancements (Not MVP)

**Phase 2:**
- Drag to reorder pending items
- Audio speed controls (0.5x, 1x, 1.5x, 2x)
- Keyboard shortcuts (Space = play/pause, Esc = close panel)
- Text input fallback (type answer instead of speaking)
- Rich transcript display (highlight keywords, timestamps)

**Phase 3:**
- Multi-device sync (Automerge enables this naturally)
- Custom notification sounds per repo
- Snooze items (defer to later, like email snooze)
- Search/filter queue items
- Export transcript to clipboard

**Phase 4:**
- Voice commands ("Skip", "Repeat", "Answer later")
- Agent response templates (canned responses for common questions)
- Integration with issue trackers (link queue item to GitHub issue)

## Open Questions

None - architecture is finalized.

## Summary

This design builds a Tauri system tray app that gives users voicemail-style control over the voice daemon's queue. By using Automerge as the state sync layer, we get real-time updates, proven architecture patterns from nteract, and a foundation for future multi-device scenarios. The 30-second auto-clear, audio replay, and repo visibility features address the core problems of managing multiple agent conversations simultaneously.

**Key innovations:**
- Automerge CRDT for state sync (proven pattern, future-proof)
- Audio persistence for replay (questions + answers)
- Voicemail UX (answer at your own pace)
- Multi-source visibility (see which repo/project per item)
- Graceful error handling (missing audio, disconnected daemon)

**Implementation approach:**
- Daemon changes: Add Automerge integration, audio recording, cleanup task
- New Tauri app: System tray UI, Automerge file watcher, Unix socket client
- Minimal RPC additions: replay_audio, cancel_item methods

This architecture enables the user to manage multi-agent conversations without feeling overwhelmed, with full control and visibility into the queue.
