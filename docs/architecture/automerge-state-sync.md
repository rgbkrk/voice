# Automerge State Synchronization

## Overview

The voice daemon uses [Automerge](https://automerge.org/) CRDT to synchronize queue state between the daemon process and the Tauri tray app. This enables real-time UI updates without polling or complex pub/sub infrastructure.

## Architecture

```
┌─────────────────┐         ┌──────────────────────┐         ┌────────────────┐
│  Voice Daemon   │         │  ~/.voice/           │         │  Tauri Tray    │
│  (Worker)       │────────>│  state.automerge     │<────────│  (File Watch)  │
└─────────────────┘ writes  └──────────────────────┘  reads  └────────────────┘
                              (Automerge CRDT doc)
```

### Data Flow

1. **Daemon writes** — Worker updates Automerge document after every queue state change
2. **File save** — Automerge document saved to `~/.voice/state.automerge`
3. **File watch** — Tray app uses `notify` to watch for file changes
4. **Tray reads** — On change event, tray loads latest Automerge document
5. **UI update** — RxJS observable emits new state to React components

## Components

### Daemon: `voice-daemon/src/automerge_state.rs`

**`AutomergeState`** wraps an `AutoCommit` document with file I/O:

```rust
pub struct AutomergeState {
    doc: AutoCommit,
    path: PathBuf,  // ~/.voice/state.automerge
}
```

**State schema** using `automorph` derives:

```rust
#[derive(Automorph)]
pub struct VoiceState {
    pub status: String,
    pub current: Option<AutomergeQueueItem>,
    pub pending: Vec<AutomergeQueueItem>,
    pub recent: Vec<AutomergeQueueItem>,
    pub audio: HashMap<String, AudioInfo>,
}
```

**Write operations:**

- `update(&mut self, state: &DaemonState)` — Replace entire state (called after queue changes)
- `remove_from_recent(&mut self, queue_id)` — Remove specific item (called by cleanup task)
- `save(&mut self) -> Result<(), String>` — Write document to disk

### Daemon: `voice-daemon/src/worker.rs`

**Sync function:**

```rust
async fn sync_automerge(
    queue: &RequestQueue,
    automerge: &Arc<tokio::sync::Mutex<AutomergeState>>,
) {
    let snapshot = queue.snapshot().await;
    let mut am = automerge.lock().await;
    am.update(&snapshot);
    if let Err(e) = am.save() {
        eprintln!("voiced: failed to save automerge doc: {}", e);
    }
}
```

**Called at:**

1. When request starts processing (`dequeue()`)
2. After request completes (success or failure)
3. When item is canceled (`cancel_item` RPC)
4. When cleanup task removes old items

### Tray App: `voice-tray/src-tauri/automerge_sync.rs`

**`FileWatcher`** uses tokio + notify for async file watching:

```rust
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    rx: mpsc::UnboundedReceiver<Result<Event, notify::Error>>,
    path: PathBuf,
}
```

**Key methods:**

- `wait_for_change(&mut self, timeout: Duration) -> Option<VoiceState>` — Async wait for file change
- `load_current(&self) -> Result<VoiceState, String>` — Immediate state load (for initial state)

**Event filtering:**

```rust
fn is_state_file_event(&self, event: &Event) -> bool {
    event.paths.iter().any(|p| {
        p.file_name() == self.path.file_name()
            && matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_))
    })
}
```

### Tray App: `voice-tray/src-tauri/main.rs`

**Spawn file watcher task:**

```rust
tauri::async_runtime::spawn(async move {
    let mut watcher = FileWatcher::new()?;
    
    // Load initial state
    let initial_state = watcher.load_current()?;
    app_handle.emit("queue-updated", initial_state)?;
    
    // Watch for changes
    loop {
        if let Some(new_state) = watcher.wait_for_change(Duration::from_secs(1)).await {
            watcher_state.update_voice_state(new_state.clone());
            update_tray_badge(&tray_app_handle, &new_state);
            app_handle.emit("queue-updated", new_state)?;
        }
    }
});
```

### Tray App: `voice-tray/src/store/index.ts`

**Zustand store** with RxJS observable for Tauri events:

```typescript
subscribeToUpdates: () => {
  const queueUpdates$ = createTauriEventObservable<VoiceState>("queue-updated");
  
  return queueUpdates$
    .pipe(tap((state) => {
      console.log("Queue updated:", state.status);
    }))
    .subscribe({
      next: (state) => get().setQueueState(state),
      error: (err) => console.error("Queue update error:", err),
    });
}
```

## Performance Characteristics

### Write Performance (Daemon)

- **Frequency**: After every queue state change (~1-5 times per request)
- **Size**: Typically <50KB for reasonable queue sizes
- **Duration**: <10ms on modern hardware (SSD)
- **Blocking**: File I/O is blocking but happens after request completes

### Read Performance (Tray App)

- **Frequency**: On file system events (1-5 per request)
- **Debouncing**: File watcher naturally debounces rapid changes
- **Parsing**: Automerge load is ~1-5ms for typical document sizes
- **UI Update**: RxJS + React re-render is <16ms (60fps)

### File Size

Automerge documents grow with history. Current implementation:

- **No compaction** — History accumulates
- **Typical size**: 50-200KB after hours of use
- **Max observed**: ~1MB after days of continuous use

Future optimization: Periodic compaction via `doc.save()` with history truncation.

## Reliability

### Crash Recovery

**Daemon crashes:**
- Last saved state persists on disk
- Tray app shows last known state
- Reconnect shows "daemon not running" error

**Tray app crashes:**
- No impact on daemon
- File watcher recreated on restart
- Loads latest state immediately

### Concurrent Access

**Multiple tray apps:**
- Each has independent file watcher
- All readers see consistent state (CRDT guarantees)
- No write conflicts (only daemon writes)

**File system race conditions:**
- `notify` handles rapid file changes
- Duplicate events filtered by `is_state_file_event()`
- Parse errors logged but don't crash watcher

### Error Handling

**Daemon write failures:**
```rust
if let Err(e) = am.save() {
    eprintln!("voiced: failed to save automerge doc: {}", e);
}
// Error logged but worker continues
```

**Tray read failures:**
```rust
match load_state(&self.path) {
    Ok(state) => Some(state),
    Err(e) => {
        log::error!("Error loading state after file change: {}", e);
        None  // Skip this update, wait for next change
    }
}
```

## Testing

### Unit Tests

**Daemon:**
- `automerge_state.rs` — State serialization, file I/O
- `worker.rs` — Sync logic (currently untested)

**Tray app:**
- `automerge_sync.rs` — Path construction, missing file handling

### Integration Tests

**Daemon:**
- `tests/daemon_integration.rs` — Full stack including automerge persistence

**Future tests:**
- Concurrent tray app connections
- State consistency after daemon restart
- Large queue handling (>100 items)

## Trade-offs

### Why Automerge?

**Pros:**
- Zero-copy file watching (no HTTP server needed)
- CRDT properties (eventual consistency)
- Simple architecture (just file I/O)
- Automatic conflict resolution (even though only daemon writes)

**Cons:**
- Document grows with history
- File I/O on every state change
- No query API (always load full document)

### Alternatives Considered

**HTTP/WebSocket:**
- ❌ Requires port management
- ❌ More complex error handling (connection issues)
- ✅ Better for network transparency
- ✅ Query API support

**SQLite:**
- ❌ Concurrent write complexity
- ❌ No automatic sync
- ✅ Smaller storage footprint
- ✅ Query support

**Redis/Message Queue:**
- ❌ External dependency
- ❌ Overkill for single-machine use
- ✅ Pub/sub naturally fits
- ✅ Better scalability

## Future Improvements

### Short-term

1. **Periodic compaction** — Truncate history to keep file size bounded
2. **Graceful degradation** — Show cached state if file read fails
3. **Metrics** — Track sync latency and file size

### Long-term

1. **Network transparency** — Allow tray app on different machine
2. **Multi-writer support** — Allow MCP clients to update state directly
3. **Query API** — Avoid loading full document for simple queries
4. **Delta updates** — Only emit changed fields to frontend

## References

- [Automerge documentation](https://automerge.org/docs/hello/)
- [automorph crate](https://docs.rs/automorph/) — Derive macro for Automerge
- [notify crate](https://docs.rs/notify/) — Cross-platform file watching
- [CRDTs explained](https://crdt.tech/)
