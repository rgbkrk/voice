//! Automerge document reader and file watcher.

use crate::types::{AudioInfo, QueueItem, VoiceState};
use automerge::{AutoCommit, ROOT};
use automorph::{Automorph, ChangeReport};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use tokio::time::Duration;

/// Temporary struct to read state from Automerge document.
#[derive(Automorph)]
struct AutomergeVoiceState {
    pub status: String,
    pub current: Option<AutomergeQueueItem>,
    pub pending: Vec<AutomergeQueueItem>,
    pub recent: Vec<AutomergeQueueItem>,
    pub audio: HashMap<String, AutomergeAudioInfo>,
}

/// Queue item stored in Automerge.
#[derive(Automorph)]
struct AutomergeQueueItem {
    pub id: String,
    pub client_id: String,
    pub method: String,
    pub status: String,
    pub created_at: u64,
    pub text_preview: Option<String>,
    pub result: Option<String>,
    pub repo: Option<String>,
    pub completed_at: Option<u64>,
    pub auto_clear_at: Option<u64>,
}

/// Audio file metadata.
#[derive(Automorph)]
struct AutomergeAudioInfo {
    pub question_path: Option<String>,
    pub answer_path: Option<String>,
    pub duration_ms: u64,
}

impl From<AutomergeQueueItem> for QueueItem {
    fn from(item: AutomergeQueueItem) -> Self {
        Self {
            id: item.id,
            client_id: item.client_id,
            method: item.method,
            status: item.status,
            created_at: item.created_at,
            text_preview: item.text_preview,
            result: item.result,
            repo: item.repo,
            completed_at: item.completed_at,
            auto_clear_at: item.auto_clear_at,
        }
    }
}

impl From<AutomergeAudioInfo> for AudioInfo {
    fn from(info: AutomergeAudioInfo) -> Self {
        Self {
            question_path: info.question_path,
            answer_path: info.answer_path,
            duration_ms: info.duration_ms,
        }
    }
}

/// Get path to daemon's Automerge state file.
pub fn state_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".voice")
        .join("state.automerge")
}

/// Load VoiceState from Automerge document file.
pub fn load_state(path: &Path) -> Result<VoiceState, String> {
    if !path.exists() {
        // File doesn't exist yet (daemon not started or no activity)
        return Ok(VoiceState::default());
    }

    let bytes =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    if bytes.is_empty() {
        // Empty file (daemon created placeholder)
        return Ok(VoiceState::default());
    }

    let doc =
        AutoCommit::load(&bytes).map_err(|e| format!("Failed to load Automerge doc: {}", e))?;

    // Use automorph to load the state from the document
    extract_voice_state(&doc)
}

/// Extract VoiceState from Automerge document using automorph.
fn extract_voice_state(doc: &AutoCommit) -> Result<VoiceState, String> {
    // Load the state using automorph
    let automerge_state: AutomergeVoiceState = AutomergeVoiceState::load(doc, &ROOT, "state")
        .map_err(|e| format!("Failed to load VoiceState from Automerge: {}", e))?;

    // Convert to the types used by the UI
    Ok(VoiceState {
        status: automerge_state.status,
        current: automerge_state.current.map(|q| q.into()),
        pending: automerge_state
            .pending
            .into_iter()
            .map(|q| q.into())
            .collect(),
        recent: automerge_state
            .recent
            .into_iter()
            .map(|q| q.into())
            .collect(),
        audio: automerge_state
            .audio
            .into_iter()
            .map(|(k, v)| (k, v.into()))
            .collect(),
    })
}

/// File system watcher for Automerge state file.
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    rx: mpsc::UnboundedReceiver<Result<Event, notify::Error>>,
    path: PathBuf,
}

impl FileWatcher {
    /// Create new file watcher for daemon state file.
    pub fn new() -> Result<Self, String> {
        let path = state_path();

        // Ensure parent directory exists for watching
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create .voice dir: {}", e))?;
        }

        let (tx, rx) = mpsc::unbounded_channel();

        let mut watcher = notify::recommended_watcher(move |res| {
            let _ = tx.send(res);
        })
        .map_err(|e| format!("Failed to create file watcher: {}", e))?;

        // Watch the parent directory (more reliable than watching file directly)
        let watch_dir = path.parent().unwrap();
        watcher
            .watch(watch_dir, RecursiveMode::NonRecursive)
            .map_err(|e| format!("Failed to watch directory: {}", e))?;

        Ok(Self {
            _watcher: watcher,
            rx,
            path,
        })
    }

    /// Wait for next file change (async). Returns new state if file changed.
    pub async fn wait_for_change(&mut self, timeout: Duration) -> Option<VoiceState> {
        match tokio::time::timeout(timeout, self.rx.recv()).await {
            Ok(Some(Ok(event))) => {
                // Check if event is for our state file
                if self.is_state_file_event(&event) {
                    // File changed, reload state
                    match load_state(&self.path) {
                        Ok(state) => Some(state),
                        Err(e) => {
                            eprintln!("Error loading state after file change: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            }
            Ok(Some(Err(e))) => {
                eprintln!("File watcher error: {}", e);
                None
            }
            Ok(None) => {
                eprintln!("File watcher disconnected");
                None
            }
            Err(_) => None, // Timeout
        }
    }

    fn is_state_file_event(&self, event: &Event) -> bool {
        event.paths.iter().any(|p| {
            p.file_name() == self.path.file_name()
                && matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_))
        })
    }

    /// Load current state immediately (non-blocking).
    pub fn load_current(&self) -> Result<VoiceState, String> {
        load_state(&self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_path_construction() {
        let path = state_path();
        assert!(path.to_string_lossy().contains(".voice"));
        assert!(path.to_string_lossy().ends_with("state.automerge"));
    }

    #[test]
    fn test_load_state_missing_file() {
        let result = load_state(Path::new("/nonexistent/path.automerge"));
        assert!(result.is_ok());
        let state = result.unwrap();
        assert_eq!(state.status, "idle");
        assert!(state.pending.is_empty());
    }
}
