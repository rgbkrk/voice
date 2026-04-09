//! Automerge document reader and file watcher.

use crate::types::{AudioInfo, QueueItem, VoiceState};
use automerge::{AutoCommit, ROOT};
use automorph::{Automorph, ChangeReport};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
