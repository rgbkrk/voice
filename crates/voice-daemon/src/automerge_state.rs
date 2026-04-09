//! Automerge document for daemon state persistence.
//!
//! The daemon maintains a single Automerge CRDT document representing
//! the complete queue state. This doc is written to ~/.voice/state.automerge
//! on every queue change, enabling real-time UI sync via file watching.

use automerge::{AutoCommit, ROOT};
use automorph::{Automorph, ChangeReport};
use std::collections::HashMap;
use std::path::PathBuf;
use voice_protocol::rpc::{DaemonState, ItemStatus, QueueItem};

/// Automerge-backed state document.
#[derive(Automorph, Clone, Debug)]
pub struct VoiceState {
    pub status: String,
    pub current: Option<AutomergeQueueItem>,
    pub pending: Vec<AutomergeQueueItem>,
    pub recent: Vec<AutomergeQueueItem>,
    pub audio: HashMap<String, AudioInfo>,
}

/// Queue item stored in Automerge (mirrors QueueItem but with Automorph).
#[derive(Automorph, Clone, Debug)]
pub struct AutomergeQueueItem {
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
#[derive(Automorph, Clone, Debug)]
pub struct AudioInfo {
    pub question_path: Option<String>,
    pub answer_path: Option<String>,
    pub duration_ms: u64,
}

impl From<&QueueItem> for AutomergeQueueItem {
    fn from(item: &QueueItem) -> Self {
        Self {
            id: item.id.clone(),
            client_id: item.client_id.clone(),
            method: item.method.clone(),
            status: match &item.status {
                ItemStatus::Queued => "Queued".to_string(),
                ItemStatus::Processing => "Processing".to_string(),
                ItemStatus::Completed => "Completed".to_string(),
                ItemStatus::Failed => "Failed".to_string(),
            },
            created_at: item.created_at,
            text_preview: item.text_preview.clone(),
            result: item.result.clone(),
            repo: item.repo.clone(),
            completed_at: item.completed_at,
            auto_clear_at: item.auto_clear_at,
        }
    }
}

pub struct AutomergeState {
    doc: AutoCommit,
    path: PathBuf,
}

impl AutomergeState {
    /// Create new Automerge state document.
    pub fn new() -> Self {
        let path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".voice")
            .join("state.automerge");

        let doc = AutoCommit::new();
        Self { doc, path }
    }

    /// Load existing document from disk, or create new if missing.
    pub fn load_or_create() -> Result<Self, String> {
        let path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".voice")
            .join("state.automerge");

        if path.exists() {
            let bytes = std::fs::read(&path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
            let doc = AutoCommit::load(&bytes)
                .map_err(|e| format!("Failed to load Automerge doc: {}", e))?;
            Ok(Self { doc, path })
        } else {
            Ok(Self::new())
        }
    }

    /// Update the document with new daemon state.
    pub fn update(&mut self, state: &DaemonState) {
        let voice_state = VoiceState {
            status: state.status.clone(),
            current: state.current.as_ref().map(|item| item.into()),
            pending: state.pending.iter().map(|item| item.into()).collect(),
            recent: state.recent.iter().map(|item| item.into()).collect(),
            audio: HashMap::new(), // Audio map updated separately via set_audio
        };

        // Use automorph to write to root
        // Note: We ignore errors here since this is called from hot path
        let _ = voice_state.save(&mut self.doc, &ROOT, "state");
    }

    /// Remove an item from recent (for cleanup task).
    pub fn remove_from_recent(&mut self, queue_id: &str) {
        if let Ok(mut state) = VoiceState::load(&self.doc, &ROOT, "state") {
            state.recent.retain(|item| item.id != queue_id);
            let _ = state.save(&mut self.doc, &ROOT, "state");
        }
    }

    /// Save document to disk.
    pub fn save(&mut self) -> Result<(), String> {
        // Ensure directory exists
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create dir {}: {}", parent.display(), e))?;
        }

        let bytes = self.doc.save();
        std::fs::write(&self.path, bytes)
            .map_err(|e| format!("Failed to write {}: {}", self.path.display(), e))?;
        Ok(())
    }
}
