//! Automerge document for daemon state persistence.
//!
//! The daemon maintains a single Automerge CRDT document representing
//! the complete queue state. This doc is written to ~/.voice/state.automerge
//! on every queue change, enabling real-time UI sync via file watching.

use automerge::{transaction::Transactable, AutoCommit, ReadDoc, ROOT};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use voice_protocol::rpc::{DaemonState, ItemStatus, QueueItem};

/// Voice state that mirrors DaemonState for serialization.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VoiceState {
    pub status: String,
    pub current: Option<QueueItemData>,
    pub pending: Vec<QueueItemData>,
    pub recent: Vec<QueueItemData>,
    pub audio: HashMap<String, AudioInfo>,
}

/// Queue item data for serialization.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QueueItemData {
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
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AudioInfo {
    pub question_path: Option<String>,
    pub answer_path: Option<String>,
    pub duration_ms: u64,
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
            current: state
                .current
                .as_ref()
                .map(|item| QueueItemData::from_queue_item(item)),
            pending: state
                .pending
                .iter()
                .map(QueueItemData::from_queue_item)
                .collect(),
            recent: state
                .recent
                .iter()
                .map(QueueItemData::from_queue_item)
                .collect(),
            audio: HashMap::new(),
        };

        if let Ok(json_str) = serde_json::to_string(&voice_state) {
            self.doc.put(ROOT, "state", json_str).ok();
        }
    }

    /// Extract voice state from Automerge document by reading the "state" key.
    pub fn extract_state(&self) -> Result<VoiceState, String> {
        use automerge::{ScalarValue, Value};

        match self.doc.get(ROOT, "state") {
            Ok(Some((Value::Scalar(scalar), _))) => {
                // For MVP, we use a simple approach: export to JSON and deserialize
                // The state is stored as a JSON string in the document
                if let ScalarValue::Str(json_str) = scalar.as_ref() {
                    let root: serde_json::Value = serde_json::from_str(json_str)
                        .map_err(|e| format!("Failed to parse state JSON: {}", e))?;

                    let state: VoiceState = serde_json::from_value(root)
                        .map_err(|e| format!("Failed to deserialize VoiceState: {}", e))?;

                    Ok(state)
                } else {
                    Err("State key is not a string".to_string())
                }
            }
            Ok(None) => Err("No 'state' key in Automerge doc".to_string()),
            _ => Err("State key is not accessible".to_string()),
        }
    }

    /// Remove an item from recent (for cleanup task).
    pub fn remove_from_recent(&mut self, queue_id: &str) -> Result<(), String> {
        let mut state = self.extract_state()?;
        state.recent.retain(|item| item.id != queue_id);

        if let Ok(json_str) = serde_json::to_string(&state) {
            self.doc.put(ROOT, "state", json_str).ok();
            Ok(())
        } else {
            Err("Failed to serialize VoiceState".to_string())
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

impl QueueItemData {
    fn from_queue_item(item: &QueueItem) -> Self {
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
