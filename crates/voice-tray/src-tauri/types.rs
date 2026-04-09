//! Types that mirror the daemon's Automerge state document.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Voice daemon queue state (read from Automerge doc).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceState {
    pub status: String,
    pub current: Option<QueueItem>,
    pub pending: Vec<QueueItem>,
    pub recent: Vec<QueueItem>,
    pub audio: HashMap<String, AudioInfo>,
}

/// A queue item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueItem {
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInfo {
    pub question_path: Option<String>,
    pub answer_path: Option<String>,
    pub duration_ms: u64,
}

impl Default for VoiceState {
    fn default() -> Self {
        Self {
            status: "idle".to_string(),
            current: None,
            pending: Vec::new(),
            recent: Vec::new(),
            audio: HashMap::new(),
        }
    }
}
