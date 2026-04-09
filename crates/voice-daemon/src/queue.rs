//! Request queue for serializing voice operations.
//!
//! All TTS/STT requests go through this queue so only one operation
//! runs at a time, preventing audio overlap between multiple clients.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tokio::sync::{Mutex, Notify};
use uuid::Uuid;

/// A voice request — what the worker will execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
pub enum VoiceRequest {
    #[serde(rename = "speak")]
    Speak {
        text: String,
        voice: Option<String>,
        speed: Option<f64>,
    },
    #[serde(rename = "listen")]
    Listen { max_duration_ms: Option<u64> },
    #[serde(rename = "converse")]
    Converse { text: String, voice: Option<String> },
}

/// Status of a queued item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ItemStatus {
    Queued,
    Processing,
    Completed,
    Failed,
}

/// A single item in the queue with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueItem {
    pub id: String,
    pub client_id: String,
    pub request: VoiceRequest,
    pub status: ItemStatus,
    pub created_at: u64,
    pub result: Option<String>,
}

/// Snapshot of daemon state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonState {
    pub status: String,
    pub current: Option<QueueItem>,
    pub pending: Vec<QueueItem>,
    pub recent: Vec<QueueItem>,
}

pub struct RequestQueue {
    items: Mutex<VecDeque<QueueItem>>,
    current: Mutex<Option<QueueItem>>,
    recent: Mutex<VecDeque<QueueItem>>,
    pub notify: Notify,
}

impl RequestQueue {
    pub fn new() -> Self {
        Self {
            items: Mutex::new(VecDeque::new()),
            current: Mutex::new(None),
            recent: Mutex::new(VecDeque::new()),
            notify: Notify::new(),
        }
    }

    // -- Typed enqueue methods ------------------------------------------------

    pub async fn enqueue_speak(
        &self,
        client_id: String,
        text: String,
        voice: Option<String>,
        speed: Option<f64>,
    ) -> String {
        self.enqueue(client_id, VoiceRequest::Speak { text, voice, speed })
            .await
    }

    pub async fn enqueue_listen(&self, client_id: String, max_duration_ms: Option<u64>) -> String {
        self.enqueue(client_id, VoiceRequest::Listen { max_duration_ms })
            .await
    }

    pub async fn enqueue_converse(
        &self,
        client_id: String,
        text: String,
        voice: Option<String>,
    ) -> String {
        self.enqueue(client_id, VoiceRequest::Converse { text, voice })
            .await
    }

    // -- Core queue operations ------------------------------------------------

    async fn enqueue(&self, client_id: String, request: VoiceRequest) -> String {
        let id = Uuid::new_v4().to_string()[..8].to_string();
        let item = QueueItem {
            id: id.clone(),
            client_id,
            request,
            status: ItemStatus::Queued,
            created_at: now_secs(),
            result: None,
        };
        self.items.lock().await.push_back(item);
        self.notify.notify_one();
        id
    }

    pub async fn dequeue(&self) -> Option<QueueItem> {
        let mut items = self.items.lock().await;
        if let Some(mut item) = items.pop_front() {
            item.status = ItemStatus::Processing;
            *self.current.lock().await = Some(item.clone());
            Some(item)
        } else {
            None
        }
    }

    pub async fn complete(&self, result: Option<String>) {
        if let Some(mut item) = self.current.lock().await.take() {
            item.status = ItemStatus::Completed;
            item.result = result;
            self.push_recent(item).await;
        }
    }

    #[allow(dead_code)]
    pub async fn fail(&self, error: String) {
        if let Some(mut item) = self.current.lock().await.take() {
            item.status = ItemStatus::Failed;
            item.result = Some(error);
            self.push_recent(item).await;
        }
    }

    pub async fn cancel_client(&self, client_id: &str) -> usize {
        let mut items = self.items.lock().await;
        let before = items.len();
        items.retain(|item| item.client_id != client_id);
        before - items.len()
    }

    pub async fn snapshot(&self) -> DaemonState {
        let current = self.current.lock().await.clone();
        let pending: Vec<QueueItem> = self.items.lock().await.iter().cloned().collect();
        let recent: Vec<QueueItem> = self.recent.lock().await.iter().cloned().collect();

        let status = match &current {
            Some(item) => match &item.request {
                VoiceRequest::Speak { .. } => "speaking",
                VoiceRequest::Listen { .. } => "listening",
                VoiceRequest::Converse { .. } => "conversing",
            },
            None if !pending.is_empty() => "queued",
            None => "idle",
        };

        DaemonState {
            status: status.to_string(),
            current,
            pending,
            recent,
        }
    }

    async fn push_recent(&self, item: QueueItem) {
        let mut recent = self.recent.lock().await;
        recent.push_front(item);
        if recent.len() > 20 {
            recent.pop_back();
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
