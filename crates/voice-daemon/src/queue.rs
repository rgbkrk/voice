//! Request queue for serializing voice operations.

use std::collections::VecDeque;
use tokio::sync::{Mutex, Notify};
use uuid::Uuid;
use voice_protocol::rpc::{DaemonState, ItemStatus, QueueItem};

/// What the worker will execute.
#[derive(Debug, Clone)]
pub enum VoiceRequest {
    Speak {
        text: String,
        voice: Option<String>,
        speed: Option<f64>,
    },
    Listen {
        max_duration_ms: Option<u64>,
    },
    Converse {
        text: String,
        voice: Option<String>,
    },
}

impl VoiceRequest {
    pub fn method(&self) -> &str {
        match self {
            Self::Speak { .. } => "speak",
            Self::Listen { .. } => "listen",
            Self::Converse { .. } => "converse",
        }
    }

    pub fn text_preview(&self) -> Option<String> {
        match self {
            Self::Speak { text, .. } | Self::Converse { text, .. } => {
                let preview: String = text.chars().take(80).collect();
                Some(preview)
            }
            Self::Listen { .. } => None,
        }
    }
}

/// Internal queue entry (richer than the protocol QueueItem).
#[derive(Debug, Clone)]
pub struct QueueEntry {
    pub id: String,
    pub client_id: String,
    pub request: VoiceRequest,
    pub status: ItemStatus,
    pub created_at: u64,
    pub result: Option<String>,
}

impl QueueEntry {
    /// Convert to the protocol's QueueItem for serialization.
    fn to_protocol(&self) -> QueueItem {
        QueueItem {
            id: self.id.clone(),
            client_id: self.client_id.clone(),
            method: self.request.method().to_string(),
            status: self.status.clone(),
            created_at: self.created_at,
            text_preview: self.request.text_preview(),
            result: self.result.clone(),
        }
    }
}

pub struct RequestQueue {
    items: Mutex<VecDeque<QueueEntry>>,
    current: Mutex<Option<QueueEntry>>,
    recent: Mutex<VecDeque<QueueEntry>>,
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

    async fn enqueue(&self, client_id: String, request: VoiceRequest) -> String {
        let id = Uuid::new_v4().to_string()[..8].to_string();
        let entry = QueueEntry {
            id: id.clone(),
            client_id,
            request,
            status: ItemStatus::Queued,
            created_at: now_secs(),
            result: None,
        };
        self.items.lock().await.push_back(entry);
        self.notify.notify_one();
        id
    }

    pub async fn dequeue(&self) -> Option<QueueEntry> {
        let mut items = self.items.lock().await;
        if let Some(mut entry) = items.pop_front() {
            entry.status = ItemStatus::Processing;
            *self.current.lock().await = Some(entry.clone());
            Some(entry)
        } else {
            None
        }
    }

    pub async fn complete(&self, result: Option<String>) {
        if let Some(mut entry) = self.current.lock().await.take() {
            entry.status = ItemStatus::Completed;
            entry.result = result;
            self.push_recent(entry).await;
        }
    }

    #[allow(dead_code)]
    pub async fn fail(&self, error: String) {
        if let Some(mut entry) = self.current.lock().await.take() {
            entry.status = ItemStatus::Failed;
            entry.result = Some(error);
            self.push_recent(entry).await;
        }
    }

    pub async fn cancel_client(&self, client_id: &str) -> usize {
        let mut items = self.items.lock().await;
        let before = items.len();
        items.retain(|e| e.client_id != client_id);
        before - items.len()
    }

    /// Snapshot using the shared protocol types.
    pub async fn snapshot(&self) -> DaemonState {
        let current = self.current.lock().await.as_ref().map(|e| e.to_protocol());
        let pending: Vec<QueueItem> = self
            .items
            .lock()
            .await
            .iter()
            .map(|e| e.to_protocol())
            .collect();
        let recent: Vec<QueueItem> = self
            .recent
            .lock()
            .await
            .iter()
            .map(|e| e.to_protocol())
            .collect();

        let status = match &current {
            Some(item) => match item.method.as_str() {
                "speak" => "speaking",
                "listen" => "listening",
                "converse" => "conversing",
                _ => "idle",
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

    async fn push_recent(&self, entry: QueueEntry) {
        let mut recent = self.recent.lock().await;
        recent.push_front(entry);
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
