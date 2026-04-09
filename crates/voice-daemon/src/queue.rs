//! Request queue for serializing voice operations.

use std::collections::{HashMap, VecDeque};
use tokio::sync::{oneshot, Mutex, Notify};
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

/// Internal queue entry.
#[derive(Debug, Clone)]
pub struct QueueEntry {
    pub id: String,
    pub client_id: String,
    pub request: VoiceRequest,
    pub status: ItemStatus,
    pub created_at: u64,
    pub result: Option<String>,
    pub completed_at: Option<u64>,
    pub repo: Option<String>,
    pub auto_clear_at: Option<u64>,
}

impl QueueEntry {
    fn to_protocol(&self) -> QueueItem {
        QueueItem {
            id: self.id.clone(),
            client_id: self.client_id.clone(),
            method: self.request.method().to_string(),
            status: self.status.clone(),
            created_at: self.created_at,
            text_preview: self.request.text_preview(),
            result: self.result.clone(),
            repo: self.repo.clone(),
            completed_at: self.completed_at,
            auto_clear_at: self.auto_clear_at,
        }
    }
}

/// Result sent through the completion channel.
#[derive(Debug, Clone)]
pub struct CompletionResult {
    pub status: ItemStatus,
    pub result: Option<String>,
}

pub struct RequestQueue {
    items: Mutex<VecDeque<QueueEntry>>,
    current: Mutex<Option<QueueEntry>>,
    recent: Mutex<VecDeque<QueueEntry>>,
    /// Completion channels: queue_id → sender. Signaled when an item finishes.
    waiters: Mutex<HashMap<String, oneshot::Sender<CompletionResult>>>,
    pub notify: Notify,
}

impl RequestQueue {
    pub fn new() -> Self {
        Self {
            items: Mutex::new(VecDeque::new()),
            current: Mutex::new(None),
            recent: Mutex::new(VecDeque::new()),
            waiters: Mutex::new(HashMap::new()),
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
            completed_at: None,
            repo: None,
            auto_clear_at: None,
        };
        self.items.lock().await.push_back(entry);
        self.notify.notify_one();
        id
    }

    /// Enqueue and atomically register a waiter. The waiter is registered
    /// *before* the item is pushed so the worker can never complete it
    /// before we start listening. Returns (queue_id, receiver).
    pub async fn enqueue_and_wait(
        &self,
        client_id: String,
        request: VoiceRequest,
    ) -> (String, oneshot::Receiver<CompletionResult>) {
        let id = Uuid::new_v4().to_string()[..8].to_string();
        let (tx, rx) = oneshot::channel();

        // Register waiter first, then push
        self.waiters.lock().await.insert(id.clone(), tx);

        let entry = QueueEntry {
            id: id.clone(),
            client_id,
            request,
            status: ItemStatus::Queued,
            created_at: now_secs(),
            result: None,
            completed_at: None,
            repo: None,
            auto_clear_at: None,
        };
        self.items.lock().await.push_back(entry);
        self.notify.notify_one();
        (id, rx)
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
            let id = entry.id.clone();
            entry.status = ItemStatus::Completed;
            entry.result = result.clone();
            self.push_recent(entry).await;
            self.signal_waiter(
                &id,
                CompletionResult {
                    status: ItemStatus::Completed,
                    result,
                },
            )
            .await;
        }
    }

    pub async fn fail(&self, error: String) {
        if let Some(mut entry) = self.current.lock().await.take() {
            let id = entry.id.clone();
            entry.status = ItemStatus::Failed;
            entry.result = Some(error.clone());
            self.push_recent(entry).await;
            self.signal_waiter(
                &id,
                CompletionResult {
                    status: ItemStatus::Failed,
                    result: Some(error),
                },
            )
            .await;
        }
    }

    pub async fn cancel_client(&self, client_id: &str) -> usize {
        let mut items = self.items.lock().await;
        let before = items.len();
        items.retain(|e| e.client_id != client_id);
        before - items.len()
    }

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

    /// Set completed_at and auto_clear_at on the current item.
    pub async fn set_auto_clear(&self, clear_delay_secs: u64) {
        if let Some(entry) = self.current.lock().await.as_mut() {
            let now = now_secs();
            entry.completed_at = Some(now);
            entry.auto_clear_at = Some(now + clear_delay_secs);
        }
    }

    async fn push_recent(&self, entry: QueueEntry) {
        let mut recent = self.recent.lock().await;
        recent.push_front(entry);
        if recent.len() > 20 {
            recent.pop_back();
        }
    }

    async fn signal_waiter(&self, id: &str, result: CompletionResult) {
        if let Some(tx) = self.waiters.lock().await.remove(id) {
            let _ = tx.send(result);
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
