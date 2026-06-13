//! Request queue for serializing voice operations.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex, Notify};
use uuid::Uuid;
use voice_audio::AudioOutputFormat;
use voice_protocol::rpc::{DaemonState, ItemStatus, QueueItem};
use voice_stream::TtsStreamEvent;

const CANCELLED_MESSAGE: &str = "Cancelled by user";

/// Parameters for a daemon TTS stream request.
#[derive(Debug, Clone)]
pub struct StreamSpeakRequest {
    pub text: String,
    pub stream_id: String,
    pub voice: Option<String>,
    pub speed: Option<f64>,
    pub sample_rate: u32,
    pub frame_ms: u32,
    pub event_tx: mpsc::Sender<TtsStreamEvent>,
}

/// What the worker will execute.
#[derive(Debug, Clone)]
pub enum VoiceRequest {
    Speak {
        text: String,
        voice: Option<String>,
        speed: Option<f64>,
    },
    /// Generate speech audio into a file without opening audio output.
    Synthesize {
        text: String,
        output_path: String,
        output_format: Option<AudioOutputFormat>,
        voice: Option<String>,
        speed: Option<f64>,
    },
    /// Generate speech audio as ordered stream events.
    StreamSpeak(StreamSpeakRequest),
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
            Self::Synthesize { .. } => "synthesize",
            Self::StreamSpeak(_) => "stream_speak",
            Self::Listen { .. } => "listen",
            Self::Converse { .. } => "converse",
        }
    }

    pub fn text_preview(&self) -> Option<String> {
        match self {
            Self::Speak { text, .. }
            | Self::Synthesize { text, .. }
            | Self::Converse { text, .. } => {
                let preview: String = text.chars().take(80).collect();
                Some(preview)
            }
            Self::StreamSpeak(request) => {
                let preview: String = request.text.chars().take(80).collect();
                Some(preview)
            }
            Self::Listen { .. } => None,
        }
    }

    fn notify_cancelled(&self) {
        if let Self::StreamSpeak(request) = self {
            let _ = request.event_tx.try_send(TtsStreamEvent::cancelled(
                request.stream_id.clone(),
                CANCELLED_MESSAGE,
            ));
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
    pub cancelled: Arc<AtomicBool>,
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

    pub async fn enqueue_synthesize(
        &self,
        client_id: String,
        text: String,
        output_path: String,
        output_format: Option<AudioOutputFormat>,
        voice: Option<String>,
        speed: Option<f64>,
    ) -> String {
        self.enqueue(
            client_id,
            VoiceRequest::Synthesize {
                text,
                output_path,
                output_format,
                voice,
                speed,
            },
        )
        .await
    }

    pub async fn enqueue_stream_speak(
        &self,
        client_id: String,
        request: StreamSpeakRequest,
    ) -> String {
        self.enqueue(client_id, VoiceRequest::StreamSpeak(request))
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
            cancelled: Arc::new(AtomicBool::new(false)),
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
            cancelled: Arc::new(AtomicBool::new(false)),
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

    pub async fn complete(&self, result: Option<String>, auto_clear_secs: Option<u64>) {
        if let Some(mut entry) = self.current.lock().await.take() {
            let id = entry.id.clone();
            entry.status = ItemStatus::Completed;
            entry.result = result.clone();

            // Set auto-clear timestamps if requested
            if let Some(delay) = auto_clear_secs {
                let now = now_secs();
                entry.completed_at = Some(now);
                entry.auto_clear_at = Some(now + delay);
            }

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
        let mut cancelled = Vec::new();

        {
            let mut current = self.current.lock().await;
            if current
                .as_ref()
                .is_some_and(|entry| entry.client_id == client_id)
            {
                if let Some(entry) = current.take() {
                    cancelled.push(entry);
                }
            }
        }

        {
            let mut items = self.items.lock().await;
            let mut index = 0;
            while index < items.len() {
                if items[index].client_id == client_id {
                    cancelled.push(items.remove(index).unwrap());
                } else {
                    index += 1;
                }
            }
        }

        let count = cancelled.len();
        for entry in cancelled {
            self.cancel_entry(entry).await;
        }
        count
    }

    /// Cancel a specific queue item by ID.
    pub async fn cancel_item(&self, queue_id: &str) -> bool {
        // Check if it's the current item
        let current_entry = {
            let mut current = self.current.lock().await;
            if let Some(entry) = current.as_ref() {
                if entry.id == queue_id {
                    current.take()
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some(entry) = current_entry {
            self.cancel_entry(entry).await;
            return true;
        }

        // Check pending queue
        let pending_entry = {
            let mut items = self.items.lock().await;
            items
                .iter()
                .position(|e| e.id == queue_id)
                .and_then(|pos| items.remove(pos))
        };

        if let Some(entry) = pending_entry {
            self.cancel_entry(entry).await;
            return true;
        }

        false
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
                "synthesize" => "synthesizing",
                "stream_speak" => "streaming",
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

    async fn signal_waiter(&self, id: &str, result: CompletionResult) {
        if let Some(tx) = self.waiters.lock().await.remove(id) {
            let _ = tx.send(result);
        }
    }

    async fn cancel_entry(&self, mut entry: QueueEntry) {
        let id = entry.id.clone();
        entry.cancelled.store(true, Ordering::SeqCst);
        entry.request.notify_cancelled();
        entry.status = ItemStatus::Failed;
        entry.result = Some(CANCELLED_MESSAGE.to_string());
        self.push_recent(entry).await;
        self.signal_waiter(
            &id,
            CompletionResult {
                status: ItemStatus::Failed,
                result: Some(CANCELLED_MESSAGE.to_string()),
            },
        )
        .await;
    }

    /// Remove a completed item from the recent list by ID.
    pub async fn remove_recent(&self, id: &str) {
        self.recent.lock().await.retain(|item| item.id != id);
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cancel_client_moves_pending_items_to_recent_and_signals_waiters() {
        let queue = RequestQueue::new();
        let (queue_id, rx) = queue
            .enqueue_and_wait(
                "client-a".to_string(),
                VoiceRequest::Speak {
                    text: "hello".to_string(),
                    voice: None,
                    speed: None,
                },
            )
            .await;

        assert_eq!(queue.cancel_client("client-a").await, 1);

        let result = rx.await.unwrap();
        assert_eq!(result.status, ItemStatus::Failed);
        assert_eq!(result.result.as_deref(), Some(CANCELLED_MESSAGE));

        let snapshot = queue.snapshot().await;
        assert!(snapshot.pending.is_empty());
        assert_eq!(snapshot.recent.len(), 1);
        assert_eq!(snapshot.recent[0].id, queue_id);
        assert_eq!(snapshot.recent[0].status, ItemStatus::Failed);
    }

    #[tokio::test]
    async fn cancel_item_signals_waiter_for_pending_item() {
        let queue = RequestQueue::new();
        let (queue_id, rx) = queue
            .enqueue_and_wait(
                "client-a".to_string(),
                VoiceRequest::Listen {
                    max_duration_ms: None,
                },
            )
            .await;

        assert!(queue.cancel_item(&queue_id).await);

        let result = rx.await.unwrap();
        assert_eq!(result.status, ItemStatus::Failed);
        assert_eq!(result.result.as_deref(), Some(CANCELLED_MESSAGE));
    }

    #[tokio::test]
    async fn cancel_item_sets_current_cancellation_flag() {
        let queue = RequestQueue::new();
        let (queue_id, rx) = queue
            .enqueue_and_wait(
                "client-a".to_string(),
                VoiceRequest::Speak {
                    text: "hello".to_string(),
                    voice: None,
                    speed: None,
                },
            )
            .await;

        let entry = queue.dequeue().await.unwrap();
        let cancelled = entry.cancelled.clone();

        assert!(queue.cancel_item(&queue_id).await);
        assert!(cancelled.load(Ordering::SeqCst));

        let result = rx.await.unwrap();
        assert_eq!(result.status, ItemStatus::Failed);
        assert_eq!(result.result.as_deref(), Some(CANCELLED_MESSAGE));
    }
}
