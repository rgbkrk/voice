//! Request queue for serializing voice operations.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex, Notify};
use uuid::Uuid;
use voice_audio::AudioOutputFormat;
use voice_protocol::rpc::{DaemonState, ItemStatus, QueueItem};
use voice_stream::TtsStreamEvent;

const CANCELLED_MESSAGE: &str = "Cancelled by user";

/// Live phase of the converse currently held in `current`. The worker writes
/// it as a converse moves speak -> listen; `converse_result` reads it to tell a
/// caller whether the mic is open. Meaningless unless `current` is a converse.
pub const CONVERSE_PHASE_IDLE: u8 = 0;
pub const CONVERSE_PHASE_SPEAKING: u8 = 1;
pub const CONVERSE_PHASE_LISTENING: u8 = 2;

/// What `converse_result` reports for a converse id.
#[derive(Debug, Clone)]
pub struct ConverseView {
    pub phase: &'static str,
    pub mic_active: bool,
    pub result: Option<String>,
    pub error: Option<String>,
}

/// Optional TTS engine controls carried with daemon requests.
#[derive(Debug, Clone, Default)]
pub struct TtsOptions {
    pub engine: Option<String>,
    pub voxtral_model: Option<String>,
    pub voxtral_max_frames: Option<usize>,
    pub voxtral_flow_steps: Option<usize>,
    pub voxtral_stream_begin_frames: Option<usize>,
    pub voxtral_kv_cache: bool,
}

/// Parameters for a daemon TTS file synthesis request.
#[derive(Debug, Clone)]
pub struct SynthesizeRequest {
    pub text: String,
    pub output_path: String,
    pub output_format: Option<AudioOutputFormat>,
    pub voice: Option<String>,
    pub speed: Option<f64>,
    pub options: TtsOptions,
}

/// Parameters for a daemon TTS stream request.
#[derive(Debug, Clone)]
pub struct StreamSpeakRequest {
    pub text: String,
    pub stream_id: String,
    pub voice: Option<String>,
    pub speed: Option<f64>,
    pub options: TtsOptions,
    pub sample_rate: u32,
    pub frame_ms: u32,
    pub event_tx: mpsc::Sender<TtsStreamEvent>,
}

/// Parameters for a daemon STT stream after the socket layer has collected PCM.
#[derive(Debug, Clone)]
pub struct StreamTranscribeRequest {
    pub stream_id: String,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// What the worker will execute.
#[derive(Debug, Clone)]
pub enum VoiceRequest {
    Speak {
        text: String,
        voice: Option<String>,
        speed: Option<f64>,
        options: TtsOptions,
    },
    /// Generate speech audio into a file without opening audio output.
    Synthesize {
        text: String,
        output_path: String,
        output_format: Option<AudioOutputFormat>,
        voice: Option<String>,
        speed: Option<f64>,
        options: TtsOptions,
    },
    /// Generate speech audio as ordered stream events.
    StreamSpeak(StreamSpeakRequest),
    /// Transcribe caller-supplied PCM frames.
    StreamTranscribe(StreamTranscribeRequest),
    Listen {
        max_duration_ms: Option<u64>,
    },
    Converse {
        text: String,
        voice: Option<String>,
        max_duration_ms: Option<u64>,
        options: TtsOptions,
    },
}

impl VoiceRequest {
    pub fn method(&self) -> &str {
        match self {
            Self::Speak { .. } => "speak",
            Self::Synthesize { .. } => "synthesize",
            Self::StreamSpeak(_) => "stream_speak",
            Self::StreamTranscribe(_) => "stream_transcribe",
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
            Self::StreamTranscribe(request) => Some(format!(
                "{} samples @ {} Hz",
                request.samples.len(),
                request.sample_rate
            )),
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
    /// Terminal converse outcomes, kept separately from `recent` so routine
    /// fire-and-forget `speak` traffic can't evict a transcript before the
    /// caller fetches it with `converse_result`.
    converse_results: Mutex<VecDeque<QueueEntry>>,
    /// Completion channels: queue_id → sender. Signaled when an item finishes.
    waiters: Mutex<HashMap<String, oneshot::Sender<CompletionResult>>>,
    /// Live speak/listen phase of the in-flight converse (see CONVERSE_PHASE_*).
    converse_phase: Arc<AtomicU8>,
    pub notify: Notify,
}

impl RequestQueue {
    pub fn new() -> Self {
        Self {
            items: Mutex::new(VecDeque::new()),
            current: Mutex::new(None),
            recent: Mutex::new(VecDeque::new()),
            converse_results: Mutex::new(VecDeque::new()),
            waiters: Mutex::new(HashMap::new()),
            converse_phase: Arc::new(AtomicU8::new(CONVERSE_PHASE_IDLE)),
            notify: Notify::new(),
        }
    }

    /// Handle to the in-flight converse phase flag. The worker stores
    /// CONVERSE_PHASE_* on it as a converse moves speak -> listen.
    pub fn converse_phase(&self) -> Arc<AtomicU8> {
        self.converse_phase.clone()
    }

    /// Snapshot of audio load taken before enqueue: (something is playing now,
    /// number of pending items ahead). Used to report converse gating.
    pub async fn audio_load(&self) -> (bool, usize) {
        let busy = self.current.lock().await.is_some();
        let pending = self.items.lock().await.len();
        (busy, pending)
    }

    /// Report the state of a converse by id: completed/failed (from the
    /// dedicated converse-results store), the live speaking/listening phase (if
    /// it is the current item), queued (if still pending), or unknown. Note that
    /// "unknown" is also returned transiently while an entry moves between
    /// collections, so callers that long-poll must not treat it as terminal.
    pub async fn converse_result(&self, id: &str) -> ConverseView {
        if let Some(entry) = self
            .converse_results
            .lock()
            .await
            .iter()
            .find(|e| e.id == id)
        {
            return match entry.status {
                ItemStatus::Failed => ConverseView {
                    phase: "failed",
                    mic_active: false,
                    result: None,
                    error: entry.result.clone(),
                },
                _ => ConverseView {
                    phase: "completed",
                    mic_active: false,
                    result: entry.result.clone(),
                    error: None,
                },
            };
        }

        if self
            .current
            .lock()
            .await
            .as_ref()
            .is_some_and(|e| e.id == id)
        {
            let listening = self.converse_phase.load(Ordering::SeqCst) == CONVERSE_PHASE_LISTENING;
            return ConverseView {
                phase: if listening { "listening" } else { "speaking" },
                mic_active: listening,
                result: None,
                error: None,
            };
        }

        if self.items.lock().await.iter().any(|e| e.id == id) {
            return ConverseView {
                phase: "queued",
                mic_active: false,
                result: None,
                error: None,
            };
        }

        ConverseView {
            phase: "unknown",
            mic_active: false,
            result: None,
            error: None,
        }
    }

    pub async fn enqueue_speak(
        &self,
        client_id: String,
        text: String,
        voice: Option<String>,
        speed: Option<f64>,
        options: TtsOptions,
    ) -> String {
        self.enqueue(
            client_id,
            VoiceRequest::Speak {
                text,
                voice,
                speed,
                options,
            },
        )
        .await
    }

    pub async fn enqueue_synthesize(
        &self,
        client_id: String,
        request: SynthesizeRequest,
    ) -> String {
        self.enqueue(
            client_id,
            VoiceRequest::Synthesize {
                text: request.text,
                output_path: request.output_path,
                output_format: request.output_format,
                voice: request.voice,
                speed: request.speed,
                options: request.options,
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
        max_duration_ms: Option<u64>,
        options: TtsOptions,
    ) -> String {
        self.enqueue(
            client_id,
            VoiceRequest::Converse {
                text,
                voice,
                max_duration_ms,
                options,
            },
        )
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
            cancelled: Arc::new(AtomicBool::new(false)),
        };
        self.items.lock().await.push_back(entry);
        self.notify.notify_one();
        (id, rx)
    }

    pub async fn dequeue(&self) -> Option<QueueEntry> {
        let mut entry = self.items.lock().await.pop_front()?;
        entry.status = ItemStatus::Processing;
        *self.current.lock().await = Some(entry.clone());
        self.notify.notify_waiters();
        Some(entry)
    }

    pub async fn complete(&self, result: Option<String>) {
        if let Some(mut entry) = self.current.lock().await.take() {
            let id = entry.id.clone();
            entry.status = ItemStatus::Completed;
            entry.result = result.clone();
            entry.completed_at = Some(now_secs());

            self.push_recent(entry).await;
            self.signal_waiter(
                &id,
                CompletionResult {
                    status: ItemStatus::Completed,
                    result,
                },
            )
            .await;
            self.notify.notify_waiters();
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
            self.notify.notify_waiters();
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
        // Converse transcripts get their own retention so they survive the high
        // churn of fire-and-forget speak items in the shared `recent` ring.
        if matches!(entry.request, VoiceRequest::Converse { .. }) {
            let mut converse = self.converse_results.lock().await;
            converse.push_front(entry.clone());
            if converse.len() > 64 {
                converse.pop_back();
            }
        }
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
        self.notify.notify_waiters();
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
                    options: TtsOptions::default(),
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
                    options: TtsOptions::default(),
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

    #[tokio::test]
    async fn converse_result_tracks_queued_speaking_listening_completed() {
        let queue = RequestQueue::new();
        let id = queue
            .enqueue_converse(
                "cli".to_string(),
                "need your response".to_string(),
                None,
                None,
                TtsOptions::default(),
            )
            .await;

        assert_eq!(queue.converse_result(&id).await.phase, "queued");

        let entry = queue.dequeue().await.expect("worker item");
        assert_eq!(entry.id, id);

        // Current item, phase still idle -> reported as speaking, mic closed.
        let view = queue.converse_result(&id).await;
        assert_eq!(view.phase, "speaking");
        assert!(!view.mic_active);

        // Worker flips to listening when the mic opens.
        queue
            .converse_phase()
            .store(CONVERSE_PHASE_LISTENING, Ordering::SeqCst);
        let view = queue.converse_result(&id).await;
        assert_eq!(view.phase, "listening");
        assert!(view.mic_active);

        queue
            .complete(Some(r#"{"heard":{"text":"yes"}}"#.to_string()))
            .await;
        let view = queue.converse_result(&id).await;
        assert_eq!(view.phase, "completed");
        assert_eq!(view.result.as_deref(), Some(r#"{"heard":{"text":"yes"}}"#));
        assert!(!view.mic_active);
    }

    #[tokio::test]
    async fn converse_result_unknown_for_missing_id() {
        let queue = RequestQueue::new();
        assert_eq!(queue.converse_result("nope").await.phase, "unknown");
    }
}
