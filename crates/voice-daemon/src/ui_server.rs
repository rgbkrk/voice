use crate::audio_recorder;
use crate::queue::{RequestQueue, VoiceRequest};
use crate::ui_state::{
    snapshot_from_daemon, UiCommandResult, UiEvent, UiSnapshot, UiTransport, UiTransportState,
    UI_INTERNAL_CLIENT_ID,
};
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode, Uri};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::{broadcast, Mutex};
use voice_protocol::rpc::ItemStatus;

#[derive(Clone)]
struct UiServerState {
    queue: Arc<RequestQueue>,
    active_track_id: Arc<Mutex<Option<String>>>,
    active_response_listens: Arc<Mutex<BTreeMap<String, Option<String>>>>,
    transport: Arc<Mutex<UiTransport>>,
    events: broadcast::Sender<UiEvent>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CommandBody {
    track_id: Option<String>,
}

pub async fn serve(queue: Arc<RequestQueue>) {
    let addr = ui_addr();
    let listener = match TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(err) => {
            eprintln!("voice daemon: failed to bind UI server at {addr}: {err}");
            return;
        }
    };

    let (event_tx, _) = broadcast::channel(64);
    let state = UiServerState {
        queue,
        active_track_id: Arc::new(Mutex::new(None)),
        active_response_listens: Arc::new(Mutex::new(BTreeMap::new())),
        transport: Arc::new(Mutex::new(UiTransport {
            state: UiTransportState::Idle,
            paused: true,
            position_seconds: 0,
        })),
        events: event_tx,
    };

    tokio::spawn(queue_event_pump(state.clone()));

    let app = router(state);

    eprintln!("voice daemon: UI listening on http://{addr}");
    if let Err(err) = axum::serve(listener, app).await {
        eprintln!("voice daemon: UI server stopped: {err}");
    }
}

fn router(state: UiServerState) -> Router {
    Router::new()
        .route("/api/ui/snapshot", get(snapshot))
        .route("/api/ui/events", get(event_stream))
        .route("/api/ui/commands/:command", post(command))
        .route("/api/ui/audio/:track_id/:part", get(audio))
        .fallback(get(asset))
        .with_state(state)
}

fn ui_addr() -> SocketAddr {
    std::env::var("VOICE_UI_ADDR")
        .ok()
        .and_then(|addr| addr.parse().ok())
        .unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 8767)))
}

async fn queue_event_pump(state: UiServerState) {
    loop {
        state.queue.notify.notified().await;
        state.broadcast_snapshot().await;
    }
}

async fn snapshot(State(state): State<UiServerState>) -> Json<UiSnapshot> {
    Json(state.snapshot().await)
}

async fn event_stream(
    State(state): State<UiServerState>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let mut receiver = state.events.subscribe();

    let stream = async_stream::stream! {
        if let Some(event) = sse_json("snapshot", &UiEvent::Snapshot(state.snapshot().await)) {
            yield Ok(event);
        }

        loop {
            match receiver.recv().await {
                Ok(event) => {
                    let name = event_name(&event);
                    if let Some(event) = sse_json(name, &event) {
                        yield Ok(event);
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    if let Some(event) = sse_json("snapshot", &UiEvent::Snapshot(state.snapshot().await)) {
                        yield Ok(event);
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keepalive"),
    )
}

async fn command(
    State(state): State<UiServerState>,
    Path(command): Path<String>,
    body: Option<Json<CommandBody>>,
) -> Json<UiCommandResult> {
    let track_id = body.and_then(|body| body.track_id.clone());
    let result = match command.as_str() {
        "play" => {
            state
                .set_active("play", track_id, UiTransportState::Playing, false)
                .await
        }
        "pause" => state.set_transport(UiTransportState::Paused, true).await,
        "next" => state.step(1).await,
        "previous" => state.step(-1).await,
        "respond" => state.respond(track_id).await,
        "cancel" => state.cancel(track_id).await,
        "clear-recent" => state.clear_recent(track_id).await,
        other => UiCommandResult {
            command: other.to_string(),
            ok: false,
            track_id: None,
            message: Some(format!("unknown command: {other}")),
        },
    };

    let _ = state.events.send(UiEvent::CommandResult(result.clone()));
    state.broadcast_snapshot().await;
    Json(result)
}

async fn audio(Path((track_id, part)): Path<(String, String)>) -> Response {
    let path = match part.as_str() {
        "prompt" | "question" => audio_recorder::question_path(&track_id),
        "answer" => audio_recorder::answer_path(&track_id),
        _ => return StatusCode::NOT_FOUND.into_response(),
    };

    match tokio::fs::read(path).await {
        Ok(bytes) => (
            [(header::CONTENT_TYPE, "audio/wav")],
            axum::body::Bytes::from(bytes),
        )
            .into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn asset(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let candidate = if path.is_empty() { "index.html" } else { path };
    let asset = voice_ui::get(candidate).or_else(|| {
        if path.starts_with("api/") {
            None
        } else {
            voice_ui::index_html()
        }
    });

    match asset {
        Some(asset) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, asset.mime)
            .body(Body::from(asset.bytes))
            .unwrap(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

fn sse_json(name: &str, event: &UiEvent) -> Option<Event> {
    Event::default()
        .event(name)
        .json_data(event)
        .map_err(|err| {
            eprintln!("voice daemon: failed to serialize UI event: {err}");
        })
        .ok()
}

fn event_name(event: &UiEvent) -> &'static str {
    match event {
        UiEvent::Snapshot(_) => "snapshot",
        UiEvent::TrackUpserted(_) => "track_upserted",
        UiEvent::TrackRemoved { .. } => "track_removed",
        UiEvent::ActiveChanged { .. } => "active_changed",
        UiEvent::TransportChanged(_) => "transport_changed",
        UiEvent::CommandResult(_) => "command_result",
        UiEvent::Error { .. } => "error",
    }
}

impl UiServerState {
    async fn snapshot(&self) -> UiSnapshot {
        snapshot_from_daemon(
            &self.queue.snapshot().await,
            self.active_track_id.lock().await.clone(),
            self.transport.lock().await.clone(),
        )
    }

    async fn broadcast_snapshot(&self) {
        let snapshot = self.snapshot().await;
        let _ = self.events.send(UiEvent::Snapshot(snapshot));
    }

    async fn set_active(
        &self,
        command: &str,
        requested_id: Option<String>,
        transport_state: UiTransportState,
        paused: bool,
    ) -> UiCommandResult {
        let snapshot = self.snapshot().await;
        let next_id = requested_id.or(snapshot.active_track_id).or_else(|| {
            snapshot
                .queue_ids
                .first()
                .cloned()
                .or_else(|| snapshot.recent_ids.first().cloned())
        });

        *self.active_track_id.lock().await = next_id.clone();
        *self.transport.lock().await = UiTransport {
            state: transport_state,
            paused,
            position_seconds: 0,
        };

        let _ = self.events.send(UiEvent::ActiveChanged {
            active_track_id: next_id.clone(),
        });
        let _ = self.events.send(UiEvent::TransportChanged(
            self.transport.lock().await.clone(),
        ));

        UiCommandResult {
            command: command.to_string(),
            ok: next_id.is_some(),
            track_id: next_id,
            message: None,
        }
    }

    async fn respond(&self, requested_id: Option<String>) -> UiCommandResult {
        let result = self
            .set_active("respond", requested_id, UiTransportState::Listening, false)
            .await;
        if !result.ok {
            return result;
        }

        let Some(track_id) = result.track_id.clone() else {
            return result;
        };

        let Ok(text) = std::env::var("VOICE_UI_TEST_RESPONSE_TEXT") else {
            self.start_microphone_response(track_id.clone()).await;
            return result;
        };
        if text.trim().is_empty() {
            self.start_microphone_response(track_id.clone()).await;
            return result;
        }

        if let Err(err) = write_synthetic_answer_audio(&track_id) {
            return UiCommandResult {
                command: "respond".to_string(),
                ok: false,
                track_id: Some(track_id),
                message: Some(err),
            };
        }

        let completed = self
            .queue
            .complete_held_item(
                &track_id,
                Some(
                    serde_json::json!({
                        "heard": {
                            "text": text,
                            "sample_rate": 16_000,
                            "audio_duration_ms": 250,
                        },
                        "source": "ui-test-response",
                    })
                    .to_string(),
                ),
            )
            .await;
        if completed {
            *self.transport.lock().await = UiTransport {
                state: UiTransportState::Idle,
                paused: true,
                position_seconds: 0,
            };
            self.broadcast_snapshot().await;
        }

        UiCommandResult {
            command: "respond".to_string(),
            ok: completed,
            track_id: Some(track_id),
            message: (!completed).then(|| "track is not a held UI item".to_string()),
        }
    }

    async fn start_microphone_response(&self, track_id: String) {
        self.active_response_listens
            .lock()
            .await
            .insert(track_id.clone(), None);
        let state = self.clone();
        tokio::spawn(async move {
            let (listen_id, completion_rx) = state
                .queue
                .enqueue_and_wait(
                    UI_INTERNAL_CLIENT_ID.to_string(),
                    VoiceRequest::Listen {
                        max_duration_ms: ui_response_max_duration_ms(),
                    },
                )
                .await;
            let still_active = {
                let mut active_listens = state.active_response_listens.lock().await;
                if let Some(active_listen_id) = active_listens.get_mut(&track_id) {
                    *active_listen_id = Some(listen_id.clone());
                    true
                } else {
                    false
                }
            };
            if !still_active {
                let _ = state.queue.cancel_item(&listen_id).await;
                state.set_response_transport_idle().await;
                state.broadcast_snapshot().await;
                return;
            }
            state.broadcast_snapshot().await;

            let finish = match completion_rx.await {
                Ok(completion) => {
                    if !state
                        .take_active_response_listen(&track_id, &listen_id)
                        .await
                    {
                        state.set_response_transport_idle().await;
                        state.broadcast_snapshot().await;
                        return;
                    }
                    if completion.status == ItemStatus::Completed {
                        let _ = copy_answer_audio(&listen_id, &track_id).await;
                        state
                            .queue
                            .complete_held_item(&track_id, completion.result)
                            .await
                    } else {
                        state
                            .queue
                            .fail_held_item(&track_id, completion.result)
                            .await
                    }
                }
                Err(_) => {
                    if !state
                        .take_active_response_listen(&track_id, &listen_id)
                        .await
                    {
                        state.set_response_transport_idle().await;
                        state.broadcast_snapshot().await;
                        return;
                    }
                    state
                        .queue
                        .fail_held_item(
                            &track_id,
                            Some("internal listen worker dropped".to_string()),
                        )
                        .await
                }
            };

            if finish {
                state.set_response_transport_idle().await;
                state.broadcast_snapshot().await;
            }
        });
    }

    async fn take_active_response_listen(&self, track_id: &str, listen_id: &str) -> bool {
        let mut active_listens = self.active_response_listens.lock().await;
        let Some(Some(active_listen_id)) = active_listens.get(track_id) else {
            return false;
        };
        if active_listen_id != listen_id {
            return false;
        }
        active_listens.remove(track_id);
        true
    }

    async fn set_response_transport_idle(&self) {
        *self.active_track_id.lock().await = None;
        *self.transport.lock().await = UiTransport {
            state: UiTransportState::Idle,
            paused: true,
            position_seconds: 0,
        };
    }

    async fn set_transport(
        &self,
        transport_state: UiTransportState,
        paused: bool,
    ) -> UiCommandResult {
        let active_track_id = self.active_track_id.lock().await.clone();
        let mut transport = self.transport.lock().await;
        transport.state = transport_state;
        transport.paused = paused;
        let _ = self
            .events
            .send(UiEvent::TransportChanged(transport.clone()));

        UiCommandResult {
            command: "pause".to_string(),
            ok: true,
            track_id: active_track_id,
            message: None,
        }
    }

    async fn step(&self, direction: isize) -> UiCommandResult {
        let snapshot = self.snapshot().await;
        let mut ids = snapshot.queue_ids.clone();
        ids.extend(snapshot.recent_ids.iter().cloned());
        let command = if direction > 0 { "next" } else { "previous" };
        if ids.is_empty() {
            return UiCommandResult {
                command: command.to_string(),
                ok: false,
                track_id: None,
                message: Some("no tracks available".to_string()),
            };
        }

        let current_index = snapshot
            .active_track_id
            .as_ref()
            .and_then(|id| ids.iter().position(|candidate| candidate == id))
            .unwrap_or(0);
        let next_index = current_index as isize + direction;
        if next_index < 0 || next_index >= ids.len() as isize {
            *self.transport.lock().await = UiTransport {
                state: UiTransportState::Idle,
                paused: true,
                position_seconds: 0,
            };
            let _ = self.events.send(UiEvent::TransportChanged(
                self.transport.lock().await.clone(),
            ));
            return UiCommandResult {
                command: command.to_string(),
                ok: false,
                track_id: snapshot.active_track_id,
                message: None,
            };
        }
        self.set_active(
            command,
            Some(ids[next_index as usize].clone()),
            UiTransportState::Playing,
            false,
        )
        .await
    }

    async fn cancel(&self, track_id: Option<String>) -> UiCommandResult {
        let track_id = track_id.or_else(|| {
            self.active_track_id
                .try_lock()
                .ok()
                .and_then(|id| id.clone())
        });
        let Some(track_id) = track_id else {
            return UiCommandResult {
                command: "cancel".to_string(),
                ok: false,
                track_id: None,
                message: Some("no active track".to_string()),
            };
        };

        let response_listen_id = self.active_response_listens.lock().await.remove(&track_id);
        if let Some(response_listen_id) = response_listen_id {
            let listen_cancelled = if let Some(response_listen_id) = response_listen_id {
                self.queue.cancel_item(&response_listen_id).await
            } else {
                true
            };
            self.set_response_transport_idle().await;
            self.broadcast_snapshot().await;
            return UiCommandResult {
                command: "cancel".to_string(),
                ok: listen_cancelled,
                track_id: Some(track_id),
                message: None,
            };
        }

        let ok = self.queue.cancel_item(&track_id).await;
        UiCommandResult {
            command: "cancel".to_string(),
            ok,
            track_id: Some(track_id),
            message: None,
        }
    }

    async fn clear_recent(&self, track_id: Option<String>) -> UiCommandResult {
        let Some(track_id) = track_id else {
            return UiCommandResult {
                command: "clear-recent".to_string(),
                ok: false,
                track_id: None,
                message: Some("track_id is required".to_string()),
            };
        };

        self.queue.remove_recent(&track_id).await;
        UiCommandResult {
            command: "clear-recent".to_string(),
            ok: true,
            track_id: Some(track_id),
            message: None,
        }
    }
}

fn write_synthetic_answer_audio(track_id: &str) -> Result<(), String> {
    let sample_rate = 16_000_u32;
    let samples = vec![0.0; sample_rate as usize / 4];
    audio_recorder::save_wav(
        &audio_recorder::answer_path(track_id),
        &samples,
        sample_rate,
    )
}

fn ui_response_max_duration_ms() -> Option<u64> {
    std::env::var("VOICE_UI_RESPONSE_MAX_DURATION_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .or(Some(30_000))
}

async fn copy_answer_audio(from_id: &str, to_id: &str) -> Result<(), String> {
    let from = audio_recorder::answer_path(from_id);
    let to = audio_recorder::answer_path(to_id);
    if !from.exists() {
        return Ok(());
    }
    if let Some(parent) = to.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|err| format!("mkdir {}: {err}", parent.display()))?;
    }
    tokio::fs::copy(&from, &to)
        .await
        .map(|_| ())
        .map_err(|err| format!("copy answer audio: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::{TtsOptions, VoiceRequest};
    use axum::body::{to_bytes, Body};
    use axum::http::{Method, Request};
    use tower::ServiceExt;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[tokio::test]
    async fn missing_static_asset_returns_not_found_for_api_path() {
        let response = asset("/api/missing".parse().unwrap()).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_response_completes_held_track_without_microphone() {
        let _guard = ENV_LOCK.lock().unwrap();
        let audio_dir = std::env::temp_dir().join(format!(
            "voice-ui-response-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let old_audio_dir = std::env::var("VOICE_AUDIO_DIR").ok();
        let old_response = std::env::var("VOICE_UI_TEST_RESPONSE_TEXT").ok();
        std::env::set_var("VOICE_AUDIO_DIR", &audio_dir);
        std::env::set_var("VOICE_UI_TEST_RESPONSE_TEXT", "Run the regression suite.");

        let queue = Arc::new(RequestQueue::new());
        let (event_tx, _) = broadcast::channel(8);
        let state = UiServerState {
            queue: queue.clone(),
            active_track_id: Arc::new(Mutex::new(None)),
            active_response_listens: Arc::new(Mutex::new(BTreeMap::new())),
            transport: Arc::new(Mutex::new(UiTransport {
                state: UiTransportState::Idle,
                paused: true,
                position_seconds: 0,
            })),
            events: event_tx,
        };
        let track_id = queue
            .enqueue_ui_held(
                "codex".to_string(),
                VoiceRequest::Converse {
                    text: "Should I run tests?".to_string(),
                    voice: None,
                    max_duration_ms: None,
                    options: TtsOptions::default(),
                },
            )
            .await;

        let result = state.respond(Some(track_id.clone())).await;
        assert!(result.ok, "{result:?}");

        let snapshot = state.snapshot().await;
        assert!(snapshot.queue_ids.is_empty());
        assert_eq!(snapshot.recent_ids, vec![track_id.clone()]);
        let track = &snapshot.tracks[&track_id];
        assert_eq!(track.answer.as_deref(), Some("Run the regression suite."));
        assert_eq!(
            track.audio.answer_url,
            Some(format!("/api/ui/audio/{track_id}/answer"))
        );

        match old_audio_dir {
            Some(value) => std::env::set_var("VOICE_AUDIO_DIR", value),
            None => std::env::remove_var("VOICE_AUDIO_DIR"),
        }
        match old_response {
            Some(value) => std::env::set_var("VOICE_UI_TEST_RESPONSE_TEXT", value),
            None => std::env::remove_var("VOICE_UI_TEST_RESPONSE_TEXT"),
        }
        let _ = std::fs::remove_dir_all(audio_dir);
    }

    #[tokio::test]
    async fn cancel_response_listen_keeps_visible_track_queued() {
        let queue = Arc::new(RequestQueue::new());
        let (event_tx, _) = broadcast::channel(8);
        let state = UiServerState {
            queue: queue.clone(),
            active_track_id: Arc::new(Mutex::new(None)),
            active_response_listens: Arc::new(Mutex::new(BTreeMap::new())),
            transport: Arc::new(Mutex::new(UiTransport {
                state: UiTransportState::Idle,
                paused: true,
                position_seconds: 0,
            })),
            events: event_tx,
        };
        let track_id = queue
            .enqueue_ui_held(
                "codex".to_string(),
                VoiceRequest::Converse {
                    text: "Should I keep listening?".to_string(),
                    voice: None,
                    max_duration_ms: None,
                    options: TtsOptions::default(),
                },
            )
            .await;
        let (listen_id, completion_rx) = queue
            .enqueue_and_wait(
                UI_INTERNAL_CLIENT_ID.to_string(),
                VoiceRequest::Listen {
                    max_duration_ms: Some(1_000),
                },
            )
            .await;
        *state.active_track_id.lock().await = Some(track_id.clone());
        *state.transport.lock().await = UiTransport {
            state: UiTransportState::Listening,
            paused: false,
            position_seconds: 0,
        };
        state
            .active_response_listens
            .lock()
            .await
            .insert(track_id.clone(), Some(listen_id));

        let result = state.cancel(Some(track_id.clone())).await;
        assert!(result.ok, "{result:?}");

        let completion = tokio::time::timeout(Duration::from_secs(1), completion_rx)
            .await
            .expect("listen cancellation should signal waiter")
            .expect("listen waiter should receive cancellation");
        assert_eq!(completion.status, ItemStatus::Failed);

        let snapshot = state.snapshot().await;
        assert_eq!(snapshot.active_track_id, None);
        assert_eq!(snapshot.transport.state, UiTransportState::Idle);
        assert_eq!(snapshot.queue_ids, vec![track_id.clone()]);
        assert_eq!(
            snapshot.tracks[&track_id].lifecycle,
            crate::ui_state::UiLifecycle::Queued
        );
    }

    #[tokio::test]
    async fn next_at_queue_end_stops_instead_of_replaying_last_track() {
        let queue = Arc::new(RequestQueue::new());
        let (event_tx, _) = broadcast::channel(8);
        let state = UiServerState {
            queue: queue.clone(),
            active_track_id: Arc::new(Mutex::new(None)),
            active_response_listens: Arc::new(Mutex::new(BTreeMap::new())),
            transport: Arc::new(Mutex::new(UiTransport {
                state: UiTransportState::Idle,
                paused: true,
                position_seconds: 0,
            })),
            events: event_tx,
        };
        let track_id = queue
            .enqueue_ui_held(
                "codex".to_string(),
                VoiceRequest::Speak {
                    text: "Only track.".to_string(),
                    voice: None,
                    speed: None,
                    options: TtsOptions::default(),
                },
            )
            .await;
        *state.active_track_id.lock().await = Some(track_id.clone());
        *state.transport.lock().await = UiTransport {
            state: UiTransportState::Playing,
            paused: false,
            position_seconds: 0,
        };

        let result = state.step(1).await;
        assert!(!result.ok, "{result:?}");
        assert_eq!(result.track_id.as_deref(), Some(track_id.as_str()));
        assert_eq!(*state.active_track_id.lock().await, Some(track_id));
        assert_eq!(
            *state.transport.lock().await,
            UiTransport {
                state: UiTransportState::Idle,
                paused: true,
                position_seconds: 0,
            }
        );
    }

    #[tokio::test]
    async fn http_snapshot_command_and_audio_round_trip_without_microphone() {
        let _guard = ENV_LOCK.lock().unwrap();
        let audio_dir = std::env::temp_dir().join(format!(
            "voice-ui-http-test-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let old_audio_dir = std::env::var("VOICE_AUDIO_DIR").ok();
        let old_response = std::env::var("VOICE_UI_TEST_RESPONSE_TEXT").ok();
        std::env::set_var("VOICE_AUDIO_DIR", &audio_dir);
        std::env::set_var("VOICE_UI_TEST_RESPONSE_TEXT", "Please open the PR.");

        let queue = Arc::new(RequestQueue::new());
        let (event_tx, _) = broadcast::channel(8);
        let state = UiServerState {
            queue: queue.clone(),
            active_track_id: Arc::new(Mutex::new(None)),
            active_response_listens: Arc::new(Mutex::new(BTreeMap::new())),
            transport: Arc::new(Mutex::new(UiTransport {
                state: UiTransportState::Idle,
                paused: true,
                position_seconds: 0,
            })),
            events: event_tx,
        };
        let track_id = queue
            .enqueue_ui_held(
                "codex".to_string(),
                VoiceRequest::Converse {
                    text: "Should I open the PR?".to_string(),
                    voice: None,
                    max_duration_ms: None,
                    options: TtsOptions::default(),
                },
            )
            .await;
        let app = router(state);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/ui/snapshot")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let snapshot = serde_json::from_slice::<UiSnapshot>(
            &to_bytes(response.into_body(), usize::MAX).await.unwrap(),
        )
        .unwrap();
        assert_eq!(snapshot.queue_ids, vec![track_id.clone()]);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/ui/commands/respond")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(format!(r#"{{"trackId":"{track_id}"}}"#)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let result = serde_json::from_slice::<UiCommandResult>(
            &to_bytes(response.into_body(), usize::MAX).await.unwrap(),
        )
        .unwrap();
        assert!(result.ok, "{result:?}");

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/ui/snapshot")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let snapshot = serde_json::from_slice::<UiSnapshot>(
            &to_bytes(response.into_body(), usize::MAX).await.unwrap(),
        )
        .unwrap();
        assert!(snapshot.queue_ids.is_empty());
        assert_eq!(snapshot.recent_ids, vec![track_id.clone()]);
        assert_eq!(
            snapshot.tracks[&track_id].answer.as_deref(),
            Some("Please open the PR.")
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/api/ui/audio/{track_id}/answer"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "audio/wav"
        );
        assert!(!to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .is_empty());

        match old_audio_dir {
            Some(value) => std::env::set_var("VOICE_AUDIO_DIR", value),
            None => std::env::remove_var("VOICE_AUDIO_DIR"),
        }
        match old_response {
            Some(value) => std::env::set_var("VOICE_UI_TEST_RESPONSE_TEXT", value),
            None => std::env::remove_var("VOICE_UI_TEST_RESPONSE_TEXT"),
        }
        let _ = std::fs::remove_dir_all(audio_dir);
    }
}
