use crate::audio_recorder;
use crate::queue::RequestQueue;
use crate::ui_state::{
    snapshot_from_daemon, UiCommandResult, UiEvent, UiSnapshot, UiTransport, UiTransportState,
};
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode, Uri};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::{broadcast, Mutex};

#[derive(Clone)]
struct UiServerState {
    queue: Arc<RequestQueue>,
    active_track_id: Arc<Mutex<Option<String>>>,
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
        transport: Arc::new(Mutex::new(UiTransport {
            state: UiTransportState::Idle,
            paused: true,
            position_seconds: 0,
        })),
        events: event_tx,
    };

    tokio::spawn(queue_event_pump(state.clone()));

    let app = Router::new()
        .route("/api/ui/snapshot", get(snapshot))
        .route("/api/ui/events", get(event_stream))
        .route("/api/ui/commands/:command", post(command))
        .route("/api/ui/audio/:track_id/:part", get(audio))
        .fallback(get(asset))
        .with_state(state);

    eprintln!("voice daemon: UI listening on http://{addr}");
    if let Err(err) = axum::serve(listener, app).await {
        eprintln!("voice daemon: UI server stopped: {err}");
    }
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
        "respond" => {
            state
                .set_active("respond", track_id, UiTransportState::Listening, false)
                .await
        }
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
        if ids.is_empty() {
            return UiCommandResult {
                command: if direction > 0 { "next" } else { "previous" }.to_string(),
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
        let next_index = (current_index as isize + direction)
            .clamp(0, ids.len().saturating_sub(1) as isize) as usize;
        self.set_active(
            if direction > 0 { "next" } else { "previous" },
            Some(ids[next_index].clone()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn missing_static_asset_returns_not_found_for_api_path() {
        let response = asset("/api/missing".parse().unwrap()).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
