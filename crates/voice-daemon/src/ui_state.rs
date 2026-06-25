use crate::audio_recorder;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use voice_protocol::rpc::{DaemonState, ItemStatus, QueueItem};

const DEFAULT_COLORS: [&str; 5] = ["#82aaff", "#f0b878", "#e89a6a", "#9ad7c2", "#b58cff"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UiSnapshot {
    pub connected: bool,
    pub ready: bool,
    pub daemon_status: String,
    pub transport: UiTransport,
    pub active_track_id: Option<String>,
    pub queue_ids: Vec<String>,
    pub recent_ids: Vec<String>,
    pub tracks: BTreeMap<String, UiTrack>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UiTransport {
    pub state: UiTransportState,
    pub paused: bool,
    pub position_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum UiTransportState {
    Idle,
    Playing,
    Paused,
    Listening,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UiTrack {
    pub id: String,
    pub agent: UiAgent,
    pub intent: UiIntent,
    pub lifecycle: UiLifecycle,
    pub prompt: String,
    pub answer: Option<String>,
    pub audio: UiAudio,
    pub created_at: u64,
    pub completed_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UiAgent {
    pub name: String,
    pub initial: String,
    pub color: String,
    pub repo: String,
    pub branch: String,
    pub model: String,
    pub session: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum UiIntent {
    Play,
    Respond,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum UiLifecycle {
    Queued,
    Preparing,
    Ready,
    Active,
    Listening,
    Completed,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
pub struct UiAudio {
    pub prompt_url: Option<String>,
    pub answer_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "type", content = "payload")]
pub enum UiEvent {
    Snapshot(UiSnapshot),
    TrackUpserted(UiTrack),
    TrackRemoved { id: String },
    ActiveChanged { active_track_id: Option<String> },
    TransportChanged(UiTransport),
    CommandResult(UiCommandResult),
    Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct UiCommandResult {
    pub command: String,
    pub ok: bool,
    pub track_id: Option<String>,
    pub message: Option<String>,
}

pub fn snapshot_from_daemon(
    daemon: &DaemonState,
    active_track_id: Option<String>,
    transport: UiTransport,
) -> UiSnapshot {
    let mut tracks = BTreeMap::new();
    let mut queue_ids = Vec::new();
    let mut recent_ids = Vec::new();

    let mut ordered_items = Vec::new();
    if let Some(current) = &daemon.current {
        ordered_items.push(current.clone());
    }
    ordered_items.extend(daemon.pending.iter().cloned());
    ordered_items.extend(daemon.recent.iter().cloned());

    for (index, item) in ordered_items.iter().enumerate() {
        let track = track_from_item(item, index, active_track_id.as_deref());
        if daemon.recent.iter().any(|recent| recent.id == item.id) {
            recent_ids.push(item.id.clone());
        } else {
            queue_ids.push(item.id.clone());
        }
        tracks.insert(track.id.clone(), track);
    }

    UiSnapshot {
        connected: true,
        ready: true,
        daemon_status: daemon.status.clone(),
        transport,
        active_track_id,
        queue_ids,
        recent_ids,
        tracks,
    }
}

fn track_from_item(item: &QueueItem, index: usize, active_track_id: Option<&str>) -> UiTrack {
    let intent = intent_for_method(&item.method);
    let lifecycle = lifecycle_for_item(item, &intent, active_track_id);
    let prompt = item
        .text_preview
        .clone()
        .or_else(|| item.result.clone())
        .unwrap_or_else(|| item.method.replace('_', " "));
    let answer = answer_from_result(item.result.as_deref());
    let client_id = item.client_id.trim();

    UiTrack {
        id: item.id.clone(),
        agent: UiAgent {
            name: agent_name(client_id),
            initial: agent_initial(client_id),
            color: DEFAULT_COLORS[index % DEFAULT_COLORS.len()].to_string(),
            repo: item
                .repo
                .clone()
                .unwrap_or_else(|| "voice daemon".to_string()),
            branch: item.method.clone(),
            model: client_id.to_string(),
            session: client_id.to_string(),
        },
        intent,
        lifecycle,
        prompt,
        answer,
        audio: audio_for_item(&item.id),
        created_at: item.created_at,
        completed_at: item.completed_at,
    }
}

fn lifecycle_for_item(
    item: &QueueItem,
    intent: &UiIntent,
    active_track_id: Option<&str>,
) -> UiLifecycle {
    if active_track_id == Some(item.id.as_str()) {
        return match intent {
            UiIntent::Respond => UiLifecycle::Listening,
            UiIntent::Play => UiLifecycle::Active,
        };
    }

    match item.status {
        ItemStatus::Queued => {
            if audio_recorder::question_path(&item.id).exists() {
                UiLifecycle::Ready
            } else {
                UiLifecycle::Queued
            }
        }
        ItemStatus::Processing => UiLifecycle::Preparing,
        ItemStatus::Completed => UiLifecycle::Completed,
        ItemStatus::Failed => UiLifecycle::Failed,
    }
}

fn intent_for_method(method: &str) -> UiIntent {
    match method {
        "converse" | "listen" | "stream_transcribe" => UiIntent::Respond,
        _ => UiIntent::Play,
    }
}

fn audio_for_item(id: &str) -> UiAudio {
    UiAudio {
        prompt_url: audio_url_if_exists(id, "prompt", &audio_recorder::question_path(id)),
        answer_url: audio_url_if_exists(id, "answer", &audio_recorder::answer_path(id)),
    }
}

fn audio_url_if_exists(id: &str, part: &str, path: &Path) -> Option<String> {
    path.exists().then(|| format!("/api/ui/audio/{id}/{part}"))
}

fn answer_from_result(result: Option<&str>) -> Option<String> {
    let result = result?;
    let value = serde_json::from_str::<serde_json::Value>(result).ok()?;
    value
        .pointer("/heard/text")
        .and_then(|text| text.as_str())
        .or_else(|| value.get("text").and_then(|text| text.as_str()))
        .map(ToOwned::to_owned)
}

fn agent_name(client_id: &str) -> String {
    let first = client_id
        .split(['.', '_', ':', '-'])
        .find(|part| !part.is_empty())
        .unwrap_or("daemon");
    let mut chars = first.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => "Daemon".to_string(),
    }
}

fn agent_initial(client_id: &str) -> String {
    client_id
        .chars()
        .find(|c| c.is_ascii_alphanumeric())
        .unwrap_or('D')
        .to_ascii_uppercase()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, method: &str, status: ItemStatus) -> QueueItem {
        QueueItem {
            id: id.to_string(),
            client_id: "codex.repo.main.gpt-5".to_string(),
            method: method.to_string(),
            status,
            held_for_ui: false,
            created_at: 10,
            text_preview: Some("Need your decision.".to_string()),
            result: None,
            repo: Some("rgbkrk/voice".to_string()),
            completed_at: None,
            auto_clear_at: None,
        }
    }

    #[test]
    fn snapshot_projects_pending_and_recent_tracks() {
        let daemon = DaemonState {
            status: "queued".to_string(),
            current: None,
            pending: vec![item("one", "converse", ItemStatus::Queued)],
            recent: vec![item("two", "speak", ItemStatus::Completed)],
        };

        let snapshot = snapshot_from_daemon(
            &daemon,
            Some("one".to_string()),
            UiTransport {
                state: UiTransportState::Listening,
                paused: false,
                position_seconds: 0,
            },
        );

        assert_eq!(snapshot.queue_ids, vec!["one"]);
        assert_eq!(snapshot.recent_ids, vec!["two"]);
        assert_eq!(snapshot.tracks["one"].intent, UiIntent::Respond);
        assert_eq!(snapshot.tracks["one"].lifecycle, UiLifecycle::Listening);
        assert_eq!(snapshot.tracks["two"].lifecycle, UiLifecycle::Completed);
    }
}
