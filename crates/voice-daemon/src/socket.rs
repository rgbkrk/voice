//! Unix socket server for the voice daemon.
//!
//! Uses the voice-protocol frame codec (length-prefixed typed frames)
//! instead of newline-delimited JSON.

use crate::config::DaemonConfig;
use crate::queue::RequestQueue;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::UnixListener;
use uuid::Uuid;
use voice_protocol::frames::{read_frame, write_frame, Frame, FrameType};
use voice_protocol::rpc::{self, Response};

pub fn socket_path() -> PathBuf {
    let dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".voice");
    std::fs::create_dir_all(&dir).ok();
    dir.join("daemon.sock")
}

pub async fn serve(
    queue: Arc<RequestQueue>,
    config: Arc<DaemonConfig>,
    automerge: Arc<tokio::sync::Mutex<crate::automerge_state::AutomergeState>>,
) {
    let path = socket_path();

    if path.exists() {
        std::fs::remove_file(&path).ok();
    }

    let listener = match UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("voiced: failed to bind {}: {}", path.display(), e);
            return;
        }
    };

    eprintln!("voiced: listening on {}", path.display());

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let queue = queue.clone();
                let config = config.clone();
                let client_id = Uuid::new_v4().to_string()[..8].to_string();
                eprintln!("voiced: client connected ({})", client_id);
                let automerge_clone = automerge.clone();
                tokio::spawn(handle_client(
                    stream,
                    queue,
                    config,
                    client_id,
                    automerge_clone,
                ));
            }
            Err(e) => eprintln!("voiced: accept error: {}", e),
        }
    }
}

async fn handle_client(
    stream: tokio::net::UnixStream,
    queue: Arc<RequestQueue>,
    config: Arc<DaemonConfig>,
    client_id: String,
    automerge: Arc<tokio::sync::Mutex<crate::automerge_state::AutomergeState>>,
) {
    let (mut reader, mut writer) = stream.into_split();

    loop {
        let frame = match read_frame(&mut reader).await {
            Ok(Some(f)) => f,
            Ok(None) => break, // EOF
            Err(e) => {
                eprintln!("voiced: read error ({}): {}", client_id, e);
                break;
            }
        };

        match frame.frame_type {
            FrameType::Request => {
                let response = match frame.json::<rpc::Request>() {
                    Ok(req) => dispatch(req, &queue, &config, &client_id, &automerge).await,
                    Err(e) => Response::error(
                        None,
                        rpc::PARSE_ERROR,
                        format!("Invalid request JSON: {}", e),
                    ),
                };

                let json = serde_json::to_vec(&response).unwrap();
                let resp_frame = Frame::response(&json);
                if write_frame(&mut writer, &resp_frame).await.is_err() {
                    break;
                }
            }
            other => {
                eprintln!(
                    "voiced: unexpected frame type {:?} from {}",
                    other, client_id
                );
            }
        }
    }

    eprintln!("voiced: client disconnected ({})", client_id);
}

async fn dispatch(
    req: rpc::Request,
    queue: &Arc<RequestQueue>,
    config: &Arc<DaemonConfig>,
    client_id: &str,
    automerge: &Arc<tokio::sync::Mutex<crate::automerge_state::AutomergeState>>,
) -> Response {
    use crate::queue::VoiceRequest;

    let wait = req
        .params
        .get("wait")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Build the voice request from params
    let voice_req = match req.method.as_str() {
        "speak" => {
            let text = req.params.get("text").and_then(|v| v.as_str());
            let Some(text) = text else {
                return Response::error(req.id, rpc::INVALID_PARAMS, "Missing param: text");
            };
            let voice = req
                .params
                .get("voice")
                .and_then(|v| v.as_str())
                .map(String::from);
            let speed = req.params.get("speed").and_then(|v| v.as_f64());
            VoiceRequest::Speak {
                text: text.to_string(),
                voice,
                speed,
            }
        }
        "listen" => {
            let max_duration_ms = req.params.get("max_duration_ms").and_then(|v| v.as_u64());
            VoiceRequest::Listen { max_duration_ms }
        }
        "converse" => {
            let text = req.params.get("text").and_then(|v| v.as_str());
            let Some(text) = text else {
                return Response::error(req.id, rpc::INVALID_PARAMS, "Missing param: text");
            };
            let voice = req
                .params
                .get("voice")
                .and_then(|v| v.as_str())
                .map(String::from);
            VoiceRequest::Converse {
                text: text.to_string(),
                voice,
            }
        }
        "replay_audio" => {
            let queue_id = req.params.get("queue_id").and_then(|v| v.as_str());
            let Some(queue_id) = queue_id else {
                return Response::error(req.id, rpc::INVALID_PARAMS, "Missing param: queue_id");
            };
            let part = req.params.get("part").and_then(|v| v.as_str());
            let Some(part) = part else {
                return Response::error(req.id, rpc::INVALID_PARAMS, "Missing param: part");
            };

            let path = match part {
                "question" => crate::audio_recorder::question_path(queue_id),
                "answer" => crate::audio_recorder::answer_path(queue_id),
                _ => {
                    return Response::error(
                        req.id,
                        rpc::INVALID_PARAMS,
                        "param 'part' must be 'question' or 'answer'",
                    );
                }
            };

            // Read WAV file
            let (samples, sample_rate) = match crate::audio_recorder::read_wav(&path) {
                Ok(result) => result,
                Err(e) => {
                    return Response::error(req.id, -32000, format!("Audio file not found: {}", e));
                }
            };

            // Play through rodio
            let duration_ms = tokio::task::spawn_blocking(move || {
                use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, Player};
                use std::num::NonZero;
                use std::time::Instant;

                let mut stream = match DeviceSinkBuilder::open_default_sink() {
                    Ok(s) => s,
                    Err(e) => return Err(format!("audio device: {}", e)),
                };
                stream.log_on_drop(false);
                let player = Player::connect_new(stream.mixer());

                let channels = NonZero::new(1u16).unwrap();
                let rate = NonZero::new(sample_rate).unwrap();
                let source = SamplesBuffer::new(channels, rate, samples);
                player.append(source);

                let started = Instant::now();
                while !player.empty() {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                Ok(started.elapsed().as_millis() as u64)
            })
            .await;

            match duration_ms {
                Ok(Ok(ms)) => {
                    return Response::success(req.id, serde_json::json!({ "duration_ms": ms }));
                }
                Ok(Err(e)) => {
                    return Response::error(req.id, -32000, format!("Playback error: {}", e));
                }
                Err(e) => {
                    return Response::error(req.id, -32000, format!("Task panicked: {}", e));
                }
            }
        }
        "cancel" => {
            let count = queue.cancel_client(client_id).await;
            return Response::success(req.id, serde_json::json!({ "cancelled_count": count }));
        }
        "cancel_item" => {
            let queue_id = req.params.get("queue_id").and_then(|v| v.as_str());
            let Some(queue_id) = queue_id else {
                return Response::error(req.id, rpc::INVALID_PARAMS, "Missing param: queue_id");
            };

            // Remove from queue (both pending and current)
            let removed = queue.cancel_item(queue_id).await;

            if removed {
                // Update Automerge state
                let snapshot = queue.snapshot().await;
                let mut am = automerge.lock().await;
                am.update(&snapshot);
                if let Err(e) = am.save() {
                    eprintln!("voiced: failed to save automerge after cancel: {}", e);
                }

                return Response::success(req.id, serde_json::json!({ "cancelled": true }));
            } else {
                return Response::success(req.id, serde_json::json!({ "cancelled": false }));
            }
        }
        "status" => {
            let state = queue.snapshot().await;
            return Response::success(req.id, serde_json::to_value(&state).unwrap());
        }
        "set_voice" => {
            let voice = req.params.get("voice").and_then(|v| v.as_str());
            let Some(voice) = voice else {
                return Response::error(req.id, rpc::INVALID_PARAMS, "Missing param: voice");
            };
            config.set_voice_name(voice.to_string());
            return Response::success(req.id, serde_json::json!({ "voice": voice }));
        }
        "set_speed" => {
            let speed = req.params.get("speed").and_then(|v| v.as_f64());
            let Some(speed) = speed else {
                return Response::error(req.id, rpc::INVALID_PARAMS, "Missing param: speed");
            };
            if speed <= 0.0 || speed > 5.0 {
                return Response::error(
                    req.id,
                    rpc::INVALID_PARAMS,
                    "Speed must be between 0 (exclusive) and 5 (inclusive)",
                );
            }
            config.set_speed(speed as f32);
            return Response::success(req.id, serde_json::json!({ "speed": speed }));
        }
        "list_voices" => {
            let voices: Vec<serde_json::Value> = voice_tts::catalog::ALL_VOICES
                .iter()
                .map(|v| {
                    let builtin = voice_tts::catalog::is_builtin(v.id);
                    serde_json::json!({
                        "id": v.id,
                        "name": v.name,
                        "language": v.language,
                        "gender": v.gender,
                        "traits": v.traits,
                        "builtin": builtin,
                    })
                })
                .collect();
            let current = config.get_voice_name();
            return Response::success(
                req.id,
                serde_json::json!({ "voices": voices, "current": current }),
            );
        }
        _ => {
            return Response::error(
                req.id,
                rpc::METHOD_NOT_FOUND,
                format!("Method not found: {}", req.method),
            );
        }
    };

    if !wait {
        // Fire-and-forget: enqueue and return immediately
        let queue_id = match voice_req {
            VoiceRequest::Speak { text, voice, speed } => {
                queue
                    .enqueue_speak(client_id.to_string(), text, voice, speed)
                    .await
            }
            VoiceRequest::Listen { max_duration_ms } => {
                queue
                    .enqueue_listen(client_id.to_string(), max_duration_ms)
                    .await
            }
            VoiceRequest::Converse { text, voice } => {
                queue
                    .enqueue_converse(client_id.to_string(), text, voice)
                    .await
            }
        };
        return Response::success(
            req.id,
            serde_json::json!({ "queue_id": queue_id, "status": "queued" }),
        );
    }

    // Wait mode: register waiter atomically with enqueue to prevent race
    let (queue_id, rx) = queue
        .enqueue_and_wait(client_id.to_string(), voice_req)
        .await;

    match rx.await {
        Ok(result) => Response::success(
            req.id,
            serde_json::json!({
                "queue_id": queue_id,
                "status": result.status,
                "result": result.result,
            }),
        ),
        Err(_) => Response::error(req.id, -32000, "Queue item dropped before completion"),
    }
}

pub fn cleanup() {
    let path = socket_path();
    if path.exists() {
        std::fs::remove_file(&path).ok();
    }
}
