//! Unix socket server for the voice daemon.
//!
//! Uses the voice-protocol frame codec (length-prefixed typed frames)
//! instead of newline-delimited JSON.

use crate::config::DaemonConfig;
use crate::queue::{RequestQueue, StreamSpeakRequest};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::UnixListener;
use uuid::Uuid;
use voice_audio::AudioOutputFormat;
use voice_protocol::frames::{read_frame, write_frame, Frame, FrameType};
use voice_protocol::rpc::{self, Response};
use voice_stream::{TtsStreamEvent, DEFAULT_FRAME_MS};

pub fn socket_path() -> PathBuf {
    let path = voice_protocol::client::daemon_socket_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    path
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
            FrameType::Request => match frame.json::<rpc::Request>() {
                Ok(req) if req.method == "stream_speak" => {
                    if dispatch_stream_speak(
                        req,
                        &queue,
                        &config,
                        &client_id,
                        &automerge,
                        &mut writer,
                    )
                    .await
                    .is_err()
                    {
                        break;
                    }
                }
                Ok(req) => {
                    let response = dispatch(req, &queue, &config, &client_id, &automerge).await;
                    if write_response(&mut writer, &response).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let response = Response::error(
                        None,
                        rpc::PARSE_ERROR,
                        format!("Invalid request JSON: {}", e),
                    );
                    if write_response(&mut writer, &response).await.is_err() {
                        break;
                    }
                }
            },
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

async fn write_response(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    response: &Response,
) -> std::io::Result<()> {
    let json = serde_json::to_vec(response).unwrap();
    write_frame(writer, &Frame::response(&json)).await
}

async fn write_stream_event(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    event: &TtsStreamEvent,
) -> std::io::Result<()> {
    let envelope = rpc::Event::new(event.event_name(), serde_json::to_value(event).unwrap());
    let json = serde_json::to_vec(&envelope).unwrap();
    write_frame(writer, &Frame::event(&json)).await
}

async fn dispatch_stream_speak(
    req: rpc::Request,
    queue: &Arc<RequestQueue>,
    config: &Arc<DaemonConfig>,
    client_id: &str,
    automerge: &Arc<tokio::sync::Mutex<crate::automerge_state::AutomergeState>>,
    writer: &mut tokio::net::unix::OwnedWriteHalf,
) -> std::io::Result<()> {
    if !(req.params.is_null() || req.params.is_object()) {
        let response = Response::error(req.id, rpc::INVALID_PARAMS, "params must be an object");
        return write_response(writer, &response).await;
    }

    let text = match required_string_param(&req, "text") {
        Ok(text) => text,
        Err(resp) => return write_response(writer, &resp).await,
    };
    let voice = match optional_voice_param(&req) {
        Ok(voice) => voice,
        Err(resp) => return write_response(writer, &resp).await,
    };
    let speed = match optional_speed_param(&req) {
        Ok(speed) => speed,
        Err(resp) => return write_response(writer, &resp).await,
    };
    let sample_rate = match optional_u32_param(&req, "sample_rate", 8_000, 96_000) {
        Ok(rate) => rate.unwrap_or(24_000),
        Err(resp) => return write_response(writer, &resp).await,
    };
    let frame_ms = match optional_u32_param(&req, "frame_ms", 5, 100) {
        Ok(frame_ms) => frame_ms.unwrap_or(DEFAULT_FRAME_MS),
        Err(resp) => return write_response(writer, &resp).await,
    };

    let stream_id = Uuid::new_v4().to_string();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<TtsStreamEvent>(16);
    let queue_id = queue
        .enqueue_stream_speak(
            client_id.to_string(),
            StreamSpeakRequest {
                text,
                stream_id: stream_id.clone(),
                voice: voice.or_else(|| Some(config.get_voice_name())),
                speed: speed.or_else(|| Some(config.get_speed() as f64)),
                sample_rate,
                frame_ms,
                event_tx,
            },
        )
        .await;
    sync_automerge(queue, automerge).await;

    let response = Response::success(
        req.id,
        serde_json::json!({
            "queue_id": queue_id,
            "stream_id": stream_id,
            "status": "queued",
        }),
    );
    write_response(writer, &response).await?;

    let mut saw_terminal = false;
    while let Some(event) = event_rx.recv().await {
        let terminal = event.is_terminal();
        write_stream_event(writer, &event).await?;
        if terminal {
            saw_terminal = true;
            break;
        }
    }

    if !saw_terminal {
        let event = TtsStreamEvent::error(stream_id, "stream closed before terminal event");
        write_stream_event(writer, &event).await?;
    }

    Ok(())
}

async fn sync_automerge(
    queue: &Arc<RequestQueue>,
    automerge: &Arc<tokio::sync::Mutex<crate::automerge_state::AutomergeState>>,
) {
    let snapshot = queue.snapshot().await;
    let mut am = automerge.lock().await;
    am.update(&snapshot);
    if let Err(e) = am.save() {
        eprintln!("voiced: failed to save automerge doc: {}", e);
    }
}

async fn dispatch(
    req: rpc::Request,
    queue: &Arc<RequestQueue>,
    config: &Arc<DaemonConfig>,
    client_id: &str,
    automerge: &Arc<tokio::sync::Mutex<crate::automerge_state::AutomergeState>>,
) -> Response {
    use crate::queue::VoiceRequest;

    if !(req.params.is_null() || req.params.is_object()) {
        return Response::error(req.id, rpc::INVALID_PARAMS, "params must be an object");
    }

    let wait = match wait_param(&req) {
        Ok(wait) => wait,
        Err(resp) => return resp,
    };

    // Build the voice request from params
    let voice_req = match req.method.as_str() {
        "speak" => {
            let text = match required_string_param(&req, "text") {
                Ok(text) => text,
                Err(resp) => return resp,
            };
            let voice = match optional_voice_param(&req) {
                Ok(voice) => voice,
                Err(resp) => return resp,
            };
            let speed = match optional_speed_param(&req) {
                Ok(speed) => speed,
                Err(resp) => return resp,
            };
            VoiceRequest::Speak { text, voice, speed }
        }
        "synthesize" => {
            let text = match required_string_param(&req, "text") {
                Ok(text) => text,
                Err(resp) => return resp,
            };
            let output_path = match required_string_param(&req, "output_path") {
                Ok(output_path) => output_path,
                Err(resp) => return resp,
            };
            let voice = match optional_voice_param(&req) {
                Ok(voice) => voice,
                Err(resp) => return resp,
            };
            let speed = match optional_speed_param(&req) {
                Ok(speed) => speed,
                Err(resp) => return resp,
            };
            let output_format = match optional_audio_output_format_param(&req) {
                Ok(output_format) => output_format,
                Err(resp) => return resp,
            };
            VoiceRequest::Synthesize {
                text,
                output_path,
                output_format,
                voice,
                speed,
            }
        }
        "listen" => {
            let max_duration_ms = match optional_duration_param(&req, "max_duration_ms") {
                Ok(duration) => duration,
                Err(resp) => return resp,
            };
            VoiceRequest::Listen { max_duration_ms }
        }
        "converse" => {
            let text = match required_string_param(&req, "text") {
                Ok(text) => text,
                Err(resp) => return resp,
            };
            let voice = match optional_voice_param(&req) {
                Ok(voice) => voice,
                Err(resp) => return resp,
            };
            VoiceRequest::Converse { text, voice }
        }
        "replay_audio" => {
            let queue_id = match required_string_param(&req, "queue_id") {
                Ok(queue_id) => queue_id,
                Err(resp) => return resp,
            };
            let part = match required_string_param(&req, "part") {
                Ok(part) => part,
                Err(resp) => return resp,
            };

            let path = match part.as_str() {
                "question" => crate::audio_recorder::question_path(&queue_id),
                "answer" => crate::audio_recorder::answer_path(&queue_id),
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
            sync_automerge(queue, automerge).await;
            return Response::success(req.id, serde_json::json!({ "cancelled_count": count }));
        }
        "cancel_item" => {
            let queue_id = match required_string_param(&req, "queue_id") {
                Ok(queue_id) => queue_id,
                Err(resp) => return resp,
            };

            // Remove from queue (both pending and current)
            let removed = queue.cancel_item(&queue_id).await;

            if removed {
                sync_automerge(queue, automerge).await;
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
            let voice = match required_voice_param(&req) {
                Ok(voice) => voice,
                Err(resp) => return resp,
            };
            config.set_voice_name(voice.clone());
            return Response::success(req.id, serde_json::json!({ "voice": voice }));
        }
        "set_speed" => {
            let speed = match required_speed_param(&req) {
                Ok(speed) => speed,
                Err(resp) => return resp,
            };
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
            VoiceRequest::Synthesize {
                text,
                output_path,
                output_format,
                voice,
                speed,
            } => {
                queue
                    .enqueue_synthesize(
                        client_id.to_string(),
                        text,
                        output_path,
                        output_format,
                        voice,
                        speed,
                    )
                    .await
            }
            VoiceRequest::StreamSpeak(_) => {
                unreachable!("stream_speak is handled before dispatch")
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
        sync_automerge(queue, automerge).await;
        return Response::success(
            req.id,
            serde_json::json!({ "queue_id": queue_id, "status": "queued" }),
        );
    }

    // Wait mode: register waiter atomically with enqueue to prevent race
    let (queue_id, rx) = queue
        .enqueue_and_wait(client_id.to_string(), voice_req)
        .await;
    sync_automerge(queue, automerge).await;

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

fn wait_param(req: &rpc::Request) -> Result<bool, Response> {
    match req.params.get("wait") {
        Some(value) => value.as_bool().ok_or_else(|| {
            Response::error(
                req.id.clone(),
                rpc::INVALID_PARAMS,
                "param 'wait' must be a boolean",
            )
        }),
        None => Ok(req.method == "synthesize"),
    }
}

fn required_string_param(req: &rpc::Request, name: &str) -> Result<String, Response> {
    match req.params.get(name) {
        Some(value) => match value.as_str() {
            Some(text) if !text.trim().is_empty() => Ok(text.to_string()),
            Some(_) => Err(Response::error(
                req.id.clone(),
                rpc::INVALID_PARAMS,
                format!("param '{}' must not be empty", name),
            )),
            None => Err(Response::error(
                req.id.clone(),
                rpc::INVALID_PARAMS,
                format!("param '{}' must be a string", name),
            )),
        },
        None => Err(Response::error(
            req.id.clone(),
            rpc::INVALID_PARAMS,
            format!("Missing param: {}", name),
        )),
    }
}

fn required_voice_param(req: &rpc::Request) -> Result<String, Response> {
    let voice = required_string_param(req, "voice")?;
    validate_voice(req, voice)
}

fn optional_voice_param(req: &rpc::Request) -> Result<Option<String>, Response> {
    if req.params.get("voice").is_none() {
        return Ok(None);
    }
    required_voice_param(req).map(Some)
}

fn validate_voice(req: &rpc::Request, voice: String) -> Result<String, Response> {
    if voice_tts::catalog::ALL_VOICES
        .iter()
        .any(|v| v.id == voice.as_str())
    {
        Ok(voice)
    } else {
        Err(Response::error(
            req.id.clone(),
            rpc::INVALID_PARAMS,
            format!("Unknown voice: {}", voice),
        ))
    }
}

fn required_speed_param(req: &rpc::Request) -> Result<f64, Response> {
    match req.params.get("speed") {
        Some(value) => match value.as_f64() {
            Some(speed) => validate_speed(req, speed),
            None => Err(Response::error(
                req.id.clone(),
                rpc::INVALID_PARAMS,
                "param 'speed' must be a number",
            )),
        },
        None => Err(Response::error(
            req.id.clone(),
            rpc::INVALID_PARAMS,
            "Missing param: speed",
        )),
    }
}

fn optional_speed_param(req: &rpc::Request) -> Result<Option<f64>, Response> {
    if req.params.get("speed").is_none() {
        return Ok(None);
    }
    required_speed_param(req).map(Some)
}

fn optional_audio_output_format_param(
    req: &rpc::Request,
) -> Result<Option<AudioOutputFormat>, Response> {
    let raw = req
        .params
        .get("format")
        .or_else(|| req.params.get("output_format"));
    let Some(value) = raw else {
        return Ok(None);
    };
    let Some(format) = value.as_str() else {
        return Err(Response::error(
            req.id.clone(),
            rpc::INVALID_PARAMS,
            "param 'format' must be a string",
        ));
    };
    AudioOutputFormat::from_name(format)
        .map(Some)
        .ok_or_else(|| {
            Response::error(
                req.id.clone(),
                rpc::INVALID_PARAMS,
                "param 'format' must be one of: wav, ogg, opus, ogg-opus",
            )
        })
}

fn validate_speed(req: &rpc::Request, speed: f64) -> Result<f64, Response> {
    if speed.is_finite() && speed > 0.0 && speed <= 5.0 {
        Ok(speed)
    } else {
        Err(Response::error(
            req.id.clone(),
            rpc::INVALID_PARAMS,
            "Speed must be between 0 (exclusive) and 5 (inclusive)",
        ))
    }
}

fn optional_duration_param(req: &rpc::Request, name: &str) -> Result<Option<u64>, Response> {
    match req.params.get(name) {
        Some(value) => match value.as_u64() {
            Some(duration) if duration > 0 => Ok(Some(duration)),
            Some(_) => Err(Response::error(
                req.id.clone(),
                rpc::INVALID_PARAMS,
                format!("param '{}' must be greater than 0", name),
            )),
            None => Err(Response::error(
                req.id.clone(),
                rpc::INVALID_PARAMS,
                format!("param '{}' must be an unsigned integer", name),
            )),
        },
        None => Ok(None),
    }
}

fn optional_u32_param(
    req: &rpc::Request,
    name: &str,
    min: u32,
    max: u32,
) -> Result<Option<u32>, Response> {
    match req.params.get(name) {
        Some(value) => match value.as_u64() {
            Some(raw) if raw >= min as u64 && raw <= max as u64 => Ok(Some(raw as u32)),
            Some(_) => Err(Response::error(
                req.id.clone(),
                rpc::INVALID_PARAMS,
                format!("param '{}' must be between {} and {}", name, min, max),
            )),
            None => Err(Response::error(
                req.id.clone(),
                rpc::INVALID_PARAMS,
                format!("param '{}' must be an unsigned integer", name),
            )),
        },
        None => Ok(None),
    }
}

pub fn cleanup() {
    let path = socket_path();
    if path.exists() {
        std::fs::remove_file(&path).ok();
    }
}
