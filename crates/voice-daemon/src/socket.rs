//! Unix socket server for the voice daemon.
//!
//! Uses the voice-protocol frame codec (length-prefixed typed frames)
//! instead of newline-delimited JSON.

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

pub async fn serve(queue: Arc<RequestQueue>) {
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
                let client_id = Uuid::new_v4().to_string()[..8].to_string();
                eprintln!("voiced: client connected ({})", client_id);
                tokio::spawn(handle_client(stream, queue, client_id));
            }
            Err(e) => eprintln!("voiced: accept error: {}", e),
        }
    }
}

async fn handle_client(
    stream: tokio::net::UnixStream,
    queue: Arc<RequestQueue>,
    client_id: String,
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
                    Ok(req) => dispatch(req, &queue, &client_id).await,
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

async fn dispatch(req: rpc::Request, queue: &Arc<RequestQueue>, client_id: &str) -> Response {
    match req.method.as_str() {
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

            let queue_id = queue
                .enqueue_speak(client_id.to_string(), text.to_string(), voice, speed)
                .await;

            Response::success(
                req.id,
                serde_json::json!({ "queue_id": queue_id, "status": "queued" }),
            )
        }

        "listen" => {
            let max_duration_ms = req.params.get("max_duration_ms").and_then(|v| v.as_u64());
            let queue_id = queue
                .enqueue_listen(client_id.to_string(), max_duration_ms)
                .await;

            Response::success(
                req.id,
                serde_json::json!({ "queue_id": queue_id, "status": "queued" }),
            )
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

            let queue_id = queue
                .enqueue_converse(client_id.to_string(), text.to_string(), voice)
                .await;

            Response::success(
                req.id,
                serde_json::json!({ "queue_id": queue_id, "status": "queued" }),
            )
        }

        "cancel" => {
            let count = queue.cancel_client(client_id).await;
            Response::success(req.id, serde_json::json!({ "cancelled_count": count }))
        }

        "status" => {
            let state = queue.snapshot().await;
            Response::success(req.id, serde_json::to_value(&state).unwrap())
        }

        _ => Response::error(
            req.id,
            rpc::METHOD_NOT_FOUND,
            format!("Method not found: {}", req.method),
        ),
    }
}

pub fn cleanup() {
    let path = socket_path();
    if path.exists() {
        std::fs::remove_file(&path).ok();
    }
}
