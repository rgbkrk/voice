//! Unix socket server for the voice daemon.
//!
//! Listens on ~/.voice/daemon.sock. Each connection is a client session.
//! Clients send newline-delimited JSON requests and get JSON responses.

use crate::queue::{RequestQueue, VoiceRequest};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct ClientRequest {
    id: Option<String>,
    #[serde(flatten)]
    request: VoiceRequest,
}

#[derive(Debug, Serialize)]
struct QueuedResponse {
    id: Option<String>,
    queue_id: String,
    status: String,
}

pub fn socket_path() -> PathBuf {
    let dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".voice");
    std::fs::create_dir_all(&dir).ok();
    dir.join("daemon.sock")
}

pub async fn serve(queue: Arc<RequestQueue>) {
    let path = socket_path();

    // Remove stale socket
    if path.exists() {
        std::fs::remove_file(&path).ok();
    }

    let listener = match UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to bind {}: {}", path.display(), e);
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
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<ClientRequest>(&line) {
            Ok(req) => handle_request(req, &queue, &client_id).await,
            Err(e) => serde_json::to_string(&serde_json::json!({
                "error": format!("parse error: {}", e)
            }))
            .unwrap(),
        };

        if writer
            .write_all(format!("{}\n", response).as_bytes())
            .await
            .is_err()
        {
            break;
        }
    }

    eprintln!("voiced: client disconnected ({})", client_id);
}

async fn handle_request(req: ClientRequest, queue: &Arc<RequestQueue>, client_id: &str) -> String {
    let request_id = req.id.clone();

    match req.request {
        VoiceRequest::Cancel => {
            let count = queue.cancel_client(client_id).await;
            serde_json::to_string(&serde_json::json!({
                "id": request_id,
                "status": "cancelled",
                "cancelled_count": count,
            }))
            .unwrap()
        }
        VoiceRequest::Status => {
            let state = queue.snapshot().await;
            serde_json::to_string(&serde_json::json!({
                "id": request_id,
                "state": state,
            }))
            .unwrap()
        }
        request => {
            let queue_id = queue.enqueue(client_id.to_string(), request).await;
            serde_json::to_string(&QueuedResponse {
                id: request_id,
                queue_id,
                status: "queued".to_string(),
            })
            .unwrap()
        }
    }
}

pub fn cleanup() {
    let path = socket_path();
    if path.exists() {
        std::fs::remove_file(&path).ok();
    }
}
