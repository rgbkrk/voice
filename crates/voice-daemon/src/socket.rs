//! Unix socket server for the voice daemon.
//!
//! Listens on ~/.voice/daemon.sock. Speaks JSON-RPC 2.0 over
//! newline-delimited streams, consistent with the MCP server protocol.

use crate::queue::RequestQueue;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// JSON-RPC types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct JsonRpcRequest {
    jsonrpc: Option<String>,
    method: String,
    #[serde(default)]
    params: Value,
    id: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

impl JsonRpcResponse {
    fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            result: Some(result),
            error: None,
            id,
        }
    }

    fn error(id: Option<Value>, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0",
            result: None,
            error: Some(JsonRpcError { code, message }),
            id,
        }
    }
}

// ---------------------------------------------------------------------------
// Socket path
// ---------------------------------------------------------------------------

pub fn socket_path() -> PathBuf {
    let dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".voice");
    std::fs::create_dir_all(&dir).ok();
    dir.join("daemon.sock")
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

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
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<JsonRpcRequest>(&line) {
            Ok(req) => dispatch(req, &queue, &client_id).await,
            Err(e) => JsonRpcResponse::error(None, -32700, format!("Parse error: {}", e)),
        };

        let json = serde_json::to_string(&response).unwrap();
        if writer
            .write_all(format!("{}\n", json).as_bytes())
            .await
            .is_err()
        {
            break;
        }
    }

    eprintln!("voiced: client disconnected ({})", client_id);
}

// ---------------------------------------------------------------------------
// Method dispatch
// ---------------------------------------------------------------------------

async fn dispatch(
    req: JsonRpcRequest,
    queue: &Arc<RequestQueue>,
    client_id: &str,
) -> JsonRpcResponse {
    match req.method.as_str() {
        "speak" => {
            let text = req.params.get("text").and_then(|v| v.as_str());
            let Some(text) = text else {
                return JsonRpcResponse::error(
                    req.id,
                    -32602,
                    "Missing required param: text".to_string(),
                );
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

            JsonRpcResponse::success(
                req.id,
                serde_json::json!({ "queue_id": queue_id, "status": "queued" }),
            )
        }

        "listen" => {
            let max_duration_ms = req.params.get("max_duration_ms").and_then(|v| v.as_u64());

            let queue_id = queue
                .enqueue_listen(client_id.to_string(), max_duration_ms)
                .await;

            JsonRpcResponse::success(
                req.id,
                serde_json::json!({ "queue_id": queue_id, "status": "queued" }),
            )
        }

        "converse" => {
            let text = req.params.get("text").and_then(|v| v.as_str());
            let Some(text) = text else {
                return JsonRpcResponse::error(
                    req.id,
                    -32602,
                    "Missing required param: text".to_string(),
                );
            };
            let voice = req
                .params
                .get("voice")
                .and_then(|v| v.as_str())
                .map(String::from);

            let queue_id = queue
                .enqueue_converse(client_id.to_string(), text.to_string(), voice)
                .await;

            JsonRpcResponse::success(
                req.id,
                serde_json::json!({ "queue_id": queue_id, "status": "queued" }),
            )
        }

        "cancel" => {
            let count = queue.cancel_client(client_id).await;
            JsonRpcResponse::success(req.id, serde_json::json!({ "cancelled_count": count }))
        }

        "status" => {
            let state = queue.snapshot().await;
            JsonRpcResponse::success(req.id, serde_json::to_value(&state).unwrap())
        }

        _ => JsonRpcResponse::error(req.id, -32601, format!("Method not found: {}", req.method)),
    }
}

pub fn cleanup() {
    let path = socket_path();
    if path.exists() {
        std::fs::remove_file(&path).ok();
    }
}
