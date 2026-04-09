//! JSON-RPC 2.0 request/response types for the voice daemon.
//!
//! These are the canonical definitions used by both the daemon and all clients
//! (MCP server, CLI, Tauri UI).

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// JSON-RPC envelope
// ---------------------------------------------------------------------------

/// A JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
}

impl Request {
    pub fn new(method: &str, params: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<Value>) -> Self {
        self.id = Some(id.into());
        self
    }
}

/// A JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
    pub id: Option<Value>,
}

impl Response {
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: Some(result),
            error: None,
            id,
        }
    }

    pub fn error(id: Option<Value>, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
            }),
            id,
        }
    }

    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
}

// Standard JSON-RPC error codes
pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;

// ---------------------------------------------------------------------------
// Daemon event (server-initiated notification)
// ---------------------------------------------------------------------------

/// An event pushed from the daemon to connected clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Event type name.
    pub event: String,
    /// Event payload.
    pub data: Value,
}

impl Event {
    pub fn new(event: &str, data: Value) -> Self {
        Self {
            event: event.to_string(),
            data,
        }
    }
}

// ---------------------------------------------------------------------------
// Voice-specific types (shared between daemon and clients)
// ---------------------------------------------------------------------------

/// Status of a queued item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ItemStatus {
    Queued,
    Processing,
    Completed,
    Failed,
}

/// A single item in the daemon's queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueItem {
    pub id: String,
    pub client_id: String,
    pub method: String,
    pub status: ItemStatus,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
}

/// Snapshot of daemon state — returned by the `status` method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonState {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current: Option<QueueItem>,
    pub pending: Vec<QueueItem>,
    pub recent: Vec<QueueItem>,
}
