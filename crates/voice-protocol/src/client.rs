//! Synchronous client for talking to the voice daemon.
//!
//! The MCP server is synchronous (no tokio runtime), so this client
//! uses std::os::unix::net::UnixStream directly.

use crate::frames::FrameType;
use crate::rpc::{self, Response};
use serde_json::Value;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

/// A synchronous client connection to the voice daemon.
pub struct DaemonClient {
    stream: UnixStream,
}

/// Get the daemon socket path.
pub fn daemon_socket_path() -> PathBuf {
    let dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".voice");
    dir.join("daemon.sock")
}

/// Check if the daemon is running (socket exists and accepts connections).
pub fn daemon_is_running() -> bool {
    let path = daemon_socket_path();
    if !path.exists() {
        return false;
    }
    UnixStream::connect(&path).is_ok()
}

impl DaemonClient {
    /// Connect to the daemon. Returns None if daemon isn't running.
    pub fn connect() -> Option<Self> {
        let path = daemon_socket_path();
        let stream = UnixStream::connect(&path).ok()?;
        stream
            .set_read_timeout(Some(Duration::from_secs(120)))
            .ok()?;
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .ok()?;
        Some(Self { stream })
    }

    /// Send a JSON-RPC request and get the response.
    pub fn call(&mut self, method: &str, params: Value) -> Result<Response, String> {
        let req = rpc::Request::new(method, params).with_id(1);
        let json = serde_json::to_vec(&req).map_err(|e| format!("serialize: {}", e))?;

        // Write frame: [4-byte length][1-byte type=Request][payload]
        let total_len = (json.len() + 1) as u32;
        self.stream
            .write_all(&total_len.to_be_bytes())
            .map_err(|e| format!("write len: {}", e))?;
        self.stream
            .write_all(&[FrameType::Request as u8])
            .map_err(|e| format!("write type: {}", e))?;
        self.stream
            .write_all(&json)
            .map_err(|e| format!("write payload: {}", e))?;
        self.stream.flush().map_err(|e| format!("flush: {}", e))?;

        // Read response frame
        let mut len_buf = [0u8; 4];
        self.stream
            .read_exact(&mut len_buf)
            .map_err(|e| format!("read len: {}", e))?;
        let total_len = u32::from_be_bytes(len_buf) as usize;

        let mut data = vec![0u8; total_len];
        self.stream
            .read_exact(&mut data)
            .map_err(|e| format!("read payload: {}", e))?;

        if data.is_empty() {
            return Err("empty response".to_string());
        }

        let _frame_type = data[0]; // Should be Response (0x02)
        let payload = &data[1..];

        serde_json::from_slice::<Response>(payload).map_err(|e| format!("parse response: {}", e))
    }

    /// Convenience: send a speak request. Returns immediately with queue_id (fire-and-forget).
    pub fn speak(
        &mut self,
        text: &str,
        voice: Option<&str>,
        speed: Option<f64>,
    ) -> Result<Response, String> {
        let mut params = serde_json::json!({"text": text, "wait": false});
        if let Some(v) = voice {
            params["voice"] = Value::String(v.to_string());
        }
        if let Some(s) = speed {
            params["speed"] = serde_json::json!(s);
        }
        self.call("speak", params)
    }

    /// Convenience: send a listen request. Blocks until transcription completes.
    pub fn listen(&mut self, max_duration_ms: Option<u64>) -> Result<Response, String> {
        let mut params = serde_json::json!({"wait": true});
        if let Some(ms) = max_duration_ms {
            params["max_duration_ms"] = serde_json::json!(ms);
        }
        self.call("listen", params)
    }

    /// Convenience: send a converse request. Blocks until speak+listen completes.
    pub fn converse(&mut self, text: &str, voice: Option<&str>) -> Result<Response, String> {
        let mut params = serde_json::json!({"text": text, "wait": true});
        if let Some(v) = voice {
            params["voice"] = Value::String(v.to_string());
        }
        self.call("converse", params)
    }

    /// Convenience: cancel all pending requests from this client.
    pub fn cancel(&mut self) -> Result<Response, String> {
        self.call("cancel", serde_json::json!({}))
    }

    /// Convenience: get daemon status.
    pub fn status(&mut self) -> Result<Response, String> {
        self.call("status", serde_json::json!({}))
    }
}
