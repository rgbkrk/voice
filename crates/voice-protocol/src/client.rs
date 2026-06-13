//! Synchronous client for talking to the voice daemon.
//!
//! The MCP server is synchronous (no tokio runtime), so this client
//! uses std::os::unix::net::UnixStream directly.

use crate::frames::{read_frame_sync, write_frame_sync, Frame, FrameType};
use crate::rpc::{self, Response};
use serde_json::Value;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

const SOCKET_ENV: &str = "VOICE_DAEMON_SOCKET";

/// A synchronous client connection to the voice daemon.
pub struct DaemonClient {
    stream: UnixStream,
}

/// Get the daemon socket path.
pub fn daemon_socket_path() -> PathBuf {
    if let Ok(path) = std::env::var(SOCKET_ENV) {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }

    let dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".voice");
    dir.join("daemon.sock")
}

/// Environment variable that overrides the daemon socket path.
pub fn daemon_socket_env_var() -> &'static str {
    SOCKET_ENV
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

        write_frame_sync(&mut self.stream, &Frame::request(&json))
            .map_err(|e| format!("write frame: {}", e))?;

        let frame = read_frame_sync(&mut self.stream)
            .map_err(|e| format!("read frame: {}", e))?
            .ok_or_else(|| "connection closed before response".to_string())?;

        if frame.frame_type != FrameType::Response {
            return Err(format!(
                "unexpected frame type {:?}; expected response",
                frame.frame_type
            ));
        }

        frame
            .json::<Response>()
            .map_err(|e| format!("parse response: {}", e))
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

    /// Convenience: set daemon default voice.
    pub fn set_voice(&mut self, voice: &str) -> Result<Response, String> {
        self.call("set_voice", serde_json::json!({ "voice": voice }))
    }

    /// Convenience: set daemon default speed.
    pub fn set_speed(&mut self, speed: f64) -> Result<Response, String> {
        self.call("set_speed", serde_json::json!({ "speed": speed }))
    }

    /// Convenience: list known voices and the current daemon default.
    pub fn list_voices(&mut self) -> Result<Response, String> {
        self.call("list_voices", serde_json::json!({}))
    }

    /// Convenience: cancel a specific queue item.
    pub fn cancel_item(&mut self, queue_id: &str) -> Result<Response, String> {
        self.call("cancel_item", serde_json::json!({ "queue_id": queue_id }))
    }

    /// Convenience: replay stored question or answer audio for a queue item.
    pub fn replay_audio(&mut self, queue_id: &str, part: &str) -> Result<Response, String> {
        self.call(
            "replay_audio",
            serde_json::json!({ "queue_id": queue_id, "part": part }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{write_frame_sync, Frame};
    use crate::rpc::Response;
    use std::io::Read;
    use std::os::unix::net::UnixListener;
    use std::sync::Mutex;
    use std::thread;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn socket_path_uses_env_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        let old = std::env::var(SOCKET_ENV).ok();
        std::env::set_var(SOCKET_ENV, "/tmp/voice-test.sock");

        assert_eq!(daemon_socket_path(), PathBuf::from("/tmp/voice-test.sock"));

        match old {
            Some(value) => std::env::set_var(SOCKET_ENV, value),
            None => std::env::remove_var(SOCKET_ENV),
        }
    }

    #[test]
    fn call_rejects_non_response_frame() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("voice-protocol-test-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 64];
            let _ = stream.read(&mut buf).unwrap();
            write_frame_sync(&mut stream, &Frame::event(b"{}")).unwrap();
        });

        let old = std::env::var(SOCKET_ENV).ok();
        std::env::set_var(SOCKET_ENV, &path);

        let mut client = DaemonClient::connect().unwrap();
        let err = client.call("status", serde_json::json!({})).unwrap_err();

        assert!(err.contains("unexpected frame type"));

        match old {
            Some(value) => std::env::set_var(SOCKET_ENV, value),
            None => std::env::remove_var(SOCKET_ENV),
        }
        let _ = std::fs::remove_file(path);
        server.join().unwrap();
    }

    #[test]
    fn call_parses_response_frame() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "voice-protocol-test-ok-{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 128];
            let _ = stream.read(&mut buf).unwrap();
            let response =
                Response::success(Some(1.into()), serde_json::json!({ "status": "idle" }));
            let payload = serde_json::to_vec(&response).unwrap();
            write_frame_sync(&mut stream, &Frame::response(&payload)).unwrap();
        });

        let old = std::env::var(SOCKET_ENV).ok();
        std::env::set_var(SOCKET_ENV, &path);

        let mut client = DaemonClient::connect().unwrap();
        let response = client.call("status", serde_json::json!({})).unwrap();
        assert_eq!(response.result.unwrap()["status"], "idle");

        match old {
            Some(value) => std::env::set_var(SOCKET_ENV, value),
            None => std::env::remove_var(SOCKET_ENV),
        }
        let _ = std::fs::remove_file(path);
        server.join().unwrap();
    }
}
