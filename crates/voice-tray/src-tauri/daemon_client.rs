//! Unix socket client for voice daemon RPC.

use serde_json::json;
use voice_protocol::client::DaemonClient as ProtocolClient;

/// Wrapper around voice-protocol's DaemonClient for Tauri commands.
pub struct DaemonClient {
    client: Option<ProtocolClient>,
}

impl DaemonClient {
    /// Create new client (not yet connected).
    pub fn new() -> Self {
        Self { client: None }
    }

    /// Ensure connected to daemon (reconnect if needed).
    fn ensure_connected(&mut self) -> Result<(), String> {
        if self.client.is_none() {
            self.client = ProtocolClient::connect();
            if self.client.is_none() {
                return Err("Daemon not running".to_string());
            }
        }
        Ok(())
    }

    /// Call RPC method with automatic reconnect on connection failure.
    fn call_with_reconnect(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<voice_protocol::rpc::Response, String> {
        self.ensure_connected()?;

        let client = self.client.as_mut().unwrap();
        match client.call(method, params.clone()) {
            Ok(resp) => Ok(resp),
            Err(_e) => {
                // Connection likely broken -- drop and retry once
                self.client = None;
                self.ensure_connected()?;
                self.client
                    .as_mut()
                    .unwrap()
                    .call(method, params)
                    .map_err(|e| format!("RPC error: {}", e))
            }
        }
    }

    /// Play question audio for a queue item.
    pub fn play_question(&mut self, queue_id: &str) -> Result<u64, String> {
        let params = json!({
            "queue_id": queue_id,
            "part": "question"
        });

        let resp = self.call_with_reconnect("replay_audio", params)?;

        if let Some(err) = resp.error {
            return Err(err.message);
        }

        let duration_ms = resp
            .result
            .as_ref()
            .and_then(|v| v.get("duration_ms"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "Missing duration_ms in response".to_string())?;

        Ok(duration_ms)
    }

    /// Play answer audio for a queue item.
    pub fn play_answer(&mut self, queue_id: &str) -> Result<u64, String> {
        let params = json!({
            "queue_id": queue_id,
            "part": "answer"
        });

        let resp = self.call_with_reconnect("replay_audio", params)?;

        if let Some(err) = resp.error {
            return Err(err.message);
        }

        let duration_ms = resp
            .result
            .as_ref()
            .and_then(|v| v.get("duration_ms"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "Missing duration_ms in response".to_string())?;

        Ok(duration_ms)
    }

    /// Cancel a queue item.
    pub fn cancel_item(&mut self, queue_id: &str) -> Result<bool, String> {
        let params = json!({
            "queue_id": queue_id
        });

        let resp = self.call_with_reconnect("cancel_item", params)?;

        if let Some(err) = resp.error {
            return Err(err.message);
        }

        let cancelled = resp
            .result
            .as_ref()
            .and_then(|v| v.get("cancelled"))
            .and_then(|v| v.as_bool())
            .ok_or_else(|| "Missing cancelled in response".to_string())?;

        Ok(cancelled)
    }

    /// Check if daemon is running.
    pub fn is_daemon_running() -> bool {
        voice_protocol::client::daemon_is_running()
    }
}
