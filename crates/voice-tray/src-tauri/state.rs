//! Shared application state.

use crate::daemon_client::DaemonClient;
use crate::types::VoiceState;
use std::sync::{Arc, Mutex};

/// Shared app state accessible from Tauri commands.
#[derive(Clone)]
pub struct AppState {
    /// Current queue state (read from Automerge doc).
    pub voice_state: Arc<Mutex<VoiceState>>,
    /// Daemon RPC client.
    pub daemon_client: Arc<Mutex<DaemonClient>>,
}

impl AppState {
    /// Create new app state with default values.
    pub fn new() -> Self {
        Self {
            voice_state: Arc::new(Mutex::new(VoiceState::default())),
            daemon_client: Arc::new(Mutex::new(DaemonClient::new())),
        }
    }

    /// Update the voice state (called by file watcher).
    pub fn update_voice_state(&self, new_state: VoiceState) {
        if let Ok(mut state) = self.voice_state.lock() {
            *state = new_state;
        }
    }

    /// Get current voice state (clone for thread safety).
    pub fn get_voice_state(&self) -> VoiceState {
        self.voice_state
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default()
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
