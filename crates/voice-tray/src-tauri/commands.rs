//! Tauri commands exposed to frontend.

use crate::daemon_client::DaemonClient;
use crate::state::AppState;
use crate::types::VoiceState;
use tauri::State;

/// Get current queue state.
#[tauri::command]
pub fn get_queue_state(state: State<AppState>) -> Result<VoiceState, String> {
    Ok(state.get_voice_state())
}

/// Play question audio for a queue item.
#[tauri::command]
pub fn play_question(state: State<AppState>, queue_id: String) -> Result<u64, String> {
    let mut client = state
        .daemon_client
        .lock()
        .map_err(|e| format!("Failed to lock daemon client: {}", e))?;
    client.play_question(&queue_id)
}

/// Play answer audio for a queue item.
#[tauri::command]
pub fn play_answer(state: State<AppState>, queue_id: String) -> Result<u64, String> {
    let mut client = state
        .daemon_client
        .lock()
        .map_err(|e| format!("Failed to lock daemon client: {}", e))?;
    client.play_answer(&queue_id)
}

/// Cancel a queue item.
#[tauri::command]
pub fn cancel_item(state: State<AppState>, queue_id: String) -> Result<bool, String> {
    let mut client = state
        .daemon_client
        .lock()
        .map_err(|e| format!("Failed to lock daemon client: {}", e))?;
    client.cancel_item(&queue_id)
}

/// Check if daemon is running.
#[tauri::command]
pub fn is_daemon_running() -> bool {
    DaemonClient::is_daemon_running()
}

/// Toggle window visibility (for debugging).
#[tauri::command]
pub fn toggle_window(window: tauri::Window) -> Result<(), String> {
    let is_visible = window.is_visible().unwrap_or(false);
    if is_visible {
        window.hide().map_err(|e| e.to_string())?;
    } else {
        window.show().map_err(|e| e.to_string())?;
        window.set_focus().map_err(|e| e.to_string())?;
    }
    Ok(())
}
