//! Voice Queue - macOS system tray app for voice daemon queue management.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod automerge_sync;
mod commands;
mod daemon_client;
mod state;
mod types;

use state::AppState;
use std::time::Duration;
use tauri::Emitter;

fn main() {
    // Initialize shared app state
    let app_state = AppState::new();

    // Clone state for file watcher thread
    let watcher_state = app_state.clone();

    tauri::Builder::default()
        .manage(app_state)
        .setup(move |app| {
            eprintln!("Voice Queue starting...");

            // Clone app handle for file watcher thread
            let app_handle = app.handle().clone();

            // Spawn file watcher in background thread
            std::thread::spawn(move || {
                eprintln!("Starting file watcher thread...");

                // Create file watcher
                let watcher = match automerge_sync::FileWatcher::new() {
                    Ok(w) => w,
                    Err(e) => {
                        eprintln!("Failed to create file watcher: {}", e);
                        return;
                    }
                };

                // Load initial state
                match watcher.load_current() {
                    Ok(initial_state) => {
                        watcher_state.update_voice_state(initial_state.clone());
                        // Emit initial state to frontend
                        if let Err(e) = app_handle.emit("queue-updated", initial_state) {
                            eprintln!("Failed to emit initial state: {}", e);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to load initial state: {}", e);
                    }
                }

                // Watch for changes in a loop
                loop {
                    if let Some(new_state) = watcher.wait_for_change(Duration::from_secs(1)) {
                        eprintln!("State file changed, updating...");
                        watcher_state.update_voice_state(new_state.clone());

                        // Emit event to frontend
                        if let Err(e) = app_handle.emit("queue-updated", new_state) {
                            eprintln!("Failed to emit queue-updated event: {}", e);
                        }
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_queue_state,
            commands::play_question,
            commands::play_answer,
            commands::cancel_item,
            commands::is_daemon_running,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
