//! Voice Queue - macOS system tray app for voice daemon queue management.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod automerge_sync;
mod commands;
mod daemon_client;
mod state;
mod types;

use state::AppState;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};
use types::VoiceState;

/// Update the tray icon badge with the number of pending items
fn update_tray_badge(app: &AppHandle, state: &VoiceState) {
    let pending_count = state.pending.len();

    if let Some(tray) = app.tray_by_id("main") {
        let badge_text = if pending_count > 0 {
            pending_count.to_string()
        } else {
            String::new()
        };

        #[cfg(target_os = "macos")]
        {
            if let Err(e) = tray.set_icon_as_template(true) {
                eprintln!("Failed to set tray icon as template: {}", e);
            }
        }

        if let Err(e) = tray.set_title(Some(&badge_text)) {
            eprintln!("Failed to set tray badge: {}", e);
        }
    }
}

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

            // Get tray handle for badge updates
            let tray_app_handle = app.handle().clone();

            // Set up tray click handler to show/hide window
            if let Some(tray) = app.tray_by_id("main-tray") {
                let window_handle = app.handle().clone();
                tray.on_tray_icon_event(move |_tray, event| {
                    if let tauri::tray::TrayIconEvent::Click { button, .. } = event {
                        if button == tauri::tray::MouseButton::Left {
                            if let Some(window) = window_handle.get_webview_window("main") {
                                let is_visible = window.is_visible().unwrap_or(false);

                                if is_visible {
                                    let _ = window.hide();
                                } else {
                                    // Position window near tray icon (top-right of screen)
                                    #[cfg(target_os = "macos")]
                                    {
                                        use tauri::LogicalPosition;
                                        if let Ok(Some(monitor)) = window.current_monitor() {
                                            let size = monitor.size();
                                            // Position in top-right, below menu bar
                                            let x = size.width as f64 - 400.0;  // 380 width + 20 padding
                                            let y = 25.0;  // Below menu bar
                                            let _ = window.set_position(LogicalPosition::new(x, y));
                                        }
                                    }

                                    let _ = window.show();
                                    let _ = window.set_focus();
                                }
                            }
                        }
                    }
                });
            }

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

                        // Update tray badge
                        update_tray_badge(&tray_app_handle, &initial_state);

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

                        // Update tray badge
                        update_tray_badge(&tray_app_handle, &new_state);

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
