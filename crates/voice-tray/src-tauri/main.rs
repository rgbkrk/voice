//! Voice Queue - macOS system tray app for voice daemon queue management.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod automerge_sync;
mod commands;
mod daemon_client;
mod state;
mod types;

use log::{error, info, warn};
use state::AppState;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};
use types::VoiceState;

/// Update the tray icon badge with the number of pending items
fn update_tray_badge(app: &AppHandle, state: &VoiceState) {
    let pending_count = state.pending.len();

    if let Some(tray) = app.tray_by_id("main-tray") {
        let badge_text = if pending_count > 0 {
            pending_count.to_string()
        } else {
            String::new()
        };

        #[cfg(target_os = "macos")]
        {
            if let Err(e) = tray.set_icon_as_template(true) {
                warn!("Failed to set tray icon as template: {}", e);
            }
        }

        if let Err(e) = tray.set_title(Some(&badge_text)) {
            warn!("Failed to set tray badge: {}", e);
        }
    }
}

fn main() {
    // Initialize shared app state
    let app_state = AppState::new();

    // Clone state for file watcher thread
    let watcher_state = app_state.clone();

    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .target(tauri_plugin_log::Target::new(
                    tauri_plugin_log::TargetKind::LogDir {
                        file_name: Some("voice-tray".to_string()),
                    },
                ))
                .level(log::LevelFilter::Info)
                .build(),
        )
        .manage(app_state)
        .setup(move |app| {
            info!("Voice Queue starting...");

            // Clone app handle for file watcher thread
            let app_handle = app.handle().clone();

            // Get tray handle for badge updates
            let tray_app_handle = app.handle().clone();

            // Set up tray (no menu, just click to show/hide)
            if let Some(tray) = app.tray_by_id("main-tray") {
                info!("Tray icon found, setting up click handler");

                use tauri::tray::MouseButton;

                // Handle direct tray click to show/hide window
                let window_handle = app.handle().clone();
                tray.on_tray_icon_event(move |_tray, event| {
                    info!("Tray icon event: {:?}", event);
                    if let tauri::tray::TrayIconEvent::Click { button, rect, .. } = event {
                        if button == MouseButton::Left {
                            if let Some(window) = window_handle.get_webview_window("main") {
                                let is_visible = window.is_visible().unwrap_or(false);
                                info!("Left click on tray, window visible: {}", is_visible);

                                if is_visible {
                                    info!("Hiding window");
                                    let _ = window.hide();
                                } else {
                                    info!("Showing window");
                                    #[cfg(target_os = "macos")]
                                    {
                                        use tauri::{PhysicalPosition, Position, Size};

                                        // Extract physical position and size from the rect
                                        if let (Position::Physical(pos), Size::Physical(size)) = (&rect.position, &rect.size) {
                                            let tray_x = pos.x;
                                            let tray_y = pos.y;
                                            let tray_width = size.width;
                                            let tray_height = size.height;

                                            info!("Tray icon rect: x={}, y={}, width={}, height={}", tray_x, tray_y, tray_width, tray_height);

                                            // Position window below the tray icon
                                            // Window width is 400px, so center it under the tray icon
                                            let window_x = (tray_x + tray_width as i32 / 2 - 200).max(0);
                                            let window_y = tray_y + tray_height as i32 + 5; // 5px gap

                                            info!("Positioning window at physical: x={}, y={}", window_x, window_y);

                                            if let Err(e) = window.set_position(PhysicalPosition::new(window_x, window_y)) {
                                                error!("Failed to set window position: {}", e);
                                            }
                                        } else {
                                            warn!("Tray rect not in physical coordinates, using fallback");
                                            // Fallback positioning
                                            let _ = window.set_position(PhysicalPosition::new(100, 40));
                                        }

                                        // Show and focus the window
                                        if let Err(e) = window.show() {
                                            error!("Error showing window: {}", e);
                                        }
                                        if let Err(e) = window.set_focus() {
                                            error!("Error focusing window: {}", e);
                                        }
                                    }

                                    #[cfg(not(target_os = "macos"))]
                                    {
                                        if let Err(e) = window.show() {
                                            error!("Error showing window: {}", e);
                                        }
                                        if let Err(e) = window.set_focus() {
                                            error!("Error focusing window: {}", e);
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            } else {
                error!("Tray icon 'main-tray' not found!");
            }

            // Window visibility is controlled by tray icon toggle
            // No auto-hide on blur - user clicks tray again to close

            // Spawn file watcher task on Tauri's async runtime
            tauri::async_runtime::spawn(async move {
                info!("Starting file watcher task...");

                // Create file watcher
                let mut watcher = match automerge_sync::FileWatcher::new() {
                    Ok(w) => w,
                    Err(e) => {
                        error!("Failed to create file watcher: {}", e);
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
                            error!("Failed to emit initial state: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Failed to load initial state: {}", e);
                    }
                }

                // Watch for changes in a loop
                loop {
                    if let Some(new_state) = watcher.wait_for_change(Duration::from_secs(1)).await {
                        info!("State file changed, updating...");
                        watcher_state.update_voice_state(new_state.clone());

                        // Update tray badge
                        update_tray_badge(&tray_app_handle, &new_state);

                        // Emit event to frontend
                        if let Err(e) = app_handle.emit("queue-updated", new_state) {
                            error!("Failed to emit queue-updated event: {}", e);
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
            commands::toggle_window,
            commands::quit_app,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
