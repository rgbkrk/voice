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

            // Set up tray menu with show/hide option
            if let Some(tray) = app.tray_by_id("main-tray") {
                info!("Tray icon found, setting up menu");

                use tauri::menu::{Menu, MenuItem};
                use tauri::tray::MouseButton;

                // Create tray menu
                let toggle_item = MenuItem::with_id(app, "toggle", "Show/Hide", true, None::<&str>)
                    .expect("Failed to create menu item");
                let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)
                    .expect("Failed to create menu item");

                let menu = Menu::with_items(app, &[&toggle_item, &quit_item])
                    .expect("Failed to create menu");

                if let Err(e) = tray.set_menu(Some(menu)) {
                    error!("Failed to set tray menu: {}", e);
                }

                // Set up menu event handler
                let window_handle = app.handle().clone();
                tray.on_menu_event(move |_app_handle, event| {
                    info!("Tray menu event: {:?}", event.id());
                    match event.id().as_ref() {
                        "toggle" => {
                            if let Some(window) = window_handle.get_webview_window("main") {
                                let is_visible = window.is_visible().unwrap_or(false);
                                info!("Toggle clicked, window visible: {}", is_visible);

                                if is_visible {
                                    info!("Hiding window");
                                    let _ = window.hide();
                                } else {
                                    info!("Showing window");
                                    // Position window near tray icon (top-right of screen)
                                    #[cfg(target_os = "macos")]
                                    {
                                        use tauri::LogicalPosition;
                                        if let Ok(Some(monitor)) = window.current_monitor() {
                                            let size = monitor.size();
                                            let x = size.width as f64 - 400.0;
                                            let y = 25.0;
                                            info!("Positioning window at ({}, {})", x, y);
                                            let _ = window.set_position(LogicalPosition::new(x, y));
                                        }
                                    }

                                    if let Err(e) = window.show() {
                                        error!("Error showing window: {}", e);
                                    }
                                    if let Err(e) = window.set_focus() {
                                        error!("Error focusing window: {}", e);
                                    }
                                }
                            }
                        }
                        "quit" => {
                            info!("Quit requested");
                            std::process::exit(0);
                        }
                        _ => {}
                    }
                });

                // Also handle direct tray click (left click to show/hide)
                let window_handle2 = app.handle().clone();
                tray.on_tray_icon_event(move |_tray, event| {
                    info!("Tray icon event: {:?}", event);
                    if let tauri::tray::TrayIconEvent::Click { button, .. } = event {
                        if button == MouseButton::Left {
                            if let Some(window) = window_handle2.get_webview_window("main") {
                                let is_visible = window.is_visible().unwrap_or(false);
                                if is_visible {
                                    let _ = window.hide();
                                } else {
                                    #[cfg(target_os = "macos")]
                                    {
                                        use tauri::LogicalPosition;
                                        if let Ok(Some(monitor)) = window.current_monitor() {
                                            let size = monitor.size();
                                            let x = size.width as f64 - 400.0;
                                            let y = 25.0;
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
            } else {
                error!("Tray icon 'main-tray' not found!");
            }

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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
