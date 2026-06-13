//! voiced — the voice daemon.
//!
//! Listens on ~/.voice/daemon.sock for TTS/STT requests and processes
//! them sequentially so multiple MCP clients never overlap audio.
//!
//! Usage:
//!   voiced              # start the daemon
//!   voiced --status     # print daemon state and exit
//!   voiced --tts-only   # start without eagerly loading STT/microphone path

mod audio_recorder;
mod automerge_state;
mod cleanup;
mod config;
mod queue;
mod socket;
mod worker;

use automerge_state::AutomergeState;
use config::DaemonConfig;
use queue::RequestQueue;
use std::sync::Arc;
use tokio::sync::Mutex;
use voice_protocol::frames::{read_frame, write_frame, Frame, FrameType};
use voice_protocol::rpc;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--status") {
        print_status().await;
        return;
    }

    let tts_only = args.iter().any(|a| a == "--tts-only");

    eprintln!("voiced: starting voice daemon");
    if tts_only {
        eprintln!("voiced: TTS-only mode enabled (STT loads only if listen/converse is requested)");
    }

    // Check if another instance is already running
    let sock_path = socket::socket_path();
    if sock_path.exists() {
        if tokio::net::UnixStream::connect(&sock_path).await.is_ok() {
            eprintln!(
                "voiced: another instance is already running at {}",
                sock_path.display()
            );
            eprintln!("voiced: use `voiced --status` to check state");
            std::process::exit(1);
        }
        eprintln!("voiced: removing stale socket");
        std::fs::remove_file(&sock_path).ok();
    }

    let queue = Arc::new(RequestQueue::new());
    let config = Arc::new(DaemonConfig::new());

    // Load or create Automerge state
    let automerge = match AutomergeState::load_or_create() {
        Ok(state) => Arc::new(Mutex::new(state)),
        Err(e) => {
            eprintln!("voiced: failed to load automerge state: {}", e);
            std::process::exit(1);
        }
    };

    // Persist an initial idle snapshot so file-watching clients can attach
    // before the first queue transition.
    {
        let snapshot = queue.snapshot().await;
        let mut am = automerge.lock().await;
        am.update(&snapshot);
        if let Err(e) = am.save() {
            eprintln!("voiced: failed to save initial automerge state: {}", e);
            std::process::exit(1);
        }
    }

    // Handle ctrl-c
    tokio::spawn({
        async move {
            tokio::signal::ctrl_c().await.ok();
            eprintln!("\nvoiced: shutting down");
            socket::cleanup();
            std::process::exit(0);
        }
    });

    // Start worker and socket server concurrently
    let worker_queue = queue.clone();
    let worker_config = config.clone();
    let worker_automerge = automerge.clone();
    tokio::spawn(async move {
        worker::run(worker_queue, worker_config, worker_automerge, tts_only).await;
    });

    // Start cleanup task
    let cleanup_queue = queue.clone();
    let cleanup_automerge = automerge.clone();
    tokio::spawn(async move {
        cleanup::run(cleanup_queue, cleanup_automerge).await;
    });

    socket::serve(queue, config, automerge).await;
}

async fn print_status() {
    let path = socket::socket_path();
    if !path.exists() {
        println!("voiced: not running (no socket at {})", path.display());
        return;
    }

    match tokio::net::UnixStream::connect(&path).await {
        Ok(stream) => {
            let (mut reader, mut writer) = stream.into_split();

            // Send a status request using the frame protocol
            let req = rpc::Request::new("status", serde_json::json!({})).with_id(1);
            let json = serde_json::to_vec(&req).unwrap();
            let frame = Frame::request(&json);
            if write_frame(&mut writer, &frame).await.is_err() {
                println!("voiced: failed to send status request");
                return;
            }

            // Read the response frame
            match read_frame(&mut reader).await {
                Ok(Some(frame)) if frame.frame_type == FrameType::Response => {
                    if let Ok(resp) = frame.json::<rpc::Response>() {
                        if let Some(result) = resp.result {
                            println!("{}", serde_json::to_string_pretty(&result).unwrap());
                        } else if let Some(err) = resp.error {
                            println!("Error: {}", err.message);
                        }
                    }
                }
                _ => println!("voiced: unexpected response"),
            }
        }
        Err(_) => {
            println!(
                "voiced: not responding (stale socket at {})",
                path.display()
            );
        }
    }
}
