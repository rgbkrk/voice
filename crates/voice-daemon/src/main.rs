//! voiced — the voice daemon.
//!
//! Listens on ~/.voice/daemon.sock for TTS/STT requests and processes
//! them sequentially so multiple MCP clients never overlap audio.
//!
//! Usage:
//!   voiced              # start the daemon
//!   voiced --status     # print daemon state and exit

mod audio_recorder;
mod automerge_state;
mod config;
mod queue;
mod socket;
mod worker;

use config::DaemonConfig;
use queue::RequestQueue;
use std::sync::Arc;
use voice_protocol::frames::{read_frame, write_frame, Frame, FrameType};
use voice_protocol::rpc;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--status") {
        print_status().await;
        return;
    }

    eprintln!("voiced: starting voice daemon");

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
    tokio::spawn(async move {
        worker::run(worker_queue, worker_config).await;
    });

    socket::serve(queue, config).await;
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
