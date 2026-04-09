//! voiced — the voice daemon.
//!
//! Listens on ~/.voice/daemon.sock for TTS/STT requests and processes
//! them sequentially so multiple MCP clients never overlap audio.
//!
//! Usage:
//!   voiced              # start the daemon
//!   voiced --status     # print daemon state and exit

mod queue;
mod socket;
mod worker;

use queue::RequestQueue;
use std::sync::Arc;

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
        // Try connecting — if it works, another daemon is running
        if tokio::net::UnixStream::connect(&sock_path).await.is_ok() {
            eprintln!(
                "voiced: another instance is already running at {}",
                sock_path.display()
            );
            eprintln!("voiced: use `voiced --status` to check state");
            std::process::exit(1);
        }
        // Stale socket, remove it
        eprintln!("voiced: removing stale socket");
        std::fs::remove_file(&sock_path).ok();
    }

    let queue = Arc::new(RequestQueue::new());

    // Handle ctrl-c
    let cleanup_queue = queue.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        eprintln!("\nvoiced: shutting down");
        let _ = cleanup_queue; // keep alive
        socket::cleanup();
        std::process::exit(0);
    });

    // Start worker and socket server concurrently
    let worker_queue = queue.clone();
    tokio::spawn(async move {
        worker::run(worker_queue).await;
    });

    socket::serve(queue).await;
}

async fn print_status() {
    let path = socket::socket_path();
    if !path.exists() {
        println!("voiced: not running (no socket at {})", path.display());
        return;
    }

    match tokio::net::UnixStream::connect(&path).await {
        Ok(stream) => {
            use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
            let (reader, mut writer) = stream.into_split();
            let req = r#"{"type":"status"}"#;
            writer.write_all(format!("{}\n", req).as_bytes()).await.ok();
            let mut lines = BufReader::new(reader).lines();
            if let Ok(Some(line)) = lines.next_line().await {
                // Pretty-print the JSON
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                    println!("{}", serde_json::to_string_pretty(&val).unwrap());
                } else {
                    println!("{}", line);
                }
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
