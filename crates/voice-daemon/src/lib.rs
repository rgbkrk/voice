//! Voice daemon runtime.
//!
//! The daemon listens on `~/.voice/daemon.sock` for TTS/STT requests and
//! processes them sequentially so multiple MCP clients never overlap audio.

mod config;
mod queue;
mod socket;
mod worker;

use config::DaemonConfig;
use queue::RequestQueue;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Default)]
pub struct DaemonOptions {
    pub tts_only: bool,
}

pub fn run_blocking(options: DaemonOptions) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| {
            eprintln!("voice daemon: failed to start async runtime: {e}");
            std::process::exit(1);
        });

    runtime.block_on(run(options));
}

pub async fn run(options: DaemonOptions) {
    eprintln!("voice daemon: starting");
    if options.tts_only {
        eprintln!(
            "voice daemon: TTS-only mode enabled (STT loads only if listen/converse is requested)"
        );
    }

    let sock_path = socket::socket_path();
    if sock_path.exists() {
        if tokio::net::UnixStream::connect(&sock_path).await.is_ok() {
            eprintln!(
                "voice daemon: another instance is already running at {}",
                sock_path.display()
            );
            eprintln!("voice daemon: use `voice daemon status` to check state");
            std::process::exit(1);
        }
        eprintln!("voice daemon: removing stale socket");
        std::fs::remove_file(&sock_path).ok();
    }

    let queue = Arc::new(RequestQueue::new());
    let config = Arc::new(DaemonConfig::new());

    tokio::spawn({
        async move {
            tokio::signal::ctrl_c().await.ok();
            eprintln!("\nvoice daemon: shutting down");
            socket::cleanup();
            std::process::exit(0);
        }
    });

    let worker_queue = queue.clone();
    let worker_config = config.clone();
    tokio::spawn(async move {
        worker::run(worker_queue, worker_config, options.tts_only).await;
    });

    socket::serve(queue, config).await;
}
