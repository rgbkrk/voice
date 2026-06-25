//! Voice daemon runtime.
//!
//! The daemon listens on `~/.voice/daemon.sock` for TTS/STT requests and
//! processes them sequentially so multiple MCP clients never overlap audio.

mod audio_recorder;
mod automerge_state;
mod cleanup;
mod config;
mod queue;
mod socket;
mod ui_server;
mod ui_state;
mod worker;

use automerge_state::AutomergeState;
use config::DaemonConfig;
use queue::RequestQueue;
use std::sync::Arc;
use tokio::sync::Mutex;

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

    let automerge = match AutomergeState::load_or_create() {
        Ok(state) => Arc::new(Mutex::new(state)),
        Err(e) => {
            eprintln!("voice daemon: failed to load automerge state: {e}");
            std::process::exit(1);
        }
    };

    {
        let snapshot = queue.snapshot().await;
        let mut am = automerge.lock().await;
        am.update(&snapshot);
        if let Err(e) = am.save() {
            eprintln!("voice daemon: failed to save initial automerge state: {e}");
            std::process::exit(1);
        }
    }

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
    let worker_automerge = automerge.clone();
    tokio::spawn(async move {
        worker::run(
            worker_queue,
            worker_config,
            worker_automerge,
            options.tts_only,
        )
        .await;
    });

    let cleanup_queue = queue.clone();
    let cleanup_automerge = automerge.clone();
    tokio::spawn(async move {
        cleanup::run(cleanup_queue, cleanup_automerge).await;
    });

    let ui_queue = queue.clone();
    tokio::spawn(async move {
        ui_server::serve(ui_queue).await;
    });

    socket::serve(queue, config, automerge).await;
}
