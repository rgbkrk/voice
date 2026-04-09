//! Background task: auto-clear completed items after 30 seconds.

use crate::{audio_recorder, automerge_state::AutomergeState, queue::RequestQueue};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{interval, Duration};

/// Run cleanup loop: every 10 seconds, remove expired items from recent.
pub async fn run(queue: Arc<RequestQueue>, automerge: Arc<Mutex<AutomergeState>>) {
    eprintln!("voiced: cleanup task started");
    let mut ticker = interval(Duration::from_secs(10));

    loop {
        ticker.tick().await;

        let snapshot = queue.snapshot().await;
        let now = now_secs();

        for item in &snapshot.recent {
            if let Some(clear_at) = item.auto_clear_at {
                if clear_at <= now {
                    eprintln!(
                        "voiced: auto-clearing item {} ({}s old)",
                        item.id,
                        now - item.created_at
                    );

                    // Delete audio files
                    if let Err(e) = audio_recorder::delete_audio(&item.id) {
                        eprintln!("voiced: failed to delete audio for {}: {}", item.id, e);
                    }

                    // Remove from Automerge doc
                    {
                        let mut am = automerge.lock().await;
                        am.remove_from_recent(&item.id);
                        if let Err(e) = am.save() {
                            eprintln!("voiced: failed to save automerge after cleanup: {}", e);
                        }
                    }

                    // Remove from in-memory queue
                    queue.remove_recent(&item.id).await;
                }
            }
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
