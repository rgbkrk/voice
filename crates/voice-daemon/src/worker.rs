//! Queue worker — processes voice requests one at a time.
//!
//! Prototype: simulates TTS/STT with delays.
//! Production: will call voice-tts and voice-stt crates.

use crate::queue::{RequestQueue, VoiceRequest};
use std::sync::Arc;

pub async fn run(queue: Arc<RequestQueue>) {
    eprintln!("voiced: worker ready");

    loop {
        queue.notify.notified().await;

        while let Some(item) = queue.dequeue().await {
            eprintln!(
                "voiced: [{}/{}] {}",
                item.id,
                item.client_id,
                short(&item.request)
            );

            match &item.request {
                VoiceRequest::Speak { text, .. } => {
                    // TODO: voice-tts generate + rodio playback
                    let words = text.split_whitespace().count();
                    let ms = (words as u64 * 200).max(500);
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                    queue.complete(Some(format!("spoke {} words", words))).await;
                }
                VoiceRequest::Listen { .. } => {
                    // TODO: voice-stt mic + transcribe
                    tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
                    queue
                        .complete(Some("(simulated transcription)".to_string()))
                        .await;
                }
                VoiceRequest::Converse { text, .. } => {
                    // TODO: speak then listen
                    let words = text.split_whitespace().count();
                    let ms = (words as u64 * 200).max(500);
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                    tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
                    queue
                        .complete(Some("(simulated converse)".to_string()))
                        .await;
                }
            }
        }
    }
}

fn short(req: &VoiceRequest) -> String {
    match req {
        VoiceRequest::Speak { text, .. } => {
            let preview: String = text.chars().take(50).collect();
            format!("speak: {}", preview)
        }
        VoiceRequest::Listen { max_duration_ms } => {
            format!("listen ({}ms)", max_duration_ms.unwrap_or(30000))
        }
        VoiceRequest::Converse { text, .. } => {
            let preview: String = text.chars().take(50).collect();
            format!("converse: {}", preview)
        }
    }
}
