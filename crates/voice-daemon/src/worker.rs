//! Queue worker — processes voice requests one at a time.
//!
//! Owns the TTS model and audio output. Runs blocking GPU inference
//! and audio playback on a dedicated thread via spawn_blocking.

use crate::queue::{RequestQueue, VoiceRequest};
use candle_core::Tensor;
use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, Player};
use std::collections::HashMap;
use std::num::NonZero;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use voice_tts::KokoroModel;

/// Shared TTS state — model, voices, config. Protected by a std::sync::Mutex
/// (not tokio) because all access is from spawn_blocking threads.
struct TtsState {
    model: KokoroModel,
    default_voice: Tensor,
    default_voice_name: String,
    voice_cache: HashMap<String, Tensor>,
    speed: f32,
    sample_rate: u32,
    repo_id: String,
}

impl TtsState {
    fn get_voice(&mut self, name: &str) -> Result<&Tensor, String> {
        if !self.voice_cache.contains_key(name) {
            let v = self
                .model
                .load_voice(name, Some(&self.repo_id))
                .map_err(|e| format!("Failed to load voice '{}': {}", name, e))?;
            self.voice_cache.insert(name.to_string(), v);
        }
        Ok(&self.voice_cache[name])
    }
}

pub async fn run(queue: Arc<RequestQueue>) {
    eprintln!("voiced: loading TTS model...");
    let start = Instant::now();

    // Load model on a blocking thread (heavy GPU init)
    let tts = match tokio::task::spawn_blocking(|| init_tts()).await {
        Ok(Ok(tts)) => Arc::new(Mutex::new(tts)),
        Ok(Err(e)) => {
            eprintln!("voiced: failed to load TTS model: {}", e);
            eprintln!("voiced: running in simulation mode");
            run_simulated(queue).await;
            return;
        }
        Err(e) => {
            eprintln!("voiced: TTS init panicked: {}", e);
            return;
        }
    };

    eprintln!(
        "voiced: TTS model loaded in {:.1}s",
        start.elapsed().as_secs_f32()
    );
    eprintln!("voiced: worker ready");

    loop {
        queue.notify.notified().await;

        while let Some(entry) = queue.dequeue().await {
            eprintln!(
                "voiced: [{}/{}] {}",
                entry.id,
                entry.client_id,
                short(&entry.request)
            );

            match &entry.request {
                VoiceRequest::Speak { text, voice, speed } => {
                    let text = text.clone();
                    let voice = voice.clone();
                    let speed = *speed;
                    let tts = tts.clone();

                    let result = tokio::task::spawn_blocking(move || {
                        speak(&tts, &text, voice.as_deref(), speed)
                    })
                    .await;

                    match result {
                        Ok(Ok(msg)) => queue.complete(Some(msg)).await,
                        Ok(Err(e)) => {
                            eprintln!("voiced: speak error: {}", e);
                            queue.fail(e).await;
                        }
                        Err(e) => {
                            eprintln!("voiced: speak panicked: {}", e);
                            queue.fail(format!("panic: {}", e)).await;
                        }
                    }
                }
                VoiceRequest::Listen { .. } => {
                    // TODO: voice-stt integration
                    eprintln!("voiced: listen not yet implemented");
                    queue
                        .complete(Some("(listen not yet implemented)".to_string()))
                        .await;
                }
                VoiceRequest::Converse { text, voice } => {
                    // Speak first, then listen (listen is stubbed for now)
                    let text = text.clone();
                    let voice = voice.clone();
                    let tts = tts.clone();

                    let speak_result = tokio::task::spawn_blocking(move || {
                        speak(&tts, &text, voice.as_deref(), None)
                    })
                    .await;

                    let spoke = match speak_result {
                        Ok(Ok(msg)) => msg,
                        Ok(Err(e)) => format!("speak error: {}", e),
                        Err(e) => format!("panic: {}", e),
                    };

                    queue
                        .complete(Some(format!(
                            "spoke: {}, listen: (not yet implemented)",
                            spoke
                        )))
                        .await;
                }
            }
        }
    }
}

const MODEL_REPO: &str = "prince-canuma/Kokoro-82M";

fn init_tts() -> Result<TtsState, String> {
    let model = voice_tts::load_model(MODEL_REPO).map_err(|e| format!("load_model: {}", e))?;
    let sample_rate = model.sample_rate;

    let default_voice_name = "af_heart".to_string();
    let voice = model
        .load_voice(&default_voice_name, Some(MODEL_REPO))
        .map_err(|e| e.to_string())?;

    let mut voice_cache = HashMap::new();
    voice_cache.insert(default_voice_name.clone(), voice.clone());

    Ok(TtsState {
        model,
        default_voice: voice,
        default_voice_name,
        voice_cache,
        speed: 1.0,
        sample_rate,
        repo_id: MODEL_REPO.to_string(),
    })
}

fn speak(
    tts: &Arc<Mutex<TtsState>>,
    text: &str,
    voice_name: Option<&str>,
    speed: Option<f64>,
) -> Result<String, String> {
    // G2P
    let chunks =
        voice_g2p::text_to_phoneme_chunks(text).map_err(|e| format!("G2P error: {}", e))?;

    if chunks.is_empty() {
        return Ok("(empty text)".to_string());
    }

    // Open audio output
    let mut stream = DeviceSinkBuilder::open_default_sink().map_err(|e| format!("audio: {}", e))?;
    stream.log_on_drop(false);
    let player = Player::connect_new(stream.mixer());

    let started = Instant::now();
    let mut total_samples = 0usize;

    {
        let mut state = tts.lock().map_err(|e| format!("lock: {}", e))?;
        let speed = speed.map(|s| s as f32).unwrap_or(state.speed);
        let sample_rate = state.sample_rate;

        let channels = NonZero::new(1u16).unwrap();
        let rate = NonZero::new(sample_rate).unwrap();

        for (i, phonemes) in chunks.iter().enumerate() {
            if phonemes.is_empty() {
                continue;
            }

            // Clone the voice tensor so we can release the borrow on state
            // before passing &mut state.model to generate.
            let voice = if let Some(name) = voice_name {
                state.get_voice(name)?.clone()
            } else {
                state.default_voice.clone()
            };

            match voice_tts::generate(&mut state.model, phonemes, &voice, speed) {
                Ok(audio) => {
                    total_samples += audio.len();
                    let source = SamplesBuffer::new(channels, rate, audio);
                    player.append(source);
                    if chunks.len() > 1 {
                        eprintln!("voiced:   chunk {}/{} generated", i + 1, chunks.len());
                    }
                }
                Err(e) => {
                    return Err(format!("generate chunk {}: {}", i + 1, e));
                }
            }
        }
    }

    // Wait for playback to finish (release the mutex while waiting)
    while !player.empty() {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    let duration_ms = started.elapsed().as_millis();
    let audio_ms = total_samples as u64 * 1000 / 24000;

    Ok(format!(
        "spoke {} chunks, {}ms audio in {}ms",
        chunks.len(),
        audio_ms,
        duration_ms
    ))
}

/// Fallback when TTS model can't load — simulate with delays.
async fn run_simulated(queue: Arc<RequestQueue>) {
    eprintln!("voiced: worker ready (simulation mode)");
    loop {
        queue.notify.notified().await;
        while let Some(entry) = queue.dequeue().await {
            eprintln!(
                "voiced: [{}/{}] {} (simulated)",
                entry.id,
                entry.client_id,
                short(&entry.request)
            );
            match &entry.request {
                VoiceRequest::Speak { text, .. } => {
                    let words = text.split_whitespace().count();
                    let ms = (words as u64 * 200).max(500);
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                    queue
                        .complete(Some(format!("simulated {} words", words)))
                        .await;
                }
                VoiceRequest::Listen { .. } => {
                    tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
                    queue.complete(Some("(simulated listen)".to_string())).await;
                }
                VoiceRequest::Converse { text, .. } => {
                    let words = text.split_whitespace().count();
                    let ms = (words as u64 * 200).max(500);
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
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
