//! Queue worker — processes voice requests one at a time.
//!
//! Owns the TTS and STT models and audio hardware. Runs blocking GPU
//! inference and audio I/O on dedicated threads via spawn_blocking.

use crate::queue::{RequestQueue, VoiceRequest};
use candle_core::Tensor;
use rodio::microphone::MicrophoneBuilder;
use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, Player};
use std::collections::HashMap;
use std::num::NonZero;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use voice_tts::KokoroModel;

const MODEL_REPO: &str = "prince-canuma/Kokoro-82M";
const STT_REPO: &str = "distil-whisper/distil-large-v3";

// -- TTS state ---------------------------------------------------------------

struct TtsState {
    model: KokoroModel,
    default_voice: Tensor,
    #[allow(dead_code)]
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

// -- Worker entry point -------------------------------------------------------

pub async fn run(queue: Arc<RequestQueue>) {
    eprintln!("voiced: loading TTS model...");
    let start = Instant::now();

    let tts = match tokio::task::spawn_blocking(init_tts).await {
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

    // Eagerly load STT model — daemon is long-lived, pay the cost once
    eprintln!("voiced: loading STT model...");
    let stt_start = Instant::now();
    let stt: Arc<Mutex<Option<voice_stt::WhisperModel>>> = match tokio::task::spawn_blocking(|| {
        voice_stt::load_model(STT_REPO).map_err(|e| format!("stt: {}", e))
    })
    .await
    {
        Ok(Ok(model)) => {
            eprintln!(
                "voiced: STT model loaded in {:.1}s",
                stt_start.elapsed().as_secs_f32()
            );
            Arc::new(Mutex::new(Some(model)))
        }
        Ok(Err(e)) => {
            eprintln!("voiced: STT model failed to load: {}", e);
            eprintln!("voiced: listen/converse will be unavailable");
            Arc::new(Mutex::new(None))
        }
        Err(e) => {
            eprintln!("voiced: STT init panicked: {}", e);
            Arc::new(Mutex::new(None))
        }
    };

    eprintln!(
        "voiced: all models ready ({:.1}s total)",
        start.elapsed().as_secs_f32()
    );

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
                VoiceRequest::Listen { max_duration_ms } => {
                    let max_ms = *max_duration_ms;
                    let stt = stt.clone();

                    let result = tokio::task::spawn_blocking(move || listen(&stt, max_ms)).await;

                    match result {
                        Ok(Ok(msg)) => queue.complete(Some(msg)).await,
                        Ok(Err(e)) => {
                            eprintln!("voiced: listen error: {}", e);
                            queue.fail(e).await;
                        }
                        Err(e) => {
                            eprintln!("voiced: listen panicked: {}", e);
                            queue.fail(format!("panic: {}", e)).await;
                        }
                    }
                }
                VoiceRequest::Converse { text, voice } => {
                    let text = text.clone();
                    let voice = voice.clone();
                    let tts = tts.clone();
                    let stt = stt.clone();

                    // Speak
                    let speak_result = tokio::task::spawn_blocking(move || {
                        let spoke = speak(&tts, &text, voice.as_deref(), None)?;
                        let heard = listen(&stt, None)?;
                        Ok::<String, String>(format!("{}, {}", spoke, heard))
                    })
                    .await;

                    match speak_result {
                        Ok(Ok(msg)) => queue.complete(Some(msg)).await,
                        Ok(Err(e)) => {
                            eprintln!("voiced: converse error: {}", e);
                            queue.fail(e).await;
                        }
                        Err(e) => {
                            eprintln!("voiced: converse panicked: {}", e);
                            queue.fail(format!("panic: {}", e)).await;
                        }
                    }
                }
            }
        }
    }
}

// -- TTS init + speak ---------------------------------------------------------

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
    let chunks =
        voice_g2p::text_to_phoneme_chunks(text).map_err(|e| format!("G2P error: {}", e))?;

    if chunks.is_empty() {
        return Ok("(empty text)".to_string());
    }

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
                Err(e) => return Err(format!("generate chunk {}: {}", i + 1, e)),
            }
        }
    }

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

// -- STT listen ---------------------------------------------------------------

fn ensure_stt(stt: &Arc<Mutex<Option<voice_stt::WhisperModel>>>) -> Result<(), String> {
    let mut guard = stt.lock().map_err(|e| format!("stt lock: {}", e))?;
    if guard.is_none() {
        eprintln!("voiced: loading STT model ({})...", STT_REPO);
        let start = Instant::now();
        let model =
            voice_stt::load_model(STT_REPO).map_err(|e| format!("stt load_model: {}", e))?;
        eprintln!(
            "voiced: STT model loaded in {:.1}s",
            start.elapsed().as_secs_f32()
        );
        *guard = Some(model);
    }
    Ok(())
}

fn listen(
    stt: &Arc<Mutex<Option<voice_stt::WhisperModel>>>,
    max_duration_ms: Option<u64>,
) -> Result<String, String> {
    ensure_stt(stt)?;

    let max_ms = max_duration_ms.unwrap_or(60000);

    eprintln!("voiced: listening (max {}ms)...", max_ms);

    // Play a ding to signal recording start
    play_tone(880.0, 0.15);
    // Brief pause so the ding finishes before mic opens
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Open mic
    let mic = MicrophoneBuilder::new()
        .default_device()
        .map_err(|e| format!("mic: no input device: {}", e))?
        .default_config()
        .map_err(|e| format!("mic: no config: {}", e))?
        .open_stream()
        .map_err(|e| format!("mic: open failed: {}", e))?;

    let sample_rate = mic.config().sample_rate.get();
    let channels = mic.config().channel_count.get();

    // Record with VAD
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let recent_peak = Arc::new(AtomicU32::new(0));
    let stop = Arc::new(AtomicBool::new(false));

    // Mic drain thread — matches the CLI's start_mic_drain pattern:
    // track local chunk_peak, publish + reset every 100 samples.
    let buf_clone = buffer.clone();
    let peak_clone = recent_peak.clone();
    let stop_clone = stop.clone();
    let mic_thread = std::thread::spawn(move || {
        let ch = channels.max(1) as usize;
        let mut chunk_peak: f32 = 0.0;
        let mut sample_count = 0usize;

        for sample in mic.into_iter() {
            if stop_clone.load(Ordering::Relaxed) {
                break;
            }

            let abs = sample.abs();
            chunk_peak = chunk_peak.max(abs);
            sample_count += 1;

            // Multi-channel: take every ch-th sample (mono mix)
            if ch == 1 {
                buf_clone.lock().unwrap().push(sample);
            } else if sample_count % ch == 0 {
                buf_clone.lock().unwrap().push(sample);
            }

            // Publish peak every ~100 samples, then reset
            if sample_count % 100 == 0 {
                peak_clone.store(chunk_peak.to_bits(), Ordering::Relaxed);
                chunk_peak = 0.0;
            }
        }
    });

    // Calibrate noise floor — sample peak amplitude over 500ms.
    // The mic may take a moment to warm up (especially Bluetooth).
    std::thread::sleep(std::time::Duration::from_millis(500));
    let noise_floor = f32::from_bits(recent_peak.swap(0f32.to_bits(), Ordering::Relaxed));
    // Threshold must be well above noise to avoid false positives.
    // Use the same heuristic as the CLI: max(noise * 3, 0.01)
    let threshold = (noise_floor * 3.0).max(0.01);
    eprintln!(
        "voiced: noise floor: {:.4}, threshold: {:.4}",
        noise_floor, threshold
    );

    // VAD state machine: wait for speech, then stop after 2s of silence.
    let started = Instant::now();
    let max_dur = std::time::Duration::from_millis(max_ms);
    let silence_timeout = std::time::Duration::from_millis(2000);
    let mut speech_detected = false;
    let mut last_speech = Instant::now();
    let mut consecutive_silent = 0u32;

    loop {
        if started.elapsed() > max_dur {
            eprintln!("voiced: max duration reached");
            break;
        }

        std::thread::sleep(std::time::Duration::from_millis(50));
        let peak = f32::from_bits(recent_peak.swap(0f32.to_bits(), Ordering::Relaxed));

        if peak > threshold {
            consecutive_silent = 0;
            if !speech_detected {
                eprintln!("voiced: speech detected (peak: {:.4})", peak);
                speech_detected = true;
            }
            last_speech = Instant::now();
        } else {
            consecutive_silent += 1;
            if speech_detected && last_speech.elapsed() > silence_timeout {
                eprintln!(
                    "voiced: silence for {:.1}s, stopping",
                    last_speech.elapsed().as_secs_f32()
                );
                break;
            }
        }
    }

    stop.store(true, Ordering::Relaxed);
    let _ = mic_thread.join();

    // Play stop tone
    play_tone(440.0, 0.1);

    let samples = match Arc::try_unwrap(buffer) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(arc) => arc.lock().unwrap().clone(),
    };

    if samples.is_empty() || !speech_detected {
        return Ok("(no speech detected)".to_string());
    }

    let duration_s = samples.len() as f32 / sample_rate as f32;
    eprintln!("voiced: recorded {:.1}s, transcribing...", duration_s);

    // Transcribe
    let mut guard = stt.lock().map_err(|e| format!("stt lock: {}", e))?;
    let model = guard.as_mut().ok_or("STT model not loaded")?;

    let result = voice_stt::transcribe_audio(model, &samples, sample_rate)
        .map_err(|e| format!("transcribe: {}", e))?;

    let text = result.text.trim().to_string();
    eprintln!("voiced: heard: {}", text);

    Ok(format!("heard: {}", text))
}

/// Play a simple sine tone (for ding/dong feedback).
fn play_tone(freq: f32, duration_secs: f32) {
    let sample_rate = 24000u32;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let envelope = if t < 0.01 {
                t / 0.01
            } else {
                (1.0 - (t - 0.01) / (duration_secs - 0.01)).max(0.0)
            };
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3 * envelope
        })
        .collect();

    if let Ok(mut stream) = DeviceSinkBuilder::open_default_sink() {
        stream.log_on_drop(false);
        let player = Player::connect_new(stream.mixer());
        let channels = NonZero::new(1u16).unwrap();
        let rate = NonZero::new(sample_rate).unwrap();
        player.append(SamplesBuffer::new(channels, rate, samples));
        while !player.empty() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}

// -- Simulation fallback ------------------------------------------------------

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
