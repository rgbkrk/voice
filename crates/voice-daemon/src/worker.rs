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
use voice_stream::{
    resample_linear, AudioEncoding, InterleavedMonoMixer, Packetizer, StreamEnded, StreamMetadata,
    TtsStreamEvent,
};
use voice_tts::KokoroModel;

use crate::audio_recorder;

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

struct StreamSpeakJob {
    text: String,
    stream_id: String,
    voice_name: Option<String>,
    speed: Option<f64>,
    sample_rate: u32,
    frame_ms: u32,
    event_tx: tokio::sync::mpsc::Sender<TtsStreamEvent>,
    cancelled: Arc<AtomicBool>,
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

async fn sync_automerge(
    queue: &RequestQueue,
    automerge: &Arc<tokio::sync::Mutex<crate::automerge_state::AutomergeState>>,
) {
    let snapshot = queue.snapshot().await;
    let mut am = automerge.lock().await;
    am.update(&snapshot);
    if let Err(e) = am.save() {
        eprintln!("voiced: failed to save automerge doc: {}", e);
    }
}

pub async fn run(
    queue: Arc<RequestQueue>,
    config: Arc<crate::config::DaemonConfig>,
    automerge: Arc<tokio::sync::Mutex<crate::automerge_state::AutomergeState>>,
    tts_only: bool,
) {
    eprintln!("voiced: loading TTS model...");
    let start = Instant::now();

    let tts = match tokio::task::spawn_blocking(init_tts).await {
        Ok(Ok(tts)) => Arc::new(Mutex::new(tts)),
        Ok(Err(e)) => {
            eprintln!("voiced: failed to load TTS model: {}", e);
            eprintln!("voiced: running in simulation mode");
            run_simulated(queue, automerge).await;
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

    let stt: Arc<Mutex<Option<voice_stt::WhisperModel>>> = if tts_only {
        eprintln!("voiced: skipping eager STT load (TTS-only mode)");
        Arc::new(Mutex::new(None))
    } else {
        // Eagerly load STT model — daemon is long-lived, pay the cost once
        eprintln!("voiced: loading STT model...");
        let stt_start = Instant::now();
        match tokio::task::spawn_blocking(|| {
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
        }
    };

    eprintln!(
        "voiced: all models ready ({:.1}s total)",
        start.elapsed().as_secs_f32()
    );

    loop {
        queue.notify.notified().await;

        while let Some(entry) = queue.dequeue().await {
            sync_automerge(&queue, &automerge).await;
            eprintln!(
                "voiced: [{}/{}] {}",
                entry.id,
                entry.client_id,
                short(&entry.request)
            );

            match &entry.request {
                VoiceRequest::Speak { text, voice, speed } => {
                    let text = text.clone();
                    // Use daemon config defaults when request doesn't specify
                    let voice = voice.clone().or_else(|| Some(config.get_voice_name()));
                    let speed = speed.or_else(|| Some(config.get_speed() as f64));
                    let tts = tts.clone();
                    let queue_id = entry.id.clone();
                    let cancelled = entry.cancelled.clone();

                    let result = tokio::task::spawn_blocking(move || {
                        speak(
                            &tts,
                            &text,
                            voice.as_deref(),
                            speed,
                            Some(&queue_id),
                            &cancelled,
                        )
                    })
                    .await;

                    match result {
                        Ok(Ok(msg)) => {
                            queue.complete(Some(msg), None).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                        Ok(Err(e)) => {
                            eprintln!("voiced: speak error: {}", e);
                            queue.fail(e).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                        Err(e) => {
                            eprintln!("voiced: speak panicked: {}", e);
                            queue.fail(format!("panic: {}", e)).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                    }
                }
                VoiceRequest::Synthesize {
                    text,
                    output_path,
                    voice,
                    speed,
                } => {
                    let text = text.clone();
                    let output_path = output_path.clone();
                    let voice = voice.clone().or_else(|| Some(config.get_voice_name()));
                    let speed = speed.or_else(|| Some(config.get_speed() as f64));
                    let tts = tts.clone();
                    let cancelled = entry.cancelled.clone();

                    let result = tokio::task::spawn_blocking(move || {
                        synthesize_to_file(
                            &tts,
                            &text,
                            &output_path,
                            voice.as_deref(),
                            speed,
                            &cancelled,
                        )
                    })
                    .await;

                    match result {
                        Ok(Ok(msg)) => {
                            queue.complete(Some(msg), None).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                        Ok(Err(e)) => {
                            eprintln!("voiced: synthesize error: {}", e);
                            queue.fail(e).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                        Err(e) => {
                            eprintln!("voiced: synthesize panicked: {}", e);
                            queue.fail(format!("panic: {}", e)).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                    }
                }
                VoiceRequest::StreamSpeak(request) => {
                    let job = StreamSpeakJob {
                        text: request.text.clone(),
                        stream_id: request.stream_id.clone(),
                        voice_name: request
                            .voice
                            .clone()
                            .or_else(|| Some(config.get_voice_name())),
                        speed: request.speed.or_else(|| Some(config.get_speed() as f64)),
                        sample_rate: request.sample_rate,
                        frame_ms: request.frame_ms,
                        event_tx: request.event_tx.clone(),
                        cancelled: entry.cancelled.clone(),
                    };
                    let tts = tts.clone();

                    let result = tokio::task::spawn_blocking(move || stream_speak(&tts, job)).await;

                    match result {
                        Ok(Ok(msg)) => {
                            queue.complete(Some(msg), None).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                        Ok(Err(e)) => {
                            eprintln!("voiced: stream_speak error: {}", e);
                            queue.fail(e).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                        Err(e) => {
                            eprintln!("voiced: stream_speak panicked: {}", e);
                            queue.fail(format!("panic: {}", e)).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                    }
                }
                VoiceRequest::Listen { max_duration_ms } => {
                    let max_ms = *max_duration_ms;
                    let stt = stt.clone();
                    let queue_id = entry.id.clone();

                    let result =
                        tokio::task::spawn_blocking(move || listen(&stt, max_ms, Some(&queue_id)))
                            .await;

                    match result {
                        Ok(Ok(msg)) => {
                            queue.complete(Some(msg), None).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                        Ok(Err(e)) => {
                            eprintln!("voiced: listen error: {}", e);
                            queue.fail(e).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                        Err(e) => {
                            eprintln!("voiced: listen panicked: {}", e);
                            queue.fail(format!("panic: {}", e)).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                    }
                }
                VoiceRequest::Converse { text, voice } => {
                    let text = text.clone();
                    let voice = voice.clone().or_else(|| Some(config.get_voice_name()));
                    let default_speed = Some(config.get_speed() as f64);
                    let tts = tts.clone();
                    let stt = stt.clone();
                    let queue_id = entry.id.clone(); // Capture for audio recording
                    let cancelled = entry.cancelled.clone();

                    // Speak then listen, return combined JSON
                    let speak_result = tokio::task::spawn_blocking(move || {
                        let spoke_json = speak(
                            &tts,
                            &text,
                            voice.as_deref(),
                            default_speed,
                            Some(&queue_id),
                            &cancelled,
                        )?;
                        let heard_json = listen(&stt, None, Some(&queue_id))?; // Pass queue_id for answer recording
                                                                               // Parse both results and combine into the converse format
                        let spoke: serde_json::Value =
                            serde_json::from_str(&spoke_json).unwrap_or_default();
                        let heard: serde_json::Value =
                            serde_json::from_str(&heard_json).unwrap_or_default();
                        Ok::<String, String>(
                            serde_json::json!({
                                "spoke": spoke,
                                "heard": heard,
                            })
                            .to_string(),
                        )
                    })
                    .await;

                    match speak_result {
                        Ok(Ok(msg)) => {
                            queue.complete(Some(msg), Some(30)).await; // Auto-clear after 30 seconds
                            sync_automerge(&queue, &automerge).await;
                        }
                        Ok(Err(e)) => {
                            eprintln!("voiced: converse error: {}", e);
                            queue.fail(e).await;
                            sync_automerge(&queue, &automerge).await;
                        }
                        Err(e) => {
                            eprintln!("voiced: converse panicked: {}", e);
                            queue.fail(format!("panic: {}", e)).await;
                            sync_automerge(&queue, &automerge).await;
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
    queue_id: Option<&str>,
    cancelled: &Arc<AtomicBool>,
) -> Result<String, String> {
    let chunks =
        voice_g2p::text_to_phoneme_chunks(text).map_err(|e| format!("G2P error: {}", e))?;

    if chunks.is_empty() {
        return Ok(serde_json::json!({"duration_ms": 0, "chunks": 0}).to_string());
    }

    let mut stream = DeviceSinkBuilder::open_default_sink().map_err(|e| format!("audio: {}", e))?;
    stream.log_on_drop(false);
    let player = Player::connect_new(stream.mixer());

    let started = Instant::now();
    let mut accumulated_audio: Vec<f32> = Vec::new();
    let sample_rate: u32;

    {
        let mut state = tts.lock().map_err(|e| format!("lock: {}", e))?;
        let speed = speed.map(|s| s as f32).unwrap_or(state.speed);
        sample_rate = state.sample_rate;
        let channels = NonZero::new(1u16).unwrap();
        let rate = NonZero::new(sample_rate).unwrap();

        for (i, phonemes) in chunks.iter().enumerate() {
            if cancelled.load(Ordering::SeqCst) {
                return Err("Cancelled by user".to_string());
            }
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
                    // Accumulate for WAV recording
                    if queue_id.is_some() {
                        accumulated_audio.extend_from_slice(&audio);
                    }

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

    // Save question audio if queue_id provided
    if let Some(qid) = queue_id {
        if !accumulated_audio.is_empty() {
            let path = audio_recorder::question_path(qid);
            audio_recorder::save_wav(&path, &accumulated_audio, sample_rate)?;
        }
    }

    let duration_ms = started.elapsed().as_millis() as u64;
    Ok(serde_json::json!({
        "duration_ms": duration_ms,
        "chunks": chunks.len(),
    })
    .to_string())
}

fn synthesize_to_file(
    tts: &Arc<Mutex<TtsState>>,
    text: &str,
    output_path: &str,
    voice_name: Option<&str>,
    speed: Option<f64>,
    cancelled: &Arc<AtomicBool>,
) -> Result<String, String> {
    let chunks =
        voice_g2p::text_to_phoneme_chunks(text).map_err(|e| format!("G2P error: {}", e))?;

    let started = Instant::now();
    let mut all_samples: Vec<f32> = Vec::new();
    let sample_rate: u32;
    let speed_used: f32;

    {
        let mut state = tts.lock().map_err(|e| format!("lock: {}", e))?;
        speed_used = speed.map(|s| s as f32).unwrap_or(state.speed);
        sample_rate = state.sample_rate;

        for (i, phonemes) in chunks.iter().enumerate() {
            if cancelled.load(Ordering::SeqCst) {
                return Err("Cancelled by user".to_string());
            }
            if phonemes.is_empty() {
                continue;
            }

            let voice = if let Some(name) = voice_name {
                state.get_voice(name)?.clone()
            } else {
                state.default_voice.clone()
            };

            match voice_tts::generate(&mut state.model, phonemes, &voice, speed_used) {
                Ok(audio) => {
                    all_samples.extend_from_slice(&audio);
                    if chunks.len() > 1 {
                        eprintln!("voiced:   chunk {}/{} synthesized", i + 1, chunks.len());
                    }
                }
                Err(e) => return Err(format!("generate chunk {}: {}", i + 1, e)),
            }
        }
    }

    if cancelled.load(Ordering::SeqCst) {
        return Err("Cancelled by user".to_string());
    }

    let path = std::path::Path::new(output_path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create output dir {}: {}", parent.display(), e))?;
        }
    }

    audio_recorder::save_wav(path, &all_samples, sample_rate)?;

    Ok(serde_json::json!({
        "output_path": output_path,
        "duration_ms": started.elapsed().as_millis() as u64,
        "chunks": chunks.len(),
        "samples": all_samples.len(),
        "sample_rate": sample_rate,
        "voice": voice_name,
        "speed": speed_used,
    })
    .to_string())
}

fn stream_speak(tts: &Arc<Mutex<TtsState>>, job: StreamSpeakJob) -> Result<String, String> {
    let StreamSpeakJob {
        text,
        stream_id,
        voice_name,
        speed,
        sample_rate,
        frame_ms,
        event_tx,
        cancelled,
    } = job;

    let started = Instant::now();
    let chunks = match voice_g2p::text_to_phoneme_chunks(&text) {
        Ok(chunks) => chunks,
        Err(e) => {
            let msg = format!("G2P error: {}", e);
            let _ = send_stream_event(
                &event_tx,
                TtsStreamEvent::error(stream_id.clone(), msg.clone()),
                &cancelled,
            );
            return Err(msg);
        }
    };

    let init_result: Result<(Tensor, u32, f32, Option<String>), String> = {
        let mut state = tts.lock().map_err(|e| format!("lock: {}", e))?;
        let voice_tensor = if let Some(name) = voice_name.as_deref() {
            state.get_voice(name)?.clone()
        } else {
            state.default_voice.clone()
        };
        Ok((
            voice_tensor,
            state.sample_rate,
            speed.map(|s| s as f32).unwrap_or(state.speed),
            voice_name
                .clone()
                .or_else(|| Some(state.default_voice_name.clone())),
        ))
    };
    let (voice_tensor, source_sample_rate, speed_used, resolved_voice) = match init_result {
        Ok(result) => result,
        Err(e) => {
            let _ = send_stream_event(
                &event_tx,
                TtsStreamEvent::error(stream_id.clone(), e.clone()),
                &cancelled,
            );
            return Err(e);
        }
    };

    let output_sample_rate = sample_rate.max(1);
    let frame_ms = frame_ms.max(1);
    let metadata = StreamMetadata {
        stream_id: stream_id.clone(),
        sample_rate: output_sample_rate,
        source_sample_rate,
        channels: 1,
        encoding: AudioEncoding::PcmS16Le,
        frame_ms,
        voice: resolved_voice.clone(),
        speed: speed_used,
        total_phoneme_chunks: chunks.len(),
    };
    send_stream_event(&event_tx, TtsStreamEvent::Started { metadata }, &cancelled)?;

    let mut packetizer = Packetizer::new(stream_id.clone(), output_sample_rate, frame_ms);

    for (i, phonemes) in chunks.iter().enumerate() {
        if cancelled.load(Ordering::SeqCst) {
            send_stream_event(
                &event_tx,
                TtsStreamEvent::cancelled(stream_id.clone(), "Cancelled by user"),
                &cancelled,
            )
            .ok();
            return Err("Cancelled by user".to_string());
        }
        if phonemes.is_empty() {
            continue;
        }

        let audio_result: Result<Vec<f32>, String> = {
            let mut state = tts.lock().map_err(|e| format!("lock: {}", e))?;
            voice_tts::generate(&mut state.model, phonemes, &voice_tensor, speed_used)
                .map_err(|e| format!("generate chunk {}: {}", i + 1, e))
        };
        let audio = match audio_result {
            Ok(audio) => audio,
            Err(e) => {
                let _ = send_stream_event(
                    &event_tx,
                    TtsStreamEvent::error(stream_id.clone(), e.clone()),
                    &cancelled,
                );
                return Err(e);
            }
        };

        let audio = if output_sample_rate == source_sample_rate {
            audio
        } else {
            resample_linear(&audio, source_sample_rate, output_sample_rate)
        };

        for frame in packetizer.push_samples(i as u32, &audio) {
            send_stream_event(&event_tx, TtsStreamEvent::Audio { frame }, &cancelled)?;
        }

        if chunks.len() > 1 {
            eprintln!(
                "voiced:   stream chunk {}/{} generated",
                i + 1,
                chunks.len()
            );
        }
    }

    if let Some(frame) = packetizer.finish(chunks.len().saturating_sub(1) as u32) {
        send_stream_event(&event_tx, TtsStreamEvent::Audio { frame }, &cancelled)?;
    }

    let samples = packetizer.samples_emitted();
    let duration_ms = if output_sample_rate == 0 {
        0
    } else {
        samples.saturating_mul(1_000) / output_sample_rate as u64
    };
    let ended = StreamEnded {
        stream_id: stream_id.clone(),
        frames: packetizer.frames_emitted(),
        samples,
        duration_ms,
        elapsed_ms: started.elapsed().as_millis() as u64,
    };
    send_stream_event(&event_tx, TtsStreamEvent::Ended(ended), &cancelled)?;

    Ok(serde_json::json!({
        "stream_id": stream_id,
        "duration_ms": duration_ms,
        "elapsed_ms": started.elapsed().as_millis() as u64,
        "chunks": chunks.len(),
        "frames": packetizer.frames_emitted(),
        "samples": samples,
        "sample_rate": output_sample_rate,
        "source_sample_rate": source_sample_rate,
        "frame_ms": frame_ms,
        "voice": resolved_voice,
        "speed": speed_used,
    })
    .to_string())
}

fn send_stream_event(
    event_tx: &tokio::sync::mpsc::Sender<TtsStreamEvent>,
    event: TtsStreamEvent,
    cancelled: &Arc<AtomicBool>,
) -> Result<(), String> {
    if cancelled.load(Ordering::SeqCst) && !event.is_terminal() {
        return Err("Cancelled by user".to_string());
    }
    event_tx
        .blocking_send(event)
        .map_err(|_| "stream receiver closed".to_string())
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
    queue_id: Option<&str>,
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
        let mut mixer = InterleavedMonoMixer::new(channels as usize);
        let mut chunk_peak: f32 = 0.0;
        let mut sample_count = 0usize;

        for sample in mic.into_iter() {
            if stop_clone.load(Ordering::Relaxed) {
                break;
            }

            let abs = sample.abs();
            chunk_peak = chunk_peak.max(abs);
            sample_count += 1;

            if let Some(mono) = mixer.push(sample) {
                buf_clone.lock().unwrap().push(mono);
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

    loop {
        if started.elapsed() > max_dur {
            eprintln!("voiced: max duration reached");
            break;
        }

        std::thread::sleep(std::time::Duration::from_millis(50));
        let peak = f32::from_bits(recent_peak.swap(0f32.to_bits(), Ordering::Relaxed));

        if peak > threshold {
            if !speech_detected {
                eprintln!("voiced: speech detected (peak: {:.4})", peak);
                speech_detected = true;
            }
            last_speech = Instant::now();
        } else if speech_detected && last_speech.elapsed() > silence_timeout {
            eprintln!(
                "voiced: silence for {:.1}s, stopping",
                last_speech.elapsed().as_secs_f32()
            );
            break;
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

    // Save answer audio if queue_id provided
    if let Some(qid) = queue_id {
        if !samples.is_empty() && speech_detected {
            let path = audio_recorder::answer_path(qid);
            // Save with original sample rate before transcription
            if let Err(e) = audio_recorder::save_wav(&path, &samples, sample_rate) {
                eprintln!("voiced: failed to save answer audio: {}", e);
            }
        }
    }

    if samples.is_empty() || !speech_detected {
        return Ok(serde_json::json!({
            "text": "",
            "tokens": 0,
            "duration_ms": started.elapsed().as_millis() as u64,
        })
        .to_string());
    }

    let duration_s = samples.len() as f32 / sample_rate as f32;
    eprintln!("voiced: recorded {:.1}s, transcribing...", duration_s);

    // Transcribe
    let mut guard = stt.lock().map_err(|e| format!("stt lock: {}", e))?;
    let model = guard.as_mut().ok_or("STT model not loaded")?;

    let result = voice_stt::transcribe_audio(model, &samples, sample_rate)
        .map_err(|e| format!("transcribe: {}", e))?;

    let text = result.text.trim().to_string();
    let duration_ms = started.elapsed().as_millis() as u64;
    eprintln!("voiced: heard: {}", text);

    Ok(serde_json::json!({
        "text": text,
        "tokens": result.tokens.len(),
        "duration_ms": duration_ms,
    })
    .to_string())
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

async fn run_simulated(
    queue: Arc<RequestQueue>,
    automerge: Arc<tokio::sync::Mutex<crate::automerge_state::AutomergeState>>,
) {
    eprintln!("voiced: worker ready (simulation mode)");
    loop {
        queue.notify.notified().await;
        while let Some(entry) = queue.dequeue().await {
            sync_automerge(&queue, &automerge).await;
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
                        .complete(Some(format!("simulated {} words", words)), None)
                        .await;
                    sync_automerge(&queue, &automerge).await;
                }
                VoiceRequest::Synthesize {
                    text, output_path, ..
                } => {
                    let words = text.split_whitespace().count();
                    let ms = (words as u64 * 50).max(100);
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                    let path = std::path::Path::new(output_path);
                    if let Some(parent) = path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let _ = audio_recorder::save_wav(path, &[], 24_000);
                    queue
                        .complete(
                            Some(
                                serde_json::json!({
                                    "output_path": output_path,
                                    "duration_ms": ms,
                                    "chunks": 0,
                                    "samples": 0,
                                    "sample_rate": 24000,
                                    "simulated": true,
                                })
                                .to_string(),
                            ),
                            None,
                        )
                        .await;
                    sync_automerge(&queue, &automerge).await;
                }
                VoiceRequest::StreamSpeak(request) => {
                    let words = request.text.split_whitespace().count();
                    let sample_rate = request.sample_rate;
                    let frame_ms = request.frame_ms;
                    let samples = (sample_rate as usize / 10).max(1);
                    let metadata = StreamMetadata {
                        stream_id: request.stream_id.clone(),
                        sample_rate,
                        source_sample_rate: sample_rate,
                        channels: 1,
                        encoding: AudioEncoding::PcmS16Le,
                        frame_ms,
                        voice: None,
                        speed: 1.0,
                        total_phoneme_chunks: 1,
                    };
                    let _ = request
                        .event_tx
                        .send(TtsStreamEvent::Started { metadata })
                        .await;
                    let mut packetizer =
                        Packetizer::new(request.stream_id.clone(), sample_rate, frame_ms);
                    for frame in packetizer.push_samples(0, &vec![0.0; samples]) {
                        let _ = request.event_tx.send(TtsStreamEvent::Audio { frame }).await;
                    }
                    if let Some(frame) = packetizer.finish(0) {
                        let _ = request.event_tx.send(TtsStreamEvent::Audio { frame }).await;
                    }
                    let ended = StreamEnded {
                        stream_id: request.stream_id.clone(),
                        frames: packetizer.frames_emitted(),
                        samples: packetizer.samples_emitted(),
                        duration_ms: packetizer.samples_emitted().saturating_mul(1_000)
                            / sample_rate.max(1) as u64,
                        elapsed_ms: words as u64,
                    };
                    let _ = request.event_tx.send(TtsStreamEvent::Ended(ended)).await;
                    queue
                        .complete(Some("simulated stream".to_string()), None)
                        .await;
                    sync_automerge(&queue, &automerge).await;
                }
                VoiceRequest::Listen { .. } => {
                    tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
                    queue
                        .complete(Some("(simulated listen)".to_string()), None)
                        .await;
                    sync_automerge(&queue, &automerge).await;
                }
                VoiceRequest::Converse { text, .. } => {
                    let words = text.split_whitespace().count();
                    let ms = (words as u64 * 200).max(500);
                    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                    queue
                        .complete(Some("(simulated converse)".to_string()), None)
                        .await;
                    sync_automerge(&queue, &automerge).await;
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
        VoiceRequest::Synthesize {
            text, output_path, ..
        } => {
            let preview: String = text.chars().take(50).collect();
            format!("synthesize: {} -> {}", preview, output_path)
        }
        VoiceRequest::StreamSpeak(request) => {
            let preview: String = request.text.chars().take(50).collect();
            format!("stream_speak: {}", preview)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automerge_state::AutomergeState;
    use crate::queue::CompletionResult;
    use voice_protocol::rpc::ItemStatus;

    async fn start_simulated_worker() -> (Arc<RequestQueue>, tokio::task::JoinHandle<()>) {
        let queue = Arc::new(RequestQueue::new());
        let automerge = Arc::new(tokio::sync::Mutex::new(AutomergeState::new()));
        let handle = tokio::spawn(run_simulated(queue.clone(), automerge));
        tokio::task::yield_now().await;
        (queue, handle)
    }

    async fn await_completion(
        rx: tokio::sync::oneshot::Receiver<CompletionResult>,
    ) -> CompletionResult {
        tokio::time::timeout(std::time::Duration::from_secs(2), rx)
            .await
            .expect("simulated worker timed out")
            .expect("completion channel closed")
    }

    #[tokio::test]
    async fn simulated_synthesize_writes_wav_and_completes() {
        let (queue, worker) = start_simulated_worker().await;
        let output_path = std::env::temp_dir().join(format!(
            "voice-daemon-simulated-synth-{}-{}.wav",
            std::process::id(),
            unique_suffix()
        ));

        let (_queue_id, rx) = queue
            .enqueue_and_wait(
                "test-client".to_string(),
                VoiceRequest::Synthesize {
                    text: "hello from simulated synthesis".to_string(),
                    output_path: output_path.to_string_lossy().to_string(),
                    voice: None,
                    speed: None,
                },
            )
            .await;

        let result = await_completion(rx).await;
        assert_eq!(result.status, ItemStatus::Completed);
        let result_json: serde_json::Value =
            serde_json::from_str(result.result.as_deref().unwrap()).unwrap();
        assert_eq!(
            result_json.get("simulated").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert!(output_path.exists(), "simulated synth should create WAV");

        let (_samples, sample_rate) = audio_recorder::read_wav(&output_path).unwrap();
        assert_eq!(sample_rate, 24_000);

        let _ = std::fs::remove_file(output_path);
        worker.abort();
    }

    #[tokio::test]
    async fn simulated_stream_emits_started_audio_and_ended_events() {
        let (queue, worker) = start_simulated_worker().await;
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(16);
        let stream_id = format!("stream-{}", unique_suffix());

        let (_queue_id, completion_rx) = queue
            .enqueue_and_wait(
                "test-client".to_string(),
                VoiceRequest::StreamSpeak(crate::queue::StreamSpeakRequest {
                    text: "hello from simulated stream".to_string(),
                    stream_id: stream_id.clone(),
                    voice: None,
                    speed: None,
                    sample_rate: 48_000,
                    frame_ms: 20,
                    event_tx,
                }),
            )
            .await;

        let mut event_names = Vec::new();
        loop {
            let event = tokio::time::timeout(std::time::Duration::from_secs(2), event_rx.recv())
                .await
                .expect("timed out waiting for stream event")
                .expect("stream event channel closed");
            let terminal = event.is_terminal();
            if let TtsStreamEvent::Audio { frame } = &event {
                assert_eq!(frame.sample_rate, 48_000);
                assert_eq!(frame.frame_ms, 20);
                assert_eq!(frame.sample_count, 960);
                assert_eq!(frame.timestamp_ms, frame.sequence * 20);
            }
            event_names.push(event.event_name().to_string());
            if terminal {
                break;
            }
        }

        assert_eq!(event_names.first().map(String::as_str), Some("tts.started"));
        assert!(event_names.iter().any(|event| event == "tts.audio"));
        assert_eq!(event_names.last().map(String::as_str), Some("tts.ended"));

        let result = await_completion(completion_rx).await;
        assert_eq!(result.status, ItemStatus::Completed);
        assert_eq!(result.result.as_deref(), Some("simulated stream"));

        worker.abort();
    }

    fn unique_suffix() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }
}
