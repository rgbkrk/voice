//! Audio chat with a local LLM via Ollama.
//!
//! Loops: mic → Whisper STT → Ollama → streaming TTS → speaker → repeat.
//!
//! Usage:
//!     voice-chat
//!     voice-chat --model llama3.2
//!     voice-chat --voice af_nicole --speed 1.1

use std::io::Write;
use std::num::NonZero;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use clap::Parser;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::ChatMessage;
use ollama_rs::Ollama;
use pulldown_cmark::{Event, Options, Parser as MdParser, Tag, TagEnd};
use rodio::buffer::SamplesBuffer;
use rodio::microphone::MicrophoneBuilder;
use rodio::{DeviceSinkBuilder, Player};
use tokio_stream::StreamExt;

const MODEL_REPO: &str = "prince-canuma/Kokoro-82M";
const STT_REPO: &str = "distil-whisper/distil-large-v3";

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

#[derive(Parser)]
#[command(about = "Audio chat with a local LLM via Ollama")]
struct Args {
    /// Ollama model name
    #[arg(short, long, default_value = "qwen3")]
    model: String,

    /// TTS voice name
    #[arg(short, long, default_value = "af_heart")]
    voice: String,

    /// TTS speed
    #[arg(short, long, default_value = "1.0")]
    speed: f32,

    /// Ollama host
    #[arg(long, default_value = "http://localhost")]
    host: String,

    /// Ollama port
    #[arg(long, default_value = "11434")]
    port: u16,

    /// System prompt
    #[arg(long)]
    system: Option<String>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    ctrlc::set_handler(|| INTERRUPTED.store(true, Ordering::SeqCst))
        .expect("Failed to set Ctrl+C handler");

    eprintln!("Loading TTS model...");
    let mut tts_model = voice_tts::load_model(MODEL_REPO).expect("Failed to load TTS model");
    let sample_rate = tts_model.sample_rate;
    let voice = tts_model
        .load_voice(&args.voice, Some(MODEL_REPO))
        .expect("Failed to load voice");

    eprintln!("Loading STT model...");
    let mut stt_model = voice_stt::load_model(STT_REPO).expect("Failed to load STT model");

    let ollama = Ollama::new(args.host, args.port);
    let history = Arc::new(Mutex::new(Vec::<ChatMessage>::new()));

    // Add system prompt if provided
    if let Some(system) = &args.system {
        history
            .lock()
            .unwrap()
            .push(ChatMessage::system(system.clone()));
    }

    eprintln!(
        "\nReady — model: {}, voice: {}, speed: {}",
        args.model, args.voice, args.speed
    );
    eprintln!("Ctrl+C to quit.\n");

    // Open mic once at startup — stays open for the whole session.
    // The Bluetooth HFP codec switch happens now, not during conversation.
    eprintln!("Opening mic...");
    let mic = PersistentMic::open().expect("Failed to open mic");

    // Kick off the conversation with a greeting
    speak(
        &mut tts_model,
        &voice,
        "Good evening.",
        args.speed,
        sample_rate,
    );

    // Conversation loop
    loop {
        if INTERRUPTED.load(Ordering::Relaxed) {
            break;
        }

        // ── Listen ──────────────────────────────────────────────────
        let user_text = match record_and_transcribe(&mic, &mut stt_model) {
            Some(text) if !text.trim().is_empty() => text.trim().to_string(),
            _ => {
                if INTERRUPTED.load(Ordering::Relaxed) {
                    break;
                }
                eprintln!("(no speech detected, listening again...)");
                continue;
            }
        };

        eprintln!("\x1b[33mYou:\x1b[0m {user_text}");

        // Check for exit
        let lower = user_text.to_lowercase();
        if ["goodbye", "bye", "quit", "exit", "stop"]
            .iter()
            .any(|w| lower.contains(w))
        {
            speak(&mut tts_model, &voice, "Goodbye!", args.speed, sample_rate);
            break;
        }

        // ── Chat with Ollama (streaming) ────────────────────────────
        let request = ChatMessageRequest::new(
            args.model.clone(),
            vec![ChatMessage::user(user_text.clone())],
        );

        let stream_result = ollama
            .send_chat_messages_with_history_stream(history.clone(), request)
            .await;

        let mut stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Ollama error: {e}");
                speak(
                    &mut tts_model,
                    &voice,
                    "Sorry, I couldn't reach the model.",
                    args.speed,
                    sample_rate,
                );
                continue;
            }
        };

        // Accumulate tokens into sentences, speak each sentence as it completes
        let mut sentence_buf = String::new();
        let mut full_response = String::new();

        // Open audio output once for the whole response
        let Ok(mut audio_stream) = DeviceSinkBuilder::open_default_sink() else {
            eprintln!("Failed to open audio output");
            continue;
        };
        audio_stream.log_on_drop(false);
        let player = Player::connect_new(audio_stream.mixer());
        let channels = NonZero::new(1u16).unwrap();
        let rate = NonZero::new(sample_rate).unwrap();

        eprint!("\x1b[36mAssistant:\x1b[0m ");

        while let Some(res) = stream.next().await {
            if INTERRUPTED.load(Ordering::Relaxed) {
                break;
            }

            let Ok(res) = res else { continue };

            // Skip the final chunk — it has done=true and content is empty or
            // duplicated. The response was already built up incrementally.
            if res.done {
                break;
            }

            let token = &res.message.content;
            full_response.push_str(token);
            sentence_buf.push_str(token);
            eprint!("{token}");
            let _ = std::io::stderr().flush();

            // Check if we have a sentence boundary
            if has_sentence_end(&sentence_buf) {
                let text = prepare_for_tts(&sentence_buf);
                if !text.trim().is_empty() {
                    generate_and_queue(
                        &mut tts_model,
                        &voice,
                        &text,
                        args.speed,
                        &player,
                        channels,
                        rate,
                    );
                }
                sentence_buf.clear();
            }
        }

        // Speak any remaining text
        if !sentence_buf.trim().is_empty() {
            let text = prepare_for_tts(&sentence_buf);
            if !text.trim().is_empty() {
                generate_and_queue(
                    &mut tts_model,
                    &voice,
                    &text,
                    args.speed,
                    &player,
                    channels,
                    rate,
                );
            }
        }

        eprintln!();

        // Wait for playback to finish
        while !player.empty() {
            if INTERRUPTED.load(Ordering::Relaxed) {
                player.stop();
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    eprintln!("\nBye!");
}

// ── TTS helpers ──────────────────────────────────────────────────────

fn generate_and_queue(
    model: &mut voice_tts::KokoroModel,
    voice: &candle_core::Tensor,
    text: &str,
    speed: f32,
    player: &Player,
    channels: NonZero<u16>,
    rate: NonZero<u32>,
) {
    let chunks = match voice_g2p::text_to_phoneme_chunks(text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\n  G2P error: {e}");
            return;
        }
    };

    for phonemes in &chunks {
        if phonemes.is_empty() {
            continue;
        }
        match voice_tts::generate(model, phonemes, voice, speed) {
            Ok(audio) => {
                player.append(SamplesBuffer::new(channels, rate, audio));
            }
            Err(e) => {
                eprintln!("\n  TTS error: {e}");
            }
        }
    }
}

fn speak(
    model: &mut voice_tts::KokoroModel,
    voice: &candle_core::Tensor,
    text: &str,
    speed: f32,
    sample_rate: u32,
) {
    let Ok(mut stream) = DeviceSinkBuilder::open_default_sink() else {
        return;
    };
    stream.log_on_drop(false);
    let player = Player::connect_new(stream.mixer());
    let channels = NonZero::new(1u16).unwrap();
    let rate = NonZero::new(sample_rate).unwrap();

    generate_and_queue(model, voice, text, speed, &player, channels, rate);

    while !player.empty() {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

// ── Persistent mic ──────────────────────────────────────────────────

/// Mic that stays open for the whole session. Avoids repeated Bluetooth
/// HFP codec switches — the switch happens once at startup.
struct PersistentMic {
    sample_rate: u32,
    buffer: Arc<Mutex<Vec<f32>>>,
    peak: Arc<std::sync::atomic::AtomicU32>,
    _stop: Arc<AtomicBool>,
}

impl PersistentMic {
    fn open() -> Result<Self, String> {
        let mic = MicrophoneBuilder::new()
            .default_device()
            .map_err(|e| format!("No input device: {e}"))?
            .default_config()
            .map_err(|e| format!("No input config: {e}"))?
            .open_stream()
            .map_err(|e| format!("Failed to open mic: {e}"))?;

        let sample_rate = mic.config().sample_rate.get();
        let channels = mic.config().channel_count.get().max(1) as usize;

        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let peak = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        let buf_clone = Arc::clone(&buffer);
        let peak_clone = Arc::clone(&peak);
        let stop_clone = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut chunk_peak: f32 = 0.0;
            let mut count = 0usize;
            for sample in mic {
                if stop_clone.load(Ordering::Relaxed) {
                    break;
                }
                chunk_peak = chunk_peak.max(sample.abs());
                count += 1;
                if channels == 1 {
                    buf_clone.lock().unwrap().push(sample);
                } else if count % channels == 0 {
                    buf_clone.lock().unwrap().push(sample);
                }
                if count % 100 == 0 {
                    peak_clone.store(chunk_peak.to_bits(), Ordering::Relaxed);
                    chunk_peak = 0.0;
                }
            }
        });

        Ok(Self {
            sample_rate,
            buffer,
            peak,
            _stop: stop,
        })
    }

    /// Clear the buffer and record with VAD. Mic stays open after.
    fn record_vad(&self) -> Option<(Vec<f32>, u32)> {
        play_ding();
        eprintln!("Listening...");

        // Clear old audio
        self.buffer.lock().unwrap().clear();

        // Calibrate noise floor (300ms)
        let cal_start = std::time::Instant::now();
        let mut max_ambient: f32 = 0.0;
        while cal_start.elapsed() < std::time::Duration::from_millis(300) {
            if INTERRUPTED.load(Ordering::Relaxed) {
                return None;
            }
            let p = f32::from_bits(self.peak.load(Ordering::Relaxed));
            max_ambient = max_ambient.max(p);
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let threshold = (max_ambient * 3.0).max(0.01);

        // VAD
        let mut speech_started = false;
        let mut silence_start: Option<std::time::Instant> = None;
        let start = std::time::Instant::now();

        loop {
            if INTERRUPTED.load(Ordering::Relaxed)
                || start.elapsed() > std::time::Duration::from_secs(60)
            {
                break;
            }
            let p = f32::from_bits(self.peak.load(Ordering::Relaxed));
            if p > threshold {
                if !speech_started {
                    speech_started = true;
                    eprintln!("Speech detected...");
                }
                silence_start = None;
            } else if speech_started {
                let ss = silence_start.get_or_insert_with(std::time::Instant::now);
                if ss.elapsed() > std::time::Duration::from_millis(1500) {
                    eprintln!("Silence detected.");
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        // Snapshot the buffer — don't stop the mic
        let samples = self.buffer.lock().unwrap().clone();
        Some((samples, self.sample_rate))
    }
}

// ── STT / Recording ─────────────────────────────────────────────────

fn record_and_transcribe(
    mic: &PersistentMic,
    stt_model: &mut voice_stt::WhisperModel,
) -> Option<String> {
    let (samples, sample_rate) = mic.record_vad()?;

    if samples.is_empty() {
        return None;
    }

    let trimmed = trim_leading_silence(&samples, sample_rate);
    if trimmed.is_empty() {
        return None;
    }

    let duration = trimmed.len() as f32 / sample_rate as f32;
    eprintln!("Transcribing {duration:.1}s...");

    match voice_stt::transcribe_audio(stt_model, &trimmed, sample_rate) {
        Ok(r) => Some(r.text),
        Err(e) => {
            eprintln!("Transcription error: {e}");
            None
        }
    }
}

fn play_ding() {
    let sample_rate = 44100u32;
    let num_samples = sample_rate as usize * 120 / 1000;

    let mut samples = Vec::with_capacity((num_samples + 882) * 2);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let envelope = (-t / 0.03f32).exp() * 0.20;
        let s = (2.0 * std::f32::consts::PI * 880.0 * t).sin() * envelope;
        samples.push(s);
        samples.push(s);
    }
    samples.extend(std::iter::repeat_n(0.0f32, 882 * 2));

    let Ok(mut stream) = DeviceSinkBuilder::open_default_sink() else {
        return;
    };
    stream.log_on_drop(false);
    let player = Player::connect_new(stream.mixer());
    player.append(SamplesBuffer::new(
        NonZero::new(2u16).unwrap(),
        NonZero::new(sample_rate).unwrap(),
        samples,
    ));
    while !player.empty() {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

fn trim_leading_silence(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let window = (sample_rate as usize) / 100; // 10ms
    let threshold: f32 = 0.005;
    let leading_context = (sample_rate as usize) / 2; // 500ms

    // Find first window above threshold
    let speech_start = samples
        .windows(window)
        .position(|w| {
            let rms = (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt();
            rms > threshold
        })
        .unwrap_or(0);

    let start = speech_start.saturating_sub(leading_context);
    samples[start..].to_vec()
}

// ── Text processing ─────────────────────────────────────────────────

fn has_sentence_end(text: &str) -> bool {
    let trimmed = text.trim_end();
    trimmed.ends_with('.')
        || trimmed.ends_with('!')
        || trimmed.ends_with('?')
        || trimmed.ends_with('\n')
}

fn prepare_for_tts(text: &str) -> String {
    let stripped = strip_markdown(text);
    apply_tech_subs(&stripped)
}

fn strip_markdown(text: &str) -> String {
    let opts = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES;
    let parser = MdParser::new_ext(text, opts);

    let mut out = String::new();
    let mut skip_depth: usize = 0;

    for event in parser {
        match event {
            Event::Start(Tag::CodeBlock(_)) | Event::Start(Tag::Image { .. }) => {
                skip_depth += 1;
            }
            Event::End(TagEnd::CodeBlock) | Event::End(TagEnd::Image) => {
                skip_depth = skip_depth.saturating_sub(1);
            }
            _ if skip_depth > 0 => {}
            Event::Text(t) => out.push_str(&t),
            Event::SoftBreak => out.push(' '),
            Event::HardBreak | Event::End(TagEnd::Paragraph) | Event::End(TagEnd::Heading(_)) => {
                out.push(' ');
            }
            Event::Code(t) => out.push_str(&t),
            _ => {}
        }
    }

    out
}

const TECH_SUBS: &[(&str, &str)] = &[
    ("JSON", "jay-sahn"),
    ("json", "jay-sahn"),
    ("YAML", "yam-ul"),
    ("yaml", "yam-ul"),
    ("TOML", "tom-ul"),
    ("toml", "tom-ul"),
    ("WASM", "waz-um"),
    ("wasm", "waz-um"),
    ("OAuth", "oh-auth"),
    ("NGINX", "engine-X"),
    ("nginx", "engine-X"),
    ("PostgreSQL", "post-gres-Q-L"),
    ("SQLite", "S-Q-lite"),
    ("macOS", "mac O S"),
    ("iOS", "eye-O-S"),
];

fn apply_tech_subs(text: &str) -> String {
    let mut result = text.to_string();
    for (from, to) in TECH_SUBS {
        result = result.replace(from, to);
    }
    result
}
