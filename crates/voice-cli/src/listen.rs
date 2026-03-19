//! Microphone recording and speech-to-text transcription.
//!
//! Debug: set `VOICE_SAVE_RECORDING=1` to save mic audio to `/tmp/voice_recording.wav`.
//!
//! Leading silence is automatically trimmed to handle Bluetooth mic latency
//! (AirPods can take ~0.5-1s before the mic starts capturing real audio).
//!
//! Captures audio from the default input device, then runs Moonshine STT
//! to produce a transcription. Two modes:
//!
//! - **Single-shot** (`voice listen`): Records until Enter or Ctrl+C,
//!   transcribes, prints the result.
//! - **Continuous** (`voice listen --continuous`): Records in segments
//!   separated by pauses, transcribes each segment as it completes.

use std::io::{self, Write};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;

use crate::{INTERRUPTED, QUIET};

/// Record audio from the default input device until interrupted.
///
/// Returns mono f32 samples at the device's native sample rate, plus the rate.
/// Recording stops when `INTERRUPTED` is set (Ctrl+C) or when `stop` is called.
pub fn record_until_interrupt() -> Result<(Vec<f32>, u32), String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

    let config = device
        .default_input_config()
        .map_err(|e| format!("Failed to get input config: {e}"))?;

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    if !QUIET.load(Ordering::Relaxed) {
        let name = device.name().unwrap_or_else(|_| "unknown".to_string());
        eprintln!("Recording from: {name} ({sample_rate}Hz, {channels}ch)");
        eprintln!("Press Enter or Ctrl+C to stop recording...");
    }

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let buffer_clone = Arc::clone(&buffer);

    let err_flag: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let err_clone = Arc::clone(&err_flag);

    let stream = match config.sample_format() {
        SampleFormat::F32 => {
            let buf = Arc::clone(&buffer_clone);
            device
                .build_input_stream(
                    &config.into(),
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        // Mix to mono on the fly
                        let mut guard = buf.lock().unwrap();
                        if channels == 1 {
                            guard.extend_from_slice(data);
                        } else {
                            for chunk in data.chunks(channels) {
                                let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                                guard.push(mono);
                            }
                        }
                    },
                    move |err| {
                        let mut guard = err_clone.lock().unwrap();
                        *guard = Some(format!("Audio input error: {err}"));
                    },
                    None,
                )
                .map_err(|e| format!("Failed to build input stream: {e}"))?
        }
        SampleFormat::I16 => {
            let buf = Arc::clone(&buffer_clone);
            device
                .build_input_stream(
                    &config.into(),
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        let mut guard = buf.lock().unwrap();
                        if channels == 1 {
                            for &s in data {
                                guard.push(s as f32 / 32768.0);
                            }
                        } else {
                            for chunk in data.chunks(channels) {
                                let mono: f32 =
                                    chunk.iter().map(|&s| s as f32 / 32768.0).sum::<f32>()
                                        / channels as f32;
                                guard.push(mono);
                            }
                        }
                    },
                    move |err| {
                        let mut guard = err_flag.lock().unwrap();
                        *guard = Some(format!("Audio input error: {err}"));
                    },
                    None,
                )
                .map_err(|e| format!("Failed to build input stream: {e}"))?
        }
        format => {
            return Err(format!("Unsupported sample format: {format:?}"));
        }
    };

    stream
        .play()
        .map_err(|e| format!("Failed to start recording: {e}"))?;

    // Wait for Enter key or Ctrl+C on a separate thread so we don't block
    // the audio callback. The INTERRUPTED flag is set by the global Ctrl+C
    // handler in main.rs.
    let enter_pressed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let enter_clone = Arc::clone(&enter_pressed);

    let stdin_thread = std::thread::spawn(move || {
        let mut line = String::new();
        let _ = io::stdin().read_line(&mut line);
        enter_clone.store(true, Ordering::SeqCst);
    });

    // Poll until interrupted or Enter pressed
    loop {
        if INTERRUPTED.load(Ordering::Relaxed) || enter_pressed.load(Ordering::Relaxed) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // Stop recording
    drop(stream);

    // Don't wait for stdin thread — it may be blocked on read_line.
    // Just detach it; it'll exit when the process does.
    drop(stdin_thread);

    let samples = match Arc::try_unwrap(buffer) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(arc) => arc.lock().unwrap().clone(),
    };

    let duration = samples.len() as f32 / sample_rate as f32;
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!(
            "Recorded {:.1}s ({} samples at {}Hz)",
            duration,
            samples.len(),
            sample_rate
        );

        // Audio level diagnostics
        if !samples.is_empty() {
            let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
            let nonzero = samples.iter().filter(|s| s.abs() > 1e-8).count();
            eprintln!(
                "Audio levels: peak={:.6}, rms={:.6}, nonzero={}/{}",
                max_abs,
                rms,
                nonzero,
                samples.len()
            );
            if max_abs < 0.001 {
                eprintln!(
                    "⚠️  Audio is nearly silent (peak={:.6}). Check your mic input level.",
                    max_abs
                );
            }
        }
    }

    // Save recording to WAV for debugging if requested
    if std::env::var("VOICE_SAVE_RECORDING").is_ok() {
        let debug_path = "/tmp/voice_recording.wav";
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        if let Ok(mut writer) = hound::WavWriter::create(debug_path, spec) {
            for &s in &samples {
                let _ = writer.write_sample(s);
            }
            let _ = writer.finalize();
            eprintln!("Saved recording to {debug_path}");
        }
    }

    Ok((samples, sample_rate))
}

/// Trim leading and trailing silence from audio samples.
///
/// Bluetooth microphones (e.g. AirPods) can take ~0.5-1s before audio
/// actually flows, producing a block of zeros at the start. Moonshine
/// is sensitive to the silence-to-speech ratio, especially on short
/// recordings — trimming silence dramatically improves accuracy.
///
/// Uses a simple energy threshold: finds the first and last sample
/// whose absolute value exceeds `threshold`, then keeps a small
/// `padding` of extra samples on each side for natural attack/release.
fn trim_silence(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    // Threshold: samples below this absolute value are considered silence.
    // -40 dBFS ≈ 0.01 amplitude — conservative enough to keep quiet speech.
    let threshold: f32 = 0.01;

    // Padding: keep 100ms of context around detected speech edges.
    let padding = (sample_rate as usize) / 10;

    let first_voice = samples.iter().position(|s| s.abs() > threshold);
    let last_voice = samples.iter().rposition(|s| s.abs() > threshold);

    match (first_voice, last_voice) {
        (Some(start), Some(end)) => {
            let start = start.saturating_sub(padding);
            let end = (end + padding).min(samples.len());
            samples[start..end].to_vec()
        }
        _ => {
            // All silence — return empty
            Vec::new()
        }
    }
}

/// Default STT model. Override with `STT_MODEL` env var.
/// moonshine-base (61M) is noticeably better than tiny (27M) for real mic audio.
const DEFAULT_STT_MODEL: &str = "UsefulSensors/moonshine-base";

fn stt_model_repo() -> String {
    std::env::var("STT_MODEL").unwrap_or_else(|_| DEFAULT_STT_MODEL.to_string())
}

/// Record from mic, transcribe with Moonshine, and print the result.
///
/// This is the main entry point for `voice listen`.
///
/// Uses moonshine-base by default. Override with `STT_MODEL` env var:
///   STT_MODEL=UsefulSensors/moonshine-tiny voice listen
pub fn listen_and_transcribe() {
    let repo = stt_model_repo();

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Loading speech-to-text model ({repo})...");
    }

    let mut model = match voice_stt::load_model(&repo) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load STT model: {e}");
            eprintln!("Model weights will be downloaded from HuggingFace on first run.");
            std::process::exit(1);
        }
    };

    let tokenizer = match voice_stt::load_tokenizer(&repo) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {e}");
            std::process::exit(1);
        }
    };

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Model loaded. Ready to listen.\n");
    }

    // Record
    let (samples, sample_rate) = match record_until_interrupt() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Recording failed: {e}");
            std::process::exit(1);
        }
    };

    if samples.is_empty() {
        eprintln!("No audio recorded.");
        return;
    }

    // Trim leading/trailing silence (Bluetooth mic latency, pauses)
    let original_len = samples.len();
    let samples = trim_silence(&samples, sample_rate);
    let trimmed_duration = samples.len() as f32 / sample_rate as f32;

    if !QUIET.load(Ordering::Relaxed) {
        let original_duration = original_len as f32 / sample_rate as f32;
        if samples.len() < original_len {
            eprintln!(
                "Trimmed silence: {:.1}s → {:.1}s of speech",
                original_duration, trimmed_duration
            );
        }
    }

    if samples.is_empty() {
        eprintln!("No speech detected in recording.");
        return;
    }

    // Transcribe
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Transcribing {:.1}s of audio...", trimmed_duration);
    }

    let result = match voice_stt::transcribe_audio_with_tokenizer(
        &mut model,
        &samples,
        sample_rate,
        &tokenizer,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Transcription failed: {e}");
            std::process::exit(1);
        }
    };

    // Reset interrupt so the process exits cleanly
    INTERRUPTED.store(false, Ordering::Relaxed);

    // Print the transcription to stdout
    println!("{}", result.text);

    if !QUIET.load(Ordering::Relaxed) {
        let _ = io::stderr().flush();
        eprintln!("\n({} tokens)", result.tokens.len());
    }
}

/// Transcribe a WAV file and print the result.
///
/// Entry point for `voice --transcribe <file>`.
pub fn transcribe_file(path: &str) {
    let repo = stt_model_repo();

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Loading speech-to-text model ({repo})...");
    }

    let mut model = match voice_stt::load_model(&repo) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load STT model: {e}");
            std::process::exit(1);
        }
    };

    let tokenizer = match voice_stt::load_tokenizer(&repo) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {e}");
            std::process::exit(1);
        }
    };

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Model loaded. Transcribing: {path}");
    }

    // Load WAV
    let reader = match hound::WavReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open {path}: {e}");
            std::process::exit(1);
        }
    };

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
    };

    // Mix to mono
    let mono: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    let duration = mono.len() as f32 / sample_rate as f32;
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!(
            "Audio: {:.1}s ({} samples at {}Hz)",
            duration,
            mono.len(),
            sample_rate
        );
    }

    let result = match voice_stt::transcribe_audio_with_tokenizer(
        &mut model,
        &mono,
        sample_rate,
        &tokenizer,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Transcription failed: {e}");
            std::process::exit(1);
        }
    };

    println!("{}", result.text);

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("\n({} tokens)", result.tokens.len());
    }
}
