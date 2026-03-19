//! Microphone recording and speech-to-text transcription.
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
    }

    Ok((samples, sample_rate))
}

/// Record from mic, transcribe with Moonshine, and print the result.
///
/// This is the main entry point for `voice listen`.
pub fn listen_and_transcribe() {
    // Load the STT model
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Loading speech-to-text model...");
    }

    let mut model = match voice_stt::load_model("UsefulSensors/moonshine-tiny") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load STT model: {e}");
            eprintln!("Model weights (~108MB) will be downloaded from HuggingFace on first run.");
            std::process::exit(1);
        }
    };

    let tokenizer = match voice_stt::load_tokenizer("UsefulSensors/moonshine-tiny") {
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

    // Transcribe
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Transcribing...");
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
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Loading speech-to-text model...");
    }

    let mut model = match voice_stt::load_model("UsefulSensors/moonshine-tiny") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load STT model: {e}");
            std::process::exit(1);
        }
    };

    let tokenizer = match voice_stt::load_tokenizer("UsefulSensors/moonshine-tiny") {
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
