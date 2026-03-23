//! Microphone recording and speech-to-text transcription.
//!
//! Debug: set `VOICE_SAVE_RECORDING=1` to save mic audio to `/tmp/voice_recording_<timestamp>.wav`.
//!
//! Leading silence is automatically trimmed to handle Bluetooth mic latency
//! (AirPods can take ~0.5-1s before the mic starts capturing real audio).
//!
//! A pleasant ding sound plays when the mic is ready to record, so you know
//! when to start speaking (especially helpful with Bluetooth mic latency).
//!
//! Three recording modes:
//!
//! - **Manual stop** (`voice listen`): Records until Enter or Ctrl+C.
//! - **VAD auto-stop** (`record_with_vad`): Records until speech is detected
//!   and then silence follows for a configurable timeout. Used by the JSON-RPC
//!   `listen` method so agents don't need to send a signal to stop recording.
//! - **Continuous** (`voice listen --continuous`): Records indefinitely,
//!   splitting on silence into segments. Each segment is transcribed as it
//!   completes while recording continues. Audio thread → segment queue →
//!   transcription thread → output.

use std::io::{self, Write};
use std::num::NonZero;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, Player};

use crate::{INTERRUPTED, QUIET};

// ── Sound configuration ───────────────────────────────────────────────

/// Cached audio samples for start/stop notification sounds.
///
/// When `None`, the default synthesized tones are used.
/// When `Some`, the provided WAV samples are played instead.
pub struct SoundConfig {
    /// Custom start-of-listening sound (replaces the default 880Hz ding).
    pub start_sound: Option<CachedSound>,
    /// Custom end-of-listening sound (replaces the default two-blip C5→E5 chime).
    pub stop_sound: Option<CachedSound>,
}

/// Pre-loaded audio samples from a WAV file, ready to play.
pub struct CachedSound {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl SoundConfig {
    pub fn new() -> Self {
        Self {
            start_sound: None,
            stop_sound: None,
        }
    }
}

impl Default for SoundConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Load a WAV file and return cached samples ready for playback.
///
/// Validates the WAV header and enforces a 30-second maximum duration
/// to prevent excessive memory use from arbitrarily large files.
pub fn load_wav_sound(path: &std::path::Path) -> Result<CachedSound, String> {
    let reader =
        hound::WavReader::open(path).map_err(|e| format!("Failed to open WAV file: {e}"))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels;

    if sample_rate == 0 {
        return Err("Invalid WAV: sample rate is 0".to_string());
    }
    if channels == 0 {
        return Err("Invalid WAV: channel count is 0".to_string());
    }

    // Cap at 30 seconds to prevent OOM from huge files
    let max_samples = sample_rate as usize * channels as usize * 30;
    let duration = reader.duration() as usize * channels as usize;
    if duration > max_samples {
        return Err(format!(
            "WAV file too long ({:.1}s); maximum is 30s for notification sounds",
            duration as f64 / (sample_rate as f64 * channels as f64)
        ));
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to read WAV samples: {e}"))?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            if bits == 0 || bits > 32 {
                return Err(format!(
                    "Unsupported bits_per_sample {bits}; expected 1..=32"
                ));
            }
            let max_val = (1u32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Failed to read WAV samples: {e}"))?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    Ok(CachedSound {
        samples,
        sample_rate,
        channels,
    })
}

/// Global sound config, accessible from recording functions.
static SOUND_CONFIG: std::sync::OnceLock<Mutex<SoundConfig>> = std::sync::OnceLock::new();

fn sound_config() -> &'static Mutex<SoundConfig> {
    SOUND_CONFIG.get_or_init(|| Mutex::new(SoundConfig::new()))
}

/// Set a custom start-of-listening sound from a WAV file.
pub fn set_start_sound(sound: Option<CachedSound>) {
    sound_config().lock().unwrap().start_sound = sound;
}

/// Set a custom end-of-listening sound from a WAV file.
pub fn set_stop_sound(sound: Option<CachedSound>) {
    sound_config().lock().unwrap().stop_sound = sound;
}

/// Play a cached sound immediately (for previewing sounds).
///
/// Returns an error if the audio output device cannot be opened.
pub fn play_cached_sound(sound: &CachedSound) -> Result<(), String> {
    let mut stream = DeviceSinkBuilder::open_default_sink()
        .map_err(|e| format!("Failed to open audio output: {e}"))?;
    stream.log_on_drop(false);
    let player = Player::connect_new(stream.mixer());

    let Some(channels) = NonZero::new(sound.channels) else {
        return Err("Invalid sound: zero channels".to_string());
    };
    let Some(rate) = NonZero::new(sound.sample_rate) else {
        return Err("Invalid sound: zero sample rate".to_string());
    };
    let source = SamplesBuffer::new(channels, rate, sound.samples.clone());
    player.append(source);

    while !player.empty() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    Ok(())
}

// ── Segment types ──────────────────────────────────────────────────────

/// A chunk of recorded audio delimited by silence boundaries.
pub struct Segment {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub segment_id: u64,
    /// Milliseconds since recording started.
    pub timestamp_ms: u64,
}

/// Result of transcribing a single segment.
#[allow(dead_code)]
pub struct SegmentResult {
    pub text: String,
    pub tokens: Vec<u32>,
    pub segment_id: u64,
    pub timestamp_ms: u64,
    /// How long inference took in milliseconds.
    pub transcribe_ms: u64,
}

// ── Ding sound ─────────────────────────────────────────────────────────

/// Play a notification sound using cached samples or synthesized defaults.
fn play_cached_or_synth(
    cached: Option<&CachedSound>,
    default_freq: f32,
    default_duration_ms: usize,
    default_decay: f32,
    default_volume: f32,
    post_delay_ms: u64,
) {
    let Ok(mut stream) = DeviceSinkBuilder::open_default_sink() else {
        return; // Silent failure — sounds are optional
    };
    stream.log_on_drop(false);
    let player = Player::connect_new(stream.mixer());

    if let Some(sound) = cached {
        let Some(channels) = NonZero::new(sound.channels) else {
            return;
        };
        let Some(rate) = NonZero::new(sound.sample_rate) else {
            return;
        };
        let source = SamplesBuffer::new(channels, rate, sound.samples.clone());
        player.append(source);
    } else {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * default_duration_ms / 1000;

        let mut samples = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let envelope = (-t / default_decay).exp() * default_volume;
            let sample = (2.0 * std::f32::consts::PI * default_freq * t).sin() * envelope;
            samples.push(sample);
        }

        let channels = NonZero::new(1u16).unwrap();
        let rate = NonZero::new(sample_rate).unwrap();
        let source = SamplesBuffer::new(channels, rate, samples);
        player.append(source);
    }

    while !player.empty() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    if post_delay_ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(post_delay_ms));
    }
}

/// Play a short pleasant ding to signal that recording has started.
///
/// Uses custom start sound if configured, otherwise synthesizes an
/// 880Hz (A5) sine tone with a gentle exponential decay over ~200ms.
fn play_ding() {
    // Clone samples under lock, then play without holding it
    let custom = sound_config()
        .lock()
        .unwrap()
        .start_sound
        .as_ref()
        .map(|s| CachedSound {
            samples: s.samples.clone(),
            sample_rate: s.sample_rate,
            channels: s.channels,
        });
    play_cached_or_synth(custom.as_ref(), 880.0, 200, 0.06, 0.15, 50);
}

/// Play two ascending blips to signal that listening has stopped.
///
/// Uses custom stop sound if configured, otherwise synthesizes two quick
/// sine blips — C5 (523Hz) then E5 (659Hz) — each with a smooth
/// sine-shaped envelope and a short gap between them. Feels like a
/// positive "got it" confirmation.
fn play_dong() {
    let custom = sound_config()
        .lock()
        .unwrap()
        .stop_sound
        .as_ref()
        .map(|s| CachedSound {
            samples: s.samples.clone(),
            sample_rate: s.sample_rate,
            channels: s.channels,
        });

    if custom.is_some() {
        play_cached_or_synth(custom.as_ref(), 0.0, 0, 0.0, 0.0, 0);
        return;
    }

    let Ok(mut stream) = DeviceSinkBuilder::open_default_sink() else {
        return;
    };
    stream.log_on_drop(false);
    let player = Player::connect_new(stream.mixer());

    let sample_rate = 44100u32;
    let pi2 = 2.0 * std::f32::consts::PI;
    let volume = 0.10f32;

    // Two blips: C5 (120ms) → gap (60ms) → E5 (120ms)
    let blip_samples = sample_rate as usize * 120 / 1000;
    let gap_samples = sample_rate as usize * 60 / 1000;

    let mut samples = Vec::with_capacity(blip_samples * 2 + gap_samples);

    // First blip: C5
    for i in 0..blip_samples {
        let t = i as f32 / sample_rate as f32;
        let envelope = (std::f32::consts::PI * i as f32 / blip_samples as f32).sin() * volume;
        samples.push((pi2 * 523.25 * t).sin() * envelope);
    }

    // Gap
    samples.extend(std::iter::repeat_n(0.0f32, gap_samples));

    // Second blip: E5
    for i in 0..blip_samples {
        let t = i as f32 / sample_rate as f32;
        let envelope = (std::f32::consts::PI * i as f32 / blip_samples as f32).sin() * volume;
        samples.push((pi2 * 659.26 * t).sin() * envelope);
    }

    let channels = NonZero::new(1u16).unwrap();
    let rate = NonZero::new(sample_rate).unwrap();
    let source = SamplesBuffer::new(channels, rate, samples);
    player.append(source);

    while !player.empty() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

// ── Mic input helpers ──────────────────────────────────────────────────

/// Open the default input device and return its config.
fn open_input_device() -> Result<(cpal::Device, cpal::SupportedStreamConfig), String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;
    let config = device
        .default_input_config()
        .map_err(|e| format!("Failed to get input config: {e}"))?;
    Ok((device, config))
}

/// Build an input stream that writes mono f32 samples into `buffer`.
///
/// If `peak_out` is `Some`, each callback also writes the chunk's peak
/// amplitude (as `f32::to_bits()`) into the atomic for VAD polling.
fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    buffer: Arc<Mutex<Vec<f32>>>,
    peak_out: Option<Arc<std::sync::atomic::AtomicU32>>,
) -> Result<cpal::Stream, String> {
    let channels = config.channels() as usize;

    match config.sample_format() {
        SampleFormat::F32 => {
            let buf = Arc::clone(&buffer);
            let peak = peak_out.clone();
            device
                .build_input_stream(
                    &config.clone().into(),
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        let mut guard = buf.lock().unwrap();
                        let mut chunk_peak: f32 = 0.0;
                        if channels == 1 {
                            for &s in data {
                                chunk_peak = chunk_peak.max(s.abs());
                                guard.push(s);
                            }
                        } else {
                            for chunk in data.chunks(channels) {
                                let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                                chunk_peak = chunk_peak.max(mono.abs());
                                guard.push(mono);
                            }
                        }
                        if let Some(ref p) = peak {
                            p.store(chunk_peak.to_bits(), Ordering::Relaxed);
                        }
                    },
                    |err| eprintln!("Audio input error: {err}"),
                    None,
                )
                .map_err(|e| format!("Failed to build input stream: {e}"))
        }
        SampleFormat::I16 => {
            let buf = Arc::clone(&buffer);
            let peak = peak_out.clone();
            device
                .build_input_stream(
                    &config.clone().into(),
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        let mut guard = buf.lock().unwrap();
                        let mut chunk_peak: f32 = 0.0;
                        if channels == 1 {
                            for &s in data {
                                let f = s as f32 / 32768.0;
                                chunk_peak = chunk_peak.max(f.abs());
                                guard.push(f);
                            }
                        } else {
                            for chunk in data.chunks(channels) {
                                let mono: f32 =
                                    chunk.iter().map(|&s| s as f32 / 32768.0).sum::<f32>()
                                        / channels as f32;
                                chunk_peak = chunk_peak.max(mono.abs());
                                guard.push(mono);
                            }
                        }
                        if let Some(ref p) = peak {
                            p.store(chunk_peak.to_bits(), Ordering::Relaxed);
                        }
                    },
                    |err| eprintln!("Audio input error: {err}"),
                    None,
                )
                .map_err(|e| format!("Failed to build input stream: {e}"))
        }
        format => Err(format!("Unsupported sample format: {format:?}")),
    }
}

/// Extract final samples from the shared buffer.
fn extract_samples(buffer: Arc<Mutex<Vec<f32>>>) -> Vec<f32> {
    match Arc::try_unwrap(buffer) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(arc) => arc.lock().unwrap().clone(),
    }
}

/// Log recording stats to stderr (unless quiet).
fn log_recording_stats(samples: &[f32], sample_rate: u32) {
    if QUIET.load(Ordering::Relaxed) || samples.is_empty() {
        return;
    }

    let duration = samples.len() as f32 / sample_rate as f32;
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    let nonzero = samples.iter().filter(|s| s.abs() > 1e-8).count();

    eprintln!(
        "Recorded {:.1}s ({} samples at {}Hz)",
        duration,
        samples.len(),
        sample_rate
    );
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

// ── Recording modes ────────────────────────────────────────────────────

/// Record audio from the default input device until Enter or Ctrl+C.
///
/// Returns mono f32 samples at the device's native sample rate, plus the rate.
pub fn record_until_interrupt() -> Result<(Vec<f32>, u32), String> {
    let (device, config) = open_input_device()?;
    let sample_rate = config.sample_rate().0;

    if !QUIET.load(Ordering::Relaxed) {
        let name = device.name().unwrap_or_else(|_| "unknown".to_string());
        eprintln!(
            "Recording from: {name} ({sample_rate}Hz, {}ch)",
            config.channels()
        );
    }

    // Start mic first so Bluetooth hardware warms up
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let stream = build_input_stream(&device, &config, Arc::clone(&buffer), None)?;

    stream
        .play()
        .map_err(|e| format!("Failed to start recording: {e}"))?;

    // Brief warmup for Bluetooth mics before the ding
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Ding signals "ready" — discard warmup audio after
    play_ding();
    {
        let mut guard = buffer.lock().unwrap();
        guard.clear();
    }

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Press Enter or Ctrl+C to stop recording...");
    }

    // Wait for Enter key or Ctrl+C
    let enter_pressed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let enter_clone = Arc::clone(&enter_pressed);

    let stdin_thread = std::thread::spawn(move || {
        let mut line = String::new();
        let _ = io::stdin().read_line(&mut line);
        enter_clone.store(true, Ordering::SeqCst);
    });

    loop {
        if INTERRUPTED.load(Ordering::Relaxed) || enter_pressed.load(Ordering::Relaxed) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    drop(stream);
    drop(stdin_thread);
    play_dong();

    let samples = extract_samples(buffer);
    log_recording_stats(&samples, sample_rate);
    maybe_save_recording(&samples, sample_rate);

    Ok((samples, sample_rate))
}

/// Record audio with voice activity detection (auto-stop on silence).
///
/// Plays a ding, starts recording, waits for speech to begin, then
/// auto-stops after `silence_timeout_ms` of silence following speech.
/// Also stops on `INTERRUPTED` flag (Ctrl+C / cancel) or `max_duration_ms`.
///
/// Returns mono f32 samples at the device's native sample rate, plus the rate.
pub fn record_with_vad(
    max_duration_ms: u64,
    silence_timeout_ms: u64,
    silence_threshold: f32,
    noise_multiplier: f32,
    calibration_ms: u64,
) -> Result<(Vec<f32>, u32), String> {
    let (device, config) = open_input_device()?;
    let sample_rate = config.sample_rate().0;

    if !QUIET.load(Ordering::Relaxed) {
        let name = device.name().unwrap_or_else(|_| "unknown".to_string());
        eprintln!(
            "Recording from: {name} ({sample_rate}Hz, {}ch)",
            config.channels()
        );
    }

    // Start the mic FIRST so Bluetooth hardware has time to spin up during
    // calibration. The ding plays AFTER calibration as the "ready" signal.
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let recent_peak: Arc<std::sync::atomic::AtomicU32> =
        Arc::new(std::sync::atomic::AtomicU32::new(0));

    let stream = build_input_stream(
        &device,
        &config,
        Arc::clone(&buffer),
        Some(Arc::clone(&recent_peak)),
    )?;

    stream
        .play()
        .map_err(|e| format!("Failed to start recording: {e}"))?;

    // Adaptive noise floor calibration — mic is already recording so
    // Bluetooth hardware warms up during this window.
    let adaptive_threshold = if calibration_ms > 0 {
        let cal_start = std::time::Instant::now();
        let calibration_duration = std::time::Duration::from_millis(calibration_ms);
        let mut max_ambient_peak: f32 = 0.0;

        while cal_start.elapsed() < calibration_duration {
            if INTERRUPTED.load(Ordering::Relaxed) {
                drop(stream);
                let samples = extract_samples(buffer);
                return Ok((samples, sample_rate));
            }
            let peak_bits = recent_peak.load(Ordering::Relaxed);
            let current_peak = f32::from_bits(peak_bits);
            max_ambient_peak = max_ambient_peak.max(current_peak);
            std::thread::sleep(std::time::Duration::from_millis(25));
        }

        let threshold = (max_ambient_peak * noise_multiplier).max(silence_threshold);

        if !QUIET.load(Ordering::Relaxed) {
            eprintln!(
                "Noise floor: {:.4}, threshold: {:.4} (×{:.1})",
                max_ambient_peak, threshold, noise_multiplier
            );
        }

        threshold
    } else {
        // Skip calibration, use raw threshold
        silence_threshold
    };

    // NOW play the ding — mic is warm, calibration is done, user hears
    // "ready" and can start speaking immediately after the tone.
    play_ding();

    // Discard all audio captured during warmup/calibration/ding.
    {
        let mut guard = buffer.lock().unwrap();
        guard.clear();
    }

    // VAD state machine
    let mut speech_started = false;
    let mut silence_start: Option<std::time::Instant> = None;
    let start_time = std::time::Instant::now();
    let max_dur = std::time::Duration::from_millis(max_duration_ms);
    let silence_dur = std::time::Duration::from_millis(silence_timeout_ms);

    loop {
        if INTERRUPTED.load(Ordering::Relaxed) {
            break;
        }

        if start_time.elapsed() >= max_dur {
            if !QUIET.load(Ordering::Relaxed) {
                eprintln!("Max recording duration reached.");
            }
            break;
        }

        let peak_bits = recent_peak.load(Ordering::Relaxed);
        let current_peak = f32::from_bits(peak_bits);
        let is_speech = current_peak > adaptive_threshold;

        if is_speech {
            if !speech_started {
                speech_started = true;
                if !QUIET.load(Ordering::Relaxed) {
                    eprintln!("Speech detected...");
                }
            }
            silence_start = None;
        } else if speech_started {
            if silence_start.is_none() {
                silence_start = Some(std::time::Instant::now());
            }
            if let Some(ss) = silence_start {
                if ss.elapsed() >= silence_dur {
                    if !QUIET.load(Ordering::Relaxed) {
                        eprintln!("Silence detected, stopping recording.");
                    }
                    break;
                }
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    drop(stream);
    play_dong();

    let samples = extract_samples(buffer);
    log_recording_stats(&samples, sample_rate);
    maybe_save_recording(&samples, sample_rate);

    Ok((samples, sample_rate))
}

// ── Continuous recording ───────────────────────────────────────────────

/// Record continuously, splitting speech into segments by silence.
///
/// Returns a receiver that yields `Segment` values as each speech segment
/// completes (silence detected after speech). Recording runs on the audio
/// thread and never stops until `INTERRUPTED` is set or `max_duration_ms`
/// is reached.
///
/// Plays a ding at the start. No dings between segments.
pub fn record_continuous(
    silence_timeout_ms: u64,
    silence_threshold: f32,
    max_duration_ms: u64,
    min_segment_ms: u64,
    max_segment_ms: u64,
    noise_multiplier: f32,
    calibration_ms: u64,
) -> Result<(mpsc::Receiver<Segment>, u32, cpal::Stream), String> {
    let (device, config) = open_input_device()?;
    let sample_rate = config.sample_rate().0;

    if !QUIET.load(Ordering::Relaxed) {
        let name = device.name().unwrap_or_else(|_| "unknown".to_string());
        eprintln!(
            "Recording from: {name} ({sample_rate}Hz, {}ch)",
            config.channels()
        );
    }

    // Start mic first so Bluetooth hardware warms up during calibration.
    let segment_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let recent_peak: Arc<std::sync::atomic::AtomicU32> =
        Arc::new(std::sync::atomic::AtomicU32::new(0));

    let stream = build_input_stream(
        &device,
        &config,
        Arc::clone(&segment_buffer),
        Some(Arc::clone(&recent_peak)),
    )?;

    stream
        .play()
        .map_err(|e| format!("Failed to start recording: {e}"))?;

    let (tx, rx) = mpsc::channel::<Segment>();

    let min_samples = (sample_rate as u64 * min_segment_ms / 1000) as usize;
    let max_samples = (sample_rate as u64 * max_segment_ms / 1000) as usize;

    // VAD monitor thread — watches peak levels, snapshots segments
    // Note: `stream` is NOT moved here — it's returned to the caller
    // who must keep it alive for the duration of recording.
    std::thread::spawn(move || {
        let mut speech_started = false;
        let mut silence_start: Option<Instant> = None;
        let mut segment_id: u64 = 0;
        let max_dur = std::time::Duration::from_millis(max_duration_ms);
        let silence_dur = std::time::Duration::from_millis(silence_timeout_ms);

        // Adaptive noise floor calibration — mic is already warm
        let adaptive_threshold = if calibration_ms > 0 {
            let cal_start = Instant::now();
            let calibration_duration = std::time::Duration::from_millis(calibration_ms);
            let mut max_ambient_peak: f32 = 0.0;

            if !QUIET.load(Ordering::Relaxed) {
                eprintln!("Calibrating noise floor...");
            }

            while cal_start.elapsed() < calibration_duration {
                if INTERRUPTED.load(Ordering::Relaxed) {
                    return;
                }
                let peak_bits = recent_peak.load(Ordering::Relaxed);
                let current_peak = f32::from_bits(peak_bits);
                max_ambient_peak = max_ambient_peak.max(current_peak);
                std::thread::sleep(std::time::Duration::from_millis(25));
            }

            let threshold = (max_ambient_peak * noise_multiplier).max(silence_threshold);

            if !QUIET.load(Ordering::Relaxed) {
                eprintln!(
                    "Noise floor: {:.4}, threshold: {:.4} (×{:.1})",
                    max_ambient_peak, threshold, noise_multiplier
                );
            }

            // Ding signals "ready" after calibration
            play_ding();

            // Clear all warmup/calibration/ding audio
            {
                let mut guard = segment_buffer.lock().unwrap();
                guard.clear();
            }

            threshold
        } else {
            // No calibration — still play ding and clear buffer
            play_ding();
            {
                let mut guard = segment_buffer.lock().unwrap();
                guard.clear();
            }
            silence_threshold
        };

        // Start the clock AFTER the ding — timestamps in segments
        // should reflect time since the user was told "ready".
        let start_time = Instant::now();

        loop {
            if INTERRUPTED.load(Ordering::Relaxed) {
                // Flush any remaining speech as a final segment
                let mut guard = segment_buffer.lock().unwrap();
                if guard.len() >= min_samples {
                    segment_id += 1;
                    let samples = std::mem::take(&mut *guard);
                    let elapsed = start_time.elapsed().as_millis() as u64;
                    let _ = tx.send(Segment {
                        samples,
                        sample_rate,
                        segment_id,
                        timestamp_ms: elapsed,
                    });
                }
                break;
            }

            if start_time.elapsed() >= max_dur {
                if !QUIET.load(Ordering::Relaxed) {
                    eprintln!("Max recording duration reached.");
                }
                // Flush remaining
                let mut guard = segment_buffer.lock().unwrap();
                if guard.len() >= min_samples {
                    segment_id += 1;
                    let samples = std::mem::take(&mut *guard);
                    let elapsed = start_time.elapsed().as_millis() as u64;
                    let _ = tx.send(Segment {
                        samples,
                        sample_rate,
                        segment_id,
                        timestamp_ms: elapsed,
                    });
                }
                break;
            }

            let peak_bits = recent_peak.load(Ordering::Relaxed);
            let current_peak = f32::from_bits(peak_bits);
            let is_speech = current_peak > adaptive_threshold;

            if is_speech {
                if !speech_started {
                    speech_started = true;
                }
                silence_start = None;
            } else if speech_started {
                if silence_start.is_none() {
                    silence_start = Some(Instant::now());
                }
                if let Some(ss) = silence_start {
                    if ss.elapsed() >= silence_dur {
                        // End of speech segment — snapshot and send
                        let segment_data = {
                            let mut guard = segment_buffer.lock().unwrap();
                            if guard.len() >= min_samples {
                                segment_id += 1;
                                let samples = std::mem::take(&mut *guard);
                                let elapsed = start_time.elapsed().as_millis() as u64;
                                Some((samples, elapsed))
                            } else {
                                // Too short — discard
                                guard.clear();
                                None
                            }
                        };

                        if let Some((samples, elapsed)) = segment_data {
                            if !QUIET.load(Ordering::Relaxed) {
                                let dur = samples.len() as f32 / sample_rate as f32;
                                eprintln!("  segment {segment_id}: {dur:.1}s of speech");
                            }

                            maybe_save_recording(&samples, sample_rate);

                            let _ = tx.send(Segment {
                                samples,
                                sample_rate,
                                segment_id,
                                timestamp_ms: elapsed,
                            });
                        }

                        speech_started = false;
                        silence_start = None;
                    }
                }
            }

            // Force-split very long segments
            let forced_split: Option<(Vec<f32>, u64, u64)> = {
                let mut guard = segment_buffer.lock().unwrap();
                if guard.len() >= max_samples && speech_started {
                    segment_id += 1;
                    let samples = std::mem::take(&mut *guard);
                    let elapsed = start_time.elapsed().as_millis() as u64;
                    Some((samples, segment_id, elapsed))
                } else {
                    None
                }
            };

            if let Some((samples, seg_id, elapsed)) = forced_split {
                if !QUIET.load(Ordering::Relaxed) {
                    let dur = samples.len() as f32 / sample_rate as f32;
                    eprintln!("  segment {seg_id}: {dur:.1}s (max length split)");
                }

                maybe_save_recording(&samples, sample_rate);

                let _ = tx.send(Segment {
                    samples,
                    sample_rate,
                    segment_id: seg_id,
                    timestamp_ms: elapsed,
                });

                // Stay in speech_started state — next audio continues the segment
                silence_start = None;
            }

            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        // Channel drops here, signaling the consumer that recording is done
    });

    Ok((rx, sample_rate, stream))
}

/// Consume segments from the recording queue and transcribe each one.
///
/// Spawns a thread that pulls segments, trims silence, resamples, and
/// runs Moonshine inference. Results are sent to the returned receiver.
///
/// The model and tokenizer are moved into this thread — they're not
/// thread-safe, so single-threaded access is correct.
pub fn transcribe_segments(
    mut model: voice_stt::WhisperModel,
    tokenizer: voice_stt::tokenizers::Tokenizer,
    segments: mpsc::Receiver<Segment>,
) -> mpsc::Receiver<SegmentResult> {
    let (tx, rx) = mpsc::channel::<SegmentResult>();

    std::thread::spawn(move || {
        for segment in segments {
            if INTERRUPTED.load(Ordering::Relaxed) {
                break;
            }

            let trimmed = trim_silence(&segment.samples, segment.sample_rate);
            if trimmed.is_empty() {
                continue;
            }

            let t0 = Instant::now();
            let result = voice_stt::transcribe_audio_with_tokenizer(
                &mut model,
                &trimmed,
                segment.sample_rate,
                &tokenizer,
            );

            let transcribe_ms = t0.elapsed().as_millis() as u64;

            match result {
                Ok(r) if !r.text.trim().is_empty() => {
                    let _ = tx.send(SegmentResult {
                        text: r.text,
                        tokens: r.tokens,
                        segment_id: segment.segment_id,
                        timestamp_ms: segment.timestamp_ms,
                        transcribe_ms,
                    });
                }
                Ok(_) => {} // empty transcription — skip
                Err(e) => {
                    if !QUIET.load(Ordering::Relaxed) {
                        eprintln!(
                            "  transcription error on segment {}: {e}",
                            segment.segment_id
                        );
                    }
                }
            }
        }
        // Channel drops here, signaling the output consumer
    });

    rx
}

/// Run continuous listen mode: record → segment → transcribe → print.
///
/// Entry point for `voice listen --continuous`.
pub fn listen_continuous() {
    let (model, tokenizer) = load_stt();

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Listening continuously... (Ctrl+C to stop)\n");
    }

    let (segments_rx, _sample_rate, _stream) = match record_continuous(
        1500,     // silence_timeout_ms
        0.01,     // silence_threshold
        u64::MAX, // max_duration_ms (unlimited — Ctrl+C to stop)
        500,      // min_segment_ms
        30_000,   // max_segment_ms
        3.0,      // noise_multiplier
        500,      // calibration_ms
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Recording failed: {e}");
            std::process::exit(1);
        }
    };

    let results_rx = transcribe_segments(model, tokenizer, segments_rx);

    let start = Instant::now();
    let mut total_segments = 0u64;

    for result in results_rx {
        total_segments += 1;
        let ts_secs = result.timestamp_ms / 1000;
        let ts_mins = ts_secs / 60;
        let ts_rem = ts_secs % 60;
        println!("[{ts_mins}:{ts_rem:02}] {}", result.text);
        let _ = io::stdout().flush();
    }

    // Reset interrupt for clean exit
    INTERRUPTED.store(false, Ordering::Relaxed);

    let total_secs = start.elapsed().as_secs_f64();
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("\n{total_segments} segments transcribed in {total_secs:.1}s");
    }
}

/// Run continuous listen for JSON-RPC, yielding segment results.
///
/// Returns a receiver of SegmentResult. The caller (jsonrpc dispatch)
/// sends notifications per result and a final response on completion.
#[allow(dead_code)]
pub fn listen_continuous_for_rpc(
    model: voice_stt::WhisperModel,
    tokenizer: voice_stt::tokenizers::Tokenizer,
    silence_timeout_ms: u64,
    max_duration_ms: u64,
    noise_multiplier: f32,
    calibration_ms: u64,
) -> Result<mpsc::Receiver<SegmentResult>, String> {
    let (segments_rx, _sample_rate, _stream) = record_continuous(
        silence_timeout_ms,
        0.01, // silence_threshold
        max_duration_ms,
        500,    // min_segment_ms
        30_000, // max_segment_ms
        noise_multiplier,
        calibration_ms,
    )?;

    Ok(transcribe_segments(model, tokenizer, segments_rx))
}

// ── Audio processing ───────────────────────────────────────────────────

/// Trim leading and trailing silence from audio samples.
///
/// Bluetooth microphones (e.g. AirPods) can take ~0.5-1s before audio
/// actually flows, producing a block of zeros at the start. Moonshine
/// is sensitive to the silence-to-speech ratio, especially on short
/// recordings — trimming silence dramatically improves accuracy.
fn trim_silence(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    // Use windowed RMS instead of single-sample threshold to avoid
    // clipping soft word onsets like "the", "a", etc.
    let window_size = (sample_rate as usize) / 100; // 10ms window
    let rms_threshold: f32 = 0.005; // ~-46 dBFS RMS

    // Keep 250ms of leading context and 100ms trailing to preserve
    // consonant onsets and natural decay.
    let lead_padding = (sample_rate as usize) / 4;
    let trail_padding = (sample_rate as usize) / 10;

    // Find first window where RMS exceeds threshold
    let first_voice = samples.windows(window_size).position(|w| {
        let rms = (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt();
        rms > rms_threshold
    });

    // Find last window where RMS exceeds threshold
    let last_voice = samples.windows(window_size).rposition(|w| {
        let rms = (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt();
        rms > rms_threshold
    });

    match (first_voice, last_voice) {
        (Some(start), Some(end)) => {
            let start = start.saturating_sub(lead_padding);
            let end = (end + window_size + trail_padding).min(samples.len());
            samples[start..end].to_vec()
        }
        _ => Vec::new(),
    }
}

// ── Debug recording save ───────────────────────────────────────────────

/// Save samples to a timestamped WAV file in /tmp for debugging.
/// Only runs if `VOICE_SAVE_RECORDING` env var is set.
fn maybe_save_recording(samples: &[f32], sample_rate: u32) {
    if std::env::var("VOICE_SAVE_RECORDING").is_err() || samples.is_empty() {
        return;
    }

    let timestamp = {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let days = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;
        let (year, month, day) = days_to_ymd(days);
        format!("{year:04}{month:02}{day:02}_{hours:02}{minutes:02}{seconds:02}")
    };

    let debug_path = format!("/tmp/voice_recording_{timestamp}.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    if let Ok(mut writer) = hound::WavWriter::create(&debug_path, spec) {
        for &s in samples {
            let _ = writer.write_sample(s);
        }
        let _ = writer.finalize();
        if !QUIET.load(Ordering::Relaxed) {
            eprintln!("Saved recording to {debug_path}");
        }
    }
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ── STT model helpers ──────────────────────────────────────────────────

/// Default STT model. Override with `STT_MODEL` env var.
/// distil-medium.en: English-only Whisper distillation, fast and accurate.
const DEFAULT_STT_MODEL: &str = "distil-whisper/distil-medium.en";

fn stt_model_repo() -> String {
    std::env::var("STT_MODEL").unwrap_or_else(|_| DEFAULT_STT_MODEL.to_string())
}

/// Load STT model and tokenizer. Prints progress to stderr unless quiet.
fn load_stt() -> (voice_stt::WhisperModel, voice_stt::tokenizers::Tokenizer) {
    let repo = stt_model_repo();

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Loading speech-to-text model ({repo})...");
    }

    let model = match voice_stt::load_model(&repo) {
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

    (model, tokenizer)
}

/// Run transcription on recorded audio with silence trimming.
fn transcribe_samples(
    model: &mut voice_stt::WhisperModel,
    tokenizer: &voice_stt::tokenizers::Tokenizer,
    samples: &[f32],
    sample_rate: u32,
) -> Option<voice_stt::TranscribeResult> {
    // Trim leading/trailing silence
    let original_len = samples.len();
    let trimmed = trim_silence(samples, sample_rate);
    let trimmed_duration = trimmed.len() as f32 / sample_rate as f32;

    if !QUIET.load(Ordering::Relaxed) && trimmed.len() < original_len {
        let original_duration = original_len as f32 / sample_rate as f32;
        eprintln!(
            "Trimmed silence: {:.1}s → {:.1}s of speech",
            original_duration, trimmed_duration
        );
    }

    if trimmed.is_empty() {
        eprintln!("No speech detected in recording.");
        return None;
    }

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Transcribing {:.1}s of audio...", trimmed_duration);
    }

    match voice_stt::transcribe_audio_with_tokenizer(model, &trimmed, sample_rate, tokenizer) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("Transcription failed: {e}");
            None
        }
    }
}

// ── Public entry points ────────────────────────────────────────────────

/// Record from mic (manual stop), transcribe, print result.
///
/// Entry point for `voice listen`.
/// Record from mic with VAD auto-stop, transcribe, print result.
///
/// Like `listen_and_transcribe` but uses voice activity detection instead
/// of waiting for Enter/Ctrl+C — stops automatically after silence.
pub fn listen_and_transcribe_auto() {
    let (mut model, tokenizer) = load_stt();

    if let Some(result) = listen_and_transcribe_vad(
        &mut model, &tokenizer, 30_000, // max_duration_ms
        2_000,  // silence_timeout_ms
        0.01,   // silence_threshold
        3.0,    // noise_multiplier
        500,    // calibration_ms
    ) {
        println!("{}", result.text);
        if !QUIET.load(Ordering::Relaxed) {
            let _ = io::stderr().flush();
            eprintln!("\n({} tokens)", result.tokens.len());
        }
    }
}

pub fn listen_and_transcribe() {
    let (mut model, tokenizer) = load_stt();

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

    // Reset interrupt so the process exits cleanly
    INTERRUPTED.store(false, Ordering::Relaxed);

    if let Some(result) = transcribe_samples(&mut model, &tokenizer, &samples, sample_rate) {
        println!("{}", result.text);
        if !QUIET.load(Ordering::Relaxed) {
            let _ = io::stderr().flush();
            eprintln!("\n({} tokens)", result.tokens.len());
        }
    }
}

/// Record from mic (VAD auto-stop), transcribe, return result.
///
/// Used by the JSON-RPC `listen` method. Returns `None` if no speech
/// was detected or transcription failed.
pub fn listen_and_transcribe_vad(
    model: &mut voice_stt::WhisperModel,
    tokenizer: &voice_stt::tokenizers::Tokenizer,
    max_duration_ms: u64,
    silence_timeout_ms: u64,
    silence_threshold: f32,
    noise_multiplier: f32,
    calibration_ms: u64,
) -> Option<voice_stt::TranscribeResult> {
    let (samples, sample_rate) = match record_with_vad(
        max_duration_ms,
        silence_timeout_ms,
        silence_threshold,
        noise_multiplier,
        calibration_ms,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Recording failed: {e}");
            return None;
        }
    };

    if INTERRUPTED.load(Ordering::Relaxed) || samples.is_empty() {
        return None;
    }

    transcribe_samples(model, tokenizer, &samples, sample_rate)
}

/// Transcribe a WAV file and print the result.
///
/// Entry point for `voice --transcribe <file>`.
pub fn transcribe_file(path: &Path) {
    let (mut model, tokenizer) = load_stt();

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Transcribing: {}", path.display());
    }

    // Load WAV
    let reader = match hound::WavReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open {}: {e}", path.display());
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
