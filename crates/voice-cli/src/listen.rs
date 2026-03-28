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

use rodio::microphone::MicrophoneBuilder;
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
    play_sound(&sound.samples, sound.sample_rate, sound.channels);
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

/// Generate the default ding samples: short 880Hz sine with exponential decay.
/// Stereo (both ears) with a short silence tail to prevent clipping.
fn synth_ding_samples() -> (Vec<f32>, u32, u16) {
    let sample_rate = 44100u32;
    let duration_ms = 120;
    let tail_ms = 20; // silence tail so the player doesn't cut the last samples
    let decay = 0.03f32;
    let volume = 0.20f32;
    let num_samples = sample_rate as usize * duration_ms / 1000;
    let tail_samples = sample_rate as usize * tail_ms / 1000;

    // Stereo: L R L R ...
    let mut samples = Vec::with_capacity((num_samples + tail_samples) * 2);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let envelope = (-t / decay).exp() * volume;
        let s = (2.0 * std::f32::consts::PI * 880.0 * t).sin() * envelope;
        samples.push(s); // L
        samples.push(s); // R
    }
    // Silence tail
    samples.extend(std::iter::repeat_n(0.0f32, tail_samples * 2));
    (samples, sample_rate, 2)
}

/// Play a sound on the default output device. Blocks until playback finishes.
fn play_sound(samples: &[f32], sample_rate: u32, channels: u16) {
    let Ok(mut stream) = DeviceSinkBuilder::open_default_sink() else {
        return;
    };
    stream.log_on_drop(false);
    let player = Player::connect_new(stream.mixer());

    let Some(ch) = NonZero::new(channels) else {
        return;
    };
    let Some(rate) = NonZero::new(sample_rate) else {
        return;
    };
    player.append(SamplesBuffer::new(ch, rate, samples.to_vec()));

    while !player.empty() {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// Cached default ding samples (synthesized once).
static DEFAULT_DING: std::sync::OnceLock<(Vec<f32>, u32, u16)> = std::sync::OnceLock::new();

/// Cached default dong samples (synthesized once).
static DEFAULT_DONG: std::sync::OnceLock<(Vec<f32>, u32, u16)> = std::sync::OnceLock::new();

/// Play a short pleasant ding to signal that recording has started.
///
/// Uses custom start sound if configured, otherwise plays a cached 880Hz sine ding.
fn play_ding() {
    let config = sound_config().lock().unwrap();
    if let Some(sound) = &config.start_sound {
        let (samples, rate, ch) = (sound.samples.clone(), sound.sample_rate, sound.channels);
        drop(config);
        play_sound(&samples, rate, ch);
    } else {
        drop(config);
        let (samples, rate, ch) = DEFAULT_DING.get_or_init(synth_ding_samples);
        play_sound(samples, *rate, *ch);
    }
}

/// Play two ascending blips to signal that listening has stopped.
///
/// Uses custom stop sound if configured, otherwise synthesizes two quick
/// sine blips — C5 (523Hz) then E5 (659Hz) — each with a smooth
/// sine-shaped envelope and a short gap between them. Feels like a
/// positive "got it" confirmation.
/// Generate the default dong samples: two ascending blips (C5 → E5).
/// Stereo (both ears) with a silence tail to prevent clipping.
fn synth_dong_samples() -> (Vec<f32>, u32, u16) {
    let sample_rate = 44100u32;
    let pi2 = 2.0 * std::f32::consts::PI;
    let volume = 0.10f32;
    let tail_ms = 20;

    let blip_samples = sample_rate as usize * 120 / 1000;
    let gap_samples = sample_rate as usize * 60 / 1000;
    let tail_samples = sample_rate as usize * tail_ms / 1000;

    let mut samples = Vec::with_capacity((blip_samples * 2 + gap_samples + tail_samples) * 2);

    // First blip: C5
    for i in 0..blip_samples {
        let t = i as f32 / sample_rate as f32;
        let envelope = (std::f32::consts::PI * i as f32 / blip_samples as f32).sin() * volume;
        let s = (pi2 * 523.25 * t).sin() * envelope;
        samples.push(s); // L
        samples.push(s); // R
    }

    // Gap (stereo silence)
    samples.extend(std::iter::repeat_n(0.0f32, gap_samples * 2));

    // Second blip: E5
    for i in 0..blip_samples {
        let t = i as f32 / sample_rate as f32;
        let envelope = (std::f32::consts::PI * i as f32 / blip_samples as f32).sin() * volume;
        let s = (pi2 * 659.26 * t).sin() * envelope;
        samples.push(s); // L
        samples.push(s); // R
    }

    // Silence tail
    samples.extend(std::iter::repeat_n(0.0f32, tail_samples * 2));

    (samples, sample_rate, 2)
}

fn play_dong() {
    let config = sound_config().lock().unwrap();
    if let Some(sound) = &config.stop_sound {
        let (samples, rate, ch) = (sound.samples.clone(), sound.sample_rate, sound.channels);
        drop(config);
        play_sound(&samples, rate, ch);
    } else {
        drop(config);
        let (samples, rate, ch) = DEFAULT_DONG.get_or_init(synth_dong_samples);
        play_sound(samples, *rate, *ch);
    }
}

// ── Mic input helpers ──────────────────────────────────────────────────

/// A pre-opened microphone that's already draining audio into a buffer.
///
/// Use in long-running processes (MCP server) to keep the mic open across
/// calls, avoiding repeated Bluetooth HFP codec switches. The first open
/// triggers the switch; subsequent recordings reuse the warm mic.
pub struct WarmMic {
    pub sample_rate: u32,
    buffer: Arc<Mutex<Vec<f32>>>,
    recent_peak: Arc<std::sync::atomic::AtomicU32>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    mic_thread: Option<std::thread::JoinHandle<()>>,
}

impl WarmMic {
    /// Open the mic and start buffering immediately.
    pub fn open() -> Result<Self, String> {
        let (_, sample_rate, mic) = open_mic()?;
        let channels = mic.config().channel_count.get();
        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let recent_peak = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let mic_thread = start_mic_drain(
            mic,
            Arc::clone(&buffer),
            Arc::clone(&recent_peak),
            Arc::clone(&stop),
            channels,
        );

        Ok(Self {
            sample_rate,
            buffer,
            recent_peak,
            stop,
            mic_thread: Some(mic_thread),
        })
    }

    /// Clear the audio buffer, discarding any previously captured audio.
    pub fn clear(&self) {
        self.buffer.lock().unwrap().clear();
    }

    /// Record with VAD using this warm mic. Plays ding, clears buffer,
    /// calibrates, then records until silence after speech.
    pub fn record_vad(
        &self,
        max_duration_ms: u64,
        silence_timeout_ms: u64,
        silence_threshold: f32,
        noise_multiplier: f32,
        calibration_ms: u64,
    ) -> Result<(Vec<f32>, u32), String> {
        play_ding();
        if !QUIET.load(Ordering::Relaxed) {
            eprintln!("Listening...");
            eprintln!("Recording from: warm mic ({}Hz)", self.sample_rate);
        }

        // Clear buffer — discard audio from before this recording
        self.clear();

        // Adaptive noise floor calibration
        let adaptive_threshold = if calibration_ms > 0 {
            let cal_start = std::time::Instant::now();
            let calibration_duration = std::time::Duration::from_millis(calibration_ms);
            let mut max_ambient_peak: f32 = 0.0;

            while cal_start.elapsed() < calibration_duration {
                if INTERRUPTED.load(Ordering::Relaxed) {
                    let samples = self.buffer.lock().unwrap().clone();
                    return Ok((samples, self.sample_rate));
                }
                let peak_bits = self.recent_peak.load(Ordering::Relaxed);
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
            silence_threshold
        };

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

            let peak_bits = self.recent_peak.load(Ordering::Relaxed);
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

        play_dong();

        // Take a snapshot of the buffer — don't stop the mic thread
        let samples = self.buffer.lock().unwrap().clone();
        log_recording_stats(&samples, self.sample_rate);
        maybe_save_recording(&samples, self.sample_rate);

        Ok((samples, self.sample_rate))
    }
}

impl Drop for WarmMic {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.mic_thread.take() {
            let _ = t.join();
        }
    }
}

/// Open a microphone via rodio and return (name, sample_rate, mic).
///
/// Uses the default input device. Rodio handles sample format conversion
/// and multi-channel mixing internally.
fn open_mic() -> Result<(String, u32, rodio::microphone::Microphone), String> {
    let mic = MicrophoneBuilder::new()
        .default_device()
        .map_err(|e| format!("No input device: {e}"))?
        .default_config()
        .map_err(|e| format!("No input config: {e}"))?
        .open_stream()
        .map_err(|e| format!("Failed to open mic: {e}"))?;

    let config = mic.config();
    let sample_rate = config.sample_rate.get();
    let name = "default".to_string(); // rodio doesn't expose device name easily

    Ok((name, sample_rate, mic))
}

/// Drain a Microphone iterator into a shared buffer on a background thread.
///
/// Returns the peak amplitude tracker. The mic thread runs until the
/// `stop` flag is set or the mic stream ends.
fn start_mic_drain(
    mic: rodio::microphone::Microphone,
    buffer: Arc<Mutex<Vec<f32>>>,
    peak_out: Arc<std::sync::atomic::AtomicU32>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    channels: u16,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let ch = channels.max(1) as usize;
        let mut chunk_peak: f32 = 0.0;
        let mut sample_count = 0usize;

        for sample in mic {
            if stop.load(Ordering::Relaxed) {
                break;
            }

            let abs = sample.abs();
            chunk_peak = chunk_peak.max(abs);
            sample_count += 1;

            // For multi-channel, mix to mono by averaging
            if ch == 1 {
                buffer.lock().unwrap().push(sample);
            } else if sample_count % ch == 0 {
                // We get interleaved samples — just take every ch-th sample
                // (rodio's SampleTypeConverter already converts to f32)
                buffer.lock().unwrap().push(sample);
            }

            // Update peak every ~100 samples to avoid atomic contention
            if sample_count % 100 == 0 {
                peak_out.store(chunk_peak.to_bits(), Ordering::Relaxed);
                chunk_peak = 0.0;
            }
        }
    })
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
    // Ding before mic so Bluetooth users hear the "ready" signal
    play_ding();

    let (name, sample_rate, mic) = open_mic()?;

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Recording from: {name} ({sample_rate}Hz)");
    }

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let peak = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let channels = mic.config().channel_count.get();

    let mic_thread = start_mic_drain(
        mic,
        Arc::clone(&buffer),
        Arc::clone(&peak),
        Arc::clone(&stop),
        channels,
    );

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

    stop.store(true, Ordering::SeqCst);
    let _ = mic_thread.join();
    drop(stdin_thread);
    play_dong();

    let samples = match Arc::try_unwrap(buffer) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(arc) => arc.lock().unwrap().clone(),
    };
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
    // Play ding BEFORE opening mic. On Bluetooth (AirPods), opening the
    // mic triggers an HFP profile switch that can swallow audio output.
    // Playing first ensures the user hears the "ready" signal.
    play_ding();
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Listening...");
    }

    let (name, sample_rate, mic) = open_mic()?;

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Recording from: {name} ({sample_rate}Hz)");
    }

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let recent_peak = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let channels = mic.config().channel_count.get();

    let mic_thread = start_mic_drain(
        mic,
        Arc::clone(&buffer),
        Arc::clone(&recent_peak),
        Arc::clone(&stop),
        channels,
    );

    // Adaptive noise floor calibration
    let adaptive_threshold = if calibration_ms > 0 {
        let cal_start = std::time::Instant::now();
        let calibration_duration = std::time::Duration::from_millis(calibration_ms);
        let mut max_ambient_peak: f32 = 0.0;

        while cal_start.elapsed() < calibration_duration {
            if INTERRUPTED.load(Ordering::Relaxed) {
                stop.store(true, Ordering::SeqCst);
                let _ = mic_thread.join();
                let samples = match Arc::try_unwrap(buffer) {
                    Ok(mutex) => mutex.into_inner().unwrap(),
                    Err(arc) => arc.lock().unwrap().clone(),
                };
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
        silence_threshold
    };

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

    stop.store(true, Ordering::SeqCst);
    let _ = mic_thread.join();
    play_dong();

    let samples = match Arc::try_unwrap(buffer) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(arc) => arc.lock().unwrap().clone(),
    };
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
/// Stop handle for continuous recording. Drop or call stop() to end.
pub struct RecordingHandle {
    stop: Arc<std::sync::atomic::AtomicBool>,
}

impl RecordingHandle {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

impl Drop for RecordingHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn record_continuous(
    silence_timeout_ms: u64,
    silence_threshold: f32,
    max_duration_ms: u64,
    min_segment_ms: u64,
    max_segment_ms: u64,
    noise_multiplier: f32,
    calibration_ms: u64,
) -> Result<(mpsc::Receiver<Segment>, u32, RecordingHandle), String> {
    // Ding before mic so Bluetooth users hear it
    play_ding();

    let (name, sample_rate, mic) = open_mic()?;

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Recording from: {name} ({sample_rate}Hz)");
    }

    let segment_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let recent_peak = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let channels = mic.config().channel_count.get();

    let _mic_thread = start_mic_drain(
        mic,
        Arc::clone(&segment_buffer),
        Arc::clone(&recent_peak),
        Arc::clone(&stop),
        channels,
    );

    let (tx, rx) = mpsc::channel::<Segment>();

    let min_samples = (sample_rate as u64 * min_segment_ms / 1000) as usize;
    let max_samples = (sample_rate as u64 * max_segment_ms / 1000) as usize;

    // VAD monitor thread
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

            // Don't clear — trim_silence handles leading silence.
            // Clearing here loses speech if user starts during calibration.

            threshold
        } else {
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

    let handle = RecordingHandle {
        stop: Arc::clone(&stop),
    };
    Ok((rx, sample_rate, handle))
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
            let result = voice_stt::transcribe_audio(&mut model, &trimmed, segment.sample_rate);

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
    let model = load_stt();

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Listening continuously... (Ctrl+C to stop)\n");
    }

    let (segments_rx, _sample_rate, _handle) = match record_continuous(
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

    let results_rx = transcribe_segments(model, segments_rx);

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
    silence_timeout_ms: u64,
    max_duration_ms: u64,
    noise_multiplier: f32,
    calibration_ms: u64,
) -> Result<mpsc::Receiver<SegmentResult>, String> {
    let (segments_rx, _sample_rate, _handle) = record_continuous(
        silence_timeout_ms,
        0.01, // silence_threshold
        max_duration_ms,
        500,    // min_segment_ms
        30_000, // max_segment_ms
        noise_multiplier,
        calibration_ms,
    )?;

    Ok(transcribe_segments(model, segments_rx))
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

    // Keep 500ms of leading context and 100ms trailing to preserve
    // soft word onsets, especially after Bluetooth mic spin-up.
    let lead_padding = (sample_rate as usize) / 2;
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
/// distil-large-v3: best accuracy. Use distil-whisper/distil-medium.en for smaller/faster.
const DEFAULT_STT_MODEL: &str = "distil-whisper/distil-large-v3";

fn stt_model_repo() -> String {
    std::env::var("STT_MODEL").unwrap_or_else(|_| DEFAULT_STT_MODEL.to_string())
}

/// Load STT model. Prints progress to stderr unless quiet.
///
/// The tokenizer is loaded internally by the model — no separate load needed.
pub fn load_stt() -> voice_stt::WhisperModel {
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

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Model loaded. Ready to listen.\n");
    }

    model
}

/// Run transcription on recorded audio with silence trimming.
fn transcribe_samples(
    model: &mut voice_stt::WhisperModel,
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

    match voice_stt::transcribe_audio(model, &trimmed, sample_rate) {
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
pub fn listen_and_transcribe() {
    let mut model = load_stt();

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

    if let Some(result) = transcribe_samples(&mut model, &samples, sample_rate) {
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

    transcribe_samples(model, &samples, sample_rate)
}

/// Like `listen_and_transcribe_vad` but uses a pre-warmed mic.
///
/// The mic stays open after recording — caller retains ownership.
pub fn listen_and_transcribe_vad_warm(
    model: &mut voice_stt::WhisperModel,
    warm_mic: &WarmMic,
    max_duration_ms: u64,
    silence_timeout_ms: u64,
    silence_threshold: f32,
    noise_multiplier: f32,
    calibration_ms: u64,
) -> Option<voice_stt::TranscribeResult> {
    let (samples, sample_rate) = match warm_mic.record_vad(
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

    transcribe_samples(model, &samples, sample_rate)
}

/// Transcribe a WAV file and print the result.
///
/// Entry point for `voice --transcribe <file>`.
pub fn transcribe_file(path: &Path) {
    let mut model = load_stt();

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

    let result = match voice_stt::transcribe_audio(&mut model, &mono, sample_rate) {
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
