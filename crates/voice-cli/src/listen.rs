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
use std::time::{Duration, Instant};

use rodio::microphone::MicrophoneBuilder;
use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, Player};
use voice_stream::InterleavedMonoMixer;

use crate::{INTERRUPTED, QUIET};

const MIC_OPEN_TIMEOUT_MS: u64 = 8_000;

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

pub struct TranscriptionAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
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

pub fn load_transcription_audio(path: &Path) -> Result<TranscriptionAudio, String> {
    let audio = voice_stt::load_audio_file(path).map_err(|e| e.to_string())?;
    Ok(TranscriptionAudio {
        samples: audio.samples,
        sample_rate: audio.sample_rate,
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

/// Snapshot of the live VAD recording buffer for best-effort partial work.
#[derive(Debug, Clone)]
pub struct VadProgressSnapshot {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub audio_ms: u64,
    pub elapsed_ms: u64,
    pub current_peak: f32,
    pub threshold: f32,
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
        self.record_vad_with_progress(
            max_duration_ms,
            silence_timeout_ms,
            silence_threshold,
            noise_multiplier,
            calibration_ms,
            None,
            |_| Ok(()),
        )
    }

    /// Record with VAD and optionally publish live buffer snapshots while
    /// speech is still active. The mic drain thread keeps capturing while the
    /// progress callback runs.
    #[allow(clippy::too_many_arguments)]
    pub fn record_vad_with_progress<F>(
        &self,
        max_duration_ms: u64,
        silence_timeout_ms: u64,
        silence_threshold: f32,
        noise_multiplier: f32,
        calibration_ms: u64,
        progress_interval_ms: Option<u64>,
        mut on_progress: F,
    ) -> Result<(Vec<f32>, u32), String>
    where
        F: FnMut(VadProgressSnapshot) -> Result<(), String>,
    {
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
        let progress_interval = progress_interval_ms
            .filter(|ms| *ms > 0)
            .map(std::time::Duration::from_millis);
        let mut last_progress_at = start_time;

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
                    last_progress_at = start_time;
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

            if speech_started
                && is_speech
                && progress_interval
                    .is_some_and(|interval| last_progress_at.elapsed() >= interval)
            {
                let samples = self.buffer.lock().unwrap().clone();
                let audio_ms =
                    samples.len().saturating_mul(1_000) as u64 / self.sample_rate.max(1) as u64;
                on_progress(VadProgressSnapshot {
                    samples,
                    sample_rate: self.sample_rate,
                    audio_ms,
                    elapsed_ms: start_time.elapsed().as_millis() as u64,
                    current_peak,
                    threshold: adaptive_threshold,
                })?;
                last_progress_at = std::time::Instant::now();
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
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(open_mic_inner());
    });

    let timeout = Duration::from_millis(MIC_OPEN_TIMEOUT_MS);
    let started = Instant::now();
    loop {
        if INTERRUPTED.load(Ordering::Relaxed) {
            return Err("Interrupted".to_string());
        }

        let elapsed = started.elapsed();
        if elapsed >= timeout {
            return Err(format!(
                "Opening microphone timed out after {}ms",
                timeout.as_millis()
            ));
        }

        let remaining = timeout.saturating_sub(elapsed);
        let poll = remaining.min(Duration::from_millis(100));
        match rx.recv_timeout(poll) {
            Ok(result) => return result,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err("Microphone open worker exited without a result".to_string());
            }
        }
    }
}

fn open_mic_inner() -> Result<(String, u32, rodio::microphone::Microphone), String> {
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
        let mut mixer = InterleavedMonoMixer::new(channels as usize);
        let mut chunk_peak: f32 = 0.0;
        let mut sample_count = 0usize;

        for sample in mic {
            if stop.load(Ordering::Relaxed) {
                break;
            }

            let abs = sample.abs();
            chunk_peak = chunk_peak.max(abs);
            sample_count += 1;

            if let Some(mono) = mixer.push(sample) {
                buffer.lock().unwrap().push(mono);
            }

            // Update peak every ~100 samples to avoid atomic contention
            if sample_count.is_multiple_of(100) {
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

/// Exercise speaker playback and mic capture so the OS surfaces any
/// permission prompt now, rather than on the hot path when an agent is
/// mid-`converse` waiting on a reply.
///
/// On macOS, microphone access is gated by TCC and the prompt fires on first
/// capture. Running this from the `voice` binary during `voice daemon install`
/// pins the grant to that binary identity, which the daemon (`voice daemon
/// start`) then inherits. Playback is best-effort and non-fatal; a denied or
/// missing mic returns `Err` so the caller can surface a clear failure.
pub fn warmup_permissions() -> Result<(), String> {
    // Speaker first: short ding. Playback isn't TCC-gated, but a silent or
    // missing output device is worth surfacing at install time.
    play_ding();

    // Mic: open the stream (this triggers the TCC prompt) and drain a brief
    // window so the OS registers real capture, then tear it down.
    let (name, sample_rate, mic) = open_mic()?;
    eprintln!("  mic:     {name} ({sample_rate}Hz)");

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

    std::thread::sleep(std::time::Duration::from_millis(400));
    stop.store(true, Ordering::SeqCst);
    let _ = mic_thread.join();

    Ok(())
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
/// runs STT inference. Results are sent to the returned receiver.
///
/// The model is moved into this thread because single-threaded access is
/// required by the decoder state.
pub fn transcribe_segments(
    mut model: voice_stt::SttModel,
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
            let result = model.transcribe_audio(&trimmed, segment.sample_rate);

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
    listen_continuous_with_options(SttLoadOptions::default());
}

pub fn listen_continuous_with_options(options: SttLoadOptions) {
    let model = load_stt_with_options(options);

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

// ── Audio processing ───────────────────────────────────────────────────

/// Trim leading and trailing silence from audio samples.
///
/// Bluetooth microphones (e.g. AirPods) can take ~0.5-1s before audio
/// actually flows, producing a block of zeros at the start. Whisper
/// STT is sensitive to the silence-to-speech ratio, especially on short
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

#[derive(Debug, Clone, Default)]
pub struct SttLoadOptions {
    pub backend: Option<voice_stt::SttBackend>,
    pub model: Option<String>,
    pub max_new_tokens: Option<usize>,
}

impl SttLoadOptions {
    pub fn is_explicit(&self) -> bool {
        self.backend.is_some() || self.model.is_some() || self.max_new_tokens.is_some()
    }
}

/// STT backend/model the CLI loads. Defaults to Whisper with the shared
/// [`voice_stt::builtin::DEFAULT_MODEL_REPO`] (distil-large-v3.5). Override with
/// `STT_BACKEND=voxtral` and/or `STT_MODEL`, or pass explicit CLI options.
fn resolve_stt_load_options(options: &SttLoadOptions) -> Result<(voice_stt::SttBackend, String), String> {
    let env_backend = match std::env::var("STT_BACKEND") {
        Ok(value) => Some(voice_stt::SttBackend::parse(value.trim()).map_err(|e| e.to_string())?),
        Err(_) => None,
    };
    let env_model = std::env::var("STT_MODEL").ok();
    let backend = options.backend.or(env_backend);
    let model = options.model.as_deref().or(env_model.as_deref());
    Ok(voice_stt::resolve_backend_and_model(backend, model))
}

/// Load STT model. Prints progress to stderr unless quiet.
///
/// The tokenizer is loaded internally by the model — no separate load needed.
pub fn load_stt() -> voice_stt::SttModel {
    load_stt_with_options(SttLoadOptions::default())
}

pub fn load_stt_with_options(options: SttLoadOptions) -> voice_stt::SttModel {
    match try_load_stt_with_options(options) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Failed to load STT model: {e}");
            eprintln!("Model weights will be downloaded from HuggingFace on first run.");
            std::process::exit(1);
        }
    }
}

pub fn try_load_stt() -> Result<voice_stt::SttModel, String> {
    try_load_stt_with_options(SttLoadOptions::default())
}

pub fn try_load_stt_with_options(options: SttLoadOptions) -> Result<voice_stt::SttModel, String> {
    let (backend, repo) = match resolve_stt_load_options(&options) {
        Ok(resolved) => resolved,
        Err(e) => return Err(format!("invalid STT configuration: {e}")),
    };

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!(
            "Loading speech-to-text model (backend={}, model={repo})...",
            backend.as_str()
        );
    }

    let mut model =
        voice_stt::load_backend_model(backend, &repo).map_err(|e| e.to_string())?;
    if let Some(max_new_tokens) = options.max_new_tokens {
        model.set_max_new_tokens(max_new_tokens);
    }

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Model loaded. Ready to listen.\n");
    }

    Ok(model)
}

/// Run transcription on recorded audio with silence trimming.
pub(crate) fn transcribe_samples(
    model: &mut voice_stt::SttModel,
    samples: &[f32],
    sample_rate: u32,
) -> Option<voice_stt::TranscribeResult> {
    transcribe_samples_inner(model, samples, sample_rate, true)
}

/// Run transcription on recorded audio with silence trimming, suppressing
/// best-effort partial failures.
pub(crate) fn transcribe_samples_best_effort(
    model: &mut voice_stt::SttModel,
    samples: &[f32],
    sample_rate: u32,
) -> Option<voice_stt::TranscribeResult> {
    transcribe_samples_inner(model, samples, sample_rate, false)
}

fn transcribe_samples_inner(
    model: &mut voice_stt::SttModel,
    samples: &[f32],
    sample_rate: u32,
    report_failures: bool,
) -> Option<voice_stt::TranscribeResult> {
    // Trim leading/trailing silence
    let original_len = samples.len();
    let trimmed = trim_silence(samples, sample_rate);
    let trimmed_duration = trimmed.len() as f32 / sample_rate as f32;

    if report_failures && !QUIET.load(Ordering::Relaxed) && trimmed.len() < original_len {
        let original_duration = original_len as f32 / sample_rate as f32;
        eprintln!(
            "Trimmed silence: {:.1}s → {:.1}s of speech",
            original_duration, trimmed_duration
        );
    }

    if trimmed.is_empty() {
        if report_failures {
            eprintln!("No speech detected in recording.");
        }
        return None;
    }

    if report_failures && !QUIET.load(Ordering::Relaxed) {
        eprintln!("Transcribing {:.1}s of audio...", trimmed_duration);
    }

    match model.transcribe_audio(&trimmed, sample_rate) {
        Ok(r) => Some(r),
        Err(e) => {
            if report_failures {
                eprintln!("Transcription failed: {e}");
            }
            None
        }
    }
}

// ── Public entry points ────────────────────────────────────────────────

/// Record from mic (manual stop), transcribe, print result.
///
/// Entry point for `voice listen`.
pub fn listen_and_transcribe() {
    listen_and_transcribe_with_options(SttLoadOptions::default());
}

pub fn listen_and_transcribe_with_options(options: SttLoadOptions) {
    let mut model = load_stt_with_options(options);

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
    model: &mut voice_stt::SttModel,
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
    model: &mut voice_stt::SttModel,
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

/// Transcribe an audio file and print the result.
///
/// Entry point for `voice --transcribe <file>`.
pub fn transcribe_file(path: &Path) {
    transcribe_file_with_options(path, SttLoadOptions::default());
}

pub fn transcribe_file_with_options(path: &Path, options: SttLoadOptions) {
    let mut model = load_stt_with_options(options);

    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("Transcribing: {}", path.display());
    }

    let audio = match load_transcription_audio(path) {
        Ok(audio) => audio,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let duration = audio.samples.len() as f32 / audio.sample_rate as f32;
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!(
            "Audio: {:.1}s ({} samples at {}Hz)",
            duration,
            audio.samples.len(),
            audio.sample_rate
        );
    }

    let result = match model.transcribe_audio(&audio.samples, audio.sample_rate) {
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
