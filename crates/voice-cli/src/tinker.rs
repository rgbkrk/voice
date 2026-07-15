//! Rust-first microphone turns for Tinker's audio-capable Inkling model.
//!
//! Capture, VAD, WAV framing, process supervision, and result reporting stay
//! in Rust. Tinker's only public audio renderer is currently distributed as a
//! Rust-backed Python extension, so a warm JSONL worker owns that narrow SDK
//! boundary.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

const DEFAULT_MODEL: &str = "thinkingmachines/Inkling";
const TINKER_AUDIO_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_INSTRUCTION: &str = "Respond naturally and concisely to the speaker. Use both the words and audible delivery, including tone, hesitation, emphasis, and emotion when relevant. Do not provide a transcript unless asked.";
const WORKER_SOURCE: &str = include_str!("tinker_worker.py");

#[derive(clap::Args, Debug)]
pub struct TinkerArgs {
    /// Existing PCM WAV to send instead of recording from the microphone
    #[arg(long, value_name = "PATH")]
    audio: Option<PathBuf>,

    /// Instruction placed before each audio turn
    #[arg(long, default_value = DEFAULT_INSTRUCTION)]
    instruction: String,

    /// Tinker audio-capable base model
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,

    /// Number of microphone turns; 0 continues until Ctrl+C
    #[arg(long, default_value_t = 1)]
    turns: u64,

    /// Maximum response tokens
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.3)]
    temperature: f32,

    /// Maximum microphone capture duration per turn in seconds
    #[arg(short, long, default_value_t = 30)]
    duration: u64,

    /// Silence after speech that commits the microphone turn
    #[arg(long, default_value_t = 900)]
    silence_timeout_ms: u64,

    /// Energy threshold floor used by microphone VAD
    #[arg(long, default_value_t = 0.01)]
    vad_threshold: f32,

    /// Multiplier applied to the calibrated ambient noise floor
    #[arg(long, default_value_t = 3.0)]
    noise_multiplier: f32,

    /// Ambient calibration duration before each microphone turn
    #[arg(long, default_value_t = 300)]
    calibration_ms: u64,

    /// Print one JSON object per result
    #[arg(long)]
    json: bool,

    /// Speak each Inkling response through the already-running voice daemon
    #[arg(long)]
    speak: bool,

    /// Loaded daemon TTS engine used by --speak
    #[arg(long, value_enum, default_value_t = super::TtsEngine::Kokoro)]
    pub(super) tts_engine: super::TtsEngine,

    /// Voice used by --speak (defaults by engine)
    #[arg(long)]
    voice: Option<String>,

    /// Speech speed used by --speak
    #[arg(long, default_value_t = 1.0)]
    speed: f32,

    /// Use the low-latency Voxtral preset when --tts-engine voxtral is selected
    #[arg(long)]
    voxtral_realtime: bool,

    /// uv executable used to provide the Tinker Python SDK boundary
    #[arg(long, default_value = "uv", hide = true)]
    uv: String,
}

#[derive(Serialize)]
struct WorkerRequest<'a> {
    id: u64,
    audio_path: &'a Path,
    instruction: &'a str,
    max_tokens: usize,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct WorkerMessage {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    id: Option<u64>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    termination: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    startup_ms: Option<f64>,
    #[serde(default)]
    render_ms: Option<f64>,
    #[serde(default)]
    sample_ms: Option<f64>,
    #[serde(default)]
    audio_ms: Option<f64>,
}

#[derive(Serialize)]
struct ResultEvent<'a> {
    turn: u64,
    text: &'a str,
    audio_ms: f64,
    capture_ms: f64,
    render_ms: f64,
    sample_ms: f64,
    round_trip_ms: f64,
    tts_engine: Option<&'a str>,
    tts_first_audio_ms: Option<f64>,
    tts_elapsed_ms: Option<f64>,
}

struct TtsPlaybackMetrics {
    first_audio_ms: f64,
    elapsed_ms: f64,
}

struct DaemonSpeaker {
    client: voice_protocol::client::DaemonClient,
    engine: super::TtsEngine,
    voice: String,
    speed: f32,
    voxtral: super::EffectiveVoxtralOptions,
}

impl DaemonSpeaker {
    fn connect(args: &TinkerArgs) -> Result<Option<Self>, String> {
        if !args.speak {
            return Ok(None);
        }
        super::validate_speed(args.speed)?;
        let voice = super::selected_voice(args.tts_engine, &args.voice);
        super::validate_voice_for_engine(args.tts_engine, &voice)?;
        let mut client = voice_protocol::client::DaemonClient::connect().ok_or_else(|| {
            "--speak requires a running voice daemon; start it with `voice daemon start`"
                .to_string()
        })?;
        if !super::daemon_supports_engine(&mut client, args.tts_engine) {
            return Err(format!(
                "voice daemon does not advertise {} support",
                args.tts_engine.as_str()
            ));
        }
        let voxtral = super::effective_voxtral_options(
            super::VOXTRAL_DEFAULT_MAX_FRAMES,
            7,
            false,
            None,
            args.voxtral_realtime,
        );
        super::validate_effective_voxtral_options(voxtral)?;
        eprintln!(
            "Using voice daemon: {} / {}.",
            args.tts_engine.as_str(),
            voice
        );
        Ok(Some(Self {
            client,
            engine: args.tts_engine,
            voice,
            speed: args.speed,
            voxtral,
        }))
    }

    fn speak(&mut self, text: &str) -> Result<TtsPlaybackMetrics, String> {
        use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, Player};
        use std::num::NonZero;

        const OUTPUT_SAMPLE_RATE: u32 = 48_000;
        const OUTPUT_FRAME_MS: u32 = 20;

        let started = Instant::now();
        let mut output = DeviceSinkBuilder::open_default_sink()
            .map_err(|error| format!("open audio output: {error}"))?;
        output.log_on_drop(false);
        let player = Player::connect_new(output.mixer());
        let channels = NonZero::new(voice_stream::WEBRTC_CHANNELS).unwrap();
        let rate = NonZero::new(OUTPUT_SAMPLE_RATE).unwrap();
        let mut first_audio_ms = None;
        let mut ended = false;
        let mut terminal_error = None;
        let response = self.client.stream_speak_with_options(
            text,
            voice_protocol::client::StreamSpeakOptions {
                voice: Some(&self.voice),
                speed: Some(self.speed as f64),
                sample_rate: Some(OUTPUT_SAMPLE_RATE),
                frame_ms: Some(OUTPUT_FRAME_MS),
                tts: super::daemon_tts_options(
                    self.engine,
                    super::DEFAULT_VOXTRAL_MODEL,
                    self.voxtral,
                ),
            },
            |event| {
                match event {
                    voice_stream::TtsStreamEvent::Audio { frame } => {
                        first_audio_ms.get_or_insert_with(|| {
                            started.elapsed().as_secs_f64() * 1_000.0
                        });
                        player.append(SamplesBuffer::new(
                            channels,
                            rate,
                            super::realtime_frame_samples_f32(&frame),
                        ));
                    }
                    voice_stream::TtsStreamEvent::Ended(_) => ended = true,
                    voice_stream::TtsStreamEvent::Error(error) => {
                        terminal_error = Some(error.message)
                    }
                    voice_stream::TtsStreamEvent::Cancelled(cancelled) => {
                        terminal_error = Some(cancelled.reason)
                    }
                    voice_stream::TtsStreamEvent::Started { .. } => {}
                }
                Ok(())
            },
        )?;
        if let Some(error) = response.error {
            return Err(format!("daemon speech failed: {}", error.message));
        }
        if let Some(error) = terminal_error {
            return Err(format!("daemon streaming speech failed: {error}"));
        }
        if !ended {
            return Err("daemon streaming speech ended without a completion event".to_string());
        }
        while !player.empty() {
            if super::interrupted() {
                player.stop();
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        Ok(TtsPlaybackMetrics {
            first_audio_ms: first_audio_ms
                .ok_or("daemon streaming speech produced no audio frames")?,
            elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        })
    }
}

struct TinkerWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl TinkerWorker {
    fn start(uv: &str, model: &str) -> Result<Self, String> {
        if std::env::var_os("TINKER_API_KEY").is_none() {
            return Err("TINKER_API_KEY is not exported in this shell".to_string());
        }

        let mut child = Command::new(uv)
            .args([
                "run",
                "--quiet",
                "--with",
                "tinker-cookbook",
                "--with",
                "tml-renderers",
                "python",
                "-u",
                "-c",
                WORKER_SOURCE,
                "--",
                model,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("failed to start `{uv}`: {error}"))?;
        let stdin = child.stdin.take().ok_or("worker stdin was not piped")?;
        let stdout = child.stdout.take().ok_or("worker stdout was not piped")?;
        let mut worker = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        let ready = worker.read_message()?;
        match ready.kind.as_str() {
            "ready" => {
                if let Some(ms) = ready.startup_ms {
                    eprintln!("Tinker worker ready in {ms:.0} ms.");
                }
                Ok(worker)
            }
            "error" => Err(ready.error.unwrap_or_else(|| "worker startup failed".into())),
            other => Err(format!("expected worker ready message, got {other:?}")),
        }
    }

    fn sample(&mut self, request: &WorkerRequest<'_>) -> Result<WorkerMessage, String> {
        serde_json::to_writer(&mut self.stdin, request)
            .map_err(|error| format!("encode worker request: {error}"))?;
        self.stdin
            .write_all(b"\n")
            .and_then(|_| self.stdin.flush())
            .map_err(|error| format!("write worker request: {error}"))?;

        loop {
            let response = self.read_message()?;
            if response.kind == "error" {
                return Err(response.error.unwrap_or_else(|| "Tinker request failed".into()));
            }
            if response.kind == "result" && response.id == Some(request.id) {
                return Ok(response);
            }
        }
    }

    fn read_message(&mut self) -> Result<WorkerMessage, String> {
        let mut line = String::new();
        loop {
            line.clear();
            let read = self
                .stdout
                .read_line(&mut line)
                .map_err(|error| format!("read worker response: {error}"))?;
            if read == 0 {
                let status = self.child.try_wait().ok().flatten();
                return Err(format!("Tinker worker exited unexpectedly ({status:?})"));
            }
            match serde_json::from_str(line.trim()) {
                Ok(message) => return Ok(message),
                Err(_) => eprintln!("Tinker worker: {}", line.trim_end()),
            }
        }
    }
}

impl Drop for TinkerWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct TemporaryWav(PathBuf);

impl TemporaryWav {
    fn from_samples(samples: &[f32], sample_rate: u32, turn: u64) -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!(
            "voice-tinker-{}-{turn}.wav",
            std::process::id()
        ));
        let normalized = if sample_rate == TINKER_AUDIO_SAMPLE_RATE {
            samples.to_vec()
        } else {
            voice_stream::resample_linear(samples, sample_rate, TINKER_AUDIO_SAMPLE_RATE)
        };
        write_pcm16_wav(&normalized, TINKER_AUDIO_SAMPLE_RATE, &path)?;
        Ok(Self(path))
    }
}

impl Drop for TemporaryWav {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub fn run(args: TinkerArgs) -> Result<(), String> {
    validate_args(&args)?;
    let mut speaker = DaemonSpeaker::connect(&args)?;
    eprintln!("Starting warm Tinker worker for {}...", args.model);
    let mut worker = TinkerWorker::start(&args.uv, &args.model)?;

    if let Some(audio) = &args.audio {
        let loaded = super::listen::load_transcription_audio(audio)?;
        let wav = TemporaryWav::from_samples(&loaded.samples, loaded.sample_rate, 1)?;
        return run_turn(&mut worker, &mut speaker, &args, 1, &wav.0, 0.0);
    }

    let mic = super::listen::WarmMic::open()?;
    let mut turn = 1;
    while args.turns == 0 || turn <= args.turns {
        if super::interrupted() {
            break;
        }
        eprintln!("Turn {turn}: speak after the ding, then pause.");
        let capture_started = Instant::now();
        let (samples, sample_rate) = mic.record_vad(
            args.duration * 1_000,
            args.silence_timeout_ms,
            args.vad_threshold,
            args.noise_multiplier,
            args.calibration_ms,
        )?;
        let capture_ms = capture_started.elapsed().as_secs_f64() * 1_000.0;
        if super::interrupted() {
            break;
        }
        if samples.is_empty() {
            eprintln!("No audio captured; skipping turn.");
            turn += 1;
            continue;
        }
        let wav = TemporaryWav::from_samples(&samples, sample_rate, turn)?;
        run_turn(
            &mut worker,
            &mut speaker,
            &args,
            turn,
            &wav.0,
            capture_ms,
        )?;
        turn += 1;
    }
    Ok(())
}

fn run_turn(
    worker: &mut TinkerWorker,
    speaker: &mut Option<DaemonSpeaker>,
    args: &TinkerArgs,
    turn: u64,
    audio_path: &Path,
    capture_ms: f64,
) -> Result<(), String> {
    if !audio_path.is_file() {
        return Err(format!("audio file does not exist: {}", audio_path.display()));
    }
    eprintln!("Sending turn {turn} to Tinker...");
    let started = Instant::now();
    let response = worker.sample(&WorkerRequest {
        id: turn,
        audio_path,
        instruction: &args.instruction,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
    })?;
    let round_trip_ms = started.elapsed().as_secs_f64() * 1_000.0;
    if speaker.is_some() && response.termination.as_deref() == Some("malformed") {
        return Err(format!(
            "Inkling did not finish a clean final answer within {} tokens; refusing to speak the partial response (increase --max-tokens)",
            args.max_tokens
        ));
    }
    let text = response.text.as_deref().unwrap_or("");
    if !args.json {
        println!("{text}");
        std::io::stdout()
            .flush()
            .map_err(|error| format!("flush response text: {error}"))?;
    }
    let tts_metrics = speaker
        .as_mut()
        .map(|speaker| speaker.speak(text))
        .transpose()?;
    let event = ResultEvent {
        turn,
        text,
        audio_ms: response.audio_ms.unwrap_or_default(),
        capture_ms,
        render_ms: response.render_ms.unwrap_or_default(),
        sample_ms: response.sample_ms.unwrap_or_default(),
        round_trip_ms,
        tts_engine: speaker.as_ref().map(|speaker| speaker.engine.as_str()),
        tts_first_audio_ms: tts_metrics.as_ref().map(|metrics| metrics.first_audio_ms),
        tts_elapsed_ms: tts_metrics.as_ref().map(|metrics| metrics.elapsed_ms),
    };
    if args.json {
        println!(
            "{}",
            serde_json::to_string(&event).map_err(|error| error.to_string())?
        );
    } else {
        eprintln!(
            "audio={:.0}ms render={:.0}ms sample={:.0}ms round-trip={round_trip_ms:.0}ms",
            event.audio_ms, event.render_ms, event.sample_ms
        );
    }
    Ok(())
}

fn validate_args(args: &TinkerArgs) -> Result<(), String> {
    if args.audio.is_some() && args.turns != 1 {
        return Err("--audio only supports --turns 1".to_string());
    }
    if args.audio.is_none() && args.duration == 0 {
        return Err("--duration must be greater than zero".to_string());
    }
    if args.silence_timeout_ms == 0 {
        return Err("--silence-timeout-ms must be greater than zero".to_string());
    }
    if !args.vad_threshold.is_finite() || args.vad_threshold < 0.0 {
        return Err("--vad-threshold must be a finite non-negative number".to_string());
    }
    if !args.noise_multiplier.is_finite() || args.noise_multiplier <= 0.0 {
        return Err("--noise-multiplier must be a finite positive number".to_string());
    }
    if args.max_tokens == 0 {
        return Err("--max-tokens must be greater than zero".to_string());
    }
    if !args.temperature.is_finite() || args.temperature < 0.0 {
        return Err("--temperature must be a finite non-negative number".to_string());
    }
    if args.speak {
        super::validate_speed(args.speed)?;
    }
    Ok(())
}

fn write_pcm16_wav(samples: &[f32], sample_rate: u32, path: &Path) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|error| format!("create WAV {}: {error}", path.display()))?;
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer
            .write_sample(pcm)
            .map_err(|error| format!("write WAV sample: {error}"))?;
    }
    writer
        .finalize()
        .map_err(|error| format!("finalize WAV {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parses_tinker_audio_experiment_options() {
        let args = super::super::Args::parse_from([
            "voice",
            "tinker",
            "--audio",
            "turn.wav",
            "--instruction",
            "Transcribe exactly",
            "--max-tokens",
            "80",
            "--temperature",
            "0",
            "--json",
            "--speak",
            "--tts-engine",
            "voxtral",
            "--voxtral-realtime",
        ]);
        let Some(super::super::Command::Tinker(args)) = args.command else {
            panic!("expected tinker command");
        };
        assert_eq!(args.audio, Some(PathBuf::from("turn.wav")));
        assert_eq!(args.instruction, "Transcribe exactly");
        assert_eq!(args.max_tokens, 80);
        assert_eq!(args.temperature, 0.0);
        assert!(args.json);
        assert!(args.speak);
        assert_eq!(args.tts_engine, super::super::TtsEngine::Voxtral);
        assert!(args.voxtral_realtime);
    }

    #[test]
    fn worker_message_accepts_ready_and_result_shapes() {
        let ready: WorkerMessage =
            serde_json::from_str(r#"{"type":"ready","startup_ms":123.4}"#).unwrap();
        assert_eq!(ready.kind, "ready");
        assert_eq!(ready.startup_ms, Some(123.4));

        let result: WorkerMessage = serde_json::from_str(
            r#"{"type":"result","id":2,"text":"hello","termination":"stop_sequence","audio_ms":500,"render_ms":3,"sample_ms":900}"#,
        )
        .unwrap();
        assert_eq!(result.id, Some(2));
        assert_eq!(result.text.as_deref(), Some("hello"));
        assert_eq!(result.termination.as_deref(), Some("stop_sequence"));
    }

    #[test]
    fn pcm16_wav_is_python_wave_compatible_shape() {
        let path = std::env::temp_dir().join(format!(
            "voice-tinker-wav-test-{}.wav",
            std::process::id()
        ));
        write_pcm16_wav(&[-1.0, 0.0, 1.0], 16_000, &path).unwrap();
        let reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 16_000);
        assert_eq!(reader.spec().bits_per_sample, 16);
        assert_eq!(reader.len(), 3);
        let _ = std::fs::remove_file(path);
    }
}
