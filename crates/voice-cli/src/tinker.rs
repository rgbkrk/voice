//! Rust-first microphone turns for Tinker's audio-capable Inkling model.
//!
//! Capture, VAD, DMel encoding, TMLv0 rendering, Tinker sampling, result
//! reporting, and optional local speech all stay in Rust.

use serde::Serialize;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tinkernel::audio::InklingAudioEncoder;
use tinkernel::renderer::{Author, Content, Message, TmlV0Renderer};
use tinkernel::types::{SamplingParams, StopCondition, StopReason};
use tinkernel::{FutureOptions, SamplingClient, TinkerClient};

const DEFAULT_MODEL: &str = "thinkingmachines/Inkling";
const TINKER_AUDIO_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_INSTRUCTION: &str = "Respond naturally and concisely to the speaker. Use both the words and audible delivery, including tone, hesitation, emphasis, and emotion when relevant. Do not provide a transcript unless asked.";
const TINKER_REQUEST_TIMEOUT: Duration = Duration::from_secs(10 * 60);

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

    /// Maximum response tokens, including Inkling's reasoning
    #[arg(long, default_value_t = 512)]
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

    /// Save each normalized input WAV and result JSON in this directory
    #[arg(long, value_name = "DIR")]
    output_dir: Option<PathBuf>,

    /// Print Inkling's completed reasoning trace dimmed on stderr
    #[arg(long)]
    show_thinking: bool,

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
}

struct NativeResponse {
    text: String,
    thinking: String,
    termination: &'static str,
    complete: bool,
    response_tokens: usize,
    audio_ms: f64,
    encode_ms: f64,
    render_ms: f64,
    sample_ms: f64,
}

#[derive(Serialize)]
struct ResultEvent<'a> {
    turn: u64,
    model: &'a str,
    text: &'a str,
    thinking: &'a str,
    termination: &'a str,
    complete: bool,
    response_tokens: usize,
    max_tokens: usize,
    temperature: f32,
    audio_ms: f64,
    capture_ms: f64,
    encode_ms: f64,
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

struct NativeTinker {
    encoder: InklingAudioEncoder,
    renderer: TmlV0Renderer,
    sampler: SamplingClient,
    runtime: tokio::runtime::Runtime,
}

impl NativeTinker {
    fn start(model: &str) -> Result<Self, String> {
        if std::env::var_os("TINKER_API_KEY").is_none() {
            return Err("TINKER_API_KEY is not exported in this shell".to_string());
        }

        let started = Instant::now();
        let encoder = InklingAudioEncoder::new();
        let renderer = TmlV0Renderer::new()
            .map_err(|error| format!("initialize Inkling renderer: {error}"))?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| format!("initialize Tinker runtime: {error}"))?;
        let sampler = runtime
            .block_on(async {
                let client = TinkerClient::from_env().await?;
                client.sampling_client(model).await
            })
            .map_err(|error| format!("initialize Tinker sampling client: {error}"))?;
        eprintln!(
            "Tinker client ready in {:.0} ms.",
            started.elapsed().as_secs_f64() * 1_000.0
        );
        Ok(Self {
            encoder,
            renderer,
            sampler,
            runtime,
        })
    }

    fn sample(
        &self,
        samples: &[f32],
        instruction: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<NativeResponse, String> {
        let audio_ms = samples.len() as f64 / f64::from(TINKER_AUDIO_SAMPLE_RATE) * 1_000.0;
        let encode_started = Instant::now();
        let dmel = self
            .encoder
            .encode(samples)
            .map_err(|error| format!("encode Inkling audio: {error}"))?;
        let encode_ms = encode_started.elapsed().as_secs_f64() * 1_000.0;

        let render_started = Instant::now();
        let prompt = self.renderer.render_for_completion(&[
            Message::user(instruction),
            Message::user_audio(dmel.into_tensor_container())
                .map_err(|error| format!("render Inkling audio message: {error}"))?,
        ]);
        let render_ms = render_started.elapsed().as_secs_f64() * 1_000.0;

        let sample_started = Instant::now();
        let response = self
            .runtime
            .block_on(async {
                self.sampler
                    .sample(
                        prompt,
                        SamplingParams {
                            max_tokens: Some(max_tokens),
                            temperature,
                            stop: Some(StopCondition::Tokens(
                                self.renderer.stop_tokens().to_vec(),
                            )),
                            ..SamplingParams::default()
                        },
                    )
                    .await?
                    .with_options(FutureOptions {
                        timeout: Some(TINKER_REQUEST_TIMEOUT),
                        ..FutureOptions::default()
                    })
                    .await_result()
                    .await
            })
            .map_err(|error| format!("sample Inkling audio: {error}"))?;
        let sample_ms = sample_started.elapsed().as_secs_f64() * 1_000.0;
        let sequence = response
            .sequences
            .into_iter()
            .next()
            .ok_or("Tinker returned no sampled sequences")?;
        let response_tokens = sequence.tokens.len();
        let (text, thinking) = completion_text(&self.renderer, &sequence.tokens)?;
        let complete = sequence.stop_reason == StopReason::Stop && !text.trim().is_empty();
        let termination = match (sequence.stop_reason, text.trim().is_empty()) {
            (StopReason::Stop, false) => "stop",
            (StopReason::Stop, true) => "missing_final_text",
            (StopReason::Length, _) => "length",
        };

        Ok(NativeResponse {
            text,
            thinking,
            termination,
            complete,
            response_tokens,
            audio_ms,
            encode_ms,
            render_ms,
            sample_ms,
        })
    }
}

fn completion_text(
    renderer: &TmlV0Renderer,
    tokens: &[u32],
) -> Result<(String, String), String> {
    let mut text = String::new();
    let mut thinking = String::new();
    for message in renderer
        .parse_completion(tokens)
        .map_err(|error| format!("parse Inkling completion: {error}"))?
    {
        match message.into_parts() {
            (Author::Model, Content::Text(part)) => text.push_str(&part),
            (Author::Model, Content::Thinking(part)) => thinking.push_str(&part),
            _ => {}
        }
    }
    Ok((text, thinking))
}

fn normalize_samples(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    if sample_rate == TINKER_AUDIO_SAMPLE_RATE {
        samples.to_vec()
    } else {
        voice_stream::resample_linear(samples, sample_rate, TINKER_AUDIO_SAMPLE_RATE)
    }
}

pub fn run(args: TinkerArgs) -> Result<(), String> {
    validate_args(&args)?;
    let mut speaker = DaemonSpeaker::connect(&args)?;
    eprintln!("Starting native Tinker client for {}...", args.model);
    let tinker = NativeTinker::start(&args.model)?;

    if let Some(audio) = &args.audio {
        let loaded = super::listen::load_transcription_audio(audio)?;
        let samples = normalize_samples(&loaded.samples, loaded.sample_rate);
        return run_turn(&tinker, &mut speaker, &args, 1, &samples, 0.0);
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
        let samples = normalize_samples(&samples, sample_rate);
        run_turn(
            &tinker,
            &mut speaker,
            &args,
            turn,
            &samples,
            capture_ms,
        )?;
        turn += 1;
    }
    Ok(())
}

fn run_turn(
    tinker: &NativeTinker,
    speaker: &mut Option<DaemonSpeaker>,
    args: &TinkerArgs,
    turn: u64,
    samples: &[f32],
    capture_ms: f64,
) -> Result<(), String> {
    if samples.is_empty() {
        return Err("cannot send an empty audio turn to Tinker".to_string());
    }
    eprintln!("Sending turn {turn} to Tinker...");
    let started = Instant::now();
    let response = tinker.sample(
        samples,
        &args.instruction,
        args.max_tokens
            .try_into()
            .map_err(|_| "--max-tokens is too large for Tinker")?,
        args.temperature,
    )?;
    let round_trip_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let termination = response.termination;
    let complete = response.complete;
    let text = response.text.as_str();
    let thinking = response.thinking.as_str();
    let response_tokens = response.response_tokens;
    if args.show_thinking {
        print_thinking(thinking);
    }
    if !args.json && complete {
        println!("{text}");
        std::io::stdout()
            .flush()
            .map_err(|error| format!("flush response text: {error}"))?;
    }
    if !complete {
        eprintln!(
            "Inkling ended without a clean final answer after {response_tokens} response tokens (budget {}); the partial response will not be spoken{}.",
            args.max_tokens,
            args.output_dir
                .as_ref()
                .map(|_| " and is available in the saved result JSON")
                .unwrap_or("")
        );
    }
    let tts_metrics = if complete {
        speaker
            .as_mut()
            .map(|speaker| speaker.speak(text))
            .transpose()?
    } else {
        None
    };
    let event = ResultEvent {
        turn,
        model: &args.model,
        text,
        thinking,
        termination,
        complete,
        response_tokens,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        audio_ms: response.audio_ms,
        capture_ms,
        encode_ms: response.encode_ms,
        render_ms: response.render_ms,
        sample_ms: response.sample_ms,
        round_trip_ms,
        tts_engine: speaker.as_ref().map(|speaker| speaker.engine.as_str()),
        tts_first_audio_ms: tts_metrics.as_ref().map(|metrics| metrics.first_audio_ms),
        tts_elapsed_ms: tts_metrics.as_ref().map(|metrics| metrics.elapsed_ms),
    };
    if let Some(output_dir) = &args.output_dir {
        save_turn_artifacts(output_dir, turn, samples, &event)?;
    }
    if args.json {
        println!(
            "{}",
            serde_json::to_string(&event).map_err(|error| error.to_string())?
        );
    } else {
        eprintln!(
            "audio={:.0}ms encode={:.0}ms render={:.0}ms sample={:.0}ms round-trip={round_trip_ms:.0}ms",
            event.audio_ms, event.encode_ms, event.render_ms, event.sample_ms
        );
    }
    Ok(())
}

fn print_thinking(thinking: &str) {
    if thinking.is_empty() {
        return;
    }
    if std::io::stderr().is_terminal() {
        eprintln!("\x1b[2;3;90mthinking\n{thinking}\x1b[0m");
    } else {
        eprintln!("thinking:\n{thinking}");
    }
}

fn save_turn_artifacts(
    output_dir: &Path,
    turn: u64,
    samples: &[f32],
    event: &ResultEvent<'_>,
) -> Result<(), String> {
    std::fs::create_dir_all(output_dir).map_err(|error| {
        format!(
            "create Tinker artifact directory {}: {error}",
            output_dir.display()
        )
    })?;
    let stem = format!("turn-{turn:04}");
    let wav_path = output_dir.join(format!("{stem}.wav"));
    let json_path = output_dir.join(format!("{stem}.json"));
    write_pcm16_wav(samples, TINKER_AUDIO_SAMPLE_RATE, &wav_path)?;
    let mut json = serde_json::to_vec_pretty(event)
        .map_err(|error| format!("encode Tinker result artifact: {error}"))?;
    json.push(b'\n');
    std::fs::write(&json_path, json)
        .map_err(|error| format!("save Tinker result JSON {}: {error}", json_path.display()))?;
    eprintln!("Saved turn {turn} artifacts to {}", output_dir.display());
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
    fn separates_native_thinking_from_final_text() {
        use tinkernel::renderer::token;

        let renderer = TmlV0Renderer::new().unwrap();
        let mut tokens = vec![token::MESSAGE_MODEL, token::CONTENT_THINKING];
        tokens.extend(renderer.encode_ordinary("considering"));
        tokens.extend([
            token::END_MESSAGE,
            token::MESSAGE_MODEL,
            token::CONTENT_TEXT,
        ]);
        tokens.extend(renderer.encode_ordinary("hello"));
        tokens.extend([token::END_MESSAGE, token::CONTENT_MODEL_END_SAMPLING]);

        let (text, thinking) = completion_text(&renderer, &tokens).unwrap();
        assert_eq!(text, "hello");
        assert_eq!(thinking, "considering");
    }

    #[test]
    fn tinker_defaults_allow_room_for_reasoning() {
        let args = super::super::Args::parse_from(["voice", "tinker"]);
        let Some(super::super::Command::Tinker(args)) = args.command else {
            panic!("expected tinker command");
        };
        assert_eq!(args.max_tokens, 512);
        assert_eq!(args.output_dir, None);
        assert!(!args.show_thinking);
    }

    #[test]
    fn saves_normalized_audio_and_result_artifacts() {
        let output_dir = std::env::temp_dir().join(format!(
            "voice-tinker-artifact-test-{}",
            std::process::id()
        ));
        let event = ResultEvent {
            turn: 2,
            model: DEFAULT_MODEL,
            text: "partial",
            thinking: "I should reason about this.",
            termination: "malformed",
            complete: false,
            response_tokens: 512,
            max_tokens: 512,
            temperature: 0.3,
            audio_ms: 0.2,
            capture_ms: 10.0,
            encode_ms: 1.5,
            render_ms: 4.0,
            sample_ms: 2_000.0,
            round_trip_ms: 2_005.0,
            tts_engine: None,
            tts_first_audio_ms: None,
            tts_elapsed_ms: None,
        };
        save_turn_artifacts(&output_dir, 2, &[-0.5, 0.0, 0.5], &event).unwrap();

        assert!(output_dir.join("turn-0002.wav").is_file());
        let saved: serde_json::Value = serde_json::from_slice(
            &std::fs::read(output_dir.join("turn-0002.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(saved["complete"], false);
        assert_eq!(saved["response_tokens"], 512);
        assert_eq!(saved["thinking"], "I should reason about this.");

        let _ = std::fs::remove_dir_all(output_dir);
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
