mod jsonrpc;
mod listen;
mod mcp;

use clap::{Parser, ValueEnum};
use pulldown_cmark::{Event, Options, Parser as MdParser, Tag, TagEnd};
use serde::Serialize;
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const MODEL_REPO: &str = "prince-canuma/Kokoro-82M";
const DEFAULT_VOXTRAL_MODEL: &str = voice_voxtral::DEFAULT_REPO;
const KOKORO_DEFAULT_VOICE: &str = "af_heart";
const VOXTRAL_DEFAULT_VOICE: &str = "casual_male";
const VOXTRAL_DEFAULT_MAX_FRAMES: usize = 256;
const VOXTRAL_REALTIME_MAX_FRAMES: usize = 56;
const VOXTRAL_REALTIME_STREAM_BEGIN_FRAMES: usize = 2;

static QUIET: AtomicBool = AtomicBool::new(false);
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Check if Ctrl+C has been pressed.
fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

/// Print an informational message to stderr, unless `--quiet` is set.
macro_rules! info {
    ($($arg:tt)*) => {
        if !QUIET.load(Ordering::Relaxed) {
            eprintln!($($arg)*);
        }
    };
}

#[derive(Parser, Debug)]
#[command(
    name = "voice",
    about = "Rust TTS & STT on Apple Silicon",
    after_help = "Examples:\n  \
                  voice Hello world\n  \
                  voxtral Hello from Voxtral\n  \
                  voice say -v am_adam \"How are you today?\"\n  \
                  voxtral say -v casual_male \"How are you today?\"\n  \
                  echo \"Hello\" | voice say\n  \
                  voice say -f speech.txt -o output.wav\n  \
                  voice say --format ogg-opus -o reply.ogg \"Hello\"\n  \
                  voice say --phonemes \"hɛloʊ wɜːld\"\n  \
                  voice say --markdown -f post.mdx\n  \
                  voice phonemes \"ChatGPT uses RuntimeStateDoc\"\n  \
                  voice stream --json \"Hello world\"\n  \
                  voice stream --output reply.ogg --format ogg-opus \"Hello world\"\n  \
                  voice stream-contract\n  \
                  voice stream-transcribe recording.ogg\n  \
                  voice listen\n  \
                  voice listen --continuous\n  \
                  voice transcribe recording.ogg\n  \
                  voice bench tts --engine kokoro --engine voxtral --runs 2 \"Hello world\"\n  \
                  voice daemon start --tts-only\n  \
                  voice daemon status\n  \
                  voice serve -v am_michael"
)]
struct Args {
    /// Suppress progress output (phonemes, chunk info, loading messages).
    /// Errors are always printed.
    #[arg(short, long, global = true)]
    quiet: bool,

    #[command(subcommand)]
    command: Option<Command>,

    /// Text to speak (shorthand for `voice say <text>` or `voxtral say <text>`)
    #[arg(trailing_var_arg = true)]
    text: Vec<String>,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Speak text aloud (default when no subcommand given)
    Say(SayArgs),

    /// Convert text to phoneme chunks without synthesis
    Phonemes(PhonemesArgs),

    /// Stream TTS audio chunks from the voice daemon
    Stream(StreamArgs),

    /// Replay an audio file or raw PCM through daemon streaming STT
    StreamTranscribe(StreamTranscribeArgs),

    /// Print the machine-readable WebRTC sidecar stream contract
    StreamContract,

    /// Speak text aloud, then listen for a response (speak + listen in one shot)
    Converse(ConverseArgs),

    /// Record from microphone and transcribe (speech-to-text)
    Listen(ListenArgs),

    /// Transcribe an audio file
    Transcribe(TranscribeArgs),

    /// Run as a JSON-RPC 2.0 server on stdin/stdout
    Serve(ServeArgs),

    /// Run as an MCP (Model Context Protocol) server on stdin/stdout
    Mcp(ServeArgs),

    /// Inspect and control a running voice daemon
    Daemon(DaemonArgs),

    /// Benchmark TTS latency without playback
    Bench(BenchArgs),
}

#[derive(clap::Args, Debug)]
struct PhonemesArgs {
    /// Text to convert
    #[arg(trailing_var_arg = true)]
    text: Vec<String>,

    /// Read text from a file (use - for stdin)
    #[arg(short = 'f', long = "input-file")]
    input_file: Option<PathBuf>,

    /// Strip markdown/MDX formatting before conversion
    #[arg(long)]
    markdown: bool,

    /// Word substitutions (pre-processing), e.g. --sub nteract=enteract
    #[arg(long = "sub", value_name = "WORD=REPLACEMENT")]
    subs: Vec<String>,

    /// Load substitutions from a file (one WORD=REPLACEMENT per line, # comments).
    /// If not set, .voice-subs is auto-discovered from the working directory upward.
    #[arg(long = "sub-file", value_name = "PATH")]
    sub_file: Option<PathBuf>,

    /// Print a JSON object with preprocessed text and phoneme chunks
    #[arg(long)]
    json: bool,
}

#[derive(clap::Args, Debug)]
struct SayArgs {
    /// Text to speak
    #[arg(trailing_var_arg = true)]
    text: Vec<String>,

    /// Read text from a file (use - for stdin)
    #[arg(short = 'f', long = "input-file", conflicts_with = "phonemes")]
    input_file: Option<PathBuf>,

    /// Raw phoneme string (IPA)
    #[arg(long)]
    phonemes: Option<String>,

    /// TTS engine to use. Defaults to kokoro for `voice`, voxtral for `voxtral`
    #[arg(long, value_enum, default_value_t = TtsEngine::Kokoro, hide_default_value = true)]
    engine: TtsEngine,

    /// Voice name. Defaults to af_heart for Kokoro, casual_male for Voxtral
    #[arg(short, long)]
    voice: Option<String>,

    /// Voxtral model path or HuggingFace repo
    #[arg(long = "voxtral-model", default_value = DEFAULT_VOXTRAL_MODEL)]
    voxtral_model: String,

    /// Maximum Voxtral audio frames to generate
    #[arg(long = "voxtral-max-frames", default_value_t = VOXTRAL_DEFAULT_MAX_FRAMES)]
    voxtral_max_frames: usize,

    /// Voxtral flow-matching steps per frame
    #[arg(long = "voxtral-flow-steps", default_value_t = 7)]
    voxtral_flow_steps: usize,

    /// Enable Voxtral language KV cache (off by default)
    #[arg(long = "voxtral-kv-cache")]
    voxtral_kv_cache: bool,

    /// Use the current opt-in low-latency Voxtral preset
    #[arg(long = "voxtral-realtime")]
    voxtral_realtime: bool,

    /// Normalize compact Voxtral numeric forms such as versions and times before synthesis
    #[arg(long = "voxtral-normalize-text")]
    voxtral_normalize_text: bool,

    /// Apply known Voxtral pronunciation aliases before synthesis
    #[arg(long = "voxtral-pronunciation-aliases")]
    voxtral_pronunciation_aliases: bool,

    /// Choose Voxtral max frames from the post-normalization synthesis text
    #[arg(long = "voxtral-auto-max-frames")]
    voxtral_auto_max_frames: bool,

    /// Override Voxtral's initial streaming codec chunk size for daemon playback
    #[arg(long = "voxtral-stream-begin-frames")]
    voxtral_stream_begin_frames: Option<usize>,

    /// Write audio to file instead of playing
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Output container/codec for --output. Defaults from extension: .wav, .ogg, .opus
    #[arg(long = "format", value_enum, requires = "output")]
    format: Option<SayOutputFormat>,

    /// Speech speed factor (1.0 = normal)
    #[arg(short, long, default_value = "1.0")]
    speed: f32,

    /// Use deterministic synthesis for reproducible evaluation output
    #[arg(long)]
    deterministic: bool,

    /// Strip markdown/MDX formatting before speaking
    #[arg(long)]
    markdown: bool,

    /// Word substitutions (pre-processing), e.g. --sub nteract=enteract
    #[arg(long = "sub", value_name = "WORD=REPLACEMENT")]
    subs: Vec<String>,

    /// Load substitutions from a file (one WORD=REPLACEMENT per line, # comments).
    /// If not set, .voice-subs is auto-discovered from the working directory upward.
    #[arg(long = "sub-file", value_name = "PATH")]
    sub_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SayOutputFormat {
    Wav,
    OggOpus,
}

impl From<SayOutputFormat> for voice_audio::AudioOutputFormat {
    fn from(format: SayOutputFormat) -> Self {
        match format {
            SayOutputFormat::Wav => Self::Wav,
            SayOutputFormat::OggOpus => Self::OggOpus,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum StreamOutputFormat {
    OggOpus,
}

impl From<StreamOutputFormat> for voice_audio::AudioOutputFormat {
    fn from(format: StreamOutputFormat) -> Self {
        match format {
            StreamOutputFormat::OggOpus => Self::OggOpus,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TtsEngine {
    Kokoro,
    Voxtral,
}

impl TtsEngine {
    fn as_str(self) -> &'static str {
        match self {
            Self::Kokoro => "kokoro",
            Self::Voxtral => "voxtral",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InvocationProfile {
    Voice,
    Voxtral,
}

impl InvocationProfile {
    fn default_tts_engine(self) -> TtsEngine {
        match self {
            Self::Voice => TtsEngine::Kokoro,
            Self::Voxtral => TtsEngine::Voxtral,
        }
    }

    fn help_name(self) -> &'static str {
        match self {
            Self::Voice => "voice",
            Self::Voxtral => "voxtral",
        }
    }
}

fn invocation_profile_from_arg0(arg0: Option<&OsStr>) -> InvocationProfile {
    let Some(arg0) = arg0 else {
        return InvocationProfile::Voice;
    };
    let stem = Path::new(arg0)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    if stem == "voxtral" {
        InvocationProfile::Voxtral
    } else {
        InvocationProfile::Voice
    }
}

fn args_contain_engine_flag(raw_args: &[OsString]) -> bool {
    raw_args.iter().skip(1).any(|arg| {
        arg.to_str()
            .is_some_and(|arg| arg == "--engine" || arg.starts_with("--engine="))
    })
}

fn apply_invocation_defaults(args: &mut Args, profile: InvocationProfile, engine_explicit: bool) {
    if profile != InvocationProfile::Voxtral || engine_explicit {
        return;
    }

    match &mut args.command {
        Some(Command::Say(args)) => args.engine = TtsEngine::Voxtral,
        Some(Command::Stream(args)) => args.engine = TtsEngine::Voxtral,
        Some(Command::Converse(args)) => args.engine = TtsEngine::Voxtral,
        Some(Command::Daemon(DaemonArgs {
            command: Some(DaemonCommand::SetVoice { engine, voice: _ }),
        })) => *engine = TtsEngine::Voxtral,
        _ => {}
    }
}

fn default_voice_for_engine(engine: TtsEngine) -> &'static str {
    match engine {
        TtsEngine::Kokoro => KOKORO_DEFAULT_VOICE,
        TtsEngine::Voxtral => VOXTRAL_DEFAULT_VOICE,
    }
}

fn selected_voice(engine: TtsEngine, voice: &Option<String>) -> String {
    voice
        .clone()
        .unwrap_or_else(|| default_voice_for_engine(engine).to_string())
}

fn validate_voice_for_engine(engine: TtsEngine, voice: &str) -> Result<(), String> {
    let known = match engine {
        TtsEngine::Kokoro => voice_tts::catalog::ALL_VOICES
            .iter()
            .any(|candidate| candidate.id == voice),
        TtsEngine::Voxtral => voice_voxtral::get_preset_voice(voice).is_some(),
    };
    if known {
        Ok(())
    } else {
        Err(format!("Unknown {} voice: {}", engine.as_str(), voice))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EffectiveVoxtralOptions {
    max_frames: usize,
    flow_steps: usize,
    kv_cache: bool,
    stream_begin_frames: Option<usize>,
}

fn effective_voxtral_options(
    max_frames: usize,
    flow_steps: usize,
    kv_cache: bool,
    stream_begin_frames: Option<usize>,
    realtime: bool,
) -> EffectiveVoxtralOptions {
    EffectiveVoxtralOptions {
        max_frames: if realtime && max_frames == VOXTRAL_DEFAULT_MAX_FRAMES {
            VOXTRAL_REALTIME_MAX_FRAMES
        } else {
            max_frames
        },
        flow_steps,
        kv_cache: kv_cache || realtime,
        stream_begin_frames: stream_begin_frames
            .or_else(|| realtime.then_some(VOXTRAL_REALTIME_STREAM_BEGIN_FRAMES)),
    }
}

fn validate_effective_voxtral_options(options: EffectiveVoxtralOptions) -> Result<(), String> {
    if options.max_frames == 0 {
        return Err("--voxtral-max-frames must be greater than zero".to_string());
    }
    if options.flow_steps == 0 {
        return Err("--voxtral-flow-steps must be greater than zero".to_string());
    }
    validate_voxtral_stream_begin_frames(options.stream_begin_frames)
}

fn validate_speed(speed: f32) -> Result<(), String> {
    if speed.is_finite() && speed > 0.0 && speed <= 5.0 {
        Ok(())
    } else {
        Err("speed must be between 0 (exclusive) and 5 (inclusive)".to_string())
    }
}

fn apply_auto_voxtral_max_frames(
    mut options: EffectiveVoxtralOptions,
    text: &str,
    enabled: bool,
) -> EffectiveVoxtralOptions {
    if enabled {
        let suggested = voice_voxtral::suggest_max_frames_for_text(text);
        options.max_frames = if options.max_frames == VOXTRAL_DEFAULT_MAX_FRAMES {
            suggested
        } else {
            options.max_frames.max(suggested)
        };
    }
    options
}

impl SayArgs {
    fn effective_voxtral_options(&self) -> EffectiveVoxtralOptions {
        effective_voxtral_options(
            self.voxtral_max_frames,
            self.voxtral_flow_steps,
            self.voxtral_kv_cache,
            self.voxtral_stream_begin_frames,
            self.voxtral_realtime,
        )
    }
}

impl StreamArgs {
    fn effective_voxtral_options(&self) -> EffectiveVoxtralOptions {
        effective_voxtral_options(
            self.voxtral_max_frames,
            self.voxtral_flow_steps,
            self.voxtral_kv_cache,
            self.voxtral_stream_begin_frames,
            self.voxtral_realtime,
        )
    }
}

impl ConverseArgs {
    fn effective_voxtral_options(&self) -> EffectiveVoxtralOptions {
        effective_voxtral_options(
            self.voxtral_max_frames,
            self.voxtral_flow_steps,
            self.voxtral_kv_cache,
            self.voxtral_stream_begin_frames,
            self.voxtral_realtime,
        )
    }
}

impl BenchTtsArgs {
    fn effective_voxtral_options(&self) -> EffectiveVoxtralOptions {
        effective_voxtral_options(
            self.voxtral_max_frames,
            self.voxtral_flow_steps,
            self.voxtral_kv_cache,
            self.voxtral_stream_begin_frames,
            self.voxtral_realtime,
        )
    }
}

fn daemon_tts_options<'a>(
    engine: TtsEngine,
    voxtral_model: &'a str,
    voxtral: EffectiveVoxtralOptions,
) -> voice_protocol::client::TtsRequestOptions<'a> {
    voice_protocol::client::TtsRequestOptions {
        engine: Some(engine.as_str()),
        voxtral_model: (engine == TtsEngine::Voxtral).then_some(voxtral_model),
        voxtral_max_frames: (engine == TtsEngine::Voxtral).then_some(voxtral.max_frames),
        voxtral_flow_steps: (engine == TtsEngine::Voxtral).then_some(voxtral.flow_steps),
        voxtral_stream_begin_frames: (engine == TtsEngine::Voxtral)
            .then_some(voxtral.stream_begin_frames)
            .flatten(),
        voxtral_kv_cache: engine == TtsEngine::Voxtral && voxtral.kv_cache,
    }
}

fn daemon_supports_engine(
    daemon: &mut voice_protocol::client::DaemonClient,
    engine: TtsEngine,
) -> bool {
    if engine == TtsEngine::Kokoro {
        return true;
    }
    daemon
        .list_voices()
        .ok()
        .and_then(|response| response.result)
        .and_then(|result| result.get("voices").cloned())
        .and_then(|voices| voices.as_array().cloned())
        .is_some_and(|voices| {
            voices.iter().any(|voice| {
                voice.get("engine").and_then(|value| value.as_str()) == Some(engine.as_str())
            })
        })
}

fn voxtral_generation_options(
    max_frames: usize,
    flow_steps: usize,
    kv_cache: bool,
    synchronize_trace: bool,
) -> voice_voxtral::VoxtralGenerationOptions {
    voice_voxtral::VoxtralGenerationOptions {
        max_frames,
        flow_steps,
        use_kv_cache: kv_cache,
        synchronize_trace,
        ..Default::default()
    }
}

#[derive(clap::Args, Debug)]
struct StreamArgs {
    /// Text to stream
    #[arg(trailing_var_arg = true)]
    text: Vec<String>,

    /// Read text from a file (use - for stdin)
    #[arg(short = 'f', long = "input-file")]
    input_file: Option<PathBuf>,

    /// TTS engine to use. Defaults to kokoro for `voice`, voxtral for `voxtral`
    #[arg(long, value_enum, default_value_t = TtsEngine::Kokoro, hide_default_value = true)]
    engine: TtsEngine,

    /// Voice name. Defaults to af_heart for Kokoro, casual_male for Voxtral
    #[arg(short, long)]
    voice: Option<String>,

    /// Voxtral model path or HuggingFace repo
    #[arg(long = "voxtral-model", default_value = DEFAULT_VOXTRAL_MODEL)]
    voxtral_model: String,

    /// Maximum Voxtral audio frames to generate
    #[arg(long = "voxtral-max-frames", default_value_t = VOXTRAL_DEFAULT_MAX_FRAMES)]
    voxtral_max_frames: usize,

    /// Voxtral flow-matching steps per frame
    #[arg(long = "voxtral-flow-steps", default_value_t = 7)]
    voxtral_flow_steps: usize,

    /// Enable Voxtral language KV cache (off by default)
    #[arg(long = "voxtral-kv-cache")]
    voxtral_kv_cache: bool,

    /// Use the current opt-in low-latency Voxtral preset
    #[arg(long = "voxtral-realtime")]
    voxtral_realtime: bool,

    /// Normalize compact Voxtral numeric forms such as versions and times before synthesis
    #[arg(long = "voxtral-normalize-text")]
    voxtral_normalize_text: bool,

    /// Apply known Voxtral pronunciation aliases before synthesis
    #[arg(long = "voxtral-pronunciation-aliases")]
    voxtral_pronunciation_aliases: bool,

    /// Choose Voxtral max frames from the post-normalization synthesis text
    #[arg(long = "voxtral-auto-max-frames")]
    voxtral_auto_max_frames: bool,

    /// Override Voxtral's initial streaming codec chunk size
    #[arg(long = "voxtral-stream-begin-frames")]
    voxtral_stream_begin_frames: Option<usize>,

    /// Speech speed factor (1.0 = normal)
    #[arg(short, long, default_value = "1.0")]
    speed: f32,

    /// Target stream sample rate
    #[arg(long = "sample-rate", default_value = "24000")]
    sample_rate: u32,

    /// Target stream frame duration in milliseconds
    #[arg(long = "frame-ms", default_value = "20")]
    frame_ms: u32,

    /// Write raw signed 16-bit little-endian mono PCM to this path (use - for stdout)
    #[arg(short = 'o', long = "raw-output", conflicts_with = "output")]
    raw_output: Option<PathBuf>,

    /// Write streamed audio to an Ogg/Opus file (use - for stdout)
    #[arg(long = "output", conflicts_with = "raw_output")]
    output: Option<PathBuf>,

    /// Output container/codec for --output. Defaults from extension: .ogg, .opus
    #[arg(long = "format", value_enum, requires = "output")]
    format: Option<StreamOutputFormat>,

    /// Print full JSON stream events instead of compact summaries
    #[arg(long)]
    json: bool,

    /// Strip markdown/MDX formatting before speaking
    #[arg(long)]
    markdown: bool,

    /// Word substitutions (pre-processing), e.g. --sub nteract=enteract
    #[arg(long = "sub", value_name = "WORD=REPLACEMENT")]
    subs: Vec<String>,

    /// Load substitutions from a file (one WORD=REPLACEMENT per line, # comments).
    /// If not set, .voice-subs is auto-discovered from the working directory upward.
    #[arg(long = "sub-file", value_name = "PATH")]
    sub_file: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
struct StreamTranscribeArgs {
    /// Path to an audio file
    #[arg(required_unless_present = "raw_input", conflicts_with = "raw_input")]
    file: Option<PathBuf>,

    /// Read raw signed 16-bit little-endian mono PCM from this path (use - for stdin)
    #[arg(long = "raw-input", value_name = "PATH", conflicts_with = "file")]
    raw_input: Option<PathBuf>,

    /// Sample rate for --raw-input
    #[arg(long = "sample-rate", default_value = "48000", requires = "raw_input")]
    sample_rate: u32,

    /// Target stream frame duration in milliseconds
    #[arg(long = "frame-ms", default_value = "20")]
    frame_ms: u32,

    /// Print full JSON STT events instead of only the transcript
    #[arg(long)]
    json: bool,
}

#[derive(clap::Args, Debug)]
struct ConverseArgs {
    /// Text to speak before listening
    #[arg(trailing_var_arg = true)]
    text: Vec<String>,

    /// TTS engine to use. Defaults to kokoro for `voice`, voxtral for `voxtral`
    #[arg(long, value_enum, default_value_t = TtsEngine::Kokoro, hide_default_value = true)]
    engine: TtsEngine,

    /// Voice name. Defaults to af_heart for Kokoro, casual_male for Voxtral
    #[arg(short, long)]
    voice: Option<String>,

    /// Voxtral model path or HuggingFace repo
    #[arg(long = "voxtral-model", default_value = DEFAULT_VOXTRAL_MODEL)]
    voxtral_model: String,

    /// Maximum Voxtral audio frames to generate
    #[arg(long = "voxtral-max-frames", default_value_t = VOXTRAL_DEFAULT_MAX_FRAMES)]
    voxtral_max_frames: usize,

    /// Voxtral flow-matching steps per frame
    #[arg(long = "voxtral-flow-steps", default_value_t = 7)]
    voxtral_flow_steps: usize,

    /// Enable Voxtral language KV cache (off by default)
    #[arg(long = "voxtral-kv-cache")]
    voxtral_kv_cache: bool,

    /// Use the current opt-in low-latency Voxtral preset
    #[arg(long = "voxtral-realtime")]
    voxtral_realtime: bool,

    // Deliberately no --voxtral-normalize-text yet: converse has separate
    // turn-taking and daemon queue behavior that should be validated in its
    // own slice before changing the spoken prompt.

    /// Override Voxtral's initial streaming codec chunk size for daemon playback
    #[arg(long = "voxtral-stream-begin-frames")]
    voxtral_stream_begin_frames: Option<usize>,

    /// Speech speed factor (1.0 = normal)
    #[arg(short, long, default_value = "1.0")]
    speed: f32,

    /// Max listen duration in seconds (after speaking)
    #[arg(short, long, default_value = "30")]
    duration: u64,

    /// Strip markdown/MDX formatting before speaking
    #[arg(long)]
    markdown: bool,

    /// Word substitutions (pre-processing), e.g. --sub nteract=enteract
    #[arg(long = "sub", value_name = "WORD=REPLACEMENT")]
    subs: Vec<String>,

    /// Load substitutions from a file (one WORD=REPLACEMENT per line, # comments).
    #[arg(long = "sub-file", value_name = "PATH")]
    sub_file: Option<PathBuf>,
}

#[derive(clap::Args, Debug)]
struct ListenArgs {
    /// Continuous mode — record and transcribe segments as you speak.
    /// Segments are split on silence and transcribed in the background.
    #[arg(long)]
    continuous: bool,
}

#[derive(clap::Args, Debug)]
struct TranscribeArgs {
    /// Path to an audio file
    file: PathBuf,
}

#[derive(clap::Args, Debug)]
struct ServeArgs {
    /// Voice name (e.g. af_heart, am_adam)
    #[arg(short, long, default_value = "af_heart")]
    voice: String,

    /// Speech speed factor (1.0 = normal)
    #[arg(short, long, default_value = "1.0")]
    speed: f32,

    /// Word substitutions (pre-processing), e.g. --sub nteract=enteract
    #[arg(long = "sub", value_name = "WORD=REPLACEMENT")]
    subs: Vec<String>,

    /// Load substitutions from a file (one WORD=REPLACEMENT per line, # comments).
    #[arg(long = "sub-file", value_name = "PATH")]
    sub_file: Option<PathBuf>,

    /// Include Metal GPU memory stats (_mem) in MCP tool responses
    #[arg(long)]
    mem: bool,
}

#[derive(clap::Args, Debug)]
struct DaemonArgs {
    #[command(subcommand)]
    command: Option<DaemonCommand>,
}

#[derive(clap::Args, Debug)]
struct BenchArgs {
    #[command(subcommand)]
    command: BenchCommand,
}

#[derive(clap::Subcommand, Debug)]
enum BenchCommand {
    /// Compare TTS stages without playback
    Tts(BenchTtsArgs),
}

#[derive(clap::Args, Debug)]
struct BenchTtsArgs {
    /// Text to synthesize
    #[arg(trailing_var_arg = true)]
    text: Vec<String>,

    /// Read text from a file (use - for stdin)
    #[arg(short = 'f', long = "input-file")]
    input_file: Option<PathBuf>,

    /// TTS engine to benchmark. Repeat to choose multiple engines; defaults to both.
    #[arg(long = "engine", value_enum)]
    engines: Vec<TtsEngine>,

    /// Kokoro voice for --engine kokoro
    #[arg(long = "kokoro-voice", default_value = KOKORO_DEFAULT_VOICE)]
    kokoro_voice: String,

    /// Voxtral voice for --engine voxtral
    #[arg(long = "voxtral-voice", default_value = VOXTRAL_DEFAULT_VOICE)]
    voxtral_voice: String,

    /// Voxtral model path or HuggingFace repo
    #[arg(long = "voxtral-model", default_value = DEFAULT_VOXTRAL_MODEL)]
    voxtral_model: String,

    /// Maximum Voxtral audio frames to generate
    #[arg(long = "voxtral-max-frames", default_value_t = VOXTRAL_DEFAULT_MAX_FRAMES)]
    voxtral_max_frames: usize,

    /// Voxtral flow-matching steps per frame
    #[arg(long = "voxtral-flow-steps", default_value_t = 7)]
    voxtral_flow_steps: usize,

    /// Enable Voxtral language KV cache
    #[arg(long = "voxtral-kv-cache")]
    voxtral_kv_cache: bool,

    /// Use the current opt-in low-latency Voxtral preset
    #[arg(long = "voxtral-realtime")]
    voxtral_realtime: bool,

    /// Synchronize Metal around Voxtral trace sections for local benchmark profiling
    #[arg(long = "voxtral-sync-trace")]
    voxtral_sync_trace: bool,

    /// Normalize compact Voxtral numeric forms such as versions and times before synthesis
    #[arg(long = "voxtral-normalize-text")]
    voxtral_normalize_text: bool,

    /// Apply known Voxtral pronunciation aliases before synthesis
    #[arg(long = "voxtral-pronunciation-aliases")]
    voxtral_pronunciation_aliases: bool,

    /// Choose Voxtral max frames from the post-normalization synthesis text
    #[arg(long = "voxtral-auto-max-frames")]
    voxtral_auto_max_frames: bool,

    /// Override Voxtral's initial streaming codec chunk size for benchmarking
    #[arg(long = "voxtral-stream-begin-frames")]
    voxtral_stream_begin_frames: Option<usize>,

    /// Benchmark daemon streaming instead of local in-process synthesis
    #[arg(long)]
    daemon: bool,

    /// Target daemon stream sample rate when --daemon is set
    #[arg(long = "stream-sample-rate", default_value_t = 24_000)]
    stream_sample_rate: u32,

    /// Target daemon stream frame duration in milliseconds when --daemon is set
    #[arg(long = "stream-frame-ms", default_value_t = 20)]
    stream_frame_ms: u32,

    /// Speech speed factor (1.0 = normal)
    #[arg(short, long, default_value = "1.0")]
    speed: f32,

    /// Number of measured synthesis runs per engine after model load
    #[arg(long = "runs", default_value_t = 1)]
    runs: usize,

    /// Use deterministic Kokoro synthesis for reproducible comparisons
    #[arg(long)]
    deterministic: bool,

    /// Strip markdown/MDX formatting before synthesis
    #[arg(long)]
    markdown: bool,

    /// Word substitutions (pre-processing), e.g. --sub nteract=enteract
    #[arg(long = "sub", value_name = "WORD=REPLACEMENT")]
    subs: Vec<String>,

    /// Load substitutions from a file (one WORD=REPLACEMENT per line, # comments).
    /// If not set, .voice-subs is auto-discovered from the working directory upward.
    #[arg(long = "sub-file", value_name = "PATH")]
    sub_file: Option<PathBuf>,

    /// Optionally write generated WAVs for inspection. Playback is never measured.
    #[arg(long = "output-dir")]
    output_dir: Option<PathBuf>,

    /// Print machine-readable JSON
    #[arg(long)]
    json: bool,
}

#[derive(clap::Subcommand, Debug)]
enum DaemonCommand {
    /// Show daemon queue state
    Status {
        /// Print the raw daemon status JSON
        #[arg(long)]
        json: bool,
    },

    /// Print the daemon Unix socket path
    Socket,

    /// List voices known to the daemon
    Voices {
        /// Print the raw daemon voices JSON
        #[arg(long)]
        json: bool,
    },

    /// Set the daemon default voice
    SetVoice {
        /// TTS engine whose default voice should be changed.
        /// Defaults to kokoro for `voice`, voxtral for `voxtral`
        #[arg(long, value_enum, default_value_t = TtsEngine::Kokoro, hide_default_value = true)]
        engine: TtsEngine,

        /// Voice name, e.g. af_heart or am_adam
        voice: String,
    },

    /// Set the daemon default TTS engine
    SetEngine {
        /// TTS engine to use for requests that do not specify --engine
        #[arg(value_enum)]
        engine: TtsEngine,

        /// Voxtral model path or HuggingFace repo
        #[arg(long = "voxtral-model")]
        voxtral_model: Option<String>,
    },

    /// Set the daemon default speech speed
    SetSpeed {
        /// Speech speed factor, between 0 and 5
        speed: f64,
    },

    /// Cancel a queued item by queue ID
    Cancel {
        /// Queue item ID returned by daemon status or speak
        queue_id: String,
    },

    /// Replay stored question or answer audio for a queue item
    Replay {
        /// Queue item ID returned by daemon status or converse
        queue_id: String,

        /// Which audio file to replay
        #[arg(short, long, value_enum, default_value_t = ReplayPart::Question)]
        part: ReplayPart,
    },

    /// Start the voice daemon
    Start {
        /// Start without eagerly loading STT/microphone support
        #[arg(long)]
        tts_only: bool,
    },

    /// Install voice daemon as a system service (macOS LaunchAgent or Linux systemd user unit)
    Install {
        /// Install the service file without starting the daemon immediately
        #[arg(long)]
        no_start: bool,
    },

    /// Stop and remove the voice daemon system service
    Uninstall,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ReplayPart {
    Question,
    Answer,
}

impl ReplayPart {
    fn as_str(self) -> &'static str {
        match self {
            Self::Question => "question",
            Self::Answer => "answer",
        }
    }
}

impl std::fmt::Display for ReplayPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

fn resolve_text(say: &SayArgs) -> Result<String, String> {
    // --phonemes takes a completely separate path
    if say.phonemes.is_some() {
        return Err("phonemes".into()); // sentinel, not a real error
    }

    // -f / --input-file
    if let Some(path) = &say.input_file {
        let text = if path.to_str() == Some("-") {
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .map_err(|e| format!("Failed to read stdin: {e}"))?;
            buf
        } else {
            std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {e}", path.display()))?
        };
        let text = text.trim().to_string();
        if text.is_empty() {
            return Err("Input file is empty".into());
        }
        return Ok(text);
    }

    // Positional text args
    if !say.text.is_empty() {
        return Ok(say.text.join(" "));
    }

    // Fall back to stdin if it's not a TTY
    if io::stdin().is_terminal() {
        return Err("No text provided. Pass text as arguments, use -f, or pipe to stdin.".into());
    }

    let mut buf = String::new();
    io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| format!("Failed to read stdin: {e}"))?;
    let text = buf.trim().to_string();
    if text.is_empty() {
        return Err("No text provided on stdin".into());
    }
    Ok(text)
}

fn resolve_stream_text(stream: &StreamArgs) -> Result<String, String> {
    if let Some(path) = &stream.input_file {
        let text = if path.to_str() == Some("-") {
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .map_err(|e| format!("Failed to read stdin: {e}"))?;
            buf
        } else {
            std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {e}", path.display()))?
        };
        let text = text.trim().to_string();
        if text.is_empty() {
            return Err("Input file is empty".into());
        }
        return Ok(text);
    }

    if !stream.text.is_empty() {
        return Ok(stream.text.join(" "));
    }

    if io::stdin().is_terminal() {
        return Err("No text provided. Pass text as arguments, use -f, or pipe to stdin.".into());
    }

    let mut buf = String::new();
    io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| format!("Failed to read stdin: {e}"))?;
    let text = buf.trim().to_string();
    if text.is_empty() {
        return Err("No text provided on stdin".into());
    }
    Ok(text)
}

enum StreamOutputWriter {
    Raw(Box<dyn Write>),
    OggOpus(voice_audio::OggOpusStreamWriter),
}

impl StreamOutputWriter {
    fn write_frame(&mut self, frame: &voice_stream::AudioFrame) -> Result<(), String> {
        let bytes = frame.payload_le_bytes();
        match self {
            Self::Raw(writer) => writer
                .write_all(&bytes)
                .map_err(|e| format!("write raw PCM: {e}")),
            Self::OggOpus(writer) => writer.write_pcm_s16le(&bytes),
        }
    }

    fn finish(self) -> Result<(), String> {
        match self {
            Self::Raw(mut writer) => writer.flush().map_err(|e| format!("flush raw output: {e}")),
            Self::OggOpus(writer) => writer.finish(),
        }
    }
}

fn resolve_phonemes_text(args: &PhonemesArgs) -> Result<String, String> {
    if let Some(path) = &args.input_file {
        let text = if path.to_str() == Some("-") {
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .map_err(|e| format!("Failed to read stdin: {e}"))?;
            buf
        } else {
            std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {e}", path.display()))?
        };
        let text = text.trim().to_string();
        if text.is_empty() {
            return Err("Input file is empty".into());
        }
        return Ok(text);
    }

    if !args.text.is_empty() {
        return Ok(args.text.join(" "));
    }

    if io::stdin().is_terminal() {
        return Err("No text provided. Pass text as arguments, use -f, or pipe to stdin.".into());
    }

    let mut buf = String::new();
    io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| format!("Failed to read stdin: {e}"))?;
    let text = buf.trim().to_string();
    if text.is_empty() {
        return Err("No text provided on stdin".into());
    }
    Ok(text)
}

/// Strip markdown/MDX to clean prose for TTS using pulldown-cmark.
///
/// Keeps text content from paragraphs, headings, list items, and block quotes.
/// Drops code blocks, inline code, images, HTML, and link URLs (keeps link text).
/// Handles YAML frontmatter (--- delimited) by skipping it before parsing.
pub(crate) fn strip_markdown(text: &str) -> String {
    // Strip YAML frontmatter before passing to pulldown-cmark
    let text = strip_frontmatter(text);

    let opts = Options::ENABLE_YAML_STYLE_METADATA_BLOCKS
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TABLES;
    let parser = MdParser::new_ext(&text, opts);

    let mut out = String::new();
    let mut skip_depth: usize = 0;

    for event in parser {
        match event {
            // Skip content inside code blocks and images
            Event::Start(Tag::CodeBlock(_)) | Event::Start(Tag::Image { .. }) => {
                skip_depth += 1;
            }
            Event::End(TagEnd::CodeBlock) | Event::End(TagEnd::Image) => {
                skip_depth = skip_depth.saturating_sub(1);
            }

            // Inside a skipped region — ignore everything
            _ if skip_depth > 0 => {}

            // Text and soft/hard breaks
            Event::Text(t) => out.push_str(&t),
            Event::SoftBreak => out.push(' '),
            Event::HardBreak => out.push('\n'),

            // Block-level boundaries → newlines for natural pauses
            Event::End(TagEnd::Paragraph)
            | Event::End(TagEnd::Heading(_))
            | Event::End(TagEnd::Item)
            | Event::End(TagEnd::BlockQuote(_)) => {
                out.push('\n');
            }

            // Inline code → just emit the text (e.g. `HashMap` → "HashMap")
            Event::Code(t) => out.push_str(&t),

            // Everything else (HTML, rules, metadata, etc.) → skip
            _ => {}
        }
    }

    out
}

/// Strip YAML frontmatter (--- delimited) from the start of text.
fn strip_frontmatter(text: &str) -> String {
    let trimmed = text.trim_start();
    if !trimmed.starts_with("---") {
        return text.to_string();
    }
    // Find the closing ---
    if let Some(rest) = trimmed.strip_prefix("---") {
        if let Some(end) = rest.find("\n---") {
            // Skip past the closing --- and its newline
            let after = &rest[end + 4..];
            return after.trim_start_matches('\n').to_string();
        }
    }
    text.to_string()
}

/// Built-in substitutions for common tech terms that G2P mispronounces.
/// These are always applied (before user subs). User subs can override.
const TECH_SUBS: &[(&str, &str)] = &[("VS Code", "V S Code")];

/// Apply word-level text substitutions (case-sensitive match, preserves replacement as-is).
fn apply_substitutions(text: &str, subs: &[(String, String)]) -> String {
    let mut result = text.to_string();
    for (from, to) in subs {
        result = result.replace(from.as_str(), to.as_str());
    }
    result
}

/// Apply built-in tech term substitutions.
fn apply_tech_subs(text: &str) -> String {
    let mut result = text.to_string();
    for (from, to) in TECH_SUBS {
        result = result.replace(from, to);
    }
    result
}

/// Parse "word=replacement" substitution strings.
fn parse_subs(raw: &[String]) -> Vec<(String, String)> {
    raw.iter()
        .filter_map(|s| {
            let (k, v) = s.split_once('=')?;
            Some((k.to_string(), v.to_string()))
        })
        .collect()
}

/// Load substitutions from a file. Format: one `WORD=REPLACEMENT` per line.
/// Wrap the replacement in /slashes/ for phoneme overrides.
/// Lines starting with `#` and blank lines are ignored.
fn load_sub_file(path: &std::path::Path) -> Result<Vec<(String, String)>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read sub-file {}: {e}", path.display()))?;
    Ok(contents
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let (k, v) = l.split_once('=')?;
            Some((k.to_string(), v.to_string()))
        })
        .collect())
}

/// Walk up from the current directory looking for a `.voice-subs` file.
/// Returns the first one found, or None.
fn find_sub_file() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join(".voice-subs");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Merge and sort substitutions so longer keys match first.
/// CLI --sub entries override --sub-file entries for the same key.
///
/// Returns (text_subs, phoneme_overrides). Values wrapped in `/slashes/`
/// are phoneme overrides passed to G2P; everything else is a text substitution.
fn collect_subs(
    cli_subs: &[String],
    file_path: Option<&std::path::Path>,
) -> (Vec<(String, String)>, HashMap<String, String>) {
    let mut map = HashMap::<String, String>::new();

    // File entries first (lower priority)
    if let Some(path) = file_path {
        match load_sub_file(path) {
            Ok(entries) => {
                for (k, v) in entries {
                    map.insert(k, v);
                }
            }
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    }

    // CLI entries override
    for (k, v) in parse_subs(cli_subs) {
        map.insert(k, v);
    }

    // Split into text subs and phoneme overrides
    let mut text_subs = Vec::new();
    let mut phoneme_overrides = HashMap::new();

    for (k, v) in map {
        if v.starts_with('/') && v.ends_with('/') && v.len() > 2 {
            // /phonemes/ → phoneme override (keyed lowercase for G2P lookup)
            phoneme_overrides.insert(k.to_lowercase(), v[1..v.len() - 1].to_string());
        } else {
            text_subs.push((k, v));
        }
    }

    // Sort text subs by key length descending so "nteract.io" matches before "nteract"
    text_subs.sort_by_key(|b| std::cmp::Reverse(b.0.len()));

    (text_subs, phoneme_overrides)
}

fn preprocess_daemon_text(
    text: String,
    engine: TtsEngine,
    markdown: bool,
    cli_subs: &[String],
    sub_file: Option<PathBuf>,
    voxtral_normalize_text: bool,
    voxtral_pronunciation_aliases: bool,
) -> String {
    let text = if markdown {
        strip_markdown(&text)
    } else {
        text
    };
    let text = apply_tech_subs(&text);
    let sub_file = sub_file.or_else(find_sub_file);
    let (subs, _phoneme_overrides) = collect_subs(cli_subs, sub_file.as_deref());
    let text = if subs.is_empty() {
        text
    } else {
        apply_substitutions(&text, &subs)
    };
    if engine == TtsEngine::Voxtral {
        apply_voxtral_text_options(
            text,
            voxtral_normalize_text,
            voxtral_pronunciation_aliases,
        )
    } else {
        text
    }
}

fn apply_voxtral_text_options(
    text: String,
    normalize_numbers: bool,
    pronunciation_aliases: bool,
) -> String {
    if normalize_numbers || pronunciation_aliases {
        voice_voxtral::normalize_tts_text_with_options(
            &text,
            voice_voxtral::VoxtralTextNormalizationOptions {
                numeric: normalize_numbers,
                pronunciation_aliases,
            },
        )
    } else {
        text
    }
}

fn main() {
    // Ctrl+C: set flag for cooperative cancellation. The generation loops
    // check this between chunks and exit cleanly, letting the current
    // Metal kernel finish before tearing down. Always prints, even in quiet mode.
    ctrlc::set_handler(|| {
        INTERRUPTED.store(true, Ordering::SeqCst);
        eprintln!("\nInterrupted.");
    })
    .expect("Failed to set Ctrl+C handler");

    let raw_args: Vec<OsString> = std::env::args_os().collect();
    let profile = invocation_profile_from_arg0(raw_args.first().map(OsString::as_os_str));
    let engine_explicit = args_contain_engine_flag(&raw_args);
    let mut args = Args::parse_from(raw_args);
    apply_invocation_defaults(&mut args, profile, engine_explicit);

    if args.quiet {
        QUIET.store(true, Ordering::Relaxed);
    }

    match args.command {
        Some(Command::Listen(listen_args)) => {
            if listen_args.continuous {
                listen::listen_continuous();
            } else if let Some(mut daemon) = voice_protocol::client::DaemonClient::connect() {
                match daemon.listen(None) {
                    Ok(resp) => {
                        if let Some(result) = resp.result {
                            if let Some(r) = result.get("result").and_then(|v| v.as_str()) {
                                println!("{}", r);
                            }
                        } else if let Some(err) = resp.error {
                            eprintln!("Daemon error: {}", err.message);
                        }
                    }
                    Err(e) => {
                        eprintln!("Daemon error: {e}, falling back to local");
                        listen::listen_and_transcribe();
                    }
                }
            } else {
                listen::listen_and_transcribe();
            }
        }
        Some(Command::Converse(converse_args)) => {
            run_converse(converse_args);
        }
        Some(Command::Transcribe(transcribe_args)) => {
            listen::transcribe_file(&transcribe_args.file);
        }
        Some(Command::Serve(serve_args)) => {
            run_serve(serve_args);
        }
        Some(Command::Mcp(serve_args)) => {
            run_mcp(serve_args);
        }
        Some(Command::Daemon(daemon_args)) => {
            run_daemon(daemon_args);
        }
        Some(Command::Bench(bench_args)) => {
            run_bench(bench_args);
        }
        Some(Command::Say(say_args)) => {
            run_say(say_args);
        }
        Some(Command::Phonemes(phonemes_args)) => {
            run_phonemes(phonemes_args);
        }
        Some(Command::Stream(stream_args)) => {
            run_stream(stream_args);
        }
        Some(Command::StreamTranscribe(stream_args)) => {
            run_stream_transcribe(stream_args);
        }
        Some(Command::StreamContract) => {
            run_stream_contract();
        }
        None => {
            // Backward compatibility: `voice Hello world` = `voice say Hello world`
            // Also: bare `voice` with piped stdin = `voice say` with stdin
            if args.text.is_empty() && io::stdin().is_terminal() {
                // No text, no pipe — show help
                Args::parse_from([profile.help_name(), "--help"]);
            } else {
                let say_args = SayArgs {
                    text: args.text,
                    input_file: None,
                    phonemes: None,
                    engine: profile.default_tts_engine(),
                    voice: None,
                    voxtral_model: DEFAULT_VOXTRAL_MODEL.to_string(),
                    voxtral_max_frames: VOXTRAL_DEFAULT_MAX_FRAMES,
                    voxtral_flow_steps: 7,
                    voxtral_kv_cache: false,
                    voxtral_realtime: false,
                    voxtral_normalize_text: false,
                    voxtral_pronunciation_aliases: false,
                    voxtral_auto_max_frames: false,
                    voxtral_stream_begin_frames: None,
                    output: None,
                    format: None,
                    speed: 1.0,
                    deterministic: false,
                    markdown: false,
                    subs: Vec::new(),
                    sub_file: None,
                };
                run_say(say_args);
            }
        }
    }
}

fn run_stream_contract() {
    let contract = voice_stream::webrtc_sidecar_contract();
    println!("{}", serde_json::to_string_pretty(&contract).unwrap());
}

fn run_daemon(args: DaemonArgs) {
    let command = args
        .command
        .unwrap_or(DaemonCommand::Status { json: false });

    match command {
        DaemonCommand::Socket => {
            println!("{}", voice_protocol::client::daemon_socket_path().display());
        }
        DaemonCommand::Status { json } => {
            let mut daemon = connect_daemon_or_exit();
            let result = daemon_response_or_exit(daemon.status());
            if json {
                println!("{}", serde_json::to_string_pretty(&result).unwrap());
            } else {
                print_daemon_status(&result);
            }
        }
        DaemonCommand::Voices { json } => {
            let mut daemon = connect_daemon_or_exit();
            let result = daemon_response_or_exit(daemon.list_voices());
            if json {
                println!("{}", serde_json::to_string_pretty(&result).unwrap());
            } else {
                print_daemon_voices(&result);
            }
        }
        DaemonCommand::SetVoice { engine, voice } => {
            let mut daemon = connect_daemon_or_exit();
            if let Err(message) = validate_voice_for_engine(engine, &voice) {
                eprintln!("{message}");
                std::process::exit(1);
            }
            let result =
                daemon_response_or_exit(daemon.set_voice_for_engine(engine.as_str(), &voice));
            let engine = result
                .get("engine")
                .and_then(|v| v.as_str())
                .unwrap_or(engine.as_str());
            let voice = result
                .get("voice")
                .and_then(|v| v.as_str())
                .unwrap_or(&voice);
            println!("engine: {engine}");
            println!("voice: {voice}");
        }
        DaemonCommand::SetEngine {
            engine,
            voxtral_model,
        } => {
            let mut daemon = connect_daemon_or_exit();
            let result = daemon_response_or_exit(
                daemon.set_engine(engine.as_str(), voxtral_model.as_deref()),
            );
            let engine = result
                .get("engine")
                .and_then(|v| v.as_str())
                .unwrap_or(engine.as_str());
            let voice = result.get("voice").and_then(|v| v.as_str()).unwrap_or("");
            let voxtral_model = result
                .get("voxtral_model")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            println!("engine: {engine}");
            if !voice.is_empty() {
                println!("voice: {voice}");
            }
            if !voxtral_model.is_empty() {
                println!("voxtral_model: {voxtral_model}");
            }
        }
        DaemonCommand::SetSpeed { speed } => {
            let mut daemon = connect_daemon_or_exit();
            let result = daemon_response_or_exit(daemon.set_speed(speed));
            let speed = result
                .get("speed")
                .and_then(|v| v.as_f64())
                .unwrap_or(speed);
            println!("speed: {speed}");
        }
        DaemonCommand::Cancel { queue_id } => {
            let mut daemon = connect_daemon_or_exit();
            let result = daemon_response_or_exit(daemon.cancel_item(&queue_id));
            let cancelled = result
                .get("cancelled")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            println!("cancelled: {cancelled}");
        }
        DaemonCommand::Replay { queue_id, part } => {
            let mut daemon = connect_daemon_or_exit();
            let result = daemon_response_or_exit(daemon.replay_audio(&queue_id, part.as_str()));
            let duration = result
                .get("duration_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "played {} audio for {} ({} ms)",
                part.as_str(),
                queue_id,
                duration
            );
        }
        DaemonCommand::Start { tts_only } => {
            voice_daemon::run_blocking(voice_daemon::DaemonOptions { tts_only });
        }
        DaemonCommand::Install { no_start } => {
            run_daemon_install(no_start);
        }
        DaemonCommand::Uninstall => {
            run_daemon_uninstall();
        }
    }
}

fn connect_daemon_or_exit() -> voice_protocol::client::DaemonClient {
    if let Some(daemon) = voice_protocol::client::DaemonClient::connect() {
        return daemon;
    }

    let socket = voice_protocol::client::daemon_socket_path();
    eprintln!("voice daemon: not running (socket: {})", socket.display());
    eprintln!("start it with `voice daemon start`");
    std::process::exit(1);
}

fn daemon_response_or_exit(
    result: Result<voice_protocol::rpc::Response, String>,
) -> serde_json::Value {
    match result {
        Ok(resp) => {
            if let Some(err) = resp.error {
                eprintln!("voice daemon: {}", err.message);
                std::process::exit(1);
            }
            resp.result.unwrap_or(serde_json::Value::Null)
        }
        Err(e) => {
            eprintln!("voice daemon: {e}");
            std::process::exit(1);
        }
    }
}

fn print_daemon_status(result: &serde_json::Value) {
    let status = result
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let pending = result
        .get("pending")
        .and_then(|v| v.as_array())
        .map(|items| items.len())
        .unwrap_or(0);
    let recent = result
        .get("recent")
        .and_then(|v| v.as_array())
        .map(|items| items.len())
        .unwrap_or(0);

    println!("status: {status}");
    println!(
        "socket: {}",
        voice_protocol::client::daemon_socket_path().display()
    );

    match result.get("current").filter(|value| !value.is_null()) {
        Some(current) => println!("current: {}", format_daemon_item(current)),
        None => println!("current: none"),
    }

    println!("pending: {pending}");
    println!("recent: {recent}");
}

fn format_daemon_item(item: &serde_json::Value) -> String {
    let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let method = item
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let status = item
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    if let Some(preview) = item.get("text_preview").and_then(|v| v.as_str()) {
        format!("{id} {method} {status}: {preview}")
    } else {
        format!("{id} {method} {status}")
    }
}

fn print_daemon_voices(result: &serde_json::Value) {
    let current = result.get("current").and_then(|v| v.as_str()).unwrap_or("");
    let voices = result
        .get("voices")
        .and_then(|v| v.as_array())
        .map(|voices| voices.as_slice())
        .unwrap_or(&[]);

    for voice in voices {
        let engine = voice
            .get("engine")
            .and_then(|v| v.as_str())
            .unwrap_or("kokoro");
        let id = voice.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let name = voice.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let language = voice.get("language").and_then(|v| v.as_str()).unwrap_or("");
        let builtin = voice
            .get("builtin")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let current_engine = result
            .get("engine")
            .and_then(|v| v.as_str())
            .unwrap_or("kokoro");
        let marker = if id == current && engine == current_engine {
            "*"
        } else {
            " "
        };
        let source = if engine == "voxtral" {
            "preset"
        } else if builtin {
            "builtin"
        } else {
            "download"
        };

        println!("{marker} {engine:<8} {id:<16} {name:<24} {language:<12} {source}");
    }
}

#[derive(Debug, Serialize)]
struct TtsBenchReport {
    text: String,
    mode: &'static str,
    runs: usize,
    speed: f32,
    output_dir: Option<String>,
    engines: Vec<TtsBenchEngineReport>,
}

#[derive(Debug, Serialize)]
struct TtsBenchEngineReport {
    engine: &'static str,
    voice: String,
    model: String,
    model_load_ms: f64,
    device_load_ms: Option<f64>,
    model_resolve_assets_ms: Option<f64>,
    model_config_load_ms: Option<f64>,
    model_tokenizer_load_ms: Option<f64>,
    model_tokenizer_validate_ms: Option<f64>,
    model_weight_metadata_ms: Option<f64>,
    model_weight_validate_ms: Option<f64>,
    module_load_ms: Option<f64>,
    cold_first_audio_ms: f64,
    cold_total_ms: f64,
    runs: Vec<TtsBenchRunReport>,
}

#[derive(Debug, Serialize)]
struct TtsBenchRunReport {
    run: usize,
    text_prep_ms: f64,
    phoneme_ms: Option<f64>,
    voice_load_ms: Option<f64>,
    synth_ms: f64,
    first_code_frame_ms: Option<f64>,
    first_audio_ms: f64,
    total_ms: f64,
    audio_duration_ms: f64,
    model_audio_duration_ms: Option<f64>,
    realtime_factor: Option<f64>,
    model_realtime_factor: Option<f64>,
    first_audio_realtime_factor: Option<f64>,
    first_audio_model_realtime_factor: Option<f64>,
    samples: usize,
    model_samples: Option<usize>,
    sample_rate: u32,
    chunks: Option<usize>,
    frames: Option<usize>,
    ended: Option<bool>,
    voxtral_max_frames: Option<usize>,
    voxtral_flow_steps: Option<usize>,
    voxtral_realtime: Option<bool>,
    voxtral_sync_trace: Option<bool>,
    voxtral_language_cache: Option<bool>,
    voxtral_stream_begin_frames: Option<usize>,
    voxtral_voice_cache_hit: Option<bool>,
    voxtral_audio_frames: Option<usize>,
    voxtral_codec_chunks: Option<usize>,
    voxtral_codec_chunks_per_second: Option<f64>,
    voxtral_prompt_ms: Option<f64>,
    voxtral_language_ms: Option<f64>,
    voxtral_language_ms_per_frame: Option<f64>,
    voxtral_acoustic_ms: Option<f64>,
    voxtral_acoustic_ms_per_frame: Option<f64>,
    voxtral_decode_loop_ms: Option<f64>,
    voxtral_decode_loop_ms_per_frame: Option<f64>,
    voxtral_codec_ms: Option<f64>,
    voxtral_codec_ms_per_chunk: Option<f64>,
    daemon_response_ms: Option<f64>,
    daemon_started_ms: Option<f64>,
    daemon_first_pcm_ms: Option<f64>,
    daemon_stream_elapsed_ms: Option<f64>,
    daemon_queue_id: Option<String>,
    daemon_stream_id: Option<String>,
    output_wav: Option<String>,
}

fn run_bench(args: BenchArgs) {
    match args.command {
        BenchCommand::Tts(args) => run_bench_tts(args),
    }
}

fn run_bench_tts(args: BenchTtsArgs) {
    if args.runs == 0 {
        eprintln!("Error: --runs must be greater than zero");
        std::process::exit(1);
    }
    if let Err(message) = validate_speed(args.speed) {
        eprintln!("Error: {message}");
        std::process::exit(1);
    }
    if let Err(message) = validate_effective_voxtral_options(args.effective_voxtral_options()) {
        eprintln!("Error: {message}");
        std::process::exit(1);
    }
    if args.daemon {
        validate_stream_frame_params(args.stream_sample_rate, args.stream_frame_ms)
            .unwrap_or_else(|e| {
                eprintln!("voice bench tts: {e}");
                std::process::exit(1);
            });
    }

    let text = match resolve_bench_text(&args) {
        Ok(text) => text,
        Err(msg) => {
            eprintln!("Error: {msg}");
            std::process::exit(1);
        }
    };
    let engines = if args.engines.is_empty() {
        vec![TtsEngine::Kokoro, TtsEngine::Voxtral]
    } else {
        args.engines.clone()
    };

    let mut reports = Vec::new();
    for engine in engines {
        let result = if args.daemon {
            bench_daemon_tts(&args, &text, engine)
        } else {
            match engine {
                TtsEngine::Kokoro => bench_kokoro_tts(&args, &text),
                TtsEngine::Voxtral => bench_voxtral_tts(&args, &text),
            }
        };
        match result {
            Ok(report) => reports.push(report),
            Err(err) => {
                eprintln!("voice bench tts: {err}");
                std::process::exit(1);
            }
        }
    }

    let report = TtsBenchReport {
        text,
        mode: if args.daemon { "daemon_stream" } else { "local" },
        runs: args.runs,
        speed: args.speed,
        output_dir: args
            .output_dir
            .as_ref()
            .map(|path| path.display().to_string()),
        engines: reports,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        print_tts_bench_report(&report);
    }
}

fn bench_kokoro_tts(args: &BenchTtsArgs, text: &str) -> Result<TtsBenchEngineReport, String> {
    validate_voice_for_engine(TtsEngine::Kokoro, &args.kokoro_voice)?;

    let model_start = Instant::now();
    let mut model = voice_tts::load_model(MODEL_REPO).map_err(|e| e.to_string())?;
    let model_load = model_start.elapsed();
    let sample_rate = model.sample_rate;
    let synthesis_mode = if args.deterministic {
        voice_tts::SynthesisMode::Deterministic
    } else {
        voice_tts::SynthesisMode::Stochastic
    };

    let mut runs = Vec::with_capacity(args.runs);
    for run_idx in 0..args.runs {
        let total_start = Instant::now();
        let text_prep_start = Instant::now();
        let (prepared, phoneme_overrides) = prepare_bench_text(text, args, TtsEngine::Kokoro);
        let text_prep = text_prep_start.elapsed();

        let phoneme_start = Instant::now();
        let phoneme_chunks = if phoneme_overrides.is_empty() {
            voice_g2p::text_to_phoneme_chunks(&prepared)
        } else {
            voice_g2p::text_to_phoneme_chunks_with_overrides(&prepared, &phoneme_overrides)
        }
        .map_err(|e| format!("kokoro G2P failed: {e}"))?;
        let phoneme = phoneme_start.elapsed();

        let voice_start = Instant::now();
        let voice = model
            .load_voice(&args.kokoro_voice, Some(MODEL_REPO))
            .map_err(|e| format!("failed to load Kokoro voice '{}': {e}", args.kokoro_voice))?;
        let voice_load = voice_start.elapsed();

        let synth_start = Instant::now();
        let mut samples = Vec::new();
        let mut first_audio = None;
        for chunk in &phoneme_chunks {
            let chunk_audio =
                voice_tts::generate_with_mode(&mut model, chunk, &voice, args.speed, synthesis_mode)
                    .map_err(|e| format!("kokoro synthesis failed: {e}"))?;
            if first_audio.is_none() {
                first_audio = Some(total_start.elapsed());
            }
            samples.extend_from_slice(&chunk_audio);
        }
        let synth = synth_start.elapsed();
        let total = total_start.elapsed();
        let total_ms = duration_ms(total);
        let first_audio_ms = duration_ms(first_audio.unwrap_or(total));
        let generated_audio_ms = audio_duration_ms(samples.len(), sample_rate);
        let output_wav = maybe_write_bench_wav(
            &args.output_dir,
            TtsEngine::Kokoro,
            &args.kokoro_voice,
            run_idx + 1,
            &samples,
            sample_rate,
        )?;

        runs.push(TtsBenchRunReport {
            run: run_idx + 1,
            text_prep_ms: duration_ms(text_prep),
            phoneme_ms: Some(duration_ms(phoneme)),
            voice_load_ms: Some(duration_ms(voice_load)),
            synth_ms: duration_ms(synth),
            first_code_frame_ms: None,
            first_audio_ms,
            total_ms,
            audio_duration_ms: generated_audio_ms,
            model_audio_duration_ms: Some(generated_audio_ms),
            realtime_factor: ratio_ms(total_ms, generated_audio_ms),
            model_realtime_factor: ratio_ms(total_ms, generated_audio_ms),
            first_audio_realtime_factor: ratio_ms(first_audio_ms, generated_audio_ms),
            first_audio_model_realtime_factor: ratio_ms(first_audio_ms, generated_audio_ms),
            samples: samples.len(),
            model_samples: Some(samples.len()),
            sample_rate,
            chunks: Some(phoneme_chunks.len()),
            frames: None,
            ended: None,
            voxtral_max_frames: None,
            voxtral_flow_steps: None,
            voxtral_realtime: None,
            voxtral_sync_trace: None,
            voxtral_language_cache: None,
            voxtral_stream_begin_frames: None,
            voxtral_voice_cache_hit: None,
            voxtral_audio_frames: None,
            voxtral_codec_chunks: None,
            voxtral_codec_chunks_per_second: None,
            voxtral_prompt_ms: None,
            voxtral_language_ms: None,
            voxtral_language_ms_per_frame: None,
            voxtral_acoustic_ms: None,
            voxtral_acoustic_ms_per_frame: None,
            voxtral_decode_loop_ms: None,
            voxtral_decode_loop_ms_per_frame: None,
            voxtral_codec_ms: None,
            voxtral_codec_ms_per_chunk: None,
            daemon_response_ms: None,
            daemon_started_ms: None,
            daemon_first_pcm_ms: None,
            daemon_stream_elapsed_ms: None,
            daemon_queue_id: None,
            daemon_stream_id: None,
            output_wav,
        });
    }

    let cold_first_audio_ms = model_load + Duration::from_secs_f64(runs[0].first_audio_ms / 1_000.0);
    let cold_total_ms = model_load + Duration::from_secs_f64(runs[0].total_ms / 1_000.0);

    Ok(TtsBenchEngineReport {
        engine: TtsEngine::Kokoro.as_str(),
        voice: args.kokoro_voice.clone(),
        model: MODEL_REPO.to_string(),
        model_load_ms: duration_ms(model_load),
        device_load_ms: None,
        model_resolve_assets_ms: None,
        model_config_load_ms: None,
        model_tokenizer_load_ms: None,
        model_tokenizer_validate_ms: None,
        model_weight_metadata_ms: None,
        model_weight_validate_ms: None,
        module_load_ms: None,
        cold_first_audio_ms: duration_ms(cold_first_audio_ms),
        cold_total_ms: duration_ms(cold_total_ms),
        runs,
    })
}

fn bench_voxtral_tts(args: &BenchTtsArgs, text: &str) -> Result<TtsBenchEngineReport, String> {
    validate_voice_for_engine(TtsEngine::Voxtral, &args.voxtral_voice)?;
    let voxtral = args.effective_voxtral_options();

    let (mut runtime, load_trace) =
        voice_voxtral::VoxtralTtsRuntime::load_default_with_trace(&args.voxtral_model)
            .map_err(|e| format!("failed to load Voxtral model '{}': {e}", args.voxtral_model))?;
    let mut runs = Vec::with_capacity(args.runs);
    for run_idx in 0..args.runs {
        let total_start = Instant::now();
        let text_prep_start = Instant::now();
        let (prepared, _phoneme_overrides) = prepare_bench_text(text, args, TtsEngine::Voxtral);
        let voxtral = apply_auto_voxtral_max_frames(
            voxtral,
            &prepared,
            args.voxtral_auto_max_frames,
        );
        let text_prep = text_prep_start.elapsed();

        let synth_start = Instant::now();
        let mut streamed_samples = 0usize;
        let (audio, trace) = runtime
            .generate_audio_streaming_with_trace(
                &prepared,
                &args.voxtral_voice,
                voxtral_generation_options(
                    voxtral.max_frames,
                    voxtral.flow_steps,
                    voxtral.kv_cache,
                    args.voxtral_sync_trace,
                ),
                voxtral_streaming_config(voxtral),
                |chunk| {
                    streamed_samples += chunk.samples.len();
                    Ok(())
                },
            )
            .map_err(|e| format!("voxtral synthesis failed: {e}"))?;
        let synth = synth_start.elapsed();
        debug_assert_eq!(streamed_samples, audio.samples.len());
        let total = total_start.elapsed();
        let total_ms = duration_ms(total);
        let first_audio_ms = duration_ms(trace.first_audio_chunk.unwrap_or(total));
        let model_audio_ms = audio_duration_ms(audio.samples.len(), audio.sample_rate);
        let output_samples = voice_audio::adjust_speed(&audio.samples, args.speed)
            .map_err(|e| format!("voxtral speed adjustment failed: {e}"))?;
        let generated_audio_ms = audio_duration_ms(output_samples.len(), audio.sample_rate);
        let language_ms = duration_ms(trace.language);
        let acoustic_ms = duration_ms(trace.acoustic);
        let decode_loop_ms = duration_ms(trace.decode_loop);
        let codec_ms = duration_ms(trace.codec);
        let output_wav = maybe_write_bench_wav(
            &args.output_dir,
            TtsEngine::Voxtral,
            &args.voxtral_voice,
            run_idx + 1,
            &output_samples,
            audio.sample_rate,
        )?;

        runs.push(TtsBenchRunReport {
            run: run_idx + 1,
            text_prep_ms: duration_ms(text_prep),
            phoneme_ms: None,
            voice_load_ms: Some(duration_ms(trace.voice_load)),
            synth_ms: duration_ms(synth),
            first_code_frame_ms: trace.first_frame.map(duration_ms),
            first_audio_ms,
            total_ms,
            audio_duration_ms: generated_audio_ms,
            model_audio_duration_ms: Some(model_audio_ms),
            realtime_factor: ratio_ms(total_ms, generated_audio_ms),
            model_realtime_factor: ratio_ms(total_ms, model_audio_ms),
            first_audio_realtime_factor: ratio_ms(first_audio_ms, generated_audio_ms),
            first_audio_model_realtime_factor: ratio_ms(first_audio_ms, model_audio_ms),
            samples: output_samples.len(),
            model_samples: Some(audio.samples.len()),
            sample_rate: audio.sample_rate,
            chunks: None,
            frames: Some(audio.frames),
            ended: Some(audio.ended),
            voxtral_max_frames: Some(voxtral.max_frames),
            voxtral_flow_steps: Some(voxtral.flow_steps),
            voxtral_realtime: Some(args.voxtral_realtime),
            voxtral_sync_trace: Some(args.voxtral_sync_trace),
            voxtral_language_cache: Some(trace.language_cache),
            voxtral_stream_begin_frames: Some(voxtral_streaming_config(voxtral).chunk_frames_at_begin),
            voxtral_voice_cache_hit: Some(trace.voice_cache_hit),
            voxtral_audio_frames: Some(audio.frames),
            voxtral_codec_chunks: Some(trace.codec_chunks),
            voxtral_codec_chunks_per_second: per_second(trace.codec_chunks, total_ms),
            voxtral_prompt_ms: Some(duration_ms(trace.prompt)),
            voxtral_language_ms: Some(language_ms),
            voxtral_language_ms_per_frame: per_unit(language_ms, audio.frames),
            voxtral_acoustic_ms: Some(acoustic_ms),
            voxtral_acoustic_ms_per_frame: per_unit(acoustic_ms, audio.frames),
            voxtral_decode_loop_ms: Some(decode_loop_ms),
            voxtral_decode_loop_ms_per_frame: per_unit(decode_loop_ms, audio.frames),
            voxtral_codec_ms: Some(codec_ms),
            voxtral_codec_ms_per_chunk: per_unit(codec_ms, trace.codec_chunks),
            daemon_response_ms: None,
            daemon_started_ms: None,
            daemon_first_pcm_ms: None,
            daemon_stream_elapsed_ms: None,
            daemon_queue_id: None,
            daemon_stream_id: None,
            output_wav,
        });
    }

    let cold_first_audio_ms =
        load_trace.total + Duration::from_secs_f64(runs[0].first_audio_ms / 1_000.0);
    let cold_total_ms = load_trace.total + Duration::from_secs_f64(runs[0].total_ms / 1_000.0);

    Ok(TtsBenchEngineReport {
        engine: TtsEngine::Voxtral.as_str(),
        voice: args.voxtral_voice.clone(),
        model: args.voxtral_model.clone(),
        model_load_ms: duration_ms(load_trace.total),
        device_load_ms: Some(duration_ms(load_trace.device_load)),
        model_resolve_assets_ms: Some(duration_ms(load_trace.model_resolve_assets)),
        model_config_load_ms: Some(duration_ms(load_trace.model_config_load)),
        model_tokenizer_load_ms: Some(duration_ms(load_trace.model_tokenizer_load)),
        model_tokenizer_validate_ms: Some(duration_ms(load_trace.model_tokenizer_validate)),
        model_weight_metadata_ms: Some(duration_ms(load_trace.model_weight_metadata)),
        model_weight_validate_ms: Some(duration_ms(load_trace.model_weight_validate)),
        module_load_ms: Some(duration_ms(load_trace.module_load)),
        cold_first_audio_ms: duration_ms(cold_first_audio_ms),
        cold_total_ms: duration_ms(cold_total_ms),
        runs,
    })
}

fn bench_daemon_tts(
    args: &BenchTtsArgs,
    text: &str,
    engine: TtsEngine,
) -> Result<TtsBenchEngineReport, String> {
    let voice = match engine {
        TtsEngine::Kokoro => args.kokoro_voice.as_str(),
        TtsEngine::Voxtral => args.voxtral_voice.as_str(),
    };
    validate_voice_for_engine(engine, voice)?;
    let voxtral = args.effective_voxtral_options();

    let mut daemon = voice_protocol::client::DaemonClient::connect()
        .ok_or_else(|| "voice daemon is not running; start it with `voice daemon start`".to_string())?;
    if !daemon_supports_engine(&mut daemon, engine) {
        return Err(format!(
            "voice daemon does not advertise {} support",
            engine.as_str()
        ));
    }

    let mut runs = Vec::with_capacity(args.runs);
    for run_idx in 0..args.runs {
        let total_start = Instant::now();
        let text_prep_start = Instant::now();
        let (prepared, _phoneme_overrides) = prepare_bench_text(text, args, engine);
        let voxtral = if engine == TtsEngine::Voxtral {
            apply_auto_voxtral_max_frames(
                voxtral,
                &prepared,
                args.voxtral_auto_max_frames,
            )
        } else {
            voxtral
        };
        let text_prep = text_prep_start.elapsed();

        let mut response_at = None;
        let mut started_at = None;
        let mut first_pcm_at = None;
        let mut queue_id = None;
        let mut stream_id = None;
        let mut samples = 0usize;
        let mut frames = 0u64;
        let mut stream_duration_ms = 0u64;
        let mut daemon_elapsed_ms = None;
        let mut terminal_error = None;

        let stream_started = Instant::now();
        let response = daemon.stream_speak_with_options_observed(
            &prepared,
            voice_protocol::client::StreamSpeakOptions {
                voice: Some(voice),
                speed: Some(args.speed as f64),
                sample_rate: Some(args.stream_sample_rate),
                frame_ms: Some(args.stream_frame_ms),
                tts: daemon_tts_options(engine, &args.voxtral_model, voxtral),
            },
            |response| {
                response_at = Some(total_start.elapsed());
                if let Some(result) = response.result.as_ref() {
                    queue_id = result
                        .get("queue_id")
                        .and_then(|value| value.as_str())
                        .map(ToOwned::to_owned);
                    stream_id = result
                        .get("stream_id")
                        .and_then(|value| value.as_str())
                        .map(ToOwned::to_owned);
                }
                Ok(())
            },
            |event| {
                match event {
                    voice_stream::TtsStreamEvent::Started { .. } => {
                        started_at.get_or_insert_with(|| total_start.elapsed());
                    }
                    voice_stream::TtsStreamEvent::Audio { frame } => {
                        first_pcm_at.get_or_insert_with(|| total_start.elapsed());
                        samples += frame.sample_count.saturating_sub(frame.padding_samples);
                    }
                    voice_stream::TtsStreamEvent::Ended(end) => {
                        frames = end.frames;
                        stream_duration_ms = end.duration_ms;
                        daemon_elapsed_ms = Some(end.elapsed_ms);
                    }
                    voice_stream::TtsStreamEvent::Error(err) => {
                        terminal_error = Some(err.message);
                    }
                    voice_stream::TtsStreamEvent::Cancelled(cancelled) => {
                        terminal_error = Some(cancelled.reason);
                    }
                }
                Ok(())
            },
        )?;
        if let Some(err) = response.error {
            return Err(format!("daemon {} stream failed: {}", engine.as_str(), err.message));
        }
        if let Some(err) = terminal_error {
            return Err(format!("daemon {} stream failed: {err}", engine.as_str()));
        }

        let worker_result = if let Some(id) = queue_id.as_deref() {
            daemon_recent_result_for_queue(&mut daemon, id)?
        } else {
            None
        };
        let first_code_frame_ms = worker_result
            .as_ref()
            .and_then(|result| result.get("first_code_frame_ms"))
            .and_then(json_number_ms);
        let voxtral_codec_chunks = worker_result
            .as_ref()
            .and_then(|result| result.get("chunks"))
            .and_then(json_number_usize)
            .filter(|_| engine == TtsEngine::Voxtral);
        let voxtral_audio_frames = worker_result
            .as_ref()
            .and_then(|result| result.get("voxtral_frames"))
            .and_then(json_number_usize)
            .filter(|_| engine == TtsEngine::Voxtral);
        let model_audio_duration_ms = worker_result
            .as_ref()
            .and_then(|result| result.get("model_audio_duration_ms"))
            .and_then(json_number_ms);
        let model_samples = worker_result
            .as_ref()
            .and_then(|result| result.get("model_samples"))
            .and_then(json_number_usize);

        let total = total_start.elapsed();
        let first_audio = first_pcm_at.unwrap_or(total);
        let total_ms = duration_ms(total);
        let first_audio_ms = duration_ms(first_audio);
        let generated_audio_ms = stream_duration_ms as f64;
        runs.push(TtsBenchRunReport {
            run: run_idx + 1,
            text_prep_ms: duration_ms(text_prep),
            phoneme_ms: None,
            voice_load_ms: None,
            synth_ms: duration_ms(stream_started.elapsed()),
            first_code_frame_ms,
            first_audio_ms,
            total_ms,
            audio_duration_ms: generated_audio_ms,
            model_audio_duration_ms,
            realtime_factor: ratio_ms(total_ms, generated_audio_ms),
            model_realtime_factor: model_audio_duration_ms.and_then(|duration| ratio_ms(total_ms, duration)),
            first_audio_realtime_factor: ratio_ms(first_audio_ms, generated_audio_ms),
            first_audio_model_realtime_factor: model_audio_duration_ms
                .and_then(|duration| ratio_ms(first_audio_ms, duration)),
            samples,
            model_samples,
            sample_rate: args.stream_sample_rate,
            chunks: None,
            frames: Some(frames as usize),
            ended: Some(true),
            voxtral_max_frames: (engine == TtsEngine::Voxtral).then_some(voxtral.max_frames),
            voxtral_flow_steps: (engine == TtsEngine::Voxtral).then_some(voxtral.flow_steps),
            voxtral_realtime: (engine == TtsEngine::Voxtral).then_some(args.voxtral_realtime),
            voxtral_sync_trace: (engine == TtsEngine::Voxtral).then_some(false),
            voxtral_language_cache: (engine == TtsEngine::Voxtral).then_some(voxtral.kv_cache),
            voxtral_stream_begin_frames: (engine == TtsEngine::Voxtral)
                .then_some(voxtral_streaming_config(voxtral).chunk_frames_at_begin),
            voxtral_voice_cache_hit: None,
            voxtral_audio_frames,
            voxtral_codec_chunks,
            voxtral_codec_chunks_per_second: voxtral_codec_chunks
                .and_then(|chunks| per_second(chunks, total_ms)),
            voxtral_prompt_ms: None,
            voxtral_language_ms: None,
            voxtral_language_ms_per_frame: None,
            voxtral_acoustic_ms: None,
            voxtral_acoustic_ms_per_frame: None,
            voxtral_decode_loop_ms: None,
            voxtral_decode_loop_ms_per_frame: None,
            voxtral_codec_ms: None,
            voxtral_codec_ms_per_chunk: None,
            daemon_response_ms: response_at.map(duration_ms),
            daemon_started_ms: started_at.map(duration_ms),
            daemon_first_pcm_ms: Some(duration_ms(first_audio)),
            daemon_stream_elapsed_ms: daemon_elapsed_ms.map(|ms| ms as f64),
            daemon_queue_id: queue_id,
            daemon_stream_id: stream_id,
            output_wav: None,
        });
    }

    let cold_first_audio_ms = runs[0].first_audio_ms;
    let cold_total_ms = runs[0].total_ms;
    let model = match engine {
        TtsEngine::Kokoro => MODEL_REPO.to_string(),
        TtsEngine::Voxtral => args.voxtral_model.clone(),
    };

    Ok(TtsBenchEngineReport {
        engine: engine.as_str(),
        voice: voice.to_string(),
        model,
        model_load_ms: 0.0,
        device_load_ms: None,
        model_resolve_assets_ms: None,
        model_config_load_ms: None,
        model_tokenizer_load_ms: None,
        model_tokenizer_validate_ms: None,
        model_weight_metadata_ms: None,
        model_weight_validate_ms: None,
        module_load_ms: None,
        cold_first_audio_ms,
        cold_total_ms,
        runs,
    })
}

fn daemon_recent_result_for_queue(
    daemon: &mut voice_protocol::client::DaemonClient,
    queue_id: &str,
) -> Result<Option<serde_json::Value>, String> {
    let response = daemon.status()?;
    if let Some(err) = response.error {
        return Err(format!("daemon status failed: {}", err.message));
    }
    let recent = response
        .result
        .as_ref()
        .and_then(|result| result.get("recent"))
        .and_then(|recent| recent.as_array());
    let Some(recent) = recent else {
        return Ok(None);
    };

    for item in recent {
        let is_match = item.get("id").and_then(|value| value.as_str()) == Some(queue_id);
        if !is_match {
            continue;
        }
        return item
            .get("result")
            .and_then(|value| value.as_str())
            .map(|json| {
                serde_json::from_str(json)
                    .map_err(|e| format!("daemon queue result for {queue_id} was not JSON: {e}"))
            })
            .transpose();
    }

    Ok(None)
}

fn json_number_ms(value: &serde_json::Value) -> Option<f64> {
    value.as_f64().or_else(|| value.as_u64().map(|value| value as f64))
}

fn json_number_usize(value: &serde_json::Value) -> Option<usize> {
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
}

fn voxtral_streaming_config(
    options: EffectiveVoxtralOptions,
) -> voice_voxtral::VoxtralStreamingConfig {
    let mut config = voice_voxtral::VoxtralStreamingConfig::default();
    if let Some(stream_begin_frames) = options.stream_begin_frames {
        config.chunk_frames_at_begin = stream_begin_frames;
    }
    config
}

fn resolve_bench_text(args: &BenchTtsArgs) -> Result<String, String> {
    if let Some(path) = &args.input_file {
        let text = if path.to_str() == Some("-") {
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .map_err(|e| format!("Failed to read stdin: {e}"))?;
            buf
        } else {
            std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {e}", path.display()))?
        };
        let text = text.trim().to_string();
        if text.is_empty() {
            return Err("Input file is empty".into());
        }
        return Ok(text);
    }

    let text = args.text.join(" ");
    if text.trim().is_empty() {
        Err("No text provided. Pass text, --input-file, or pipe stdin with --input-file -".into())
    } else {
        Ok(text)
    }
}

fn prepare_bench_text(
    text: &str,
    args: &BenchTtsArgs,
    engine: TtsEngine,
) -> (String, HashMap<String, String>) {
    let text = if args.markdown {
        strip_markdown(text)
    } else {
        text.to_string()
    };
    let sub_file = args.sub_file.clone().or_else(find_sub_file);
    let (subs, phoneme_overrides) = collect_subs(&args.subs, sub_file.as_deref());
    let text = apply_tech_subs(&text);
    let text = if subs.is_empty() {
        text
    } else {
        apply_substitutions(&text, &subs)
    };
    let text = if engine == TtsEngine::Voxtral {
        apply_voxtral_text_options(
            text,
            args.voxtral_normalize_text,
            args.voxtral_pronunciation_aliases,
        )
    } else {
        text
    };
    (text, phoneme_overrides)
}

fn maybe_write_bench_wav(
    output_dir: &Option<PathBuf>,
    engine: TtsEngine,
    voice: &str,
    run: usize,
    samples: &[f32],
    sample_rate: u32,
) -> Result<Option<String>, String> {
    let Some(output_dir) = output_dir else {
        return Ok(None);
    };
    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("failed to create {}: {e}", output_dir.display()))?;
    let path = output_dir.join(format!(
        "{}-{}-run{}.wav",
        engine.as_str(),
        file_safe_label(voice),
        run
    ));
    voice_audio::save_wav(samples, &path, sample_rate)?;
    Ok(Some(path.display().to_string()))
}

fn file_safe_label(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn audio_duration_ms(samples: usize, sample_rate: u32) -> f64 {
    samples as f64 / sample_rate as f64 * 1_000.0
}

fn ratio_ms(numerator_ms: f64, denominator_ms: f64) -> Option<f64> {
    (denominator_ms > 0.0).then_some(numerator_ms / denominator_ms)
}

fn per_second(count: usize, elapsed_ms: f64) -> Option<f64> {
    (elapsed_ms > 0.0).then_some(count as f64 / (elapsed_ms / 1_000.0))
}

fn per_unit(total_ms: f64, units: usize) -> Option<f64> {
    (units > 0).then_some(total_ms / units as f64)
}

fn format_optional_ms(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.1}"))
        .unwrap_or_else(|| "-".to_string())
}

fn print_tts_bench_report(report: &TtsBenchReport) {
    println!("tts_bench.text={}", report.text);
    println!("tts_bench.mode={}", report.mode);
    println!("tts_bench.runs={}", report.runs);
    println!("tts_bench.speed={}", report.speed);
    if let Some(output_dir) = &report.output_dir {
        println!("tts_bench.output_dir={output_dir}");
    }
    for engine in &report.engines {
        println!(
            "tts_bench.engine={} voice={} model_load_ms={:.1} cold_first_audio_ms={:.1} cold_total_ms={:.1} device_load_ms={} model_resolve_assets_ms={} model_config_load_ms={} model_tokenizer_load_ms={} model_tokenizer_validate_ms={} model_weight_metadata_ms={} model_weight_validate_ms={} module_load_ms={} model={}",
            engine.engine,
            engine.voice,
            engine.model_load_ms,
            engine.cold_first_audio_ms,
            engine.cold_total_ms,
            format_optional_ms(engine.device_load_ms),
            format_optional_ms(engine.model_resolve_assets_ms),
            format_optional_ms(engine.model_config_load_ms),
            format_optional_ms(engine.model_tokenizer_load_ms),
            format_optional_ms(engine.model_tokenizer_validate_ms),
            format_optional_ms(engine.model_weight_metadata_ms),
            format_optional_ms(engine.model_weight_validate_ms),
            format_optional_ms(engine.module_load_ms),
            engine.model
        );
        for run in &engine.runs {
            let mut line = format!(
                "tts_bench.run engine={} run={} first_audio_ms={:.1} total_ms={:.1} synth_ms={:.1} audio_ms={:.1} model_audio_ms={} realtime_factor={} model_realtime_factor={} first_audio_realtime_factor={} first_audio_model_realtime_factor={} phoneme_ms={} first_code_frame_ms={} voxtral_realtime={} voxtral_sync_trace={} voxtral_max_frames={} voxtral_flow_steps={} voxtral_stream_begin_frames={} voxtral_audio_frames={} voxtral_codec_chunks={} voxtral_codec_chunks_per_second={} voxtral_language_ms_per_frame={} voxtral_acoustic_ms_per_frame={} voxtral_decode_loop_ms_per_frame={} voxtral_codec_ms_per_chunk={} output_wav={}",
                engine.engine,
                run.run,
                run.first_audio_ms,
                run.total_ms,
                run.synth_ms,
                run.audio_duration_ms,
                format_optional_ms(run.model_audio_duration_ms),
                run.realtime_factor
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                run.model_realtime_factor
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                run.first_audio_realtime_factor
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                run.first_audio_model_realtime_factor
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                format_optional_ms(run.phoneme_ms),
                format_optional_ms(run.first_code_frame_ms),
                run.voxtral_realtime
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                run.voxtral_sync_trace
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                run.voxtral_max_frames
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                run.voxtral_flow_steps
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                run.voxtral_stream_begin_frames
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                run.voxtral_audio_frames
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                run.voxtral_codec_chunks
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                run.voxtral_codec_chunks_per_second
                    .map(|value| format!("{value:.2}"))
                    .unwrap_or_else(|| "-".to_string()),
                format_optional_ms(run.voxtral_language_ms_per_frame),
                format_optional_ms(run.voxtral_acoustic_ms_per_frame),
                format_optional_ms(run.voxtral_decode_loop_ms_per_frame),
                format_optional_ms(run.voxtral_codec_ms_per_chunk),
                run.output_wav.as_deref().unwrap_or("")
            );
            if let Some(first_pcm_ms) = run.daemon_first_pcm_ms {
                line.push_str(&format!(
                    " daemon_response_ms={} daemon_started_ms={} daemon_first_pcm_ms={first_pcm_ms:.1} daemon_stream_elapsed_ms={} daemon_queue_id={} daemon_stream_id={}",
                    format_optional_ms(run.daemon_response_ms),
                    format_optional_ms(run.daemon_started_ms),
                    format_optional_ms(run.daemon_stream_elapsed_ms),
                    run.daemon_queue_id.as_deref().unwrap_or(""),
                    run.daemon_stream_id.as_deref().unwrap_or("")
                ));
            }
            println!("{line}");
        }
    }
}

fn run_serve(serve_args: ServeArgs) {
    let model = load_tts_model(std::thread::spawn(|| voice_tts::load_model(MODEL_REPO)));

    let voice = match model.load_voice(&serve_args.voice, Some(MODEL_REPO)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load voice '{}': {e}", serve_args.voice);
            std::process::exit(1);
        }
    };

    let sample_rate = model.sample_rate;
    let sub_file = serve_args.sub_file.clone().or_else(find_sub_file);

    jsonrpc::run(jsonrpc::ServerConfig {
        model,
        voice,
        voice_name: serve_args.voice,
        speed: serve_args.speed,
        sample_rate,
        repo_id: MODEL_REPO.to_string(),
        cli_subs: serve_args.subs,
        sub_file_path: sub_file,
    });
}

fn run_mcp(serve_args: ServeArgs) {
    let model = load_tts_model(std::thread::spawn(|| voice_tts::load_model(MODEL_REPO)));

    let voice = match model.load_voice(&serve_args.voice, Some(MODEL_REPO)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load voice '{}': {e}", serve_args.voice);
            std::process::exit(1);
        }
    };

    let sample_rate = model.sample_rate;
    let sub_file = serve_args.sub_file.clone().or_else(find_sub_file);

    mcp::run(mcp::ServerConfig {
        model,
        voice,
        voice_name: serve_args.voice,
        speed: serve_args.speed,
        sample_rate,
        repo_id: MODEL_REPO.to_string(),
        cli_subs: serve_args.subs,
        sub_file_path: sub_file,
        mem_stats: serve_args.mem,
    });
}

fn run_phonemes(args: PhonemesArgs) {
    let text = match resolve_phonemes_text(&args) {
        Ok(text) => text,
        Err(msg) => {
            eprintln!("Error: {msg}");
            std::process::exit(1);
        }
    };

    let text = if args.markdown {
        strip_markdown(&text)
    } else {
        text
    };
    let sub_file = args.sub_file.clone().or_else(find_sub_file);
    if let Some(ref path) = sub_file {
        info!("Using substitutions from {}", path.display());
    }
    let (subs, phoneme_overrides) = collect_subs(&args.subs, sub_file.as_deref());
    let text = apply_tech_subs(&text);
    let text = if subs.is_empty() {
        text
    } else {
        apply_substitutions(&text, &subs)
    };

    let chunks = if phoneme_overrides.is_empty() {
        voice_g2p::text_to_phoneme_chunks(&text)
    } else {
        voice_g2p::text_to_phoneme_chunks_with_overrides(&text, &phoneme_overrides)
    };
    let chunks = match chunks {
        Ok(chunks) => chunks,
        Err(e) => {
            eprintln!("G2P error: {e}");
            std::process::exit(1);
        }
    };

    if args.json {
        let output = serde_json::json!({
            "text": text,
            "chunks": chunks,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        for chunk in chunks {
            println!("{chunk}");
        }
    }
}

fn run_say(say_args: SayArgs) {
    if let Err(message) = validate_speed(say_args.speed) {
        eprintln!("Error: {message}");
        std::process::exit(1);
    }

    let output_format = match say_args.output.as_ref() {
        Some(output_path) => match resolve_say_output_format(output_path, say_args.format) {
            Ok(format) => Some(format),
            Err(msg) => {
                eprintln!("Error: {msg}");
                std::process::exit(1);
            }
        },
        None => None,
    };

    let voice = selected_voice(say_args.engine, &say_args.voice);
    if let Err(message) = validate_voice_for_engine(say_args.engine, &voice) {
        eprintln!("{message}");
        std::process::exit(1);
    }
    let voxtral = say_args.effective_voxtral_options();
    if let Err(message) = validate_effective_voxtral_options(voxtral) {
        eprintln!("Error: {message}");
        std::process::exit(1);
    }

    // If the daemon is running, delegate normal playback and file synthesis to it.
    // `--phonemes` stays local because the daemon RPC accepts text and runs its
    // own G2P pipeline.
    // `--deterministic` stays local because the daemon protocol does not expose
    // synthesis-mode selection yet.
    if say_args.phonemes.is_none() && !say_args.deterministic {
        if let Some(mut daemon) = voice_protocol::client::DaemonClient::connect() {
            if !daemon_supports_engine(&mut daemon, say_args.engine) {
                eprintln!(
                    "Daemon does not advertise {} support, falling back to local",
                    say_args.engine.as_str()
                );
            } else {
                let text = match resolve_text(&say_args) {
                    Ok(t) => t,
                    Err(msg) => {
                        eprintln!("Error: {msg}");
                        std::process::exit(1);
                    }
                };
                let text = preprocess_daemon_text(
                    text,
                    say_args.engine,
                    say_args.markdown,
                    &say_args.subs,
                    say_args.sub_file.clone(),
                    say_args.voxtral_normalize_text,
                    say_args.voxtral_pronunciation_aliases,
                );
                let voxtral = apply_auto_voxtral_max_frames(
                    voxtral,
                    &text,
                    say_args.voxtral_auto_max_frames,
                );
                let tts_options =
                    daemon_tts_options(say_args.engine, &say_args.voxtral_model, voxtral);

                let daemon_result = if let Some(output_path) = &say_args.output {
                    daemon.synthesize_with_format_and_options(
                        &text,
                        &output_path.to_string_lossy(),
                        output_format.map(|format| format.as_str()),
                        Some(&voice),
                        Some(say_args.speed as f64),
                        tts_options,
                    )
                } else {
                    daemon.speak_with_options(
                        &text,
                        Some(&voice),
                        Some(say_args.speed as f64),
                        tts_options,
                    )
                };

                match daemon_result {
                    Ok(resp) if resp.error.is_none() => {
                        let failed = resp
                            .result
                            .as_ref()
                            .and_then(|r| r.get("status"))
                            .and_then(|s| s.as_str())
                            == Some("failed");
                        if failed {
                            eprintln!("Daemon synthesis failed, falling back to local");
                        } else if output_format == Some(voice_audio::AudioOutputFormat::OggOpus) {
                            if let Some(output_path) = &say_args.output {
                                if voice_audio::is_ogg_opus_file(output_path) {
                                    return;
                                }
                                let _ = std::fs::remove_file(output_path);
                                eprintln!("Daemon output was not Ogg/Opus, falling back to local");
                            }
                        } else {
                            return;
                        }
                    }
                    Ok(resp) => {
                        if let Some(err) = resp.error {
                            eprintln!("Daemon error: {}, falling back to local", err.message);
                        }
                    }
                    Err(e) => {
                        eprintln!("Daemon error: {e}, falling back to local");
                    }
                }
            }
        }
    }

    if say_args.engine == TtsEngine::Voxtral {
        run_voxtral_say(say_args, &voice, output_format);
        return;
    }

    // Start model loading in a background thread immediately — this is the
    // slowest startup step (~200ms) and can run while we resolve text + G2P.
    let model_handle = std::thread::spawn(|| voice_tts::load_model(MODEL_REPO));

    // Resolve phoneme chunks (text resolution + G2P are fast with the
    // embedded perceptron tagger, ~1-2ms total).
    let phoneme_chunks: Vec<String> = if let Some(phonemes) = &say_args.phonemes {
        vec![phonemes.clone()]
    } else {
        match resolve_text(&say_args) {
            Ok(text) => {
                let text = if say_args.markdown {
                    strip_markdown(&text)
                } else {
                    text
                };
                let sub_file = say_args.sub_file.clone().or_else(find_sub_file);
                if let Some(ref path) = sub_file {
                    info!("Using substitutions from {}", path.display());
                }
                let (subs, phoneme_overrides) = collect_subs(&say_args.subs, sub_file.as_deref());
                let text = apply_tech_subs(&text);
                let text = if subs.is_empty() {
                    text
                } else {
                    apply_substitutions(&text, &subs)
                };
                info!("Converting text to phonemes...");
                let chunks_result = if phoneme_overrides.is_empty() {
                    voice_g2p::text_to_phoneme_chunks(&text)
                } else {
                    voice_g2p::text_to_phoneme_chunks_with_overrides(&text, &phoneme_overrides)
                };
                match chunks_result {
                    Ok(chunks) => {
                        for (i, chunk) in chunks.iter().enumerate() {
                            info!("  chunk {}: {}", i + 1, chunk);
                        }
                        chunks
                    }
                    Err(e) => {
                        eprintln!("G2P error: {e}");
                        std::process::exit(1);
                    }
                }
            }
            Err(msg) => {
                eprintln!("Error: {msg}");
                std::process::exit(1);
            }
        }
    };

    let mut model = load_tts_model(model_handle);
    let sample_rate = model.sample_rate;
    let synthesis_mode = if say_args.deterministic {
        voice_tts::SynthesisMode::Deterministic
    } else {
        voice_tts::SynthesisMode::Stochastic
    };

    // Load voice (fast for builtins — embedded in binary, ~5ms).
    // Must happen after model is loaded so we can share its Metal device.
    let voice_tensor = match model.load_voice(&voice, Some(MODEL_REPO)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load voice '{}': {e}", voice);
            eprintln!("Available voices include: af_heart, af_bella, af_nicole, af_sarah, af_sky,");
            eprintln!("  am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis");
            std::process::exit(1);
        }
    };

    if let Some(output_path) = &say_args.output {
        if let Err(e) = generate_to_file(
            &mut model,
            &voice_tensor,
            &phoneme_chunks,
            say_args.speed,
            sample_rate,
            synthesis_mode,
            FileOutput {
                path: output_path.as_path(),
                format: output_format.expect("output format resolved when output path is set"),
            },
        ) {
            eprintln!("Failed to write audio: {e}");
            std::process::exit(1);
        }
    } else {
        stream_playback(
            &mut model,
            &voice_tensor,
            &phoneme_chunks,
            say_args.speed,
            sample_rate,
            synthesis_mode,
        );
    }
}

fn run_voxtral_say(
    say_args: SayArgs,
    voice: &str,
    output_format: Option<voice_audio::AudioOutputFormat>,
) {
    if say_args.phonemes.is_some() {
        eprintln!("Error: --phonemes is only supported with --engine kokoro");
        std::process::exit(1);
    }

    let text = match resolve_text(&say_args) {
        Ok(text) => text,
        Err(msg) => {
            eprintln!("Error: {msg}");
            std::process::exit(1);
        }
    };
    let text = preprocess_daemon_text(
        text,
        say_args.engine,
        say_args.markdown,
        &say_args.subs,
        say_args.sub_file.clone(),
        say_args.voxtral_normalize_text,
        say_args.voxtral_pronunciation_aliases,
    );
    let voxtral = apply_auto_voxtral_max_frames(
        say_args.effective_voxtral_options(),
        &text,
        say_args.voxtral_auto_max_frames,
    );
    if let Err(message) = validate_effective_voxtral_options(voxtral) {
        eprintln!("Error: {message}");
        std::process::exit(1);
    }

    info!("Loading Voxtral TTS model ({})...", say_args.voxtral_model);
    let mut runtime = match voice_voxtral::VoxtralTtsRuntime::load_default(&say_args.voxtral_model)
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!(
                "Failed to load Voxtral model '{}': {e}",
                say_args.voxtral_model
            );
            std::process::exit(1);
        }
    };
    info!("Generating Voxtral audio with voice {}...", voice);
    let audio = match runtime.generate_audio(
        &text,
        voice,
        voxtral_generation_options(voxtral.max_frames, voxtral.flow_steps, voxtral.kv_cache, false),
    ) {
        Ok(audio) => audio,
        Err(e) => {
            eprintln!("Voxtral synthesis failed: {e}");
            std::process::exit(1);
        }
    };
    let samples = match voice_audio::adjust_speed(&audio.samples, say_args.speed) {
        Ok(samples) => samples,
        Err(e) => {
            eprintln!("Voxtral speed adjustment failed: {e}");
            std::process::exit(1);
        }
    };

    if let Some(output_path) = &say_args.output {
        let format = output_format.expect("output format resolved when output path is set");
        if let Err(e) = voice_audio::save_audio(&samples, output_path, audio.sample_rate, format) {
            eprintln!("Failed to write audio: {e}");
            std::process::exit(1);
        }
    } else if let Err(e) = play_samples(&samples, audio.sample_rate) {
        eprintln!("Audio playback failed: {e}");
        std::process::exit(1);
    }
}

fn resolve_say_output_format(
    output_path: &std::path::Path,
    explicit: Option<SayOutputFormat>,
) -> Result<voice_audio::AudioOutputFormat, String> {
    voice_audio::resolve_output_format(output_path, explicit.map(Into::into))
}

fn run_stream(stream_args: StreamArgs) {
    if let Err(message) = validate_speed(stream_args.speed) {
        eprintln!("Error: {message}");
        std::process::exit(1);
    }

    let voice = selected_voice(stream_args.engine, &stream_args.voice);
    if let Err(message) = validate_voice_for_engine(stream_args.engine, &voice) {
        eprintln!("{message}");
        std::process::exit(1);
    }
    let voxtral = stream_args.effective_voxtral_options();

    let text = match resolve_stream_text(&stream_args) {
        Ok(t) => t,
        Err(msg) => {
            eprintln!("Error: {msg}");
            std::process::exit(1);
        }
    };
    let text = preprocess_daemon_text(
        text,
        stream_args.engine,
        stream_args.markdown,
        &stream_args.subs,
        stream_args.sub_file.clone(),
        stream_args.voxtral_normalize_text,
        stream_args.voxtral_pronunciation_aliases,
    );
    let voxtral = apply_auto_voxtral_max_frames(
        voxtral,
        &text,
        stream_args.voxtral_auto_max_frames,
    );

    validate_stream_frame_params(stream_args.sample_rate, stream_args.frame_ms).unwrap_or_else(
        |e| {
            eprintln!("voice stream: {e}");
            std::process::exit(1);
        },
    );
    if let Err(message) = validate_effective_voxtral_options(voxtral) {
        eprintln!("voice stream: {message}");
        std::process::exit(1);
    }

    let raw_to_stdout = stream_args
        .raw_output
        .as_ref()
        .is_some_and(|path| path.as_os_str() == std::ffi::OsStr::new("-"));
    let output_to_stdout = stream_args
        .output
        .as_ref()
        .is_some_and(|path| path.as_os_str() == std::ffi::OsStr::new("-"));
    let binary_to_stdout = raw_to_stdout || output_to_stdout;
    if binary_to_stdout && stream_args.json {
        eprintln!("Error: --json cannot be combined with binary stream output to stdout");
        std::process::exit(1);
    }

    let mut output_writer: Option<StreamOutputWriter> = if let Some(path) = &stream_args.raw_output
    {
        Some(StreamOutputWriter::Raw(
            if path.as_os_str() == std::ffi::OsStr::new("-") {
                Box::new(io::BufWriter::new(io::stdout())) as Box<dyn Write>
            } else {
                if let Some(parent) = path.parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                            eprintln!("Failed to create {}: {e}", parent.display());
                            std::process::exit(1);
                        });
                    }
                }
                Box::new(std::fs::File::create(path).unwrap_or_else(|e| {
                    eprintln!("Failed to create {}: {e}", path.display());
                    std::process::exit(1);
                })) as Box<dyn Write>
            },
        ))
    } else if let Some(path) = &stream_args.output {
        let format = resolve_stream_output_format(path, stream_args.format).unwrap_or_else(|e| {
            eprintln!("voice stream: {e}");
            std::process::exit(1);
        });
        match format {
            voice_audio::AudioOutputFormat::OggOpus => {
                let writer =
                    voice_audio::OggOpusStreamWriter::create(path, stream_args.sample_rate)
                        .unwrap_or_else(|e| {
                            eprintln!("voice stream: {e}");
                            std::process::exit(1);
                        });
                Some(StreamOutputWriter::OggOpus(writer))
            }
            voice_audio::AudioOutputFormat::Wav => unreachable!(),
        }
    } else {
        None
    };

    let mut daemon = connect_daemon_or_exit();
    if !daemon_supports_engine(&mut daemon, stream_args.engine) {
        eprintln!(
            "voice daemon does not advertise {} support",
            stream_args.engine.as_str()
        );
        std::process::exit(1);
    }
    let mut terminal_error: Option<String> = None;
    let mut frame_count = 0u64;
    let emit_summaries =
        should_emit_stream_summaries(stream_args.json, QUIET.load(Ordering::Relaxed));

    let result = daemon.stream_speak_with_options(
        &text,
        voice_protocol::client::StreamSpeakOptions {
            voice: Some(&voice),
            speed: Some(stream_args.speed as f64),
            sample_rate: Some(stream_args.sample_rate),
            frame_ms: Some(stream_args.frame_ms),
            tts: daemon_tts_options(stream_args.engine, &stream_args.voxtral_model, voxtral),
        },
        |event| {
            if stream_args.json {
                println!("{}", serde_json::to_string(&event).unwrap());
            }

            match event {
                voice_stream::TtsStreamEvent::Started { metadata } => {
                    if emit_summaries {
                        let line = format!(
                            "started stream={} rate={}Hz frame={}ms encoding={:?}",
                            metadata.stream_id,
                            metadata.sample_rate,
                            metadata.frame_ms,
                            metadata.encoding
                        );
                        if binary_to_stdout {
                            eprintln!("{line}");
                        } else {
                            println!("{line}");
                        }
                    }
                }
                voice_stream::TtsStreamEvent::Audio { frame } => {
                    frame_count += 1;
                    if let Some(writer) = output_writer.as_mut() {
                        writer.write_frame(&frame)?;
                    }
                    if emit_summaries {
                        let line = format!(
                            "audio seq={} samples={} padding={}",
                            frame.sequence, frame.sample_count, frame.padding_samples
                        );
                        if binary_to_stdout {
                            eprintln!("{line}");
                        } else {
                            println!("{line}");
                        }
                    }
                }
                voice_stream::TtsStreamEvent::Ended(end) => {
                    if emit_summaries {
                        let line = format!(
                            "ended stream={} frames={} samples={} duration_ms={}",
                            end.stream_id, end.frames, end.samples, end.duration_ms
                        );
                        if binary_to_stdout {
                            eprintln!("{line}");
                        } else {
                            println!("{line}");
                        }
                    }
                }
                voice_stream::TtsStreamEvent::Error(err) => {
                    terminal_error = Some(err.message.clone());
                    if emit_summaries {
                        let line = format!("error stream={}: {}", err.stream_id, err.message);
                        if binary_to_stdout {
                            eprintln!("{line}");
                        } else {
                            println!("{line}");
                        }
                    }
                }
                voice_stream::TtsStreamEvent::Cancelled(cancelled) => {
                    terminal_error = Some(cancelled.reason.clone());
                    if emit_summaries {
                        let line = format!(
                            "cancelled stream={}: {}",
                            cancelled.stream_id, cancelled.reason
                        );
                        if binary_to_stdout {
                            eprintln!("{line}");
                        } else {
                            println!("{line}");
                        }
                    }
                }
            }

            Ok(())
        },
    );

    match result {
        Ok(resp) => {
            if let Some(err) = resp.error {
                eprintln!("voice daemon: {}", err.message);
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("voice daemon stream: {e}");
            std::process::exit(1);
        }
    }

    if let Some(err) = terminal_error {
        eprintln!("voice stream failed: {err}");
        std::process::exit(1);
    }

    if let Some(writer) = output_writer {
        writer.finish().unwrap_or_else(|e| {
            eprintln!("Failed to finish stream output: {e}");
            std::process::exit(1);
        });
    }

    if frame_count == 0 {
        eprintln!("voice stream produced no audio frames");
        std::process::exit(1);
    }
}

fn should_emit_stream_summaries(json: bool, quiet: bool) -> bool {
    !json && !quiet
}

fn run_stream_transcribe(stream_args: StreamTranscribeArgs) {
    let input = match load_stream_transcribe_input(&stream_args) {
        Ok(input) => input,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    validate_stream_frame_params(input.sample_rate, stream_args.frame_ms).unwrap_or_else(|e| {
        eprintln!("voice stream-transcribe: {e}");
        std::process::exit(1);
    });

    if !QUIET.load(Ordering::Relaxed) {
        let duration = input.sample_count as f64 / input.sample_rate as f64;
        eprintln!(
            "Streaming: {} ({:.1}s, {}Hz, {} frames)",
            input.label,
            duration,
            input.sample_rate,
            input.frames.len()
        );
    }

    let mut daemon = connect_daemon_or_exit();
    let mut transcript: Option<String> = None;
    let mut terminal_error: Option<String> = None;
    let mut received_terminal = false;

    let result = daemon.stream_transcribe(
        &input.frames,
        input.sample_rate,
        stream_args.frame_ms,
        |event| {
            if stream_args.json {
                println!("{}", serde_json::to_string(&event).unwrap());
            }

            match event.event.as_str() {
                "stt.transcribed" => {
                    received_terminal = true;
                    transcript = Some(
                        event
                            .data
                            .get("text")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default()
                            .to_string(),
                    );
                }
                "stt.error" => {
                    received_terminal = true;
                    terminal_error = Some(
                        event
                            .data
                            .get("message")
                            .and_then(|value| value.as_str())
                            .unwrap_or("stream transcription failed")
                            .to_string(),
                    );
                }
                _ => {}
            }

            Ok(())
        },
    );

    match result {
        Ok(resp) => {
            if let Some(err) = resp.error {
                eprintln!("voice daemon: {}", err.message);
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("voice daemon stream-transcribe: {e}");
            std::process::exit(1);
        }
    }

    if let Some(err) = terminal_error {
        eprintln!("voice stream-transcribe failed: {err}");
        std::process::exit(1);
    }

    if !received_terminal {
        eprintln!("voice stream-transcribe produced no terminal event");
        std::process::exit(1);
    }

    if !stream_args.json {
        println!("{}", transcript.unwrap_or_default());
    }
}

struct StreamTranscribeInput {
    label: String,
    sample_rate: u32,
    sample_count: usize,
    frames: Vec<Vec<i16>>,
}

fn load_stream_transcribe_input(
    stream_args: &StreamTranscribeArgs,
) -> Result<StreamTranscribeInput, String> {
    if let Some(path) = &stream_args.raw_input {
        validate_stream_frame_params(stream_args.sample_rate, stream_args.frame_ms)
            .map_err(|e| format!("voice stream-transcribe: {e}"))?;
        return load_raw_pcm_s16le_input(path, stream_args.sample_rate, stream_args.frame_ms);
    }

    let file = stream_args
        .file
        .as_ref()
        .ok_or_else(|| "Path to an audio file or --raw-input is required".to_string())?;
    let audio = listen::load_transcription_audio(file)?;
    validate_stream_frame_params(audio.sample_rate, stream_args.frame_ms)
        .map_err(|e| format!("voice stream-transcribe: {e}"))?;
    let frames =
        voice_stream::pcm_s16le_frames(&audio.samples, audio.sample_rate, stream_args.frame_ms);
    Ok(StreamTranscribeInput {
        label: file.display().to_string(),
        sample_rate: audio.sample_rate,
        sample_count: audio.samples.len(),
        frames,
    })
}

fn load_raw_pcm_s16le_input(
    path: &Path,
    sample_rate: u32,
    frame_ms: u32,
) -> Result<StreamTranscribeInput, String> {
    let bytes = if path.as_os_str() == std::ffi::OsStr::new("-") {
        let mut bytes = Vec::new();
        io::stdin()
            .read_to_end(&mut bytes)
            .map_err(|e| format!("Failed to read raw PCM from stdin: {e}"))?;
        bytes
    } else {
        std::fs::read(path)
            .map_err(|e| format!("Failed to read raw PCM {}: {e}", path.display()))?
    };
    let frame_samples = voice_stream::samples_per_frame(sample_rate, frame_ms);
    let frames = pcm_s16le_bytes_to_frames(&bytes, frame_samples)?;
    let sample_count = bytes.len() / 2;
    Ok(StreamTranscribeInput {
        label: if path.as_os_str() == std::ffi::OsStr::new("-") {
            "stdin raw pcm_s16le".to_string()
        } else {
            format!("{} raw pcm_s16le", path.display())
        },
        sample_rate,
        sample_count,
        frames,
    })
}

fn pcm_s16le_bytes_to_frames(bytes: &[u8], frame_samples: usize) -> Result<Vec<Vec<i16>>, String> {
    if bytes.is_empty() {
        return Err("Raw PCM input is empty".to_string());
    }
    if bytes.len() % 2 != 0 {
        return Err(format!(
            "Raw PCM input has {} bytes; expected an even number of bytes for s16le samples",
            bytes.len()
        ));
    }

    let samples: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();
    Ok(samples
        .chunks(frame_samples.max(1))
        .map(|chunk| chunk.to_vec())
        .collect())
}

fn validate_stream_frame_params(sample_rate: u32, frame_ms: u32) -> Result<(), String> {
    if sample_rate == 0 {
        return Err("sample rate must be greater than 0".to_string());
    }
    if frame_ms == 0 {
        return Err("frame-ms must be greater than 0".to_string());
    }

    Ok(())
}

fn validate_voxtral_stream_begin_frames(value: Option<usize>) -> Result<(), String> {
    if value == Some(0) {
        return Err("--voxtral-stream-begin-frames must be greater than zero".to_string());
    }
    Ok(())
}

fn resolve_stream_output_format(
    path: &Path,
    explicit: Option<StreamOutputFormat>,
) -> Result<voice_audio::AudioOutputFormat, String> {
    let explicit = explicit.map(voice_audio::AudioOutputFormat::from);
    let format = if path.as_os_str() == std::ffi::OsStr::new("-") {
        explicit.ok_or_else(|| {
            "voice stream --output - requires --format ogg-opus so stdout is unambiguous"
                .to_string()
        })?
    } else {
        voice_audio::resolve_output_format(path, explicit)?
    };

    if format != voice_audio::AudioOutputFormat::OggOpus {
        return Err(
            "voice stream --output currently supports only Ogg/Opus; use .ogg/.opus or --format ogg-opus, or use --raw-output for PCM"
                .to_string(),
        );
    }

    Ok(format)
}

fn run_converse(args: ConverseArgs) {
    if args.text.is_empty() {
        eprintln!("Error: No text provided. Usage: voice converse <text>");
        std::process::exit(1);
    }
    if let Err(message) = validate_speed(args.speed) {
        eprintln!("Error: {message}");
        std::process::exit(1);
    }

    let voice = selected_voice(args.engine, &args.voice);
    if let Err(message) = validate_voice_for_engine(args.engine, &voice) {
        eprintln!("{message}");
        std::process::exit(1);
    }
    let voxtral = args.effective_voxtral_options();
    if let Err(message) = validate_effective_voxtral_options(voxtral) {
        eprintln!("Error: {message}");
        std::process::exit(1);
    }

    let sub_file = args.sub_file.clone().or_else(find_sub_file);
    let (subs, phoneme_overrides) = collect_subs(&args.subs, sub_file.as_deref());

    let text = args.text.join(" ");
    let text = if args.markdown {
        strip_markdown(&text)
    } else {
        text
    };
    let text = apply_tech_subs(&text);
    let text = if subs.is_empty() {
        text
    } else {
        apply_substitutions(&text, &subs)
    };

    // Delegate TTS playback to daemon if available, but keep microphone capture
    // in the foreground CLI process. Background daemon mic capture can wedge in
    // CoreAudio on macOS, while daemon TTS is the fast queued path we want.
    if let Some(mut daemon) = voice_protocol::client::DaemonClient::connect() {
        if !daemon_supports_engine(&mut daemon, args.engine) {
            eprintln!(
                "Daemon does not advertise {} support, falling back to local",
                args.engine.as_str()
            );
        } else {
            let stt_handle = std::thread::spawn(listen::load_stt);
            match daemon.speak_with_options_and_wait(
                &text,
                Some(&voice),
                Some(args.speed as f64),
                daemon_tts_options(
                    args.engine,
                    &args.voxtral_model,
                    voxtral,
                ),
                true,
            ) {
                Ok(resp) if resp.error.is_none() => {
                    let failed = resp
                        .result
                        .as_ref()
                        .and_then(|r| r.get("status"))
                        .and_then(|s| s.as_str())
                        == Some("failed");
                    if failed {
                        let _ = stt_handle.join();
                        eprintln!("Daemon speak failed, falling back to local");
                    } else {
                        if interrupted() {
                            std::process::exit(130);
                        }
                        finish_converse_listen(stt_handle, args.duration);
                        return;
                    }
                }
                Ok(resp) => {
                    let _ = stt_handle.join();
                    if let Some(err) = resp.error {
                        eprintln!("Daemon error: {}, falling back to local", err.message);
                    }
                }
                Err(e) => {
                    let _ = stt_handle.join();
                    eprintln!("Daemon error: {e}, falling back to local");
                }
            }
        }
    }

    if args.engine == TtsEngine::Voxtral {
        let stt_handle = std::thread::spawn(listen::load_stt);
        info!("Loading Voxtral TTS model ({})...", args.voxtral_model);
        let mut runtime = match voice_voxtral::VoxtralTtsRuntime::load_default(&args.voxtral_model) {
            Ok(runtime) => runtime,
            Err(e) => {
                eprintln!("Failed to load Voxtral model '{}': {e}", args.voxtral_model);
                std::process::exit(1);
            }
        };
        let audio = match runtime.generate_audio(
            &text,
            &voice,
            voxtral_generation_options(voxtral.max_frames, voxtral.flow_steps, voxtral.kv_cache, false),
        ) {
            Ok(audio) => audio,
            Err(e) => {
                eprintln!("Voxtral synthesis failed: {e}");
                std::process::exit(1);
            }
        };
        let samples = match voice_audio::adjust_speed(&audio.samples, args.speed) {
            Ok(samples) => samples,
            Err(e) => {
                eprintln!("Voxtral speed adjustment failed: {e}");
                std::process::exit(1);
            }
        };
        if let Err(e) = play_samples(&samples, audio.sample_rate) {
            eprintln!("Audio playback failed: {e}");
            std::process::exit(1);
        }
        if interrupted() {
            std::process::exit(130);
        }
        finish_converse_listen(stt_handle, args.duration);
        return;
    }

    let model_handle = std::thread::spawn(|| voice_tts::load_model(MODEL_REPO));

    info!("Converting text to phonemes...");
    let phoneme_chunks = if phoneme_overrides.is_empty() {
        voice_g2p::text_to_phoneme_chunks(&text)
    } else {
        voice_g2p::text_to_phoneme_chunks_with_overrides(&text, &phoneme_overrides)
    };
    let phoneme_chunks = match phoneme_chunks {
        Ok(chunks) => {
            for (i, chunk) in chunks.iter().enumerate() {
                info!("  chunk {}: {}", i + 1, chunk);
            }
            chunks
        }
        Err(e) => {
            eprintln!("G2P error: {e}");
            std::process::exit(1);
        }
    };

    let mut model = load_tts_model(model_handle);
    let sample_rate = model.sample_rate;

    let voice_tensor = match model.load_voice(&voice, Some(MODEL_REPO)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load voice '{}': {e}", voice);
            std::process::exit(1);
        }
    };

    // Start loading STT model in background while TTS plays
    let stt_handle = std::thread::spawn(listen::load_stt);

    stream_playback(
        &mut model,
        &voice_tensor,
        &phoneme_chunks,
        args.speed,
        sample_rate,
        voice_tts::SynthesisMode::Stochastic,
    );

    if interrupted() {
        std::process::exit(130);
    }

    finish_converse_listen(stt_handle, args.duration);
}

fn finish_converse_listen(
    stt_handle: std::thread::JoinHandle<voice_stt::WhisperModel>,
    duration: u64,
) {
    // STT should be loaded by now (TTS playback took seconds)
    let mut stt_model = stt_handle.join().expect("STT load panicked");

    // Listen for response (VAD auto-stop — no Enter key needed)
    if let Some(result) = listen::listen_and_transcribe_vad(
        &mut stt_model,
        duration * 1_000, // max_duration_ms
        1_500,            // silence_timeout_ms
        0.01,             // silence_threshold
        3.0,              // noise_multiplier
        300,              // calibration_ms
    ) {
        println!("{}", result.text);
        if !QUIET.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = std::io::stderr().flush();
            eprintln!("\n({} tokens)", result.tokens.len());
        }
    }
}

/// Wait for TTS model loading to finish and handle errors.
fn load_tts_model(
    handle: std::thread::JoinHandle<
        std::result::Result<voice_tts::KokoroModel, voice_tts::VoicersError>,
    >,
) -> voice_tts::KokoroModel {
    match handle.join().expect("model loading thread panicked") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            eprintln!("The model will be downloaded from HuggingFace on first run.");
            eprintln!("Check your network connection and try again.");
            std::process::exit(1);
        }
    }
}

struct FileOutput<'a> {
    path: &'a Path,
    format: voice_audio::AudioOutputFormat,
}

/// Batch-generate all chunks and write a single audio file.
fn generate_to_file(
    model: &mut voice_tts::KokoroModel,
    voice: &candle_core::Tensor,
    chunks: &[String],
    speed: f32,
    sample_rate: u32,
    synthesis_mode: voice_tts::SynthesisMode,
    output: FileOutput<'_>,
) -> Result<(), String> {
    info!("Generating audio...");
    let mut all_samples: Vec<f32> = Vec::new();

    for (i, phonemes) in chunks.iter().enumerate() {
        if interrupted() {
            break;
        }
        if phonemes.is_empty() {
            continue;
        }
        if chunks.len() > 1 {
            info!("  generating chunk {}/{}...", i + 1, chunks.len());
        }
        match voice_tts::generate_with_mode(model, phonemes, voice, speed, synthesis_mode) {
            Ok(audio) => {
                all_samples.extend_from_slice(&audio);
            }
            Err(e) => {
                eprintln!("Failed to generate audio for chunk {}: {e}", i + 1);
                std::process::exit(1);
            }
        }
    }

    if interrupted() {
        std::process::exit(130);
    }

    voice_audio::save_audio(&all_samples, output.path, sample_rate, output.format)?;
    info!("Saved {} to {}", output.format, output.path.display());
    Ok(())
}

/// Generate audio chunks and stream them to the speakers via rodio.
///
/// Each chunk is appended to the player as soon as it's generated. rodio
/// plays them sequentially on its audio thread, so the first chunk starts
/// playing while subsequent chunks are still being generated.
fn play_samples(samples: &[f32], sample_rate: u32) -> Result<(), String> {
    use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, Player};
    use std::num::NonZero;

    let mut stream =
        DeviceSinkBuilder::open_default_sink().map_err(|e| format!("open audio output: {e}"))?;
    stream.log_on_drop(false);
    let player = Player::connect_new(stream.mixer());
    let channels = NonZero::new(1u16).unwrap();
    let rate = NonZero::new(sample_rate).unwrap();
    player.append(SamplesBuffer::new(channels, rate, samples.to_vec()));

    while !player.empty() {
        if interrupted() {
            player.stop();
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    Ok(())
}

fn stream_playback(
    model: &mut voice_tts::KokoroModel,
    voice: &candle_core::Tensor,
    chunks: &[String],
    speed: f32,
    sample_rate: u32,
    synthesis_mode: voice_tts::SynthesisMode,
) {
    use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, Player};
    use std::num::NonZero;

    let mut stream = DeviceSinkBuilder::open_default_sink().expect("Failed to open audio output");
    stream.log_on_drop(false);
    let player = Player::connect_new(stream.mixer());

    let channels = NonZero::new(1u16).unwrap();
    let rate = NonZero::new(sample_rate).unwrap();

    for (i, phonemes) in chunks.iter().enumerate() {
        if interrupted() {
            break;
        }
        if phonemes.is_empty() {
            continue;
        }
        if chunks.len() > 1 {
            info!("  generating chunk {}/{}...", i + 1, chunks.len());
        }
        match voice_tts::generate_with_mode(model, phonemes, voice, speed, synthesis_mode) {
            Ok(audio) => {
                let source = SamplesBuffer::new(channels, rate, audio);
                player.append(source);
            }
            Err(e) => {
                eprintln!("Failed to generate audio for chunk {}: {e}", i + 1);
                std::process::exit(1);
            }
        }
    }

    // Wait for playback to finish, checking for Ctrl+C periodically
    // so we can exit cleanly without blocking on sleep_until_end().
    while !player.empty() {
        if interrupted() {
            player.stop();
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// Resolve the path to the `voice` binary.
///
/// Checks a sibling of the current executable first (covers release archives
/// and cargo target directories), then falls back to PATH.
fn find_voice_binary() -> Option<std::path::PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if exe.is_file() {
            return Some(exe);
        }
    }

    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join("voice");
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}

#[cfg(target_os = "macos")]
fn run_daemon_install(no_start: bool) {
    let voice_path = match find_voice_binary() {
        Some(p) => p,
        None => {
            eprintln!("error: voice binary not found on PATH or as the current executable");
            eprintln!("install it first with: cargo install voice");
            std::process::exit(1);
        }
    };

    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
    let agents_dir = home.join("Library/LaunchAgents");
    std::fs::create_dir_all(&agents_dir).unwrap_or_else(|e| {
        eprintln!("error: could not create {}: {e}", agents_dir.display());
        std::process::exit(1);
    });

    let plist_path = agents_dir.join("com.rgbkrk.voice.voiced.plist");
    let voice_str = voice_path.display().to_string();

    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.rgbkrk.voice.voiced</string>
  <key>ProgramArguments</key>
  <array>
    <string>{voice_str}</string>
    <string>daemon</string>
    <string>start</string>
    <string>--tts-only</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/voiced.out.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/voiced.err.log</string>
</dict>
</plist>
"#
    );

    std::fs::write(&plist_path, &plist).unwrap_or_else(|e| {
        eprintln!("error: could not write {}: {e}", plist_path.display());
        std::process::exit(1);
    });

    println!("Installing voice daemon as a LaunchAgent...");
    println!("  voice:   {voice_str}");
    println!("  plist:   {}", plist_path.display());

    let uid = unsafe { libc::getuid() };
    let target = format!("gui/{uid}");
    let label = format!("gui/{uid}/com.rgbkrk.voice.voiced");

    // Unload any existing instance silently before re-loading
    let _ = std::process::Command::new("launchctl")
        .args(["bootout", &target, &plist_path.to_string_lossy()])
        .output();

    let bootstrap = std::process::Command::new("launchctl")
        .args(["bootstrap", &target, &plist_path.to_string_lossy()])
        .output();

    match bootstrap {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            eprintln!(
                "warning: launchctl bootstrap exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Err(e) => {
            eprintln!("warning: launchctl bootstrap failed: {e}");
        }
    }

    let _ = std::process::Command::new("launchctl")
        .args(["enable", &label])
        .output();

    println!("  loaded:  {label}");

    if !no_start {
        let kick = std::process::Command::new("launchctl")
            .args(["kickstart", "-k", &label])
            .output();
        match kick {
            Ok(out) if out.status.success() => {}
            Ok(out) => {
                eprintln!(
                    "warning: launchctl kickstart exited {}: {}",
                    out.status,
                    String::from_utf8_lossy(&out.stderr).trim()
                );
            }
            Err(e) => eprintln!("warning: launchctl kickstart failed: {e}"),
        }

        // Give daemon a moment to start before printing status
        std::thread::sleep(std::time::Duration::from_millis(800));
        println!();
        if let Some(mut daemon) = voice_protocol::client::DaemonClient::connect() {
            let result = daemon_response_or_exit(daemon.status());
            print_daemon_status(&result);
        } else {
            eprintln!("note: daemon did not respond yet — try `voice daemon status` in a moment");
        }
    }
}

#[cfg(target_os = "linux")]
fn run_daemon_install(no_start: bool) {
    let voice_path = match find_voice_binary() {
        Some(p) => p,
        None => {
            eprintln!("error: voice binary not found on PATH or as the current executable");
            eprintln!("install it first with: cargo install voice");
            std::process::exit(1);
        }
    };

    let config_dir = dirs::config_dir().unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
            .join(".config")
    });
    let systemd_dir = config_dir.join("systemd/user");
    std::fs::create_dir_all(&systemd_dir).unwrap_or_else(|e| {
        eprintln!("error: could not create {}: {e}", systemd_dir.display());
        std::process::exit(1);
    });

    let unit_path = systemd_dir.join("voiced.service");
    let legacy_unit_path = systemd_dir.join("voice-daemon.service");
    let voice_str = voice_path.display().to_string();

    let unit = format!(
        "[Unit]\nDescription=Voice daemon\nAfter=default.target\n\n\
         [Service]\nType=simple\nExecStart={voice_str} daemon start --tts-only\nRestart=on-failure\nRestartSec=2\n\n\
         [Install]\nWantedBy=default.target\n"
    );

    std::fs::write(&unit_path, &unit).unwrap_or_else(|e| {
        eprintln!("error: could not write {}: {e}", unit_path.display());
        std::process::exit(1);
    });

    println!("Installing voice daemon as a systemd user service...");
    println!("  voice:   {voice_str}");
    println!("  unit:    {}", unit_path.display());

    if legacy_unit_path.exists() {
        println!("  legacy: disabling voice-daemon.service");
        let legacy_disable = std::process::Command::new("systemctl")
            .args(["--user", "disable", "--now", "voice-daemon.service"])
            .output();
        match legacy_disable {
            Ok(out) if out.status.success() => {}
            Ok(out) => eprintln!(
                "warning: systemctl disable legacy voice-daemon.service exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ),
            Err(e) => {
                eprintln!("warning: systemctl disable legacy voice-daemon.service failed: {e}")
            }
        }
    }

    let reload = std::process::Command::new("systemctl")
        .args(["--user", "daemon-reload"])
        .output();
    if let Err(e) = reload {
        eprintln!("warning: systemctl daemon-reload failed: {e}");
    }

    let _ = std::process::Command::new("systemctl")
        .args(["--user", "enable", "voiced.service"])
        .output();

    if !no_start {
        let start = std::process::Command::new("systemctl")
            .args(["--user", "start", "voiced.service"])
            .output();
        match start {
            Ok(out) if out.status.success() => {}
            Ok(out) => {
                eprintln!(
                    "warning: systemctl start exited {}: {}",
                    out.status,
                    String::from_utf8_lossy(&out.stderr).trim()
                );
            }
            Err(e) => eprintln!("warning: systemctl start failed: {e}"),
        }

        std::thread::sleep(std::time::Duration::from_millis(800));
        println!();
        if let Some(mut daemon) = voice_protocol::client::DaemonClient::connect() {
            let result = daemon_response_or_exit(daemon.status());
            print_daemon_status(&result);
        } else {
            eprintln!("note: daemon did not respond yet — try `voice daemon status` in a moment");
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn run_daemon_install(_no_start: bool) {
    eprintln!("error: voice daemon install is not supported on this platform");
    eprintln!("See docs/daemon.md for manual setup instructions.");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn run_daemon_uninstall() {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
    let plist_path = home.join("Library/LaunchAgents/com.rgbkrk.voice.voiced.plist");

    if !plist_path.exists() {
        eprintln!("voice daemon service not installed (plist not found)");
        std::process::exit(1);
    }

    let uid = unsafe { libc::getuid() };
    let target = format!("gui/{uid}");

    let bootout = std::process::Command::new("launchctl")
        .args(["bootout", &target, &plist_path.to_string_lossy()])
        .output();
    match bootout {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            eprintln!(
                "warning: launchctl bootout exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Err(e) => eprintln!("warning: launchctl bootout failed: {e}"),
    }

    std::fs::remove_file(&plist_path).unwrap_or_else(|e| {
        eprintln!("error: could not remove {}: {e}", plist_path.display());
        std::process::exit(1);
    });

    println!("Uninstalled voice daemon LaunchAgent.");
    println!("  removed: {}", plist_path.display());
}

#[cfg(target_os = "linux")]
fn run_daemon_uninstall() {
    let config_dir = dirs::config_dir().unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
            .join(".config")
    });
    let systemd_dir = config_dir.join("systemd/user");
    let unit_path = systemd_dir.join("voiced.service");
    let legacy_unit_path = systemd_dir.join("voice-daemon.service");

    if !unit_path.exists() && !legacy_unit_path.exists() {
        eprintln!("voice daemon service not installed (unit file not found)");
        std::process::exit(1);
    }

    if unit_path.exists() {
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "disable", "--now", "voiced.service"])
            .output();
    }
    if legacy_unit_path.exists() {
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "disable", "--now", "voice-daemon.service"])
            .output();
    }

    let _ = std::process::Command::new("systemctl")
        .args(["--user", "daemon-reload"])
        .output();

    let mut removed = Vec::new();
    for path in [&unit_path, &legacy_unit_path] {
        if path.exists() {
            std::fs::remove_file(path).unwrap_or_else(|e| {
                eprintln!("error: could not remove {}: {e}", path.display());
                std::process::exit(1);
            });
            removed.push(path.display().to_string());
        }
    }

    println!("Uninstalled voice daemon systemd user service.");
    for path in removed {
        println!("  removed: {path}");
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn run_daemon_uninstall() {
    eprintln!("error: voice daemon uninstall is not supported on this platform");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_stream_frame_params_rejects_zero_values() {
        assert!(validate_stream_frame_params(0, 20).is_err());
        assert!(validate_stream_frame_params(48_000, 0).is_err());
        assert!(validate_stream_frame_params(48_000, 20).is_ok());
    }

    #[test]
    fn validate_voxtral_stream_begin_frames_rejects_zero() {
        assert!(validate_voxtral_stream_begin_frames(Some(0)).is_err());
        assert!(validate_voxtral_stream_begin_frames(Some(2)).is_ok());
        assert!(validate_voxtral_stream_begin_frames(None).is_ok());
    }

    #[test]
    fn voxtral_realtime_preset_expands_default_options() {
        let options =
            effective_voxtral_options(VOXTRAL_DEFAULT_MAX_FRAMES, 7, false, None, true);

        assert_eq!(options.max_frames, VOXTRAL_REALTIME_MAX_FRAMES);
        assert_eq!(options.flow_steps, 7);
        assert!(options.kv_cache);
        assert_eq!(
            options.stream_begin_frames,
            Some(VOXTRAL_REALTIME_STREAM_BEGIN_FRAMES)
        );
    }

    #[test]
    fn voxtral_realtime_preset_preserves_explicit_overrides() {
        let options = effective_voxtral_options(80, 5, false, Some(3), true);

        assert_eq!(options.max_frames, 80);
        assert_eq!(options.flow_steps, 5);
        assert!(options.kv_cache);
        assert_eq!(options.stream_begin_frames, Some(3));
    }

    #[test]
    fn auto_voxtral_max_frames_uses_text_estimate_without_lowering_explicit_caps() {
        let default_options =
            effective_voxtral_options(VOXTRAL_DEFAULT_MAX_FRAMES, 7, false, None, false);
        assert_eq!(
            apply_auto_voxtral_max_frames(
                default_options,
                "Vox trell should pronounce Vox trell clearly in a short answer.",
                true,
            )
            .max_frames,
            56
        );

        let explicit_high = effective_voxtral_options(80, 7, false, None, false);
        assert_eq!(
            apply_auto_voxtral_max_frames(explicit_high, "hello world", true).max_frames,
            80
        );

        let realtime = effective_voxtral_options(VOXTRAL_DEFAULT_MAX_FRAMES, 7, false, None, true);
        assert_eq!(
            apply_auto_voxtral_max_frames(
                realtime,
                "Vox trell should pronounce Vox trell clearly in a short answer.",
                true,
            )
            .max_frames,
            56
        );
    }

    #[test]
    fn pcm_s16le_bytes_to_frames_decodes_little_endian_samples() {
        let bytes = [
            0x00, 0x00, // 0
            0xff, 0x7f, // i16::MAX
            0x00, 0x80, // i16::MIN
            0xff, 0xff, // -1
            0x01, 0x00, // 1
        ];
        let frames = pcm_s16le_bytes_to_frames(&bytes, 2).unwrap();

        assert_eq!(frames, vec![vec![0, i16::MAX], vec![i16::MIN, -1], vec![1]]);
    }

    #[test]
    fn pcm_s16le_bytes_to_frames_rejects_empty_or_odd_bytes() {
        assert!(pcm_s16le_bytes_to_frames(&[], 2).is_err());
        let err = pcm_s16le_bytes_to_frames(&[0, 1, 2], 2).unwrap_err();
        assert!(err.contains("even number"));
    }

    #[test]
    fn resolve_stream_output_format_accepts_ogg_opus_paths() {
        assert_eq!(
            resolve_stream_output_format(Path::new("reply.ogg"), None).unwrap(),
            voice_audio::AudioOutputFormat::OggOpus
        );
        assert_eq!(
            resolve_stream_output_format(Path::new("reply.opus"), None).unwrap(),
            voice_audio::AudioOutputFormat::OggOpus
        );
        assert_eq!(
            resolve_stream_output_format(Path::new("reply"), Some(StreamOutputFormat::OggOpus))
                .unwrap(),
            voice_audio::AudioOutputFormat::OggOpus
        );
    }

    #[test]
    fn resolve_stream_output_format_rejects_wav_and_stdout_without_format() {
        assert!(resolve_stream_output_format(Path::new("reply.wav"), None).is_err());
        assert!(resolve_stream_output_format(Path::new("reply"), None).is_err());
        assert!(resolve_stream_output_format(Path::new("-"), None).is_err());
        assert_eq!(
            resolve_stream_output_format(Path::new("-"), Some(StreamOutputFormat::OggOpus))
                .unwrap(),
            voice_audio::AudioOutputFormat::OggOpus
        );
    }

    #[test]
    fn stream_summaries_are_suppressed_for_json_or_quiet_output() {
        assert!(should_emit_stream_summaries(false, false));
        assert!(!should_emit_stream_summaries(true, false));
        assert!(!should_emit_stream_summaries(false, true));
        assert!(!should_emit_stream_summaries(true, true));
    }

    #[test]
    fn voxtral_invocation_defaults_say_to_voxtral() {
        let raw_args: Vec<OsString> = ["voxtral", "say", "hello"]
            .into_iter()
            .map(OsString::from)
            .collect();
        let profile = invocation_profile_from_arg0(raw_args.first().map(OsString::as_os_str));
        let engine_explicit = args_contain_engine_flag(&raw_args);
        let mut args = Args::parse_from(raw_args);

        apply_invocation_defaults(&mut args, profile, engine_explicit);

        match args.command {
            Some(Command::Say(args)) => assert_eq!(args.engine, TtsEngine::Voxtral),
            other => panic!("expected say command, got {other:?}"),
        }
    }

    #[test]
    fn voxtral_invocation_preserves_explicit_engine() {
        let raw_args: Vec<OsString> = ["voxtral", "say", "--engine", "kokoro", "hello"]
            .into_iter()
            .map(OsString::from)
            .collect();
        let profile = invocation_profile_from_arg0(raw_args.first().map(OsString::as_os_str));
        let engine_explicit = args_contain_engine_flag(&raw_args);
        let mut args = Args::parse_from(raw_args);

        apply_invocation_defaults(&mut args, profile, engine_explicit);

        match args.command {
            Some(Command::Say(args)) => assert_eq!(args.engine, TtsEngine::Kokoro),
            other => panic!("expected say command, got {other:?}"),
        }
    }

    #[test]
    fn parses_voxtral_text_options_for_say_stream_and_bench() {
        let say = Args::parse_from([
            "voice",
            "say",
            "--engine",
            "voxtral",
            "--voxtral-normalize-text",
            "--voxtral-pronunciation-aliases",
            "--voxtral-auto-max-frames",
            "Read ticket A17.",
        ]);
        match say.command {
            Some(Command::Say(args)) => {
                assert!(args.voxtral_normalize_text);
                assert!(args.voxtral_pronunciation_aliases);
                assert!(args.voxtral_auto_max_frames);
            }
            other => panic!("expected say command, got {other:?}"),
        }

        let stream = Args::parse_from([
            "voice",
            "stream",
            "--engine",
            "voxtral",
            "--voxtral-normalize-text",
            "--voxtral-pronunciation-aliases",
            "--voxtral-auto-max-frames",
            "Read ticket A17.",
        ]);
        match stream.command {
            Some(Command::Stream(args)) => {
                assert!(args.voxtral_normalize_text);
                assert!(args.voxtral_pronunciation_aliases);
                assert!(args.voxtral_auto_max_frames);
            }
            other => panic!("expected stream command, got {other:?}"),
        }

        let bench = Args::parse_from([
            "voice",
            "bench",
            "tts",
            "--engine",
            "voxtral",
            "--voxtral-normalize-text",
            "--voxtral-pronunciation-aliases",
            "--voxtral-auto-max-frames",
            "Read ticket A17.",
        ]);
        match bench.command {
            Some(Command::Bench(BenchArgs {
                command: BenchCommand::Tts(args),
            })) => {
                assert!(args.voxtral_normalize_text);
                assert!(args.voxtral_pronunciation_aliases);
                assert!(args.voxtral_auto_max_frames);
            }
            other => panic!("expected bench tts command, got {other:?}"),
        }
    }

    #[test]
    fn preprocess_voxtral_text_normalization_is_opt_in() {
        let text = "Read ticket A17, version 2.4.1, at 9:30 PM.".to_string();

        assert_eq!(
            preprocess_daemon_text(
                text.clone(),
                TtsEngine::Voxtral,
                false,
                &[],
                None,
                false,
                false
            ),
            text
        );
        assert_eq!(
            preprocess_daemon_text(
                text.clone(),
                TtsEngine::Kokoro,
                false,
                &[],
                None,
                true,
                true
            ),
            text
        );
        assert_eq!(
            preprocess_daemon_text(text, TtsEngine::Voxtral, false, &[], None, true, false),
            "Read ticket A seventeen, version two point four point one, at nine thirty PM."
        );

        assert_eq!(
            preprocess_daemon_text(
                "Voxtral reads A17.".to_string(),
                TtsEngine::Voxtral,
                false,
                &[],
                None,
                true,
                true
            ),
            "Vox trell reads A seventeen."
        );
    }

    #[test]
    fn prepare_bench_text_normalizes_only_voxtral_engine() {
        let args = BenchTtsArgs {
            text: vec![],
            input_file: None,
            engines: vec![],
            kokoro_voice: KOKORO_DEFAULT_VOICE.to_string(),
            voxtral_voice: VOXTRAL_DEFAULT_VOICE.to_string(),
            voxtral_model: DEFAULT_VOXTRAL_MODEL.to_string(),
            voxtral_max_frames: VOXTRAL_DEFAULT_MAX_FRAMES,
            voxtral_flow_steps: 7,
            voxtral_kv_cache: false,
            voxtral_realtime: false,
            voxtral_sync_trace: false,
            voxtral_normalize_text: true,
            voxtral_pronunciation_aliases: false,
            voxtral_auto_max_frames: false,
            voxtral_stream_begin_frames: None,
            daemon: false,
            stream_sample_rate: 24_000,
            stream_frame_ms: 20,
            speed: 1.0,
            runs: 1,
            deterministic: false,
            markdown: false,
            subs: vec![],
            sub_file: None,
            output_dir: None,
            json: false,
        };

        let text = "Read ticket A17.";
        assert_eq!(
            prepare_bench_text(text, &args, TtsEngine::Kokoro).0,
            "Read ticket A17."
        );
        assert_eq!(
            prepare_bench_text(text, &args, TtsEngine::Voxtral).0,
            "Read ticket A seventeen."
        );
    }

    #[test]
    fn tts_bench_report_serializes_load_trace_fields() {
        let report = TtsBenchReport {
            text: "hello".to_string(),
            mode: "local",
            runs: 1,
            speed: 1.0,
            output_dir: None,
            engines: vec![TtsBenchEngineReport {
                engine: "voxtral",
                voice: "casual_male".to_string(),
                model: "mistralai/Voxtral-4B-TTS-2603".to_string(),
                model_load_ms: 12.0,
                device_load_ms: Some(1.0),
                model_resolve_assets_ms: Some(2.0),
                model_config_load_ms: Some(3.0),
                model_tokenizer_load_ms: Some(4.0),
                model_tokenizer_validate_ms: Some(5.0),
                model_weight_metadata_ms: Some(6.0),
                model_weight_validate_ms: Some(7.0),
                module_load_ms: Some(8.0),
                cold_first_audio_ms: 20.0,
                cold_total_ms: 30.0,
                runs: vec![],
            }],
        };

        let value = serde_json::to_value(report).unwrap();
        let engine = &value["engines"][0];
        assert_eq!(engine["device_load_ms"], 1.0);
        assert_eq!(engine["model_resolve_assets_ms"], 2.0);
        assert_eq!(engine["model_config_load_ms"], 3.0);
        assert_eq!(engine["model_tokenizer_load_ms"], 4.0);
        assert_eq!(engine["model_tokenizer_validate_ms"], 5.0);
        assert_eq!(engine["model_weight_metadata_ms"], 6.0);
        assert_eq!(engine["model_weight_validate_ms"], 7.0);
        assert_eq!(engine["module_load_ms"], 8.0);
    }

    #[test]
    fn voice_invocation_keeps_kokoro_default() {
        let raw_args: Vec<OsString> = ["voice", "say", "hello"]
            .into_iter()
            .map(OsString::from)
            .collect();
        let profile = invocation_profile_from_arg0(raw_args.first().map(OsString::as_os_str));
        let engine_explicit = args_contain_engine_flag(&raw_args);
        let mut args = Args::parse_from(raw_args);

        apply_invocation_defaults(&mut args, profile, engine_explicit);

        match args.command {
            Some(Command::Say(args)) => assert_eq!(args.engine, TtsEngine::Kokoro),
            other => panic!("expected say command, got {other:?}"),
        }
    }

    #[test]
    fn stream_transcribe_input_accepts_ogg_opus_files() {
        if !command_available("ffmpeg") {
            eprintln!(
                "skipping stream-transcribe Ogg/Opus input test because ffmpeg is not on PATH"
            );
            return;
        }

        let wav_path = temp_audio_path("stream_transcribe_source", "wav");
        let ogg_path = temp_audio_path("stream_transcribe_source", "ogg");
        let sample_rate = 24_000u32;
        let samples: Vec<f32> = (0..sample_rate / 10)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
            for sample in &samples {
                writer.write_sample(*sample).unwrap();
            }
            writer.finalize().unwrap();
        }

        let encode = std::process::Command::new("ffmpeg")
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-y")
            .arg("-i")
            .arg(&wav_path)
            .arg("-ac")
            .arg("1")
            .arg("-ar")
            .arg("48000")
            .arg("-c:a")
            .arg("libopus")
            .arg(&ogg_path)
            .output()
            .unwrap();

        if !encode.status.success() {
            eprintln!(
                "skipping stream-transcribe Ogg/Opus input test because ffmpeg encode failed: {}",
                String::from_utf8_lossy(&encode.stderr)
            );
            let _ = std::fs::remove_file(&wav_path);
            let _ = std::fs::remove_file(&ogg_path);
            return;
        }

        let args = StreamTranscribeArgs {
            file: Some(ogg_path.clone()),
            raw_input: None,
            sample_rate: 48_000,
            frame_ms: 20,
            json: false,
        };
        let input = load_stream_transcribe_input(&args).unwrap();
        let _ = std::fs::remove_file(&wav_path);
        let _ = std::fs::remove_file(&ogg_path);

        assert_eq!(input.sample_rate, 16_000);
        assert!(input.sample_count > 0);
        assert!(!input.frames.is_empty());
    }

    fn temp_audio_path(label: &str, ext: &str) -> PathBuf {
        let pid = std::process::id();
        let tid = std::thread::current().id();
        std::env::temp_dir().join(format!("voice_cli_{label}_{pid}_{tid:?}.{ext}"))
    }

    fn command_available(command: &str) -> bool {
        std::process::Command::new(command)
            .arg("-version")
            .output()
            .is_ok()
    }
}
