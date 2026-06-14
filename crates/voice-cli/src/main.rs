mod jsonrpc;
mod listen;
mod mcp;

use clap::{Parser, ValueEnum};
use pulldown_cmark::{Event, Options, Parser as MdParser, Tag, TagEnd};
use std::collections::HashMap;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

const MODEL_REPO: &str = "prince-canuma/Kokoro-82M";

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
                  voice say -v am_adam \"How are you today?\"\n  \
                  echo \"Hello\" | voice say\n  \
                  voice say -f speech.txt -o output.wav\n  \
                  voice say --format ogg-opus -o reply.ogg \"Hello\"\n  \
                  voice say --phonemes \"hɛloʊ wɜːld\"\n  \
                  voice say --markdown -f post.mdx\n  \
                  voice phonemes \"ChatGPT uses RuntimeStateDoc\"\n  \
                  voice stream --json \"Hello world\"\n  \
                  voice stream --output reply.ogg --format ogg-opus \"Hello world\"\n  \
                  voice stream-transcribe recording.wav\n  \
                  voice listen\n  \
                  voice listen --continuous\n  \
                  voice transcribe recording.wav\n  \
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

    /// Text to speak (shorthand for `voice say <text>`)
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

    /// Replay WAV or raw PCM audio through daemon streaming STT
    StreamTranscribe(StreamTranscribeArgs),

    /// Speak text aloud, then listen for a response (speak + listen in one shot)
    Converse(ConverseArgs),

    /// Record from microphone and transcribe (speech-to-text)
    Listen(ListenArgs),

    /// Transcribe a WAV audio file
    Transcribe(TranscribeArgs),

    /// Run as a JSON-RPC 2.0 server on stdin/stdout
    Serve(ServeArgs),

    /// Run as an MCP (Model Context Protocol) server on stdin/stdout
    Mcp(ServeArgs),

    /// Inspect and control a running voice daemon
    Daemon(DaemonArgs),
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

    /// Voice name (e.g. af_heart, am_adam)
    #[arg(short, long, default_value = "af_heart")]
    voice: String,

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

#[derive(clap::Args, Debug)]
struct StreamArgs {
    /// Text to stream
    #[arg(trailing_var_arg = true)]
    text: Vec<String>,

    /// Read text from a file (use - for stdin)
    #[arg(short = 'f', long = "input-file")]
    input_file: Option<PathBuf>,

    /// Voice name (e.g. af_heart, am_adam)
    #[arg(short, long, default_value = "af_heart")]
    voice: String,

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
    format: Option<SayOutputFormat>,

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
    /// Path to WAV audio file
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

    /// Voice name (e.g. af_heart, am_adam)
    #[arg(short, long, default_value = "af_heart")]
    voice: String,

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
    /// Path to WAV audio file
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
        /// Voice name, e.g. af_heart or am_adam
        voice: String,
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
    markdown: bool,
    cli_subs: &[String],
    sub_file: Option<PathBuf>,
) -> String {
    let text = if markdown {
        strip_markdown(&text)
    } else {
        text
    };
    let text = apply_tech_subs(&text);
    let sub_file = sub_file.or_else(find_sub_file);
    let (subs, _phoneme_overrides) = collect_subs(cli_subs, sub_file.as_deref());
    if subs.is_empty() {
        text
    } else {
        apply_substitutions(&text, &subs)
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

    let args = Args::parse();

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
        None => {
            // Backward compatibility: `voice Hello world` = `voice say Hello world`
            // Also: bare `voice` with piped stdin = `voice say` with stdin
            if args.text.is_empty() && io::stdin().is_terminal() {
                // No text, no pipe — show help
                Args::parse_from(["voice", "--help"]);
            } else {
                let say_args = SayArgs {
                    text: args.text,
                    input_file: None,
                    phonemes: None,
                    voice: "af_heart".to_string(),
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
        DaemonCommand::SetVoice { voice } => {
            let mut daemon = connect_daemon_or_exit();
            let result = daemon_response_or_exit(daemon.set_voice(&voice));
            let voice = result
                .get("voice")
                .and_then(|v| v.as_str())
                .unwrap_or(&voice);
            println!("voice: {voice}");
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
        let id = voice.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let name = voice.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let language = voice.get("language").and_then(|v| v.as_str()).unwrap_or("");
        let builtin = voice
            .get("builtin")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let marker = if id == current { "*" } else { " " };
        let source = if builtin { "builtin" } else { "download" };

        println!("{marker} {id:<14} {name:<24} {language:<12} {source}");
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

    // If the daemon is running, delegate normal playback and file synthesis to it.
    // `--phonemes` stays local because the daemon RPC accepts text and runs its
    // own G2P pipeline.
    // `--deterministic` stays local because the daemon protocol does not expose
    // synthesis-mode selection yet.
    if say_args.phonemes.is_none() && !say_args.deterministic {
        if let Some(mut daemon) = voice_protocol::client::DaemonClient::connect() {
            let text = match resolve_text(&say_args) {
                Ok(t) => t,
                Err(msg) => {
                    eprintln!("Error: {msg}");
                    std::process::exit(1);
                }
            };
            let text = preprocess_daemon_text(
                text,
                say_args.markdown,
                &say_args.subs,
                say_args.sub_file.clone(),
            );

            let daemon_result = if let Some(output_path) = &say_args.output {
                daemon.synthesize_with_format(
                    &text,
                    &output_path.to_string_lossy(),
                    output_format.map(|format| format.as_str()),
                    Some(&say_args.voice),
                    Some(say_args.speed as f64),
                )
            } else {
                daemon.speak(&text, Some(&say_args.voice), Some(say_args.speed as f64))
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
                    } else {
                        if output_format == Some(voice_audio::AudioOutputFormat::OggOpus) {
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
    let voice = match model.load_voice(&say_args.voice, Some(MODEL_REPO)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load voice '{}': {e}", say_args.voice);
            eprintln!("Available voices include: af_heart, af_bella, af_nicole, af_sarah, af_sky,");
            eprintln!("  am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis");
            std::process::exit(1);
        }
    };

    if let Some(output_path) = &say_args.output {
        if let Err(e) = generate_to_file(
            &mut model,
            &voice,
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
            &voice,
            &phoneme_chunks,
            say_args.speed,
            sample_rate,
            synthesis_mode,
        );
    }
}

fn resolve_say_output_format(
    output_path: &std::path::Path,
    explicit: Option<SayOutputFormat>,
) -> Result<voice_audio::AudioOutputFormat, String> {
    voice_audio::resolve_output_format(output_path, explicit.map(Into::into))
}

fn run_stream(stream_args: StreamArgs) {
    let text = match resolve_stream_text(&stream_args) {
        Ok(t) => t,
        Err(msg) => {
            eprintln!("Error: {msg}");
            std::process::exit(1);
        }
    };
    let text = preprocess_daemon_text(
        text,
        stream_args.markdown,
        &stream_args.subs,
        stream_args.sub_file.clone(),
    );

    validate_stream_frame_params(stream_args.sample_rate, stream_args.frame_ms).unwrap_or_else(
        |e| {
            eprintln!("voice stream: {e}");
            std::process::exit(1);
        },
    );

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
    let mut terminal_error: Option<String> = None;
    let mut frame_count = 0u64;

    let result = daemon.stream_speak(
        &text,
        Some(&stream_args.voice),
        Some(stream_args.speed as f64),
        Some(stream_args.sample_rate),
        Some(stream_args.frame_ms),
        |event| {
            if stream_args.json {
                println!("{}", serde_json::to_string(&event).unwrap());
            }

            match event {
                voice_stream::TtsStreamEvent::Started { metadata } => {
                    if !stream_args.json {
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
                    if !stream_args.json {
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
                    if !stream_args.json {
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
                    if !stream_args.json {
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
                    if !stream_args.json {
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
        .ok_or_else(|| "Path to WAV audio file or --raw-input is required".to_string())?;
    let audio = listen::load_transcription_wav(file)?;
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

fn resolve_stream_output_format(
    path: &Path,
    explicit: Option<SayOutputFormat>,
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

    let text = args.text.join(" ");

    // Delegate to daemon if available
    if let Some(mut daemon) = voice_protocol::client::DaemonClient::connect() {
        match daemon.converse(&text, Some(&args.voice)) {
            Ok(resp) => {
                // Extract and print the heard text
                if let Some(result) = resp.result {
                    if let Some(r) = result.get("result").and_then(|v| v.as_str()) {
                        println!("{}", r);
                    }
                } else if let Some(err) = resp.error {
                    eprintln!("Daemon error: {}", err.message);
                }
                return;
            }
            Err(e) => {
                eprintln!("Daemon error: {e}, falling back to local");
            }
        }
    }

    let model_handle = std::thread::spawn(|| voice_tts::load_model(MODEL_REPO));

    let sub_file = args.sub_file.clone().or_else(find_sub_file);
    let (subs, phoneme_overrides) = collect_subs(&args.subs, sub_file.as_deref());

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

    let voice = match model.load_voice(&args.voice, Some(MODEL_REPO)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load voice '{}': {e}", args.voice);
            std::process::exit(1);
        }
    };

    // Start loading STT model in background while TTS plays
    let stt_handle = std::thread::spawn(listen::load_stt);

    stream_playback(
        &mut model,
        &voice,
        &phoneme_chunks,
        args.speed,
        sample_rate,
        voice_tts::SynthesisMode::Stochastic,
    );

    if interrupted() {
        std::process::exit(130);
    }

    // STT should be loaded by now (TTS playback took seconds)
    let mut stt_model = stt_handle.join().expect("STT load panicked");

    // Listen for response (VAD auto-stop — no Enter key needed)
    if let Some(result) = listen::listen_and_transcribe_vad(
        &mut stt_model,
        args.duration * 1_000, // max_duration_ms
        1_500,                 // silence_timeout_ms
        0.01,                  // silence_threshold
        3.0,                   // noise_multiplier
        300,                   // calibration_ms
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
    let unit_path = config_dir.join("systemd/user/voiced.service");

    if !unit_path.exists() {
        eprintln!("voice daemon service not installed (unit file not found)");
        std::process::exit(1);
    }

    let _ = std::process::Command::new("systemctl")
        .args(["--user", "disable", "--now", "voiced.service"])
        .output();

    let _ = std::process::Command::new("systemctl")
        .args(["--user", "daemon-reload"])
        .output();

    std::fs::remove_file(&unit_path).unwrap_or_else(|e| {
        eprintln!("error: could not remove {}: {e}", unit_path.display());
        std::process::exit(1);
    });

    println!("Uninstalled voice daemon systemd user service.");
    println!("  removed: {}", unit_path.display());
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
            resolve_stream_output_format(Path::new("reply"), Some(SayOutputFormat::OggOpus))
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
            resolve_stream_output_format(Path::new("-"), Some(SayOutputFormat::OggOpus)).unwrap(),
            voice_audio::AudioOutputFormat::OggOpus
        );
    }
}
