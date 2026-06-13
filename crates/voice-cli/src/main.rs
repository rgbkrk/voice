mod jsonrpc;
mod listen;
mod mcp;

use clap::Parser;
use pulldown_cmark::{Event, Options, Parser as MdParser, Tag, TagEnd};
use std::collections::HashMap;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
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
                  voice say --phonemes \"hɛloʊ wɜːld\"\n  \
                  voice say --markdown -f post.mdx\n  \
                  voice phonemes \"ChatGPT uses RuntimeStateDoc\"\n  \
                  voice stream --json \"Hello world\"\n  \
                  voice listen\n  \
                  voice listen --continuous\n  \
                  voice transcribe recording.wav\n  \
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

    /// Write WAV to file instead of playing
    #[arg(short, long)]
    output: Option<PathBuf>,

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

    /// Write raw signed 16-bit little-endian mono PCM to this path
    #[arg(short = 'o', long = "raw-output")]
    raw_output: Option<PathBuf>,

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
    }
}

fn connect_daemon_or_exit() -> voice_protocol::client::DaemonClient {
    if let Some(daemon) = voice_protocol::client::DaemonClient::connect() {
        return daemon;
    }

    let socket = voice_protocol::client::daemon_socket_path();
    eprintln!("voice daemon: not running (socket: {})", socket.display());
    eprintln!("start it with `voiced`");
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
                daemon.synthesize(
                    &text,
                    &output_path.to_string_lossy(),
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
                    if !failed {
                        return;
                    }
                    eprintln!("Daemon synthesis failed, falling back to local");
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
        generate_to_file(
            &mut model,
            &voice,
            &phoneme_chunks,
            say_args.speed,
            sample_rate,
            synthesis_mode,
            output_path,
        );
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

    let mut raw_writer = stream_args.raw_output.as_ref().map(|path| {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                    eprintln!("Failed to create {}: {e}", parent.display());
                    std::process::exit(1);
                });
            }
        }
        std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("Failed to create {}: {e}", path.display());
            std::process::exit(1);
        })
    });

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
                        println!(
                            "started stream={} rate={}Hz frame={}ms encoding={:?}",
                            metadata.stream_id,
                            metadata.sample_rate,
                            metadata.frame_ms,
                            metadata.encoding
                        );
                    }
                }
                voice_stream::TtsStreamEvent::Audio { frame } => {
                    frame_count += 1;
                    if let Some(file) = raw_writer.as_mut() {
                        file.write_all(&frame.payload_le_bytes())
                            .map_err(|e| format!("write raw PCM: {e}"))?;
                    }
                    if !stream_args.json {
                        println!(
                            "audio seq={} samples={} padding={}",
                            frame.sequence, frame.sample_count, frame.padding_samples
                        );
                    }
                }
                voice_stream::TtsStreamEvent::Ended(end) => {
                    if !stream_args.json {
                        println!(
                            "ended stream={} frames={} samples={} duration_ms={}",
                            end.stream_id, end.frames, end.samples, end.duration_ms
                        );
                    }
                }
                voice_stream::TtsStreamEvent::Error(err) => {
                    terminal_error = Some(err.message.clone());
                    if !stream_args.json {
                        println!("error stream={}: {}", err.stream_id, err.message);
                    }
                }
                voice_stream::TtsStreamEvent::Cancelled(cancelled) => {
                    terminal_error = Some(cancelled.reason.clone());
                    if !stream_args.json {
                        println!(
                            "cancelled stream={}: {}",
                            cancelled.stream_id, cancelled.reason
                        );
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

    if let Some(mut file) = raw_writer {
        file.flush().unwrap_or_else(|e| {
            eprintln!("Failed to flush raw output: {e}");
            std::process::exit(1);
        });
    }

    if frame_count == 0 {
        eprintln!("voice stream produced no audio frames");
        std::process::exit(1);
    }
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

/// Batch-generate all chunks and write a single WAV file.
fn generate_to_file(
    model: &mut voice_tts::KokoroModel,
    voice: &candle_core::Tensor,
    chunks: &[String],
    speed: f32,
    sample_rate: u32,
    synthesis_mode: voice_tts::SynthesisMode,
    output_path: &PathBuf,
) {
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

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(output_path, spec).expect("Failed to create WAV");
    for s in &all_samples {
        writer.write_sample(*s).expect("Failed to write sample");
    }
    writer.finalize().expect("Failed to finalize WAV");
    info!("Saved to {}", output_path.display());
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
