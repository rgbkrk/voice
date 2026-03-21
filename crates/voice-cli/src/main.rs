mod jsonrpc;
mod listen;
mod mcp;

use clap::Parser;
use pulldown_cmark::{Event, Options, Parser as MdParser, Tag, TagEnd};
use std::collections::HashMap;
use std::io::{self, IsTerminal, Read};
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
                  voice say --phonemes \"hɛloʊ wɜːld\"\n  \
                  voice say --markdown -f post.mdx\n  \
                  voice listen\n  \
                  voice listen --continuous\n  \
                  voice transcribe recording.wav\n  \
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
const TECH_SUBS: &[(&str, &str)] = &[
    ("JSON", "jay-sahn"),
    ("json", "jay-sahn"),
    ("Json", "jay-sahn"),
    ("YAML", "yam-ul"),
    ("yaml", "yam-ul"),
    ("TOML", "tom-ul"),
    ("toml", "tom-ul"),
    ("WASM", "waz-um"),
    ("wasm", "waz-um"),
    ("OAuth", "oh-auth"),
    ("oauth", "oh-auth"),
    ("NGINX", "engine-X"),
    ("nginx", "engine-X"),
    ("PostgreSQL", "post-gres-Q-L"),
    ("CRDTs", "C R D Ts"),
    ("CRDT", "C R D T"),
    ("SQLite", "S-Q-lite"),
    ("WiFi", "why-fye"),
    ("iOS", "eye-O-S"),
    ("macOS", "mac O S"),
    ("VS Code", "V S Code"),
];

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
    text_subs.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    (text_subs, phoneme_overrides)
}

/// Download `mlx.metallib` from the GitHub release matching this binary's version.
///
/// The release workflow uploads the metallib as a standalone asset alongside
/// the binary tarball.  This lets `cargo binstall` users (who skip the build
/// step entirely) get the Metal kernels on first run.
fn download_metallib(dest: &Path) -> Result<(), String> {
    let repo = env!("CARGO_PKG_REPOSITORY");
    let version = env!("CARGO_PKG_VERSION");
    let url = format!("{repo}/releases/download/v{version}/mlx.metallib");

    info!("Downloading mlx.metallib from GitHub release v{version}...");

    let resp = ureq::get(&url)
        .call()
        .map_err(|e| format!("Failed to download mlx.metallib: {e}"))?;

    let size_msg = resp
        .header("content-length")
        .and_then(|s| s.parse::<u64>().ok())
        .map(|len| format!(" ({} MB)", len / 1_048_576))
        .unwrap_or_default();
    info!("Downloading mlx.metallib{size_msg}...");

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create {}: {e}", parent.display()))?;
    }

    // Write to a temp file then rename for atomicity
    let tmp = dest.with_extension("metallib.tmp");
    let mut file = std::fs::File::create(&tmp)
        .map_err(|e| format!("Failed to create {}: {e}", tmp.display()))?;

    std::io::copy(&mut resp.into_reader(), &mut file)
        .map_err(|e| format!("Failed to write mlx.metallib: {e}"))?;

    std::fs::rename(&tmp, dest)
        .map_err(|e| format!("Failed to rename to {}: {e}", dest.display()))?;

    info!("Saved mlx.metallib to {}", dest.display());
    Ok(())
}

/// Ensure `mlx.metallib` is co-located with the voice binary.
///
/// MLX searches for the metallib next to the running binary before falling back
/// to the compile-time `METAL_PATH`. When installed via `cargo install`, the
/// binary lands in `~/.cargo/bin/` but the build tree is gone — so we copy
/// the metallib from `~/.mlx/lib/` (where our build.rs places it) to sit next
/// to the binary, making the co-located search succeed.
///
/// When installed via `cargo binstall`, there's no build step at all — so we
/// download the metallib from the matching GitHub release on first run.
fn ensure_metallib() {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return,
    };
    let exe_dir = match exe.parent() {
        Some(d) => d,
        None => return,
    };

    let colocated = exe_dir.join("mlx.metallib");
    if colocated.exists() {
        return; // Already there — nothing to do
    }

    // Check the stable location where build.rs copies it
    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => return,
    };
    let stable = Path::new(&home)
        .join(".mlx")
        .join("lib")
        .join("mlx.metallib");

    if !stable.exists() {
        // Not built locally — try downloading from the GitHub release
        if let Err(e) = download_metallib(&stable) {
            eprintln!("{e}");
            return;
        }
    }

    // Copy to sit next to the binary
    if let Err(e) = std::fs::copy(&stable, &colocated) {
        // Non-fatal — the model load error handler will print guidance
        eprintln!(
            "Note: could not copy mlx.metallib to {}: {}",
            colocated.display(),
            e
        );
    }
}

fn main() {
    // Ensure MLX metallib is discoverable (fixes `cargo install` from crates.io)
    ensure_metallib();

    // Ctrl+C: set flag for cooperative cancellation. The generation loops
    // check this between chunks and exit cleanly, letting MLX finish its
    // current kernel before tearing down. Always prints, even in quiet mode.
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
        Some(Command::Say(say_args)) => {
            run_say(say_args);
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
                    markdown: false,
                    subs: Vec::new(),
                    sub_file: None,
                };
                run_say(say_args);
            }
        }
    }
}

fn run_serve(serve_args: ServeArgs) {
    let model_handle = std::thread::spawn(|| voice_tts::load_model(MODEL_REPO));

    let voice = match voice_tts::load_voice(&serve_args.voice, Some(MODEL_REPO)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load voice '{}': {e}", serve_args.voice);
            std::process::exit(1);
        }
    };

    let model = load_tts_model(model_handle);
    let sample_rate = model.sample_rate as u32;
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
    let model_handle = std::thread::spawn(|| voice_tts::load_model(MODEL_REPO));

    let voice = match voice_tts::load_voice(&serve_args.voice, Some(MODEL_REPO)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load voice '{}': {e}", serve_args.voice);
            std::process::exit(1);
        }
    };

    let model = load_tts_model(model_handle);
    let sample_rate = model.sample_rate as u32;
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
    });
}

fn run_say(say_args: SayArgs) {
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

    // Load voice (fast for builtins — embedded in binary, ~5ms).
    let voice = match voice_tts::load_voice(&say_args.voice, Some(MODEL_REPO)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load voice '{}': {e}", say_args.voice);
            eprintln!("Available voices include: af_heart, af_bella, af_nicole, af_sarah, af_sky,");
            eprintln!("  am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis");
            std::process::exit(1);
        }
    };

    let mut model = load_tts_model(model_handle);
    let sample_rate = model.sample_rate as u32;

    if let Some(output_path) = &say_args.output {
        generate_to_file(
            &mut model,
            &voice,
            &phoneme_chunks,
            say_args.speed,
            sample_rate,
            output_path,
        );
    } else {
        stream_playback(
            &mut model,
            &voice,
            &phoneme_chunks,
            say_args.speed,
            sample_rate,
        );
    }
}

fn run_converse(args: ConverseArgs) {
    if args.text.is_empty() {
        eprintln!("Error: No text provided. Usage: voice converse <text>");
        std::process::exit(1);
    }

    let text = args.text.join(" ");
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

    let voice = match voice_tts::load_voice(&args.voice, Some(MODEL_REPO)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load voice '{}': {e}", args.voice);
            std::process::exit(1);
        }
    };

    let mut model = load_tts_model(model_handle);
    let sample_rate = model.sample_rate as u32;

    stream_playback(&mut model, &voice, &phoneme_chunks, args.speed, sample_rate);

    if interrupted() {
        std::process::exit(130);
    }

    // Listen for response
    listen::listen_and_transcribe();
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
            let msg = format!("{e}");
            if msg.contains("metallib") || msg.contains("metal") {
                eprintln!("Failed to load model: {e}");
                eprintln!();
                eprintln!("This is a known issue with `cargo install` on Apple Silicon.");
                eprintln!("The MLX Metal shader library (mlx.metallib) was not copied");
                eprintln!("next to the installed binary.");
                eprintln!();
                eprintln!("Fix: build from source instead:");
                eprintln!();
                eprintln!("  git clone https://github.com/rgbkrk/voice.git");
                eprintln!("  cd voice");
                eprintln!("  cargo install --path crates/voice-cli");
                eprintln!();
                eprintln!("Or copy the metallib manually:");
                eprintln!();
                eprintln!("  cp target/release/build/mlx-sys-*/out/build/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib ~/.cargo/bin/");
            } else {
                eprintln!("Failed to load model: {e}");
                eprintln!("The model will be downloaded from HuggingFace on first run.");
                eprintln!("Check your network connection and try again.");
            }
            std::process::exit(1);
        }
    }
}

/// Batch-generate all chunks and write a single WAV file.
fn generate_to_file(
    model: &mut voice_tts::KokoroModel,
    voice: &voice_tts::Array,
    chunks: &[String],
    speed: f32,
    sample_rate: u32,
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
        match voice_tts::generate(model, phonemes, voice, speed) {
            Ok(audio) => {
                all_samples.extend_from_slice(audio.as_slice());
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
    voice: &voice_tts::Array,
    chunks: &[String],
    speed: f32,
    sample_rate: u32,
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
        match voice_tts::generate(model, phonemes, voice, speed) {
            Ok(audio) => {
                let samples: Vec<f32> = audio.as_slice().to_vec();
                let source = SamplesBuffer::new(channels, rate, samples);
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
