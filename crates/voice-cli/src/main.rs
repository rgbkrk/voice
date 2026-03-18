use clap::Parser;
use std::collections::HashMap;
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;

const MODEL_REPO: &str = "prince-canuma/Kokoro-82M";

#[derive(Parser, Debug)]
#[command(
    name = "voice",
    about = "Kokoro TTS from the command line",
    after_help = "If no text, --phonemes, or -f is given, reads from stdin.\n\
                  A .voice-subs file is auto-discovered from the working directory upward.\n\n\
                  Examples:\n  \
                  voice Hello world\n  \
                  voice -v am_adam \"How are you today?\"\n  \
                  echo \"Hello\" | voice\n  \
                  voice -f speech.txt -o output.wav\n  \
                  voice --phonemes \"hɛloʊ wɜːld\"\n  \
                  voice --markdown -f post.mdx\n  \
                  voice --sub nteract=enteract -f post.mdx\n  \
                  voice --sub-file .voice-subs --markdown -f post.mdx"
)]
struct Args {
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

fn resolve_text(args: &Args) -> Result<String, String> {
    // --phonemes takes a completely separate path
    if args.phonemes.is_some() {
        return Err("phonemes".into()); // sentinel, not a real error
    }

    // -f / --input-file
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

    // Positional text args
    if !args.text.is_empty() {
        return Ok(args.text.join(" "));
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

/// Strip markdown formatting from text, producing clean prose for TTS.
fn strip_markdown(text: &str) -> String {
    let mut out = String::new();
    let mut in_frontmatter = false;
    let mut frontmatter_count = 0;

    for line in text.lines() {
        let trimmed = line.trim();

        // Handle YAML frontmatter (--- delimited)
        if trimmed == "---" {
            frontmatter_count += 1;
            if frontmatter_count == 1 {
                in_frontmatter = true;
                continue;
            } else if frontmatter_count == 2 {
                in_frontmatter = false;
                continue;
            }
        }
        if in_frontmatter {
            continue;
        }

        // Skip blank lines
        if trimmed.is_empty() {
            out.push('\n');
            continue;
        }

        let mut line = line.to_string();

        // Strip heading markers: "## Foo" -> "Foo"
        if line.trim_start().starts_with('#') {
            line = line
                .trim_start()
                .trim_start_matches('#')
                .trim_start()
                .to_string();
        }

        // Strip bold markers: **text** and __text__
        line = line.replace("**", "");
        line = line.replace("__", "");

        // Strip remaining italic markers: *text*
        line = line.replace("*", "");

        // Strip inline code backticks
        line = line.replace("`", "");

        // Strip numbered list prefixes: "1. Foo" -> "Foo"
        let stripped = line.trim_start();
        if let Some(rest) = stripped.strip_prefix(|c: char| c.is_ascii_digit()) {
            // Handle multi-digit: consume remaining digits
            let rest = rest.trim_start_matches(|c: char| c.is_ascii_digit());
            if let Some(rest) = rest.strip_prefix(". ") {
                line = rest.to_string();
            }
        }

        // Strip bullet markers: "- Foo" or "* Foo" (already stripped * above, handle -)
        let stripped = line.trim_start();
        if let Some(rest) = stripped.strip_prefix("- ") {
            line = rest.to_string();
        }

        out.push_str(&line);
        out.push('\n');
    }

    out
}

/// Apply word-level text substitutions (case-insensitive on match, preserves replacement as-is).
fn apply_substitutions(text: &str, subs: &[(String, String)]) -> String {
    let mut result = text.to_string();
    for (from, to) in subs {
        result = result.replace(from.as_str(), to.as_str());
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

fn main() {
    let args = Args::parse();

    // Start model loading in a background thread immediately — this is the
    // slowest startup step (~200ms) and can run while we resolve text + G2P.
    let model_handle = std::thread::spawn(|| voice_tts::load_model(MODEL_REPO));

    // Resolve phoneme chunks (text resolution + G2P are fast with the
    // embedded perceptron tagger, ~1-2ms total).
    let phoneme_chunks: Vec<String> = if let Some(phonemes) = &args.phonemes {
        vec![phonemes.clone()]
    } else {
        match resolve_text(&args) {
            Ok(text) => {
                let text = if args.markdown {
                    strip_markdown(&text)
                } else {
                    text
                };
                // Resolve sub file: explicit --sub-file, or auto-discover .voice-subs
                let sub_file = args.sub_file.clone().or_else(find_sub_file);
                if let Some(ref path) = sub_file {
                    eprintln!("Using substitutions from {}", path.display());
                }
                let (subs, phoneme_overrides) = collect_subs(&args.subs, sub_file.as_deref());
                let text = if subs.is_empty() {
                    text
                } else {
                    apply_substitutions(&text, &subs)
                };
                eprintln!("Converting text to phonemes...");
                let chunks_result = if phoneme_overrides.is_empty() {
                    voice_g2p::text_to_phoneme_chunks(&text)
                } else {
                    voice_g2p::text_to_phoneme_chunks_with_overrides(&text, &phoneme_overrides)
                };
                match chunks_result {
                    Ok(chunks) => {
                        for (i, chunk) in chunks.iter().enumerate() {
                            eprintln!("  chunk {}: {}", i + 1, chunk);
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
    let voice = match voice_tts::load_voice(&args.voice, Some(MODEL_REPO)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load voice '{}': {e}", args.voice);
            eprintln!("Available voices include: af_heart, af_bella, af_nicole, af_sarah, af_sky,");
            eprintln!("  am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis");
            std::process::exit(1);
        }
    };

    // Wait for model loading to finish.
    let mut model = match model_handle.join().expect("model loading thread panicked") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            eprintln!("The model will be downloaded from HuggingFace on first run.");
            eprintln!("Check your network connection and try again.");
            std::process::exit(1);
        }
    };

    let sample_rate = model.sample_rate as u32;

    if let Some(output_path) = &args.output {
        // File output: batch generate all chunks, then write WAV
        // (WAV header needs total sample count upfront).
        generate_to_file(
            &mut model,
            &voice,
            &phoneme_chunks,
            args.speed,
            sample_rate,
            output_path,
        );
    } else {
        // Streaming playback: generate chunks and feed them to rodio as
        // they're ready. The first chunk starts playing immediately while
        // subsequent chunks are generated in the background.
        stream_playback(&mut model, &voice, &phoneme_chunks, args.speed, sample_rate);
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
    eprintln!("Generating audio...");
    let mut all_samples: Vec<f32> = Vec::new();

    for (i, phonemes) in chunks.iter().enumerate() {
        if phonemes.is_empty() {
            continue;
        }
        if chunks.len() > 1 {
            eprintln!("  generating chunk {}/{}...", i + 1, chunks.len());
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
    eprintln!("Saved to {}", output_path.display());
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
        if phonemes.is_empty() {
            continue;
        }
        if chunks.len() > 1 {
            eprintln!("  generating chunk {}/{}...", i + 1, chunks.len());
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

    player.sleep_until_end();
}
