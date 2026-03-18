use clap::Parser;
use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;

const MODEL_REPO: &str = "prince-canuma/Kokoro-82M";

#[derive(Parser, Debug)]
#[command(
    name = "voice",
    about = "Kokoro TTS from the command line",
    after_help = "If no text, --phonemes, or -f is given, reads from stdin.\n\n\
                  Examples:\n  \
                  voice Hello world\n  \
                  voice -v am_adam \"How are you today?\"\n  \
                  echo \"Hello\" | voice\n  \
                  voice -f speech.txt -o output.wav\n  \
                  voice --phonemes \"hɛloʊ wɜːld\""
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

fn main() {
    let args = Args::parse();

    // Resolve phoneme chunks
    let phoneme_chunks: Vec<String> = if let Some(phonemes) = &args.phonemes {
        vec![phonemes.clone()]
    } else {
        match resolve_text(&args) {
            Ok(text) => {
                eprintln!("Converting text to phonemes...");
                match voice_g2p::text_to_phoneme_chunks(&text) {
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

    eprintln!("Loading voice '{}'...", args.voice);
    let voice = match voice_tts::load_voice(&args.voice, Some(MODEL_REPO)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load voice '{}': {e}", args.voice);
            eprintln!("Available voices include: af_heart, af_bella, af_nicole, af_sarah, af_sky,");
            eprintln!("  am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis");
            std::process::exit(1);
        }
    };

    eprintln!("Loading model...");
    let mut model = match voice_tts::load_model(MODEL_REPO) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            eprintln!("The model will be downloaded from HuggingFace on first run.");
            eprintln!("Check your network connection and try again.");
            std::process::exit(1);
        }
    };

    let sample_rate = model.sample_rate as u32;

    eprintln!("Generating audio...");
    let mut all_samples: Vec<f32> = Vec::new();

    for (i, phonemes) in phoneme_chunks.iter().enumerate() {
        if phonemes.is_empty() {
            continue;
        }
        if phoneme_chunks.len() > 1 {
            eprintln!("  generating chunk {}/{}...", i + 1, phoneme_chunks.len());
        }
        match voice_tts::generate(&mut model, phonemes, &voice, args.speed) {
            Ok(audio) => {
                all_samples.extend_from_slice(audio.as_slice());
            }
            Err(e) => {
                eprintln!("Failed to generate audio for chunk {}: {e}", i + 1);
                std::process::exit(1);
            }
        }
    }

    if let Some(output_path) = &args.output {
        // Write WAV to file
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
    } else {
        // Play directly from memory
        play_samples(&all_samples, sample_rate);
    }
}

fn play_samples(samples: &[f32], sample_rate: u32) {
    use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, Player};
    use std::num::NonZero;

    let mut stream = DeviceSinkBuilder::open_default_sink().expect("Failed to open audio output");
    stream.log_on_drop(false);
    let player = Player::connect_new(stream.mixer());

    let source = SamplesBuffer::new(
        NonZero::new(1u16).unwrap(),
        NonZero::new(sample_rate).unwrap(),
        samples.to_vec(),
    );
    player.append(source);
    player.sleep_until_end();
}
