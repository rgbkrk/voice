use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "voicers", about = "Kokoro TTS from the command line")]
struct Args {
    /// HuggingFace model repo or local path
    #[arg(long, default_value = "prince-canuma/Kokoro-82M")]
    model: String,

    /// Phoneme string to synthesize
    #[arg(long)]
    phonemes: String,

    /// Voice name (e.g. af_heart, am_adam)
    #[arg(long, default_value = "af_heart")]
    voice: String,

    /// Output WAV file path
    #[arg(long)]
    output: Option<PathBuf>,

    /// Play audio after generation
    #[arg(long)]
    play: bool,

    /// Speech speed factor (1.0 = normal)
    #[arg(long, default_value = "1.0")]
    speed: f32,
}

fn main() {
    let args = Args::parse();

    eprintln!("Loading model from {}...", args.model);
    let mut model = voicers::load_model(&args.model).expect("Failed to load model");

    eprintln!("Loading voice '{}'...", args.voice);
    let voice = voicers::load_voice(&args.voice, Some(&args.model)).expect("Failed to load voice");

    eprintln!("Generating audio...");
    let audio =
        voicers::generate(&mut model, &args.phonemes, &voice, args.speed)
            .expect("Failed to generate audio");

    let sample_rate = model.sample_rate as u32;

    // Save to file
    let output_path = args
        .output
        .unwrap_or_else(|| PathBuf::from("output.wav"));
    voicers::save_wav(&audio, &output_path, sample_rate).expect("Failed to save WAV");
    eprintln!("Saved to {}", output_path.display());

    // Play if requested
    if args.play {
        play_wav(&output_path);
    }
}

fn play_wav(path: &std::path::Path) {
    use rodio::{Decoder, OutputStream, Sink};
    use std::fs::File;
    use std::io::BufReader;

    let (_stream, stream_handle) = OutputStream::try_default().expect("Failed to open audio output");
    let sink = Sink::try_new(&stream_handle).expect("Failed to create sink");

    let file = File::open(path).expect("Failed to open WAV file");
    let source = Decoder::new(BufReader::new(file)).expect("Failed to decode WAV");
    sink.append(source);
    sink.sleep_until_end();
}
