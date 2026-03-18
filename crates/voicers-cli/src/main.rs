use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "voicers", about = "Kokoro TTS from the command line")]
struct Args {
    /// HuggingFace model repo or local path
    #[arg(long, default_value = "prince-canuma/Kokoro-82M")]
    model: String,

    /// Plain English text to synthesize
    #[arg(long, group = "input")]
    text: Option<String>,

    /// Raw phoneme string to synthesize (IPA)
    #[arg(long, group = "input")]
    phonemes: Option<String>,

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

    // Resolve phoneme chunks from either --text or --phonemes
    let phoneme_chunks: Vec<String> = if let Some(text) = &args.text {
        eprintln!("Converting text to phonemes...");
        match voicers_g2p::text_to_phoneme_chunks(text) {
            Ok(chunks) => {
                for (i, chunk) in chunks.iter().enumerate() {
                    eprintln!("  chunk {}: {}", i + 1, chunk);
                }
                chunks
            }
            Err(e) => {
                eprintln!("G2P error: {}", e);
                std::process::exit(1);
            }
        }
    } else if let Some(phonemes) = &args.phonemes {
        vec![phonemes.clone()]
    } else {
        eprintln!("Error: provide either --text or --phonemes");
        std::process::exit(1);
    };

    eprintln!("Loading model from {}...", args.model);
    let mut model = voicers::load_model(&args.model).expect("Failed to load model");

    eprintln!("Loading voice '{}'...", args.voice);
    let voice = voicers::load_voice(&args.voice, Some(&args.model)).expect("Failed to load voice");

    let sample_rate = model.sample_rate as u32;

    // Generate audio for each chunk and collect samples
    eprintln!("Generating audio...");
    let mut all_samples: Vec<f32> = Vec::new();

    for (i, phonemes) in phoneme_chunks.iter().enumerate() {
        if phonemes.is_empty() {
            continue;
        }
        if phoneme_chunks.len() > 1 {
            eprintln!("  generating chunk {}/{}...", i + 1, phoneme_chunks.len());
        }
        let audio = voicers::generate(&mut model, phonemes, &voice, args.speed)
            .expect("Failed to generate audio");

        let samples: Vec<f32> = audio.as_slice().to_vec();
        all_samples.extend_from_slice(&samples);
    }

    // Save combined audio
    let output_path = args
        .output
        .unwrap_or_else(|| PathBuf::from("output.wav"));

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&output_path, spec).expect("Failed to create WAV");
    for s in &all_samples {
        writer.write_sample(*s).expect("Failed to write sample");
    }
    writer.finalize().expect("Failed to finalize WAV");
    eprintln!("Saved to {}", output_path.display());

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
