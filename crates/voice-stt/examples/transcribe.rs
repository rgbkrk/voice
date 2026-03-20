//! Example: transcribe a WAV file using Moonshine.
//!
//! Usage:
//!     cargo run -p voice-stt --example transcribe -- /path/to/audio.wav
//!
//! The WAV file must be 16kHz mono (16-bit or 32-bit float).
//! Generate a test file with:
//!     voice -o test.wav "Hello, this is a test."
//!     ffmpeg -i test.wav -ar 16000 -acodec pcm_s16le test_16k.wav

use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        eprintln!();
        eprintln!("The WAV file must be 16kHz mono.");
        eprintln!("Generate a test file with:");
        eprintln!("  voice -o test.wav \"Hello, this is a test.\"");
        eprintln!("  ffmpeg -i test.wav -ar 16000 -acodec pcm_s16le test_16k.wav");
        std::process::exit(1);
    }
    let audio_path = &args[1];

    let repo =
        env::var("MOONSHINE_MODEL").unwrap_or_else(|_| "UsefulSensors/moonshine-tiny".to_string());

    eprintln!("Loading model: {repo}");
    let t0 = Instant::now();
    let mut model = voice_stt::load_model(&repo).expect("Failed to load model");
    eprintln!("Model loaded in {:.2}s", t0.elapsed().as_secs_f64());

    eprintln!("Loading tokenizer...");
    let tokenizer = voice_stt::load_tokenizer(&repo).expect("Failed to load tokenizer");

    eprintln!("Transcribing: {audio_path}");
    let t1 = Instant::now();

    // Load the WAV file
    let samples = load_wav_16k(audio_path);
    let duration_secs = samples.len() as f64 / 16000.0;
    eprintln!(
        "Audio: {:.2}s ({} samples at 16kHz)",
        duration_secs,
        samples.len()
    );

    let result =
        voice_stt::transcribe_audio_with_tokenizer(&mut model, &samples, 16000, &tokenizer)
            .expect("Transcription failed");

    let elapsed = t1.elapsed().as_secs_f64();
    let rtf = elapsed / duration_secs;

    println!("{}", result.text);
    eprintln!();
    eprintln!(
        "Tokens: {} generated in {:.2}s",
        result.tokens.len(),
        elapsed
    );
    eprintln!("RTF: {:.2}x (< 1.0 = faster than real-time)", rtf);
}

/// Load a 16kHz WAV file as mono f32 samples.
///
/// Handles both 16-bit integer and 32-bit float WAV formats.
fn load_wav_16k(path: &str) -> Vec<f32> {
    let reader = hound::WavReader::open(path).unwrap_or_else(|e| {
        eprintln!("Failed to open {path}: {e}");
        std::process::exit(1);
    });

    let spec = reader.spec();
    if spec.sample_rate != 16000 {
        eprintln!(
            "Error: WAV sample rate is {}Hz, expected 16000Hz.",
            spec.sample_rate
        );
        eprintln!("Convert with: ffmpeg -i {path} -ar 16000 -acodec pcm_s16le output_16k.wav");
        std::process::exit(1);
    }

    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
    };

    // Mix to mono if needed
    if channels > 1 {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    }
}
