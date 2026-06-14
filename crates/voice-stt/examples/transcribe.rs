//! Example: transcribe an audio file using Whisper.
//!
//! Usage:
//!     cargo run -p voice-stt --example transcribe -- /path/to/audio.ogg
//!
//! Generate a test file with:
//!     voice say --format ogg-opus -o test.ogg "Hello, this is a test."

use std::env;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio-file>", args[0]);
        eprintln!();
        eprintln!("WAV input is decoded directly; compressed formats use ffmpeg.");
        eprintln!("Generate a test file with:");
        eprintln!("  voice say --format ogg-opus -o test.ogg \"Hello, this is a test.\"");
        std::process::exit(1);
    }
    let audio_path = &args[1];

    let repo =
        env::var("STT_MODEL").unwrap_or_else(|_| "distil-whisper/distil-medium.en".to_string());

    eprintln!("Loading model: {repo}");
    let t0 = Instant::now();
    let mut model = voice_stt::load_model(&repo).expect("Failed to load model");
    eprintln!("Model loaded in {:.2}s", t0.elapsed().as_secs_f64());

    eprintln!("Transcribing: {audio_path}");
    let t1 = Instant::now();

    let audio = voice_stt::load_audio_file(Path::new(audio_path)).expect("Failed to decode audio");
    let duration_secs = audio.samples.len() as f64 / audio.sample_rate as f64;
    eprintln!(
        "Audio: {:.2}s ({} samples at {}Hz)",
        duration_secs,
        audio.samples.len(),
        audio.sample_rate
    );

    let result = voice_stt::transcribe_audio(&mut model, &audio.samples, audio.sample_rate)
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
