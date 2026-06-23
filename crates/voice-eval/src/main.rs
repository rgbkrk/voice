use std::path::PathBuf;

use clap::Parser;
use serde::Serialize;

const DEFAULT_TTS_MODEL: &str = "prince-canuma/Kokoro-82M";
const DEFAULT_TTS_BACKEND: &str = "kokoro";
const DEFAULT_STT_BACKEND: &str = "whisper";

#[derive(Debug, Parser)]
#[command(
    name = "voice-eval",
    about = "Evaluate generated TTS audio by transcribing it with Whisper STT"
)]
struct Args {
    /// Text to synthesize and evaluate.
    #[arg(long, default_value = "The quick brown fox jumps over the lazy dog.")]
    text: String,

    /// Existing audio file to transcribe and score instead of synthesizing.
    #[arg(long = "input-wav")]
    input_wav: Option<PathBuf>,

    /// Reference text to score against when --input-wav is used.
    #[arg(long = "expected-text")]
    expected_text: Option<String>,

    /// TTS backend to synthesize with.
    #[arg(long = "tts-backend", default_value = DEFAULT_TTS_BACKEND)]
    tts_backend: String,

    /// TTS voice name.
    #[arg(short, long, default_value = "af_heart")]
    voice: String,

    /// TTS model repo or local directory.
    #[arg(long = "tts-model", default_value = DEFAULT_TTS_MODEL)]
    tts_model: String,

    /// STT backend to transcribe with.
    #[arg(long = "stt-backend", default_value = DEFAULT_STT_BACKEND)]
    stt_backend: String,

    /// STT model repo or local directory.
    #[arg(long = "stt-model")]
    stt_model: Option<String>,

    /// Maximum generated tokens for Voxtral STT probes.
    #[arg(long = "stt-max-tokens")]
    stt_max_tokens: Option<usize>,

    /// Speech speed factor.
    #[arg(short, long, default_value_t = 1.0)]
    speed: f32,

    /// Use Kokoro stochastic synthesis instead of deterministic synthesis.
    #[arg(long)]
    stochastic: bool,

    /// Maximum autoregressive frames for Voxtral synthesis.
    #[arg(long = "max-frames", default_value_t = 256)]
    max_frames: usize,

    /// Deterministic seed for Voxtral flow noise.
    #[arg(long, default_value_t = 0x5658_5452_414c)]
    seed: u64,

    /// Flow matching steps per Voxtral audio frame.
    #[arg(long = "flow-steps", default_value_t = 7)]
    flow_steps: usize,

    /// Write the generated TTS audio to this WAV path.
    #[arg(long = "output-wav")]
    output_wav: Option<PathBuf>,

    /// Print the report as pretty JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Serialize)]
struct EvalReport {
    text: String,
    voice: String,
    tts_model: String,
    stt_backend: String,
    stt_model: String,
    synthesis_mode: &'static str,
    sample_rate: u32,
    sample_count: usize,
    duration_seconds: f32,
    phoneme_chunks: Vec<String>,
    transcription: String,
    normalized_reference: String,
    normalized_hypothesis: String,
    reference_word_count: usize,
    word_error_count: usize,
    word_error_rate: Option<f32>,
    stt_token_count: usize,
    stt_sample_rate: u32,
    input_wav: Option<PathBuf>,
    output_wav: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let reference_text = args
        .expected_text
        .clone()
        .unwrap_or_else(|| args.text.clone());
    let (samples, sample_rate, phoneme_chunks, synthesis_mode) =
        if let Some(input_wav) = &args.input_wav {
            let audio = voice_stt::load_audio_file(input_wav)?;
            (audio.samples, audio.sample_rate, Vec::new(), "input-wav")
        } else if args.tts_backend == "voxtral" {
            let (samples, sample_rate) = synthesize_voxtral(&args)?;
            (samples, sample_rate, Vec::new(), "voxtral-native")
        } else if args.tts_backend == "kokoro" {
            let phoneme_chunks = voice_g2p::text_to_phoneme_chunks(&args.text)?;
            let mode = if args.stochastic {
                voice_tts::SynthesisMode::Stochastic
            } else {
                voice_tts::SynthesisMode::Deterministic
            };
            let (samples, sample_rate) = synthesize_kokoro(&args, &phoneme_chunks, mode)?;
            (
                samples,
                sample_rate,
                phoneme_chunks,
                if args.stochastic {
                    "stochastic"
                } else {
                    "deterministic"
                },
            )
        } else {
            return Err(format!(
                "unsupported --tts-backend {:?}; expected kokoro or voxtral",
                args.tts_backend
            )
            .into());
        };

    if args.input_wav.is_some() && args.output_wav.is_some() {
        return Err("--output-wav cannot be used with --input-wav".into());
    }

    if let Some(path) = &args.output_wav {
        voice_tts::save_wav(&samples, path, sample_rate)?;
    }

    let stt_backend = voice_stt::SttBackend::parse(&args.stt_backend)?;
    let stt_model_path = args
        .stt_model
        .clone()
        .unwrap_or_else(|| voice_stt::default_model_for_backend(stt_backend).to_string());
    let mut stt_model = voice_stt::load_backend_model(stt_backend, &stt_model_path)?;
    if let Some(max_new_tokens) = args.stt_max_tokens {
        stt_model.set_max_new_tokens(max_new_tokens);
    }
    let transcription = stt_model.transcribe_audio(&samples, sample_rate)?;
    let wer = word_error_rate(&reference_text, &transcription.text);
    let duration_seconds = if sample_rate == 0 {
        0.0
    } else {
        samples.len() as f32 / sample_rate as f32
    };

    let report = EvalReport {
        text: reference_text,
        voice: args.voice,
        tts_model: args.tts_model,
        stt_backend: stt_backend.as_str().to_string(),
        stt_model: stt_model_path,
        synthesis_mode,
        sample_rate,
        sample_count: samples.len(),
        duration_seconds,
        phoneme_chunks,
        transcription: transcription.text,
        normalized_reference: wer.reference_words.join(" "),
        normalized_hypothesis: wer.hypothesis_words.join(" "),
        reference_word_count: wer.reference_words.len(),
        word_error_count: wer.distance,
        word_error_rate: wer.rate,
        stt_token_count: transcription.tokens.len(),
        stt_sample_rate: transcription.sample_rate,
        input_wav: args.input_wav,
        output_wav: args.output_wav,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_text_report(&report);
    }

    Ok(())
}

fn synthesize_kokoro(
    args: &Args,
    phoneme_chunks: &[String],
    synthesis_mode: voice_tts::SynthesisMode,
) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut model = voice_tts::load_model(&args.tts_model)?;
    let sample_rate = model.sample_rate;
    let voice = model.load_voice(&args.voice, Some(&args.tts_model))?;
    let mut samples = Vec::new();

    for phonemes in phoneme_chunks {
        if phonemes.is_empty() {
            continue;
        }
        let chunk = voice_tts::generate_with_mode(
            &mut model,
            phonemes,
            &voice,
            args.speed,
            synthesis_mode,
        )?;
        samples.extend_from_slice(&chunk);
    }

    Ok((samples, sample_rate))
}

fn synthesize_voxtral(args: &Args) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let model = voice_voxtral::VoxtralModel::load(&args.tts_model)?;
    let audio = model.generate_audio_default(
        &args.text,
        &args.voice,
        voice_voxtral::VoxtralGenerationOptions {
            max_frames: args.max_frames,
            seed: args.seed,
            flow_steps: args.flow_steps,
            ..Default::default()
        },
    )?;
    Ok((audio.samples, audio.sample_rate))
}

fn print_text_report(report: &EvalReport) {
    println!("reference: {}", report.text);
    println!("hypothesis: {}", report.transcription);
    println!("normalized reference: {}", report.normalized_reference);
    println!("normalized hypothesis: {}", report.normalized_hypothesis);
    match report.word_error_rate {
        Some(rate) => println!(
            "WER: {:.2}% ({}/{})",
            rate * 100.0,
            report.word_error_count,
            report.reference_word_count
        ),
        None => println!("WER: n/a (empty reference)"),
    }
    println!(
        "audio: {:.2}s, {} Hz, {} samples",
        report.duration_seconds, report.sample_rate, report.sample_count
    );
    println!(
        "stt: {} ({} Hz) using {}",
        report.stt_backend, report.stt_sample_rate, report.stt_model
    );
    if let Some(path) = &report.output_wav {
        println!("wav: {}", path.display());
    }
    if let Some(path) = &report.input_wav {
        println!("input wav: {}", path.display());
    }
}

#[derive(Debug, PartialEq)]
struct Wer {
    reference_words: Vec<String>,
    hypothesis_words: Vec<String>,
    distance: usize,
    rate: Option<f32>,
}

fn word_error_rate(reference: &str, hypothesis: &str) -> Wer {
    let reference_words = normalize_words(reference);
    let hypothesis_words = normalize_words(hypothesis);
    let distance = levenshtein_words(&reference_words, &hypothesis_words);
    let rate = if reference_words.is_empty() {
        None
    } else {
        Some(distance as f32 / reference_words.len() as f32)
    };

    Wer {
        reference_words,
        hypothesis_words,
        distance,
        rate,
    }
}

fn normalize_words(text: &str) -> Vec<String> {
    let mut normalized = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '\'' {
            normalized.extend(ch.to_lowercase());
        } else {
            normalized.push(' ');
        }
    }
    normalized.split_whitespace().map(str::to_string).collect()
}

fn levenshtein_words(reference: &[String], hypothesis: &[String]) -> usize {
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut current = vec![0; hypothesis.len() + 1];

    for (i, reference_word) in reference.iter().enumerate() {
        current[0] = i + 1;
        for (j, hypothesis_word) in hypothesis.iter().enumerate() {
            let substitution = previous[j] + usize::from(reference_word != hypothesis_word);
            let insertion = current[j] + 1;
            let deletion = previous[j + 1] + 1;
            current[j + 1] = substitution.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[hypothesis.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_words_for_eval() {
        assert_eq!(
            normalize_words("Hello, WORLD! It isn't 1999."),
            vec!["hello", "world", "it", "isn't", "1999"]
        );
    }

    #[test]
    fn computes_exact_word_error_rate() {
        let wer = word_error_rate("hello world", "hello world");
        assert_eq!(wer.distance, 0);
        assert_eq!(wer.rate, Some(0.0));
    }

    #[test]
    fn computes_substitution_word_error_rate() {
        let wer = word_error_rate("the quick brown fox", "the quick blue fox");
        assert_eq!(wer.distance, 1);
        assert_eq!(wer.rate, Some(0.25));
    }

    #[test]
    fn parses_voxtral_stt_backend_args() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--input-wav",
            "/tmp/known.wav",
            "--expected-text",
            "known words",
            "--stt-backend",
            "voxtral",
            "--stt-model",
            "/tmp/voxtral-realtime",
            "--stt-max-tokens",
            "4",
        ])
        .unwrap();

        assert_eq!(args.stt_backend, "voxtral");
        assert_eq!(args.stt_model.as_deref(), Some("/tmp/voxtral-realtime"));
        assert_eq!(args.stt_max_tokens, Some(4));
        assert_eq!(args.input_wav, Some(PathBuf::from("/tmp/known.wav")));
    }
}
