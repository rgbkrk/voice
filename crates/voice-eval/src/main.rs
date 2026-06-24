use std::f64::consts::PI;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use serde::Serialize;

const DEFAULT_TTS_MODEL: &str = "prince-canuma/Kokoro-82M";
const DEFAULT_TTS_BACKEND: &str = "kokoro";
const DEFAULT_STT_BACKEND: &str = "whisper";
const DEFAULT_KOKORO_VOICE: &str = "af_heart";
const DEFAULT_VOXTRAL_VOICE: &str = "casual_male";

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

    /// Text to synthesize when it should differ from the scored reference text.
    #[arg(long = "synthesis-text", value_name = "TEXT")]
    synthesis_text: Option<String>,

    /// TTS backend to synthesize with.
    #[arg(long = "tts-backend", default_value = DEFAULT_TTS_BACKEND)]
    tts_backend: String,

    /// TTS voice name.
    #[arg(short, long)]
    voice: Option<String>,

    /// TTS model repo or local directory. Defaults to the backend's known model.
    #[arg(long = "tts-model")]
    tts_model: Option<String>,

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

    /// Run a Voxtral quality/speed matrix instead of a single evaluation.
    #[arg(long = "voxtral-matrix")]
    voxtral_matrix: bool,

    /// Additional prompt for --voxtral-matrix. Repeat for multiple prompts.
    #[arg(long = "matrix-text", value_name = "TEXT")]
    matrix_texts: Vec<String>,

    /// Synthesis text paired by index with --matrix-text.
    #[arg(long = "matrix-synthesis-text", value_name = "TEXT")]
    matrix_synthesis_texts: Vec<String>,

    /// Comma-separated max-frame values for --voxtral-matrix.
    #[arg(
        long = "matrix-max-frames",
        value_delimiter = ',',
        default_value = "32,40"
    )]
    matrix_max_frames: Vec<usize>,

    /// Comma-separated flow-step values for --voxtral-matrix.
    #[arg(
        long = "matrix-flow-steps",
        value_delimiter = ',',
        default_value = "5,6,7"
    )]
    matrix_flow_steps: Vec<usize>,

    /// Enable Voxtral language KV cache for Voxtral synthesis.
    #[arg(long = "voxtral-kv-cache")]
    voxtral_kv_cache: bool,

    /// Normalize compact Voxtral numeric synthesis text forms such as versions and times.
    #[arg(long = "voxtral-normalize-text")]
    voxtral_normalize_text: bool,

    /// Initial streaming codec chunk size for Voxtral matrix timing.
    #[arg(long = "voxtral-stream-begin-frames", default_value_t = 2)]
    voxtral_stream_begin_frames: usize,

    /// Synchronize Metal around Voxtral trace sections for profiling.
    #[arg(long = "voxtral-sync-trace")]
    voxtral_sync_trace: bool,

    /// Collect EOS rank/margin diagnostics for Voxtral semantic logits.
    #[arg(long = "voxtral-eos-scores")]
    voxtral_eos_scores: bool,

    /// Deterministic seed for Voxtral flow noise.
    #[arg(long, default_value_t = 0x5658_5452_414c)]
    seed: u64,

    /// Flow matching steps per Voxtral audio frame.
    #[arg(long = "flow-steps", default_value_t = 7)]
    flow_steps: usize,

    /// Write the generated TTS audio to this WAV path.
    #[arg(long = "output-wav")]
    output_wav: Option<PathBuf>,

    /// Directory for generated WAVs when --voxtral-matrix is set.
    #[arg(long = "output-dir")]
    output_dir: Option<PathBuf>,

    /// Directory for grayscale PGM spectrograms.
    #[arg(long = "spectrogram-dir")]
    spectrogram_dir: Option<PathBuf>,

    /// Print the report as pretty JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Serialize)]
struct AudioSegment {
    start_seconds: f64,
    end_seconds: f64,
    duration_seconds: f64,
    peak_dbfs: f64,
    rms_dbfs: f64,
}

#[derive(Debug, Clone, Serialize)]
struct AudioDiagnostics {
    duration_seconds: f64,
    peak_dbfs: f64,
    rms_dbfs: f64,
    active_threshold_dbfs: f64,
    active_duration_seconds: f64,
    active_span_seconds: f64,
    active_ratio: Option<f64>,
    active_segment_count: usize,
    long_gap_count: usize,
    longest_internal_gap_seconds: f64,
    gap_seconds: Vec<f64>,
    leading_silence_seconds: f64,
    trailing_silence_seconds: f64,
    leading_fragment_seconds: Option<f64>,
    trailing_fragment_seconds: Option<f64>,
    segments: Vec<AudioSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MatrixCase {
    reference_text: String,
    synthesis_text: String,
}

#[derive(Debug, Serialize)]
struct EvalReport {
    text: String,
    synthesis_text: String,
    voice: String,
    tts_model: String,
    stt_backend: String,
    stt_model: String,
    synthesis_mode: &'static str,
    voxtral_text_normalization: bool,
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
    audio_diagnostics: AudioDiagnostics,
    input_wav: Option<PathBuf>,
    output_wav: Option<PathBuf>,
    spectrogram_pgm: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct VoxtralMatrixReport {
    voice: String,
    tts_model: String,
    stt_backend: String,
    stt_model: String,
    model_load_ms: f64,
    matrix_max_frames: Vec<usize>,
    matrix_flow_steps: Vec<usize>,
    stream_begin_frames: usize,
    kv_cache: bool,
    sync_trace: bool,
    text_normalization: bool,
    eos_scores: bool,
    seed: u64,
    output_dir: Option<PathBuf>,
    spectrogram_dir: Option<PathBuf>,
    rows: Vec<VoxtralMatrixRow>,
}

#[derive(Debug, Serialize)]
struct VoxtralMatrixRow {
    text_index: usize,
    text: String,
    synthesis_text: String,
    max_frames: usize,
    flow_steps: usize,
    first_code_frame_ms: Option<f64>,
    first_audio_ms: Option<f64>,
    total_ms: f64,
    audio_duration_ms: f64,
    realtime_factor: Option<f64>,
    samples: usize,
    sample_rate: u32,
    audio_frames: usize,
    ended: bool,
    eos_frame: Option<usize>,
    semantic_code_count: usize,
    semantic_tail_codes: Vec<u32>,
    semantic_tail_unique_count: usize,
    semantic_tail_repeat_count: usize,
    semantic_eos_rank_tail: Vec<usize>,
    semantic_eos_margin_tail: Vec<f32>,
    semantic_best_eos_rank: Option<usize>,
    semantic_best_eos_margin: Option<f32>,
    codec_chunks: usize,
    language_ms: f64,
    language_ms_per_frame: Option<f64>,
    acoustic_ms: f64,
    acoustic_ms_per_frame: Option<f64>,
    decode_loop_ms: f64,
    decode_loop_ms_per_frame: Option<f64>,
    codec_ms: f64,
    codec_ms_per_chunk: Option<f64>,
    transcription: String,
    normalized_reference: String,
    normalized_hypothesis: String,
    reference_word_count: usize,
    word_error_count: usize,
    word_error_rate: Option<f32>,
    stt_token_count: usize,
    audio_diagnostics: AudioDiagnostics,
    output_wav: Option<PathBuf>,
    spectrogram_pgm: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.voxtral_matrix {
        return run_voxtral_matrix(&args);
    }

    let tts_model_path = tts_model_path(&args);
    let tts_voice = tts_voice(&args);
    if args.input_wav.is_some() && args.synthesis_text.is_some() {
        return Err("--synthesis-text cannot be used with --input-wav".into());
    }
    if args.input_wav.is_some() && args.voxtral_normalize_text {
        return Err("--voxtral-normalize-text cannot be used with --input-wav".into());
    }
    let reference_text = args
        .expected_text
        .clone()
        .unwrap_or_else(|| args.text.clone());
    let synthesis_text = args
        .synthesis_text
        .clone()
        .unwrap_or_else(|| args.text.clone());
    let synthesis_text = maybe_normalize_voxtral_text(&args, synthesis_text);
    let (samples, sample_rate, phoneme_chunks, synthesis_mode) = if let Some(input_wav) =
        &args.input_wav
    {
        let audio = voice_stt::load_audio_file(input_wav)?;
        (audio.samples, audio.sample_rate, Vec::new(), "input-wav")
    } else if args.tts_backend == "voxtral" {
        let (samples, sample_rate) = synthesize_voxtral(&args, &synthesis_text, &tts_voice)?;
        (samples, sample_rate, Vec::new(), "voxtral-native")
    } else if args.tts_backend == "kokoro" {
        let phoneme_chunks = voice_g2p::text_to_phoneme_chunks(&synthesis_text)?;
        let mode = if args.stochastic {
            voice_tts::SynthesisMode::Stochastic
        } else {
            voice_tts::SynthesisMode::Deterministic
        };
        let (samples, sample_rate) = synthesize_kokoro(&args, &phoneme_chunks, mode, &tts_voice)?;
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
    let audio_diagnostics = analyze_audio(&samples, sample_rate);
    let spectrogram_pgm = maybe_write_spectrogram_pgm(
        args.spectrogram_dir.as_deref(),
        "eval.pgm",
        &samples,
        sample_rate,
    )?;

    let report = EvalReport {
        text: reference_text,
        synthesis_text,
        voice: tts_voice,
        tts_model: tts_model_path,
        stt_backend: stt_backend.as_str().to_string(),
        stt_model: stt_model_path,
        synthesis_mode,
        voxtral_text_normalization: args.tts_backend == "voxtral" && args.voxtral_normalize_text,
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
        audio_diagnostics,
        input_wav: args.input_wav.clone(),
        output_wav: args.output_wav.clone(),
        spectrogram_pgm,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_text_report(&report);
    }

    Ok(())
}

fn run_voxtral_matrix(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.input_wav.is_some() {
        return Err("--voxtral-matrix cannot be used with --input-wav".into());
    }
    if args.output_wav.is_some() {
        return Err("--output-wav cannot be used with --voxtral-matrix; use --output-dir".into());
    }
    if args.expected_text.is_some() {
        return Err("--expected-text cannot be used with --voxtral-matrix".into());
    }
    if args.synthesis_text.is_some() {
        return Err(
            "--synthesis-text cannot be used with --voxtral-matrix; use --matrix-synthesis-text"
                .into(),
        );
    }
    if args.tts_backend != "voxtral" {
        return Err("--voxtral-matrix requires --tts-backend voxtral".into());
    }
    if args.matrix_max_frames.is_empty() {
        return Err("--matrix-max-frames must include at least one value".into());
    }
    if args.matrix_flow_steps.is_empty() {
        return Err("--matrix-flow-steps must include at least one value".into());
    }
    if args.matrix_max_frames.contains(&0) {
        return Err("--matrix-max-frames values must be greater than zero".into());
    }
    if args.matrix_flow_steps.contains(&0) {
        return Err("--matrix-flow-steps values must be greater than zero".into());
    }
    if args.voxtral_stream_begin_frames == 0 {
        return Err("--voxtral-stream-begin-frames must be greater than zero".into());
    }

    let cases = matrix_cases(args)?;
    let tts_model_path = tts_model_path(args);
    let tts_voice = tts_voice(args);
    let (mut runtime, load_trace) =
        voice_voxtral::VoxtralTtsRuntime::load_default_with_trace(&tts_model_path)?;
    runtime.preload_voice(&tts_voice)?;

    let stt_backend = voice_stt::SttBackend::parse(&args.stt_backend)?;
    let stt_model_path = args
        .stt_model
        .clone()
        .unwrap_or_else(|| voice_stt::default_model_for_backend(stt_backend).to_string());
    let mut stt_model = voice_stt::load_backend_model(stt_backend, &stt_model_path)?;
    if let Some(max_new_tokens) = args.stt_max_tokens {
        stt_model.set_max_new_tokens(max_new_tokens);
    }

    let mut rows = Vec::new();
    for (text_index, case) in cases.iter().enumerate() {
        for &max_frames in &args.matrix_max_frames {
            for &flow_steps in &args.matrix_flow_steps {
                let streaming = voice_voxtral::VoxtralStreamingConfig {
                    chunk_frames_at_begin: args.voxtral_stream_begin_frames,
                    ..Default::default()
                };
                let options = voice_voxtral::VoxtralGenerationOptions {
                    max_frames,
                    seed: args.seed,
                    flow_steps,
                    use_kv_cache: args.voxtral_kv_cache,
                    synchronize_trace: args.voxtral_sync_trace,
                    trace_semantic_scores: args.voxtral_eos_scores,
                    ..Default::default()
                };

                let run_start = Instant::now();
                let (audio, trace) = runtime.generate_audio_streaming_with_trace(
                    &case.synthesis_text,
                    &tts_voice,
                    options,
                    streaming,
                    |_| Ok(()),
                )?;
                let total = run_start.elapsed();
                let output_wav = maybe_write_matrix_wav(
                    args.output_dir.as_deref(),
                    text_index,
                    max_frames,
                    flow_steps,
                    &audio.samples,
                    audio.sample_rate,
                )?;
                let spectrogram_pgm = maybe_write_spectrogram_pgm(
                    args.spectrogram_dir.as_deref(),
                    &format!(
                        "voxtral-text{}-max{}-flow{}.pgm",
                        text_index + 1,
                        max_frames,
                        flow_steps
                    ),
                    &audio.samples,
                    audio.sample_rate,
                )?;

                let transcription =
                    stt_model.transcribe_audio(&audio.samples, audio.sample_rate)?;
                let wer = word_error_rate(&case.reference_text, &transcription.text);
                let audio_diagnostics = analyze_audio(&audio.samples, audio.sample_rate);
                let total_ms = duration_ms(total);
                let audio_duration_ms = audio_duration_ms(audio.samples.len(), audio.sample_rate);
                let language_ms = duration_ms(trace.language);
                let acoustic_ms = duration_ms(trace.acoustic);
                let decode_loop_ms = duration_ms(trace.decode_loop);
                let codec_ms = duration_ms(trace.codec);
                let semantic_tail_codes = tail_semantic_codes(&trace.semantic_codes, 8);
                let semantic_tail_unique_count = unique_count(&semantic_tail_codes);
                let semantic_tail_repeat_count = repeated_tail_count(&trace.semantic_codes);
                let semantic_eos_rank_tail = tail_values(&trace.semantic_eos_ranks, 8);
                let semantic_eos_margin_tail = tail_values(&trace.semantic_eos_margins, 8);
                let semantic_best_eos_rank = trace.semantic_eos_ranks.iter().copied().min();
                let semantic_best_eos_margin =
                    trace.semantic_eos_margins.iter().copied().reduce(f32::min);

                rows.push(VoxtralMatrixRow {
                    text_index,
                    text: case.reference_text.clone(),
                    synthesis_text: case.synthesis_text.clone(),
                    max_frames,
                    flow_steps,
                    first_code_frame_ms: trace.first_frame.map(duration_ms),
                    first_audio_ms: trace.first_audio_chunk.map(duration_ms),
                    total_ms,
                    audio_duration_ms,
                    realtime_factor: ratio_ms(total_ms, audio_duration_ms),
                    samples: audio.samples.len(),
                    sample_rate: audio.sample_rate,
                    audio_frames: audio.frames,
                    ended: audio.ended,
                    eos_frame: trace.eos_frame,
                    semantic_code_count: trace.semantic_codes.len(),
                    semantic_tail_codes,
                    semantic_tail_unique_count,
                    semantic_tail_repeat_count,
                    semantic_eos_rank_tail,
                    semantic_eos_margin_tail,
                    semantic_best_eos_rank,
                    semantic_best_eos_margin,
                    codec_chunks: trace.codec_chunks,
                    language_ms,
                    language_ms_per_frame: per_unit(language_ms, audio.frames),
                    acoustic_ms,
                    acoustic_ms_per_frame: per_unit(acoustic_ms, audio.frames),
                    decode_loop_ms,
                    decode_loop_ms_per_frame: per_unit(decode_loop_ms, audio.frames),
                    codec_ms,
                    codec_ms_per_chunk: per_unit(codec_ms, trace.codec_chunks),
                    transcription: transcription.text,
                    normalized_reference: wer.reference_words.join(" "),
                    normalized_hypothesis: wer.hypothesis_words.join(" "),
                    reference_word_count: wer.reference_words.len(),
                    word_error_count: wer.distance,
                    word_error_rate: wer.rate,
                    stt_token_count: transcription.tokens.len(),
                    audio_diagnostics,
                    output_wav,
                    spectrogram_pgm,
                });
            }
        }
    }

    let report = VoxtralMatrixReport {
        voice: tts_voice,
        tts_model: tts_model_path,
        stt_backend: stt_backend.as_str().to_string(),
        stt_model: stt_model_path,
        model_load_ms: duration_ms(load_trace.total),
        matrix_max_frames: args.matrix_max_frames.clone(),
        matrix_flow_steps: args.matrix_flow_steps.clone(),
        stream_begin_frames: args.voxtral_stream_begin_frames,
        kv_cache: args.voxtral_kv_cache,
        sync_trace: args.voxtral_sync_trace,
        text_normalization: args.voxtral_normalize_text,
        eos_scores: args.voxtral_eos_scores,
        seed: args.seed,
        output_dir: args.output_dir.clone(),
        spectrogram_dir: args.spectrogram_dir.clone(),
        rows,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_voxtral_matrix_report(&report);
    }

    Ok(())
}

fn tts_model_path(args: &Args) -> String {
    args.tts_model.clone().unwrap_or_else(|| {
        if args.tts_backend == "voxtral" {
            voice_voxtral::DEFAULT_REPO.to_string()
        } else {
            DEFAULT_TTS_MODEL.to_string()
        }
    })
}

fn tts_voice(args: &Args) -> String {
    args.voice.clone().unwrap_or_else(|| {
        if args.tts_backend == "voxtral" {
            DEFAULT_VOXTRAL_VOICE.to_string()
        } else {
            DEFAULT_KOKORO_VOICE.to_string()
        }
    })
}

fn matrix_cases(args: &Args) -> Result<Vec<MatrixCase>, Box<dyn std::error::Error>> {
    let reference_texts = if args.matrix_texts.is_empty() {
        vec![args.text.clone()]
    } else {
        args.matrix_texts.clone()
    };
    if reference_texts.iter().any(|text| text.trim().is_empty()) {
        return Err("--matrix-text values must not be empty".into());
    }
    if args
        .matrix_synthesis_texts
        .iter()
        .any(|text| text.trim().is_empty())
    {
        return Err("--matrix-synthesis-text values must not be empty".into());
    }

    let synthesis_texts = if args.matrix_synthesis_texts.is_empty() {
        reference_texts.clone()
    } else {
        if args.matrix_synthesis_texts.len() != reference_texts.len() {
            return Err(format!(
                "--matrix-synthesis-text count ({}) must match --matrix-text count ({})",
                args.matrix_synthesis_texts.len(),
                reference_texts.len()
            )
            .into());
        }
        args.matrix_synthesis_texts.clone()
    };

    Ok(reference_texts
        .into_iter()
        .zip(synthesis_texts)
        .map(|(reference_text, synthesis_text)| MatrixCase {
            reference_text,
            synthesis_text: maybe_normalize_voxtral_text(args, synthesis_text),
        })
        .collect())
}

fn maybe_normalize_voxtral_text(args: &Args, text: String) -> String {
    if args.tts_backend == "voxtral" && args.voxtral_normalize_text {
        voice_voxtral::normalize_tts_text(&text)
    } else {
        text
    }
}

fn synthesize_kokoro(
    args: &Args,
    phoneme_chunks: &[String],
    synthesis_mode: voice_tts::SynthesisMode,
    voice_name: &str,
) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let tts_model_path = tts_model_path(args);
    let mut model = voice_tts::load_model(&tts_model_path)?;
    let sample_rate = model.sample_rate;
    let voice = model.load_voice(voice_name, Some(&tts_model_path))?;
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

fn synthesize_voxtral(
    args: &Args,
    text: &str,
    voice_name: &str,
) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let tts_model_path = tts_model_path(args);
    let model = voice_voxtral::VoxtralModel::load(&tts_model_path)?;
    let audio = model.generate_audio_default(
        text,
        voice_name,
        voice_voxtral::VoxtralGenerationOptions {
            max_frames: args.max_frames,
            seed: args.seed,
            flow_steps: args.flow_steps,
            use_kv_cache: args.voxtral_kv_cache,
            synchronize_trace: args.voxtral_sync_trace,
            ..Default::default()
        },
    )?;
    Ok((audio.samples, audio.sample_rate))
}

fn maybe_write_matrix_wav(
    output_dir: Option<&Path>,
    text_index: usize,
    max_frames: usize,
    flow_steps: usize,
    samples: &[f32],
    sample_rate: u32,
) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let Some(output_dir) = output_dir else {
        return Ok(None);
    };
    std::fs::create_dir_all(output_dir)?;
    let path = output_dir.join(format!(
        "voxtral-text{}-max{}-flow{}.wav",
        text_index + 1,
        max_frames,
        flow_steps
    ));
    voice_tts::save_wav(samples, &path, sample_rate)?;
    Ok(Some(path))
}

fn maybe_write_spectrogram_pgm(
    spectrogram_dir: Option<&Path>,
    file_name: &str,
    samples: &[f32],
    sample_rate: u32,
) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let Some(spectrogram_dir) = spectrogram_dir else {
        return Ok(None);
    };
    std::fs::create_dir_all(spectrogram_dir)?;
    let path = spectrogram_dir.join(file_name);
    let image = spectrogram_pgm(samples, sample_rate);
    let mut file = std::fs::File::create(&path)?;
    file.write_all(&image)?;
    Ok(Some(path))
}

fn spectrogram_pgm(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    const WINDOW_SIZE: usize = 256;
    const HOP_SIZE: usize = 128;
    const BINS: usize = 96;
    const DB_RANGE: f64 = 80.0;

    let frame_count = if samples.is_empty() {
        1
    } else {
        ((samples.len().saturating_sub(1)) / HOP_SIZE) + 1
    };
    let mut magnitudes = vec![0.0_f64; frame_count * BINS];
    let mut max_db = -120.0_f64;

    for frame in 0..frame_count {
        let start = frame * HOP_SIZE;
        for bin in 0..BINS {
            let mut real = 0.0_f64;
            let mut imag = 0.0_f64;
            for n in 0..WINDOW_SIZE {
                let sample = samples.get(start + n).copied().unwrap_or(0.0) as f64;
                let window = 0.5 - 0.5 * ((2.0 * PI * n as f64) / (WINDOW_SIZE - 1) as f64).cos();
                let angle = 2.0 * PI * bin as f64 * n as f64 / WINDOW_SIZE as f64;
                real += sample * window * angle.cos();
                imag -= sample * window * angle.sin();
            }
            let magnitude = (real.mul_add(real, imag * imag)).sqrt();
            let db = dbfs_amplitude(magnitude.max(1.0e-12));
            magnitudes[frame * BINS + bin] = db;
            max_db = max_db.max(db);
        }
    }

    let floor_db = max_db - DB_RANGE;
    let mut image = Vec::new();
    image.extend_from_slice(
        format!("P5\n# sample_rate {sample_rate}\n{frame_count} {BINS}\n255\n").as_bytes(),
    );
    for display_bin in (0..BINS).rev() {
        for frame in 0..frame_count {
            let db = magnitudes[frame * BINS + display_bin];
            let normalized = ((db - floor_db) / DB_RANGE).clamp(0.0, 1.0);
            image.push((normalized * 255.0).round() as u8);
        }
    }
    image
}

fn analyze_audio(samples: &[f32], sample_rate: u32) -> AudioDiagnostics {
    const FRAME_MS: u32 = 20;
    const HOP_MS: u32 = 10;
    const MERGE_GAP_SECONDS: f64 = 0.10;
    const MIN_SEGMENT_SECONDS: f64 = 0.03;
    const LONG_GAP_SECONDS: f64 = 0.25;
    const FRAGMENT_SECONDS: f64 = 0.25;

    let duration_seconds = seconds(samples.len(), sample_rate);
    let peak = peak_amplitude(samples);
    let rms = rms_amplitude(samples);
    let peak_dbfs = dbfs_amplitude(peak);
    let rms_dbfs = dbfs_amplitude(rms);
    let active_threshold_dbfs = (peak_dbfs - 35.0).max(-55.0);

    if samples.is_empty() || sample_rate == 0 {
        return AudioDiagnostics {
            duration_seconds,
            peak_dbfs,
            rms_dbfs,
            active_threshold_dbfs,
            active_duration_seconds: 0.0,
            active_span_seconds: 0.0,
            active_ratio: None,
            active_segment_count: 0,
            long_gap_count: 0,
            longest_internal_gap_seconds: 0.0,
            gap_seconds: Vec::new(),
            leading_silence_seconds: duration_seconds,
            trailing_silence_seconds: 0.0,
            leading_fragment_seconds: None,
            trailing_fragment_seconds: None,
            segments: Vec::new(),
        };
    }

    let frame_len = samples_for_duration(sample_rate, FRAME_MS as f64 / 1_000.0);
    let hop_len = samples_for_duration(sample_rate, HOP_MS as f64 / 1_000.0);
    let merge_gap_samples = samples_for_duration(sample_rate, MERGE_GAP_SECONDS);
    let min_segment_samples = samples_for_duration(sample_rate, MIN_SEGMENT_SECONDS);

    let mut raw_segments = Vec::new();
    let mut current: Option<(usize, usize)> = None;
    let mut start = 0;
    while start < samples.len() {
        let end = (start + frame_len).min(samples.len());
        let frame_rms_dbfs = dbfs_amplitude(rms_amplitude(&samples[start..end]));
        let is_active = frame_rms_dbfs >= active_threshold_dbfs && frame_rms_dbfs > -90.0;
        if is_active {
            match current {
                Some((segment_start, segment_end)) => {
                    if start.saturating_sub(segment_end) <= merge_gap_samples {
                        current = Some((segment_start, end));
                    } else {
                        push_raw_segment(
                            &mut raw_segments,
                            segment_start,
                            segment_end,
                            min_segment_samples,
                        );
                        current = Some((start, end));
                    }
                }
                None => current = Some((start, end)),
            }
        }
        start += hop_len;
    }
    if let Some((segment_start, segment_end)) = current {
        push_raw_segment(
            &mut raw_segments,
            segment_start,
            segment_end,
            min_segment_samples,
        );
    }

    let segments: Vec<AudioSegment> = raw_segments
        .into_iter()
        .map(|(start, end)| {
            let segment_samples = &samples[start..end];
            let start_seconds = seconds(start, sample_rate);
            let end_seconds = seconds(end, sample_rate);
            AudioSegment {
                start_seconds,
                end_seconds,
                duration_seconds: end_seconds - start_seconds,
                peak_dbfs: dbfs_amplitude(peak_amplitude(segment_samples)),
                rms_dbfs: dbfs_amplitude(rms_amplitude(segment_samples)),
            }
        })
        .collect();

    let active_duration_seconds: f64 = segments
        .iter()
        .map(|segment| segment.duration_seconds)
        .sum();
    let active_span_seconds = match (segments.first(), segments.last()) {
        (Some(first), Some(last)) => last.end_seconds - first.start_seconds,
        _ => 0.0,
    };
    let active_ratio =
        (active_span_seconds > 0.0).then_some(active_duration_seconds / active_span_seconds);
    let gap_seconds: Vec<f64> = segments
        .windows(2)
        .map(|pair| pair[1].start_seconds - pair[0].end_seconds)
        .collect();
    let longest_internal_gap_seconds = gap_seconds.iter().copied().fold(0.0, f64::max);
    let long_gap_count = gap_seconds
        .iter()
        .filter(|gap_seconds| **gap_seconds >= LONG_GAP_SECONDS)
        .count();
    let leading_silence_seconds = segments
        .first()
        .map(|segment| segment.start_seconds)
        .unwrap_or(duration_seconds);
    let trailing_silence_seconds = segments
        .last()
        .map(|segment| (duration_seconds - segment.end_seconds).max(0.0))
        .unwrap_or(0.0);
    let leading_fragment_seconds = (segments.len() > 1)
        .then(|| segments.first())
        .flatten()
        .and_then(|segment| {
            (segment.duration_seconds < FRAGMENT_SECONDS).then_some(segment.duration_seconds)
        });
    let trailing_fragment_seconds = (segments.len() > 1)
        .then(|| segments.last())
        .flatten()
        .and_then(|segment| {
            (segment.duration_seconds < FRAGMENT_SECONDS).then_some(segment.duration_seconds)
        });

    AudioDiagnostics {
        duration_seconds,
        peak_dbfs,
        rms_dbfs,
        active_threshold_dbfs,
        active_duration_seconds,
        active_span_seconds,
        active_ratio,
        active_segment_count: segments.len(),
        long_gap_count,
        longest_internal_gap_seconds,
        gap_seconds,
        leading_silence_seconds,
        trailing_silence_seconds,
        leading_fragment_seconds,
        trailing_fragment_seconds,
        segments,
    }
}

fn push_raw_segment(
    segments: &mut Vec<(usize, usize)>,
    start: usize,
    end: usize,
    min_segment_samples: usize,
) {
    if end.saturating_sub(start) >= min_segment_samples {
        segments.push((start, end));
    }
}

fn samples_for_duration(sample_rate: u32, duration_seconds: f64) -> usize {
    ((sample_rate as f64 * duration_seconds).round() as usize).max(1)
}

fn seconds(samples: usize, sample_rate: u32) -> f64 {
    if sample_rate == 0 {
        0.0
    } else {
        samples as f64 / sample_rate as f64
    }
}

fn peak_amplitude(samples: &[f32]) -> f64 {
    samples
        .iter()
        .map(|sample| sample.abs() as f64)
        .fold(0.0, f64::max)
}

fn rms_amplitude(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f64 = samples
        .iter()
        .map(|sample| {
            let sample = *sample as f64;
            sample * sample
        })
        .sum();
    (sum_squares / samples.len() as f64).sqrt()
}

fn dbfs_amplitude(amplitude: f64) -> f64 {
    if amplitude <= 0.0 {
        -120.0
    } else {
        20.0 * amplitude.log10()
    }
}

fn print_voxtral_matrix_report(report: &VoxtralMatrixReport) {
    println!("voxtral_matrix.voice={}", report.voice);
    println!("voxtral_matrix.model={}", report.tts_model);
    println!(
        "voxtral_matrix.stt={} model={}",
        report.stt_backend, report.stt_model
    );
    println!("voxtral_matrix.model_load_ms={:.1}", report.model_load_ms);
    println!(
        "voxtral_matrix.max_frames={:?} flow_steps={:?} stream_begin_frames={} kv_cache={} sync_trace={} text_normalization={} eos_scores={}",
        report.matrix_max_frames,
        report.matrix_flow_steps,
        report.stream_begin_frames,
        report.kv_cache,
        report.sync_trace,
        report.text_normalization,
        report.eos_scores
    );
    if let Some(output_dir) = &report.output_dir {
        println!("voxtral_matrix.output_dir={}", output_dir.display());
    }
    if let Some(spectrogram_dir) = &report.spectrogram_dir {
        println!(
            "voxtral_matrix.spectrogram_dir={}",
            spectrogram_dir.display()
        );
    }
    for row in &report.rows {
        println!(
            concat!(
                "voxtral_matrix.row text_index={} max_frames={} flow_steps={} ",
                "reference_text={:?} synthesis_text={:?} ",
                "first_audio_ms={} total_ms={:.1} audio_ms={:.1} rtf={} ",
                "wer={} errors={}/{} frames={} ended={} chunks={} ",
                "eos_frame={} semantic_count={} semantic_tail={:?} ",
                "semantic_tail_unique={} semantic_tail_repeat={} ",
                "eos_rank_tail={:?} eos_margin_tail={:?} ",
                "best_eos_rank={} best_eos_margin={} ",
                "segments={} long_gaps={} longest_gap_s={:.3} active_ratio={} ",
                "lead_frag_s={} trail_frag_s={} spectrogram={} transcript={:?}"
            ),
            row.text_index,
            row.max_frames,
            row.flow_steps,
            row.text,
            row.synthesis_text,
            format_optional_f64(row.first_audio_ms),
            row.total_ms,
            row.audio_duration_ms,
            format_optional_f64(row.realtime_factor),
            format_optional_f32(row.word_error_rate),
            row.word_error_count,
            row.reference_word_count,
            row.audio_frames,
            row.ended,
            row.codec_chunks,
            format_optional_usize(row.eos_frame),
            row.semantic_code_count,
            row.semantic_tail_codes,
            row.semantic_tail_unique_count,
            row.semantic_tail_repeat_count,
            row.semantic_eos_rank_tail,
            row.semantic_eos_margin_tail,
            format_optional_usize(row.semantic_best_eos_rank),
            format_optional_f32(row.semantic_best_eos_margin),
            row.audio_diagnostics.active_segment_count,
            row.audio_diagnostics.long_gap_count,
            row.audio_diagnostics.longest_internal_gap_seconds,
            format_optional_f64(row.audio_diagnostics.active_ratio),
            format_optional_f64(row.audio_diagnostics.leading_fragment_seconds),
            format_optional_f64(row.audio_diagnostics.trailing_fragment_seconds),
            format_optional_path(row.spectrogram_pgm.as_ref()),
            row.transcription
        );
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn audio_duration_ms(samples: usize, sample_rate: u32) -> f64 {
    if sample_rate == 0 {
        0.0
    } else {
        samples as f64 / sample_rate as f64 * 1_000.0
    }
}

fn ratio_ms(numerator_ms: f64, denominator_ms: f64) -> Option<f64> {
    (denominator_ms > 0.0).then_some(numerator_ms / denominator_ms)
}

fn per_unit(total_ms: f64, units: usize) -> Option<f64> {
    (units > 0).then_some(total_ms / units as f64)
}

fn tail_semantic_codes(codes: &[u32], max_len: usize) -> Vec<u32> {
    let start = codes.len().saturating_sub(max_len);
    codes[start..].to_vec()
}

fn tail_values<T: Clone>(values: &[T], max_len: usize) -> Vec<T> {
    let start = values.len().saturating_sub(max_len);
    values[start..].to_vec()
}

fn unique_count(codes: &[u32]) -> usize {
    let mut sorted = codes.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.len()
}

fn repeated_tail_count(codes: &[u32]) -> usize {
    let Some(&last) = codes.last() else {
        return 0;
    };
    codes.iter().rev().take_while(|&&code| code == last).count()
}

fn format_optional_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_optional_f32(value: Option<f32>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_optional_path(path: Option<&PathBuf>) -> String {
    path.map(|path| path.display().to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn print_text_report(report: &EvalReport) {
    println!("reference: {}", report.text);
    if report.synthesis_text != report.text {
        println!("synthesis text: {}", report.synthesis_text);
    }
    if report.voxtral_text_normalization {
        println!("voxtral text normalization: enabled");
    }
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
        concat!(
            "audio diagnostics: peak={:.1} dBFS rms={:.1} dBFS threshold={:.1} dBFS ",
            "segments={} active_ratio={} long_gaps={} longest_gap_s={:.3} ",
            "leading_silence_s={:.3} trailing_silence_s={:.3} ",
            "lead_frag_s={} trail_frag_s={}"
        ),
        report.audio_diagnostics.peak_dbfs,
        report.audio_diagnostics.rms_dbfs,
        report.audio_diagnostics.active_threshold_dbfs,
        report.audio_diagnostics.active_segment_count,
        format_optional_f64(report.audio_diagnostics.active_ratio),
        report.audio_diagnostics.long_gap_count,
        report.audio_diagnostics.longest_internal_gap_seconds,
        report.audio_diagnostics.leading_silence_seconds,
        report.audio_diagnostics.trailing_silence_seconds,
        format_optional_f64(report.audio_diagnostics.leading_fragment_seconds),
        format_optional_f64(report.audio_diagnostics.trailing_fragment_seconds)
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
    if let Some(path) = &report.spectrogram_pgm {
        println!("spectrogram: {}", path.display());
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

    #[test]
    fn parses_synthesis_text_for_pronunciation_eval() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--text",
            "Voxtral should sound clear.",
            "--synthesis-text",
            "Vox trahl should sound clear.",
            "--voxtral-normalize-text",
        ])
        .unwrap();

        assert_eq!(args.text, "Voxtral should sound clear.");
        assert_eq!(
            args.synthesis_text.as_deref(),
            Some("Vox trahl should sound clear.")
        );
        assert!(args.voxtral_normalize_text);
    }

    #[test]
    fn parses_voxtral_matrix_args() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-matrix",
            "--matrix-text",
            "A fast reply should arrive naturally.",
            "--matrix-text",
            "Voxtral pronunciation should stay clear.",
            "--matrix-synthesis-text",
            "A fast reply should arrive naturally.",
            "--matrix-synthesis-text",
            "Vox trahl pronunciation should stay clear.",
            "--matrix-max-frames",
            "32,40",
            "--matrix-flow-steps",
            "5,6,7",
            "--voxtral-kv-cache",
            "--voxtral-eos-scores",
            "--voxtral-stream-begin-frames",
            "2",
            "--output-dir",
            "/tmp/voxtral-matrix",
            "--spectrogram-dir",
            "/tmp/voxtral-spectrograms",
        ])
        .unwrap();

        assert!(args.voxtral_matrix);
        assert_eq!(
            args.matrix_texts,
            vec![
                "A fast reply should arrive naturally.",
                "Voxtral pronunciation should stay clear."
            ]
        );
        assert_eq!(
            args.matrix_synthesis_texts,
            vec![
                "A fast reply should arrive naturally.",
                "Vox trahl pronunciation should stay clear."
            ]
        );
        assert_eq!(args.matrix_max_frames, vec![32, 40]);
        assert_eq!(args.matrix_flow_steps, vec![5, 6, 7]);
        assert!(args.voxtral_kv_cache);
        assert!(args.voxtral_eos_scores);
        assert_eq!(args.voxtral_stream_begin_frames, 2);
        assert_eq!(args.output_dir, Some(PathBuf::from("/tmp/voxtral-matrix")));
        assert_eq!(
            args.spectrogram_dir,
            Some(PathBuf::from("/tmp/voxtral-spectrograms"))
        );
    }

    #[test]
    fn defaults_tts_model_from_backend() {
        let kokoro_args = Args::try_parse_from(["voice-eval"]).unwrap();
        assert_eq!(tts_model_path(&kokoro_args), DEFAULT_TTS_MODEL);

        let voxtral_args =
            Args::try_parse_from(["voice-eval", "--tts-backend", "voxtral"]).unwrap();
        assert_eq!(tts_model_path(&voxtral_args), voice_voxtral::DEFAULT_REPO);
    }

    #[test]
    fn defaults_tts_voice_from_backend() {
        let kokoro_args = Args::try_parse_from(["voice-eval"]).unwrap();
        assert_eq!(tts_voice(&kokoro_args), DEFAULT_KOKORO_VOICE);

        let voxtral_args =
            Args::try_parse_from(["voice-eval", "--tts-backend", "voxtral"]).unwrap();
        assert_eq!(tts_voice(&voxtral_args), DEFAULT_VOXTRAL_VOICE);

        let explicit_voxtral_args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voice",
            "neutral_female",
        ])
        .unwrap();
        assert_eq!(tts_voice(&explicit_voxtral_args), "neutral_female");
    }

    #[test]
    fn matrix_cases_falls_back_to_primary_text() {
        let args =
            Args::try_parse_from(["voice-eval", "--voxtral-matrix", "--text", "hello"]).unwrap();

        assert_eq!(
            matrix_cases(&args).unwrap(),
            vec![MatrixCase {
                reference_text: "hello".to_string(),
                synthesis_text: "hello".to_string(),
            }]
        );
    }

    #[test]
    fn matrix_cases_pair_reference_and_synthesis_texts() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-matrix",
            "--matrix-text",
            "Voxtral sounds clear.",
            "--matrix-text",
            "Read ticket A17.",
            "--matrix-synthesis-text",
            "Vox trahl sounds clear.",
            "--matrix-synthesis-text",
            "Read ticket A seventeen.",
        ])
        .unwrap();

        assert_eq!(
            matrix_cases(&args).unwrap(),
            vec![
                MatrixCase {
                    reference_text: "Voxtral sounds clear.".to_string(),
                    synthesis_text: "Vox trahl sounds clear.".to_string(),
                },
                MatrixCase {
                    reference_text: "Read ticket A17.".to_string(),
                    synthesis_text: "Read ticket A seventeen.".to_string(),
                }
            ]
        );
    }

    #[test]
    fn matrix_cases_apply_opt_in_voxtral_text_normalization() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-matrix",
            "--voxtral-normalize-text",
            "--matrix-text",
            "Read ticket A17, version 2.4.1, at 9:30 PM.",
        ])
        .unwrap();

        assert_eq!(
            matrix_cases(&args).unwrap(),
            vec![MatrixCase {
                reference_text: "Read ticket A17, version 2.4.1, at 9:30 PM.".to_string(),
                synthesis_text:
                    "Read ticket A seventeen, version two point four point one, at nine thirty PM."
                        .to_string(),
            }]
        );
    }

    #[test]
    fn matrix_cases_require_matching_synthesis_count() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--voxtral-matrix",
            "--matrix-text",
            "Voxtral sounds clear.",
            "--matrix-text",
            "Read ticket A17.",
            "--matrix-synthesis-text",
            "Vox trahl sounds clear.",
        ])
        .unwrap();

        let err = matrix_cases(&args).unwrap_err().to_string();
        assert!(
            err.contains("--matrix-synthesis-text count (1) must match --matrix-text count (2)")
        );
    }

    #[test]
    fn semantic_tail_helpers_summarize_generation_codes() {
        let codes = vec![10, 11, 12, 12, 12];

        assert_eq!(tail_semantic_codes(&codes, 3), vec![12, 12, 12]);
        assert_eq!(tail_values(&codes, 2), vec![12, 12]);
        assert_eq!(unique_count(&tail_semantic_codes(&codes, 4)), 2);
        assert_eq!(repeated_tail_count(&codes), 3);
        assert_eq!(repeated_tail_count(&[]), 0);
    }

    #[test]
    fn audio_diagnostics_keep_continuous_audio_in_one_segment() {
        let sample_rate = 1_000;
        let samples = tone_samples(sample_rate, 1.0, 0.5);

        let diagnostics = analyze_audio(&samples, sample_rate);

        assert_eq!(diagnostics.active_segment_count, 1);
        assert_eq!(diagnostics.long_gap_count, 0);
        assert!(diagnostics.longest_internal_gap_seconds < 0.01);
        assert!(diagnostics.leading_fragment_seconds.is_none());
        assert!(diagnostics.trailing_fragment_seconds.is_none());
    }

    #[test]
    fn audio_diagnostics_flag_islands_and_long_internal_gaps() {
        let sample_rate = 1_000;
        let mut samples = Vec::new();
        samples.extend(tone_samples(sample_rate, 0.08, 0.5));
        samples.extend(silence_samples(sample_rate, 0.40));
        samples.extend(tone_samples(sample_rate, 0.50, 0.5));
        samples.extend(silence_samples(sample_rate, 0.30));
        samples.extend(tone_samples(sample_rate, 0.10, 0.5));

        let diagnostics = analyze_audio(&samples, sample_rate);

        assert_eq!(diagnostics.active_segment_count, 3);
        assert_eq!(diagnostics.long_gap_count, 2);
        assert!(diagnostics.longest_internal_gap_seconds > 0.25);
        assert!(diagnostics.leading_fragment_seconds.is_some());
        assert!(diagnostics.trailing_fragment_seconds.is_some());
    }

    #[test]
    fn spectrogram_pgm_has_a_valid_grayscale_header() {
        let image = spectrogram_pgm(&tone_samples(1_000, 0.10, 0.5), 1_000);

        assert!(image.starts_with(b"P5\n# sample_rate 1000\n"));
        assert!(image.len() > b"P5\n# sample_rate 1000\n1 96\n255\n".len());
    }

    fn tone_samples(sample_rate: u32, duration_seconds: f64, amplitude: f32) -> Vec<f32> {
        let sample_count = (sample_rate as f64 * duration_seconds).round() as usize;
        (0..sample_count)
            .map(|index| {
                let phase = 2.0 * PI * 120.0 * index as f64 / sample_rate as f64;
                amplitude * phase.sin() as f32
            })
            .collect()
    }

    fn silence_samples(sample_rate: u32, duration_seconds: f64) -> Vec<f32> {
        vec![0.0; (sample_rate as f64 * duration_seconds).round() as usize]
    }
}
