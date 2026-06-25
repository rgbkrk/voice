use std::collections::BTreeMap;
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
const VOXTRAL_QUALITY_SUITE_PROMPTS: &[&str] = &[
    "hello world",
    "A fast reply should arrive naturally.",
    "Voxtral should pronounce its own made-up name clearly.",
    "Please pause, then continue; do not add extra words.",
    "Read ticket A17, version 2.4.1, at 9:30 PM.",
    "If I ask a quick question, can you answer in one sentence?",
    "The voice should stay steady across a longer reply, even when the sentence reaches the realtime frame cap.",
];

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

    /// Treat compact time transcripts such as 1205 a.m. as equivalent to 12:05 AM for WER.
    #[arg(long = "time-token-equivalence")]
    time_token_equivalence: bool,

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

    /// Run the canonical varied Voxtral quality prompt suite.
    #[arg(long = "voxtral-quality-suite")]
    voxtral_quality_suite: bool,

    /// Additional prompt for --voxtral-matrix. Repeat for multiple prompts.
    #[arg(long = "matrix-text", value_name = "TEXT")]
    matrix_texts: Vec<String>,

    /// Synthesis text paired by index with --matrix-text.
    #[arg(long = "matrix-synthesis-text", value_name = "TEXT")]
    matrix_synthesis_texts: Vec<String>,

    /// Voxtral voice for --voxtral-matrix. Repeat or comma-separate for multiple voices.
    #[arg(long = "matrix-voice", value_name = "VOICE", value_delimiter = ',')]
    matrix_voices: Vec<String>,

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

    /// Speech speed values for --voxtral-matrix. Repeat or comma-separate for multiple speeds.
    #[arg(long = "matrix-speed", value_name = "SPEED", value_delimiter = ',')]
    matrix_speeds: Vec<f32>,

    /// Enable Voxtral language KV cache for Voxtral synthesis.
    #[arg(long = "voxtral-kv-cache")]
    voxtral_kv_cache: bool,

    /// Normalize compact Voxtral numeric synthesis text forms such as versions and times.
    #[arg(long = "voxtral-normalize-text")]
    voxtral_normalize_text: bool,

    /// Apply known Voxtral pronunciation aliases before synthesis.
    #[arg(long = "voxtral-pronunciation-aliases")]
    voxtral_pronunciation_aliases: bool,

    /// Choose Voxtral max frames from the post-normalization synthesis text.
    #[arg(long = "voxtral-auto-max-frames")]
    voxtral_auto_max_frames: bool,

    /// Initial streaming codec chunk size for Voxtral matrix timing.
    #[arg(long = "voxtral-stream-begin-frames", default_value_t = 2)]
    voxtral_stream_begin_frames: usize,

    /// Synchronize Metal around Voxtral trace sections for profiling.
    #[arg(long = "voxtral-sync-trace")]
    voxtral_sync_trace: bool,

    /// Collect EOS rank/margin diagnostics for Voxtral semantic logits.
    #[arg(long = "voxtral-eos-scores")]
    voxtral_eos_scores: bool,

    /// Extra frames to allow after --max-frames when EOS is close at the cap.
    #[arg(long = "voxtral-eos-guard-frames", default_value_t = 0)]
    voxtral_eos_guard_frames: usize,

    /// Maximum EOS rank that can activate --voxtral-eos-guard-frames.
    #[arg(long = "voxtral-eos-guard-rank", default_value_t = 2)]
    voxtral_eos_guard_rank: usize,

    /// Maximum selected-minus-EOS logit margin that can activate the EOS guard.
    #[arg(long = "voxtral-eos-guard-margin", default_value_t = 0.5)]
    voxtral_eos_guard_margin: f32,

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
    first_active_seconds: Option<f64>,
    active_segment_count: usize,
    long_gap_count: usize,
    longest_internal_gap_seconds: f64,
    gap_seconds: Vec<f64>,
    leading_silence_seconds: f64,
    trailing_silence_seconds: f64,
    leading_fragment_seconds: Option<f64>,
    trailing_fragment_seconds: Option<f64>,
    energy_peak_count: usize,
    energy_peak_rate_hz: Option<f64>,
    energy_peak_interval_mean_seconds: Option<f64>,
    energy_peak_interval_cv: Option<f64>,
    spectral_flux_peak_count: usize,
    spectral_flux_peak_rate_hz: Option<f64>,
    spectral_flux_interval_mean_seconds: Option<f64>,
    spectral_flux_interval_cv: Option<f64>,
    spectral_flux_mean: Option<f64>,
    spectral_flux_cv: Option<f64>,
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
    voxtral_pronunciation_aliases: bool,
    voxtral_auto_max_frames: bool,
    sample_rate: u32,
    sample_count: usize,
    duration_seconds: f32,
    phoneme_chunks: Vec<String>,
    transcription: String,
    time_token_equivalence: bool,
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
    matrix_voices: Vec<String>,
    tts_model: String,
    stt_backend: String,
    stt_model: String,
    model_load_ms: f64,
    matrix_max_frames: Vec<usize>,
    matrix_flow_steps: Vec<usize>,
    matrix_speeds: Vec<f32>,
    stream_begin_frames: usize,
    kv_cache: bool,
    sync_trace: bool,
    text_normalization: bool,
    pronunciation_aliases: bool,
    auto_max_frames: bool,
    eos_scores: bool,
    eos_guard_frames: usize,
    eos_guard_max_rank: usize,
    eos_guard_max_margin: f32,
    time_token_equivalence: bool,
    seed: u64,
    output_dir: Option<PathBuf>,
    spectrogram_dir: Option<PathBuf>,
    quality_summary: VoxtralQualitySummary,
    rows: Vec<VoxtralMatrixRow>,
}

#[derive(Debug, Serialize)]
struct VoxtralQualitySummary {
    total_rows: usize,
    clean_rows: usize,
    suspect_rows: usize,
    ended_rows: usize,
    zero_wer_rows: usize,
    transcript_correct_artifact_rows: usize,
    by_setting: Vec<VoxtralSettingQualitySummary>,
}

#[derive(Debug, Serialize)]
struct VoxtralSettingQualitySummary {
    voice: String,
    speed: f32,
    max_frames: usize,
    flow_steps: usize,
    rows: usize,
    clean_rows: usize,
    suspect_rows: usize,
    ended_rows: usize,
    zero_wer_rows: usize,
    transcript_correct_artifact_rows: usize,
    average_word_error_rate: Option<f32>,
    average_realtime_factor: Option<f64>,
    average_model_realtime_factor: Option<f64>,
    average_first_active_audio_ms: Option<f64>,
    quality_flags: Vec<QualityFlag>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QualityFlag {
    WordError,
    DidNotEnd,
    NoActiveAudio,
    ExtraActiveSegments,
    LongInternalGap,
    LeadingFragment,
    TrailingFragment,
    IrregularSpectralFlux,
}

impl Serialize for QualityFlag {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl QualityFlag {
    fn as_str(self) -> &'static str {
        match self {
            QualityFlag::WordError => "word_error",
            QualityFlag::DidNotEnd => "did_not_end",
            QualityFlag::NoActiveAudio => "no_active_audio",
            QualityFlag::ExtraActiveSegments => "extra_active_segments",
            QualityFlag::LongInternalGap => "long_internal_gap",
            QualityFlag::LeadingFragment => "leading_fragment",
            QualityFlag::TrailingFragment => "trailing_fragment",
            QualityFlag::IrregularSpectralFlux => "irregular_spectral_flux",
        }
    }
}

#[derive(Debug, Serialize)]
struct VoxtralMatrixRow {
    text_index: usize,
    voice: String,
    text: String,
    synthesis_text: String,
    speed: f32,
    max_frames: usize,
    flow_steps: usize,
    first_code_frame_ms: Option<f64>,
    first_audio_ms: Option<f64>,
    first_active_audio_ms: Option<f64>,
    total_ms: f64,
    model_audio_duration_ms: f64,
    model_realtime_factor: Option<f64>,
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
    quality_suspect: bool,
    quality_flags: Vec<QualityFlag>,
    stt_token_count: usize,
    audio_diagnostics: AudioDiagnostics,
    output_wav: Option<PathBuf>,
    spectrogram_pgm: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.voxtral_matrix || args.voxtral_quality_suite {
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
    let wer =
        word_error_rate_with_options(&reference_text, &transcription.text, wer_options(&args));
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
        voxtral_pronunciation_aliases: args.tts_backend == "voxtral"
            && args.voxtral_pronunciation_aliases,
        voxtral_auto_max_frames: args.tts_backend == "voxtral" && args.voxtral_auto_max_frames,
        sample_rate,
        sample_count: samples.len(),
        duration_seconds,
        phoneme_chunks,
        transcription: transcription.text,
        time_token_equivalence: args.time_token_equivalence,
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
    if args.voxtral_quality_suite && !args.matrix_texts.is_empty() {
        return Err("--voxtral-quality-suite cannot be combined with --matrix-text".into());
    }
    if args.voxtral_quality_suite && !args.matrix_synthesis_texts.is_empty() {
        return Err(
            "--voxtral-quality-suite cannot be combined with --matrix-synthesis-text".into(),
        );
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
    let matrix_voices = matrix_voices(args)?;
    let matrix_speeds = matrix_speeds(args)?;
    if args.voxtral_stream_begin_frames == 0 {
        return Err("--voxtral-stream-begin-frames must be greater than zero".into());
    }

    let cases = matrix_cases(args)?;
    let tts_model_path = tts_model_path(args);
    let (mut runtime, load_trace) =
        voice_voxtral::VoxtralTtsRuntime::load_default_with_trace(&tts_model_path)?;
    for voice in &matrix_voices {
        runtime.preload_voice(voice)?;
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

    let mut rows = Vec::new();
    for voice in &matrix_voices {
        for (text_index, case) in cases.iter().enumerate() {
            for &speed in &matrix_speeds {
                for &configured_max_frames in &args.matrix_max_frames {
                    for &flow_steps in &args.matrix_flow_steps {
                        let max_frames = effective_voxtral_max_frames(
                            args,
                            &case.synthesis_text,
                            configured_max_frames,
                        );
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
                            eos_guard_frames: args.voxtral_eos_guard_frames,
                            eos_guard_max_rank: args.voxtral_eos_guard_rank,
                            eos_guard_max_margin: args.voxtral_eos_guard_margin,
                            ..Default::default()
                        };

                        let run_start = Instant::now();
                        let (audio, trace) = runtime.generate_audio_streaming_with_trace(
                            &case.synthesis_text,
                            voice,
                            options,
                            streaming,
                            |_| Ok(()),
                        )?;
                        let total = run_start.elapsed();
                        let samples = voice_audio::adjust_speed(&audio.samples, speed)?;
                        let output_wav = maybe_write_matrix_wav(
                            args.output_dir.as_deref(),
                            MatrixArtifactName {
                                voice,
                                speed,
                                text_index,
                                max_frames,
                                flow_steps,
                            },
                            &samples,
                            audio.sample_rate,
                        )?;
                        let spectrogram_pgm = maybe_write_spectrogram_pgm(
                            args.spectrogram_dir.as_deref(),
                            &matrix_artifact_file_name(
                                MatrixArtifactName {
                                    voice,
                                    speed,
                                    text_index,
                                    max_frames,
                                    flow_steps,
                                },
                                "pgm",
                            ),
                            &samples,
                            audio.sample_rate,
                        )?;

                        let transcription =
                            stt_model.transcribe_audio(&samples, audio.sample_rate)?;
                        let wer = word_error_rate_with_options(
                            &case.reference_text,
                            &transcription.text,
                            wer_options(args),
                        );
                        let audio_diagnostics = analyze_audio(&samples, audio.sample_rate);
                        let total_ms = duration_ms(total);
                        let model_audio_duration_ms =
                            audio_duration_ms(audio.samples.len(), audio.sample_rate);
                        let audio_duration_ms = audio_duration_ms(samples.len(), audio.sample_rate);
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
                        let first_code_frame_ms = trace.first_frame.map(duration_ms);
                        let first_audio_ms = trace.first_audio_chunk.map(duration_ms);
                        let first_active_audio_ms =
                            first_active_audio_ms(first_audio_ms, &audio_diagnostics);
                        let quality_flags =
                            voxtral_quality_flags(wer.distance, audio.ended, &audio_diagnostics);

                        rows.push(VoxtralMatrixRow {
                            text_index,
                            voice: voice.clone(),
                            text: case.reference_text.clone(),
                            synthesis_text: case.synthesis_text.clone(),
                            speed,
                            max_frames,
                            flow_steps,
                            first_code_frame_ms,
                            first_audio_ms,
                            first_active_audio_ms,
                            total_ms,
                            model_audio_duration_ms,
                            model_realtime_factor: ratio_ms(total_ms, model_audio_duration_ms),
                            audio_duration_ms,
                            realtime_factor: ratio_ms(total_ms, audio_duration_ms),
                            samples: samples.len(),
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
                            quality_suspect: !quality_flags.is_empty(),
                            quality_flags,
                            stt_token_count: transcription.tokens.len(),
                            audio_diagnostics,
                            output_wav,
                            spectrogram_pgm,
                        });
                    }
                }
            }
        }
    }

    let report = VoxtralMatrixReport {
        voice: matrix_voices.join(","),
        matrix_voices,
        tts_model: tts_model_path,
        stt_backend: stt_backend.as_str().to_string(),
        stt_model: stt_model_path,
        model_load_ms: duration_ms(load_trace.total),
        matrix_max_frames: args.matrix_max_frames.clone(),
        matrix_flow_steps: args.matrix_flow_steps.clone(),
        matrix_speeds,
        stream_begin_frames: args.voxtral_stream_begin_frames,
        kv_cache: args.voxtral_kv_cache,
        sync_trace: args.voxtral_sync_trace,
        text_normalization: args.voxtral_normalize_text,
        pronunciation_aliases: args.voxtral_pronunciation_aliases,
        auto_max_frames: args.voxtral_auto_max_frames,
        eos_scores: args.voxtral_eos_scores,
        eos_guard_frames: args.voxtral_eos_guard_frames,
        eos_guard_max_rank: args.voxtral_eos_guard_rank,
        eos_guard_max_margin: args.voxtral_eos_guard_margin,
        time_token_equivalence: args.time_token_equivalence,
        seed: args.seed,
        output_dir: args.output_dir.clone(),
        spectrogram_dir: args.spectrogram_dir.clone(),
        quality_summary: summarize_voxtral_quality(&rows),
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

fn matrix_voices(args: &Args) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let voices = if args.matrix_voices.is_empty() {
        vec![tts_voice(args)]
    } else {
        args.matrix_voices.clone()
    };
    for voice in &voices {
        if voice_voxtral::get_preset_voice(voice).is_none() {
            return Err(format!("unknown Voxtral voice for --matrix-voice: {voice}").into());
        }
    }
    Ok(voices)
}

fn matrix_speeds(args: &Args) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let speeds = if args.matrix_speeds.is_empty() {
        vec![args.speed]
    } else {
        args.matrix_speeds.clone()
    };
    for speed in &speeds {
        validate_speed(*speed)?;
    }
    Ok(speeds)
}

fn validate_speed(speed: f32) -> Result<(), Box<dyn std::error::Error>> {
    if speed.is_finite() && speed > 0.0 {
        Ok(())
    } else {
        Err(format!("speed values must be finite and greater than zero, got {speed}").into())
    }
}

fn matrix_cases(args: &Args) -> Result<Vec<MatrixCase>, Box<dyn std::error::Error>> {
    let reference_texts = if args.voxtral_quality_suite {
        VOXTRAL_QUALITY_SUITE_PROMPTS
            .iter()
            .map(|prompt| (*prompt).to_string())
            .collect()
    } else if args.matrix_texts.is_empty() {
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
    if args.tts_backend == "voxtral"
        && (args.voxtral_normalize_text || args.voxtral_pronunciation_aliases)
    {
        voice_voxtral::normalize_tts_text_with_options(
            &text,
            voice_voxtral::VoxtralTextNormalizationOptions {
                numeric: args.voxtral_normalize_text,
                pronunciation_aliases: args.voxtral_pronunciation_aliases,
            },
        )
    } else {
        text
    }
}

fn effective_voxtral_max_frames(args: &Args, synthesis_text: &str, configured: usize) -> usize {
    if args.voxtral_auto_max_frames {
        configured.max(voice_voxtral::suggest_max_frames_for_text(synthesis_text))
    } else {
        configured
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
            max_frames: effective_voxtral_max_frames(args, text, args.max_frames),
            seed: args.seed,
            flow_steps: args.flow_steps,
            use_kv_cache: args.voxtral_kv_cache,
            synchronize_trace: args.voxtral_sync_trace,
            trace_semantic_scores: args.voxtral_eos_scores,
            eos_guard_frames: args.voxtral_eos_guard_frames,
            eos_guard_max_rank: args.voxtral_eos_guard_rank,
            eos_guard_max_margin: args.voxtral_eos_guard_margin,
            ..Default::default()
        },
    )?;
    let samples = voice_audio::adjust_speed(&audio.samples, args.speed)?;
    Ok((samples, audio.sample_rate))
}

#[derive(Debug, Clone, Copy)]
struct MatrixArtifactName<'a> {
    voice: &'a str,
    speed: f32,
    text_index: usize,
    max_frames: usize,
    flow_steps: usize,
}

fn maybe_write_matrix_wav(
    output_dir: Option<&Path>,
    name: MatrixArtifactName<'_>,
    samples: &[f32],
    sample_rate: u32,
) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let Some(output_dir) = output_dir else {
        return Ok(None);
    };
    std::fs::create_dir_all(output_dir)?;
    let path = output_dir.join(matrix_artifact_file_name(name, "wav"));
    voice_tts::save_wav(samples, &path, sample_rate)?;
    Ok(Some(path))
}

fn matrix_artifact_file_name(name: MatrixArtifactName<'_>, extension: &str) -> String {
    format!(
        "voxtral-{}-speed{}-text{}-max{}-flow{}.{}",
        file_stem(name.voice),
        file_stem(&format_speed(name.speed)),
        name.text_index + 1,
        name.max_frames,
        name.flow_steps,
        extension
    )
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
            first_active_seconds: None,
            active_segment_count: 0,
            long_gap_count: 0,
            longest_internal_gap_seconds: 0.0,
            gap_seconds: Vec::new(),
            leading_silence_seconds: duration_seconds,
            trailing_silence_seconds: 0.0,
            leading_fragment_seconds: None,
            trailing_fragment_seconds: None,
            energy_peak_count: 0,
            energy_peak_rate_hz: None,
            energy_peak_interval_mean_seconds: None,
            energy_peak_interval_cv: None,
            spectral_flux_peak_count: 0,
            spectral_flux_peak_rate_hz: None,
            spectral_flux_interval_mean_seconds: None,
            spectral_flux_interval_cv: None,
            spectral_flux_mean: None,
            spectral_flux_cv: None,
            segments: Vec::new(),
        };
    }

    let frame_len = samples_for_duration(sample_rate, FRAME_MS as f64 / 1_000.0);
    let hop_len = samples_for_duration(sample_rate, HOP_MS as f64 / 1_000.0);
    let merge_gap_samples = samples_for_duration(sample_rate, MERGE_GAP_SECONDS);
    let min_segment_samples = samples_for_duration(sample_rate, MIN_SEGMENT_SECONDS);

    let mut raw_segments = Vec::new();
    let mut energy_frames = Vec::new();
    let mut current: Option<(usize, usize)> = None;
    let mut start = 0;
    while start < samples.len() {
        let end = (start + frame_len).min(samples.len());
        let frame_rms = rms_amplitude(&samples[start..end]);
        let frame_rms_dbfs = dbfs_amplitude(frame_rms);
        energy_frames.push((
            seconds((start + end) / 2, sample_rate),
            frame_rms,
            frame_rms_dbfs,
        ));
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
    let first_active_seconds = segments.first().map(|segment| segment.start_seconds);
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
    let rhythm = rhythm_diagnostics(&energy_frames, active_threshold_dbfs, active_span_seconds);
    let spectral_flux = spectral_flux_diagnostics(
        samples,
        sample_rate,
        active_threshold_dbfs,
        active_span_seconds,
    );

    AudioDiagnostics {
        duration_seconds,
        peak_dbfs,
        rms_dbfs,
        active_threshold_dbfs,
        active_duration_seconds,
        active_span_seconds,
        active_ratio,
        first_active_seconds,
        active_segment_count: segments.len(),
        long_gap_count,
        longest_internal_gap_seconds,
        gap_seconds,
        leading_silence_seconds,
        trailing_silence_seconds,
        leading_fragment_seconds,
        trailing_fragment_seconds,
        energy_peak_count: rhythm.energy_peak_count,
        energy_peak_rate_hz: rhythm.energy_peak_rate_hz,
        energy_peak_interval_mean_seconds: rhythm.energy_peak_interval_mean_seconds,
        energy_peak_interval_cv: rhythm.energy_peak_interval_cv,
        spectral_flux_peak_count: spectral_flux.peak_count,
        spectral_flux_peak_rate_hz: spectral_flux.peak_rate_hz,
        spectral_flux_interval_mean_seconds: spectral_flux.interval_mean_seconds,
        spectral_flux_interval_cv: spectral_flux.interval_cv,
        spectral_flux_mean: spectral_flux.mean,
        spectral_flux_cv: spectral_flux.cv,
        segments,
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct RhythmDiagnostics {
    energy_peak_count: usize,
    energy_peak_rate_hz: Option<f64>,
    energy_peak_interval_mean_seconds: Option<f64>,
    energy_peak_interval_cv: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct SpectralFluxDiagnostics {
    peak_count: usize,
    peak_rate_hz: Option<f64>,
    interval_mean_seconds: Option<f64>,
    interval_cv: Option<f64>,
    mean: Option<f64>,
    cv: Option<f64>,
}

fn rhythm_diagnostics(
    frames: &[(f64, f64, f64)],
    active_threshold_dbfs: f64,
    active_span_seconds: f64,
) -> RhythmDiagnostics {
    const MIN_PEAK_DISTANCE_SECONDS: f64 = 0.08;

    if frames.len() < 3 {
        return RhythmDiagnostics::default();
    }

    let mut peaks = Vec::<(f64, f64)>::new();
    for window in frames.windows(3) {
        let previous = window[0];
        let current = window[1];
        let next = window[2];
        let is_peak =
            current.2 >= active_threshold_dbfs && current.1 >= previous.1 && current.1 > next.1;
        if !is_peak {
            continue;
        }
        if let Some(last) = peaks.last_mut() {
            if current.0 - last.0 < MIN_PEAK_DISTANCE_SECONDS {
                if current.1 > last.1 {
                    *last = (current.0, current.1);
                }
                continue;
            }
        }
        peaks.push((current.0, current.1));
    }

    let intervals = peaks
        .windows(2)
        .map(|pair| pair[1].0 - pair[0].0)
        .collect::<Vec<_>>();
    let interval_mean = mean(&intervals);
    let interval_cv = coefficient_of_variation(&intervals);

    RhythmDiagnostics {
        energy_peak_count: peaks.len(),
        energy_peak_rate_hz: (active_span_seconds > 0.0)
            .then_some(peaks.len() as f64 / active_span_seconds),
        energy_peak_interval_mean_seconds: interval_mean,
        energy_peak_interval_cv: interval_cv,
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then_some(values.iter().sum::<f64>() / values.len() as f64)
}

fn coefficient_of_variation(values: &[f64]) -> Option<f64> {
    let mean = mean(values)?;
    (mean > 0.0).then(|| {
        let variance = values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / values.len() as f64;
        variance.sqrt() / mean
    })
}

fn spectral_flux_diagnostics(
    samples: &[f32],
    sample_rate: u32,
    active_threshold_dbfs: f64,
    active_span_seconds: f64,
) -> SpectralFluxDiagnostics {
    const WINDOW_SIZE: usize = 512;
    const HOP_SIZE: usize = 128;
    const BINS: usize = 96;
    const MIN_PEAK_DISTANCE_SECONDS: f64 = 0.08;

    if samples.is_empty() || sample_rate == 0 {
        return SpectralFluxDiagnostics::default();
    }

    let frame_count = ((samples.len().saturating_sub(1)) / HOP_SIZE) + 1;
    if frame_count < 3 {
        return SpectralFluxDiagnostics::default();
    }

    let mut previous: Option<Vec<f64>> = None;
    let mut frames = Vec::<(f64, f64, f64)>::with_capacity(frame_count);
    for frame in 0..frame_count {
        let start = frame * HOP_SIZE;
        let end = (start + WINDOW_SIZE).min(samples.len());
        let frame_rms_dbfs = dbfs_amplitude(rms_amplitude(&samples[start..end]));
        let magnitudes = log_spectrum(samples, start, WINDOW_SIZE, BINS);
        let Some(previous_magnitudes) = previous.as_ref() else {
            previous = Some(magnitudes);
            continue;
        };
        let flux = magnitudes
            .iter()
            .zip(previous_magnitudes.iter())
            .map(|(current, previous)| (current - previous).max(0.0))
            .sum::<f64>();
        previous = Some(magnitudes);
        frames.push((
            seconds(start + (WINDOW_SIZE / 2), sample_rate),
            flux,
            frame_rms_dbfs,
        ));
    }

    let active_flux = frames
        .iter()
        .filter(|(_, _, dbfs)| *dbfs >= active_threshold_dbfs)
        .map(|(_, flux, _)| *flux)
        .collect::<Vec<_>>();
    let Some(flux_mean) = mean(&active_flux) else {
        return SpectralFluxDiagnostics::default();
    };
    let flux_cv = coefficient_of_variation(&active_flux);
    let flux_stddev = flux_cv.map(|cv| cv * flux_mean).unwrap_or(0.0);
    let peak_threshold = flux_mean + (flux_stddev * 0.5);

    let mut peaks = Vec::<(f64, f64)>::new();
    for window in frames.windows(3) {
        let previous = window[0];
        let current = window[1];
        let next = window[2];
        let is_peak = current.2 >= active_threshold_dbfs
            && current.1 >= peak_threshold
            && current.1 >= previous.1
            && current.1 > next.1;
        if !is_peak {
            continue;
        }
        if let Some(last) = peaks.last_mut() {
            if current.0 - last.0 < MIN_PEAK_DISTANCE_SECONDS {
                if current.1 > last.1 {
                    *last = (current.0, current.1);
                }
                continue;
            }
        }
        peaks.push((current.0, current.1));
    }

    let intervals = peaks
        .windows(2)
        .map(|pair| pair[1].0 - pair[0].0)
        .collect::<Vec<_>>();

    SpectralFluxDiagnostics {
        peak_count: peaks.len(),
        peak_rate_hz: (active_span_seconds > 0.0)
            .then_some(peaks.len() as f64 / active_span_seconds),
        interval_mean_seconds: mean(&intervals),
        interval_cv: coefficient_of_variation(&intervals),
        mean: Some(flux_mean),
        cv: flux_cv,
    }
}

fn log_spectrum(samples: &[f32], start: usize, window_size: usize, bins: usize) -> Vec<f64> {
    let mut magnitudes = Vec::with_capacity(bins);
    for bin in 0..bins {
        let mut real = 0.0_f64;
        let mut imag = 0.0_f64;
        for n in 0..window_size {
            let sample = samples.get(start + n).copied().unwrap_or(0.0) as f64;
            let window = 0.5 - 0.5 * ((2.0 * PI * n as f64) / (window_size - 1) as f64).cos();
            let angle = 2.0 * PI * bin as f64 * n as f64 / window_size as f64;
            real += sample * window * angle.cos();
            imag -= sample * window * angle.sin();
        }
        let magnitude = (real.mul_add(real, imag * imag)).sqrt();
        magnitudes.push((magnitude + 1.0e-9).ln());
    }
    magnitudes
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
        "voxtral_matrix.quality_suspect_rows={}/{}",
        report.quality_summary.suspect_rows, report.quality_summary.total_rows
    );
    println!(
        "voxtral_matrix.clean_rows={} ended_rows={} zero_wer_rows={} transcript_correct_artifact_rows={}",
        report.quality_summary.clean_rows,
        report.quality_summary.ended_rows,
        report.quality_summary.zero_wer_rows,
        report.quality_summary.transcript_correct_artifact_rows
    );
    println!(
        "voxtral_matrix.voices={:?} speeds={:?} max_frames={:?} flow_steps={:?} stream_begin_frames={} kv_cache={} sync_trace={} text_normalization={} pronunciation_aliases={} auto_max_frames={} eos_scores={} eos_guard_frames={} eos_guard_max_rank={} eos_guard_max_margin={:.3} time_token_equivalence={}",
        report.matrix_voices,
        report.matrix_speeds,
        report.matrix_max_frames,
        report.matrix_flow_steps,
        report.stream_begin_frames,
        report.kv_cache,
        report.sync_trace,
        report.text_normalization,
        report.pronunciation_aliases,
        report.auto_max_frames,
        report.eos_scores,
        report.eos_guard_frames,
        report.eos_guard_max_rank,
        report.eos_guard_max_margin,
        report.time_token_equivalence
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
    for setting in &report.quality_summary.by_setting {
        println!(
            "voxtral_matrix.setting voice={} speed={} max_frames={} flow_steps={} clean_rows={}/{} suspect_rows={} ended_rows={} zero_wer_rows={} transcript_correct_artifact_rows={} avg_wer={} avg_rtf={} avg_model_rtf={} avg_first_active_audio_ms={} quality_flags={}",
            setting.voice,
            format_speed(setting.speed),
            setting.max_frames,
            setting.flow_steps,
            setting.clean_rows,
            setting.rows,
            setting.suspect_rows,
            setting.ended_rows,
            setting.zero_wer_rows,
            setting.transcript_correct_artifact_rows,
            format_optional_f32(setting.average_word_error_rate),
            format_optional_f64(setting.average_realtime_factor),
            format_optional_f64(setting.average_model_realtime_factor),
            format_optional_f64(setting.average_first_active_audio_ms),
            format_quality_flags(&setting.quality_flags)
        );
    }
    for row in &report.rows {
        println!(
            concat!(
                "voxtral_matrix.row text_index={} voice={} speed={} max_frames={} flow_steps={} ",
                "reference_text={:?} synthesis_text={:?} ",
                "first_audio_ms={} first_active_audio_ms={} total_ms={:.1} ",
                "model_audio_ms={:.1} model_rtf={} audio_ms={:.1} rtf={} ",
                "wer={} errors={}/{} frames={} ended={} chunks={} ",
                "quality_suspect={} quality_flags={} ",
                "eos_frame={} semantic_count={} semantic_tail={:?} ",
                "semantic_tail_unique={} semantic_tail_repeat={} ",
                "eos_rank_tail={:?} eos_margin_tail={:?} ",
                "best_eos_rank={} best_eos_margin={} ",
                "segments={} long_gaps={} longest_gap_s={:.3} active_ratio={} ",
                "lead_frag_s={} trail_frag_s={} energy_peaks={} energy_peak_rate_hz={} ",
                "energy_peak_interval_s={} energy_peak_interval_cv={} ",
                "spectral_flux_peaks={} spectral_flux_rate_hz={} spectral_flux_interval_s={} ",
                "spectral_flux_interval_cv={} spectral_flux_mean={} spectral_flux_cv={} ",
                "spectrogram={} transcript={:?}"
            ),
            row.text_index,
            row.voice,
            format_speed(row.speed),
            row.max_frames,
            row.flow_steps,
            row.text,
            row.synthesis_text,
            format_optional_f64(row.first_audio_ms),
            format_optional_f64(row.first_active_audio_ms),
            row.total_ms,
            row.model_audio_duration_ms,
            format_optional_f64(row.model_realtime_factor),
            row.audio_duration_ms,
            format_optional_f64(row.realtime_factor),
            format_optional_f32(row.word_error_rate),
            row.word_error_count,
            row.reference_word_count,
            row.audio_frames,
            row.ended,
            row.codec_chunks,
            row.quality_suspect,
            format_quality_flags(&row.quality_flags),
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
            row.audio_diagnostics.energy_peak_count,
            format_optional_f64(row.audio_diagnostics.energy_peak_rate_hz),
            format_optional_f64(row.audio_diagnostics.energy_peak_interval_mean_seconds),
            format_optional_f64(row.audio_diagnostics.energy_peak_interval_cv),
            row.audio_diagnostics.spectral_flux_peak_count,
            format_optional_f64(row.audio_diagnostics.spectral_flux_peak_rate_hz),
            format_optional_f64(row.audio_diagnostics.spectral_flux_interval_mean_seconds),
            format_optional_f64(row.audio_diagnostics.spectral_flux_interval_cv),
            format_optional_f64(row.audio_diagnostics.spectral_flux_mean),
            format_optional_f64(row.audio_diagnostics.spectral_flux_cv),
            format_optional_path(row.spectrogram_pgm.as_ref()),
            row.transcription
        );
    }
}

fn voxtral_quality_flags(
    word_error_count: usize,
    ended: bool,
    diagnostics: &AudioDiagnostics,
) -> Vec<QualityFlag> {
    let mut flags = Vec::new();
    if word_error_count > 0 {
        flags.push(QualityFlag::WordError);
    }
    if !ended {
        flags.push(QualityFlag::DidNotEnd);
    }
    if diagnostics.active_segment_count == 0 {
        flags.push(QualityFlag::NoActiveAudio);
    } else {
        if diagnostics.active_segment_count > 1 {
            flags.push(QualityFlag::ExtraActiveSegments);
        }
        if diagnostics.long_gap_count > 0 {
            flags.push(QualityFlag::LongInternalGap);
        }
        if diagnostics.leading_fragment_seconds.is_some() {
            flags.push(QualityFlag::LeadingFragment);
        }
        if diagnostics.trailing_fragment_seconds.is_some() {
            flags.push(QualityFlag::TrailingFragment);
        }
        if has_irregular_spectral_flux(diagnostics) {
            flags.push(QualityFlag::IrregularSpectralFlux);
        }
    }
    flags
}

fn has_irregular_spectral_flux(diagnostics: &AudioDiagnostics) -> bool {
    const MIN_PEAKS: usize = 8;
    const MIN_INTERVAL_CV: f64 = 0.55;
    const MIN_FLUX_CV: f64 = 0.60;

    diagnostics.spectral_flux_peak_count >= MIN_PEAKS
        && diagnostics
            .spectral_flux_interval_cv
            .is_some_and(|cv| cv >= MIN_INTERVAL_CV)
        && diagnostics
            .spectral_flux_cv
            .is_some_and(|cv| cv >= MIN_FLUX_CV)
}

#[derive(Debug, Default)]
struct QualityAccumulator {
    rows: usize,
    clean_rows: usize,
    suspect_rows: usize,
    ended_rows: usize,
    zero_wer_rows: usize,
    transcript_correct_artifact_rows: usize,
    word_error_rate_sum: f64,
    word_error_rate_count: usize,
    realtime_factor_sum: f64,
    realtime_factor_count: usize,
    model_realtime_factor_sum: f64,
    model_realtime_factor_count: usize,
    first_active_audio_ms_sum: f64,
    first_active_audio_ms_count: usize,
    quality_flags: Vec<QualityFlag>,
}

impl QualityAccumulator {
    fn add_row(&mut self, row: &VoxtralMatrixRow) {
        self.rows += 1;
        if row.quality_suspect {
            self.suspect_rows += 1;
        } else {
            self.clean_rows += 1;
        }
        if row.ended {
            self.ended_rows += 1;
        }
        if row.word_error_count == 0 {
            self.zero_wer_rows += 1;
        }
        if row.word_error_count == 0 && artifact_quality_flags(&row.quality_flags) {
            self.transcript_correct_artifact_rows += 1;
        }
        if let Some(word_error_rate) = row.word_error_rate {
            self.word_error_rate_sum += f64::from(word_error_rate);
            self.word_error_rate_count += 1;
        }
        if let Some(realtime_factor) = row.realtime_factor {
            self.realtime_factor_sum += realtime_factor;
            self.realtime_factor_count += 1;
        }
        if let Some(model_realtime_factor) = row.model_realtime_factor {
            self.model_realtime_factor_sum += model_realtime_factor;
            self.model_realtime_factor_count += 1;
        }
        if let Some(first_active_audio_ms) = row.first_active_audio_ms {
            self.first_active_audio_ms_sum += first_active_audio_ms;
            self.first_active_audio_ms_count += 1;
        }
        for flag in &row.quality_flags {
            if !self.quality_flags.contains(flag) {
                self.quality_flags.push(*flag);
            }
        }
    }

    fn average_word_error_rate(&self) -> Option<f32> {
        (self.word_error_rate_count > 0)
            .then_some((self.word_error_rate_sum / self.word_error_rate_count as f64) as f32)
    }

    fn average_realtime_factor(&self) -> Option<f64> {
        (self.realtime_factor_count > 0)
            .then_some(self.realtime_factor_sum / self.realtime_factor_count as f64)
    }

    fn average_model_realtime_factor(&self) -> Option<f64> {
        (self.model_realtime_factor_count > 0)
            .then_some(self.model_realtime_factor_sum / self.model_realtime_factor_count as f64)
    }

    fn average_first_active_audio_ms(&self) -> Option<f64> {
        (self.first_active_audio_ms_count > 0)
            .then_some(self.first_active_audio_ms_sum / self.first_active_audio_ms_count as f64)
    }
}

fn summarize_voxtral_quality(rows: &[VoxtralMatrixRow]) -> VoxtralQualitySummary {
    let mut total = QualityAccumulator::default();
    let mut by_setting = BTreeMap::<(String, i32, usize, usize), QualityAccumulator>::new();
    for row in rows {
        total.add_row(row);
        by_setting
            .entry((
                row.voice.clone(),
                speed_key(row.speed),
                row.max_frames,
                row.flow_steps,
            ))
            .or_default()
            .add_row(row);
    }

    VoxtralQualitySummary {
        total_rows: total.rows,
        clean_rows: total.clean_rows,
        suspect_rows: total.suspect_rows,
        ended_rows: total.ended_rows,
        zero_wer_rows: total.zero_wer_rows,
        transcript_correct_artifact_rows: total.transcript_correct_artifact_rows,
        by_setting: by_setting
            .into_iter()
            .map(
                |((voice, speed, max_frames, flow_steps), setting)| VoxtralSettingQualitySummary {
                    voice,
                    speed: speed_from_key(speed),
                    max_frames,
                    flow_steps,
                    rows: setting.rows,
                    clean_rows: setting.clean_rows,
                    suspect_rows: setting.suspect_rows,
                    ended_rows: setting.ended_rows,
                    zero_wer_rows: setting.zero_wer_rows,
                    transcript_correct_artifact_rows: setting.transcript_correct_artifact_rows,
                    average_word_error_rate: setting.average_word_error_rate(),
                    average_realtime_factor: setting.average_realtime_factor(),
                    average_model_realtime_factor: setting.average_model_realtime_factor(),
                    average_first_active_audio_ms: setting.average_first_active_audio_ms(),
                    quality_flags: setting.quality_flags,
                },
            )
            .collect(),
    }
}

fn speed_key(speed: f32) -> i32 {
    (speed * 1_000.0).round() as i32
}

fn speed_from_key(speed: i32) -> f32 {
    speed as f32 / 1_000.0
}

fn artifact_quality_flags(flags: &[QualityFlag]) -> bool {
    flags.iter().any(|flag| {
        matches!(
            flag,
            QualityFlag::NoActiveAudio
                | QualityFlag::ExtraActiveSegments
                | QualityFlag::LongInternalGap
                | QualityFlag::LeadingFragment
                | QualityFlag::TrailingFragment
                | QualityFlag::IrregularSpectralFlux
        )
    })
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

fn first_active_audio_ms(
    first_audio_ms: Option<f64>,
    diagnostics: &AudioDiagnostics,
) -> Option<f64> {
    if diagnostics.active_segment_count == 0 {
        return None;
    }
    first_audio_ms
        .map(|first_audio_ms| first_audio_ms + diagnostics.leading_silence_seconds * 1_000.0)
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

fn format_speed(speed: f32) -> String {
    let mut formatted = format!("{speed:.3}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.pop();
    }
    formatted
}

fn file_stem(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn format_quality_flags(flags: &[QualityFlag]) -> String {
    if flags.is_empty() {
        return "none".to_string();
    }
    flags
        .iter()
        .map(|flag| flag.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

fn print_text_report(report: &EvalReport) {
    println!("reference: {}", report.text);
    if report.synthesis_text != report.text {
        println!("synthesis text: {}", report.synthesis_text);
    }
    if report.voxtral_text_normalization {
        println!("voxtral text normalization: enabled");
    }
    if report.voxtral_pronunciation_aliases {
        println!("voxtral pronunciation aliases: enabled");
    }
    if report.voxtral_auto_max_frames {
        println!("voxtral auto max frames: enabled");
    }
    if report.time_token_equivalence {
        println!("time token equivalence: enabled");
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
            "first_active_s={} leading_silence_s={:.3} trailing_silence_s={:.3} ",
            "lead_frag_s={} trail_frag_s={} energy_peaks={} energy_peak_rate_hz={} ",
            "energy_peak_interval_s={} energy_peak_interval_cv={} ",
            "spectral_flux_peaks={} spectral_flux_rate_hz={} spectral_flux_interval_s={} ",
            "spectral_flux_interval_cv={} spectral_flux_mean={} spectral_flux_cv={}"
        ),
        report.audio_diagnostics.peak_dbfs,
        report.audio_diagnostics.rms_dbfs,
        report.audio_diagnostics.active_threshold_dbfs,
        report.audio_diagnostics.active_segment_count,
        format_optional_f64(report.audio_diagnostics.active_ratio),
        report.audio_diagnostics.long_gap_count,
        report.audio_diagnostics.longest_internal_gap_seconds,
        format_optional_f64(report.audio_diagnostics.first_active_seconds),
        report.audio_diagnostics.leading_silence_seconds,
        report.audio_diagnostics.trailing_silence_seconds,
        format_optional_f64(report.audio_diagnostics.leading_fragment_seconds),
        format_optional_f64(report.audio_diagnostics.trailing_fragment_seconds),
        report.audio_diagnostics.energy_peak_count,
        format_optional_f64(report.audio_diagnostics.energy_peak_rate_hz),
        format_optional_f64(report.audio_diagnostics.energy_peak_interval_mean_seconds),
        format_optional_f64(report.audio_diagnostics.energy_peak_interval_cv),
        report.audio_diagnostics.spectral_flux_peak_count,
        format_optional_f64(report.audio_diagnostics.spectral_flux_peak_rate_hz),
        format_optional_f64(report.audio_diagnostics.spectral_flux_interval_mean_seconds),
        format_optional_f64(report.audio_diagnostics.spectral_flux_interval_cv),
        format_optional_f64(report.audio_diagnostics.spectral_flux_mean),
        format_optional_f64(report.audio_diagnostics.spectral_flux_cv)
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct WerOptions {
    time_token_equivalence: bool,
}

fn wer_options(args: &Args) -> WerOptions {
    WerOptions {
        time_token_equivalence: args.time_token_equivalence,
    }
}

#[cfg(test)]
fn word_error_rate(reference: &str, hypothesis: &str) -> Wer {
    word_error_rate_with_options(reference, hypothesis, WerOptions::default())
}

fn word_error_rate_with_options(reference: &str, hypothesis: &str, options: WerOptions) -> Wer {
    let reference_words = normalize_words_with_options(reference, options);
    let hypothesis_words = normalize_words_with_options(hypothesis, options);
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

#[cfg(test)]
fn normalize_words(text: &str) -> Vec<String> {
    normalize_words_with_options(text, WerOptions::default())
}

fn normalize_words_with_options(text: &str, options: WerOptions) -> Vec<String> {
    let mut normalized = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '\'' {
            normalized.extend(ch.to_lowercase());
        } else {
            normalized.push(' ');
        }
    }
    let words = normalized
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let words = normalize_known_compound_tokens(words);
    if options.time_token_equivalence {
        normalize_time_tokens(words)
    } else {
        words
    }
}

fn normalize_known_compound_tokens(words: Vec<String>) -> Vec<String> {
    let mut normalized = Vec::with_capacity(words.len());
    let mut index = 0;
    while index < words.len() {
        if words.get(index).is_some_and(|word| word == "real")
            && words.get(index + 1).is_some_and(|word| word == "time")
        {
            normalized.push("realtime".to_string());
            index += 2;
        } else {
            normalized.push(words[index].clone());
            index += 1;
        }
    }
    normalized
}

fn normalize_time_tokens(words: Vec<String>) -> Vec<String> {
    let mut normalized = Vec::with_capacity(words.len());
    let mut index = 0;
    while index < words.len() {
        let word = &words[index];
        if (word == "a" || word == "p") && words.get(index + 1).is_some_and(|next| next == "m") {
            normalized.push(format!("{word}m"));
            index += 2;
            continue;
        }
        if has_adjacent_meridiem(&words, index) {
            if let Some((hour, minute)) = split_compact_time_token(word) {
                normalized.push(hour);
                normalized.push(minute);
            } else if should_canonicalize_time_digit(&words, index) {
                normalized.push(canonical_digit_token(word));
            } else {
                normalized.push(word.clone());
            }
        } else {
            normalized.push(word.clone());
        }
        index += 1;
    }
    normalized
}

fn has_adjacent_meridiem(words: &[String], index: usize) -> bool {
    index
        .checked_sub(1)
        .and_then(|previous| words.get(previous))
        .is_some_and(|word| is_meridiem_token(word))
        || index
            .checked_sub(2)
            .is_some_and(|previous| is_meridiem_pair(&words[previous], &words[index - 1]))
        || words
            .get(index + 1)
            .is_some_and(|word| is_meridiem_token(word))
        || words
            .get(index + 1)
            .zip(words.get(index + 2))
            .is_some_and(|(first, second)| is_meridiem_pair(first, second))
}

fn is_meridiem_token(word: &str) -> bool {
    word == "am" || word == "pm"
}

fn is_meridiem_pair(first: &str, second: &str) -> bool {
    (first == "a" || first == "p") && second == "m"
}

fn split_compact_time_token(token: &str) -> Option<(String, String)> {
    if !(token.len() == 3 || token.len() == 4) || !token.as_bytes().iter().all(u8::is_ascii_digit) {
        return None;
    }
    let hour_len = token.len() - 2;
    let (hour, minute) = token.split_at(hour_len);
    let hour_value = hour.parse::<u32>().ok()?;
    let minute_value = minute.parse::<u32>().ok()?;
    if !(1..=23).contains(&hour_value) || minute_value > 59 {
        return None;
    }
    Some((canonical_digit_token(hour), canonical_digit_token(minute)))
}

fn should_canonicalize_time_digit(words: &[String], index: usize) -> bool {
    let word = &words[index];
    (word.len() == 1 || word.len() == 2)
        && word.as_bytes().iter().all(u8::is_ascii_digit)
        && has_adjacent_meridiem(words, index)
}

fn canonical_digit_token(token: &str) -> String {
    let trimmed = token.trim_start_matches('0');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
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
    fn normalizes_known_compound_tokens_for_eval() {
        let equivalent = word_error_rate(
            "The sentence reaches the realtime frame cap.",
            "The sentence reaches the real-time frame cap.",
        );
        assert_eq!(equivalent.distance, 0);
        assert_eq!(
            equivalent.reference_words,
            vec!["the", "sentence", "reaches", "the", "realtime", "frame", "cap"]
        );
        assert_eq!(
            equivalent.hypothesis_words,
            vec!["the", "sentence", "reaches", "the", "realtime", "frame", "cap"]
        );

        let spaced = word_error_rate(
            "The sentence reaches the realtime frame cap.",
            "The sentence reaches the real time frame cap.",
        );
        assert_eq!(spaced.distance, 0);
    }

    #[test]
    fn time_token_equivalence_is_opt_in_for_eval() {
        let default = word_error_rate("Use API on CPU at 12:05 AM.", "Use API on CPU at 1205 a.m.");
        assert_ne!(default.distance, 0);
        assert_eq!(
            default.reference_words,
            vec!["use", "api", "on", "cpu", "at", "12", "05", "am"]
        );
        assert_eq!(
            default.hypothesis_words,
            vec!["use", "api", "on", "cpu", "at", "1205", "a", "m"]
        );

        let equivalent = word_error_rate_with_options(
            "Use API on CPU at 12:05 AM.",
            "Use API on CPU at 1205 a.m.",
            WerOptions {
                time_token_equivalence: true,
            },
        );
        assert_eq!(equivalent.distance, 0);
        assert_eq!(
            equivalent.reference_words,
            vec!["use", "api", "on", "cpu", "at", "12", "5", "am"]
        );
        assert_eq!(
            equivalent.hypothesis_words,
            vec!["use", "api", "on", "cpu", "at", "12", "5", "am"]
        );
    }

    #[test]
    fn time_token_equivalence_preserves_non_time_numbers() {
        assert_eq!(
            normalize_words_with_options(
                "Ticket 100, room 2000, agent 007, code 05.",
                WerOptions {
                    time_token_equivalence: true,
                },
            ),
            vec!["ticket", "100", "room", "2000", "agent", "007", "code", "05"]
        );
    }

    #[test]
    fn time_token_equivalence_handles_single_token_meridiem() {
        let equivalent = word_error_rate_with_options(
            "Use API on CPU at 12:05 PM.",
            "Use API on CPU at 1205 pm.",
            WerOptions {
                time_token_equivalence: true,
            },
        );
        assert_eq!(equivalent.distance, 0);
        assert_eq!(
            equivalent.reference_words,
            vec!["use", "api", "on", "cpu", "at", "12", "5", "pm"]
        );
        assert_eq!(
            equivalent.hypothesis_words,
            vec!["use", "api", "on", "cpu", "at", "12", "5", "pm"]
        );
    }

    #[test]
    fn time_token_equivalence_matches_compact_time_on_one_side() {
        let equivalent = word_error_rate_with_options(
            "Start at 12 05 PM.",
            "Start at 1205 PM.",
            WerOptions {
                time_token_equivalence: true,
            },
        );
        assert_eq!(equivalent.distance, 0);
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
            "--time-token-equivalence",
        ])
        .unwrap();

        assert_eq!(args.stt_backend, "voxtral");
        assert_eq!(args.stt_model.as_deref(), Some("/tmp/voxtral-realtime"));
        assert_eq!(args.stt_max_tokens, Some(4));
        assert!(args.time_token_equivalence);
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
            "--voxtral-pronunciation-aliases",
            "--voxtral-auto-max-frames",
            "--time-token-equivalence",
        ])
        .unwrap();

        assert_eq!(args.text, "Voxtral should sound clear.");
        assert_eq!(
            args.synthesis_text.as_deref(),
            Some("Vox trahl should sound clear.")
        );
        assert!(args.voxtral_normalize_text);
        assert!(args.voxtral_pronunciation_aliases);
        assert!(args.voxtral_auto_max_frames);
        assert!(args.time_token_equivalence);
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
            "--voxtral-pronunciation-aliases",
            "--voxtral-auto-max-frames",
            "--time-token-equivalence",
            "--voxtral-eos-scores",
            "--voxtral-eos-guard-frames",
            "8",
            "--voxtral-eos-guard-rank",
            "3",
            "--voxtral-eos-guard-margin",
            "0.75",
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
        assert!(args.voxtral_pronunciation_aliases);
        assert!(args.voxtral_auto_max_frames);
        assert!(args.time_token_equivalence);
        assert!(args.voxtral_eos_scores);
        assert_eq!(args.voxtral_eos_guard_frames, 8);
        assert_eq!(args.voxtral_eos_guard_rank, 3);
        assert_eq!(args.voxtral_eos_guard_margin, 0.75);
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
    fn matrix_cases_apply_opt_in_voxtral_pronunciation_aliases() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-matrix",
            "--voxtral-pronunciation-aliases",
            "--matrix-text",
            "Voxtral reads A17.",
            "--matrix-text",
            "Kokoro should stay unchanged.",
        ])
        .unwrap();

        assert_eq!(
            matrix_cases(&args).unwrap(),
            vec![
                MatrixCase {
                    reference_text: "Voxtral reads A17.".to_string(),
                    synthesis_text: "Vox trell reads A17.".to_string(),
                },
                MatrixCase {
                    reference_text: "Kokoro should stay unchanged.".to_string(),
                    synthesis_text: "Kokoro should stay unchanged.".to_string(),
                }
            ]
        );
    }

    #[test]
    fn voxtral_quality_suite_uses_canonical_prompt_set() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-quality-suite",
        ])
        .unwrap();

        let cases = matrix_cases(&args).unwrap();
        assert_eq!(cases.len(), VOXTRAL_QUALITY_SUITE_PROMPTS.len());
        assert_eq!(cases[0].reference_text, "hello world");
        assert_eq!(
            cases[2].reference_text,
            "Voxtral should pronounce its own made-up name clearly."
        );
        assert_eq!(
            cases.last().unwrap().reference_text,
            "The voice should stay steady across a longer reply, even when the sentence reaches the realtime frame cap."
        );
        assert!(cases
            .iter()
            .all(|case| case.reference_text == case.synthesis_text));
    }

    #[test]
    fn voxtral_quality_suite_applies_opt_in_synthesis_rewrites() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-quality-suite",
            "--voxtral-normalize-text",
            "--voxtral-pronunciation-aliases",
        ])
        .unwrap();

        let cases = matrix_cases(&args).unwrap();
        assert_eq!(
            cases[2].reference_text,
            "Voxtral should pronounce its own made-up name clearly."
        );
        assert_eq!(
            cases[2].synthesis_text,
            "Vox trell should pronounce its own made-up name clearly."
        );
        assert_eq!(
            cases[4].reference_text,
            "Read ticket A17, version 2.4.1, at 9:30 PM."
        );
        assert_eq!(
            cases[4].synthesis_text,
            "Read ticket A seventeen, version two point four point one, at nine thirty PM."
        );
    }

    #[test]
    fn voxtral_quality_suite_rejects_custom_matrix_texts() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-quality-suite",
            "--matrix-text",
            "custom prompt",
        ])
        .unwrap();

        let err = run_voxtral_matrix(&args).unwrap_err().to_string();
        assert!(err.contains("--voxtral-quality-suite cannot be combined with --matrix-text"));
    }

    #[test]
    fn voxtral_quality_suite_rejects_custom_matrix_synthesis_texts() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-quality-suite",
            "--matrix-synthesis-text",
            "custom synthesis",
        ])
        .unwrap();

        let err = run_voxtral_matrix(&args).unwrap_err().to_string();
        assert!(
            err.contains("--voxtral-quality-suite cannot be combined with --matrix-synthesis-text")
        );
    }

    #[test]
    fn auto_voxtral_max_frames_uses_synthesis_text_estimate_without_lowering_explicit_caps() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-auto-max-frames",
            "--max-frames",
            "40",
        ])
        .unwrap();

        assert_eq!(
            effective_voxtral_max_frames(
                &args,
                "Vox trell should pronounce Vox trell clearly in a short answer.",
                args.max_frames,
            ),
            56
        );

        let explicit_high = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-auto-max-frames",
            "--max-frames",
            "200",
        ])
        .unwrap();
        assert_eq!(
            effective_voxtral_max_frames(&explicit_high, "hello world", explicit_high.max_frames),
            200
        );

        let fixed = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--max-frames",
            "40",
        ])
        .unwrap();
        assert_eq!(
            effective_voxtral_max_frames(&fixed, "hello world", fixed.max_frames),
            40
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
    fn matrix_voices_default_to_selected_voice_and_validate_known_voices() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-matrix",
            "--voice",
            "nl_female",
        ])
        .unwrap();
        assert_eq!(matrix_voices(&args).unwrap(), vec!["nl_female"]);

        let multi = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-matrix",
            "--matrix-voice",
            "casual_male,nl_female",
        ])
        .unwrap();
        assert_eq!(
            matrix_voices(&multi).unwrap(),
            vec!["casual_male", "nl_female"]
        );

        let bad = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-matrix",
            "--matrix-voice",
            "not_a_voice",
        ])
        .unwrap();
        let err = matrix_voices(&bad).unwrap_err().to_string();
        assert!(err.contains("unknown Voxtral voice"));
    }

    #[test]
    fn matrix_speeds_default_to_speed_and_validate_positive_values() {
        let args = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-matrix",
            "--speed",
            "1.15",
        ])
        .unwrap();
        assert_eq!(matrix_speeds(&args).unwrap(), vec![1.15]);

        let multi = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-matrix",
            "--matrix-speed",
            "1.0,1.15",
        ])
        .unwrap();
        assert_eq!(matrix_speeds(&multi).unwrap(), vec![1.0, 1.15]);

        let bad = Args::try_parse_from([
            "voice-eval",
            "--tts-backend",
            "voxtral",
            "--voxtral-matrix",
            "--matrix-speed",
            "0.0",
        ])
        .unwrap();
        let err = matrix_speeds(&bad).unwrap_err().to_string();
        assert!(err.contains("speed values must be finite and greater than zero"));
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
        assert!(diagnostics.first_active_seconds.unwrap_or(1.0) < 0.01);
        assert!(diagnostics.leading_fragment_seconds.is_none());
        assert!(diagnostics.trailing_fragment_seconds.is_none());
    }

    #[test]
    fn audio_diagnostics_report_first_active_sample_after_leading_silence() {
        let sample_rate = 1_000;
        let mut samples = silence_samples(sample_rate, 0.25);
        samples.extend(tone_samples(sample_rate, 0.50, 0.5));

        let diagnostics = analyze_audio(&samples, sample_rate);

        let first_active_seconds = diagnostics.first_active_seconds.unwrap();
        assert_eq!(diagnostics.active_segment_count, 1);
        assert!((0.23..=0.26).contains(&first_active_seconds));
        assert_eq!(first_active_seconds, diagnostics.leading_silence_seconds);
        assert_eq!(
            first_active_audio_ms(Some(120.0), &diagnostics)
                .unwrap()
                .round(),
            (120.0 + first_active_seconds * 1_000.0).round()
        );
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
    fn audio_diagnostics_report_energy_peak_rhythm() {
        let sample_rate = 1_000;
        let mut samples = Vec::new();
        for _ in 0..4 {
            samples.extend(tone_samples(sample_rate, 0.08, 0.5));
            samples.extend(tone_samples(sample_rate, 0.08, 0.1));
        }

        let diagnostics = analyze_audio(&samples, sample_rate);

        assert!(diagnostics.energy_peak_count >= 3);
        assert!(diagnostics.energy_peak_rate_hz.unwrap_or(0.0) > 1.0);
        assert!(diagnostics.energy_peak_interval_mean_seconds.unwrap_or(0.0) > 0.05);
    }

    #[test]
    fn audio_diagnostics_report_spectral_flux_onsets() {
        let sample_rate = 8_000;
        let mut samples = Vec::new();
        for _ in 0..6 {
            samples.extend(tone_samples(sample_rate, 0.08, 0.5));
            samples.extend(silence_samples(sample_rate, 0.12));
        }
        let diagnostics = analyze_audio(&samples, sample_rate);

        assert!(diagnostics.spectral_flux_peak_count >= 4);
        assert!(diagnostics.spectral_flux_peak_rate_hz.unwrap_or(0.0) > 2.0);
        assert!(
            diagnostics
                .spectral_flux_interval_mean_seconds
                .unwrap_or(0.0)
                > 0.05
        );
        assert!(diagnostics.spectral_flux_mean.unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn voxtral_quality_flags_keep_clean_rows_unflagged() {
        let sample_rate = 1_000;
        let diagnostics = analyze_audio(&tone_samples(sample_rate, 1.0, 0.5), sample_rate);

        assert_eq!(voxtral_quality_flags(0, true, &diagnostics), Vec::new());
        assert_eq!(format_quality_flags(&[]), "none");
    }

    #[test]
    fn voxtral_quality_flags_mark_transcript_and_cap_failures() {
        let sample_rate = 1_000;
        let diagnostics = analyze_audio(&tone_samples(sample_rate, 1.0, 0.5), sample_rate);

        assert_eq!(
            voxtral_quality_flags(2, false, &diagnostics),
            vec![QualityFlag::WordError, QualityFlag::DidNotEnd]
        );
        assert_eq!(
            format_quality_flags(&[QualityFlag::WordError, QualityFlag::DidNotEnd]),
            "word_error,did_not_end"
        );
    }

    #[test]
    fn voxtral_quality_flags_mark_transcript_correct_audio_artifacts() {
        let sample_rate = 1_000;
        let mut samples = Vec::new();
        samples.extend(tone_samples(sample_rate, 0.08, 0.5));
        samples.extend(silence_samples(sample_rate, 0.40));
        samples.extend(tone_samples(sample_rate, 0.50, 0.5));
        samples.extend(silence_samples(sample_rate, 0.30));
        samples.extend(tone_samples(sample_rate, 0.10, 0.5));
        let diagnostics = analyze_audio(&samples, sample_rate);

        assert_eq!(
            voxtral_quality_flags(0, true, &diagnostics),
            vec![
                QualityFlag::ExtraActiveSegments,
                QualityFlag::LongInternalGap,
                QualityFlag::LeadingFragment,
                QualityFlag::TrailingFragment
            ]
        );
    }

    #[test]
    fn voxtral_quality_flags_mark_irregular_spectral_flux() {
        let mut diagnostics = audio_diagnostics_for_test();
        diagnostics.spectral_flux_peak_count = 12;
        diagnostics.spectral_flux_interval_cv = Some(0.70);
        diagnostics.spectral_flux_cv = Some(0.80);

        assert_eq!(
            voxtral_quality_flags(0, true, &diagnostics),
            vec![QualityFlag::IrregularSpectralFlux]
        );
    }

    #[test]
    fn voxtral_quality_flags_mark_silent_rows() {
        let sample_rate = 1_000;
        let diagnostics = analyze_audio(&silence_samples(sample_rate, 0.5), sample_rate);

        assert_eq!(
            voxtral_quality_flags(0, true, &diagnostics),
            vec![QualityFlag::NoActiveAudio]
        );
    }

    #[test]
    fn voxtral_quality_summary_groups_rows_by_setting() {
        let rows = vec![
            matrix_row_for_test(32, 5, true, 0, Some(0.0), Some(1.1), Some(350.0), vec![]),
            matrix_row_for_test(
                32,
                5,
                false,
                2,
                Some(0.25),
                Some(1.3),
                Some(420.0),
                vec![QualityFlag::WordError, QualityFlag::DidNotEnd],
            ),
            matrix_row_for_test(
                40,
                7,
                true,
                0,
                Some(0.0),
                Some(0.9),
                Some(300.0),
                vec![
                    QualityFlag::ExtraActiveSegments,
                    QualityFlag::LeadingFragment,
                ],
            ),
        ];

        let summary = summarize_voxtral_quality(&rows);

        assert_eq!(summary.total_rows, 3);
        assert_eq!(summary.clean_rows, 1);
        assert_eq!(summary.suspect_rows, 2);
        assert_eq!(summary.ended_rows, 2);
        assert_eq!(summary.zero_wer_rows, 2);
        assert_eq!(summary.transcript_correct_artifact_rows, 1);
        assert_eq!(summary.by_setting.len(), 2);

        let setting_32_5 = &summary.by_setting[0];
        assert_eq!(setting_32_5.voice, "casual_male");
        assert_eq!(setting_32_5.speed, 1.0);
        assert_eq!(setting_32_5.max_frames, 32);
        assert_eq!(setting_32_5.flow_steps, 5);
        assert_eq!(setting_32_5.rows, 2);
        assert_eq!(setting_32_5.clean_rows, 1);
        assert_eq!(setting_32_5.suspect_rows, 1);
        assert_eq!(
            setting_32_5.quality_flags,
            vec![QualityFlag::WordError, QualityFlag::DidNotEnd,]
        );
        assert_eq!(setting_32_5.average_word_error_rate, Some(0.125));
        assert!(
            (setting_32_5.average_realtime_factor.unwrap() - 1.2).abs() < 0.000_001,
            "unexpected average realtime factor: {:?}",
            setting_32_5.average_realtime_factor
        );
        assert!(
            (setting_32_5.average_model_realtime_factor.unwrap() - 1.2).abs() < 0.000_001,
            "unexpected average model realtime factor: {:?}",
            setting_32_5.average_model_realtime_factor
        );
        assert_eq!(setting_32_5.average_first_active_audio_ms, Some(385.0));

        let setting_40_7 = &summary.by_setting[1];
        assert_eq!(setting_40_7.voice, "casual_male");
        assert_eq!(setting_40_7.speed, 1.0);
        assert_eq!(setting_40_7.max_frames, 40);
        assert_eq!(setting_40_7.flow_steps, 7);
        assert_eq!(setting_40_7.transcript_correct_artifact_rows, 1);
        assert_eq!(
            setting_40_7.quality_flags,
            vec![
                QualityFlag::ExtraActiveSegments,
                QualityFlag::LeadingFragment,
            ]
        );
    }

    #[test]
    fn voxtral_quality_summary_keeps_voice_and_speed_settings_separate() {
        let rows = vec![
            matrix_row_for_test_with_voice_speed("casual_male", 1.0, 40, 7, vec![]),
            matrix_row_for_test_with_voice_speed(
                "nl_female",
                1.0,
                40,
                7,
                vec![QualityFlag::WordError],
            ),
            matrix_row_for_test_with_voice_speed(
                "nl_female",
                1.15,
                40,
                7,
                vec![QualityFlag::LongInternalGap],
            ),
        ];

        let summary = summarize_voxtral_quality(&rows);

        assert_eq!(summary.by_setting.len(), 3);
        assert_eq!(summary.by_setting[0].voice, "casual_male");
        assert_eq!(summary.by_setting[0].speed, 1.0);
        assert_eq!(summary.by_setting[1].voice, "nl_female");
        assert_eq!(summary.by_setting[1].speed, 1.0);
        assert_eq!(summary.by_setting[2].voice, "nl_female");
        assert_eq!(summary.by_setting[2].speed, 1.15);
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

    #[allow(clippy::too_many_arguments)]
    fn matrix_row_for_test(
        max_frames: usize,
        flow_steps: usize,
        ended: bool,
        word_error_count: usize,
        word_error_rate: Option<f32>,
        realtime_factor: Option<f64>,
        first_active_audio_ms: Option<f64>,
        quality_flags: Vec<QualityFlag>,
    ) -> VoxtralMatrixRow {
        matrix_row_for_test_full(
            "casual_male",
            1.0,
            max_frames,
            flow_steps,
            ended,
            word_error_count,
            word_error_rate,
            realtime_factor,
            first_active_audio_ms,
            quality_flags,
        )
    }

    fn matrix_row_for_test_with_voice_speed(
        voice: &str,
        speed: f32,
        max_frames: usize,
        flow_steps: usize,
        quality_flags: Vec<QualityFlag>,
    ) -> VoxtralMatrixRow {
        matrix_row_for_test_full(
            voice,
            speed,
            max_frames,
            flow_steps,
            quality_flags.is_empty(),
            usize::from(quality_flags.contains(&QualityFlag::WordError)),
            Some(0.0),
            Some(1.0),
            Some(300.0),
            quality_flags,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn matrix_row_for_test_full(
        voice: &str,
        speed: f32,
        max_frames: usize,
        flow_steps: usize,
        ended: bool,
        word_error_count: usize,
        word_error_rate: Option<f32>,
        realtime_factor: Option<f64>,
        first_active_audio_ms: Option<f64>,
        quality_flags: Vec<QualityFlag>,
    ) -> VoxtralMatrixRow {
        let quality_suspect = !quality_flags.is_empty();
        VoxtralMatrixRow {
            text_index: 0,
            voice: voice.to_string(),
            text: "reference".to_string(),
            synthesis_text: "synthesis".to_string(),
            speed,
            max_frames,
            flow_steps,
            first_code_frame_ms: Some(10.0),
            first_audio_ms: Some(250.0),
            first_active_audio_ms,
            total_ms: 1000.0,
            model_audio_duration_ms: 900.0,
            model_realtime_factor: realtime_factor,
            audio_duration_ms: 900.0,
            realtime_factor,
            samples: 24_000,
            sample_rate: 24_000,
            audio_frames: max_frames,
            ended,
            eos_frame: ended.then_some(audio_frames_for_test(max_frames)),
            semantic_code_count: max_frames,
            semantic_tail_codes: Vec::new(),
            semantic_tail_unique_count: 0,
            semantic_tail_repeat_count: 0,
            semantic_eos_rank_tail: Vec::new(),
            semantic_eos_margin_tail: Vec::new(),
            semantic_best_eos_rank: None,
            semantic_best_eos_margin: None,
            codec_chunks: 1,
            language_ms: 100.0,
            language_ms_per_frame: Some(2.5),
            acoustic_ms: 200.0,
            acoustic_ms_per_frame: Some(5.0),
            decode_loop_ms: 300.0,
            decode_loop_ms_per_frame: Some(7.5),
            codec_ms: 30.0,
            codec_ms_per_chunk: Some(30.0),
            transcription: "hypothesis".to_string(),
            normalized_reference: "reference".to_string(),
            normalized_hypothesis: "hypothesis".to_string(),
            reference_word_count: 4,
            word_error_count,
            word_error_rate,
            quality_suspect,
            quality_flags,
            stt_token_count: 4,
            audio_diagnostics: audio_diagnostics_for_test(),
            output_wav: None,
            spectrogram_pgm: None,
        }
    }

    fn audio_frames_for_test(max_frames: usize) -> usize {
        max_frames.saturating_sub(1)
    }

    fn audio_diagnostics_for_test() -> AudioDiagnostics {
        AudioDiagnostics {
            duration_seconds: 1.0,
            peak_dbfs: -1.0,
            rms_dbfs: -12.0,
            active_threshold_dbfs: -41.0,
            active_duration_seconds: 1.0,
            active_span_seconds: 1.0,
            active_ratio: Some(1.0),
            first_active_seconds: Some(0.0),
            active_segment_count: 1,
            long_gap_count: 0,
            longest_internal_gap_seconds: 0.0,
            gap_seconds: Vec::new(),
            leading_silence_seconds: 0.0,
            trailing_silence_seconds: 0.0,
            leading_fragment_seconds: None,
            trailing_fragment_seconds: None,
            energy_peak_count: 0,
            energy_peak_rate_hz: None,
            energy_peak_interval_mean_seconds: None,
            energy_peak_interval_cv: None,
            spectral_flux_peak_count: 0,
            spectral_flux_peak_rate_hz: None,
            spectral_flux_interval_mean_seconds: None,
            spectral_flux_interval_cv: None,
            spectral_flux_mean: None,
            spectral_flux_cv: None,
            segments: vec![AudioSegment {
                start_seconds: 0.0,
                end_seconds: 1.0,
                duration_seconds: 1.0,
                peak_dbfs: -1.0,
                rms_dbfs: -12.0,
            }],
        }
    }
}
