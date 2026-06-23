use std::env;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use candle_core::{DType, Device};
use serde::Serialize;
use voice_voxtral::{VoxtralGenerationOptions, VoxtralTtsRuntime};

#[derive(Debug)]
struct Args {
    model_dir: String,
    device: PerfDevice,
    dtype: DType,
    voices: Vec<String>,
    text: String,
    max_frames: usize,
    flow_steps: usize,
    seed: u64,
    warm_runs: usize,
    output_dir: Option<PathBuf>,
    use_kv_cache: bool,
    json: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum PerfDevice {
    #[cfg_attr(not(target_os = "macos"), default)]
    Cpu,
    #[cfg_attr(target_os = "macos", default)]
    Metal,
}

#[derive(Debug, Serialize)]
struct PerfReport {
    model_dir: String,
    device: PerfDevice,
    dtype: String,
    text: String,
    voices: Vec<String>,
    warm_runs: usize,
    max_frames: usize,
    flow_steps: usize,
    seed: u64,
    use_kv_cache: bool,
    output_dir: Option<String>,
    module_loads: usize,
    load: LoadReport,
    generations: Vec<GenerationReport>,
    totals: TotalsReport,
}

#[derive(Debug, Serialize)]
struct LoadReport {
    model_ms: f64,
    module_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Serialize)]
struct GenerationReport {
    run: usize,
    voice: String,
    frames: usize,
    ended: bool,
    samples: usize,
    sample_rate: u32,
    language_cache: bool,
    voice_cache_hit: bool,
    cached_voices_after: usize,
    voice_load_ms: f64,
    prompt_ms: f64,
    language_ms: f64,
    acoustic_ms: f64,
    decode_loop_ms: f64,
    first_frame_ms: Option<f64>,
    codec_ms: f64,
    total_ms: f64,
    output_wav: Option<String>,
}

#[derive(Debug, Serialize)]
struct TotalsReport {
    generation_ms: f64,
    voice_cache_hits: usize,
    voice_cache_misses: usize,
    cached_voices: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut model_dir: Option<String> = None;
        let mut device = PerfDevice::default();
        let mut dtype = DType::F16;
        let mut voices = vec!["casual_male".to_string()];
        let mut text = "Voxtral sounds better when the runtime stays warm.".to_string();
        let mut max_frames = 50usize;
        let mut flow_steps = 7usize;
        let mut seed = 0x5658_5452_414c;
        let mut warm_runs = 1usize;
        let mut output_dir = None;
        let mut use_kv_cache = false;
        let mut json = false;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-dir" | "--model" => {
                    model_dir = Some(args.next().ok_or("--model-dir requires a value")?);
                }
                "--device" => {
                    device = parse_device(&args.next().ok_or("--device requires a value")?)?;
                }
                "--dtype" => {
                    dtype = parse_dtype(&args.next().ok_or("--dtype requires a value")?)?;
                }
                "--voices" => {
                    voices = parse_voices(&args.next().ok_or("--voices requires a value")?)?;
                }
                "--text" => {
                    text = args.next().ok_or("--text requires a value")?;
                }
                "--max-frames" => {
                    max_frames = args
                        .next()
                        .ok_or("--max-frames requires a value")?
                        .parse()?;
                }
                "--flow-steps" => {
                    flow_steps = args
                        .next()
                        .ok_or("--flow-steps requires a value")?
                        .parse()?;
                }
                "--seed" => {
                    seed = args.next().ok_or("--seed requires a value")?.parse()?;
                }
                "--warm-runs" => {
                    warm_runs = args.next().ok_or("--warm-runs requires a value")?.parse()?;
                }
                "--output-dir" => {
                    output_dir = Some(PathBuf::from(
                        args.next().ok_or("--output-dir requires a value")?,
                    ));
                }
                "--kv-cache" => use_kv_cache = true,
                "--json" => json = true,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument {arg:?}").into()),
            }
        }

        let model_dir =
            model_dir.ok_or("missing --model-dir PATH or HuggingFace repo id for Voxtral TTS")?;
        if voices.is_empty() {
            return Err("--voices must include at least one voice".into());
        }
        if max_frames == 0 {
            return Err("--max-frames must be greater than zero".into());
        }
        if flow_steps == 0 {
            return Err("--flow-steps must be greater than zero".into());
        }
        if warm_runs == 0 {
            return Err("--warm-runs must be greater than zero".into());
        }

        Ok(Self {
            model_dir,
            device,
            dtype,
            voices,
            text,
            max_frames,
            flow_steps,
            seed,
            warm_runs,
            output_dir,
            use_kv_cache,
            json,
        })
    }
}

impl PerfDevice {
    fn load(self) -> Result<Device, Box<dyn Error>> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Metal => Device::new_metal(0).map_err(|e| e.to_string().into()),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let device = args.device.load()?;
    let (mut runtime, load_trace) =
        VoxtralTtsRuntime::load_with_trace(&args.model_dir, args.dtype, device)?;

    let mut generations = Vec::with_capacity(args.voices.len() * args.warm_runs);
    for run in 0..args.warm_runs {
        for voice in &args.voices {
            let (audio, trace) = runtime.generate_audio_with_trace(
                &args.text,
                voice,
                VoxtralGenerationOptions {
                    max_frames: args.max_frames,
                    seed: args.seed,
                    flow_steps: args.flow_steps,
                    use_kv_cache: args.use_kv_cache,
                    ..Default::default()
                },
            )?;
            let output_wav = if let Some(output_dir) = &args.output_dir {
                fs::create_dir_all(output_dir)?;
                let path = output_dir.join(format!("run{}_{}.wav", run + 1, file_stem(voice)));
                write_wav_pcm16(&path, &audio.samples, audio.sample_rate)?;
                Some(path.display().to_string())
            } else {
                None
            };
            generations.push(GenerationReport {
                run: run + 1,
                voice: voice.clone(),
                frames: audio.frames,
                ended: audio.ended,
                samples: audio.samples.len(),
                sample_rate: audio.sample_rate,
                language_cache: trace.language_cache,
                voice_cache_hit: trace.voice_cache_hit,
                cached_voices_after: runtime.cached_voice_count(),
                voice_load_ms: ms(trace.voice_load),
                prompt_ms: ms(trace.prompt),
                language_ms: ms(trace.language),
                acoustic_ms: ms(trace.acoustic),
                decode_loop_ms: ms(trace.decode_loop),
                first_frame_ms: trace.first_frame.map(ms),
                codec_ms: ms(trace.codec),
                total_ms: ms(trace.total),
                output_wav,
            });
        }
    }

    let generation_ms = generations
        .iter()
        .map(|generation| generation.total_ms)
        .sum::<f64>();
    let voice_cache_hits = generations
        .iter()
        .filter(|generation| generation.voice_cache_hit)
        .count();
    let voice_cache_misses = generations.len() - voice_cache_hits;
    let report = PerfReport {
        model_dir: args.model_dir,
        device: args.device,
        dtype: format!("{:?}", args.dtype),
        text: args.text,
        voices: args.voices,
        warm_runs: args.warm_runs,
        max_frames: args.max_frames,
        flow_steps: args.flow_steps,
        seed: args.seed,
        use_kv_cache: args.use_kv_cache,
        output_dir: args
            .output_dir
            .as_ref()
            .map(|path| path.display().to_string()),
        module_loads: 1,
        load: LoadReport {
            model_ms: ms(load_trace.model_load),
            module_ms: ms(load_trace.module_load),
            total_ms: ms(load_trace.total),
        },
        totals: TotalsReport {
            generation_ms,
            voice_cache_hits,
            voice_cache_misses,
            cached_voices: runtime.cached_voice_count(),
        },
        generations,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human(&report);
    }

    Ok(())
}

fn parse_device(raw: &str) -> Result<PerfDevice, Box<dyn Error>> {
    match raw {
        "cpu" => Ok(PerfDevice::Cpu),
        "metal" => Ok(PerfDevice::Metal),
        _ => Err(format!("unsupported device {raw:?}; expected cpu or metal").into()),
    }
}

fn parse_dtype(raw: &str) -> Result<DType, Box<dyn Error>> {
    match raw {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        _ => Err(format!("unsupported dtype {raw:?}; expected f32, f16, or bf16").into()),
    }
}

fn parse_voices(raw: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let voices = raw
        .split(',')
        .map(str::trim)
        .filter(|voice| !voice.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if voices.is_empty() {
        Err("--voices must include at least one non-empty voice".into())
    } else {
        Ok(voices)
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn print_human(report: &PerfReport) {
    println!("voxtral_perf.model_dir={}", report.model_dir);
    println!("voxtral_perf.device={:?}", report.device);
    println!("voxtral_perf.dtype={}", report.dtype);
    println!("voxtral_perf.kv_cache={}", report.use_kv_cache);
    println!("voxtral_perf.module_loads={}", report.module_loads);
    println!("voxtral_perf.load.model_ms={:.1}", report.load.model_ms);
    println!("voxtral_perf.load.module_ms={:.1}", report.load.module_ms);
    println!("voxtral_perf.load.total_ms={:.1}", report.load.total_ms);
    for generation in &report.generations {
        println!(
            "voxtral_perf.generation.run={} voice={} kv_cache={} cache_hit={} frames={} total_ms={:.1} first_frame_ms={:.1} language_ms={:.1} acoustic_ms={:.1} codec_ms={:.1} output_wav={}",
            generation.run,
            generation.voice,
            generation.language_cache,
            generation.voice_cache_hit,
            generation.frames,
            generation.total_ms,
            generation.first_frame_ms.unwrap_or(0.0),
            generation.language_ms,
            generation.acoustic_ms,
            generation.codec_ms,
            generation.output_wav.as_deref().unwrap_or("")
        );
    }
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

fn write_wav_pcm16(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), Box<dyn Error>> {
    let data_bytes = samples.len() as u32 * 2;
    let riff_size = 36u32
        .checked_add(data_bytes)
        .ok_or("WAV output is too large")?;
    let mut file = fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&riff_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&(sample_rate * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_bytes.to_le_bytes())?;
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        file.write_all(&pcm.to_le_bytes())?;
    }
    Ok(())
}

fn print_help() {
    println!(
        "Usage: voxtral-perf --model-dir PATH [--voices casual_male,casual_female] [--text TEXT] [--warm-runs N] [--max-frames N] [--flow-steps N] [--kv-cache] [--output-dir DIR] [--device cpu|metal] [--dtype f32|f16|bf16] [--json]"
    );
}
