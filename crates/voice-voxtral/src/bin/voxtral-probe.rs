use std::env;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear_no_bias, VarBuilder};
use voice_voxtral::{
    build_prompt_embeddings, build_prompt_token_ids, load_voice_embedding,
    VoxtralGenerationOptions, VoxtralModel, WeightComponent,
};

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    device: ProbeDevice,
    dtype: DType,
    load_norm: bool,
    linear_smoke: bool,
    load_modules: bool,
    acoustic_forward: bool,
    language_forward: bool,
    text: String,
    voice: String,
    output_wav: Option<PathBuf>,
    max_frames: usize,
    seed: u64,
    flow_steps: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeDevice {
    Cpu,
    Metal,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let device = args.device.load()?;

    println!("voxtral_probe.model_dir={}", args.model_dir.display());
    println!("voxtral_probe.device={:?}", args.device);
    println!("voxtral_probe.dtype={:?}", args.dtype);

    let model = VoxtralModel::load_from_dir(&args.model_dir)?;
    let config = model.config();
    let summary = model
        .checkpoint_summary()
        .ok_or("loaded model did not include checkpoint metadata")?;

    println!("config.model_type={}", config.model_type);
    println!("config.dim={}", config.dim);
    println!("config.n_layers={}", config.n_layers);
    println!("config.vocab_size={}", config.vocab_size);
    println!("config.sample_rate={}", config.sample_rate());
    println!("config.frame_rate={}", config.frame_rate());
    println!("config.num_codebooks={}", config.num_codebooks());
    println!(
        "config.acoustic_codebooks={}",
        config.multimodal.audio_model_args.n_acoustic_codebook
    );
    println!(
        "config.semantic_codebook_size={}",
        config.multimodal.audio_model_args.semantic_codebook_size
    );
    println!(
        "config.acoustic_codebook_size={}",
        config.multimodal.audio_model_args.acoustic_codebook_size
    );
    println!("checkpoint.file_len_bytes={}", summary.file_len);
    println!("checkpoint.data_len_bytes={}", summary.data_len);
    println!("checkpoint.header_len_bytes={}", summary.header_len);
    println!("checkpoint.tensor_count={}", summary.tensor_count);
    println!(
        "checkpoint.expected_tensor_count={}",
        summary.expected_tensor_count
    );
    println!(
        "checkpoint.bf16_data_gib={:.3}",
        summary.data_len as f64 / 1024_f64.powi(3)
    );
    println!(
        "checkpoint.f32_equivalent_gib={:.3}",
        summary.data_len as f64 * 2.0 / 1024_f64.powi(3)
    );
    for component in [
        WeightComponent::LanguageModel,
        WeightComponent::AcousticTransformer,
        WeightComponent::AudioTokenizer,
        WeightComponent::MultimodalEmbeddings,
        WeightComponent::FinalNorm,
        WeightComponent::Other,
    ] {
        let count = summary
            .component_counts
            .get(&component)
            .copied()
            .unwrap_or(0);
        println!("checkpoint.component.{component:?}={count}");
    }

    if let Some(tokenizer) = model.tokenizer() {
        println!(
            "tokenizer.output_audio_token={:?}",
            tokenizer.special_token_id("[OUTPUT_AUDIO]")
        );
        println!(
            "tokenizer.begin_audio_token={:?}",
            tokenizer.special_token_id("[BEGIN_AUDIO]")
        );
        println!(
            "tokenizer.casual_male_audio_tokens={:?}",
            tokenizer.voice_audio_tokens("casual_male")
        );
    }

    if args.load_norm {
        let norm = model.load_norm_weight(args.dtype, &device)?;
        println!("probe.load_norm.dims={:?}", norm.dims());
        println!("probe.load_norm.dtype={:?}", norm.dtype());
    }

    if args.linear_smoke {
        let audio_model = &config.multimodal.audio_model_args;
        let acoustic = &audio_model.acoustic_transformer_args;
        let vb = VarBuilder::zeros(args.dtype, &device);
        let projection = linear_no_bias(
            audio_model.n_acoustic_codebook,
            acoustic.dim,
            vb.pp("linear_smoke"),
        )?;
        let input_2d = Tensor::zeros((1, audio_model.n_acoustic_codebook), args.dtype, &device)?;
        let input_3d = input_2d.reshape((1, 1, audio_model.n_acoustic_codebook))?;
        let flat = input_3d.reshape((1, audio_model.n_acoustic_codebook))?;
        println!(
            "probe.linear_smoke.weight_dims={:?}",
            projection.weight().dims()
        );
        println!("probe.linear_smoke.input_2d_dims={:?}", input_2d.dims());
        println!("probe.linear_smoke.input_3d_dims={:?}", input_3d.dims());
        println!("probe.linear_smoke.flat_dims={:?}", flat.dims());
        println!(
            "probe.linear_smoke.out_2d_dims={:?}",
            projection.forward(&input_2d)?.dims()
        );
        println!(
            "probe.linear_smoke.out_flat_dims={:?}",
            projection.forward(&flat)?.dims()
        );
    }

    if args.load_modules
        || args.acoustic_forward
        || args.language_forward
        || args.output_wav.is_some()
    {
        let modules = model.load_inference_modules(args.dtype, &device)?;
        println!(
            "probe.load_modules.language_layers={}",
            modules.language.layers.len()
        );
        println!(
            "probe.load_modules.acoustic_layers={}",
            modules.acoustic.layers.len()
        );
        println!(
            "probe.load_modules.token_embedding_dims={:?}",
            modules.embeddings.tok_embeddings.embeddings().dims()
        );
        println!(
            "probe.load_modules.acoustic.input_projection={:?}",
            modules.acoustic.input_projection.weight().dims()
        );
        println!(
            "probe.load_modules.acoustic.time_projection={:?}",
            modules.acoustic.time_projection.weight().dims()
        );
        println!(
            "probe.load_modules.acoustic.llm_projection={:?}",
            modules.acoustic.llm_projection.weight().dims()
        );
        println!(
            "probe.load_modules.acoustic.acoustic_codebook_output={:?}",
            modules.acoustic.acoustic_codebook_output.weight().dims()
        );
        println!(
            "probe.load_modules.codec.semantic_embedding={:?}",
            modules.codec.codebook.semantic_embedding.dims()
        );
        println!(
            "probe.load_modules.codec.input_conv={:?}",
            modules.codec.input_conv.weight().dims()
        );
        println!(
            "probe.load_modules.codec.stages={}",
            modules.codec.stages.len()
        );
        for (stage_idx, stage) in modules.codec.stages.iter().enumerate() {
            println!(
                "probe.load_modules.codec.stage.{stage_idx}.layers={}",
                stage.layers.len()
            );
            println!(
                "probe.load_modules.codec.stage.{stage_idx}.window={}",
                stage.window_size
            );
            if let Some(upsample) = &stage.upsample {
                println!(
                    "probe.load_modules.codec.stage.{stage_idx}.upsample={:?}",
                    upsample.weight().dims()
                );
            }
        }
        println!(
            "probe.load_modules.codec.output_proj={:?}",
            modules.codec.output_proj.weight().dims()
        );

        if args.language_forward {
            let tokenizer = model
                .tokenizer()
                .ok_or("model did not include tekken.json")?;
            let encoder = tokenizer.encoder()?;
            let text_token_ids = encoder.encode(&args.text)?;
            let assets = model
                .assets()
                .ok_or("model did not include resolved assets")?;
            let voice_path = assets
                .voice_embeddings
                .get(&args.voice)
                .ok_or_else(|| format!("voice embedding {:?} was not resolved", args.voice))?;
            let voice_embeddings = load_voice_embedding(voice_path, args.dtype, &device)?;
            let voice_frames = voice_embeddings.dim(0)?;
            if let Some(expected_voice_frames) = tokenizer.voice_audio_tokens(&args.voice) {
                if expected_voice_frames != voice_frames {
                    return Err(format!(
                        "voice {:?} has {voice_frames} rows but tekken metadata expects {expected_voice_frames}",
                        args.voice
                    )
                    .into());
                }
            }
            let prompt = build_prompt_token_ids(config, tokenizer, voice_frames, &text_token_ids)?;
            let prompt_embeddings =
                build_prompt_embeddings(&modules.embeddings, &prompt, &voice_embeddings, &device)?;
            let audio_token_id = usize::try_from(config.multimodal.audio_model_args.audio_token_id)
                .map_err(|_| "audio_token_id must be non-negative")?;
            let audio_embedding = modules
                .embeddings
                .token_embeddings(&[audio_token_id], &device)?;
            let decode_embeddings = Tensor::cat(&[prompt_embeddings, audio_embedding], 1)?;
            let hidden =
                modules
                    .language
                    .forward_causal(&decode_embeddings, 0, config.rope_theta)?;
            let last_hidden = hidden
                .narrow(1, decode_embeddings.dim(1)? - 1, 1)?
                .reshape((1, config.dim))?;

            println!("probe.language_forward.voice={}", args.voice);
            println!("probe.language_forward.text={:?}", args.text);
            println!(
                "probe.language_forward.text_token_count={}",
                text_token_ids.len()
            );
            println!("probe.language_forward.voice_frames={voice_frames}");
            println!(
                "probe.language_forward.prompt_token_count={}",
                prompt.input_ids.len()
            );
            println!(
                "probe.language_forward.decode_input_dims={:?}",
                decode_embeddings.dims()
            );
            println!("probe.language_forward.hidden_dims={:?}", hidden.dims());
            println!(
                "probe.language_forward.last_hidden_dims={:?}",
                last_hidden.dims()
            );
            println!(
                "probe.language_forward.last_hidden_dtype={:?}",
                last_hidden.dtype()
            );

            let initial_noise = Tensor::zeros(
                (1, config.multimodal.audio_model_args.n_acoustic_codebook),
                args.dtype,
                &device,
            )?;
            let frame_codes = modules.acoustic.predict_frame_codes_from_noise(
                config,
                &last_hidden,
                &initial_noise,
                &[0.0, 1.0],
                1.2,
            )?;
            println!(
                "probe.language_forward.frame_code_dims={:?}",
                frame_codes.dims()
            );
            println!(
                "probe.language_forward.frame_code_dtype={:?}",
                frame_codes.dtype()
            );
            let codec_codes = frame_codes.reshape((1, config.num_codebooks(), 1))?;
            let waveform = modules.codec.decode_codes_to_waveform(&codec_codes)?;
            println!(
                "probe.language_forward.codec_waveform_dims={:?}",
                waveform.dims()
            );
            println!(
                "probe.language_forward.codec_waveform_dtype={:?}",
                waveform.dtype()
            );
        }

        if let Some(output_wav) = &args.output_wav {
            let audio = model.generate_audio(
                &args.text,
                &args.voice,
                args.dtype,
                &device,
                VoxtralGenerationOptions {
                    max_frames: args.max_frames,
                    seed: args.seed,
                    flow_steps: args.flow_steps,
                    ..Default::default()
                },
            )?;
            write_wav_pcm16(output_wav, &audio.samples, audio.sample_rate)?;

            println!("probe.generate_wav.frames={}", audio.frames);
            println!("probe.generate_wav.ended={}", audio.ended);
            println!("probe.generate_wav.samples={}", audio.samples.len());
            println!("probe.generate_wav.output={}", output_wav.display());
        }

        if args.acoustic_forward {
            let audio_model = &config.multimodal.audio_model_args;
            let acoustic = &audio_model.acoustic_transformer_args;
            let x_t = Tensor::zeros((1, audio_model.n_acoustic_codebook), args.dtype, &device)?;
            let llm_hidden = Tensor::zeros((1, acoustic.input_dim), args.dtype, &device)?;
            let timestep = Tensor::new(&[0.0f32], &device)?.to_dtype(args.dtype)?;
            let velocity = modules
                .acoustic
                .predict_velocity(&x_t, &llm_hidden, &timestep)?;
            println!("probe.acoustic_forward.velocity_dims={:?}", velocity.dims());
            let frame_codes = modules.acoustic.predict_frame_codes_from_noise(
                config,
                &llm_hidden,
                &x_t,
                &[0.0, 1.0],
                1.2,
            )?;
            println!(
                "probe.acoustic_forward.frame_code_dims={:?}",
                frame_codes.dims()
            );
            println!(
                "probe.acoustic_forward.frame_code_dtype={:?}",
                frame_codes.dtype()
            );
            let codec_codes = frame_codes.reshape((1, config.num_codebooks(), 1))?;
            let codec_latents = modules.codec.decode_code_embeddings(&codec_codes)?;
            println!(
                "probe.acoustic_forward.codec_latent_dims={:?}",
                codec_latents.dims()
            );
            println!(
                "probe.acoustic_forward.codec_latent_dtype={:?}",
                codec_latents.dtype()
            );
            let codec_input = modules.codec.forward_input_projection(&codec_latents)?;
            println!(
                "probe.acoustic_forward.codec_input_projection_dims={:?}",
                codec_input.dims()
            );
            println!(
                "probe.acoustic_forward.codec_input_projection_dtype={:?}",
                codec_input.dtype()
            );
            let codec_stage0 = modules.codec.forward_stage_transformers(0, &codec_input)?;
            println!(
                "probe.acoustic_forward.codec_stage0_transformer_dims={:?}",
                codec_stage0.dims()
            );
            println!(
                "probe.acoustic_forward.codec_stage0_transformer_dtype={:?}",
                codec_stage0.dtype()
            );
            if let Some(upsampled) = modules.codec.forward_stage_upsample(0, &codec_stage0)? {
                println!(
                    "probe.acoustic_forward.codec_stage0_upsample_dims={:?}",
                    upsampled.dims()
                );
                println!(
                    "probe.acoustic_forward.codec_stage0_upsample_dtype={:?}",
                    upsampled.dtype()
                );
            }
            let waveform = modules.codec.decode_codes_to_waveform(&codec_codes)?;
            println!(
                "probe.acoustic_forward.codec_waveform_dims={:?}",
                waveform.dims()
            );
            println!(
                "probe.acoustic_forward.codec_waveform_dtype={:?}",
                waveform.dtype()
            );
        }
    }

    Ok(())
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut model_dir = None;
        let mut device = ProbeDevice::Cpu;
        let mut dtype = DType::F32;
        let mut load_norm = false;
        let mut linear_smoke = false;
        let mut load_modules = false;
        let mut acoustic_forward = false;
        let mut language_forward = false;
        let mut text = "Voxtral support is running from native Rust.".to_string();
        let mut voice = "casual_male".to_string();
        let mut output_wav = None;
        let mut max_frames = 8;
        let mut seed = 0x5658_5452_414c;
        let mut flow_steps = 7;

        let mut raw = env::args().skip(1);
        while let Some(arg) = raw.next() {
            match arg.as_str() {
                "--model-dir" => {
                    model_dir = Some(PathBuf::from(
                        raw.next().ok_or("--model-dir requires a path")?,
                    ));
                }
                "--device" => {
                    device = parse_device(&raw.next().ok_or("--device requires cpu or metal")?)?;
                }
                "--dtype" => {
                    dtype = parse_dtype(&raw.next().ok_or("--dtype requires f32, f16, or bf16")?)?;
                }
                "--load-norm" => load_norm = true,
                "--linear-smoke" => linear_smoke = true,
                "--load-modules" => load_modules = true,
                "--acoustic-forward" => {
                    acoustic_forward = true;
                    load_modules = true;
                }
                "--language-forward" => {
                    language_forward = true;
                    load_modules = true;
                }
                "--text" => {
                    text = raw.next().ok_or("--text requires a value")?;
                }
                "--voice" => {
                    voice = raw.next().ok_or("--voice requires a value")?;
                }
                "--output-wav" => {
                    output_wav = Some(PathBuf::from(
                        raw.next().ok_or("--output-wav requires a path")?,
                    ));
                    load_modules = true;
                }
                "--max-frames" => {
                    max_frames = raw.next().ok_or("--max-frames requires a value")?.parse()?;
                }
                "--seed" => {
                    seed = raw.next().ok_or("--seed requires a value")?.parse()?;
                }
                "--flow-steps" => {
                    flow_steps = raw.next().ok_or("--flow-steps requires a value")?.parse()?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                unknown => return Err(format!("unknown argument {unknown:?}").into()),
            }
        }

        let model_dir = model_dir
            .or_else(|| env::var_os("VOXTRAL_LOCAL_DIR").map(PathBuf::from))
            .ok_or("pass --model-dir or set VOXTRAL_LOCAL_DIR")?;

        Ok(Self {
            model_dir,
            device,
            dtype,
            load_norm,
            linear_smoke,
            load_modules,
            acoustic_forward,
            language_forward,
            text,
            voice,
            output_wav,
            max_frames,
            seed,
            flow_steps,
        })
    }
}

impl ProbeDevice {
    fn load(self) -> Result<Device, Box<dyn Error>> {
        match self {
            ProbeDevice::Cpu => Ok(Device::Cpu),
            ProbeDevice::Metal => {
                #[cfg(target_os = "macos")]
                {
                    Device::new_metal(0).map_err(|e| e.into())
                }

                #[cfg(not(target_os = "macos"))]
                {
                    Err("Metal device is only available on macOS".into())
                }
            }
        }
    }
}

fn parse_device(raw: &str) -> Result<ProbeDevice, Box<dyn Error>> {
    match raw {
        "cpu" => Ok(ProbeDevice::Cpu),
        "metal" => Ok(ProbeDevice::Metal),
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

fn print_help() {
    println!(
        "Usage: voxtral-probe --model-dir PATH [--device cpu|metal] [--dtype f32|f16|bf16] [--load-norm] [--linear-smoke] [--load-modules] [--acoustic-forward] [--language-forward] [--voice VOICE] [--text TEXT] [--output-wav PATH] [--max-frames N] [--flow-steps N] [--seed N]"
    );
}

fn write_wav_pcm16(
    path: &PathBuf,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

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
        let sample = sample.clamp(-1.0, 1.0);
        let pcm = (sample * i16::MAX as f32) as i16;
        file.write_all(&pcm.to_le_bytes())?;
    }
    Ok(())
}
