use std::collections::BTreeMap;
use std::f32::consts::TAU;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};

use crate::{
    build_prompt_embeddings, build_prompt_token_ids, load_voice_embedding, plan_codec_chunk,
    Result, VoxtralCodecChunk, VoxtralError, VoxtralInferenceModules, VoxtralModel,
    VoxtralStreamingConfig, VoxtralTokenizerMetadata,
};

#[derive(Debug, Clone)]
pub struct VoxtralGenerationOptions {
    pub max_frames: usize,
    pub seed: u64,
    pub flow_steps: usize,
    pub cfg_alpha: f32,
    pub use_kv_cache: bool,
    pub synchronize_trace: bool,
    pub trace_semantic_scores: bool,
    pub eos_guard_frames: usize,
    pub eos_guard_max_rank: usize,
    pub eos_guard_max_margin: f32,
}

#[derive(Debug, Clone)]
pub struct VoxtralGeneratedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub frames: usize,
    pub ended: bool,
}

#[derive(Debug, Clone)]
pub struct VoxtralGeneratedAudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub chunk_index: usize,
    pub context_frames: usize,
    pub chunk_frames: usize,
    pub generated_frames: usize,
    pub finished: bool,
}

#[derive(Debug, Clone, Default)]
pub struct VoxtralGenerationTrace {
    pub language_cache: bool,
    pub voice_cache_hit: bool,
    pub semantic_codes: Vec<u32>,
    pub eos_frame: Option<usize>,
    pub semantic_eos_ranks: Vec<usize>,
    pub semantic_eos_margins: Vec<f32>,
    pub voice_load: Duration,
    pub prompt: Duration,
    pub language: Duration,
    pub acoustic: Duration,
    pub decode_loop: Duration,
    pub codec: Duration,
    pub codec_chunks: usize,
    pub first_frame: Option<Duration>,
    pub first_audio_chunk: Option<Duration>,
    pub total: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct VoxtralRuntimeLoadTrace {
    pub model_load: Duration,
    pub module_load: Duration,
    pub total: Duration,
}

pub struct VoxtralTtsRuntime {
    model: VoxtralModel,
    modules: VoxtralInferenceModules,
    dtype: DType,
    device: Device,
    voice_cache: BTreeMap<String, Tensor>,
}

impl Default for VoxtralGenerationOptions {
    fn default() -> Self {
        Self {
            max_frames: 256,
            seed: 0x5658_5452_414c,
            flow_steps: 7,
            cfg_alpha: 1.2,
            use_kv_cache: false,
            synchronize_trace: false,
            trace_semantic_scores: false,
            eos_guard_frames: 0,
            eos_guard_max_rank: 2,
            eos_guard_max_margin: 0.5,
        }
    }
}

impl VoxtralTtsRuntime {
    pub fn load(path_or_repo: &str, dtype: DType, device: Device) -> Result<Self> {
        Ok(Self::load_with_trace(path_or_repo, dtype, device)?.0)
    }

    pub fn load_with_trace(
        path_or_repo: &str,
        dtype: DType,
        device: Device,
    ) -> Result<(Self, VoxtralRuntimeLoadTrace)> {
        let total_start = Instant::now();
        let model_start = Instant::now();
        let model = VoxtralModel::load(path_or_repo)?;
        let model_load = model_start.elapsed();
        Self::from_model_with_trace(model, dtype, device, total_start, model_load)
    }

    pub fn load_from_dir(
        dir: impl AsRef<std::path::Path>,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        Ok(Self::load_from_dir_with_trace(dir, dtype, device)?.0)
    }

    pub fn load_from_dir_with_trace(
        dir: impl AsRef<std::path::Path>,
        dtype: DType,
        device: Device,
    ) -> Result<(Self, VoxtralRuntimeLoadTrace)> {
        let total_start = Instant::now();
        let model_start = Instant::now();
        let model = VoxtralModel::load_from_dir(dir)?;
        let model_load = model_start.elapsed();
        Self::from_model_with_trace(model, dtype, device, total_start, model_load)
    }

    pub fn load_default(path_or_repo: &str) -> Result<Self> {
        Ok(Self::load_default_with_trace(path_or_repo)?.0)
    }

    pub fn load_default_with_trace(path_or_repo: &str) -> Result<(Self, VoxtralRuntimeLoadTrace)> {
        #[cfg(target_os = "macos")]
        {
            let device = Device::new_metal(0).map_err(|e| VoxtralError::Candle(e.to_string()))?;
            Self::load_with_trace(path_or_repo, DType::F16, device)
        }

        #[cfg(not(target_os = "macos"))]
        {
            Self::load_with_trace(path_or_repo, DType::F32, Device::Cpu)
        }
    }

    fn from_model_with_trace(
        model: VoxtralModel,
        dtype: DType,
        device: Device,
        total_start: Instant,
        model_load: Duration,
    ) -> Result<(Self, VoxtralRuntimeLoadTrace)> {
        let module_start = Instant::now();
        let modules = model.load_inference_modules(dtype, &device)?;
        let module_load = module_start.elapsed();
        let trace = VoxtralRuntimeLoadTrace {
            model_load,
            module_load,
            total: total_start.elapsed(),
        };

        Ok((
            Self {
                model,
                modules,
                dtype,
                device,
                voice_cache: BTreeMap::new(),
            },
            trace,
        ))
    }

    pub fn config(&self) -> &crate::VoxtralConfig {
        self.model.config()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn cached_voice_count(&self) -> usize {
        self.voice_cache.len()
    }

    pub fn preload_voice(&mut self, voice: &str) -> Result<()> {
        self.voice_embedding(voice).map(|_| ())
    }

    pub fn generate_audio(
        &mut self,
        text: &str,
        voice: &str,
        options: VoxtralGenerationOptions,
    ) -> Result<VoxtralGeneratedAudio> {
        Ok(self.generate_audio_with_trace(text, voice, options)?.0)
    }

    pub fn generate_audio_with_trace(
        &mut self,
        text: &str,
        voice: &str,
        options: VoxtralGenerationOptions,
    ) -> Result<(VoxtralGeneratedAudio, VoxtralGenerationTrace)> {
        let total_start = Instant::now();
        let (voice_embeddings, voice_cache_hit, voice_load) = self.voice_embedding(voice)?;
        let (audio, mut trace) = generate_audio_inner(GenerateAudioInner {
            config: self.model.config(),
            tokenizer: self
                .model
                .tokenizer()
                .ok_or_else(|| VoxtralError::InvalidTokenizer("missing tekken.json".into()))?,
            modules: &self.modules,
            text,
            voice,
            voice_embeddings: &voice_embeddings,
            device: &self.device,
            options,
        })?;
        trace.voice_cache_hit = voice_cache_hit;
        trace.voice_load = voice_load;
        trace.total = total_start.elapsed();
        Ok((audio, trace))
    }

    pub fn generate_audio_streaming_with_trace<F>(
        &mut self,
        text: &str,
        voice: &str,
        options: VoxtralGenerationOptions,
        streaming: VoxtralStreamingConfig,
        on_chunk: F,
    ) -> Result<(VoxtralGeneratedAudio, VoxtralGenerationTrace)>
    where
        F: FnMut(&VoxtralGeneratedAudioChunk) -> Result<()>,
    {
        let total_start = Instant::now();
        let (voice_embeddings, voice_cache_hit, voice_load) = self.voice_embedding(voice)?;
        let (audio, mut trace) = generate_audio_inner_streaming(
            GenerateAudioInner {
                config: self.model.config(),
                tokenizer: self
                    .model
                    .tokenizer()
                    .ok_or_else(|| VoxtralError::InvalidTokenizer("missing tekken.json".into()))?,
                modules: &self.modules,
                text,
                voice,
                voice_embeddings: &voice_embeddings,
                device: &self.device,
                options,
            },
            streaming,
            on_chunk,
        )?;
        trace.voice_cache_hit = voice_cache_hit;
        trace.voice_load = voice_load;
        trace.total = total_start.elapsed();
        Ok((audio, trace))
    }

    fn voice_embedding(&mut self, voice: &str) -> Result<(Tensor, bool, Duration)> {
        if let Some(embedding) = self.voice_cache.get(voice) {
            return Ok((embedding.clone(), true, Duration::ZERO));
        }

        let start = Instant::now();
        let voice_path = self.model.resolve_voice_embedding_path(voice)?;
        let embedding = load_voice_embedding(&voice_path, self.dtype, &self.device)?;
        let elapsed = start.elapsed();
        self.voice_cache
            .insert(voice.to_string(), embedding.clone());
        Ok((embedding, false, elapsed))
    }
}

impl VoxtralModel {
    pub fn generate_audio_default(
        &self,
        text: &str,
        voice: &str,
        options: VoxtralGenerationOptions,
    ) -> Result<VoxtralGeneratedAudio> {
        #[cfg(target_os = "macos")]
        {
            let device = Device::new_metal(0).map_err(|e| VoxtralError::Candle(e.to_string()))?;
            self.generate_audio(text, voice, DType::F16, &device, options)
        }

        #[cfg(not(target_os = "macos"))]
        {
            self.generate_audio(text, voice, DType::F32, &Device::Cpu, options)
        }
    }

    pub fn generate_audio(
        &self,
        text: &str,
        voice: &str,
        dtype: DType,
        device: &Device,
        options: VoxtralGenerationOptions,
    ) -> Result<VoxtralGeneratedAudio> {
        let config = self.config();
        let tokenizer = self
            .tokenizer()
            .ok_or_else(|| VoxtralError::InvalidTokenizer("missing tekken.json".into()))?;
        let modules = self.load_inference_modules(dtype, device)?;
        let voice_path = self.resolve_voice_embedding_path(voice)?;
        let voice_embeddings = load_voice_embedding(voice_path, dtype, device)?;
        Ok(generate_audio_inner(GenerateAudioInner {
            config,
            tokenizer,
            modules: &modules,
            text,
            voice,
            voice_embeddings: &voice_embeddings,
            device,
            options,
        })?
        .0)
    }
}

struct GenerateAudioInner<'a> {
    config: &'a crate::VoxtralConfig,
    tokenizer: &'a VoxtralTokenizerMetadata,
    modules: &'a VoxtralInferenceModules,
    text: &'a str,
    voice: &'a str,
    voice_embeddings: &'a Tensor,
    device: &'a Device,
    options: VoxtralGenerationOptions,
}

fn generate_audio_inner(
    request: GenerateAudioInner<'_>,
) -> Result<(VoxtralGeneratedAudio, VoxtralGenerationTrace)> {
    let GenerateAudioInner {
        config,
        tokenizer,
        modules,
        text,
        voice,
        voice_embeddings,
        device,
        options,
    } = request;
    let total_start = Instant::now();
    let mut trace = VoxtralGenerationTrace::default();
    let dtype = voice_embeddings.dtype();

    let prompt_start = Instant::now();
    let encoder = tokenizer.encoder()?;
    let text_token_ids = encoder.encode(text)?;
    let voice_frames = candle(voice_embeddings.dim(0))?;
    if let Some(expected_voice_frames) = tokenizer.voice_audio_tokens(voice) {
        if expected_voice_frames != voice_frames {
            return Err(VoxtralError::InvalidCheckpoint(format!(
                "voice {voice:?} has {voice_frames} rows but tekken metadata expects {expected_voice_frames}"
            )));
        }
    }

    let prompt = build_prompt_token_ids(config, tokenizer, voice_frames, &text_token_ids)?;
    let prompt_embeddings =
        build_prompt_embeddings(&modules.embeddings, &prompt, voice_embeddings, device)?;
    let audio_token_id = usize::try_from(config.multimodal.audio_model_args.audio_token_id)
        .map_err(|_| {
            VoxtralError::InvalidConfig(format!(
                "audio_token_id must be non-negative, got {}",
                config.multimodal.audio_model_args.audio_token_id
            ))
        })?;
    let audio_embedding = candle(
        modules
            .embeddings
            .token_embeddings(&[audio_token_id], device),
    )?;
    let mut decode_embeddings = candle(Tensor::cat(&[prompt_embeddings, audio_embedding], 1))?;
    let mut decode_input = decode_embeddings.clone();
    let mut language_cache = options.use_kv_cache.then(|| modules.language.new_cache());
    let timesteps = flow_timesteps(options.flow_steps)?;
    trace.language_cache = options.use_kv_cache;
    trace.prompt = prompt_start.elapsed();

    let loop_start = Instant::now();
    let mut code_frames = Vec::with_capacity(options.max_frames * config.num_codebooks());
    let mut ended = false;
    let mut eos_guard_active = false;
    let max_decode_frames = options.max_frames.saturating_add(options.eos_guard_frames);

    for frame_idx in 0..max_decode_frames {
        if frame_idx >= options.max_frames && !eos_guard_active {
            break;
        }
        eos_guard_active = false;

        let language_start = Instant::now();
        let hidden = if let Some(cache) = language_cache.as_mut() {
            let start_pos = cache.len();
            candle(modules.language.forward_causal_cached(
                &decode_input,
                start_pos,
                config.rope_theta,
                cache,
            ))?
        } else {
            candle(
                modules
                    .language
                    .forward_causal(&decode_embeddings, 0, config.rope_theta),
            )?
        };
        let last_pos = candle(hidden.dim(1))? - 1;
        let last_hidden = candle(
            hidden
                .narrow(1, last_pos, 1)
                .and_then(|hidden| hidden.reshape((1, config.dim))),
        )?;
        sync_trace(device, options.synchronize_trace)?;
        trace.language += language_start.elapsed();

        let acoustic_start = Instant::now();
        let initial_noise = deterministic_noise(
            options.seed,
            frame_idx,
            config.multimodal.audio_model_args.n_acoustic_codebook,
            dtype,
            device,
        )?;
        let frame_codes = candle(modules.acoustic.predict_frame_codes_from_noise(
            config,
            &last_hidden,
            &initial_noise,
            &timesteps,
            options.cfg_alpha,
        ))?;
        sync_trace(device, options.synchronize_trace)?;
        trace.acoustic += acoustic_start.elapsed();
        trace
            .first_frame
            .get_or_insert_with(|| loop_start.elapsed());

        let frame = candle(frame_codes.to_vec2::<u32>())?.remove(0);
        let semantic_code = frame[0];
        let should_check_guard =
            options.eos_guard_frames > 0 && frame_idx + 1 >= options.max_frames;
        let semantic_score = if options.trace_semantic_scores || should_check_guard {
            Some(semantic_eos_rank_and_margin(
                config,
                modules,
                &last_hidden,
                semantic_code,
            )?)
        } else {
            None
        };
        if let Some((rank, margin)) = semantic_score {
            if options.trace_semantic_scores {
                trace.semantic_eos_ranks.push(rank);
                trace.semantic_eos_margins.push(margin);
            }
            eos_guard_active = should_extend_eos_guard(&options, should_check_guard, rank, margin);
        }
        if semantic_code == 1 {
            trace.eos_frame = Some(frame_idx);
            ended = true;
            break;
        }
        trace.semantic_codes.push(semantic_code);
        code_frames.extend_from_slice(&frame);

        let next_embedding = candle(
            modules
                .embeddings
                .audio_codes_embedding(config, &frame_codes),
        )?;
        let next_embedding = candle(next_embedding.unsqueeze(1))?;
        if language_cache.is_some() {
            decode_input = next_embedding;
        } else {
            decode_embeddings = candle(Tensor::cat(&[decode_embeddings, next_embedding], 1))?;
        }
    }
    trace.decode_loop = loop_start.elapsed();

    let frames = code_frames.len() / config.num_codebooks();
    if frames == 0 {
        return Err(VoxtralError::Unsupported(
            "generation produced no audio frames".into(),
        ));
    }

    let codec_start = Instant::now();
    let codes = candle(Tensor::from_vec(
        code_frames,
        (frames, config.num_codebooks()),
        device,
    ))?;
    let codes = candle(codes.transpose(0, 1))?;
    let codes = candle(codes.unsqueeze(0))?;
    let waveform = candle(modules.codec.decode_codes_to_waveform(&codes))?;
    sync_trace(device, options.synchronize_trace)?;
    let samples = candle(waveform.to_dtype(DType::F32))?
        .to_vec3::<f32>()
        .map_err(|e| VoxtralError::Candle(e.to_string()))?[0][0]
        .clone();
    trace.codec = codec_start.elapsed();
    trace.total = total_start.elapsed();

    Ok((
        VoxtralGeneratedAudio {
            samples,
            sample_rate: config.sample_rate(),
            frames,
            ended,
        },
        trace,
    ))
}

fn generate_audio_inner_streaming<F>(
    request: GenerateAudioInner<'_>,
    streaming: VoxtralStreamingConfig,
    mut on_chunk: F,
) -> Result<(VoxtralGeneratedAudio, VoxtralGenerationTrace)>
where
    F: FnMut(&VoxtralGeneratedAudioChunk) -> Result<()>,
{
    let GenerateAudioInner {
        config,
        tokenizer,
        modules,
        text,
        voice,
        voice_embeddings,
        device,
        options,
    } = request;
    let total_start = Instant::now();
    let mut trace = VoxtralGenerationTrace::default();
    let dtype = voice_embeddings.dtype();

    let prompt_start = Instant::now();
    let encoder = tokenizer.encoder()?;
    let text_token_ids = encoder.encode(text)?;
    let voice_frames = candle(voice_embeddings.dim(0))?;
    if let Some(expected_voice_frames) = tokenizer.voice_audio_tokens(voice) {
        if expected_voice_frames != voice_frames {
            return Err(VoxtralError::InvalidCheckpoint(format!(
                "voice {voice:?} has {voice_frames} rows but tekken metadata expects {expected_voice_frames}"
            )));
        }
    }

    let prompt = build_prompt_token_ids(config, tokenizer, voice_frames, &text_token_ids)?;
    let prompt_embeddings =
        build_prompt_embeddings(&modules.embeddings, &prompt, voice_embeddings, device)?;
    let audio_token_id = usize::try_from(config.multimodal.audio_model_args.audio_token_id)
        .map_err(|_| {
            VoxtralError::InvalidConfig(format!(
                "audio_token_id must be non-negative, got {}",
                config.multimodal.audio_model_args.audio_token_id
            ))
        })?;
    let audio_embedding = candle(
        modules
            .embeddings
            .token_embeddings(&[audio_token_id], device),
    )?;
    let mut decode_embeddings = candle(Tensor::cat(&[prompt_embeddings, audio_embedding], 1))?;
    let mut decode_input = decode_embeddings.clone();
    let mut language_cache = options.use_kv_cache.then(|| modules.language.new_cache());
    let timesteps = flow_timesteps(options.flow_steps)?;
    trace.language_cache = options.use_kv_cache;
    trace.prompt = prompt_start.elapsed();

    let loop_start = Instant::now();
    let mut frame_history: Vec<Vec<u32>> = Vec::with_capacity(options.max_frames);
    let mut emitted_frames = 0usize;
    let mut emitted_samples = Vec::new();
    let mut ended = false;
    let mut eos_guard_active = false;
    let max_decode_frames = options.max_frames.saturating_add(options.eos_guard_frames);

    for frame_idx in 0..max_decode_frames {
        if frame_idx >= options.max_frames && !eos_guard_active {
            break;
        }
        eos_guard_active = false;

        let language_start = Instant::now();
        let hidden = if let Some(cache) = language_cache.as_mut() {
            let start_pos = cache.len();
            candle(modules.language.forward_causal_cached(
                &decode_input,
                start_pos,
                config.rope_theta,
                cache,
            ))?
        } else {
            candle(
                modules
                    .language
                    .forward_causal(&decode_embeddings, 0, config.rope_theta),
            )?
        };
        let last_pos = candle(hidden.dim(1))? - 1;
        let last_hidden = candle(
            hidden
                .narrow(1, last_pos, 1)
                .and_then(|hidden| hidden.reshape((1, config.dim))),
        )?;
        sync_trace(device, options.synchronize_trace)?;
        trace.language += language_start.elapsed();

        let acoustic_start = Instant::now();
        let initial_noise = deterministic_noise(
            options.seed,
            frame_idx,
            config.multimodal.audio_model_args.n_acoustic_codebook,
            dtype,
            device,
        )?;
        let frame_codes = candle(modules.acoustic.predict_frame_codes_from_noise(
            config,
            &last_hidden,
            &initial_noise,
            &timesteps,
            options.cfg_alpha,
        ))?;
        sync_trace(device, options.synchronize_trace)?;
        trace.acoustic += acoustic_start.elapsed();
        trace
            .first_frame
            .get_or_insert_with(|| loop_start.elapsed());

        let frame = candle(frame_codes.to_vec2::<u32>())?.remove(0);
        let semantic_code = frame[0];
        let should_check_guard =
            options.eos_guard_frames > 0 && frame_idx + 1 >= options.max_frames;
        let semantic_score = if options.trace_semantic_scores || should_check_guard {
            Some(semantic_eos_rank_and_margin(
                config,
                modules,
                &last_hidden,
                semantic_code,
            )?)
        } else {
            None
        };
        if let Some((rank, margin)) = semantic_score {
            if options.trace_semantic_scores {
                trace.semantic_eos_ranks.push(rank);
                trace.semantic_eos_margins.push(margin);
            }
            eos_guard_active = should_extend_eos_guard(&options, should_check_guard, rank, margin);
        }
        if semantic_code == 1 {
            trace.eos_frame = Some(frame_idx);
            ended = true;
            break;
        }
        trace.semantic_codes.push(semantic_code);
        frame_history.push(frame.clone());

        maybe_emit_streaming_chunk(
            config,
            modules,
            device,
            streaming,
            false,
            &frame_history,
            &mut emitted_frames,
            &mut emitted_samples,
            &mut trace,
            options.synchronize_trace,
            total_start,
            &mut on_chunk,
        )?;

        let next_embedding = candle(
            modules
                .embeddings
                .audio_codes_embedding(config, &frame_codes),
        )?;
        let next_embedding = candle(next_embedding.unsqueeze(1))?;
        if language_cache.is_some() {
            decode_input = next_embedding;
        } else {
            decode_embeddings = candle(Tensor::cat(&[decode_embeddings, next_embedding], 1))?;
        }
    }
    trace.decode_loop = loop_start.elapsed();

    if frame_history.is_empty() {
        return Err(VoxtralError::Unsupported(
            "generation produced no audio frames".into(),
        ));
    }

    maybe_emit_streaming_chunk(
        config,
        modules,
        device,
        streaming,
        true,
        &frame_history,
        &mut emitted_frames,
        &mut emitted_samples,
        &mut trace,
        options.synchronize_trace,
        total_start,
        &mut on_chunk,
    )?;

    trace.total = total_start.elapsed();

    Ok((
        VoxtralGeneratedAudio {
            samples: emitted_samples,
            sample_rate: config.sample_rate(),
            frames: emitted_frames,
            ended,
        },
        trace,
    ))
}

#[allow(clippy::too_many_arguments)]
fn maybe_emit_streaming_chunk<F>(
    config: &crate::VoxtralConfig,
    modules: &VoxtralInferenceModules,
    device: &Device,
    streaming: VoxtralStreamingConfig,
    finished: bool,
    frame_history: &[Vec<u32>],
    emitted_frames: &mut usize,
    emitted_samples: &mut Vec<f32>,
    trace: &mut VoxtralGenerationTrace,
    synchronize_trace: bool,
    total_start: Instant,
    on_chunk: &mut F,
) -> Result<()>
where
    F: FnMut(&VoxtralGeneratedAudioChunk) -> Result<()>,
{
    let Some(chunk) = plan_codec_chunk(frame_history, *emitted_frames, streaming, finished)? else {
        return Ok(());
    };
    if chunk.chunk_frames == 0 || *emitted_frames >= frame_history.len() {
        return Ok(());
    }
    let new_frames = frame_history.len() - *emitted_frames;
    if chunk.chunk_frames != new_frames {
        return Err(VoxtralError::Unsupported(format!(
            "streaming codec chunk mismatch: planned {} new frames but {} remain",
            chunk.chunk_frames, new_frames
        )));
    }

    let codec_start = Instant::now();
    let samples = decode_codec_chunk_samples(config, modules, device, &chunk)?;
    sync_trace(device, synchronize_trace)?;
    trace.codec += codec_start.elapsed();
    trace.codec_chunks += 1;
    trace.first_audio_chunk.get_or_insert(total_start.elapsed());

    let audio_chunk = VoxtralGeneratedAudioChunk {
        samples,
        sample_rate: config.sample_rate(),
        chunk_index: trace.codec_chunks - 1,
        context_frames: chunk.context_frames,
        chunk_frames: chunk.chunk_frames,
        generated_frames: frame_history.len(),
        finished: chunk.finished,
    };
    on_chunk(&audio_chunk)?;
    emitted_samples.extend_from_slice(&audio_chunk.samples);
    *emitted_frames += audio_chunk.chunk_frames;
    Ok(())
}

fn decode_codec_chunk_samples(
    config: &crate::VoxtralConfig,
    modules: &VoxtralInferenceModules,
    device: &Device,
    chunk: &VoxtralCodecChunk,
) -> Result<Vec<f32>> {
    let num_codebooks = config.num_codebooks();
    if chunk.frames.is_empty() {
        return Ok(Vec::new());
    }

    let mut code_frames = Vec::with_capacity(chunk.frames.len() * num_codebooks);
    for frame in &chunk.frames {
        if frame.len() != num_codebooks {
            return Err(VoxtralError::InvalidConfig(format!(
                "codec frame has {} codebooks but config expects {num_codebooks}",
                frame.len()
            )));
        }
        code_frames.extend_from_slice(frame);
    }

    let codes = candle(Tensor::from_vec(
        code_frames,
        (chunk.frames.len(), num_codebooks),
        device,
    ))?;
    let codes = candle(codes.transpose(0, 1))?;
    let codes = candle(codes.unsqueeze(0))?;
    let waveform = candle(modules.codec.decode_codes_to_waveform(&codes))?;
    let samples = candle(waveform.to_dtype(DType::F32))?
        .to_vec3::<f32>()
        .map_err(|e| VoxtralError::Candle(e.to_string()))?[0][0]
        .clone();

    if samples.is_empty() {
        return Ok(samples);
    }
    if samples.len() % chunk.frames.len() != 0 {
        return Err(VoxtralError::Unsupported(format!(
            "codec chunk produced {} samples for {} frames",
            samples.len(),
            chunk.frames.len()
        )));
    }
    let samples_per_frame = samples.len() / chunk.frames.len();
    let start = chunk.context_frames * samples_per_frame;
    let len = chunk.chunk_frames * samples_per_frame;
    let end = start + len;
    if end > samples.len() {
        return Err(VoxtralError::Unsupported(format!(
            "codec chunk sample window {start}..{end} exceeds {} samples",
            samples.len()
        )));
    }
    Ok(samples[start..end].to_vec())
}

fn semantic_eos_rank_and_margin(
    config: &crate::VoxtralConfig,
    modules: &VoxtralInferenceModules,
    llm_hidden: &Tensor,
    selected_code: u32,
) -> Result<(usize, f32)> {
    let logits = candle(modules.acoustic.semantic_logits(config, llm_hidden))?;
    let row = candle(logits.to_dtype(DType::F32))?
        .to_vec2::<f32>()
        .map_err(|e| VoxtralError::Candle(e.to_string()))?
        .remove(0);
    let eos_logit = *row
        .get(1)
        .ok_or_else(|| VoxtralError::InvalidConfig("semantic logits missing EOS index".into()))?;
    let selected_idx = usize::try_from(selected_code).map_err(|_| {
        VoxtralError::InvalidConfig(format!("semantic code {selected_code} does not fit usize"))
    })?;
    let selected_logit = *row.get(selected_idx).ok_or_else(|| {
        VoxtralError::InvalidConfig(format!(
            "selected semantic code {selected_code} exceeds semantic logits length {}",
            row.len()
        ))
    })?;
    let rank = 1 + row.iter().filter(|&&logit| logit > eos_logit).count();
    Ok((rank, selected_logit - eos_logit))
}

fn should_extend_eos_guard(
    options: &VoxtralGenerationOptions,
    should_check_guard: bool,
    eos_rank: usize,
    eos_margin: f32,
) -> bool {
    should_check_guard
        && options.eos_guard_frames > 0
        && eos_rank <= options.eos_guard_max_rank
        && eos_margin <= options.eos_guard_max_margin
}

fn flow_timesteps(steps: usize) -> Result<Vec<f32>> {
    if steps == 0 {
        return Err(VoxtralError::InvalidConfig(
            "flow_steps must be greater than zero".into(),
        ));
    }
    Ok((0..=steps).map(|step| step as f32 / steps as f32).collect())
}

fn deterministic_noise(
    seed: u64,
    frame_idx: usize,
    len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let frame_seed = (frame_idx as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut rng = XorShift64::new(seed ^ frame_seed);
    let mut values = Vec::with_capacity(len);
    while values.len() < len {
        let u1 = rng.next_f32().max(f32::MIN_POSITIVE);
        let u2 = rng.next_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        values.push(radius * (TAU * u2).cos());
        if values.len() < len {
            values.push(radius * (TAU * u2).sin());
        }
    }
    let noise = candle(Tensor::from_vec(values, (1, len), device))?;
    candle(noise.to_dtype(dtype))
}

fn sync_trace(device: &Device, synchronize_trace: bool) -> Result<()> {
    if synchronize_trace {
        candle(device.synchronize())?;
    }
    Ok(())
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x1234_5678_abcd_ef01
            } else {
                seed
            },
        }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        ((x >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn candle<T>(result: candle_core::Result<T>) -> Result<T> {
    result.map_err(|e| VoxtralError::Candle(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{should_extend_eos_guard, VoxtralGenerationOptions};

    #[test]
    fn eos_guard_extends_only_near_cap_when_eos_is_close() {
        let options = VoxtralGenerationOptions {
            eos_guard_frames: 8,
            eos_guard_max_rank: 2,
            eos_guard_max_margin: 0.5,
            ..Default::default()
        };

        assert!(should_extend_eos_guard(&options, true, 2, 0.25));
        assert!(!should_extend_eos_guard(&options, false, 2, 0.25));
        assert!(!should_extend_eos_guard(&options, true, 3, 0.25));
        assert!(!should_extend_eos_guard(&options, true, 2, 0.75));
        assert!(!should_extend_eos_guard(
            &VoxtralGenerationOptions::default(),
            true,
            1,
            0.0
        ));
    }
}
