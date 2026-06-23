use std::f32::consts::TAU;

use candle_core::{DType, Device, Tensor};

use crate::{
    build_prompt_embeddings, build_prompt_token_ids, load_voice_embedding, Result, VoxtralError,
    VoxtralModel,
};

#[derive(Debug, Clone)]
pub struct VoxtralGenerationOptions {
    pub max_frames: usize,
    pub seed: u64,
    pub flow_steps: usize,
    pub cfg_alpha: f32,
}

#[derive(Debug, Clone)]
pub struct VoxtralGeneratedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub frames: usize,
    pub ended: bool,
}

impl Default for VoxtralGenerationOptions {
    fn default() -> Self {
        Self {
            max_frames: 256,
            seed: 0x5658_5452_414c,
            flow_steps: 7,
            cfg_alpha: 1.2,
        }
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
        let assets = self.assets().ok_or_else(|| {
            VoxtralError::InvalidCheckpoint("model was loaded without resolved assets".into())
        })?;
        let voice_path = assets.voice_embeddings.get(voice).ok_or_else(|| {
            VoxtralError::InvalidCheckpoint(format!("voice embedding {voice:?} was not resolved"))
        })?;
        let modules = self.load_inference_modules(dtype, device)?;

        let encoder = tokenizer.encoder()?;
        let text_token_ids = encoder.encode(text)?;
        let voice_embeddings = load_voice_embedding(voice_path, dtype, device)?;
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
            build_prompt_embeddings(&modules.embeddings, &prompt, &voice_embeddings, device)?;
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
        let timesteps = flow_timesteps(options.flow_steps)?;
        let mut code_frames = Vec::with_capacity(options.max_frames * config.num_codebooks());
        let mut ended = false;

        for frame_idx in 0..options.max_frames {
            let hidden = candle(modules.language.forward_causal(
                &decode_embeddings,
                0,
                config.rope_theta,
            ))?;
            let last_pos = candle(decode_embeddings.dim(1))? - 1;
            let last_hidden = candle(
                hidden
                    .narrow(1, last_pos, 1)
                    .and_then(|hidden| hidden.reshape((1, config.dim))),
            )?;
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
            let frame = candle(frame_codes.to_vec2::<u32>())?.remove(0);
            if frame[0] == 1 {
                ended = true;
                break;
            }
            code_frames.extend_from_slice(&frame);

            let next_embedding = candle(
                modules
                    .embeddings
                    .audio_codes_embedding(config, &frame_codes),
            )?;
            let next_embedding = candle(next_embedding.unsqueeze(1))?;
            decode_embeddings = candle(Tensor::cat(&[decode_embeddings, next_embedding], 1))?;
        }

        let frames = code_frames.len() / config.num_codebooks();
        if frames == 0 {
            return Err(VoxtralError::Unsupported(
                "generation produced no audio frames".into(),
            ));
        }
        let codes = candle(Tensor::from_vec(
            code_frames,
            (frames, config.num_codebooks()),
            device,
        ))?;
        let codes = candle(codes.transpose(0, 1))?;
        let codes = candle(codes.unsqueeze(0))?;
        let waveform = candle(modules.codec.decode_codes_to_waveform(&codes))?;
        let samples = candle(waveform.to_dtype(DType::F32))?
            .to_vec3::<f32>()
            .map_err(|e| VoxtralError::Candle(e.to_string()))?[0][0]
            .clone();

        Ok(VoxtralGeneratedAudio {
            samples,
            sample_rate: config.sample_rate(),
            frames,
            ended,
        })
    }
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
