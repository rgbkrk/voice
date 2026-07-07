use std::collections::VecDeque;

use crate::{
    build_realtime_streaming_prompt_with_left_pad, realtime_num_delay_tokens,
    realtime_raw_audio_length_per_token, Result, VoxtralError, VoxtralRealtimeConfig,
    VoxtralTokenizerMetadata, REALTIME_DEFAULT_OFFLINE_BUFFER_TOKENS,
};

/// Model-native scheduling parameters for Voxtral Realtime STT.
///
/// This mirrors the upstream realtime buffer contract: a first prompt window
/// consumes the streaming prefix, then every generated text token is fed back
/// while the audio side advances by one raw-audio token.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxtralRealtimeStreamConfig {
    pub sample_rate: u32,
    pub raw_audio_length_per_token: usize,
    pub look_ahead_samples: usize,
    pub look_back_samples: usize,
    pub left_pad_tokens: usize,
    pub delay_tokens: usize,
    pub right_pad_tokens: usize,
    pub prompt_token_ids: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoxtralRealtimeStreamWindow {
    pub sequence: usize,
    pub input_token_ids: Vec<usize>,
    pub audio_samples: Vec<f32>,
    pub frame_start_sample: usize,
    pub frame_end_sample: usize,
    pub stride_start_sample: usize,
    pub stride_end_sample: usize,
    pub input_token_start: usize,
    pub input_token_end: usize,
    pub is_initial: bool,
}

#[derive(Debug, Clone)]
pub struct VoxtralRealtimeStreamBuffer {
    config: VoxtralRealtimeStreamConfig,
    samples: Vec<f32>,
    token_queue: VecDeque<usize>,
    stride_start_sample: usize,
    stride_end_sample: usize,
    input_token_position: usize,
    sequence: usize,
    finished: bool,
}

impl VoxtralRealtimeStreamConfig {
    pub fn from_metadata(
        config: &VoxtralRealtimeConfig,
        tokenizer: &VoxtralTokenizerMetadata,
    ) -> Result<Self> {
        let delay_ms = tokenizer.audio.transcription_delay_ms.ok_or_else(|| {
            VoxtralError::InvalidTokenizer("missing realtime transcription_delay_ms".into())
        })?;
        let delay_tokens = realtime_num_delay_tokens(config, delay_ms)?;
        Self::from_metadata_with_delay_tokens(config, tokenizer, delay_tokens)
    }

    pub fn from_metadata_with_delay_tokens(
        config: &VoxtralRealtimeConfig,
        tokenizer: &VoxtralTokenizerMetadata,
        delay_tokens: usize,
    ) -> Result<Self> {
        let raw_audio_length_per_token = realtime_raw_audio_length_per_token(config)?;
        let sample_rate = config.sample_rate();
        if tokenizer.audio.sampling_rate != sample_rate {
            return Err(VoxtralError::InvalidTokenizer(format!(
                "tokenizer sampling_rate={} but realtime config sample_rate={sample_rate}",
                tokenizer.audio.sampling_rate
            )));
        }
        if (tokenizer.audio.frame_rate - config.frame_rate()).abs() > f64::EPSILON {
            return Err(VoxtralError::InvalidTokenizer(format!(
                "tokenizer frame_rate={} but realtime config frame_rate={}",
                tokenizer.audio.frame_rate,
                config.frame_rate()
            )));
        }

        let left_pad_tokens = tokenizer.audio.streaming_n_left_pad_tokens.ok_or_else(|| {
            VoxtralError::InvalidTokenizer("missing realtime streaming_n_left_pad_tokens".into())
        })?;
        let look_ahead_samples = ms_to_samples(
            tokenizer.audio.streaming_look_ahead_ms.ok_or_else(|| {
                VoxtralError::InvalidTokenizer("missing realtime streaming_look_ahead_ms".into())
            })?,
            sample_rate,
        )?;
        let look_back_samples = ms_to_samples(
            tokenizer.audio.streaming_look_back_ms.ok_or_else(|| {
                VoxtralError::InvalidTokenizer("missing realtime streaming_look_back_ms".into())
            })?,
            sample_rate,
        )?;
        let right_pad_tokens = delay_tokens
            .checked_add(1)
            .and_then(|tokens| tokens.checked_add(REALTIME_DEFAULT_OFFLINE_BUFFER_TOKENS))
            .ok_or_else(|| VoxtralError::InvalidConfig("right pad token overflow".into()))?;
        let prompt = build_realtime_streaming_prompt_with_left_pad(left_pad_tokens, delay_tokens);

        Ok(Self {
            sample_rate,
            raw_audio_length_per_token,
            look_ahead_samples,
            look_back_samples,
            left_pad_tokens,
            delay_tokens,
            right_pad_tokens,
            prompt_token_ids: prompt.input_ids,
        })
    }

    pub fn left_pad_samples(&self) -> usize {
        self.left_pad_tokens * self.raw_audio_length_per_token
    }

    pub fn right_pad_samples(&self) -> usize {
        self.right_pad_tokens * self.raw_audio_length_per_token
    }

    pub fn initial_stride_end_sample(&self) -> usize {
        self.prompt_token_ids.len() * self.raw_audio_length_per_token
    }
}

impl VoxtralRealtimeStreamBuffer {
    pub fn new(config: VoxtralRealtimeStreamConfig) -> Result<Self> {
        if config.raw_audio_length_per_token == 0 {
            return Err(VoxtralError::InvalidConfig(
                "raw audio length per token must be greater than zero".into(),
            ));
        }
        if config.prompt_token_ids.is_empty() {
            return Err(VoxtralError::InvalidConfig(
                "realtime stream prompt must not be empty".into(),
            ));
        }

        let mut samples = Vec::new();
        samples.resize(config.left_pad_samples(), 0.0);
        let token_queue = config.prompt_token_ids.iter().copied().collect();
        let stride_end_sample = config.initial_stride_end_sample();

        Ok(Self {
            config,
            samples,
            token_queue,
            stride_start_sample: 0,
            stride_end_sample,
            input_token_position: 0,
            sequence: 0,
            finished: false,
        })
    }

    pub fn config(&self) -> &VoxtralRealtimeStreamConfig {
        &self.config
    }

    pub fn buffered_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn queued_tokens(&self) -> usize {
        self.token_queue.len()
    }

    pub fn push_audio_16khz(&mut self, samples: &[f32]) -> Result<()> {
        if self.finished {
            return Err(VoxtralError::InvalidConfig(
                "cannot push audio after realtime stream finish".into(),
            ));
        }
        self.samples.extend_from_slice(samples);
        Ok(())
    }

    pub fn push_generated_token(&mut self, token: usize) {
        self.token_queue.push_back(token);
    }

    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        let align_pad_samples = (self.config.raw_audio_length_per_token
            - (self.samples.len() % self.config.raw_audio_length_per_token))
            % self.config.raw_audio_length_per_token;
        self.samples
            .extend(std::iter::repeat_n(0.0, align_pad_samples));
        self.samples
            .extend(std::iter::repeat_n(0.0, self.config.right_pad_samples()));
        self.finished = true;
    }

    pub fn next_window(&mut self) -> Result<Option<VoxtralRealtimeStreamWindow>> {
        let stride_samples = self
            .stride_end_sample
            .checked_sub(self.stride_start_sample)
            .ok_or_else(|| VoxtralError::InvalidConfig("invalid realtime stream stride".into()))?;
        if !stride_samples.is_multiple_of(self.config.raw_audio_length_per_token) {
            return Err(VoxtralError::InvalidConfig(format!(
                "stream stride {stride_samples} is not divisible by raw audio token size {}",
                self.config.raw_audio_length_per_token
            )));
        }
        let token_count = stride_samples / self.config.raw_audio_length_per_token;
        if self.token_queue.len() < token_count {
            return Ok(None);
        }

        let frame_start = self
            .stride_start_sample
            .saturating_sub(self.config.look_back_samples);
        let frame_end = self
            .stride_end_sample
            .checked_add(self.config.look_ahead_samples)
            .ok_or_else(|| VoxtralError::InvalidConfig("stream frame end overflow".into()))?;
        if self.samples.len() < frame_end {
            return Ok(None);
        }

        let input_token_ids = self.token_queue.drain(..token_count).collect::<Vec<_>>();
        let input_token_start = self.input_token_position;
        let input_token_end = input_token_start + token_count;
        let window = VoxtralRealtimeStreamWindow {
            sequence: self.sequence,
            input_token_ids,
            audio_samples: self.samples[frame_start..frame_end].to_vec(),
            frame_start_sample: frame_start,
            frame_end_sample: frame_end,
            stride_start_sample: self.stride_start_sample,
            stride_end_sample: self.stride_end_sample,
            input_token_start,
            input_token_end,
            is_initial: self.sequence == 0,
        };

        self.sequence += 1;
        self.input_token_position = input_token_end;
        self.stride_start_sample = self.stride_end_sample;
        self.stride_end_sample = self
            .stride_end_sample
            .checked_add(self.config.raw_audio_length_per_token)
            .ok_or_else(|| VoxtralError::InvalidConfig("stream stride overflow".into()))?;

        Ok(Some(window))
    }
}

fn ms_to_samples(ms: f64, sample_rate: u32) -> Result<usize> {
    if !ms.is_finite() || ms < 0.0 {
        return Err(VoxtralError::InvalidConfig(format!(
            "streaming millisecond value must be finite and non-negative, got {ms}"
        )));
    }
    let samples = ms * sample_rate as f64 / 1000.0;
    let rounded = samples.round();
    if (samples - rounded).abs() > 1e-6 {
        return Err(VoxtralError::InvalidConfig(format!(
            "{ms}ms at {sample_rate}Hz does not map to an integral sample count"
        )));
    }
    Ok(rounded as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        REALTIME_BOS_TOKEN_ID, REALTIME_SAMPLE_RATE, REALTIME_STREAMING_PAD_TOKEN_ID,
        REALTIME_TRANSCRIPTION_FORMAT,
    };

    fn tiny_realtime_config() -> VoxtralRealtimeConfig {
        VoxtralRealtimeConfig {
            dim: 8,
            n_layers: 1,
            head_dim: 4,
            hidden_dim: 16,
            n_heads: 2,
            n_kv_heads: 1,
            use_biases: false,
            causal: true,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-5,
            vocab_size: 64,
            model_parallel: 1,
            tied_embeddings: true,
            sliding_window: 32,
            model_max_length: 128,
            multimodal: crate::VoxtralRealtimeMultimodalConfig {
                whisper_model_args: crate::VoxtralRealtimeWhisperModelConfig {
                    encoder_args: crate::VoxtralRealtimeAudioEncoderConfig {
                        audio_encoding_args: crate::VoxtralRealtimeAudioEncodingConfig {
                            sampling_rate: REALTIME_SAMPLE_RATE,
                            frame_rate: 12.5,
                            num_mel_bins: 4,
                            hop_length: 160,
                            window_size: 400,
                            chunk_length_s: None,
                            global_log_mel_max: 1.5,
                            transcription_format: REALTIME_TRANSCRIPTION_FORMAT.to_string(),
                        },
                        dim: 8,
                        n_layers: 1,
                        head_dim: 4,
                        hidden_dim: 16,
                        n_heads: 2,
                        vocab_size: 64,
                        n_kv_heads: 1,
                        use_biases: true,
                        use_cache: false,
                        rope_theta: 1_000_000.0,
                        causal: true,
                        norm_eps: 1e-5,
                        pos_embed: "rope".to_string(),
                        max_source_positions: None,
                        ffn_type: "swiglu".to_string(),
                        norm_type: "rms_norm".to_string(),
                        sliding_window: 16,
                    },
                    downsample_args: crate::VoxtralRealtimeDownsampleConfig {
                        downsample_factor: 4,
                    },
                },
            },
            ada_rms_norm_t_cond: true,
            ada_rms_norm_t_cond_dim: Some(32),
        }
    }

    fn tokenizer_json() -> String {
        serde_json::json!({
            "config": {
                "pattern": ".",
                "num_vocab_tokens": 0,
                "default_vocab_size": 64,
                "default_num_special_tokens": 35,
                "version": "v1"
            },
            "vocab": [],
            "special_tokens": (0..35).map(|rank| {
                let token = match rank {
                    1 => "<s>",
                    2 => "</s>",
                    24 => "[AUDIO]",
                    25 => "[BEGIN_AUDIO]",
                    26 => "[OUTPUT_AUDIO]",
                    32 => "[STREAMING_PAD]",
                    33 => "[STREAMING_WORD]",
                    34 => "[REPEAT_AUDIO_TEXT]",
                    _ => "<reserved>",
                };
                serde_json::json!({
                    "rank": rank,
                    "token_str": token,
                    "is_control": true
                })
            }).collect::<Vec<_>>(),
            "audio": {
                "sampling_rate": REALTIME_SAMPLE_RATE,
                "frame_rate": 12.5,
                "audio_encoding_config": {
                    "num_mel_bins": 4,
                    "hop_length": 160,
                    "window_size": 400
                },
                "transcription_delay_ms": 480,
                "streaming_look_ahead_ms": 2.5,
                "streaming_look_back_ms": 52.5,
                "streaming_n_left_pad_tokens": 32,
                "transcription_format": REALTIME_TRANSCRIPTION_FORMAT
            }
        })
        .to_string()
    }

    fn tokenizer() -> VoxtralTokenizerMetadata {
        VoxtralTokenizerMetadata::from_json_str(&tokenizer_json()).unwrap()
    }

    #[test]
    fn builds_stream_config_from_realtime_metadata() {
        let config = tiny_realtime_config();
        let stream = VoxtralRealtimeStreamConfig::from_metadata(&config, &tokenizer()).unwrap();

        assert_eq!(stream.sample_rate, REALTIME_SAMPLE_RATE);
        assert_eq!(stream.raw_audio_length_per_token, 1280);
        assert_eq!(stream.look_ahead_samples, 40);
        assert_eq!(stream.look_back_samples, 840);
        assert_eq!(stream.left_pad_tokens, 32);
        assert_eq!(stream.delay_tokens, 6);
        assert_eq!(stream.right_pad_tokens, 17);
        assert_eq!(stream.left_pad_samples(), 40_960);
        assert_eq!(stream.right_pad_samples(), 21_760);
        assert_eq!(stream.initial_stride_end_sample(), 49_920);
        assert_eq!(stream.prompt_token_ids.len(), 39);
        assert_eq!(stream.prompt_token_ids[0], REALTIME_BOS_TOKEN_ID);
        assert!(stream.prompt_token_ids[1..]
            .iter()
            .all(|token| *token == REALTIME_STREAMING_PAD_TOKEN_ID));
    }

    #[test]
    fn emits_initial_window_then_waits_for_audio_and_token_feedback() {
        let config = tiny_realtime_config();
        let stream_config =
            VoxtralRealtimeStreamConfig::from_metadata(&config, &tokenizer()).unwrap();
        let mut buffer = VoxtralRealtimeStreamBuffer::new(stream_config).unwrap();

        assert_eq!(buffer.buffered_samples(), 40_960);
        assert_eq!(buffer.queued_tokens(), 39);
        assert!(buffer.next_window().unwrap().is_none());

        buffer.push_audio_16khz(&vec![0.5; 9_000]).unwrap();
        let initial = buffer.next_window().unwrap().unwrap();
        assert!(initial.is_initial);
        assert_eq!(initial.sequence, 0);
        assert_eq!(initial.input_token_ids.len(), 39);
        assert_eq!(initial.frame_start_sample, 0);
        assert_eq!(initial.frame_end_sample, 49_960);
        assert_eq!(initial.stride_start_sample, 0);
        assert_eq!(initial.stride_end_sample, 49_920);
        assert_eq!(initial.audio_samples.len(), 49_960);

        assert!(buffer.next_window().unwrap().is_none());
        buffer.push_generated_token(41);
        assert!(buffer.next_window().unwrap().is_none());

        buffer.push_audio_16khz(&vec![0.25; 1_280]).unwrap();
        let next = buffer.next_window().unwrap().unwrap();
        assert!(!next.is_initial);
        assert_eq!(next.sequence, 1);
        assert_eq!(next.input_token_ids, vec![41]);
        assert_eq!(next.frame_start_sample, 49_080);
        assert_eq!(next.frame_end_sample, 51_240);
        assert_eq!(next.stride_start_sample, 49_920);
        assert_eq!(next.stride_end_sample, 51_200);
        assert_eq!(next.audio_samples.len(), 2_160);
    }

    #[test]
    fn finish_adds_alignment_and_right_padding_once() {
        let config = tiny_realtime_config();
        let stream_config =
            VoxtralRealtimeStreamConfig::from_metadata(&config, &tokenizer()).unwrap();
        let mut buffer = VoxtralRealtimeStreamBuffer::new(stream_config).unwrap();
        buffer.push_audio_16khz(&vec![1.0; 1_281]).unwrap();

        buffer.finish();
        let once = buffer.buffered_samples();
        buffer.finish();

        assert_eq!(buffer.buffered_samples(), once);
        assert_eq!(once, 40_960 + 1_281 + 1_279 + 21_760);
        assert!(buffer.push_audio_16khz(&[1.0]).is_err());
    }
}
