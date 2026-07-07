use candle_core::Tensor;

use crate::{
    realtime_log_mel_spectrogram, Result, VoxtralError, VoxtralRealtimeStreamBuffer,
    VoxtralRealtimeStreamConfig, VoxtralRealtimeStreamWindow, VoxtralRealtimeTranscriber,
    VoxtralRealtimeTranscriptionOptions, REALTIME_EOS_TOKEN_ID,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralRealtimeStreamStep {
    pub sequence: usize,
    pub token: usize,
    pub text: String,
    pub reached_eos: bool,
    pub output_token_count: usize,
    pub frame_start_sample: usize,
    pub frame_end_sample: usize,
    pub input_token_start: usize,
    pub input_token_end: usize,
}

/// Stateful Voxtral Realtime decoding session.
///
/// This implements the model-native token/audio feedback loop. It intentionally
/// does not cache decoder KV state yet; each step recomputes the current text
/// history while preserving the externally visible streaming contract.
pub struct VoxtralRealtimeStreamSession<'a> {
    transcriber: &'a VoxtralRealtimeTranscriber,
    buffer: VoxtralRealtimeStreamBuffer,
    options: VoxtralRealtimeTranscriptionOptions,
    input_token_ids: Vec<usize>,
    output_tokens: Vec<usize>,
    audio_embeddings: Option<Tensor>,
    done: bool,
}

impl VoxtralRealtimeTranscriber {
    pub fn stream_session(
        &self,
        config: VoxtralRealtimeStreamConfig,
        options: VoxtralRealtimeTranscriptionOptions,
    ) -> Result<VoxtralRealtimeStreamSession<'_>> {
        VoxtralRealtimeStreamSession::new(self, config, options)
    }
}

impl<'a> VoxtralRealtimeStreamSession<'a> {
    pub fn new(
        transcriber: &'a VoxtralRealtimeTranscriber,
        config: VoxtralRealtimeStreamConfig,
        options: VoxtralRealtimeTranscriptionOptions,
    ) -> Result<Self> {
        Ok(Self {
            transcriber,
            buffer: VoxtralRealtimeStreamBuffer::new(config)?,
            options,
            input_token_ids: Vec::new(),
            output_tokens: Vec::new(),
            audio_embeddings: None,
            done: false,
        })
    }

    pub fn push_audio_16khz(&mut self, samples: &[f32]) -> Result<()> {
        self.buffer.push_audio_16khz(samples)
    }

    pub fn push_generated_token_for_test(&mut self, token: usize) {
        self.buffer.push_generated_token(token);
    }

    pub fn finish(&mut self) {
        self.buffer.finish();
    }

    pub fn output_tokens(&self) -> &[usize] {
        &self.output_tokens
    }

    pub fn text(&self) -> String {
        decode_text(
            &self.transcriber.token_decoder,
            self.output_tokens.iter().copied(),
        )
    }

    pub fn next_step(&mut self) -> Result<Option<VoxtralRealtimeStreamStep>> {
        if self.done || self.output_tokens.len() >= self.options.max_new_tokens {
            return Ok(None);
        }
        let Some(window) = self.buffer.next_window()? else {
            return Ok(None);
        };

        let window_embeddings = self.transcriber.encode_stream_window_embeddings(&window)?;
        self.audio_embeddings = Some(match &self.audio_embeddings {
            Some(existing) => {
                Tensor::cat(&[existing, &window_embeddings], 1).map_err(candle_err)?
            }
            None => window_embeddings,
        });
        self.input_token_ids
            .extend(window.input_token_ids.iter().copied());

        let audio_embeddings = self
            .audio_embeddings
            .as_ref()
            .expect("audio embeddings are set above");
        let generation = self
            .transcriber
            .text_decoder
            .greedy_decode_audio_embeddings_with_prompt(
                &self.transcriber.token_embeddings,
                audio_embeddings,
                &self.input_token_ids,
                self.options.delay_tokens,
                1,
            )
            .map_err(candle_err)?;
        let Some(token) = generation.generated_tokens.first().copied() else {
            return Ok(None);
        };

        let reached_eos = token == REALTIME_EOS_TOKEN_ID;
        self.output_tokens.push(token);
        if reached_eos {
            self.done = true;
        } else {
            self.buffer.push_generated_token(token);
        }
        let text = self.text();

        Ok(Some(VoxtralRealtimeStreamStep {
            sequence: window.sequence,
            token,
            text,
            reached_eos,
            output_token_count: self.output_tokens.len(),
            frame_start_sample: window.frame_start_sample,
            frame_end_sample: window.frame_end_sample,
            input_token_start: window.input_token_start,
            input_token_end: window.input_token_end,
        }))
    }
}

impl VoxtralRealtimeTranscriber {
    pub fn encode_stream_window_embeddings(
        &self,
        window: &VoxtralRealtimeStreamWindow,
    ) -> Result<Tensor> {
        let mel = realtime_log_mel_spectrogram(&self.config, &window.audio_samples)?;
        let input_features = Tensor::from_vec(
            mel.to_channel_major(),
            (1, mel.mel_bins, mel.frames),
            self.token_embeddings.tok_embeddings.embeddings().device(),
        )
        .map_err(candle_err)?
        .to_dtype(self.token_embeddings.tok_embeddings.embeddings().dtype())
        .map_err(candle_err)?;
        let audio_start_pos = window
            .input_token_start
            .checked_mul(self.config.downsample_factor())
            .ok_or_else(|| VoxtralError::InvalidConfig("stream audio position overflow".into()))?;
        let embeddings = self
            .audio_modules
            .forward(&input_features, audio_start_pos)
            .map_err(candle_err)?;
        let actual_tokens = embeddings.dim(1).map_err(candle_err)?;
        let expected_tokens = window.input_token_ids.len();
        if actual_tokens < expected_tokens {
            return Err(VoxtralError::Candle(format!(
                "stream window produced {actual_tokens} audio embeddings for {expected_tokens} input tokens"
            )));
        }
        if actual_tokens == expected_tokens {
            return Ok(embeddings);
        }
        embeddings
            .narrow(1, actual_tokens - expected_tokens, expected_tokens)
            .map_err(candle_err)
    }
}

fn candle_err(err: candle_core::Error) -> VoxtralError {
    VoxtralError::Candle(err.to_string())
}

fn decode_text(
    decoder: &crate::VoxtralTekkenDecoder,
    tokens: impl IntoIterator<Item = usize>,
) -> String {
    let text_tokens = tokens
        .into_iter()
        .filter(|token| *token != REALTIME_EOS_TOKEN_ID)
        .collect::<Vec<_>>();
    decoder.decode(&text_tokens).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        VoxtralRealtimeAudioModules, VoxtralRealtimeConfig, VoxtralRealtimeDownsampleConfig,
        VoxtralRealtimeInferenceModules, VoxtralRealtimeMultimodalConfig,
        VoxtralRealtimeWhisperModelConfig, VoxtralTokenizerMetadata, REALTIME_SAMPLE_RATE,
        REALTIME_TRANSCRIPTION_FORMAT,
    };
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

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
            multimodal: VoxtralRealtimeMultimodalConfig {
                whisper_model_args: VoxtralRealtimeWhisperModelConfig {
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
                    downsample_args: VoxtralRealtimeDownsampleConfig {
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

    fn transcriber() -> VoxtralRealtimeTranscriber {
        let config = tiny_realtime_config();
        let modules = VoxtralRealtimeInferenceModules::load(
            &config,
            VarBuilder::zeros(DType::F32, &Device::Cpu),
        )
        .unwrap();
        let audio_modules =
            VoxtralRealtimeAudioModules::load(&config, VarBuilder::zeros(DType::F32, &Device::Cpu))
                .unwrap();
        let decoder = crate::VoxtralRealtimeTextDecoder::load(
            &config,
            VarBuilder::zeros(DType::F32, &Device::Cpu),
        )
        .unwrap();
        let tokenizer = VoxtralTokenizerMetadata::from_json_str(&tokenizer_json()).unwrap();
        VoxtralRealtimeTranscriber::new(
            config,
            modules.token_embeddings,
            audio_modules,
            decoder,
            tokenizer.decoder().unwrap(),
        )
    }

    #[test]
    fn streaming_session_waits_for_audio_then_emits_token_steps() {
        let transcriber = transcriber();
        let tokenizer = VoxtralTokenizerMetadata::from_json_str(&tokenizer_json()).unwrap();
        let stream_config =
            VoxtralRealtimeStreamConfig::from_metadata(&transcriber.config, &tokenizer).unwrap();
        let mut session = transcriber
            .stream_session(
                stream_config,
                VoxtralRealtimeTranscriptionOptions {
                    delay_tokens: 6,
                    max_new_tokens: 2,
                },
            )
            .unwrap();

        assert!(session.next_step().unwrap().is_none());
        session.push_audio_16khz(&vec![0.0; 9_000]).unwrap();
        let first = session.next_step().unwrap().unwrap();
        assert_eq!(first.sequence, 0);
        assert_eq!(first.input_token_start, 0);
        assert_eq!(first.input_token_end, 39);
        assert_eq!(first.output_token_count, 1);

        assert!(session.next_step().unwrap().is_none());
        session.push_audio_16khz(&vec![0.0; 1_280]).unwrap();
        let second = session.next_step().unwrap().unwrap();
        assert_eq!(second.sequence, 1);
        assert_eq!(second.input_token_start, 39);
        assert_eq!(second.input_token_end, 40);
        assert_eq!(second.output_token_count, 2);
        assert_eq!(session.output_tokens().len(), 2);
        assert!(session.next_step().unwrap().is_none());
    }
}
