//! Greedy decode loop for Whisper with KV-caching.
//!
//! Ported from the candle whisper example with simplifications for
//! library use (no timestamps mode by default, greedy decoding with
//! temperature fallback).

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::models::whisper::{self as m, Config};
use tokenizers::Tokenizer;

/// Result of decoding a single audio segment.
#[derive(Debug, Clone)]
pub struct DecodingResult {
    /// Decoded token IDs (including special tokens).
    pub tokens: Vec<u32>,
    /// Decoded text with special tokens stripped.
    pub text: String,
    /// Average log-probability of decoded tokens.
    pub avg_logprob: f64,
    /// Probability that the segment contains no speech.
    pub no_speech_prob: f64,
    /// Temperature used for this decode.
    pub temperature: f64,
}

/// Whisper decoder wrapping the model, tokenizer, and mel filters.
///
/// Handles greedy autoregressive decoding with KV-caching.
pub struct WhisperDecoder {
    model: m::model::Whisper,
    config: Config,
    device: Device,
    mel_filters: Vec<f32>,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    eot_token: u32,
    transcribe_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl WhisperDecoder {
    /// Create a new decoder.
    ///
    /// `language_token` should be `None` for `.en` models and `Some(token_id)`
    /// for multilingual models (e.g. the `<|en|>` token).
    pub fn new(
        model: m::model::Whisper,
        config: Config,
        tokenizer: Tokenizer,
        mel_filters: Vec<f32>,
        device: Device,
        language_token: Option<u32>,
    ) -> candle_core::Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok())
            .ok_or_else(|| {
                candle_core::Error::Msg("unable to find any no-speech token".to_string())
            })?;

        // Build suppress_tokens mask
        let suppress_tokens: Vec<f32> = (0..config.vocab_size as u32)
            .map(|i| {
                if config.suppress_tokens.contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), &device)?;

        Ok(Self {
            model,
            config,
            device,
            mel_filters,
            tokenizer,
            suppress_tokens,
            sot_token,
            eot_token,
            transcribe_token,
            no_speech_token,
            no_timestamps_token,
            language_token,
        })
    }

    /// Access the model config.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Access the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Access the mel filters.
    pub fn mel_filters(&self) -> &[f32] {
        &self.mel_filters
    }

    /// Transcribe PCM audio samples (mono, 16kHz f32).
    ///
    /// Audio longer than 30 seconds is truncated to the first 30s.
    /// For typical voice-command audio (<30s), this produces a single segment.
    pub fn transcribe(&mut self, samples: &[f32]) -> candle_core::Result<DecodingResult> {
        use candle_transformers::models::whisper::N_SAMPLES;

        // Truncate to 30s (Whisper's max chunk size)
        let max_samples = N_SAMPLES; // 480000 = 30s at 16kHz
        let samples = if samples.len() > max_samples {
            &samples[..max_samples]
        } else {
            samples
        };

        let mel = crate::pcm_to_mel(&self.config, samples, &self.mel_filters, &self.device)?;

        // Ensure mel frames don't exceed encoder's max source positions
        let (_, _, mel_frames) = mel.dims3()?;
        let mel = if mel_frames > self.config.max_source_positions {
            mel.narrow(2, 0, self.config.max_source_positions)?
        } else {
            mel
        };

        self.decode_with_fallback(&mel)
    }

    /// Transcribe and return just the text + tokens (convenience).
    pub fn transcribe_text(&mut self, samples: &[f32]) -> candle_core::Result<(String, Vec<u32>)> {
        let result = self.transcribe(samples)?;
        Ok((result.text, result.tokens))
    }

    /// Decode a mel spectrogram segment at the given temperature.
    ///
    /// Temperature 0.0 = greedy (argmax), >0 = softmax sampling.
    fn decode(&mut self, mel: &Tensor, temperature: f64) -> candle_core::Result<DecodingResult> {
        let audio_features = self.model.encoder.forward(mel, true)?;

        let sample_len = self.config.max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];

        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        tokens.push(self.transcribe_token);
        tokens.push(self.no_timestamps_token);

        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = self
                .model
                .decoder
                .forward(&tokens_t, &audio_features, i == 0)?;

            // Extract no-speech probability on first iteration
            if i == 0 {
                let logits = self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            let logits = logits.broadcast_add(&self.suppress_tokens)?;

            let next_token = if temperature > 0.0 {
                sample_token(&logits, temperature)?
            } else {
                // Greedy: argmax on GPU, only download the single index
                logits.argmax(0)?.to_scalar::<u32>()?
            };

            tokens.push(next_token);

            // Compute log-prob on GPU, download single scalar
            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;

            if next_token == self.eot_token || tokens.len() > self.config.max_target_positions {
                break;
            }

            // Detect repetition loops: if the last N tokens are all the same,
            // the model is hallucinating on silence — bail early.
            if tokens.len() >= 8 {
                let last = tokens[tokens.len() - 1];
                let repeating = tokens[tokens.len() - 8..].iter().all(|&t| t == last);
                if repeating {
                    // Trim the repeated tokens
                    while tokens.last() == Some(&last) {
                        tokens.pop();
                    }
                    break;
                }
            }

            sum_logprob += prob.ln();
        }

        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature,
        })
    }

    /// Decode with temperature fallback for robustness.
    ///
    /// Tries greedy first (T=0), falls back to higher temperatures if
    /// average log-probability is too low.
    fn decode_with_fallback(&mut self, mel: &Tensor) -> candle_core::Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr = self.decode(mel, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(_) => continue,
            }
        }
        unreachable!()
    }
}

/// Resolve a special token string to its ID.
fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

/// Sample a token from logits using temperature scaling.
fn sample_token(logits: &Tensor, temperature: f64) -> candle_core::Result<u32> {
    use rand::distr::weighted::WeightedIndex;
    use rand::distr::Distribution;
    use rand::SeedableRng;

    let prs = softmax(&(logits / temperature)?, 0)?;
    let logits_v: Vec<f32> = prs.to_vec1()?;
    let distr = WeightedIndex::new(&logits_v)
        .map_err(|e| candle_core::Error::Msg(format!("WeightedIndex error: {e}")))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(299792458);
    Ok(distr.sample(&mut rng) as u32)
}
