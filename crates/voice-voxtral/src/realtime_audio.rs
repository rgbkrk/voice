use crate::{Result, VoxtralError, VoxtralRealtimeConfig};

pub const REALTIME_MIN_MEL_VALUE: f32 = 1e-10;

#[derive(Debug, Clone, PartialEq)]
pub struct VoxtralRealtimeMelFilters {
    pub mel_bins: usize,
    pub frequency_bins: usize,
    /// Row-major `[mel_bins, frequency_bins]`.
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VoxtralRealtimeMelSpectrogram {
    pub frames: usize,
    pub mel_bins: usize,
    /// Row-major `[frames, mel_bins]`.
    pub values: Vec<f32>,
}

impl VoxtralRealtimeMelSpectrogram {
    pub fn as_frame_major(&self) -> &[f32] {
        &self.values
    }

    pub fn to_channel_major(&self) -> Vec<f32> {
        let mut values = vec![0.0; self.values.len()];
        for frame in 0..self.frames {
            for mel in 0..self.mel_bins {
                values[mel * self.frames + frame] = self.values[frame * self.mel_bins + mel];
            }
        }
        values
    }
}

pub fn realtime_mel_filters(config: &VoxtralRealtimeConfig) -> Result<VoxtralRealtimeMelFilters> {
    let encoding = realtime_audio_encoding(config);
    let mel_bins = encoding.num_mel_bins;
    let n_fft = encoding.window_size;
    if n_fft == 0 || !n_fft.is_multiple_of(2) {
        return Err(VoxtralError::InvalidConfig(format!(
            "realtime window_size must be non-zero and even, got {n_fft}"
        )));
    }
    if mel_bins == 0 {
        return Err(VoxtralError::InvalidConfig(
            "realtime num_mel_bins must be greater than zero".into(),
        ));
    }

    let frequency_bins = 1 + n_fft / 2;
    let max_frequency = encoding.sampling_rate as f32 / 2.0;
    let mel_min = hertz_to_mel(0.0);
    let mel_max = hertz_to_mel(max_frequency);

    let fft_freqs: Vec<f32> = (0..frequency_bins)
        .map(|idx| max_frequency * idx as f32 / (frequency_bins - 1) as f32)
        .collect();

    let filter_freqs: Vec<f32> = (0..mel_bins + 2)
        .map(|idx| {
            let mel = mel_min + (mel_max - mel_min) * idx as f32 / (mel_bins + 1) as f32;
            mel_to_hertz(mel)
        })
        .collect();

    let mut filter_diff = Vec::with_capacity(mel_bins + 1);
    for idx in 0..mel_bins + 1 {
        filter_diff.push((filter_freqs[idx + 1] - filter_freqs[idx]).max(1e-6));
    }

    let mut values = vec![0.0; mel_bins * frequency_bins];
    for mel in 0..mel_bins {
        let enorm = 2.0 / (filter_freqs[mel + 2] - filter_freqs[mel]);
        for (freq_idx, freq) in fft_freqs.iter().enumerate() {
            let down = (*freq - filter_freqs[mel]) / filter_diff[mel];
            let up = (filter_freqs[mel + 2] - *freq) / filter_diff[mel + 1];
            values[mel * frequency_bins + freq_idx] = down.min(up).max(0.0) * enorm;
        }
    }

    Ok(VoxtralRealtimeMelFilters {
        mel_bins,
        frequency_bins,
        values,
    })
}

pub fn realtime_log_mel_spectrogram(
    config: &VoxtralRealtimeConfig,
    samples: &[f32],
) -> Result<VoxtralRealtimeMelSpectrogram> {
    realtime_log_mel_spectrogram_with_center(config, samples, true)
}

pub fn realtime_log_mel_spectrogram_with_center(
    config: &VoxtralRealtimeConfig,
    samples: &[f32],
    center: bool,
) -> Result<VoxtralRealtimeMelSpectrogram> {
    let encoding = realtime_audio_encoding(config);
    let n_fft = encoding.window_size;
    let hop = encoding.hop_length;
    if n_fft == 0 || hop == 0 {
        return Err(VoxtralError::InvalidConfig(
            "realtime window_size and hop_length must be greater than zero".into(),
        ));
    }
    if samples.is_empty() {
        return Err(VoxtralError::InvalidConfig(
            "realtime mel spectrogram requires at least one audio sample".into(),
        ));
    }

    let filters = realtime_mel_filters(config)?;
    let padded = if center {
        reflect_pad(samples, n_fft / 2)
    } else {
        samples.to_vec()
    };
    let total_frames = padded
        .len()
        .checked_sub(n_fft)
        .map(|remaining| remaining / hop + 1)
        .unwrap_or(0);
    let frames = total_frames.saturating_sub(1);
    if frames == 0 {
        return Err(VoxtralError::InvalidConfig(format!(
            "realtime mel spectrogram input is too short: {} samples",
            samples.len()
        )));
    }

    let window = periodic_hann_window(n_fft);
    let (dft_cos, dft_sin) = dft_tables(n_fft, filters.frequency_bins);
    let min_log = encoding.global_log_mel_max as f32 - 8.0;
    let mut values = vec![0.0; frames * filters.mel_bins];

    for frame in 0..frames {
        let start = frame * hop;
        let mut power = vec![0.0; filters.frequency_bins];

        for freq in 0..filters.frequency_bins {
            let mut re = 0.0;
            let mut im = 0.0;
            let cos_row = &dft_cos[freq * n_fft..(freq + 1) * n_fft];
            let sin_row = &dft_sin[freq * n_fft..(freq + 1) * n_fft];
            for idx in 0..n_fft {
                let sample = padded[start + idx] * window[idx];
                re += sample * cos_row[idx];
                im += sample * sin_row[idx];
            }
            power[freq] = re * re + im * im;
        }

        for mel in 0..filters.mel_bins {
            let filter =
                &filters.values[mel * filters.frequency_bins..(mel + 1) * filters.frequency_bins];
            let mel_energy = filter
                .iter()
                .zip(power.iter())
                .map(|(weight, power)| weight * power)
                .sum::<f32>();
            let log_spec = mel_energy.max(REALTIME_MIN_MEL_VALUE).log10().max(min_log);
            values[frame * filters.mel_bins + mel] = (log_spec + 4.0) / 4.0;
        }
    }

    Ok(VoxtralRealtimeMelSpectrogram {
        frames,
        mel_bins: filters.mel_bins,
        values,
    })
}

fn realtime_audio_encoding(
    config: &VoxtralRealtimeConfig,
) -> &crate::VoxtralRealtimeAudioEncodingConfig {
    &config
        .multimodal
        .whisper_model_args
        .encoder_args
        .audio_encoding_args
}

fn hertz_to_mel(freq: f32) -> f32 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / f32::ln(6.4);
    let mels = 3.0 * freq / 200.0;
    if freq >= min_log_hertz {
        min_log_mel + f32::ln(freq / min_log_hertz) * logstep
    } else {
        mels
    }
}

fn mel_to_hertz(mel: f32) -> f32 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = f32::ln(6.4) / 27.0;
    let freq = 200.0 * mel / 3.0;
    if mel >= min_log_mel {
        min_log_hertz * f32::exp(logstep * (mel - min_log_mel))
    } else {
        freq
    }
}

fn periodic_hann_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| 0.5 * (1.0 - f32::cos(2.0 * std::f32::consts::PI * idx as f32 / len as f32)))
        .collect()
}

fn dft_tables(n_fft: usize, frequency_bins: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0.0; frequency_bins * n_fft];
    let mut sin = vec![0.0; frequency_bins * n_fft];
    for freq in 0..frequency_bins {
        for idx in 0..n_fft {
            let angle = 2.0 * std::f32::consts::PI * freq as f32 * idx as f32 / n_fft as f32;
            cos[freq * n_fft + idx] = angle.cos();
            sin[freq * n_fft + idx] = angle.sin();
        }
    }
    (cos, sin)
}

fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    for idx in 0..pad {
        let src = pad - idx;
        padded.push(samples.get(src).copied().unwrap_or(0.0));
    }
    padded.extend_from_slice(samples);
    for idx in 0..pad {
        let src = samples.len().checked_sub(2 + idx);
        padded.push(src.and_then(|src| samples.get(src)).copied().unwrap_or(0.0));
    }
    padded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{plan_realtime_audio_padding, VoxtralRealtimeConfig};

    const REALTIME_PARAMS_JSON: &str = r#"{
      "dim": 3072,
      "n_layers": 26,
      "head_dim": 128,
      "hidden_dim": 9216,
      "n_heads": 32,
      "n_kv_heads": 8,
      "use_biases": false,
      "causal": true,
      "rope_theta": 1000000.0,
      "norm_eps": 0.00001,
      "vocab_size": 131072,
      "model_parallel": 1,
      "tied_embeddings": true,
      "sliding_window": 8192,
      "model_max_length": 131072,
      "multimodal": {
        "whisper_model_args": {
          "encoder_args": {
            "audio_encoding_args": {
              "sampling_rate": 16000,
              "frame_rate": 12.5,
              "num_mel_bins": 128,
              "hop_length": 160,
              "window_size": 400,
              "chunk_length_s": null,
              "global_log_mel_max": 1.5,
              "transcription_format": "streaming"
            },
            "dim": 1280,
            "n_layers": 32,
            "head_dim": 64,
            "hidden_dim": 5120,
            "n_heads": 32,
            "vocab_size": 131072,
            "n_kv_heads": 32,
            "use_biases": true,
            "use_cache": false,
            "rope_theta": 1000000.0,
            "causal": true,
            "norm_eps": 0.00001,
            "pos_embed": "rope",
            "max_source_positions": null,
            "ffn_type": "swiglu",
            "norm_type": "rms_norm",
            "sliding_window": 750
          },
          "downsample_args": {
            "downsample_factor": 4
          }
        }
      },
      "ada_rms_norm_t_cond": true,
      "ada_rms_norm_t_cond_dim": 32
    }"#;

    fn realtime_config() -> VoxtralRealtimeConfig {
        VoxtralRealtimeConfig::from_json_str(REALTIME_PARAMS_JSON).unwrap()
    }

    #[test]
    fn builds_slaney_mel_filters_with_realtime_shape() {
        let config = realtime_config();
        let filters = realtime_mel_filters(&config).unwrap();

        assert_eq!(filters.mel_bins, 128);
        assert_eq!(filters.frequency_bins, 201);
        assert_eq!(filters.values.len(), 128 * 201);
        assert!(filters.values.iter().all(|value| value.is_finite()));
        assert!(filters.values.iter().all(|value| *value >= 0.0));
        assert!(filters.values.iter().any(|value| *value > 0.0));
    }

    #[test]
    fn silence_log_mel_uses_fixed_global_floor() {
        let config = realtime_config();
        let silence = vec![0.0; 16_000];
        let mel = realtime_log_mel_spectrogram(&config, &silence).unwrap();
        let expected = ((1.5f32 - 8.0) + 4.0) / 4.0;

        assert_eq!(mel.frames, 100);
        assert_eq!(mel.mel_bins, 128);
        assert_eq!(mel.values.len(), 100 * 128);
        assert!(mel
            .values
            .iter()
            .all(|value| (*value - expected).abs() < 1e-6));
    }

    #[test]
    fn padded_offline_audio_has_one_mel_frame_per_hop() {
        let config = realtime_config();
        let samples = vec![0.0; 16_000];
        let plan = plan_realtime_audio_padding(&config, samples.len(), 6).unwrap();
        let padded = crate::pad_realtime_audio(&samples, &plan);
        let mel = realtime_log_mel_spectrogram(&config, &padded).unwrap();

        assert_eq!(mel.frames, plan.padded_samples / 160);
        assert_eq!(mel.frames, 496);
        assert_eq!(mel.to_channel_major().len(), mel.values.len());
    }
}
