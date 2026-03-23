//! Mel spectrogram computation for Whisper.

use candle_core::{Device, Tensor};
use candle_transformers::models::whisper::{audio, Config};

/// Pre-computed mel filter banks embedded in the binary.
const MEL_FILTERS_80: &[u8] = include_bytes!("../data/melfilters.bytes");
const MEL_FILTERS_128: &[u8] = include_bytes!("../data/melfilters128.bytes");

/// Load mel filter coefficients for the given config.
///
/// Returns the filter bank as a flat `Vec<f32>`. The filters are embedded
/// in the binary so no filesystem or network access is needed.
pub fn load_mel_filters(config: &Config) -> Result<Vec<f32>, String> {
    let mel_bytes = match config.num_mel_bins {
        80 => MEL_FILTERS_80,
        128 => MEL_FILTERS_128,
        n => {
            return Err(format!(
                "unsupported num_mel_bins: {n} (expected 80 or 128)"
            ))
        }
    };

    let mut filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut filters);
    Ok(filters)
}

/// Convert PCM audio samples to a mel spectrogram tensor on the given device.
///
/// Input: mono f32 samples at 16kHz.
/// Output: tensor of shape `(1, num_mel_bins, num_frames)`.
pub fn pcm_to_mel(
    config: &Config,
    samples: &[f32],
    mel_filters: &[f32],
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mel = audio::pcm_to_mel(config, samples, mel_filters);
    let mel_len = mel.len();
    Tensor::from_vec(
        mel,
        (1, config.num_mel_bins, mel_len / config.num_mel_bins),
        device,
    )
}
