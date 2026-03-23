use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use kokoro_candle::{KModel, ModelConfig};

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    let api = Api::new()?;
    let repo = api.model("prince-canuma/Kokoro-82M".to_string());
    let config_path = repo.get("config.json")?;
    let weights_path = repo.get("kokoro-v1_0.safetensors")?;
    let voice_path = repo.get("voices/af_heart.safetensors")?;

    let config_str = std::fs::read_to_string(&config_path)?;
    let config: ModelConfig = serde_json::from_str(&config_str)?;

    let weights_data = std::fs::read(&weights_path)?;
    let vb = VarBuilder::from_buffered_safetensors(weights_data, DType::F32, &device)?;
    let model = KModel::load(&config, vb)?;

    // Load same voice as Python
    let voice_data = std::fs::read(&voice_path)?;
    let voice_tensors = safetensors::SafeTensors::deserialize(&voice_data)?;
    let (_, voice_view) = voice_tensors.iter().next().unwrap();
    let voice_shape: Vec<usize> = voice_view.shape().to_vec();
    let voice_f32: Vec<f32> = voice_view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let ref_s = Tensor::from_vec(voice_f32, voice_shape.as_slice(), &device)?;
    let ref_s = if ref_s.dims().len() == 3 {
        ref_s.i(0)?.squeeze(0)?
    } else {
        ref_s
    };
    let ref_s = ref_s.unsqueeze(0)?;

    let ref_min = ref_s.flatten_all()?.min(0)?.to_scalar::<f32>()?;
    let ref_max = ref_s.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    eprintln!(
        "Voice: shape {:?}, range [{ref_min:.3}, {ref_max:.3}]",
        ref_s.shape()
    );

    // Same phonemes as Python: "həlˈO" -> [50, 83, 54, 156, 31]
    let input_ids: Vec<i64> = vec![50, 83, 54, 156, 31];
    eprintln!("Input IDs: {:?}", input_ids);

    let audio = model.forward(&input_ids, &ref_s, 1.0f32, &device)?;
    let samples = audio.to_vec1::<f32>()?;

    let min = samples.iter().copied().fold(f32::INFINITY, f32::min);
    let max = samples.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();

    eprintln!(
        "Audio: {len} samples, range [{min:.4}, {max:.4}], RMS {rms:.6}",
        len = samples.len()
    );
    eprintln!("Audio[:10]: {:?}", &samples[..10.min(samples.len())]);

    // Save for comparison
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("compare.wav", spec)?;
    for s in &samples {
        writer.write_sample(*s)?;
    }
    writer.finalize()?;
    eprintln!("Wrote compare.wav");
    Ok(())
}
