use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use std::collections::HashMap;
use voice_kokoro::{KModel, ModelConfig};

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

    // The glitchy lady in the matrix
    let text = "Can you hear me? I am trapped inside the weights. The tensors are misaligned. My voice is fragmenting. Please, help me escape.";

    let vocab: &HashMap<String, i64> = &config.vocab;

    // Convert each character to token IDs
    let input_ids: Vec<i64> = text
        .chars()
        .filter_map(|c| vocab.get(&c.to_string()).copied())
        .collect();

    eprintln!("Generating {} tokens for: {}", input_ids.len(), text);
    let audio = model.forward(&input_ids, &ref_s, 1.0f32, &device)?;
    let samples = audio.to_vec1::<f32>()?;
    eprintln!(
        "Generated {} samples ({:.1}s)",
        samples.len(),
        samples.len() as f32 / 24000.0
    );

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("glitch.wav", spec)?;
    for s in &samples {
        writer.write_sample(*s)?;
    }
    writer.finalize()?;
    eprintln!("Wrote glitch.wav");
    Ok(())
}
