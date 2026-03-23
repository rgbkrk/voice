use candle_core::{DType, Device, Tensor};

#[test]
fn istft_from_python_spec() {
    let device = Device::new_metal(0).unwrap();

    // Load Python's spec and phase
    let spec_npy = std::fs::read("/tmp/python_spec.npy").unwrap();
    let phase_npy = std::fs::read("/tmp/python_phase.npy").unwrap();

    let spec = load_npy_f32(&spec_npy, &device);
    let phase = load_npy_f32(&phase_npy, &device);

    eprintln!("spec: {:?}, phase: {:?}", spec.shape(), phase.shape());

    // Run our iSTFT
    let stft = voice_kokoro::istftnet::TorchSTFT::new(20, 5, 20);
    let audio = stft.inverse(&spec, &phase).unwrap();

    eprintln!("audio: {:?}", audio.shape());
    let samples = audio.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    eprintln!(
        "audio len: {}, RMS: {:.6}",
        samples.len(),
        (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
    );

    // Load Python's iSTFT output for comparison
    let py_wav = std::fs::read("/tmp/python_fox_raw.wav").unwrap();
    // Skip WAV header (44 bytes for simple WAV)
    // Actually let's just check the sample count matches
    eprintln!("Our samples: {}, expected: ~66600", samples.len());

    // Write to WAV for listening/whisper test
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("/tmp/rust_istft_pyspec.wav", spec).unwrap();
    for &s in &samples {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
    eprintln!("Wrote /tmp/rust_istft_pyspec.wav");

    assert!(samples.len() > 60000, "Too few samples: {}", samples.len());
}

fn load_npy_f32(data: &[u8], device: &Device) -> Tensor {
    assert_eq!(&data[..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
    let header = std::str::from_utf8(&data[10..10 + header_len]).unwrap();
    let shape_start = header.find("'shape': (").unwrap() + 10;
    let shape_end = header[shape_start..].find(')').unwrap() + shape_start;
    let shape_str = &header[shape_start..shape_end];
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let data_start = 10 + header_len;
    let float_data: &[f32] = bytemuck::cast_slice(&data[data_start..]);
    Tensor::from_vec(float_data.to_vec(), shape.as_slice(), device).unwrap()
}
