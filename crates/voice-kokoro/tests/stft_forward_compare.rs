use candle_core::{Device, Tensor};

#[test]
#[ignore]
fn compare_stft_forward_with_python() {
    let device = Device::new_metal(0).unwrap();

    let x_npy = std::fs::read("/tmp/stft_test_x.npy").unwrap();
    let mag_npy = std::fs::read("/tmp/stft_test_mag.npy").unwrap();
    let phase_npy = std::fs::read("/tmp/stft_test_phase.npy").unwrap();

    let x = load_npy_f32(&x_npy, &device);
    let expected_mag = load_npy_f32(&mag_npy, &Device::Cpu);
    let expected_phase = load_npy_f32(&phase_npy, &Device::Cpu);

    let stft = voice_kokoro::istftnet::TorchSTFT::new(20, 5, 20, &device, candle_core::DType::F32)
        .unwrap();
    let (mag, phase) = stft.transform(&x).unwrap();

    let mag_cpu = mag.to_device(&Device::Cpu).unwrap();
    let phase_cpu = phase.to_device(&Device::Cpu).unwrap();

    // Compare magnitudes
    let our_mag = mag_cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let exp_mag = expected_mag
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mag_mse: f32 = our_mag
        .iter()
        .zip(exp_mag.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / our_mag.len() as f32;

    eprintln!(
        "Mag shapes: ours {:?}, Python {:?}",
        mag_cpu.shape(),
        expected_mag.shape()
    );
    eprintln!("Mag[0,:5,0] ours:   {:?}", &our_mag[..5]);
    eprintln!("Mag[0,:5,0] Python: {:?}", &exp_mag[..5]);
    eprintln!("Mag MSE: {mag_mse:.8}");

    // Compare phases
    let our_phase = phase_cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let exp_phase = expected_phase
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let phase_mse: f32 = our_phase
        .iter()
        .zip(exp_phase.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / our_phase.len() as f32;

    eprintln!("Phase[0,:5,0] ours:   {:?}", &our_phase[..5]);
    eprintln!("Phase[0,:5,0] Python: {:?}", &exp_phase[..5]);
    eprintln!("Phase MSE: {phase_mse:.8}");

    assert!(mag_mse < 0.01, "Magnitude MSE too high: {mag_mse}");
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
