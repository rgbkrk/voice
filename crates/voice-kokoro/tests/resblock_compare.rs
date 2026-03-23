use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use voice_kokoro::istftnet::AdaINResBlock1;

#[test]
#[ignore]
fn compare_resblock_with_python() {
    let device = Device::new_metal(0).unwrap();

    // Load test tensors saved from Python
    let x_npy = std::fs::read("/tmp/rb_test_x.npy").unwrap();
    let s_npy = std::fs::read("/tmp/rb_test_s.npy").unwrap();
    let out_npy = std::fs::read("/tmp/rb_test_out.npy").unwrap();
    let adain0_npy = std::fs::read("/tmp/rb_test_adain0.npy").unwrap();

    let x = load_npy_f32(&x_npy, &device);
    let s = load_npy_f32(&s_npy, &device);
    let expected_out = load_npy_f32(&out_npy, &Device::Cpu);
    let expected_adain0 = load_npy_f32(&adain0_npy, &Device::Cpu);

    eprintln!("x: {:?}, s: {:?}", x.shape(), s.shape());

    // Load resblock weights from safetensors
    let weights_path = find_weights();
    let weights_data = std::fs::read(&weights_path).unwrap();
    let vb = VarBuilder::from_buffered_safetensors(weights_data, DType::F32, &device).unwrap();

    let rb0 = AdaINResBlock1::load(
        256,
        3,
        &[1, 3, 5],
        128,
        vb.pp("decoder.generator.resblocks.0"),
    )
    .unwrap();

    // Run forward
    let out = rb0.forward(&x, &s).unwrap();
    let out_cpu = out.to_device(&Device::Cpu).unwrap();

    let out_vec = out_cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let exp_vec = expected_out
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let rms_out = rms(&out_vec);
    let rms_exp = rms(&exp_vec);
    eprintln!("Rust rb0 RMS: {rms_out:.6}");
    eprintln!("Python rb0 RMS: {rms_exp:.6}");

    // Compare element-wise
    let len = out_vec.len().min(exp_vec.len());
    let mse: f32 = out_vec[..len]
        .iter()
        .zip(exp_vec[..len].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / len as f32;
    let max_err: f32 = out_vec[..len]
        .iter()
        .zip(exp_vec[..len].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("MSE: {mse:.8}");
    eprintln!("Max error: {max_err:.6}");

    assert!(mse < 0.01, "Resblock MSE too high: {mse}");
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

fn load_npy_f32(data: &[u8], device: &Device) -> Tensor {
    // Simple NPY parser for f32 arrays
    // NPY format: magic + version + header_len + header + data
    assert_eq!(&data[..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
    let header = std::str::from_utf8(&data[10..10 + header_len]).unwrap();

    // Parse shape from header
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

fn find_weights() -> String {
    let home = std::env::var("HOME").unwrap();
    let cache =
        format!("{home}/.cache/huggingface/hub/models--prince-canuma--Kokoro-82M/snapshots");
    let snapshot = std::fs::read_dir(&cache)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    snapshot
        .join("kokoro-v1_0.safetensors")
        .to_string_lossy()
        .to_string()
}
