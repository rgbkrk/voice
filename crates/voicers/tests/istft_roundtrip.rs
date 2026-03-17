use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;

#[test]
fn test_istft_roundtrip() {
    let sr = 24000.0f32;
    let freq = 440.0f32;
    let n_samples = 12000; // 0.5s
    let signal: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5)
        .collect();
    let signal_arr = Array::from_slice(&signal, &[n_samples as i32]);

    let n_fft = 20;
    let hop = 5;

    let stft_result =
        voicers::dsp::stft(&signal_arr, Some(n_fft), Some(hop), Some(n_fft), Some(true)).unwrap();
    let mag = stft_result.abs().unwrap();
    let phase = voicers::dsp::mlx_angle(&stft_result).unwrap();

    let mag_t = mag.transpose_axes(&[1, 0]).unwrap();
    let phase_t = phase.transpose_axes(&[1, 0]).unwrap();
    let mag_b = mag_t
        .reshape(&[1, mag_t.shape()[0], mag_t.shape()[1]])
        .unwrap();
    let phase_b = phase_t
        .reshape(&[1, phase_t.shape()[0], phase_t.shape()[1]])
        .unwrap();

    let stft_obj = voicers::dsp::MlxStft::new(n_fft, hop, n_fft).unwrap();
    let reconstructed = stft_obj.inverse(&mag_b, &phase_b).unwrap();
    reconstructed.eval().unwrap();

    let recon = reconstructed.index(0);
    recon.eval().unwrap();
    let recon_data: &[f32] = recon.as_slice();

    let orig_peak: f32 = signal.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let recon_peak: f32 = recon_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    eprintln!(
        "Original: {} samples, peak: {:.4}",
        signal.len(),
        orig_peak
    );
    eprintln!(
        "Reconstructed: {} samples, peak: {:.4}",
        recon_data.len(),
        recon_peak
    );

    // Print first 20 samples for debugging
    let n_compare = 20.min(recon_data.len());
    for i in 0..n_compare {
        eprintln!(
            "  [{:3}] orig={:+.6}, recon={:+.6}",
            i, signal[i], recon_data[i]
        );
    }

    // Peak should be close
    assert!(
        (orig_peak - recon_peak).abs() < 0.05,
        "Peak mismatch: orig={}, recon={}",
        orig_peak,
        recon_peak
    );
}
