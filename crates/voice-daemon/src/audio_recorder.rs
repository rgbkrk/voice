//! Audio recording utilities for persisting TTS output and mic input.

use hound::{WavSpec, WavWriter};
use std::path::{Path, PathBuf};

/// Audio storage directory.
pub fn audio_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".voice")
        .join("audio")
}

/// Path for question audio (TTS output).
pub fn question_path(queue_id: &str) -> PathBuf {
    audio_dir().join(format!("{}-q.wav", queue_id))
}

/// Path for answer audio (mic input).
pub fn answer_path(queue_id: &str) -> PathBuf {
    audio_dir().join(format!("{}-a.wav", queue_id))
}

/// Save audio samples to WAV file.
pub fn save_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    // Ensure directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
    }

    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)
        .map_err(|e| format!("create WAV {}: {}", path.display(), e))?;

    for &sample in samples {
        writer
            .write_sample(sample)
            .map_err(|e| format!("write sample: {}", e))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("finalize WAV: {}", e))?;

    Ok(())
}

/// Read audio samples from WAV file.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let reader =
        hound::WavReader::open(path).map_err(|e| format!("open WAV {}: {}", path.display(), e))?;

    let sample_rate = reader.spec().sample_rate;
    let samples: Result<Vec<f32>, _> = reader.into_samples::<f32>().collect();
    let samples = samples.map_err(|e| format!("read samples: {}", e))?;

    Ok((samples, sample_rate))
}

/// Delete audio files for a queue item.
pub fn delete_audio(queue_id: &str) -> Result<(), String> {
    let q_path = question_path(queue_id);
    let a_path = answer_path(queue_id);

    // Ignore errors if files don't exist
    let _ = std::fs::remove_file(q_path);
    let _ = std::fs::remove_file(a_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_and_read_wav() {
        let tmpdir = std::env::temp_dir();
        let path = tmpdir.join("test-audio.wav");

        // Generate 0.5s of 440Hz sine wave
        let sample_rate = 24000u32;
        let num_samples = (sample_rate as f32 * 0.5) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        // Save
        save_wav(&path, &samples, sample_rate).expect("save_wav failed");

        // Read back
        let (read_samples, read_rate) = read_wav(&path).expect("read_wav failed");

        assert_eq!(read_rate, sample_rate);
        assert_eq!(read_samples.len(), samples.len());

        // Check first few samples match (allow small floating-point error)
        for i in 0..10 {
            let diff = (read_samples[i] - samples[i]).abs();
            assert!(
                diff < 0.0001,
                "Sample {} mismatch: {} vs {}",
                i,
                read_samples[i],
                samples[i]
            );
        }

        // Cleanup
        std::fs::remove_file(path).ok();
    }
}
