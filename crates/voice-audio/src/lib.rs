//! Audio container and codec helpers shared by the CLI and daemon.

use hound::{WavSpec, WavWriter};
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioOutputFormat {
    Wav,
    OggOpus,
}

impl AudioOutputFormat {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "wav" | "wave" => Some(Self::Wav),
            "ogg" | "opus" | "ogg-opus" | "ogg_opus" => Some(Self::OggOpus),
            _ => None,
        }
    }

    pub fn from_path(path: &Path) -> Option<Self> {
        let extension = path.extension()?.to_str()?;
        Self::from_name(extension)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::OggOpus => "ogg-opus",
        }
    }

    pub fn mime_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::OggOpus => "audio/ogg; codecs=opus",
        }
    }
}

impl fmt::Display for AudioOutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub fn save_audio(
    samples: &[f32],
    path: &Path,
    sample_rate: u32,
    format: AudioOutputFormat,
) -> Result<(), String> {
    match format {
        AudioOutputFormat::Wav => save_wav(samples, path, sample_rate),
        AudioOutputFormat::OggOpus => save_ogg_opus(samples, path, sample_rate),
    }
}

/// Adjust PCM duration by linear resampling while keeping the declared sample rate unchanged.
///
/// This is a lightweight playback/eval helper, not a pitch-preserving time-stretch
/// algorithm. Values above `1.0` shorten the sample buffer; values below `1.0`
/// lengthen it.
pub fn adjust_speed(samples: &[f32], speed: f32) -> Result<Vec<f32>, String> {
    if !speed.is_finite() || speed <= 0.0 {
        return Err(format!(
            "speed must be finite and greater than zero, got {speed}"
        ));
    }
    if samples.is_empty() || (speed - 1.0).abs() <= f32::EPSILON {
        return Ok(samples.to_vec());
    }

    let output_len = ((samples.len() as f64) / f64::from(speed)).round() as usize;
    let output_len = output_len.max(1);
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let source = index as f64 * f64::from(speed);
        let left = source.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let fraction = (source - left as f64) as f32;
        let sample = samples[left] * (1.0 - fraction) + samples[right] * fraction;
        output.push(sample);
    }
    Ok(output)
}

pub fn resolve_output_format(
    path: &Path,
    explicit: Option<AudioOutputFormat>,
) -> Result<AudioOutputFormat, String> {
    let inferred = AudioOutputFormat::from_path(path);
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());

    if let Some(explicit) = explicit {
        if let Some(inferred) = inferred {
            if inferred != explicit {
                return Err(format!(
                    "output extension .{} implies {}, but requested format is {}",
                    extension.as_deref().unwrap_or(""),
                    inferred,
                    explicit
                ));
            }
        } else if let Some(extension) = extension {
            return Err(format!(
                "unsupported output extension .{} for format {}; use .wav, .ogg, .opus, or omit the extension",
                extension, explicit
            ));
        }
        return Ok(explicit);
    }

    if let Some(inferred) = inferred {
        return Ok(inferred);
    }

    if let Some(extension) = extension {
        return Err(format!(
            "unsupported output extension .{}; use format wav or ogg-opus with .wav, .ogg, or .opus",
            extension
        ));
    }

    Ok(AudioOutputFormat::Wav)
}

pub fn save_wav(samples: &[f32], path: &Path, sample_rate: u32) -> Result<(), String> {
    ensure_parent_dir(path)?;

    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer =
        WavWriter::create(path, spec).map_err(|e| format!("create WAV {}: {e}", path.display()))?;

    for &sample in samples {
        writer
            .write_sample(sample)
            .map_err(|e| format!("write WAV sample: {e}"))?;
        writer
            .write_sample(sample)
            .map_err(|e| format!("write WAV sample: {e}"))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("finalize WAV {}: {e}", path.display()))?;

    Ok(())
}

pub fn save_ogg_opus(samples: &[f32], path: &Path, sample_rate: u32) -> Result<(), String> {
    ensure_parent_dir(path)?;

    let mut child = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-f")
        .arg("f32le")
        .arg("-ar")
        .arg(sample_rate.to_string())
        .arg("-ac")
        .arg("1")
        .arg("-i")
        .arg("pipe:0")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("48000")
        .arg("-c:a")
        .arg("libopus")
        .arg("-b:a")
        .arg("32k")
        .arg("-vbr")
        .arg("on")
        .arg("-application")
        .arg("voip")
        .arg("-f")
        .arg("ogg")
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn ffmpeg for Ogg/Opus output: {e}"))?;

    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| "open ffmpeg stdin".to_string())?;
        for sample in samples {
            stdin
                .write_all(&sample.clamp(-1.0, 1.0).to_le_bytes())
                .map_err(|e| format!("write PCM to ffmpeg: {e}"))?;
        }
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("wait for ffmpeg: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ffmpeg Ogg/Opus encode failed with {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    if !is_ogg_opus_file(path) {
        return Err(format!(
            "ffmpeg output is not an Ogg/Opus file: {}",
            path.display()
        ));
    }

    Ok(())
}

pub struct OggOpusStreamWriter {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    output_path: Option<PathBuf>,
}

impl OggOpusStreamWriter {
    pub fn create(path: &Path, input_sample_rate: u32) -> Result<Self, String> {
        let to_stdout = path.as_os_str() == "-";
        if !to_stdout {
            ensure_parent_dir(path)?;
        }

        let mut command = Command::new("ffmpeg");
        command
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-y")
            .arg("-f")
            .arg("s16le")
            .arg("-ar")
            .arg(input_sample_rate.to_string())
            .arg("-ac")
            .arg("1")
            .arg("-i")
            .arg("pipe:0")
            .arg("-ac")
            .arg("1")
            .arg("-ar")
            .arg("48000")
            .arg("-c:a")
            .arg("libopus")
            .arg("-b:a")
            .arg("32k")
            .arg("-vbr")
            .arg("on")
            .arg("-application")
            .arg("voip")
            .arg("-f")
            .arg("ogg");

        if to_stdout {
            command.arg("pipe:1").stdout(Stdio::inherit());
        } else {
            command.arg(path).stdout(Stdio::null());
        }

        let mut child = command
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawn ffmpeg for streamed Ogg/Opus output: {e}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "open ffmpeg stdin".to_string())?;

        Ok(Self {
            child: Some(child),
            stdin: Some(stdin),
            output_path: (!to_stdout).then(|| path.to_path_buf()),
        })
    }

    pub fn write_pcm_s16le(&mut self, bytes: &[u8]) -> Result<(), String> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| "ffmpeg stdin is closed".to_string())?;
        stdin
            .write_all(bytes)
            .map_err(|e| format!("write PCM to ffmpeg: {e}"))
    }

    pub fn finish(mut self) -> Result<(), String> {
        drop(self.stdin.take());
        let Some(child) = self.child.take() else {
            return Ok(());
        };
        let output = child
            .wait_with_output()
            .map_err(|e| format!("wait for ffmpeg: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "ffmpeg streamed Ogg/Opus encode failed with {}: {}",
                output.status,
                stderr.trim()
            ));
        }

        if let Some(path) = &self.output_path {
            if !is_ogg_opus_file(path) {
                return Err(format!(
                    "ffmpeg output is not an Ogg/Opus file: {}",
                    path.display()
                ));
            }
        }

        Ok(())
    }
}

impl Drop for OggOpusStreamWriter {
    fn drop(&mut self) {
        drop(self.stdin.take());
        if let Some(mut child) = self.child.take() {
            match child.try_wait() {
                Ok(Some(_)) => {}
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        }
    }
}

pub fn is_ogg_opus_file(path: &Path) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    bytes.starts_with(b"OggS") && bytes.windows(b"OpusHead".len()).any(|w| w == b"OpusHead")
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create output dir {}: {e}", parent.display()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn temp_path(extension: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "voice_audio_test_{}_{}.{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed"),
            extension
        ))
    }

    fn sine_wave(sample_rate: u32) -> Vec<f32> {
        let len = (sample_rate as f32 * 0.1) as usize;
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.25
            })
            .collect()
    }

    fn command_available(command: &str) -> bool {
        Command::new(command).arg("-version").output().is_ok()
    }

    fn ffprobe_audio_stream(path: &Path) -> Option<HashMap<String, String>> {
        if !command_available("ffprobe") {
            eprintln!("skipping ffprobe stream-shape check because ffprobe is not on PATH");
            return None;
        }

        let output = Command::new("ffprobe")
            .arg("-v")
            .arg("error")
            .arg("-select_streams")
            .arg("a:0")
            .arg("-show_entries")
            .arg("stream=codec_name,sample_rate,channels")
            .arg("-of")
            .arg("default=noprint_wrappers=1")
            .arg(path)
            .output()
            .expect("run ffprobe");

        assert!(
            output.status.success(),
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        Some(
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter_map(|line| {
                    let (key, value) = line.split_once('=')?;
                    Some((key.to_string(), value.to_string()))
                })
                .collect(),
        )
    }

    fn assert_ffprobe_ogg_opus_shape(path: &Path) {
        let Some(stream) = ffprobe_audio_stream(path) else {
            return;
        };
        assert_eq!(stream.get("codec_name").map(String::as_str), Some("opus"));
        assert_eq!(stream.get("sample_rate").map(String::as_str), Some("48000"));
        assert_eq!(stream.get("channels").map(String::as_str), Some("1"));
    }

    #[test]
    fn output_format_parses_names_and_extensions() {
        assert_eq!(
            AudioOutputFormat::from_name("ogg-opus"),
            Some(AudioOutputFormat::OggOpus)
        );
        assert_eq!(
            AudioOutputFormat::from_name("ogg"),
            Some(AudioOutputFormat::OggOpus)
        );
        assert_eq!(
            AudioOutputFormat::from_path(Path::new("reply.opus")),
            Some(AudioOutputFormat::OggOpus)
        );
        assert_eq!(
            AudioOutputFormat::from_path(Path::new("reply.wav")),
            Some(AudioOutputFormat::Wav)
        );
        assert_eq!(AudioOutputFormat::from_name("mp3"), None);
        assert_eq!(
            AudioOutputFormat::OggOpus.mime_type(),
            "audio/ogg; codecs=opus"
        );
    }

    #[test]
    fn resolve_output_format_rejects_misleading_extensions() {
        assert_eq!(
            resolve_output_format(Path::new("reply.ogg"), None).unwrap(),
            AudioOutputFormat::OggOpus
        );
        assert_eq!(
            resolve_output_format(Path::new("reply"), None).unwrap(),
            AudioOutputFormat::Wav
        );
        assert!(
            resolve_output_format(Path::new("reply.wav"), Some(AudioOutputFormat::OggOpus))
                .is_err()
        );
        assert!(resolve_output_format(Path::new("reply.mp3"), None).is_err());
    }

    #[test]
    fn save_wav_writes_riff_file() {
        let path = temp_path("wav");
        let samples = sine_wave(24_000);
        save_wav(&samples, &path, 24_000).expect("save wav");
        let bytes = std::fs::read(&path).expect("read wav");
        assert!(bytes.starts_with(b"RIFF"));
        assert_eq!(&bytes[8..12], b"WAVE");

        let mut reader = hound::WavReader::open(&path).expect("read saved wav");
        let spec = reader.spec();
        assert_eq!(spec.channels, 2);
        assert_eq!(spec.sample_rate, 24_000);
        assert_eq!(spec.bits_per_sample, 32);
        assert_eq!(spec.sample_format, hound::SampleFormat::Float);

        let decoded = reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .expect("decode saved wav samples");
        assert_eq!(decoded.len(), samples.len() * 2);
        for (frame, expected) in decoded.chunks_exact(2).zip(samples.iter().copied()) {
            assert_eq!(frame[0], expected);
            assert_eq!(frame[1], expected);
        }
        let rms = (decoded.iter().map(|sample| sample * sample).sum::<f32>()
            / decoded.len() as f32)
            .sqrt();
        assert!(rms > 0.0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn adjust_speed_keeps_identity_at_normal_speed() {
        let samples = vec![0.0, 0.25, 0.5, 0.25, 0.0];

        assert_eq!(adjust_speed(&samples, 1.0).unwrap(), samples);
    }

    #[test]
    fn adjust_speed_shortens_and_lengthens_sample_count() {
        let samples = vec![0.0; 100];

        assert_eq!(adjust_speed(&samples, 2.0).unwrap().len(), 50);
        assert_eq!(adjust_speed(&samples, 0.5).unwrap().len(), 200);
    }

    #[test]
    fn adjust_speed_rejects_invalid_speed() {
        assert!(adjust_speed(&[0.0], 0.0).is_err());
        assert!(adjust_speed(&[0.0], f32::NAN).is_err());
    }

    #[test]
    fn save_ogg_opus_writes_opus_mono_48khz() {
        if !command_available("ffmpeg") {
            eprintln!("skipping Ogg/Opus encode test because ffmpeg is not on PATH");
            return;
        }

        let path = temp_path("ogg");
        let samples = sine_wave(24_000);
        save_ogg_opus(&samples, &path, 24_000).expect("save ogg opus");
        assert!(is_ogg_opus_file(&path));
        assert_ffprobe_ogg_opus_shape(&path);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn stream_ogg_opus_writer_writes_opus_mono_48khz() {
        if !command_available("ffmpeg") {
            eprintln!("skipping streamed Ogg/Opus encode test because ffmpeg is not on PATH");
            return;
        }

        let path = temp_path("stream.ogg");
        let samples = sine_wave(24_000);
        let mut writer = OggOpusStreamWriter::create(&path, 24_000).expect("create stream writer");
        for chunk in samples.chunks(480) {
            let mut bytes = Vec::with_capacity(chunk.len() * 2);
            for sample in chunk {
                let clamped = sample.clamp(-1.0, 1.0);
                let pcm = if clamped >= 0.0 {
                    (clamped * i16::MAX as f32).round() as i16
                } else {
                    (clamped * 32768.0).round() as i16
                };
                bytes.extend_from_slice(&pcm.to_le_bytes());
            }
            writer.write_pcm_s16le(&bytes).expect("write stream pcm");
        }
        writer.finish().expect("finish stream writer");
        assert!(is_ogg_opus_file(&path));
        assert_ffprobe_ogg_opus_shape(&path);
        let _ = std::fs::remove_file(path);
    }
}
