//! Transport-neutral streaming audio events.
//!
//! The daemon and clients use these types to describe TTS as ordered audio
//! frames instead of only completed files or local playback. The payload is
//! signed 16-bit mono PCM so it can be written directly to a raw stream,
//! wrapped in WAV, or fed to an Opus/WebRTC adapter.

use serde::{Deserialize, Serialize};

pub const DEFAULT_FRAME_MS: u32 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioEncoding {
    PcmS16Le,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamMetadata {
    pub stream_id: String,
    pub sample_rate: u32,
    pub source_sample_rate: u32,
    pub channels: u16,
    pub encoding: AudioEncoding,
    pub frame_ms: u32,
    pub voice: Option<String>,
    pub speed: f32,
    pub total_phoneme_chunks: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioFrame {
    pub stream_id: String,
    pub sequence: u64,
    pub chunk_index: u32,
    pub offset_samples: u64,
    pub timestamp_ms: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub encoding: AudioEncoding,
    pub frame_ms: u32,
    pub sample_count: usize,
    pub padding_samples: usize,
    pub samples: Vec<i16>,
}

impl AudioFrame {
    pub fn payload_le_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.samples.len() * 2);
        for sample in &self.samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamEnded {
    pub stream_id: String,
    pub frames: u64,
    pub samples: u64,
    pub duration_ms: u64,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamError {
    pub stream_id: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamCancelled {
    pub stream_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TtsStreamEvent {
    Started { metadata: StreamMetadata },
    Audio { frame: AudioFrame },
    Ended(StreamEnded),
    Error(StreamError),
    Cancelled(StreamCancelled),
}

impl TtsStreamEvent {
    pub fn event_name(&self) -> &'static str {
        match self {
            Self::Started { .. } => "tts.started",
            Self::Audio { .. } => "tts.audio",
            Self::Ended(_) => "tts.ended",
            Self::Error(_) => "tts.error",
            Self::Cancelled(_) => "tts.cancelled",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Ended(_) | Self::Error(_) | Self::Cancelled(_))
    }

    pub fn cancelled(stream_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Cancelled(StreamCancelled {
            stream_id: stream_id.into(),
            reason: reason.into(),
        })
    }

    pub fn error(stream_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Error(StreamError {
            stream_id: stream_id.into(),
            message: message.into(),
        })
    }
}

/// Fixed-duration packetizer for mono f32 PCM.
///
/// `push_samples` emits only full frames. `finish` emits the remaining tail,
/// padded with silence to a full frame, which keeps the output Opus-friendly.
#[derive(Debug, Clone)]
pub struct Packetizer {
    stream_id: String,
    sample_rate: u32,
    channels: u16,
    frame_ms: u32,
    samples_per_frame: usize,
    next_sequence: u64,
    offset_samples: u64,
    pending: Vec<f32>,
}

impl Packetizer {
    pub fn new(stream_id: impl Into<String>, sample_rate: u32, frame_ms: u32) -> Self {
        let sample_rate = sample_rate.max(1);
        let frame_ms = frame_ms.max(1);
        let samples_per_frame = ((sample_rate as u64 * frame_ms as u64) / 1_000).max(1) as usize;
        Self {
            stream_id: stream_id.into(),
            sample_rate,
            channels: 1,
            frame_ms,
            samples_per_frame,
            next_sequence: 0,
            offset_samples: 0,
            pending: Vec::new(),
        }
    }

    pub fn samples_per_frame(&self) -> usize {
        self.samples_per_frame
    }

    pub fn frames_emitted(&self) -> u64 {
        self.next_sequence
    }

    pub fn samples_emitted(&self) -> u64 {
        self.offset_samples
    }

    pub fn push_samples(&mut self, chunk_index: u32, samples: &[f32]) -> Vec<AudioFrame> {
        self.pending.extend_from_slice(samples);
        let mut frames = Vec::new();

        while self.pending.len() >= self.samples_per_frame {
            let frame_samples: Vec<f32> = self.pending.drain(..self.samples_per_frame).collect();
            frames.push(self.make_frame(chunk_index, frame_samples, 0));
        }

        frames
    }

    pub fn finish(&mut self, chunk_index: u32) -> Option<AudioFrame> {
        if self.pending.is_empty() {
            return None;
        }

        let padding = self.samples_per_frame - self.pending.len();
        let mut frame_samples = std::mem::take(&mut self.pending);
        frame_samples.resize(self.samples_per_frame, 0.0);
        Some(self.make_frame(chunk_index, frame_samples, padding))
    }

    fn make_frame(
        &mut self,
        chunk_index: u32,
        samples: Vec<f32>,
        padding_samples: usize,
    ) -> AudioFrame {
        let sequence = self.next_sequence;
        let offset_samples = self.offset_samples;
        self.next_sequence += 1;
        self.offset_samples += samples.len() as u64;

        AudioFrame {
            stream_id: self.stream_id.clone(),
            sequence,
            chunk_index,
            offset_samples,
            timestamp_ms: offset_samples.saturating_mul(1_000) / self.sample_rate as u64,
            sample_rate: self.sample_rate,
            channels: self.channels,
            encoding: AudioEncoding::PcmS16Le,
            frame_ms: self.frame_ms,
            sample_count: samples.len(),
            padding_samples,
            samples: samples.into_iter().map(f32_to_i16).collect(),
        }
    }
}

pub fn f32_to_i16(sample: f32) -> i16 {
    let clamped = sample.clamp(-1.0, 1.0);
    if clamped >= 0.0 {
        (clamped * i16::MAX as f32).round() as i16
    } else {
        (clamped * 32768.0).round() as i16
    }
}

pub fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == 0 || target_rate == 0 {
        return Vec::new();
    }
    if source_rate == target_rate {
        return samples.to_vec();
    }
    if samples.len() == 1 {
        return vec![samples[0]];
    }

    let output_len =
        (samples.len() as u64 * target_rate as u64).div_ceil(source_rate as u64) as usize;
    let scale = source_rate as f64 / target_rate as f64;

    (0..output_len)
        .map(|i| {
            let pos = i as f64 * scale;
            let left = pos.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = (pos - left as f64) as f32;
            samples[left] * (1.0 - frac) + samples[right] * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packetizer_emits_full_frames_and_pads_tail() {
        let mut packetizer = Packetizer::new("s", 24_000, 20);
        assert_eq!(packetizer.samples_per_frame(), 480);

        let frames = packetizer.push_samples(0, &vec![0.25; 1_000]);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].sequence, 0);
        assert_eq!(frames[0].offset_samples, 0);
        assert_eq!(frames[1].sequence, 1);
        assert_eq!(frames[1].offset_samples, 480);
        assert_eq!(frames[1].timestamp_ms, 20);

        let tail = packetizer.finish(0).unwrap();
        assert_eq!(tail.sequence, 2);
        assert_eq!(tail.sample_count, 480);
        assert_eq!(tail.padding_samples, 440);
    }

    #[test]
    fn converts_f32_to_i16_with_clamping() {
        assert_eq!(f32_to_i16(0.0), 0);
        assert_eq!(f32_to_i16(1.0), 32767);
        assert_eq!(f32_to_i16(-1.0), -32768);
        assert_eq!(f32_to_i16(2.0), 32767);
        assert_eq!(f32_to_i16(-2.0), -32768);
    }

    #[test]
    fn resample_linear_doubles_sample_count() {
        let out = resample_linear(&[0.0, 1.0, 0.0], 24_000, 48_000);
        assert_eq!(out.len(), 6);
        assert!((out[1] - 0.5).abs() < 0.001);
    }
}
