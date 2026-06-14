//! Transport-neutral streaming audio events.
//!
//! The daemon and clients use these types to describe TTS as ordered audio
//! frames instead of only completed files or local playback. The payload is
//! signed 16-bit mono PCM so it can be written directly to a raw stream,
//! wrapped in WAV, or fed to an Opus/WebRTC adapter.

use serde::{Deserialize, Serialize};

pub const DEFAULT_FRAME_MS: u32 = 20;
pub const PCM_S16LE_BYTES_PER_SAMPLE: usize = 2;

/// WebRTC-friendly local PCM contract used by Hermes/WhatsApp sidecar work.
///
/// WebRTC carries Opus over RTP on the wire, but the local `voice` boundary
/// stays as signed 16-bit little-endian mono PCM so callers can choose whether
/// to feed frames to a sidecar, an Opus encoder, or test fixtures.
pub const WEBRTC_SAMPLE_RATE: u32 = 48_000;
pub const WEBRTC_CHANNELS: u16 = 1;
pub const WEBRTC_FRAME_MS: u32 = DEFAULT_FRAME_MS;
pub const WEBRTC_SAMPLES_PER_FRAME: usize =
    WEBRTC_SAMPLE_RATE as usize * WEBRTC_FRAME_MS as usize / 1_000;
pub const WEBRTC_FRAME_BYTES: usize =
    WEBRTC_SAMPLES_PER_FRAME * WEBRTC_CHANNELS as usize * PCM_S16LE_BYTES_PER_SAMPLE;
pub const WEBRTC_DEFAULT_DRAIN_BYTES: usize = WEBRTC_FRAME_BYTES * 50;
pub const WEBRTC_MAX_OUTBOUND_QUEUE_BYTES: usize =
    WEBRTC_SAMPLE_RATE as usize * WEBRTC_CHANNELS as usize * PCM_S16LE_BYTES_PER_SAMPLE * 10;
pub const WEBRTC_MAX_INBOUND_QUEUE_BYTES: usize =
    WEBRTC_SAMPLE_RATE as usize * WEBRTC_CHANNELS as usize * PCM_S16LE_BYTES_PER_SAMPLE * 10;
pub const WEBRTC_MAX_DRAIN_WAIT_MS: u32 = 5_000;

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
        let frame_samples = samples_per_frame(sample_rate, frame_ms);
        Self {
            stream_id: stream_id.into(),
            sample_rate,
            channels: 1,
            frame_ms,
            samples_per_frame: frame_samples,
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

#[derive(Debug, Clone)]
pub struct InterleavedMonoMixer {
    channels: usize,
    pending_sum: f32,
    pending_count: usize,
}

impl InterleavedMonoMixer {
    pub fn new(channels: usize) -> Self {
        Self {
            channels: channels.max(1),
            pending_sum: 0.0,
            pending_count: 0,
        }
    }

    pub fn push(&mut self, sample: f32) -> Option<f32> {
        if self.channels == 1 {
            return Some(sample);
        }

        self.pending_sum += sample;
        self.pending_count += 1;

        if self.pending_count == self.channels {
            let mixed = self.pending_sum / self.channels as f32;
            self.pending_sum = 0.0;
            self.pending_count = 0;
            Some(mixed)
        } else {
            None
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

pub fn pcm_s16le_frames(samples: &[f32], sample_rate: u32, frame_ms: u32) -> Vec<Vec<i16>> {
    let frame_samples = samples_per_frame(sample_rate, frame_ms);
    samples
        .chunks(frame_samples)
        .map(|chunk| chunk.iter().copied().map(f32_to_i16).collect())
        .collect()
}

pub fn samples_per_frame(sample_rate: u32, frame_ms: u32) -> usize {
    ((u64::from(sample_rate.max(1)) * u64::from(frame_ms.max(1))) / 1_000).max(1) as usize
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
    fn pcm_s16le_frames_splits_by_frame_duration() {
        let samples = vec![0.0; 1_000];
        let frames = pcm_s16le_frames(&samples, 1_000, 20);

        assert_eq!(frames.len(), 50);
        assert!(frames.iter().all(|frame| frame.len() == 20));
    }

    #[test]
    fn pcm_s16le_frames_keeps_short_final_frame() {
        let samples = vec![0.0; 25];
        let frames = pcm_s16le_frames(&samples, 1_000, 20);

        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].len(), 20);
        assert_eq!(frames[1].len(), 5);
    }

    #[test]
    fn samples_per_frame_uses_minimum_one_sample() {
        assert_eq!(samples_per_frame(0, 0), 1);
        assert_eq!(samples_per_frame(1, 1), 1);
    }

    #[test]
    fn webrtc_pcm_constants_describe_twenty_ms_mono_frames() {
        assert_eq!(WEBRTC_SAMPLE_RATE, 48_000);
        assert_eq!(WEBRTC_CHANNELS, 1);
        assert_eq!(WEBRTC_FRAME_MS, 20);
        assert_eq!(WEBRTC_SAMPLES_PER_FRAME, 960);
        assert_eq!(WEBRTC_FRAME_BYTES, 1_920);
        assert_eq!(
            WEBRTC_SAMPLES_PER_FRAME,
            samples_per_frame(WEBRTC_SAMPLE_RATE, WEBRTC_FRAME_MS)
        );
    }

    #[test]
    fn webrtc_sidecar_json_contract_matches_stream_constants() {
        let contract_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/contracts/webrtc-sidecar-v1.json");
        let bytes = std::fs::read(&contract_path).unwrap_or_else(|err| {
            panic!("read {}: {err}", contract_path.display());
        });
        let contract: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_else(|err| {
            panic!("parse {}: {err}", contract_path.display());
        });

        assert_eq!(contract["contract"], "voice.webrtc_sidecar");
        assert_eq!(contract["version"], 1);

        let audio = &contract["audio"];
        assert_eq!(audio["sample_rate"], WEBRTC_SAMPLE_RATE);
        assert_eq!(audio["channels"], WEBRTC_CHANNELS);
        assert_eq!(audio["frame_ms"], WEBRTC_FRAME_MS);
        assert_eq!(audio["encoding"], "pcm_s16le");
        assert_eq!(audio["bytes_per_sample"], PCM_S16LE_BYTES_PER_SAMPLE);
        assert_eq!(audio["samples_per_frame"], WEBRTC_SAMPLES_PER_FRAME);
        assert_eq!(audio["frame_bytes"], WEBRTC_FRAME_BYTES);
        assert_eq!(audio["default_drain_bytes"], WEBRTC_DEFAULT_DRAIN_BYTES);
        assert_eq!(
            audio["max_outbound_queue_bytes"],
            WEBRTC_MAX_OUTBOUND_QUEUE_BYTES
        );
        assert_eq!(
            audio["max_inbound_queue_bytes"],
            WEBRTC_MAX_INBOUND_QUEUE_BYTES
        );
        assert_eq!(audio["max_drain_wait_ms"], WEBRTC_MAX_DRAIN_WAIT_MS);
    }

    #[test]
    fn interleaved_mono_mixer_passes_mono_samples_through() {
        let mut mixer = InterleavedMonoMixer::new(1);
        assert_eq!(mixer.push(0.25), Some(0.25));
        assert_eq!(mixer.push(-0.5), Some(-0.5));
    }

    #[test]
    fn interleaved_mono_mixer_averages_stereo_frames() {
        let mut mixer = InterleavedMonoMixer::new(2);
        let input = [1.0, -1.0, 0.5, 0.25];
        let output: Vec<f32> = input
            .into_iter()
            .filter_map(|sample| mixer.push(sample))
            .collect();

        assert_eq!(output, vec![0.0, 0.375]);
    }

    #[test]
    fn interleaved_mono_mixer_treats_zero_channels_as_mono() {
        let mut mixer = InterleavedMonoMixer::new(0);
        assert_eq!(mixer.push(0.75), Some(0.75));
    }

    #[test]
    fn interleaved_mono_mixer_holds_incomplete_frame() {
        let mut mixer = InterleavedMonoMixer::new(2);
        assert_eq!(mixer.push(1.0), None);
    }

    #[test]
    fn resample_linear_doubles_sample_count() {
        let out = resample_linear(&[0.0, 1.0, 0.0], 24_000, 48_000);
        assert_eq!(out.len(), 6);
        assert!((out[1] - 0.5).abs() < 0.001);
    }
}
