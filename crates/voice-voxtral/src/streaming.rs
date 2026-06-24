use crate::{Result, VoxtralError};

/// vLLM-Omni's normal codec chunk size after the initial warm-up chunks.
pub const DEFAULT_CODEC_CHUNK_FRAMES: usize = 25;

/// vLLM-Omni emits smaller codec chunks while the request has not yet reached
/// `DEFAULT_CODEC_CHUNK_FRAMES` total frames.
pub const DEFAULT_CODEC_CHUNK_FRAMES_AT_BEGIN: usize = 5;

/// vLLM-Omni prepends this many prior frames as left context for the codec.
pub const DEFAULT_CODEC_LEFT_CONTEXT_FRAMES: usize = 25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxtralStreamingConfig {
    /// Normal chunk size in generated audio-code frames.
    pub chunk_frames: usize,
    /// Initial chunk size while the request has not yet reached `chunk_frames`.
    pub chunk_frames_at_begin: usize,
    /// Prior frames to include before the chunk frames for codec context.
    pub left_context_frames: usize,
}

impl Default for VoxtralStreamingConfig {
    fn default() -> Self {
        Self {
            chunk_frames: DEFAULT_CODEC_CHUNK_FRAMES,
            chunk_frames_at_begin: DEFAULT_CODEC_CHUNK_FRAMES_AT_BEGIN,
            left_context_frames: DEFAULT_CODEC_LEFT_CONTEXT_FRAMES,
        }
    }
}

impl VoxtralStreamingConfig {
    fn validate(&self) -> Result<()> {
        if self.chunk_frames == 0 {
            return Err(VoxtralError::InvalidConfig(
                "codec chunk_frames must be greater than zero".to_string(),
            ));
        }
        if self.chunk_frames_at_begin == 0 {
            return Err(VoxtralError::InvalidConfig(
                "codec chunk_frames_at_begin must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

/// One generator-to-codec streaming payload.
///
/// `frames` contains left-context frames followed by the new chunk frames, which
/// matches vLLM-Omni's `generator2tokenizer_async_chunk` payload after removing
/// transport-specific wrapper types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralCodecChunk {
    pub context_frames: usize,
    pub chunk_frames: usize,
    pub frames: Vec<Vec<u32>>,
    pub finished: bool,
}

impl VoxtralCodecChunk {
    /// Return the codec prompt shape used by vLLM-Omni:
    /// `[context_frames, chunk_frames, flattened_codebook_frames...]`.
    pub fn to_prompt_codes(&self) -> Vec<u32> {
        let frame_code_count: usize = self.frames.iter().map(Vec::len).sum();
        let mut codes = Vec::with_capacity(2 + frame_code_count);
        codes.push(self.context_frames as u32);
        codes.push(self.chunk_frames as u32);
        for frame in &self.frames {
            codes.extend(frame.iter().copied());
        }
        codes
    }
}

/// Plan the next generator-to-codec chunk from all generated code frames so far.
///
/// Returns `None` when streaming should wait for more frames. Returns an empty
/// finished chunk when the request ends before producing audio frames.
pub fn plan_codec_chunk(
    frames: &[Vec<u32>],
    emitted_frames: usize,
    config: VoxtralStreamingConfig,
    finished: bool,
) -> Result<Option<VoxtralCodecChunk>> {
    config.validate()?;

    let generated_frames = frames.len();
    if emitted_frames > generated_frames {
        return Err(VoxtralError::InvalidConfig(format!(
            "emitted_frames {emitted_frames} exceeds generated_frames {generated_frames}"
        )));
    }
    if generated_frames == 0 {
        if finished {
            return Ok(Some(VoxtralCodecChunk {
                context_frames: 0,
                chunk_frames: 0,
                frames: Vec::new(),
                finished: true,
            }));
        }
        return Ok(None);
    }
    if emitted_frames == generated_frames {
        return Ok(None);
    }

    let chunk_frames = if emitted_frames < config.chunk_frames {
        let remaining_initial_frames = config.chunk_frames - emitted_frames;
        config
            .chunk_frames_at_begin
            .min(remaining_initial_frames)
            .max(1)
    } else {
        config.chunk_frames
    };
    let available_frames = generated_frames - emitted_frames;
    if available_frames < chunk_frames && !finished {
        return Ok(None);
    }

    let current_chunk_frames = available_frames.min(chunk_frames);
    let end = emitted_frames + current_chunk_frames;
    let start = emitted_frames.saturating_sub(config.left_context_frames);
    let context_frames = emitted_frames - start;

    Ok(Some(VoxtralCodecChunk {
        context_frames,
        chunk_frames: current_chunk_frames,
        frames: frames[start..end].to_vec(),
        finished,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(count: usize) -> Vec<Vec<u32>> {
        (0..count)
            .map(|idx| vec![idx as u32, idx as u32 + 100, idx as u32 + 200])
            .collect()
    }

    #[test]
    fn waits_for_the_first_streaming_boundary() {
        let config = VoxtralStreamingConfig::default();

        assert_eq!(
            plan_codec_chunk(&frames(4), 0, config, false).unwrap(),
            None
        );

        let chunk = plan_codec_chunk(&frames(5), 0, config, false)
            .unwrap()
            .unwrap();
        assert_eq!(chunk.context_frames, 0);
        assert_eq!(chunk.chunk_frames, 5);
        assert_eq!(chunk.frames.len(), 5);
    }

    #[test]
    fn includes_left_context_for_followup_chunks() {
        let config = VoxtralStreamingConfig::default();

        let chunk = plan_codec_chunk(&frames(50), 25, config, false)
            .unwrap()
            .unwrap();

        assert_eq!(chunk.context_frames, 25);
        assert_eq!(chunk.chunk_frames, 25);
        assert_eq!(chunk.frames.len(), 50);
        assert_eq!(chunk.frames[0], vec![0, 100, 200]);
        assert_eq!(chunk.frames[49], vec![49, 149, 249]);
    }

    #[test]
    fn flushes_partial_final_chunks_with_context() {
        let config = VoxtralStreamingConfig::default();

        let chunk = plan_codec_chunk(&frames(7), 5, config, true)
            .unwrap()
            .unwrap();

        assert_eq!(chunk.context_frames, 5);
        assert_eq!(chunk.chunk_frames, 2);
        assert_eq!(chunk.frames.len(), 7);
        assert!(chunk.finished);
    }

    #[test]
    fn emits_empty_finished_marker() {
        let config = VoxtralStreamingConfig::default();

        let chunk = plan_codec_chunk(&[], 0, config, true).unwrap().unwrap();

        assert_eq!(chunk.context_frames, 0);
        assert_eq!(chunk.chunk_frames, 0);
        assert_eq!(chunk.frames, Vec::<Vec<u32>>::new());
        assert_eq!(chunk.to_prompt_codes(), vec![0, 0]);
    }

    #[test]
    fn flattens_prompt_codes_with_header() {
        let config = VoxtralStreamingConfig::default();
        let chunk = plan_codec_chunk(&frames(5), 0, config, false)
            .unwrap()
            .unwrap();

        assert_eq!(
            chunk.to_prompt_codes(),
            vec![0, 5, 0, 100, 200, 1, 101, 201, 2, 102, 202, 3, 103, 203, 4, 104, 204]
        );
    }

    #[test]
    fn rejects_zero_chunk_size() {
        let config = VoxtralStreamingConfig {
            chunk_frames: 0,
            ..VoxtralStreamingConfig::default()
        };

        assert!(plan_codec_chunk(&frames(1), 0, config, false).is_err());
    }

    #[test]
    fn finishes_non_divisor_begin_chunks_at_normal_boundary() {
        let config = VoxtralStreamingConfig {
            chunk_frames_at_begin: 3,
            ..VoxtralStreamingConfig::default()
        };

        let chunk = plan_codec_chunk(&frames(25), 24, config, false)
            .unwrap()
            .unwrap();

        assert_eq!(chunk.context_frames, 24);
        assert_eq!(chunk.chunk_frames, 1);
        assert_eq!(chunk.frames.len(), 25);
        assert!(!chunk.finished);
    }

    #[test]
    fn flushes_final_non_divisor_begin_chunk_without_mismatch() {
        let config = VoxtralStreamingConfig {
            chunk_frames_at_begin: 3,
            ..VoxtralStreamingConfig::default()
        };

        let chunk = plan_codec_chunk(&frames(40), 25, config, true)
            .unwrap()
            .unwrap();

        assert_eq!(chunk.context_frames, 25);
        assert_eq!(chunk.chunk_frames, 15);
        assert_eq!(chunk.frames.len(), 40);
        assert!(chunk.finished);
    }
}
