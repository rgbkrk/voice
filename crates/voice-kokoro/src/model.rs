//! KModel — the top-level Kokoro TTS model.
//!
//! Ported from kokoro/model.py

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

use crate::albert::CustomAlbert;
use crate::config::ModelConfig;
use crate::istftnet::Decoder;
use crate::modules::{ProsodyPredictor, TextEncoder};

/// The top-level Kokoro-82M model.
pub struct KModel {
    bert: CustomAlbert,
    bert_encoder: nn::Linear,
    predictor: ProsodyPredictor,
    text_encoder: TextEncoder,
    decoder: Decoder,
    context_length: usize,
}

impl KModel {
    /// Load model weights from a VarBuilder (typically backed by safetensors).
    pub fn load(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let bert = CustomAlbert::load(&config.plbert, config.n_token, vb.pp("bert"))?;

        let bert_encoder = nn::linear(
            config.plbert.hidden_size,
            config.hidden_dim,
            vb.pp("bert_encoder"),
        )?;

        let predictor = ProsodyPredictor::load(
            config.style_dim,
            config.hidden_dim,
            config.n_layer,
            config.max_dur,
            vb.pp("predictor"),
        )?;

        let text_encoder = TextEncoder::load(
            config.hidden_dim,
            config.text_encoder_kernel_size,
            config.n_layer,
            config.n_token,
            vb.pp("text_encoder"),
        )?;

        let istft = &config.istftnet;
        let decoder = Decoder::load(
            config.hidden_dim,
            config.style_dim,
            config.n_mels,
            &istft.resblock_kernel_sizes,
            &istft.upsample_rates,
            istft.upsample_initial_channel,
            &istft.resblock_dilation_sizes,
            &istft.upsample_kernel_sizes,
            istft.gen_istft_n_fft,
            istft.gen_istft_hop_size,
            vb.pp("decoder"),
        )?;

        Ok(Self {
            bert,
            bert_encoder,
            predictor,
            text_encoder,
            decoder,
            context_length: config.plbert.max_position_embeddings,
        })
    }

    /// Run inference.
    ///
    /// * `input_ids` - phoneme token IDs (without BOS/EOS padding; the model adds [0, ..., 0])
    /// * `ref_s` - reference style tensor [1, 256] (first 128 = acoustic style, last 128 = prosody style)
    /// * `speed` - speed factor (1.0 = normal)
    ///
    /// Returns: audio waveform tensor [samples]
    pub fn forward(
        &self,
        input_ids: &[i64],
        ref_s: &Tensor,
        speed: f32,
        device: &Device,
    ) -> Result<Tensor> {
        // Pad with BOS=0 and EOS=0
        let mut padded = Vec::with_capacity(input_ids.len() + 2);
        padded.push(0i64);
        padded.extend_from_slice(input_ids);
        padded.push(0i64);

        let seq_len = padded.len();
        assert!(
            seq_len <= self.context_length,
            "Input too long: {} > {}",
            seq_len,
            self.context_length
        );

        let input_ids_t = Tensor::new(&padded[..], device)?.unsqueeze(0)?; // [1, T]
        let input_lengths = Tensor::new(&[seq_len as i64][..], device)?; // [1]

        // Text mask: [1, T], true where position >= length (i.e., padding)
        // Python: mask = torch.gt(mask+1, lengths.unsqueeze(1))
        // For single-sequence inference with no padding, this is all false (0).
        // For single-sequence inference with no padding, mask is all false
        let mask_data: Vec<f32> = (0..seq_len)
            .map(|i| if i + 1 > seq_len { 1.0 } else { 0.0 })
            .collect();
        let text_mask = Tensor::new(&mask_data[..], device)?.unsqueeze(0)?; // [1, T]
        let text_mask = text_mask.gt(0.5)?; // [1, T] u8
        let text_mask_f = text_mask.to_dtype(DType::F32)?;

        // BERT: attention_mask is 1=attend, 0=pad (inverse of text_mask)
        let attention_mask = (1.0 - &text_mask_f)?;

        // ALBERT forward
        let bert_dur = self.bert.forward(
            &input_ids_t.to_dtype(DType::U32)?,
            &attention_mask.to_dtype(DType::F32)?,
        )?; // [1, T, hidden_size]

        // bert_encoder: project hidden_size -> hidden_dim
        let d_en = self.bert_encoder.forward(&bert_dur)?; // [1, T, hidden_dim]
        let d_en = d_en.transpose(1, 2)?; // [1, hidden_dim, T]

        // Style split: ref_s[:, 128:] = prosody style, ref_s[:, :128] = acoustic style
        let s = ref_s.narrow(1, 128, 128)?; // [1, 128] prosody style

        // Duration encoder
        let d = self
            .predictor
            .text_encoder
            .forward(&d_en, &s, &input_lengths, &text_mask)?; // [1, T, hidden_dim]

        // Duration LSTM: needs [1, T, hidden_dim + style_dim]
        let (b, t_dur, _) = d.dims3()?;
        let s_exp = s.unsqueeze(1)?.expand(&[b, t_dur, 128])?;
        let d_cat = Tensor::cat(&[&d, &s_exp], 2)?; // [1, T, hidden_dim + 128]
        let x = self.predictor.lstm.forward(&d_cat)?; // [1, T, hidden_dim]

        // Duration projection
        let duration = self.predictor.duration_proj.forward(&x)?; // [1, T, max_dur]
        let duration = candle_nn::ops::sigmoid(&duration)?;
        let duration = duration.sum(2)?; // [1, T]
        let speed_t = Tensor::new(speed, device)?.to_dtype(duration.dtype())?;
        let duration = duration.broadcast_div(&speed_t)?;
        let pred_dur = duration.round()?.clamp(1.0f32, f32::MAX)?;

        // Build alignment matrix on GPU via repeat_interleave-style expansion.
        //
        // The alignment matrix is a [T, total_frames] binary matrix where each
        // token row has 1s in the frame columns it occupies. Instead of building
        // this with a CPU loop (which requires GPU→CPU→GPU round-trip), we:
        //   1. Download durations once (small: ~100 i64 values)
        //   2. Build per-row one-hot slices and concatenate on GPU
        //
        // For typical phoneme counts (50-150 tokens), the duration download is
        // ~400 bytes — negligible vs the old approach which also downloaded them.
        let pred_dur_vec: Vec<i64> = pred_dur
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
            .iter()
            .map(|&v| v.max(1.0) as i64)
            .collect();

        let total_frames: usize = pred_dur_vec.iter().sum::<i64>() as usize;

        // Build alignment rows on GPU: each row is [0..0, 1..1, 0..0]
        let mut rows: Vec<Tensor> = Vec::with_capacity(seq_len);
        let mut frame_offset = 0usize;
        for (token_idx, &dur) in pred_dur_vec.iter().enumerate() {
            let dur = dur as usize;
            if token_idx < seq_len && dur > 0 && frame_offset + dur <= total_frames {
                // Build row: zeros(frame_offset) | ones(dur) | zeros(remaining)
                let remaining = total_frames - frame_offset - dur;
                let row = if frame_offset == 0 && remaining == 0 {
                    Tensor::ones((1, total_frames), DType::F32, device)?
                } else {
                    let mut parts: Vec<Tensor> = Vec::new();
                    if frame_offset > 0 {
                        parts.push(Tensor::zeros((1, frame_offset), DType::F32, device)?);
                    }
                    parts.push(Tensor::ones((1, dur), DType::F32, device)?);
                    if remaining > 0 {
                        parts.push(Tensor::zeros((1, remaining), DType::F32, device)?);
                    }
                    Tensor::cat(&parts, 1)?
                };
                rows.push(row);
            } else {
                rows.push(Tensor::zeros((1, total_frames), DType::F32, device)?);
            }
            frame_offset += dur;
        }
        let pred_aln_trg = Tensor::cat(&rows, 0)?.unsqueeze(0)?; // [1, T, total_frames]

        // en = d.transpose(-1,-2) @ pred_aln_trg
        // d is [1, T, hidden_dim] -> transpose to [1, hidden_dim, T]
        let d_t = d.transpose(1, 2)?;
        let en = d_t.matmul(&pred_aln_trg)?; // [1, hidden_dim, total_frames]

        // F0 and N prediction
        let (f0_pred, n_pred) = self.predictor.f0_n_train(&en, &s)?;

        // Text encoder
        let t_en = self.text_encoder.forward(
            &input_ids_t.to_dtype(DType::U32)?,
            &input_lengths,
            &text_mask,
        )?; // [1, hidden_dim, T]

        // asr = t_en @ pred_aln_trg
        let asr = t_en.matmul(&pred_aln_trg)?; // [1, hidden_dim, total_frames]

        // Decoder
        let s_acoustic = ref_s.narrow(1, 0, 128)?; // [1, 128]
        let audio = self.decoder.forward(&asr, &f0_pred, &n_pred, &s_acoustic)?;

        // audio: [1, 1, samples] -> [samples]
        audio.squeeze(0)?.squeeze(0)
    }
}
