use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{Linear, LinearBuilder};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{clip, sigmoid};
use mlx_rs::Array;
use std::collections::HashMap;

use crate::config::{AlbertConfig, ModelConfig};
use voice_nn::albert::{CustomAlbert, CustomAlbertInput};
use voice_nn::prosody::{DurationEncoderInput, ProsodyPredictor};
use voice_nn::text_encoder::{TextEncoder, TextEncoderInput};
use voice_nn::vocoder::decoder::{Decoder, DecoderInput};

#[derive(Debug, ModuleParameters)]
pub struct KokoroModel {
    #[param]
    pub bert: CustomAlbert,
    #[param]
    pub bert_encoder: Linear,
    #[param]
    pub predictor: ProsodyPredictor,
    #[param]
    pub text_encoder: TextEncoder,
    #[param]
    pub decoder: Decoder,

    pub vocab: HashMap<String, i32>,
    pub context_length: i32,
    pub sample_rate: i32,
}

impl KokoroModel {
    pub fn new(config: &ModelConfig) -> Result<Self, Exception> {
        let albert_config = AlbertConfig {
            vocab_size: config.n_token,
            ..config.plbert.clone()
        };

        let bert = CustomAlbert::new(&albert_config)?;
        let bert_encoder =
            LinearBuilder::new(albert_config.hidden_size, config.hidden_dim).build()?;

        let predictor = ProsodyPredictor::new(
            config.style_dim,
            config.hidden_dim,
            config.n_layer,
            config.max_dur,
            config.dropout,
        )?;

        let text_encoder = TextEncoder::new(
            config.hidden_dim,
            config.text_encoder_kernel_size,
            config.n_layer,
            config.n_token,
        )?;

        let istft = &config.istftnet;
        let decoder = Decoder::new(
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
        )?;

        Ok(Self {
            bert,
            bert_encoder,
            predictor,
            text_encoder,
            decoder,
            vocab: config.vocab.clone(),
            context_length: albert_config.max_position_embeddings,
            sample_rate: config.sample_rate,
        })
    }

    /// Generate audio from phoneme string and voice embedding.
    ///
    /// - `phonemes`: A phoneme string (e.g. "h ɛ l oʊ")
    /// - `ref_s`: Voice/style embedding, shape (1, 256)
    /// - `speed`: Speed factor (1.0 = normal)
    pub fn generate(
        &mut self,
        phonemes: &str,
        ref_s: &Array,
        speed: f32,
    ) -> Result<Array, Exception> {
        // Disable nn.Dropout in all modules (eval mode) — Python MLX modules default
        // to eval mode where nn.Dropout is a no-op. mlx-rs defaults to training=true.
        // Note: The prosody predictor intentionally uses raw mx.dropout(p=0.5) at
        // inference which is NOT an nn.Dropout and is always active.
        self.bert.training_mode(false);
        self.text_encoder.training_mode(false);
        self.decoder.training_mode(false);
        self.predictor.training_mode(false);

        // Map phonemes to token IDs
        let input_ids: Vec<i32> = phonemes
            .chars()
            .filter_map(|c| self.vocab.get(&c.to_string()).copied())
            .collect();

        assert!(
            input_ids.len() + 2 <= self.context_length as usize,
            "Input too long: {} + 2 > {}",
            input_ids.len(),
            self.context_length
        );

        // Build input_ids array: [0, ...ids, 0] (BOS/EOS)
        let mut ids_with_bos_eos = vec![0i32];
        ids_with_bos_eos.extend_from_slice(&input_ids);
        ids_with_bos_eos.push(0);
        let seq_len = ids_with_bos_eos.len() as i32;
        let input_ids_arr = Array::from_slice(&ids_with_bos_eos, &[1, seq_len]);

        // input_lengths
        let input_lengths = Array::from_slice(&[seq_len], &[1]);

        // text_mask: arange(seq_len)[None, :] + 1 > input_lengths[:, None]
        let arange = Array::arange::<_, i32>(None, seq_len, None)?;
        let arange = arange.reshape(&[1, seq_len])?;
        let one = Array::from_int(1);
        let arange_plus_one = &arange + &one;
        let lengths_expanded = input_lengths.reshape(&[1, 1])?;
        let text_mask = arange_plus_one.gt(&lengths_expanded)?;

        // ALBERT encoder
        let mask_int = text_mask
            .logical_not()?
            .as_dtype(mlx_rs::Dtype::Int32)?;
        let bert_output = self.bert.forward(CustomAlbertInput {
            input_ids: &input_ids_arr,
            token_type_ids: None,
            attention_mask: Some(&mask_int),
        })?;

        // Project ALBERT output to hidden dim, transpose to (B, hidden, seq)
        let d_en = self
            .bert_encoder
            .forward(&bert_output.encoder_output)?
            .transpose_axes(&[0, 2, 1])?;

        // Extract style from ref_s
        // ref_s is a voice pack of shape (510, 1, 256).
        // Index by phoneme count - 1 to get the style for this length.
        let phoneme_count = input_ids.len() as i32; // excludes BOS/EOS
        let ref_s = if ref_s.ndim() == 3 {
            ref_s.index(phoneme_count - 1)
        } else {
            ref_s.clone()
        };

        let s = ref_s.index((.., 128..));

        // Duration encoding
        let d = self.predictor.text_encoder.forward(DurationEncoderInput {
            x: &d_en,
            style: &s,
            text_lengths: &input_lengths,
            mask: &text_mask,
        })?;

        // Duration LSTM
        let (lstm_out, _) = self.predictor.lstm.forward(&d)?;

        // Duration projection
        let duration = self.predictor.duration_proj.forward(&lstm_out)?;
        let duration = sigmoid(&duration)?;
        let duration_sum = duration.sum_axes(&[-1], false)?;
        let speed_arr = Array::from_f32(speed);
        let duration_scaled = &duration_sum / &speed_arr;

        // Round and clip durations
        let rounded = duration_scaled.round(None)?;
        let pred_dur = clip(&rounded, (1.0f32, ()))?;
        let pred_dur = pred_dur.as_dtype(mlx_rs::Dtype::Int32)?;
        // Take first batch element
        let pred_dur = pred_dur.index(0);
        pred_dur.eval()?;

        // Build alignment matrix on CPU
        let pred_dur_slice: &[i32] = pred_dur.as_slice();
        let total_frames: i32 = pred_dur_slice.iter().sum();

        let mut indices = Vec::with_capacity(total_frames as usize);
        for (i, &n) in pred_dur_slice.iter().enumerate() {
            for _ in 0..n {
                indices.push(i as i32);
            }
        }

        // Build pred_aln_trg[i, j] = 1 where i = indices[j]
        let mut aln_data = vec![0.0f32; (seq_len * total_frames) as usize];
        for (j, &idx) in indices.iter().enumerate() {
            aln_data[(idx * total_frames + j as i32) as usize] = 1.0;
        }
        let pred_aln_trg = Array::from_slice(&aln_data, &[seq_len, total_frames]);
        let pred_aln_trg = pred_aln_trg.reshape(&[1, seq_len, total_frames])?;

        // en = d^T @ pred_aln_trg
        let d_t = d.transpose_axes(&[0, 2, 1])?;
        let en = d_t.matmul(&pred_aln_trg)?;

        // F0 and N prediction
        let (f0_pred, n_pred) = self.predictor.f0_n_train(&en, &s)?;

        // Text encoder
        let t_en = self.text_encoder.forward(TextEncoderInput {
            x: &input_ids_arr,
            input_lengths: &input_lengths,
            mask: &text_mask,
        })?;

        // ASR = t_en @ pred_aln_trg
        let asr = t_en.matmul(&pred_aln_trg)?;

        // Decoder: generate audio
        let speaker_style = ref_s.index((.., ..128));
        let audio = self.decoder.forward(DecoderInput {
            asr: &asr,
            f0_curve: &f0_pred,
            n: &n_pred,
            s: &speaker_style,
        })?;

        // audio may be (1, 1, samples), (1, samples, 1), etc.
        // Squeeze all size-1 dimensions
        let audio = audio.squeeze()?;

        audio.eval()?;
        Ok(audio)
    }
}
