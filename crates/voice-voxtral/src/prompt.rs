use std::ops::Range;

use candle_core::{Device, Tensor};

use crate::{
    Result, VoxtralConfig, VoxtralError, VoxtralMultimodalEmbeddings, VoxtralTokenizerMetadata,
};

/// Speech-prompt token that marks the transition from reference audio to text.
pub const VOXTRAL_REPEAT_AUDIO_TEXT_TOKEN_ID: usize = 35;

/// Speech-prompt token that marks the transition from text to generated audio.
pub const VOXTRAL_NEXT_AUDIO_TEXT_TOKEN_ID: usize = 36;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralPrompt {
    pub input_ids: Vec<usize>,
    pub voice_range: Range<usize>,
    pub text_range: Range<usize>,
}

pub fn build_prompt_token_ids(
    config: &VoxtralConfig,
    tokenizer: &VoxtralTokenizerMetadata,
    voice_frames: usize,
    text_token_ids: &[usize],
) -> Result<VoxtralPrompt> {
    let bos = token_id(config.multimodal.bos_token_id, "bos_token_id")?;
    let audio = token_id(
        config.multimodal.audio_model_args.audio_token_id,
        "audio_token_id",
    )?;
    let begin_audio = token_id(
        config.multimodal.audio_model_args.begin_audio_token_id,
        "begin_audio_token_id",
    )?;
    let next_audio_text = required_special_token(tokenizer, "[NEXT_AUDIO_TEXT]")?;
    let repeat_audio_text = required_special_token(tokenizer, "[REPEAT_AUDIO_TEXT]")?;

    let mut input_ids = Vec::with_capacity(2 + voice_frames + 1 + text_token_ids.len() + 2);
    input_ids.push(bos);
    input_ids.push(begin_audio);

    let voice_start = input_ids.len();
    input_ids.extend(std::iter::repeat_n(audio, voice_frames));
    let voice_end = input_ids.len();

    input_ids.push(next_audio_text);
    let text_start = input_ids.len();
    input_ids.extend_from_slice(text_token_ids);
    let text_end = input_ids.len();
    input_ids.push(repeat_audio_text);
    input_ids.push(begin_audio);

    Ok(VoxtralPrompt {
        input_ids,
        voice_range: voice_start..voice_end,
        text_range: text_start..text_end,
    })
}

pub fn build_prompt_embeddings(
    embeddings: &VoxtralMultimodalEmbeddings,
    prompt: &VoxtralPrompt,
    voice_embeddings: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let voice_frames = prompt.voice_range.len();
    let (voice_rows, voice_dim) = voice_embeddings
        .dims2()
        .map_err(|e| VoxtralError::Candle(e.to_string()))?;
    if voice_rows != voice_frames {
        return Err(VoxtralError::InvalidCheckpoint(format!(
            "voice embedding has {voice_rows} rows but prompt expects {voice_frames}"
        )));
    }
    if voice_dim != embeddings.tok_embeddings.hidden_size() {
        return Err(VoxtralError::InvalidCheckpoint(format!(
            "voice embedding hidden dim {voice_dim} does not match token embedding dim {}",
            embeddings.tok_embeddings.hidden_size()
        )));
    }

    if voice_frames == 0 {
        return embeddings
            .token_embeddings(&prompt.input_ids, device)
            .map_err(|e| VoxtralError::Candle(e.to_string()));
    }

    let prefix = embeddings
        .token_embeddings(&prompt.input_ids[..prompt.voice_range.start], device)
        .map_err(|e| VoxtralError::Candle(e.to_string()))?;
    let suffix = embeddings
        .token_embeddings(&prompt.input_ids[prompt.voice_range.end..], device)
        .map_err(|e| VoxtralError::Candle(e.to_string()))?;
    let voice = voice_embeddings
        .to_device(device)
        .and_then(|tensor| tensor.unsqueeze(0))
        .map_err(|e| VoxtralError::Candle(e.to_string()))?;
    Tensor::cat(&[prefix, voice, suffix], 1).map_err(|e| VoxtralError::Candle(e.to_string()))
}

fn token_id(value: i64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        VoxtralError::InvalidConfig(format!("{name} must be non-negative, got {value}"))
    })
}

fn required_special_token(tokenizer: &VoxtralTokenizerMetadata, token: &str) -> Result<usize> {
    tokenizer.special_token_id(token).ok_or_else(|| {
        VoxtralError::InvalidTokenizer(format!("missing required speech token {token}"))
    })
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    use super::*;
    use crate::{
        tokenizer::tests::tokenizer_json, transformer::tests::tiny_config, VoxtralInferenceModules,
        VoxtralTokenizerMetadata,
    };

    #[test]
    fn builds_speech_request_prompt_token_ids() {
        let config = tiny_config();
        let tokenizer = VoxtralTokenizerMetadata::from_json_str(&tokenizer_json()).unwrap();
        let prompt = build_prompt_token_ids(&config, &tokenizer, 3, &[1000, 1001]).unwrap();

        assert_eq!(
            prompt.input_ids,
            vec![
                1,
                25,
                24,
                24,
                24,
                VOXTRAL_NEXT_AUDIO_TEXT_TOKEN_ID,
                1000,
                1001,
                VOXTRAL_REPEAT_AUDIO_TEXT_TOKEN_ID,
                25,
            ]
        );
        assert_eq!(prompt.voice_range, 2..5);
        assert_eq!(prompt.text_range, 6..8);
    }

    #[test]
    fn replaces_voice_token_positions_with_voice_embeddings() {
        let config = tiny_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let modules = VoxtralInferenceModules::load(&config, vb).unwrap();
        let tokenizer = VoxtralTokenizerMetadata::from_json_str(&tokenizer_json()).unwrap();
        let prompt = build_prompt_token_ids(&config, &tokenizer, 2, &[2]).unwrap();
        let voice_embeddings =
            Tensor::from_vec(vec![1.0f32; 2 * config.dim], (2, config.dim), &Device::Cpu).unwrap();

        let prompt_embeddings = build_prompt_embeddings(
            &modules.embeddings,
            &prompt,
            &voice_embeddings,
            &Device::Cpu,
        )
        .unwrap();
        let values = prompt_embeddings.to_vec3::<f32>().unwrap();

        assert_eq!(
            prompt_embeddings.dims(),
            &[1, prompt.input_ids.len(), config.dim]
        );
        assert_eq!(values[0][prompt.voice_range.start][0], 1.0);
        assert_eq!(values[0][prompt.voice_range.end - 1][0], 1.0);
        assert_eq!(values[0][0][0], 0.0);
        assert_eq!(values[0][prompt.voice_range.end][0], 0.0);
    }
}
