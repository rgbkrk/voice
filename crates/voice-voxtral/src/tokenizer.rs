use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::{Result, VoxtralConfig, VoxtralError, SAMPLE_RATE, VOXTRAL_PRESET_VOICES};

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTokenizerMetadata {
    pub config: TekkenConfig,
    pub special_tokens: Vec<TekkenSpecialToken>,
    pub vocab: Vec<TekkenVocabToken>,
    pub audio: TekkenAudioMetadata,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TekkenConfig {
    pub pattern: String,
    pub num_vocab_tokens: usize,
    pub default_vocab_size: usize,
    pub default_num_special_tokens: usize,
    pub version: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TekkenSpecialToken {
    pub rank: usize,
    pub token_str: String,
    pub is_control: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TekkenVocabToken {
    pub rank: usize,
    pub token_bytes: String,
    pub token_str: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TekkenAudioMetadata {
    pub sampling_rate: u32,
    pub frame_rate: f64,
    pub audio_encoding_config: TekkenAudioEncodingConfig,
    pub chunk_length_s: f64,
    pub voice_num_audio_tokens: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TekkenAudioEncodingConfig {
    pub num_mel_bins: usize,
    pub hop_length: usize,
    pub window_size: usize,
}

impl VoxtralTokenizerMetadata {
    pub fn from_json_str(json: &str) -> Result<Self> {
        let metadata: Self = serde_json::from_str(json)?;
        metadata.validate_self()?;
        Ok(metadata)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        Self::from_json_str(&json)
    }

    pub fn special_token_id(&self, token: &str) -> Option<usize> {
        self.special_tokens
            .iter()
            .find(|entry| entry.token_str == token)
            .map(|entry| entry.rank)
    }

    pub fn voice_audio_tokens(&self, voice: &str) -> Option<usize> {
        self.audio.voice_num_audio_tokens.get(voice).copied()
    }

    pub fn validate_for_config(&self, config: &VoxtralConfig) -> Result<()> {
        if self.config.default_vocab_size != config.vocab_size {
            return Err(VoxtralError::InvalidTokenizer(format!(
                "tokenizer default_vocab_size={} but params vocab_size={}",
                self.config.default_vocab_size, config.vocab_size
            )));
        }

        let encoding = &config.multimodal.audio_model_args.audio_encoding_args;
        if self.audio.sampling_rate != SAMPLE_RATE
            || self.audio.sampling_rate != config.sample_rate()
        {
            return Err(VoxtralError::InvalidTokenizer(format!(
                "expected {SAMPLE_RATE} Hz audio tokenizer metadata, got {}",
                self.audio.sampling_rate
            )));
        }

        if (self.audio.frame_rate - encoding.frame_rate).abs() > f64::EPSILON {
            return Err(VoxtralError::InvalidTokenizer(format!(
                "audio tokenizer frame_rate={} but params frame_rate={}",
                self.audio.frame_rate, encoding.frame_rate
            )));
        }

        let audio_model = &config.multimodal.audio_model_args;
        self.expect_special_token("<s>", config.multimodal.bos_token_id as usize)?;
        self.expect_special_token("[AUDIO]", audio_model.audio_token_id as usize)?;
        self.expect_special_token("[BEGIN_AUDIO]", audio_model.begin_audio_token_id as usize)?;
        self.expect_special_token("[OUTPUT_AUDIO]", 26)?;

        for voice in VOXTRAL_PRESET_VOICES {
            if config.voice_id(voice.id).is_none() {
                return Err(VoxtralError::InvalidTokenizer(format!(
                    "voice {} exists in builtins but not params.json",
                    voice.id
                )));
            }
            if self.voice_audio_tokens(voice.id).is_none() {
                return Err(VoxtralError::InvalidTokenizer(format!(
                    "voice {} exists in params.json but not tekken audio metadata",
                    voice.id
                )));
            }
        }

        Ok(())
    }

    fn validate_self(&self) -> Result<()> {
        if self.vocab.len() != self.config.num_vocab_tokens {
            return Err(VoxtralError::InvalidTokenizer(format!(
                "vocab has {} entries but config declares {}",
                self.vocab.len(),
                self.config.num_vocab_tokens
            )));
        }
        if self.special_tokens.len() != self.config.default_num_special_tokens {
            return Err(VoxtralError::InvalidTokenizer(format!(
                "special_tokens has {} entries but config declares {}",
                self.special_tokens.len(),
                self.config.default_num_special_tokens
            )));
        }
        if self.audio.sampling_rate != SAMPLE_RATE {
            return Err(VoxtralError::InvalidTokenizer(format!(
                "expected {SAMPLE_RATE} Hz audio tokenizer metadata, got {}",
                self.audio.sampling_rate
            )));
        }
        self.expect_ranked_vocab()?;
        self.expect_ranked_special_tokens()?;
        Ok(())
    }

    fn expect_ranked_vocab(&self) -> Result<()> {
        for (index, entry) in self.vocab.iter().enumerate() {
            if entry.rank != index {
                return Err(VoxtralError::InvalidTokenizer(format!(
                    "vocab entry {index} has rank {}",
                    entry.rank
                )));
            }
        }
        Ok(())
    }

    fn expect_ranked_special_tokens(&self) -> Result<()> {
        for (index, entry) in self.special_tokens.iter().enumerate() {
            if entry.rank != index {
                return Err(VoxtralError::InvalidTokenizer(format!(
                    "special token entry {index} has rank {}",
                    entry.rank
                )));
            }
        }
        Ok(())
    }

    fn expect_special_token(&self, token: &str, expected_id: usize) -> Result<()> {
        match self.special_token_id(token) {
            Some(actual) if actual == expected_id => Ok(()),
            Some(actual) => Err(VoxtralError::InvalidTokenizer(format!(
                "special token {token} has id {actual}, expected {expected_id}"
            ))),
            None => Err(VoxtralError::InvalidTokenizer(format!(
                "missing special token {token}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMS_JSON: &str = crate::config::tests::PARAMS_JSON;

    fn tokenizer_json() -> String {
        let vocab = (0..6)
            .map(|rank| {
                format!(r#"{{"rank":{rank},"token_bytes":"AA==","token_str":"token-{rank}"}}"#)
            })
            .collect::<Vec<_>>()
            .join(",");
        let mut special_tokens = vec![
            r#"{"rank":0,"token_str":"<unk>","is_control":true}"#.to_string(),
            r#"{"rank":1,"token_str":"<s>","is_control":true}"#.to_string(),
            r#"{"rank":2,"token_str":"</s>","is_control":true}"#.to_string(),
        ];
        for rank in 3..24 {
            special_tokens.push(format!(
                r#"{{"rank":{rank},"token_str":"<SPECIAL_{rank}>","is_control":true}}"#
            ));
        }
        special_tokens.push(r#"{"rank":24,"token_str":"[AUDIO]","is_control":true}"#.to_string());
        special_tokens
            .push(r#"{"rank":25,"token_str":"[BEGIN_AUDIO]","is_control":true}"#.to_string());
        special_tokens
            .push(r#"{"rank":26,"token_str":"[OUTPUT_AUDIO]","is_control":true}"#.to_string());
        special_tokens
            .push(r#"{"rank":27,"token_str":"<SPECIAL_27>","is_control":true}"#.to_string());

        format!(
            r#"{{
              "config": {{
                "pattern": "\\s+",
                "num_vocab_tokens": 6,
                "default_vocab_size": 131072,
                "default_num_special_tokens": 28,
                "version": "v7"
              }},
              "special_tokens": [{}],
              "vocab": [{vocab}],
              "audio": {{
                "sampling_rate": 24000,
                "frame_rate": 12.5,
                "audio_encoding_config": {{
                  "num_mel_bins": 128,
                  "hop_length": 160,
                  "window_size": 400
                }},
                "chunk_length_s": 30.0,
                "voice_num_audio_tokens": {{
                  "casual_female": 214,
                  "casual_male": 147,
                  "cheerful_female": 132,
                  "neutral_female": 218,
                  "neutral_male": 169,
                  "pt_male": 144,
                  "pt_female": 175,
                  "nl_male": 138,
                  "nl_female": 146,
                  "it_male": 168,
                  "it_female": 172,
                  "fr_male": 97,
                  "fr_female": 97,
                  "es_male": 208,
                  "es_female": 138,
                  "de_male": 163,
                  "de_female": 147,
                  "ar_male": 67,
                  "hi_male": 94,
                  "hi_female": 86
                }}
              }}
            }}"#,
            special_tokens.join(",")
        )
    }

    #[test]
    fn validates_tokenizer_metadata_against_config() {
        let config = VoxtralConfig::from_json_str(PARAMS_JSON).unwrap();
        let tokenizer = VoxtralTokenizerMetadata::from_json_str(&tokenizer_json()).unwrap();

        tokenizer.validate_for_config(&config).unwrap();
        assert_eq!(tokenizer.special_token_id("[OUTPUT_AUDIO]"), Some(26));
        assert_eq!(tokenizer.voice_audio_tokens("casual_male"), Some(147));
    }

    #[test]
    fn rejects_bad_special_token_id() {
        let config = VoxtralConfig::from_json_str(PARAMS_JSON).unwrap();
        let raw = tokenizer_json().replace(
            r#""rank":24,"token_str":"[AUDIO]""#,
            r#""rank":24,"token_str":"[WRONG]""#,
        );
        let tokenizer = VoxtralTokenizerMetadata::from_json_str(&raw).unwrap();
        let err = tokenizer.validate_for_config(&config).unwrap_err();

        assert!(matches!(err, VoxtralError::InvalidTokenizer(_)));
    }

    #[test]
    fn validates_local_tekken_when_env_is_set() {
        let Ok(dir) = std::env::var("VOXTRAL_LOCAL_DIR") else {
            return;
        };
        let dir = std::path::Path::new(&dir);

        let config = VoxtralConfig::from_path(dir.join(crate::CONFIG_FILE)).unwrap();
        let tokenizer =
            VoxtralTokenizerMetadata::from_path(dir.join(crate::TOKENIZER_FILE)).unwrap();

        tokenizer.validate_for_config(&config).unwrap();
        assert_eq!(tokenizer.config.num_vocab_tokens, 150_000);
        assert_eq!(tokenizer.config.default_vocab_size, config.vocab_size);
        assert_eq!(tokenizer.special_token_id("[AUDIO]"), Some(24));
        assert_eq!(tokenizer.special_token_id("[BEGIN_AUDIO]"), Some(25));
        assert_eq!(tokenizer.special_token_id("[OUTPUT_AUDIO]"), Some(26));
        assert_eq!(tokenizer.voice_audio_tokens("casual_male"), Some(147));
    }
}
