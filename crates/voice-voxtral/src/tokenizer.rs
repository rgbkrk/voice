use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use fancy_regex::Regex;
use serde::Deserialize;

use crate::{
    Result, VoxtralConfig, VoxtralError, SAMPLE_RATE, VOXTRAL_NEXT_AUDIO_TEXT_TOKEN_ID,
    VOXTRAL_PRESET_VOICES, VOXTRAL_REPEAT_AUDIO_TEXT_TOKEN_ID,
};

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralTokenizerMetadata {
    pub config: TekkenConfig,
    pub special_tokens: Vec<TekkenSpecialToken>,
    pub vocab: Vec<TekkenVocabToken>,
    pub audio: TekkenAudioMetadata,
}

#[derive(Debug, Clone)]
pub struct VoxtralTekkenEncoder {
    mergeable_ranks: HashMap<Vec<u8>, usize>,
    pattern: Regex,
    token_id_offset: usize,
    default_vocab_size: usize,
}

#[derive(Debug, Clone)]
pub struct VoxtralTekkenDecoder {
    token_bytes: HashMap<usize, Vec<u8>>,
    token_id_offset: usize,
    default_vocab_size: usize,
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
    #[serde(default)]
    pub chunk_length_s: Option<f64>,
    #[serde(default)]
    pub voice_num_audio_tokens: BTreeMap<String, usize>,
    #[serde(default)]
    pub transcription_delay_ms: Option<usize>,
    #[serde(default)]
    pub streaming_look_ahead_ms: Option<f64>,
    #[serde(default)]
    pub streaming_look_back_ms: Option<f64>,
    #[serde(default)]
    pub streaming_n_left_pad_tokens: Option<usize>,
    #[serde(default)]
    pub transcription_format: Option<String>,
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

    pub fn encoder(&self) -> Result<VoxtralTekkenEncoder> {
        VoxtralTekkenEncoder::from_metadata(self)
    }

    pub fn decoder(&self) -> Result<VoxtralTekkenDecoder> {
        VoxtralTekkenDecoder::from_metadata(self)
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
        self.expect_special_token("[REPEAT_AUDIO_TEXT]", VOXTRAL_REPEAT_AUDIO_TEXT_TOKEN_ID)?;
        self.expect_special_token("[NEXT_AUDIO_TEXT]", VOXTRAL_NEXT_AUDIO_TEXT_TOKEN_ID)?;

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
        if self.audio.sampling_rate != SAMPLE_RATE && self.audio.sampling_rate != 16_000 {
            return Err(VoxtralError::InvalidTokenizer(format!(
                "expected {SAMPLE_RATE} Hz TTS or 16000 Hz realtime audio tokenizer metadata, got {}",
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

impl VoxtralTekkenEncoder {
    pub fn from_metadata(metadata: &VoxtralTokenizerMetadata) -> Result<Self> {
        let usable_vocab = metadata
            .config
            .default_vocab_size
            .checked_sub(metadata.config.default_num_special_tokens)
            .ok_or_else(|| {
                VoxtralError::InvalidTokenizer(format!(
                    "default_vocab_size {} is smaller than default_num_special_tokens {}",
                    metadata.config.default_vocab_size, metadata.config.default_num_special_tokens
                ))
            })?;
        let mut mergeable_ranks = HashMap::with_capacity(usable_vocab.min(metadata.vocab.len()));
        for token in &metadata.vocab {
            if token.rank >= usable_vocab {
                continue;
            }
            mergeable_ranks.insert(decode_base64(&token.token_bytes)?, token.rank);
        }
        let pattern = Regex::new(&metadata.config.pattern).map_err(|e| {
            VoxtralError::InvalidTokenizer(format!("invalid Tekken regex pattern: {e}"))
        })?;

        Ok(Self {
            mergeable_ranks,
            pattern,
            token_id_offset: metadata.config.default_num_special_tokens,
            default_vocab_size: metadata.config.default_vocab_size,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let mut token_ids = Vec::new();
        let mut cursor = 0;
        for piece in self.pattern.find_iter(text) {
            let piece = piece.map_err(|e| {
                VoxtralError::InvalidTokenizer(format!("Tekken regex matching failed: {e}"))
            })?;
            if piece.start() != cursor {
                return Err(VoxtralError::InvalidTokenizer(format!(
                    "Tekken regex did not cover input bytes {}..{}",
                    cursor,
                    piece.start()
                )));
            }
            let piece_tokens = self.encode_piece(piece.as_str().as_bytes())?;
            token_ids.extend(
                piece_tokens
                    .into_iter()
                    .map(|rank| self.token_id_offset + rank),
            );
            cursor = piece.end();
        }
        if cursor != text.len() {
            return Err(VoxtralError::InvalidTokenizer(format!(
                "Tekken regex did not cover trailing input bytes {cursor}..{}",
                text.len()
            )));
        }

        Ok(token_ids)
    }

    fn encode_piece(&self, bytes: &[u8]) -> Result<Vec<usize>> {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(rank) = self.mergeable_ranks.get(bytes) {
            return Ok(vec![*rank]);
        }

        let mut parts = (0..bytes.len()).map(|idx| idx..idx + 1).collect::<Vec<_>>();

        loop {
            let mut best = None;
            for idx in 0..parts.len().saturating_sub(1) {
                let candidate = parts[idx].start..parts[idx + 1].end;
                if let Some(rank) = self.mergeable_ranks.get(&bytes[candidate]) {
                    if best
                        .map(|(_best_idx, best_rank)| *rank < best_rank)
                        .unwrap_or(true)
                    {
                        best = Some((idx, *rank));
                    }
                }
            }

            let Some((idx, _rank)) = best else {
                break;
            };
            parts[idx].end = parts[idx + 1].end;
            parts.remove(idx + 1);
        }

        let mut ranks = Vec::with_capacity(parts.len());
        for part in parts {
            let rank = self
                .mergeable_ranks
                .get(&bytes[part.clone()])
                .ok_or_else(|| {
                    VoxtralError::InvalidTokenizer(format!(
                        "no Tekken token for byte span {:?}",
                        &bytes[part]
                    ))
                })?;
            if self.token_id_offset + *rank >= self.default_vocab_size {
                return Err(VoxtralError::InvalidTokenizer(format!(
                    "Tekken rank {rank} produces token id {} outside default vocab size {}",
                    self.token_id_offset + *rank,
                    self.default_vocab_size
                )));
            }
            ranks.push(*rank);
        }

        Ok(ranks)
    }
}

impl VoxtralTekkenDecoder {
    pub fn from_metadata(metadata: &VoxtralTokenizerMetadata) -> Result<Self> {
        let usable_vocab = metadata
            .config
            .default_vocab_size
            .checked_sub(metadata.config.default_num_special_tokens)
            .ok_or_else(|| {
                VoxtralError::InvalidTokenizer(format!(
                    "default_vocab_size {} is smaller than default_num_special_tokens {}",
                    metadata.config.default_vocab_size, metadata.config.default_num_special_tokens
                ))
            })?;
        let mut token_bytes = HashMap::with_capacity(usable_vocab.min(metadata.vocab.len()));
        for token in &metadata.vocab {
            if token.rank >= usable_vocab {
                continue;
            }
            token_bytes.insert(token.rank, decode_base64(&token.token_bytes)?);
        }

        Ok(Self {
            token_bytes,
            token_id_offset: metadata.config.default_num_special_tokens,
            default_vocab_size: metadata.config.default_vocab_size,
        })
    }

    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for token_id in token_ids {
            if let Some(token_bytes) = self.token_bytes(*token_id) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn decode_token(&self, token_id: usize) -> Option<String> {
        self.token_bytes(token_id)
            .map(|bytes| String::from_utf8_lossy(bytes).to_string())
    }

    pub fn token_bytes(&self, token_id: usize) -> Option<&[u8]> {
        if token_id < self.token_id_offset || token_id >= self.default_vocab_size {
            return None;
        }
        self.token_bytes
            .get(&(token_id - self.token_id_offset))
            .map(Vec::as_slice)
    }
}

fn decode_base64(input: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut value = 0u32;
    let mut bits = 0u32;

    for byte in input.bytes() {
        let Some(decoded) = base64_value(byte) else {
            if byte == b'=' {
                break;
            }
            return Err(VoxtralError::InvalidTokenizer(format!(
                "invalid base64 byte 0x{byte:02x}"
            )));
        };

        value = (value << 6) | decoded as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((value >> bits) & 0xff) as u8);
        }
    }

    Ok(out)
}

fn base64_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const PARAMS_JSON: &str = crate::config::tests::PARAMS_JSON;

    pub(crate) fn tokenizer_json() -> String {
        let vocab = [
            r#"{"rank":0,"token_bytes":"YQ==","token_str":"a"}"#,
            r#"{"rank":1,"token_bytes":"Yg==","token_str":"b"}"#,
            r#"{"rank":2,"token_bytes":"Yw==","token_str":"c"}"#,
            r#"{"rank":3,"token_bytes":"YmM=","token_str":"bc"}"#,
            r#"{"rank":4,"token_bytes":"YWI=","token_str":"ab"}"#,
            r#"{"rank":5,"token_bytes":"IA==","token_str":" "}"#,
            r#"{"rank":6,"token_bytes":"IQ==","token_str":"!"}"#,
        ]
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
        for rank in 28..35 {
            special_tokens.push(format!(
                r#"{{"rank":{rank},"token_str":"<SPECIAL_{rank}>","is_control":true}}"#
            ));
        }
        special_tokens
            .push(r#"{"rank":35,"token_str":"[REPEAT_AUDIO_TEXT]","is_control":true}"#.to_string());
        special_tokens
            .push(r#"{"rank":36,"token_str":"[NEXT_AUDIO_TEXT]","is_control":true}"#.to_string());

        format!(
            r#"{{
              "config": {{
                "pattern": "[^\\s]+|\\s+",
                "num_vocab_tokens": 7,
                "default_vocab_size": 131072,
                "default_num_special_tokens": 37,
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
    fn encodes_text_with_rank_priority_bpe() {
        let tokenizer = VoxtralTokenizerMetadata::from_json_str(&tokenizer_json()).unwrap();
        let encoder = tokenizer.encoder().unwrap();

        assert_eq!(encoder.encode("abc").unwrap(), vec![37, 40]);
    }

    #[test]
    fn encodes_regex_chunks_and_single_byte_tokens() {
        let tokenizer = VoxtralTokenizerMetadata::from_json_str(&tokenizer_json()).unwrap();
        let encoder = tokenizer.encoder().unwrap();

        assert_eq!(encoder.encode("ab a!").unwrap(), vec![41, 42, 37, 43]);
    }

    #[test]
    fn decodes_token_ids_to_text_and_skips_specials() {
        let tokenizer = VoxtralTokenizerMetadata::from_json_str(&tokenizer_json()).unwrap();
        let decoder = tokenizer.decoder().unwrap();

        assert_eq!(decoder.decode(&[1, 41, 42, 37, 43, 2]), "ab a!");
        assert_eq!(decoder.decode_token(41).as_deref(), Some("ab"));
        assert_eq!(decoder.decode_token(1), None);
    }

    #[test]
    fn loads_local_tekken_decoder_when_env_is_set() {
        let Ok(dir) = std::env::var("VOXTRAL_REALTIME_LOCAL_DIR") else {
            return;
        };
        let tokenizer = VoxtralTokenizerMetadata::from_path(
            std::path::Path::new(&dir).join(crate::REALTIME_TOKENIZER_FILE),
        )
        .unwrap();
        let decoder = tokenizer.decoder().unwrap();

        assert_eq!(tokenizer.config.default_num_special_tokens, 1000);
        assert_eq!(decoder.decode(&[1, 2, 32]), "");
        assert!(decoder.decode_token(1000).is_some());
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
        assert_eq!(tokenizer.special_token_id("[REPEAT_AUDIO_TEXT]"), Some(35));
        assert_eq!(tokenizer.special_token_id("[NEXT_AUDIO_TEXT]"), Some(36));
        assert_eq!(tokenizer.voice_audio_tokens("casual_male"), Some(147));

        let encoder = tokenizer.encoder().unwrap();
        let token_ids = encoder
            .encode("Voxtral support is running from native Rust on this Mac.")
            .unwrap();
        assert!(!token_ids.is_empty());
        assert!(token_ids
            .iter()
            .all(|token_id| *token_id < tokenizer.config.default_vocab_size));
    }
}
