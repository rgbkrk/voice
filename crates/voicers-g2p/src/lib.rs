use std::process::Command;

#[derive(Debug, thiserror::Error)]
pub enum G2pError {
    #[error("espeak-ng not found. Install with: brew install espeak-ng")]
    EspeakNotFound,
    #[error("espeak-ng failed: {0}")]
    EspeakFailed(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convert English text to a Kokoro-compatible phoneme string via espeak-ng.
pub fn english_to_phonemes(text: &str) -> Result<String, G2pError> {
    let output = Command::new("espeak-ng")
        .args(["--ipa", "-q", "-v", "en-us", text])
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                G2pError::EspeakNotFound
            } else {
                G2pError::Io(e)
            }
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(G2pError::EspeakFailed(stderr));
    }

    let ipa = String::from_utf8_lossy(&output.stdout);
    // espeak-ng may output multiple lines (one per clause). Join with space.
    let joined: String = ipa
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    Ok(espeak_ipa_to_kokoro(&joined))
}

/// Post-process espeak-ng IPA output into Kokoro phoneme format.
///
/// The Kokoro model was trained with misaki G2P output, which uses collapsed
/// diphthongs (capital letters) and affricate ligatures. espeak-ng outputs
/// expanded IPA, so we convert to match.
pub fn espeak_ipa_to_kokoro(ipa: &str) -> String {
    let mut s = ipa.to_string();

    // Multi-character replacements (longest match first to avoid partial matches)

    // Affricates: two-char sequences → ligature characters
    s = s.replace("dʒ", "ʤ");
    s = s.replace("tʃ", "ʧ");

    // NURSE vowel: ɜːɹ → ɜɹ (remove length mark before rhotic)
    s = s.replace("ɜːɹ", "ɜɹ");
    // NURSE without explicit rhotic: ɜː → ɜɹ (American English adds rhotic)
    s = s.replace("ɜː", "ɜɹ");

    // Diphthongs: two-char IPA → single capital letter tokens
    // Order matters: do these after affricates but before single-char cleanup
    s = s.replace("aɪ", "I");
    s = s.replace("aʊ", "W");
    s = s.replace("eɪ", "A");
    s = s.replace("oʊ", "O");
    s = s.replace("ɔɪ", "Y");

    // Long vowels: remove remaining length marks (the model doesn't use them
    // for most vowels since misaki doesn't produce them)
    s = s.replace('ː', "");

    // Rhotacized schwa: ɚ is in vocab (token 85), keep as-is
    // ɾ (flap) → T for American English (matches misaki pipeline)
    s = s.replace('ɾ', "T");

    // ɡ (IPA g, U+0261) should map to ɡ (token 92) — espeak already uses this

    s
}

/// Split text into chunks whose phoneme representations fit within the model's
/// 510-character context limit.
///
/// Strategy: split on newlines first, then on sentence boundaries if needed.
pub fn text_to_phoneme_chunks(text: &str) -> Result<Vec<String>, G2pError> {
    const MAX_PHONEME_LEN: usize = 500; // leave margin below 510

    let mut chunks = Vec::new();

    // Split on newlines first
    for paragraph in text.split('\n') {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }

        let phonemes = english_to_phonemes(paragraph)?;
        if phonemes.len() <= MAX_PHONEME_LEN {
            chunks.push(phonemes);
            continue;
        }

        // Too long — split on sentence boundaries
        let sentences = split_sentences(paragraph);
        let mut current_phonemes = String::new();

        for sentence in &sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let sent_phonemes = english_to_phonemes(sentence)?;

            if current_phonemes.is_empty() {
                current_phonemes = sent_phonemes;
            } else if current_phonemes.len() + 1 + sent_phonemes.len() <= MAX_PHONEME_LEN {
                current_phonemes.push(' ');
                current_phonemes.push_str(&sent_phonemes);
            } else {
                chunks.push(current_phonemes);
                current_phonemes = sent_phonemes;
            }
        }

        if !current_phonemes.is_empty() {
            chunks.push(current_phonemes);
        }
    }

    if chunks.is_empty() {
        chunks.push(String::new());
    }

    Ok(chunks)
}

/// Split text into sentences on `.!?` boundaries, keeping the punctuation
/// attached to the preceding sentence.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            sentences.push(current.clone());
            current.clear();
        }
    }

    // Remaining text without terminal punctuation
    if !current.trim().is_empty() {
        sentences.push(current);
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affricate_conversion() {
        assert_eq!(espeak_ipa_to_kokoro("dʒʌmp"), "ʤʌmp");
        assert_eq!(espeak_ipa_to_kokoro("tʃɪp"), "ʧɪp");
    }

    #[test]
    fn test_diphthong_collapse() {
        assert_eq!(espeak_ipa_to_kokoro("haɪ"), "hI");
        assert_eq!(espeak_ipa_to_kokoro("naʊ"), "nW");
        assert_eq!(espeak_ipa_to_kokoro("deɪ"), "dA");
        assert_eq!(espeak_ipa_to_kokoro("goʊ"), "gO");
        assert_eq!(espeak_ipa_to_kokoro("bɔɪ"), "bY");
    }

    #[test]
    fn test_nurse_vowel() {
        assert_eq!(espeak_ipa_to_kokoro("wɜːɹld"), "wɜɹld");
        assert_eq!(espeak_ipa_to_kokoro("bɜːd"), "bɜɹd");
    }

    #[test]
    fn test_length_mark_removal() {
        assert_eq!(espeak_ipa_to_kokoro("siː"), "si");
        assert_eq!(espeak_ipa_to_kokoro("fuːd"), "fud");
    }

    #[test]
    fn test_flap_to_t() {
        assert_eq!(espeak_ipa_to_kokoro("wɑɾɚ"), "wɑTɚ");
    }

    #[test]
    fn test_full_espeak_output() {
        // espeak-ng output for "Hello world"
        let input = "həlˈoʊ wˈɜːld";
        let expected = "həlˈO wˈɜɹld";
        assert_eq!(espeak_ipa_to_kokoro(input), expected);
    }

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("Hello world. How are you? I'm fine!");
        assert_eq!(sentences, vec!["Hello world.", " How are you?", " I'm fine!"]);
    }

    #[test]
    fn test_english_to_phonemes() {
        // This test requires espeak-ng to be installed
        match english_to_phonemes("Hello") {
            Ok(phonemes) => {
                assert!(!phonemes.is_empty());
                // Should contain the O diphthong (collapsed from oʊ)
                assert!(phonemes.contains('O'), "Expected O diphthong in: {}", phonemes);
            }
            Err(G2pError::EspeakNotFound) => {
                eprintln!("Skipping test: espeak-ng not installed");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }
}
