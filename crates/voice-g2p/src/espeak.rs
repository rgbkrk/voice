//! Per-word espeak-ng fallback phonemizer, ported from misaki's `espeak.py`.
//!
//! This differs from the sentence-level `english_to_phonemes` in `lib.rs`:
//! 1. Phonemizes one word at a time (not full sentences)
//! 2. Uses `--tie=^` so multi-character IPA sequences are explicitly tied
//! 3. Applies misaki's exact E2M (espeak-to-misaki) mapping table

use std::process::Command;

/// Espeak-to-Misaki replacement pairs, sorted by key length descending
/// so longest-match-first replacement works correctly.
pub(crate) const E2M: &[(&str, &str)] = &[
    // 4+ character sequences
    ("\u{0294}\u{02CC}n\u{0329}", "t\u{1D4A}n"), // ʔˌn̩ → tᵊn
    // 3 character sequences
    ("\u{0294}n", "t\u{1D4A}n"),        // ʔn → tᵊn
    ("\u{0259}\u{005E}l", "\u{1D4A}l"), // ə^l → ᵊl
    // 2 character sequences (tied diphthongs/affricates)
    ("a\u{005E}\u{026A}", "I"),        // a^ɪ → I
    ("a\u{005E}\u{028A}", "W"),        // a^ʊ → W
    ("d\u{005E}\u{0292}", "\u{02A4}"), // d^ʒ → ʤ
    ("e\u{005E}\u{026A}", "A"),        // e^ɪ → A
    ("t\u{005E}\u{0283}", "\u{02A7}"), // t^ʃ → ʧ
    ("\u{0254}\u{005E}\u{026A}", "Y"), // ɔ^ɪ → Y
    ("\u{02B2}O", "jO"),               // ʲO → jO
    ("\u{02B2}Q", "jQ"),               // ʲQ → jQ
    // 1 character sequences
    ("\u{0303}", ""),                 // nasalization diacritic → remove
    ("e", "A"),                       // bare e → A
    ("r", "\u{0279}"),                // r → ɹ
    ("x", "k"),                       // velar fricative → k
    ("\u{00E7}", "k"),                // ç → k
    ("\u{0250}", "\u{0259}"),         // ɐ → ə
    ("\u{025A}", "\u{0259}\u{0279}"), // ɚ → əɹ
    ("\u{026C}", "l"),                // ɬ → l
    ("\u{0294}", "t"),                // ʔ → t
    ("\u{02B2}", ""),                 // bare ʲ → remove
];

/// Per-word espeak-ng fallback, ported from misaki's `EspeakFallback`.
pub struct EspeakFallback {
    british: bool,
    espeak_path: String,
}

impl EspeakFallback {
    /// Create a new fallback with US English and default PATH lookup.
    pub fn new() -> Self {
        Self {
            british: false,
            espeak_path: "espeak-ng".to_string(),
        }
    }

    /// Create a new fallback with a custom espeak-ng binary path.
    pub fn with_path(espeak_path: String) -> Self {
        Self {
            british: false,
            espeak_path,
        }
    }

    /// Check if espeak-ng is available on the system.
    pub fn is_available(&self) -> bool {
        Command::new(&self.espeak_path)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Convert a single word to Kokoro-compatible phonemes via espeak-ng.
    ///
    /// Returns `Some((phonemes, 2))` on success, or `None` if espeak-ng
    /// is unavailable or produces no output. Rating 2 indicates espeak fallback.
    pub fn convert_word(&self, word: &str) -> Option<(String, u8)> {
        let lang = if self.british { "en-gb" } else { "en-us" };

        let output = Command::new(&self.espeak_path)
            .args(["--ipa", "-q", "-v", lang, "--tie=^", word])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let raw = String::from_utf8_lossy(&output.stdout);
        let ps = raw.trim();
        if ps.is_empty() {
            return None;
        }

        if self.british {
            // British path kept inline (not extracted since bronze is US-only)
            let mut ps = ps.to_string();
            for &(old, new) in E2M {
                ps = ps.replace(old, new);
            }
            ps = replace_syllabic_mark(&ps);
            ps = ps.replace("e^ə", "\u{025B}\u{02D0}"); // e^ə → ɛː
            ps = ps.replace("i\u{0259}", "\u{026A}\u{0259}"); // iə → ɪə
            ps = ps.replace("\u{0259}^\u{028A}", "Q"); // ə^ʊ → Q
            ps = ps.replace('^', "");
            ps = ps.replace('\u{027E}', "T");
            ps = ps.replace('\u{0294}', "t");
            Some((ps, 2))
        } else {
            Some((apply_e2m_us(ps), 2))
        }
    }
}

impl Default for EspeakFallback {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle the syllabic consonant diacritic (U+0329 COMBINING VERTICAL LINE BELOW).
/// Pattern: any non-whitespace char followed by U+0329 → ᵊ + that char.
/// Then remove any remaining U+0329.
pub(crate) fn replace_syllabic_mark(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut result = String::with_capacity(input.len());
    let mut i = 0;

    while i < chars.len() {
        if i + 1 < chars.len() && chars[i + 1] == '\u{0329}' && !chars[i].is_whitespace() {
            // Replace (\S)\u0329 with ᵊ\1
            result.push('\u{1D4A}'); // ᵊ
            result.push(chars[i]);
            i += 2; // skip both the consonant and the combining mark
        } else if chars[i] == '\u{0329}' {
            // Remove any remaining U+0329 that didn't match the pattern
            i += 1;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Convert raw espeak-ng IPA output (with tie markers) to Kokoro phonemes.
///
/// Applies the E2M mapping table, syllabic mark handling, US-English vowel
/// adjustments, tie marker removal, and legacy conversions.
pub fn apply_e2m_us(raw_ipa: &str) -> String {
    let mut ps = raw_ipa.to_string();

    // Apply E2M replacements (longest-match-first, already sorted by key length desc)
    for &(old, new) in E2M {
        ps = ps.replace(old, new);
    }

    // Handle syllabic consonant diacritic U+0329
    ps = replace_syllabic_mark(&ps);

    // US-English adjustments
    ps = ps.replace("o^\u{028A}", "O"); // o^ʊ → O
    ps = ps.replace("\u{025C}\u{02D0}\u{0279}", "\u{025C}\u{0279}"); // ɜːɹ → ɜɹ
    ps = ps.replace("\u{025C}\u{02D0}", "\u{025C}\u{0279}"); // ɜː → ɜɹ
    ps = ps.replace("\u{026A}\u{0259}", "i\u{0259}"); // ɪə → iə
    ps = ps.replace('\u{02D0}', ""); // remove remaining ː

    // Remove remaining tie markers
    ps = ps.replace('^', "");

    // Legacy conversion
    ps = ps.replace('\u{027E}', "T"); // ɾ → T
    ps = ps.replace('\u{0294}', "t"); // ʔ → t

    ps
}

/// Sentence-level espeak-ng phonemization (no tie marker).
///
/// This wraps the same espeak-ng subprocess call used by `english_to_phonemes`
/// in `lib.rs`, but returns `Option<String>` for convenience in fallback chains.
pub fn espeak_sentence(text: &str, espeak_path: &str) -> Option<String> {
    let output = Command::new(espeak_path)
        .args(["--ipa", "-q", "-v", "en-us", text])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let ipa = String::from_utf8_lossy(&output.stdout);
    let joined: String = ipa
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    if joined.is_empty() {
        return None;
    }

    // Apply the same post-processing as lib.rs
    Some(crate::espeak_ipa_to_kokoro(&joined))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syllabic_mark_replacement() {
        // n followed by U+0329 → ᵊn
        let input = format!("n{}", '\u{0329}');
        assert_eq!(replace_syllabic_mark(&input), "\u{1D4A}n");
    }

    #[test]
    fn test_syllabic_mark_in_context() {
        // "bɑtl̩" (bottle with syllabic l)
        let input = format!("b\u{0251}tl{}", '\u{0329}');
        assert_eq!(replace_syllabic_mark(&input), "b\u{0251}t\u{1D4A}l");
    }

    #[test]
    fn test_e2m_affricate_tie() {
        // With tie marker: d^ʒ → ʤ
        let mut s = "d^ʒ".to_string();
        for &(old, new) in E2M {
            s = s.replace(old, new);
        }
        assert_eq!(s, "\u{02A4}");
    }

    #[test]
    fn test_e2m_diphthong_tie() {
        // a^ɪ → I
        let mut s = "a^ɪ".to_string();
        for &(old, new) in E2M {
            s = s.replace(old, new);
        }
        assert_eq!(s, "I");
    }

    #[test]
    fn test_apply_e2m_us_goat_vowel() {
        // o^ʊ → O (goat diphthong with tie marker)
        let input = "h\u{0259}l\u{02C8}o^\u{028A}";
        let result = apply_e2m_us(input);
        assert!(
            result.contains('O'),
            "Expected O diphthong in: {result}"
        );
        assert!(
            !result.contains('^'),
            "Tie markers should be removed: {result}"
        );
    }

    #[test]
    fn test_apply_e2m_us_affricates() {
        // d^ʒ → ʤ, t^ʃ → ʧ
        assert!(apply_e2m_us("d^\u{0292}\u{028C}mp").contains('\u{02A4}'));
        assert!(apply_e2m_us("t^\u{0283}\u{026A}p").contains('\u{02A7}'));
    }

    #[test]
    fn test_apply_e2m_us_nurse_vowel() {
        // ɜːɹ → ɜɹ (nurse vowel, remove length mark)
        let result = apply_e2m_us("w\u{025C}\u{02D0}\u{0279}ld");
        assert_eq!(result, "w\u{025C}\u{0279}ld");
    }

    #[test]
    fn test_convert_word_available() {
        let fb = EspeakFallback::new();
        if !fb.is_available() {
            eprintln!("Skipping test: espeak-ng not installed");
            return;
        }
        let result = fb.convert_word("hello");
        assert!(
            result.is_some(),
            "espeak-ng should produce output for 'hello'"
        );
        let (ps, rating) = result.unwrap();
        assert_eq!(rating, 2);
        assert!(!ps.is_empty());
        // Should contain the O diphthong
        assert!(
            ps.contains('O'),
            "Expected O diphthong in phonemes for 'hello': {}",
            ps
        );
    }

    #[test]
    fn test_espeak_sentence_available() {
        let fb = EspeakFallback::new();
        if !fb.is_available() {
            eprintln!("Skipping test: espeak-ng not installed");
            return;
        }

        let result = espeak_sentence("Hello world", "espeak-ng");
        assert!(result.is_some());
        let ps = result.unwrap();
        assert!(!ps.is_empty());
        assert!(ps.contains('O'), "Expected O in: {}", ps);
    }
}
