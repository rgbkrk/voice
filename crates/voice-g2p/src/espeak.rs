//! OOV fallback phonemizer plus espeak-ng conversion helpers.
//!
//! Runtime G2P uses the embedded fallback in this module so `voice` does not
//! require an `espeak-ng` binary on fresh machines. The espeak conversion
//! helpers are still used by the offline bronze-dictionary generator.

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

/// Per-word fallback for OOV pronunciations.
pub struct EspeakFallback {
    espeak_path: String,
}

impl EspeakFallback {
    /// Create a new fallback with US English and default PATH lookup.
    pub fn new() -> Self {
        Self {
            espeak_path: "espeak-ng".to_string(),
        }
    }

    /// Create a new fallback with a custom espeak-ng binary path.
    pub fn with_path(espeak_path: String) -> Self {
        Self { espeak_path }
    }

    /// Check if espeak-ng is available on the system.
    ///
    /// Runtime fallback no longer depends on this; it is retained for tests and
    /// offline dictionary-generation tooling.
    pub fn is_available(&self) -> bool {
        Command::new(&self.espeak_path)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Convert a single OOV word to Kokoro-compatible phonemes.
    ///
    /// Returns `Some((phonemes, 2))` on success. Rating 2 matches the former
    /// legacy espeak fallback priority.
    pub fn convert_word(&self, word: &str) -> Option<(String, u8)> {
        embedded_oov_word(word).map(|phonemes| (phonemes, 2))
    }
}

impl Default for EspeakFallback {
    fn default() -> Self {
        Self::new()
    }
}

fn embedded_oov_word(word: &str) -> Option<String> {
    let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
    if trimmed.is_empty() {
        return None;
    }

    let mut parts = Vec::new();
    let mut current = String::new();
    let mut current_kind: Option<CharKind> = None;

    for ch in trimmed.chars() {
        let Some(kind) = CharKind::of(ch) else {
            flush_oov_part(&mut parts, &mut current, current_kind.take());
            continue;
        };

        if current_kind.is_some_and(|k| k != kind) {
            flush_oov_part(&mut parts, &mut current, current_kind.take());
        }
        current_kind = Some(kind);
        current.push(ch);
    }
    flush_oov_part(&mut parts, &mut current, current_kind);

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CharKind {
    Letter,
    Digit,
}

impl CharKind {
    fn of(ch: char) -> Option<Self> {
        if ch.is_ascii_alphabetic() {
            Some(Self::Letter)
        } else if ch.is_ascii_digit() {
            Some(Self::Digit)
        } else {
            None
        }
    }
}

fn flush_oov_part(parts: &mut Vec<String>, current: &mut String, kind: Option<CharKind>) {
    if current.is_empty() {
        return;
    }
    match kind {
        Some(CharKind::Letter) => parts.push(letters_to_phonemes(current)),
        Some(CharKind::Digit) => {
            parts.extend(
                current
                    .chars()
                    .filter_map(digit_name_phonemes)
                    .map(str::to_string),
            );
        }
        None => {}
    }
    current.clear();
}

fn letters_to_phonemes(word: &str) -> String {
    if should_spell_letters(word) {
        return spell_letters(word);
    }

    rough_word_phonemes(&word.to_lowercase())
}

fn should_spell_letters(word: &str) -> bool {
    let letters = word.chars().filter(|c| c.is_ascii_alphabetic()).count();
    letters <= 3 || (letters <= 8 && word.chars().all(|c| c.is_ascii_uppercase()))
}

fn spell_letters(word: &str) -> String {
    word.chars()
        .filter_map(letter_name_phonemes)
        .collect::<Vec<_>>()
        .join(" ")
}

fn rough_word_phonemes(word: &str) -> String {
    let chars: Vec<char> = word.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    let mut stressed = false;

    while i < chars.len() {
        let remaining = &word[i..];
        if let Some((grapheme, phoneme)) = DIGRAPHS
            .iter()
            .find(|(grapheme, _)| remaining.starts_with(*grapheme))
        {
            push_phoneme(&mut out, phoneme, is_vowel_phoneme(phoneme), &mut stressed);
            i += grapheme.len();
            continue;
        }

        let ch = chars[i];
        if ch == 'e' && i == chars.len() - 1 && stressed {
            i += 1;
            continue;
        }

        if let Some(phoneme) = single_letter_phoneme(ch, chars.get(i + 1).copied()) {
            push_phoneme(&mut out, phoneme, is_vowel_phoneme(phoneme), &mut stressed);
        }
        i += ch.len_utf8();
    }

    if out.is_empty() {
        spell_letters(word)
    } else {
        out
    }
}

fn push_phoneme(out: &mut String, phoneme: &str, is_vowel: bool, stressed: &mut bool) {
    if is_vowel && !*stressed {
        out.push('\u{02C8}');
        *stressed = true;
    }
    out.push_str(phoneme);
}

fn is_vowel_phoneme(phoneme: &str) -> bool {
    phoneme.chars().any(|c| {
        matches!(
            c,
            'A' | 'I'
                | 'O'
                | 'W'
                | 'Y'
                | 'Q'
                | 'i'
                | 'u'
                | '\u{00E6}'
                | '\u{0251}'
                | '\u{0254}'
                | '\u{0259}'
                | '\u{025B}'
                | '\u{026A}'
                | '\u{028A}'
                | '\u{028C}'
                | '\u{025C}'
        )
    })
}

const DIGRAPHS: &[(&str, &str)] = &[
    ("tion", "\u{0283}\u{1D4A}n"),
    ("ough", "O"),
    ("augh", "\u{0254}"),
    ("eigh", "A"),
    ("igh", "I"),
    ("air", "\u{025B}\u{0279}"),
    ("ear", "i\u{0259}\u{0279}"),
    ("er", "\u{025C}\u{0279}"),
    ("ir", "\u{025C}\u{0279}"),
    ("ur", "\u{025C}\u{0279}"),
    ("ar", "\u{0251}\u{0279}"),
    ("or", "\u{0254}\u{0279}"),
    ("ch", "\u{02A7}"),
    ("sh", "\u{0283}"),
    ("th", "\u{03B8}"),
    ("ph", "f"),
    ("wh", "w"),
    ("ck", "k"),
    ("ng", "\u{014B}"),
    ("qu", "kw"),
    ("ee", "i"),
    ("ea", "i"),
    ("oo", "u"),
    ("ai", "A"),
    ("ay", "A"),
    ("oa", "O"),
    ("ow", "O"),
    ("ou", "W"),
];

fn single_letter_phoneme(ch: char, next: Option<char>) -> Option<&'static str> {
    match ch {
        'a' => Some("\u{00E6}"),
        'b' => Some("b"),
        'c' if next.is_some_and(|n| matches!(n, 'e' | 'i' | 'y')) => Some("s"),
        'c' => Some("k"),
        'd' => Some("d"),
        'e' => Some("\u{025B}"),
        'f' => Some("f"),
        'g' if next.is_some_and(|n| matches!(n, 'e' | 'i' | 'y')) => Some("\u{02A4}"),
        'g' => Some("\u{0261}"),
        'h' => Some("h"),
        'i' => Some("\u{026A}"),
        'j' => Some("\u{02A4}"),
        'k' => Some("k"),
        'l' => Some("l"),
        'm' => Some("m"),
        'n' => Some("n"),
        'o' => Some("\u{0251}"),
        'p' => Some("p"),
        'q' => Some("k"),
        'r' => Some("\u{0279}"),
        's' => Some("s"),
        't' => Some("t"),
        'u' => Some("\u{028C}"),
        'v' => Some("v"),
        'w' => Some("w"),
        'x' => Some("ks"),
        'y' => Some("i"),
        'z' => Some("z"),
        _ => None,
    }
}

fn letter_name_phonemes(ch: char) -> Option<&'static str> {
    match ch.to_ascii_lowercase() {
        'a' => Some("\u{02C8}A"),
        'b' => Some("b\u{02C8}i"),
        'c' => Some("s\u{02C8}i"),
        'd' => Some("d\u{02C8}i"),
        'e' => Some("\u{02C8}i"),
        'f' => Some("\u{02C8}\u{025B}f"),
        'g' => Some("\u{02A4}\u{02C8}i"),
        'h' => Some("\u{02C8}A\u{02A7}"),
        'i' => Some("\u{02C8}I"),
        'j' => Some("\u{02A4}\u{02C8}A"),
        'k' => Some("k\u{02C8}A"),
        'l' => Some("\u{02C8}\u{025B}l"),
        'm' => Some("\u{02C8}\u{025B}m"),
        'n' => Some("\u{02C8}\u{025B}n"),
        'o' => Some("\u{02C8}O"),
        'p' => Some("p\u{02C8}i"),
        'q' => Some("kj\u{02C8}u"),
        'r' => Some("\u{02C8}\u{0251}\u{0279}"),
        's' => Some("\u{02C8}\u{025B}s"),
        't' => Some("t\u{02C8}i"),
        'u' => Some("j\u{02C8}u"),
        'v' => Some("v\u{02C8}i"),
        'w' => Some("d\u{02C8}\u{028C}b\u{1D4A}l j\u{02CC}u"),
        'x' => Some("\u{02C8}\u{025B}ks"),
        'y' => Some("w\u{02C8}I"),
        'z' => Some("z\u{02C8}i"),
        _ => None,
    }
}

fn digit_name_phonemes(ch: char) -> Option<&'static str> {
    match ch {
        '0' => Some("z\u{02C8}i\u{0279}O"),
        '1' => Some("w\u{02C8}\u{028C}n"),
        '2' => Some("t\u{02C8}u"),
        '3' => Some("\u{03B8}\u{0279}\u{02C8}i"),
        '4' => Some("f\u{02C8}\u{0254}\u{0279}"),
        '5' => Some("f\u{02C8}Iv"),
        '6' => Some("s\u{02C8}\u{026A}ks"),
        '7' => Some("s\u{02C8}\u{025B}v\u{1D4A}n"),
        '8' => Some("\u{02C8}A\u{02A7}"),
        '9' => Some("n\u{02C8}In"),
        _ => None,
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
/// Runtime G2P does not use this. It remains available for offline tooling and
/// compatibility with callers that still want to post-process espeak output.
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
        assert!(result.contains('O'), "Expected O diphthong in: {result}");
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
    fn test_convert_word_embedded_without_espeak() {
        let fb = EspeakFallback::with_path("/definitely/missing/espeak-ng".into());
        let result = fb.convert_word("neologismxyz");
        assert!(result.is_some(), "embedded fallback should cover OOV words");
        let (ps, rating) = result.unwrap();
        assert_eq!(rating, 2);
        assert!(!ps.is_empty());
        assert!(
            ps.contains('\u{02C8}'),
            "Expected embedded fallback to assign primary stress: {}",
            ps
        );
    }

    #[test]
    fn test_convert_word_embedded_handles_digits() {
        let fb = EspeakFallback::with_path("/definitely/missing/espeak-ng".into());
        let (ps, rating) = fb.convert_word("TTS2").unwrap();
        assert_eq!(rating, 2);
        assert!(ps.contains("t\u{02C8}i"), "Expected T spelling in: {ps}");
        assert!(ps.contains("t\u{02C8}u"), "Expected 2 spelling in: {ps}");
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
