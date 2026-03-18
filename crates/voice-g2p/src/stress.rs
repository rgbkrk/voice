//! Phoneme stress constants and the `apply_stress` function,
//! ported from misaki's `en.py`.

// ---------------------------------------------------------------------------
// Character-set constants (small sets — plain &str + .contains() is fine)
// ---------------------------------------------------------------------------

pub const DIPHTHONGS: &str = "AIOQWYʤʧ";

pub const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";

pub const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";

pub const US_VOCAB: &str = "AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔ";
pub const GB_VOCAB: &str = "AIQWYabdfhijklmnpstuvwzðŋɑɒɔəɛɜɡɪɹʃʊʌʒʤʧˈˌːθᵊ";

pub const STRESSES: &str = "ˌˈ";
pub const PRIMARY_STRESS: char = 'ˈ';
pub const SECONDARY_STRESS: char = 'ˌ';

pub const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";

pub const SUBTOKEN_JUNKS: &str = "',-._''/";
pub const PUNCTS: &str = ";:,.!?—…\u{201C}\u{201D}\u{201E}";
pub const NON_QUOTE_PUNCTS: &str = ";:,.!?—…";

pub const PUNCT_TAGS: &[&str] = &[
    ".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP",
];

/// Mapping from punct tags to their phoneme representations.
pub fn punct_tag_phoneme(tag: &str) -> Option<&'static str> {
    match tag {
        "-LRB-" => Some("("),
        "-RRB-" => Some(")"),
        "``" => Some("\u{201C}"),   // left double quotation mark
        "\"\"" => Some("\u{201D}"), // right double quotation mark
        "''" => Some("\u{201D}"),   // right double quotation mark
        _ => None,
    }
}

/// Currency symbol -> (unit_name, subunit_name)
pub fn currency_names(symbol: char) -> Option<(&'static str, &'static str)> {
    match symbol {
        '$' => Some(("dollar", "cent")),
        '£' => Some(("pound", "pence")),
        '€' => Some(("euro", "cent")),
        _ => None,
    }
}

pub const ORDINALS: &[&str] = &["st", "nd", "rd", "th"];

pub fn add_symbol_name(c: char) -> Option<&'static str> {
    match c {
        '.' => Some("dot"),
        '/' => Some("slash"),
        _ => None,
    }
}

pub fn symbol_name(c: char) -> Option<&'static str> {
    match c {
        '%' => Some("percent"),
        '&' => Some("and"),
        '+' => Some("plus"),
        '@' => Some("at"),
        _ => None,
    }
}

/// Ordinals recognized in the LEXICON_ORDS set: `'`, `-`, `A-Z`, `a-z`.
pub fn is_lexicon_ord(c: char) -> bool {
    matches!(c, '\'' | '-' | 'A'..='Z' | 'a'..='z')
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Weight of a phoneme string: diphthongs count as 2, everything else as 1.
pub fn stress_weight(ps: &str) -> usize {
    if ps.is_empty() {
        return 0;
    }
    ps.chars()
        .map(|c| if DIPHTHONGS.contains(c) { 2 } else { 1 })
        .sum()
}

fn contains_vowel(ps: &str) -> bool {
    ps.chars().any(|c| VOWELS.contains(c))
}

/// Move each stress mark so that it immediately precedes its nearest
/// following vowel (sorted by position).
fn restress(ps: &str) -> String {
    let chars: Vec<char> = ps.chars().collect();
    // (sort_key, char)  — we use f64 keys so we can insert at i - 0.5
    let mut ips: Vec<(f64, char)> = chars
        .iter()
        .enumerate()
        .map(|(i, &c)| (i as f64, c))
        .collect();

    // For every stress mark, find the next vowel index.
    let stress_moves: Vec<(usize, usize)> = chars
        .iter()
        .enumerate()
        .filter(|(_, c)| STRESSES.contains(**c))
        .filter_map(|(i, _)| {
            chars[i..]
                .iter()
                .enumerate()
                .find(|(_, c)| VOWELS.contains(**c))
                .map(|(offset, _)| (i, i + offset))
        })
        .collect();

    for (i, vowel_idx) in stress_moves {
        ips[i].0 = vowel_idx as f64 - 0.5;
    }

    ips.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    ips.iter().map(|(_, c)| c).collect()
}

// ---------------------------------------------------------------------------
// apply_stress
// ---------------------------------------------------------------------------

/// Apply a stress level to a phoneme string.
///
/// * `None`      — return `ps` unchanged
/// * `< -1.0`    — strip all stress marks
/// * `-1.0`      — demote primary to secondary (strip secondary)
/// * `0.0 / -0.5` when primary already present — same as -1
/// * `0.0 / 0.5 / 1.0` when no stress present — prepend secondary + restress
/// * `>= 1.0` when secondary present but no primary — promote to primary
/// * `> 1.0` when no stress present — prepend primary + restress
pub fn apply_stress(ps: &str, stress: Option<f32>) -> String {
    let stress = match stress {
        Some(s) => s,
        None => return ps.to_string(),
    };

    let has_primary = ps.contains(PRIMARY_STRESS);
    let has_secondary = ps.contains(SECONDARY_STRESS);
    let has_any_stress = has_primary || has_secondary;

    if stress < -1.0 {
        // Strip all stress marks.
        ps.replace([PRIMARY_STRESS, SECONDARY_STRESS], "")
    } else if (stress - (-1.0)).abs() < f32::EPSILON
        || ((stress == 0.0 || stress == -0.5) && has_primary)
    {
        // Demote: remove secondary, then replace primary with secondary.
        ps.replace(SECONDARY_STRESS, "")
            .replace(PRIMARY_STRESS, &SECONDARY_STRESS.to_string())
    } else if [0.0_f32, 0.5, 1.0]
        .iter()
        .any(|v| (stress - v).abs() < f32::EPSILON)
        && !has_any_stress
    {
        if !contains_vowel(ps) {
            return ps.to_string();
        }
        let mut s = String::new();
        s.push(SECONDARY_STRESS);
        s.push_str(ps);
        restress(&s)
    } else if stress >= 1.0 && !has_primary && has_secondary {
        ps.replace(SECONDARY_STRESS, &PRIMARY_STRESS.to_string())
    } else if stress > 1.0 && !has_any_stress {
        if !contains_vowel(ps) {
            return ps.to_string();
        }
        let mut s = String::new();
        s.push(PRIMARY_STRESS);
        s.push_str(ps);
        restress(&s)
    } else {
        ps.to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- stress_weight -------------------------------------------------------

    #[test]
    fn stress_weight_empty() {
        assert_eq!(stress_weight(""), 0);
    }

    #[test]
    fn stress_weight_plain() {
        // 'b' is not a diphthong → weight 1 per char
        assert_eq!(stress_weight("bk"), 2);
    }

    #[test]
    fn stress_weight_with_diphthongs() {
        // 'A' is a diphthong (2), 'b' is not (1)
        assert_eq!(stress_weight("Ab"), 3);
    }

    // -- apply_stress: None → identity ---------------------------------------

    #[test]
    fn stress_none_returns_unchanged() {
        assert_eq!(apply_stress("hˈɛloʊ", None), "hˈɛloʊ");
    }

    // -- apply_stress: < -1 → strip all marks --------------------------------

    #[test]
    fn stress_strip_all() {
        assert_eq!(apply_stress("hˈɛˌloʊ", Some(-2.0)), "hɛloʊ");
    }

    // -- apply_stress: -1 → demote primary to secondary ----------------------

    #[test]
    fn stress_demote_primary() {
        let result = apply_stress("hˈɛloʊ", Some(-1.0));
        assert!(
            result.contains(SECONDARY_STRESS),
            "should contain secondary stress"
        );
        assert!(
            !result.contains(PRIMARY_STRESS),
            "should not contain primary stress"
        );
        assert_eq!(result, "hˌɛloʊ");
    }

    // -- apply_stress: 0 with primary already present → demote ---------------

    #[test]
    fn stress_zero_with_primary_demotes() {
        let result = apply_stress("hˈɛloʊ", Some(0.0));
        assert_eq!(result, "hˌɛloʊ");
    }

    // -- apply_stress: 0 with no stress → add secondary + restress -----------

    #[test]
    fn stress_zero_no_stress_adds_secondary() {
        let result = apply_stress("hɛloʊ", Some(0.0));
        assert!(result.contains(SECONDARY_STRESS));
        // The secondary stress should appear before the first vowel.
        let stress_pos = result.find(SECONDARY_STRESS).unwrap();
        let vowel_pos = result
            .char_indices()
            .find(|(_, c)| VOWELS.contains(*c))
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            stress_pos < vowel_pos,
            "secondary stress ({}) should precede first vowel ({})",
            stress_pos,
            vowel_pos
        );
    }

    // -- apply_stress: 1 with secondary but no primary → promote -------------

    #[test]
    fn stress_one_promotes_secondary() {
        let result = apply_stress("hˌɛloʊ", Some(1.0));
        assert_eq!(result, "hˈɛloʊ");
    }

    // -- apply_stress: > 1 with no stress → add primary + restress -----------

    #[test]
    fn stress_high_no_stress_adds_primary() {
        let result = apply_stress("hɛloʊ", Some(2.0));
        assert!(result.contains(PRIMARY_STRESS));
        assert!(!result.contains(SECONDARY_STRESS));
    }

    // -- apply_stress: no vowels → return unchanged --------------------------

    #[test]
    fn stress_no_vowels_unchanged() {
        // Consonants only — stress insertion branches should bail.
        assert_eq!(apply_stress("bkd", Some(0.0)), "bkd");
        assert_eq!(apply_stress("bkd", Some(2.0)), "bkd");
    }

    // -- apply_stress: fallthrough → return unchanged ------------------------

    #[test]
    fn stress_fallthrough_unchanged() {
        // Already has primary, stress=1.5 — doesn't match any branch that
        // would change it.
        let input = "hˈɛloʊ";
        assert_eq!(apply_stress(input, Some(1.5)), input);
    }

    // -- restress moves stress before vowel ----------------------------------

    #[test]
    fn restress_basic() {
        // secondary stress at start, first vowel is 'ɛ' at index 1
        let result = restress("ˌhɛloʊ");
        // Stress should move to just before 'ɛ'
        let chars: Vec<char> = result.chars().collect();
        let stress_idx = chars.iter().position(|&c| c == SECONDARY_STRESS).unwrap();
        let vowel_idx = chars.iter().position(|&c| VOWELS.contains(c)).unwrap();
        assert_eq!(
            stress_idx + 1,
            vowel_idx,
            "stress should be immediately before first vowel"
        );
    }
}
