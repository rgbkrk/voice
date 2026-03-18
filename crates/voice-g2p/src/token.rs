use std::collections::BTreeSet;

/// Per-token metadata that mirrors misaki's dynamic `Underscore` dict,
/// but as a concrete struct with named fields.
#[derive(Clone, Debug, Default)]
pub struct Underscore {
    /// Whether this token is the head of a compound/phrase.
    pub is_head: bool,
    /// An alias (e.g. from abbreviation expansion).
    pub alias: Option<String>,
    /// Lexical stress weight override.
    pub stress: Option<f32>,
    /// Currency symbol associated with this token.
    pub currency: Option<char>,
    /// Sorted set of single-char number-formatting flags.
    pub num_flags: String,
    /// Whether a space should be inserted before this token's phonemes.
    pub prespace: bool,
    /// Quality rating (lower is better).
    pub rating: Option<u8>,
}

/// A single token produced by the G2P front-end.
#[derive(Clone, Debug)]
pub struct MToken {
    pub text: String,
    pub tag: String,
    pub whitespace: String,
    pub phonemes: Option<String>,
    pub underscore: Underscore,
}

impl MToken {
    /// Create a new token with the given text, tag, and trailing whitespace.
    /// All other fields are set to their defaults.
    pub fn new(
        text: impl Into<String>,
        tag: impl Into<String>,
        whitespace: impl Into<String>,
    ) -> Self {
        Self {
            text: text.into(),
            tag: tag.into(),
            whitespace: whitespace.into(),
            phonemes: None,
            underscore: Underscore::default(),
        }
    }
}

/// Contextual information threaded through English G2P rules.
#[derive(Clone, Debug, Default)]
pub struct TokenContext {
    pub future_vowel: Option<bool>,
    pub future_to: bool,
}

/// Characters treated as diphthongs (or affricates) for stress-weight counting.
/// Each counts as 2; all other phoneme characters count as 1.
const DIPHTHONGS: &[char] = &['A', 'I', 'O', 'Q', 'W', 'Y', '\u{02A4}', '\u{02A7}'];
// ʤ = U+02A4, ʧ = U+02A7

/// Compute the stress weight of a phoneme string.
/// Diphthong / affricate characters count as 2, everything else as 1.
pub fn stress_weight(ps: Option<&str>) -> usize {
    match ps {
        None => 0,
        Some(s) => s
            .chars()
            .map(|c| if DIPHTHONGS.contains(&c) { 2 } else { 1 })
            .sum(),
    }
}

/// Merge a slice of tokens into a single token, combining their text,
/// phonemes, and underscore metadata.
///
/// `unk` controls unknown-phoneme handling:
/// - `None` — the merged token gets `phonemes = None`.
/// - `Some(fallback)` — tokens whose phonemes are `None` use `fallback`;
///   tokens with `prespace` get a space inserted before their phonemes.
///
/// # Panics
///
/// Panics if `tokens` is empty.
pub fn merge_tokens(tokens: &[MToken], unk: Option<&str>) -> MToken {
    assert!(!tokens.is_empty(), "merge_tokens called with empty slice");

    // --- Collect underscore aggregates ---
    let stresses: BTreeSet<u32> = tokens
        .iter()
        .filter_map(|tk| tk.underscore.stress.map(|s: f32| s.to_bits()))
        .collect();

    let currencies: BTreeSet<char> = tokens
        .iter()
        .filter_map(|tk| tk.underscore.currency)
        .collect();

    let ratings: Vec<Option<u8>> = tokens.iter().map(|tk| tk.underscore.rating).collect();

    let num_flags: String = {
        let mut chars: BTreeSet<char> = BTreeSet::new();
        for tk in tokens {
            for c in tk.underscore.num_flags.chars() {
                chars.insert(c);
            }
        }
        chars.into_iter().collect()
    };

    // --- Phonemes ---
    let phonemes = match unk {
        None => None,
        Some(fallback) => {
            let mut out = String::new();
            for tk in tokens {
                if tk.underscore.prespace
                    && !out.is_empty()
                    && !out.ends_with(char::is_whitespace)
                    && tk.phonemes.is_some()
                {
                    out.push(' ');
                }
                match &tk.phonemes {
                    Some(p) => out.push_str(p),
                    None => out.push_str(fallback),
                }
            }
            Some(out)
        }
    };

    // --- Text: join all but last with their whitespace, then last text only ---
    let text = {
        let mut t = String::new();
        for (i, tk) in tokens.iter().enumerate() {
            t.push_str(&tk.text);
            if i < tokens.len() - 1 {
                t.push_str(&tk.whitespace);
            }
        }
        t
    };

    // --- Tag: pick the tag from the token with the highest "case score" ---
    let tag = tokens
        .iter()
        .max_by_key(|tk| {
            tk.text
                .chars()
                .map(|c: char| if c.is_lowercase() { 1usize } else { 2 })
                .sum::<usize>()
        })
        .unwrap()
        .tag
        .clone();

    // --- Merged underscore ---
    let merged_stress = if stresses.len() == 1 {
        Some(f32::from_bits(stresses.into_iter().next().unwrap()))
    } else {
        None
    };

    let merged_currency = if currencies.is_empty() {
        None
    } else {
        currencies.into_iter().max()
    };

    let merged_rating = if ratings.contains(&None) {
        None
    } else {
        ratings.into_iter().flatten().min()
    };

    MToken {
        text,
        tag,
        whitespace: tokens.last().unwrap().whitespace.clone(),
        phonemes,
        underscore: Underscore {
            is_head: tokens[0].underscore.is_head,
            alias: None,
            stress: merged_stress,
            currency: merged_currency,
            num_flags,
            prespace: tokens[0].underscore.prespace,
            rating: merged_rating,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_weight_none() {
        assert_eq!(stress_weight(None), 0);
    }

    #[test]
    fn test_stress_weight_plain() {
        // 3 regular chars
        assert_eq!(stress_weight(Some("abc")), 3);
    }

    #[test]
    fn test_stress_weight_diphthongs() {
        // A counts 2, b counts 1 => 3
        assert_eq!(stress_weight(Some("Ab")), 3);
    }

    #[test]
    fn test_merge_tokens_basic() {
        let tokens = vec![
            MToken {
                text: "can".into(),
                tag: "NN".into(),
                whitespace: "".into(),
                phonemes: Some("k\u{00E6}n".into()),
                underscore: Underscore::default(),
            },
            MToken {
                text: "not".into(),
                tag: "RB".into(),
                whitespace: " ".into(),
                phonemes: Some("n\u{0251}t".into()),
                underscore: Underscore::default(),
            },
        ];

        let merged = merge_tokens(&tokens, Some("?"));
        assert_eq!(merged.text, "cannot");
        assert_eq!(merged.phonemes, Some("k\u{00E6}nn\u{0251}t".into()));
        assert_eq!(merged.whitespace, " ");
    }

    #[test]
    fn test_merge_tokens_prespace() {
        let mut tk1 = MToken::new("a", "DT", " ");
        tk1.phonemes = Some("\u{0259}".into());

        let mut tk2 = MToken::new("b", "NN", "");
        tk2.phonemes = Some("bi".into());
        tk2.underscore.prespace = true;

        let merged = merge_tokens(&[tk1, tk2], Some("?"));
        assert_eq!(merged.phonemes, Some("\u{0259} bi".into()));
    }

    #[test]
    fn test_merge_tokens_unk_none() {
        let tokens = vec![MToken::new("hello", "NN", " ")];
        let merged = merge_tokens(&tokens, None);
        assert!(merged.phonemes.is_none());
    }

    #[test]
    fn test_merge_tokens_unk_fallback() {
        let tokens = vec![MToken::new("xyz", "NN", "")];
        let merged = merge_tokens(&tokens, Some("?"));
        assert_eq!(merged.phonemes, Some("?".into()));
    }

    #[test]
    fn test_merge_tokens_rating() {
        let mut tk1 = MToken::new("a", "X", "");
        tk1.underscore.rating = Some(3);
        let mut tk2 = MToken::new("b", "X", "");
        tk2.underscore.rating = Some(5);

        let merged = merge_tokens(&[tk1, tk2], None);
        assert_eq!(merged.underscore.rating, Some(3));
    }

    #[test]
    fn test_merge_tokens_rating_with_none() {
        let mut tk1 = MToken::new("a", "X", "");
        tk1.underscore.rating = Some(3);
        let tk2 = MToken::new("b", "X", "");
        // tk2.underscore.rating is None by default

        let merged = merge_tokens(&[tk1, tk2], None);
        assert_eq!(merged.underscore.rating, None);
    }
}
