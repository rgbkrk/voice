pub mod espeak;
pub mod lexicon;
pub mod number;
pub mod stress;
pub mod tagger;
pub mod token;
pub mod tokenizer;

use std::collections::HashMap;
use std::sync::OnceLock;

use espeak::EspeakFallback;
use lexicon::Lexicon;
use stress::{apply_stress, CONSONANTS, NON_QUOTE_PUNCTS, PRIMARY_STRESS, SUBTOKEN_JUNKS, VOWELS};
use token::{merge_tokens, MToken, TokenContext};
use tokenizer::TokenOrGroup;

#[derive(Debug, thiserror::Error)]
pub enum G2pError {
    #[error("espeak-ng not found. Install with: brew install espeak-ng")]
    EspeakNotFound,
    #[error("espeak-ng failed: {0}")]
    EspeakFailed(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Configuration for external tool paths used by the G2P pipeline.
#[derive(Debug, Clone)]
pub struct G2PConfig {
    /// Path to the `espeak-ng` binary for fallback pronunciation.
    /// Defaults to `"espeak-ng"` (PATH lookup).
    pub espeak_path: String,
}

impl Default for G2PConfig {
    fn default() -> Self {
        Self {
            espeak_path: "espeak-ng".to_string(),
        }
    }
}

/// The main G2P pipeline, ported from misaki's `en.G2P.__call__()`.
pub struct G2P {
    lexicon: Lexicon,
    fallback: EspeakFallback,
    unk: String,
    overrides: HashMap<String, String>,
}

fn global_g2p() -> &'static G2P {
    static INSTANCE: OnceLock<G2P> = OnceLock::new();
    INSTANCE.get_or_init(G2P::new)
}

impl G2P {
    pub fn new() -> Self {
        Self::with_config(G2PConfig::default())
    }

    pub fn with_config(config: G2PConfig) -> Self {
        Self {
            lexicon: Lexicon::new(),
            fallback: EspeakFallback::with_path(config.espeak_path),
            unk: String::new(),
            overrides: Self::builtin_overrides(),
        }
    }

    /// Words whose default lexicon/espeak phonemes are wrong or misleading.
    fn builtin_overrides() -> HashMap<String, String> {
        const ENTRIES: &[(&str, &str)] = &[
            ("demo", "dˈɛmO"),
            ("demos", "dˈɛmOz"),
            ("demultiplex", "dˌimˈʌltɪplɛks"),
            ("demultiplexing", "dˌimˈʌltɪplɛksɪŋ"),
            ("demux", "dˌimˈʌks"),
            ("demuxing", "dˌimˈʌksɪŋ"),
            ("jupyter", "ʤˈupɪTəɹ"),
            ("nteract", "ˈɛntəɹˌækt"),
            ("todo", "tˈudu"),
            // Developer acronyms and initialisms
            ("ipynb", "nˈOtbˌʊk fˈIl"),
            ("pr", "pˈi ˈɑɹ"),
            ("prs", "pˈi ˈɑɹz"),
            ("rxjs", "ˈɑɹ ˈɛks ʤˈA ˈɛs"),
            ("tsconfig", "tˈi ˈɛs kˌɑnfˈɪɡ"),
            ("vitest", "vˈItˌɛst"),
        ];
        ENTRIES
            .iter()
            .map(|(k, v)| ((*k).into(), (*v).into()))
            .collect()
    }

    /// Set custom word-to-phoneme overrides (builder pattern).
    ///
    /// Overrides map lowercase words to phoneme strings, checked before
    /// the lexicon and espeak fallback.
    pub fn with_overrides(mut self, overrides: HashMap<String, String>) -> Self {
        self.overrides.extend(overrides);
        self
    }

    /// Full pipeline: text -> phoneme string.
    ///
    /// Mirrors misaki `G2P.__call__()` from en.py:679-738.
    pub fn convert(&self, text: &str) -> Result<String, G2pError> {
        // 1. Tokenize and POS-tag (embedded perceptron tagger)
        let tokens = tokenizer::tokenize(text);

        // 2. fold_left: merge non-head tokens
        let tokens = tokenizer::fold_left(tokens);

        // 3. retokenize: subtokenize, handle punctuation/currency
        let mut items = tokenizer::retokenize(tokens);

        // 4. Right-to-left resolution with TokenContext
        let mut ctx = TokenContext::default();

        for item in items.iter_mut().rev() {
            match item {
                TokenOrGroup::Single(ref mut w) => {
                    self.resolve_single_token(w, &ctx);
                    ctx = Self::token_context(&ctx, w.phonemes.as_deref(), w);
                }
                TokenOrGroup::Group(ref mut group) => {
                    self.resolve_group(group, &ctx);
                    if let Some(first) = group.first() {
                        ctx = Self::token_context(&ctx, first.phonemes.as_deref(), first);
                    }
                }
            }
        }

        // 5. Merge groups into single tokens
        let tokens: Vec<MToken> = items
            .into_iter()
            .map(|item| match item {
                TokenOrGroup::Single(tok) => tok,
                TokenOrGroup::Group(group) => merge_tokens(&group, Some(&self.unk)),
            })
            .collect();

        // 6. Legacy conversion: ɾ->T, ʔ->t
        let result: String = tokens
            .iter()
            .map(|tk| {
                let ps = match &tk.phonemes {
                    Some(p) => p.replace('ɾ', "T").replace('ʔ', "t"),
                    None => self.unk.clone(),
                };
                format!("{}{}", ps, tk.whitespace)
            })
            .collect();

        Ok(result)
    }

    /// Resolve a single (non-grouped) token.
    fn resolve_single_token(&self, w: &mut MToken, ctx: &TokenContext) {
        if w.phonemes.is_some() {
            return;
        }

        // Check custom overrides before lexicon/espeak fallback
        let lookup_key = w.text.to_lowercase();
        if let Some(ps) = self.overrides.get(&lookup_key) {
            w.phonemes = Some(ps.clone());
            w.underscore.rating = Some(5); // highest priority
            return;
        }
        let (ps, rating) = self.lexicon.call(
            &w.text,
            w.underscore.alias.as_deref(),
            &w.tag,
            w.underscore.stress,
            w.underscore.currency,
            w.underscore.is_head,
            &w.underscore.num_flags,
            ctx,
        );
        if let Some(ps) = ps {
            w.phonemes = Some(ps);
            w.underscore.rating = rating;
            return;
        }

        if let Some((ps, rating)) = self.fallback.convert_word(&w.text) {
            w.phonemes = Some(ps);
            w.underscore.rating = Some(rating);
        }
    }

    /// Resolve a group of subtokens using the left-expand/right-shrink algorithm.
    ///
    /// Ported from en.py:694-731.
    fn resolve_group(&self, group: &mut [MToken], ctx: &TokenContext) {
        // Check overrides for the whole merged text before the expand/shrink loop
        let merged_text: String = group.iter().map(|tk| tk.text.as_str()).collect();
        let lookup_key = merged_text.to_lowercase();
        if let Some(ps) = self.overrides.get(&lookup_key) {
            group[0].phonemes = Some(ps.clone());
            group[0].underscore.rating = Some(5);
            for tk in group.iter_mut().skip(1) {
                tk.phonemes = Some(String::new());
                tk.underscore.rating = Some(5);
            }
            return;
        }

        let n = group.len();
        let mut left = 0;
        let mut right = n;
        let mut should_fallback = false;

        while left < right {
            let has_existing = group[left..right]
                .iter()
                .any(|tk| tk.underscore.alias.is_some() || tk.phonemes.is_some());

            let (ps, rating) = if has_existing {
                (None, None)
            } else {
                let merged = merge_tokens(&group[left..right], None);
                self.lexicon.call(
                    &merged.text,
                    merged.underscore.alias.as_deref(),
                    &merged.tag,
                    merged.underscore.stress,
                    merged.underscore.currency,
                    merged.underscore.is_head,
                    &merged.underscore.num_flags,
                    ctx,
                )
            };

            if let Some(ps) = ps {
                group[left].phonemes = Some(ps);
                group[left].underscore.rating = rating;
                for x in &mut group[left + 1..right] {
                    x.phonemes = Some(String::new());
                    x.underscore.rating = rating;
                }
                right = left;
                left = 0;
            } else if left + 1 < right {
                left += 1;
            } else {
                right -= 1;
                let tk = &mut group[right];
                if tk.phonemes.is_none() {
                    if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                        tk.phonemes = Some(String::new());
                        tk.underscore.rating = Some(3);
                    } else {
                        should_fallback = true;
                        break;
                    }
                }
                left = 0;
            }
        }

        if should_fallback {
            let merged = merge_tokens(group, None);
            if let Some((ps, rating)) = self.fallback.convert_word(&merged.text) {
                group[0].phonemes = Some(ps);
                group[0].underscore.rating = Some(rating);
                for j in 1..group.len() {
                    group[j].phonemes = Some(String::new());
                    group[j].underscore.rating = group[0].underscore.rating;
                }
            }
        } else {
            Self::resolve_tokens(group);
        }
    }

    /// Update TokenContext based on resolved phonemes and token.
    ///
    /// Ported from en.py:646-650.
    fn token_context(ctx: &TokenContext, ps: Option<&str>, token: &MToken) -> TokenContext {
        let mut vowel = ctx.future_vowel;

        if let Some(ps) = ps {
            for c in ps.chars() {
                let is_vowel = VOWELS.contains(c);
                let is_consonant = CONSONANTS.contains(c);
                let is_punct = NON_QUOTE_PUNCTS.contains(c);

                if is_vowel || is_consonant || is_punct {
                    vowel = if is_punct { None } else { Some(is_vowel) };
                    break;
                }
            }
        }

        let future_to = matches!(token.text.as_str(), "to" | "To")
            || (token.text == "TO" && matches!(token.tag.as_str(), "TO" | "IN"));

        TokenContext {
            future_vowel: vowel,
            future_to,
        }
    }

    /// Normalize stress across a group of resolved subtokens.
    ///
    /// Ported from en.py:652-677.
    fn resolve_tokens(tokens: &mut [MToken]) {
        if tokens.is_empty() {
            return;
        }

        let text: String = tokens
            .iter()
            .enumerate()
            .map(|(i, tk)| {
                if i < tokens.len() - 1 {
                    format!("{}{}", tk.text, tk.whitespace)
                } else {
                    tk.text.clone()
                }
            })
            .collect();

        let has_space = text.contains(' ') || text.contains('/');
        let char_classes: std::collections::HashSet<u8> = text
            .chars()
            .filter(|c| !SUBTOKEN_JUNKS.contains(*c))
            .map(|c| {
                if c.is_alphabetic() {
                    0
                } else if c.is_ascii_digit() {
                    1
                } else {
                    2
                }
            })
            .collect();
        let prespace = has_space || char_classes.len() > 1;

        let n = tokens.len();
        for (i, tk) in tokens.iter_mut().enumerate() {
            if tk.phonemes.is_none() {
                let last = i == n - 1;
                if last
                    && tk.text.len() == 1
                    && NON_QUOTE_PUNCTS.contains(tk.text.chars().next().unwrap_or(' '))
                {
                    tk.phonemes = Some(tk.text.clone());
                    tk.underscore.rating = Some(3);
                } else if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                    tk.phonemes = Some(String::new());
                    tk.underscore.rating = Some(3);
                }
            } else if i > 0 && !tk.underscore.prespace {
                tk.underscore.prespace = prespace;
            }
        }

        if prespace {
            return;
        }

        let indices: Vec<(bool, usize, usize)> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, tk)| {
                tk.phonemes.as_ref().filter(|p| !p.is_empty()).map(|p| {
                    let has_primary = p.contains(PRIMARY_STRESS);
                    let weight = token::stress_weight(Some(p));
                    (has_primary, weight, i)
                })
            })
            .collect();

        if indices.len() == 2 && tokens[indices[0].2].text.len() == 1 {
            let i = indices[1].2;
            if let Some(ref ps) = tokens[i].phonemes {
                tokens[i].phonemes = Some(apply_stress(ps, Some(-0.5)));
            }
            return;
        }

        if indices.len() < 2 {
            return;
        }
        let primary_count: usize = indices.iter().filter(|(b, _, _)| *b).count();
        if primary_count <= indices.len().div_ceil(2) {
            return;
        }

        let mut sorted = indices.clone();
        sorted.sort();
        let half = sorted.len() / 2;
        for &(_, _, i) in &sorted[..half] {
            if let Some(ref ps) = tokens[i].phonemes {
                tokens[i].phonemes = Some(apply_stress(ps, Some(-0.5)));
            }
        }
    }
}

impl Default for G2P {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Public API (backward-compatible)
// ---------------------------------------------------------------------------

/// Convert English text to a Kokoro-compatible phoneme string.
///
/// Uses misaki-style dictionary lookup with espeak-ng fallback for unknown words.
pub fn english_to_phonemes(text: &str) -> Result<String, G2pError> {
    global_g2p().convert(text)
}

/// Convert English text to phonemes with custom word overrides.
///
/// Overrides map lowercase words to phoneme strings, checked before
/// the lexicon and espeak fallback.
pub fn english_to_phonemes_with_overrides(
    text: &str,
    overrides: &HashMap<String, String>,
) -> Result<String, G2pError> {
    let g2p = G2P::new().with_overrides(overrides.clone());
    g2p.convert(text)
}

/// Post-process espeak-ng IPA output into Kokoro phoneme format.
///
/// Kept for backward compatibility. New code should use `english_to_phonemes()`.
pub fn espeak_ipa_to_kokoro(ipa: &str) -> String {
    let mut s = ipa.to_string();

    s = s.replace("dʒ", "ʤ");
    s = s.replace("tʃ", "ʧ");
    s = s.replace("ɜːɹ", "ɜɹ");
    s = s.replace("ɜː", "ɜɹ");
    s = s.replace("aɪ", "I");
    s = s.replace("aʊ", "W");
    s = s.replace("eɪ", "A");
    s = s.replace("oʊ", "O");
    s = s.replace("ɔɪ", "Y");
    s = s.replace('ː', "");
    s = s.replace('ɾ', "T");

    s
}

/// Split text into chunks whose phoneme representations fit within the model's
/// 510-character context limit.
pub fn text_to_phoneme_chunks(text: &str) -> Result<Vec<String>, G2pError> {
    const MAX_PHONEME_LEN: usize = 500;

    let mut chunks = Vec::new();

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

/// Split text into chunks whose phoneme representations fit within the model's
/// 510-character context limit, with custom word-to-phoneme overrides.
///
/// Overrides map lowercase words to phoneme strings, checked before
/// the lexicon and espeak fallback.
pub fn text_to_phoneme_chunks_with_overrides(
    text: &str,
    overrides: &HashMap<String, String>,
) -> Result<Vec<String>, G2pError> {
    const MAX_PHONEME_LEN: usize = 500;

    let mut chunks = Vec::new();

    for paragraph in text.split('\n') {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }

        let phonemes = english_to_phonemes_with_overrides(paragraph, overrides)?;
        if phonemes.len() <= MAX_PHONEME_LEN {
            chunks.push(phonemes);
            continue;
        }

        let sentences = split_sentences(paragraph);
        let mut current_phonemes = String::new();

        for sentence in &sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let sent_phonemes = english_to_phonemes_with_overrides(sentence, overrides)?;

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
        let input = "həlˈoʊ wˈɜːld";
        let expected = "həlˈO wˈɜɹld";
        assert_eq!(espeak_ipa_to_kokoro(input), expected);
    }

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("Hello world. How are you? I'm fine!");
        assert_eq!(
            sentences,
            vec!["Hello world.", " How are you?", " I'm fine!"]
        );
    }

    #[test]
    fn test_g2p_convert_hello() {
        let g2p = G2P::new();
        let result = g2p.convert("hello").unwrap();
        assert!(!result.is_empty());
        assert!(
            result.contains('O') || result.contains('o'),
            "Expected phonemes for 'hello', got: {}",
            result
        );
    }

    #[test]
    fn test_g2p_convert_sentence() {
        let g2p = G2P::new();
        let result = g2p.convert("Hello world").unwrap();
        assert!(!result.is_empty());
        assert!(
            result.contains(' '),
            "Expected space between words in: {}",
            result
        );
    }

    #[test]
    fn test_g2p_convert_the_context() {
        let g2p = G2P::new();
        let result = g2p.convert("the apple").unwrap();
        assert!(
            result.contains("ði"),
            "Expected 'ði' (the before vowel) in: {}",
            result
        );
    }

    #[test]
    fn test_g2p_convert_number() {
        let g2p = G2P::new();
        let result = g2p.convert("42").unwrap();
        assert!(!result.is_empty(), "Should produce phonemes for numbers");
    }

    #[test]
    fn test_english_to_phonemes_api() {
        let result = english_to_phonemes("hello world");
        assert!(result.is_ok());
        let phonemes = result.unwrap();
        assert!(!phonemes.is_empty());
    }

    // -- Punctuation preservation tests --------------------------------------

    #[test]
    fn test_period_preserved() {
        let result = english_to_phonemes("Hello.").unwrap();
        assert!(
            result.contains('.'),
            "Period should appear in phonemes: {result}"
        );
    }

    #[test]
    fn test_comma_preserved() {
        let result = english_to_phonemes("Hello, world.").unwrap();
        assert!(
            result.contains(','),
            "Comma should appear in phonemes: {result}"
        );
        assert!(
            result.contains('.'),
            "Period should appear in phonemes: {result}"
        );
    }

    #[test]
    fn test_question_mark_preserved() {
        let result = english_to_phonemes("Hello?").unwrap();
        assert!(
            result.contains('?'),
            "Question mark should appear in phonemes: {result}"
        );
    }

    #[test]
    fn test_exclamation_preserved() {
        let result = english_to_phonemes("Hello!").unwrap();
        assert!(
            result.contains('!'),
            "Exclamation mark should appear in phonemes: {result}"
        );
    }

    #[test]
    fn test_two_sentences_have_period_between() {
        let result = english_to_phonemes("Hello. World.").unwrap();
        // Should have at least one period (ideally two) in the phoneme output
        let period_count = result.chars().filter(|c| *c == '.').count();
        assert!(
            period_count >= 1,
            "Expected period(s) between sentences, got: {result}"
        );
    }

    #[test]
    fn test_mixed_punctuation() {
        let result = english_to_phonemes("Wait! What? Really.").unwrap();
        assert!(
            result.contains('!'),
            "Exclamation should appear in phonemes: {result}"
        );
        assert!(
            result.contains('?'),
            "Question mark should appear in phonemes: {result}"
        );
        assert!(
            result.contains('.'),
            "Period should appear in phonemes: {result}"
        );
    }

    #[test]
    fn test_semicolon_preserved() {
        let result = english_to_phonemes("Hello; world.").unwrap();
        assert!(
            result.contains(';'),
            "Semicolon should appear in phonemes: {result}"
        );
    }

    #[test]
    fn test_builtin_overrides() {
        let g2p = G2P::new();
        assert_eq!(g2p.convert("demos").unwrap(), "dˈɛmOz");
        assert_eq!(g2p.convert("demo").unwrap(), "dˈɛmO");
        assert_eq!(g2p.convert("TODO").unwrap(), "tˈudu");
        assert_eq!(g2p.convert("demuxing").unwrap(), "dˌimˈʌksɪŋ");
        assert_eq!(g2p.convert("demux").unwrap(), "dˌimˈʌks");
        assert_eq!(g2p.convert("demultiplexing").unwrap(), "dˌimˈʌltɪplɛksɪŋ");
        assert_eq!(g2p.convert("demultiplex").unwrap(), "dˌimˈʌltɪplɛks");
        assert_eq!(g2p.convert("Jupyter").unwrap(), "ʤˈupɪTəɹ");
        assert_eq!(g2p.convert("nteract").unwrap(), "ˈɛntəɹˌækt");
        assert_eq!(g2p.convert("vitest").unwrap(), "vˈItˌɛst");
        assert_eq!(g2p.convert("tsconfig").unwrap(), "tˈi ˈɛs kˌɑnfˈɪɡ");
        assert_eq!(g2p.convert("ipynb").unwrap(), "nˈOtbˌʊk fˈIl");
        assert_eq!(g2p.convert("PR").unwrap(), "pˈi ˈɑɹ");
        assert_eq!(g2p.convert("PRs").unwrap(), "pˈi ˈɑɹz");
    }

    // -- camelCase tests ------------------------------------------------------

    #[test]
    fn test_camel_case_spaced_phonemes() {
        let g2p = G2P::new();

        // Two-part camelCase
        let result = g2p.convert("useEffect").unwrap();
        assert!(
            result.contains(' '),
            "camelCase should produce space-separated phonemes: {result}"
        );

        // Three-part camelCase
        let result = g2p.convert("fromTauriEvent").unwrap();
        let spaces = result.chars().filter(|c| *c == ' ').count();
        assert!(
            spaces >= 2,
            "Three-part camelCase should have 2+ spaces: {result}"
        );

        // Single word should not gain a space
        let result = g2p.convert("hello").unwrap();
        assert!(
            !result.contains(' '),
            "Single word should not have spaces: {result}"
        );
    }
}
