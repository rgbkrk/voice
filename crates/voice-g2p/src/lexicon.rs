/// Lexicon-based dictionary lookup for English G2P,
/// ported from misaki's `Lexicon` class in `en.py`.
use std::collections::HashMap;

use serde::Deserialize;

use crate::number::{int_to_ordinal, int_to_words, int_to_year};
use crate::stress::{
    apply_stress, currency_names, is_lexicon_ord, ORDINALS, PRIMARY_STRESS, SECONDARY_STRESS,
    US_TAUS,
};
use crate::token::TokenContext;

// ---------------------------------------------------------------------------
// Dictionary entry types
// ---------------------------------------------------------------------------

/// A single dictionary entry: either a plain phoneme string or a POS-tagged map.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum LexEntry {
    Simple(String),
    Tagged(HashMap<String, Option<String>>),
}

// ---------------------------------------------------------------------------
// Embedded dictionaries
// ---------------------------------------------------------------------------

const US_GOLD_JSON: &str = include_str!("../data/us_gold.json");
const US_SILVER_JSON: &str = include_str!("../data/us_silver.json");

// ---------------------------------------------------------------------------
// Symbol tables (mirrors Python ADD_SYMBOLS / SYMBOLS)
// ---------------------------------------------------------------------------

fn add_symbol_word(word: &str) -> Option<&'static str> {
    match word {
        "." => Some("dot"),
        "/" => Some("slash"),
        _ => None,
    }
}

fn symbol_word(word: &str) -> Option<&'static str> {
    match word {
        "%" => Some("percent"),
        "&" => Some("and"),
        "+" => Some("plus"),
        "@" => Some("at"),
        _ => None,
    }
}

fn is_symbol(word: &str) -> bool {
    matches!(word, "%" | "&" | "+" | "@")
}

fn is_add_symbol(word: &str) -> bool {
    matches!(word, "." | "/")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_digit_str(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// Check whether a word is all ASCII-alphabetic.
fn is_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphabetic())
}

/// Check whether all characters in a word are in the LEXICON_ORDS set.
fn all_lexicon_ords(s: &str) -> bool {
    s.chars().all(is_lexicon_ord)
}

// ---------------------------------------------------------------------------
// Lexicon
// ---------------------------------------------------------------------------

pub struct Lexicon {
    /// Cap-stress values: (lowercase_capitalized, ALL_UPPER)
    cap_stresses: (f32, f32),
    golds: HashMap<String, LexEntry>,
    silvers: HashMap<String, LexEntry>,
}

impl Default for Lexicon {
    fn default() -> Self {
        Self::new()
    }
}

impl Lexicon {
    /// Create a new US-English lexicon (british=false hardcoded).
    pub fn new() -> Self {
        let golds_raw: HashMap<String, LexEntry> =
            serde_json::from_str(US_GOLD_JSON).expect("failed to parse us_gold.json");
        let silvers_raw: HashMap<String, LexEntry> =
            serde_json::from_str(US_SILVER_JSON).expect("failed to parse us_silver.json");

        Self {
            cap_stresses: (0.5, 2.0),
            golds: Self::grow_dictionary(golds_raw),
            silvers: Self::grow_dictionary(silvers_raw),
        }
    }

    /// Expand a dictionary by adding case variants:
    /// - lowercase keys get a Capitalized variant
    /// - Capitalized keys get a lowercase variant
    ///   Original entries override grown entries.
    fn grow_dictionary(d: HashMap<String, LexEntry>) -> HashMap<String, LexEntry> {
        let mut e: HashMap<String, LexEntry> = HashMap::new();
        for (k, v) in d.iter() {
            if k.chars().count() < 2 {
                continue;
            }
            let lower = k.to_lowercase();
            let capitalized = capitalize(&lower);
            if *k == lower {
                // Key is all lowercase
                if *k != capitalized {
                    e.insert(capitalized, v.clone());
                }
            } else if *k == capitalized {
                // Key is Capitalized
                e.insert(lower, v.clone());
            }
        }
        // Original overrides grown
        for (k, v) in d {
            e.insert(k, v);
        }
        e
    }

    // -----------------------------------------------------------------------
    // get_NNP: spell out as letter-by-letter phonemes
    // -----------------------------------------------------------------------

    pub fn get_nnp(&self, word: &str) -> (Option<String>, Option<u8>) {
        let mut ps_parts: Vec<String> = Vec::new();
        for c in word.chars() {
            if !c.is_alphabetic() {
                continue;
            }
            let upper = c.to_uppercase().to_string();
            match self.golds.get(&upper) {
                Some(entry) => {
                    let s = match entry {
                        LexEntry::Simple(s) => s.clone(),
                        LexEntry::Tagged(map) => match map.get("DEFAULT").and_then(|v| v.clone()) {
                            Some(s) => s,
                            None => return (None, None),
                        },
                    };
                    ps_parts.push(s);
                }
                None => return (None, None),
            }
        }
        if ps_parts.iter().any(|p| p.is_empty()) {
            // None in ps check: if any lookup returned empty/missing
        }
        let joined = ps_parts.join("");
        let stressed = apply_stress(&joined, Some(0.0));
        // rsplit on SECONDARY_STRESS, then rejoin with PRIMARY_STRESS
        let parts: Vec<&str> = stressed.rsplitn(2, SECONDARY_STRESS).collect();
        let result = if parts.len() == 2 {
            // rsplitn gives [after, before] — we want before + PRIMARY + after
            format!("{}{}{}", parts[1], PRIMARY_STRESS, parts[0])
        } else {
            stressed
        };
        (Some(result), Some(3))
    }

    // -----------------------------------------------------------------------
    // get_parent_tag
    // -----------------------------------------------------------------------

    pub fn get_parent_tag(tag: &str) -> &str {
        if tag.starts_with("VB") {
            "VERB"
        } else if tag.starts_with("NN") {
            "NOUN"
        } else if tag.starts_with("ADV") || tag.starts_with("RB") {
            "ADV"
        } else if tag.starts_with("ADJ") || tag.starts_with("JJ") {
            "ADJ"
        } else {
            tag
        }
    }

    // -----------------------------------------------------------------------
    // get_special_case
    // -----------------------------------------------------------------------

    pub fn get_special_case(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f32>,
        ctx: &TokenContext,
    ) -> (Option<String>, Option<u8>) {
        if tag == "ADD" && is_add_symbol(word) {
            let sym = add_symbol_word(word).unwrap();
            return self.lookup(sym, None, Some(-0.5), ctx);
        } else if is_symbol(word) {
            let sym = symbol_word(word).unwrap();
            return self.lookup(sym, None, None, ctx);
        } else if self.is_dotted_abbreviation(word) {
            return self.get_nnp(word);
        } else if word == "a" || word == "A" {
            let ps = if tag == "DT" {
                "ɐ".to_string()
            } else {
                format!("{}A", PRIMARY_STRESS)
            };
            return (Some(ps), Some(4));
        } else if word == "am" || word == "Am" || word == "AM" {
            if tag.starts_with("NN") {
                return self.get_nnp(word);
            } else if ctx.future_vowel.is_none()
                || word != "am"
                || (stress.is_some() && stress.unwrap() > 0.0)
            {
                let ps = self.resolve_simple_gold("am");
                return (Some(ps), Some(4));
            }
            return (Some("ɐm".to_string()), Some(4));
        } else if word == "an" || word == "An" || word == "AN" {
            if word == "AN" && tag.starts_with("NN") {
                return self.get_nnp(word);
            }
            return (Some("ɐn".to_string()), Some(4));
        } else if word == "I" && tag == "PRP" {
            return (Some(format!("{}I", SECONDARY_STRESS)), Some(4));
        } else if (word == "by" || word == "By" || word == "BY")
            && Self::get_parent_tag(tag) == "ADV"
        {
            return (Some(format!("b{}I", PRIMARY_STRESS)), Some(4));
        } else if word == "to" || word == "To" || (word == "TO" && (tag == "TO" || tag == "IN")) {
            let ps = match ctx.future_vowel {
                None => self.resolve_simple_gold("to"),
                Some(false) => "tə".to_string(),
                Some(true) => "tʊ".to_string(),
            };
            return (Some(ps), Some(4));
        } else if word == "in" || word == "In" || (word == "IN" && tag != "NNP") {
            let stress_mark = if ctx.future_vowel.is_none() || tag != "IN" {
                PRIMARY_STRESS.to_string()
            } else {
                String::new()
            };
            return (Some(format!("{}ɪn", stress_mark)), Some(4));
        } else if word == "the" || word == "The" || (word == "THE" && tag == "DT") {
            let ps = if ctx.future_vowel == Some(true) {
                "ði"
            } else {
                "ðə"
            };
            return (Some(ps.to_string()), Some(4));
        } else if tag == "IN" && is_vs(word) {
            return self.lookup("versus", None, None, ctx);
        } else if word == "used" || word == "Used" || word == "USED" {
            if (tag == "VBD" || tag == "JJ") && ctx.future_to {
                let ps = self.resolve_tagged_gold("used", "VBD");
                return (Some(ps), Some(4));
            }
            let ps = self.resolve_tagged_gold("used", "DEFAULT");
            return (Some(ps), Some(4));
        }
        (None, None)
    }

    // -----------------------------------------------------------------------
    // is_known
    // -----------------------------------------------------------------------

    pub fn is_known(&self, word: &str, _tag: &str) -> bool {
        if self.golds.contains_key(word) || is_symbol(word) || self.silvers.contains_key(word) {
            return true;
        } else if !is_alpha(word) || !all_lexicon_ords(word) {
            return false;
        } else if word.chars().count() == 1
            || (word == word.to_uppercase() && self.golds.contains_key(&word.to_lowercase()))
        {
            return true;
        }
        // word[1:] == word[1:].upper()
        let rest: String = word.chars().skip(1).collect();
        rest == rest.to_uppercase()
    }

    // -----------------------------------------------------------------------
    // lookup
    // -----------------------------------------------------------------------

    pub fn lookup(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f32>,
        ctx: &TokenContext,
    ) -> (Option<String>, Option<u8>) {
        let mut word = word.to_string();
        let tag_str: Option<String> = tag.map(|t| t.to_string());
        let mut is_nnp: Option<bool> = None;

        if word == word.to_uppercase() && !self.golds.contains_key(&word) {
            word = word.to_lowercase();
            is_nnp = Some(tag == Some("NNP"));
        }

        let (mut ps_entry, mut rating): (Option<LexEntry>, u8) = match self.golds.get(&word) {
            Some(entry) => (Some(entry.clone()), 4),
            None => (None, 4),
        };

        if ps_entry.is_none() && is_nnp != Some(true) {
            if let Some(entry) = self.silvers.get(&word) {
                ps_entry = Some(entry.clone());
                rating = 3;
            }
        }

        // Resolve tagged entries to a plain string
        let ps_str: Option<String> = match ps_entry {
            Some(LexEntry::Simple(s)) => Some(s),
            Some(LexEntry::Tagged(map)) => {
                let resolved_tag = if ctx.future_vowel.is_none() && map.contains_key("None") {
                    "None".to_string()
                } else {
                    let t = tag_str.as_deref().unwrap_or("DEFAULT");
                    if map.contains_key(t) {
                        t.to_string()
                    } else {
                        Self::get_parent_tag(t).to_string()
                    }
                };
                map.get(&resolved_tag)
                    .cloned()
                    .unwrap_or_else(|| map.get("DEFAULT").cloned().unwrap_or(None))
            }
            None => None,
        };

        if ps_str.is_none()
            || (is_nnp == Some(true)
                && ps_str
                    .as_ref()
                    .map(|s| !s.contains(PRIMARY_STRESS))
                    .unwrap_or(true))
        {
            let (nnp_ps, nnp_rating) = self.get_nnp(&word);
            if nnp_ps.is_some() {
                return (nnp_ps, nnp_rating);
            }
        }

        match ps_str {
            Some(s) => (Some(apply_stress(&s, stress)), Some(rating)),
            None => (None, None),
        }
    }

    // -----------------------------------------------------------------------
    // Suffix inflection helpers
    // -----------------------------------------------------------------------

    fn suffix_s(&self, stem: &str) -> Option<String> {
        if stem.is_empty() {
            return None;
        }
        let last = stem.chars().last().unwrap();
        if "ptkfθ".contains(last) {
            Some(format!("{}s", stem))
        } else if "szʃʒʧʤ".contains(last) {
            // US: ᵻz
            Some(format!("{}ᵻz", stem))
        } else {
            Some(format!("{}z", stem))
        }
    }

    pub fn stem_s(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f32>,
        ctx: &TokenContext,
    ) -> (Option<String>, Option<u8>) {
        if word.len() < 3 || !word.ends_with('s') {
            return (None, None);
        }
        let stem = if !word.ends_with("ss") && self.is_known(&word[..word.len() - 1], tag) {
            word[..word.len() - 1].to_string()
        } else if (word.ends_with("'s")
            || (word.len() > 4 && word.ends_with("es") && !word.ends_with("ies")))
            && self.is_known(&word[..word.len() - 2], tag)
        {
            word[..word.len() - 2].to_string()
        } else if word.len() > 4
            && word.ends_with("ies")
            && self.is_known(&format!("{}y", &word[..word.len() - 3]), tag)
        {
            format!("{}y", &word[..word.len() - 3])
        } else {
            return (None, None);
        };

        let (stem_ps, rating) = self.lookup(&stem, Some(tag), stress, ctx);
        match stem_ps {
            Some(ref s) => (self.suffix_s(s), rating),
            None => (None, None),
        }
    }

    fn suffix_ed(&self, stem: &str) -> Option<String> {
        if stem.is_empty() {
            return None;
        }
        let last = stem.chars().last().unwrap();
        if "pkfθʃsʧ".contains(last) {
            Some(format!("{}t", stem))
        } else if last == 'd' {
            // US: ᵻd
            Some(format!("{}ᵻd", stem))
        } else if last != 't' {
            Some(format!("{}d", stem))
        } else {
            let chars: Vec<char> = stem.chars().collect();
            if chars.len() < 2 {
                Some(format!("{}ɪd", stem))
            } else {
                let second_last = chars[chars.len() - 2];
                if US_TAUS.contains(second_last) {
                    // stem[:-1] + 'ɾᵻd'
                    let prefix: String = chars[..chars.len() - 1].iter().collect();
                    Some(format!("{}ɾᵻd", prefix))
                } else {
                    Some(format!("{}ᵻd", stem))
                }
            }
        }
    }

    pub fn stem_ed(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f32>,
        ctx: &TokenContext,
    ) -> (Option<String>, Option<u8>) {
        if word.len() < 4 || !word.ends_with('d') {
            return (None, None);
        }
        let stem = if !word.ends_with("dd") && self.is_known(&word[..word.len() - 1], tag) {
            word[..word.len() - 1].to_string()
        } else if word.len() > 4
            && word.ends_with("ed")
            && !word.ends_with("eed")
            && self.is_known(&word[..word.len() - 2], tag)
        {
            word[..word.len() - 2].to_string()
        } else {
            return (None, None);
        };

        let (stem_ps, rating) = self.lookup(&stem, Some(tag), stress, ctx);
        match stem_ps {
            Some(ref s) => (self.suffix_ed(s), rating),
            None => (None, None),
        }
    }

    fn suffix_ing(&self, stem: &str) -> Option<String> {
        if stem.is_empty() {
            return None;
        }
        let chars: Vec<char> = stem.chars().collect();
        if chars.len() > 1
            && chars[chars.len() - 1] == 't'
            && US_TAUS.contains(chars[chars.len() - 2])
        {
            let prefix: String = chars[..chars.len() - 1].iter().collect();
            Some(format!("{}ɾɪŋ", prefix))
        } else {
            Some(format!("{}ɪŋ", stem))
        }
    }

    pub fn stem_ing(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f32>,
        ctx: &TokenContext,
    ) -> (Option<String>, Option<u8>) {
        if word.len() < 5 || !word.ends_with("ing") {
            return (None, None);
        }
        let base = &word[..word.len() - 3];
        let stem = if word.len() > 5 && self.is_known(base, tag) {
            base.to_string()
        } else if self.is_known(&format!("{}e", base), tag) {
            format!("{}e", base)
        } else if word.len() > 5
            && self.check_doubled_consonant_ing(word)
            && self.is_known(&word[..word.len() - 4], tag)
        {
            word[..word.len() - 4].to_string()
        } else {
            return (None, None);
        };

        let (stem_ps, rating) = self.lookup(&stem, Some(tag), stress, ctx);
        match stem_ps {
            Some(ref s) => (self.suffix_ing(s), rating),
            None => (None, None),
        }
    }

    /// Check for doubled consonant + "ing" at end of word, or "cking" at end.
    /// Replaces regex `([bcdgklmnprstvxz])\1ing$|cking$`.
    fn check_doubled_consonant_ing(&self, word: &str) -> bool {
        if word.ends_with("cking") {
            return true;
        }
        // Check for doubled consonant + "ing"
        let bytes = word.as_bytes();
        if bytes.len() < 5 {
            return false;
        }
        let before_ing = &word[..word.len() - 3];
        let chars: Vec<char> = before_ing.chars().collect();
        if chars.len() < 2 {
            return false;
        }
        let last = chars[chars.len() - 1];
        let second_last = chars[chars.len() - 2];
        last == second_last && "bcdgklmnprstvxz".contains(last)
    }

    // -----------------------------------------------------------------------
    // get_word
    // -----------------------------------------------------------------------

    pub fn get_word(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f32>,
        ctx: &TokenContext,
    ) -> (Option<String>, Option<u8>) {
        let (ps, rating) = self.get_special_case(word, tag, stress, ctx);
        if ps.is_some() {
            return (ps, rating);
        }

        let wl = word.to_lowercase();
        let mut word = word.to_string();

        // Check if we should lowercase the word for lookup
        if word.chars().count() > 1
            && word.replace("'", "").chars().all(|c| c.is_alphabetic())
            && word != wl
            && (tag != "NNP" || word.chars().count() > 7)
            && !self.golds.contains_key(&word)
            && !self.silvers.contains_key(&word)
            && (word == word.to_uppercase() || is_lower_after_first(&word))
            && (self.golds.contains_key(&wl)
                || self.silvers.contains_key(&wl)
                || self.stem_s(&wl, tag, stress, ctx).0.is_some()
                || self.stem_ed(&wl, tag, stress, ctx).0.is_some()
                || self.stem_ing(&wl, tag, stress, ctx).0.is_some())
        {
            word = wl;
        }

        if self.is_known(&word, tag) {
            return self.lookup(&word, Some(tag), stress, ctx);
        } else if word.ends_with("s'") {
            let alt = format!("{}'s", &word[..word.len() - 2]);
            if self.is_known(&alt, tag) {
                return self.lookup(&alt, Some(tag), stress, ctx);
            }
        } else if word.ends_with("'") {
            let alt = &word[..word.len() - 1];
            if self.is_known(alt, tag) {
                return self.lookup(alt, Some(tag), stress, ctx);
            }
        }

        let (s_ps, s_rating) = self.stem_s(&word, tag, stress, ctx);
        if s_ps.is_some() {
            return (s_ps, s_rating);
        }
        let (ed_ps, ed_rating) = self.stem_ed(&word, tag, stress, ctx);
        if ed_ps.is_some() {
            return (ed_ps, ed_rating);
        }
        let ing_stress = if stress.is_none() { Some(0.5) } else { stress };
        let (ing_ps, ing_rating) = self.stem_ing(&word, tag, ing_stress, ctx);
        if ing_ps.is_some() {
            return (ing_ps, ing_rating);
        }

        (None, None)
    }

    // -----------------------------------------------------------------------
    // is_number / is_currency / get_number / append_currency / numeric_if_needed
    // -----------------------------------------------------------------------

    /// Check if `word` is a number (possibly with suffix like "st", "nd", "s", etc.).
    pub fn is_number(word: &str, is_head: bool) -> bool {
        // Must contain at least one digit
        if !word.chars().any(|c| c.is_ascii_digit()) {
            return false;
        }
        let suffixes: &[&str] = &["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"];
        let mut w = word;
        let owned: String;
        for s in suffixes {
            if w.ends_with(s) {
                owned = w[..w.len() - s.len()].to_string();
                w = &owned;
                break;
            }
        }
        w.chars().enumerate().all(|(i, c)| {
            c.is_ascii_digit() || c == ',' || c == '.' || (is_head && i == 0 && c == '-')
        })
    }

    /// Check if a numeric word is a valid currency representation.
    pub fn is_currency(word: &str) -> bool {
        if !word.contains('.') {
            return true;
        }
        if word.matches('.').count() > 1 {
            return false;
        }
        let parts: Vec<&str> = word.split('.').collect();
        if parts.len() < 2 {
            return true;
        }
        let cents = parts[1];
        cents.len() < 3 || cents.chars().all(|c| c == '0')
    }

    /// Convert a number word to phonemes, handling ordinals, years, currency, etc.
    pub fn get_number(
        &self,
        word: &str,
        currency: Option<char>,
        is_head: bool,
        num_flags: &str,
    ) -> (Option<String>, Option<u8>) {
        // Extract suffix
        let suffixes: &[&str] = &["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"];
        let mut suffix: Option<&str> = None;
        let mut word_str: String = word.to_string();

        // Search for suffix match using the actual word suffix
        for s in suffixes {
            if word_str.ends_with(s) {
                // Check that stripping the suffix leaves at least something
                let prefix = &word_str[..word_str.len() - s.len()];
                if !prefix.is_empty() {
                    // For ordinals (st, nd, rd, th), only match if prefix is all digits
                    if ORDINALS.contains(s) {
                        if is_digit_str(prefix) {
                            suffix = Some(s);
                            word_str = prefix.to_string();
                        }
                    } else {
                        suffix = Some(s);
                        word_str = prefix.to_string();
                    }
                    break;
                }
            }
        }

        let mut result: Vec<(String, u8)> = Vec::new();

        if word_str.starts_with('-') {
            let (ps, rating) = self.lookup("minus", None, None, &TokenContext::default());
            if let Some(ps) = ps {
                result.push((ps, rating.unwrap_or(4)));
            }
            word_str = word_str[1..].to_string();
        }

        let has_currency = currency.is_some() && currency.and_then(currency_names).is_some();

        // Helper closure: extend result with number-to-word lookup
        let extend_num = |result: &mut Vec<(String, u8)>,
                          num: &str,
                          first: bool,
                          escape: bool,
                          num_flags: &str,
                          lexicon: &Lexicon| {
            let words_str = if escape {
                num.to_string()
            } else {
                let n: i64 = num.parse().unwrap_or(0);
                int_to_words(n)
            };
            // Split on non-alpha chars
            let splits: Vec<&str> = if escape {
                words_str
                    .split(|c: char| !c.is_ascii_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect()
            } else {
                words_str.split_whitespace().collect()
            };
            for (i, w) in splits.iter().enumerate() {
                if *w != "and" || num_flags.contains('&') {
                    if first && i == 0 && splits.len() > 1 && *w == "one" && num_flags.contains('a')
                    {
                        result.push(("ə".to_string(), 4));
                    } else {
                        let stress = if *w == "point" { Some(-2.0) } else { None };
                        let (ps, rating) =
                            lexicon.lookup(w, None, stress, &TokenContext::default());
                        if let Some(ps) = ps {
                            result.push((ps, rating.unwrap_or(4)));
                        }
                    }
                } else if *w == "and" && num_flags.contains('n') && !result.is_empty() {
                    let last = result.last_mut().unwrap();
                    last.0.push_str("ən");
                }
            }
        };

        let is_ordinal_suffix = suffix.map(|s| ORDINALS.contains(&s)).unwrap_or(false);

        if is_digit_str(&word_str) && is_ordinal_suffix {
            // Ordinal number
            let n: i64 = word_str.parse().unwrap_or(0);
            let ordinal_words = int_to_ordinal(n);
            extend_num(&mut result, &ordinal_words, true, true, num_flags, self);
        } else if result.is_empty()
            && word_str.len() == 4
            && !has_currency
            && is_digit_str(&word_str)
        {
            // Year
            let n: i64 = word_str.parse().unwrap_or(0);
            let year_words = int_to_year(n);
            extend_num(&mut result, &year_words, true, true, num_flags, self);
        } else if !is_head && !word_str.contains('.') {
            // Non-head, no decimal
            let num = word_str.replace(',', "");
            let num_bytes = num.as_bytes();
            if (!num.is_empty() && num_bytes[0] == b'0') || num.len() > 3 {
                // Spell out digit by digit
                for n in num.chars() {
                    let digit = n.to_string();
                    extend_num(&mut result, &digit, false, false, num_flags, self);
                }
            } else if num.len() == 3 && !num.ends_with("00") {
                // Three-digit: first digit, then rest
                extend_num(&mut result, &num[..1], true, false, num_flags, self);
                if num.as_bytes()[1] == b'0' {
                    let (ps, rating) = self.lookup("O", None, Some(-2.0), &TokenContext::default());
                    if let Some(ps) = ps {
                        result.push((ps, rating.unwrap_or(4)));
                    }
                    extend_num(&mut result, &num[2..3], false, false, num_flags, self);
                } else {
                    extend_num(&mut result, &num[1..], false, false, num_flags, self);
                }
            } else {
                extend_num(&mut result, &num, true, false, num_flags, self);
            }
        } else if word_str.matches('.').count() > 1 || !is_head {
            // Multiple decimals or non-head
            let cleaned = word_str.replace(',', "");
            let mut first = true;
            for num in cleaned.split('.') {
                if num.is_empty() {
                    // skip
                } else if (!num.is_empty() && num.as_bytes()[0] == b'0')
                    || (num.len() != 2 && num[1..].chars().any(|c| c != '0'))
                {
                    // Spell digit by digit
                    for n in num.chars() {
                        let digit = n.to_string();
                        extend_num(&mut result, &digit, false, false, num_flags, self);
                    }
                } else {
                    extend_num(&mut result, num, first, false, num_flags, self);
                }
                first = false;
            }
        } else if has_currency && Self::is_currency(&word_str) {
            // Currency handling
            let currency_ch = currency.unwrap();
            let (unit_name, subunit_name) = currency_names(currency_ch).unwrap();
            let cleaned = word_str.replace(',', "");
            let parts: Vec<&str> = cleaned.split('.').collect();

            let mut pairs: Vec<(i64, &str)> = Vec::new();
            for (idx, part) in parts.iter().enumerate() {
                let n: i64 = if part.is_empty() {
                    0
                } else {
                    part.parse().unwrap_or(0)
                };
                let unit = if idx == 0 { unit_name } else { subunit_name };
                pairs.push((n, unit));
            }

            if pairs.len() > 1 {
                if pairs[1].0 == 0 {
                    pairs.truncate(1);
                } else if pairs[0].0 == 0 {
                    pairs.remove(0);
                }
            }

            for (i, (num, unit)) in pairs.iter().enumerate() {
                if i > 0 {
                    let (ps, rating) = self.lookup("and", None, None, &TokenContext::default());
                    if let Some(ps) = ps {
                        result.push((ps, rating.unwrap_or(4)));
                    }
                }
                let num_str = num.to_string();
                extend_num(&mut result, &num_str, i == 0, false, num_flags, self);

                // Pluralize: unit + "s" if abs(num) != 1 and unit != "pence"
                if num.abs() != 1 && *unit != "pence" {
                    let pluralized = format!("{}s", unit);
                    let (ps, rating) =
                        self.stem_s(&pluralized, "NN", None, &TokenContext::default());
                    if let Some(ps) = ps {
                        result.push((ps, rating.unwrap_or(4)));
                    }
                } else {
                    let (ps, rating) = self.lookup(unit, None, None, &TokenContext::default());
                    if let Some(ps) = ps {
                        result.push((ps, rating.unwrap_or(4)));
                    }
                }
            }
        } else {
            // General case
            let converted = if is_digit_str(&word_str) {
                let n: i64 = word_str.parse().unwrap_or(0);
                int_to_words(n)
            } else if !word_str.contains('.') {
                let cleaned = word_str.replace(',', "");
                let n: i64 = cleaned.parse().unwrap_or(0);
                if is_ordinal_suffix {
                    int_to_ordinal(n)
                } else {
                    int_to_words(n)
                }
            } else {
                let cleaned = word_str.replace(',', "");
                if let Some(after_dot) = cleaned.strip_prefix('.') {
                    // ".123" -> "point one two three"
                    let digits: Vec<String> = after_dot
                        .chars()
                        .map(|c| {
                            let n: i64 = c.to_digit(10).unwrap_or(0) as i64;
                            int_to_words(n)
                        })
                        .collect();
                    format!("point {}", digits.join(" "))
                } else {
                    // Parse as float -> num2words equivalent
                    // Use our own decimal handling
                    let f: f64 = cleaned.parse().unwrap_or(0.0);
                    float_to_words(f)
                }
            };
            extend_num(&mut result, &converted, true, true, num_flags, self);
        }

        if result.is_empty() {
            return (None, None);
        }

        let ps: String = result
            .iter()
            .map(|(p, _)| p.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let rating: u8 = result.iter().map(|(_, r)| *r).min().unwrap_or(4);

        // Apply suffix
        match suffix {
            Some("s") | Some("'s") => (self.suffix_s(&ps), Some(rating)),
            Some("ed") | Some("'d") => (self.suffix_ed(&ps), Some(rating)),
            Some("ing") => (self.suffix_ing(&ps), Some(rating)),
            _ => (Some(ps), Some(rating)),
        }
    }

    /// Append a currency unit word to phonemes.
    pub fn append_currency(&self, ps: &str, currency: Option<char>) -> String {
        let currency = match currency {
            Some(c) => c,
            None => return ps.to_string(),
        };
        let (unit_name, _) = match currency_names(currency) {
            Some(names) => names,
            None => return ps.to_string(),
        };
        let pluralized = format!("{}s", unit_name);
        let (currency_ps, _) = self.stem_s(&pluralized, "NN", None, &TokenContext::default());
        match currency_ps {
            Some(cps) => format!("{} {}", ps, cps),
            None => ps.to_string(),
        }
    }

    /// Convert full-width or Unicode digits to ASCII equivalents.
    pub fn numeric_if_needed(c: char) -> char {
        if !c.is_numeric() || c.is_ascii_digit() {
            return c;
        }
        // Try to get the numeric value
        if let Some(n) = c.to_digit(10) {
            // It's a digit with integer value
            char::from_digit(n, 10).unwrap_or(c)
        } else {
            c
        }
    }

    // -----------------------------------------------------------------------
    // __call__ entry point
    // -----------------------------------------------------------------------

    /// The main entry point: look up a token's phonemes given context.
    /// Returns `(Option<phonemes>, Option<rating>)`.
    #[allow(clippy::too_many_arguments)]
    pub fn call(
        &self,
        text: &str,
        alias: Option<&str>,
        tag: &str,
        stress_override: Option<f32>,
        currency: Option<char>,
        is_head: bool,
        num_flags: &str,
        ctx: &TokenContext,
    ) -> (Option<String>, Option<u8>) {
        // Replace curly quotes with straight apostrophe
        let word = alias.unwrap_or(text);
        let word = word.replace(['\u{2018}', '\u{2019}'], "'");
        // NFKC normalize
        use unicode_normalization::UnicodeNormalization;
        let word: String = word.nfkc().collect();
        // Convert Unicode digits
        let word: String = word.chars().map(Self::numeric_if_needed).collect();

        let stress = if word == word.to_lowercase() {
            None
        } else if word == word.to_uppercase() {
            Some(self.cap_stresses.1)
        } else {
            Some(self.cap_stresses.0)
        };

        let (ps, rating) = self.get_word(&word, tag, stress, ctx);
        if let Some(ps) = ps {
            let with_currency = self.append_currency(&ps, currency);
            let final_ps = apply_stress(&with_currency, stress_override);
            return (Some(final_ps), rating);
        }

        if Self::is_number(&word, is_head) {
            let (ps, rating) = self.get_number(&word, currency, is_head, num_flags);
            if let Some(ref ps) = ps {
                let final_ps = apply_stress(ps, stress_override);
                return (Some(final_ps), rating);
            }
            return (ps, rating);
        }

        if !word.chars().all(is_lexicon_ord) {
            return (None, None);
        }

        (None, None)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Check if word matches pattern: contains '.' in the middle, all alpha when dots removed,
    /// and longest segment between dots is < 3 chars.
    fn is_dotted_abbreviation(&self, word: &str) -> bool {
        let stripped = word.trim_matches('.');
        if !stripped.contains('.') {
            return false;
        }
        let without_dots = word.replace('.', "");
        if !is_alpha(&without_dots) {
            return false;
        }
        let max_segment_len = word.split('.').map(|s| s.len()).max().unwrap_or(0);
        max_segment_len < 3
    }

    /// Resolve a simple (non-tagged) gold dictionary entry to its string value.
    fn resolve_simple_gold(&self, key: &str) -> String {
        match self.golds.get(key) {
            Some(LexEntry::Simple(s)) => s.clone(),
            Some(LexEntry::Tagged(map)) => map
                .get("DEFAULT")
                .and_then(|v| v.clone())
                .unwrap_or_default(),
            None => String::new(),
        }
    }

    /// Resolve a tagged gold dictionary entry for a specific POS tag.
    fn resolve_tagged_gold(&self, key: &str, tag: &str) -> String {
        match self.golds.get(key) {
            Some(LexEntry::Tagged(map)) => {
                map.get(tag).and_then(|v| v.clone()).unwrap_or_else(|| {
                    map.get("DEFAULT")
                        .and_then(|v| v.clone())
                        .unwrap_or_default()
                })
            }
            Some(LexEntry::Simple(s)) => s.clone(),
            None => String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Capitalize the first character of a string.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => {
            let upper: String = c.to_uppercase().collect();
            format!("{}{}", upper, chars.as_str())
        }
    }
}

/// Check if `word[1:]` is all lowercase.
fn is_lower_after_first(word: &str) -> bool {
    word.chars()
        .skip(1)
        .all(|c| c.is_lowercase() || !c.is_alphabetic())
}

/// Check if a word matches the `(?i)vs\.?$` pattern.
fn is_vs(word: &str) -> bool {
    let w = word.to_lowercase();
    w == "vs" || w == "vs."
}

/// Simple float-to-words for numbers like 3.14 -> "three point one four".
fn float_to_words(f: f64) -> String {
    let s = format!("{}", f);
    if let Some(dot_pos) = s.find('.') {
        let integer_part = &s[..dot_pos];
        let decimal_part = &s[dot_pos + 1..];
        let int_n: i64 = integer_part.parse().unwrap_or(0);
        let int_words = int_to_words(int_n);
        let decimal_words: Vec<String> = decimal_part
            .chars()
            .map(|c| {
                let n = c.to_digit(10).unwrap_or(0) as i64;
                int_to_words(n)
            })
            .collect();
        format!("{} point {}", int_words, decimal_words.join(" "))
    } else {
        int_to_words(f as i64)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grow_dictionary() {
        let mut d = HashMap::new();
        d.insert("hello".to_string(), LexEntry::Simple("hɛlˈO".to_string()));
        d.insert("World".to_string(), LexEntry::Simple("wˈɜɹld".to_string()));
        let grown = Lexicon::grow_dictionary(d);
        // "hello" -> grew "Hello"
        assert!(grown.contains_key("Hello"));
        // "World" -> grew "world"
        assert!(grown.contains_key("world"));
        // originals preserved
        assert!(grown.contains_key("hello"));
        assert!(grown.contains_key("World"));
    }

    #[test]
    fn test_lexicon_new() {
        let lex = Lexicon::new();
        // Should have loaded dictionaries
        assert!(!lex.golds.is_empty());
        assert!(!lex.silvers.is_empty());
        // "hello" should be in gold
        assert!(lex.golds.contains_key("hello"));
    }

    #[test]
    fn test_get_parent_tag() {
        assert_eq!(Lexicon::get_parent_tag("VBD"), "VERB");
        assert_eq!(Lexicon::get_parent_tag("VBZ"), "VERB");
        assert_eq!(Lexicon::get_parent_tag("NNS"), "NOUN");
        assert_eq!(Lexicon::get_parent_tag("NNP"), "NOUN");
        assert_eq!(Lexicon::get_parent_tag("RB"), "ADV");
        assert_eq!(Lexicon::get_parent_tag("JJ"), "ADJ");
        assert_eq!(Lexicon::get_parent_tag("IN"), "IN");
    }

    #[test]
    fn test_is_number() {
        assert!(Lexicon::is_number("42", true));
        assert!(Lexicon::is_number("1,000", true));
        assert!(Lexicon::is_number("3.14", true));
        assert!(Lexicon::is_number("-5", true));
        assert!(Lexicon::is_number("1st", true));
        assert!(Lexicon::is_number("42nd", true));
        assert!(!Lexicon::is_number("hello", true));
        assert!(!Lexicon::is_number("-5", false)); // negative only allowed for is_head
    }

    #[test]
    fn test_is_currency() {
        assert!(Lexicon::is_currency("42"));
        assert!(Lexicon::is_currency("3.14"));
        assert!(!Lexicon::is_currency("1.2.3"));
        assert!(Lexicon::is_currency("100.99"));
        assert!(!Lexicon::is_currency("100.999")); // 3+ cents digits, not all zero
        assert!(Lexicon::is_currency("100.000")); // 3 digits but all zero
    }

    #[test]
    fn test_numeric_if_needed() {
        assert_eq!(Lexicon::numeric_if_needed('5'), '5');
        assert_eq!(Lexicon::numeric_if_needed('a'), 'a');
    }

    #[test]
    fn test_capitalize() {
        assert_eq!(capitalize("hello"), "Hello");
        assert_eq!(capitalize("HELLO"), "HELLO");
        assert_eq!(capitalize(""), "");
    }

    #[test]
    fn test_is_vs() {
        assert!(is_vs("vs"));
        assert!(is_vs("Vs"));
        assert!(is_vs("VS"));
        assert!(is_vs("vs."));
        assert!(is_vs("Vs."));
        assert!(!is_vs("versus"));
    }

    #[test]
    fn test_is_dotted_abbreviation() {
        let lex = Lexicon::new();
        assert!(lex.is_dotted_abbreviation("U.S.A."));
        assert!(lex.is_dotted_abbreviation("U.S."));
        assert!(!lex.is_dotted_abbreviation("hello"));
        assert!(!lex.is_dotted_abbreviation(".hello"));
        assert!(!lex.is_dotted_abbreviation("test.long.segment"));
    }

    #[test]
    fn test_lookup_simple() {
        let lex = Lexicon::new();
        let ctx = TokenContext::default();
        let (ps, rating) = lex.lookup("hello", Some("NN"), None, &ctx);
        assert!(ps.is_some());
        assert_eq!(rating, Some(4));
    }

    #[test]
    fn test_special_case_a() {
        let lex = Lexicon::new();
        let ctx = TokenContext::default();
        let (ps, rating) = lex.get_special_case("a", "DT", None, &ctx);
        assert_eq!(ps, Some("ɐ".to_string()));
        assert_eq!(rating, Some(4));
    }

    #[test]
    fn test_special_case_the() {
        let lex = Lexicon::new();

        let ctx_vowel = TokenContext {
            future_vowel: Some(true),
            future_to: false,
        };
        let (ps, _) = lex.get_special_case("the", "DT", None, &ctx_vowel);
        assert_eq!(ps, Some("ði".to_string()));

        let ctx_cons = TokenContext {
            future_vowel: Some(false),
            future_to: false,
        };
        let (ps, _) = lex.get_special_case("the", "DT", None, &ctx_cons);
        assert_eq!(ps, Some("ðə".to_string()));
    }

    #[test]
    fn test_float_to_words() {
        let result = float_to_words(std::f64::consts::PI);
        assert!(result.starts_with("three point"));
    }

    #[test]
    fn test_suffix_s() {
        let lex = Lexicon::new();
        // Voiceless: p -> +s
        assert_eq!(lex.suffix_s("stɑp"), Some("stɑps".to_string()));
        // Sibilant: s -> +ᵻz
        assert_eq!(lex.suffix_s("bʌs"), Some("bʌsᵻz".to_string()));
        // Voiced: d -> +z
        assert_eq!(lex.suffix_s("plAd"), Some("plAdz".to_string()));
    }

    #[test]
    fn test_suffix_ed() {
        let lex = Lexicon::new();
        // Voiceless: k -> +t
        assert_eq!(lex.suffix_ed("wɑk"), Some("wɑkt".to_string()));
        // d -> +ᵻd
        assert_eq!(lex.suffix_ed("plAd"), Some("plAdᵻd".to_string()));
        // Voiced (not t or d): n -> +d
        assert_eq!(lex.suffix_ed("plæn"), Some("plænd".to_string()));
    }

    #[test]
    fn test_suffix_ing() {
        let lex = Lexicon::new();
        // Normal: +ɪŋ
        assert_eq!(lex.suffix_ing("plA"), Some("plAɪŋ".to_string()));
    }

    #[test]
    fn test_check_doubled_consonant_ing() {
        let lex = Lexicon::new();
        assert!(lex.check_doubled_consonant_ing("running"));
        assert!(lex.check_doubled_consonant_ing("sitting"));
        assert!(lex.check_doubled_consonant_ing("kicking"));
        assert!(!lex.check_doubled_consonant_ing("eating"));
    }
}
