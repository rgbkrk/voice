//! Tokenizer for English G2P.
//!
//! Splits text into [`MToken`]s using whitespace-based word boundaries, then
//! assigns POS tags via an embedded averaged perceptron tagger (no external
//! processes required). espeak-ng subprocess is still used downstream as an
//! OOV pronunciation fallback, but tokenization itself is fully self-contained.

use std::sync::OnceLock;

use fancy_regex::Regex;

use crate::stress::{punct_tag_phoneme, PUNCTS, PUNCT_TAGS, SUBTOKEN_JUNKS};
use crate::tagger;
use crate::token::MToken;

/// Characters that should be split off the **end** of a word token.
const TRAILING_PUNCT: &str = ".!?…,;:)—\"\u{201D}\u{201E}\u{2019}";

/// Characters that should be split off the **start** of a word token.
const LEADING_PUNCT: &str = "($£€¥\"\u{201C}\u{2018}";

/// Returns true if every character in `s` is a punctuation character from PUNCTS.
fn is_all_puncts(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| PUNCTS.contains(c))
}

/// Returns true if `s` looks like a number (digits with optional commas/dots).
///
/// A trailing dot with no digit after it is NOT number-like — that's a
/// sentence-ending period (e.g. `"3."` → `["3", "."]`), matching spaCy's
/// behaviour. Internal dots with digits on both sides are fine (`"3.14"`).
fn is_number_like(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if !s
        .chars()
        .all(|c| c.is_ascii_digit() || c == ',' || c == '.')
    {
        return false;
    }
    // Must contain at least one digit
    if !s.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }
    // Trailing dot/comma without a following digit → not a number
    // (e.g. "3." is sentence-ending, "3," is a list)
    !s.ends_with('.') && !s.ends_with(',')
}

/// Returns true if every character in `s` is a currency symbol.
fn is_currency(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| matches!(c, '$' | '£' | '€'))
}

/// Assign a simple POS tag based on the token text.
fn simple_tag(text: &str) -> &'static str {
    if is_currency(text) {
        "$"
    } else if text == "(" {
        "-LRB-"
    } else if text == ")" {
        "-RRB-"
    } else if is_all_puncts(text) {
        // Classify punctuation
        let first = text.chars().next().unwrap();
        if matches!(first, '.' | '!' | '?' | '…') {
            "."
        } else if first == ',' {
            ","
        } else if matches!(first, ':' | ';' | '—' | '-') {
            ":"
        } else {
            "."
        }
    } else if is_number_like(text) {
        "CD"
    } else {
        "DEFAULT"
    }
}

/// Assign a POS tag for a leading (opening) punctuation character.
///
/// Matches spaCy's tagging: `(` → `-LRB-`, `"` → ` `` `, etc.
fn leading_punct_tag(text: &str) -> &'static str {
    match text {
        "(" => "-LRB-",
        "\"" | "\u{201C}" | "\u{2018}" => "``",
        _ => simple_tag(text),
    }
}

/// Assign a POS tag for a trailing (closing) punctuation character.
///
/// Matches spaCy's tagging: `)` → `-RRB-`, `"` → `''`, etc.
fn trailing_punct_tag(text: &str) -> &'static str {
    match text {
        ")" => "-RRB-",
        "\"" | "\u{201D}" | "\u{201E}" | "\u{2019}" => "''",
        _ => simple_tag(text),
    }
}

/// Split leading and trailing punctuation from a word token into separate tokens.
///
/// For example, `"Hello,"` (with whitespace `" "`) becomes:
///   - `MToken { text: "Hello", whitespace: "", tag: "DEFAULT" }`
///   - `MToken { text: ",",     whitespace: " ", tag: "," }`
///
/// Pure-punctuation tokens like `"!?!?"` or `"..."` are split into individual
/// characters so each one maps to its own model pause token.
///
/// Number-like and currency tokens are returned as-is.
fn split_punct(word: &str, whitespace: &str) -> Vec<MToken> {
    if word.is_empty() || is_number_like(word) || is_currency(word) {
        let tag = simple_tag(word);
        let mut tok = MToken::new(word, tag, whitespace);
        tok.underscore.is_head = true;
        return vec![tok];
    }

    // Pure-punctuation tokens: split into individual characters so each maps
    // to its own model vocab entry (e.g. "!?!?" → ["!", "?", "!", "?"],
    // "..." → [".", ".", "."]). Single-char punct passes through as-is.
    if is_all_puncts(word) {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() == 1 {
            let tag = simple_tag(word);
            let mut tok = MToken::new(word, tag, whitespace);
            tok.underscore.is_head = true;
            return vec![tok];
        }
        let mut result = Vec::new();
        for (i, ch) in chars.iter().enumerate() {
            let s: String = std::iter::once(*ch).collect();
            let is_last = i == chars.len() - 1;
            let tok_ws = if is_last { whitespace } else { "" };
            let tag = simple_tag(&s);
            let mut tok = MToken::new(&s, tag, tok_ws);
            tok.underscore.is_head = true;
            result.push(tok);
        }
        return result;
    }

    let chars: Vec<char> = word.chars().collect();
    let len = chars.len();

    // Find how many leading chars are in LEADING_PUNCT
    let leading = chars
        .iter()
        .take_while(|c| LEADING_PUNCT.contains(**c))
        .count();

    // Find how many trailing chars are in TRAILING_PUNCT
    let trailing = chars
        .iter()
        .rev()
        .take_while(|c| TRAILING_PUNCT.contains(**c))
        .count();

    // If leading + trailing consume the entire token (e.g. "()", "\"\""),
    // emit each character individually with positional tags rather than
    // returning the whole thing as one opaque token.
    if leading + trailing >= len {
        let mut result = Vec::new();
        for (i, ch) in chars.iter().enumerate() {
            let s: String = std::iter::once(*ch).collect();
            let is_last = i == chars.len() - 1;
            let tok_ws = if is_last { whitespace } else { "" };
            let tag = if LEADING_PUNCT.contains(*ch) {
                leading_punct_tag(&s)
            } else {
                trailing_punct_tag(&s)
            };
            let mut tok = MToken::new(&s, tag, tok_ws);
            tok.underscore.is_head = true;
            result.push(tok);
        }
        return result;
    }

    let mut result = Vec::new();

    // Emit leading punct tokens (one per character, like spaCy)
    for i in 0..leading {
        let ch: String = chars[i..=i].iter().collect();
        let tag = leading_punct_tag(&ch);
        let mut tok = MToken::new(&ch, tag, "");
        tok.underscore.is_head = true;
        result.push(tok);
    }

    // Emit the core word (no whitespace — trailing punct gets it)
    let core: String = chars[leading..len - trailing].iter().collect();
    let core_ws = if trailing > 0 { "" } else { whitespace };
    let tag = simple_tag(&core);
    let mut tok = MToken::new(&core, tag, core_ws);
    tok.underscore.is_head = true;
    result.push(tok);

    // Emit trailing punct tokens (one per character, like spaCy)
    for i in 0..trailing {
        let ch: String = chars[len - trailing + i..=len - trailing + i]
            .iter()
            .collect();
        let is_last = i == trailing - 1;
        let tok_ws = if is_last { whitespace } else { "" };
        let tag = trailing_punct_tag(&ch);
        let mut tok = MToken::new(&ch, tag, tok_ws);
        tok.underscore.is_head = true;
        result.push(tok);
    }

    result
}

/// Tokenize text using simple whitespace splitting with heuristic POS tags.
///
/// After splitting on whitespace, leading and trailing punctuation is detached
/// from word tokens so that `.` `,` `!` `?` etc. become their own tokens with
/// proper POS tags. This matches spaCy's behaviour and ensures punctuation
/// flows through the G2P pipeline as pause tokens for the Kokoro model.
pub fn tokenize_simple(text: &str) -> Vec<MToken> {
    let mut raw_tokens: Vec<(&str, &str)> = Vec::new(); // (word, whitespace)
    let mut chars = text.char_indices().peekable();
    let mut current_word_start: Option<usize> = None;

    while let Some(&(i, c)) = chars.peek() {
        if c.is_whitespace() {
            // If we had a word being accumulated, finalize it
            if let Some(start) = current_word_start.take() {
                let word = &text[start..i];
                // Gather the whitespace
                let ws_start = i;
                while let Some(&(_, wc)) = chars.peek() {
                    if wc.is_whitespace() {
                        chars.next();
                    } else {
                        break;
                    }
                }
                let ws_end = chars.peek().map(|&(idx, _)| idx).unwrap_or(text.len());
                let ws = &text[ws_start..ws_end];
                raw_tokens.push((word, ws));
            } else {
                // Leading whitespace — skip
                chars.next();
            }
        } else {
            if current_word_start.is_none() {
                current_word_start = Some(i);
            }
            chars.next();
        }
    }

    // Handle the last word (no trailing whitespace)
    if let Some(start) = current_word_start {
        let word = &text[start..];
        raw_tokens.push((word, ""));
    }

    // Split leading/trailing punctuation from each token
    let mut tokens = Vec::new();
    for (word, ws) in raw_tokens {
        tokens.extend(split_punct(word, ws));
    }

    tokens
}

// ---------------------------------------------------------------------------
// Combined entry point
// ---------------------------------------------------------------------------

/// Tokenize text using the simple word splitter, then apply POS tags from
/// the embedded perceptron tagger.
///
/// Heuristic tags for punctuation, numbers, and currency are kept as-is
/// (the perceptron wasn't trained on those token shapes). All other tokens
/// get their tag overwritten by the perceptron's prediction.
pub fn tokenize(text: &str) -> Vec<MToken> {
    let mut tokens = tokenize_simple(text);

    // Collect words for the perceptron tagger. Owned strings break the
    // borrow on `tokens` so we can mutate them afterwards.
    let words_owned: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();
    let words: Vec<&str> = words_owned.iter().map(|s| s.as_str()).collect();
    let tags = tagger::global_tagger().tag(&words);

    for (tok, tag) in tokens.iter_mut().zip(tags.iter()) {
        // Keep heuristic tags for punctuation/numbers/currency — the perceptron
        // doesn't handle those well. Override everything tagged "DEFAULT" by
        // the simple tagger.
        if tok.tag == "DEFAULT" {
            tok.tag = tag.tag.clone();
        }
    }

    tokens
}

// ---------------------------------------------------------------------------
// Subtokenize
// ---------------------------------------------------------------------------

/// Lazily compiled regex for subtokenization.
/// Ported from misaki en.py:57-58.
fn subtokenize_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?x)
            ^[''']+ |
            \p{Lu}(?=\p{Lu}\p{Ll}) |
            (?:^-)?(?:\d?[,.]?\d)+ |
            [-_]+ |
            [''']{2,} |
            \p{L}*?(?:[''']\p{L})*?\p{Ll}(?=\p{Lu}) |
            \p{L}+(?:[''']\p{L})* |
            [^-_\p{L}'''\d] |
            [''']+$
            ",
        )
        .expect("subtokenize regex should compile")
    })
}

/// Split a word into subtokens using the English subtokenization pattern.
pub fn subtokenize(word: &str) -> Vec<String> {
    let re = subtokenize_regex();
    let mut results = Vec::new();
    let mut pos = 0;
    while pos < word.len() {
        if let Ok(Some(m)) = re.find(&word[pos..]) {
            let start = m.start();
            let end = m.end();
            if start == end {
                // Zero-length match — advance by one character to avoid infinite loop
                pos += word[pos..].chars().next().map_or(1, |c| c.len_utf8());
                continue;
            }
            results.push(word[pos + start..pos + end].to_string());
            pos += end;
        } else {
            break;
        }
    }

    if results.is_empty() {
        vec![word.to_string()]
    } else {
        results
    }
}

// ---------------------------------------------------------------------------
// fold_left
// ---------------------------------------------------------------------------

/// Merge adjacent tokens where the second token is not a head.
///
/// Ported from misaki en.py:594-599.
pub fn fold_left(tokens: Vec<MToken>) -> Vec<MToken> {
    let mut result: Vec<MToken> = Vec::new();

    for tok in tokens {
        if !tok.underscore.is_head && !result.is_empty() {
            // Merge into the previous token
            let prev = result.last_mut().unwrap();
            prev.text.push_str(&prev.whitespace);
            prev.text.push_str(&tok.text);
            prev.whitespace = tok.whitespace;
            // Merge phonemes: if both have phonemes, concatenate
            if let Some(ref tp) = tok.phonemes {
                if let Some(ref mut pp) = prev.phonemes {
                    pp.push_str(tp);
                } else {
                    prev.phonemes = tok.phonemes.clone();
                }
            }
        } else {
            result.push(tok);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Retokenize
// ---------------------------------------------------------------------------

/// A retokenized element: either a single token or a group of adjacent
/// subtokens that had no whitespace between them.
#[derive(Clone, Debug)]
pub enum TokenOrGroup {
    Single(MToken),
    Group(Vec<MToken>),
}

/// Retokenize: process tokens after subtokenization, handling punctuation,
/// currency, and grouping adjacent subtokens.
///
/// Ported from misaki en.py:601-643.
pub fn retokenize(tokens: Vec<MToken>) -> Vec<TokenOrGroup> {
    let mut output: Vec<TokenOrGroup> = Vec::new();
    let mut pending_currency: Option<char> = None;

    for tok in tokens {
        // If alias or phonemes are already set, pass through
        if tok.underscore.alias.is_some() || tok.phonemes.is_some() {
            output.push(TokenOrGroup::Single(tok));
            continue;
        }

        let subtokens = subtokenize(&tok.text);

        if subtokens.len() <= 1 {
            // Single token — apply classification
            let mut tok = tok;

            if tok.tag == "$" && is_currency(&tok.text) {
                // Currency symbol: set empty phonemes, rating 4
                tok.phonemes = Some(String::new());
                tok.underscore.rating = Some(4);
                pending_currency = tok.text.chars().next();
                output.push(TokenOrGroup::Single(tok));
            } else if tok.tag == ":" && tok.text.chars().all(|c| c == '-' || c == '—') {
                // Dash: set phoneme to em-dash
                tok.phonemes = Some("\u{2014}".to_string());
                tok.underscore.rating = Some(3);
                output.push(TokenOrGroup::Single(tok));
            } else if PUNCT_TAGS.contains(&tok.tag.as_str())
                && !tok.text.chars().all(|c| c.is_alphabetic())
            {
                // Punctuation token
                if let Some(ph) = punct_tag_phoneme(&tok.tag) {
                    tok.phonemes = Some(ph.to_string());
                } else {
                    // Filter through PUNCTS or NON_QUOTE_PUNCTS
                    let filtered: String =
                        tok.text.chars().filter(|c| PUNCTS.contains(*c)).collect();
                    if !filtered.is_empty() {
                        tok.phonemes = Some(filtered);
                    } else {
                        tok.phonemes = Some(String::new());
                    }
                }
                tok.underscore.rating = Some(3);
                output.push(TokenOrGroup::Single(tok));
            } else {
                // Apply pending currency
                if let Some(cur) = pending_currency {
                    if tok.tag == "CD" {
                        tok.underscore.currency = Some(cur);
                    } else {
                        pending_currency = None;
                    }
                }
                output.push(TokenOrGroup::Single(tok));
            }

            continue;
        }

        // Multiple subtokens: create MToken for each and group them
        let mut group: Vec<MToken> = Vec::new();

        for (i, sub_text) in subtokens.iter().enumerate() {
            let is_last = i == subtokens.len() - 1;
            let is_junk = sub_text.chars().all(|c| SUBTOKEN_JUNKS.contains(c));

            let mut sub_tok = MToken::new(
                sub_text.as_str(),
                if is_junk {
                    ":".to_string()
                } else {
                    tok.tag.clone()
                },
                if is_last {
                    tok.whitespace.clone()
                } else {
                    String::new()
                },
            );
            sub_tok.underscore.is_head = i == 0;

            if is_junk {
                sub_tok.phonemes = Some(String::new());
                sub_tok.underscore.rating = Some(3);
            }

            // Apply pending currency to CD-tagged subtokens
            if let Some(cur) = pending_currency {
                if sub_tok.tag == "CD" {
                    sub_tok.underscore.currency = Some(cur);
                }
            }

            if !group.is_empty() && !is_last {
                // Continue building the group
                group.push(sub_tok);
            } else if !group.is_empty() && is_last {
                group.push(sub_tok);
                output.push(TokenOrGroup::Group(std::mem::take(&mut group)));
            } else if !is_last {
                group.push(sub_tok);
            } else {
                // Single subtoken that is also the last
                output.push(TokenOrGroup::Single(sub_tok));
            }
        }

        // Flush any remaining group
        if !group.is_empty() {
            output.push(TokenOrGroup::Group(group));
        }

        // Clear currency after non-CD token
        if tok.tag != "$" && tok.tag != "CD" {
            pending_currency = None;
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- tokenize_simple tests ------------------------------------------------

    #[test]
    fn simple_basic_sentence() {
        let tokens = tokenize_simple("Hello world");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[0].tag, "DEFAULT");
        assert_eq!(tokens[0].whitespace, " ");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[1].tag, "DEFAULT");
        assert_eq!(tokens[1].whitespace, "");
    }

    #[test]
    fn simple_punctuation() {
        let tokens = tokenize_simple("Hello, world!");
        // Comma and exclamation are now split off
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[0].whitespace, "");
        assert_eq!(tokens[1].text, ",");
        assert_eq!(tokens[1].tag, ",");
        assert_eq!(tokens[1].whitespace, " ");
        assert_eq!(tokens[2].text, "world");
        assert_eq!(tokens[2].whitespace, "");
        assert_eq!(tokens[3].text, "!");
        assert_eq!(tokens[3].tag, ".");
        assert_eq!(tokens[3].whitespace, "");
    }

    #[test]
    fn simple_standalone_punct() {
        let tokens = tokenize_simple("a . b");
        assert_eq!(tokens[1].text, ".");
        assert_eq!(tokens[1].tag, ".");
    }

    #[test]
    fn split_period_from_word() {
        let tokens = tokenize_simple("Hello.");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[1].text, ".");
        assert_eq!(tokens[1].tag, ".");
    }

    #[test]
    fn split_question_mark() {
        let tokens = tokenize_simple("Really?");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "Really");
        assert_eq!(tokens[1].text, "?");
        assert_eq!(tokens[1].tag, ".");
    }

    #[test]
    fn split_leading_paren() {
        let tokens = tokenize_simple("(Hello)");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "(");
        assert_eq!(tokens[1].text, "Hello");
        assert_eq!(tokens[2].text, ")");
    }

    #[test]
    fn split_leading_currency() {
        let tokens = tokenize_simple("$100");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "$");
        assert_eq!(tokens[0].tag, "$");
        assert_eq!(tokens[1].text, "100");
        assert_eq!(tokens[1].tag, "CD");
    }

    #[test]
    fn no_split_contraction() {
        // ASCII apostrophe is NOT in LEADING_PUNCT or TRAILING_PUNCT
        let tokens = tokenize_simple("don't");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "don't");
    }

    #[test]
    fn no_split_decimal() {
        // "3.14" is recognized as number-like, no split
        let tokens = tokenize_simple("3.14");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "3.14");
        assert_eq!(tokens[0].tag, "CD");
    }

    #[test]
    fn split_multiple_trailing() {
        let tokens = tokenize_simple("What!?");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "What");
        assert_eq!(tokens[1].text, "!");
        assert_eq!(tokens[2].text, "?");
    }

    #[test]
    fn split_sentence_periods() {
        let tokens = tokenize_simple("Hello. World.");
        // "Hello." → ["Hello", "."], "World." → ["World", "."]
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[1].text, ".");
        assert_eq!(tokens[1].tag, ".");
        assert_eq!(tokens[1].whitespace, " ");
        assert_eq!(tokens[2].text, "World");
        assert_eq!(tokens[3].text, ".");
        assert_eq!(tokens[3].tag, ".");
    }

    #[test]
    fn pure_punct_split_into_chars() {
        // "..." → three separate "." tokens
        let tokens = tokenize_simple("...");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, ".");
        assert_eq!(tokens[1].text, ".");
        assert_eq!(tokens[2].text, ".");
    }

    #[test]
    fn pure_mixed_punct_split() {
        let tokens = tokenize_simple("!?!?");
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "!");
        assert_eq!(tokens[1].text, "?");
        assert_eq!(tokens[2].text, "!");
        assert_eq!(tokens[3].text, "?");
    }

    #[test]
    fn single_punct_not_split_further() {
        let tokens = tokenize_simple(".");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, ".");
        assert_eq!(tokens[0].tag, ".");
    }

    #[test]
    fn split_parens_with_tags() {
        let tokens = tokenize_simple("(Hello)");
        assert_eq!(tokens[0].text, "(");
        assert_eq!(tokens[0].tag, "-LRB-");
        assert_eq!(tokens[2].text, ")");
        assert_eq!(tokens[2].tag, "-RRB-");
    }

    #[test]
    fn split_ascii_quotes() {
        let tokens = tokenize_simple("\"Hello\"");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "\"");
        assert_eq!(tokens[0].tag, "``"); // opening
        assert_eq!(tokens[2].text, "\"");
        assert_eq!(tokens[2].tag, "''"); // closing
    }

    #[test]
    fn split_quoted_sentence_with_period() {
        // She said, "hello." → She | said | , | " | hello | . | "
        let tokens = tokenize_simple("She said, \"hello.\"");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&","), "comma should be split: {texts:?}");
        assert!(texts.contains(&"."), "period should be split: {texts:?}");
        assert!(
            texts.iter().filter(|t| **t == "\"").count() == 2,
            "two quotes should be split: {texts:?}"
        );
    }

    #[test]
    fn split_number_trailing_period() {
        // "3." at end of sentence → ["3", "."] (period is sentence-ending, not decimal)
        // Matches spaCy: "I have 3." → [I, have, 3, .]
        let tokens = tokenize_simple("I have 3.");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"3"), "number should be separate: {texts:?}");
        assert!(texts.contains(&"."), "period should be split: {texts:?}");
    }

    #[test]
    fn no_split_decimal_number() {
        // "3.14" stays together — dot has digits on both sides
        let tokens = tokenize_simple("He scored 3.14 points.");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(
            texts.contains(&"3.14"),
            "decimal should stay together: {texts:?}"
        );
    }

    #[test]
    fn split_number_trailing_comma() {
        // "Buy 3, get 1" → "3" and "," split
        let tokens = tokenize_simple("Buy 3, get 1.");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"3"), "number should be separate: {texts:?}");
        assert!(texts.contains(&","), "comma should be split: {texts:?}");
    }

    // Known edge cases: pure-punctuation inputs like "...", "()", "!?!?" are
    // degenerate (no words). They pass through as single tokens via
    // is_all_puncts/simple_tag but may produce empty or unexpected phonemes
    // downstream. This is acceptable — these inputs don't occur in real speech.

    #[test]
    fn simple_currency() {
        let tokens = tokenize_simple("$ 100");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "$");
        assert_eq!(tokens[0].tag, "$");
        assert_eq!(tokens[1].text, "100");
        assert_eq!(tokens[1].tag, "CD");
    }

    #[test]
    fn simple_number() {
        let tokens = tokenize_simple("42 1,000 3.14");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].tag, "CD");
        assert_eq!(tokens[1].tag, "CD");
        assert_eq!(tokens[2].tag, "CD");
    }

    #[test]
    fn simple_dash_tag() {
        let tokens = tokenize_simple("a — b");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[1].text, "—");
        assert_eq!(tokens[1].tag, ":");
    }

    #[test]
    fn simple_all_heads() {
        let tokens = tokenize_simple("one two three");
        for tok in &tokens {
            assert!(tok.underscore.is_head);
        }
    }

    #[test]
    fn simple_empty_input() {
        let tokens = tokenize_simple("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn simple_whitespace_only() {
        let tokens = tokenize_simple("   ");
        assert!(tokens.is_empty());
    }

    #[test]
    fn simple_multiple_spaces() {
        let tokens = tokenize_simple("hello   world");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].whitespace, "   ");
    }

    // -- subtokenize tests ----------------------------------------------------

    #[test]
    fn subtokenize_simple_word() {
        let result = subtokenize("hello");
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn subtokenize_camel_case() {
        let result = subtokenize("iPhone");
        // 'i' is lowercase before 'P' uppercase → should split
        assert!(
            result.len() >= 2,
            "Expected split for camelCase: {:?}",
            result
        );
    }

    #[test]
    fn subtokenize_number_word() {
        let result = subtokenize("test123");
        assert!(
            result.len() >= 2,
            "Expected split for word+number: {:?}",
            result
        );
    }

    #[test]
    fn subtokenize_hyphenated() {
        let result = subtokenize("well-known");
        assert!(result.len() >= 3, "Expected split on hyphen: {:?}", result);
        assert!(result.contains(&"-".to_string()));
    }

    #[test]
    fn subtokenize_apostrophe() {
        let result = subtokenize("don't");
        assert!(!result.is_empty(), "Should handle apostrophe: {:?}", result);
    }

    #[test]
    fn subtokenize_all_caps() {
        let result = subtokenize("NASA");
        // All uppercase letters — should stay as one or split per character
        assert!(!result.is_empty());
    }

    #[test]
    fn subtokenize_leading_quotes() {
        let result = subtokenize("'hello");
        assert_eq!(result[0], "'");
    }

    #[test]
    fn subtokenize_trailing_quotes() {
        let result = subtokenize("hello'");
        assert!(result.last().unwrap().contains('\''));
    }

    #[test]
    fn subtokenize_digits_with_comma() {
        let result = subtokenize("1,000");
        // Should keep number together
        assert!(
            result.contains(&"1,000".to_string()),
            "Expected number to stay together: {:?}",
            result
        );
    }

    // -- fold_left tests ------------------------------------------------------

    #[test]
    fn fold_left_merges_non_heads() {
        let tok1 = {
            let mut t = MToken::new("can", "NN", "");
            t.underscore.is_head = true;
            t.phonemes = Some("k".into());
            t
        };
        let tok2 = {
            let mut t = MToken::new("not", "RB", " ");
            t.underscore.is_head = false;
            t.phonemes = Some("n".into());
            t
        };

        let result = fold_left(vec![tok1, tok2]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "cannot");
        assert_eq!(result[0].phonemes, Some("kn".into()));
        assert_eq!(result[0].whitespace, " ");
    }

    #[test]
    fn fold_left_preserves_heads() {
        let tok1 = {
            let mut t = MToken::new("a", "DT", " ");
            t.underscore.is_head = true;
            t
        };
        let tok2 = {
            let mut t = MToken::new("b", "NN", "");
            t.underscore.is_head = true;
            t
        };

        let result = fold_left(vec![tok1, tok2]);
        assert_eq!(result.len(), 2);
    }

    // -- retokenize tests -----------------------------------------------------

    #[test]
    fn test_retokenize_period_gets_phoneme() {
        let mut tok = MToken::new(".", ".", "");
        tok.underscore.is_head = true;
        let result = retokenize(vec![tok]);
        assert_eq!(result.len(), 1);
        match &result[0] {
            TokenOrGroup::Single(t) => {
                assert_eq!(t.phonemes.as_deref(), Some("."));
            }
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn test_retokenize_comma_gets_phoneme() {
        let mut tok = MToken::new(",", ",", "");
        tok.underscore.is_head = true;
        let result = retokenize(vec![tok]);
        assert_eq!(result.len(), 1);
        match &result[0] {
            TokenOrGroup::Single(t) => {
                assert_eq!(t.phonemes.as_deref(), Some(","));
            }
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn test_retokenize_exclamation_gets_phoneme() {
        let mut tok = MToken::new("!", ".", "");
        tok.underscore.is_head = true;
        let result = retokenize(vec![tok]);
        assert_eq!(result.len(), 1);
        match &result[0] {
            TokenOrGroup::Single(t) => {
                assert_eq!(t.phonemes.as_deref(), Some("!"));
            }
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn test_retokenize_dash_becomes_emdash() {
        let mut tok = MToken::new("-", ":", "");
        tok.underscore.is_head = true;
        let result = retokenize(vec![tok]);
        assert_eq!(result.len(), 1);
        match &result[0] {
            TokenOrGroup::Single(t) => {
                assert_eq!(t.phonemes.as_deref(), Some("\u{2014}"));
            }
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn test_retokenize_bracket_tags() {
        let mut tok_l = MToken::new("(", "-LRB-", "");
        tok_l.underscore.is_head = true;
        let result_l = retokenize(vec![tok_l]);
        assert_eq!(result_l.len(), 1);
        match &result_l[0] {
            TokenOrGroup::Single(t) => {
                assert_eq!(t.phonemes.as_deref(), Some("("));
            }
            _ => panic!("expected Single for left bracket"),
        }

        let mut tok_r = MToken::new(")", "-RRB-", "");
        tok_r.underscore.is_head = true;
        let result_r = retokenize(vec![tok_r]);
        assert_eq!(result_r.len(), 1);
        match &result_r[0] {
            TokenOrGroup::Single(t) => {
                assert_eq!(t.phonemes.as_deref(), Some(")"));
            }
            _ => panic!("expected Single for right bracket"),
        }
    }

    #[test]
    fn test_retokenize_currency_silent() {
        let mut tok = MToken::new("$", "$", "");
        tok.underscore.is_head = true;
        let result = retokenize(vec![tok]);
        assert_eq!(result.len(), 1);
        match &result[0] {
            TokenOrGroup::Single(t) => {
                assert_eq!(t.phonemes.as_deref(), Some(""));
                assert_eq!(t.underscore.rating, Some(4));
            }
            _ => panic!("expected Single"),
        }
    }

    #[test]
    fn test_retokenize_groups_adjacent() {
        let mut tok = MToken::new("well-known", "JJ", " ");
        tok.underscore.is_head = true;
        let result = retokenize(vec![tok]);
        assert_eq!(result.len(), 1);
        match &result[0] {
            TokenOrGroup::Group(group) => {
                assert!(
                    group.len() >= 2,
                    "expected multiple subtokens in group, got {:?}",
                    group
                );
            }
            _ => panic!("expected Group, got {:?}", result[0]),
        }
    }
}
