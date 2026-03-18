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

/// Returns true if every character in `s` is a punctuation character from PUNCTS.
fn is_all_puncts(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| PUNCTS.contains(c))
}

/// Returns true if every character in `s` is a digit, comma, or dot.
fn is_number_like(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_digit() || c == ',' || c == '.')
}

/// Returns true if every character in `s` is a currency symbol.
fn is_currency(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| matches!(c, '$' | '£' | '€'))
}

/// Assign a simple POS tag based on the token text.
fn simple_tag(text: &str) -> &'static str {
    if is_currency(text) {
        "$"
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

/// Tokenize text using simple whitespace splitting with heuristic POS tags.
pub fn tokenize_simple(text: &str) -> Vec<MToken> {
    let mut tokens = Vec::new();
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
                let tag = simple_tag(word);
                let mut tok = MToken::new(word, tag, ws);
                tok.underscore.is_head = true;
                tokens.push(tok);
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
        let tag = simple_tag(word);
        let mut tok = MToken::new(word, tag, "");
        tok.underscore.is_head = true;
        tokens.push(tok);
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
        // "Hello," is one token, "world!" is another
        assert_eq!(tokens.len(), 2);
        // The comma is part of "Hello," — the simple tokenizer doesn't split on punct within words
        assert_eq!(tokens[0].text, "Hello,");
        assert_eq!(tokens[1].text, "world!");
    }

    #[test]
    fn simple_standalone_punct() {
        let tokens = tokenize_simple("a . b");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[1].text, ".");
        assert_eq!(tokens[1].tag, ".");
    }

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
        assert!(result.len() >= 1, "Should handle apostrophe: {:?}", result);
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
}
