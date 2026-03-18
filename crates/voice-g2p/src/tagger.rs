//! Averaged perceptron POS tagger.
//!
//! A pure-Rust reimplementation of the NLTK-style averaged perceptron tagger,
//! ported from misaki-rs. Replaces the previous approach of shelling out to
//! spaCy via `uv run` for POS tagging.
//!
//! Model data (weights, classes, tags) is embedded at compile time via
//! `include_str!` and lazily parsed into a `PerceptronTagger` singleton
//! on first access through [`global_tagger()`].

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Raw model data baked into the binary.
const WEIGHTS_JSON: &str = include_str!("../data/tagger/weights.json");
const CLASSES_TXT: &str = include_str!("../data/tagger/classes.txt");
const TAGS_JSON: &str = include_str!("../data/tagger/tags.json");

/// Global singleton tagger, initialized once on first access.
static TAGGER: OnceLock<PerceptronTagger> = OnceLock::new();

/// Returns a reference to the lazily-initialized global [`PerceptronTagger`].
///
/// The first call parses the embedded JSON weights (~5.7 MB) and tag mappings.
/// Subsequent calls return the cached instance with no overhead.
pub fn global_tagger() -> &'static PerceptronTagger {
    TAGGER.get_or_init(|| PerceptronTagger::new(WEIGHTS_JSON, CLASSES_TXT, TAGS_JSON))
}

/// A linear classifier that scores classes by summing weighted feature values.
///
/// Each feature maps to a set of per-class weights. Prediction is the class
/// with the highest total score across all active features.
#[derive(Debug, Serialize, Deserialize)]
pub struct AveragedPerceptron {
    /// Mapping from feature name -> (class name -> weight).
    pub feature_weights: HashMap<String, HashMap<String, f32>>,
    /// Ordered list of all known classes (POS tags). Used as a tiebreaker:
    /// when scores are equal, the class appearing first wins.
    pub classes: Vec<String>,
}

impl AveragedPerceptron {
    /// Construct from raw JSON weights and a newline-delimited classes list.
    pub fn new(weights_json: &str, classes_txt: &str) -> Self {
        let feature_weights: HashMap<String, HashMap<String, f32>> =
            serde_json::from_str(weights_json).expect("Failed to parse tagger weights.json");
        let classes: Vec<String> = classes_txt
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        Self {
            feature_weights,
            classes,
        }
    }

    /// Score all classes against the given feature vector and return the
    /// best class along with its score.
    pub fn predict(&self, word_features: HashMap<String, usize>) -> (&str, f32) {
        let mut scores: HashMap<&str, f32> = HashMap::new();

        for (feature, value) in &word_features {
            if *value == 0 {
                continue;
            }
            if let Some(weights) = self.feature_weights.get(feature.as_str()) {
                let v = *value as f32;
                for (label, weight) in weights {
                    *scores.entry(label.as_str()).or_insert(0.0) += weight * v;
                }
            }
        }

        let class = self
            .classes
            .iter()
            .max_by(|a, b| {
                let sa = scores.get(a.as_str()).unwrap_or(&0.0);
                let sb = scores.get(b.as_str()).unwrap_or(&0.0);
                sa.partial_cmp(sb).unwrap()
            })
            .expect("classes list must not be empty");

        let max_score = *scores.get(class.as_str()).unwrap_or(&0.0);
        (class.as_str(), max_score)
    }
}

/// POS tagger built on an [`AveragedPerceptron`] model.
///
/// Maintains a lookup table (`tags`) for high-frequency unambiguous words that
/// can be tagged without running the perceptron, and falls back to feature-based
/// prediction for everything else.
pub struct PerceptronTagger {
    model: AveragedPerceptron,
    /// Unambiguous word -> tag mapping (e.g. "the" -> "DT").
    tags: HashMap<String, String>,
}

impl PerceptronTagger {
    /// Build from raw data strings (JSON weights, newline-delimited classes,
    /// JSON tag map).
    pub fn new(weights_json: &str, classes_txt: &str, tags_json: &str) -> Self {
        let tags: HashMap<String, String> =
            serde_json::from_str(tags_json).expect("Failed to parse tagger tags.json");
        Self {
            model: AveragedPerceptron::new(weights_json, classes_txt),
            tags,
        }
    }

    /// Tag a slice of words, returning a `Vec<Tag>` with one entry per word.
    ///
    /// Uses bigram tag history as context features. Words present in the
    /// unambiguous `tags` table are assigned directly (conf = 1.0); all
    /// others go through the perceptron.
    pub fn tag<'a>(&self, words: &[&'a str]) -> Vec<Tag<'a>> {
        let mut prev = "-START-".to_string();
        let mut prev2 = "-START2-".to_string();
        let mut output = Vec::with_capacity(words.len());

        // Build context array with sentinel tokens on both ends.
        let mut context: Vec<&str> = Vec::with_capacity(words.len() + 4);
        context.push("-START-");
        context.push("-START2-");
        for &token in words {
            context.push(Self::normalize(token));
        }
        context.push("-END-");
        context.push("-END2-");

        for (i, &token) in words.iter().enumerate() {
            if let Some(tag) = self.tags.get(token) {
                output.push(Tag {
                    word: token,
                    tag: tag.clone(),
                    conf: 1.0,
                });
                prev2 = prev;
                prev = tag.clone();
            } else {
                let features = Self::get_features(i + 2, token, &context, &prev, &prev2);
                let (tag, conf) = self.model.predict(features);
                let tag_string = tag.to_string();
                output.push(Tag {
                    word: token,
                    tag: tag_string.clone(),
                    conf,
                });
                prev2 = prev;
                prev = tag_string;
            }
        }

        output
    }

    /// Normalize a token for the context window.
    ///
    /// Replaces hyphenated non-prefix tokens with `!HYPHEN`, 4-digit numbers
    /// with `!YEAR`, and other digit-leading tokens with `!DIGITS`.
    fn normalize(token: &str) -> &str {
        if token.contains('-') && !token.starts_with('-') {
            "!HYPHEN"
        } else if token.len() == 4 && token.parse::<usize>().is_ok() {
            "!YEAR"
        } else if token.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            "!DIGITS"
        } else {
            token
        }
    }

    /// Extract feature vector for the word at position `i` in `context`.
    ///
    /// Features encode the word itself, its prefix/suffix, surrounding words
    /// and their suffixes, and the previous two predicted tags.
    fn get_features(
        i: usize,
        word: &str,
        context: &[&str],
        prev: &str,
        prev2: &str,
    ) -> HashMap<String, usize> {
        let mut features = HashMap::with_capacity(14);

        features.insert("bias".to_string(), 1);

        // Suffix (last 3 chars) of current word.
        let suffix = suffix3(word);
        features.insert(format!("i suffix {}", suffix), 1);

        // First character of current word.
        let pref1: String = word.chars().take(1).collect();
        features.insert(format!("i pref1 {}", pref1), 1);

        // Tag history features.
        features.insert(format!("i-1 tag {}", prev), 1);
        features.insert(format!("i-2 tag {}", prev2), 1);
        features.insert(format!("i tag+i-2 tag {} {}", prev, prev2), 1);

        // Current word in context.
        features.insert(format!("i word {}", context[i]), 1);
        features.insert(format!("i-1 tag+i word {} {}", prev, context[i]), 1);

        // Surrounding words.
        features.insert(format!("i-1 word {}", context[i - 1]), 1);
        features.insert(format!("i-2 word {}", context[i - 2]), 1);
        features.insert(format!("i+1 word {}", context[i + 1]), 1);
        features.insert(format!("i+2 word {}", context[i + 2]), 1);

        // Suffixes of neighboring words.
        features.insert(format!("i+1 suffix {}", suffix3(context[i + 1])), 1);
        features.insert(format!("i-1 suffix {}", suffix3(context[i - 1])), 1);

        features
    }
}

/// Last 3 characters of a string (or fewer if the string is shorter).
fn suffix3(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let start = chars.len().saturating_sub(3);
    chars[start..].iter().collect()
}

/// A single tagged word.
pub struct Tag<'a> {
    /// The original word (borrows from the input slice).
    pub word: &'a str,
    /// Predicted POS tag (owned because the global static lifetime makes
    /// borrowing from the tagger's internal maps ergonomically painful).
    pub tag: String,
    /// Confidence score — sum of weighted features for the predicted class,
    /// or 1.0 for words resolved via the unambiguous tag table.
    pub conf: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_tagger_loads() {
        let tagger = global_tagger();
        assert!(!tagger.model.classes.is_empty());
        assert!(!tagger.tags.is_empty());
    }

    #[test]
    fn test_tag_simple_sentence() {
        let tagger = global_tagger();
        let words = vec!["The", "cat", "sat", "on", "the", "mat"];
        let tags = tagger.tag(&words);
        assert_eq!(tags.len(), words.len());
        for tag in &tags {
            assert!(!tag.tag.is_empty());
        }
    }

    #[test]
    fn test_suffix3() {
        assert_eq!(suffix3("hello"), "llo");
        assert_eq!(suffix3("hi"), "hi");
        assert_eq!(suffix3("a"), "a");
        assert_eq!(suffix3(""), "");
    }
}
