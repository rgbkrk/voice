#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VoxtralTextNormalizationOptions {
    /// Expand compact numeric forms that Voxtral currently handles poorly.
    pub numeric: bool,
    /// Rewrite known hard-to-pronounce project names into pronounceable hints.
    pub pronunciation_aliases: bool,
}

pub const DEFAULT_SUGGESTED_MAX_FRAMES: usize = 32;
pub const SUGGESTED_MAX_FRAMES_CAP: usize = 64;

/// Normalize numeric text forms that Voxtral currently handles poorly when
/// they are left as compact written forms.
pub fn normalize_tts_text(text: &str) -> String {
    normalize_tts_text_with_options(
        text,
        VoxtralTextNormalizationOptions {
            numeric: true,
            pronunciation_aliases: false,
        },
    )
}

pub fn normalize_tts_text_with_options(
    text: &str,
    options: VoxtralTextNormalizationOptions,
) -> String {
    let text = if options.numeric {
        normalize_numeric_tts_text(text)
    } else {
        text.to_string()
    };

    if options.pronunciation_aliases {
        apply_pronunciation_aliases(&text)
    } else {
        text
    }
}

pub fn suggest_max_frames_for_text(text: &str) -> usize {
    let word_count = count_speech_tokens(text);
    let raw_frames = if word_count <= 4 {
        DEFAULT_SUGGESTED_MAX_FRAMES
    } else {
        DEFAULT_SUGGESTED_MAX_FRAMES + (word_count - 4) * 3
    };
    round_up_to_multiple(raw_frames, 8)
        .clamp(DEFAULT_SUGGESTED_MAX_FRAMES, SUGGESTED_MAX_FRAMES_CAP)
}

fn count_speech_tokens(text: &str) -> usize {
    let mut count = 0;
    let mut in_token = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            if !in_token {
                count += 1;
                in_token = true;
            }
        } else {
            in_token = false;
        }
    }
    count
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    debug_assert!(multiple > 0);
    if value == 0 {
        0
    } else {
        value.div_ceil(multiple) * multiple
    }
}

fn normalize_numeric_tts_text(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    let bytes = text.as_bytes();

    while index < bytes.len() {
        let byte = bytes[index];
        if byte.is_ascii_digit() {
            if let Some((replacement, end)) = parse_time(text, index) {
                output.push_str(&replacement);
                index = end;
                continue;
            }
            if let Some((replacement, end)) = parse_dotted_number(text, index) {
                output.push_str(&replacement);
                index = end;
                continue;
            }
            let number = parse_ascii_digits(text, index);
            output.push_str(&number_to_words_for_token(number.value, number.digits));
            index = number.end;
            continue;
        }

        if byte.is_ascii_uppercase() {
            if let Some((replacement, end)) = parse_letter_number(text, index) {
                output.push_str(&replacement);
                index = end;
                continue;
            }
        }

        let ch = text[index..]
            .chars()
            .next()
            .expect("index is inside a valid UTF-8 string");
        output.push(ch);
        index += ch.len_utf8();
    }

    output
}

fn apply_pronunciation_aliases(text: &str) -> String {
    replace_ascii_word(text, "Voxtral", "Vox trell")
}

fn replace_ascii_word(text: &str, word: &str, replacement: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let word_bytes = word.as_bytes();
    let mut index = 0;

    while index < text.len() {
        let candidate_end = index.saturating_add(word.len());
        let matches_word = bytes
            .get(index..candidate_end)
            .is_some_and(|candidate| candidate.eq_ignore_ascii_case(word_bytes));
        if matches_word && is_word_boundary(text, index, candidate_end) {
            output.push_str(replacement);
            index = candidate_end;
            continue;
        }

        let ch = text[index..]
            .chars()
            .next()
            .expect("index is inside a valid UTF-8 string");
        output.push(ch);
        index += ch.len_utf8();
    }

    output
}

fn is_word_boundary(text: &str, start: usize, end: usize) -> bool {
    let bytes = text.as_bytes();
    let before = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
    let after = bytes
        .get(end)
        .is_none_or(|byte| !byte.is_ascii_alphanumeric());
    before && after
}

fn parse_time(text: &str, start: usize) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    let hour = parse_ascii_digits(text, start);
    if hour.value == 0 || hour.value > 23 || bytes.get(hour.end) != Some(&b':') {
        return None;
    }
    let minute_start = hour.end + 1;
    if !bytes.get(minute_start).is_some_and(u8::is_ascii_digit)
        || !bytes.get(minute_start + 1).is_some_and(u8::is_ascii_digit)
    {
        return None;
    }
    if bytes.get(minute_start + 2).is_some_and(u8::is_ascii_digit) {
        return None;
    }
    let minute =
        ((bytes[minute_start] - b'0') as u32) * 10 + (bytes[minute_start + 1] - b'0') as u32;
    if minute > 59 {
        return None;
    }

    let mut parts = Vec::new();
    parts.push(number_to_words_for_token(hour.value, hour.digits));
    if minute == 0 {
        parts.push("o'clock".to_string());
    } else if minute < 10 {
        parts.push(format!("oh {}", number_to_words(minute)));
    } else {
        parts.push(number_to_words(minute));
    }

    Some((parts.join(" "), minute_start + 2))
}

fn parse_dotted_number(text: &str, start: usize) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    let first = parse_ascii_digits(text, start);
    let mut end = first.end;
    if bytes.get(end) != Some(&b'.') || !bytes.get(end + 1).is_some_and(u8::is_ascii_digit) {
        return None;
    }

    let mut parts = vec![number_to_words_for_token(first.value, first.digits)];
    while bytes.get(end) == Some(&b'.') && bytes.get(end + 1).is_some_and(u8::is_ascii_digit) {
        let number = parse_ascii_digits(text, end + 1);
        parts.push("point".to_string());
        parts.push(number_to_words_for_token(number.value, number.digits));
        end = number.end;
    }

    Some((parts.join(" "), end))
}

fn parse_letter_number(text: &str, start: usize) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    if start > 0 && bytes[start - 1].is_ascii_alphanumeric() {
        return None;
    }
    if !bytes[start].is_ascii_uppercase() || !bytes.get(start + 1).is_some_and(u8::is_ascii_digit) {
        return None;
    }
    let letter = bytes[start] as char;
    let number = parse_ascii_digits(text, start + 1);
    let end = number.end;
    if bytes
        .get(end)
        .is_some_and(|byte| byte.is_ascii_alphanumeric())
    {
        return None;
    }
    Some((
        format!(
            "{} {}",
            letter,
            number_to_words_for_token(number.value, number.digits)
        ),
        end,
    ))
}

struct NumberToken<'a> {
    value: u32,
    end: usize,
    digits: &'a str,
}

fn parse_ascii_digits(text: &str, start: usize) -> NumberToken<'_> {
    let bytes = text.as_bytes();
    let mut end = start;
    let mut value = 0u32;
    while let Some(byte) = bytes.get(end).filter(|byte| byte.is_ascii_digit()) {
        value = value
            .saturating_mul(10)
            .saturating_add((*byte - b'0') as u32);
        end += 1;
    }
    NumberToken {
        value,
        end,
        digits: &text[start..end],
    }
}

fn number_to_words_for_token(number: u32, digits: &str) -> String {
    if digits.len() > 1 && digits.as_bytes()[0] == b'0' {
        return digit_words(digits.as_bytes());
    }
    number_to_words(number)
}

fn number_to_words(number: u32) -> String {
    match number {
        0 => "zero".to_string(),
        1..=19 => small_number_word(number).to_string(),
        20..=99 => {
            let tens = number / 10;
            let ones = number % 10;
            if ones == 0 {
                tens_word(tens).to_string()
            } else {
                format!("{} {}", tens_word(tens), small_number_word(ones))
            }
        }
        100..=999 => {
            let hundreds = number / 100;
            let rest = number % 100;
            if rest == 0 {
                format!("{} hundred", small_number_word(hundreds))
            } else {
                format!(
                    "{} hundred {}",
                    small_number_word(hundreds),
                    number_to_words(rest)
                )
            }
        }
        1000..=9999 => {
            let thousands = number / 1000;
            let rest = number % 1000;
            if rest == 0 {
                format!("{} thousand", small_number_word(thousands))
            } else {
                format!(
                    "{} thousand {}",
                    small_number_word(thousands),
                    number_to_words(rest)
                )
            }
        }
        _ => digit_words(number.to_string().as_bytes()),
    }
}

fn digit_words(digits: &[u8]) -> String {
    digits
        .iter()
        .filter_map(|digit| digit.is_ascii_digit().then_some((*digit - b'0') as u32))
        .map(number_to_words)
        .collect::<Vec<_>>()
        .join(" ")
}

fn small_number_word(number: u32) -> &'static str {
    match number {
        0 => "zero",
        1 => "one",
        2 => "two",
        3 => "three",
        4 => "four",
        5 => "five",
        6 => "six",
        7 => "seven",
        8 => "eight",
        9 => "nine",
        10 => "ten",
        11 => "eleven",
        12 => "twelve",
        13 => "thirteen",
        14 => "fourteen",
        15 => "fifteen",
        16 => "sixteen",
        17 => "seventeen",
        18 => "eighteen",
        19 => "nineteen",
        _ => unreachable!("small number out of range"),
    }
}

fn tens_word(tens: u32) -> &'static str {
    match tens {
        2 => "twenty",
        3 => "thirty",
        4 => "forty",
        5 => "fifty",
        6 => "sixty",
        7 => "seventy",
        8 => "eighty",
        9 => "ninety",
        _ => unreachable!("tens out of range"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        normalize_tts_text, normalize_tts_text_with_options, suggest_max_frames_for_text,
        VoxtralTextNormalizationOptions,
    };

    #[test]
    fn leaves_plain_prompt_unchanged() {
        assert_eq!(normalize_tts_text("hello world"), "hello world");
        assert_eq!(
            normalize_tts_text("Voxtral should sound clear."),
            "Voxtral should sound clear."
        );
    }

    #[test]
    fn expands_ticket_version_and_time() {
        assert_eq!(
            normalize_tts_text("Read ticket A17, version 2.4.1, at 9:30 PM."),
            "Read ticket A seventeen, version two point four point one, at nine thirty PM."
        );
    }

    #[test]
    fn preserves_acronyms_while_expanding_time() {
        assert_eq!(
            normalize_tts_text("Use API on CPU at 12:05 AM."),
            "Use API on CPU at twelve oh five AM."
        );
    }

    #[test]
    fn preserves_trailing_sentence_period_after_dotted_number() {
        assert_eq!(
            normalize_tts_text("Ship version 10.2."),
            "Ship version ten point two."
        );
    }

    #[test]
    fn spells_general_leading_zero_tokens_digit_by_digit() {
        assert_eq!(
            normalize_tts_text("Use code 007."),
            "Use code zero zero seven."
        );
    }

    #[test]
    fn pronunciation_aliases_are_opt_in_and_word_bounded() {
        assert_eq!(
            normalize_tts_text_with_options(
                "Voxtral should say voxtral clearly.",
                VoxtralTextNormalizationOptions {
                    numeric: false,
                    pronunciation_aliases: false,
                }
            ),
            "Voxtral should say voxtral clearly."
        );
        assert_eq!(
            normalize_tts_text_with_options(
                "Voxtral should say voxtral clearly.",
                VoxtralTextNormalizationOptions {
                    numeric: false,
                    pronunciation_aliases: true,
                }
            ),
            "Vox trell should say Vox trell clearly."
        );
        assert_eq!(
            normalize_tts_text_with_options(
                "Voxtralized text should not change.",
                VoxtralTextNormalizationOptions {
                    numeric: false,
                    pronunciation_aliases: true,
                }
            ),
            "Voxtralized text should not change."
        );
    }

    #[test]
    fn pronunciation_aliases_do_not_panic_on_utf8_boundary_overlap() {
        assert_eq!(
            normalize_tts_text_with_options(
                "Voxtraé should remain unchanged.",
                VoxtralTextNormalizationOptions {
                    numeric: false,
                    pronunciation_aliases: true,
                }
            ),
            "Voxtraé should remain unchanged."
        );
    }

    #[test]
    fn text_normalization_options_can_combine_numeric_and_pronunciation_rewrites() {
        assert_eq!(
            normalize_tts_text_with_options(
                "Voxtral reads A17 at 9:30 PM.",
                VoxtralTextNormalizationOptions {
                    numeric: true,
                    pronunciation_aliases: true,
                }
            ),
            "Vox trell reads A seventeen at nine thirty PM."
        );
    }

    #[test]
    fn suggested_frame_budget_scales_with_preprocessed_text_length() {
        assert_eq!(suggest_max_frames_for_text("hello world"), 32);
        assert_eq!(
            suggest_max_frames_for_text("A fast reply should arrive naturally."),
            40
        );
        assert_eq!(
            suggest_max_frames_for_text(
                "Vox trell should pronounce Vox trell clearly in a short answer."
            ),
            56
        );
        assert_eq!(
            suggest_max_frames_for_text(
                "Read ticket A seventeen, version two point four point one, at nine thirty PM."
            ),
            64
        );
        assert_eq!(
            suggest_max_frames_for_text("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen"),
            64
        );
    }
}
