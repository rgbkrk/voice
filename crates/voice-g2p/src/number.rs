/// English number-to-words conversion, replacing Python's `num2words` library.
///
/// Three public functions cover cardinal, ordinal, and year forms.
/// "and" is intentionally omitted (misaki's `extend_num` strips it).
const ONES: &[&str] = &[
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const TENS: &[&str] = &[
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// Convert an integer to English cardinal words: 42 -> "forty two"
pub fn int_to_words(n: i64) -> String {
    if n == 0 {
        return "zero".to_string();
    }
    if n < 0 {
        return format!("minus {}", int_to_words(-n));
    }
    let mut parts = Vec::new();
    convert_chunk(n as u64, &mut parts);
    parts.join(" ")
}

/// Recursively convert a positive number into word parts.
fn convert_chunk(n: u64, parts: &mut Vec<String>) {
    if n == 0 {
        return;
    }

    const SCALES: &[(u64, &str)] = &[
        (1_000_000_000_000, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
        (100, "hundred"),
    ];

    for &(divisor, name) in SCALES {
        if n >= divisor {
            let high = n / divisor;
            let low = n % divisor;
            convert_chunk(high, parts);
            parts.push(name.to_string());
            convert_chunk(low, parts);
            return;
        }
    }

    // n is 1..99
    if n >= 20 {
        let t = n / 10;
        let o = n % 10;
        parts.push(TENS[t as usize].to_string());
        if o > 0 {
            parts.push(ONES[o as usize].to_string());
        }
    } else {
        // 1..19
        parts.push(ONES[n as usize].to_string());
    }
}

/// Convert an integer to English ordinal words: 42 -> "forty second"
pub fn int_to_ordinal(n: i64) -> String {
    let cardinal = int_to_words(n);

    // Split into words; only the last word gets the ordinal treatment.
    let mut words: Vec<&str> = cardinal.split_whitespace().collect();
    if words.is_empty() {
        return cardinal;
    }

    let last = words.pop().unwrap();
    let ordinal_last = make_ordinal(last);

    if words.is_empty() {
        ordinal_last
    } else {
        format!("{} {}", words.join(" "), ordinal_last)
    }
}

/// Apply ordinal suffix to a single cardinal word.
fn make_ordinal(word: &str) -> String {
    // Irregular ordinals
    match word {
        "one" => return "first".to_string(),
        "two" => return "second".to_string(),
        "three" => return "third".to_string(),
        "five" => return "fifth".to_string(),
        "eight" => return "eighth".to_string(),
        "nine" => return "ninth".to_string(),
        "twelve" => return "twelfth".to_string(),
        _ => {}
    }

    // Words ending in "y" -> "ieth"
    if let Some(stem) = word.strip_suffix('y') {
        return format!("{stem}ieth");
    }

    // Regular: append "th"
    format!("{word}th")
}

/// Convert an integer to English year words: 1984 -> "nineteen eighty four"
pub fn int_to_year(n: i64) -> String {
    // Only handle 4-digit positive years specially
    if !(1000..=9999).contains(&n) {
        return int_to_words(n);
    }

    let century = n / 100; // e.g. 19 for 1984
    let remainder = n % 100; // e.g. 84 for 1984

    if remainder == 0 {
        // 2000 -> "two thousand", 1900 -> "nineteen hundred"
        if century % 10 == 0 {
            // e.g. 2000: century=20 -> "twenty" hundreds = "two thousand"
            let thousands = n / 1000;
            return format!("{} thousand", int_to_words(thousands));
        } else {
            // e.g. 1900: "nineteen hundred"
            return format!("{} hundred", int_to_words(century));
        }
    }

    // Century is a multiple of 10 (e.g. 2001, 2024 where century=20)
    if century % 10 == 0 {
        let thousands = n / 1000;
        if remainder < 10 {
            // 2001 -> "twenty oh one"
            return format!(
                "{} oh {}",
                int_to_words(thousands * 10),
                int_to_words(remainder)
            );
        } else {
            // 2024 -> "twenty twenty four"
            return format!(
                "{} {}",
                int_to_words(thousands * 10),
                int_to_words(remainder)
            );
        }
    }

    // General 4-digit: split into two halves
    // 1984 -> "nineteen" + "eighty four"
    if remainder < 10 {
        // 1906 -> "nineteen oh six"
        format!("{} oh {}", int_to_words(century), int_to_words(remainder))
    } else {
        format!("{} {}", int_to_words(century), int_to_words(remainder))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Cardinal tests ---

    #[test]
    fn test_zero() {
        assert_eq!(int_to_words(0), "zero");
    }

    #[test]
    fn test_one() {
        assert_eq!(int_to_words(1), "one");
    }

    #[test]
    fn test_thirteen() {
        assert_eq!(int_to_words(13), "thirteen");
    }

    #[test]
    fn test_twenty() {
        assert_eq!(int_to_words(20), "twenty");
    }

    #[test]
    fn test_forty_two() {
        assert_eq!(int_to_words(42), "forty two");
    }

    #[test]
    fn test_hundred() {
        assert_eq!(int_to_words(100), "one hundred");
    }

    #[test]
    fn test_hundred_one() {
        assert_eq!(int_to_words(101), "one hundred one");
    }

    #[test]
    fn test_thousand() {
        assert_eq!(int_to_words(1000), "one thousand");
    }

    #[test]
    fn test_thousand_one() {
        assert_eq!(int_to_words(1001), "one thousand one");
    }

    #[test]
    fn test_million() {
        assert_eq!(int_to_words(1_000_000), "one million");
    }

    #[test]
    fn test_negative() {
        assert_eq!(int_to_words(-42), "minus forty two");
    }

    #[test]
    fn test_negative_one() {
        assert_eq!(int_to_words(-1), "minus one");
    }

    #[test]
    fn test_large_number() {
        assert_eq!(
            int_to_words(1_234_567),
            "one million two hundred thirty four thousand five hundred sixty seven"
        );
    }

    // --- Ordinal tests ---

    #[test]
    fn test_ordinal_first() {
        assert_eq!(int_to_ordinal(1), "first");
    }

    #[test]
    fn test_ordinal_second() {
        assert_eq!(int_to_ordinal(2), "second");
    }

    #[test]
    fn test_ordinal_third() {
        assert_eq!(int_to_ordinal(3), "third");
    }

    #[test]
    fn test_ordinal_fourth() {
        assert_eq!(int_to_ordinal(4), "fourth");
    }

    #[test]
    fn test_ordinal_fifth() {
        assert_eq!(int_to_ordinal(5), "fifth");
    }

    #[test]
    fn test_ordinal_eighth() {
        assert_eq!(int_to_ordinal(8), "eighth");
    }

    #[test]
    fn test_ordinal_ninth() {
        assert_eq!(int_to_ordinal(9), "ninth");
    }

    #[test]
    fn test_ordinal_twelfth() {
        assert_eq!(int_to_ordinal(12), "twelfth");
    }

    #[test]
    fn test_ordinal_twenty_first() {
        assert_eq!(int_to_ordinal(21), "twenty first");
    }

    #[test]
    fn test_ordinal_forty_second() {
        assert_eq!(int_to_ordinal(42), "forty second");
    }

    #[test]
    fn test_ordinal_hundredth() {
        assert_eq!(int_to_ordinal(100), "one hundredth");
    }

    #[test]
    fn test_ordinal_twentieth() {
        assert_eq!(int_to_ordinal(20), "twentieth");
    }

    #[test]
    fn test_ordinal_thirtieth() {
        assert_eq!(int_to_ordinal(30), "thirtieth");
    }

    // --- Year tests ---

    #[test]
    fn test_year_1984() {
        assert_eq!(int_to_year(1984), "nineteen eighty four");
    }

    #[test]
    fn test_year_2000() {
        assert_eq!(int_to_year(2000), "two thousand");
    }

    #[test]
    fn test_year_2001() {
        assert_eq!(int_to_year(2001), "twenty oh one");
    }

    #[test]
    fn test_year_2024() {
        assert_eq!(int_to_year(2024), "twenty twenty four");
    }

    #[test]
    fn test_year_1900() {
        assert_eq!(int_to_year(1900), "nineteen hundred");
    }

    #[test]
    fn test_year_1066() {
        assert_eq!(int_to_year(1066), "ten sixty six");
    }

    #[test]
    fn test_year_1906() {
        assert_eq!(int_to_year(1906), "nineteen oh six");
    }

    #[test]
    fn test_year_small() {
        // Below 1000, falls back to int_to_words
        assert_eq!(int_to_year(999), "nine hundred ninety nine");
    }
}
