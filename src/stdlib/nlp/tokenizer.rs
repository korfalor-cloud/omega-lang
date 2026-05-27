/// Text tokenization, stemming, and N-gram generation.

use std::collections::HashMap;

/// Tokenizer with configurable options.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    lowercase: bool,
    strip_punctuation: bool,
    min_token_length: usize,
    max_token_length: usize,
    stop_words: Vec<String>,
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            lowercase: true,
            strip_punctuation: true,
            min_token_length: 1,
            max_token_length: usize::MAX,
            stop_words: Vec::new(),
        }
    }

    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    pub fn with_stop_words(mut self, words: &[&str]) -> Self {
        self.stop_words = words.iter().map(|s| s.to_lowercase()).collect();
        self
    }

    pub fn with_min_length(mut self, len: usize) -> Self {
        self.min_token_length = len;
        self
    }

    pub fn with_max_length(mut self, len: usize) -> Self {
        self.max_token_length = len;
        self
    }

    /// Tokenize text into words.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let processed = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let tokens: Vec<String> = processed
            .split(|c: char| c.is_whitespace() || (self.strip_punctuation && c.is_ascii_punctuation()))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .filter(|s| s.len() >= self.min_token_length && s.len() <= self.max_token_length)
            .filter(|s| !self.stop_words.contains(s))
            .collect();

        tokens
    }

    /// Tokenize into sentences.
    pub fn tokenize_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);
            if ch == '.' || ch == '!' || ch == '?' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }

        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }

        sentences
    }

    /// Generate character N-grams.
    pub fn char_ngrams(&self, text: &str, n: usize) -> Vec<String> {
        if n == 0 || text.len() < n {
            return Vec::new();
        }
        let chars: Vec<char> = text.chars().collect();
        (0..=chars.len() - n)
            .map(|i| chars[i..i + n].iter().collect())
            .collect()
    }

    /// Generate word N-grams.
    pub fn word_ngrams(&self, text: &str, n: usize) -> Vec<Vec<String>> {
        let tokens = self.tokenize(text);
        if n == 0 || tokens.len() < n {
            return Vec::new();
        }
        (0..=tokens.len() - n)
            .map(|i| tokens[i..i + n].to_vec())
            .collect()
    }

    pub fn default_stop_words() -> Vec<&'static str> {
        vec![
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "or", "if", "while", "that", "this", "which", "who",
            "whom", "what", "it", "its", "i", "me", "my", "we", "our", "you",
            "your", "he", "him", "his", "she", "her", "they", "them", "their",
        ]
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Porter Stemmer (simplified English stemmer).
pub struct PorterStemmer;

impl PorterStemmer {
    pub fn stem(word: &str) -> String {
        let word = word.to_lowercase();
        if word.len() <= 2 {
            return word;
        }

        let mut result = word.clone();

        // Step 1a
        if result.ends_with("sses") {
            result.truncate(result.len() - 2);
        } else if result.ends_with("ies") {
            result.truncate(result.len() - 2);
        } else if !result.ends_with("ss") && result.ends_with('s') {
            result.truncate(result.len() - 1);
        }

        // Step 1b
        if result.ends_with("eed") {
            if Self::measure(&result[..result.len() - 3]) > 0 {
                result.truncate(result.len() - 1);
            }
        } else if result.ends_with("ed") && Self::contains_vowel(&result[..result.len() - 2]) {
            result.truncate(result.len() - 2);
            result = Self::step1b_post(&result);
        } else if result.ends_with("ing") && Self::contains_vowel(&result[..result.len() - 3]) {
            result.truncate(result.len() - 3);
            result = Self::step1b_post(&result);
        }

        // Step 1c
        if result.ends_with('y') && Self::contains_vowel(&result[..result.len() - 1]) {
            result.truncate(result.len() - 1);
            result.push('i');
        }

        // Step 2
        let step2_pairs = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
            ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
            ("alli", "al"), ("entli", "ent"), ("eli", "e"),
            ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
            ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
            ("iviti", "ive"), ("biliti", "ble"),
        ];
        for (suffix, replacement) in &step2_pairs {
            if result.ends_with(suffix) {
                let stem = &result[..result.len() - suffix.len()];
                if Self::measure(stem) > 0 {
                    result = format!("{}{}", stem, replacement);
                }
                break;
            }
        }

        // Step 3
        let step3_pairs = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"),
            ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
        ];
        for (suffix, replacement) in &step3_pairs {
            if result.ends_with(suffix) {
                let stem = &result[..result.len() - suffix.len()];
                if Self::measure(stem) > 0 {
                    result = format!("{}{}", stem, replacement);
                }
                break;
            }
        }

        // Step 4
        let step4_suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible",
            "ant", "ement", "ment", "ent", "ion", "ou", "ism",
            "ate", "iti", "ous", "ive", "ize",
        ];
        for suffix in &step4_suffixes {
            if result.ends_with(suffix) {
                let stem = &result[..result.len() - suffix.len()];
                if *suffix == "ion" {
                    if Self::measure(stem) > 1 && (stem.ends_with('s') || stem.ends_with('t')) {
                        result = stem.to_string();
                    }
                } else if Self::measure(stem) > 1 {
                    result = stem.to_string();
                }
                break;
            }
        }

        // Step 5a
        if result.ends_with('e') {
            let stem = &result[..result.len() - 1];
            if Self::measure(stem) > 1 || (Self::measure(stem) == 1 && !Self::ends_with_cvc(stem)) {
                result = stem.to_string();
            }
        }

        // Step 5b
        if result.ends_with('l') && result.ends_with("ll") && Self::measure(&result) > 1 {
            result.truncate(result.len() - 1);
        }

        result
    }

    fn step1b_post(word: &str) -> String {
        if word.ends_with("at") || word.ends_with("bl") || word.ends_with("iz") {
            format!("{}e", word)
        } else if Self::ends_with_double_consonant(word) && !word.ends_with('l')
            && !word.ends_with('s') && !word.ends_with('z')
        {
            word[..word.len() - 1].to_string()
        } else if Self::measure(word) == 1 && Self::ends_with_cvc(word) {
            format!("{}e", word)
        } else {
            word.to_string()
        }
    }

    fn measure(word: &str) -> usize {
        let mut m = 0;
        let mut prev_vowel = false;
        let chars: Vec<char> = word.chars().collect();

        for &ch in &chars {
            let is_vowel = Self::is_vowel_char(ch);
            if is_vowel && !prev_vowel {
                // Start of vowel sequence - don't count
            } else if !is_vowel && prev_vowel {
                m += 1;
            }
            prev_vowel = is_vowel;
        }
        m
    }

    fn contains_vowel(word: &str) -> bool {
        word.chars().any(|c| Self::is_vowel_char(c))
    }

    fn is_vowel_char(c: char) -> bool {
        matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
    }

    fn ends_with_double_consonant(word: &str) -> bool {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() < 2 {
            return false;
        }
        let n = chars.len();
        chars[n - 1] == chars[n - 2] && !Self::is_vowel_char(chars[n - 1])
    }

    fn ends_with_cvc(word: &str) -> bool {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() < 3 {
            return false;
        }
        let n = chars.len();
        !Self::is_vowel_char(chars[n - 1])
            && Self::is_vowel_char(chars[n - 2])
            && !Self::is_vowel_char(chars[n - 3])
            && !matches!(chars[n - 1], 'w' | 'x' | 'y')
    }
}

/// Word frequency counter.
pub struct WordCounter {
    counts: HashMap<String, usize>,
    total: usize,
}

impl WordCounter {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            total: 0,
        }
    }

    pub fn add(&mut self, word: &str) {
        let entry = self.counts.entry(word.to_lowercase()).or_insert(0);
        *entry += 1;
        self.total += 1;
    }

    pub fn add_text(&mut self, text: &str, tokenizer: &Tokenizer) {
        for token in tokenizer.tokenize(text) {
            self.add(&token);
        }
    }

    pub fn count(&self, word: &str) -> usize {
        self.counts.get(&word.to_lowercase()).copied().unwrap_or(0)
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn unique_words(&self) -> usize {
        self.counts.len()
    }

    pub fn most_common(&self, n: usize) -> Vec<(&str, usize)> {
        let mut pairs: Vec<(&str, usize)> = self.counts.iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.into_iter().take(n).collect()
    }

    pub fn vocabulary(&self) -> Vec<&str> {
        self.counts.keys().map(|s| s.as_str()).collect()
    }

    pub fn frequency(&self, word: &str) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.count(word) as f64 / self.total as f64
    }
}

impl Default for WordCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Levenshtein edit distance between two strings.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n]
}

/// Jaro similarity between two strings.
pub fn jaro_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 || b_len == 0 {
        return 0.0;
    }

    let match_distance = a_len.max(b_len) / 2 - 1;
    let mut a_matches = vec![false; a_len];
    let mut b_matches = vec![false; b_len];
    let mut matches = 0;
    let mut transpositions = 0;

    for i in 0..a_len {
        let start = if i >= match_distance { i - match_distance } else { 0 };
        let end = (i + match_distance + 1).min(b_len);

        for j in start..end {
            if b_matches[j] || a_chars[i] != b_chars[j] {
                continue;
            }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    let mut k = 0;
    for i in 0..a_len {
        if !a_matches[i] {
            continue;
        }
        while !b_matches[k] {
            k += 1;
        }
        if a_chars[i] != b_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    (m / a_len as f64 + m / b_len as f64 + (m - transpositions as f64 / 2.0) / m) / 3.0
}

/// Jaro-Winkler similarity.
pub fn jaro_winkler_similarity(a: &str, b: &str) -> f64 {
    let jaro = jaro_similarity(a, b);
    let prefix = a.chars().zip(b.chars()).take(4).take_while(|(a, b)| a == b).count();
    jaro + prefix as f64 * 0.1 * (1.0 - jaro)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let tok = Tokenizer::new();
        let tokens = tok.tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_ngrams() {
        let tok = Tokenizer::new();
        let bigrams = tok.word_ngrams("the cat sat on the mat", 2);
        assert!(!bigrams.is_empty());
    }

    #[test]
    fn test_stemmer() {
        assert_eq!(PorterStemmer::stem("running"), "run");
        assert_eq!(PorterStemmer::stem("cats"), "cat");
        assert_eq!(PorterStemmer::stem("generously"), "generous");
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
    }

    #[test]
    fn test_jaro() {
        assert!(jaro_similarity("abc", "abc") > 0.99);
        assert!(jaro_similarity("abc", "xyz") < 0.5);
    }

    #[test]
    fn test_word_counter() {
        let mut counter = WordCounter::new();
        let tok = Tokenizer::new();
        counter.add_text("the cat sat on the mat", &tok);
        assert_eq!(counter.count("the"), 2);
        assert_eq!(counter.total(), 6);
    }
}
