/// Sentiment analysis using lexicon-based approach and n-gram features.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SentimentAnalyzer {
    positive_words: HashMap<String, f64>,
    negative_words: HashMap<String, f64>,
    intensifiers: HashMap<String, f64>,
    negators: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub score: f64,
    pub magnitude: f64,
    pub label: SentimentLabel,
    pub positive_count: usize,
    pub negative_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SentimentLabel {
    VeryPositive,
    Positive,
    Neutral,
    Negative,
    VeryNegative,
}

impl SentimentAnalyzer {
    pub fn new() -> Self {
        let mut positive = HashMap::new();
        let mut negative = HashMap::new();
        let mut intensifiers = HashMap::new();

        // Positive words with scores
        let pos_words = [
            ("good", 0.6), ("great", 0.8), ("excellent", 0.9), ("amazing", 0.9),
            ("wonderful", 0.85), ("fantastic", 0.85), ("love", 0.8), ("happy", 0.7),
            ("joy", 0.8), ("beautiful", 0.7), ("awesome", 0.8), ("brilliant", 0.8),
            ("perfect", 0.9), ("nice", 0.5), ("pleasant", 0.6), ("enjoy", 0.7),
            ("delightful", 0.8), ("superb", 0.85), ("magnificent", 0.9), ("glorious", 0.85),
            ("outstanding", 0.9), ("remarkable", 0.8), ("exceptional", 0.85), ("impressive", 0.75),
            ("best", 0.9), ("better", 0.6), ("improve", 0.5), ("improved", 0.6),
            ("improving", 0.6), ("positive", 0.5), ("profit", 0.6), ("gain", 0.6),
            ("success", 0.7), ("successful", 0.7), ("win", 0.7), ("winner", 0.7),
            ("recommend", 0.6), ("recommended", 0.7), ("approve", 0.6), ("praise", 0.7),
            ("celebrate", 0.7), ("triumph", 0.8), ("victory", 0.8), ("excel", 0.7),
            ("thrilled", 0.8), ("excited", 0.7), ("eager", 0.6), ("grateful", 0.7),
            ("thankful", 0.7), ("blessed", 0.7), ("proud", 0.7), ("confident", 0.6),
            ("hopeful", 0.6), ("optimistic", 0.6), ("inspire", 0.7), ("inspired", 0.7),
            ("creative", 0.5), ("innovative", 0.6), ("elegant", 0.6), ("graceful", 0.6),
            ("kind", 0.6), ("generous", 0.6), ("gentle", 0.5), ("caring", 0.6),
            ("compassionate", 0.7), ("loyal", 0.6), ("honest", 0.5), ("trustworthy", 0.6),
            ("reliable", 0.5), ("efficient", 0.5), ("effective", 0.5), ("productive", 0.5),
        ];
        for (word, score) in &pos_words {
            positive.insert(word.to_string(), *score);
        }

        // Negative words with scores
        let neg_words = [
            ("bad", -0.6), ("terrible", -0.9), ("horrible", -0.9), ("awful", -0.85),
            ("dreadful", -0.85), ("hate", -0.8), ("angry", -0.7), ("sad", -0.6),
            ("depressed", -0.8), ("miserable", -0.8), ("ugly", -0.6), ("disgusting", -0.8),
            ("nasty", -0.7), ("poor", -0.5), ("worst", -0.9), ("worse", -0.7),
            ("disappoint", -0.6), ("disappointed", -0.7), ("failure", -0.8), ("fail", -0.7),
            ("lost", -0.6), ("loss", -0.7), ("lose", -0.7), ("loser", -0.8),
            ("wrong", -0.5), ("error", -0.5), ("mistake", -0.5), ("problem", -0.5),
            ("issue", -0.4), ("bug", -0.4), ("broken", -0.6), ("damage", -0.6),
            ("destroy", -0.8), ("destroyed", -0.8), ("ruin", -0.7), ("ruined", -0.7),
            ("pain", -0.7), ("suffer", -0.7), ("suffering", -0.7), ("sick", -0.6),
            ("ill", -0.5), ("disease", -0.6), ("death", -0.8), ("dead", -0.7),
            ("kill", -0.8), ("killed", -0.8), ("murder", -0.9), ("war", -0.7),
            ("attack", -0.7), ("attacked", -0.7), ("threat", -0.6), ("danger", -0.7),
            ("dangerous", -0.7), ("risk", -0.4), ("risky", -0.5), ("fear", -0.6),
            ("scared", -0.6), ("terrified", -0.8), ("panic", -0.7), ("anxious", -0.5),
            ("anxiety", -0.6), ("stress", -0.5), ("stressed", -0.6), ("worried", -0.5),
            ("concern", -0.4), ("concerned", -0.5), ("trouble", -0.5), ("difficult", -0.4),
            ("hard", -0.3), ("impossible", -0.5), ("never", -0.4), ("refuse", -0.5),
            ("reject", -0.5), ("deny", -0.5), ("complain", -0.5), ("complaint", -0.5),
            ("blame", -0.5), ("guilty", -0.6), ("shame", -0.6), ("embarrass", -0.5),
            ("humiliate", -0.7), ("insult", -0.6), ("offend", -0.5), ("offensive", -0.6),
            ("abuse", -0.8), ("abusive", -0.8), ("cruel", -0.7), ("evil", -0.8),
            ("wicked", -0.7), ("corrupt", -0.7), ("fraud", -0.7), ("scam", -0.8),
            ("cheat", -0.6), ("liar", -0.7), ("lie", -0.6), ("false", -0.5),
            ("fake", -0.5), ("crash", -0.6), ("collapse", -0.7), ("recession", -0.7),
            ("crisis", -0.7), ("panic", -0.7), ("plunge", -0.6), ("dump", -0.5),
        ];
        for (word, score) in &neg_words {
            negative.insert(word.to_string(), *score);
        }

        // Intensifiers
        let int_words = [
            ("very", 1.5), ("extremely", 2.0), ("incredibly", 2.0), ("absolutely", 2.0),
            ("completely", 1.8), ("totally", 1.8), ("really", 1.5), ("quite", 1.3),
            ("rather", 1.2), ("somewhat", 0.8), ("slightly", 0.5), ("barely", 0.3),
            ("hardly", 0.3), ("so", 1.5), ("super", 1.5), ("most", 1.5),
        ];
        for (word, multiplier) in &int_words {
            intensifiers.insert(word.to_string(), *multiplier);
        }

        Self {
            positive_words: positive,
            negative_words: negative,
            intensifiers,
            negators: vec![
                "not", "no", "never", "neither", "nobody", "nothing",
                "nowhere", "nor", "cannot", "can't", "won't", "don't",
                "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
                "hasn't", "haven't", "hadn't", "wouldn't", "shouldn't", "couldn't",
            ].into_iter().map(String::from).collect(),
        }
    }

    /// Analyze sentiment of text.
    pub fn analyze(&self, text: &str) -> SentimentResult {
        let words: Vec<String> = text.split_whitespace()
            .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        let mut score = 0.0;
        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut negated = false;
        let mut intensifier = 1.0;

        for (i, word) in words.iter().enumerate() {
            // Check for negators
            if self.negators.contains(word) {
                negated = true;
                continue;
            }

            // Check for intensifiers
            if let Some(&mult) = self.intensifiers.get(word) {
                intensifier = mult;
                continue;
            }

            // Score the word
            let word_score = if let Some(&s) = self.positive_words.get(word) {
                positive_count += 1;
                s
            } else if let Some(&s) = self.negative_words.get(word) {
                negative_count += 1;
                s
            } else {
                0.0
            };

            if word_score != 0.0 {
                let final_score = if negated {
                    -word_score * intensifier
                } else {
                    word_score * intensifier
                };
                score += final_score;
            }

            // Reset modifiers
            negated = false;
            intensifier = 1.0;
        }

        let magnitude = score.abs();
        let label = if score > 0.5 {
            SentimentLabel::VeryPositive
        } else if score > 0.1 {
            SentimentLabel::Positive
        } else if score < -0.5 {
            SentimentLabel::VeryNegative
        } else if score < -0.1 {
            SentimentLabel::Negative
        } else {
            SentimentLabel::Neutral
        };

        SentimentResult {
            score,
            magnitude,
            label,
            positive_count,
            negative_count,
        }
    }

    /// Analyze sentiment of multiple sentences.
    pub fn analyze_paragraph(&self, text: &str) -> Vec<SentimentResult> {
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        sentences.iter().map(|s| self.analyze(s)).collect()
    }

    /// Average sentiment across sentences.
    pub fn average_sentiment(&self, text: &str) -> f64 {
        let results = self.analyze_paragraph(text);
        if results.is_empty() {
            return 0.0;
        }
        let sum: f64 = results.iter().map(|r| r.score).sum();
        sum / results.len() as f64
    }

    /// Add custom positive word.
    pub fn add_positive(&mut self, word: &str, score: f64) {
        self.positive_words.insert(word.to_lowercase(), score.abs());
    }

    /// Add custom negative word.
    pub fn add_negative(&mut self, word: &str, score: f64) {
        self.negative_words.insert(word.to_lowercase(), -score.abs());
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Naive Bayes text classifier.
pub struct TextClassifier {
    class_word_counts: HashMap<String, HashMap<String, usize>>,
    class_counts: HashMap<String, usize>,
    vocabulary: HashMap<String, usize>,
    total_docs: usize,
}

impl TextClassifier {
    pub fn new() -> Self {
        Self {
            class_word_counts: HashMap::new(),
            class_counts: HashMap::new(),
            vocabulary: HashMap::new(),
            total_docs: 0,
        }
    }

    /// Train on a labeled document.
    pub fn train(&mut self, class: &str, tokens: &[String]) {
        self.total_docs += 1;
        *self.class_counts.entry(class.to_string()).or_insert(0) += 1;

        let word_counts = self.class_word_counts.entry(class.to_string()).or_insert_with(HashMap::new);
        for token in tokens {
            *word_counts.entry(token.clone()).or_insert(0) += 1;
            self.vocabulary.entry(token.clone()).or_insert(self.vocabulary.len());
        }
    }

    /// Predict class for a document.
    pub fn predict(&self, tokens: &[String]) -> Option<(String, f64)> {
        if self.total_docs == 0 {
            return None;
        }

        let mut best_class = String::new();
        let mut best_score = f64::NEG_INFINITY;

        let vocab_size = self.vocabulary.len() as f64;

        for (class, &count) in &self.class_counts {
            let prior = (count as f64 / self.total_docs as f64).ln();
            let word_counts = self.class_word_counts.get(class).unwrap();
            let total_words: usize = word_counts.values().sum();
            let total_words_f = total_words as f64;

            let mut log_likelihood = 0.0;
            for token in tokens {
                let token_count = word_counts.get(token).copied().unwrap_or(0) as f64;
                // Laplace smoothing
                let prob = (token_count + 1.0) / (total_words_f + vocab_size);
                log_likelihood += prob.ln();
            }

            let score = prior + log_likelihood;
            if score > best_score {
                best_score = score;
                best_class = class.clone();
            }
        }

        Some((best_class, best_score))
    }

    pub fn class_count(&self, class: &str) -> usize {
        self.class_counts.get(class).copied().unwrap_or(0)
    }

    pub fn classes(&self) -> Vec<&str> {
        self.class_counts.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for TextClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_sentiment() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("This is a great and wonderful experience!");
        assert!(result.score > 0.0);
        assert_eq!(result.label, SentimentLabel::VeryPositive);
    }

    #[test]
    fn test_negative_sentiment() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("This is terrible and horrible.");
        assert!(result.score < 0.0);
    }

    #[test]
    fn test_negation() {
        let analyzer = SentimentAnalyzer::new();
        let pos = analyzer.analyze("good");
        let neg = analyzer.analyze("not good");
        assert!(pos.score > neg.score);
    }

    #[test]
    fn test_intensifier() {
        let analyzer = SentimentAnalyzer::new();
        let normal = analyzer.analyze("good");
        let intense = analyzer.analyze("very good");
        assert!(intense.score > normal.score);
    }

    #[test]
    fn test_text_classifier() {
        let mut classifier = TextClassifier::new();
        classifier.train("positive", &["good".into(), "great".into(), "excellent".into()]);
        classifier.train("negative", &["bad".into(), "terrible".into(), "awful".into()]);

        let result = classifier.predict(&["good".into(), "great".into()]);
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "positive");
    }
}
