/// Tokenization: BPE, WordPiece, SentencePiece-style tokenizers.

use std::collections::HashMap;

/// Byte Pair Encoding (BPE) tokenizer.
pub struct BPETokenizer {
    pub vocab: HashMap<String, usize>,
    pub inv_vocab: HashMap<usize, String>,
    pub merges: Vec<(String, String)>,
    pub vocab_size: usize,
}

impl BPETokenizer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            inv_vocab: HashMap::new(),
            merges: Vec::new(),
            vocab_size: 0,
        }
    }

    /// Train BPE on corpus.
    pub fn train(&mut self, corpus: &[String], num_merges: usize) {
        // Initialize with character vocabulary
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();

        for text in corpus {
            let words: Vec<String> = text.split_whitespace().map(|w| w.to_string()).collect();
            for word in words {
                let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                *word_freqs.entry(chars).or_insert(0) += 1;
            }
        }

        // Create initial vocabulary
        let mut all_chars: Vec<String> = word_freqs.values()
            .flat_map(|word| word.iter())
            .cloned()
            .collect();
        all_chars.sort();
        all_chars.dedup();

        for (i, ch) in all_chars.iter().enumerate() {
            self.vocab.insert(ch.clone(), i);
            self.inv_vocab.insert(i, ch.clone());
        }
        self.vocab_size = all_chars.len();

        // Merge iterations
        for _ in 0..num_merges {
            // Count pairs
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for (word, &freq) in &word_freqs {
                for i in 0..word.len() - 1 {
                    *pair_counts.entry((word[i].clone(), word[i + 1].clone())).or_insert(0) += freq;
                }
            }

            // Find most frequent pair
            let best_pair = pair_counts.iter()
                .max_by_key(|(_, &count)| count)
                .map(|(pair, _)| pair.clone());

            if let Some((a, b)) = best_pair {
                let merged = format!("{}{}", a, b);
                self.merges.push((a.clone(), b.clone()));

                if !self.vocab.contains_key(&merged) {
                    self.vocab.insert(merged.clone(), self.vocab_size);
                    self.inv_vocab.insert(self.vocab_size, merged.clone());
                    self.vocab_size += 1;
                }

                // Update word frequencies
                let mut new_word_freqs = HashMap::new();
                for (word, &freq) in &word_freqs {
                    let mut new_word = Vec::new();
                    let mut i = 0;
                    while i < word.len() {
                        if i < word.len() - 1 && word[i] == a && word[i + 1] == b {
                            new_word.push(merged.clone());
                            i += 2;
                        } else {
                            new_word.push(word[i].clone());
                            i += 1;
                        }
                    }
                    *new_word_freqs.entry(new_word).or_insert(0) += freq;
                }
                word_freqs = new_word_freqs;
            } else {
                break;
            }
        }
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            // Apply merges
            for (a, b) in &self.merges {
                let mut new_chars = Vec::new();
                let mut i = 0;
                while i < chars.len() {
                    if i < chars.len() - 1 && chars[i] == *a && chars[i + 1] == *b {
                        new_chars.push(format!("{}{}", a, b));
                        i += 2;
                    } else {
                        new_chars.push(chars[i].clone());
                        i += 1;
                    }
                }
                chars = new_chars;
            }

            for ch in chars {
                if let Some(&id) = self.vocab.get(&ch) {
                    tokens.push(id);
                }
            }
        }

        tokens
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|id| self.inv_vocab.get(id))
            .cloned()
            .collect::<Vec<String>>()
            .join("")
    }
}

/// WordPiece tokenizer (used in BERT).
pub struct WordPieceTokenizer {
    pub vocab: HashMap<String, usize>,
    pub inv_vocab: HashMap<usize, String>,
    pub unk_token: String,
    pub prefix: String,
}

impl WordPieceTokenizer {
    pub fn new(vocab: HashMap<String, usize>, unk_token: &str, prefix: &str) -> Self {
        let inv_vocab: HashMap<usize, String> = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Self {
            vocab,
            inv_vocab,
            unk_token: unk_token.to_string(),
            prefix: prefix.to_string(),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            let chars: Vec<char> = word.chars().collect();
            let mut start = 0;
            let mut word_tokens = Vec::new();

            while start < chars.len() {
                let mut end = chars.len();
                let mut found = false;

                while start < end {
                    let mut subword: String = chars[start..end].iter().collect();
                    if start > 0 {
                        subword = format!("{}{}", self.prefix, subword);
                    }

                    if self.vocab.contains_key(&subword) {
                        word_tokens.push(self.vocab[&subword]);
                        found = true;
                        break;
                    }
                    end -= 1;
                }

                if !found {
                    word_tokens.push(*self.vocab.get(&self.unk_token).unwrap_or(&0));
                    start += 1;
                } else {
                    start = end;
                }
            }

            tokens.extend(word_tokens);
        }

        tokens
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|id| self.inv_vocab.get(id))
            .map(|s| {
                if s.starts_with(&self.prefix) {
                    s[self.prefix.len()..].to_string()
                } else {
                    format!(" {}", s)
                }
            })
            .collect()
    }
}

/// Character-level tokenizer.
pub struct CharTokenizer {
    pub char_to_id: HashMap<char, usize>,
    pub id_to_char: HashMap<usize, char>,
    pub vocab_size: usize,
}

impl CharTokenizer {
    pub fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        for (i, &ch) in chars.iter().enumerate() {
            char_to_id.insert(ch, i);
            id_to_char.insert(i, ch);
        }

        Self { char_to_id, id_to_char, vocab_size: chars.len() }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_id.get(&c).copied())
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_char.get(&id))
            .collect()
    }
}

/// N-gram language model for tokenization.
pub struct NGramTokenizer {
    pub n: usize,
    pub vocab: HashMap<String, usize>,
    pub inv_vocab: HashMap<usize, String>,
}

impl NGramTokenizer {
    pub fn new(n: usize) -> Self {
        Self { n, vocab: HashMap::new(), inv_vocab: HashMap::new() }
    }

    pub fn build_vocab(&mut self, text: &str, min_count: usize) {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let chars: Vec<char> = text.chars().collect();

        for window in chars.windows(self.n) {
            let ngram: String = window.iter().collect();
            *counts.entry(ngram).or_insert(0) += 1;
        }

        let mut id = 0;
        for (ngram, count) in counts {
            if count >= min_count {
                self.vocab.insert(ngram.clone(), id);
                self.inv_vocab.insert(id, ngram);
                id += 1;
            }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let chars: Vec<char> = text.chars().collect();
        chars.windows(self.n)
            .filter_map(|window| {
                let ngram: String = window.iter().collect();
                self.vocab.get(&ngram).copied()
            })
            .collect()
    }
}

/// Unigram tokenizer (SentencePiece style).
pub struct UnigramTokenizer {
    pub vocab: HashMap<String, f64>, // token -> log probability
    pub vocab_size: usize,
}

impl UnigramTokenizer {
    pub fn new(vocab: HashMap<String, f64>) -> Self {
        let vocab_size = vocab.len();
        Self { vocab, vocab_size }
    }

    /// Viterbi segmentation.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let n = text.len();
        let chars: Vec<char> = text.chars().collect();

        // Dynamic programming
        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut best_prev = vec![0usize; n + 1];
        best_score[0] = 0.0;

        for i in 0..n {
            for j in i + 1..=n {
                let substring: String = chars[i..j].iter().collect();
                if let Some(&score) = self.vocab.get(&substring) {
                    let new_score = best_score[i] + score;
                    if new_score > best_score[j] {
                        best_score[j] = new_score;
                        best_prev[j] = i;
                    }
                }
            }
        }

        // Backtrack
        let mut tokens = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let start = best_prev[pos];
            let token: String = chars[start..pos].iter().collect();
            tokens.push(self.vocab_id(&token));
            pos = start;
        }

        tokens.reverse();
        tokens
    }

    fn vocab_id(&self, token: &str) -> usize {
        self.vocab.get(token).map(|_| {
            // Simplified: use hash as ID
            let mut hash = 0usize;
            for ch in token.chars() {
                hash = hash.wrapping_mul(31).wrapping_add(ch as usize);
            }
            hash % self.vocab_size
        }).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe() {
        let mut tokenizer = BPETokenizer::new();
        let corpus = vec![
            "low lower newest widest".to_string(),
            "low low low low".to_string(),
        ];
        tokenizer.train(&corpus, 5);

        let encoded = tokenizer.encode("low");
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_char_tokenizer() {
        let tokenizer = CharTokenizer::new("hello world");
        let encoded = tokenizer.encode("hello");
        assert_eq!(encoded.len(), 5);

        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_wordpiece() {
        let mut vocab = HashMap::new();
        vocab.insert("low".to_string(), 0);
        vocab.insert("##er".to_string(), 1);
        vocab.insert("##est".to_string(), 2);
        vocab.insert("[UNK]".to_string(), 3);

        let tokenizer = WordPieceTokenizer::new(vocab, "[UNK]", "##");
        let encoded = tokenizer.encode("lower");
        assert!(!encoded.is_empty());
    }
}
