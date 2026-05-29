/// Embeddings: word2vec, GloVe-style, positional encoding.

use std::collections::HashMap;

/// Word2Vec (Skip-gram with negative sampling).
pub struct Word2Vec {
    pub vocab: HashMap<String, usize>,
    pub word_embeddings: Vec<Vec<f64>>,
    pub context_embeddings: Vec<Vec<f64>>,
    pub embedding_dim: usize,
    pub learning_rate: f64,
    seed: u64,
}

impl Word2Vec {
    pub fn new(vocab_size: usize, embedding_dim: usize, learning_rate: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / embedding_dim as f64).sqrt();

        Self {
            vocab: HashMap::new(),
            word_embeddings: (0..vocab_size).map(|_| (0..embedding_dim).map(|_| rand(scale)).collect()).collect(),
            context_embeddings: (0..vocab_size).map(|_| (0..embedding_dim).map(|_| rand(scale)).collect()).collect(),
            embedding_dim, learning_rate, seed,
        }
    }

    pub fn build_vocab(&mut self, corpus: &[String], min_count: usize) {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for text in corpus {
            for word in text.split_whitespace() {
                *counts.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        let mut id = 0;
        for (word, count) in counts {
            if count >= min_count {
                self.vocab.insert(word, id);
                id += 1;
            }
        }
    }

    pub fn train_skipgram(&mut self, corpus: &[String], window_size: usize, n_negative: usize, epochs: usize) {
        for _ in 0..epochs {
            for text in corpus {
                let words: Vec<usize> = text.split_whitespace()
                    .filter_map(|w| self.vocab.get(w).copied())
                    .collect();

                for (i, &center) in words.iter().enumerate() {
                    let start = i.saturating_sub(window_size);
                    let end = (i + window_size + 1).min(words.len());

                    for j in start..end {
                        if i == j { continue; }
                        let context = words[j];

                        // Positive sample
                        self.update_pair(center, context, 1.0);

                        // Negative samples
                        for _ in 0..n_negative {
                            let neg = self.sample_negative();
                            self.update_pair(center, neg, 0.0);
                        }
                    }
                }
            }
        }
    }

    fn update_pair(&mut self, center: usize, context: usize, label: f64) {
        let dot: f64 = self.word_embeddings[center].iter()
            .zip(self.context_embeddings[context].iter())
            .map(|(w, c)| w * c)
            .sum();

        let sigmoid = 1.0 / (1.0 + (-dot).exp());
        let grad = (sigmoid - label) * self.learning_rate;

        for i in 0..self.embedding_dim {
            let w_grad = grad * self.context_embeddings[context][i];
            let c_grad = grad * self.word_embeddings[center][i];
            self.word_embeddings[center][i] -= w_grad;
            self.context_embeddings[context][i] -= c_grad;
        }
    }

    fn sample_negative(&mut self) -> usize {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as usize) % self.word_embeddings.len()
    }

    pub fn get_embedding(&self, word: &str) -> Option<&Vec<f64>> {
        self.vocab.get(word).map(|&id| &self.word_embeddings[id])
    }

    pub fn similarity(&self, word1: &str, word2: &str) -> f64 {
        let emb1 = self.get_embedding(word1);
        let emb2 = self.get_embedding(word2);

        match (emb1, emb2) {
            (Some(e1), Some(e2)) => cosine_similarity(e1, e2),
            _ => 0.0,
        }
    }

    pub fn analogy(&self, a: &str, b: &str, c: &str) -> Option<String> {
        let emb_a = self.get_embedding(a)?;
        let emb_b = self.get_embedding(b)?;
        let emb_c = self.get_embedding(c)?;

        let target: Vec<f64> = emb_a.iter().zip(emb_b.iter()).zip(emb_c.iter())
            .map(|((&a, &b), &c)| b - a + c)
            .collect();

        self.vocab.iter()
            .filter(|(word, _)| *word != a && *word != b && *word != c)
            .max_by(|(_, id1), (_, id2)| {
                let sim1 = cosine_similarity(&target, &self.word_embeddings[**id1]);
                let sim2 = cosine_similarity(&target, &self.word_embeddings[**id2]);
                sim1.partial_cmp(&sim2).unwrap()
            })
            .map(|(word, _)| word.clone())
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a * norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 }
}

/// Sinusoidal positional encoding (Transformer-style).
pub fn sinusoidal_positional_encoding(seq_len: usize, d_model: usize) -> Vec<Vec<f64>> {
    let mut pe = vec![vec![0.0; d_model]; seq_len];

    for pos in 0..seq_len {
        for i in 0..d_model / 2 {
            let angle = pos as f64 / 10000.0_f64.powf(2.0 * i as f64 / d_model as f64);
            pe[pos][2 * i] = angle.sin();
            pe[pos][2 * i + 1] = angle.cos();
        }
    }

    pe
}

/// Learned positional embeddings.
pub struct PositionalEmbedding {
    pub embeddings: Vec<Vec<f64>>,
    pub max_len: usize,
}

impl PositionalEmbedding {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        let mut seed = 42u64;
        let embeddings = (0..max_len).map(|_| {
            (0..d_model).map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.02 - 0.01
            }).collect()
        }).collect();

        Self { embeddings, max_len }
    }

    pub fn get(&self, position: usize) -> &[f64] {
        &self.embeddings[position.min(self.max_len - 1)]
    }
}

/// Rotary Position Embedding (RoPE).
pub fn rotary_embedding(x: &[f64], position: usize, theta: f64) -> Vec<f64> {
    let d = x.len();
    let mut result = vec![0.0; d];

    for i in 0..d / 2 {
        let angle = position as f64 * theta.powf(-(2 * i) as f64 / d as f64);
        let cos = angle.cos();
        let sin = angle.sin();

        result[2 * i] = x[2 * i] * cos - x[2 * i + 1] * sin;
        result[2 * i + 1] = x[2 * i] * sin + x[2 * i + 1] * cos;
    }

    result
}

/// ALiBi (Attention with Linear Biases).
pub fn alibi_bias(seq_len: usize, n_heads: usize) -> Vec<Vec<f64>> {
    let slopes: Vec<f64> = (0..n_heads).map(|h| {
        2.0_f64.powf(-8.0 * (h + 1) as f64 / n_heads as f64)
    }).collect();

    let mut biases = vec![vec![0.0; seq_len]; seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let distance = (i as f64 - j as f64).abs();
            biases[i][j] = -slopes[0] * distance; // Use first head's slope
        }
    }

    biases
}

/// Token embedding layer.
pub struct TokenEmbedding {
    pub embeddings: Vec<Vec<f64>>,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
}

impl TokenEmbedding {
    pub fn new(vocab_size: usize, embedding_dim: usize, padding_idx: Option<usize>) -> Self {
        let mut seed = 42u64;
        let mut embeddings: Vec<Vec<f64>> = (0..vocab_size).map(|_| {
            (0..embedding_dim).map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.1 - 0.05
            }).collect()
        }).collect();

        if let Some(idx) = padding_idx {
            if idx < vocab_size {
                embeddings[idx] = vec![0.0; embedding_dim];
            }
        }

        Self { embeddings, vocab_size, embedding_dim, padding_idx }
    }

    pub fn forward(&self, token_ids: &[usize]) -> Vec<Vec<f64>> {
        token_ids.iter().map(|&id| {
            if id < self.vocab_size {
                self.embeddings[id].clone()
            } else {
                vec![0.0; self.embedding_dim]
            }
        }).collect()
    }

    pub fn get_embedding(&self, token_id: usize) -> &[f64] {
        &self.embeddings[token_id.min(self.vocab_size - 1)]
    }
}

/// Bag of Embeddings.
pub fn bag_of_embeddings(embeddings: &[Vec<f64>]) -> Vec<f64> {
    if embeddings.is_empty() { return Vec::new(); }
    let dim = embeddings[0].len();
    let n = embeddings.len() as f64;

    let mut result = vec![0.0; dim];
    for emb in embeddings {
        for (i, val) in emb.iter().enumerate() {
            result[i] += val;
        }
    }
    for val in result.iter_mut() { *val /= n; }
    result
}

/// TF-IDF weighted embeddings.
pub fn tfidf_weighted_embeddings(embeddings: &[Vec<f64>], tfidf_weights: &[f64]) -> Vec<f64> {
    if embeddings.is_empty() { return Vec::new(); }
    let dim = embeddings[0].len();

    let mut result = vec![0.0; dim];
    let mut total_weight = 0.0;

    for (emb, &weight) in embeddings.iter().zip(tfidf_weights.iter()) {
        for (i, val) in emb.iter().enumerate() {
            result[i] += weight * val;
        }
        total_weight += weight;
    }

    if total_weight > 0.0 {
        for val in result.iter_mut() { *val /= total_weight; }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positional_encoding() {
        let pe = sinusoidal_positional_encoding(10, 8);
        assert_eq!(pe.len(), 10);
        assert_eq!(pe[0].len(), 8);
    }

    #[test]
    fn test_token_embedding() {
        let embedding = TokenEmbedding::new(100, 16, Some(0));
        let tokens = vec![1, 2, 3];
        let embedded = embedding.forward(&tokens);
        assert_eq!(embedded.len(), 3);
        assert_eq!(embedded[0].len(), 16);
        assert_eq!(embedded[0], vec![0.0; 16]); // padding
    }

    #[test]
    fn test_rotary() {
        let x = vec![1.0, 0.0, 1.0, 0.0];
        let result = rotary_embedding(&x, 1, 10000.0);
        assert_eq!(result.len(), 4);
    }
}
