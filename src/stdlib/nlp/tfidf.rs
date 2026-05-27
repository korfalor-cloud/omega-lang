/// TF-IDF (Term Frequency-Inverse Document Frequency) implementation.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TfIdf {
    documents: Vec<Document>,
    vocabulary: HashMap<String, usize>,
    idf: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct Document {
    pub terms: HashMap<String, usize>,
    pub total_terms: usize,
    pub length: usize,
}

#[derive(Debug, Clone)]
pub struct TfIdfScore {
    pub term: String,
    pub tf: f64,
    pub idf: f64,
    pub tfidf: f64,
}

impl TfIdf {
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
            vocabulary: HashMap::new(),
            idf: HashMap::new(),
        }
    }

    /// Add a document (pre-tokenized).
    pub fn add_document(&mut self, tokens: &[String]) {
        let mut terms = HashMap::new();
        for token in tokens {
            let entry = terms.entry(token.clone()).or_insert(0);
            *entry += 1;
            self.vocabulary.entry(token.clone()).or_insert(self.vocabulary.len());
        }
        let total = tokens.len();
        self.documents.push(Document {
            terms,
            total_terms: total,
            length: total,
        });
        self.recompute_idf();
    }

    /// Add multiple documents.
    pub fn add_documents(&mut self, documents: &[Vec<String>]) {
        for doc in documents {
            self.add_document(doc);
        }
    }

    fn recompute_idf(&mut self) {
        let n = self.documents.len() as f64;
        self.idf.clear();

        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        for doc in &self.documents {
            for term in doc.terms.keys() {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
        }

        for (term, df) in &doc_freq {
            // Smoothed IDF
            let idf_val = (1.0 + n / (*df as f64 + 1.0)).ln() + 1.0;
            self.idf.insert(term.clone(), idf_val);
        }
    }

    /// Term frequency of a term in a document.
    pub fn tf(&self, doc_index: usize, term: &str) -> f64 {
        if doc_index >= self.documents.len() {
            return 0.0;
        }
        let doc = &self.documents[doc_index];
        let count = doc.terms.get(term).copied().unwrap_or(0);
        if doc.total_terms == 0 {
            0.0
        } else {
            count as f64 / doc.total_terms as f64
        }
    }

    /// Log-normalized term frequency.
    pub fn tf_log(&self, doc_index: usize, term: &str) -> f64 {
        if doc_index >= self.documents.len() {
            return 0.0;
        }
        let doc = &self.documents[doc_index];
        let count = doc.terms.get(term).copied().unwrap_or(0) as f64;
        if count > 0.0 {
            1.0 + count.ln()
        } else {
            0.0
        }
    }

    /// Double-normalized TF (K normalization, k=0.5).
    pub fn tf_k_norm(&self, doc_index: usize, term: &str, k: f64) -> f64 {
        if doc_index >= self.documents.len() {
            return 0.0;
        }
        let doc = &self.documents[doc_index];
        let count = doc.terms.get(term).copied().unwrap_or(0) as f64;
        let max_freq = doc.terms.values().max().copied().unwrap_or(0) as f64;
        if max_freq == 0.0 {
            0.0
        } else {
            k + (1.0 - k) * count / max_freq
        }
    }

    /// Inverse document frequency of a term.
    pub fn idf(&self, term: &str) -> f64 {
        self.idf.get(term).copied().unwrap_or(0.0)
    }

    /// TF-IDF score for a term in a document.
    pub fn tfidf(&self, doc_index: usize, term: &str) -> f64 {
        self.tf(doc_index, term) * self.idf(term)
    }

    /// All TF-IDF scores for a document, sorted by score descending.
    pub fn document_scores(&self, doc_index: usize) -> Vec<TfIdfScore> {
        if doc_index >= self.documents.len() {
            return Vec::new();
        }

        let doc = &self.documents[doc_index];
        let mut scores: Vec<TfIdfScore> = doc.terms.keys()
            .map(|term| {
                let tf = self.tf(doc_index, term);
                let idf = self.idf(term);
                TfIdfScore {
                    term: term.clone(),
                    tf,
                    idf,
                    tfidf: tf * idf,
                }
            })
            .collect();

        scores.sort_by(|a, b| b.tfidf.partial_cmp(&a.tfidf).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Top N keywords for a document.
    pub fn keywords(&self, doc_index: usize, n: usize) -> Vec<(String, f64)> {
        self.document_scores(doc_index)
            .into_iter()
            .take(n)
            .map(|s| (s.term, s.tfidf))
            .collect()
    }

    /// Cosine similarity between two documents.
    pub fn cosine_similarity(&self, doc_a: usize, doc_b: usize) -> f64 {
        if doc_a >= self.documents.len() || doc_b >= self.documents.len() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for term in self.vocabulary.keys() {
            let tfidf_a = self.tfidf(doc_a, term);
            let tfidf_b = self.tfidf(doc_b, term);
            dot_product += tfidf_a * tfidf_b;
            norm_a += tfidf_a * tfidf_a;
            norm_b += tfidf_b * tfidf_b;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            dot_product / denom
        }
    }

    /// Similarity matrix for all documents.
    pub fn similarity_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.documents.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let sim = self.cosine_similarity(i, j);
                matrix[i][j] = sim;
                matrix[j][i] = sim;
            }
        }

        matrix
    }

    /// Query a document against the corpus.
    pub fn query(&self, query_terms: &[String]) -> Vec<(usize, f64)> {
        let mut scores: Vec<(usize, f64)> = self.documents.iter().enumerate()
            .map(|(i, _)| {
                let score: f64 = query_terms.iter()
                    .map(|term| self.tfidf(i, term))
                    .sum();
                (i, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn vocabulary(&self) -> Vec<&str> {
        self.vocabulary.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for TfIdf {
    fn default() -> Self {
        Self::new()
    }
}

/// BM25 scoring.
pub struct BM25 {
    documents: Vec<Document>,
    avg_doc_length: f64,
    k1: f64,
    b: f64,
    idf: HashMap<String, f64>,
}

impl BM25 {
    pub fn new(k1: f64, b: f64) -> Self {
        Self {
            documents: Vec::new(),
            avg_doc_length: 0.0,
            k1,
            b,
            idf: HashMap::new(),
        }
    }

    pub fn add_document(&mut self, tokens: &[String]) {
        let mut terms = HashMap::new();
        for token in tokens {
            *terms.entry(token.clone()).or_insert(0) += 1;
        }
        let total = tokens.len();
        self.documents.push(Document {
            terms,
            total_terms: total,
            length: total,
        });
        self.recompute();
    }

    fn recompute(&mut self) {
        let n = self.documents.len() as f64;
        if n == 0.0 {
            return;
        }

        let total_len: usize = self.documents.iter().map(|d| d.length).sum();
        self.avg_doc_length = total_len as f64 / n;

        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        for doc in &self.documents {
            for term in doc.terms.keys() {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
        }

        self.idf.clear();
        for (term, df) in &doc_freq {
            let idf_val = ((n - *df as f64 + 0.5) / (*df as f64 + 0.5) + 1.0).ln();
            self.idf.insert(term.clone(), idf_val);
        }
    }

    /// Score a document against a query.
    pub fn score(&self, doc_index: usize, query_terms: &[String]) -> f64 {
        if doc_index >= self.documents.len() {
            return 0.0;
        }
        let doc = &self.documents[doc_index];

        query_terms.iter().map(|term| {
            let tf = doc.terms.get(term).copied().unwrap_or(0) as f64;
            let idf = self.idf.get(term).copied().unwrap_or(0.0);
            let numerator = tf * (self.k1 + 1.0);
            let denominator = tf + self.k1 * (1.0 - self.b + self.b * doc.length as f64 / self.avg_doc_length);
            idf * numerator / denominator
        }).sum()
    }

    /// Rank documents by BM25 score for a query.
    pub fn query(&self, query_terms: &[String]) -> Vec<(usize, f64)> {
        let mut scores: Vec<(usize, f64)> = (0..self.documents.len())
            .map(|i| (i, self.score(i, query_terms)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(s: &str) -> Vec<String> {
        s.split_whitespace().map(|s| s.to_lowercase().to_string()).collect()
    }

    #[test]
    fn test_tfidf_basic() {
        let mut tfidf = TfIdf::new();
        tfidf.add_document(&tokens("the cat sat on the mat"));
        tfidf.add_document(&tokens("the dog chased the cat"));
        tfidf.add_document(&tokens("the bird flew over the tree"));

        assert!(tfidf.tfidf(0, "cat") > 0.0);
        assert!(tfidf.tfidf(0, "mat") > 0.0);
    }

    #[test]
    fn test_keywords() {
        let mut tfidf = TfIdf::new();
        tfidf.add_document(&tokens("machine learning is a subset of artificial intelligence"));
        let keywords = tfidf.keywords(0, 3);
        assert_eq!(keywords.len(), 3);
    }

    #[test]
    fn test_bm25() {
        let mut bm25 = BM25::new(1.2, 0.75);
        bm25.add_document(&tokens("the cat sat on the mat"));
        bm25.add_document(&tokens("the dog chased the cat"));

        let results = bm25.query(&tokens("cat"));
        assert!(!results.is_empty());
    }
}
