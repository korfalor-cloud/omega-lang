//! Advanced string processing algorithms.
//!
//! Provides multi-pattern matching (Aho-Corasick), edit distance (Levenshtein),
//! longest common subsequence, similarity metrics (Jaro, Jaccard), and suffix
//! array construction with search.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Aho-Corasick multi-pattern matcher
// ---------------------------------------------------------------------------

/// A node in the Aho-Corasick trie.
#[derive(Debug, Default)]
struct AcNode {
    children: HashMap<char, usize>,
    fail: usize,
    output: Vec<usize>, // indices into the pattern list
}

/// Aho-Corasick automaton for searching multiple patterns in a single pass.
#[derive(Debug)]
pub struct AhoCorasick {
    nodes: Vec<AcNode>,
    patterns: Vec<String>,
}

impl AhoCorasick {
    /// Build the automaton from an iterator of pattern strings.
    pub fn new<I, S>(patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let patterns: Vec<String> = patterns.into_iter().map(Into::into).collect();
        let mut nodes = vec![AcNode::default()];

        // Phase 1: build the trie
        for (idx, pat) in patterns.iter().enumerate() {
            let mut cur = 0usize;
            for ch in pat.chars() {
                let next = *nodes[cur].children.entry(ch).or_insert_with(|| {
                    nodes.push(AcNode::default());
                    nodes.len() - 1
                });
                cur = next;
            }
            nodes[cur].output.push(idx);
        }

        // Phase 2: build failure links via BFS
        let mut queue = VecDeque::new();
        // Depth-1 nodes fail to root
        for &child in nodes[0].children.values() {
            nodes[child].fail = 0;
            queue.push_back(child);
        }

        while let Some(r) = queue.pop_front() {
            let keys: Vec<char> = nodes[r].children.keys().copied().collect();
            for ch in keys {
                let u = nodes[r].children[&ch];
                let mut f = nodes[r].fail;
                while f != 0 && !nodes[f].children.contains_key(&ch) {
                    f = nodes[f].fail;
                }
                nodes[u].fail = match nodes[f].children.get(&ch) {
                    Some(&v) if v != u => v,
                    _ => 0,
                };
                // Merge output of failure link
                let out = nodes[nodes[u].fail].output.clone();
                nodes[u].output.extend(out);
                queue.push_back(u);
            }
        }

        AhoCorasick { nodes, patterns }
    }

    /// Search `text` and return `(start, end, pattern_index)` for every match.
    pub fn find_all(&self, text: &str) -> Vec<(usize, usize, usize)> {
        let mut results = Vec::new();
        let mut state = 0usize;
        for (i, ch) in text.char_indices() {
            while state != 0 && !self.nodes[state].children.contains_key(&ch) {
                state = self.nodes[state].fail;
            }
            state = match self.nodes[state].children.get(&ch) {
                Some(&next) => next,
                None => 0,
            };
            for &pat_idx in &self.nodes[state].output {
                let pat_len = self.patterns[pat_idx].len();
                results.push((i + ch.len_utf8() - pat_len, i + ch.len_utf8(), pat_idx));
            }
        }
        results
    }

    /// Return `true` if any pattern appears in `text`.
    pub fn is_match(&self, text: &str) -> bool {
        let mut state = 0usize;
        for ch in text.chars() {
            while state != 0 && !self.nodes[state].children.contains_key(&ch) {
                state = self.nodes[state].fail;
            }
            state = match self.nodes[state].children.get(&ch) {
                Some(&next) => next,
                None => 0,
            };
            if !self.nodes[state].output.is_empty() {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Levenshtein distance
// ---------------------------------------------------------------------------

/// Compute the Levenshtein (edit) distance between two strings.
///
/// Time complexity: O(m * n) where m and n are the byte lengths.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// ---------------------------------------------------------------------------
// Longest Common Subsequence
// ---------------------------------------------------------------------------

/// Return the length of the longest common subsequence of `a` and `b`.
///
/// Time complexity: O(m * n).
pub fn lcs_len(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 || n == 0 {
        return 0;
    }

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            curr[j] = if a_chars[i - 1] == b_chars[j - 1] {
                prev[j - 1] + 1
            } else {
                prev[j].max(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Return the longest common subsequence as a `String`.
///
/// Uses backtracking on the full DP table.  Time and space: O(m * n).
pub fn lcs_string(a: &str, b: &str) -> String {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 || n == 0 {
        return String::new();
    }

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a_chars[i - 1] == b_chars[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }

    // Backtrack
    let mut result = Vec::new();
    let (mut i, mut j) = (m, n);
    while i > 0 && j > 0 {
        if a_chars[i - 1] == b_chars[j - 1] {
            result.push(a_chars[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    result.reverse();
    result.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Jaro similarity
// ---------------------------------------------------------------------------

/// Compute the Jaro similarity between two strings.
///
/// Returns a value in `[0.0, 1.0]` where `1.0` means identical.
pub fn jaro(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 && n == 0 { return 1.0; }
    if m == 0 || n == 0 { return 0.0; }

    let match_distance = m.max(n) / 2 - 1;
    let mut a_matches = vec![false; m];
    let mut b_matches = vec![false; n];
    let mut matches = 0f64;
    let mut transpositions = 0f64;

    // Match phase
    for i in 0..m {
        let lo = if i >= match_distance { i - match_distance } else { 0 };
        let lo = lo.min(n);
        let hi = (i + match_distance + 1).min(n);
        for j in lo..hi {
            if b_matches[j] || a_chars[i] != b_chars[j] {
                continue;
            }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1.0;
            break;
        }
    }

    if matches == 0.0 {
        return 0.0;
    }

    // Transposition phase
    let mut k = 0usize;
    for i in 0..m {
        if !a_matches[i] { continue; }
        while !b_matches[k] { k += 1; }
        if a_chars[i] != b_chars[k] {
            transpositions += 1.0;
        }
        k += 1;
    }

    (matches / m as f64
        + matches / n as f64
        + (matches - transpositions / 2.0) / matches)
        / 3.0
}

// ---------------------------------------------------------------------------
// Jaro-Winkler similarity
// ---------------------------------------------------------------------------

/// Compute the Jaro-Winkler similarity, which gives a higher score to strings
/// that match from the beginning.
///
/// The optional `p` parameter controls the prefix weight (default 0.1, max 0.25).
pub fn jaro_winkler(a: &str, b: &str, p: Option<f64>) -> f64 {
    let j = jaro(a, b);
    let p = p.unwrap_or(0.1).min(0.25);
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let max_prefix = 4.min(a_chars.len().min(b_chars.len()));

    let mut l = 0usize;
    while l < max_prefix && a_chars[l] == b_chars[l] {
        l += 1;
    }

    j + l as f64 * p * (1.0 - j)
}

// ---------------------------------------------------------------------------
// Jaccard similarity
// ---------------------------------------------------------------------------

/// Compute Jaccard similarity between two strings using character n-grams.
///
/// `n` controls the n-gram size (commonly 2 for bigrams).
/// Returns a value in `[0.0, 1.0]`.
pub fn jaccard_ngram(a: &str, b: &str, n: usize) -> f64 {
    if n == 0 { return 1.0; }
    let a_ngrams = ngram_set(a, n);
    let b_ngrams = ngram_set(b, n);

    if a_ngrams.is_empty() && b_ngrams.is_empty() {
        return 1.0;
    }

    let intersection = a_ngrams.intersection(&b_ngrams).count();
    let union = a_ngrams.union(&b_ngrams).count();

    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

fn ngram_set(s: &str, n: usize) -> HashSet<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < n {
        let mut set = HashSet::new();
        set.insert(s.to_string());
        return set;
    }
    chars
        .windows(n)
        .map(|w| w.iter().collect::<String>())
        .collect()
}

// ---------------------------------------------------------------------------
// Jaccard set similarity (token-level)
// ---------------------------------------------------------------------------

/// Compute Jaccard similarity treating inputs as sets of whitespace-separated
/// tokens.
pub fn jaccard_tokens(a: &str, b: &str) -> f64 {
    let a_set: HashSet<&str> = a.split_whitespace().collect();
    let b_set: HashSet<&str> = b.split_whitespace().collect();

    if a_set.is_empty() && b_set.is_empty() {
        return 1.0;
    }

    let intersection = a_set.intersection(&b_set).count();
    let union = a_set.union(&b_set).count();

    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

// ---------------------------------------------------------------------------
// Suffix array
// ---------------------------------------------------------------------------

/// A suffix array: an array of starting indices into a string, sorted by the
/// suffix they represent.
#[derive(Debug, Clone)]
pub struct SuffixArray {
    text: String,
    /// Sorted suffix indices.
    sa: Vec<usize>,
    /// Inverse suffix array: rank of the suffix starting at position `i`.
    rank: Vec<usize>,
}

impl SuffixArray {
    /// Build a suffix array for the given text using the naive O(n^2 log n)
    /// approach.  This is simple and correct; for very large inputs a
    /// SA-IS or DC3 implementation would be preferred.
    pub fn new(text: &str) -> Self {
        let n = text.len();
        let mut sa: Vec<usize> = (0..n).collect();
        sa.sort_by(|&a, &b| text[a..].cmp(&text[b..]));

        let mut rank = vec![0usize; n];
        for (i, &pos) in sa.iter().enumerate() {
            rank[pos] = i;
        }

        SuffixArray {
            text: text.to_string(),
            sa,
            rank,
        }
    }

    /// Return the suffix array (sorted indices).
    pub fn as_slice(&self) -> &[usize] {
        &self.sa
    }

    /// Return the rank array.
    pub fn rank(&self) -> &[usize] {
        &self.rank
    }

    /// Return the original text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Return the suffix starting at position `i` in the sorted array.
    pub fn suffix_at(&self, idx: usize) -> &str {
        &self.text[self.sa[idx]..]
    }

    /// Binary search for `query` among the sorted suffixes.
    ///
    /// Returns the range of suffix array indices whose suffixes start with
    /// `query`.  The range is empty when the query is not found.
    pub fn search(&self, query: &str) -> std::ops::Range<usize> {
        let n = self.sa.len();
        if query.is_empty() {
            return 0..n;
        }

        // Lower bound
        let lo = {
            let (mut lo, mut hi) = (0, n);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if self.text[self.sa[mid]..].cmp(query) == std::cmp::Ordering::Less {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        };

        // Upper bound
        let hi = {
            let (mut lo, mut hi) = (lo, n);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let suffix = &self.text[self.sa[mid]..];
                if suffix.len() < query.len() || suffix[..query.len()].cmp(query) != std::cmp::Ordering::Greater {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        };

        lo..hi
    }

    /// Count how many times `query` occurs in the text.
    pub fn count(&self, query: &str) -> usize {
        self.search(query).len()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Aho-Corasick -------------------------------------------------------

    #[test]
    fn ac_single_pattern() {
        let ac = AhoCorasick::new(["hello"]);
        let hits = ac.find_all("say hello world");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0], (4, 9, 0));
    }

    #[test]
    fn ac_multiple_patterns() {
        let ac = AhoCorasick::new(["he", "she", "his", "hers"]);
        let hits = ac.find_all("ahishers");
        // "his" at 1..4, "he" at 4..6, "she" at 3..6, "hers" at 4..8
        assert!(hits.len() >= 4);
        assert!(ac.is_match("ahishers"));
        assert!(!ac.is_match("nothing"));
    }

    #[test]
    fn ac_overlapping() {
        let ac = AhoCorasick::new(["ab", "abc", "bc"]);
        let hits = ac.find_all("abc");
        // Should find "ab", "abc", "bc"
        assert!(hits.len() >= 3);
    }

    #[test]
    fn ac_no_match() {
        let ac = AhoCorasick::new(["xyz", "qrs"]);
        assert!(ac.find_all("hello world").is_empty());
        assert!(!ac.is_match("hello world"));
    }

    #[test]
    fn ac_unicode() {
        let ac = AhoCorasick::new(["cafe", "naive"]);
        assert!(ac.is_match("a naive cafe"));
    }

    // -- Levenshtein --------------------------------------------------------

    #[test]
    fn lev_identical() {
        assert_eq!(levenshtein("kitten", "kitten"), 0);
    }

    #[test]
    fn lev_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn lev_classic() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("saturday", "sunday"), 3);
    }

    #[test]
    fn lev_single_char() {
        assert_eq!(levenshtein("a", "b"), 1);
        assert_eq!(levenshtein("a", "a"), 0);
    }

    // -- LCS ----------------------------------------------------------------

    #[test]
    fn lcs_length_classic() {
        assert_eq!(lcs_len("ABCBDAB", "BDCAB"), 4);
    }

    #[test]
    fn lcs_length_identical() {
        assert_eq!(lcs_len("abcdef", "abcdef"), 6);
    }

    #[test]
    fn lcs_length_empty() {
        assert_eq!(lcs_len("", "abc"), 0);
        assert_eq!(lcs_len("abc", ""), 0);
    }

    #[test]
    fn lcs_string_basic() {
        let s = lcs_string("ABCBDAB", "BDCAB");
        assert_eq!(s.len(), 4);
        // The LCS is one of: BCAB, BDAB, BDCAB (length 4)
        assert!(s == "BCBA" || s == "BDAB" || s == "BCAB");
    }

    #[test]
    fn lcs_string_identical() {
        assert_eq!(lcs_string("hello", "hello"), "hello");
    }

    // -- Jaro ---------------------------------------------------------------

    #[test]
    fn jaro_identical() {
        assert!((jaro("abc", "abc") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jaro_empty() {
        assert!((jaro("", "") - 1.0).abs() < 1e-10);
        assert!((jaro("abc", "")).abs() < 1e-10);
        assert!((jaro("", "abc")).abs() < 1e-10);
    }

    #[test]
    fn jaro_classic() {
        // "MARTHA" vs "MARHTA" -> ~0.944
        let v = jaro("MARTHA", "MARHTA");
        assert!((v - 0.944444).abs() < 0.001);
    }

    #[test]
    fn jaro_completely_different() {
        let v = jaro("abc", "xyz");
        assert!(v < 0.01);
    }

    // -- Jaro-Winkler -------------------------------------------------------

    #[test]
    fn jw_identical() {
        assert!((jaro_winkler("abc", "abc", None) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jw_boosts_prefix() {
        let j = jaro("MARTHA", "MARHTA");
        let jw = jaro_winkler("MARTHA", "MARHTA", None);
        assert!(jw > j, "Winkler should boost the Jaro score for shared prefix");
    }

    // -- Jaccard ------------------------------------------------------------

    #[test]
    fn jaccard_identical() {
        assert!((jaccard_ngram("hello", "hello", 2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jaccard_no_overlap() {
        let v = jaccard_ngram("abc", "xyz", 2);
        assert!(v < 0.01);
    }

    #[test]
    fn jaccard_partial() {
        let v = jaccard_ngram("night", "nacht", 2);
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn jaccard_tokens_basic() {
        let v = jaccard_tokens("the cat sat", "the dog sat");
        // intersection = {the, sat}, union = {the, cat, sat, dog} => 2/4 = 0.5
        assert!((v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn jaccard_tokens_identical() {
        assert!((jaccard_tokens("a b c", "a b c") - 1.0).abs() < 1e-10);
    }

    // -- Suffix array -------------------------------------------------------

    #[test]
    fn sa_basic_build() {
        let sa = SuffixArray::new("banana");
        // Suffixes sorted: a, ana, anana, banana, na, nana
        assert_eq!(sa.as_slice(), &[5, 3, 1, 0, 4, 2]);
    }

    #[test]
    fn sa_search_found() {
        let sa = SuffixArray::new("banana");
        let range = sa.search("ana");
        // "ana" starts at positions 1 and 3
        assert_eq!(range.len(), 2);
    }

    #[test]
    fn sa_search_not_found() {
        let sa = SuffixArray::new("banana");
        assert_eq!(sa.search("xyz").len(), 0);
    }

    #[test]
    fn sa_count() {
        let sa = SuffixArray::new("abracadabra");
        assert_eq!(sa.count("abra"), 2);
        assert_eq!(sa.count("a"), 5);
        assert_eq!(sa.count("bra"), 2);
        assert_eq!(sa.count("cad"), 1);
    }

    #[test]
    fn sa_suffix_at() {
        let sa = SuffixArray::new("banana");
        assert_eq!(sa.suffix_at(0), "a");
        assert_eq!(sa.suffix_at(5), "nana");
    }

    #[test]
    fn sa_rank() {
        let sa = SuffixArray::new("banana");
        let rank = sa.rank();
        // rank[5] = 0 (suffix "a" is first)
        assert_eq!(rank[5], 0);
        // rank[0] = 3 (suffix "banana" is fourth)
        assert_eq!(rank[0], 3);
    }

    #[test]
    fn sa_empty_text() {
        let sa = SuffixArray::new("");
        assert!(sa.as_slice().is_empty());
        assert_eq!(sa.search("").len(), 0);
    }

    #[test]
    fn sa_single_char() {
        let sa = SuffixArray::new("a");
        assert_eq!(sa.as_slice(), &[0]);
        assert_eq!(sa.count("a"), 1);
        assert_eq!(sa.count("b"), 0);
    }
}
