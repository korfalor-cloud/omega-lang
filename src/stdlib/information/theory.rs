/// Information theory: entropy, mutual information, channel coding, source coding.

use std::collections::HashMap;

/// Shannon entropy of a probability distribution.
pub fn entropy(probs: &[f64]) -> f64 {
    -probs.iter().filter(|&&p| p > 0.0).map(|&p| p * p.log2()).sum::<f64>()
}

/// Entropy of a discrete random variable given values.
pub fn discrete_entropy(values: &[f64]) -> f64 {
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &v in values {
        let key = (v * 1e9) as i64;
        *counts.entry(key).or_insert(0) += 1;
    }
    let n = values.len() as f64;
    let probs: Vec<f64> = counts.values().map(|&c| c as f64 / n).collect();
    entropy(&probs)
}

/// Joint entropy of two discrete random variables.
pub fn joint_entropy(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut counts: HashMap<(i64, i64), usize> = HashMap::new();
    for (&a, &b) in x.iter().zip(y.iter()) {
        let key = ((a * 1e9) as i64, (b * 1e9) as i64);
        *counts.entry(key).or_insert(0) += 1;
    }
    let n = x.len() as f64;
    let probs: Vec<f64> = counts.values().map(|&c| c as f64 / n).collect();
    entropy(&probs)
}

/// Conditional entropy H(Y|X).
pub fn conditional_entropy(x: &[f64], y: &[f64]) -> f64 {
    joint_entropy(x, y) - discrete_entropy(x)
}

/// Mutual information I(X;Y).
pub fn mutual_information(x: &[f64], y: &[f64]) -> f64 {
    discrete_entropy(x) + discrete_entropy(y) - joint_entropy(x, y)
}

/// Normalized mutual information.
pub fn normalized_mutual_information(x: &[f64], y: &[f64]) -> f64 {
    let mi = mutual_information(x, y);
    let hx = discrete_entropy(x);
    let hy = discrete_entropy(y);
    if hx + hy == 0.0 { 0.0 } else { 2.0 * mi / (hx + hy) }
}

/// Kullback-Leibler divergence D_KL(P || Q).
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    p.iter().zip(q.iter())
        .filter(|(&pi, &qi)| pi > 0.0 && qi > 0.0)
        .map(|(&pi, &qi)| pi * (pi / qi).log2())
        .sum()
}

/// Jensen-Shannon divergence.
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&a, &b)| (a + b) / 2.0).collect();
    (kl_divergence(p, &m) + kl_divergence(q, &m)) / 2.0
}

/// Cross-entropy H(P, Q) = -sum P(x) log Q(x).
pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    -p.iter().zip(q.iter())
        .filter(|(_, &qi)| qi > 0.0)
        .map(|(&pi, &qi)| pi * qi.log2())
        .sum()
}

/// Huffman coding.
pub struct HuffmanTree {
    pub root: HuffmanNode,
}

#[derive(Debug, Clone)]
pub enum HuffmanNode {
    Leaf { symbol: usize, frequency: usize },
    Internal { left: Box<HuffmanNode>, right: Box<HuffmanNode>, frequency: usize },
}

impl HuffmanNode {
    pub fn frequency(&self) -> usize {
        match self {
            HuffmanNode::Leaf { frequency, .. } => *frequency,
            HuffmanNode::Internal { frequency, .. } => *frequency,
        }
    }
}

impl HuffmanTree {
    pub fn build(symbols: &[(usize, usize)]) -> Self { // (symbol, frequency)
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut heap: BinaryHeap<Reverse<(usize, HuffmanNode)>> = BinaryHeap::new();
        for &(symbol, freq) in symbols {
            heap.push(Reverse((freq, HuffmanNode::Leaf { symbol, frequency: freq })));
        }

        while heap.len() > 1 {
            let Reverse((f1, left)) = heap.pop().unwrap();
            let Reverse((f2, right)) = heap.pop().unwrap();
            let parent = HuffmanNode::Internal {
                frequency: f1 + f2,
                left: Box::new(left),
                right: Box::new(right),
            };
            heap.push(Reverse((f1 + f2, parent)));
        }

        let Reverse((_, root)) = heap.pop().unwrap();
        Self { root }
    }

    pub fn codes(&self) -> HashMap<usize, Vec<u8>> {
        let mut codes = HashMap::new();
        self.build_codes(&self.root, &mut Vec::new(), &mut codes);
        codes
    }

    fn build_codes(&self, node: &HuffmanNode, path: &mut Vec<u8>, codes: &mut HashMap<usize, Vec<u8>>) {
        match node {
            HuffmanNode::Leaf { symbol, .. } => {
                codes.insert(*symbol, path.clone());
            }
            HuffmanNode::Internal { left, right, .. } => {
                path.push(0);
                self.build_codes(left, path, codes);
                path.pop();
                path.push(1);
                self.build_codes(right, path, codes);
                path.pop();
            }
        }
    }

    pub fn encode(&self, data: &[usize]) -> Vec<u8> {
        let codes = self.codes();
        let mut bits = Vec::new();
        for &symbol in data {
            if let Some(code) = codes.get(&symbol) {
                bits.extend(code);
            }
        }
        bits
    }

    pub fn decode(&self, bits: &[u8]) -> Vec<usize> {
        let mut result = Vec::new();
        let mut node = &self.root;

        for &bit in bits {
            match node {
                HuffmanNode::Internal { left, right, .. } => {
                    node = if bit == 0 { left } else { right };
                    if let HuffmanNode::Leaf { symbol, .. } = node {
                        result.push(*symbol);
                        node = &self.root;
                    }
                }
                _ => {}
            }
        }

        result
    }
}

/// Arithmetic coding.
pub struct ArithmeticCoder {
    pub precision: u32,
}

impl ArithmeticCoder {
    pub fn new(precision: u32) -> Self {
        Self { precision }
    }

    pub fn encode(&self, data: &[usize], frequencies: &HashMap<usize, usize>) -> (u64, u64) {
        let total: u64 = frequencies.values().sum::<usize>() as u64;
        let max_val = 1u64 << self.precision;

        let mut cumulative: HashMap<usize, u64> = HashMap::new();
        let mut cum = 0u64;
        for i in 0..frequencies.len() {
            if let Some(&freq) = frequencies.get(&i) {
                cumulative.insert(i, cum);
                cum += freq as u64;
            }
        }

        let mut low = 0u64;
        let mut high = max_val;

        for &symbol in data {
            let range = high - low;
            let sym_low = cumulative.get(&symbol).copied().unwrap_or(0);
            let sym_high = sym_low + frequencies.get(&symbol).copied().unwrap_or(0) as u64;

            high = low + (range * sym_high) / total;
            low = low + (range * sym_low) / total;
        }

        (low, high)
    }

    pub fn decode(&self, code: u64, length: usize, frequencies: &HashMap<usize, usize>) -> Vec<usize> {
        let total: u64 = frequencies.values().sum::<usize>() as u64;
        let max_val = 1u64 << self.precision;

        let mut cumulative: Vec<(u64, u64, usize)> = Vec::new(); // (low, high, symbol)
        let mut cum = 0u64;
        for i in 0..frequencies.len() {
            if let Some(&freq) = frequencies.get(&i) {
                cumulative.push((cum, cum + freq as u64, i));
                cum += freq as u64;
            }
        }

        let mut result = Vec::new();
        let mut low = 0u64;
        let mut high = max_val;
        let mut val = code;

        for _ in 0..length {
            let range = high - low;
            let scaled = ((val - low) * total) / range;

            for &(sym_low, sym_high, symbol) in &cumulative {
                if scaled >= sym_low && scaled < sym_high {
                    result.push(symbol);
                    high = low + (range * sym_high) / total;
                    low = low + (range * sym_low) / total;
                    break;
                }
            }
        }

        result
    }
}

/// Channel capacity: binary symmetric channel.
pub fn bsc_capacity(error_prob: f64) -> f64 {
    if error_prob <= 0.0 || error_prob >= 1.0 {
        return 1.0;
    }
    1.0 + error_prob * error_prob.log2() + (1.0 - error_prob) * (1.0 - error_prob).log2()
}

/// Binary erasure channel capacity.
pub fn bec_capacity(erase_prob: f64) -> f64 {
    1.0 - erase_prob
}

/// Hamming distance between two bit strings.
pub fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

/// Hamming weight (number of 1-bits).
pub fn hamming_weight(a: &[u8]) -> usize {
    a.iter().filter(|&&x| x == 1).count()
}

/// Rate-distortion function (for binary source with Hamming distortion).
pub fn rate_distortion(distortion: f64) -> f64 {
    if distortion <= 0.0 { return 1.0; }
    if distortion >= 0.5 { return 0.0; }
    1.0 + distortion * distortion.log2() + (1.0 - distortion) * (1.0 - distortion).log2()
}

/// Kolmogorov complexity approximation via Lempel-Ziv complexity.
pub fn lz_complexity(data: &[u8]) -> usize {
    let n = data.len();
    if n == 0 { return 0; }

    let mut complexity = 1;
    let mut i = 1;
    let mut k = 1;
    let mut l = 1;

    while i + l <= n {
        if data[i..i + l] == data[k..k + l] {
            l += 1;
        } else {
            complexity += 1;
            i += l;
            if i > k {
                k = i;
            }
            l = 1;
        }
        if i + l > n { break; }
    }

    complexity
}

/// Rényi entropy of order alpha.
pub fn renyi_entropy(probs: &[f64], alpha: f64) -> f64 {
    if (alpha - 1.0).abs() < 1e-10 {
        return entropy(probs);
    }
    let sum: f64 = probs.iter().filter(|&&p| p > 0.0).map(|&p| p.powf(alpha)).sum();
    sum.ln() / ((alpha - 1.0) * 2.0_f64.ln())
}

/// Tsallis entropy.
pub fn tsallis_entropy(probs: &[f64], q: f64) -> f64 {
    if (q - 1.0).abs() < 1e-10 {
        return entropy(probs);
    }
    let sum: f64 = probs.iter().filter(|&&p| p > 0.0).map(|&p| p.powf(q)).sum();
    (1.0 - sum) / (q - 1.0)
}

/// Fisher information for a parametric family.
pub fn fisher_information(scores: &[f64]) -> f64 {
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy() {
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        assert!((entropy(&uniform) - 2.0).abs() < 1e-10);

        let certain = vec![1.0, 0.0, 0.0, 0.0];
        assert!((entropy(&certain)).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.25, 0.75];
        assert!(kl_divergence(&p, &q) > 0.0);
    }

    #[test]
    fn test_huffman() {
        let symbols = vec![(0, 45), (1, 13), (2, 12), (3, 16), (4, 9), (5, 5)];
        let tree = HuffmanTree::build(&symbols);
        let codes = tree.codes();

        // Most frequent symbol should have shortest code
        assert!(codes[&0].len() <= codes[&5].len());

        // Test encode/decode roundtrip
        let data = vec![0, 1, 2, 3, 4, 5, 0, 0];
        let encoded = tree.encode(&data);
        let decoded = tree.decode(&encoded);
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_bsc_capacity() {
        assert!((bsc_capacity(0.0) - 1.0).abs() < 1e-10);
        assert!((bsc_capacity(0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_lz_complexity() {
        let simple = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let complex = vec![0, 1, 0, 1, 1, 0, 1, 1];
        assert!(lz_complexity(&simple) < lz_complexity(&complex));
    }
}
