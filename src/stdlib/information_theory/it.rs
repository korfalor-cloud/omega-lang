//! Comprehensive information theory module.
//!
//! Provides entropy, divergence measures, channel capacity calculations,
//! rate-distortion theory, and source coding algorithms.

use std::collections::BinaryHeap;
use std::cmp::Reverse;

// ---------------------------------------------------------------------------
// Core entropy measures
// ---------------------------------------------------------------------------

/// Shannon entropy H(X) = -sum p(x) log2 p(x).
/// Returns 0.0 when the distribution is empty or all zeros.
pub fn entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Conditional entropy H(Y|X) = H(X,Y) - H(X).
/// `joint` is the joint distribution P(X,Y) laid out row-major with
/// `cols` columns (one per Y value).
pub fn conditional_entropy(joint: &[f64], cols: usize) -> f64 {
    if joint.is_empty() || cols == 0 {
        return 0.0;
    }
    let h_xy = entropy(joint);
    let rows = joint.len() / cols;
    let mut marginal_x = Vec::with_capacity(rows);
    for r in 0..rows {
        let row_sum: f64 = joint[r * cols..(r + 1) * cols].iter().sum();
        marginal_x.push(row_sum);
    }
    h_xy - entropy(&marginal_x)
}

/// Mutual information I(X;Y) = H(Y) - H(Y|X).
/// `joint` is P(X,Y) row-major with `cols` columns.
pub fn mutual_information(joint: &[f64], cols: usize) -> f64 {
    if joint.is_empty() || cols == 0 {
        return 0.0;
    }
    let mut marginal_y = vec![0.0; cols];
    for (i, &p) in joint.iter().enumerate() {
        marginal_y[i % cols] += p;
    }
    entropy(&marginal_y) - conditional_entropy(joint, cols)
}

/// Cross entropy H_p(q) = -sum p(x) log2 q(x).
pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have equal length");
    p.iter()
        .zip(q.iter())
        .filter(|(&pi, &qi)| pi > 0.0 && qi > 0.0)
        .map(|(&pi, &qi)| -pi * qi.log2())
        .sum()
}

// ---------------------------------------------------------------------------
// Divergence measures
// ---------------------------------------------------------------------------

/// Kullback-Leibler divergence D_KL(p || q) = sum p(x) log2(p(x)/q(x)).
/// Returns infinity when q(x)=0 and p(x)>0.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have equal length");
    p.iter()
        .zip(q.iter())
        .filter(|(&pi, _)| pi > 0.0)
        .map(|(&pi, &qi)| {
            if qi <= 0.0 {
                f64::INFINITY
            } else {
                pi * (pi / qi).log2()
            }
        })
        .sum()
}

/// Jensen-Shannon divergence JSD(p, q) = 0.5*D_KL(p||m) + 0.5*D_KL(q||m)
/// where m = 0.5*(p+q).  Bounded in [0, 1].
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have equal length");
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| 0.5 * (pi + qi)).collect();
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

/// Total variation distance TV(p, q) = 0.5 * sum |p(x) - q(x)|.
pub fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    0.5 * p.iter().zip(q.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>()
}

// ---------------------------------------------------------------------------
// Channel capacity: Binary Symmetric Channel & Binary Erasure Channel
// ---------------------------------------------------------------------------

/// Binary Symmetric Channel capacity C = 1 - H(p) where p is the crossover
/// probability.
pub fn bsc_capacity(crossover: f64) -> f64 {
    assert!((0.0..=1.0).contains(&crossover));
    1.0 - entropy(&[crossover, 1.0 - crossover])
}

/// Binary Erasure Channel capacity C = 1 - erasure_prob
/// where erasure_prob is the probability of erasure.
pub fn bec_capacity(erasure: f64) -> f64 {
    assert!((0.0..=1.0).contains(&erasure));
    1.0 - erasure
}

/// Channel capacity for a discrete memoryless channel given by its transition
/// matrix (row-major, rows=input, cols=output) via the Blahut-Arimoto algorithm.
/// Returns the capacity in bits per channel use.
pub fn channel_capacity_blahut_arimoto(
    transition: &[Vec<f64>],
    max_iter: usize,
    tol: f64,
) -> f64 {
    let n_inputs = transition.len();
    if n_inputs == 0 {
        return 0.0;
    }
    let n_outputs = transition[0].len();
    // Initialise uniform input distribution
    let mut q = vec![1.0 / n_inputs as f64; n_inputs];

    for _ in 0..max_iter {
        // Compute output distribution r(y) = sum_x q(x)*W(y|x)
        let mut r = vec![0.0; n_outputs];
        for x in 0..n_inputs {
            for y in 0..n_outputs {
                r[y] += q[x] * transition[x][y];
            }
        }
        // Update q(x) proportional to exp(D_KL(W(y|x) || r))
        let mut q_new = Vec::with_capacity(n_inputs);
        for x in 0..n_inputs {
            let exponent: f64 = transition[x]
                .iter()
                .zip(r.iter())
                .filter(|(&w, &ri)| w > 0.0 && ri > 0.0)
                .map(|(&w, &ri)| w * (w / ri).log2())
                .sum();
            q_new.push(exponent.exp());
        }
        let sum_q: f64 = q_new.iter().sum();
        for v in q_new.iter_mut() {
            *v /= sum_q;
        }
        let delta: f64 = q.iter().zip(q_new.iter()).map(|(a, b)| (a - b).abs()).sum();
        q = q_new;
        if delta < tol {
            break;
        }
    }
    // Capacity = sum_x q(x) * D_KL(W(y|x) || r)
    let mut r = vec![0.0; n_outputs];
    for x in 0..n_inputs {
        for y in 0..n_outputs {
            r[y] += q[x] * transition[x][y];
        }
    }
    let mut cap = 0.0;
    for x in 0..n_inputs {
        let kl: f64 = transition[x]
            .iter()
            .zip(r.iter())
            .filter(|(&w, &ri)| w > 0.0 && ri > 0.0)
            .map(|(&w, &ri)| w * (w / ri).log2())
            .sum();
        cap += q[x] * kl;
    }
    cap
}

// ---------------------------------------------------------------------------
// Rate-distortion theory (binary symmetric source, Hamming distortion)
// ---------------------------------------------------------------------------

/// Rate-distortion function for a binary symmetric source with Hamming
/// distortion: R(D) = H(p) - H(D) for 0 <= D <= p, where p is the source
/// bias P(X=1).  Returns 0 for D >= p.
pub fn rate_distortion_binary(p_source: f64, d: f64) -> f64 {
    assert!((0.0..=1.0).contains(&p_source));
    if d >= p_source || d >= 1.0 - p_source {
        return 0.0;
    }
    let h_source = entropy(&[p_source, 1.0 - p_source]);
    let h_d = entropy(&[d, 1.0 - d]);
    (h_source - h_d).max(0.0)
}

// ---------------------------------------------------------------------------
// Source coding: Huffman
// ---------------------------------------------------------------------------

#[derive(Debug, Eq, PartialEq)]
struct HuffNode {
    freq: u64,
    symbol: Option<u16>,
    left: Option<Box<HuffNode>>,
    right: Option<Box<HuffNode>>,
}

impl Ord for HuffNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap
        other.freq.cmp(&self.freq)
    }
}
impl PartialOrd for HuffNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Build a Huffman code table mapping symbol -> codeword (as Vec<u8> of 0/1).
/// `freqs` maps symbol index to frequency; zero-frequency symbols are skipped.
pub fn huffman_encode_table(freqs: &[u64]) -> Vec<(u16, Vec<u8>)> {
    let mut heap: BinaryHeap<Reverse<HuffNode>> = BinaryHeap::new();
    for (i, &f) in freqs.iter().enumerate() {
        if f > 0 {
            heap.push(Reverse(HuffNode {
                freq: f,
                symbol: Some(i as u16),
                left: None,
                right: None,
            }));
        }
    }
    if heap.is_empty() {
        return Vec::new();
    }
    if heap.len() == 1 {
        let node = heap.pop().unwrap().0;
        return vec![(node.symbol.unwrap(), vec![0])];
    }
    while heap.len() > 1 {
        let a = heap.pop().unwrap().0;
        let b = heap.pop().unwrap().0;
        let parent = HuffNode {
            freq: a.freq + b.freq,
            symbol: None,
            left: Some(Box::new(a)),
            right: Some(Box::new(b)),
        };
        heap.push(Reverse(parent));
    }
    let root = heap.pop().unwrap().0;
    let mut table = Vec::new();
    fn walk(node: &HuffNode, prefix: &mut Vec<u8>, table: &mut Vec<(u16, Vec<u8>)>) {
        if let Some(sym) = node.symbol {
            table.push((sym, prefix.clone()));
            return;
        }
        if let Some(ref left) = node.left {
            prefix.push(0);
            walk(left, prefix, table);
            prefix.pop();
        }
        if let Some(ref right) = node.right {
            prefix.push(1);
            walk(right, prefix, table);
            prefix.pop();
        }
    }
    walk(&root, &mut Vec::new(), &mut table);
    table
}

/// Average codeword length for a Huffman code given frequency distribution.
pub fn huffman_avg_length(freqs: &[u64]) -> f64 {
    let table = huffman_encode_table(freqs);
    let total: u64 = freqs.iter().sum();
    if total == 0 {
        return 0.0;
    }
    table
        .iter()
        .map(|&(sym, ref codeword)| {
            freqs[sym as usize] as f64 * codeword.len() as f64 / total as f64
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Source coding: Arithmetic coding (fixed-precision, 32-bit)
// ---------------------------------------------------------------------------

/// Encode a sequence of symbols into a Vec of bits using arithmetic coding.
/// `cum_freq` is the cumulative frequency table (length = alphabet_size + 1),
/// where cum_freq[i] is the cumulative count of symbols < i.
/// `total` is the total frequency count (cum_freq[alphabet_size]).
pub fn arithmetic_encode(
    symbols: &[u16],
    cum_freq: &[u64],
    total: u64,
) -> Vec<u8> {
    let precision = 32u32;
    let full: u64 = 1u64 << precision;
    let half: u64 = full >> 1;
    let quarter: u64 = half >> 1;

    let mut low: u64 = 0;
    let mut high: u64 = full - 1;
    let mut pending: u32 = 0;
    let mut bits: Vec<u8> = Vec::new();

    let emit = |bit: u8, pending: &mut u32, bits: &mut Vec<u8>| {
        bits.push(bit);
        for _ in 0..*pending {
            bits.push(bit ^ 1);
        }
        *pending = 0;
    };

    for &sym in symbols {
        let range = high - low + 1;
        let sym_lo = cum_freq[sym as usize];
        let sym_hi = cum_freq[sym as usize + 1];
        high = low + (range * sym_hi / total) - 1;
        low = low + (range * sym_lo / total);

        loop {
            if high < half {
                emit(0, &mut pending, &mut bits);
                low = low << 1;
                high = (high << 1) | 1;
            } else if low >= half {
                emit(1, &mut pending, &mut bits);
                low = (low - half) << 1;
                high = ((high - half) << 1) | 1;
            } else if low >= quarter && high < 3 * quarter {
                pending += 1;
                low = (low - quarter) << 1;
                high = ((high - quarter) << 1) | 1;
            } else {
                break;
            }
        }
    }
    // Flush remaining bits
    pending += 1;
    if low < quarter {
        emit(0, &mut pending, &mut bits);
    } else {
        emit(1, &mut pending, &mut bits);
    }
    bits
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_entropy_uniform() {
        let p = vec![0.25; 4];
        assert!((entropy(&p) - 2.0).abs() < EPS);
    }

    #[test]
    fn test_entropy_deterministic() {
        assert!((entropy(&[1.0, 0.0, 0.0])).abs() < EPS);
    }

    #[test]
    fn test_entropy_empty() {
        assert!((entropy(&[])).abs() < EPS);
    }

    #[test]
    fn test_conditional_entropy_independent() {
        // P(X,Y) uniform over 2x2 => H(Y|X) = H(Y) = 1
        let joint = vec![0.25; 4];
        let h_y = entropy(&[0.5, 0.5]);
        assert!((conditional_entropy(&joint, 2) - h_y).abs() < EPS);
    }

    #[test]
    fn test_mutual_information_perfect() {
        // Y = X perfectly correlated: joint = diag(0.5, 0.5)
        let joint = vec![0.5, 0.0, 0.0, 0.5];
        assert!((mutual_information(&joint, 2) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_mutual_information_independent() {
        let joint = vec![0.25; 4];
        assert!((mutual_information(&joint, 2)).abs() < EPS);
    }

    #[test]
    fn test_cross_entropy() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        assert!((cross_entropy(&p, &q) - entropy(&p)).abs() < EPS);
    }

    #[test]
    fn test_kl_divergence_same() {
        let p = vec![0.3, 0.7];
        assert!((kl_divergence(&p, &p)).abs() < EPS);
    }

    #[test]
    fn test_kl_divergence_asymmetric() {
        let p = vec![0.25, 0.75];
        let q = vec![0.5, 0.5];
        let d_pq = kl_divergence(&p, &q);
        let d_qp = kl_divergence(&q, &p);
        assert!(d_pq > 0.0);
        assert!(d_qp > 0.0);
        assert!((d_pq - d_qp).abs() > 1e-6, "KL should be asymmetric");
    }

    #[test]
    fn test_js_divergence_same() {
        let p = vec![0.1, 0.9];
        assert!((js_divergence(&p, &p)).abs() < EPS);
    }

    #[test]
    fn test_js_divergence_symmetric() {
        let p = vec![0.25, 0.75];
        let q = vec![0.75, 0.25];
        assert!((js_divergence(&p, &q) - js_divergence(&q, &p)).abs() < EPS);
    }

    #[test]
    fn test_js_divergence_bounded() {
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        // Extreme case: JSD = 1 bit
        assert!((js_divergence(&p, &q) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_total_variation() {
        let p = vec![0.3, 0.7];
        let q = vec![0.7, 0.3];
        assert!((total_variation(&p, &q) - 0.4).abs() < EPS);
    }

    #[test]
    fn test_bsc_no_noise() {
        assert!((bsc_capacity(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_bsc_max_noise() {
        assert!((bsc_capacity(0.5)).abs() < EPS);
    }

    #[test]
    fn test_bec_no_erasure() {
        assert!((bec_capacity(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_bec_full_erasure() {
        assert!((bec_capacity(1.0)).abs() < EPS);
    }

    #[test]
    fn test_blahut_arimoto_bsc() {
        // BSC with crossover 0.1 => capacity ~ 0.531
        let p = 0.1;
        let transition = vec![
            vec![1.0 - p, p],
            vec![p, 1.0 - p],
        ];
        let cap = channel_capacity_blahut_arimoto(&transition, 1000, 1e-10);
        let expected = bsc_capacity(p);
        assert!((cap - expected).abs() < 1e-4, "got {cap}, expected {expected}");
    }

    #[test]
    fn test_blahut_arimoto_noiseless() {
        // Noiseless binary channel
        let transition = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let cap = channel_capacity_blahut_arimoto(&transition, 1000, 1e-10);
        assert!((cap - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rate_distortion_no_distortion() {
        let r = rate_distortion_binary(0.5, 0.0);
        assert!((r - 1.0).abs() < EPS);
    }

    #[test]
    fn test_rate_distortion_max_distortion() {
        let r = rate_distortion_binary(0.5, 0.5);
        assert!((r).abs() < EPS);
    }

    #[test]
    fn test_huffman_basic() {
        let freqs = vec![45, 13, 12, 16, 9, 5]; // classic example
        let table = huffman_encode_table(&freqs);
        assert_eq!(table.len(), 6);
        // Average length should be less than entropy + 1
        let avg = huffman_avg_length(&freqs);
        let total: u64 = freqs.iter().sum();
        let probs: Vec<f64> = freqs.iter().map(|&f| f as f64 / total as f64).collect();
        let h = entropy(&probs);
        assert!(avg >= h, "avg length must be >= entropy");
        assert!(avg < h + 1.0, "Huffman is within 1 bit of entropy");
    }

    #[test]
    fn test_huffman_single_symbol() {
        let freqs = vec![0, 100, 0];
        let table = huffman_encode_table(&freqs);
        assert_eq!(table.len(), 1);
        assert_eq!(table[0].1, vec![0]);
    }

    #[test]
    fn test_arithmetic_coding_roundtrip() {
        // Encode uniform binary source and verify bit count is reasonable
        let symbols: Vec<u16> = vec![0, 1, 0, 0, 1, 1, 0, 1, 0, 0];
        let cum_freq = vec![0u64, 1, 2]; // alphabet size 2, uniform
        let total = 2u64;
        let bits = arithmetic_encode(&symbols, cum_freq, total);
        // 10 symbols with entropy 1 bit each => ~10 bits
        assert!(
            bits.len() <= 20,
            "encoded length should be reasonable: got {}",
            bits.len()
        );
    }

    #[test]
    fn test_arithmetic_skewed_source() {
        // Highly skewed source: symbol 0 is very frequent
        let symbols: Vec<u16> = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let cum_freq = vec![0u64, 9, 10];
        let total = 10u64;
        let bits = arithmetic_encode(&symbols, cum_freq, total);
        // Entropy is low, should compress well
        assert!(bits.len() < 10, "skewed source should compress: got {} bits", bits.len());
    }

    #[test]
    fn test_entropy_binary_coin() {
        // Fair coin: H = 1 bit
        let h = entropy(&[0.5, 0.5]);
        assert!((h - 1.0).abs() < EPS);
    }

    #[test]
    fn test_channel_capacity_ternary_symmetric() {
        // 3-ary symmetric channel with crossover prob p
        // C = log2(3) - H(p/(n-1), ..., p/(n-1), 1-p) for n=3
        let p = 0.3;
        let q = p / 2.0;
        let transition = vec![
            vec![1.0 - p, q, q],
            vec![q, 1.0 - p, q],
            vec![q, q, 1.0 - p],
        ];
        let cap = channel_capacity_blahut_arimoto(&transition, 2000, 1e-12);
        let expected = 3.0_f64.log2() - entropy(&[1.0 - p, q, q]);
        assert!(
            (cap - expected).abs() < 1e-4,
            "ternary symmetric: got {cap}, expected {expected}"
        );
    }

    #[test]
    fn test_mutual_information_3x3() {
        // 3x3 joint distribution
        let joint = vec![
            0.1, 0.1, 0.1,
            0.1, 0.2, 0.1,
            0.05, 0.1, 0.15,
        ];
        let mi = mutual_information(&joint, 3);
        assert!(mi >= 0.0, "MI must be non-negative");
        // H(Y) >= I(X;Y) >= 0
        let mut marginal_y = [0.0f64; 3];
        for (i, &p) in joint.iter().enumerate() {
            marginal_y[i % 3] += p;
        }
        let h_y = entropy(&marginal_y);
        assert!(mi <= h_y + EPS);
    }
}
