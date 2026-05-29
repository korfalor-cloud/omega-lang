//! Advanced random number generators and distributions.
//!
//! Provides Mersenne Twister, PCG, and Xoshiro256 PRNGs along with
//! statistical distributions (normal, exponential, gamma, beta) and
//! sampling algorithms (reservoir, weighted).

// ---------------------------------------------------------------------------
// RNG trait
// ---------------------------------------------------------------------------

/// Common interface every advanced RNG implements.
pub trait RngCore {
    fn next_u64(&mut self) -> u64;

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn next_range(&mut self, lo: u64, hi: u64) -> u64 {
        if lo >= hi {
            return lo;
        }
        lo + self.next_u64() % (hi - lo)
    }
}

// ---------------------------------------------------------------------------
// Mersenne Twister (MT19937-64)
// ---------------------------------------------------------------------------

pub struct MersenneTwister {
    mt: [u64; 312],
    idx: usize,
}

impl MersenneTwister {
    const NN: usize = 312;
    const MM: usize = 156;
    const MATRIX_A: u64 = 0xB5026F5AA96619E9;
    const UM: u64 = 0xFFFFFFFF80000000;
    const LM: u64 = 0x7FFFFFFF;

    pub fn new(seed: u64) -> Self {
        let mut mt = Self {
            mt: [0u64; Self::NN],
            idx: Self::NN,
        };
        mt.seed(seed);
        mt
    }

    fn seed(&mut self, seed: u64) {
        self.mt[0] = seed;
        for i in 1..Self::NN {
            self.mt[i] = 6364136223846793005u64
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 62))
                .wrapping_add(i as u64);
        }
    }

    fn twist(&mut self) {
        for i in 0..Self::NN {
            let x = (self.mt[i] & Self::UM) | (self.mt[(i + 1) % Self::NN] & Self::LM);
            self.mt[i] = self.mt[(i + Self::MM) % Self::NN] ^ (x >> 1);
            if x & 1 != 0 {
                self.mt[i] ^= Self::MATRIX_A;
            }
        }
        self.idx = 0;
    }
}

impl RngCore for MersenneTwister {
    fn next_u64(&mut self) -> u64 {
        if self.idx >= Self::NN {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;

        y ^= (y >> 29) & 0x5555555555555555;
        y ^= (y << 17) & 0x71D67FFFEDA60000;
        y ^= (y << 37) & 0xFFF7EEE000000000;
        y ^= y >> 43;
        y
    }
}

// ---------------------------------------------------------------------------
// PCG (Permuted Congruential Generator) — PCG-XSH-RR 64/32
// ---------------------------------------------------------------------------

pub struct PcgRng {
    state: u64,
    inc: u64,
}

impl PcgRng {
    pub fn new(seed: u64, seq: u64) -> Self {
        let mut rng = Self { state: 0, inc: (seq << 1) | 1 };
        rng.state = rng.state.wrapping_add(seed).wrapping_mul(6364136223846793005).wrapping_add(rng.inc);
        rng.next_u64(); // advance once
        rng
    }
}

impl RngCore for PcgRng {
    fn next_u64(&mut self) -> u64 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);

        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        ((xorshifted >> rot) | (xorshifted << ((!rot as u32).wrapping_add(1) & 31))) as u64
    }
}

// ---------------------------------------------------------------------------
// Xoshiro256**
// ---------------------------------------------------------------------------

pub struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to fill 4 words from a single seed.
        let mut s = [0u64; 4];
        let mut z = seed;
        for item in s.iter_mut() {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut v = z;
            v = (v ^ (v >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94D049BB133111EB);
            *item = v ^ (v >> 31);
        }
        Self { s }
    }

    fn rotl(x: u64, k: u32) -> u64 {
        (x << k) | (x >> (64 - k))
    }
}

impl RngCore for Xoshiro256 {
    fn next_u64(&mut self) -> u64 {
        let result = Self::rotl(self.s[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = Self::rotl(self.s[3], 45);

        result
    }
}

// ---------------------------------------------------------------------------
// Distributions
// ---------------------------------------------------------------------------

/// Box-Muller: standard normal N(0,1), then scale.
pub fn normal<R: RngCore>(rng: &mut R, mean: f64, std_dev: f64) -> f64 {
    let u1: f64 = rng.next_f64().max(1e-30);
    let u2 = rng.next_f64();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + std_dev * z
}

/// Inverse-transform exponential.
pub fn exponential<R: RngCore>(rng: &mut R, lambda: f64) -> f64 {
    -rng.next_f64().max(1e-30).ln() / lambda
}

/// Gamma(alpha, beta) via Marsaglia & Tsang's method for alpha >= 1,
/// falling back to the small-shape trick for alpha < 1.
pub fn gamma<R: RngCore>(rng: &mut R, shape: f64, scale: f64) -> f64 {
    assert!(shape > 0.0 && scale > 0.0, "gamma: shape and scale must be > 0");

    if shape < 1.0 {
        // For small shape: Gamma(a) = Gamma(a+1) * U^{1/a}
        return gamma(rng, shape + 1.0, scale) * rng.next_f64().max(1e-30).powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let mut x;
        let mut v;
        loop {
            x = normal(rng, 0.0, 1.0);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        let v = v * v * v;
        let u = rng.next_f64().max(1e-30);

        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v * scale;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v * scale;
        }
    }
}

/// Beta(a, b) via two Gamma draws.
pub fn beta<R: RngCore>(rng: &mut R, a: f64, b: f64) -> f64 {
    assert!(a > 0.0 && b > 0.0, "beta: a and b must be > 0");
    let ga = gamma(rng, a, 1.0);
    let gb = gamma(rng, b, 1.0);
    ga / (ga + gb)
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Reservoir sampling (Algorithm R) — selects `k` items uniformly at random
/// from an iterator of unknown length, using O(k) memory.
pub fn reservoir_sample<T: Clone, I: Iterator<Item = T>>(iter: I, k: usize) -> Vec<T> {
    use super::random::OmegaRandom;
    let mut rng = OmegaRandom::new();
    let mut reservoir: Vec<T> = Vec::with_capacity(k);

    for (i, item) in iter.enumerate() {
        if i < k {
            reservoir.push(item);
        } else {
            let j = rng.next_int_range(0, (i as i64) + 1) as usize;
            if j < k {
                reservoir[j] = item;
            }
        }
    }
    reservoir
}

/// Weighted sampling without replacement using Efraimidis-Spirakis.
/// Returns indices into the `weights` slice.
pub fn weighted_sample_without_replacement(weights: &[f64], k: usize) -> Vec<usize> {
    use super::random::OmegaRandom;
    let mut rng = OmegaRandom::new();

    assert!(k <= weights.len(), "cannot sample more items than available");

    // Compute key = U^{1/w} for each item, then pick the k largest keys.
    let mut keys: Vec<(f64, usize)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| {
            assert!(w >= 0.0, "weights must be non-negative");
            let u = rng.next_float().max(1e-30);
            (u.powf(1.0 / w.max(1e-30)), i)
        })
        .collect();

    keys.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    keys[..k].iter().map(|(_, i)| *i).collect()
}

/// Weighted sampling with replacement: draw `k` items from the given
/// distribution, returning indices.
pub fn weighted_sample_with_replacement(weights: &[f64], k: usize) -> Vec<usize> {
    use super::random::OmegaRandom;
    let mut rng = OmegaRandom::new();

    let total: f64 = weights.iter().sum();
    assert!(total > 0.0, "total weight must be positive");

    let cdf: Vec<f64> = weights
        .iter()
        .scan(0.0, |acc, w| {
            *acc += w / total;
            Some(*acc)
        })
        .collect();

    (0..k)
        .map(|_| {
            let u = rng.next_float();
            cdf.iter()
                .position(|&p| p >= u)
                .unwrap_or(weights.len() - 1)
        })
        .collect()
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLES: usize = 10_000;

    // -- RNG smoke tests ------------------------------------------------

    #[test]
    fn mersenne_twister_deterministic() {
        let mut a = MersenneTwister::new(42);
        let mut b = MersenneTwister::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn pcg_deterministic() {
        let mut a = PcgRng::new(42, 1);
        let mut b = PcgRng::new(42, 1);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn xoshiro256_deterministic() {
        let mut a = Xoshiro256::new(42);
        let mut b = Xoshiro256::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn rngs_produce_different_sequences_with_different_seeds() {
        let mut a = MersenneTwister::new(1);
        let mut b = MersenneTwister::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn f64_range_is_0_to_1() {
        let mut rng = Xoshiro256::new(7);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "out of range: {v}");
        }
    }

    // -- Distribution tests ---------------------------------------------

    #[test]
    fn normal_distribution_mean() {
        let mut rng = MersenneTwister::new(123);
        let sum: f64 = (0..SAMPLES).map(|_| normal(&mut rng, 5.0, 1.0)).sum();
        let mean = sum / SAMPLES as f64;
        assert!((mean - 5.0).abs() < 0.1, "mean={mean}");
    }

    #[test]
    fn exponential_distribution_mean() {
        let mut rng = PcgRng::new(1, 1);
        let lambda = 2.0;
        let sum: f64 = (0..SAMPLES).map(|_| exponential(&mut rng, lambda)).sum();
        let mean = sum / SAMPLES as f64;
        // E[X] = 1/lambda
        assert!((mean - 1.0 / lambda).abs() < 0.05, "mean={mean}");
    }

    #[test]
    fn gamma_distribution_mean() {
        let mut rng = Xoshiro256::new(99);
        let shape = 3.0;
        let scale = 2.0;
        let sum: f64 = (0..SAMPLES).map(|_| gamma(&mut rng, shape, scale)).sum();
        let mean = sum / SAMPLES as f64;
        // E[X] = shape * scale
        assert!((mean - shape * scale).abs() < 0.5, "mean={mean}");
    }

    #[test]
    fn beta_distribution_bounded() {
        let mut rng = PcgRng::new(7, 3);
        for _ in 0..SAMPLES {
            let v = beta(&mut rng, 2.0, 5.0);
            assert!((0.0..=1.0).contains(&v), "beta out of range: {v}");
        }
    }

    #[test]
    fn beta_distribution_mean() {
        let mut rng = MersenneTwister::new(55);
        let a = 2.0;
        let b = 5.0;
        let sum: f64 = (0..SAMPLES).map(|_| beta(&mut rng, a, b)).sum();
        let mean = sum / SAMPLES as f64;
        // E[X] = a / (a + b)
        assert!((mean - a / (a + b)).abs() < 0.05, "mean={mean}");
    }

    #[test]
    fn gamma_small_shape() {
        let mut rng = Xoshiro256::new(11);
        // shape < 1 exercises the recursive path
        let v = gamma(&mut rng, 0.5, 1.0);
        assert!(v.is_finite() && v >= 0.0);
    }

    // -- Sampling tests -------------------------------------------------

    #[test]
    fn reservoir_sample_respects_size() {
        let data: Vec<i32> = (0..100).collect();
        let sample = reservoir_sample(data.into_iter(), 10);
        assert_eq!(sample.len(), 10);
    }

    #[test]
    fn reservoir_sample_from_small_input() {
        let data = vec![1, 2, 3];
        let sample = reservoir_sample(data.into_iter(), 5);
        assert_eq!(sample.len(), 3); // cannot exceed input size
    }

    #[test]
    fn weighted_sample_without_replacement_unique() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample = weighted_sample_without_replacement(&weights, 3);
        assert_eq!(sample.len(), 3);
        // All indices should be valid and unique
        for idx in &sample {
            assert!(*idx < weights.len());
        }
        let mut sorted = sample.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "sample contained duplicates");
    }

    #[test]
    fn weighted_sample_with_replacement_respects_length() {
        let weights = vec![1.0, 1.0, 1.0];
        let sample = weighted_sample_with_replacement(&weights, 100);
        assert_eq!(sample.len(), 100);
        for idx in &sample {
            assert!(*idx < weights.len());
        }
    }

    #[test]
    fn weighted_sample_with_replacement_favors_heavy() {
        let weights = vec![0.0, 0.0, 1.0]; // only index 2 has weight
        let sample = weighted_sample_with_replacement(&weights, 50);
        assert!(sample.iter().all(|&i| i == 2));
    }
}
