/// Statistical distributions: PDF, CDF, sampling, parameter estimation.

use std::f64::consts::PI;

/// Normal (Gaussian) distribution.
#[derive(Debug, Clone)]
pub struct Normal {
    pub mean: f64,
    pub std_dev: f64,
}

impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> Self { Self { mean, std_dev } }

    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        (-0.5 * z * z).exp() / (self.std_dev * (2.0 * PI).sqrt())
    }

    pub fn cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + erf((x - self.mean) / (self.std_dev * 2.0_f64.sqrt())))
    }

    pub fn quantile(&self, p: f64) -> f64 {
        self.mean + self.std_dev * normal_quantile(p)
    }

    pub fn sample(&self, seed: &mut u64) -> f64 {
        self.mean + self.std_dev * box_muller(seed)
    }

    pub fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        -0.5 * z * z - (self.std_dev * (2.0 * PI).sqrt()).ln()
    }
}

/// Exponential distribution.
#[derive(Debug, Clone)]
pub struct Exponential {
    pub rate: f64,
}

impl Exponential {
    pub fn new(rate: f64) -> Self { Self { rate } }

    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { self.rate * (-self.rate * x).exp() }
    }

    pub fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { 1.0 - (-self.rate * x).exp() }
    }

    pub fn quantile(&self, p: f64) -> f64 {
        -(1.0 - p).ln() / self.rate
    }

    pub fn sample(&self, seed: &mut u64) -> f64 {
        -pseudo_rand(seed).ln() / self.rate
    }

    pub fn mean(&self) -> f64 { 1.0 / self.rate }
    pub fn variance(&self) -> f64 { 1.0 / (self.rate * self.rate) }
}

/// Poisson distribution.
#[derive(Debug, Clone)]
pub struct Poisson {
    pub lambda: f64,
}

impl Poisson {
    pub fn new(lambda: f64) -> Self { Self { lambda } }

    pub fn pmf(&self, k: u32) -> f64 {
        (-self.lambda).exp() * self.lambda.powi(k as i32) / factorial(k) as f64
    }

    pub fn cdf(&self, k: u32) -> f64 {
        (0..=k).map(|i| self.pmf(i)).sum()
    }

    pub fn sample(&self, seed: &mut u64) -> u32 {
        let l = (-self.lambda).exp();
        let mut k = 0u32;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= pseudo_rand(seed);
            if p < l {
                return k - 1;
            }
        }
    }

    pub fn mean(&self) -> f64 { self.lambda }
    pub fn variance(&self) -> f64 { self.lambda }
}

/// Binomial distribution.
#[derive(Debug, Clone)]
pub struct Binomial {
    pub n: u32,
    pub p: f64,
}

impl Binomial {
    pub fn new(n: u32, p: f64) -> Self { Self { n, p } }

    pub fn pmf(&self, k: u32) -> f64 {
        binomial_coeff(self.n, k) as f64 * self.p.powi(k as i32) * (1.0 - self.p).powi((self.n - k) as i32)
    }

    pub fn cdf(&self, k: u32) -> f64 {
        (0..=k).map(|i| self.pmf(i)).sum()
    }

    pub fn sample(&self, seed: &mut u64) -> u32 {
        (0..self.n).filter(|_| pseudo_rand(seed) < self.p).count() as u32
    }

    pub fn mean(&self) -> f64 { self.n as f64 * self.p }
    pub fn variance(&self) -> f64 { self.n as f64 * self.p * (1.0 - self.p) }
}

/// Beta distribution.
#[derive(Debug, Clone)]
pub struct Beta {
    pub alpha: f64,
    pub beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> Self { Self { alpha, beta } }

    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 { return 0.0; }
        let b = beta_func(self.alpha, self.beta);
        x.powf(self.alpha - 1.0) * (1.0 - x).powf(self.beta - 1.0) / b
    }

    pub fn mean(&self) -> f64 { self.alpha / (self.alpha + self.beta) }
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        self.alpha * self.beta / (ab * ab * (ab + 1.0))
    }

    pub fn sample(&self, seed: &mut u64) -> f64 {
        let x = gamma_sample(self.alpha, seed);
        let y = gamma_sample(self.beta, seed);
        x / (x + y)
    }
}

/// Gamma distribution.
#[derive(Debug, Clone)]
pub struct Gamma {
    pub shape: f64,
    pub scale: f64,
}

impl Gamma {
    pub fn new(shape: f64, scale: f64) -> Self { Self { shape, scale } }

    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        let k = self.shape;
        let theta = self.scale;
        x.powf(k - 1.0) * (-x / theta).exp() / (theta.powf(k) * gamma_func(k))
    }

    pub fn mean(&self) -> f64 { self.shape * self.scale }
    pub fn variance(&self) -> f64 { self.shape * self.scale * self.scale }

    pub fn sample(&self, seed: &mut u64) -> f64 {
        gamma_sample(self.shape, seed) * self.scale
    }
}

/// Student's t-distribution.
#[derive(Debug, Clone)]
pub struct StudentT {
    pub df: f64,
}

impl StudentT {
    pub fn new(df: f64) -> Self { Self { df } }

    pub fn pdf(&self, x: f64) -> f64 {
        let v = self.df;
        gamma_func((v + 1.0) / 2.0) / ((v * PI).sqrt() * gamma_func(v / 2.0))
            * (1.0 + x * x / v).powf(-(v + 1.0) / 2.0)
    }

    pub fn cdf(&self, x: f64) -> f64 {
        let v = self.df;
        let t = x / v.sqrt();
        let p = 0.5 + t * gamma_func((v + 1.0) / 2.0) / ((v * PI).sqrt() * gamma_func(v / 2.0));
        p.clamp(0.0, 1.0) // Approximation
    }

    pub fn mean(&self) -> f64 { if self.df > 1.0 { 0.0 } else { f64::NAN } }
    pub fn variance(&self) -> f64 {
        if self.df > 2.0 { self.df / (self.df - 2.0) } else { f64::INFINITY }
    }
}

/// Chi-squared distribution.
#[derive(Debug, Clone)]
pub struct ChiSquared {
    pub df: f64,
}

impl ChiSquared {
    pub fn new(df: f64) -> Self { Self { df } }

    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        let k = self.df;
        x.powf(k / 2.0 - 1.0) * (-x / 2.0).exp() / (2.0_f64.powf(k / 2.0) * gamma_func(k / 2.0))
    }

    pub fn mean(&self) -> f64 { self.df }
    pub fn variance(&self) -> f64 { 2.0 * self.df }

    pub fn sample(&self, seed: &mut u64) -> f64 {
        2.0 * gamma_sample(self.df / 2.0, seed)
    }
}

/// Log-normal distribution.
#[derive(Debug, Clone)]
pub struct LogNormal {
    pub mu: f64,
    pub sigma: f64,
}

impl LogNormal {
    pub fn new(mu: f64, sigma: f64) -> Self { Self { mu, sigma } }

    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        let z = (x.ln() - self.mu) / self.sigma;
        (-0.5 * z * z).exp() / (x * self.sigma * (2.0 * PI).sqrt())
    }

    pub fn mean(&self) -> f64 { (self.mu + self.sigma * self.sigma / 2.0).exp() }
    pub fn variance(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        ((s2 - 1.0).exp() * (2.0 * self.mu + s2).exp()) - (2.0 * self.mu + s2).exp()
    }

    pub fn sample(&self, seed: &mut u64) -> f64 {
        (self.mu + self.sigma * box_muller(seed)).exp()
    }
}

/// Weibull distribution.
#[derive(Debug, Clone)]
pub struct Weibull {
    pub shape: f64,
    pub scale: f64,
}

impl Weibull {
    pub fn new(shape: f64, scale: f64) -> Self { Self { shape, scale } }

    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { return 0.0; }
        let k = self.shape;
        let lambda = self.scale;
        (k / lambda) * (x / lambda).powf(k - 1.0) * (-(x / lambda).powf(k)).exp()
    }

    pub fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 { return 0.0; }
        1.0 - (-(x / self.scale).powf(self.shape)).exp()
    }

    pub fn mean(&self) -> f64 {
        self.scale * gamma_func(1.0 + 1.0 / self.shape)
    }
}

/// Uniform distribution.
#[derive(Debug, Clone)]
pub struct Uniform {
    pub lo: f64,
    pub hi: f64,
}

impl Uniform {
    pub fn new(lo: f64, hi: f64) -> Self { Self { lo, hi } }

    pub fn pdf(&self, x: f64) -> f64 {
        if x >= self.lo && x <= self.hi { 1.0 / (self.hi - self.lo) } else { 0.0 }
    }

    pub fn cdf(&self, x: f64) -> f64 {
        if x < self.lo { 0.0 } else if x > self.hi { 1.0 } else { (x - self.lo) / (self.hi - self.lo) }
    }

    pub fn sample(&self, seed: &mut u64) -> f64 {
        self.lo + pseudo_rand(seed) * (self.hi - self.lo)
    }

    pub fn mean(&self) -> f64 { (self.lo + self.hi) / 2.0 }
    pub fn variance(&self) -> f64 { (self.hi - self.lo).powi(2) / 12.0 }
}

/// Mixture model.
pub struct MixtureModel {
    pub weights: Vec<f64>,
    pub components: Vec<Normal>,
}

impl MixtureModel {
    pub fn new(weights: Vec<f64>, components: Vec<Normal>) -> Self {
        Self { weights, components }
    }

    pub fn pdf(&self, x: f64) -> f64 {
        self.weights.iter().zip(self.components.iter())
            .map(|(w, c)| w * c.pdf(x))
            .sum()
    }

    pub fn sample(&self, seed: &mut u64) -> f64 {
        let r = pseudo_rand(seed);
        let mut cumulative = 0.0;
        for (w, c) in self.weights.iter().zip(self.components.iter()) {
            cumulative += w;
            if r < cumulative {
                return c.sample(seed);
            }
        }
        self.components.last().unwrap().sample(seed)
    }

    /// EM algorithm for parameter estimation.
    pub fn fit(data: &[f64], k: usize, max_iter: usize) -> Self {
        let n = data.len();
        let mut seed: u64 = 42;

        // Initialize with k-means-like approach
        let mut means: Vec<f64> = (0..k).map(|i| data[i * n / k]).collect();
        let mut stds = vec![1.0; k];
        let mut weights = vec![1.0 / k as f64; k];
        let mut responsibilities = vec![vec![0.0; k]; n];

        for _ in 0..max_iter {
            // E-step
            for i in 0..n {
                let total: f64 = (0..k).map(|j| {
                    weights[j] * normal_pdf(data[i], means[j], stds[j])
                }).sum();
                for j in 0..k {
                    responsibilities[i][j] = if total > 0.0 {
                        weights[j] * normal_pdf(data[i], means[j], stds[j]) / total
                    } else {
                        1.0 / k as f64
                    };
                }
            }

            // M-step
            for j in 0..k {
                let nk: f64 = responsibilities.iter().map(|r| r[j]).sum();
                if nk < 1e-10 { continue; }
                means[j] = data.iter().enumerate().map(|(i, &x)| responsibilities[i][j] * x).sum::<f64>() / nk;
                stds[j] = (data.iter().enumerate()
                    .map(|(i, &x)| responsibilities[i][j] * (x - means[j]).powi(2))
                    .sum::<f64>() / nk).sqrt().max(0.01);
                weights[j] = nk / n as f64;
            }
        }

        let components: Vec<Normal> = means.into_iter().zip(stds.into_iter())
            .map(|(m, s)| Normal::new(m, s))
            .collect();
        Self::new(weights, components)
    }
}

fn normal_pdf(x: f64, mean: f64, std: f64) -> f64 {
    let z = (x - mean) / std;
    (-0.5 * z * z).exp() / (std * (2.0 * PI).sqrt())
}

/// Descriptive statistics.
pub struct DescriptiveStats {
    pub data: Vec<f64>,
}

impl DescriptiveStats {
    pub fn new(data: Vec<f64>) -> Self { Self { data } }

    pub fn mean(&self) -> f64 {
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    pub fn variance(&self) -> f64 {
        let m = self.mean();
        self.data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / self.data.len() as f64
    }

    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }

    pub fn skewness(&self) -> f64 {
        let m = self.mean();
        let s = self.std_dev();
        let n = self.data.len() as f64;
        self.data.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / n
    }

    pub fn kurtosis(&self) -> f64 {
        let m = self.mean();
        let s = self.std_dev();
        let n = self.data.len() as f64;
        self.data.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / n - 3.0
    }

    pub fn median(&self) -> f64 {
        let mut sorted = self.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
    }

    pub fn percentile(&self, p: f64) -> f64 {
        let mut sorted = self.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (p / 100.0 * (sorted.len() - 1) as f64) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    pub fn mode(&self) -> f64 {
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for &x in &self.data {
            let bucket = (x * 100.0) as i64;
            *counts.entry(bucket).or_insert(0) += 1;
        }
        counts.into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, _)| k as f64 / 100.0)
            .unwrap_or(0.0)
    }

    pub fn covariance(&self, other: &DescriptiveStats) -> f64 {
        let m1 = self.mean();
        let m2 = other.mean();
        self.data.iter().zip(other.data.iter())
            .map(|(x, y)| (x - m1) * (y - m2))
            .sum::<f64>() / self.data.len() as f64
    }

    pub fn correlation(&self, other: &DescriptiveStats) -> f64 {
        self.covariance(other) / (self.std_dev() * other.std_dev())
    }

    pub fn entropy(&self, bins: usize) -> f64 {
        let min = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        if range < 1e-10 { return 0.0; }

        let mut counts = vec![0usize; bins];
        for &x in &self.data {
            let bin = ((x - min) / range * bins as f64).min(bins as f64 - 1.0) as usize;
            counts[bin] += 1;
        }

        let n = self.data.len() as f64;
        -counts.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                p * p.ln()
            })
            .sum::<f64>()
    }
}

/// Hypothesis testing.
pub struct HypothesisTest;

impl HypothesisTest {
    /// Two-sample t-test.
    pub fn t_test(sample1: &[f64], sample2: &[f64]) -> (f64, f64) {
        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;
        let m1 = sample1.iter().sum::<f64>() / n1;
        let m2 = sample2.iter().sum::<f64>() / n2;
        let v1 = sample1.iter().map(|x| (x - m1).powi(2)).sum::<f64>() / (n1 - 1.0);
        let v2 = sample2.iter().map(|x| (x - m2).powi(2)).sum::<f64>() / (n2 - 1.0);

        let se = (v1 / n1 + v2 / n2).sqrt();
        let t = (m1 - m2) / se;

        // Welch-Satterthwaite degrees of freedom
        let df = (v1 / n1 + v2 / n2).powi(2)
            / ((v1 / n1).powi(2) / (n1 - 1.0) + (v2 / n2).powi(2) / (n2 - 1.0));

        (t, df)
    }

    /// Chi-squared test for independence.
    pub fn chi_squared_test(observed: &[Vec<u32>]) -> f64 {
        let rows = observed.len();
        let cols = observed[0].len();
        let total: u32 = observed.iter().flat_map(|r| r.iter()).sum();
        let row_sums: Vec<u32> = observed.iter().map(|r| r.iter().sum()).collect();
        let col_sums: Vec<u32> = (0..cols).map(|c| observed.iter().map(|r| r[c]).sum()).collect();

        let mut chi2 = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                let expected = row_sums[i] as f64 * col_sums[j] as f64 / total as f64;
                chi2 += (observed[i][j] as f64 - expected).powi(2) / expected;
            }
        }
        chi2
    }

    /// Kolmogorov-Smirnov test statistic.
    pub fn ks_test(sample: &[f64], cdf: impl Fn(f64) -> f64) -> f64 {
        let mut sorted = sample.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len() as f64;

        let mut max_diff = 0.0;
        for (i, &x) in sorted.iter().enumerate() {
            let emp = (i + 1) as f64 / n;
            let theoretical = cdf(x);
            max_diff = max_diff.max((emp - theoretical).abs());
        }
        max_diff
    }
}

// Helper functions

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn normal_quantile(p: f64) -> f64 {
    // Rational approximation (Abramowitz and Stegun)
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if (p - 0.5).abs() < 1e-10 { return 0.0; }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -z } else { z }
}

fn factorial(n: u32) -> u64 {
    (1..=n as u64).product()
}

fn binomial_coeff(n: u32, k: u32) -> u64 {
    if k > n { return 0; }
    let k = k.min(n - k);
    let mut result = 1u64;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

fn beta_func(a: f64, b: f64) -> f64 {
    gamma_func(a) * gamma_func(b) / gamma_func(a + b)
}

fn gamma_func(x: f64) -> f64 {
    // Stirling approximation with Lanczos coefficients
    if x < 0.5 {
        return PI / ((PI * x).sin() * gamma_func(1.0 - x));
    }
    let x = x - 1.0;
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let mut sum = c[0];
    for i in 1..(g as usize + 2) {
        sum += c[i] / (x + i as f64);
    }

    let t = x + g + 0.5;
    (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
}

fn gamma_sample(shape: f64, seed: &mut u64) -> f64 {
    // Marsaglia and Tsang's method
    if shape < 1.0 {
        return gamma_sample(shape + 1.0, seed) * pseudo_rand(seed).powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let mut x;
        let mut v;
        loop {
            x = box_muller(seed);
            v = 1.0 + c * x;
            if v > 0.0 { break; }
        }
        v = v * v * v;
        let u = pseudo_rand(seed);
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

fn box_muller(seed: &mut u64) -> f64 {
    let u1 = pseudo_rand(seed).max(1e-10);
    let u2 = pseudo_rand(seed);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn pseudo_rand(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / (1u64 << 31) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal() {
        let n = Normal::new(0.0, 1.0);
        assert!((n.pdf(0.0) - 0.3989).abs() < 0.001);
        assert!((n.cdf(0.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_exponential() {
        let e = Exponential::new(1.0);
        assert!((e.mean() - 1.0).abs() < 0.001);
        assert!((e.cdf(1.0) - 0.6321).abs() < 0.01);
    }

    #[test]
    fn test_poisson() {
        let p = Poisson::new(3.0);
        assert!((p.mean() - 3.0).abs() < 0.001);
        assert!((p.pmf(0) - 0.0498).abs() < 0.01);
    }

    #[test]
    fn test_descriptive() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = DescriptiveStats::new(data);
        assert!((stats.mean() - 3.0).abs() < 0.001);
        assert!((stats.median() - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_mixture() {
        let data: Vec<f64> = (0..100).map(|i| if i < 50 { i as f64 * 0.1 } else { 10.0 + i as f64 * 0.1 }).collect();
        let model = MixtureModel::fit(&data, 2, 50);
        assert!(model.weights.len() == 2);
    }

    #[test]
    fn test_t_test() {
        let s1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let (t, _) = HypothesisTest::t_test(&s1, &s2);
        assert!(t < 0.0); // s1 < s2
    }
}
