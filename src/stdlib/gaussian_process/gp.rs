/// Gaussian Process regression and classification.

use std::f64::consts::PI;

/// Kernel (covariance) functions.
pub trait Kernel: Clone {
    fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64;
}

/// Squared Exponential (RBF) kernel.
#[derive(Clone)]
pub struct SquaredExponential {
    pub length_scale: f64,
    pub variance: f64,
}

impl SquaredExponential {
    pub fn new(length_scale: f64, variance: f64) -> Self {
        Self { length_scale, variance }
    }
}

impl Kernel for SquaredExponential {
    fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dist_sq: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        self.variance * (-0.5 * dist_sq / (self.length_scale * self.length_scale)).exp()
    }
}

/// Matérn 3/2 kernel.
#[derive(Clone)]
pub struct Matern32 {
    pub length_scale: f64,
    pub variance: f64,
}

impl Matern32 {
    pub fn new(length_scale: f64, variance: f64) -> Self {
        Self { length_scale, variance }
    }
}

impl Kernel for Matern32 {
    fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dist: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let r = (3.0_f64).sqrt() * dist / self.length_scale;
        self.variance * (1.0 + r) * (-r).exp()
    }
}

/// Matérn 5/2 kernel.
#[derive(Clone)]
pub struct Matern52 {
    pub length_scale: f64,
    pub variance: f64,
}

impl Matern52 {
    pub fn new(length_scale: f64, variance: f64) -> Self {
        Self { length_scale, variance }
    }
}

impl Kernel for Matern52 {
    fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dist: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let r = (5.0_f64).sqrt() * dist / self.length_scale;
        self.variance * (1.0 + r + r * r / 3.0) * (-r).exp()
    }
}

/// Rational Quadratic kernel.
#[derive(Clone)]
pub struct RationalQuadratic {
    pub length_scale: f64,
    pub variance: f64,
    pub alpha: f64,
}

impl RationalQuadratic {
    pub fn new(length_scale: f64, variance: f64, alpha: f64) -> Self {
        Self { length_scale, variance, alpha }
    }
}

impl Kernel for RationalQuadratic {
    fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dist_sq: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        self.variance * (1.0 + dist_sq / (2.0 * self.alpha * self.length_scale * self.length_scale)).powf(-self.alpha)
    }
}

/// Periodic kernel.
#[derive(Clone)]
pub struct PeriodicKernel {
    pub length_scale: f64,
    pub variance: f64,
    pub period: f64,
}

impl PeriodicKernel {
    pub fn new(length_scale: f64, variance: f64, period: f64) -> Self {
        Self { length_scale, variance, period }
    }
}

impl Kernel for PeriodicKernel {
    fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dist: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let sin_val = (PI * dist / self.period).sin();
        self.variance * (-2.0 * sin_val * sin_val / (self.length_scale * self.length_scale)).exp()
    }
}

/// Linear kernel.
#[derive(Clone)]
pub struct LinearKernel {
    pub variance: f64,
    pub offset: f64,
}

impl LinearKernel {
    pub fn new(variance: f64, offset: f64) -> Self {
        Self { variance, offset }
    }
}

impl Kernel for LinearKernel {
    fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dot: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
        self.variance * (dot + self.offset)
    }
}

/// Gaussian Process regression.
pub struct GaussianProcess<K: Kernel> {
    pub kernel: K,
    pub noise: f64,
    pub x_train: Vec<Vec<f64>>,
    pub y_train: Vec<f64>,
    pub k_inv: Option<Vec<Vec<f64>>>,
    pub alpha: Option<Vec<f64>>,
}

impl<K: Kernel> GaussianProcess<K> {
    pub fn new(kernel: K, noise: f64) -> Self {
        Self {
            kernel,
            noise,
            x_train: Vec::new(),
            y_train: Vec::new(),
            k_inv: None,
            alpha: None,
        }
    }

    /// Fit the GP to training data.
    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) {
        let n = x.len();
        assert_eq!(n, y.len());

        // Compute kernel matrix with noise
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = self.kernel.evaluate(&x[i], &x[j]);
                if i == j {
                    k[i][j] += self.noise * self.noise;
                }
            }
        }

        // Cholesky decomposition
        let l = cholesky(&k);

        // Solve L * alpha = y
        let alpha = forward_substitution(&l, &y);

        // Solve L^T * v = alpha
        let alpha = backward_substitution(&transpose(&l), &alpha);

        self.x_train = x;
        self.y_train = y;
        self.k_inv = Some(mat_invert(&k).unwrap_or_else(|| vec![vec![0.0; n]; n]));
        self.alpha = Some(alpha);
    }

    /// Predict mean and variance at new points.
    pub fn predict(&self, x_test: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
        let n_train = self.x_train.len();
        let n_test = x_test.len();

        let alpha = self.alpha.as_ref().expect("GP not fitted");

        // Compute cross-covariance K_star
        let mut k_star = vec![vec![0.0; n_train]; n_test];
        for i in 0..n_test {
            for j in 0..n_train {
                k_star[i][j] = self.kernel.evaluate(&x_test[i], &self.x_train[j]);
            }
        }

        // Predictive mean: K_star * alpha
        let mean: Vec<f64> = k_star.iter()
            .map(|row| row.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum())
            .collect();

        // Predictive variance: K_star_star - K_star * K_inv * K_star^T
        let k_inv = self.k_inv.as_ref().expect("GP not fitted");
        let variance: Vec<f64> = (0..n_test).map(|i| {
            let k_star_star = self.kernel.evaluate(&x_test[i], &x_test[i]);
            let k_star_k_inv: Vec<f64> = (0..n_train)
                .map(|j| (0..n_train).map(|l| k_star[i][l] * k_inv[l][j]).sum())
                .collect();
            let var = k_star_star - k_star_k_inv.iter().zip(k_star[i].iter()).map(|(a, b)| a * b).sum::<f64>();
            var.max(0.0)
        }).collect();

        (mean, variance)
    }

    /// Log marginal likelihood.
    pub fn log_likelihood(&self) -> f64 {
        let n = self.y_train.len();
        let alpha = self.alpha.as_ref().expect("GP not fitted");
        let k_inv = self.k_inv.as_ref().expect("GP not fitted");

        let data_fit: f64 = self.y_train.iter().zip(alpha.iter()).map(|(y, a)| y * a).sum();
        let log_det = (0..n).map(|i| k_inv[i][i].ln()).sum::<f64>();

        -0.5 * data_fit - 0.5 * log_det - 0.5 * n as f64 * (2.0 * PI).ln()
    }
}

/// GP Classification using Laplace approximation.
pub struct GPClassification<K: Kernel> {
    pub kernel: K,
    pub x_train: Vec<Vec<f64>>,
    pub y_train: Vec<f64>, // +1 or -1
    pub f: Vec<f64>,        // Latent function values
}

impl<K: Kernel> GPClassification<K> {
    pub fn new(kernel: K) -> Self {
        Self {
            kernel,
            x_train: Vec::new(),
            y_train: Vec::new(),
            f: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>, max_iter: usize) {
        let n = x.len();

        // Compute kernel matrix
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = self.kernel.evaluate(&x[i], &x[j]);
            }
        }

        // Initialize f
        self.f = vec![0.0; n];

        for _ in 0..max_iter {
            // Compute W and gradient
            let pi: Vec<f64> = self.f.iter().map(|&f| sigmoid(f)).collect();
            let w: Vec<f64> = pi.iter().map(|&p| p * (1.0 - p)).collect();
            let grad: Vec<f64> = y.iter().zip(pi.iter()).map(|(&y, &p)| y - p).collect();

            // Newton update: f_new = K * (I + W * K)^-1 * (W * f + grad)
            let wk: Vec<Vec<f64>> = (0..n).map(|i| {
                (0..n).map(|j| w[i] * k[i][j]).collect()
            }).collect();

            let mut iwk = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    iwk[i][j] = if i == j { 1.0 + wk[i][j] } else { wk[i][j] };
                }
            }

            if let Some(iwk_inv) = mat_invert(&iwk) {
                let wf_grad: Vec<f64> = (0..n).map(|i| w[i] * self.f[i] + grad[i]).collect();
                let update: Vec<f64> = (0..n).map(|i| {
                    (0..n).map(|j| k[i][j] * iwk_inv[j].iter().zip(wf_grad.iter()).map(|(a, b)| a * b).sum::<f64>()).sum()
                }).collect();

                self.f = update;
            }
        }

        self.x_train = x;
        self.y_train = y;
    }

    pub fn predict(&self, x_test: &[Vec<f64>]) -> Vec<f64> {
        let n_train = self.x_train.len();
        let pi: Vec<f64> = self.f.iter().map(|&f| sigmoid(f)).collect();

        x_test.iter().map(|x| {
            let k_star: Vec<f64> = self.x_train.iter()
                .map(|x_train| self.kernel.evaluate(x, x_train))
                .collect();

            let f_star: f64 = k_star.iter().zip(pi.iter()).map(|(k, p)| k * (2.0 * p - 1.0)).sum();
            sigmoid(f_star)
        }).collect()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn cholesky(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }

            if i == j {
                l[i][j] = (m[i][i] - sum).sqrt();
            } else {
                l[i][j] = (m[i][j] - sum) / l[j][j];
            }
        }
    }

    l
}

fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        x[i] = b[i];
        for j in 0..i {
            x[i] -= l[i][j] * x[j];
        }
        x[i] /= l[i][i];
    }
    x
}

fn backward_substitution(u: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in (i + 1)..n {
            x[i] -= u[i][j] * x[j];
        }
        x[i] /= u[i][i];
    }
    x
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = m.len();
    let cols = m[0].len();
    let mut t = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = m[i][j];
        }
    }
    t
}

fn mat_invert(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n { aug[i][j] = matrix[i][j]; }
        aug[i][n + i] = 1.0;
    }

    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-10 { return None; }

        for j in 0..(2 * n) { aug[col][j] /= pivot; }

        for row in 0..n {
            if row == col { continue; }
            let factor = aug[row][col];
            for j in 0..(2 * n) { aug[row][j] -= factor * aug[col][j]; }
        }
    }

    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n { inv[i][j] = aug[i][n + j]; }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gp_regression() {
        let kernel = SquaredExponential::new(1.0, 1.0);
        let mut gp = GaussianProcess::new(kernel, 0.1);

        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![0.0, 1.0, 4.0, 9.0];

        gp.fit(x, y);

        let x_test = vec![vec![1.5], vec![2.5]];
        let (mean, variance) = gp.predict(&x_test);

        assert_eq!(mean.len(), 2);
        assert_eq!(variance.len(), 2);
        // Predictions should be reasonable
        assert!(mean[0] > 0.0 && mean[0] < 4.0);
        assert!(variance[0] >= 0.0);
    }

    #[test]
    fn test_gp_log_likelihood() {
        let kernel = SquaredExponential::new(1.0, 1.0);
        let mut gp = GaussianProcess::new(kernel, 0.1);

        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0, 4.0];

        gp.fit(x, y);
        let ll = gp.log_likelihood();
        assert!(ll.is_finite());
    }
}
