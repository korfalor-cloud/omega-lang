/// Gaussian Mixture Model using EM algorithm.

/// Univariate Gaussian.
#[derive(Clone, Debug)]
pub struct Gaussian {
    pub mean: f64,
    pub variance: f64,
}

impl Gaussian {
    pub fn new(mean: f64, variance: f64) -> Self {
        Self { mean, variance }
    }

    pub fn pdf(&self, x: f64) -> f64 {
        let std = self.variance.sqrt();
        let z = (x - self.mean) / std;
        (-0.5 * z * z).exp() / (std * (2.0 * std::f64::consts::PI).sqrt())
    }

    pub fn log_pdf(&self, x: f64) -> f64 {
        let std = self.variance.sqrt();
        let z = (x - self.mean) / std;
        -0.5 * z * z - std.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
}

/// Multivariate Gaussian.
#[derive(Clone, Debug)]
pub struct MultivariateGaussian {
    pub mean: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub precision: Vec<Vec<f64>>,
    pub log_det: f64,
}

impl MultivariateGaussian {
    pub fn new(mean: Vec<f64>, covariance: Vec<Vec<f64>>) -> Self {
        let precision = mat_invert(&covariance).unwrap_or_else(|| {
            let n = mean.len();
            let mut m = vec![vec![0.0; n]; n];
            for i in 0..n { m[i][i] = 1.0; }
            m
        });
        let log_det = log_determinant(&covariance);

        Self { mean, covariance, precision, log_det }
    }

    pub fn pdf(&self, x: &[f64]) -> f64 {
        self.log_pdf(x).exp()
    }

    pub fn log_pdf(&self, x: &[f64]) -> f64 {
        let n = self.mean.len() as f64;
        let diff: Vec<f64> = x.iter().zip(self.mean.iter()).map(|(a, b)| a - b).collect();

        // Mahalanobis distance: diff^T * precision * diff
        let mut mahal = 0.0;
        for i in 0..diff.len() {
            for j in 0..diff.len() {
                mahal += diff[i] * self.precision[i][j] * diff[j];
            }
        }

        -0.5 * n * (2.0 * std::f64::consts::PI).ln() - 0.5 * self.log_det - 0.5 * mahal
    }

    pub fn sample(&self, seed: u64) -> Vec<f64> {
        let n = self.mean.len();
        let l = cholesky(&self.covariance);

        // Generate standard normal samples
        let mut rng = seed;
        let mut rand_normal = || -> f64 {
            let u1 = ((rng >> 33) as f64) / (1u64 << 31) as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = ((rng >> 33) as f64) / (1u64 << 31) as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let z: Vec<f64> = (0..n).map(|_| rand_normal()).collect();

        // x = mean + L * z
        (0..n).map(|i| {
            self.mean[i] + (0..=i).map(|j| l[i][j] * z[j]).sum::<f64>()
        }).collect()
    }
}

/// Gaussian Mixture Model.
pub struct GaussianMixtureModel {
    pub n_components: usize,
    pub weights: Vec<f64>,
    pub components: Vec<Gaussian>,
    pub converged: bool,
}

impl GaussianMixtureModel {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            weights: vec![1.0 / n_components as f64; n_components],
            components: Vec::new(),
            converged: false,
        }
    }

    /// Fit using EM algorithm.
    pub fn fit(&mut self, data: &[f64], max_iter: usize, tol: f64) {
        let n = data.len();
        let k = self.n_components;

        // Initialize components using k-means-like initialization
        self.components = self.initialize_components(data, k);

        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for iter in 0..max_iter {
            // E-step: compute responsibilities
            let mut responsibilities = vec![vec![0.0; k]; n];

            for i in 0..n {
                let mut total = 0.0;
                for j in 0..k {
                    responsibilities[i][j] = self.weights[j] * self.components[j].pdf(data[i]);
                    total += responsibilities[i][j];
                }
                if total > 0.0 {
                    for j in 0..k {
                        responsibilities[i][j] /= total;
                    }
                }
            }

            // M-step: update parameters
            let nk: Vec<f64> = (0..k).map(|j| {
                (0..n).map(|i| responsibilities[i][j]).sum()
            }).collect();

            for j in 0..k {
                if nk[j] < 1e-10 { continue; }

                // Update weight
                self.weights[j] = nk[j] / n as f64;

                // Update mean
                let new_mean: f64 = (0..n).map(|i| responsibilities[i][j] * data[i]).sum::<f64>() / nk[j];

                // Update variance
                let new_variance: f64 = (0..n).map(|i| {
                    responsibilities[i][j] * (data[i] - new_mean).powi(2)
                }).sum::<f64>() / nk[j];

                self.components[j] = Gaussian::new(new_mean, new_variance.max(1e-6));
            }

            // Compute log-likelihood
            let log_likelihood: f64 = data.iter().map(|&x| {
                let mut total = 0.0;
                for j in 0..k {
                    total += self.weights[j] * self.components[j].pdf(x);
                }
                total.max(1e-300).ln()
            }).sum();

            // Check convergence
            if (log_likelihood - prev_log_likelihood).abs() < tol {
                self.converged = true;
                break;
            }
            prev_log_likelihood = log_likelihood;
        }
    }

    /// Predict cluster assignments.
    pub fn predict(&self, data: &[f64]) -> Vec<usize> {
        data.iter().map(|&x| {
            (0..self.n_components)
                .max_by(|&a, &b| {
                    let pa = self.weights[a] * self.components[a].pdf(x);
                    let pb = self.weights[b] * self.components[b].pdf(x);
                    pa.partial_cmp(&pb).unwrap()
                })
                .unwrap()
        }).collect()
    }

    /// Compute log-likelihood.
    pub fn log_likelihood(&self, data: &[f64]) -> f64 {
        data.iter().map(|&x| {
            let mut total = 0.0;
            for j in 0..self.n_components {
                total += self.weights[j] * self.components[j].pdf(x);
            }
            total.max(1e-300).ln()
        }).sum()
    }

    /// BIC (Bayesian Information Criterion).
    pub fn bic(&self, data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let k = self.n_components as f64;
        let ll = self.log_likelihood(data);
        let n_params = 3.0 * k - 1.0; // mean, variance, weight per component
        -2.0 * ll + n_params * n.ln()
    }

    fn initialize_components(&self, data: &[f64], k: usize) -> Vec<Gaussian> {
        let n = data.len();
        let mut rng = 42u64;
        let mut rand = || -> usize {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as usize) % n
        };

        // Random initialization
        let mut means: Vec<f64> = Vec::new();
        while means.len() < k {
            let idx = rand();
            let val = data[idx];
            if !means.contains(&val) {
                means.push(val);
            }
        }

        let global_var: f64 = data.iter().map(|&x| (x - data.iter().sum::<f64>() / n as f64).powi(2)).sum::<f64>() / n as f64;

        means.into_iter().map(|m| Gaussian::new(m, global_var / k as f64)).collect()
    }
}

/// Multivariate GMM.
pub struct MultivariateGMM {
    pub n_components: usize,
    pub weights: Vec<f64>,
    pub components: Vec<MultivariateGaussian>,
}

impl MultivariateGMM {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            weights: vec![1.0 / n_components as f64; n_components],
            components: Vec::new(),
        }
    }

    pub fn fit(&mut self, data: &[Vec<f64>], max_iter: usize, tol: f64) {
        let n = data.len();
        let k = self.n_components;
        let dim = data[0].len();

        // Initialize
        self.components = self.initialize(data, k);

        let mut prev_ll = f64::NEG_INFINITY;

        for _ in 0..max_iter {
            // E-step
            let mut resp = vec![vec![0.0; k]; n];
            for i in 0..n {
                let mut total = 0.0;
                for j in 0..k {
                    resp[i][j] = self.weights[j] * self.components[j].pdf(&data[i]);
                    total += resp[i][j];
                }
                if total > 0.0 {
                    for j in 0..k { resp[i][j] /= total; }
                }
            }

            // M-step
            let nk: Vec<f64> = (0..k).map(|j| {
                (0..n).map(|i| resp[i][j]).sum()
            }).collect();

            for j in 0..k {
                if nk[j] < 1e-10 { continue; }

                self.weights[j] = nk[j] / n as f64;

                // Update mean
                let new_mean: Vec<f64> = (0..dim).map(|d| {
                    (0..n).map(|i| resp[i][j] * data[i][d]).sum::<f64>() / nk[j]
                }).collect();

                // Update covariance
                let mut new_cov = vec![vec![0.0; dim]; dim];
                for i in 0..n {
                    let diff: Vec<f64> = data[i].iter().zip(new_mean.iter()).map(|(a, b)| a - b).collect();
                    for d1 in 0..dim {
                        for d2 in 0..dim {
                            new_cov[d1][d2] += resp[i][j] * diff[d1] * diff[d2];
                        }
                    }
                }
                for d1 in 0..dim {
                    for d2 in 0..dim {
                        new_cov[d1][d2] /= nk[j];
                    }
                }
                // Add regularization
                for d in 0..dim {
                    new_cov[d][d] += 1e-6;
                }

                self.components[j] = MultivariateGaussian::new(new_mean, new_cov);
            }

            // Check convergence
            let ll: f64 = data.iter().map(|x| {
                let mut total = 0.0;
                for j in 0..k {
                    total += self.weights[j] * self.components[j].pdf(x);
                }
                total.max(1e-300).ln()
            }).sum();

            if (ll - prev_ll).abs() < tol { break; }
            prev_ll = ll;
        }
    }

    fn initialize(&self, data: &[Vec<f64>], k: usize) -> Vec<MultivariateGaussian> {
        let n = data.len();
        let dim = data[0].len();
        let mut rng = 42u64;

        let mut indices: Vec<usize> = Vec::new();
        while indices.len() < k {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = ((rng >> 33) as usize) % n;
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }

        indices.into_iter().map(|idx| {
            let mean = data[idx].clone();
            let mut cov = vec![vec![0.0; dim]; dim];
            for d in 0..dim { cov[d][d] = 1.0; }
            MultivariateGaussian::new(mean, cov)
        }).collect()
    }
}

fn cholesky(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j { sum += l[i][k] * l[j][k]; }
            if i == j {
                l[i][j] = (m[i][i] - sum).sqrt();
            } else {
                l[i][j] = (m[i][j] - sum) / l[j][j];
            }
        }
    }
    l
}

fn log_determinant(m: &[Vec<f64>]) -> f64 {
    let l = cholesky(m);
    2.0 * (0..m.len()).map(|i| l[i][i].ln()).sum::<f64>()
}

fn mat_invert(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n { aug[i][j] = m[i][j]; }
        aug[i][n + i] = 1.0;
    }
    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[max_row][col].abs() { max_row = row; }
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
    fn test_gmm() {
        let mut gmm = GaussianMixtureModel::new(2);
        let data: Vec<f64> = (0..50).map(|i| if i < 25 { i as f64 * 0.1 } else { 10.0 + i as f64 * 0.1 }).collect();
        gmm.fit(&data, 100, 1e-6);

        let predictions = gmm.predict(&data);
        assert_eq!(predictions.len(), data.len());
        // Should separate the two clusters
        assert_ne!(predictions[0], predictions[49]);
    }

    #[test]
    fn test_gaussian() {
        let g = Gaussian::new(0.0, 1.0);
        assert!(g.pdf(0.0) > g.pdf(1.0));
        assert!(g.log_pdf(0.0) > g.log_pdf(1.0));
    }
}
