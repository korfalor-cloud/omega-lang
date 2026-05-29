/// Variational inference: mean-field VI, ELBO, variational autoencoders.

/// Variational distribution family.
pub trait VariationalFamily {
    fn sample(&self, seed: u64) -> f64;
    fn log_prob(&self, x: f64) -> f64;
    fn entropy(&self) -> f64;
}

/// Diagonal Gaussian variational family.
#[derive(Clone)]
pub struct DiagonalGaussian {
    pub mean: Vec<f64>,
    pub log_var: Vec<f64>,
}

impl DiagonalGaussian {
    pub fn new(mean: Vec<f64>, log_var: Vec<f64>) -> Self {
        Self { mean, log_var }
    }

    pub fn sample(&self, seed: u64) -> Vec<f64> {
        let mut rng = seed;
        self.mean.iter().zip(self.log_var.iter()).map(|(&m, &lv)| {
            let u1 = ((rng >> 33) as f64) / (1u64 << 31) as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = ((rng >> 33) as f64) / (1u64 << 31) as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let z = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            m + (lv / 2.0).exp() * z
        }).collect()
    }

    pub fn log_prob(&self, x: &[f64]) -> f64 {
        self.mean.iter().zip(self.log_var.iter()).zip(x.iter())
            .map(|((&m, &lv), &x)| {
                let var = lv.exp();
                let z = (x - m) / var.sqrt();
                -0.5 * z * z - 0.5 * lv - 0.5 * (2.0 * std::f64::consts::PI).ln()
            })
            .sum()
    }

    pub fn entropy(&self) -> f64 {
        self.log_var.iter().map(|&lv| {
            0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln() + 0.5 * lv
        }).sum()
    }

    pub fn kl_divergence(&self, prior_mean: &[f64], prior_log_var: &[f64]) -> f64 {
        self.mean.iter().zip(self.log_var.iter())
            .zip(prior_mean.iter().zip(prior_log_var.iter()))
            .map(|((&q_m, &q_lv), (&p_m, &p_lv))| {
                let q_var = q_lv.exp();
                let p_var = p_lv.exp();
                0.5 * (p_lv - q_lv + (q_var + (q_m - p_m).powi(2)) / p_var - 1.0)
            })
            .sum()
    }
}

/// Mean-field Variational Inference for Bayesian linear regression.
pub struct BayesianLinearVI {
    pub n_features: usize,
    pub weight_mean: Vec<f64>,
    pub weight_log_var: Vec<f64>,
    pub prior_mean: Vec<f64>,
    pub prior_log_var: Vec<f64>,
    pub noise_log_var: f64,
}

impl BayesianLinearVI {
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            weight_mean: vec![0.0; n_features],
            weight_log_var: vec![0.0; n_features],
            prior_mean: vec![0.0; n_features],
            prior_log_var: vec![0.0; n_features],
            noise_log_var: 0.0,
        }
    }

    /// Compute ELBO (Evidence Lower Bound).
    pub fn elbo(&self, x: &[Vec<f64>], y: &[f64], n_samples: usize) -> f64 {
        let n = y.len();
        let mut rng = 42u64;

        // Expected log-likelihood
        let mut expected_ll = 0.0;
        for _ in 0..n_samples {
            // Sample weights
            let w = DiagonalGaussian::new(self.weight_mean.clone(), self.weight_log_var.clone()).sample(rng);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);

            for (xi, &yi) in x.iter().zip(y.iter()) {
                let pred: f64 = w.iter().zip(xi.iter()).map(|(wi, xi)| wi * xi).sum();
                let noise_var = self.noise_log_var.exp();
                let residual = yi - pred;
                expected_ll += -0.5 * residual * residual / noise_var - 0.5 * self.noise_log_var;
            }
        }
        expected_ll /= n_samples as f64;

        // KL divergence
        let kl = DiagonalGaussian::new(self.weight_mean.clone(), self.weight_log_var.clone())
            .kl_divergence(&self.prior_mean, &self.prior_log_var);

        expected_ll - kl
    }

    /// Update variational parameters using gradient ascent (simplified).
    pub fn update(&mut self, x: &[Vec<f64>], y: &[f64], learning_rate: f64, n_samples: usize) {
        let n = y.len();
        let mut rng = 42u64;

        // Compute gradients (simplified)
        let mut grad_mean = vec![0.0; self.n_features];
        let mut grad_log_var = vec![0.0; self.n_features];

        for _ in 0..n_samples {
            let w = DiagonalGaussian::new(self.weight_mean.clone(), self.weight_log_var.clone()).sample(rng);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);

            for (xi, &yi) in x.iter().zip(y.iter()) {
                let pred: f64 = w.iter().zip(xi.iter()).map(|(wi, xi)| wi * xi).sum();
                let residual = yi - pred;
                let noise_var = self.noise_log_var.exp();

                for j in 0..self.n_features {
                    grad_mean[j] += residual * xi[j] / noise_var;
                    grad_log_var[j] += 0.5 * (residual * xi[j]).powi(2) / noise_var - 0.5;
                }
            }
        }

        // KL gradient
        for j in 0..self.n_features {
            let prior_var = self.prior_log_var[j].exp();
            let q_var = self.weight_log_var[j].exp();
            grad_mean[j] -= (self.weight_mean[j] - self.prior_mean[j]) / prior_var;
            grad_log_var[j] -= 0.5 * (q_var / prior_var - 1.0);
        }

        // Update
        for j in 0..self.n_features {
            self.weight_mean[j] += learning_rate * grad_mean[j] / n as f64;
            self.weight_log_var[j] += learning_rate * grad_log_var[j] / n as f64;
        }
    }

    /// Predict with uncertainty.
    pub fn predict(&self, x: &[f64], n_samples: usize) -> (f64, f64) {
        let mut rng = 42u64;
        let mut predictions = Vec::new();

        for _ in 0..n_samples {
            let w = DiagonalGaussian::new(self.weight_mean.clone(), self.weight_log_var.clone()).sample(rng);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let pred: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            predictions.push(pred);
        }

        let mean = predictions.iter().sum::<f64>() / n_samples as f64;
        let variance = predictions.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n_samples as f64;
        (mean, variance)
    }
}

/// Variational Autoencoder (simplified, 1D).
pub struct VariationalAutoencoder {
    pub input_dim: usize,
    pub latent_dim: usize,
    pub hidden_dim: usize,
    // Encoder
    pub enc_w1: Vec<Vec<f64>>,
    pub enc_b1: Vec<f64>,
    pub enc_w_mean: Vec<Vec<f64>>,
    pub enc_b_mean: Vec<f64>,
    pub enc_w_logvar: Vec<Vec<f64>>,
    pub enc_b_logvar: Vec<f64>,
    // Decoder
    pub dec_w1: Vec<Vec<f64>>,
    pub dec_b1: Vec<f64>,
    pub dec_w_out: Vec<Vec<f64>>,
    pub dec_b_out: Vec<f64>,
}

impl VariationalAutoencoder {
    pub fn new(input_dim: usize, latent_dim: usize, hidden_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale1 = (2.0 / input_dim as f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();
        let scale3 = (2.0 / latent_dim as f64).sqrt();

        Self {
            input_dim, latent_dim, hidden_dim,
            enc_w1: (0..hidden_dim).map(|_| (0..input_dim).map(|_| rand(scale1)).collect()).collect(),
            enc_b1: vec![0.0; hidden_dim],
            enc_w_mean: (0..latent_dim).map(|_| (0..hidden_dim).map(|_| rand(scale2)).collect()).collect(),
            enc_b_mean: vec![0.0; latent_dim],
            enc_w_logvar: (0..latent_dim).map(|_| (0..hidden_dim).map(|_| rand(scale2)).collect()).collect(),
            enc_b_logvar: vec![0.0; latent_dim],
            dec_w1: (0..hidden_dim).map(|_| (0..latent_dim).map(|_| rand(scale3)).collect()).collect(),
            dec_b1: vec![0.0; hidden_dim],
            dec_w_out: (0..input_dim).map(|_| (0..hidden_dim).map(|_| rand(scale2)).collect()).collect(),
            dec_b_out: vec![0.0; input_dim],
        }
    }

    /// Encode: x -> (mean, log_var).
    pub fn encode(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Hidden layer
        let h: Vec<f64> = self.enc_w1.iter().zip(self.enc_b1.iter()).map(|(w, &b)| {
            let sum: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b;
            sum.tanh()
        }).collect();

        // Mean
        let mean: Vec<f64> = self.enc_w_mean.iter().zip(self.enc_b_mean.iter()).map(|(w, &b)| {
            w.iter().zip(h.iter()).map(|(wi, hi)| wi * hi).sum::<f64>() + b
        }).collect();

        // Log variance
        let log_var: Vec<f64> = self.enc_w_logvar.iter().zip(self.enc_b_logvar.iter()).map(|(w, &b)| {
            let sum: f64 = w.iter().zip(h.iter()).map(|(wi, hi)| wi * hi).sum::<f64>() + b;
            sum.max(-10.0).min(10.0) // Clamp for stability
        }).collect();

        (mean, log_var)
    }

    /// Reparameterization trick.
    pub fn reparameterize(&self, mean: &[f64], log_var: &[f64], seed: u64) -> Vec<f64> {
        let mut rng = seed;
        mean.iter().zip(log_var.iter()).map(|(&m, &lv)| {
            let u1 = ((rng >> 33) as f64) / (1u64 << 31) as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = ((rng >> 33) as f64) / (1u64 << 31) as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let eps = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            m + (lv / 2.0).exp() * eps
        }).collect()
    }

    /// Decode: z -> x_recon.
    pub fn decode(&self, z: &[f64]) -> Vec<f64> {
        // Hidden layer
        let h: Vec<f64> = self.dec_w1.iter().zip(self.dec_b1.iter()).map(|(w, &b)| {
            let sum: f64 = w.iter().zip(z.iter()).map(|(wi, zi)| wi * zi).sum::<f64>() + b;
            sum.tanh()
        }).collect();

        // Output layer (sigmoid for binary data)
        self.dec_w_out.iter().zip(self.dec_b_out.iter()).map(|(w, &b)| {
            let sum: f64 = w.iter().zip(h.iter()).map(|(wi, hi)| wi * hi).sum::<f64>() + b;
            sigmoid(sum)
        }).collect()
    }

    /// Forward pass.
    pub fn forward(&self, x: &[f64], seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mean, log_var) = self.encode(x);
        let z = self.reparameterize(&mean, &log_var, seed);
        let x_recon = self.decode(&z);
        (x_recon, z, mean, log_var)
    }

    /// Reconstruction loss (binary cross-entropy).
    pub fn reconstruction_loss(&self, x: &[f64], x_recon: &[f64]) -> f64 {
        -x.iter().zip(x_recon.iter())
            .map(|(&xi, &ri)| {
                let ri = ri.max(1e-15).min(1.0 - 1e-15);
                xi * ri.ln() + (1.0 - xi) * (1.0 - ri).ln()
            })
            .sum::<f64>()
    }

    /// KL divergence loss.
    pub fn kl_loss(&self, mean: &[f64], log_var: &[f64]) -> f64 {
        -0.5 * mean.iter().zip(log_var.iter())
            .map(|(&m, &lv)| 1.0 + lv - m * m - lv.exp())
            .sum::<f64>()
    }

    /// Total loss (ELBO).
    pub fn loss(&self, x: &[f64], x_recon: &[f64], mean: &[f64], log_var: &[f64]) -> f64 {
        self.reconstruction_loss(x, x_recon) + self.kl_loss(mean, log_var)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Importance Weighted Autoencoder (IWAE) bound.
pub fn iwae_elbo(vae: &VariationalAutoencoder, x: &[f64], n_samples: usize) -> f64 {
    let mut rng = 42u64;
    let mut log_weights = Vec::new();

    for _ in 0..n_samples {
        let (x_recon, _, mean, log_var) = vae.forward(x, rng);
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);

        let log_p_x_given_z = -vae.reconstruction_loss(x, &x_recon);
        let log_p_z = -0.5 * mean.iter().map(|m| m * m).sum::<f64>();
        let log_q_z_given_x = -0.5 * mean.iter().zip(log_var.iter())
            .map(|(&m, &lv)| lv + m * m / lv.exp())
            .sum::<f64>();

        log_weights.push(log_p_x_given_z + log_p_z - log_q_z_given_x);
    }

    // Log-sum-exp trick
    let max_lw = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = log_weights.iter().map(|lw| (lw - max_lw).exp()).sum();
    max_lw + (sum_exp / n_samples as f64).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_linear_vi() {
        let n = 50;
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 / n as f64]).collect();
        let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi[0] + 0.5).collect();

        let mut vi = BayesianLinearVI::new(1);
        vi.prior_mean = vec![0.0];
        vi.prior_log_var = vec![0.0];

        for _ in 0..100 {
            vi.update(&x, &y, 0.01, 10);
        }

        let (pred_mean, pred_var) = vi.predict(&[0.5], 100);
        assert!((pred_mean - 1.5).abs() < 0.5);
    }

    #[test]
    fn test_vae() {
        let vae = VariationalAutoencoder::new(4, 2, 8);
        let x = vec![1.0, 0.0, 1.0, 0.0];
        let (x_recon, z, mean, log_var) = vae.forward(&x, 42);

        assert_eq!(x_recon.len(), 4);
        assert_eq!(z.len(), 2);
        assert_eq!(mean.len(), 2);
        assert_eq!(log_var.len(), 2);

        let loss = vae.loss(&x, &x_recon, &mean, &log_var);
        assert!(loss.is_finite());
    }
}
