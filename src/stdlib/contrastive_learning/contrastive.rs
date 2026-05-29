/// Contrastive learning: SimCLR, MoCo, BYOL.

/// SimCLR contrastive loss (NT-Xent).
pub fn nt_xent_loss(z_i: &[f64], z_j: &[f64], all_negatives: &[Vec<f64>], temperature: f64) -> f64 {
    let sim_pos = cosine_similarity(z_i, z_j) / temperature;

    let mut sims = vec![sim_pos];
    for neg in all_negatives {
        sims.push(cosine_similarity(z_i, neg) / temperature);
    }

    let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = sims.iter().map(|s| (s - max_sim).exp()).sum();
    -(sim_pos - max_sim - sum_exp.ln())
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a * norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 }
}

/// SimCLR framework.
pub struct SimCLR {
    pub encoder_dim: usize,
    pub projection_dim: usize,
    pub encoder_weights: Vec<Vec<f64>>,
    pub projection_weights: Vec<Vec<f64>>,
    pub temperature: f64,
}

impl SimCLR {
    pub fn new(encoder_dim: usize, projection_dim: usize, input_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale_enc = (2.0 / input_dim as f64).sqrt();
        let scale_proj = (2.0 / encoder_dim as f64).sqrt();

        Self {
            encoder_dim, projection_dim,
            encoder_weights: (0..encoder_dim).map(|_| (0..input_dim).map(|_| rand(scale_enc)).collect()).collect(),
            projection_weights: (0..projection_dim).map(|_| (0..encoder_dim).map(|_| rand(scale_proj)).collect()).collect(),
            temperature: 0.5,
        }
    }

    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        self.encoder_weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            sum.tanh()
        }).collect()
    }

    pub fn project(&self, h: &[f64]) -> Vec<f64> {
        self.projection_weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(h.iter()).map(|(wi, hi)| wi * hi).sum();
            sum
        }).collect()
    }

    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let h = self.encode(x);
        self.project(&h)
    }

    pub fn contrastive_loss(&self, x_i: &[f64], x_j: &[f64], negatives: &[Vec<f64>]) -> f64 {
        let z_i = self.forward(x_i);
        let z_j = self.forward(x_j);
        let z_neg: Vec<Vec<f64>> = negatives.iter().map(|x| self.forward(x)).collect();
        nt_xent_loss(&z_i, &z_j, &z_neg, self.temperature)
    }
}

/// MoCo (Momentum Contrast).
pub struct MoCo {
    pub encoder_dim: usize,
    pub queue_size: usize,
    pub momentum: f64,
    pub temperature: f64,
    pub encoder_weights: Vec<Vec<f64>>,
    pub momentum_encoder_weights: Vec<Vec<f64>>,
    pub queue: Vec<Vec<f64>>,
    pub queue_ptr: usize,
}

impl MoCo {
    pub fn new(encoder_dim: usize, input_dim: usize, queue_size: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();
        let encoder_weights: Vec<Vec<f64>> = (0..encoder_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect();
        let momentum_encoder_weights = encoder_weights.clone();
        let queue = vec![vec![0.0; encoder_dim]; queue_size];

        Self {
            encoder_dim, queue_size, momentum: 0.999, temperature: 0.07,
            encoder_weights, momentum_encoder_weights, queue, queue_ptr: 0,
        }
    }

    pub fn encode(&self, x: &[f64], use_momentum: bool) -> Vec<f64> {
        let weights = if use_momentum { &self.momentum_encoder_weights } else { &self.encoder_weights };
        weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            sum.tanh()
        }).collect()
    }

    pub fn update_momentum(&mut self) {
        for (m, e) in self.momentum_encoder_weights.iter_mut().zip(self.encoder_weights.iter()) {
            for (mi, ei) in m.iter_mut().zip(e.iter()) {
                *mi = self.momentum * *mi + (1.0 - self.momentum) * *ei;
            }
        }
    }

    pub fn enqueue(&mut self, key: Vec<f64>) {
        self.queue[self.queue_ptr] = key;
        self.queue_ptr = (self.queue_ptr + 1) % self.queue_size;
    }

    pub fn contrastive_loss(&self, q: &[f64], k: &[f64]) -> f64 {
        let pos_sim = cosine_similarity(q, k) / self.temperature;

        let neg_sims: Vec<f64> = self.queue.iter()
            .map(|neg| cosine_similarity(q, neg) / self.temperature)
            .collect();

        let mut all_sims = vec![pos_sim];
        all_sims.extend_from_slice(&neg_sims);

        let max_sim = all_sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = all_sims.iter().map(|s| (s - max_sim).exp()).sum();
        -(pos_sim - max_sim - sum_exp.ln())
    }
}

/// BYOL (Bootstrap Your Own Latent).
pub struct BYOL {
    pub encoder_dim: usize,
    pub predictor_dim: usize,
    pub encoder_weights: Vec<Vec<f64>>,
    pub target_encoder_weights: Vec<Vec<f64>>,
    pub predictor_weights: Vec<Vec<f64>>,
    pub momentum: f64,
}

impl BYOL {
    pub fn new(encoder_dim: usize, predictor_dim: usize, input_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale_enc = (2.0 / input_dim as f64).sqrt();
        let scale_pred = (2.0 / encoder_dim as f64).sqrt();

        let encoder_weights: Vec<Vec<f64>> = (0..encoder_dim).map(|_| (0..input_dim).map(|_| rand(scale_enc)).collect()).collect();
        let target_encoder_weights = encoder_weights.clone();
        let predictor_weights = (0..predictor_dim).map(|_| (0..encoder_dim).map(|_| rand(scale_pred)).collect()).collect();

        Self {
            encoder_dim, predictor_dim, encoder_weights, target_encoder_weights,
            predictor_weights, momentum: 0.996,
        }
    }

    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        self.encoder_weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            sum.tanh()
        }).collect()
    }

    pub fn target_encode(&self, x: &[f64]) -> Vec<f64> {
        self.target_encoder_weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            sum.tanh()
        }).collect()
    }

    pub fn predict(&self, h: &[f64]) -> Vec<f64> {
        self.predictor_weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(h.iter()).map(|(wi, hi)| wi * hi).sum();
            sum
        }).collect()
    }

    pub fn loss(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let h1 = self.encode(x1);
        let h2 = self.encode(x2);
        let p1 = self.predict(&h1);
        let p2 = self.predict(&h2);
        let z1 = self.target_encode(x1);
        let z2 = self.target_encode(x2);

        let loss1 = cosine_similarity(&p1, &z2);
        let loss2 = cosine_similarity(&p2, &z1);
        -(loss1 + loss2) / 2.0
    }

    pub fn update_target(&mut self) {
        for (t, e) in self.target_encoder_weights.iter_mut().zip(self.encoder_weights.iter()) {
            for (ti, ei) in t.iter_mut().zip(e.iter()) {
                *ti = self.momentum * *ti + (1.0 - self.momentum) * *ei;
            }
        }
    }
}

/// Barlow Twins redundancy reduction.
pub fn barlow_twins_loss(z_a: &[Vec<f64>], z_b: &[Vec<f64>], lambda: f64) -> f64 {
    let n = z_a.len();
    let d = z_a[0].len();

    // Normalize
    let mean_a: Vec<f64> = (0..d).map(|j| z_a.iter().map(|z| z[j]).sum::<f64>() / n as f64).collect();
    let mean_b: Vec<f64> = (0..d).map(|j| z_b.iter().map(|z| z[j]).sum::<f64>() / n as f64).collect();

    let std_a: Vec<f64> = (0..d).map(|j| {
        (z_a.iter().map(|z| (z[j] - mean_a[j]).powi(2)).sum::<f64>() / n as f64).sqrt().max(1e-8)
    }).collect();
    let std_b: Vec<f64> = (0..d).map(|j| {
        (z_b.iter().map(|z| (z[j] - mean_b[j]).powi(2)).sum::<f64>() / n as f64).sqrt().max(1e-8)
    }).collect();

    // Cross-correlation matrix
    let mut c = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..n {
                c[i][j] += ((z_a[k][i] - mean_a[i]) / std_a[i]) * ((z_b[k][j] - mean_b[j]) / std_b[j]);
            }
            c[i][j] /= n as f64;
        }
    }

    // Loss
    let mut loss = 0.0;
    for i in 0..d {
        for j in 0..d {
            if i == j {
                loss += (1.0 - c[i][j]).powi(2);
            } else {
                loss += lambda * c[i][j].powi(2);
            }
        }
    }

    loss
}

/// VICReg (Variance-Invariance-Covariance Regularization).
pub fn vicreg_loss(z_a: &[Vec<f64>], z_b: &[Vec<f64>], gamma: f64, mu: f64, nu: f64) -> f64 {
    let n = z_a.len();
    let d = z_a[0].len();

    // Invariance loss
    let invariance: f64 = z_a.iter().zip(z_b.iter())
        .map(|(a, b)| a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>())
        .sum::<f64>() / (n as f64 * d as f64);

    // Variance loss
    let mean_a: Vec<f64> = (0..d).map(|j| z_a.iter().map(|z| z[j]).sum::<f64>() / n as f64).collect();
    let var_a: Vec<f64> = (0..d).map(|j| {
        z_a.iter().map(|z| (z[j] - mean_a[j]).powi(2)).sum::<f64>() / n as f64
    }).collect();
    let variance_a: f64 = var_a.iter().map(|v| (gamma - v.sqrt()).max(0.0)).sum::<f64>() / d as f64;

    let mean_b: Vec<f64> = (0..d).map(|j| z_b.iter().map(|z| z[j]).sum::<f64>() / n as f64).collect();
    let var_b: Vec<f64> = (0..d).map(|j| {
        z_b.iter().map(|z| (z[j] - mean_b[j]).powi(2)).sum::<f64>() / n as f64
    }).collect();
    let variance_b: f64 = var_b.iter().map(|v| (gamma - v.sqrt()).max(0.0)).sum::<f64>() / d as f64;

    // Covariance loss
    let mut cov_a = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            cov_a[i][j] = z_a.iter().map(|z| (z[i] - mean_a[i]) * (z[j] - mean_a[j])).sum::<f64>() / n as f64;
        }
    }
    let covariance: f64 = (0..d).map(|i| {
        (0..d).filter(|&j| i != j).map(|j| cov_a[i][j].powi(2)).sum::<f64>()
    }).sum::<f64>() / d as f64;

    mu * invariance + gamma * (variance_a + variance_b) + nu * covariance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simclr() {
        let simclr = SimCLR::new(8, 4, 10);
        let x = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let z = simclr.forward(&x);
        assert_eq!(z.len(), 4);
    }

    #[test]
    fn test_moco() {
        let mut moco = MoCo::new(4, 10, 100);
        let q = moco.encode(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], false);
        let k = moco.encode(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], true);
        let loss = moco.contrastive_loss(&q, &k);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_nt_xent() {
        let z_i = vec![1.0, 0.0, 0.0];
        let z_j = vec![0.9, 0.1, 0.0];
        let negatives = vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let loss = nt_xent_loss(&z_i, &z_j, &negatives, 0.5);
        assert!(loss > 0.0);
    }
}
