/// Multi-task learning: hard/soft parameter sharing, task weighting, uncertainty weighting.

/// Multi-task network with hard parameter sharing.
pub struct HardSharingMTL {
    pub shared_layers: Vec<Vec<f64>>,
    pub task_heads: Vec<Vec<Vec<f64>>>,
    pub n_tasks: usize,
    pub learning_rate: f64,
}

impl HardSharingMTL {
    pub fn new(input_dim: usize, hidden_dim: usize, n_tasks: usize, output_dim: usize, learning_rate: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale_in = (2.0 / input_dim as f64).sqrt();
        let scale_hid = (2.0 / hidden_dim as f64).sqrt();

        Self {
            learning_rate, n_tasks,
            shared_layers: (0..hidden_dim).map(|_| (0..input_dim).map(|_| rand(scale_in)).collect()).collect(),
            task_heads: (0..n_tasks).map(|_| {
                (0..output_dim).map(|_| (0..hidden_dim).map(|_| rand(scale_hid)).collect()).collect()
            }).collect(),
        }
    }

    pub fn shared_forward(&self, x: &[f64]) -> Vec<f64> {
        self.shared_layers.iter().map(|w| {
            w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
        }).collect()
    }

    pub fn task_forward(&self, task_id: usize, shared_features: &[f64]) -> Vec<f64> {
        self.task_heads[task_id].iter().map(|w| {
            w.iter().zip(shared_features.iter()).map(|(wi, fi)| wi * fi).sum()
        }).collect()
    }

    pub fn forward(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let shared = self.shared_forward(x);
        (0..self.n_tasks).map(|t| self.task_forward(t, &shared)).collect()
    }

    pub fn multi_task_loss(&self, x: &[f64], targets: &[Vec<f64>], task_weights: &[f64]) -> f64 {
        let shared = self.shared_forward(x);
        let mut total_loss = 0.0;

        for (t, target) in targets.iter().enumerate() {
            let pred = self.task_forward(t, &shared);
            let task_loss: f64 = pred.iter().zip(target.iter()).map(|(p, t)| (p - t).powi(2)).sum();
            total_loss += task_weights[t] * task_loss;
        }

        total_loss
    }
}

/// Uncertainty weighting (Kendall et al.).
pub struct UncertaintyWeighting {
    pub log_vars: Vec<f64>, // Log variance per task
    pub n_tasks: usize,
}

impl UncertaintyWeighting {
    pub fn new(n_tasks: usize) -> Self {
        Self {
            log_vars: vec![0.0; n_tasks],
            n_tasks,
        }
    }

    /// Compute weighted loss with uncertainty.
    pub fn weighted_loss(&self, task_losses: &[f64]) -> f64 {
        let mut total = 0.0;
        for (i, loss) in task_losses.iter().enumerate() {
            let precision = (-self.log_vars[i]).exp();
            total += precision * loss + self.log_vars[i];
        }
        total
    }

    /// Get task weights from uncertainties.
    pub fn get_weights(&self) -> Vec<f64> {
        self.log_vars.iter().map(|&lv| (-lv).exp()).collect()
    }

    /// Update log variances.
    pub fn update(&mut self, task_losses: &[f64], learning_rate: f64) {
        for (i, loss) in task_losses.iter().enumerate() {
            let precision = (-self.log_vars[i]).exp();
            let grad = 0.5 - 0.5 * precision * loss;
            self.log_vars[i] += learning_rate * grad;
        }
    }
}

/// Gradient normalization (GradNorm).
pub struct GradNorm {
    pub task_losses: Vec<f64>,
    pub initial_losses: Vec<f64>,
    pub weights: Vec<f64>,
    pub alpha: f64,
    pub learning_rate: f64,
}

impl GradNorm {
    pub fn new(n_tasks: usize, alpha: f64, learning_rate: f64) -> Self {
        Self {
            task_losses: vec![0.0; n_tasks],
            initial_losses: vec![1.0; n_tasks],
            weights: vec![1.0 / n_tasks as f64; n_tasks],
            alpha, learning_rate,
        }
    }

    /// Update weights based on gradient norms.
    pub fn update_weights(&mut self, gradient_norms: &[f64]) {
        let avg_norm: f64 = gradient_norms.iter().sum::<f64>() / gradient_norms.len() as f64;

        for (i, (norm, loss)) in gradient_norms.iter().zip(self.task_losses.iter()).enumerate() {
            let relative_inverse_rate = loss / self.initial_losses[i];
            let target_norm = avg_norm * relative_inverse_rate.powf(self.alpha);

            // Update weight to bring gradient norm closer to target
            let grad_diff = norm - target_norm;
            self.weights[i] -= self.learning_rate * grad_diff;
            self.weights[i] = self.weights[i].max(0.01);
        }

        // Normalize weights
        let sum: f64 = self.weights.iter().sum();
        for w in self.weights.iter_mut() { *w /= sum; }
    }
}

/// Multi-task attention network.
pub struct TaskAttentionMTL {
    pub shared_dim: usize,
    pub n_tasks: usize,
    pub attention_weights: Vec<Vec<f64>>,
    pub shared_weights: Vec<Vec<f64>>,
}

impl TaskAttentionMTL {
    pub fn new(input_dim: usize, shared_dim: usize, n_tasks: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();

        Self {
            shared_dim, n_tasks,
            attention_weights: (0..n_tasks).map(|_| (0..shared_dim).map(|_| rand(scale)).collect()).collect(),
            shared_weights: (0..shared_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn forward(&self, x: &[f64], task_id: usize) -> Vec<f64> {
        // Shared representation
        let shared: Vec<f64> = self.shared_weights.iter().map(|w| {
            w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
        }).collect();

        // Task-specific attention
        let attention: Vec<f64> = self.attention_weights[task_id].iter().zip(shared.iter())
            .map(|(a, s)| a * s)
            .collect();

        // Element-wise multiplication
        attention.iter().zip(shared.iter()).map(|(a, s)| a * s).collect()
    }
}

/// Tensor factorization for multi-task learning (MTL with shared factors).
pub struct TensorFactorizationMTL {
    pub n_tasks: usize,
    pub rank: usize,
    pub shared_factor: Vec<Vec<f64>>,
    pub task_factors: Vec<Vec<Vec<f64>>>,
}

impl TensorFactorizationMTL {
    pub fn new(n_tasks: usize, input_dim: usize, rank: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / (input_dim * rank) as f64).sqrt();

        Self {
            n_tasks, rank,
            shared_factor: (0..rank).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
            task_factors: (0..n_tasks).map(|_| (0..rank).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn forward(&self, x: &[f64], task_id: usize) -> Vec<f64> {
        // Project to rank dimension
        let rank_repr: Vec<f64> = self.shared_factor.iter().map(|w| {
            w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
        }).collect();

        // Task-specific projection
        self.task_factors[task_id].iter().map(|w| {
            w.iter().zip(rank_repr.iter()).map(|(wi, ri)| wi * ri).sum()
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_sharing() {
        let mtl = HardSharingMTL::new(4, 8, 3, 2, 0.01);
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let outputs = mtl.forward(&x);
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].len(), 2);
    }

    #[test]
    fn test_uncertainty_weighting() {
        let mut uw = UncertaintyWeighting::new(3);
        let losses = vec![0.5, 0.3, 0.8];
        let weighted = uw.weighted_loss(&losses);
        assert!(weighted.is_finite());

        let weights = uw.get_weights();
        assert_eq!(weights.len(), 3);
    }

    #[test]
    fn test_task_attention() {
        let mtl = TaskAttentionMTL::new(4, 8, 3);
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let out = mtl.forward(&x, 0);
        assert_eq!(out.len(), 8);
    }
}
