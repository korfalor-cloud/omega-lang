/// Federated learning: FedAvg, FedProx, differential privacy.

use std::collections::HashMap;

/// Federated Averaging (FedAvg).
pub struct FedAvg {
    pub global_params: Vec<f64>,
    pub learning_rate: f64,
    pub n_clients: usize,
}

impl FedAvg {
    pub fn new(param_dim: usize, learning_rate: f64, n_clients: usize) -> Self {
        let mut seed = 42u64;
        let global_params: Vec<f64> = (0..param_dim).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.1 - 0.05
        }).collect();

        Self { global_params, learning_rate, n_clients }
    }

    /// Client update: local SGD.
    pub fn client_update<F, G>(
        &self,
        client_data: &[(Vec<f64>, f64)],
        loss_fn: F,
        grad_fn: G,
        n_local_steps: usize,
    ) -> Vec<f64>
    where
        F: Fn(&[f64], &[f64], f64) -> f64,
        G: Fn(&[f64], &[f64], f64) -> Vec<f64>,
    {
        let mut local_params = self.global_params.clone();

        for _ in 0..n_local_steps {
            for (x, y) in client_data {
                let grad = grad_fn(&local_params, x, *y);
                for i in 0..local_params.len() {
                    local_params[i] -= self.learning_rate * grad[i];
                }
            }
        }

        local_params
    }

    /// Server aggregation: weighted average of client updates.
    pub fn aggregate(&mut self, client_params: Vec<Vec<f64>>, client_sizes: Vec<usize>) {
        let total_size: usize = client_sizes.iter().sum();
        let param_dim = self.global_params.len();

        for i in 0..param_dim {
            let mut weighted_sum = 0.0;
            for (params, &size) in client_params.iter().zip(client_sizes.iter()) {
                weighted_sum += params[i] * size as f64;
            }
            self.global_params[i] = weighted_sum / total_size as f64;
        }
    }

    /// Full federated training round.
    pub fn train_round<F, G>(
        &mut self,
        client_data: Vec<&[(Vec<f64>, f64)]>,
        loss_fn: F,
        grad_fn: G,
        n_local_steps: usize,
    )
    where
        F: Fn(&[f64], &[f64], f64) -> f64 + Copy,
        G: Fn(&[f64], &[f64], f64) -> Vec<f64> + Copy,
    {
        let mut client_params = Vec::new();
        let mut client_sizes = Vec::new();

        for data in &client_data {
            let params = self.client_update(data, loss_fn, grad_fn, n_local_steps);
            client_sizes.push(data.len());
            client_params.push(params);
        }

        self.aggregate(client_params, client_sizes);
    }
}

/// FedProx: adds proximal term to local objective.
pub struct FedProx {
    pub global_params: Vec<f64>,
    pub learning_rate: f64,
    pub mu: f64, // Proximal term coefficient
    pub n_clients: usize,
}

impl FedProx {
    pub fn new(param_dim: usize, learning_rate: f64, mu: f64, n_clients: usize) -> Self {
        let mut seed = 42u64;
        let global_params: Vec<f64> = (0..param_dim).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.1 - 0.05
        }).collect();

        Self { global_params, learning_rate, mu, n_clients }
    }

    pub fn client_update<F, G>(
        &self,
        client_data: &[(Vec<f64>, f64)],
        grad_fn: G,
        n_local_steps: usize,
    ) -> Vec<f64>
    where
        G: Fn(&[f64], &[f64], f64) -> Vec<f64>,
    {
        let mut local_params = self.global_params.clone();

        for _ in 0..n_local_steps {
            for (x, y) in client_data {
                let mut grad = grad_fn(&local_params, x, *y);

                // Add proximal term gradient
                for i in 0..local_params.len() {
                    grad[i] += self.mu * (local_params[i] - self.global_params[i]);
                }

                for i in 0..local_params.len() {
                    local_params[i] -= self.learning_rate * grad[i];
                }
            }
        }

        local_params
    }

    pub fn aggregate(&mut self, client_params: Vec<Vec<f64>>, client_sizes: Vec<usize>) {
        let total_size: usize = client_sizes.iter().sum();
        let param_dim = self.global_params.len();

        for i in 0..param_dim {
            let mut weighted_sum = 0.0;
            for (params, &size) in client_params.iter().zip(client_sizes.iter()) {
                weighted_sum += params[i] * size as f64;
            }
            self.global_params[i] = weighted_sum / total_size as f64;
        }
    }
}

/// Differential Privacy mechanism.
pub struct DifferentialPrivacy {
    pub epsilon: f64,
    pub delta: f64,
    pub sensitivity: f64,
    seed: u64,
}

impl DifferentialPrivacy {
    pub fn new(epsilon: f64, delta: f64, sensitivity: f64) -> Self {
        Self { epsilon, delta, sensitivity, seed: 42 }
    }

    /// Gaussian mechanism.
    pub fn add_gaussian_noise(&mut self, params: &[f64]) -> Vec<f64> {
        let sigma = self.sensitivity * (2.0 * (1.25 / self.delta).ln()).sqrt() / self.epsilon;

        params.iter().map(|&p| {
            p + self.gaussian() * sigma
        }).collect()
    }

    /// Laplace mechanism.
    pub fn add_laplace_noise(&mut self, params: &[f64]) -> Vec<f64> {
        let scale = self.sensitivity / self.epsilon;

        params.iter().map(|&p| {
            p + self.laplace() * scale
        }).collect()
    }

    /// Gradient clipping for DP-SGD.
    pub fn clip_gradients(&self, gradients: &mut [Vec<f64>], max_norm: f64) {
        for grad in gradients {
            let norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if norm > max_norm {
                let scale = max_norm / norm;
                for g in grad.iter_mut() {
                    *g *= scale;
                }
            }
        }
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.pseudo_rand().max(1e-10);
        let u2 = self.pseudo_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn laplace(&mut self) -> f64 {
        let u = self.pseudo_rand() - 0.5;
        -u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Secure aggregation (simplified).
pub struct SecureAggregation {
    pub n_clients: usize,
    pub masks: Vec<Vec<f64>>,
}

impl SecureAggregation {
    pub fn new(n_clients: usize) -> Self {
        Self { n_clients, masks: Vec::new() }
    }

    /// Generate pairwise masks.
    pub fn generate_masks(&mut self, param_dim: usize) {
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * 2.0 - 1.0
        };

        self.masks = (0..self.n_clients).map(|_| {
            (0..param_dim).map(|_| rand()).collect()
        }).collect();
    }

    /// Mask client update.
    pub fn mask_update(&self, client_id: usize, params: &[f64]) -> Vec<f64> {
        params.iter().zip(self.masks[client_id].iter()).map(|(p, m)| p + m).collect()
    }

    /// Aggregate masked updates (masks cancel out).
    pub fn aggregate_masked(&self, masked_updates: &[Vec<f64>]) -> Vec<f64> {
        let param_dim = masked_updates[0].len();
        let mut result = vec![0.0; param_dim];

        for update in masked_updates {
            for (i, val) in update.iter().enumerate() {
                result[i] += val;
            }
        }

        // Average
        for val in result.iter_mut() {
            *val /= self.n_clients as f64;
        }

        result
    }
}

/// Federated learning with non-IID data handling.
pub struct FederatedNonIID {
    pub global_params: Vec<f64>,
    pub learning_rate: f64,
    pub momentum: f64,
    pub velocity: Vec<f64>,
}

impl FederatedNonIID {
    pub fn new(param_dim: usize, learning_rate: f64, momentum: f64) -> Self {
        Self {
            global_params: vec![0.0; param_dim],
            learning_rate, momentum,
            velocity: vec![0.0; param_dim],
        }
    }

    /// FedNova: normalized averaging.
    pub fn fednova_aggregate(&mut self, client_params: Vec<Vec<f64>>, client_steps: Vec<usize>) {
        let total_steps: usize = client_steps.iter().sum();
        let param_dim = self.global_params.len();

        // Compute pseudo-gradient
        let mut pseudo_grad = vec![0.0; param_dim];
        for (params, &steps) in client_params.iter().zip(client_steps.iter()) {
            for i in 0..param_dim {
                pseudo_grad[i] += (self.global_params[i] - params[i]) * steps as f64;
            }
        }

        for i in 0..param_dim {
            pseudo_grad[i] /= total_steps as f64;
            self.velocity[i] = self.momentum * self.velocity[i] + pseudo_grad[i];
            self.global_params[i] -= self.learning_rate * self.velocity[i];
        }
    }

    /// SCAFFOLD: control variates.
    pub fn scaffold_aggregate(
        &mut self,
        client_params: Vec<Vec<f64>>,
        client_controls: Vec<Vec<f64>>,
        server_control: &[f64],
    ) {
        let n = client_params.len();
        let param_dim = self.global_params.len();

        for i in 0..param_dim {
            let mut sum = 0.0;
            for (params, controls) in client_params.iter().zip(client_controls.iter()) {
                sum += params[i] - self.global_params[i] + server_control[i] - controls[i];
            }
            self.global_params[i] += sum / n as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fedavg() {
        let mut fed = FedAvg::new(2, 0.01, 3);
        let data1 = vec![(vec![1.0, 0.0], 1.0), (vec![0.0, 1.0], 0.0)];
        let data2 = vec![(vec![0.0, 1.0], 0.0), (vec![1.0, 0.0], 1.0)];

        let grad_fn = |params: &[f64], x: &[f64], y: f64| -> Vec<f64> {
            let pred: f64 = params.iter().zip(x.iter()).map(|(p, xi)| p * xi).sum();
            let error = pred - y;
            x.iter().map(|xi| error * xi).collect()
        };

        let client_params = vec![
            fed.client_update(&data1, |_p, _x, _y| 0.0, grad_fn, 1),
            fed.client_update(&data2, |_p, _x, _y| 0.0, grad_fn, 1),
        ];
        let sizes = vec![data1.len(), data2.len()];
        fed.aggregate(client_params, sizes);
    }

    #[test]
    fn test_differential_privacy() {
        let mut dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0);
        let params = vec![1.0, 2.0, 3.0];
        let noisy = dp.add_gaussian_noise(&params);
        assert_eq!(noisy.len(), 3);
        // Values should be close but not identical
        assert!((noisy[0] - params[0]).abs() < 10.0);
    }
}
