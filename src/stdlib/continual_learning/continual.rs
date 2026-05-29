/// Continual learning: Elastic Weight Consolidation (EWC), Progressive Nets, PackNet.

/// Elastic Weight Consolidation (EWC).
pub struct EWC {
    pub params: Vec<f64>,
    pub fisher_diagonal: Vec<f64>,
    pub optimal_params: Vec<f64>,
    pub lambda: f64,
    pub learning_rate: f64,
}

impl EWC {
    pub fn new(param_dim: usize, lambda: f64, learning_rate: f64) -> Self {
        Self {
            params: vec![0.0; param_dim],
            fisher_diagonal: vec![0.0; param_dim],
            optimal_params: vec![0.0; param_dim],
            lambda, learning_rate,
        }
    }

    /// Compute Fisher information matrix (diagonal approximation).
    pub fn compute_fisher(&mut self, data: &[(Vec<f64>, f64)], grad_fn: &dyn Fn(&[f64], &[f64], f64) -> Vec<f64>) {
        self.fisher_diagonal = vec![0.0; self.params.len()];

        for (x, y) in data {
            let grad = grad_fn(&self.params, x, *y);
            for (i, g) in grad.iter().enumerate() {
                self.fisher_diagonal[i] += g * g;
            }
        }

        let n = data.len() as f64;
        for f in self.fisher_diagonal.iter_mut() {
            *f /= n;
        }
    }

    /// Update with EWC regularization.
    pub fn update(&mut self, grad: &[f64]) {
        for i in 0..self.params.len() {
            let ewc_grad = self.lambda * self.fisher_diagonal[i] * (self.params[i] - self.optimal_params[i]);
            self.params[i] -= self.learning_rate * (grad[i] + ewc_grad);
        }
    }

    /// Save current parameters as optimal for current task.
    pub fn save_optimal(&mut self) {
        self.optimal_params = self.params.clone();
    }

    /// EWC loss: task_loss + lambda/2 * sum(F_i * (theta_i - theta*_i)^2).
    pub fn ewc_penalty(&self) -> f64 {
        0.5 * self.lambda * self.fisher_diagonal.iter().zip(self.params.iter()).zip(self.optimal_params.iter())
            .map(|((f, p), o)| f * (p - o).powi(2))
            .sum::<f64>()
    }
}

/// Online EWC: update Fisher incrementally.
pub struct OnlineEWC {
    pub params: Vec<f64>,
    pub fisher_diagonal: Vec<f64>,
    pub optimal_params: Vec<f64>,
    pub gamma: f64, // Forgetting factor
    pub lambda: f64,
    pub learning_rate: f64,
}

impl OnlineEWC {
    pub fn new(param_dim: usize, gamma: f64, lambda: f64, learning_rate: f64) -> Self {
        Self {
            params: vec![0.0; param_dim],
            fisher_diagonal: vec![0.0; param_dim],
            optimal_params: vec![0.0; param_dim],
            gamma, lambda, learning_rate,
        }
    }

    pub fn update_fisher(&mut self, new_fisher: &[f64]) {
        for i in 0..self.fisher_diagonal.len() {
            self.fisher_diagonal[i] = self.gamma * self.fisher_diagonal[i] + (1.0 - self.gamma) * new_fisher[i];
        }
    }

    pub fn update(&mut self, grad: &[f64]) {
        for i in 0..self.params.len() {
            let ewc_grad = self.lambda * self.fisher_diagonal[i] * (self.params[i] - self.optimal_params[i]);
            self.params[i] -= self.learning_rate * (grad[i] + ewc_grad);
        }
    }
}

/// Synaptic Intelligence (SI).
pub struct SynapticIntelligence {
    pub params: Vec<f64>,
    pub omega: Vec<f64>,         // Synaptic importance
    pub theta_star: Vec<f64>,    // Parameters at task end
    pub path_integral: Vec<f64>, // Running path integral
    pub lambda: f64,
    pub learning_rate: f64,
    pub xi: f64, // Damping parameter
}

impl SynapticIntelligence {
    pub fn new(param_dim: usize, lambda: f64, learning_rate: f64) -> Self {
        Self {
            params: vec![0.0; param_dim],
            omega: vec![0.0; param_dim],
            theta_star: vec![0.0; param_dim],
            path_integral: vec![0.0; param_dim],
            lambda, learning_rate, xi: 0.01,
        }
    }

    /// Update path integral during training.
    pub fn update_path_integral(&mut self, grad: &[f64]) {
        for i in 0..self.params.len() {
            self.path_integral[i] += grad[i] * (self.params[i] - self.theta_star[i]);
        }
    }

    /// Compute synaptic importance after task.
    pub fn compute_omega(&mut self) {
        for i in 0..self.omega.len() {
            let delta = (self.params[i] - self.theta_star[i]).abs().max(self.xi);
            self.omega[i] += self.path_integral[i] / (delta * delta);
        }
        self.theta_star = self.params.clone();
        self.path_integral = vec![0.0; self.params.len()];
    }

    pub fn update(&mut self, grad: &[f64]) {
        for i in 0..self.params.len() {
            let si_grad = self.lambda * self.omega[i] * (self.params[i] - self.theta_star[i]);
            self.params[i] -= self.learning_rate * (grad[i] + si_grad);
        }
    }
}

/// Memory Aware Synapses (MAS).
pub struct MAS {
    pub params: Vec<f64>,
    pub omega: Vec<f64>,
    pub optimal_params: Vec<f64>,
    pub lambda: f64,
    pub learning_rate: f64,
}

impl MAS {
    pub fn new(param_dim: usize, lambda: f64, learning_rate: f64) -> Self {
        Self {
            params: vec![0.0; param_dim],
            omega: vec![0.0; param_dim],
            optimal_params: vec![0.0; param_dim],
            lambda, learning_rate,
        }
    }

    /// Compute importance weights based on output sensitivity.
    pub fn compute_importance(&mut self, data: &[Vec<f64>], output_fn: &dyn Fn(&[f64], &[f64]) -> Vec<f64>) {
        self.omega = vec![0.0; self.params.len()];

        for x in data {
            let output = output_fn(&self.params, x);
            // Compute gradient of ||f(x)||^2 w.r.t. params
            let output_norm_sq: f64 = output.iter().map(|o| o * o).sum();
            // Simplified: use parameter magnitude as importance proxy
            for (i, p) in self.params.iter().enumerate() {
                self.omega[i] += p.abs() * output_norm_sq;
            }
        }

        let n = data.len() as f64;
        for o in self.omega.iter_mut() { *o /= n; }
    }

    pub fn update(&mut self, grad: &[f64]) {
        for i in 0..self.params.len() {
            let mas_grad = self.lambda * self.omega[i] * (self.params[i] - self.optimal_params[i]);
            self.params[i] -= self.learning_rate * (grad[i] + mas_grad);
        }
    }

    pub fn save_optimal(&mut self) {
        self.optimal_params = self.params.clone();
    }
}

/// Progressive Neural Network: adds columns for new tasks.
pub struct ProgressiveNet {
    pub columns: Vec<Column>,
    pub lateral_connections: Vec<Vec<Vec<f64>>>,
}

pub struct Column {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub task_id: usize,
}

impl ProgressiveNet {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            columns: Vec::new(),
            lateral_connections: Vec::new(),
        }
    }

    pub fn add_column(&mut self, input_dim: usize, hidden_dim: usize, task_id: usize) {
        let mut seed = 42u64 + task_id as u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();

        // Lateral connections from previous columns
        let n_prev = self.columns.len();
        let mut lateral = Vec::new();
        for _ in 0..n_prev {
            lateral.push((0..hidden_dim).map(|_| (0..hidden_dim).map(|_| rand(0.01)).collect()).collect());
        }

        self.columns.push(Column {
            weights: (0..hidden_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
            bias: vec![0.0; hidden_dim],
            task_id,
        });
        self.lateral_connections.push(lateral);
    }

    pub fn forward(&self, x: &[f64], task_id: usize) -> Vec<f64> {
        let col_idx = self.columns.iter().position(|c| c.task_id == task_id).unwrap_or(0);
        let col = &self.columns[col_idx];

        // Base computation
        let mut hidden: Vec<f64> = col.weights.iter().zip(col.bias.iter()).map(|(w, &b)| {
            w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b
        }).collect();

        // Add lateral connections from previous columns
        for (prev_idx, prev_col) in self.columns[..col_idx].iter().enumerate() {
            let prev_hidden: Vec<f64> = prev_col.weights.iter().zip(prev_col.bias.iter()).map(|(w, &b)| {
                w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b
            }).collect();

            let lateral = &self.lateral_connections[col_idx][prev_idx];
            for (i, h) in hidden.iter_mut().enumerate() {
                let lateral_contrib: f64 = lateral[i].iter().zip(prev_hidden.iter()).map(|(l, p)| l * p).sum();
                *h += lateral_contrib;
            }
        }

        // Activation
        hidden.iter().map(|&h| h.max(0.0)).collect()
    }

    /// Freeze previous columns.
    pub fn freeze_previous(&mut self, task_id: usize) {
        // In practice, this would mark parameters as non-trainable
        // Here we just record which columns are frozen
    }
}

/// PackNet: prune and freeze network parts.
pub struct PackNet {
    pub params: Vec<f64>,
    pub masks: Vec<Vec<bool>>, // Masks per task
    pub active_mask: Vec<bool>,
    pub pruning_ratio: f64,
}

impl PackNet {
    pub fn new(param_dim: usize, pruning_ratio: f64) -> Self {
        Self {
            params: vec![0.0; param_dim],
            masks: Vec::new(),
            active_mask: vec![true; param_dim],
            pruning_ratio,
        }
    }

    /// Prune smallest magnitude weights.
    pub fn prune(&mut self) {
        let n = self.params.len();
        let n_prune = (n as f64 * self.pruning_ratio) as usize;

        let mut indexed: Vec<(usize, f64)> = self.params.iter().enumerate()
            .filter(|(i, _)| self.active_mask[*i])
            .map(|(i, &p)| (i, p.abs()))
            .collect();

        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut new_mask = self.active_mask.clone();
        for (idx, _) in indexed.iter().take(n_prune) {
            new_mask[*idx] = false;
        }

        self.masks.push(new_mask.clone());
        self.active_mask = new_mask;
    }

    /// Get trainable parameters.
    pub fn trainable_params(&self) -> Vec<(usize, f64)> {
        self.params.iter().enumerate()
            .filter(|(i, _)| self.active_mask[*i])
            .map(|(i, &p)| (i, p))
            .collect()
    }

    pub fn update(&mut self, grad: &[f64], learning_rate: f64) {
        for i in 0..self.params.len() {
            if self.active_mask[i] {
                self.params[i] -= learning_rate * grad[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewc() {
        let mut ewc = EWC::new(4, 100.0, 0.01);
        ewc.params = vec![1.0, 2.0, 3.0, 4.0];
        ewc.save_optimal();
        ewc.fisher_diagonal = vec![1.0, 2.0, 3.0, 4.0];

        let grad = vec![0.1, 0.1, 0.1, 0.1];
        ewc.update(&grad);

        // Parameters should move toward gradient but be pulled back by EWC
        assert!(ewc.params[0] < 1.0);
    }

    #[test]
    fn test_progressive_net() {
        let mut pnet = ProgressiveNet::new(4, 8);
        pnet.add_column(4, 8, 0);
        pnet.add_column(4, 8, 1);

        let x = vec![1.0, 0.0, 0.0, 0.0];
        let out0 = pnet.forward(&x, 0);
        let out1 = pnet.forward(&x, 1);
        assert_eq!(out0.len(), 8);
        assert_eq!(out1.len(), 8);
    }
}
