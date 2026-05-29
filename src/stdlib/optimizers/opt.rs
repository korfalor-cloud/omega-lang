/// Optimizers: AdamW, RAdam, LAMB, Novograd.

/// AdamW optimizer.
pub struct AdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub t: usize,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}

impl AdamW {
    pub fn new(param_dim: usize, lr: f64, beta1: f64, beta2: f64, epsilon: f64, weight_decay: f64) -> Self {
        Self {
            lr, beta1, beta2, epsilon, weight_decay, t: 0,
            m: vec![0.0; param_dim],
            v: vec![0.0; param_dim],
        }
    }

    pub fn step(&mut self, params: &mut [f64], gradients: &[f64]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradients[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradients[i] * gradients[i];

            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            // Decoupled weight decay
            params[i] *= 1.0 - self.lr * self.weight_decay;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

/// RAdam (Rectified Adam).
pub struct RAdam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub t: usize,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}

impl RAdam {
    pub fn new(param_dim: usize, lr: f64) -> Self {
        Self {
            lr, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, weight_decay: 0.0, t: 0,
            m: vec![0.0; param_dim],
            v: vec![0.0; param_dim],
        }
    }

    pub fn step(&mut self, params: &mut [f64], gradients: &[f64]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);

        // Compute rho_inf
        let rho_inf = 2.0 / (1.0 - self.beta2) - 1.0;
        let rho_t = rho_inf - 2.0 * self.t as f64 * self.beta2.powi(self.t as i32) / (1.0 - self.beta2.powi(self.t as i32));

        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradients[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradients[i] * gradients[i];

            let m_hat = self.m[i] / bc1;

            if rho_t > 4.0 {
                // Rectified update
                let l = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)).sqrt();
                let v_hat = (self.v[i] / (1.0 - self.beta2.powi(self.t as i32))).sqrt();
                params[i] -= self.lr * l * m_hat / (v_hat + self.epsilon);
            } else {
                // SGD-like update
                params[i] -= self.lr * m_hat;
            }

            // Weight decay
            if self.weight_decay > 0.0 {
                params[i] -= self.lr * self.weight_decay * params[i];
            }
        }
    }
}

/// LAMB (Layer-wise Adaptive Moments optimizer for Batch training).
pub struct LAMB {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub t: usize,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}

impl LAMB {
    pub fn new(param_dim: usize, lr: f64) -> Self {
        Self {
            lr, beta1: 0.9, beta2: 0.999, epsilon: 1e-6, weight_decay: 0.01, t: 0,
            m: vec![0.0; param_dim],
            v: vec![0.0; param_dim],
        }
    }

    pub fn step(&mut self, params: &mut [f64], gradients: &[f64]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        let mut updates = Vec::new();

        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradients[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradients[i] * gradients[i];

            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            let r = m_hat / (v_hat.sqrt() + self.epsilon) + self.weight_decay * params[i];
            updates.push(r);
        }

        // Layer-wise trust ratio
        let param_norm: f64 = params.iter().map(|p| p * p).sum::<f64>().sqrt();
        let update_norm: f64 = updates.iter().map(|u| u * u).sum::<f64>().sqrt();

        let trust_ratio = if update_norm > 0.0 && param_norm > 0.0 {
            param_norm / update_norm
        } else {
            1.0
        };

        for (p, u) in params.iter_mut().zip(updates.iter()) {
            *p -= self.lr * trust_ratio * u;
        }
    }
}

/// Novograd.
pub struct Novograd {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub t: usize,
    pub m: Vec<f64>,
    pub v: f64,
}

impl Novograd {
    pub fn new(param_dim: usize, lr: f64) -> Self {
        Self {
            lr, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, weight_decay: 0.01, t: 0,
            m: vec![0.0; param_dim],
            v: 0.0,
        }
    }

    pub fn step(&mut self, params: &mut [f64], gradients: &[f64]) {
        self.t += 1;

        // Update v (second moment of gradient norm)
        let grad_norm_sq: f64 = gradients.iter().map(|g| g * g).sum();
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad_norm_sq;

        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let v_hat = self.v / (1.0 - self.beta2.powi(self.t as i32));

        for i in 0..params.len() {
            // Normalize gradient by v
            let g_norm = gradients[i] / (v_hat.sqrt() + self.epsilon);

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g_norm;
            let m_hat = self.m[i] / bc1;

            params[i] -= self.lr * (m_hat + self.weight_decay * params[i]);
        }
    }
}

/// AdaBound.
pub struct AdaBound {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub final_lr: f64,
    pub gamma: f64,
    pub t: usize,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}

impl AdaBound {
    pub fn new(param_dim: usize, lr: f64, final_lr: f64) -> Self {
        Self {
            lr, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, final_lr, gamma: 1e-3, t: 0,
            m: vec![0.0; param_dim],
            v: vec![0.0; param_dim],
        }
    }

    pub fn step(&mut self, params: &mut [f64], gradients: &[f64]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        // Compute bounds
        let lower_bound = self.final_lr * (1.0 - 1.0 / (self.gamma * self.t as f64 + 1.0));
        let upper_bound = self.final_lr * (1.0 + 1.0 / (self.gamma * self.t as f64));

        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradients[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradients[i] * gradients[i];

            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            let step_size = (self.lr / (v_hat.sqrt() + self.epsilon)).max(lower_bound).min(upper_bound);
            params[i] -= step_size * m_hat;
        }
    }
}

/// AggMo (Aggregated Momentum).
pub struct AggMo {
    pub lr: f64,
    pub betas: Vec<f64>,
    pub velocities: Vec<Vec<f64>>,
}

impl AggMo {
    pub fn new(param_dim: usize, lr: f64, betas: Vec<f64>) -> Self {
        let velocities = betas.iter().map(|_| vec![0.0; param_dim]).collect();
        Self { lr, betas, velocities }
    }

    pub fn step(&mut self, params: &mut [f64], gradients: &[f64]) {
        let mut updates = vec![0.0; params.len()];

        for (beta_idx, &beta) in self.betas.iter().enumerate() {
            for i in 0..params.len() {
                self.velocities[beta_idx][i] = beta * self.velocities[beta_idx][i] + gradients[i];
                updates[i] += self.velocities[beta_idx][i];
            }
        }

        let n = self.betas.len() as f64;
        for (p, u) in params.iter_mut().zip(updates.iter()) {
            *p -= self.lr * u / n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw() {
        let mut opt = AdamW::new(4, 0.01, 0.9, 0.999, 1e-8, 0.01);
        let mut params = vec![1.0, 2.0, 3.0, 4.0];
        let grads = vec![0.1, 0.2, 0.3, 0.4];
        opt.step(&mut params, &grads);
        assert!(params[0] < 1.0);
    }

    #[test]
    fn test_radam() {
        let mut opt = RAdam::new(4, 0.01);
        let mut params = vec![1.0, 2.0, 3.0, 4.0];
        let grads = vec![0.1, 0.2, 0.3, 0.4];
        opt.step(&mut params, &grads);
        assert!(params[0] < 1.0);
    }

    #[test]
    fn test_lamb() {
        let mut opt = LAMB::new(4, 0.01);
        let mut params = vec![1.0, 2.0, 3.0, 4.0];
        let grads = vec![0.1, 0.2, 0.3, 0.4];
        opt.step(&mut params, &grads);
        assert!(params[0] < 1.0);
    }
}
