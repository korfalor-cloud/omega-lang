/// Lifelong learning: knowledge distillation, gradient episodic memory.

/// Learning without Forgetting (LwF).
pub struct LearningWithoutForgetting {
    pub params: Vec<f64>,
    pub old_params: Vec<f64>,
    pub temperature: f64,
    pub lambda: f64,
    pub learning_rate: f64,
}

impl LearningWithoutForgetting {
    pub fn new(param_dim: usize, temperature: f64, lambda: f64, learning_rate: f64) -> Self {
        Self {
            params: vec![0.0; param_dim],
            old_params: vec![0.0; param_dim],
            temperature, lambda, learning_rate,
        }
    }

    /// Compute knowledge distillation loss.
    pub fn kd_loss(&self, new_logits: &[f64], old_logits: &[f64]) -> f64 {
        let t = self.temperature;
        let new_probs = softmax_temperature(new_logits, t);
        let old_probs = softmax_temperature(old_logits, t);

        -new_probs.iter().zip(old_probs.iter())
            .filter(|(_, &o)| o > 0.0)
            .map(|(n, o)| o * (n.max(1e-15)).ln())
            .sum::<f64>()
    }

    /// Update with LwF: new_task_loss + lambda * kd_loss.
    pub fn update(&mut self, new_grad: &[f64], kd_grad: &[f64]) {
        for i in 0..self.params.len() {
            self.params[i] -= self.learning_rate * (new_grad[i] + self.lambda * kd_grad[i]);
        }
    }

    pub fn save_old_params(&mut self) {
        self.old_params = self.params.clone();
    }
}

fn softmax_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logits.iter().map(|l| l / temperature).collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Gradient Episodic Memory (GEM).
pub struct GEM {
    pub params: Vec<f64>,
    pub memory: Vec<Vec<(Vec<f64>, f64)>>, // Per-task memory
    pub learning_rate: f64,
    pub margin: f64,
}

impl GEM {
    pub fn new(param_dim: usize, learning_rate: f64, margin: f64) -> Self {
        Self {
            params: vec![0.0; param_dim],
            memory: Vec::new(),
            learning_rate, margin,
        }
    }

    /// Project gradient to satisfy constraints from old tasks.
    pub fn project_gradient(&self, grad: &[f64], task_id: usize) -> Vec<f64> {
        let mut projected = grad.to_vec();

        for (t, task_memory) in self.memory.iter().enumerate() {
            if t == task_id { continue; }

            // Compute reference gradient for old task
            let mut ref_grad = vec![0.0; self.params.len()];
            for (x, y) in task_memory {
                let error = self.predict(x) - y;
                for (i, xi) in x.iter().enumerate() {
                    if i < ref_grad.len() {
                        ref_grad[i] += error * xi;
                    }
                }
            }
            let n = task_memory.len() as f64;
            for g in ref_grad.iter_mut() { *g /= n; }

            // Check constraint: <grad, ref_grad> >= -margin
            let dot: f64 = projected.iter().zip(ref_grad.iter()).map(|(a, b)| a * b).sum();
            if dot < -self.margin {
                // Project grad onto ref_grad
                let ref_norm_sq: f64 = ref_grad.iter().map(|g| g * g).sum();
                if ref_norm_sq > 1e-10 {
                    let scale = (dot + self.margin) / ref_norm_sq;
                    for i in 0..projected.len() {
                        projected[i] -= scale * ref_grad[i];
                    }
                }
            }
        }

        projected
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        self.params.iter().zip(x.iter()).map(|(p, xi)| p * xi).sum()
    }

    pub fn update(&mut self, grad: &[f64], task_id: usize) {
        let projected = self.project_gradient(grad, task_id);
        for i in 0..self.params.len() {
            self.params[i] -= self.learning_rate * projected[i];
        }
    }

    pub fn add_to_memory(&mut self, task_id: usize, x: Vec<f64>, y: f64) {
        while self.memory.len() <= task_id {
            self.memory.push(Vec::new());
        }
        self.memory[task_id].push((x, y));
    }
}

/// A-GEM (Averaged GEM).
pub struct AGEM {
    pub params: Vec<f64>,
    pub memory: Vec<(Vec<f64>, f64)>,
    pub learning_rate: f64,
}

impl AGEM {
    pub fn new(param_dim: usize, learning_rate: f64) -> Self {
        Self {
            params: vec![0.0; param_dim],
            memory: Vec::new(),
            learning_rate,
        }
    }

    /// Compute reference gradient from memory.
    pub fn reference_gradient(&self) -> Vec<f64> {
        let mut ref_grad = vec![0.0; self.params.len()];
        for (x, y) in &self.memory {
            let error = self.predict(x) - y;
            for (i, xi) in x.iter().enumerate() {
                if i < ref_grad.len() {
                    ref_grad[i] += error * xi;
                }
            }
        }
        let n = self.memory.len() as f64;
        for g in ref_grad.iter_mut() { *g /= n.max(1.0); }
        ref_grad
    }

    pub fn project_gradient(&self, grad: &[f64]) -> Vec<f64> {
        let ref_grad = self.reference_gradient();
        let dot: f64 = grad.iter().zip(ref_grad.iter()).map(|(a, b)| a * b).sum();

        if dot >= 0.0 {
            return grad.to_vec();
        }

        let ref_norm_sq: f64 = ref_grad.iter().map(|g| g * g).sum();
        if ref_norm_sq < 1e-10 {
            return grad.to_vec();
        }

        let scale = dot / ref_norm_sq;
        grad.iter().zip(ref_grad.iter()).map(|(g, r)| g - scale * r).collect()
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        self.params.iter().zip(x.iter()).map(|(p, xi)| p * xi).sum()
    }

    pub fn update(&mut self, grad: &[f64]) {
        let projected = self.project_gradient(grad);
        for i in 0..self.params.len() {
            self.params[i] -= self.learning_rate * projected[i];
        }
    }
}

/// Dark Experience Replay (DER).
pub struct DarkExperienceReplay {
    pub params: Vec<f64>,
    pub buffer: Vec<(Vec<f64>, Vec<f64>, usize)>, // (input, logits, task_id)
    pub buffer_size: usize,
    pub alpha: f64,
    pub learning_rate: f64,
}

impl DarkExperienceReplay {
    pub fn new(param_dim: usize, buffer_size: usize, alpha: f64, learning_rate: f64) -> Self {
        Self {
            params: vec![0.0; param_dim],
            buffer: Vec::new(),
            buffer_size, alpha, learning_rate,
        }
    }

    pub fn add_to_buffer(&mut self, x: Vec<f64>, logits: Vec<f64>, task_id: usize) {
        if self.buffer.len() >= self.buffer_size {
            // Remove oldest
            self.buffer.remove(0);
        }
        self.buffer.push((x, logits, task_id));
    }

    pub fn replay_loss(&self, current_logits_fn: &dyn Fn(&[f64], &[f64]) -> Vec<f64>) -> f64 {
        let mut loss = 0.0;
        for (x, old_logits, _) in &self.buffer {
            let current_logits = current_logits_fn(&self.params, x);
            // KL divergence between old and current logits
            let old_probs = softmax(old_logits);
            let current_probs = softmax(&current_logits);
            loss += kl_divergence(&old_probs, &current_probs);
        }
        loss / self.buffer.len().max(1) as f64
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter())
        .filter(|(pi, _)| **pi > 0.0)
        .map(|(pi, qi)| pi * (pi / qi.max(1e-15)).ln())
        .sum()
}

/// Experience Replay for continual learning.
pub struct ExperienceReplayBuffer {
    pub buffer: Vec<(Vec<f64>, f64)>,
    pub capacity: usize,
    seed: u64,
}

impl ExperienceReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { buffer: Vec::new(), capacity, seed: 42 }
    }

    pub fn add(&mut self, x: Vec<f64>, y: f64) {
        if self.buffer.len() >= self.capacity {
            let idx = (self.pseudo_rand() * self.buffer.len() as f64) as usize;
            self.buffer[idx] = (x, y);
        } else {
            self.buffer.push((x, y));
        }
    }

    pub fn sample(&mut self, batch_size: usize) -> Vec<(Vec<f64>, f64)> {
        let n = self.buffer.len();
        let mut batch = Vec::new();
        for _ in 0..batch_size.min(n) {
            let idx = (self.pseudo_rand() * n as f64) as usize % n;
            batch.push(self.buffer[idx].clone());
        }
        batch
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lwf() {
        let mut lwf = LearningWithoutForgetting::new(4, 2.0, 0.5, 0.01);
        lwf.save_old_params();

        let new_logits = vec![1.0, 2.0, 3.0];
        let old_logits = vec![0.5, 2.5, 2.0];
        let kd = lwf.kd_loss(&new_logits, &old_logits);
        assert!(kd > 0.0);
    }

    #[test]
    fn test_gem() {
        let mut gem = GEM::new(4, 0.01, 0.5);
        gem.add_to_memory(0, vec![1.0, 0.0, 0.0, 0.0], 1.0);
        gem.add_to_memory(0, vec![0.0, 1.0, 0.0, 0.0], 0.0);

        let grad = vec![0.1, 0.1, 0.1, 0.1];
        let projected = gem.project_gradient(&grad, 1);
        assert_eq!(projected.len(), 4);
    }
}
