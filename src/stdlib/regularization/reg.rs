/// Regularization techniques: dropout, weight decay, spectral normalization.

/// Dropout layer.
pub struct Dropout {
    pub rate: f64,
    seed: u64,
}

impl Dropout {
    pub fn new(rate: f64) -> Self {
        Self { rate, seed: 42 }
    }

    pub fn forward(&mut self, input: &[f64], training: bool) -> Vec<f64> {
        if !training || self.rate <= 0.0 {
            return input.to_vec();
        }

        let scale = 1.0 / (1.0 - self.rate);
        input.iter().map(|&x| {
            self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((self.seed >> 33) as f64) / (1u64 << 31) as f64;
            if r > self.rate { x * scale } else { 0.0 }
        }).collect()
    }
}

/// Weight decay (L2 regularization).
pub fn weight_decay(params: &mut [f64], gradients: &mut [f64], lr: f64, wd: f64) {
    for (p, g) in params.iter_mut().zip(gradients.iter_mut()) {
        *g += wd * *p;
        *p -= lr * *g;
    }
}

/// Spectral normalization.
pub struct SpectralNorm {
    pub weight: Vec<Vec<f64>>,
    pub u: Vec<f64>,
    pub n_power_iterations: usize,
}

impl SpectralNorm {
    pub fn new(weight: Vec<Vec<f64>>) -> Self {
        let rows = weight.len();
        let cols = weight[0].len();
        let mut seed = 42u64;
        let u: Vec<f64> = (0..rows).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64
        }).collect();

        Self { weight, u, n_power_iterations: 1 }
    }

    /// Compute spectral norm and normalized weight.
    pub fn normalize(&mut self) -> (f64, Vec<Vec<f64>>) {
        let rows = self.weight.len();
        let cols = self.weight[0].len();

        // Power iteration
        let mut u = self.u.clone();
        for _ in 0..self.n_power_iterations {
            // v = W^T u
            let mut v = vec![0.0; cols];
            for j in 0..cols {
                for i in 0..rows {
                    v[j] += self.weight[i][j] * u[i];
                }
            }
            let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in v.iter_mut() { *x /= v_norm.max(1e-10); }

            // u = W v
            let mut new_u = vec![0.0; rows];
            for i in 0..rows {
                for j in 0..cols {
                    new_u[i] += self.weight[i][j] * v[j];
                }
            }
            let u_norm: f64 = new_u.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in new_u.iter_mut() { *x /= u_norm.max(1e-10); }
            u = new_u;
        }

        // sigma = u^T W v
        let mut v = vec![0.0; cols];
        for j in 0..cols {
            for i in 0..rows {
                v[j] += self.weight[i][j] * u[i];
            }
        }
        let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v.iter_mut() { *x /= v_norm.max(1e-10); }

        let sigma: f64 = u.iter().enumerate().map(|(i, &ui)| {
            let wv: f64 = self.weight[i].iter().zip(v.iter()).map(|(w, vj)| w * vj).sum();
            ui * wv
        }).sum();

        self.u = u;

        // Return normalized weight
        let normalized: Vec<Vec<f64>> = self.weight.iter().map(|row| {
            row.iter().map(|&w| w / sigma.max(1e-10)).collect()
        }).collect();

        (sigma, normalized)
    }
}

/// Layer normalization.
pub struct LayerNorm {
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub epsilon: f64,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            epsilon: 1e-5,
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let n = input.len() as f64;
        let mean = input.iter().sum::<f64>() / n;
        let variance = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = (variance + self.epsilon).sqrt();

        input.iter().enumerate().map(|(i, &x)| {
            self.gamma[i] * (x - mean) / std + self.beta[i]
        }).collect()
    }
}

/// Batch normalization.
pub struct BatchNorm {
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
    pub momentum: f64,
    pub epsilon: f64,
}

impl BatchNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            running_mean: vec![0.0; dim],
            running_var: vec![1.0; dim],
            momentum: 0.1,
            epsilon: 1e-5,
        }
    }

    pub fn forward(&mut self, input: &[f64], training: bool) -> Vec<f64> {
        if training {
            let mean = input.iter().sum::<f64>() / input.len() as f64;
            let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / input.len() as f64;

            for i in 0..self.running_mean.len().min(input.len()) {
                self.running_mean[i] = (1.0 - self.momentum) * self.running_mean[i] + self.momentum * mean;
                self.running_var[i] = (1.0 - self.momentum) * self.running_var[i] + self.momentum * var;
            }

            input.iter().enumerate().map(|(i, &x)| {
                self.gamma[i] * (x - mean) / (var + self.epsilon).sqrt() + self.beta[i]
            }).collect()
        } else {
            input.iter().enumerate().map(|(i, &x)| {
                self.gamma[i] * (x - self.running_mean[i]) / (self.running_var[i] + self.epsilon).sqrt() + self.beta[i]
            }).collect()
        }
    }
}

/// Gradient clipping.
pub fn clip_gradients(gradients: &mut [f64], max_norm: f64) {
    let norm: f64 = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in gradients.iter_mut() {
            *g *= scale;
        }
    }
}

/// Early stopping.
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f64,
    pub best_value: f64,
    pub counter: usize,
    pub should_stop: bool,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience, min_delta,
            best_value: f64::INFINITY,
            counter: 0,
            should_stop: false,
        }
    }

    pub fn update(&mut self, value: f64) -> bool {
        if value < self.best_value - self.min_delta {
            self.best_value = value;
            self.counter = 0;
        } else {
            self.counter += 1;
            if self.counter >= self.patience {
                self.should_stop = true;
            }
        }
        self.should_stop
    }
}

/// Mixup regularization.
pub fn mixup_data(x1: &[f64], x2: &[f64], alpha: f64, seed: u64) -> (Vec<f64>, f64) {
    let mut rng = seed;
    let lambda = if alpha > 0.0 {
        let u = ((rng >> 33) as f64) / (1u64 << 31) as f64;
        u.powf(alpha) / (u.powf(alpha) + (1.0 - u).powf(alpha))
    } else {
        0.5
    };

    let mixed: Vec<f64> = x1.iter().zip(x2.iter()).map(|(&a, &b)| lambda * a + (1.0 - lambda) * b).collect();
    (mixed, lambda)
}

/// Label smoothing.
pub fn label_smoothing(labels: &[usize], n_classes: usize, smoothing: f64) -> Vec<Vec<f64>> {
    labels.iter().map(|&label| {
        (0..n_classes).map(|i| {
            if i == label { 1.0 - smoothing } else { smoothing / (n_classes - 1) as f64 }
        }).collect()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout() {
        let mut dropout = Dropout::new(0.5);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = dropout.forward(&input, true);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = ln.forward(&input);
        let mean: f64 = output.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_early_stopping() {
        let mut es = EarlyStopping::new(3, 0.01);
        assert!(!es.update(1.0));
        assert!(!es.update(0.9));
        assert!(!es.update(0.89));
        assert!(!es.update(0.88));
        assert!(es.update(0.87)); // Should stop
    }
}
