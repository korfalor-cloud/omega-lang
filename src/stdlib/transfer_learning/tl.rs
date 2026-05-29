/// Transfer learning: fine-tuning, feature extraction, domain adaptation.

/// Transfer learning with fine-tuning.
pub struct FineTuning {
    pub pretrained_weights: Vec<Vec<f64>>,
    pub task_weights: Vec<Vec<f64>>,
    pub freeze_layers: usize,
    pub learning_rate: f64,
}

impl FineTuning {
    pub fn new(pretrained_weights: Vec<Vec<f64>>, task_dim: usize, freeze_layers: usize, learning_rate: f64) -> Self {
        let feature_dim = pretrained_weights[0].len();
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / feature_dim as f64).sqrt();

        Self {
            pretrained_weights, freeze_layers, learning_rate,
            task_weights: (0..task_dim).map(|_| (0..feature_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    /// Extract features using pretrained weights.
    pub fn extract_features(&self, x: &[f64]) -> Vec<f64> {
        let mut current = x.to_vec();
        for (i, layer) in self.pretrained_weights.iter().enumerate() {
            current = layer.iter().map(|w| {
                w.iter().zip(current.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
            }).collect();
        }
        current
    }

    /// Forward through task-specific head.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let features = self.extract_features(x);
        self.task_weights.iter().map(|w| {
            w.iter().zip(features.iter()).map(|(wi, fi)| wi * fi).sum()
        }).collect()
    }

    /// Update only task head and unfrozen layers.
    pub fn update(&mut self, x: &[f64], target: &[f64]) -> f64 {
        let features = self.extract_features(x);
        let pred = self.forward(x);

        let loss: f64 = pred.iter().zip(target.iter()).map(|(p, t)| (p - t).powi(2)).sum();

        // Update task weights
        for (i, w_row) in self.task_weights.iter_mut().enumerate() {
            let error = pred[i] - target[i];
            for (j, w) in w_row.iter_mut().enumerate() {
                *w -= self.learning_rate * 2.0 * error * features[j];
            }
        }

        loss
    }
}

/// Feature extraction transfer (freeze all pretrained).
pub struct FeatureExtraction {
    pub pretrained_weights: Vec<Vec<f64>>,
    pub classifier_weights: Vec<Vec<f64>>,
    pub learning_rate: f64,
}

impl FeatureExtraction {
    pub fn new(pretrained_weights: Vec<Vec<f64>>, n_classes: usize, learning_rate: f64) -> Self {
        let feature_dim = pretrained_weights.last().unwrap().len();
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / feature_dim as f64).sqrt();

        Self {
            pretrained_weights, learning_rate,
            classifier_weights: (0..n_classes).map(|_| (0..feature_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn extract_features(&self, x: &[f64]) -> Vec<f64> {
        let mut current = x.to_vec();
        for layer in &self.pretrained_weights {
            current = layer.iter().map(|w| {
                w.iter().zip(current.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
            }).collect();
        }
        current
    }

    pub fn classify(&self, x: &[f64]) -> Vec<f64> {
        let features = self.extract_features(x);
        let logits: Vec<f64> = self.classifier_weights.iter().map(|w| {
            w.iter().zip(features.iter()).map(|(wi, fi)| wi * fi).sum()
        }).collect();
        softmax(&logits)
    }

    pub fn update(&mut self, x: &[f64], class_label: usize) -> f64 {
        let features = self.extract_features(x);
        let probs = self.classify(x);

        let loss = -probs[class_label].max(1e-15).ln();

        // Update classifier only
        for (i, w_row) in self.classifier_weights.iter_mut().enumerate() {
            let error = probs[i] - if i == class_label { 1.0 } else { 0.0 };
            for (j, w) in w_row.iter_mut().enumerate() {
                *w -= self.learning_rate * error * features[j];
            }
        }

        loss
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Gradual unfreezing strategy.
pub struct GradualUnfreezing {
    pub total_layers: usize,
    pub unfreeze_per_epoch: usize,
    pub current_frozen: usize,
}

impl GradualUnfreezing {
    pub fn new(total_layers: usize, unfreeze_per_epoch: usize) -> Self {
        Self {
            total_layers, unfreeze_per_epoch,
            current_frozen: total_layers, // Start fully frozen
        }
    }

    /// Get number of frozen layers for current epoch.
    pub fn frozen_layers(&self, epoch: usize) -> usize {
        let unfrozen = (epoch + 1) * self.unfreeze_per_epoch;
        self.total_layers.saturating_sub(unfrozen)
    }

    /// Update frozen count.
    pub fn epoch_update(&mut self) {
        self.current_frozen = self.current_frozen.saturating_sub(self.unfreeze_per_epoch);
    }
}

/// Discriminative fine-tuning: different learning rates per layer.
pub struct DiscriminativeFineTuning {
    pub base_lr: f64,
    pub lr_decay: f64,
    pub n_layers: usize,
}

impl DiscriminativeFineTuning {
    pub fn new(base_lr: f64, lr_decay: f64, n_layers: usize) -> Self {
        Self { base_lr, lr_decay, n_layers }
    }

    /// Get learning rate for specific layer.
    pub fn get_lr(&self, layer_idx: usize) -> f64 {
        // Earlier layers get smaller learning rates
        self.base_lr * self.lr_decay.powi((self.n_layers - 1 - layer_idx) as i32)
    }

    /// Get all learning rates.
    pub fn get_all_lrs(&self) -> Vec<f64> {
        (0..self.n_layers).map(|i| self.get_lr(i)).collect()
    }
}

/// Transferability estimation.
pub fn estimate_transferability(source_features: &[Vec<f64>], target_features: &[Vec<f64>]) -> f64 {
    // Compute MMD between source and target features
    let n_s = source_features.len();
    let n_t = target_features.len();

    let mut xx = 0.0;
    for i in 0..n_s {
        for j in i + 1..n_s {
            let dist: f64 = source_features[i].iter().zip(source_features[j].iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            xx += (-dist).exp();
        }
    }
    xx /= (n_s * (n_s - 1)) as f64;

    let mut yy = 0.0;
    for i in 0..n_t {
        for j in i + 1..n_t {
            let dist: f64 = target_features[i].iter().zip(target_features[j].iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            yy += (-dist).exp();
        }
    }
    yy /= (n_t * (n_t - 1)) as f64;

    let mut xy = 0.0;
    for i in 0..n_s {
        for j in 0..n_t {
            let dist: f64 = source_features[i].iter().zip(target_features[j].iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            xy += (-dist).exp();
        }
    }
    xy /= (n_s * n_t) as f64;

    // Lower MMD = more transferable
    1.0 / (1.0 + (xx + yy - 2.0 * xy).abs())
}

/// Feature space transfer: align feature distributions.
pub fn align_features(source: &[Vec<f64>], target: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let d = source[0].len();
    let mean_s = compute_mean(source);
    let mean_t = compute_mean(target);

    // Simple mean alignment
    source.iter().map(|x| {
        x.iter().zip(mean_s.iter()).zip(mean_t.iter())
            .map(|((xi, ms), mt)| xi - ms + mt)
            .collect()
    }).collect()
}

fn compute_mean(data: &[Vec<f64>]) -> Vec<f64> {
    let d = data[0].len();
    let n = data.len() as f64;
    (0..d).map(|j| data.iter().map(|x| x[j]).sum::<f64>() / n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradual_unfreezing() {
        let mut gu = GradualUnfreezing::new(10, 2);
        assert_eq!(gu.frozen_layers(0), 8); // Unfreeze 2 at epoch 0
        assert_eq!(gu.frozen_layers(4), 0); // All unfrozen by epoch 4
    }

    #[test]
    fn test_discriminative_lr() {
        let dft = DiscriminativeFineTuning::new(0.01, 0.1, 5);
        let lrs = dft.get_all_lrs();
        assert!(lrs[0] > lrs[4]); // Earlier layers have smaller LR
    }

    #[test]
    fn test_transferability() {
        let source = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let target = vec![vec![1.1, 0.1], vec![0.1, 1.1]];
        let score = estimate_transferability(&source, &target);
        assert!(score > 0.0 && score <= 1.0);
    }
}
