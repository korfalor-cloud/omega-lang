/// Ensemble methods: bagging, boosting, stacking, random forest.

/// Bagging ensemble.
pub struct BaggingEnsemble {
    pub models: Vec<Vec<f64>>,
    pub n_models: usize,
    pub sample_ratio: f64,
    seed: u64,
}

impl BaggingEnsemble {
    pub fn new(n_models: usize, sample_ratio: f64) -> Self {
        Self { models: Vec::new(), n_models, sample_ratio, seed: 42 }
    }

    /// Train ensemble with bootstrap sampling.
    pub fn train<F>(&mut self, data: &[(Vec<f64>, f64)], train_fn: F)
    where
        F: Fn(&[(Vec<f64>, f64)]) -> Vec<f64>,
    {
        let n = data.len();
        let sample_size = (n as f64 * self.sample_ratio) as usize;

        for _ in 0..self.n_models {
            // Bootstrap sample
            let mut sample = Vec::new();
            for _ in 0..sample_size {
                let idx = ((self.seed >> 33) as usize) % n;
                self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                sample.push(data[idx].clone());
            }

            let model = train_fn(&sample);
            self.models.push(model);
        }
    }

    /// Predict by averaging.
    pub fn predict(&self, x: &[f64]) -> f64 {
        let predictions: Vec<f64> = self.models.iter().map(|model| {
            model.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum()
        }).collect();

        predictions.iter().sum::<f64>() / predictions.len() as f64
    }

    /// Predict with confidence (standard deviation).
    pub fn predict_with_confidence(&self, x: &[f64]) -> (f64, f64) {
        let predictions: Vec<f64> = self.models.iter().map(|model| {
            model.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum()
        }).collect();

        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / predictions.len() as f64;
        (mean, variance.sqrt())
    }
}

/// AdaBoost.
pub struct AdaBoost {
    pub models: Vec<Vec<f64>>,
    pub alphas: Vec<f64>,
    pub n_models: usize,
}

impl AdaBoost {
    pub fn new(n_models: usize) -> Self {
        Self { models: Vec::new(), alphas: Vec::new(), n_models }
    }

    /// Train AdaBoost.
    pub fn train<F>(&mut self, data: &[(Vec<f64>, f64)], train_fn: F)
    where
        F: Fn(&[(Vec<f64>, f64)]) -> Vec<f64>,
    {
        let n = data.len();
        let mut weights = vec![1.0 / n as f64; n];

        for _ in 0..self.n_models {
            // Weighted sample
            let mut weighted_data = data.to_vec();
            // In practice, would sample with weights

            let model = train_fn(&weighted_data);

            // Compute weighted error
            let mut error = 0.0;
            for ((x, y), &w) in data.iter().zip(weights.iter()) {
                let pred: f64 = model.iter().zip(x.iter()).map(|(m, xi)| m * xi).sum();
                if (pred > 0.0) != (*y > 0.0) {
                    error += w;
                }
            }

            error = error.max(1e-10).min(1.0 - 1e-10);
            let alpha = 0.5 * ((1.0 - error) / error).ln();

            // Update weights
            for ((x, y), w) in data.iter().zip(weights.iter_mut()) {
                let pred: f64 = model.iter().zip(x.iter()).map(|(m, xi)| m * xi).sum();
                if (pred > 0.0) != (*y > 0.0) {
                    *w *= (alpha).exp();
                } else {
                    *w *= (-alpha).exp();
                }
            }

            // Normalize
            let total: f64 = weights.iter().sum();
            for w in weights.iter_mut() { *w /= total; }

            self.models.push(model);
            self.alphas.push(alpha);
        }
    }

    /// Predict by weighted vote.
    pub fn predict(&self, x: &[f64]) -> f64 {
        let weighted_sum: f64 = self.models.iter().zip(self.alphas.iter()).map(|(model, alpha)| {
            let pred: f64 = model.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
            alpha * pred.signum()
        }).sum();

        weighted_sum.signum()
    }
}

/// Gradient Boosting.
pub struct GradientBoosting {
    pub models: Vec<Vec<f64>>,
    pub learning_rate: f64,
    pub n_models: usize,
}

impl GradientBoosting {
    pub fn new(n_models: usize, learning_rate: f64) -> Self {
        Self { models: Vec::new(), learning_rate, n_models }
    }

    /// Train gradient boosting.
    pub fn train<F>(&mut self, data: &[(Vec<f64>, f64)], train_fn: F)
    where
        F: Fn(&[(Vec<f64>, f64)]) -> Vec<f64>,
    {
        // Initialize with mean
        let mean: f64 = data.iter().map(|(_, y)| y).sum::<f64>() / data.len() as f64;
        self.models.push(vec![mean]);

        for _ in 1..self.n_models {
            // Compute residuals
            let residuals: Vec<(Vec<f64>, f64)> = data.iter().map(|(x, y)| {
                let pred = self.predict(x);
                (x.clone(), y - pred)
            }).collect();

            let model = train_fn(&residuals);
            self.models.push(model);
        }
    }

    /// Predict by summing all models.
    pub fn predict(&self, x: &[f64]) -> f64 {
        let mut pred = 0.0;
        for (i, model) in self.models.iter().enumerate() {
            if i == 0 {
                pred += model[0]; // Base prediction
            } else {
                let contribution: f64 = model.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
                pred += self.learning_rate * contribution;
            }
        }
        pred
    }
}

/// Stacking ensemble.
pub struct StackingEnsemble {
    pub base_models: Vec<Vec<f64>>,
    pub meta_model: Vec<f64>,
    pub learning_rate: f64,
}

impl StackingEnsemble {
    pub fn new(n_base_models: usize, learning_rate: f64) -> Self {
        Self {
            base_models: Vec::new(),
            meta_model: vec![0.0; n_base_models],
            learning_rate,
        }
    }

    /// Get base model predictions (meta-features).
    pub fn get_meta_features(&self, x: &[f64]) -> Vec<f64> {
        self.base_models.iter().map(|model| {
            model.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum()
        }).collect()
    }

    /// Predict using stacking.
    pub fn predict(&self, x: &[f64]) -> f64 {
        let meta_features = self.get_meta_features(x);
        self.meta_model.iter().zip(meta_features.iter()).map(|(w, f)| w * f).sum()
    }

    /// Train meta-model.
    pub fn train_meta(&mut self, data: &[(Vec<f64>, f64)], n_epochs: usize) {
        for _ in 0..n_epochs {
            for (x, y) in data {
                let meta_features = self.get_meta_features(x);
                let pred = self.predict(x);
                let error = pred - y;

                for (w, f) in self.meta_model.iter_mut().zip(meta_features.iter()) {
                    *w -= self.learning_rate * 2.0 * error * f;
                }
            }
        }
    }
}

/// Random Forest (simplified).
pub struct RandomForest {
    pub trees: Vec<DecisionStump>,
    pub n_trees: usize,
    pub feature_sample_ratio: f64,
    seed: u64,
}

#[derive(Clone)]
pub struct DecisionStump {
    pub feature_idx: usize,
    pub threshold: f64,
    pub left_value: f64,
    pub right_value: f64,
}

impl RandomForest {
    pub fn new(n_trees: usize, feature_sample_ratio: f64) -> Self {
        Self { trees: Vec::new(), n_trees, feature_sample_ratio, seed: 42 }
    }

    pub fn train(&mut self, data: &[(Vec<f64>, f64)]) {
        let n = data.len();
        let n_features = data[0].0.len();
        let n_sample_features = (n_features as f64 * self.feature_sample_ratio) as usize;

        for _ in 0..self.n_trees {
            // Bootstrap sample
            let mut sample = Vec::new();
            for _ in 0..n {
                let idx = ((self.seed >> 33) as usize) % n;
                self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                sample.push(data[idx].clone());
            }

            // Random feature subset
            let mut features: Vec<usize> = (0..n_features).collect();
            for i in (1..n_features).rev() {
                let j = ((self.seed >> 33) as usize) % (i + 1);
                self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                features.swap(i, j);
            }
            let selected_features = &features[..n_sample_features];

            // Find best split
            let mut best_stump = DecisionStump { feature_idx: 0, threshold: 0.0, left_value: 0.0, right_value: 0.0 };
            let mut best_error = f64::INFINITY;

            for &feat in selected_features {
                let values: Vec<f64> = sample.iter().map(|(x, _)| x[feat]).collect();
                let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                for threshold_idx in 0..10 {
                    let threshold = min_val + (max_val - min_val) * threshold_idx as f64 / 10.0;

                    let left: Vec<f64> = sample.iter().filter(|(x, _)| x[feat] <= threshold).map(|(_, y)| *y).collect();
                    let right: Vec<f64> = sample.iter().filter(|(x, _)| x[feat] > threshold).map(|(_, y)| *y).collect();

                    if left.is_empty() || right.is_empty() { continue; }

                    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
                    let right_mean = right.iter().sum::<f64>() / right.len() as f64;

                    let error: f64 = left.iter().map(|y| (y - left_mean).powi(2)).sum::<f64>()
                        + right.iter().map(|y| (y - right_mean).powi(2)).sum::<f64>();

                    if error < best_error {
                        best_error = error;
                        best_stump = DecisionStump {
                            feature_idx: feat,
                            threshold,
                            left_value: left_mean,
                            right_value: right_mean,
                        };
                    }
                }
            }

            self.trees.push(best_stump);
        }
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let predictions: Vec<f64> = self.trees.iter().map(|tree| {
            if x[tree.feature_idx] <= tree.threshold {
                tree.left_value
            } else {
                tree.right_value
            }
        }).collect();

        predictions.iter().sum::<f64>() / predictions.len() as f64
    }
}

/// Weighted ensemble.
pub struct WeightedEnsemble {
    pub models: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
}

impl WeightedEnsemble {
    pub fn new() -> Self {
        Self { models: Vec::new(), weights: Vec::new() }
    }

    pub fn add_model(&mut self, model: Vec<f64>, weight: f64) {
        self.models.push(model);
        self.weights.push(weight);
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let weighted_sum: f64 = self.models.iter().zip(self.weights.iter()).map(|(model, weight)| {
            let pred: f64 = model.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
            weight * pred
        }).sum();

        let total_weight: f64 = self.weights.iter().sum();
        weighted_sum / total_weight
    }

    /// Optimize weights using validation data.
    pub fn optimize_weights(&mut self, val_data: &[(Vec<f64>, f64)], n_iterations: usize, learning_rate: f64) {
        for _ in 0..n_iterations {
            for (x, y) in val_data {
                let pred = self.predict(x);
                let error = pred - y;

                for (i, (model, weight)) in self.models.iter().zip(self.weights.iter_mut()).enumerate() {
                    let model_pred: f64 = model.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum();
                    *weight -= learning_rate * 2.0 * error * model_pred;
                    *weight = weight.max(0.0); // Non-negative weights
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bagging() {
        let mut ensemble = BaggingEnsemble::new(5, 0.8);
        let data = vec![
            (vec![1.0, 0.0], 1.0),
            (vec![0.0, 1.0], -1.0),
        ];
        ensemble.train(&data, |data| vec![1.0, -1.0]);
        let pred = ensemble.predict(&[1.0, 0.0]);
        assert!(pred.is_finite());
    }

    #[test]
    fn test_random_forest() {
        let mut rf = RandomForest::new(10, 0.5);
        let data = vec![
            (vec![1.0, 0.0, 0.0], 1.0),
            (vec![0.0, 1.0, 0.0], -1.0),
            (vec![0.0, 0.0, 1.0], 0.5),
        ];
        rf.train(&data);
        let pred = rf.predict(&[1.0, 0.0, 0.0]);
        assert!(pred.is_finite());
    }
}
