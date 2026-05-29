/// Domain adaptation: adversarial, maximum mean discrepancy, correlation alignment.

/// Maximum Mean Discrepancy (MMD) with Gaussian kernel.
pub fn mmd_gaussian(source: &[Vec<f64>], target: &[Vec<f64>], sigma: f64) -> f64 {
    let n = source.len();
    let m = target.len();

    let mut xx = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            xx += gaussian_kernel(&source[i], &source[j], sigma);
        }
    }
    xx /= (n * (n - 1)) as f64;

    let mut yy = 0.0;
    for i in 0..m {
        for j in i + 1..m {
            yy += gaussian_kernel(&target[i], &target[j], sigma);
        }
    }
    yy /= (m * (m - 1)) as f64;

    let mut xy = 0.0;
    for i in 0..n {
        for j in 0..m {
            xy += gaussian_kernel(&source[i], &target[j], sigma);
        }
    }
    xy /= (n * m) as f64;

    (xx + yy - 2.0 * xy).max(0.0)
}

fn gaussian_kernel(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    let dist_sq: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    (-dist_sq / (2.0 * sigma * sigma)).exp()
}

/// Deep Domain Adaptation: domain classifier.
pub struct DomainClassifier {
    pub feature_dim: usize,
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub learning_rate: f64,
}

impl DomainClassifier {
    pub fn new(feature_dim: usize, learning_rate: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / feature_dim as f64).sqrt();
        Self {
            feature_dim, learning_rate,
            weights: (0..1).map(|_| (0..feature_dim).map(|_| rand(scale)).collect()).collect(),
            bias: vec![0.0],
        }
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        let logit: f64 = self.weights[0].iter().zip(features.iter()).map(|(w, f)| w * f).sum::<f64>() + self.bias[0];
        sigmoid(logit)
    }

    pub fn update(&mut self, features: &[f64], is_source: bool, learning_rate: f64) {
        let pred = self.predict(features);
        let target = if is_source { 1.0 } else { 0.0 };
        let error = pred - target;

        for (i, w) in self.weights[0].iter_mut().enumerate() {
            *w -= learning_rate * error * features[i.min(features.len() - 1)];
        }
        self.bias[0] -= learning_rate * error;
    }

    pub fn domain_loss(&self, features: &[f64], is_source: bool) -> f64 {
        let pred = self.predict(features);
        let target = if is_source { 1.0 } else { 0.0 };
        -(target * pred.max(1e-15).ln() + (1.0 - target) * (1.0 - pred).max(1e-15).ln())
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// CORAL (CORrelation ALignment).
pub fn coral(source: &[Vec<f64>], target: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let d = source[0].len();
    let n_s = source.len();
    let n_t = target.len();

    // Compute covariance matrices
    let mean_s = compute_mean(source);
    let mean_t = compute_mean(target);

    let cov_s = compute_covariance(source, &mean_s);
    let cov_t = compute_covariance(target, &mean_t);

    // Whiten source: C_s^{-1/2}
    let cov_s_sqrt_inv = matrix_sqrt_inv(&cov_s);
    // Re-color: C_t^{1/2}
    let cov_t_sqrt = matrix_sqrt(&cov_t);

    // Transform: X_s * C_s^{-1/2} * C_t^{1/2}
    let transform = mat_mul(&cov_s_sqrt_inv, &cov_t_sqrt);

    source.iter().map(|x| {
        let centered: Vec<f64> = x.iter().zip(mean_s.iter()).map(|(xi, mi)| xi - mi).collect();
        let transformed = mat_vec_mul(&transform, &centered);
        transformed.iter().zip(mean_t.iter()).map(|(ti, mi)| ti + mi).collect()
    }).collect()
}

fn compute_mean(data: &[Vec<f64>]) -> Vec<f64> {
    let d = data[0].len();
    let n = data.len() as f64;
    (0..d).map(|j| data.iter().map(|x| x[j]).sum::<f64>() / n).collect()
}

fn compute_covariance(data: &[Vec<f64>], mean: &[f64]) -> Vec<Vec<f64>> {
    let d = mean.len();
    let n = data.len();
    let mut cov = vec![vec![0.0; d]; d];

    for x in data {
        for i in 0..d {
            for j in 0..d {
                cov[i][j] += (x[i] - mean[i]) * (x[j] - mean[j]);
            }
        }
    }

    for row in cov.iter_mut() {
        for val in row.iter_mut() {
            *val /= n as f64;
        }
    }

    cov
}

fn matrix_sqrt(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    // Simplified: use identity (proper implementation would use eigendecomposition)
    let n = m.len();
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        result[i][i] = m[i][i].sqrt();
    }
    result
}

fn matrix_sqrt_inv(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        result[i][i] = if m[i][i] > 1e-10 { 1.0 / m[i][i].sqrt() } else { 1.0 };
    }
    result
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
}

/// Adversarial Domain Adaptation with Gradient Reversal Layer.
pub struct AdversarialDA {
    pub feature_dim: usize,
    pub encoder_weights: Vec<Vec<f64>>,
    pub classifier_weights: Vec<Vec<f64>>,
    pub domain_weights: Vec<Vec<f64>>,
    pub learning_rate: f64,
}

impl AdversarialDA {
    pub fn new(feature_dim: usize, n_classes: usize, learning_rate: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / feature_dim as f64).sqrt();

        Self {
            feature_dim, learning_rate,
            encoder_weights: (0..feature_dim).map(|_| (0..feature_dim).map(|_| rand(scale)).collect()).collect(),
            classifier_weights: (0..n_classes).map(|_| (0..feature_dim).map(|_| rand(scale)).collect()).collect(),
            domain_weights: (0..1).map(|_| (0..feature_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        self.encoder_weights.iter().map(|w| {
            w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
        }).collect()
    }

    pub fn classify(&self, features: &[f64]) -> Vec<f64> {
        let logits: Vec<f64> = self.classifier_weights.iter().map(|w| {
            w.iter().zip(features.iter()).map(|(wi, fi)| wi * fi).sum()
        }).collect();
        softmax(&logits)
    }

    pub fn domain_predict(&self, features: &[f64]) -> f64 {
        let logit: f64 = self.domain_weights[0].iter().zip(features.iter()).map(|(w, f)| w * f).sum();
        sigmoid(logit)
    }

    /// Train step with gradient reversal.
    pub fn train_step(&mut self, x: &[f64], class_label: usize, is_source: bool) -> f64 {
        let features = self.encode(x);
        let class_probs = self.classify(&features);
        let domain_pred = self.domain_predict(&features);

        // Classification loss (only for source)
        let class_loss = if is_source {
            -class_probs[class_label].max(1e-15).ln()
        } else {
            0.0
        };

        // Domain loss
        let domain_target = if is_source { 1.0 } else { 0.0 };
        let domain_loss = -(domain_target * domain_pred.max(1e-15).ln()
            + (1.0 - domain_target) * (1.0 - domain_pred).max(1e-15).ln());

        // Update classifier
        if is_source {
            for (i, w_row) in self.classifier_weights.iter_mut().enumerate() {
                let grad = class_probs[i] - if i == class_label { 1.0 } else { 0.0 };
                for (j, w) in w_row.iter_mut().enumerate() {
                    *w -= self.learning_rate * grad * features[j];
                }
            }
        }

        // Update domain classifier (with gradient reversal for encoder)
        let domain_grad = domain_pred - domain_target;
        for (j, w) in self.domain_weights[0].iter_mut().enumerate() {
            *w -= self.learning_rate * domain_grad * features[j];
        }

        // Gradient reversal: encoder tries to fool domain classifier
        for (i, w_row) in self.encoder_weights.iter_mut().enumerate() {
            for (j, w) in w_row.iter_mut().enumerate() {
                // Reversed gradient
                *w += self.learning_rate * domain_grad * x[j.min(x.len() - 1)] / self.feature_dim as f64;
            }
        }

        class_loss + domain_loss
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Domain-Adversarial Neural Network (DANN).
pub struct DANN {
    pub feature_extractor: Vec<Vec<f64>>,
    pub task_classifier: Vec<Vec<f64>>,
    pub domain_classifier: Vec<Vec<f64>>,
    pub learning_rate: f64,
    pub lambda: f64, // Gradient reversal coefficient
}

impl DANN {
    pub fn new(input_dim: usize, feature_dim: usize, n_classes: usize, learning_rate: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale_in = (2.0 / input_dim as f64).sqrt();
        let scale_feat = (2.0 / feature_dim as f64).sqrt();

        Self {
            learning_rate, lambda: 1.0,
            feature_extractor: (0..feature_dim).map(|_| (0..input_dim).map(|_| rand(scale_in)).collect()).collect(),
            task_classifier: (0..n_classes).map(|_| (0..feature_dim).map(|_| rand(scale_feat)).collect()).collect(),
            domain_classifier: (0..1).map(|_| (0..feature_dim).map(|_| rand(scale_feat)).collect()).collect(),
        }
    }

    pub fn extract_features(&self, x: &[f64]) -> Vec<f64> {
        self.feature_extractor.iter().map(|w| {
            w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
        }).collect()
    }

    pub fn predict_class(&self, features: &[f64]) -> Vec<f64> {
        let logits: Vec<f64> = self.task_classifier.iter().map(|w| {
            w.iter().zip(features.iter()).map(|(wi, fi)| wi * fi).sum()
        }).collect();
        softmax(&logits)
    }

    pub fn predict_domain(&self, features: &[f64]) -> f64 {
        let logit: f64 = self.domain_classifier[0].iter().zip(features.iter()).map(|(w, f)| w * f).sum();
        sigmoid(logit)
    }
}

/// Domain-Adversarial Training.
pub struct DomainAdversarialTraining {
    pub source_features: Vec<Vec<f64>>,
    pub target_features: Vec<Vec<f64>>,
    pub mmd_weight: f64,
}

impl DomainAdversarialTraining {
    pub fn new(mmd_weight: f64) -> Self {
        Self {
            source_features: Vec::new(),
            target_features: Vec::new(),
            mmd_weight,
        }
    }

    pub fn add_source(&mut self, features: Vec<f64>) {
        self.source_features.push(features);
    }

    pub fn add_target(&mut self, features: Vec<f64>) {
        self.target_features.push(features);
    }

    pub fn compute_mmd_loss(&self, sigma: f64) -> f64 {
        self.mmd_weight * mmd_gaussian(&self.source_features, &self.target_features, sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmd() {
        let source = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let target = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let mmd = mmd_gaussian(&source, &target, 1.0);
        assert!(mmd > 0.0);
    }

    #[test]
    fn test_coral() {
        let source = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let target = vec![vec![2.0, 3.0], vec![4.0, 5.0], vec![6.0, 7.0]];
        let aligned = coral(&source, &target);
        assert_eq!(aligned.len(), 3);
        assert_eq!(aligned[0].len(), 2);
    }

    #[test]
    fn test_domain_classifier() {
        let mut dc = DomainClassifier::new(4, 0.01);
        let features = vec![1.0, 0.0, 0.0, 0.0];
        let pred = dc.predict(&features);
        assert!(pred > 0.0 && pred < 1.0);
    }
}
