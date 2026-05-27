/// Logistic regression for binary classification.

#[derive(Debug, Clone)]
pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    epochs: usize,
    lambda: f64,
    losses: Vec<f64>,
    threshold: f64,
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate: 0.01,
            epochs: 1000,
            lambda: 0.0,
            losses: Vec::new(),
            threshold: 0.5,
        }
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn regularization(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        assert!(!x.is_empty() && x.len() == y.len());
        let n_features = x[0].len();
        let n_samples = x.len() as f64;

        self.weights = vec![0.0; n_features];
        self.bias = 0.0;
        self.losses.clear();

        for _ in 0..self.epochs {
            let mut total_loss = 0.0;
            let mut dw = vec![0.0; n_features];
            let mut db = 0.0;

            for i in 0..x.len() {
                let z = self.dot(&x[i]) + self.bias;
                let a = Self::sigmoid(z);
                let error = a - y[i];

                for j in 0..n_features {
                    dw[j] += error * x[i][j];
                }
                db += error;

                // Binary cross-entropy
                let eps = 1e-15;
                total_loss -= y[i] * (a + eps).ln() + (1.0 - y[i]) * (1.0 - a + eps).ln();
            }

            for j in 0..n_features {
                dw[j] = dw[j] / n_samples + self.lambda * self.weights[j];
            }
            db /= n_samples;

            for j in 0..n_features {
                self.weights[j] -= self.learning_rate * dw[j];
            }
            self.bias -= self.learning_rate * db;

            self.losses.push(total_loss / n_samples);
        }
    }

    fn dot(&self, x: &[f64]) -> f64 {
        self.weights.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum()
    }

    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| Self::sigmoid(self.dot(row) + self.bias)).collect()
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        self.predict_proba(x).iter().map(|&p| if p >= self.threshold { 1.0 } else { 0.0 }).collect()
    }

    pub fn accuracy(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let correct = predictions.iter().zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-10)
            .count();
        correct as f64 / y.len() as f64
    }

    pub fn precision(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let tp = predictions.iter().zip(y.iter()).filter(|(p, t)| **p == 1.0 && **t == 1.0).count() as f64;
        let fp = predictions.iter().zip(y.iter()).filter(|(p, t)| **p == 1.0 && **t == 0.0).count() as f64;
        if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) }
    }

    pub fn recall(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let tp = predictions.iter().zip(y.iter()).filter(|(p, t)| **p == 1.0 && **t == 1.0).count() as f64;
        let fn_count = predictions.iter().zip(y.iter()).filter(|(p, t)| **p == 0.0 && **t == 1.0).count() as f64;
        if tp + fn_count == 0.0 { 0.0 } else { tp / (tp + fn_count) }
    }

    pub fn f1_score(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let p = self.precision(x, y);
        let r = self.recall(x, y);
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }

    pub fn confusion_matrix(&self, x: &[Vec<f64>], y: &[f64]) -> [[usize; 2]; 2] {
        let predictions = self.predict(x);
        let mut cm = [[0usize; 2]; 2];
        for (p, t) in predictions.iter().zip(y.iter()) {
            let pi = *p as usize;
            let ti = *t as usize;
            cm[ti][pi] += 1;
        }
        cm
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn losses(&self) -> &[f64] {
        &self.losses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_regression_and_gate() {
        let x = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        let y = vec![0.0, 0.0, 0.0, 1.0];

        let mut model = LogisticRegression::new()
            .learning_rate(0.5)
            .epochs(5000);
        model.fit(&x, &y);

        let predictions = model.predict(&x);
        assert_eq!(predictions[3], 1.0);
        assert_eq!(predictions[0], 0.0);
    }

    #[test]
    fn test_sigmoid() {
        let s = LogisticRegression::sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        assert!(LogisticRegression::sigmoid(10.0) > 0.99);
        assert!(LogisticRegression::sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_confusion_matrix() {
        let x = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        let y = vec![0.0, 0.0, 0.0, 1.0];

        let mut model = LogisticRegression::new().epochs(5000);
        model.fit(&x, &y);

        let cm = model.confusion_matrix(&x, &y);
        assert_eq!(cm[0][0], 3); // TN
        assert_eq!(cm[1][1], 1); // TP
    }

    #[test]
    fn test_metrics() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![0.0, 0.0, 1.0, 1.0];

        let mut model = LogisticRegression::new().epochs(5000);
        model.fit(&x, &y);

        let acc = model.accuracy(&x, &y);
        assert!(acc >= 0.5);
    }
}
