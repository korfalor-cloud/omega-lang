/// Linear regression using gradient descent with optional regularization.

#[derive(Debug, Clone)]
pub struct LinearRegression {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    epochs: usize,
    regularization: Regularization,
    lambda: f64,
    losses: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum Regularization {
    None,
    L1,
    L2,
    ElasticNet(f64),
}

impl LinearRegression {
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate: 0.01,
            epochs: 1000,
            regularization: Regularization::None,
            lambda: 0.01,
            losses: Vec::new(),
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

    pub fn l2_regularization(mut self, lambda: f64) -> Self {
        self.regularization = Regularization::L2;
        self.lambda = lambda;
        self
    }

    pub fn l1_regularization(mut self, lambda: f64) -> Self {
        self.regularization = Regularization::L1;
        self.lambda = lambda;
        self
    }

    pub fn elastic_net(mut self, lambda: f64, ratio: f64) -> Self {
        self.regularization = Regularization::ElasticNet(ratio);
        self.lambda = lambda;
        self
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        assert!(!x.is_empty() && x.len() == y.len());
        let n_features = x[0].len();
        let n_samples = x.len() as f64;

        self.weights = vec![0.0; n_features];
        self.bias = 0.0;
        self.losses.clear();

        for _ in 0..self.epochs {
            let mut predictions = Vec::with_capacity(x.len());
            for row in x {
                predictions.push(self.predict_single(row));
            }

            let mut dw = vec![0.0; n_features];
            let mut db = 0.0;

            for i in 0..x.len() {
                let error = predictions[i] - y[i];
                for j in 0..n_features {
                    dw[j] += error * x[i][j];
                }
                db += error;
            }

            for j in 0..n_features {
                dw[j] /= n_samples;
            }
            db /= n_samples;

            // Apply regularization gradient
            match &self.regularization {
                Regularization::None => {}
                Regularization::L1 => {
                    for j in 0..n_features {
                        dw[j] += self.lambda * self.weights[j].signum();
                    }
                }
                Regularization::L2 => {
                    for j in 0..n_features {
                        dw[j] += self.lambda * 2.0 * self.weights[j];
                    }
                }
                Regularization::ElasticNet(ratio) => {
                    for j in 0..n_features {
                        dw[j] += self.lambda * ratio * self.weights[j].signum();
                        dw[j] += self.lambda * (1.0 - ratio) * 2.0 * self.weights[j];
                    }
                }
            }

            for j in 0..n_features {
                self.weights[j] -= self.learning_rate * dw[j];
            }
            self.bias -= self.learning_rate * db;

            // Compute MSE loss
            let mse: f64 = predictions.iter().zip(y.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>() / n_samples;
            self.losses.push(mse);
        }
    }

    fn predict_single(&self, x: &[f64]) -> f64 {
        let mut result = self.bias;
        for (w, xi) in self.weights.iter().zip(x.iter()) {
            result += w * xi;
        }
        result
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| self.predict_single(row)).collect()
    }

    pub fn r_squared(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;

        let ss_res: f64 = y.iter().zip(predictions.iter())
            .map(|(yi, pi)| (yi - pi).powi(2))
            .sum();
        let ss_tot: f64 = y.iter()
            .map(|yi| (yi - mean_y).powi(2))
            .sum();

        1.0 - ss_res / ss_tot
    }

    pub fn mae(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        predictions.iter().zip(y.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>() / y.len() as f64
    }

    pub fn rmse(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let mse: f64 = self.predict(x).iter().zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / y.len() as f64;
        mse.sqrt()
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

/// Polynomial feature expansion
pub fn polynomial_features(x: &[Vec<f64>], degree: usize) -> Vec<Vec<f64>> {
    x.iter().map(|row| {
        let mut expanded = row.clone();
        for d in 2..=degree {
            for val in row {
                expanded.push(val.powi(d as i32));
            }
        }
        expanded
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_linear_regression() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let mut model = LinearRegression::new()
            .learning_rate(0.01)
            .epochs(5000);
        model.fit(&x, &y);

        let pred = model.predict(&[vec![6.0]]);
        assert!((pred[0] - 12.0).abs() < 0.5);
    }

    #[test]
    fn test_multiple_features() {
        let x = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        let y = vec![3.0, 6.0, 9.0];

        let mut model = LinearRegression::new()
            .learning_rate(0.01)
            .epochs(5000);
        model.fit(&x, &y);

        let r2 = model.r_squared(&x, &y);
        assert!(r2 > 0.9);
    }

    #[test]
    fn test_l2_regularization() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0];

        let mut model = LinearRegression::new()
            .learning_rate(0.01)
            .epochs(3000)
            .l2_regularization(0.001);
        model.fit(&x, &y);

        assert!(!model.weights().is_empty());
    }

    #[test]
    fn test_polynomial_features() {
        let x = vec![vec![2.0], vec![3.0]];
        let expanded = polynomial_features(&x, 2);
        assert_eq!(expanded[0], vec![2.0, 4.0]);
        assert_eq!(expanded[1], vec![3.0, 9.0]);
    }

    #[test]
    fn test_metrics() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![2.0, 4.0, 6.0];

        let mut model = LinearRegression::new().epochs(5000);
        model.fit(&x, &y);

        let mae = model.mae(&x, &y);
        let rmse = model.rmse(&x, &y);
        assert!(mae < 1.0);
        assert!(rmse < 1.0);
    }
}
