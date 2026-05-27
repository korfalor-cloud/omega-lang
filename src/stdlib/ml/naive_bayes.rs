/// Gaussian Naive Bayes classifier.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct NaiveBayes {
    class_priors: HashMap<i64, f64>,
    means: HashMap<i64, Vec<f64>>,
    variances: HashMap<i64, Vec<f64>>,
    classes: Vec<i64>,
    smoothing: f64,
}

impl NaiveBayes {
    pub fn new() -> Self {
        Self {
            class_priors: HashMap::new(),
            means: HashMap::new(),
            variances: HashMap::new(),
            classes: Vec::new(),
            smoothing: 1e-9,
        }
    }

    pub fn smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing;
        self
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        assert!(!x.is_empty() && x.len() == y.len());
        let n_features = x[0].len();
        let n_samples = x.len() as f64;

        // Group by class
        let mut groups: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            groups.entry(label as i64).or_insert_with(Vec::new).push(i);
        }

        self.classes = groups.keys().cloned().collect();
        self.classes.sort();

        for (&class, indices) in &groups {
            let n_class = indices.len() as f64;
            self.class_priors.insert(class, n_class / n_samples);

            // Compute mean per feature
            let mut mean = vec![0.0; n_features];
            for &i in indices {
                for j in 0..n_features {
                    mean[j] += x[i][j];
                }
            }
            for val in mean.iter_mut() {
                *val /= n_class;
            }

            // Compute variance per feature
            let mut variance = vec![0.0; n_features];
            for &i in indices {
                for j in 0..n_features {
                    variance[j] += (x[i][j] - mean[j]).powi(2);
                }
            }
            for val in variance.iter_mut() {
                *val = *val / n_class + self.smoothing;
            }

            self.means.insert(class, mean);
            self.variances.insert(class, variance);
        }
    }

    fn log_likelihood(&self, x: &[f64], class: i64) -> f64 {
        let mean = self.means.get(&class).unwrap();
        let variance = self.variances.get(&class).unwrap();

        let mut log_lik = 0.0;
        for j in 0..x.len() {
            let diff = x[j] - mean[j];
            log_lik -= 0.5 * (2.0 * std::f64::consts::PI * variance[j]).ln();
            log_lik -= 0.5 * diff * diff / variance[j];
        }
        log_lik
    }

    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Vec<HashMap<i64, f64>> {
        x.iter().map(|row| {
            let mut log_probs: HashMap<i64, f64> = HashMap::new();
            for &class in &self.classes {
                let log_prior = self.class_priors[&class].ln();
                log_probs.insert(class, log_prior + self.log_likelihood(row, class));
            }

            // Convert log probabilities to probabilities using log-sum-exp trick
            let max_log = log_probs.values().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut probs: HashMap<i64, f64> = HashMap::new();
            let mut sum = 0.0;
            for (&class, &log_prob) in &log_probs {
                let prob = (log_prob - max_log).exp();
                probs.insert(class, prob);
                sum += prob;
            }
            for prob in probs.values_mut() {
                *prob /= sum;
            }
            probs
        }).collect()
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<i64> {
        self.predict_proba(x).iter()
            .map(|probs| *probs.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0)
            .collect()
    }

    pub fn accuracy(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let correct = predictions.iter().zip(y.iter())
            .filter(|(p, t)| **p == *t as i64)
            .count();
        correct as f64 / y.len() as f64
    }

    pub fn classes(&self) -> &[i64] {
        &self.classes
    }

    pub fn class_priors(&self) -> &HashMap<i64, f64> {
        &self.class_priors
    }
}

/// Multinomial Naive Bayes for count data
#[derive(Debug, Clone)]
pub struct MultinomialNaiveBayes {
    class_log_priors: HashMap<i64, f64>,
    feature_log_probs: HashMap<i64, Vec<f64>>,
    classes: Vec<i64>,
    alpha: f64,
}

impl MultinomialNaiveBayes {
    pub fn new(alpha: f64) -> Self {
        Self {
            class_log_priors: HashMap::new(),
            feature_log_probs: HashMap::new(),
            classes: Vec::new(),
            alpha,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let n_features = x[0].len();
        let n_samples = x.len() as f64;

        let mut groups: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            groups.entry(label as i64).or_insert_with(Vec::new).push(i);
        }

        self.classes = groups.keys().cloned().collect();
        self.classes.sort();

        for (&class, indices) in &groups {
            let n_class = indices.len() as f64;
            self.class_log_priors.insert(class, (n_class / n_samples).ln());

            let mut feature_counts = vec![0.0; n_features];
            let mut total_count = 0.0;
            for &i in indices {
                for j in 0..n_features {
                    feature_counts[j] += x[i][j];
                    total_count += x[i][j];
                }
            }

            let mut log_probs = vec![0.0; n_features];
            let denominator = total_count + self.alpha * n_features as f64;
            for j in 0..n_features {
                log_probs[j] = ((feature_counts[j] + self.alpha) / denominator).ln();
            }

            self.feature_log_probs.insert(class, log_probs);
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<i64> {
        x.iter().map(|row| {
            let mut best_class = self.classes[0];
            let mut best_score = f64::NEG_INFINITY;

            for &class in &self.classes {
                let mut score = self.class_log_priors[&class];
                let log_probs = &self.feature_log_probs[&class];
                for (j, &val) in row.iter().enumerate() {
                    score += val * log_probs[j];
                }
                if score > best_score {
                    best_score = score;
                    best_class = class;
                }
            }
            best_class
        }).collect()
    }

    pub fn accuracy(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let correct = predictions.iter().zip(y.iter())
            .filter(|(p, t)| **p == *t as i64)
            .count();
        correct as f64 / y.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_naive_bayes() {
        let x = vec![
            vec![1.0, 1.0], vec![1.0, 2.0], vec![2.0, 1.0],
            vec![5.0, 5.0], vec![5.0, 6.0], vec![6.0, 5.0],
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut nb = NaiveBayes::new();
        nb.fit(&x, &y);

        let predictions = nb.predict(&[vec![1.5, 1.5], vec![5.5, 5.5]]);
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 1);
    }

    #[test]
    fn test_predict_proba() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![8.0], vec![9.0], vec![10.0]];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut nb = NaiveBayes::new();
        nb.fit(&x, &y);

        let probas = nb.predict_proba(&[vec![1.5]]);
        assert!(probas[0][&0] > 0.5);
    }

    #[test]
    fn test_multinomial_nb() {
        let x = vec![
            vec![1.0, 0.0, 0.0], vec![1.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0],
        ];
        let y = vec![0.0, 0.0, 1.0, 1.0];

        let mut mnb = MultinomialNaiveBayes::new(1.0);
        mnb.fit(&x, &y);

        let predictions = mnb.predict(&[vec![1.0, 0.0, 0.0]]);
        assert_eq!(predictions[0], 0);
    }
}
