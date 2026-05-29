/// Calibration: temperature scaling, Platt scaling, isotonic regression.

/// Temperature scaling for calibration.
pub struct TemperatureScaling {
    pub temperature: f64,
    pub learning_rate: f64,
}

impl TemperatureScaling {
    pub fn new(learning_rate: f64) -> Self {
        Self { temperature: 1.0, learning_rate }
    }

    /// Calibrate logits with temperature.
    pub fn calibrate(&self, logits: &[f64]) -> Vec<f64> {
        let scaled: Vec<f64> = logits.iter().map(|l| l / self.temperature).collect();
        softmax(&scaled)
    }

    /// Optimize temperature on validation set.
    pub fn fit(&mut self, logits: &[Vec<f64>], labels: &[usize], n_iterations: usize) {
        for _ in 0..n_iterations {
            let mut grad = 0.0;

            for (logit, &label) in logits.iter().zip(labels.iter()) {
                let scaled: Vec<f64> = logit.iter().map(|l| l / self.temperature).collect();
                let probs = softmax(&scaled);

                // Gradient of NLL w.r.t. temperature
                for (i, (l, p)) in logit.iter().zip(probs.iter()).enumerate() {
                    let dp_dt = if i == label {
                        -l / (self.temperature * self.temperature) * p * (1.0 - p)
                    } else {
                        l / (self.temperature * self.temperature) * probs[label] * p
                    };
                    grad += dp_dt / probs[label].max(1e-15);
                }
            }

            self.temperature -= self.learning_rate * grad / logits.len() as f64;
            self.temperature = self.temperature.max(0.1);
        }
    }

    /// Expected Calibration Error.
    pub fn ece(&self, probs: &[Vec<f64>], labels: &[usize], n_bins: usize) -> f64 {
        let n = probs.len();
        let mut bin_correct = vec![0usize; n_bins];
        let mut bin_confidence = vec![0.0f64; n_bins];
        let mut bin_count = vec![0usize; n_bins];

        for (prob, &label) in probs.iter().zip(labels.iter()) {
            let predicted = prob.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            let confidence = prob[predicted];
            let bin = (confidence * n_bins as f64) as usize;
            let bin = bin.min(n_bins - 1);

            bin_count[bin] += 1;
            bin_confidence[bin] += confidence;
            if predicted == label {
                bin_correct[bin] += 1;
            }
        }

        let mut ece = 0.0;
        for i in 0..n_bins {
            if bin_count[i] > 0 {
                let avg_confidence = bin_confidence[i] / bin_count[i] as f64;
                let accuracy = bin_correct[i] as f64 / bin_count[i] as f64;
                ece += bin_count[i] as f64 / n as f64 * (accuracy - avg_confidence).abs();
            }
        }

        ece
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Platt scaling.
pub struct PlattScaling {
    pub a: f64,
    pub b: f64,
    pub learning_rate: f64,
}

impl PlattScaling {
    pub fn new(learning_rate: f64) -> Self {
        Self { a: 0.0, b: 0.0, learning_rate }
    }

    /// Calibrate scores.
    pub fn calibrate(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| sigmoid(self.a * s + self.b)).collect()
    }

    /// Fit Platt scaling.
    pub fn fit(&mut self, scores: &[f64], labels: &[bool], n_iterations: usize) {
        for _ in 0..n_iterations {
            for (&score, &label) in scores.iter().zip(labels.iter()) {
                let pred = sigmoid(self.a * score + self.b);
                let target = if label { 1.0 } else { 0.0 };
                let error = pred - target;

                self.a -= self.learning_rate * error * score;
                self.b -= self.learning_rate * error;
            }
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Isotonic regression for calibration.
pub struct IsotonicRegression {
    pub thresholds: Vec<f64>,
    pub calibrated_values: Vec<f64>,
}

impl IsotonicRegression {
    pub fn new() -> Self {
        Self { thresholds: Vec::new(), calibrated_values: Vec::new() }
    }

    /// Fit isotonic regression using Pool Adjacent Violators algorithm.
    pub fn fit(&mut self, scores: &[f64], labels: &[f64]) {
        let n = scores.len();
        let mut paired: Vec<(f64, f64)> = scores.iter().zip(labels.iter()).map(|(&s, &l)| (s, l)).collect();
        paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let sorted_scores: Vec<f64> = paired.iter().map(|(s, _)| *s).collect();
        let sorted_labels: Vec<f64> = paired.iter().map(|(_, l)| *l).collect();

        // Pool Adjacent Violators
        let mut pools: Vec<(f64, usize)> = sorted_labels.iter().enumerate().map(|(i, &l)| (l, 1)).collect();

        loop {
            let mut violated = false;
            for i in 0..pools.len() - 1 {
                if pools[i].0 > pools[i + 1].0 {
                    // Merge pools
                    let total_count = pools[i].1 + pools[i + 1].1;
                    let total_value = pools[i].0 * pools[i].1 as f64 + pools[i + 1].0 * pools[i + 1].1 as f64;
                    pools[i] = (total_value / total_count as f64, total_count);
                    pools[i + 1] = pools[i];
                    violated = true;
                }
            }
            if !violated { break; }
        }

        // Extract results
        self.thresholds = Vec::new();
        self.calibrated_values = Vec::new();
        let mut idx = 0;
        for (value, count) in pools {
            if self.calibrated_values.last() != Some(&value) {
                self.thresholds.push(sorted_scores[idx]);
                self.calibrated_values.push(value);
            }
            idx += count;
        }
    }

    /// Calibrate new scores.
    pub fn calibrate(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&score| {
            // Find appropriate bin
            let bin = self.thresholds.iter().enumerate()
                .rev()
                .find(|(_, &threshold)| score >= threshold)
                .map(|(i, _)| i)
                .unwrap_or(0);

            self.calibrated_values[bin.min(self.calibrated_values.len() - 1)]
        }).collect()
    }
}

/// Reliability diagram data.
pub fn reliability_diagram(probs: &[Vec<f64>], labels: &[usize], n_bins: usize) -> Vec<(f64, f64, usize)> {
    let mut bin_correct = vec![0usize; n_bins];
    let mut bin_confidence = vec![0.0f64; n_bins];
    let mut bin_count = vec![0usize; n_bins];

    for (prob, &label) in probs.iter().zip(labels.iter()) {
        let predicted = prob.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let confidence = prob[predicted];
        let bin = (confidence * n_bins as f64) as usize;
        let bin = bin.min(n_bins - 1);

        bin_count[bin] += 1;
        bin_confidence[bin] += confidence;
        if predicted == label {
            bin_correct[bin] += 1;
        }
    }

    (0..n_bins).map(|i| {
        let avg_confidence = if bin_count[i] > 0 { bin_confidence[i] / bin_count[i] as f64 } else { 0.0 };
        let accuracy = if bin_count[i] > 0 { bin_correct[i] as f64 / bin_count[i] as f64 } else { 0.0 };
        (avg_confidence, accuracy, bin_count[i])
    }).collect()
}

/// Brier score.
pub fn brier_score(probs: &[Vec<f64>], labels: &[usize]) -> f64 {
    let n = probs.len();
    let total: f64 = probs.iter().zip(labels.iter()).map(|(prob, &label)| {
        prob.iter().enumerate().map(|(i, &p)| {
            let target = if i == label { 1.0 } else { 0.0 };
            (p - target).powi(2)
        }).sum::<f64>()
    }).sum();

    total / n as f64
}

/// Negative log-likelihood.
pub fn nll(probs: &[Vec<f64>], labels: &[usize]) -> f64 {
    let n = probs.len();
    let total: f64 = probs.iter().zip(labels.iter()).map(|(prob, &label)| {
        -prob[label].max(1e-15).ln()
    }).sum();

    total / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_scaling() {
        let mut ts = TemperatureScaling::new(0.01);
        let logits = vec![vec![2.0, 1.0, 0.1], vec![0.1, 2.0, 1.0]];
        let labels = vec![0, 1];

        ts.fit(&logits, &labels, 100);
        assert!(ts.temperature > 0.0);

        let calibrated = ts.calibrate(&logits[0]);
        let sum: f64 = calibrated.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_platt_scaling() {
        let mut ps = PlattScaling::new(0.01);
        let scores = vec![1.0, 0.5, -0.5, -1.0];
        let labels = vec![true, true, false, false];
        ps.fit(&scores, &labels, 100);

        let calibrated = ps.calibrate(&scores);
        assert!(calibrated[0] > calibrated[2]);
    }

    #[test]
    fn test_brier_score() {
        let probs = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        let labels = vec![0, 1];
        let bs = brier_score(&probs, &labels);
        assert!(bs < 0.1); // Should be low for well-calibrated
    }
}
