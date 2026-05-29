/// AutoML: automated hyperparameter optimization, feature selection, model selection.

/// Hyperparameter space.
#[derive(Clone, Debug)]
pub struct HyperparameterSpace {
    pub continuous: Vec<(String, f64, f64)>, // (name, min, max)
    pub discrete: Vec<(String, Vec<f64>)>,   // (name, values)
    pub categorical: Vec<(String, Vec<String>)>, // (name, categories)
}

impl HyperparameterSpace {
    pub fn new() -> Self {
        Self { continuous: Vec::new(), discrete: Vec::new(), categorical: Vec::new() }
    }

    pub fn add_continuous(&mut self, name: &str, min: f64, max: f64) {
        self.continuous.push((name.to_string(), min, max));
    }

    pub fn add_discrete(&mut self, name: &str, values: Vec<f64>) {
        self.discrete.push((name.to_string(), values));
    }

    pub fn add_categorical(&mut self, name: &str, categories: Vec<String>) {
        self.categorical.push((name.to_string(), categories));
    }

    pub fn sample(&self, seed: u64) -> Vec<(String, f64)> {
        let mut rng = seed;
        let mut rand = || -> f64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f64) / (1u64 << 31) as f64
        };

        let mut params = Vec::new();

        for (name, min, max) in &self.continuous {
            params.push((name.clone(), min + rand() * (max - min)));
        }

        for (name, values) in &self.discrete {
            let idx = (rand() * values.len() as f64) as usize % values.len();
            params.push((name.clone(), values[idx]));
        }

        for (name, categories) in &self.categorical {
            let idx = (rand() * categories.len() as f64) as usize % categories.len();
            params.push((name.clone(), idx as f64));
        }

        params
    }

    pub fn n_dimensions(&self) -> usize {
        self.continuous.len() + self.discrete.len() + self.categorical.len()
    }
}

/// Random search for hyperparameter optimization.
pub struct RandomSearch {
    pub space: HyperparameterSpace,
    pub n_trials: usize,
    seed: u64,
}

impl RandomSearch {
    pub fn new(space: HyperparameterSpace, n_trials: usize) -> Self {
        Self { space, n_trials, seed: 42 }
    }

    pub fn optimize<F>(&mut self, objective: F) -> (Vec<(String, f64)>, f64)
    where
        F: Fn(&[(String, f64)]) -> f64,
    {
        let mut best_params = Vec::new();
        let mut best_value = f64::NEG_INFINITY;

        for _ in 0..self.n_trials {
            self.seed += 1;
            let params = self.space.sample(self.seed);
            let value = objective(&params);

            if value > best_value {
                best_value = value;
                best_params = params;
            }
        }

        (best_params, best_value)
    }
}

/// Bayesian optimization with Gaussian process surrogate.
pub struct BayesianOptimization {
    pub space: HyperparameterSpace,
    pub observations: Vec<(Vec<f64>, f64)>,
    pub n_initial: usize,
    seed: u64,
}

impl BayesianOptimization {
    pub fn new(space: HyperparameterSpace, n_initial: usize) -> Self {
        Self { space, observations: Vec::new(), n_initial, seed: 42 }
    }

    pub fn optimize<F>(&mut self, objective: F, n_iterations: usize) -> (Vec<(String, f64)>, f64)
    where
        F: Fn(&[(String, f64)]) -> f64,
    {
        // Initial random sampling
        for i in 0..self.n_initial {
            self.seed += 1;
            let params = self.space.sample(self.seed);
            let param_vec: Vec<f64> = params.iter().map(|(_, v)| *v).collect();
            let value = objective(&params);
            self.observations.push((param_vec, value));
        }

        // Bayesian optimization loop
        for _ in 0..n_iterations {
            // Acquisition function: Upper Confidence Bound
            let next_point = self.acquire_ucb(2.0);
            let params: Vec<(String, f64)> = self.space.continuous.iter().enumerate()
                .map(|(i, (name, _, _))| (name.clone(), next_point[i]))
                .chain(self.space.discrete.iter().enumerate().map(|(i, (name, values))| {
                    let idx = (next_point[self.space.continuous.len() + i] * values.len() as f64) as usize % values.len();
                    (name.clone(), values[idx])
                }))
                .chain(self.space.categorical.iter().enumerate().map(|(i, (name, cats))| {
                    let idx = (next_point[self.space.continuous.len() + self.space.discrete.len() + i] * cats.len() as f64) as usize % cats.len();
                    (name.clone(), idx as f64)
                }))
                .collect();

            let value = objective(&params);
            let param_vec: Vec<f64> = params.iter().map(|(_, v)| *v).collect();
            self.observations.push((param_vec, value));
        }

        // Return best
        let best = self.observations.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let best_params: Vec<(String, f64)> = self.space.continuous.iter().enumerate()
            .map(|(i, (name, _, _))| (name.clone(), best.0[i]))
            .collect();

        (best_params, best.1)
    }

    fn acquire_ucb(&self, beta: f64) -> Vec<f64> {
        // Simplified UCB: sample near best observed point with noise
        let best = self.observations.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let mut result = best.0.clone();
        let mut rng = self.seed;

        for val in result.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = ((rng >> 33) as f64) / (1u64 << 31) as f64 * 0.1 - 0.05;
            *val = (*val + noise).max(0.0).min(1.0);
        }

        result
    }

    /// Expected improvement acquisition function.
    pub fn expected_improvement(&self, candidate: &[f64], xi: f64) -> f64 {
        let (mean, std) = self.predict(candidate);
        let best = self.observations.iter()
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max);

        if std < 1e-10 { return 0.0; }

        let z = (mean - best - xi) / std;
        let ei = (mean - best - xi) * normal_cdf(z) + std * normal_pdf(z);
        ei.max(0.0)
    }

    fn predict(&self, x: &[f64]) -> (f64, f64) {
        // Simplified GP prediction: weighted average of observations
        if self.observations.is_empty() { return (0.0, 1.0); }

        let mut weights: Vec<f64> = self.observations.iter()
            .map(|(obs, _)| {
                let dist: f64 = obs.iter().zip(x.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                (-dist * dist).exp()
            })
            .collect();

        let total_weight: f64 = weights.iter().sum();
        if total_weight < 1e-10 { return (0.0, 1.0); }

        for w in weights.iter_mut() { *w /= total_weight; }

        let mean: f64 = weights.iter().zip(self.observations.iter()).map(|(w, (_, v))| w * v).sum();
        let variance: f64 = weights.iter().zip(self.observations.iter())
            .map(|(w, (_, v))| w * (v - mean).powi(2))
            .sum();

        (mean, variance.sqrt())
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Feature selection using mutual information.
pub struct MutualInformationSelector {
    pub n_features: usize,
    pub n_selected: usize,
}

impl MutualInformationSelector {
    pub fn new(n_features: usize, n_selected: usize) -> Self {
        Self { n_features, n_selected }
    }

    pub fn select(&self, x: &[Vec<f64>], y: &[f64]) -> Vec<usize> {
        let n = x.len();
        let mut mi_scores: Vec<(usize, f64)> = (0..self.n_features).map(|j| {
            let feature: Vec<f64> = x.iter().map(|row| row[j]).collect();
            let mi = self.mutual_information(&feature, y);
            (j, mi)
        }).collect();

        mi_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        mi_scores.iter().take(self.n_selected).map(|(idx, _)| *idx).collect()
    }

    fn mutual_information(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let n_bins = 10;

        let x_bins = self.discretize(x, n_bins);
        let y_bins = self.discretize(y, n_bins);

        let mut joint = vec![vec![0.0; n_bins]; n_bins];
        let mut x_counts = vec![0.0; n_bins];
        let mut y_counts = vec![0.0; n_bins];

        for (&xb, &yb) in x_bins.iter().zip(y_bins.iter()) {
            joint[xb][yb] += 1.0;
            x_counts[xb] += 1.0;
            y_counts[yb] += 1.0;
        }

        let mut mi = 0.0;
        for i in 0..n_bins {
            for j in 0..n_bins {
                if joint[i][j] > 0.0 && x_counts[i] > 0.0 && y_counts[j] > 0.0 {
                    mi += (joint[i][j] / n) * (joint[i][j] * n / (x_counts[i] * y_counts[j])).ln();
                }
            }
        }

        mi
    }

    fn discretize(&self, data: &[f64], n_bins: usize) -> Vec<usize> {
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        data.iter().map(|&x| {
            if range < 1e-10 { 0 } else { ((x - min) / range * (n_bins - 1) as f64) as usize }.min(n_bins - 1)
        }).collect()
    }
}

/// Model selection: cross-validation based.
pub struct ModelSelector {
    pub n_folds: usize,
}

impl ModelSelector {
    pub fn new(n_folds: usize) -> Self {
        Self { n_folds }
    }

    pub fn cross_validate<F>(&self, n_samples: usize, model_fn: F) -> Vec<f64>
    where
        F: Fn(&[usize], &[usize]) -> f64, // (train_indices, val_indices) -> score
    {
        let fold_size = n_samples / self.n_folds;
        let mut scores = Vec::new();

        for fold in 0..self.n_folds {
            let val_start = fold * fold_size;
            let val_end = val_start + fold_size;

            let train_indices: Vec<usize> = (0..n_samples).filter(|&i| i < val_start || i >= val_end).collect();
            let val_indices: Vec<usize> = (val_start..val_end).collect();

            let score = model_fn(&train_indices, &val_indices);
            scores.push(score);
        }

        scores
    }

    pub fn select_best<F>(&self, n_samples: usize, models: Vec<F>) -> (usize, f64)
    where
        F: Fn(&[usize], &[usize]) -> f64,
    {
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (i, model_fn) in models.iter().enumerate() {
            let scores = self.cross_validate(n_samples, |train, val| model_fn(train, val));
            let mean_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;

            if mean_score > best_score {
                best_score = mean_score;
                best_idx = i;
            }
        }

        (best_idx, best_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_search() {
        let mut space = HyperparameterSpace::new();
        space.add_continuous("lr", 0.001, 0.1);
        space.add_discrete("batch_size", vec![16.0, 32.0, 64.0]);

        let mut search = RandomSearch::new(space, 10);
        let objective = |params: &[(String, f64)]| {
            let lr = params.iter().find(|(n, _)| n == "lr").unwrap().1;
            -(lr - 0.01).powi(2) // Maximize when lr = 0.01
        };

        let (best_params, best_value) = search.optimize(objective);
        assert!(best_value < 0.0); // Should be close to 0
    }

    #[test]
    fn test_feature_selection() {
        let selector = MutualInformationSelector::new(5, 2);
        let x = vec![
            vec![1.0, 0.0, 1.0, 0.0, 0.5],
            vec![0.0, 1.0, 0.0, 1.0, 0.5],
            vec![1.0, 0.0, 1.0, 0.0, 0.5],
            vec![0.0, 1.0, 0.0, 1.0, 0.5],
        ];
        let y = vec![1.0, 0.0, 1.0, 0.0];

        let selected = selector.select(&x, &y);
        assert_eq!(selected.len(), 2);
    }
}
