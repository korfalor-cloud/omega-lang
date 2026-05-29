/// Sample selection: influence functions, data Shapley, memorization.

/// Influence function: estimate effect of removing a training point.
pub struct InfluenceFunction {
    pub params: Vec<f64>,
    pub damping: f64,
}

impl InfluenceFunction {
    pub fn new(param_dim: usize, damping: f64) -> Self {
        Self {
            params: vec![0.0; param_dim],
            damping,
        }
    }

    /// Compute inverse Hessian-vector product (conjugate gradient).
    pub fn ihvp(&self, grad_fn: &dyn Fn(&[f64]) -> Vec<f64>, v: &[f64], n_iter: usize) -> Vec<f64> {
        let n = self.params.len();
        let mut x = vec![0.0; n];
        let mut r = v.to_vec();
        let mut p = r.clone();
        let mut rsold: f64 = r.iter().map(|ri| ri * ri).sum();

        for _ in 0..n_iter {
            // Hessian-vector product (finite difference approximation)
            let hp = self.hessian_vector_product(grad_fn, &p);

            let alpha = rsold / (p.iter().zip(hp.iter()).map(|(pi, hi)| pi * hi).sum::<f64>() + self.damping);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * hp[i];
            }

            let rsnew: f64 = r.iter().map(|ri| ri * ri).sum();
            if rsnew < 1e-10 { break; }

            let beta = rsnew / rsold;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rsold = rsnew;
        }

        x
    }

    fn hessian_vector_product(&self, grad_fn: &dyn Fn(&[f64]) -> Vec<f64>, v: &[f64]) -> Vec<f64> {
        let eps = 1e-5;
        let n = self.params.len();

        let grad_plus: Vec<f64> = (0..n).map(|i| {
            let mut p = self.params.clone();
            p[i] += eps * v[i];
            grad_fn(&p)[i]
        }).collect();

        let grad_minus: Vec<f64> = (0..n).map(|i| {
            let mut p = self.params.clone();
            p[i] -= eps * v[i];
            grad_fn(&p)[i]
        }).collect();

        (0..n).map(|i| (grad_plus[i] - grad_minus[i]) / (2.0 * eps)).collect()
    }

    /// Compute influence of training point on test loss.
    pub fn influence(&self, train_grad: &[f64], test_grad: &[f64], n_iter: usize) -> f64 {
        let ihvp = self.ihvp(&|_| train_grad.to_vec(), test_grad, n_iter);
        -train_grad.iter().zip(ihvp.iter()).map(|(a, b)| a * b).sum::<f64>()
    }
}

/// Data Shapley: approximate Shapley values for data points.
pub struct DataShapley {
    pub n_samples: usize,
    pub n_permutations: usize,
    seed: u64,
}

impl DataShapley {
    pub fn new(n_samples: usize, n_permutations: usize) -> Self {
        Self { n_samples, n_permutations, seed: 42 }
    }

    /// Compute Shapley values using Monte Carlo sampling.
    pub fn compute<F>(&mut self, performance_fn: F) -> Vec<f64>
    where
        F: Fn(&[usize]) -> f64,
    {
        let mut shapley = vec![0.0; self.n_samples];

        for _ in 0..self.n_permutations {
            // Random permutation
            let mut perm: Vec<usize> = (0..self.n_samples).collect();
            for i in (1..self.n_samples).rev() {
                self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = ((self.seed >> 33) as usize) % (i + 1);
                perm.swap(i, j);
            }

            // Compute marginal contributions
            let mut coalition = Vec::new();
            let mut prev_perf = performance_fn(&coalition);

            for &idx in &perm {
                coalition.push(idx);
                let curr_perf = performance_fn(&coalition);
                shapley[idx] += curr_perf - prev_perf;
                prev_perf = curr_perf;
            }
        }

        // Average
        for s in shapley.iter_mut() {
            *s /= self.n_permutations as f64;
        }

        shapley
    }
}

/// Leave-one-out (LOO) importance.
pub fn loo_importance<F>(n_samples: usize, performance_fn: F) -> Vec<f64>
where
    F: Fn(&[usize]) -> f64,
{
    let all: Vec<usize> = (0..n_samples).collect();
    let full_perf = performance_fn(&all);

    (0..n_samples).map(|i| {
        let subset: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
        let loo_perf = performance_fn(&subset);
        full_perf - loo_perf
    }).collect()
}

/// Memorization score: difference between training and validation performance.
pub fn memorization_score(train_correct: &[bool], val_correct: &[bool]) -> Vec<f64> {
    train_correct.iter().zip(val_correct.iter())
        .map(|(&train, &val)| {
            if train && !val { 1.0 }       // Memorized
            else if train && val { 0.0 }   // Generalized
            else if !train { -1.0 }        // Not learned
            else { 0.0 }
        })
        .collect()
}

/// Data valuation using KNN.
pub fn knn_valuation(features: &[Vec<f64>], labels: &[usize], k: usize) -> Vec<f64> {
    let n = features.len();

    (0..n).map(|i| {
        let mut distances: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let dist: f64 = features[i].iter().zip(features[j].iter())
                    .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                (j, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Count correct predictions among k nearest neighbors
        let correct = distances.iter().take(k)
            .filter(|&&(idx, _)| labels[idx] == labels[i])
            .count();

        correct as f64 / k as f64
    }).collect()
}

/// TracIn: trace the training trajectory influence.
pub struct TracIn {
    pub checkpoints: Vec<Vec<f64>>,
    pub learning_rates: Vec<f64>,
}

impl TracIn {
    pub fn new(checkpoints: Vec<Vec<f64>>, learning_rates: Vec<f64>) -> Self {
        Self { checkpoints, learning_rates }
    }

    /// Compute influence of training point on test point.
    pub fn influence(&self, train_grad_fn: &dyn Fn(&[f64], &[f64], f64) -> Vec<f64>,
                     test_grad_fn: &dyn Fn(&[f64], &[f64], f64) -> Vec<f64>,
                     train_x: &[f64], train_y: f64,
                     test_x: &[f64], test_y: f64) -> f64 {
        let mut total = 0.0;

        for (checkpoint, &lr) in self.checkpoints.iter().zip(self.learning_rates.iter()) {
            let train_grad = train_grad_fn(checkpoint, train_x, train_y);
            let test_grad = test_grad_fn(checkpoint, test_x, test_y);

            total += lr * train_grad.iter().zip(test_grad.iter()).map(|(a, b)| a * b).sum::<f64>();
        }

        total
    }
}

/// Data pruning: remove low-value training examples.
pub fn prune_data(importance_scores: &[f64], keep_ratio: f64) -> Vec<usize> {
    let n = importance_scores.len();
    let n_keep = (n as f64 * keep_ratio) as usize;

    let mut indexed: Vec<(usize, f64)> = importance_scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Descending

    indexed.iter().take(n_keep).map(|(i, _)| *i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_shapley() {
        let mut shapley = DataShapley::new(3, 100);
        let perf_fn = |indices: &[usize]| -> f64 {
            indices.len() as f64 * 0.3
        };
        let values = shapley.compute(perf_fn);
        assert_eq!(values.len(), 3);
        // Each point should have roughly equal contribution
        for v in &values {
            assert!((v - 0.3).abs() < 0.2);
        }
    }

    #[test]
    fn test_loo() {
        let n = 5;
        let perf_fn = |indices: &[usize]| -> f64 {
            indices.len() as f64
        };
        let importance = loo_importance(n, perf_fn);
        assert_eq!(importance.len(), n);
        // Removing any point should reduce performance by 1
        for imp in &importance {
            assert!((imp - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_prune() {
        let scores = vec![0.9, 0.1, 0.8, 0.2, 0.7];
        let kept = prune_data(&scores, 0.6);
        assert_eq!(kept.len(), 3);
        assert!(kept.contains(&0)); // Highest score
        assert!(kept.contains(&2));
        assert!(kept.contains(&4));
    }
}
