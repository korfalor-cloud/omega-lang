/// Active learning: uncertainty sampling, query-by-committee, expected model change.

/// Uncertainty sampling strategies.
pub struct UncertaintySampling {
    pub strategy: SamplingStrategy,
}

#[derive(Clone, Debug)]
pub enum SamplingStrategy {
    Margin,
    Entropy,
    LeastConfident,
}

impl UncertaintySampling {
    pub fn new(strategy: SamplingStrategy) -> Self {
        Self { strategy }
    }

    /// Select most uncertain sample.
    pub fn select(&self, unlabeled: &[Vec<f64>], model: &dyn Fn(&[f64]) -> Vec<f64>) -> usize {
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (i, x) in unlabeled.iter().enumerate() {
            let probs = model(x);
            let score = match self.strategy {
                SamplingStrategy::Margin => self.margin_score(&probs),
                SamplingStrategy::Entropy => self.entropy_score(&probs),
                SamplingStrategy::LeastConfident => self.least_confident_score(&probs),
            };

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        best_idx
    }

    fn margin_score(&self, probs: &[f64]) -> f64 {
        if probs.len() < 2 { return 0.0; }
        let mut sorted = probs.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        -(sorted[0] - sorted[1]) // Negative because we want to maximize
    }

    fn entropy_score(&self, probs: &[f64]) -> f64 {
        -probs.iter().filter(|&&p| p > 0.0).map(|&p| p * p.ln()).sum::<f64>()
    }

    fn least_confident_score(&self, probs: &[f64]) -> f64 {
        let max_prob = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        -(1.0 - max_prob)
    }

    /// Batch selection.
    pub fn select_batch(&self, unlabeled: &[Vec<f64>], model: &dyn Fn(&[f64]) -> Vec<f64>, batch_size: usize) -> Vec<usize> {
        let mut selected = Vec::new();
        let mut remaining: Vec<usize> = (0..unlabeled.len()).collect();

        for _ in 0..batch_size.min(unlabeled.len()) {
            let mut best_idx = 0;
            let mut best_score = f64::NEG_INFINITY;

            for &idx in &remaining {
                let probs = model(&unlabeled[idx]);
                let score = match self.strategy {
                    SamplingStrategy::Margin => self.margin_score(&probs),
                    SamplingStrategy::Entropy => self.entropy_score(&probs),
                    SamplingStrategy::LeastConfident => self.least_confident_score(&probs),
                };

                if score > best_score {
                    best_score = score;
                    best_idx = idx;
                }
            }

            selected.push(best_idx);
            remaining.retain(|&x| x != best_idx);
        }

        selected
    }
}

/// Query by Committee.
pub struct QueryByCommittee {
    pub n_committee: usize,
    pub disagreement: DisagreementMeasure,
}

#[derive(Clone, Debug)]
pub enum DisagreementMeasure {
    VoteEntropy,
    KLDisagreement,
    ConsensusEntropy,
}

impl QueryByCommittee {
    pub fn new(n_committee: usize, disagreement: DisagreementMeasure) -> Self {
        Self { n_committee, disagreement }
    }

    /// Select sample with highest disagreement.
    pub fn select(&self, unlabeled: &[Vec<f64>], committee: &[&dyn Fn(&[f64]) -> Vec<f64>]) -> usize {
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (i, x) in unlabeled.iter().enumerate() {
            let predictions: Vec<Vec<f64>> = committee.iter().map(|model| model(x)).collect();
            let score = match self.disagreement {
                DisagreementMeasure::VoteEntropy => self.vote_entropy(&predictions),
                DisagreementMeasure::KLDisagreement => self.kl_disagreement(&predictions),
                DisagreementMeasure::ConsensusEntropy => self.consensus_entropy(&predictions),
            };

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        best_idx
    }

    fn vote_entropy(&self, predictions: &[Vec<f64>]) -> f64 {
        let n = predictions.len();
        let n_classes = predictions[0].len();

        // Get majority vote for each class
        let mut votes = vec![0usize; n_classes];
        for pred in predictions {
            let best_class = pred.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            votes[best_class] += 1;
        }

        // Entropy of votes
        let n_f64 = n as f64;
        -votes.iter()
            .filter(|&&v| v > 0)
            .map(|&v| {
                let p = v as f64 / n_f64;
                p * p.ln()
            })
            .sum::<f64>()
    }

    fn kl_disagreement(&self, predictions: &[Vec<f64>]) -> f64 {
        let n = predictions.len();
        let n_classes = predictions[0].len();

        // Average prediction
        let avg: Vec<f64> = (0..n_classes).map(|c| {
            predictions.iter().map(|p| p[c]).sum::<f64>() / n as f64
        }).collect();

        // Average KL divergence from each committee member to average
        predictions.iter().map(|p| {
            p.iter().zip(avg.iter())
                .filter(|(pi, _)| **pi > 0.0)
                .map(|(pi, ai)| pi * (pi / ai.max(1e-15)).ln())
                .sum::<f64>()
        }).sum::<f64>() / n as f64
    }

    fn consensus_entropy(&self, predictions: &[Vec<f64>]) -> f64 {
        let n = predictions.len();
        let n_classes = predictions[0].len();

        let avg: Vec<f64> = (0..n_classes).map(|c| {
            predictions.iter().map(|p| p[c]).sum::<f64>() / n as f64
        }).collect();

        -avg.iter().filter(|&&p| p > 0.0).map(|&p| p * p.ln()).sum::<f64>()
    }
}

/// Expected Model Change.
pub struct ExpectedModelChange {
    pub learning_rate: f64,
}

impl ExpectedModelChange {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    /// Select sample that would cause largest gradient.
    pub fn select(&self, unlabeled: &[Vec<f64>], model: &dyn Fn(&[f64]) -> Vec<f64>,
                  grad_fn: &dyn Fn(&[f64], &[f64]) -> Vec<f64>) -> usize {
        let mut best_idx = 0;
        let mut best_magnitude = 0.0;

        for (i, x) in unlabeled.iter().enumerate() {
            let probs = model(x);
            let predicted_class = probs.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            // Compute gradient assuming predicted class is correct
            let mut pseudo_label = vec![0.0; probs.len()];
            pseudo_label[predicted_class] = 1.0;

            let grad = grad_fn(x, &pseudo_label);
            let magnitude: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

            if magnitude > best_magnitude {
                best_magnitude = magnitude;
                best_idx = i;
            }
        }

        best_idx
    }
}

/// Density-weighted active learning.
pub struct DensityWeightedAL {
    pub beta: f64, // Balance between uncertainty and density
}

impl DensityWeightedAL {
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }

    /// Select based on uncertainty * density^beta.
    pub fn select(&self, unlabeled: &[Vec<f64>], model: &dyn Fn(&[f64]) -> Vec<f64>) -> usize {
        let n = unlabeled.len();

        // Compute density for each point (average similarity to other points)
        let densities: Vec<f64> = unlabeled.iter().map(|x| {
            let sim_sum: f64 = unlabeled.iter().map(|other| {
                let dist: f64 = x.iter().zip(other.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                (-dist).exp()
            }).sum();
            sim_sum / n as f64
        }).collect();

        // Compute uncertainty
        let uncertainties: Vec<f64> = unlabeled.iter().map(|x| {
            let probs = model(x);
            -probs.iter().filter(|&&p| p > 0.0).map(|&p| p * p.ln()).sum::<f64>()
        }).collect();

        // Combined score
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for i in 0..n {
            let score = uncertainties[i] * densities[i].powf(self.beta);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        best_idx
    }
}

/// Coreset active learning.
pub struct CoresetAL;

impl CoresetAL {
    /// Select points that are farthest from existing labeled set (greedy).
    pub fn select_batch(labeled: &[Vec<f64>], unlabeled: &[Vec<f64>], batch_size: usize) -> Vec<usize> {
        let mut selected = Vec::new();
        let mut remaining: Vec<usize> = (0..unlabeled.len()).collect();

        for _ in 0..batch_size.min(unlabeled.len()) {
            let mut best_idx = 0;
            let mut best_min_dist = f64::NEG_INFINITY;

            for &idx in &remaining {
                // Minimum distance to any labeled point
                let min_dist = labeled.iter()
                    .map(|l| {
                        unlabeled[idx].iter().zip(l.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
                    })
                    .fold(f64::INFINITY, f64::min);

                // Also consider already selected points
                let min_dist = selected.iter()
                    .map(|&s| {
                        unlabeled[idx].iter().zip(unlabeled[s].iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
                    })
                    .fold(min_dist, f64::min);

                if min_dist > best_min_dist {
                    best_min_dist = min_dist;
                    best_idx = idx;
                }
            }

            selected.push(best_idx);
            remaining.retain(|&x| x != best_idx);
        }

        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_sampling() {
        let us = UncertaintySampling::new(SamplingStrategy::Entropy);
        let unlabeled = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
        let model = |x: &[f64]| -> Vec<f64> {
            if x[0] > 0.7 { vec![0.9, 0.1] } else { vec![0.5, 0.5] }
        };
        let idx = us.select(&unlabeled, &model);
        assert_eq!(idx, 1); // More uncertain
    }

    #[test]
    fn test_qbc() {
        let qbc = QueryByCommittee::new(3, DisagreementMeasure::VoteEntropy);
        let unlabeled = vec![vec![1.0, 0.0], vec![0.5, 0.5]];
        let committee: Vec<&dyn Fn(&[f64]) -> Vec<f64>> = vec![
            &|_| vec![0.9, 0.1],
            &|_| vec![0.1, 0.9],
            &|_| vec![0.5, 0.5],
        ];
        let idx = qbc.select(&unlabeled, &committee);
        assert!(idx < 2);
    }

    #[test]
    fn test_coreset() {
        let labeled = vec![vec![0.0, 0.0]];
        let unlabeled = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![5.0, 5.0]];
        let selected = CoresetAL::select_batch(&labeled, &unlabeled, 2);
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&2)); // Farthest point
    }
}
