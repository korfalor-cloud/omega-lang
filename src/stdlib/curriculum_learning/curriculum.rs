/// Curriculum learning: data ordering, difficulty estimation, self-paced learning.

/// Curriculum strategy.
pub trait CurriculumStrategy {
    fn select_batch(&self, difficulties: &[f64], epoch: usize, batch_size: usize) -> Vec<usize>;
}

/// Linear curriculum: gradually include harder examples.
pub struct LinearCurriculum {
    pub total_epochs: usize,
}

impl LinearCurriculum {
    pub fn new(total_epochs: usize) -> Self {
        Self { total_epochs }
    }
}

impl CurriculumStrategy for LinearCurriculum {
    fn select_batch(&self, difficulties: &[f64], epoch: usize, batch_size: usize) -> Vec<usize> {
        let progress = epoch as f64 / self.total_epochs as f64;
        let max_difficulty = progress; // Linearly increase difficulty threshold

        let mut indexed: Vec<(usize, f64)> = difficulties.iter().enumerate()
            .map(|(i, &d)| (i, d))
            .filter(|(_, d)| *d <= max_difficulty)
            .collect();

        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        indexed.iter().take(batch_size).map(|(i, _)| *i).collect()
    }
}

/// Exponential curriculum.
pub struct ExponentialCurriculum {
    pub total_epochs: usize,
    pub rate: f64,
}

impl ExponentialCurriculum {
    pub fn new(total_epochs: usize, rate: f64) -> Self {
        Self { total_epochs, rate }
    }
}

impl CurriculumStrategy for ExponentialCurriculum {
    fn select_batch(&self, difficulties: &[f64], epoch: usize, batch_size: usize) -> Vec<usize> {
        let progress = epoch as f64 / self.total_epochs as f64;
        let max_difficulty = 1.0 - (-self.rate * progress).exp();

        let mut indexed: Vec<(usize, f64)> = difficulties.iter().enumerate()
            .map(|(i, &d)| (i, d))
            .filter(|(_, d)| *d <= max_difficulty)
            .collect();

        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        indexed.iter().take(batch_size).map(|(i, _)| *i).collect()
    }
}

/// Self-paced learning.
pub struct SelfPacedLearning {
    pub lambda: f64, // Regularization parameter
}

impl SelfPacedLearning {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Compute sample weights based on loss values.
    pub fn compute_weights(&self, losses: &[f64]) -> Vec<f64> {
        losses.iter().map(|&loss| {
            if loss <= self.lambda {
                1.0 - loss / self.lambda
            } else {
                0.0
            }
        }).collect()
    }

    /// Select samples with non-zero weights.
    pub fn select_samples(&self, losses: &[f64]) -> Vec<usize> {
        let weights = self.compute_weights(losses);
        weights.iter().enumerate()
            .filter(|(_, &w)| w > 0.0)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Teacher-guided curriculum.
pub struct TeacherCurriculum {
    pub difficulty_scores: Vec<f64>,
    pub paces: Vec<f64>, // Learning pace for each difficulty level
}

impl TeacherCurriculum {
    pub fn new(difficulty_scores: Vec<f64>) -> Self {
        let n = difficulty_scores.len();
        Self {
            difficulty_scores,
            paces: vec![1.0; n],
        }
    }

    /// Update pace based on performance.
    pub fn update_pace(&mut self, sample_idx: usize, correct: bool) {
        if correct {
            self.paces[sample_idx] *= 1.1;
        } else {
            self.paces[sample_idx] *= 0.9;
        }
    }

    /// Select batch based on pace and difficulty.
    pub fn select_batch(&self, epoch: usize, batch_size: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f64)> = self.difficulty_scores.iter().enumerate()
            .map(|(i, &d)| {
                let epoch_factor = epoch as f64;
                let score = d * self.paces[i] + epoch_factor * 0.01;
                (i, score)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        scored.iter().take(batch_size).map(|(i, _)| *i).collect()
    }
}

/// Anti-curriculum: start with harder examples.
pub struct AntiCurriculum {
    pub total_epochs: usize,
}

impl AntiCurriculum {
    pub fn new(total_epochs: usize) -> Self {
        Self { total_epochs }
    }

    pub fn select_batch(&self, difficulties: &[f64], epoch: usize, batch_size: usize) -> Vec<usize> {
        let progress = epoch as f64 / self.total_epochs as f64;
        let min_difficulty = 1.0 - progress; // Start with hard, end with easy

        let mut indexed: Vec<(usize, f64)> = difficulties.iter().enumerate()
            .map(|(i, &d)| (i, d))
            .filter(|(_, d)| *d >= min_difficulty)
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Descending

        indexed.iter().take(batch_size).map(|(i, _)| *i).collect()
    }
}

/// Rooted curriculum: p(t) = min(1, sqrt(t/T * (1 - p0^2) + p0^2)).
pub struct RootedCurriculum {
    pub total_epochs: usize,
    pub p0: f64,
}

impl RootedCurriculum {
    pub fn new(total_epochs: usize, p0: f64) -> Self {
        Self { total_epochs, p0 }
    }

    pub fn difficulty_threshold(&self, epoch: usize) -> f64 {
        let t = epoch as f64 / self.total_epochs as f64;
        (t * (1.0 - self.p0 * self.p0) + self.p0 * self.p0).sqrt().min(1.0)
    }
}

impl CurriculumStrategy for RootedCurriculum {
    fn select_batch(&self, difficulties: &[f64], epoch: usize, batch_size: usize) -> Vec<usize> {
        let threshold = self.difficulty_threshold(epoch);

        let mut indexed: Vec<(usize, f64)> = difficulties.iter().enumerate()
            .map(|(i, &d)| (i, d))
            .filter(|(_, d)| *d <= threshold)
            .collect();

        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        indexed.iter().take(batch_size).map(|(i, _)| *i).collect()
    }
}

/// Transfer curriculum: order tasks by similarity to target.
pub struct TransferCurriculum {
    pub source_difficulties: Vec<f64>,
    pub target_similarity: Vec<f64>,
}

impl TransferCurriculum {
    pub fn new(source_difficulties: Vec<f64>, target_similarity: Vec<f64>) -> Self {
        Self { source_difficulties, target_similarity }
    }

    pub fn select_batch(&self, epoch: usize, batch_size: usize) -> Vec<usize> {
        let n = self.source_difficulties.len();
        let progress = epoch as f64 / 100.0; // Assume 100 epochs

        let mut scored: Vec<(usize, f64)> = (0..n).map(|i| {
            // Combine difficulty and similarity
            let score = self.source_difficulties[i] * (1.0 - progress)
                + (1.0 - self.target_similarity[i]) * progress;
            (i, score)
        }).collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        scored.iter().take(batch_size).map(|(i, _)| *i).collect()
    }
}

/// Spaced repetition for curriculum.
pub struct SpacedRepetition {
    pub easiness: Vec<f64>,
    pub interval: Vec<f64>,
    pub repetitions: Vec<usize>,
}

impl SpacedRepetition {
    pub fn new(n_items: usize) -> Self {
        Self {
            easiness: vec![2.5; n_items],
            interval: vec![1.0; n_items],
            repetitions: vec![0; n_items],
        }
    }

    /// Update after reviewing item with quality (0-5).
    pub fn update(&mut self, item: usize, quality: usize) {
        let q = quality as f64;

        if q >= 3.0 {
            if self.repetitions[item] == 0 {
                self.interval[item] = 1.0;
            } else if self.repetitions[item] == 1 {
                self.interval[item] = 6.0;
            } else {
                self.interval[item] *= self.easiness[item];
            }
            self.repetitions[item] += 1;
        } else {
            self.repetitions[item] = 0;
            self.interval[item] = 1.0;
        }

        self.easiness[item] = (self.easiness[item] + 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02)).max(1.3);
    }

    /// Get items due for review.
    pub fn get_due_items(&self, current_day: f64) -> Vec<usize> {
        self.interval.iter().enumerate()
            .filter(|(_, &interval)| interval <= current_day)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get next review day for item.
    pub fn next_review(&self, item: usize) -> f64 {
        self.interval[item]
    }
}

/// Difficulty estimation using loss-based method.
pub fn estimate_difficulty_losses(losses: &[f64]) -> Vec<f64> {
    let max_loss = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_loss == 0.0 { return vec![0.0; losses.len()]; }
    losses.iter().map(|l| l / max_loss).collect()
}

/// Difficulty estimation using gradient norm.
pub fn estimate_difficulty_gradients(gradients: &[Vec<f64>]) -> Vec<f64> {
    gradients.iter().map(|g| {
        g.iter().map(|gi| gi * gi).sum::<f64>().sqrt()
    }).collect()
}

/// Uncertainty-based difficulty estimation.
pub fn estimate_difficulty_uncertainty(predictions: &[Vec<f64>]) -> Vec<f64> {
    predictions.iter().map(|preds| {
        let mean = preds.iter().sum::<f64>() / preds.len() as f64;
        preds.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / preds.len() as f64
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_curriculum() {
        let curriculum = LinearCurriculum::new(100);
        let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9];

        // Early epoch: only easy examples
        let batch = curriculum.select_batch(&difficulties, 10, 3);
        assert!(batch.iter().all(|&i| difficulties[i] <= 0.1));

        // Late epoch: include harder examples
        let batch = curriculum.select_batch(&difficulties, 90, 5);
        assert!(batch.len() == 5);
    }

    #[test]
    fn test_self_paced() {
        let spl = SelfPacedLearning::new(0.5);
        let losses = vec![0.1, 0.3, 0.6, 0.8, 1.0];
        let weights = spl.compute_weights(&losses);
        assert!(weights[0] > 0.0);
        assert!(weights[1] > 0.0);
        assert_eq!(weights[2], 0.0); // loss > lambda
    }

    #[test]
    fn test_spaced_repetition() {
        let mut sr = SpacedRepetition::new(10);
        sr.update(0, 5); // Perfect
        assert!(sr.easiness[0] > 2.5);
        assert_eq!(sr.interval[0], 1.0);
    }
}
