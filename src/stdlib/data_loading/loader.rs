/// Data loading: batching, shuffling, sampling, and data pipelines.

/// Dataset trait.
pub trait Dataset {
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Option<(Vec<f64>, f64)>;
}

/// In-memory dataset.
pub struct InMemoryDataset {
    pub data: Vec<(Vec<f64>, f64)>,
}

impl InMemoryDataset {
    pub fn new(data: Vec<(Vec<f64>, f64)>) -> Self {
        Self { data }
    }
}

impl Dataset for InMemoryDataset {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, idx: usize) -> Option<(Vec<f64>, f64)> {
        self.data.get(idx).cloned()
    }
}

/// DataLoader with batching and shuffling.
pub struct DataLoader {
    pub indices: Vec<usize>,
    pub batch_size: usize,
    pub current: usize,
    pub shuffle: bool,
    seed: u64,
}

impl DataLoader {
    pub fn new(n_samples: usize, batch_size: usize, shuffle: bool) -> Self {
        let indices: Vec<usize> = (0..n_samples).collect();
        Self { indices, batch_size, current: 0, shuffle, seed: 42 }
    }

    pub fn reset(&mut self) {
        self.current = 0;
        if self.shuffle {
            self.shuffle();
        }
    }

    fn shuffle(&mut self) {
        let n = self.indices.len();
        for i in (1..n).rev() {
            self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = ((self.seed >> 33) as usize) % (i + 1);
            self.indices.swap(i, j);
        }
    }

    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.current >= self.indices.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch = self.indices[self.current..end].to_vec();
        self.current = end;
        Some(batch)
    }

    pub fn n_batches(&self) -> usize {
        (self.indices.len() + self.batch_size - 1) / self.batch_size
    }
}

/// Weighted random sampler.
pub struct WeightedSampler {
    pub weights: Vec<f64>,
    pub cumulative: Vec<f64>,
    seed: u64,
}

impl WeightedSampler {
    pub fn new(weights: Vec<f64>) -> Self {
        let total: f64 = weights.iter().sum();
        let mut cumulative = Vec::new();
        let mut sum = 0.0;
        for &w in &weights {
            sum += w / total;
            cumulative.push(sum);
        }
        Self { weights, cumulative, seed: 42 }
    }

    pub fn sample(&mut self, n: usize) -> Vec<usize> {
        let mut result = Vec::new();
        for _ in 0..n {
            let r = self.pseudo_rand();
            let idx = self.cumulative.iter().position(|&c| c >= r).unwrap_or(self.cumulative.len() - 1);
            result.push(idx);
        }
        result
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Stratified sampler.
pub struct StratifiedSampler {
    pub class_indices: Vec<Vec<usize>>,
    pub n_samples_per_class: usize,
    seed: u64,
}

impl StratifiedSampler {
    pub fn new(labels: &[usize], n_classes: usize, n_samples_per_class: usize) -> Self {
        let mut class_indices = vec![Vec::new(); n_classes];
        for (i, &label) in labels.iter().enumerate() {
            class_indices[label].push(i);
        }
        Self { class_indices, n_samples_per_class, seed: 42 }
    }

    pub fn sample(&mut self) -> Vec<usize> {
        let mut result = Vec::new();
        for class_idx in 0..self.class_indices.len() {
            let n = self.class_indices[class_idx].len();
            for _ in 0..self.n_samples_per_class.min(n) {
                let idx = ((self.seed >> 33) as usize) % n;
                self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                result.push(self.class_indices[class_idx][idx]);
            }
        }
        result
    }
}

/// Data pipeline with transforms.
pub struct DataPipeline {
    pub transforms: Vec<Box<dyn Transform>>,
}

pub trait Transform {
    fn apply(&self, data: &[f64]) -> Vec<f64>;
}

pub struct Normalize {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl Transform for Normalize {
    fn apply(&self, data: &[f64]) -> Vec<f64> {
        data.iter().zip(self.mean.iter()).zip(self.std.iter())
            .map(|((&d, &m), &s)| (d - m) / s.max(1e-10))
            .collect()
    }
}

pub struct StandardScaler {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl StandardScaler {
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        let n = data.len();
        let d = data[0].len();

        self.mean = (0..d).map(|j| data.iter().map(|x| x[j]).sum::<f64>() / n as f64).collect();
        self.std = (0..d).map(|j| {
            let variance = data.iter().map(|x| (x[j] - self.mean[j]).powi(2)).sum::<f64>() / n as f64;
            variance.sqrt().max(1e-10)
        }).collect();
    }

    pub fn transform(&self, data: &[f64]) -> Vec<f64> {
        data.iter().zip(self.mean.iter()).zip(self.std.iter())
            .map(|((&d, &m), &s)| (d - m) / s)
            .collect()
    }
}

/// Min-max scaler.
pub struct MinMaxScaler {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub feature_range: (f64, f64),
}

impl MinMaxScaler {
    pub fn new(feature_range: (f64, f64)) -> Self {
        Self { min: Vec::new(), max: Vec::new(), feature_range }
    }

    pub fn fit(&mut self, data: &[Vec<f64>]) {
        let d = data[0].len();
        self.min = (0..d).map(|j| data.iter().map(|x| x[j]).fold(f64::INFINITY, f64::min)).collect();
        self.max = (0..d).map(|j| data.iter().map(|x| x[j]).fold(f64::NEG_INFINITY, f64::max)).collect();
    }

    pub fn transform(&self, data: &[f64]) -> Vec<f64> {
        data.iter().zip(self.min.iter()).zip(self.max.iter())
            .map(|((&d, &min), &max)| {
                let range = max - min;
                if range < 1e-10 { self.feature_range.0 } else {
                    self.feature_range.0 + (d - min) / range * (self.feature_range.1 - self.feature_range.0)
                }
            })
            .collect()
    }
}

/// Collation functions.
pub fn collate_batch(batch: &[(Vec<f64>, f64)]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x: Vec<Vec<f64>> = batch.iter().map(|(x, _)| x.clone()).collect();
    let y: Vec<f64> = batch.iter().map(|(_, y)| *y).collect();
    (x, y)
}

pub fn pad_sequence(sequences: &[Vec<f64>], max_len: usize, pad_value: f64) -> Vec<Vec<f64>> {
    sequences.iter().map(|seq| {
        let mut padded = seq.clone();
        padded.resize(max_len, pad_value);
        padded
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader() {
        let mut loader = DataLoader::new(10, 3, true);
        loader.reset();

        let batch1 = loader.next_batch().unwrap();
        assert_eq!(batch1.len(), 3);

        let batch2 = loader.next_batch().unwrap();
        assert_eq!(batch2.len(), 3);
    }

    #[test]
    fn test_weighted_sampler() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let mut sampler = WeightedSampler::new(weights);
        let samples = sampler.sample(100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_standard_scaler() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut scaler = StandardScaler { mean: Vec::new(), std: Vec::new() };
        scaler.fit(&data);
        let transformed = scaler.transform(&[3.0, 4.0]);
        assert!(transformed[0].abs() < 0.1);
    }
}
