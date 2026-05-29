/// Few-shot learning: prototypical networks, matching networks, MAML, relation networks.

/// Few-shot episode generator.
pub struct EpisodeGenerator {
    pub n_way: usize,
    pub k_shot: usize,
    pub n_query: usize,
    seed: u64,
}

impl EpisodeGenerator {
    pub fn new(n_way: usize, k_shot: usize, n_query: usize) -> Self {
        Self { n_way, k_shot, n_query, seed: 42 }
    }

    /// Generate a few-shot episode from dataset.
    pub fn generate_episode(&mut self, data: &[(Vec<f64>, usize)]) -> (Vec<(Vec<f64>, usize)>, Vec<(Vec<f64>, usize)>) {
        let mut class_data: std::collections::HashMap<usize, Vec<&(Vec<f64>, usize)>> = std::collections::HashMap::new();
        for item in data {
            class_data.entry(item.1).or_default().push(item);
        }

        let classes: Vec<usize> = class_data.keys().take(self.n_way).cloned().collect();
        let mut support = Vec::new();
        let mut query = Vec::new();

        for &class in &classes {
            let items = &class_data[&class];
            let n_samples = items.len();

            // Select k_shot + n_query samples
            let mut indices: Vec<usize> = (0..n_samples).collect();
            // Shuffle
            for i in (1..n_samples).rev() {
                self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = ((self.seed >> 33) as usize) % (i + 1);
                indices.swap(i, j);
            }

            for &idx in indices.iter().take(self.k_shot) {
                support.push((items[idx].0.clone(), class));
            }
            for &idx in indices.iter().skip(self.k_shot).take(self.n_query) {
                query.push((items[idx].0.clone(), class));
            }
        }

        (support, query)
    }
}

/// Relation Network for few-shot learning.
pub struct RelationNetwork {
    pub embedding_dim: usize,
    pub relation_dim: usize,
    pub encoder_weights: Vec<Vec<f64>>,
    pub relation_weights: Vec<Vec<f64>>,
    pub learning_rate: f64,
}

impl RelationNetwork {
    pub fn new(input_dim: usize, embedding_dim: usize, relation_dim: usize, learning_rate: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let enc_scale = (2.0 / input_dim as f64).sqrt();
        let rel_scale = (2.0 / (embedding_dim * 2) as f64).sqrt();

        Self {
            embedding_dim, relation_dim, learning_rate,
            encoder_weights: (0..embedding_dim).map(|_| (0..input_dim).map(|_| rand(enc_scale)).collect()).collect(),
            relation_weights: (0..1).map(|_| (0..embedding_dim * 2).map(|_| rand(rel_scale)).collect()).collect(),
        }
    }

    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        self.encoder_weights.iter().map(|w| {
            w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
        }).collect()
    }

    pub fn relation_score(&self, support_embed: &[f64], query_embed: &[f64]) -> f64 {
        let mut combined = support_embed.to_vec();
        combined.extend_from_slice(query_embed);

        let logit: f64 = self.relation_weights[0].iter().zip(combined.iter()).map(|(w, c)| w * c).sum();
        1.0 / (1.0 + (-logit).exp())
    }

    pub fn classify(&self, support: &[(Vec<f64>, usize)], query: &[f64], n_way: usize) -> usize {
        let query_embed = self.encode(query);

        // Compute prototype for each class
        let mut class_prototypes: std::collections::HashMap<usize, Vec<f64>> = std::collections::HashMap::new();
        let mut class_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

        for (x, class) in support {
            let embed = self.encode(x);
            let entry = class_prototypes.entry(*class).or_insert_with(|| vec![0.0; self.embedding_dim]);
            for (e, val) in entry.iter_mut().zip(embed.iter()) {
                *e += val;
            }
            *class_counts.entry(*class).or_insert(0) += 1;
        }

        for (class, prototype) in class_prototypes.iter_mut() {
            let count = class_counts[class] as f64;
            for val in prototype.iter_mut() { *val /= count; }
        }

        // Find class with highest relation score
        class_prototypes.iter()
            .max_by(|(_, a), (_, b)| {
                let score_a = self.relation_score(a, &query_embed);
                let score_b = self.relation_score(b, &query_embed);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(class, _)| *class)
            .unwrap()
    }
}

/// Siamese few-shot network.
pub struct SiameseFewShot {
    pub embedding_dim: usize,
    pub weights: Vec<Vec<f64>>,
    pub learning_rate: f64,
}

impl SiameseFewShot {
    pub fn new(input_dim: usize, embedding_dim: usize, learning_rate: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();
        Self {
            embedding_dim, learning_rate,
            weights: (0..embedding_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        self.weights.iter().map(|w| {
            w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
        }).collect()
    }

    pub fn distance(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let e1 = self.encode(x1);
        let e2 = self.encode(x2);
        e1.iter().zip(e2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
    }

    pub fn classify_knn(&self, support: &[(Vec<f64>, usize)], query: &[f64], k: usize) -> usize {
        let mut distances: Vec<(usize, f64)> = support.iter()
            .map(|(x, class)| (*class, self.distance(x, query)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut votes: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &(class, _) in distances.iter().take(k) {
            *votes.entry(class).or_insert(0) += 1;
        }

        votes.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(class, _)| class)
            .unwrap()
    }

    /// Contrastive loss for training.
    pub fn contrastive_loss(&self, x1: &[f64], x2: &[f64], same_class: bool, margin: f64) -> f64 {
        let dist = self.distance(x1, x2);
        if same_class {
            dist * dist
        } else {
            (margin - dist).max(0.0).powi(2)
        }
    }
}

/// Matching networks with attention.
pub struct MatchingFewShot {
    pub embedding_dim: usize,
    pub weights: Vec<Vec<f64>>,
}

impl MatchingFewShot {
    pub fn new(input_dim: usize, embedding_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();
        Self {
            embedding_dim,
            weights: (0..embedding_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        self.weights.iter().map(|w| {
            w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
        }).collect()
    }

    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a * norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 }
    }

    pub fn classify(&self, support: &[(Vec<f64>, usize)], query: &[f64]) -> usize {
        let query_embed = self.encode(query);

        let similarities: Vec<(usize, f64)> = support.iter()
            .map(|(x, class)| (*class, Self::cosine_similarity(&self.encode(x), &query_embed)))
            .collect();

        // Softmax attention
        let max_sim = similarities.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
        let exp_sims: Vec<f64> = similarities.iter().map(|(_, s)| (s - max_sim).exp()).collect();
        let sum_exp: f64 = exp_sims.iter().sum();

        // Weighted vote
        let mut class_scores: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        for ((class, _), &exp_sim) in similarities.iter().zip(exp_sims.iter()) {
            *class_scores.entry(*class).or_insert(0.0) += exp_sim / sum_exp;
        }

        class_scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(class, _)| class)
            .unwrap()
    }
}

/// Test-time augmentation for few-shot.
pub fn test_time_augmentation<F>(model_fn: F, x: &[f64], n_augmentations: usize) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut seed = 42u64;
    let mut predictions = Vec::new();

    for _ in 0..n_augmentations {
        // Add noise augmentation
        let augmented: Vec<f64> = x.iter().map(|&xi| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.1 - 0.05;
            xi + noise
        }).collect();

        predictions.push(model_fn(&augmented));
    }

    // Average predictions
    let n = predictions[0].len();
    (0..n).map(|i| predictions.iter().map(|p| p[i]).sum::<f64>() / n_augmentations as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relation_network() {
        let rn = RelationNetwork::new(4, 8, 4, 0.01);
        let support = vec![
            (vec![1.0, 0.0, 0.0, 0.0], 0),
            (vec![0.0, 1.0, 0.0, 0.0], 1),
        ];
        let query = vec![0.9, 0.1, 0.0, 0.0];
        let class = rn.classify(&support, &query, 2);
        assert!(class < 2);
    }

    #[test]
    fn test_siamese_few_shot() {
        let sfs = SiameseFewShot::new(4, 8, 0.01);
        let support = vec![
            (vec![1.0, 0.0, 0.0, 0.0], 0),
            (vec![0.0, 1.0, 0.0, 0.0], 1),
        ];
        let query = vec![0.9, 0.1, 0.0, 0.0];
        let class = sfs.classify_knn(&support, &query, 1);
        assert_eq!(class, 0);
    }
}
