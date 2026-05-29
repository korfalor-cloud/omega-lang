/// Meta-learning: MAML, prototypical networks, matching networks.

/// MAML (Model-Agnostic Meta-Learning).
pub struct MAML {
    pub inner_lr: f64,
    pub outer_lr: f64,
    pub n_inner_steps: usize,
    pub params: Vec<f64>,
    pub param_dim: usize,
}

impl MAML {
    pub fn new(param_dim: usize, inner_lr: f64, outer_lr: f64, n_inner_steps: usize) -> Self {
        let mut seed = 42u64;
        let params: Vec<f64> = (0..param_dim).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.1 - 0.05
        }).collect();

        Self { inner_lr, outer_lr, n_inner_steps, params, param_dim }
    }

    /// Inner loop: adapt parameters to task.
    pub fn adapt<F, G>(&self, task_data: &[(Vec<f64>, f64)], loss_fn: F, grad_fn: G) -> Vec<f64>
    where
        F: Fn(&[f64], &[f64], f64) -> f64,
        G: Fn(&[f64], &[f64], f64) -> Vec<f64>,
    {
        let mut adapted_params = self.params.clone();

        for _ in 0..self.n_inner_steps {
            let mut grad = vec![0.0; self.param_dim];
            for (x, y) in task_data {
                let g = grad_fn(&adapted_params, x, *y);
                for i in 0..self.param_dim {
                    grad[i] += g[i];
                }
            }
            for i in 0..self.param_dim {
                adapted_params[i] -= self.inner_lr * grad[i] / task_data.len() as f64;
            }
        }

        adapted_params
    }

    /// Outer loop: update meta-parameters.
    pub fn meta_update<F, G>(
        &mut self,
        tasks: &[Vec<(Vec<f64>, f64)>],
        loss_fn: F,
        grad_fn: G,
    )
    where
        F: Fn(&[f64], &[f64], f64) -> f64 + Copy,
        G: Fn(&[f64], &[f64], f64) -> Vec<f64> + Copy,
    {
        let mut meta_grad = vec![0.0; self.param_dim];

        for task in tasks {
            // Split task into support and query
            let split = task.len() / 2;
            let support = &task[..split];
            let query = &task[split..];

            // Adapt to task
            let adapted = self.adapt(support, loss_fn, grad_fn);

            // Compute gradient on query set
            for (x, y) in query {
                let g = grad_fn(&adapted, x, *y);
                for i in 0..self.param_dim {
                    meta_grad[i] += g[i];
                }
            }
        }

        // Update meta-parameters
        for i in 0..self.param_dim {
            self.params[i] -= self.outer_lr * meta_grad[i] / tasks.len() as f64;
        }
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        self.params.iter().zip(x.iter()).map(|(p, xi)| p * xi).sum()
    }
}

/// Prototypical Networks.
pub struct PrototypicalNetwork {
    pub embedding_dim: usize,
    pub input_dim: usize,
    pub weights: Vec<Vec<f64>>,
}

impl PrototypicalNetwork {
    pub fn new(input_dim: usize, embedding_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();
        Self {
            embedding_dim, input_dim,
            weights: (0..embedding_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn embed(&self, x: &[f64]) -> Vec<f64> {
        self.weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            sum.tanh()
        }).collect()
    }

    pub fn compute_prototypes(&self, support: &[(Vec<f64>, usize)], n_classes: usize) -> Vec<Vec<f64>> {
        let mut prototypes = vec![vec![0.0; self.embedding_dim]; n_classes];
        let mut counts = vec![0usize; n_classes];

        for (x, label) in support {
            let embedding = self.embed(x);
            for (i, &val) in embedding.iter().enumerate() {
                prototypes[*label][i] += val;
            }
            counts[*label] += 1;
        }

        for (i, prototype) in prototypes.iter_mut().enumerate() {
            if counts[i] > 0 {
                for val in prototype.iter_mut() {
                    *val /= counts[i] as f64;
                }
            }
        }

        prototypes
    }

    pub fn classify(&self, x: &[f64], prototypes: &[Vec<f64>]) -> usize {
        let embedding = self.embed(x);

        prototypes.iter().enumerate()
            .min_by(|(_, a), (_, b)| {
                let dist_a: f64 = a.iter().zip(embedding.iter()).map(|(ai, ei)| (ai - ei).powi(2)).sum();
                let dist_b: f64 = b.iter().zip(embedding.iter()).map(|(bi, ei)| (bi - ei).powi(2)).sum();
                dist_a.partial_cmp(&dist_b).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap()
    }
}

/// Matching Networks.
pub struct MatchingNetwork {
    pub embedding_dim: usize,
    pub input_dim: usize,
    pub weights: Vec<Vec<f64>>,
}

impl MatchingNetwork {
    pub fn new(input_dim: usize, embedding_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();
        Self {
            embedding_dim, input_dim,
            weights: (0..embedding_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn embed(&self, x: &[f64]) -> Vec<f64> {
        self.weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            sum.tanh()
        }).collect()
    }

    /// Cosine similarity.
    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a * norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 }
    }

    pub fn classify(&self, query: &[f64], support: &[(Vec<f64>, usize)], n_classes: usize) -> usize {
        let query_embed = self.embed(query);
        let support_embeds: Vec<(Vec<f64>, usize)> = support.iter()
            .map(|(x, label)| (self.embed(x), *label))
            .collect();

        // Compute attention weights
        let similarities: Vec<f64> = support_embeds.iter()
            .map(|(embed, _)| Self::cosine_similarity(&query_embed, embed))
            .collect();

        let max_sim = similarities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sims: Vec<f64> = similarities.iter().map(|s| (s - max_sim).exp()).collect();
        let sum_exp: f64 = exp_sims.iter().sum();
        let attention: Vec<f64> = exp_sims.iter().map(|e| e / sum_exp).collect();

        // Weighted vote
        let mut class_scores = vec![0.0; n_classes];
        for ((_, label), &attn) in support_embeds.iter().zip(attention.iter()) {
            class_scores[*label] += attn;
        }

        class_scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }
}

/// Siamese Network for few-shot learning.
pub struct SiameseNetwork {
    pub input_dim: usize,
    pub embedding_dim: usize,
    pub weights: Vec<Vec<f64>>,
}

impl SiameseNetwork {
    pub fn new(input_dim: usize, embedding_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();
        Self {
            input_dim, embedding_dim,
            weights: (0..embedding_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn embed(&self, x: &[f64]) -> Vec<f64> {
        self.weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            sum.tanh()
        }).collect()
    }

    pub fn distance(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let e1 = self.embed(x1);
        let e2 = self.embed(x2);
        e1.iter().zip(e2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
    }

    pub fn are_same_class(&self, x1: &[f64], x2: &[f64], threshold: f64) -> bool {
        self.distance(x1, x2) < threshold
    }
}

/// Reptile meta-learning algorithm.
pub struct Reptile {
    pub params: Vec<f64>,
    pub inner_lr: f64,
    pub outer_lr: f64,
    pub n_inner_steps: usize,
}

impl Reptile {
    pub fn new(param_dim: usize, inner_lr: f64, outer_lr: f64, n_inner_steps: usize) -> Self {
        let mut seed = 42u64;
        let params: Vec<f64> = (0..param_dim).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.1 - 0.05
        }).collect();

        Self { params, inner_lr, outer_lr, n_inner_steps }
    }

    pub fn adapt<F, G>(&self, task_data: &[(Vec<f64>, f64)], loss_fn: F, grad_fn: G) -> Vec<f64>
    where
        F: Fn(&[f64], &[f64], f64) -> f64,
        G: Fn(&[f64], &[f64], f64) -> Vec<f64>,
    {
        let mut adapted = self.params.clone();

        for _ in 0..self.n_inner_steps {
            let mut grad = vec![0.0; self.params.len()];
            for (x, y) in task_data {
                let g = grad_fn(&adapted, x, *y);
                for i in 0..self.params.len() {
                    grad[i] += g[i];
                }
            }
            for i in 0..self.params.len() {
                adapted[i] -= self.inner_lr * grad[i] / task_data.len() as f64;
            }
        }

        adapted
    }

    pub fn meta_update<F, G>(&mut self, tasks: &[Vec<(Vec<f64>, f64)>], loss_fn: F, grad_fn: G)
    where
        F: Fn(&[f64], &[f64], f64) -> f64 + Copy,
        G: Fn(&[f64], &[f64], f64) -> Vec<f64> + Copy,
    {
        let original = self.params.clone();

        for task in tasks {
            let adapted = self.adapt(task, loss_fn, grad_fn);
            for i in 0..self.params.len() {
                self.params[i] += self.outer_lr * (adapted[i] - original[i]) / tasks.len() as f64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prototypical_network() {
        let proto_net = PrototypicalNetwork::new(2, 4);
        let support = vec![
            (vec![1.0, 0.0], 0),
            (vec![0.0, 1.0], 1),
        ];
        let prototypes = proto_net.compute_prototypes(&support, 2);
        assert_eq!(prototypes.len(), 2);

        let class = proto_net.classify(&[0.9, 0.1], &prototypes);
        assert_eq!(class, 0);
    }

    #[test]
    fn test_matching_network() {
        let match_net = MatchingNetwork::new(2, 4);
        let support = vec![
            (vec![1.0, 0.0], 0),
            (vec![0.0, 1.0], 1),
        ];
        let class = match_net.classify(&[0.9, 0.1], &support, 2);
        assert_eq!(class, 0);
    }

    #[test]
    fn test_siamese() {
        let siamese = SiameseNetwork::new(2, 4);
        let dist = siamese.distance(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(dist > 0.0);
    }
}
