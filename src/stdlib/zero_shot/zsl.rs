/// Zero-shot learning: attribute-based, embedding-based, generative.

/// Attribute-based zero-shot learning.
pub struct AttributeZSL {
    pub class_attributes: Vec<Vec<f64>>, // Attribute vector per class
    pub attribute_dim: usize,
    pub n_classes: usize,
    pub classifier_weights: Vec<Vec<f64>>,
}

impl AttributeZSL {
    pub fn new(class_attributes: Vec<Vec<f64>>, input_dim: usize) -> Self {
        let attribute_dim = class_attributes[0].len();
        let n_classes = class_attributes.len();
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();
        Self {
            class_attributes, attribute_dim, n_classes,
            classifier_weights: (0..attribute_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    /// Predict attributes from input.
    pub fn predict_attributes(&self, x: &[f64]) -> Vec<f64> {
        self.classifier_weights.iter().map(|w| {
            w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum()
        }).collect()
    }

    /// Classify by nearest neighbor in attribute space.
    pub fn classify(&self, x: &[f64]) -> usize {
        let predicted_attrs = self.predict_attributes(x);

        self.class_attributes.iter().enumerate()
            .min_by(|(_, a), (_, b)| {
                let dist_a: f64 = a.iter().zip(predicted_attrs.iter()).map(|(ai, pi)| (ai - pi).powi(2)).sum();
                let dist_b: f64 = b.iter().zip(predicted_attrs.iter()).map(|(bi, pi)| (bi - pi).powi(2)).sum();
                dist_a.partial_cmp(&dist_b).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Cosine similarity based classification.
    pub fn classify_cosine(&self, x: &[f64]) -> usize {
        let predicted_attrs = self.predict_attributes(x);

        self.class_attributes.iter().enumerate()
            .max_by(|(_, a), (_, b)| {
                let sim_a = cosine_similarity(a, &predicted_attrs);
                let sim_b = cosine_similarity(b, &predicted_attrs);
                sim_a.partial_cmp(&sim_b).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap()
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a * norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 }
}

/// Embedding-based zero-shot learning (DeViSE-style).
pub struct EmbeddingZSL {
    pub class_embeddings: Vec<Vec<f64>>,
    pub visual_weights: Vec<Vec<f64>>,
    pub embedding_dim: usize,
    pub learning_rate: f64,
}

impl EmbeddingZSL {
    pub fn new(class_embeddings: Vec<Vec<f64>>, input_dim: usize, learning_rate: f64) -> Self {
        let embedding_dim = class_embeddings[0].len();
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();
        Self {
            class_embeddings, learning_rate, embedding_dim,
            visual_weights: (0..embedding_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    /// Project visual feature to embedding space.
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        self.visual_weights.iter().map(|w| {
            w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum()
        }).collect()
    }

    /// Classify by nearest class embedding.
    pub fn classify(&self, x: &[f64]) -> usize {
        let projected = self.project(x);

        self.class_embeddings.iter().enumerate()
            .min_by(|(_, a), (_, b)| {
                let dist_a: f64 = a.iter().zip(projected.iter()).map(|(ai, pi)| (ai - pi).powi(2)).sum();
                let dist_b: f64 = b.iter().zip(projected.iter()).map(|(bi, pi)| (bi - pi).powi(2)).sum();
                dist_a.partial_cmp(&dist_b).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Contrastive loss for training.
    pub fn contrastive_loss(&self, x: &[f64], correct_class: usize) -> f64 {
        let projected = self.project(x);
        let correct_embed = &self.class_embeddings[correct_class];

        let pos_dist: f64 = projected.iter().zip(correct_embed.iter()).map(|(a, b)| (a - b).powi(2)).sum();

        let mut neg_distances = Vec::new();
        for (i, embed) in self.class_embeddings.iter().enumerate() {
            if i != correct_class {
                let dist: f64 = projected.iter().zip(embed.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                neg_distances.push(dist);
            }
        }

        let min_neg = neg_distances.iter().cloned().fold(f64::INFINITY, f64::min);
        (pos_dist - min_neg + 1.0).max(0.0) // Margin-based loss
    }
}

/// Generative zero-shot learning: generate features for unseen classes.
pub struct GenerativeZSL {
    pub noise_dim: usize,
    pub feature_dim: usize,
    pub generator_weights: Vec<Vec<f64>>,
    pub discriminator_weights: Vec<Vec<f64>>,
    pub learning_rate: f64,
}

impl GenerativeZSL {
    pub fn new(noise_dim: usize, feature_dim: usize, learning_rate: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let gen_scale = (2.0 / noise_dim as f64).sqrt();
        let dis_scale = (2.0 / feature_dim as f64).sqrt();

        Self {
            noise_dim, feature_dim, learning_rate,
            generator_weights: (0..feature_dim).map(|_| (0..noise_dim).map(|_| rand(gen_scale)).collect()).collect(),
            discriminator_weights: (0..1).map(|_| (0..feature_dim).map(|_| rand(dis_scale)).collect()).collect(),
        }
    }

    /// Generate features from noise.
    pub fn generate(&self, noise: &[f64]) -> Vec<f64> {
        self.generator_weights.iter().map(|w| {
            w.iter().zip(noise.iter()).map(|(wi, ni)| wi * ni).sum::<f64>().tanh()
        }).collect()
    }

    /// Discriminate real vs generated features.
    pub fn discriminate(&self, features: &[f64]) -> f64 {
        let logit: f64 = self.discriminator_weights[0].iter().zip(features.iter()).map(|(w, f)| w * f).sum();
        1.0 / (1.0 + (-logit).exp())
    }

    /// Generate features for unseen class.
    pub fn generate_for_class(&self, class_attributes: &[f64], n_samples: usize) -> Vec<Vec<f64>> {
        let mut seed = 42u64;
        (0..n_samples).map(|_| {
            let noise: Vec<f64> = (0..self.noise_dim).map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) as f64) / (1u64 << 31) as f64 * 2.0 - 1.0
            }).collect();
            self.generate(&noise)
        }).collect()
    }
}

/// Transductive zero-shot learning.
pub struct TransductiveZSL {
    pub seen_classes: Vec<Vec<f64>>,
    pub unseen_classes: Vec<Vec<f64>>,
    pub labeled_data: Vec<(Vec<f64>, usize)>,
    pub unlabeled_data: Vec<Vec<f64>>,
    pub learning_rate: f64,
}

impl TransductiveZSL {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            seen_classes: Vec::new(),
            unseen_classes: Vec::new(),
            labeled_data: Vec::new(),
            unlabeled_data: Vec::new(),
            learning_rate,
        }
    }

    /// Label propagation from seen to unseen classes.
    pub fn label_propagation(&self, k: usize) -> Vec<usize> {
        self.unlabeled_data.iter().map(|x| {
            // Find k nearest neighbors in labeled data
            let mut distances: Vec<(usize, f64)> = self.labeled_data.iter().enumerate()
                .map(|(i, (labeled_x, _))| {
                    let dist: f64 = x.iter().zip(labeled_x.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                    (i, dist)
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Vote among k nearest
            let mut class_votes: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            for &(idx, _) in distances.iter().take(k) {
                let class = self.labeled_data[idx].1;
                *class_votes.entry(class).or_insert(0) += 1;
            }

            class_votes.into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(class, _)| class)
                .unwrap_or(0)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_zsl() {
        let class_attrs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let zsl = AttributeZSL::new(class_attrs, 4);
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let class = zsl.classify(&x);
        assert!(class < 3);
    }

    #[test]
    fn test_embedding_zsl() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let zsl = EmbeddingZSL::new(embeddings, 4, 0.01);
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let class = zsl.classify(&x);
        assert!(class < 2);
    }
}
