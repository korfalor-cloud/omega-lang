/// Self-supervised learning: masked autoencoders, rotation prediction, jigsaw puzzles.

/// Masked Autoencoder (MAE).
pub struct MaskedAutoencoder {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    pub mask_ratio: f64,
    pub encoder_weights: Vec<Vec<f64>>,
    pub decoder_weights: Vec<Vec<f64>>,
}

impl MaskedAutoencoder {
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize, mask_ratio: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale_enc = (2.0 / input_dim as f64).sqrt();
        let scale_dec = (2.0 / latent_dim as f64).sqrt();

        Self {
            input_dim, hidden_dim, latent_dim, mask_ratio,
            encoder_weights: (0..hidden_dim).map(|_| (0..input_dim).map(|_| rand(scale_enc)).collect()).collect(),
            decoder_weights: (0..input_dim).map(|_| (0..latent_dim).map(|_| rand(scale_dec)).collect()).collect(),
        }
    }

    pub fn mask_input(&self, x: &[f64], seed: u64) -> (Vec<f64>, Vec<usize>) {
        let mut rng = seed;
        let n_masks = (self.input_dim as f64 * self.mask_ratio) as usize;
        let mut indices: Vec<usize> = (0..self.input_dim).collect();

        // Fisher-Yates shuffle
        for i in (1..self.input_dim).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = ((rng >> 33) as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mask_indices: Vec<usize> = indices[..n_masks].to_vec();
        let mut masked = x.to_vec();
        for &idx in &mask_indices {
            masked[idx] = 0.0; // Mask with zero
        }

        (masked, mask_indices)
    }

    pub fn encode(&self, masked_input: &[f64]) -> Vec<f64> {
        self.encoder_weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(masked_input.iter()).map(|(wi, xi)| wi * xi).sum();
            sum.tanh()
        }).collect()
    }

    pub fn decode(&self, latent: &[f64]) -> Vec<f64> {
        self.decoder_weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(latent.iter()).map(|(wi, li)| wi * li).sum();
            sum
        }).collect()
    }

    pub fn forward(&self, x: &[f64], seed: u64) -> (Vec<f64>, Vec<usize>) {
        let (masked, mask_indices) = self.mask_input(x, seed);
        let latent = self.encode(&masked);
        let reconstructed = self.decode(&latent);
        (reconstructed, mask_indices)
    }

    pub fn loss(&self, x: &[f64], reconstructed: &[f64], mask_indices: &[usize]) -> f64 {
        // Only compute loss on masked positions
        mask_indices.iter().map(|&idx| {
            (x[idx] - reconstructed[idx]).powi(2)
        }).sum::<f64>() / mask_indices.len() as f64
    }
}

/// Rotation prediction pretext task.
pub struct RotationPredictor {
    pub feature_dim: usize,
    pub n_rotations: usize,
    pub weights: Vec<Vec<f64>>,
}

impl RotationPredictor {
    pub fn new(feature_dim: usize, n_rotations: usize, input_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();
        Self {
            feature_dim, n_rotations,
            weights: (0..feature_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn extract_features(&self, x: &[f64]) -> Vec<f64> {
        self.weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            sum.tanh()
        }).collect()
    }

    pub fn predict_rotation(&self, features: &[f64]) -> Vec<f64> {
        // Simple linear classifier
        let mut logits = vec![0.0; self.n_rotations];
        for (i, logit) in logits.iter_mut().enumerate() {
            *logit = features.iter().enumerate().map(|(j, &f)| {
                f * ((i * j) as f64 * 0.1).sin()
            }).sum();
        }
        softmax(&logits)
    }

    pub fn loss(&self, x: &[f64], rotation_label: usize) -> f64 {
        let features = self.extract_features(x);
        let probs = self.predict_rotation(&features);
        -probs[rotation_label].max(1e-15).ln()
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Jigsaw puzzle pretext task.
pub struct JigsawPredictor {
    pub n_patches: usize,
    pub n_permutations: usize,
    pub patch_dim: usize,
    pub weights: Vec<Vec<f64>>,
}

impl JigsawPredictor {
    pub fn new(n_patches: usize, n_permutations: usize, patch_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let input_dim = n_patches * patch_dim;
        let scale = (2.0 / input_dim as f64).sqrt();

        Self {
            n_patches, n_permutations, patch_dim,
            weights: (0..n_permutations).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn extract_features(&self, patches: &[Vec<f64>]) -> Vec<f64> {
        let mut features = Vec::new();
        for patch in patches {
            features.extend_from_slice(patch);
        }
        features
    }

    pub fn predict_permutation(&self, patches: &[Vec<f64>]) -> Vec<f64> {
        let features = self.extract_features(patches);
        let logits: Vec<f64> = self.weights.iter().map(|w| {
            w.iter().zip(features.iter()).map(|(wi, fi)| wi * fi).sum()
        }).collect();
        softmax(&logits)
    }

    pub fn loss(&self, patches: &[Vec<f64>], permutation_label: usize) -> f64 {
        let probs = self.predict_permutation(patches);
        -probs[permutation_label].max(1e-15).ln()
    }
}

/// Contrastive Predictive Coding (CPC).
pub struct CPC {
    pub encoder_dim: usize,
    pub predictor_dim: usize,
    pub n_negative: usize,
    pub encoder_weights: Vec<Vec<f64>>,
    pub predictor_weights: Vec<Vec<f64>>,
    pub temperature: f64,
}

impl CPC {
    pub fn new(encoder_dim: usize, predictor_dim: usize, input_dim: usize, n_negative: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale_enc = (2.0 / input_dim as f64).sqrt();
        let scale_pred = (2.0 / encoder_dim as f64).sqrt();

        Self {
            encoder_dim, predictor_dim, n_negative,
            encoder_weights: (0..encoder_dim).map(|_| (0..input_dim).map(|_| rand(scale_enc)).collect()).collect(),
            predictor_weights: (0..predictor_dim).map(|_| (0..encoder_dim).map(|_| rand(scale_pred)).collect()).collect(),
            temperature: 0.1,
        }
    }

    pub fn encode(&self, x: &[f64]) -> Vec<f64> {
        self.encoder_weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            sum.tanh()
        }).collect()
    }

    pub fn predict(&self, context: &[f64]) -> Vec<f64> {
        self.predictor_weights.iter().map(|w| {
            w.iter().zip(context.iter()).map(|(wi, ci)| wi * ci).sum()
        }).collect()
    }

    pub fn loss(&self, sequence: &[Vec<f64>], negatives: &[Vec<Vec<f64>>]) -> f64 {
        let t = sequence.len();
        let mut total_loss = 0.0;

        for i in 0..t - 1 {
            // Context: average of encodings up to time i
            let context: Vec<f64> = (0..self.encoder_dim).map(|d| {
                (0..=i).map(|j| self.encode(&sequence[j])[d]).sum::<f64>() / (i + 1) as f64
            }).collect();

            let prediction = self.predict(&context);
            let target = self.encode(&sequence[i + 1]);

            // Positive similarity
            let pos_sim = cosine_similarity(&prediction, &target) / self.temperature;

            // Negative similarities
            let neg_sims: Vec<f64> = negatives[i].iter()
                .map(|neg| cosine_similarity(&prediction, &self.encode(neg)) / self.temperature)
                .collect();

            let mut all_sims = vec![pos_sim];
            all_sims.extend_from_slice(&neg_sims);

            let max_sim = all_sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = all_sims.iter().map(|s| (s - max_sim).exp()).sum();
            total_loss += -(pos_sim - max_sim - sum_exp.ln());
        }

        total_loss / (t - 1) as f64
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a * norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 }
}

/// DINO (Self-Distillation with No Labels).
pub struct DINO {
    pub student_dim: usize,
    pub teacher_dim: usize,
    pub student_weights: Vec<Vec<f64>>,
    pub teacher_weights: Vec<Vec<f64>>,
    pub center: Vec<f64>,
    pub momentum: f64,
    pub temperature_student: f64,
    pub temperature_teacher: f64,
}

impl DINO {
    pub fn new(student_dim: usize, teacher_dim: usize, input_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();
        let student_weights: Vec<Vec<f64>> = (0..student_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect();
        let teacher_weights = student_weights.clone();

        Self {
            student_dim, teacher_dim, student_weights, teacher_weights,
            center: vec![0.0; teacher_dim],
            momentum: 0.996,
            temperature_student: 0.1,
            temperature_teacher: 0.04,
        }
    }

    pub fn student_forward(&self, x: &[f64]) -> Vec<f64> {
        self.student_weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            sum.tanh()
        }).collect()
    }

    pub fn teacher_forward(&self, x: &[f64]) -> Vec<f64> {
        self.teacher_weights.iter().map(|w| {
            let sum: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            sum.tanh()
        }).collect()
    }

    pub fn loss(&self, x_global: &[f64], x_local: &[f64]) -> f64 {
        let student_global = softmax(&self.student_forward(x_global).iter().map(|&z| z / self.temperature_student).collect::<Vec<f64>>());
        let student_local = softmax(&self.student_forward(x_local).iter().map(|&z| z / self.temperature_student).collect::<Vec<f64>>());
        let teacher_global = softmax(&self.teacher_forward(x_global).iter().map(|&z| (z - self.center.iter().sum::<f64>() / self.center.len() as f64) / self.temperature_teacher).collect::<Vec<f64>>());

        // Cross-entropy: student_local predicts teacher_global
        -teacher_global.iter().zip(student_local.iter())
            .map(|(t, s)| t * s.max(1e-15).ln())
            .sum::<f64>()
    }

    pub fn update_teacher(&mut self) {
        for (t, s) in self.teacher_weights.iter_mut().zip(self.student_weights.iter()) {
            for (ti, si) in t.iter_mut().zip(s.iter()) {
                *ti = self.momentum * *ti + (1.0 - self.momentum) * *si;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mae() {
        let mae = MaskedAutoencoder::new(10, 8, 4, 0.75);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (reconstructed, mask_indices) = mae.forward(&x, 42);
        assert_eq!(reconstructed.len(), 10);
        assert_eq!(mask_indices.len(), 7); // 75% of 10
    }

    #[test]
    fn test_rotation_predictor() {
        let rp = RotationPredictor::new(8, 4, 10);
        let x = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let loss = rp.loss(&x, 0);
        assert!(loss > 0.0);
    }
}
