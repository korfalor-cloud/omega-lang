/// Data augmentation: noise injection, mixup, cutout, feature space augmentation.

/// Add Gaussian noise.
pub fn add_gaussian_noise(data: &[f64], std: f64, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    data.iter().map(|&x| {
        let u1 = ((rng >> 33) as f64) / (1u64 << 31) as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = ((rng >> 33) as f64) / (1u64 << 31) as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let noise = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        x + noise * std
    }).collect()
}

/// Mixup augmentation.
pub fn mixup(x1: &[f64], y1: &[f64], x2: &[f64], y2: &[f64], alpha: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = seed;
    let lambda = if alpha > 0.0 {
        // Sample from Beta(alpha, alpha)
        let u = ((rng >> 33) as f64) / (1u64 << 31) as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Simplified Beta sampling
        u.powf(alpha) / (u.powf(alpha) + (1.0 - u).powf(alpha))
    } else {
        0.5
    };

    let mixed_x: Vec<f64> = x1.iter().zip(x2.iter()).map(|(&a, &b)| lambda * a + (1.0 - lambda) * b).collect();
    let mixed_y: Vec<f64> = y1.iter().zip(y2.iter()).map(|(&a, &b)| lambda * a + (1.0 - lambda) * b).collect();
    (mixed_x, mixed_y)
}

/// Cutout augmentation (zero out random features).
pub fn cutout(data: &[f64], n_features: usize, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    let mut result = data.to_vec();
    let n = data.len();

    for _ in 0..n_features.min(n) {
        let idx = ((rng >> 33) as usize) % n;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        result[idx] = 0.0;
    }

    result
}

/// Feature masking augmentation.
pub fn feature_masking(data: &[f64], mask_prob: f64, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    data.iter().map(|&x| {
        let r = ((rng >> 33) as f64) / (1u64 << 31) as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if r < mask_prob { 0.0 } else { x }
    }).collect()
}

/// Random scaling augmentation.
pub fn random_scale(data: &[f64], scale_range: (f64, f64), seed: u64) -> Vec<f64> {
    let mut rng = seed;
    let scale = scale_range.0 + ((rng >> 33) as f64) / (1u64 << 31) as f64 * (scale_range.1 - scale_range.0);
    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    data.iter().map(|&x| x * scale).collect()
}

/// Random shift augmentation.
pub fn random_shift(data: &[f64], shift_range: (f64, f64), seed: u64) -> Vec<f64> {
    let mut rng = seed;
    let shift = shift_range.0 + ((rng >> 33) as f64) / (1u64 << 31) as f64 * (shift_range.1 - shift_range.0);
    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    data.iter().map(|&x| x + shift).collect()
}

/// Time series augmentation: window slicing.
pub fn window_slice(data: &[f64], window_size: usize, seed: u64) -> Vec<f64> {
    let n = data.len();
    if window_size >= n { return data.to_vec(); }

    let mut rng = seed;
    let start = ((rng >> 33) as usize) % (n - window_size);
    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);

    data[start..start + window_size].to_vec()
}

/// Time series augmentation: permutation.
pub fn temporal_permutation(data: &[f64], n_segments: usize, seed: u64) -> Vec<f64> {
    let n = data.len();
    let segment_size = n / n_segments;
    let mut segments: Vec<Vec<f64>> = data.chunks(segment_size).map(|c| c.to_vec()).collect();

    // Shuffle segments
    let mut rng = seed;
    for i in (1..segments.len()).rev() {
        let j = ((rng >> 33) as usize) % (i + 1);
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        segments.swap(i, j);
    }

    segments.into_iter().flatten().collect()
}

/// Magnitude warping.
pub fn magnitude_warp(data: &[f64], sigma: f64, seed: u64) -> Vec<f64> {
    let n = data.len();
    let mut rng = seed;

    // Generate smooth random curve
    let warps: Vec<f64> = (0..n).map(|_| {
        let u1 = ((rng >> 33) as f64) / (1u64 << 31) as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = ((rng >> 33) as f64) / (1u64 << 31) as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let noise = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        1.0 + noise * sigma
    }).collect();

    data.iter().zip(warps.iter()).map(|(&x, &w)| x * w).collect()
}

/// SMOTE (Synthetic Minority Over-sampling Technique).
pub fn smote(minority_samples: &[Vec<f64>], n_synthetic: usize, k: usize, seed: u64) -> Vec<Vec<f64>> {
    let n = minority_samples.len();
    let mut rng = seed;
    let mut synthetic = Vec::new();

    for _ in 0..n_synthetic {
        // Pick random sample
        let idx = ((rng >> 33) as usize) % n;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);

        // Find k nearest neighbors
        let mut distances: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != idx)
            .map(|j| {
                let dist: f64 = minority_samples[idx].iter().zip(minority_samples[j].iter())
                    .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                (j, dist)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Pick random neighbor
        let neighbor_idx = distances[((rng >> 33) as usize) % k.min(distances.len())].0;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);

        // Interpolate
        let lambda = ((rng >> 33) as f64) / (1u64 << 31) as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);

        let new_sample: Vec<f64> = minority_samples[idx].iter()
            .zip(minority_samples[neighbor_idx].iter())
            .map(|(&a, &b)| a + lambda * (b - a))
            .collect();

        synthetic.push(new_sample);
    }

    synthetic
}

/// ADASYN (Adaptive Synthetic Sampling).
pub fn adasyn(minority_samples: &[Vec<f64>], majority_count: usize, seed: u64) -> Vec<Vec<f64>> {
    let n = minority_samples.len();
    let imbalance_ratio = n as f64 / majority_count as f64;

    // Compute density for each sample
    let densities: Vec<f64> = minority_samples.iter().map(|x| {
        let neighbor_count = minority_samples.iter().filter(|other| {
            let dist: f64 = x.iter().zip(other.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            dist < 1.0
        }).count();
        neighbor_count as f64 / n as f64
    }).collect();

    let total_density: f64 = densities.iter().sum();
    let n_synthetic = ((majority_count - n) as f64 * imbalance_ratio) as usize;

    let mut rng = seed;
    let mut synthetic = Vec::new();

    for (i, sample) in minority_samples.iter().enumerate() {
        let n_generate = (densities[i] / total_density * n_synthetic as f64) as usize;

        for _ in 0..n_generate {
            let neighbor_idx = ((rng >> 33) as usize) % n;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);

            let lambda = ((rng >> 33) as f64) / (1u64 << 31) as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);

            let new_sample: Vec<f64> = sample.iter()
                .zip(minority_samples[neighbor_idx].iter())
                .map(|(&a, &b)| a + lambda * (b - a))
                .collect();

            synthetic.push(new_sample);
        }
    }

    synthetic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_noise() {
        let data = vec![1.0, 2.0, 3.0];
        let noisy = add_gaussian_noise(&data, 0.1, 42);
        assert_eq!(noisy.len(), 3);
        for (orig, noisy) in data.iter().zip(noisy.iter()) {
            assert!((orig - noisy).abs() < 1.0);
        }
    }

    #[test]
    fn test_mixup() {
        let x1 = vec![1.0, 0.0];
        let y1 = vec![1.0, 0.0];
        let x2 = vec![0.0, 1.0];
        let y2 = vec![0.0, 1.0];
        let (mx, my) = mixup(&x1, &y1, &x2, &y2, 1.0, 42);
        assert_eq!(mx.len(), 2);
        assert_eq!(my.len(), 2);
    }

    #[test]
    fn test_smote() {
        let minority = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let synthetic = smote(&minority, 5, 2, 42);
        assert_eq!(synthetic.len(), 5);
        assert_eq!(synthetic[0].len(), 2);
    }
}
