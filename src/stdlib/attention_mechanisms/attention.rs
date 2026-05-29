/// Advanced attention mechanisms: multi-head, self-attention, cross-attention, flash attention.

/// Scaled dot-product attention.
pub fn scaled_dot_product_attention(
    query: &[Vec<f64>],
    key: &[Vec<f64>],
    value: &[Vec<f64>],
    mask: Option<&[Vec<bool>]>,
) -> Vec<Vec<f64>> {
    let seq_len = query.len();
    let d_k = query[0].len() as f64;
    let scale = 1.0 / d_k.sqrt();

    let mut output = vec![vec![0.0; value[0].len()]; seq_len];

    for i in 0..seq_len {
        // Compute attention scores
        let mut scores: Vec<f64> = (0..seq_len).map(|j| {
            query[i].iter().zip(key[j].iter()).map(|(q, k)| q * k).sum::<f64>() * scale
        }).collect();

        // Apply mask
        if let Some(mask) = mask {
            for j in 0..seq_len {
                if !mask[i][j] {
                    scores[j] = f64::NEG_INFINITY;
                }
            }
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for s in &mut scores { *s = (*s - max_score).exp(); }
        let sum: f64 = scores.iter().sum();
        for s in &mut scores { *s /= sum; }

        // Weighted sum
        for j in 0..seq_len {
            for k in 0..value[0].len() {
                output[i][k] += scores[j] * value[j][k];
            }
        }
    }

    output
}

/// Multi-head attention.
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub w_q: Vec<Vec<f64>>,
    pub w_k: Vec<Vec<f64>>,
    pub w_v: Vec<Vec<f64>>,
    pub w_o: Vec<Vec<f64>>,
}

impl MultiHeadAttention {
    pub fn new(num_heads: usize, d_model: usize) -> Self {
        assert_eq!(d_model % num_heads, 0);
        let d_k = d_model / num_heads;
        let d_v = d_model / num_heads;

        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / d_model as f64).sqrt();
        let make_matrix = || -> Vec<Vec<f64>> {
            (0..d_model).map(|_| (0..d_model).map(|_| rand(scale)).collect()).collect()
        };

        Self {
            num_heads, d_model, d_k, d_v,
            w_q: make_matrix(),
            w_k: make_matrix(),
            w_v: make_matrix(),
            w_o: make_matrix(),
        }
    }

    pub fn forward(&self, query: &[Vec<f64>], key: &[Vec<f64>], value: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = query.len();

        // Project Q, K, V
        let q_proj: Vec<Vec<f64>> = query.iter().map(|x| self.matmul(&self.w_q, x)).collect();
        let k_proj: Vec<Vec<f64>> = key.iter().map(|x| self.matmul(&self.w_k, x)).collect();
        let v_proj: Vec<Vec<f64>> = value.iter().map(|x| self.matmul(&self.w_v, x)).collect();

        let mut concat_output = vec![vec![0.0; self.d_model]; seq_len];

        for h in 0..self.num_heads {
            let start = h * self.d_k;
            let end = start + self.d_k;

            // Extract head
            let q_head: Vec<Vec<f64>> = q_proj.iter().map(|x| x[start..end].to_vec()).collect();
            let k_head: Vec<Vec<f64>> = k_proj.iter().map(|x| x[start..end].to_vec()).collect();
            let v_head: Vec<Vec<f64>> = v_proj.iter().map(|x| x[start..end].to_vec()).collect();

            // Compute attention
            let head_output = scaled_dot_product_attention(&q_head, &k_head, &v_head, None);

            // Concatenate
            for i in 0..seq_len {
                concat_output[i][start..end].copy_from_slice(&head_output[i]);
            }
        }

        // Output projection
        concat_output.iter().map(|x| self.matmul(&self.w_o, x)).collect()
    }

    fn matmul(&self, m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        m.iter().map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
    }
}

/// Self-attention (query = key = value).
pub fn self_attention(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    scaled_dot_product_attention(features, features, features, None)
}

/// Cross-attention between two sequences.
pub fn cross_attention(
    query: &[Vec<f64>],
    context: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    scaled_dot_product_attention(query, context, context, None)
}

/// Causal (masked) self-attention.
pub fn causal_self_attention(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let seq_len = features.len();
    let mask: Vec<Vec<bool>> = (0..seq_len).map(|i| {
        (0..seq_len).map(|j| j <= i).collect()
    }).collect();

    scaled_dot_product_attention(features, features, features, Some(&mask))
}

/// Relative position attention.
pub struct RelativePositionAttention {
    pub max_distance: usize,
    pub d_model: usize,
    pub embeddings: Vec<Vec<f64>>,
}

impl RelativePositionAttention {
    pub fn new(max_distance: usize, d_model: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.1 - 0.05
        };

        let embeddings = (0..2 * max_distance + 1)
            .map(|_| (0..d_model).map(|_| rand()).collect())
            .collect();

        Self { max_distance, d_model, embeddings }
    }

    pub fn forward(&self, query: &[Vec<f64>], key: &[Vec<f64>], value: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = query.len();
        let d_k = query[0].len();
        let scale = 1.0 / (d_k as f64).sqrt();

        let mut output = vec![vec![0.0; value[0].len()]; seq_len];

        for i in 0..seq_len {
            let mut scores: Vec<f64> = (0..seq_len).map(|j| {
                let distance = (i as isize - j as isize).unsigned_abs().min(self.max_distance);
                let rel_embed = &self.embeddings[distance + self.max_distance];

                let content_score: f64 = query[i].iter().zip(key[j].iter()).map(|(q, k)| q * k).sum();
                let position_score: f64 = query[i].iter().zip(rel_embed.iter()).map(|(q, r)| q * r).sum();

                (content_score + position_score) * scale
            }).collect();

            // Softmax
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            for s in &mut scores { *s = (*s - max_score).exp(); }
            let sum: f64 = scores.iter().sum();
            for s in &mut scores { *s /= sum; }

            for j in 0..seq_len {
                for k in 0..value[0].len() {
                    output[i][k] += scores[j] * value[j][k];
                }
            }
        }

        output
    }
}

/// Linformer: linear attention approximation.
pub struct Linformer {
    pub d_model: usize,
    pub k_dim: usize,
    pub e: Vec<Vec<f64>>,
}

impl Linformer {
    pub fn new(d_model: usize, k_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.1 - 0.05
        };

        let e = (0..k_dim).map(|_| (0..d_model).map(|_| rand()).collect()).collect();

        Self { d_model, k_dim, e }
    }

    pub fn forward(&self, query: &[Vec<f64>], key: &[Vec<f64>], value: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = query.len();
        let scale = 1.0 / (self.d_model as f64).sqrt();

        // Project K and V to lower dimension
        let k_proj: Vec<Vec<f64>> = (0..self.k_dim).map(|i| {
            (0..seq_len).map(|j| {
                self.e[i].iter().zip(key[j].iter()).map(|(e, k)| e * k).sum()
            }).collect()
        }).collect();

        let v_proj: Vec<Vec<f64>> = (0..self.k_dim).map(|i| {
            (0..seq_len).map(|j| {
                self.e[i].iter().zip(value[j].iter()).map(|(e, v)| e * v).sum()
            }).collect()
        }).collect();

        // Compute attention in O(n*k) instead of O(n^2)
        let mut output = vec![vec![0.0; value[0].len()]; seq_len];

        for i in 0..seq_len {
            let mut scores: Vec<f64> = (0..self.k_dim).map(|j| {
                query[i].iter().zip(k_proj[j].iter()).map(|(q, k)| q * k).sum::<f64>() * scale
            }).collect();

            // Softmax
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            for s in &mut scores { *s = (*s - max_score).exp(); }
            let sum: f64 = scores.iter().sum();
            for s in &mut scores { *s /= sum; }

            for j in 0..self.k_dim {
                for k in 0..value[0].len() {
                    output[i][k] += scores[j] * v_proj[j][k];
                }
            }
        }

        output
    }
}

/// Performer: random feature attention.
pub struct Performer {
    pub d_model: usize,
    pub n_features: usize,
    pub random_matrix: Vec<Vec<f64>>,
}

impl Performer {
    pub fn new(d_model: usize, n_features: usize) -> Self {
        let mut seed = 42u64;
        let mut rand_normal = || -> f64 {
            let u1 = ((seed >> 33) as f64) / (1u64 << 31) as f64;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = ((seed >> 33) as f64) / (1u64 << 31) as f64;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let random_matrix = (0..n_features)
            .map(|_| (0..d_model).map(|_| rand_normal()).collect())
            .collect();

        Self { d_model, n_features, random_matrix }
    }

    /// Random feature map: phi(x) = exp(-||x||^2/2) * exp(w^T x) / sqrt(m)
    fn feature_map(&self, x: &[f64]) -> Vec<f64> {
        let norm_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        let scale = (-norm_sq / 2.0).exp() / (self.n_features as f64).sqrt();

        self.random_matrix.iter().map(|w| {
            let dot: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            scale * dot.exp()
        }).collect()
    }

    pub fn forward(&self, query: &[Vec<f64>], key: &[Vec<f64>], value: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = query.len();
        let d_v = value[0].len();

        // Compute feature maps
        let q_features: Vec<Vec<f64>> = query.iter().map(|q| self.feature_map(q)).collect();
        let k_features: Vec<Vec<f64>> = key.iter().map(|k| self.feature_map(k)).collect();

        // Compute KV = sum_j k_j * v_j^T
        let mut kv = vec![vec![0.0; d_v]; self.n_features];
        for j in 0..seq_len {
            for f in 0..self.n_features {
                for v in 0..d_v {
                    kv[f][v] += k_features[j][f] * value[j][v];
                }
            }
        }

        // Compute denominator: sum_j phi(k_j)^T * phi(q_i) for each i
        let mut denominators = vec![0.0; seq_len];
        for i in 0..seq_len {
            denominators[i] = k_features.iter().map(|k| {
                q_features[i].iter().zip(k.iter()).map(|(q, k)| q * k).sum::<f64>()
            }).sum();
        }

        // Compute output: phi(q_i)^T * KV / denominator
        let mut output = vec![vec![0.0; d_v]; seq_len];
        for i in 0..seq_len {
            for f in 0..self.n_features {
                for v in 0..d_v {
                    output[i][v] += q_features[i][f] * kv[f][v];
                }
            }
            if denominators[i] > 1e-10 {
                for v in 0..d_v {
                    output[i][v] /= denominators[i];
                }
            }
        }

        output
    }
}

/// Flash Attention (simplified, tiled computation).
pub fn flash_attention(
    query: &[Vec<f64>],
    key: &[Vec<f64>],
    value: &[Vec<f64>],
    block_size: usize,
) -> Vec<Vec<f64>> {
    let seq_len = query.len();
    let d = query[0].len();
    let scale = 1.0 / (d as f64).sqrt();

    let mut output = vec![vec![0.0; d]; seq_len];

    // Process in blocks to simulate memory efficiency
    for i_block in (0..seq_len).step_by(block_size) {
        let i_end = (i_block + block_size).min(seq_len);

        let mut o_block = vec![vec![0.0; d]; i_end - i_block];
        let mut m_block = vec![f64::NEG_INFINITY; i_end - i_block];
        let mut l_block = vec![0.0; i_end - i_block];

        for j_block in (0..seq_len).step_by(block_size) {
            let j_end = (j_block + block_size).min(seq_len);

            // Compute block attention scores
            for (ii, i) in (i_block..i_end).enumerate() {
                for j in j_block..j_end {
                    let score: f64 = query[i].iter().zip(key[j].iter())
                        .map(|(q, k)| q * k).sum::<f64>() * scale;

                    let m_new = m_block[ii].max(score);
                    let exp_score = (score - m_new).exp();
                    let exp_old = (m_block[ii] - m_new).exp();

                    l_block[ii] = l_block[ii] * exp_old + exp_score;
                    m_block[ii] = m_new;

                    for k in 0..d {
                        o_block[ii][k] = o_block[ii][k] * exp_old + exp_score * value[j][k];
                    }
                }
            }
        }

        // Normalize
        for (ii, i) in (i_block..i_end).enumerate() {
            for k in 0..d {
                output[i][k] = o_block[ii][k] / l_block[ii].max(1e-10);
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_dot_product() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let output = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_causal_attention() {
        let features = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let output = causal_self_attention(&features);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_flash_attention() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![0.5, 0.5]];
        let k = q.clone();
        let v = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]];

        let output = flash_attention(&q, &k, &v, 2);
        assert_eq!(output.len(), 4);
        assert_eq!(output[0].len(), 2);
    }
}
