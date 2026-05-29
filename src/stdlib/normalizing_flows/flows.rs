/// Normalizing flows: planar, radial, affine coupling, real NVP.

/// Planar flow: f(z) = z + u * tanh(w^T z + b).
pub struct PlanarFlow {
    pub w: Vec<f64>,
    pub u: Vec<f64>,
    pub b: f64,
}

impl PlanarFlow {
    pub fn new(dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.1 - 0.05
        };

        Self {
            w: (0..dim).map(|_| rand()).collect(),
            u: (0..dim).map(|_| rand()).collect(),
            b: rand(),
        }
    }

    pub fn forward(&self, z: &[f64]) -> Vec<f64> {
        let w_dot_z: f64 = self.w.iter().zip(z.iter()).map(|(w, z)| w * z).sum();
        let activation = (w_dot_z + self.b).tanh();

        z.iter().zip(self.u.iter()).map(|(zi, ui)| zi + ui * activation).collect()
    }

    pub fn log_det_jacobian(&self, z: &[f64]) -> f64 {
        let w_dot_z: f64 = self.w.iter().zip(z.iter()).map(|(w, z)| w * z).sum();
        let tanh_val = (w_dot_z + self.b).tanh();
        let dtanh = 1.0 - tanh_val * tanh_val;

        let w_dot_u: f64 = self.w.iter().zip(self.u.iter()).map(|(w, u)| w * u).sum();
        (1.0 + w_dot_u * dtanh).abs().ln()
    }
}

/// Radial flow: f(z) = z + beta * (z - z0) / (alpha + ||z - z0||).
pub struct RadialFlow {
    pub z0: Vec<f64>,
    pub alpha: f64,
    pub beta: f64,
}

impl RadialFlow {
    pub fn new(z0: Vec<f64>) -> Self {
        Self { z0, alpha: 1.0, beta: 0.5 }
    }

    pub fn forward(&self, z: &[f64]) -> Vec<f64> {
        let dist: f64 = z.iter().zip(self.z0.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let scale = self.beta / (self.alpha + dist);

        z.iter().zip(self.z0.iter()).map(|(zi, z0i)| zi + scale * (zi - z0i)).collect()
    }

    pub fn log_det_jacobian(&self, z: &[f64]) -> f64 {
        let dim = z.len();
        let dist: f64 = z.iter().zip(self.z0.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let r = self.alpha + dist;

        ((1.0 + self.beta / r).powi(dim as i32 - 1) * (1.0 + self.beta / r + self.beta * dist / (r * r))).abs().ln()
    }
}

/// Affine coupling layer.
pub struct AffineCoupling {
    pub dim: usize,
    pub mask: Vec<bool>,
    pub scale_weights: Vec<Vec<f64>>,
    pub scale_bias: Vec<f64>,
    pub translate_weights: Vec<Vec<f64>>,
    pub translate_bias: Vec<f64>,
}

impl AffineCoupling {
    pub fn new(dim: usize, mask: Vec<bool>, hidden_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / dim as f64).sqrt();

        Self {
            dim, mask,
            scale_weights: (0..dim).map(|_| (0..dim).map(|_| rand(scale)).collect()).collect(),
            scale_bias: vec![0.0; dim],
            translate_weights: (0..dim).map(|_| (0..dim).map(|_| rand(scale)).collect()).collect(),
            translate_bias: vec![0.0; dim],
        }
    }

    pub fn forward(&self, z: &[f64]) -> Vec<f64> {
        // Compute scale and translation from masked dimensions
        let mut log_scale = vec![0.0; self.dim];
        let mut translate = vec![0.0; self.dim];

        for i in 0..self.dim {
            if !self.mask[i] {
                let mut s = self.scale_bias[i];
                let mut t = self.translate_bias[i];
                for j in 0..self.dim {
                    if self.mask[j] {
                        s += self.scale_weights[i][j] * z[j];
                        t += self.translate_weights[i][j] * z[j];
                    }
                }
                log_scale[i] = s;
                translate[i] = t;
            }
        }

        // Apply transformation
        z.iter().enumerate().map(|(i, &zi)| {
            if self.mask[i] {
                zi
            } else {
                zi * log_scale[i].exp() + translate[i]
            }
        }).collect()
    }

    pub fn log_det_jacobian(&self, z: &[f64]) -> f64 {
        let mut log_det = 0.0;

        for i in 0..self.dim {
            if !self.mask[i] {
                let mut s = self.scale_bias[i];
                for j in 0..self.dim {
                    if self.mask[j] {
                        s += self.scale_weights[i][j] * z[j];
                    }
                }
                log_det += s;
            }
        }

        log_det
    }
}

/// Real NVP (Non-Volume Preserving) - stack of affine coupling layers.
pub struct RealNVP {
    pub layers: Vec<AffineCoupling>,
}

impl RealNVP {
    pub fn new(dim: usize, n_layers: usize, hidden_dim: usize) -> Self {
        let mut layers = Vec::new();

        for i in 0..n_layers {
            let mask: Vec<bool> = (0..dim).map(|j| (j + i) % 2 == 0).collect();
            layers.push(AffineCoupling::new(dim, mask, hidden_dim));
        }

        Self { layers }
    }

    pub fn forward(&self, z: &[f64]) -> Vec<f64> {
        let mut result = z.to_vec();
        for layer in &self.layers {
            result = layer.forward(&result);
        }
        result
    }

    pub fn log_det_jacobian(&self, z: &[f64]) -> f64 {
        let mut total = 0.0;
        let mut current = z.to_vec();
        for layer in &self.layers {
            total += layer.log_det_jacobian(&current);
            current = layer.forward(&current);
        }
        total
    }
}

/// MADE (Masked Autoencoder for Distribution Estimation).
pub struct MADE {
    pub dim: usize,
    pub hidden_dim: usize,
    pub weights1: Vec<Vec<f64>>,
    pub bias1: Vec<f64>,
    pub weights2_mu: Vec<Vec<f64>>,
    pub weights2_sigma: Vec<Vec<f64>>,
    pub bias2_mu: Vec<f64>,
    pub bias2_sigma: Vec<f64>,
    pub mask1: Vec<Vec<f64>>,
    pub mask2_mu: Vec<Vec<f64>>,
    pub mask2_sigma: Vec<Vec<f64>>,
}

impl MADE {
    pub fn new(dim: usize, hidden_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale1 = (2.0 / dim as f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();

        // Create masks
        let input_degrees: Vec<usize> = (0..dim).collect();
        let hidden_degrees: Vec<usize> = (0..hidden_dim).map(|i| (i % (dim - 1)) + 1).collect();

        let mask1: Vec<Vec<f64>> = (0..hidden_dim).map(|h| {
            (0..dim).map(|d| if input_degrees[d] <= hidden_degrees[h] { 1.0 } else { 0.0 }).collect()
        }).collect();

        let mask2_mu: Vec<Vec<f64>> = (0..dim).map(|d| {
            (0..hidden_dim).map(|h| if hidden_degrees[h] < input_degrees[d] { 1.0 } else { 0.0 }).collect()
        }).collect();

        let mask2_sigma = mask2_mu.clone();

        Self {
            dim, hidden_dim,
            weights1: (0..hidden_dim).map(|_| (0..dim).map(|_| rand(scale1)).collect()).collect(),
            bias1: vec![0.0; hidden_dim],
            weights2_mu: (0..dim).map(|_| (0..hidden_dim).map(|_| rand(scale2)).collect()).collect(),
            weights2_sigma: (0..dim).map(|_| (0..hidden_dim).map(|_| rand(scale2)).collect()).collect(),
            bias2_mu: vec![0.0; dim],
            bias2_sigma: vec![0.0; dim],
            mask1, mask2_mu, mask2_sigma,
        }
    }

    pub fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Hidden layer with mask
        let h: Vec<f64> = (0..self.hidden_dim).map(|i| {
            let mut sum = self.bias1[i];
            for j in 0..self.dim {
                sum += self.weights1[i][j] * self.mask1[i][j] * x[j];
            }
            sum.max(0.0) // ReLU
        }).collect();

        // Output layers
        let mu: Vec<f64> = (0..self.dim).map(|i| {
            let mut sum = self.bias2_mu[i];
            for j in 0..self.hidden_dim {
                sum += self.weights2_mu[i][j] * self.mask2_mu[i][j] * h[j];
            }
            sum
        }).collect();

        let log_sigma: Vec<f64> = (0..self.dim).map(|i| {
            let mut sum = self.bias2_sigma[i];
            for j in 0..self.hidden_dim {
                sum += self.weights2_sigma[i][j] * self.mask2_sigma[i][j] * h[j];
            }
            sum.max(-10.0).min(10.0)
        }).collect();

        (mu, log_sigma)
    }

    pub fn log_prob(&self, x: &[f64]) -> f64 {
        let (mu, log_sigma) = self.forward(x);
        let sigma: Vec<f64> = log_sigma.iter().map(|&ls| ls.exp()).collect();

        -0.5 * x.iter().zip(mu.iter()).zip(sigma.iter())
            .map(|((&xi, &mi), &si)| ((xi - mi) / si).powi(2) + 2.0 * si.ln())
            .sum::<f64>()
    }
}

/// Autoregressive flow using MADE.
pub struct AutoregressiveFlow {
    pub made: MADE,
    pub dim: usize,
}

impl AutoregressiveFlow {
    pub fn new(dim: usize, hidden_dim: usize) -> Self {
        Self {
            made: MADE::new(dim, hidden_dim),
            dim,
        }
    }

    pub fn forward(&self, z: &[f64]) -> Vec<f64> {
        let (mu, log_sigma) = self.made.forward(z);
        z.iter().zip(mu.iter()).zip(log_sigma.iter())
            .map(|((&zi, &mi), &lsi)| mi + zi * lsi.exp())
            .collect()
    }

    pub fn log_det_jacobian(&self, z: &[f64]) -> f64 {
        let (_, log_sigma) = self.made.forward(z);
        log_sigma.iter().sum()
    }
}

/// Inverse autoregressive flow.
pub struct InverseAutoregressiveFlow {
    pub made: MADE,
    pub dim: usize,
}

impl InverseAutoregressiveFlow {
    pub fn new(dim: usize, hidden_dim: usize) -> Self {
        Self {
            made: MADE::new(dim, hidden_dim),
            dim,
        }
    }

    pub fn forward(&self, z: &[f64]) -> Vec<f64> {
        let (mu, log_sigma) = self.made.forward(z);
        let sigma: Vec<f64> = log_sigma.iter().map(|&ls| ls.exp()).collect();

        // x = sigma * z + (1 - sigma) * mu (simplified)
        z.iter().zip(mu.iter()).zip(sigma.iter())
            .map(|((&zi, &mi), &si)| si * zi + (1.0 - si) * mi)
            .collect()
    }

    pub fn log_det_jacobian(&self, z: &[f64]) -> f64 {
        let (_, log_sigma) = self.made.forward(z);
        -log_sigma.iter().sum::<f64>()
    }
}

/// Flow model: chain of flows.
pub struct FlowModel {
    pub flows: Vec<Box<dyn Flow>>,
}

pub trait Flow {
    fn forward(&self, z: &[f64]) -> Vec<f64>;
    fn log_det_jacobian(&self, z: &[f64]) -> f64;
}

impl Flow for PlanarFlow {
    fn forward(&self, z: &[f64]) -> Vec<f64> { self.forward(z) }
    fn log_det_jacobian(&self, z: &[f64]) -> f64 { self.log_det_jacobian(z) }
}

impl Flow for RadialFlow {
    fn forward(&self, z: &[f64]) -> Vec<f64> { self.forward(z) }
    fn log_det_jacobian(&self, z: &[f64]) -> f64 { self.log_det_jacobian(z) }
}

impl FlowModel {
    pub fn new() -> Self {
        Self { flows: Vec::new() }
    }

    pub fn add_flow(&mut self, flow: Box<dyn Flow>) {
        self.flows.push(flow);
    }

    pub fn forward(&self, z: &[f64]) -> Vec<f64> {
        let mut result = z.to_vec();
        for flow in &self.flows {
            result = flow.forward(&result);
        }
        result
    }

    pub fn total_log_det_jacobian(&self, z: &[f64]) -> f64 {
        let mut total = 0.0;
        let mut current = z.to_vec();
        for flow in &self.flows {
            total += flow.log_det_jacobian(&current);
            current = flow.forward(&current);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planar_flow() {
        let flow = PlanarFlow::new(2);
        let z = vec![0.5, -0.5];
        let result = flow.forward(&z);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_affine_coupling() {
        let mask = vec![true, false, true, false];
        let flow = AffineCoupling::new(4, mask, 8);
        let z = vec![1.0, 2.0, 3.0, 4.0];
        let result = flow.forward(&z);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 1.0); // Masked dimensions unchanged
        assert_eq!(result[2], 3.0);
    }

    #[test]
    fn test_real_nvp() {
        let nvp = RealNVP::new(4, 3, 8);
        let z = vec![1.0, 2.0, 3.0, 4.0];
        let result = nvp.forward(&z);
        assert_eq!(result.len(), 4);
    }
}
