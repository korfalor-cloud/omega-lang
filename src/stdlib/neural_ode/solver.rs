/// Neural ODE: ODE solvers integrated with neural networks.

/// Neural ODE layer: learns the dynamics function.
pub struct NeuralODE {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub weights_ih: Vec<Vec<f64>>,
    pub weights_hh: Vec<Vec<f64>>,
    pub weights_ho: Vec<Vec<f64>>,
    pub bias_h: Vec<f64>,
    pub bias_o: Vec<f64>,
    pub dt: f64,
    pub n_steps: usize,
}

impl NeuralODE {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale_ih = (1.0 / input_dim as f64).sqrt();
        let scale_hh = (1.0 / hidden_dim as f64).sqrt();
        let scale_ho = (1.0 / hidden_dim as f64).sqrt();

        Self {
            input_dim, hidden_dim, output_dim,
            weights_ih: (0..hidden_dim).map(|_| (0..input_dim).map(|_| rand(scale_ih)).collect()).collect(),
            weights_hh: (0..hidden_dim).map(|_| (0..hidden_dim).map(|_| rand(scale_hh)).collect()).collect(),
            weights_ho: (0..output_dim).map(|_| (0..hidden_dim).map(|_| rand(scale_ho)).collect()).collect(),
            bias_h: vec![0.0; hidden_dim],
            bias_o: vec![0.0; output_dim],
            dt: 0.1,
            n_steps: 10,
        }
    }

    /// Forward pass: ODE integration using Euler method.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut h = vec![0.0; self.hidden_dim];

        for _ in 0..self.n_steps {
            let dh = self.dynamics(input, &h);
            for i in 0..self.hidden_dim {
                h[i] += dh[i] * self.dt;
            }
        }

        // Output layer
        self.weights_ho.iter().zip(self.bias_o.iter()).map(|(wo, &bo)| {
            wo.iter().zip(h.iter()).map(|(w, hi)| w * hi).sum::<f64>() + bo
        }).collect()
    }

    /// Dynamics function: dh/dt = f(t, h, input).
    fn dynamics(&self, input: &[f64], h: &[f64]) -> Vec<f64> {
        let mut dh = vec![0.0; self.hidden_dim];

        for i in 0..self.hidden_dim {
            let mut sum = self.bias_h[i];
            for j in 0..self.input_dim {
                sum += self.weights_ih[i][j] * input[j];
            }
            for j in 0..self.hidden_dim {
                sum += self.weights_hh[i][j] * h[j];
            }
            dh[i] = sum.tanh();
        }

        dh
    }

    /// Forward with RK4 integration.
    pub fn forward_rk4(&self, input: &[f64]) -> Vec<f64> {
        let mut h = vec![0.0; self.hidden_dim];

        for _ in 0..self.n_steps {
            let k1 = self.dynamics(input, &h);
            let h_k1: Vec<f64> = h.iter().zip(k1.iter()).map(|(hi, k)| hi + k * self.dt / 2.0).collect();
            let k2 = self.dynamics(input, &h_k1);
            let h_k2: Vec<f64> = h.iter().zip(k2.iter()).map(|(hi, k)| hi + k * self.dt / 2.0).collect();
            let k3 = self.dynamics(input, &h_k2);
            let h_k3: Vec<f64> = h.iter().zip(k3.iter()).map(|(hi, k)| hi + k * self.dt).collect();
            let k4 = self.dynamics(input, &h_k3);

            for i in 0..self.hidden_dim {
                h[i] += self.dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
        }

        self.weights_ho.iter().zip(self.bias_o.iter()).map(|(wo, &bo)| {
            wo.iter().zip(h.iter()).map(|(w, hi)| w * hi).sum::<f64>() + bo
        }).collect()
    }

    /// Adjoint method for memory-efficient backpropagation.
    pub fn adjoint_method(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_steps;
        let dt = self.dt;

        // Forward pass: store states
        let mut states = vec![vec![0.0; self.hidden_dim]; n + 1];
        for i in 0..n {
            let dh = self.dynamics(input, &states[i]);
            for j in 0..self.hidden_dim {
                states[i + 1][j] = states[i][j] + dh[j] * dt;
            }
        }

        // Compute output
        let output: Vec<f64> = self.weights_ho.iter().zip(self.bias_o.iter()).map(|(wo, &bo)| {
            wo.iter().zip(states[n].iter()).map(|(w, hi)| w * hi).sum::<f64>() + bo
        }).collect();

        // Backward pass: adjoint dynamics
        let mut adjoint = vec![0.0; self.hidden_dim];
        // Initialize adjoint from output gradient (simplified: use output as gradient)
        for i in 0..self.output_dim {
            for j in 0..self.hidden_dim {
                adjoint[j] += self.weights_ho[i][j] * output[i];
            }
        }

        // Integrate adjoint backward
        for i in (0..n).rev() {
            let d_adj = self.adjoint_dynamics(input, &states[i], &adjoint);
            for j in 0..self.hidden_dim {
                adjoint[j] -= d_adj[j] * dt;
            }
        }

        (output, adjoint)
    }

    fn adjoint_dynamics(&self, input: &[f64], state: &[f64], adjoint: &[f64]) -> Vec<f64> {
        // Simplified adjoint dynamics
        let dh = self.dynamics(input, state);
        let mut d_adj = vec![0.0; self.hidden_dim];

        for i in 0..self.hidden_dim {
            let tanh_deriv = 1.0 - dh[i] * dh[i];
            for j in 0..self.hidden_dim {
                d_adj[i] += adjoint[j] * self.weights_hh[j][i] * tanh_deriv;
            }
        }

        d_adj
    }
}

/// Continuous normalizing flows.
pub struct ContinuousNormalizingFlow {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
    pub dt: f64,
    pub n_steps: usize,
}

impl ContinuousNormalizingFlow {
    pub fn new(dim: usize, hidden_dim: usize, n_layers: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for layer in 0..n_layers {
            let in_dim = if layer == 0 { dim } else { hidden_dim };
            let out_dim = if layer == n_layers - 1 { dim } else { hidden_dim };
            let scale = (1.0 / in_dim as f64).sqrt();

            weights.push((0..out_dim).map(|_| (0..in_dim).map(|_| rand(scale)).collect()).collect());
            biases.push(vec![0.0; out_dim]);
        }

        Self { dim, hidden_dim, n_layers, weights, biases, dt: 0.1, n_steps: 10 }
    }

    /// Forward: transform z to x.
    pub fn forward(&self, z: &[f64]) -> Vec<f64> {
        let mut x = z.to_vec();

        for _ in 0..self.n_steps {
            let dx = self.network(&x);
            for i in 0..self.dim {
                x[i] += dx[i] * self.dt;
            }
        }

        x
    }

    /// Compute trace of Jacobian (for log-density computation).
    pub fn trace_jacobian(&self, z: &[f64]) -> f64 {
        let eps = 1e-5;
        let mut trace = 0.0;

        for i in 0..self.dim {
            let mut z_plus = z.to_vec();
            let mut z_minus = z.to_vec();
            z_plus[i] += eps;
            z_minus[i] -= eps;

            let f_plus = self.network(&z_plus);
            let f_minus = self.network(&z_minus);

            trace += (f_plus[i] - f_minus[i]) / (2.0 * eps);
        }

        trace
    }

    fn network(&self, x: &[f64]) -> Vec<f64> {
        let mut h = x.to_vec();

        for layer in 0..self.n_layers {
            let mut new_h = Vec::new();
            for i in 0..self.weights[layer].len() {
                let mut sum = self.biases[layer][i];
                for j in 0..h.len() {
                    sum += self.weights[layer][i][j] * h[j];
                }
                new_h.push(if layer < self.n_layers - 1 { sum.tanh() } else { sum });
            }
            h = new_h;
        }

        h
    }
}

/// Neural Controlled Differential Equations (Neural CDE).
pub struct NeuralCDE {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub dt: f64,
}

impl NeuralCDE {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.1 - 0.05
        };

        Self {
            input_dim,
            hidden_dim,
            weights: (0..hidden_dim).map(|_| (0..input_dim).map(|_| rand()).collect()).collect(),
            bias: vec![0.0; hidden_dim],
            dt: 0.1,
        }
    }

    /// Forward pass: solve CDE using Euler method.
    pub fn forward(&self, path: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut h = vec![0.0; self.hidden_dim];
        let mut outputs = vec![h.clone()];

        for t in 1..path.len() {
            let d_path: Vec<f64> = path[t].iter().zip(path[t - 1].iter()).map(|(a, b)| a - b).collect();
            let dh = self.dynamics(&h, &d_path);
            for i in 0..self.hidden_dim {
                h[i] += dh[i];
            }
            outputs.push(h.clone());
        }

        outputs
    }

    fn dynamics(&self, h: &[f64], dx: &[f64]) -> Vec<f64> {
        let mut dh = vec![0.0; self.hidden_dim];

        for i in 0..self.hidden_dim {
            let mut sum = self.bias[i];
            for j in 0..self.input_dim {
                sum += self.weights[i][j] * dx[j];
            }
            // Multiply by hidden state (controlled by input)
            dh[i] = sum * (1.0 + h[i].tanh());
        }

        dh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_ode() {
        let node = NeuralODE::new(2, 8, 2);
        let input = vec![1.0, 0.5];
        let output = node.forward(&input);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_neural_ode_rk4() {
        let node = NeuralODE::new(2, 8, 2);
        let input = vec![1.0, 0.5];
        let output = node.forward_rk4(&input);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_cnf() {
        let cnf = ContinuousNormalizingFlow::new(2, 8, 2);
        let z = vec![0.5, -0.5];
        let x = cnf.forward(&z);
        assert_eq!(x.len(), 2);
    }

    #[test]
    fn test_neural_cde() {
        let cde = NeuralCDE::new(2, 4);
        let path = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.2],
            vec![0.3, 0.5],
            vec![0.6, 0.8],
        ];
        let outputs = cde.forward(&path);
        assert_eq!(outputs.len(), 4);
        assert_eq!(outputs[0].len(), 4);
    }
}
