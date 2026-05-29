/// MCMC samplers: Hamiltonian Monte Carlo, NUTS, slice sampling.

/// Hamiltonian Monte Carlo sampler.
pub struct HMC {
    pub step_size: f64,
    pub n_leapfrog: usize,
    seed: u64,
}

impl HMC {
    pub fn new(step_size: f64, n_leapfrog: usize) -> Self {
        Self { step_size, n_leapfrog, seed: 42 }
    }

    pub fn sample<F, G>(&mut self, initial: &[f64], log_prob: F, grad_log_prob: G, n_samples: usize) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let dim = initial.len();
        let mut current = initial.to_vec();
        let mut samples = Vec::new();

        for _ in 0..n_samples {
            // Sample momentum
            let momentum: Vec<f64> = (0..dim).map(|_| self.gaussian()).collect();

            // Leapfrog integration
            let mut q = current.clone();
            let mut p = momentum.clone();

            // Half step for momentum
            let grad = grad_log_prob(&q);
            for i in 0..dim {
                p[i] += 0.5 * self.step_size * grad[i];
            }

            for _ in 0..self.n_leapfrog - 1 {
                // Full step for position
                for i in 0..dim {
                    q[i] += self.step_size * p[i];
                }
                // Full step for momentum
                let grad = grad_log_prob(&q);
                for i in 0..dim {
                    p[i] += self.step_size * grad[i];
                }
            }

            // Half step for momentum
            let grad = grad_log_prob(&q);
            for i in 0..dim {
                p[i] += 0.5 * self.step_size * grad[i];
            }

            // Negate momentum
            for p_i in &mut p { *p_i = -*p_i; }

            // Metropolis accept/reject
            let current_h = -log_prob(&current) + 0.5 * momentum.iter().map(|p| p * p).sum::<f64>();
            let proposed_h = -log_prob(&q) + 0.5 * p.iter().map(|p| p * p).sum::<f64>();

            let log_alpha = -proposed_h + current_h;
            if self.pseudo_rand() < log_alpha.exp() {
                current = q;
            }

            samples.push(current.clone());
        }

        samples
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.pseudo_rand().max(1e-10);
        let u2 = self.pseudo_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Slice sampler.
pub struct SliceSampler {
    pub width: f64,
    pub max_iter: usize,
    seed: u64,
}

impl SliceSampler {
    pub fn new(width: f64, max_iter: usize) -> Self {
        Self { width, max_iter, seed: 42 }
    }

    pub fn sample_1d<F>(&mut self, initial: f64, log_prob: F, n_samples: usize) -> Vec<f64>
    where
        F: Fn(f64) -> f64,
    {
        let mut current = initial;
        let mut samples = Vec::new();

        for _ in 0..n_samples {
            // Sample vertical level
            let u = self.pseudo_rand();
            let log_y = log_prob(current) + u.ln();

            // Find interval
            let mut left = current - self.width * self.pseudo_rand();
            let mut right = left + self.width;

            while log_prob(left) > log_y {
                left -= self.width;
            }
            while log_prob(right) > log_y {
                right += self.width;
            }

            // Slice sample
            for _ in 0..self.max_iter {
                let proposal = left + self.pseudo_rand() * (right - left);
                if log_prob(proposal) > log_y {
                    current = proposal;
                    break;
                }
                if proposal < current {
                    left = proposal;
                } else {
                    right = proposal;
                }
            }

            samples.push(current);
        }

        samples
    }

    pub fn sample_nd<F>(&mut self, initial: &[f64], log_prob: F, n_samples: usize) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        let dim = initial.len();
        let mut current = initial.to_vec();
        let mut samples = Vec::new();

        for _ in 0..n_samples {
            for d in 0..dim {
                let x_d = current[d];
                let log_prob_d = |v: f64| {
                    let mut x = current.clone();
                    x[d] = v;
                    log_prob(&x)
                };

                let u = self.pseudo_rand();
                let log_y = log_prob_d(x_d) + u.ln();

                let mut left = x_d - self.width * self.pseudo_rand();
                let mut right = left + self.width;

                while log_prob_d(left) > log_y { left -= self.width; }
                while log_prob_d(right) > log_y { right += self.width; }

                for _ in 0..self.max_iter {
                    let proposal = left + self.pseudo_rand() * (right - left);
                    if log_prob_d(proposal) > log_y {
                        current[d] = proposal;
                        break;
                    }
                    if proposal < x_d {
                        left = proposal;
                    } else {
                        right = proposal;
                    }
                }
            }
            samples.push(current.clone());
        }

        samples
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// No-U-Turn Sampler (NUTS) - simplified version.
pub struct NUTS {
    pub step_size: f64,
    pub max_depth: usize,
    seed: u64,
}

impl NUTS {
    pub fn new(step_size: f64, max_depth: usize) -> Self {
        Self { step_size, max_depth, seed: 42 }
    }

    pub fn sample<F, G>(&mut self, initial: &[f64], log_prob: F, grad_log_prob: G, n_samples: usize) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let dim = initial.len();
        let mut current = initial.to_vec();
        let mut samples = Vec::new();

        for _ in 0..n_samples {
            // Sample momentum
            let momentum: Vec<f64> = (0..dim).map(|_| self.gaussian()).collect();

            // Slice variable
            let u = self.pseudo_rand();
            let log_probCurrent = -log_prob(&current);
            let log_slice = log_probCurrent - 0.5 * momentum.iter().map(|p| p * p).sum::<f64>() + u.ln();

            let mut q_minus = current.clone();
            let mut q_plus = current.clone();
            let mut p_minus = momentum.clone();
            let mut p_plus = momentum.clone();

            let mut j = 0;
            let mut n = 1;
            let mut s = true;
            let mut q_new = current.clone();

            while s && j < self.max_depth {
                // Choose direction
                let direction = if self.pseudo_rand() < 0.5 { -1 } else { 1 };

                let (q_candidate, p_candidate, n_candidate, s_candidate) = if direction == -1 {
                    self.build_tree(&q_minus, &p_minus, log_slice, direction, j, &log_prob, &grad_log_prob)
                } else {
                    self.build_tree(&q_plus, &p_plus, log_slice, direction, j, &log_prob, &grad_log_prob)
                };

                if s_candidate {
                    let accept_prob = n_candidate as f64 / n as f64;
                    if self.pseudo_rand() < accept_prob {
                        q_new = q_candidate;
                    }
                }

                n += n_candidate;
                s = s_candidate && self.no_u_turn(&q_minus, &q_plus, &p_minus, &p_plus);

                if direction == -1 {
                    q_minus = q_candidate;
                    p_minus = p_candidate;
                } else {
                    q_plus = q_candidate;
                    p_plus = p_candidate;
                }

                j += 1;
            }

            current = q_new;
            samples.push(current.clone());
        }

        samples
    }

    fn build_tree<F, G>(&mut self, q: &[f64], p: &[f64], log_slice: f64, direction: i32, depth: usize,
                         log_prob: &F, grad_log_prob: &G) -> (Vec<f64>, Vec<f64>, usize, bool)
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let dim = q.len();

        if depth == 0 {
            // Base case: single leapfrog step
            let mut q_new = q.to_vec();
            let mut p_new = p.to_vec();

            let grad = grad_log_prob(&q_new);
            for i in 0..dim {
                p_new[i] += 0.5 * self.step_size * direction as f64 * grad[i];
                q_new[i] += self.step_size * direction as f64 * p_new[i];
            }
            let grad = grad_log_prob(&q_new);
            for i in 0..dim {
                p_new[i] += 0.5 * self.step_size * direction as f64 * grad[i];
            }

            let log_prob_new = -log_prob(&q_new);
            let log_hamiltonian = log_prob_new - 0.5 * p_new.iter().map(|p| p * p).sum::<f64>();
            let n = if log_hamiltonian > log_slice { 1 } else { 0 };
            let s = log_hamiltonian > log_slice - 1000.0;

            return (q_new, p_new, n, s);
        }

        // Recursive case
        let (q_minus, p_minus, n_minus, s_minus) = self.build_tree(q, p, log_slice, direction, depth - 1, log_prob, grad_log_prob);
        let (q_plus, p_plus, n_plus, s_plus) = self.build_tree(q, p, log_slice, direction, depth - 1, log_prob, grad_log_prob);

        let (q_new, n_new, s_new) = if s_minus && s_plus {
            let accept = if n_minus + n_plus > 0 {
                self.pseudo_rand() < n_minus as f64 / (n_minus + n_plus) as f64
            } else {
                false
            };
            if accept { (q_minus, n_minus + n_plus, s_minus && s_plus && self.no_u_turn(&q_minus, &q_plus, &p_minus, &p_plus)) }
            else { (q_plus, n_minus + n_plus, s_minus && s_plus && self.no_u_turn(&q_minus, &q_plus, &p_minus, &p_plus)) }
        } else if s_minus {
            (q_minus, n_minus, s_minus)
        } else {
            (q_plus, n_plus, s_plus)
        };

        (q_new, p_minus, n_new, s_new)
    }

    fn no_u_turn(&self, q_minus: &[f64], q_plus: &[f64], p_minus: &[f64], p_plus: &[f64]) -> bool {
        let dq: Vec<f64> = q_plus.iter().zip(q_minus.iter()).map(|(a, b)| a - b).collect();
        let dot_minus: f64 = dq.iter().zip(p_minus.iter()).map(|(d, p)| d * p).sum();
        let dot_plus: f64 = dq.iter().zip(p_plus.iter()).map(|(d, p)| d * p).sum();
        dot_minus >= 0.0 && dot_plus >= 0.0
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.pseudo_rand().max(1e-10);
        let u2 = self.pseudo_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Elliptical Slice Sampler.
pub struct EllipticalSlice {
    seed: u64,
}

impl EllipticalSlice {
    pub fn new() -> Self {
        Self { seed: 42 }
    }

    pub fn sample<F>(&mut self, initial: &[f64], prior_sample: &[f64], log_likelihood: F, n_samples: usize) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        let dim = initial.len();
        let mut current = initial.to_vec();
        let mut samples = Vec::new();

        for _ in 0..n_samples {
            // Sample from prior
            let nu: Vec<f64> = (0..dim).map(|i| {
                self.gaussian() * prior_sample[i].abs().max(1.0)
            }).collect();

            let u = self.pseudo_rand();
            let log_y = log_likelihood(&current) + u.ln();

            let mut theta = self.pseudo_rand() * 2.0 * std::f64::consts::PI;
            let theta_min = theta - 2.0 * std::f64::consts::PI;
            let theta_max = theta;

            loop {
                let proposal: Vec<f64> = current.iter().zip(nu.iter()).map(|(c, n)| {
                    c * theta.cos() + n * theta.sin()
                }).collect();

                if log_likelihood(&proposal) > log_y {
                    current = proposal;
                    break;
                }

                if theta < 0.0 {
                    theta_min = theta;
                } else {
                    theta_max = theta;
                }

                theta = theta_min + self.pseudo_rand() * (theta_max - theta_min);
            }

            samples.push(current.clone());
        }

        samples
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.pseudo_rand().max(1e-10);
        let u2 = self.pseudo_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Diagnostic: effective sample size.
pub fn effective_sample_size(samples: &[Vec<f64>]) -> Vec<f64> {
    let n_samples = samples.len();
    if n_samples < 2 { return vec![0.0; samples[0].len()]; }

    let dim = samples[0].len();
    let max_lag = n_samples / 4;

    (0..dim).map(|d| {
        let values: Vec<f64> = samples.iter().map(|s| s[d]).collect();
        let mean = values.iter().sum::<f64>() / n_samples as f64;
        let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;

        if var < 1e-15 { return n_samples as f64; }

        let mut sum_rho = 1.0;
        for lag in 1..=max_lag {
            let cov: f64 = (0..n_samples - lag).map(|i| {
                (values[i] - mean) * (values[i + lag] - mean)
            }).sum::<f64>() / n_samples as f64;
            let rho = cov / var;
            if rho < 0.05 { break; }
            sum_rho += 2.0 * rho;
        }

        n_samples as f64 / sum_rho.max(1.0)
    }).collect()
}

/// R-hat convergence diagnostic.
pub fn r_hat(chains: &[Vec<Vec<f64>>]) -> Vec<f64> {
    let n_chains = chains.len();
    let n_samples = chains[0].len();
    let dim = chains[0][0].len();

    (0..dim).map(|d| {
        let chain_means: Vec<f64> = chains.iter().map(|chain| {
            chain.iter().map(|s| s[d]).sum::<f64>() / n_samples as f64
        }).collect();

        let chain_vars: Vec<f64> = chains.iter().zip(chain_means.iter()).map(|(chain, &mean)| {
            chain.iter().map(|s| (s[d] - mean).powi(2)).sum::<f64>() / (n_samples - 1) as f64
        }).collect();

        let overall_mean = chain_means.iter().sum::<f64>() / n_chains as f64;

        let b = n_samples as f64 * chain_means.iter().map(|m| (m - overall_mean).powi(2)).sum::<f64>() / (n_chains - 1) as f64;
        let w = chain_vars.iter().sum::<f64>() / n_chains as f64;

        let var_plus = (n_samples - 1) as f64 / n_samples as f64 * w + b / n_samples as f64;
        (var_plus / w).sqrt()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_sampler() {
        let mut sampler = SliceSampler::new(1.0, 100);
        let log_prob = |x: f64| -0.5 * x * x; // Standard normal
        let samples = sampler.sample_1d(0.0, log_prob, 1000);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.5);
    }

    #[test]
    fn test_hmc() {
        let mut hmc = HMC::new(0.1, 10);
        let log_prob = |x: &[f64]| -0.5 * x.iter().map(|xi| xi * xi).sum::<f64>();
        let grad = |x: &[f64]| x.iter().map(|&xi| -xi).collect();
        let samples = hmc.sample(&[0.0, 0.0], log_prob, grad, 100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_ess() {
        let samples: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64 * 0.01, (i as f64 * 0.01).sin()]).collect();
        let ess = effective_sample_size(&samples);
        assert_eq!(ess.len(), 2);
        assert!(ess[0] > 0.0);
    }
}
