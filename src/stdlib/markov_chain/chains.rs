/// Markov chains and Hidden Markov Models.

use std::collections::HashMap;

/// Discrete-time Markov chain.
pub struct MarkovChain {
    pub num_states: usize,
    pub transition: Vec<Vec<f64>>,
    pub initial: Vec<f64>,
    seed: u64,
}

impl MarkovChain {
    pub fn new(transition: Vec<Vec<f64>>, initial: Vec<f64>) -> Self {
        let num_states = transition.len();
        Self { num_states, transition, initial, seed: 42 }
    }

    /// Stationary distribution (power iteration).
    pub fn stationary_distribution(&self, max_iter: usize, tolerance: f64) -> Vec<f64> {
        let mut pi = self.initial.clone();

        for _ in 0..max_iter {
            let mut new_pi = vec![0.0; self.num_states];
            for i in 0..self.num_states {
                for j in 0..self.num_states {
                    new_pi[j] += pi[i] * self.transition[i][j];
                }
            }

            let diff: f64 = pi.iter().zip(new_pi.iter()).map(|(a, b)| (a - b).abs()).sum();
            pi = new_pi;
            if diff < tolerance { break; }
        }

        pi
    }

    /// Simulate chain for n steps.
    pub fn simulate(&mut self, steps: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut state = self.sample_from(&self.initial);
        path.push(state);

        for _ in 0..steps {
            state = self.sample_from(&self.transition[state]);
            path.push(state);
        }

        path
    }

    /// n-step transition matrix.
    pub fn n_step_transition(&self, n: usize) -> Vec<Vec<f64>> {
        let mut result = self.identity();
        let mut base = self.transition.clone();

        let mut exp = n;
        while exp > 0 {
            if exp % 2 == 1 {
                result = self.matrix_mul(&result, &base);
            }
            base = self.matrix_mul(&base, &base);
            exp /= 2;
        }

        result
    }

    /// Expected first passage time from state i to state j.
    pub fn first_passage_time(&self, from: usize, to: usize, max_iter: usize) -> f64 {
        if from == to { return 0.0; }

        // Solve system of equations using iteration
        let mut times = vec![0.0; self.num_states];

        for _ in 0..max_iter {
            let mut new_times = vec![0.0; self.num_states];
            for i in 0..self.num_states {
                if i == to {
                    new_times[i] = 0.0;
                    continue;
                }
                new_times[i] = 1.0;
                for j in 0..self.num_states {
                    new_times[i] += self.transition[i][j] * times[j];
                }
            }
            times = new_times;
        }

        times[from]
    }

    /// Mean recurrence time for a state.
    pub fn mean_recurrence_time(&self, state: usize) -> f64 {
        let pi = self.stationary_distribution(1000, 1e-10);
        if pi[state] > 0.0 { 1.0 / pi[state] } else { f64::INFINITY }
    }

    /// Absorption probability (probability of reaching absorbing state j from i).
    pub fn absorption_probability(&self, from: usize, to: usize, max_iter: usize) -> f64 {
        let mut probs = vec![0.0; self.num_states];
        probs[to] = 1.0;

        for _ in 0..max_iter {
            let mut new_probs = probs.clone();
            for i in 0..self.num_states {
                if i == to { continue; }
                new_probs[i] = 0.0;
                for j in 0..self.num_states {
                    new_probs[i] += self.transition[i][j] * probs[j];
                }
            }
            probs = new_probs;
        }

        probs[from]
    }

    fn identity(&self) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; self.num_states]; self.num_states];
        for i in 0..self.num_states { m[i][i] = 1.0; }
        m
    }

    fn matrix_mul(&self, a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = self.num_states;
        let mut result = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        result
    }

    fn sample_from(&mut self, probs: &[f64]) -> usize {
        let r = self.pseudo_rand();
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum { return i; }
        }
        probs.len() - 1
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Hidden Markov Model.
pub struct HMM {
    pub num_states: usize,
    pub num_observations: usize,
    pub transition: Vec<Vec<f64>>,   // A[i][j] = P(state j | state i)
    pub emission: Vec<Vec<f64>>,     // B[i][o] = P(observation o | state i)
    pub initial: Vec<f64>,           // pi[i] = P(initial state i)
}

impl HMM {
    pub fn new(
        transition: Vec<Vec<f64>>,
        emission: Vec<Vec<f64>>,
        initial: Vec<f64>,
    ) -> Self {
        Self {
            num_states: transition.len(),
            num_observations: emission[0].len(),
            transition, emission, initial,
        }
    }

    /// Forward algorithm: P(observations | model).
    pub fn forward(&self, observations: &[usize]) -> (Vec<Vec<f64>>, f64) {
        let t = observations.len();
        let n = self.num_states;
        let mut alpha = vec![vec![0.0; n]; t];

        // Initialization
        for i in 0..n {
            alpha[0][i] = self.initial[i] * self.emission[i][observations[0]];
        }

        // Induction
        for t_idx in 1..t {
            for j in 0..n {
                alpha[t_idx][j] = (0..n)
                    .map(|i| alpha[t_idx - 1][i] * self.transition[i][j])
                    .sum::<f64>() * self.emission[j][observations[t_idx]];
            }
        }

        let prob = alpha[t - 1].iter().sum();
        (alpha, prob)
    }

    /// Backward algorithm.
    pub fn backward(&self, observations: &[usize]) -> Vec<Vec<f64>> {
        let t = observations.len();
        let n = self.num_states;
        let mut beta = vec![vec![0.0; n]; t];

        // Initialization
        for i in 0..n {
            beta[t - 1][i] = 1.0;
        }

        // Induction
        for t_idx in (0..t - 1).rev() {
            for i in 0..n {
                beta[t_idx][i] = (0..n)
                    .map(|j| self.transition[i][j] * self.emission[j][observations[t_idx + 1]] * beta[t_idx + 1][j])
                    .sum();
            }
        }

        beta
    }

    /// Viterbi algorithm: most likely state sequence.
    pub fn viterbi(&self, observations: &[usize]) -> (Vec<usize>, f64) {
        let t = observations.len();
        let n = self.num_states;

        let mut delta = vec![vec![0.0; n]; t];
        let mut psi = vec![vec![0usize; n]; t];

        // Initialization
        for i in 0..n {
            delta[0][i] = self.initial[i] * self.emission[i][observations[0]];
            psi[0][i] = 0;
        }

        // Recursion
        for t_idx in 1..t {
            for j in 0..n {
                let (best_i, best_val) = (0..n)
                    .map(|i| (i, delta[t_idx - 1][i] * self.transition[i][j]))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                delta[t_idx][j] = best_val * self.emission[j][observations[t_idx]];
                psi[t_idx][j] = best_i;
            }
        }

        // Termination
        let (best_last, best_prob) = (0..n)
            .map(|i| (i, delta[t - 1][i]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        // Backtrack
        let mut path = vec![0usize; t];
        path[t - 1] = best_last;
        for t_idx in (1..t).rev() {
            path[t_idx - 1] = psi[t_idx][path[t_idx]];
        }

        (path, best_prob)
    }

    /// Baum-Welch algorithm for parameter estimation.
    pub fn baum_welch(&mut self, observations: &[usize], max_iter: usize, tolerance: f64) {
        let t = observations.len();
        let n = self.num_states;

        for _ in 0..max_iter {
            let (alpha, prob) = self.forward(observations);
            let beta = self.backward(observations);

            if prob < 1e-300 { break; }

            // Compute gamma
            let mut gamma = vec![vec![0.0; n]; t];
            for t_idx in 0..t {
                for i in 0..n {
                    gamma[t_idx][i] = alpha[t_idx][i] * beta[t_idx][i] / prob;
                }
            }

            // Compute xi
            let mut xi = vec![vec![vec![0.0; n]; n]; t - 1];
            for t_idx in 0..t - 1 {
                for i in 0..n {
                    for j in 0..n {
                        xi[t_idx][i][j] = alpha[t_idx][i] * self.transition[i][j]
                            * self.emission[j][observations[t_idx + 1]] * beta[t_idx + 1][j] / prob;
                    }
                }
            }

            // Update initial probabilities
            self.initial = gamma[0].clone();

            // Update transition probabilities
            for i in 0..n {
                let denom: f64 = (0..t - 1).map(|t_idx| gamma[t_idx][i]).sum();
                for j in 0..n {
                    let numer: f64 = (0..t - 1).map(|t_idx| xi[t_idx][i][j]).sum();
                    self.transition[i][j] = if denom > 0.0 { numer / denom } else { self.transition[i][j] };
                }
            }

            // Update emission probabilities
            for i in 0..n {
                let denom: f64 = gamma.iter().map(|g| g[i]).sum();
                for o in 0..self.num_observations {
                    let numer: f64 = (0..t).filter(|&t_idx| observations[t_idx] == o).map(|t_idx| gamma[t_idx][i]).sum();
                    self.emission[i][o] = if denom > 0.0 { numer / denom } else { self.emission[i][o] };
                }
            }

            // Check convergence
            let (_, new_prob) = self.forward(observations);
            if (new_prob - prob).abs() < tolerance {
                break;
            }
        }
    }

    /// Generate random observation sequence.
    pub fn generate(&self, length: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
        let mut rng = seed;
        let mut rand = || -> f64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f64) / (1u64 << 31) as f64
        };

        let mut states = Vec::new();
        let mut obs = Vec::new();

        // Sample initial state
        let r = rand();
        let mut cum = 0.0;
        let mut state = 0;
        for (i, &p) in self.initial.iter().enumerate() {
            cum += p;
            if r < cum { state = i; break; }
        }

        for _ in 0..length {
            states.push(state);

            // Sample observation
            let r = rand();
            let mut cum = 0.0;
            for (o, &p) in self.emission[state].iter().enumerate() {
                cum += p;
                if r < cum { obs.push(o); break; }
            }

            // Transition
            let r = rand();
            let mut cum = 0.0;
            for (j, &p) in self.transition[state].iter().enumerate() {
                cum += p;
                if r < cum { state = j; break; }
            }
        }

        (states, obs)
    }
}

/// MCMC sampler (Metropolis-Hastings).
pub struct MetropolisHastings {
    pub log_prob: Box<dyn Fn(&[f64]) -> f64>,
    pub proposal_std: Vec<f64>,
    seed: u64,
}

impl MetropolisHastings {
    pub fn new(log_prob: Box<dyn Fn(&[f64]) -> f64>, proposal_std: Vec<f64>) -> Self {
        Self { log_prob, proposal_std, seed: 42 }
    }

    pub fn sample(&mut self, initial: &[f64], n_samples: usize) -> Vec<Vec<f64>> {
        let dim = initial.len();
        let mut current = initial.to_vec();
        let mut current_lp = (self.log_prob)(&current);
        let mut samples = Vec::new();

        for _ in 0..n_samples {
            // Propose
            let proposal: Vec<f64> = current.iter().zip(self.proposal_std.iter())
                .map(|(&x, &std)| x + self.gaussian() * std)
                .collect();

            let proposal_lp = (self.log_prob)(&proposal);

            // Accept/reject
            let log_alpha = proposal_lp - current_lp;
            if self.pseudo_rand() < log_alpha.exp() {
                current = proposal;
                current_lp = proposal_lp;
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

/// Gibbs sampler.
pub struct GibbsSampler {
    pub conditionals: Vec<Box<dyn Fn(&[f64], usize) -> f64>>,
    seed: u64,
}

impl GibbsSampler {
    pub fn new(conditionals: Vec<Box<dyn Fn(&[f64], usize) -> f64>>) -> Self {
        Self { conditionals, seed: 42 }
    }

    pub fn sample(&mut self, initial: &[f64], n_samples: usize) -> Vec<Vec<f64>> {
        let dim = initial.len();
        let mut current = initial.to_vec();
        let mut samples = Vec::new();

        for _ in 0..n_samples {
            for i in 0..dim {
                current[i] = (self.conditionals[i])(&current, i);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markov_stationary() {
        let transition = vec![
            vec![0.7, 0.3],
            vec![0.4, 0.6],
        ];
        let initial = vec![1.0, 0.0];
        let mc = MarkovChain::new(transition, initial);
        let pi = mc.stationary_distribution(1000, 1e-10);
        // Stationary distribution: [0.571, 0.429]
        assert!((pi[0] - 4.0 / 7.0).abs() < 0.01);
    }

    #[test]
    fn test_hmm_viterbi() {
        // Simple weather model: Sunny(0), Rainy(1)
        let transition = vec![
            vec![0.7, 0.3],
            vec![0.4, 0.6],
        ];
        let emission = vec![
            vec![0.8, 0.2], // Sunny: Walk(0), Shop(1)
            vec![0.4, 0.6], // Rainy: Walk(0), Shop(1)
        ];
        let initial = vec![0.6, 0.4];

        let hmm = HMM::new(transition, emission, initial);
        let observations = vec![0, 1, 0]; // Walk, Shop, Walk
        let (path, _prob) = hmm.viterbi(&observations);
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_hmm_forward() {
        let transition = vec![
            vec![0.7, 0.3],
            vec![0.4, 0.6],
        ];
        let emission = vec![
            vec![0.8, 0.2],
            vec![0.4, 0.6],
        ];
        let initial = vec![0.6, 0.4];

        let hmm = HMM::new(transition, emission, initial);
        let observations = vec![0, 1];
        let (_, prob) = hmm.forward(&observations);
        assert!(prob > 0.0);
    }
}
