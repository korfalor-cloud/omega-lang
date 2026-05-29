/// Imitation learning: behavioral cloning, DAgger, inverse reinforcement learning.

/// Behavioral cloning: learn policy from expert demonstrations.
pub struct BehavioralCloning {
    pub state_dim: usize,
    pub action_dim: usize,
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub learning_rate: f64,
}

impl BehavioralCloning {
    pub fn new(state_dim: usize, action_dim: usize, learning_rate: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / state_dim as f64).sqrt();
        Self {
            state_dim, action_dim, learning_rate,
            weights: (0..action_dim).map(|_| (0..state_dim).map(|_| rand(scale)).collect()).collect(),
            bias: vec![0.0; action_dim],
        }
    }

    pub fn predict(&self, state: &[f64]) -> Vec<f64> {
        self.weights.iter().zip(self.bias.iter()).map(|(w, &b)| {
            w.iter().zip(state.iter()).map(|(wi, si)| wi * si).sum::<f64>() + b
        }).collect()
    }

    pub fn train_step(&mut self, state: &[f64], expert_action: &[f64]) -> f64 {
        let predicted = self.predict(state);

        // MSE loss gradient
        let mut loss = 0.0;
        for (i, (pred, expert)) in predicted.iter().zip(expert_action.iter()).enumerate() {
            let error = pred - expert;
            loss += error * error;

            // Update weights
            for j in 0..self.state_dim {
                self.weights[i][j] -= self.learning_rate * 2.0 * error * state[j];
            }
            self.bias[i] -= self.learning_rate * 2.0 * error;
        }

        loss / self.action_dim as f64
    }

    pub fn fit(&mut self, states: &[Vec<f64>], actions: &[Vec<f64>], epochs: usize) {
        for _ in 0..epochs {
            for (state, action) in states.iter().zip(actions.iter()) {
                self.train_step(state, action);
            }
        }
    }
}

/// DAgger (Dataset Aggregation).
pub struct DAgger {
    pub policy: BehavioralCloning,
    pub expert_data: Vec<(Vec<f64>, Vec<f64>)>,
    pub learning_rate: f64,
}

impl DAgger {
    pub fn new(state_dim: usize, action_dim: usize, learning_rate: f64) -> Self {
        Self {
            policy: BehavioralCloning::new(state_dim, action_dim, learning_rate),
            expert_data: Vec::new(),
            learning_rate,
        }
    }

    /// Add expert demonstration.
    pub fn add_expert_data(&mut self, states: Vec<Vec<f64>>, actions: Vec<Vec<f64>>) {
        for (s, a) in states.into_iter().zip(actions.into_iter()) {
            self.expert_data.push((s, a));
        }
    }

    /// DAgger iteration: collect data using current policy, label with expert.
    pub fn dagger_iteration<F>(&mut self, expert_label: F, n_samples: usize)
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        // Collect new data using current policy
        let mut rng = 42u64;
        for _ in 0..n_samples {
            // Generate random state (simplified)
            let state: Vec<f64> = (0..self.policy.state_dim).map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((rng >> 33) as f64) / (1u64 << 31) as f64 * 2.0 - 1.0
            }).collect();

            // Get expert label for this state
            let expert_action = expert_label(&state);
            self.expert_data.push((state, expert_action));
        }

        // Re-train policy on all data
        let states: Vec<Vec<f64>> = self.expert_data.iter().map(|(s, _)| s.clone()).collect();
        let actions: Vec<Vec<f64>> = self.expert_data.iter().map(|(_, a)| a.clone()).collect();
        self.policy.fit(&states, &actions, 10);
    }

    pub fn predict(&self, state: &[f64]) -> Vec<f64> {
        self.policy.predict(state)
    }
}

/// Inverse Reinforcement Learning (MaxEntropy IRL).
pub struct MaxEntropyIRL {
    pub state_dim: usize,
    pub action_dim: usize,
    pub reward_weights: Vec<f64>,
    pub learning_rate: f64,
}

impl MaxEntropyIRL {
    pub fn new(state_dim: usize, action_dim: usize, learning_rate: f64) -> Self {
        Self {
            state_dim, action_dim, learning_rate,
            reward_weights: vec![0.0; state_dim],
        }
    }

    pub fn reward(&self, state: &[f64]) -> f64 {
        self.reward_weights.iter().zip(state.iter()).map(|(w, s)| w * s).sum()
    }

    /// Compute expected state visitation frequencies.
    pub fn expected_svf(&self, policy: &dyn Fn(&[f64]) -> Vec<f64>, n_states: usize, n_trajectories: usize) -> Vec<f64> {
        let mut svf = vec![0.0; n_states];
        let mut rng = 42u64;

        for _ in 0..n_trajectories {
            let mut state = vec![0.0; self.state_dim];
            for _ in 0..100 { // Max trajectory length
                let state_idx = ((state.iter().sum::<f64>().abs() * 10.0) as usize).min(n_states - 1);
                svf[state_idx] += 1.0;

                let action = policy(&state);
                // Simple transition (simplified)
                for i in 0..self.state_dim {
                    state[i] += action.get(i).copied().unwrap_or(0.0) * 0.1;
                    state[i] = state[i].max(-1.0).min(1.0);
                }
            }
        }

        let total: f64 = svf.iter().sum();
        for s in svf.iter_mut() { *s /= total; }
        svf
    }

    /// Update reward weights.
    pub fn update(&mut self, expert_svf: &[f64], policy_svf: &[f64]) {
        for (i, (e, p)) in expert_svf.iter().zip(policy_svf.iter()).enumerate() {
            let grad = e - p;
            self.reward_weights[i.min(self.state_dim - 1)] += self.learning_rate * grad;
        }
    }
}

/// Generative Adversarial Imitation Learning (GAIL).
pub struct GAIL {
    pub state_dim: usize,
    pub action_dim: usize,
    pub discriminator_weights: Vec<Vec<f64>>,
    pub policy_weights: Vec<Vec<f64>>,
    pub learning_rate: f64,
}

impl GAIL {
    pub fn new(state_dim: usize, action_dim: usize, learning_rate: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let input_dim = state_dim + action_dim;
        let scale = (2.0 / input_dim as f64).sqrt();

        Self {
            state_dim, action_dim, learning_rate,
            discriminator_weights: vec![vec![rand(scale); input_dim]],
            policy_weights: (0..action_dim).map(|_| (0..state_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn discriminator(&self, state: &[f64], action: &[f64]) -> f64 {
        let mut input = state.to_vec();
        input.extend_from_slice(action);
        let logit: f64 = self.discriminator_weights[0].iter().zip(input.iter()).map(|(w, x)| w * x).sum();
        1.0 / (1.0 + (-logit).exp()) // sigmoid
    }

    pub fn policy(&self, state: &[f64]) -> Vec<f64> {
        self.policy_weights.iter().map(|w| {
            w.iter().zip(state.iter()).map(|(wi, si)| wi * si).sum()
        }).collect()
    }

    pub fn train_discriminator_step(&mut self, expert_state: &[f64], expert_action: &[f64],
                                      policy_state: &[f64], policy_action: &[f64]) {
        let d_expert = self.discriminator(expert_state, expert_action);
        let d_policy = self.discriminator(policy_state, policy_action);

        // Discriminator loss: -log(D(expert)) - log(1 - D(policy))
        let grad_expert = -1.0 / d_expert.max(1e-10);
        let grad_policy = 1.0 / (1.0 - d_policy).max(1e-10);

        let mut expert_input = expert_state.to_vec();
        expert_input.extend_from_slice(expert_action);
        let mut policy_input = policy_state.to_vec();
        policy_input.extend_from_slice(policy_action);

        for (i, w) in self.discriminator_weights[0].iter_mut().enumerate() {
            *w -= self.learning_rate * (grad_expert * expert_input[i] + grad_policy * policy_input[i]);
        }
    }

    pub fn train_policy_step(&mut self, state: &[f64]) {
        let action = self.policy(state);
        let d = self.discriminator(state, &action);

        // Policy gradient using discriminator as reward
        let reward = -(1.0 - d).max(1e-10).ln();
        for (i, w_row) in self.policy_weights.iter_mut().enumerate() {
            for (j, w) in w_row.iter_mut().enumerate() {
                *w += self.learning_rate * reward * state[j.min(state.len() - 1)];
            }
        }
    }
}

/// Dataset for imitation learning.
pub struct ImitationDataset {
    pub states: Vec<Vec<f64>>,
    pub actions: Vec<Vec<f64>>,
    pub rewards: Vec<f64>,
}

impl ImitationDataset {
    pub fn new() -> Self {
        Self { states: Vec::new(), actions: Vec::new(), rewards: Vec::new() }
    }

    pub fn add(&mut self, state: Vec<f64>, action: Vec<f64>, reward: f64) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn sample_batch(&self, batch_size: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = seed;
        let n = self.states.len();
        let mut batch_states = Vec::new();
        let mut batch_actions = Vec::new();
        let mut batch_rewards = Vec::new();

        for _ in 0..batch_size.min(n) {
            let idx = ((rng >> 33) as usize) % n;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            batch_states.push(self.states[idx].clone());
            batch_actions.push(self.actions[idx].clone());
            batch_rewards.push(self.rewards[idx]);
        }

        (batch_states, batch_actions, batch_rewards)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavioral_cloning() {
        let mut bc = BehavioralCloning::new(2, 2, 0.01);
        let states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let actions = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        bc.fit(&states, &actions, 100);

        let pred = bc.predict(&[1.0, 0.0]);
        assert!(pred[0] > pred[1]);
    }

    #[test]
    fn test_gail() {
        let mut gail = GAIL::new(2, 2, 0.01);
        let d = gail.discriminator(&[1.0, 0.0], &[1.0, 0.0]);
        assert!(d > 0.0 && d < 1.0);
    }
}
