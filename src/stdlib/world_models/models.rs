/// World models: environment dynamics learning, latent imagination.

/// Environment model that learns transition dynamics.
pub struct EnvironmentModel {
    pub state_dim: usize,
    pub action_dim: usize,
    pub hidden_dim: usize,
    pub encoder_weights: Vec<Vec<f64>>,
    pub dynamics_weights: Vec<Vec<f64>>,
    pub reward_weights: Vec<f64>,
    pub decoder_weights: Vec<Vec<f64>>,
}

impl EnvironmentModel {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let enc_scale = (2.0 / state_dim as f64).sqrt();
        let dyn_scale = (2.0 / (hidden_dim + action_dim) as f64).sqrt();
        let dec_scale = (2.0 / hidden_dim as f64).sqrt();

        Self {
            state_dim, action_dim, hidden_dim,
            encoder_weights: (0..hidden_dim).map(|_| (0..state_dim).map(|_| rand(enc_scale)).collect()).collect(),
            dynamics_weights: (0..hidden_dim).map(|_| (0..hidden_dim + action_dim).map(|_| rand(dyn_scale)).collect()).collect(),
            reward_weights: (0..hidden_dim).map(|_| rand(dec_scale)).collect(),
            decoder_weights: (0..state_dim).map(|_| (0..hidden_dim).map(|_| rand(dec_scale)).collect()).collect(),
        }
    }

    pub fn encode(&self, state: &[f64]) -> Vec<f64> {
        self.encoder_weights.iter().map(|w| {
            w.iter().zip(state.iter()).map(|(wi, si)| wi * si).sum::<f64>().tanh()
        }).collect()
    }

    pub fn decode(&self, latent: &[f64]) -> Vec<f64> {
        self.decoder_weights.iter().map(|w| {
            w.iter().zip(latent.iter()).map(|(wi, li)| wi * li).sum()
        }).collect()
    }

    pub fn predict_next_latent(&self, latent: &[f64], action: &[f64]) -> Vec<f64> {
        let mut input = latent.to_vec();
        input.extend_from_slice(action);

        self.dynamics_weights.iter().map(|w| {
            w.iter().zip(input.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
        }).collect()
    }

    pub fn predict_reward(&self, latent: &[f64]) -> f64 {
        self.reward_weights.iter().zip(latent.iter()).map(|(w, l)| w * l).sum()
    }

    pub fn imagine_trajectory(&self, initial_state: &[f64], actions: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut latent = self.encode(initial_state);
        let mut states = vec![self.decode(&latent)];
        let mut rewards = Vec::new();

        for action in actions {
            latent = self.predict_next_latent(&latent, action);
            states.push(self.decode(&latent));
            rewards.push(self.predict_reward(&latent));
        }

        (states, rewards)
    }
}

/// Dreamer-style world model with RSSM (Recurrent State-Space Model).
pub struct RSSM {
    pub deterministic_dim: usize,
    pub stochastic_dim: usize,
    pub action_dim: usize,
    pub hidden_dim: usize,
    pub prior_weights: Vec<Vec<f64>>,
    pub posterior_weights: Vec<Vec<f64>>,
    pub rnn_weights: Vec<Vec<f64>>,
}

impl RSSM {
    pub fn new(deterministic_dim: usize, stochastic_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let prior_scale = (2.0 / (deterministic_dim + stochastic_dim) as f64).sqrt();
        let post_scale = (2.0 / (deterministic_dim + stochastic_dim + hidden_dim) as f64).sqrt();
        let rnn_scale = (2.0 / (stochastic_dim + action_dim) as f64).sqrt();

        Self {
            deterministic_dim, stochastic_dim, action_dim, hidden_dim,
            prior_weights: (0..stochastic_dim * 2).map(|_| (0..deterministic_dim + stochastic_dim).map(|_| rand(prior_scale)).collect()).collect(),
            posterior_weights: (0..stochastic_dim * 2).map(|_| (0..deterministic_dim + stochastic_dim + hidden_dim).map(|_| rand(post_scale)).collect()).collect(),
            rnn_weights: (0..deterministic_dim).map(|_| (0..stochastic_dim + action_dim).map(|_| rand(rnn_scale)).collect()).collect(),
        }
    }

    pub fn prior(&self, deterministic: &[f64], stochastic: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut input = deterministic.to_vec();
        input.extend_from_slice(stochastic);

        let outputs: Vec<f64> = self.prior_weights.iter().map(|w| {
            w.iter().zip(input.iter()).map(|(wi, xi)| wi * xi).sum()
        }).collect();

        let mean = outputs[..self.stochastic_dim].to_vec();
        let log_std = outputs[self.stochastic_dim..].iter().map(|&x| x.max(-10.0).min(2.0)).collect();
        (mean, log_std)
    }

    pub fn posterior(&self, deterministic: &[f64], stochastic: &[f64], observation: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut input = deterministic.to_vec();
        input.extend_from_slice(stochastic);
        input.extend_from_slice(observation);

        let outputs: Vec<f64> = self.posterior_weights.iter().map(|w| {
            w.iter().zip(input.iter()).map(|(wi, xi)| wi * xi).sum()
        }).collect();

        let mean = outputs[..self.stochastic_dim].to_vec();
        let log_std = outputs[self.stochastic_dim..].iter().map(|&x| x.max(-10.0).min(2.0)).collect();
        (mean, log_std)
    }

    pub fn rnn_step(&self, stochastic: &[f64], action: &[f64]) -> Vec<f64> {
        let mut input = stochastic.to_vec();
        input.extend_from_slice(action);

        self.rnn_weights.iter().map(|w| {
            w.iter().zip(input.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
        }).collect()
    }

    pub fn imagine(&self, initial_stochastic: &[f64], actions: &[Vec<f64>]) -> Vec<(Vec<f64>, Vec<f64>)> {
        let mut stochastic = initial_stochastic.to_vec();
        let mut deterministic = vec![0.0; self.deterministic_dim];
        let mut trajectory = Vec::new();

        for action in actions {
            deterministic = self.rnn_step(&stochastic, action);
            let (mean, log_std) = self.prior(&deterministic, &stochastic);
            // Sample from prior
            stochastic = mean; // Simplified: use mean
            trajectory.push((deterministic.clone(), stochastic.clone()));
        }

        trajectory
    }
}

/// Actor-Critic in imagination (Dreamer).
pub struct DreamerActorCritic {
    pub state_dim: usize,
    pub action_dim: usize,
    pub actor_weights: Vec<Vec<f64>>,
    pub critic_weights: Vec<f64>,
    pub learning_rate: f64,
    pub discount: f64,
    pub lambda_: f64,
}

impl DreamerActorCritic {
    pub fn new(state_dim: usize, action_dim: usize, learning_rate: f64, discount: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / state_dim as f64).sqrt();
        Self {
            state_dim, action_dim, learning_rate, discount, lambda_: 0.95,
            actor_weights: (0..action_dim).map(|_| (0..state_dim).map(|_| rand(scale)).collect()).collect(),
            critic_weights: (0..state_dim).map(|_| rand(scale)).collect(),
        }
    }

    pub fn actor(&self, state: &[f64]) -> Vec<f64> {
        let logits: Vec<f64> = self.actor_weights.iter().map(|w| {
            w.iter().zip(state.iter()).map(|(wi, si)| wi * si).sum()
        }).collect();
        softmax(&logits)
    }

    pub fn critic(&self, state: &[f64]) -> f64 {
        self.critic_weights.iter().zip(state.iter()).map(|(w, s)| w * s).sum()
    }

    pub fn compute_lambda_returns(&self, rewards: &[f64], values: &[f64]) -> Vec<f64> {
        let t = rewards.len();
        let mut returns = vec![0.0; t];
        returns[t - 1] = rewards[t - 1] + self.discount * values[t - 1];

        for i in (0..t - 1).rev() {
            returns[i] = rewards[i] + self.discount * ((1.0 - self.lambda_) * values[i] + self.lambda_ * returns[i + 1]);
        }

        returns
    }

    pub fn update_critic(&mut self, states: &[Vec<f64>], returns: &[f64]) {
        for (state, &ret) in states.iter().zip(returns.iter()) {
            let value = self.critic(state);
            let error = ret - value;
            for (w, s) in self.critic_weights.iter_mut().zip(state.iter()) {
                *w += self.learning_rate * 2.0 * error * s;
            }
        }
    }

    pub fn update_actor(&mut self, states: &[Vec<f64>], actions: &[Vec<f64>], returns: &[f64]) {
        for (state, (action, &ret)) in states.iter().zip(actions.iter().zip(returns.iter())) {
            let value = self.critic(state);
            let advantage = ret - value;
            let action_probs = self.actor(state);

            for (i, w_row) in self.actor_weights.iter_mut().enumerate() {
                let grad = advantage * (action[i] - action_probs[i]);
                for (j, w) in w_row.iter_mut().enumerate() {
                    *w += self.learning_rate * grad * state[j];
                }
            }
        }
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// PlaNet (Deep Planning Network).
pub struct PlaNet {
    pub env_model: EnvironmentModel,
    pub horizon: usize,
    pub n_candidates: usize,
    seed: u64,
}

impl PlaNet {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize, horizon: usize, n_candidates: usize) -> Self {
        Self {
            env_model: EnvironmentModel::new(state_dim, action_dim, hidden_dim),
            horizon, n_candidates,
            seed: 42,
        }
    }

    pub fn plan(&mut self, current_state: &[f64]) -> Vec<f64> {
        let mut best_actions = vec![0.0; self.env_model.action_dim];
        let mut best_reward = f64::NEG_INFINITY;

        for _ in 0..self.n_candidates {
            // Random shooting
            let actions: Vec<Vec<f64>> = (0..self.horizon).map(|_| {
                (0..self.env_model.action_dim).map(|_| {
                    self.pseudo_rand() * 2.0 - 1.0
                }).collect()
            }).collect();

            let (_, rewards) = self.env_model.imagine_trajectory(current_state, &actions);
            let total_reward: f64 = rewards.iter().enumerate().map(|(i, r)| r * self.env_model.hidden_dim as f64.powi(-(i as i32))).sum();

            if total_reward > best_reward {
                best_reward = total_reward;
                best_actions = actions[0].clone();
            }
        }

        best_actions
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
    fn test_environment_model() {
        let model = EnvironmentModel::new(4, 2, 8);
        let state = vec![1.0, 0.0, 0.0, 0.0];
        let action = vec![1.0, 0.0];

        let latent = model.encode(&state);
        assert_eq!(latent.len(), 8);

        let next_latent = model.predict_next_latent(&latent, &action);
        assert_eq!(next_latent.len(), 8);

        let reward = model.predict_reward(&latent);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_rssm() {
        let rssm = RSSM::new(8, 4, 2, 8);
        let stochastic = vec![0.0; 4];
        let action = vec![1.0, 0.0];

        let deterministic = rssm.rnn_step(&stochastic, &action);
        assert_eq!(deterministic.len(), 8);

        let (mean, log_std) = rssm.prior(&deterministic, &stochastic);
        assert_eq!(mean.len(), 4);
        assert_eq!(log_std.len(), 4);
    }
}
