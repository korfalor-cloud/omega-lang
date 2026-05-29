/// Offline RL: Conservative Q-Learning (CQL), Implicit Q-Learning (IQL).

use std::collections::HashMap;

/// Conservative Q-Learning (CQL).
pub struct CQL {
    pub state_dim: usize,
    pub action_dim: usize,
    pub q_weights: Vec<Vec<f64>>,
    pub q_bias: Vec<f64>,
    pub learning_rate: f64,
    pub alpha: f64, // CQL regularization coefficient
    pub discount: f64,
}

impl CQL {
    pub fn new(state_dim: usize, action_dim: usize, learning_rate: f64, alpha: f64, discount: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / state_dim as f64).sqrt();
        Self {
            state_dim, action_dim, learning_rate, alpha, discount,
            q_weights: (0..action_dim).map(|_| (0..state_dim).map(|_| rand(scale)).collect()).collect(),
            q_bias: vec![0.0; action_dim],
        }
    }

    pub fn q_value(&self, state: &[f64], action: usize) -> f64 {
        let q: Vec<f64> = self.q_weights.iter().zip(self.q_bias.iter()).map(|(w, &b)| {
            w.iter().zip(state.iter()).map(|(wi, si)| wi * si).sum::<f64>() + b
        }).collect();
        q[action.min(self.action_dim - 1)]
    }

    pub fn best_action(&self, state: &[f64]) -> usize {
        (0..self.action_dim)
            .max_by(|&a, &b| self.q_value(state, a).partial_cmp(&self.q_value(state, b)).unwrap())
            .unwrap()
    }

    /// CQL loss: standard TD loss + alpha * (logsumexp Q(s,a) - E[Q(s,a)]).
    pub fn cql_loss(&self, state: &[f64], action: usize, reward: f64, next_state: &[f64], done: bool) -> f64 {
        let q = self.q_value(state, action);
        let next_q = if done { 0.0 } else { self.q_value(next_state, self.best_action(next_state)) };
        let target = reward + self.discount * next_q;

        // TD error
        let td_error = q - target;
        let td_loss = td_error * td_error;

        // CQL regularization
        let all_q: Vec<f64> = (0..self.action_dim).map(|a| self.q_value(state, a)).collect();
        let max_q = all_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let logsumexp = max_q + all_q.iter().map(|qi| (qi - max_q).exp()).sum::<f64>().ln();
        let data_q = q;
        let cql_reg = self.alpha * (logsumexp - data_q);

        td_loss + cql_reg
    }

    pub fn update(&mut self, state: &[f64], action: usize, reward: f64, next_state: &[f64], done: bool) {
        let q = self.q_value(state, action);
        let next_q = if done { 0.0 } else { self.q_value(next_state, self.best_action(next_state)) };
        let target = reward + self.discount * next_q;
        let td_error = q - target;

        // CQL gradient (simplified)
        let all_q: Vec<f64> = (0..self.action_dim).map(|a| self.q_value(state, a)).collect();
        let max_q = all_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let softmax_probs: Vec<f64> = all_q.iter().map(|qi| (qi - max_q).exp()).collect();
        let sum_exp: f64 = softmax_probs.iter().sum();

        for i in 0..self.action_dim {
            let grad = 2.0 * td_error * state[i % state.len()]
                + self.alpha * (softmax_probs[i] / sum_exp - if i == action { 1.0 } else { 0.0 });
            for j in 0..self.state_dim {
                self.q_weights[i][j] -= self.learning_rate * grad * state[j];
            }
            self.q_bias[i] -= self.learning_rate * grad;
        }
    }
}

/// Implicit Q-Learning (IQL).
pub struct IQL {
    pub state_dim: usize,
    pub action_dim: usize,
    pub q_weights: Vec<Vec<f64>>,
    pub v_weights: Vec<f64>,
    pub learning_rate: f64,
    pub expectile: f64,
    pub discount: f64,
    pub tau: f64, // For asymmetric loss
}

impl IQL {
    pub fn new(state_dim: usize, action_dim: usize, learning_rate: f64, expectile: f64, discount: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / state_dim as f64).sqrt();
        Self {
            state_dim, action_dim, learning_rate, expectile, discount, tau: 0.7,
            q_weights: (0..action_dim).map(|_| (0..state_dim).map(|_| rand(scale)).collect()).collect(),
            v_weights: (0..state_dim).map(|_| rand(scale)).collect(),
        }
    }

    pub fn q_value(&self, state: &[f64], action: usize) -> f64 {
        let q: Vec<f64> = self.q_weights.iter().map(|w| {
            w.iter().zip(state.iter()).map(|(wi, si)| wi * si).sum()
        }).collect();
        q[action.min(self.action_dim - 1)]
    }

    pub fn v_value(&self, state: &[f64]) -> f64 {
        self.v_weights.iter().zip(state.iter()).map(|(w, s)| w * s).sum()
    }

    /// Asymmetric L2 loss (expectile regression).
    fn expectile_loss(&self, diff: f64) -> f64 {
        let weight = if diff > 0.0 { self.expectile } else { 1.0 - self.expectile };
        weight * diff * diff
    }

    pub fn update_value(&mut self, state: &[f64], action: usize) {
        let q = self.q_value(state, action);
        let v = self.v_value(state);
        let diff = q - v;

        // Update V with expectile loss
        let grad = if diff > 0.0 {
            2.0 * self.expectile * diff
        } else {
            2.0 * (1.0 - self.expectile) * diff
        };

        for i in 0..self.state_dim {
            self.v_weights[i] += self.learning_rate * grad * state[i];
        }
    }

    pub fn update_q(&mut self, state: &[f64], action: usize, reward: f64, next_state: &[f64], done: bool) {
        let v_next = if done { 0.0 } else { self.v_value(next_state) };
        let target = reward + self.discount * v_next;
        let q = self.q_value(state, action);
        let td_error = q - target;

        for i in 0..self.state_dim {
            self.q_weights[action][i] -= self.learning_rate * 2.0 * td_error * state[i];
        }
    }
}

/// Decision Transformer (simplified).
pub struct DecisionTransformer {
    pub state_dim: usize,
    pub action_dim: usize,
    pub context_length: usize,
    pub hidden_dim: usize,
    pub state_weights: Vec<Vec<f64>>,
    pub action_weights: Vec<Vec<f64>>,
    pub reward_weights: Vec<f64>,
}

impl DecisionTransformer {
    pub fn new(state_dim: usize, action_dim: usize, context_length: usize, hidden_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / (state_dim + action_dim + 1) as f64).sqrt();

        Self {
            state_dim, action_dim, context_length, hidden_dim,
            state_weights: (0..hidden_dim).map(|_| (0..state_dim).map(|_| rand(scale)).collect()).collect(),
            action_weights: (0..hidden_dim).map(|_| (0..action_dim).map(|_| rand(scale)).collect()).collect(),
            reward_weights: (0..hidden_dim).map(|_| rand(scale)).collect(),
        }
    }

    pub fn predict_action(&self, states: &[Vec<f64>], actions: &[Vec<f64>], rewards: &[f64]) -> Vec<f64> {
        // Simple attention-like mechanism
        let mut context = vec![0.0; self.hidden_dim];

        for (i, (state, action)) in states.iter().zip(actions.iter()).enumerate() {
            let reward = rewards.get(i).copied().unwrap_or(0.0);

            // Embed state, action, reward
            let state_embed: Vec<f64> = self.state_weights.iter().map(|w| {
                w.iter().zip(state.iter()).map(|(wi, si)| wi * si).sum()
            }).collect();

            let action_embed: Vec<f64> = self.action_weights.iter().map(|w| {
                w.iter().zip(action.iter()).map(|(wi, ai)| wi * ai).sum()
            }).collect();

            // Update context
            for j in 0..self.hidden_dim {
                context[j] += state_embed[j] + action_embed[j] + self.reward_weights[j] * reward;
            }
        }

        // Decode action from context
        let action: Vec<f64> = (0..self.action_dim).map(|i| {
            let scale = (i + 1) as f64 / self.action_dim as f64;
            context[i % self.hidden_dim] * scale
        }).collect();

        action
    }
}

/// BCQ (Batch-Constrained Q-Learning).
pub struct BCQ {
    pub state_dim: usize,
    pub action_dim: usize,
    pub q_weights: Vec<Vec<f64>>,
    pub policy_weights: Vec<Vec<f64>>,
    pub learning_rate: f64,
    pub discount: f64,
    pub threshold: f64,
}

impl BCQ {
    pub fn new(state_dim: usize, action_dim: usize, learning_rate: f64, discount: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / state_dim as f64).sqrt();
        Self {
            state_dim, action_dim, learning_rate, discount, threshold: 0.3,
            q_weights: (0..action_dim).map(|_| (0..state_dim).map(|_| rand(scale)).collect()).collect(),
            policy_weights: (0..action_dim).map(|_| (0..state_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn q_value(&self, state: &[f64], action: usize) -> f64 {
        self.q_weights[action].iter().zip(state.iter()).map(|(w, s)| w * s).sum()
    }

    pub fn policy_prob(&self, state: &[f64], action: usize) -> f64 {
        let logits: Vec<f64> = self.policy_weights.iter().map(|w| {
            w.iter().zip(state.iter()).map(|(wi, si)| wi * si).sum()
        }).collect();
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_logit = (logits[action] - max_logit).exp();
        let sum_exp: f64 = logits.iter().map(|l| (l - max_logit).exp()).sum();
        exp_logit / sum_exp
    }

    pub fn best_action(&self, state: &[f64]) -> usize {
        // Select action that maximizes Q among likely actions
        (0..self.action_dim)
            .filter(|&a| self.policy_prob(state, a) > self.threshold)
            .max_by(|&a, &b| self.q_value(state, a).partial_cmp(&self.q_value(state, b)).unwrap())
            .unwrap_or(0)
    }

    pub fn update(&mut self, state: &[f64], action: usize, reward: f64, next_state: &[f64], done: bool) {
        let q = self.q_value(state, action);
        let next_action = self.best_action(next_state);
        let next_q = if done { 0.0 } else { self.q_value(next_state, next_action) };
        let target = reward + self.discount * next_q;
        let td_error = q - target;

        // Update Q
        for j in 0..self.state_dim {
            self.q_weights[action][j] -= self.learning_rate * 2.0 * td_error * state[j];
        }

        // Update policy
        for i in 0..self.action_dim {
            let grad = if i == action { 1.0 - self.policy_prob(state, i) } else { -self.policy_prob(state, i) };
            for j in 0..self.state_dim {
                self.policy_weights[i][j] += self.learning_rate * td_error * grad * state[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cql() {
        let mut cql = CQL::new(2, 4, 0.01, 1.0, 0.99);
        cql.update(&[1.0, 0.0], 0, 1.0, &[0.0, 1.0], false);
        let q = cql.q_value(&[1.0, 0.0], 0);
        assert!(q.is_finite());
    }

    #[test]
    fn test_iql() {
        let mut iql = IQL::new(2, 4, 0.01, 0.7, 0.99);
        iql.update_value(&[1.0, 0.0], 0);
        iql.update_q(&[1.0, 0.0], 0, 1.0, &[0.0, 1.0], false);
    }
}
