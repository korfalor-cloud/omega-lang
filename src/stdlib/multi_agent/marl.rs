/// Multi-agent RL: independent learners, QMIX, MADDPG.

use std::collections::HashMap;

/// Independent Q-Learning (each agent learns independently).
pub struct IndependentQLearning {
    pub n_agents: usize,
    pub n_actions: usize,
    pub q_tables: Vec<HashMap<(usize, usize), f64>>,
    pub learning_rate: f64,
    pub discount: f64,
    pub epsilon: f64,
    seed: u64,
}

impl IndependentQLearning {
    pub fn new(n_agents: usize, n_actions: usize, learning_rate: f64, discount: f64, epsilon: f64) -> Self {
        Self {
            n_agents, n_actions, learning_rate, discount, epsilon,
            q_tables: vec![HashMap::new(); n_agents],
            seed: 42,
        }
    }

    pub fn choose_action(&mut self, agent: usize, state: usize) -> usize {
        if self.pseudo_rand() < self.epsilon {
            (self.pseudo_rand() * self.n_actions as f64) as usize % self.n_actions
        } else {
            self.best_action(agent, state)
        }
    }

    pub fn best_action(&self, agent: usize, state: usize) -> usize {
        (0..self.n_actions)
            .max_by(|&a, &b| {
                let qa = self.q_tables[agent].get(&(state, a)).copied().unwrap_or(0.0);
                let qb = self.q_tables[agent].get(&(state, b)).copied().unwrap_or(0.0);
                qa.partial_cmp(&qb).unwrap()
            })
            .unwrap()
    }

    pub fn update(&mut self, agent: usize, state: usize, action: usize, reward: f64, next_state: usize) {
        let current_q = self.q_tables[agent].get(&(state, action)).copied().unwrap_or(0.0);
        let max_next_q = (0..self.n_actions)
            .map(|a| self.q_tables[agent].get(&(next_state, a)).copied().unwrap_or(0.0))
            .fold(f64::NEG_INFINITY, f64::max);

        let new_q = current_q + self.learning_rate * (reward + self.discount * max_next_q - current_q);
        self.q_tables[agent].insert((state, action), new_q);
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// QMIX: monotonic value factorization.
pub struct QMIX {
    pub n_agents: usize,
    pub n_actions: usize,
    pub agent_q_weights: Vec<Vec<Vec<f64>>>,
    pub mixing_weights: Vec<Vec<f64>>,
    pub mixing_bias: Vec<f64>,
    pub learning_rate: f64,
    pub discount: f64,
}

impl QMIX {
    pub fn new(n_agents: usize, n_actions: usize, hidden_dim: usize, learning_rate: f64, discount: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / n_actions as f64).sqrt();
        let mix_scale = (2.0 / (n_agents * hidden_dim) as f64).sqrt();

        Self {
            n_agents, n_actions, learning_rate, discount,
            agent_q_weights: (0..n_agents).map(|_| {
                (0..hidden_dim).map(|_| (0..n_actions).map(|_| rand(scale)).collect()).collect()
            }).collect(),
            mixing_weights: (0..hidden_dim).map(|_| (0..n_agents).map(|_| rand(mix_scale)).collect()).collect(),
            mixing_bias: vec![0.0; hidden_dim],
        }
    }

    pub fn agent_q_values(&self, agent: usize, state: &[f64]) -> Vec<f64> {
        (0..self.n_actions).map(|a| {
            self.agent_q_weights[agent].iter().map(|w| {
                w.iter().zip(state.iter()).map(|(wi, si)| wi * si).sum::<f64>()
            }).sum::<f64>()
        }).collect()
    }

    pub fn mixing_q(&self, agent_q_values: &[Vec<f64>], state: &[f64]) -> f64 {
        // Monotonic mixing: w_i(state) >= 0
        let mut total_q = 0.0;
        for (i, q_vals) in agent_q_values.iter().enumerate() {
            let best_q = q_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let weight: f64 = self.mixing_weights.iter().map(|w| {
                w[i].abs() // Ensure positive
            }).sum::<f64>() / self.mixing_weights.len() as f64;
            total_q += weight * best_q;
        }
        total_q
    }

    pub fn choose_actions(&self, states: &[Vec<f64>]) -> Vec<usize> {
        states.iter().enumerate().map(|(i, state)| {
            let q_vals = self.agent_q_values(i, state);
            q_vals.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        }).collect()
    }
}

/// MADDPG (Multi-Agent DDPG).
pub struct MADDPG {
    pub n_agents: usize,
    pub state_dim: usize,
    pub action_dim: usize,
    pub actor_weights: Vec<Vec<Vec<f64>>>,
    pub critic_weights: Vec<Vec<f64>>,
    pub learning_rate: f64,
    pub discount: f64,
    pub tau: f64,
}

impl MADDPG {
    pub fn new(n_agents: usize, state_dim: usize, action_dim: usize, learning_rate: f64, discount: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let actor_scale = (2.0 / state_dim as f64).sqrt();
        let critic_scale = (2.0 / (n_agents * (state_dim + action_dim)) as f64).sqrt();

        Self {
            n_agents, state_dim, action_dim, learning_rate, discount, tau: 0.01,
            actor_weights: (0..n_agents).map(|_| {
                (0..action_dim).map(|_| (0..state_dim).map(|_| rand(actor_scale)).collect()).collect()
            }).collect(),
            critic_weights: (0..n_agents).map(|_| {
                (0..n_agents * (state_dim + action_dim)).map(|_| rand(critic_scale)).collect()
            }).collect(),
        }
    }

    pub fn actor(&self, agent: usize, state: &[f64]) -> Vec<f64> {
        self.actor_weights[agent].iter().map(|w| {
            let sum: f64 = w.iter().zip(state.iter()).map(|(wi, si)| wi * si).sum();
            sum.tanh()
        }).collect()
    }

    pub fn critic(&self, agent: usize, all_states: &[Vec<f64>], all_actions: &[Vec<f64>]) -> f64 {
        let mut input = Vec::new();
        for (s, a) in all_states.iter().zip(all_actions.iter()) {
            input.extend_from_slice(s);
            input.extend_from_slice(a);
        }
        self.critic_weights[agent].iter().zip(input.iter()).map(|(w, x)| w * x).sum()
    }

    pub fn choose_actions(&self, states: &[Vec<f64>]) -> Vec<Vec<f64>> {
        states.iter().enumerate().map(|(i, state)| self.actor(i, state)).collect()
    }
}

/// Communication learning between agents.
pub struct CommNet {
    pub n_agents: usize,
    pub hidden_dim: usize,
    pub message_weights: Vec<Vec<Vec<f64>>>,
    pub combine_weights: Vec<Vec<f64>>,
}

impl CommNet {
    pub fn new(n_agents: usize, input_dim: usize, hidden_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();
        let comm_scale = (2.0 / hidden_dim as f64).sqrt();

        Self {
            n_agents, hidden_dim,
            message_weights: (0..n_agents).map(|_| {
                (0..hidden_dim).map(|_| (0..hidden_dim).map(|_| rand(comm_scale)).collect()).collect()
            }).collect(),
            combine_weights: (0..n_agents).map(|_| (0..hidden_dim * 2).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn forward(&self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // Encode inputs to hidden
        let hidden: Vec<Vec<f64>> = inputs.iter().map(|x| {
            x.iter().map(|&xi| xi.tanh()).collect()
        }).collect();

        // Compute messages
        let mut messages = vec![vec![0.0; self.hidden_dim]; self.n_agents];
        for i in 0..self.n_agents {
            for j in 0..self.n_agents {
                if i != j {
                    for k in 0..self.hidden_dim {
                        messages[i][k] += self.message_weights[i][k].iter().zip(hidden[j].iter())
                            .map(|(w, h)| w * h).sum::<f64>();
                    }
                }
            }
            // Average messages
            for m in messages[i].iter_mut() { *m /= (self.n_agents - 1) as f64; }
        }

        // Combine hidden and messages
        (0..self.n_agents).map(|i| {
            let mut combined = hidden[i].clone();
            combined.extend_from_slice(&messages[i]);
            self.combine_weights[i].iter().zip(combined.iter()).map(|(w, c)| w * c).sum::<f64>();
            vec![combined.iter().map(|&c| c.tanh()).sum::<f64>() / combined.len() as f64; self.hidden_dim]
        }).collect()
    }
}

/// Mean Field Game approximation.
pub struct MeanFieldQ {
    pub n_actions: usize,
    pub q_table: HashMap<(usize, usize, usize), f64>, // (state, action, mean_action) -> Q
    pub learning_rate: f64,
    pub discount: f64,
}

impl MeanFieldQ {
    pub fn new(n_actions: usize, learning_rate: f64, discount: f64) -> Self {
        Self { n_actions, q_table: HashMap::new(), learning_rate, discount }
    }

    pub fn q_value(&self, state: usize, action: usize, mean_action: usize) -> f64 {
        self.q_table.get(&(state, action, mean_action)).copied().unwrap_or(0.0)
    }

    pub fn best_action(&self, state: usize, mean_action: usize) -> usize {
        (0..self.n_actions)
            .max_by(|&a, &b| {
                self.q_value(state, a, mean_action).partial_cmp(&self.q_value(state, b, mean_action)).unwrap()
            })
            .unwrap()
    }

    pub fn update(&mut self, state: usize, action: usize, mean_action: usize, reward: f64, next_state: usize, next_mean_action: usize) {
        let current_q = self.q_value(state, action, mean_action);
        let max_next_q = (0..self.n_actions)
            .map(|a| self.q_value(next_state, a, next_mean_action))
            .fold(f64::NEG_INFINITY, f64::max);

        let new_q = current_q + self.learning_rate * (reward + self.discount * max_next_q - current_q);
        self.q_table.insert((state, action, mean_action), new_q);
    }
}

/// Compute mean action from population.
pub fn compute_mean_action(actions: &[usize], n_actions: usize) -> usize {
    let mut counts = vec![0usize; n_actions];
    for &a in actions {
        counts[a] += 1;
    }
    counts.iter().enumerate()
        .max_by_key(|(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_ql() {
        let mut iql = IndependentQLearning::new(3, 4, 0.1, 0.99, 0.1);
        iql.update(0, 0, 1, 1.0, 1);
        assert!(iql.q_tables[0].get(&(0, 1)).is_some());
    }

    #[test]
    fn test_mean_field() {
        let mut mfq = MeanFieldQ::new(4, 0.1, 0.99);
        mfq.update(0, 1, 2, 1.0, 1, 0);
        assert!(mfq.q_value(0, 1, 2) > 0.0);
    }
}
