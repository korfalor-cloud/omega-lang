/// Reinforcement learning: Q-learning, SARSA, policy gradient, actor-critic.

use std::collections::HashMap;

/// Q-Learning agent.
pub struct QLearning {
    pub q_table: HashMap<(usize, usize), f64>,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub epsilon: f64,
    pub n_actions: usize,
    seed: u64,
}

impl QLearning {
    pub fn new(n_actions: usize, learning_rate: f64, discount_factor: f64, epsilon: f64) -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor,
            epsilon,
            n_actions,
            seed: 42,
        }
    }

    pub fn get_q(&self, state: usize, action: usize) -> f64 {
        self.q_table.get(&(state, action)).copied().unwrap_or(0.0)
    }

    pub fn set_q(&mut self, state: usize, action: usize, value: f64) {
        self.q_table.insert((state, action), value);
    }

    pub fn choose_action(&mut self, state: usize) -> usize {
        if self.pseudo_rand() < self.epsilon {
            // Explore
            (self.pseudo_rand() * self.n_actions as f64) as usize % self.n_actions
        } else {
            // Exploit
            self.best_action(state)
        }
    }

    pub fn best_action(&self, state: usize) -> usize {
        (0..self.n_actions)
            .max_by(|&a, &b| {
                self.get_q(state, a).partial_cmp(&self.get_q(state, b)).unwrap()
            })
            .unwrap_or(0)
    }

    pub fn update(&mut self, state: usize, action: usize, reward: f64, next_state: usize) {
        let current_q = self.get_q(state, action);
        let max_next_q = (0..self.n_actions)
            .map(|a| self.get_q(next_state, a))
            .fold(f64::NEG_INFINITY, f64::max);

        let new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q);
        self.set_q(state, action, new_q);
    }

    pub fn decay_epsilon(&mut self, decay: f64) {
        self.epsilon *= decay;
        self.epsilon = self.epsilon.max(0.01);
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// SARSA agent.
pub struct SARSA {
    pub q_table: HashMap<(usize, usize), f64>,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub epsilon: f64,
    pub n_actions: usize,
    seed: u64,
}

impl SARSA {
    pub fn new(n_actions: usize, learning_rate: f64, discount_factor: f64, epsilon: f64) -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor,
            epsilon,
            n_actions,
            seed: 42,
        }
    }

    pub fn choose_action(&mut self, state: usize) -> usize {
        if self.pseudo_rand() < self.epsilon {
            (self.pseudo_rand() * self.n_actions as f64) as usize % self.n_actions
        } else {
            self.best_action(state)
        }
    }

    pub fn best_action(&self, state: usize) -> usize {
        (0..self.n_actions)
            .max_by(|&a, &b| {
                self.get_q(state, a).partial_cmp(&self.get_q(state, b)).unwrap()
            })
            .unwrap_or(0)
    }

    pub fn get_q(&self, state: usize, action: usize) -> f64 {
        self.q_table.get(&(state, action)).copied().unwrap_or(0.0)
    }

    pub fn update(&mut self, state: usize, action: usize, reward: f64, next_state: usize, next_action: usize) {
        let current_q = self.get_q(state, action);
        let next_q = self.get_q(next_state, next_action);
        let new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q);
        self.q_table.insert((state, action), new_q);
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Double Q-Learning.
pub struct DoubleQLearning {
    pub q1: HashMap<(usize, usize), f64>,
    pub q2: HashMap<(usize, usize), f64>,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub epsilon: f64,
    pub n_actions: usize,
    seed: u64,
}

impl DoubleQLearning {
    pub fn new(n_actions: usize, learning_rate: f64, discount_factor: f64, epsilon: f64) -> Self {
        Self {
            q1: HashMap::new(),
            q2: HashMap::new(),
            learning_rate,
            discount_factor,
            epsilon,
            n_actions,
            seed: 42,
        }
    }

    pub fn choose_action(&mut self, state: usize) -> usize {
        if self.pseudo_rand() < self.epsilon {
            (self.pseudo_rand() * self.n_actions as f64) as usize % self.n_actions
        } else {
            self.best_action(state)
        }
    }

    pub fn best_action(&self, state: usize) -> usize {
        (0..self.n_actions)
            .max_by(|&a, &b| {
                let qa = self.q1.get(&(state, a)).copied().unwrap_or(0.0) + self.q2.get(&(state, a)).copied().unwrap_or(0.0);
                let qb = self.q1.get(&(state, b)).copied().unwrap_or(0.0) + self.q2.get(&(state, b)).copied().unwrap_or(0.0);
                qa.partial_cmp(&qb).unwrap()
            })
            .unwrap_or(0)
    }

    pub fn update(&mut self, state: usize, action: usize, reward: f64, next_state: usize) {
        if self.pseudo_rand() < 0.5 {
            // Update Q1
            let current_q = self.q1.get(&(state, action)).copied().unwrap_or(0.0);
            let best_next_action = (0..self.n_actions)
                .max_by(|&a, &b| {
                    self.q1.get(&(next_state, a)).copied().unwrap_or(0.0)
                        .partial_cmp(&self.q1.get(&(next_state, b)).copied().unwrap_or(0.0))
                        .unwrap()
                })
                .unwrap_or(0);
            let next_q = self.q2.get(&(next_state, best_next_action)).copied().unwrap_or(0.0);
            let new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q);
            self.q1.insert((state, action), new_q);
        } else {
            // Update Q2
            let current_q = self.q2.get(&(state, action)).copied().unwrap_or(0.0);
            let best_next_action = (0..self.n_actions)
                .max_by(|&a, &b| {
                    self.q2.get(&(next_state, a)).copied().unwrap_or(0.0)
                        .partial_cmp(&self.q2.get(&(next_state, b)).copied().unwrap_or(0.0))
                        .unwrap()
                })
                .unwrap_or(0);
            let next_q = self.q1.get(&(next_state, best_next_action)).copied().unwrap_or(0.0);
            let new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q);
            self.q2.insert((state, action), new_q);
        }
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Policy gradient (REINFORCE) agent.
pub struct PolicyGradient {
    pub n_states: usize,
    pub n_actions: usize,
    pub policy: Vec<Vec<f64>>, // policy[state][action] = probability
    pub learning_rate: f64,
    pub discount_factor: f64,
    seed: u64,
}

impl PolicyGradient {
    pub fn new(n_states: usize, n_actions: usize, learning_rate: f64, discount_factor: f64) -> Self {
        let policy = vec![vec![1.0 / n_actions as f64; n_actions]; n_states];
        Self { n_states, n_actions, policy, learning_rate, discount_factor, seed: 42 }
    }

    pub fn choose_action(&mut self, state: usize) -> usize {
        let r = self.pseudo_rand();
        let mut cum = 0.0;
        for (a, &p) in self.policy[state].iter().enumerate() {
            cum += p;
            if r < cum { return a; }
        }
        self.n_actions - 1
    }

    pub fn update(&mut self, episode: &[(usize, usize, f64)]) {
        let t = episode.len();

        // Compute returns
        let mut returns = vec![0.0; t];
        returns[t - 1] = episode[t - 1].2;
        for i in (0..t - 1).rev() {
            returns[i] = episode[i].2 + self.discount_factor * returns[i + 1];
        }

        // Normalize returns
        let mean: f64 = returns.iter().sum::<f64>() / t as f64;
        let std = (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / t as f64).sqrt().max(1e-8);
        let normalized: Vec<f64> = returns.iter().map(|r| (r - mean) / std).collect();

        // Update policy
        for (i, &(state, action, _)) in episode.iter().enumerate() {
            let g = normalized[i];
            for a in 0..self.n_actions {
                let grad = if a == action { 1.0 - self.policy[state][a] } else { -self.policy[state][a] };
                self.policy[state][a] += self.learning_rate * g * grad;
            }

            // Normalize policy
            let sum: f64 = self.policy[state].iter().sum();
            for p in &mut self.policy[state] { *p /= sum; }
        }
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Actor-Critic agent.
pub struct ActorCritic {
    pub n_states: usize,
    pub n_actions: usize,
    pub actor: Vec<Vec<f64>>,   // Policy probabilities
    pub critic: Vec<f64>,        // State values
    pub actor_lr: f64,
    pub critic_lr: f64,
    pub discount_factor: f64,
    seed: u64,
}

impl ActorCritic {
    pub fn new(n_states: usize, n_actions: usize, actor_lr: f64, critic_lr: f64, discount_factor: f64) -> Self {
        Self {
            n_states, n_actions,
            actor: vec![vec![1.0 / n_actions as f64; n_actions]; n_states],
            critic: vec![0.0; n_states],
            actor_lr, critic_lr, discount_factor,
            seed: 42,
        }
    }

    pub fn choose_action(&mut self, state: usize) -> usize {
        let r = self.pseudo_rand();
        let mut cum = 0.0;
        for (a, &p) in self.actor[state].iter().enumerate() {
            cum += p;
            if r < cum { return a; }
        }
        self.n_actions - 1
    }

    pub fn update(&mut self, state: usize, action: usize, reward: f64, next_state: usize) {
        let td_error = reward + self.discount_factor * self.critic[next_state] - self.critic[state];

        // Update critic
        self.critic[state] += self.critic_lr * td_error;

        // Update actor
        for a in 0..self.n_actions {
            let grad = if a == action { 1.0 - self.actor[state][a] } else { -self.actor[state][a] };
            self.actor[state][a] += self.actor_lr * td_error * grad;
        }

        // Normalize
        let sum: f64 = self.actor[state].iter().sum();
        for p in &mut self.actor[state] { *p /= sum; }
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Monte Carlo Tree Search for RL.
pub struct MCTSRL {
    pub exploration: f64,
    pub n_simulations: usize,
    seed: u64,
}

impl MCTSRL {
    pub fn new(exploration: f64, n_simulations: usize) -> Self {
        Self { exploration, n_simulations, seed: 42 }
    }

    pub fn search(&mut self, state: usize, n_actions: usize, transition: &dyn Fn(usize, usize) -> (usize, f64, bool)) -> usize {
        let mut visits = vec![0usize; n_actions];
        let mut values = vec![0.0f64; n_actions];

        for _ in 0..self.n_simulations {
            let action = (self.pseudo_rand() * n_actions as f64) as usize % n_actions;
            let (next_state, reward, terminal) = transition(state, action);

            let value = if terminal {
                reward
            } else {
                reward + self.rollout(next_state, n_actions, transition, 100)
            };

            visits[action] += 1;
            values[action] += (value - values[action]) / visits[action] as f64;
        }

        // UCB1 action selection
        let total_visits: usize = visits.iter().sum();
        (0..n_actions)
            .max_by(|&a, &b| {
                let ucb_a = if visits[a] == 0 {
                    f64::INFINITY
                } else {
                    values[a] + self.exploration * ((total_visits as f64).ln() / visits[a] as f64).sqrt()
                };
                let ucb_b = if visits[b] == 0 {
                    f64::INFINITY
                } else {
                    values[b] + self.exploration * ((total_visits as f64).ln() / visits[b] as f64).sqrt()
                };
                ucb_a.partial_cmp(&ucb_b).unwrap()
            })
            .unwrap_or(0)
    }

    fn rollout(&mut self, mut state: usize, n_actions: usize, transition: &dyn Fn(usize, usize) -> (usize, f64, bool), max_steps: usize) -> f64 {
        let mut total_reward = 0.0;
        let mut gamma = 1.0;

        for _ in 0..max_steps {
            let action = (self.pseudo_rand() * n_actions as f64) as usize % n_actions;
            let (next_state, reward, terminal) = transition(state, action);
            total_reward += gamma * reward;
            gamma *= 0.99;
            state = next_state;
            if terminal { break; }
        }

        total_reward
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Experience replay buffer.
pub struct ReplayBuffer {
    pub buffer: Vec<(usize, usize, f64, usize, bool)>,
    pub capacity: usize,
    pub position: usize,
    seed: u64,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { buffer: Vec::new(), capacity, position: 0, seed: 42 }
    }

    pub fn push(&mut self, state: usize, action: usize, reward: f64, next_state: usize, done: bool) {
        if self.buffer.len() < self.capacity {
            self.buffer.push((state, action, reward, next_state, done));
        } else {
            self.buffer[self.position] = (state, action, reward, next_state, done);
        }
        self.position = (self.position + 1) % self.capacity;
    }

    pub fn sample(&mut self, batch_size: usize) -> Vec<(usize, usize, f64, usize, bool)> {
        let n = self.buffer.len();
        let mut batch = Vec::new();
        for _ in 0..batch_size.min(n) {
            let idx = (self.pseudo_rand() * n as f64) as usize % n;
            batch.push(self.buffer[idx]);
        }
        batch
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
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
    fn test_q_learning() {
        let mut agent = QLearning::new(4, 0.1, 0.99, 0.1);
        // Simple test: update Q-value
        agent.update(0, 1, 1.0, 1);
        assert!(agent.get_q(0, 1) > 0.0);
    }

    #[test]
    fn test_sarsa() {
        let mut agent = SARSA::new(4, 0.1, 0.99, 0.1);
        agent.update(0, 1, 1.0, 1, 2);
        assert!(agent.get_q(0, 1) > 0.0);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);
        buffer.push(0, 1, 1.0, 1, false);
        buffer.push(1, 2, 2.0, 2, true);
        assert_eq!(buffer.len(), 2);

        let batch = buffer.sample(2);
        assert_eq!(batch.len(), 2);
    }
}
