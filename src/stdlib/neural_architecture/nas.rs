/// Neural Architecture Search: DARTS, evolution-based NAS, weight sharing.

/// Architecture encoding.
#[derive(Clone, Debug)]
pub struct Architecture {
    pub n_layers: usize,
    pub operations: Vec<usize>, // 0: conv3x3, 1: conv5x5, 2: maxpool, 3: avgpool, 4: skip
    pub connections: Vec<usize>, // Which previous layer to connect
}

impl Architecture {
    pub fn new(n_layers: usize) -> Self {
        Self {
            n_layers,
            operations: vec![0; n_layers],
            connections: vec![0; n_layers],
        }
    }

    pub fn random(n_layers: usize, n_ops: usize, seed: u64) -> Self {
        let mut rng = seed;
        let mut rand = |max: usize| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as usize) % max
        };

        Self {
            n_layers,
            operations: (0..n_layers).map(|_| rand(n_ops)).collect(),
            connections: (0..n_layers).map(|i| if i > 0 { rand(i) } else { 0 }).collect(),
        }
    }

    pub fn mutate(&self, n_ops: usize, seed: u64) -> Self {
        let mut rng = seed;
        let mut rand = |max: usize| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as usize) % max
        };

        let mut arch = self.clone();
        let layer = rand(self.n_layers);

        if rand(2) == 0 {
            arch.operations[layer] = rand(n_ops);
        } else if layer > 0 {
            arch.connections[layer] = rand(layer);
        }

        arch
    }
}

/// DARTS (Differentiable Architecture Search).
pub struct DARTS {
    pub n_nodes: usize,
    pub n_ops: usize,
    pub alphas: Vec<Vec<f64>>, // Architecture parameters
    pub weights: Vec<Vec<f64>>, // Network weights
}

impl DARTS {
    pub fn new(n_nodes: usize, n_ops: usize, input_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.01
        };

        Self {
            n_nodes, n_ops,
            alphas: (0..n_nodes).map(|_| (0..n_ops).map(|_| rand()).collect()).collect(),
            weights: (0..n_nodes).map(|_| (0..input_dim).map(|_| rand()).collect()).collect(),
        }
    }

    /// Compute mixed operation: sum of softmax(alpha) * op(x).
    pub fn mixed_op(&self, node: usize, x: &[f64]) -> Vec<f64> {
        let probs = softmax(&self.alphas[node]);
        let dim = x.len();

        let mut result = vec![0.0; dim];
        for (op_idx, &prob) in probs.iter().enumerate() {
            let op_result = self.apply_op(op_idx, x);
            for i in 0..dim {
                result[i] += prob * op_result[i];
            }
        }

        result
    }

    fn apply_op(&self, op: usize, x: &[f64]) -> Vec<f64> {
        match op {
            0 => x.to_vec(), // Identity
            1 => x.iter().map(|&xi| xi * 0.5).collect(), // Scale
            2 => x.iter().map(|&xi| xi.max(0.0)).collect(), // ReLU
            3 => x.iter().map(|&xi| xi.tanh()).collect(), // Tanh
            4 => x.iter().map(|&xi| 1.0 / (1.0 + (-xi).exp())).collect(), // Sigmoid
            _ => x.to_vec(),
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut node_outputs: Vec<Vec<f64>> = Vec::new();

        for node in 0..self.n_nodes {
            // Aggregate inputs from previous nodes
            let mut agg = if node_outputs.is_empty() {
                input.to_vec()
            } else {
                let last = node_outputs.last().unwrap();
                self.mixed_op(node, last)
            };

            node_outputs.push(agg);
        }

        node_outputs.last().unwrap().clone()
    }

    pub fn update_alphas(&mut self, gradients: &[Vec<f64>], learning_rate: f64) {
        for (alpha_row, grad_row) in self.alphas.iter_mut().zip(gradients.iter()) {
            for (alpha, grad) in alpha_row.iter_mut().zip(grad_row.iter()) {
                *alpha -= learning_rate * grad;
            }
        }
    }

    pub fn get_architecture(&self) -> Architecture {
        let mut arch = Architecture::new(self.n_nodes);
        for (i, alpha) in self.alphas.iter().enumerate() {
            arch.operations[i] = alpha.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
        }
        arch
    }
}

/// Evolution-based NAS.
pub struct EvolutionaryNAS {
    pub population_size: usize,
    pub n_layers: usize,
    pub n_ops: usize,
    pub mutation_rate: f64,
    pub tournament_size: usize,
    seed: u64,
}

impl EvolutionaryNAS {
    pub fn new(population_size: usize, n_layers: usize, n_ops: usize) -> Self {
        Self {
            population_size, n_layers, n_ops, mutation_rate: 0.2, tournament_size: 3,
            seed: 42,
        }
    }

    pub fn evolve<F>(&mut self, fitness_fn: F, n_generations: usize) -> Architecture
    where
        F: Fn(&Architecture) -> f64,
    {
        // Initialize population
        let mut population: Vec<Architecture> = (0..self.population_size)
            .map(|i| Architecture::random(self.n_layers, self.n_ops, self.seed + i as u64))
            .collect();

        let mut best_arch = population[0].clone();
        let mut best_fitness = f64::NEG_INFINITY;

        for gen in 0..n_generations {
            // Evaluate fitness
            let fitnesses: Vec<f64> = population.iter().map(|arch| fitness_fn(arch)).collect();

            // Track best
            for (arch, &fit) in population.iter().zip(fitnesses.iter()) {
                if fit > best_fitness {
                    best_fitness = fit;
                    best_arch = arch.clone();
                }
            }

            // Selection and reproduction
            let mut new_population = Vec::new();

            // Elitism: keep best
            let best_idx = fitnesses.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            new_population.push(population[best_idx].clone());

            while new_population.len() < self.population_size {
                // Tournament selection
                let parent1 = self.tournament_select(&population, &fitnesses);
                let parent2 = self.tournament_select(&population, &fitnesses);

                // Crossover
                let child = self.crossover(&parent1, &parent2);

                // Mutation
                let child = if self.pseudo_rand() < self.mutation_rate {
                    child.mutate(self.n_ops, self.seed)
                } else {
                    child
                };

                new_population.push(child);
            }

            population = new_population;
        }

        best_arch
    }

    fn tournament_select(&mut self, population: &[Architecture], fitnesses: &[f64]) -> Architecture {
        let mut best_idx = 0;
        let mut best_fitness = f64::NEG_INFINITY;

        for _ in 0..self.tournament_size {
            let idx = (self.pseudo_rand() * population.len() as f64) as usize % population.len();
            if fitnesses[idx] > best_fitness {
                best_fitness = fitnesses[idx];
                best_idx = idx;
            }
        }

        population[best_idx].clone()
    }

    fn crossover(&self, parent1: &Architecture, parent2: &Architecture) -> Architecture {
        let mut child = parent1.clone();
        let crossover_point = (self.pseudo_rand() * self.n_layers as f64) as usize % self.n_layers;

        for i in crossover_point..self.n_layers {
            child.operations[i] = parent2.operations[i];
            child.connections[i] = parent2.connections[i];
        }

        child
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Weight sharing NAS (one-shot).
pub struct WeightSharingNAS {
    pub n_ops: usize,
    pub shared_weights: Vec<Vec<Vec<f64>>>,
    pub supernet: Vec<Vec<f64>>,
}

impl WeightSharingNAS {
    pub fn new(n_ops: usize, input_dim: usize, hidden_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / input_dim as f64).sqrt();

        Self {
            n_ops,
            shared_weights: (0..n_ops).map(|_| {
                (0..hidden_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect()
            }).collect(),
            supernet: (0..hidden_dim).map(|_| (0..input_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn forward_with_arch(&self, x: &[f64], arch: &Architecture) -> Vec<f64> {
        let mut current = x.to_vec();

        for layer in 0..arch.n_layers {
            let op = arch.operations[layer];
            let weights = &self.shared_weights[op];

            current = weights.iter().map(|w| {
                w.iter().zip(current.iter()).map(|(wi, xi)| wi * xi).sum::<f64>().tanh()
            }).collect();
        }

        current
    }

    pub fn train_step(&mut self, x: &[f64], y: &[f64], arch: &Architecture, learning_rate: f64) -> f64 {
        let pred = self.forward_with_arch(x, arch);
        let loss: f64 = pred.iter().zip(y.iter()).map(|(p, y)| (p - y).powi(2)).sum();

        // Simplified gradient update on shared weights
        let error: Vec<f64> = pred.iter().zip(y.iter()).map(|(p, y)| p - y).collect();
        let op = arch.operations[0]; // Update first op's weights
        for (i, w_row) in self.shared_weights[op].iter_mut().enumerate() {
            for (j, w) in w_row.iter_mut().enumerate() {
                *w -= learning_rate * 2.0 * error[i.min(error.len() - 1)] * x[j.min(x.len() - 1)];
            }
        }

        loss
    }
}

/// ProxylessNAS: hardware-aware NAS.
pub struct ProxylessNAS {
    pub n_ops: usize,
    pub alphas: Vec<Vec<f64>>,
    pub latency_table: Vec<f64>, // Latency for each operation
    pub target_latency: f64,
}

impl ProxylessNAS {
    pub fn new(n_ops: usize, n_layers: usize, latency_table: Vec<f64>, target_latency: f64) -> Self {
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * 0.01
        };

        Self {
            n_ops,
            alphas: (0..n_layers).map(|_| (0..n_ops).map(|_| rand()).collect()).collect(),
            latency_table,
            target_latency,
        }
    }

    /// Expected latency given current architecture weights.
    pub fn expected_latency(&self) -> f64 {
        self.alphas.iter().map(|alpha| {
            let probs = softmax(alpha);
            probs.iter().zip(self.latency_table.iter()).map(|(p, l)| p * l).sum::<f64>()
        }).sum()
    }

    /// Hardware-aware loss: accuracy loss + lambda * latency penalty.
    pub fn loss(&self, accuracy_loss: f64, lambda: f64) -> f64 {
        let latency = self.expected_latency();
        let latency_penalty = lambda * (latency - self.target_latency).max(0.0).powi(2);
        accuracy_loss + latency_penalty
    }

    pub fn get_architecture(&self) -> Vec<usize> {
        self.alphas.iter().map(|alpha| {
            alpha.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap()
        }).collect()
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_darts() {
        let darts = DARTS::new(4, 5, 8);
        let input = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let output = darts.forward(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_evolutionary_nas() {
        let mut nas = EvolutionaryNAS::new(10, 4, 5);
        let fitness_fn = |arch: &Architecture| {
            arch.operations.iter().map(|&o| o as f64).sum::<f64>()
        };
        let best = nas.evolve(fitness_fn, 5);
        assert_eq!(best.n_layers, 4);
    }

    #[test]
    fn test_architecture_mutation() {
        let arch = Architecture::random(4, 5, 42);
        let mutated = arch.mutate(5, 43);
        // At least one thing should be different
        assert!(arch.operations != mutated.operations || arch.connections != mutated.connections);
    }
}
