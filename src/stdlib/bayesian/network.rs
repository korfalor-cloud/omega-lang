/// Bayesian network: directed acyclic graph of probabilistic variables.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct BayesianNetwork {
    nodes: Vec<Node>,
    adjacency: HashMap<usize, Vec<usize>>, // parent -> children
    name_to_id: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub name: String,
    pub states: Vec<String>,
    pub parents: Vec<usize>,
    pub cpt: CPT,
}

#[derive(Debug, Clone)]
pub enum CPT {
    /// For root nodes: probability of each state.
    Root(Vec<f64>),
    /// For child nodes: conditional probabilities indexed by parent state combination.
    Conditional {
        /// Each row is [parent_state_combo] -> probabilities for each state.
        table: Vec<Vec<f64>>,
        /// Parent state counts for indexing.
        parent_state_counts: Vec<usize>,
    },
}

impl BayesianNetwork {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            adjacency: HashMap::new(),
            name_to_id: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, name: &str, states: &[&str]) -> usize {
        let id = self.nodes.len();
        self.name_to_id.insert(name.to_string(), id);
        self.nodes.push(Node {
            id,
            name: name.to_string(),
            states: states.iter().map(|s| s.to_string()).collect(),
            parents: Vec::new(),
            cpt: CPT::Root(vec![1.0 / states.len() as f64; states.len()]),
        });
        id
    }

    pub fn add_edge(&mut self, parent: usize, child: usize) {
        self.adjacency.entry(parent).or_insert_with(Vec::new).push(child);
        self.nodes[child].parents.push(parent);
    }

    pub fn set_root_probabilities(&mut self, node: usize, probs: &[f64]) -> Result<(), String> {
        if let CPT::Root(_) = &self.nodes[node].cpt {
            if probs.len() != self.nodes[node].states.len() {
                return Err("Probability count must match state count".to_string());
            }
            let sum: f64 = probs.iter().sum();
            if (sum - 1.0).abs() > 0.01 {
                return Err(format!("Probabilities must sum to 1.0, got {}", sum));
            }
            self.nodes[node].cpt = CPT::Root(probs.to_vec());
            Ok(())
        } else {
            Err("Node is not a root node".to_string())
        }
    }

    pub fn set_conditional_probabilities(&mut self, node: usize, table: Vec<Vec<f64>>) -> Result<(), String> {
        let parents = self.nodes[node].parents.clone();
        if parents.is_empty() {
            return Err("Node has no parents".to_string());
        }

        let parent_state_counts: Vec<usize> = parents.iter()
            .map(|&p| self.nodes[p].states.len())
            .collect();

        let expected_rows: usize = parent_state_counts.iter().product();
        let expected_cols = self.nodes[node].states.len();

        if table.len() != expected_rows {
            return Err(format!("Expected {} rows, got {}", expected_rows, table.len()));
        }

        for (i, row) in table.iter().enumerate() {
            if row.len() != expected_cols {
                return Err(format!("Row {} expected {} columns, got {}", i, expected_cols, row.len()));
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 0.01 {
                return Err(format!("Row {} probabilities must sum to 1.0, got {}", i, sum));
            }
        }

        self.nodes[node].cpt = CPT::Conditional { table, parent_state_counts };
        Ok(())
    }

    /// Get the probability of a node being in a specific state given parent states.
    pub fn get_probability(&self, node: usize, state: usize, parent_states: &[usize]) -> f64 {
        match &self.nodes[node].cpt {
            CPT::Root(probs) => probs[state],
            CPT::Conditional { table, parent_state_counts } => {
                let index = self.compute_cpt_index(parent_states, parent_state_counts);
                table[index][state]
            }
        }
    }

    fn compute_cpt_index(&self, parent_states: &[usize], parent_state_counts: &[usize]) -> usize {
        let mut index = 0;
        let mut multiplier = 1;
        for i in (0..parent_states.len()).rev() {
            index += parent_states[i] * multiplier;
            multiplier *= parent_state_counts[i];
        }
        index
    }

    /// Enumerate all nodes.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn get_node(&self, name: &str) -> Option<&Node> {
        self.name_to_id.get(name).map(|&id| &self.nodes[id])
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get children of a node.
    pub fn children(&self, node: usize) -> &[usize] {
        self.adjacency.get(&node).map_or(&[], |v| v.as_slice())
    }

    /// Check if the network is a valid DAG (no cycles).
    pub fn is_valid(&self) -> bool {
        let mut visited = vec![false; self.nodes.len()];
        let mut rec_stack = vec![false; self.nodes.len()];

        for node in 0..self.nodes.len() {
            if !visited[node] {
                if self.has_cycle_dfs(node, &mut visited, &mut rec_stack) {
                    return false;
                }
            }
        }
        true
    }

    fn has_cycle_dfs(&self, node: usize, visited: &mut [bool], rec_stack: &mut [bool]) -> bool {
        visited[node] = true;
        rec_stack[node] = true;

        if let Some(children) = self.adjacency.get(&node) {
            for &child in children {
                if !visited[child] {
                    if self.has_cycle_dfs(child, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack[child] {
                    return true;
                }
            }
        }

        rec_stack[node] = false;
        false
    }

    /// Topological ordering of nodes.
    pub fn topological_order(&self) -> Vec<usize> {
        let mut visited = vec![false; self.nodes.len()];
        let mut order = Vec::new();

        for node in 0..self.nodes.len() {
            if !visited[node] {
                self.topo_dfs(node, &mut visited, &mut order);
            }
        }

        order.reverse();
        order
    }

    fn topo_dfs(&self, node: usize, visited: &mut [bool], order: &mut Vec<usize>) {
        visited[node] = true;
        if let Some(children) = self.adjacency.get(&node) {
            for &child in children {
                if !visited[child] {
                    self.topo_dfs(child, visited, order);
                }
            }
        }
        order.push(node);
    }

    /// Sample a value from the network using forward sampling.
    pub fn forward_sample(&self) -> HashMap<usize, usize> {
        let mut seed: u64 = 42;
        let order = self.topological_order();
        let mut assignments = HashMap::new();

        for &node_id in &order {
            let node = &self.nodes[node_id];
            let parent_states: Vec<usize> = node.parents.iter()
                .map(|&p| *assignments.get(&p).unwrap_or(&0))
                .collect();

            let probs = match &node.cpt {
                CPT::Root(p) => p.clone(),
                CPT::Conditional { table, parent_state_counts } => {
                    let idx = self.compute_cpt_index(&parent_states, parent_state_counts);
                    table[idx].clone()
                }
            };

            // Sample from categorical distribution
            let r = pseudo_rand(&mut seed);
            let mut cumulative = 0.0;
            let mut chosen = 0;
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r < cumulative {
                    chosen = i;
                    break;
                }
            }

            assignments.insert(node_id, chosen);
        }

        assignments
    }

    /// Exact inference by enumeration.
    pub fn query(&self, query_node: usize, evidence: &HashMap<usize, usize>) -> Vec<f64> {
        let num_states = self.nodes[query_node].states.len();
        let mut probs = vec![0.0; num_states];

        // Enumerate all possible assignments
        let hidden_nodes: Vec<usize> = (0..self.nodes.len())
            .filter(|&n| n != query_node && !evidence.contains_key(&n))
            .collect();

        let all_assignments = self.enumerate_assignments(&hidden_nodes);

        for state in 0..num_states {
            for assignment in &all_assignments {
                let mut full_assignment = evidence.clone();
                full_assignment.insert(query_node, state);
                for (i, &h) in hidden_nodes.iter().enumerate() {
                    full_assignment.insert(h, assignment[i]);
                }

                let joint = self.joint_probability(&full_assignment);
                probs[state] += joint;
            }
        }

        // Normalize
        let total: f64 = probs.iter().sum();
        if total > 0.0 {
            for p in &mut probs {
                *p /= total;
            }
        }

        probs
    }

    fn enumerate_assignments(&self, nodes: &[usize]) -> Vec<Vec<usize>> {
        if nodes.is_empty() {
            return vec![Vec::new()];
        }

        let first = nodes[0];
        let rest = &nodes[1..];
        let rest_assignments = self.enumerate_assignments(rest);

        let mut result = Vec::new();
        for state in 0..self.nodes[first].states.len() {
            for mut rest_assign in rest_assignments.clone() {
                rest_assign.insert(0, state);
                result.push(rest_assign);
            }
        }
        result
    }

    fn joint_probability(&self, assignment: &HashMap<usize, usize>) -> f64 {
        let order = self.topological_order();
        let mut prob = 1.0;

        for &node_id in &order {
            let node = &self.nodes[node_id];
            let state = match assignment.get(&node_id) {
                Some(&s) => s,
                None => return 0.0,
            };

            let parent_states: Vec<usize> = node.parents.iter()
                .map(|&p| *assignment.get(&p).unwrap_or(&0))
                .collect();

            let p = self.get_probability(node_id, state, &parent_states);
            prob *= p;
        }

        prob
    }
}

fn pseudo_rand(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / (1u64 << 31) as f64
}

impl Default for BayesianNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Markov Chain: state transitions with probabilities.
#[derive(Debug)]
pub struct MarkovChain {
    states: Vec<String>,
    transition: Vec<Vec<f64>>,
    initial: Vec<f64>,
    name_to_id: HashMap<String, usize>,
}

impl MarkovChain {
    pub fn new(states: &[&str], initial: &[f64], transition: Vec<Vec<f64>>) -> Result<Self, String> {
        let n = states.len();
        if initial.len() != n {
            return Err("Initial distribution length must match state count".to_string());
        }
        if transition.len() != n {
            return Err("Transition matrix must be NxN".to_string());
        }
        for (i, row) in transition.iter().enumerate() {
            if row.len() != n {
                return Err(format!("Transition matrix row {} must have {} columns", i, n));
            }
        }

        let mut name_to_id = HashMap::new();
        for (i, &s) in states.iter().enumerate() {
            name_to_id.insert(s.to_string(), i);
        }

        Ok(Self {
            states: states.iter().map(|s| s.to_string()).collect(),
            transition,
            initial: initial.to_vec(),
            name_to_id,
        })
    }

    pub fn step(&self, current_state: usize) -> usize {
        let mut seed: u64 = current_state as u64;
        let r = pseudo_rand(&mut seed);
        let mut cumulative = 0.0;
        for (i, &p) in self.transition[current_state].iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return i;
            }
        }
        self.states.len() - 1
    }

    pub fn simulate(&self, start: usize, steps: usize) -> Vec<usize> {
        let mut path = vec![start];
        let mut current = start;
        for _ in 0..steps {
            current = self.step(current);
            path.push(current);
        }
        path
    }

    pub fn stationary_distribution(&self, iterations: usize) -> Vec<f64> {
        let n = self.states.len();
        let mut dist = self.initial.clone();

        for _ in 0..iterations {
            let mut new_dist = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_dist[j] += dist[i] * self.transition[i][j];
                }
            }
            dist = new_dist;
        }

        dist
    }

    pub fn state_name(&self, id: usize) -> &str {
        &self.states[id]
    }

    pub fn get_state(&self, name: &str) -> Option<usize> {
        self.name_to_id.get(name).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_network() {
        let mut bn = BayesianNetwork::new();
        let cloudy = bn.add_node("Cloudy", &["true", "false"]);
        let sprinkler = bn.add_node("Sprinkler", &["on", "off"]);
        let rain = bn.add_node("Rain", &["yes", "no"]);

        bn.add_edge(cloudy, sprinkler);
        bn.add_edge(cloudy, rain);

        bn.set_root_probabilities(cloudy, &[0.5, 0.5]).unwrap();
        bn.set_conditional_probabilities(sprinkler, vec![
            vec![0.1, 0.9], // cloudy=true -> P(sprinkler=on)=0.1
            vec![0.5, 0.5], // cloudy=false -> P(sprinkler=on)=0.5
        ]).unwrap();
        bn.set_conditional_probabilities(rain, vec![
            vec![0.8, 0.2], // cloudy=true -> P(rain=yes)=0.8
            vec![0.2, 0.8], // cloudy=false -> P(rain=yes)=0.2
        ]).unwrap();

        assert!(bn.is_valid());
        assert_eq!(bn.node_count(), 3);
    }

    #[test]
    fn test_markov_chain() {
        let mc = MarkovChain::new(
            &["Sunny", "Rainy"],
            &[0.6, 0.4],
            vec![
                vec![0.8, 0.2],
                vec![0.4, 0.6],
            ],
        ).unwrap();

        let stationary = mc.stationary_distribution(100);
        assert!((stationary[0] - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_forward_sample() {
        let mut bn = BayesianNetwork::new();
        let a = bn.add_node("A", &["0", "1"]);
        let b = bn.add_node("B", &["0", "1"]);
        bn.add_edge(a, b);
        bn.set_root_probabilities(a, &[0.5, 0.5]).unwrap();
        bn.set_conditional_probabilities(b, vec![vec![0.9, 0.1], vec![0.1, 0.9]]).unwrap();

        let sample = bn.forward_sample();
        assert!(sample.contains_key(&a));
        assert!(sample.contains_key(&b));
    }
}
