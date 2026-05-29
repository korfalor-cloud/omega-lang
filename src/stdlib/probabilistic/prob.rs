/// Comprehensive probabilistic programming: Bayesian inference (MCMC, VI),
/// probabilistic graphical models, belief propagation, and expectation propagation.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Distributions
// ---------------------------------------------------------------------------

/// A univariate Gaussian distribution.
#[derive(Clone, Debug)]
pub struct Gaussian {
    pub mean: f64,
    pub variance: f64,
}

impl Gaussian {
    pub fn new(mean: f64, variance: f64) -> Self {
        Self { mean, variance }
    }

    pub fn log_prob(&self, x: f64) -> f64 {
        let diff = x - self.mean;
        -0.5 * diff * diff / self.variance - 0.5 * (2.0 * std::f64::consts::PI * self.variance).ln()
    }

    pub fn sample(&self, rng: &mut u64) -> f64 {
        let u1 = pseudo_rand(rng).max(1e-10);
        let u2 = pseudo_rand(rng);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        self.mean + self.variance.sqrt() * z
    }

    /// Precision (1/variance).
    pub fn precision(&self) -> f64 {
        1.0 / self.variance
    }

    /// Natural parameters: eta1 = mean/variance, eta2 = -1/(2*variance).
    pub fn natural_params(&self) -> (f64, f64) {
        (self.mean / self.variance, -0.5 / self.variance)
    }

    /// Construct from natural parameters.
    pub fn from_natural(eta1: f64, eta2: f64) -> Self {
        let variance = -0.5 / eta2;
        let mean = eta1 * variance;
        Self { mean, variance }
    }

    /// KL(q || p) between two Gaussians.
    pub fn kl_divergence(&self, other: &Gaussian) -> f64 {
        let ratio = self.variance / other.variance;
        let mean_diff = self.mean - other.mean;
        0.5 * (ratio - 1.0 - ratio.ln() + mean_diff * mean_diff / other.variance)
    }
}

/// A categorical / discrete distribution over a finite set of outcomes.
#[derive(Clone, Debug)]
pub struct Categorical {
    pub probs: Vec<f64>,
}

impl Categorical {
    pub fn new(probs: Vec<f64>) -> Self {
        let total: f64 = probs.iter().sum();
        Self {
            probs: probs.iter().map(|p| p / total).collect(),
        }
    }

    pub fn log_prob(&self, index: usize) -> f64 {
        self.probs[index].max(1e-30).ln()
    }

    pub fn sample(&self, rng: &mut u64) -> usize {
        let u = pseudo_rand(rng);
        let mut cumulative = 0.0;
        for (i, &p) in self.probs.iter().enumerate() {
            cumulative += p;
            if u <= cumulative {
                return i;
            }
        }
        self.probs.len() - 1
    }

    /// Multiply element-wise with another Categorical (unnormalized).
    pub fn pointwise_mul(&self, other: &Categorical) -> Categorical {
        let unnormalized: Vec<f64> = self
            .probs
            .iter()
            .zip(other.probs.iter())
            .map(|(a, b)| a * b)
            .collect();
        Categorical::new(unnormalized)
    }
}

// ---------------------------------------------------------------------------
// Probabilistic Graphical Model (directed, discrete)
// ---------------------------------------------------------------------------

/// A node in a discrete Bayesian network.
#[derive(Clone, Debug)]
pub struct PgmNode {
    pub name: String,
    /// Conditional probability table: key = parent value assignments, value = distribution.
    pub cpt: HashMap<Vec<usize>, Categorical>,
    pub n_states: usize,
}

impl PgmNode {
    pub fn new(name: &str, n_states: usize) -> Self {
        Self {
            name: name.to_string(),
            cpt: HashMap::new(),
            n_states,
        }
    }

    pub fn set_cpt(&mut self, parent_values: Vec<usize>, probs: Vec<f64>) {
        self.cpt.insert(parent_values, Categorical::new(probs));
    }

    pub fn get_distribution(&self, parent_values: &[usize]) -> Option<&Categorical> {
        self.cpt.get(parent_values)
    }
}

/// A discrete Bayesian network (directed graphical model).
#[derive(Clone, Debug)]
pub struct BayesianNetwork {
    pub nodes: Vec<PgmNode>,
    /// adjacency: edges[i] = list of parent indices for node i
    pub edges: Vec<Vec<usize>>,
}

impl BayesianNetwork {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: PgmNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        self.edges.push(Vec::new());
        idx
    }

    pub fn add_edge(&mut self, parent: usize, child: usize) {
        self.edges[child].push(parent);
    }

    /// Sample from the joint distribution by ancestral sampling.
    pub fn sample(&self, rng: &mut u64) -> Vec<usize> {
        let n = self.nodes.len();
        let mut values = vec![0usize; n];

        // Process in topological order (assume nodes are added in topo order).
        for i in 0..n {
            let parent_vals: Vec<usize> = self.edges[i].iter().map(|&p| values[p]).collect();
            let dist = self.nodes[i].get_distribution(&parent_vals);
            if let Some(d) = dist {
                values[i] = d.sample(rng);
            }
        }
        values
    }

    /// Compute joint log-probability of a complete assignment.
    pub fn log_prob(&self, values: &[usize]) -> f64 {
        let mut lp = 0.0;
        for i in 0..self.nodes.len() {
            let parent_vals: Vec<usize> = self.edges[i].iter().map(|&p| values[p]).collect();
            if let Some(d) = self.nodes[i].get_distribution(&parent_vals) {
                lp += d.log_prob(values[i]);
            }
        }
        lp
    }
}

// ---------------------------------------------------------------------------
// Factor Graph and Belief Propagation
// ---------------------------------------------------------------------------

/// A factor in a factor graph: scope of variable indices and a table of values.
#[derive(Clone, Debug)]
pub struct Factor {
    pub scope: Vec<usize>,
    pub table: HashMap<Vec<usize>, f64>,
}

impl Factor {
    pub fn new(scope: Vec<usize>) -> Self {
        Self {
            scope,
            table: HashMap::new(),
        }
    }

    pub fn set(&mut self, assignment: Vec<usize>, value: f64) {
        self.table.insert(assignment, value);
    }

    pub fn get(&self, assignment: &[usize]) -> f64 {
        *self.table.get(assignment).unwrap_or(&0.0)
    }

    /// Marginalise out a variable by summing over its values.
    pub fn marginalize(&self, var: usize, n_states: usize) -> Factor {
        let pos = self.scope.iter().position(|&v| v == var);
        let mut new_scope = self.scope.clone();
        if let Some(p) = pos {
            new_scope.remove(p);
        }

        let mut result = Factor::new(new_scope);
        for (assignment, &value) in &self.table {
            if let Some(p) = pos {
                let mut new_assignment = assignment.clone();
                new_assignment.remove(p);
                let entry = result.table.entry(new_assignment).or_insert(0.0);
                *entry += value;
            }
        }
        result
    }

    /// Restrict a variable to a particular value.
    pub fn restrict(&self, var: usize, value: usize) -> Factor {
        let pos = self.scope.iter().position(|&v| v == var);
        let mut new_scope = self.scope.clone();
        if let Some(p) = pos {
            new_scope.remove(p);
        }

        let mut result = Factor::new(new_scope);
        for (assignment, &fv) in &self.table {
            if let Some(p) = pos {
                if assignment[p] == value {
                    let mut new_assignment = assignment.clone();
                    new_assignment.remove(p);
                    result.table.insert(new_assignment, fv);
                }
            }
        }
        result
    }

    /// Multiply two factors (join).
    pub fn multiply(&self, other: &Factor) -> Factor {
        // Union of scopes preserving order.
        let mut new_scope = self.scope.clone();
        for &v in &other.scope {
            if !new_scope.contains(&v) {
                new_scope.push(v);
            }
        }

        let mut result = Factor::new(new_scope.clone());
        let self_var_count = self.scope.len();
        let other_var_count = other.scope.len();

        // Enumerate all assignments of the combined scope.
        let dims: Vec<usize> = new_scope
            .iter()
            .map(|v| {
                self.table
                    .keys()
                    .chain(other.table.keys())
                    .filter_map(|k| {
                        let pos_self = self.scope.iter().position(|s| s == v);
                        let pos_other = other.scope.iter().position(|s| s == v);
                        let mut max_val = 0usize;
                        if let Some(p) = pos_self {
                            for a in self.table.keys() {
                                max_val = max_val.max(a[p]);
                            }
                        }
                        if let Some(p) = pos_other {
                            for a in other.table.keys() {
                                max_val = max_val.max(a[p]);
                            }
                        }
                        Some(max_val + 1)
                    })
                    .next()
                    .unwrap_or(1)
            })
            .collect();

        // Simple enumeration for small factors.
        let total_assignments: usize = dims.iter().product();
        for idx in 0..total_assignments {
            let mut assignment = vec![0usize; new_scope.len()];
            let mut tmp = idx;
            for j in (0..new_scope.len()).rev() {
                assignment[j] = tmp % dims[j];
                tmp /= dims[j];
            }

            // Build sub-assignments for self and other.
            let self_assign: Vec<usize> = self
                .scope
                .iter()
                .map(|v| {
                    let pos = new_scope.iter().position(|n| n == v).unwrap();
                    assignment[pos]
                })
                .collect();
            let other_assign: Vec<usize> = other
                .scope
                .iter()
                .map(|v| {
                    let pos = new_scope.iter().position(|n| n == v).unwrap();
                    assignment[pos]
                })
                .collect();

            let val = self.get(&self_assign) * other.get(&other_assign);
            if val != 0.0 {
                result.set(assignment, val);
            }
        }
        result
    }

    /// Divide by another factor (element-wise, on the shared scope).
    pub fn divide(&self, other: &Factor) -> Factor {
        let mut result = self.clone();
        for (assignment, value) in result.table.iter_mut() {
            let other_assign: Vec<usize> = other
                .scope
                .iter()
                .map(|v| {
                    let pos = self.scope.iter().position(|s| s == v).unwrap();
                    assignment[pos]
                })
                .collect();
            let denom = other.get(&other_assign);
            if denom > 1e-30 {
                *value /= denom;
            }
        }
        result
    }
}

/// Loopy belief propagation on a factor graph.
pub struct BeliefPropagation {
    pub factors: Vec<Factor>,
    pub n_vars: usize,
    pub var_states: Vec<usize>,
}

impl BeliefPropagation {
    pub fn new(n_vars: usize, var_states: Vec<usize>) -> Self {
        Self {
            factors: Vec::new(),
            n_vars,
            var_states,
        }
    }

    pub fn add_factor(&mut self, factor: Factor) {
        self.factors.push(factor);
    }

    /// Compute exact marginals via variable elimination (for small models).
    pub fn exact_marginals(&self) -> Vec<Categorical> {
        let mut result = Vec::new();

        for v in 0..self.n_vars {
            // Multiply all factors containing v, then marginalise out everything except v.
            let mut product: Option<Factor> = None;
            for f in &self.factors {
                if f.scope.contains(&v) {
                    product = Some(match product {
                        Some(p) => p.multiply(f),
                        None => f.clone(),
                    });
                }
            }

            if let Some(mut p) = product {
                // Marginalise out all variables except v.
                let vars_to_remove: Vec<usize> = p
                    .scope
                    .iter()
                    .filter(|&&var| var != v)
                    .copied()
                    .collect();
                for var in vars_to_remove {
                    p = p.marginalize(var, self.var_states[var]);
                }

                let total: f64 = p.table.values().sum();
                let probs: Vec<f64> = (0..self.var_states[v])
                    .map(|s| {
                        let val = p.table.get(&vec![s]).copied().unwrap_or(0.0);
                        if total > 1e-30 {
                            val / total
                        } else {
                            1.0 / self.var_states[v] as f64
                        }
                    })
                    .collect();
                result.push(Categorical::new(probs));
            } else {
                // No factor: uniform.
                result.push(Categorical::new(vec![1.0; self.var_states[v]]));
            }
        }
        result
    }

    /// Loopy BP message passing (fixed iterations).
    pub fn loopy_bp(&self, max_iter: usize) -> Vec<Categorical> {
        // Simplified: for each variable, compute the product of all incoming messages
        // from adjacent factors, then normalise.
        let mut messages: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

        // Initialise messages to uniform.
        for (fi, f) in self.factors.iter().enumerate() {
            for &v in &f.scope {
                let n = self.var_states[v];
                messages.insert((fi, v), vec![1.0 / n as f64; n]);
                messages.insert((v, fi), vec![1.0 / n as f64; n]);
            }
        }

        for _iter in 0..max_iter {
            // Variable -> Factor messages: product of all incoming factor messages except f.
            for v in 0..self.n_vars {
                let adjacent_factors: Vec<usize> = self
                    .factors
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| f.scope.contains(&v))
                    .map(|(i, _)| i)
                    .collect();

                for &fi in &adjacent_factors {
                    let mut msg = vec![1.0; self.var_states[v]];
                    for &fj in &adjacent_factors {
                        if fj != fi {
                            if let Some(m) = messages.get(&(fj, v)) {
                                for s in 0..self.var_states[v] {
                                    msg[s] *= m[s];
                            }
                        }
                    }
                    }
                    // Normalise
                    let total: f64 = msg.iter().sum();
                    if total > 1e-30 {
                        for m in msg.iter_mut() {
                            *m /= total;
                        }
                    }
                    messages.insert((v, fi), msg);
                }
            }

            // Factor -> Variable messages: marginalise factor * product of incoming var msgs except v.
            for (fi, f) in self.factors.iter().enumerate() {
                for &v in &f.scope {
                    let mut msg = vec![0.0; self.var_states[v]];
                    for (assignment, &fval) in &f.table {
                        let v_pos = f.scope.iter().position(|&x| x == v).unwrap();
                        let s = assignment[v_pos];
                        let mut weight = fval;
                        for (j, &other_v) in f.scope.iter().enumerate() {
                            if other_v != v {
                                if let Some(m) = messages.get(&(other_v, fi)) {
                                    weight *= m[assignment[j]];
                                }
                            }
                        }
                        msg[s] += weight;
                    }
                    let total: f64 = msg.iter().sum();
                    if total > 1e-30 {
                        for m in msg.iter_mut() {
                            *m /= total;
                        }
                    }
                    messages.insert((fi, v), msg);
                }
            }
        }

        // Compute beliefs.
        let mut beliefs = Vec::new();
        for v in 0..self.n_vars {
            let mut belief = vec![1.0; self.var_states[v]];
            for (fi, f) in self.factors.iter().enumerate() {
                if f.scope.contains(&v) {
                    if let Some(m) = messages.get(&(fi, v)) {
                        for s in 0..self.var_states[v] {
                            belief[s] *= m[s];
                        }
                    }
                }
            }
            let total: f64 = belief.iter().sum();
            if total > 1e-30 {
                for b in belief.iter_mut() {
                    *b /= total;
                }
            }
            beliefs.push(Categorical::new(belief));
        }
        beliefs
    }
}

// ---------------------------------------------------------------------------
// Expectation Propagation
// ---------------------------------------------------------------------------

/// Approximate inference via Expectation Propagation for a Gaussian model.
/// P(x) ~ prod_i f_i(x), each f_i is approximated by a Gaussian cavity.
pub struct ExpectationPropagation {
    pub n_dims: usize,
    /// Site approximations (Gaussian natural parameters per site).
    pub site_eta1: Vec<Vec<f64>>,
    pub site_eta2: Vec<Vec<f64>>,
    /// Prior as Gaussian natural parameters.
    pub prior_eta1: Vec<f64>,
    pub prior_eta2: Vec<f64>,
}

impl ExpectationPropagation {
    pub fn new(n_dims: usize, prior: &Gaussian) -> Self {
        let (eta1, eta2) = prior.natural_params();
        Self {
            n_dims,
            site_eta1: Vec::new(),
            site_eta2: Vec::new(),
            prior_eta1: vec![eta1; n_dims],
            prior_eta2: vec![eta2; n_dims],
        }
    }

    pub fn add_site(&mut self) {
        self.site_eta1.push(vec![0.0; self.n_dims]);
        self.site_eta2.push(vec![0.0; self.n_dims]);
    }

    /// Compute the current posterior (product of all sites + prior).
    pub fn posterior(&self) -> Vec<Gaussian> {
        let mut eta1 = self.prior_eta1.clone();
        let mut eta2 = self.prior_eta2.clone();

        for site in &self.site_eta1 {
            for (e, s) in eta1.iter_mut().zip(site.iter()) {
                *e += s;
            }
        }
        for site in &self.site_eta2 {
            for (e, s) in eta2.iter_mut().zip(site.iter()) {
                *e += s;
            }
        }

        (0..self.n_dims)
            .map(|d| Gaussian::from_natural(eta1[d], eta2[d]))
            .collect()
    }

    /// Compute cavity distribution for site i (posterior / site_i).
    pub fn cavity(&self, site_idx: usize) -> Vec<Gaussian> {
        let mut eta1 = self.prior_eta1.clone();
        let mut eta2 = self.prior_eta2.clone();

        for (i, site) in self.site_eta1.iter().enumerate() {
            if i != site_idx {
                for (e, s) in eta1.iter_mut().zip(site.iter()) {
                    *e += s;
                }
            }
        }
        for (i, site) in self.site_eta2.iter().enumerate() {
            if i != site_idx {
                for (e, s) in eta2.iter_mut().zip(site.iter()) {
                    *e += s;
                }
            }
        }

        (0..self.n_dims)
            .map(|d| Gaussian::from_natural(eta1[d], eta2[d]))
            .collect()
    }

    /// Update a single site given a moment-matching target.
    /// `target_moment` = (new_mean, new_variance) from the tilted distribution.
    pub fn update_site(&mut self, site_idx: usize, target_moments: &[(f64, f64)]) {
        let cavity = self.cavity(site_idx);
        for d in 0..self.n_dims {
            let cav = &cavity[d];
            let (t_mean, t_var) = target_moments[d];

            // New site natural params = target - cavity
            let (t_eta1, t_eta2) = (t_mean / t_var, -0.5 / t_var);
            let (c_eta1, c_eta2) = cav.natural_params();
            self.site_eta1[site_idx][d] = t_eta1 - c_eta1;
            self.site_eta2[site_idx][d] = t_eta2 - c_eta2;
        }
    }
}

// ---------------------------------------------------------------------------
// MCMC: Metropolis-Hastings
// ---------------------------------------------------------------------------

/// A generic Metropolis-Hastings sampler with a Gaussian proposal.
pub struct MetropolisHastings {
    pub dim: usize,
    pub proposal_var: f64,
    seed: u64,
}

impl MetropolisHastings {
    pub fn new(dim: usize, proposal_var: f64) -> Self {
        Self {
            dim,
            proposal_var,
            seed: 42,
        }
    }

    pub fn sample<F: Fn(&[f64]) -> f64>(
        &mut self,
        initial: &[f64],
        log_target: F,
        n_samples: usize,
    ) -> Vec<Vec<f64>> {
        let mut current = initial.to_vec();
        let mut lp_current = log_target(&current);
        let mut samples = Vec::new();

        for _ in 0..n_samples {
            // Propose
            let proposal: Vec<f64> = current
                .iter()
                .map(|&c| c + self.gaussian() * self.proposal_var.sqrt())
                .collect();
            let lp_proposal = log_target(&proposal);

            // Accept / reject
            let log_alpha = lp_proposal - lp_current;
            if pseudo_rand(&mut self.seed) < log_alpha.exp().min(1.0) {
                current = proposal;
                lp_current = lp_proposal;
            }
            samples.push(current.clone());
        }
        samples
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = pseudo_rand(&mut self.seed).max(1e-10);
        let u2 = pseudo_rand(&mut self.seed);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Variational Inference helpers
// ---------------------------------------------------------------------------

/// Mean-field variational Bayes update for a conjugate Gaussian model.
/// Returns updated (mean, variance) for each dimension.
pub fn mean_field_update(
    prior: &[Gaussian],
    data: &[f64],
    likelihood_var: f64,
) -> Vec<Gaussian> {
    let n = data.len() as f64;
    let data_mean = data.iter().sum::<f64>() / n;

    prior
        .iter()
        .map(|p| {
            let post_prec = p.precision() + n / likelihood_var;
            let post_var = 1.0 / post_prec;
            let post_mean = post_var * (p.mean * p.precision() + n * data_mean / likelihood_var);
            Gaussian::new(post_mean, post_var)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random number generator (xorshift64).
fn pseudo_rand(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_log_prob() {
        let g = Gaussian::new(0.0, 1.0);
        let lp = g.log_prob(0.0);
        assert!((lp - (-0.5 * (2.0 * std::f64::consts::PI).ln())).abs() < 1e-10);

        let g2 = Gaussian::new(3.0, 0.25);
        assert!(g2.log_prob(3.0) > g2.log_prob(0.0));
    }

    #[test]
    fn test_gaussian_natural_params_roundtrip() {
        let g = Gaussian::new(2.0, 4.0);
        let (eta1, eta2) = g.natural_params();
        let g2 = Gaussian::from_natural(eta1, eta2);
        assert!((g.mean - g2.mean).abs() < 1e-10);
        assert!((g.variance - g2.variance).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_kl_divergence() {
        let g = Gaussian::new(0.0, 1.0);
        let kl = g.kl_divergence(&g);
        assert!(kl.abs() < 1e-10);

        let g2 = Gaussian::new(1.0, 1.0);
        assert!(g.kl_divergence(&g2) > 0.0);
    }

    #[test]
    fn test_categorical() {
        let c = Categorical::new(vec![1.0, 2.0, 1.0]);
        assert!((c.probs[0] - 0.25).abs() < 1e-10);
        assert!((c.probs[1] - 0.5).abs() < 1e-10);

        let mut rng = 42u64;
        let s = c.sample(&mut rng);
        assert!(s < 3);
    }

    #[test]
    fn test_categorical_pointwise_mul() {
        let a = Categorical::new(vec![0.5, 0.5]);
        let b = Categorical::new(vec![0.8, 0.2]);
        let c = a.pointwise_mul(&b);
        // Unnormalised: [0.4, 0.1], normalised: [0.8, 0.2]
        assert!((c.probs[0] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_bayesian_network_sample_and_logprob() {
        let mut bn = BayesianNetwork::new();

        // Node 0: A (2 states), no parents.
        let mut a = PgmNode::new("A", 2);
        a.set_cpt(vec![], vec![0.4, 0.6]);
        bn.add_node(a);

        // Node 1: B (2 states), parent A.
        let mut b = PgmNode::new("B", 2);
        b.set_cpt(vec![0], vec![0.9, 0.1]); // A=0 -> B probs
        b.set_cpt(vec![1], vec![0.2, 0.8]); // A=1 -> B probs
        let b_idx = bn.add_node(b);
        bn.add_edge(0, b_idx);

        let mut rng = 123u64;
        let sample = bn.sample(&mut rng);
        assert_eq!(sample.len(), 2);
        assert!(sample[0] < 2);
        assert!(sample[1] < 2);

        let lp = bn.log_prob(&[0, 0]);
        let expected = (0.4_f64).ln() + (0.9_f64).ln();
        assert!((lp - expected).abs() < 1e-10);
    }

    #[test]
    fn test_factor_marginalize() {
        let mut f = Factor::new(vec![0, 1]);
        f.set(vec![0, 0], 0.1);
        f.set(vec![0, 1], 0.2);
        f.set(vec![1, 0], 0.3);
        f.set(vec![1, 1], 0.4);

        let marg = f.marginalize(1, 2);
        // sum over var 1: f(0,0)+f(0,1)=0.3, f(1,0)+f(1,1)=0.7
        assert!((marg.get(&[0]) - 0.3).abs() < 1e-10);
        assert!((marg.get(&[1]) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_factor_multiply() {
        let mut f1 = Factor::new(vec![0]);
        f1.set(vec![0], 2.0);
        f1.set(vec![1], 3.0);

        let mut f2 = Factor::new(vec![1]);
        f2.set(vec![0], 5.0);
        f2.set(vec![1], 7.0);

        let prod = f1.multiply(&f2);
        assert!((prod.get(&[0, 0]) - 10.0).abs() < 1e-10);
        assert!((prod.get(&[0, 1]) - 14.0).abs() < 1e-10);
        assert!((prod.get(&[1, 0]) - 15.0).abs() < 1e-10);
        assert!((prod.get(&[1, 1]) - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_belief_propagation_exact() {
        // Two binary variables, one factor connecting them.
        let mut bp = BeliefPropagation::new(2, vec![2, 2]);

        let mut f = Factor::new(vec![0, 1]);
        f.set(vec![0, 0], 0.9);
        f.set(vec![0, 1], 0.1);
        f.set(vec![1, 0], 0.1);
        f.set(vec![1, 1], 0.9);
        bp.add_factor(f);

        let marginals = bp.exact_marginals();
        // Both should be symmetric: P(0)=P(1)=0.5
        assert!((marginals[0].probs[0] - 0.5).abs() < 1e-10);
        assert!((marginals[1].probs[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_belief_propagation_asymmetric() {
        // Two binary variables, prefer (0,0) and (1,1) but asymmetrically.
        let mut bp = BeliefPropagation::new(2, vec![2, 2]);

        let mut f = Factor::new(vec![0, 1]);
        f.set(vec![0, 0], 8.0);
        f.set(vec![0, 1], 1.0);
        f.set(vec![1, 0], 1.0);
        f.set(vec![1, 1], 0.5);
        bp.add_factor(f);

        // Add a prior on variable 0 favouring state 0.
        let mut prior = Factor::new(vec![0]);
        prior.set(vec![0], 3.0);
        prior.set(vec![1], 1.0);
        bp.add_factor(prior);

        let marginals = bp.exact_marginals();
        // Variable 0 should lean towards state 0.
        assert!(marginals[0].probs[0] > marginals[0].probs[1]);
    }

    #[test]
    fn test_loopy_bp() {
        let mut bp = BeliefPropagation::new(3, vec![2, 2, 2]);

        // Chain: 0 -- 1 -- 2
        let mut f01 = Factor::new(vec![0, 1]);
        f01.set(vec![0, 0], 0.9);
        f01.set(vec![0, 1], 0.1);
        f01.set(vec![1, 0], 0.1);
        f01.set(vec![1, 1], 0.9);
        bp.add_factor(f01);

        let mut f12 = Factor::new(vec![1, 2]);
        f12.set(vec![0, 0], 0.7);
        f12.set(vec![0, 1], 0.3);
        f12.set(vec![1, 0], 0.3);
        f12.set(vec![1, 1], 0.7);
        bp.add_factor(f12);

        let marginals = bp.loopy_bp(10);
        assert_eq!(marginals.len(), 3);
        // All should be valid distributions.
        for m in &marginals {
            let total: f64 = m.probs.iter().sum();
            assert!((total - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_expectation_propagation() {
        let prior = Gaussian::new(0.0, 10.0);
        let mut ep = ExpectationPropagation::new(2, &prior);

        ep.add_site();
        ep.add_site();

        // Simulate site updates with target moments that shift the posterior.
        ep.update_site(0, &[(1.0, 1.0), (0.5, 2.0)]);
        ep.update_site(1, &[(2.0, 0.5), (1.5, 1.0)]);

        let post = ep.posterior();
        assert_eq!(post.len(), 2);
        // Posterior mean should be shifted towards the targets.
        assert!(post[0].mean > 0.0);
        assert!(post[1].mean > 0.0);
    }

    #[test]
    fn test_metropolis_hastings() {
        // Sample from a standard Gaussian.
        let mut mh = MetropolisHastings::new(1, 0.5);
        let log_target = |x: &[f64]| -0.5 * x[0] * x[0];
        let samples = mh.sample(&[0.0], log_target, 5000);

        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.5, "mean={}", mean);
    }

    #[test]
    fn test_mean_field_update() {
        let prior = vec![Gaussian::new(0.0, 1.0); 1];
        let data: Vec<f64> = vec![5.0, 5.1, 4.9, 5.0, 5.05];
        let posterior = mean_field_update(&prior, &data, 1.0);

        // Posterior mean should be close to data mean (~5.0).
        assert!((posterior[0].mean - 5.0).abs() < 0.2);
        // Posterior variance should be smaller than prior.
        assert!(posterior[0].variance < 1.0);
    }

    #[test]
    fn test_factor_restrict() {
        let mut f = Factor::new(vec![0, 1]);
        f.set(vec![0, 0], 1.0);
        f.set(vec![0, 1], 2.0);
        f.set(vec![1, 0], 3.0);
        f.set(vec![1, 1], 4.0);

        let restricted = f.restrict(0, 1);
        assert_eq!(restricted.scope.len(), 1);
        assert!((restricted.get(&[0]) - 3.0).abs() < 1e-10);
        assert!((restricted.get(&[1]) - 4.0).abs() < 1e-10);
    }
}
