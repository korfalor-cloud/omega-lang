/// Numerical optimization: gradient descent, Newton's method, genetic algorithm, simulated annealing.

/// Minimize a function using gradient descent.
pub fn gradient_descent<F, G>(
    f: F,
    grad: G,
    x0: &[f64],
    learning_rate: f64,
    max_iter: usize,
    tolerance: f64,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let mut x = x0.to_vec();
    let mut lr = learning_rate;

    for _ in 0..max_iter {
        let g = grad(&x);

        // Check convergence
        let grad_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
        if grad_norm < tolerance {
            break;
        }

        // Update
        let new_x: Vec<f64> = x.iter().zip(g.iter()).map(|(xi, gi)| xi - lr * gi).collect();

        // Backtracking line search
        let old_f = f(&x);
        let new_f = f(&new_x);
        if new_f > old_f {
            lr *= 0.5;
            continue;
        }

        x = new_x;
        lr = (lr * 1.1).min(learning_rate * 10.0);
    }

    let value = f(&x);
    (x, value)
}

/// Minimize using Adam optimizer.
pub fn adam<F, G>(
    f: F,
    grad: G,
    x0: &[f64],
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    max_iter: usize,
    tolerance: f64,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut m = vec![0.0; n]; // First moment
    let mut v = vec![0.0; n]; // Second moment

    for t in 1..=max_iter {
        let g = grad(&x);

        // Check convergence
        let grad_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
        if grad_norm < tolerance {
            break;
        }

        // Update biased moments
        for i in 0..n {
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
        }

        // Bias correction
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);

        for i in 0..n {
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            x[i] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
        }
    }

    let value = f(&x);
    (x, value)
}

/// Newton's method for finding roots.
pub fn newton_root<F, D>(
    f: F,
    df: D,
    x0: f64,
    max_iter: usize,
    tolerance: f64,
) -> Option<f64>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    let mut x = x0;
    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() < tolerance {
            return Some(x);
        }
        let dfx = df(x);
        if dfx.abs() < 1e-15 {
            return None;
        }
        x -= fx / dfx;
    }
    Some(x)
}

/// Bisection method for root finding.
pub fn bisection<F>(
    f: F,
    mut a: f64,
    mut b: f64,
    max_iter: usize,
    tolerance: f64,
) -> Option<f64>
where
    F: Fn(f64) -> f64,
{
    let fa = f(a);
    let fb = f(b);
    if fa * fb > 0.0 {
        return None; // No sign change
    }

    for _ in 0..max_iter {
        let mid = (a + b) / 2.0;
        let fmid = f(mid);
        if fmid.abs() < tolerance || (b - a) / 2.0 < tolerance {
            return Some(mid);
        }
        if fa * fmid < 0.0 {
            b = mid;
        } else {
            a = mid;
        }
    }
    Some((a + b) / 2.0)
}

/// Secant method for root finding.
pub fn secant<F>(
    f: F,
    x0: f64,
    x1: f64,
    max_iter: usize,
    tolerance: f64,
) -> Option<f64>
where
    F: Fn(f64) -> f64,
{
    let mut x_prev = x0;
    let mut x_curr = x1;

    for _ in 0..max_iter {
        let f_prev = f(x_prev);
        let f_curr = f(x_curr);
        if f_curr.abs() < tolerance {
            return Some(x_curr);
        }
        let denom = f_curr - f_prev;
        if denom.abs() < 1e-15 {
            return None;
        }
        let x_next = x_curr - f_curr * (x_curr - x_prev) / denom;
        x_prev = x_curr;
        x_curr = x_next;
    }
    Some(x_curr)
}

/// Simplex method for linear programming (2D).
pub fn simplex_2d(
    c: &[f64],           // Objective coefficients
    a: &[Vec<f64>],      // Constraint matrix
    b: &[f64],           // Constraint RHS
) -> Option<(Vec<f64>, f64)> {
    let n = c.len();
    let m = b.len();

    if n != 2 || m == 0 {
        return None;
    }

    // Evaluate at all vertices
    let mut best_x = vec![0.0; n];
    let mut best_val = f64::INFINITY;
    let mut found = false;

    // Generate candidate points (intersections of constraints)
    let mut candidates = vec![vec![0.0; n]; 0];

    // Origin
    candidates.push(vec![0.0; n]);

    // Axis intercepts
    for i in 0..m {
        if a[i][0].abs() > 1e-10 {
            candidates.push(vec![b[i] / a[i][0], 0.0]);
        }
        if a[i][1].abs() > 1e-10 {
            candidates.push(vec![0.0, b[i] / a[i][1]]);
        }
    }

    // Pairwise intersections
    for i in 0..m {
        for j in (i + 1)..m {
            let det = a[i][0] * a[j][1] - a[i][1] * a[j][0];
            if det.abs() > 1e-10 {
                let x = (b[i] * a[j][1] - b[j] * a[i][1]) / det;
                let y = (a[i][0] * b[j] - a[j][0] * b[i]) / det;
                candidates.push(vec![x, y]);
            }
        }
    }

    // Check feasibility and find minimum
    for point in candidates {
        if point.iter().any(|&v| v < -1e-10) {
            continue;
        }
        let feasible = a.iter().zip(b.iter()).all(|(ai, bi)| {
            ai.iter().zip(point.iter()).map(|(aij, xj)| aij * xj).sum::<f64>() <= *bi + 1e-10
        });
        if feasible {
            let val: f64 = c.iter().zip(point.iter()).map(|(ci, xi)| ci * xi).sum();
            if val < best_val {
                best_val = val;
                best_x = point;
                found = true;
            }
        }
    }

    if found {
        Some((best_x, best_val))
    } else {
        None
    }
}

/// Genetic algorithm for optimization.
pub struct GeneticAlgorithm {
    population_size: usize,
    dimensions: usize,
    bounds: Vec<(f64, f64)>,
    mutation_rate: f64,
    crossover_rate: f64,
    elitism: usize,
    seed: u64,
}

impl GeneticAlgorithm {
    pub fn new(dimensions: usize, bounds: Vec<(f64, f64)>) -> Self {
        Self {
            population_size: 100,
            dimensions,
            bounds,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elitism: 2,
            seed: 42,
        }
    }

    pub fn with_population(mut self, size: usize) -> Self { self.population_size = size; self }
    pub fn with_mutation_rate(mut self, rate: f64) -> Self { self.mutation_rate = rate; self }
    pub fn with_crossover_rate(mut self, rate: f64) -> Self { self.crossover_rate = rate; self }
    pub fn with_elitism(mut self, n: usize) -> Self { self.elitism = n; self }

    fn random(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }

    fn init_population(&mut self) -> Vec<Vec<f64>> {
        (0..self.population_size).map(|_| {
            (0..self.dimensions).map(|d| {
                let (lo, hi) = self.bounds[d];
                lo + self.random() * (hi - lo)
            }).collect()
        }).collect()
    }

    fn tournament_select(&mut self, fitness: &[f64]) -> usize {
        let a = (self.random() * fitness.len() as f64) as usize % fitness.len();
        let b = (self.random() * fitness.len() as f64) as usize % fitness.len();
        if fitness[a] < fitness[b] { a } else { b }
    }

    fn crossover(&mut self, p1: &[f64], p2: &[f64]) -> Vec<f64> {
        if self.random() > self.crossover_rate {
            return p1.to_vec();
        }
        let point = (self.random() * self.dimensions as f64) as usize % self.dimensions;
        let mut child = p1[..point].to_vec();
        child.extend_from_slice(&p2[point..]);
        child
    }

    fn mutate(&mut self, individual: &mut [f64]) {
        for d in 0..self.dimensions {
            if self.random() < self.mutation_rate {
                let (lo, hi) = self.bounds[d];
                let range = hi - lo;
                individual[d] += (self.random() - 0.5) * range * 0.1;
                individual[d] = individual[d].clamp(lo, hi);
            }
        }
    }

    /// Minimize a fitness function.
    pub fn minimize<F: Fn(&[f64]) -> f64>(&mut self, fitness: F, generations: usize) -> (Vec<f64>, f64) {
        let mut population = self.init_population();

        for _ in 0..generations {
            let fitness_vals: Vec<f64> = population.iter().map(|ind| fitness(ind)).collect();

            // Sort by fitness
            let mut indices: Vec<usize> = (0..population.len()).collect();
            indices.sort_by(|&a, &b| fitness_vals[a].partial_cmp(&fitness_vals[b]).unwrap_or(std::cmp::Ordering::Equal));

            let mut new_population = Vec::new();

            // Elitism
            for i in 0..self.elitism.min(indices.len()) {
                new_population.push(population[indices[i]].clone());
            }

            // Generate rest
            while new_population.len() < self.population_size {
                let p1 = self.tournament_select(&fitness_vals);
                let p2 = self.tournament_select(&fitness_vals);
                let mut child = self.crossover(&population[p1], &population[p2]);
                self.mutate(&mut child);
                new_population.push(child);
            }

            population = new_population;
        }

        // Return best
        let fitness_vals: Vec<f64> = population.iter().map(|ind| fitness(ind)).collect();
        let best_idx = fitness_vals.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        (population[best_idx].clone(), fitness_vals[best_idx])
    }
}

/// Simulated annealing optimizer.
pub fn simulated_annealing<F>(
    f: F,
    x0: &[f64],
    bounds: &[(f64, f64)],
    initial_temp: f64,
    cooling_rate: f64,
    max_iter: usize,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let mut seed: u64 = 42;
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut best_x = x.clone();
    let mut best_fx = fx;
    let mut temp = initial_temp;

    for _ in 0..max_iter {
        // Generate neighbor
        let mut neighbor = x.clone();
        let d = (pseudo_rand(&mut seed) * x.len() as f64) as usize % x.len();
        let (lo, hi) = bounds[d];
        let range = hi - lo;
        neighbor[d] += (pseudo_rand(&mut seed) - 0.5) * range * 0.1 * temp / initial_temp;
        neighbor[d] = neighbor[d].clamp(lo, hi);

        let f_neighbor = f(&neighbor);
        let delta = f_neighbor - fx;

        if delta < 0.0 || pseudo_rand(&mut seed) < (-delta / temp).exp() {
            x = neighbor;
            fx = f_neighbor;
            if fx < best_fx {
                best_x = x.clone();
                best_fx = fx;
            }
        }

        temp *= cooling_rate;
    }

    (best_x, best_fx)
}

fn pseudo_rand(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / (1u64 << 31) as f64
}

/// Particle Swarm Optimization.
pub fn particle_swarm<F>(
    f: F,
    bounds: &[(f64, f64)],
    num_particles: usize,
    max_iter: usize,
    w: f64,
    c1: f64,
    c2: f64,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let dim = bounds.len();
    let mut seed: u64 = 77;

    // Initialize particles
    let mut positions: Vec<Vec<f64>> = (0..num_particles).map(|_| {
        (0..dim).map(|d| {
            let (lo, hi) = bounds[d];
            lo + pseudo_rand(&mut seed) * (hi - lo)
        }).collect()
    }).collect();

    let mut velocities: Vec<Vec<f64>> = vec![vec![0.0; dim]; num_particles];
    let mut personal_best = positions.clone();
    let mut personal_best_fitness: Vec<f64> = positions.iter().map(|p| f(p)).collect();

    let mut gbest_idx = personal_best_fitness.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut gbest = personal_best[gbest_idx].clone();
    let mut gbest_fitness = personal_best_fitness[gbest_idx];

    for _ in 0..max_iter {
        for i in 0..num_particles {
            for d in 0..dim {
                let r1 = pseudo_rand(&mut seed);
                let r2 = pseudo_rand(&mut seed);
                velocities[i][d] = w * velocities[i][d]
                    + c1 * r1 * (personal_best[i][d] - positions[i][d])
                    + c2 * r2 * (gbest[d] - positions[i][d]);
                positions[i][d] += velocities[i][d];
                positions[i][d] = positions[i][d].clamp(bounds[d].0, bounds[d].1);
            }

            let fitness = f(&positions[i]);
            if fitness < personal_best_fitness[i] {
                personal_best[i] = positions[i].clone();
                personal_best_fitness[i] = fitness;
                if fitness < gbest_fitness {
                    gbest = positions[i].clone();
                    gbest_fitness = fitness;
                }
            }
        }
    }

    (gbest, gbest_fitness)
}

/// Golden section search for 1D minimization.
pub fn golden_section_search<F>(
    f: F,
    mut a: f64,
    mut b: f64,
    tolerance: f64,
) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let resphi = 2.0 - phi;

    let mut x1 = a + resphi * (b - a);
    let mut x2 = b - resphi * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    while (b - a).abs() > tolerance {
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + resphi * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - resphi * (b - a);
            f2 = f(x2);
        }
    }

    let x_opt = (a + b) / 2.0;
    (x_opt, f(x_opt))
}

/// Nelder-Mead simplex method.
pub fn nelder_mead<F>(
    f: F,
    x0: &[f64],
    initial_step: f64,
    max_iter: usize,
    tolerance: f64,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;

    // Initialize simplex
    let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..n {
        let mut point = x0.to_vec();
        point[i] += initial_step;
        simplex.push(point);
    }

    for _ in 0..max_iter {
        // Sort by fitness
        simplex.sort_by(|a, b| f(a).partial_cmp(&f(b)).unwrap_or(std::cmp::Ordering::Equal));

        // Check convergence
        let best = f(&simplex[0]);
        let worst = f(&simplex[n]);
        if (worst - best).abs() < tolerance {
            break;
        }

        // Centroid of all but worst
        let centroid: Vec<f64> = (0..n).map(|d| {
            simplex[..n].iter().map(|s| s[d]).sum::<f64>() / n as f64
        }).collect();

        // Reflection
        let reflected: Vec<f64> = centroid.iter().zip(simplex[n].iter())
            .map(|(c, w)| c + alpha * (c - w))
            .collect();
        let f_reflected = f(&reflected);

        if f_reflected < f(&simplex[n - 1]) && f_reflected >= f(&simplex[0]) {
            simplex[n] = reflected;
        } else if f_reflected < f(&simplex[0]) {
            // Expansion
            let expanded: Vec<f64> = centroid.iter().zip(reflected.iter())
                .map(|(c, r)| c + gamma * (r - c))
                .collect();
            if f(&expanded) < f_reflected {
                simplex[n] = expanded;
            } else {
                simplex[n] = reflected;
            }
        } else {
            // Contraction
            let contracted: Vec<f64> = centroid.iter().zip(simplex[n].iter())
                .map(|(c, w)| c + rho * (w - c))
                .collect();
            if f(&contracted) < f(&simplex[n]) {
                simplex[n] = contracted;
            } else {
                // Shrink
                for i in 1..=n {
                    for d in 0..n {
                        simplex[i][d] = simplex[0][d] + sigma * (simplex[i][d] - simplex[0][d]);
                    }
                }
            }
        }
    }

    simplex.sort_by(|a, b| f(a).partial_cmp(&f(b)).unwrap_or(std::cmp::Ordering::Equal));
    (simplex[0].clone(), f(&simplex[0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_gradient_descent() {
        // Minimize (x-3)^2 + (y-5)^2
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 5.0).powi(2);
        let grad = |x: &[f64]| vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] - 5.0)];

        let (x, _) = gradient_descent(f, grad, &[0.0, 0.0], 0.1, 1000, 1e-8);
        assert!(approx_eq(x[0], 3.0, 0.01));
        assert!(approx_eq(x[1], 5.0, 0.01));
    }

    #[test]
    fn test_newton_root() {
        // Find root of x^2 - 4
        let f = |x: f64| x * x - 4.0;
        let df = |x: f64| 2.0 * x;
        let root = newton_root(f, df, 3.0, 100, 1e-10).unwrap();
        assert!(approx_eq(root, 2.0, 1e-8));
    }

    #[test]
    fn test_bisection() {
        let f = |x: f64| x * x - 2.0;
        let root = bisection(f, 0.0, 2.0, 100, 1e-10).unwrap();
        assert!(approx_eq(root, 2.0_f64.sqrt(), 1e-8));
    }

    #[test]
    fn test_golden_section() {
        let f = |x: f64| (x - 2.5).powi(2);
        let (x, _) = golden_section_search(f, 0.0, 5.0, 1e-8);
        assert!(approx_eq(x, 2.5, 0.01));
    }

    #[test]
    fn test_simplex_2d() {
        let c = vec![1.0, 1.0];
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let b = vec![4.0, 4.0, 6.0];
        let result = simplex_2d(&c, &a, &b);
        assert!(result.is_some());
    }
}
