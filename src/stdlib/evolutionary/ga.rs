/// Evolutionary algorithms: genetic algorithm, differential evolution, multi-objective.

use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct Individual {
    pub genes: Vec<f64>,
    pub fitness: f64,
    pub objectives: Vec<f64>,
}

impl Individual {
    pub fn new(genes: Vec<f64>) -> Self {
        Self { genes, fitness: 0.0, objectives: Vec::new() }
    }
}

/// Tournament selection.
pub fn tournament_select(population: &[Individual], tournament_size: usize, seed: &mut u64) -> usize {
    let mut best = (pseudo_rand(seed) * population.len() as f64) as usize % population.len();
    for _ in 1..tournament_size {
        let idx = (pseudo_rand(seed) * population.len() as f64) as usize % population.len();
        if population[idx].fitness < population[best].fitness {
            best = idx;
        }
    }
    best
}

/// Simulated Binary Crossover (SBX).
pub fn sbx_crossover(p1: &[f64], p2: &[f64], eta: f64, seed: &mut u64) -> (Vec<f64>, Vec<f64>) {
    let n = p1.len();
    let mut c1 = vec![0.0; n];
    let mut c2 = vec![0.0; n];

    for i in 0..n {
        if pseudo_rand(seed) < 0.5 {
            if (p1[i] - p2[i]).abs() > 1e-10 {
                let (y1, y2) = if p1[i] < p2[i] { (p1[i], p2[i]) } else { (p2[i], p1[i]) };
                let rand = pseudo_rand(seed);
                let beta = 1.0 + 2.0 * y1 / (y2 - y1 + 1e-10);
                let alpha = 2.0 - beta.powf(-(eta + 1.0));

                let betaq = if rand <= 1.0 / alpha {
                    (rand * alpha).powf(1.0 / (eta + 1.0))
                } else {
                    (1.0 / (2.0 - rand * alpha)).powf(1.0 / (eta + 1.0))
                };

                c1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
                c2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1));
            } else {
                c1[i] = p1[i];
                c2[i] = p2[i];
            }
        } else {
            c1[i] = p1[i];
            c2[i] = p2[i];
        }
    }

    (c1, c2)
}

/// Polynomial mutation.
pub fn polynomial_mutate(individual: &mut [f64], bounds: &[(f64, f64)], eta: f64, rate: f64, seed: &mut u64) {
    for i in 0..individual.len() {
        if pseudo_rand(seed) < rate {
            let (lo, hi) = bounds[i];
            let delta = hi - lo;
            let u = pseudo_rand(seed);

            let delta_q = if u < 0.5 {
                let val = (2.0 * u).powf(1.0 / (eta + 1.0));
                val - 1.0
            } else {
                let val = (2.0 * (1.0 - u)).powf(1.0 / (eta + 1.0));
                1.0 - val
            };

            individual[i] += delta_q * delta * 0.1;
            individual[i] = individual[i].clamp(lo, hi);
        }
    }
}

/// Differential Evolution.
pub struct DifferentialEvolution {
    population_size: usize,
    dimensions: usize,
    bounds: Vec<(f64, f64)>,
    f_scale: f64,
    crossover_rate: f64,
    seed: u64,
}

impl DifferentialEvolution {
    pub fn new(dimensions: usize, bounds: Vec<(f64, f64)>) -> Self {
        Self {
            population_size: 50,
            dimensions,
            bounds,
            f_scale: 0.8,
            crossover_rate: 0.9,
            seed: 42,
        }
    }

    pub fn with_population(mut self, size: usize) -> Self { self.population_size = size; self }
    pub fn with_f_scale(mut self, f: f64) -> Self { self.f_scale = f; self }
    pub fn with_crossover_rate(mut self, cr: f64) -> Self { self.crossover_rate = cr; self }

    fn init_population(&mut self) -> Vec<Vec<f64>> {
        (0..self.population_size).map(|_| {
            (0..self.dimensions).map(|d| {
                let (lo, hi) = self.bounds[d];
                lo + pseudo_rand(&mut self.seed) * (hi - lo)
            }).collect()
        }).collect()
    }

    /// Minimize a fitness function using DE/rand/1/bin.
    pub fn minimize<F: Fn(&[f64]) -> f64>(&mut self, fitness: F, generations: usize) -> (Vec<f64>, f64) {
        let mut population = self.init_population();
        let mut fitness_vals: Vec<f64> = population.iter().map(|ind| fitness(ind)).collect();

        let mut best_idx = fitness_vals.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        for _ in 0..generations {
            for i in 0..self.population_size {
                // Select three distinct random individuals
                let mut indices = Vec::new();
                while indices.len() < 3 {
                    let idx = (pseudo_rand(&mut self.seed) * self.population_size as f64) as usize % self.population_size;
                    if idx != i && !indices.contains(&idx) {
                        indices.push(idx);
                    }
                }

                let (a, b, c) = (indices[0], indices[1], indices[2]);

                // Mutation: v = x_a + F * (x_b - x_c)
                let mut trial = vec![0.0; self.dimensions];
                let j_rand = (pseudo_rand(&mut self.seed) * self.dimensions as f64) as usize % self.dimensions;

                for j in 0..self.dimensions {
                    if pseudo_rand(&mut self.seed) < self.crossover_rate || j == j_rand {
                        trial[j] = population[a][j] + self.f_scale * (population[b][j] - population[c][j]);
                        let (lo, hi) = self.bounds[j];
                        trial[j] = trial[j].clamp(lo, hi);
                    } else {
                        trial[j] = population[i][j];
                    }
                }

                let trial_fitness = fitness(&trial);
                if trial_fitness < fitness_vals[i] {
                    population[i] = trial;
                    fitness_vals[i] = trial_fitness;
                    if trial_fitness < fitness_vals[best_idx] {
                        best_idx = i;
                    }
                }
            }
        }

        (population[best_idx].clone(), fitness_vals[best_idx])
    }
}

/// NSGA-II multi-objective optimization.
pub struct NSGA2 {
    population_size: usize,
    dimensions: usize,
    num_objectives: usize,
    bounds: Vec<(f64, f64)>,
    seed: u64,
}

impl NSGA2 {
    pub fn new(dimensions: usize, num_objectives: usize, bounds: Vec<(f64, f64)>) -> Self {
        Self {
            population_size: 100,
            dimensions,
            num_objectives,
            bounds,
            seed: 42,
        }
    }

    pub fn with_population(mut self, size: usize) -> Self { self.population_size = size; self }

    fn init_population(&mut self) -> Vec<Individual> {
        (0..self.population_size).map(|_| {
            let genes: Vec<f64> = (0..self.dimensions).map(|d| {
                let (lo, hi) = self.bounds[d];
                lo + pseudo_rand(&mut self.seed) * (hi - lo)
            }).collect();
            Individual::new(genes)
        }).collect()
    }

    /// Minimize multiple objectives simultaneously.
    pub fn minimize<F>(&mut self, objectives: F, generations: usize) -> Vec<Individual>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let mut population = self.init_population();

        // Evaluate objectives
        for ind in &mut population {
            ind.objectives = objectives(&ind.genes);
            ind.fitness = ind.objectives.iter().sum(); // aggregate for selection
        }

        for gen in 0..generations {
            // Create offspring
            let mut offspring = Vec::new();
            while offspring.len() < self.population_size {
                let p1_idx = tournament_select(&population, 3, &mut self.seed);
                let p2_idx = tournament_select(&population, 3, &mut self.seed);

                let (c1_genes, c2_genes) = sbx_crossover(
                    &population[p1_idx].genes,
                    &population[p2_idx].genes,
                    20.0,
                    &mut self.seed,
                );

                let mut child1 = Individual::new(c1_genes);
                let mut child2 = Individual::new(c2_genes);

                polynomial_mutate(&mut child1.genes, &self.bounds, 20.0, 1.0 / self.dimensions as f64, &mut self.seed);
                polynomial_mutate(&mut child2.genes, &self.bounds, 20.0, 1.0 / self.dimensions as f64, &mut self.seed);

                child1.objectives = objectives(&child1.genes);
                child1.fitness = child1.objectives.iter().sum();
                child2.objectives = objectives(&child2.genes);
                child2.fitness = child2.objectives.iter().sum();

                offspring.push(child1);
                offspring.push(child2);
            }

            // Combine parent and offspring
            population.extend(offspring);

            // Non-dominated sorting
            let fronts = non_dominated_sort(&population);

            // Crowding distance
            let mut new_pop = Vec::new();
            for front in &fronts {
                if new_pop.len() + front.len() <= self.population_size {
                    for &idx in front {
                        new_pop.push(population[idx].clone());
                    }
                } else {
                    let remaining = self.population_size - new_pop.len();
                    let mut with_distance: Vec<(usize, f64)> = front.iter()
                        .map(|&idx| {
                            let d = crowding_distance(&population, front, self.num_objectives);
                            (idx, d[front.iter().position(|&i| i == idx).unwrap_or(0)])
                        })
                        .collect();
                    with_distance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                    for &(idx, _) in with_distance.iter().take(remaining) {
                        new_pop.push(population[idx].clone());
                    }
                    break;
                }
            }

            population = new_pop;
        }

        population
    }
}

/// Non-dominated sorting.
fn non_dominated_sort(population: &[Individual]) -> Vec<Vec<usize>> {
    let n = population.len();
    let mut domination_count = vec![0usize; n];
    let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&population[i].objectives, &population[j].objectives) {
                dominated_set[i].push(j);
                domination_count[j] += 1;
            } else if dominates(&population[j].objectives, &population[i].objectives) {
                dominated_set[j].push(i);
                domination_count[i] += 1;
            }
        }
        if domination_count[i] == 0 {
            current_front.push(i);
        }
    }

    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut at_least_one_better = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            at_least_one_better = true;
        }
    }
    at_least_one_better
}

/// Crowding distance for a front.
fn crowding_distance(population: &[Individual], front: &[usize], num_objectives: usize) -> Vec<f64> {
    let n = front.len();
    let mut distances = vec![f64::INFINITY; n];

    if n <= 2 {
        return distances;
    }

    for m in 0..num_objectives {
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            population[front[a]].objectives[m]
                .partial_cmp(&population[front[b]].objectives[m])
                .unwrap_or(Ordering::Equal)
        });

        let obj_min = population[front[sorted_indices[0]]].objectives[m];
        let obj_max = population[front[sorted_indices[n - 1]]].objectives[m];
        let range = obj_max - obj_min;

        if range < 1e-10 {
            continue;
        }

        distances[sorted_indices[0]] = 0.0;
        distances[sorted_indices[n - 1]] = 0.0;

        for i in 1..n - 1 {
            let prev = population[front[sorted_indices[i - 1]]].objectives[m];
            let next = population[front[sorted_indices[i + 1]]].objectives[m];
            distances[sorted_indices[i]] += (next - prev) / range;
        }
    }

    distances
}

fn pseudo_rand(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / (1u64 << 31) as f64
}

/// Fitness-proportionate (roulette wheel) selection.
pub fn roulette_select(fitness: &[f64], seed: &mut u64) -> usize {
    let total: f64 = fitness.iter().sum();
    if total == 0.0 {
        return (pseudo_rand(seed) * fitness.len() as f64) as usize % fitness.len();
    }
    let r = pseudo_rand(seed) * total;
    let mut cumulative = 0.0;
    for (i, &f) in fitness.iter().enumerate() {
        cumulative += f;
        if r <= cumulative {
            return i;
        }
    }
    fitness.len() - 1
}

/// Uniform crossover.
pub fn uniform_crossover(p1: &[f64], p2: &[f64], seed: &mut u64) -> (Vec<f64>, Vec<f64>) {
    let mut c1 = Vec::with_capacity(p1.len());
    let mut c2 = Vec::with_capacity(p2.len());

    for i in 0..p1.len() {
        if pseudo_rand(seed) < 0.5 {
            c1.push(p1[i]);
            c2.push(p2[i]);
        } else {
            c1.push(p2[i]);
            c2.push(p1[i]);
        }
    }

    (c1, c2)
}

/// Gaussian mutation.
pub fn gaussian_mutate(individual: &mut [f64], sigma: f64, bounds: &[(f64, f64)], seed: &mut u64) {
    for i in 0..individual.len() {
        let z = box_muller(seed);
        individual[i] += z * sigma;
        let (lo, hi) = bounds[i];
        individual[i] = individual[i].clamp(lo, hi);
    }
}

fn box_muller(seed: &mut u64) -> f64 {
    let u1 = pseudo_rand(seed);
    let u2 = pseudo_rand(seed);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Elitism: preserve top N individuals.
pub fn elitism(population: &mut Vec<Individual>, n: usize) {
    population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal));
    population.truncate(n.max(population.len()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differential_evolution() {
        let mut de = DifferentialEvolution::new(2, vec![(-5.0, 5.0), (-5.0, 5.0)]);
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let (best, fitness) = de.minimize(f, 100);
        assert!(fitness < 0.1);
    }

    #[test]
    fn test_nsga2() {
        let mut nsga = NSGA2::new(2, 2, vec![(0.0, 1.0), (0.0, 1.0)]);
        let objectives = |x: &[f64]| vec![x[0], x[1]]; // Minimize both
        let pop = nsga.minimize(objectives, 10);
        assert!(!pop.is_empty());
    }

    #[test]
    fn test_dominates() {
        assert!(dominates(&[1.0, 2.0], &[2.0, 3.0]));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 3.0]));
        assert!(!dominates(&[1.0, 3.0], &[1.0, 2.0]));
    }
}
