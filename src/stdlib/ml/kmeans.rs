/// K-Means clustering with K-Means++ initialization.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct KMeans {
    k: usize,
    max_iterations: usize,
    tolerance: f64,
    centroids: Vec<Vec<f64>>,
    labels: Vec<usize>,
    inertia: f64,
    iterations_run: usize,
}

impl KMeans {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iterations: 300,
            tolerance: 1e-4,
            centroids: Vec::new(),
            labels: Vec::new(),
            inertia: 0.0,
            iterations_run: 0,
        }
    }

    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn fit(&mut self, x: &[Vec<f64>]) {
        assert!(!x.is_empty());
        assert!(self.k <= x.len());

        self.centroids = self.kmeans_plus_plus_init(x);
        self.labels = vec![0; x.len()];

        for iter in 0..self.max_iterations {
            // Assign points to nearest centroid
            for (i, point) in x.iter().enumerate() {
                self.labels[i] = self.nearest_centroid(point);
            }

            // Compute new centroids
            let mut new_centroids = vec![vec![0.0; x[0].len()]; self.k];
            let mut counts = vec![0usize; self.k];

            for (i, point) in x.iter().enumerate() {
                let cluster = self.labels[i];
                for (j, val) in point.iter().enumerate() {
                    new_centroids[cluster][j] += val;
                }
                counts[cluster] += 1;
            }

            for (c, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[c] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[c] as f64;
                    }
                } else {
                    // Reinitialize empty cluster
                    *centroid = x[iter % x.len()].clone();
                }
            }

            // Check convergence
            let mut max_shift = 0.0;
            for (old, new) in self.centroids.iter().zip(new_centroids.iter()) {
                let shift = euclidean_distance(old, new);
                if shift > max_shift {
                    max_shift = shift;
                }
            }

            self.centroids = new_centroids;
            self.iterations_run = iter + 1;

            if max_shift < self.tolerance {
                break;
            }
        }

        // Compute inertia
        self.inertia = x.iter().enumerate()
            .map(|(i, point)| euclidean_distance(point, &self.centroids[self.labels[i]]).powi(2))
            .sum();
    }

    fn kmeans_plus_plus_init(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut centroids = Vec::new();

        // Choose first centroid randomly (use first point as deterministic)
        centroids.push(x[0].clone());

        for _ in 1..self.k {
            let mut distances: Vec<f64> = x.iter()
                .map(|point| {
                    centroids.iter()
                        .map(|c| euclidean_distance(point, c).powi(2))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();

            let total: f64 = distances.iter().sum();
            if total == 0.0 {
                centroids.push(x[centroids.len() % x.len()].clone());
                continue;
            }

            // Choose next centroid proportional to distance squared
            let mut cumulative = 0.0;
            let target = total * 0.5; // Deterministic midpoint selection
            let mut chosen = 0;
            for (i, &dist) in distances.iter().enumerate() {
                cumulative += dist;
                if cumulative >= target {
                    chosen = i;
                    break;
                }
            }
            centroids.push(x[chosen].clone());
        }

        centroids
    }

    fn nearest_centroid(&self, point: &[f64]) -> usize {
        self.centroids.iter().enumerate()
            .min_by(|(_, a), (_, b)| {
                euclidean_distance(point, a).partial_cmp(&euclidean_distance(point, b)).unwrap()
            })
            .unwrap().0
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        x.iter().map(|point| self.nearest_centroid(point)).collect()
    }

    pub fn centroids(&self) -> &[Vec<f64>] {
        &self.centroids
    }

    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    pub fn inertia(&self) -> f64 {
        self.inertia
    }

    pub fn iterations_run(&self) -> usize {
        self.iterations_run
    }

    pub fn silhouette_score(&self, x: &[Vec<f64>]) -> f64 {
        if x.len() <= 1 || self.k <= 1 {
            return 0.0;
        }

        let mut total_score = 0.0;

        for i in 0..x.len() {
            let cluster = self.labels[i];

            // a(i): mean distance to same cluster
            let same_cluster: Vec<usize> = (0..x.len())
                .filter(|&j| j != i && self.labels[j] == cluster)
                .collect();

            let a = if same_cluster.is_empty() {
                0.0
            } else {
                same_cluster.iter()
                    .map(|&j| euclidean_distance(&x[i], &x[j]))
                    .sum::<f64>() / same_cluster.len() as f64
            };

            // b(i): min mean distance to other clusters
            let mut min_b = f64::INFINITY;
            for c in 0..self.k {
                if c == cluster {
                    continue;
                }
                let other_cluster: Vec<usize> = (0..x.len())
                    .filter(|&j| self.labels[j] == c)
                    .collect();

                if !other_cluster.is_empty() {
                    let mean_dist = other_cluster.iter()
                        .map(|&j| euclidean_distance(&x[i], &x[j]))
                        .sum::<f64>() / other_cluster.len() as f64;
                    if mean_dist < min_b {
                        min_b = mean_dist;
                    }
                }
            }

            let s = if a == 0.0 && min_b == f64::INFINITY {
                0.0
            } else {
                (min_b - a) / a.max(min_b)
            };

            total_score += s;
        }

        total_score / x.len() as f64
    }

    pub fn elbow_method(x: &[Vec<f64>], max_k: usize) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
        for k in 1..=max_k.min(x.len()) {
            let mut model = KMeans::new(k);
            model.fit(x);
            results.push((k, model.inertia()));
        }
        results
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_two_clusters() {
        let x = vec![
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0],
            vec![5.0, 5.0], vec![5.1, 5.1], vec![5.0, 5.2],
        ];

        let mut kmeans = KMeans::new(2);
        kmeans.fit(&x);

        assert_eq!(kmeans.labels().len(), 6);
        assert_ne!(kmeans.labels()[0], kmeans.labels()[5]);
    }

    #[test]
    fn test_silhouette_score() {
        let x = vec![
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.0, 0.1],
            vec![5.0, 5.0], vec![5.1, 5.1], vec![5.0, 5.1],
        ];

        let mut kmeans = KMeans::new(2);
        kmeans.fit(&x);

        let score = kmeans.silhouette_score(&x);
        assert!(score > 0.5); // Should be high for well-separated clusters
    }

    #[test]
    fn test_elbow_method() {
        let x = vec![
            vec![0.0, 0.0], vec![0.1, 0.1],
            vec![5.0, 5.0], vec![5.1, 5.1],
            vec![10.0, 10.0], vec![10.1, 10.1],
        ];

        let results = KMeans::elbow_method(&x, 4);
        assert_eq!(results.len(), 4);
        // Inertia should decrease with more clusters
        assert!(results[0].1 >= results[1].1);
    }

    #[test]
    fn test_predict() {
        let x = vec![
            vec![0.0, 0.0], vec![0.1, 0.1],
            vec![5.0, 5.0], vec![5.1, 5.1],
        ];

        let mut kmeans = KMeans::new(2);
        kmeans.fit(&x);

        let predictions = kmeans.predict(&[vec![0.05, 0.05], vec![5.05, 5.05]]);
        assert_ne!(predictions[0], predictions[1]);
    }
}
