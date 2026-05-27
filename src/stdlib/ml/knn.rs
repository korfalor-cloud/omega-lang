/// K-Nearest Neighbors classifier.

#[derive(Debug, Clone)]
pub struct KNN {
    k: usize,
    distance: DistanceMetric,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    weighted: bool,
}

#[derive(Debug, Clone)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Minkowski(f64),
    Chebyshev,
    Cosine,
}

impl KNN {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            distance: DistanceMetric::Euclidean,
            x_train: Vec::new(),
            y_train: Vec::new(),
            weighted: false,
        }
    }

    pub fn distance(mut self, metric: DistanceMetric) -> Self {
        self.distance = metric;
        self
    }

    pub fn weighted(mut self, weighted: bool) -> Self {
        self.weighted = weighted;
        self
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        self.x_train = x.to_vec();
        self.y_train = y.to_vec();
    }

    fn distance_between(&self, a: &[f64], b: &[f64]) -> f64 {
        match &self.distance {
            DistanceMetric::Euclidean => {
                a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>().sqrt()
            }
            DistanceMetric::Manhattan => {
                a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).abs()).sum()
            }
            DistanceMetric::Minkowski(p) => {
                a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).abs().powf(*p)).sum::<f64>().powf(1.0 / p)
            }
            DistanceMetric::Chebyshev => {
                a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).abs()).fold(0.0_f64, f64::max)
            }
            DistanceMetric::Cosine => {
                let dot: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
                let norm_a: f64 = a.iter().map(|ai| ai.powi(2)).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|bi| bi.powi(2)).sum::<f64>().sqrt();
                if norm_a == 0.0 || norm_b == 0.0 { 1.0 } else { 1.0 - dot / (norm_a * norm_b) }
            }
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| self.predict_single(row)).collect()
    }

    fn predict_single(&self, x: &[f64]) -> f64 {
        let mut distances: Vec<(f64, f64)> = self.x_train.iter().zip(self.y_train.iter())
            .map(|(train_x, train_y)| (self.distance_between(x, train_x), *train_y))
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let k_nearest = &distances[..self.k.min(distances.len())];

        if self.weighted {
            let mut weighted_votes: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
            for (dist, label) in k_nearest {
                let weight = if *dist < 1e-10 { 1e10 } else { 1.0 / dist };
                *weighted_votes.entry(*label as i64).or_insert(0.0) += weight;
            }
            weighted_votes.into_iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0 as f64
        } else {
            let mut votes: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
            for (_, label) in k_nearest {
                *votes.entry(*label as i64).or_insert(0) += 1;
            }
            votes.into_iter().max_by_key(|&(_, count)| count).unwrap().0 as f64
        }
    }

    pub fn accuracy(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let correct = predictions.iter().zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-10)
            .count();
        correct as f64 / y.len() as f64
    }

    pub fn leave_one_out_cv(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let mut correct = 0;
        for i in 0..x.len() {
            let mut distances: Vec<(f64, f64)> = Vec::new();
            for j in 0..x.len() {
                if i != j {
                    distances.push((self.distance_between(&x[i], &x[j]), y[j]));
                }
            }
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let k_nearest = &distances[..self.k.min(distances.len())];

            let mut votes: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
            for (_, label) in k_nearest {
                *votes.entry(*label as i64).or_insert(0) += 1;
            }
            let predicted = votes.into_iter().max_by_key(|&(_, count)| count).unwrap().0 as f64;
            if (predicted - y[i]).abs() < 1e-10 {
                correct += 1;
            }
        }
        correct as f64 / x.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_simple() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        let y = vec![0.0, 0.0, 1.0, 1.0];

        let mut knn = KNN::new(3);
        knn.fit(&x, &y);

        let pred = knn.predict(&[vec![0.1, 0.1]]);
        assert_eq!(pred[0], 0.0);
    }

    #[test]
    fn test_knn_weighted() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        let y = vec![0.0, 0.0, 1.0, 1.0];

        let mut knn = KNN::new(3).weighted(true);
        knn.fit(&x, &y);

        let pred = knn.predict(&[vec![0.9, 0.9]]);
        assert_eq!(pred[0], 0.0);
    }

    #[test]
    fn test_knn_manhattan() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![5.0, 5.0], vec![6.0, 6.0]];
        let y = vec![0.0, 0.0, 1.0, 1.0];

        let mut knn = KNN::new(2).distance(DistanceMetric::Manhattan);
        knn.fit(&x, &y);

        let pred = knn.predict(&[vec![0.5, 0.5]]);
        assert_eq!(pred[0], 0.0);
    }

    #[test]
    fn test_loocv() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        let y = vec![0.0, 0.0, 1.0, 1.0];

        let knn = KNN::new(1);
        let cv = knn.leave_one_out_cv(&x, &y);
        assert!(cv >= 0.0 && cv <= 1.0);
    }
}
