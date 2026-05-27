/// Principal Component Analysis.

#[derive(Debug, Clone)]
pub struct PCA {
    n_components: usize,
    components: Vec<Vec<f64>>,
    mean: Vec<f64>,
    eigenvalues: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            components: Vec::new(),
            mean: Vec::new(),
            eigenvalues: Vec::new(),
            explained_variance_ratio: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>]) {
        assert!(!x.is_empty());
        let n_samples = x.len();
        let n_features = x[0].len();
        let n_components = self.n_components.min(n_features);

        // Compute mean
        self.mean = vec![0.0; n_features];
        for row in x {
            for j in 0..n_features {
                self.mean[j] += row[j];
            }
        }
        for val in self.mean.iter_mut() {
            *val /= n_samples as f64;
        }

        // Center data
        let centered: Vec<Vec<f64>> = x.iter()
            .map(|row| row.iter().zip(self.mean.iter()).map(|(v, m)| v - m).collect())
            .collect();

        // Compute covariance matrix
        let mut cov = vec![vec![0.0; n_features]; n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for row in &centered {
                    sum += row[i] * row[j];
                }
                cov[i][j] = sum / (n_samples - 1) as f64;
            }
        }

        // Jacobi eigendecomposition
        let (eigenvalues, eigenvectors) = jacobi_eigen(&cov, 1000, 1e-10);

        // Sort by eigenvalue (descending)
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

        self.eigenvalues = indices.iter().map(|&i| eigenvalues[i]).collect();
        self.components = (0..n_components)
            .map(|k| (0..n_features).map(|j| eigenvectors[j][indices[k]]).collect())
            .collect();

        // Explained variance ratio
        let total_variance: f64 = self.eigenvalues.iter().sum();
        self.explained_variance_ratio = self.eigenvalues.iter()
            .take(n_components)
            .map(|ev| ev / total_variance)
            .collect();
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter().map(|row| {
            let centered: Vec<f64> = row.iter().zip(self.mean.iter()).map(|(v, m)| v - m).collect();
            self.components.iter()
                .map(|component| centered.iter().zip(component.iter()).map(|(c, w)| c * w).sum())
                .collect()
        }).collect()
    }

    pub fn inverse_transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter().map(|row| {
            let mut result = self.mean.clone();
            for (k, &val) in row.iter().enumerate() {
                for j in 0..result.len() {
                    result[j] += val * self.components[k][j];
                }
            }
            result
        }).collect()
    }

    pub fn fit_transform(&mut self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.fit(x);
        self.transform(x)
    }

    pub fn reconstruction_error(&self, x: &[Vec<f64>]) -> f64 {
        let transformed = self.transform(x);
        let reconstructed = self.inverse_transform(&transformed);

        x.iter().zip(reconstructed.iter())
            .map(|(orig, recon)| {
                orig.iter().zip(recon.iter())
                    .map(|(o, r)| (o - r).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>() / x.len() as f64
    }

    pub fn cumulative_variance(&self) -> Vec<f64> {
        let mut cumulative = Vec::new();
        let mut sum = 0.0;
        for ratio in &self.explained_variance_ratio {
            sum += ratio;
            cumulative.push(sum);
        }
        cumulative
    }

    pub fn components(&self) -> &[Vec<f64>] {
        &self.components
    }

    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    pub fn explained_variance_ratio(&self) -> &[f64] {
        &self.explained_variance_ratio
    }
}

fn jacobi_eigen(matrix: &[Vec<f64>], max_iter: usize, tol: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let (mut p, mut q) = (0, 1);
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Compute rotation
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-15 {
            std::f64::consts::PI / 4.0
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        a[p][p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for i in 0..n {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                a[i][p] = c * aip + s * aiq;
                a[p][i] = a[i][p];
                a[i][q] = -s * aip + c * aiq;
                a[q][i] = a[i][q];
            }
        }

        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip + s * viq;
            v[i][q] = -s * vip + c * viq;
        }
    }

    let eigenvalues = (0..n).map(|i| a[i][i]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_2d() {
        let x = vec![
            vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0],
            vec![4.0, 5.0], vec![5.0, 6.0],
        ];

        let mut pca = PCA::new(1);
        let transformed = pca.fit_transform(&x);

        assert_eq!(transformed[0].len(), 1);
        assert!(pca.explained_variance_ratio()[0] > 0.9);
    }

    #[test]
    fn test_reconstruction() {
        let x = vec![
            vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0], vec![4.0, 5.0, 6.0],
        ];

        let mut pca = PCA::new(2);
        let transformed = pca.fit_transform(&x);
        let reconstructed = pca.inverse_transform(&transformed);

        assert_eq!(reconstructed.len(), 4);
        assert_eq!(reconstructed[0].len(), 3);
    }

    #[test]
    fn test_cumulative_variance() {
        let x = vec![
            vec![1.0, 0.0, 0.0], vec![2.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0], vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 2.0],
        ];

        let mut pca = PCA::new(3);
        pca.fit(&x);

        let cumulative = pca.cumulative_variance();
        assert!(cumulative[2] > 0.99); // Should explain almost all variance
    }

    #[test]
    fn test_jacobi_eigen() {
        let matrix = vec![
            vec![4.0, 2.0],
            vec![2.0, 3.0],
        ];

        let (eigenvalues, _) = jacobi_eigen(&matrix, 100, 1e-10);
        // Eigenvalues of [[4,2],[2,3]] are approximately 5.56 and 1.44
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sorted[0] - 5.56).abs() < 0.1);
        assert!((sorted[1] - 1.44).abs() < 0.1);
    }
}
