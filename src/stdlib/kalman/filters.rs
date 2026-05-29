/// Extended Kalman Filter and Unscented Kalman Filter.

/// Extended Kalman Filter.
pub struct ExtendedKalmanFilter {
    pub state_dim: usize,
    pub obs_dim: usize,
    pub state: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub process_noise: Vec<Vec<f64>>,
    pub measurement_noise: Vec<Vec<f64>>,
}

impl ExtendedKalmanFilter {
    pub fn new(state_dim: usize, obs_dim: usize) -> Self {
        Self {
            state_dim,
            obs_dim,
            state: vec![0.0; state_dim],
            covariance: Self::identity(state_dim),
            process_noise: Self::identity(state_dim),
            measurement_noise: Self::identity(obs_dim),
        }
    }

    fn identity(n: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n { m[i][i] = 1.0; }
        m
    }

    /// Predict step with nonlinear state transition.
    pub fn predict<F>(&mut self, f: F, jacobian_f: &[Vec<f64>], dt: f64)
    where
        F: Fn(&[f64], f64) -> Vec<f64>,
    {
        // Predict state
        self.state = f(&self.state, dt);

        // Predict covariance: P = F * P * F^T + Q
        let fp = Self::mat_mul(jacobian_f, &self.covariance);
        let ft = Self::transpose(jacobian_f);
        let fpft = Self::mat_mul(&fp, &ft);
        for i in 0..self.state_dim {
            for j in 0..self.state_dim {
                self.covariance[i][j] = fpft[i][j] + self.process_noise[i][j];
            }
        }
    }

    /// Update step with nonlinear observation.
    pub fn update<H>(&mut self, measurement: &[f64], h: H, jacobian_h: &[Vec<f64>])
    where
        H: Fn(&[f64]) -> Vec<f64>,
    {
        let predicted_obs = h(&self.state);

        // Innovation
        let innovation: Vec<f64> = measurement.iter().zip(predicted_obs.iter())
            .map(|(z, h)| z - h)
            .collect();

        // S = H * P * H^T + R
        let hp = Self::mat_mul(jacobian_h, &self.covariance);
        let ht = Self::transpose(jacobian_h);
        let hph = Self::mat_mul(&hp, &ht);
        let mut s = hph;
        for i in 0..self.obs_dim {
            for j in 0..self.obs_dim {
                s[i][j] += self.measurement_noise[i][j];
            }
        }

        // K = P * H^T * S^-1
        let ph = Self::mat_mul(&self.covariance, &ht);
        if let Some(s_inv) = Self::mat_invert(&s) {
            let k = Self::mat_mul(&ph, &s_inv);

            // Update state
            let ky = Self::mat_vec_mul(&k, &innovation);
            for i in 0..self.state_dim {
                self.state[i] += ky[i];
            }

            // Update covariance
            let kh = Self::mat_mul(&k, jacobian_h);
            let mut i_minus_kh = Self::identity(self.state_dim);
            for i in 0..self.state_dim {
                for j in 0..self.state_dim {
                    i_minus_kh[i][j] -= kh[i][j];
                }
            }
            self.covariance = Self::mat_mul(&i_minus_kh, &self.covariance);
        }
    }

    fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = a.len();
        let cols = b[0].len();
        let inner = a[0].len();
        let mut result = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..inner {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        result
    }

    fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        m.iter().map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
    }

    fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = m.len();
        let cols = m[0].len();
        let mut t = vec![vec![0.0; rows]; cols];
        for i in 0..rows {
            for j in 0..cols {
                t[j][i] = m[i][j];
            }
        }
        t
    }

    fn mat_invert(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        let n = matrix.len();
        let mut aug = vec![vec![0.0; 2 * n]; n];
        for i in 0..n {
            for j in 0..n { aug[i][j] = matrix[i][j]; }
            aug[i][n + i] = 1.0;
        }

        for col in 0..n {
            let mut max_row = col;
            for row in (col + 1)..n {
                if aug[row][col].abs() > aug[max_row][col].abs() {
                    max_row = row;
                }
            }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            if pivot.abs() < 1e-10 { return None; }

            for j in 0..(2 * n) { aug[col][j] /= pivot; }

            for row in 0..n {
                if row == col { continue; }
                let factor = aug[row][col];
                for j in 0..(2 * n) { aug[row][j] -= factor * aug[col][j]; }
            }
        }

        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n { inv[i][j] = aug[i][n + j]; }
        }
        Some(inv)
    }
}

/// Unscented Kalman Filter.
pub struct UnscentedKalmanFilter {
    pub state_dim: usize,
    pub obs_dim: usize,
    pub state: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub process_noise: Vec<Vec<f64>>,
    pub measurement_noise: Vec<Vec<f64>>,
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
}

impl UnscentedKalmanFilter {
    pub fn new(state_dim: usize, obs_dim: usize) -> Self {
        Self {
            state_dim,
            obs_dim,
            state: vec![0.0; state_dim],
            covariance: Self::identity(state_dim),
            process_noise: Self::identity(state_dim),
            measurement_noise: Self::identity(obs_dim),
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        }
    }

    fn identity(n: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n { m[i][i] = 1.0; }
        m
    }

    /// Generate sigma points.
    fn sigma_points(&self) -> Vec<Vec<f64>> {
        let n = self.state_dim;
        let lambda = self.alpha * self.alpha * (n as f64 + self.kappa) - n as f64;

        // Compute matrix square root using Cholesky
        let scaled_cov: Vec<Vec<f64>> = self.covariance.iter()
            .map(|row| row.iter().map(|&x| x * (n as f64 + lambda)).collect())
            .collect();

        let l = Self::cholesky(&scaled_cov);

        let mut points = Vec::new();
        points.push(self.state.clone());

        for i in 0..n {
            let mut p_plus = self.state.clone();
            let mut p_minus = self.state.clone();
            for j in 0..n {
                p_plus[j] += l[j][i];
                p_minus[j] -= l[j][i];
            }
            points.push(p_plus);
            points.push(p_minus);
        }

        points
    }

    fn weights(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.state_dim;
        let lambda = self.alpha * self.alpha * (n as f64 + self.kappa) - n as f64;
        let total = 2 * n + 1;

        let mut wm = vec![0.0; total];
        let mut wc = vec![0.0; total];

        wm[0] = lambda / (n as f64 + lambda);
        wc[0] = lambda / (n as f64 + lambda) + (1.0 - self.alpha * self.alpha + self.beta);

        for i in 1..total {
            wm[i] = 1.0 / (2.0 * (n as f64 + lambda));
            wc[i] = 1.0 / (2.0 * (n as f64 + lambda));
        }

        (wm, wc)
    }

    /// Predict step.
    pub fn predict<F>(&mut self, f: F, dt: f64)
    where
        F: Fn(&[f64], f64) -> Vec<f64>,
    {
        let sigma = self.sigma_points();
        let (wm, wc) = self.weights();

        // Propagate sigma points
        let propagated: Vec<Vec<f64>> = sigma.iter().map(|s| f(s, dt)).collect();

        // Predicted state
        self.state = vec![0.0; self.state_dim];
        for (i, p) in propagated.iter().enumerate() {
            for j in 0..self.state_dim {
                self.state[j] += wm[i] * p[j];
            }
        }

        // Predicted covariance
        self.covariance = vec![vec![0.0; self.state_dim]; self.state_dim];
        for (i, p) in propagated.iter().enumerate() {
            let diff: Vec<f64> = p.iter().zip(self.state.iter()).map(|(a, b)| a - b).collect();
            for j in 0..self.state_dim {
                for k in 0..self.state_dim {
                    self.covariance[j][k] += wc[i] * diff[j] * diff[k];
                }
            }
        }

        // Add process noise
        for i in 0..self.state_dim {
            for j in 0..self.state_dim {
                self.covariance[i][j] += self.process_noise[i][j];
            }
        }
    }

    /// Update step.
    pub fn update<H>(&mut self, measurement: &[f64], h: H)
    where
        H: Fn(&[f64]) -> Vec<f64>,
    {
        let sigma = self.sigma_points();
        let (wm, wc) = self.weights();

        // Predicted observations
        let predicted_obs: Vec<Vec<f64>> = sigma.iter().map(|s| h(s)).collect();

        // Mean predicted observation
        let mut z_mean = vec![0.0; self.obs_dim];
        for (i, p) in predicted_obs.iter().enumerate() {
            for j in 0..self.obs_dim {
                z_mean[j] += wm[i] * p[j];
            }
        }

        // Innovation covariance S
        let mut s = vec![vec![0.0; self.obs_dim]; self.obs_dim];
        for (i, p) in predicted_obs.iter().enumerate() {
            let dz: Vec<f64> = p.iter().zip(z_mean.iter()).map(|(a, b)| a - b).collect();
            for j in 0..self.obs_dim {
                for k in 0..self.obs_dim {
                    s[j][k] += wc[i] * dz[j] * dz[k];
                }
            }
        }
        for i in 0..self.obs_dim {
            for j in 0..self.obs_dim {
                s[i][j] += self.measurement_noise[i][j];
            }
        }

        // Cross covariance Pxz
        let mut pxz = vec![vec![0.0; self.obs_dim]; self.state_dim];
        for (i, p) in propagated_sigma(sigma.len(), &sigma, &predicted_obs, &self.state, &z_mean).iter().enumerate() {
            for j in 0..self.state_dim {
                for k in 0..self.obs_dim {
                    pxz[j][k] += wc[i] * p.0[j] * p.1[k];
                }
            }
        }

        // Kalman gain K = Pxz * S^-1
        if let Some(s_inv) = Self::mat_invert(&s) {
            let k = Self::mat_mul(&pxz, &s_inv);

            // Innovation
            let innovation: Vec<f64> = measurement.iter().zip(z_mean.iter()).map(|(z, h)| z - h).collect();

            // Update state
            let ky = Self::mat_vec_mul(&k, &innovation);
            for i in 0..self.state_dim {
                self.state[i] += ky[i];
            }

            // Update covariance
            let ks = Self::mat_mul(&k, &s);
            let kt = Self::transpose(&k);
            let kskt = Self::mat_mul(&ks, &kt);
            for i in 0..self.state_dim {
                for j in 0..self.state_dim {
                    self.covariance[i][j] -= kskt[i][j];
                }
            }
        }
    }

    fn cholesky(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = m.len();
        let mut l = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i][k] * l[j][k];
                }

                if i == j {
                    l[i][j] = (m[i][i] - sum).sqrt();
                } else {
                    l[i][j] = (m[i][j] - sum) / l[j][j];
                }
            }
        }

        l
    }

    fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = a.len();
        let cols = b[0].len();
        let inner = a[0].len();
        let mut result = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                for k in 0..inner {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        result
    }

    fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        m.iter().map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
    }

    fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = m.len();
        let cols = m[0].len();
        let mut t = vec![vec![0.0; rows]; cols];
        for i in 0..rows {
            for j in 0..cols {
                t[j][i] = m[i][j];
            }
        }
        t
    }

    fn mat_invert(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        let n = matrix.len();
        let mut aug = vec![vec![0.0; 2 * n]; n];
        for i in 0..n {
            for j in 0..n { aug[i][j] = matrix[i][j]; }
            aug[i][n + i] = 1.0;
        }

        for col in 0..n {
            let mut max_row = col;
            for row in (col + 1)..n {
                if aug[row][col].abs() > aug[max_row][col].abs() {
                    max_row = row;
                }
            }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            if pivot.abs() < 1e-10 { return None; }

            for j in 0..(2 * n) { aug[col][j] /= pivot; }

            for row in 0..n {
                if row == col { continue; }
                let factor = aug[row][col];
                for j in 0..(2 * n) { aug[row][j] -= factor * aug[col][j]; }
            }
        }

        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n { inv[i][j] = aug[i][n + j]; }
        }
        Some(inv)
    }
}

fn propagated_sigma(n: usize, sigma: &[Vec<f64>], predicted_obs: &[Vec<f64>], state: &[f64], z_mean: &[f64]) -> Vec<(Vec<f64>, Vec<f64>)> {
    sigma.iter().zip(predicted_obs.iter()).map(|(s, o)| {
        let dx: Vec<f64> = s.iter().zip(state.iter()).map(|(a, b)| a - b).collect();
        let dz: Vec<f64> = o.iter().zip(z_mean.iter()).map(|(a, b)| a - b).collect();
        (dx, dz)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ekf() {
        let mut ekf = ExtendedKalmanFilter::new(2, 1);
        ekf.state = vec![0.0, 1.0];

        let f = |state: &[f64], dt: f64| -> Vec<f64> {
            vec![state[0] + state[1] * dt, state[1]]
        };

        let jacobian_f = vec![
            vec![1.0, 0.1],
            vec![0.0, 1.0],
        ];

        ekf.predict(f, &jacobian_f, 0.1);
        assert!(ekf.state[0] > 0.0);
    }

    #[test]
    fn test_ukf() {
        let mut ukf = UnscentedKalmanFilter::new(2, 1);
        ukf.state = vec![0.0, 1.0];

        let f = |state: &[f64], dt: f64| -> Vec<f64> {
            vec![state[0] + state[1] * dt, state[1]]
        };

        ukf.predict(f, 0.1);
        assert!(ukf.state[0] > 0.0);
    }
}
