/// Stochastic processes: Brownian motion, Poisson process, geometric Brownian motion.

/// Standard Brownian motion generator.
pub struct BrownianMotion {
    seed: u64,
    pub dt: f64,
}

impl BrownianMotion {
    pub fn new(dt: f64) -> Self {
        Self { seed: 42, dt }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generate a path of length n.
    pub fn generate(&mut self, n: usize) -> Vec<f64> {
        let mut path = vec![0.0];
        for _ in 0..n {
            let dw = self.gaussian() * self.dt.sqrt();
            path.push(path.last().unwrap() + dw);
        }
        path
    }

    /// Generate multiple independent paths.
    pub fn generate_paths(&mut self, n_paths: usize, n_steps: usize) -> Vec<Vec<f64>> {
        (0..n_paths).map(|_| self.generate(n_steps)).collect()
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.pseudo_rand().max(1e-10);
        let u2 = self.pseudo_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Geometric Brownian motion: dS = mu*S*dt + sigma*S*dW.
pub struct GeometricBrownianMotion {
    pub mu: f64,
    pub sigma: f64,
    pub dt: f64,
    seed: u64,
}

impl GeometricBrownianMotion {
    pub fn new(mu: f64, sigma: f64, dt: f64) -> Self {
        Self { mu, sigma, dt, seed: 42 }
    }

    pub fn generate(&mut self, s0: f64, n: usize) -> Vec<f64> {
        let mut path = vec![s0];
        for _ in 0..n {
            let s = *path.last().unwrap();
            let dw = self.gaussian() * self.dt.sqrt();
            let ds = self.mu * s * self.dt + self.sigma * s * dw;
            path.push(s + ds);
        }
        path
    }

    /// Generate using exact solution: S(t) = S0 * exp((mu - sigma^2/2)t + sigma*W(t)).
    pub fn generate_exact(&mut self, s0: f64, t: f64, n: usize) -> Vec<f64> {
        let dt = t / n as f64;
        let mut path = vec![s0];
        let mut w = 0.0;

        for i in 0..n {
            w += self.gaussian() * dt.sqrt();
            let s = s0 * ((self.mu - 0.5 * self.sigma * self.sigma) * (i + 1) as f64 * dt + self.sigma * w).exp();
            path.push(s);
        }

        path
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.pseudo_rand().max(1e-10);
        let u2 = self.pseudo_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Poisson process.
pub struct PoissonProcess {
    pub rate: f64,
    seed: u64,
}

impl PoissonProcess {
    pub fn new(rate: f64) -> Self {
        Self { rate, seed: 42 }
    }

    /// Generate inter-arrival times.
    pub fn generate_interarrivals(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| {
            let u = self.pseudo_rand().max(1e-10);
            -u.ln() / self.rate
        }).collect()
    }

    /// Generate event times up to time T.
    pub fn generate_events(&mut self, t_max: f64) -> Vec<f64> {
        let mut events = Vec::new();
        let mut t = 0.0;

        loop {
            let u = self.pseudo_rand().max(1e-10);
            t += -u.ln() / self.rate;
            if t > t_max { break; }
            events.push(t);
        }

        events
    }

    /// Count process: N(t) = number of events up to time t.
    pub fn count_at_time(&mut self, t: f64) -> usize {
        let events = self.generate_events(t);
        events.len()
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Ornstein-Uhlenbeck process: dX = theta*(mu - X)*dt + sigma*dW.
pub struct OrnsteinUhlenbeck {
    pub theta: f64,
    pub mu: f64,
    pub sigma: f64,
    pub dt: f64,
    seed: u64,
}

impl OrnsteinUhlenbeck {
    pub fn new(theta: f64, mu: f64, sigma: f64, dt: f64) -> Self {
        Self { theta, mu, sigma, dt, seed: 42 }
    }

    pub fn generate(&mut self, x0: f64, n: usize) -> Vec<f64> {
        let mut path = vec![x0];
        for _ in 0..n {
            let x = *path.last().unwrap();
            let dx = self.theta * (self.mu - x) * self.dt + self.sigma * self.gaussian() * self.dt.sqrt();
            path.push(x + dx);
        }
        path
    }

    /// Variance at time t.
    pub fn variance(&self, t: f64) -> f64 {
        self.sigma * self.sigma / (2.0 * self.theta) * (1.0 - (-2.0 * self.theta * t).exp())
    }

    /// Autocorrelation at lag tau.
    pub fn autocorrelation(&self, tau: f64) -> f64 {
        (-self.theta * tau.abs()).exp()
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.pseudo_rand().max(1e-10);
        let u2 = self.pseudo_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Merton jump-diffusion model: dS = mu*S*dt + sigma*S*dW + J*S*dN.
pub struct MertonJumpDiffusion {
    pub mu: f64,
    pub sigma: f64,
    pub jump_intensity: f64,
    pub jump_mean: f64,
    pub jump_std: f64,
    pub dt: f64,
    seed: u64,
}

impl MertonJumpDiffusion {
    pub fn new(mu: f64, sigma: f64, jump_intensity: f64, jump_mean: f64, jump_std: f64, dt: f64) -> Self {
        Self { mu, sigma, jump_intensity, jump_mean, jump_std, dt, seed: 42 }
    }

    pub fn generate(&mut self, s0: f64, n: usize) -> Vec<f64> {
        let mut path = vec![s0];
        for _ in 0..n {
            let s = *path.last().unwrap();
            let dw = self.gaussian() * self.dt.sqrt();

            // Jump component
            let mut jump = 0.0;
            if self.pseudo_rand() < self.jump_intensity * self.dt {
                jump = self.jump_mean + self.jump_std * self.gaussian();
            }

            let ds = self.mu * s * self.dt + self.sigma * s * dw + s * jump;
            path.push((s + ds).max(0.0));
        }
        path
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.pseudo_rand().max(1e-10);
        let u2 = self.pseudo_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Cox-Ingersoll-Ross process: dX = theta*(mu - X)*dt + sigma*sqrt(X)*dW.
pub struct CIRProcess {
    pub theta: f64,
    pub mu: f64,
    pub sigma: f64,
    pub dt: f64,
    seed: u64,
}

impl CIRProcess {
    pub fn new(theta: f64, mu: f64, sigma: f64, dt: f64) -> Self {
        Self { theta, mu, sigma, dt, seed: 42 }
    }

    pub fn generate(&mut self, x0: f64, n: usize) -> Vec<f64> {
        let mut path = vec![x0.max(0.0)];
        for _ in 0..n {
            let x = *path.last().unwrap();
            let sqrt_x = x.max(0.0).sqrt();
            let dx = self.theta * (self.mu - x) * self.dt + self.sigma * sqrt_x * self.gaussian() * self.dt.sqrt();
            path.push((x + dx).max(0.0));
        }
        path
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.pseudo_rand().max(1e-10);
        let u2 = self.pseudo_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Heston stochastic volatility model.
pub struct HestonModel {
    pub kappa: f64,   // Mean reversion speed
    pub theta: f64,   // Long-run variance
    pub xi: f64,      // Vol of vol
    pub rho: f64,     // Correlation
    pub r: f64,       // Risk-free rate
    pub dt: f64,
    seed: u64,
}

impl HestonModel {
    pub fn new(kappa: f64, theta: f64, xi: f64, rho: f64, r: f64, dt: f64) -> Self {
        Self { kappa, theta, xi, rho, r, dt, seed: 42 }
    }

    pub fn generate(&mut self, s0: f64, v0: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut s_path = vec![s0];
        let mut v_path = vec![v0.max(0.0)];

        for _ in 0..n {
            let s = *s_path.last().unwrap();
            let v = *v_path.last().unwrap().max(0.0);

            let z1 = self.gaussian();
            let z2 = self.rho * z1 + (1.0 - self.rho * self.rho).sqrt() * self.gaussian();

            let sqrt_v = v.max(0.0).sqrt();
            let ds = self.r * s * self.dt + sqrt_v * s * z1 * self.dt.sqrt();
            let dv = self.kappa * (self.theta - v) * self.dt + self.xi * sqrt_v * z2 * self.dt.sqrt();

            s_path.push(s + ds);
            v_path.push((v + dv).max(0.0));
        }

        (s_path, v_path)
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.pseudo_rand().max(1e-10);
        let u2 = self.pseudo_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Fractional Brownian motion (Cholesky method).
pub struct FractionalBrownianMotion {
    pub hurst: f64,
    seed: u64,
}

impl FractionalBrownianMotion {
    pub fn new(hurst: f64) -> Self {
        assert!(hurst > 0.0 && hurst < 1.0, "Hurst parameter must be in (0, 1)");
        Self { hurst, seed: 42 }
    }

    /// Generate fBm path using Cholesky decomposition of covariance matrix.
    pub fn generate(&mut self, n: usize) -> Vec<f64> {
        let h = self.hurst;

        // Covariance function: R(s,t) = 0.5 * (|s|^(2H) + |t|^(2H) - |s-t|^(2H))
        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let s = (i + 1) as f64;
                let t = (j + 1) as f64;
                cov[i][j] = 0.5 * (s.powf(2.0 * h) + t.powf(2.0 * h) - (s - t).abs().powf(2.0 * h));
            }
        }

        // Cholesky decomposition
        let l = Self::cholesky(&cov);

        // Generate independent normals
        let z: Vec<f64> = (0..n).map(|_| self.gaussian()).collect();

        // Multiply L * z
        let mut path = vec![0.0];
        for i in 0..n {
            let val: f64 = l[i].iter().zip(z.iter()).map(|(a, b)| a * b).sum();
            path.push(val);
        }

        path
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

    fn gaussian(&mut self) -> f64 {
        let u1 = self.pseudo_rand().max(1e-10);
        let u2 = self.pseudo_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
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
    fn test_brownian() {
        let mut bm = BrownianMotion::new(0.01);
        let path = bm.generate(100);
        assert_eq!(path.len(), 101);
        assert_eq!(path[0], 0.0);
    }

    #[test]
    fn test_gbm() {
        let mut gbm = GeometricBrownianMotion::new(0.05, 0.2, 0.01);
        let path = gbm.generate(100.0, 100);
        assert_eq!(path.len(), 101);
        assert_eq!(path[0], 100.0);
        // Prices should be positive
        assert!(path.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn test_poisson() {
        let mut pp = PoissonProcess::new(10.0);
        let events = pp.generate_events(1.0);
        // Expected ~10 events in time 1.0
        assert!(events.len() > 3 && events.len() < 20);
    }

    #[test]
    fn test_ou_process() {
        let mut ou = OrnsteinUhlenbeck::new(1.0, 0.0, 0.3, 0.01);
        let path = ou.generate(0.0, 1000);
        assert_eq!(path.len(), 1001);
        // Should revert to mean
        let mean: f64 = path[500..].iter().sum::<f64>() / 501.0;
        assert!(mean.abs() < 2.0);
    }
}
