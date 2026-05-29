/// Time series analysis: ARIMA, exponential smoothing, Holt-Winters.

/// Simple Exponential Smoothing.
pub struct SimpleExpSmoothing {
    pub alpha: f64,
}

impl SimpleExpSmoothing {
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0);
        Self { alpha }
    }

    pub fn fit(&self, data: &[f64]) -> Vec<f64> {
        if data.is_empty() { return Vec::new(); }

        let mut smoothed = vec![data[0]];
        for i in 1..data.len() {
            let s = self.alpha * data[i] + (1.0 - self.alpha) * smoothed[i - 1];
            smoothed.push(s);
        }
        smoothed
    }

    pub fn forecast(&self, data: &[f64], steps: usize) -> Vec<f64> {
        let smoothed = self.fit(data);
        let last = *smoothed.last().unwrap();
        vec![last; steps]
    }
}

/// Double Exponential Smoothing (Holt's method).
pub struct HoltMethod {
    pub alpha: f64,
    pub beta: f64,
}

impl HoltMethod {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    pub fn fit(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        if data.len() < 2 { return (data.to_vec(), vec![0.0; data.len()]); }

        let mut level = vec![data[0]];
        let mut trend = vec![data[1] - data[0]];

        for i in 1..data.len() {
            let l = self.alpha * data[i] + (1.0 - self.alpha) * (level[i - 1] + trend[i - 1]);
            let b = self.beta * (l - level[i - 1]) + (1.0 - self.beta) * trend[i - 1];
            level.push(l);
            trend.push(b);
        }

        (level, trend)
    }

    pub fn forecast(&self, data: &[f64], steps: usize) -> Vec<f64> {
        let (level, trend) = self.fit(data);
        let l = *level.last().unwrap();
        let b = *trend.last().unwrap();

        (1..=steps).map(|h| l + h as f64 * b).collect()
    }
}

/// Triple Exponential Smoothing (Holt-Winters).
pub struct HoltWinters {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub season_length: usize,
    pub additive: bool,
}

impl HoltWinters {
    pub fn new(alpha: f64, beta: f64, gamma: f64, season_length: usize, additive: bool) -> Self {
        Self { alpha, beta, gamma, season_length, additive }
    }

    pub fn fit(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = data.len();
        let m = self.season_length;
        assert!(n >= 2 * m, "Need at least 2 seasons of data");

        let mut level = vec![0.0; n];
        let mut trend = vec![0.0; n];
        let mut seasonal = vec![0.0; n];

        // Initialize: average of first season for level, average difference for trend
        let first_season_avg: f64 = data[..m].iter().sum::<f64>() / m as f64;
        level[m - 1] = first_season_avg;

        let second_season_avg: f64 = data[m..2 * m].iter().sum::<f64>() / m as f64;
        trend[m - 1] = (second_season_avg - first_season_avg) / m as f64;

        // Initialize seasonal
        if self.additive {
            for i in 0..m {
                seasonal[i] = data[i] - first_season_avg;
            }
        } else {
            for i in 0..m {
                seasonal[i] = data[i] / first_season_avg;
            }
        }

        // Recursive update
        for i in m..n {
            if self.additive {
                level[i] = self.alpha * (data[i] - seasonal[i - m]) + (1.0 - self.alpha) * (level[i - 1] + trend[i - 1]);
                trend[i] = self.beta * (level[i] - level[i - 1]) + (1.0 - self.beta) * trend[i - 1];
                seasonal[i] = self.gamma * (data[i] - level[i]) + (1.0 - self.gamma) * seasonal[i - m];
            } else {
                level[i] = self.alpha * (data[i] / seasonal[i - m]) + (1.0 - self.alpha) * (level[i - 1] + trend[i - 1]);
                trend[i] = self.beta * (level[i] - level[i - 1]) + (1.0 - self.beta) * trend[i - 1];
                seasonal[i] = self.gamma * (data[i] / level[i]) + (1.0 - self.gamma) * seasonal[i - m];
            }
        }

        (level, trend, seasonal)
    }

    pub fn forecast(&self, data: &[f64], steps: usize) -> Vec<f64> {
        let (level, trend, seasonal) = self.fit(data);
        let n = data.len();
        let m = self.season_length;
        let l = *level.last().unwrap();
        let b = *trend.last().unwrap();

        (1..=steps).map(|h| {
            let season_idx = (n - m + ((h - 1) % m));
            let s = seasonal[season_idx];
            if self.additive {
                l + h as f64 * b + s
            } else {
                (l + h as f64 * b) * s
            }
        }).collect()
    }
}

/// ARIMA(p, d, q) model.
pub struct ARIMA {
    pub p: usize, // AR order
    pub d: usize, // Differencing order
    pub q: usize, // MA order
    pub ar_coeffs: Vec<f64>,
    pub ma_coeffs: Vec<f64>,
    pub intercept: f64,
}

impl ARIMA {
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            p, d, q,
            ar_coeffs: vec![0.0; p],
            ma_coeffs: vec![0.0; q],
            intercept: 0.0,
        }
    }

    /// Difference the series d times.
    pub fn difference(data: &[f64], d: usize) -> Vec<f64> {
        let mut result = data.to_vec();
        for _ in 0..d {
            let diff: Vec<f64> = result.windows(2).map(|w| w[1] - w[0]).collect();
            result = diff;
        }
        result
    }

    /// Inverse differencing.
    pub fn undifference(diffed: &[f64], original: &[f64], d: usize) -> Vec<f64> {
        let mut result = diffed.to_vec();
        for _ in 0..d {
            let mut undiffed = vec![original[original.len() - result.len() - 1]];
            for &val in &result {
                undiffed.push(undiffed.last().unwrap() + val);
            }
            result = undiffed;
        }
        result
    }

    /// Fit using conditional least squares (simplified).
    pub fn fit(&mut self, data: &[f64]) {
        let diffed = Self::difference(data, self.d);
        let n = diffed.len();
        let max_lag = self.p.max(self.q);

        if n <= max_lag { return; }

        // Simple estimation: use autocorrelation
        let mean: f64 = diffed.iter().sum::<f64>() / n as f64;
        self.intercept = mean;

        // Estimate AR coefficients using Yule-Walker
        if self.p > 0 {
            let acf = self.autocorrelation(&diffed, self.p);
            // Solve Toeplitz system (simplified: use first p autocorrelations)
            for i in 0..self.p {
                self.ar_coeffs[i] = acf.get(i + 1).copied().unwrap_or(0.0);
            }
        }

        // MA coefficients (simplified: set to small values)
        for i in 0..self.q {
            self.ma_coeffs[i] = 0.1;
        }
    }

    /// One-step-ahead prediction.
    pub fn predict_one(&self, data: &[f64]) -> f64 {
        let diffed = Self::difference(data, self.d);
        let n = diffed.len();
        let mean: f64 = diffed.iter().sum::<f64>() / n as f64;

        let mut pred = self.intercept;

        // AR component
        for i in 0..self.p {
            if n > i {
                pred += self.ar_coeffs[i] * (diffed[n - 1 - i] - mean);
            }
        }

        // MA component (simplified: use zero for innovations)
        // In practice, would track past prediction errors

        pred + mean
    }

    /// Forecast multiple steps ahead.
    pub fn forecast(&self, data: &[f64], steps: usize) -> Vec<f64> {
        let mut forecasts = Vec::new();
        let mut extended = data.to_vec();

        for _ in 0..steps {
            let pred = self.predict_one(&extended);
            forecasts.push(pred);
            extended.push(pred);
        }

        // Inverse difference if needed
        if self.d > 0 {
            Self::undifference(&forecasts, data, self.d)[1..].to_vec()
        } else {
            forecasts
        }
    }

    fn autocorrelation(&self, data: &[f64], max_lag: usize) -> Vec<f64> {
        let n = data.len();
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        if var == 0.0 { return vec![0.0; max_lag]; }

        (1..=max_lag).map(|lag| {
            let cov: f64 = (0..n - lag).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / n as f64;
            cov / var
        }).collect()
    }
}

/// ARIMA model selection using AIC.
pub fn select_arima(data: &[f64], max_p: usize, max_d: usize, max_q: usize) -> (usize, usize, usize) {
    let mut best_aic = f64::INFINITY;
    let mut best_order = (0, 0, 0);

    for d in 0..=max_d {
        let diffed = ARIMA::difference(data, d);
        let n = diffed.len();

        for p in 0..=max_p {
            for q in 0..=max_q {
                if p + q == 0 { continue; }

                let mut model = ARIMA::new(p, d, q);
                model.fit(data);

                // Compute residual variance
                let residuals = compute_residuals(&diffed, &model);
                let mse: f64 = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;

                if mse > 0.0 {
                    let k = p + q + 1;
                    let aic = n as f64 * mse.ln() + 2.0 * k as f64;
                    if aic < best_aic {
                        best_aic = aic;
                        best_order = (p, d, q);
                    }
                }
            }
        }
    }

    best_order
}

fn compute_residuals(data: &[f64], model: &ARIMA) -> Vec<f64> {
    let n = data.len();
    let max_lag = model.p.max(model.q);
    if n <= max_lag { return vec![0.0; n]; }

    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let mut residuals = vec![0.0; n];

    for i in max_lag..n {
        let mut pred = model.intercept;
        for j in 0..model.p {
            pred += model.ar_coeffs[j] * (data[i - 1 - j] - mean);
        }
        residuals[i] = data[i] - pred - mean;
    }

    residuals
}

/// Autoregressive model of order p.
pub struct AR {
    pub p: usize,
    pub coefficients: Vec<f64>,
    pub intercept: f64,
}

impl AR {
    pub fn new(p: usize) -> Self {
        Self { p, coefficients: vec![0.0; p], intercept: 0.0 }
    }

    pub fn fit(&mut self, data: &[f64]) {
        let n = data.len();
        if n <= self.p { return; }

        // Solve Yule-Walker equations
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        self.intercept = mean;

        let acf = self.autocorrelation(data, self.p);

        // Levinson-Durbin recursion for Toeplitz system
        let mut r = vec![1.0; self.p + 1];
        for i in 1..=self.p {
            r[i] = acf[i - 1];
        }

        let mut a = vec![0.0; self.p + 1];
        let mut e = r[0];

        for i in 1..=self.p {
            let mut sum = r[i];
            for j in 1..i {
                sum -= a[j] * r[i - j];
            }
            let k = sum / e;
            a[i] = k;
            for j in 1..i {
                let temp = a[j] - k * a[i - j];
                a[j] = temp;
            }
            e *= 1.0 - k * k;
        }

        self.coefficients = a[1..].to_vec();
    }

    pub fn predict_one(&self, data: &[f64]) -> f64 {
        let n = data.len();
        let mean = self.intercept;
        let mut pred = mean;
        for i in 0..self.p {
            if n > i {
                pred += self.coefficients[i] * (data[n - 1 - i] - mean);
            }
        }
        pred
    }

    pub fn forecast(&self, data: &[f64], steps: usize) -> Vec<f64> {
        let mut forecasts = Vec::new();
        let mut extended = data.to_vec();
        for _ in 0..steps {
            let pred = self.predict_one(&extended);
            forecasts.push(pred);
            extended.push(pred);
        }
        forecasts
    }

    fn autocorrelation(&self, data: &[f64], max_lag: usize) -> Vec<f64> {
        let n = data.len();
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        if var == 0.0 { return vec![0.0; max_lag]; }
        (1..=max_lag).map(|lag| {
            let cov: f64 = (0..n - lag).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / n as f64;
            cov / var
        }).collect()
    }
}

/// Moving average for trend extraction.
pub fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window { return data.to_vec(); }
    data.windows(window).map(|w| w.iter().sum::<f64>() / window as f64).collect()
}

/// Exponentially weighted moving average.
pub fn ewma(data: &[f64], span: usize) -> Vec<f64> {
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut result = vec![data[0]];
    for i in 1..data.len() {
        result.push(alpha * data[i] + (1.0 - alpha) * result[i - 1]);
    }
    result
}

/// Seasonal decomposition (simple).
pub fn seasonal_decompose(data: &[f64], period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    if n < 2 * period {
        return (data.to_vec(), vec![0.0; n], vec![0.0; n]);
    }

    // Trend: centered moving average
    let trend = moving_average(data, period);
    let offset = (period - 1) / 2;
    let mut full_trend = vec![0.0; n];
    for i in 0..trend.len() {
        if i + offset < n {
            full_trend[i + offset] = trend[i];
        }
    }

    // Detrend
    let detrended: Vec<f64> = data.iter().zip(full_trend.iter()).map(|(d, t)| d - t).collect();

    // Seasonal: average by position in season
    let mut seasonal_means = vec![0.0; period];
    let mut counts = vec![0usize; period];
    for i in 0..n {
        if full_trend[i] != 0.0 {
            seasonal_means[i % period] += detrended[i];
            counts[i % period] += 1;
        }
    }
    for i in 0..period {
        if counts[i] > 0 {
            seasonal_means[i] /= counts[i] as f64;
        }
    }

    let seasonal: Vec<f64> = (0..n).map(|i| seasonal_means[i % period]).collect();
    let residual: Vec<f64> = data.iter().zip(full_trend.iter()).zip(seasonal.iter())
        .map(|((d, t), s)| d - t - s)
        .collect();

    (full_trend, seasonal, residual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_exp_smoothing() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ses = SimpleExpSmoothing::new(0.3);
        let smoothed = ses.fit(&data);
        assert_eq!(smoothed.len(), 5);
        assert_eq!(smoothed[0], 1.0);
    }

    #[test]
    fn test_holt() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let holt = HoltMethod::new(0.3, 0.1);
        let forecast = holt.forecast(&data, 3);
        assert_eq!(forecast.len(), 3);
        assert!(forecast[0] > 7.0);
    }

    #[test]
    fn test_holt_winters() {
        let data: Vec<f64> = (0..24).map(|i| {
            (i as f64 * 0.5) + (i % 4) as f64 * 2.0
        }).collect();
        let hw = HoltWinters::new(0.3, 0.1, 0.1, 4, true);
        let forecast = hw.forecast(&data, 4);
        assert_eq!(forecast.len(), 4);
    }

    #[test]
    fn test_arima() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() + 0.5 * (i as f64 * 0.3).cos()).collect();
        let mut arima = ARIMA::new(2, 1, 1);
        arima.fit(&data);
        let forecast = arima.forecast(&data, 5);
        assert_eq!(forecast.len(), 5);
    }

    #[test]
    fn test_seasonal_decompose() {
        let data: Vec<f64> = (0..48).map(|i| {
            i as f64 + (i % 12) as f64 * 2.0 + (i as f64 * 0.1).sin()
        }).collect();
        let (trend, seasonal, residual) = seasonal_decompose(&data, 12);
        assert_eq!(trend.len(), 48);
        assert_eq!(seasonal.len(), 48);
        assert_eq!(residual.len(), 48);
    }
}
