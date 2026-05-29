/// Function approximation: interpolation, splines, polynomial fitting.

/// Linear interpolation.
pub fn linear_interp(x: &[f64], y: &[f64], xi: f64) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();

    if xi <= x[0] { return y[0]; }
    if xi >= x[n - 1] { return y[n - 1]; }

    // Binary search for interval
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x[mid] <= xi { lo = mid; } else { hi = mid; }
    }

    let t = (xi - x[lo]) / (x[hi] - x[lo]);
    y[lo] + t * (y[hi] - y[lo])
}

/// Polynomial interpolation (Lagrange).
pub fn lagrange_interp(x: &[f64], y: &[f64], xi: f64) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut result = 0.0;

    for i in 0..n {
        let mut term = y[i];
        for j in 0..n {
            if i != j {
                term *= (xi - x[j]) / (x[i] - x[j]);
            }
        }
        result += term;
    }

    result
}

/// Newton's divided difference interpolation.
pub struct NewtonInterpolator {
    pub x: Vec<f64>,
    pub coefficients: Vec<f64>,
}

impl NewtonInterpolator {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        let mut coefficients = y.clone();

        for j in 1..n {
            for i in (j..n).rev() {
                coefficients[i] = (coefficients[i] - coefficients[i - 1]) / (x[i] - x[i - j]);
            }
        }

        Self { x, coefficients }
    }

    pub fn evaluate(&self, xi: f64) -> f64 {
        let n = self.coefficients.len();
        let mut result = self.coefficients[n - 1];

        for i in (0..n - 1).rev() {
            result = result * (xi - self.x[i]) + self.coefficients[i];
        }

        result
    }
}

/// Cubic spline interpolation.
pub struct CubicSpline {
    pub x: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: Vec<f64>,
}

impl CubicSpline {
    /// Natural cubic spline (second derivatives at endpoints are zero).
    pub fn natural(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        let m = n - 1;

        // Compute h_i = x_{i+1} - x_i
        let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();

        // Set up tridiagonal system for c_i (second derivatives / 6)
        let mut alpha = vec![0.0; n];
        for i in 1..m {
            alpha[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
        }

        // Solve tridiagonal system
        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];

        for i in 1..m {
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        let mut c = vec![0.0; n];
        let mut b = vec![0.0; m];
        let mut d = vec![0.0; m];

        for j in (0..m).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }

        Self { x, a: y[..m].to_vec(), b, c: c[..m].to_vec(), d }
    }

    /// Clamped cubic spline (specified first derivatives at endpoints).
    pub fn clamped(x: Vec<f64>, y: Vec<f64>, d0: f64, dn: f64) -> Self {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        let m = n - 1;

        let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();

        let mut alpha = vec![0.0; n];
        alpha[0] = 3.0 * ((y[1] - y[0]) / h[0] - d0);
        alpha[m] = 3.0 * (dn - (y[m] - y[m - 1]) / h[m - 1]);
        for i in 1..m {
            alpha[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
        }

        let mut l = vec![0.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];

        l[0] = 2.0 * h[0];
        mu[0] = 0.5;
        z[0] = alpha[0] / l[0];

        for i in 1..m {
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        l[m] = h[m - 1] * (2.0 - mu[m - 1]);
        z[m] = (alpha[m] - h[m - 1] * z[m - 1]) / l[m];

        let mut c = vec![0.0; n];
        c[m] = z[m];
        let mut b = vec![0.0; m];
        let mut d = vec![0.0; m];

        for j in (0..m).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }

        Self { x, a: y[..m].to_vec(), b, c: c[..m].to_vec(), d }
    }

    pub fn evaluate(&self, xi: f64) -> f64 {
        let n = self.x.len();

        if xi <= self.x[0] { return self.a[0]; }
        if xi >= self.x[n - 1] { return self.a[self.a.len() - 1]; }

        // Find interval
        let mut idx = 0;
        for i in 0..self.a.len() {
            if xi < self.x[i + 1] { idx = i; break; }
        }

        let dx = xi - self.x[idx];
        self.a[idx] + self.b[idx] * dx + self.c[idx] * dx * dx + self.d[idx] * dx * dx * dx
    }

    pub fn evaluate_derivative(&self, xi: f64) -> f64 {
        let n = self.x.len();

        if xi <= self.x[0] { return self.b[0]; }
        if xi >= self.x[n - 1] { return self.b[self.b.len() - 1]; }

        let mut idx = 0;
        for i in 0..self.a.len() {
            if xi < self.x[i + 1] { idx = i; break; }
        }

        let dx = xi - self.x[idx];
        self.b[idx] + 2.0 * self.c[idx] * dx + 3.0 * self.d[idx] * dx * dx
    }
}

/// Least squares polynomial fit.
pub fn poly_fit(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    let m = degree + 1;

    // Build Vandermonde matrix
    let mut a = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            a[i][j] = x[i].powi(j as i32);
        }
    }

    // A^T * A
    let mut ata = vec![vec![0.0; m]; m];
    let mut aty = vec![0.0; m];
    for i in 0..m {
        for j in 0..m {
            for k in 0..n {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
        for k in 0..n {
            aty[i] += a[k][i] * y[k];
        }
    }

    // Solve normal equations
    solve_linear_system(&mut ata, &mut aty).unwrap_or(vec![0.0; m])
}

fn solve_linear_system(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> Option<Vec<f64>> {
    let n = a.len();

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[max_row][col].abs() {
                max_row = row;
            }
        }
        a.swap(col, max_row);
        b.swap(col, max_row);

        let pivot = a[col][col];
        if pivot.abs() < 1e-12 { return None; }

        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in (i + 1)..n {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }

    Some(x)
}

/// Evaluate polynomial at a point.
pub fn poly_eval(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// B-spline basis functions.
pub struct BSpline {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub control_points: Vec<f64>,
}

impl BSpline {
    pub fn new(degree: usize, knots: Vec<f64>, control_points: Vec<f64>) -> Self {
        Self { degree, knots, control_points }
    }

    /// Evaluate B-spline at parameter t using de Boor's algorithm.
    pub fn evaluate(&self, t: f64) -> f64 {
        let n = self.control_points.len();
        let k = self.degree;

        // Find knot span
        let mut span = k;
        for i in k..n {
            if t < self.knots[i + 1] {
                span = i;
                break;
            }
        }

        // Initialize
        let mut d: Vec<f64> = (0..=k).map(|j| self.control_points[span - k + j]).collect();

        // de Boor's algorithm
        for r in 1..=k {
            for j in (r..=k).rev() {
                let left = span - k + j;
                let right = span + 1 + r - j;
                let denom = self.knots[right] - self.knots[left];
                if denom.abs() < 1e-15 {
                    d[j] = d[j - 1];
                } else {
                    let alpha = (t - self.knots[left]) / denom;
                    d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
                }
            }
        }

        d[k]
    }

    pub fn evaluate_range(&self, t_min: f64, t_max: f64, n_points: usize) -> Vec<(f64, f64)> {
        (0..n_points).map(|i| {
            let t = t_min + (t_max - t_min) * i as f64 / (n_points - 1) as f64;
            (t, self.evaluate(t))
        }).collect()
    }
}

/// Bezier curve.
pub struct BezierCurve {
    pub control_points: Vec<(f64, f64)>,
}

impl BezierCurve {
    pub fn new(control_points: Vec<(f64, f64)>) -> Self {
        Self { control_points }
    }

    /// Evaluate at parameter t using de Casteljau's algorithm.
    pub fn evaluate(&self, t: f64) -> (f64, f64) {
        let mut points = self.control_points.clone();

        while points.len() > 1 {
            let mut next = Vec::new();
            for i in 0..points.len() - 1 {
                let x = (1.0 - t) * points[i].0 + t * points[i + 1].0;
                let y = (1.0 - t) * points[i].1 + t * points[i + 1].1;
                next.push((x, y));
            }
            points = next;
        }

        points[0]
    }

    /// Evaluate derivative at parameter t.
    pub fn derivative(&self, t: f64) -> (f64, f64) {
        let n = self.control_points.len();
        if n < 2 { return (0.0, 0.0); }

        let derived: Vec<(f64, f64)> = (0..n - 1).map(|i| {
            let dx = (n - 1) as f64 * (self.control_points[i + 1].0 - self.control_points[i].0);
            let dy = (n - 1) as f64 * (self.control_points[i + 1].1 - self.control_points[i].1);
            (dx, dy)
        }).collect();

        BezierCurve::new(derived).evaluate(t)
    }

    pub fn evaluate_range(&self, n_points: usize) -> Vec<(f64, f64)> {
        (0..n_points).map(|i| {
            let t = i as f64 / (n_points - 1) as f64;
            self.evaluate(t)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_spline() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let spline = CubicSpline::natural(x, y);

        // Should interpolate at data points
        assert!((spline.evaluate(0.0) - 0.0).abs() < 1e-10);
        assert!((spline.evaluate(1.0) - 1.0).abs() < 1e-10);
        assert!((spline.evaluate(2.0) - 4.0).abs() < 1e-10);
        assert!((spline.evaluate(3.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_lagrange() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 4.0, 9.0];
        let val = lagrange_interp(&x, &y, 1.5);
        // For y = x^2 + 2x + 1, f(1.5) = 2.25 + 3 + 1 = 6.25
        assert!((val - 6.25).abs() < 0.01);
    }

    #[test]
    fn test_bezier() {
        let curve = BezierCurve::new(vec![(0.0, 0.0), (1.0, 2.0), (3.0, 2.0)]);
        let p0 = curve.evaluate(0.0);
        let p1 = curve.evaluate(1.0);
        assert!((p0.0 - 0.0).abs() < 1e-10);
        assert!((p0.1 - 0.0).abs() < 1e-10);
        assert!((p1.0 - 3.0).abs() < 1e-10);
        assert!((p1.1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_poly_fit() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&x| 2.0 * x * x + 3.0 * x + 1.0).collect();
        let coeffs = poly_fit(&x, &y, 2);
        assert!((coeffs[0] - 1.0).abs() < 0.1);
        assert!((coeffs[1] - 3.0).abs() < 0.1);
        assert!((coeffs[2] - 2.0).abs() < 0.1);
    }
}
