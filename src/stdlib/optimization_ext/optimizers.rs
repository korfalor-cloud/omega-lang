/// Advanced optimization: trust region, line search, BFGS, L-BFGS, conjugate gradient.

/// Backtracking line search.
pub fn backtracking_line_search<F, G>(
    f: F,
    grad: G,
    x: &[f64],
    direction: &[f64],
    initial_step: f64,
    c: f64,
    rho: f64,
    max_iter: usize,
) -> f64
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let f0 = f(x);
    let g0 = grad(x);
    let directional_grad: f64 = g0.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();

    let mut step = initial_step;
    for _ in 0..max_iter {
        let x_new: Vec<f64> = x.iter().zip(direction.iter()).map(|(xi, di)| xi + step * di).collect();
        if f(&x_new) <= f0 + c * step * directional_grad {
            return step;
        }
        step *= rho;
    }
    step
}

/// Wolfe line search (satisfies both Armijo and curvature conditions).
pub fn wolfe_line_search<F, G>(
    f: F,
    grad: G,
    x: &[f64],
    direction: &[f64],
    c1: f64,
    c2: f64,
    max_iter: usize,
) -> f64
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let f0 = f(x);
    let g0 = grad(x);
    let directional_grad: f64 = g0.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();

    let mut step = 1.0;
    let mut prev_step = 0.0;
    let mut prev_f = f0;

    for _ in 0..max_iter {
        let x_new: Vec<f64> = x.iter().zip(direction.iter()).map(|(xi, di)| xi + step * di).collect();
        let f_new = f(&x_new);

        // Armijo condition
        if f_new > f0 + c1 * step * directional_grad {
            return zoom(f, grad, x, direction, prev_step, step, f0, directional_grad, c1, c2, max_iter);
        }

        let g_new = grad(&x_new);
        let new_directional_grad: f64 = g_new.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();

        // Curvature condition
        if new_directional_grad.abs() <= -c2 * directional_grad {
            return step;
        }

        if new_directional_grad >= 0.0 {
            return zoom(f, grad, x, direction, step, prev_step, f0, directional_grad, c1, c2, max_iter);
        }

        prev_step = step;
        prev_f = f_new;
        step *= 2.0;
    }

    step
}

fn zoom<F, G>(
    f: F,
    grad: G,
    x: &[f64],
    direction: &[f64],
    mut lo: f64,
    mut hi: f64,
    f0: f64,
    directional_grad: f64,
    c1: f64,
    c2: f64,
    max_iter: usize,
) -> f64
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    for _ in 0..max_iter {
        let step = (lo + hi) / 2.0;
        let x_new: Vec<f64> = x.iter().zip(direction.iter()).map(|(xi, di)| xi + step * di).collect();
        let f_new = f(&x_new);

        if f_new > f0 + c1 * step * directional_grad || f_new >= f(&(x.iter().zip(direction.iter()).map(|(xi, di)| xi + lo * di).collect::<Vec<f64>>())) {
            hi = step;
        } else {
            let g_new = grad(&x_new);
            let new_directional_grad: f64 = g_new.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();

            if new_directional_grad.abs() <= -c2 * directional_grad {
                return step;
            }

            if new_directional_grad * (hi - lo) >= 0.0 {
                hi = lo;
            }
            lo = step;
        }
    }
    (lo + hi) / 2.0
}

/// BFGS quasi-Newton method.
pub fn bfgs<F, G>(
    f: F,
    grad: G,
    initial: &[f64],
    tol: f64,
    max_iter: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = initial.len();
    let mut x = initial.to_vec();
    let mut h_inv = identity(n);

    for _ in 0..max_iter {
        let g = grad(&x);

        // Check convergence
        let g_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
        if g_norm < tol { break; }

        // Search direction: d = -H^{-1} * g
        let direction: Vec<f64> = (0..n).map(|i| {
            -(0..n).map(|j| h_inv[i][j] * g[j]).sum::<f64>()
        }).collect();

        // Line search
        let step = backtracking_line_search(&f, &grad, &x, &direction, 1.0, 1e-4, 0.5, 50);

        // Update x
        let x_new: Vec<f64> = x.iter().zip(direction.iter()).map(|(xi, di)| xi + step * di).collect();
        let g_new = grad(&x_new);

        // BFGS update
        let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(a, b)| a - b).collect();

        let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        if sy.abs() < 1e-15 { continue; }

        let rho = 1.0 / sy;

        // H = (I - rho*s*y') * H * (I - rho*y*s') + rho*s*s'
        let mut new_h = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut val = h_inv[i][j];
                val -= rho * (s[i] * (0..n).map(|k| h_inv[k][j] * y[k]).sum::<f64>()
                    + y[i] * (0..n).map(|k| h_inv[i][k] * s[k]).sum::<f64>());
                val += rho * rho * (sy + (0..n).map(|k| y[k] * (0..n).map(|l| h_inv[k][l] * y[l]).sum::<f64>()).sum::<f64>()) * s[i] * s[j];
                val += rho * s[i] * s[j];
                new_h[i][j] = val;
            }
        }
        h_inv = new_h;
        x = x_new;
    }

    x
}

/// L-BFGS (Limited-memory BFGS).
pub fn l_bfgs<F, G>(
    f: F,
    grad: G,
    initial: &[f64],
    m: usize, // History size
    tol: f64,
    max_iter: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = initial.len();
    let mut x = initial.to_vec();
    let mut s_history: Vec<Vec<f64>> = Vec::new();
    let mut y_history: Vec<Vec<f64>> = Vec::new();
    let mut rho_history: Vec<f64> = Vec::new();

    for _ in 0..max_iter {
        let g = grad(&x);

        let g_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
        if g_norm < tol { break; }

        // Compute direction using two-loop recursion
        let mut q = g.clone();
        let k = s_history.len();
        let mut alpha = vec![0.0; k];

        for i in (0..k).rev() {
            alpha[i] = rho_history[i] * s_history[i].iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f64>();
            for j in 0..n {
                q[j] -= alpha[i] * y_history[i][j];
            }
        }

        // Initial Hessian approximation
        let gamma = if k > 0 {
            let sy = s_history[k - 1].iter().zip(y_history[k - 1].iter()).map(|(a, b)| a * b).sum::<f64>();
            let yy = y_history[k - 1].iter().map(|y| y * y).sum::<f64>();
            if yy > 1e-15 { sy / yy } else { 1.0 }
        } else {
            1.0
        };

        let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();

        for i in 0..k {
            let beta = rho_history[i] * y_history[i].iter().zip(r.iter()).map(|(a, b)| a * b).sum::<f64>();
            for j in 0..n {
                r[j] += s_history[i][j] * (alpha[i] - beta);
            }
        }

        let direction: Vec<f64> = r.iter().map(|ri| -ri).collect();

        // Line search
        let step = backtracking_line_search(&f, &grad, &x, &direction, 1.0, 1e-4, 0.5, 50);

        // Update
        let x_new: Vec<f64> = x.iter().zip(direction.iter()).map(|(xi, di)| xi + step * di).collect();
        let g_new = grad(&x_new);

        let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        if sy.abs() > 1e-15 {
            if s_history.len() >= m {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
            s_history.push(s);
            y_history.push(y);
            rho_history.push(1.0 / sy);
        }

        x = x_new;
    }

    x
}

/// Conjugate gradient method (Fletcher-Reeves).
pub fn conjugate_gradient<F, G>(
    f: F,
    grad: G,
    initial: &[f64],
    tol: f64,
    max_iter: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = initial.len();
    let mut x = initial.to_vec();
    let mut g = grad(&x);
    let mut d: Vec<f64> = g.iter().map(|gi| -gi).collect();

    for _ in 0..max_iter {
        let g_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
        if g_norm < tol { break; }

        // Line search
        let step = backtracking_line_search(&f, &grad, &x, &d, 1.0, 1e-4, 0.5, 50);

        // Update
        let x_new: Vec<f64> = x.iter().zip(d.iter()).map(|(xi, di)| xi + step * di).collect();
        let g_new = grad(&x_new);

        // Fletcher-Reeves beta
        let g_old_norm_sq: f64 = g.iter().map(|gi| gi * gi).sum();
        let g_new_norm_sq: f64 = g_new.iter().map(|gi| gi * gi).sum();

        let beta = if g_old_norm_sq > 1e-15 { g_new_norm_sq / g_old_norm_sq } else { 0.0 };

        // Update direction
        for i in 0..n {
            d[i] = -g_new[i] + beta * d[i];
        }

        x = x_new;
        g = g_new;
    }

    x
}

/// Trust region method.
pub fn trust_region<F, G, H>(
    f: F,
    grad: G,
    hessian: H,
    initial: &[f64],
    max_radius: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
    H: Fn(&[f64]) -> Vec<Vec<f64>>,
{
    let n = initial.len();
    let mut x = initial.to_vec();
    let mut radius = max_radius / 4.0;

    for _ in 0..max_iter {
        let g = grad(&x);
        let g_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
        if g_norm < tol { break; }

        let h = hessian(&x);

        // Solve trust region subproblem (Cauchy point)
        let direction = cauchy_point(&g, &h, radius);

        // Compute actual vs predicted reduction
        let x_new: Vec<f64> = x.iter().zip(direction.iter()).map(|(xi, di)| xi + di).collect();
        let actual_reduction = f(&x) - f(&x_new);

        // Predicted reduction: -g'*d - 0.5*d'*H*d
        let gd: f64 = g.iter().zip(direction.iter()).map(|(a, b)| a * b).sum();
        let hd: Vec<f64> = (0..n).map(|i| {
            (0..n).map(|j| h[i][j] * direction[j]).sum::<f64>()
        }).collect();
        let dhd: f64 = direction.iter().zip(hd.iter()).map(|(a, b)| a * b).sum();
        let predicted_reduction = -gd - 0.5 * dhd;

        let rho = if predicted_reduction.abs() > 1e-15 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };

        // Update radius
        if rho < 0.25 {
            radius *= 0.25;
        } else if rho > 0.75 && (direction.iter().map(|d| d * d).sum::<f64>().sqrt() - radius).abs() < 1e-10 {
            radius = (2.0 * radius).min(max_radius);
        }

        // Accept or reject step
        if rho > 0.1 {
            x = x_new;
        }
    }

    x
}

fn cauchy_point(g: &[f64], h: &[Vec<f64>], radius: f64) -> Vec<f64> {
    let n = g.len();
    let g_hg: f64 = (0..n).map(|i| g[i] * (0..n).map(|j| h[i][j] * g[j]).sum::<f64>()).sum();
    let g_norm = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();

    if g_hg <= 0.0 {
        // Steepest descent direction scaled to trust region
        let scale = radius / g_norm;
        return g.iter().map(|gi| -scale * gi).collect();
    }

    let scale = (g_norm.powi(3) / (radius * g_hg)).min(1.0);
    let tau = scale / g_norm;

    g.iter().map(|gi| -tau * gi).collect()
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = 1.0; }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfgs() {
        // Rosenbrock function
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let grad = |x: &[f64]| vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
            200.0 * (x[1] - x[0] * x[0]),
        ];

        let result = bfgs(f, grad, &[-1.0, 1.0], 1e-6, 1000);
        assert!((result[0] - 1.0).abs() < 0.1);
        assert!((result[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_conjugate_gradient() {
        // Quadratic: f(x) = 0.5 * x'Ax - b'x
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0];

        let f = |x: &[f64]| {
            let ax: Vec<f64> = a.iter().map(|row| row.iter().zip(x).map(|(ai, xi)| ai * xi).sum()).collect();
            0.5 * x.iter().zip(ax.iter()).map(|(xi, ai)| xi * ai).sum::<f64>()
                - b.iter().zip(x.iter()).map(|(bi, xi)| bi * xi).sum::<f64>()
        };

        let grad = |x: &[f64]| {
            let ax: Vec<f64> = a.iter().map(|row| row.iter().zip(x).map(|(ai, xi)| ai * xi).sum()).collect();
            ax.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect()
        };

        let result = conjugate_gradient(f, grad, &[0.0, 0.0], 1e-6, 100);
        // Solution: A^-1 * b
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        let expected_x = (a[1][1] * b[0] - a[0][1] * b[1]) / det;
        let expected_y = (a[0][0] * b[1] - a[1][0] * b[0]) / det;
        assert!((result[0] - expected_x).abs() < 0.1);
        assert!((result[1] - expected_y).abs() < 0.1);
    }
}
