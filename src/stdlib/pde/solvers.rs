/// PDE solvers: finite difference methods for heat, wave, and Laplace equations.

/// 1D Heat equation: du/dt = alpha * d²u/dx²
/// Using explicit finite difference (forward Euler in time, central difference in space).
pub fn heat_equation_1d(
    initial: &[f64],
    alpha: f64,
    dx: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<Vec<f64>> {
    let n = initial.len();
    let r = alpha * dt / (dx * dx);

    assert!(r <= 0.5, "Stability condition violated: r = {} > 0.5", r);

    let mut u = initial.to_vec();
    let mut history = vec![u.clone()];

    for _ in 0..n_steps {
        let mut u_new = vec![0.0; n];
        u_new[0] = u[0]; // Boundary condition
        u_new[n - 1] = u[n - 1]; // Boundary condition

        for i in 1..n - 1 {
            u_new[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
        }

        u = u_new;
        history.push(u.clone());
    }

    history
}

/// 1D Heat equation with Crank-Nicolson (implicit, unconditionally stable).
pub fn heat_equation_crank_nicolson(
    initial: &[f64],
    alpha: f64,
    dx: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<Vec<f64>> {
    let n = initial.len();
    let r = alpha * dt / (2.0 * dx * dx);

    let mut u = initial.to_vec();
    let mut history = vec![u.clone()];

    for _ in 0..n_steps {
        // Set up tridiagonal system: A * u_new = B * u
        let mut a_lower = vec![0.0; n];
        let mut a_diag = vec![1.0; n];
        let mut a_upper = vec![0.0; n];
        let mut rhs = vec![0.0; n];

        for i in 1..n - 1 {
            a_lower[i] = -r;
            a_diag[i] = 1.0 + 2.0 * r;
            a_upper[i] = -r;

            rhs[i] = (1.0 - 2.0 * r) * u[i] + r * (u[i + 1] + u[i - 1]);
        }

        // Boundary conditions
        a_diag[0] = 1.0;
        a_diag[n - 1] = 1.0;
        rhs[0] = u[0];
        rhs[n - 1] = u[n - 1];

        // Solve tridiagonal system (Thomas algorithm)
        u = thomas_algorithm(&a_lower, &a_diag, &a_upper, &rhs);
        history.push(u.clone());
    }

    history
}

/// Thomas algorithm for tridiagonal systems.
fn thomas_algorithm(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut x = vec![0.0; n];

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let m = a[i] / (b[i] - a[i] * c_prime[i - 1]);
        c_prime[i] = c[i] / (b[i] - a[i] * c_prime[i - 1]);
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / (b[i] - a[i] * c_prime[i - 1]);
    }

    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    x
}

/// 1D Wave equation: d²u/dt² = c² * d²u/dx²
pub fn wave_equation_1d(
    initial_u: &[f64],
    initial_v: &[f64],
    c: f64,
    dx: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<Vec<f64>> {
    let n = initial_u.len();
    let r = c * dt / dx;

    assert!(r <= 1.0, "Stability condition violated: r = {} > 1.0", r);

    let mut u_prev = initial_u.to_vec();
    let mut u_curr: Vec<f64> = initial_u.iter().zip(initial_v.iter())
        .map(|(u, v)| u + v * dt)
        .collect();

    let mut history = vec![u_prev.clone(), u_curr.clone()];

    for _ in 1..n_steps {
        let mut u_next = vec![0.0; n];
        u_next[0] = 0.0; // Boundary condition
        u_next[n - 1] = 0.0; // Boundary condition

        for i in 1..n - 1 {
            u_next[i] = 2.0 * u_curr[i] - u_prev[i]
                + r * r * (u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1]);
        }

        u_prev = u_curr;
        u_curr = u_next;
        history.push(u_curr.clone());
    }

    history
}

/// 2D Laplace equation: d²u/dx² + d²u/dy² = 0
/// Solved using Jacobi iteration.
pub fn laplace_2d(
    initial: &[Vec<f64>],
    boundary: &[(usize, usize, f64)],
    tol: f64,
    max_iter: usize,
) -> Vec<Vec<f64>> {
    let ny = initial.len();
    let nx = initial[0].len();

    let mut u = initial.to_vec();

    // Set boundary conditions
    for &(x, y, val) in boundary {
        if y < ny && x < nx {
            u[y][x] = val;
        }
    }

    for iter in 0..max_iter {
        let mut max_diff = 0.0;
        let mut u_new = u.clone();

        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                // Skip boundary points
                if boundary.iter().any(|&(bx, by, _)| bx == x && by == y) {
                    continue;
                }

                u_new[y][x] = 0.25 * (u[y + 1][x] + u[y - 1][x] + u[y][x + 1] + u[y][x - 1]);
                max_diff = max_diff.max((u_new[y][x] - u[y][x]).abs());
            }
        }

        u = u_new;

        if max_diff < tol {
            break;
        }
    }

    u
}

/// 2D Laplace equation using Gauss-Seidel (faster convergence).
pub fn laplace_2d_gauss_seidel(
    initial: &[Vec<f64>],
    boundary: &[(usize, usize, f64)],
    tol: f64,
    max_iter: usize,
) -> Vec<Vec<f64>> {
    let ny = initial.len();
    let nx = initial[0].len();

    let mut u = initial.to_vec();

    for &(x, y, val) in boundary {
        if y < ny && x < nx {
            u[y][x] = val;
        }
    }

    for _ in 0..max_iter {
        let mut max_diff = 0.0;

        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                if boundary.iter().any(|&(bx, by, _)| bx == x && by == y) {
                    continue;
                }

                let old = u[y][x];
                u[y][x] = 0.25 * (u[y + 1][x] + u[y - 1][x] + u[y][x + 1] + u[y][x - 1]);
                max_diff = max_diff.max((u[y][x] - old).abs());
            }
        }

        if max_diff < tol {
            break;
        }
    }

    u
}

/// 2D Heat equation: du/dt = alpha * (d²u/dx² + d²u/dy²)
pub fn heat_equation_2d(
    initial: &[Vec<f64>],
    alpha: f64,
    dx: f64,
    dy: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<Vec<Vec<f64>>> {
    let ny = initial.len();
    let nx = initial[0].len();

    let rx = alpha * dt / (dx * dx);
    let ry = alpha * dt / (dy * dy);

    assert!(rx + ry <= 0.5, "Stability condition violated");

    let mut u = initial.to_vec();
    let mut history = vec![u.clone()];

    for _ in 0..n_steps {
        let mut u_new = u.clone();

        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                u_new[y][x] = u[y][x]
                    + rx * (u[y][x + 1] - 2.0 * u[y][x] + u[y][x - 1])
                    + ry * (u[y + 1][x] - 2.0 * u[y][x] + u[y - 1][x]);
            }
        }

        u = u_new;
        history.push(u.clone());
    }

    history
}

/// 2D Wave equation: d²u/dt² = c² * (d²u/dx² + d²u/dy²)
pub fn wave_equation_2d(
    initial_u: &[Vec<f64>],
    c: f64,
    dx: f64,
    dy: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<Vec<Vec<f64>>> {
    let ny = initial_u.len();
    let nx = initial_u[0].len();

    let rx = c * dt / dx;
    let ry = c * dt / dy;

    assert!(rx * rx + ry * ry <= 1.0, "Stability condition violated");

    let mut u_prev = initial_u.to_vec();
    let mut u_curr = initial_u.to_vec(); // Assuming zero initial velocity

    let mut history = vec![u_prev.clone(), u_curr.clone()];

    for _ in 1..n_steps {
        let mut u_next = vec![vec![0.0; nx]; ny];

        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                u_next[y][x] = 2.0 * u_curr[y][x] - u_prev[y][x]
                    + rx * rx * (u_curr[y][x + 1] - 2.0 * u_curr[y][x] + u_curr[y][x - 1])
                    + ry * ry * (u_curr[y + 1][x] - 2.0 * u_curr[y][x] + u_curr[y - 1][x]);
            }
        }

        u_prev = u_curr;
        u_curr = u_next;
        history.push(u_curr.clone());
    }

    history
}

/// Advection equation: du/dt + v * du/dx = 0
/// Using upwind scheme.
pub fn advection_1d(
    initial: &[f64],
    v: f64,
    dx: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<Vec<f64>> {
    let n = initial.len();
    let c = v * dt / dx; // Courant number

    assert!(c.abs() <= 1.0, "Stability condition violated");

    let mut u = initial.to_vec();
    let mut history = vec![u.clone()];

    for _ in 0..n_steps {
        let mut u_new = vec![0.0; n];

        if v > 0.0 {
            // Upwind (backward difference)
            u_new[0] = u[0]; // Inflow boundary
            for i in 1..n {
                u_new[i] = u[i] - c * (u[i] - u[i - 1]);
            }
        } else {
            // Upwind (forward difference)
            u_new[n - 1] = u[n - 1]; // Outflow boundary
            for i in 0..n - 1 {
                u_new[i] = u[i] - c * (u[i + 1] - u[i]);
            }
        }

        u = u_new;
        history.push(u.clone());
    }

    history
}

/// Burgers' equation: du/dt + u * du/dx = nu * d²u/dx²
/// Using Lax-Friedrichs scheme.
pub fn burgers_equation(
    initial: &[f64],
    nu: f64,
    dx: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<Vec<f64>> {
    let n = initial.len();

    let mut u = initial.to_vec();
    let mut history = vec![u.clone()];

    for _ in 0..n_steps {
        let mut u_new = vec![0.0; n];
        u_new[0] = u[0];
        u_new[n - 1] = u[n - 1];

        for i in 1..n - 1 {
            // Lax-Friedrichs
            let avg = 0.5 * (u[i + 1] + u[i - 1]);
            let flux = 0.25 * (u[i + 1] * u[i + 1] - u[i - 1] * u[i - 1]);
            let viscous = nu * (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx);
            u_new[i] = avg - 0.5 * dt / dx * flux + dt * viscous;
        }

        u = u_new;
        history.push(u.clone());
    }

    history
}

/// 2D Poisson equation: d²u/dx² + d²u/dy² = f(x,y)
/// Using Jacobi iteration.
pub fn poisson_2d(
    initial: &[Vec<f64>],
    f: &dyn Fn(f64, f64) -> f64,
    dx: f64,
    dy: f64,
    boundary: &[(usize, usize, f64)],
    tol: f64,
    max_iter: usize,
) -> Vec<Vec<f64>> {
    let ny = initial.len();
    let nx = initial[0].len();

    let mut u = initial.to_vec();

    for &(x, y, val) in boundary {
        if y < ny && x < nx {
            u[y][x] = val;
        }
    }

    let dx2 = dx * dx;
    let dy2 = dy * dy;
    let denom = 2.0 * (dx2 + dy2);

    for _ in 0..max_iter {
        let mut max_diff = 0.0;
        let mut u_new = u.clone();

        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                if boundary.iter().any(|&(bx, by, _)| bx == x && by == y) {
                    continue;
                }

                let fx = f(x as f64 * dx, y as f64 * dy);
                u_new[y][x] = ((u[y][x + 1] + u[y][x - 1]) * dy2
                    + (u[y + 1][x] + u[y - 1][x]) * dx2
                    - dx2 * dy2 * fx) / denom;

                max_diff = max_diff.max((u_new[y][x] - u[y][x]).abs());
            }
        }

        u = u_new;

        if max_diff < tol {
            break;
        }
    }

    u
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_equation() {
        let initial: Vec<f64> = (0..100).map(|i| {
            if i > 30 && i < 70 { 1.0 } else { 0.0 }
        }).collect();

        let result = heat_equation_1d(&initial, 0.01, 0.01, 0.001, 100);
        assert_eq!(result.len(), 101);

        // Heat should spread out
        let final_state = result.last().unwrap();
        let max_initial = initial.iter().cloned().fold(0.0f64, f64::max);
        let max_final = final_state.iter().cloned().fold(0.0f64, f64::max);
        assert!(max_final <= max_initial);
    }

    #[test]
    fn test_wave_equation() {
        let n = 100;
        let initial_u: Vec<f64> = (0..n).map(|i| {
            let x = i as f64 / n as f64;
            (std::f64::consts::PI * x).sin()
        }).collect();

        let initial_v = vec![0.0; n];

        let result = wave_equation_1d(&initial_u, &initial_v, 1.0, 0.01, 0.005, 200);
        assert_eq!(result.len(), 201);
    }

    #[test]
    fn test_laplace() {
        let n = 20;
        let initial = vec![vec![0.0; n]; n];
        let boundary = vec![
            (0, 0, 0.0), (n - 1, 0, 0.0),
            (0, n - 1, 1.0), (n - 1, n - 1, 1.0),
        ];

        let result = laplace_2d(&initial, &boundary, 1e-6, 10000);
        assert_eq!(result.len(), n);
    }

    #[test]
    fn test_advection() {
        let n = 100;
        let initial: Vec<f64> = (0..n).map(|i| {
            if i > 30 && i < 70 { 1.0 } else { 0.0 }
        }).collect();

        let result = advection_1d(&initial, 1.0, 0.01, 0.005, 100);
        assert_eq!(result.len(), 101);
    }
}
