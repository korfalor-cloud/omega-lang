/// ODE solvers: Euler, RK4, RK45 (adaptive), Verlet, symplectic.

/// Euler method.
pub fn euler<F>(f: F, y0: &[f64], t0: f64, t_end: f64, dt: f64) -> Vec<(f64, Vec<f64>)>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let mut results = vec![(t0, y0.to_vec())];
    let mut t = t0;
    let mut y = y0.to_vec();

    while t < t_end {
        let actual_dt = dt.min(t_end - t);
        let dy = f(t, &y);
        for i in 0..y.len() {
            y[i] += dy[i] * actual_dt;
        }
        t += actual_dt;
        results.push((t, y.clone()));
    }

    results
}

/// 4th-order Runge-Kutta method.
pub fn rk4<F>(f: F, y0: &[f64], t0: f64, t_end: f64, dt: f64) -> Vec<(f64, Vec<f64>)>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let mut results = vec![(t0, y0.to_vec())];
    let mut t = t0;
    let mut y = y0.to_vec();

    while t < t_end {
        let actual_dt = dt.min(t_end - t);

        let k1 = f(t, &y);
        let y_k1: Vec<f64> = y.iter().zip(k1.iter()).map(|(y, k)| y + k * actual_dt / 2.0).collect();
        let k2 = f(t + actual_dt / 2.0, &y_k1);
        let y_k2: Vec<f64> = y.iter().zip(k2.iter()).map(|(y, k)| y + k * actual_dt / 2.0).collect();
        let k3 = f(t + actual_dt / 2.0, &y_k2);
        let y_k3: Vec<f64> = y.iter().zip(k3.iter()).map(|(y, k)| y + k * actual_dt).collect();
        let k4 = f(t + actual_dt, &y_k3);

        for i in 0..y.len() {
            y[i] += actual_dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        t += actual_dt;
        results.push((t, y.clone()));
    }

    results
}

/// RK45 (Dormand-Prince) adaptive step method.
pub fn rk45<F>(f: F, y0: &[f64], t0: f64, t_end: f64, tol: f64) -> Vec<(f64, Vec<f64>)>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let mut results = vec![(t0, y0.to_vec())];
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut dt = 0.01;

    // Dormand-Prince coefficients
    let a21 = 1.0 / 5.0;
    let a31 = 3.0 / 40.0;
    let a32 = 9.0 / 40.0;
    let a41 = 44.0 / 45.0;
    let a42 = -56.0 / 15.0;
    let a43 = 32.0 / 9.0;
    let a51 = 19372.0 / 6561.0;
    let a52 = -25360.0 / 2187.0;
    let a53 = 64448.0 / 6561.0;
    let a54 = -212.0 / 729.0;
    let a61 = 9017.0 / 3168.0;
    let a62 = -355.0 / 33.0;
    let a63 = 46732.0 / 5247.0;
    let a64 = 49.0 / 176.0;
    let a65 = -5103.0 / 18656.0;

    let c2 = 1.0 / 5.0;
    let c3 = 3.0 / 10.0;
    let c4 = 4.0 / 5.0;
    let c5 = 8.0 / 9.0;

    // 5th order weights
    let b1 = 35.0 / 384.0;
    let b3 = 500.0 / 1113.0;
    let b4 = 125.0 / 192.0;
    let b5 = -2187.0 / 6784.0;
    let b6 = 11.0 / 84.0;

    // 4th order weights (for error estimation)
    let e1 = 71.0 / 57600.0;
    let e3 = -71.0 / 16695.0;
    let e4 = 71.0 / 1920.0;
    let e5 = -17253.0 / 339200.0;
    let e6 = 22.0 / 525.0;
    let e7 = -1.0 / 40.0;

    while t < t_end {
        let n = y.len();

        let k1 = f(t, &y);
        let y2: Vec<f64> = (0..n).map(|i| y[i] + dt * a21 * k1[i]).collect();
        let k2 = f(t + c2 * dt, &y2);
        let y3: Vec<f64> = (0..n).map(|i| y[i] + dt * (a31 * k1[i] + a32 * k2[i])).collect();
        let k3 = f(t + c3 * dt, &y3);
        let y4: Vec<f64> = (0..n).map(|i| y[i] + dt * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i])).collect();
        let k4 = f(t + c4 * dt, &y4);
        let y5: Vec<f64> = (0..n).map(|i| y[i] + dt * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i])).collect();
        let k5 = f(t + c5 * dt, &y5);
        let y6: Vec<f64> = (0..n).map(|i| y[i] + dt * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i])).collect();
        let k6 = f(t + dt, &y6);

        // 5th order solution
        let y_new: Vec<f64> = (0..n).map(|i| {
            y[i] + dt * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i])
        }).collect();

        // Error estimate
        let k7 = f(t + dt, &y_new);
        let error: f64 = (0..n).map(|i| {
            let e = dt * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
            e * e
        }).sum::<f64>().sqrt();

        // Accept or reject step
        if error <= tol || dt < 1e-10 {
            t += dt;
            y = y_new;
            results.push((t, y.clone()));
        }

        // Adjust step size
        if error > 0.0 {
            let safety = 0.9;
            let factor = safety * (tol / error).powf(0.2);
            dt = dt * factor.max(0.2).min(5.0);
        }

        if t + dt > t_end {
            dt = t_end - t;
        }
    }

    results
}

/// Velocity Verlet (symplectic integrator for Hamiltonian systems).
pub fn verlet<F>(force: F, x0: &[f64], v0: &[f64], t0: f64, t_end: f64, dt: f64) -> Vec<(f64, Vec<f64>, Vec<f64>)>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut results = vec![(t0, x0.to_vec(), v0.to_vec())];
    let mut t = t0;
    let mut x = x0.to_vec();
    let mut v = v0.to_vec();
    let mut a = force(&x);

    while t < t_end {
        let actual_dt = dt.min(t_end - t);

        // Update position
        let x_new: Vec<f64> = x.iter().zip(v.iter()).zip(a.iter())
            .map(|((xi, vi), ai)| xi + vi * actual_dt + 0.5 * ai * actual_dt * actual_dt)
            .collect();

        // Update acceleration
        let a_new = force(&x_new);

        // Update velocity
        let v_new: Vec<f64> = v.iter().zip(a.iter()).zip(a_new.iter())
            .map(|((vi, ai), ai_new)| vi + 0.5 * (ai + ai_new) * actual_dt)
            .collect();

        x = x_new;
        v = v_new;
        a = a_new;
        t += actual_dt;
        results.push((t, x.clone(), v.clone()));
    }

    results
}

/// Störmer-Verlet (for second-order ODEs: x'' = f(x, x')).
pub fn stormer_verlet<F>(f: F, x0: f64, v0: f64, t0: f64, t_end: f64, dt: f64) -> Vec<(f64, f64, f64)>
where
    F: Fn(f64, f64) -> f64,
{
    let mut results = vec![(t0, x0, v0)];
    let mut t = t0;
    let mut x = x0;
    let mut v = v0;

    while t < t_end {
        let actual_dt = dt.min(t_end - t);
        let half_dt = actual_dt / 2.0;

        // Half-step velocity
        let a = f(x, v);
        let v_half = v + half_dt * a;

        // Full-step position
        let x_new = x + actual_dt * v_half;

        // Half-step velocity
        let a_new = f(x_new, v_half);
        let v_new = v_half + half_dt * a_new;

        x = x_new;
        v = v_new;
        t += actual_dt;
        results.push((t, x, v));
    }

    results
}

/// Solve boundary value problem using shooting method.
pub fn bvp_shoot<F>(
    f: F,
    x0: f64, x1: f64,
    y_start: f64, y_end: f64,
    v_guess: f64,
    tol: f64,
    max_iter: usize,
) -> Vec<(f64, f64)>
where
    F: Fn(f64, f64, f64) -> f64, // f(x, y, y') = y''
{
    let mut v = v_guess;

    for _ in 0..max_iter {
        // Shoot with current guess
        let result = shoot_ode(&f, x0, y_start, v, x1, 0.001);
        let y_final = result.last().unwrap().1;

        if (y_final - y_end).abs() < tol {
            return result;
        }

        // Adjust using secant method
        let v2 = v + 0.001;
        let result2 = shoot_ode(&f, x0, y_start, v2, x1, 0.001);
        let y_final2 = result2.last().unwrap().1;

        let slope = (y_final2 - y_final) / 0.001;
        if slope.abs() < 1e-15 { break; }
        v -= (y_final - y_end) / slope;
    }

    shoot_ode(&f, x0, y_start, v, x1, 0.001)
}

fn shoot_ode<F>(f: F, x0: f64, y0: f64, v0: f64, x_end: f64, dx: f64) -> Vec<(f64, f64)>
where
    F: Fn(f64, f64, f64) -> f64,
{
    let mut results = vec![(x0, y0)];
    let mut x = x0;
    let mut y = y0;
    let mut v = v0;

    while x < x_end {
        let actual_dx = dx.min(x_end - x);

        // RK4 for second-order ODE
        let k1y = v;
        let k1v = f(x, y, v);

        let k2y = v + 0.5 * actual_dx * k1v;
        let k2v = f(x + 0.5 * actual_dx, y + 0.5 * actual_dx * k1y, k2y);

        let k3y = v + 0.5 * actual_dx * k2v;
        let k3v = f(x + 0.5 * actual_dx, y + 0.5 * actual_dx * k2y, k3y);

        let k4y = v + actual_dx * k3v;
        let k4v = f(x + actual_dx, y + actual_dx * k3y, k4y);

        y += actual_dx / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        v += actual_dx / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        x += actual_dx;

        results.push((x, y));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rk4() {
        // dy/dt = -y, y(0) = 1
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let result = rk4(f, &[1.0], 0.0, 1.0, 0.01);
        let y_final = result.last().unwrap().1[0];
        assert!((y_final - (-1.0_f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn test_rk45() {
        // dy/dt = -y, y(0) = 1
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let result = rk45(f, &[1.0], 0.0, 1.0, 1e-8);
        let y_final = result.last().unwrap().1[0];
        assert!((y_final - (-1.0_f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn test_verlet() {
        // Simple harmonic oscillator: F = -x
        let force = |x: &[f64]| vec![-x[0]];
        let result = verlet(force, &[1.0], &[0.0], 0.0, 2.0 * std::f64::consts::PI, 0.01);
        let (t, x, _) = result.last().unwrap();
        // Should return to initial position after one period
        assert!((x[0] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_euler() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let result = euler(f, &[1.0], 0.0, 1.0, 0.001);
        let y_final = result.last().unwrap().1[0];
        assert!((y_final - (-1.0_f64).exp()).abs() < 0.01);
    }
}
