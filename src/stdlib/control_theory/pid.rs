/// Control theory: PID controller, state-space, transfer functions.

/// PID controller with anti-windup.
pub struct PIDController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub setpoint: f64,
    pub output_min: f64,
    pub output_max: f64,
    pub integral: f64,
    pub prev_error: f64,
    pub prev_derivative: f64,
    pub derivative_filter: f64, // 0.0 = no filter, 1.0 = full filter
    pub first_update: bool,
}

impl PIDController {
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp, ki, kd,
            setpoint: 0.0,
            output_min: f64::NEG_INFINITY,
            output_max: f64::INFINITY,
            integral: 0.0,
            prev_error: 0.0,
            prev_derivative: 0.0,
            derivative_filter: 0.1,
            first_update: true,
        }
    }

    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.output_min = min;
        self.output_max = max;
        self
    }

    pub fn with_setpoint(mut self, setpoint: f64) -> Self {
        self.setpoint = setpoint;
        self
    }

    pub fn update(&mut self, measurement: f64, dt: f64) -> f64 {
        let error = self.setpoint - measurement;

        // Proportional
        let p_term = self.kp * error;

        // Integral with anti-windup
        self.integral += error * dt;
        let i_term = self.ki * self.integral;

        // Derivative with filtering
        let raw_derivative = if self.first_update {
            self.first_update = false;
            0.0
        } else {
            (error - self.prev_error) / dt
        };

        let filtered_derivative = self.derivative_filter * self.prev_derivative
            + (1.0 - self.derivative_filter) * raw_derivative;
        let d_term = self.kd * filtered_derivative;

        // Compute output
        let output = p_term + i_term + d_term;

        // Clamp output
        let clamped = output.max(self.output_min).min(self.output_max);

        // Anti-windup: if output was clamped, undo integral
        if (output - clamped).abs() > 1e-10 {
            self.integral -= error * dt;
        }

        self.prev_error = error;
        self.prev_derivative = filtered_derivative;

        clamped
    }

    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.prev_derivative = 0.0;
        self.first_update = true;
    }
}

/// State-space model: dx/dt = Ax + Bu, y = Cx + Du.
pub struct StateSpace {
    pub a: Vec<Vec<f64>>, // n x n
    pub b: Vec<Vec<f64>>, // n x m
    pub c: Vec<Vec<f64>>, // p x n
    pub d: Vec<Vec<f64>>, // p x m
    pub state: Vec<f64>,
    pub n: usize, // state dimension
    pub m: usize, // input dimension
    pub p: usize, // output dimension
}

impl StateSpace {
    pub fn new(
        a: Vec<Vec<f64>>, b: Vec<Vec<f64>>,
        c: Vec<Vec<f64>>, d: Vec<Vec<f64>>,
    ) -> Self {
        let n = a.len();
        let m = b[0].len();
        let p = c.len();
        Self { a, b, c, d, state: vec![0.0; n], n, m, p }
    }

    /// Discrete-time simulation (Euler method).
    pub fn step(&mut self, input: &[f64], dt: f64) -> Vec<f64> {
        assert_eq!(input.len(), self.m);

        // dx = (Ax + Bu) * dt
        let mut dx = vec![0.0; self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                dx[i] += self.a[i][j] * self.state[j];
            }
            for j in 0..self.m {
                dx[i] += self.b[i][j] * input[j];
            }
            dx[i] *= dt;
        }

        // Update state
        for i in 0..self.n {
            self.state[i] += dx[i];
        }

        // Output: y = Cx + Du
        let mut output = vec![0.0; self.p];
        for i in 0..self.p {
            for j in 0..self.n {
                output[i] += self.c[i][j] * self.state[j];
            }
            for j in 0..self.m {
                output[i] += self.d[i][j] * input[j];
            }
        }

        output
    }

    /// Simulate over time.
    pub fn simulate(&mut self, inputs: &[Vec<f64>], dt: f64) -> Vec<Vec<f64>> {
        inputs.iter().map(|u| self.step(u, dt)).collect()
    }

    /// Check controllability.
    pub fn is_controllable(&self) -> bool {
        // Controllability matrix: [B, AB, A^2B, ...]
        let mut cm = vec![vec![0.0; self.n * self.m]; self.n];

        let mut ab_power = self.b.clone();
        for k in 0..self.n {
            for i in 0..self.n {
                for j in 0..self.m {
                    cm[i][k * self.m + j] = ab_power[i][j];
                }
            }
            // A * ab_power
            let mut new_ab = vec![vec![0.0; self.m]; self.n];
            for i in 0..self.n {
                for j in 0..self.m {
                    for l in 0..self.n {
                        new_ab[i][j] += self.a[i][l] * ab_power[l][j];
                    }
                }
            }
            ab_power = new_ab;
        }

        // Check rank
        matrix_rank(&cm) == self.n
    }

    /// Check observability.
    pub fn is_observable(&self) -> bool {
        // Observability matrix: [C; CA; CA^2; ...]
        let mut om = vec![vec![0.0; self.n]; self.p * self.n];

        let mut ca_power = self.c.clone();
        for k in 0..self.n {
            for i in 0..self.p {
                for j in 0..self.n {
                    om[k * self.p + i][j] = ca_power[i][j];
                }
            }
            // ca_power * A
            let mut new_ca = vec![vec![0.0; self.n]; self.p];
            for i in 0..self.p {
                for j in 0..self.n {
                    for l in 0..self.n {
                        new_ca[i][j] += ca_power[i][l] * self.a[l][j];
                    }
                }
            }
            ca_power = new_ca;
        }

        matrix_rank(&om) == self.n
    }
}

fn matrix_rank(m: &[Vec<f64>]) -> usize {
    if m.is_empty() || m[0].is_empty() { return 0; }
    let rows = m.len();
    let cols = m[0].len();

    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut rank = 0;

    for col in 0..cols {
        // Find pivot
        let mut pivot_row = None;
        for row in rank..rows {
            if a[row][col].abs() > 1e-10 {
                pivot_row = Some(row);
                break;
            }
        }

        if let Some(pr) = pivot_row {
            a.swap(rank, pr);
            let pivot = a[rank][col];
            for j in col..cols {
                a[rank][j] /= pivot;
            }
            for row in 0..rows {
                if row != rank && a[row][col].abs() > 1e-10 {
                    let factor = a[row][col];
                    for j in col..cols {
                        a[row][j] -= factor * a[rank][j];
                    }
                }
            }
            rank += 1;
        }
    }

    rank
}

/// Transfer function: H(s) = num(s) / den(s).
pub struct TransferFunction {
    pub numerator: Vec<f64>,   // Coefficients of s^n, s^(n-1), ..., s^0
    pub denominator: Vec<f64>,
}

impl TransferFunction {
    pub fn new(numerator: Vec<f64>, denominator: Vec<f64>) -> Self {
        Self { numerator, denominator }
    }

    /// Evaluate H(s) for a complex value s = (re, im).
    pub fn evaluate(&self, re: f64, im: f64) -> (f64, f64) {
        let num = poly_eval_complex(&self.numerator, re, im);
        let den = poly_eval_complex(&self.denominator, re, im);
        complex_div(num, den)
    }

    /// Evaluate on imaginary axis (frequency response).
    pub fn frequency_response(&self, omegas: &[f64]) -> Vec<(f64, f64, f64)> {
        omegas.iter().map(|&w| {
            let (re, im) = self.evaluate(0.0, w);
            let magnitude = (re * re + im * im).sqrt();
            let phase = im.atan2(re);
            (w, magnitude, phase)
        }).collect()
    }

    /// Convert to state-space (controllable canonical form).
    pub fn to_state_space(&self) -> StateSpace {
        let n = self.denominator.len() - 1;
        let a0 = self.denominator[0];

        // A matrix (companion form)
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n - 1 {
            a[i][i + 1] = 1.0;
        }
        for i in 0..n {
            a[n - 1][i] = -self.denominator[n - i] / a0;
        }

        // B matrix
        let mut b = vec![vec![0.0]; n];
        b[n - 1][0] = 1.0 / a0;

        // C matrix
        let mut c = vec![vec![0.0; n]];
        let num_offset = self.denominator.len() - self.numerator.len();
        for (i, &coeff) in self.numerator.iter().enumerate() {
            c[0][n - 1 - num_offset - i] = coeff / a0;
        }

        // D matrix
        let d = vec![vec![0.0]];

        StateSpace::new(a, b, c, d)
    }

    /// Parallel connection: H1 + H2.
    pub fn parallel(&self, other: &TransferFunction) -> TransferFunction {
        let num1 = poly_mul(&self.numerator, &other.denominator);
        let num2 = poly_mul(&other.numerator, &self.denominator);
        let den = poly_mul(&self.denominator, &other.denominator);
        let num = poly_add(&num1, &num2);
        TransferFunction::new(num, den)
    }

    /// Series connection: H1 * H2.
    pub fn series(&self, other: &TransferFunction) -> TransferFunction {
        let num = poly_mul(&self.numerator, &other.numerator);
        let den = poly_mul(&self.denominator, &other.denominator);
        TransferFunction::new(num, den)
    }

    /// Feedback connection: H1 / (1 + H1 * H2).
    pub fn feedback(&self, controller: &TransferFunction) -> TransferFunction {
        let open_num = poly_mul(&self.numerator, &controller.numerator);
        let open_den = poly_mul(&self.denominator, &controller.denominator);
        let feedback_den = poly_add(&open_den, &open_num);
        TransferFunction::new(self.numerator.clone(), feedback_den)
    }
}

fn poly_eval_complex(coeffs: &[f64], re: f64, im: f64) -> (f64, f64) {
    let mut result_re = 0.0;
    let mut result_im = 0.0;
    let mut power_re = 1.0;
    let mut power_im = 0.0;

    for &c in coeffs.iter().rev() {
        result_re += c * power_re;
        result_im += c * power_im;
        // power *= (re + im*i)
        let new_re = power_re * re - power_im * im;
        let new_im = power_re * im + power_im * re;
        power_re = new_re;
        power_im = new_im;
    }

    (result_re, result_im)
}

fn complex_div((a_re, a_im): (f64, f64), (b_re, b_im): (f64, f64)) -> (f64, f64) {
    let denom = b_re * b_re + b_im * b_im;
    if denom.abs() < 1e-30 { return (0.0, 0.0); }
    ((a_re * b_re + a_im * b_im) / denom, (a_im * b_re - a_re * b_im) / denom)
}

fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

fn poly_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len().max(b.len());
    let mut result = vec![0.0; len];
    for (i, &ai) in a.iter().enumerate() { result[len - a.len() + i] += ai; }
    for (i, &bi) in b.iter().enumerate() { result[len - b.len() + i] += bi; }
    result
}

/// LQR (Linear Quadratic Regulator) gain computation.
pub fn lqr(a: &[Vec<f64>], b: &[Vec<f64>], q: &[Vec<f64>], r: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();

    // Solve Riccati equation iteratively: PA + A'P - PBR^{-1}B'P + Q = 0
    let mut p = q.to_vec();

    let r_inv = mat_invert(r).unwrap_or_else(|| vec![vec![0.0; m]; m]);

    for _ in 0..1000 {
        // BR^{-1}B'P
        let brinv = mat_mul(b, &r_inv);
        let brinvbt = mat_mul(&brinv, &transpose(b));
        let brinvbtp = mat_mul(&brinvbt, &p);

        // PA + A'P - PBR^{-1}B'P + Q
        let pa = mat_mul(&p, a);
        let at = transpose(a);
        let atp = mat_mul(&at, &p);

        let mut delta = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                delta[i][j] = pa[i][j] + atp[i][j] - brinvbtp[i][j] + q[i][j];
            }
        }

        // Update P
        let max_delta = delta.iter().flat_map(|row| row.iter()).map(|x| x.abs()).fold(0.0f64, f64::max);
        for i in 0..n {
            for j in 0..n {
                p[i][j] += delta[i][j] * 0.01;
            }
        }

        if max_delta < 1e-8 { break; }
    }

    // K = R^{-1}B'P
    let r_inv = mat_invert(r).unwrap_or_else(|| vec![vec![0.0; m]; m]);
    let bt = transpose(b);
    let btp = mat_mul(&bt, &p);
    mat_mul(&r_inv, &btp)
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

fn mat_invert(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n { aug[i][j] = m[i][j]; }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid() {
        let mut pid = PIDController::new(1.0, 0.1, 0.01)
            .with_setpoint(10.0)
            .with_limits(-100.0, 100.0);

        let mut measurement = 0.0;
        for _ in 0..100 {
            let output = pid.update(measurement, 0.1);
            measurement += output * 0.1;
        }
        assert!((measurement - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_state_space() {
        let ss = StateSpace::new(
            vec![vec![0.0, 1.0], vec![-2.0, -3.0]],
            vec![vec![0.0], vec![1.0]],
            vec![vec![1.0, 0.0]],
            vec![vec![0.0]],
        );
        assert!(ss.is_controllable());
        assert!(ss.is_observable());
    }

    #[test]
    fn test_transfer_function() {
        // H(s) = 1 / (s + 1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]);
        let (re, im) = tf.evaluate(0.0, 0.0); // H(0) = 1
        assert!((re - 1.0).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }
}
