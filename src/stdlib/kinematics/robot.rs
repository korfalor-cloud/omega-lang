/// Robotics kinematics: forward/inverse kinematics, DH parameters, trajectory planning.

use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct DHParams {
    pub a: f64,     // link length
    pub alpha: f64, // link twist
    pub d: f64,     // link offset
    pub theta: f64, // joint angle
}

impl DHParams {
    pub fn new(a: f64, alpha: f64, d: f64, theta: f64) -> Self {
        Self { a, alpha, d, theta }
    }

    /// Compute the 4x4 transformation matrix for this DH parameter set.
    pub fn transformation_matrix(&self) -> [[f64; 4]; 4] {
        let (ct, st) = (self.theta.cos(), self.theta.sin());
        let (ca, sa) = (self.alpha.cos(), self.alpha.sin());

        [
            [ct, -st * ca,  st * sa, self.a * ct],
            [st,  ct * ca, -ct * sa, self.a * st],
            [0.0, sa,       ca,      self.d      ],
            [0.0, 0.0,      0.0,     1.0         ],
        ]
    }
}

/// Multiply two 4x4 matrices.
pub fn mat4_mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// Extract position and orientation (ZYX Euler) from transformation matrix.
pub fn decompose_transform(m: &[[f64; 4]; 4]) -> ([f64; 3], [f64; 3]) {
    let x = m[0][3];
    let y = m[1][3];
    let z = m[2][3];

    let pitch = (-m[2][0]).asin();
    let (roll, yaw) = if pitch.cos().abs() > 1e-10 {
        (m[2][1].atan2(m[2][2]), m[1][0].atan2(m[0][0]))
    } else {
        (0.0, m[0][1].atan2(m[1][1]))
    };

    ([x, y, z], [roll, pitch, yaw])
}

/// Serial robot arm with DH parameters.
pub struct SerialArm {
    pub joints: Vec<DHParams>,
    pub joint_limits: Vec<(f64, f64)>,
    pub base_transform: [[f64; 4]; 4],
}

impl SerialArm {
    pub fn new(dh_params: Vec<DHParams>, joint_limits: Vec<(f64, f64)>) -> Self {
        Self {
            joints: dh_params,
            joint_limits,
            base_transform: identity_mat4(),
        }
    }

    pub fn dof(&self) -> usize {
        self.joints.len()
    }

    /// Forward kinematics: compute end-effector pose from joint angles.
    pub fn forward_kinematics(&self, angles: &[f64]) -> ([f64; 3], [f64; 3]) {
        let mut transform = self.base_transform;
        for (i, joint) in self.joints.iter().enumerate() {
            let mut params = joint.clone();
            if i < angles.len() {
                params.theta += angles[i];
            }
            transform = mat4_mul(&transform, &params.transformation_matrix());
        }
        decompose_transform(&transform)
    }

    /// Compute the full chain of link transforms.
    pub fn link_transforms(&self, angles: &[f64]) -> Vec<[[f64; 4]; 4]> {
        let mut transforms = vec![self.base_transform];
        let mut current = self.base_transform;
        for (i, joint) in self.joints.iter().enumerate() {
            let mut params = joint.clone();
            if i < angles.len() {
                params.theta += angles[i];
            }
            current = mat4_mul(&current, &params.transformation_matrix());
            transforms.push(current);
        }
        transforms
    }

    /// Jacobian matrix (6 x n) for end-effector.
    pub fn jacobian(&self, angles: &[f64]) -> Vec<Vec<f64>> {
        let n = self.joints.len();
        let transforms = self.link_transforms(angles);
        let end_pos = decompose_transform(&transforms[n]).0;

        let mut jac = vec![vec![0.0; n]; 6];

        for i in 0..n {
            let z_i = [transforms[i][0][2], transforms[i][1][2], transforms[i][2][2]];
            let p_i = [transforms[i][0][3], transforms[i][1][3], transforms[i][2][3]];

            // Revolute joint
            let cross = cross_product(&z_i, &[
                end_pos[0] - p_i[0],
                end_pos[1] - p_i[1],
                end_pos[2] - p_i[2],
            ]);
            jac[0][i] = cross[0];
            jac[1][i] = cross[1];
            jac[2][i] = cross[2];
            jac[3][i] = z_i[0];
            jac[4][i] = z_i[1];
            jac[5][i] = z_i[2];
        }

        jac
    }

    /// Inverse kinematics using damped least squares (Levenberg-Marquardt).
    pub fn inverse_kinematics(
        &self,
        target_pos: &[f64; 3],
        target_rpy: &[f64; 3],
        initial_angles: &[f64],
        max_iter: usize,
        tolerance: f64,
        damping: f64,
    ) -> Option<Vec<f64>> {
        let n = self.joints.len();
        let mut angles = initial_angles.to_vec();

        for _ in 0..max_iter {
            let (pos, rpy) = self.forward_kinematics(&angles);

            let mut error = vec![0.0; 6];
            for i in 0..3 {
                error[i] = target_pos[i] - pos[i];
                error[i + 3] = angle_diff(target_rpy[i], rpy[i]);
            }

            let err_norm: f64 = error.iter().map(|e| e * e).sum::<f64>().sqrt();
            if err_norm < tolerance {
                return Some(angles);
            }

            let jac = self.jacobian(&angles);
            let jt = transpose(&jac);
            let jtj = mat_mul(&jac, &jt);

            // Damped least squares: dq = J^T * (J * J^T + lambda^2 * I)^-1 * error
            let mut damped = jtj;
            for i in 0..6 {
                damped[i][i] += damping * damping;
            }

            if let Some(inv) = mat_invert_6x6(&damped) {
                let jte = mat_vec_mul(&inv, &error);
                let dq = mat_vec_mul(&jt, &jte);

                for i in 0..n {
                    angles[i] += dq[i];
                    let (lo, hi) = self.joint_limits[i];
                    angles[i] = angles[i].clamp(lo, hi);
                }
            } else {
                return None;
            }
        }

        None // Did not converge
    }
}

fn cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let d = a - b;
    ((d + PI) % (2.0 * PI) - PI + 2.0 * PI) % (2.0 * PI) - PI
}

fn identity_mat4() -> [[f64; 4]; 4] {
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 { m[i][i] = 1.0; }
    m
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

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
}

/// Invert a 6x6 matrix using Gauss-Jordan elimination.
fn mat_invert_6x6(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = 6;
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = matrix[i][j];
        }
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
        if pivot.abs() < 1e-10 {
            return None;
        }

        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row == col { continue; }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Some(inv)
}

/// Trajectory planning: cubic polynomial interpolation.
pub struct CubicTrajectory {
    pub a0: f64,
    pub a1: f64,
    pub a2: f64,
    pub a3: f64,
    pub duration: f64,
}

impl CubicTrajectory {
    pub fn new(start: f64, end: f64, start_vel: f64, end_vel: f64, duration: f64) -> Self {
        let a0 = start;
        let a1 = start_vel;
        let a2 = (3.0 * (end - start) - 2.0 * start_vel * duration - end_vel * duration) / (duration * duration);
        let a3 = (2.0 * (start - end) + (start_vel + end_vel) * duration) / (duration * duration * duration);
        Self { a0, a1, a2, a3, duration }
    }

    pub fn position(&self, t: f64) -> f64 {
        let t = t.min(self.duration);
        self.a0 + self.a1 * t + self.a2 * t * t + self.a3 * t * t * t
    }

    pub fn velocity(&self, t: f64) -> f64 {
        let t = t.min(self.duration);
        self.a1 + 2.0 * self.a2 * t + 3.0 * self.a3 * t * t
    }

    pub fn acceleration(&self, t: f64) -> f64 {
        let t = t.min(self.duration);
        2.0 * self.a2 + 6.0 * self.a3 * t
    }
}

/// Trapezoidal velocity profile.
pub struct TrapezoidalProfile {
    pub max_vel: f64,
    pub max_acc: f64,
    pub distance: f64,
    pub t_accel: f64,
    pub t_cruise: f64,
    pub t_decel: f64,
}

impl TrapezoidalProfile {
    pub fn new(distance: f64, max_vel: f64, max_acc: f64) -> Self {
        let t_accel = max_vel / max_acc;
        let d_accel = 0.5 * max_acc * t_accel * t_accel;

        if 2.0 * d_accel > distance {
            // Triangle profile (no cruise phase)
            let t_accel = (distance / max_acc).sqrt();
            Self {
                max_vel: max_acc * t_accel,
                max_acc,
                distance,
                t_accel,
                t_cruise: 0.0,
                t_decel: t_accel,
            }
        } else {
            let d_cruise = distance - 2.0 * d_accel;
            let t_cruise = d_cruise / max_vel;
            Self {
                max_vel,
                max_acc,
                distance,
                t_accel,
                t_cruise,
                t_decel: t_accel,
            }
        }
    }

    pub fn total_time(&self) -> f64 {
        self.t_accel + self.t_cruise + self.t_decel
    }

    pub fn position(&self, t: f64) -> f64 {
        let t = t.min(self.total_time());
        if t < self.t_accel {
            0.5 * self.max_acc * t * t
        } else if t < self.t_accel + self.t_cruise {
            let d_accel = 0.5 * self.max_acc * self.t_accel * self.t_accel;
            d_accel + self.max_vel * (t - self.t_accel)
        } else {
            let d_accel = 0.5 * self.max_acc * self.t_accel * self.t_accel;
            let d_cruise = self.max_vel * self.t_cruise;
            let t_decel = t - self.t_accel - self.t_cruise;
            d_accel + d_cruise + self.max_vel * t_decel - 0.5 * self.max_acc * t_decel * t_decel
        }
    }

    pub fn velocity(&self, t: f64) -> f64 {
        let t = t.min(self.total_time());
        if t < self.t_accel {
            self.max_acc * t
        } else if t < self.t_accel + self.t_cruise {
            self.max_vel
        } else {
            let t_decel = t - self.t_accel - self.t_cruise;
            self.max_vel - self.max_acc * t_decel
        }
    }
}

/// Multi-joint trajectory with via points.
pub struct JointTrajectory {
    pub segments: Vec<(CubicTrajectory, CubicTrajectory)>, // (pos_traj, vel_traj per joint)
    pub joint_count: usize,
}

impl JointTrajectory {
    pub fn from_via_points(waypoints: &[Vec<f64>], durations: &[f64]) -> Self {
        let n = waypoints[0].len();
        let mut segments = Vec::new();

        for seg in 0..durations.len() {
            let start = &waypoints[seg];
            let end = &waypoints[seg + 1];
            let vel_start: Vec<f64> = if seg == 0 {
                vec![0.0; n]
            } else {
                // Approximate velocity from previous segment
                let prev_end = &waypoints[seg];
                let prev_start = &waypoints[seg - 1];
                prev_end.iter().zip(prev_start.iter()).map(|(e, s)| (e - s) / durations[seg - 1]).collect()
            };
            let vel_end: Vec<f64> = if seg == durations.len() - 1 {
                vec![0.0; n]
            } else {
                let next_end = &waypoints[seg + 2];
                next_end.iter().zip(end.iter()).map(|(ne, e)| (ne - e) / durations[seg + 1]).collect()
            };

            // For simplicity, use 2D segment storage (would be n-dimensional in production)
            let traj1 = CubicTrajectory::new(start[0], end[0], vel_start[0], vel_end[0], durations[seg]);
            let traj2 = if n > 1 {
                CubicTrajectory::new(start[1], end[1], vel_start[1], vel_end[1], durations[seg])
            } else {
                CubicTrajectory::new(0.0, 0.0, 0.0, 0.0, durations[seg])
            };
            segments.push((traj1, traj2));
        }

        Self { segments, joint_count: n }
    }
}

/// PID controller for joint control.
pub struct PIDController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub integral: f64,
    pub prev_error: f64,
    pub output_min: f64,
    pub output_max: f64,
}

impl PIDController {
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp, ki, kd,
            integral: 0.0,
            prev_error: 0.0,
            output_min: f64::NEG_INFINITY,
            output_max: f64::INFINITY,
        }
    }

    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.output_min = min;
        self.output_max = max;
        self
    }

    pub fn update(&mut self, error: f64, dt: f64) -> f64 {
        self.integral += error * dt;
        let derivative = if dt > 0.0 { (error - self.prev_error) / dt } else { 0.0 };
        self.prev_error = error;

        let output = self.kp * error + self.ki * self.integral + self.kd * derivative;
        output.clamp(self.output_min, self.output_max)
    }

    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_kinematics() {
        // 2-link planar arm
        let arm = SerialArm::new(
            vec![
                DHParams::new(1.0, 0.0, 0.0, 0.0),
                DHParams::new(1.0, 0.0, 0.0, 0.0),
            ],
            vec![(-PI, PI), (-PI, PI)],
        );

        // Fully extended
        let (pos, _) = arm.forward_kinematics(&[0.0, 0.0]);
        assert!((pos[0] - 2.0).abs() < 0.001);
        assert!(pos[1].abs() < 0.001);

        // 90 degree bend
        let (pos, _) = arm.forward_kinematics(&[0.0, PI / 2.0]);
        assert!((pos[0] - 1.0).abs() < 0.001);
        assert!((pos[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_inverse_kinematics() {
        let arm = SerialArm::new(
            vec![
                DHParams::new(1.0, 0.0, 0.0, 0.0),
                DHParams::new(1.0, 0.0, 0.0, 0.0),
            ],
            vec![(-PI, PI), (-PI, PI)],
        );

        let target = [1.5, 0.5, 0.0];
        let rpy = [0.0, 0.0, 0.0];
        let result = arm.inverse_kinematics(&target, &rpy, &[0.1, 0.1], 100, 0.001, 0.1);
        assert!(result.is_some());
        let angles = result.unwrap();
        let (pos, _) = arm.forward_kinematics(&angles);
        assert!((pos[0] - target[0]).abs() < 0.01);
        assert!((pos[1] - target[1]).abs() < 0.01);
    }

    #[test]
    fn test_trapezoidal_profile() {
        let profile = TrapezoidalProfile::new(10.0, 2.0, 1.0);
        assert!((profile.position(profile.total_time()) - 10.0).abs() < 0.01);
        assert!((profile.velocity(0.0)).abs() < 0.01);
    }

    #[test]
    fn test_pid() {
        let mut pid = PIDController::new(1.0, 0.1, 0.01);
        let mut output = 0.0;
        let mut error = 10.0;
        for _ in 0..100 {
            output = pid.update(error, 0.1);
            error -= output * 0.1;
        }
        assert!(error.abs() < 1.0);
    }
}
