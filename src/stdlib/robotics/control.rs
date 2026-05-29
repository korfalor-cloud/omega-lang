/// Robotics control: path planning, SLAM, sensor fusion, motion control.

use std::collections::{HashMap, BinaryHeap, HashSet};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridPos {
    pub x: i32,
    pub y: i32,
}

impl GridPos {
    pub fn new(x: i32, y: i32) -> Self { Self { x, y } }
    pub fn manhattan(self, other: Self) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
    pub fn euclidean(self, other: Self) -> f64 {
        ((self.x - other.x).pow(2) + (self.y - other.y).pow(2)) as f64
    }
}

/// A* path planner on a 2D grid.
pub struct AStarPlanner {
    pub width: i32,
    pub height: i32,
    obstacles: HashSet<GridPos>,
    costs: HashMap<GridPos, f64>,
}

impl AStarPlanner {
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            width,
            height,
            obstacles: HashSet::new(),
            costs: HashMap::new(),
        }
    }

    pub fn set_obstacle(&mut self, pos: GridPos) {
        self.obstacles.insert(pos);
    }

    pub fn set_cost(&mut self, pos: GridPos, cost: f64) {
        self.costs.insert(pos, cost);
    }

    pub fn is_valid(&self, pos: GridPos) -> bool {
        pos.x >= 0 && pos.x < self.width && pos.y >= 0 && pos.y < self.height && !self.obstacles.contains(&pos)
    }

    pub fn neighbors(&self, pos: GridPos) -> Vec<GridPos> {
        let dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)];
        dirs.iter()
            .map(|(dx, dy)| GridPos::new(pos.x + dx, pos.y + dy))
            .filter(|p| self.is_valid(*p))
            .collect()
    }

    pub fn find_path(&self, start: GridPos, goal: GridPos) -> Option<Vec<GridPos>> {
        let mut open = BinaryHeap::new();
        let mut came_from: HashMap<GridPos, GridPos> = HashMap::new();
        let mut g_score: HashMap<GridPos, f64> = HashMap::new();
        let mut closed: HashSet<GridPos> = HashSet::new();

        g_score.insert(start, 0.0);
        open.push(Node {
            pos: start,
            f_score: start.euclidean(goal),
        });

        while let Some(current) = open.pop() {
            if current.pos == goal {
                return Some(self.reconstruct_path(&came_from, goal));
            }

            if closed.contains(&current.pos) {
                continue;
            }
            closed.insert(current.pos);

            for neighbor in self.neighbors(current.pos) {
                if closed.contains(&neighbor) {
                    continue;
                }

                let move_cost = if (neighbor.x - current.pos.x).abs() + (neighbor.y - current.pos.y).abs() == 2 {
                    std::f64::consts::SQRT_2
                } else {
                    1.0
                };

                let terrain_cost = self.costs.get(&neighbor).copied().unwrap_or(1.0);
                let tentative_g = g_score[&current.pos] + move_cost * terrain_cost;

                if tentative_g < *g_score.get(&neighbor).unwrap_or(&f64::INFINITY) {
                    came_from.insert(neighbor, current.pos);
                    g_score.insert(neighbor, tentative_g);
                    let f = tentative_g + neighbor.euclidean(goal);
                    open.push(Node { pos: neighbor, f_score: f });
                }
            }
        }

        None
    }

    fn reconstruct_path(&self, came_from: &HashMap<GridPos, GridPos>, goal: GridPos) -> Vec<GridPos> {
        let mut path = vec![goal];
        let mut current = goal;
        while let Some(&prev) = came_from.get(&current) {
            path.push(prev);
            current = prev;
        }
        path.reverse();
        path
    }
}

#[derive(Clone)]
struct Node {
    pos: GridPos,
    f_score: f64,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool { self.f_score == other.f_score }
}

impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f_score.partial_cmp(&self.f_score).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// RRT (Rapidly-exploring Random Tree) path planner.
pub struct RRTPlanner {
    pub width: f64,
    pub height: f64,
    pub step_size: f64,
    pub max_iter: usize,
    pub goal_bias: f64,
    obstacles: Vec<Circle>,
    seed: u64,
}

#[derive(Debug, Clone)]
pub struct Circle {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
}

impl RRTPlanner {
    pub fn new(width: f64, height: f64, step_size: f64) -> Self {
        Self {
            width, height, step_size,
            max_iter: 10000,
            goal_bias: 0.1,
            obstacles: Vec::new(),
            seed: 42,
        }
    }

    pub fn add_obstacle(&mut self, obs: Circle) {
        self.obstacles.push(obs);
    }

    pub fn is_collision_free(&self, x: f64, y: f64) -> bool {
        for obs in &self.obstacles {
            let dx = x - obs.x;
            let dy = y - obs.y;
            if dx * dx + dy * dy < obs.radius * obs.radius {
                return false;
            }
        }
        x >= 0.0 && x < self.width && y >= 0.0 && y < self.height
    }

    pub fn find_path(&mut self, start: (f64, f64), goal: (f64, f64)) -> Option<Vec<(f64, f64)>> {
        let mut nodes = vec![start];
        let mut parents: Vec<i32> = vec![-1];

        for _ in 0..self.max_iter {
            // Sample with goal bias
            let (rx, ry) = if self.pseudo_rand() < self.goal_bias {
                goal
            } else {
                (self.pseudo_rand() * self.width, self.pseudo_rand() * self.height)
            };

            // Find nearest node
            let nearest = nodes.iter().enumerate()
                .min_by_key(|(_, (nx, ny))| ((nx - rx).powi(2) + (ny - ry).powi(2)) as i64)
                .map(|(i, _)| i)
                .unwrap();

            let (nx, ny) = nodes[nearest];
            let dx = rx - nx;
            let dy = ry - ny;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < 1e-10 { continue; }

            let step = self.step_size.min(dist);
            let new_x = nx + dx / dist * step;
            let new_y = ny + dy / dist * step;

            if !self.is_collision_free(new_x, new_y) {
                continue;
            }

            // Check line segment collision
            if !self.line_collision_free(nx, ny, new_x, new_y) {
                continue;
            }

            nodes.push((new_x, new_y));
            parents.push(nearest as i32);

            // Check if goal reached
            let dg = ((new_x - goal.0).powi(2) + (new_y - goal.1).powi(2)).sqrt();
            if dg < self.step_size {
                nodes.push(goal);
                parents.push((nodes.len() - 2) as i32);
                return Some(self.build_path(&nodes, &parents));
            }
        }

        None
    }

    fn line_collision_free(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> bool {
        let steps = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt() / (self.step_size * 0.5) as f64;
        let steps = steps.max(1.0) as i32;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let x = x1 + (x2 - x1) * t;
            let y = y1 + (y2 - y1) * t;
            if !self.is_collision_free(x, y) {
                return false;
            }
        }
        true
    }

    fn build_path(&self, nodes: &[(f64, f64)], parents: &[i32]) -> Vec<(f64, f64)> {
        let mut path = Vec::new();
        let mut current = (nodes.len() - 1) as i32;
        while current >= 0 {
            path.push(nodes[current as usize]);
            current = parents[current as usize];
        }
        path.reverse();
        path
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Occupancy grid for SLAM.
pub struct OccupancyGrid {
    pub width: usize,
    pub height: usize,
    pub resolution: f64, // meters per cell
    cells: Vec<f64>,     // log-odds
    pub origin_x: f64,
    pub origin_y: f64,
}

impl OccupancyGrid {
    pub fn new(width: usize, height: usize, resolution: f64) -> Self {
        Self {
            width, height, resolution,
            cells: vec![0.0; width * height],
            origin_x: 0.0,
            origin_y: 0.0,
        }
    }

    pub fn cell_index(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let cx = ((x - self.origin_x) / self.resolution) as i32;
        let cy = ((y - self.origin_y) / self.resolution) as i32;
        if cx >= 0 && cx < self.width as i32 && cy >= 0 && cy < self.height as i32 {
            Some((cx as usize, cy as usize))
        } else {
            None
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        let log_odds = self.cells[y * self.width + x];
        1.0 / (1.0 + (-log_odds).exp())
    }

    pub fn set(&mut self, x: usize, y: usize, log_odds: f64) {
        self.cells[y * self.width + x] = log_odds;
    }

    pub fn update_ray(&mut self, sensor_x: f64, sensor_y: f64, hit_x: f64, hit_y: f64, occupied_prob: f64, free_prob: f64) {
        let occupied_lodds = (occupied_prob / (1.0 - occupied_prob)).ln();
        let free_lodds = (free_prob / (1.0 - free_prob)).ln();

        // Bresenham line from sensor to hit point
        let start = self.cell_index(sensor_x, sensor_y);
        let end = self.cell_index(hit_x, hit_y);

        if let (Some((sx, sy)), Some((ex, ey))) = (start, end) {
            let points = bresenham(sx as i32, sy as i32, ex as i32, ey as i32);
            for (i, &(px, py)) in points.iter().enumerate() {
                if px >= 0 && px < self.width as i32 && py >= 0 && py < self.height as i32 {
                    let idx = py as usize * self.width + px as usize;
                    if i == points.len() - 1 {
                        self.cells[idx] += occupied_lodds;
                    } else {
                        self.cells[idx] += free_lodds;
                    }
                }
            }
        }
    }
}

fn bresenham(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let mut points = Vec::new();
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = x0;
    let mut y = y0;

    loop {
        points.push((x, y));
        if x == x1 && y == y1 { break; }
        let e2 = 2 * err;
        if e2 > -dy { err -= dy; x += sx; }
        if e2 < dx { err += dx; y += sy; }
    }
    points
}

/// Kalman filter for sensor fusion.
pub struct KalmanFilter {
    pub state: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub process_noise: Vec<Vec<f64>>,
    pub measurement_noise: Vec<Vec<f64>>,
    pub state_transition: Vec<Vec<f64>>,
    pub observation: Vec<Vec<f64>>,
}

impl KalmanFilter {
    pub fn new(dim_state: usize, dim_obs: usize) -> Self {
        Self {
            state: vec![0.0; dim_state],
            covariance: Self::identity(dim_state),
            process_noise: Self::identity(dim_state),
            measurement_noise: Self::identity(dim_obs),
            state_transition: Self::identity(dim_state),
            observation: vec![vec![0.0; dim_state]; dim_obs],
        }
    }

    fn identity(n: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n { m[i][i] = 1.0; }
        m
    }

    pub fn predict(&mut self) {
        // x = F * x
        let new_state = Self::mat_vec_mul(&self.state_transition, &self.state);
        self.state = new_state;

        // P = F * P * F^T + Q
        let fp = Self::mat_mul(&self.state_transition, &self.covariance);
        let ft = Self::transpose(&self.state_transition);
        let fpft = Self::mat_mul(&fp, &ft);
        for i in 0..self.covariance.len() {
            for j in 0..self.covariance.len() {
                self.covariance[i][j] = fpft[i][j] + self.process_noise[i][j];
            }
        }
    }

    pub fn update(&mut self, measurement: &[f64]) {
        let n = self.state.len();
        let m = measurement.len();

        // y = z - H * x
        let hx = Self::mat_vec_mul(&self.observation, &self.state);
        let innovation: Vec<f64> = measurement.iter().zip(hx.iter()).map(|(z, h)| z - h).collect();

        // S = H * P * H^T + R
        let hp = Self::mat_mul(&self.observation, &self.covariance);
        let ht = Self::transpose(&self.observation);
        let hph = Self::mat_mul(&hp, &ht);
        let mut s = hph;
        for i in 0..m {
            for j in 0..m {
                s[i][j] += self.measurement_noise[i][j];
            }
        }

        // K = P * H^T * S^-1
        let ph = Self::mat_mul(&self.covariance, &ht);
        if let Some(s_inv) = Self::mat_invert(&s) {
            let k = Self::mat_mul(&ph, &s_inv);

            // x = x + K * y
            let ky = Self::mat_vec_mul(&k, &innovation);
            for i in 0..n {
                self.state[i] += ky[i];
            }

            // P = (I - K * H) * P
            let kh = Self::mat_mul(&k, &self.observation);
            let mut i_minus_kh = Self::identity(n);
            for i in 0..n {
                for j in 0..n {
                    i_minus_kh[i][j] -= kh[i][j];
                }
            }
            self.covariance = Self::mat_mul(&i_minus_kh, &self.covariance);
        }
    }

    fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        m.iter().map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
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

/// Particle filter for localization.
pub struct ParticleFilter {
    pub particles: Vec<Particle>,
    pub num_particles: usize,
    seed: u64,
}

#[derive(Debug, Clone)]
pub struct Particle {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub weight: f64,
}

impl ParticleFilter {
    pub fn new(num_particles: usize) -> Self {
        Self {
            particles: Vec::new(),
            num_particles,
            seed: 42,
        }
    }

    pub fn initialize(&mut self, x_range: (f64, f64), y_range: (f64, f64)) {
        self.particles.clear();
        for _ in 0..self.num_particles {
            self.particles.push(Particle {
                x: x_range.0 + self.pseudo_rand() * (x_range.1 - x_range.0),
                y: y_range.0 + self.pseudo_rand() * (y_range.1 - y_range.0),
                theta: self.pseudo_rand() * 2.0 * std::f64::consts::PI,
                weight: 1.0 / self.num_particles as f64,
            });
        }
    }

    pub fn predict(&mut self, dx: f64, dy: f64, dtheta: f64, noise_x: f64, noise_y: f64, noise_theta: f64) {
        for p in &mut self.particles {
            p.x += dx + self.gaussian() * noise_x;
            p.y += dy + self.gaussian() * noise_y;
            p.theta += dtheta + self.gaussian() * noise_theta;
        }
    }

    pub fn update_weights(&mut self, landmarks: &[(f64, f64)], measurements: &[(f64, f64)], sensor_noise: f64) {
        for p in &mut self.particles {
            let mut likelihood = 1.0;
            for &(mx, my) in measurements {
                let mut min_dist = f64::INFINITY;
                for &(lx, ly) in landmarks {
                    let dx = p.x - lx;
                    let dy = p.y - ly;
                    let predicted = (dx * dx + dy * dy).sqrt();
                    let measured = (mx * mx + my * my).sqrt();
                    let dist = (predicted - measured).abs();
                    min_dist = min_dist.min(dist);
                }
                let prob = (-0.5 * min_dist * min_dist / (sensor_noise * sensor_noise)).exp();
                likelihood *= prob;
            }
            p.weight = likelihood;
        }

        // Normalize
        let total: f64 = self.particles.iter().map(|p| p.weight).sum();
        if total > 0.0 {
            for p in &mut self.particles {
                p.weight /= total;
            }
        }
    }

    pub fn resample(&mut self) {
        let mut new_particles = Vec::with_capacity(self.num_particles);
        let weights: Vec<f64> = self.particles.iter().map(|p| p.weight).collect();
        let step = 1.0 / self.num_particles as f64;
        let mut r = self.pseudo_rand() * step;
        let mut cumulative = 0.0;
        let mut idx = 0;

        for _ in 0..self.num_particles {
            while cumulative < r && idx < weights.len() {
                cumulative += weights[idx];
                idx += 1;
            }
            if idx > 0 {
                new_particles.push(self.particles[idx - 1].clone());
            } else {
                new_particles.push(self.particles[0].clone());
            }
            r += step;
        }

        self.particles = new_particles;
    }

    pub fn estimate(&self) -> (f64, f64, f64) {
        let total_weight: f64 = self.particles.iter().map(|p| p.weight).sum();
        let x: f64 = self.particles.iter().map(|p| p.x * p.weight).sum::<f64>() / total_weight;
        let y: f64 = self.particles.iter().map(|p| p.y * p.weight).sum::<f64>() / total_weight;
        let theta: f64 = self.particles.iter().map(|p| p.theta * p.weight).sum::<f64>() / total_weight;
        (x, y, theta)
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.pseudo_rand().max(1e-10);
        let u2 = self.pseudo_rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Differential drive kinematics.
pub struct DifferentialDrive {
    pub wheelbase: f64,
    pub wheel_radius: f64,
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl DifferentialDrive {
    pub fn new(wheelbase: f64, wheel_radius: f64) -> Self {
        Self { wheelbase, wheel_radius, x: 0.0, y: 0.0, theta: 0.0 }
    }

    pub fn update(&mut self, left_omega: f64, right_omega: f64, dt: f64) {
        let v_left = left_omega * self.wheel_radius;
        let v_right = right_omega * self.wheel_radius;
        let v = (v_left + v_right) / 2.0;
        let omega = (v_right - v_left) / self.wheelbase;

        self.x += v * self.theta.cos() * dt;
        self.y += v * self.theta.sin() * dt;
        self.theta += omega * dt;
    }

    pub fn inverse_kinematics(&self, v: f64, omega: f64) -> (f64, f64) {
        let v_left = v - omega * self.wheelbase / 2.0;
        let v_right = v + omega * self.wheelbase / 2.0;
        (v_left / self.wheel_radius, v_right / self.wheel_radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_astar() {
        let mut planner = AStarPlanner::new(10, 10);
        planner.set_obstacle(GridPos::new(5, 5));
        planner.set_obstacle(GridPos::new(5, 4));
        planner.set_obstacle(GridPos::new(5, 3));

        let path = planner.find_path(GridPos::new(0, 0), GridPos::new(9, 9));
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(!path.is_empty());
        assert_eq!(path[0], GridPos::new(0, 0));
        assert_eq!(path.last(), Some(&GridPos::new(9, 9)));
    }

    #[test]
    fn test_kalman_filter() {
        let mut kf = KalmanFilter::new(2, 1);
        kf.state_transition = vec![vec![1.0, 1.0], vec![0.0, 1.0]];
        kf.observation = vec![vec![1.0, 0.0]];

        kf.predict();
        kf.update(&[1.0]);

        assert!(kf.state[0].abs() > 0.0);
    }

    #[test]
    fn test_particle_filter() {
        let mut pf = ParticleFilter::new(100);
        pf.initialize((0.0, 10.0), (0.0, 10.0));
        assert_eq!(pf.particles.len(), 100);

        pf.predict(1.0, 0.0, 0.0, 0.1, 0.1, 0.01);
        let (x, _, _) = pf.estimate();
        assert!(x > 0.0);
    }

    #[test]
    fn test_occupancy_grid() {
        let mut grid = OccupancyGrid::new(100, 100, 0.1);
        grid.update_ray(0.0, 0.0, 1.0, 0.0, 0.9, 0.4);
        // Should have free cells along the ray and occupied at the end
        assert!(grid.get(9, 0) > 0.5); // Occupied
    }

    #[test]
    fn test_differential_drive() {
        let mut dd = DifferentialDrive::new(0.5, 0.1);
        dd.update(10.0, 10.0, 1.0); // Both wheels same speed = straight
        assert!(dd.x > 0.0);
        assert!(dd.y.abs() < 0.01);
    }
}
