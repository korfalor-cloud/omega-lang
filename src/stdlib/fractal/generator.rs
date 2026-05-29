/// Fractal generation: Mandelbrot, Julia, Sierpinski, L-systems, IFS.

use std::collections::HashMap;

/// Mandelbrot set computation.
pub struct Mandelbrot {
    pub max_iter: usize,
    pub width: usize,
    pub height: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

impl Mandelbrot {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            max_iter: 1000,
            width, height,
            x_min: -2.5, x_max: 1.0,
            y_min: -1.25, y_max: 1.25,
        }
    }

    pub fn compute(&self) -> Vec<Vec<usize>> {
        let mut grid = vec![vec![0; self.width]; self.height];

        for py in 0..self.height {
            for px in 0..self.width {
                let x0 = self.x_min + (px as f64 / self.width as f64) * (self.x_max - self.x_min);
                let y0 = self.y_min + (py as f64 / self.height as f64) * (self.y_max - self.y_min);

                let mut x = 0.0;
                let mut y = 0.0;
                let mut iter = 0;

                while x * x + y * y <= 4.0 && iter < self.max_iter {
                    let xtemp = x * x - y * y + x0;
                    y = 2.0 * x * y + y0;
                    x = xtemp;
                    iter += 1;
                }

                grid[py][px] = iter;
            }
        }

        grid
    }

    /// Compute with smooth coloring.
    pub fn compute_smooth(&self) -> Vec<Vec<f64>> {
        let mut grid = vec![vec![0.0; self.width]; self.height];

        for py in 0..self.height {
            for px in 0..self.width {
                let x0 = self.x_min + (px as f64 / self.width as f64) * (self.x_max - self.x_min);
                let y0 = self.y_min + (py as f64 / self.height as f64) * (self.y_max - self.y_min);

                let mut x = 0.0;
                let mut y = 0.0;
                let mut iter = 0;

                while x * x + y * y <= 4.0 && iter < self.max_iter {
                    let xtemp = x * x - y * y + x0;
                    y = 2.0 * x * y + y0;
                    x = xtemp;
                    iter += 1;
                }

                if iter < self.max_iter {
                    let log_zn = (x * x + y * y).ln();
                    let nu = (log_zn / 2.0_f64.ln()).ln() / 2.0_f64.ln();
                    grid[py][px] = iter as f64 + 1.0 - nu;
                } else {
                    grid[py][px] = 0.0;
                }
            }
        }

        grid
    }

    /// Render as ASCII art.
    pub fn render_ascii(&self) -> String {
        let grid = self.compute();
        let chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
        let max_val = *grid.iter().flat_map(|row| row.iter()).max().unwrap_or(&1);

        let mut result = String::new();
        for row in &grid {
            for &val in row {
                let idx = if val >= max_val {
                    chars.len() - 1
                } else {
                    (val * (chars.len() - 1)) / max_val.max(1)
                };
                result.push(chars[idx]);
            }
            result.push('\n');
        }
        result
    }
}

/// Julia set computation.
pub struct Julia {
    pub max_iter: usize,
    pub width: usize,
    pub height: usize,
    pub c_real: f64,
    pub c_imag: f64,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

impl Julia {
    pub fn new(width: usize, height: usize, c_real: f64, c_imag: f64) -> Self {
        Self {
            max_iter: 1000,
            width, height,
            c_real, c_imag,
            x_min: -2.0, x_max: 2.0,
            y_min: -1.5, y_max: 1.5,
        }
    }

    pub fn compute(&self) -> Vec<Vec<usize>> {
        let mut grid = vec![vec![0; self.width]; self.height];

        for py in 0..self.height {
            for px in 0..self.width {
                let mut x = self.x_min + (px as f64 / self.width as f64) * (self.x_max - self.x_min);
                let mut y = self.y_min + (py as f64 / self.height as f64) * (self.y_max - self.y_min);
                let mut iter = 0;

                while x * x + y * y <= 4.0 && iter < self.max_iter {
                    let xtemp = x * x - y * y + self.c_real;
                    y = 2.0 * x * y + self.c_imag;
                    x = xtemp;
                    iter += 1;
                }

                grid[py][px] = iter;
            }
        }

        grid
    }
}

/// Burning Ship fractal.
pub struct BurningShip {
    pub max_iter: usize,
    pub width: usize,
    pub height: usize,
}

impl BurningShip {
    pub fn new(width: usize, height: usize) -> Self {
        Self { max_iter: 1000, width, height }
    }

    pub fn compute(&self) -> Vec<Vec<usize>> {
        let mut grid = vec![vec![0; self.width]; self.height];

        for py in 0..self.height {
            for px in 0..self.width {
                let x0 = -2.5 + (px as f64 / self.width as f64) * 3.5;
                let y0 = -2.0 + (py as f64 / self.height as f64) * 3.0;

                let mut x = 0.0;
                let mut y = 0.0;
                let mut iter = 0;

                while x * x + y * y <= 4.0 && iter < self.max_iter {
                    let xtemp = x * x - y * y + x0;
                    y = (2.0 * x * y).abs() + y0;
                    x = xtemp;
                    iter += 1;
                }

                grid[py][px] = iter;
            }
        }

        grid
    }
}

/// Sierpinski triangle.
pub struct Sierpinski {
    pub depth: usize,
}

impl Sierpinski {
    pub fn new(depth: usize) -> Self {
        Self { depth }
    }

    pub fn compute(&self) -> Vec<Vec<bool>> {
        let size = 2usize.pow(self.depth as u32);
        let mut grid = vec![vec![true; size]; size];

        self.sierpinski_recurse(&mut grid, 0, 0, size);

        grid
    }

    fn sierpinski_recurse(&self, grid: &mut Vec<Vec<bool>>, x: usize, y: usize, size: usize) {
        if size <= 1 { return; }

        let half = size / 2;

        // Remove the middle triangle
        for i in 0..half {
            for j in (half - i)..(half + i + 1) {
                if y + i < grid.len() && x + j < grid[0].len() {
                    grid[y + i][x + j] = false;
                }
            }
        }

        // Recurse into three corners
        self.sierpinski_recurse(grid, x, y, half);
        self.sierpinski_recurse(grid, x + half, y, half);
        self.sierpinski_recurse(grid, x + half / 2, y + half, half);
    }

    pub fn render_ascii(&self) -> String {
        let grid = self.compute();
        let mut result = String::new();
        for row in &grid {
            for &val in row {
                result.push(if val { '*' } else { ' ' });
            }
            result.push('\n');
        }
        result
    }
}

/// L-system generator.
pub struct LSystem {
    pub axiom: String,
    pub rules: HashMap<char, String>,
    pub iterations: usize,
}

impl LSystem {
    pub fn new(axiom: &str, iterations: usize) -> Self {
        Self {
            axiom: axiom.to_string(),
            rules: HashMap::new(),
            iterations,
        }
    }

    pub fn add_rule(&mut self, from: char, to: &str) {
        self.rules.insert(from, to.to_string());
    }

    pub fn generate(&self) -> String {
        let mut current = self.axiom.clone();

        for _ in 0..self.iterations {
            let mut next = String::new();
            for ch in current.chars() {
                if let Some(replacement) = self.rules.get(&ch) {
                    next.push_str(replacement);
                } else {
                    next.push(ch);
                }
            }
            current = next;
        }

        current
    }

    /// Interpret L-system as turtle graphics and return points.
    pub fn interpret(&self, step: f64, angle: f64) -> Vec<(f64, f64)> {
        let instruction = self.generate();
        let mut points = Vec::new();
        let mut x = 0.0;
        let mut y = 0.0;
        let mut dir = 90.0_f64.to_radians(); // Pointing up
        let mut stack: Vec<(f64, f64, f64)> = Vec::new();

        points.push((x, y));

        for ch in instruction.chars() {
            match ch {
                'F' | 'A' | 'B' => {
                    x += step * dir.cos();
                    y += step * dir.sin();
                    points.push((x, y));
                }
                'f' => {
                    x += step * dir.cos();
                    y += step * dir.sin();
                }
                '+' => {
                    dir += angle;
                }
                '-' => {
                    dir -= angle;
                }
                '[' => {
                    stack.push((x, y, dir));
                }
                ']' => {
                    if let Some((px, py, pd)) = stack.pop() {
                        x = px;
                        y = py;
                        dir = pd;
                        points.push((x, y));
                    }
                }
                _ => {}
            }
        }

        points
    }
}

/// Preset L-systems.
pub fn koch_curve(iterations: usize) -> LSystem {
    let mut ls = LSystem::new("F", iterations);
    ls.add_rule('F', "F+F-F-F+F");
    ls
}

pub fn sierpinski_triangle_lsystem(iterations: usize) -> LSystem {
    let mut ls = LSystem::new("F-G-G", iterations);
    ls.add_rule('F', "F-G+F+G-F");
    ls.add_rule('G', "GG");
    ls
}

pub fn dragon_curve(iterations: usize) -> LSystem {
    let mut ls = LSystem::new("FX", iterations);
    ls.add_rule('X', "X+YF+");
    ls.add_rule('Y', "-FX-Y");
    ls
}

pub fn plant(iterations: usize) -> LSystem {
    let mut ls = LSystem::new("X", iterations);
    ls.add_rule('X', "F+[[X]-X]-F[-FX]+X");
    ls.add_rule('F', "FF");
    ls
}

/// Iterated Function System for Sierpinski triangle.
pub struct IFS {
    pub transforms: Vec<(f64, f64, f64, f64, f64, f64)>, // (a, b, c, d, e, f) for ax+by+e, cx+dy+f
    pub probabilities: Vec<f64>,
    seed: u64,
}

impl IFS {
    pub fn sierpinski() -> Self {
        Self {
            transforms: vec![
                (0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
                (0.5, 0.0, 0.0, 0.5, 0.5, 0.0),
                (0.5, 0.0, 0.0, 0.5, 0.25, 0.5),
            ],
            probabilities: vec![1.0 / 3.0; 3],
            seed: 42,
        }
    }

    pub fn barnsley_fern() -> Self {
        Self {
            transforms: vec![
                (0.0, 0.0, 0.0, 0.16, 0.0, 0.0),
                (0.85, 0.04, -0.04, 0.85, 0.0, 1.6),
                (0.2, -0.26, 0.23, 0.22, 0.0, 1.6),
                (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44),
            ],
            probabilities: vec![0.01, 0.85, 0.07, 0.07],
            seed: 42,
        }
    }

    pub fn generate(&mut self, n: usize) -> Vec<(f64, f64)> {
        let mut points = Vec::with_capacity(n);
        let mut x = 0.0;
        let mut y = 0.0;

        for _ in 0..n {
            let r = self.pseudo_rand();
            let mut cum = 0.0;
            let mut idx = 0;
            for (i, &p) in self.probabilities.iter().enumerate() {
                cum += p;
                if r < cum {
                    idx = i;
                    break;
                }
            }

            let (a, b, c, d, e, f) = self.transforms[idx];
            let new_x = a * x + b * y + e;
            let new_y = c * x + d * y + f;
            x = new_x;
            y = new_y;
            points.push((x, y));
        }

        points
    }

    fn pseudo_rand(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.seed >> 33) as f64) / (1u64 << 31) as f64
    }
}

/// Newton's fractal for polynomial roots.
pub struct NewtonFractal {
    pub coefficients: Vec<(f64, f64)>, // Complex coefficients
    pub roots: Vec<(f64, f64)>,
    pub max_iter: usize,
    pub tolerance: f64,
}

impl NewtonFractal {
    /// Create for z^3 - 1 = 0.
    pub fn cubic_roots() -> Self {
        Self {
            coefficients: vec![(-1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
            roots: vec![
                (1.0, 0.0),
                (-0.5, 0.866025),
                (-0.5, -0.866025),
            ],
            max_iter: 100,
            tolerance: 1e-6,
        }
    }

    pub fn compute(&self, width: usize, height: usize, x_range: (f64, f64), y_range: (f64, f64)) -> Vec<Vec<usize>> {
        let mut grid = vec![vec![0; width]; height];

        for py in 0..height {
            for px in 0..width {
                let x = x_range.0 + (px as f64 / width as f64) * (x_range.1 - x_range.0);
                let y = y_range.0 + (py as f64 / height as f64) * (y_range.1 - y_range.0);

                let mut zx = x;
                let mut zy = y;

                for iter in 0..self.max_iter {
                    // Evaluate polynomial and derivative
                    let (fx, fy) = self.eval_poly(zx, zy);
                    let (dfx, dfy) = self.eval_deriv(zx, zy);

                    // z = z - f(z)/f'(z)
                    let denom = dfx * dfx + dfy * dfy;
                    if denom < 1e-15 { break; }

                    let dx = (fx * dfx + fy * dfy) / denom;
                    let dy = (fy * dfx - fx * dfy) / denom;

                    zx -= dx;
                    zy -= dy;

                    // Check which root we converged to
                    for (i, &(rx, ry)) in self.roots.iter().enumerate() {
                        if (zx - rx).powi(2) + (zy - ry).powi(2) < self.tolerance * self.tolerance {
                            grid[py][px] = i + 1;
                            break;
                        }
                    }

                    if dx * dx + dy * dy < self.tolerance * self.tolerance {
                        break;
                    }
                }
            }
        }

        grid
    }

    fn eval_poly(&self, x: f64, y: f64) -> (f64, f64) {
        let mut rx = 0.0;
        let mut ry = 0.0;
        let mut px = 1.0;
        let mut py = 0.0;

        for &(cx, cy) in &self.coefficients {
            // result += c * power
            rx += cx * px - cy * py;
            ry += cx * py + cy * px;
            // power *= z
            let new_px = px * x - py * y;
            let new_py = px * y + py * x;
            px = new_px;
            py = new_py;
        }

        (rx, ry)
    }

    fn eval_deriv(&self, x: f64, y: f64) -> (f64, f64) {
        let mut rx = 0.0;
        let mut ry = 0.0;
        let mut px = 1.0;
        let mut py = 0.0;

        for (i, &(cx, cy)) in self.coefficients.iter().enumerate() {
            if i == 0 { continue; }
            let n = i as f64;
            rx += n * (cx * px - cy * py);
            ry += n * (cx * py + cy * px);
            let new_px = px * x - py * y;
            let new_py = px * y + py * x;
            px = new_px;
            py = new_py;
        }

        (rx, ry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mandelbrot() {
        let m = Mandelbrot::new(100, 100);
        let grid = m.compute();
        assert_eq!(grid.len(), 100);
        assert_eq!(grid[0].len(), 100);
    }

    #[test]
    fn test_julia() {
        let j = Julia::new(50, 50, -0.7, 0.27015);
        let grid = j.compute();
        assert_eq!(grid.len(), 50);
    }

    #[test]
    fn test_lsystem() {
        let mut ls = LSystem::new("F", 2);
        ls.add_rule('F', "F+F-F");
        assert_eq!(ls.generate(), "F+F-F+F+F-F-F+F-F");

        let koch = koch_curve(1);
        assert_eq!(koch.generate(), "F+F-F-F+F");
    }

    #[test]
    fn test_sierpinski() {
        let s = Sierpinski::new(3);
        let grid = s.compute();
        assert_eq!(grid.len(), 8);
        assert_eq!(grid[0].len(), 8);
    }

    #[test]
    fn test_ifs() {
        let mut ifs = IFS::sierpinski();
        let points = ifs.generate(1000);
        assert_eq!(points.len(), 1000);
        for &(x, y) in &points {
            assert!(x >= 0.0 && x <= 1.0);
            assert!(y >= 0.0 && y <= 1.0);
        }
    }
}
