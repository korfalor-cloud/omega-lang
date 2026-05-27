/// Cellular automata: Game of Life, 1D rules, and custom automata.

use std::collections::HashMap;

/// 2D Game of Life.
#[derive(Debug, Clone)]
pub struct GameOfLife {
    cells: Vec<Vec<bool>>,
    width: usize,
    height: usize,
    generation: u64,
}

impl GameOfLife {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            cells: vec![vec![false; width]; height],
            width,
            height,
            generation: 0,
        }
    }

    pub fn set_cell(&mut self, x: usize, y: usize, alive: bool) {
        if x < self.width && y < self.height {
            self.cells[y][x] = alive;
        }
    }

    pub fn get_cell(&self, x: usize, y: usize) -> bool {
        if x < self.width && y < self.height {
            self.cells[y][x]
        } else {
            false
        }
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn generation(&self) -> u64 { self.generation }

    pub fn alive_count(&self) -> usize {
        self.cells.iter().flat_map(|row| row.iter()).filter(|&&c| c).count()
    }

    fn count_neighbors(&self, x: usize, y: usize) -> usize {
        let mut count = 0;
        for dy in [-1i32, 0, 1].iter() {
            for dx in [-1i32, 0, 1].iter() {
                if *dx == 0 && *dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                    if self.cells[ny as usize][nx as usize] {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    /// Step forward one generation with standard B3/S23 rules.
    pub fn step(&mut self) {
        let mut next = vec![vec![false; self.width]; self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                let neighbors = self.count_neighbors(x, y);
                let alive = self.cells[y][x];
                next[y][x] = match (alive, neighbors) {
                    (true, 2) | (true, 3) => true,
                    (false, 3) => true,
                    _ => false,
                };
            }
        }

        self.cells = next;
        self.generation += 1;
    }

    /// Step with custom birth/survive rules.
    pub fn step_custom(&mut self, birth: &[usize], survive: &[usize]) {
        let mut next = vec![vec![false; self.width]; self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                let neighbors = self.count_neighbors(x, y);
                let alive = self.cells[y][x];
                next[y][x] = if alive {
                    survive.contains(&neighbors)
                } else {
                    birth.contains(&neighbors)
                };
            }
        }

        self.cells = next;
        self.generation += 1;
    }

    /// Clear all cells.
    pub fn clear(&mut self) {
        for row in &mut self.cells {
            row.fill(false);
        }
        self.generation = 0;
    }

    /// Randomize cells with given density (0.0 to 1.0).
    pub fn randomize(&mut self, density: f64) {
        let mut state: u64 = 42;
        for y in 0..self.height {
            for x in 0..self.width {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let rand = ((state >> 33) as f64) / (1u64 << 31) as f64;
                self.cells[y][x] = rand < density;
            }
        }
    }

    /// Glider pattern at position.
    pub fn add_glider(&mut self, x: usize, y: usize) {
        self.set_cell(x + 1, y, true);
        self.set_cell(x + 2, y + 1, true);
        self.set_cell(x, y + 2, true);
        self.set_cell(x + 1, y + 2, true);
        self.set_cell(x + 2, y + 2, true);
    }

    /// Blinker pattern at position.
    pub fn add_blinker(&mut self, x: usize, y: usize) {
        self.set_cell(x, y, true);
        self.set_cell(x + 1, y, true);
        self.set_cell(x + 2, y, true);
    }

    /// Pulsar pattern at position (period 3).
    pub fn add_pulsar(&mut self, x: usize, y: usize) {
        let pattern = [
            (2, 0), (3, 0), (4, 0), (8, 0), (9, 0), (10, 0),
            (0, 2), (5, 2), (7, 2), (12, 2),
            (0, 3), (5, 3), (7, 3), (12, 3),
            (0, 4), (5, 4), (7, 4), (12, 4),
            (2, 5), (3, 5), (4, 5), (8, 5), (9, 5), (10, 5),
            (2, 7), (3, 7), (4, 7), (8, 7), (9, 7), (10, 7),
            (0, 8), (5, 8), (7, 8), (12, 8),
            (0, 9), (5, 9), (7, 9), (12, 9),
            (0, 10), (5, 10), (7, 10), (12, 10),
            (2, 12), (3, 12), (4, 12), (8, 12), (9, 12), (10, 12),
        ];
        for &(dx, dy) in &pattern {
            self.set_cell(x + dx, y + dy, true);
        }
    }

    /// Gosper glider gun.
    pub fn add_gosper_gun(&mut self, x: usize, y: usize) {
        let pattern = [
            (24, 0),
            (22, 1), (24, 1),
            (12, 2), (13, 2), (20, 2), (21, 2), (34, 2), (35, 2),
            (11, 3), (15, 3), (20, 3), (21, 3), (34, 3), (35, 3),
            (0, 4), (1, 4), (10, 4), (16, 4), (20, 4), (21, 4),
            (0, 5), (1, 5), (10, 5), (14, 5), (16, 5), (17, 5), (22, 5), (24, 5),
            (10, 6), (16, 6), (24, 6),
            (11, 7), (15, 7),
            (12, 8), (13, 8),
        ];
        for &(dx, dy) in &pattern {
            self.set_cell(x + dx, y + dy, true);
        }
    }

    /// Render to text.
    pub fn to_string(&self) -> String {
        let mut s = String::new();
        for y in 0..self.height {
            for x in 0..self.width {
                s.push(if self.cells[y][x] { '#' } else { '.' });
            }
            s.push('\n');
        }
        s
    }
}

/// 1D elementary cellular automaton.
#[derive(Debug, Clone)]
pub struct CellularAutomaton1D {
    cells: Vec<bool>,
    rule: u8,
    generation: u64,
}

impl CellularAutomaton1D {
    pub fn new(size: usize, rule: u8) -> Self {
        let mut cells = vec![false; size];
        cells[size / 2] = true;
        Self { cells, rule, generation: 0 }
    }

    pub fn set_cell(&mut self, index: usize, alive: bool) {
        if index < self.cells.len() {
            self.cells[index] = alive;
        }
    }

    pub fn step(&mut self) {
        let n = self.cells.len();
        let mut next = vec![false; n];

        for i in 0..n {
            let left = if i > 0 { self.cells[i - 1] } else { false };
            let center = self.cells[i];
            let right = if i + 1 < n { self.cells[i + 1] } else { false };

            let pattern = (left as u8) << 2 | (center as u8) << 1 | (right as u8);
            next[i] = (self.rule >> pattern) & 1 == 1;
        }

        self.cells = next;
        self.generation += 1;
    }

    pub fn step_n(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    pub fn cells(&self) -> &[bool] {
        &self.cells
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn alive_count(&self) -> usize {
        self.cells.iter().filter(|&&c| c).count()
    }

    pub fn to_string(&self) -> String {
        self.cells.iter().map(|&c| if c { '#' } else { '.' }).collect()
    }

    /// Generate a visual history of the automaton.
    pub fn history(&mut self, steps: usize) -> Vec<String> {
        let mut result = Vec::new();
        result.push(self.to_string());
        for _ in 0..steps {
            self.step();
            result.push(self.to_string());
        }
        result
    }
}

/// Brian's Brain automaton (3-state).
#[derive(Debug, Clone)]
pub struct BriansBrain {
    cells: Vec<Vec<u8>>, // 0=dead, 1=dying, 2=alive
    width: usize,
    height: usize,
    generation: u64,
}

impl BriansBrain {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            cells: vec![vec![0; width]; height],
            width,
            height,
            generation: 0,
        }
    }

    pub fn set_cell(&mut self, x: usize, y: usize, state: u8) {
        if x < self.width && y < self.height {
            self.cells[y][x] = state.min(2);
        }
    }

    pub fn step(&mut self) {
        let mut next = vec![vec![0u8; self.width]; self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                match self.cells[y][x] {
                    2 => next[y][x] = 1, // alive -> dying
                    1 => next[y][x] = 0, // dying -> dead
                    0 => {
                        // dead -> alive if exactly 2 alive neighbors
                        let mut alive_neighbors = 0;
                        for dy in [-1i32, 0, 1].iter() {
                            for dx in [-1i32, 0, 1].iter() {
                                if *dx == 0 && *dy == 0 { continue; }
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                                    if self.cells[ny as usize][nx as usize] == 2 {
                                        alive_neighbors += 1;
                                    }
                                }
                            }
                        }
                        if alive_neighbors == 2 {
                            next[y][x] = 2;
                        }
                    }
                    _ => {}
                }
            }
        }

        self.cells = next;
        self.generation += 1;
    }

    pub fn generation(&self) -> u64 { self.generation }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_of_life_block() {
        let mut gol = GameOfLife::new(4, 4);
        gol.set_cell(1, 1, true);
        gol.set_cell(1, 2, true);
        gol.set_cell(2, 1, true);
        gol.set_cell(2, 2, true);

        gol.step();
        assert_eq!(gol.alive_count(), 4); // Block is stable
    }

    #[test]
    fn test_game_of_life_blinker() {
        let mut gol = GameOfLife::new(5, 5);
        gol.add_blinker(1, 2);

        assert_eq!(gol.alive_count(), 3);
        gol.step();
        // Blinker oscillates between horizontal and vertical
        assert_eq!(gol.alive_count(), 3);
    }

    #[test]
    fn test_rule_110() {
        let mut ca = CellularAutomaton1D::new(21, 110);
        ca.step_n(10);
        assert!(ca.generation() == 10);
    }

    #[test]
    fn test_rule_30() {
        let mut ca = CellularAutomaton1D::new(21, 30);
        let history = ca.history(10);
        assert_eq!(history.len(), 11);
    }

    #[test]
    fn test_brians_brain() {
        let mut bb = BriansBrain::new(10, 10);
        bb.set_cell(5, 5, 2);
        bb.step();
        // Cell should be dying now
        assert_eq!(bb.generation(), 1);
    }
}
