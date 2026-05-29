/// Game tree search: minimax, alpha-beta, MCTS.

use std::collections::HashMap;

/// Game state trait for search algorithms.
pub trait GameState: Clone {
    fn is_terminal(&self) -> bool;
    fn evaluate(&self) -> f64; // Positive = maximizing player wins
    fn generate_moves(&self) -> Vec<Self>;
    fn apply_move(&self, m: &Self) -> Self;
    fn current_player(&self) -> i32; // 1 or -1
}

/// Minimax algorithm with depth limit.
pub fn minimax<S: GameState>(state: &S, depth: usize, maximizing: bool) -> f64 {
    if depth == 0 || state.is_terminal() {
        return state.evaluate();
    }

    let moves = state.generate_moves();

    if maximizing {
        let mut value = f64::NEG_INFINITY;
        for m in &moves {
            let child = state.apply_move(m);
            value = value.max(minimax(&child, depth - 1, false));
        }
        value
    } else {
        let mut value = f64::INFINITY;
        for m in &moves {
            let child = state.apply_move(m);
            value = value.min(minimax(&child, depth - 1, true));
        }
        value
    }
}

/// Alpha-beta pruning.
pub fn alpha_beta<S: GameState>(state: &S, depth: usize, mut alpha: f64, mut beta: f64, maximizing: bool) -> f64 {
    if depth == 0 || state.is_terminal() {
        return state.evaluate();
    }

    let moves = state.generate_moves();

    if maximizing {
        let mut value = f64::NEG_INFINITY;
        for m in &moves {
            let child = state.apply_move(m);
            value = value.max(alpha_beta(&child, depth - 1, alpha, beta, false));
            alpha = alpha.max(value);
            if alpha >= beta { break; } // Beta cutoff
        }
        value
    } else {
        let mut value = f64::INFINITY;
        for m in &moves {
            let child = state.apply_move(m);
            value = value.min(alpha_beta(&child, depth - 1, alpha, beta, true));
            beta = beta.min(value);
            if alpha >= beta { break; } // Alpha cutoff
        }
        value
    }
}

/// Find best move using alpha-beta.
pub fn best_move<S: GameState>(state: &S, depth: usize) -> Option<S> {
    let moves = state.generate_moves();
    if moves.is_empty() { return None; }

    let maximizing = state.current_player() == 1;
    let mut best = None;
    let mut best_val = if maximizing { f64::NEG_INFINITY } else { f64::INFINITY };

    for m in &moves {
        let child = state.apply_move(m);
        let val = alpha_beta(&child, depth - 1, f64::NEG_INFINITY, f64::INFINITY, !maximizing);
        if (maximizing && val > best_val) || (!maximizing && val < best_val) {
            best_val = val;
            best = Some(child);
        }
    }

    best
}

/// Transposition table entry.
#[derive(Clone)]
struct TTEntry {
    depth: usize,
    value: f64,
    flag: TTFlag,
}

#[derive(Clone, PartialEq)]
enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

/// Alpha-beta with transposition table.
pub fn alpha_beta_tt<S: GameState + std::hash::Hash + Eq>(
    state: &S, depth: usize, mut alpha: f64, mut beta: f64, maximizing: bool,
    tt: &mut HashMap<S, TTEntry>
) -> f64 {
    if let Some(entry) = tt.get(state) {
        if entry.depth >= depth {
            match entry.flag {
                TTFlag::Exact => return entry.value,
                TTFlag::LowerBound => alpha = alpha.max(entry.value),
                TTFlag::UpperBound => beta = beta.min(entry.value),
            }
            if alpha >= beta { return entry.value; }
        }
    }

    if depth == 0 || state.is_terminal() {
        return state.evaluate();
    }

    let moves = state.generate_moves();
    let value;

    if maximizing {
        value = f64::NEG_INFINITY;
        let mut val = value;
        for m in &moves {
            let child = state.apply_move(m);
            val = val.max(alpha_beta_tt(&child, depth - 1, alpha, beta, false, tt));
            alpha = alpha.max(val);
            if alpha >= beta { break; }
        }
        value.max(val)
    } else {
        value = f64::INFINITY;
        let mut val = value;
        for m in &moves {
            let child = state.apply_move(m);
            val = val.min(alpha_beta_tt(&child, depth - 1, alpha, beta, true, tt));
            beta = beta.min(val);
            if alpha >= beta { break; }
        }
        value.min(val)
    };

    let flag = if value <= alpha {
        TTFlag::UpperBound
    } else if value >= beta {
        TTFlag::LowerBound
    } else {
        TTFlag::Exact
    };

    tt.insert(state.clone(), TTEntry { depth, value, flag });
    value
}

/// Iterative deepening.
pub fn iterative_deepening<S: GameState + std::hash::Hash + Eq>(state: &S, max_depth: usize) -> Option<S> {
    let mut best = None;
    let mut tt = HashMap::new();

    for depth in 1..=max_depth {
        let moves = state.generate_moves();
        if moves.is_empty() { return None; }

        let maximizing = state.current_player() == 1;
        let mut best_val = if maximizing { f64::NEG_INFINITY } else { f64::INFINITY };

        for m in &moves {
            let child = state.apply_move(m);
            let val = alpha_beta_tt(&child, depth - 1, f64::NEG_INFINITY, f64::INFINITY, !maximizing, &mut tt);
            if (maximizing && val > best_val) || (!maximizing && val < best_val) {
                best_val = val;
                best = Some(child);
            }
        }
    }

    best
}

/// Monte Carlo Tree Search.
pub struct MCTS {
    pub exploration: f64,
    pub simulations: usize,
    seed: u64,
}

impl MCTS {
    pub fn new(exploration: f64, simulations: usize) -> Self {
        Self { exploration, simulations, seed: 42 }
    }

    pub fn search<S: GameState>(&mut self, state: &S) -> Option<S> {
        let mut tree: HashMap<S, MCTSNode> = HashMap::new();
        tree.insert(state.clone(), MCTSNode::new());

        for _ in 0..self.simulations {
            let (path, leaf) = self.select(state, &tree);
            let value = self.simulate(&leaf);
            self.backpropagate(&path, value, &mut tree);
        }

        // Pick most visited child
        let root = tree.get(state)?;
        let moves = state.generate_moves();
        moves.into_iter()
            .max_by_key(|m| {
                let child = state.apply_move(m);
                tree.get(&child).map(|n| n.visits).unwrap_or(0)
            })
    }

    fn select<S: GameState>(&self, state: &S, tree: &HashMap<S, MCTSNode>) -> (Vec<S>, S) {
        let mut path = vec![state.clone()];
        let mut current = state.clone();

        loop {
            let node = tree.get(&current);
            if current.is_terminal() || node.map_or(true, |n| n.visits == 0) {
                return (path, current);
            }

            let moves = current.generate_moves();
            if moves.is_empty() {
                return (path, current);
            }

            let total_visits = node.unwrap().visits as f64;
            let mut best_child = None;
            let mut best_ucb = f64::NEG_INFINITY;

            for m in &moves {
                let child = current.apply_move(m);
                let child_node = tree.get(&child);

                let ucb = if let Some(cn) = child_node {
                    if cn.visits == 0 {
                        f64::INFINITY
                    } else {
                        let exploitation = cn.value / cn.visits as f64;
                        let exploration = self.exploration * (total_visits.ln() / cn.visits as f64).sqrt();
                        exploitation + exploration
                    }
                } else {
                    f64::INFINITY // Unvisited nodes are explored first
                };

                if ucb > best_ucb {
                    best_ucb = ucb;
                    best_child = Some(child);
                }
            }

            if let Some(next) = best_child {
                path.push(next.clone());
                current = next;
            } else {
                break;
            }
        }

        let leaf = path.last().cloned().unwrap_or(current);
        (path, leaf)
    }

    fn simulate<S: GameState>(&mut self, state: &S) -> f64 {
        let mut current = state.clone();
        let mut depth = 0;

        while !current.is_terminal() && depth < 100 {
            let moves = current.generate_moves();
            if moves.is_empty() { break; }

            let idx = self.pseudo_rand() as usize % moves.len();
            current = current.apply_move(&moves[idx]);
            depth += 1;
        }

        current.evaluate()
    }

    fn backpropagate<S: GameState + Eq + std::hash::Hash>(&self, path: &[S], value: f64, tree: &mut HashMap<S, MCTSNode>) {
        for state in path.iter().rev() {
            let node = tree.entry(state.clone()).or_insert_with(MCTSNode::new);
            node.visits += 1;
            node.value += value;
        }
    }

    fn pseudo_rand(&mut self) -> u64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.seed >> 33
    }
}

#[derive(Clone)]
struct MCTSNode {
    visits: usize,
    value: f64,
}

impl MCTSNode {
    fn new() -> Self {
        Self { visits: 0, value: 0.0 }
    }
}

// ─── Tic-Tac-Toe Example ───

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TicTacToe {
    pub board: [[i8; 3]; 3],
    pub player: i8, // 1 or -1
}

impl TicTacToe {
    pub fn new() -> Self {
        Self { board: [[0; 3]; 3], player: 1 }
    }
}

impl GameState for TicTacToe {
    fn is_terminal(&self) -> bool {
        self.winner() != 0 || self.board.iter().all(|row| row.iter().all(|&c| c != 0))
    }

    fn evaluate(&self) -> f64 {
        match self.winner() {
            1 => 100.0,
            -1 => -100.0,
            _ => 0.0,
        }
    }

    fn generate_moves(&self) -> Vec<Self> {
        let mut moves = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                if self.board[i][j] == 0 {
                    let mut new_board = self.board;
                    new_board[i][j] = self.player;
                    moves.push(Self { board: new_board, player: -self.player });
                }
            }
        }
        moves
    }

    fn apply_move(&self, m: &Self) -> Self {
        m.clone()
    }

    fn current_player(&self) -> i32 {
        self.player as i32
    }
}

impl TicTacToe {
    fn winner(&self) -> i8 {
        // Rows
        for row in &self.board {
            if row[0] != 0 && row[0] == row[1] && row[1] == row[2] {
                return row[0];
            }
        }
        // Columns
        for j in 0..3 {
            if self.board[0][j] != 0 && self.board[0][j] == self.board[1][j] && self.board[1][j] == self.board[2][j] {
                return self.board[0][j];
            }
        }
        // Diagonals
        if self.board[0][0] != 0 && self.board[0][0] == self.board[1][1] && self.board[1][1] == self.board[2][2] {
            return self.board[0][0];
        }
        if self.board[0][2] != 0 && self.board[0][2] == self.board[1][1] && self.board[1][1] == self.board[2][0] {
            return self.board[0][2];
        }
        0
    }
}

// ─── N-Queens ───

pub fn n_queens(n: usize) -> Vec<Vec<usize>> {
    let mut solutions = Vec::new();
    let mut board = vec![0usize; n];
    solve_n_queens(&mut board, 0, &mut solutions);
    solutions
}

fn solve_n_queens(board: &mut [usize], row: usize, solutions: &mut Vec<Vec<usize>>) {
    let n = board.len();
    if row == n {
        solutions.push(board.to_vec());
        return;
    }

    for col in 0..n {
        if is_safe(board, row, col) {
            board[row] = col;
            solve_n_queens(board, row + 1, solutions);
        }
    }
}

fn is_safe(board: &[usize], row: usize, col: usize) -> bool {
    for r in 0..row {
        if board[r] == col || (board[r] as i32 - col as i32).abs() == (r as i32 - row as i32).abs() {
            return false;
        }
    }
    true
}

/// Knight's tour using Warnsdorff's heuristic.
pub fn knights_tour(n: usize, start_x: usize, start_y: usize) -> Option<Vec<Vec<usize>>> {
    let mut board = vec![vec![0usize; n]; n];
    let moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)];

    board[start_x][start_y] = 1;
    let total = n * n;

    if warnsdorff(&mut board, start_x as i32, start_y as i32, 2, total, &moves, n) {
        Some(board)
    } else {
        None
    }
}

fn warnsdorff(board: &mut Vec<Vec<usize>>, x: i32, y: i32, step: usize, total: usize, moves: &[(i32, i32); 8], n: usize) -> bool {
    if step > total { return true; }

    let mut next_moves: Vec<(i32, i32, usize)> = Vec::new();

    for &(dx, dy) in moves {
        let nx = x + dx;
        let ny = y + dy;
        if nx >= 0 && nx < n as i32 && ny >= 0 && ny < n as i32 && board[nx as usize][ny as usize] == 0 {
            let degree = moves.iter().filter(|&&(ddx, ddy)| {
                let nnx = nx + ddx;
                let nny = ny + ddy;
                nnx >= 0 && nnx < n as i32 && nny >= 0 && nny < n as i32 && board[nnx as usize][nny as usize] == 0
            }).count();
            next_moves.push((nx, ny, degree));
        }
    }

    next_moves.sort_by_key(|&(_, _, d)| d);

    for (nx, ny, _) in next_moves {
        board[nx as usize][ny as usize] = step;
        if warnsdorff(board, nx, ny, step + 1, total, moves, n) {
            return true;
        }
        board[nx as usize][ny as usize] = 0;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tic_tac_toe_minimax() {
        let game = TicTacToe::new();
        let val = minimax(&game, 9, true);
        assert_eq!(val, 0.0); // Perfect play = draw
    }

    #[test]
    fn test_alpha_beta() {
        let game = TicTacToe::new();
        let val = alpha_beta(&game, 9, f64::NEG_INFINITY, f64::INFINITY, true);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_n_queens() {
        let solutions = n_queens(8);
        assert_eq!(solutions.len(), 92);
    }

    #[test]
    fn test_n_queens_4() {
        let solutions = n_queens(4);
        assert_eq!(solutions.len(), 2);
    }

    #[test]
    fn test_mcts() {
        let mut mcts = MCTS::new(1.414, 1000);
        let game = TicTacToe::new();
        let best = mcts.search(&game);
        assert!(best.is_some());
    }
}
