/// Constraint Satisfaction Problem solver with backtracking, arc consistency, and search heuristics.

use std::collections::{HashMap, HashSet, VecDeque};

pub type VarId = usize;
pub type Value = i32;

#[derive(Debug, Clone)]
pub struct CSP {
    variables: Vec<Variable>,
    constraints: Vec<Constraint>,
    adjacency: HashMap<VarId, Vec<usize>>, // var -> constraint indices
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub id: VarId,
    pub name: String,
    pub domain: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub variables: Vec<VarId>,
    pub kind: ConstraintKind,
}

#[derive(Debug, Clone)]
pub enum ConstraintKind {
    AllDifferent,
    NotEqual(VarId, VarId),
    Equal(VarId, VarId),
    LessThan(VarId, VarId),
    GreaterThan(VarId, VarId),
    SumEquals(Vec<VarId>, Value),
    Custom(Box<dyn Fn(&HashMap<VarId, Value>) -> bool>),
}

impl std::fmt::Debug for ConstraintKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintKind::AllDifferent => write!(f, "AllDifferent"),
            ConstraintKind::NotEqual(a, b) => write!(f, "NotEqual({}, {})", a, b),
            ConstraintKind::Equal(a, b) => write!(f, "Equal({}, {})", a, b),
            ConstraintKind::LessThan(a, b) => write!(f, "LessThan({}, {})", a, b),
            ConstraintKind::GreaterThan(a, b) => write!(f, "GreaterThan({}, {})", a, b),
            ConstraintKind::SumEquals(vars, val) => write!(f, "SumEquals({:?} = {})", vars, val),
            ConstraintKind::Custom(_) => write!(f, "Custom(...)"),
        }
    }
}

impl CSP {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            constraints: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    pub fn add_variable(&mut self, name: &str, domain: &[Value]) -> VarId {
        let id = self.variables.len();
        self.variables.push(Variable {
            id,
            name: name.to_string(),
            domain: domain.to_vec(),
        });
        id
    }

    pub fn add_constraint(&mut self, constraint: Constraint) {
        let idx = self.constraints.len();
        for &var in &constraint.variables {
            self.adjacency.entry(var).or_insert_with(Vec::new).push(idx);
        }
        self.constraints.push(constraint);
    }

    pub fn add_not_equal(&mut self, a: VarId, b: VarId) {
        self.add_constraint(Constraint {
            variables: vec![a, b],
            kind: ConstraintKind::NotEqual(a, b),
        });
    }

    pub fn add_less_than(&mut self, a: VarId, b: VarId) {
        self.add_constraint(Constraint {
            variables: vec![a, b],
            kind: ConstraintKind::LessThan(a, b),
        });
    }

    pub fn add_greater_than(&mut self, a: VarId, b: VarId) {
        self.add_constraint(Constraint {
            variables: vec![a, b],
            kind: ConstraintKind::GreaterThan(a, b),
        });
    }

    pub fn add_all_different(&mut self, vars: &[VarId]) {
        self.add_constraint(Constraint {
            variables: vars.to_vec(),
            kind: ConstraintKind::AllDifferent,
        });
    }

    pub fn add_sum_equals(&mut self, vars: &[VarId], target: Value) {
        self.add_constraint(Constraint {
            variables: vars.to_vec(),
            kind: ConstraintKind::SumEquals(vars.to_vec(), target),
        });
    }

    pub fn add_custom(&mut self, vars: Vec<VarId>, check: Box<dyn Fn(&HashMap<VarId, Value>) -> bool>) {
        self.add_constraint(Constraint {
            variables: vars.clone(),
            kind: ConstraintKind::Custom(check),
        });
    }

    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    pub fn domain(&self, var: VarId) -> &[Value] {
        &self.variables[var].domain
    }

    pub fn set_domain(&mut self, var: VarId, domain: Vec<Value>) {
        self.variables[var].domain = domain;
    }

    /// Check if an assignment is consistent with all constraints.
    pub fn is_consistent(&self, assignment: &HashMap<VarId, Value>) -> bool {
        for constraint in &self.constraints {
            if !self.check_constraint(constraint, assignment) {
                return false;
            }
        }
        true
    }

    fn check_constraint(&self, constraint: &Constraint, assignment: &HashMap<VarId, Value>) -> bool {
        match &constraint.kind {
            ConstraintKind::NotEqual(a, b) => {
                match (assignment.get(a), assignment.get(b)) {
                    (Some(va), Some(vb)) => va != vb,
                    _ => true,
                }
            }
            ConstraintKind::Equal(a, b) => {
                match (assignment.get(a), assignment.get(b)) {
                    (Some(va), Some(vb)) => va == vb,
                    _ => true,
                }
            }
            ConstraintKind::LessThan(a, b) => {
                match (assignment.get(a), assignment.get(b)) {
                    (Some(va), Some(vb)) => va < vb,
                    _ => true,
                }
            }
            ConstraintKind::GreaterThan(a, b) => {
                match (assignment.get(a), assignment.get(b)) {
                    (Some(va), Some(vb)) => va > vb,
                    _ => true,
                }
            }
            ConstraintKind::AllDifferent => {
                let values: Vec<&Value> = constraint.variables.iter()
                    .filter_map(|v| assignment.get(v))
                    .collect();
                let unique: HashSet<&Value> = values.iter().copied().collect();
                values.len() == unique.len()
            }
            ConstraintKind::SumEquals(vars, target) => {
                let all_assigned = vars.iter().all(|v| assignment.contains_key(v));
                if !all_assigned {
                    return true;
                }
                let sum: Value = vars.iter().map(|v| assignment[v]).sum();
                sum == *target
            }
            ConstraintKind::Custom(check) => check(assignment),
        }
    }

    /// Solve using backtracking with forward checking.
    pub fn solve(&self) -> Option<HashMap<VarId, Value>> {
        let mut domains: Vec<Vec<Value>> = self.variables.iter().map(|v| v.domain.clone()).collect();
        let mut assignment = HashMap::new();

        // Apply arc consistency first
        self.ac3(&mut domains);

        self.backtrack(&mut assignment, &mut domains)
    }

    fn backtrack(
        &self,
        assignment: &mut HashMap<VarId, Value>,
        domains: &mut Vec<Vec<Value>>,
    ) -> Option<HashMap<VarId, Value>> {
        if assignment.len() == self.variables.len() {
            return Some(assignment.clone());
        }

        // Select variable using MRV (Minimum Remaining Values)
        let var = self.select_unassigned(assignment, domains);

        // Try values in order (optionally with LCV heuristic)
        let values: Vec<Value> = domains[var].clone();
        for &value in &values {
            assignment.insert(var, value);

            if self.is_consistent(assignment) {
                // Forward checking: reduce domains
                let mut new_domains = domains.clone();
                if self.forward_check(var, value, &mut new_domains, assignment) {
                    if let Some(result) = self.backtrack(assignment, &mut new_domains) {
                        return Some(result);
                    }
                }
            }

            assignment.remove(&var);
        }

        None
    }

    /// MRV heuristic: select variable with smallest domain.
    fn select_unassigned(&self, assignment: &HashMap<VarId, Value>, domains: &[Vec<Value>]) -> VarId {
        let mut best_var = usize::MAX;
        let mut best_count = usize::MAX;

        for (i, var) in self.variables.iter().enumerate() {
            if assignment.contains_key(&i) {
                continue;
            }
            let count = domains[i].len();
            if count < best_count {
                best_count = count;
                best_var = i;
            }
        }

        best_var
    }

    /// Forward checking: after assigning var=value, reduce domains of neighbors.
    fn forward_check(
        &self,
        var: VarId,
        value: Value,
        domains: &mut Vec<Vec<Value>>,
        assignment: &HashMap<VarId, Value>,
    ) -> bool {
        if let Some(constraint_indices) = self.adjacency.get(&var) {
            for &cidx in constraint_indices {
                let constraint = &self.constraints[cidx];
                for &other_var in &constraint.variables {
                    if other_var == var || assignment.contains_key(&other_var) {
                        continue;
                    }

                    // Remove inconsistent values from other_var's domain
                    let original_len = domains[other_var].len();
                    domains[other_var].retain(|&other_val| {
                        let mut test = assignment.clone();
                        test.insert(var, value);
                        test.insert(other_var, other_val);
                        self.check_constraint(constraint, &test)
                    });

                    if domains[other_var].is_empty() {
                        domains[other_var] = vec![value]; // restore for backtracking
                        return false;
                    }
                }
            }
        }
        true
    }

    /// AC-3 arc consistency algorithm.
    pub fn ac3(&self, domains: &mut Vec<Vec<Value>>) {
        let mut queue: VecDeque<(VarId, VarId)> = VecDeque::new();

        // Initialize with all arcs
        for constraint in &self.constraints {
            for i in 0..constraint.variables.len() {
                for j in 0..constraint.variables.len() {
                    if i != j {
                        queue.push_back((constraint.variables[i], constraint.variables[j]));
                    }
                }
            }
        }

        while let Some((xi, xj)) = queue.pop_front() {
            if self.revise(domains, xi, xj) {
                if domains[xi].is_empty() {
                    return;
                }
                if let Some(indices) = self.adjacency.get(&xi) {
                    for &cidx in indices {
                        let constraint = &self.constraints[cidx];
                        for &xk in &constraint.variables {
                            if xk != xi && xk != xj {
                                queue.push_back((xk, xi));
                            }
                        }
                    }
                }
            }
        }
    }

    fn revise(&self, domains: &mut Vec<Vec<Value>>, xi: VarId, xj: VarId) -> bool {
        let mut revised = false;
        let original = domains[xi].clone();

        domains[xi].retain(|&val_xi| {
            // Check if there exists a value in xj's domain that satisfies constraints
            domains[xj].iter().any(|&val_xj| {
                let mut assignment = HashMap::new();
                assignment.insert(xi, val_xi);
                assignment.insert(xj, val_xj);

                // Check all constraints involving both xi and xj
                if let Some(indices) = self.adjacency.get(&xi) {
                    for &cidx in indices {
                        let constraint = &self.constraints[cidx];
                        if constraint.variables.contains(&xj) {
                            if !self.check_constraint(constraint, &assignment) {
                                return false;
                            }
                        }
                    }
                }
                true
            })
        });

        domains[xi].len() != original.len()
    }

    /// Find all solutions.
    pub fn solve_all(&self) -> Vec<HashMap<VarId, Value>> {
        let mut domains: Vec<Vec<Value>> = self.variables.iter().map(|v| v.domain.clone()).collect();
        self.ac3(&mut domains);
        let mut results = Vec::new();
        let mut assignment = HashMap::new();
        self.backtrack_all(&mut assignment, &mut domains, &mut results);
        results
    }

    fn backtrack_all(
        &self,
        assignment: &mut HashMap<VarId, Value>,
        domains: &mut Vec<Vec<Value>>,
        results: &mut Vec<HashMap<VarId, Value>>,
    ) {
        if assignment.len() == self.variables.len() {
            results.push(assignment.clone());
            return;
        }

        let var = self.select_unassigned(assignment, domains);
        let values: Vec<Value> = domains[var].clone();

        for &value in &values {
            assignment.insert(var, value);

            if self.is_consistent(assignment) {
                let mut new_domains = domains.clone();
                if self.forward_check(var, value, &mut new_domains, assignment) {
                    self.backtrack_all(assignment, &mut new_domains, results);
                }
            }

            assignment.remove(&var);
        }
    }

    /// Count solutions (without storing them all).
    pub fn count_solutions(&self) -> usize {
        let mut domains: Vec<Vec<Value>> = self.variables.iter().map(|v| v.domain.clone()).collect();
        self.ac3(&mut domains);
        let mut assignment = HashMap::new();
        self.count_backtrack(&mut assignment, &mut domains)
    }

    fn count_backtrack(
        &self,
        assignment: &mut HashMap<VarId, Value>,
        domains: &mut Vec<Vec<Value>>,
    ) -> usize {
        if assignment.len() == self.variables.len() {
            return 1;
        }

        let var = self.select_unassigned(assignment, domains);
        let values: Vec<Value> = domains[var].clone();
        let mut count = 0;

        for &value in &values {
            assignment.insert(var, value);
            if self.is_consistent(assignment) {
                let mut new_domains = domains.clone();
                if self.forward_check(var, value, &mut new_domains, assignment) {
                    count += self.count_backtrack(assignment, &mut new_domains);
                }
            }
            assignment.remove(&var);
        }

        count
    }
}

impl Default for CSP {
    fn default() -> Self {
        Self::new()
    }
}

/// Sudoku solver using CSP.
pub fn solve_sudoku(grid: &[[i32; 9]; 9]) -> Option<[[i32; 9]; 9]> {
    let mut csp = CSP::new();

    // Create 81 variables
    let mut vars = [[0usize; 9]; 9];
    for r in 0..9 {
        for c in 0..9 {
            let domain = if grid[r][c] != 0 {
                vec![grid[r][c]]
            } else {
                (1..=9).collect()
            };
            vars[r][c] = csp.add_variable(&format!("r{}c{}", r, c), &domain);
        }
    }

    // Row constraints
    for r in 0..9 {
        let row_vars: Vec<VarId> = (0..9).map(|c| vars[r][c]).collect();
        csp.add_all_different(&row_vars);
    }

    // Column constraints
    for c in 0..9 {
        let col_vars: Vec<VarId> = (0..9).map(|r| vars[r][c]).collect();
        csp.add_all_different(&col_vars);
    }

    // Box constraints
    for box_r in 0..3 {
        for box_c in 0..3 {
            let mut box_vars = Vec::new();
            for r in (box_r * 3)..(box_r * 3 + 3) {
                for c in (box_c * 3)..(box_c * 3 + 3) {
                    box_vars.push(vars[r][c]);
                }
            }
            csp.add_all_different(&box_vars);
        }
    }

    let solution = csp.solve()?;
    let mut result = [[0i32; 9]; 9];
    for r in 0..9 {
        for c in 0..9 {
            result[r][c] = solution[&vars[r][c]];
        }
    }
    Some(result)
}

/// N-Queens solver.
pub fn solve_n_queens(n: usize) -> Option<Vec<usize>> {
    let mut csp = CSP::new();

    let vars: Vec<VarId> = (0..n).map(|i| {
        csp.add_variable(&format!("q{}", i), &(0..n as i32).collect::<Vec<_>>())
    }).collect();

    // All queens in different columns
    csp.add_all_different(&vars);

    // No two queens on same diagonal
    for i in 0..n {
        for j in (i + 1)..n {
            let vi = vars[i];
            let vj = vars[j];
            let diff = (j - i) as i32;

            csp.add_custom(
                vec![vi, vj],
                Box::new(move |assignment: &HashMap<VarId, Value>| {
                    if let (Some(&qi), Some(&qj)) = (assignment.get(&vi), assignment.get(&vj)) {
                        (qi - qj).abs() != diff
                    } else {
                        true
                    }
                }),
            );
        }
    }

    let solution = csp.solve()?;
    Some(vars.iter().map(|&v| solution[&v] as usize).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_csp() {
        let mut csp = CSP::new();
        let x = csp.add_variable("x", &[1, 2, 3]);
        let y = csp.add_variable("y", &[1, 2, 3]);
        csp.add_not_equal(x, y);
        csp.add_less_than(x, y);

        let solution = csp.solve();
        assert!(solution.is_some());
        let sol = solution.unwrap();
        assert!(sol[&x] < sol[&y]);
        assert!(sol[&x] != sol[&y]);
    }

    #[test]
    fn test_sudoku() {
        let mut grid = [[0i32; 9]; 9];
        // A few given values
        grid[0][0] = 5; grid[0][1] = 3; grid[0][4] = 7;
        grid[1][0] = 6; grid[1][3] = 1; grid[1][4] = 9; grid[1][5] = 5;
        grid[2][1] = 9; grid[2][2] = 8; grid[2][7] = 6;
        grid[3][0] = 8; grid[3][4] = 6; grid[3][8] = 3;
        grid[4][0] = 4; grid[4][3] = 8; grid[4][5] = 3; grid[4][8] = 1;
        grid[5][0] = 7; grid[5][4] = 2; grid[5][8] = 6;
        grid[6][1] = 6; grid[6][6] = 2; grid[6][7] = 8;
        grid[7][3] = 4; grid[7][4] = 1; grid[7][5] = 9; grid[7][8] = 5;
        grid[8][4] = 8; grid[8][7] = 7; grid[8][8] = 9;

        let solution = solve_sudoku(&grid);
        assert!(solution.is_some());
    }

    #[test]
    fn test_n_queens() {
        let solution = solve_n_queens(8);
        assert!(solution.is_some());
        let queens = solution.unwrap();
        assert_eq!(queens.len(), 8);
        // Verify no conflicts
        for i in 0..8 {
            for j in (i + 1)..8 {
                assert_ne!(queens[i], queens[j]);
                assert!((queens[i] as i32 - queens[j] as i32).abs() != (j - i) as i32);
            }
        }
    }

    #[test]
    fn test_all_different() {
        let mut csp = CSP::new();
        let a = csp.add_variable("a", &[1, 2]);
        let b = csp.add_variable("b", &[1, 2]);
        let c = csp.add_variable("c", &[1, 2]);
        csp.add_all_different(&[a, b, c]);

        // Impossible: 3 variables, 2 values each, all different
        let solution = csp.solve();
        assert!(solution.is_none());
    }
}
