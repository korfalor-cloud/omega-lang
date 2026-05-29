/// Topological data analysis: simplicial complexes, persistent homology, Mapper algorithm.

use std::collections::{HashMap, HashSet, BTreeMap};

pub type Vertex = usize;
pub type Simplex = Vec<Vertex>;

/// Simplicial complex: a collection of simplices closed under face inclusion.
#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    pub simplices: Vec<Simplex>,
    pub max_dimension: usize,
}

impl SimplicialComplex {
    pub fn new() -> Self {
        Self { simplices: Vec::new(), max_dimension: 0 }
    }

    pub fn add_simplex(&mut self, simplex: Simplex) {
        let dim = simplex.len().saturating_sub(1);
        self.max_dimension = self.max_dimension.max(dim);
        self.simplices.push(simplex);
    }

    pub fn add_vertex(&mut self, v: Vertex) {
        self.add_simplex(vec![v]);
    }

    pub fn add_edge(&mut self, a: Vertex, b: Vertex) {
        self.add_simplex(vec![a, b]);
    }

    pub fn add_triangle(&mut self, a: Vertex, b: Vertex, c: Vertex) {
        self.add_simplex(vec![a, b, c]);
    }

    /// Get all simplices of a given dimension.
    pub fn simplices_of_dim(&self, dim: usize) -> Vec<&Simplex> {
        self.simplices.iter().filter(|s| s.len() == dim + 1).collect()
    }

    /// Get all faces of a simplex.
    pub fn faces(simplex: &Simplex) -> Vec<Simplex> {
        let mut faces = Vec::new();
        let n = simplex.len();
        // Generate all subsets of size n-1
        for i in 0..n {
            let face: Simplex = simplex.iter().enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &v)| v)
                .collect();
            faces.push(face);
        }
        faces
    }

    /// Euler characteristic: sum of (-1)^dim * count.
    pub fn euler_characteristic(&self) -> i64 {
        let mut chi = 0i64;
        for dim in 0..=self.max_dimension {
            let count = self.simplices_of_dim(dim).len() as i64;
            if dim % 2 == 0 {
                chi += count;
            } else {
                chi -= count;
            }
        }
        chi
    }

    /// Compute boundary matrix for a given dimension.
    pub fn boundary_matrix(&self, dim: usize) -> Vec<Vec<i32>> {
        let k_simplices = self.simplices_of_dim(dim);
        let km1_simplices = self.simplices_of_dim(dim.saturating_sub(1));

        let mut matrix = vec![vec![0; km1_simplices.len()]; k_simplices.len()];

        for (i, simplex) in k_simplices.iter().enumerate() {
            let faces = Self::faces(simplex);
            for (j, face) in km1_simplices.iter().enumerate() {
                if let Some(pos) = faces.iter().position(|f| f == face) {
                    matrix[i][j] = if pos % 2 == 0 { 1 } else { -1 };
                }
            }
        }

        matrix
    }
}

/// Rips complex built from a distance matrix.
pub fn vietoris_rips(distances: &[Vec<f64>], n: usize, epsilon: f64) -> SimplicialComplex {
    let mut complex = SimplicialComplex::new();

    // Add vertices
    for i in 0..n {
        complex.add_vertex(i);
    }

    // Add edges
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if distances[i][j] <= epsilon {
                edges.push((i, j));
                complex.add_edge(i, j);
            }
        }
    }

    // Add higher simplices (up to triangles for simplicity)
    for i in 0..n {
        for &(j, k) in edges.iter().filter(|&&(a, _)| a == i).map(|&(a, b)| (a, b)).collect::<Vec<_>>().iter() {
            if j != i && k != i && distances[j][k] <= epsilon {
                let mut tri = vec![i, j, k];
                tri.sort();
                tri.dedup();
                if tri.len() == 3 && !complex.simplices.contains(&tri) {
                    complex.add_triangle(tri[0], tri[1], tri[2]);
                }
            }
        }
    }

    complex
}

/// Alpha complex (subset of Delaunay triangulation).
pub fn alpha_complex(points: &[(f64, f64)], alpha: f64) -> SimplicialComplex {
    let n = points.len();
    let mut complex = SimplicialComplex::new();

    for i in 0..n {
        complex.add_vertex(i);
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = points[i].0 - points[j].0;
            let dy = points[i].1 - points[j].1;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist <= 2.0 * alpha {
                complex.add_edge(i, j);
            }
        }
    }

    // Check triangles
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let circumradius = circumradius_2d(points[i], points[j], points[k]);
                if circumradius <= alpha {
                    complex.add_triangle(i, j, k);
                }
            }
        }
    }

    complex
}

fn circumradius_2d(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    let ax = a.0; let ay = a.1;
    let bx = b.0; let by = b.1;
    let cx = c.0; let cy = c.1;

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-10 { return f64::INFINITY; }

    let ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;

    ((ux - ax).powi(2) + (uy - ay).powi(2)).sqrt()
}

/// Persistent homology: compute birth-death pairs.
pub struct PersistentHomology;

impl PersistentHomology {
    /// Compute persistence pairs from a filtration.
    pub fn compute(filtration: &[(f64, Simplex)]) -> Vec<(usize, f64, f64)> {
        // Simplified: track connected components (H0)
        let mut pairs = Vec::new();
        let mut uf = UnionFind::new(filtration.len() * 2); // over-allocate
        let mut birth: HashMap<usize, f64> = HashMap::new();
        let max_vertex = filtration.iter()
            .flat_map(|(_, s)| s.iter())
            .copied()
            .max()
            .unwrap_or(0);

        let mut vertex_birth: HashMap<usize, f64> = HashMap::new();

        for (time, simplex) in filtration {
            if simplex.len() == 1 {
                let v = simplex[0];
                vertex_birth.insert(v, *time);
                birth.insert(v, *time);
            } else if simplex.len() == 2 {
                let a = simplex[0];
                let b = simplex[1];
                let root_a = uf.find(a);
                let root_b = uf.find(b);

                if root_a != root_b {
                    let birth_a = *birth.get(&root_a).unwrap_or(time);
                    let birth_b = *birth.get(&root_b).unwrap_or(time);

                    let (younger, older) = if birth_a > birth_b {
                        (root_a, root_b)
                    } else {
                        (root_b, root_a)
                    };

                    pairs.push((0, birth[&younger], *time));
                    birth.remove(&younger);
                    uf.union(a, b);
                    let new_root = uf.find(a);
                    birth.insert(new_root, birth[&older]);
                }
            }
        }

        // Remaining components die at infinity
        for (_, &b) in &birth {
            pairs.push((0, b, f64::INFINITY));
        }

        pairs
    }
}

/// Union-Find for persistent homology.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        if root_x == root_y { return; }
        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
        } else {
            self.parent[root_y] = root_x;
            self.rank[root_x] += 1;
        }
    }
}

/// Mapper algorithm: topological summary of high-dimensional data.
pub struct Mapper;

impl Mapper {
    /// Simple 1D mapper: cover data with intervals, cluster each pullback, build nerve.
    pub fn compute(data: &[Vec<f64>], filter_values: &[f64], num_intervals: usize, overlap: f64) -> SimplicialComplex {
        let min_val = filter_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = filter_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        let interval_size = range / num_intervals as f64 * (1.0 + overlap);

        let mut covers: Vec<Vec<usize>> = vec![Vec::new(); num_intervals];
        for (i, &val) in filter_values.iter().enumerate() {
            for j in 0..num_intervals {
                let center = min_val + (j as f64 + 0.5) * range / num_intervals as f64;
                if (val - center).abs() <= interval_size / 2.0 {
                    covers[j].push(i);
                }
            }
        }

        // Cluster each cover set
        let mut cluster_labels: HashMap<usize, usize> = HashMap::new();
        let mut node_id = 0;

        for (j, cover) in covers.iter().enumerate() {
            let clusters = simple_dbscan(data, cover, 0.5, 2);
            for cluster in clusters {
                for &point in &cluster {
                    cluster_labels.insert(point, node_id);
                }
                node_id += 1;
            }
        }

        // Build nerve complex
        let mut complex = SimplicialComplex::new();
        for i in 0..node_id {
            complex.add_vertex(i);
        }

        // Connect nodes that share data points
        for i in 0..num_intervals {
            for j in (i + 1)..num_intervals {
                let nodes_i: HashSet<usize> = covers[i].iter()
                    .filter_map(|p| cluster_labels.get(p))
                    .copied()
                    .collect();
                let nodes_j: HashSet<usize> = covers[j].iter()
                    .filter_map(|p| cluster_labels.get(p))
                    .copied()
                    .collect();

                for &ni in &nodes_i {
                    for &nj in &nodes_j {
                        if ni != nj {
                            complex.add_edge(ni, nj);
                        }
                    }
                }
            }
        }

        complex
    }
}

fn simple_dbscan(data: &[Vec<f64>], indices: &[usize], eps: f64, min_pts: usize) -> Vec<Vec<usize>> {
    let mut clusters = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();

    for &idx in indices {
        if visited.contains(&idx) { continue; }
        visited.insert(idx);

        let mut neighbors = Vec::new();
        for &other in indices {
            if other == idx { continue; }
            let dist: f64 = data[idx].iter().zip(data[other].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist <= eps {
                neighbors.push(other);
            }
        }

        if neighbors.len() >= min_pts {
            let mut cluster = vec![idx];
            let mut queue: Vec<usize> = neighbors.clone();
            while let Some(n) = queue.pop() {
                if visited.contains(&n) { continue; }
                visited.insert(n);
                cluster.push(n);

                let mut n_neighbors = Vec::new();
                for &other in indices {
                    if other == n { continue; }
                    let dist: f64 = data[n].iter().zip(data[other].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if dist <= eps {
                        n_neighbors.push(other);
                    }
                }
                if n_neighbors.len() >= min_pts {
                    queue.extend(n_neighbors);
                }
            }
            clusters.push(cluster);
        }
    }

    clusters
}

/// Betti numbers: number of k-dimensional holes.
pub fn betti_numbers(complex: &SimplicialComplex) -> Vec<usize> {
    let mut betti = Vec::new();
    for dim in 0..=complex.max_dimension {
        let boundary_k = complex.boundary_matrix(dim);
        let boundary_kp1 = if dim + 1 <= complex.max_dimension {
            complex.boundary_matrix(dim + 1)
        } else {
            Vec::new()
        };

        let rank_k = matrix_rank(&boundary_k);
        let rank_kp1 = if boundary_kp1.is_empty() { 0 } else { matrix_rank(&boundary_kp1) };

        let n_k = complex.simplices_of_dim(dim).len();
        betti.push(n_k - rank_k - rank_kp1);
    }
    betti
}

fn matrix_rank(matrix: &[Vec<i32>]) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() { return 0; }
    let mut m = matrix.to_vec();
    let rows = m.len();
    let cols = m[0].len();
    let mut rank = 0;

    for col in 0..cols {
        let mut pivot_row = None;
        for row in rank..rows {
            if m[row][col] != 0 {
                pivot_row = Some(row);
                break;
            }
        }

        if let Some(pr) = pivot_row {
            m.swap(rank, pr);
            let pivot = m[rank][col];
            for j in 0..cols {
                m[rank][j] /= pivot; // Integer division (approximate)
            }

            for row in 0..rows {
                if row == rank || m[row][col] == 0 { continue; }
                let factor = m[row][col];
                for j in 0..cols {
                    m[row][j] -= factor * m[rank][j];
                }
            }
            rank += 1;
        }
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplicial_complex() {
        let mut sc = SimplicialComplex::new();
        sc.add_triangle(0, 1, 2);
        sc.add_edge(1, 3);

        assert_eq!(sc.simplices_of_dim(0).len(), 0); // no standalone vertices added
        assert_eq!(sc.simplices_of_dim(1).len(), 1); // edge (1,3)
        assert_eq!(sc.simplices_of_dim(2).len(), 1); // triangle
    }

    #[test]
    fn test_euler_characteristic() {
        let mut sc = SimplicialComplex::new();
        sc.add_vertex(0);
        sc.add_vertex(1);
        sc.add_vertex(2);
        sc.add_edge(0, 1);
        sc.add_edge(1, 2);
        sc.add_edge(0, 2);
        sc.add_triangle(0, 1, 2);

        // V - E + F = 3 - 3 + 1 = 1
        assert_eq!(sc.euler_characteristic(), 1);
    }

    #[test]
    fn test_vietoris_rips() {
        let points = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let mut distances = vec![vec![0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let dx = points[i].0 - points[j].0;
                let dy = points[i].1 - points[j].1;
                distances[i][j] = (dx * dx + dy * dy).sqrt();
            }
        }

        let complex = vietoris_rips(&distances, 3, 1.5);
        assert!(complex.simplices.len() > 3); // vertices + edges
    }

    #[test]
    fn test_persistent_homology() {
        let filtration = vec![
            (0.0, vec![0]),
            (0.0, vec![1]),
            (0.5, vec![0, 1]),
            (1.0, vec![2]),
            (1.5, vec![1, 2]),
        ];
        let pairs = PersistentHomology::compute(&filtration);
        assert!(!pairs.is_empty());
    }
}
