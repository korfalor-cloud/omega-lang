/// Weighted graph with Dijkstra, Bellman-Ford, A*, and Floyd-Warshall.

use std::collections::{HashMap, HashSet, BinaryHeap, VecDeque};
use std::cmp::Reverse;

#[derive(Debug, Clone)]
pub struct WeightedGraph {
    adjacency: HashMap<usize, Vec<(usize, f64)>>,
    node_count: usize,
    directed: bool,
}

#[derive(Debug, Clone)]
pub struct WeightedPath {
    pub path: Vec<usize>,
    pub cost: f64,
}

impl WeightedGraph {
    pub fn new(directed: bool) -> Self {
        Self {
            adjacency: HashMap::new(),
            node_count: 0,
            directed,
        }
    }

    pub fn add_node(&mut self) -> usize {
        let id = self.node_count;
        self.adjacency.entry(id).or_insert_with(Vec::new);
        self.node_count += 1;
        id
    }

    pub fn add_nodes(&mut self, count: usize) -> Vec<usize> {
        (0..count).map(|_| self.add_node()).collect()
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.adjacency.entry(from).or_insert_with(Vec::new).push((to, weight));
        if !self.directed {
            self.adjacency.entry(to).or_insert_with(Vec::new).push((from, weight));
        }
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn neighbors(&self, node: usize) -> &[(usize, f64)] {
        self.adjacency.get(&node).map_or(&[], |v| v.as_slice())
    }

    /// Dijkstra's shortest path algorithm.
    pub fn dijkstra(&self, start: usize) -> HashMap<usize, WeightedPath> {
        let mut dist: HashMap<usize, f64> = HashMap::new();
        let mut prev: HashMap<usize, Option<usize>> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(start, 0.0);
        prev.insert(start, None);
        heap.push(Reverse((OrderedFloat(0.0), start)));

        while let Some(Reverse((OrderedFloat(d), node))) = heap.pop() {
            if d > *dist.get(&node).unwrap_or(&f64::INFINITY) {
                continue;
            }
            for &(neighbor, weight) in self.neighbors(node) {
                let new_dist = d + weight;
                if new_dist < *dist.get(&neighbor).unwrap_or(&f64::INFINITY) {
                    dist.insert(neighbor, new_dist);
                    prev.insert(neighbor, Some(node));
                    heap.push(Reverse((OrderedFloat(new_dist), neighbor)));
                }
            }
        }

        let mut result = HashMap::new();
        for (&target, &cost) in &dist {
            let path = self.reconstruct_path(&prev, start, target);
            result.insert(target, WeightedPath { path, cost });
        }
        result
    }

    /// Dijkstra shortest path between two specific nodes.
    pub fn dijkstra_path(&self, start: usize, end: usize) -> Option<WeightedPath> {
        let mut dist: HashMap<usize, f64> = HashMap::new();
        let mut prev: HashMap<usize, Option<usize>> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(start, 0.0);
        prev.insert(start, None);
        heap.push(Reverse((OrderedFloat(0.0), start)));

        while let Some(Reverse((OrderedFloat(d), node))) = heap.pop() {
            if node == end {
                let path = self.reconstruct_path(&prev, start, end);
                return Some(WeightedPath { path, cost: d });
            }
            if d > *dist.get(&node).unwrap_or(&f64::INFINITY) {
                continue;
            }
            for &(neighbor, weight) in self.neighbors(node) {
                let new_dist = d + weight;
                if new_dist < *dist.get(&neighbor).unwrap_or(&f64::INFINITY) {
                    dist.insert(neighbor, new_dist);
                    prev.insert(neighbor, Some(node));
                    heap.push(Reverse((OrderedFloat(new_dist), neighbor)));
                }
            }
        }
        None
    }

    /// Bellman-Ford algorithm (handles negative weights).
    pub fn bellman_ford(&self, start: usize) -> Result<HashMap<usize, WeightedPath>, String> {
        let edges = self.all_edges();
        let mut dist: HashMap<usize, f64> = HashMap::new();
        let mut prev: HashMap<usize, Option<usize>> = HashMap::new();

        dist.insert(start, 0.0);
        prev.insert(start, None);

        for _ in 0..self.node_count - 1 {
            for &(from, to, weight) in &edges {
                let d = *dist.get(&from).unwrap_or(&f64::INFINITY);
                if d + weight < *dist.get(&to).unwrap_or(&f64::INFINITY) {
                    dist.insert(to, d + weight);
                    prev.insert(to, Some(from));
                }
                if !self.directed {
                    let d2 = *dist.get(&to).unwrap_or(&f64::INFINITY);
                    if d2 + weight < *dist.get(&from).unwrap_or(&f64::INFINITY) {
                        dist.insert(from, d2 + weight);
                        prev.insert(from, Some(to));
                    }
                }
            }
        }

        // Check for negative cycles
        for &(from, to, weight) in &edges {
            let d = *dist.get(&from).unwrap_or(&f64::INFINITY);
            if d + weight < *dist.get(&to).unwrap_or(&f64::INFINITY) {
                return Err("Graph contains a negative-weight cycle".to_string());
            }
        }

        let mut result = HashMap::new();
        for (&target, &cost) in &dist {
            let path = self.reconstruct_path(&prev, start, target);
            result.insert(target, WeightedPath { path, cost });
        }
        Ok(result)
    }

    /// Floyd-Warshall all-pairs shortest paths.
    pub fn floyd_warshall(&self) -> Vec<Vec<f64>> {
        let n = self.node_count;
        let mut dist = vec![vec![f64::INFINITY; n]; n];

        for i in 0..n {
            dist[i][i] = 0.0;
        }

        for (&from, neighbors) in &self.adjacency {
            for &(to, weight) in neighbors {
                dist[from][to] = dist[from][to].min(weight);
            }
        }

        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if dist[i][k] + dist[k][j] < dist[i][j] {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }

        dist
    }

    /// A* search with a heuristic function.
    pub fn astar<F>(&self, start: usize, goal: usize, heuristic: F) -> Option<WeightedPath>
    where
        F: Fn(usize) -> f64,
    {
        let mut g_score: HashMap<usize, f64> = HashMap::new();
        let mut prev: HashMap<usize, Option<usize>> = HashMap::new();
        let mut heap = BinaryHeap::new();

        g_score.insert(start, 0.0);
        prev.insert(start, None);
        let f = heuristic(start);
        heap.push(Reverse((OrderedFloat(f), start)));

        while let Some(Reverse((OrderedFloat(_), node))) = heap.pop() {
            if node == goal {
                let cost = *g_score.get(&goal).unwrap();
                let path = self.reconstruct_path(&prev, start, goal);
                return Some(WeightedPath { path, cost });
            }

            let current_g = *g_score.get(&node).unwrap_or(&f64::INFINITY);
            for &(neighbor, weight) in self.neighbors(node) {
                let tentative_g = current_g + weight;
                if tentative_g < *g_score.get(&neighbor).unwrap_or(&f64::INFINITY) {
                    g_score.insert(neighbor, tentative_g);
                    prev.insert(neighbor, Some(node));
                    let f = tentative_g + heuristic(neighbor);
                    heap.push(Reverse((OrderedFloat(f), neighbor)));
                }
            }
        }
        None
    }

    /// Prim's minimum spanning tree.
    pub fn prim_mst(&self) -> Vec<(usize, usize, f64)> {
        if self.node_count == 0 {
            return Vec::new();
        }

        let mut in_mst = vec![false; self.node_count];
        let mut heap = BinaryHeap::new();
        let mut mst = Vec::new();

        in_mst[0] = true;
        for &(neighbor, weight) in self.neighbors(0) {
            heap.push(Reverse((OrderedFloat(weight), 0, neighbor)));
        }

        while let Some(Reverse((OrderedFloat(w), from, to))) = heap.pop() {
            if in_mst[to] {
                continue;
            }
            in_mst[to] = true;
            mst.push((from, to, w));

            for &(neighbor, weight) in self.neighbors(to) {
                if !in_mst[neighbor] {
                    heap.push(Reverse((OrderedFloat(weight), to, neighbor)));
                }
            }
        }

        mst
    }

    fn all_edges(&self) -> Vec<(usize, usize, f64)> {
        let mut edges = Vec::new();
        for (&from, neighbors) in &self.adjacency {
            for &(to, weight) in neighbors {
                if self.directed || from <= to {
                    edges.push((from, to, weight));
                }
            }
        }
        edges
    }

    fn reconstruct_path(&self, prev: &HashMap<usize, Option<usize>>, start: usize, end: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = end;
        path.push(current);
        while let Some(Some(p)) = prev.get(&current) {
            path.push(*p);
            current = *p;
        }
        if current != start {
            return Vec::new();
        }
        path.reverse();
        path
    }
}

/// Wrapper for f64 to make it orderable in BinaryHeap.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Network flow using Edmonds-Karp (Ford-Fulkerson with BFS).
pub struct MaxFlow {
    capacity: Vec<Vec<f64>>,
    flow: Vec<Vec<f64>>,
    n: usize,
}

impl MaxFlow {
    pub fn new(n: usize) -> Self {
        Self {
            capacity: vec![vec![0.0; n]; n],
            flow: vec![vec![0.0; n]; n],
            n,
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, cap: f64) {
        self.capacity[from][to] += cap;
    }

    pub fn compute(&mut self, source: usize, sink: usize) -> f64 {
        let mut total_flow = 0.0;

        loop {
            let path = self.bfs_augmenting_path(source, sink);
            let Some((path, bottleneck)) = path else {
                break;
            };

            total_flow += bottleneck;

            for i in 0..path.len() - 1 {
                let u = path[i];
                let v = path[i + 1];
                self.flow[u][v] += bottleneck;
                self.flow[v][u] -= bottleneck;
            }
        }

        total_flow
    }

    fn bfs_augmenting_path(&self, source: usize, sink: usize) -> Option<(Vec<usize>, f64)> {
        let mut parent = vec![usize::MAX; self.n];
        let mut min_cap = vec![f64::MAX; self.n];
        let mut visited = vec![false; self.n];
        let mut queue = VecDeque::new();

        visited[source] = true;
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            for v in 0..self.n {
                let residual = self.capacity[u][v] - self.flow[u][v];
                if !visited[v] && residual > 0.0 {
                    visited[v] = true;
                    parent[v] = u;
                    min_cap[v] = min_cap[u].min(residual);
                    queue.push_back(v);
                    if v == sink {
                        let mut path = Vec::new();
                        let mut current = sink;
                        path.push(current);
                        while current != source {
                            current = parent[current];
                            path.push(current);
                        }
                        path.reverse();
                        return Some((path, min_cap[sink]));
                    }
                }
            }
        }
        None
    }

    pub fn min_cut(&self, source: usize) -> HashSet<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(source);
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            for v in 0..self.n {
                let residual = self.capacity[u][v] - self.flow[u][v];
                if !visited.contains(&v) && residual > 0.0 {
                    visited.insert(v);
                    queue.push_back(v);
                }
            }
        }
        visited
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dijkstra() {
        let mut g = WeightedGraph::new(true);
        g.add_nodes(5);
        g.add_edge(0, 1, 4.0);
        g.add_edge(0, 2, 1.0);
        g.add_edge(2, 1, 2.0);
        g.add_edge(1, 3, 1.0);
        g.add_edge(2, 3, 5.0);

        let result = g.dijkstra_path(0, 3).unwrap();
        assert!((result.cost - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_floyd_warshall() {
        let mut g = WeightedGraph::new(true);
        g.add_nodes(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 4.0);

        let dist = g.floyd_warshall();
        assert!((dist[0][2] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_flow() {
        let mut mf = MaxFlow::new(4);
        mf.add_edge(0, 1, 3.0);
        mf.add_edge(0, 2, 2.0);
        mf.add_edge(1, 2, 1.0);
        mf.add_edge(1, 3, 2.0);
        mf.add_edge(2, 3, 3.0);

        let flow = mf.compute(0, 3);
        assert!((flow - 4.0).abs() < f64::EPSILON);
    }
}
