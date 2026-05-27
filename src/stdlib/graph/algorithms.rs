/// Graph data structure and algorithms.

use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Reverse;

#[derive(Debug, Clone)]
pub struct Graph {
    adjacency: HashMap<usize, Vec<usize>>,
    node_count: usize,
    directed: bool,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
}

#[derive(Debug, Clone)]
pub struct PathResult {
    pub path: Vec<usize>,
    pub distance: usize,
}

#[derive(Debug, Clone)]
pub struct TopologicalSort {
    pub order: Vec<usize>,
    pub has_cycle: bool,
}

#[derive(Debug, Clone)]
pub struct ConnectedComponents {
    pub components: Vec<Vec<usize>>,
    pub component_id: HashMap<usize, usize>,
}

impl Graph {
    pub fn new(directed: bool) -> Self {
        Self {
            adjacency: HashMap::new(),
            node_count: 0,
            directed,
        }
    }

    pub fn with_capacity(nodes: usize, directed: bool) -> Self {
        Self {
            adjacency: HashMap::with_capacity(nodes),
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

    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.adjacency.entry(from).or_insert_with(Vec::new).push(to);
        if !self.directed {
            self.adjacency.entry(to).or_insert_with(Vec::new).push(from);
        }
    }

    pub fn add_edges(&mut self, edges: &[(usize, usize)]) {
        for &(from, to) in edges {
            self.add_edge(from, to);
        }
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn edge_count(&self) -> usize {
        let total: usize = self.adjacency.values().map(|v| v.len()).sum();
        if self.directed { total } else { total / 2 }
    }

    pub fn neighbors(&self, node: usize) -> &[usize] {
        self.adjacency.get(&node).map_or(&[], |v| v.as_slice())
    }

    pub fn has_node(&self, node: usize) -> bool {
        self.adjacency.contains_key(&node)
    }

    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        self.adjacency.get(&from).map_or(false, |neighbors| neighbors.contains(&to))
    }

    pub fn degree(&self, node: usize) -> usize {
        self.adjacency.get(&node).map_or(0, |v| v.len())
    }

    pub fn nodes(&self) -> Vec<usize> {
        self.adjacency.keys().copied().collect()
    }

    pub fn edges(&self) -> Vec<Edge> {
        let mut edges = Vec::new();
        for (&from, neighbors) in &self.adjacency {
            for &to in neighbors {
                if self.directed || from <= to {
                    edges.push(Edge { from, to });
                }
            }
        }
        edges
    }

    pub fn reverse(&self) -> Self {
        let mut reversed = Graph::new(self.directed);
        for i in 0..self.node_count {
            reversed.add_node();
        }
        for (&from, neighbors) in &self.adjacency {
            for &to in neighbors {
                reversed.adjacency.entry(to).or_insert_with(Vec::new).push(from);
            }
        }
        reversed
    }

    pub fn subgraph(&self, nodes: &HashSet<usize>) -> Self {
        let mut sub = Graph::new(self.directed);
        let mut id_map = HashMap::new();
        for &node in nodes {
            let new_id = sub.add_node();
            id_map.insert(node, new_id);
        }
        for (&from, neighbors) in &self.adjacency {
            if let Some(&new_from) = id_map.get(&from) {
                for &to in neighbors {
                    if let Some(&new_to) = id_map.get(&to) {
                        sub.adjacency.entry(new_from).or_insert_with(Vec::new).push(new_to);
                    }
                }
            }
        }
        sub
    }

    /// Breadth-first search from a start node.
    pub fn bfs(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut order = Vec::new();

        visited.insert(start);
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &neighbor in self.neighbors(node) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        order
    }

    /// Depth-first search from a start node.
    pub fn dfs(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.dfs_recursive(start, &mut visited, &mut order);
        order
    }

    fn dfs_recursive(&self, node: usize, visited: &mut HashSet<usize>, order: &mut Vec<usize>) {
        if !visited.insert(node) {
            return;
        }
        order.push(node);
        for &neighbor in self.neighbors(node) {
            self.dfs_recursive(neighbor, visited, order);
        }
    }

    /// Shortest path (unweighted) using BFS.
    pub fn shortest_path(&self, from: usize, to: usize) -> Option<PathResult> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<usize, usize> = HashMap::new();

        visited.insert(from);
        queue.push_back((from, 0usize));

        while let Some((node, dist)) = queue.pop_front() {
            if node == to {
                let mut path = Vec::new();
                let mut current = to;
                path.push(current);
                while let Some(&p) = parent.get(&current) {
                    path.push(p);
                    current = p;
                }
                path.reverse();
                return Some(PathResult { path, distance: dist });
            }
            for &neighbor in self.neighbors(node) {
                if visited.insert(neighbor) {
                    parent.insert(neighbor, node);
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }
        None
    }

    /// Check if the graph has a cycle (for directed graphs).
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for node in 0..self.node_count {
            if !visited.contains(&node) {
                if self.has_cycle_dfs(node, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        false
    }

    fn has_cycle_dfs(&self, node: usize, visited: &mut HashSet<usize>, rec_stack: &mut HashSet<usize>) -> bool {
        visited.insert(node);
        rec_stack.insert(node);

        for &neighbor in self.neighbors(node) {
            if !visited.contains(&neighbor) {
                if self.has_cycle_dfs(neighbor, visited, rec_stack) {
                    return true;
                }
            } else if rec_stack.contains(&neighbor) {
                return true;
            }
        }

        rec_stack.remove(&node);
        false
    }

    /// Topological sort (only for DAGs).
    pub fn topological_sort(&self) -> TopologicalSort {
        let mut in_degree = vec![0usize; self.node_count];
        for neighbors in self.adjacency.values() {
            for &to in neighbors {
                in_degree[to] += 1;
            }
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..self.node_count {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        let mut order = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &neighbor in self.neighbors(node) {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        TopologicalSort {
            has_cycle: order.len() != self.node_count,
            order,
        }
    }

    /// Find connected components (undirected graphs).
    pub fn connected_components(&self) -> ConnectedComponents {
        let mut visited = HashSet::new();
        let mut components = Vec::new();
        let mut component_id = HashMap::new();

        for node in 0..self.node_count {
            if !visited.contains(&node) {
                let component = self.bfs_from(node, &mut visited);
                let id = components.len();
                for &n in &component {
                    component_id.insert(n, id);
                }
                components.push(component);
            }
        }

        ConnectedComponents { components, component_id }
    }

    fn bfs_from(&self, start: usize, visited: &mut HashSet<usize>) -> Vec<usize> {
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        visited.insert(start);
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            component.push(node);
            for &neighbor in self.neighbors(node) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        component
    }

    /// Is the graph bipartite? (2-colorable)
    pub fn is_bipartite(&self) -> bool {
        let mut color = vec![0i8; self.node_count]; // 0 = uncolored, 1 = group A, -1 = group B

        for start in 0..self.node_count {
            if color[start] != 0 {
                continue;
            }
            let mut queue = VecDeque::new();
            color[start] = 1;
            queue.push_back(start);

            while let Some(node) = queue.pop_front() {
                for &neighbor in self.neighbors(node) {
                    if color[neighbor] == 0 {
                        color[neighbor] = -color[node];
                        queue.push_back(neighbor);
                    } else if color[neighbor] == color[node] {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// All nodes reachable from `start`.
    pub fn reachable_from(&self, start: usize) -> HashSet<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start);
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            for &neighbor in self.neighbors(node) {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        visited
    }

    /// Eccentricity of a node (max shortest-path distance to any other node).
    pub fn eccentricity(&self, node: usize) -> usize {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(node);
        queue.push_back((node, 0usize));
        let mut max_dist = 0;

        while let Some((n, dist)) = queue.pop_front() {
            max_dist = dist;
            for &neighbor in self.neighbors(n) {
                if visited.insert(neighbor) {
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }
        max_dist
    }

    /// Diameter of the graph (longest shortest path).
    pub fn diameter(&self) -> usize {
        (0..self.node_count)
            .map(|n| self.eccentricity(n))
            .max()
            .unwrap_or(0)
    }

    /// Radius of the graph (minimum eccentricity).
    pub fn radius(&self) -> usize {
        (0..self.node_count)
            .map(|n| self.eccentricity(n))
            .min()
            .unwrap_or(0)
    }

    /// Center nodes (nodes with eccentricity == radius).
    pub fn center(&self) -> Vec<usize> {
        let r = self.radius();
        (0..self.node_count)
            .filter(|&n| self.eccentricity(n) == r)
            .collect()
    }

    /// Density: ratio of actual edges to possible edges.
    pub fn density(&self) -> f64 {
        let n = self.node_count as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let max_edges = if self.directed { n * (n - 1.0) } else { n * (n - 1.0) / 2.0 };
        self.edge_count() as f64 / max_edges
    }

    /// Clustering coefficient of a node.
    pub fn clustering_coefficient(&self, node: usize) -> f64 {
        let neighbors: Vec<usize> = self.neighbors(node).to_vec();
        let k = neighbors.len();
        if k < 2 {
            return 0.0;
        }

        let neighbor_set: HashSet<usize> = neighbors.iter().copied().collect();
        let mut triangles = 0;

        for i in 0..k {
            for j in (i + 1)..k {
                if self.has_edge(neighbors[i], neighbors[j]) || self.has_edge(neighbors[j], neighbors[i]) {
                    triangles += 1;
                }
            }
        }

        let max_triangles = k * (k - 1) / 2;
        triangles as f64 / max_triangles as f64
    }

    /// Average clustering coefficient of the graph.
    pub fn average_clustering_coefficient(&self) -> f64 {
        if self.node_count == 0 {
            return 0.0;
        }
        let sum: f64 = (0..self.node_count)
            .map(|n| self.clustering_coefficient(n))
            .sum();
        sum / self.node_count as f64
    }
}

/// Minimum spanning tree using Kruskal's algorithm.
pub fn kruskal_mst(node_count: usize, edges: &[(usize, usize, f64)]) -> Vec<(usize, usize, f64)> {
    let mut sorted_edges = edges.to_vec();
    sorted_edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut uf = UnionFind::new(node_count);
    let mut mst = Vec::new();

    for (from, to, weight) in sorted_edges {
        if uf.union(from, to) {
            mst.push((from, to, weight));
            if mst.len() == node_count - 1 {
                break;
            }
        }
    }
    mst
}

/// Union-Find (Disjoint Set Union) data structure.
#[derive(Debug)]
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);
        if root_x == root_y {
            return false;
        }
        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
            self.size[root_y] += self.size[root_x];
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
            self.size[root_x] += self.size[root_y];
        } else {
            self.parent[root_y] = root_x;
            self.rank[root_x] += 1;
            self.size[root_x] += self.size[root_y];
        }
        true
    }

    pub fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    pub fn component_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.size[root]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs_dfs() {
        let mut g = Graph::new(false);
        g.add_nodes(5);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 3);
        g.add_edge(2, 4);

        let bfs = g.bfs(0);
        assert_eq!(bfs.len(), 5);

        let dfs = g.dfs(0);
        assert_eq!(dfs.len(), 5);
    }

    #[test]
    fn test_shortest_path() {
        let mut g = Graph::new(true);
        g.add_nodes(5);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(0, 3);
        g.add_edge(3, 4);
        g.add_edge(4, 2);

        let result = g.shortest_path(0, 2).unwrap();
        assert_eq!(result.distance, 2);
    }

    #[test]
    fn test_topological_sort() {
        let mut g = Graph::new(true);
        g.add_nodes(4);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 3);
        g.add_edge(2, 3);

        let ts = g.topological_sort();
        assert!(!ts.has_cycle);
        assert_eq!(ts.order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = Graph::new(true);
        g.add_nodes(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        assert!(!g.has_cycle());

        g.add_edge(2, 0);
        assert!(g.has_cycle());
    }

    #[test]
    fn test_connected_components() {
        let mut g = Graph::new(false);
        g.add_nodes(6);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(3, 4);

        let cc = g.connected_components();
        assert_eq!(cc.components.len(), 3);
    }

    #[test]
    fn test_bipartite() {
        let mut g = Graph::new(false);
        g.add_nodes(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 0);
        assert!(g.is_bipartite());

        g.add_edge(0, 2);
        assert!(!g.is_bipartite());
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);
        assert!(uf.union(0, 1));
        assert!(uf.union(2, 3));
        assert!(!uf.union(0, 1)); // already connected
        assert!(uf.union(1, 3));
        assert!(uf.connected(0, 3));
    }

    #[test]
    fn test_clustering_coefficient() {
        let mut g = Graph::new(false);
        g.add_nodes(4);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 2);
        g.add_edge(2, 3);

        let cc = g.clustering_coefficient(0);
        assert!((cc - 1.0).abs() < f64::EPSILON); // triangle 0-1-2
    }
}
