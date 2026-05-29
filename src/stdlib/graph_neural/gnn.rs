/// Graph Neural Networks: message passing, graph convolution, graph attention.

/// Graph representation.
#[derive(Clone)]
pub struct Graph {
    pub n_nodes: usize,
    pub edges: Vec<(usize, usize)>,
    pub node_features: Vec<Vec<f64>>,
    pub edge_features: Vec<Vec<f64>>,
}

impl Graph {
    pub fn new(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            edges: Vec::new(),
            node_features: Vec::new(),
            edge_features: Vec::new(),
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.push((from, to));
    }

    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.edges.iter()
            .filter(|&&(f, _)| f == node)
            .map(|&(_, t)| t)
            .collect()
    }

    pub fn adjacency_matrix(&self) -> Vec<Vec<f64>> {
        let mut adj = vec![vec![0.0; self.n_nodes]; self.n_nodes];
        for &(from, to) in &self.edges {
            adj[from][to] = 1.0;
            adj[to][from] = 1.0; // Undirected
        }
        adj
    }

    pub fn degree_matrix(&self) -> Vec<Vec<f64>> {
        let adj = self.adjacency_matrix();
        let mut deg = vec![vec![0.0; self.n_nodes]; self.n_nodes];
        for i in 0..self.n_nodes {
            deg[i][i] = adj[i].iter().sum();
        }
        deg
    }
}

/// Message Passing Neural Network layer.
pub struct MPNNLayer {
    pub node_dim: usize,
    pub edge_dim: usize,
    pub hidden_dim: usize,
    pub message_weights: Vec<Vec<f64>>,
    pub update_weights: Vec<Vec<f64>>,
    pub edge_weights: Vec<Vec<f64>>,
}

impl MPNNLayer {
    pub fn new(node_dim: usize, edge_dim: usize, hidden_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale_m = (2.0 / (node_dim * 2 + edge_dim) as f64).sqrt();
        let scale_u = (2.0 / (node_dim + hidden_dim) as f64).sqrt();

        Self {
            node_dim, edge_dim, hidden_dim,
            message_weights: (0..hidden_dim).map(|_| (0..node_dim * 2 + edge_dim).map(|_| rand(scale_m)).collect()).collect(),
            update_weights: (0..node_dim).map(|_| (0..node_dim + hidden_dim).map(|_| rand(scale_u)).collect()).collect(),
            edge_weights: (0..hidden_dim).map(|_| (0..edge_dim).map(|_| rand(scale_m)).collect()).collect(),
        }
    }

    pub fn forward(&self, graph: &Graph) -> Vec<Vec<f64>> {
        let n = graph.n_nodes;
        let mut messages = vec![vec![0.0; self.hidden_dim]; n];

        // Message computation
        for &(from, to) in &graph.edges {
            let edge_feat = graph.edge_features.get(from * n + to).cloned().unwrap_or(vec![0.0; self.edge_dim]);
            let mut msg_input = graph.node_features[from].clone();
            msg_input.extend_from_slice(&graph.node_features[to]);
            msg_input.extend_from_slice(&edge_feat);

            let msg: Vec<f64> = self.message_weights.iter().map(|w| {
                let sum: f64 = w.iter().zip(msg_input.iter()).map(|(wi, xi)| wi * xi).sum();
                sum.tanh()
            }).collect();

            for i in 0..self.hidden_dim {
                messages[to][i] += msg[i];
            }
        }

        // Update
        let mut new_features = Vec::new();
        for i in 0..n {
            let mut update_input = graph.node_features[i].clone();
            update_input.extend_from_slice(&messages[i]);

            let new_feat: Vec<f64> = self.update_weights.iter().map(|w| {
                let sum: f64 = w.iter().zip(update_input.iter()).map(|(wi, xi)| wi * xi).sum();
                sum.tanh()
            }).collect();

            new_features.push(new_feat);
        }

        new_features
    }
}

/// Graph Convolutional Network layer.
pub struct GCNLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

impl GCNLayer {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / in_dim as f64).sqrt();
        Self {
            in_dim, out_dim,
            weights: (0..out_dim).map(|_| (0..in_dim).map(|_| rand(scale)).collect()).collect(),
            bias: vec![0.0; out_dim],
        }
    }

    pub fn forward(&self, graph: &Graph) -> Vec<Vec<f64>> {
        let adj = graph.adjacency_matrix();
        let n = graph.n_nodes;

        // Compute D^{-1/2} A D^{-1/2}
        let deg: Vec<f64> = adj.iter().map(|row| row.iter().sum::<f64>()).collect();
        let deg_inv_sqrt: Vec<f64> = deg.iter().map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }).collect();

        let mut norm_adj = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                norm_adj[i][j] = deg_inv_sqrt[i] * adj[i][j] * deg_inv_sqrt[j];
            }
        }

        // A_norm * X * W
        let mut result = vec![vec![0.0; self.out_dim]; n];
        for i in 0..n {
            for j in 0..n {
                if norm_adj[i][j] != 0.0 {
                    for k in 0..self.out_dim {
                        let sum: f64 = self.weights[k].iter().zip(graph.node_features[j].iter())
                            .map(|(w, x)| w * x)
                            .sum();
                        result[i][k] += norm_adj[i][j] * sum;
                    }
                }
            }
            for k in 0..self.out_dim {
                result[i][k] += self.bias[k];
                result[i][k] = result[i][k].max(0.0); // ReLU
            }
        }

        result
    }
}

/// Graph Attention Network layer.
pub struct GATLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub n_heads: usize,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub attention_a: Vec<Vec<f64>>,
    pub leaky_relu_slope: f64,
}

impl GATLayer {
    pub fn new(in_dim: usize, out_dim: usize, n_heads: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / in_dim as f64).sqrt();

        Self {
            in_dim, out_dim, n_heads,
            weights: (0..n_heads).map(|_| {
                (0..out_dim).map(|_| (0..in_dim).map(|_| rand(scale)).collect()).collect()
            }).collect(),
            attention_a: (0..n_heads).map(|_| vec![rand(0.1), rand(0.1)]).collect(),
            leaky_relu_slope: 0.2,
        }
    }

    pub fn forward(&self, graph: &Graph) -> Vec<Vec<f64>> {
        let n = graph.n_nodes;
        let mut all_head_outputs = Vec::new();

        for head in 0..self.n_heads {
            // Linear transformation
            let mut h = vec![vec![0.0; self.out_dim]; n];
            for i in 0..n {
                for j in 0..self.out_dim {
                    h[i][j] = self.weights[head][j].iter().zip(graph.node_features[i].iter())
                        .map(|(w, x)| w * x)
                        .sum();
                }
            }

            // Compute attention coefficients
            let mut attention = vec![vec![0.0; n]; n];
            for &(from, to) in &graph.edges {
                let e = self.leaky_relu(
                    self.attention_a[head][0] * h[from].iter().sum::<f64>() +
                    self.attention_a[head][1] * h[to].iter().sum::<f64>()
                );
                attention[from][to] = e.exp();
            }

            // Softmax over neighbors
            for i in 0..n {
                let sum: f64 = attention[i].iter().sum();
                if sum > 0.0 {
                    for j in 0..n {
                        attention[i][j] /= sum;
                    }
                }
            }

            // Weighted aggregation
            let mut output = vec![vec![0.0; self.out_dim]; n];
            for i in 0..n {
                for j in 0..n {
                    if attention[i][j] > 0.0 {
                        for k in 0..self.out_dim {
                            output[i][k] += attention[i][j] * h[j][k];
                        }
                    }
                }
                // ELU activation
                for k in 0..self.out_dim {
                    output[i][k] = if output[i][k] > 0.0 { output[i][k] } else { (output[i][k]).exp() - 1.0 };
                }
            }

            all_head_outputs.push(output);
        }

        // Concatenate heads
        let mut result = vec![vec![0.0; self.out_dim * self.n_heads]; n];
        for (h, head_output) in all_head_outputs.iter().enumerate() {
            for i in 0..n {
                for j in 0..self.out_dim {
                    result[i][h * self.out_dim + j] = head_output[i][j];
                }
            }
        }

        result
    }

    fn leaky_relu(&self, x: f64) -> f64 {
        if x > 0.0 { x } else { self.leaky_relu_slope * x }
    }
}

/// Graph pooling: min-cut pooling.
pub struct MinCutPool {
    pub in_dim: usize,
    pub n_clusters: usize,
    pub weights: Vec<Vec<f64>>,
}

impl MinCutPool {
    pub fn new(in_dim: usize, n_clusters: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale = (2.0 / in_dim as f64).sqrt();
        Self {
            in_dim, n_clusters,
            weights: (0..n_clusters).map(|_| (0..in_dim).map(|_| rand(scale)).collect()).collect(),
        }
    }

    pub fn compute_assignment(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = features.len();
        let mut s = vec![vec![0.0; self.n_clusters]; n];

        for i in 0..n {
            for j in 0..self.n_clusters {
                s[i][j] = self.weights[j].iter().zip(features[i].iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    .exp();
            }
            // Softmax
            let sum: f64 = s[i].iter().sum();
            for j in 0..self.n_clusters {
                s[i][j] /= sum;
            }
        }

        s
    }

    pub fn pool(&self, graph: &Graph, features: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let s = self.compute_assignment(features);
        let adj = graph.adjacency_matrix();
        let n = graph.n_nodes;

        // Pooled features: S^T * X
        let mut pooled = vec![vec![0.0; features[0].len()]; self.n_clusters];
        for j in 0..self.n_clusters {
            for i in 0..n {
                for k in 0..features[0].len() {
                    pooled[j][k] += s[i][j] * features[i][k];
                }
            }
        }

        // Pooled adjacency: S^T * A * S
        let mut pooled_adj = vec![vec![0.0; self.n_clusters]; self.n_clusters];
        for i in 0..self.n_clusters {
            for j in 0..self.n_clusters {
                for a in 0..n {
                    for b in 0..n {
                        pooled_adj[i][j] += s[a][i] * adj[a][b] * s[b][j];
                    }
                }
            }
        }

        (pooled, pooled_adj)
    }
}

/// Graph-level readout functions.
pub struct GraphReadout;

impl GraphReadout {
    /// Sum pooling.
    pub fn sum(features: &[Vec<f64>]) -> Vec<f64> {
        let dim = features[0].len();
        let mut result = vec![0.0; dim];
        for node in features {
            for (i, &val) in node.iter().enumerate() {
                result[i] += val;
            }
        }
        result
    }

    /// Mean pooling.
    pub fn mean(features: &[Vec<f64>]) -> Vec<f64> {
        let n = features.len() as f64;
        let result = Self::sum(features);
        result.iter().map(|&x| x / n).collect()
    }

    /// Max pooling.
    pub fn max(features: &[Vec<f64>]) -> Vec<f64> {
        let dim = features[0].len();
        let mut result = vec![f64::NEG_INFINITY; dim];
        for node in features {
            for (i, &val) in node.iter().enumerate() {
                result[i] = result[i].max(val);
            }
        }
        result
    }

    /// Sort pooling (sort nodes by feature sum, take top-k).
    pub fn sort_pool(features: &[Vec<f64>], k: usize) -> Vec<f64> {
        let mut scored: Vec<(f64, &Vec<f64>)> = features.iter()
            .map(|f| (f.iter().sum::<f64>(), f))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut result = Vec::new();
        for (_, feat) in scored.iter().take(k) {
            result.extend_from_slice(feat);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcn() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.node_features = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let gcn = GCNLayer::new(2, 4);
        let output = gcn.forward(&graph);
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 4);
    }

    #[test]
    fn test_gat() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.node_features = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let gat = GATLayer::new(2, 4, 2);
        let output = gat.forward(&graph);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_readout() {
        let features = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];

        let sum = GraphReadout::sum(&features);
        assert_eq!(sum, vec![9.0, 12.0]);

        let mean = GraphReadout::mean(&features);
        assert_eq!(mean, vec![3.0, 4.0]);

        let max = GraphReadout::max(&features);
        assert_eq!(max, vec![5.0, 6.0]);
    }
}
