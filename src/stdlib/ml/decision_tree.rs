/// Decision tree classifier with entropy-based splitting.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DecisionTree {
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    criterion: SplitCriterion,
    root: Option<Box<Node>>,
    feature_importance: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum SplitCriterion {
    Gini,
    Entropy,
}

#[derive(Debug, Clone)]
struct Node {
    feature: Option<usize>,
    threshold: Option<f64>,
    value: Option<f64>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    samples: usize,
    class_distribution: HashMap<i64, usize>,
}

impl DecisionTree {
    pub fn new() -> Self {
        Self {
            max_depth: 10,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: SplitCriterion::Gini,
            root: None,
            feature_importance: Vec::new(),
        }
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn min_samples_split(mut self, min: usize) -> Self {
        self.min_samples_split = min;
        self
    }

    pub fn min_samples_leaf(mut self, min: usize) -> Self {
        self.min_samples_leaf = min;
        self
    }

    pub fn criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        assert!(!x.is_empty() && x.len() == y.len());
        let n_features = x[0].len();
        self.feature_importance = vec![0.0; n_features];

        let indices: Vec<usize> = (0..x.len()).collect();
        self.root = Some(self.build_tree(x, y, &indices, 0));
    }

    fn build_tree(&mut self, x: &[Vec<f64>], y: &[f64], indices: &[usize], depth: usize) -> Box<Node> {
        let class_dist = self.class_distribution(y, indices);
        let majority = self.majority_class(&class_dist);

        // Check stopping conditions
        if depth >= self.max_depth
            || indices.len() < self.min_samples_split
            || self.is_pure(y, indices)
        {
            return Box::new(Node {
                feature: None,
                threshold: None,
                value: Some(majority),
                left: None,
                right: None,
                samples: indices.len(),
                class_distribution: class_dist,
            });
        }

        let (best_feature, best_threshold, best_gain) = self.find_best_split(x, y, indices);

        if best_gain <= 0.0 {
            return Box::new(Node {
                feature: None,
                threshold: None,
                value: Some(majority),
                left: None,
                right: None,
                samples: indices.len(),
                class_distribution: class_dist,
            });
        }

        self.feature_importance[best_feature] += best_gain * indices.len() as f64;

        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices.iter()
            .partition(|&&i| x[i][best_feature] <= best_threshold);

        if left_indices.len() < self.min_samples_leaf || right_indices.len() < self.min_samples_leaf {
            return Box::new(Node {
                feature: None,
                threshold: None,
                value: Some(majority),
                left: None,
                right: None,
                samples: indices.len(),
                class_distribution: class_dist,
            });
        }

        let left = self.build_tree(x, y, &left_indices, depth + 1);
        let right = self.build_tree(x, y, &right_indices, depth + 1);

        Box::new(Node {
            feature: Some(best_feature),
            threshold: Some(best_threshold),
            value: None,
            left: Some(left),
            right: Some(right),
            samples: indices.len(),
            class_distribution: class_dist,
        })
    }

    fn find_best_split(&self, x: &[Vec<f64>], y: &[f64], indices: &[usize]) -> (usize, f64, f64) {
        let n_features = x[0].len();
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;

        let parent_impurity = self.impurity(y, indices);

        for feature in 0..n_features {
            let mut values: Vec<f64> = indices.iter().map(|&i| x[i][feature]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values.dedup();

            for i in 0..values.len().saturating_sub(1) {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                let (left, right): (Vec<usize>, Vec<usize>) = indices.iter()
                    .partition(|&&idx| x[idx][feature] <= threshold);

                if left.is_empty() || right.is_empty() {
                    continue;
                }

                let left_impurity = self.impurity(y, &left);
                let right_impurity = self.impurity(y, &right);

                let left_weight = left.len() as f64 / indices.len() as f64;
                let right_weight = right.len() as f64 / indices.len() as f64;

                let gain = parent_impurity - left_weight * left_impurity - right_weight * right_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature;
                    best_threshold = threshold;
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    fn impurity(&self, y: &[f64], indices: &[usize]) -> f64 {
        match self.criterion {
            SplitCriterion::Gini => self.gini(y, indices),
            SplitCriterion::Entropy => self.entropy(y, indices),
        }
    }

    fn gini(&self, y: &[f64], indices: &[usize]) -> f64 {
        let dist = self.class_distribution(y, indices);
        let total = indices.len() as f64;
        1.0 - dist.values().map(|&count| (count as f64 / total).powi(2)).sum::<f64>()
    }

    fn entropy(&self, y: &[f64], indices: &[usize]) -> f64 {
        let dist = self.class_distribution(y, indices);
        let total = indices.len() as f64;
        -dist.values()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total;
                p * p.log2()
            })
            .sum::<f64>()
    }

    fn class_distribution(&self, y: &[f64], indices: &[usize]) -> HashMap<i64, usize> {
        let mut dist = HashMap::new();
        for &i in indices {
            *dist.entry(y[i] as i64).or_insert(0) += 1;
        }
        dist
    }

    fn majority_class(&self, dist: &HashMap<i64, usize>) -> f64 {
        *dist.iter().max_by_key(|&(_, count)| count).unwrap().0 as f64
    }

    fn is_pure(&self, y: &[f64], indices: &[usize]) -> bool {
        let first = y[indices[0]] as i64;
        indices.iter().all(|&i| y[i] as i64 == first)
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|row| self.predict_single(row)).collect()
    }

    fn predict_single(&self, x: &[f64]) -> f64 {
        match &self.root {
            Some(node) => self.traverse(node, x),
            None => 0.0,
        }
    }

    fn traverse(&self, node: &Node, x: &[f64]) -> f64 {
        if let Some(value) = node.value {
            return value;
        }

        let feature = node.feature.unwrap();
        let threshold = node.threshold.unwrap();

        if x[feature] <= threshold {
            self.traverse(node.left.as_ref().unwrap(), x)
        } else {
            self.traverse(node.right.as_ref().unwrap(), x)
        }
    }

    pub fn accuracy(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict(x);
        let correct = predictions.iter().zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-10)
            .count();
        correct as f64 / y.len() as f64
    }

    pub fn feature_importance(&self) -> Vec<f64> {
        let total: f64 = self.feature_importance.iter().sum();
        if total == 0.0 {
            return self.feature_importance.clone();
        }
        self.feature_importance.iter().map(|f| f / total).collect()
    }

    pub fn depth(&self) -> usize {
        self.node_depth(self.root.as_ref().unwrap())
    }

    fn node_depth(&self, node: &Node) -> usize {
        match (&node.left, &node.right) {
            (Some(left), Some(right)) => 1 + self.node_depth(left).max(self.node_depth(right)),
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_tree_xor() {
        let x = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        let y = vec![0.0, 1.0, 1.0, 0.0];

        let mut tree = DecisionTree::new().max_depth(5);
        tree.fit(&x, &y);

        let acc = tree.accuracy(&x, &y);
        assert_eq!(acc, 1.0);
    }

    #[test]
    fn test_decision_tree_simple() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![0.0, 0.0, 1.0, 1.0, 1.0];

        let mut tree = DecisionTree::new();
        tree.fit(&x, &y);

        let acc = tree.accuracy(&x, &y);
        assert!(acc >= 0.8);
    }

    #[test]
    fn test_feature_importance() {
        let x = vec![
            vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 0.0],
            vec![0.0, 1.0], vec![0.0, 2.0], vec![0.0, 3.0],
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut tree = DecisionTree::new();
        tree.fit(&x, &y);

        let importance = tree.feature_importance();
        assert!(importance[1] > 0.0); // Second feature should be important
    }

    #[test]
    fn test_gini_vs_entropy() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![0.0, 0.0, 1.0, 1.0];

        let mut gini_tree = DecisionTree::new().criterion(SplitCriterion::Gini);
        gini_tree.fit(&x, &y);

        let mut entropy_tree = DecisionTree::new().criterion(SplitCriterion::Entropy);
        entropy_tree.fit(&x, &y);

        // Both should work
        assert!(gini_tree.accuracy(&x, &y) >= 0.5);
        assert!(entropy_tree.accuracy(&x, &y) >= 0.5);
    }
}
