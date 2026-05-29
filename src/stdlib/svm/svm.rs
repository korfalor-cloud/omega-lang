/// Support Vector Machine classifier with multiple kernels.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Kernel {
    Linear,
    Polynomial { degree: usize, coef0: f64 },
    RBF { gamma: f64 },
    Sigmoid { gamma: f64, coef0: f64 },
}

impl Kernel {
    pub fn evaluate(&self, x1: &[f64], x2: &[f64]) -> f64 {
        match self {
            Kernel::Linear => dot(x1, x2),
            Kernel::Polynomial { degree, coef0 } => {
                (dot(x1, x2) + coef0).powi(*degree as i32)
            }
            Kernel::RBF { gamma } => {
                let dist_sq: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                (-gamma * dist_sq).exp()
            }
            Kernel::Sigmoid { gamma, coef0 } => {
                (gamma * dot(x1, x2) + coef0).tanh()
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Support Vector Machine using simplified SMO (Sequential Minimal Optimization).
#[derive(Debug)]
pub struct SVM {
    kernel: Kernel,
    c: f64,
    tolerance: f64,
    max_iterations: usize,
    alphas: Vec<f64>,
    b: f64,
    support_vectors: Vec<Vec<f64>>,
    support_labels: Vec<f64>,
    support_alphas: Vec<f64>,
    trained: bool,
}

impl SVM {
    pub fn new(kernel: Kernel, c: f64) -> Self {
        Self {
            kernel,
            c,
            tolerance: 1e-3,
            max_iterations: 1000,
            alphas: Vec::new(),
            b: 0.0,
            support_vectors: Vec::new(),
            support_labels: Vec::new(),
            support_alphas: Vec::new(),
            trained: false,
        }
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self { self.tolerance = tol; self }
    pub fn with_max_iterations(mut self, max: usize) -> Self { self.max_iterations = max; self }

    /// Compute the kernel matrix for all training samples.
    fn compute_kernel_matrix(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut km = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let k = self.kernel.evaluate(&x[i], &x[j]);
                km[i][j] = k;
                km[j][i] = k;
            }
        }
        km
    }

    /// Train the SVM using simplified SMO.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), String> {
        if x.len() != y.len() {
            return Err("Feature and label counts must match".to_string());
        }
        if x.is_empty() {
            return Err("Training data is empty".to_string());
        }

        let n = x.len();
        self.alphas = vec![0.0; n];
        self.b = 0.0;

        let km = self.compute_kernel_matrix(x);

        let mut num_changed = 0;
        let mut examine_all = true;
        let mut iteration = 0;

        while (num_changed > 0 || examine_all) && iteration < self.max_iterations {
            num_changed = 0;
            iteration += 1;

            if examine_all {
                for i in 0..n {
                    num_changed += self.examine_example(i, x, y, &km);
                }
            } else {
                for i in 0..n {
                    if self.alphas[i] > 0.0 && self.alphas[i] < self.c {
                        num_changed += self.examine_example(i, x, y, &km);
                    }
                }
            }

            if examine_all {
                examine_all = false;
            } else if num_changed == 0 {
                examine_all = true;
            }
        }

        // Extract support vectors
        self.support_vectors.clear();
        self.support_labels.clear();
        self.support_alphas.clear();

        for i in 0..n {
            if self.alphas[i] > 1e-10 {
                self.support_vectors.push(x[i].clone());
                self.support_labels.push(y[i]);
                self.support_alphas.push(self.alphas[i]);
            }
        }

        self.trained = true;
        Ok(())
    }

    fn examine_example(&mut self, i2: usize, x: &[Vec<f64>], y: &[f64], km: &[Vec<f64>]) -> usize {
        let alpha2 = self.alphas[i2];
        let y2 = y[i2];
        let e2 = self.compute_error(i2, x, y, km);

        let r2 = e2 * y2;
        if (r2 < -self.tolerance && alpha2 < self.c) || (r2 > self.tolerance && alpha2 > 0.0) {
            // Try to find i1 using heuristic
            let i1 = self.select_second_alpha(i2, e2, x, y, km);
            if self.take_step(i1, i2, x, y, km) {
                return 1;
            }

            // Try random non-bound alphas
            let mut indices: Vec<usize> = (0..x.len()).filter(|&j| j != i2 && self.alphas[j] > 0.0 && self.alphas[j] < self.c).collect();
            let seed = iteration_seed(i2);
            shuffle_with_seed(&mut indices, seed);
            for &i1 in &indices {
                if self.take_step(i1, i2, x, y, km) {
                    return 1;
                }
            }

            // Try all examples
            let mut all: Vec<usize> = (0..x.len()).filter(|&j| j != i2).collect();
            shuffle_with_seed(&mut all, seed + 1);
            for &i1 in &all {
                if self.take_step(i1, i2, x, y, km) {
                    return 1;
                }
            }
        }

        0
    }

    fn take_step(&mut self, i1: usize, i2: usize, x: &[Vec<f64>], y: &[f64], km: &[Vec<f64>]) -> bool {
        if i1 == i2 {
            return false;
        }

        let alpha1 = self.alphas[i1];
        let alpha2 = self.alphas[i2];
        let y1 = y[i1];
        let y2 = y[i2];

        let e1 = self.compute_error(i1, x, y, km);
        let e2 = self.compute_error(i2, x, y, km);

        let s = y1 * y2;

        let (l, h) = if y1 != y2 {
            let l = (0.0_f64).max(alpha2 - alpha1);
            let h = (self.c).min(self.c + alpha2 - alpha1);
            (l, h)
        } else {
            let l = (0.0_f64).max(alpha2 + alpha1 - self.c);
            let h = (self.c).min(alpha2 + alpha1);
            (l, h)
        };

        if (l - h).abs() < 1e-10 {
            return false;
        }

        let k11 = km[i1][i1];
        let k12 = km[i1][i2];
        let k22 = km[i2][i2];
        let eta = k11 + k22 - 2.0 * k12;

        let mut a2;
        if eta > 0.0 {
            a2 = alpha2 + y2 * (e1 - e2) / eta;
            if a2 < l { a2 = l; }
            else if a2 > h { a2 = h; }
        } else {
            // Objective function at a2=l and a2=h
            let f1 = y1 * (e1 + self.b) - alpha1 * k11 - s * alpha2 * k12;
            let f2 = y2 * (e2 + self.b) - s * alpha1 * k12 - alpha2 * k22;
            let l1 = alpha1 + s * (alpha2 - l);
            let h1 = alpha1 + s * (alpha2 - h);
            let mut lobj = l1 * f1 + l * f2 + 0.5 * l1 * l1 * k11 + 0.5 * l * l * k22 + s * l * l1 * k12;
            let mut hobj = h1 * f1 + h * f2 + 0.5 * h1 * h1 * k11 + 0.5 * h * h * k22 + s * h * h1 * k12;
            if lobj < hobj - 1e-10 { a2 = l; }
            else if lobj > hobj + 1e-10 { a2 = h; }
            else { a2 = alpha2; }
        }

        if (a2 - alpha2).abs() < 1e-10 * (a2 + alpha2 + 1e-10) {
            return false;
        }

        let a1 = alpha1 + s * (alpha2 - a2);

        // Update threshold
        let b1 = e1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self.b;
        let b2 = e2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self.b;

        self.b = if a1 > 0.0 && a1 < self.c {
            b1
        } else if a2 > 0.0 && a2 < self.c {
            b2
        } else {
            (b1 + b2) / 2.0
        };

        self.alphas[i1] = a1;
        self.alphas[i2] = a2;
        true
    }

    fn compute_error(&self, i: usize, x: &[Vec<f64>], y: &[f64], km: &[Vec<f64>]) -> f64 {
        let mut sum = 0.0;
        for j in 0..x.len() {
            if self.alphas[j] > 0.0 {
                sum += self.alphas[j] * y[j] * km[j][i];
            }
        }
        sum + self.b - y[i]
    }

    fn select_second_alpha(&self, i2: usize, e2: f64, x: &[Vec<f64>], y: &[f64], km: &[Vec<f64>]) -> usize {
        let mut best_i = 0;
        let mut best_delta = 0.0;
        let mut found = false;

        for i in 0..x.len() {
            if self.alphas[i] > 0.0 && self.alphas[i] < self.c {
                let e1 = self.compute_error(i, x, y, km);
                let delta = (e1 - e2).abs();
                if !found || delta > best_delta {
                    best_i = i;
                    best_delta = delta;
                    found = true;
                }
            }
        }

        if found {
            best_i
        } else {
            // Random
            let mut indices: Vec<usize> = (0..x.len()).filter(|&j| j != i2).collect();
            shuffle_with_seed(&mut indices, iteration_seed(i2));
            indices[0]
        }
    }

    /// Predict class label for a single sample.
    pub fn predict(&self, x: &[f64]) -> f64 {
        if !self.trained {
            return 0.0;
        }
        let mut sum = 0.0;
        for (sv, (&label, &alpha)) in self.support_vectors.iter()
            .zip(self.support_labels.iter().zip(self.support_alphas.iter()))
        {
            sum += alpha * label * self.kernel.evaluate(x, sv);
        }
        if sum + self.b >= 0.0 { 1.0 } else { -1.0 }
    }

    /// Predict class labels for multiple samples.
    pub fn predict_batch(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|sample| self.predict(sample)).collect()
    }

    /// Compute accuracy on test data.
    pub fn accuracy(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict_batch(x);
        let correct = predictions.iter().zip(y.iter()).filter(|(p, y)| **p == **y).count();
        correct as f64 / y.len() as f64
    }

    /// Number of support vectors.
    pub fn support_vector_count(&self) -> usize {
        self.support_vectors.len()
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }
}

fn iteration_seed(i: usize) -> u64 {
    (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

fn shuffle_with_seed(v: &mut [usize], seed: u64) {
    let mut state = seed;
    for i in (1..v.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        v.swap(i, j);
    }
}

/// One-vs-Rest multi-class SVM.
pub struct MultiClassSVM {
    classifiers: Vec<SVM>,
    classes: Vec<f64>,
}

impl MultiClassSVM {
    pub fn new(kernel: Kernel, c: f64) -> Self {
        Self {
            classifiers: Vec::new(),
            classes: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), String> {
        self.classes = unique_values(y);
        self.classifiers.clear();

        for &class in &self.classes {
            let binary_y: Vec<f64> = y.iter().map(|&yi| if yi == class { 1.0 } else { -1.0 }).collect();
            let mut svm = SVM::new(Kernel::RBF { gamma: 0.5 }, 1.0);
            svm.fit(x, &binary_y)?;
            self.classifiers.push(svm);
        }

        Ok(())
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let mut best_class = self.classes[0];
        let mut best_score = f64::NEG_INFINITY;

        for (i, svm) in self.classifiers.iter().enumerate() {
            let mut sum = 0.0;
            for (sv, (&label, &alpha)) in svm.support_vectors.iter()
                .zip(svm.support_labels.iter().zip(svm.support_alphas.iter()))
            {
                sum += alpha * label * svm.kernel.evaluate(x, sv);
            }
            let score = sum + svm.b;
            if score > best_score {
                best_score = score;
                best_class = self.classes[i];
            }
        }

        best_class
    }

    pub fn predict_batch(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|sample| self.predict(sample)).collect()
    }

    pub fn accuracy(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        let predictions = self.predict_batch(x);
        let correct = predictions.iter().zip(y.iter()).filter(|(p, y)| **p == **y).count();
        correct as f64 / y.len() as f64
    }
}

fn unique_values(v: &[f64]) -> Vec<f64> {
    let mut unique: Vec<f64> = v.to_vec();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique.dedup_by(|a, b| (a - b).abs() < 1e-10);
    unique
}

/// Compute confusion matrix for binary classification.
pub fn confusion_matrix(predicted: &[f64], actual: &[f64]) -> (usize, usize, usize, usize) {
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_ = 0;

    for (p, a) in predicted.iter().zip(actual.iter()) {
        match (*p > 0.0, *a > 0.0) {
            (true, true) => tp += 1,
            (false, false) => tn += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
        }
    }

    (tp, tn, fp, fn_)
}

/// Precision, recall, F1 score.
pub fn classification_metrics(predicted: &[f64], actual: &[f64]) -> (f64, f64, f64) {
    let (tp, _, fp, fn_) = confusion_matrix(predicted, actual);
    let precision = if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 };
    let recall = if tp + fn_ == 0 { 0.0 } else { tp as f64 / (tp + fn_) as f64 };
    let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
    (precision, recall, f1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svm_linear() {
        let x = vec![
            vec![1.0, 1.0], vec![2.0, 2.0], vec![2.0, 1.0],
            vec![-1.0, -1.0], vec![-2.0, -2.0], vec![-2.0, -1.0],
        ];
        let y = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

        let mut svm = SVM::new(Kernel::Linear, 1.0);
        svm.fit(&x, &y).unwrap();

        assert_eq!(svm.predict(&[3.0, 3.0]), 1.0);
        assert_eq!(svm.predict(&[-3.0, -3.0]), -1.0);
    }

    #[test]
    fn test_svm_rbf() {
        let x = vec![
            vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0],
            vec![5.0, 5.0], vec![6.0, 5.0], vec![5.0, 6.0],
        ];
        let y = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

        let mut svm = SVM::new(Kernel::RBF { gamma: 0.5 }, 1.0);
        svm.fit(&x, &y).unwrap();

        assert_eq!(svm.predict(&[0.5, 0.5]), 1.0);
        assert_eq!(svm.predict(&[5.5, 5.5]), -1.0);
    }

    #[test]
    fn test_kernel_evaluations() {
        let x1 = vec![1.0, 2.0, 3.0];
        let x2 = vec![4.0, 5.0, 6.0];

        let linear = Kernel::Linear.evaluate(&x1, &x2);
        assert!((linear - 32.0).abs() < 1e-10);

        let rbf = Kernel::RBF { gamma: 0.1 }.evaluate(&x1, &x2);
        assert!(rbf > 0.0 && rbf < 1.0);
    }

    #[test]
    fn test_confusion_matrix() {
        let predicted = vec![1.0, 1.0, -1.0, -1.0];
        let actual = vec![1.0, -1.0, -1.0, 1.0];
        let (tp, tn, fp, fn_) = confusion_matrix(&predicted, &actual);
        assert_eq!(tp, 1);
        assert_eq!(tn, 1);
        assert_eq!(fp, 1);
        assert_eq!(fn_, 1);
    }
}
