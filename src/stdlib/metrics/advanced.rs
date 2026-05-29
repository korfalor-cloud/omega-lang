/// Advanced ML classification metrics: ROC AUC, PR AUC, F-beta, MCC, Cohen's kappa,
/// confusion matrix utilities, and classification reports.

/// Confusion matrix for multi-class classification.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    matrix: Vec<Vec<usize>>,
    num_classes: usize,
}

impl ConfusionMatrix {
    /// Build a confusion matrix from predicted and true labels.
    pub fn new(y_true: &[usize], y_pred: &[usize], num_classes: usize) -> Self {
        assert_eq!(y_true.len(), y_pred.len(), "label arrays must have equal length");
        let mut matrix = vec![vec![0usize; num_classes]; num_classes];
        for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
            assert!(t < num_classes && p < num_classes, "label out of range");
            matrix[t][p] += 1;
        }
        Self { matrix, num_classes }
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Access raw matrix: rows = true, columns = predicted.
    pub fn as_slice(&self) -> &[Vec<usize>] {
        &self.matrix
    }

    /// True positives for each class.
    pub fn true_positives(&self) -> Vec<usize> {
        (0..self.num_classes).map(|i| self.matrix[i][i]).collect()
    }

    /// False positives for each class.
    pub fn false_positives(&self) -> Vec<usize> {
        (0..self.num_classes)
            .map(|i| (0..self.num_classes).filter(|&j| j != i).map(|j| self.matrix[j][i]).sum())
            .collect()
    }

    /// False negatives for each class.
    pub fn false_negatives(&self) -> Vec<usize> {
        (0..self.num_classes)
            .map(|i| (0..self.num_classes).filter(|&j| j != i).map(|j| self.matrix[i][j]).sum())
            .collect()
    }

    /// True negatives for each class.
    pub fn true_negatives(&self) -> Vec<usize> {
        let total: usize = self.matrix.iter().flatten().sum();
        (0..self.num_classes)
            .map(|i| total - self.matrix[i].iter().sum::<usize>()
                - (0..self.num_classes).map(|j| self.matrix[j][i]).sum::<usize>()
                + self.matrix[i][i])
            .collect()
    }

    /// Overall accuracy.
    pub fn accuracy(&self) -> f64 {
        let correct: usize = self.true_positives().iter().sum();
        let total: usize = self.matrix.iter().flatten().sum();
        if total == 0 { 0.0 } else { correct as f64 / total as f64 }
    }

    /// Per-class precision.
    pub fn precision_per_class(&self) -> Vec<f64> {
        let tp = self.true_positives();
        let fp = self.false_positives();
        (0..self.num_classes)
            .map(|i| if tp[i] + fp[i] == 0 { 0.0 } else { tp[i] as f64 / (tp[i] + fp[i]) as f64 })
            .collect()
    }

    /// Per-class recall.
    pub fn recall_per_class(&self) -> Vec<f64> {
        let tp = self.true_positives();
        let fn_ = self.false_negatives();
        (0..self.num_classes)
            .map(|i| if tp[i] + fn_[i] == 0 { 0.0 } else { tp[i] as f64 / (tp[i] + fn_[i]) as f64 })
            .collect()
    }

    /// Per-class F1 score.
    pub fn f1_per_class(&self) -> Vec<f64> {
        let p = self.precision_per_class();
        let r = self.recall_per_class();
        (0..self.num_classes)
            .map(|i| if p[i] + r[i] == 0.0 { 0.0 } else { 2.0 * p[i] * r[i] / (p[i] + r[i]) })
            .collect()
    }

    /// Support (number of true instances) for each class.
    pub fn support(&self) -> Vec<usize> {
        self.matrix.iter().map(|row| row.iter().sum()).collect()
    }
}

/// Computes F-beta score given precision and recall.
pub fn f_beta_score(precision: f64, recall: f64, beta: f64) -> f64 {
    if precision + recall == 0.0 {
        return 0.0;
    }
    let b2 = beta * beta;
    (1.0 + b2) * precision * recall / (b2 * precision + recall)
}

/// Computes the F1 score (F-beta with beta=1).
pub fn f1_score(precision: f64, recall: f64) -> f64 {
    f_beta_score(precision, recall, 1.0)
}

/// Macro-averaged F1 score across classes.
pub fn macro_f1(cm: &ConfusionMatrix) -> f64 {
    let f1s = cm.f1_per_class();
    f1s.iter().sum::<f64>() / f1s.len() as f64
}

/// Weighted-averaged F1 score across classes.
pub fn weighted_f1(cm: &ConfusionMatrix) -> f64 {
    let f1s = cm.f1_per_class();
    let support = cm.support();
    let total: usize = support.iter().sum();
    if total == 0 { return 0.0; }
    f1s.iter().zip(support.iter()).map(|(&f, &s)| f * s as f64).sum::<f64>() / total as f64
}

/// Matthews Correlation Coefficient (binary case, using confusion matrix entries).
pub fn mcc(tp: usize, fp: usize, fn_: usize, tn: usize) -> f64 {
    let num = (tp as f64) * (tn as f64) - (fp as f64) * (fn_ as f64);
    let den = ((tp as f64 + fp as f64) * (tp as f64 + fn_ as f64)
        * (tn as f64 + fp as f64) * (tn as f64 + fn_ as f64))
        .sqrt();
    if den == 0.0 { 0.0 } else { num / den }
}

/// Multi-class MCC from a confusion matrix.
pub fn multiclass_mcc(cm: &ConfusionMatrix) -> f64 {
    let k = cm.num_classes();
    let m = cm.as_slice();
    let n: usize = m.iter().flatten().sum();
    if n == 0 {
        return 0.0;
    }

    let mut sum_kk = 0.0_f64;
    let mut sum_ck_sq = 0.0_f64;
    let mut sum_kc_sq = 0.0_f64;

    for k_idx in 0..k {
        for c in 0..k {
            sum_kk += m[k_idx][c] as f64 * m[c][k_idx] as f64;
        }
    }

    for c in 0..k {
        let col_sum: usize = (0..k).map(|k_idx| m[k_idx][c]).sum();
        let row_sum: usize = m[c].iter().sum();
        sum_ck_sq += col_sum as f64 * col_sum as f64;
        sum_kc_sq += row_sum as f64 * row_sum as f64;
    }

    let num = sum_kk - (1.0 / (n as f64)) * sum_kc_sq * sum_ck_sq;
    let den = (1.0 - (1.0 / (n as f64)) * sum_kc_sq).sqrt()
        * (1.0 - (1.0 / (n as f64)) * sum_ck_sq).sqrt();

    if den == 0.0 { 0.0 } else { num / den }
}

/// Cohen's Kappa statistic measuring inter-rater agreement.
pub fn cohens_kappa(y_true: &[usize], y_pred: &[usize], num_classes: usize) -> f64 {
    let cm = ConfusionMatrix::new(y_true, y_pred, num_classes);
    let n: usize = cm.as_slice().iter().flatten().sum();
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;

    let observed_agreement = cm.accuracy();
    let row_marginals = cm.support();
    let col_marginals: Vec<usize> = (0..num_classes)
        .map(|j| (0..num_classes).map(|i| cm.as_slice()[i][j]).sum())
        .collect();

    let expected_agreement: f64 = row_marginals.iter().zip(col_marginals.iter())
        .map(|(&r, &c)| (r as f64) * (c as f64))
        .sum::<f64>() / (n_f * n_f);

    if (expected_agreement - 1.0).abs() < 1e-15 {
        1.0
    } else {
        (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    }
}

/// Computes ROC AUC using the trapezoidal rule.
/// `y_score` are predicted probabilities; `y_true` are binary labels (0 or 1).
pub fn roc_auc(y_true: &[usize], y_score: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_score.len());
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }

    let mut pairs: Vec<(f64, usize)> = y_score.iter().zip(y_true.iter())
        .map(|(&s, &t)| (s, t))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = y_true.iter().filter(|&&t| t == 1).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }

    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut prev_score = pairs[0].0 + 1.0; // ensures first point starts at (0,0)
    let mut auc = 0.0_f64;
    let mut prev_fpr = 0.0_f64;
    let mut prev_tpr = 0.0_f64;

    for &(score, label) in &pairs {
        if (score - prev_score).abs() > 1e-15 {
            let fpr = fp / n_neg;
            let tpr = tp / n_pos;
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
            prev_fpr = fpr;
            prev_tpr = tpr;
            prev_score = score;
        }
        if label == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
    }
    // Final point
    let fpr = fp / n_neg;
    let tpr = tp / n_pos;
    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;

    auc
}

/// Computes precision-recall AUC using the trapezoidal rule.
/// Sorts by descending score and tracks precision/recall at each threshold.
pub fn precision_recall_auc(y_true: &[usize], y_score: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_score.len());
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }

    let mut pairs: Vec<(f64, usize)> = y_score.iter().zip(y_true.iter())
        .map(|(&s, &t)| (s, t))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = y_true.iter().filter(|&&t| t == 1).count() as f64;
    if n_pos == 0.0 {
        return 0.0;
    }

    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut auc = 0.0_f64;
    let mut prev_recall = 1.0_f64;

    for &(_score, label) in &pairs {
        if label == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let precision = tp / (tp + fp);
        let recall = tp / n_pos;
        // Integrate with respect to recall (decreasing)
        auc += (prev_recall - recall) * precision;
        prev_recall = recall;
    }

    auc
}

/// Builds a full classification report as a formatted string.
pub fn classification_report(
    cm: &ConfusionMatrix,
    class_names: Option<&[&str]>,
) -> String {
    let precision = cm.precision_per_class();
    let recall = cm.recall_per_class();
    let f1 = cm.f1_per_class();
    let support = cm.support();
    let k = cm.num_classes();

    let mut report = String::new();
    report.push_str(&format!(
        "{:<15} {:>10} {:>10} {:>10} {:>10}\n",
        "", "precision", "recall", "f1-score", "support"
    ));
    report.push_str(&format!("{:-<55}\n", ""));

    let total_support: usize = support.iter().sum();
    let mut weighted_p = 0.0_f64;
    let mut weighted_r = 0.0_f64;
    let mut weighted_f = 0.0_f64;

    for i in 0..k {
        let name = class_names
            .map(|n| n[i].to_string())
            .unwrap_or_else(|| format!("class {}", i));
        report.push_str(&format!(
            "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            name, precision[i], recall[i], f1[i], support[i]
        ));
        if total_support > 0 {
            let w = support[i] as f64;
            weighted_p += precision[i] * w;
            weighted_r += recall[i] * w;
            weighted_f += f1[i] * w;
        }
    }

    report.push_str(&format!("{:-<55}\n", ""));
    let acc = cm.accuracy();

    let macro_p: f64 = precision.iter().sum::<f64>() / k as f64;
    let macro_r: f64 = recall.iter().sum::<f64>() / k as f64;
    let macro_f: f64 = f1.iter().sum::<f64>() / k as f64;

    report.push_str(&format!(
        "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
        "macro avg", macro_p, macro_r, macro_f, total_support
    ));
    if total_support > 0 {
        report.push_str(&format!(
            "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "weighted avg",
            weighted_p / total_support as f64,
            weighted_r / total_support as f64,
            weighted_f / total_support as f64,
            total_support
        ));
    }
    report.push_str(&format!(
        "\naccuracy: {:.4}  ({} samples)",
        acc, total_support
    ));

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix_basic() {
        // 2-class: true = [0,0,1,1], pred = [0,1,0,1]
        let y_true = [0, 0, 1, 1];
        let y_pred = [0, 1, 0, 1];
        let cm = ConfusionMatrix::new(&y_true, &y_pred, 2);
        assert_eq!(cm.as_slice()[0][0], 1); // TN
        assert_eq!(cm.as_slice()[0][1], 1); // FP
        assert_eq!(cm.as_slice()[1][0], 1); // FN
        assert_eq!(cm.as_slice()[1][1], 1); // TP
    }

    #[test]
    fn test_confusion_matrix_perfect() {
        let y_true = [0, 0, 0, 1, 1, 1];
        let y_pred = [0, 0, 0, 1, 1, 1];
        let cm = ConfusionMatrix::new(&y_true, &y_pred, 2);
        assert!((cm.accuracy() - 1.0).abs() < 1e-10);
        let p = cm.precision_per_class();
        assert!((p[0] - 1.0).abs() < 1e-10);
        assert!((p[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_and_f_beta() {
        let p = 0.8;
        let r = 0.6;
        let f1 = f1_score(p, r);
        assert!((f1 - 0.685714).abs() < 1e-4);
        // F2 weights recall higher
        let f2 = f_beta_score(p, r, 2.0);
        assert!(f2 > f1); // recall-weighted should be closer to recall
    }

    #[test]
    fn test_mcc_binary() {
        // Perfect classifier
        assert!((mcc(3, 0, 0, 3) - 1.0).abs() < 1e-10);
        // Random
        let val = mcc(2, 2, 2, 2);
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_worst_case() {
        // All predictions wrong for binary
        assert!((mcc(0, 3, 3, 0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_multiclass_mcc() {
        // 3-class perfect classification
        let y_true = [0, 0, 1, 1, 2, 2];
        let y_pred = [0, 0, 1, 1, 2, 2];
        let cm = ConfusionMatrix::new(&y_true, &y_pred, 3);
        let val = multiclass_mcc(&cm);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cohens_kappa_perfect() {
        let y_true = [0, 0, 1, 1, 2, 2];
        let y_pred = [0, 0, 1, 1, 2, 2];
        let kappa = cohens_kappa(&y_true, &y_pred, 3);
        assert!((kappa - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cohens_kappa_random() {
        // Balanced 2-class with random-looking predictions: kappa should be ~0
        let y_true = [0, 0, 1, 1];
        let y_pred = [0, 1, 0, 1];
        let kappa = cohens_kappa(&y_true, &y_pred, 2);
        assert!((kappa - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_roc_auc_perfect_separation() {
        let y_true = [0, 0, 0, 1, 1, 1];
        let y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        let auc = roc_auc(&y_true, &y_score);
        assert!((auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_roc_auc_inverse() {
        let y_true = [0, 0, 0, 1, 1, 1];
        let y_score = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
        let auc = roc_auc(&y_true, &y_score);
        assert!((auc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_roc_auc_random() {
        let y_true = [0, 1, 0, 1];
        let y_score = [0.5, 0.5, 0.5, 0.5];
        let auc = roc_auc(&y_true, &y_score);
        assert!((auc - 0.5).abs() < 0.15); // near 0.5 for tied scores
    }

    #[test]
    fn test_pr_auc_perfect() {
        let y_true = [1, 1, 1, 0, 0, 0];
        let y_score = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
        let auc = precision_recall_auc(&y_true, &y_score);
        assert!((auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pr_auc_decreasing() {
        let y_true = [1, 0, 1, 0, 1];
        let y_score = [0.9, 0.8, 0.7, 0.6, 0.5];
        let auc = precision_recall_auc(&y_true, &y_score);
        assert!(auc > 0.5); // should be decent since positives score high
    }

    #[test]
    fn test_macro_and_weighted_f1() {
        let y_true = [0, 0, 0, 1, 1, 1];
        let y_pred = [0, 0, 1, 1, 1, 0];
        let cm = ConfusionMatrix::new(&y_true, &y_pred, 2);
        let mf = macro_f1(&cm);
        let wf = weighted_f1(&cm);
        assert!(mf > 0.5);
        // With balanced classes, macro and weighted should be equal
        assert!((mf - wf).abs() < 1e-10);
    }

    #[test]
    fn test_multiclass_report() {
        let y_true = [0, 0, 1, 1, 2, 2];
        let y_pred = [0, 1, 1, 1, 2, 0];
        let cm = ConfusionMatrix::new(&y_true, &y_pred, 3);
        let report = classification_report(&cm, Some(&["cat", "dog", "bird"]));
        assert!(report.contains("precision"));
        assert!(report.contains("cat"));
        assert!(report.contains("dog"));
        assert!(report.contains("bird"));
        assert!(report.contains("macro avg"));
        assert!(report.contains("weighted avg"));
    }

    #[test]
    fn test_empty_inputs() {
        let auc = roc_auc(&[], &[]);
        assert_eq!(auc, 0.0);
        let pr = precision_recall_auc(&[], &[]);
        assert_eq!(pr, 0.0);
    }

    #[test]
    fn test_support_counts() {
        let y_true = [0, 0, 0, 1, 1, 2, 2, 2, 2];
        let y_pred = [0, 0, 1, 1, 1, 2, 2, 0, 2];
        let cm = ConfusionMatrix::new(&y_true, &y_pred, 3);
        assert_eq!(cm.support(), vec![3, 2, 4]);
    }
}
