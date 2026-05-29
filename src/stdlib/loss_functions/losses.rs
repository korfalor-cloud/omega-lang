/// Loss functions: focal loss, dice loss, contrastive loss, triplet loss.

/// Focal loss for imbalanced classification.
pub fn focal_loss(logits: &[f64], target: usize, gamma: f64, alpha: f64) -> f64 {
    let probs = softmax(logits);
    let p = probs[target];
    -alpha * (1.0 - p).powf(gamma) * p.max(1e-15).ln()
}

/// Dice loss for segmentation.
pub fn dice_loss(predicted: &[f64], target: &[f64], smooth: f64) -> f64 {
    let intersection: f64 = predicted.iter().zip(target.iter()).map(|(p, t)| p * t).sum();
    let sum: f64 = predicted.iter().map(|p| p * p).sum::<f64>() + target.iter().map(|t| t * t).sum::<f64>();
    1.0 - (2.0 * intersection + smooth) / (sum + smooth)
}

/// IoU loss.
pub fn iou_loss(predicted: &[f64], target: &[f64], smooth: f64) -> f64 {
    let intersection: f64 = predicted.iter().zip(target.iter()).map(|(p, t)| p.min(*t)).sum();
    let union: f64 = predicted.iter().zip(target.iter()).map(|(p, t)| p.max(*t)).sum();
    1.0 - (intersection + smooth) / (union + smooth)
}

/// Contrastive loss.
pub fn contrastive_loss(embedding1: &[f64], embedding2: &[f64], same_class: bool, margin: f64) -> f64 {
    let dist: f64 = embedding1.iter().zip(embedding2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    if same_class {
        dist * dist
    } else {
        (margin - dist).max(0.0).powi(2)
    }
}

/// Triplet loss.
pub fn triplet_loss(anchor: &[f64], positive: &[f64], negative: &[f64], margin: f64) -> f64 {
    let pos_dist: f64 = anchor.iter().zip(positive.iter()).map(|(a, p)| (a - p).powi(2)).sum::<f64>().sqrt();
    let neg_dist: f64 = anchor.iter().zip(negative.iter()).map(|(a, n)| (a - n).powi(2)).sum::<f64>().sqrt();
    (pos_dist - neg_dist + margin).max(0.0)
}

/// Center loss.
pub fn center_loss(embedding: &[f64], center: &[f64]) -> f64 {
    embedding.iter().zip(center.iter()).map(|(e, c)| (e - c).powi(2)).sum::<f64>() / 2.0
}

/// Cosine embedding loss.
pub fn cosine_embedding_loss(embedding1: &[f64], embedding2: &[f64], same_class: bool, margin: f64) -> f64 {
    let dot: f64 = embedding1.iter().zip(embedding2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f64 = embedding1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2: f64 = embedding2.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cosine = dot / (norm1 * norm2).max(1e-10);

    if same_class {
        1.0 - cosine
    } else {
        (cosine - margin).max(0.0)
    }
}

/// Hinge loss.
pub fn hinge_loss(scores: &[f64], target: usize) -> f64 {
    let target_score = scores[target];
    let mut max_other = f64::NEG_INFINITY;
    for (i, &s) in scores.iter().enumerate() {
        if i != target {
            max_other = max_other.max(s);
        }
    }
    (max_other - target_score + 1.0).max(0.0)
}

/// Squared hinge loss.
pub fn squared_hinge_loss(scores: &[f64], target: usize) -> f64 {
    let h = hinge_loss(scores, target);
    h * h
}

/// KL divergence loss.
pub fn kl_divergence_loss(predicted: &[f64], target: &[f64]) -> f64 {
    target.iter().zip(predicted.iter())
        .filter(|(t, _)| **t > 0.0)
        .map(|(t, p)| t * (t / p.max(1e-15)).ln())
        .sum()
}

/// Jensen-Shannon divergence loss.
pub fn js_divergence_loss(predicted: &[f64], target: &[f64]) -> f64 {
    let m: Vec<f64> = predicted.iter().zip(target.iter()).map(|(p, t)| (p + t) / 2.0).collect();
    (kl_divergence_loss(predicted, &m) + kl_divergence_loss(target, &m)) / 2.0
/// Loss functions: focal loss, dice loss, contrastive loss, triplet loss.

/// Focal loss for imbalanced classification.
pub fn focal_loss(logits: &[f64], target: usize, gamma: f64, alpha: f64) -> f64 {
    let probs = softmax(logits);
    let p = probs[target];
    -alpha * (1.0 - p).powf(gamma) * p.max(1e-15).ln()
}

/// Dice loss for segmentation.
pub fn dice_loss(predicted: &[f64], target: &[f64], smooth: f64) -> f64 {
    let intersection: f64 = predicted.iter().zip(target.iter()).map(|(p, t)| p * t).sum();
    let sum: f64 = predicted.iter().map(|p| p * p).sum::<f64>() + target.iter().map(|t| t * t).sum::<f64>();
    1.0 - (2.0 * intersection + smooth) / (sum + smooth)
}

/// IoU loss.
pub fn iou_loss(predicted: &[f64], target: &[f64], smooth: f64) -> f64 {
    let intersection: f64 = predicted.iter().zip(target.iter()).map(|(p, t)| p.min(*t)).sum();
    let union: f64 = predicted.iter().zip(target.iter()).map(|(p, t)| p.max(*t)).sum();
    1.0 - (intersection + smooth) / (union + smooth)
}

/// Contrastive loss.
pub fn contrastive_loss(embedding1: &[f64], embedding2: &[f64], same_class: bool, margin: f64) -> f64 {
    let dist: f64 = embedding1.iter().zip(embedding2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    if same_class {
        dist * dist
    } else {
        (margin - dist).max(0.0).powi(2)
    }
}

/// Triplet loss.
pub fn triplet_loss(anchor: &[f64], positive: &[f64], negative: &[f64], margin: f64) -> f64 {
    let pos_dist: f64 = anchor.iter().zip(positive.iter()).map(|(a, p)| (a - p).powi(2)).sum::<f64>().sqrt();
    let neg_dist: f64 = anchor.iter().zip(negative.iter()).map(|(a, n)| (a - n).powi(2)).sum::<f64>().sqrt();
    (pos_dist - neg_dist + margin).max(0.0)
}

/// Center loss.
pub fn center_loss(embedding: &[f64], center: &[f64]) -> f64 {
    embedding.iter().zip(center.iter()).map(|(e, c)| (e - c).powi(2)).sum::<f64>() / 2.0
}

/// Cosine embedding loss.
pub fn cosine_embedding_loss(embedding1: &[f64], embedding2: &[f64], same_class: bool, margin: f64) -> f64 {
    let dot: f64 = embedding1.iter().zip(embedding2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f64 = embedding1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2: f64 = embedding2.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cosine = dot / (norm1 * norm2).max(1e-10);

    if same_class {
        1.0 - cosine
    } else {
        (cosine - margin).max(0.0)
    }
}

/// Hinge loss.
pub fn hinge_loss(scores: &[f64], target: usize) -> f64 {
    let target_score = scores[target];
    let mut max_other = f64::NEG_INFINITY;
    for (i, &s) in scores.iter().enumerate() {
        if i != target {
            max_other = max_other.max(s);
        }
    }
    (max_other - target_score + 1.0).max(0.0)
}

/// Squared hinge loss.
pub fn squared_hinge_loss(scores: &[f64], target: usize) -> f64 {
    let h = hinge_loss(scores, target);
    h * h
}

/// KL divergence loss.
pub fn kl_divergence_loss(predicted: &[f64], target: &[f64]) -> f64 {
    target.iter().zip(predicted.iter())
        .filter(|(t, _)| **t > 0.0)
        .map(|(t, p)| t * (t / p.max(1e-15)).ln())
        .sum()
}

/// Jensen-Shannon divergence loss.
pub fn js_divergence_loss(predicted: &[f64], target: &[f64]) -> f64 {
    let m: Vec<f64> = predicted.iter().zip(target.iter()).map(|(p, t)| (p + t) / 2.0).collect();
    (kl_divergence_loss(predicted, &m) + kl_divergence_loss(target, &m)) / 2.0
}

/// Huber loss.
pub fn huber_loss(predicted: f64, target: f64, delta: f64) -> f64 {
    let diff = (predicted - target).abs();
    if diff <= delta {
        0.5 * diff * diff
    } else {
        delta * (diff - 0.5 * delta)
    }
}

/// Log-cosh loss.
pub fn log_cosh_loss(predicted: f64, target: f64) -> f64 {
    let diff = predicted - target;
    (diff.cosh()).ln()
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_focal_loss() {
        let logits = vec![2.0, 1.0, 0.1];
        let loss = focal_loss(&logits, 0, 2.0, 0.25);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_triplet_loss() {
        let anchor = vec![1.0, 0.0];
        let positive = vec![0.9, 0.1];
        let negative = vec![0.0, 1.0];
        let loss = triplet_loss(&anchor, &positive, &negative, 1.0);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_dice_loss() {
        let pred = vec![0.9, 0.1, 0.8];
        let target = vec![1.0, 0.0, 1.0];
        let loss = dice_loss(&pred, &target, 1.0);
        assert!(loss >= 0.0 && loss <= 1.0);
    }
}
