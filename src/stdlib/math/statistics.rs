use crate::errors::{OmegaError, OmegaResult};

pub fn mean(data: &[f64]) -> OmegaResult<f64> {
    if data.is_empty() {
        return Err(OmegaError::ValueError {
            message: "Cannot compute mean of empty dataset".to_string(),
        });
    }
    Ok(data.iter().sum::<f64>() / data.len() as f64)
}

pub fn median(data: &[f64]) -> OmegaResult<f64> {
    if data.is_empty() {
        return Err(OmegaError::ValueError {
            message: "Cannot compute median of empty dataset".to_string(),
        });
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Ok((sorted[mid - 1] + sorted[mid]) / 2.0)
    } else {
        Ok(sorted[mid])
    }
}

pub fn mode(data: &[f64]) -> OmegaResult<f64> {
    if data.is_empty() {
        return Err(OmegaError::ValueError {
            message: "Cannot compute mode of empty dataset".to_string(),
        });
    }
    let mut counts = std::collections::HashMap::new();
    for &value in data {
        *counts.entry(value.to_bits()).or_insert(0usize) += 1;
    }
    let max_count = counts.values().max().unwrap();
    for (&bits, &count) in &counts {
        if count == *max_count {
            return Ok(f64::from_bits(bits));
        }
    }
    unreachable!()
}

pub fn variance(data: &[f64]) -> OmegaResult<f64> {
    if data.len() < 2 {
        return Err(OmegaError::ValueError {
            message: "Need at least 2 data points for variance".to_string(),
        });
    }
    let m = mean(data)?;
    let sum_sq: f64 = data.iter().map(|x| (x - m).powi(2)).sum();
    Ok(sum_sq / (data.len() - 1) as f64)
}

pub fn population_variance(data: &[f64]) -> OmegaResult<f64> {
    if data.is_empty() {
        return Err(OmegaError::ValueError {
            message: "Cannot compute variance of empty dataset".to_string(),
        });
    }
    let m = mean(data)?;
    let sum_sq: f64 = data.iter().map(|x| (x - m).powi(2)).sum();
    Ok(sum_sq / data.len() as f64)
}

pub fn standard_deviation(data: &[f64]) -> OmegaResult<f64> {
    Ok(variance(data)?.sqrt())
}

pub fn population_standard_deviation(data: &[f64]) -> OmegaResult<f64> {
    Ok(population_variance(data)?.sqrt())
}

pub fn min(data: &[f64]) -> OmegaResult<f64> {
    data.iter().cloned().reduce(f64::min).ok_or_else(|| OmegaError::ValueError {
        message: "Cannot compute min of empty dataset".to_string(),
    })
}

pub fn max(data: &[f64]) -> OmegaResult<f64> {
    data.iter().cloned().reduce(f64::max).ok_or_else(|| OmegaError::ValueError {
        message: "Cannot compute max of empty dataset".to_string(),
    })
}

pub fn range(data: &[f64]) -> OmegaResult<f64> {
    Ok(max(data)? - min(data)?)
}

pub fn sum(data: &[f64]) -> f64 {
    data.iter().sum()
}

pub fn product(data: &[f64]) -> f64 {
    data.iter().product()
}

pub fn percentile(data: &[f64], p: f64) -> OmegaResult<f64> {
    if data.is_empty() {
        return Err(OmegaError::ValueError {
            message: "Cannot compute percentile of empty dataset".to_string(),
        });
    }
    if p < 0.0 || p > 100.0 {
        return Err(OmegaError::ValueError {
            message: "Percentile must be between 0 and 100".to_string(),
        });
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = (p / 100.0 * (sorted.len() - 1) as f64) as usize;
    Ok(sorted[index])
}

pub fn quartiles(data: &[f64]) -> OmegaResult<(f64, f64, f64)> {
    Ok((percentile(data, 25.0)?, median(data)?, percentile(data, 75.0)?))
}

pub fn iqr(data: &[f64]) -> OmegaResult<f64> {
    let (q1, _, q3) = quartiles(data)?;
    Ok(q3 - q1)
}

pub fn covariance(x: &[f64], y: &[f64]) -> OmegaResult<f64> {
    if x.len() != y.len() {
        return Err(OmegaError::ValueError {
            message: "Datasets must have same length".to_string(),
        });
    }
    if x.len() < 2 {
        return Err(OmegaError::ValueError {
            message: "Need at least 2 data points".to_string(),
        });
    }
    let mx = mean(x)?;
    let my = mean(y)?;
    let sum: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| (xi - mx) * (yi - my)).sum();
    Ok(sum / (x.len() - 1) as f64)
}

pub fn correlation(x: &[f64], y: &[f64]) -> OmegaResult<f64> {
    let cov = covariance(x, y)?;
    let sx = standard_deviation(x)?;
    let sy = standard_deviation(y)?;
    if sx == 0.0 || sy == 0.0 {
        return Err(OmegaError::ValueError {
            message: "Cannot compute correlation with zero variance".to_string(),
        });
    }
    Ok(cov / (sx * sy))
}

pub fn linear_regression(x: &[f64], y: &[f64]) -> OmegaResult<(f64, f64)> {
    if x.len() != y.len() {
        return Err(OmegaError::ValueError {
            message: "Datasets must have same length".to_string(),
        });
    }
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    Ok((slope, intercept))
}

pub fn z_score(value: f64, mean: f64, std_dev: f64) -> f64 {
    if std_dev == 0.0 {
        0.0
    } else {
        (value - mean) / std_dev
    }
}

pub fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || window > data.len() {
        return data.to_vec();
    }
    let mut result = Vec::new();
    for i in 0..=data.len() - window {
        let window_mean: f64 = data[i..i + window].iter().sum::<f64>() / window as f64;
        result.push(window_mean);
    }
    result
}

pub fn exponential_moving_average(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut result = vec![data[0]];
    for i in 1..data.len() {
        let ema = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        result.push(ema);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&data).unwrap(), 3.0);
    }

    #[test]
    fn test_median() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(median(&data).unwrap(), 3.0);
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&data2).unwrap(), 2.5);
    }

    #[test]
    fn test_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = variance(&data).unwrap();
        assert!((var - 4.0).abs() < 1e-10);
    }
}
