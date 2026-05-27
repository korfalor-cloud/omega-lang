/// Histogram metric for tracking distributions.

#[derive(Debug)]
pub struct Histogram {
    name: String,
    buckets: Vec<Bucket>,
    count: u64,
    sum: f64,
    description: String,
}

#[derive(Debug, Clone)]
struct Bucket {
    upper_bound: f64,
    count: u64,
}

impl Histogram {
    pub fn new(name: &str, buckets: &[f64]) -> Self {
        let mut bucket_list: Vec<Bucket> = buckets.iter()
            .map(|&b| Bucket { upper_bound: b, count: 0 })
            .collect();
        bucket_list.sort_by(|a, b| a.upper_bound.partial_cmp(&b.upper_bound).unwrap());

        Self {
            name: name.to_string(),
            buckets: bucket_list,
            count: 0,
            sum: 0.0,
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn observe(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;

        for bucket in &mut self.buckets {
            if value <= bucket.upper_bound {
                bucket.count += 1;
            }
        }
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn sum(&self) -> f64 {
        self.sum
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    pub fn bucket_counts(&self) -> Vec<(f64, u64)> {
        self.buckets.iter().map(|b| (b.upper_bound, b.count)).collect()
    }

    pub fn percentile(&self, p: f64) -> f64 {
        let target = (self.count as f64 * p / 100.0) as u64;
        let mut cumulative = 0;

        for bucket in &self.buckets {
            cumulative += bucket.count;
            if cumulative >= target {
                return bucket.upper_bound;
            }
        }

        self.buckets.last().map_or(0.0, |b| b.upper_bound)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn reset(&mut self) {
        for bucket in &mut self.buckets {
            bucket.count = 0;
        }
        self.count = 0;
        self.sum = 0.0;
    }
}

/// Summary for tracking quantiles
#[derive(Debug)]
pub struct Summary {
    name: String,
    values: Vec<f64>,
    max_size: usize,
    count: u64,
    sum: f64,
}

impl Summary {
    pub fn new(name: &str, max_size: usize) -> Self {
        Self {
            name: name.to_string(),
            values: Vec::new(),
            max_size,
            count: 0,
            sum: 0.0,
        }
    }

    pub fn observe(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;

        if self.values.len() < self.max_size {
            self.values.push(value);
        } else {
            // Reservoir sampling
            let idx = (self.count as usize) % self.max_size;
            self.values[idx] = value;
        }
    }

    pub fn quantile(&mut self, q: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }

        self.values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((q / 100.0) * (self.values.len() - 1) as f64) as usize;
        self.values[idx]
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn sum(&self) -> f64 {
        self.sum
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum / self.count as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram() {
        let mut hist = Histogram::new("latency", &[10.0, 50.0, 100.0, 500.0]);
        hist.observe(25.0);
        hist.observe(75.0);
        hist.observe(150.0);

        assert_eq!(hist.count(), 3);
        assert_eq!(hist.mean(), 83.33333333333333);
    }

    #[test]
    fn test_histogram_percentiles() {
        let mut hist = Histogram::new("test", &[10.0, 20.0, 30.0, 40.0, 50.0]);
        for i in 1..=100 {
            hist.observe(i as f64);
        }

        let p50 = hist.percentile(50.0);
        let p99 = hist.percentile(99.0);
        assert!(p50 < p99);
    }

    #[test]
    fn test_summary() {
        let mut summary = Summary::new("test", 1000);
        for i in 1..=100 {
            summary.observe(i as f64);
        }

        assert_eq!(summary.count(), 100);
        assert_eq!(summary.mean(), 50.5);
    }
}
