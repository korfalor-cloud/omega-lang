/// Counter metric for tracking cumulative values.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Counter {
    name: String,
    value: f64,
    labels: HashMap<String, String>,
    description: String,
}

impl Counter {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            value: 0.0,
            labels: HashMap::new(),
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    pub fn inc(&mut self) {
        self.value += 1.0;
    }

    pub fn inc_by(&mut self, amount: f64) {
        self.value += amount;
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }

    pub fn reset(&mut self) {
        self.value = 0.0;
    }
}

/// Rate counter for tracking events per second
#[derive(Debug)]
pub struct RateCounter {
    window_size: u64,
    buckets: Vec<(u64, u64)>,
    current_bucket: usize,
}

impl RateCounter {
    pub fn new(window_size: u64, buckets: usize) -> Self {
        Self {
            window_size,
            buckets: vec![(0, 0); buckets],
            current_bucket: 0,
        }
    }

    pub fn record(&mut self, timestamp: u64, count: u64) {
        let bucket_idx = (timestamp % self.buckets.len() as u64) as usize;
        if bucket_idx != self.current_bucket {
            self.buckets[bucket_idx] = (timestamp, 0);
            self.current_bucket = bucket_idx;
        }
        self.buckets[bucket_idx].1 += count;
    }

    pub fn rate(&self, current_time: u64) -> f64 {
        let cutoff = current_time.saturating_sub(self.window_size);
        let total: u64 = self.buckets.iter()
            .filter(|(ts, _)| *ts >= cutoff)
            .map(|(_, count)| count)
            .sum();
        total as f64 / self.window_size as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let mut counter = Counter::new("requests");
        counter.inc();
        counter.inc_by(5.0);
        assert_eq!(counter.value(), 6.0);
    }

    #[test]
    fn test_counter_reset() {
        let mut counter = Counter::new("test");
        counter.inc_by(10.0);
        counter.reset();
        assert_eq!(counter.value(), 0.0);
    }

    #[test]
    fn test_rate_counter() {
        let mut rate = RateCounter::new(60, 60);
        rate.record(100, 10);
        rate.record(101, 5);
        assert!(rate.rate(101) > 0.0);
    }
}
