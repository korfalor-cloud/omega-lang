/// Gauge metric for tracking instantaneous values.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Gauge {
    name: String,
    value: f64,
    labels: HashMap<String, String>,
    description: String,
    min_value: f64,
    max_value: f64,
}

impl Gauge {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            value: 0.0,
            labels: HashMap::new(),
            description: String::new(),
            min_value: f64::MAX,
            max_value: f64::MIN,
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

    pub fn set(&mut self, value: f64) {
        self.value = value;
        if value < self.min_value {
            self.min_value = value;
        }
        if value > self.max_value {
            self.max_value = value;
        }
    }

    pub fn inc(&mut self) {
        self.set(self.value + 1.0);
    }

    pub fn dec(&mut self) {
        self.set(self.value - 1.0);
    }

    pub fn inc_by(&mut self, amount: f64) {
        self.set(self.value + amount);
    }

    pub fn dec_by(&mut self, amount: f64) {
        self.set(self.value - amount);
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn min(&self) -> f64 {
        self.min_value
    }

    pub fn max(&self) -> f64 {
        self.max_value
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }

    pub fn reset(&mut self) {
        self.value = 0.0;
        self.min_value = f64::MAX;
        self.max_value = f64::MIN;
    }
}

/// Moving average gauge
#[derive(Debug)]
pub struct MovingAverageGauge {
    values: Vec<f64>,
    window_size: usize,
    current_index: usize,
    sum: f64,
}

impl MovingAverageGauge {
    pub fn new(window_size: usize) -> Self {
        Self {
            values: vec![0.0; window_size],
            window_size,
            current_index: 0,
            sum: 0.0,
        }
    }

    pub fn record(&mut self, value: f64) {
        self.sum -= self.values[self.current_index];
        self.values[self.current_index] = value;
        self.sum += value;
        self.current_index = (self.current_index + 1) % self.window_size;
    }

    pub fn average(&self) -> f64 {
        self.sum / self.window_size as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauge() {
        let mut gauge = Gauge::new("temperature");
        gauge.set(25.0);
        assert_eq!(gauge.value(), 25.0);

        gauge.inc();
        assert_eq!(gauge.value(), 26.0);

        gauge.dec_by(5.0);
        assert_eq!(gauge.value(), 21.0);
    }

    #[test]
    fn test_gauge_min_max() {
        let mut gauge = Gauge::new("test");
        gauge.set(10.0);
        gauge.set(5.0);
        gauge.set(15.0);

        assert_eq!(gauge.min(), 5.0);
        assert_eq!(gauge.max(), 15.0);
    }

    #[test]
    fn test_moving_average() {
        let mut gauge = MovingAverageGauge::new(3);
        gauge.record(10.0);
        gauge.record(20.0);
        gauge.record(30.0);

        assert_eq!(gauge.average(), 20.0);

        gauge.record(40.0);
        assert_eq!(gauge.average(), 30.0); // (20+30+40)/3
    }
}
