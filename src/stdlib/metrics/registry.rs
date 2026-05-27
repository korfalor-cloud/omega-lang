/// Metrics registry for managing all metrics.

use super::counter::Counter;
use super::gauge::Gauge;
use super::histogram::Histogram;
use std::collections::HashMap;

#[derive(Debug)]
pub struct MetricsRegistry {
    counters: HashMap<String, Counter>,
    gauges: HashMap<String, Gauge>,
    histograms: HashMap<String, Histogram>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
        }
    }

    pub fn register_counter(&mut self, counter: Counter) {
        self.counters.insert(counter.name().to_string(), counter);
    }

    pub fn register_gauge(&mut self, gauge: Gauge) {
        self.gauges.insert(gauge.name().to_string(), gauge);
    }

    pub fn register_histogram(&mut self, histogram: Histogram) {
        self.histograms.insert(histogram.name().to_string(), histogram);
    }

    pub fn counter(&self, name: &str) -> Option<&Counter> {
        self.counters.get(name)
    }

    pub fn counter_mut(&mut self, name: &str) -> Option<&mut Counter> {
        self.counters.get_mut(name)
    }

    pub fn gauge(&self, name: &str) -> Option<&Gauge> {
        self.gauges.get(name)
    }

    pub fn gauge_mut(&mut self, name: &str) -> Option<&mut Gauge> {
        self.gauges.get_mut(name)
    }

    pub fn histogram(&self, name: &str) -> Option<&Histogram> {
        self.histograms.get(name)
    }

    pub fn histogram_mut(&mut self, name: &str) -> Option<&mut Histogram> {
        self.histograms.get_mut(name)
    }

    pub fn counter_names(&self) -> Vec<&str> {
        self.counters.keys().map(|s| s.as_str()).collect()
    }

    pub fn gauge_names(&self) -> Vec<&str> {
        self.gauges.keys().map(|s| s.as_str()).collect()
    }

    pub fn histogram_names(&self) -> Vec<&str> {
        self.histograms.keys().map(|s| s.as_str()).collect()
    }

    pub fn reset_all(&mut self) {
        for counter in self.counters.values_mut() {
            counter.reset();
        }
        for gauge in self.gauges.values_mut() {
            gauge.reset();
        }
        for histogram in self.histograms.values_mut() {
            histogram.reset();
        }
    }

    pub fn to_text(&self) -> String {
        let mut output = String::new();

        for counter in self.counters.values() {
            output.push_str(&format!("# TYPE {} counter\n", counter.name()));
            output.push_str(&format!("{} {}\n", counter.name(), counter.value()));
        }

        for gauge in self.gauges.values() {
            output.push_str(&format!("# TYPE {} gauge\n", gauge.name()));
            output.push_str(&format!("{} {}\n", gauge.name(), gauge.value()));
        }

        for histogram in self.histograms.values() {
            output.push_str(&format!("# TYPE {} histogram\n", histogram.name()));
            for (bound, count) in histogram.bucket_counts() {
                output.push_str(&format!("{}_bucket{{le=\"{}\"}} {}\n", histogram.name(), bound, count));
            }
            output.push_str(&format!("{}_count {}\n", histogram.name(), histogram.count()));
            output.push_str(&format!("{}_sum {}\n", histogram.name(), histogram.sum()));
        }

        output
    }
}

/// Timer for measuring durations
#[derive(Debug)]
pub struct Timer {
    start: std::time::Instant,
    name: String,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Self {
            start: std::time::Instant::now(),
            name: name.to_string(),
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1000.0
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry() {
        let mut registry = MetricsRegistry::new();

        let counter = Counter::new("requests").with_description("Total requests");
        registry.register_counter(counter);

        let gauge = Gauge::new("connections").with_description("Active connections");
        registry.register_gauge(gauge);

        let histogram = Histogram::new("latency", &[10.0, 50.0, 100.0]);
        registry.register_histogram(histogram);

        assert!(registry.counter("requests").is_some());
        assert!(registry.gauge("connections").is_some());
        assert!(registry.histogram("latency").is_some());
    }

    #[test]
    fn test_registry_text_output() {
        let mut registry = MetricsRegistry::new();
        registry.register_counter(Counter::new("test_counter"));
        registry.register_gauge(Gauge::new("test_gauge"));

        registry.counter_mut("test_counter").unwrap().inc_by(42.0);
        registry.gauge_mut("test_gauge").unwrap().set(3.14);

        let text = registry.to_text();
        assert!(text.contains("test_counter 42"));
        assert!(text.contains("test_gauge 3.14"));
    }

    #[test]
    fn test_timer() {
        let timer = Timer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10.0);
    }
}
