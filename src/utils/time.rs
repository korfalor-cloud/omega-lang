use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use chrono::{DateTime, Utc, Local, NaiveDate, NaiveTime, NaiveDateTime};

pub fn now_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub fn now_iso8601() -> String {
    let now: DateTime<Utc> = Utc::now();
    now.to_rfc3339()
}

pub fn now_local() -> String {
    let now: DateTime<Local> = Local::now();
    now.format("%Y-%m-%d %H:%M:%S").to_string()
}

pub fn format_timestamp(ts: u64, fmt: &str) -> String {
    let naive = NaiveDateTime::from_timestamp_opt(ts as i64, 0);
    match naive {
        Some(dt) => dt.format(fmt).to_string(),
        None => "Invalid timestamp".to_string(),
    }
}

pub fn parse_date(s: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()
}

pub fn parse_time(s: &str) -> Option<NaiveTime> {
    NaiveTime::parse_from_str(s, "%H:%M:%S").ok()
}

pub fn parse_datetime(s: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").ok()
}

pub fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

pub fn sleep_ms(ms: u64) {
    std::thread::sleep(Duration::from_millis(ms));
}

pub fn sleep_secs(secs: f64) {
    std::thread::sleep(Duration::from_secs_f64(secs));
}

pub fn timer<F, T>(f: F) -> (T, f64)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let elapsed = elapsed_ms(start);
    (result, elapsed)
}

pub fn benchmark<F>(name: &str, iterations: u64, f: F) -> BenchmarkResult
where
    F: Fn(),
{
    let mut times = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        f();
        times.push(elapsed_ms(start));
    }

    let total: f64 = times.iter().sum();
    let mean = total / iterations as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let variance = times.iter()
        .map(|t| (t - mean).powi(2))
        .sum::<f64>() / iterations as f64;
    let std_dev = variance.sqrt();

    BenchmarkResult {
        name: name.to_string(),
        iterations,
        total_ms: total,
        mean_ms: mean,
        min_ms: min,
        max_ms: max,
        std_dev_ms: std_dev,
    }
}

#[derive(Debug)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: u64,
    pub total_ms: f64,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub std_dev_ms: f64,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} iterations, mean={:.3}ms, min={:.3}ms, max={:.3}ms, stddev={:.3}ms",
            self.name, self.iterations, self.mean_ms, self.min_ms, self.max_ms, self.std_dev_ms)
    }
}
