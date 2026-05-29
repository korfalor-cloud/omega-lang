/// Advanced logging: rotation, async buffering, aggregation, and filtering.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use super::logger::{LogEntry, LogLevel};

// ---------------------------------------------------------------------------
// FilteredLogger — composable level + key/value filters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LogFilter {
    pub min_level: LogLevel,
    pub max_level: Option<LogLevel>,
    pub module_pattern: Option<String>,
    pub field_filters: HashMap<String, String>,
}

impl LogFilter {
    pub fn new(min_level: LogLevel) -> Self {
        Self {
            min_level,
            max_level: None,
            module_pattern: None,
            field_filters: HashMap::new(),
        }
    }

    pub fn with_max_level(mut self, level: LogLevel) -> Self {
        self.max_level = Some(level);
        self
    }

    pub fn with_module(mut self, pattern: &str) -> Self {
        self.module_pattern = Some(pattern.to_string());
        self
    }

    pub fn with_field_match(mut self, key: &str, value: &str) -> Self {
        self.field_filters.insert(key.to_string(), value.to_string());
        self
    }

    pub fn matches(&self, entry: &LogEntry) -> bool {
        if entry.level < self.min_level {
            return false;
        }
        if let Some(ref max) = self.max_level {
            if entry.level > *max {
                return false;
            }
        }
        if let Some(ref pattern) = self.module_pattern {
            match &entry.module {
                Some(m) if m.contains(pattern.as_str()) => {}
                _ => return false,
            }
        }
        for (k, v) in &self.field_filters {
            match entry.fields.get(k) {
                Some(val) if val == v => {}
                _ => return false,
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// LogRotation — time-based and size-based rotation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum RotationPolicy {
    /// Rotate after the log exceeds `max_bytes`.
    SizeBytes(usize),
    /// Rotate after `max_entries` entries.
    EntryCount(usize),
    /// Rotate every `seconds`.
    TimeInterval(u64),
}

#[derive(Debug)]
pub struct RotatingBuffer {
    policy: RotationPolicy,
    current: Vec<LogEntry>,
    archives: Vec<Vec<LogEntry>>,
    max_archives: usize,
    bytes_written: usize,
    last_rotation_ts: u64,
}

impl RotatingBuffer {
    pub fn new(policy: RotationPolicy) -> Self {
        Self {
            policy,
            current: Vec::new(),
            archives: Vec::new(),
            max_archives: 5,
            bytes_written: 0,
            last_rotation_ts: now_ts(),
        }
    }

    pub fn with_max_archives(mut self, n: usize) -> Self {
        self.max_archives = n;
        self
    }

    fn should_rotate(&self) -> bool {
        match &self.policy {
            RotationPolicy::SizeBytes(max) => self.bytes_written >= *max,
            RotationPolicy::EntryCount(max) => self.current.len() >= *max,
            RotationPolicy::TimeInterval(secs) => now_ts().saturating_sub(self.last_rotation_ts) >= *secs,
        }
    }

    fn rotate(&mut self) {
        if self.current.is_empty() {
            return;
        }
        if self.archives.len() >= self.max_archives {
            self.archives.remove(0);
        }
        self.archives.push(std::mem::take(&mut self.current));
        self.bytes_written = 0;
        self.last_rotation_ts = now_ts();
    }

    /// Push a single entry; triggers rotation when the policy threshold is met.
    pub fn push(&mut self, entry: LogEntry) {
        self.bytes_written += entry.message.len() + 64; // rough overhead
        self.current.push(entry);
        if self.should_rotate() {
            self.rotate();
        }
    }

    pub fn current(&self) -> &[LogEntry] {
        &self.current
    }

    pub fn archives(&self) -> &[Vec<LogEntry>] {
        &self.archives
    }

    /// Return every entry across current + archived segments.
    pub fn all_entries(&self) -> Vec<&LogEntry> {
        let mut out: Vec<&LogEntry> = self.archives.iter().flatten().collect();
        out.extend(self.current.iter());
        out
    }
}

// ---------------------------------------------------------------------------
// AsyncLogBuffer — lock-free-style async buffer backed by Arc<Mutex>
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AsyncLogBuffer {
    inner: Arc<Mutex<AsyncBufferInner>>,
}

#[derive(Debug)]
struct AsyncBufferInner {
    queue: VecDeque<LogEntry>,
    capacity: usize,
    dropped: u64,
}

impl AsyncLogBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AsyncBufferInner {
                queue: VecDeque::with_capacity(capacity),
                capacity,
                dropped: 0,
            })),
        }
    }

    /// Enqueue a log entry. Returns `false` if the buffer is full and the entry was dropped.
    pub fn enqueue(&self, entry: LogEntry) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if inner.queue.len() >= inner.capacity {
            inner.dropped += 1;
            return false;
        }
        inner.queue.push_back(entry);
        true
    }

    /// Drain up to `n` entries from the front of the queue.
    pub fn drain(&self, n: usize) -> Vec<LogEntry> {
        let mut inner = self.inner.lock().unwrap();
        let take = n.min(inner.queue.len());
        inner.queue.drain(..take).collect()
    }

    /// Drain all pending entries.
    pub fn drain_all(&self) -> Vec<LogEntry> {
        let mut inner = self.inner.lock().unwrap();
        inner.queue.drain(..).collect()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().queue.is_empty()
    }

    pub fn dropped(&self) -> u64 {
        self.inner.lock().unwrap().dropped
    }

    pub fn capacity(&self) -> usize {
        self.inner.lock().unwrap().capacity
    }
}

// ---------------------------------------------------------------------------
// LogAggregator — counts + min/max/avg per level and per field key
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct LevelStats {
    pub count: u64,
    pub first_ts: Option<u64>,
    pub last_ts: Option<u64>,
}

#[derive(Debug)]
pub struct LogAggregator {
    level_counts: HashMap<LogLevel, LevelStats>,
    field_value_counts: HashMap<String, HashMap<String, u64>>,
    total: u64,
}

impl LogAggregator {
    pub fn new() -> Self {
        Self {
            level_counts: HashMap::new(),
            field_value_counts: HashMap::new(),
            total: 0,
        }
    }

    pub fn ingest(&mut self, entry: &LogEntry) {
        self.total += 1;
        let stats = self.level_counts.entry(entry.level).or_default();
        stats.count += 1;
        if stats.first_ts.is_none() {
            stats.first_ts = Some(entry.timestamp);
        }
        stats.last_ts = Some(entry.timestamp);

        for (k, v) in &entry.fields {
            *self.field_value_counts
                .entry(k.clone())
                .or_default()
                .entry(v.clone())
                .or_insert(0) += 1;
        }
    }

    pub fn ingest_batch(&mut self, entries: &[LogEntry]) {
        for e in entries {
            self.ingest(e);
        }
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn count_for_level(&self, level: LogLevel) -> u64 {
        self.level_counts.get(&level).map_or(0, |s| s.count)
    }

    pub fn stats_for_level(&self, level: LogLevel) -> Option<&LevelStats> {
        self.level_counts.get(&level)
    }

    pub fn field_value_counts(&self) -> &HashMap<String, HashMap<String, u64>> {
        &self.field_value_counts
    }

    pub fn reset(&mut self) {
        self.level_counts.clear();
        self.field_value_counts.clear();
        self.total = 0;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(level: LogLevel, msg: &str) -> LogEntry {
        LogEntry {
            level,
            message: msg.to_string(),
            timestamp: now_ts(),
            fields: HashMap::new(),
            module: None,
            line: None,
        }
    }

    fn entry_with_fields(level: LogLevel, msg: &str, fields: Vec<(&str, &str)>) -> LogEntry {
        LogEntry {
            level,
            message: msg.to_string(),
            timestamp: now_ts(),
            fields: fields.into_iter().map(|(k, v)| (k.into(), v.into())).collect(),
            module: None,
            line: None,
        }
    }

    // -- LogFilter ----------------------------------------------------------

    #[test]
    fn filter_by_min_level() {
        let f = LogFilter::new(LogLevel::Warn);
        assert!(!f.matches(&entry(LogLevel::Info, "no")));
        assert!(f.matches(&entry(LogLevel::Warn, "yes")));
        assert!(f.matches(&entry(LogLevel::Error, "yes")));
    }

    #[test]
    fn filter_by_range() {
        let f = LogFilter::new(LogLevel::Debug).with_max_level(LogLevel::Warn);
        assert!(!f.matches(&entry(LogLevel::Trace, "out")));
        assert!(f.matches(&entry(LogLevel::Debug, "in")));
        assert!(f.matches(&entry(LogLevel::Warn, "in")));
        assert!(!f.matches(&entry(LogLevel::Error, "out")));
    }

    #[test]
    fn filter_by_field() {
        let f = LogFilter::new(LogLevel::Trace).with_field_match("svc", "auth");
        let good = entry_with_fields(LogLevel::Info, "ok", vec![("svc", "auth")]);
        let bad = entry_with_fields(LogLevel::Info, "no", vec![("svc", "db")]);
        assert!(f.matches(&good));
        assert!(!f.matches(&bad));
    }

    // -- RotatingBuffer -----------------------------------------------------

    #[test]
    fn rotation_by_count() {
        let mut buf = RotatingBuffer::new(RotationPolicy::EntryCount(3));
        for i in 0..5 {
            buf.push(entry(LogLevel::Info, &format!("msg {i}")));
        }
        // First 3 entries rotated, next 2 in current
        assert_eq!(buf.current().len(), 2);
        assert_eq!(buf.archives().len(), 1);
        assert_eq!(buf.archives()[0].len(), 3);
    }

    #[test]
    fn rotation_max_archives() {
        let mut buf = RotatingBuffer::new(RotationPolicy::EntryCount(2)).with_max_archives(2);
        for i in 0..8 {
            buf.push(entry(LogLevel::Info, &format!("{i}")));
        }
        assert!(buf.archives().len() <= 2);
    }

    #[test]
    fn all_entries_flattened() {
        let mut buf = RotatingBuffer::new(RotationPolicy::EntryCount(2));
        for i in 0..6 {
            buf.push(entry(LogLevel::Info, &format!("{i}")));
        }
        let all = buf.all_entries();
        assert_eq!(all.len(), 6);
    }

    // -- AsyncLogBuffer -----------------------------------------------------

    #[test]
    fn async_enqueue_and_drain() {
        let buf = AsyncLogBuffer::new(8);
        for i in 0..5 {
            assert!(buf.enqueue(entry(LogLevel::Info, &format!("{i}"))));
        }
        assert_eq!(buf.len(), 5);

        let batch = buf.drain(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(buf.len(), 2);

        let rest = buf.drain_all();
        assert_eq!(rest.len(), 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn async_overflow_drops() {
        let buf = AsyncLogBuffer::new(2);
        assert!(buf.enqueue(entry(LogLevel::Info, "a")));
        assert!(buf.enqueue(entry(LogLevel::Info, "b")));
        assert!(!buf.enqueue(entry(LogLevel::Info, "c"))); // dropped
        assert_eq!(buf.dropped(), 1);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn async_buffer_clone_shares_state() {
        let buf1 = AsyncLogBuffer::new(16);
        let buf2 = buf1.clone();
        buf1.enqueue(entry(LogLevel::Info, "shared"));
        assert_eq!(buf2.len(), 1);
    }

    // -- LogAggregator ------------------------------------------------------

    #[test]
    fn aggregator_counts_levels() {
        let mut agg = LogAggregator::new();
        agg.ingest(&entry(LogLevel::Info, "a"));
        agg.ingest(&entry(LogLevel::Info, "b"));
        agg.ingest(&entry(LogLevel::Error, "c"));

        assert_eq!(agg.total(), 3);
        assert_eq!(agg.count_for_level(LogLevel::Info), 2);
        assert_eq!(agg.count_for_level(LogLevel::Error), 1);
        assert_eq!(agg.count_for_level(LogLevel::Warn), 0);
    }

    #[test]
    fn aggregator_tracks_field_values() {
        let mut agg = LogAggregator::new();
        agg.ingest(&entry_with_fields(LogLevel::Info, "a", vec![("region", "us")]));
        agg.ingest(&entry_with_fields(LogLevel::Info, "b", vec![("region", "eu")]));
        agg.ingest(&entry_with_fields(LogLevel::Info, "c", vec![("region", "us")]));

        let regions = agg.field_value_counts().get("region").unwrap();
        assert_eq!(regions.get("us"), Some(&2));
        assert_eq!(regions.get("eu"), Some(&1));
    }

    #[test]
    fn aggregator_reset() {
        let mut agg = LogAggregator::new();
        agg.ingest(&entry(LogLevel::Info, "x"));
        agg.reset();
        assert_eq!(agg.total(), 0);
        assert_eq!(agg.count_for_level(LogLevel::Info), 0);
    }

    #[test]
    fn aggregator_ingest_batch() {
        let entries: Vec<LogEntry> = (0..10).map(|i| entry(LogLevel::Warn, &format!("{i}"))).collect();
        let mut agg = LogAggregator::new();
        agg.ingest_batch(&entries);
        assert_eq!(agg.total(), 10);
        assert_eq!(agg.count_for_level(LogLevel::Warn), 10);
    }
}
