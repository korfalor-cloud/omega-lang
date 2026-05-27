/// Time-to-live cache with automatic expiration.

use std::collections::HashMap;
use std::hash::Hash;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct TtlCache<K, V> {
    entries: HashMap<K, TtlEntry<V>>,
    default_ttl: Duration,
}

#[derive(Debug)]
struct TtlEntry<V> {
    value: V,
    expires_at: Instant,
}

impl<K: Eq + Hash + Clone, V> TtlCache<K, V> {
    pub fn new(default_ttl: Duration) -> Self {
        Self {
            entries: HashMap::new(),
            default_ttl,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.cleanup();
        self.entries.get(key).map(|entry| &entry.value)
    }

    pub fn put(&mut self, key: K, value: V) {
        self.entries.insert(key, TtlEntry {
            value,
            expires_at: Instant::now() + self.default_ttl,
        });
    }

    pub fn put_with_ttl(&mut self, key: K, value: V, ttl: Duration) {
        self.entries.insert(key, TtlEntry {
            value,
            expires_at: Instant::now() + ttl,
        });
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.entries.remove(key).map(|entry| entry.value)
    }

    pub fn contains_key(&mut self, key: &K) -> bool {
        self.cleanup();
        self.entries.contains_key(key)
    }

    pub fn len(&mut self) -> usize {
        self.cleanup();
        self.entries.len()
    }

    pub fn is_empty(&mut self) -> bool {
        self.cleanup();
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    fn cleanup(&mut self) {
        let now = Instant::now();
        self.entries.retain(|_, entry| entry.expires_at > now);
    }

    pub fn expired_count(&self) -> usize {
        let now = Instant::now();
        self.entries.values().filter(|e| e.expires_at <= now).count()
    }

    pub fn set_default_ttl(&mut self, ttl: Duration) {
        self.default_ttl = ttl;
    }
}

/// Sliding window rate limiter
#[derive(Debug)]
pub struct RateLimiter {
    window_size: Duration,
    max_requests: usize,
    requests: Vec<Instant>,
}

impl RateLimiter {
    pub fn new(window_size: Duration, max_requests: usize) -> Self {
        Self {
            window_size,
            max_requests,
            requests: Vec::new(),
        }
    }

    pub fn try_acquire(&mut self) -> bool {
        let now = Instant::now();
        self.requests.retain(|t| now.duration_since(*t) < self.window_size);

        if self.requests.len() < self.max_requests {
            self.requests.push(now);
            true
        } else {
            false
        }
    }

    pub fn available_permits(&mut self) -> usize {
        let now = Instant::now();
        self.requests.retain(|t| now.duration_since(*t) < self.window_size);
        self.max_requests - self.requests.len()
    }

    pub fn reset(&mut self) {
        self.requests.clear();
    }
}

/// Token bucket rate limiter
#[derive(Debug)]
pub struct TokenBucket {
    capacity: f64,
    tokens: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl TokenBucket {
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    pub fn try_acquire(&mut self, tokens: f64) -> bool {
        self.refill();
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }

    pub fn available_tokens(&mut self) -> f64 {
        self.refill();
        self.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_ttl_cache_basic() {
        let mut cache = TtlCache::new(Duration::from_secs(1));
        cache.put("key", "value");
        assert_eq!(cache.get(&"key"), Some(&"value"));
    }

    #[test]
    fn test_ttl_cache_expiration() {
        let mut cache = TtlCache::new(Duration::from_millis(50));
        cache.put("key", "value");
        thread::sleep(Duration::from_millis(100));
        assert!(cache.get(&"key").is_none());
    }

    #[test]
    fn test_ttl_custom() {
        let mut cache = TtlCache::new(Duration::from_secs(10));
        cache.put_with_ttl("short", "value", Duration::from_millis(50));
        cache.put("long", "value");

        thread::sleep(Duration::from_millis(100));
        assert!(cache.get(&"short").is_none());
        assert_eq!(cache.get(&"long"), Some(&"value"));
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(Duration::from_secs(1), 3);
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(!limiter.try_acquire());
    }

    #[test]
    fn test_token_bucket() {
        let mut bucket = TokenBucket::new(10.0, 1.0);
        assert!(bucket.try_acquire(5.0));
        assert!(bucket.try_acquire(5.0));
        assert!(!bucket.try_acquire(1.0));
    }
}
