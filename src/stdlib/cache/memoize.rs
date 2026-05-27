/// Memoization / caching decorator for pure functions.

use std::collections::HashMap;
use std::hash::Hash;

/// Generic memoizer that caches results of a function call.
#[derive(Debug)]
pub struct Memoizer<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    cache: HashMap<K, V>,
    hits: u64,
    misses: u64,
    max_size: usize,
}

impl<K, V> Memoizer<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            hits: 0,
            misses: 0,
            max_size: usize::MAX,
        }
    }

    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size.min(1024)),
            hits: 0,
            misses: 0,
            max_size,
        }
    }

    /// Get a cached value or compute it with the provided closure.
    pub fn get_or_insert_with<F>(&mut self, key: K, f: F) -> &V
    where
        F: FnOnce() -> V,
    {
        if !self.cache.contains_key(&key) {
            if self.cache.len() >= self.max_size {
                // Evict oldest entry (arbitrary choice since HashMap has no order)
                if let Some(first_key) = self.cache.keys().next().cloned() {
                    self.cache.remove(&first_key);
                }
            }
            let value = f();
            self.cache.insert(key.clone(), value);
            self.misses += 1;
        } else {
            self.hits += 1;
        }
        self.cache.get(&key).unwrap()
    }

    /// Get a cached value if it exists.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.cache.get(key)
    }

    /// Insert a value directly.
    pub fn insert(&mut self, key: K, value: V) {
        if self.cache.len() >= self.max_size && !self.cache.contains_key(&key) {
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(key, value);
    }

    /// Check if a key is cached.
    pub fn contains_key(&self, key: &K) -> bool {
        self.cache.contains_key(key)
    }

    /// Remove a specific entry.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.cache.remove(key)
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Cache hit count.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache miss count.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hit rate as a fraction in [0, 1].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Maximum cache size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// All cached keys.
    pub fn keys(&self) -> Vec<&K> {
        self.cache.keys().collect()
    }
}

impl<K, V> Default for Memoizer<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A memoized function wrapper that automatically caches results.
#[derive(Debug)]
pub struct MemoizedFn<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    memoizer: Memoizer<K, V>,
    call_count: u64,
}

impl<K, V> MemoizedFn<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    pub fn new() -> Self {
        Self {
            memoizer: Memoizer::new(),
            call_count: 0,
        }
    }

    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            memoizer: Memoizer::with_max_size(max_size),
            call_count: 0,
        }
    }

    pub fn call<F>(&mut self, key: K, f: F) -> &V
    where
        F: FnOnce(&K) -> V,
    {
        self.call_count += 1;
        let memoizer = &mut self.memoizer;
        if !memoizer.contains_key(&key) {
            let value = f(&key);
            memoizer.insert(key.clone(), value);
        }
        memoizer.get(&key).unwrap()
    }

    pub fn memoizer(&self) -> &Memoizer<K, V> {
        &self.memoizer
    }

    pub fn memoizer_mut(&mut self) -> &mut Memoizer<K, V> {
        &mut self.memoizer
    }

    pub fn call_count(&self) -> u64 {
        self.call_count
    }

    pub fn reset(&mut self) {
        self.memoizer.clear();
        self.call_count = 0;
    }
}

impl<K, V> Default for MemoizedFn<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Cache with automatic expiration based on a TTL (time-to-live) in seconds.
#[derive(Debug)]
pub struct ExpiringCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    entries: HashMap<K, (V, u64)>,
    ttl_seconds: u64,
}

impl<K, V> ExpiringCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    pub fn new(ttl_seconds: u64) -> Self {
        Self {
            entries: HashMap::new(),
            ttl_seconds,
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        let expiry = current_timestamp() + self.ttl_seconds;
        self.entries.insert(key, (value, expiry));
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        let now = current_timestamp();
        if let Some((value, expiry)) = self.entries.get(key) {
            if now < *expiry {
                return Some(value);
            }
        }
        None
    }

    pub fn get_or_insert_with<F>(&mut self, key: K, f: F) -> &V
    where
        F: FnOnce() -> V,
    {
        let now = current_timestamp();
        let needs_insert = match self.entries.get(&key) {
            Some((_, expiry)) => now >= *expiry,
            None => true,
        };
        if needs_insert {
            let value = f();
            self.entries.insert(key.clone(), (value, now + self.ttl_seconds));
        }
        &self.entries.get(&key).unwrap().0
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.entries.remove(key).map(|(v, _)| v)
    }

    pub fn cleanup(&mut self) {
        let now = current_timestamp();
        self.entries.retain(|_, (_, expiry)| now < *expiry);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memoizer_basic() {
        let mut memo = Memoizer::new();
        let result = memo.get_or_insert_with(42, || 100);
        assert_eq!(*result, 100);
        assert_eq!(memo.len(), 1);

        // Second call should return cached value
        let result = memo.get_or_insert_with(42, || 999);
        assert_eq!(*result, 100);
        assert_eq!(memo.hits(), 1);
        assert_eq!(memo.misses(), 1);
    }

    #[test]
    fn test_memoizer_max_size() {
        let mut memo = Memoizer::with_max_size(2);
        memo.insert("a", 1);
        memo.insert("b", 2);
        assert_eq!(memo.len(), 2);

        // Inserting a third should evict one
        memo.insert("c", 3);
        assert_eq!(memo.len(), 2);
    }

    #[test]
    fn test_memoizer_hit_rate() {
        let mut memo = Memoizer::new();
        memo.get_or_insert_with("x", || 1);
        memo.get_or_insert_with("x", || 2);
        memo.get_or_insert_with("y", || 3);

        assert_eq!(memo.hits(), 1);
        assert_eq!(memo.misses(), 2);
        let rate = memo.hit_rate();
        assert!((rate - 1.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memoized_fn() {
        let mut f = MemoizedFn::new();
        let r1 = f.call(5, |n| n * 2);
        assert_eq!(*r1, 10);

        let r2 = f.call(5, |n| n * 100);
        assert_eq!(*r2, 10); // cached

        assert_eq!(f.call_count(), 2);
        assert_eq!(f.memoizer().hits(), 1);
    }

    #[test]
    fn test_expiring_cache() {
        let mut cache = ExpiringCache::new(3600);
        cache.insert("key", "value");
        assert_eq!(cache.get(&"key"), Some(&"value"));
        assert_eq!(cache.len(), 1);

        cache.remove(&"key");
        assert_eq!(cache.get(&"key"), None);
    }
}
