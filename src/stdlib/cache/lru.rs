/// LRU (Least Recently Used) cache implementation.

use std::collections::HashMap;
use std::hash::Hash;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct LruCache<K, V> {
    capacity: usize,
    map: HashMap<K, CacheEntry<V>>,
    access_order: Vec<K>,
}

#[derive(Debug)]
struct CacheEntry<V> {
    value: V,
    inserted_at: Instant,
    last_accessed: Instant,
    access_count: u64,
}

impl<K: Eq + Hash + Clone, V> LruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            access_order: Vec::new(),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(entry) = self.map.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;

            // Move to front of access order
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                let k = self.access_order.remove(pos);
                self.access_order.push(k);
            }

            Some(&entry.value)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if let Some(entry) = self.map.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;

            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                let k = self.access_order.remove(pos);
                self.access_order.push(k);
            }

            Some(&mut entry.value)
        } else {
            None
        }
    }

    pub fn put(&mut self, key: K, value: V) {
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            self.evict();
        }

        let now = Instant::now();
        self.map.insert(key.clone(), CacheEntry {
            value,
            inserted_at: now,
            last_accessed: now,
            access_count: 0,
        });
        self.access_order.push(key);
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.access_order.retain(|k| k != key);
        self.map.remove(key).map(|entry| entry.value)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.access_order.clear();
    }

    fn evict(&mut self) {
        if let Some(oldest_key) = self.access_order.first().cloned() {
            self.access_order.remove(0);
            self.map.remove(&oldest_key);
        }
    }

    pub fn keys(&self) -> Vec<&K> {
        self.access_order.iter().collect()
    }

    pub fn values(&self) -> Vec<&V> {
        self.access_order.iter()
            .filter_map(|k| self.map.get(k))
            .map(|entry| &entry.value)
            .collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.access_order.iter()
            .filter_map(move |k| self.map.get(k).map(|entry| (k, &entry.value)))
    }

    pub fn stats(&self) -> CacheStats {
        let total_accesses: u64 = self.map.values().map(|e| e.access_count).sum();
        CacheStats {
            size: self.map.len(),
            capacity: self.capacity,
            total_accesses,
            hit_rate: 0.0, // Would need separate tracking
        }
    }
}

#[derive(Debug)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub total_accesses: u64,
    pub hit_rate: f64,
}

/// LFU (Least Frequently Used) cache
#[derive(Debug)]
pub struct LfuCache<K, V> {
    capacity: usize,
    map: HashMap<K, (V, u64)>,
    min_freq: u64,
}

impl<K: Eq + Hash + Clone, V> LfuCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            min_freq: 0,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some((value, freq)) = self.map.get_mut(key) {
            *freq += 1;
            Some(value)
        } else {
            None
        }
    }

    pub fn put(&mut self, key: K, value: V) {
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            self.evict_lfu();
        }
        self.map.insert(key, (value, 1));
    }

    fn evict_lfu(&mut self) {
        if let Some(key) = self.map.iter()
            .min_by_key(|(_, (_, freq))| *freq)
            .map(|(k, _)| k.clone())
        {
            self.map.remove(&key);
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_basic() {
        let mut cache = LruCache::new(3);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);

        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), Some(&3));
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = LruCache::new(2);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3); // Should evict "a"

        assert!(cache.get(&"a").is_none());
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), Some(&3));
    }

    #[test]
    fn test_lru_access_updates_order() {
        let mut cache = LruCache::new(2);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.get(&"a"); // Access "a" to make it recently used
        cache.put("c", 3); // Should evict "b"

        assert_eq!(cache.get(&"a"), Some(&1));
        assert!(cache.get(&"b").is_none());
        assert_eq!(cache.get(&"c"), Some(&3));
    }

    #[test]
    fn test_lru_remove() {
        let mut cache = LruCache::new(3);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.remove(&"a");

        assert!(cache.get(&"a").is_none());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lfu_cache() {
        let mut cache = LfuCache::new(3);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);

        cache.get(&"a");
        cache.get(&"a");
        cache.get(&"b");

        cache.put("d", 4); // Should evict "c" (lowest frequency)

        assert!(cache.get(&"c").is_none());
        assert_eq!(cache.get(&"a"), Some(&1));
    }
}
