/// Advanced caching: ARC, write-through/write-back, and eviction policies.

use std::collections::{HashMap, VecDeque, HashSet};
use std::hash::Hash;

// ---------------------------------------------------------------------------
// Eviction policy trait
// ---------------------------------------------------------------------------

pub trait EvictionPolicy<K> {
    fn record_access(&mut self, key: &K);
    fn record_insert(&mut self, key: &K);
    fn choose_victim(&mut self) -> Option<K>;
    fn remove(&mut self, key: &K);
}

// ---------------------------------------------------------------------------
// LRU eviction policy
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct LruPolicy<K> {
    order: VecDeque<K>,
}

impl<K: Eq + Hash + Clone> LruPolicy<K> {
    pub fn new() -> Self {
        Self { order: VecDeque::new() }
    }
}

impl<K: Eq + Hash + Clone> EvictionPolicy<K> for LruPolicy<K> {
    fn record_access(&mut self, key: &K) {
        self.order.retain(|k| k != key);
        self.order.push_back(key.clone());
    }

    fn record_insert(&mut self, key: &K) {
        self.order.retain(|k| k != key);
        self.order.push_back(key.clone());
    }

    fn choose_victim(&mut self) -> Option<K> {
        self.order.pop_front()
    }

    fn remove(&mut self, key: &K) {
        self.order.retain(|k| k != key);
    }
}

// ---------------------------------------------------------------------------
// LFU eviction policy
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct LfuPolicy<K> {
    freq: HashMap<K, u64>,
}

impl<K: Eq + Hash + Clone> LfuPolicy<K> {
    pub fn new() -> Self {
        Self { freq: HashMap::new() }
    }
}

impl<K: Eq + Hash + Clone> EvictionPolicy<K> for LfuPolicy<K> {
    fn record_access(&mut self, key: &K) {
        *self.freq.entry(key.clone()).or_insert(0) += 1;
    }

    fn record_insert(&mut self, key: &K) {
        self.freq.entry(key.clone()).or_insert(1);
    }

    fn choose_victim(&mut self) -> Option<K> {
        let victim = self.freq.iter()
            .min_by_key(|(_, f)| **f)
            .map(|(k, _)| k.clone());
        if let Some(ref k) = victim {
            self.freq.remove(k);
        }
        victim
    }

    fn remove(&mut self, key: &K) {
        self.freq.remove(key);
    }
}

// ---------------------------------------------------------------------------
// Generic cache with pluggable eviction policy
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct PolicyCache<K, V, P: EvictionPolicy<K>> {
    capacity: usize,
    map: HashMap<K, V>,
    policy: P,
    hits: u64,
    misses: u64,
}

impl<K: Eq + Hash + Clone, V, P: EvictionPolicy<K>> PolicyCache<K, V, P> {
    pub fn new(capacity: usize, policy: P) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            policy,
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            self.policy.record_access(key);
            self.hits += 1;
            self.map.get(key)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            if let Some(victim) = self.policy.choose_victim() {
                self.map.remove(&victim);
            }
        }
        self.policy.record_insert(&key);
        self.map.insert(key, value);
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.policy.remove(key);
        self.map.remove(key)
    }

    pub fn len(&self) -> usize { self.map.len() }
    pub fn is_empty(&self) -> bool { self.map.is_empty() }
    pub fn capacity(&self) -> usize { self.capacity }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}

// ---------------------------------------------------------------------------
// ARC (Adaptive Replacement Cache)
// ---------------------------------------------------------------------------

/// Adapts between recency (LRU) and frequency (LFU) based on workload.
#[derive(Debug)]
pub struct ArcCache<K, V> {
    capacity: usize,
    p: usize, // target size for T1 (recency)

    t1: HashMap<K, V>, // recent items
    t2: HashMap<K, V>, // frequent items
    b1: HashSet<K>,    // ghost entries evicted from t1
    b2: HashSet<K>,    // ghost entries evicted from t2

    t1_order: VecDeque<K>,
    t2_order: VecDeque<K>,
    b1_order: VecDeque<K>,
    b2_order: VecDeque<K>,
}

impl<K: Eq + Hash + Clone, V> ArcCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            p: 0,
            t1: HashMap::new(),
            t2: HashMap::new(),
            b1: HashSet::new(),
            b2: HashSet::new(),
            t1_order: VecDeque::new(),
            t2_order: VecDeque::new(),
            b1_order: VecDeque::new(),
            b2_order: VecDeque::new(),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        // Hit in T1 -> promote to T2
        if self.t1.contains_key(key) {
            let val = self.t1.remove(key).unwrap();
            self.t1_order.retain(|k| k != key);
            self.t2.insert(key.clone(), val);
            self.t2_order.push_back(key.clone());
            return self.t2.get(key);
        }
        // Hit in T2 -> refresh order
        if self.t2.contains_key(key) {
            self.t2_order.retain(|k| k != key);
            self.t2_order.push_back(key.clone());
            return self.t2.get(key);
        }
        None
    }

    pub fn put(&mut self, key: K, value: V) {
        // Already in cache: update
        if self.t1.remove(&key).is_some() {
            self.t1_order.retain(|k| k != &key);
            self.t2.insert(key.clone(), value);
            self.t2_order.push_back(key);
            return;
        }
        if self.t2.remove(&key).is_some() {
            self.t2_order.retain(|k| k != &key);
            self.t2.insert(key.clone(), value);
            self.t2_order.push_back(key);
            return;
        }

        // Case I: ghost hit in B1 -> increase p (favor recency)
        if self.b1.remove(&key) {
            self.b1_order.retain(|k| k != &key);
            let delta = if self.b1.len() >= self.b2.len() {
                1
            } else {
                (self.b2.len() / (self.b1.len().max(1))) as usize
            };
            self.p = (self.p + delta).min(self.capacity);
        }
        // Case II: ghost hit in B2 -> decrease p (favor frequency)
        else if self.b2.remove(&key) {
            self.b2_order.retain(|k| k != &key);
            let delta = if self.b2.len() >= self.b1.len() {
                1
            } else {
                (self.b1.len() / (self.b2.len().max(1))) as usize
            };
            self.p = self.p.saturating_sub(delta);
        }

        // Evict if necessary
        let total = self.t1.len() + self.t2.len();
        if total >= self.capacity {
            if self.t1.len() > self.p || (self.t2.is_empty() && self.t1.len() == self.capacity) {
                // Evict from T1
                if let Some(victim) = self.t1_order.pop_front() {
                    self.t1.remove(&victim);
                    self.b1.insert(victim.clone());
                    self.b1_order.push_back(victim);
                    // Trim B1
                    while self.b1.len() + self.b2.len() > self.capacity {
                        if let Some(old) = self.b1_order.pop_front() {
                            self.b1.remove(&old);
                        }
                    }
                }
            } else {
                // Evict from T2
                if let Some(victim) = self.t2_order.pop_front() {
                    self.t2.remove(&victim);
                    self.b2.insert(victim.clone());
                    self.b2_order.push_back(victim);
                    while self.b1.len() + self.b2.len() > self.capacity {
                        if let Some(old) = self.b2_order.pop_front() {
                            self.b2.remove(&old);
                        }
                    }
                }
            }
        }

        self.t1.insert(key.clone(), value);
        self.t1_order.push_back(key);
    }

    pub fn len(&self) -> usize {
        self.t1.len() + self.t2.len()
    }

    pub fn is_empty(&self) -> bool {
        self.t1.is_empty() && self.t2.is_empty()
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.t1.contains_key(key) || self.t2.contains_key(key)
    }
}

// ---------------------------------------------------------------------------
// Write-through cache: every write is immediately persisted via callback
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct WriteThroughCache<K, V> {
    store: HashMap<K, V>,
    capacity: usize,
    write_count: u64,
}

impl<K: Eq + Hash + Clone, V: Clone> WriteThroughCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self { store: HashMap::new(), capacity, write_count: 0 }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.store.get(key)
    }

    /// Write to cache and invoke the persist callback immediately.
    pub fn put(&mut self, key: K, value: V, persist: impl Fn(&K, &V)) {
        persist(&key, &value);
        self.write_count += 1;
        if self.store.len() >= self.capacity && !self.store.contains_key(&key) {
            // evict arbitrary key
            if let Some(k) = self.store.keys().next().cloned() {
                self.store.remove(&k);
            }
        }
        self.store.insert(key, value);
    }

    pub fn write_count(&self) -> u64 { self.write_count }
    pub fn len(&self) -> usize { self.store.len() }
}

// ---------------------------------------------------------------------------
// Write-back cache: writes are buffered and flushed in batches
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct WriteBackCache<K, V> {
    store: HashMap<K, V>,
    dirty: HashSet<K>,
    capacity: usize,
    flush_threshold: usize,
    flush_count: u64,
}

impl<K: Eq + Hash + Clone, V: Clone> WriteBackCache<K, V> {
    pub fn new(capacity: usize, flush_threshold: usize) -> Self {
        Self {
            store: HashMap::new(),
            dirty: HashSet::new(),
            capacity,
            flush_threshold,
            flush_count: 0,
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.store.get(key)
    }

    pub fn put(&mut self, key: K, value: V) {
        if self.store.len() >= self.capacity && !self.store.contains_key(&key) {
            self.flush_dirty(|_, _| {});
            if self.store.len() >= self.capacity {
                if let Some(k) = self.store.keys().next().cloned() {
                    self.store.remove(&k);
                    self.dirty.remove(&k);
                }
            }
        }
        self.store.insert(key.clone(), value);
        self.dirty.insert(key);

        if self.dirty.len() >= self.flush_threshold {
            self.flush_dirty(|_, _| {});
        }
    }

    /// Flush all dirty entries using the provided persistence callback.
    pub fn flush_dirty(&mut self, persist: impl Fn(&K, &V)) {
        for key in self.dirty.iter() {
            if let Some(val) = self.store.get(key) {
                persist(key, val);
            }
        }
        self.flush_count += 1;
        self.dirty.clear();
    }

    pub fn dirty_count(&self) -> usize { self.dirty.len() }
    pub fn flush_count(&self) -> u64 { self.flush_count }
    pub fn len(&self) -> usize { self.store.len() }
}

// ---------------------------------------------------------------------------
// Clock eviction (approximation of LRU with lower overhead)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ClockCache<K, V> {
    capacity: usize,
    entries: Vec<Option<(K, V, bool)>>, // (key, value, reference_bit)
    hand: usize,
}

impl<K: Eq + Hash + Clone, V> ClockCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            entries.push(None);
        }
        Self { capacity, entries, hand: 0 }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        for entry in self.entries.iter_mut() {
            if let Some((k, v, ref mut bit)) = entry {
                if k == key {
                    *bit = true;
                    return Some(v);
                }
            }
        }
        None
    }

    pub fn put(&mut self, key: K, value: V) {
        // Check if key already exists
        for entry in self.entries.iter_mut() {
            if let Some((k, _, ref mut slot)) = entry {
                if k == &key {
                    *slot = (key, value, true);
                    return;
                }
            }
        }

        // Find slot using clock algorithm
        loop {
            if self.entries[self.hand].is_none() {
                self.entries[self.hand] = Some((key, value, false));
                self.hand = (self.hand + 1) % self.capacity;
                return;
            }

            let ref_bit = self.entries[self.hand].as_ref().unwrap().2;
            if ref_bit {
                self.entries[self.hand].as_mut().unwrap().2 = false;
                self.hand = (self.hand + 1) % self.capacity;
            } else {
                self.entries[self.hand] = Some((key, value, false));
                self.hand = (self.hand + 1) % self.capacity;
                return;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_policy_cache() {
        let mut cache = PolicyCache::new(3, LruPolicy::new());
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);

        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), Some(&3));

        cache.insert("d", 4); // evicts "a" (LRU)
        assert!(cache.get(&"a").is_none());
    }

    #[test]
    fn test_lfu_policy_cache() {
        let mut cache = PolicyCache::new(3, LfuPolicy::new());
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);

        cache.get(&"a");
        cache.get(&"a");
        cache.get(&"b");

        cache.insert("d", 4); // evicts "c" (lowest freq)
        assert!(cache.get(&"c").is_none());
        assert_eq!(cache.get(&"a"), Some(&1));
    }

    #[test]
    fn test_hit_rate_tracking() {
        let mut cache = PolicyCache::new(2, LruPolicy::new());
        cache.insert("x", 10);
        cache.get(&"x");
        cache.get(&"x");
        cache.get(&"y"); // miss

        assert!((cache.hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_arc_basic() {
        let mut cache = ArcCache::new(4);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        cache.put("d", 4);

        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn test_arc_promotes_to_t2() {
        let mut cache = ArcCache::new(4);
        cache.put("a", 1);
        cache.put("b", 2);

        // Access "a" twice to promote T1 -> T2
        cache.get(&"a");
        cache.get(&"a");

        // Fill remaining capacity
        cache.put("c", 3);
        cache.put("d", 4);

        // "a" should survive in T2; new insert evicts from T1
        cache.put("e", 5);
        assert_eq!(cache.get(&"a"), Some(&1));
    }

    #[test]
    fn test_arc_ghost_adaptation() {
        let mut cache = ArcCache::new(2);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3); // evicts "a" to B1
        cache.put("a", 1); // ghost hit in B1 -> increase p
        assert!(cache.contains_key(&"a"));
    }

    #[test]
    fn test_write_through() {
        let mut persisted = vec![];
        let mut cache = WriteThroughCache::new(3);

        cache.put("a", 1, |k, v| persisted.push((*k, *v)));
        cache.put("b", 2, |k, v| persisted.push((*k, *v)));

        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(persisted.len(), 2);
        assert_eq!(cache.write_count(), 2);
    }

    #[test]
    fn test_write_back_buffers_and_flushes() {
        let mut cache = WriteBackCache::new(10, 3);
        cache.put("a", 1);
        cache.put("b", 2);
        assert_eq!(cache.dirty_count(), 2);

        cache.put("c", 3); // triggers auto-flush at threshold
        assert_eq!(cache.dirty_count(), 0);
        assert_eq!(cache.flush_count(), 1);
    }

    #[test]
    fn test_write_back_manual_flush() {
        let mut cache = WriteBackCache::new(10, 100);
        cache.put("x", 42);
        cache.put("y", 99);

        let mut flushed = vec![];
        cache.flush_dirty(|k, v| flushed.push((*k, *v)));

        assert_eq!(flushed.len(), 2);
        assert_eq!(cache.dirty_count(), 0);
    }

    #[test]
    fn test_clock_cache_basic() {
        let mut cache = ClockCache::new(3);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);

        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_clock_cache_eviction() {
        let mut cache = ClockCache::new(2);
        cache.put("a", 1);
        cache.put("b", 2);

        // Access "b" to set reference bit
        cache.get(&"b");
        cache.put("c", 3); // evicts "a" (ref bit cleared on "b")

        assert!(cache.get(&"a").is_none());
        assert_eq!(cache.get(&"b"), Some(&2));
    }

    #[test]
    fn test_clock_cache_update() {
        let mut cache = ClockCache::new(2);
        cache.put("k", 1);
        cache.put("k", 2);
        assert_eq!(cache.get(&"k"), Some(&2));
        assert_eq!(cache.len(), 1);
    }
}
