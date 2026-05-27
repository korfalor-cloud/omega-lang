use std::collections::HashMap;
use std::collections::VecDeque;
use crate::vm::stack::Value;

pub struct OmegaLRUCache {
    capacity: usize,
    map: HashMap<String, Value>,
    order: VecDeque<String>,
}

impl OmegaLRUCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&Value> {
        if self.map.contains_key(key) {
            self.order.retain(|k| k != key);
            self.order.push_front(key.to_string());
            self.map.get(key)
        } else {
            None
        }
    }

    pub fn put(&mut self, key: String, value: Value) {
        if self.map.contains_key(&key) {
            self.order.retain(|k| k != &key);
        } else if self.map.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_back() {
                self.map.remove(&oldest);
            }
        }
        self.map.insert(key.clone(), value);
        self.order.push_front(key);
    }

    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.order.retain(|k| k != key);
        self.map.remove(key)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    pub fn keys(&self) -> Vec<&String> {
        self.order.iter().collect()
    }
}

impl Clone for OmegaLRUCache {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            map: self.map.clone(),
            order: self.order.clone(),
        }
    }
}
