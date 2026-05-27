use std::collections::BTreeMap;
use crate::vm::stack::Value;

pub struct OmegaSortedMap {
    data: BTreeMap<String, Value>,
}

impl OmegaSortedMap {
    pub fn new() -> Self {
        Self { data: BTreeMap::new() }
    }

    pub fn insert(&mut self, key: String, value: Value) -> Option<Value> {
        self.data.insert(key, value)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.data.remove(key)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn first(&self) -> Option<(&String, &Value)> {
        self.data.iter().next()
    }

    pub fn last(&self) -> Option<(&String, &Value)> {
        self.data.iter().next_back()
    }

    pub fn range(&self, start: &str, end: &str) -> Vec<(&String, &Value)> {
        self.data.range(start.to_string()..end.to_string()).collect()
    }

    pub fn range_inclusive(&self, start: &str, end: &str) -> Vec<(&String, &Value)> {
        self.data.range(start.to_string()..=end.to_string()).collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.data.iter()
    }

    pub fn keys(&self) -> Vec<&String> {
        self.data.keys().collect()
    }

    pub fn values(&self) -> Vec<&Value> {
        self.data.values().collect()
    }
}

impl Clone for OmegaSortedMap {
    fn clone(&self) -> Self {
        Self { data: self.data.clone() }
    }
}
