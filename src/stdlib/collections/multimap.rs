use std::collections::HashMap;
use crate::vm::stack::Value;

pub struct OmegaMultimap {
    data: HashMap<String, Vec<Value>>,
}

impl OmegaMultimap {
    pub fn new() -> Self {
        Self { data: HashMap::new() }
    }

    pub fn insert(&mut self, key: String, value: Value) {
        self.data.entry(key).or_insert_with(Vec::new).push(value);
    }

    pub fn get(&self, key: &str) -> Option<&Vec<Value>> {
        self.data.get(key)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Vec<Value>> {
        self.data.get_mut(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<Vec<Value>> {
        self.data.remove(key)
    }

    pub fn remove_value(&mut self, key: &str, value: &Value) -> bool {
        if let Some(values) = self.data.get_mut(key) {
            let len_before = values.len();
            values.retain(|v| v != value);
            values.len() < len_before
        } else {
            false
        }
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    pub fn contains_value(&self, key: &str, value: &Value) -> bool {
        self.data.get(key).map_or(false, |values| values.contains(value))
    }

    pub fn len(&self) -> usize {
        self.data.values().map(|v| v.len()).sum()
    }

    pub fn key_count(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn keys(&self) -> Vec<&String> {
        self.data.keys().collect()
    }

    pub fn values_flat(&self) -> Vec<&Value> {
        self.data.values().flat_map(|v| v.iter()).collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Vec<Value>)> {
        self.data.iter()
    }

    pub fn flatten(&self) -> Vec<(&String, &Value)> {
        self.data.iter()
            .flat_map(|(k, values)| values.iter().map(move |v| (k, v)))
            .collect()
    }
}

impl Clone for OmegaMultimap {
    fn clone(&self) -> Self {
        Self { data: self.data.clone() }
    }
}
