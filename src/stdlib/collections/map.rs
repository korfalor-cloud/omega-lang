use std::collections::HashMap;
use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;

pub struct OmegaMap {
    data: HashMap<String, Value>,
    insertion_order: Vec<String>,
}

impl OmegaMap {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            insertion_order: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
            insertion_order: Vec::with_capacity(capacity),
        }
    }

    pub fn insert(&mut self, key: String, value: Value) -> Option<Value> {
        if !self.data.contains_key(&key) {
            self.insertion_order.push(key.clone());
        }
        self.data.insert(key, value)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.data.get_mut(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.insertion_order.retain(|k| k != key);
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
        self.insertion_order.clear();
    }

    pub fn keys(&self) -> Vec<&String> {
        self.insertion_order.iter().collect()
    }

    pub fn values(&self) -> Vec<&Value> {
        self.insertion_order.iter()
            .filter_map(|k| self.data.get(k))
            .collect()
    }

    pub fn entries(&self) -> Vec<(&String, &Value)> {
        self.insertion_order.iter()
            .filter_map(|k| self.data.get(k).map(|v| (k, v)))
            .collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.insertion_order.iter()
            .filter_map(move |k| self.data.get(k).map(|v| (k, v)))
    }

    pub fn merge(&mut self, other: &Self) {
        for (key, value) in &other.data {
            if !self.data.contains_key(key) {
                self.insertion_order.push(key.clone());
            }
            self.data.insert(key.clone(), value.clone());
        }
    }

    pub fn extend(&mut self, other: impl IntoIterator<Item = (String, Value)>) {
        for (key, value) in other {
            self.insert(key, value);
        }
    }

    pub fn map_values(&self, f: impl Fn(&Value) -> Value) -> Self {
        let mut result = Self::new();
        for key in &self.insertion_order {
            if let Some(value) = self.data.get(key) {
                result.insert(key.clone(), f(value));
            }
        }
        result
    }

    pub fn filter(&self, f: impl Fn(&str, &Value) -> bool) -> Self {
        let mut result = Self::new();
        for key in &self.insertion_order {
            if let Some(value) = self.data.get(key) {
                if f(key, value) {
                    result.insert(key.clone(), value.clone());
                }
            }
        }
        result
    }

    pub fn find_key(&self, f: impl Fn(&Value) -> bool) -> Option<&String> {
        self.insertion_order.iter().find(|k| {
            self.data.get(k.as_str()).map_or(false, |v| f(v))
        })
    }

    pub fn find_value(&self, f: impl Fn(&str) -> bool) -> Option<&Value> {
        self.insertion_order.iter().find(|k| f(k)).and_then(|k| self.data.get(k))
    }

    pub fn any(&self, f: impl Fn(&str, &Value) -> bool) -> bool {
        self.insertion_order.iter().any(|k| {
            self.data.get(k).map_or(false, |v| f(k, v))
        })
    }

    pub fn all(&self, f: impl Fn(&str, &Value) -> bool) -> bool {
        self.insertion_order.iter().all(|k| {
            self.data.get(k).map_or(false, |v| f(k, v))
        })
    }

    pub fn fold<T>(&self, init: T, f: impl Fn(T, &str, &Value) -> T) -> T {
        self.insertion_order.iter().fold(init, |acc, k| {
            self.data.get(k).map_or(acc, |v| f(acc, k, v))
        })
    }

    pub fn group_by(&self, key_fn: impl Fn(&str, &Value) -> String) -> HashMap<String, Self> {
        let mut groups: HashMap<String, Self> = HashMap::new();
        for key in &self.insertion_order {
            if let Some(value) = self.data.get(key) {
                let group_key = key_fn(key, value);
                groups.entry(group_key)
                    .or_insert_with(Self::new)
                    .insert(key.clone(), value.clone());
            }
        }
        groups
    }

    pub fn sort_by_key(&self, f: impl Fn(&str) -> String) -> Self {
        let mut sorted_keys: Vec<&String> = self.insertion_order.iter().collect();
        sorted_keys.sort_by(|a, b| f(a).cmp(&f(b)));

        let mut result = Self::new();
        for key in sorted_keys {
            if let Some(value) = self.data.get(key) {
                result.insert(key.clone(), value.clone());
            }
        }
        result
    }

    pub fn sort_by_value(&self, f: impl Fn(&Value) -> String) -> Self {
        let mut sorted_keys: Vec<&String> = self.insertion_order.iter().collect();
        sorted_keys.sort_by(|a, b| {
            let va = self.data.get(a.as_str()).map(|v| f(v)).unwrap_or_default();
            let vb = self.data.get(b.as_str()).map(|v| f(v)).unwrap_or_default();
            va.cmp(&vb)
        });

        let mut result = Self::new();
        for key in sorted_keys {
            if let Some(value) = self.data.get(key) {
                result.insert(key.clone(), value.clone());
            }
        }
        result
    }

    pub fn invert(&self) -> HashMap<String, String> {
        self.data.iter()
            .map(|(k, v)| (v.format_display(), k.clone()))
            .collect()
    }

    pub fn to_json(&self) -> String {
        let entries: Vec<String> = self.insertion_order.iter()
            .filter_map(|k| {
                self.data.get(k).map(|v| {
                    format!("\"{}\":{}", k, value_to_json(v))
                })
            })
            .collect();
        format!("{{{}}}", entries.join(","))
    }
}

fn value_to_json(value: &Value) -> String {
    match value {
        Value::None => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(value_to_json).collect();
            format!("[{}]", items.join(","))
        }
        Value::Map(map) => {
            let items: Vec<String> = map.iter()
                .map(|(k, v)| format!("\"{}\":{}", k.format_display(), value_to_json(v)))
                .collect();
            format!("{{{}}}", items.join(","))
        }
        Value::Tuple(tuple) => {
            let items: Vec<String> = tuple.iter().map(value_to_json).collect();
            format!("[{}]", items.join(","))
        }
        _ => "null".to_string(),
    }
}

impl Clone for OmegaMap {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            insertion_order: self.insertion_order.clone(),
        }
    }
}

impl std::fmt::Debug for OmegaMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OmegaMap({})", self.data.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_basic() {
        let mut map = OmegaMap::new();
        map.insert("a".to_string(), Value::Integer(1));
        map.insert("b".to_string(), Value::Integer(2));
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("a"), Some(&Value::Integer(1)));
    }

    #[test]
    fn test_map_order() {
        let mut map = OmegaMap::new();
        map.insert("c".to_string(), Value::Integer(3));
        map.insert("a".to_string(), Value::Integer(1));
        map.insert("b".to_string(), Value::Integer(2));
        let keys: Vec<&String> = map.keys().into_iter().collect();
        assert_eq!(keys, vec!["c", "a", "b"]);
    }
}
