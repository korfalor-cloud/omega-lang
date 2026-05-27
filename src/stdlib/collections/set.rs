use std::collections::HashSet;
use crate::vm::stack::Value;

pub struct OmegaSet {
    data: HashSet<String>,
    values: Vec<Value>,
}

impl OmegaSet {
    pub fn new() -> Self {
        Self {
            data: HashSet::new(),
            values: Vec::new(),
        }
    }

    pub fn insert(&mut self, value: Value) -> bool {
        let key = value.format_display();
        if self.data.insert(key) {
            self.values.push(value);
            true
        } else {
            false
        }
    }

    pub fn contains(&self, value: &Value) -> bool {
        self.data.contains(&value.format_display())
    }

    pub fn remove(&mut self, value: &Value) -> bool {
        let key = value.format_display();
        if self.data.remove(&key) {
            self.values.retain(|v| v.format_display() != key);
            true
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.values.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.values.iter()
    }

    pub fn union(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for value in &other.values {
            result.insert(value.clone());
        }
        result
    }

    pub fn intersection(&self, other: &Self) -> Self {
        let mut result = Self::new();
        for value in &self.values {
            if other.contains(value) {
                result.insert(value.clone());
            }
        }
        result
    }

    pub fn difference(&self, other: &Self) -> Self {
        let mut result = Self::new();
        for value in &self.values {
            if !other.contains(value) {
                result.insert(value.clone());
            }
        }
        result
    }

    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let mut result = Self::new();
        for value in &self.values {
            if !other.contains(value) {
                result.insert(value.clone());
            }
        }
        for value in &other.values {
            if !self.contains(value) {
                result.insert(value.clone());
            }
        }
        result
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        self.values.iter().all(|v| other.contains(v))
    }

    pub fn is_superset(&self, other: &Self) -> bool {
        other.values.iter().all(|v| self.contains(v))
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.values.iter().all(|v| !other.contains(v))
    }

    pub fn to_vec(&self) -> Vec<Value> {
        self.values.clone()
    }
}

impl Clone for OmegaSet {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            values: self.values.clone(),
        }
    }
}

impl std::fmt::Debug for OmegaSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OmegaSet({})", self.data.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_basic() {
        let mut set = OmegaSet::new();
        assert!(set.insert(Value::Integer(1)));
        assert!(set.insert(Value::Integer(2)));
        assert!(!set.insert(Value::Integer(1)));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_set_operations() {
        let mut a = OmegaSet::new();
        a.insert(Value::Integer(1));
        a.insert(Value::Integer(2));
        a.insert(Value::Integer(3));

        let mut b = OmegaSet::new();
        b.insert(Value::Integer(2));
        b.insert(Value::Integer(3));
        b.insert(Value::Integer(4));

        let union = a.union(&b);
        assert_eq!(union.len(), 4);

        let intersection = a.intersection(&b);
        assert_eq!(intersection.len(), 2);

        let difference = a.difference(&b);
        assert_eq!(difference.len(), 1);
    }
}
