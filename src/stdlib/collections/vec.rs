use std::collections::HashMap;
use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;

/// Omega Vector - a dynamic array implementation
pub struct OmegaVec {
    data: Vec<Value>,
}

impl OmegaVec {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { data: Vec::with_capacity(capacity) }
    }

    pub fn from_values(values: Vec<Value>) -> Self {
        Self { data: values }
    }

    pub fn push(&mut self, value: Value) {
        self.data.push(value);
    }

    pub fn pop(&mut self) -> Option<Value> {
        self.data.pop()
    }

    pub fn get(&self, index: i64) -> OmegaResult<&Value> {
        let idx = self.normalize_index(index)?;
        self.data.get(idx).ok_or_else(|| OmegaError::IndexOutOfBounds {
            index,
            length: self.data.len(),
        })
    }

    pub fn set(&mut self, index: i64, value: Value) -> OmegaResult<()> {
        let idx = self.normalize_index(index)?;
        if idx < self.data.len() {
            self.data[idx] = value;
            Ok(())
        } else {
            Err(OmegaError::IndexOutOfBounds {
                index,
                length: self.data.len(),
            })
        }
    }

    pub fn insert(&mut self, index: i64, value: Value) -> OmegaResult<()> {
        let idx = self.normalize_index(index)?;
        if idx <= self.data.len() {
            self.data.insert(idx, value);
            Ok(())
        } else {
            Err(OmegaError::IndexOutOfBounds {
                index,
                length: self.data.len(),
            })
        }
    }

    pub fn remove(&mut self, index: i64) -> OmegaResult<Value> {
        let idx = self.normalize_index(index)?;
        if idx < self.data.len() {
            Ok(self.data.remove(idx))
        } else {
            Err(OmegaError::IndexOutOfBounds {
                index,
                length: self.data.len(),
            })
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
    }

    pub fn contains(&self, value: &Value) -> bool {
        self.data.contains(value)
    }

    pub fn first(&self) -> Option<&Value> {
        self.data.first()
    }

    pub fn last(&self) -> Option<&Value> {
        self.data.last()
    }

    pub fn iter(&self) -> std::slice::Iter<Value> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Value> {
        self.data.iter_mut()
    }

    pub fn sort(&mut self) {
        self.data.sort_by(|a, b| {
            match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
                (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                (Value::String(a), Value::String(b)) => a.cmp(b),
                _ => std::cmp::Ordering::Equal,
            }
        });
    }

    pub fn sort_by(&mut self, compare: impl Fn(&Value, &Value) -> std::cmp::Ordering) {
        self.data.sort_by(compare);
    }

    pub fn reverse(&mut self) {
        self.data.reverse();
    }

    pub fn dedup(&mut self) {
        self.data.dedup();
    }

    pub fn dedup_by(&mut self, same_bucket: impl Fn(&Value, &Value) -> bool) {
        self.data.dedup_by(same_bucket);
    }

    pub fn binary_search(&self, value: &Value) -> Result<usize, usize> {
        self.data.binary_search_by(|probe| {
            match (probe, value) {
                (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
                (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                (Value::String(a), Value::String(b)) => a.cmp(b),
                _ => std::cmp::Ordering::Equal,
            }
        })
    }

    pub fn rotate_left(&mut self, mid: usize) {
        self.data.rotate_left(mid);
    }

    pub fn rotate_right(&mut self, mid: usize) {
        self.data.rotate_right(mid);
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.data.swap(a, b);
    }

    pub fn fill(&mut self, value: Value) {
        self.data.fill(value);
    }

    pub fn resize(&mut self, new_len: usize, value: Value) {
        self.data.resize(new_len, value);
    }

    pub fn truncate(&mut self, len: usize) {
        self.data.truncate(len);
    }

    pub fn split_off(&mut self, at: usize) -> Self {
        Self { data: self.data.split_off(at) }
    }

    pub fn append(&mut self, other: &mut Self) {
        self.data.append(&mut other.data);
    }

    pub fn extend(&mut self, other: impl IntoIterator<Item = Value>) {
        self.data.extend(other);
    }

    pub fn map(&self, f: impl Fn(&Value) -> Value) -> Self {
        Self { data: self.data.iter().map(f).collect() }
    }

    pub fn filter(&self, f: impl Fn(&Value) -> bool) -> Self {
        Self { data: self.data.iter().filter(|v| f(v)).cloned().collect() }
    }

    pub fn fold(&self, init: Value, f: impl Fn(Value, &Value) -> Value) -> Value {
        self.data.iter().fold(init, |acc, v| f(acc, v))
    }

    pub fn reduce(&self, f: impl Fn(&Value, &Value) -> Value) -> Option<Value> {
        let mut iter = self.data.iter();
        let first = iter.next()?.clone();
        Some(iter.fold(first, |acc, v| f(&acc, v)))
    }

    pub fn find(&self, f: impl Fn(&Value) -> bool) -> Option<&Value> {
        self.data.iter().find(|v| f(v))
    }

    pub fn find_index(&self, f: impl Fn(&Value) -> bool) -> Option<usize> {
        self.data.iter().position(|v| f(v))
    }

    pub fn every(&self, f: impl Fn(&Value) -> bool) -> bool {
        self.data.iter().all(|v| f(v))
    }

    pub fn some(&self, f: impl Fn(&Value) -> bool) -> bool {
        self.data.iter().any(|v| f(v))
    }

    pub fn flatten(&self) -> Self {
        let mut result = Vec::new();
        for value in &self.data {
            if let Value::Array(arr) = value {
                result.extend(arr.iter().cloned());
            } else {
                result.push(value.clone());
            }
        }
        Self { data: result }
    }

    pub fn flat_map(&self, f: impl Fn(&Value) -> Vec<Value>) -> Self {
        let mut result = Vec::new();
        for value in &self.data {
            result.extend(f(value));
        }
        Self { data: result }
    }

    pub fn zip(&self, other: &Self) -> Self {
        let len = self.data.len().min(other.data.len());
        let data = (0..len)
            .map(|i| Value::Tuple(vec![self.data[i].clone(), other.data[i].clone()]))
            .collect();
        Self { data }
    }

    pub fn unzip(&self) -> (Self, Self) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for value in &self.data {
            if let Value::Tuple(tuple) = value {
                if tuple.len() >= 2 {
                    left.push(tuple[0].clone());
                    right.push(tuple[1].clone());
                }
            }
        }
        (Self { data: left }, Self { data: right })
    }

    pub fn windows(&self, size: usize) -> Vec<Self> {
        self.data.windows(size)
            .map(|w| Self { data: w.to_vec() })
            .collect()
    }

    pub fn chunks(&self, size: usize) -> Vec<Self> {
        self.data.chunks(size)
            .map(|c| Self { data: c.to_vec() })
            .collect()
    }

    pub fn group_by(&self, key_fn: impl Fn(&Value) -> Value) -> HashMap<String, Self> {
        let mut groups: HashMap<String, Vec<Value>> = HashMap::new();
        for value in &self.data {
            let key = key_fn(value).format_display();
            groups.entry(key).or_default().push(value.clone());
        }
        groups.into_iter().map(|(k, v)| (k, Self { data: v })).collect()
    }

    pub fn unique(&self) -> Self {
        let mut seen = Vec::new();
        let mut result = Vec::new();
        for value in &self.data {
            let key = value.format_display();
            if !seen.contains(&key) {
                seen.push(key);
                result.push(value.clone());
            }
        }
        Self { data: result }
    }

    pub fn count(&self, value: &Value) -> usize {
        self.data.iter().filter(|v| *v == value).count()
    }

    pub fn sum(&self) -> Value {
        let mut sum = 0i64;
        let mut float_sum = 0.0f64;
        let mut is_float = false;

        for value in &self.data {
            match value {
                Value::Integer(n) => sum += n,
                Value::Float(f) => {
                    is_float = true;
                    float_sum += f;
                }
                _ => {}
            }
        }

        if is_float {
            Value::Float(float_sum + sum as f64)
        } else {
            Value::Integer(sum)
        }
    }

    pub fn product(&self) -> Value {
        let mut product = 1i64;
        let mut float_product = 1.0f64;
        let mut is_float = false;

        for value in &self.data {
            match value {
                Value::Integer(n) => product *= n,
                Value::Float(f) => {
                    is_float = true;
                    float_product *= f;
                }
                _ => {}
            }
        }

        if is_float {
            Value::Float(float_product * product as f64)
        } else {
            Value::Integer(product)
        }
    }

    pub fn min(&self) -> Option<&Value> {
        self.data.iter().min_by(|a, b| {
            match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
                (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                _ => std::cmp::Ordering::Equal,
            }
        })
    }

    pub fn max(&self) -> Option<&Value> {
        self.data.iter().max_by(|a, b| {
            match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
                (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                _ => std::cmp::Ordering::Equal,
            }
        })
    }

    pub fn average(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        let sum: f64 = self.data.iter().map(|v| match v {
            Value::Integer(n) => *n as f64,
            Value::Float(f) => *f,
            _ => 0.0,
        }).sum();
        Some(sum / self.data.len() as f64)
    }

    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.data.shuffle(&mut rng);
    }

    pub fn sample(&self, n: usize) -> Self {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut data = self.data.clone();
        data.shuffle(&mut rng);
        data.truncate(n);
        Self { data }
    }

    pub fn partition(&self, f: impl Fn(&Value) -> bool) -> (Self, Self) {
        let mut true_vec = Vec::new();
        let mut false_vec = Vec::new();
        for value in &self.data {
            if f(value) {
                true_vec.push(value.clone());
            } else {
                false_vec.push(value.clone());
            }
        }
        (Self { data: true_vec }, Self { data: false_vec })
    }

    pub fn take(&self, n: usize) -> Self {
        Self { data: self.data.iter().take(n).cloned().collect() }
    }

    pub fn skip(&self, n: usize) -> Self {
        Self { data: self.data.iter().skip(n).cloned().collect() }
    }

    pub fn take_while(&self, f: impl Fn(&Value) -> bool) -> Self {
        Self { data: self.data.iter().take_while(|v| f(v)).cloned().collect() }
    }

    pub fn skip_while(&self, f: impl Fn(&Value) -> bool) -> Self {
        Self { data: self.data.iter().skip_while(|v| f(v)).cloned().collect() }
    }

    pub fn enumerate(&self) -> Self {
        let data = self.data.iter().enumerate()
            .map(|(i, v)| Value::Tuple(vec![Value::Integer(i as i64), v.clone()]))
            .collect();
        Self { data }
    }

    pub fn intersperse(&self, separator: &Value) -> Self {
        let mut result = Vec::new();
        for (i, value) in self.data.iter().enumerate() {
            if i > 0 {
                result.push(separator.clone());
            }
            result.push(value.clone());
        }
        Self { data: result }
    }

    pub fn join(&self, separator: &str) -> String {
        self.data.iter()
            .map(|v| v.format_display())
            .collect::<Vec<_>>()
            .join(separator)
    }

    fn normalize_index(&self, index: i64) -> OmegaResult<usize> {
        if index < 0 {
            let idx = self.data.len() as i64 + index;
            if idx < 0 {
                Err(OmegaError::IndexOutOfBounds {
                    index,
                    length: self.data.len(),
                })
            } else {
                Ok(idx as usize)
            }
        } else {
            Ok(index as usize)
        }
    }
}

impl Clone for OmegaVec {
    fn clone(&self) -> Self {
        Self { data: self.data.clone() }
    }
}

impl std::fmt::Debug for OmegaVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OmegaVec({})", self.data.len())
    }
}

impl FromIterator<Value> for OmegaVec {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        Self { data: iter.into_iter().collect() }
    }
}

impl IntoIterator for OmegaVec {
    type Item = Value;
    type IntoIter = std::vec::IntoIter<Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_basic() {
        let mut v = OmegaVec::new();
        v.push(Value::Integer(1));
        v.push(Value::Integer(2));
        v.push(Value::Integer(3));
        assert_eq!(v.len(), 3);
        assert_eq!(v.get(0), Ok(&Value::Integer(1)));
    }

    #[test]
    fn test_vec_negative_index() {
        let mut v = OmegaVec::new();
        v.push(Value::Integer(1));
        v.push(Value::Integer(2));
        v.push(Value::Integer(3));
        assert_eq!(v.get(-1), Ok(&Value::Integer(3)));
        assert_eq!(v.get(-2), Ok(&Value::Integer(2)));
    }

    #[test]
    fn test_vec_sort() {
        let mut v = OmegaVec::from_values(vec![
            Value::Integer(3),
            Value::Integer(1),
            Value::Integer(2),
        ]);
        v.sort();
        assert_eq!(v.get(0), Ok(&Value::Integer(1)));
        assert_eq!(v.get(1), Ok(&Value::Integer(2)));
        assert_eq!(v.get(2), Ok(&Value::Integer(3)));
    }

    #[test]
    fn test_vec_map_filter() {
        let v = OmegaVec::from_values(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]);
        let mapped = v.map(|v| match v {
            Value::Integer(n) => Value::Integer(n * 2),
            _ => v.clone(),
        });
        assert_eq!(mapped.get(0), Ok(&Value::Integer(2)));
        assert_eq!(mapped.get(1), Ok(&Value::Integer(4)));
        assert_eq!(mapped.get(2), Ok(&Value::Integer(6)));
    }
}
