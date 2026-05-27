use std::collections::VecDeque;
use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;

pub struct OmegaDeque {
    data: VecDeque<Value>,
}

impl OmegaDeque {
    pub fn new() -> Self {
        Self { data: VecDeque::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { data: VecDeque::with_capacity(capacity) }
    }

    pub fn push_front(&mut self, value: Value) {
        self.data.push_front(value);
    }

    pub fn push_back(&mut self, value: Value) {
        self.data.push_back(value);
    }

    pub fn pop_front(&mut self) -> Option<Value> {
        self.data.pop_front()
    }

    pub fn pop_back(&mut self) -> Option<Value> {
        self.data.pop_back()
    }

    pub fn front(&self) -> Option<&Value> {
        self.data.front()
    }

    pub fn back(&self) -> Option<&Value> {
        self.data.back()
    }

    pub fn get(&self, index: usize) -> Option<&Value> {
        self.data.get(index)
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

    pub fn rotate_left(&mut self, n: usize) {
        self.data.rotate_left(n);
    }

    pub fn rotate_right(&mut self, n: usize) {
        self.data.rotate_right(n);
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.data.swap(a, b);
    }

    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.data.iter()
    }

    pub fn to_vec(&self) -> Vec<Value> {
        self.data.iter().cloned().collect()
    }
}

impl Clone for OmegaDeque {
    fn clone(&self) -> Self {
        Self { data: self.data.clone() }
    }
}
