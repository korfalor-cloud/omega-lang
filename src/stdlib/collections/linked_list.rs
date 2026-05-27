use std::collections::LinkedList;
use crate::vm::stack::Value;

pub struct OmegaLinkedList {
    data: LinkedList<Value>,
}

impl OmegaLinkedList {
    pub fn new() -> Self {
        Self { data: LinkedList::new() }
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

    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.data.iter()
    }

    pub fn append(&mut self, other: &mut Self) {
        self.data.append(&mut other.data);
    }

    pub fn split_off(&mut self, at: usize) -> Self {
        Self { data: self.data.split_off(at) }
    }
}

impl Clone for OmegaLinkedList {
    fn clone(&self) -> Self {
        Self { data: self.data.clone() }
    }
}
