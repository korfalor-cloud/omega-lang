use std::collections::BinaryHeap;
use std::cmp::Reverse;
use crate::vm::stack::Value;

pub struct OmegaHeap {
    data: BinaryHeap<i64>,
    min_heap: BinaryHeap<Reverse<i64>>,
    is_min: bool,
}

impl OmegaHeap {
    pub fn new() -> Self {
        Self {
            data: BinaryHeap::new(),
            min_heap: BinaryHeap::new(),
            is_min: false,
        }
    }

    pub fn new_min() -> Self {
        Self {
            data: BinaryHeap::new(),
            min_heap: BinaryHeap::new(),
            is_min: true,
        }
    }

    pub fn push(&mut self, value: i64) {
        if self.is_min {
            self.min_heap.push(Reverse(value));
        } else {
            self.data.push(value);
        }
    }

    pub fn pop(&mut self) -> Option<i64> {
        if self.is_min {
            self.min_heap.pop().map(|Reverse(v)| v)
        } else {
            self.data.pop()
        }
    }

    pub fn peek(&self) -> Option<&i64> {
        if self.is_min {
            self.min_heap.peek().map(|Reverse(v)| v)
        } else {
            self.data.peek()
        }
    }

    pub fn len(&self) -> usize {
        if self.is_min {
            self.min_heap.len()
        } else {
            self.data.len()
        }
    }

    pub fn is_empty(&self) -> bool {
        if self.is_min {
            self.min_heap.is_empty()
        } else {
            self.data.is_empty()
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.min_heap.clear();
    }
}
