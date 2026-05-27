use crate::vm::stack::Value;

pub struct OmegaRingBuffer {
    data: Vec<Option<Value>>,
    capacity: usize,
    head: usize,
    tail: usize,
    len: usize,
}

impl OmegaRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: (0..capacity).map(|_| None).collect(),
            capacity,
            head: 0,
            tail: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, value: Value) -> Option<Value> {
        let old = self.data[self.tail].take();
        self.data[self.tail] = Some(value);
        self.tail = (self.tail + 1) % self.capacity;
        if self.len == self.capacity {
            self.head = (self.head + 1) % self.capacity;
            old
        } else {
            self.len += 1;
            None
        }
    }

    pub fn pop(&mut self) -> Option<Value> {
        if self.len == 0 {
            return None;
        }
        let value = self.data[self.head].take();
        self.head = (self.head + 1) % self.capacity;
        self.len -= 1;
        value
    }

    pub fn peek(&self) -> Option<&Value> {
        if self.len == 0 {
            return None;
        }
        self.data[self.head].as_ref()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    pub fn clear(&mut self) {
        for item in &mut self.data {
            *item = None;
        }
        self.head = 0;
        self.tail = 0;
        self.len = 0;
    }

    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        let mut items = Vec::new();
        let mut pos = self.head;
        for _ in 0..self.len {
            if let Some(ref value) = self.data[pos] {
                items.push(value);
            }
            pos = (pos + 1) % self.capacity;
        }
        items.into_iter()
    }

    pub fn to_vec(&self) -> Vec<Value> {
        self.iter().cloned().collect()
    }
}

impl Clone for OmegaRingBuffer {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            capacity: self.capacity,
            head: self.head,
            tail: self.tail,
            len: self.len,
        }
    }
}
