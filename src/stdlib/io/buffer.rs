use crate::vm::stack::Value;

pub struct OmegaBuffer {
    data: Vec<u8>,
    position: usize,
}

impl OmegaBuffer {
    pub fn new() -> Self {
        Self { data: Vec::new(), position: 0 }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { data: Vec::with_capacity(capacity), position: 0 }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self { data: bytes.to_vec(), position: 0 }
    }

    pub fn write(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }

    pub fn write_byte(&mut self, byte: u8) {
        self.data.push(byte);
    }

    pub fn read(&mut self, count: usize) -> Vec<u8> {
        let end = (self.position + count).min(self.data.len());
        let result = self.data[self.position..end].to_vec();
        self.position = end;
        result
    }

    pub fn read_byte(&mut self) -> Option<u8> {
        if self.position < self.data.len() {
            let byte = self.data[self.position];
            self.position += 1;
            Some(byte)
        } else {
            None
        }
    }

    pub fn peek(&self) -> Option<u8> {
        self.data.get(self.position).copied()
    }

    pub fn seek(&mut self, position: usize) {
        self.position = position.min(self.data.len());
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.position = 0;
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.data.clone()
    }

    pub fn to_string(&self) -> String {
        String::from_utf8_lossy(&self.data).to_string()
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

impl Clone for OmegaBuffer {
    fn clone(&self) -> Self {
        Self { data: self.data.clone(), position: self.position }
    }
}
