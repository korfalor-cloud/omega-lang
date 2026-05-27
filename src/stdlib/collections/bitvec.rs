pub struct OmegaBitVec {
    data: Vec<u64>,
    len: usize,
}

impl OmegaBitVec {
    pub fn new() -> Self {
        Self { data: Vec::new(), len: 0 }
    }

    pub fn with_capacity(bits: usize) -> Self {
        Self {
            data: Vec::with_capacity((bits + 63) / 64),
            len: 0,
        }
    }

    pub fn push(&mut self, bit: bool) {
        let word_index = self.len / 64;
        let bit_index = self.len % 64;
        if word_index >= self.data.len() {
            self.data.push(0);
        }
        if bit {
            self.data[word_index] |= 1 << bit_index;
        }
        self.len += 1;
    }

    pub fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len {
            return None;
        }
        let word_index = index / 64;
        let bit_index = index % 64;
        self.data.get(word_index).map(|word| (word >> bit_index) & 1 == 1)
    }

    pub fn set(&mut self, index: usize, bit: bool) {
        if index >= self.len {
            return;
        }
        let word_index = index / 64;
        let bit_index = index % 64;
        if bit {
            self.data[word_index] |= 1 << bit_index;
        } else {
            self.data[word_index] &= !(1 << bit_index);
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.len = 0;
    }

    pub fn count_ones(&self) -> usize {
        self.data.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn count_zeros(&self) -> usize {
        self.len - self.count_ones()
    }

    pub fn and(&self, other: &Self) -> Self {
        let len = self.len.min(other.len);
        let mut result = Self::with_capacity(len);
        for i in 0..(len + 63) / 64 {
            let a = self.data.get(i).copied().unwrap_or(0);
            let b = other.data.get(i).copied().unwrap_or(0);
            result.data.push(a & b);
        }
        result.len = len;
        result
    }

    pub fn or(&self, other: &Self) -> Self {
        let len = self.len.max(other.len);
        let mut result = Self::with_capacity(len);
        for i in 0..(len + 63) / 64 {
            let a = self.data.get(i).copied().unwrap_or(0);
            let b = other.data.get(i).copied().unwrap_or(0);
            result.data.push(a | b);
        }
        result.len = len;
        result
    }

    pub fn xor(&self, other: &Self) -> Self {
        let len = self.len.max(other.len);
        let mut result = Self::with_capacity(len);
        for i in 0..(len + 63) / 64 {
            let a = self.data.get(i).copied().unwrap_or(0);
            let b = other.data.get(i).copied().unwrap_or(0);
            result.data.push(a ^ b);
        }
        result.len = len;
        result
    }

    pub fn not(&self) -> Self {
        let mut result = Self::with_capacity(self.len);
        for word in &self.data {
            result.data.push(!word);
        }
        result.len = self.len;
        result
    }

    pub fn shift_left(&self, n: usize) -> Self {
        let mut result = Self::with_capacity(self.len + n);
        let word_shift = n / 64;
        let bit_shift = n % 64;
        for i in 0..(self.len + 63) / 64 {
            let word = self.data.get(i).copied().unwrap_or(0);
            if i + word_shift < result.data.len() {
                result.data[i + word_shift] |= word << bit_shift;
            }
            if bit_shift > 0 && i + word_shift + 1 < result.data.len() {
                result.data[i + word_shift + 1] |= word >> (64 - bit_shift);
            }
        }
        result.len = self.len + n;
        result
    }

    pub fn shift_right(&self, n: usize) -> Self {
        if n >= self.len {
            return Self::new();
        }
        let mut result = Self::with_capacity(self.len - n);
        let word_shift = n / 64;
        let bit_shift = n % 64;
        for i in word_shift..(self.len + 63) / 64 {
            let word = self.data.get(i).copied().unwrap_or(0);
            let mut new_word = word >> bit_shift;
            if bit_shift > 0 && i + 1 < self.data.len() {
                new_word |= self.data[i + 1] << (64 - bit_shift);
            }
            result.data.push(new_word);
        }
        result.len = self.len - n;
        result
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.data.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut data = Vec::new();
        for chunk in bytes.chunks(8) {
            let mut bytes = [0u8; 8];
            for (i, &b) in chunk.iter().enumerate() {
                bytes[i] = b;
            }
            data.push(u64::from_le_bytes(bytes));
        }
        let len = bytes.len() * 8;
        Self { data, len }
    }
}

impl Clone for OmegaBitVec {
    fn clone(&self) -> Self {
        Self { data: self.data.clone(), len: self.len }
    }
}
