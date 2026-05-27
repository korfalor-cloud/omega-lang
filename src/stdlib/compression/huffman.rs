use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;

#[derive(Debug, Clone)]
struct HuffmanNode {
    char: Option<u8>,
    frequency: usize,
    left: Option<Box<HuffmanNode>>,
    right: Option<Box<HuffmanNode>>,
}

impl HuffmanNode {
    fn new_leaf(char: u8, frequency: usize) -> Self {
        Self {
            char: Some(char),
            frequency,
            left: None,
            right: None,
        }
    }

    fn new_internal(left: HuffmanNode, right: HuffmanNode) -> Self {
        Self {
            char: None,
            frequency: left.frequency + right.frequency,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }

    fn is_leaf(&self) -> bool {
        self.char.is_some()
    }
}

impl PartialEq for HuffmanNode {
    fn eq(&self, other: &Self) -> bool {
        self.frequency == other.frequency
    }
}

impl Eq for HuffmanNode {}

impl PartialOrd for HuffmanNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HuffmanNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.frequency.cmp(&other.frequency)
    }
}

pub struct HuffmanCoder {
    codes: HashMap<u8, String>,
    tree: Option<HuffmanNode>,
}

impl HuffmanCoder {
    pub fn new() -> Self {
        Self {
            codes: HashMap::new(),
            tree: None,
        }
    }

    pub fn build(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        // Count frequencies
        let mut frequencies: HashMap<u8, usize> = HashMap::new();
        for &byte in data {
            *frequencies.entry(byte).or_insert(0) += 1;
        }

        // Build priority queue
        let mut heap: BinaryHeap<Reverse<HuffmanNode>> = BinaryHeap::new();
        for (&char, &freq) in &frequencies {
            heap.push(Reverse(HuffmanNode::new_leaf(char, freq)));
        }

        // Build tree
        while heap.len() > 1 {
            let Reverse(left) = heap.pop().unwrap();
            let Reverse(right) = heap.pop().unwrap();
            let parent = HuffmanNode::new_internal(left, right);
            heap.push(Reverse(parent));
        }

        if let Some(Reverse(root)) = heap.pop() {
            self.codes.clear();
            self.build_codes(&root, String::new());
            self.tree = Some(root);
        }
    }

    fn build_codes(&mut self, node: &HuffmanNode, code: String) {
        if let Some(char) = node.char {
            self.codes.insert(char, code);
        } else {
            if let Some(ref left) = node.left {
                self.build_codes(left, format!("{}0", code));
            }
            if let Some(ref right) = node.right {
                self.build_codes(right, format!("{}1", code));
            }
        }
    }

    pub fn encode(&self, data: &[u8]) -> Vec<u8> {
        let mut bits = String::new();

        for &byte in data {
            if let Some(code) = self.codes.get(&byte) {
                bits.push_str(code);
            }
        }

        // Convert bits to bytes
        let mut result = Vec::new();
        for chunk in bits.as_bytes().chunks(8) {
            let s = std::str::from_utf8(chunk).unwrap_or("0");
            let padded = format!("{:0<8}", s);
            if let Ok(byte) = u8::from_str_radix(&padded, 2) {
                result.push(byte);
            }
        }

        // Store original length for decoding
        let len = data.len() as u32;
        result.extend_from_slice(&len.to_be_bytes());

        result
    }

    pub fn decode(&self, encoded: &[u8]) -> Vec<u8> {
        if encoded.len() < 4 {
            return Vec::new();
        }

        // Extract original length
        let len_bytes = &encoded[encoded.len() - 4..];
        let original_len = u32::from_be_bytes([
            len_bytes[0],
            len_bytes[1],
            len_bytes[2],
            len_bytes[3],
        ]) as usize;

        let encoded_data = &encoded[..encoded.len() - 4];

        // Convert bytes to bits
        let mut bits = String::new();
        for &byte in encoded_data {
            bits.push_str(&format!("{:08b}", byte));
        }

        // Decode using tree
        let mut result = Vec::new();
        if let Some(ref tree) = self.tree {
            let mut current = tree;

            for bit in bits.chars() {
                if current.is_leaf() {
                    result.push(current.char.unwrap());
                    current = tree;
                    if result.len() >= original_len {
                        break;
                    }
                }

                match bit {
                    '0' => {
                        if let Some(ref left) = current.left {
                            current = left;
                        }
                    }
                    '1' => {
                        if let Some(ref right) = current.right {
                            current = right;
                        }
                    }
                    _ => {}
                }
            }

            if current.is_leaf() && result.len() < original_len {
                result.push(current.char.unwrap());
            }
        }

        result
    }

    pub fn codes(&self) -> &HashMap<u8, String> {
        &self.codes
    }

    pub fn compression_ratio(&self, original: &[u8], compressed: &[u8]) -> f64 {
        if original.is_empty() {
            return 0.0;
        }
        compressed.len() as f64 / original.len() as f64
    }
}

// Frequency analysis
pub fn frequency_analysis(data: &[u8]) -> HashMap<u8, usize> {
    let mut frequencies: HashMap<u8, usize> = HashMap::new();
    for &byte in data {
        *frequencies.entry(byte).or_insert(0) += 1;
    }
    frequencies
}

pub fn sorted_frequencies(data: &[u8]) -> Vec<(u8, usize)> {
    let mut freqs: Vec<(u8, usize)> = frequency_analysis(data).into_iter().collect();
    freqs.sort_by(|a, b| b.1.cmp(&a.1));
    freqs
}
