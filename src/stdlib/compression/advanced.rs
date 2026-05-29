//! Advanced compression algorithms.
//!
//! This module provides implementations of various compression algorithms:
//! - Huffman coding (frequency-based variable-length encoding)
//! - LZ77 (sliding-window dictionary compression)
//! - LZ78 (sequential dictionary compression)
//! - Run-length encoding (repeated-byte compression)
//! - Delta encoding (difference-based compression)
//! - Burrows-Wheeler transform (block-sorting compression)

use std::collections::{BTreeMap, HashMap, VecDeque};

/// Huffman tree node for advanced Huffman coding.
#[derive(Debug, Clone)]
enum HuffmanNode {
    Leaf {
        byte: u8,
        freq: usize,
    },
    Internal {
        freq: usize,
        left: Box<HuffmanNode>,
        right: Box<HuffmanNode>,
    },
}

impl HuffmanNode {
    fn freq(&self) -> usize {
        match self {
            HuffmanNode::Leaf { freq, .. } => *freq,
            HuffmanNode::Internal { freq, .. } => *freq,
        }
    }
}

/// Advanced Huffman coder with canonical code generation.
pub struct HuffmanAdvanced;

impl HuffmanAdvanced {
    /// Build a Huffman tree from input data.
    fn build_tree(data: &[u8]) -> Option<HuffmanNode> {
        let mut freq_map: HashMap<u8, usize> = HashMap::new();
        for &byte in data {
            *freq_map.entry(byte).or_insert(0) += 1;
        }

        let mut nodes: Vec<HuffmanNode> = freq_map
            .into_iter()
            .map(|(byte, freq)| HuffmanNode::Leaf { byte, freq })
            .collect();

        if nodes.is_empty() {
            return None;
        }

        while nodes.len() > 1 {
            nodes.sort_by(|a, b| b.freq().cmp(&a.freq()));
            let left = nodes.pop().unwrap();
            let right = nodes.pop().unwrap();
            let merged = HuffmanNode::Internal {
                freq: left.freq() + right.freq(),
                left: Box::new(left),
                right: Box::new(right),
            };
            nodes.push(merged);
        }

        nodes.pop()
    }

    /// Generate canonical Huffman codes from the tree.
    fn build_codes(root: &HuffmanNode) -> HashMap<u8, Vec<u8>> {
        let mut codes = HashMap::new();
        let mut stack = vec![(root, Vec::new())];

        while let Some((node, path)) = stack.pop() {
            match node {
                HuffmanNode::Leaf { byte, .. } => {
                    codes.insert(*byte, if path.is_empty() { vec![0] } else { path });
                }
                HuffmanNode::Internal { left, right, .. } => {
                    let mut left_path = path.clone();
                    left_path.push(0);
                    stack.push((left, left_path));

                    let mut right_path = path;
                    right_path.push(1);
                    stack.push((right, right_path));
                }
            }
        }

        codes
    }

    /// Compress data using Huffman coding.
    pub fn compress(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let tree = match Self::build_tree(data) {
            Some(t) => t,
            None => return Vec::new(),
        };
        let codes = Self::build_codes(&tree);

        let mut output = Vec::new();
        // Store original length
        output.extend_from_slice(&(data.len() as u32).to_le_bytes());
        // Store code table
        output.push(codes.len() as u8);
        for (byte, code) in &codes {
            output.push(*byte);
            output.push(code.len() as u8);
            // Pack bits
            let mut packed = 0u8;
            let mut bit_pos = 0;
            for bit in code {
                packed |= bit << (7 - bit_pos);
                bit_pos += 1;
                if bit_pos == 8 {
                    output.push(packed);
                    packed = 0;
                    bit_pos = 0;
                }
            }
            if bit_pos > 0 {
                output.push(packed);
            }
        }

        // Encode data
        let mut current_byte = 0u8;
        let mut bit_pos = 0;
        for &byte in data {
            for &bit in &codes[&byte] {
                current_byte |= bit << (7 - bit_pos);
                bit_pos += 1;
                if bit_pos == 8 {
                    output.push(current_byte);
                    current_byte = 0;
                    bit_pos = 0;
                }
            }
        }
        if bit_pos > 0 {
            output.push(current_byte);
        }

        output
    }

    /// Decompress Huffman-coded data.
    pub fn decompress(data: &[u8]) -> Vec<u8> {
        if data.len() < 6 {
            return Vec::new();
        }

        let original_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let table_size = data[4] as usize;
        let mut pos = 5;

        // Read code table
        let mut table = Vec::new();
        for _ in 0..table_size {
            let byte = data[pos];
            let code_len = data[pos + 1] as usize;
            pos += 2;
            let bytes_needed = (code_len + 7) / 8;
            let mut code = Vec::new();
            for i in 0..bytes_needed {
                let b = data[pos + i];
                for bit in 0..8.min(code_len - i * 8) {
                    code.push((b >> (7 - bit)) & 1);
                }
            }
            pos += bytes_needed;
            code.truncate(code_len);
            table.push((byte, code));
        }

        // Build decoding lookup
        let mut output = Vec::with_capacity(original_len);
        let mut current_bits = Vec::new();

        while output.len() < original_len && pos < data.len() {
            let byte = data[pos];
            pos += 1;
            for bit in 0..8 {
                current_bits.push((byte >> (7 - bit)) & 1);
                // Try to match against table
                for (symbol, code) in &table {
                    if current_bits == *code {
                        output.push(*symbol);
                        current_bits.clear();
                        break;
                    }
                }
            }
        }

        output.truncate(original_len);
        output
    }
}

/// LZ77 sliding-window compressor.
pub struct Lz77;

impl Lz77 {
    const WINDOW_SIZE: usize = 4096;
    const LOOKAHEAD_SIZE: usize = 18;

    /// Compress data using LZ77 algorithm.
    pub fn compress(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            let search_start = pos.saturating_sub(Self::WINDOW_SIZE);
            let mut best_offset = 0u16;
            let mut best_length = 0u16;

            // Find longest match in window
            for offset in (search_start..pos).rev() {
                let mut length = 0u16;
                while pos + (length as usize) < data.len()
                    && length < Self::LOOKAHEAD_SIZE as u16
                    && data[offset + length as usize] == data[pos + length as usize]
                {
                    length += 1;
                }
                if length > best_length {
                    best_length = length;
                    best_offset = (pos - offset) as u16;
                }
            }

            if best_length >= 3 {
                // Emit (offset, length, next_byte)
                let next = if pos + best_length as usize < data.len() {
                    data[pos + best_length as usize]
                } else {
                    0
                };
                output.push(1); // flag: match
                output.extend_from_slice(&best_offset.to_le_bytes());
                output.push(best_length as u8);
                output.push(next);
                pos += best_length as usize + 1;
            } else {
                // Emit literal
                output.push(0); // flag: literal
                output.push(data[pos]);
                pos += 1;
            }
        }

        output
    }

    /// Decompress LZ77 data.
    pub fn decompress(data: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            let flag = data[pos];
            pos += 1;

            if flag == 0 {
                // Literal
                if pos < data.len() {
                    output.push(data[pos]);
                    pos += 1;
                }
            } else {
                // Match
                if pos + 3 >= data.len() {
                    break;
                }
                let offset = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                let length = data[pos + 2] as usize;
                let next = data[pos + 3];
                pos += 4;

                let start = output.len().saturating_sub(offset);
                for i in 0..length {
                    if start + i < output.len() {
                        output.push(output[start + i]);
                    }
                }
                output.push(next);
            }
        }

        output
    }
}

/// LZ78 dictionary-based compressor.
pub struct Lz78;

impl Lz78 {
    /// Compress data using LZ78 algorithm.
    pub fn compress(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::new();
        let mut dict: HashMap<Vec<u8>, u32> = HashMap::new();
        let mut current = Vec::new();
        let mut dict_size: u32 = 1;

        for &byte in data {
            current.push(byte);
            if !dict.contains_key(&current) {
                let prefix_idx = if current.len() > 1 {
                    let prefix = &current[..current.len() - 1];
                    *dict.get(prefix).unwrap_or(&0)
                } else {
                    0
                };

                output.extend_from_slice(&prefix_idx.to_le_bytes());
                output.push(byte);

                if dict_size < u32::MAX {
                    dict.insert(current.clone(), dict_size);
                    dict_size += 1;
                }
                current.clear();
            }
        }

        // Flush remaining
        if !current.is_empty() {
            let idx = *dict.get(&current).unwrap_or(&0);
            output.extend_from_slice(&idx.to_le_bytes());
            output.push(0);
        }

        output
    }

    /// Decompress LZ78 data.
    pub fn decompress(data: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let mut dict: Vec<Vec<u8>> = vec![Vec::new()]; // index 0 = empty
        let mut pos = 0;

        while pos + 4 < data.len() {
            let idx = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            let byte = data[pos + 4];
            pos += 5;

            let mut entry = if (idx as usize) < dict.len() {
                dict[idx as usize].clone()
            } else {
                Vec::new()
            };
            entry.push(byte);
            output.extend_from_slice(&entry);
            dict.push(entry);
        }

        output
    }
}

/// Run-length encoder (advanced variant with escape byte).
pub struct RunLengthAdvanced;

impl RunLengthAdvanced {
    const ESCAPE: u8 = 0xFF;

    /// Compress data using run-length encoding.
    pub fn compress(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::new();
        let mut i = 0;

        while i < data.len() {
            let byte = data[i];
            let mut count = 1u8;

            while i + (count as usize) < data.len()
                && data[i + (count as usize)] == byte
                && count < 255
            {
                count += 1;
            }

            if count >= 3 || byte == Self::ESCAPE {
                output.push(Self::ESCAPE);
                output.push(count);
                output.push(byte);
            } else {
                for _ in 0..count {
                    output.push(byte);
                }
            }

            i += count as usize;
        }

        output
    }

    /// Decompress run-length encoded data.
    pub fn decompress(data: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let mut i = 0;

        while i < data.len() {
            if data[i] == Self::ESCAPE && i + 2 < data.len() {
                let count = data[i + 1];
                let byte = data[i + 2];
                for _ in 0..count {
                    output.push(byte);
                }
                i += 3;
            } else {
                output.push(data[i]);
                i += 1;
            }
        }

        output
    }
}

/// Delta encoder for sequential numeric data.
pub struct DeltaCoder;

impl DeltaCoder {
    /// Compress data using delta encoding (byte-level differences).
    pub fn compress(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::with_capacity(data.len() + 1);
        output.push(data[0]);

        for i in 1..data.len() {
            let delta = data[i].wrapping_sub(data[i - 1]);
            output.push(delta);
        }

        output
    }

    /// Decompress delta-encoded data.
    pub fn decompress(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::with_capacity(data.len());
        output.push(data[0]);

        for i in 1..data.len() {
            let value = output[i - 1].wrapping_add(data[i]);
            output.push(value);
        }

        output
    }

    /// Compress using delta encoding with variable-length integers.
    pub fn compress_varint(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut output = Vec::new();
        output.push(data[0]);

        for i in 1..data.len() {
            let delta = data[i] as i16 - data[i - 1] as i16;
            // Encode as zigzag + varint
            let zigzag = ((delta << 1) ^ (delta >> 15)) as u16;
            Self::write_varint(&mut output, zigzag);
        }

        output
    }

    fn write_varint(output: &mut Vec<u8>, mut value: u16) {
        while value >= 0x80 {
            output.push((value & 0x7F) as u8 | 0x80);
            value >>= 7;
        }
        output.push(value as u8);
    }
}

/// Burrows-Wheeler Transform for block-sorting compression.
pub struct BurrowsWheeler;

impl BurrowsWheeler {
    /// Apply the Burrows-Wheeler transform.
    pub fn transform(data: &[u8]) -> (Vec<u8>, usize) {
        if data.is_empty() {
            return (Vec::new(), 0);
        }

        let len = data.len();
        // Create all rotations
        let mut indices: Vec<usize> = (0..len).collect();
        indices.sort_by(|&a, &b| {
            for i in 0..len {
                let byte_a = data[(a + i) % len];
                let byte_b = data[(b + i) % len];
                if byte_a != byte_b {
                    return byte_a.cmp(&byte_b);
                }
            }
            std::cmp::Ordering::Equal
        });

        // Get last column
        let last_column: Vec<u8> = indices.iter().map(|&i| data[(i + len - 1) % len]).collect();

        // Find original position
        let original_pos = indices.iter().position(|&i| i == 0).unwrap_or(0);

        (last_column, original_pos)
    }

    /// Inverse Burrows-Wheeler transform.
    pub fn inverse_transform(data: &[u8], original_pos: usize) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let len = data.len();
        // Build first column by sorting
        let mut sorted: Vec<(u8, usize)> = data.iter().copied().zip(0..len).collect();
        sorted.sort();

        // Build transformation vector
        let mut counts: BTreeMap<u8, usize> = BTreeMap::new();
        let mut t = vec![0usize; len];
        for i in 0..len {
            let count = counts.entry(data[i]).or_insert(0);
            t[i] = *count;
            *count += 1;
        }

        // Build LF mapping
        let mut lf = vec![0usize; len];
        let mut cumul: BTreeMap<u8, usize> = BTreeMap::new();
        let mut total = 0;
        for (&byte, &count) in &counts {
            cumul.insert(byte, total);
            total += count;
        }
        for i in 0..len {
            lf[i] = cumul[&data[i]] + t[i];
        }

        // Reconstruct
        let mut output = vec![0u8; len];
        let mut idx = original_pos;
        for i in (0..len).rev() {
            output[i] = data[idx];
            idx = lf[idx];
        }

        output
    }

    /// Compress using BWT + move-to-front + RLE.
    pub fn compress(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        // Apply BWT
        let (transformed, pos) = Self::transform(data);

        // Move-to-front encoding
        let mut alphabet: Vec<u8> = (0..=255).collect();
        let mut mtf_output = Vec::new();
        for &byte in &transformed {
            let idx = alphabet.iter().position(|&b| b == byte).unwrap();
            mtf_output.push(idx as u8);
            let val = alphabet.remove(idx);
            alphabet.insert(0, val);
        }

        // Pack output: [pos(4 bytes), mtf data...]
        let mut output = Vec::new();
        output.extend_from_slice(&(pos as u32).to_le_bytes());
        output.extend_from_slice(&(data.len() as u32).to_le_bytes());
        output.extend(mtf_output);

        output
    }

    /// Decompress BWT-compressed data.
    pub fn decompress(data: &[u8]) -> Vec<u8> {
        if data.len() < 8 {
            return Vec::new();
        }

        let pos = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let original_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let mtf_data = &data[8..];

        // Inverse move-to-front
        let mut alphabet: Vec<u8> = (0..=255).collect();
        let mut transformed = Vec::new();
        for &idx in mtf_data {
            let byte = alphabet[idx as usize];
            transformed.push(byte);
            let val = alphabet.remove(idx as usize);
            alphabet.insert(0, val);
        }

        // Inverse BWT
        Self::inverse_transform(&transformed, pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_roundtrip() {
        let data = b"aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnnooopppqqqrrrssstttuuuvvvwwwxxxyyyyzzz";
        let compressed = HuffmanAdvanced::compress(data);
        let decompressed = HuffmanAdvanced::decompress(&compressed);
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_huffman_empty() {
        assert!(HuffmanAdvanced::compress(b"").is_empty());
        assert!(HuffmanAdvanced::decompress(&[]).is_empty());
    }

    #[test]
    fn test_huffman_single_byte() {
        let data = b"a";
        let compressed = HuffmanAdvanced::compress(data);
        let decompressed = HuffmanAdvanced::decompress(&compressed);
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_lz77_roundtrip() {
        let data = b"the quick brown fox jumps over the lazy dog the quick brown fox";
        let compressed = Lz77::compress(data);
        let decompressed = Lz77::decompress(&compressed);
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_lz77_empty() {
        assert!(Lz77::compress(b"").is_empty());
    }

    #[test]
    fn test_lz77_no_repetition() {
        let data = b"abcdefghij";
        let compressed = Lz77::compress(data);
        let decompressed = Lz77::decompress(&compressed);
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_lz78_roundtrip() {
        let data = b"abracadabra abracadabra";
        let compressed = Lz78::compress(data);
        let decompressed = Lz78::decompress(&compressed);
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_lz78_empty() {
        assert!(Lz78::compress(b"").is_empty());
    }

    #[test]
    fn test_run_length_advanced_roundtrip() {
        let data = b"aaabbbcccdddeee11111111111111111111";
        let compressed = RunLengthAdvanced::compress(data);
        let decompressed = RunLengthAdvanced::decompress(&compressed);
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_run_length_advanced_empty() {
        assert!(RunLengthAdvanced::compress(b"").is_empty());
    }

    #[test]
    fn test_delta_coder_roundtrip() {
        let data = vec![10, 12, 15, 20, 18, 25, 30, 28];
        let compressed = DeltaCoder::compress(&data);
        let decompressed = DeltaCoder::decompress(&compressed);
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_delta_coder_empty() {
        assert!(DeltaCoder::compress(&[]).is_empty());
        assert!(DeltaCoder::decompress(&[]).is_empty());
    }

    #[test]
    fn test_delta_coder_wrapping() {
        let data = vec![250, 5, 200, 10];
        let compressed = DeltaCoder::compress(&data);
        let decompressed = DeltaCoder::decompress(&compressed);
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_bwt_transform_inverse() {
        let data = b"banana";
        let (transformed, pos) = BurrowsWheeler::transform(data);
        let recovered = BurrowsWheeler::inverse_transform(&transformed, pos);
        assert_eq!(data.to_vec(), recovered);
    }

    #[test]
    fn test_bwt_compress_roundtrip() {
        let data = b"the burrows-wheeler transform is a block-sorting compression algorithm";
        let compressed = BurrowsWheeler::compress(data);
        let decompressed = BurrowsWheeler::decompress(&compressed);
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_bwt_empty() {
        let (transformed, pos) = BurrowsWheeler::transform(b"");
        assert!(transformed.is_empty());
        assert_eq!(pos, 0);
    }

    #[test]
    fn test_delta_varint_roundtrip() {
        let data = vec![100, 102, 105, 110, 108, 112];
        let compressed = DeltaCoder::compress_varint(&data);
        let decompressed = DeltaCoder::decompress(&compressed);
        assert_eq!(data, decompressed);
    }
}
