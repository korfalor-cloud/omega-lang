pub struct RunLengthCoder;

impl RunLengthCoder {
    pub fn encode(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < data.len() {
            let current = data[i];
            let mut count = 1u8;

            while i + count as usize < data.len()
                && data[i + count as usize] == current
                && count < 255
            {
                count += 1;
            }

            result.push(count);
            result.push(current);
            i += count as usize;
        }

        result
    }

    pub fn decode(encoded: &[u8]) -> Vec<u8> {
        if encoded.len() % 2 != 0 {
            return Vec::new();
        }

        let mut result = Vec::new();

        for chunk in encoded.chunks(2) {
            let count = chunk[0];
            let byte = chunk[1];

            for _ in 0..count {
                result.push(byte);
            }
        }

        result
    }

    pub fn encode_string(s: &str) -> String {
        let bytes = s.as_bytes();
        let encoded = Self::encode(bytes);

        let mut result = String::new();
        for chunk in encoded.chunks(2) {
            result.push_str(&format!("{}{}", chunk[0], chunk[1] as char));
        }

        result
    }

    pub fn decode_string(s: &str) -> String {
        let mut encoded = Vec::new();
        let mut chars = s.chars();

        while let Some(count_char) = chars.next() {
            if let Some(byte_char) = chars.next() {
                if let Some(count) = count_char.to_digit(10) {
                    encoded.push(count as u8);
                    encoded.push(byte_char as u8);
                }
            }
        }

        let decoded = Self::decode(&encoded);
        String::from_utf8_lossy(&decoded).to_string()
    }

    pub fn compression_ratio(original: &[u8], encoded: &[u8]) -> f64 {
        if original.is_empty() {
            return 0.0;
        }
        encoded.len() as f64 / original.len() as f64
    }
}

// RLE for strings with longer runs
pub fn rle_encode_verbose(data: &[u8]) -> String {
    if data.is_empty() {
        return String::new();
    }

    let mut result = String::new();
    let mut i = 0;

    while i < data.len() {
        let current = data[i];
        let mut count = 1;

        while i + count < data.len() && data[i + count] == current {
            count += 1;
        }

        if count > 1 {
            result.push_str(&format!("{}{}", count, current as char));
        } else {
            result.push(current as char);
        }

        i += count;
    }

    result
}

pub fn rle_decode_verbose(s: &str) -> Vec<u8> {
    let mut result = Vec::new();
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c.is_ascii_digit() {
            let mut num_str = String::new();
            num_str.push(c);

            while let Some(&next) = chars.peek() {
                if next.is_ascii_digit() {
                    num_str.push(next);
                    chars.next();
                } else {
                    break;
                }
            }

            if let Some(byte_char) = chars.next() {
                let count: usize = num_str.parse().unwrap_or(1);
                for _ in 0..count {
                    result.push(byte_char as u8);
                }
            }
        } else {
            result.push(c as u8);
        }
    }

    result
}
