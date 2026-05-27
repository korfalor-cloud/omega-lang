pub struct OmegaCipher;

impl OmegaCipher {
    // XOR cipher (simple, not secure)
    pub fn xor_encrypt(data: &[u8], key: &[u8]) -> Vec<u8> {
        if key.is_empty() {
            return data.to_vec();
        }

        data.iter()
            .enumerate()
            .map(|(i, &b)| b ^ key[i % key.len()])
            .collect()
    }

    pub fn xor_decrypt(data: &[u8], key: &[u8]) -> Vec<u8> {
        // XOR is symmetric
        Self::xor_encrypt(data, key)
    }

    // Caesar cipher
    pub fn caesar_encrypt(text: &str, shift: u8) -> String {
        text.chars()
            .map(|c| {
                if c.is_ascii_lowercase() {
                    (b'a' + (c as u8 - b'a' + shift) % 26) as char
                } else if c.is_ascii_uppercase() {
                    (b'A' + (c as u8 - b'A' + shift) % 26) as char
                } else {
                    c
                }
            })
            .collect()
    }

    pub fn caesar_decrypt(text: &str, shift: u8) -> String {
        Self::caesar_encrypt(text, 26 - shift)
    }

    // ROT13
    pub fn rot13(text: &str) -> String {
        Self::caesar_encrypt(text, 13)
    }

    // Vigenere cipher
    pub fn vigenere_encrypt(text: &str, key: &str) -> String {
        if key.is_empty() {
            return text.to_string();
        }

        let key_bytes: Vec<u8> = key.bytes().collect();
        let mut key_index = 0;

        text.chars()
            .map(|c| {
                if c.is_ascii_lowercase() {
                    let shift = key_bytes[key_index % key_bytes.len()] - b'a';
                    key_index += 1;
                    (b'a' + (c as u8 - b'a' + shift) % 26) as char
                } else if c.is_ascii_uppercase() {
                    let shift = key_bytes[key_index % key_bytes.len()] - b'A';
                    key_index += 1;
                    (b'A' + (c as u8 - b'A' + shift) % 26) as char
                } else {
                    c
                }
            })
            .collect()
    }

    pub fn vigenere_decrypt(text: &str, key: &str) -> String {
        if key.is_empty() {
            return text.to_string();
        }

        let key_bytes: Vec<u8> = key.bytes().collect();
        let mut key_index = 0;

        text.chars()
            .map(|c| {
                if c.is_ascii_lowercase() {
                    let shift = 26 - (key_bytes[key_index % key_bytes.len()] - b'a');
                    key_index += 1;
                    (b'a' + (c as u8 - b'a' + shift) % 26) as char
                } else if c.is_ascii_uppercase() {
                    let shift = 26 - (key_bytes[key_index % key_bytes.len()] - b'A');
                    key_index += 1;
                    (b'A' + (c as u8 - b'A' + shift) % 26) as char
                } else {
                    c
                }
            })
            .collect()
    }

    // Base64 encode/decode
    const BASE64_CHARS: &'static [u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn base64_encode(data: &[u8]) -> String {
        let mut result = String::new();
        let chunks = data.chunks_exact(3);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let b0 = chunk[0] as u32;
            let b1 = chunk[1] as u32;
            let b2 = chunk[2] as u32;
            let triple = (b0 << 16) | (b1 << 8) | b2;

            result.push(Self::BASE64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
            result.push(Self::BASE64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
            result.push(Self::BASE64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
            result.push(Self::BASE64_CHARS[(triple & 0x3F) as usize] as char);
        }

        match remainder.len() {
            1 => {
                let b0 = remainder[0] as u32;
                result.push(Self::BASE64_CHARS[((b0 >> 2) & 0x3F) as usize] as char);
                result.push(Self::BASE64_CHARS[((b0 << 4) & 0x3F) as usize] as char);
                result.push('=');
                result.push('=');
            }
            2 => {
                let b0 = remainder[0] as u32;
                let b1 = remainder[1] as u32;
                let triple = (b0 << 8) | b1;
                result.push(Self::BASE64_CHARS[((triple >> 10) & 0x3F) as usize] as char);
                result.push(Self::BASE64_CHARS[((triple >> 4) & 0x3F) as usize] as char);
                result.push(Self::BASE64_CHARS[((triple << 2) & 0x3F) as usize] as char);
                result.push('=');
            }
            _ => {}
        }

        result
    }

    pub fn base64_decode(encoded: &str) -> Result<Vec<u8>, String> {
        let mut result = Vec::new();
        let clean: String = encoded.chars().filter(|c| !c.is_whitespace()).collect();

        if clean.len() % 4 != 0 {
            return Err("Invalid base64 length".to_string());
        }

        for chunk in clean.as_bytes().chunks(4) {
            let a = Self::base64_index(chunk[0])?;
            let b = Self::base64_index(chunk[1])?;

            result.push((a << 2 | b >> 4) as u8);

            if chunk[2] != b'=' {
                let c = Self::base64_index(chunk[2])?;
                result.push((b << 4 | c >> 2) as u8);
            }

            if chunk[3] != b'=' {
                let d = Self::base64_index(chunk[3])?;
                result.push((c << 6 | d) as u8);
            }
        }

        Ok(result)
    }

    fn base64_index(c: u8) -> Result<u8, String> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("Invalid base64 character: {}", c as char)),
        }
    }

    // Hex encode/decode
    pub fn hex_encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{:02x}", b)).collect()
    }

    pub fn hex_decode(hex: &str) -> Result<Vec<u8>, String> {
        if hex.len() % 2 != 0 {
            return Err("Invalid hex length".to_string());
        }

        (0..hex.len())
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(&hex[i..i + 2], 16)
                    .map_err(|e| format!("Invalid hex: {}", e))
            })
            .collect()
    }

    // Simple substitution cipher
    pub fn substitution_encrypt(text: &str, key: &str) -> Result<String, String> {
        if key.len() != 26 {
            return Err("Key must be 26 characters".to_string());
        }

        let key_lower = key.to_lowercase();
        let key_upper = key.to_uppercase();

        Ok(text
            .chars()
            .map(|c| {
                if c.is_ascii_lowercase() {
                    let index = (c as u8 - b'a') as usize;
                    key_lower.as_bytes()[index] as char
                } else if c.is_ascii_uppercase() {
                    let index = (c as u8 - b'A') as usize;
                    key_upper.as_bytes()[index] as char
                } else {
                    c
                }
            })
            .collect())
    }

    pub fn substitution_decrypt(text: &str, key: &str) -> Result<String, String> {
        if key.len() != 26 {
            return Err("Key must be 26 characters".to_string());
        }

        let key_lower = key.to_lowercase();
        Ok(text
            .chars()
            .map(|c| {
                if c.is_ascii_lowercase() {
                    if let Some(pos) = key_lower.find(c) {
                        (b'a' + pos as u8) as char
                    } else {
                        c
                    }
                } else if c.is_ascii_uppercase() {
                    let lower = c.to_lowercase().next().unwrap();
                    if let Some(pos) = key_lower.find(lower) {
                        (b'A' + pos as u8) as char
                    } else {
                        c
                    }
                } else {
                    c
                }
            })
            .collect())
    }
}
