/// Error-correcting codes: Hamming, Reed-Solomon, convolutional codes, Golay.

/// Hamming(7,4) encoder/decoder.
pub struct HammingCode;

impl HammingCode {
    /// Encode 4 data bits into 7 bits (with 3 parity bits).
    pub fn encode(data: &[u8; 4]) -> [u8; 7] {
        let mut codeword = [0u8; 7];
        // Positions: 1,2,3,4,5,6,7 (1-indexed)
        // Data bits at positions 3,5,6,7
        codeword[2] = data[0]; // position 3
        codeword[4] = data[1]; // position 5
        codeword[5] = data[2]; // position 6
        codeword[6] = data[3]; // position 7

        // Parity bits at positions 1,2,4
        codeword[0] = codeword[2] ^ codeword[4] ^ codeword[6]; // p1 covers 1,3,5,7
        codeword[1] = codeword[2] ^ codeword[5] ^ codeword[6]; // p2 covers 2,3,6,7
        codeword[3] = codeword[4] ^ codeword[5] ^ codeword[6]; // p4 covers 4,5,6,7

        codeword
    }

    /// Decode and correct single-bit errors.
    pub fn decode(codeword: &[u8; 7]) -> ([u8; 4], usize) {
        let mut syndrome = 0usize;
        if (codeword[0] ^ codeword[2] ^ codeword[4] ^ codeword[6]) != 0 { syndrome |= 1; }
        if (codeword[1] ^ codeword[2] ^ codeword[5] ^ codeword[6]) != 0 { syndrome |= 2; }
        if (codeword[3] ^ codeword[4] ^ codeword[5] ^ codeword[6]) != 0 { syndrome |= 4; }

        let mut corrected = *codeword;
        if syndrome > 0 && syndrome <= 7 {
            corrected[syndrome - 1] ^= 1; // Flip the error bit
        }

        let data = [corrected[2], corrected[4], corrected[5], corrected[6]];
        (data, syndrome)
    }

    /// Encode a byte into two Hamming(7,4) codewords.
    pub fn encode_byte(byte: u8) -> [u8; 14] {
        let lo = [
            (byte >> 0) & 1, (byte >> 1) & 1, (byte >> 2) & 1, (byte >> 3) & 1
        ];
        let hi = [
            (byte >> 4) & 1, (byte >> 5) & 1, (byte >> 6) & 1, (byte >> 7) & 1
        ];
        let cw_lo = Self::encode(&lo);
        let cw_hi = Self::encode(&hi);

        let mut result = [0u8; 14];
        result[..7].copy_from_slice(&cw_lo);
        result[7..].copy_from_slice(&cw_hi);
        result
    }
}

/// Convolutional code (rate 1/2, constraint length 3).
pub struct ConvolutionalCode {
    pub g1: u8, // Generator polynomial 1 (0b111 = 7)
    pub g2: u8, // Generator polynomial 2 (0b101 = 5)
    pub state: u8,
}

impl ConvolutionalCode {
    pub fn new() -> Self {
        Self { g1: 0b111, g2: 0b101, state: 0 }
    }

    /// Encode a bit sequence.
    pub fn encode(&mut self, data: &[u8]) -> Vec<u8> {
        self.state = 0;
        let mut output = Vec::new();

        for &bit in data {
            self.state = ((self.state << 1) | bit) & 0b111;
            output.push(Self::parity(self.state & self.g1));
            output.push(Self::parity(self.state & self.g2));
        }

        // Flush
        for _ in 0..2 {
            self.state = (self.state << 1) & 0b111;
            output.push(Self::parity(self.state & self.g1));
            output.push(Self::parity(self.state & self.g2));
        }

        output
    }

    /// Viterbi decoder.
    pub fn decode(&self, received: &[u8]) -> Vec<u8> {
        let n_states = 4; // 2^(K-1) where K=3
        let msg_len = received.len() / 2;

        // Path metrics
        let mut path_metric = vec![u32::MAX; n_states];
        path_metric[0] = 0;

        let mut survivor: Vec<Vec<u8>> = vec![Vec::new(); n_states];

        for t in 0..msg_len {
            let r0 = received[2 * t];
            let r1 = received[2 * t + 1];
            let mut new_metric = vec![u32::MAX; n_states];
            let mut new_survivor: Vec<Vec<u8>> = vec![Vec::new(); n_states];

            for s in 0..n_states {
                if path_metric[s] == u32::MAX { continue; }

                for input in 0..2u8 {
                    let next_state = ((s << 1) | input as usize) & (n_states - 1);
                    let out0 = Self::parity((s as u8) << 1 | input) & 1;
                    let out1 = Self::parity(((s as u8) << 1 | input) & self.g2);

                    let dist = (r0 ^ out0) as u32 + (r1 ^ out1) as u32;
                    let metric = path_metric[s] + dist;

                    if metric < new_metric[next_state] {
                        new_metric[next_state] = metric;
                        new_survivor[next_state] = survivor[s].clone();
                        new_survivor[next_state].push(input);
                    }
                }
            }

            path_metric = new_metric;
            survivor = new_survivor;
        }

        // Return best path
        let best = path_metric.iter().enumerate()
            .min_by_key(|(_, m)| *m)
            .map(|(i, _)| i)
            .unwrap_or(0);

        survivor[best].clone()
    }

    fn parity(mut x: u8) -> u8 {
        let mut p = 0;
        while x != 0 {
            p ^= x & 1;
            x >>= 1;
        }
        p
    }
}

/// Golay code G(23,12) - perfect code.
pub struct GolayCode;

impl GolayCode {
    /// Generator matrix for Golay(23,12).
    const G: [[u8; 23]; 12] = [
        [1,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,0,1,1,1,0,0,0,1],
        [0,1,0,0,0,0,0,0,0,0,0,0, 1,1,0,1,1,1,0,0,0,1,0],
        [0,0,1,0,0,0,0,0,0,0,0,0, 1,0,1,1,1,0,0,0,1,0,1],
        [0,0,0,1,0,0,0,0,0,0,0,0, 0,1,1,1,0,0,0,1,0,1,1],
        [0,0,0,0,1,0,0,0,0,0,0,0, 1,1,1,0,0,0,1,0,1,1,0],
        [0,0,0,0,0,1,0,0,0,0,0,0, 1,1,0,0,0,1,0,1,1,0,1],
        [0,0,0,0,0,0,1,0,0,0,0,0, 1,0,0,0,1,0,1,1,0,1,1],
        [0,0,0,0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,1,0,1,1,1],
        [0,0,0,0,0,0,0,0,1,0,0,0, 0,0,1,0,1,1,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,1,0,0, 0,1,0,1,1,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0, 1,0,1,1,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1, 0,1,1,0,1,1,1,0,0,0,1],
    ];

    pub fn encode(data: &[u8; 12]) -> [u8; 23] {
        let mut codeword = [0u8; 23];
        for i in 0..12 {
            if data[i] != 0 {
                for j in 0..23 {
                    codeword[j] ^= Self::G[i][j];
                }
            }
        }
        codeword
    }

    /// Weight of a binary vector.
    fn weight(v: &[u8]) -> usize {
        v.iter().filter(|&&x| x != 0).count()
    }

    /// Syndrome computation.
    fn syndrome(codeword: &[u8; 23]) -> [u8; 11] {
        // H is the parity check matrix
        let h: [[u8; 23]; 11] = [
            [1,1,1,0,1,1,1,0,0,0,1, 1,0,0,0,0,0,0,0,0,0,0,0],
            [1,1,0,1,1,1,0,0,0,1,0, 0,1,0,0,0,0,0,0,0,0,0,0],
            [1,0,1,1,1,0,0,0,1,0,1, 0,0,1,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,0,0,0,1,0,1,1, 0,0,0,1,0,0,0,0,0,0,0,0],
            [1,1,1,0,0,0,1,0,1,1,0, 0,0,0,0,1,0,0,0,0,0,0,0],
            [1,1,0,0,0,1,0,1,1,0,1, 0,0,0,0,0,1,0,0,0,0,0,0],
            [1,0,0,0,1,0,1,1,0,1,1, 0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,1,0,1,1,0,1,1,1, 0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,1,0,1,1,0,1,1,1,0, 0,0,0,0,0,0,0,0,1,0,0,0],
            [0,1,0,1,1,0,1,1,1,0,0, 0,0,0,0,0,0,0,0,0,1,0,0],
            [1,0,1,1,0,1,1,1,0,0,0, 0,0,0,0,0,0,0,0,0,0,1,0],
        ];

        let mut s = [0u8; 11];
        for i in 0..11 {
            for j in 0..23 {
                s[i] ^= h[i][j] & codeword[j];
            }
        }
        s
    }
}

/// Reed-Solomon code over GF(2^8).
pub struct ReedSolomon {
    pub n: usize,  // Codeword length
    pub k: usize,  // Message length
    pub t: usize,  // Error correction capability
}

impl ReedSolomon {
    pub fn new(n: usize, k: usize) -> Self {
        let t = (n - k) / 2;
        Self { n, k, t }
    }

    /// Multiply in GF(2^8) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1.
    fn gf_mul(a: u8, b: u8) -> u8 {
        let mut result = 0u8;
        let mut a = a;
        let mut b = b;
        for _ in 0..8 {
            if b & 1 != 0 {
                result ^= a;
            }
            let carry = a & 0x80;
            a <<= 1;
            if carry != 0 {
                a ^= 0x1d; // x^8 + x^4 + x^3 + x^2 + 1
            }
            b >>= 1;
        }
        result
    }

    fn gf_pow(mut base: u8, mut exp: u8) -> u8 {
        let mut result = 1u8;
        while exp > 0 {
            if exp & 1 != 0 {
                result = Self::gf_mul(result, base);
            }
            base = Self::gf_mul(base, base);
            exp >>= 1;
        }
        result
    }

    fn gf_inv(a: u8) -> u8 {
        // a^254 = a^(-1) in GF(2^8)
        Self::gf_pow(a, 254)
    }

    /// Generate generator polynomial for RS(n,k).
    fn generator_poly(&self) -> Vec<u8> {
        let mut g = vec![1u8];
        for i in 0..(self.n - self.k) {
            let alpha_i = Self::gf_pow(2, i as u8);
            let mut new_g = vec![0u8; g.len() + 1];
            for j in 0..g.len() {
                new_g[j] ^= g[j];
                new_g[j + 1] ^= Self::gf_mul(g[j], alpha_i);
            }
            g = new_g;
        }
        g
    }

    /// Encode message polynomial.
    pub fn encode(&self, message: &[u8]) -> Vec<u8> {
        let g = self.generator_poly();
        let mut codeword = vec![0u8; self.n];

        // Copy message
        for i in 0..message.len().min(self.k) {
            codeword[i] = message[i];
        }

        // Systematic encoding: divide by generator
        for i in 0..self.k {
            let coef = codeword[i];
            if coef != 0 {
                for j in 0..g.len() {
                    codeword[i + j] ^= Self::gf_mul(g[j], coef);
                }
            }
        }

        // Copy message back (systematic)
        for i in 0..message.len().min(self.k) {
            codeword[i] = message[i];
        }

        codeword
    }

    /// Compute syndrome polynomial.
    fn syndromes(&self, codeword: &[u8]) -> Vec<u8> {
        let mut s = vec![0u8; self.n - self.k];
        for i in 0..(self.n - self.k) {
            let alpha_i = Self::gf_pow(2, i as u8);
            let mut val = 0u8;
            for &c in codeword.iter().rev() {
                val = Self::gf_mul(val, alpha_i) ^ c;
            }
            s[i] = val;
        }
        s
    }

    /// Check if codeword has errors.
    pub fn has_errors(&self, codeword: &[u8]) -> bool {
        self.syndromes(codeword).iter().any(|&s| s != 0)
    }
}

/// Run-Length Encoding.
pub struct RLE;

impl RLE {
    pub fn encode(data: &[u8]) -> Vec<(u8, u32)> {
        if data.is_empty() { return Vec::new(); }

        let mut result = Vec::new();
        let mut current = data[0];
        let mut count = 1u32;

        for &byte in &data[1..] {
            if byte == current {
                count += 1;
                if count == u32::MAX {
                    result.push((current, count));
                    count = 0;
                }
            } else {
                if count > 0 {
                    result.push((current, count));
                }
                current = byte;
                count = 1;
            }
        }
        if count > 0 {
            result.push((current, count));
        }

        result
    }

    pub fn decode(encoded: &[(u8, u32)]) -> Vec<u8> {
        let mut result = Vec::new();
        for &(byte, count) in encoded {
            for _ in 0..count {
                result.push(byte);
            }
        }
        result
    }
}

/// Base64 encoding.
pub struct Base64;

impl Base64 {
    const CHARS: &'static [u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn encode(data: &[u8]) -> String {
        let mut result = String::new();
        let chunks = data.chunks(3);

        for chunk in chunks {
            let b0 = chunk[0] as u32;
            let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
            let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };

            let triple = (b0 << 16) | (b1 << 8) | b2;

            result.push(Self::CHARS[((triple >> 18) & 0x3F) as usize] as char);
            result.push(Self::CHARS[((triple >> 12) & 0x3F) as usize] as char);

            if chunk.len() > 1 {
                result.push(Self::CHARS[((triple >> 6) & 0x3F) as usize] as char);
            } else {
                result.push('=');
            }

            if chunk.len() > 2 {
                result.push(Self::CHARS[(triple & 0x3F) as usize] as char);
            } else {
                result.push('=');
            }
        }

        result
    }

    pub fn decode(encoded: &str) -> Result<Vec<u8>, &'static str> {
        let clean: String = encoded.chars().filter(|c| *c != '=').collect();
        let mut result = Vec::new();
        let chars: Vec<u8> = clean.chars().map(|c| {
            Self::CHARS.iter().position(|&x| x == c as u8).map(|p| p as u8).ok_or("Invalid character")
        }).collect::<Result<Vec<u8>, _>>()?;

        for chunk in chars.chunks(4) {
            let b0 = chunk[0] as u32;
            let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
            let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
            let b3 = if chunk.len() > 3 { chunk[3] as u32 } else { 0 };

            let triple = (b0 << 18) | (b1 << 12) | (b2 << 6) | b3;

            result.push((triple >> 16) as u8);
            if chunk.len() > 2 {
                result.push((triple >> 8) as u8);
            }
            if chunk.len() > 3 {
                result.push(triple as u8);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming() {
        let data = [1u8, 0, 1, 1];
        let encoded = HammingCode::encode(&data);
        let (decoded, syndrome) = HammingCode::decode(&encoded);
        assert_eq!(decoded, data);
        assert_eq!(syndrome, 0);

        // Introduce single error
        let mut corrupted = encoded;
        corrupted[3] ^= 1;
        let (decoded, syndrome) = HammingCode::decode(&corrupted);
        assert_eq!(decoded, data);
        assert_eq!(syndrome, 5); // Error at position 5
    }

    #[test]
    fn test_rle() {
        let data = vec![1, 1, 1, 2, 2, 3, 3, 3, 3];
        let encoded = RLE::encode(&data);
        assert_eq!(encoded, vec![(1, 3), (2, 2), (3, 4)]);
        let decoded = RLE::decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base64() {
        let data = b"Hello, World!";
        let encoded = Base64::encode(data);
        let decoded = Base64::decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_convolutional() {
        let mut cc = ConvolutionalCode::new();
        let data = vec![1, 0, 1, 1, 0];
        let encoded = cc.encode(&data);
        assert_eq!(encoded.len(), (data.len() + 2) * 2); // Rate 1/2 + flush

        let decoded = cc.decode(&encoded);
        assert_eq!(&decoded[..data.len()], &data[..]);
    }
}
