use std::collections::HashMap;

// ============================================================
// AES-128 Encryption (Educational Implementation)
// ============================================================

pub struct Aes128 {
    round_keys: [[u8; 4]; 44],
}

impl Aes128 {
    const SBOX: [u8; 256] = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
    ];

    const RCON: [u8; 11] = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

    pub fn new(key: &[u8; 16]) -> Self {
        let mut round_keys = [[0u8; 4]; 44];

        for i in 0..4 {
            for j in 0..4 {
                round_keys[i][j] = key[i * 4 + j];
            }
        }

        for i in 4..44 {
            let mut temp = round_keys[i - 1];
            if i % 4 == 0 {
                temp = Self::sub_word(Self::rot_word(temp));
                temp[0] ^= Self::RCON[i / 4];
            }
            for j in 0..4 {
                round_keys[i][j] = round_keys[i - 4][j] ^ temp[j];
            }
        }

        Self { round_keys }
    }

    fn sub_word(word: [u8; 4]) -> [u8; 4] {
        [
            Self::SBOX[word[0] as usize],
            Self::SBOX[word[1] as usize],
            Self::SBOX[word[2] as usize],
            Self::SBOX[word[3] as usize],
        ]
    }

    fn rot_word(word: [u8; 4]) -> [u8; 4] {
        [word[1], word[2], word[3], word[0]]
    }

    pub fn encrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        let mut state = [[0u8; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                state[j][i] = block[i * 4 + j];
            }
        }

        Self::add_round_key(&mut state, &self.round_keys, 0);

        for round in 1..10 {
            Self::sub_bytes(&mut state);
            Self::shift_rows(&mut state);
            Self::mix_columns(&mut state);
            Self::add_round_key(&mut state, &self.round_keys, round);
        }

        Self::sub_bytes(&mut state);
        Self::shift_rows(&mut state);
        Self::add_round_key(&mut state, &self.round_keys, 10);

        let mut output = [0u8; 16];
        for i in 0..4 {
            for j in 0..4 {
                output[i * 4 + j] = state[j][i];
            }
        }
        output
    }

    fn sub_bytes(state: &mut [[u8; 4]; 4]) {
        for row in state.iter_mut() {
            for byte in row.iter_mut() {
                *byte = Self::SBOX[*byte as usize];
            }
        }
    }

    fn shift_rows(state: &mut [[u8; 4]; 4]) {
        let temp = state[1][0];
        state[1][0] = state[1][1];
        state[1][1] = state[1][2];
        state[1][2] = state[1][3];
        state[1][3] = temp;

        let (a, b) = (state[2][0], state[2][1]);
        state[2][0] = state[2][2];
        state[2][1] = state[2][3];
        state[2][2] = a;
        state[2][3] = b;

        let temp = state[3][3];
        state[3][3] = state[3][2];
        state[3][2] = state[3][1];
        state[3][1] = state[3][0];
        state[3][0] = temp;
    }

    fn gmul(a: u8, b: u8) -> u8 {
        let mut p = 0u8;
        let mut a_val = a;
        let mut b_val = b;
        for _ in 0..8 {
            if b_val & 1 != 0 {
                p ^= a_val;
            }
            let hi = a_val & 0x80;
            a_val <<= 1;
            if hi != 0 {
                a_val ^= 0x1b;
            }
            b_val >>= 1;
        }
        p
    }

    fn mix_columns(state: &mut [[u8; 4]; 4]) {
        for col in 0..4 {
            let s0 = state[0][col];
            let s1 = state[1][col];
            let s2 = state[2][col];
            let s3 = state[3][col];

            state[0][col] = Self::gmul(s0, 2) ^ Self::gmul(s1, 3) ^ s2 ^ s3;
            state[1][col] = s0 ^ Self::gmul(s1, 2) ^ Self::gmul(s2, 3) ^ s3;
            state[2][col] = s0 ^ s1 ^ Self::gmul(s2, 2) ^ Self::gmul(s3, 3);
            state[3][col] = Self::gmul(s0, 3) ^ s1 ^ s2 ^ Self::gmul(s3, 2);
        }
    }

    fn add_round_key(state: &mut [[u8; 4]; 4], round_keys: &[[u8; 4]; 44], round: usize) {
        for i in 0..4 {
            for j in 0..4 {
                state[j][i] ^= round_keys[round * 4 + i][j];
            }
        }
    }

    pub fn encrypt_cbc(plaintext: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Vec<u8> {
        let aes = Self::new(key);
        let mut padded = plaintext.to_vec();
        let pad_len = 16 - (padded.len() % 16);
        padded.extend(std::iter::repeat(pad_len as u8).take(pad_len));

        let mut ciphertext = Vec::new();
        let mut prev_block = *iv;

        for chunk in padded.chunks(16) {
            let mut block = [0u8; 16];
            for i in 0..16 {
                block[i] = chunk[i] ^ prev_block[i];
            }
            let encrypted = aes.encrypt_block(&block);
            ciphertext.extend_from_slice(&encrypted);
            prev_block = encrypted;
        }

        ciphertext
    }
}

// ============================================================
// RSA Implementation
// ============================================================

pub struct RsaKeyPair {
    pub public_key: (u64, u64),  // (e, n)
    pub private_key: (u64, u64), // (d, n)
}

impl RsaKeyPair {
    pub fn generate(p: u64, q: u64) -> Result<Self, String> {
        if !Self::is_prime(p) || !Self::is_prime(q) {
            return Err("Both p and q must be prime".to_string());
        }
        if p == q {
            return Err("p and q must be different".to_string());
        }

        let n = p.checked_mul(q).ok_or("Overflow computing n")?;
        let phi = (p - 1) * (q - 1);
        let e = 65537u64;

        if Self::gcd(e, phi) != 1 {
            return Err("e and phi(n) are not coprime".to_string());
        }

        let d = Self::mod_inverse(e, phi)?;

        Ok(Self {
            public_key: (e, n),
            private_key: (d, n),
        })
    }

    pub fn encrypt(message: u64, public_key: (u64, u64)) -> u64 {
        let (e, n) = public_key;
        Self::mod_pow(message, e, n)
    }

    pub fn decrypt(ciphertext: u64, private_key: (u64, u64)) -> u64 {
        let (d, n) = private_key;
        Self::mod_pow(ciphertext, d, n)
    }

    pub fn sign(message: u64, private_key: (u64, u64)) -> u64 {
        let (d, n) = private_key;
        Self::mod_pow(message % n, d, n)
    }

    pub fn verify(message: u64, signature: u64, public_key: (u64, u64)) -> bool {
        let (e, n) = public_key;
        let recovered = Self::mod_pow(signature, e, n);
        recovered == message % n
    }

    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n < 4 {
            return true;
        }
        if n % 2 == 0 || n % 3 == 0 {
            return false;
        }
        let mut i = 5u64;
        while i * i <= n {
            if n % i == 0 || n % (i + 2) == 0 {
                return false;
            }
            i += 6;
        }
        true
    }

    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }

    fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
        if modulus == 1 {
            return 0;
        }
        let mut result = 1u64;
        base %= modulus;
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.wrapping_mul(base) % modulus;
            }
            exp >>= 1;
            base = base.wrapping_mul(base) % modulus;
        }
        result
    }

    fn mod_inverse(a: u64, m: u64) -> Result<u64, String> {
        let (g, x, _) = Self::extended_gcd(a as i64, m as i64);
        if g != 1 {
            return Err("Modular inverse does not exist".to_string());
        }
        Ok(((x % m as i64 + m as i64) % m as i64) as u64)
    }

    fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
        if a == 0 {
            return (b, 0, 1);
        }
        let (g, x, y) = Self::extended_gcd(b % a, a);
        (g, y - (b / a) * x, x)
    }
}

// ============================================================
// Elliptic Curve (secp256k1-like over small field)
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcPoint {
    pub x: Option<u64>,
    pub y: Option<u64>,
}

pub struct EllipticCurve {
    pub a: u64,
    pub b: u64,
    pub p: u64,
    pub g: EcPoint,
    pub n: u64,
}

impl EllipticCurve {
    pub fn secp256k1_small() -> Self {
        // Small prime field for educational use
        Self {
            a: 0,
            b: 7,
            p: 997,
            g: EcPoint {
                x: Some(439),
                y: Some(723),
            },
            n: 1009,
        }
    }

    pub fn point_at_infinity() -> EcPoint {
        EcPoint { x: None, y: None }
    }

    pub fn point_add(&self, p1: EcPoint, p2: EcPoint) -> EcPoint {
        if p1.x.is_none() {
            return p2;
        }
        if p2.x.is_none() {
            return p1;
        }

        let (x1, y1) = (p1.x.unwrap(), p1.y.unwrap());
        let (x2, y2) = (p2.x.unwrap(), p2.y.unwrap());

        if x1 == x2 && y1 != y2 {
            return Self::point_at_infinity();
        }

        let m = if x1 == x2 && y1 == y2 {
            let num = (3 * x1 * x1 + self.a) % self.p;
            let den = Self::mod_inv((2 * y1) % self.p, self.p).unwrap();
            (num * den) % self.p
        } else {
            let num = (self.p + y2 - y1) % self.p;
            let den = Self::mod_inv((self.p + x2 - x1) % self.p, self.p).unwrap();
            (num * den) % self.p
        };

        let x3 = (self.p + m * m - x1 - x2) % self.p;
        let y3 = (self.p + m * ((self.p + x1 - x3) % self.p) - y1) % self.p;

        EcPoint {
            x: Some(x3),
            y: Some(y3),
        }
    }

    pub fn scalar_mul(&self, k: u64, point: EcPoint) -> EcPoint {
        let mut result = Self::point_at_infinity();
        let mut addend = point;
        let mut remaining = k;

        while remaining > 0 {
            if remaining & 1 == 1 {
                result = self.point_add(result, addend);
            }
            addend = self.point_add(addend, addend);
            remaining >>= 1;
        }

        result
    }

    pub fn generate_keypair(&self, private_key: u64) -> (u64, EcPoint) {
        let public_key = self.scalar_mul(private_key, self.g);
        (private_key, public_key)
    }

    fn mod_inv(a: u64, m: u64) -> Option<u64> {
        let (g, x, _) = Self::extended_gcd(a as i64, m as i64);
        if g != 1 {
            None
        } else {
            Some(((x % m as i64 + m as i64) % m as i64) as u64)
        }
    }

    fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
        if a == 0 {
            return (b, 0, 1);
        }
        let (g, x, y) = Self::extended_gcd(b % a, a);
        (g, y - (b / a) * x, x)
    }
}

// ============================================================
// Digital Signatures (ECDSA-like)
// ============================================================

pub struct EcdsaSignature {
    pub r: u64,
    pub s: u64,
}

pub struct DigitalSignature;

impl DigitalSignature {
    pub fn ecdsa_sign(
        curve: &EllipticCurve,
        message_hash: u64,
        private_key: u64,
        k: u64,
    ) -> Result<EcdsaSignature, String> {
        let r_point = curve.scalar_mul(k, curve.g);
        let r = r_point.x.ok_or("R is point at infinity, choose different k")? % curve.n;
        if r == 0 {
            return Err("r cannot be zero".to_string());
        }

        let k_inv = EllipticCurve::mod_inv(k, curve.n)
            .ok_or("k has no modular inverse")?;
        let s = (k_inv * (message_hash + r * private_key)) % curve.n;
        if s == 0 {
            return Err("s cannot be zero".to_string());
        }

        Ok(EcdsaSignature { r, s })
    }

    pub fn ecdsa_verify(
        curve: &EllipticCurve,
        message_hash: u64,
        signature: &EcdsaSignature,
        public_key: EcPoint,
    ) -> Result<bool, String> {
        if signature.r == 0 || signature.s >= curve.n {
            return Ok(false);
        }

        let s_inv = EllipticCurve::mod_inv(signature.s, curve.n)
            .ok_or("s has no modular inverse")?;

        let u1 = (message_hash * s_inv) % curve.n;
        let u2 = (signature.r * s_inv) % curve.n;

        let point = curve.point_add(
            curve.scalar_mul(u1, curve.g),
            curve.scalar_mul(u2, public_key),
        );

        match point.x {
            Some(x) => Ok(x % curve.n == signature.r),
            None => Ok(false),
        }
    }
}

// ============================================================
// Key Derivation Functions
// ============================================================

pub struct Kdf;

impl Kdf {
    // PBKDF2-like key derivation using HMAC
    pub fn pbkdf2(password: &[u8], salt: &[u8], iterations: u32, dk_len: usize) -> Vec<u8> {
        let mut derived_key = Vec::new();
        let mut block_num = 1u32;

        while derived_key.len() < dk_len {
            let mut block = Self::hmac_sha256_simple(password, &Self::u32_to_be_bytes(block_num, salt));
            let mut u = block.clone();

            for _ in 1..iterations {
                u = Self::hmac_sha256_simple(password, &u);
                for (a, b) in block.iter_mut().zip(u.iter()) {
                    *a ^= b;
                }
            }

            derived_key.extend_from_slice(&block);
            block_num += 1;
        }

        derived_key.truncate(dk_len);
        derived_key
    }

    // HKDF (HMAC-based Key Derivation Function)
    pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> Vec<u8> {
        Self::hmac_sha256_simple(salt, ikm)
    }

    pub fn hkdf_expand(prk: &[u8], info: &[u8], length: usize) -> Vec<u8> {
        let mut okm = Vec::new();
        let mut t = Vec::new();
        let mut counter = 1u8;

        while okm.len() < length {
            let mut input = t.clone();
            input.extend_from_slice(info);
            input.push(counter);

            t = Self::hmac_sha256_simple(prk, &input);
            okm.extend_from_slice(&t);
            counter += 1;
        }

        okm.truncate(length);
        okm
    }

    // Simple HMAC-SHA256 (XOR-based for demonstration)
    fn hmac_sha256_simple(key: &[u8], data: &[u8]) -> Vec<u8> {
        const BLOCK_SIZE: usize = 64;
        let mut key_padded = vec![0u8; BLOCK_SIZE];

        if key.len() > BLOCK_SIZE {
            let hashed = Self::simple_sha256(key);
            key_padded[..32].copy_from_slice(&hashed);
        } else {
            key_padded[..key.len()].copy_from_slice(key);
        }

        let mut ipad = vec![0x36u8; BLOCK_SIZE];
        let mut opad = vec![0x5cu8; BLOCK_SIZE];

        for i in 0..BLOCK_SIZE {
            ipad[i] ^= key_padded[i];
            opad[i] ^= key_padded[i];
        }

        let mut inner = ipad;
        inner.extend_from_slice(data);
        let inner_hash = Self::simple_sha256(&inner);

        let mut outer = opad;
        outer.extend_from_slice(&inner_hash);
        Self::simple_sha256(&outer)
    }

    // Simple SHA-256-like hash (XOR-compression for demonstration)
    fn simple_sha256(data: &[u8]) -> Vec<u8> {
        let mut state = [
            0x6a09e667u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ];

        let mut padded = data.to_vec();
        let len = padded.len();
        padded.push(0x80);
        while (padded.len() % 64) != 56 {
            padded.push(0);
        }
        padded.extend_from_slice(&(len as u64 * 8).to_be_bytes());

        for chunk in padded.chunks(64) {
            let mut w = [0u32; 16];
            for (i, c) in chunk.chunks(4).enumerate() {
                w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
            }

            let mut working = state;
            for i in 0..16 {
                let s0 = working[0].rotate_right(2) ^ working[0].rotate_right(13) ^ working[0].rotate_right(22);
                let maj = (working[0] & working[1]) ^ (working[0] & working[2]) ^ (working[1] & working[2]);
                let t2 = s0.wrapping_add(maj);
                let s1 = working[4].rotate_right(6) ^ working[4].rotate_right(11) ^ working[4].rotate_right(25);
                let ch = (working[4] & working[5]) ^ ((!working[4]) & working[6]);
                let t1 = working[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(0x428a2f98)
                    .wrapping_add(w[i]);

                working[7] = working[6];
                working[6] = working[5];
                working[5] = working[4];
                working[4] = working[3].wrapping_add(t1);
                working[3] = working[2];
                working[2] = working[1];
                working[1] = working[0];
                working[0] = t1.wrapping_add(t2);
            }

            for i in 0..8 {
                state[i] = state[i].wrapping_add(working[i]);
            }
        }

        state.iter().flat_map(|x| x.to_be_bytes().to_vec()).collect()
    }

    fn u32_to_be_bytes(n: u32, salt: &[u8]) -> Vec<u8> {
        let mut result = salt.to_vec();
        result.extend_from_slice(&n.to_be_bytes());
        result
    }

    // Simple key stretching
    pub fn stretch_key(key: &[u8], rounds: u32) -> Vec<u8> {
        let mut result = key.to_vec();
        for _ in 0..rounds {
            result = Self::simple_sha256(&result);
        }
        result
    }
}

// ============================================================
// Diffie-Hellman Key Exchange
// ============================================================

pub struct DiffieHellman;

impl DiffieHellman {
    pub fn generate_shared_secret(private_key: u64, other_public: u64, p: u64) -> u64 {
        Self::mod_pow(other_public, private_key, p)
    }

    pub fn compute_public_key(g: u64, private_key: u64, p: u64) -> u64 {
        Self::mod_pow(g, private_key, p)
    }

    fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
        if modulus == 1 {
            return 0;
        }
        let mut result = 1u64;
        base %= modulus;
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.wrapping_mul(base) % modulus;
            }
            exp >>= 1;
            base = base.wrapping_mul(base) % modulus;
        }
        result
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aes_encrypt_block() {
        let key = [0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                   0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c];
        let plaintext = [0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
                         0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34];
        let aes = Aes128::new(&key);
        let ciphertext = aes.encrypt_block(&plaintext);
        assert_ne!(ciphertext, plaintext);
        assert_eq!(ciphertext.len(), 16);
    }

    #[test]
    fn test_aes_cbc_encrypt() {
        let key = [1u8; 16];
        let iv = [2u8; 16];
        let plaintext = b"Hello, AES encryption!";
        let ciphertext = Aes128::encrypt_cbc(plaintext, &key, &iv);
        assert!(ciphertext.len() >= plaintext.len());
        assert_eq!(ciphertext.len() % 16, 0);
    }

    #[test]
    fn test_aes_deterministic() {
        let key = [3u8; 16];
        let iv = [4u8; 16];
        let data = b"test data for determinism check";
        let c1 = Aes128::encrypt_cbc(data, &key, &iv);
        let c2 = Aes128::encrypt_cbc(data, &key, &iv);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_rsa_key_generation() {
        let keypair = RsaKeyPair::generate(61, 53).unwrap();
        assert_eq!(keypair.public_key.1, 61 * 53);
        assert_ne!(keypair.public_key.0, keypair.private_key.0);
    }

    #[test]
    fn test_rsa_encrypt_decrypt() {
        let keypair = RsaKeyPair::generate(61, 53).unwrap();
        let message = 42u64;
        let ciphertext = RsaKeyPair::encrypt(message, keypair.public_key);
        let decrypted = RsaKeyPair::decrypt(ciphertext, keypair.private_key);
        assert_eq!(decrypted, message);
    }

    #[test]
    fn test_rsa_sign_verify() {
        let keypair = RsaKeyPair::generate(61, 53).unwrap();
        let message = 123u64;
        let signature = RsaKeyPair::sign(message, keypair.private_key);
        assert!(RsaKeyPair::verify(message, signature, keypair.public_key));
    }

    #[test]
    fn test_rsa_verify_wrong_message() {
        let keypair = RsaKeyPair::generate(61, 53).unwrap();
        let message = 123u64;
        let signature = RsaKeyPair::sign(message, keypair.private_key);
        assert!(!RsaKeyPair::verify(456, signature, keypair.public_key));
    }

    #[test]
    fn test_rsa_non_prime_rejection() {
        let result = RsaKeyPair::generate(15, 53);
        assert!(result.is_err());
    }

    #[test]
    fn test_ec_point_add() {
        let curve = EllipticCurve::secp256k1_small();
        let inf = EllipticCurve::point_at_infinity();
        let g = curve.g;

        // Identity element
        assert_eq!(curve.point_add(g, inf), g);
        assert_eq!(curve.point_add(inf, g), g);
    }

    #[test]
    fn test_ec_scalar_mul() {
        let curve = EllipticCurve::secp256k1_small();
        let point = curve.scalar_mul(1, curve.g);
        assert_eq!(point, curve.g);
    }

    #[test]
    fn test_ec_generate_keypair() {
        let curve = EllipticCurve::secp256k1_small();
        let (priv_key, pub_key) = curve.generate_keypair(42);
        assert_eq!(priv_key, 42);
        assert!(pub_key.x.is_some());
        assert!(pub_key.y.is_some());
    }

    #[test]
    fn test_ecdsa_sign_verify() {
        let curve = EllipticCurve::secp256k1_small();
        let private_key = 42u64;
        let (_, public_key) = curve.generate_keypair(private_key);
        let message_hash = 123u64;
        let k = 7u64;

        let sig = DigitalSignature::ecdsa_sign(&curve, message_hash, private_key, k).unwrap();
        let valid = DigitalSignature::ecdsa_verify(&curve, message_hash, &sig, public_key).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_ecdsa_verify_wrong_message() {
        let curve = EllipticCurve::secp256k1_small();
        let private_key = 42u64;
        let (_, public_key) = curve.generate_keypair(private_key);
        let message_hash = 123u64;
        let k = 7u64;

        let sig = DigitalSignature::ecdsa_sign(&curve, message_hash, private_key, k).unwrap();
        let valid = DigitalSignature::ecdsa_verify(&curve, 999, &sig, public_key).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_pbkdf2_output_length() {
        let password = b"password123";
        let salt = b"salt";
        let dk = Kdf::pbkdf2(password, salt, 100, 32);
        assert_eq!(dk.len(), 32);
    }

    #[test]
    fn test_pbkdf2_deterministic() {
        let password = b"test";
        let salt = b"fixed";
        let dk1 = Kdf::pbkdf2(password, salt, 50, 16);
        let dk2 = Kdf::pbkdf2(password, salt, 50, 16);
        assert_eq!(dk1, dk2);
    }

    #[test]
    fn test_hkdf_extract_expand() {
        let ikm = b"input key material";
        let salt = b"salt value";
        let info = b"context info";

        let prk = Kdf::hkdf_extract(salt, ikm);
        assert_eq!(prk.len(), 32);

        let okm = Kdf::hkdf_expand(&prk, info, 42);
        assert_eq!(okm.len(), 42);
    }

    #[test]
    fn test_key_stretching() {
        let key = b"stretch";
        let result = Kdf::stretch_key(key, 100);
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_diffie_hellman() {
        let p = 997u64;
        let g = 5u64;
        let alice_private = 23u64;
        let bob_private = 17u64;

        let alice_public = DiffieHellman::compute_public_key(g, alice_private, p);
        let bob_public = DiffieHellman::compute_public_key(g, bob_private, p);

        let alice_shared = DiffieHellman::generate_shared_secret(alice_private, bob_public, p);
        let bob_shared = DiffieHellman::generate_shared_secret(bob_private, alice_public, p);

        assert_eq!(alice_shared, bob_shared);
    }

    #[test]
    fn test_gmul() {
        assert_eq!(Aes128::gmul(0x57, 0x13), 0xfe);
        assert_eq!(Aes128::gmul(0x02, 0x80), 0x01);
    }
}
