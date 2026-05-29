/// Advanced cryptographic primitives: RSA, ECC, Diffie-Hellman, ECDSA, SHA-256, SHA-3.

// ---------------------------------------------------------------------------
// SHA-256
// ---------------------------------------------------------------------------

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256(message: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    let bit_len = (message.len() as u64) * 8;
    let mut padded = message.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];
        for (i, c) in chunk.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K256[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, &val) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// SHA-3 (Keccak-256)
// ---------------------------------------------------------------------------

const KECCAK_ROUNDS: usize = 24;

const RC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];

const ROT: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

const PI: [[usize; 5]; 5] = [
    [0, 3, 1, 4, 2],
    [1, 4, 2, 0, 3],
    [2, 0, 3, 1, 4],
    [3, 1, 4, 2, 0],
    [4, 2, 0, 3, 1],
];

fn keccak_f1600(state: &mut [[u64; 5]; 5]) {
    for round in 0..KECCAK_ROUNDS {
        // Theta
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4];
        }
        let mut d = [0u64; 5];
        for x in 0..5 {
            d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
        }
        for x in 0..5 {
            for y in 0..5 {
                state[x][y] ^= d[x];
            }
        }

        // Rho and Pi
        let mut b = [[0u64; 5]; 5];
        for x in 0..5 {
            for y in 0..5 {
                b[y][PI[x][y]] = state[x][y].rotate_left(ROT[x][y]);
            }
        }

        // Chi
        for x in 0..5 {
            for y in 0..5 {
                state[x][y] = b[x][y] ^ ((!b[(x + 1) % 5][y]) & b[(x + 2) % 5][y]);
            }
        }

        // Iota
        state[0][0] ^= RC[round];
    }
}

fn sha3_256(message: &[u8]) -> [u8; 32] {
    let rate = 136; // (1600 - 2*256) / 8
    let mut state = [[0u64; 5]; 5];

    let mut padded = message.to_vec();
    padded.push(0x06); // SHA-3 domain separation
    while padded.len() % rate != 0 {
        padded.push(0);
    }
    let last = padded.len() - 1;
    padded[last] |= 0x80;

    for chunk in padded.chunks(rate) {
        for (i, byte) in chunk.iter().enumerate() {
            let lane = i / 8;
            let offset = (i % 8) * 8;
            let x = lane % 5;
            let y = lane / 5;
            state[x][y] ^= (*byte as u64) << offset;
        }
        keccak_f1600(&mut state);
    }

    let mut out = [0u8; 32];
    for i in 0..32 {
        let lane = i / 8;
        let offset = (i % 8) * 8;
        let x = lane % 5;
        let y = lane / 5;
        out[i] = (state[x][y] >> offset) as u8;
    }
    out
}

// ---------------------------------------------------------------------------
// Big integer helpers (fixed-width, enough for educational RSA)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct BigInt {
    digits: Vec<u64>,
}

impl BigInt {
    fn zero() -> Self {
        Self { digits: vec![0] }
    }

    fn from_u64(v: u64) -> Self {
        Self { digits: vec![v] }
    }

    fn from_bytes_be(bytes: &[u8]) -> Self {
        let mut digits = Vec::new();
        let mut i = bytes.len();
        while i > 0 {
            let start = if i >= 8 { i - 8 } else { 0 };
            let mut limb = 0u64;
            for &b in &bytes[start..i] {
                limb = (limb << 8) | (b as u64);
            }
            digits.push(limb);
            i = start;
        }
        if digits.is_empty() {
            digits.push(0);
        }
        Self { digits }
    }

    fn to_bytes_be(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &limb in self.digits.iter().rev() {
            bytes.extend_from_slice(&limb.to_be_bytes());
        }
        // trim leading zeros but keep at least one byte
        while bytes.len() > 1 && bytes[0] == 0 {
            bytes.remove(0);
        }
        bytes
    }

    fn is_zero(&self) -> bool {
        self.digits.iter().all(|&d| d == 0)
    }

    fn is_even(&self) -> bool {
        self.digits[0] & 1 == 0
    }

    fn cmp(&self, other: &BigInt) -> std::cmp::Ordering {
        let max_len = self.digits.len().max(other.digits.len());
        for i in (0..max_len).rev() {
            let a = if i < self.digits.len() { self.digits[i] } else { 0 };
            let b = if i < other.digits.len() { other.digits[i] } else { 0 };
            if a != b {
                return a.cmp(&b);
            }
        }
        std::cmp::Ordering::Equal
    }

    fn add(&self, other: &BigInt) -> BigInt {
        let max_len = self.digits.len().max(other.digits.len());
        let mut result = Vec::with_capacity(max_len + 1);
        let mut carry = 0u64;
        for i in 0..max_len {
            let a = if i < self.digits.len() { self.digits[i] } else { 0 };
            let b = if i < other.digits.len() { other.digits[i] } else { 0 };
            let (s1, o1) = a.overflowing_add(b);
            let (s2, o2) = s1.overflowing_add(carry);
            result.push(s2);
            carry = (o1 as u64) + (o2 as u64);
        }
        if carry > 0 {
            result.push(carry);
        }
        BigInt { digits: result }
    }

    fn sub(&self, other: &BigInt) -> BigInt {
        let mut result = Vec::with_capacity(self.digits.len());
        let mut borrow = 0i64;
        for i in 0..self.digits.len() {
            let a = self.digits[i] as i128;
            let b = if i < other.digits.len() { other.digits[i] as i128 } else { 0 };
            let diff = a - b - (borrow as i128);
            if diff < 0 {
                result.push((diff + (1i128 << 64)) as u64);
                borrow = 1;
            } else {
                result.push(diff as u64);
                borrow = 0;
            }
        }
        while result.len() > 1 && result.last() == Some(&0) {
            result.pop();
        }
        BigInt { digits: result }
    }

    fn mul(&self, other: &BigInt) -> BigInt {
        let mut result = vec![0u64; self.digits.len() + other.digits.len()];
        for i in 0..self.digits.len() {
            let mut carry = 0u128;
            for j in 0..other.digits.len() {
                let idx = i + j;
                let prod = (self.digits[i] as u128) * (other.digits[j] as u128)
                    + (result[idx] as u128)
                    + carry;
                result[idx] = prod as u64;
                carry = prod >> 64;
            }
            result[i + other.digits.len()] += carry as u64;
        }
        while result.len() > 1 && result.last() == Some(&0) {
            result.pop();
        }
        BigInt { digits: result }
    }

    fn div_mod(&self, divisor: &BigInt) -> (BigInt, BigInt) {
        assert!(!divisor.is_zero(), "division by zero");
        if self.cmp(divisor) == std::cmp::Ordering::Less {
            return (BigInt::zero(), self.clone());
        }
        let mut quotient = vec![0u64; self.digits.len()];
        let mut remainder = BigInt::zero();

        for i in (0..self.digits.len()).rev() {
            remainder.digits.insert(0, 0);
            if remainder.digits.len() > 1 && remainder.digits.last() == Some(&0) {
                remainder.digits.pop();
            }
            remainder = remainder.add(&BigInt::from_u64(self.digits[i]));

            // Binary search for the quotient digit
            let mut lo = 0u64;
            let mut hi = u64::MAX;
            while lo < hi {
                let mid = lo.wrapping_add(hi.wrapping_add(1 - lo) / 2);
                let trial = divisor.mul(&BigInt::from_u64(mid));
                if trial.cmp(&remainder) != std::cmp::Ordering::Greater {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            quotient[i] = lo;
            let subtracted = divisor.mul(&BigInt::from_u64(lo));
            remainder = remainder.sub(&subtracted);
        }

        while quotient.len() > 1 && quotient.last() == Some(&0) {
            quotient.pop();
        }
        (BigInt { digits: quotient }, remainder)
    }

    fn modpow(&self, exp: &BigInt, modulus: &BigInt) -> BigInt {
        let mut result = BigInt::from_u64(1);
        let mut base = self.div_mod(modulus).1;
        for i in 0..exp.digits.len() {
            let mut word = exp.digits[i];
            for _ in 0..64 {
                if word & 1 == 1 {
                    result = result.mul(&base).div_mod(modulus).1;
                }
                base = base.mul(&base).div_mod(modulus).1;
                word >>= 1;
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// RSA
// ---------------------------------------------------------------------------

/// A simple RSA keypair (educational, small primes).
#[derive(Debug)]
pub struct RsaKeyPair {
    pub n: Vec<u8>,
    pub e: Vec<u8>,
    pub d: Vec<u8>,
}

impl RsaKeyPair {
    /// Generate an RSA keypair from two prime numbers.
    pub fn from_primes(p: u64, q: u64) -> Self {
        let bn_p = BigInt::from_u64(p);
        let bn_q = BigInt::from_u64(q);
        let n = bn_p.mul(&bn_q);
        let phi = bn_p.sub(&BigInt::from_u64(1)).mul(&bn_q.sub(&BigInt::from_u64(1)));
        let e = BigInt::from_u64(65537);
        let d = modinv(&e, &phi);
        RsaKeyPair {
            n: n.to_bytes_be(),
            e: e.to_bytes_be(),
            d: d.to_bytes_be(),
        }
    }

    /// Encrypt a message (as a big integer) with the public key.
    pub fn encrypt(&self, plaintext: &[u8]) -> Vec<u8> {
        let m = BigInt::from_bytes_be(plaintext);
        let n = BigInt::from_bytes_be(&self.n);
        let e = BigInt::from_bytes_be(&self.e);
        m.modpow(&e, &n).to_bytes_be()
    }

    /// Decrypt a ciphertext (as a big integer) with the private key.
    pub fn decrypt(&self, ciphertext: &[u8]) -> Vec<u8> {
        let c = BigInt::from_bytes_be(ciphertext);
        let n = BigInt::from_bytes_be(&self.n);
        let d = BigInt::from_bytes_be(&self.d);
        c.modpow(&d, &n).to_bytes_be()
    }
}

fn egcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if b.is_zero() {
        return (a.clone(), BigInt::from_u64(1), BigInt::zero());
    }
    let (q, r) = a.div_mod(b);
    let (g, x, y) = egcd(b, &r);
    (g, y.clone(), x.sub(&q.mul(&y)))
}

fn modinv(a: &BigInt, m: &BigInt) -> BigInt {
    let (_, x, _) = egcd(a, m);
    let (_, rem) = x.div_mod(m);
    if rem.cmp(&BigInt::zero()) == std::cmp::Ordering::Less {
        rem.add(m)
    } else {
        rem
    }
}

// ---------------------------------------------------------------------------
// Elliptic Curve (secp256k1-like over a prime field)
// ---------------------------------------------------------------------------

/// A point on an elliptic curve y^2 = x^3 + ax + b (mod p).
#[derive(Clone, Debug, PartialEq)]
pub enum EcPoint {
    Infinity,
    Affine { x: BigInt, y: BigInt },
}

/// secp256k1-like curve parameters.
#[derive(Debug)]
pub struct EllipticCurve {
    pub a: BigInt,
    pub b: BigInt,
    pub p: BigInt,
    pub g: EcPoint,
    pub n: BigInt,
}

impl EllipticCurve {
    /// Create a small educational curve for demonstration.
    /// y^2 = x^3 + ax + b  (mod p)
    pub fn new(a: u64, b: u64, p: u64, gx: u64, gy: u64, n: u64) -> Self {
        Self {
            a: BigInt::from_u64(a),
            b: BigInt::from_u64(b),
            p: BigInt::from_u64(p),
            g: EcPoint::Affine { x: BigInt::from_u64(gx), y: BigInt::from_u64(gy) },
            n: BigInt::from_u64(n),
        }
    }

    pub fn point_add(&self, p1: &EcPoint, p2: &EcPoint) -> EcPoint {
        match (p1, p2) {
            (EcPoint::Infinity, q) => q.clone(),
            (p, EcPoint::Infinity) => p.clone(),
            (
                EcPoint::Affine { x: x1, y: y1 },
                EcPoint::Affine { x: x2, y: y2 },
            ) => {
                if x1 == x2 && y1 != y2 {
                    return EcPoint::Infinity;
                }

                let lambda = if x1 == x2 && y1 == y2 {
                    // Point doubling
                    let three_x2 = x1.mul(x1).mul(&BigInt::from_u64(3)).div_mod(&self.p).1;
                    let numerator = three_x2.add(&self.a);
                    let two_y = y1.mul(&BigInt::from_u64(2));
                    let denom_inv = modinv(&two_y.div_mod(&self.p).1, &self.p);
                    numerator.mul(&denom_inv).div_mod(&self.p).1
                } else {
                    let numerator = y2.sub(y1);
                    let denominator = x2.sub(x1);
                    let denom_inv = modinv(&denominator.div_mod(&self.p).1, &self.p);
                    numerator.mul(&denom_inv).div_mod(&self.p).1
                };

                let x3 = lambda
                    .mul(&lambda)
                    .sub(x1)
                    .sub(x2)
                    .div_mod(&self.p)
                    .1;
                let y3 = lambda
                    .mul(&x1.sub(&x3))
                    .sub(y1)
                    .div_mod(&self.p)
                    .1;

                EcPoint::Affine { x: x3, y: y3 }
            }
        }
    }

    /// Scalar multiplication using double-and-add.
    pub fn scalar_mul(&self, k: &BigInt, point: &EcPoint) -> EcPoint {
        let mut result = EcPoint::Infinity;
        let mut temp = point.clone();
        for i in 0..k.digits.len() {
            let mut word = k.digits[i];
            for _ in 0..64 {
                if word & 1 == 1 {
                    result = self.point_add(&result, &temp);
                }
                temp = self.point_add(&temp, &temp);
                word >>= 1;
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Diffie-Hellman Key Exchange
// ---------------------------------------------------------------------------

/// Diffie-Hellman key exchange parameters and computation.
#[derive(Debug)]
pub struct DiffieHellman {
    pub p: BigInt,
    pub g: BigInt,
}

impl DiffieHellman {
    pub fn new(p: u64, g: u64) -> Self {
        Self {
            p: BigInt::from_u64(p),
            g: BigInt::from_u64(g),
        }
    }

    /// Compute public key: g^private_key mod p
    pub fn public_key(&self, private_key: &BigInt) -> BigInt {
        self.g.modpow(private_key, &self.p)
    }

    /// Compute shared secret: other_public^private_key mod p
    pub fn shared_secret(&self, private_key: &BigInt, other_public: &BigInt) -> BigInt {
        other_public.modpow(private_key, &self.p)
    }
}

// ---------------------------------------------------------------------------
// ECDSA (Elliptic Curve Digital Signature Algorithm)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EcdsaSignature {
    pub r: Vec<u8>,
    pub s: Vec<u8>,
}

/// ECDSA signer using a provided elliptic curve.
#[derive(Debug)]
pub struct Ecdsa {
    curve: EllipticCurve,
}

impl Ecdsa {
    pub fn new(curve: EllipticCurve) -> Self {
        Self { curve }
    }

    /// Sign a message hash with a private key.
    pub fn sign(&self, hash: &[u8], private_key: &BigInt, k: &BigInt) -> EcdsaSignature {
        let z = BigInt::from_bytes_be(hash);

        // R = k * G
        let r_point = self.curve.scalar_mul(k, &self.curve.g);
        let rx = match &r_point {
            EcPoint::Affine { x, .. } => x.clone(),
            EcPoint::Infinity => panic!("k produced point at infinity"),
        };
        let r = rx.div_mod(&self.curve.n).1;

        // s = k^{-1} * (z + r*d) mod n
        let k_inv = modinv(k, &self.curve.n);
        let d = private_key;
        let rd = r.mul(d).div_mod(&self.curve.n).1;
        let s = k_inv.mul(&z.add(&rd)).div_mod(&self.curve.n).1;

        EcdsaSignature {
            r: r.to_bytes_be(),
            s: s.to_bytes_be(),
        }
    }

    /// Verify a signature against a message hash and public key.
    pub fn verify(&self, hash: &[u8], signature: &EcdsaSignature, public_key: &EcPoint) -> bool {
        let r = BigInt::from_bytes_be(&signature.r);
        let s = BigInt::from_bytes_be(&signature.s);
        let z = BigInt::from_bytes_be(hash);

        if r.is_zero() || s.is_zero() {
            return false;
        }

        let s_inv = modinv(&s, &self.curve.n);
        let u1 = z.mul(&s_inv).div_mod(&self.curve.n).1;
        let u2 = r.mul(&s_inv).div_mod(&self.curve.n).1;

        let point = self.curve.point_add(
            &self.curve.scalar_mul(&u1, &self.curve.g),
            &self.curve.scalar_mul(&u2, public_key),
        );

        match point {
            EcPoint::Infinity => false,
            EcPoint::Affine { x, .. } => x.div_mod(&self.curve.n).1 == r,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- SHA-256 ---

    #[test]
    fn test_sha256_empty() {
        let hash = sha256(b"");
        assert_eq!(
            hex_encode(&hash),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_abc() {
        let hash = sha256(b"abc");
        assert_eq!(
            hex_encode(&hash),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_sha256_hello_world() {
        let hash = sha256(b"hello world");
        assert_eq!(
            hex_encode(&hash),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    // --- SHA-3 ---

    #[test]
    fn test_sha3_empty() {
        let hash = sha3_256(b"");
        assert_eq!(
            hex_encode(&hash),
            "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
        );
    }

    #[test]
    fn test_sha3_abc() {
        let hash = sha3_256(b"abc");
        assert_eq!(
            hex_encode(&hash),
            "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
        );
    }

    // --- BigInt ---

    #[test]
    fn test_bigint_add() {
        let a = BigInt::from_u64(100);
        let b = BigInt::from_u64(200);
        assert_eq!(a.add(&b), BigInt::from_u64(300));
    }

    #[test]
    fn test_bigint_mul() {
        let a = BigInt::from_u64(1000);
        let b = BigInt::from_u64(1000);
        assert_eq!(a.mul(&b), BigInt::from_u64(1_000_000));
    }

    #[test]
    fn test_bigint_divmod() {
        let a = BigInt::from_u64(17);
        let b = BigInt::from_u64(5);
        let (q, r) = a.div_mod(&b);
        assert_eq!(q, BigInt::from_u64(3));
        assert_eq!(r, BigInt::from_u64(2));
    }

    #[test]
    fn test_bigint_modpow() {
        let base = BigInt::from_u64(3);
        let exp = BigInt::from_u64(13);
        let modulus = BigInt::from_u64(7);
        assert_eq!(base.modpow(&exp, &modulus), BigInt::from_u64(3)); // 3^13 mod 7 = 3
    }

    #[test]
    fn test_bigint_bytes_roundtrip() {
        let val = BigInt::from_u64(0xDEADBEEF);
        let bytes = val.to_bytes_be();
        let back = BigInt::from_bytes_be(&bytes);
        assert_eq!(val, back);
    }

    // --- RSA ---

    #[test]
    fn test_rsa_encrypt_decrypt() {
        let keypair = RsaKeyPair::from_primes(61, 53);
        let plaintext = vec![42];
        let ciphertext = keypair.encrypt(&plaintext);
        let decrypted = keypair.decrypt(&ciphertext);
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_rsa_larger_primes() {
        let keypair = RsaKeyPair::from_primes(3557, 2503);
        let msg = BigInt::from_u64(12345);
        let plaintext = msg.to_bytes_be();
        let ciphertext = keypair.encrypt(&plaintext);
        let decrypted = keypair.decrypt(&ciphertext);
        assert_eq!(decrypted, plaintext);
    }

    // --- Diffie-Hellman ---

    #[test]
    fn test_diffie_hellman_shared_secret() {
        // Safe prime p = 23, generator g = 5
        let dh = DiffieHellman::new(23, 5);

        let alice_private = BigInt::from_u64(6);
        let bob_private = BigInt::from_u64(15);

        let alice_public = dh.public_key(&alice_private);
        let bob_public = dh.public_key(&bob_private);

        let secret_a = dh.shared_secret(&alice_private, &bob_public);
        let secret_b = dh.shared_secret(&bob_private, &alice_public);

        assert_eq!(secret_a, secret_b);
        assert_eq!(secret_a, BigInt::from_u64(2)); // 5^(6*15) mod 23 = 2
    }

    #[test]
    fn test_diffie_hellman_larger_prime() {
        let dh = DiffieHellman::new(7919, 2);
        let a_priv = BigInt::from_u64(123);
        let b_priv = BigInt::from_u64(456);

        let a_pub = dh.public_key(&a_priv);
        let b_pub = dh.public_key(&b_priv);

        assert_eq!(
            dh.shared_secret(&a_priv, &b_pub),
            dh.shared_secret(&b_priv, &a_pub)
        );
    }

    // --- Elliptic Curve ---

    #[test]
    fn test_ec_point_add_identity() {
        let curve = EllipticCurve::new(2, 3, 97, 3, 6, 5);
        let p = EcPoint::Affine {
            x: BigInt::from_u64(3),
            y: BigInt::from_u64(6),
        };
        assert_eq!(curve.point_add(&p, &EcPoint::Infinity), p);
        assert_eq!(curve.point_add(&EcPoint::Infinity, &p), p);
    }

    #[test]
    fn test_ec_scalar_mul() {
        let curve = EllipticCurve::new(2, 3, 97, 3, 6, 5);
        let g = curve.g.clone();
        let two_g = curve.scalar_mul(&BigInt::from_u64(2), &g);
        let three_g = curve.scalar_mul(&BigInt::from_u64(3), &g);
        // 3G should equal 2G + G
        let two_g_plus_g = curve.point_add(&two_g, &g);
        assert_eq!(three_g, two_g_plus_g);
    }

    // --- ECDSA ---

    #[test]
    fn test_ecdsa_sign_verify() {
        let curve = EllipticCurve::new(2, 3, 97, 3, 6, 5);
        let ecdsa = Ecdsa::new(curve);

        let private_key = BigInt::from_u64(3);
        let public_key = ecdsa.curve.scalar_mul(&private_key, &ecdsa.curve.g);

        let message_hash = sha256(b"test message");
        let k = BigInt::from_u64(2);

        let sig = ecdsa.sign(&message_hash, &private_key, &k);
        assert!(ecdsa.verify(&message_hash, &sig, &public_key));
    }

    #[test]
    fn test_ecdsa_wrong_message() {
        let curve = EllipticCurve::new(2, 3, 97, 3, 6, 5);
        let ecdsa = Ecdsa::new(curve);

        let private_key = BigInt::from_u64(3);
        let public_key = ecdsa.curve.scalar_mul(&private_key, &ecdsa.curve.g);

        let hash1 = sha256(b"message one");
        let hash2 = sha256(b"message two");
        let k = BigInt::from_u64(2);

        let sig = ecdsa.sign(&hash1, &private_key, &k);
        assert!(!ecdsa.verify(&hash2, &sig, &public_key));
    }

    #[test]
    fn test_ecdsa_different_keys() {
        let curve = EllipticCurve::new(2, 3, 97, 3, 6, 5);
        let ecdsa = Ecdsa::new(curve);

        let key_a = BigInt::from_u64(3);
        let key_b = BigInt::from_u64(4);
        let pub_a = ecdsa.curve.scalar_mul(&key_a, &ecdsa.curve.g);

        let hash = sha256(b"hello");
        let k = BigInt::from_u64(2);

        let sig = ecdsa.sign(&hash, &key_b, &k);
        assert!(!ecdsa.verify(&hash, &sig, &pub_a));
    }

    // --- Integration ---

    #[test]
    fn test_full_signature_workflow() {
        // Alice signs, Bob verifies with public key
        let curve = EllipticCurve::new(2, 3, 97, 3, 6, 5);
        let ecdsa = Ecdsa::new(curve);

        let alice_private = BigInt::from_u64(5);
        let alice_public = ecdsa.curve.scalar_mul(&alice_private, &ecdsa.curve.g);

        let message = b"transfer 100 coins to bob";
        let hash = sha256(message);
        let k = BigInt::from_u64(3);

        let signature = ecdsa.sign(&hash, &alice_private, &k);
        assert!(ecdsa.verify(&hash, &signature, &alice_public));
    }
}

// --- Helpers ---

fn hex_encode(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}
