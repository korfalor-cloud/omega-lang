/// Cryptographic protocols: commitments, zero-knowledge proofs, secret sharing, oblivious transfer.

use std::collections::HashMap;

/// Pedersen commitment scheme (simplified, using modular arithmetic).
pub struct PedersenCommitment {
    pub p: u64,  // Large prime
    pub g: u64,  // Generator
    pub h: u64,  // Second generator (discrete log unknown to committer)
}

impl PedersenCommitment {
    pub fn new(p: u64, g: u64, h: u64) -> Self {
        Self { p, g, h }
    }

    /// Commit to value v with randomness r: C = g^v * h^r mod p.
    pub fn commit(&self, v: u64, r: u64) -> u64 {
        let gv = mod_pow(self.g, v, self.p);
        let hr = mod_pow(self.h, r, self.p);
        (gv * hr) % self.p
    }

    /// Verify a commitment opening.
    pub fn verify(&self, commitment: u64, v: u64, r: u64) -> bool {
        commitment == self.commit(v, r)
    }

    /// Homomorphic property: commit(a, r1) * commit(b, r2) = commit(a+b, r1+r2).
    pub fn add_commitments(&self, c1: u64, c2: u64) -> u64 {
        (c1 * c2) % self.p
    }
}

/// Shamir's Secret Sharing.
pub struct ShamirSharing {
    pub prime: u64,
}

impl ShamirSharing {
    pub fn new(prime: u64) -> Self {
        Self { prime }
    }

    /// Split secret into n shares with threshold t.
    pub fn split(&self, secret: u64, n: usize, t: usize, seed: &mut u64) -> Vec<(u64, u64)> {
        assert!(t <= n);
        assert!(t >= 2);

        // Generate random polynomial coefficients
        let mut coeffs = vec![secret];
        for _ in 1..t {
            coeffs.push(pseudo_rand_u64(seed) % self.prime);
        }

        // Evaluate polynomial at points 1..=n
        (1..=n as u64).map(|x| {
            let mut y = 0u64;
            let mut xpow = 1u64;
            for &c in &coeffs {
                y = (y + c * xpow % self.prime) % self.prime;
                xpow = xpow * x % self.prime;
            }
            (x, y)
        }).collect()
    }

    /// Reconstruct secret from shares using Lagrange interpolation.
    pub fn reconstruct(&self, shares: &[(u64, u64)]) -> u64 {
        let mut secret = 0i64;
        let p = self.prime as i64;

        for (i, &(xi, yi)) in shares.iter().enumerate() {
            let mut num = 1i64;
            let mut den = 1i64;

            for (j, &(xj, _)) in shares.iter().enumerate() {
                if i == j { continue; }
                num = (num * (-(xj as i64)).rem_euclid(p)) % p;
                den = (den * ((xi as i64) - (xj as i64)).rem_euclid(p)) % p;
            }

            let den_inv = mod_inverse(den as u64, self.prime) as i64;
            let lagrange = (num * den_inv).rem_euclid(p);
            secret = (secret + (yi as i64) * lagrange).rem_euclid(p);
        }

        secret as u64
    }
}

/// Simple zero-knowledge proof of discrete log equality (Chaum-Pedersen).
pub struct DLogEqualityProof {
    pub p: u64,
    pub g: u64,
    pub h: u64,
}

#[derive(Debug, Clone)]
pub struct DLogProof {
    pub commitment_a: u64,
    pub commitment_b: u64,
    pub response: u64,
}

impl DLogEqualityProof {
    pub fn new(p: u64, g: u64, h: u64) -> Self {
        Self { p, g, h }
    }

    /// Prove that y1 = g^x and y2 = h^x for the same x.
    pub fn prove(&self, x: u64, y1: u64, y2: u64, k: u64) -> DLogProof {
        let a1 = mod_pow(self.g, k, self.p);
        let a2 = mod_pow(self.h, k, self.p);

        // Challenge (Fiat-Shamir heuristic)
        let c = hash_to_u64(y1, y2, a1, a2) % self.p;

        // Response: r = k - c * x (mod p-1)
        let pm1 = self.p - 1;
        let r = ((k as i64 - (c as i64 * x as i64)).rem_euclid(pm1 as i64)) as u64;

        DLogProof {
            commitment_a: a1,
            commitment_b: a2,
            response: r,
        }
    }

    /// Verify the proof.
    pub fn verify(&self, y1: u64, y2: u64, proof: &DLogProof) -> bool {
        let c = hash_to_u64(y1, y2, proof.commitment_a, proof.commitment_b) % self.p;

        // Check g^r * y1^c == a1
        let lhs1 = (mod_pow(self.g, proof.response, self.p) * mod_pow(y1, c, self.p)) % self.p;
        if lhs1 != proof.commitment_a { return false; }

        // Check h^r * y2^c == a2
        let lhs2 = (mod_pow(self.h, proof.response, self.p) * mod_pow(y2, c, self.p)) % self.p;
        lhs2 == proof.commitment_b
    }
}

/// Oblivious Transfer (1-out-of-2 OT) using RSA-like construction.
pub struct ObliviousTransfer {
    pub p: u64,
    pub g: u64,
}

impl ObliviousTransfer {
    pub fn new(p: u64, g: u64) -> Self {
        Self { p, g }
    }

    /// Sender: generate public key and encrypted messages.
    pub fn sender_prepare(
        &self,
        m0: u64,
        m1: u64,
        sk: u64,
    ) -> (u64, u64, u64) {
        let pk = mod_pow(self.g, sk, self.p);
        // In real OT, messages would be encrypted; here simplified
        (pk, m0, m1)
    }

    /// Receiver: choose which message to receive.
    pub fn receiver_choose(
        &self,
        choice: usize,
        pk: u64,
        seed: &mut u64,
    ) -> (u64, u64) {
        let r = pseudo_rand_u64(seed) % (self.p - 1) + 1;
        let k0 = mod_pow(pk, r, self.p);
        let kr = mod_pow(self.g, r, self.p);
        let k1 = if choice == 0 { k0 } else { (kr * mod_inverse(pk, self.p)) % self.p };
        (k0, k1)
    }
}

/// Commitment scheme using hash function.
pub struct HashCommitment;

impl HashCommitment {
    pub fn commit(value: u64, nonce: u64) -> u64 {
        simple_hash(value, nonce)
    }

    pub fn verify(commitment: u64, value: u64, nonce: u64) -> bool {
        commitment == simple_hash(value, nonce)
    }
}

/// Sigma protocol for proving knowledge of a preimage.
pub struct SigmaProtocol;

impl SigmaProtocol {
    /// Prove knowledge of x such that y = H(x).
    pub fn prove_commit(x: u64, seed: &mut u64) -> (u64, u64) {
        let r = pseudo_rand_u64(seed);
        let commitment = simple_hash(r, 0);
        (commitment, r)
    }

    pub fn respond(r: u64, x: u64, challenge: u64) -> u64 {
        // Simplified: z = r + challenge * x
        r.wrapping_add(challenge.wrapping_mul(x))
    }

    pub fn verify(y: u64, commitment: u64, challenge: u64, response: u64) -> bool {
        // Simplified verification
        let check = simple_hash(response.wrapping_sub(challenge.wrapping_mul(y)), 0);
        check == commitment
    }
}

/// Additive secret sharing (for arithmetic circuits).
pub struct AdditiveSharing {
    pub prime: u64,
}

impl AdditiveSharing {
    pub fn new(prime: u64) -> Self { Self { prime } }

    pub fn share(&self, secret: u64, num_shares: usize, seed: &mut u64) -> Vec<u64> {
        let mut shares = Vec::new();
        let mut sum = 0u64;
        for _ in 0..num_shares - 1 {
            let s = pseudo_rand_u64(seed) % self.prime;
            shares.push(s);
            sum = (sum + s) % self.prime;
        }
        shares.push((self.prime + secret - sum) % self.prime);
        shares
    }

    pub fn reconstruct(&self, shares: &[u64]) -> u64 {
        shares.iter().fold(0u64, |acc, &s| (acc + s) % self.prime)
    }

    pub fn add_shares(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x + y) % self.prime).collect()
    }

    pub fn scalar_mul_share(&self, shares: &[u64], c: u64) -> Vec<u64> {
        shares.iter().map(|&s| (s * c) % self.prime).collect()
    }
}

/// Verifiable Secret Sharing (Feldman's scheme).
pub struct FeldmanVSS {
    pub p: u64,
    pub g: u64,
}

impl FeldmanVSS {
    pub fn new(p: u64, g: u64) -> Self { Self { p, g } }

    pub fn share(&self, secret: u64, n: usize, t: usize, seed: &mut u64) -> (Vec<(u64, u64)>, Vec<u64>) {
        let mut coeffs = vec![secret];
        for _ in 1..t {
            coeffs.push(pseudo_rand_u64(seed) % self.p);
        }

        // Commitments to coefficients
        let commitments: Vec<u64> = coeffs.iter().map(|&c| mod_pow(self.g, c, self.p)).collect();

        // Shares
        let shares: Vec<(u64, u64)> = (1..=n as u64).map(|x| {
            let mut y = 0u64;
            let mut xpow = 1u64;
            for &c in &coeffs {
                y = (y + c * xpow % self.p) % self.p;
                xpow = xpow * x % self.p;
            }
            (x, y)
        }).collect();

        (shares, commitments)
    }

    pub fn verify_share(&self, x: u64, y: u64, commitments: &[u64]) -> bool {
        let lhs = mod_pow(self.g, y, self.p);
        let rhs: u64 = commitments.iter().enumerate()
            .map(|(i, &c)| mod_pow(c, mod_pow(x, i as u64, self.p), self.p))
            .fold(1u64, |acc, v| (acc * v) % self.p);
        lhs == rhs
    }
}

// Helper functions

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % modulus;
        }
        exp >>= 1;
        base = base * base % modulus;
    }
    result
}

fn mod_inverse(a: u64, m: u64) -> u64 {
    let (mut t, mut newt) = (0i64, 1i64);
    let (mut r, mut newr) = (m as i64, a as i64);
    while newr != 0 {
        let quotient = r / newr;
        let tmp = newt;
        newt = t - quotient * newt;
        t = tmp;
        let tmp = newr;
        newr = r - quotient * newr;
        r = tmp;
    }
    if t < 0 { t += m as i64; }
    t as u64
}

fn simple_hash(a: u64, b: u64) -> u64 {
    let mut h = 14695981039346656037u64;
    for &byte in &a.to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(1099511628211);
    }
    for &byte in &b.to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn hash_to_u64(a: u64, b: u64, c: u64, d: u64) -> u64 {
    let mut h = 14695981039346656037u64;
    for &v in &[a, b, c, d] {
        for byte in v.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(1099511628211);
        }
    }
    h
}

fn pseudo_rand_u64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pedersen_commitment() {
        let pc = PedersenCommitment::new(101, 3, 7);
        let commitment = pc.commit(42, 17);
        assert!(pc.verify(commitment, 42, 17));
        assert!(!pc.verify(commitment, 42, 18));
        assert!(!pc.verify(commitment, 43, 17));
    }

    #[test]
    fn test_shamir_secret_sharing() {
        let ss = ShamirSharing::new(101);
        let mut seed = 42;
        let shares = ss.split(42, 5, 3, &mut seed);

        // Reconstruct with any 3 shares
        let reconstructed = ss.reconstruct(&shares[..3]);
        assert_eq!(reconstructed, 42);

        // Try with different 3 shares
        let reconstructed = ss.reconstruct(&shares[2..5]);
        assert_eq!(reconstructed, 42);
    }

    #[test]
    fn test_additive_sharing() {
        let ash = AdditiveSharing::new(101);
        let mut seed = 42;
        let shares = ash.share(42, 3, &mut seed);
        assert_eq!(ash.reconstruct(&shares), 42);

        let shares2 = ash.share(58, 3, &mut seed);
        let sum = ash.add_shares(&shares, &shares2);
        assert_eq!(ash.reconstruct(&sum), 0); // (42+58)%101 = 0
    }

    #[test]
    fn test_dlog_equality() {
        let proof = DLogEqualityProof::new(101, 3, 7);
        let x = 5;
        let y1 = mod_pow(3, x, 101);
        let y2 = mod_pow(7, x, 101);
        let k = 13;
        let p = proof.prove(x, y1, y2, k);
        assert!(proof.verify(y1, y2, &p));
    }

    #[test]
    fn test_hash_commitment() {
        let commitment = HashCommitment::commit(42, 12345);
        assert!(HashCommitment::verify(commitment, 42, 12345));
        assert!(!HashCommitment::verify(commitment, 42, 12346));
    }
}
