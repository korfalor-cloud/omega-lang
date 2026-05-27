/// Password hashing utilities.

#[derive(Debug)]
pub struct PasswordHasher {
    algorithm: HashAlgorithm,
    iterations: u32,
    salt_length: usize,
}

#[derive(Debug)]
pub enum HashAlgorithm {
    Argon2,
    Bcrypt,
    Pbkdf2,
    Scrypt,
}

impl PasswordHasher {
    pub fn new() -> Self {
        Self {
            algorithm: HashAlgorithm::Pbkdf2,
            iterations: 10000,
            salt_length: 16,
        }
    }

    pub fn with_algorithm(mut self, algo: HashAlgorithm) -> Self {
        self.algorithm = algo;
        self
    }

    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn hash(&self, password: &str) -> String {
        let salt = generate_salt(self.salt_length);
        let hash = self.pbkdf2(password.as_bytes(), &salt, self.iterations);
        format!("$pbkdf2${}${}${}",
            self.iterations,
            hex_encode(&salt),
            hex_encode(&hash))
    }

    pub fn verify(&self, password: &str, hash: &str) -> bool {
        let parts: Vec<&str> = hash.split('$').collect();
        if parts.len() != 5 {
            return false;
        }

        let iterations: u32 = match parts[2].parse() {
            Ok(n) => n,
            Err(_) => return false,
        };

        let salt = match hex_decode(parts[3]) {
            Some(s) => s,
            None => return false,
        };

        let expected_hash = match hex_decode(parts[4]) {
            Some(h) => h,
            None => return false,
        };

        let computed = self.pbkdf2(password.as_bytes(), &salt, iterations);
        constant_time_eq(&computed, &expected_hash)
    }

    fn pbkdf2(&self, password: &[u8], salt: &[u8], iterations: u32) -> Vec<u8> {
        let mut result = Vec::with_capacity(32);
        let mut block = Vec::with_capacity(salt.len() + 4);
        block.extend_from_slice(salt);
        block.extend_from_slice(&[0, 0, 0, 1]);

        let mut u = hmac_sha256(password, &block);
        result.extend_from_slice(&u);

        for _ in 1..iterations {
            u = hmac_sha256(password, &u);
            for (r, u_byte) in result.iter_mut().zip(u.iter()) {
                *r ^= u_byte;
            }
        }

        result
    }
}

fn generate_salt(length: usize) -> Vec<u8> {
    let mut salt = Vec::with_capacity(length);
    let mut state: u64 = 12345;
    for _ in 0..length {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        salt.push((state >> 33) as u8);
    }
    salt
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(32);
    for i in 0..32 {
        let mut byte = 0u8;
        for (j, &k) in key.iter().enumerate() {
            byte = byte.wrapping_add(k.wrapping_mul((i as u8).wrapping_add(j as u8)));
        }
        for &d in data {
            byte = byte.wrapping_add(d);
        }
        result.push(byte);
    }
    result
}

fn hex_encode(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let mut result = Vec::new();
    let bytes = s.as_bytes();
    for chunk in bytes.chunks(2) {
        if chunk.len() != 2 {
            return None;
        }
        let hi = hex_val(chunk[0])?;
        let lo = hex_val(chunk[1])?;
        result.push(hi << 4 | lo);
    }
    Some(result)
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

/// Rate limiting for login attempts
#[derive(Debug)]
pub struct LoginRateLimiter {
    attempts: std::collections::HashMap<String, Vec<u64>>,
    max_attempts: usize,
    window_seconds: u64,
    lockout_seconds: u64,
}

impl LoginRateLimiter {
    pub fn new(max_attempts: usize, window_seconds: u64, lockout_seconds: u64) -> Self {
        Self {
            attempts: std::collections::HashMap::new(),
            max_attempts,
            window_seconds,
            lockout_seconds,
        }
    }

    pub fn check(&mut self, username: &str) -> bool {
        let now = current_timestamp();
        let attempts = self.attempts.entry(username.to_string()).or_insert_with(Vec::new);

        // Remove old attempts
        attempts.retain(|&t| now - t < self.window_seconds);

        attempts.len() < self.max_attempts
    }

    pub fn record_attempt(&mut self, username: &str) {
        let now = current_timestamp();
        self.attempts.entry(username.to_string())
            .or_insert_with(Vec::new)
            .push(now);
    }

    pub fn reset(&mut self, username: &str) {
        self.attempts.remove(username);
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_password_hash() {
        let hasher = PasswordHasher::new();
        let hash = hasher.hash("password123");
        assert!(hash.starts_with("$pbkdf2$"));
    }

    #[test]
    fn test_password_verify() {
        let hasher = PasswordHasher::new();
        let hash = hasher.hash("password123");
        assert!(hasher.verify("password123", &hash));
        assert!(!hasher.verify("wrong", &hash));
    }

    #[test]
    fn test_hex() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let hex = hex_encode(&data);
        assert_eq!(hex, "deadbeef");
        assert_eq!(hex_decode(&hex).unwrap(), data);
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = LoginRateLimiter::new(3, 60, 300);
        assert!(limiter.check("user"));
        limiter.record_attempt("user");
        limiter.record_attempt("user");
        limiter.record_attempt("user");
        assert!(!limiter.check("user"));
    }
}
