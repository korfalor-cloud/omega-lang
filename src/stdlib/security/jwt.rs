/// JSON Web Token implementation.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Jwt {
    header: JwtHeader,
    claims: JwtClaims,
    signature: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct JwtHeader {
    pub algorithm: String,
    pub typ: String,
}

#[derive(Debug, Clone)]
pub struct JwtClaims {
    pub issuer: Option<String>,
    pub subject: Option<String>,
    pub audience: Option<String>,
    pub expiration: Option<u64>,
    pub not_before: Option<u64>,
    pub issued_at: Option<u64>,
    pub jwt_id: Option<String>,
    pub custom: HashMap<String, String>,
}

impl Jwt {
    pub fn new() -> Self {
        Self {
            header: JwtHeader {
                algorithm: "HS256".to_string(),
                typ: "JWT".to_string(),
            },
            claims: JwtClaims {
                issuer: None,
                subject: None,
                audience: None,
                expiration: None,
                not_before: None,
                issued_at: Some(current_timestamp()),
                jwt_id: None,
                custom: HashMap::new(),
            },
            signature: Vec::new(),
        }
    }

    pub fn set_algorithm(&mut self, alg: &str) {
        self.header.algorithm = alg.to_string();
    }

    pub fn set_issuer(&mut self, issuer: &str) {
        self.claims.issuer = Some(issuer.to_string());
    }

    pub fn set_subject(&mut self, subject: &str) {
        self.claims.subject = Some(subject.to_string());
    }

    pub fn set_audience(&mut self, audience: &str) {
        self.claims.audience = Some(audience.to_string());
    }

    pub fn set_expiration(&mut self, exp: u64) {
        self.claims.expiration = Some(exp);
    }

    pub fn set_expiration_seconds(&mut self, seconds: u64) {
        self.claims.expiration = Some(current_timestamp() + seconds);
    }

    pub fn set_not_before(&mut self, nbf: u64) {
        self.claims.not_before = Some(nbf);
    }

    pub fn set_jwt_id(&mut self, jti: &str) {
        self.claims.jwt_id = Some(jti.to_string());
    }

    pub fn set_claim(&mut self, key: &str, value: &str) {
        self.claims.custom.insert(key.to_string(), value.to_string());
    }

    pub fn get_claim(&self, key: &str) -> Option<&str> {
        self.claims.custom.get(key).map(|s| s.as_str())
    }

    pub fn sign(&mut self, secret: &[u8]) {
        let payload = self.encode_payload();
        self.signature = hmac_sha256(secret, payload.as_bytes());
    }

    pub fn verify(&self, secret: &[u8]) -> bool {
        let payload = self.encode_payload();
        let expected = hmac_sha256(secret, payload.as_bytes());
        self.signature == expected
    }

    pub fn is_expired(&self) -> bool {
        match self.claims.expiration {
            Some(exp) => current_timestamp() >= exp,
            None => false,
        }
    }

    pub fn is_valid(&self, secret: &[u8]) -> bool {
        self.verify(secret) && !self.is_expired()
    }

    fn encode_payload(&self) -> String {
        let header = format!(r#"{{"alg":"{}","typ":"{}"}}"#, self.header.algorithm, self.header.typ);
        let mut claims = String::from("{");

        if let Some(iss) = &self.claims.issuer {
            claims.push_str(&format!("\"iss\":\"{}\",", iss));
        }
        if let Some(sub) = &self.claims.subject {
            claims.push_str(&format!("\"sub\":\"{}\",", sub));
        }
        if let Some(aud) = &self.claims.audience {
            claims.push_str(&format!("\"aud\":\"{}\",", aud));
        }
        if let Some(exp) = self.claims.expiration {
            claims.push_str(&format!("\"exp\":{},", exp));
        }
        if let Some(nbf) = self.claims.not_before {
            claims.push_str(&format!("\"nbf\":{},", nbf));
        }
        if let Some(iat) = self.claims.issued_at {
            claims.push_str(&format!("\"iat\":{},", iat));
        }
        if let Some(jti) = &self.claims.jwt_id {
            claims.push_str(&format!("\"jti\":\"{}\",", jti));
        }
        claims.push('}');

        format!("{}.{}", base64_encode(header.as_bytes()), base64_encode(claims.as_bytes()))
    }

    pub fn to_string(&self) -> String {
        let payload = self.encode_payload();
        format!("{}.{}", payload, base64_encode(&self.signature))
    }

    pub fn from_string(token: &str) -> Option<Self> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return None;
        }

        let header_bytes = base64_decode(parts[0])?;
        let claims_bytes = base64_decode(parts[1])?;
        let signature = base64_decode(parts[2])?;

        let header_str = String::from_utf8_lossy(&header_bytes);
        let claims_str = String::from_utf8_lossy(&claims_bytes);

        // Parse header
        let algorithm = if header_str.contains("HS256") { "HS256" } else { "none" };

        // Parse claims
        let mut claims = JwtClaims {
            issuer: None,
            subject: None,
            audience: None,
            expiration: None,
            not_before: None,
            issued_at: None,
            jwt_id: None,
            custom: HashMap::new(),
        };

        // Simple JSON parsing for claims
        if let Some(exp_str) = extract_json_number(&claims_str, "exp") {
            claims.expiration = Some(exp_str as u64);
        }

        Some(Self {
            header: JwtHeader {
                algorithm: algorithm.to_string(),
                typ: "JWT".to_string(),
            },
            claims,
            signature,
        })
    }
}

fn extract_json_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\":", key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = &json[start..];
    let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).map_or(0, |&b| b as u32);
        let b2 = chunk.get(2).map_or(0, |&b| b as u32);
        let n = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((n >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((n >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(n & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.replace('-', "+").replace('_', "/");
    let mut result = Vec::new();
    let chars: Vec<u8> = s.bytes().collect();
    for chunk in chars.chunks(4) {
        if chunk.len() != 4 {
            return None;
        }
        let a = base64_val(chunk[0])?;
        let b = base64_val(chunk[1])?;
        let c = if chunk[2] == b'=' { 0 } else { base64_val(chunk[2])? };
        let d = if chunk[3] == b'=' { 0 } else { base64_val(chunk[3])? };
        let n = (a as u32) << 18 | (b as u32) << 12 | (c as u32) << 6 | d as u32;
        result.push((n >> 16) as u8);
        if chunk[2] != b'=' {
            result.push((n >> 8 & 0xFF) as u8);
        }
        if chunk[3] != b'=' {
            result.push((n & 0xFF) as u8);
        }
    }
    Some(result)
}

fn base64_val(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    // Simplified HMAC-SHA256 placeholder
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jwt_creation() {
        let mut jwt = Jwt::new();
        jwt.set_issuer("test");
        jwt.set_subject("user123");
        jwt.set_expiration_seconds(3600);

        assert_eq!(jwt.claims.issuer, Some("test".to_string()));
        assert_eq!(jwt.claims.subject, Some("user123".to_string()));
    }

    #[test]
    fn test_jwt_sign_verify() {
        let mut jwt = Jwt::new();
        jwt.set_issuer("test");
        jwt.sign(b"secret");

        assert!(jwt.verify(b"secret"));
        assert!(!jwt.verify(b"wrong"));
    }

    #[test]
    fn test_jwt_expiration() {
        let mut jwt = Jwt::new();
        jwt.set_expiration(0); // Already expired
        assert!(jwt.is_expired());

        jwt.set_expiration(current_timestamp() + 3600);
        assert!(!jwt.is_expired());
    }

    #[test]
    fn test_jwt_custom_claims() {
        let mut jwt = Jwt::new();
        jwt.set_claim("role", "admin");
        assert_eq!(jwt.get_claim("role"), Some("admin"));
    }

    #[test]
    fn test_base64() {
        let data = b"Hello, World!";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }
}
