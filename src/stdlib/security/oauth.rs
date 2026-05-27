/// OAuth 2.0 implementation.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct OAuth {
    client_id: String,
    client_secret: String,
    redirect_uri: String,
    authorization_endpoint: String,
    token_endpoint: String,
    scopes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AccessToken {
    pub token: String,
    pub token_type: String,
    pub expires_in: u64,
    pub refresh_token: Option<String>,
    pub scope: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AuthorizationCode {
    pub code: String,
    pub state: Option<String>,
}

impl OAuth {
    pub fn new(client_id: &str, client_secret: &str) -> Self {
        Self {
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
            redirect_uri: String::new(),
            authorization_endpoint: String::new(),
            token_endpoint: String::new(),
            scopes: Vec::new(),
        }
    }

    pub fn with_redirect_uri(mut self, uri: &str) -> Self {
        self.redirect_uri = uri.to_string();
        self
    }

    pub fn with_endpoints(mut self, auth: &str, token: &str) -> Self {
        self.authorization_endpoint = auth.to_string();
        self.token_endpoint = token.to_string();
        self
    }

    pub fn with_scopes(mut self, scopes: &[&str]) -> Self {
        self.scopes = scopes.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn get_authorization_url(&self, state: &str) -> String {
        let scope = self.scopes.join(" ");
        format!(
            "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&state={}",
            self.authorization_endpoint,
            url_encode(&self.client_id),
            url_encode(&self.redirect_uri),
            url_encode(&scope),
            url_encode(state)
        )
    }

    pub fn exchange_code(&self, code: &str) -> Result<AccessToken, String> {
        // In real implementation, would make HTTP request to token endpoint
        Ok(AccessToken {
            token: format!("access_{}", code),
            token_type: "Bearer".to_string(),
            expires_in: 3600,
            refresh_token: Some("refresh_token".to_string()),
            scope: self.scopes.clone(),
        })
    }

    pub fn refresh_token(&self, refresh_token: &str) -> Result<AccessToken, String> {
        // In real implementation, would make HTTP request
        Ok(AccessToken {
            token: "new_access_token".to_string(),
            token_type: "Bearer".to_string(),
            expires_in: 3600,
            refresh_token: Some(refresh_token.to_string()),
            scope: self.scopes.clone(),
        })
    }
}

/// PKCE (Proof Key for Code Exchange) extension
#[derive(Debug, Clone)]
pub struct Pkce {
    code_verifier: String,
    code_challenge: String,
    method: String,
}

impl Pkce {
    pub fn new() -> Self {
        let verifier = generate_random_string(43);
        let challenge = sha256_base64url(&verifier);
        Self {
            code_verifier: verifier,
            code_challenge: challenge,
            method: "S256".to_string(),
        }
    }

    pub fn code_verifier(&self) -> &str {
        &self.code_verifier
    }

    pub fn code_challenge(&self) -> &str {
        &self.code_challenge
    }

    pub fn method(&self) -> &str {
        &self.method
    }
}

fn generate_random_string(length: usize) -> String {
    let chars: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~".chars().collect();
    let mut result = String::new();
    let mut state: u64 = 42;
    for _ in 0..length {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = (state >> 33) as usize % chars.len();
        result.push(chars[idx]);
    }
    result
}

fn sha256_base64url(input: &str) -> String {
    // Simplified SHA256 placeholder
    let hash = simple_hash(input.as_bytes());
    base64url_encode(&hash)
}

fn simple_hash(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(32);
    for i in 0..32 {
        let mut byte = 0u8;
        for &d in data {
            byte = byte.wrapping_add(d.wrapping_mul(i as u8 + 1));
        }
        result.push(byte);
    }
    result
}

fn base64url_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
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
        }
        if chunk.len() > 2 {
            result.push(CHARS[(n & 0x3F) as usize] as char);
        }
    }
    result
}

fn url_encode(s: &str) -> String {
    s.bytes().map(|b| match b {
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
            (b as char).to_string()
        }
        _ => format!("%{:02X}", b),
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oauth_authorization_url() {
        let oauth = OAuth::new("client_id", "secret")
            .with_redirect_uri("https://example.com/callback")
            .with_endpoints("https://auth.example.com/authorize", "https://auth.example.com/token")
            .with_scopes(&["openid", "profile"]);

        let url = oauth.get_authorization_url("random_state");
        assert!(url.contains("client_id=client_id"));
        assert!(url.contains("scope=openid"));
    }

    #[test]
    fn test_pkce() {
        let pkce = Pkce::new();
        assert_eq!(pkce.code_verifier().len(), 43);
        assert!(!pkce.code_challenge().is_empty());
        assert_eq!(pkce.method(), "S256");
    }

    #[test]
    fn test_url_encode() {
        assert_eq!(url_encode("hello world"), "hello%20world");
        assert_eq!(url_encode("foo&bar"), "foo%26bar");
    }
}
