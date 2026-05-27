use std::collections::HashMap;
use crate::errors::{OmegaError, OmegaResult};

#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
    pub timeout_ms: u64,
}

impl HttpRequest {
    pub fn new(method: &str, url: &str) -> Self {
        Self {
            method: method.to_uppercase(),
            url: url.to_string(),
            headers: HashMap::new(),
            body: None,
            timeout_ms: 30000,
        }
    }

    pub fn get(url: &str) -> Self {
        Self::new("GET", url)
    }

    pub fn post(url: &str) -> Self {
        Self::new("POST", url)
    }

    pub fn put(url: &str) -> Self {
        Self::new("PUT", url)
    }

    pub fn delete(url: &str) -> Self {
        Self::new("DELETE", url)
    }

    pub fn header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    pub fn body(mut self, body: &str) -> Self {
        self.body = Some(body.to_string());
        self
    }

    pub fn json(mut self, json: &str) -> Self {
        self.headers.insert("Content-Type".to_string(), "application/json".to_string());
        self.body = Some(json.to_string());
        self
    }

    pub fn timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    pub fn send(&self) -> OmegaResult<HttpResponse> {
        // Simplified HTTP client - in production would use reqwest or hyper
        Err(OmegaError::NetworkError {
            message: "HTTP client not yet implemented - use std::net::tcp for raw connections".to_string(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl HttpResponse {
    pub fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }

    pub fn is_redirect(&self) -> bool {
        self.status >= 300 && self.status < 400
    }

    pub fn is_client_error(&self) -> bool {
        self.status >= 400 && self.status < 500
    }

    pub fn is_server_error(&self) -> bool {
        self.status >= 500
    }

    pub fn content_type(&self) -> Option<&String> {
        self.headers.get("Content-Type").or(self.headers.get("content-type"))
    }

    pub fn content_length(&self) -> Option<usize> {
        self.headers.get("Content-Length")
            .or(self.headers.get("content-length"))
            .and_then(|s| s.parse().ok())
    }

    pub fn json(&self) -> OmegaResult<serde_json::Value> {
        serde_json::from_str(&self.body).map_err(|e| OmegaError::ValueError {
            message: format!("Invalid JSON: {}", e),
        })
    }
}
