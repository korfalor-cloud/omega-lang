/// HTTP middleware framework.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub query: HashMap<String, String>,
    pub body: Option<String>,
    pub params: HashMap<String, String>,
}

impl Request {
    pub fn new(method: &str, path: &str) -> Self {
        Self {
            method: method.to_string(),
            path: path.to_string(),
            headers: HashMap::new(),
            query: HashMap::new(),
            body: None,
            params: HashMap::new(),
        }
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(&name.to_lowercase()).map(|s| s.as_str())
    }

    pub fn query_param(&self, name: &str) -> Option<&str> {
        self.query.get(name).map(|s| s.as_str())
    }

    pub fn param(&self, name: &str) -> Option<&str> {
        self.params.get(name).map(|s| s.as_str())
    }

    pub fn content_type(&self) -> Option<&str> {
        self.header("content-type")
    }

    pub fn is_json(&self) -> bool {
        self.content_type().map_or(false, |ct| ct.contains("application/json"))
    }
}

#[derive(Debug, Clone)]
pub struct Response {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl Response {
    pub fn new(status: u16) -> Self {
        Self {
            status,
            headers: HashMap::new(),
            body: String::new(),
        }
    }

    pub fn ok() -> Self {
        Self::new(200)
    }

    pub fn not_found() -> Self {
        Self::new(404).with_body("Not Found")
    }

    pub fn internal_error() -> Self {
        Self::new(500).with_body("Internal Server Error")
    }

    pub fn json(status: u16, body: &str) -> Self {
        Self::new(status)
            .with_header("Content-Type", "application/json")
            .with_body(body)
    }

    pub fn html(status: u16, body: &str) -> Self {
        Self::new(status)
            .with_header("Content-Type", "text/html")
            .with_body(body)
    }

    pub fn redirect(url: &str) -> Self {
        Self::new(302)
            .with_header("Location", url)
    }

    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.insert(name.to_string(), value.to_string());
        self
    }

    pub fn with_body(mut self, body: &str) -> Self {
        self.body = body.to_string();
        self
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(|s| s.as_str())
    }
}

pub type MiddlewareFn = Box<dyn Fn(&Request, &mut Response) -> bool>;

pub struct Middleware {
    name: String,
    handler: MiddlewareFn,
}

impl Middleware {
    pub fn new<F: Fn(&Request, &mut Response) -> bool + 'static>(name: &str, handler: F) -> Self {
        Self {
            name: name.to_string(),
            handler: Box::new(handler),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn execute(&self, request: &Request, response: &mut Response) -> bool {
        (self.handler)(request, response)
    }
}

/// CORS middleware
pub fn cors_middleware(origins: &[&str]) -> Middleware {
    let origins: Vec<String> = origins.iter().map(|s| s.to_string()).collect();
    Middleware::new("cors", move |req, res| {
        let origin = req.header("origin").unwrap_or("*");
        if origins.contains(&"*".to_string()) || origins.iter().any(|o| o == origin) {
            res.headers.insert("Access-Control-Allow-Origin".to_string(), origin.to_string());
            res.headers.insert("Access-Control-Allow-Methods".to_string(), "GET, POST, PUT, DELETE, OPTIONS".to_string());
            res.headers.insert("Access-Control-Allow-Headers".to_string(), "Content-Type, Authorization".to_string());
        }
        true
    })
}

/// Rate limiting middleware
pub fn rate_limit_middleware(max_requests: usize, window_seconds: u64) -> Middleware {
    Middleware::new("rate_limit", move |_req, _res| {
        // Placeholder - in real implementation would track request counts
        true
    })
}

/// Logging middleware
pub fn logging_middleware() -> Middleware {
    Middleware::new("logging", |req, res| {
        // In real implementation would log to file/stdout
        true
    })
}

/// Authentication middleware
pub fn auth_middleware() -> Middleware {
    Middleware::new("auth", |req, res| {
        if req.header("authorization").is_none() {
            *res = Response::new(401).with_body("Unauthorized");
            return false;
        }
        true
    })
}

/// Compression middleware
pub fn compression_middleware() -> Middleware {
    Middleware::new("compression", |_req, res| {
        res.headers.insert("Content-Encoding".to_string(), "gzip".to_string());
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request() {
        let mut req = Request::new("GET", "/api/users");
        req.headers.insert("content-type".to_string(), "application/json".to_string());
        assert!(req.is_json());
    }

    #[test]
    fn test_response_builder() {
        let res = Response::ok()
            .with_header("X-Custom", "value")
            .with_body("hello");
        assert_eq!(res.status, 200);
        assert_eq!(res.body, "hello");
    }

    #[test]
    fn test_json_response() {
        let res = Response::json(200, r#"{"status":"ok"}"#);
        assert_eq!(res.header("Content-Type"), Some("application/json"));
    }

    #[test]
    fn test_redirect() {
        let res = Response::redirect("/login");
        assert_eq!(res.status, 302);
        assert_eq!(res.header("Location"), Some("/login"));
    }

    #[test]
    fn test_middleware_chain() {
        let cors = cors_middleware(&["*"]);
        let mut res = Response::ok();
        let req = Request::new("GET", "/");
        cors.execute(&req, &mut res);
        assert!(res.headers.contains_key("Access-Control-Allow-Origin"));
    }
}
