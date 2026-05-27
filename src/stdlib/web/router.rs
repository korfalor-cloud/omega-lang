/// HTTP router with path matching and middleware support.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Router {
    routes: Vec<Route>,
    middleware: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Route {
    pub method: HttpMethod,
    pub path: String,
    pub handler: String,
    pub middleware: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
}

impl HttpMethod {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "GET" => Some(HttpMethod::GET),
            "POST" => Some(HttpMethod::POST),
            "PUT" => Some(HttpMethod::PUT),
            "DELETE" => Some(HttpMethod::DELETE),
            "PATCH" => Some(HttpMethod::PATCH),
            "HEAD" => Some(HttpMethod::HEAD),
            "OPTIONS" => Some(HttpMethod::OPTIONS),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatchedRoute {
    pub handler: String,
    pub params: HashMap<String, String>,
    pub middleware: Vec<String>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            middleware: Vec::new(),
        }
    }

    pub fn get(mut self, path: &str, handler: &str) -> Self {
        self.routes.push(Route {
            method: HttpMethod::GET,
            path: path.to_string(),
            handler: handler.to_string(),
            middleware: Vec::new(),
        });
        self
    }

    pub fn post(mut self, path: &str, handler: &str) -> Self {
        self.routes.push(Route {
            method: HttpMethod::POST,
            path: path.to_string(),
            handler: handler.to_string(),
            middleware: Vec::new(),
        });
        self
    }

    pub fn put(mut self, path: &str, handler: &str) -> Self {
        self.routes.push(Route {
            method: HttpMethod::PUT,
            path: path.to_string(),
            handler: handler.to_string(),
            middleware: Vec::new(),
        });
        self
    }

    pub fn delete(mut self, path: &str, handler: &str) -> Self {
        self.routes.push(Route {
            method: HttpMethod::DELETE,
            path: path.to_string(),
            handler: handler.to_string(),
            middleware: Vec::new(),
        });
        self
    }

    pub fn route(mut self, method: HttpMethod, path: &str, handler: &str) -> Self {
        self.routes.push(Route {
            method,
            path: path.to_string(),
            handler: handler.to_string(),
            middleware: Vec::new(),
        });
        self
    }

    pub fn with_middleware(mut self, name: &str) -> Self {
        self.middleware.push(name.to_string());
        self
    }

    pub fn match_route(&self, method: &HttpMethod, path: &str) -> Option<MatchedRoute> {
        for route in &self.routes {
            if &route.method != method {
                continue;
            }
            if let Some(params) = match_path(&route.path, path) {
                let mut middleware = self.middleware.clone();
                middleware.extend(route.middleware.clone());
                return Some(MatchedRoute {
                    handler: route.handler.clone(),
                    params,
                    middleware,
                });
            }
        }
        None
    }

    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    pub fn group(mut self, prefix: &str, routes: Router) -> Self {
        for route in routes.routes {
            self.routes.push(Route {
                method: route.method,
                path: format!("{}{}", prefix, route.path),
                handler: route.handler,
                middleware: route.middleware,
            });
        }
        self
    }
}

fn match_path(pattern: &str, path: &str) -> Option<HashMap<String, String>> {
    let pattern_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();

    if pattern_parts.len() != path_parts.len() {
        return None;
    }

    let mut params = HashMap::new();

    for (pattern_part, path_part) in pattern_parts.iter().zip(path_parts.iter()) {
        if pattern_part.starts_with(':') {
            params.insert(pattern_part[1..].to_string(), path_part.to_string());
        } else if pattern_part.starts_with('*') {
            params.insert(pattern_part[1..].to_string(), path_part.to_string());
        } else if pattern_part != path_part {
            return None;
        }
    }

    Some(params)
}

/// URL builder
pub struct UrlBuilder {
    scheme: String,
    host: String,
    port: Option<u16>,
    path: String,
    query: Vec<(String, String)>,
    fragment: Option<String>,
}

impl UrlBuilder {
    pub fn new(scheme: &str, host: &str) -> Self {
        Self {
            scheme: scheme.to_string(),
            host: host.to_string(),
            port: None,
            path: String::new(),
            query: Vec::new(),
            fragment: None,
        }
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    pub fn path(mut self, path: &str) -> Self {
        self.path = path.to_string();
        self
    }

    pub fn query(mut self, key: &str, value: &str) -> Self {
        self.query.push((key.to_string(), value.to_string()));
        self
    }

    pub fn fragment(mut self, fragment: &str) -> Self {
        self.fragment = Some(fragment.to_string());
        self
    }

    pub fn build(&self) -> String {
        let mut url = format!("{}://{}", self.scheme, self.host);
        if let Some(port) = self.port {
            url.push_str(&format!(":{}", port));
        }
        if !self.path.is_empty() {
            if !self.path.starts_with('/') {
                url.push('/');
            }
            url.push_str(&self.path);
        }
        if !self.query.is_empty() {
            url.push('?');
            url.push_str(&self.query.iter()
                .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
                .collect::<Vec<_>>()
                .join("&"));
        }
        if let Some(fragment) = &self.fragment {
            url.push('#');
            url.push_str(fragment);
        }
        url
    }
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
    fn test_route_matching() {
        let router = Router::new()
            .get("/", "home")
            .get("/users/:id", "user_detail")
            .post("/users", "create_user");

        let matched = router.match_route(&HttpMethod::GET, "/users/42").unwrap();
        assert_eq!(matched.handler, "user_detail");
        assert_eq!(matched.params.get("id").unwrap(), "42");
    }

    #[test]
    fn test_no_match() {
        let router = Router::new().get("/", "home");
        assert!(router.match_route(&HttpMethod::POST, "/").is_none());
    }

    #[test]
    fn test_url_builder() {
        let url = UrlBuilder::new("https", "example.com")
            .port(8080)
            .path("/api/users")
            .query("page", "1")
            .query("limit", "10")
            .build();
        assert_eq!(url, "https://example.com:8080/api/users?page=1&limit=10");
    }

    #[test]
    fn test_url_encode() {
        assert_eq!(url_encode("hello world"), "hello%20world");
        assert_eq!(url_encode("foo&bar"), "foo%26bar");
    }

    #[test]
    fn test_route_group() {
        let api_routes = Router::new()
            .get("/users", "list_users")
            .get("/users/:id", "get_user");

        let router = Router::new()
            .group("/api", api_routes);

        let matched = router.match_route(&HttpMethod::GET, "/api/users/5").unwrap();
        assert_eq!(matched.handler, "get_user");
    }
}
