/// Advanced web framework with HTTP server, WebSocket, and middleware chain.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// HTTP Server
// ---------------------------------------------------------------------------

/// Configuration for the HTTP server.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub request_timeout_secs: u64,
    pub keep_alive: bool,
    pub max_body_size: usize,
}

impl ServerConfig {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            host: host.to_string(),
            port,
            max_connections: 1024,
            request_timeout_secs: 30,
            keep_alive: true,
            max_body_size: 1024 * 1024, // 1 MB
        }
    }

    pub fn max_connections(mut self, n: usize) -> Self {
        self.max_connections = n;
        self
    }

    pub fn request_timeout(mut self, secs: u64) -> Self {
        self.request_timeout_secs = secs;
        self
    }

    pub fn keep_alive(mut self, enabled: bool) -> Self {
        self.keep_alive = enabled;
        self
    }

    pub fn max_body_size(mut self, bytes: usize) -> Self {
        self.max_body_size = bytes;
        self
    }

    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Represents a running HTTP server (metadata only; no real I/O).
#[derive(Debug)]
pub struct HttpServer {
    config: ServerConfig,
    state: ServerState,
    request_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ServerState {
    Stopped,
    Running,
    Paused,
}

impl HttpServer {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            state: ServerState::Stopped,
            request_count: 0,
        }
    }

    pub fn start(&mut self) -> Result<(), String> {
        if self.state == ServerState::Running {
            return Err("server already running".into());
        }
        self.state = ServerState::Running;
        Ok(())
    }

    pub fn stop(&mut self) {
        self.state = ServerState::Stopped;
    }

    pub fn pause(&mut self) {
        if self.state == ServerState::Running {
            self.state = ServerState::Paused;
        }
    }

    pub fn resume(&mut self) {
        if self.state == ServerState::Paused {
            self.state = ServerState::Running;
        }
    }

    pub fn state(&self) -> &ServerState {
        &self.state
    }

    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    pub fn record_request(&mut self) {
        self.request_count += 1;
    }

    pub fn request_count(&self) -> u64 {
        self.request_count
    }
}

// ---------------------------------------------------------------------------
// Advanced Request / Response
// ---------------------------------------------------------------------------

/// Parsed HTTP request with body support.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub version: String,
    pub headers: HashMap<String, String>,
    pub query_params: HashMap<String, String>,
    pub path_params: HashMap<String, String>,
    pub body: Vec<u8>,
    pub remote_addr: Option<String>,
    pub timestamp: u64,
}

impl HttpRequest {
    pub fn new(method: &str, path: &str) -> Self {
        Self {
            method: method.to_string(),
            path: path.to_string(),
            version: "HTTP/1.1".to_string(),
            headers: HashMap::new(),
            query_params: HashMap::new(),
            path_params: HashMap::new(),
            body: Vec::new(),
            remote_addr: None,
            timestamp: 0,
        }
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_lowercase())
            .map(|s| s.as_str())
    }

    pub fn set_header(&mut self, name: &str, value: &str) {
        self.headers
            .insert(name.to_lowercase(), value.to_string());
    }

    pub fn query(&self, name: &str) -> Option<&str> {
        self.query_params.get(name).map(|s| s.as_str())
    }

    pub fn param(&self, name: &str) -> Option<&str> {
        self.path_params.get(name).map(|s| s.as_str())
    }

    pub fn body_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.body).ok()
    }

    pub fn content_length(&self) -> usize {
        self.body.len()
    }

    pub fn is_json(&self) -> bool {
        self.header("content-type")
            .map_or(false, |ct| ct.contains("application/json"))
    }

    pub fn accept(&self) -> &str {
        self.header("accept").unwrap_or("*/*")
    }

    pub fn user_agent(&self) -> &str {
        self.header("user-agent").unwrap_or("unknown")
    }
}

/// HTTP response builder.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub status_text: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn new(status: u16) -> Self {
        let status_text = status_text_for(status);
        Self {
            status,
            status_text: status_text.to_string(),
            headers: HashMap::new(),
            body: Vec::new(),
        }
    }

    pub fn ok() -> Self {
        Self::new(200)
    }

    pub fn created() -> Self {
        Self::new(201)
    }

    pub fn no_content() -> Self {
        Self::new(204)
    }

    pub fn bad_request() -> Self {
        Self::new(400).with_text_body("Bad Request")
    }

    pub fn unauthorized() -> Self {
        Self::new(401).with_text_body("Unauthorized")
    }

    pub fn forbidden() -> Self {
        Self::new(403).with_text_body("Forbidden")
    }

    pub fn not_found() -> Self {
        Self::new(404).with_text_body("Not Found")
    }

    pub fn internal_error() -> Self {
        Self::new(500).with_text_body("Internal Server Error")
    }

    pub fn json(status: u16, body: &str) -> Self {
        Self::new(status)
            .with_header("Content-Type", "application/json")
            .with_text_body(body)
    }

    pub fn html(status: u16, body: &str) -> Self {
        Self::new(status)
            .with_header("Content-Type", "text/html; charset=utf-8")
            .with_text_body(body)
    }

    pub fn redirect(url: &str) -> Self {
        Self::new(302).with_header("Location", url)
    }

    pub fn permanent_redirect(url: &str) -> Self {
        Self::new(301).with_header("Location", url)
    }

    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.insert(name.to_string(), value.to_string());
        self
    }

    pub fn with_text_body(mut self, body: &str) -> Self {
        self.body = body.as_bytes().to_vec();
        self
    }

    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body = body;
        self
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(|s| s.as_str())
    }

    pub fn body_str(&self) -> &str {
        std::str::from_utf8(&self.body).unwrap_or("")
    }

    /// Serialize the response to raw HTTP bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = format!("HTTP/1.1 {} {}\r\n", self.status, self.status_text);
        let mut headers = self.headers.clone();
        headers
            .entry("Content-Length".to_string())
            .or_insert_with(|| self.body.len().to_string());
        for (k, v) in &headers {
            out.push_str(&format!("{}: {}\r\n", k, v));
        }
        out.push_str("\r\n");
        let mut bytes = out.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }
}

fn status_text_for(code: u16) -> &'static str {
    match code {
        100 => "Continue",
        101 => "Switching Protocols",
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

// ---------------------------------------------------------------------------
// Advanced Router with path parameters and route prefixes
// ---------------------------------------------------------------------------

/// A single route entry.
#[derive(Debug, Clone)]
pub struct AdvancedRoute {
    pub method: String,
    pub pattern: String,
    pub handler: String,
    pub middleware: Vec<String>,
}

/// Router supporting method-based registration, groups, and path params.
#[derive(Debug, Clone)]
pub struct AdvancedRouter {
    routes: Vec<AdvancedRoute>,
    global_middleware: Vec<String>,
    prefix: String,
}

impl AdvancedRouter {
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            global_middleware: Vec::new(),
            prefix: String::new(),
        }
    }

    pub fn with_prefix(prefix: &str) -> Self {
        Self {
            routes: Vec::new(),
            global_middleware: Vec::new(),
            prefix: prefix.to_string(),
        }
    }

    pub fn get(mut self, path: &str, handler: &str) -> Self {
        self.add_route("GET", path, handler);
        self
    }

    pub fn post(mut self, path: &str, handler: &str) -> Self {
        self.add_route("POST", path, handler);
        self
    }

    pub fn put(mut self, path: &str, handler: &str) -> Self {
        self.add_route("PUT", path, handler);
        self
    }

    pub fn delete(mut self, path: &str, handler: &str) -> Self {
        self.add_route("DELETE", path, handler);
        self
    }

    pub fn patch(mut self, path: &str, handler: &str) -> Self {
        self.add_route("PATCH", path, handler);
        self
    }

    pub fn middleware(mut self, name: &str) -> Self {
        self.global_middleware.push(name.to_string());
        self
    }

    pub fn group(mut self, prefix: &str, sub: AdvancedRouter) -> Self {
        let full_prefix = format!("{}{}", self.prefix, prefix);
        for mut route in sub.routes {
            route.pattern = format!("{}{}", full_prefix, route.pattern);
            self.routes.push(route);
        }
        self
    }

    fn add_route(&mut self, method: &str, path: &str, handler: &str) {
        let full = format!("{}{}", self.prefix, path);
        self.routes.push(AdvancedRoute {
            method: method.to_string(),
            pattern: full,
            handler: handler.to_string(),
            middleware: Vec::new(),
        });
    }

    /// Match a request to a route, extracting path parameters.
    pub fn resolve(&self, method: &str, path: &str) -> Option<MatchResult> {
        for route in &self.routes {
            if route.method != method {
                continue;
            }
            if let Some(params) = match_pattern(&route.pattern, path) {
                let mut mw = self.global_middleware.clone();
                mw.extend(route.middleware.clone());
                return Some(MatchResult {
                    handler: route.handler.clone(),
                    params,
                    middleware: mw,
                });
            }
        }
        None
    }

    pub fn routes(&self) -> &[AdvancedRoute] {
        &self.routes
    }

    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub handler: String,
    pub params: HashMap<String, String>,
    pub middleware: Vec<String>,
}

/// Match a URL pattern (with `:param` and `*glob`) against a concrete path.
fn match_pattern(pattern: &str, path: &str) -> Option<HashMap<String, String>> {
    let pat_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();

    // Support trailing wildcard that consumes the rest.
    if pat_parts.last().map_or(false, |p| p.starts_with('*')) {
        if path_parts.len() < pat_parts.len() - 1 {
            return None;
        }
        let mut params = HashMap::new();
        for (pp, pv) in pat_parts[..pat_parts.len() - 1].iter().zip(path_parts.iter()) {
            if pp.starts_with(':') {
                params.insert(pp[1..].to_string(), pv.to_string());
            } else if pp != pv {
                return None;
            }
        }
        let glob_name = &pat_parts.last().unwrap()[1..];
        let rest: Vec<&str> = path_parts[pat_parts.len() - 1..].to_vec();
        params.insert(glob_name.to_string(), rest.join("/"));
        return Some(params);
    }

    if pat_parts.len() != path_parts.len() {
        return None;
    }

    let mut params = HashMap::new();
    for (pp, pv) in pat_parts.iter().zip(path_parts.iter()) {
        if pp.starts_with(':') {
            params.insert(pp[1..].to_string(), pv.to_string());
        } else if pp != pv {
            return None;
        }
    }
    Some(params)
}

// ---------------------------------------------------------------------------
// Middleware chain
// ---------------------------------------------------------------------------

/// A composable middleware chain that processes requests through an ordered
/// list of stages.
pub struct MiddlewareChain {
    stages: Vec<MiddlewareStage>,
}

#[derive(Debug, Clone)]
pub struct MiddlewareStage {
    pub name: String,
    pub priority: i32,
    pub enabled: bool,
}

impl MiddlewareChain {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    pub fn add(&mut self, name: &str, priority: i32) {
        self.stages.push(MiddlewareStage {
            name: name.to_string(),
            priority,
            enabled: true,
        });
    }

    pub fn disable(&mut self, name: &str) {
        if let Some(s) = self.stages.iter_mut().find(|s| s.name == name) {
            s.enabled = false;
        }
    }

    pub fn enable(&mut self, name: &str) {
        if let Some(s) = self.stages.iter_mut().find(|s| s.name == name) {
            s.enabled = true;
        }
    }

    /// Return the enabled stages sorted by priority (lowest first).
    pub fn ordered(&self) -> Vec<&MiddlewareStage> {
        let mut v: Vec<&MiddlewareStage> =
            self.stages.iter().filter(|s| s.enabled).collect();
        v.sort_by_key(|s| s.priority);
        v
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

// ---------------------------------------------------------------------------
// WebSocket support
// ---------------------------------------------------------------------------

/// A WebSocket frame.
#[derive(Debug, Clone, PartialEq)]
pub struct WsFrame {
    pub opcode: WsOpcode,
    pub payload: Vec<u8>,
    pub fin: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WsOpcode {
    Text,
    Binary,
    Close,
    Ping,
    Pong,
}

impl WsOpcode {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b & 0x0F {
            0x01 => Some(WsOpcode::Text),
            0x02 => Some(WsOpcode::Binary),
            0x08 => Some(WsOpcode::Close),
            0x09 => Some(WsOpcode::Ping),
            0x0A => Some(WsOpcode::Pong),
            _ => None,
        }
    }

    pub fn to_byte(&self) -> u8 {
        match self {
            WsOpcode::Text => 0x01,
            WsOpcode::Binary => 0x02,
            WsOpcode::Close => 0x08,
            WsOpcode::Ping => 0x09,
            WsOpcode::Pong => 0x0A,
        }
    }
}

impl WsFrame {
    pub fn text(payload: &str) -> Self {
        Self {
            opcode: WsOpcode::Text,
            payload: payload.as_bytes().to_vec(),
            fin: true,
        }
    }

    pub fn binary(payload: &[u8]) -> Self {
        Self {
            opcode: WsOpcode::Binary,
            payload: payload.to_vec(),
            fin: true,
        }
    }

    pub fn close(code: u16, reason: &str) -> Self {
        let mut payload = code.to_be_bytes().to_vec();
        payload.extend_from_slice(reason.as_bytes());
        Self {
            opcode: WsOpcode::Close,
            payload,
            fin: true,
        }
    }

    pub fn ping(payload: &[u8]) -> Self {
        Self {
            opcode: WsOpcode::Ping,
            payload: payload.to_vec(),
            fin: true,
        }
    }

    pub fn pong(payload: &[u8]) -> Self {
        Self {
            opcode: WsOpcode::Pong,
            payload: payload.to_vec(),
            fin: true,
        }
    }

    pub fn payload_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.payload).ok()
    }

    /// Encode the frame into bytes (server-to-client, no masking).
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let first = if self.fin { 0x80 } else { 0x00 } | self.opcode.to_byte();
        out.push(first);

        let len = self.payload.len();
        if len < 126 {
            out.push(len as u8);
        } else if len < 65536 {
            out.push(126);
            out.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            out.push(127);
            out.extend_from_slice(&(len as u64).to_be_bytes());
        }

        out.extend_from_slice(&self.payload);
        out
    }
}

/// Tracks a WebSocket connection's metadata.
#[derive(Debug)]
pub struct WebSocketConnection {
    pub id: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub state: WsState,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub connected_at: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WsState {
    Connecting,
    Open,
    Closing,
    Closed,
}

impl WebSocketConnection {
    pub fn new(id: &str, path: &str) -> Self {
        Self {
            id: id.to_string(),
            path: path.to_string(),
            headers: HashMap::new(),
            state: WsState::Connecting,
            messages_sent: 0,
            messages_received: 0,
            connected_at: 0,
        }
    }

    pub fn open(&mut self) {
        self.state = WsState::Open;
    }

    pub fn close(&mut self) {
        self.state = WsState::Closed;
    }

    pub fn is_open(&self) -> bool {
        self.state == WsState::Open
    }

    pub fn record_sent(&mut self) {
        self.messages_sent += 1;
    }

    pub fn record_received(&mut self) {
        self.messages_received += 1;
    }
}

/// Simple hub that manages a set of WebSocket connections.
#[derive(Debug)]
pub struct WebSocketHub {
    connections: HashMap<String, WebSocketConnection>,
}

impl WebSocketHub {
    pub fn new() -> Self {
        Self {
            connections: HashMap::new(),
        }
    }

    pub fn register(&mut self, conn: WebSocketConnection) {
        self.connections.insert(conn.id.clone(), conn);
    }

    pub fn remove(&mut self, id: &str) -> Option<WebSocketConnection> {
        self.connections.remove(id)
    }

    pub fn get(&self, id: &str) -> Option<&WebSocketConnection> {
        self.connections.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut WebSocketConnection> {
        self.connections.get_mut(id)
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    pub fn active_ids(&self) -> Vec<&str> {
        self.connections
            .values()
            .filter(|c| c.is_open())
            .map(|c| c.id.as_str())
            .collect()
    }

    /// Broadcast a text message to all open connections.
    pub fn broadcast(&mut self, message: &str) {
        let frame = WsFrame::text(message);
        let _ = frame; // encoding would happen over real transport
        for conn in self.connections.values_mut() {
            if conn.is_open() {
                conn.record_sent();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Server ---------------------------------------------------------------

    #[test]
    fn test_server_lifecycle() {
        let cfg = ServerConfig::new("127.0.0.1", 8080)
            .max_connections(256)
            .request_timeout(60);
        let mut srv = HttpServer::new(cfg);
        assert_eq!(*srv.state(), ServerState::Stopped);

        srv.start().unwrap();
        assert_eq!(*srv.state(), ServerState::Running);
        assert_eq!(srv.start().err().unwrap(), "server already running");

        srv.pause();
        assert_eq!(*srv.state(), ServerState::Paused);
        srv.resume();
        assert_eq!(*srv.state(), ServerState::Running);

        srv.stop();
        assert_eq!(*srv.state(), ServerState::Stopped);
    }

    #[test]
    fn test_server_config() {
        let cfg = ServerConfig::new("0.0.0.0", 3000)
            .keep_alive(false)
            .max_body_size(4096);
        assert_eq!(cfg.bind_address(), "0.0.0.0:3000");
        assert!(!cfg.keep_alive);
        assert_eq!(cfg.max_body_size, 4096);
    }

    #[test]
    fn test_server_request_count() {
        let cfg = ServerConfig::new("localhost", 0);
        let mut srv = HttpServer::new(cfg);
        assert_eq!(srv.request_count(), 0);
        srv.record_request();
        srv.record_request();
        assert_eq!(srv.request_count(), 2);
    }

    // -- Request / Response ---------------------------------------------------

    #[test]
    fn test_request_basics() {
        let mut req = HttpRequest::new("POST", "/api/data");
        req.set_header("Content-Type", "application/json");
        req.set_header("Authorization", "Bearer tok");
        req.body = b"{\"x\":1}".to_vec();

        assert!(req.is_json());
        assert_eq!(req.header("authorization"), Some("Bearer tok"));
        assert_eq!(req.body_str().unwrap(), "{\"x\":1}");
        assert_eq!(req.content_length(), 7);
    }

    #[test]
    fn test_response_builder() {
        let res = HttpResponse::ok()
            .with_header("X-Request-Id", "abc")
            .with_text_body("hello");
        assert_eq!(res.status, 200);
        assert_eq!(res.status_text, "OK");
        assert_eq!(res.header("X-Request-Id"), Some("abc"));
        assert_eq!(res.body_str(), "hello");
    }

    #[test]
    fn test_response_json() {
        let res = HttpResponse::json(201, r#"{"id":1}"#);
        assert_eq!(res.status, 201);
        assert_eq!(res.header("Content-Type"), Some("application/json"));
    }

    #[test]
    fn test_response_to_bytes() {
        let res = HttpResponse::ok().with_text_body("OK");
        let raw = String::from_utf8(res.to_bytes()).unwrap();
        assert!(raw.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(raw.contains("Content-Length: 2\r\n"));
        assert!(raw.ends_with("OK"));
    }

    #[test]
    fn test_status_texts() {
        assert_eq!(status_text_for(404), "Not Found");
        assert_eq!(status_text_for(429), "Too Many Requests");
        assert_eq!(status_text_for(999), "Unknown");
    }

    #[test]
    fn test_redirect_responses() {
        let r301 = HttpResponse::permanent_redirect("/new");
        assert_eq!(r301.status, 301);
        assert_eq!(r301.header("Location"), Some("/new"));

        let r302 = HttpResponse::redirect("/login");
        assert_eq!(r302.status, 302);
    }

    // -- Router ---------------------------------------------------------------

    #[test]
    fn test_router_resolve() {
        let router = AdvancedRouter::new()
            .get("/", "index")
            .get("/users/:id", "get_user")
            .post("/users", "create_user")
            .put("/users/:id", "update_user")
            .delete("/users/:id", "delete_user");

        let m = router.resolve("GET", "/").unwrap();
        assert_eq!(m.handler, "index");

        let m = router.resolve("GET", "/users/42").unwrap();
        assert_eq!(m.handler, "get_user");
        assert_eq!(m.params.get("id").unwrap(), "42");

        let m = router.resolve("DELETE", "/users/7").unwrap();
        assert_eq!(m.handler, "delete_user");

        assert!(router.resolve("GET", "/missing").is_none());
        assert!(router.resolve("PATCH", "/").is_none());
    }

    #[test]
    fn test_router_prefix() {
        let router = AdvancedRouter::with_prefix("/api/v1")
            .get("/items", "list_items")
            .get("/items/:id", "get_item");

        let m = router.resolve("GET", "/api/v1/items").unwrap();
        assert_eq!(m.handler, "list_items");

        let m = router.resolve("GET", "/api/v1/items/99").unwrap();
        assert_eq!(m.params.get("id").unwrap(), "99");
    }

    #[test]
    fn test_router_group() {
        let api = AdvancedRouter::new()
            .get("/users", "list_users")
            .post("/users", "create_user");

        let router = AdvancedRouter::new().group("/api", api);

        assert!(router.resolve("GET", "/api/users").is_some());
        assert!(router.resolve("POST", "/api/users").is_some());
        assert!(router.resolve("GET", "/users").is_none());
    }

    #[test]
    fn test_glob_wildcard() {
        let router = AdvancedRouter::new().get("/files/*path", "serve_file");

        let m = router.resolve("GET", "/files/docs/readme.md").unwrap();
        assert_eq!(m.handler, "serve_file");
        assert_eq!(m.params.get("path").unwrap(), "docs/readme.md");
    }

    // -- Middleware chain ------------------------------------------------------

    #[test]
    fn test_middleware_ordering() {
        let mut chain = MiddlewareChain::new();
        chain.add("cors", 10);
        chain.add("auth", 5);
        chain.add("logging", 1);

        let ordered = chain.ordered();
        let names: Vec<&str> = ordered.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["logging", "auth", "cors"]);
    }

    #[test]
    fn test_middleware_disable() {
        let mut chain = MiddlewareChain::new();
        chain.add("cors", 10);
        chain.add("auth", 5);
        chain.disable("auth");

        assert_eq!(chain.ordered().len(), 1);
        assert_eq!(chain.ordered()[0].name, "cors");

        chain.enable("auth");
        assert_eq!(chain.ordered().len(), 2);
    }

    // -- WebSocket ------------------------------------------------------------

    #[test]
    fn test_ws_frame_encode_text() {
        let frame = WsFrame::text("hello");
        assert!(frame.fin);
        assert_eq!(frame.opcode, WsOpcode::Text);
        assert_eq!(frame.payload_str().unwrap(), "hello");

        let encoded = frame.encode();
        assert_eq!(encoded[0], 0x81); // fin + text opcode
        assert_eq!(encoded[1], 5);    // payload length
        assert_eq!(&encoded[2..], b"hello");
    }

    #[test]
    fn test_ws_frame_binary() {
        let frame = WsFrame::binary(&[0x01, 0x02, 0x03]);
        let encoded = frame.encode();
        assert_eq!(encoded[0], 0x82); // fin + binary opcode
        assert_eq!(encoded[1], 3);
    }

    #[test]
    fn test_ws_frame_close() {
        let frame = WsFrame::close(1000, "bye");
        assert_eq!(frame.opcode, WsOpcode::Close);
        assert_eq!(frame.payload_str().unwrap(), "\x00\x00bye"); // code bytes + reason
    }

    #[test]
    fn test_ws_frame_ping_pong() {
        let ping = WsFrame::ping(b"hi");
        let pong = WsFrame::pong(b"hi");
        assert_eq!(ping.opcode, WsOpcode::Ping);
        assert_eq!(pong.opcode, WsOpcode::Pong);
        assert_eq!(pong.payload, b"hi");
    }

    #[test]
    fn test_ws_opcode_from_byte() {
        assert_eq!(WsOpcode::from_byte(0x01), Some(WsOpcode::Text));
        assert_eq!(WsOpcode::from_byte(0x02), Some(WsOpcode::Binary));
        assert_eq!(WsOpcode::from_byte(0x08), Some(WsOpcode::Close));
        assert_eq!(WsOpcode::from_byte(0x09), Some(WsOpcode::Ping));
        assert_eq!(WsOpcode::from_byte(0x0A), Some(WsOpcode::Pong));
        assert_eq!(WsOpcode::from_byte(0x0F), None);
    }

    #[test]
    fn test_ws_connection_lifecycle() {
        let mut conn = WebSocketConnection::new("ws-1", "/chat");
        assert_eq!(conn.state, WsState::Connecting);
        assert!(!conn.is_open());

        conn.open();
        assert!(conn.is_open());

        conn.record_sent();
        conn.record_received();
        conn.record_sent();
        assert_eq!(conn.messages_sent, 2);
        assert_eq!(conn.messages_received, 1);

        conn.close();
        assert_eq!(conn.state, WsState::Closed);
        assert!(!conn.is_open());
    }

    #[test]
    fn test_ws_hub() {
        let mut hub = WebSocketHub::new();
        assert_eq!(hub.connection_count(), 0);

        let mut c1 = WebSocketConnection::new("a", "/ws");
        c1.open();
        let mut c2 = WebSocketConnection::new("b", "/ws");
        c2.open();
        hub.register(c1);
        hub.register(c2);
        assert_eq!(hub.connection_count(), 2);

        let ids = hub.active_ids();
        assert_eq!(ids.len(), 2);

        hub.broadcast("hello everyone");
        assert_eq!(hub.get("a").unwrap().messages_sent, 1);
        assert_eq!(hub.get("b").unwrap().messages_sent, 1);

        hub.remove("a");
        assert_eq!(hub.connection_count(), 1);
        assert!(hub.get("a").is_none());
    }

    #[test]
    fn test_ws_frame_long_payload() {
        // Payload >= 126 bytes uses 2-byte length header
        let big = vec![b'x'; 300];
        let frame = WsFrame::binary(&big);
        let encoded = frame.encode();
        assert_eq!(encoded[1], 126);
        let len = u16::from_be_bytes([encoded[2], encoded[3]]);
        assert_eq!(len, 300);
    }
}
