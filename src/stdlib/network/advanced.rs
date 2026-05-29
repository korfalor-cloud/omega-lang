use std::collections::HashMap;
use std::io::{Read, Write, BufRead, BufReader};
use std::net::{TcpListener, TcpStream, UdpSocket, SocketAddr, ToSocketAddrs, Shutdown};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use crate::errors::{OmegaError, OmegaResult};

// ---------------------------------------------------------------------------
// TCP Client with auto-reconnect
// ---------------------------------------------------------------------------

pub struct TcpClient {
    addr: String,
    stream: Option<TcpStream>,
    connect_timeout_ms: u64,
    read_timeout_ms: Option<u64>,
    write_timeout_ms: Option<u64>,
    max_retries: u32,
}

impl TcpClient {
    pub fn new(addr: &str) -> Self {
        Self {
            addr: addr.to_string(),
            stream: None,
            connect_timeout_ms: 5000,
            read_timeout_ms: None,
            write_timeout_ms: None,
            max_retries: 3,
        }
    }

    pub fn connect_timeout(mut self, ms: u64) -> Self {
        self.connect_timeout_ms = ms;
        self
    }

    pub fn read_timeout(mut self, ms: u64) -> Self {
        self.read_timeout_ms = Some(ms);
        self
    }

    pub fn write_timeout(mut self, ms: u64) -> Self {
        self.write_timeout_ms = Some(ms);
        self
    }

    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    fn do_connect(&self) -> OmegaResult<TcpStream> {
        let sock_addr: SocketAddr = self.addr.parse().map_err(|_| OmegaError::NetworkError {
            message: format!("Invalid address: {}", self.addr),
        })?;
        let stream = TcpStream::connect_timeout(&sock_addr, Duration::from_millis(self.connect_timeout_ms))
            .map_err(|e| OmegaError::NetworkError {
                message: format!("Connection failed: {}", e),
            })?;
        if let Some(ms) = self.read_timeout_ms {
            let _ = stream.set_read_timeout(Some(Duration::from_millis(ms)));
        }
        if let Some(ms) = self.write_timeout_ms {
            let _ = stream.set_write_timeout(Some(Duration::from_millis(ms)));
        }
        Ok(stream)
    }

    pub fn connect(&mut self) -> OmegaResult<()> {
        let mut last_err = String::new();
        for _ in 0..self.max_retries {
            match self.do_connect() {
                Ok(s) => { self.stream = Some(s); return Ok(()); }
                Err(e) => { last_err = e.to_string(); }
            }
        }
        Err(OmegaError::NetworkError {
            message: format!("Failed after {} retries: {}", self.max_retries, last_err),
        })
    }

    pub fn send(&mut self, data: &[u8]) -> OmegaResult<usize> {
        if self.stream.is_none() { self.connect()?; }
        let stream = self.stream.as_mut().unwrap();
        stream.write(data).map_err(|e| {
            self.stream = None;
            OmegaError::NetworkError { message: e.to_string() }
        })
    }

    pub fn send_all(&mut self, data: &[u8]) -> OmegaResult<()> {
        if self.stream.is_none() { self.connect()?; }
        let stream = self.stream.as_mut().unwrap();
        stream.write_all(data).map_err(|e| {
            self.stream = None;
            OmegaError::NetworkError { message: e.to_string() }
        })
    }

    pub fn recv(&mut self, buf: &mut [u8]) -> OmegaResult<usize> {
        if self.stream.is_none() { self.connect()?; }
        let stream = self.stream.as_mut().unwrap();
        stream.read(buf).map_err(|e| {
            self.stream = None;
            OmegaError::NetworkError { message: e.to_string() }
        })
    }

    pub fn recv_line(&mut self) -> OmegaResult<String> {
        if self.stream.is_none() { self.connect()?; }
        let mut line = String::new();
        {
            let stream = self.stream.as_mut().unwrap();
            let mut reader = BufReader::new(stream);
            reader.read_line(&mut line).map_err(|e| {
                self.stream = None;
                OmegaError::NetworkError { message: e.to_string() }
            })?;
        }
        Ok(line)
    }

    pub fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    pub fn close(&mut self) -> OmegaResult<()> {
        if let Some(ref s) = self.stream {
            s.shutdown(Shutdown::Both).map_err(|e| OmegaError::NetworkError {
                message: e.to_string(),
            })?;
        }
        self.stream = None;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TCP Server with per-connection handler
// ---------------------------------------------------------------------------

pub struct TcpServer {
    listener: TcpListener,
}

pub struct TcpConnection {
    stream: TcpStream,
    peer: SocketAddr,
}

impl TcpConnection {
    pub fn peer_addr(&self) -> SocketAddr { self.peer }

    pub fn send(&mut self, data: &[u8]) -> OmegaResult<usize> {
        self.stream.write(data).map_err(|e| OmegaError::NetworkError { message: e.to_string() })
    }

    pub fn send_all(&mut self, data: &[u8]) -> OmegaResult<()> {
        self.stream.write_all(data).map_err(|e| OmegaError::NetworkError { message: e.to_string() })
    }

    pub fn recv(&mut self, buf: &mut [u8]) -> OmegaResult<usize> {
        self.stream.read(buf).map_err(|e| OmegaError::NetworkError { message: e.to_string() })
    }

    pub fn recv_line(&mut self) -> OmegaResult<String> {
        let mut line = String::new();
        BufReader::new(&self.stream).read_line(&mut line)
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })?;
        Ok(line)
    }

    pub fn set_read_timeout(&self, ms: u64) -> OmegaResult<()> {
        self.stream.set_read_timeout(Some(Duration::from_millis(ms)))
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })
    }

    pub fn close(&self) -> OmegaResult<()> {
        self.stream.shutdown(Shutdown::Both)
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })
    }
}

impl TcpServer {
    pub fn bind(addr: &str) -> OmegaResult<Self> {
        let listener = TcpListener::bind(addr).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })?;
        Ok(Self { listener })
    }

    pub fn local_addr(&self) -> OmegaResult<SocketAddr> {
        self.listener.local_addr().map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn accept(&self) -> OmegaResult<TcpConnection> {
        let (stream, peer) = self.listener.accept().map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })?;
        Ok(TcpConnection { stream, peer })
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> OmegaResult<()> {
        self.listener.set_nonblocking(nonblocking).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn incoming(&self) -> impl Iterator<Item = OmegaResult<TcpConnection>> + '_ {
        self.listener.incoming().map(|r| {
            let (stream, peer) = r.map_err(|e| OmegaError::NetworkError {
                message: e.to_string(),
            })?;
            Ok(TcpConnection { stream, peer })
        })
    }
}

// ---------------------------------------------------------------------------
// UDP Socket wrapper with helpers
// ---------------------------------------------------------------------------

pub struct AdvancedUdpSocket {
    socket: UdpSocket,
}

impl AdvancedUdpSocket {
    pub fn bind(addr: &str) -> OmegaResult<Self> {
        let socket = UdpSocket::bind(addr).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })?;
        Ok(Self { socket })
    }

    pub fn connect(&self, addr: &str) -> OmegaResult<()> {
        self.socket.connect(addr).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn send(&self, data: &[u8]) -> OmegaResult<usize> {
        self.socket.send(data).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn send_to(&self, data: &[u8], addr: &str) -> OmegaResult<usize> {
        self.socket.send_to(data, addr).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn recv(&self, buf: &mut [u8]) -> OmegaResult<usize> {
        self.socket.recv(buf).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn recv_from(&self, buf: &mut [u8]) -> OmegaResult<(usize, SocketAddr)> {
        self.socket.recv_from(buf).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn recv_timeout(&self, buf: &mut [u8], timeout_ms: u64) -> OmegaResult<(usize, SocketAddr)> {
        self.socket.set_read_timeout(Some(Duration::from_millis(timeout_ms)))
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })?;
        self.socket.recv_from(buf).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn set_broadcast(&self, broadcast: bool) -> OmegaResult<()> {
        self.socket.set_broadcast(broadcast).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn join_multicast_v4(&self, multiaddr: &str) -> OmegaResult<()> {
        let addr: std::net::Ipv4Addr = multiaddr.parse().map_err(|_| OmegaError::NetworkError {
            message: format!("Invalid multicast address: {}", multiaddr),
        })?;
        self.socket.join_multicast_v4(&addr, &std::net::Ipv4Addr::UNSPECIFIED)
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })
    }

    pub fn local_addr(&self) -> OmegaResult<SocketAddr> {
        self.socket.local_addr().map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn try_clone(&self) -> OmegaResult<Self> {
        let socket = self.socket.try_clone().map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })?;
        Ok(Self { socket })
    }
}

// ---------------------------------------------------------------------------
// HTTP Client (raw TCP, no external deps)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HttpClient {
    connect_timeout_ms: u64,
    read_timeout_ms: u64,
    user_agent: String,
}

#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status_code: u16,
    pub status_text: String,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl HttpClient {
    pub fn new() -> Self {
        Self {
            connect_timeout_ms: 5000,
            read_timeout_ms: 30000,
            user_agent: "omega-lang/1.0".to_string(),
        }
    }

    pub fn connect_timeout(mut self, ms: u64) -> Self {
        self.connect_timeout_ms = ms;
        self
    }

    pub fn read_timeout(mut self, ms: u64) -> Self {
        self.read_timeout_ms = ms;
        self
    }

    pub fn user_agent(mut self, ua: &str) -> Self {
        self.user_agent = ua.to_string();
        self
    }

    /// Perform a simple HTTP/1.1 GET request.
    pub fn get(&self, url: &str) -> OmegaResult<HttpResponse> {
        let (host, port, path) = parse_url(url)?;
        let addr = format!("{}:{}", host, port);

        let sock_addr: SocketAddr = (host.as_str(), port).to_socket_addrs()
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })?
            .next()
            .ok_or_else(|| OmegaError::NetworkError { message: "DNS resolution returned no results".into() })?;

        let mut stream = TcpStream::connect_timeout(&sock_addr, Duration::from_millis(self.connect_timeout_ms))
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })?;
        stream.set_read_timeout(Some(Duration::from_millis(self.read_timeout_ms)))
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })?;

        let request = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: {}\r\nConnection: close\r\nAccept: */*\r\n\r\n",
            path, host, self.user_agent
        );
        stream.write_all(request.as_bytes()).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })?;

        read_http_response(&mut stream)
    }

    /// Perform a simple HTTP/1.1 POST request with a body.
    pub fn post(&self, url: &str, content_type: &str, body: &str) -> OmegaResult<HttpResponse> {
        let (host, port, path) = parse_url(url)?;
        let sock_addr: SocketAddr = (host.as_str(), port).to_socket_addrs()
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })?
            .next()
            .ok_or_else(|| OmegaError::NetworkError { message: "DNS resolution returned no results".into() })?;

        let mut stream = TcpStream::connect_timeout(&sock_addr, Duration::from_millis(self.connect_timeout_ms))
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })?;
        stream.set_read_timeout(Some(Duration::from_millis(self.read_timeout_ms)))
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })?;

        let request = format!(
            "POST {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            path, host, self.user_agent, content_type, body.len(), body
        );
        stream.write_all(request.as_bytes()).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })?;

        read_http_response(&mut stream)
    }
}

impl Default for HttpClient {
    fn default() -> Self { Self::new() }
}

impl HttpResponse {
    pub fn is_success(&self) -> bool { (200..300).contains(&self.status_code) }
    pub fn is_redirect(&self) -> bool { (300..400).contains(&self.status_code) }
    pub fn is_client_error(&self) -> bool { (400..500).contains(&self.status_code) }
    pub fn is_server_error(&self) -> bool { self.status_code >= 500 }
}

fn parse_url(url: &str) -> OmegaResult<(String, u16, String)> {
    let rest = url.strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    let (host_port, path) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].to_string()),
        None => (rest, "/".to_string()),
    };
    let (host, port) = match host_port.rfind(':') {
        Some(i) => {
            let p: u16 = host_port[i + 1..].parse().map_err(|_| OmegaError::NetworkError {
                message: format!("Invalid port in URL: {}", url),
            })?;
            (host_port[..i].to_string(), p)
        }
        None => (host_port.to_string(), 80),
    };
    Ok((host, port, path))
}

fn read_http_response(stream: &mut TcpStream) -> OmegaResult<HttpResponse> {
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).map_err(|e| OmegaError::NetworkError {
        message: e.to_string(),
    })?;

    let parts: Vec<&str> = status_line.trim().splitn(3, ' ').collect();
    let status_code: u16 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let status_text = parts.get(2).unwrap_or(&"").to_string();

    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() { break; }
        if let Some((key, value)) = trimmed.split_once(':') {
            headers.insert(key.trim().to_lowercase(), value.trim().to_string());
        }
    }

    let mut body = String::new();
    reader.read_to_string(&mut body).map_err(|e| OmegaError::NetworkError {
        message: e.to_string(),
    })?;

    Ok(HttpResponse { status_code, status_text, headers, body })
}

// ---------------------------------------------------------------------------
// DNS Resolver
// ---------------------------------------------------------------------------

pub struct DnsResolver {
    cache: Arc<Mutex<HashMap<String, (Vec<String>, Instant)>>>,
    cache_ttl: Duration,
}

impl DnsResolver {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            cache_ttl: Duration::from_secs(300),
        }
    }

    pub fn cache_ttl(mut self, secs: u64) -> Self {
        self.cache_ttl = Duration::from_secs(secs);
        self
    }

    /// Resolve a hostname to a list of IP addresses, using a TTL cache.
    pub fn resolve(&self, host: &str) -> OmegaResult<Vec<String>> {
        {
            let cache = self.cache.lock().unwrap();
            if let Some((addrs, ts)) = cache.get(host) {
                if ts.elapsed() < self.cache_ttl {
                    return Ok(addrs.clone());
                }
            }
        }
        let addrs: Vec<String> = (host, 0).to_socket_addrs()
            .map_err(|e| OmegaError::NetworkError {
                message: format!("DNS resolution failed for '{}': {}", host, e),
            })?
            .map(|a| a.ip().to_string())
            .collect();
        if addrs.is_empty() {
            return Err(OmegaError::NetworkError {
                message: format!("No addresses found for '{}'", host),
            });
        }
        self.cache.lock().unwrap().insert(host.to_string(), (addrs.clone(), Instant::now()));
        Ok(addrs)
    }

    /// Resolve to a single IP address.
    pub fn resolve_one(&self, host: &str) -> OmegaResult<String> {
        self.resolve(host).map(|mut v| v.remove(0))
    }

    /// Flush the DNS cache.
    pub fn flush_cache(&self) {
        self.cache.lock().unwrap().clear();
    }

    /// Return a snapshot of cache entries.
    pub fn cache_entries(&self) -> Vec<(String, Vec<String>)> {
        self.cache.lock().unwrap()
            .iter()
            .map(|(k, (v, _))| (k.clone(), v.clone()))
            .collect()
    }
}

impl Default for DnsResolver {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Connection Pool (for TCP clients)
// ---------------------------------------------------------------------------

struct PooledConnection {
    stream: TcpStream,
    last_used: Instant,
}

pub struct ConnectionPool {
    addr: String,
    pool: Arc<Mutex<Vec<PooledConnection>>>,
    max_size: usize,
    max_idle_secs: u64,
    connect_timeout_ms: u64,
}

impl ConnectionPool {
    pub fn new(addr: &str) -> Self {
        Self {
            addr: addr.to_string(),
            pool: Arc::new(Mutex::new(Vec::new())),
            max_size: 8,
            max_idle_secs: 300,
            connect_timeout_ms: 5000,
        }
    }

    pub fn max_size(mut self, n: usize) -> Self {
        self.max_size = n;
        self
    }

    pub fn max_idle_secs(mut self, secs: u64) -> Self {
        self.max_idle_secs = secs;
        self
    }

    pub fn connect_timeout_ms(mut self, ms: u64) -> Self {
        self.connect_timeout_ms = ms;
        self
    }

    fn make_connection(&self) -> OmegaResult<TcpStream> {
        let sock_addr: SocketAddr = self.addr.parse().map_err(|_| OmegaError::NetworkError {
            message: format!("Invalid address: {}", self.addr),
        })?;
        TcpStream::connect_timeout(&sock_addr, Duration::from_millis(self.connect_timeout_ms))
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })
    }

    /// Acquire a connection from the pool, creating one if needed.
    pub fn acquire(&self) -> OmegaResult<TcpStream> {
        let mut pool = self.pool.lock().unwrap();
        let now = Instant::now();

        // Evict stale connections
        pool.retain(|c| now.duration_since(c.last_used).as_secs() < self.max_idle_secs);

        if let Some(conn) = pool.pop() {
            return Ok(conn.stream);
        }
        drop(pool);
        self.make_connection()
    }

    /// Return a connection to the pool for reuse.
    pub fn release(&self, stream: TcpStream) {
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < self.max_size {
            pool.push(PooledConnection {
                stream,
                last_used: Instant::now(),
            });
        }
        // else: drop the connection (pool full)
    }

    /// Number of idle connections currently in the pool.
    pub fn idle_count(&self) -> usize {
        self.pool.lock().unwrap().len()
    }

    /// Drain and close all idle connections.
    pub fn drain(&self) {
        self.pool.lock().unwrap().clear();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcp_client_server_echo() {
        let server = TcpServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap().to_string();

        std::thread::spawn(move || {
            let mut conn = server.accept().unwrap();
            let mut buf = [0u8; 128];
            let n = conn.recv(&mut buf).unwrap();
            conn.send_all(&buf[..n]).unwrap();
        });

        let mut client = TcpClient::new(&addr).connect_timeout(1000).max_retries(1);
        client.connect().unwrap();
        client.send_all(b"hello").unwrap();
        let mut buf = [0u8; 128];
        let n = client.recv(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"hello");
        client.close().unwrap();
    }

    #[test]
    fn test_tcp_client_recv_line() {
        let server = TcpServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap().to_string();

        std::thread::spawn(move || {
            let mut conn = server.accept().unwrap();
            conn.send_all(b"line one\nline two\n").unwrap();
        });

        let mut client = TcpClient::new(&addr).connect_timeout(1000).max_retries(1);
        client.connect().unwrap();
        let line = client.recv_line().unwrap();
        assert_eq!(line, "line one\n");
        client.close().unwrap();
    }

    #[test]
    fn test_tcp_server_incoming() {
        let server = TcpServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap().to_string();

        let handle = std::thread::spawn(move || {
            let mut client = TcpClient::new(&addr).connect_timeout(1000).max_retries(1);
            client.connect().unwrap();
            client.send_all(b"ping").unwrap();
            client.close().unwrap();
        });

        for conn in server.incoming().take(1) {
            let mut c = conn.unwrap();
            let mut buf = [0u8; 64];
            let n = c.recv(&mut buf).unwrap();
            assert_eq!(&buf[..n], b"ping");
        }
        handle.join().unwrap();
    }

    #[test]
    fn test_udp_send_recv() {
        let sock1 = AdvancedUdpSocket::bind("127.0.0.1:0").unwrap();
        let sock2 = AdvancedUdpSocket::bind("127.0.0.1:0").unwrap();
        let addr1 = sock1.local_addr().unwrap().to_string();
        let addr2 = sock2.local_addr().unwrap().to_string();

        sock1.send_to(b"udp-hello", &addr2).unwrap();
        let mut buf = [0u8; 64];
        let (n, from) = sock2.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"udp-hello");
        assert_eq!(from.port(), sock1.local_addr().unwrap().port());
    }

    #[test]
    fn test_udp_try_clone() {
        let sock = AdvancedUdpSocket::bind("127.0.0.1:0").unwrap();
        let clone = sock.try_clone().unwrap();
        assert_eq!(
            sock.local_addr().unwrap().port(),
            clone.local_addr().unwrap().port()
        );
    }

    #[test]
    fn test_parse_url_basic() {
        let (host, port, path) = parse_url("http://example.com/foo/bar").unwrap();
        assert_eq!(host, "example.com");
        assert_eq!(port, 80);
        assert_eq!(path, "/foo/bar");
    }

    #[test]
    fn test_parse_url_with_port() {
        let (host, port, path) = parse_url("http://localhost:8080/api").unwrap();
        assert_eq!(host, "localhost");
        assert_eq!(port, 8080);
        assert_eq!(path, "/api");
    }

    #[test]
    fn test_parse_url_no_path() {
        let (host, _port, path) = parse_url("http://example.com").unwrap();
        assert_eq!(host, "example.com");
        assert_eq!(path, "/");
    }

    #[test]
    fn test_dns_resolve_localhost() {
        let resolver = DnsResolver::new();
        let addrs = resolver.resolve("localhost").unwrap();
        assert!(addrs.contains(&"127.0.0.1".to_string()));
    }

    #[test]
    fn test_dns_cache_hit() {
        let resolver = DnsResolver::new();
        let _ = resolver.resolve("localhost").unwrap();
        let entries = resolver.cache_entries();
        assert!(entries.iter().any(|(k, _)| k == "localhost"));
    }

    #[test]
    fn test_dns_flush_cache() {
        let resolver = DnsResolver::new();
        let _ = resolver.resolve("localhost").unwrap();
        assert!(!resolver.cache_entries().is_empty());
        resolver.flush_cache();
        assert!(resolver.cache_entries().is_empty());
    }

    #[test]
    fn test_dns_resolve_one() {
        let resolver = DnsResolver::new();
        let ip = resolver.resolve_one("localhost").unwrap();
        assert!(ip == "127.0.0.1" || ip == "::1");
    }

    #[test]
    fn test_connection_pool_acquire_release() {
        let server = TcpServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap().to_string();

        std::thread::spawn(move || {
            for _ in 0..2 {
                let mut conn = server.accept().unwrap();
                let mut buf = [0u8; 32];
                let n = conn.recv(&mut buf).unwrap();
                conn.send_all(&buf[..n]).unwrap();
            }
        });

        let pool = ConnectionPool::new(&addr).max_size(4).connect_timeout_ms(1000);
        assert_eq!(pool.idle_count(), 0);

        let mut s1 = pool.acquire().unwrap();
        s1.write_all(b"a").unwrap();
        let mut buf = [0u8; 32];
        let n = s1.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"a");

        pool.release(s1);
        assert_eq!(pool.idle_count(), 1);

        let mut s2 = pool.acquire().unwrap();
        s2.write_all(b"b").unwrap();
        let n = s2.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"b");
        pool.release(s2);

        pool.drain();
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn test_http_response_status_helpers() {
        let resp = HttpResponse {
            status_code: 200,
            status_text: "OK".into(),
            headers: HashMap::new(),
            body: String::new(),
        };
        assert!(resp.is_success());
        assert!(!resp.is_redirect());
        assert!(!resp.is_client_error());
        assert!(!resp.is_server_error());
    }

    #[test]
    fn test_tcp_client_reconnect_on_failure() {
        // Client should retry and fail after max_retries
        let client = TcpClient::new("127.0.0.1:1")
            .connect_timeout(100)
            .max_retries(2);
        let result = client.connect();
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Failed after 2 retries"));
    }

    #[test]
    fn test_udp_recv_timeout() {
        let sock = AdvancedUdpSocket::bind("127.0.0.1:0").unwrap();
        let mut buf = [0u8; 64];
        let result = sock.recv_timeout(&mut buf, 50);
        assert!(result.is_err()); // should time out
    }

    #[test]
    fn test_http_client_default() {
        let client = HttpClient::default();
        assert_eq!(client.user_agent, "omega-lang/1.0");
        assert_eq!(client.connect_timeout_ms, 5000);
    }
}
