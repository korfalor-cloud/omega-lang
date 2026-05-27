use std::net::{UdpSocket, SocketAddr};
use std::time::Duration;
use crate::errors::{OmegaError, OmegaResult};

pub struct OmegaUdpSocket {
    socket: UdpSocket,
}

impl OmegaUdpSocket {
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

    pub fn set_read_timeout(&self, timeout_ms: Option<u64>) -> OmegaResult<()> {
        self.socket.set_read_timeout(timeout_ms.map(Duration::from_millis))
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })
    }

    pub fn set_write_timeout(&self, timeout_ms: Option<u64>) -> OmegaResult<()> {
        self.socket.set_write_timeout(timeout_ms.map(Duration::from_millis))
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })
    }

    pub fn set_broadcast(&self, broadcast: bool) -> OmegaResult<()> {
        self.socket.set_broadcast(broadcast).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn join_multicast(&self, addr: &str) -> OmegaResult<()> {
        let addr: SocketAddr = addr.parse().map_err(|_| OmegaError::NetworkError {
            message: format!("Invalid multicast address: {}", addr),
        })?;
        if let std::net::IpAddr::V4(v4) = addr.ip() {
            self.socket.join_multicast_v4(&v4, &std::net::Ipv4Addr::UNSPECIFIED)
                .map_err(|e| OmegaError::NetworkError { message: e.to_string() })
        } else {
            Err(OmegaError::NetworkError {
                message: "IPv6 multicast not supported".to_string(),
            })
        }
    }

    pub fn local_addr(&self) -> OmegaResult<SocketAddr> {
        self.socket.local_addr().map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }
}
