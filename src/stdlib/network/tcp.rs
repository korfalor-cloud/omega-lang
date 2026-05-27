use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, SocketAddr};
use std::time::Duration;
use crate::errors::{OmegaError, OmegaResult};

pub struct OmegaTcpListener {
    listener: TcpListener,
}

impl OmegaTcpListener {
    pub fn bind(addr: &str) -> OmegaResult<Self> {
        let listener = TcpListener::bind(addr).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })?;
        Ok(Self { listener })
    }

    pub fn accept(&self) -> OmegaResult<(OmegaTcpStream, SocketAddr)> {
        let (stream, addr) = self.listener.accept().map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })?;
        Ok((OmegaTcpStream { stream }, addr))
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> OmegaResult<()> {
        self.listener.set_nonblocking(nonblocking).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn local_addr(&self) -> OmegaResult<SocketAddr> {
        self.listener.local_addr().map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }
}

pub struct OmegaTcpStream {
    stream: TcpStream,
}

impl OmegaTcpStream {
    pub fn connect(addr: &str) -> OmegaResult<Self> {
        let stream = TcpStream::connect(addr).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })?;
        Ok(Self { stream })
    }

    pub fn connect_timeout(addr: &str, timeout_ms: u64) -> OmegaResult<Self> {
        let addr: SocketAddr = addr.parse().map_err(|_| OmegaError::NetworkError {
            message: format!("Invalid address: {}", addr),
        })?;
        let stream = TcpStream::connect_timeout(&addr, Duration::from_millis(timeout_ms))
            .map_err(|e| OmegaError::NetworkError {
                message: e.to_string(),
            })?;
        Ok(Self { stream })
    }

    pub fn read(&mut self, buf: &mut [u8]) -> OmegaResult<usize> {
        self.stream.read(buf).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn read_to_string(&mut self) -> OmegaResult<String> {
        let mut s = String::new();
        self.stream.read_to_string(&mut s).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })?;
        Ok(s)
    }

    pub fn read_exact(&mut self, count: usize) -> OmegaResult<Vec<u8>> {
        let mut buf = vec![0u8; count];
        self.stream.read_exact(&mut buf).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })?;
        Ok(buf)
    }

    pub fn write(&mut self, data: &[u8]) -> OmegaResult<usize> {
        self.stream.write(data).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn write_all(&mut self, data: &[u8]) -> OmegaResult<()> {
        self.stream.write_all(data).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn flush(&mut self) -> OmegaResult<()> {
        self.stream.flush().map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn set_read_timeout(&self, timeout_ms: Option<u64>) -> OmegaResult<()> {
        self.stream.set_read_timeout(timeout_ms.map(Duration::from_millis))
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })
    }

    pub fn set_write_timeout(&self, timeout_ms: Option<u64>) -> OmegaResult<()> {
        self.stream.set_write_timeout(timeout_ms.map(Duration::from_millis))
            .map_err(|e| OmegaError::NetworkError { message: e.to_string() })
    }

    pub fn set_nodelay(&self, nodelay: bool) -> OmegaResult<()> {
        self.stream.set_nodelay(nodelay).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn peer_addr(&self) -> OmegaResult<SocketAddr> {
        self.stream.peer_addr().map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn local_addr(&self) -> OmegaResult<SocketAddr> {
        self.stream.local_addr().map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }

    pub fn shutdown(&self) -> OmegaResult<()> {
        self.stream.shutdown(std::net::Shutdown::Both).map_err(|e| OmegaError::NetworkError {
            message: e.to_string(),
        })
    }
}
