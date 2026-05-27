use std::net::ToSocketAddrs;
use crate::errors::{OmegaError, OmegaResult};

pub fn resolve(host: &str) -> OmegaResult<Vec<String>> {
    let addrs = (host, 0).to_socket_addrs().map_err(|e| OmegaError::NetworkError {
        message: format!("DNS resolution failed for '{}': {}", host, e),
    })?;
    Ok(addrs.map(|a| a.ip().to_string()).collect())
}

pub fn resolve_one(host: &str) -> OmegaResult<String> {
    let addrs = resolve(host)?;
    addrs.into_iter().next().ok_or_else(|| OmegaError::NetworkError {
        message: format!("No addresses found for '{}'", host),
    })
}

pub fn reverse_lookup(ip: &str) -> OmegaResult<String> {
    use std::net::IpAddr;
    let ip: IpAddr = ip.parse().map_err(|_| OmegaError::ValueError {
        message: format!("Invalid IP address: {}", ip),
    })?;
    let addr = std::net::SocketAddr::new(ip, 0);
    let hostname = addr.to_socket_addrs()
        .map_err(|e| OmegaError::NetworkError { message: e.to_string() })?
        .next()
        .map(|a| a.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    Ok(hostname)
}

pub fn is_valid_ip(ip: &str) -> bool {
    ip.parse::<std::net::IpAddr>().is_ok()
}

pub fn is_valid_ipv4(ip: &str) -> bool {
    ip.parse::<std::net::Ipv4Addr>().is_ok()
}

pub fn is_valid_ipv6(ip: &str) -> bool {
    ip.parse::<std::net::Ipv6Addr>().is_ok()
}
