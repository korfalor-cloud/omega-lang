use crate::errors::{OmegaError, OmegaResult};

#[derive(Debug, Clone)]
pub struct OmegaUrl {
    pub scheme: String,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub path: String,
    pub query: Option<String>,
    pub fragment: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
}

impl OmegaUrl {
    pub fn parse(url: &str) -> OmegaResult<Self> {
        // Simple URL parser
        let (scheme, rest) = if let Some(pos) = url.find("://") {
            (url[..pos].to_string(), &url[pos + 3..])
        } else {
            return Err(OmegaError::ValueError {
                message: format!("Invalid URL: {}", url),
            });
        };

        let (authority, path_with_query) = if let Some(pos) = rest.find('/') {
            (&rest[..pos], &rest[pos..])
        } else {
            (rest, "/")
        };

        let (userinfo, host_port) = if let Some(pos) = authority.find('@') {
            (Some(&authority[..pos]), &authority[pos + 1..])
        } else {
            (None, authority)
        };

        let (username, password) = if let Some(info) = userinfo {
            if let Some(pos) = info.find(':') {
                (Some(info[..pos].to_string()), Some(info[pos + 1..].to_string()))
            } else {
                (Some(info.to_string()), None)
            }
        } else {
            (None, None)
        };

        let (host, port) = if let Some(pos) = host_port.find(':') {
            let port: u16 = host_port[pos + 1..].parse().map_err(|_| OmegaError::ValueError {
                message: format!("Invalid port: {}", &host_port[pos + 1..]),
            })?;
            (Some(host_port[..pos].to_string()), Some(port))
        } else {
            (Some(host_port.to_string()), None)
        };

        let (path, query_fragment) = if let Some(pos) = path_with_query.find('?') {
            (&path_with_query[..pos], &path_with_query[pos + 1..])
        } else {
            (path_with_query, "")
        };

        let (query, fragment) = if let Some(pos) = query_fragment.find('#') {
            (Some(query_fragment[..pos].to_string()), Some(query_fragment[pos + 1..].to_string()))
        } else if query_fragment.is_empty() {
            (None, None)
        } else {
            (Some(query_fragment.to_string()), None)
        };

        Ok(Self {
            scheme,
            host,
            port,
            path: path.to_string(),
            query,
            fragment,
            username,
            password,
        })
    }

    pub fn to_string(&self) -> String {
        let mut url = format!("{}://", self.scheme);
        if let Some(user) = &self.username {
            url.push_str(user);
            if let Some(pass) = &self.password {
                url.push(':');
                url.push_str(pass);
            }
            url.push('@');
        }
        if let Some(host) = &self.host {
            url.push_str(host);
        }
        if let Some(port) = self.port {
            url.push_str(&format!(":{}", port));
        }
        url.push_str(&self.path);
        if let Some(query) = &self.query {
            url.push('?');
            url.push_str(query);
        }
        if let Some(fragment) = &self.fragment {
            url.push('#');
            url.push_str(fragment);
        }
        url
    }

    pub fn base_url(&self) -> String {
        let mut url = format!("{}://", self.scheme);
        if let Some(host) = &self.host {
            url.push_str(host);
        }
        if let Some(port) = self.port {
            url.push_str(&format!(":{}", port));
        }
        url
    }

    pub fn query_params(&self) -> Vec<(String, String)> {
        self.query.as_ref().map(|q| {
            q.split('&')
                .filter_map(|param| {
                    let mut parts = param.splitn(2, '=');
                    let key = parts.next()?.to_string();
                    let value = parts.next().unwrap_or("").to_string();
                    Some((key, value))
                })
                .collect()
        }).unwrap_or_default()
    }

    pub fn join(&self, path: &str) -> OmegaResult<Self> {
        let new_path = if path.starts_with('/') {
            path.to_string()
        } else {
            let base = self.path.rfind('/').map(|i| &self.path[..i + 1]).unwrap_or("/");
            format!("{}{}", base, path)
        };
        Ok(Self {
            scheme: self.scheme.clone(),
            host: self.host.clone(),
            port: self.port,
            path: new_path,
            query: None,
            fragment: None,
            username: self.username.clone(),
            password: self.password.clone(),
        })
    }
}

pub fn encode(s: &str) -> String {
    let mut result = String::new();
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(byte as char);
            }
            b' ' => result.push('+'),
            _ => result.push_str(&format!("%{:02X}", byte)),
        }
    }
    result
}

pub fn decode(s: &str) -> OmegaResult<String> {
    let mut result = Vec::new();
    let mut bytes = s.bytes();
    while let Some(byte) = bytes.next() {
        match byte {
            b'+' => result.push(b' '),
            b'%' => {
                let hex: Vec<u8> = bytes.by_ref().take(2).collect();
                if hex.len() != 2 {
                    return Err(OmegaError::ValueError {
                        message: "Invalid percent encoding".to_string(),
                    });
                }
                let value = u8::from_str_radix(
                    std::str::from_utf8(&hex).unwrap_or(""), 16
                ).map_err(|_| OmegaError::ValueError {
                    message: "Invalid hex in percent encoding".to_string(),
                })?;
                result.push(value);
            }
            _ => result.push(byte),
        }
    }
    String::from_utf8(result).map_err(|e| OmegaError::EncodingError {
        message: e.to_string(),
    })
}
