use base64::{Engine as _, engine::general_purpose};
use crate::errors::{OmegaError, OmegaResult};

pub fn base64_encode(data: &[u8]) -> String {
    general_purpose::STANDARD.encode(data)
}

pub fn base64_decode(s: &str) -> OmegaResult<Vec<u8>> {
    general_purpose::STANDARD.decode(s).map_err(|e| OmegaError::EncodingError {
        message: format!("Base64 decode error: {}", e),
    })
}

pub fn base64url_encode(data: &[u8]) -> String {
    general_purpose::URL_SAFE_NO_PAD.encode(data)
}

pub fn base64url_decode(s: &str) -> OmegaResult<Vec<u8>> {
    general_purpose::URL_SAFE_NO_PAD.decode(s).map_err(|e| OmegaError::EncodingError {
        message: format!("Base64 URL decode error: {}", e),
    })
}

pub fn hex_encode(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}

pub fn hex_decode(s: &str) -> OmegaResult<Vec<u8>> {
    if s.len() % 2 != 0 {
        return Err(OmegaError::EncodingError {
            message: "Hex string must have even length".to_string(),
        });
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| OmegaError::EncodingError {
                message: format!("Hex decode error: {}", e),
            })
        })
        .collect()
}

pub fn url_encode(s: &str) -> String {
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

pub fn url_decode(s: &str) -> OmegaResult<String> {
    let mut result = Vec::new();
    let mut bytes = s.bytes();
    while let Some(byte) = bytes.next() {
        match byte {
            b'+' => result.push(b' '),
            b'%' => {
                let hex: Vec<u8> = bytes.by_ref().take(2).collect();
                if hex.len() != 2 {
                    return Err(OmegaError::EncodingError {
                        message: "Invalid percent encoding".to_string(),
                    });
                }
                let value = u8::from_str_radix(
                    std::str::from_utf8(&hex).unwrap_or(""), 16
                ).map_err(|_| OmegaError::EncodingError {
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

pub fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

pub fn html_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

pub fn compress(data: &[u8]) -> OmegaResult<Vec<u8>> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).map_err(|e| OmegaError::EncodingError {
        message: e.to_string(),
    })?;
    encoder.finish().map_err(|e| OmegaError::EncodingError {
        message: e.to_string(),
    })
}

pub fn decompress(data: &[u8]) -> OmegaResult<Vec<u8>> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let mut decoder = GzDecoder::new(data);
    let mut result = Vec::new();
    decoder.read_to_end(&mut result).map_err(|e| OmegaError::EncodingError {
        message: e.to_string(),
    })?;
    Ok(result)
}
