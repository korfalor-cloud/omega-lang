use std::io::{self, BufRead, Read};
use crate::errors::{OmegaError, OmegaResult};

pub struct OmegaStdin {
    reader: io::Stdin,
}

impl OmegaStdin {
    pub fn new() -> Self {
        Self { reader: io::stdin() }
    }

    pub fn read_line(&self) -> OmegaResult<String> {
        let mut line = String::new();
        self.reader.lock().read_line(&mut line).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(line.trim_end().to_string())
    }

    pub fn read_to_string(&self) -> OmegaResult<String> {
        let mut contents = String::new();
        self.reader.lock().read_to_string(&mut contents).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(contents)
    }

    pub fn read_bytes(&self, count: usize) -> OmegaResult<Vec<u8>> {
        let mut buffer = vec![0u8; count];
        let bytes_read = self.reader.lock().read(&mut buffer).map_err(|e| OmegaError::IoError(e.to_string()))?;
        buffer.truncate(bytes_read);
        Ok(buffer)
    }

    pub fn read_char(&self) -> OmegaResult<char> {
        let mut buf = [0u8; 4];
        let mut i = 0;
        loop {
            let bytes_read = self.reader.lock().read(&mut buf[i..i+1]).map_err(|e| OmegaError::IoError(e.to_string()))?;
            if bytes_read == 0 {
                return Err(OmegaError::IoError("End of input".to_string()));
            }
            i += 1;
            if let Ok(s) = std::str::from_utf8(&buf[..i]) {
                if let Some(c) = s.chars().next() {
                    return Ok(c);
                }
            }
            if i >= 4 {
                return Err(OmegaError::EncodingError {
                    message: "Invalid UTF-8".to_string(),
                });
            }
        }
    }

    pub fn read_int(&self) -> OmegaResult<i64> {
        let line = self.read_line()?;
        line.parse().map_err(|_| OmegaError::ValueError {
            message: format!("Cannot parse '{}' as integer", line),
        })
    }

    pub fn read_float(&self) -> OmegaResult<f64> {
        let line = self.read_line()?;
        line.parse().map_err(|_| OmegaError::ValueError {
            message: format!("Cannot parse '{}' as float", line),
        })
    }

    pub fn read_bool(&self) -> OmegaResult<bool> {
        let line = self.read_line()?.to_lowercase();
        match line.as_str() {
            "true" | "yes" | "1" | "y" => Ok(true),
            "false" | "no" | "0" | "n" => Ok(false),
            _ => Err(OmegaError::ValueError {
                message: format!("Cannot parse '{}' as bool", line),
            }),
        }
    }
}
