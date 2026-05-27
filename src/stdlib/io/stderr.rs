use std::io::{self, Write};
use crate::errors::{OmegaError, OmegaResult};

pub struct OmegaStderr {
    writer: io::Stderr,
}

impl OmegaStderr {
    pub fn new() -> Self {
        Self { writer: io::stderr() }
    }

    pub fn write(&self, data: &str) -> OmegaResult<()> {
        self.writer.lock().write_all(data.as_bytes()).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn write_line(&self, data: &str) -> OmegaResult<()> {
        let mut writer = self.writer.lock();
        writer.write_all(data.as_bytes()).map_err(|e| OmegaError::IoError(e.to_string()))?;
        writer.write_all(b"\n").map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(())
    }

    pub fn write_bytes(&self, data: &[u8]) -> OmegaResult<()> {
        self.writer.lock().write_all(data).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn flush(&self) -> OmegaResult<()> {
        self.writer.lock().flush().map_err(|e| OmegaError::IoError(e.to_string()))
    }
}
