use std::io::{self, Write, BufWriter};
use crate::errors::{OmegaError, OmegaResult};

pub struct OmegaWriter<W: Write> {
    writer: BufWriter<W>,
}

impl<W: Write> OmegaWriter<W> {
    pub fn new(inner: W) -> Self {
        Self { writer: BufWriter::new(inner) }
    }

    pub fn write(&mut self, data: &[u8]) -> OmegaResult<usize> {
        self.writer.write(data).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn write_all(&mut self, data: &[u8]) -> OmegaResult<()> {
        self.writer.write_all(data).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn write_string(&mut self, data: &str) -> OmegaResult<()> {
        self.writer.write_all(data.as_bytes()).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn write_line(&mut self, data: &str) -> OmegaResult<()> {
        self.writer.write_all(data.as_bytes()).map_err(|e| OmegaError::IoError(e.to_string()))?;
        self.writer.write_all(b"\n").map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(())
    }

    pub fn write_byte(&mut self, byte: u8) -> OmegaResult<()> {
        self.writer.write_all(&[byte]).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn write_u16(&mut self, value: u16) -> OmegaResult<()> {
        self.writer.write_all(&value.to_le_bytes()).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn write_u32(&mut self, value: u32) -> OmegaResult<()> {
        self.writer.write_all(&value.to_le_bytes()).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn write_u64(&mut self, value: u64) -> OmegaResult<()> {
        self.writer.write_all(&value.to_le_bytes()).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn write_i32(&mut self, value: i32) -> OmegaResult<()> {
        self.writer.write_all(&value.to_le_bytes()).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn write_i64(&mut self, value: i64) -> OmegaResult<()> {
        self.writer.write_all(&value.to_le_bytes()).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn write_f32(&mut self, value: f32) -> OmegaResult<()> {
        self.writer.write_all(&value.to_le_bytes()).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn write_f64(&mut self, value: f64) -> OmegaResult<()> {
        self.writer.write_all(&value.to_le_bytes()).map_err(|e| OmegaError::IoError(e.to_string()))
    }

    pub fn flush(&mut self) -> OmegaResult<()> {
        self.writer.flush().map_err(|e| OmegaError::IoError(e.to_string()))
    }
}
