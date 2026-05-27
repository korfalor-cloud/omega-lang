use std::io::{self, Read, BufRead, BufReader};
use crate::errors::{OmegaError, OmegaResult};

pub struct OmegaReader<R: Read> {
    reader: BufReader<R>,
}

impl<R: Read> OmegaReader<R> {
    pub fn new(inner: R) -> Self {
        Self { reader: BufReader::new(inner) }
    }

    pub fn read_line(&mut self) -> OmegaResult<String> {
        let mut line = String::new();
        self.reader.read_line(&mut line).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(line)
    }

    pub fn read_lines(&mut self) -> OmegaResult<Vec<String>> {
        let mut lines = Vec::new();
        loop {
            let mut line = String::new();
            let bytes_read = self.reader.read_line(&mut line).map_err(|e| OmegaError::IoError(e.to_string()))?;
            if bytes_read == 0 {
                break;
            }
            lines.push(line.trim_end().to_string());
        }
        Ok(lines)
    }

    pub fn read_to_string(&mut self) -> OmegaResult<String> {
        let mut contents = String::new();
        self.reader.read_to_string(&mut contents).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(contents)
    }

    pub fn read_bytes(&mut self, count: usize) -> OmegaResult<Vec<u8>> {
        let mut buffer = vec![0u8; count];
        let bytes_read = self.reader.read(&mut buffer).map_err(|e| OmegaError::IoError(e.to_string()))?;
        buffer.truncate(bytes_read);
        Ok(buffer)
    }

    pub fn read_byte(&mut self) -> OmegaResult<u8> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(buf[0])
    }

    pub fn read_u16(&mut self) -> OmegaResult<u16> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(u16::from_le_bytes(buf))
    }

    pub fn read_u32(&mut self) -> OmegaResult<u32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(u32::from_le_bytes(buf))
    }

    pub fn read_u64(&mut self) -> OmegaResult<u64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(u64::from_le_bytes(buf))
    }

    pub fn read_i32(&mut self) -> OmegaResult<i32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(i32::from_le_bytes(buf))
    }

    pub fn read_i64(&mut self) -> OmegaResult<i64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(i64::from_le_bytes(buf))
    }

    pub fn read_f32(&mut self) -> OmegaResult<f32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(f32::from_le_bytes(buf))
    }

    pub fn read_f64(&mut self) -> OmegaResult<f64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(f64::from_le_bytes(buf))
    }
}
