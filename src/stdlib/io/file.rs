use std::fs;
use std::io::{self, Read, Write, BufRead, BufReader, BufWriter, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;

pub struct OmegaFile {
    path: PathBuf,
    handle: Option<fs::File>,
    mode: FileMode,
    position: u64,
}

#[derive(Debug, Clone)]
pub enum FileMode {
    Read,
    Write,
    Append,
    ReadWrite,
    WriteCreate,
    ReadWriteCreate,
}

impl OmegaFile {
    pub fn open(path: &str) -> OmegaResult<Self> {
        let file = fs::File::open(path).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(Self {
            path: PathBuf::from(path),
            handle: Some(file),
            mode: FileMode::Read,
            position: 0,
        })
    }

    pub fn create(path: &str) -> OmegaResult<Self> {
        let file = fs::File::create(path).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(Self {
            path: PathBuf::from(path),
            handle: Some(file),
            mode: FileMode::Write,
            position: 0,
        })
    }

    pub fn append(path: &str) -> OmegaResult<Self> {
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(Self {
            path: PathBuf::from(path),
            handle: Some(file),
            mode: FileMode::Append,
            position: 0,
        })
    }

    pub fn read_to_string(&mut self) -> OmegaResult<String> {
        let mut contents = String::new();
        if let Some(ref mut file) = self.handle {
            file.read_to_string(&mut contents).map_err(|e| OmegaError::IoError(e.to_string()))?;
        }
        Ok(contents)
    }

    pub fn read_bytes(&mut self, count: usize) -> OmegaResult<Vec<u8>> {
        let mut buffer = vec![0u8; count];
        if let Some(ref mut file) = self.handle {
            let bytes_read = file.read(&mut buffer).map_err(|e| OmegaError::IoError(e.to_string()))?;
            buffer.truncate(bytes_read);
        }
        Ok(buffer)
    }

    pub fn read_line(&mut self) -> OmegaResult<String> {
        let mut line = String::new();
        if let Some(ref mut file) = self.handle {
            let mut reader = BufReader::new(file);
            reader.read_line(&mut line).map_err(|e| OmegaError::IoError(e.to_string()))?;
        }
        Ok(line)
    }

    pub fn read_lines(&mut self) -> OmegaResult<Vec<String>> {
        let contents = self.read_to_string()?;
        Ok(contents.lines().map(String::from).collect())
    }

    pub fn write_string(&mut self, data: &str) -> OmegaResult<usize> {
        if let Some(ref mut file) = self.handle {
            let bytes_written = file.write(data.as_bytes()).map_err(|e| OmegaError::IoError(e.to_string()))?;
            Ok(bytes_written)
        } else {
            Err(OmegaError::IoError("File not open".to_string()))
        }
    }

    pub fn write_bytes(&mut self, data: &[u8]) -> OmegaResult<usize> {
        if let Some(ref mut file) = self.handle {
            let bytes_written = file.write(data).map_err(|e| OmegaError::IoError(e.to_string()))?;
            Ok(bytes_written)
        } else {
            Err(OmegaError::IoError("File not open".to_string()))
        }
    }

    pub fn write_line(&mut self, data: &str) -> OmegaResult<usize> {
        self.write_string(&format!("{}\n", data))
    }

    pub fn flush(&mut self) -> OmegaResult<()> {
        if let Some(ref mut file) = self.handle {
            file.flush().map_err(|e| OmegaError::IoError(e.to_string()))?;
        }
        Ok(())
    }

    pub fn seek(&mut self, position: u64) -> OmegaResult<()> {
        if let Some(ref mut file) = self.handle {
            file.seek(SeekFrom::Start(position)).map_err(|e| OmegaError::IoError(e.to_string()))?;
            self.position = position;
        }
        Ok(())
    }

    pub fn seek_from_end(&mut self, offset: i64) -> OmegaResult<()> {
        if let Some(ref mut file) = self.handle {
            file.seek(SeekFrom::End(offset)).map_err(|e| OmegaError::IoError(e.to_string()))?;
        }
        Ok(())
    }

    pub fn seek_from_current(&mut self, offset: i64) -> OmegaResult<()> {
        if let Some(ref mut file) = self.handle {
            file.seek(SeekFrom::Current(offset)).map_err(|e| OmegaError::IoError(e.to_string()))?;
        }
        Ok(())
    }

    pub fn metadata(&self) -> OmegaResult<FileMetadata> {
        let metadata = fs::metadata(&self.path).map_err(|e| OmegaError::IoError(e.to_string()))?;
        Ok(FileMetadata {
            size: metadata.len(),
            is_file: metadata.is_file(),
            is_dir: metadata.is_dir(),
            readonly: metadata.permissions().readonly(),
            modified: metadata.modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn close(&mut self) {
        self.handle = None;
    }
}

#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub size: u64,
    pub is_file: bool,
    pub is_dir: bool,
    pub readonly: bool,
    pub modified: u64,
}

// Filesystem operations
pub fn read_to_string(path: &str) -> OmegaResult<String> {
    fs::read_to_string(path).map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn read_bytes(path: &str) -> OmegaResult<Vec<u8>> {
    fs::read(path).map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn write_string(path: &str, data: &str) -> OmegaResult<()> {
    fs::write(path, data).map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn write_bytes(path: &str, data: &[u8]) -> OmegaResult<()> {
    fs::write(path, data).map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn create_dir(path: &str) -> OmegaResult<()> {
    fs::create_dir(path).map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn create_dir_all(path: &str) -> OmegaResult<()> {
    fs::create_dir_all(path).map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn remove_file(path: &str) -> OmegaResult<()> {
    fs::remove_file(path).map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn remove_dir(path: &str) -> OmegaResult<()> {
    fs::remove_dir(path).map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn remove_dir_all(path: &str) -> OmegaResult<()> {
    fs::remove_dir_all(path).map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn rename(from: &str, to: &str) -> OmegaResult<()> {
    fs::rename(from, to).map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn copy(from: &str, to: &str) -> OmegaResult<u64> {
    fs::copy(from, to).map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn exists(path: &str) -> bool {
    Path::new(path).exists()
}

pub fn is_file(path: &str) -> bool {
    Path::new(path).is_file()
}

pub fn is_dir(path: &str) -> bool {
    Path::new(path).is_dir()
}

pub fn metadata(path: &str) -> OmegaResult<FileMetadata> {
    let metadata = fs::metadata(path).map_err(|e| OmegaError::IoError(e.to_string()))?;
    Ok(FileMetadata {
        size: metadata.len(),
        is_file: metadata.is_file(),
        is_dir: metadata.is_dir(),
        readonly: metadata.permissions().readonly(),
        modified: metadata.modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0),
    })
}

pub fn read_dir(path: &str) -> OmegaResult<Vec<String>> {
    let entries = fs::read_dir(path).map_err(|e| OmegaError::IoError(e.to_string()))?;
    let mut result = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| OmegaError::IoError(e.to_string()))?;
        result.push(entry.path().to_string_lossy().to_string());
    }
    Ok(result)
}

pub fn read_dir_recursive(path: &str) -> OmegaResult<Vec<String>> {
    let mut result = Vec::new();
    read_dir_recursive_impl(Path::new(path), &mut result)?;
    Ok(result)
}

fn read_dir_recursive_impl(path: &Path, result: &mut Vec<String>) -> OmegaResult<()> {
    let entries = fs::read_dir(path).map_err(|e| OmegaError::IoError(e.to_string()))?;
    for entry in entries {
        let entry = entry.map_err(|e| OmegaError::IoError(e.to_string()))?;
        let path = entry.path();
        result.push(path.to_string_lossy().to_string());
        if path.is_dir() {
            read_dir_recursive_impl(&path, result)?;
        }
    }
    Ok(())
}

pub fn temp_dir() -> String {
    std::env::temp_dir().to_string_lossy().to_string()
}

pub fn current_dir() -> OmegaResult<String> {
    std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| OmegaError::IoError(e.to_string()))
}

pub fn set_current_dir(path: &str) -> OmegaResult<()> {
    std::env::set_current_dir(path).map_err(|e| OmegaError::IoError(e.to_string()))
}
