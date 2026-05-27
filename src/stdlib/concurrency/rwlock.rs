use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use crate::errors::{OmegaError, OmegaResult};

pub struct OmegaRwLock<T> {
    inner: RwLock<T>,
}

impl<T> OmegaRwLock<T> {
    pub fn new(data: T) -> Self {
        Self { inner: RwLock::new(data) }
    }

    pub fn read(&self) -> OmegaResult<RwLockReadGuard<T>> {
        self.inner.read().map_err(|e| OmegaError::LockError {
            message: format!("Read lock poisoned: {}", e),
        })
    }

    pub fn write(&self) -> OmegaResult<RwLockWriteGuard<T>> {
        self.inner.write().map_err(|e| OmegaError::LockError {
            message: format!("Write lock poisoned: {}", e),
        })
    }

    pub fn try_read(&self) -> OmegaResult<RwLockReadGuard<T>> {
        self.inner.try_read().map_err(|e| OmegaError::LockError {
            message: format!("Try read lock failed: {}", e),
        })
    }

    pub fn try_write(&self) -> OmegaResult<RwLockWriteGuard<T>> {
        self.inner.try_write().map_err(|e| OmegaError::LockError {
            message: format!("Try write lock failed: {}", e),
        })
    }

    pub fn is_poisoned(&self) -> bool {
        self.inner.is_poisoned()
    }

    pub fn into_inner(self) -> OmegaResult<T> {
        self.inner.into_inner().map_err(|e| OmegaError::LockError {
            message: format!("Into inner failed: {}", e),
        })
    }
}
