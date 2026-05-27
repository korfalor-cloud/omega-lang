use std::sync::{Mutex, MutexGuard};
use crate::errors::{OmegaError, OmegaResult};

pub struct OmegaMutex<T> {
    inner: Mutex<T>,
}

impl<T> OmegaMutex<T> {
    pub fn new(data: T) -> Self {
        Self { inner: Mutex::new(data) }
    }

    pub fn lock(&self) -> OmegaResult<MutexGuard<T>> {
        self.inner.lock().map_err(|e| OmegaError::LockError {
            message: format!("Lock poisoned: {}", e),
        })
    }

    pub fn try_lock(&self) -> OmegaResult<MutexGuard<T>> {
        self.inner.try_lock().map_err(|e| OmegaError::LockError {
            message: format!("Try lock failed: {}", e),
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

impl<T: Clone> Clone for OmegaMutex<T> {
    fn clone(&self) -> Self {
        Self::new(self.lock().unwrap().clone())
    }
}
