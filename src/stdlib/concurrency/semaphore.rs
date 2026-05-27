use std::sync::{Arc, Condvar, Mutex};
use crate::errors::{OmegaError, OmegaResult};

pub struct OmegaSemaphore {
    state: Arc<(Mutex<usize>, Condvar)>,
}

impl OmegaSemaphore {
    pub fn new(permits: usize) -> Self {
        Self {
            state: Arc::new((Mutex::new(permits), Condvar::new())),
        }
    }

    pub fn acquire(&self) -> OmegaResult<SemaphoreGuard> {
        let (lock, cvar) = &*self.state;
        let mut permits = lock.lock().map_err(|e| OmegaError::LockError {
            message: e.to_string(),
        })?;
        while *permits == 0 {
            permits = cvar.wait(permits).map_err(|e| OmegaError::LockError {
                message: e.to_string(),
            })?;
        }
        *permits -= 1;
        Ok(SemaphoreGuard { state: self.state.clone() })
    }

    pub fn try_acquire(&self) -> OmegaResult<Option<SemaphoreGuard>> {
        let (lock, _) = &*self.state;
        let mut permits = lock.lock().map_err(|e| OmegaError::LockError {
            message: e.to_string(),
        })?;
        if *permits > 0 {
            *permits -= 1;
            Ok(Some(SemaphoreGuard { state: self.state.clone() }))
        } else {
            Ok(None)
        }
    }

    pub fn available_permits(&self) -> usize {
        let (lock, _) = &*self.state;
        *lock.lock().unwrap()
    }

    pub fn add_permits(&self, n: usize) {
        let (lock, cvar) = &*self.state;
        let mut permits = lock.lock().unwrap();
        *permits += n;
        cvar.notify_all();
    }
}

impl Clone for OmegaSemaphore {
    fn clone(&self) -> Self {
        Self { state: self.state.clone() }
    }
}

pub struct SemaphoreGuard {
    state: Arc<(Mutex<usize>, Condvar)>,
}

impl Drop for SemaphoreGuard {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.state;
        let mut permits = lock.lock().unwrap();
        *permits += 1;
        cvar.notify_one();
    }
}
