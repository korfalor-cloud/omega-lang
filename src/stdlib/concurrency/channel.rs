use std::sync::mpsc;
use std::time::Duration;
use crate::errors::{OmegaError, OmegaResult};

pub fn channel<T: Send + 'static>() -> (Sender<T>, Receiver<T>) {
    let (tx, rx) = mpsc::channel();
    (Sender { inner: tx }, Receiver { inner: rx })
}

pub fn sync_channel<T: Send + 'static>(bound: usize) -> (SyncSender<T>, Receiver<T>) {
    let (tx, rx) = mpsc::sync_channel(bound);
    (SyncSender { inner: tx }, Receiver { inner: rx })
}

pub struct Sender<T: Send> {
    inner: mpsc::Sender<T>,
}

impl<T: Send + 'static> Sender<T> {
    pub fn send(&self, value: T) -> OmegaResult<()> {
        self.inner.send(value).map_err(|_| OmegaError::ChannelError {
            message: "Send failed - channel closed".to_string(),
        })
    }

    pub fn try_send(&self, value: T) -> OmegaResult<()> {
        self.inner.send(value).map_err(|_| OmegaError::ChannelError {
            message: "Try send failed - channel closed".to_string(),
        })
    }
}

impl<T: Send + 'static> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

pub struct SyncSender<T: Send> {
    inner: mpsc::SyncSender<T>,
}

impl<T: Send + 'static> SyncSender<T> {
    pub fn send(&self, value: T) -> OmegaResult<()> {
        self.inner.send(value).map_err(|_| OmegaError::ChannelError {
            message: "Send failed - channel closed".to_string(),
        })
    }

    pub fn try_send(&self, value: T) -> OmegaResult<()> {
        self.inner.try_send(value).map_err(|e| OmegaError::ChannelError {
            message: format!("Try send failed: {}", e),
        })
    }
}

pub struct Receiver<T: Send> {
    inner: mpsc::Receiver<T>,
}

impl<T: Send + 'static> Receiver<T> {
    pub fn recv(&self) -> OmegaResult<T> {
        self.inner.recv().map_err(|_| OmegaError::ChannelError {
            message: "Receive failed - channel closed".to_string(),
        })
    }

    pub fn try_recv(&self) -> OmegaResult<T> {
        self.inner.try_recv().map_err(|e| OmegaError::ChannelError {
            message: format!("Try receive failed: {}", e),
        })
    }

    pub fn recv_timeout(&self, timeout_ms: u64) -> OmegaResult<T> {
        self.inner.recv_timeout(Duration::from_millis(timeout_ms)).map_err(|e| OmegaError::ChannelError {
            message: format!("Receive timeout: {}", e),
        })
    }

    pub fn iter(&self) -> mpsc::Iter<T> {
        self.inner.iter()
    }

    pub fn try_iter(&self) -> mpsc::TryIter<T> {
        self.inner.try_iter()
    }
}
