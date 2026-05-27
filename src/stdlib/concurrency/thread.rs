use std::thread;
use std::time::Duration;
use crate::errors::{OmegaError, OmegaResult};

pub fn spawn(f: impl FnOnce() + Send + 'static) -> OmegaResult<JoinHandle> {
    let handle = thread::spawn(f);
    Ok(JoinHandle { handle: Some(handle) })
}

pub fn sleep_ms(ms: u64) {
    thread::sleep(Duration::from_millis(ms));
}

pub fn sleep_secs(secs: f64) {
    thread::sleep(Duration::from_secs_f64(secs));
}

pub fn yield_now() {
    thread::yield_now();
}

pub fn current_thread_id() -> String {
    format!("{:?}", thread::current().id())
}

pub fn available_parallelism() -> usize {
    thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

pub struct JoinHandle {
    handle: Option<thread::JoinHandle<()>>,
}

impl JoinHandle {
    pub fn join(&mut self) -> OmegaResult<()> {
        if let Some(handle) = self.handle.take() {
            handle.join().map_err(|_| OmegaError::ThreadError {
                message: "Thread panicked".to_string(),
            })?;
        }
        Ok(())
    }

    pub fn is_finished(&self) -> bool {
        self.handle.as_ref().map_or(true, |h| h.is_finished())
    }
}

pub struct Builder {
    name: Option<String>,
    stack_size: Option<usize>,
}

impl Builder {
    pub fn new() -> Self {
        Self { name: None, stack_size: None }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub fn stack_size(mut self, size: usize) -> Self {
        self.stack_size = Some(size);
        self
    }

    pub fn spawn(self, f: impl FnOnce() + Send + 'static) -> OmegaResult<JoinHandle> {
        let mut builder = thread::Builder::new();
        if let Some(name) = self.name {
            builder = builder.name(name);
        }
        if let Some(size) = self.stack_size {
            builder = builder.stack_size(size);
        }
        let handle = builder.spawn(f).map_err(|e| OmegaError::ThreadError {
            message: e.to_string(),
        })?;
        Ok(JoinHandle { handle: Some(handle) })
    }
}
