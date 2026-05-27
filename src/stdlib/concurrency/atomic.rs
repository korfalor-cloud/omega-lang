use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

pub struct OmegaAtomicBool {
    inner: AtomicBool,
}

impl OmegaAtomicBool {
    pub fn new(value: bool) -> Self {
        Self { inner: AtomicBool::new(value) }
    }

    pub fn load(&self) -> bool {
        self.inner.load(Ordering::SeqCst)
    }

    pub fn store(&self, value: bool) {
        self.inner.store(value, Ordering::SeqCst)
    }

    pub fn swap(&self, value: bool) -> bool {
        self.inner.swap(value, Ordering::SeqCst)
    }

    pub fn compare_and_swap(&self, current: bool, new: bool) -> bool {
        self.inner.compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap_or_else(|x| x)
    }

    pub fn fetch_and(&self, value: bool) -> bool {
        self.inner.fetch_and(value, Ordering::SeqCst)
    }

    pub fn fetch_or(&self, value: bool) -> bool {
        self.inner.fetch_or(value, Ordering::SeqCst)
    }

    pub fn fetch_xor(&self, value: bool) -> bool {
        self.inner.fetch_xor(value, Ordering::SeqCst)
    }

    pub fn fetch_nand(&self, value: bool) -> bool {
        self.inner.fetch_nand(value, Ordering::SeqCst)
    }

    pub fn not(&self) -> bool {
        self.inner.fetch_nand(true, Ordering::SeqCst)
    }
}

impl Clone for OmegaAtomicBool {
    fn clone(&self) -> Self {
        Self::new(self.load())
    }
}

pub struct OmegaAtomicI64 {
    inner: AtomicI64,
}

impl OmegaAtomicI64 {
    pub fn new(value: i64) -> Self {
        Self { inner: AtomicI64::new(value) }
    }

    pub fn load(&self) -> i64 {
        self.inner.load(Ordering::SeqCst)
    }

    pub fn store(&self, value: i64) {
        self.inner.store(value, Ordering::SeqCst)
    }

    pub fn swap(&self, value: i64) -> i64 {
        self.inner.swap(value, Ordering::SeqCst)
    }

    pub fn compare_and_swap(&self, current: i64, new: i64) -> i64 {
        self.inner.compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap_or_else(|x| x)
    }

    pub fn fetch_add(&self, value: i64) -> i64 {
        self.inner.fetch_add(value, Ordering::SeqCst)
    }

    pub fn fetch_sub(&self, value: i64) -> i64 {
        self.inner.fetch_sub(value, Ordering::SeqCst)
    }

    pub fn fetch_and(&self, value: i64) -> i64 {
        self.inner.fetch_and(value, Ordering::SeqCst)
    }

    pub fn fetch_or(&self, value: i64) -> i64 {
        self.inner.fetch_or(value, Ordering::SeqCst)
    }

    pub fn fetch_xor(&self, value: i64) -> i64 {
        self.inner.fetch_xor(value, Ordering::SeqCst)
    }

    pub fn fetch_max(&self, value: i64) -> i64 {
        self.inner.fetch_max(value, Ordering::SeqCst)
    }

    pub fn fetch_min(&self, value: i64) -> i64 {
        self.inner.fetch_min(value, Ordering::SeqCst)
    }

    pub fn increment(&self) -> i64 {
        self.fetch_add(1) + 1
    }

    pub fn decrement(&self) -> i64 {
        self.fetch_sub(1) - 1
    }
}

impl Clone for OmegaAtomicI64 {
    fn clone(&self) -> Self {
        Self::new(self.load())
    }
}

pub struct OmegaAtomicU64 {
    inner: AtomicU64,
}

impl OmegaAtomicU64 {
    pub fn new(value: u64) -> Self {
        Self { inner: AtomicU64::new(value) }
    }

    pub fn load(&self) -> u64 {
        self.inner.load(Ordering::SeqCst)
    }

    pub fn store(&self, value: u64) {
        self.inner.store(value, Ordering::SeqCst)
    }

    pub fn swap(&self, value: u64) -> u64 {
        self.inner.swap(value, Ordering::SeqCst)
    }

    pub fn compare_and_swap(&self, current: u64, new: u64) -> u64 {
        self.inner.compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap_or_else(|x| x)
    }

    pub fn fetch_add(&self, value: u64) -> u64 {
        self.inner.fetch_add(value, Ordering::SeqCst)
    }

    pub fn fetch_sub(&self, value: u64) -> u64 {
        self.inner.fetch_sub(value, Ordering::SeqCst)
    }

    pub fn fetch_and(&self, value: u64) -> u64 {
        self.inner.fetch_and(value, Ordering::SeqCst)
    }

    pub fn fetch_or(&self, value: u64) -> u64 {
        self.inner.fetch_or(value, Ordering::SeqCst)
    }

    pub fn fetch_xor(&self, value: u64) -> u64 {
        self.inner.fetch_xor(value, Ordering::SeqCst)
    }

    pub fn fetch_max(&self, value: u64) -> u64 {
        self.inner.fetch_max(value, Ordering::SeqCst)
    }

    pub fn fetch_min(&self, value: u64) -> u64 {
        self.inner.fetch_min(value, Ordering::SeqCst)
    }

    pub fn increment(&self) -> u64 {
        self.fetch_add(1) + 1
    }

    pub fn decrement(&self) -> u64 {
        self.fetch_sub(1) - 1
    }
}

impl Clone for OmegaAtomicU64 {
    fn clone(&self) -> Self {
        Self::new(self.load())
    }
}
