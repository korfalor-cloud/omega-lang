use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::thread;
use std::time::{Duration, Instant};
use crate::errors::{OmegaError, OmegaResult};

// ---------------------------------------------------------------------------
// Thread Pool
// ---------------------------------------------------------------------------

struct Task(Box<dyn FnOnce() + Send + 'static>);

pub struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    tx: Option<std::sync::mpsc::Sender<Task>>,
    size: usize,
}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "ThreadPool size must be > 0");
        let (tx, rx) = std::sync::mpsc::channel::<Task>();
        let rx = Arc::new(Mutex::new(rx));
        let mut workers = Vec::with_capacity(size);
        for id in 0..size {
            let rx = Arc::clone(&rx);
            workers.push(thread::spawn(move || {
                loop {
                    let task = rx.lock().unwrap().recv();
                    match task {
                        Ok(task) => (task.0)(),
                        Err(_) => break,
                    }
                }
                let _ = id;
            }));
        }
        Self { workers, tx: Some(tx), size }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn execute<F: FnOnce() + Send + 'static>(&self, f: F) -> OmegaResult<()> {
        self.tx
            .as_ref()
            .ok_or_else(|| OmegaError::ThreadError {
                message: "ThreadPool is shut down".to_string(),
            })?
            .send(Task(Box::new(f)))
            .map_err(|_| OmegaError::ThreadError {
                message: "Failed to enqueue task".to_string(),
            })
    }

    pub fn shutdown(mut self) {
        drop(self.tx.take());
        for w in self.workers {
            let _ = w.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Work-Stealing Deque (single-producer / multi-consumer)
// ---------------------------------------------------------------------------

pub struct WorkStealer<T: Send> {
    deque: Arc<Mutex<VecDeque<T>>>,
}

impl<T: Send> WorkStealer<T> {
    pub fn new() -> Self {
        Self { deque: Arc::new(Mutex::new(VecDeque::new())) }
    }

    /// Push a task onto the local end (producer side).
    pub fn push(&self, item: T) {
        self.deque.lock().unwrap().push_back(item);
    }

    /// Pop from the local end (LIFO for the owning worker).
    pub fn pop_local(&self) -> Option<T> {
        self.deque.lock().unwrap().pop_back()
    }

    /// Steal from the remote end (FIFO) -- used by other workers.
    pub fn steal(&self) -> Option<T> {
        self.deque.lock().unwrap().pop_front()
    }

    pub fn len(&self) -> usize {
        self.deque.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.deque.lock().unwrap().is_empty()
    }
}

impl<T: Send> Clone for WorkStealer<T> {
    fn clone(&self) -> Self {
        Self { deque: Arc::clone(&self.deque) }
    }
}

/// A simple work-stealing thread pool.
pub struct WorkStealingPool {
    stealers: Vec<WorkStealer<Box<dyn FnOnce() + Send>>>,
    workers: Vec<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
}

impl WorkStealingPool {
    pub fn new(num_workers: usize) -> Self {
        assert!(num_workers > 0);
        let shutdown = Arc::new(AtomicBool::new(false));
        let stealers: Vec<_> = (0..num_workers).map(|_| WorkStealer::new()).collect();
        let mut workers = Vec::with_capacity(num_workers);

        for id in 0..num_workers {
            let stealers = stealers.clone();
            let shutdown = Arc::clone(&shutdown);
            workers.push(thread::spawn(move || {
                loop {
                    // Try local deque first (LIFO).
                    if let Some(task) = stealers[id].pop_local() {
                        task();
                        continue;
                    }
                    // Try stealing from peers (FIFO).
                    let mut found = false;
                    for (j, s) in stealers.iter().enumerate() {
                        if j == id { continue; }
                        if let Some(task) = s.steal() {
                            task();
                            found = true;
                            break;
                        }
                    }
                    if found { continue; }
                    if shutdown.load(Ordering::Relaxed) && stealers.iter().all(|s| s.is_empty()) {
                        break;
                    }
                    thread::yield_now();
                }
            }));
        }
        Self { stealers, workers, shutdown }
    }

    pub fn submit<F: FnOnce() + Send + 'static>(&self, worker_id: usize, f: F) {
        self.stealers[worker_id % self.stealers.len()].push(Box::new(f));
    }

    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::Relaxed);
        for w in self.workers {
            let _ = w.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Lock-Free Stack (Treiber stack)
// ---------------------------------------------------------------------------

struct Node<T> {
    data: T,
    next: *mut Node<T>,
}

pub struct LockFreeStack<T> {
    head: AtomicPtr<Node<T>>,
    len: AtomicUsize,
}

unsafe impl<T: Send> Send for LockFreeStack<T> {}
unsafe impl<T: Send> Sync for LockFreeStack<T> {}

impl<T> LockFreeStack<T> {
    pub fn new() -> Self {
        Self { head: AtomicPtr::new(std::ptr::null_mut()), len: AtomicUsize::new(0) }
    }

    pub fn push(&self, data: T) {
        let node = Box::into_raw(Box::new(Node { data, next: std::ptr::null_mut() }));
        loop {
            let head = self.head.load(Ordering::SeqCst);
            unsafe { (*node).next = head; }
            if self.head.compare_exchange(
                head, node, Ordering::SeqCst, Ordering::SeqCst,
            ).is_ok() {
                self.len.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
    }

    pub fn pop(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::SeqCst);
            if head.is_null() {
                return None;
            }
            let next = unsafe { (*head).next };
            if self.head.compare_exchange(
                head, next, Ordering::SeqCst, Ordering::SeqCst,
            ).is_ok() {
                self.len.fetch_sub(1, Ordering::Relaxed);
                let node = unsafe { Box::from_raw(head) };
                return Some(node.data);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

// ---------------------------------------------------------------------------
// Lock-Free Counter
// ---------------------------------------------------------------------------

pub struct AtomicCounter {
    value: AtomicU64,
}

impl AtomicCounter {
    pub fn new(initial: u64) -> Self {
        Self { value: AtomicU64::new(initial) }
    }

    pub fn increment(&self) -> u64 {
        self.value.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn decrement(&self) -> u64 {
        self.value.fetch_sub(1, Ordering::Relaxed).saturating_sub(1)
    }

    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Simple Async Runtime (cooperative, poll-based)
// ---------------------------------------------------------------------------

type BoxFuture = Pin<Box<dyn std::future::Future<Output = ()> + Send>>;

fn noop_clone(_: *const ()) -> RawWaker { RawWaker::new(std::ptr::null(), &VTABLE) }
fn noop(_: *const ()) {}
static VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);

fn noop_waker() -> Waker {
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

pub struct MiniRuntime {
    tasks: VecDeque<BoxFuture>,
    completed: usize,
}

impl MiniRuntime {
    pub fn new() -> Self {
        Self { tasks: VecDeque::new(), completed: 0 }
    }

    pub fn spawn<F>(&mut self, future: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        self.tasks.push_back(Box::pin(future));
    }

    /// Run all spawned futures to completion (busy-poll).
    pub fn block_on_all(&mut self) {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        loop {
            let mut pending = VecDeque::new();
            let mut made_progress = false;
            while let Some(mut fut) = self.tasks.pop_front() {
                match fut.as_mut().poll(&mut cx) {
                    Poll::Ready(()) => {
                        self.completed += 1;
                        made_progress = true;
                    }
                    Poll::Pending => pending.push_back(fut),
                }
            }
            self.tasks = pending;
            if self.tasks.is_empty() {
                break;
            }
            if !made_progress {
                // All pending and no progress -- break to avoid infinite loop.
                break;
            }
        }
    }

    pub fn completed_count(&self) -> usize {
        self.completed
    }

    pub fn pending_count(&self) -> usize {
        self.tasks.len()
    }
}

// ---------------------------------------------------------------------------
// MPSC Channel (bounded, blocking)
// ---------------------------------------------------------------------------

pub struct OmegaMpscSender<T: Send> {
    inner: Arc<MpscInner<T>>,
}

impl<T: Send> Clone for OmegaMpscSender<T> {
    fn clone(&self) -> Self {
        self.inner.sender_count.fetch_add(1, Ordering::SeqCst);
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<T: Send> Drop for OmegaMpscSender<T> {
    fn drop(&mut self) {
        if self.inner.sender_count.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.inner.cond.notify_all();
        }
    }
}

pub struct OmegaMpscReceiver<T: Send> {
    inner: Arc<MpscInner<T>>,
}

struct MpscInner<T: Send> {
    queue: Mutex<VecDeque<T>>,
    cond: Condvar,
    sender_count: AtomicUsize,
    capacity: usize,
}

impl<T: Send> OmegaMpscSender<T> {
    pub fn send(&self, value: T) -> OmegaResult<()> {
        let mut q = self.inner.queue.lock().map_err(|e| OmegaError::ChannelError {
            message: format!("Lock poisoned: {}", e),
        })?;
        while self.inner.capacity > 0 && q.len() >= self.inner.capacity {
            q = self.inner.cond.wait(q).map_err(|e| OmegaError::ChannelError {
                message: format!("Wait failed: {}", e),
            })?;
        }
        q.push_back(value);
        self.inner.cond.notify_one();
        Ok(())
    }
}

impl<T: Send> OmegaMpscReceiver<T> {
    pub fn recv(&self) -> OmegaResult<T> {
        let mut q = self.inner.queue.lock().map_err(|e| OmegaError::ChannelError {
            message: format!("Lock poisoned: {}", e),
        })?;
        loop {
            if let Some(val) = q.pop_front() {
                self.inner.cond.notify_one();
                return Ok(val);
            }
            if self.inner.sender_count.load(Ordering::SeqCst) == 0 {
                return Err(OmegaError::ChannelError {
                    message: "Channel closed".to_string(),
                });
            }
            q = self.inner.cond.wait(q).map_err(|e| OmegaError::ChannelError {
                message: format!("Wait failed: {}", e),
            })?;
        }
    }

    pub fn try_recv(&self) -> OmegaResult<T> {
        let mut q = self.inner.queue.lock().map_err(|e| OmegaError::ChannelError {
            message: format!("Lock poisoned: {}", e),
        })?;
        q.pop_front().ok_or_else(|| OmegaError::ChannelError {
            message: "Channel empty".to_string(),
        })
    }
}

pub fn omega_mpsc_channel<T: Send>(capacity: usize) -> (OmegaMpscSender<T>, OmegaMpscReceiver<T>) {
    let inner = Arc::new(MpscInner {
        queue: Mutex::new(VecDeque::new()),
        cond: Condvar::new(),
        sender_count: AtomicUsize::new(1),
        capacity,
    });
    (OmegaMpscSender { inner: Arc::clone(&inner) }, OmegaMpscReceiver { inner })
}

// ---------------------------------------------------------------------------
// Broadcast Channel (fan-out to all receivers)
// ---------------------------------------------------------------------------

struct BroadcastInner<T: Send + Clone> {
    queue: Mutex<VecDeque<T>>,
    cond: Condvar,
    capacity: usize,
    version: AtomicU64,
}

pub struct BroadcastSender<T: Send + Clone> {
    inner: Arc<BroadcastInner<T>>,
}

impl<T: Send + Clone> Clone for BroadcastSender<T> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

pub struct BroadcastReceiver<T: Send + Clone> {
    inner: Arc<BroadcastInner<T>>,
    position: Mutex<usize>,
}

impl<T: Send + Clone> Clone for BroadcastReceiver<T> {
    fn clone(&self) -> Self {
        let pos = *self.position.lock().unwrap();
        Self { inner: Arc::clone(&self.inner), position: Mutex::new(pos) }
    }
}

impl<T: Send + Clone> BroadcastSender<T> {
    pub fn send(&self, value: T) -> OmegaResult<()> {
        let mut q = self.inner.queue.lock().map_err(|e| OmegaError::ChannelError {
            message: format!("Lock poisoned: {}", e),
        })?;
        if self.inner.capacity > 0 && q.len() >= self.inner.capacity {
            q.pop_front();
        }
        q.push_back(value);
        self.inner.version.fetch_add(1, Ordering::SeqCst);
        self.inner.cond.notify_all();
        Ok(())
    }
}

impl<T: Send + Clone> BroadcastReceiver<T> {
    pub fn recv(&self) -> OmegaResult<T> {
        loop {
            let mut q = self.inner.queue.lock().map_err(|e| OmegaError::ChannelError {
                message: format!("Lock poisoned: {}", e),
            })?;
            let mut pos = self.position.lock().map_err(|e| OmegaError::ChannelError {
                message: format!("Lock poisoned: {}", e),
            })?;
            if *pos < q.len() {
                let val = q[*pos].clone();
                *pos += 1;
                return Ok(val);
            }
            drop(pos);
            q = self.inner.cond.wait(q).map_err(|e| OmegaError::ChannelError {
                message: format!("Wait failed: {}", e),
            })?;
            drop(q);
        }
    }
}

pub fn broadcast_channel<T: Send + Clone>(capacity: usize) -> (BroadcastSender<T>, BroadcastReceiver<T>) {
    let inner = Arc::new(BroadcastInner {
        queue: Mutex::new(VecDeque::new()),
        cond: Condvar::new(),
        capacity,
        version: AtomicU64::new(0),
    });
    (BroadcastSender { inner: Arc::clone(&inner) }, BroadcastReceiver { inner, position: Mutex::new(0) })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    // -- ThreadPool ---------------------------------------------------------

    #[test]
    fn test_thread_pool_basic() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        for _ in 0..20 {
            let c = Arc::clone(&counter);
            pool.execute(move || { c.fetch_add(1, Ordering::SeqCst); }).unwrap();
        }
        pool.shutdown();
        assert_eq!(counter.load(Ordering::SeqCst), 20);
    }

    #[test]
    fn test_thread_pool_size() {
        let pool = ThreadPool::new(2);
        assert_eq!(pool.size(), 2);
        pool.shutdown();
    }

    // -- WorkStealer --------------------------------------------------------

    #[test]
    fn test_work_stealer_push_pop() {
        let ws = WorkStealer::new();
        ws.push(10);
        ws.push(20);
        assert_eq!(ws.pop_local(), Some(20)); // LIFO
        assert_eq!(ws.steal(), Some(10));     // FIFO from other end
        assert!(ws.is_empty());
    }

    #[test]
    fn test_work_stealing_pool() {
        let pool = WorkStealingPool::new(2);
        let counter = Arc::new(AtomicUsize::new(0));
        for i in 0..10 {
            let c = Arc::clone(&counter);
            pool.submit(i % 2, move || { c.fetch_add(1, Ordering::SeqCst); });
        }
        pool.shutdown();
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    // -- LockFreeStack ------------------------------------------------------

    #[test]
    fn test_lock_free_stack_push_pop() {
        let stack = LockFreeStack::new();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        assert_eq!(stack.len(), 3);
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn test_lock_free_stack_concurrent() {
        let stack = Arc::new(LockFreeStack::new());
        let num_threads = 4;
        let ops_per_thread = 1000;
        let mut handles = Vec::new();

        for _ in 0..num_threads {
            let s = Arc::clone(&stack);
            handles.push(thread::spawn(move || {
                for i in 0..ops_per_thread {
                    s.push(i);
                    let _ = s.pop();
                }
            }));
        }
        for h in handles { h.join().unwrap(); }
        // Stack may or may not be empty depending on interleaving, just verify no crash.
        let _ = stack.len();
    }

    // -- AtomicCounter ------------------------------------------------------

    #[test]
    fn test_atomic_counter() {
        let counter = AtomicCounter::new(0);
        assert_eq!(counter.get(), 0);
        assert_eq!(counter.increment(), 1);
        assert_eq!(counter.increment(), 2);
        assert_eq!(counter.decrement(), 1);
        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_atomic_counter_concurrent() {
        let counter = Arc::new(AtomicCounter::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let c = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 { c.increment(); }
            }));
        }
        for h in handles { h.join().unwrap(); }
        assert_eq!(counter.get(), 8000);
    }

    // -- MiniRuntime --------------------------------------------------------

    #[test]
    fn test_mini_runtime_ready_immediately() {
        let mut rt = MiniRuntime::new();
        rt.spawn(async { /* immediately ready */ });
        rt.block_on_all();
        assert_eq!(rt.completed_count(), 1);
        assert_eq!(rt.pending_count(), 0);
    }

    // -- OmegaMpscChannel ---------------------------------------------------

    #[test]
    fn test_omega_mpsc_send_recv() {
        let (tx, rx) = omega_mpsc_channel::<i32>(16);
        tx.send(42).unwrap();
        tx.send(43).unwrap();
        assert_eq!(rx.recv().unwrap(), 42);
        assert_eq!(rx.recv().unwrap(), 43);
    }

    #[test]
    fn test_omega_mpsc_try_recv_empty() {
        let (_tx, rx) = omega_mpsc_channel::<i32>(4);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_omega_mpsc_multi_producer() {
        let (tx, rx) = omega_mpsc_channel::<i32>(100);
        let mut handles = Vec::new();
        for i in 0..4 {
            let tx = tx.clone();
            handles.push(thread::spawn(move || {
                tx.send(i).unwrap();
            }));
        }
        for h in handles { h.join().unwrap(); }
        let mut values = Vec::new();
        for _ in 0..4 {
            values.push(rx.recv().unwrap());
        }
        values.sort();
        assert_eq!(values, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_omega_mpsc_close_drains() {
        let (tx, rx) = omega_mpsc_channel::<i32>(16);
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        drop(tx);
        assert_eq!(rx.recv().unwrap(), 1);
        assert_eq!(rx.recv().unwrap(), 2);
        assert!(rx.recv().is_err());
    }

    // -- BroadcastChannel ---------------------------------------------------

    #[test]
    fn test_broadcast_basic() {
        let (tx, rx1) = broadcast_channel::<String>(64);
        let rx2 = rx1.clone();
        tx.send("hello".to_string()).unwrap();
        assert_eq!(rx1.recv().unwrap(), "hello");
        assert_eq!(rx2.recv().unwrap(), "hello");
    }

    #[test]
    fn test_broadcast_independent_positions() {
        let (tx, rx1) = broadcast_channel::<i32>(32);
        let rx2 = rx1.clone();
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        assert_eq!(rx1.recv().unwrap(), 1);
        assert_eq!(rx2.recv().unwrap(), 1);
        assert_eq!(rx2.recv().unwrap(), 2);
    }
}
