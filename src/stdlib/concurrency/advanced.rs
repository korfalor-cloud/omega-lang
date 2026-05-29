use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::thread;
use crate::errors::{OmegaError, OmegaResult};
// ========================= Thread Pool =====================================
struct Task(Box<dyn FnOnce() + Send + 'static>);
pub struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    tx: Option<mpsc::Sender<Task>>,
    size: usize,
}
impl ThreadPool {
    pub fn new(size: usize) -> Self {
        assert!(size > 0);
        let (tx, rx) = mpsc::channel::<Task>();
        let rx = Arc::new(Mutex::new(rx));
        let workers = (0..size).map(|_| {
            let rx = Arc::clone(&rx);
            thread::spawn(move || { while let Ok(task) = rx.lock().unwrap().recv() { (task.0)(); } })
        }).collect();
        Self { workers, tx: Some(tx), size }
    }
    pub fn size(&self) -> usize { self.size }
    pub fn execute<F: FnOnce() + Send + 'static>(&self, f: F) -> OmegaResult<()> {
        self.tx.as_ref().ok_or_else(|| OmegaError::ThreadError { message: "Pool shut down".into() })?
            .send(Task(Box::new(f))).map_err(|_| OmegaError::ThreadError { message: "Enqueue failed".into() })
    }
    pub fn shutdown(mut self) {
        drop(self.tx.take());
        for w in self.workers { let _ = w.join(); }
    }
}
// ========================= Work Stealing ===================================
pub struct WorkStealer<T: Send> { deque: Arc<Mutex<VecDeque<T>>> }
impl<T: Send> WorkStealer<T> {
    pub fn new() -> Self { Self { deque: Arc::new(Mutex::new(VecDeque::new())) } }
    pub fn push(&self, item: T) { self.deque.lock().unwrap().push_back(item); }
    pub fn pop_local(&self) -> Option<T> { self.deque.lock().unwrap().pop_back() }
    pub fn steal(&self) -> Option<T> { self.deque.lock().unwrap().pop_front() }
    pub fn is_empty(&self) -> bool { self.deque.lock().unwrap().is_empty() }
}
impl<T: Send> Clone for WorkStealer<T> {
    fn clone(&self) -> Self { Self { deque: Arc::clone(&self.deque) } }
}
pub struct WorkStealingPool {
    stealers: Vec<WorkStealer<Box<dyn FnOnce() + Send>>>,
    workers: Vec<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
}
impl WorkStealingPool {
    pub fn new(n: usize) -> Self {
        let flag = Arc::new(AtomicBool::new(false));
        let stealers: Vec<_> = (0..n).map(|_| WorkStealer::new()).collect();
        let workers = (0..n).map(|id| {
            let stealers = stealers.clone(); let flag = Arc::clone(&flag);
            thread::spawn(move || loop {
                if let Some(t) = stealers[id].pop_local() { t(); continue; }
                let mut stole = false;
                for (j, s) in stealers.iter().enumerate() {
                    if j != id { if let Some(t) = s.steal() { t(); stole = true; break; } }
                }
                if stole { continue; }
                if flag.load(Ordering::Relaxed) && stealers.iter().all(|s| s.is_empty()) { break; }
                thread::yield_now();
            })
        }).collect();
        Self { stealers, workers, shutdown: flag }
    }
    pub fn submit<F: FnOnce() + Send + 'static>(&self, wid: usize, f: F) {
        self.stealers[wid % self.stealers.len()].push(Box::new(f));
    }
    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::Relaxed);
        for w in self.workers { let _ = w.join(); }
    }
}
// ========================= Lock-Free Stack (Treiber) =======================
struct Node<T> { data: T, next: *mut Node<T> }
pub struct LockFreeStack<T> { head: AtomicPtr<Node<T>>, len: AtomicUsize }
unsafe impl<T: Send> Send for LockFreeStack<T> {}
unsafe impl<T: Send> Sync for LockFreeStack<T> {}
impl<T> LockFreeStack<T> {
    pub fn new() -> Self { Self { head: AtomicPtr::new(std::ptr::null_mut()), len: AtomicUsize::new(0) } }
    pub fn push(&self, data: T) {
        let node = Box::into_raw(Box::new(Node { data, next: std::ptr::null_mut() }));
        loop {
            let head = self.head.load(Ordering::SeqCst);
            unsafe { (*node).next = head; }
            if self.head.compare_exchange(head, node, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                self.len.fetch_add(1, Ordering::Relaxed); return;
            }
        }
    }
    pub fn pop(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::SeqCst);
            if head.is_null() { return None; }
            let next = unsafe { (*head).next };
            if self.head.compare_exchange(head, next, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                self.len.fetch_sub(1, Ordering::Relaxed);
                return Some(unsafe { Box::from_raw(head) }.data);
            }
        }
    }
    pub fn len(&self) -> usize { self.len.load(Ordering::Relaxed) }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}
impl<T> Drop for LockFreeStack<T> { fn drop(&mut self) { while self.pop().is_some() {} } }
// ========================= Lock-Free Counter ===============================
pub struct AtomicCounter { value: AtomicU64 }
impl AtomicCounter {
    pub fn new(init: u64) -> Self { Self { value: AtomicU64::new(init) } }
    pub fn increment(&self) -> u64 { self.value.fetch_add(1, Ordering::Relaxed) + 1 }
    pub fn decrement(&self) -> u64 { self.value.fetch_sub(1, Ordering::Relaxed).saturating_sub(1) }
    pub fn get(&self) -> u64 { self.value.load(Ordering::Relaxed) }
    pub fn reset(&self) { self.value.store(0, Ordering::Relaxed); }
}
// ========================= Mini Async Runtime ==============================
type BoxFuture = Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
fn noop_waker() -> Waker {
    fn clone(_: *const ()) -> RawWaker { RawWaker::new(std::ptr::null(), &VTABLE) }
    fn noop(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}
pub struct MiniRuntime { tasks: VecDeque<BoxFuture>, completed: usize }
impl MiniRuntime {
    pub fn new() -> Self { Self { tasks: VecDeque::new(), completed: 0 } }
    pub fn spawn<F: std::future::Future<Output = ()> + Send + 'static>(&mut self, f: F) {
        self.tasks.push_back(Box::pin(f));
    }
    pub fn block_on_all(&mut self) {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        loop {
            let mut pending = VecDeque::new();
            let mut progress = false;
            while let Some(mut fut) = self.tasks.pop_front() {
                match fut.as_mut().poll(&mut cx) {
                    Poll::Ready(()) => { self.completed += 1; progress = true; }
                    Poll::Pending => pending.push_back(fut),
                }
            }
            self.tasks = pending;
            if self.tasks.is_empty() || !progress { break; }
        }
    }
    pub fn completed_count(&self) -> usize { self.completed }
    pub fn pending_count(&self) -> usize { self.tasks.len() }
}
// ========================= MPSC Channel (bounded) ==========================
struct MpscInner<T: Send> { queue: Mutex<VecDeque<T>>, cond: Condvar, senders: AtomicUsize, cap: usize }
pub struct MpscSender<T: Send> { inner: Arc<MpscInner<T>> }
pub struct MpscReceiver<T: Send> { inner: Arc<MpscInner<T>> }
impl<T: Send> Clone for MpscSender<T> {
    fn clone(&self) -> Self {
        self.inner.senders.fetch_add(1, Ordering::SeqCst);
        Self { inner: Arc::clone(&self.inner) }
    }
}
impl<T: Send> Drop for MpscSender<T> {
    fn drop(&mut self) {
        if self.inner.senders.fetch_sub(1, Ordering::SeqCst) == 1 { self.inner.cond.notify_all(); }
    }
}
impl<T: Send> MpscSender<T> {
    pub fn send(&self, val: T) -> OmegaResult<()> {
        let mut q = self.inner.queue.lock().unwrap();
        while self.inner.cap > 0 && q.len() >= self.inner.cap {
            q = self.inner.cond.wait(q).unwrap();
        }
        q.push_back(val);
        self.inner.cond.notify_one();
        Ok(())
    }
}
impl<T: Send> MpscReceiver<T> {
    pub fn recv(&self) -> OmegaResult<T> {
        let mut q = self.inner.queue.lock().unwrap();
        loop {
            if let Some(v) = q.pop_front() { self.inner.cond.notify_one(); return Ok(v); }
            if self.inner.senders.load(Ordering::SeqCst) == 0 {
                return Err(OmegaError::ChannelError { message: "Closed".into() });
            }
            q = self.inner.cond.wait(q).unwrap();
        }
    }
    pub fn try_recv(&self) -> OmegaResult<T> {
        self.inner.queue.lock().unwrap().pop_front()
            .ok_or_else(|| OmegaError::ChannelError { message: "Empty".into() })
    }
}
pub fn mpsc_channel<T: Send>(cap: usize) -> (MpscSender<T>, MpscReceiver<T>) {
    let inner = Arc::new(MpscInner {
        queue: Mutex::new(VecDeque::new()), cond: Condvar::new(),
        senders: AtomicUsize::new(1), cap,
    });
    (MpscSender { inner: Arc::clone(&inner) }, MpscReceiver { inner })
}
// ========================= Broadcast Channel ===============================
struct BcastInner<T: Send + Clone> { queue: Mutex<VecDeque<T>>, cond: Condvar, cap: usize }
pub struct BcastSender<T: Send + Clone> { inner: Arc<BcastInner<T>> }
pub struct BcastReceiver<T: Send + Clone> { inner: Arc<BcastInner<T>>, pos: Mutex<usize> }
impl<T: Send + Clone> Clone for BcastSender<T> {
    fn clone(&self) -> Self { Self { inner: Arc::clone(&self.inner) } }
}
impl<T: Send + Clone> Clone for BcastReceiver<T> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner), pos: Mutex::new(*self.pos.lock().unwrap()) }
    }
}
impl<T: Send + Clone> BcastSender<T> {
    pub fn send(&self, val: T) -> OmegaResult<()> {
        let mut q = self.inner.queue.lock().unwrap();
        if self.inner.cap > 0 && q.len() >= self.inner.cap { q.pop_front(); }
        q.push_back(val);
        self.inner.cond.notify_all();
        Ok(())
    }
}
impl<T: Send + Clone> BcastReceiver<T> {
    pub fn recv(&self) -> OmegaResult<T> {
        loop {
            let q = self.inner.queue.lock().unwrap();
            let mut p = self.pos.lock().unwrap();
            if *p < q.len() { let v = q[*p].clone(); *p += 1; return Ok(v); }
            drop(p); drop(q);
            let mut q = self.inner.queue.lock().unwrap();
            q = self.inner.cond.wait(q).unwrap();
            drop(q);
        }
    }
}
pub fn broadcast_channel<T: Send + Clone>(cap: usize) -> (BcastSender<T>, BcastReceiver<T>) {
    let inner = Arc::new(BcastInner { queue: Mutex::new(VecDeque::new()), cond: Condvar::new(), cap });
    (BcastSender { inner: Arc::clone(&inner) }, BcastReceiver { inner, pos: Mutex::new(0) })
}
// ===========================================================================
// Tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    #[test]
    fn thread_pool_executes_tasks() {
        let pool = ThreadPool::new(4);
        let n = Arc::new(AtomicUsize::new(0));
        for _ in 0..20 { let c = Arc::clone(&n); pool.execute(move || { c.fetch_add(1, Ordering::SeqCst); }).unwrap(); }
        pool.shutdown();
        assert_eq!(n.load(Ordering::SeqCst), 20);
    }
    #[test]
    fn work_stealer_lifo_fifo() {
        let ws = WorkStealer::new();
        ws.push(10); ws.push(20);
        assert_eq!(ws.pop_local(), Some(20));
        assert_eq!(ws.steal(), Some(10));
        assert!(ws.is_empty());
    }
    #[test]
    fn work_stealing_pool_runs_all() {
        let pool = WorkStealingPool::new(2);
        let n = Arc::new(AtomicUsize::new(0));
        for i in 0..10 { let c = Arc::clone(&n); pool.submit(i % 2, move || { c.fetch_add(1, Ordering::SeqCst); }); }
        pool.shutdown();
        assert_eq!(n.load(Ordering::SeqCst), 10);
    }
    #[test]
    fn lock_free_stack_push_pop() {
        let s = LockFreeStack::new();
        s.push(1); s.push(2); s.push(3);
        assert_eq!(s.len(), 3);
        assert_eq!(s.pop(), Some(3));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.pop(), Some(1));
        assert!(s.pop().is_none());
    }
    #[test]
    fn lock_free_stack_concurrent() {
        let s = Arc::new(LockFreeStack::new());
        let handles: Vec<_> = (0..4).map(|_| {
            let s = Arc::clone(&s);
            thread::spawn(move || { for i in 0..1000 { s.push(i); let _ = s.pop(); } })
        }).collect();
        for h in handles { h.join().unwrap(); }
    }
    #[test]
    fn atomic_counter_ops() {
        let c = AtomicCounter::new(0);
        assert_eq!(c.increment(), 1);
        assert_eq!(c.increment(), 2);
        assert_eq!(c.decrement(), 1);
        c.reset();
        assert_eq!(c.get(), 0);
    }
    #[test]
    fn atomic_counter_concurrent() {
        let c = Arc::new(AtomicCounter::new(0));
        let h: Vec<_> = (0..8).map(|_| {
            let c = Arc::clone(&c);
            thread::spawn(move || { for _ in 0..1000 { c.increment(); } })
        }).collect();
        for t in h { t.join().unwrap(); }
        assert_eq!(c.get(), 8000);
    }
    #[test]
    fn mini_runtime_completes() {
        let mut rt = MiniRuntime::new();
        rt.spawn(async {});
        rt.block_on_all();
        assert_eq!(rt.completed_count(), 1);
        assert_eq!(rt.pending_count(), 0);
    }
    #[test]
    fn mpsc_basic() {
        let (tx, rx) = mpsc_channel::<i32>(16);
        tx.send(42).unwrap();
        assert_eq!(rx.recv().unwrap(), 42);
    }
    #[test]
    fn mpsc_multi_producer() {
        let (tx, rx) = mpsc_channel::<i32>(100);
        let h: Vec<_> = (0..4u32).map(|i| { let tx = tx.clone(); thread::spawn(move || { tx.send(i as i32).unwrap(); }) }).collect();
        for t in h { t.join().unwrap(); }
        let mut v: Vec<_> = (0..4).map(|_| rx.recv().unwrap()).collect();
        v.sort();
        assert_eq!(v, vec![0, 1, 2, 3]);
    }
    #[test]
    fn mpsc_close_drains() {
        let (tx, rx) = mpsc_channel::<i32>(16);
        tx.send(1).unwrap(); tx.send(2).unwrap();
        drop(tx);
        assert_eq!(rx.recv().unwrap(), 1);
        assert_eq!(rx.recv().unwrap(), 2);
        assert!(rx.recv().is_err());
    }
    #[test]
    fn broadcast_fan_out() {
        let (tx, rx1) = broadcast_channel::<String>(64);
        let rx2 = rx1.clone();
        tx.send("hello".into()).unwrap();
        assert_eq!(rx1.recv().unwrap(), "hello");
        assert_eq!(rx2.recv().unwrap(), "hello");
    }
    #[test]
    fn broadcast_independent_positions() {
        let (tx, rx1) = broadcast_channel::<i32>(32);
        let rx2 = rx1.clone();
        tx.send(1).unwrap(); tx.send(2).unwrap();
        assert_eq!(rx1.recv().unwrap(), 1);
        assert_eq!(rx2.recv().unwrap(), 1);
        assert_eq!(rx2.recv().unwrap(), 2);
    }
}
