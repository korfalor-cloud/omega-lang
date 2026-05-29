/// Advanced queue structures: binary heap, deque, circular buffer, work queue, message queue.
use std::collections::{HashMap, VecDeque};

#[derive(Debug)]
pub struct BinaryHeap<T> { data: Vec<T>, cmp: fn(&T, &T) -> std::cmp::Ordering }

impl<T> BinaryHeap<T> {
    pub fn new(cmp: fn(&T, &T) -> std::cmp::Ordering) -> Self { Self { data: Vec::new(), cmp } }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn peek(&self) -> Option<&T> { self.data.first() }

    pub fn push(&mut self, item: T) {
        self.data.push(item);
        self.sift_up(self.data.len() - 1);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() { return None; }
        let last = self.data.len() - 1;
        self.data.swap(0, last);
        let item = self.data.pop().unwrap();
        if !self.data.is_empty() { self.sift_down(0); }
        Some(item)
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let p = (idx - 1) / 2;
            if (self.cmp)(&self.data[idx], &self.data[p]) == std::cmp::Ordering::Greater {
                self.data.swap(idx, p); idx = p;
            } else { break; }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.data.len();
        loop {
            let (l, r, mut best) = (2 * idx + 1, 2 * idx + 2, idx);
            if l < len && (self.cmp)(&self.data[l], &self.data[best]) == std::cmp::Ordering::Greater { best = l; }
            if r < len && (self.cmp)(&self.data[r], &self.data[best]) == std::cmp::Ordering::Greater { best = r; }
            if best != idx { self.data.swap(idx, best); idx = best; } else { break; }
        }
    }
}

#[derive(Debug)]
pub struct Deque<T> { front: Vec<T>, back: Vec<T> }

impl<T> Deque<T> {
    pub fn new() -> Self { Self { front: Vec::new(), back: Vec::new() } }
    pub fn push_front(&mut self, item: T) { self.front.push(item); }
    pub fn push_back(&mut self, item: T) { self.back.push(item); }
    pub fn front(&self) -> Option<&T> { self.front.last().or_else(|| self.back.first()) }
    pub fn back(&self) -> Option<&T> { self.back.last().or_else(|| self.front.first()) }
    pub fn len(&self) -> usize { self.front.len() + self.back.len() }
    pub fn is_empty(&self) -> bool { self.front.is_empty() && self.back.is_empty() }

    pub fn pop_front(&mut self) -> Option<T> {
        if let Some(v) = self.front.pop() { return Some(v); }
        self.back.reverse();
        std::mem::swap(&mut self.front, &mut self.back);
        self.front.pop()
    }

    pub fn pop_back(&mut self) -> Option<T> {
        if let Some(v) = self.back.pop() { return Some(v); }
        self.front.reverse();
        std::mem::swap(&mut self.front, &mut self.back);
        self.back.pop()
    }
}

impl<T> Default for Deque<T> { fn default() -> Self { Self::new() } }

#[derive(Debug)]
pub struct CircularBuffer<T> { buf: Vec<Option<T>>, cap: usize, head: usize, len: usize }

impl<T> CircularBuffer<T> {
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0, "capacity must be > 0");
        Self { buf: (0..cap).map(|_| None).collect(), cap, head: 0, len: 0 }
    }
    pub fn len(&self) -> usize { self.len }
    pub fn capacity(&self) -> usize { self.cap }
    pub fn is_full(&self) -> bool { self.len == self.cap }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn push(&mut self, item: T) -> Option<T> {
        let evicted = if self.len == self.cap { self.buf[self.head].take() } else { self.len += 1; None };
        self.buf[self.head] = Some(item);
        self.head = (self.head + 1) % self.cap;
        evicted
    }

    pub fn pop_oldest(&mut self) -> Option<T> {
        if self.len == 0 { return None; }
        let tail = (self.head + self.cap - self.len) % self.cap;
        self.len -= 1;
        self.buf[tail].take()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let (cap, len, head) = (self.cap, self.len, self.head);
        (0..len).filter_map(move |i| self.buf[(head + cap - len + i) % cap].as_ref())
    }
}

#[derive(Debug, Clone)]
pub struct WorkItem<T> { pub task: T, pub priority: u32, gen: u64 }

#[derive(Debug)]
pub struct WorkQueue<T> { heap: Vec<WorkItem<T>>, generation: u64 }

impl<T> WorkQueue<T> {
    pub fn new() -> Self { Self { heap: Vec::new(), generation: 0 } }
    pub fn len(&self) -> usize { self.heap.len() }
    pub fn is_empty(&self) -> bool { self.heap.is_empty() }
    pub fn peek(&self) -> Option<(&T, u32)> { self.heap.first().map(|w| (&w.task, w.priority)) }

    pub fn enqueue(&mut self, task: T, priority: u32) {
        let gen = self.generation;
        self.generation += 1;
        self.heap.push(WorkItem { task, priority, gen });
        self.sift_up(self.heap.len() - 1);
    }

    pub fn dequeue(&mut self) -> Option<T> {
        if self.heap.is_empty() { return None; }
        let last = self.heap.len() - 1;
        self.heap.swap(0, last);
        let item = self.heap.pop().unwrap();
        if !self.heap.is_empty() { self.sift_down(0); }
        Some(item.task)
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let p = (idx - 1) / 2;
            if self.has_priority(idx, p) { self.heap.swap(idx, p); idx = p; } else { break; }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        loop {
            let (l, r, mut best) = (2 * idx + 1, 2 * idx + 2, idx);
            if l < self.heap.len() && self.has_priority(l, best) { best = l; }
            if r < self.heap.len() && self.has_priority(r, best) { best = r; }
            if best != idx { self.heap.swap(idx, best); idx = best; } else { break; }
        }
    }

    fn has_priority(&self, a: usize, b: usize) -> bool {
        let (ia, ib) = (&self.heap[a], &self.heap[b]);
        ia.priority < ib.priority || (ia.priority == ib.priority && ia.gen < ib.gen)
    }
}

impl<T> Default for WorkQueue<T> { fn default() -> Self { Self::new() } }

#[derive(Debug)]
pub struct SimpleMessageQueue<T: Clone> { topics: HashMap<String, VecDeque<T>> }

impl<T: Clone> SimpleMessageQueue<T> {
    pub fn new() -> Self { Self { topics: HashMap::new() } }
    pub fn topic_len(&self, t: &str) -> usize { self.topics.get(t).map_or(0, |q| q.len()) }
    pub fn total_messages(&self) -> usize { self.topics.values().map(|q| q.len()).sum() }
    pub fn topics(&self) -> Vec<&str> { self.topics.keys().map(|s| s.as_str()).collect() }
    pub fn peek(&self, t: &str) -> Option<&T> { self.topics.get(t)?.front() }

    pub fn publish(&mut self, topic: &str, msg: T) {
        self.topics.entry(topic.to_string()).or_default().push_back(msg);
    }
    pub fn consume(&mut self, t: &str) -> Option<T> { self.topics.get_mut(t)?.pop_front() }
    pub fn purge(&mut self, t: &str) { if let Some(q) = self.topics.get_mut(t) { q.clear(); } }
}

impl<T: Clone> Default for SimpleMessageQueue<T> { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_heap_orders_correctly() {
        let mut h = BinaryHeap::new(|a: &i32, b: &i32| a.cmp(b));
        h.push(3); h.push(1); h.push(4); h.push(1); h.push(5);
        assert_eq!(h.pop(), Some(5));
        assert_eq!(h.pop(), Some(4));
        assert_eq!(h.pop(), Some(3));
        assert_eq!(h.pop(), Some(1));
        assert_eq!(h.pop(), Some(1));
        assert!(h.is_empty());
    }

    #[test]
    fn binary_heap_peek_and_empty() {
        let mut h = BinaryHeap::new(|a: &i32, b: &i32| a.cmp(b));
        assert!(h.peek().is_none());
        h.push(10);
        assert_eq!(h.peek(), Some(&10));
        h.push(20);
        assert_eq!(h.peek(), Some(&20));
    }

    #[test]
    fn deque_front_and_back() {
        let mut d = Deque::new();
        d.push_back(1); d.push_back(2); d.push_front(0);
        assert_eq!(d.pop_front(), Some(0));
        assert_eq!(d.pop_front(), Some(1));
        assert_eq!(d.pop_front(), Some(2));
        assert!(d.is_empty());
    }

    #[test]
    fn deque_pop_back() {
        let mut d = Deque::new();
        d.push_back(10); d.push_back(20); d.push_front(5);
        assert_eq!(d.pop_back(), Some(20));
        assert_eq!(d.pop_back(), Some(10));
        assert_eq!(d.pop_back(), Some(5));
    }

    #[test]
    fn circular_buffer_evicts_oldest() {
        let mut cb = CircularBuffer::new(3);
        assert!(cb.push(1).is_none());
        assert!(cb.push(2).is_none());
        assert!(cb.push(3).is_none());
        assert!(cb.is_full());
        assert_eq!(cb.push(4), Some(1));
        let items: Vec<&i32> = cb.iter().collect();
        assert_eq!(items, vec![&2, &3, &4]);
    }

    #[test]
    fn circular_buffer_pop_oldest() {
        let mut cb = CircularBuffer::new(3);
        cb.push(10); cb.push(20); cb.push(30);
        assert_eq!(cb.pop_oldest(), Some(10));
        assert_eq!(cb.pop_oldest(), Some(20));
        assert_eq!(cb.len(), 1);
    }

    #[test]
    fn work_queue_respects_priority() {
        let mut wq = WorkQueue::new();
        wq.enqueue("low", 10); wq.enqueue("high", 1); wq.enqueue("medium", 5);
        assert_eq!(wq.dequeue(), Some("high"));
        assert_eq!(wq.dequeue(), Some("medium"));
        assert_eq!(wq.dequeue(), Some("low"));
    }

    #[test]
    fn work_queue_fifo_on_tie() {
        let mut wq = WorkQueue::new();
        wq.enqueue("first", 1); wq.enqueue("second", 1); wq.enqueue("third", 1);
        assert_eq!(wq.dequeue(), Some("first"));
        assert_eq!(wq.dequeue(), Some("second"));
        assert_eq!(wq.dequeue(), Some("third"));
    }

    #[test]
    fn message_queue_publish_consume() {
        let mut mq = SimpleMessageQueue::new();
        mq.publish("events", "hello");
        mq.publish("events", "world");
        assert_eq!(mq.topic_len("events"), 2);
        assert_eq!(mq.consume("events"), Some("hello"));
        assert_eq!(mq.consume("events"), Some("world"));
        assert!(mq.consume("events").is_none());
    }

    #[test]
    fn message_queue_multi_topic_and_purge() {
        let mut mq = SimpleMessageQueue::new();
        mq.publish("a", 1); mq.publish("b", 2);
        assert_eq!(mq.total_messages(), 2);
        let mut ts: Vec<&str> = mq.topics(); ts.sort();
        assert_eq!(ts, vec!["a", "b"]);
        mq.purge("a");
        assert_eq!(mq.topic_len("a"), 0);
    }
}
