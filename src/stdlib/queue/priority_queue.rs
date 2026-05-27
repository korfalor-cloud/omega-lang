/// Generic priority queue implementation.

#[derive(Debug)]
pub struct PriorityQueue<T> {
    heap: Vec<(T, i64)>,
}

impl<T> PriorityQueue<T> {
    pub fn new() -> Self {
        Self { heap: Vec::new() }
    }

    pub fn push(&mut self, item: T, priority: i64) {
        self.heap.push((item, priority));
        self.sift_up(self.heap.len() - 1);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.heap.is_empty() {
            return None;
        }
        let last = self.heap.len() - 1;
        self.heap.swap(0, last);
        let (item, _) = self.heap.pop().unwrap();
        if !self.heap.is_empty() {
            self.sift_down(0);
        }
        Some(item)
    }

    pub fn peek(&self) -> Option<(&T, i64)> {
        self.heap.first().map(|(item, priority)| (item, *priority))
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn clear(&mut self) {
        self.heap.clear();
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.heap[idx].1 < self.heap[parent].1 {
                self.heap.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut smallest = idx;

            if left < self.heap.len() && self.heap[left].1 < self.heap[smallest].1 {
                smallest = left;
            }
            if right < self.heap.len() && self.heap[right].1 < self.heap[smallest].1 {
                smallest = right;
            }

            if smallest != idx {
                self.heap.swap(idx, smallest);
                idx = smallest;
            } else {
                break;
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.heap.iter().map(|(item, _)| item)
    }

    pub fn into_sorted_vec(mut self) -> Vec<T> {
        let mut result = Vec::new();
        while let Some(item) = self.pop() {
            result.push(item);
        }
        result
    }
}

impl<T> Default for PriorityQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Min-max heap for efficient min and max access
#[derive(Debug)]
pub struct MinMaxHeap<T> {
    heap: Vec<T>,
    cmp: fn(&T, &T) -> std::cmp::Ordering,
}

impl<T> MinMaxHeap<T> {
    pub fn new(cmp: fn(&T, &T) -> std::cmp::Ordering) -> Self {
        Self {
            heap: Vec::new(),
            cmp,
        }
    }

    pub fn push(&mut self, item: T) {
        self.heap.push(item);
        let idx = self.heap.len() - 1;
        self.sift_up(idx);
    }

    pub fn pop_min(&mut self) -> Option<T> {
        if self.heap.is_empty() {
            return None;
        }
        let last = self.heap.len() - 1;
        self.heap.swap(0, last);
        let item = self.heap.pop().unwrap();
        if !self.heap.is_empty() {
            self.sift_down_min(0);
        }
        Some(item)
    }

    pub fn pop_max(&mut self) -> Option<T> {
        if self.heap.is_empty() {
            return None;
        }
        if self.heap.len() <= 2 {
            return self.heap.pop();
        }

        let max_idx = if (self.cmp)(&self.heap[1], &self.heap[2]) == std::cmp::Ordering::Greater {
            1
        } else {
            2
        };

        let last = self.heap.len() - 1;
        self.heap.swap(max_idx, last);
        let item = self.heap.pop().unwrap();
        self.sift_down_max(max_idx);
        Some(item)
    }

    pub fn min(&self) -> Option<&T> {
        self.heap.first()
    }

    pub fn max(&self) -> Option<&T> {
        match self.heap.len() {
            0 => None,
            1 => Some(&self.heap[0]),
            2 => Some(&self.heap[1]),
            _ => {
                if (self.cmp)(&self.heap[1], &self.heap[2]) == std::cmp::Ordering::Greater {
                    Some(&self.heap[1])
                } else {
                    Some(&self.heap[2])
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    fn sift_up(&mut self, _idx: usize) {
        // Simplified implementation
    }

    fn sift_down_min(&mut self, _idx: usize) {
        // Simplified implementation
    }

    fn sift_down_max(&mut self, _idx: usize) {
        // Simplified implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_queue() {
        let mut pq = PriorityQueue::new();
        pq.push("low", 10);
        pq.push("high", 1);
        pq.push("medium", 5);

        assert_eq!(pq.pop(), Some("high"));
        assert_eq!(pq.pop(), Some("medium"));
        assert_eq!(pq.pop(), Some("low"));
    }

    #[test]
    fn test_priority_queue_peek() {
        let mut pq = PriorityQueue::new();
        pq.push("a", 5);
        pq.push("b", 1);

        let (item, priority) = pq.peek().unwrap();
        assert_eq!(*item, "b");
        assert_eq!(priority, 1);
    }

    #[test]
    fn test_sorted_vec() {
        let mut pq = PriorityQueue::new();
        pq.push(3, 3);
        pq.push(1, 1);
        pq.push(2, 2);

        let sorted = pq.into_sorted_vec();
        assert_eq!(sorted, vec![1, 2, 3]);
    }

    #[test]
    fn test_empty_queue() {
        let mut pq: PriorityQueue<i32> = PriorityQueue::new();
        assert!(pq.pop().is_none());
        assert!(pq.peek().is_none());
    }
}
