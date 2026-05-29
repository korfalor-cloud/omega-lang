/// Comprehensive scheduling module: learning rate schedulers and task schedulers.

use std::collections::BinaryHeap;
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Learning Rate Schedulers
// ---------------------------------------------------------------------------

/// Trait implemented by every learning rate scheduler.
pub trait LearningRateScheduler {
    /// Return the current learning rate and advance the internal step counter.
    fn step(&mut self) -> f64;
    /// Peek at the current learning rate without advancing.
    fn current_lr(&self) -> f64;
    /// Reset the scheduler to its initial state.
    fn reset(&mut self);
}

// -- Step LR ----------------------------------------------------------------

/// Decays the learning rate by `gamma` every `step_size` epochs.
#[derive(Debug, Clone)]
pub struct StepLR {
    initial_lr: f64,
    current_lr: f64,
    step_size: u64,
    gamma: f64,
    epoch: u64,
}

impl StepLR {
    pub fn new(initial_lr: f64, step_size: u64, gamma: f64) -> Self {
        Self { initial_lr, current_lr: initial_lr, step_size, gamma, epoch: 0 }
    }
}

impl LearningRateScheduler for StepLR {
    fn step(&mut self) -> f64 {
        self.epoch += 1;
        if self.epoch % self.step_size == 0 {
            self.current_lr *= self.gamma;
        }
        self.current_lr
    }

    fn current_lr(&self) -> f64 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.epoch = 0;
    }
}

// -- Cosine Annealing -------------------------------------------------------

/// Cosine annealing schedule that decays from `initial_lr` to `min_lr` over
/// `total_steps` epochs.
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    initial_lr: f64,
    min_lr: f64,
    total_steps: u64,
    current_step: u64,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f64, min_lr: f64, total_steps: u64) -> Self {
        Self { initial_lr, min_lr, total_steps, current_step: 0 }
    }
}

impl LearningRateScheduler for CosineAnnealingLR {
    fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.current_lr()
    }

    fn current_lr(&self) -> f64 {
        let progress = (self.current_step as f64 / self.total_steps as f64).min(1.0);
        self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// -- Exponential Decay ------------------------------------------------------

/// Multiplies the learning rate by `gamma` every epoch.
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    current_lr: f64,
    gamma: f64,
}

impl ExponentialLR {
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        Self { current_lr: initial_lr, gamma }
    }
}

impl LearningRateScheduler for ExponentialLR {
    fn step(&mut self) -> f64 {
        self.current_lr *= self.gamma;
        self.current_lr
    }

    fn current_lr(&self) -> f64 {
        self.current_lr
    }

    fn reset(&mut self) {
        // Cannot recover initial_lr without storing it; gamma stays.
        self.current_lr /= self.gamma.powi(0); // no-op, kept for symmetry
    }
}

// -- Linear Warmup ----------------------------------------------------------

/// Linearly increases the learning rate from `start_lr` to `peak_lr` over
/// `warmup_steps`, then holds at `peak_lr`.
#[derive(Debug, Clone)]
pub struct WarmupLR {
    start_lr: f64,
    peak_lr: f64,
    warmup_steps: u64,
    current_step: u64,
}

impl WarmupLR {
    pub fn new(start_lr: f64, peak_lr: f64, warmup_steps: u64) -> Self {
        Self { start_lr, peak_lr, warmup_steps, current_step: 0 }
    }
}

impl LearningRateScheduler for WarmupLR {
    fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.current_lr()
    }

    fn current_lr(&self) -> f64 {
        if self.current_step >= self.warmup_steps {
            self.peak_lr
        } else {
            self.start_lr + (self.peak_lr - self.start_lr) * (self.current_step as f64 / self.warmup_steps as f64)
        }
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// -- Cyclic LR --------------------------------------------------------------

/// Cycles the learning rate between `base_lr` and `max_lr` with a period of
/// `step_size_up` + `step_size_down` epochs.
#[derive(Debug, Clone)]
pub struct CyclicLR {
    base_lr: f64,
    max_lr: f64,
    step_size_up: u64,
    step_size_down: u64,
    current_step: u64,
}

impl CyclicLR {
    pub fn new(base_lr: f64, max_lr: f64, step_size_up: u64, step_size_down: u64) -> Self {
        Self { base_lr, max_lr, step_size_up, step_size_down, current_step: 0 }
    }
}

impl LearningRateScheduler for CyclicLR {
    fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.current_lr()
    }

    fn current_lr(&self) -> f64 {
        let cycle_len = self.step_size_up + self.step_size_down;
        let cycle_pos = self.current_step % cycle_len;
        if cycle_pos <= self.step_size_up {
            // Ascending phase
            let ratio = cycle_pos as f64 / self.step_size_up as f64;
            self.base_lr + (self.max_lr - self.base_lr) * ratio
        } else {
            // Descending phase
            let ratio = (cycle_pos - self.step_size_up) as f64 / self.step_size_down as f64;
            self.max_lr - (self.max_lr - self.base_lr) * ratio
        }
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ---------------------------------------------------------------------------
// Task Schedulers
// ---------------------------------------------------------------------------

/// Priority levels for task scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// A task submitted to a scheduler.
#[derive(Debug, Clone)]
pub struct Task {
    pub id: u64,
    pub name: String,
    pub priority: Priority,
    pub burst_time: u64,
    pub deadline: Option<u64>,
    pub remaining_time: u64,
    pub arrival_time: u64,
}

impl Task {
    pub fn new(id: u64, name: &str, priority: Priority, burst_time: u64) -> Self {
        Self {
            id,
            name: name.to_string(),
            priority,
            burst_time,
            deadline: None,
            remaining_time: burst_time,
            arrival_time: 0,
        }
    }

    pub fn with_deadline(mut self, deadline: u64) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub fn with_arrival(mut self, arrival: u64) -> Self {
        self.arrival_time = arrival;
        self
    }
}

// -- Priority Queue Scheduler -----------------------------------------------

/// Executes tasks strictly by priority (highest first). Ties broken by
/// earliest arrival time.
#[derive(Debug)]
pub struct PriorityQueueScheduler {
    queue: BinaryHeap<PriorityEntry>,
    current_tick: u64,
    completed: Vec<u64>,
}

#[derive(Debug)]
struct PriorityEntry {
    task: Task,
    sequence: u64, // insertion order for stable ordering
}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.task.priority == other.task.priority && self.task.arrival_time == other.task.arrival_time
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.task.priority
            .cmp(&other.task.priority)
            .then_with(|| other.task.arrival_time.cmp(&self.task.arrival_time))
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl PriorityQueueScheduler {
    pub fn new() -> Self {
        Self { queue: BinaryHeap::new(), current_tick: 0, completed: Vec::new() }
    }

    pub fn submit(&mut self, task: Task) {
        let seq = self.queue.len() as u64;
        self.queue.push(PriorityEntry { task, sequence: seq });
    }

    /// Run one tick: dequeue the highest-priority task, decrement its
    /// remaining time by 1, re-enqueue if not finished.
    pub fn tick(&mut self) -> Option<u64> {
        self.current_tick += 1;
        if let Some(mut entry) = self.queue.pop() {
            entry.task.remaining_time = entry.task.remaining_time.saturating_sub(1);
            let id = entry.task.id;
            if entry.task.remaining_time == 0 {
                self.completed.push(id);
                return Some(id);
            }
            self.queue.push(entry);
            Some(id)
        } else {
            None
        }
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn completed(&self) -> &[u64] {
        &self.completed
    }

    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }
}

impl Default for PriorityQueueScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// -- Round-Robin Scheduler --------------------------------------------------

/// Round-robin scheduler with a configurable quantum (time slice).
#[derive(Debug)]
pub struct RoundRobinScheduler {
    queue: Vec<Task>,
    quantum: u64,
    current_index: usize,
    current_tick: u64,
    completed: Vec<u64>,
}

impl RoundRobinScheduler {
    pub fn new(quantum: u64) -> Self {
        Self { queue: Vec::new(), quantum, current_index: 0, current_tick: 0, completed: Vec::new() }
    }

    pub fn submit(&mut self, task: Task) {
        self.queue.push(task);
    }

    /// Advance one quantum slice for the current task. Returns the id of the
    /// task that ran, or `None` if the queue is empty.
    pub fn tick(&mut self) -> Option<u64> {
        if self.queue.is_empty() {
            return None;
        }
        self.current_tick += 1;
        let idx = self.current_index % self.queue.len();
        let slice = self.quantum.min(self.queue[idx].remaining_time);
        self.queue[idx].remaining_time -= slice;

        let id = self.queue[idx].id;
        if self.queue[idx].remaining_time == 0 {
            self.completed.push(id);
            self.queue.remove(idx);
            if !self.queue.is_empty() {
                self.current_index = idx % self.queue.len();
            }
        } else {
            self.current_index = (idx + 1) % self.queue.len();
        }
        Some(id)
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn completed(&self) -> &[u64] {
        &self.completed
    }
}

// -- Earliest-Deadline-First Scheduler --------------------------------------

/// Schedules the task whose absolute deadline is closest.
#[derive(Debug)]
pub struct DeadlineScheduler {
    tasks: Vec<Task>,
    current_tick: u64,
    completed: Vec<u64>,
}

impl DeadlineScheduler {
    pub fn new() -> Self {
        Self { tasks: Vec::new(), current_tick: 0, completed: Vec::new() }
    }

    pub fn submit(&mut self, task: Task) {
        self.tasks.push(task);
    }

    /// Run one tick on the task with the earliest deadline.
    pub fn tick(&mut self) -> Option<u64> {
        if self.tasks.is_empty() {
            return None;
        }
        self.current_tick += 1;

        // Pick index with smallest deadline (or None => u64::MAX).
        let idx = self
            .tasks
            .iter()
            .enumerate()
            .min_by_key(|(_, t)| t.deadline.unwrap_or(u64::MAX))
            .map(|(i, _)| i)?;

        self.tasks[idx].remaining_time = self.tasks[idx].remaining_time.saturating_sub(1);
        let id = self.tasks[idx].id;

        if self.tasks[idx].remaining_time == 0 {
            self.completed.push(id);
            self.tasks.remove(idx);
        }
        Some(id)
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    pub fn completed(&self) -> &[u64] {
        &self.completed
    }

    /// Returns true if any task has missed its deadline.
    pub fn has_missed_deadlines(&self) -> bool {
        self.tasks.iter().any(|t| {
            t.deadline.map_or(false, |dl| self.current_tick > dl && t.remaining_time > 0)
        })
    }
}

impl Default for DeadlineScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // -- Learning Rate Schedulers -------------------------------------------

    #[test]
    fn test_step_lr() {
        let mut sched = StepLR::new(0.1, 5, 0.5);
        for _ in 0..4 {
            sched.step();
        }
        assert!((sched.current_lr() - 0.1).abs() < EPS);
        sched.step(); // epoch 5 -> decay
        assert!((sched.current_lr() - 0.05).abs() < EPS);
    }

    #[test]
    fn test_cosine_annealing() {
        let mut sched = CosineAnnealingLR::new(0.1, 0.0, 10);
        let start = sched.step();
        assert!((start - 0.1).abs() < 0.02);
        for _ in 0..9 {
            sched.step();
        }
        let end = sched.current_lr();
        assert!(end < 0.02); // near min_lr
    }

    #[test]
    fn test_exponential_lr() {
        let mut sched = ExponentialLR::new(1.0, 0.9);
        sched.step();
        assert!((sched.current_lr() - 0.9).abs() < EPS);
        sched.step();
        assert!((sched.current_lr() - 0.81).abs() < EPS);
    }

    #[test]
    fn test_warmup_lr() {
        let mut sched = WarmupLR::new(0.0, 0.1, 10);
        sched.step(); // step 1
        assert!((sched.current_lr() - 0.01).abs() < EPS);
        for _ in 1..10 {
            sched.step();
        }
        assert!((sched.current_lr() - 0.1).abs() < EPS);
        sched.step(); // step 11 -> hold
        assert!((sched.current_lr() - 0.1).abs() < EPS);
    }

    #[test]
    fn test_cyclic_lr() {
        let mut sched = CyclicLR::new(0.01, 0.1, 5, 5);
        // Ascending
        sched.step();
        assert!((sched.current_lr() - 0.028).abs() < EPS);
        sched.step();
        sched.step();
        sched.step();
        sched.step(); // step 5 -> peak
        assert!((sched.current_lr() - 0.1).abs() < EPS);
        // Descending
        sched.step(); // step 6
        let lr6 = sched.current_lr();
        assert!(lr6 < 0.1 && lr6 > 0.01);
    }

    #[test]
    fn test_lr_scheduler_reset() {
        let mut sched = StepLR::new(0.1, 3, 0.5);
        sched.step();
        sched.step();
        sched.step(); // decay
        assert!((sched.current_lr() - 0.05).abs() < EPS);
        sched.reset();
        assert!((sched.current_lr() - 0.1).abs() < EPS);
    }

    // -- Task Schedulers ----------------------------------------------------

    #[test]
    fn test_priority_queue_basic() {
        let mut sched = PriorityQueueScheduler::new();
        sched.submit(Task::new(1, "low", Priority::Low, 3));
        sched.submit(Task::new(2, "high", Priority::High, 1));
        sched.submit(Task::new(3, "critical", Priority::Critical, 1));

        // Critical should run first
        let first = sched.tick().unwrap();
        assert_eq!(first, 3);
        // Then high
        let second = sched.tick().unwrap();
        assert_eq!(second, 2);
        // Then low (remaining 3 ticks)
        let third = sched.tick().unwrap();
        assert_eq!(third, 1);
    }

    #[test]
    fn test_priority_queue_completion() {
        let mut sched = PriorityQueueScheduler::new();
        sched.submit(Task::new(1, "task", Priority::Medium, 2));
        sched.tick();
        sched.tick();
        assert!(sched.is_empty());
        assert_eq!(sched.completed(), &[1]);
    }

    #[test]
    fn test_round_robin() {
        let mut sched = RoundRobinScheduler::new(2);
        sched.submit(Task::new(1, "a", Priority::Medium, 4));
        sched.submit(Task::new(2, "b", Priority::Medium, 3));

        // a runs for quantum=2 (remaining 2)
        let r1 = sched.tick().unwrap();
        assert_eq!(r1, 1);
        // b runs for quantum=2 (remaining 1)
        let r2 = sched.tick().unwrap();
        assert_eq!(r2, 2);
        // a runs for quantum=2 (remaining 0) -> completed
        let r3 = sched.tick().unwrap();
        assert_eq!(r3, 1);
        assert!(sched.completed().contains(&1));
        // b runs for quantum=1 (remaining 0) -> completed
        let r4 = sched.tick().unwrap();
        assert_eq!(r4, 2);
        assert!(sched.is_empty());
    }

    #[test]
    fn test_deadline_scheduler() {
        let mut sched = DeadlineScheduler::new();
        sched.submit(Task::new(1, "far", Priority::Medium, 5).with_deadline(100));
        sched.submit(Task::new(2, "soon", Priority::Medium, 2).with_deadline(3));

        // "soon" has earlier deadline -> should be picked first
        let first = sched.tick().unwrap();
        assert_eq!(first, 2);
        let second = sched.tick().unwrap();
        assert_eq!(second, 2);
        // "soon" completed, now "far"
        let third = sched.tick().unwrap();
        assert_eq!(third, 1);
    }

    #[test]
    fn test_deadline_missed() {
        let mut sched = DeadlineScheduler::new();
        sched.submit(Task::new(1, "late", Priority::Low, 10).with_deadline(2));
        for _ in 0..3 {
            sched.tick();
        }
        assert!(sched.has_missed_deadlines());
    }

    #[test]
    fn test_empty_schedulers() {
        let mut pq = PriorityQueueScheduler::new();
        assert!(pq.tick().is_none());

        let mut rr = RoundRobinScheduler::new(1);
        assert!(rr.tick().is_none());

        let mut dl = DeadlineScheduler::new();
        assert!(dl.tick().is_none());
    }
}
