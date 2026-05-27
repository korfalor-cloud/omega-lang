/// Reactive programming: Observables, observers, operators.

use std::collections::HashMap;

pub type ObserverId = usize;

/// Event that can be emitted through an observable.
#[derive(Debug, Clone)]
pub enum Event<T: Clone> {
    Next(T),
    Error(String),
    Complete,
}

/// Trait for observing events.
pub trait Observer<T: Clone> {
    fn on_next(&mut self, value: &T);
    fn on_error(&mut self, error: &str);
    fn on_complete(&mut self);
}

/// Simple callback-based observer.
pub struct CallbackObserver<T: Clone> {
    on_next: Box<dyn FnMut(&T)>,
    on_error: Box<dyn FnMut(&str)>,
    on_complete: Box<dyn FnMut()>,
}

impl<T: Clone> CallbackObserver<T> {
    pub fn new(
        on_next: impl FnMut(&T) + 'static,
        on_error: impl FnMut(&str) + 'static,
        on_complete: impl FnMut() + 'static,
    ) -> Self {
        Self {
            on_next: Box::new(on_next),
            on_error: Box::new(on_error),
            on_complete: Box::new(on_complete),
        }
    }
}

impl<T: Clone> Observer<T> for CallbackObserver<T> {
    fn on_next(&mut self, value: &T) {
        (self.on_next)(value);
    }
    fn on_error(&mut self, error: &str) {
        (self.on_error)(error);
    }
    fn on_complete(&mut self) {
        (self.on_complete)();
    }
}

/// Subject: both an observable and an observer.
pub struct Subject<T: Clone + 'static> {
    observers: HashMap<ObserverId, Box<dyn Observer<T>>>,
    next_id: ObserverId,
    completed: bool,
}

impl<T: Clone + 'static> Subject<T> {
    pub fn new() -> Self {
        Self {
            observers: HashMap::new(),
            next_id: 0,
            completed: false,
        }
    }

    pub fn subscribe(&mut self, observer: impl Observer<T> + 'static) -> ObserverId {
        let id = self.next_id;
        self.next_id += 1;
        self.observers.insert(id, Box::new(observer));
        id
    }

    pub fn unsubscribe(&mut self, id: ObserverId) {
        self.observers.remove(&id);
    }

    pub fn next(&mut self, value: T) {
        if self.completed {
            return;
        }
        for observer in self.observers.values_mut() {
            observer.on_next(&value);
        }
    }

    pub fn error(&mut self, error: &str) {
        for observer in self.observers.values_mut() {
            observer.on_error(error);
        }
        self.completed = true;
    }

    pub fn complete(&mut self) {
        for observer in self.observers.values_mut() {
            observer.on_complete();
        }
        self.completed = true;
    }

    pub fn observer_count(&self) -> usize {
        self.observers.len()
    }

    pub fn is_completed(&self) -> bool {
        self.completed
    }
}

impl<T: Clone + 'static> Default for Subject<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// BehaviorSubject: emits the most recent value to new subscribers.
pub struct BehaviorSubject<T: Clone + 'static> {
    current: T,
    subject: Subject<T>,
}

impl<T: Clone + 'static> BehaviorSubject<T> {
    pub fn new(initial: T) -> Self {
        Self {
            current: initial,
            subject: Subject::new(),
        }
    }

    pub fn subscribe(&mut self, mut observer: impl Observer<T> + 'static) -> ObserverId {
        observer.on_next(&self.current);
        self.subject.subscribe(observer)
    }

    pub fn next(&mut self, value: T) {
        self.current = value.clone();
        self.subject.next(value);
    }

    pub fn value(&self) -> &T {
        &self.current
    }

    pub fn complete(&mut self) {
        self.subject.complete();
    }
}

/// ReplaySubject: buffers N values and replays them to new subscribers.
pub struct ReplaySubject<T: Clone + 'static> {
    buffer: Vec<T>,
    buffer_size: usize,
    subject: Subject<T>,
}

impl<T: Clone + 'static> ReplaySubject<T> {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            buffer: Vec::new(),
            buffer_size,
            subject: Subject::new(),
        }
    }

    pub fn subscribe(&mut self, mut observer: impl Observer<T> + 'static) -> ObserverId {
        for value in &self.buffer {
            observer.on_next(value);
        }
        self.subject.subscribe(observer)
    }

    pub fn next(&mut self, value: T) {
        self.buffer.push(value.clone());
        if self.buffer.len() > self.buffer_size {
            self.buffer.remove(0);
        }
        self.subject.next(value);
    }

    pub fn complete(&mut self) {
        self.subject.complete();
    }

    pub fn buffer(&self) -> &[T] {
        &self.buffer
    }
}

/// Observable stream that can be transformed with operators.
pub struct Observable<T: Clone + 'static> {
    values: Vec<T>,
    completed: bool,
}

impl<T: Clone + 'static> Observable<T> {
    pub fn of(values: Vec<T>) -> Self {
        Self { values, completed: true }
    }

    pub fn empty() -> Self {
        Self { values: Vec::new(), completed: true }
    }

    pub fn from_iter(iter: impl IntoIterator<Item = T>) -> Self {
        Self {
            values: iter.into_iter().collect(),
            completed: true,
        }
    }

    /// Map each value through a function.
    pub fn map<U: Clone + 'static, F: Fn(T) -> U>(self, f: F) -> Observable<U> {
        Observable {
            values: self.values.into_iter().map(f).collect(),
            completed: self.completed,
        }
    }

    /// Filter values by predicate.
    pub fn filter<F: Fn(&T) -> bool>(self, predicate: F) -> Self {
        Observable {
            values: self.values.into_iter().filter(|v| predicate(v)).collect(),
            completed: self.completed,
        }
    }

    /// Take first N values.
    pub fn take(self, n: usize) -> Self {
        Observable {
            values: self.values.into_iter().take(n).collect(),
            completed: self.completed,
        }
    }

    /// Skip first N values.
    pub fn skip(self, n: usize) -> Self {
        Observable {
            values: self.values.into_iter().skip(n).collect(),
            completed: self.completed,
        }
    }

    /// Flatten an Observable of iterables.
    pub fn flatten(self) -> Observable<<T as IntoIterator>::Item>
    where
        T: IntoIterator,
        <T as IntoIterator>::Item: Clone + 'static,
    {
        Observable {
            values: self.values.into_iter().flat_map(|v| v.into_iter()).collect(),
            completed: self.completed,
        }
    }

    /// Reduce to a single value.
    pub fn reduce<F: Fn(T, T) -> T>(self, f: F) -> Option<T> {
        self.values.into_iter().reduce(f)
    }

    /// Collect all values.
    pub fn collect(self) -> Vec<T> {
        self.values
    }

    /// For each value, call a function.
    pub fn for_each<F: FnMut(&T)>(self, mut f: F) {
        for value in &self.values {
            f(value);
        }
    }

    /// Count values.
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Check if any value matches predicate.
    pub fn any<F: Fn(&T) -> bool>(&self, predicate: F) -> bool {
        self.values.iter().any(|v| predicate(v))
    }

    /// Check if all values match predicate.
    pub fn all<F: Fn(&T) -> bool>(&self, predicate: F) -> bool {
        self.values.iter().all(|v| predicate(v))
    }

    /// Get first value.
    pub fn first(&self) -> Option<&T> {
        self.values.first()
    }

    /// Get last value.
    pub fn last(&self) -> Option<&T> {
        self.values.last()
    }

    /// Get value at index.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    /// Distinct values (consecutive dedup).
    pub fn distinct_consecutive(self) -> Self
    where
        T: PartialEq,
    {
        let mut result = Vec::new();
        for value in self.values {
            if result.last() != Some(&value) {
                result.push(value);
            }
        }
        Observable { values: result, completed: self.completed }
    }

    /// Scan (running accumulation).
    pub fn scan<U: Clone + 'static, F: Fn(U, &T) -> U>(self, initial: U, f: F) -> Observable<U> {
        let mut acc = initial;
        let mut result = Vec::new();
        for value in &self.values {
            acc = f(acc, value);
            result.push(acc.clone());
        }
        Observable { values: result, completed: self.completed }
    }

    /// Buffer values into groups of N.
    pub fn buffer(self, count: usize) -> Observable<Vec<T>> {
        let chunks: Vec<Vec<T>> = self.values.chunks(count).map(|c| c.to_vec()).collect();
        Observable { values: chunks, completed: self.completed }
    }

    /// Window into groups of N (sliding).
    pub fn window(self, size: usize) -> Observable<Vec<T>> {
        let windows: Vec<Vec<T>> = self.values.windows(size).map(|w| w.to_vec()).collect();
        Observable { values: windows, completed: self.completed }
    }

    /// Merge with another observable.
    pub fn merge(self, other: Self) -> Self {
        let mut values = self.values;
        values.extend(other.values);
        Observable { values, completed: self.completed }
    }

    /// Zip two observables together.
    pub fn zip<U: Clone + 'static>(self, other: Observable<U>) -> Observable<(T, U)> {
        let values: Vec<(T, U)> = self.values.into_iter()
            .zip(other.values.into_iter())
            .collect();
        Observable { values, completed: self.completed }
    }

    /// Sort values.
    pub fn sort_by<F: Fn(&T, &T) -> std::cmp::Ordering>(self, compare: F) -> Self {
        let mut values = self.values;
        values.sort_by(compare);
        Observable { values, completed: self.completed }
    }

    /// Reverse the order.
    pub fn reverse(self) -> Self {
        let mut values = self.values;
        values.reverse();
        Observable { values, completed: self.completed }
    }
}

/// Create an observable from a range of numbers.
pub fn range(start: i64, end: i64) -> Observable<i64> {
    Observable::from_iter(start..end)
}

/// Create an observable that emits at intervals (simulated).
pub fn interval(count: usize) -> Observable<usize> {
    Observable::from_iter(0..count)
}

/// Combine multiple observables into one.
pub fn combine_latest<T: Clone + 'static>(observables: Vec<Observable<T>>) -> Observable<Vec<T>> {
    if observables.is_empty() {
        return Observable::empty();
    }

    let min_len = observables.iter().map(|o| o.values.len()).min().unwrap_or(0);
    let mut result = Vec::new();

    for i in 0..min_len {
        let combined: Vec<T> = observables.iter().map(|o| o.values[i].clone()).collect();
        result.push(combined);
    }

    Observable::of(result)
}

/// ForkJoin: wait for all observables to complete, then emit last values.
pub fn fork_join<T: Clone + 'static>(observables: Vec<Observable<T>>) -> Observable<Vec<T>> {
    let last_values: Vec<T> = observables.iter()
        .filter_map(|o| o.values.last().cloned())
        .collect();

    if last_values.is_empty() {
        Observable::empty()
    } else {
        Observable::of(vec![last_values])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observable_map_filter() {
        let obs = Observable::of(vec![1, 2, 3, 4, 5]);
        let result = obs
            .filter(|&x| x % 2 == 0)
            .map(|x| x * 10)
            .collect();
        assert_eq!(result, vec![20, 40]);
    }

    #[test]
    fn test_observable_reduce() {
        let obs = Observable::of(vec![1, 2, 3, 4, 5]);
        let sum = obs.reduce(|a, b| a + b);
        assert_eq!(sum, Some(15));
    }

    #[test]
    fn test_subject() {
        let mut subject = Subject::new();
        let values = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let v = values.clone();

        subject.subscribe(CallbackObserver::new(
            move |x: &i32| v.borrow_mut().push(*x),
            |_| {},
            || {},
        ));

        subject.next(1);
        subject.next(2);
        subject.next(3);

        assert_eq!(*values.borrow(), vec![1, 2, 3]);
    }

    #[test]
    fn test_observable_scan() {
        let obs = Observable::of(vec![1, 2, 3, 4]);
        let result = obs.scan(0, |acc, x| acc + x).collect();
        assert_eq!(result, vec![1, 3, 6, 10]);
    }

    #[test]
    fn test_observable_zip() {
        let a = Observable::of(vec![1, 2, 3]);
        let b = Observable::of(vec!["a", "b", "c"]);
        let result = a.zip(b).collect();
        assert_eq!(result, vec![(1, "a"), (2, "b"), (3, "c")]);
    }

    #[test]
    fn test_range() {
        let result = range(0, 5).collect();
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_buffer_window() {
        let obs = Observable::of(vec![1, 2, 3, 4, 5]);
        let buffered = obs.clone().buffer(2).collect();
        assert_eq!(buffered, vec![vec![1, 2], vec![3, 4], vec![5]]);

        let windowed = Observable::of(vec![1, 2, 3, 4, 5]).window(3).collect();
        assert_eq!(windowed, vec![vec![1, 2, 3], vec![2, 3, 4], vec![3, 4, 5]]);
    }
}
