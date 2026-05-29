use std::collections::HashMap;
use std::time::Instant;
use crate::errors::{OmegaError, OmegaResult};

// ---------------------------------------------------------------------------
// Property-based testing
// ---------------------------------------------------------------------------

pub trait Arbitrary: Clone + std::fmt::Debug {
    fn generate(seed: u64) -> Self;
    fn shrink(&self) -> Vec<Self>;
}

pub struct PropertyTest<T: Arbitrary + 'static> {
    name: String,
    generator: Box<dyn Fn(u64) -> T>,
    predicate: Box<dyn Fn(&T) -> bool>,
    trials: usize,
}

impl<T: Arbitrary + 'static> PropertyTest<T> {
    pub fn new(
        name: &str,
        gen: impl Fn(u64) -> T + 'static,
        pred: impl Fn(&T) -> bool + 'static,
    ) -> Self {
        Self { name: name.into(), generator: Box::new(gen), predicate: Box::new(pred), trials: 100 }
    }

    pub fn trials(mut self, n: usize) -> Self { self.trials = n; self }

    pub fn run(&self) -> OmegaResult<()> {
        for i in 0..self.trials {
            let val = (self.generator)(i as u64);
            if !(self.predicate)(&val) {
                let minimal = self.minimize(&val);
                return Err(OmegaError::AssertionError {
                    message: format!("Property '{}' failed at trial {}, minimal counterexample: {:?}", self.name, i, minimal),
                });
            }
        }
        Ok(())
    }

    fn minimize(&self, val: &T) -> T {
        let mut cur = val.clone();
        loop {
            match cur.shrink().into_iter().find(|c| !(self.predicate)(c)) {
                Some(smaller) => cur = smaller,
                None => return cur,
            }
        }
    }
}

impl Arbitrary for i64 {
    fn generate(seed: u64) -> Self {
        (seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) >> 33) as i64
    }
    fn shrink(&self) -> Vec<Self> {
        let mut v = vec![0];
        if *self > 1 { v.push(self / 2); }
        if *self < -1 { v.push(self / 2); }
        v
    }
}

impl Arbitrary for bool {
    fn generate(seed: u64) -> Self { seed % 2 == 0 }
    fn shrink(&self) -> Vec<Self> { if *self { vec![false] } else { vec![] } }
}

impl Arbitrary for String {
    fn generate(seed: u64) -> Self {
        (0..(seed % 20) as usize).map(|i| ((seed.wrapping_add(i as u64) % 26) + b'a') as char).collect()
    }
    fn shrink(&self) -> Vec<Self> {
        if self.is_empty() { vec![] } else { vec![String::new(), self[..self.len() / 2].into()] }
    }
}

// ---------------------------------------------------------------------------
// Fuzzing
// ---------------------------------------------------------------------------

pub struct Fuzzer { corpus: Vec<Vec<u8>>, iterations: usize, max_len: usize }

#[derive(Debug)]
pub struct FuzzResult { pub iterations: usize, pub crashes: Vec<FuzzCrash> }
#[derive(Debug)]
pub struct FuzzCrash { pub iteration: usize, pub input: Vec<u8>, pub error: String }

impl Fuzzer {
    pub fn new() -> Self { Self { corpus: vec![vec![0u8]], iterations: 1000, max_len: 256 } }
    pub fn iterations(mut self, n: usize) -> Self { self.iterations = n; self }
    pub fn corpus(mut self, c: Vec<Vec<u8>>) -> Self { self.corpus = c; self }

    pub fn run(&self, target: impl Fn(&[u8]) -> OmegaResult<()>) -> FuzzResult {
        let mut crashes = Vec::new();
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE;
        for i in 0..self.iterations {
            let input = self.mutate(&self.corpus[i % self.corpus.len()], &mut seed);
            if let Err(e) = target(&input) {
                crashes.push(FuzzCrash { iteration: i, input, error: e.to_string() });
            }
        }
        FuzzResult { iterations: self.iterations, crashes }
    }

    fn mutate(&self, base: &[u8], seed: &mut u64) -> Vec<u8> {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut d = base.to_vec();
        match *seed % 4 {
            0 if !d.is_empty() => { let b = *seed as usize % d.len(); d[b] ^= 1 << ((*seed >> 8) % 8); }
            1 => { let p = *seed as usize % (d.len() + 1); d.insert(p, (*seed >> 16) as u8); }
            2 if !d.is_empty() => { let p = *seed as usize % d.len(); d[p] = (*seed >> 16) as u8; }
            _ if d.len() > 1 => { d.truncate((*seed as usize % d.len()).max(1)); }
            _ => {}
        }
        d.truncate(self.max_len);
        d
    }
}

// ---------------------------------------------------------------------------
// Benchmark utilities
// ---------------------------------------------------------------------------

pub struct Benchmark { name: String, iterations: usize, warmup: usize }

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String, pub iterations: usize, pub mean_ms: f64,
    pub min_ms: f64, pub max_ms: f64, pub median_ms: f64, pub stddev_ms: f64,
}

impl Benchmark {
    pub fn new(name: &str) -> Self { Self { name: name.into(), iterations: 100, warmup: 10 } }
    pub fn iterations(mut self, n: usize) -> Self { self.iterations = n; self }

    pub fn run<F: Fn()>(&self, f: F) -> BenchmarkResult {
        for _ in 0..self.warmup { f(); }
        let mut times: Vec<f64> = (0..self.iterations).map(|_| {
            let s = Instant::now(); f(); s.elapsed().as_secs_f64() * 1000.0
        }).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
        BenchmarkResult {
            name: self.name.clone(), iterations: self.iterations,
            mean_ms: mean, min_ms: times[0], max_ms: *times.last().unwrap(),
            median_ms: times[times.len() / 2], stddev_ms: var.sqrt(),
        }
    }
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {} iters | mean={:.4}ms median={:.4}ms min={:.4}ms max={:.4}ms stddev={:.4}ms",
            self.name, self.iterations, self.mean_ms, self.median_ms, self.min_ms, self.max_ms, self.stddev_ms)
    }
}

// ---------------------------------------------------------------------------
// Mock objects
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CallRecord { pub method: String, pub args: Vec<String>, pub returned: String }

pub struct MockFn {
    name: String, calls: Vec<CallRecord>, queue: Vec<String>, idx: usize, default: String,
}

impl MockFn {
    pub fn new(name: &str, default: &str) -> Self {
        Self { name: name.into(), calls: Vec::new(), queue: Vec::new(), idx: 0, default: default.into() }
    }
    pub fn enqueue(&mut self, resp: &str) { self.queue.push(resp.into()); }
    pub fn call(&mut self, args: Vec<String>) -> String {
        let ret = if self.idx < self.queue.len() { let r = self.queue[self.idx].clone(); self.idx += 1; r }
                  else { self.default.clone() };
        self.calls.push(CallRecord { method: self.name.clone(), args, returned: ret.clone() });
        ret
    }
    pub fn call_count(&self) -> usize { self.calls.len() }
    pub fn was_called_with(&self, args: &[&str]) -> bool {
        let owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        self.calls.iter().any(|c| c.args == owned)
    }
    pub fn last_call(&self) -> Option<&CallRecord> { self.calls.last() }
    pub fn reset(&mut self) { self.calls.clear(); self.idx = 0; }
}

pub struct MockObject { name: String, methods: HashMap<String, MockFn> }

impl MockObject {
    pub fn new(name: &str) -> Self { Self { name: name.into(), methods: HashMap::new() } }
    pub fn add_method(&mut self, method: &str, default: &str) {
        self.methods.insert(method.into(), MockFn::new(method, default));
    }
    pub fn call(&mut self, method: &str, args: Vec<String>) -> OmegaResult<String> {
        self.methods.get_mut(method).map(|m| m.call(args))
            .ok_or_else(|| OmegaError::AttributeError { attr: method.into(), ty: self.name.clone() })
    }
    pub fn method_calls(&self, method: &str) -> Option<&[CallRecord]> {
        self.methods.get(method).map(|m| m.calls.as_slice())
    }
    pub fn total_calls(&self) -> usize { self.methods.values().map(|m| m.call_count()).sum() }
    pub fn reset_all(&mut self) { for m in self.methods.values_mut() { m.reset(); } }
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

pub struct Fixture<T> {
    name: String,
    factory: Box<dyn Fn() -> T>,
    teardown: Option<Box<dyn FnMut(&mut T)>>,
    instances: Vec<T>,
}

impl<T> Fixture<T> {
    pub fn new(name: &str, factory: impl Fn() -> T + 'static) -> Self {
        Self { name: name.into(), factory: Box::new(factory), teardown: None, instances: Vec::new() }
    }
    pub fn with_teardown(mut self, f: impl FnMut(&mut T) + 'static) -> Self {
        self.teardown = Some(Box::new(f)); self
    }
    pub fn create(&mut self) -> &T {
        self.instances.push((self.factory)());
        self.instances.last().unwrap()
    }
    pub fn create_owned(&self) -> T { (self.factory)() }
    pub fn cleanup(&mut self) {
        if let Some(ref mut t) = self.teardown { for inst in &mut self.instances { t(inst); } }
        self.instances.clear();
    }
    pub fn name(&self) -> &str { &self.name }
}

impl<T> Drop for Fixture<T> {
    fn drop(&mut self) { self.cleanup(); }
}

pub struct FixtureRegistry { paths: Vec<String> }

impl FixtureRegistry {
    pub fn new() -> Self { Self { paths: Vec::new() } }
    pub fn temp_path(&mut self, prefix: &str, ext: &str) -> String {
        let p = format!("{}_{}.{}", prefix, self.paths.len(), ext);
        self.paths.push(p.clone()); p
    }
    pub fn temp_dir(&mut self, prefix: &str) -> String {
        let p = format!("{}_{}", prefix, self.paths.len());
        self.paths.push(p.clone()); p
    }
    pub fn paths(&self) -> &[String] { &self.paths }
    pub fn cleanup(&mut self) { self.paths.clear(); }
}

impl Drop for FixtureRegistry {
    fn drop(&mut self) { self.cleanup(); }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_passes_when_predicate_holds() {
        PropertyTest::new("non-negative", |s| i64::generate(s).abs(), |v| *v >= 0)
            .trials(200).run().unwrap();
    }

    #[test]
    fn property_fails_on_violation() {
        let err = PropertyTest::new("always false", |s| i64::generate(s), |_| false)
            .trials(5).run().unwrap_err();
        assert!(err.to_string().contains("always false"));
    }

    #[test]
    fn shrink_converges_to_zero() {
        let mut val = 42i64;
        for _ in 0..50 {
            match val.shrink().into_iter().find(|c| *c != 0) {
                Some(v) => val = v,
                None => { assert_eq!(val, 0); return; }
            }
        }
    }

    #[test]
    fn fuzzer_no_crash_on_clean_target() {
        let r = Fuzzer::new().iterations(20).run(|_| Ok(()));
        assert!(r.crashes.is_empty());
    }

    #[test]
    fn fuzzer_detects_error() {
        let r = Fuzzer::new().iterations(50).run(|d| {
            if d.contains(&0xFF) { Err(OmegaError::RuntimeError { message: "bad".into(), span: None }) }
            else { Ok(()) }
        });
        for c in &r.crashes { assert!(c.error.contains("bad")); }
    }

    #[test]
    fn benchmark_reports_statistics() {
        let r = Benchmark::new("noop").iterations(50).run(|| { let _ = (0..100).fold(0u64, |a, x| a.wrapping_add(x)); });
        assert!(r.mean_ms >= 0.0);
        assert!(r.min_ms <= r.max_ms);
        let _ = format!("{}", r);
    }

    #[test]
    fn mock_fn_tracks_and_replays() {
        let mut m = MockFn::new("f", "default");
        m.enqueue("a"); m.enqueue("b");
        assert_eq!(m.call(vec!["x".into()]), "a");
        assert_eq!(m.call(vec!["y".into()]), "b");
        assert_eq!(m.call(vec!["z".into()]), "default");
        assert_eq!(m.call_count(), 3);
        assert!(m.was_called_with(&["x"]));
        assert!(!m.was_called_with(&["zzz"]));
    }

    #[test]
    fn mock_fn_reset() {
        let mut m = MockFn::new("g", "hi");
        m.call(vec![]); m.reset();
        assert_eq!(m.call_count(), 0);
        assert!(m.last_call().is_none());
    }

    #[test]
    fn mock_object_dispatches() {
        let mut obj = MockObject::new("Svc");
        obj.add_method("get", "404");
        obj.add_method("post", "ok");
        assert_eq!(obj.call("get", vec!["/a".into()]).unwrap(), "404");
        assert_eq!(obj.call("post", vec!["/b".into()]).unwrap(), "ok");
        assert_eq!(obj.total_calls(), 2);
        assert!(obj.call("del", vec![]).is_err());
    }

    #[test]
    fn mock_object_method_query() {
        let mut obj = MockObject::new("Api");
        obj.add_method("fetch", "0");
        obj.call("fetch", vec!["a".into()]).unwrap();
        obj.call("fetch", vec!["b".into()]).unwrap();
        assert_eq!(obj.method_calls("fetch").unwrap().len(), 2);
    }

    #[test]
    fn fixture_creates_sequential_instances() {
        let mut n = 0u64;
        let mut f = Fixture::new("ctr", move || { n += 1; n });
        assert_eq!(*f.create(), 1);
        assert_eq!(*f.create(), 2);
        assert_eq!(*f.create_owned(), 3);
    }

    #[test]
    fn fixture_registry_unique_paths() {
        let mut reg = FixtureRegistry::new();
        let a = reg.temp_path("t", "txt");
        let b = reg.temp_path("t", "txt");
        assert_ne!(a, b);
        let d = reg.temp_dir("ws");
        assert!(d.starts_with("ws_"));
        assert_eq!(reg.paths().len(), 3);
    }
}
