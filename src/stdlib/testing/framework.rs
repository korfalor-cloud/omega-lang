use std::collections::HashMap;
use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;

pub struct TestSuite {
    name: String,
    tests: Vec<TestCase>,
    setup: Option<Box<dyn Fn()>>,
    teardown: Option<Box<dyn Fn()>>,
}

pub struct TestCase {
    name: String,
    test_fn: Box<dyn Fn() -> OmegaResult<()>>,
    should_panic: bool,
    timeout_ms: Option<u64>,
    tags: Vec<String>,
}

pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub duration_ms: f64,
    pub error: Option<String>,
}

pub struct TestReport {
    pub results: Vec<TestResult>,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub duration_ms: f64,
}

impl TestSuite {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tests: Vec::new(),
            setup: None,
            teardown: None,
        }
    }

    pub fn add_test(&mut self, name: &str, test_fn: impl Fn() -> OmegaResult<()> + 'static) {
        self.tests.push(TestCase {
            name: name.to_string(),
            test_fn: Box::new(test_fn),
            should_panic: false,
            timeout_ms: None,
            tags: Vec::new(),
        });
    }

    pub fn add_test_with_config(
        &mut self,
        name: &str,
        test_fn: impl Fn() -> OmegaResult<()> + 'static,
        should_panic: bool,
        timeout_ms: Option<u64>,
        tags: Vec<String>,
    ) {
        self.tests.push(TestCase {
            name: name.to_string(),
            test_fn: Box::new(test_fn),
            should_panic,
            timeout_ms,
            tags,
        });
    }

    pub fn set_setup(&mut self, setup: impl Fn() + 'static) {
        self.setup = Some(Box::new(setup));
    }

    pub fn set_teardown(&mut self, teardown: impl Fn() + 'static) {
        self.teardown = Some(Box::new(teardown));
    }

    pub fn run(&self) -> TestReport {
        let mut results = Vec::new();
        let start = std::time::Instant::now();

        for test in &self.tests {
            if let Some(setup) = &self.setup {
                setup();
            }

            let test_start = std::time::Instant::now();
            let result = if test.should_panic {
                match (test.test_fn)() {
                    Ok(_) => TestResult {
                        name: test.name.clone(),
                        passed: false,
                        duration_ms: test_start.elapsed().as_secs_f64() * 1000.0,
                        error: Some("Expected panic but test succeeded".to_string()),
                    },
                    Err(_) => TestResult {
                        name: test.name.clone(),
                        passed: true,
                        duration_ms: test_start.elapsed().as_secs_f64() * 1000.0,
                        error: None,
                    },
                }
            } else {
                match (test.test_fn)() {
                    Ok(_) => TestResult {
                        name: test.name.clone(),
                        passed: true,
                        duration_ms: test_start.elapsed().as_secs_f64() * 1000.0,
                        error: None,
                    },
                    Err(e) => TestResult {
                        name: test.name.clone(),
                        passed: false,
                        duration_ms: test_start.elapsed().as_secs_f64() * 1000.0,
                        error: Some(e.to_string()),
                    },
                }
            };

            results.push(result);

            if let Some(teardown) = &self.teardown {
                teardown();
            }
        }

        let passed = results.iter().filter(|r| r.passed).count();
        let failed = results.iter().filter(|r| !r.passed).count();

        TestReport {
            total: results.len(),
            passed,
            failed,
            skipped: 0,
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            results,
        }
    }
}

pub fn assert(condition: bool) -> OmegaResult<()> {
    if condition {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: "Assertion failed".to_string(),
        })
    }
}

pub fn assert_eq<T: PartialEq + std::fmt::Debug>(a: &T, b: &T) -> OmegaResult<()> {
    if a == b {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: format!("Expected {:?} to equal {:?}", a, b),
        })
    }
}

pub fn assert_ne<T: PartialEq + std::fmt::Debug>(a: &T, b: &T) -> OmegaResult<()> {
    if a != b {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: format!("Expected {:?} to not equal {:?}", a, b),
        })
    }
}

pub fn assert_lt<T: PartialOrd + std::fmt::Debug>(a: &T, b: &T) -> OmegaResult<()> {
    if a < b {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: format!("Expected {:?} < {:?}", a, b),
        })
    }
}

pub fn assert_le<T: PartialOrd + std::fmt::Debug>(a: &T, b: &T) -> OmegaResult<()> {
    if a <= b {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: format!("Expected {:?} <= {:?}", a, b),
        })
    }
}

pub fn assert_gt<T: PartialOrd + std::fmt::Debug>(a: &T, b: &T) -> OmegaResult<()> {
    if a > b {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: format!("Expected {:?} > {:?}", a, b),
        })
    }
}

pub fn assert_ge<T: PartialOrd + std::fmt::Debug>(a: &T, b: &T) -> OmegaResult<()> {
    if a >= b {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: format!("Expected {:?} >= {:?}", a, b),
        })
    }
}

pub fn assert_approx_eq(a: f64, b: f64, epsilon: f64) -> OmegaResult<()> {
    if (a - b).abs() < epsilon {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: format!("Expected {} ≈ {} (epsilon={})", a, b, epsilon),
        })
    }
}

pub fn assert_contains<T: PartialEq>(collection: &[T], item: &T) -> OmegaResult<()> {
    if collection.contains(item) {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: "Collection does not contain expected item".to_string(),
        })
    }
}

pub fn assert_starts_with(s: &str, prefix: &str) -> OmegaResult<()> {
    if s.starts_with(prefix) {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: format!("Expected '{}' to start with '{}'", s, prefix),
        })
    }
}

pub fn assert_ends_with(s: &str, suffix: &str) -> OmegaResult<()> {
    if s.ends_with(suffix) {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: format!("Expected '{}' to end with '{}'", s, suffix),
        })
    }
}

pub fn assert_matches(s: &str, pattern: &str) -> OmegaResult<()> {
    if s.contains(pattern) {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: format!("Expected '{}' to match '{}'", s, pattern),
        })
    }
}

pub fn assert_panics(f: impl FnOnce()) -> OmegaResult<()> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    if result.is_err() {
        Ok(())
    } else {
        Err(OmegaError::AssertionError {
            message: "Expected panic but function succeeded".to_string(),
        })
    }
}
