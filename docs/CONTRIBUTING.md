# Contributing to Omega

Thank you for your interest in contributing to Omega! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Rust 1.70 or later
- Git

### Building

```bash
git clone https://github.com/omega-lang/omega.git
cd omega
cargo build
```

### Running Tests

```bash
cargo test
```

### Running Benchmarks

```bash
cargo bench
```

## Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`cargo test`)
5. Run clippy (`cargo clippy`)
6. Format code (`cargo fmt`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Code Style

### Rust Code

- Follow the Rust style guide
- Use `cargo fmt` to format code
- Use `cargo clippy` to catch common mistakes
- Write documentation for public APIs
- Add tests for new functionality

### Omega Code

- Use snake_case for variables and functions
- Use PascalCase for types and structs
- Use SCREAMING_SNAKE_CASE for constants
- Add doc comments for public items
- Keep functions small and focused

## Testing

### Unit Tests

Place unit tests in the same file as the code they test:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example() {
        assert_eq!(1 + 1, 2);
    }
}
```

### Integration Tests

Place integration tests in the `tests/` directory:

```rust
// tests/integration_test.rs
use omega_lang::*;

#[test]
fn test_full_pipeline() {
    // Test the full compilation pipeline
}
```

### Benchmarks

Place benchmarks in the `benches/` directory:

```rust
// benches/my_bench.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_example(c: &mut Criterion) {
    c.bench_function("example", |b| {
        b.iter(|| {
            // Code to benchmark
        })
    });
}

criterion_group!(benches, bench_example);
criterion_main!(benches);
```

## Documentation

### Code Documentation

Use `///` for public item documentation:

```rust
/// Adds two numbers together.
///
/// # Arguments
///
/// * `a` - The first number
/// * `b` - The second number
///
/// # Returns
///
/// The sum of `a` and `b`
///
/// # Examples
///
/// ```
/// let result = add(2, 3);
/// assert_eq!(result, 5);
/// ```
pub fn add(a: i64, b: i64) -> i64 {
    a + b
}
```

### Module Documentation

Use `//!` for module-level documentation:

```//! This module provides mathematical functions.

// Module code here
```

## Pull Request Guidelines

1. **One feature per PR**: Keep pull requests focused on a single feature or fix
2. **Write descriptive commit messages**: Explain what and why, not how
3. **Add tests**: All new code should have tests
4. **Update documentation**: Keep docs in sync with code changes
5. **Keep it clean**: Remove debug code, unused imports, etc.

## Issue Reporting

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: How to reproduce the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: OS, Rust version, etc.

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

### Review Checklist

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Code is formatted with `cargo fmt`
- [ ] No clippy warnings
- [ ] Documentation is updated
- [ ] Tests are added for new functionality

## Community

- **Discord**: Join our Discord server
- **GitHub Discussions**: Use GitHub Discussions for questions
- **Twitter**: Follow us on Twitter

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
