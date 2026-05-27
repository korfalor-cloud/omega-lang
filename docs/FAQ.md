# Frequently Asked Questions

## General

### What is Omega?

Omega is a modern, statically-typed programming language designed for developer experience, safety, and performance. It features first-class functions, pattern matching, error handling, and a rich standard library.

### What can I build with Omega?

You can build a wide range of applications:
- Command-line tools
- Web servers and APIs
- Data processing pipelines
- System utilities
- Libraries and frameworks
- Scripts and automation

### Is Omega production-ready?

Omega is currently in active development. While the core language is functional, some features may change. We recommend using it for learning and experimentation.

### How does Omega compare to other languages?

| Feature | Omega | Rust | Python | Go |
|---------|-------|------|--------|-----|
| Static typing | Yes | Yes | No | Yes |
| Memory safety | Yes | Yes | GC | GC |
| Pattern matching | Yes | Yes | Limited | No |
| Error handling | Result | Result | Exceptions | Error values |
| Concurrency | Async/await | Async/await | Async/await | Goroutines |
| Learning curve | Medium | High | Low | Medium |

## Installation

### How do I install Omega?

```bash
# From source
git clone https://github.com/omega-lang/omega.git
cd omega
cargo install --path .

# From package manager
cargo install omega-lang
```

### What are the system requirements?

- Operating system: Linux, macOS, Windows
- Memory: 512MB minimum
- Disk space: 100MB for installation

### How do I update Omega?

```bash
cargo install omega-lang --force
```

## Language Features

### How do I declare variables?

```omega
let x = 42           // Immutable
let mut y = 10       // Mutable
const PI = 3.14159   // Constant
```

### How do I define functions?

```omega
fn add(a: i64, b: i64) -> i64 {
    return a + b
}

// Short form
fn double(x) => x * 2

// Lambda
let triple = |x| x * 3
```

### How does error handling work?

```omega
// Result type
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0 {
        return Err("Division by zero")
    }
    return Ok(a / b)
}

// Try-catch
try {
    let result = risky_operation()
} catch (e) {
    println("Error: {}", e)
}
```

### How do I use pattern matching?

```omega
match value {
    0 => println("zero"),
    1..=9 => println("single digit"),
    (x, y) => println("tuple: {}, {}", x, y),
    [first, ...rest] => println("array with first: {}", first),
    _ => println("other"),
}
```

### How do I work with collections?

```omega
// Arrays
let arr = [1, 2, 3, 4, 5]
arr.push(6)
arr.map(|x| x * 2)
arr.filter(|x| x > 3)

// Maps
let map = {"key": "value"}
map["key"]
map.has("key")
map.keys()
```

### How do I handle concurrency?

```omega
// Threads
let handle = thread::spawn(|| {
    println("Hello from thread!")
})
handle.join()

// Channels
let (tx, rx) = channel::channel()
tx.send("Hello!")
let msg = rx.recv()

// Async
async fn fetch(url: String) -> String {
    let response = await http.get(url)
    return response.body()
}
```

## Tooling

### How do I run a program?

```bash
omega run program.omega
```

### How do I start the REPL?

```bash
omega repl
```

### How do I format code?

```bash
omega fmt program.omega
```

### How do I run tests?

```bash
omega test tests/
```

### How do I lint code?

```bash
omega lint program.omega
```

### Is there IDE support?

Yes! We provide:
- VS Code extension
- JetBrains plugin
- Language Server Protocol (LSP) implementation
- Vim/Neovim plugin

## Performance

### Is Omega fast?

Omega compiles to bytecode and runs on a virtual machine. Performance is comparable to other interpreted languages. For critical paths, you can use the JIT compiler for native code execution.

### How do I optimize my code?

1. Use appropriate data structures
2. Avoid unnecessary allocations
3. Use iterators instead of loops
4. Profile before optimizing
5. Consider using the JIT compiler

### Does Omega support parallelism?

Yes! Omega supports:
- Thread-based parallelism
- Async/await for I/O-bound tasks
- Channels for message passing
- Shared state with mutexes and locks

## Contributing

### How can I contribute?

1. Report bugs on GitHub
2. Submit pull requests
3. Improve documentation
4. Write tests
5. Share your projects

### Where can I get help?

- GitHub Issues
- Discord community
- Stack Overflow tag: omega-lang
- Documentation

### What's the development roadmap?

1. **Phase 1**: Core language (current)
2. **Phase 2**: Standard library
3. **Phase 3**: Tooling and ecosystem
4. **Phase 4**: Performance optimization
5. **Phase 5**: Production readiness
