# Omega Programming Language

A modern, safe, and fast programming language with a focus on developer experience.

## Features

- **Static typing** with type inference
- **First-class functions** and closures
- **Pattern matching** and destructuring
- **Structs and enums** with methods
- **Traits** for polymorphism
- **Modules** for code organization
- **Error handling** with Result type
- **Concurrency** with threads and channels
- **Async/await** for asynchronous programming
- **Memory safety** without garbage collection (optional GC)
- **Rich standard library**

## Syntax

### Variables

```omega
let x = 42
let mut y = 10
y += 5

const PI = 3.14159
```

### Functions

```omega
fn add(a: i64, b: i64) -> i64 {
    return a + b
}

fn greet(name) {
    println("Hello, {}!", name)
}

// Lambda
let double = |x| x * 2
```

### Control Flow

```omega
if x > 0 {
    println("positive")
} else if x < 0 {
    println("negative")
} else {
    println("zero")
}

while x > 0 {
    x -= 1
}

for i in 0..10 {
    println(i)
}

match value {
    0 => println("zero"),
    1..=9 => println("single digit"),
    _ => println("large"),
}
```

### Data Structures

```omega
// Arrays
let arr = [1, 2, 3, 4, 5]

// Maps
let map = {"key": "value", "count": 42}

// Tuples
let tuple = (1, "hello", true)

// Structs
struct Point {
    x: f64,
    y: f64,
}

// Enums
enum Option<T> {
    Some(T),
    None,
}
```

### Error Handling

```omega
try {
    let result = risky_operation()
} catch (e) {
    println("Error: {}", e)
} finally {
    cleanup()
}

// Result type
fn divide(a, b) -> Result<f64, String> {
    if b == 0 {
        return Err("Division by zero")
    }
    return Ok(a / b)
}
```

### Concurrency

```omega
use std::thread
use std::channel

let (tx, rx) = channel::channel()

thread::spawn(|| {
    tx.send("Hello from thread!")
})

let message = rx.recv()
```

### Modules

```omega
mod math {
    pub fn add(a, b) {
        return a + b
    }
}

use math::add
```

## Standard Library

### Collections
- `Vec` - Dynamic array
- `Map` - Hash map
- `Set` - Hash set
- `Deque` - Double-ended queue
- `Heap` - Binary heap

### I/O
- `File` - File operations
- `Stdin/Stdout/Stderr` - Standard streams
- `Buffer` - In-memory buffer

### Math
- Trigonometric functions
- Statistics
- Linear algebra
- Complex numbers
- Rational numbers

### String
- Case conversion
- Trimming and padding
- Splitting and joining
- Pattern matching

### Network
- TCP/UDP sockets
- HTTP client
- URL parsing
- DNS resolution

### Concurrency
- Threads
- Channels
- Mutexes
- Read-write locks
- Semaphores
- Atomics

### Serialization
- JSON
- YAML
- TOML
- CSV

### Testing
- Test framework
- Assertions
- Benchmarks

## CLI Usage

```bash
# Run a program
omega run program.omega

# Start REPL
omega repl

# Compile
omega compile program.omega -o output

# Format
omega fmt program.omega

# Test
omega test tests/
```
