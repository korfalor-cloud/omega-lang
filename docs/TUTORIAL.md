# Omega Tutorial

## Hello World

Let's start with the classic Hello World program:

```omega
println("Hello, World!")
```

Save this as `hello.omega` and run it:

```bash
omega run hello.omega
```

## Variables

Variables are declared using `let`:

```omega
let x = 42
let name = "Omega"
let pi = 3.14159
let is_active = true
```

Mutable variables use `mut`:

```omega
let mut count = 0
count += 1
println(count)  // 1
```

Constants use `const`:

```omega
const PI = 3.14159
const MAX_SIZE = 100
```

## Data Types

### Integers

```omega
let small: i8 = 127
let medium: i32 = 1000000
let big: i64 = 9999999999
let unsigned: u64 = 100
```

### Floats

```omega
let single: f32 = 3.14
let double: f64 = 3.14159265358979
```

### Strings

```omega
let greeting = "Hello"
let message = "Hello, {}!"
println(message, "World")
```

### Booleans

```omega
let is_true = true
let is_false = false
```

### Arrays

```omega
let numbers = [1, 2, 3, 4, 5]
let first = numbers[0]
let length = numbers.len()
```

### Maps

```omega
let person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}
let name = person["name"]
```

### Tuples

```omega
let point = (10, 20)
let (x, y) = point
```

## Functions

### Basic Functions

```omega
fn add(a: i64, b: i64) -> i64 {
    return a + b
}

let result = add(3, 4)
```

### Short Functions

```omega
fn square(x: i64) -> i64 {
    return x * x
}
```

### Lambda Functions

```omega
let double = |x| x * 2
let result = double(5)  // 10
```

### Higher-Order Functions

```omega
fn apply(f: fn(i64) -> i64, x: i64) -> i64 {
    return f(x)
}

let result = apply(|x| x * 2, 5)  // 10
```

## Control Flow

### If/Else

```omega
let x = 42

if x > 0 {
    println("positive")
} else if x < 0 {
    println("negative")
} else {
    println("zero")
}
```

### While Loops

```omega
let mut i = 0
while i < 10 {
    println(i)
    i += 1
}
```

### For Loops

```omega
for i in 0..10 {
    println(i)
}

// With step
for i in (0..10).step(2) {
    println(i)
}
```

### Match Expressions

```omega
let x = 42

match x {
    0 => println("zero"),
    1..=9 => println("single digit"),
    10..=99 => println("double digit"),
    _ => println("large number"),
}
```

## Structs

### Defining Structs

```omega
struct Point {
    x: f64,
    y: f64,
}
```

### Creating Instances

```omega
let p = Point { x: 10.0, y: 20.0 }
println(p.x)  // 10.0
```

### Methods

```omega
impl Point {
    fn new(x: f64, y: f64) -> Point {
        return Point { x, y }
    }

    fn distance(&self, other: Point) -> f64 {
        let dx = self.x - other.x
        let dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)
    }
}
```

## Enums

### Defining Enums

```omega
enum Color {
    Red,
    Green,
    Blue,
    Custom(i64, i64, i64),
}
```

### Using Enums

```omega
let color = Color::Custom(255, 0, 0)

match color {
    Color::Red => println("red"),
    Color::Green => println("green"),
    Color::Blue => println("blue"),
    Color::Custom(r, g, b) => println("rgb({},{},{})", r, g, b),
}
```

## Error Handling

### Result Type

```omega
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0 {
        return Err("Division by zero")
    }
    return Ok(a / b)
}

let result = divide(10, 3)
match result {
    Ok(value) => println("Result: {}", value),
    Err(e) => println("Error: {}", e),
}
```

### Try/Catch

```omega
try {
    let data = risky_operation()
    println(data)
} catch (e) {
    println("Error: {}", e)
} finally {
    cleanup()
}
```

## Collections

### Vec

```omega
let mut numbers = vec![1, 2, 3]
numbers.push(4)
numbers.pop()
numbers.sort()

for n in numbers {
    println(n)
}
```

### Map

```omega
let mut scores = {}
scores["Alice"] = 95
scores["Bob"] = 87

for (name, score) in scores {
    println("{}: {}", name, score)
}
```

### Set

```omega
let mut unique = set![1, 2, 3]
unique.add(4)
unique.remove(1)
```

## Concurrency

### Threads

```omega
let handle = thread::spawn(|| {
    println("Hello from thread!")
})
handle.join()
```

### Channels

```omega
let (tx, rx) = channel::channel()

thread::spawn(|| {
    tx.send("Hello!")
})

let message = rx.recv()
println(message)
```

### Async/Await

```omega
async fn fetch_data(url: String) -> String {
    let response = await http.get(url)
    return response.body()
}

let data = await fetch_data("https://api.example.com")
```

## Modules

### Defining Modules

```omega
mod math {
    pub fn add(a: i64, b: i64) -> i64 {
        return a + b
    }

    pub fn sub(a: i64, b: i64) -> i64 {
        return a - b
    }
}
```

### Using Modules

```omega
use math::add
use math::sub

let result = add(10, 5)  // 15
```

## Traits

### Defining Traits

```omega
trait Printable {
    fn print(&self)
}
```

### Implementing Traits

```omega
impl Printable for Point {
    fn print(&self) {
        println("({}, {})", self.x, self.y)
    }
}
```

## Generics

### Generic Functions

```omega
fn identity<T>(x: T) -> T {
    return x
}

let num = identity(42)
let str = identity("hello")
```

### Generic Structs

```omega
struct Container<T> {
    value: T,
}

impl<T> Container<T> {
    fn new(value: T) -> Container<T> {
        return Container { value }
    }

    fn get(&self) -> T {
        return self.value
    }
}
```

## Pattern Matching

### Basic Patterns

```omega
match value {
    0 => println("zero"),
    1 => println("one"),
    _ => println("other"),
}
```

### Destructuring

```omega
match point {
    (0, 0) => println("origin"),
    (x, 0) => println("on x-axis at {}", x),
    (0, y) => println("on y-axis at {}", y),
    (x, y) => println("at ({}, {})", x, y),
}
```

### Guard Clauses

```omega
match x {
    n if n > 0 => println("positive"),
    n if n < 0 => println("negative"),
    _ => println("zero"),
}
```

## Next Steps

- Read the [Language Reference](LANGUAGE.md)
- Explore the [Standard Library](STDLIB.md)
- Check out the [Examples](../examples/)
- Join the community on Discord
