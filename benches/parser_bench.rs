use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use omega_lang::lexer::scanner::Scanner;
use omega_lang::parser::parser::Parser;

fn parse(source: &str) {
    let mut parser = Parser::new(source);
    parser.parse().unwrap();
}

fn bench_parse_small(c: &mut Criterion) {
    c.bench_function("parse_small", |b| {
        b.iter(|| parse(black_box("let x = 42")))
    });
}

fn bench_parse_function(c: &mut Criterion) {
    let source = r#"
fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}
"#;
    c.bench_function("parse_function", |b| {
        b.iter(|| parse(black_box(source)))
    });
}

fn bench_parse_struct(c: &mut Criterion) {
    let source = r#"
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl Point {
    fn new(x: f64, y: f64, z: f64) -> Point {
        return Point { x, y, z }
    }

    fn distance(&self, other: Point) -> f64 {
        let dx = self.x - other.x
        let dy = self.y - other.y
        let dz = self.z - other.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
}
"#;
    c.bench_function("parse_struct", |b| {
        b.iter(|| parse(black_box(source)))
    });
}

fn bench_parse_enum(c: &mut Criterion) {
    let source = r#"
enum Result<T, E> {
    Ok(T),
    Err(E),
}

impl<T, E> Result<T, E> {
    fn is_ok(&self) -> bool {
        match self {
            Result::Ok(_) => true,
            Result::Err(_) => false,
        }
    }

    fn unwrap(&self) -> T {
        match self {
            Result::Ok(value) => return value,
            Result::Err(e) => panic("unwrap on Err: {}", e),
        }
    }
}
"#;
    c.bench_function("parse_enum", |b| {
        b.iter(|| parse(black_box(source)))
    });
}

fn bench_parse_match(c: &mut Criterion) {
    let source = r#"
match value {
    0 => println("zero"),
    1..=9 => println("single digit"),
    10..=99 => println("double digit"),
    100..=999 => println("triple digit"),
    _ => println("large number"),
}
"#;
    c.bench_function("parse_match", |b| {
        b.iter(|| parse(black_box(source)))
    });
}

fn bench_parse_expressions(c: &mut Criterion) {
    let expressions = vec![
        "1 + 2 * 3 - 4 / 5 % 6",
        "a && b || c && !d || e",
        "(x > 0) && (y < 100) || (z == 0)",
        "arr[0].field.method(arg1, arg2, arg3)",
        "fn(a, b, c) { return a + b + c }",
    ];

    for expr in expressions {
        c.bench_with_input(
            BenchmarkId::new("parse_expression", expr.len()),
            expr,
            |b, expr| {
                b.iter(|| parse(black_box(expr)))
            },
        );
    }
}

fn bench_parse_large_program(c: &mut Criterion) {
    let source = r#"
mod math {
    pub fn add(a: i64, b: i64) -> i64 {
        return a + b
    }

    pub fn sub(a: i64, b: i64) -> i64 {
        return a - b
    }

    pub fn mul(a: i64, b: i64) -> i64 {
        return a * b
    }

    pub fn div(a: f64, b: f64) -> Result<f64, String> {
        if b == 0 {
            return Err("Division by zero")
        }
        return Ok(a / b)
    }
}

mod utils {
    pub fn factorial(n: i64) -> i64 {
        if n <= 1 {
            return 1
        }
        return n * factorial(n - 1)
    }

    pub fn is_palindrome(s: String) -> bool {
        let reversed = s.chars().rev().collect()
        return s == reversed
    }
}

fn main() {
    let result = math::add(10, 20)
    println("10 + 20 = {}", result)

    let fact = utils::factorial(10)
    println("10! = {}", fact)

    for i in 0..20 {
        if i % 2 == 0 {
            println("{} is even", i)
        } else {
            println("{} is odd", i)
        }
    }
}
"#;
    c.bench_function("parse_large_program", |b| {
        b.iter(|| parse(black_box(source)))
    });
}

criterion_group!(
    benches,
    bench_parse_small,
    bench_parse_function,
    bench_parse_struct,
    bench_parse_enum,
    bench_parse_match,
    bench_parse_expressions,
    bench_parse_large_program,
);
criterion_main!(benches);
