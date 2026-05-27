use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use omega_lang::lexer::scanner::Scanner;

fn bench_scan_small(c: &mut Criterion) {
    c.bench_function("scan_small", |b| {
        b.iter(|| {
            let mut scanner = Scanner::new("let x = 42");
            scanner.scan().unwrap()
        })
    });
}

fn bench_scan_medium(c: &mut Criterion) {
    let source = r#"
fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}
"#;
    c.bench_function("scan_medium", |b| {
        b.iter(|| {
            let mut scanner = Scanner::new(source);
            scanner.scan().unwrap()
        })
    });
}

fn bench_scan_large(c: &mut Criterion) {
    let source = r#"
fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false
    }
    for i in 2..n {
        if n % i == 0 {
            return false
        }
    }
    return true
}

fn quicksort(arr: [i64]) -> [i64] {
    if arr.len() <= 1 {
        return arr
    }
    let pivot = arr[0]
    let left = []
    let right = []
    for i in 1..arr.len() {
        if arr[i] < pivot {
            left.push(arr[i])
        } else {
            right.push(arr[i])
        }
    }
    return [...quicksort(left), pivot, ...quicksort(right)]
}

fn main() {
    for i in 0..100 {
        if is_prime(i) {
            println("{} is prime", i)
        }
    }
    let arr = [5, 3, 8, 1, 9, 2, 7, 4, 6]
    println("Sorted: {}", quicksort(arr))
}
"#;
    c.bench_function("scan_large", |b| {
        b.iter(|| {
            let mut scanner = Scanner::new(source);
            scanner.scan().unwrap()
        })
    });
}

fn bench_scan_expressions(c: &mut Criterion) {
    let expressions = vec![
        "1 + 2 * 3 - 4 / 5",
        "a && b || c && !d",
        "(x > 0) && (y < 100)",
        "arr[i].field.method()",
        "fn(a, b, c) => a + b + c",
    ];

    for expr in expressions {
        c.bench_with_input(
            BenchmarkId::new("scan_expression", expr.len()),
            expr,
            |b, expr| {
                b.iter(|| {
                    let mut scanner = Scanner::new(expr);
                    scanner.scan().unwrap()
                })
            },
        );
    }
}

fn bench_scan_strings(c: &mut Criterion) {
    let strings = vec![
        r#""hello""#,
        r#""hello world with spaces""#,
        r#""escaped\nnewlines\tand\ttabs""#,
        r#""very long string with lots of characters that goes on and on""#,
    ];

    for s in strings {
        c.bench_with_input(
            BenchmarkId::new("scan_string", s.len()),
            s,
            |b, s| {
                b.iter(|| {
                    let mut scanner = Scanner::new(s);
                    scanner.scan().unwrap()
                })
            },
        );
    }
}

fn bench_scan_numbers(c: &mut Criterion) {
    let numbers = vec![
        "42",
        "3.14159",
        "0xFF",
        "0b101010",
        "0o777",
        "1000000000",
    ];

    for num in numbers {
        c.bench_with_input(
            BenchmarkId::new("scan_number", num),
            num,
            |b, num| {
                b.iter(|| {
                    let mut scanner = Scanner::new(num);
                    scanner.scan().unwrap()
                })
            },
        );
    }
}

criterion_group!(
    benches,
    bench_scan_small,
    bench_scan_medium,
    bench_scan_large,
    bench_scan_expressions,
    bench_scan_strings,
    bench_scan_numbers,
);
criterion_main!(benches);
