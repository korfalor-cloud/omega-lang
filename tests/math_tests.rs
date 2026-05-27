use omega_lang::stdlib::math::functions::*;
use omega_lang::stdlib::math::statistics::*;

#[test]
fn test_abs() {
    assert_eq!(omega_abs(-42), 42);
    assert_eq!(omega_abs(42), 42);
    assert_eq!(omega_abs(0), 0);
}

#[test]
fn test_min() {
    assert_eq!(omega_min(1, 2), 1);
    assert_eq!(omega_min(2, 1), 1);
    assert_eq!(omega_min(5, 5), 5);
}

#[test]
fn test_max() {
    assert_eq!(omega_max(1, 2), 2);
    assert_eq!(omega_max(2, 1), 2);
    assert_eq!(omega_max(5, 5), 5);
}

#[test]
fn test_clamp() {
    assert_eq!(omega_clamp(5, 0, 10), 5);
    assert_eq!(omega_clamp(-5, 0, 10), 0);
    assert_eq!(omega_clamp(15, 0, 10), 10);
}

#[test]
fn test_sqrt() {
    assert!((omega_sqrt(4.0) - 2.0).abs() < 0.0001);
    assert!((omega_sqrt(9.0) - 3.0).abs() < 0.0001);
    assert!((omega_sqrt(0.0) - 0.0).abs() < 0.0001);
}

#[test]
fn test_pow() {
    assert!((omega_pow(2.0, 10.0) - 1024.0).abs() < 0.0001);
    assert!((omega_pow(3.0, 3.0) - 27.0).abs() < 0.0001);
    assert!((omega_pow(5.0, 0.0) - 1.0).abs() < 0.0001);
}

#[test]
fn test_log() {
    assert!((omega_ln(std::f64::consts::E) - 1.0).abs() < 0.0001);
    assert!((omega_log2(8.0) - 3.0).abs() < 0.0001);
    assert!((omega_log10(100.0) - 2.0).abs() < 0.0001);
}

#[test]
fn test_trig() {
    assert!((omega_sin(0.0) - 0.0).abs() < 0.0001);
    assert!((omega_cos(0.0) - 1.0).abs() < 0.0001);
    assert!((omega_sin(std::f64::consts::PI / 2.0) - 1.0).abs() < 0.0001);
}

#[test]
fn test_gcd() {
    assert_eq!(omega_gcd(12, 8), 4);
    assert_eq!(omega_gcd(15, 10), 5);
    assert_eq!(omega_gcd(7, 13), 1);
}

#[test]
fn test_lcm() {
    assert_eq!(omega_lcm(4, 6), 12);
    assert_eq!(omega_lcm(3, 5), 15);
    assert_eq!(omega_lcm(6, 8), 24);
}

#[test]
fn test_factorial() {
    assert_eq!(omega_factorial(0), 1);
    assert_eq!(omega_factorial(1), 1);
    assert_eq!(omega_factorial(5), 120);
    assert_eq!(omega_factorial(10), 3628800);
}

#[test]
fn test_fibonacci() {
    assert_eq!(omega_fibonacci(0), 0);
    assert_eq!(omega_fibonacci(1), 1);
    assert_eq!(omega_fibonacci(10), 55);
    assert_eq!(omega_fibonacci(20), 6765);
}

#[test]
fn test_is_prime() {
    assert!(!omega_is_prime(0));
    assert!(!omega_is_prime(1));
    assert!(omega_is_prime(2));
    assert!(omega_is_prime(3));
    assert!(!omega_is_prime(4));
    assert!(omega_is_prime(17));
    assert!(!omega_is_prime(15));
}

#[test]
fn test_mean() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!((omega_mean(&data) - 3.0).abs() < 0.0001);
}

#[test]
fn test_median() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!((omega_median(&data) - 3.0).abs() < 0.0001);
}

#[test]
fn test_mode() {
    let data = vec![1, 2, 2, 3, 3, 3];
    assert_eq!(omega_mode(&data), Some(3));
}

#[test]
fn test_variance() {
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let var = omega_variance(&data);
    assert!((var - 4.0).abs() < 0.0001);
}

#[test]
fn test_std_dev() {
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let std = omega_std_dev(&data);
    assert!((std - 2.0).abs() < 0.0001);
}

#[test]
fn test_correlation() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let corr = omega_correlation(&x, &y);
    assert!((corr - 1.0).abs() < 0.0001);
}

#[test]
fn test_linear_regression() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let (slope, intercept) = omega_linear_regression(&x, &y);
    assert!((slope - 2.0).abs() < 0.0001);
    assert!((intercept - 0.0).abs() < 0.0001);
}

#[test]
fn test_z_score() {
    let z = omega_z_score(7.0, 5.0, 2.0);
    assert!((z - 1.0).abs() < 0.0001);
}

#[test]
fn test_moving_average() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let ma = omega_moving_average(&data, 3);
    assert_eq!(ma.len(), 3);
    assert!((ma[0] - 2.0).abs() < 0.0001);
    assert!((ma[1] - 3.0).abs() < 0.0001);
    assert!((ma[2] - 4.0).abs() < 0.0001);
}
