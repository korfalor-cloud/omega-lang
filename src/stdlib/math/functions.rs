use crate::errors::{OmegaError, OmegaResult};
use std::f64::consts;

pub const PI: f64 = consts::PI;
pub const E: f64 = consts::E;
pub const TAU: f64 = consts::TAU;
pub const SQRT_2: f64 = consts::SQRT_2;
pub const LN_2: f64 = consts::LN_2;
pub const LN_10: f64 = consts::LN_10;
pub const LOG2_E: f64 = consts::LOG2_E;
pub const LOG10_E: f64 = consts::LOG10_E;

pub fn abs(x: f64) -> f64 {
    x.abs()
}

pub fn floor(x: f64) -> f64 {
    x.floor()
}

pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

pub fn round(x: f64) -> f64 {
    x.round()
}

pub fn trunc(x: f64) -> f64 {
    x.trunc()
}

pub fn fract(x: f64) -> f64 {
    x.fract()
}

pub fn sqrt(x: f64) -> OmegaResult<f64> {
    if x < 0.0 {
        Err(OmegaError::ValueError {
            message: "Cannot take square root of negative number".to_string(),
        })
    } else {
        Ok(x.sqrt())
    }
}

pub fn cbrt(x: f64) -> f64 {
    x.cbrt()
}

pub fn pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

pub fn exp(x: f64) -> f64 {
    x.exp()
}

pub fn exp2(x: f64) -> f64 {
    x.exp2()
}

pub fn ln(x: f64) -> OmegaResult<f64> {
    if x <= 0.0 {
        Err(OmegaError::ValueError {
            message: "Cannot take logarithm of non-positive number".to_string(),
        })
    } else {
        Ok(x.ln())
    }
}

pub fn log2(x: f64) -> OmegaResult<f64> {
    if x <= 0.0 {
        Err(OmegaError::ValueError {
            message: "Cannot take logarithm of non-positive number".to_string(),
        })
    } else {
        Ok(x.log2())
    }
}

pub fn log10(x: f64) -> OmegaResult<f64> {
    if x <= 0.0 {
        Err(OmegaError::ValueError {
            message: "Cannot take logarithm of non-positive number".to_string(),
        })
    } else {
        Ok(x.log10())
    }
}

pub fn log(x: f64, base: f64) -> OmegaResult<f64> {
    if x <= 0.0 || base <= 0.0 || base == 1.0 {
        Err(OmegaError::ValueError {
            message: "Invalid arguments for logarithm".to_string(),
        })
    } else {
        Ok(x.log(base))
    }
}

pub fn sin(x: f64) -> f64 {
    x.sin()
}

pub fn cos(x: f64) -> f64 {
    x.cos()
}

pub fn tan(x: f64) -> f64 {
    x.tan()
}

pub fn asin(x: f64) -> OmegaResult<f64> {
    if x < -1.0 || x > 1.0 {
        Err(OmegaError::ValueError {
            message: "asin requires argument in [-1, 1]".to_string(),
        })
    } else {
        Ok(x.asin())
    }
}

pub fn acos(x: f64) -> OmegaResult<f64> {
    if x < -1.0 || x > 1.0 {
        Err(OmegaError::ValueError {
            message: "acos requires argument in [-1, 1]".to_string(),
        })
    } else {
        Ok(x.acos())
    }
}

pub fn atan(x: f64) -> f64 {
    x.atan()
}

pub fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

pub fn sinh(x: f64) -> f64 {
    x.sinh()
}

pub fn cosh(x: f64) -> f64 {
    x.cosh()
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn asinh(x: f64) -> f64 {
    x.asinh()
}

pub fn acosh(x: f64) -> OmegaResult<f64> {
    if x < 1.0 {
        Err(OmegaError::ValueError {
            message: "acosh requires argument >= 1".to_string(),
        })
    } else {
        Ok(x.acosh())
    }
}

pub fn atanh(x: f64) -> OmegaResult<f64> {
    if x <= -1.0 || x >= 1.0 {
        Err(OmegaError::ValueError {
            message: "atanh requires argument in (-1, 1)".to_string(),
        })
    } else {
        Ok(x.atanh())
    }
}

pub fn degrees(x: f64) -> f64 {
    x * 180.0 / PI
}

pub fn radians(x: f64) -> f64 {
    x * PI / 180.0
}

pub fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

pub fn clamp(x: f64, min: f64, max: f64) -> f64 {
    x.max(min).min(max)
}

pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn sign(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
}

pub fn is_nan(x: f64) -> bool {
    x.is_nan()
}

pub fn is_infinite(x: f64) -> bool {
    x.is_infinite()
}

pub fn is_finite(x: f64) -> bool {
    x.is_finite()
}

pub fn is_normal(x: f64) -> bool {
    x.is_normal()
}

pub fn gcd(a: i64, b: i64) -> i64 {
    let mut a = a.abs();
    let mut b = b.abs();
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

pub fn lcm(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        0
    } else {
        (a.abs() / gcd(a, b)) * b.abs()
    }
}

pub fn factorial(n: u64) -> OmegaResult<u64> {
    if n > 20 {
        return Err(OmegaError::OverflowError {
            message: "Factorial overflow for n > 20".to_string(),
        });
    }
    let mut result = 1u64;
    for i in 2..=n {
        result *= i;
    }
    Ok(result)
}

pub fn fibonacci(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let mut a = 0u64;
    let mut b = 1u64;
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}

pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

pub fn primes_up_to(n: u64) -> Vec<u64> {
    if n < 2 {
        return Vec::new();
    }
    let mut sieve = vec![true; (n + 1) as usize];
    sieve[0] = false;
    sieve[1] = false;
    let mut i = 2;
    while i * i <= n {
        if sieve[i as usize] {
            let mut j = i * i;
            while j <= n {
                sieve[j as usize] = false;
                j += i;
            }
        }
        i += 1;
    }
    sieve.iter().enumerate().filter(|(_, &is_p)| is_p).map(|(i, _)| i as u64).collect()
}

pub fn prime_factors(n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut n = n;
    let mut d = 2;
    while d * d <= n {
        while n % d == 0 {
            factors.push(d);
            n /= d;
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_math() {
        assert_eq!(abs(-5.0), 5.0);
        assert_eq!(floor(3.7), 3.0);
        assert_eq!(ceil(3.2), 4.0);
        assert_eq!(round(3.5), 4.0);
    }

    #[test]
    fn test_trig() {
        assert!((sin(PI / 2.0) - 1.0).abs() < 1e-10);
        assert!((cos(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gcd_lcm() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(lcm(12, 8), 24);
    }

    #[test]
    fn test_primes() {
        assert!(is_prime(7));
        assert!(!is_prime(4));
        assert_eq!(primes_up_to(10), vec![2, 3, 5, 7]);
    }
}
