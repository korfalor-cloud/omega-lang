use std::fmt;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Rational {
    pub numerator: i64,
    pub denominator: i64,
}

impl Rational {
    pub fn new(numerator: i64, denominator: i64) -> Self {
        if denominator == 0 {
            panic!("Denominator cannot be zero");
        }
        let mut r = Self { numerator, denominator };
        r.simplify();
        r
    }

    pub fn zero() -> Self {
        Self { numerator: 0, denominator: 1 }
    }

    pub fn one() -> Self {
        Self { numerator: 1, denominator: 1 }
    }

    pub fn from_integer(value: i64) -> Self {
        Self { numerator: value, denominator: 1 }
    }

    fn simplify(&mut self) {
        if self.numerator == 0 {
            self.denominator = 1;
            return;
        }
        let gcd = gcd(self.numerator.abs(), self.denominator.abs());
        self.numerator /= gcd;
        self.denominator /= gcd;
        if self.denominator < 0 {
            self.numerator = -self.numerator;
            self.denominator = -self.denominator;
        }
    }

    pub fn add(&self, other: &Rational) -> Rational {
        let num = self.numerator * other.denominator + other.numerator * self.denominator;
        let den = self.denominator * other.denominator;
        Rational::new(num, den)
    }

    pub fn sub(&self, other: &Rational) -> Rational {
        let num = self.numerator * other.denominator - other.numerator * self.denominator;
        let den = self.denominator * other.denominator;
        Rational::new(num, den)
    }

    pub fn mul(&self, other: &Rational) -> Rational {
        Rational::new(self.numerator * other.numerator, self.denominator * other.denominator)
    }

    pub fn div(&self, other: &Rational) -> Rational {
        Rational::new(self.numerator * other.denominator, self.denominator * other.numerator)
    }

    pub fn neg(&self) -> Rational {
        Rational::new(-self.numerator, self.denominator)
    }

    pub fn abs(&self) -> Rational {
        Rational::new(self.numerator.abs(), self.denominator.abs())
    }

    pub fn reciprocal(&self) -> Rational {
        Rational::new(self.denominator, self.numerator)
    }

    pub fn to_f64(&self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    pub fn is_zero(&self) -> bool {
        self.numerator == 0
    }

    pub fn is_positive(&self) -> bool {
        self.numerator > 0
    }

    pub fn is_negative(&self) -> bool {
        self.numerator < 0
    }

    pub fn is_integer(&self) -> bool {
        self.denominator == 1
    }

    pub fn floor(&self) -> i64 {
        self.numerator / self.denominator
    }

    pub fn ceil(&self) -> i64 {
        (self.numerator + self.denominator - 1) / self.denominator
    }

    pub fn pow(&self, exp: i64) -> Rational {
        if exp == 0 {
            return Rational::one();
        }
        if exp < 0 {
            return self.reciprocal().pow(-exp);
        }
        Rational::new(
            self.numerator.pow(exp as u32),
            self.denominator.pow(exp as u32),
        )
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let lhs = self.numerator * other.denominator;
        let rhs = other.numerator * self.denominator;
        lhs.cmp(&rhs)
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}
