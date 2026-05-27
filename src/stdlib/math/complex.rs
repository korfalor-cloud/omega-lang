#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn zero() -> Self {
        Self { real: 0.0, imag: 0.0 }
    }

    pub fn one() -> Self {
        Self { real: 1.0, imag: 0.0 }
    }

    pub fn i() -> Self {
        Self { real: 0.0, imag: 1.0 }
    }

    pub fn from_polar(magnitude: f64, angle: f64) -> Self {
        Self {
            real: magnitude * angle.cos(),
            imag: magnitude * angle.sin(),
        }
    }

    pub fn add(&self, other: &Complex) -> Complex {
        Complex {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }

    pub fn sub(&self, other: &Complex) -> Complex {
        Complex {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }

    pub fn mul(&self, other: &Complex) -> Complex {
        Complex {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }

    pub fn div(&self, other: &Complex) -> Complex {
        let denom = other.real * other.real + other.imag * other.imag;
        Complex {
            real: (self.real * other.real + self.imag * other.imag) / denom,
            imag: (self.imag * other.real - self.real * other.imag) / denom,
        }
    }

    pub fn conjugate(&self) -> Complex {
        Complex { real: self.real, imag: -self.imag }
    }

    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    pub fn angle(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    pub fn normalize(&self) -> Complex {
        let mag = self.magnitude();
        if mag == 0.0 {
            *self
        } else {
            Complex {
                real: self.real / mag,
                imag: self.imag / mag,
            }
        }
    }

    pub fn exp(&self) -> Complex {
        let r = self.real.exp();
        Complex {
            real: r * self.imag.cos(),
            imag: r * self.imag.sin(),
        }
    }

    pub fn ln(&self) -> Complex {
        Complex {
            real: self.magnitude().ln(),
            imag: self.angle(),
        }
    }

    pub fn pow(&self, exponent: &Complex) -> Complex {
        if self.magnitude() == 0.0 {
            return Complex::zero();
        }
        (self.ln().mul(exponent)).exp()
    }

    pub fn sqrt(&self) -> Complex {
        let r = self.magnitude();
        let theta = self.angle();
        Complex::from_polar(r.sqrt(), theta / 2.0)
    }

    pub fn sin(&self) -> Complex {
        Complex {
            real: self.real.sin() * self.imag.cosh(),
            imag: self.real.cos() * self.imag.sinh(),
        }
    }

    pub fn cos(&self) -> Complex {
        Complex {
            real: self.real.cos() * self.imag.cosh(),
            imag: -self.real.sin() * self.imag.sinh(),
        }
    }

    pub fn tan(&self) -> Complex {
        self.sin().div(&self.cos())
    }

    pub fn is_zero(&self) -> bool {
        self.real == 0.0 && self.imag == 0.0
    }

    pub fn is_real(&self) -> bool {
        self.imag == 0.0
    }

    pub fn is_imaginary(&self) -> bool {
        self.real == 0.0
    }
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.imag >= 0.0 {
            write!(f, "{}+{}i", self.real, self.imag)
        } else {
            write!(f, "{}{}i", self.real, self.imag)
        }
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        (self.real - other.real).abs() < 1e-10 && (self.imag - other.imag).abs() < 1e-10
    }
}
