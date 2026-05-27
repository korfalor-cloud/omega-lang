/// RGBA pixel representation.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Pixel {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub fn gray(value: u8) -> Self {
        Self { r: value, g: value, b: value, a: 255 }
    }

    pub fn black() -> Self {
        Self::gray(0)
    }

    pub fn white() -> Self {
        Self::gray(255)
    }

    pub fn red() -> Self {
        Self::rgb(255, 0, 0)
    }

    pub fn green() -> Self {
        Self::rgb(0, 255, 0)
    }

    pub fn blue() -> Self {
        Self::rgb(0, 0, 255)
    }

    pub fn transparent() -> Self {
        Self { r: 0, g: 0, b: 0, a: 0 }
    }

    pub fn luminance(&self) -> u8 {
        (0.299 * self.r as f64 + 0.587 * self.g as f64 + 0.114 * self.b as f64) as u8
    }

    pub fn to_hsl(&self) -> (f64, f64, f64) {
        let r = self.r as f64 / 255.0;
        let g = self.g as f64 / 255.0;
        let b = self.b as f64 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let l = (max + min) / 2.0;

        if max == min {
            return (0.0, 0.0, l);
        }

        let d = max - min;
        let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };

        let h = if max == r {
            ((g - b) / d + if g < b { 6.0 } else { 0.0 }) / 6.0
        } else if max == g {
            ((b - r) / d + 2.0) / 6.0
        } else {
            ((r - g) / d + 4.0) / 6.0
        };

        (h * 360.0, s, l)
    }

    pub fn from_hsl(h: f64, s: f64, l: f64) -> Self {
        if s == 0.0 {
            let v = (l * 255.0) as u8;
            return Self::gray(v);
        }

        let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
        let p = 2.0 * l - q;
        let h = h / 360.0;

        let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
        let g = hue_to_rgb(p, q, h);
        let b = hue_to_rgb(p, q, h - 1.0 / 3.0);

        Self::rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
    }

    pub fn blend(&self, other: &Pixel, alpha: f64) -> Pixel {
        let a = alpha.clamp(0.0, 1.0);
        Pixel::rgb(
            (self.r as f64 * (1.0 - a) + other.r as f64 * a) as u8,
            (self.g as f64 * (1.0 - a) + other.g as f64 * a) as u8,
            (self.b as f64 * (1.0 - a) + other.b as f64 * a) as u8,
        )
    }

    pub fn invert(&self) -> Pixel {
        Pixel::rgb(255 - self.r, 255 - self.g, 255 - self.b)
    }

    pub fn distance(&self, other: &Pixel) -> f64 {
        ((self.r as f64 - other.r as f64).powi(2)
            + (self.g as f64 - other.g as f64).powi(2)
            + (self.b as f64 - other.b as f64).powi(2))
        .sqrt()
    }
}

fn hue_to_rgb(p: f64, q: f64, t: f64) -> f64 {
    let t = if t < 0.0 { t + 1.0 } else if t > 1.0 { t - 1.0 } else { t };
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 1.0 / 2.0 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}

impl std::fmt::Display for Pixel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rgba({}, {}, {}, {})", self.r, self.g, self.b, self.a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_creation() {
        let p = Pixel::rgb(255, 128, 0);
        assert_eq!(p.r, 255);
        assert_eq!(p.g, 128);
        assert_eq!(p.b, 0);
        assert_eq!(p.a, 255);
    }

    #[test]
    fn test_luminance() {
        assert_eq!(Pixel::black().luminance(), 0);
        assert_eq!(Pixel::white().luminance(), 255);
    }

    #[test]
    fn test_hsl_roundtrip() {
        let p = Pixel::rgb(128, 64, 192);
        let (h, s, l) = p.to_hsl();
        let p2 = Pixel::from_hsl(h, s, l);
        // Allow some rounding error
        assert!((p.r as i32 - p2.r as i32).abs() <= 2);
    }

    #[test]
    fn test_blend() {
        let a = Pixel::black();
        let b = Pixel::white();
        let blended = a.blend(&b, 0.5);
        assert_eq!(blended.r, 127);
    }

    #[test]
    fn test_invert() {
        let p = Pixel::rgb(100, 150, 200);
        let inv = p.invert();
        assert_eq!(inv.r, 155);
        assert_eq!(inv.g, 105);
        assert_eq!(inv.b, 55);
    }
}
