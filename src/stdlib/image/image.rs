/// Image representation with pixel data.

use super::pixel::Pixel;

#[derive(Debug, Clone)]
pub struct Image {
    width: usize,
    height: usize,
    pixels: Vec<Pixel>,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![Pixel::black(); width * height],
        }
    }

    pub fn filled(width: usize, height: usize, color: Pixel) -> Self {
        Self {
            width,
            height,
            pixels: vec![color; width * height],
        }
    }

    pub fn from_pixels(width: usize, height: usize, pixels: Vec<Pixel>) -> Self {
        assert_eq!(pixels.len(), width * height);
        Self { width, height, pixels }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn get_pixel(&self, x: usize, y: usize) -> Pixel {
        assert!(x < self.width && y < self.height);
        self.pixels[y * self.width + x]
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, pixel: Pixel) {
        assert!(x < self.width && y < self.height);
        self.pixels[y * self.width + x] = pixel;
    }

    pub fn get_pixel_safe(&self, x: i32, y: i32) -> Option<Pixel> {
        if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
            Some(self.pixels[y as usize * self.width + x as usize])
        } else {
            None
        }
    }

    pub fn get_pixel_clamped(&self, x: i32, y: i32) -> Pixel {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        self.pixels[y * self.width + x]
    }

    pub fn fill(&mut self, color: Pixel) {
        self.pixels.fill(color);
    }

    pub fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: Pixel) {
        for dy in 0..h {
            for dx in 0..w {
                let px = x + dx;
                let py = y + dy;
                if px < self.width && py < self.height {
                    self.set_pixel(px, py, color);
                }
            }
        }
    }

    pub fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: Pixel) {
        // Bresenham's line algorithm
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut x = x0;
        let mut y = y0;

        loop {
            if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
                self.set_pixel(x as usize, y as usize, color);
            }
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    pub fn draw_circle(&mut self, cx: i32, cy: i32, radius: i32, color: Pixel) {
        // Midpoint circle algorithm
        let mut x = radius;
        let mut y = 0;
        let mut err = 1 - radius;

        while x >= y {
            self.plot_circle_points(cx, cy, x, y, color);
            y += 1;
            if err < 0 {
                err += 2 * y + 1;
            } else {
                x -= 1;
                err += 2 * (y - x) + 1;
            }
        }
    }

    fn plot_circle_points(&mut self, cx: i32, cy: i32, x: i32, y: i32, color: Pixel) {
        let points = [
            (cx + x, cy + y), (cx - x, cy + y),
            (cx + x, cy - y), (cx - x, cy - y),
            (cx + y, cy + x), (cx - y, cy + x),
            (cx + y, cy - x), (cx - y, cy - x),
        ];
        for (px, py) in &points {
            if *px >= 0 && *px < self.width as i32 && *py >= 0 && *py < self.height as i32 {
                self.set_pixel(*px as usize, *py as usize, color);
            }
        }
    }

    pub fn draw_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: Pixel) {
        for dx in 0..w {
            if x + dx < self.width {
                if y < self.height {
                    self.set_pixel(x + dx, y, color);
                }
                if y + h - 1 < self.height {
                    self.set_pixel(x + dx, y + h - 1, color);
                }
            }
        }
        for dy in 0..h {
            if y + dy < self.height {
                if x < self.width {
                    self.set_pixel(x, y + dy, color);
                }
                if x + w - 1 < self.width {
                    self.set_pixel(x + w - 1, y + dy, color);
                }
            }
        }
    }

    pub fn sub_image(&self, x: usize, y: usize, w: usize, h: usize) -> Image {
        let mut img = Image::new(w, h);
        for dy in 0..h {
            for dx in 0..w {
                if x + dx < self.width && y + dy < self.height {
                    img.set_pixel(dx, dy, self.get_pixel(x + dx, y + dy));
                }
            }
        }
        img
    }

    pub fn paste(&mut self, other: &Image, x: usize, y: usize) {
        for dy in 0..other.height() {
            for dx in 0..other.width() {
                let px = x + dx;
                let py = y + dy;
                if px < self.width && py < self.height {
                    self.set_pixel(px, py, other.get_pixel(dx, dy));
                }
            }
        }
    }

    pub fn pixels(&self) -> &[Pixel] {
        &self.pixels
    }

    pub fn pixels_mut(&mut self) -> &mut [Pixel] {
        &mut self.pixels
    }

    pub fn to_ppm(&self) -> String {
        let mut ppm = format!("P3\n{} {}\n255\n", self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let p = self.get_pixel(x, y);
                ppm.push_str(&format!("{} {} {} ", p.r, p.g, p.b));
            }
            ppm.push('\n');
        }
        ppm
    }

    pub fn histogram(&self) -> [u32; 256] {
        let mut hist = [0u32; 256];
        for pixel in &self.pixels {
            hist[pixel.luminance() as usize] += 1;
        }
        hist
    }

    pub fn mean_color(&self) -> Pixel {
        let (mut r, mut g, mut b) = (0u64, 0u64, 0u64);
        for pixel in &self.pixels {
            r += pixel.r as u64;
            g += pixel.g as u64;
            b += pixel.b as u64;
        }
        let n = self.pixels.len() as u64;
        Pixel::rgb((r / n) as u8, (g / n) as u8, (b / n) as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_creation() {
        let img = Image::new(100, 100);
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 100);
    }

    #[test]
    fn test_pixel_access() {
        let mut img = Image::new(10, 10);
        img.set_pixel(5, 5, Pixel::red());
        assert_eq!(img.get_pixel(5, 5), Pixel::red());
    }

    #[test]
    fn test_fill_rect() {
        let mut img = Image::new(10, 10);
        img.fill_rect(2, 2, 3, 3, Pixel::blue());
        assert_eq!(img.get_pixel(3, 3), Pixel::blue());
        assert_eq!(img.get_pixel(1, 1), Pixel::black());
    }

    #[test]
    fn test_draw_line() {
        let mut img = Image::new(10, 10);
        img.draw_line(0, 0, 9, 9, Pixel::white());
        assert_eq!(img.get_pixel(0, 0), Pixel::white());
        assert_eq!(img.get_pixel(5, 5), Pixel::white());
    }

    #[test]
    fn test_draw_circle() {
        let mut img = Image::new(20, 20);
        img.draw_circle(10, 10, 5, Pixel::green());
        assert_eq!(img.get_pixel(10, 5), Pixel::green());
        assert_eq!(img.get_pixel(15, 10), Pixel::green());
    }

    #[test]
    fn test_sub_image() {
        let mut img = Image::new(10, 10);
        img.fill(Pixel::red());
        let sub = img.sub_image(2, 2, 3, 3);
        assert_eq!(sub.width(), 3);
        assert_eq!(sub.get_pixel(0, 0), Pixel::red());
    }

    #[test]
    fn test_histogram() {
        let mut img = Image::new(10, 10);
        img.fill(Pixel::white());
        let hist = img.histogram();
        assert_eq!(hist[255], 100);
    }

    #[test]
    fn test_ppm() {
        let img = Image::new(2, 2);
        let ppm = img.to_ppm();
        assert!(ppm.starts_with("P3\n2 2\n255\n"));
    }
}
