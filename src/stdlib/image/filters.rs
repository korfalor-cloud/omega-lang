/// Image filters and convolutions.

use super::image::Image;
use super::pixel::Pixel;

pub fn grayscale(image: &Image) -> Image {
    let mut result = Image::new(image.width(), image.height());
    for y in 0..image.height() {
        for x in 0..image.width() {
            let p = image.get_pixel(x, y);
            result.set_pixel(x, y, Pixel::gray(p.luminance()));
        }
    }
    result
}

pub fn invert(image: &Image) -> Image {
    let mut result = Image::new(image.width(), image.height());
    for y in 0..image.height() {
        for x in 0..image.width() {
            result.set_pixel(x, y, image.get_pixel(x, y).invert());
        }
    }
    result
}

pub fn brightness(image: &Image, amount: i32) -> Image {
    let mut result = Image::new(image.width(), image.height());
    for y in 0..image.height() {
        for x in 0..image.width() {
            let p = image.get_pixel(x, y);
            result.set_pixel(x, y, Pixel::rgb(
                (p.r as i32 + amount).clamp(0, 255) as u8,
                (p.g as i32 + amount).clamp(0, 255) as u8,
                (p.b as i32 + amount).clamp(0, 255) as u8,
            ));
        }
    }
    result
}

pub fn contrast(image: &Image, factor: f64) -> Image {
    let mut result = Image::new(image.width(), image.height());
    let factor = factor.clamp(0.0, 2.0);
    for y in 0..image.height() {
        for x in 0..image.width() {
            let p = image.get_pixel(x, y);
            result.set_pixel(x, y, Pixel::rgb(
                ((p.r as f64 - 128.0) * factor + 128.0).clamp(0.0, 255.0) as u8,
                ((p.g as f64 - 128.0) * factor + 128.0).clamp(0.0, 255.0) as u8,
                ((p.b as f64 - 128.0) * factor + 128.0).clamp(0.0, 255.0) as u8,
            ));
        }
    }
    result
}

pub fn threshold(image: &Image, threshold: u8) -> Image {
    let mut result = Image::new(image.width(), image.height());
    for y in 0..image.height() {
        for x in 0..image.width() {
            let lum = image.get_pixel(x, y).luminance();
            result.set_pixel(x, y, if lum >= threshold { Pixel::white() } else { Pixel::black() });
        }
    }
    result
}

pub fn sepia(image: &Image) -> Image {
    let mut result = Image::new(image.width(), image.height());
    for y in 0..image.height() {
        for x in 0..image.width() {
            let p = image.get_pixel(x, y);
            let r = p.r as f64;
            let g = p.g as f64;
            let b = p.b as f64;
            result.set_pixel(x, y, Pixel::rgb(
                (0.393 * r + 0.769 * g + 0.189 * b).min(255.0) as u8,
                (0.349 * r + 0.686 * g + 0.168 * b).min(255.0) as u8,
                (0.272 * r + 0.534 * g + 0.131 * b).min(255.0) as u8,
            ));
        }
    }
    result
}

pub fn convolution(image: &Image, kernel: &[Vec<f64>]) -> Image {
    let k_size = kernel.len();
    assert!(k_size % 2 == 1 && kernel[0].len() == k_size);
    let k_half = k_size / 2;

    let mut result = Image::new(image.width(), image.height());

    for y in 0..image.height() {
        for x in 0..image.width() {
            let (mut r_sum, mut g_sum, mut b_sum) = (0.0, 0.0, 0.0);

            for ky in 0..k_size {
                for kx in 0..k_size {
                    let px = x as i32 + kx as i32 - k_half as i32;
                    let py = y as i32 + ky as i32 - k_half as i32;
                    let p = image.get_pixel_clamped(px, py);
                    let weight = kernel[ky][kx];
                    r_sum += p.r as f64 * weight;
                    g_sum += p.g as f64 * weight;
                    b_sum += p.b as f64 * weight;
                }
            }

            result.set_pixel(x, y, Pixel::rgb(
                r_sum.clamp(0.0, 255.0) as u8,
                g_sum.clamp(0.0, 255.0) as u8,
                b_sum.clamp(0.0, 255.0) as u8,
            ));
        }
    }
    result
}

pub fn blur(image: &Image) -> Image {
    let kernel = vec![
        vec![1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        vec![1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        vec![1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
    ];
    convolution(image, &kernel)
}

pub fn gaussian_blur(image: &Image) -> Image {
    let kernel = vec![
        vec![1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
        vec![2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
        vec![1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
    ];
    convolution(image, &kernel)
}

pub fn sharpen(image: &Image) -> Image {
    let kernel = vec![
        vec![0.0, -1.0, 0.0],
        vec![-1.0, 5.0, -1.0],
        vec![0.0, -1.0, 0.0],
    ];
    convolution(image, &kernel)
}

pub fn edge_detect(image: &Image) -> Image {
    let kernel = vec![
        vec![-1.0, -1.0, -1.0],
        vec![-1.0, 8.0, -1.0],
        vec![-1.0, -1.0, -1.0],
    ];
    convolution(image, &kernel)
}

pub fn emboss(image: &Image) -> Image {
    let kernel = vec![
        vec![-2.0, -1.0, 0.0],
        vec![-1.0, 1.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    convolution(image, &kernel)
}

pub fn sobel_x(image: &Image) -> Image {
    let kernel = vec![
        vec![-1.0, 0.0, 1.0],
        vec![-2.0, 0.0, 2.0],
        vec![-1.0, 0.0, 1.0],
    ];
    convolution(image, &kernel)
}

pub fn sobel_y(image: &Image) -> Image {
    let kernel = vec![
        vec![-1.0, -2.0, -1.0],
        vec![0.0, 0.0, 0.0],
        vec![1.0, 2.0, 1.0],
    ];
    convolution(image, &kernel)
}

pub fn median_filter(image: &Image, radius: usize) -> Image {
    let mut result = Image::new(image.width(), image.height());

    for y in 0..image.height() {
        for x in 0..image.width() {
            let mut r_vals = Vec::new();
            let mut g_vals = Vec::new();
            let mut b_vals = Vec::new();

            for dy in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    let p = image.get_pixel_clamped(x as i32 + dx, y as i32 + dy);
                    r_vals.push(p.r);
                    g_vals.push(p.g);
                    b_vals.push(p.b);
                }
            }

            r_vals.sort();
            g_vals.sort();
            b_vals.sort();

            let mid = r_vals.len() / 2;
            result.set_pixel(x, y, Pixel::rgb(r_vals[mid], g_vals[mid], b_vals[mid]));
        }
    }
    result
}

pub fn color_filter(image: &Image, r_mul: f64, g_mul: f64, b_mul: f64) -> Image {
    let mut result = Image::new(image.width(), image.height());
    for y in 0..image.height() {
        for x in 0..image.width() {
            let p = image.get_pixel(x, y);
            result.set_pixel(x, y, Pixel::rgb(
                (p.r as f64 * r_mul).min(255.0) as u8,
                (p.g as f64 * g_mul).min(255.0) as u8,
                (p.b as f64 * b_mul).min(255.0) as u8,
            ));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image() -> Image {
        let mut img = Image::new(10, 10);
        img.fill(Pixel::rgb(128, 128, 128));
        img.set_pixel(5, 5, Pixel::red());
        img
    }

    #[test]
    fn test_grayscale() {
        let img = test_image();
        let gray = grayscale(&img);
        assert_eq!(gray.get_pixel(5, 5), Pixel::gray(76));
    }

    #[test]
    fn test_invert_filter() {
        let img = test_image();
        let inv = invert(&img);
        assert_eq!(inv.get_pixel(0, 0), Pixel::rgb(127, 127, 127));
    }

    #[test]
    fn test_brightness() {
        let img = test_image();
        let bright = brightness(&img, 50);
        assert!(bright.get_pixel(0, 0).r > 128);
    }

    #[test]
    fn test_threshold() {
        let img = test_image();
        let binary = threshold(&img, 100);
        assert_eq!(binary.get_pixel(0, 0), Pixel::white());
    }

    #[test]
    fn test_blur() {
        let img = test_image();
        let blurred = blur(&img);
        assert_eq!(blurred.width(), 10);
    }

    #[test]
    fn test_sharpen() {
        let img = test_image();
        let sharpened = sharpen(&img);
        assert_eq!(sharpened.width(), 10);
    }

    #[test]
    fn test_edge_detect() {
        let img = test_image();
        let edges = edge_detect(&img);
        assert_eq!(edges.width(), 10);
    }

    #[test]
    fn test_median_filter() {
        let img = test_image();
        let filtered = median_filter(&img, 1);
        assert_eq!(filtered.width(), 10);
    }
}
