/// Image transformations: resize, rotate, flip, crop.

use super::image::Image;
use super::pixel::Pixel;

pub fn resize_nearest(image: &Image, new_width: usize, new_height: usize) -> Image {
    let mut result = Image::new(new_width, new_height);
    let x_ratio = image.width() as f64 / new_width as f64;
    let y_ratio = image.height() as f64 / new_height as f64;

    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = (x as f64 * x_ratio) as usize;
            let src_y = (y as f64 * y_ratio) as usize;
            result.set_pixel(x, y, image.get_pixel(
                src_x.min(image.width() - 1),
                src_y.min(image.height() - 1),
            ));
        }
    }
    result
}

pub fn resize_bilinear(image: &Image, new_width: usize, new_height: usize) -> Image {
    let mut result = Image::new(new_width, new_height);
    let x_ratio = (image.width() - 1) as f64 / (new_width - 1).max(1) as f64;
    let y_ratio = (image.height() - 1) as f64 / (new_height - 1).max(1) as f64;

    for y in 0..new_height {
        for x in 0..new_width {
            let gx = x as f64 * x_ratio;
            let gy = y as f64 * y_ratio;

            let x_low = gx.floor() as usize;
            let y_low = gy.floor() as usize;
            let x_high = (x_low + 1).min(image.width() - 1);
            let y_high = (y_low + 1).min(image.height() - 1);

            let x_frac = gx - gx.floor();
            let y_frac = gy - gy.floor();

            let p00 = image.get_pixel(x_low, y_low);
            let p10 = image.get_pixel(x_high, y_low);
            let p01 = image.get_pixel(x_low, y_high);
            let p11 = image.get_pixel(x_high, y_high);

            let r = bilinear_interp(p00.r, p10.r, p01.r, p11.r, x_frac, y_frac);
            let g = bilinear_interp(p00.g, p10.g, p01.g, p11.g, x_frac, y_frac);
            let b = bilinear_interp(p00.b, p10.b, p01.b, p11.b, x_frac, y_frac);

            result.set_pixel(x, y, Pixel::rgb(r, g, b));
        }
    }
    result
}

fn bilinear_interp(v00: u8, v10: u8, v01: u8, v11: u8, fx: f64, fy: f64) -> u8 {
    let top = v00 as f64 * (1.0 - fx) + v10 as f64 * fx;
    let bottom = v01 as f64 * (1.0 - fx) + v11 as f64 * fx;
    (top * (1.0 - fy) + bottom * fy) as u8
}

pub fn flip_horizontal(image: &Image) -> Image {
    let mut result = Image::new(image.width(), image.height());
    let w = image.width();
    for y in 0..image.height() {
        for x in 0..w {
            result.set_pixel(x, y, image.get_pixel(w - 1 - x, y));
        }
    }
    result
}

pub fn flip_vertical(image: &Image) -> Image {
    let mut result = Image::new(image.width(), image.height());
    let h = image.height();
    for y in 0..h {
        for x in 0..image.width() {
            result.set_pixel(x, y, image.get_pixel(x, h - 1 - y));
        }
    }
    result
}

pub fn rotate_90(image: &Image) -> Image {
    let w = image.width();
    let h = image.height();
    let mut result = Image::new(h, w);
    for y in 0..h {
        for x in 0..w {
            result.set_pixel(h - 1 - y, x, image.get_pixel(x, y));
        }
    }
    result
}

pub fn rotate_180(image: &Image) -> Image {
    let w = image.width();
    let h = image.height();
    let mut result = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            result.set_pixel(w - 1 - x, h - 1 - y, image.get_pixel(x, y));
        }
    }
    result
}

pub fn rotate_270(image: &Image) -> Image {
    let w = image.width();
    let h = image.height();
    let mut result = Image::new(h, w);
    for y in 0..h {
        for x in 0..w {
            result.set_pixel(y, w - 1 - x, image.get_pixel(x, y));
        }
    }
    result
}

pub fn rotate(image: &Image, angle_degrees: f64) -> Image {
    let w = image.width();
    let h = image.height();
    let mut result = Image::filled(w, h, Pixel::black());

    let angle = angle_degrees.to_radians();
    let cos = angle.cos();
    let sin = angle.sin();
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;

    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;

            let src_x = (cos * dx + sin * dy + cx) as i32;
            let src_y = (-sin * dx + cos * dy + cy) as i32;

            if let Some(pixel) = image.get_pixel_safe(src_x, src_y) {
                result.set_pixel(x, y, pixel);
            }
        }
    }
    result
}

pub fn crop(image: &Image, x: usize, y: usize, width: usize, height: usize) -> Image {
    image.sub_image(x, y, width, height)
}

pub fn pad(image: &Image, top: usize, right: usize, bottom: usize, left: usize, color: Pixel) -> Image {
    let new_width = left + image.width() + right;
    let new_height = top + image.height() + bottom;
    let mut result = Image::filled(new_width, new_height, color);
    result.paste(image, left, top);
    result
}

pub fn tile(image: &Image, tiles_x: usize, tiles_y: usize) -> Image {
    let mut result = Image::new(image.width() * tiles_x, image.height() * tiles_y);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            result.paste(image, tx * image.width(), ty * image.height());
        }
    }
    result
}

pub fn mosaic(images: &[Image]) -> Image {
    if images.is_empty() {
        return Image::new(0, 0);
    }

    let max_width = images.iter().map(|i| i.width()).max().unwrap();
    let max_height = images.iter().map(|i| i.height()).max().unwrap();
    let cols = (images.len() as f64).sqrt().ceil() as usize;
    let rows = (images.len() + cols - 1) / cols;

    let mut result = Image::new(max_width * cols, max_height * rows);
    for (idx, img) in images.iter().enumerate() {
        let col = idx % cols;
        let row = idx / cols;
        result.paste(img, col * max_width, row * max_height);
    }
    result
}

pub fn blend_images(a: &Image, b: &Image, alpha: f64) -> Image {
    assert_eq!(a.width(), b.width());
    assert_eq!(a.height(), b.height());

    let mut result = Image::new(a.width(), a.height());
    for y in 0..a.height() {
        for x in 0..a.width() {
            result.set_pixel(x, y, a.get_pixel(x, y).blend(&b.get_pixel(x, y), alpha));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image() -> Image {
        let mut img = Image::new(10, 10);
        img.fill(Pixel::red());
        img.fill_rect(5, 5, 5, 5, Pixel::blue());
        img
    }

    #[test]
    fn test_resize_nearest() {
        let img = test_image();
        let resized = resize_nearest(&img, 20, 20);
        assert_eq!(resized.width(), 20);
        assert_eq!(resized.height(), 20);
    }

    #[test]
    fn test_resize_bilinear() {
        let img = test_image();
        let resized = resize_bilinear(&img, 20, 20);
        assert_eq!(resized.width(), 20);
    }

    #[test]
    fn test_flip_horizontal() {
        let mut img = Image::new(3, 1);
        img.set_pixel(0, 0, Pixel::red());
        img.set_pixel(2, 0, Pixel::blue());

        let flipped = flip_horizontal(&img);
        assert_eq!(flipped.get_pixel(0, 0), Pixel::blue());
        assert_eq!(flipped.get_pixel(2, 0), Pixel::red());
    }

    #[test]
    fn test_rotate_90() {
        let mut img = Image::new(2, 3);
        img.set_pixel(0, 0, Pixel::red());
        img.set_pixel(1, 2, Pixel::blue());

        let rotated = rotate_90(&img);
        assert_eq!(rotated.width(), 3);
        assert_eq!(rotated.height(), 2);
    }

    #[test]
    fn test_crop() {
        let img = test_image();
        let cropped = crop(&img, 2, 2, 5, 5);
        assert_eq!(cropped.width(), 5);
        assert_eq!(cropped.height(), 5);
    }

    #[test]
    fn test_pad() {
        let img = test_image();
        let padded = pad(&img, 5, 5, 5, 5, Pixel::green());
        assert_eq!(padded.width(), 20);
        assert_eq!(padded.height(), 20);
    }

    #[test]
    fn test_tile() {
        let img = Image::new(5, 5);
        let tiled = tile(&img, 3, 2);
        assert_eq!(tiled.width(), 15);
        assert_eq!(tiled.height(), 10);
    }

    #[test]
    fn test_blend_images() {
        let a = Image::filled(5, 5, Pixel::black());
        let b = Image::filled(5, 5, Pixel::white());
        let blended = blend_images(&a, &b, 0.5);
        assert_eq!(blended.get_pixel(0, 0), Pixel::rgb(127, 127, 127));
    }
}
