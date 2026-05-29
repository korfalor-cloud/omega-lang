/// Advanced image processing: color spaces, morphology, feature detection, segmentation.

use super::image::Image;
use super::pixel::Pixel;

// ---------------------------------------------------------------------------
// Color-space representations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hsv { pub h: f64, pub s: f64, pub v: f64 }

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lab { pub l: f64, pub a: f64, pub b: f64 }

// ---------------------------------------------------------------------------
// Color-space conversions
// ---------------------------------------------------------------------------

pub fn rgb_to_hsv(p: Pixel) -> Hsv {
    let (r, g, b) = (p.r as f64 / 255.0, p.g as f64 / 255.0, p.b as f64 / 255.0);
    let (max, min) = (r.max(g).max(b), r.min(g).min(b));
    let d = max - min;
    let h = if d == 0.0 { 0.0 } else if max == r { 60.0 * (((g - b) / d) % 6.0) }
        else if max == g { 60.0 * ((b - r) / d + 2.0) }
        else { 60.0 * ((r - g) / d + 4.0) };
    let h = if h < 0.0 { h + 360.0 } else { h };
    Hsv { h, s: if max == 0.0 { 0.0 } else { d / max }, v: max }
}

pub fn hsv_to_pixel(hsv: Hsv) -> Pixel {
    let c = hsv.v * hsv.s;
    let x = c * (1.0 - ((hsv.h / 60.0) % 2.0 - 1.0).abs());
    let m = hsv.v - c;
    let (r, g, b) = match hsv.h as u32 / 60 {
        0 => (c, x, 0.0), 1 => (x, c, 0.0), 2 => (0.0, c, x),
        3 => (0.0, x, c), 4 => (x, 0.0, c), _ => (c, 0.0, x),
    };
    Pixel::rgb(((r + m) * 255.0) as u8, ((g + m) * 255.0) as u8, ((b + m) * 255.0) as u8)
}

pub fn rgb_to_lab(p: Pixel) -> Lab {
    let lin = |c: u8| { let v = c as f64 / 255.0; if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } };
    let (r, g, b) = (lin(p.r), lin(p.g), lin(p.b));
    let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
    let f = |t: f64| if t > 0.008856 { t.powf(1.0 / 3.0) } else { 7.787 * t + 16.0 / 116.0 };
    Lab { l: 116.0 * f(y) - 16.0, a: 500.0 * (f(x / 0.95047) - f(y)), b: 200.0 * (f(y) - f(z / 1.08883)) }
}

pub fn lab_to_pixel(lab: Lab) -> Pixel {
    let fy = (lab.l + 16.0) / 116.0;
    let finv = |t: f64| if t.powi(3) > 0.008856 { t.powi(3) } else { (t - 16.0 / 116.0) / 7.787 };
    let (x, y, z) = (finv(lab.a / 500.0 + fy) * 0.95047, finv(fy), finv(fy - lab.b / 200.0) * 1.08883);
    let (rl, gl, bl) = (x * 3.2404542 + y * -1.5371385 + z * -0.4985314,
                        x * -0.9692660 + y * 1.8760108 + z * 0.0415560,
                        x * 0.0556434 + y * -0.2040259 + z * 1.0572252);
    let gamma = |v: f64| { let v = v.clamp(0.0, 1.0); if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 } };
    Pixel::rgb((gamma(rl) * 255.0) as u8, (gamma(gl) * 255.0) as u8, (gamma(bl) * 255.0) as u8)
}

// ---------------------------------------------------------------------------
// Advanced filters
// ---------------------------------------------------------------------------

/// Bilateral filter: smooths while preserving edges.
pub fn bilateral_filter(image: &Image, spatial_sigma: f64, range_sigma: f64) -> Image {
    let (w, h) = (image.width(), image.height());
    let radius = (spatial_sigma * 2.0).ceil() as i32;
    let (s2, r2) = (2.0 * spatial_sigma * spatial_sigma, 2.0 * range_sigma * range_sigma);
    let mut result = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let center = image.get_pixel(x, y);
            let (mut wr, mut wg, mut wb, mut sr, mut sg, mut sb) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let p = image.get_pixel_clamped(x as i32 + dx, y as i32 + dy);
                    let sp = (-((dx * dx + dy * dy) as f64) / s2).exp();
                    let dr = (p.r as f64 - center.r as f64).powi(2);
                    let dg = (p.g as f64 - center.g as f64).powi(2);
                    let db = (p.b as f64 - center.b as f64).powi(2);
                    let (cr, cg, cb) = (sp * (-dr / r2).exp(), sp * (-dg / r2).exp(), sp * (-db / r2).exp());
                    wr += cr; wg += cg; wb += cb;
                    sr += cr * p.r as f64; sg += cg * p.g as f64; sb += cb * p.b as f64;
                }
            }
            result.set_pixel(x, y, Pixel::rgb((sr / wr).clamp(0.0, 255.0) as u8,
                                               (sg / wg).clamp(0.0, 255.0) as u8,
                                               (sb / wb).clamp(0.0, 255.0) as u8));
        }
    }
    result
}

/// Unsharp mask: sharpen by subtracting a blurred copy.
pub fn unsharp_mask(image: &Image, amount: f64) -> Image {
    use super::filters::gaussian_blur;
    let blurred = gaussian_blur(image);
    let mut result = Image::new(image.width(), image.height());
    for y in 0..image.height() {
        for x in 0..image.width() {
            let (o, b) = (image.get_pixel(x, y), blurred.get_pixel(x, y));
            result.set_pixel(x, y, Pixel::rgb(
                ((o.r as f64 + amount * (o.r as f64 - b.r as f64)).clamp(0.0, 255.0)) as u8,
                ((o.g as f64 + amount * (o.g as f64 - b.g as f64)).clamp(0.0, 255.0)) as u8,
                ((o.b as f64 + amount * (o.b as f64 - b.b as f64)).clamp(0.0, 255.0)) as u8,
            ));
        }
    }
    result
}

/// Sobel magnitude edge strength map.
pub fn sobel_magnitude(image: &Image) -> Image {
    use super::filters::{sobel_x, sobel_y};
    let (gx, gy) = (sobel_x(image), sobel_y(image));
    let mut result = Image::new(image.width(), image.height());
    for y in 0..image.height() {
        for x in 0..image.width() {
            let (px, py) = (gx.get_pixel(x, y), gy.get_pixel(x, y));
            let mag = ((px.r as f64).powi(2) + (py.r as f64).powi(2)).sqrt().min(255.0) as u8;
            result.set_pixel(x, y, Pixel::gray(mag));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Morphological operations
// ---------------------------------------------------------------------------

fn morph_op(image: &Image, radius: usize, keep_max: bool) -> Image {
    let (w, h) = (image.width(), image.height());
    let mut result = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut best = image.get_pixel(x, y);
            for dy in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    let p = image.get_pixel_clamped(x as i32 + dx, y as i32 + dy);
                    if (keep_max && p.luminance() > best.luminance()) ||
                       (!keep_max && p.luminance() < best.luminance()) { best = p; }
                }
            }
            result.set_pixel(x, y, best);
        }
    }
    result
}

pub fn dilate(image: &Image, radius: usize) -> Image { morph_op(image, radius, true) }
pub fn erode(image: &Image, radius: usize) -> Image { morph_op(image, radius, false) }
pub fn opening(image: &Image, radius: usize) -> Image { dilate(&erode(image, radius), radius) }
pub fn closing(image: &Image, radius: usize) -> Image { erode(&dilate(image, radius), radius) }

pub fn morphological_gradient(image: &Image, radius: usize) -> Image {
    let (d, e) = (dilate(image, radius), erode(image, radius));
    let mut result = Image::new(image.width(), image.height());
    for y in 0..image.height() {
        for x in 0..image.width() {
            let diff = d.get_pixel(x, y).luminance() as i32 - e.get_pixel(x, y).luminance() as i32;
            result.set_pixel(x, y, Pixel::gray(diff.clamp(0, 255) as u8));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Harris corner detector
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corner { pub x: usize, pub y: usize, pub response: f64 }

pub fn harris_corners(image: &Image, k: f64, threshold: f64) -> Vec<Corner> {
    use super::filters::{sobel_x, sobel_y};
    let (gx, gy) = (sobel_x(image), sobel_y(image));
    let (w, h) = (image.width(), image.height());
    let mut ixx = vec![0.0f64; w * h];
    let mut iyy = vec![0.0f64; w * h];
    let mut ixy = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (gx.get_pixel(x, y).luminance() as f64 - 128.0,
                            gy.get_pixel(x, y).luminance() as f64 - 128.0);
            let idx = y * w + x;
            ixx[idx] = dx * dx; iyy[idx] = dy * dy; ixy[idx] = dx * dy;
        }
    }
    let win = |arr: &[f64], cx: usize, cy: usize| -> f64 {
        let mut s = 0.0;
        for dy in -1i32..=1 { for dx in -1i32..=1 {
            s += arr[(cy as i32 + dy).clamp(0, h as i32 - 1) as usize * w
                   + (cx as i32 + dx).clamp(0, w as i32 - 1) as usize];
        }}
        s
    };
    let mut corners = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let (sx, sy, sxy) = (win(&ixx, x, y), win(&iyy, x, y), win(&ixy, x, y));
            let r = sx * sy - sxy * sxy - k * (sx + sy).powi(2);
            if r > threshold { corners.push(Corner { x, y, response: r }); }
        }
    }
    corners.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
    corners
}

// ---------------------------------------------------------------------------
// K-means color segmentation
// ---------------------------------------------------------------------------

pub fn kmeans_segment(image: &Image, k: usize, max_iter: usize) -> Image {
    assert!(k > 0);
    let pixels = image.pixels();
    let n = pixels.len();
    let mut centroids: Vec<[f64; 3]> = (0..k)
        .map(|i| { let p = pixels[(i * n / k).min(n - 1)]; [p.r as f64, p.g as f64, p.b as f64] })
        .collect();
    let mut assignments = vec![0usize; n];
    for _ in 0..max_iter {
        let mut changed = false;
        for (i, p) in pixels.iter().enumerate() {
            let v = [p.r as f64, p.g as f64, p.b as f64];
            let mut best = 0;
            let mut best_d = f64::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d = (v[0] - c[0]).powi(2) + (v[1] - c[1]).powi(2) + (v[2] - c[2]).powi(2);
                if d < best_d { best_d = d; best = ci; }
            }
            if assignments[i] != best { assignments[i] = best; changed = true; }
        }
        if !changed { break; }
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0usize; k];
        for (i, p) in pixels.iter().enumerate() {
            let c = assignments[i];
            sums[c][0] += p.r as f64; sums[c][1] += p.g as f64; sums[c][2] += p.b as f64;
            counts[c] += 1;
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                centroids[ci] = [sums[ci][0] / counts[ci] as f64,
                                 sums[ci][1] / counts[ci] as f64,
                                 sums[ci][2] / counts[ci] as f64];
            }
        }
    }
    let mut result = Image::new(image.width(), image.height());
    for (i, &c) in assignments.iter().enumerate() {
        let cent = centroids[c];
        result.pixels_mut()[i] = Pixel::rgb(cent[0] as u8, cent[1] as u8, cent[2] as u8);
    }
    result
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_hsv_roundtrip() {
        let orig = Pixel::rgb(100, 180, 50);
        let back = hsv_to_pixel(rgb_to_hsv(orig));
        assert!((back.r as i32 - orig.r as i32).abs() <= 1);
        assert!((back.g as i32 - orig.g as i32).abs() <= 1);
        assert!((back.b as i32 - orig.b as i32).abs() <= 1);
    }

    #[test]
    fn test_hsv_black() {
        let hsv = rgb_to_hsv(Pixel::black());
        assert_eq!(hsv.v, 0.0);
        assert_eq!(hsv.s, 0.0);
    }

    #[test]
    fn test_rgb_lab_roundtrip() {
        let orig = Pixel::rgb(120, 60, 200);
        let back = lab_to_pixel(rgb_to_lab(orig));
        assert!((back.r as i32 - orig.r as i32).abs() <= 2);
        assert!((back.g as i32 - orig.g as i32).abs() <= 2);
        assert!((back.b as i32 - orig.b as i32).abs() <= 2);
    }

    #[test]
    fn test_lab_white() {
        let lab = rgb_to_lab(Pixel::white());
        assert!((lab.l - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_bilateral_filter_preserves_size() {
        let img = Image::filled(8, 8, Pixel::gray(100));
        let out = bilateral_filter(&img, 1.0, 30.0);
        assert_eq!(out.width(), 8);
        assert_eq!(out.height(), 8);
    }

    #[test]
    fn test_unsharp_mask() {
        let mut img = Image::new(10, 10);
        img.fill(Pixel::gray(128));
        img.set_pixel(5, 5, Pixel::white());
        let sharp = unsharp_mask(&img, 1.5);
        assert!(sharp.get_pixel(5, 5).luminance() >= 128);
    }

    #[test]
    fn test_sobel_magnitude_on_edge() {
        let mut img = Image::new(10, 10);
        for y in 0..10 { for x in 0..10 {
            img.set_pixel(x, y, if x < 5 { Pixel::black() } else { Pixel::white() });
        }}
        let mag = sobel_magnitude(&img);
        assert!(mag.get_pixel(5, 5).luminance() > 0);
    }

    #[test]
    fn test_dilate_expands_bright() {
        let mut img = Image::new(10, 10);
        img.set_pixel(5, 5, Pixel::white());
        assert!(dilate(&img, 1).get_pixel(6, 5).luminance() > 0);
    }

    #[test]
    fn test_erode_shrinks_bright() {
        let mut img = Image::new(10, 10);
        img.fill(Pixel::white());
        img.set_pixel(5, 5, Pixel::black());
        assert!(erode(&img, 1).get_pixel(6, 5).luminance() < 255);
    }

    #[test]
    fn test_opening_removes_noise() {
        let mut img = Image::new(10, 10);
        img.set_pixel(5, 5, Pixel::white());
        assert_eq!(opening(&img, 1).get_pixel(5, 5), Pixel::black());
    }

    #[test]
    fn test_closing_fills_holes() {
        let mut img = Image::new(10, 10);
        img.fill(Pixel::white());
        img.set_pixel(5, 5, Pixel::black());
        assert_eq!(closing(&img, 1).get_pixel(5, 5), Pixel::white());
    }

    #[test]
    fn test_morphological_gradient() {
        let mut img = Image::new(10, 10);
        for y in 3..7 { for x in 3..7 { img.set_pixel(x, y, Pixel::white()); } }
        assert!(morphological_gradient(&img, 1).get_pixel(3, 5).luminance() > 0);
    }

    #[test]
    fn test_harris_corners_finds_corner() {
        let mut img = Image::new(20, 20);
        for y in 0..10 { for x in 0..20 { img.set_pixel(x, y, Pixel::white()); } }
        for y in 10..20 { for x in 0..10 { img.set_pixel(x, y, Pixel::white()); } }
        let corners = harris_corners(&img, 0.04, 1e6);
        let found = corners.iter().any(|c| (c.x as i32 - 10).unsigned_abs() <= 3
                                          && (c.y as i32 - 10).unsigned_abs() <= 3);
        assert!(found, "expected corner near (10,10), got {:?}", corners);
    }

    #[test]
    fn test_harris_blank() {
        assert!(harris_corners(&Image::filled(20, 20, Pixel::gray(128)), 0.04, 1e4).is_empty());
    }

    #[test]
    fn test_kmeans_two_colors() {
        let mut img = Image::new(10, 10);
        for y in 0..10 { for x in 0..10 {
            img.set_pixel(x, y, if x < 5 { Pixel::black() } else { Pixel::white() });
        }}
        let seg = kmeans_segment(&img, 2, 20);
        assert!(seg.get_pixel(0, 5).distance(&seg.get_pixel(9, 5)) > 50.0);
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let img = Image::filled(5, 5, Pixel::rgb(80, 80, 80));
        let seg = kmeans_segment(&img, 1, 5);
        for y in 0..5 { for x in 0..5 {
            assert!(seg.get_pixel(x, y).distance(&Pixel::rgb(80, 80, 80)) < 5.0);
        }}
    }
}
