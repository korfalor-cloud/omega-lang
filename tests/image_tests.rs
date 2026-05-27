use omega_lang::stdlib::image::pixel::Pixel;
use omega_lang::stdlib::image::image::Image;
use omega_lang::stdlib::image::filters::*;
use omega_lang::stdlib::image::transforms::*;

#[test]
fn test_pixel_creation() {
    let p = Pixel::rgb(255, 128, 0);
    assert_eq!(p.r, 255);
    assert_eq!(p.g, 128);
    assert_eq!(p.b, 0);
    assert_eq!(p.a, 255);
}

#[test]
fn test_pixel_luminance() {
    assert_eq!(Pixel::black().luminance(), 0);
    assert_eq!(Pixel::white().luminance(), 255);
}

#[test]
fn test_pixel_hsl_roundtrip() {
    let p = Pixel::rgb(128, 64, 192);
    let (h, s, l) = p.to_hsl();
    let p2 = Pixel::from_hsl(h, s, l);
    assert!((p.r as i32 - p2.r as i32).abs() <= 2);
}

#[test]
fn test_image_creation() {
    let img = Image::new(100, 100);
    assert_eq!(img.width(), 100);
    assert_eq!(img.height(), 100);
}

#[test]
fn test_image_pixel_access() {
    let mut img = Image::new(10, 10);
    img.set_pixel(5, 5, Pixel::red());
    assert_eq!(img.get_pixel(5, 5), Pixel::red());
}

#[test]
fn test_image_fill() {
    let mut img = Image::new(10, 10);
    img.fill(Pixel::blue());
    assert_eq!(img.get_pixel(0, 0), Pixel::blue());
    assert_eq!(img.get_pixel(9, 9), Pixel::blue());
}

#[test]
fn test_image_fill_rect() {
    let mut img = Image::new(10, 10);
    img.fill_rect(2, 2, 3, 3, Pixel::green());
    assert_eq!(img.get_pixel(3, 3), Pixel::green());
    assert_eq!(img.get_pixel(1, 1), Pixel::black());
}

#[test]
fn test_image_draw_line() {
    let mut img = Image::new(10, 10);
    img.draw_line(0, 0, 9, 9, Pixel::white());
    assert_eq!(img.get_pixel(0, 0), Pixel::white());
    assert_eq!(img.get_pixel(5, 5), Pixel::white());
}

#[test]
fn test_image_draw_circle() {
    let mut img = Image::new(20, 20);
    img.draw_circle(10, 10, 5, Pixel::green());
    assert_eq!(img.get_pixel(10, 5), Pixel::green());
    assert_eq!(img.get_pixel(15, 10), Pixel::green());
}

#[test]
fn test_grayscale_filter() {
    let mut img = Image::new(10, 10);
    img.fill(Pixel::rgb(128, 128, 128));
    let gray = grayscale(&img);
    assert_eq!(gray.get_pixel(0, 0), Pixel::gray(128));
}

#[test]
fn test_invert_filter() {
    let mut img = Image::new(10, 10);
    img.fill(Pixel::rgb(100, 150, 200));
    let inv = invert(&img);
    assert_eq!(inv.get_pixel(0, 0), Pixel::rgb(155, 105, 55));
}

#[test]
fn test_brightness_filter() {
    let mut img = Image::new(10, 10);
    img.fill(Pixel::rgb(100, 100, 100));
    let bright = brightness(&img, 50);
    assert_eq!(bright.get_pixel(0, 0).r, 150);
}

#[test]
fn test_threshold_filter() {
    let mut img = Image::new(10, 10);
    img.fill(Pixel::rgb(200, 200, 200));
    let binary = threshold(&img, 128);
    assert_eq!(binary.get_pixel(0, 0), Pixel::white());
}

#[test]
fn test_blur_filter() {
    let mut img = Image::new(10, 10);
    img.fill(Pixel::rgb(100, 100, 100));
    let blurred = blur(&img);
    assert_eq!(blurred.width(), 10);
}

#[test]
fn test_sharpen_filter() {
    let mut img = Image::new(10, 10);
    img.fill(Pixel::rgb(100, 100, 100));
    let sharpened = sharpen(&img);
    assert_eq!(sharpened.width(), 10);
}

#[test]
fn test_edge_detect_filter() {
    let mut img = Image::new(10, 10);
    img.fill(Pixel::rgb(100, 100, 100));
    let edges = edge_detect(&img);
    assert_eq!(edges.width(), 10);
}

#[test]
fn test_resize_nearest() {
    let img = Image::new(10, 10);
    let resized = resize_nearest(&img, 20, 20);
    assert_eq!(resized.width(), 20);
    assert_eq!(resized.height(), 20);
}

#[test]
fn test_resize_bilinear() {
    let img = Image::new(10, 10);
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

    let rotated = rotate_90(&img);
    assert_eq!(rotated.width(), 3);
    assert_eq!(rotated.height(), 2);
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
fn test_ppm_export() {
    let img = Image::new(2, 2);
    let ppm = img.to_ppm();
    assert!(ppm.starts_with("P3\n2 2\n255\n"));
}

#[test]
fn test_blend_images() {
    let a = Image::filled(5, 5, Pixel::black());
    let b = Image::filled(5, 5, Pixel::white());
    let blended = blend_images(&a, &b, 0.5);
    assert_eq!(blended.get_pixel(0, 0), Pixel::rgb(127, 127, 127));
}
