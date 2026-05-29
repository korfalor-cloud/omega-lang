/// Data visualization: line, bar, scatter, heatmap, contour, 3D surface,
/// histogram, box plot, violin plot, and animation support.

use std::f64::consts::PI;

// ── Core types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PlotCanvas {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Vec<char>>,
}

impl PlotCanvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![vec![' '; width]; height],
        }
    }

    pub fn set(&mut self, x: usize, y: usize, ch: char) {
        if x < self.width && y < self.height {
            self.pixels[y][x] = ch;
        }
    }

    pub fn render(&self) -> String {
        self.pixels.iter().map(|row| row.iter().collect::<String>()).collect::<Vec<_>>().join("\n")
    }
}

// ── Line plot ───────────────────────────────────────────────────────────────

pub fn line_plot(points: &[(f64, f64)], width: usize, height: usize) -> PlotCanvas {
    let mut canvas = PlotCanvas::new(width, height);
    if points.len() < 2 {
        return canvas;
    }
    let x_min = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let x_range = (x_max - x_min).max(1e-12);
    let y_range = (y_max - y_min).max(1e-12);

    let to_canvas = |x: f64, y: f64| -> (usize, usize) {
        let cx = ((x - x_min) / x_range * (width - 1) as f64).round() as usize;
        let cy = ((y_max - y) / y_range * (height - 1) as f64).round() as usize;
        (cx.min(width - 1), cy.min(height - 1))
    };

    for w in points.windows(2) {
        let (x0, y0) = to_canvas(w[0].0, w[0].1);
        let (x1, y1) = to_canvas(w[1].0, w[1].1);
        bresenham_line(&mut canvas, x0, y0, x1, y1, '·');
    }
    for &(x, y) in points {
        let (cx, cy) = to_canvas(x, y);
        canvas.set(cx, cy, '●');
    }
    canvas
}

// ── Bar chart ───────────────────────────────────────────────────────────────

pub fn bar_chart(values: &[f64], labels: &[&str], width: usize, height: usize) -> PlotCanvas {
    let mut canvas = PlotCanvas::new(width, height);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1e-12);
    let bar_count = values.len().max(1);
    let bar_w = (width / bar_count).max(1);

    for (i, &v) in values.iter().enumerate() {
        let bar_h = ((v / max_val) * (height - 1) as f64).round() as usize;
        let x_start = i * bar_w;
        for row in (height - bar_h)..height {
            for col in x_start..(x_start + bar_w).min(width) {
                canvas.set(col, row, '█');
            }
        }
        // label
        if i < labels.len() {
            let label = labels[i];
            let lx = x_start + bar_w / 2;
            for (j, ch) in label.chars().enumerate() {
                canvas.set(lx + j, height - bar_h - 1, ch);
            }
        }
    }
    canvas
}

// ── Scatter plot ────────────────────────────────────────────────────────────

pub fn scatter_plot(points: &[(f64, f64)], width: usize, height: usize) -> PlotCanvas {
    let mut canvas = PlotCanvas::new(width, height);
    if points.is_empty() {
        return canvas;
    }
    let x_min = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let x_range = (x_max - x_min).max(1e-12);
    let y_range = (y_max - y_min).max(1e-12);

    for &(x, y) in points {
        let cx = ((x - x_min) / x_range * (width - 1) as f64).round() as usize;
        let cy = ((y_max - y) / y_range * (height - 1) as f64).round() as usize;
        canvas.set(cx.min(width - 1), cy.min(height - 1), '●');
    }
    canvas
}

// ── Heatmap ─────────────────────────────────────────────────────────────────

pub fn heatmap(data: &[Vec<f64>], width: usize, height: usize) -> PlotCanvas {
    let mut canvas = PlotCanvas::new(width, height);
    if data.is_empty() || data[0].is_empty() {
        return canvas;
    }
    let rows = data.len();
    let cols = data[0].len();
    let min = data.iter().flat_map(|r| r.iter()).cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().flat_map(|r| r.iter()).cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-12);
    let shades = [' ', '░', '▒', '▓', '█'];

    for r in 0..rows {
        for c in 0..cols {
            let norm = (data[r][c] - min) / range;
            let si = (norm * (shades.len() - 1) as f64).round() as usize;
            let ch = shades[si.min(shades.len() - 1)];
            let px = c * width / cols;
            let py = r * height / rows;
            canvas.set(px.min(width - 1), py.min(height - 1), ch);
        }
    }
    canvas
}

// ── Contour plot ────────────────────────────────────────────────────────────

pub fn contour_plot<F: Fn(f64, f64) -> f64>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    levels: &[f64],
    width: usize,
    height: usize,
) -> PlotCanvas {
    let mut canvas = PlotCanvas::new(width, height);
    let symbols = ['·', '○', '◎', '◉', '●'];

    for py in 0..height {
        for px in 0..width {
            let x = x_range.0 + (px as f64 / (width - 1).max(1) as f64) * (x_range.1 - x_range.0);
            let y = y_range.1 - (py as f64 / (height - 1).max(1) as f64) * (y_range.1 - y_range.0);
            let val = f(x, y);
            for (i, &lvl) in levels.iter().enumerate() {
                let threshold = (y_range.1 - y_range.0).abs() / (height as f64 * 2.0);
                if (val - lvl).abs() < threshold {
                    canvas.set(px, py, symbols[i % symbols.len()]);
                    break;
                }
            }
        }
    }
    canvas
}

// ── 3D surface (isometric projection) ───────────────────────────────────────

pub fn surface_3d<F: Fn(f64, f64) -> f64>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: usize,
    width: usize,
    height: usize,
) -> PlotCanvas {
    let mut canvas = PlotCanvas::new(width, height);
    let mut z_buffer = vec![vec![f64::NEG_INFINITY; width]; height];
    let shades = [' ', '·', '░', '▒', '▓', '█'];
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;

    for iy in 0..resolution {
        for ix in 0..resolution {
            let x = x_range.0 + (ix as f64 / (resolution - 1).max(1) as f64) * (x_range.1 - x_range.0);
            let y = y_range.0 + (iy as f64 / (resolution - 1).max(1) as f64) * (y_range.1 - y_range.0);
            let z = f(x, y);

            // Isometric projection
            let sx = (x - (x_range.0 + x_range.1) / 2.0) * 0.7;
            let sy = (y - (y_range.0 + y_range.1) / 2.0) * 0.7;
            let px = (cx + sx * 10.0 - sy * 10.0) as i32;
            let py = (cy - z * 2.0 + sx * 5.0 + sy * 5.0) as i32;

            if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                let (ux, uy) = (px as usize, py as usize);
                if z > z_buffer[uy][ux] {
                    z_buffer[uy][ux] = z;
                    let shade_idx = ((z - y_range.0) / (y_range.1 - y_range.0).max(1e-12)
                        * (shades.len() - 1) as f64)
                        .round() as usize;
                    canvas.set(ux, uy, shades[shade_idx.min(shades.len() - 1)]);
                }
            }
        }
    }
    canvas
}

// ── Histogram ───────────────────────────────────────────────────────────────

pub fn histogram(values: &[f64], bins: usize, width: usize, height: usize) -> PlotCanvas {
    let mut canvas = PlotCanvas::new(width, height);
    if values.is_empty() || bins == 0 {
        return canvas;
    }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_w = ((max - min) / bins as f64).max(1e-12);
    let mut counts = vec![0usize; bins];
    for &v in values {
        let b = ((v - min) / bin_w).floor() as usize;
        counts[b.min(bins - 1)] += 1;
    }
    let max_count = *counts.iter().max().unwrap_or(&1) as f64;
    let col_w = (width / bins).max(1);

    for (i, &cnt) in counts.iter().enumerate() {
        let bar_h = ((cnt as f64 / max_count) * (height - 1) as f64).round() as usize;
        let x_start = i * col_w;
        for row in (height - bar_h)..height {
            for col in x_start..(x_start + col_w).min(width) {
                canvas.set(col, row, '█');
            }
        }
    }
    canvas
}

// ── Box plot ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BoxPlotData {
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub outliers: Vec<f64>,
}

pub fn box_plot_stats(values: &[f64]) -> BoxPlotData {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let q1 = percentile(&sorted, 25.0);
    let median = percentile(&sorted, 50.0);
    let q3 = percentile(&sorted, 75.0);
    let iqr = q3 - q1;
    let lo = q1 - 1.5 * iqr;
    let hi = q3 + 1.5 * iqr;
    let min = sorted.iter().copied().find(|&v| v >= lo).unwrap_or(sorted[0]);
    let max = sorted.iter().copied().rev().find(|&v| v <= hi).unwrap_or(sorted[n - 1]);
    let outliers = sorted.iter().copied().filter(|&v| v < lo || v > hi).collect();
    BoxPlotData { min, q1, median, q3, max, outliers }
}

pub fn render_box_plot(bp: &BoxPlotData, width: usize) -> PlotCanvas {
    let mut canvas = PlotCanvas::new(width, 5);
    let lo = bp.min;
    let hi = bp.max;
    let range = (hi - lo).max(1e-12);

    let scale = |v: f64| -> usize {
        ((v - lo) / range * (width - 1) as f64).round() as usize
    };

    let q1x = scale(bp.q1);
    let qx = scale(bp.median);
    let q3x = scale(bp.q3);
    // whiskers
    let wlo = scale(bp.min);
    let whi = scale(bp.max);

    for x in wlo..=q1x.min(width - 1) {
        canvas.set(x, 2, '─');
    }
    for x in q3x..=whi.min(width - 1) {
        canvas.set(x, 2, '─');
    }
    for x in q1x..=q3x.min(width - 1) {
        canvas.set(x, 1, '─');
        canvas.set(x, 3, '─');
    }
    for row in 1..=3 {
        canvas.set(q1x.min(width - 1), row, '│');
        canvas.set(q3x.min(width - 1), row, '│');
    }
    canvas.set(qx.min(width - 1), 2, '◆');
    canvas.set(wlo.min(width - 1), 2, '├');
    canvas.set(whi.min(width - 1), 2, '┤');
    canvas
}

// ── Violin plot ─────────────────────────────────────────────────────────────

pub fn violin_plot(values: &[f64], bins: usize, width: usize, height: usize) -> PlotCanvas {
    let mut canvas = PlotCanvas::new(width, height);
    if values.is_empty() || bins == 0 {
        return canvas;
    }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_w = ((max - min) / bins as f64).max(1e-12);
    let mut counts = vec![0usize; bins];
    for &v in values {
        let b = ((v - min) / bin_w).floor() as usize;
        counts[b.min(bins - 1)] += 1;
    }
    let max_count = *counts.iter().max().unwrap_or(&1) as f64;
    let mid = width / 2;

    for (i, &cnt) in counts.iter().enumerate() {
        let row = i * height / bins;
        let half = ((cnt as f64 / max_count) * mid as f64).round() as usize;
        for dx in 0..half.min(mid) {
            canvas.set(mid - dx - 1, row.min(height - 1), '▓');
            canvas.set(mid + dx, row.min(height - 1), '▓');
        }
        canvas.set(mid, row.min(height - 1), '│');
    }
    canvas
}

// ── Animation support ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Animation {
    pub frames: Vec<PlotCanvas>,
    pub frame_rate_ms: u64,
}

impl Animation {
    pub fn new(frame_rate_ms: u64) -> Self {
        Self { frames: Vec::new(), frame_rate_ms }
    }

    pub fn add_frame(&mut self, frame: PlotCanvas) {
        self.frames.push(frame);
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn render_frame(&self, index: usize) -> Option<String> {
        self.frames.get(index).map(|f| f.render())
    }

    pub fn render_all(&self) -> Vec<String> {
        self.frames.iter().map(|f| f.render()).collect()
    }
}

/// Generate a rotating sine-wave animation.
pub fn animate_sine_wave(width: usize, height: usize, frames: usize) -> Animation {
    let mut anim = Animation::new(100);
    for f in 0..frames {
        let phase = 2.0 * PI * f as f64 / frames as f64;
        let pts: Vec<(f64, f64)> = (0..width)
            .map(|i| {
                let x = i as f64;
                let y = (x * 0.1 + phase).sin();
                (x, y)
            })
            .collect();
        anim.add_frame(line_plot(&pts, width, height));
    }
    anim
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn bresenham_line(canvas: &mut PlotCanvas, x0: usize, y0: usize, x1: usize, y1: usize, ch: char) {
    let dx = (x1 as i32 - x0 as i32).abs();
    let dy = -(y1 as i32 - y0 as i32).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut cx = x0 as i32;
    let mut cy = y0 as i32;
    let w = canvas.width as i32;
    let h = canvas.height as i32;

    loop {
        if cx >= 0 && cx < w && cy >= 0 && cy < h {
            canvas.set(cx as usize, cy as usize, ch);
        }
        if cx == x1 as i32 && cy == y1 as i32 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            cx += sx;
        }
        if e2 <= dx {
            err += dx;
            cy += sy;
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let rank = p / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (rank - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_plot() {
        let pts = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)];
        let c = line_plot(&pts, 20, 10);
        assert_eq!(c.width, 20);
        assert_eq!(c.height, 10);
        let r = c.render();
        assert!(!r.is_empty());
    }

    #[test]
    fn test_bar_chart() {
        let vals = vec![3.0, 7.0, 5.0, 9.0];
        let labels = vec!["A", "B", "C", "D"];
        let c = bar_chart(&vals, &labels, 40, 10);
        assert_eq!(c.width, 40);
        assert!(c.render().contains('█'));
    }

    #[test]
    fn test_scatter_plot() {
        let pts: Vec<(f64, f64)> = (0..50).map(|i| (i as f64, (i as f64 * 0.3).sin())).collect();
        let c = scatter_plot(&pts, 30, 15);
        assert!(c.render().contains('●'));
    }

    #[test]
    fn test_heatmap() {
        let data: Vec<Vec<f64>> = (0..5).map(|r| (0..5).map(|c| (r * c) as f64).collect()).collect();
        let c = heatmap(&data, 10, 10);
        assert_eq!(c.width, 10);
        assert!(!c.render().is_empty());
    }

    #[test]
    fn test_contour_plot() {
        let f = |x: f64, y: f64| x * x + y * y;
        let levels = vec![1.0, 4.0, 9.0];
        let c = contour_plot(f, (-3.0, 3.0), (-3.0, 3.0), &levels, 20, 20);
        assert!(!c.render().is_empty());
    }

    #[test]
    fn test_surface_3d() {
        let f = |x: f64, y: f64| (x * x + y * y).sqrt();
        let c = surface_3d(f, (-2.0, 2.0), (-2.0, 2.0), 15, 30, 20);
        assert!(!c.render().is_empty());
    }

    #[test]
    fn test_histogram() {
        let vals: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let c = histogram(&vals, 10, 40, 10);
        assert!(c.render().contains('█'));
    }

    #[test]
    fn test_box_plot_stats() {
        let vals: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let bp = box_plot_stats(&vals);
        assert!((bp.median - 10.5).abs() < 0.1);
        assert!(bp.q1 < bp.median);
        assert!(bp.median < bp.q3);
    }

    #[test]
    fn test_render_box_plot() {
        let vals: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let bp = box_plot_stats(&vals);
        let c = render_box_plot(&bp, 40);
        assert_eq!(c.height, 5);
        assert!(c.render().contains('◆'));
    }

    #[test]
    fn test_violin_plot() {
        let vals: Vec<f64> = (0..100).map(|i| (i as f64 * 0.05).sin()).collect();
        let c = violin_plot(&vals, 20, 30, 20);
        assert!(!c.render().is_empty());
    }

    #[test]
    fn test_animation() {
        let anim = animate_sine_wave(20, 10, 5);
        assert_eq!(anim.frame_count(), 5);
        assert!(anim.render_frame(0).is_some());
        assert!(anim.render_frame(99).is_none());
        let all = anim.render_all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_plot_canvas_set_and_render() {
        let mut c = PlotCanvas::new(5, 3);
        c.set(2, 1, 'X');
        let r = c.render();
        assert!(r.contains('X'));
    }

    #[test]
    fn test_empty_inputs() {
        let c = line_plot(&[], 10, 10);
        assert_eq!(c.width, 10);
        let c = scatter_plot(&[], 10, 10);
        assert_eq!(c.width, 10);
        let c = histogram(&[], 0, 10, 10);
        assert_eq!(c.width, 10);
        let c = heatmap(&[], 10, 10);
        assert_eq!(c.width, 10);
    }
}
