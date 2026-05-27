/// Charting and data visualization: line, bar, scatter, histogram, pie, heatmap.

#[derive(Debug, Clone)]
pub struct Chart {
    title: String,
    width: usize,
    height: usize,
    x_label: String,
    y_label: String,
    series: Vec<Series>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    auto_range: bool,
}

#[derive(Debug, Clone)]
pub struct Series {
    pub name: String,
    pub data: Vec<(f64, f64)>,
    pub style: PlotStyle,
}

#[derive(Debug, Clone)]
pub enum PlotStyle {
    Line { color: char, marker: char },
    Bar { fill: char },
    Scatter { marker: char },
    Area { fill: char },
}

impl Chart {
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            width: 80,
            height: 24,
            x_label: String::new(),
            y_label: String::new(),
            series: Vec::new(),
            x_min: f64::INFINITY,
            x_max: f64::NEG_INFINITY,
            y_min: f64::INFINITY,
            y_max: f64::NEG_INFINITY,
            auto_range: true,
        }
    }

    pub fn with_size(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_x_label(mut self, label: &str) -> Self {
        self.x_label = label.to_string();
        self
    }

    pub fn with_y_label(mut self, label: &str) -> Self {
        self.y_label = label.to_string();
        self
    }

    pub fn with_range(mut self, x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        self.x_min = x_min;
        self.x_max = x_max;
        self.y_min = y_min;
        self.y_max = y_max;
        self.auto_range = false;
        self
    }

    pub fn add_line_series(&mut self, name: &str, data: &[(f64, f64)]) {
        self.update_range(data);
        self.series.push(Series {
            name: name.to_string(),
            data: data.to_vec(),
            style: PlotStyle::Line { color: '█', marker: '●' },
        });
    }

    pub fn add_scatter_series(&mut self, name: &str, data: &[(f64, f64)]) {
        self.update_range(data);
        self.series.push(Series {
            name: name.to_string(),
            data: data.to_vec(),
            style: PlotStyle::Scatter { marker: '●' },
        });
    }

    pub fn add_bar_series(&mut self, name: &str, data: &[(f64, f64)]) {
        self.update_range(data);
        self.series.push(Series {
            name: name.to_string(),
            data: data.to_vec(),
            style: PlotStyle::Bar { fill: '█' },
        });
    }

    fn update_range(&mut self, data: &[(f64, f64)]) {
        if !self.auto_range {
            return;
        }
        for &(x, y) in data {
            self.x_min = self.x_min.min(x);
            self.x_max = self.x_max.max(x);
            self.y_min = self.y_min.min(y);
            self.y_max = self.y_max.max(y);
        }
    }

    /// Render chart as ASCII art.
    pub fn render(&self) -> String {
        if self.series.is_empty() {
            return format!("{}\n[No data]", self.title);
        }

        let x_min = if self.auto_range { self.x_min } else { self.x_min };
        let x_max = if self.auto_range { self.x_max.max(self.x_min + 1.0) } else { self.x_max };
        let y_min = if self.auto_range { self.y_min } else { self.y_min };
        let y_max = if self.auto_range { self.y_max.max(self.y_min + 1.0) } else { self.y_max };

        let plot_width = self.width.saturating_sub(8);
        let plot_height = self.height.saturating_sub(4);

        if plot_width == 0 || plot_height == 0 {
            return format!("{}\n[Chart too small]", self.title);
        }

        let mut grid = vec![vec![' '; plot_width]; plot_height];

        // Plot each series
        for series in &self.series {
            match &series.style {
                PlotStyle::Line { color, .. } => {
                    let mut prev_px: Option<usize> = None;
                    let mut prev_py: Option<usize> = None;
                    for &(x, y) in &series.data {
                        let px = ((x - x_min) / (x_max - x_min) * (plot_width - 1) as f64) as usize;
                        let py = ((y - y_min) / (y_max - y_min) * (plot_height - 1) as f64) as usize;
                        let px = px.min(plot_width - 1);
                        let py = (plot_height - 1).saturating_sub(py.min(plot_height - 1));

                        grid[py][px] = *color;

                        // Draw line between points
                        if let (Some(ppx), Some(ppy)) = (prev_px, prev_py) {
                            draw_line_on_grid(&mut grid, ppy, ppx, py, px, *color);
                        }
                        prev_px = Some(px);
                        prev_py = Some(py);
                    }
                }
                PlotStyle::Scatter { marker } => {
                    for &(x, y) in &series.data {
                        let px = ((x - x_min) / (x_max - x_min) * (plot_width - 1) as f64) as usize;
                        let py = ((y - y_min) / (y_max - y_min) * (plot_height - 1) as f64) as usize;
                        let px = px.min(plot_width - 1);
                        let py = (plot_height - 1).saturating_sub(py.min(plot_height - 1));
                        grid[py][px] = *marker;
                    }
                }
                PlotStyle::Bar { fill } => {
                    for &(x, y) in &series.data {
                        let px = ((x - x_min) / (x_max - x_min) * (plot_width - 1) as f64) as usize;
                        let px = px.min(plot_width - 1);
                        let bar_top = ((y - y_min) / (y_max - y_min) * (plot_height - 1) as f64) as usize;
                        let bar_top = (plot_height - 1).saturating_sub(bar_top.min(plot_height - 1));
                        for row in bar_top..plot_height {
                            grid[row][px] = *fill;
                        }
                    }
                }
                PlotStyle::Area { fill } => {
                    let mut prev_px: Option<usize> = None;
                    for &(x, y) in &series.data {
                        let px = ((x - x_min) / (x_max - x_min) * (plot_width - 1) as f64) as usize;
                        let py = ((y - y_min) / (y_max - y_min) * (plot_height - 1) as f64) as usize;
                        let px = px.min(plot_width - 1);
                        let py = (plot_height - 1).saturating_sub(py.min(plot_height - 1));

                        // Fill area below
                        for row in py..plot_height {
                            grid[row][px] = *fill;
                        }
                        prev_px = Some(px);
                    }
                }
            }
        }

        // Build output
        let mut output = String::new();

        // Title
        let padding = (self.width.saturating_sub(self.title.len())) / 2;
        output.push_str(&" ".repeat(padding));
        output.push_str(&self.title);
        output.push('\n');

        // Y-axis labels and grid
        let y_label_width = 7;
        for row in 0..plot_height {
            let y_val = y_max - (row as f64 / (plot_height - 1) as f64) * (y_max - y_min);
            if row == 0 || row == plot_height - 1 || row == plot_height / 2 {
                output.push_str(&format!("{:>y_label_width$.1} │", y_val));
            } else {
                output.push_str(&format!("{:>y_label_width$} │", ""));
            }
            for col in 0..plot_width {
                output.push(grid[row][col]);
            }
            output.push('\n');
        }

        // X-axis
        output.push_str(&format!("{:>y_label_width$} └", ""));
        output.push_str(&"─".repeat(plot_width));
        output.push('\n');

        // X-axis labels
        output.push_str(&format!("{:>y_label_width$}  {:.1}", "", x_min));
        let mid_x = (x_min + x_max) / 2.0;
        let mid_pos = plot_width / 2;
        let left_pad = mid_pos.saturating_sub(format!("{:.1}", mid_x).len() / 2);
        output.push_str(&" ".repeat(left_map_padding(left_pad, y_label_width + 2, plot_width, &format!("{:.1}", mid_x))));
        output.push_str(&format!("{:.1}", mid_x));
        output.push_str(&format!("{:>width$.1}", x_max, width = plot_width.saturating_sub(left_pad + format!("{:.1}", mid_x).len())));
        output.push('\n');

        // Legend
        if !self.series.is_empty() {
            output.push_str("\nLegend: ");
            for series in &self.series {
                output.push_str(&format!("{} ", series.name));
            }
            output.push('\n');
        }

        output
    }
}

fn left_map_padding(left_pad: usize, prefix: usize, plot_width: usize, mid_str: &str) -> usize {
    let expected = prefix + plot_width / 2;
    let actual = prefix + left_pad;
    if expected > actual { expected - actual } else { 0 }
}

fn draw_line_on_grid(grid: &mut Vec<Vec<char>>, y0: usize, x0: usize, y1: usize, x1: usize, ch: char) {
    // Bresenham's line algorithm
    let dx = (x1 as i32 - x0 as i32).abs();
    let dy = (y1 as i32 - y0 as i32).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut cx = x0 as i32;
    let mut cy = y0 as i32;
    let height = grid.len() as i32;
    let width = grid[0].len() as i32;

    loop {
        if cx >= 0 && cx < width && cy >= 0 && cy < height {
            grid[cy as usize][cx as usize] = ch;
        }
        if cx == x1 as i32 && cy == y1 as i32 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            cx += sx;
        }
        if e2 < dx {
            err += dx;
            cy += sy;
        }
    }
}

/// Generate a histogram as a bar chart.
pub fn histogram(values: &[f64], bins: usize) -> Chart {
    if values.is_empty() || bins == 0 {
        return Chart::new("Histogram");
    }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    let bin_width = range / bins as f64;

    let mut counts = vec![0usize; bins];
    for &v in values {
        let bin = ((v - min) / bin_width).floor() as usize;
        let bin = bin.min(bins - 1);
        counts[bin] += 1;
    }

    let data: Vec<(f64, f64)> = counts.iter().enumerate()
        .map(|(i, &c)| (min + (i as f64 + 0.5) * bin_width, c as f64))
        .collect();

    let mut chart = Chart::new("Histogram");
    chart.add_bar_series("frequency", &data);
    chart
}

/// Box plot statistics.
#[derive(Debug, Clone)]
pub struct BoxPlotStats {
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub mean: f64,
    pub outliers: Vec<f64>,
}

pub fn box_plot_stats(values: &[f64]) -> BoxPlotStats {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let median = percentile(&sorted, 50.0);
    let q1 = percentile(&sorted, 25.0);
    let q3 = percentile(&sorted, 75.0);
    let iqr = q3 - q1;
    let lower_fence = q1 - 1.5 * iqr;
    let upper_fence = q3 + 1.5 * iqr;

    let min = sorted.iter().cloned().find(|&v| v >= lower_fence).unwrap_or(sorted[0]);
    let max = sorted.iter().rev().cloned().find(|&v| v <= upper_fence).unwrap_or(sorted[n - 1]);

    let outliers: Vec<f64> = sorted.iter().cloned().filter(|&v| v < lower_fence || v > upper_fence).collect();
    let mean = sorted.iter().sum::<f64>() / n as f64;

    BoxPlotStats { min, q1, median, q3, max, mean, outliers }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let rank = p / 100.0 * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (rank - lower as f64) * (sorted[upper] - sorted[lower])
    }
}

/// Moving average smoothing for chart data.
pub fn moving_average(data: &[(f64, f64)], window: usize) -> Vec<(f64, f64)> {
    if data.is_empty() || window == 0 {
        return Vec::new();
    }

    let mut result = Vec::new();
    for i in 0..data.len() {
        let start = if i >= window { i - window + 1 } else { 0 };
        let count = i - start + 1;
        let sum: f64 = data[start..=i].iter().map(|&(_, y)| y).sum();
        result.push((data[i].0, sum / count as f64));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chart_render() {
        let mut chart = Chart::new("Test Chart");
        chart.add_line_series("line1", &[(0.0, 0.0), (1.0, 1.0), (2.0, 0.5)]);
        let rendered = chart.render();
        assert!(rendered.contains("Test Chart"));
    }

    #[test]
    fn test_histogram() {
        let values: Vec<f64> = (0..100).map(|i| (i as f64).sin() * 10.0).collect();
        let chart = histogram(&values, 10);
        assert!(!chart.series.is_empty());
    }

    #[test]
    fn test_box_plot() {
        let values: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let stats = box_plot_stats(&values);
        assert!((stats.median - 10.5).abs() < 0.1);
    }
}
