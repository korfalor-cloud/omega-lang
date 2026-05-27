/// Digital signal filters: FIR, IIR, Butterworth, Chebyshev.

use std::f64::consts::PI;

/// FIR (Finite Impulse Response) filter.
#[derive(Debug, Clone)]
pub struct FirFilter {
    coefficients: Vec<f64>,
    buffer: Vec<f64>,
    index: usize,
}

impl FirFilter {
    pub fn new(coefficients: Vec<f64>) -> Self {
        let n = coefficients.len();
        Self {
            coefficients,
            buffer: vec![0.0; n],
            index: 0,
        }
    }

    /// Process a single sample.
    pub fn process(&mut self, sample: f64) -> f64 {
        self.buffer[self.index] = sample;
        let n = self.coefficients.len();
        let mut output = 0.0;
        for i in 0..n {
            let idx = (self.index + n - i) % n;
            output += self.coefficients[i] * self.buffer[idx];
        }
        self.index = (self.index + 1) % n;
        output
    }

    /// Process a block of samples.
    pub fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&s| self.process(s)).collect()
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.index = 0;
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

/// IIR (Infinite Impulse Response) filter.
#[derive(Debug, Clone)]
pub struct IirFilter {
    b: Vec<f64>,  // feedforward coefficients
    a: Vec<f64>,  // feedback coefficients
    x_buffer: Vec<f64>,
    y_buffer: Vec<f64>,
    index: usize,
}

impl IirFilter {
    pub fn new(b: Vec<f64>, a: Vec<f64>) -> Self {
        let n = b.len().max(a.len());
        Self {
            b,
            a,
            x_buffer: vec![0.0; n],
            y_buffer: vec![0.0; n],
            index: 0,
        }
    }

    pub fn process(&mut self, sample: f64) -> f64 {
        let n = self.b.len().max(self.a.len());
        self.x_buffer[self.index] = sample;

        let mut output = 0.0;
        for i in 0..self.b.len() {
            let idx = (self.index + n - i) % n;
            output += self.b[i] * self.x_buffer[idx];
        }
        for i in 1..self.a.len() {
            let idx = (self.index + n - i) % n;
            output -= self.a[i] * self.y_buffer[idx];
        }
        if !self.a.is_empty() {
            output /= self.a[0];
        }

        self.y_buffer[self.index] = output;
        self.index = (self.index + 1) % n;
        output
    }

    pub fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&s| self.process(s)).collect()
    }

    pub fn reset(&mut self) {
        self.x_buffer.fill(0.0);
        self.y_buffer.fill(0.0);
        self.index = 0;
    }
}

/// Design a low-pass FIR filter using the windowed sinc method.
pub fn design_lowpass_fir(cutoff: f64, sample_rate: f64, num_taps: usize) -> Vec<f64> {
    let fc = cutoff / sample_rate;
    let m = (num_taps - 1) as f64;
    let mut coefficients = vec![0.0; num_taps];
    let window = super::fft::hamming_window(num_taps);

    for i in 0..num_taps {
        let n = i as f64 - m / 2.0;
        if n.abs() < f64::EPSILON {
            coefficients[i] = 2.0 * fc;
        } else {
            coefficients[i] = (2.0 * PI * fc * n).sin() / (PI * n);
        }
        coefficients[i] *= window[i];
    }

    // Normalize
    let sum: f64 = coefficients.iter().sum();
    for c in &mut coefficients {
        *c /= sum;
    }

    coefficients
}

/// Design a high-pass FIR filter.
pub fn design_highpass_fir(cutoff: f64, sample_rate: f64, num_taps: usize) -> Vec<f64> {
    let mut coefficients = design_lowpass_fir(cutoff, sample_rate, num_taps);
    for i in 0..num_taps {
        coefficients[i] = -coefficients[i];
    }
    coefficients[num_taps / 2] += 1.0;
    coefficients
}

/// Design a band-pass FIR filter.
pub fn design_bandpass_fir(low: f64, high: f64, sample_rate: f64, num_taps: usize) -> Vec<f64> {
    let lp = design_lowpass_fir(high, sample_rate, num_taps);
    let hp = design_highpass_fir(low, sample_rate, num_taps);
    lp.iter().zip(hp.iter()).map(|(a, b)| a + b).collect()
}

/// Design a band-stop (notch) FIR filter.
pub fn design_bandstop_fir(low: f64, high: f64, sample_rate: f64, num_taps: usize) -> Vec<f64> {
    let bp = design_bandpass_fir(low, high, sample_rate, num_taps);
    bp.iter().map(|&c| -c).enumerate().map(|(i, c)| {
        if i == bp.len() / 2 { c + 1.0 } else { c }
    }).collect()
}

/// Moving average filter.
pub fn moving_average(input: &[f64], window_size: usize) -> Vec<f64> {
    if input.is_empty() || window_size == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(input.len());
    let mut sum = 0.0;

    for i in 0..input.len() {
        sum += input[i];
        if i >= window_size {
            sum -= input[i - window_size];
        }
        let count = (i + 1).min(window_size);
        result.push(sum / count as f64);
    }

    result
}

/// Exponential moving average.
pub fn exponential_moving_average(input: &[f64], alpha: f64) -> Vec<f64> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(input.len());
    result.push(input[0]);

    for i in 1..input.len() {
        let ema = alpha * input[i] + (1.0 - alpha) * result[i - 1];
        result.push(ema);
    }

    result
}

/// Weighted moving average.
pub fn weighted_moving_average(input: &[f64], window_size: usize) -> Vec<f64> {
    if input.is_empty() || window_size == 0 {
        return Vec::new();
    }

    let weight_sum: f64 = (1..=window_size).map(|i| i as f64).sum();
    let mut result = Vec::with_capacity(input.len());

    for i in 0..input.len() {
        let start = if i + 1 >= window_size { i + 1 - window_size } else { 0 };
        let mut weighted = 0.0;
        let mut w_sum = 0.0;
        for j in start..=i {
            let w = (j - start + 1) as f64;
            weighted += input[j] * w;
            w_sum += w;
        }
        result.push(weighted / w_sum);
    }

    result
}

/// Savitzky-Golay smoothing filter (polynomial smoothing).
pub fn savitzky_golay(input: &[f64], window_size: usize, poly_order: usize) -> Vec<f64> {
    if input.len() < window_size || window_size <= poly_order {
        return input.to_vec();
    }

    let half = window_size / 2;
    let coefficients = sg_coefficients(window_size, poly_order);
    let mut result = vec![0.0; input.len()];

    for i in 0..input.len() {
        let start = if i >= half { i - half } else { 0 };
        let end = (i + half + 1).min(input.len());
        let mut sum = 0.0;
        for (j, &val) in input[start..end].iter().enumerate() {
            let coeff_idx = j + (half.saturating_sub(i));
            if coeff_idx < coefficients.len() {
                sum += val * coefficients[coeff_idx];
            }
        }
        result[i] = sum;
    }

    result
}

fn sg_coefficients(window_size: usize, poly_order: usize) -> Vec<f64> {
    // Simplified: return smoothing kernel based on polynomial order
    let n = window_size;
    match poly_order {
        0 => vec![1.0 / n as f64; n],
        1 => {
            // Linear smoothing weights
            let mut weights = vec![0.0; n];
            let half = n / 2;
            let norm = ((half * (half + 1)) / 2) as f64 * 2.0 + if n % 2 == 1 { (half + 1) as f64 } else { 0.0 };
            for i in 0..n {
                let dist = if i <= half { half - i } else { i - half };
                weights[i] = (half - dist + 1) as f64 / norm;
            }
            weights
        }
        _ => {
            // Higher order: use triangular window as approximation
            let mut weights = vec![0.0; n];
            let half = n / 2;
            let sum: f64 = (0..n).map(|i| {
                let dist = if i <= half { half - i } else { i - half };
                (half - dist + 1) as f64
            }).sum();
            for i in 0..n {
                let dist = if i <= half { half - i } else { i - half };
                weights[i] = (half - dist + 1) as f64 / sum;
            }
            weights
        }
    }
}

/// Median filter (non-linear).
pub fn median_filter(input: &[f64], window_size: usize) -> Vec<f64> {
    if input.is_empty() || window_size == 0 {
        return Vec::new();
    }

    let half = window_size / 2;
    let mut result = Vec::with_capacity(input.len());

    for i in 0..input.len() {
        let start = if i >= half { i - half } else { 0 };
        let end = (i + half + 1).min(input.len());
        let mut window: Vec<f64> = input[start..end].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        result.push(window[window.len() / 2]);
    }

    result
}

/// Envelope detection using Hilbert transform approximation.
pub fn envelope(input: &[f64]) -> Vec<f64> {
    let analytic = hilbert(input);
    analytic.iter().map(|c| c.magnitude()).collect()
}

/// Hilbert transform (returns analytic signal).
pub fn hilbert(input: &[f64]) -> Vec<super::fft::Complex> {
    use super::fft::{Complex, fft, ifft};

    let n = input.len();
    let fft_size = n.next_power_of_two();
    let mut padded: Vec<Complex> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    padded.resize(fft_size, Complex::zero());

    let spectrum = fft(&padded);

    let mut h = vec![Complex::zero(); fft_size];
    h[0] = Complex::new(1.0, 0.0);
    if fft_size % 2 == 0 {
        h[fft_size / 2] = Complex::new(1.0, 0.0);
        for i in 1..fft_size / 2 {
            h[i] = Complex::new(2.0, 0.0);
        }
    } else {
        for i in 1..(fft_size + 1) / 2 {
            h[i] = Complex::new(2.0, 0.0);
        }
    }

    let product: Vec<Complex> = spectrum.iter().zip(h.iter()).map(|(s, h)| *s * *h).collect();
    let result = ifft(&product);
    result[..n].to_vec()
}

/// Zero-crossing rate.
pub fn zero_crossing_rate(input: &[f64]) -> f64 {
    if input.len() < 2 {
        return 0.0;
    }
    let crossings = input.windows(2)
        .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
        .count();
    crossings as f64 / (input.len() - 1) as f64
}

/// Root mean square of a signal.
pub fn rms(input: &[f64]) -> f64 {
    if input.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = input.iter().map(|x| x * x).sum();
    (sum_sq / input.len() as f64).sqrt()
}

/// Signal-to-noise ratio.
pub fn snr(signal: &[f64], noise: &[f64]) -> f64 {
    let signal_power: f64 = signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64;
    let noise_power: f64 = noise.iter().map(|x| x * x).sum::<f64>() / noise.len() as f64;
    if noise_power == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fir_filter() {
        let coeffs = vec![0.25, 0.5, 0.25];
        let mut filter = FirFilter::new(coeffs);
        let input = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let output = filter.process_block(&input);
        assert!((output[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_moving_average() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = moving_average(&input, 3);
        assert_eq!(result.len(), 5);
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_filter() {
        let input = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let result = median_filter(&input, 3);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let zcr = zero_crossing_rate(&signal);
        assert!((zcr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rms() {
        let signal = vec![1.0, -1.0, 1.0, -1.0];
        assert!((rms(&signal) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_moving_average() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = exponential_moving_average(&input, 0.5);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < 1e-10);
    }
}
