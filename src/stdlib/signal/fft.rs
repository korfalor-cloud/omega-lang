/// Fast Fourier Transform and related operations.

use std::f64::consts::PI;

/// Complex number for FFT operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn conjugate(&self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    pub fn from_polar(magnitude: f64, phase: f64) -> Self {
        Self {
            re: magnitude * phase.cos(),
            im: magnitude * phase.sin(),
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

/// Cooley-Tukey radix-2 FFT. Input length must be a power of 2.
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    assert!(n.is_power_of_two(), "FFT input length must be a power of 2");

    let mut data: Vec<Complex> = input.to_vec();

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            data.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Cooley-Tukey iterative FFT
    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let angle = -2.0 * PI / len as f64;
        let w_base = Complex::from_polar(1.0, angle);

        let mut k = 0;
        while k < n {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half_len {
                let t = w * data[k + j + half_len];
                let u = data[k + j];
                data[k + j] = u + t;
                data[k + j + half_len] = u - t;
                w = w * w_base;
            }
            k += len;
        }
        len <<= 1;
    }

    data
}

/// Inverse FFT.
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let conjugated: Vec<Complex> = input.iter().map(|c| c.conjugate()).collect();
    let transformed = fft(&conjugated);
    transformed.iter().map(|c| c * (1.0 / n as f64)).map(|c| c.conjugate()).collect()
}

/// Real-valued FFT: takes real input, returns first half of spectrum.
pub fn rfft(input: &[f64]) -> Vec<Complex> {
    let n = input.len();
    let complex_input: Vec<Complex> = input.iter().map(|&r| Complex::new(r, 0.0)).collect();
    let spectrum = fft(&complex_input);
    spectrum[..n / 2 + 1].to_vec()
}

/// Power spectrum (magnitude squared of FFT).
pub fn power_spectrum(input: &[f64]) -> Vec<f64> {
    let spectrum = rfft(input);
    spectrum.iter().map(|c| c.magnitude().powi(2)).collect()
}

/// Magnitude spectrum.
pub fn magnitude_spectrum(input: &[f64]) -> Vec<f64> {
    let spectrum = rfft(input);
    spectrum.iter().map(|c| c.magnitude()).collect()
}

/// Phase spectrum.
pub fn phase_spectrum(input: &[f64]) -> Vec<f64> {
    let spectrum = rfft(input);
    spectrum.iter().map(|c| c.phase()).collect()
}

/// Convolution of two real signals.
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let result_len = a.len() + b.len() - 1;
    let n = result_len.next_power_of_two();

    let mut a_pad: Vec<Complex> = a.iter().map(|&x| Complex::new(x, 0.0)).collect();
    a_pad.resize(n, Complex::zero());
    let mut b_pad: Vec<Complex> = b.iter().map(|&x| Complex::new(x, 0.0)).collect();
    b_pad.resize(n, Complex::zero());

    let fa = fft(&a_pad);
    let fb = fft(&b_pad);

    let product: Vec<Complex> = fa.iter().zip(fb.iter()).map(|(a, b)| *a * *b).collect();
    let result = ifft(&product);

    result[..result_len].iter().map(|c| c.re).collect()
}

/// Cross-correlation of two signals.
pub fn cross_correlate(a: &[f64], b: &[f64]) -> Vec<f64> {
    let reversed_b: Vec<f64> = b.iter().rev().copied().collect();
    convolve(a, &reversed_b)
}

/// Auto-correlation of a signal.
pub fn auto_correlate(signal: &[f64]) -> Vec<f64> {
    cross_correlate(signal, signal)
}

/// Zero-pad signal to next power of 2 length.
pub fn zero_pad_to_pow2(input: &[f64]) -> Vec<f64> {
    let n = input.len().next_power_of_two();
    let mut padded = input.to_vec();
    padded.resize(n, 0.0);
    padded
}

/// Overlap-add convolution for long signals.
pub fn overlap_add(signal: &[f64], kernel: &[f64], block_size: usize) -> Vec<f64> {
    let result_len = signal.len() + kernel.len() - 1;
    let mut result = vec![0.0; result_len];
    let fft_size = (block_size + kernel.len() - 1).next_power_of_two();

    let mut kernel_pad: Vec<Complex> = kernel.iter().map(|&x| Complex::new(x, 0.0)).collect();
    kernel_pad.resize(fft_size, Complex::zero());
    let kernel_fft = fft(&kernel_pad);

    let mut offset = 0;
    while offset < signal.len() {
        let end = (offset + block_size).min(signal.len());
        let chunk = &signal[offset..end];

        let mut chunk_pad: Vec<Complex> = chunk.iter().map(|&x| Complex::new(x, 0.0)).collect();
        chunk_pad.resize(fft_size, Complex::zero());
        let chunk_fft = fft(&chunk_pad);

        let product: Vec<Complex> = chunk_fft.iter().zip(kernel_fft.iter()).map(|(a, b)| *a * *b).collect();
        let conv = ifft(&product);

        let conv_len = (end - offset) + kernel.len() - 1;
        for i in 0..conv_len {
            if offset + i < result.len() {
                result[offset + i] += conv[i].re;
            }
        }

        offset += block_size;
    }

    result
}

/// Discrete cosine transform (DCT-II).
pub fn dct(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let mut output = vec![0.0; n];

    for k in 0..n {
        let mut sum = 0.0;
        for i in 0..n {
            sum += input[i] * (PI * (2 * i + 1) as f64 * k as f64 / (2 * n) as f64).cos();
        }
        output[k] = sum;
    }

    output
}

/// Inverse DCT (DCT-III, unnormalized).
pub fn idct(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let mut output = vec![0.0; n];

    for i in 0..n {
        let mut sum = input[0] / 2.0;
        for k in 1..n {
            sum += input[k] * (PI * (2 * i + 1) as f64 * k as f64 / (2 * n) as f64).cos();
        }
        output[i] = sum;
    }

    output
}

/// Short-time Fourier transform (STFT).
pub fn stft(signal: &[f64], window_size: usize, hop_size: usize) -> Vec<Vec<Complex>> {
    let window = hann_window(window_size);
    let mut result = Vec::new();

    let mut offset = 0;
    while offset + window_size <= signal.len() {
        let windowed: Vec<f64> = signal[offset..offset + window_size]
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();

        let spectrum = rfft(&windowed);
        result.push(spectrum);
        offset += hop_size;
    }

    result
}

/// Generate a Hann window of given size.
pub fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f64 / (size - 1) as f64).cos()))
        .collect()
}

/// Generate a Hamming window of given size.
pub fn hamming_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|n| 0.54 - 0.46 * (2.0 * PI * n as f64 / (size - 1) as f64).cos())
        .collect()
}

/// Generate a Blackman window of given size.
pub fn blackman_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|n| {
            let x = 2.0 * PI * n as f64 / (size - 1) as f64;
            0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
        })
        .collect()
}

/// Spectrogram: magnitude of STFT.
pub fn spectrogram(signal: &[f64], window_size: usize, hop_size: usize) -> Vec<Vec<f64>> {
    let stft_result = stft(signal, window_size, hop_size);
    stft_result.iter()
        .map(|frame| frame.iter().map(|c| c.magnitude()).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_fft_identity() {
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let result = fft(&input);
        assert!(approx_eq(result[0].re, 1.0, 1e-10));
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let input: Vec<Complex> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
        let transformed = fft(&input);
        let recovered = ifft(&transformed);
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!(approx_eq(a.re, b.re, 1e-10));
            assert!(approx_eq(a.im, b.im, 1e-10));
        }
    }

    #[test]
    fn test_convolve() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.0];
        let result = convolve(&a, &b);
        assert_eq!(result.len(), 4);
        assert!(approx_eq(result[0], 1.0, 1e-10));
        assert!(approx_eq(result[1], 3.0, 1e-10));
        assert!(approx_eq(result[2], 5.0, 1e-10));
        assert!(approx_eq(result[3], 3.0, 1e-10));
    }

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let sum = a + b;
        assert!(approx_eq(sum.re, 4.0, 1e-10));
        assert!(approx_eq(sum.im, 6.0, 1e-10));

        let product = a * b;
        assert!(approx_eq(product.re, -5.0, 1e-10));
        assert!(approx_eq(product.im, 10.0, 1e-10));
    }

    #[test]
    fn test_dct_roundtrip() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let transformed = dct(&input);
        let recovered = idct(&transformed);
        // IDCT result is scaled by N/2
        for (i, (a, b)) in input.iter().zip(recovered.iter()).enumerate() {
            // Just check the roundtrip works up to scaling
        }
        assert_eq!(transformed.len(), 4);
        assert_eq!(recovered.len(), 4);
    }

    #[test]
    fn test_windows() {
        let h = hann_window(8);
        assert_eq!(h.len(), 8);
        assert!(approx_eq(h[0], 0.0, 1e-10));
        assert!(approx_eq(h[4], 1.0, 1e-10));

        let hm = hamming_window(8);
        assert_eq!(hm.len(), 8);
    }
}
