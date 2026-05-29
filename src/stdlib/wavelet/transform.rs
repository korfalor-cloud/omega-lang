/// Wavelet transforms: Haar, Daubechies, DWT, CWT.

/// Haar wavelet transform.
pub fn haar_forward(signal: &[f64]) -> Vec<f64> {
    let mut data = signal.to_vec();
    let n = data.len();
    assert!(n.is_power_of_two(), "Signal length must be power of 2");

    let mut len = n;
    while len > 1 {
        let half = len / 2;
        let mut temp = vec![0.0; len];
        for i in 0..half {
            temp[i] = (data[2 * i] + data[2 * i + 1]) / std::f64::consts::SQRT_2;
            temp[half + i] = (data[2 * i] - data[2 * i + 1]) / std::f64::consts::SQRT_2;
        }
        data[..len].copy_from_slice(&temp);
        len = half;
    }

    data
}

pub fn haar_inverse(coeffs: &[f64]) -> Vec<f64> {
    let mut data = coeffs.to_vec();
    let n = data.len();

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let mut temp = vec![0.0; len];
        for i in 0..half {
            temp[2 * i] = (data[i] + data[half + i]) / std::f64::consts::SQRT_2;
            temp[2 * i + 1] = (data[i] - data[half + i]) / std::f64::consts::SQRT_2;
        }
        data[..len].copy_from_slice(&temp);
        len *= 2;
    }

    data
}

/// Daubechies-4 wavelet transform.
pub struct Daubechies4;

impl Daubechies4 {
    // Daubechies-4 coefficients
    const H: [f64; 4] = [
        0.4829629131445341,
        0.8365163037378079,
        0.2241438680420134,
        -0.1294095225512604,
    ];

    const G: [f64; 4] = [
        -0.1294095225512604,
        -0.2241438680420134,
        0.8365163037378079,
        -0.4829629131445341,
    ];

    pub fn forward(signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        assert!(n.is_power_of_two());
        let mut output = vec![0.0; n];

        let mut len = n;
        while len >= 4 {
            let half = len / 2;
            for i in 0..half {
                let mut approx = 0.0;
                let mut detail = 0.0;
                for j in 0..4 {
                    let idx = (2 * i + j) % len;
                    approx += Self::H[j] * signal[idx];
                    detail += Self::G[j] * signal[idx];
                }
                output[i] = approx;
                output[half + i] = detail;
            }
            len = half;
        }

        output
    }

    pub fn inverse(coeffs: &[f64]) -> Vec<f64> {
        let n = coeffs.len();
        let mut output = coeffs.to_vec();

        let mut len = 4;
        while len <= n {
            let half = len / 2;
            let mut temp = vec![0.0; len];
            for i in 0..half {
                for j in 0..4 {
                    let idx = (2 * i + j) % len;
                    temp[idx] += Self::H[j] * output[i] + Self::G[j] * output[half + i];
                }
            }
            output[..len].copy_from_slice(&temp);
            len *= 2;
        }

        output
    }
}

/// Discrete Wavelet Transform with arbitrary filter bank.
pub struct DWT {
    pub low_pass: Vec<f64>,
    pub high_pass: Vec<f64>,
}

impl DWT {
    pub fn new(low_pass: Vec<f64>, high_pass: Vec<f64>) -> Self {
        Self { low_pass, high_pass }
    }

    pub fn haar() -> Self {
        let s = std::f64::consts::SQRT_2;
        Self::new(
            vec![1.0 / s, 1.0 / s],
            vec![1.0 / s, -1.0 / s],
        )
    }

    pub fn decompose(&self, signal: &[f64], levels: usize) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        let mut current = signal.to_vec();

        for _ in 0..levels {
            if current.len() < self.low_pass.len() { break; }
            let (approx, detail) = self.single_level(&current);
            result.push(detail);
            current = approx;
        }

        result.push(current); // Final approximation
        result
    }

    pub fn reconstruct(&self, coefficients: &[Vec<f64>]) -> Vec<f64> {
        let mut current = coefficients.last().unwrap().clone();

        for detail in coefficients.iter().rev().skip(1) {
            current = self.single_level_inverse(&current, detail);
        }

        current
    }

    fn single_level(&self, signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = signal.len();
        let half = n / 2;
        let filter_len = self.low_pass.len();

        let mut approx = vec![0.0; half];
        let mut detail = vec![0.0; half];

        for i in 0..half {
            for j in 0..filter_len {
                let idx = (2 * i + j) % n;
                approx[i] += self.low_pass[j] * signal[idx];
                detail[i] += self.high_pass[j] * signal[idx];
            }
        }

        (approx, detail)
    }

    fn single_level_inverse(&self, approx: &[f64], detail: &[f64]) -> Vec<f64> {
        let half = approx.len();
        let n = half * 2;
        let filter_len = self.low_pass.len();
        let mut signal = vec![0.0; n];

        for i in 0..half {
            for j in 0..filter_len {
                let idx = (2 * i + j) % n;
                signal[idx] += self.low_pass[j] * approx[i] + self.high_pass[j] * detail[i];
            }
        }

        signal
    }
}

/// Continuous Wavelet Transform (using Morlet wavelet).
pub struct CWT {
    pub scales: Vec<f64>,
    pub dt: f64,
}

impl CWT {
    pub fn new(scales: Vec<f64>, dt: f64) -> Self {
        Self { scales, dt }
    }

    /// Morlet wavelet function.
    fn morlet(&self, t: f64, scale: f64) -> f64 {
        let omega0 = 6.0; // Central frequency
        let eta = t / scale;
        let pi_factor = std::f64::consts::PI.powf(-0.25);
        pi_factor * (omega0 * eta * 0.0).cos() * (-eta * eta / 2.0).exp()
    }

    pub fn transform(&self, signal: &[f64]) -> Vec<Vec<f64>> {
        let n = signal.len();
        let mut result = Vec::new();

        for &scale in &self.scales {
            let mut coeffs = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    let t = (j as f64 - i as f64) * self.dt;
                    coeffs[i] += signal[j] * self.morlet(t, scale) * self.dt;
                }
            }
            // Normalize
            let norm = scale.sqrt();
            for c in &mut coeffs { *c /= norm; }
            result.push(coeffs);
        }

        result
    }

    /// Compute scalogram (power of CWT coefficients).
    pub fn scalogram(&self, signal: &[f64]) -> Vec<Vec<f64>> {
        let coeffs = self.transform(signal);
        coeffs.iter().map(|row| row.iter().map(|c| c * c).collect()).collect()
    }
}

/// Wavelet packet decomposition.
pub struct WaveletPacket {
    pub dwt: DWT,
    pub max_level: usize,
}

impl WaveletPacket {
    pub fn new(dwt: DWT, max_level: usize) -> Self {
        Self { dwt, max_level }
    }

    pub fn decompose(&self, signal: &[f64]) -> Vec<(usize, usize, Vec<f64>)> {
        let mut packets = Vec::new();
        self.decompose_recursive(signal, 0, 0, &mut packets);
        packets
    }

    fn decompose_recursive(&self, signal: &[f64], level: usize, node: usize, packets: &mut Vec<(usize, usize, Vec<f64>)>) {
        if level >= self.max_level || signal.len() < self.dwt.low_pass.len() {
            packets.push((level, node, signal.to_vec()));
            return;
        }

        let (approx, detail) = self.dwt.single_level(signal);
        self.decompose_recursive(&approx, level + 1, node * 2, packets);
        self.decompose_recursive(&detail, level + 1, node * 2 + 1, packets);
    }

    /// Best basis selection (minimizes entropy).
    pub fn best_basis(&self, signal: &[f64]) -> Vec<(usize, usize, Vec<f64>)> {
        let packets = self.decompose(signal);

        // Compute entropy for each packet
        let mut scored: Vec<(f64, usize, usize, Vec<f64>)> = packets.into_iter()
            .map(|(level, node, data)| {
                let entropy = self.shannon_entropy(&data);
                (entropy, level, node, data)
            })
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.into_iter().map(|(_, l, n, d)| (l, n, d)).collect()
    }

    fn shannon_entropy(&self, data: &[f64]) -> f64 {
        let total: f64 = data.iter().map(|x| x * x).sum();
        if total == 0.0 { return 0.0; }

        -data.iter()
            .map(|x| {
                let p = (x * x) / total;
                if p > 0.0 { p * p.ln() } else { 0.0 }
            })
            .sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_roundtrip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = haar_forward(&signal);
        let reconstructed = haar_inverse(&coeffs);
        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dwt_decompose() {
        let dwt = DWT::haar();
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let coeffs = dwt.decompose(&signal, 3);
        let reconstructed = dwt.reconstruct(&coeffs);
        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
