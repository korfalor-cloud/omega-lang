/// Quantum computing simulator: qubits, gates, circuits, measurement.

use std::f64::consts::PI;

/// Complex number for quantum state amplitudes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self { Self { re, im } }
    pub fn zero() -> Self { Self { re: 0.0, im: 0.0 } }
    pub fn one() -> Self { Self { re: 1.0, im: 0.0 } }

    pub fn magnitude_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn conjugate(&self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self { re: r * theta.cos(), im: r * theta.sin() }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self { re: self.re + rhs.re, im: self.im + rhs.im } }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self { re: self.re - rhs.re, im: self.im - rhs.im } }
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
    fn mul(self, rhs: f64) -> Self { Self { re: self.re * rhs, im: self.im * rhs } }
}

/// Quantum state vector for N qubits (2^N amplitudes).
#[derive(Debug, Clone)]
pub struct QuantumState {
    amplitudes: Vec<Complex>,
    num_qubits: usize,
}

impl QuantumState {
    /// Create |0...0⟩ state for n qubits.
    pub fn new(num_qubits: usize) -> Self {
        let size = 1 << num_qubits;
        let mut amplitudes = vec![Complex::zero(); size];
        amplitudes[0] = Complex::one();
        Self { amplitudes, num_qubits }
    }

    /// Create state from amplitudes.
    pub fn from_amplitudes(amplitudes: Vec<Complex>) -> Result<Self, String> {
        let size = amplitudes.len();
        if !size.is_power_of_two() {
            return Err("Amplitude count must be a power of 2".to_string());
        }
        let num_qubits = size.trailing_zeros() as usize;

        // Normalize
        let norm: f64 = amplitudes.iter().map(|a| a.magnitude_squared()).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err("Cannot normalize zero state".to_string());
        }
        let normalized: Vec<Complex> = amplitudes.iter().map(|a| *a * (1.0 / norm)).collect();

        Ok(Self { amplitudes: normalized, num_qubits })
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub fn amplitudes(&self) -> &[Complex] {
        &self.amplitudes
    }

    /// Apply a single-qubit gate to the specified qubit.
    pub fn apply_single_gate(&mut self, qubit: usize, gate: &[[Complex; 2]; 2]) {
        assert!(qubit < self.num_qubits);
        let size = self.amplitudes.len();
        let mask = 1 << qubit;

        let mut i = 0;
        while i < size {
            if i & mask == 0 {
                let j = i | mask;
                let a0 = self.amplitudes[i];
                let a1 = self.amplitudes[j];
                self.amplitudes[i] = gate[0][0] * a0 + gate[0][1] * a1;
                self.amplitudes[j] = gate[1][0] * a0 + gate[1][1] * a1;
            }
            i += 1;
        }
    }

    /// Apply a two-qubit gate (controlled gate).
    pub fn apply_two_qubit_gate(&mut self, control: usize, target: usize, gate: &[[Complex; 2]; 2]) {
        assert!(control < self.num_qubits && target < self.num_qubits && control != target);
        let size = self.amplitudes.len();
        let control_mask = 1 << control;
        let target_mask = 1 << target;

        let mut i = 0;
        while i < size {
            // Only apply when control qubit is |1⟩
            if i & control_mask != 0 && i & target_mask == 0 {
                let j = i | target_mask;
                let a0 = self.amplitudes[i];
                let a1 = self.amplitudes[j];
                self.amplitudes[i] = gate[0][0] * a0 + gate[0][1] * a1;
                self.amplitudes[j] = gate[1][0] * a0 + gate[1][1] * a1;
            }
            i += 1;
        }
    }

    /// Measure all qubits, collapsing the state.
    pub fn measure(&mut self) -> Vec<u8> {
        let mut result = vec![0u8; self.num_qubits];
        let probabilities: Vec<f64> = self.amplitudes.iter().map(|a| a.magnitude_squared()).collect();

        // Pick outcome based on probabilities
        let r = pseudo_random(&mut 12345u64);
        let mut cumulative = 0.0;
        let mut outcome = 0;
        for (i, &p) in probabilities.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                outcome = i;
                break;
            }
        }

        // Extract bit values
        for q in 0..self.num_qubits {
            result[q] = ((outcome >> q) & 1) as u8;
        }

        // Collapse state
        self.amplitudes.fill(Complex::zero());
        self.amplitudes[outcome] = Complex::one();

        result
    }

    /// Measure a single qubit without collapsing the full state.
    pub fn measure_qubit(&mut self, qubit: usize) -> u8 {
        let mask = 1 << qubit;
        let prob_one: f64 = self.amplitudes.iter().enumerate()
            .filter(|(i, _)| i & mask != 0)
            .map(|(_, a)| a.magnitude_squared())
            .sum();

        let r = pseudo_random(&mut 67890u64);
        let result = if r < prob_one { 1u8 } else { 0u8 };

        // Collapse the measured qubit
        let norm_factor = if result == 1 { 1.0 / prob_one.sqrt() } else { 1.0 / (1.0 - prob_one).sqrt() };
        for i in 0..self.amplitudes.len() {
            let bit_set = (i >> qubit) & 1 == 1;
            if (result == 1) != bit_set {
                self.amplitudes[i] = Complex::zero();
            } else {
                self.amplitudes[i] = self.amplitudes[i] * norm_factor;
            }
        }

        result
    }

    /// Probability of measuring a specific qubit as |1⟩.
    pub fn probability_one(&self, qubit: usize) -> f64 {
        let mask = 1 << qubit;
        self.amplitudes.iter().enumerate()
            .filter(|(i, _)| i & mask != 0)
            .map(|(_, a)| a.magnitude_squared())
            .sum()
    }

    /// Get the probability distribution over all basis states.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.magnitude_squared()).collect()
    }

    /// Entangle two qubits using CNOT.
    pub fn cnot(&mut self, control: usize, target: usize) {
        let not_gate = [[Complex::zero(), Complex::one()],
                        [Complex::one(), Complex::zero()]];
        self.apply_two_qubit_gate(control, target, &not_gate);
    }

    /// Create a Bell state (|00⟩ + |11⟩) / √2
    pub fn bell_state() -> Self {
        let mut state = QuantumState::new(2);
        state.hadamard(0);
        state.cnot(0, 1);
        state
    }

    /// Tensor product of two quantum states.
    pub fn tensor_product(a: &Self, b: &Self) -> Self {
        let new_qubits = a.num_qubits + b.num_qubits;
        let size = 1 << new_qubits;
        let mut amplitudes = vec![Complex::zero(); size];

        for (i, ai) in a.amplitudes.iter().enumerate() {
            for (j, bj) in b.amplitudes.iter().enumerate() {
                amplitudes[i * b.amplitudes.len() + j] = *ai * *bj;
            }
        }

        Self { amplitudes, num_qubits: new_qubits }
    }

    /// Trace out (discard) a qubit, returning reduced state.
    pub fn partial_trace(&self, discard_qubit: usize) -> Self {
        let new_qubits = self.num_qubits - 1;
        let new_size = 1 << new_qubits;
        let mut reduced_amps = vec![Complex::zero(); new_size];

        for i in 0..self.amplitudes.len() {
            let bit = (i >> discard_qubit) & 1;
            let mut new_idx = 0;
            for q in 0..self.num_qubits {
                if q == discard_qubit { continue; }
                let q_bit = (i >> q) & 1;
                let new_q = if q > discard_qubit { q - 1 } else { q };
                new_idx |= q_bit << new_q;
            }
            reduced_amps[new_idx] = reduced_amps[new_idx] + self.amplitudes[i].conjugate() * self.amplitudes[i];
        }

        // Take sqrt of diagonal elements (simplified)
        for amp in &mut reduced_amps {
            *amp = Complex::new(amp.re.sqrt(), 0.0);
        }

        Self { amplitudes: reduced_amps, num_qubits: new_qubits }
    }
}

fn pseudo_random(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / (1u64 << 31) as f64
}

// === Standard Quantum Gates ===

/// Pauli-X gate (NOT gate).
pub fn pauli_x() -> [[Complex; 2]; 2] {
    [[Complex::zero(), Complex::one()],
     [Complex::one(), Complex::zero()]]
}

/// Pauli-Y gate.
pub fn pauli_y() -> [[Complex; 2]; 2] {
    [[Complex::zero(), Complex::new(0.0, -1.0)],
     [Complex::new(0.0, 1.0), Complex::zero()]]
}

/// Pauli-Z gate.
pub fn pauli_z() -> [[Complex; 2]; 2] {
    [[Complex::one(), Complex::zero()],
     [Complex::zero(), Complex::new(-1.0, 0.0)]]
}

/// Hadamard gate.
pub fn hadamard() -> [[Complex; 2]; 2] {
    let h = 1.0 / 2.0f64.sqrt();
    [[Complex::new(h, 0.0), Complex::new(h, 0.0)],
     [Complex::new(h, 0.0), Complex::new(-h, 0.0)]]
}

/// Phase gate (S gate).
pub fn phase_gate() -> [[Complex; 2]; 2] {
    [[Complex::one(), Complex::zero()],
     [Complex::zero(), Complex::new(0.0, 1.0)]]
}

/// T gate (π/8 gate).
pub fn t_gate() -> [[Complex; 2]; 2] {
    [[Complex::one(), Complex::zero()],
     [Complex::zero(), Complex::from_polar(1.0, PI / 4.0)]]
}

/// Rotation-X gate by angle theta.
pub fn rx(theta: f64) -> [[Complex; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [[Complex::new(c, 0.0), Complex::new(0.0, -s)],
     [Complex::new(0.0, -s), Complex::new(c, 0.0)]]
}

/// Rotation-Y gate by angle theta.
pub fn ry(theta: f64) -> [[Complex; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [[Complex::new(c, 0.0), Complex::new(-s, 0.0)],
     [Complex::new(s, 0.0), Complex::new(c, 0.0)]]
}

/// Rotation-Z gate by angle theta.
pub fn rz(theta: f64) -> [[Complex; 2]; 2] {
    [[Complex::from_polar(1.0, -theta / 2.0), Complex::zero()],
     [Complex::zero(), Complex::from_polar(1.0, theta / 2.0)]]
}

/// CNOT (Controlled-NOT) gate as a 4x4 matrix.
pub fn cnot_matrix() -> [[Complex; 4]; 4] {
    let mut gate = [[Complex::zero(); 4]; 4];
    gate[0][0] = Complex::one();
    gate[1][1] = Complex::one();
    gate[2][3] = Complex::one();
    gate[3][2] = Complex::one();
    gate
}

/// SWAP gate.
pub fn swap_matrix() -> [[Complex; 4]; 4] {
    let mut gate = [[Complex::zero(); 4]; 4];
    gate[0][0] = Complex::one();
    gate[1][2] = Complex::one();
    gate[2][1] = Complex::one();
    gate[3][3] = Complex::one();
    gate
}

/// Toffoli (CCNOT) gate as 8x8 matrix.
pub fn toffoli_matrix() -> Vec<Vec<Complex>> {
    let mut gate = vec![vec![Complex::zero(); 8]; 8];
    for i in 0..6 {
        gate[i][i] = Complex::one();
    }
    gate[6][7] = Complex::one();
    gate[7][6] = Complex::one();
    gate
}

/// Quantum circuit builder.
pub struct QuantumCircuit {
    num_qubits: usize,
    gates: Vec<CircuitGate>,
}

#[derive(Debug, Clone)]
struct CircuitGate {
    name: String,
    qubits: Vec<usize>,
    matrix: Vec<Vec<Complex>>,
}

impl QuantumCircuit {
    pub fn new(num_qubits: usize) -> Self {
        Self { num_qubits, gates: Vec::new() }
    }

    pub fn hadamard(&mut self, qubit: usize) -> &mut Self {
        let h = hadamard();
        self.gates.push(CircuitGate {
            name: "H".to_string(),
            qubits: vec![qubit],
            matrix: h.iter().map(|row| row.to_vec()).collect(),
        });
        self
    }

    pub fn x(&mut self, qubit: usize) -> &mut Self {
        let x = pauli_x();
        self.gates.push(CircuitGate {
            name: "X".to_string(),
            qubits: vec![qubit],
            matrix: x.iter().map(|row| row.to_vec()).collect(),
        });
        self
    }

    pub fn y(&mut self, qubit: usize) -> &mut Self {
        let y = pauli_y();
        self.gates.push(CircuitGate {
            name: "Y".to_string(),
            qubits: vec![qubit],
            matrix: y.iter().map(|row| row.to_vec()).collect(),
        });
        self
    }

    pub fn z(&mut self, qubit: usize) -> &mut Self {
        let z = pauli_z();
        self.gates.push(CircuitGate {
            name: "Z".to_string(),
            qubits: vec![qubit],
            matrix: z.iter().map(|row| row.to_vec()).collect(),
        });
        self
    }

    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.gates.push(CircuitGate {
            name: "CNOT".to_string(),
            qubits: vec![control, target],
            matrix: Vec::new(), // handled specially
        });
        self
    }

    pub fn measure_all(&mut self) -> &mut Self {
        self.gates.push(CircuitGate {
            name: "MEASURE".to_string(),
            qubits: (0..self.num_qubits).collect(),
            matrix: Vec::new(),
        });
        self
    }

    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    pub fn depth(&self) -> usize {
        // Simplified depth = gate count (actual depth depends on qubit assignments)
        self.gates.len()
    }

    /// Execute the circuit and return measurement results.
    pub fn execute(&self) -> Vec<u8> {
        let mut state = QuantumState::new(self.num_qubits);

        for gate in &self.gates {
            match gate.name.as_str() {
                "CNOT" => {
                    state.cnot(gate.qubits[0], gate.qubits[1]);
                }
                "MEASURE" => {
                    // Don't measure yet, just note it
                }
                _ => {
                    if gate.qubits.len() == 1 && !gate.matrix.is_empty() {
                        let mut arr = [[Complex::zero(); 2]; 2];
                        for i in 0..2 {
                            for j in 0..2 {
                                arr[i][j] = gate.matrix[i][j];
                            }
                        }
                        state.apply_single_gate(gate.qubits[0], &arr);
                    }
                }
            }
        }

        state.measure()
    }

    /// Run the circuit many times and return statistics.
    pub fn run_shots(&self, shots: usize) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for _ in 0..shots {
            let result = self.execute();
            let key: String = result.iter().map(|b| if *b == 0 { '0' } else { '1' }).collect();
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }
}

use std::collections::HashMap;

/// Quantum teleportation protocol.
pub fn teleport(state: &QuantumState) -> QuantumState {
    // Create Bell pair on qubits 1 and 2
    let mut system = QuantumState::tensor_product(state, &QuantumState::bell_state());

    // Alice applies CNOT and Hadamard
    system.cnot(0, 1);
    system.apply_single_gate(0, &hadamard());

    // Measure qubits 0 and 1
    let m0 = system.measure_qubit(0);
    let m1 = system.measure_qubit(1);

    // Bob applies corrections based on measurement
    if m1 == 1 {
        system.apply_single_gate(2, &pauli_x());
    }
    if m0 == 1 {
        system.apply_single_gate(2, &pauli_z());
    }

    system
}

/// Quantum Fourier Transform.
pub fn qft(state: &mut QuantumState) {
    let n = state.num_qubits();
    for i in (0..n).rev() {
        state.apply_single_gate(i, &hadamard());
        for j in 0..i {
            let angle = PI / (1 << (i - j)) as f64;
            let phase = [[Complex::one(), Complex::zero()],
                         [Complex::zero(), Complex::from_polar(1.0, angle)]];
            // Controlled phase (simplified: apply to target with control j)
            state.apply_two_qubit_gate(j, i, &phase);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_single_qubit() {
        let mut state = QuantumState::new(1);
        state.apply_single_gate(0, &hadamard());
        let probs = state.probabilities();
        assert!(approx_eq(probs[0], 0.5, 1e-10));
        assert!(approx_eq(probs[1], 0.5, 1e-10));
    }

    #[test]
    fn test_bell_state() {
        let bell = QuantumState::bell_state();
        let probs = bell.probabilities();
        assert!(approx_eq(probs[0], 0.5, 1e-10));
        assert!(approx_eq(probs[3], 0.5, 1e-10));
        assert!(approx_eq(probs[1], 0.0, 1e-10));
        assert!(approx_eq(probs[2], 0.0, 1e-10));
    }

    #[test]
    fn test_cnot() {
        let mut state = QuantumState::new(2);
        state.apply_single_gate(0, &pauli_x()); // |10⟩
        state.cnot(0, 1);                        // |11⟩
        let probs = state.probabilities();
        assert!(approx_eq(probs[3], 1.0, 1e-10));
    }

    #[test]
    fn test_tensor_product() {
        let a = QuantumState::new(1);
        let b = QuantumState::new(1);
        let ab = QuantumState::tensor_product(&a, &b);
        assert_eq!(ab.num_qubits(), 2);
        assert!(approx_eq(ab.probabilities()[0], 1.0, 1e-10));
    }

    #[test]
    fn test_circuit() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.hadamard(0).cnot(0, 1).measure_all();
        assert_eq!(circuit.gate_count(), 3);
    }
}
