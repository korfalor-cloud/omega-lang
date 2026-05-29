/// Neural network from scratch: layers, activations, loss functions, backpropagation.

use std::f64::consts::PI;

// ─── Activation Functions ───

pub trait Activation: Clone {
    fn forward(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}

#[derive(Clone)]
pub struct ReLU;
impl Activation for ReLU {
    fn forward(&self, x: f64) -> f64 { x.max(0.0) }
    fn derivative(&self, x: f64) -> f64 { if x > 0.0 { 1.0 } else { 0.0 } }
}

#[derive(Clone)]
pub struct LeakyReLU(pub f64);
impl Activation for LeakyReLU {
    fn forward(&self, x: f64) -> f64 { if x > 0.0 { x } else { self.0 * x } }
    fn derivative(&self, x: f64) -> f64 { if x > 0.0 { 1.0 } else { self.0 } }
}

#[derive(Clone)]
pub struct Sigmoid;
impl Activation for Sigmoid {
    fn forward(&self, x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
    fn derivative(&self, x: f64) -> f64 { let s = self.forward(x); s * (1.0 - s) }
}

#[derive(Clone)]
pub struct Tanh;
impl Activation for Tanh {
    fn forward(&self, x: f64) -> f64 { x.tanh() }
    fn derivative(&self, x: f64) -> f64 { 1.0 - x.tanh().powi(2) }
}

#[derive(Clone)]
pub struct ELU(pub f64);
impl Activation for ELU {
    fn forward(&self, x: f64) -> f64 { if x > 0.0 { x } else { self.0 * (x.exp() - 1.0) } }
    fn derivative(&self, x: f64) -> f64 { if x > 0.0 { 1.0 } else { self.0 * x.exp() } }
}

#[derive(Clone)]
pub struct Swish;
impl Activation for Swish {
    fn forward(&self, x: f64) -> f64 { x / (1.0 + (-x).exp()) }
    fn derivative(&self, x: f64) -> f64 {
        let s = 1.0 / (1.0 + (-x).exp());
        s + x * s * (1.0 - s)
    }
}

#[derive(Clone)]
pub struct GELU;
impl Activation for GELU {
    fn forward(&self, x: f64) -> f64 {
        0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }
    fn derivative(&self, x: f64) -> f64 {
        let c = (2.0 / PI).sqrt();
        let inner = c * (x + 0.044715 * x.powi(3));
        let tanh_val = inner.tanh();
        let sech2 = 1.0 - tanh_val * tanh_val;
        0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x * x)
    }
}

#[derive(Clone)]
pub struct Softplus;
impl Activation for Softplus {
    fn forward(&self, x: f64) -> f64 { (1.0 + x.exp()).ln() }
    fn derivative(&self, x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
}

// ─── Loss Functions ───

pub trait Loss {
    fn forward(&self, predicted: &[f64], target: &[f64]) -> f64;
    fn derivative(&self, predicted: &[f64], target: &[f64]) -> Vec<f64>;
}

#[derive(Clone)]
pub struct MSELoss;
impl Loss for MSELoss {
    fn forward(&self, predicted: &[f64], target: &[f64]) -> f64 {
        predicted.iter().zip(target.iter()).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / predicted.len() as f64
    }
    fn derivative(&self, predicted: &[f64], target: &[f64]) -> Vec<f64> {
        let n = predicted.len() as f64;
        predicted.iter().zip(target.iter()).map(|(p, t)| 2.0 * (p - t) / n).collect()
    }
}

#[derive(Clone)]
pub struct MAELoss;
impl Loss for MAELoss {
    fn forward(&self, predicted: &[f64], target: &[f64]) -> f64 {
        predicted.iter().zip(target.iter()).map(|(p, t)| (p - t).abs()).sum::<f64>() / predicted.len() as f64
    }
    fn derivative(&self, predicted: &[f64], target: &[f64]) -> Vec<f64> {
        let n = predicted.len() as f64;
        predicted.iter().zip(target.iter()).map(|(p, t)| {
            if p > t { 1.0 / n } else if p < t { -1.0 / n } else { 0.0 }
        }).collect()
    }
}

#[derive(Clone)]
pub struct HuberLoss(pub f64);
impl Loss for HuberLoss {
    fn forward(&self, predicted: &[f64], target: &[f64]) -> f64 {
        predicted.iter().zip(target.iter()).map(|(p, t)| {
            let d = (p - t).abs();
            if d <= self.0 { 0.5 * d * d } else { self.0 * (d - 0.5 * self.0) }
        }).sum::<f64>() / predicted.len() as f64
    }
    fn derivative(&self, predicted: &[f64], target: &[f64]) -> Vec<f64> {
        let n = predicted.len() as f64;
        predicted.iter().zip(target.iter()).map(|(p, t)| {
            let d = p - t;
            if d.abs() <= self.0 { d / n } else { self.0 * d.signum() / n }
        }).collect()
    }
}

#[derive(Clone)]
pub struct CrossEntropyLoss;
impl Loss for CrossEntropyLoss {
    fn forward(&self, predicted: &[f64], target: &[f64]) -> f64 {
        -predicted.iter().zip(target.iter())
            .map(|(p, t)| t * (p.max(1e-15)).ln())
            .sum::<f64>()
    }
    fn derivative(&self, predicted: &[f64], target: &[f64]) -> Vec<f64> {
        predicted.iter().zip(target.iter())
            .map(|(p, t)| -t / p.max(1e-15))
            .collect()
    }
}

#[derive(Clone)]
pub struct BinaryCrossEntropyLoss;
impl Loss for BinaryCrossEntropyLoss {
    fn forward(&self, predicted: &[f64], target: &[f64]) -> f64 {
        -predicted.iter().zip(target.iter())
            .map(|(p, t)| {
                let p = p.max(1e-15).min(1.0 - 1e-15);
                t * p.ln() + (1.0 - t) * (1.0 - p).ln()
            })
            .sum::<f64>() / predicted.len() as f64
    }
    fn derivative(&self, predicted: &[f64], target: &[f64]) -> Vec<f64> {
        let n = predicted.len() as f64;
        predicted.iter().zip(target.iter())
            .map(|(p, t)| {
                let p = p.max(1e-15).min(1.0 - 1e-15);
                (-t / p + (1.0 - t) / (1.0 - p)) / n
            })
            .collect()
    }
}

// ─── Layer ───

#[derive(Clone)]
pub struct DenseLayer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: Option<Box<dyn Activation>>,
    // Cache for backprop
    pub last_input: Vec<f64>,
    pub last_z: Vec<f64>,
    pub last_output: Vec<f64>,
    // Gradients
    pub weight_grads: Vec<Vec<f64>>,
    pub bias_grads: Vec<f64>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation: Option<Box<dyn Activation>>) -> Self {
        // Xavier initialization
        let scale = (2.0 / input_size as f64).sqrt();
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((seed >> 33) as f64) / (1u64 << 31) as f64;
            (u * 2.0 - 1.0) * scale
        };

        let weights = (0..output_size).map(|_| {
            (0..input_size).map(|_| rand()).collect()
        }).collect();

        Self {
            weights,
            biases: vec![0.0; output_size],
            activation,
            last_input: Vec::new(),
            last_z: Vec::new(),
            last_output: Vec::new(),
            weight_grads: vec![vec![0.0; input_size]; output_size],
            bias_grads: vec![0.0; output_size],
        }
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.last_input = input.to_vec();

        // z = W * x + b
        let z: Vec<f64> = self.weights.iter().zip(self.biases.iter()).map(|(w, &b)| {
            w.iter().zip(input.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b
        }).collect();

        self.last_z = z.clone();

        // Apply activation
        let output = if let Some(ref act) = self.activation {
            z.iter().map(|&x| act.forward(x)).collect()
        } else {
            z
        };

        self.last_output = output.clone();
        output
    }

    pub fn backward(&mut self, grad_output: &[f64]) -> Vec<f64> {
        let n = self.weights.len();    // output size
        let m = self.weights[0].len(); // input size

        // Apply activation derivative
        let grad_z: Vec<f64> = if let Some(ref act) = self.activation {
            grad_output.iter().zip(self.last_z.iter())
                .map(|(g, &z)| g * act.derivative(z))
                .collect()
        } else {
            grad_output.to_vec()
        };

        // Weight gradients: dW = grad_z * input^T
        for i in 0..n {
            for j in 0..m {
                self.weight_grads[i][j] = grad_z[i] * self.last_input[j];
            }
            self.bias_grads[i] = grad_z[i];
        }

        // Input gradients: dx = W^T * grad_z
        let mut grad_input = vec![0.0; m];
        for j in 0..m {
            for i in 0..n {
                grad_input[j] += self.weights[i][j] * grad_z[i];
            }
        }

        grad_input
    }

    pub fn zero_grads(&mut self) {
        for row in &mut self.weight_grads {
            for v in row.iter_mut() { *v = 0.0; }
        }
        for v in &mut self.bias_grads { *v = 0.0; }
    }
}

// ─── Optimizers ───

pub trait Optimizer {
    fn step(&mut self, layers: &mut [DenseLayer], learning_rate: f64);
}

#[derive(Clone)]
pub struct SGDMomentum {
    pub momentum: f64,
    pub velocity_w: Vec<Vec<Vec<f64>>>,
    pub velocity_b: Vec<Vec<f64>>,
}

impl SGDMomentum {
    pub fn new(momentum: f64) -> Self {
        Self { momentum, velocity_w: Vec::new(), velocity_b: Vec::new() }
    }
}

impl Optimizer for SGDMomentum {
    fn step(&mut self, layers: &mut [DenseLayer], learning_rate: f64) {
        if self.velocity_w.is_empty() {
            self.velocity_w = layers.iter().map(|l| {
                l.weight_grads.iter().map(|row| vec![0.0; row.len()]).collect()
            }).collect();
            self.velocity_b = layers.iter().map(|l| vec![0.0; l.biases.len()]).collect();
        }

        for (i, layer) in layers.iter_mut().enumerate() {
            for j in 0..layer.weights.len() {
                for k in 0..layer.weights[j].len() {
                    self.velocity_w[i][j][k] = self.momentum * self.velocity_w[i][j][k] - learning_rate * layer.weight_grads[j][k];
                    layer.weights[j][k] += self.velocity_w[i][j][k];
                }
                self.velocity_b[i][j] = self.momentum * self.velocity_b[i][j] - learning_rate * layer.bias_grads[j];
                layer.biases[j] += self.velocity_b[i][j];
            }
        }
    }
}

#[derive(Clone)]
pub struct Adam {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub t: usize,
    pub m_w: Vec<Vec<Vec<f64>>>,
    pub v_w: Vec<Vec<Vec<f64>>>,
    pub m_b: Vec<Vec<f64>>,
    pub v_b: Vec<Vec<f64>>,
}

impl Adam {
    pub fn new(beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            beta1, beta2, epsilon, t: 0,
            m_w: Vec::new(), v_w: Vec::new(),
            m_b: Vec::new(), v_b: Vec::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, layers: &mut [DenseLayer], learning_rate: f64) {
        if self.m_w.is_empty() {
            self.m_w = layers.iter().map(|l| {
                l.weight_grads.iter().map(|row| vec![0.0; row.len()]).collect()
            }).collect();
            self.v_w = self.m_w.clone();
            self.m_b = layers.iter().map(|l| vec![0.0; l.biases.len()]).collect();
            self.v_b = self.m_b.clone();
        }

        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, layer) in layers.iter_mut().enumerate() {
            for j in 0..layer.weights.len() {
                for k in 0..layer.weights[j].len() {
                    let g = layer.weight_grads[j][k];
                    self.m_w[i][j][k] = self.beta1 * self.m_w[i][j][k] + (1.0 - self.beta1) * g;
                    self.v_w[i][j][k] = self.beta2 * self.v_w[i][j][k] + (1.0 - self.beta2) * g * g;
                    let m_hat = self.m_w[i][j][k] / bc1;
                    let v_hat = self.v_w[i][j][k] / bc2;
                    layer.weights[j][k] -= learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                }
                let g = layer.bias_grads[j];
                self.m_b[i][j] = self.beta1 * self.m_b[i][j] + (1.0 - self.beta1) * g;
                self.v_b[i][j] = self.beta2 * self.v_b[i][j] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m_b[i][j] / bc1;
                let v_hat = self.v_b[i][j] / bc2;
                layer.biases[j] -= learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
    }
}

// ─── Batch Normalization ───

#[derive(Clone)]
pub struct BatchNorm {
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
    pub momentum: f64,
    pub epsilon: f64,
    pub size: usize,
    // Cache
    pub last_norm: Vec<f64>,
    pub last_inv_std: f64,
}

impl BatchNorm {
    pub fn new(size: usize) -> Self {
        Self {
            gamma: vec![1.0; size],
            beta: vec![0.0; size],
            running_mean: vec![0.0; size],
            running_var: vec![1.0; size],
            momentum: 0.1,
            epsilon: 1e-5,
            size,
            last_norm: Vec::new(),
            last_inv_std: 0.0,
        }
    }

    pub fn forward(&mut self, input: &[f64], training: bool) -> Vec<f64> {
        if training {
            let mean: f64 = input.iter().sum::<f64>() / input.len() as f64;
            let var: f64 = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / input.len() as f64;
            let inv_std = 1.0 / (var + self.epsilon).sqrt();

            self.last_norm = input.iter().map(|x| (x - mean) * inv_std).collect();
            self.last_inv_std = inv_std;

            // Update running stats
            for i in 0..self.size.min(input.len()) {
                self.running_mean[i] = (1.0 - self.momentum) * self.running_mean[i] + self.momentum * mean;
                self.running_var[i] = (1.0 - self.momentum) * self.running_var[i] + self.momentum * var;
            }

            input.iter().enumerate().map(|(i, &x)| {
                self.gamma[i] * (x - mean) * inv_std + self.beta[i]
            }).collect()
        } else {
            input.iter().enumerate().map(|(i, &x)| {
                let inv_std = 1.0 / (self.running_var[i] + self.epsilon).sqrt();
                self.gamma[i] * (x - self.running_mean[i]) * inv_std + self.beta[i]
            }).collect()
        }
    }
}

// ─── Dropout ───

#[derive(Clone)]
pub struct Dropout {
    pub rate: f64,
    seed: u64,
}

impl Dropout {
    pub fn new(rate: f64) -> Self {
        Self { rate, seed: 42 }
    }

    pub fn forward(&mut self, input: &[f64], training: bool) -> Vec<f64> {
        if !training || self.rate <= 0.0 {
            return input.to_vec();
        }
        let scale = 1.0 / (1.0 - self.rate);
        input.iter().map(|&x| {
            self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((self.seed >> 33) as f64) / (1u64 << 31) as f64;
            if r > self.rate { x * scale } else { 0.0 }
        }).collect()
    }
}

// ─── Neural Network ───

pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>,
    pub loss: Box<dyn Loss>,
    pub optimizer: Box<dyn Optimizer>,
    pub learning_rate: f64,
    pub training_history: Vec<f64>,
}

impl NeuralNetwork {
    pub fn new(loss: Box<dyn Loss>, optimizer: Box<dyn Optimizer>, learning_rate: f64) -> Self {
        Self {
            layers: Vec::new(),
            loss,
            optimizer,
            learning_rate,
            training_history: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, layer: DenseLayer) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current
    }

    pub fn backward(&mut self, grad: &[f64]) {
        let mut current_grad = grad.to_vec();
        for layer in self.layers.iter_mut().rev() {
            current_grad = layer.backward(&current_grad);
        }
    }

    pub fn train_step(&mut self, input: &[f64], target: &[f64]) -> f64 {
        let output = self.forward(input);
        let loss_val = self.loss.forward(&output, target);
        let grad = self.loss.derivative(&output, target);
        self.backward(&grad);
        self.optimizer.step(&mut self.layers, self.learning_rate);
        loss_val
    }

    pub fn fit(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], epochs: usize, batch_size: usize) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut count = 0;

            for (input, target) in inputs.iter().zip(targets.iter()) {
                total_loss += self.train_step(input, target);
                count += 1;

                if count % batch_size == 0 {
                    for layer in &mut self.layers {
                        layer.zero_grads();
                    }
                }
            }

            let avg_loss = total_loss / count as f64;
            self.training_history.push(avg_loss);

            if epoch % 100 == 0 {
                // Log progress (in a real implementation)
            }
        }
    }

    pub fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        self.forward(input)
    }

    pub fn predict_batch(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        inputs.iter().map(|input| self.forward(input)).collect()
    }
}

// ─── Convolutional Layer ───

#[derive(Clone)]
pub struct Conv1D {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub kernels: Vec<Vec<Vec<f64>>>, // [out_ch][in_ch][kernel]
    pub biases: Vec<f64>,
    pub kernel_grads: Vec<Vec<Vec<f64>>>,
    pub bias_grads: Vec<f64>,
    pub last_input: Vec<Vec<f64>>,
}

impl Conv1D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 - 0.5
        };

        let kernels = (0..out_channels).map(|_| {
            (0..in_channels).map(|_| {
                (0..kernel_size).map(|_| rand()).collect()
            }).collect()
        }).collect();

        Self {
            in_channels, out_channels, kernel_size, stride, padding,
            kernels,
            biases: vec![0.0; out_channels],
            kernel_grads: vec![vec![vec![0.0; kernel_size]; in_channels]; out_channels],
            bias_grads: vec![0.0; out_channels],
            last_input: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.last_input = input.to_vec();
        let len = input[0].len();
        let padded_len = len + 2 * self.padding;
        let out_len = (padded_len - self.kernel_size) / self.stride + 1;

        // Pad input
        let padded: Vec<Vec<f64>> = input.iter().map(|ch| {
            let mut p = vec![0.0; self.padding];
            p.extend(ch);
            p.extend(vec![0.0; self.padding]);
            p
        }).collect();

        let mut output = vec![vec![0.0; out_len]; self.out_channels];

        for oc in 0..self.out_channels {
            for i in 0..out_len {
                let start = i * self.stride;
                let mut sum = self.biases[oc];
                for ic in 0..self.in_channels {
                    for k in 0..self.kernel_size {
                        sum += self.kernels[oc][ic][k] * padded[ic][start + k];
                    }
                }
                output[oc][i] = sum;
            }
        }

        output
    }

    pub fn backward(&mut self, grad_output: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let len = self.last_input[0].len();
        let padded_len = len + 2 * self.padding;
        let out_len = grad_output[0].len();

        // Pad input
        let padded: Vec<Vec<f64>> = self.last_input.iter().map(|ch| {
            let mut p = vec![0.0; self.padding];
            p.extend(ch);
            p.extend(vec![0.0; self.padding]);
            p
        }).collect();

        let mut grad_padded = vec![vec![0.0; padded_len]; self.in_channels];

        for oc in 0..self.out_channels {
            self.bias_grads[oc] += grad_output[oc].iter().sum::<f64>();
            for i in 0..out_len {
                let start = i * self.stride;
                for ic in 0..self.in_channels {
                    for k in 0..self.kernel_size {
                        self.kernel_grads[oc][ic][k] += grad_output[oc][i] * padded[ic][start + k];
                        grad_padded[ic][start + k] += self.kernels[oc][ic][k] * grad_output[oc][i];
                    }
                }
            }
        }

        // Remove padding
        grad_padded.iter().map(|ch| {
            ch[self.padding..ch.len() - self.padding].to_vec()
        }).collect()
    }
}

// ─── Recurrent Layer ───

#[derive(Clone)]
pub struct RNNLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub w_ih: Vec<Vec<f64>>, // input to hidden
    pub w_hh: Vec<Vec<f64>>, // hidden to hidden
    pub b_h: Vec<f64>,
    pub w_ho: Vec<Vec<f64>>, // hidden to output
    pub b_o: Vec<f64>,
    // Cache
    pub hidden_states: Vec<Vec<f64>>,
    pub inputs: Vec<Vec<f64>>,
}

impl RNNLayer {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = |scale: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * scale - scale / 2.0
        };

        let scale_ih = (1.0 / input_size as f64).sqrt();
        let scale_hh = (1.0 / hidden_size as f64).sqrt();

        Self {
            input_size, hidden_size,
            w_ih: (0..hidden_size).map(|_| (0..input_size).map(|_| rand(scale_ih)).collect()).collect(),
            w_hh: (0..hidden_size).map(|_| (0..hidden_size).map(|_| rand(scale_hh)).collect()).collect(),
            b_h: vec![0.0; hidden_size],
            w_ho: (0..output_size).map(|_| (0..hidden_size).map(|_| rand(scale_hh)).collect()).collect(),
            b_o: vec![0.0; output_size],
            hidden_states: Vec::new(),
            inputs: Vec::new(),
        }
    }

    pub fn forward(&mut self, sequence: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.hidden_states.clear();
        self.inputs = sequence.to_vec();
        let mut h = vec![0.0; self.hidden_size];
        self.hidden_states.push(h.clone());

        let mut outputs = Vec::new();

        for input in sequence {
            // h = tanh(W_ih * x + W_hh * h + b_h)
            let mut new_h = vec![0.0; self.hidden_size];
            for i in 0..self.hidden_size {
                let mut sum = self.b_h[i];
                for j in 0..self.input_size {
                    sum += self.w_ih[i][j] * input[j];
                }
                for j in 0..self.hidden_size {
                    sum += self.w_hh[i][j] * h[j];
                }
                new_h[i] = sum.tanh();
            }
            h = new_h;
            self.hidden_states.push(h.clone());

            // output = W_ho * h + b_o
            let output: Vec<f64> = self.w_ho.iter().zip(self.b_o.iter()).map(|(wo, &bo)| {
                wo.iter().zip(h.iter()).map(|(w, hi)| w * hi).sum::<f64>() + bo
            }).collect();
            outputs.push(output);
        }

        outputs
    }
}

// ─── LSTM Layer ───

#[derive(Clone)]
pub struct LSTMLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub w: Vec<Vec<f64>>, // Combined weights [4*hidden_size][input_size + hidden_size]
    pub b: Vec<f64>,
    pub outputs: Vec<Vec<f64>>,
    pub cell_states: Vec<Vec<f64>>,
}

impl LSTMLayer {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let combined = input_size + hidden_size;
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * (2.0 / combined as f64).sqrt() - (1.0 / combined as f64).sqrt()
        };

        Self {
            input_size, hidden_size,
            w: (0..4 * hidden_size).map(|_| (0..combined).map(|_| rand()).collect()).collect(),
            b: vec![0.0; 4 * hidden_size],
            outputs: Vec::new(),
            cell_states: Vec::new(),
        }
    }

    pub fn forward(&mut self, sequence: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.outputs.clear();
        self.cell_states.clear();

        let mut h = vec![0.0; self.hidden_size];
        let mut c = vec![0.0; self.hidden_size];
        self.outputs.push(h.clone());
        self.cell_states.push(c.clone());

        let mut outputs = Vec::new();

        for input in sequence {
            // Concatenate input and hidden state
            let mut combined = input.clone();
            combined.extend_from_slice(&h);

            // Compute gates
            let gates: Vec<f64> = self.w.iter().zip(self.b.iter()).map(|(wi, &bi)| {
                wi.iter().zip(combined.iter()).map(|(w, x)| w * x).sum::<f64>() + bi
            }).collect();

            let hs = self.hidden_size;
            let i_gate: Vec<f64> = gates[0..hs].iter().map(|&x| sigmoid(x)).collect();
            let f_gate: Vec<f64> = gates[hs..2*hs].iter().map(|&x| sigmoid(x)).collect();
            let g_gate: Vec<f64> = gates[2*hs..3*hs].iter().map(|&x| x.tanh()).collect();
            let o_gate: Vec<f64> = gates[3*hs..4*hs].iter().map(|&x| sigmoid(x)).collect();

            // Update cell state and hidden state
            c = f_gate.iter().zip(c.iter()).map(|(f, ci)| f * ci).zip(
                i_gate.iter().zip(g_gate.iter()).map(|(ig, g)| ig * g)
            ).map(|(a, b)| a + b).collect();

            h = o_gate.iter().zip(c.iter()).map(|(o, ci)| o * ci.tanh()).collect();

            self.outputs.push(h.clone());
            self.cell_states.push(c.clone());
            outputs.push(h.clone());
        }

        outputs
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ─── Transformer ───

#[derive(Clone)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    pub w_q: Vec<Vec<f64>>,
    pub w_k: Vec<Vec<f64>>,
    pub w_v: Vec<Vec<f64>>,
    pub w_o: Vec<Vec<f64>>,
}

impl MultiHeadAttention {
    pub fn new(num_heads: usize, d_model: usize) -> Self {
        assert_eq!(d_model % num_heads, 0);
        let d_k = d_model / num_heads;
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 * (2.0 / d_model as f64).sqrt() - (1.0 / d_model as f64).sqrt()
        };

        let make_matrix = || -> Vec<Vec<f64>> {
            (0..d_model).map(|_| (0..d_model).map(|_| rand()).collect()).collect()
        };

        Self {
            num_heads, d_model, d_k,
            w_q: make_matrix(),
            w_k: make_matrix(),
            w_v: make_matrix(),
            w_o: make_matrix(),
        }
    }

    pub fn forward(&self, query: &[Vec<f64>], key: &[Vec<f64>], value: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = query.len();
        let d_k = self.d_k;

        // Project Q, K, V
        let q_proj: Vec<Vec<f64>> = query.iter().map(|x| mat_vec_mul(&self.w_q, x)).collect();
        let k_proj: Vec<Vec<f64>> = key.iter().map(|x| mat_vec_mul(&self.w_k, x)).collect();
        let v_proj: Vec<Vec<f64>> = value.iter().map(|x| mat_vec_mul(&self.w_v, x)).collect();

        let mut concat_output = vec![vec![0.0; self.d_model]; seq_len];

        for h in 0..self.num_heads {
            let start = h * d_k;
            let end = start + d_k;

            // Extract head slices
            let q_head: Vec<Vec<f64>> = q_proj.iter().map(|x| x[start..end].to_vec()).collect();
            let k_head: Vec<Vec<f64>> = k_proj.iter().map(|x| x[start..end].to_vec()).collect();
            let v_head: Vec<Vec<f64>> = v_proj.iter().map(|x| x[start..end].to_vec()).collect();

            // Attention scores: Q * K^T / sqrt(d_k)
            let scale = 1.0 / (d_k as f64).sqrt();
            for i in 0..seq_len {
                let mut scores: Vec<f64> = (0..seq_len).map(|j| {
                    q_head[i].iter().zip(k_head[j].iter()).map(|(a, b)| a * b).sum::<f64>() * scale
                }).collect();

                // Softmax
                let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                for s in &mut scores { *s = (*s - max_score).exp(); }
                let sum: f64 = scores.iter().sum();
                for s in &mut scores { *s /= sum; }

                // Weighted sum of values
                for j in 0..seq_len {
                    for k in 0..d_k {
                        concat_output[i][start + k] += scores[j] * v_head[j][k];
                    }
                }
            }
        }

        // Output projection
        concat_output.iter().map(|x| mat_vec_mul(&self.w_o, x)).collect()
    }
}

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
}

// ─── K-means Clustering ───

pub struct KMeans {
    pub k: usize,
    pub centroids: Vec<Vec<f64>>,
    pub max_iter: usize,
    pub tolerance: f64,
}

impl KMeans {
    pub fn new(k: usize, max_iter: usize) -> Self {
        Self {
            k,
            centroids: Vec::new(),
            max_iter,
            tolerance: 1e-4,
        }
    }

    pub fn fit(&mut self, data: &[Vec<f64>]) -> Vec<usize> {
        let n = data.len();
        let dim = data[0].len();

        // Initialize centroids (k-means++ style)
        let mut seed = 42u64;
        let mut rand_f64 = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64
        };

        self.centroids.clear();
        let first = (rand_f64() * n as f64) as usize % n;
        self.centroids.push(data[first].clone());

        while self.centroids.len() < self.k {
            let distances: Vec<f64> = data.iter().map(|point| {
                self.centroids.iter().map(|c| euclidean_dist(point, c)).fold(f64::INFINITY, f64::min).powi(2)
            }).collect();
            let total: f64 = distances.iter().sum();
            let mut r = rand_f64() * total;
            for (i, &d) in distances.iter().enumerate() {
                r -= d;
                if r <= 0.0 {
                    self.centroids.push(data[i].clone());
                    break;
                }
            }
        }

        let mut assignments = vec![0; n];

        for _ in 0..self.max_iter {
            // Assign to nearest centroid
            for (i, point) in data.iter().enumerate() {
                let (min_idx, _) = self.centroids.iter().enumerate()
                    .min_by(|(_, a), (_, b)| euclidean_dist(point, a).partial_cmp(&euclidean_dist(point, b)).unwrap())
                    .unwrap();
                assignments[i] = min_idx;
            }

            // Update centroids
            let mut new_centroids = vec![vec![0.0; dim]; self.k];
            let mut counts = vec![0usize; self.k];
            for (i, point) in data.iter().enumerate() {
                let c = assignments[i];
                counts[c] += 1;
                for j in 0..dim {
                    new_centroids[c][j] += point[j];
                }
            }
            for c in 0..self.k {
                if counts[c] > 0 {
                    for j in 0..dim {
                        new_centroids[c][j] /= counts[c] as f64;
                    }
                }
            }

            // Check convergence
            let max_shift = self.centroids.iter().zip(new_centroids.iter())
                .map(|(old, new)| euclidean_dist(old, new))
                .fold(0.0f64, f64::max);

            self.centroids = new_centroids;
            if max_shift < self.tolerance { break; }
        }

        assignments
    }

    pub fn predict(&self, point: &[f64]) -> usize {
        self.centroids.iter().enumerate()
            .min_by(|(_, a), (_, b)| euclidean_dist(point, a).partial_cmp(&euclidean_dist(point, b)).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    pub fn inertia(&self, data: &[Vec<f64>], assignments: &[usize]) -> f64 {
        data.iter().zip(assignments.iter())
            .map(|(p, &c)| euclidean_dist(p, &self.centroids[c]).powi(2))
            .sum()
    }
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_layer() {
        let mut layer = DenseLayer::new(3, 2, Some(Box::new(ReLU)));
        let input = vec![1.0, 2.0, 3.0];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_xor_network() {
        let mut net = NeuralNetwork::new(
            Box::new(MSELoss),
            Box::new(Adam::new(0.9, 0.999, 1e-8)),
            0.01,
        );
        net.add_layer(DenseLayer::new(2, 8, Some(Box::new(Tanh))));
        net.add_layer(DenseLayer::new(8, 1, Some(Box::new(Sigmoid))));

        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![0.0],
        ];

        // Train
        for _ in 0..1000 {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                net.train_step(input, target);
            }
        }

        // Check predictions
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let pred = net.predict(input);
            assert!((pred[0] - target[0]).abs() < 0.3, "Input: {:?}, Pred: {}, Target: {}", input, pred[0], target[0]);
        }
    }

    #[test]
    fn test_lstm() {
        let mut lstm = LSTMLayer::new(3, 4);
        let sequence = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let outputs = lstm.forward(&sequence);
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].len(), 4);
    }

    #[test]
    fn test_attention() {
        let attn = MultiHeadAttention::new(2, 4);
        let seq = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let output = attn.forward(&seq, &seq, &seq);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 4);
    }

    #[test]
    fn test_kmeans() {
        let mut km = KMeans::new(2, 100);
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![5.0, 5.0],
            vec![5.1, 5.1],
            vec![5.2, 5.0],
        ];
        let assignments = km.fit(&data);
        // Points near (0,0) should be in one cluster, points near (5,5) in another
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[3], assignments[4]);
        assert_ne!(assignments[0], assignments[3]);
    }
}
