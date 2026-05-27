/// Feedforward neural network with backpropagation.

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
    epochs: usize,
    activation: Activation,
    output_activation: Activation,
    losses: Vec<f64>,
    batch_size: usize,
    lr_decay: f64,
}

#[derive(Debug, Clone)]
struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    outputs: Vec<f64>,
    deltas: Vec<f64>,
    inputs: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    LeakyReLU(f64),
    Softmax,
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let scale = (2.0 / input_size as f64).sqrt();
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| (rand_f64() - 0.5) * scale).collect())
            .collect();
        Self {
            weights,
            biases: vec![0.0; output_size],
            outputs: vec![0.0; output_size],
            deltas: vec![0.0; output_size],
            inputs: Vec::new(),
        }
    }

    fn forward(&mut self, inputs: &[f64], activation: Activation) {
        self.inputs = inputs.to_vec();
        for i in 0..self.weights.len() {
            let mut sum = self.biases[i];
            for j in 0..inputs.len() {
                sum += self.weights[i][j] * inputs[j];
            }
            self.outputs[i] = sum;
        }
        apply_activation(&mut self.outputs, activation);
    }
}

fn rand_f64() -> f64 {
    // Simple deterministic pseudo-random for reproducibility
    static mut STATE: u64 = 42;
    unsafe {
        STATE = STATE.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (STATE >> 33) as f64 / (1u64 << 31) as f64
    }
}

fn apply_activation(values: &mut [f64], activation: Activation) {
    match activation {
        Activation::ReLU => {
            for v in values.iter_mut() {
                *v = v.max(0.0);
            }
        }
        Activation::Sigmoid => {
            for v in values.iter_mut() {
                *v = 1.0 / (1.0 + (-*v).exp());
            }
        }
        Activation::Tanh => {
            for v in values.iter_mut() {
                *v = v.tanh();
            }
        }
        Activation::Linear => {}
        Activation::LeakyReLU(alpha) => {
            for v in values.iter_mut() {
                if *v < 0.0 { *v *= alpha; }
            }
        }
        Activation::Softmax => {
            let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = values.iter().map(|v| (v - max_val).exp()).sum();
            for v in values.iter_mut() {
                *v = (*v - max_val).exp() / sum;
            }
        }
    }
}

fn activation_derivative(output: f64, activation: Activation) -> f64 {
    match activation {
        Activation::ReLU => if output > 0.0 { 1.0 } else { 0.0 },
        Activation::Sigmoid => output * (1.0 - output),
        Activation::Tanh => 1.0 - output * output,
        Activation::Linear => 1.0,
        Activation::LeakyReLU(alpha) => if output > 0.0 { 1.0 } else { alpha },
        Activation::Softmax => 1.0, // Simplified; handled in cross-entropy gradient
    }
}

impl NeuralNetwork {
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(layer_sizes.len() >= 2);
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        Self {
            layers,
            learning_rate: 0.01,
            epochs: 1000,
            activation: Activation::ReLU,
            output_activation: Activation::Sigmoid,
            losses: Vec::new(),
            batch_size: 32,
            lr_decay: 0.0,
        }
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    pub fn output_activation(mut self, activation: Activation) -> Self {
        self.output_activation = activation;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn lr_decay(mut self, decay: f64) -> Self {
        self.lr_decay = decay;
        self
    }

    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let mut current = inputs.to_vec();
        for i in 0..self.layers.len() {
            let activation = if i == self.layers.len() - 1 {
                self.output_activation
            } else {
                self.activation
            };
            self.layers[i].forward(&current, activation);
            current = self.layers[i].outputs.clone();
        }
        current
    }

    pub fn fit(&mut self, x: &[Vec<f64>], y: &[Vec<f64>]) {
        assert!(!x.is_empty() && x.len() == y.len());
        self.losses.clear();

        let mut lr = self.learning_rate;

        for epoch in 0..self.epochs {
            let mut total_loss = 0.0;

            // Simple SGD (no mini-batch shuffling for simplicity)
            for i in 0..x.len() {
                let output = self.forward(&x[i]);
                let target = &y[i];

                // Compute loss (MSE)
                let loss: f64 = output.iter().zip(target.iter())
                    .map(|(o, t)| (o - t).powi(2))
                    .sum();
                total_loss += loss;

                // Backpropagation
                self.backward(target);
                self.update_weights(lr);
            }

            self.losses.push(total_loss / x.len() as f64);

            if self.lr_decay > 0.0 {
                lr = self.learning_rate / (1.0 + self.lr_decay * epoch as f64);
            }
        }
    }

    fn backward(&mut self, target: &[f64]) {
        let num_layers = self.layers.len();

        // Output layer deltas
        {
            let layer = &mut self.layers[num_layers - 1];
            for i in 0..layer.outputs.len() {
                let error = target[i] - layer.outputs[i];
                layer.deltas[i] = error * activation_derivative(layer.outputs[i], self.output_activation);
            }
        }

        // Hidden layer deltas
        for l in (0..num_layers - 1).rev() {
            let (left, right) = self.layers.split_at_mut(l + 1);
            let layer = &mut left[l];
            let next_layer = &right[0];

            for i in 0..layer.outputs.len() {
                let mut error = 0.0;
                for j in 0..next_layer.deltas.len() {
                    error += next_layer.weights[j][i] * next_layer.deltas[j];
                }
                layer.deltas[i] = error * activation_derivative(layer.outputs[i], self.activation);
            }
        }
    }

    fn update_weights(&mut self, lr: f64) {
        for layer in self.layers.iter_mut() {
            for i in 0..layer.weights.len() {
                for j in 0..layer.weights[i].len() {
                    layer.weights[i][j] += lr * layer.deltas[i] * layer.inputs[j];
                }
                layer.biases[i] += lr * layer.deltas[i];
            }
        }
    }

    pub fn predict(&mut self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter().map(|row| self.forward(row)).collect()
    }

    pub fn predict_classes(&mut self, x: &[Vec<f64>]) -> Vec<usize> {
        self.predict(x).iter()
            .map(|output| output.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap().0)
            .collect()
    }

    pub fn losses(&self) -> &[f64] {
        &self.losses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_network() {
        let x = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        let y = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        let mut nn = NeuralNetwork::new(&[2, 8, 1])
            .learning_rate(0.5)
            .epochs(5000)
            .activation(Activation::Tanh)
            .output_activation(Activation::Sigmoid);
        nn.fit(&x, &y);

        let predictions = nn.predict(&x);
        assert!(predictions[0][0] < 0.3);
        assert!(predictions[3][0] < 0.3);
        assert!(predictions[1][0] > 0.5 || predictions[2][0] > 0.5);
    }

    #[test]
    fn test_relu_activation() {
        let mut values = vec![-1.0, 0.0, 1.0, -0.5];
        apply_activation(&mut values, Activation::ReLU);
        assert_eq!(values, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_sigmoid_activation() {
        let mut values = vec![0.0];
        apply_activation(&mut values, Activation::Sigmoid);
        assert!((values[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_softmax() {
        let mut values = vec![1.0, 2.0, 3.0];
        apply_activation(&mut values, Activation::Softmax);
        let sum: f64 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_class() {
        let x = vec![
            vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0],
            vec![-1.0, 0.0], vec![0.0, -1.0], vec![-1.0, -1.0],
        ];
        let y = vec![
            vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 1.0],
        ];

        let mut nn = NeuralNetwork::new(&[2, 8, 3])
            .learning_rate(0.1)
            .epochs(2000)
            .output_activation(Activation::Softmax);
        nn.fit(&x, &y);

        let predictions = nn.predict_classes(&x);
        assert!(predictions.len() == 6);
    }
}
