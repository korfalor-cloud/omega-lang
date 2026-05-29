/// Tensor operations: multi-dimensional arrays with broadcasting, einsum, and common operations.

use std::ops::{Add, Mul, Sub};
use std::fmt;

/// A multi-dimensional tensor.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        assert_eq!(data.len(), shape.iter().product::<usize>(), "Data length doesn't match shape");
        Self { data, shape, strides }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::new(vec![1.0; size], shape)
    }

    pub fn fill(shape: Vec<usize>, value: f64) -> Self {
        let size = shape.iter().product();
        Self::new(vec![value; size], shape)
    }

    pub fn randn(shape: Vec<usize>, seed: u64) -> Self {
        let size = shape.iter().product();
        let mut s = seed;
        let data: Vec<f64> = (0..size).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = ((s >> 33) as f64) / (1u64 << 31) as f64;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = ((s >> 33) as f64) / (1u64 << 31) as f64;
            (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }).collect();
        Self::new(data, shape)
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    fn index_to_offset(&self, indices: &[usize]) -> usize {
        indices.iter().zip(self.strides.iter()).map(|(i, s)| i * s).sum()
    }

    pub fn get(&self, indices: &[usize]) -> f64 {
        self.data[self.index_to_offset(indices)]
    }

    pub fn set(&mut self, indices: &[usize], value: f64) {
        let offset = self.index_to_offset(indices);
        self.data[offset] = value;
    }

    /// Reshape tensor (total size must match).
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        assert_eq!(self.data.len(), new_shape.iter().product::<usize>(), "Shape mismatch");
        Self::new(self.data.clone(), new_shape)
    }

    /// Transpose (reverse dimensions).
    pub fn transpose(&self) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.reverse();
        let new_strides = Self::compute_strides(&new_shape);

        let mut new_data = vec![0.0; self.data.len()];
        let mut indices = vec![0usize; self.ndim()];

        for i in 0..self.data.len() {
            // Convert flat index to multi-dimensional indices
            let mut temp = i;
            for d in 0..self.ndim() {
                indices[d] = temp / self.strides[d];
                temp %= self.strides[d];
            }

            // Reverse indices for transpose
            let mut new_offset = 0;
            for d in 0..self.ndim() {
                new_offset += indices[d] * new_strides[d];
            }
            new_data[new_offset] = self.data[i];
        }

        Self::new(new_data, new_shape)
    }

    /// Matrix multiplication (2D only).
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2);
        assert_eq!(other.ndim(), 2);
        assert_eq!(self.shape[1], other.shape[0]);

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Tensor::new(result, vec![m, n])
    }

    /// Element-wise operations with broadcasting.
    pub fn broadcast_add(&self, other: &Tensor) -> Tensor {
        let (result_shape, a_data, b_data) = self.broadcast_tensors(self, other);
        let data = a_data.iter().zip(b_data.iter()).map(|(a, b)| a + b).collect();
        Tensor::new(data, result_shape)
    }

    pub fn broadcast_sub(&self, other: &Tensor) -> Tensor {
        let (result_shape, a_data, b_data) = self.broadcast_tensors(self, other);
        let data = a_data.iter().zip(b_data.iter()).map(|(a, b)| a - b).collect();
        Tensor::new(data, result_shape)
    }

    pub fn broadcast_mul(&self, other: &Tensor) -> Tensor {
        let (result_shape, a_data, b_data) = self.broadcast_tensors(self, other);
        let data = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).collect();
        Tensor::new(data, result_shape)
    }

    fn broadcast_tensors(a: &Tensor, b: &Tensor) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
        let max_ndim = a.ndim().max(b.ndim());
        let mut a_shape = vec![1; max_ndim - a.ndim()];
        a_shape.extend_from_slice(&a.shape);
        let mut b_shape = vec![1; max_ndim - b.ndim()];
        b_shape.extend_from_slice(&b.shape);

        let result_shape: Vec<usize> = a_shape.iter().zip(b_shape.iter()).map(|(&a, &b)| a.max(b)).collect();

        let total: usize = result_shape.iter().product();
        let mut a_data = vec![0.0; total];
        let mut b_data = vec![0.0; total];

        let mut indices = vec![0usize; max_ndim];
        for i in 0..total {
            let mut temp = i;
            for d in (0..max_ndim).rev() {
                indices[d] = temp % result_shape[d];
                temp /= result_shape[d];
            }

            let a_idx: Vec<usize> = indices.iter().zip(a_shape.iter()).map(|(&i, &s)| if s == 1 { 0 } else { i }).collect();
            let b_idx: Vec<usize> = indices.iter().zip(b_shape.iter()).map(|(&i, &s)| if s == 1 { 0 } else { i }).collect();

            a_data[i] = a.get(&a_idx);
            b_data[i] = b.get(&b_idx);
        }

        (result_shape, a_data, b_data)
    }

    /// Sum along an axis.
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        assert!(axis < self.ndim());
        let mut new_shape = self.shape.clone();
        new_shape[axis] = 1;

        let result_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; result_size];

        let mut indices = vec![0usize; self.ndim()];
        for i in 0..self.data.len() {
            let mut temp = i;
            for d in (0..self.ndim()).rev() {
                indices[d] = temp % self.shape[d];
                temp /= self.shape[d];
            }

            let mut out_idx = indices.clone();
            out_idx[axis] = 0;
            let offset: usize = out_idx.iter().zip(new_shape[1..].iter().chain(std::iter::once(&1)))
                .scan(0, |acc, (&i, &s)| { let v = *acc; *acc = *acc * s + i; Some(v) })
                .last().unwrap_or(0);

            result[offset] += self.data[i];
        }

        Tensor::new(result, new_shape).squeeze(axis)
    }

    /// Mean along an axis.
    pub fn mean_axis(&self, axis: usize) -> Tensor {
        let n = self.shape[axis] as f64;
        let summed = self.sum_axis(axis);
        let data = summed.data.iter().map(|&x| x / n).collect();
        Tensor::new(data, summed.shape)
    }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self, axis: usize) -> Tensor {
        assert!(self.shape[axis] == 1);
        let mut new_shape = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() { new_shape.push(1); }
        self.reshape(new_shape)
    }

    /// Add dimension of size 1.
    pub fn unsqueeze(&self, axis: usize) -> Tensor {
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        self.reshape(new_shape)
    }

    /// Concatenate tensors along an axis.
    pub fn concatenate(tensors: &[Tensor], axis: usize) -> Tensor {
        assert!(!tensors.is_empty());
        let first = &tensors[0];
        assert!(axis < first.ndim());

        let mut new_shape = first.shape.clone();
        for t in &tensors[1..] {
            assert_eq!(t.ndim(), first.ndim());
            for d in 0..first.ndim() {
                if d == axis {
                    new_shape[d] += t.shape[d];
                } else {
                    assert_eq!(t.shape[d], first.shape[d]);
                }
            }
        }

        let mut result = Vec::new();
        // Simple concatenation for 1D case
        if first.ndim() == 1 {
            for t in tensors {
                result.extend_from_slice(&t.data);
            }
        } else {
            // General case: iterate over non-axis dimensions
            let total: usize = new_shape.iter().product();
            let mut indices = vec![0usize; first.ndim()];

            for _ in 0..total {
                let mut src_idx = indices.clone();
                let mut offset = 0;
                let mut found = false;

                for (t_idx, t) in tensors.iter().enumerate() {
                    let axis_start = if t_idx == 0 { 0 } else {
                        tensors[..t_idx].iter().map(|t| t.shape[axis]).sum()
                    };
                    let axis_end = axis_start + t.shape[axis];

                    if indices[axis] >= axis_start && indices[axis] < axis_end {
                        src_idx[axis] = indices[axis] - axis_start;
                        result.push(t.get(&src_idx));
                        found = true;
                        break;
                    }
                }

                if !found {
                    result.push(0.0);
                }

                // Increment indices
                for d in (0..first.ndim()).rev() {
                    indices[d] += 1;
                    if indices[d] < new_shape[d] { break; }
                    indices[d] = 0;
                }
            }
        }

        Tensor::new(result, new_shape)
    }

    /// Softmax along an axis.
    pub fn softmax(&self, axis: usize) -> Tensor {
        let axis_size = self.shape[axis];
        let mut result = self.data.clone();

        // For each slice along the axis
        let outer_size: usize = self.shape[..axis].iter().product();
        let inner_size: usize = self.shape[axis + 1..].iter().product();

        for o in 0..outer_size {
            for i in 0..inner_size {
                let base = o * axis_size * inner_size + i;
                let mut max_val = f64::NEG_INFINITY;
                for a in 0..axis_size {
                    let idx = base + a * inner_size;
                    max_val = max_val.max(result[idx]);
                }
                let mut sum = 0.0;
                for a in 0..axis_size {
                    let idx = base + a * inner_size;
                    result[idx] = (result[idx] - max_val).exp();
                    sum += result[idx];
                }
                for a in 0..axis_size {
                    let idx = base + a * inner_size;
                    result[idx] /= sum;
                }
            }
        }

        Tensor::new(result, self.shape.clone())
    }

    /// Argmax along an axis.
    pub fn argmax(&self, axis: usize) -> Tensor {
        let axis_size = self.shape[axis];
        let outer_size: usize = self.shape[..axis].iter().product();
        let inner_size: usize = self.shape[axis + 1..].iter().product();

        let mut new_shape = self.shape.clone();
        new_shape[axis] = 1;
        let result_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; result_size];

        for o in 0..outer_size {
            for i in 0..inner_size {
                let base = o * axis_size * inner_size + i;
                let mut max_idx = 0;
                let mut max_val = f64::NEG_INFINITY;
                for a in 0..axis_size {
                    let idx = base + a * inner_size;
                    if self.data[idx] > max_val {
                        max_val = self.data[idx];
                        max_idx = a;
                    }
                }
                let out_base = o * inner_size + i;
                result[out_base] = max_idx as f64;
            }
        }

        Tensor::new(result, new_shape).squeeze(axis)
    }

    /// Clip values.
    pub fn clip(&self, min: f64, max: f64) -> Tensor {
        let data = self.data.iter().map(|&x| x.max(min).min(max)).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Apply function element-wise.
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Tensor {
        let data = self.data.iter().map(|&x| f(x)).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Dot product (1D tensors).
    pub fn dot(&self, other: &Tensor) -> f64 {
        assert_eq!(self.ndim(), 1);
        assert_eq!(other.ndim(), 1);
        assert_eq!(self.shape, other.shape);
        self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum()
    }

    /// Outer product (1D tensors).
    pub fn outer(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 1);
        assert_eq!(other.ndim(), 1);
        let m = self.shape[0];
        let n = other.shape[0];
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = self.data[i] * other.data[j];
            }
        }
        Tensor::new(data, vec![m, n])
    }

    /// Frobenius norm.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Diagonal (2D square tensor).
    pub fn diagonal(&self) -> Tensor {
        assert_eq!(self.ndim(), 2);
        assert_eq!(self.shape[0], self.shape[1]);
        let n = self.shape[0];
        let data = (0..n).map(|i| self.data[i * n + i]).collect();
        Tensor::new(data, vec![n])
    }

    /// Trace (sum of diagonal).
    pub fn trace(&self) -> f64 {
        self.diagonal().data.iter().sum()
    }

    /// Flatten to 1D.
    pub fn flatten(&self) -> Tensor {
        self.reshape(vec![self.data.len()])
    }

    /// Create identity matrix.
    pub fn eye(n: usize) -> Tensor {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::new(data, vec![n, n])
    }

    /// Stack tensors along a new axis.
    pub fn stack(tensors: &[Tensor], axis: usize) -> Tensor {
        let unsqueezed: Vec<Tensor> = tensors.iter().map(|t| t.unsqueeze(axis)).collect();
        Self::concatenate(&unsqueezed, axis)
    }

    /// Split tensor along axis.
    pub fn split(&self, sections: &[usize], axis: usize) -> Vec<Tensor> {
        assert_eq!(sections.iter().sum::<usize>(), self.shape[axis]);
        let mut result = Vec::new();
        let mut start = 0;
        for &size in sections {
            let mut indices = vec![0usize; self.ndim()];
            let mut chunk_data = Vec::new();

            // Extract slice
            for i in 0..self.data.len() {
                let mut temp = i;
                for d in (0..self.ndim()).rev() {
                    indices[d] = temp % self.shape[d];
                    temp /= self.shape[d];
                }
                if indices[axis] >= start && indices[axis] < start + size {
                    chunk_data.push(self.data[i]);
                }
            }

            let mut new_shape = self.shape.clone();
            new_shape[axis] = size;
            result.push(Tensor::new(chunk_data, new_shape));
            start += size;
        }
        result
    }

    /// Einsum (simplified: supports "ij,jk->ik", "ij->i", "ij->", "i,i->", "ij,ij->ij").
    pub fn einsum(subscripts: &str, operands: &[Tensor]) -> Tensor {
        let parts: Vec<&str> = subscripts.split("->").collect();
        let input_subs: Vec<&str> = parts[0].split(',').collect();
        let output_sub = parts.get(1).copied().unwrap_or("");

        // Matrix multiply: ij,jk->ik
        if input_subs.len() == 2 && input_subs[0] == "ij" && input_subs[1] == "jk" && output_sub == "ik" {
            return operands[0].matmul(&operands[1]);
        }

        // Trace/sum: ij->
        if input_subs.len() == 1 && input_subs[0] == "ij" && output_sub == "" {
            return Tensor::new(vec![operands[0].data.iter().sum()], vec![1]);
        }

        // Row sum: ij->i
        if input_subs.len() == 1 && input_subs[0] == "ij" && output_sub == "i" {
            return operands[0].sum_axis(1);
        }

        // Dot product: i,i->
        if input_subs.len() == 2 && input_subs[0] == "i" && input_subs[1] == "i" && output_sub == "" {
            return Tensor::new(vec![operands[0].dot(&operands[1])], vec![1]);
        }

        // Element-wise: ij,ij->ij
        if input_subs.len() == 2 && input_subs[0] == "ij" && input_subs[1] == "ij" {
            if output_sub == "ij" || output_sub == "" {
                return operands[0].broadcast_mul(&operands[1]);
            }
        }

        // Fallback: return first operand
        operands[0].clone()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, data={:?})", self.shape, &self.data[..self.data.len().min(10)])
    }
}

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        self.broadcast_add(other)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, other: &Tensor) -> Tensor {
        self.broadcast_sub(other)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Tensor {
        self.broadcast_mul(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[1, 1]), 4.0);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::eye(2);
        let c = a.matmul(&b);
        assert_eq!(c.data, a.data);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tt = t.transpose();
        assert_eq!(tt.shape, vec![3, 2]);
        assert_eq!(tt.get(&[0, 0]), 1.0);
        assert_eq!(tt.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_broadcast() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![10.0], vec![1]);
        let c = a.broadcast_add(&b);
        assert_eq!(c.data, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let s = t.softmax(0);
        let sum: f64 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_einsum() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = Tensor::einsum("ij,jk->ik", &[a, b]);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.get(&[0, 0]), 1.0 * 5.0 + 2.0 * 7.0);
    }
}
